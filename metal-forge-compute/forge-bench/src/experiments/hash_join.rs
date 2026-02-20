//! Hash join experiment: GPU open-addressing hash join vs CPU HashMap join.
//!
//! Tests hash_join_build + hash_join_probe kernels.
//! Build phase: insert build keys into an open-addressing hash table via atomic CAS.
//! Probe phase: lookup probe keys, emit (build_idx, probe_idx) match pairs.
//!
//! Hash table uses ~2x build_count entries to keep load factor < 0.5.
//! GPU may be slower than CPU for random-access patterns -- that's the data we want.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer, BenchTimer, GpuTimer,
    HashJoinParams, MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// Sentinel value for empty hash table slots (must match HASH_EMPTY in shader).
const HASH_EMPTY: u32 = 0xFFFF_FFFF;

/// Hash join experiment comparing GPU atomic hash table vs CPU HashMap.
pub struct HashJoinExperiment {
    /// Build table keys.
    build_keys: Vec<u32>,
    /// Probe table keys.
    probe_keys: Vec<u32>,
    /// Hash table size (next power of 2 >= 2 * build_count).
    table_size: usize,
    /// Metal buffer for build keys.
    buf_build_keys: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for probe keys.
    buf_probe_keys: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for hash table (table_size * 2 u32s: [key, value] pairs).
    buf_hash_table: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for HashJoinParams.
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for output match pairs (uint2: build_idx, probe_idx).
    buf_output_pairs: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for atomic match count (single u32).
    buf_match_count: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU match count from last run.
    gpu_match_count: u32,
    /// CPU match count from last run.
    cpu_match_count: u32,
    /// Current element count (build side).
    size: usize,
}

impl HashJoinExperiment {
    pub fn new() -> Self {
        Self {
            build_keys: Vec::new(),
            probe_keys: Vec::new(),
            table_size: 0,
            buf_build_keys: None,
            buf_probe_keys: None,
            buf_hash_table: None,
            buf_params: None,
            buf_output_pairs: None,
            buf_match_count: None,
            pso_cache: PsoCache::new(),
            gpu_match_count: 0,
            cpu_match_count: 0,
            size: 0,
        }
    }

    /// Initialize hash table to HASH_EMPTY sentinel values.
    fn clear_hash_table(&self) {
        if let Some(ref buf) = self.buf_hash_table {
            unsafe {
                let ptr = buf.contents().as_ptr() as *mut u32;
                for i in 0..(self.table_size * 2) {
                    *ptr.add(i) = HASH_EMPTY;
                }
            }
        }
    }

    /// Zero the match count buffer.
    fn zero_match_count(&self) {
        if let Some(ref buf) = self.buf_match_count {
            unsafe {
                let ptr = buf.contents().as_ptr() as *mut u32;
                *ptr = 0;
            }
        }
    }
}

impl Experiment for HashJoinExperiment {
    fn name(&self) -> &str {
        "hash_join"
    }

    fn description(&self) -> &str {
        "Hash join (open-addressing): GPU atomic CAS build + linear probe vs CPU HashMap"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![
            100_000,   // 100K
            1_000_000, // 1M
        ]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        let build_count = size;
        let probe_count = size;

        // Generate build keys: values in range [0, build_count * 4)
        // to get ~25% join selectivity with probe keys in same range
        let key_range = (build_count * 4) as u32;
        self.build_keys = gen
            .uniform_u32(build_count)
            .iter()
            .map(|v| v % key_range)
            .collect();
        self.probe_keys = gen
            .uniform_u32(probe_count)
            .iter()
            .map(|v| v % key_range)
            .collect();

        // Hash table: next power of 2 >= 2 * build_count (load factor < 0.5)
        self.table_size = (build_count * 2).next_power_of_two();

        // Allocate Metal buffers
        self.buf_build_keys = Some(alloc_buffer_with_data(&ctx.device, &self.build_keys));
        self.buf_probe_keys = Some(alloc_buffer_with_data(&ctx.device, &self.probe_keys));

        // Hash table: table_size * 2 u32s (key + value interleaved)
        self.buf_hash_table = Some(alloc_buffer(
            &ctx.device,
            self.table_size * 2 * std::mem::size_of::<u32>(),
        ));
        self.clear_hash_table();

        // Params (shared for build and probe)
        let params = HashJoinParams {
            build_count: build_count as u32,
            probe_count: probe_count as u32,
            table_size: self.table_size as u32,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Output pairs: worst case = probe_count matches (each probe finds a match)
        // Use uint2 = 8 bytes per pair
        self.buf_output_pairs = Some(alloc_buffer(
            &ctx.device,
            probe_count * 2 * std::mem::size_of::<u32>(),
        ));

        // Atomic match count: single u32
        self.buf_match_count = Some(alloc_buffer(&ctx.device, std::mem::size_of::<u32>()));
        self.zero_match_count();

        // Pre-warm PSO cache
        self.pso_cache
            .get_or_create(ctx.library(), "hash_join_build");
        self.pso_cache
            .get_or_create(ctx.library(), "hash_join_probe");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        // Reset state
        self.clear_hash_table();
        self.zero_match_count();

        let build_count = self.build_keys.len();
        let probe_count = self.probe_keys.len();

        // Single command buffer with two dispatches: build then probe
        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");

        // --- Build phase ---
        {
            let pso_build = self
                .pso_cache
                .get_or_create(ctx.library(), "hash_join_build");

            let build_keys = self.buf_build_keys.as_ref().expect("setup not called");
            let hash_table = self.buf_hash_table.as_ref().expect("setup not called");
            let params = self.buf_params.as_ref().expect("setup not called");

            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            dispatch_1d(
                &encoder,
                pso_build,
                &[
                    (build_keys.as_ref(), 0),
                    (hash_table.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                build_count,
            );

            encoder.endEncoding();
        }

        // --- Probe phase ---
        {
            let pso_probe = self
                .pso_cache
                .get_or_create(ctx.library(), "hash_join_probe");

            let probe_keys = self.buf_probe_keys.as_ref().expect("setup not called");
            let hash_table = self.buf_hash_table.as_ref().expect("setup not called");
            let params = self.buf_params.as_ref().expect("setup not called");
            let output_pairs = self.buf_output_pairs.as_ref().expect("setup not called");
            let match_count = self.buf_match_count.as_ref().expect("setup not called");

            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            dispatch_1d(
                &encoder,
                pso_probe,
                &[
                    (probe_keys.as_ref(), 0),
                    (hash_table.as_ref(), 1),
                    (params.as_ref(), 2),
                    (output_pairs.as_ref(), 3),
                    (match_count.as_ref(), 4),
                ],
                probe_count,
            );

            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read back match count
        let match_count = self.buf_match_count.as_ref().expect("setup not called");
        self.gpu_match_count = unsafe { read_buffer::<u32>(match_count.as_ref()) };

        GpuTimer::elapsed_ms(&cmd_buf).unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();

        // Build phase: HashMap<key, Vec<index>>
        let mut table: HashMap<u32, Vec<usize>> = HashMap::with_capacity(self.build_keys.len());
        for (idx, &key) in self.build_keys.iter().enumerate() {
            table.entry(key).or_default().push(idx);
        }

        // Probe phase: count matches
        let mut count: u32 = 0;
        for &key in &self.probe_keys {
            if let Some(indices) = table.get(&key) {
                count += indices.len() as u32;
            }
        }
        self.cpu_match_count = count;

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // GPU uses last-writer-wins for duplicate build keys (open addressing, one slot per key),
        // so GPU match count <= CPU match count when build has duplicates.
        // For validation, we compare against a deduplicated CPU baseline.
        let mut dedup_table: HashMap<u32, usize> = HashMap::with_capacity(self.build_keys.len());
        for (idx, &key) in self.build_keys.iter().enumerate() {
            dedup_table.insert(key, idx); // last writer wins, like GPU
        }

        let mut expected_matches: u32 = 0;
        for &key in &self.probe_keys {
            if dedup_table.contains_key(&key) {
                expected_matches += 1;
            }
        }

        if self.gpu_match_count == expected_matches {
            Ok(())
        } else {
            Err(format!(
                "Match count mismatch: GPU={} expected={} (dedup build, size={})",
                self.gpu_match_count, expected_matches, self.size
            ))
        }
    }

    fn metrics(&self, elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let seconds = elapsed_ms / 1000.0;

        // Total joins per second (build + probe combined)
        let total_keys = (self.build_keys.len() + self.probe_keys.len()) as f64;
        let joins_per_sec = if seconds > 0.0 {
            total_keys / seconds
        } else {
            0.0
        };
        m.insert("joins_per_sec".to_string(), joins_per_sec);

        // Match count and selectivity
        m.insert("match_count".to_string(), self.gpu_match_count as f64);
        let selectivity = if !self.probe_keys.is_empty() {
            self.gpu_match_count as f64 / self.probe_keys.len() as f64 * 100.0
        } else {
            0.0
        };
        m.insert("selectivity_pct".to_string(), selectivity);

        // Elements
        m.insert("build_count".to_string(), self.build_keys.len() as f64);
        m.insert("probe_count".to_string(), self.probe_keys.len() as f64);

        // Memory traffic: build reads keys + writes hash table; probe reads keys + reads table + writes pairs
        let build_bytes = self.build_keys.len() as f64 * 4.0 + self.table_size as f64 * 8.0;
        let probe_bytes = self.probe_keys.len() as f64 * 4.0 + self.gpu_match_count as f64 * 8.0; // output pairs
        let total_bytes = build_bytes + probe_bytes;
        let gbs = if seconds > 0.0 {
            total_bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
