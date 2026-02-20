//! Cache Hierarchy Probe via pointer chasing.
//!
//! Measures memory LATENCY (not bandwidth) at different working set sizes.
//! Each load depends on the previous load's result, defeating prefetching.
//!
//! Latency jumps reveal cache boundaries:
//!   - < ~8KB:  L1 cache (~4 cycles, ~1.5ns)
//!   - ~8KB-24MB: SLC (~30 cycles, ~12ns)
//!   - > ~24MB: DRAM (~100+ cycles, ~40ns)
//!
//! The Rust host creates a Sattolo-cycle random permutation for each
//! working set size, ensuring every element is visited exactly once.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, ExploitParams, GpuTimer, MetalContext,
    PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

// Working set sizes: 4KB to 64MB (in u32 elements)
const WORKING_SETS: &[(usize, &str)] = &[
    (1024, "4KB"),         // 4 KB — should be pure L1
    (4096, "16KB"),        // 16 KB — L1 capacity
    (16384, "64KB"),       // 64 KB — past L1
    (65536, "256KB"),      // 256 KB — L2 range
    (262144, "1MB"),       // 1 MB — deep SLC
    (1048576, "4MB"),      // 4 MB — SLC capacity test
    (4194304, "16MB"),     // 16 MB — SLC boundary
    (16777216, "64MB"),    // 64 MB — pure DRAM
];

const CHASE_STEPS: u32 = 10000;
const NUM_THREADS: usize = 1024; // Few threads to avoid saturating bandwidth

pub struct CacheProbeExperiment {
    bufs_chain: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    latencies_ns: Vec<(usize, String, f64)>,
    size: usize,
}

impl CacheProbeExperiment {
    pub fn new() -> Self {
        Self {
            bufs_chain: Vec::new(),
            buf_output: None,
            pso_cache: PsoCache::new(),
            latencies_ns: Vec::new(),
            size: 0,
        }
    }

    /// Create a Sattolo cycle (random cyclic permutation) of length n.
    /// Every element is visited exactly once before returning to start.
    fn sattolo_cycle(n: usize) -> Vec<u32> {
        let mut perm: Vec<u32> = (0..n as u32).collect();
        // Sattolo's algorithm
        let mut rng_state = 0x12345678u64;
        for i in (1..n).rev() {
            // Simple LCG for deterministic results
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % i; // j in [0, i-1]
            perm.swap(i, j);
        }
        perm
    }
}

impl Experiment for CacheProbeExperiment {
    fn name(&self) -> &str {
        "cache_probe"
    }

    fn description(&self) -> &str {
        "Cache hierarchy probe: pointer chasing at 4KB-64MB to find L1/SLC/DRAM boundaries"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        // Size parameter doesn't matter much — we control working set internally
        vec![1024]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, _gen: &mut DataGenerator) {
        self.size = size;
        self.bufs_chain.clear();
        self.latencies_ns.clear();

        // Create pointer-chase buffers for each working set size
        for (ws_elems, _label) in WORKING_SETS {
            let chain = Self::sattolo_cycle(*ws_elems);
            self.bufs_chain
                .push(alloc_buffer_with_data(&ctx.device, &chain));
        }

        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            NUM_THREADS * std::mem::size_of::<u32>(),
        ));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_cache_probe");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let output = self.buf_output.as_ref().unwrap();
        self.latencies_ns.clear();

        for (i, (ws_elems, label)) in WORKING_SETS.iter().enumerate() {
            let chain_buf = &self.bufs_chain[i];

            let params = ExploitParams {
                element_count: NUM_THREADS as u32,
                num_passes: CHASE_STEPS,
                mode: *ws_elems as u32,
                _pad: 0,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_cache_probe");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (chain_buf.as_ref(), 0),
                    (output.as_ref(), 1),
                    (buf_params.as_ref(), 2),
                ],
                NUM_THREADS,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let gpu_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
            // Latency per chase step in nanoseconds
            let ns_per_step = if gpu_ms > 0.0 {
                (gpu_ms * 1e6) / CHASE_STEPS as f64
            } else {
                0.0
            };

            self.latencies_ns
                .push((*ws_elems, label.to_string(), ns_per_step));
        }

        // Return the 1MB latency as representative
        self.latencies_ns
            .iter()
            .find(|(e, _, _)| *e == 262144)
            .map(|(_, _, ns)| *ns / 1e6) // convert back to ms for API
            .unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU pointer chase at 1MB working set
        let chain = Self::sattolo_cycle(262144);
        let mut idx = 0usize;
        for _ in 0..CHASE_STEPS * 100 {
            idx = chain[idx] as usize;
        }
        std::hint::black_box(idx);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // Check that we got results for all working set sizes
        if self.latencies_ns.len() == WORKING_SETS.len() {
            Ok(())
        } else {
            Err(format!(
                "Expected {} results, got {}",
                WORKING_SETS.len(),
                self.latencies_ns.len()
            ))
        }
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        for (ws_elems, label, ns_per_step) in &self.latencies_ns {
            m.insert(format!("latency_{}_ns", label), *ns_per_step);
            m.insert(format!("ws_{}_bytes", label), (*ws_elems * 4) as f64);
        }

        // Compute latency ratios to detect cache boundaries
        if self.latencies_ns.len() >= 2 {
            for i in 1..self.latencies_ns.len() {
                let (_, prev_label, prev_ns) = &self.latencies_ns[i - 1];
                let (_, curr_label, curr_ns) = &self.latencies_ns[i];
                if *prev_ns > 0.0 {
                    let ratio = curr_ns / prev_ns;
                    m.insert(
                        format!("jump_{}_to_{}", prev_label, curr_label),
                        ratio,
                    );
                }
            }
        }

        m
    }
}
