//! SLC Lock-Free V3: AoS (Array-of-Structs) interleaved layout.
//!
//! Key optimization: interleave key+value as adjacent u32 pairs.
//! Layout: [key0, val0, key1, val1, ...] — 8 bytes per slot.
//! One 128B cache line fetch gets 16 complete key-value pairs.
//! Eliminates the second random DRAM access for the value array.
//!
//! At DRAM sizes (>16MB table), this halves random memory accesses
//! per found key: 1 fetch instead of 2 (key array + value array).
//!
//! Uses murmur3 hash (from V2) + 32-bit atomic CAS on key slots.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, GpuOsParams, GpuTimer, MetalContext,
    PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

const EMPTY_KEY: u32 = 0xFFFFFFFF;
const QUEUE_CYCLES: u32 = 10;

fn next_power_of_2(n: u32) -> u32 {
    if n <= 1 {
        return 1;
    }
    1u32 << (32 - (n - 1).leading_zeros())
}

pub struct SlcLockfreeV3Experiment {
    /// AoS table: interleaved [key, val, key, val, ...] as u32 pairs
    buf_slots: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_input_keys: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_input_values: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    buf_queue_ht: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_ring: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_q_result: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    input_keys: Vec<u32>,
    input_values: Vec<u32>,

    insert_ms: f64,
    lookup_ms: f64,
    mixed_ms: f64,
    queue_ms: f64,
    lookup_hits: u32,

    pso_cache: PsoCache,
    size: usize,
    capacity: u32,
}

impl SlcLockfreeV3Experiment {
    pub fn new() -> Self {
        Self {
            buf_slots: None,
            buf_input_keys: None,
            buf_input_values: None,
            buf_output: None,
            buf_queue_ht: None,
            buf_ring: None,
            buf_q_result: None,
            input_keys: Vec::new(),
            input_values: Vec::new(),
            insert_ms: 0.0,
            lookup_ms: 0.0,
            mixed_ms: 0.0,
            queue_ms: 0.0,
            lookup_hits: 0,
            pso_cache: PsoCache::new(),
            size: 0,
            capacity: 0,
        }
    }

    fn clear_table(&self) {
        if let Some(ref buf) = self.buf_slots {
            unsafe {
                let ptr = buf.contents().as_ptr() as *mut u8;
                // Fill with 0xFF → each u32 = 0xFFFFFFFF (EMPTY_KEY)
                // capacity * 2 u32s (key+value pairs) * 4 bytes = capacity * 8
                std::ptr::write_bytes(ptr, 0xFF, self.capacity as usize * 8);
            }
        }
    }
}

impl Experiment for SlcLockfreeV3Experiment {
    fn name(&self) -> &str {
        "slc_lockfree_v3"
    }

    fn description(&self) -> &str {
        "SLC lock-free V3: AoS interleaved layout (1 fetch = key + value)"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;
        self.capacity = next_power_of_2((size * 2) as u32);

        self.input_keys = gen
            .uniform_u32(size)
            .into_iter()
            .map(|k| if k == EMPTY_KEY { 0xFFFFFFFE } else { k })
            .collect();
        self.input_values = gen.uniform_u32(size);

        // AoS table: capacity slots * 2 u32s per slot * 4 bytes = capacity * 8
        self.buf_slots = Some(alloc_buffer(
            &ctx.device,
            self.capacity as usize * 2 * std::mem::size_of::<u32>(),
        ));
        self.buf_input_keys = Some(alloc_buffer_with_data(&ctx.device, &self.input_keys));
        self.buf_input_values = Some(alloc_buffer_with_data(&ctx.device, &self.input_values));
        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));

        let q_cap = next_power_of_2(size as u32);
        self.buf_queue_ht = Some(alloc_buffer(
            &ctx.device,
            2 * std::mem::size_of::<u32>(),
        ));
        self.buf_ring = Some(alloc_buffer(
            &ctx.device,
            q_cap as usize * std::mem::size_of::<u32>(),
        ));
        self.buf_q_result = Some(alloc_buffer(
            &ctx.device,
            std::mem::size_of::<u32>(),
        ));

        // V3 kernels
        self.pso_cache
            .get_or_create(ctx.library(), "gpuos_ht_insert_v3");
        self.pso_cache
            .get_or_create(ctx.library(), "gpuos_ht_lookup_v3");
        self.pso_cache
            .get_or_create(ctx.library(), "gpuos_ht_mixed_v3");
        self.pso_cache
            .get_or_create(ctx.library(), "gpuos_queue_throughput");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let slots = self.buf_slots.as_ref().unwrap();
        let input_keys = self.buf_input_keys.as_ref().unwrap();
        let input_values = self.buf_input_values.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();

        // V3 uses different buffer binding order (no separate keys/values table)
        let params = GpuOsParams {
            capacity: self.capacity,
            num_ops: self.size as u32,
            num_queues: 0,
            mode: 0,
        };

        // ── INSERT V3 (AoS interleaved, 32-bit atomic CAS on key) ──
        self.clear_table();
        let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "gpuos_ht_insert_v3");
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            pso,
            &[
                (slots.as_ref(), 0),
                (input_keys.as_ref(), 1),
                (input_values.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            self.size,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        self.insert_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);

        // ── LOOKUP V3 (AoS interleaved — key+value in same cache line) ──
        let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "gpuos_ht_lookup_v3");
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            pso,
            &[
                (slots.as_ref(), 0),
                (input_keys.as_ref(), 1),
                (output.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            self.size,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        self.lookup_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);

        // Count lookup hits
        self.lookup_hits = 0;
        unsafe {
            let ptr = output.contents().as_ptr() as *const u32;
            for i in 0..self.size {
                if *ptr.add(i) != EMPTY_KEY {
                    self.lookup_hits += 1;
                }
            }
        }

        // ── MIXED V3 (AoS, concurrent insert+lookup) ──
        self.clear_table();
        let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "gpuos_ht_mixed_v3");
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            pso,
            &[
                (slots.as_ref(), 0),
                (input_keys.as_ref(), 1),
                (output.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            self.size,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        self.mixed_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);

        // ── MPMC QUEUE (same as V1/V2 for comparison) ──
        let q_cap = next_power_of_2(self.size as u32);
        let q_params = GpuOsParams {
            capacity: q_cap,
            num_ops: self.size as u32,
            num_queues: QUEUE_CYCLES,
            mode: 0,
        };

        let ht_buf = self.buf_queue_ht.as_ref().unwrap();
        let q_result = self.buf_q_result.as_ref().unwrap();
        unsafe {
            let ptr = ht_buf.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 8);
            let ptr = q_result.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 4);
        }

        let buf_qp = alloc_buffer_with_data(&ctx.device, &[q_params]);
        let ring = self.buf_ring.as_ref().unwrap();
        let cmd = ctx.queue.commandBuffer().unwrap();
        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "gpuos_queue_throughput");
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            pso,
            &[
                (ht_buf.as_ref(), 0),
                (ring.as_ref(), 1),
                (q_result.as_ref(), 2),
                (buf_qp.as_ref(), 3),
            ],
            self.size,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        self.queue_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);

        self.insert_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        let mut map = std::collections::HashMap::with_capacity(self.size);
        for i in 0..self.size {
            map.insert(self.input_keys[i], self.input_values[i]);
        }
        let mut hits = 0u64;
        for i in 0..self.size {
            if map.get(&self.input_keys[i]).is_some() {
                hits += 1;
            }
        }
        std::hint::black_box(hits);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        let hit_pct = self.lookup_hits as f64 / self.size as f64 * 100.0;
        if hit_pct < 50.0 {
            return Err(format!(
                "Low lookup hit rate: {:.1}% ({} / {})",
                hit_pct, self.lookup_hits, self.size
            ));
        }
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("insert_ms".to_string(), self.insert_ms);
        m.insert("lookup_ms".to_string(), self.lookup_ms);
        m.insert("mixed_ms".to_string(), self.mixed_ms);
        m.insert("queue_ms".to_string(), self.queue_ms);

        let size = self.size as f64;
        if self.insert_ms > 0.0 {
            m.insert("insert_mops".to_string(), size / self.insert_ms / 1000.0);
        }
        if self.lookup_ms > 0.0 {
            m.insert("lookup_mops".to_string(), size / self.lookup_ms / 1000.0);
        }
        if self.mixed_ms > 0.0 {
            m.insert("mixed_mops".to_string(), size / self.mixed_ms / 1000.0);
        }

        let total_q_ops = size * QUEUE_CYCLES as f64 * 2.0;
        if self.queue_ms > 0.0 {
            m.insert("queue_mops".to_string(), total_q_ops / self.queue_ms / 1000.0);
        }

        let hit_pct = self.lookup_hits as f64 / self.size as f64 * 100.0;
        m.insert("lookup_hit_pct".to_string(), hit_pct);

        // AoS table: capacity * 2 u32s * 4 bytes = capacity * 8
        let table_mb = (self.capacity as f64 * 8.0) / (1024.0 * 1024.0);
        m.insert("table_mb".to_string(), table_mb);

        m
    }
}
