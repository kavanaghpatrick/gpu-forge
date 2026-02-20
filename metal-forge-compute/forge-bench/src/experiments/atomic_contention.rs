//! Atomic Contention Scaling experiment.
//!
//! Measures atomic_fetch_add throughput at different contention levels:
//!   (A) 1 target -- all threads contend on single location (worst case)
//!   (B) per-simdgroup -- SIMD reduce then 1 atomic per 32 threads
//!   (C) per-threadgroup -- SIMD reduce then 1 atomic per 256 threads
//!   (D) per-thread private -- accumulate locally, 1 atomic per TG
//!
//! Shows how SIMD reduce + hierarchical atomics eliminates contention.

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

const VARIANT_NAMES: [&str; 4] = [
    "exploit_atomic_single",
    "exploit_atomic_per_simdgroup",
    "exploit_atomic_per_tg",
    "exploit_atomic_private",
];

pub struct AtomicContentionExperiment {
    data: Vec<u32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    variant_times: [f64; 4],
    size: usize,
}

impl AtomicContentionExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            variant_times: [0.0; 4],
            size: 0,
        }
    }
}

impl Experiment for AtomicContentionExperiment {
    fn name(&self) -> &str {
        "atomic_contention"
    }

    fn description(&self) -> &str {
        "Atomic contention scaling: single target vs SIMD reduce + hierarchical atomics"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        self.data = gen.uniform_u32(size);

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output buffer: needs enough space for all variants
        // atomic_private uses tg_id as index, so needs ceil(size/256) * 4 entries minimum
        // Use 64KB to be safe for all variants
        let output_size = 65536 * std::mem::size_of::<u32>();
        self.buf_output = Some(alloc_buffer(&ctx.device, output_size));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        for name in &VARIANT_NAMES {
            self.pso_cache.get_or_create(ctx.library(), name);
        }
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        for (i, name) in VARIANT_NAMES.iter().enumerate() {
            // Zero output buffer before each variant
            unsafe {
                let ptr = output.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(ptr, 0, 65536 * 4);
            }

            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self.pso_cache.get_or_create(ctx.library(), name);
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.variant_times[i] = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // Return the "best" (private) time as the GPU time
        self.variant_times[3]
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU: sum all byte values (same work as GPU)
        let mut sum: u64 = 0;
        for &val in &self.data {
            sum += (val & 0xFF) as u64;
            sum += ((val >> 8) & 0xFF) as u64;
            sum += ((val >> 16) & 0xFF) as u64;
            sum += ((val >> 24) & 0xFF) as u64;
        }
        std::hint::black_box(sum);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // All variants should have non-zero timing
        for (i, &t) in self.variant_times.iter().enumerate() {
            if t == 0.0 {
                return Err(format!("Variant {} ({}) has zero timing", i, VARIANT_NAMES[i]));
            }
        }
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("single_ms".to_string(), self.variant_times[0]);
        m.insert("per_simdgroup_ms".to_string(), self.variant_times[1]);
        m.insert("per_tg_ms".to_string(), self.variant_times[2]);
        m.insert("private_ms".to_string(), self.variant_times[3]);

        // Contention relief ratio: single / private
        let relief = if self.variant_times[3] > 0.0 {
            self.variant_times[0] / self.variant_times[3]
        } else {
            0.0
        };
        m.insert("contention_relief_x".to_string(), relief);

        // Bandwidth for private variant (best case)
        let bytes = size as f64 * 4.0; // uint = 4 bytes
        let gbs = if self.variant_times[3] > 0.0 {
            bytes / (self.variant_times[3] / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
