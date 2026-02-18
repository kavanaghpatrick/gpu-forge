//! Bitonic Sort in SIMD Registers — Zero Memory Sort.
//!
//! Sorts 32 elements using ONLY simd_shuffle_xor — no threadgroup memory,
//! no barriers, no global memory access after the initial load.
//!
//! A bitonic sorting network for 32 elements has 5 stages with 15 total
//! compare-and-swap steps. Each step uses simd_shuffle_xor(val, distance)
//! to fetch the partner's value, then keeps min or max based on position.
//!
//! This is the fastest possible sort for 32 elements on any GPU:
//! 15 register shuffles vs 15 shared-memory loads + 15 barriers.

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

pub struct SimdSort32Experiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_simd: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_tg: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_simd_ms: f64,
    gpu_tg_ms: f64,
    size: usize,
}

impl SimdSort32Experiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output_simd: None,
            buf_output_tg: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_simd_ms: 0.0,
            gpu_tg_ms: 0.0,
            size: 0,
        }
    }

    fn validate_sorted(data: &[f32]) -> bool {
        // Check that every group of 32 is sorted
        for chunk in data.chunks(32) {
            for i in 1..chunk.len() {
                if chunk[i] < chunk[i - 1] {
                    return false;
                }
            }
        }
        true
    }
}

impl Experiment for SimdSort32Experiment {
    fn name(&self) -> &str {
        "simd_sort32"
    }

    fn description(&self) -> &str {
        "Bitonic sort 32 elements: SIMD shuffle (zero memory) vs threadgroup memory"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        // Must be multiple of 32
        vec![1_000_000, 10_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        // Round to multiple of 32
        let size = (size / 32) * 32;
        self.size = size;
        self.data = gen.uniform_f32(size);

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));
        self.buf_output_simd = Some(alloc_buffer(&ctx.device, size * 4));
        self.buf_output_tg = Some(alloc_buffer(&ctx.device, size * 4));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_simd_sort32");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_tg_sort32");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        // SIMD shuffle sort
        {
            let output = self.buf_output_simd.as_ref().unwrap();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_simd_sort32");
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
            self.gpu_simd_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // Threadgroup memory sort
        {
            let output = self.buf_output_tg.as_ref().unwrap();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_tg_sort32");
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
            self.gpu_tg_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        self.gpu_simd_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU: sort each group of 32 elements
        let mut output = self.data.clone();
        for chunk in output.chunks_mut(32) {
            chunk.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        std::hint::black_box(&output);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        let simd_output = self.buf_output_simd.as_ref().unwrap();
        let result: Vec<f32> =
            unsafe { forge_primitives::read_buffer_slice(simd_output.as_ref(), self.size) };

        if Self::validate_sorted(&result) {
            Ok(())
        } else {
            // Find first unsorted position
            for (i, chunk) in result.chunks(32).enumerate() {
                for j in 1..chunk.len() {
                    if chunk[j] < chunk[j - 1] {
                        return Err(format!(
                            "Group {} unsorted at pos {}: {} > {}",
                            i,
                            j,
                            chunk[j - 1],
                            chunk[j]
                        ));
                    }
                }
            }
            Err("Unknown sort error".to_string())
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("simd_sort_ms".to_string(), self.gpu_simd_ms);
        m.insert("tg_sort_ms".to_string(), self.gpu_tg_ms);

        // SIMD vs threadgroup speedup
        if self.gpu_simd_ms > 0.0 {
            m.insert(
                "tg_vs_simd_x".to_string(),
                self.gpu_tg_ms / self.gpu_simd_ms,
            );
        }

        // Elements sorted per second
        let num_groups = size / 32;
        if self.gpu_simd_ms > 0.0 {
            let groups_per_sec = num_groups as f64 / (self.gpu_simd_ms / 1000.0);
            m.insert("simd_groups_per_sec".to_string(), groups_per_sec);
            m.insert(
                "simd_elems_per_sec".to_string(),
                groups_per_sec * 32.0,
            );
        }

        // Bandwidth: read + write
        let bytes = size as f64 * 4.0 * 2.0;
        if self.gpu_simd_ms > 0.0 {
            m.insert(
                "simd_gbs".to_string(),
                bytes / (self.gpu_simd_ms / 1000.0) / 1e9,
            );
        }

        m
    }
}
