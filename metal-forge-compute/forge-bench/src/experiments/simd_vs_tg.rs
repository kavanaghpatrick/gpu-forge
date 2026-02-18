//! SIMD Shuffle vs Threadgroup Memory bandwidth experiment.
//!
//! Exploits the undocumented 4x bandwidth gap between Apple GPU's SIMD shuffle
//! fabric (256 bytes/cycle) and threadgroup memory (64 bytes/cycle).
//!
//! Both kernels perform identical reductions -- the only difference is the
//! inter-thread communication mechanism. The shuffle version should be ~4x faster.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer_slice, BenchTimer, ExploitParams,
    GpuTimer, MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

pub struct SimdVsTgExperiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_simd: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_tg: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_simd_ms: f64,
    gpu_tg_ms: f64,
    simd_result: f64,
    tg_result: f64,
    cpu_result: f64,
    size: usize,
}

impl SimdVsTgExperiment {
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
            simd_result: 0.0,
            tg_result: 0.0,
            cpu_result: 0.0,
            size: 0,
        }
    }
}

impl Experiment for SimdVsTgExperiment {
    fn name(&self) -> &str {
        "simd_vs_tg"
    }

    fn description(&self) -> &str {
        "SIMD shuffle (256 B/cycle) vs threadgroup memory (64 B/cycle) reduction"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // Generate float data as groups of 4 (float4)
        // Use small values to avoid float precision issues in large sums
        self.data = gen
            .uniform_u32(size)
            .iter()
            .map(|&v| (v % 100) as f32 * 0.01)
            .collect();

        // Input: size floats packed as float4 (size/4 float4 elements)
        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output buffers: one per-simdgroup partial per TG
        // With 256 threads/TG and 32 threads/simdgroup = 8 simdgroups/TG
        // num_tgs = ceil(size / (4*256)) for float4 input
        let float4_count = size.div_ceil(4);
        let num_tgs = float4_count.div_ceil(256);
        let num_partials = num_tgs * 8; // 8 simdgroups per TG

        self.buf_output_simd = Some(alloc_buffer(
            &ctx.device,
            num_partials * std::mem::size_of::<f32>(),
        ));
        self.buf_output_tg = Some(alloc_buffer(
            &ctx.device,
            num_tgs * std::mem::size_of::<f32>(),
        ));

        let params = ExploitParams {
            element_count: float4_count as u32,
            num_passes: 1,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_simd_shuffle_reduce");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_threadgroup_reduce");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output_simd = self.buf_output_simd.as_ref().unwrap();
        let output_tg = self.buf_output_tg.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        let float4_count = self.size.div_ceil(4);
        let num_tgs = float4_count.div_ceil(256);

        // --- Run SIMD shuffle version ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_simd_shuffle_reduce");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output_simd.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                float4_count,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_simd_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);

            // Sum partials on CPU
            let num_partials = num_tgs * 8;
            let partials: Vec<f32> =
                unsafe { read_buffer_slice::<f32>(output_simd.as_ref(), num_partials) };
            self.simd_result = partials.iter().map(|&x| x as f64).sum();
        }

        // --- Run threadgroup version ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_threadgroup_reduce");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output_tg.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                float4_count,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_tg_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);

            let partials: Vec<f32> =
                unsafe { read_buffer_slice::<f32>(output_tg.as_ref(), num_tgs) };
            self.tg_result = partials.iter().map(|&x| x as f64).sum();
        }

        // Return the SIMD time as the "GPU time" (faster path)
        self.gpu_simd_ms
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();
        self.cpu_result = self.data.iter().map(|&x| x as f64).sum();
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // Float reduction order differs, so allow generous tolerance
        let rel_err = if self.cpu_result.abs() > 1e-6 {
            ((self.simd_result - self.cpu_result) / self.cpu_result).abs()
        } else {
            (self.simd_result - self.cpu_result).abs()
        };

        if rel_err < 0.01 {
            Ok(())
        } else {
            Err(format!(
                "SIMD result ({:.2}) too far from CPU ({:.2}), rel_err={:.4}",
                self.simd_result, self.cpu_result, rel_err
            ))
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let bytes = size as f64 * 4.0; // f32 = 4 bytes

        m.insert("simd_shuffle_ms".to_string(), self.gpu_simd_ms);
        m.insert("threadgroup_ms".to_string(), self.gpu_tg_ms);

        let speedup = if self.gpu_simd_ms > 0.0 {
            self.gpu_tg_ms / self.gpu_simd_ms
        } else {
            0.0
        };
        m.insert("shuffle_speedup_x".to_string(), speedup);

        // Effective bandwidth for each path
        let simd_gbs = if self.gpu_simd_ms > 0.0 {
            bytes / (self.gpu_simd_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        let tg_gbs = if self.gpu_tg_ms > 0.0 {
            bytes / (self.gpu_tg_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("simd_gb_per_sec".to_string(), simd_gbs);
        m.insert("tg_gb_per_sec".to_string(), tg_gbs);
        m.insert("gb_per_sec".to_string(), simd_gbs);

        m
    }
}
