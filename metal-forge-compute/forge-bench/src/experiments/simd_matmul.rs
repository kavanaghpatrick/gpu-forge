//! simdgroup_matrix Actual TOPS Measurement.
//!
//! Measures the real throughput of Apple Silicon's hardware matrix multiply units
//! using simdgroup_float8x8 and simdgroup_half8x8 multiply_accumulate.
//!
//! Each simdgroup_multiply_accumulate(c, a, b, c) performs:
//!   `C[8x8] += A[8x8] * B[8x8]`  →  512 FMAs  →  1024 FLOPs
//!
//! Data stays in registers (no memory access after init) to measure
//! pure matrix unit throughput, not memory bandwidth.
//!
//! Expected: F16 should show ~2x FLOPS vs F32 if dedicated half-precision
//! matrix units exist (like NVIDIA's tensor cores for FP16).

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

const NUM_PASSES: u32 = 5000;

pub struct SimdMatmulExperiment {
    buf_output_f32: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_f16: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    f32_ms: f64,
    f16_ms: f64,
    size: usize,
}

impl SimdMatmulExperiment {
    pub fn new() -> Self {
        Self {
            buf_output_f32: None,
            buf_output_f16: None,
            pso_cache: PsoCache::new(),
            f32_ms: 0.0,
            f16_ms: 0.0,
            size: 0,
        }
    }
}

impl Experiment for SimdMatmulExperiment {
    fn name(&self) -> &str {
        "simd_matmul"
    }

    fn description(&self) -> &str {
        "simdgroup_matrix throughput: F32 and F16 multiply-accumulate TOPS"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, _gen: &mut DataGenerator) {
        self.size = size;

        // Small output buffers — kernels only store one 8x8 tile
        self.buf_output_f32 = Some(alloc_buffer(&ctx.device, 256)); // 8*8*4 = 256
        self.buf_output_f16 = Some(alloc_buffer(&ctx.device, 128)); // 8*8*2 = 128

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_simd_matmul_f32");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_simd_matmul_f16");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let output_f32 = self.buf_output_f32.as_ref().unwrap();
        let output_f16 = self.buf_output_f16.as_ref().unwrap();

        let params = ExploitParams {
            element_count: self.size as u32,
            num_passes: NUM_PASSES,
            mode: 0,
            _pad: 0,
        };
        let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

        // ── F32 simdgroup_matrix ──
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_simd_matmul_f32");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[(output_f32.as_ref(), 0), (buf_params.as_ref(), 1)],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.f32_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // ── F16 simdgroup_matrix ──
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_simd_matmul_f16");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[(output_f16.as_ref(), 0), (buf_params.as_ref(), 1)],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.f16_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        self.f32_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU: 8x8 matrix multiply NUM_PASSES times
        let mut c = [[0.0f32; 8]; 8];
        let a = [[0.001f32; 8]; 8];
        let b = [[0.5f32; 8]; 8];

        for _ in 0..NUM_PASSES {
            for i in 0..8 {
                for j in 0..8 {
                    let mut sum = c[i][j];
                    for k in 0..8 {
                        sum += a[i][k] * b[k][j];
                    }
                    c[i][j] = sum;
                }
            }
        }
        std::hint::black_box(c);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("f32_ms".to_string(), self.f32_ms);
        m.insert("f16_ms".to_string(), self.f16_ms);

        // Each simdgroup does num_passes multiply-accumulate ops
        // Each op = 512 FMAs = 1024 FLOPs
        let num_simdgroups = self.size as f64 / 32.0;
        let total_fmas = num_simdgroups * NUM_PASSES as f64 * 512.0;
        let total_flops = total_fmas * 2.0; // count mul + add separately

        if self.f32_ms > 0.0 {
            let tflops_f32 = total_flops / (self.f32_ms / 1000.0) / 1e12;
            m.insert("f32_tflops".to_string(), tflops_f32);
            m.insert("f32_gflops".to_string(), tflops_f32 * 1000.0);
        }

        if self.f16_ms > 0.0 {
            let tflops_f16 = total_flops / (self.f16_ms / 1000.0) / 1e12;
            m.insert("f16_tflops".to_string(), tflops_f16);
            m.insert("f16_gflops".to_string(), tflops_f16 * 1000.0);
        }

        // F16 vs F32 ratio — reveals dedicated half-precision matrix units
        if self.f32_ms > 0.0 && self.f16_ms > 0.0 {
            m.insert("f16_vs_f32_speedup".to_string(), self.f32_ms / self.f16_ms);
        }

        m
    }
}
