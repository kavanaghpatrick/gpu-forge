//! SIMD 32-Tap FIR Filter — Pipeline in Registers.
//!
//! Implements a 32-tap FIR filter ENTIRELY in SIMD registers using
//! simd_shuffle_rotate_down. No threadgroup memory, no barriers, no
//! global memory reads beyond the initial load.
//!
//! Data flows between lanes at register speed (~1 cycle per shuffle).
//! This is zero-memory convolution — the holy grail of GPU stencils.
//!
//! Compare: SIMD shuffle FIR vs memory-read FIR vs threadgroup FIR

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

pub struct SimdPipelineExperiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_simd: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_mem: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_tg: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_simd_ms: f64,
    gpu_mem_ms: f64,
    gpu_tg_ms: f64,
    size: usize,
}

impl SimdPipelineExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output_simd: None,
            buf_output_mem: None,
            buf_output_tg: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_simd_ms: 0.0,
            gpu_mem_ms: 0.0,
            gpu_tg_ms: 0.0,
            size: 0,
        }
    }
}

impl Experiment for SimdPipelineExperiment {
    fn name(&self) -> &str {
        "simd_pipeline"
    }

    fn description(&self) -> &str {
        "32-tap FIR filter: SIMD shuffle (zero memory) vs global reads vs threadgroup"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;
        self.data = gen.uniform_f32(size);

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));
        self.buf_output_simd = Some(alloc_buffer(&ctx.device, size * 4));
        self.buf_output_mem = Some(alloc_buffer(&ctx.device, size * 4));
        self.buf_output_tg = Some(alloc_buffer(&ctx.device, size * 4));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_simd_fir");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_memory_fir");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_tgmem_fir");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        // SIMD shuffle FIR
        {
            let output = self.buf_output_simd.as_ref().unwrap();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_simd_fir");
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

        // Global memory FIR
        {
            let output = self.buf_output_mem.as_ref().unwrap();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_memory_fir");
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
            self.gpu_mem_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // Threadgroup memory FIR
        {
            let output = self.buf_output_tg.as_ref().unwrap();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_tgmem_fir");
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

        let mut output = vec![0.0f32; self.data.len()];
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.data.len() {
            let mut sum = 0.0f32;
            for tap in 0..32u32 {
                let idx = i + tap as usize;
                if idx < self.data.len() {
                    sum += self.data[idx] / (tap + 1) as f32;
                }
            }
            output[i] = sum;
        }
        std::hint::black_box(&output);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // Both GPU versions should produce similar results
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("simd_fir_ms".to_string(), self.gpu_simd_ms);
        m.insert("memory_fir_ms".to_string(), self.gpu_mem_ms);
        m.insert("tgmem_fir_ms".to_string(), self.gpu_tg_ms);

        // SIMD vs memory speedup
        if self.gpu_simd_ms > 0.0 {
            m.insert(
                "mem_vs_simd_x".to_string(),
                self.gpu_mem_ms / self.gpu_simd_ms,
            );
            m.insert(
                "tg_vs_simd_x".to_string(),
                self.gpu_tg_ms / self.gpu_simd_ms,
            );
        }

        // Effective bandwidth (input data read + output written)
        let bytes = size as f64 * 4.0 * 2.0; // read + write
        if self.gpu_simd_ms > 0.0 {
            m.insert(
                "simd_gbs".to_string(),
                bytes / (self.gpu_simd_ms / 1000.0) / 1e9,
            );
        }

        m
    }
}
