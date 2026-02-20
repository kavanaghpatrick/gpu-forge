//! Dynamic Caching Pool Monopolization experiment.
//!
//! M3/M4 Dynamic Caching: ~208 KB shared pool per GPU core, dynamically split
//! between registers, threadgroup memory, tile cache, and buffer L1.
//!
//! Trick: using ZERO threadgroup memory lets the full pool go to registers
//! and buffer L1 cache. Deliberately wasting 16 KB of threadgroup memory
//! reduces the pool available for L1, hurting bandwidth-bound kernels.
//!
//! Both kernels do identical work (8 float4 loads per thread + SIMD reduce).
//! The zero-TG version should be measurably faster due to larger L1 cache.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer_slice, BenchTimer,
    ExploitParams, GpuTimer, MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

pub struct DynamicCacheExperiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_zero_tg_ms: f64,
    gpu_heavy_tg_ms: f64,
    zero_tg_result: f64,
    heavy_tg_result: f64,
    cpu_result: f64,
    size: usize,
}

impl DynamicCacheExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_zero_tg_ms: 0.0,
            gpu_heavy_tg_ms: 0.0,
            zero_tg_result: 0.0,
            heavy_tg_result: 0.0,
            cpu_result: 0.0,
            size: 0,
        }
    }
}

impl Experiment for DynamicCacheExperiment {
    fn name(&self) -> &str {
        "dynamic_cache"
    }

    fn description(&self) -> &str {
        "Dynamic Caching exploit: zero-TG (full L1 pool) vs 16KB-TG (reduced L1)"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // Generate float4 data: size float4 elements = size * 4 floats
        self.data = gen
            .uniform_u32(size * 4)
            .iter()
            .map(|&v| (v % 100) as f32 * 0.01)
            .collect();

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output: each kernel writes one partial per threadgroup
        // zero-tg: 8 simdgroups/TG, each thread handles 8 float4 = ceil(size / (256*8)) TGs
        let num_tgs = size.div_ceil(256 * 8);
        let max_partials = num_tgs * 8; // simdgroup partials for zero-TG version
        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            max_partials.max(num_tgs) * std::mem::size_of::<f32>(),
        ));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_zero_tg_reduce");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_heavy_tg_reduce");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        // Each thread handles 8 float4, so total_threads = ceil(size / 8)
        let total_threads = self.size.div_ceil(8);
        let num_tgs = total_threads.div_ceil(256);

        // --- Zero-TG version (full Dynamic Caching pool for L1) ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_zero_tg_reduce");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                total_threads,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_zero_tg_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);

            let num_partials = num_tgs * 8;
            let partials: Vec<f32> =
                unsafe { read_buffer_slice::<f32>(output.as_ref(), num_partials) };
            self.zero_tg_result = partials.iter().map(|&x| x as f64).sum();
        }

        // --- Heavy-TG version (16 KB threadgroup, reduced L1 pool) ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_heavy_tg_reduce");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                total_threads,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_heavy_tg_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);

            let partials: Vec<f32> =
                unsafe { read_buffer_slice::<f32>(output.as_ref(), num_tgs) };
            self.heavy_tg_result = partials.iter().map(|&x| x as f64).sum();
        }

        self.gpu_zero_tg_ms
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();
        self.cpu_result = self.data.iter().map(|&x| x as f64).sum();
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        let rel_err = if self.cpu_result.abs() > 1e-6 {
            ((self.zero_tg_result - self.cpu_result) / self.cpu_result).abs()
        } else {
            (self.zero_tg_result - self.cpu_result).abs()
        };

        if rel_err < 0.02 {
            Ok(())
        } else {
            Err(format!(
                "Zero-TG result ({:.2}) too far from CPU ({:.2}), rel_err={:.4}",
                self.zero_tg_result, self.cpu_result, rel_err
            ))
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let bytes = size as f64 * 16.0; // float4 = 16 bytes per element

        m.insert("zero_tg_ms".to_string(), self.gpu_zero_tg_ms);
        m.insert("heavy_tg_ms".to_string(), self.gpu_heavy_tg_ms);

        let speedup = if self.gpu_zero_tg_ms > 0.0 {
            self.gpu_heavy_tg_ms / self.gpu_zero_tg_ms
        } else {
            0.0
        };
        m.insert("pool_monopoly_speedup_x".to_string(), speedup);

        let zero_gbs = if self.gpu_zero_tg_ms > 0.0 {
            bytes / (self.gpu_zero_tg_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), zero_gbs);

        m
    }
}
