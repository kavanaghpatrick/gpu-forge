//! Threadgroup Memory Gradient experiment.
//!
//! Maps the Dynamic Caching curve by sweeping threadgroup memory from
//! 256 bytes to 32 KB. All 7 kernel variants do identical work (4 float4
//! loads per thread + SIMD reduce). The only difference is the amount
//! of threadgroup memory allocated, which steals from the Dynamic Caching pool.
//!
//! Expected: smooth degradation curve showing exactly how much bandwidth
//! each KB of threadgroup memory costs. The zero-TG version from
//! exploit_zero_tg_reduce serves as the 0 KB baseline.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, ExploitParams, GpuTimer,
    MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// Kernel names and their TG memory sizes in bytes.
const TG_VARIANTS: [(&str, usize); 8] = [
    ("exploit_zero_tg_reduce", 0),        // from exploits.metal
    ("exploit_tg_256b_reduce", 256),
    ("exploit_tg_1kb_reduce", 1024),
    ("exploit_tg_2kb_reduce", 2048),
    ("exploit_tg_4kb_reduce", 4096),
    ("exploit_tg_8kb_reduce", 8192),
    ("exploit_tg_16kb_reduce", 16384),
    ("exploit_tg_32kb_reduce", 32768),
];

pub struct TgGradientExperiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    variant_times: Vec<(usize, f64)>, // (tg_bytes, ms)
    best_ms: f64,
    cpu_result: f64,
    size: usize,
}

impl TgGradientExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            variant_times: Vec::new(),
            best_ms: 0.0,
            cpu_result: 0.0,
            size: 0,
        }
    }
}

impl Experiment for TgGradientExperiment {
    fn name(&self) -> &str {
        "tg_gradient"
    }

    fn description(&self) -> &str {
        "Dynamic Caching curve: TG memory 0→32KB, mapping pool allocation cost"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // float4 elements
        self.data = gen
            .uniform_u32(size * 4)
            .iter()
            .map(|&v| (v % 100) as f32 * 0.01)
            .collect();

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output: enough for any variant's partials
        // zero_tg: 8 simdgroups/TG * ceil(size/(256*8)) = max partials
        let total_threads = size.div_ceil(8); // 8 float4 per thread for zero-TG
        let max_tgs = total_threads.div_ceil(256);
        // TG gradient variants use 4 float4 per thread, so ceil(size/(256*4)) TGs
        let max_tgs_gradient = size.div_ceil(4).div_ceil(256);
        let max_partials = (max_tgs * 8).max(max_tgs_gradient * 8);

        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            max_partials * std::mem::size_of::<f32>(),
        ));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Pre-warm all variant PSOs
        for &(kernel_name, _) in &TG_VARIANTS {
            self.pso_cache
                .get_or_create(ctx.library(), kernel_name);
        }
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        self.variant_times.clear();

        for &(kernel_name, tg_bytes) in &TG_VARIANTS {
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), kernel_name);

            // Determine total_threads based on kernel variant
            let total_threads = if tg_bytes == 0 {
                // zero_tg: 8 float4 per thread
                self.size.div_ceil(8)
            } else {
                // tg_gradient: 4 float4 per thread
                self.size.div_ceil(4)
            };

            // Warmup
            for _ in 0..2 {
                let cmd = ctx.queue.commandBuffer().unwrap();
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
            }

            // Measured runs (median of 3)
            let mut times = Vec::new();
            for _ in 0..3 {
                let cmd = ctx.queue.commandBuffer().unwrap();
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
                times.push(GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0));
            }

            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = times[1];
            self.variant_times.push((tg_bytes, median));
        }

        self.best_ms = self
            .variant_times
            .iter()
            .map(|&(_, ms)| ms)
            .fold(f64::MAX, f64::min);

        self.best_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();
        self.cpu_result = self.data.iter().map(|&x| x as f64).sum();
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.variant_times.len() == TG_VARIANTS.len() {
            Ok(())
        } else {
            Err(format!(
                "Expected {} variants, got {}",
                TG_VARIANTS.len(),
                self.variant_times.len()
            ))
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let bytes = size as f64 * 16.0; // float4 = 16 bytes

        for &(tg_bytes, ms) in &self.variant_times {
            let label = if tg_bytes == 0 {
                "0kb".to_string()
            } else if tg_bytes < 1024 {
                format!("{}b", tg_bytes)
            } else {
                format!("{}kb", tg_bytes / 1024)
            };

            m.insert(format!("tg_{}_ms", label), ms);

            let gbs = if ms > 0.0 {
                bytes / (ms / 1000.0) / 1e9
            } else {
                0.0
            };
            m.insert(format!("tg_{}_gbs", label), gbs);
        }

        // Zero-TG BW for the table
        if let Some(&(_, ms)) = self.variant_times.first() {
            let gbs = if ms > 0.0 {
                bytes / (ms / 1000.0) / 1e9
            } else {
                0.0
            };
            m.insert("gb_per_sec".to_string(), gbs);
        }

        // Cost per KB of TG memory (ms penalty)
        if self.variant_times.len() >= 2 {
            let zero_ms = self.variant_times[0].1;
            let full_ms = self.variant_times.last().unwrap().1;
            let full_kb = self.variant_times.last().unwrap().0 as f64 / 1024.0;
            if full_kb > 0.0 {
                let ms_per_kb = (full_ms - zero_ms) / full_kb;
                m.insert("ms_penalty_per_kb_tg".to_string(), ms_per_kb);
            }
        }

        m
    }
}
