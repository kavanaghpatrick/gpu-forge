//! SLC Cache Residency experiment.
//!
//! Exploits Apple Silicon's GPU-inclusive SLC property: all GPU data is backed
//! by SLC, and CPU activity does NOT evict GPU data.
//!
//! Multi-pass algorithms on SLC-sized working sets (~24 MB on M4 Pro) get
//! SLC cache hits on pass 2+, dramatically faster than pass 1 (DRAM load).
//!
//! Test: run 8 identical passes over the same buffer. Measure per-pass timing.
//! Pass 1 = cold (DRAM). Pass 2-8 = warm (SLC hits). Compare latencies.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer_with_data, dispatch_1d, ExploitParams, GpuTimer, MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// Number of passes to run over the same data.
const NUM_PASSES: u32 = 8;

pub struct SlcResidencyExperiment {
    data: Vec<f32>,
    buf_data: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    pass_times_ms: Vec<f64>,
    size: usize,
}

impl SlcResidencyExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_data: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            pass_times_ms: Vec::new(),
            size: 0,
        }
    }
}

impl Experiment for SlcResidencyExperiment {
    fn name(&self) -> &str {
        "slc_residency"
    }

    fn description(&self) -> &str {
        "SLC cache residency: multi-pass latency (pass 1 DRAM vs pass 2+ SLC hits)"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        // float4 elements: 16 bytes each
        // 1M float4 = 16 MB (fits in SLC ~24 MB)
        // 4M float4 = 64 MB (exceeds SLC)
        vec![
            1_000_000,  // 16 MB -- fits SLC
            4_000_000,  // 64 MB -- exceeds SLC
            16_000_000, // 256 MB -- way over SLC
        ]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // Generate float data (we'll treat it as float4 in the shader)
        self.data = gen
            .uniform_u32(size * 4) // size float4 elements = size*4 floats
            .iter()
            .map(|&v| (v % 1000) as f32 * 0.001)
            .collect();

        self.buf_data = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: NUM_PASSES,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_slc_multipass");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let data = self.buf_data.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();
        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "exploit_slc_multipass");

        self.pass_times_ms.clear();

        // Run each pass as a separate command buffer to get per-pass GPU timing
        for _ in 0..NUM_PASSES {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[(data.as_ref(), 0), (params.as_ref(), 1)],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
            self.pass_times_ms.push(ms);
        }

        // Return total time
        self.pass_times_ms.iter().sum()
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // Same computation on CPU: N passes of multiply-add
        for _ in 0..NUM_PASSES {
            for val in self.data.iter_mut() {
                *val = *val * 1.00001 + 0.5;
            }
        }

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // No strict validation needed -- we're measuring timing, not correctness.
        // Just verify we got timing data.
        if self.pass_times_ms.len() == NUM_PASSES as usize {
            Ok(())
        } else {
            Err(format!(
                "Expected {} pass times, got {}",
                NUM_PASSES,
                self.pass_times_ms.len()
            ))
        }
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let bytes = size as f64 * 16.0; // float4 = 16 bytes (read + write = 32 bytes per element)

        // Per-pass timing
        for (i, &ms) in self.pass_times_ms.iter().enumerate() {
            m.insert(format!("pass_{}_ms", i + 1), ms);
        }

        // Pass 1 (cold) vs average of pass 2+ (warm)
        if let Some(&pass1) = self.pass_times_ms.first() {
            m.insert("pass1_cold_ms".to_string(), pass1);

            if self.pass_times_ms.len() > 1 {
                let warm_avg: f64 =
                    self.pass_times_ms[1..].iter().sum::<f64>() / (NUM_PASSES - 1) as f64;
                m.insert("warm_avg_ms".to_string(), warm_avg);

                let slc_speedup = if warm_avg > 0.0 {
                    pass1 / warm_avg
                } else {
                    0.0
                };
                m.insert("slc_speedup_x".to_string(), slc_speedup);

                // Warm bandwidth (reads + writes = 2x data)
                let warm_gbs = if warm_avg > 0.0 {
                    (bytes * 2.0) / (warm_avg / 1000.0) / 1e9
                } else {
                    0.0
                };
                m.insert("gb_per_sec".to_string(), warm_gbs);
            }
        }

        m.insert(
            "working_set_mb".to_string(),
            (size as f64 * 16.0) / 1e6,
        );

        m
    }
}
