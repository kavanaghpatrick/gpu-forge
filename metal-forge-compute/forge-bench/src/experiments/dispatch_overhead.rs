//! Dispatch Overhead Isolation.
//!
//! Measures the exact cost of Metal dispatch by running a trivial kernel
//! (writes a single uint) varying numbers of times.
//!
//! Two modes:
//!   A) N encoders in ONE command buffer — measures per-encoder overhead
//!   B) N SEPARATE command buffers — measures per-command-buffer overhead
//!
//! The difference reveals: firmware scheduling, encoder setup, and commit costs.

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

const DISPATCH_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256];

pub struct DispatchOverheadExperiment {
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    // (count, single_cmdbuf_ms, multi_cmdbuf_ms)
    results: Vec<(usize, f64, f64)>,
    size: usize,
}

impl DispatchOverheadExperiment {
    pub fn new() -> Self {
        Self {
            buf_output: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            results: Vec::new(),
            size: 0,
        }
    }
}

impl Experiment for DispatchOverheadExperiment {
    fn name(&self) -> &str {
        "dispatch_overhead"
    }

    fn description(&self) -> &str {
        "Dispatch overhead: per-encoder vs per-command-buffer cost with trivial kernel"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1024]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, _gen: &mut DataGenerator) {
        self.size = size;
        self.results.clear();

        self.buf_output = Some(alloc_buffer(&ctx.device, 256));

        let params = ExploitParams {
            element_count: 1,
            num_passes: 1,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_dispatch_trivial");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();
        self.results.clear();

        for &count in DISPATCH_COUNTS {
            // ── Mode A: N encoders in ONE command buffer ──
            let cmd_single = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_dispatch_trivial");

            for _ in 0..count {
                let enc = cmd_single.computeCommandEncoder().unwrap();
                dispatch_1d(
                    &enc,
                    pso,
                    &[(output.as_ref(), 0), (params.as_ref(), 1)],
                    1,
                );
                enc.endEncoding();
            }
            cmd_single.commit();
            cmd_single.waitUntilCompleted();
            let single_ms = GpuTimer::elapsed_ms(&cmd_single).unwrap_or(0.0);

            // ── Mode B: N SEPARATE command buffers ──
            let start = std::time::Instant::now();
            for _ in 0..count {
                let cmd = ctx.queue.commandBuffer().unwrap();
                let pso_b = self
                    .pso_cache
                    .get_or_create(ctx.library(), "exploit_dispatch_trivial");
                let enc = cmd.computeCommandEncoder().unwrap();
                dispatch_1d(
                    &enc,
                    pso_b,
                    &[(output.as_ref(), 0), (params.as_ref(), 1)],
                    1,
                );
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }
            let multi_ms = start.elapsed().as_secs_f64() * 1000.0;

            self.results.push((count, single_ms, multi_ms));
        }

        // Return single-dispatch overhead as primary
        self.results
            .first()
            .map(|(_, s, _)| *s)
            .unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();
        // CPU does nothing meaningful — just baseline measurement
        let mut v = 0u64;
        for i in 0..1000 {
            v = v.wrapping_add(i);
        }
        std::hint::black_box(v);
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.results.len() == DISPATCH_COUNTS.len() {
            Ok(())
        } else {
            Err(format!(
                "Expected {} results, got {}",
                DISPATCH_COUNTS.len(),
                self.results.len()
            ))
        }
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        for (count, single_ms, multi_ms) in &self.results {
            m.insert(format!("single_{}_ms", count), *single_ms);
            m.insert(format!("multi_{}_ms", count), *multi_ms);

            // Per-dispatch overhead
            if *count > 0 {
                m.insert(
                    format!("per_encoder_{}_us", count),
                    single_ms * 1000.0 / *count as f64,
                );
                m.insert(
                    format!("per_cmdbuf_{}_us", count),
                    multi_ms * 1000.0 / *count as f64,
                );
            }
        }

        // Derive amortized costs from the largest batch
        if let Some((count, single_ms, multi_ms)) = self.results.last() {
            m.insert(
                "amortized_encoder_us".to_string(),
                single_ms * 1000.0 / *count as f64,
            );
            m.insert(
                "amortized_cmdbuf_us".to_string(),
                multi_ms * 1000.0 / *count as f64,
            );
            if *single_ms > 0.0 {
                m.insert(
                    "cmdbuf_vs_encoder_ratio".to_string(),
                    multi_ms / single_ms,
                );
            }
        }

        m
    }
}
