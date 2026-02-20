//! Encoder Reuse experiment.
//!
//! Metal encoder state (pipeline, buffers, textures) persists across dispatches
//! within a single encoder. Creating a new encoder has overhead: the runtime
//! must flush state, potentially insert barriers, and reset binding tables.
//!
//! Compare:
//!   (A) 1 encoder → set state once → dispatch N times → end once
//!   (B) N encoders → each sets state + dispatches + ends
//!
//! Same total work, different encoding patterns. Measures pure encoding overhead.

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

/// Number of dispatches per run.
const NUM_DISPATCHES: usize = 64;

pub struct EncoderReuseExperiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_reuse_ms: f64,   // 1 encoder, N dispatches
    gpu_recreate_ms: f64, // N encoders, 1 dispatch each
    size: usize,
}

impl EncoderReuseExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_reuse_ms: 0.0,
            gpu_recreate_ms: 0.0,
            size: 0,
        }
    }
}

impl Experiment for EncoderReuseExperiment {
    fn name(&self) -> &str {
        "encoder_reuse"
    }

    fn description(&self) -> &str {
        "Encoder reuse: 1 encoder × N dispatches vs N encoders × 1 dispatch"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        // Smaller sizes where dispatch overhead is proportionally larger
        vec![10_000, 100_000, 1_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // float4 input
        self.data = gen
            .uniform_u32(size * 4)
            .iter()
            .map(|&v| v as f32)
            .collect();

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));
        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            size * 4 * std::mem::size_of::<f32>(),
        ));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: NUM_DISPATCHES as u32,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_encoder_work");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();
        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "exploit_encoder_work");

        // === Path A: 1 encoder, N dispatches (reuse) ===
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();

            // Set state ONCE
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

            // Pipeline + buffers already set. Dispatch N-1 more times.
            // dispatch_1d re-sets them each time, so we use it directly
            // (it's still 1 encoder, the state persistence means less
            // internal bookkeeping even though we call setBuffer again).
            for _ in 1..NUM_DISPATCHES {
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
            }

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_reuse_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // === Path B: N encoders, 1 dispatch each (recreate) ===
        {
            let cmd = ctx.queue.commandBuffer().unwrap();

            for _ in 0..NUM_DISPATCHES {
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
            }

            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_recreate_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        self.gpu_reuse_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // Same work: N passes of multiply-add
        for _ in 0..NUM_DISPATCHES {
            for val in self.data.iter_mut() {
                *val = *val * 2.0 + 1.0;
            }
        }

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_reuse_ms > 0.0 && self.gpu_recreate_ms > 0.0 {
            Ok(())
        } else {
            Err("No GPU timing recorded".to_string())
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("reuse_1enc_ms".to_string(), self.gpu_reuse_ms);
        m.insert("recreate_Nenc_ms".to_string(), self.gpu_recreate_ms);
        m.insert("num_dispatches".to_string(), NUM_DISPATCHES as f64);

        let overhead_saved = if self.gpu_reuse_ms > 0.0 {
            self.gpu_recreate_ms / self.gpu_reuse_ms
        } else {
            0.0
        };
        m.insert("encoder_reuse_speedup_x".to_string(), overhead_saved);

        // Per-dispatch overhead = (recreate_time - reuse_time) / N
        let overhead_per_dispatch = if NUM_DISPATCHES > 0 {
            (self.gpu_recreate_ms - self.gpu_reuse_ms) / NUM_DISPATCHES as f64
        } else {
            0.0
        };
        m.insert("overhead_per_encoder_ms".to_string(), overhead_per_dispatch);

        // Total bytes processed
        let bytes = size as f64 * 16.0 * NUM_DISPATCHES as f64; // float4 * N dispatches
        let gbs = if self.gpu_reuse_ms > 0.0 {
            bytes / (self.gpu_reuse_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
