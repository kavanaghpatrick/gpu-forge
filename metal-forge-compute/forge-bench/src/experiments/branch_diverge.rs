//! Branch Divergence Cost experiment.
//!
//! Apple GPU uses SIMD execution (32-wide). When threads in a SIMD take
//! different branches, both paths execute (thread masking).
//!
//! Compare:
//!   (A) Uniform: all threads take same branch
//!   (B) Alternating: even/odd threads diverge (50%)
//!   (C) Random: random branch per thread (worst case for 2-path)
//!   (D) Deep diverge: each SIMD lane takes different iteration depth

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
    "exploit_branch_uniform",
    "exploit_branch_alternating",
    "exploit_branch_random",
    "exploit_branch_deep_diverge",
];

pub struct BranchDivergeExperiment {
    data_f32: Vec<f32>,
    data_u32: Vec<u32>,
    buf_input_f32: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_input_u32: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    variant_times: [f64; 4],
    size: usize,
}

impl BranchDivergeExperiment {
    pub fn new() -> Self {
        Self {
            data_f32: Vec::new(),
            data_u32: Vec::new(),
            buf_input_f32: None,
            buf_input_u32: None,
            buf_output: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            variant_times: [0.0; 4],
            size: 0,
        }
    }
}

impl Experiment for BranchDivergeExperiment {
    fn name(&self) -> &str {
        "branch_diverge"
    }

    fn description(&self) -> &str {
        "Branch divergence cost: uniform vs alternating vs random vs deep per-lane divergence"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // Float data for uniform/alternating kernels (float4 input)
        // Values 0.0-0.99 so all < 2.0 threshold (uniform path A)
        self.data_f32 = gen
            .uniform_u32(size * 4)
            .iter()
            .map(|&v| (v % 100) as f32 * 0.01)
            .collect();

        // Uint keys for random/deep_diverge kernels
        self.data_u32 = gen.uniform_u32(size);

        self.buf_input_f32 = Some(alloc_buffer_with_data(&ctx.device, &self.data_f32));
        self.buf_input_u32 = Some(alloc_buffer_with_data(&ctx.device, &self.data_u32));

        // Output: float4 per element = 16 bytes
        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            size * 4 * std::mem::size_of::<f32>(),
        ));

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
        let input_f32 = self.buf_input_f32.as_ref().unwrap();
        let input_u32 = self.buf_input_u32.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        for (i, name) in VARIANT_NAMES.iter().enumerate() {
            // Uniform and alternating use float4 input (buffer 0)
            // Random and deep_diverge use uint keys (buffer 0)
            let input_buf: &ProtocolObject<dyn MTLBuffer> = if i < 2 {
                input_f32.as_ref()
            } else {
                input_u32.as_ref()
            };

            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self.pso_cache.get_or_create(ctx.library(), name);
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input_buf, 0),
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

        // Return uniform (best case) as the GPU time
        self.variant_times[0]
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU: same multiply-add chain (8 iterations, uniform path)
        let mut acc = 0.0f64;
        for chunk in self.data_f32.chunks(4) {
            let mut v: [f64; 4] = [
                chunk.first().copied().unwrap_or(0.0) as f64,
                chunk.get(1).copied().unwrap_or(0.0) as f64,
                chunk.get(2).copied().unwrap_or(0.0) as f64,
                chunk.get(3).copied().unwrap_or(0.0) as f64,
            ];
            for _ in 0..8 {
                for x in v.iter_mut() {
                    *x = *x * 1.001 + 0.5;
                }
            }
            acc += v[0] + v[1] + v[2] + v[3];
        }
        std::hint::black_box(acc);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        for (i, &t) in self.variant_times.iter().enumerate() {
            if t == 0.0 {
                return Err(format!(
                    "Variant {} ({}) has zero timing",
                    i, VARIANT_NAMES[i]
                ));
            }
        }
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("uniform_ms".to_string(), self.variant_times[0]);
        m.insert("alternating_ms".to_string(), self.variant_times[1]);
        m.insert("random_ms".to_string(), self.variant_times[2]);
        m.insert("deep_diverge_ms".to_string(), self.variant_times[3]);

        // Divergence penalty: each variant vs uniform
        for i in 1..4 {
            let label = match i {
                1 => "alternating",
                2 => "random",
                3 => "deep_diverge",
                _ => unreachable!(),
            };
            let penalty = if self.variant_times[0] > 0.0 {
                self.variant_times[i] / self.variant_times[0]
            } else {
                0.0
            };
            m.insert(format!("{}_penalty_x", label), penalty);
        }

        // Bandwidth for uniform (data read: float4 per element = 16 bytes)
        let bytes = size as f64 * 16.0;
        let gbs = if self.variant_times[0] > 0.0 {
            bytes / (self.variant_times[0] / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
