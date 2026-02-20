//! Instruction-Level Parallelism (ILP) Width + Dual-Issue Probe experiment.
//!
//! Part A: ILP Width
//! Runs 1, 2, 4, or 8 INDEPENDENT FMA chains per thread. Throughput scales
//! linearly until the ALU pipeline is saturated. The saturation point reveals
//! how many FMA instructions the GPU can have in-flight per thread.
//!
//! Part B: Dual-Issue
//! Tests whether FP32 and INT32 ops execute on separate pipelines.
//! If dual-issue exists, interleaving FP+INT should give ~2x throughput
//! vs pure FP or pure INT alone.

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

const ILP_KERNELS: &[(&str, u32)] = &[
    ("exploit_ilp_1", 1),
    ("exploit_ilp_2", 2),
    ("exploit_ilp_4", 4),
    ("exploit_ilp_8", 8),
];

const DUAL_KERNELS: &[&str] = &[
    "exploit_fp32_only",
    "exploit_int32_only",
    "exploit_dual_issue",
];

const NUM_PASSES: u32 = 1000;

pub struct IlpWidthExperiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    ilp_times: [f64; 4],
    dual_times: [f64; 3],
    size: usize,
}

impl IlpWidthExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            ilp_times: [0.0; 4],
            dual_times: [0.0; 3],
            size: 0,
        }
    }
}

impl Experiment for IlpWidthExperiment {
    fn name(&self) -> &str {
        "ilp_width"
    }

    fn description(&self) -> &str {
        "ILP width (1-8 chains) + FP32/INT32 dual-issue probe"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;
        self.data = gen.uniform_f32(size);

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));
        self.buf_output = Some(alloc_buffer(&ctx.device, size * 4));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: NUM_PASSES,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        for (name, _) in ILP_KERNELS {
            self.pso_cache.get_or_create(ctx.library(), name);
        }
        for name in DUAL_KERNELS {
            self.pso_cache.get_or_create(ctx.library(), name);
        }
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        // ILP width variants
        for (i, (name, _)) in ILP_KERNELS.iter().enumerate() {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self.pso_cache.get_or_create(ctx.library(), name);
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
            self.ilp_times[i] = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // Dual-issue variants
        for (i, name) in DUAL_KERNELS.iter().enumerate() {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self.pso_cache.get_or_create(ctx.library(), name);
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
            self.dual_times[i] = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // Return single-chain ILP time as primary
        self.ilp_times[0]
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        let mut sum = 0.0f64;
        for &v in &self.data {
            let mut a = v;
            for _ in 0..NUM_PASSES {
                a = a.mul_add(a, 1.0);
            }
            sum += a as f64;
        }
        std::hint::black_box(sum);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        // ILP times and GFLOP rates
        for (i, (_, chains)) in ILP_KERNELS.iter().enumerate() {
            m.insert(format!("ilp{}_ms", chains), self.ilp_times[i]);

            // Each chain does num_passes FMAs (2 FLOP each)
            let flops = self.size as f64 * NUM_PASSES as f64 * *chains as f64 * 2.0;
            let gflops = if self.ilp_times[i] > 0.0 {
                flops / (self.ilp_times[i] / 1000.0) / 1e9
            } else {
                0.0
            };
            m.insert(format!("ilp{}_gflops", chains), gflops);
        }

        // ILP speedup relative to 1-chain
        if self.ilp_times[0] > 0.0 {
            for (i, (_, chains)) in ILP_KERNELS.iter().enumerate() {
                // Normalize: speedup = (chains * time_1chain) / time_Nchains
                let speedup = (*chains as f64 * self.ilp_times[0]) / self.ilp_times[i];
                m.insert(format!("ilp{}_speedup_x", chains), speedup);
            }
        }

        // Dual-issue times
        m.insert("fp32_ms".to_string(), self.dual_times[0]);
        m.insert("int32_ms".to_string(), self.dual_times[1]);
        m.insert("mixed_ms".to_string(), self.dual_times[2]);

        // Dual-issue ratio: if < 1.0, FP+INT overlap (separate pipelines)
        // mixed_time / max(fp_time, int_time)
        let max_single = self.dual_times[0].max(self.dual_times[1]);
        if max_single > 0.0 {
            let dual_ratio = self.dual_times[2] / max_single;
            m.insert("dual_issue_ratio".to_string(), dual_ratio);
            // If ratio ≈ 1.0: dual-issue works (FP and INT overlap)
            // If ratio ≈ 2.0: no dual-issue (FP and INT serialize)
        }

        m
    }
}
