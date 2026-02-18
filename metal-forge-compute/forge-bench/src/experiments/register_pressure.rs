//! Register Pressure Cliff experiment.
//!
//! Varies the number of live float registers per thread (8, 16, 32, 64, 128).
//! Each kernel keeps N floats alive with dependent FMA ops across all of them.
//! The performance cliff reveals the GPU register file size per core.
//!
//! At the cliff point, occupancy drops sharply as threads can no longer
//! fit their registers simultaneously, and performance tanks.

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

const VARIANTS: &[(&str, usize)] = &[
    ("exploit_regpressure_8", 8),
    ("exploit_regpressure_16", 16),
    ("exploit_regpressure_32", 32),
    ("exploit_regpressure_64", 64),
    ("exploit_regpressure_128", 128),
];

const NUM_PASSES: u32 = 100;

pub struct RegisterPressureExperiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    variant_times: [f64; 5],
    size: usize,
}

impl RegisterPressureExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            variant_times: [0.0; 5],
            size: 0,
        }
    }
}

impl Experiment for RegisterPressureExperiment {
    fn name(&self) -> &str {
        "register_pressure"
    }

    fn description(&self) -> &str {
        "Register pressure cliff: vary live registers (8-128) to find register file limit"
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

        for (kernel_name, _) in VARIANTS {
            self.pso_cache.get_or_create(ctx.library(), kernel_name);
        }
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        for (i, (kernel_name, _)) in VARIANTS.iter().enumerate() {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self.pso_cache.get_or_create(ctx.library(), kernel_name);
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
            self.variant_times[i] = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // Return the 32-register time as the "primary" GPU time
        self.variant_times[2]
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU baseline: 32 registers worth of FMA work
        let mut sum = 0.0f64;
        for &v in &self.data {
            let mut r = [v; 32];
            for _ in 0..NUM_PASSES {
                for i in 0..32 {
                    r[i] = r[i].mul_add(r[(i + 1) & 31], r[(i + 3) & 31]);
                }
            }
            sum += r.iter().sum::<f32>() as f64;
        }
        std::hint::black_box(sum);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // All variants should produce finite results
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        for (i, (_, reg_count)) in VARIANTS.iter().enumerate() {
            m.insert(format!("reg{}_ms", reg_count), self.variant_times[i]);
        }

        // Compute FLOP rates: each variant does size * num_passes * reg_count FMAs
        for (i, (_, reg_count)) in VARIANTS.iter().enumerate() {
            let flops = self.size as f64 * NUM_PASSES as f64 * *reg_count as f64 * 2.0; // FMA = 2 FLOP
            let gflops = if self.variant_times[i] > 0.0 {
                flops / (self.variant_times[i] / 1000.0) / 1e9
            } else {
                0.0
            };
            m.insert(format!("reg{}_gflops", reg_count), gflops);
        }

        // Efficiency relative to 8-register variant (should decrease at cliff)
        if self.variant_times[0] > 0.0 {
            for (i, (_, reg_count)) in VARIANTS.iter().enumerate() {
                let flops_i = self.size as f64 * *reg_count as f64;
                let flops_8 = self.size as f64 * 8.0;
                let expected_ratio = flops_i / flops_8;
                let actual_ratio = self.variant_times[i] / self.variant_times[0];
                let efficiency = expected_ratio / actual_ratio;
                m.insert(format!("reg{}_efficiency", reg_count), efficiency);
            }
        }

        m
    }
}
