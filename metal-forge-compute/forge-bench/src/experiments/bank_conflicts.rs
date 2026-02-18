//! Threadgroup Memory Bank Conflict Probe.
//!
//! Tests whether Apple Silicon GPU threadgroup memory has a banking structure
//! (like NVIDIA's 32 banks).
//!
//! Each SIMD group accesses threadgroup memory at a parameterized stride:
//!   stride 1:  threads hit consecutive addresses (no conflicts expected)
//!   stride 2:  every other address
//!   stride 4:  every 4th address
//!   stride 8:  every 8th
//!   stride 16: every 16th
//!   stride 32: all threads same bank (if 32 banks of 4B → 32-way conflict)
//!
//! If bank conflicts exist: high strides show proportional slowdowns.
//! If no conflicts: all strides show similar performance.

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

const STRIDES: &[u32] = &[1, 2, 4, 8, 16, 32];
const NUM_PASSES: u32 = 10000;

pub struct BankConflictsExperiment {
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    stride_times: Vec<(u32, f64)>, // (stride, ms)
    size: usize,
}

impl BankConflictsExperiment {
    pub fn new() -> Self {
        Self {
            buf_output: None,
            pso_cache: PsoCache::new(),
            stride_times: Vec::new(),
            size: 0,
        }
    }
}

impl Experiment for BankConflictsExperiment {
    fn name(&self) -> &str {
        "bank_conflicts"
    }

    fn description(&self) -> &str {
        "TG memory bank conflicts: stride sweep to detect banking structure"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, _gen: &mut DataGenerator) {
        self.size = size;
        self.stride_times.clear();

        self.buf_output = Some(alloc_buffer(&ctx.device, size * 4));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_bank_test");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let output = self.buf_output.as_ref().unwrap();
        self.stride_times.clear();

        for &stride in STRIDES {
            let params = ExploitParams {
                element_count: self.size as u32,
                num_passes: NUM_PASSES,
                mode: stride,
                _pad: 0,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_bank_test");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[(output.as_ref(), 0), (buf_params.as_ref(), 1)],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let gpu_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
            self.stride_times.push((stride, gpu_ms));
        }

        // Return stride-1 time as primary
        self.stride_times
            .first()
            .map(|(_, ms)| *ms)
            .unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU: simulate TG memory access pattern
        let mut mem = vec![0.0f32; 8192];
        let mut sum = 0.0f64;
        for p in 0..NUM_PASSES {
            for lane in 0..32u32 {
                let addr = (lane * 1) as usize & 8191;
                mem[addr] = (lane + p) as f32;
                sum += mem[addr] as f64;
            }
        }
        std::hint::black_box(sum);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.stride_times.len() == STRIDES.len() {
            Ok(())
        } else {
            Err(format!(
                "Expected {} strides, got {}",
                STRIDES.len(),
                self.stride_times.len()
            ))
        }
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        let baseline = self
            .stride_times
            .first()
            .map(|(_, ms)| *ms)
            .unwrap_or(1.0);

        for (stride, ms) in &self.stride_times {
            m.insert(format!("stride_{}_ms", stride), *ms);
            // Slowdown relative to stride-1 (no conflicts)
            if baseline > 0.0 {
                m.insert(format!("stride_{}_ratio", stride), ms / baseline);
            }
        }

        // Max slowdown (indicates worst bank conflict)
        if baseline > 0.0 {
            let max_ratio = self
                .stride_times
                .iter()
                .map(|(_, ms)| ms / baseline)
                .fold(0.0f64, f64::max);
            m.insert("max_bank_penalty_x".to_string(), max_ratio);
        }

        m
    }
}
