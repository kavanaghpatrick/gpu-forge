//! Divergent Task Dispatch + SIMD Finite State Machine.
//!
//! The ultimate "each SIMD lane is a CPU" experiment.
//!
//! Part A: Divergent Tasks
//! Each group of 8 lanes executes STRUCTURALLY DIFFERENT computation:
//!   - Lanes  0-7:  Cryptographic hash (integer multiply-xor)
//!   - Lanes  8-15: Polynomial evaluation (FMA chain)
//!   - Lanes 16-23: Bit manipulation (popcount, clz, shift)
//!   - Lanes 24-31: Transcendental math (sin, cos)
//!
//! On NVIDIA: 4 code paths serialize → ~4x slowdown.
//! On Apple Silicon: claimed zero divergence → should match uniform.
//!
//! Part B: SIMD FSM
//! Each lane independently runs a finite state machine with different
//! transitions per state — 32 independent programs in one SIMD group.

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

const NUM_PASSES: u32 = 100;

const TASK_KERNELS: &[&str] = &[
    "exploit_divergent_tasks",
    "exploit_uniform_hash",
    "exploit_uniform_poly",
    "exploit_uniform_bits",
    "exploit_uniform_trig",
    "exploit_simd_fsm",
];

pub struct SimdTaskqueueExperiment {
    data_f32: Vec<f32>,
    data_u32: Vec<u32>,
    buf_input_f32: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_input_u32: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_u32: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    divergent_ms: f64,
    uniform_hash_ms: f64,
    uniform_poly_ms: f64,
    uniform_bits_ms: f64,
    uniform_trig_ms: f64,
    fsm_ms: f64,
    size: usize,
}

impl SimdTaskqueueExperiment {
    pub fn new() -> Self {
        Self {
            data_f32: Vec::new(),
            data_u32: Vec::new(),
            buf_input_f32: None,
            buf_input_u32: None,
            buf_output: None,
            buf_output_u32: None,
            pso_cache: PsoCache::new(),
            divergent_ms: 0.0,
            uniform_hash_ms: 0.0,
            uniform_poly_ms: 0.0,
            uniform_bits_ms: 0.0,
            uniform_trig_ms: 0.0,
            fsm_ms: 0.0,
            size: 0,
        }
    }
}

impl Experiment for SimdTaskqueueExperiment {
    fn name(&self) -> &str {
        "simd_taskqueue"
    }

    fn description(&self) -> &str {
        "Divergent task dispatch (4 code paths) + SIMD FSM: zero divergence cost test"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;
        self.data_f32 = gen.uniform_f32(size);
        self.data_u32 = gen.uniform_u32(size);

        self.buf_input_f32 = Some(alloc_buffer_with_data(&ctx.device, &self.data_f32));
        self.buf_input_u32 = Some(alloc_buffer_with_data(&ctx.device, &self.data_u32));
        self.buf_output = Some(alloc_buffer(&ctx.device, size * 4));
        self.buf_output_u32 = Some(alloc_buffer(&ctx.device, size * 4));

        for name in TASK_KERNELS {
            self.pso_cache.get_or_create(ctx.library(), name);
        }
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input_f32 = self.buf_input_f32.as_ref().unwrap();
        let input_u32 = self.buf_input_u32.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let output_u32 = self.buf_output_u32.as_ref().unwrap();

        let params = ExploitParams {
            element_count: self.size as u32,
            num_passes: NUM_PASSES,
            mode: 0,
            _pad: 0,
        };
        let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

        // Run all float kernels: divergent + 4 uniform baselines
        let float_kernels = [
            "exploit_divergent_tasks",
            "exploit_uniform_hash",
            "exploit_uniform_poly",
            "exploit_uniform_bits",
            "exploit_uniform_trig",
        ];
        let mut float_times = [0.0f64; 5];
        for (i, name) in float_kernels.iter().enumerate() {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self.pso_cache.get_or_create(ctx.library(), name);
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input_f32.as_ref(), 0),
                    (output.as_ref(), 1),
                    (buf_params.as_ref(), 2),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            float_times[i] = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }
        self.divergent_ms = float_times[0];
        self.uniform_hash_ms = float_times[1];
        self.uniform_poly_ms = float_times[2];
        self.uniform_bits_ms = float_times[3];
        self.uniform_trig_ms = float_times[4];

        // SIMD FSM (uses u32 input/output)
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_simd_fsm");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input_u32.as_ref(), 0),
                    (output_u32.as_ref(), 1),
                    (buf_params.as_ref(), 2),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.fsm_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        self.divergent_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU: run hash as baseline
        let mut sum = 0u64;
        for &v in &self.data_f32 {
            let mut h = v.to_bits();
            for _ in 0..NUM_PASSES {
                h = h.wrapping_mul(0x9E3779B9);
                h ^= h >> 16;
                h = h.wrapping_mul(0x85EBCA6B);
                h ^= h >> 13;
            }
            sum = sum.wrapping_add(h as u64);
        }
        std::hint::black_box(sum);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("divergent_ms".to_string(), self.divergent_ms);
        m.insert("uniform_hash_ms".to_string(), self.uniform_hash_ms);
        m.insert("uniform_poly_ms".to_string(), self.uniform_poly_ms);
        m.insert("uniform_bits_ms".to_string(), self.uniform_bits_ms);
        m.insert("uniform_trig_ms".to_string(), self.uniform_trig_ms);
        m.insert("fsm_ms".to_string(), self.fsm_ms);

        // Key metric: divergence penalty
        // If zero divergence: divergent_time ≈ max(individual uniform times)
        // If serialized: divergent_time ≈ sum(individual uniform times)
        let max_uniform = self
            .uniform_hash_ms
            .max(self.uniform_poly_ms)
            .max(self.uniform_bits_ms)
            .max(self.uniform_trig_ms);
        let sum_uniform = self.uniform_hash_ms
            + self.uniform_poly_ms
            + self.uniform_bits_ms
            + self.uniform_trig_ms;

        if max_uniform > 0.0 {
            // divergence_penalty = 1.0 means zero cost (matches slowest task)
            // divergence_penalty = 4.0 means full serialization
            let penalty = self.divergent_ms / max_uniform;
            m.insert("divergence_penalty_x".to_string(), penalty);
        }

        if sum_uniform > 0.0 {
            // serialization_ratio: 0.0 = perfectly parallel, 1.0 = fully serial
            // (divergent - max) / (sum - max)
            let serial_ratio = if sum_uniform > max_uniform {
                (self.divergent_ms - max_uniform) / (sum_uniform - max_uniform)
            } else {
                0.0
            };
            m.insert("serialization_ratio".to_string(), serial_ratio);
        }

        m
    }
}
