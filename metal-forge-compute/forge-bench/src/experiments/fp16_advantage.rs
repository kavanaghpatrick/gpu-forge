//! FP16 Triple Advantage experiment.
//!
//! FP16 provides >2x speedup from three compounding effects:
//!   (a) 2x ALU throughput (hardware)
//!   (b) Lower register dependency penalty (0.56 vs 0.84 cycles)
//!   (c) Half-sized values = 2x register cache capacity
//!
//! FP16↔FP32 conversions are FREE on Apple GPU.
//! We use 32 multiply-add iterations to make this compute-bound (not BW-bound),
//! isolating the ALU throughput + register effects.

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

pub struct Fp16AdvantageExperiment {
    data_f32: Vec<f32>,
    data_f16: Vec<u16>, // half-precision stored as u16 bits
    buf_input_f32: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_f32: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_input_f16: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_f16: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_fp16_ms: f64,
    gpu_fp32_ms: f64,
    size: usize,
}

impl Fp16AdvantageExperiment {
    pub fn new() -> Self {
        Self {
            data_f32: Vec::new(),
            data_f16: Vec::new(),
            buf_input_f32: None,
            buf_output_f32: None,
            buf_input_f16: None,
            buf_output_f16: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_fp16_ms: 0.0,
            gpu_fp32_ms: 0.0,
            size: 0,
        }
    }
}

/// Convert f32 to IEEE 754 half-precision (f16) stored as u16.
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;

    if exp == 0 {
        // Zero or denorm → f16 zero
        return sign as u16;
    }
    if exp == 0xFF {
        // Inf or NaN
        if frac == 0 {
            return (sign | 0x7C00) as u16;
        } else {
            return (sign | 0x7E00) as u16; // quiet NaN
        }
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow → inf
    }
    if new_exp <= 0 {
        return sign as u16; // underflow → zero
    }

    let new_frac = frac >> 13;
    (sign | ((new_exp as u32) << 10) | new_frac) as u16
}

impl Experiment for Fp16AdvantageExperiment {
    fn name(&self) -> &str {
        "fp16_advantage"
    }

    fn description(&self) -> &str {
        "FP16 triple advantage: throughput + register + cache compounding vs FP32"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // float4 element count
        let float4_count = size;

        // Generate f32 data (small values to stay in f16 range)
        self.data_f32 = gen
            .uniform_u32(float4_count * 4)
            .iter()
            .map(|&v| (v % 100) as f32 * 0.01)
            .collect();

        // Convert to f16 (each float4 = 4 half values = 8 bytes = 4 u16)
        self.data_f16 = self.data_f32.iter().map(|&v| f32_to_f16(v)).collect();

        // Allocate f32 buffers: float4 = 16 bytes each
        self.buf_input_f32 = Some(alloc_buffer_with_data(&ctx.device, &self.data_f32));
        self.buf_output_f32 = Some(alloc_buffer(
            &ctx.device,
            float4_count * 4 * std::mem::size_of::<f32>(),
        ));

        // Allocate f16 buffers: half4 = 8 bytes each
        self.buf_input_f16 = Some(alloc_buffer_with_data(&ctx.device, &self.data_f16));
        self.buf_output_f16 = Some(alloc_buffer(
            &ctx.device,
            float4_count * 4 * std::mem::size_of::<u16>(),
        ));

        let params = ExploitParams {
            element_count: float4_count as u32,
            num_passes: 32, // iterations
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_fp16_compute");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_fp32_compute");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let params = self.buf_params.as_ref().unwrap();
        let float4_count = self.size;

        // --- FP16 ---
        {
            let input = self.buf_input_f16.as_ref().unwrap();
            let output = self.buf_output_f16.as_ref().unwrap();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_fp16_compute");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                float4_count,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_fp16_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // --- FP32 ---
        {
            let input = self.buf_input_f32.as_ref().unwrap();
            let output = self.buf_output_f32.as_ref().unwrap();
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_fp32_compute");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                float4_count,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_fp32_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        self.gpu_fp16_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // CPU f32 baseline: same 32 iterations
        for val in self.data_f32.iter_mut() {
            let mut v = *val;
            for _ in 0..32 {
                v = v * 1.001 + 0.5;
            }
            *val = v;
        }

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_fp16_ms > 0.0 && self.gpu_fp32_ms > 0.0 {
            Ok(())
        } else {
            Err("No GPU timing recorded".to_string())
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("fp16_ms".to_string(), self.gpu_fp16_ms);
        m.insert("fp32_ms".to_string(), self.gpu_fp32_ms);

        let advantage = if self.gpu_fp16_ms > 0.0 {
            self.gpu_fp32_ms / self.gpu_fp16_ms
        } else {
            0.0
        };
        m.insert("fp16_advantage_x".to_string(), advantage);

        // FP16 GFLOPS: size float4 elements * 4 components * 32 iterations * 2 ops (mul+add)
        let flops = size as f64 * 4.0 * 32.0 * 2.0;
        let fp16_gflops = if self.gpu_fp16_ms > 0.0 {
            flops / (self.gpu_fp16_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        let fp32_gflops = if self.gpu_fp32_ms > 0.0 {
            flops / (self.gpu_fp32_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("fp16_gflops".to_string(), fp16_gflops);
        m.insert("fp32_gflops".to_string(), fp32_gflops);

        // Use FP16 bandwidth for the table
        let bytes = size as f64 * 8.0 * 2.0; // half4 read + write = 16 bytes
        let gbs = if self.gpu_fp16_ms > 0.0 {
            bytes / (self.gpu_fp16_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
