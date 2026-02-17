//! Reduce experiment: GPU vs CPU parallel reduction.
//!
//! Tests reduce_sum_u32 kernel. Setup generates uniform u32 data,
//! creates Metal buffers (input + atomic output), and compares GPU
//! result against rayon CPU baseline.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer, BenchTimer, MetalContext,
    PsoCache, ReduceParams,
};

use crate::cpu_baselines::rayon_reduce;
use crate::data_gen::DataGenerator;

use super::Experiment;

/// Reduce sum experiment using atomic global accumulation.
pub struct ReduceExperiment {
    /// Input data kept for CPU baseline and validation.
    data: Vec<u32>,
    /// Metal buffer holding the input data.
    input_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for atomic output result (single u32).
    output_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for ReduceParams constant.
    params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU result from last run.
    gpu_result: u64,
    /// CPU result from last run.
    cpu_result: u64,
    /// Current element count.
    size: usize,
}

impl ReduceExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            input_buffer: None,
            output_buffer: None,
            params_buffer: None,
            pso_cache: PsoCache::new(),
            gpu_result: 0,
            cpu_result: 0,
            size: 0,
        }
    }

    /// Zero the output buffer before each GPU run.
    fn zero_output_buffer(&self) {
        if let Some(ref buf) = self.output_buffer {
            unsafe {
                let ptr = buf.contents().as_ptr() as *mut u32;
                *ptr = 0;
            }
        }
    }
}

impl Experiment for ReduceExperiment {
    fn name(&self) -> &str {
        "reduce"
    }

    fn description(&self) -> &str {
        "Parallel reduction (sum_u32): 3-level SIMD->threadgroup->atomic"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![
            100_000,     // 100K
            1_000_000,   // 1M
            10_000_000,  // 10M
            100_000_000, // 100M
        ]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;
        self.data = gen.uniform_u32(size);

        // Input buffer: N x u32
        self.input_buffer = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output buffer: single atomic u32 (zeroed)
        self.output_buffer = Some(alloc_buffer(&ctx.device, std::mem::size_of::<u32>()));
        self.zero_output_buffer();

        // Params buffer
        let params = ReduceParams {
            element_count: size as u32,
            _pad: [0; 3],
        };
        self.params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Pre-warm PSO cache
        self.pso_cache.get_or_create(ctx.library(), "reduce_sum_u32");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        // Zero output before each run
        self.zero_output_buffer();

        let input = self.input_buffer.as_ref().expect("setup not called");
        let output = self.output_buffer.as_ref().expect("setup not called");
        let params = self.params_buffer.as_ref().expect("setup not called");

        let pso = self.pso_cache.get_or_create(ctx.library(), "reduce_sum_u32");

        let timer = BenchTimer::start();

        // Create command buffer and encoder
        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        // Dispatch reduce kernel
        dispatch_1d(
            &encoder,
            pso,
            &[
                (input.as_ref(), 0),
                (output.as_ref(), 1),
                (params.as_ref(), 2),
            ],
            self.size,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let elapsed = timer.stop();

        // Read back result
        self.gpu_result = unsafe { read_buffer::<u32>(output.as_ref()) } as u64;

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();
        self.cpu_result = rayon_reduce::par_sum_u32(&self.data);
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_result == self.cpu_result {
            Ok(())
        } else {
            Err(format!(
                "GPU sum ({}) != CPU sum ({}), diff = {}",
                self.gpu_result,
                self.cpu_result,
                (self.gpu_result as i128 - self.cpu_result as i128).unsigned_abs()
            ))
        }
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let bytes = size as f64 * 4.0; // u32 = 4 bytes
        let seconds = elapsed_ms / 1000.0;
        let gbs = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        // Bandwidth utilization requires hardware info -- caller can compute from gb_per_sec
        m.insert("bytes_processed".to_string(), bytes);
        m.insert("elements".to_string(), size as f64);

        m
    }
}
