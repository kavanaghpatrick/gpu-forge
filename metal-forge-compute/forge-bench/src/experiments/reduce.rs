//! Reduce experiment: GPU vs CPU parallel reduction.
//!
//! Tests two-pass atomic-free reduce_sum_u32_v2 + reduce_sum_partials kernels.
//! Setup generates uniform u32 data, creates Metal buffers, and compares
//! GPU result against rayon CPU baseline.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer, BenchTimer, GpuTimer,
    MetalContext, PsoCache, ReduceParams,
};

use rayon::prelude::*;

use crate::data_gen::DataGenerator;

use super::Experiment;

/// Elements processed per threadgroup: 256 threads * 4 elements/thread.
const ELEMENTS_PER_TG: usize = 256 * 4; // 1024

/// Max partials that reduce_sum_partials can handle in a single TG dispatch
/// (256 threads * 4 elements/thread = 1024).
const MAX_SINGLE_TG_PARTIALS: usize = 1024;

/// Two-pass atomic-free reduce sum experiment.
pub struct ReduceExperiment {
    /// Input data kept for CPU baseline and validation.
    data: Vec<u32>,
    /// Metal buffer holding the input data.
    input_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for final output result (single u32).
    output_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for ReduceParams constant (pass 1: element_count = N).
    params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for per-TG partial sums from pass 1.
    partials_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for ReduceParams for pass 2 (element_count = num_partials).
    partials_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for level-2 partial sums (only used when >1024 partials).
    partials2_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for ReduceParams for pass 3 (element_count = num_partials2).
    partials2_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU result from last run.
    gpu_result: u64,
    /// CPU result from last run.
    cpu_result: u64,
    /// Current element count.
    size: usize,
    /// Number of partials from pass 1.
    num_partials: usize,
}

impl ReduceExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            input_buffer: None,
            output_buffer: None,
            params_buffer: None,
            partials_buffer: None,
            partials_params_buffer: None,
            partials2_buffer: None,
            partials2_params_buffer: None,
            pso_cache: PsoCache::new(),
            gpu_result: 0,
            cpu_result: 0,
            size: 0,
            num_partials: 0,
        }
    }
}

impl Experiment for ReduceExperiment {
    fn name(&self) -> &str {
        "reduce"
    }

    fn description(&self) -> &str {
        "Parallel reduction (sum_u32): two-pass atomic-free SIMD+TG reduce"
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

        // Number of threadgroups (= number of partials) for pass 1
        self.num_partials = size.div_ceil(ELEMENTS_PER_TG);

        // Input buffer: N x u32
        self.input_buffer = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output buffer: single u32
        self.output_buffer = Some(alloc_buffer(&ctx.device, std::mem::size_of::<u32>()));

        // Partials buffer: one u32 per threadgroup from pass 1
        self.partials_buffer = Some(alloc_buffer(
            &ctx.device,
            self.num_partials * std::mem::size_of::<u32>(),
        ));

        // Params buffer for pass 1 (element_count = N)
        let params = ReduceParams {
            element_count: size as u32,
            _pad: [0; 3],
        };
        self.params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Params buffer for pass 2 (element_count = num_partials)
        let partials_params = ReduceParams {
            element_count: self.num_partials as u32,
            _pad: [0; 3],
        };
        self.partials_params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[partials_params]));

        // If num_partials > MAX_SINGLE_TG_PARTIALS, we need a 3rd level
        if self.num_partials > MAX_SINGLE_TG_PARTIALS {
            let num_partials2 = self.num_partials.div_ceil(ELEMENTS_PER_TG);
            self.partials2_buffer = Some(alloc_buffer(
                &ctx.device,
                num_partials2 * std::mem::size_of::<u32>(),
            ));
            let partials2_params = ReduceParams {
                element_count: num_partials2 as u32,
                _pad: [0; 3],
            };
            self.partials2_params_buffer =
                Some(alloc_buffer_with_data(&ctx.device, &[partials2_params]));
        }

        // Pre-warm PSO cache for both kernels
        self.pso_cache
            .get_or_create(ctx.library(), "reduce_sum_u32_v2");
        self.pso_cache
            .get_or_create(ctx.library(), "reduce_sum_partials");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.input_buffer.as_ref().expect("setup not called");
        let output = self.output_buffer.as_ref().expect("setup not called");
        let params = self.params_buffer.as_ref().expect("setup not called");
        let partials = self.partials_buffer.as_ref().expect("setup not called");
        let partials_params = self
            .partials_params_buffer
            .as_ref()
            .expect("setup not called");

        // Single command buffer for all passes
        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");

        // === Pass 1: reduce_sum_u32_v2 ===
        // Each thread handles 4 elements, so total_threads = ceil(N / 4).
        // dispatch_1d computes ceil(total_threads / 256) = ceil(N / 1024) threadgroups.
        {
            let pso_v2 = self
                .pso_cache
                .get_or_create(ctx.library(), "reduce_sum_u32_v2");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");
            dispatch_1d(
                &encoder,
                pso_v2,
                &[
                    (input.as_ref(), 0),
                    (partials.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                self.size.div_ceil(4),
            );
            encoder.endEncoding();
        }

        if self.num_partials <= MAX_SINGLE_TG_PARTIALS {
            // === Pass 2: reduce_sum_partials (single TG) ===
            // Reduces partials array into final result.
            let pso_partials = self
                .pso_cache
                .get_or_create(ctx.library(), "reduce_sum_partials");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");
            dispatch_1d(
                &encoder,
                pso_partials,
                &[
                    (partials.as_ref(), 0),
                    (output.as_ref(), 1),
                    (partials_params.as_ref(), 2),
                ],
                self.num_partials.div_ceil(4),
            );
            encoder.endEncoding();
        } else {
            // === 3-level reduction: partials -> partials2 -> result ===
            let partials2 = self.partials2_buffer.as_ref().expect("setup not called");
            let partials2_params = self
                .partials2_params_buffer
                .as_ref()
                .expect("setup not called");

            let num_partials2 = self.num_partials.div_ceil(ELEMENTS_PER_TG);

            // Pass 2: reduce partials -> partials2 using reduce_sum_u32_v2
            {
                let pso_v2 = self
                    .pso_cache
                    .get_or_create(ctx.library(), "reduce_sum_u32_v2");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                dispatch_1d(
                    &encoder,
                    pso_v2,
                    &[
                        (partials.as_ref(), 0),
                        (partials2.as_ref(), 1),
                        (partials_params.as_ref(), 2),
                    ],
                    self.num_partials.div_ceil(4),
                );
                encoder.endEncoding();
            }

            // Pass 3: reduce partials2 -> result using reduce_sum_partials (single TG)
            {
                let pso_partials = self
                    .pso_cache
                    .get_or_create(ctx.library(), "reduce_sum_partials");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                dispatch_1d(
                    &encoder,
                    pso_partials,
                    &[
                        (partials2.as_ref(), 0),
                        (output.as_ref(), 1),
                        (partials2_params.as_ref(), 2),
                    ],
                    num_partials2.div_ceil(4),
                );
                encoder.endEncoding();
            }
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let elapsed = GpuTimer::elapsed_ms(&cmd_buf).unwrap_or(0.0);

        // Read back result
        self.gpu_result = unsafe { read_buffer::<u32>(output.as_ref()) } as u64;

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();
        // Compute wrapping u32 sum to match GPU's u32 arithmetic.
        // u32 addition is associative under modular arithmetic, so the
        // result is independent of evaluation order.
        self.cpu_result = self.data.par_iter().copied().map(|x| x as u64).sum::<u64>();
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // Compare as u32 (wrapping) since the GPU operates on u32.
        // At large N the u64 sum overflows u32, but both GPU and CPU
        // produce the same wrapped result.
        let gpu_u32 = self.gpu_result as u32;
        let cpu_u32 = self.cpu_result as u32;
        if gpu_u32 == cpu_u32 {
            Ok(())
        } else {
            Err(format!(
                "GPU sum ({}) != CPU sum ({}), diff = {}",
                gpu_u32,
                cpu_u32,
                (gpu_u32 as i64 - cpu_u32 as i64).unsigned_abs()
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
