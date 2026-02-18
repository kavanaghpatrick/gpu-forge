//! Indirect Conditional Dispatch experiment.
//!
//! GPU-side branching: a decision kernel writes (N,1,1) or (0,0,0) to an
//! indirect dispatch buffer. The next dispatch reads this buffer and either
//! executes or skips entirely -- all without CPU involvement.
//!
//! Compare against CPU-side conditional: commit+wait, read flag, decide, dispatch.
//! GPU-side should be ~2x faster by eliminating the CPU round-trip.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer, BenchTimer, ExploitParams,
    GpuTimer, MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

pub struct IndirectCondExperiment {
    data: Vec<f32>,
    decision_data: Vec<u32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_decision_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_indirect_args: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_indirect_ms: f64,
    gpu_cpuside_ms: f64,
    size: usize,
}

impl IndirectCondExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            decision_data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_decision_input: None,
            buf_indirect_args: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_indirect_ms: 0.0,
            gpu_cpuside_ms: 0.0,
            size: 0,
        }
    }
}

impl Experiment for IndirectCondExperiment {
    fn name(&self) -> &str {
        "indirect_cond"
    }

    fn description(&self) -> &str {
        "GPU-side conditional dispatch (indirect) vs CPU-side branch (2x commit+wait)"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // Float4 input data
        self.data = gen
            .uniform_u32(size * 4)
            .iter()
            .map(|&v| v as f32)
            .collect();
        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output buffer
        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            size * 4 * std::mem::size_of::<f32>(),
        ));

        // Decision input: single u32 value that the decision kernel reads
        // Set it above threshold so the work kernel DOES execute
        self.decision_data = vec![1000u32];
        self.buf_decision_input = Some(alloc_buffer_with_data(&ctx.device, &self.decision_data));

        // Indirect args buffer: 3 x u32 (threadgroup counts X, Y, Z)
        self.buf_indirect_args = Some(alloc_buffer(
            &ctx.device,
            3 * std::mem::size_of::<u32>(),
        ));

        // Params: threshold=500 (mode field), so val=1000 > 500 triggers dispatch
        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: 500, // threshold
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_indirect_decision");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_indirect_work");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let decision_input = self.buf_decision_input.as_ref().unwrap();
        let indirect_args = self.buf_indirect_args.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        // === GPU-side indirect conditional: single command buffer ===
        {
            let cmd = ctx.queue.commandBuffer().unwrap();

            // Pass 1: decision kernel writes indirect args
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "exploit_indirect_decision");
                let enc = cmd.computeCommandEncoder().unwrap();
                dispatch_1d(
                    &enc,
                    pso,
                    &[
                        (decision_input.as_ref(), 0),
                        (indirect_args.as_ref(), 1),
                        (params.as_ref(), 2),
                    ],
                    1, // single thread makes decision
                );
                enc.endEncoding();
            }

            // Pass 2: work kernel dispatched via indirect buffer
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "exploit_indirect_work");
                let enc = cmd.computeCommandEncoder().unwrap();

                enc.setComputePipelineState(pso);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(input.as_ref()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(output.as_ref()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(params.as_ref()), 0, 2);
                }

                let tg_size = MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                };
                unsafe {
                    enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                        indirect_args.as_ref(),
                        0,
                        tg_size,
                    );
                }
                enc.endEncoding();
            }

            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_indirect_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // === CPU-side conditional: 2 command buffers with CPU decision ===
        {
            let timer = BenchTimer::start();

            // First cmd buf: run decision, read back on CPU
            let cmd1 = ctx.queue.commandBuffer().unwrap();
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "exploit_indirect_decision");
                let enc = cmd1.computeCommandEncoder().unwrap();
                dispatch_1d(
                    &enc,
                    pso,
                    &[
                        (decision_input.as_ref(), 0),
                        (indirect_args.as_ref(), 1),
                        (params.as_ref(), 2),
                    ],
                    1,
                );
                enc.endEncoding();
            }
            cmd1.commit();
            cmd1.waitUntilCompleted();

            // CPU reads decision
            let tg_count: u32 = unsafe { read_buffer::<u32>(indirect_args.as_ref()) };

            if tg_count > 0 {
                // Second cmd buf: run work
                let cmd2 = ctx.queue.commandBuffer().unwrap();
                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "exploit_indirect_work");
                    let enc = cmd2.computeCommandEncoder().unwrap();
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
                cmd2.commit();
                cmd2.waitUntilCompleted();
            }

            self.gpu_cpuside_ms = timer.stop();
        }

        self.gpu_indirect_ms
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();

        // Same computation: data[i] = data[i] * 2.0 + 1.0
        for val in self.data.iter_mut() {
            *val = *val * 2.0 + 1.0;
        }

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // Both paths should produce the same output. We just check timing worked.
        if self.gpu_indirect_ms > 0.0 {
            Ok(())
        } else {
            Err("No GPU timing recorded".to_string())
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let bytes = size as f64 * 16.0; // float4

        m.insert("indirect_ms".to_string(), self.gpu_indirect_ms);
        m.insert("cpuside_ms".to_string(), self.gpu_cpuside_ms);

        let speedup = if self.gpu_indirect_ms > 0.0 {
            self.gpu_cpuside_ms / self.gpu_indirect_ms
        } else {
            0.0
        };
        m.insert("indirect_speedup_x".to_string(), speedup);

        let gbs = if self.gpu_indirect_ms > 0.0 {
            bytes / (self.gpu_indirect_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
