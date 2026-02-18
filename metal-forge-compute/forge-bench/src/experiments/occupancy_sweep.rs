//! Occupancy Sweep experiment.
//!
//! On NVIDIA: more occupancy = better (hides latency via warp scheduling).
//! On Apple with Dynamic Caching: fewer threads per core = more pool per thread
//!   = larger L1 cache per thread = better for bandwidth-bound kernels.
//!
//! This experiment dispatches the SAME kernel with threadgroup sizes
//! 32, 64, 128, 256, 512, 1024 and measures bandwidth at each level.
//! Expected: a non-obvious sweet spot, not monotonically increasing.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLPipelineOption, MTLSize,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, ExploitParams, GpuTimer, MetalContext,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// Threadgroup sizes to sweep.
const TG_SIZES: [usize; 6] = [32, 64, 128, 256, 512, 1024];

pub struct OccupancySweepExperiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO compiled with maxTotalThreadsPerThreadgroup=1024 to allow full sweep.
    pso: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    tg_times: Vec<(usize, f64)>, // (tg_size, ms)
    best_tg_ms: f64,
    size: usize,
}

impl OccupancySweepExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_params: None,
            pso: None,
            tg_times: Vec::new(),
            best_tg_ms: 0.0,
            size: 0,
        }
    }

    fn dispatch_with_tg_size(
        &self,
        ctx: &MetalContext,
        tg_size: usize,
    ) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();
        let pso = self.pso.as_ref().unwrap();

        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();

        enc.setComputePipelineState(pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(params.as_ref()), 0, 2);
        }

        // Each thread processes 4 float4, so total threads = ceil(size / 4)
        let total_threads = self.size.div_ceil(4);
        let num_tgs = total_threads.div_ceil(tg_size);

        let grid = MTLSize {
            width: num_tgs,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        };

        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0)
    }
}

impl Experiment for OccupancySweepExperiment {
    fn name(&self) -> &str {
        "occupancy_sweep"
    }

    fn description(&self) -> &str {
        "Occupancy inversion: sweep TG size 32→1024, lower occupancy may win"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // float4 elements
        self.data = gen
            .uniform_u32(size * 4)
            .iter()
            .map(|&v| (v % 100) as f32 * 0.01)
            .collect();

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output: max possible partials (for TG=32: size/4/32 * 1 simdgroup)
        let max_partials = size.div_ceil(4).div_ceil(32) * 32; // generous
        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            max_partials * std::mem::size_of::<f32>(),
        ));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Compile PSO with maxTotalThreadsPerThreadgroup=1024
        // (PsoCache hardcodes 256 which blocks TG sizes 512/1024)
        let fn_name = NSString::from_str("exploit_occupancy_bw");
        #[allow(deprecated)]
        let function = ctx
            .library()
            .newFunctionWithName(&fn_name)
            .expect("exploit_occupancy_bw not found in metallib");

        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setMaxTotalThreadsPerThreadgroup(1024);
        unsafe {
            descriptor.setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        }

        self.pso = Some(
            ctx.device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &descriptor,
                    MTLPipelineOption::None,
                    None,
                )
                .expect("Failed to create occupancy_sweep PSO"),
        );
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        self.tg_times.clear();

        for &tg_size in &TG_SIZES {
            // Run 3 warmups then 3 measured
            for _ in 0..2 {
                self.dispatch_with_tg_size(ctx, tg_size);
            }

            let mut times = Vec::new();
            for _ in 0..3 {
                times.push(self.dispatch_with_tg_size(ctx, tg_size));
            }

            let median = {
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                times[1] // median of 3
            };

            self.tg_times.push((tg_size, median));
        }

        // Best time
        self.best_tg_ms = self
            .tg_times
            .iter()
            .map(|&(_, ms)| ms)
            .fold(f64::MAX, f64::min);

        self.best_tg_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();
        let sum: f64 = self.data.iter().map(|&x| x as f64).sum();
        std::hint::black_box(sum);
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.tg_times.len() == TG_SIZES.len() {
            Ok(())
        } else {
            Err(format!(
                "Expected {} TG measurements, got {}",
                TG_SIZES.len(),
                self.tg_times.len()
            ))
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let bytes = size as f64 * 16.0; // float4 = 16 bytes

        // Per-TG-size timing
        for &(tg_size, ms) in &self.tg_times {
            m.insert(format!("tg_{}_ms", tg_size), ms);

            let gbs = if ms > 0.0 {
                bytes / (ms / 1000.0) / 1e9
            } else {
                0.0
            };
            m.insert(format!("tg_{}_gbs", tg_size), gbs);
        }

        // Best TG size and its bandwidth
        if let Some(&(best_tg, best_ms)) = self
            .tg_times
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            m.insert("best_tg_size".to_string(), best_tg as f64);
            m.insert("best_ms".to_string(), best_ms);

            let best_gbs = if best_ms > 0.0 {
                bytes / (best_ms / 1000.0) / 1e9
            } else {
                0.0
            };
            m.insert("gb_per_sec".to_string(), best_gbs);
        }

        // Worst TG size for contrast
        if let Some(&(worst_tg, worst_ms)) = self
            .tg_times
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            m.insert("worst_tg_size".to_string(), worst_tg as f64);
            m.insert("worst_ms".to_string(), worst_ms);

            let inversion_ratio = if self.best_tg_ms > 0.0 {
                worst_ms / self.best_tg_ms
            } else {
                0.0
            };
            m.insert("occupancy_inversion_x".to_string(), inversion_ratio);
        }

        m
    }
}
