//! Memory Stride Bandwidth experiment.
//!
//! Measures effective bandwidth at different access strides.
//! Apple GPU L1 is 8KB, cache line is 128 bytes.
//! Sequential access (stride=1) should be optimal.
//! Strided access wastes cache lines proportionally.
//!
//! stride=1: threads read consecutive float4 elements (coalesced)
//! stride=32: threads read every 32nd float4 (128-byte gaps, 1/32 cache utilization)

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

const STRIDES: [usize; 6] = [1, 2, 4, 8, 16, 32];
const STRIDE_KERNEL_NAMES: [&str; 6] = [
    "exploit_stride_1",
    "exploit_stride_2",
    "exploit_stride_4",
    "exploit_stride_8",
    "exploit_stride_16",
    "exploit_stride_32",
];

pub struct StrideBandwidthExperiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    stride_times: [f64; 6],
    size: usize,
}

impl StrideBandwidthExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            stride_times: [0.0; 6],
            size: 0,
        }
    }
}

impl Experiment for StrideBandwidthExperiment {
    fn name(&self) -> &str {
        "stride_bandwidth"
    }

    fn description(&self) -> &str {
        "Memory stride bandwidth: sequential vs strided float4 access (cache line waste)"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // Input: size float4 elements = size * 4 floats
        // Need enough for stride=32: threads access up to input[tid*32]
        // So input buffer needs size float4 elements minimum
        self.data = gen
            .uniform_u32(size * 4)
            .iter()
            .map(|&v| (v % 1000) as f32 * 0.001)
            .collect();

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output: one float per simdgroup leader, max = ceil(size/32) floats
        let max_outputs = size.div_ceil(32) * 8; // generous
        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            max_outputs * std::mem::size_of::<f32>(),
        ));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        for name in &STRIDE_KERNEL_NAMES {
            self.pso_cache.get_or_create(ctx.library(), name);
        }
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        for (i, name) in STRIDE_KERNEL_NAMES.iter().enumerate() {
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
            self.stride_times[i] = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // Return stride=1 time as the GPU time
        self.stride_times[0]
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // Sequential sum of float4 data
        let mut sum = 0.0f64;
        for chunk in self.data.chunks(4) {
            let v: f64 = chunk.iter().map(|&x| x as f64).sum();
            sum += v;
        }
        std::hint::black_box(sum);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        for (i, &t) in self.stride_times.iter().enumerate() {
            if t == 0.0 {
                return Err(format!(
                    "Stride {} ({}) has zero timing",
                    STRIDES[i], STRIDE_KERNEL_NAMES[i]
                ));
            }
        }
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        // Per-stride timing and bandwidth
        for (i, &stride) in STRIDES.iter().enumerate() {
            m.insert(format!("stride_{}_ms", stride), self.stride_times[i]);

            // Active threads = size / stride; each reads 1 float4 = 16 bytes
            let active_threads = size / stride;
            let useful_bytes = active_threads as f64 * 16.0;
            let gbs = if self.stride_times[i] > 0.0 {
                useful_bytes / (self.stride_times[i] / 1000.0) / 1e9
            } else {
                0.0
            };
            m.insert(format!("stride_{}_gbs", stride), gbs);
        }

        // Stride degradation ratio: stride_32 / stride_1
        let degradation = if self.stride_times[0] > 0.0 {
            self.stride_times[5] / self.stride_times[0]
        } else {
            0.0
        };
        m.insert("stride_degradation_x".to_string(), degradation);

        // Use stride=1 bandwidth as the headline number
        let bytes = size as f64 * 16.0;
        let gbs = if self.stride_times[0] > 0.0 {
            bytes / (self.stride_times[0] / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
