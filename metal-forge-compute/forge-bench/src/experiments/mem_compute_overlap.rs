//! Memory-Compute Overlap (Latency Hiding) Probe.
//!
//! Tests whether Apple M4 GPU overlaps memory loads with ALU compute
//! within the same SIMD group.
//!
//! Three kernels:
//!   A) Memory-only: strided reads, sum results
//!   B) Compute-only: pure FMA chains, no memory after initial load
//!   C) Interleaved: alternating loads + FMAs
//!
//! If overlap exists: interleaved_time ≈ max(mem, compute), not mem + compute.
//! The overlap_ratio = 1 - (interleaved / (mem + compute)) measures efficiency.

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

const KERNELS: &[&str] = &[
    "exploit_mem_only",
    "exploit_compute_only",
    "exploit_mem_compute_interleaved",
];

const NUM_PASSES: u32 = 200;

pub struct MemComputeOverlapExperiment {
    data: Vec<f32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    times: [f64; 3], // mem_only, compute_only, interleaved
    size: usize,
}

impl MemComputeOverlapExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output: None,
            pso_cache: PsoCache::new(),
            times: [0.0; 3],
            size: 0,
        }
    }
}

impl Experiment for MemComputeOverlapExperiment {
    fn name(&self) -> &str {
        "mem_compute_overlap"
    }

    fn description(&self) -> &str {
        "Memory-compute overlap: does Apple GPU hide memory latency with ALU work?"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;
        self.data = gen.uniform_f32(size);

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));
        self.buf_output = Some(alloc_buffer(&ctx.device, size * 4));

        for name in KERNELS {
            self.pso_cache.get_or_create(ctx.library(), name);
        }
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();

        let params = ExploitParams {
            element_count: self.size as u32,
            num_passes: NUM_PASSES,
            mode: 0,
            _pad: 0,
        };
        let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

        for (i, name) in KERNELS.iter().enumerate() {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self.pso_cache.get_or_create(ctx.library(), name);
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output.as_ref(), 1),
                    (buf_params.as_ref(), 2),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.times[i] = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // Return interleaved time as primary
        self.times[2]
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        let mut sum = 0.0f64;
        let mut v = 1.0f64;
        for &val in &self.data {
            sum += val as f64;
            for _ in 0..4 {
                v = v.mul_add(1.00001, 0.00001);
            }
        }
        std::hint::black_box((sum, v));

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, _size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("mem_only_ms".to_string(), self.times[0]);
        m.insert("compute_only_ms".to_string(), self.times[1]);
        m.insert("interleaved_ms".to_string(), self.times[2]);

        let sum = self.times[0] + self.times[1];
        let max = self.times[0].max(self.times[1]);

        // Overlap ratio: 1.0 = perfect overlap, 0.0 = no overlap (serial)
        // overlap = 1 - (interleaved - max) / (sum - max)
        if sum > max && sum > 0.0 {
            let overlap = 1.0 - (self.times[2] - max) / (sum - max);
            m.insert("overlap_ratio".to_string(), overlap.clamp(0.0, 1.0));
        }

        // Speedup vs serial execution
        if self.times[2] > 0.0 {
            m.insert("serial_vs_interleaved_x".to_string(), sum / self.times[2]);
        }

        // Individual bandwidth/compute rates
        if self.times[0] > 0.0 {
            let bytes_read = self.size as f64 * 4.0 * NUM_PASSES as f64;
            m.insert(
                "mem_bandwidth_gbs".to_string(),
                (bytes_read / 1e9) / (self.times[0] / 1000.0),
            );
        }

        if self.times[1] > 0.0 {
            let flops = self.size as f64 * NUM_PASSES as f64 * 4.0 * 2.0; // 4 FMAs per pass
            m.insert(
                "compute_gflops".to_string(),
                flops / (self.times[1] / 1000.0) / 1e9,
            );
        }

        m
    }
}
