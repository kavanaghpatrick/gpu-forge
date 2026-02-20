//! Filter experiment: GPU columnar filter vs CPU rayon parallel filter.
//!
//! Tests a simple greater-than predicate on u32 data. The GPU kernel uses
//! simd_sum for efficient per-threadgroup match counting with atomic
//! accumulation to a global counter.
//!
//! Uses threshold = u32::MAX / 2 for ~50% selectivity on uniform random data.
//! Validates match count equality between GPU and CPU.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, read_buffer, BenchTimer, FilterBenchParams, GpuTimer,
    MetalContext, PsoCache,
};

use crate::cpu_baselines::rayon_filter;
use crate::data_gen::DataGenerator;

use super::Experiment;

/// Filter experiment comparing GPU atomic-counted filter vs rayon parallel filter.
pub struct FilterExperiment {
    /// Input data kept for CPU baseline and validation.
    data: Vec<u32>,
    /// Threshold for the predicate (value > threshold).
    threshold: u32,
    /// Metal buffer holding input data.
    input_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for atomic match count output (single u32).
    output_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for FilterBenchParams.
    params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU match count from last run.
    gpu_match_count: u32,
    /// CPU match count from last run.
    cpu_match_count: u32,
    /// Current element count.
    size: usize,
}

impl FilterExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            threshold: 0,
            input_buffer: None,
            output_buffer: None,
            params_buffer: None,
            pso_cache: PsoCache::new(),
            gpu_match_count: 0,
            cpu_match_count: 0,
            size: 0,
        }
    }

    /// Zero the output buffer (atomic counter) before each GPU run.
    fn zero_output_buffer(&self) {
        if let Some(ref buf) = self.output_buffer {
            unsafe {
                let ptr = buf.contents().as_ptr() as *mut u32;
                *ptr = 0;
            }
        }
    }
}

impl Experiment for FilterExperiment {
    fn name(&self) -> &str {
        "filter"
    }

    fn description(&self) -> &str {
        "Columnar filter (u32 > threshold): GPU simd_sum+atomic vs rayon par_filter"
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

        // Use u32::MAX / 2 as threshold for ~50% selectivity on uniform random data
        self.threshold = u32::MAX / 2;

        // Input buffer: N x u32
        self.input_buffer = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output buffer: single atomic u32 (match count)
        self.output_buffer = Some(alloc_buffer(&ctx.device, std::mem::size_of::<u32>()));
        self.zero_output_buffer();

        // Params buffer
        let params = FilterBenchParams {
            element_count: size as u32,
            threshold: self.threshold,
            _pad: [0; 2],
        };
        self.params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Pre-warm PSO cache
        self.pso_cache
            .get_or_create(ctx.library(), "filter_count_gt");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        // Zero output before each run
        self.zero_output_buffer();

        let input = self.input_buffer.as_ref().expect("setup not called");
        let output = self.output_buffer.as_ref().expect("setup not called");
        let params = self.params_buffer.as_ref().expect("setup not called");

        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "filter_count_gt");

        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        // Each thread handles 4 elements via uint4 vectorized loads
        forge_primitives::dispatch_1d(
            &encoder,
            pso,
            &[
                (input.as_ref(), 0),
                (output.as_ref(), 1),
                (params.as_ref(), 2),
            ],
            self.size.div_ceil(4),
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read back match count
        self.gpu_match_count = unsafe { read_buffer::<u32>(output.as_ref()) };

        GpuTimer::elapsed_ms(&cmd_buf).unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();
        let result = rayon_filter::par_filter_gt(&self.data, self.threshold);
        self.cpu_match_count = result.len() as u32;
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_match_count == self.cpu_match_count {
            Ok(())
        } else {
            Err(format!(
                "Match count mismatch: GPU={} CPU={} (threshold={}, size={})",
                self.gpu_match_count, self.cpu_match_count, self.threshold, self.size
            ))
        }
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let seconds = elapsed_ms / 1000.0;

        // Rows per second
        let rows_per_sec = if seconds > 0.0 {
            size as f64 / seconds
        } else {
            0.0
        };
        m.insert("rows_per_sec".to_string(), rows_per_sec);
        m.insert("elements".to_string(), size as f64);

        // Selectivity percentage
        let selectivity = if size > 0 {
            self.gpu_match_count as f64 / size as f64 * 100.0
        } else {
            0.0
        };
        m.insert("selectivity_pct".to_string(), selectivity);
        m.insert("match_count".to_string(), self.gpu_match_count as f64);

        // Bandwidth: read input(N*4) bytes
        let bytes = size as f64 * 4.0;
        let gbs = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
