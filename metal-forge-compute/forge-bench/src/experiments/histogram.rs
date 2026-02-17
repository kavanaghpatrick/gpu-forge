//! Histogram experiment: GPU shared-memory histogram vs CPU sequential baseline.
//!
//! Tests histogram_256 kernel. Setup generates uniform u32 data,
//! creates Metal buffers (input + global histogram output), and compares
//! GPU result against sequential CPU histogram.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer_slice, BenchTimer,
    HistogramParams, MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// Number of histogram bins for the 256-bin kernel.
const NUM_BINS: u32 = 256;

/// Histogram experiment using shared-memory 256-bin accumulation.
pub struct HistogramExperiment {
    /// Input data kept for CPU baseline and validation.
    data: Vec<u32>,
    /// Metal buffer holding the input data.
    input_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for global histogram output (256 x u32).
    output_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for HistogramParams constant.
    params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU histogram result from last run.
    gpu_result: Vec<u32>,
    /// CPU histogram result from last run.
    cpu_result: Vec<u32>,
    /// Current element count.
    size: usize,
}

impl HistogramExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            input_buffer: None,
            output_buffer: None,
            params_buffer: None,
            pso_cache: PsoCache::new(),
            gpu_result: Vec::new(),
            cpu_result: Vec::new(),
            size: 0,
        }
    }

    /// Zero the output histogram buffer before each GPU run.
    fn zero_output_buffer(&self) {
        if let Some(ref buf) = self.output_buffer {
            unsafe {
                let ptr = buf.contents().as_ptr() as *mut u32;
                for i in 0..NUM_BINS as usize {
                    *ptr.add(i) = 0;
                }
            }
        }
    }
}

impl Experiment for HistogramExperiment {
    fn name(&self) -> &str {
        "histogram"
    }

    fn description(&self) -> &str {
        "Shared-memory 256-bin histogram: threadgroup atomics -> global merge"
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

        // Output buffer: NUM_BINS x u32 (zeroed)
        self.output_buffer = Some(alloc_buffer(
            &ctx.device,
            NUM_BINS as usize * std::mem::size_of::<u32>(),
        ));
        self.zero_output_buffer();

        // Params buffer
        let params = HistogramParams {
            element_count: size as u32,
            num_bins: NUM_BINS,
            _pad: [0; 2],
        };
        self.params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Pre-warm PSO cache
        self.pso_cache.get_or_create(ctx.library(), "histogram_256");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        // Zero output before each run
        self.zero_output_buffer();

        let input = self.input_buffer.as_ref().expect("setup not called");
        let output = self.output_buffer.as_ref().expect("setup not called");
        let params = self.params_buffer.as_ref().expect("setup not called");

        let pso = self.pso_cache.get_or_create(ctx.library(), "histogram_256");

        let timer = BenchTimer::start();

        // Create command buffer and encoder
        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        // Dispatch histogram kernel
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

        // Read back histogram
        self.gpu_result = unsafe { read_buffer_slice::<u32>(output.as_ref(), NUM_BINS as usize) };

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();

        // Sequential histogram: accumulate into Vec<u32>
        let mut hist = vec![0u32; NUM_BINS as usize];
        for &value in &self.data {
            let bin = (value % NUM_BINS) as usize;
            hist[bin] += 1;
        }
        self.cpu_result = hist;

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_result.len() != self.cpu_result.len() {
            return Err(format!(
                "Histogram size mismatch: GPU {} bins vs CPU {} bins",
                self.gpu_result.len(),
                self.cpu_result.len()
            ));
        }

        for (bin, (gpu, cpu)) in self
            .gpu_result
            .iter()
            .zip(self.cpu_result.iter())
            .enumerate()
        {
            if gpu != cpu {
                return Err(format!(
                    "Histogram bin {} mismatch: GPU={} CPU={}, diff={}",
                    bin,
                    gpu,
                    cpu,
                    (*gpu as i64 - *cpu as i64).unsigned_abs()
                ));
            }
        }

        // Verify total counts equal element count
        let gpu_total: u64 = self.gpu_result.iter().map(|&x| x as u64).sum();
        let cpu_total: u64 = self.cpu_result.iter().map(|&x| x as u64).sum();
        if gpu_total != self.size as u64 {
            return Err(format!(
                "GPU histogram total ({}) != element count ({})",
                gpu_total, self.size
            ));
        }
        if cpu_total != self.size as u64 {
            return Err(format!(
                "CPU histogram total ({}) != element count ({})",
                cpu_total, self.size
            ));
        }

        Ok(())
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let bytes = size as f64 * 4.0; // u32 = 4 bytes
        let seconds = elapsed_ms / 1000.0;

        // Elements per second
        let elements_per_sec = if seconds > 0.0 {
            size as f64 / seconds
        } else {
            0.0
        };
        m.insert("elements_per_sec".to_string(), elements_per_sec);

        // Bins per second (each element touches one bin)
        m.insert("bins_per_sec".to_string(), elements_per_sec);

        // GB/s bandwidth
        let gbs = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m.insert("bytes_processed".to_string(), bytes);
        m.insert("elements".to_string(), size as f64);
        m.insert("num_bins".to_string(), NUM_BINS as f64);

        m
    }
}
