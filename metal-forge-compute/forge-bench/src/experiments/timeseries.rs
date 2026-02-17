//! Time series analytics experiment: GPU moving average vs CPU sequential sliding window.
//!
//! Tests a simple moving average over a price series with configurable window size.
//! The GPU kernel has each thread compute the average over its window by reading
//! window_size elements -- simple but memory-bandwidth-bound.
//!
//! Also supports VWAP (volume-weighted average price) with price and volume arrays.
//! Validates GPU result against CPU reference with relative error < 1e-3.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, read_buffer_slice, BenchTimer, MetalContext, PsoCache,
    TimeSeriesParams,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// Default window size for moving average.
const WINDOW_SIZE: u32 = 20;

/// Time series analytics experiment comparing GPU moving average vs CPU sequential.
pub struct TimeSeriesExperiment {
    /// Input price data.
    prices: Vec<f32>,
    /// Input volume data (for VWAP).
    volumes: Vec<f32>,
    /// Metal buffer for price data.
    buf_prices: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for volume data.
    buf_volumes: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for output (moving average result).
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for TimeSeriesParams.
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU result from last run.
    gpu_result: Vec<f32>,
    /// CPU result from last run.
    cpu_result: Vec<f32>,
    /// Current tick count.
    size: usize,
    /// Window size.
    window_size: u32,
}

impl TimeSeriesExperiment {
    pub fn new() -> Self {
        Self {
            prices: Vec::new(),
            volumes: Vec::new(),
            buf_prices: None,
            buf_volumes: None,
            buf_output: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_result: Vec::new(),
            cpu_result: Vec::new(),
            size: 0,
            window_size: WINDOW_SIZE,
        }
    }
}

impl Experiment for TimeSeriesExperiment {
    fn name(&self) -> &str {
        "timeseries"
    }

    fn description(&self) -> &str {
        "Time series moving average: GPU parallel window vs CPU sequential"
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
        let (prices, volumes) = gen.time_series(size);
        self.prices = prices;
        self.volumes = volumes;

        // Input buffers
        self.buf_prices = Some(alloc_buffer_with_data(&ctx.device, &self.prices));
        self.buf_volumes = Some(alloc_buffer_with_data(&ctx.device, &self.volumes));

        // Output buffer: N x f32
        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<f32>(),
        ));

        // Params buffer
        let params = TimeSeriesParams {
            tick_count: size as u32,
            window_size: self.window_size,
            op_type: 0, // moving average
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Pre-warm PSO cache
        self.pso_cache
            .get_or_create(ctx.library(), "timeseries_moving_avg");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let buf_prices = self.buf_prices.as_ref().expect("setup not called").clone();
        let buf_output = self.buf_output.as_ref().expect("setup not called").clone();
        let buf_params = self.buf_params.as_ref().expect("setup not called").clone();

        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "timeseries_moving_avg");

        let timer = BenchTimer::start();

        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        forge_primitives::dispatch_1d(
            &encoder,
            pso,
            &[
                (buf_prices.as_ref(), 0),
                (buf_output.as_ref(), 1),
                (buf_params.as_ref(), 2),
            ],
            self.size,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let elapsed = timer.stop();

        // Read back result
        self.gpu_result = unsafe { read_buffer_slice::<f32>(buf_output.as_ref(), self.size) };

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();

        let n = self.prices.len();
        let w = self.window_size as usize;
        let mut result = vec![0.0f32; n];

        // Sequential sliding window moving average
        let mut window_sum: f64 = 0.0;
        for i in 0..n {
            window_sum += self.prices[i] as f64;
            if i >= w {
                window_sum -= self.prices[i - w] as f64;
            }
            let count = if i >= w - 1 { w } else { i + 1 };
            result[i] = (window_sum / count as f64) as f32;
        }

        self.cpu_result = result;
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_result.len() != self.cpu_result.len() {
            return Err(format!(
                "Result size mismatch: GPU {} vs CPU {}",
                self.gpu_result.len(),
                self.cpu_result.len()
            ));
        }

        let mut max_rel_err: f64 = 0.0;
        let mut max_err_idx = 0;

        for (i, (gpu, cpu)) in self
            .gpu_result
            .iter()
            .zip(self.cpu_result.iter())
            .enumerate()
        {
            let abs_err = (*gpu as f64 - *cpu as f64).abs();
            let denom = (*cpu as f64).abs().max(1e-8);
            let rel_err = abs_err / denom;

            if rel_err > max_rel_err {
                max_rel_err = rel_err;
                max_err_idx = i;
            }
        }

        if max_rel_err > 1e-3 {
            return Err(format!(
                "Max relative error {:.6e} at index {}: GPU={:.6} CPU={:.6} (threshold 1e-3)",
                max_rel_err, max_err_idx, self.gpu_result[max_err_idx], self.cpu_result[max_err_idx]
            ));
        }

        Ok(())
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let seconds = elapsed_ms / 1000.0;

        // Ticks per second
        let ticks_per_sec = if seconds > 0.0 {
            size as f64 / seconds
        } else {
            0.0
        };
        m.insert("ticks_per_sec".to_string(), ticks_per_sec);
        m.insert("ticks".to_string(), size as f64);
        m.insert("window_size".to_string(), self.window_size as f64);

        // Bandwidth: each thread reads ~window_size floats + writes 1 float
        // Effective bytes = N * (window_size + 1) * 4
        let bytes = size as f64 * (self.window_size as f64 + 1.0) * 4.0;
        let gbs = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
