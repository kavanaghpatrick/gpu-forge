//! GEMV experiment: GPU matrix-vector multiply vs CPU Accelerate cblas_sgemv.
//!
//! Tests gemv_f32 kernel with vectorized float4 loads for bandwidth optimization.
//! Sizes are interpreted as matrix dimension M (square MxM matrix by default).
//! y[M] = A[M,N] * x[N] where M=N=size.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer_slice, BenchTimer, GemmParams,
    MetalContext, PsoCache,
};

use crate::cpu_baselines::accelerate;
use crate::data_gen::DataGenerator;

use super::Experiment;

/// GEMV experiment using vectorized loads for bandwidth optimization.
pub struct GemvExperiment {
    /// Matrix A data (M x N, row-major).
    data_a: Vec<f32>,
    /// Input vector x (N elements).
    data_x: Vec<f32>,
    /// Metal buffer for matrix A.
    buf_a: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for input vector x.
    buf_x: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for output vector y.
    buf_y: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for GemmParams constant (reusing M, N fields).
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU result from last run.
    gpu_result: Vec<f32>,
    /// CPU result from last run.
    cpu_result: Vec<f32>,
    /// Matrix rows (M).
    dim_m: usize,
    /// Matrix cols (N).
    dim_n: usize,
}

impl GemvExperiment {
    pub fn new() -> Self {
        Self {
            data_a: Vec::new(),
            data_x: Vec::new(),
            buf_a: None,
            buf_x: None,
            buf_y: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_result: Vec::new(),
            cpu_result: Vec::new(),
            dim_m: 0,
            dim_n: 0,
        }
    }
}

impl Experiment for GemvExperiment {
    fn name(&self) -> &str {
        "gemv"
    }

    fn description(&self) -> &str {
        "GEMV (vectorized float4): y = A * x, FP32"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![256, 768, 1024, 2048, 4096]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        // Square matrix: M = N = size
        self.dim_m = size;
        self.dim_n = size;

        // Generate random FP32 matrix and vector with values in [-1, 1]
        self.data_a = gen.uniform_f32(self.dim_m * self.dim_n);
        self.data_x = gen.uniform_f32(self.dim_n);

        // Allocate Metal buffers
        self.buf_a = Some(alloc_buffer_with_data(&ctx.device, &self.data_a));
        self.buf_x = Some(alloc_buffer_with_data(&ctx.device, &self.data_x));
        self.buf_y = Some(alloc_buffer(
            &ctx.device,
            self.dim_m * std::mem::size_of::<f32>(),
        ));

        // Reuse GemmParams: M=rows, N=cols, K unused
        let params = GemmParams {
            m: self.dim_m as u32,
            n: self.dim_n as u32,
            k: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Pre-warm PSO cache
        self.pso_cache
            .get_or_create(ctx.library(), "gemv_f32");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let buf_a = self.buf_a.as_ref().expect("setup not called");
        let buf_x = self.buf_x.as_ref().expect("setup not called");
        let buf_y = self.buf_y.as_ref().expect("setup not called");
        let buf_params = self.buf_params.as_ref().expect("setup not called");

        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "gemv_f32");

        let timer = BenchTimer::start();

        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        // 1D dispatch: one thread per output row
        dispatch_1d(
            &encoder,
            pso,
            &[
                (buf_a.as_ref(), 0),
                (buf_x.as_ref(), 1),
                (buf_y.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            self.dim_m,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let elapsed = timer.stop();

        // Read back result vector
        self.gpu_result =
            unsafe { read_buffer_slice::<f32>(buf_y.as_ref(), self.dim_m) };

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();

        self.cpu_result = vec![0.0f32; self.dim_m];
        accelerate::sgemv(
            self.dim_m,
            self.dim_n,
            &self.data_a,
            &self.data_x,
            &mut self.cpu_result,
        );

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

        if max_rel_err > 1e-4 {
            return Err(format!(
                "Max relative error {:.6e} at y[{}]: GPU={:.6} CPU={:.6} (threshold 1e-4)",
                max_rel_err, max_err_idx, self.gpu_result[max_err_idx],
                self.cpu_result[max_err_idx]
            ));
        }

        Ok(())
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let dim_m = self.dim_m as f64;
        let dim_n = self.dim_n as f64;

        // Memory traffic: read A (M*N*4) + read x (N*4) + write y (M*4) bytes
        let bytes = (dim_m * dim_n + dim_n + dim_m) * 4.0;
        let seconds = elapsed_ms / 1000.0;
        let gbs = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };

        m.insert("gb_per_sec".to_string(), gbs);
        m.insert("bytes_processed".to_string(), bytes);

        // FLOPS: 2*M*N (one multiply + one add per element)
        let flops = 2.0 * dim_m * dim_n;
        let gflops = if seconds > 0.0 {
            flops / seconds / 1e9
        } else {
            0.0
        };

        m.insert("gflops".to_string(), gflops);
        m.insert("matrix_rows".to_string(), dim_m);
        m.insert("matrix_cols".to_string(), dim_n);
        m.insert("matrix_dim".to_string(), size as f64);

        m
    }
}
