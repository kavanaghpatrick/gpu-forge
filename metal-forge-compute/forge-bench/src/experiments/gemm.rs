//! GEMM experiment: GPU tiled matrix multiply vs CPU Accelerate cblas_sgemm.
//!
//! Tests gemm_naive_f32 kernel with shared-memory tiling (16x16 tiles).
//! Sizes are interpreted as matrix dimension (e.g., 256 = 256x256 square matrices).
//! C[M,N] = A[M,K] * B[K,N] where M=N=K=size.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_2d, read_buffer_slice, BenchTimer, GemmParams,
    MetalContext, PsoCache,
};

use crate::cpu_baselines::accelerate;
use crate::data_gen::DataGenerator;

use super::Experiment;

/// Tile size must match TILE_SIZE in gemm.metal.
const TILE_SIZE: usize = 16;

/// GEMM experiment using tiled shared-memory approach.
pub struct GemmExperiment {
    /// Matrix A data (M x K, row-major).
    data_a: Vec<f32>,
    /// Matrix B data (K x N, row-major).
    data_b: Vec<f32>,
    /// Metal buffer for matrix A.
    buf_a: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for matrix B.
    buf_b: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for output matrix C.
    buf_c: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for GemmParams constant.
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU result from last run.
    gpu_result: Vec<f32>,
    /// CPU result from last run.
    cpu_result: Vec<f32>,
    /// Matrix dimension M.
    dim_m: usize,
    /// Matrix dimension N.
    dim_n: usize,
    /// Matrix dimension K.
    dim_k: usize,
}

impl GemmExperiment {
    pub fn new() -> Self {
        Self {
            data_a: Vec::new(),
            data_b: Vec::new(),
            buf_a: None,
            buf_b: None,
            buf_c: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_result: Vec::new(),
            cpu_result: Vec::new(),
            dim_m: 0,
            dim_n: 0,
            dim_k: 0,
        }
    }
}

impl Experiment for GemmExperiment {
    fn name(&self) -> &str {
        "gemm"
    }

    fn description(&self) -> &str {
        "Tiled GEMM (16x16 shared memory): C = A * B, FP32"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![256, 512, 1024, 2048, 4096]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        // Square matrices: M = N = K = size
        self.dim_m = size;
        self.dim_n = size;
        self.dim_k = size;

        // Generate random FP32 matrices with values in [-1, 1]
        self.data_a = gen.uniform_f32(self.dim_m * self.dim_k);
        self.data_b = gen.uniform_f32(self.dim_k * self.dim_n);

        // Allocate Metal buffers
        self.buf_a = Some(alloc_buffer_with_data(&ctx.device, &self.data_a));
        self.buf_b = Some(alloc_buffer_with_data(&ctx.device, &self.data_b));
        self.buf_c = Some(alloc_buffer(
            &ctx.device,
            self.dim_m * self.dim_n * std::mem::size_of::<f32>(),
        ));

        let params = GemmParams {
            m: self.dim_m as u32,
            n: self.dim_n as u32,
            k: self.dim_k as u32,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Pre-warm PSO cache
        self.pso_cache
            .get_or_create(ctx.library(), "gemm_naive_f32");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let buf_a = self.buf_a.as_ref().expect("setup not called");
        let buf_b = self.buf_b.as_ref().expect("setup not called");
        let buf_c = self.buf_c.as_ref().expect("setup not called");
        let buf_params = self.buf_params.as_ref().expect("setup not called");

        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "gemm_naive_f32");

        let timer = BenchTimer::start();

        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        // Dispatch 2D: threadgroups = (ceil(N/16), ceil(M/16)), threads per TG = (16, 16)
        let tg_x = self.dim_n.div_ceil(TILE_SIZE);
        let tg_y = self.dim_m.div_ceil(TILE_SIZE);

        dispatch_2d(
            &encoder,
            pso,
            &[
                (buf_a.as_ref(), 0),
                (buf_b.as_ref(), 1),
                (buf_c.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            tg_x,
            tg_y,
            TILE_SIZE,
            TILE_SIZE,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let elapsed = timer.stop();

        // Read back result matrix
        self.gpu_result =
            unsafe { read_buffer_slice::<f32>(buf_c.as_ref(), self.dim_m * self.dim_n) };

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();

        self.cpu_result = vec![0.0f32; self.dim_m * self.dim_n];
        accelerate::sgemm(
            self.dim_m,
            self.dim_n,
            self.dim_k,
            &self.data_a,
            &self.data_b,
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

        if max_rel_err > 1e-3 {
            let row = max_err_idx / self.dim_n;
            let col = max_err_idx % self.dim_n;
            return Err(format!(
                "Max relative error {:.6e} at C[{},{}]: GPU={:.6} CPU={:.6} (threshold 1e-3)",
                max_rel_err, row, col, self.gpu_result[max_err_idx], self.cpu_result[max_err_idx]
            ));
        }

        Ok(())
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let dim = size as f64;

        // GEMM FLOPS = 2 * M * N * K (multiply-add per output element)
        let flops = 2.0 * dim * dim * dim;
        let seconds = elapsed_ms / 1000.0;
        let gflops = if seconds > 0.0 {
            flops / seconds / 1e9
        } else {
            0.0
        };

        m.insert("gflops".to_string(), gflops);
        m.insert("flops".to_string(), flops);
        m.insert("matrix_dim".to_string(), dim);

        // Memory traffic: read A (M*K) + read B (K*N) + write C (M*N), all f32
        let bytes = (dim * dim + dim * dim + dim * dim) * 4.0;
        let gbs = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);
        m.insert("bytes_processed".to_string(), bytes);

        m
    }
}
