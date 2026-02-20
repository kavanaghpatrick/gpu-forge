//! Criterion benchmark: CubeCL-generated vs hand-written Metal attention kernels.
//!
//! Side-by-side wall-clock comparison at identical matrix sizes.
//! CubeCL runs Q*K^T matmul via wgpu Metal backend.
//! Hand-written runs full Flash Attention via objc2-metal dispatch.
//!
//! NOTE: This is NOT an apples-to-apples comparison.
//! - CubeCL: naive Q*K^T matmul only (score computation, ~1/4 of attention)
//! - Hand-written: full FlashAttention-2 (Q*K^T + softmax + P*V + online accumulation)
//! The hand-written kernel does ~4x more FLOPs per dispatch, yet we expect it to be
//! faster due to simdgroup_matrix, threadgroup memory tiling, and function constants.
//!
//! Both use wall-clock timing (Instant::now) for fairness since they use different
//! GPU dispatch mechanisms (wgpu vs objc2-metal).
//!
//! Feature-gated: requires `--features cubecl` to compile.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::{Duration, Instant};

use attention_proto::device::GpuDevice;
use attention_proto::encode::{alloc_buffer, alloc_buffer_with_data};
use attention_proto::pipeline::{PsoCache, PsoKey};
use attention_proto::proto5_cubecl::run_cubecl_matmul;
use attention_proto::types::AttentionParams;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Generate deterministic pseudo-random f32 data via LCG.
fn lcg_data(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Run one GPU dispatch of flash attention and wait for completion.
/// Returns wall-clock time in seconds (includes dispatch overhead).
fn wall_clock_flash(
    gpu: &GpuDevice,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    o_buf: &ProtocolObject<dyn MTLBuffer>,
    params_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    block_r: u32,
    num_heads: u32,
) -> f64 {
    let num_row_blocks = (seq_len as u64 + block_r as u64 - 1) / block_r as u64;
    let threadgroups_per_grid = MTLSize {
        width: num_row_blocks as usize,
        height: num_heads as usize,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    let start = Instant::now();

    let cmd_buf = gpu
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = cmd_buf
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(o_buf), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(params_buf), 0, 4);
    }

    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threads_per_threadgroup);
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    start.elapsed().as_secs_f64()
}

/// Run CubeCL Q*K^T matmul via wgpu and return wall-clock time in seconds.
/// Uses the public run_cubecl_matmul function which handles buffer creation,
/// kernel dispatch, and readback through the wgpu Metal backend.
fn wall_clock_cubecl(seq_len: usize, head_dim: usize, q: &[f32], k: &[f32]) -> f64 {
    let start = Instant::now();
    let _result = run_cubecl_matmul(seq_len, head_dim, q, k)
        .expect("CubeCL matmul failed");
    start.elapsed().as_secs_f64()
}

fn bench_cubecl_comparison(c: &mut Criterion) {
    let gpu = GpuDevice::shared();

    let head_dim: usize = 64;
    let block_r: u32 = 16;
    let block_c: u32 = 64;
    let num_heads: u32 = 1;

    // Compile Flash Attention PSO once
    let pso_key = PsoKey::simple("flash_attention")
        .with_uint(0, head_dim as u32)
        .with_uint(1, block_r)
        .with_uint(2, block_c);
    let mut cache = PsoCache::new(gpu.library.clone());
    let pso = cache.get_or_compile(&pso_key);

    let mut group = c.benchmark_group("cubecl_vs_handwritten");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(3));

    // Print header
    eprintln!();
    eprintln!("=== CubeCL vs Hand-Written Comparison ===");
    eprintln!("NOTE: CubeCL = naive Q*K^T matmul only (score computation)");
    eprintln!("      Hand-written = full FlashAttention-2 (Q*K^T + softmax + P*V)");
    eprintln!("      Hand-written does ~4x more FLOPs but uses simdgroup_matrix + tiling");
    eprintln!();

    for &seq_len in &[256usize, 512, 1024] {
        let n_elements = seq_len * head_dim;

        // FLOPs for Q*K^T matmul: 2 * N * N * D (multiply-add)
        let matmul_flops: u64 = 2 * (seq_len as u64) * (seq_len as u64) * (head_dim as u64);
        // FLOPs for full attention: 4 * N * N * D (two matmuls: QK^T and PV)
        let attention_flops: u64 = 4 * (seq_len as u64) * (seq_len as u64) * (head_dim as u64);

        // Pre-allocate data
        let q_data = lcg_data(n_elements, 42);
        let k_data = lcg_data(n_elements, 137);
        let v_data = lcg_data(n_elements, 256);

        // Pre-allocate Metal buffers for flash attention
        let q_buf = alloc_buffer_with_data(&gpu.device, &q_data);
        let k_buf = alloc_buffer_with_data(&gpu.device, &k_data);
        let v_buf = alloc_buffer_with_data(&gpu.device, &v_data);
        let o_buf = alloc_buffer(&gpu.device, n_elements * std::mem::size_of::<f32>());
        let params = AttentionParams::flash(seq_len as u32, head_dim as u32, num_heads);
        let params_buf = alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&params));

        // Warmup: 4 dispatches each
        for _ in 0..4 {
            wall_clock_flash(
                gpu, pso, &q_buf, &k_buf, &v_buf, &o_buf, &params_buf,
                seq_len, block_r, num_heads,
            );
        }
        for _ in 0..4 {
            wall_clock_cubecl(seq_len, head_dim, &q_data, &k_data);
        }

        // Criterion benchmark: CubeCL
        group.bench_with_input(
            BenchmarkId::new(format!("cubecl_matmul_N={seq_len}"), seq_len),
            &seq_len,
            |b, &_n| {
                b.iter_custom(|iters| {
                    let mut total = 0.0f64;
                    for _ in 0..iters {
                        total += wall_clock_cubecl(seq_len, head_dim, &q_data, &k_data);
                    }
                    Duration::from_secs_f64(total)
                });
            },
        );

        // Criterion benchmark: Hand-written Flash Attention
        group.bench_with_input(
            BenchmarkId::new(format!("handwritten_flash_N={seq_len}"), seq_len),
            &seq_len,
            |b, &_n| {
                b.iter_custom(|iters| {
                    let mut total = 0.0f64;
                    for _ in 0..iters {
                        total += wall_clock_flash(
                            gpu, pso, &q_buf, &k_buf, &v_buf, &o_buf, &params_buf,
                            seq_len, block_r, num_heads,
                        );
                    }
                    Duration::from_secs_f64(total)
                });
            },
        );

        // Quick TFLOPS estimate (10 samples each)
        let mut cubecl_times = Vec::with_capacity(10);
        let mut flash_times = Vec::with_capacity(10);

        for _ in 0..10 {
            cubecl_times.push(wall_clock_cubecl(seq_len, head_dim, &q_data, &k_data));
        }
        for _ in 0..10 {
            flash_times.push(wall_clock_flash(
                gpu, pso, &q_buf, &k_buf, &v_buf, &o_buf, &params_buf,
                seq_len, block_r, num_heads,
            ));
        }

        let cubecl_mean = cubecl_times.iter().sum::<f64>() / cubecl_times.len() as f64;
        let flash_mean = flash_times.iter().sum::<f64>() / flash_times.len() as f64;

        let cubecl_tflops = matmul_flops as f64 / cubecl_mean / 1e12;
        let flash_tflops = attention_flops as f64 / flash_mean / 1e12;

        // Ratio: compare on equal footing (both as TFLOPS for their respective workloads)
        let ratio = if flash_tflops > 0.0 {
            cubecl_tflops / flash_tflops
        } else {
            0.0
        };

        eprintln!(
            "[TFLOPS] N={seq_len} D={head_dim}:"
        );
        eprintln!(
            "  CubeCL (Q*K^T only):     {cubecl_tflops:.4} TFLOPS, mean={cubecl_us:.1}us",
            cubecl_us = cubecl_mean * 1e6,
        );
        eprintln!(
            "  Hand-written (full attn): {flash_tflops:.4} TFLOPS, mean={flash_us:.1}us",
            flash_us = flash_mean * 1e6,
        );
        eprintln!(
            "  TFLOPS ratio (CubeCL/Hand-written): {ratio:.2}x"
        );
        eprintln!(
            "  Wall-clock ratio: {wall_ratio:.2}x (CubeCL/Hand-written)",
            wall_ratio = cubecl_mean / flash_mean,
        );
        eprintln!();
    }

    group.finish();

    // Final summary
    eprintln!("=== Summary ===");
    eprintln!("CubeCL uses wgpu Metal backend (scalar kernels, no simdgroup_matrix)");
    eprintln!("Hand-written uses objc2-metal (simdgroup_matrix, threadgroup tiling, function constants)");
    eprintln!("CubeCL expected to be 30-60% slower even for simple matmul due to wgpu abstraction overhead");
    eprintln!("Hand-written does 4x more work (full attention) yet likely faster or comparable in wall-clock");
}

criterion_group!(benches, bench_cubecl_comparison);
criterion_main!(benches);
