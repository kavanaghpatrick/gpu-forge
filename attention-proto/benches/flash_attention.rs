//! Criterion benchmark for Proto 1 Flash Attention kernel.
//!
//! Sweeps seq_len with fixed tile config (BLOCK_R=16, BLOCK_C=64, D=64)
//! and reports TFLOPS. Uses iter_custom with GPU timing for accurate
//! hardware-side measurements.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

use attention_proto::device::GpuDevice;
use attention_proto::encode::{alloc_buffer, alloc_buffer_with_data};
use attention_proto::pipeline::{PsoCache, PsoKey};
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
            // LCG: state = (state * 6364136223846793005 + 1442695040888963407) mod 2^64
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [-1, 1] range
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Run one GPU dispatch of flash attention and return GPU time in seconds.
fn gpu_dispatch_flash(
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
        width: 32, // One simdgroup
        height: 1,
        depth: 1,
    };

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

    let gpu_start = cmd_buf.GPUStartTime();
    let gpu_end = cmd_buf.GPUEndTime();
    gpu_end - gpu_start
}

fn bench_flash_attention(c: &mut Criterion) {
    let gpu = GpuDevice::shared();

    let head_dim: usize = 64;
    let block_r: u32 = 16;
    let block_c: u32 = 64;
    let num_heads: u32 = 1;

    // Compile PSO once (function constants: HEAD_DIM=64, BLOCK_R=16, BLOCK_C=64)
    let pso_key = PsoKey::simple("flash_attention")
        .with_uint(0, head_dim as u32)
        .with_uint(1, block_r)
        .with_uint(2, block_c);
    let mut cache = PsoCache::new(gpu.library.clone());
    let pso = cache.get_or_compile(&pso_key);

    let mut group = c.benchmark_group("flash_attention");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(5));

    for &seq_len in &[256usize, 512, 1024, 2048] {
        let n_elements = seq_len * head_dim;

        // Compute FLOPS: 2 matmuls (QK^T and PV), each 2*N*N*D
        let flops: u64 = 4 * (seq_len as u64) * (seq_len as u64) * (head_dim as u64);

        // Set throughput for criterion's built-in reporting
        group.throughput(Throughput::Elements(flops));

        // Pre-allocate buffers
        let q_data = lcg_data(n_elements, 42);
        let k_data = lcg_data(n_elements, 137);
        let v_data = lcg_data(n_elements, 256);

        let q_buf = alloc_buffer_with_data(&gpu.device, &q_data);
        let k_buf = alloc_buffer_with_data(&gpu.device, &k_data);
        let v_buf = alloc_buffer_with_data(&gpu.device, &v_data);
        let o_buf = alloc_buffer(&gpu.device, n_elements * std::mem::size_of::<f32>());
        let params = AttentionParams::flash(seq_len as u32, head_dim as u32, num_heads);
        let params_buf = alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&params));

        // GPU warmup: 8 throwaway dispatches to stabilize clocks
        for _ in 0..8 {
            gpu_dispatch_flash(
                gpu, pso, &q_buf, &k_buf, &v_buf, &o_buf, &params_buf,
                seq_len, block_r, num_heads,
            );
        }

        group.bench_with_input(
            BenchmarkId::new(
                format!("N={seq_len}_D={head_dim}_Br={block_r}_Bc={block_c}"),
                seq_len,
            ),
            &seq_len,
            |b, &seq_len| {
                b.iter_custom(|iters| {
                    let mut total_gpu_secs = 0.0f64;
                    for _ in 0..iters {
                        let gpu_time = gpu_dispatch_flash(
                            gpu, pso, &q_buf, &k_buf, &v_buf, &o_buf, &params_buf,
                            seq_len, block_r, num_heads,
                        );
                        total_gpu_secs += gpu_time;
                    }
                    // Convert GPU seconds to Duration for criterion
                    Duration::from_secs_f64(total_gpu_secs)
                });
            },
        );

        // After benchmark, compute and print TFLOPS for this config
        // Run 10 samples for a quick TFLOPS estimate
        let mut times = Vec::with_capacity(10);
        for _ in 0..10 {
            let t = gpu_dispatch_flash(
                gpu, pso, &q_buf, &k_buf, &v_buf, &o_buf, &params_buf,
                seq_len, block_r, num_heads,
            );
            times.push(t);
        }
        let mean_time: f64 = times.iter().sum::<f64>() / times.len() as f64;
        let tflops = flops as f64 / mean_time / 1e12;

        // Compute CV (coefficient of variation)
        let variance: f64 = times
            .iter()
            .map(|&t| (t - mean_time).powi(2))
            .sum::<f64>()
            / times.len() as f64;
        let std_dev = variance.sqrt();
        let cv = if mean_time > 0.0 {
            std_dev / mean_time * 100.0
        } else {
            0.0
        };

        eprintln!(
            "[TFLOPS] N={seq_len} D={head_dim} Br={block_r} Bc={block_c}: \
             {tflops:.3} TFLOPS, mean={mean_time_us:.1}us, CV={cv:.1}%",
            mean_time_us = mean_time * 1e6,
        );
    }

    group.finish();
}

criterion_group!(benches, bench_flash_attention);
criterion_main!(benches);
