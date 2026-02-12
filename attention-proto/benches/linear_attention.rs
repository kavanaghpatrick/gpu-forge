//! Criterion benchmark for Proto 6: FLA Linear Attention.
//!
//! Compares linear attention (chunk_h + CPU prefix sum + chunk_o) vs
//! Proto 1 softmax flash attention at multiple sequence lengths.
//! Reports GPU time, total wall-clock time, and identifies the crossover
//! point where linear attention becomes faster.
//!
//! Key insight: Linear attention is O(N * D^2) per chunk vs O(N^2 * D) for softmax.
//! Theoretical crossover at seq_len ~ D/2, but CPU prefix sum overhead raises it.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::{Duration, Instant};

use attention_proto::device::GpuDevice;
use attention_proto::encode::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
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
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Run one full linear attention pass (chunk_h GPU + CPU prefix sum + chunk_o GPU).
/// Returns (total_wall_secs, chunk_h_gpu_secs, chunk_o_gpu_secs).
fn dispatch_linear_attention(
    gpu: &GpuDevice,
    chunk_h_pso: &ProtocolObject<dyn MTLComputePipelineState>,
    chunk_o_pso: &ProtocolObject<dyn MTLComputePipelineState>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    h_deltas_buf: &ProtocolObject<dyn MTLBuffer>,
    o_buf: &ProtocolObject<dyn MTLBuffer>,
    params_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    head_dim: usize,
    chunk_size: usize,
    threads_per_tg_h: usize,
    threads_per_tg_o: usize,
) -> (f64, f64, f64) {
    let num_chunks = seq_len / chunk_size;
    let dd = head_dim * head_dim;
    let wall_start = Instant::now();

    // Pass 1: chunk_h kernel
    let chunk_h_gpu_time;
    {
        let cmd_buf = gpu
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer for chunk_h");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder for chunk_h");

        encoder.setComputePipelineState(chunk_h_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(h_deltas_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(params_buf), 0, 3);
        }

        let threadgroups = MTLSize {
            width: num_chunks,
            height: 1,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: threads_per_tg_h,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, tg_size);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        chunk_h_gpu_time = cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime();
    }

    // CPU prefix sum: H_cumulative[c] = sum(delta_H[0..=c])
    let h_deltas_size = num_chunks * dd;
    let h_deltas: Vec<f32> = unsafe { read_buffer_slice(h_deltas_buf, h_deltas_size) };

    let mut h_cumulative = vec![0.0f32; num_chunks * dd];
    h_cumulative[..dd].copy_from_slice(&h_deltas[..dd]);
    for c in 1..num_chunks {
        for elem in 0..dd {
            h_cumulative[c * dd + elem] =
                h_cumulative[(c - 1) * dd + elem] + h_deltas[c * dd + elem];
        }
    }

    // Upload H_cumulative to GPU
    let h_cumulative_buf = alloc_buffer_with_data(&gpu.device, &h_cumulative);

    // Pass 2: chunk_o kernel
    let chunk_o_gpu_time;
    {
        let cmd_buf = gpu
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer for chunk_o");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder for chunk_o");

        encoder.setComputePipelineState(chunk_o_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*h_cumulative_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(o_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(params_buf), 0, 3);
        }

        let threadgroups = MTLSize {
            width: num_chunks,
            height: 1,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: threads_per_tg_o,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, tg_size);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        chunk_o_gpu_time = cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime();
    }

    let wall_elapsed = wall_start.elapsed().as_secs_f64();
    (wall_elapsed, chunk_h_gpu_time, chunk_o_gpu_time)
}

/// Run one GPU dispatch of flash attention and return GPU time in seconds.
fn dispatch_flash(
    gpu: &GpuDevice,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    v_buf: &ProtocolObject<dyn MTLBuffer>,
    o_buf: &ProtocolObject<dyn MTLBuffer>,
    params_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    block_r: u32,
) -> f64 {
    let num_row_blocks = (seq_len as u64 + block_r as u64 - 1) / block_r as u64;
    let threadgroups = MTLSize {
        width: num_row_blocks as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: 32,
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

    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    encoder.endEncoding();

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()
}

/// Compute mean and coefficient of variation from a sample of times.
fn stats(times: &[f64]) -> (f64, f64) {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / n;
    let cv = if mean > 0.0 {
        variance.sqrt() / mean * 100.0
    } else {
        0.0
    };
    (mean, cv)
}

fn bench_linear_attention(c: &mut Criterion) {
    let gpu = GpuDevice::shared();

    let head_dim: usize = 64;
    let chunk_size: usize = 32;
    let flash_block_r: u32 = 16;
    let flash_block_c: u32 = 64;

    // Compile linear attention PSOs
    let chunk_h_key = PsoKey::simple("chunk_h")
        .with_uint(0, head_dim as u32)
        .with_uint(4, chunk_size as u32);
    let mut chunk_h_cache = PsoCache::new(gpu.library.clone());
    let chunk_h_pso = chunk_h_cache.get_or_compile(&chunk_h_key);
    let max_threads_h = chunk_h_pso.maxTotalThreadsPerThreadgroup();
    let threads_per_tg_h = (head_dim * head_dim).min(max_threads_h);

    let chunk_o_key = PsoKey::simple("chunk_o")
        .with_uint(0, head_dim as u32)
        .with_uint(4, chunk_size as u32);
    let mut chunk_o_cache = PsoCache::new(gpu.library.clone());
    let chunk_o_pso = chunk_o_cache.get_or_compile(&chunk_o_key);
    let max_threads_o = chunk_o_pso.maxTotalThreadsPerThreadgroup();
    let threads_per_tg_o = (chunk_size * head_dim).min(max_threads_o);

    // Compile flash attention PSO (baseline comparison)
    let flash_key = PsoKey::simple("flash_attention")
        .with_uint(0, head_dim as u32)
        .with_uint(1, flash_block_r)
        .with_uint(2, flash_block_c);
    let mut flash_cache = PsoCache::new(gpu.library.clone());
    let flash_pso = flash_cache.get_or_compile(&flash_key);

    let mut group = c.benchmark_group("linear_attention");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(3));

    // Results for summary table
    let mut summary: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::new();
    // (seq_len, linear_wall_us, linear_gpu_us, flash_gpu_us, linear_tflops, flash_tflops)

    for &seq_len in &[256usize, 512, 1024] {
        let n_elements = seq_len * head_dim;
        let num_chunks = seq_len / chunk_size;
        let dd = head_dim * head_dim;

        // Generate data
        let q_data = lcg_data(n_elements, 42);
        let k_data = lcg_data(n_elements, 137);
        let v_data = lcg_data(n_elements, 256);

        // Allocate linear attention buffers
        let q_buf = alloc_buffer_with_data(&gpu.device, &q_data);
        let k_buf = alloc_buffer_with_data(&gpu.device, &k_data);
        let v_buf = alloc_buffer_with_data(&gpu.device, &v_data);
        let h_deltas_buf = alloc_buffer(
            &gpu.device,
            num_chunks * dd * std::mem::size_of::<f32>(),
        );
        let o_linear_buf = alloc_buffer(&gpu.device, n_elements * std::mem::size_of::<f32>());
        let linear_params = AttentionParams {
            seq_len: seq_len as u32,
            head_dim: head_dim as u32,
            ..Default::default()
        };
        let linear_params_buf =
            alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&linear_params));

        // Allocate flash attention buffers
        let o_flash_buf = alloc_buffer(&gpu.device, n_elements * std::mem::size_of::<f32>());
        let flash_params = AttentionParams::flash(seq_len as u32, head_dim as u32, 1);
        let flash_params_buf =
            alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&flash_params));

        // Warmup: 4 dispatches of each
        for _ in 0..4 {
            dispatch_linear_attention(
                gpu,
                chunk_h_pso,
                chunk_o_pso,
                &k_buf,
                &v_buf,
                &q_buf,
                &h_deltas_buf,
                &o_linear_buf,
                &linear_params_buf,
                seq_len,
                head_dim,
                chunk_size,
                threads_per_tg_h,
                threads_per_tg_o,
            );
            dispatch_flash(
                gpu,
                flash_pso,
                &q_buf,
                &k_buf,
                &v_buf,
                &o_flash_buf,
                &flash_params_buf,
                seq_len,
                flash_block_r,
            );
        }

        // Benchmark: linear attention (wall-clock includes CPU prefix sum)
        group.bench_with_input(
            BenchmarkId::new(format!("linear_N={seq_len}"), seq_len),
            &seq_len,
            |b, &_seq_len| {
                b.iter_custom(|iters| {
                    let mut total = 0.0f64;
                    for _ in 0..iters {
                        let (wall, _h, _o) = dispatch_linear_attention(
                            gpu,
                            chunk_h_pso,
                            chunk_o_pso,
                            &k_buf,
                            &v_buf,
                            &q_buf,
                            &h_deltas_buf,
                            &o_linear_buf,
                            &linear_params_buf,
                            seq_len,
                            head_dim,
                            chunk_size,
                            threads_per_tg_h,
                            threads_per_tg_o,
                        );
                        total += wall;
                    }
                    Duration::from_secs_f64(total)
                });
            },
        );

        // Benchmark: flash attention (GPU time only, baseline)
        group.bench_with_input(
            BenchmarkId::new(format!("flash_N={seq_len}"), seq_len),
            &seq_len,
            |b, &_seq_len| {
                b.iter_custom(|iters| {
                    let mut total = 0.0f64;
                    for _ in 0..iters {
                        total += dispatch_flash(
                            gpu,
                            flash_pso,
                            &q_buf,
                            &k_buf,
                            &v_buf,
                            &o_flash_buf,
                            &flash_params_buf,
                            seq_len,
                            flash_block_r,
                        );
                    }
                    Duration::from_secs_f64(total)
                });
            },
        );

        // Post-benchmark: collect 10 samples for summary stats
        let mut linear_wall_times = Vec::with_capacity(10);
        let mut linear_gpu_times = Vec::with_capacity(10);
        let mut flash_times = Vec::with_capacity(10);
        for _ in 0..10 {
            let (wall, h_gpu, o_gpu) = dispatch_linear_attention(
                gpu,
                chunk_h_pso,
                chunk_o_pso,
                &k_buf,
                &v_buf,
                &q_buf,
                &h_deltas_buf,
                &o_linear_buf,
                &linear_params_buf,
                seq_len,
                head_dim,
                chunk_size,
                threads_per_tg_h,
                threads_per_tg_o,
            );
            linear_wall_times.push(wall);
            linear_gpu_times.push(h_gpu + o_gpu);
            flash_times.push(dispatch_flash(
                gpu,
                flash_pso,
                &q_buf,
                &k_buf,
                &v_buf,
                &o_flash_buf,
                &flash_params_buf,
                seq_len,
                flash_block_r,
            ));
        }

        let (linear_wall_mean, linear_wall_cv) = stats(&linear_wall_times);
        let (linear_gpu_mean, linear_gpu_cv) = stats(&linear_gpu_times);
        let (flash_mean, flash_cv) = stats(&flash_times);

        // FLOPS calculations:
        // Linear: 2 * N * D^2 (chunk_h: K^T*V outer products) + 2 * N * D^2 (chunk_o: Q*H matmul) = 4 * N * D^2
        let linear_flops = 4.0 * seq_len as f64 * (head_dim as f64).powi(2);
        // Flash: 4 * N^2 * D (two matmuls QK^T and PV, each 2*N*N*D)
        let flash_flops = 4.0 * (seq_len as f64).powi(2) * head_dim as f64;

        let linear_tflops = linear_flops / linear_wall_mean / 1e12;
        let flash_tflops = flash_flops / flash_mean / 1e12;

        summary.push((
            seq_len,
            linear_wall_mean * 1e6,
            linear_gpu_mean * 1e6,
            flash_mean * 1e6,
            linear_tflops,
            flash_tflops,
        ));

        let cpu_overhead_us = (linear_wall_mean - linear_gpu_mean) * 1e6;
        let ratio = if flash_mean > 0.0 {
            linear_wall_mean / flash_mean
        } else {
            0.0
        };

        eprintln!(
            "[Linear vs Flash] N={seq_len} D={head_dim} chunk={chunk_size}:"
        );
        eprintln!(
            "  Linear (wall): {:.1}us (CV={:.1}%) — includes CPU prefix sum",
            linear_wall_mean * 1e6,
            linear_wall_cv
        );
        eprintln!(
            "  Linear (GPU):  {:.1}us (CV={:.1}%) — chunk_h + chunk_o only",
            linear_gpu_mean * 1e6,
            linear_gpu_cv
        );
        eprintln!(
            "  CPU overhead:  {:.1}us (prefix sum + buffer readback/upload)",
            cpu_overhead_us
        );
        eprintln!(
            "  Flash (GPU):   {:.1}us (CV={:.1}%)",
            flash_mean * 1e6,
            flash_cv
        );
        eprintln!("  Ratio (linear_wall / flash_gpu): {:.2}x", ratio);
        eprintln!(
            "  Linear TFLOPS: {:.4} (FLOPs={:.1}M)",
            linear_tflops,
            linear_flops / 1e6
        );
        eprintln!(
            "  Flash  TFLOPS: {:.4} (FLOPs={:.1}M)",
            flash_tflops,
            flash_flops / 1e6
        );
    }

    group.finish();

    // Print summary table
    eprintln!();
    eprintln!("=== Linear Attention vs Flash Attention Summary ===");
    eprintln!("Config: D={head_dim}, chunk_size={chunk_size}, Br={flash_block_r}, Bc={flash_block_c}");
    eprintln!(
        "{:<10} {:>14} {:>14} {:>14} {:>12} {:>12}",
        "seq_len", "linear_wall", "linear_gpu", "flash_gpu", "lin_TFLOPS", "fla_TFLOPS"
    );
    for &(seq_len, lin_wall, lin_gpu, flash, lin_tf, fla_tf) in &summary {
        eprintln!(
            "{:<10} {:>12.1}us {:>12.1}us {:>12.1}us {:>12.4} {:>12.4}",
            seq_len, lin_wall, lin_gpu, flash, lin_tf, fla_tf
        );
    }
    eprintln!();

    // Crossover analysis
    eprintln!("=== Crossover Analysis ===");
    eprintln!(
        "Theoretical crossover: seq_len ~ D/2 = {} (where linear FLOPs = flash FLOPs)",
        head_dim / 2
    );
    eprintln!("  Linear FLOPs = 4 * N * D^2 = 4 * N * {}", head_dim * head_dim);
    eprintln!("  Flash FLOPs  = 4 * N^2 * D = 4 * N^2 * {head_dim}");
    eprintln!("  Equal when N = D = {head_dim}");
    eprintln!();
    eprintln!("Practical crossover is higher due to:");
    eprintln!("  1. CPU prefix sum overhead (buffer readback + prefix sum + re-upload)");
    eprintln!("  2. Two separate command buffer submissions (vs one for flash)");
    eprintln!("  3. Linear attention kernel uses scalar dots (no simdgroup_matrix)");
    eprintln!();

    // Identify crossover from measured data
    let mut crossover_found = false;
    for &(seq_len, lin_wall, _lin_gpu, flash, _lin_tf, _fla_tf) in &summary {
        if lin_wall < flash {
            eprintln!(
                "CROSSOVER: Linear faster at N={seq_len} ({:.1}us vs {:.1}us)",
                lin_wall, flash
            );
            crossover_found = true;
            break;
        }
    }
    if !crossover_found {
        eprintln!(
            "No crossover found in tested range (N=256..1024). Linear attention is slower"
        );
        eprintln!(
            "at all tested seq_lens. Expected: with GPU prefix sum kernel, crossover"
        );
        eprintln!("would occur at much larger N where O(N*D^2) << O(N^2*D).");
        if let Some(last) = summary.last() {
            let ratio = last.1 / last.3;
            // Estimate crossover: linear_wall scales ~linearly, flash scales ~quadratically
            // At current N=1024, ratio = linear/flash
            // Flash at N_cross = flash_1024 * (N_cross/1024)^2
            // Linear at N_cross ~ linear_1024 * (N_cross/1024)
            // Crossover when linear_1024 * (N_cross/1024) = flash_1024 * (N_cross/1024)^2
            // => N_cross = 1024 * linear_1024 / flash_1024 = 1024 * ratio
            let estimated_crossover = (1024.0 * ratio) as usize;
            eprintln!(
                "Estimated crossover (extrapolated): N ~ {} (assuming linear scales as O(N), flash as O(N^2))",
                estimated_crossover
            );
        }
    }
    eprintln!("=============================================");
}

criterion_group!(benches, bench_linear_attention);
criterion_main!(benches);
