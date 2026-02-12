//! Criterion benchmark for Proto 7: RoPE/ALiBi/GQA variant overhead.
//!
//! Measures per-variant microseconds:
//! - Base flash attention (no variants) as baseline
//! - RoPE standalone (apply_rope kernel)
//! - ALiBi fused into flash attention (ALIBI_ENABLED function constant)
//! - GQA standalone (gqa_remap kernel) at group sizes 1, 2, 4, 8
//!
//! Config: N=2048, D=64, heads=32.
//! D=64 used because flash_attention.metal has #define TILE_D 64.
//! Reports us per variant and % overhead vs base attention time.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/// Compute mean and coefficient of variation from a sample of times (in seconds).
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

// ---------------------------------------------------------------------------
// GPU dispatch helpers
// ---------------------------------------------------------------------------

/// Dispatch flash attention (multi-head) and return GPU time in seconds.
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
    let threadgroups = MTLSize {
        width: num_row_blocks as usize,
        height: num_heads as usize,
        depth: 1,
    };
    let tg_size = MTLSize {
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

    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, tg_size);
    encoder.endEncoding();

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()
}

/// Dispatch RoPE kernel and return GPU time in seconds.
fn gpu_dispatch_rope(
    gpu: &GpuDevice,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    k_buf: &ProtocolObject<dyn MTLBuffer>,
    params_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    head_dim: usize,
) -> f64 {
    let grid = MTLSize {
        width: seq_len,
        height: head_dim / 2,
        depth: 1,
    };
    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    let tg_side = if max_threads >= 256 {
        16
    } else {
        (max_threads as f64).sqrt() as usize
    };
    let tg_size = MTLSize {
        width: tg_side,
        height: tg_side,
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
        encoder.setBuffer_offset_atIndex(Some(params_buf), 0, 2);
    }

    encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
    encoder.endEncoding();

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()
}

/// Dispatch GQA remap kernel and return GPU time in seconds.
fn gpu_dispatch_gqa(
    gpu: &GpuDevice,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    k_full_buf: &ProtocolObject<dyn MTLBuffer>,
    k_expanded_buf: &ProtocolObject<dyn MTLBuffer>,
    params_buf: &ProtocolObject<dyn MTLBuffer>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> f64 {
    let grid = MTLSize {
        width: num_heads,
        height: seq_len,
        depth: head_dim,
    };
    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    let (tg_w, tg_h, tg_d) = if max_threads >= 512 {
        (8, 8, 8)
    } else if max_threads >= 64 {
        (4, 4, 4)
    } else {
        (2, 2, 2)
    };
    let tg_size = MTLSize {
        width: tg_w,
        height: tg_h,
        depth: tg_d,
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
        encoder.setBuffer_offset_atIndex(Some(k_full_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k_expanded_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(params_buf), 0, 2);
    }

    encoder.dispatchThreads_threadsPerThreadgroup(grid, tg_size);
    encoder.endEncoding();

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

fn bench_variant_overhead(c: &mut Criterion) {
    let gpu = GpuDevice::shared();

    // Config: N=2048, D=64 (flash kernel TILE_D limit), heads=32
    let seq_len: usize = 2048;
    let head_dim: usize = 64;
    let num_heads: u32 = 32;
    let block_r: u32 = 16;
    let block_c: u32 = 64;

    // Total elements per multi-head tensor
    let total = num_heads as usize * seq_len * head_dim;

    // -----------------------------------------------------------------------
    // Compile all PSOs upfront (separate PsoCache instances to avoid borrows)
    // -----------------------------------------------------------------------

    // Base flash attention (ALIBI_ENABLED=false)
    let base_key = PsoKey::simple("flash_attention")
        .with_uint(0, head_dim as u32)
        .with_uint(1, block_r)
        .with_uint(2, block_c)
        .with_bool(4, false);
    let mut base_cache = PsoCache::new(gpu.library.clone());
    let base_pso = base_cache.get_or_compile(&base_key);

    // Flash attention with ALiBi enabled
    let alibi_key = PsoKey::simple("flash_attention")
        .with_uint(0, head_dim as u32)
        .with_uint(1, block_r)
        .with_uint(2, block_c)
        .with_bool(4, true);
    let mut alibi_cache = PsoCache::new(gpu.library.clone());
    let alibi_pso = alibi_cache.get_or_compile(&alibi_key);

    // RoPE kernel (no function constants)
    let rope_key = PsoKey::simple("apply_rope");
    let mut rope_cache = PsoCache::new(gpu.library.clone());
    let rope_pso = rope_cache.get_or_compile(&rope_key);

    // GQA remap kernel (no function constants)
    let gqa_key = PsoKey::simple("gqa_remap");
    let mut gqa_cache = PsoCache::new(gpu.library.clone());
    let gqa_pso = gqa_cache.get_or_compile(&gqa_key);

    // -----------------------------------------------------------------------
    // Allocate shared data buffers
    // -----------------------------------------------------------------------

    // Multi-head Q/K/V for flash attention: [num_heads, seq_len, head_dim]
    let q_data = lcg_data(total, 42);
    let k_data = lcg_data(total, 137);
    let v_data = lcg_data(total, 256);

    let q_buf = alloc_buffer_with_data(&gpu.device, &q_data);
    let k_buf = alloc_buffer_with_data(&gpu.device, &k_data);
    let v_buf = alloc_buffer_with_data(&gpu.device, &v_data);
    let o_buf = alloc_buffer(&gpu.device, total * std::mem::size_of::<f32>());

    // Flash params (multi-head)
    let flash_params = AttentionParams::flash(seq_len as u32, head_dim as u32, num_heads);
    let flash_params_buf = alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&flash_params));

    // RoPE buffers: single-head [seq_len, head_dim]
    let rope_elems = seq_len * head_dim;
    let rope_q_data = lcg_data(rope_elems, 77);
    let rope_k_data = lcg_data(rope_elems, 88);
    let rope_q_buf = alloc_buffer_with_data(&gpu.device, &rope_q_data);
    let rope_k_buf = alloc_buffer_with_data(&gpu.device, &rope_k_data);
    let rope_params = AttentionParams {
        seq_len: seq_len as u32,
        head_dim: head_dim as u32,
        num_heads: 1,
        ..Default::default()
    };
    let rope_params_buf = alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&rope_params));

    // -----------------------------------------------------------------------
    // Warmup: 8 dispatches of base, ALiBi, and RoPE
    // -----------------------------------------------------------------------
    for _ in 0..8 {
        gpu_dispatch_flash(
            gpu, base_pso, &q_buf, &k_buf, &v_buf, &o_buf, &flash_params_buf,
            seq_len, block_r, num_heads,
        );
        gpu_dispatch_flash(
            gpu, alibi_pso, &q_buf, &k_buf, &v_buf, &o_buf, &flash_params_buf,
            seq_len, block_r, num_heads,
        );
        gpu_dispatch_rope(
            gpu, rope_pso, &rope_q_buf, &rope_k_buf, &rope_params_buf,
            seq_len, head_dim,
        );
    }

    // -----------------------------------------------------------------------
    // Criterion benchmark group
    // -----------------------------------------------------------------------
    let mut group = c.benchmark_group("variant_overhead");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));

    // 1. Base flash attention (no variants)
    group.bench_function("base_flash_N=2048_D=64_H=32", |b| {
        b.iter_custom(|iters| {
            let mut total_gpu = 0.0f64;
            for _ in 0..iters {
                total_gpu += gpu_dispatch_flash(
                    gpu, base_pso, &q_buf, &k_buf, &v_buf, &o_buf, &flash_params_buf,
                    seq_len, block_r, num_heads,
                );
            }
            Duration::from_secs_f64(total_gpu)
        });
    });

    // 2. RoPE standalone (single-head, pre-attention transform)
    group.bench_function("rope_standalone_N=2048_D=64", |b| {
        b.iter_custom(|iters| {
            let mut total_gpu = 0.0f64;
            for _ in 0..iters {
                total_gpu += gpu_dispatch_rope(
                    gpu, rope_pso, &rope_q_buf, &rope_k_buf, &rope_params_buf,
                    seq_len, head_dim,
                );
            }
            Duration::from_secs_f64(total_gpu)
        });
    });

    // 3. ALiBi fused into flash attention
    group.bench_function("alibi_fused_N=2048_D=64_H=32", |b| {
        b.iter_custom(|iters| {
            let mut total_gpu = 0.0f64;
            for _ in 0..iters {
                total_gpu += gpu_dispatch_flash(
                    gpu, alibi_pso, &q_buf, &k_buf, &v_buf, &o_buf, &flash_params_buf,
                    seq_len, block_r, num_heads,
                );
            }
            Duration::from_secs_f64(total_gpu)
        });
    });

    // 4. GQA remap at different group sizes: 1 (MHA=copy), 2, 4, 8
    for &group_size in &[1u32, 2, 4, 8] {
        let num_kv_heads = num_heads / group_size;
        let kv_elems = num_kv_heads as usize * seq_len * head_dim;
        let expanded_elems = num_heads as usize * seq_len * head_dim;

        let gqa_k_data = lcg_data(kv_elems, 300 + group_size as u64);
        let gqa_k_full_buf = alloc_buffer_with_data(&gpu.device, &gqa_k_data);
        let gqa_k_expanded_buf = alloc_buffer(
            &gpu.device,
            expanded_elems * std::mem::size_of::<f32>(),
        );
        let gqa_params = AttentionParams::gqa(
            seq_len as u32,
            head_dim as u32,
            num_heads,
            num_kv_heads,
        );
        let gqa_params_buf =
            alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&gqa_params));

        // Warmup GQA
        for _ in 0..8 {
            gpu_dispatch_gqa(
                gpu, gqa_pso, &gqa_k_full_buf, &gqa_k_expanded_buf, &gqa_params_buf,
                num_heads as usize, seq_len, head_dim,
            );
        }

        group.bench_with_input(
            BenchmarkId::new(format!("gqa_remap_gs={group_size}"), group_size),
            &group_size,
            |b, &_gs| {
                b.iter_custom(|iters| {
                    let mut total_gpu = 0.0f64;
                    for _ in 0..iters {
                        total_gpu += gpu_dispatch_gqa(
                            gpu, gqa_pso, &gqa_k_full_buf, &gqa_k_expanded_buf,
                            &gqa_params_buf, num_heads as usize, seq_len, head_dim,
                        );
                    }
                    Duration::from_secs_f64(total_gpu)
                });
            },
        );
    }

    group.finish();

    // -----------------------------------------------------------------------
    // Post-benchmark: collect 10 samples for summary statistics
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("=== Variant Overhead Summary ===");
    eprintln!("Config: N={seq_len}, D={head_dim}, heads={num_heads}");
    eprintln!();

    // Base flash attention timing
    let mut base_times = Vec::with_capacity(10);
    for _ in 0..10 {
        base_times.push(gpu_dispatch_flash(
            gpu, base_pso, &q_buf, &k_buf, &v_buf, &o_buf, &flash_params_buf,
            seq_len, block_r, num_heads,
        ));
    }
    let (base_mean, base_cv) = stats(&base_times);
    let base_us = base_mean * 1e6;
    eprintln!(
        "  Base flash attention:     {:>8.1}us (CV={:.1}%)",
        base_us, base_cv
    );

    // RoPE standalone timing
    let mut rope_times = Vec::with_capacity(10);
    for _ in 0..10 {
        rope_times.push(gpu_dispatch_rope(
            gpu, rope_pso, &rope_q_buf, &rope_k_buf, &rope_params_buf,
            seq_len, head_dim,
        ));
    }
    let (rope_mean, rope_cv) = stats(&rope_times);
    let rope_us = rope_mean * 1e6;
    let rope_pct = if base_mean > 0.0 {
        rope_mean / base_mean * 100.0
    } else {
        0.0
    };
    eprintln!(
        "  RoPE standalone:          {:>8.1}us (CV={:.1}%)  = {:.1}% of base",
        rope_us, rope_cv, rope_pct
    );

    // ALiBi fused timing
    let mut alibi_times = Vec::with_capacity(10);
    for _ in 0..10 {
        alibi_times.push(gpu_dispatch_flash(
            gpu, alibi_pso, &q_buf, &k_buf, &v_buf, &o_buf, &flash_params_buf,
            seq_len, block_r, num_heads,
        ));
    }
    let (alibi_mean, alibi_cv) = stats(&alibi_times);
    let alibi_us = alibi_mean * 1e6;
    let alibi_overhead_us = (alibi_mean - base_mean) * 1e6;
    let alibi_pct = if base_mean > 0.0 {
        (alibi_mean - base_mean) / base_mean * 100.0
    } else {
        0.0
    };
    eprintln!(
        "  ALiBi fused:              {:>8.1}us (CV={:.1}%)  overhead: {:>+.1}us ({:>+.1}%)",
        alibi_us, alibi_cv, alibi_overhead_us, alibi_pct
    );

    // GQA remap at different group sizes
    eprintln!();
    eprintln!("  GQA remap (standalone copy kernel):");
    eprintln!(
        "  {:>10} {:>10} {:>10} {:>12} {:>12}",
        "group_size", "kv_heads", "us", "% of base", "CV%"
    );

    for &group_size in &[1u32, 2, 4, 8] {
        let num_kv_heads = num_heads / group_size;
        let kv_elems = num_kv_heads as usize * seq_len * head_dim;
        let expanded_elems = num_heads as usize * seq_len * head_dim;

        let gqa_k_data = lcg_data(kv_elems, 300 + group_size as u64);
        let gqa_k_full_buf = alloc_buffer_with_data(&gpu.device, &gqa_k_data);
        let gqa_k_expanded_buf = alloc_buffer(
            &gpu.device,
            expanded_elems * std::mem::size_of::<f32>(),
        );
        let gqa_params = AttentionParams::gqa(
            seq_len as u32,
            head_dim as u32,
            num_heads,
            num_kv_heads,
        );
        let gqa_params_buf =
            alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&gqa_params));

        let mut gqa_times = Vec::with_capacity(10);
        for _ in 0..10 {
            gqa_times.push(gpu_dispatch_gqa(
                gpu, gqa_pso, &gqa_k_full_buf, &gqa_k_expanded_buf, &gqa_params_buf,
                num_heads as usize, seq_len, head_dim,
            ));
        }
        let (gqa_mean, gqa_cv) = stats(&gqa_times);
        let gqa_us = gqa_mean * 1e6;
        let gqa_pct = if base_mean > 0.0 {
            gqa_mean / base_mean * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {:>10} {:>10} {:>8.1}us {:>10.1}% {:>10.1}%",
            group_size, num_kv_heads, gqa_us, gqa_pct, gqa_cv
        );
    }

    // Summary
    eprintln!();
    eprintln!("=== Per-Variant Cost Summary ===");
    eprintln!("  Base attention time: {:.1}us", base_us);
    eprintln!(
        "  RoPE (pre-attention):     {:.1}us = {:.1}% of attention",
        rope_us, rope_pct
    );
    eprintln!(
        "  ALiBi (fused overhead):   {:+.1}us = {:+.1}% of attention",
        alibi_overhead_us, alibi_pct
    );
    eprintln!("  GQA remap (pre-attention): see group-size table above");
    eprintln!("===================================");
}

criterion_group!(benches, bench_variant_overhead);
criterion_main!(benches);
