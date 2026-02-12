//! Criterion benchmark for Proto 3 PagedAttention V2.
//!
//! Measures paged attention (partition + reduce) vs Proto 1 contiguous flash attention
//! at context lengths 256/512/1024, page_size=16, D=64.
//! Reports overhead % of page-table indirection.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

use attention_proto::device::GpuDevice;
use attention_proto::encode::{alloc_buffer, alloc_buffer_with_data};
use attention_proto::pipeline::{PsoCache, PsoKey};
use attention_proto::proto3_paged::create_paged_kv_cache;
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

/// Run one GPU dispatch of paged attention (partition + reduce) and return GPU time in seconds.
fn gpu_dispatch_paged(
    gpu: &GpuDevice,
    partition_pso: &ProtocolObject<dyn MTLComputePipelineState>,
    reduce_pso: &ProtocolObject<dyn MTLComputePipelineState>,
    q_buf: &ProtocolObject<dyn MTLBuffer>,
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    pt_buf: &ProtocolObject<dyn MTLBuffer>,
    o_partial_buf: &ProtocolObject<dyn MTLBuffer>,
    m_partial_buf: &ProtocolObject<dyn MTLBuffer>,
    l_partial_buf: &ProtocolObject<dyn MTLBuffer>,
    o_final_buf: &ProtocolObject<dyn MTLBuffer>,
    params_buf: &ProtocolObject<dyn MTLBuffer>,
    num_query_blocks: usize,
    num_partitions: usize,
    block_r: usize,
    head_dim: usize,
) -> f64 {
    let cmd_buf = gpu
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");

    // Pass 1: Partition kernel
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");
        encoder.setComputePipelineState(partition_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(kv_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(pt_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o_partial_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(m_partial_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(l_partial_buf), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(params_buf), 0, 6);
        }
        let threadgroups = MTLSize {
            width: num_query_blocks,
            height: num_partitions,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        encoder.endEncoding();
    }

    // Pass 2: Reduce kernel
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");
        encoder.setComputePipelineState(reduce_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(o_partial_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(m_partial_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(l_partial_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(o_final_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(params_buf), 0, 4);
        }
        let threadgroups = MTLSize {
            width: num_query_blocks,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: block_r * head_dim,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        encoder.endEncoding();
    }

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let gpu_start = cmd_buf.GPUStartTime();
    let gpu_end = cmd_buf.GPUEndTime();
    gpu_end - gpu_start
}

/// Run one GPU dispatch of contiguous flash attention and return GPU time in seconds.
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

    encoder
        .dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threads_per_threadgroup);
    encoder.endEncoding();

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let gpu_start = cmd_buf.GPUStartTime();
    let gpu_end = cmd_buf.GPUEndTime();
    gpu_end - gpu_start
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

fn bench_paged_attention(c: &mut Criterion) {
    let gpu = GpuDevice::shared();

    let head_dim: usize = 64;
    let page_size: usize = 16;
    let block_r: usize = 16;
    let num_partitions: usize = 1;
    let flash_block_r: u32 = 16;
    let flash_block_c: u32 = 64;
    let num_heads: u32 = 1;

    // Compile paged attention PSOs (partition + reduce)
    let partition_key = PsoKey::simple("paged_attention_partition")
        .with_uint(0, head_dim as u32)
        .with_uint(1, page_size as u32);
    let mut partition_cache = PsoCache::new(gpu.library.clone());
    let partition_pso = partition_cache.get_or_compile(&partition_key);

    let reduce_key = PsoKey::simple("paged_attention_reduce")
        .with_uint(0, head_dim as u32)
        .with_uint(1, page_size as u32);
    let mut reduce_cache = PsoCache::new(gpu.library.clone());
    let reduce_pso = reduce_cache.get_or_compile(&reduce_key);

    // Compile contiguous flash attention PSO (baseline)
    let flash_key = PsoKey::simple("flash_attention")
        .with_uint(0, head_dim as u32)
        .with_uint(1, flash_block_r)
        .with_uint(2, flash_block_c);
    let mut flash_cache = PsoCache::new(gpu.library.clone());
    let flash_pso = flash_cache.get_or_compile(&flash_key);

    let mut group = c.benchmark_group("paged_attention");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(3));

    // Collect results for summary printout
    let mut summary: Vec<(usize, f64, f64, f64)> = Vec::new(); // (seq_len, paged_us, flash_us, overhead%)

    for &seq_len in &[256usize, 512, 1024] {
        let n_elements = seq_len * head_dim;
        let num_query_blocks = (seq_len + block_r - 1) / block_r;
        let num_pages = (seq_len + page_size - 1) / page_size;

        // Generate deterministic data
        let q_data = lcg_data(n_elements, 42);
        let k_data = lcg_data(n_elements, 137);
        let v_data = lcg_data(n_elements, 256);

        // Build paged KV cache from contiguous K/V
        let cache = create_paged_kv_cache(&k_data, &v_data, seq_len, head_dim, page_size);

        // Allocate paged attention buffers
        let q_buf = alloc_buffer_with_data(&gpu.device, &q_data);
        let kv_buf = alloc_buffer_with_data(&gpu.device, &cache.kv_data);
        let pt_buf = alloc_buffer_with_data(&gpu.device, &cache.page_table);

        let total_pb = num_query_blocks * num_partitions;
        let o_partial_buf = alloc_buffer(
            &gpu.device,
            total_pb * block_r * head_dim * std::mem::size_of::<f32>(),
        );
        let ml_size = total_pb * block_r * std::mem::size_of::<f32>();
        let m_partial_buf = alloc_buffer(&gpu.device, ml_size);
        let l_partial_buf = alloc_buffer(&gpu.device, ml_size);
        let o_paged_buf =
            alloc_buffer(&gpu.device, n_elements * std::mem::size_of::<f32>());

        let paged_params = AttentionParams::paged(
            seq_len as u32,
            head_dim as u32,
            1,
            page_size as u32,
            num_pages as u32,
            seq_len as u32,
            num_partitions as u32,
        );
        let paged_params_buf =
            alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&paged_params));

        // Allocate contiguous flash attention buffers
        let k_buf = alloc_buffer_with_data(&gpu.device, &k_data);
        let v_buf = alloc_buffer_with_data(&gpu.device, &v_data);
        let o_flash_buf =
            alloc_buffer(&gpu.device, n_elements * std::mem::size_of::<f32>());
        let flash_params = AttentionParams::flash(seq_len as u32, head_dim as u32, num_heads);
        let flash_params_buf =
            alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&flash_params));

        // Warmup: 8 dispatches of each
        for _ in 0..8 {
            gpu_dispatch_paged(
                gpu,
                partition_pso,
                reduce_pso,
                &q_buf,
                &kv_buf,
                &pt_buf,
                &o_partial_buf,
                &m_partial_buf,
                &l_partial_buf,
                &o_paged_buf,
                &paged_params_buf,
                num_query_blocks,
                num_partitions,
                block_r,
                head_dim,
            );
            gpu_dispatch_flash(
                gpu,
                flash_pso,
                &q_buf,
                &k_buf,
                &v_buf,
                &o_flash_buf,
                &flash_params_buf,
                seq_len,
                flash_block_r,
                num_heads,
            );
        }

        // Benchmark: paged attention
        group.bench_with_input(
            BenchmarkId::new(format!("paged_N={seq_len}"), seq_len),
            &seq_len,
            |b, &seq_len| {
                let _ = seq_len;
                b.iter_custom(|iters| {
                    let mut total = 0.0f64;
                    for _ in 0..iters {
                        total += gpu_dispatch_paged(
                            gpu,
                            partition_pso,
                            reduce_pso,
                            &q_buf,
                            &kv_buf,
                            &pt_buf,
                            &o_partial_buf,
                            &m_partial_buf,
                            &l_partial_buf,
                            &o_paged_buf,
                            &paged_params_buf,
                            num_query_blocks,
                            num_partitions,
                            block_r,
                            head_dim,
                        );
                    }
                    Duration::from_secs_f64(total)
                });
            },
        );

        // Benchmark: contiguous flash (baseline)
        group.bench_with_input(
            BenchmarkId::new(format!("contiguous_N={seq_len}"), seq_len),
            &seq_len,
            |b, &seq_len| {
                let _ = seq_len;
                b.iter_custom(|iters| {
                    let mut total = 0.0f64;
                    for _ in 0..iters {
                        total += gpu_dispatch_flash(
                            gpu,
                            flash_pso,
                            &q_buf,
                            &k_buf,
                            &v_buf,
                            &o_flash_buf,
                            &flash_params_buf,
                            seq_len,
                            flash_block_r,
                            num_heads,
                        );
                    }
                    Duration::from_secs_f64(total)
                });
            },
        );

        // Post-benchmark: collect 10 samples for summary stats
        let mut paged_times = Vec::with_capacity(10);
        let mut flash_times = Vec::with_capacity(10);
        for _ in 0..10 {
            paged_times.push(gpu_dispatch_paged(
                gpu,
                partition_pso,
                reduce_pso,
                &q_buf,
                &kv_buf,
                &pt_buf,
                &o_partial_buf,
                &m_partial_buf,
                &l_partial_buf,
                &o_paged_buf,
                &paged_params_buf,
                num_query_blocks,
                num_partitions,
                block_r,
                head_dim,
            ));
            flash_times.push(gpu_dispatch_flash(
                gpu,
                flash_pso,
                &q_buf,
                &k_buf,
                &v_buf,
                &o_flash_buf,
                &flash_params_buf,
                seq_len,
                flash_block_r,
                num_heads,
            ));
        }

        let (paged_mean, paged_cv) = stats(&paged_times);
        let (flash_mean, flash_cv) = stats(&flash_times);
        let overhead_pct = if flash_mean > 0.0 {
            (paged_mean - flash_mean) / flash_mean * 100.0
        } else {
            0.0
        };

        summary.push((seq_len, paged_mean * 1e6, flash_mean * 1e6, overhead_pct));

        eprintln!(
            "[PagedAttention] N={seq_len} page_size={page_size} D={head_dim} partitions={num_partitions}:"
        );
        eprintln!(
            "  Paged:      {:.1}us (CV={:.1}%)",
            paged_mean * 1e6,
            paged_cv
        );
        eprintln!(
            "  Contiguous: {:.1}us (CV={:.1}%)",
            flash_mean * 1e6,
            flash_cv
        );
        eprintln!("  Overhead:   {:.1}%", overhead_pct);
    }

    group.finish();

    // Print summary table
    eprintln!();
    eprintln!("=== PagedAttention V2 Overhead Summary ===");
    eprintln!(
        "{:<10} {:>12} {:>12} {:>10}",
        "seq_len", "paged(us)", "contig(us)", "overhead%"
    );
    for (seq_len, paged_us, flash_us, overhead) in &summary {
        eprintln!(
            "{:<10} {:>12.1} {:>12.1} {:>10.1}",
            seq_len, paged_us, flash_us, overhead
        );
    }
    eprintln!("==========================================");
}

criterion_group!(benches, bench_paged_attention);
criterion_main!(benches);
