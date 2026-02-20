//! Criterion benchmark for Proto 2: Function stitching overhead measurement.
//!
//! Compiles 3 PSOs with STITCH_MODE 0/1/2 (monolithic/always_inline/noinline),
//! runs identical Flash Attention workloads (N=1024, D=64), and measures GPU
//! time via GPUStartTime/GPUEndTime. Reports ns/call overhead per mode.
//!
//! The goal is to determine whether factoring inner-loop operations into
//! separate functions introduces measurable overhead on Apple M4 GPU.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::ptr::NonNull;
use std::time::Duration;

use attention_proto::device::GpuDevice;
use attention_proto::encode::{alloc_buffer, alloc_buffer_with_data};
use attention_proto::types::AttentionParams;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLSize,
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

/// Compile a PSO for flash_attention_stitched with given STITCH_MODE.
///
/// Returns an owned Retained<> to avoid borrow checker issues with PsoCache.
fn compile_stitched_pso(
    library: &ProtocolObject<dyn MTLLibrary>,
    device: &ProtocolObject<dyn MTLDevice>,
    head_dim: u32,
    block_r: u32,
    block_c: u32,
    stitch_mode: u32,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let constant_values = MTLFunctionConstantValues::new();

    // Index 0: HEAD_DIM
    let hd = head_dim;
    unsafe {
        let ptr = NonNull::new(&hd as *const u32 as *mut std::ffi::c_void).unwrap();
        constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::UInt, 0);
    }

    // Index 1: BLOCK_R_FC
    let br = block_r;
    unsafe {
        let ptr = NonNull::new(&br as *const u32 as *mut std::ffi::c_void).unwrap();
        constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::UInt, 1);
    }

    // Index 2: BLOCK_C_FC
    let bc = block_c;
    unsafe {
        let ptr = NonNull::new(&bc as *const u32 as *mut std::ffi::c_void).unwrap();
        constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::UInt, 2);
    }

    // Index 3: STITCH_MODE
    let sm = stitch_mode;
    unsafe {
        let ptr = NonNull::new(&sm as *const u32 as *mut std::ffi::c_void).unwrap();
        constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::UInt, 3);
    }

    let fn_name = NSString::from_str("flash_attention_stitched");
    let function = library
        .newFunctionWithName_constantValues_error(&fn_name, &constant_values)
        .unwrap_or_else(|e| {
            panic!(
                "Failed to create function for STITCH_MODE={stitch_mode}: {:?}",
                e
            )
        });

    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|e| {
            panic!(
                "Failed to compile PSO for STITCH_MODE={stitch_mode}: {:?}",
                e
            )
        })
}

/// Run one GPU dispatch of flash_attention_stitched and return GPU time in seconds.
fn gpu_dispatch_stitched(
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

    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        threadgroups_per_grid,
        threads_per_threadgroup,
    );
    encoder.endEncoding();

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let gpu_start = cmd_buf.GPUStartTime();
    let gpu_end = cmd_buf.GPUEndTime();
    gpu_end - gpu_start
}

/// Stitch mode labels.
const MODE_NAMES: [&str; 3] = ["monolithic", "always_inline", "noinline"];

fn bench_function_stitch(c: &mut Criterion) {
    let gpu = GpuDevice::shared();

    let seq_len: usize = 1024;
    let head_dim: usize = 64;
    let block_r: u32 = 16;
    let block_c: u32 = 64;
    let num_heads: u32 = 1;

    // Compile 3 owned PSOs: one per STITCH_MODE (0=monolithic, 1=always_inline, 2=noinline)
    let psos: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>> = (0..3u32)
        .map(|mode| {
            compile_stitched_pso(&gpu.library, &gpu.device, head_dim as u32, block_r, block_c, mode)
        })
        .collect();

    // Allocate Q/K/V/O buffers for N=1024, D=64
    let n_elements = seq_len * head_dim;
    let q_data = lcg_data(n_elements, 42);
    let k_data = lcg_data(n_elements, 137);
    let v_data = lcg_data(n_elements, 256);

    let q_buf = alloc_buffer_with_data(&gpu.device, &q_data);
    let k_buf = alloc_buffer_with_data(&gpu.device, &k_data);
    let v_buf = alloc_buffer_with_data(&gpu.device, &v_data);
    let o_buf = alloc_buffer(&gpu.device, n_elements * std::mem::size_of::<f32>());
    let params = AttentionParams::flash(seq_len as u32, head_dim as u32, num_heads);
    let params_buf = alloc_buffer_with_data(&gpu.device, std::slice::from_ref(&params));

    // GPU warmup: 4 dispatches per mode to stabilize clocks and caches
    for pso in &psos {
        for _ in 0..4 {
            gpu_dispatch_stitched(
                gpu, pso, &q_buf, &k_buf, &v_buf, &o_buf, &params_buf, seq_len, block_r,
                num_heads,
            );
        }
    }

    // --- Criterion benchmark group ---
    let mut group = c.benchmark_group("function_stitch");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    for (mode, pso) in psos.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new(MODE_NAMES[mode], mode),
            &mode,
            |b, &_mode| {
                b.iter_custom(|iters| {
                    let mut total_gpu_secs = 0.0f64;
                    for _ in 0..iters {
                        let gpu_time = gpu_dispatch_stitched(
                            gpu, pso, &q_buf, &k_buf, &v_buf, &o_buf, &params_buf, seq_len,
                            block_r, num_heads,
                        );
                        total_gpu_secs += gpu_time;
                    }
                    Duration::from_secs_f64(total_gpu_secs)
                });
            },
        );
    }

    group.finish();

    // --- Post-benchmark summary: 100 iterations per mode for statistical comparison ---
    let num_samples = 100;
    let mut mode_times: Vec<Vec<f64>> = vec![Vec::with_capacity(num_samples); 3];

    for (mode, pso) in psos.iter().enumerate() {
        // Extra warmup before measurement pass
        for _ in 0..8 {
            gpu_dispatch_stitched(
                gpu, pso, &q_buf, &k_buf, &v_buf, &o_buf, &params_buf, seq_len, block_r,
                num_heads,
            );
        }

        for _ in 0..num_samples {
            let t = gpu_dispatch_stitched(
                gpu, pso, &q_buf, &k_buf, &v_buf, &o_buf, &params_buf, seq_len, block_r,
                num_heads,
            );
            mode_times[mode].push(t);
        }
    }

    // Compute statistics
    eprintln!("\n=== Function Stitching Overhead Summary ===");
    eprintln!(
        "Workload: N={seq_len}, D={head_dim}, Br={block_r}, Bc={block_c}, heads={num_heads}"
    );
    eprintln!("Samples per mode: {num_samples}\n");

    let mut means = [0.0f64; 3];
    let mut stddevs = [0.0f64; 3];

    for mode in 0..3 {
        let times = &mode_times[mode];
        let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
        let variance: f64 =
            times.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
        let std_dev = variance.sqrt();
        let cv = if mean > 0.0 {
            std_dev / mean * 100.0
        } else {
            0.0
        };
        let mean_us = mean * 1e6;
        let std_us = std_dev * 1e6;

        means[mode] = mean;
        stddevs[mode] = std_dev;

        // Compute FLOPS: 2 matmuls (QK^T and PV), each 2*N*N*D
        let flops: u64 = 4 * (seq_len as u64) * (seq_len as u64) * (head_dim as u64);
        let tflops = flops as f64 / mean / 1e12;

        eprintln!(
            "  {:<14} mean={mean_us:>8.1}us  std={std_us:>6.1}us  CV={cv:>4.1}%  TFLOPS={tflops:.4}",
            MODE_NAMES[mode],
        );
    }

    // Overhead comparison vs monolithic baseline
    eprintln!("\n--- Overhead vs monolithic (mode 0) ---");
    let baseline_ns = means[0] * 1e9;

    for mode in 1..3 {
        let mode_ns = means[mode] * 1e9;
        let overhead_ns = mode_ns - baseline_ns;
        let overhead_pct = if means[0] > 0.0 {
            (means[mode] - means[0]) / means[0] * 100.0
        } else {
            0.0
        };

        // Compute per-call overhead: each KV block iteration calls 3 functions
        let num_kv_blocks = (seq_len as u64 + block_c as u64 - 1) / block_c as u64;
        let calls_per_dispatch = num_kv_blocks * 3; // compute_scores + scale_scores + softmax_accumulate
        let overhead_per_call_ns = if calls_per_dispatch > 0 {
            overhead_ns / calls_per_dispatch as f64
        } else {
            0.0
        };

        eprintln!(
            "  {:<14} overhead={overhead_ns:>+8.1}ns total ({overhead_pct:>+5.2}%)  \
             ~{overhead_per_call_ns:.1}ns/call ({calls_per_dispatch} calls/dispatch)",
            MODE_NAMES[mode],
        );
    }

    // Determine if within noise floor
    let noise_floor_ns = stddevs[0] * 1e9 * 2.0; // 2-sigma noise threshold
    eprintln!("\n--- Statistical significance ---");
    eprintln!("  Noise floor (2*sigma of monolithic): {noise_floor_ns:.1}ns");

    for mode in 1..3 {
        let overhead_ns = (means[mode] - means[0]) * 1e9;
        let significant = overhead_ns.abs() > noise_floor_ns;
        let verdict = if significant {
            "SIGNIFICANT"
        } else {
            "within noise"
        };
        eprintln!(
            "  {:<14} |{overhead_ns:.1}ns| vs {noise_floor_ns:.1}ns threshold -> {verdict}",
            MODE_NAMES[mode],
        );
    }

    eprintln!();
}

criterion_group!(benches, bench_function_stitch);
criterion_main!(benches);
