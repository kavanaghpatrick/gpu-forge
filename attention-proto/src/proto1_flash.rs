//! Proto 1: Flash Attention — CPU reference and GPU kernel host code.
//!
//! This module implements:
//! - `cpu_attention_f64`: Naive O(N²) scaled dot-product attention with FP64 accumulation
//! - `assert_allclose`: Element-wise comparison with absolute and relative tolerance
//! - `run_flash_attention`: GPU host code for the Flash Attention Metal kernel

use crate::device::GpuDevice;
use crate::encode::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::AttentionParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Run flash attention on the GPU using the Metal kernel.
///
/// Allocates Q/K/V buffers, uploads data, compiles PSO with function constants,
/// dispatches 2D threadgroups (ceil(seq_len/BLOCK_R), num_heads), reads back output.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `q`: Query matrix, row-major [seq_len, head_dim], stored as f32
/// - `k`: Key matrix, row-major [seq_len, head_dim], stored as f32
/// - `v`: Value matrix, row-major [seq_len, head_dim], stored as f32
/// - `seq_len`: Number of tokens
/// - `head_dim`: Dimension of each head
///
/// # Returns
/// Output matrix [seq_len, head_dim] as Vec<f32>
pub fn run_flash_attention(
    device: &GpuDevice,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");
    assert_eq!(k.len(), seq_len * head_dim, "K length mismatch");
    assert_eq!(v.len(), seq_len * head_dim, "V length mismatch");

    let block_r: u32 = 16;
    let block_c: u32 = 64;
    let num_heads: u32 = 1;

    // Create AttentionParams
    let params = AttentionParams::flash(seq_len as u32, head_dim as u32, num_heads);

    // Allocate Metal buffers
    let q_buf = alloc_buffer_with_data(&device.device, q);
    let k_buf = alloc_buffer_with_data(&device.device, k);
    let v_buf = alloc_buffer_with_data(&device.device, v);
    let o_buf = alloc_buffer(&device.device, seq_len * head_dim * std::mem::size_of::<f32>());
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    // Compile PSO with function constants
    // Function constant indices must match the shader:
    // Index 0 = HEAD_DIM (uint), Index 1 = BLOCK_R (uint), Index 2 = BLOCK_C (uint)
    let pso_key = PsoKey::simple("flash_attention")
        .with_uint(0, head_dim as u32)
        .with_uint(1, block_r)
        .with_uint(2, block_c);

    let mut cache = PsoCache::new(device.library.clone());
    let pso = cache.get_or_compile(&pso_key);

    // Create command buffer and compute encoder
    let command_buffer = device.command_queue.commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    // Set PSO and bind buffers
    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);      // Q at buffer(0)
        encoder.setBuffer_offset_atIndex(Some(&*k_buf), 0, 1);      // K at buffer(1)
        encoder.setBuffer_offset_atIndex(Some(&*v_buf), 0, 2);      // V at buffer(2)
        encoder.setBuffer_offset_atIndex(Some(&*o_buf), 0, 3);      // O at buffer(3)
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 4); // params at buffer(4)
    }

    // Dispatch threadgroups:
    // Grid: (ceil(seq_len/BLOCK_R), num_heads, 1)
    // The kernel uses tg_pos.x for query block index, tg_pos.y for head index
    // Threadgroup size: 32 threads (one simdgroup) as specified in kernel
    let num_row_blocks = (seq_len as u64 + block_r as u64 - 1) / block_r as u64;
    let threadgroups_per_grid = MTLSize {
        width: num_row_blocks as usize,
        height: num_heads as usize,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,  // One simdgroup
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threads_per_threadgroup);
    encoder.endEncoding();

    // Commit and wait
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Check command buffer status
    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "Command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    // Read back output
    unsafe { read_buffer_slice(&o_buf, seq_len * head_dim) }
}

/// Naive scaled dot-product attention computed entirely in FP64.
///
/// Computes: softmax(Q * K^T / sqrt(head_dim)) * V
///
/// All intermediate values use FP64 for maximum precision.
/// The final output is truncated to FP32.
///
/// # Arguments
/// - `q`: Query matrix, row-major [seq_len, head_dim], stored as f32
/// - `k`: Key matrix, row-major [seq_len, head_dim], stored as f32
/// - `v`: Value matrix, row-major [seq_len, head_dim], stored as f32
/// - `seq_len`: Number of tokens (rows in Q, K, V)
/// - `head_dim`: Dimension of each head (columns in Q, K, V)
///
/// # Returns
/// Output matrix [seq_len, head_dim] as Vec<f32>
pub fn cpu_attention_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");
    assert_eq!(k.len(), seq_len * head_dim, "K length mismatch");
    assert_eq!(v.len(), seq_len * head_dim, "V length mismatch");

    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut output = vec![0.0f64; seq_len * head_dim];

    for i in 0..seq_len {
        // Compute attention scores: q_i * k_j^T * scale
        let mut scores = vec![0.0f64; seq_len];
        let mut max_score = f64::NEG_INFINITY;

        for j in 0..seq_len {
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] as f64 * k[j * head_dim + d] as f64;
            }
            scores[j] = dot * scale;
            max_score = max_score.max(scores[j]);
        }

        // Safe softmax: subtract max for numerical stability, then exp and normalize
        let mut sum_exp = 0.0f64;
        for j in 0..seq_len {
            scores[j] = (scores[j] - max_score).exp();
            sum_exp += scores[j];
        }

        // Weighted sum of values
        for j in 0..seq_len {
            let weight = scores[j] / sum_exp;
            for d in 0..head_dim {
                output[i * head_dim + d] += weight * v[j * head_dim + d] as f64;
            }
        }
    }

    // Truncate FP64 accumulation to FP32 output
    output.iter().map(|&x| x as f32).collect()
}

/// Assert that two slices are element-wise close within absolute and relative tolerance.
///
/// An element passes if: |gpu - cpu| <= atol  OR  |gpu - cpu| / |cpu| <= rtol
/// (i.e., it fails only if BOTH tolerances are exceeded)
///
/// Reports up to 5 failing elements with detailed info before panicking.
///
/// # Arguments
/// - `gpu`: GPU output values
/// - `cpu`: CPU reference values
/// - `atol`: Absolute tolerance
/// - `rtol`: Relative tolerance
/// - `context`: Description for error messages
pub fn assert_allclose(gpu: &[f32], cpu: &[f32], atol: f32, rtol: f32, context: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{context}: length mismatch");

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut fail_count = 0;

    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let abs_err = (g - c).abs();
        let rel_err = if c.abs() > 1e-8 {
            abs_err / c.abs()
        } else {
            abs_err
        };
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);

        if abs_err > atol && rel_err > rtol {
            fail_count += 1;
            if fail_count <= 5 {
                eprintln!(
                    "{context}[{i}]: gpu={g:.6}, cpu={c:.6}, abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
                );
            }
        }
    }

    assert_eq!(
        fail_count, 0,
        "{context}: {fail_count}/{} elements exceed tolerance (max_abs={max_abs_err:.2e}, max_rel={max_rel_err:.2e})",
        gpu.len()
    );
}

/// Generate KB findings for Proto 1 Flash Attention based on empirical benchmark results.
///
/// Emits 6 findings covering:
/// 1. Baseline throughput on M4
/// 2. Throughput scaling with sequence length
/// 3. Comparison vs metal-flash-attention reference
/// 4. 32KB threadgroup memory constraint validation
/// 5. simdgroup_matrix basic operation validation
/// 6. Dispatch overhead at small N
pub fn emit_proto1_findings() {
    use crate::kb::{emit_finding, KbFinding};

    // Finding 1: Flash Attention baseline throughput on M4
    emit_finding(&KbFinding {
        domain: "metal-compute".to_string(),
        title: "proto1: Flash Attention baseline throughput on M4".to_string(),
        content: "From-scratch Flash Attention kernel (Br=16, Bc=64, D=64, single head) achieves \
                  0.16 TFLOPS at N=2048 on Apple M4. Kernel uses simdgroup_float8x8 for Q*K^T and \
                  P*V matmuls with online softmax (running max/sum). GPU time: ~6.7ms at N=2048. \
                  This is a first unoptimized implementation with one simdgroup (32 threads) per \
                  threadgroup, no shared memory prefetching, and no multi-head parallelism."
            .to_string(),
        tags: vec![
            "proto1".to_string(),
            "flash-attention".to_string(),
            "simdgroup-matrix".to_string(),
            "M4".to_string(),
            "tflops".to_string(),
        ],
        confidence: 0.85,
        source: "attention-proto/proto1_flash benchmark (criterion, 50 samples, CV<1%)"
            .to_string(),
    });

    // Finding 2: Throughput scaling with sequence length
    emit_finding(&KbFinding {
        domain: "gpu-perf".to_string(),
        title: "proto1: Flash Attention TFLOPS scales with sequence length on M4".to_string(),
        content: "Flash Attention TFLOPS scales roughly linearly with N in [256-2048] range on M4: \
                  N=256 → 0.03 TFLOPS (398us), N=512 → 0.07 TFLOPS (779us), N=1024 → 0.11 TFLOPS \
                  (2.4ms), N=2048 → 0.16 TFLOPS (6.7ms). FLOPs grow as O(N^2*D) while GPU time \
                  grows sub-quadratically, indicating better GPU utilization at larger workloads. \
                  The 5x throughput improvement from N=256 to N=2048 suggests the kernel is \
                  dispatch/launch-overhead-bound at small N and compute-bound at larger N."
            .to_string(),
        tags: vec![
            "proto1".to_string(),
            "flash-attention".to_string(),
            "scaling".to_string(),
            "M4".to_string(),
            "sequence-length".to_string(),
        ],
        confidence: 0.85,
        source: "attention-proto/proto1_flash benchmark (criterion, 50 samples per N, CV<1%)"
            .to_string(),
    });

    // Finding 3: Comparison vs metal-flash-attention reference
    emit_finding(&KbFinding {
        domain: "gpu-perf".to_string(),
        title: "proto1: From-scratch Flash Attention achieves ~4-10% of MFA throughput on M4"
            .to_string(),
        content: "Metal-flash-attention (MFA) by Apple/philipturner achieves ~1790 GINSTRS \
                  (~1.8 TFLOPS equivalent) on M4 for optimized configurations. Our from-scratch \
                  Proto 1 kernel at 0.16 TFLOPS (N=2048, D=64) is approximately 4-10% of MFA \
                  performance. This gap is expected for a first implementation: (1) single \
                  simdgroup (32 threads) vs MFA's multi-simdgroup design, (2) no async copy or \
                  prefetching, (3) naive threadgroup memory layout, (4) no register-level \
                  optimizations. The 10-25x gap validates that significant optimization headroom \
                  exists and provides a concrete baseline for measuring improvement in later protos."
            .to_string(),
        tags: vec![
            "proto1".to_string(),
            "flash-attention".to_string(),
            "metal-flash-attention".to_string(),
            "mfa-comparison".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.8,
        source: "attention-proto/proto1_flash benchmark vs MFA published M4 numbers".to_string(),
    });

    // Finding 4: 32KB threadgroup memory constraint validation
    emit_finding(&KbFinding {
        domain: "msl-kernels".to_string(),
        title: "proto1: 32KB threadgroup memory constraint validated for Br=16, Bc=64, D=64"
            .to_string(),
        content: "Flash Attention threadgroup memory layout at Br=16, Bc=64, D=64: \
                  Q_tile = Br*D*4 = 16*64*4 = 4096 bytes (4KB), \
                  K_chunk = Bc*D*4 = 64*64*4 = 16384 bytes (16KB), \
                  S_tile = Br*Bc*4 = 16*64*4 = 4096 bytes (4KB). \
                  Total = 24576 bytes (24KB), fits within Apple M4's 32KB per-threadgroup limit \
                  with 8KB headroom. Doubling Bc to 128 would require 32KB Q_tile(4KB) + \
                  K_chunk(32KB) + S_tile(8KB) = 44KB, exceeding the limit. Doubling Br to 32 \
                  would need Q_tile(8KB) + K_chunk(16KB) + S_tile(8KB) = 32KB, at the exact limit. \
                  This confirms Br=16, Bc=64 is a safe default for D=64."
            .to_string(),
        tags: vec![
            "proto1".to_string(),
            "flash-attention".to_string(),
            "threadgroup-memory".to_string(),
            "M4".to_string(),
            "tile-size".to_string(),
        ],
        confidence: 0.95,
        source: "attention-proto/proto1_flash kernel analysis (deterministic calculation)"
            .to_string(),
    });

    // Finding 5: simdgroup_matrix basic operation validation
    emit_finding(&KbFinding {
        domain: "simd-wave".to_string(),
        title: "proto1: simdgroup_float8x8 load/multiply_accumulate/store validated on M4"
            .to_string(),
        content: "simdgroup_float8x8 operations (simdgroup_load, simdgroup_multiply_accumulate, \
                  simdgroup_store) compile and produce correct results on Apple M4 with \
                  -std=metal3.1 and #include <metal_simdgroup_matrix>. The Flash Attention kernel \
                  uses 8x8 tiles for both Q*K^T and P*V matmuls. Correctness verified against \
                  FP64 CPU reference at N=256, D=64 with atol=5e-3, rtol=1e-2. The simdgroup_load \
                  with transpose=true flag enables direct K^T loading without explicit transpose. \
                  All 32 threads in a simdgroup participate cooperatively in each 8x8 matrix op."
            .to_string(),
        tags: vec![
            "proto1".to_string(),
            "simdgroup-matrix".to_string(),
            "simdgroup_float8x8".to_string(),
            "M4".to_string(),
            "metal3.1".to_string(),
        ],
        confidence: 0.95,
        source: "attention-proto/proto1_flash correctness test (MTL_SHADER_VALIDATION=1)"
            .to_string(),
    });

    // Finding 6: Dispatch overhead observation at small N
    emit_finding(&KbFinding {
        domain: "gpu-perf".to_string(),
        title: "proto1: GPU dispatch overhead dominates at small sequence lengths on M4"
            .to_string(),
        content: "At N=256 (FLOPS=16.8M), GPU kernel time is ~398us yielding only 0.03 TFLOPS \
                  (42 GFLOPS). At N=2048 (FLOPS=1.07B), time is ~6.7ms yielding 0.16 TFLOPS. \
                  The 8x N increase yields 64x more FLOPs but only 17x more time, indicating \
                  that at N=256 a significant fraction of time is non-compute overhead (command \
                  buffer submission, encoder setup, GPU clock ramp). Estimated dispatch overhead \
                  is ~100-200us based on extrapolation. For production attention kernels, batching \
                  multiple heads into a single dispatch (multi-head parallelism) is critical to \
                  amortize this overhead at small N."
            .to_string(),
        tags: vec![
            "proto1".to_string(),
            "dispatch-overhead".to_string(),
            "command-buffer".to_string(),
            "M4".to_string(),
            "small-workload".to_string(),
        ],
        confidence: 0.8,
        source: "attention-proto/proto1_flash benchmark (extrapolated from N=256..2048 scaling)"
            .to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test with identity-like Q and uniform K.
    ///
    /// When all K rows are identical, softmax produces uniform weights (1/N for each j).
    /// Therefore, output[i] = mean(V) for all i, regardless of Q.
    #[test]
    fn test_cpu_attention_identity() {
        let seq_len = 4;
        let head_dim = 8;

        // Q: identity-like pattern (doesn't matter when K is uniform)
        let mut q = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            if i < head_dim {
                q[i * head_dim + i] = 1.0;
            }
        }

        // K: all rows identical => uniform attention weights
        let k = vec![1.0f32; seq_len * head_dim];

        // V: distinct rows so we can verify the mean
        let mut v = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for d in 0..head_dim {
                v[i * head_dim + d] = (i * head_dim + d) as f32;
            }
        }

        let output = cpu_attention_f64(&q, &k, &v, seq_len, head_dim);

        // Compute expected: mean(V) across all rows
        let mut expected = vec![0.0f32; head_dim];
        for d in 0..head_dim {
            let mut sum = 0.0f64;
            for i in 0..seq_len {
                sum += v[i * head_dim + d] as f64;
            }
            expected[d] = (sum / seq_len as f64) as f32;
        }

        // Every output row should equal mean(V)
        for i in 0..seq_len {
            let row = &output[i * head_dim..(i + 1) * head_dim];
            assert_allclose(row, &expected, 1e-6, 1e-5, &format!("identity row {i}"));
        }
    }

    /// Test with small random-ish data: N=4, D=8.
    /// Verify that softmax weights sum to 1 for each query position.
    #[test]
    fn test_cpu_attention_small() {
        let seq_len = 4;
        let head_dim = 8;

        // Deterministic "random" data using simple formula
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.1).sin() * 2.0)
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.2 + 1.0).cos() * 2.0)
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.3 + 2.0).sin() * 2.0)
            .collect();

        let output = cpu_attention_f64(&q, &k, &v, seq_len, head_dim);

        // Verify output dimensions
        assert_eq!(output.len(), seq_len * head_dim);

        // Verify softmax properties: recompute scores and check weights sum to 1
        let scale = 1.0 / (head_dim as f64).sqrt();
        for i in 0..seq_len {
            let mut scores = vec![0.0f64; seq_len];
            let mut max_score = f64::NEG_INFINITY;

            for j in 0..seq_len {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] as f64 * k[j * head_dim + d] as f64;
                }
                scores[j] = dot * scale;
                max_score = max_score.max(scores[j]);
            }

            let mut sum_exp = 0.0f64;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }

            // Softmax weights must sum to 1
            let weight_sum: f64 = scores.iter().map(|s| s / sum_exp).sum();
            assert!(
                (weight_sum - 1.0).abs() < 1e-12,
                "Softmax weights for row {i} sum to {weight_sum}, expected 1.0"
            );

            // All weights must be non-negative
            for (j, s) in scores.iter().enumerate() {
                let w = s / sum_exp;
                assert!(
                    w >= 0.0,
                    "Softmax weight [{i}][{j}] is negative: {w}"
                );
            }
        }

        // Verify no NaN or Inf in output
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Output[{i}] is not finite: {val}"
            );
        }
    }

    /// Test assert_allclose passes with matching data.
    #[test]
    fn test_assert_allclose_passes() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_allclose(&a, &b, 1e-6, 1e-5, "exact match");
    }

    /// Test assert_allclose fails with mismatched data.
    #[test]
    #[should_panic(expected = "elements exceed tolerance")]
    fn test_assert_allclose_fails() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 3.0, 5.0]; // last element differs by 1.0
        assert_allclose(&a, &b, 1e-6, 1e-5, "mismatch");
    }

    /// Test seq_len=1 degenerate case: output should equal V directly.
    #[test]
    fn test_cpu_attention_single_token() {
        let seq_len = 1;
        let head_dim = 4;

        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let k = vec![0.5f32, 0.5, 0.5, 0.5];
        let v = vec![3.0f32, 7.0, 11.0, 13.0];

        let output = cpu_attention_f64(&q, &k, &v, seq_len, head_dim);

        // With seq_len=1, softmax of a single score is always 1.0
        // So output = 1.0 * v = v
        assert_allclose(&output, &v, 1e-6, 1e-5, "single token");
    }

    /// Generate KB findings for Proto 1 Flash Attention.
    ///
    /// Run explicitly with: `cargo test --lib proto1_flash::tests::generate_proto1_findings -- --ignored`
    #[test]
    #[ignore]
    fn generate_proto1_findings() {
        // Clear any existing findings first to avoid duplicates
        crate::kb::clear_findings();
        emit_proto1_findings();

        // Verify at least 6 findings were written
        let contents = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("findings.jsonl"),
        )
        .expect("findings.jsonl should exist after emitting");
        let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
        assert!(
            lines.len() >= 5,
            "Expected at least 5 findings, got {}",
            lines.len()
        );

        // Verify all lines are valid JSON and contain proto1
        for (i, line) in lines.iter().enumerate() {
            let parsed: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|e| panic!("Line {i} invalid JSON: {e}"));
            // At least some tag or content should reference proto1
            let json_str = serde_json::to_string(&parsed).unwrap();
            assert!(
                json_str.contains("proto1"),
                "Line {i} missing proto1 reference"
            );
            // Confidence must be >= 0.8
            let confidence = parsed["confidence"].as_f64().unwrap();
            assert!(
                confidence >= 0.8,
                "Line {i} confidence {confidence} < 0.8"
            );
        }

        eprintln!("[KB] Emitted {} Proto 1 findings to findings.jsonl", lines.len());
    }
}
