//! Proto 6: FLA Linear Attention — CPU reference and GPU kernel host code.
//!
//! This module implements:
//! - `cpu_linear_attention_f64`: Chunk-based linear attention recurrence with FP64 accumulation
//! - `run_linear_attention`: Two-pass GPU dispatch (chunk_h then chunk_o) with CPU prefix sum
//!
//! Linear attention replaces softmax(Q*K^T)*V with a linear recurrence:
//!   H_c = H_{c-1} + K_chunk^T * V_chunk  (D x D outer product accumulation)
//!   O_chunk = Q_chunk * H_c              (C x D = C x D * D x D)
//!
//! This is the core mechanism from Flash Linear Attention (FLA).

use crate::device::GpuDevice;
use crate::encode::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::AttentionParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Run chunk-based linear attention on the GPU using two Metal kernels.
///
/// Two-pass dispatch:
/// 1. `chunk_h`: Each threadgroup computes delta_H[chunk] = K_chunk^T * V_chunk (D x D)
/// 2. CPU prefix sum: H_cumulative[c] = H_cumulative[c-1] + delta_H[c]
/// 3. `chunk_o`: Each threadgroup computes O_chunk = Q_chunk * H_cumulative (C x D)
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `q`: Query matrix [seq_len, head_dim]
/// - `k`: Key matrix [seq_len, head_dim]
/// - `v`: Value matrix [seq_len, head_dim]
/// - `seq_len`: Number of tokens (must be divisible by chunk_size)
/// - `head_dim`: Dimension of each head
/// - `chunk_size`: Number of tokens per chunk
///
/// # Returns
/// Output matrix [seq_len, head_dim] as Vec<f32>
pub fn run_linear_attention(
    device: &GpuDevice,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    chunk_size: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");
    assert_eq!(k.len(), seq_len * head_dim, "K length mismatch");
    assert_eq!(v.len(), seq_len * head_dim, "V length mismatch");
    assert!(
        seq_len.is_multiple_of(chunk_size),
        "seq_len must be divisible by chunk_size"
    );

    let num_chunks = seq_len / chunk_size;

    // Build AttentionParams — reuse existing struct, only seq_len and head_dim matter
    let params = AttentionParams {
        seq_len: seq_len as u32,
        head_dim: head_dim as u32,
        ..Default::default()
    };

    // Allocate Metal buffers for input data
    let k_buf = alloc_buffer_with_data(&device.device, k);
    let v_buf = alloc_buffer_with_data(&device.device, v);
    let q_buf = alloc_buffer_with_data(&device.device, q);
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    // H_deltas buffer: [num_chunks, head_dim, head_dim]
    let h_deltas_size = num_chunks * head_dim * head_dim;
    let h_deltas_buf = alloc_buffer(
        &device.device,
        h_deltas_size * std::mem::size_of::<f32>(),
    );

    // Compile chunk_h PSO with function constants: HEAD_DIM=index(0), CHUNK_SIZE=index(4)
    let chunk_h_key = PsoKey::simple("chunk_h")
        .with_uint(0, head_dim as u32)
        .with_uint(4, chunk_size as u32);
    let mut chunk_h_cache = PsoCache::new(device.library.clone());
    let chunk_h_pso = chunk_h_cache.get_or_compile(&chunk_h_key);

    // Determine threadgroup size for chunk_h: min(D*D, maxTotalThreadsPerThreadgroup)
    let max_threads_h = chunk_h_pso.maxTotalThreadsPerThreadgroup();
    let threads_per_tg_h = (head_dim * head_dim).min(max_threads_h);

    // ====================================================================
    // Pass 1: chunk_h kernel — compute delta_H per chunk
    // Grid: num_chunks threadgroups, each with threads_per_tg_h threads
    // ====================================================================
    {
        let command_buffer = device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer for chunk_h");

        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("Failed to create compute encoder for chunk_h");

        encoder.setComputePipelineState(chunk_h_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*k_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*v_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*h_deltas_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 3);
        }

        let threadgroups = MTLSize {
            width: num_chunks,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: threads_per_tg_h,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        let status = command_buffer.status();
        assert_eq!(
            status,
            objc2_metal::MTLCommandBufferStatus::Completed,
            "chunk_h command buffer failed with status {:?}. Error: {:?}",
            status,
            command_buffer.error()
        );
    }

    // ====================================================================
    // CPU prefix sum: H_cumulative[c] = sum(delta_H[0..=c])
    // Read back delta_H, compute prefix sum, upload H_cumulative
    // ====================================================================
    let h_deltas: Vec<f32> = unsafe { read_buffer_slice(&h_deltas_buf, h_deltas_size) };

    let dd = head_dim * head_dim;
    let mut h_cumulative = vec![0.0f32; num_chunks * dd];

    // H_cumulative[0] = H_deltas[0]
    h_cumulative[..dd].copy_from_slice(&h_deltas[..dd]);

    // H_cumulative[c] = H_cumulative[c-1] + H_deltas[c] (element-wise D x D matrix addition)
    for c in 1..num_chunks {
        for elem in 0..dd {
            h_cumulative[c * dd + elem] =
                h_cumulative[(c - 1) * dd + elem] + h_deltas[c * dd + elem];
        }
    }

    // Upload H_cumulative to GPU
    let h_cumulative_buf = alloc_buffer_with_data(&device.device, &h_cumulative);

    // Output buffer
    let o_buf = alloc_buffer(
        &device.device,
        seq_len * head_dim * std::mem::size_of::<f32>(),
    );

    // Compile chunk_o PSO
    let chunk_o_key = PsoKey::simple("chunk_o")
        .with_uint(0, head_dim as u32)
        .with_uint(4, chunk_size as u32);
    let mut chunk_o_cache = PsoCache::new(device.library.clone());
    let chunk_o_pso = chunk_o_cache.get_or_compile(&chunk_o_key);

    // Determine threadgroup size for chunk_o: min(C*D, maxTotalThreadsPerThreadgroup)
    let max_threads_o = chunk_o_pso.maxTotalThreadsPerThreadgroup();
    let threads_per_tg_o = (chunk_size * head_dim).min(max_threads_o);

    // ====================================================================
    // Pass 2: chunk_o kernel — compute O = Q * H_cumulative
    // Grid: num_chunks threadgroups, each with threads_per_tg_o threads
    // ====================================================================
    {
        let command_buffer = device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer for chunk_o");

        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("Failed to create compute encoder for chunk_o");

        encoder.setComputePipelineState(chunk_o_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*h_cumulative_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 3);
        }

        let threadgroups = MTLSize {
            width: num_chunks,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: threads_per_tg_o,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        let status = command_buffer.status();
        assert_eq!(
            status,
            objc2_metal::MTLCommandBufferStatus::Completed,
            "chunk_o command buffer failed with status {:?}. Error: {:?}",
            status,
            command_buffer.error()
        );
    }

    // Read back output
    unsafe { read_buffer_slice(&o_buf, seq_len * head_dim) }
}

/// Chunk-based linear attention computed entirely in FP64.
///
/// Implements the recurrence:
///   H_0 = 0 (D x D zero matrix)
///   For each chunk c of chunk_size tokens:
///     H_c = H_{c-1} + sum_{t in chunk} K[t]^T * V[t]  (outer product accumulation)
///     O_chunk = Q_chunk * H_c                          (matrix-vector products)
///
/// All intermediate values use FP64 for maximum precision.
/// The final output is truncated to FP32.
///
/// # Arguments
/// - `q`: Query matrix, row-major [seq_len, head_dim], stored as f32
/// - `k`: Key matrix, row-major [seq_len, head_dim], stored as f32
/// - `v`: Value matrix, row-major [seq_len, head_dim], stored as f32
/// - `seq_len`: Number of tokens (must be divisible by chunk_size)
/// - `head_dim`: Dimension of each head
/// - `chunk_size`: Number of tokens per chunk
///
/// # Returns
/// Output matrix [seq_len, head_dim] as Vec<f32>
///
/// # Panics
/// Panics if seq_len is not divisible by chunk_size, or if input lengths don't match.
pub fn cpu_linear_attention_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    chunk_size: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");
    assert_eq!(k.len(), seq_len * head_dim, "K length mismatch");
    assert_eq!(v.len(), seq_len * head_dim, "V length mismatch");
    assert!(
        seq_len.is_multiple_of(chunk_size),
        "seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
    );
    assert!(chunk_size > 0, "chunk_size must be > 0");
    assert!(head_dim > 0, "head_dim must be > 0");

    let num_chunks = seq_len / chunk_size;

    // H: D x D hidden state matrix, accumulated across chunks
    let mut h = vec![0.0f64; head_dim * head_dim];

    // Output: seq_len x head_dim
    let mut output = vec![0.0f64; seq_len * head_dim];

    for c in 0..num_chunks {
        let start = c * chunk_size;

        // Update H: H += sum_{t in chunk} K[t]^T * V[t] (outer product accumulation)
        // For each token t in the chunk, accumulate the outer product of K[t] and V[t]
        // H[i][j] += K[t][i] * V[t][j]
        for t in 0..chunk_size {
            let token_idx = start + t;
            for i in 0..head_dim {
                let k_val = k[token_idx * head_dim + i] as f64;
                for j in 0..head_dim {
                    let v_val = v[token_idx * head_dim + j] as f64;
                    h[i * head_dim + j] += k_val * v_val;
                }
            }
        }

        // Compute output: O_chunk = Q_chunk * H
        // For each token t in the chunk: O[t][j] = sum_i Q[t][i] * H[i][j]
        for t in 0..chunk_size {
            let token_idx = start + t;
            for j in 0..head_dim {
                let mut sum = 0.0f64;
                for i in 0..head_dim {
                    sum += q[token_idx * head_dim + i] as f64 * h[i * head_dim + j];
                }
                output[token_idx * head_dim + j] = sum;
            }
        }
    }

    // Truncate FP64 accumulation to FP32 output
    output.iter().map(|&x| x as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test with small inputs: N=16, D=4, chunk_size=4.
    /// Verify output is not all zeros (i.e., the recurrence actually computes something).
    #[test]
    fn test_linear_attention_small() {
        let seq_len = 16;
        let head_dim = 4;
        let chunk_size = 4;

        // Deterministic data using simple formula
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.1).sin() * 2.0)
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.2 + 1.0).cos() * 2.0)
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.3 + 2.0).sin() * 2.0)
            .collect();

        let output = cpu_linear_attention_f64(&q, &k, &v, seq_len, head_dim, chunk_size);

        // Verify output dimensions
        assert_eq!(output.len(), seq_len * head_dim);

        // Verify output is not all zeros — the recurrence should produce nonzero values
        let all_zero = output.iter().all(|&x| x == 0.0);
        assert!(!all_zero, "Output should not be all zeros");

        // Verify no NaN or Inf in output
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Output[{i}] is not finite: {val}"
            );
        }

        // Verify later chunks produce different values than earlier chunks
        // (because H accumulates across chunks)
        let first_chunk_out = &output[0..chunk_size * head_dim];
        let last_chunk_out =
            &output[(seq_len - chunk_size) * head_dim..seq_len * head_dim];
        let same = first_chunk_out
            .iter()
            .zip(last_chunk_out.iter())
            .all(|(&a, &b)| (a - b).abs() < 1e-10);
        assert!(
            !same,
            "First and last chunk outputs should differ (H accumulates)"
        );
    }

    /// Test with identity-like inputs to verify output pattern.
    ///
    /// When Q = I (identity rows), K = I (identity rows), V = known values:
    /// - After chunk 0 (tokens 0..chunk_size): H = sum of e_t * V[t]^T outer products
    ///   For identity K rows, K[t] = e_t, so H[t][j] = V[t][j] for t < D
    /// - O[t] = Q[t] * H = e_t * H = H[t] (the t-th row of H)
    ///
    /// With Q=I, K=I, chunk_size=D: after first chunk, H = diag-ish matrix,
    /// output row t = H[t] = V[t] for t < D.
    #[test]
    fn test_linear_attention_identity() {
        let head_dim = 4;
        let chunk_size = 4;
        let seq_len = 4; // Single chunk, same as head_dim

        // Q = identity matrix rows (e_0, e_1, e_2, e_3)
        let mut q = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            q[i * head_dim + i] = 1.0;
        }

        // K = identity matrix rows (same as Q)
        let mut k = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            k[i * head_dim + i] = 1.0;
        }

        // V = known values
        let v: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // V[0]
            5.0, 6.0, 7.0, 8.0, // V[1]
            9.0, 10.0, 11.0, 12.0, // V[2]
            13.0, 14.0, 15.0, 16.0, // V[3]
        ];

        let output = cpu_linear_attention_f64(&q, &k, &v, seq_len, head_dim, chunk_size);

        // With Q=I, K=I, single chunk:
        // H[i][j] = sum_t K[t][i] * V[t][j] = V[i][j] (since K[t][i] = delta(t,i))
        // O[t][j] = sum_i Q[t][i] * H[i][j] = H[t][j] = V[t][j]
        //
        // So output should equal V exactly.
        for i in 0..seq_len {
            for j in 0..head_dim {
                let idx = i * head_dim + j;
                let expected = v[idx];
                let got = output[idx];
                assert!(
                    (got - expected).abs() < 1e-6,
                    "output[{i}][{j}] = {got}, expected {expected}"
                );
            }
        }
    }

    /// Test that chunk accumulation works correctly across multiple chunks.
    ///
    /// With two chunks and Q=I, K=I:
    /// After chunk 0: H = diag(V[0..D])
    /// After chunk 1: H = diag(V[0..D]) + diag(V[D..2D])
    /// Output for chunk 1 tokens uses the accumulated H.
    #[test]
    fn test_linear_attention_accumulation() {
        let head_dim = 2;
        let chunk_size = 2;
        let seq_len = 4; // Two chunks

        // Q = identity rows repeated: [e0, e1, e0, e1]
        let q: Vec<f32> = vec![
            1.0, 0.0, // token 0: e0
            0.0, 1.0, // token 1: e1
            1.0, 0.0, // token 2: e0
            0.0, 1.0, // token 3: e1
        ];

        // K = identity rows repeated: [e0, e1, e0, e1]
        let k: Vec<f32> = vec![
            1.0, 0.0, // token 0: e0
            0.0, 1.0, // token 1: e1
            1.0, 0.0, // token 2: e0
            0.0, 1.0, // token 3: e1
        ];

        // V = known values
        let v: Vec<f32> = vec![
            10.0, 20.0, // V[0]
            30.0, 40.0, // V[1]
            50.0, 60.0, // V[2]
            70.0, 80.0, // V[3]
        ];

        let output = cpu_linear_attention_f64(&q, &k, &v, seq_len, head_dim, chunk_size);

        // Chunk 0 (tokens 0,1):
        //   H after chunk 0:
        //     H[0][0] += K[0][0]*V[0][0] = 1*10 = 10
        //     H[0][1] += K[0][0]*V[0][1] = 1*20 = 20
        //     H[1][0] += K[1][1]*V[1][0] = 1*30 = 30
        //     H[1][1] += K[1][1]*V[1][1] = 1*40 = 40
        //   H = [[10, 20], [30, 40]]
        //
        //   O[0] = Q[0] * H = e0 * H = [10, 20]
        //   O[1] = Q[1] * H = e1 * H = [30, 40]
        assert!((output[0] - 10.0).abs() < 1e-6, "O[0][0]={}", output[0]);
        assert!((output[1] - 20.0).abs() < 1e-6, "O[0][1]={}", output[1]);
        assert!((output[2] - 30.0).abs() < 1e-6, "O[1][0]={}", output[2]);
        assert!((output[3] - 40.0).abs() < 1e-6, "O[1][1]={}", output[3]);

        // Chunk 1 (tokens 2,3):
        //   H after chunk 1:
        //     H[0][0] += K[2][0]*V[2][0] = 1*50 = 10+50 = 60
        //     H[0][1] += K[2][0]*V[2][1] = 1*60 = 20+60 = 80
        //     H[1][0] += K[3][1]*V[3][0] = 1*70 = 30+70 = 100
        //     H[1][1] += K[3][1]*V[3][1] = 1*80 = 40+80 = 120
        //   H = [[60, 80], [100, 120]]
        //
        //   O[2] = Q[2] * H = e0 * H = [60, 80]
        //   O[3] = Q[3] * H = e1 * H = [100, 120]
        assert!((output[4] - 60.0).abs() < 1e-6, "O[2][0]={}", output[4]);
        assert!((output[5] - 80.0).abs() < 1e-6, "O[2][1]={}", output[5]);
        assert!((output[6] - 100.0).abs() < 1e-6, "O[3][0]={}", output[6]);
        assert!((output[7] - 120.0).abs() < 1e-6, "O[3][1]={}", output[7]);
    }

    /// Test that seq_len not divisible by chunk_size panics.
    #[test]
    #[should_panic(expected = "must be divisible by chunk_size")]
    fn test_linear_attention_bad_chunk_size() {
        let q = vec![1.0f32; 15]; // seq_len=5, head_dim=3
        let k = vec![1.0f32; 15];
        let v = vec![1.0f32; 15];
        cpu_linear_attention_f64(&q, &k, &v, 5, 3, 2); // 5 % 2 != 0
    }
}
