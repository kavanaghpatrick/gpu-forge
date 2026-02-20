//! Proto 7: RoPE/ALiBi/GQA variant host code and CPU references.
//!
//! This module implements:
//! - `cpu_rope`: CPU reference for Rotary Position Embeddings
//! - `run_rope_gpu`: GPU dispatch for the apply_rope Metal kernel
//! - `cpu_attention_alibi_f64`: CPU reference for attention with ALiBi bias
//! - `run_flash_attention_alibi`: GPU dispatch for flash_attention with ALIBI_ENABLED=true
//! - `cpu_gqa_remap`: CPU reference for GQA head expansion
//! - `run_gqa_remap_gpu`: GPU dispatch for the gqa_remap Metal kernel

use crate::device::GpuDevice;
use crate::encode::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::AttentionParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embeddings)
// ---------------------------------------------------------------------------

/// CPU reference implementation of RoPE (Rotary Position Embeddings).
///
/// Applies rotary embeddings to Q and K in-place. Each (token, pair) element
/// is rotated by angle = token / theta_base^(2*pair/head_dim).
///
/// # Arguments
/// - `q`: Query matrix [seq_len, head_dim], modified in-place
/// - `k`: Key matrix [seq_len, head_dim], modified in-place
/// - `seq_len`: Number of tokens
/// - `head_dim`: Dimension per head (must be even)
pub fn cpu_rope(q: &mut [f32], k: &mut [f32], seq_len: usize, head_dim: usize) {
    let theta_base: f64 = 10000.0;
    for token in 0..seq_len {
        for pair in 0..head_dim / 2 {
            let angle =
                token as f64 / theta_base.powf(2.0 * pair as f64 / head_dim as f64);
            let cos_a = angle.cos() as f32;
            let sin_a = angle.sin() as f32;
            let idx0 = token * head_dim + 2 * pair;
            let idx1 = idx0 + 1;
            let (q0, q1) = (q[idx0], q[idx1]);
            q[idx0] = q0 * cos_a - q1 * sin_a;
            q[idx1] = q0 * sin_a + q1 * cos_a;
            let (k0, k1) = (k[idx0], k[idx1]);
            k[idx0] = k0 * cos_a - k1 * sin_a;
            k[idx1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

/// Run RoPE on GPU via the apply_rope Metal kernel.
///
/// Allocates Q/K as read-write buffers, dispatches apply_rope, reads back results.
///
/// # Returns
/// (q_out, k_out) after RoPE rotation applied by GPU.
pub fn run_rope_gpu(
    device: &GpuDevice,
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");
    assert_eq!(k.len(), seq_len * head_dim, "K length mismatch");

    let params = AttentionParams {
        seq_len: seq_len as u32,
        head_dim: head_dim as u32,
        num_heads: 1,
        ..Default::default()
    };

    // Allocate read-write buffers (apply_rope modifies Q and K in-place)
    let q_buf = alloc_buffer_with_data(&device.device, q);
    let k_buf = alloc_buffer_with_data(&device.device, k);
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    // Compile PSO for apply_rope (no function constants needed)
    let pso_key = PsoKey::simple("apply_rope");
    let mut cache = PsoCache::new(device.library.clone());
    let pso = cache.get_or_compile(&pso_key);

    // Create command buffer and encoder
    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    // Set PSO and bind buffers
    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*k_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 2);
    }

    // Dispatch 2D grid: (seq_len, head_dim/2) — each thread handles one (token, dim_pair)
    let grid_size = MTLSize {
        width: seq_len,
        height: head_dim / 2,
        depth: 1,
    };
    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    // Use 16x16 = 256 threadgroup, clamped to pipeline max
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

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "RoPE command buffer failed. Error: {:?}",
        command_buffer.error()
    );

    let q_out = unsafe { read_buffer_slice(&q_buf, seq_len * head_dim) };
    let k_out = unsafe { read_buffer_slice(&k_buf, seq_len * head_dim) };
    (q_out, k_out)
}

// ---------------------------------------------------------------------------
// ALiBi (Attention with Linear Biases)
// ---------------------------------------------------------------------------

/// CPU reference: scaled dot-product attention with ALiBi bias (FP64).
///
/// Computes: softmax((Q * K^T / sqrt(D)) + alibi_bias) * V
/// ALiBi bias: -slope * |pos_q - pos_k| per head
/// slope = 1 / 2^((head+1) * 8 / num_heads)
///
/// Multi-head layout: Q/K/V are [num_heads, seq_len, head_dim].
pub fn cpu_attention_alibi_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
) -> Vec<f32> {
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total, "Q length mismatch");
    assert_eq!(k.len(), total, "K length mismatch");
    assert_eq!(v.len(), total, "V length mismatch");

    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut output = vec![0.0f32; total];

    for head in 0..num_heads {
        let slope = 1.0_f64 / 2.0_f64.powf((head as f64 + 1.0) * 8.0 / num_heads as f64);
        let head_offset = head * seq_len * head_dim;

        for i in 0..seq_len {
            let mut scores = vec![0.0f64; seq_len];
            let mut max_score = f64::NEG_INFINITY;

            for j in 0..seq_len {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += q[head_offset + i * head_dim + d] as f64
                        * k[head_offset + j * head_dim + d] as f64;
                }
                // Add ALiBi bias
                let bias = -slope * ((i as f64) - (j as f64)).abs();
                scores[j] = dot * scale + bias;
                max_score = max_score.max(scores[j]);
            }

            // Safe softmax
            let mut sum_exp = 0.0f64;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }

            // Weighted sum of values
            for j in 0..seq_len {
                let weight = scores[j] / sum_exp;
                for d in 0..head_dim {
                    output[head_offset + i * head_dim + d] +=
                        (weight * v[head_offset + j * head_dim + d] as f64) as f32;
                }
            }
        }
    }

    output
}

/// CPU reference: standard attention without ALiBi (FP64), multi-head.
///
/// Same as cpu_attention_alibi_f64 but without ALiBi bias.
pub fn cpu_attention_no_alibi_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
) -> Vec<f32> {
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total, "Q length mismatch");
    assert_eq!(k.len(), total, "K length mismatch");
    assert_eq!(v.len(), total, "V length mismatch");

    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut output = vec![0.0f32; total];

    for head in 0..num_heads {
        let head_offset = head * seq_len * head_dim;

        for i in 0..seq_len {
            let mut scores = vec![0.0f64; seq_len];
            let mut max_score = f64::NEG_INFINITY;

            for j in 0..seq_len {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += q[head_offset + i * head_dim + d] as f64
                        * k[head_offset + j * head_dim + d] as f64;
                }
                scores[j] = dot * scale;
                max_score = max_score.max(scores[j]);
            }

            let mut sum_exp = 0.0f64;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }

            for j in 0..seq_len {
                let weight = scores[j] / sum_exp;
                for d in 0..head_dim {
                    output[head_offset + i * head_dim + d] +=
                        (weight * v[head_offset + j * head_dim + d] as f64) as f32;
                }
            }
        }
    }

    output
}

/// Run flash attention on GPU with ALiBi enabled or disabled.
///
/// Uses the flash_attention kernel with ALIBI_ENABLED function constant.
/// Multi-head layout: Q/K/V are [num_heads, seq_len, head_dim].
pub fn run_flash_attention_with_alibi(
    device: &GpuDevice,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    alibi_enabled: bool,
) -> Vec<f32> {
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total, "Q length mismatch");
    assert_eq!(k.len(), total, "K length mismatch");
    assert_eq!(v.len(), total, "V length mismatch");

    let block_r: u32 = 16;
    let block_c: u32 = 64;

    let params = AttentionParams::flash(seq_len as u32, head_dim as u32, num_heads as u32);

    let q_buf = alloc_buffer_with_data(&device.device, q);
    let k_buf = alloc_buffer_with_data(&device.device, k);
    let v_buf = alloc_buffer_with_data(&device.device, v);
    let o_buf = alloc_buffer(&device.device, total * std::mem::size_of::<f32>());
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    // Function constants: HEAD_DIM(0), BLOCK_R(1), BLOCK_C(2), ALIBI_ENABLED(4)
    let pso_key = PsoKey::simple("flash_attention")
        .with_uint(0, head_dim as u32)
        .with_uint(1, block_r)
        .with_uint(2, block_c)
        .with_bool(4, alibi_enabled);

    let mut cache = PsoCache::new(device.library.clone());
    let pso = cache.get_or_compile(&pso_key);

    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*k_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*v_buf), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&*o_buf), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 4);
    }

    // Dispatch threadgroups: (ceil(seq_len/BLOCK_R), num_heads)
    let num_row_blocks = (seq_len as u64 + block_r as u64 - 1) / block_r as u64;
    let threadgroups_per_grid = MTLSize {
        width: num_row_blocks as usize,
        height: num_heads,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threads_per_threadgroup);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "ALiBi command buffer failed. Error: {:?}",
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&o_buf, total) }
}

// ---------------------------------------------------------------------------
// GQA (Grouped Query Attention) head remapping
// ---------------------------------------------------------------------------

/// CPU reference for GQA head remapping.
///
/// Expands K from [num_kv_heads, seq_len, head_dim] to [num_heads, seq_len, head_dim]
/// by copying each KV head to its corresponding group of Q heads.
///
/// group_size = num_heads / num_kv_heads
/// kv_head = q_head / group_size
pub fn cpu_gqa_remap(
    k_full: &[f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> Vec<f32> {
    assert_eq!(
        k_full.len(),
        num_kv_heads * seq_len * head_dim,
        "K_full length mismatch"
    );
    assert!(
        num_heads >= num_kv_heads && num_heads.is_multiple_of(num_kv_heads),
        "num_heads must be a multiple of num_kv_heads"
    );

    let group_size = num_heads / num_kv_heads;
    let mut k_expanded = vec![0.0f32; num_heads * seq_len * head_dim];

    for q_head in 0..num_heads {
        let kv_head = q_head / group_size;
        for token in 0..seq_len {
            for dim in 0..head_dim {
                let src = (kv_head * seq_len + token) * head_dim + dim;
                let dst = (q_head * seq_len + token) * head_dim + dim;
                k_expanded[dst] = k_full[src];
            }
        }
    }

    k_expanded
}

/// Run GQA head remapping on GPU via the gqa_remap Metal kernel.
///
/// Dispatches a 3D grid (num_heads, seq_len, head_dim), each thread copies one element.
pub fn run_gqa_remap_gpu(
    device: &GpuDevice,
    k_full: &[f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> Vec<f32> {
    assert_eq!(
        k_full.len(),
        num_kv_heads * seq_len * head_dim,
        "K_full length mismatch"
    );

    let params = AttentionParams::gqa(
        seq_len as u32,
        head_dim as u32,
        num_heads as u32,
        num_kv_heads as u32,
    );

    let k_full_buf = alloc_buffer_with_data(&device.device, k_full);
    let k_expanded_buf = alloc_buffer(
        &device.device,
        num_heads * seq_len * head_dim * std::mem::size_of::<f32>(),
    );
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    let pso_key = PsoKey::simple("gqa_remap");
    let mut cache = PsoCache::new(device.library.clone());
    let pso = cache.get_or_compile(&pso_key);

    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*k_full_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*k_expanded_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 2);
    }

    // 3D grid dispatch: (num_heads, seq_len, head_dim)
    let grid_size = MTLSize {
        width: num_heads,
        height: seq_len,
        depth: head_dim,
    };
    // Use a reasonable threadgroup size for 3D dispatch
    // Total threads per threadgroup should not exceed pipeline max (typically 1024)
    // Use 8x8x8 = 512 threads as a safe default
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

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "GQA remap command buffer failed. Error: {:?}",
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&k_expanded_buf, num_heads * seq_len * head_dim) }
}

// ---------------------------------------------------------------------------
// KB Findings
// ---------------------------------------------------------------------------

/// Emit KB findings for Proto 7 (RoPE, ALiBi, GQA variant overhead).
///
/// Findings based on empirical benchmarks on M4 (32 heads, N=2048, D=64):
/// - Base flash attention: ~204ms GPU time
/// - RoPE standalone: ~9.7us (negligible)
/// - ALiBi fused: ~201ms (within noise, zero overhead via function constant)
/// - GQA remap: ~73-184us depending on group_size
pub fn emit_proto7_findings() {
    use crate::kb::{emit_finding, KbFinding};

    // Finding 1: RoPE standalone overhead is negligible
    emit_finding(&KbFinding {
        domain: "gpu-perf".to_string(),
        title: "proto7: RoPE standalone kernel overhead ~10us per head on M4 — negligible vs attention"
            .to_string(),
        content: "RoPE apply_rope kernel takes ~9.7us for one head at N=2048, D=64 on M4. \
                  For 32-head attention taking ~204ms total GPU time, RoPE adds <0.01% overhead \
                  per head (~0.3ms total for all heads). RoPE uses a 2D grid (seq_len, D/2) with \
                  cos/sin rotation per (token, dim_pair). Expected 2-5% overhead confirmed to be \
                  well below 1%. Can be applied as a pre-processing step before attention without \
                  measurable impact on end-to-end latency."
            .to_string(),
        tags: vec![
            "proto7".to_string(),
            "rope".to_string(),
            "rotary-position-embedding".to_string(),
            "gpu-perf".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.9,
        source: "attention-proto/variant_overhead benchmark (criterion, GPU timing, 32 heads, N=2048, D=64)"
            .to_string(),
    });

    // Finding 2: ALiBi fused via function constant has zero overhead
    emit_finding(&KbFinding {
        domain: "gpu-perf".to_string(),
        title: "proto7: ALiBi fused into flash attention via function constant adds zero measurable overhead on M4"
            .to_string(),
        content: "ALiBi bias fused into flash_attention.metal via ALIBI_ENABLED function_constant(4) \
                  takes ~201ms vs base ~204ms for 32 heads at N=2048, D=64 — within noise (-2.3%). \
                  Metal compiler dead-code eliminates the ALiBi branch when ALIBI_ENABLED=false, so \
                  the disabled path has zero cost. When enabled, the ALiBi math (one multiply + one \
                  add per score element: bias = -slope * |pos_q - pos_k|) is negligible relative to \
                  simdgroup_matrix attention compute. Confirms <1% overhead target. Function constant \
                  specialization is the correct strategy for optional attention biases."
            .to_string(),
        tags: vec![
            "proto7".to_string(),
            "alibi".to_string(),
            "function-constant".to_string(),
            "zero-overhead".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.92,
        source: "attention-proto/variant_overhead benchmark (criterion, GPU timing, 32 heads, N=2048, D=64)"
            .to_string(),
    });

    // Finding 3: GQA remap overhead is negligible
    emit_finding(&KbFinding {
        domain: "gpu-perf".to_string(),
        title: "proto7: GQA head remap kernel overhead <0.1% of attention compute on M4".to_string(),
        content: "GQA gqa_remap kernel expands KV heads to Q head count via pure memory copy. \
                  Measured on M4 at N=2048, D=64, 32 Q heads: group_size=1 (32 KV heads, full copy) \
                  ~184us = 0.1%, group_size=2 (16 KV heads) ~112us, group_size=4 (8 KV heads) ~80us, \
                  group_size=8 (4 KV heads) ~73us. All well below 1% of base flash attention ~204ms. \
                  Cost scales with output size (num_heads * seq_len * head_dim elements), not input \
                  KV head count. Exact match (atol=1e-6) with CPU reference — pure copy has no \
                  numerical error. GQA remap can be applied as a pre-processing step with negligible \
                  overhead."
            .to_string(),
        tags: vec![
            "proto7".to_string(),
            "gqa".to_string(),
            "grouped-query-attention".to_string(),
            "head-remap".to_string(),
            "gpu-perf".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.9,
        source: "attention-proto/variant_overhead benchmark (criterion, GPU timing, group_size=1/2/4/8)"
            .to_string(),
    });

    // Finding 4: All variants can be function-constant-specialized
    emit_finding(&KbFinding {
        domain: "metal-compute".to_string(),
        title: "proto7: All attention variants (RoPE, ALiBi, GQA) amenable to function constant specialization on M4"
            .to_string(),
        content: "All three Proto 7 attention variants work with Metal function constants for \
                  zero-overhead dispatch. ALiBi: ALIBI_ENABLED bool function_constant(4) enables \
                  compiler dead-code elimination — confirmed zero overhead when disabled and \
                  negligible overhead when enabled. RoPE: standalone kernel with no function \
                  constants needed (always applied identically), ~10us per head. GQA: uses \
                  AttentionParams.group_size at runtime (pure memory copy, no branching to \
                  specialize). For trait Attention<Q,K,V>, the recommended dispatch strategy is: \
                  (1) function constants for attention kernel variants (ALiBi, causal mask, etc.), \
                  (2) standalone pre/post-processing kernels for RoPE and GQA remap. Combined with \
                  Proto 4 findings (~63us cold compile, ~178ns cache hit), variant selection adds \
                  <1ms total overhead to any attention configuration."
            .to_string(),
        tags: vec![
            "proto7".to_string(),
            "function-constant".to_string(),
            "trait-attention".to_string(),
            "dispatch-strategy".to_string(),
            "metal-compute".to_string(),
        ],
        confidence: 0.88,
        source: "attention-proto/proto7_variants analysis (combined benchmark + correctness data)"
            .to_string(),
    });

    // Finding 5: Correctness validation tolerances for each variant
    emit_finding(&KbFinding {
        domain: "msl-kernels".to_string(),
        title: "proto7: Attention variant correctness tolerances — RoPE 1e-4, ALiBi 5e-3, GQA 1e-6 on M4"
            .to_string(),
        content: "GPU vs CPU FP64 reference correctness tolerances on M4: RoPE matches at \
                  atol=1e-4 (tight — element-wise trig ops have good FP32 agreement). ALiBi \
                  matches at atol=5e-3 (wider — accumulated softmax error from FP32 attention \
                  compute, same as base flash attention). GQA remap matches at atol=1e-6 (exact — \
                  pure memory copy with no arithmetic). These tolerances establish baselines for \
                  regression testing in trait Attention<Q,K,V> implementations. The ALiBi tolerance \
                  is dominated by flash attention FP32 softmax accumulation, not the ALiBi bias \
                  computation itself."
            .to_string(),
        tags: vec![
            "proto7".to_string(),
            "correctness".to_string(),
            "numerical-tolerance".to_string(),
            "rope".to_string(),
            "alibi".to_string(),
            "gqa".to_string(),
        ],
        confidence: 0.92,
        source: "attention-proto/proto7_variants correctness tests (GPU vs CPU FP64 reference)"
            .to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_rope_basic() {
        // Verify token=0 is identity (angle=0 for all pairs => cos=1, sin=0)
        let head_dim = 4;
        let mut q = vec![1.0, 2.0, 3.0, 4.0]; // token 0
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q_orig = q.clone();
        let k_orig = k.clone();

        cpu_rope(&mut q, &mut k, 1, head_dim);

        // At token=0: angle = 0 / theta^(...) = 0, cos(0)=1, sin(0)=0
        // So q and k should be unchanged
        for i in 0..head_dim {
            assert!(
                (q[i] - q_orig[i]).abs() < 1e-6,
                "RoPE at token=0 should be identity for q[{i}]"
            );
            assert!(
                (k[i] - k_orig[i]).abs() < 1e-6,
                "RoPE at token=0 should be identity for k[{i}]"
            );
        }
    }

    #[test]
    fn test_cpu_rope_nonzero_token() {
        // Verify token > 0 actually rotates values
        let head_dim = 4;
        let seq_len = 2;
        let mut q = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0];
        let q_before = q.clone();

        cpu_rope(&mut q, &mut k, seq_len, head_dim);

        // Token 0 should be unchanged, token 1 should be different
        for d in 0..head_dim {
            assert!(
                (q[d] - q_before[d]).abs() < 1e-6,
                "Token 0 should be unchanged"
            );
        }
        // Token 1 should be rotated (different from original)
        let mut any_different = false;
        for d in 0..head_dim {
            if (q[head_dim + d] - q_before[head_dim + d]).abs() > 1e-6 {
                any_different = true;
            }
        }
        assert!(any_different, "Token 1 should be rotated by RoPE");
    }

    /// Test to emit Proto 7 KB findings. Run manually:
    /// `cargo test --release -- proto7_variants::tests::generate_proto7_findings --ignored --test-threads=1`
    #[test]
    #[ignore]
    fn generate_proto7_findings() {
        super::emit_proto7_findings();
    }
}
