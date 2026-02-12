//! Integration tests for attention-proto prototypes.
//!
//! Tests run with --test-threads=1 to avoid Metal device contention.
//! Enable MTL_SHADER_VALIDATION=1 for GPU-side validation.

use attention_proto::device::GpuDevice;
use attention_proto::proto1_flash::{assert_allclose, cpu_attention_f64, run_flash_attention};
use attention_proto::proto3_paged::{create_paged_kv_cache, run_paged_attention};
use attention_proto::proto6_fla::{cpu_linear_attention_f64, run_linear_attention};
use attention_proto::proto7_variants::{
    cpu_attention_alibi_f64, cpu_attention_no_alibi_f64, cpu_gqa_remap, cpu_rope,
    run_flash_attention_with_alibi, run_gqa_remap_gpu, run_rope_gpu,
};

/// Generate deterministic pseudo-random f32 values using a simple LCG.
fn random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            // Simple LCG: state = (state * 6364136223846793005 + 1) mod 2^64
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1, 1] range for reasonable attention values
            let bits = (state >> 33) as u32; // top 31 bits
            (bits as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0
        })
        .collect()
}

mod proto1 {
    use super::*;

    /// Correctness test: GPU flash attention vs CPU FP64 reference.
    ///
    /// N=256, D=64, single head. Tolerances account for FP32 vs FP64 differences
    /// and tiled computation order differences in the flash attention algorithm.
    #[test]
    fn test_flash_correctness() {
        let device = GpuDevice::shared();
        let seq_len = 256;
        let head_dim = 64;

        // Generate deterministic random Q, K, V
        let q = random_f32(seq_len * head_dim, 42);
        let k = random_f32(seq_len * head_dim, 137);
        let v = random_f32(seq_len * head_dim, 999);

        // Run GPU flash attention
        let gpu_result = run_flash_attention(device, &q, &k, &v, seq_len, head_dim);

        // Run CPU FP64 reference
        let cpu_result = cpu_attention_f64(&q, &k, &v, seq_len, head_dim);

        // Verify dimensions
        assert_eq!(gpu_result.len(), seq_len * head_dim);
        assert_eq!(cpu_result.len(), seq_len * head_dim);

        // Verify no NaN or Inf in GPU output
        for (i, &val) in gpu_result.iter().enumerate() {
            assert!(
                val.is_finite(),
                "GPU output[{i}] is not finite: {val}"
            );
        }

        // Compare GPU vs CPU with tolerance for FP32 computation
        // atol=5e-3, rtol=1e-2 accounts for:
        // - FP32 vs FP64 accumulation differences
        // - Online softmax vs naive softmax numerical differences
        // - Tiled computation reordering
        assert_allclose(&gpu_result, &cpu_result, 5e-3, 1e-2, "flash_attention N=256 D=64");
    }
}

mod proto3 {
    use super::*;

    /// Correctness test: paged attention vs CPU FP64 reference.
    ///
    /// N=64, D=64, page_size=16, num_partitions=1 (simplest case — no reduce needed
    /// but reduce kernel still runs with 1 partition for code coverage).
    ///
    /// Tolerances: atol=1e-3, rtol=1e-2 (FP32 online softmax with paging overhead).
    /// Paged attention uses scalar dot products (not simdgroup_matrix) so may have
    /// slightly different rounding from Proto 1.
    #[test]
    fn test_paged_correctness() {
        let device = GpuDevice::shared();
        let seq_len = 64;
        let head_dim = 64;
        let page_size = 16;
        let num_partitions = 1;

        // Generate deterministic random Q, K, V
        let q = random_f32(seq_len * head_dim, 42);
        let k = random_f32(seq_len * head_dim, 137);
        let v = random_f32(seq_len * head_dim, 999);

        // Build paged KV cache (fragmented physical page order)
        let cache = create_paged_kv_cache(&k, &v, seq_len, head_dim, page_size);

        // Run paged attention on GPU
        let gpu_result = run_paged_attention(device, &q, &cache, seq_len, num_partitions);

        // Run CPU FP64 reference (contiguous, no paging)
        let cpu_result = cpu_attention_f64(&q, &k, &v, seq_len, head_dim);

        // Verify dimensions
        assert_eq!(gpu_result.len(), seq_len * head_dim);
        assert_eq!(cpu_result.len(), seq_len * head_dim);

        // Verify no NaN or Inf in GPU output
        for (i, &val) in gpu_result.iter().enumerate() {
            assert!(
                val.is_finite(),
                "GPU output[{i}] is not finite: {val}"
            );
        }

        // Compare GPU paged attention vs CPU reference
        assert_allclose(
            &gpu_result,
            &cpu_result,
            1e-3,
            1e-2,
            "paged_attention N=64 D=64 pages=16 parts=1",
        );
    }

    /// Threadgroup memory budget test for PagedAttention V2.
    ///
    /// Verifies threadgroup memory usage at varying page sizes (8/16/32/64/128 tokens)
    /// against the 32KB Metal threadgroup memory limit.
    ///
    /// Threadgroup memory layout per page_size (BLOCK_R=16, head_dim=64):
    ///   Q_tile  = BLOCK_R * head_dim * 4 bytes  (query tile, fixed)
    ///   K_page  = page_size * head_dim * 4 bytes (key page)
    ///   V_page  = page_size * head_dim * 4 bytes (value page)
    ///   S_buf   = BLOCK_R * page_size * 4 bytes  (score buffer)
    ///   Total   = Q_tile + K_page + V_page + S_buf
    ///
    /// Expected results for D=64:
    ///   page_size=8:   Q(4KB) + K(2KB) + V(2KB) + S(0.5KB) =  8.5KB  <= 32KB
    ///   page_size=16:  Q(4KB) + K(4KB) + V(4KB) + S(1KB)   = 13.0KB  <= 32KB
    ///   page_size=32:  Q(4KB) + K(8KB) + V(8KB) + S(2KB)   = 22.0KB  <= 32KB
    ///   page_size=64:  Q(4KB) + K(16KB) + V(16KB) + S(4KB) = 40.0KB  > 32KB!
    ///   page_size=128: Q(4KB) + K(32KB) + V(32KB) + S(8KB) = 76.0KB  > 32KB!
    #[test]
    fn test_threadgroup_budget() {
        const BLOCK_R: usize = 16;
        const HEAD_DIM: usize = 64;
        const MAX_THREADGROUP_MEMORY: usize = 32 * 1024; // 32KB

        let page_sizes: &[usize] = &[8, 16, 32, 64, 128];

        println!();
        println!("PagedAttention V2 Threadgroup Memory Budget (BLOCK_R={BLOCK_R}, D={HEAD_DIM})");
        println!("{:-<72}", "");
        println!(
            "{:>10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>6}",
            "page_size", "Q_tile", "K_page", "V_page", "S_buf", "Total", "Fits?"
        );
        println!("{:-<72}", "");

        let mut exceeds_32kb = Vec::new();

        for &page_size in page_sizes {
            let q_tile = BLOCK_R * HEAD_DIM * 4;
            let k_page = page_size * HEAD_DIM * 4;
            let v_page = page_size * HEAD_DIM * 4;
            let s_buf = BLOCK_R * page_size * 4;
            let total = q_tile + k_page + v_page + s_buf;

            let fits = total <= MAX_THREADGROUP_MEMORY;
            let fits_str = if fits { "YES" } else { "NO" };

            println!(
                "{:>10} {:>7.1}KB {:>7.1}KB {:>7.1}KB {:>7.1}KB {:>9.1}KB {:>6}",
                page_size,
                q_tile as f64 / 1024.0,
                k_page as f64 / 1024.0,
                v_page as f64 / 1024.0,
                s_buf as f64 / 1024.0,
                total as f64 / 1024.0,
                fits_str,
            );

            if !fits {
                exceeds_32kb.push((page_size, total));
            }
        }

        println!("{:-<72}", "");

        // Document which page sizes exceed 32KB
        if !exceeds_32kb.is_empty() {
            println!("Page sizes exceeding 32KB threadgroup limit:");
            for (ps, total) in &exceeds_32kb {
                println!(
                    "  page_size={ps}: {:.1}KB ({:.1}KB over limit)",
                    *total as f64 / 1024.0,
                    (*total as f64 - MAX_THREADGROUP_MEMORY as f64) / 1024.0,
                );
            }
        }

        // Assert: our chosen config (page_size=16) fits within 32KB
        let chosen_page_size = 16;
        let chosen_total = BLOCK_R * HEAD_DIM * 4
            + chosen_page_size * HEAD_DIM * 4
            + chosen_page_size * HEAD_DIM * 4
            + BLOCK_R * chosen_page_size * 4;
        assert!(
            chosen_total <= MAX_THREADGROUP_MEMORY,
            "Chosen config page_size={chosen_page_size} uses {}KB, exceeds 32KB limit!",
            chosen_total as f64 / 1024.0,
        );
        println!(
            "\nChosen config (page_size={chosen_page_size}): {:.1}KB -- WITHIN 32KB budget",
            chosen_total as f64 / 1024.0,
        );

        // Assert: page_size=32 also fits (max viable page size for D=64)
        let ps32_total = BLOCK_R * HEAD_DIM * 4
            + 32 * HEAD_DIM * 4
            + 32 * HEAD_DIM * 4
            + BLOCK_R * 32 * 4;
        assert!(
            ps32_total <= MAX_THREADGROUP_MEMORY,
            "page_size=32 uses {}KB, exceeds 32KB limit!",
            ps32_total as f64 / 1024.0,
        );

        // Assert: page_size=64 does NOT fit (documents the boundary)
        let ps64_total = BLOCK_R * HEAD_DIM * 4
            + 64 * HEAD_DIM * 4
            + 64 * HEAD_DIM * 4
            + BLOCK_R * 64 * 4;
        assert!(
            ps64_total > MAX_THREADGROUP_MEMORY,
            "Expected page_size=64 to exceed 32KB but it uses only {}KB",
            ps64_total as f64 / 1024.0,
        );

        // Verify exact byte counts match expectations
        assert_eq!(chosen_total, 13 * 1024, "page_size=16 should be exactly 13KB");
        assert_eq!(ps32_total, 22 * 1024, "page_size=32 should be exactly 22KB");

        println!("\nMax viable page_size for D={HEAD_DIM} within 32KB: 32 tokens");
        println!("page_size >= 64 requires either smaller BLOCK_R or reduced D");
    }
}

mod proto6 {
    use super::*;

    /// Correctness test: GPU linear attention vs CPU FP64 reference.
    ///
    /// N=128, D=64, chunk_size=32. Tolerances account for FP32 GPU accumulation
    /// vs FP64 CPU reference. Linear attention does not use softmax so the main
    /// error source is FP32 dot product accumulation over D=64 dimensions and
    /// chunk_size=32 outer product accumulations per chunk.
    ///
    /// atol=1e-3, rtol=1e-2: relaxed because values accumulate across chunks,
    /// and FP32 rounding compounds with larger H matrix elements.
    #[test]
    fn test_fla_correctness() {
        let device = GpuDevice::shared();
        let seq_len = 128;
        let head_dim = 64;
        let chunk_size = 32;

        // Generate deterministic random Q, K, V
        // Use small magnitude to keep accumulated values reasonable
        let q = random_f32(seq_len * head_dim, 42);
        let k = random_f32(seq_len * head_dim, 137);
        let v = random_f32(seq_len * head_dim, 999);

        // Run GPU linear attention (two-pass: chunk_h then chunk_o)
        let gpu_result =
            run_linear_attention(device, &q, &k, &v, seq_len, head_dim, chunk_size);

        // Run CPU FP64 reference
        let cpu_result =
            cpu_linear_attention_f64(&q, &k, &v, seq_len, head_dim, chunk_size);

        // Verify dimensions
        assert_eq!(gpu_result.len(), seq_len * head_dim);
        assert_eq!(cpu_result.len(), seq_len * head_dim);

        // Verify no NaN or Inf in GPU output
        for (i, &val) in gpu_result.iter().enumerate() {
            assert!(
                val.is_finite(),
                "GPU output[{i}] is not finite: {val}"
            );
        }

        // Compare GPU vs CPU
        assert_allclose(
            &gpu_result,
            &cpu_result,
            1e-3,
            1e-2,
            "fla_linear_attention N=128 D=64 chunk=32",
        );
    }
}

mod proto7 {
    use super::*;

    /// Correctness test: GPU RoPE vs CPU reference.
    ///
    /// N=64, D=64. Compares GPU apply_rope kernel output vs CPU RoPE reference.
    /// Tolerances: atol=1e-4, rtol=1e-3 (RoPE is element-wise, FP32 trig should match well).
    #[test]
    fn test_rope_correctness() {
        let device = GpuDevice::shared();
        let seq_len = 64;
        let head_dim = 64;

        let q = random_f32(seq_len * head_dim, 42);
        let k = random_f32(seq_len * head_dim, 137);

        // GPU RoPE
        let (gpu_q, gpu_k) = run_rope_gpu(device, &q, &k, seq_len, head_dim);

        // CPU RoPE reference
        let mut cpu_q = q.clone();
        let mut cpu_k = k.clone();
        cpu_rope(&mut cpu_q, &mut cpu_k, seq_len, head_dim);

        // Verify dimensions
        assert_eq!(gpu_q.len(), seq_len * head_dim);
        assert_eq!(gpu_k.len(), seq_len * head_dim);

        // Verify no NaN or Inf in GPU output
        for (i, &val) in gpu_q.iter().chain(gpu_k.iter()).enumerate() {
            assert!(val.is_finite(), "GPU RoPE output[{i}] is not finite: {val}");
        }

        // Compare GPU vs CPU
        assert_allclose(&gpu_q, &cpu_q, 1e-4, 1e-3, "rope_q N=64 D=64");
        assert_allclose(&gpu_k, &cpu_k, 1e-4, 1e-3, "rope_k N=64 D=64");
    }

    /// Correctness test: ALiBi attention vs CPU reference.
    ///
    /// Verifies that:
    /// 1. ALiBi-enabled output differs from vanilla attention (bias is applied)
    /// 2. ALiBi GPU output matches CPU ALiBi reference within tolerance
    ///
    /// N=64, D=64, num_heads=4.
    /// Uses smaller N to keep flash attention within reasonable tolerance.
    #[test]
    fn test_alibi_correctness() {
        let device = GpuDevice::shared();
        let seq_len = 64;
        let head_dim = 64;
        let num_heads = 4;
        let total = num_heads * seq_len * head_dim;

        let q = random_f32(total, 42);
        let k = random_f32(total, 137);
        let v = random_f32(total, 999);

        // GPU: run with ALiBi enabled
        let gpu_alibi =
            run_flash_attention_with_alibi(device, &q, &k, &v, seq_len, head_dim, num_heads, true);

        // GPU: run without ALiBi
        let gpu_vanilla = run_flash_attention_with_alibi(
            device, &q, &k, &v, seq_len, head_dim, num_heads, false,
        );

        // CPU: ALiBi reference
        let cpu_alibi =
            cpu_attention_alibi_f64(&q, &k, &v, seq_len, head_dim, num_heads);

        // CPU: vanilla reference
        let cpu_vanilla =
            cpu_attention_no_alibi_f64(&q, &k, &v, seq_len, head_dim, num_heads);

        // Verify dimensions
        assert_eq!(gpu_alibi.len(), total);
        assert_eq!(gpu_vanilla.len(), total);

        // Verify no NaN or Inf
        for (i, &val) in gpu_alibi.iter().enumerate() {
            assert!(
                val.is_finite(),
                "GPU ALiBi output[{i}] is not finite: {val}"
            );
        }

        // 1. ALiBi output must differ from vanilla
        let mut diff_count = 0;
        let mut max_diff = 0.0f32;
        for (a, v) in gpu_alibi.iter().zip(gpu_vanilla.iter()) {
            let d = (a - v).abs();
            if d > 1e-6 {
                diff_count += 1;
            }
            max_diff = max_diff.max(d);
        }
        assert!(
            diff_count > total / 2,
            "ALiBi should produce different output than vanilla. Only {diff_count}/{total} elements differ (max_diff={max_diff:.6})"
        );
        eprintln!(
            "[ALiBi] {diff_count}/{total} elements differ from vanilla (max_diff={max_diff:.4})"
        );

        // 2. GPU ALiBi should match CPU ALiBi reference
        // Flash attention uses tiled computation with online softmax, so tolerances
        // are wider than element-wise ops like RoPE.
        assert_allclose(
            &gpu_alibi,
            &cpu_alibi,
            5e-3,
            1e-2,
            "alibi_attention N=64 D=64 H=4",
        );

        // 3. GPU vanilla should match CPU vanilla reference
        assert_allclose(
            &gpu_vanilla,
            &cpu_vanilla,
            5e-3,
            1e-2,
            "vanilla_attention N=64 D=64 H=4",
        );
    }

    /// Correctness test: GPU GQA head remapping vs CPU reference.
    ///
    /// N=32, num_heads=8, num_kv_heads=2 (group_size=4).
    /// Each of the 2 KV heads should be replicated to 4 Q heads.
    /// Tolerances: exact match expected (pure copy, no arithmetic).
    #[test]
    fn test_gqa_correctness() {
        let device = GpuDevice::shared();
        let seq_len = 32;
        let head_dim = 64;
        let num_heads = 8;
        let num_kv_heads = 2;

        let k_full = random_f32(num_kv_heads * seq_len * head_dim, 42);

        // GPU GQA remap
        let gpu_expanded =
            run_gqa_remap_gpu(device, &k_full, seq_len, head_dim, num_heads, num_kv_heads);

        // CPU GQA remap reference
        let cpu_expanded = cpu_gqa_remap(&k_full, seq_len, head_dim, num_heads, num_kv_heads);

        // Verify dimensions
        assert_eq!(gpu_expanded.len(), num_heads * seq_len * head_dim);
        assert_eq!(cpu_expanded.len(), num_heads * seq_len * head_dim);

        // Verify no NaN or Inf
        for (i, &val) in gpu_expanded.iter().enumerate() {
            assert!(
                val.is_finite(),
                "GPU GQA output[{i}] is not finite: {val}"
            );
        }

        // GQA remap is a pure copy — should be exactly equal (atol=0, rtol=0)
        // Use small tolerance for floating point buffer roundtrip
        assert_allclose(
            &gpu_expanded,
            &cpu_expanded,
            1e-6,
            1e-6,
            "gqa_remap N=32 D=64 H=8 KV=2",
        );

        // Additionally verify the group structure:
        // Q heads 0-3 should all match KV head 0
        // Q heads 4-7 should all match KV head 1
        let group_size = num_heads / num_kv_heads;
        for kv_head in 0..num_kv_heads {
            let kv_offset = kv_head * seq_len * head_dim;
            for g in 0..group_size {
                let q_head = kv_head * group_size + g;
                let q_offset = q_head * seq_len * head_dim;
                for idx in 0..seq_len * head_dim {
                    assert_eq!(
                        gpu_expanded[q_offset + idx],
                        k_full[kv_offset + idx],
                        "GQA group mismatch: q_head={q_head} kv_head={kv_head} idx={idx}"
                    );
                }
            }
        }
    }
}
