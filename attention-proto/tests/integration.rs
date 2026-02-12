//! Integration tests for attention-proto prototypes.
//!
//! Tests run with --test-threads=1 to avoid Metal device contention.
//! Enable MTL_SHADER_VALIDATION=1 for GPU-side validation.

use attention_proto::device::GpuDevice;
use attention_proto::proto1_flash::{assert_allclose, cpu_attention_f64, run_flash_attention};

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
