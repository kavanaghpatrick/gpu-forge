//! AttentionParams — shared #[repr(C)] type matching shaders/types.h.
//!
//! This struct is passed directly to Metal compute kernels via setBytes or buffer,
//! so its layout must exactly match the MSL `struct AttentionParams` in types.h.

/// Attention kernel parameters shared between Rust host and Metal shaders.
///
/// Layout: 16 x u32/f32 fields = 64 bytes, 4-byte aligned.
/// Must be kept in sync with `shaders/types.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AttentionParams {
    /// Sequence length N
    pub seq_len: u32,
    /// Head dimension D
    pub head_dim: u32,
    /// Number of Q heads
    pub num_heads: u32,
    /// Number of KV heads (== num_heads for MHA, < for GQA)
    pub num_kv_heads: u32,
    /// Tile rows (queries)
    pub block_r: u32,
    /// Tile columns (keys)
    pub block_c: u32,
    /// Scaling factor: 1/sqrt(D)
    pub scale: f32,
    /// Variant selector: 0=standard, 1=RoPE, 2=ALiBi, 3=GQA
    pub variant: u32,
    /// Tokens per page (paged attention)
    pub page_size: u32,
    /// Total pages allocated (paged attention)
    pub num_pages: u32,
    /// Maximum context length (paged attention)
    pub max_context_len: u32,
    /// Partitioned reduce count (paged attention)
    pub num_partitions: u32,
    /// Explicit padding to reach 64 bytes
    pub _pad0: u32,
    /// Explicit padding
    pub _pad1: u32,
    /// Explicit padding
    pub _pad2: u32,
    /// Explicit padding
    pub _pad3: u32,
}

impl Default for AttentionParams {
    fn default() -> Self {
        Self {
            seq_len: 256,
            head_dim: 64,
            num_heads: 1,
            num_kv_heads: 1,
            block_r: 16,
            block_c: 64,
            scale: 1.0 / (64.0_f32).sqrt(),
            variant: 0,
            page_size: 16,
            num_pages: 0,
            max_context_len: 0,
            num_partitions: 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        }
    }
}

impl AttentionParams {
    /// Create params for standard (non-paged) flash attention.
    pub fn flash(seq_len: u32, head_dim: u32, num_heads: u32) -> Self {
        Self {
            seq_len,
            head_dim,
            num_heads,
            num_kv_heads: num_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
            ..Default::default()
        }
    }

    /// Create params for GQA (grouped-query attention).
    pub fn gqa(seq_len: u32, head_dim: u32, num_heads: u32, num_kv_heads: u32) -> Self {
        Self {
            seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
            variant: 3,
            ..Default::default()
        }
    }

    /// Create params for paged attention.
    pub fn paged(
        seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        page_size: u32,
        num_pages: u32,
        max_context_len: u32,
        num_partitions: u32,
    ) -> Self {
        Self {
            seq_len,
            head_dim,
            num_heads,
            num_kv_heads: num_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
            page_size,
            num_pages,
            max_context_len,
            num_partitions,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_attention_params_layout() {
        // Total size must be exactly 64 bytes
        assert_eq!(mem::size_of::<AttentionParams>(), 64);

        // Alignment must be 4 bytes (matching MSL uint/float alignment)
        assert_eq!(mem::align_of::<AttentionParams>(), 4);

        // Verify each field offset matches the MSL struct layout
        assert_eq!(mem::offset_of!(AttentionParams, seq_len), 0);
        assert_eq!(mem::offset_of!(AttentionParams, head_dim), 4);
        assert_eq!(mem::offset_of!(AttentionParams, num_heads), 8);
        assert_eq!(mem::offset_of!(AttentionParams, num_kv_heads), 12);
        assert_eq!(mem::offset_of!(AttentionParams, block_r), 16);
        assert_eq!(mem::offset_of!(AttentionParams, block_c), 20);
        assert_eq!(mem::offset_of!(AttentionParams, scale), 24);
        assert_eq!(mem::offset_of!(AttentionParams, variant), 28);
        assert_eq!(mem::offset_of!(AttentionParams, page_size), 32);
        assert_eq!(mem::offset_of!(AttentionParams, num_pages), 36);
        assert_eq!(mem::offset_of!(AttentionParams, max_context_len), 40);
        assert_eq!(mem::offset_of!(AttentionParams, num_partitions), 44);
        assert_eq!(mem::offset_of!(AttentionParams, _pad0), 48);
        assert_eq!(mem::offset_of!(AttentionParams, _pad1), 52);
        assert_eq!(mem::offset_of!(AttentionParams, _pad2), 56);
        assert_eq!(mem::offset_of!(AttentionParams, _pad3), 60);
    }

    #[test]
    fn test_default_params() {
        let params = AttentionParams::default();
        assert_eq!(params.seq_len, 256);
        assert_eq!(params.head_dim, 64);
        assert_eq!(params.num_heads, 1);
        assert_eq!(params.num_kv_heads, 1);
        assert_eq!(params.block_r, 16);
        assert_eq!(params.block_c, 64);
        assert!((params.scale - 0.125).abs() < 1e-6); // 1/sqrt(64) = 0.125
        assert_eq!(params.variant, 0);
        assert_eq!(params._pad0, 0);
        assert_eq!(params._pad1, 0);
        assert_eq!(params._pad2, 0);
        assert_eq!(params._pad3, 0);
    }

    #[test]
    fn test_flash_constructor() {
        let params = AttentionParams::flash(512, 128, 8);
        assert_eq!(params.seq_len, 512);
        assert_eq!(params.head_dim, 128);
        assert_eq!(params.num_heads, 8);
        assert_eq!(params.num_kv_heads, 8); // MHA: num_kv_heads == num_heads
        assert!((params.scale - 1.0 / (128.0_f32).sqrt()).abs() < 1e-6);
        assert_eq!(params.variant, 0);
    }

    #[test]
    fn test_gqa_constructor() {
        let params = AttentionParams::gqa(1024, 64, 32, 8);
        assert_eq!(params.num_heads, 32);
        assert_eq!(params.num_kv_heads, 8);
        assert_eq!(params.variant, 3);
    }

    #[test]
    fn test_paged_constructor() {
        let params = AttentionParams::paged(2048, 64, 16, 32, 128, 4096, 4);
        assert_eq!(params.page_size, 32);
        assert_eq!(params.num_pages, 128);
        assert_eq!(params.max_context_len, 4096);
        assert_eq!(params.num_partitions, 4);
    }
}
