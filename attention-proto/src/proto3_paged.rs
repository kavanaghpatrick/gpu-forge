//! Proto 3: PagedAttention V2 — Host code for paged attention with page table indirection.
//!
//! This module implements:
//! - `PagedKVCache`: Page table simulation with fragmented physical page ordering
//! - `create_paged_kv_cache`: Splits K/V into pages and shuffles physical layout
//! - `run_paged_attention`: Two-pass GPU dispatch (partition then reduce)

use crate::device::GpuDevice;
use crate::encode::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::AttentionParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Simulated paged KV cache with page table indirection.
///
/// Pages are stored in a flat buffer with interleaved K and V per page:
/// `kv_data[page][0=K/1=V][token][dim]`
///
/// The `page_table` maps logical page indices to physical page indices,
/// simulating memory fragmentation in a real KV cache allocator.
pub struct PagedKVCache {
    pub page_size: usize,
    pub head_dim: usize,
    pub num_physical_pages: usize,
    /// Flat KV data: [num_pages, 2, page_size, head_dim] interleaved K and V
    pub kv_data: Vec<f32>,
    /// Logical page -> physical page mapping
    pub page_table: Vec<u32>,
}

/// Create a paged KV cache from contiguous K and V tensors.
///
/// Splits K and V into pages of `page_size` tokens each, then shuffles
/// the physical page order to simulate memory fragmentation.
///
/// # Arguments
/// - `k`: Key tensor [seq_len, head_dim]
/// - `v`: Value tensor [seq_len, head_dim]
/// - `seq_len`: Number of tokens
/// - `head_dim`: Dimension per head
/// - `page_size`: Tokens per page
///
/// # Returns
/// A `PagedKVCache` with shuffled physical pages and page table mapping
pub fn create_paged_kv_cache(
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    page_size: usize,
) -> PagedKVCache {
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);

    let num_pages = (seq_len + page_size - 1) / page_size;

    // Build a shuffled mapping: logical page -> physical page
    // Use a deterministic shuffle (reverse order) for reproducibility
    let mut page_table: Vec<u32> = (0..num_pages).map(|i| i as u32).collect();
    // Reverse to simulate fragmentation
    page_table.reverse();

    // Allocate KV data: [num_pages, 2, page_size, head_dim]
    let page_stride = 2 * page_size * head_dim; // floats per physical page (K + V)
    let mut kv_data = vec![0.0f32; num_pages * page_stride];

    // Fill pages: logical page `lp` maps to physical page `page_table[lp]`
    for lp in 0..num_pages {
        let pp = page_table[lp] as usize;
        let token_start = lp * page_size;
        let token_count = (seq_len - token_start).min(page_size);

        for t in 0..token_count {
            let src_row = token_start + t;
            for d in 0..head_dim {
                // K at offset 0 within page
                kv_data[pp * page_stride + t * head_dim + d] = k[src_row * head_dim + d];
                // V at offset page_size * head_dim within page
                kv_data[pp * page_stride + page_size * head_dim + t * head_dim + d] =
                    v[src_row * head_dim + d];
            }
        }
    }

    PagedKVCache {
        page_size,
        head_dim,
        num_physical_pages: num_pages,
        kv_data,
        page_table,
    }
}

/// Run paged attention on the GPU using the two-pass Metal kernels.
///
/// Pass 1: `paged_attention_partition` — each threadgroup processes one query block
///         and one partition of KV pages, producing partial O/m/l.
/// Pass 2: `paged_attention_reduce` — combines partial outputs across partitions
///         using log-sum-exp reduction.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `q`: Query tensor [seq_len, head_dim]
/// - `cache`: PagedKVCache with KV data and page table
/// - `seq_len`: Number of query tokens
/// - `num_partitions`: Number of partitions to split KV pages across
///
/// # Returns
/// Output tensor [seq_len, head_dim] as Vec<f32>
pub fn run_paged_attention(
    device: &GpuDevice,
    q: &[f32],
    cache: &PagedKVCache,
    seq_len: usize,
    num_partitions: usize,
) -> Vec<f32> {
    let head_dim = cache.head_dim;
    let page_size = cache.page_size;
    let num_pages = cache.num_physical_pages;

    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");

    let block_r: usize = 16; // TILE_Q in kernel
    let num_query_blocks = (seq_len + block_r - 1) / block_r;

    // Build AttentionParams
    let params = AttentionParams::paged(
        seq_len as u32,
        head_dim as u32,
        1, // single head
        page_size as u32,
        num_pages as u32,
        seq_len as u32, // max_context_len = seq_len (all tokens valid)
        num_partitions as u32,
    );

    // Allocate Metal buffers
    let q_buf = alloc_buffer_with_data(&device.device, q);
    let kv_buf = alloc_buffer_with_data(&device.device, &cache.kv_data);
    let pt_buf = alloc_buffer_with_data(&device.device, &cache.page_table);
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    // Partial output buffers for partition kernel
    // Layout: pb_idx = block_row * num_partitions + partition
    // Total pb entries = num_query_blocks * num_partitions
    let total_pb = num_query_blocks * num_partitions;
    let o_partial_size = total_pb * block_r * head_dim;
    let ml_partial_size = total_pb * block_r;

    let o_partial_buf =
        alloc_buffer(&device.device, o_partial_size * std::mem::size_of::<f32>());
    let m_partial_buf =
        alloc_buffer(&device.device, ml_partial_size * std::mem::size_of::<f32>());
    let l_partial_buf =
        alloc_buffer(&device.device, ml_partial_size * std::mem::size_of::<f32>());

    // Final output buffer
    let o_final_buf =
        alloc_buffer(&device.device, seq_len * head_dim * std::mem::size_of::<f32>());

    // Compile PSOs with function constants
    // paged_attention.metal: HEAD_DIM=index(0), PAGE_SIZE=index(1)
    // Compile each PSO with a separate PsoCache to avoid borrow checker issues
    let partition_key = PsoKey::simple("paged_attention_partition")
        .with_uint(0, head_dim as u32)
        .with_uint(1, page_size as u32);
    let mut partition_cache = PsoCache::new(device.library.clone());
    let partition_pso = partition_cache.get_or_compile(&partition_key);

    let reduce_key = PsoKey::simple("paged_attention_reduce")
        .with_uint(0, head_dim as u32)
        .with_uint(1, page_size as u32);
    let mut reduce_cache = PsoCache::new(device.library.clone());
    let reduce_pso = reduce_cache.get_or_compile(&reduce_key);

    // Create command buffer
    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");

    // ====================================================================
    // Pass 1: Partition kernel
    // Grid: (num_query_blocks, num_partitions)
    // Threadgroup: 32 threads (one simdgroup)
    // ====================================================================
    {
        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encoder.setComputePipelineState(partition_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*kv_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*pt_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*o_partial_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&*m_partial_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&*l_partial_buf), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 6);
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

    // ====================================================================
    // Pass 2: Reduce kernel
    // Grid: (num_query_blocks, 1)
    // Threadgroup: TILE_Q * TILE_D = 16 * 64 = 1024 threads
    // ====================================================================
    {
        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encoder.setComputePipelineState(reduce_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*o_partial_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*m_partial_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*l_partial_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*o_final_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 4);
        }

        let threadgroups = MTLSize {
            width: num_query_blocks,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: block_r * head_dim, // 16 * 64 = 1024
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        encoder.endEncoding();
    }

    // Commit and wait
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Check status
    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "Command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    // Read back final output
    unsafe { read_buffer_slice(&o_final_buf, seq_len * head_dim) }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Basic unit test: verify page table construction with a small example.
    #[test]
    fn test_paged_cache_construction() {
        let seq_len = 32;
        let head_dim = 4;
        let page_size = 16;

        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i + 1000) as f32)
            .collect();

        let cache = create_paged_kv_cache(&k, &v, seq_len, head_dim, page_size);

        // 32 tokens / 16 per page = 2 pages
        assert_eq!(cache.num_physical_pages, 2);
        assert_eq!(cache.page_table.len(), 2);

        // Reversed page table: logical 0 -> physical 1, logical 1 -> physical 0
        assert_eq!(cache.page_table[0], 1);
        assert_eq!(cache.page_table[1], 0);

        // Verify K data for logical page 0 (stored in physical page 1)
        let pp = cache.page_table[0] as usize;
        let page_stride = 2 * page_size * head_dim;
        for t in 0..page_size {
            for d in 0..head_dim {
                let expected = k[t * head_dim + d];
                let actual = cache.kv_data[pp * page_stride + t * head_dim + d];
                assert_eq!(
                    actual, expected,
                    "K mismatch at page 0, token {t}, dim {d}"
                );
            }
        }

        // Verify V data for logical page 0
        for t in 0..page_size {
            for d in 0..head_dim {
                let expected = v[t * head_dim + d];
                let actual =
                    cache.kv_data[pp * page_stride + page_size * head_dim + t * head_dim + d];
                assert_eq!(
                    actual, expected,
                    "V mismatch at page 0, token {t}, dim {d}"
                );
            }
        }
    }
}
