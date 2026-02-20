#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// PagedAttention V2 — Phase 2: Reduce Kernel
//
// Combines partial outputs from all partitions using log-sum-exp reduction.
// Each threadgroup handles one query block. Each thread handles one dimension
// of one query row, iterating over all partitions.
//
// Input layout (from phase 1 partition kernel):
//   pb_idx = block_row * num_partitions + partition
//   O_partial[pb_idx * TILE_Q * D + row * D + d]
//   m_partial[pb_idx * TILE_Q + row]
//   l_partial[pb_idx * TILE_Q + row]
//
// Output:
//   O_final[seq_len, head_dim]
// ---------------------------------------------------------------------------

#define TILE_Q  16   // must match partition kernel
#define TILE_D  64   // must match partition kernel

// Function constants set by host at PSO compile time.
constant uint HEAD_DIM   [[function_constant(0)]];
constant uint PAGE_SIZE  [[function_constant(1)]];

kernel void paged_attention_reduce(
    device const float* O_partial [[buffer(0)]],  // [num_blocks * num_partitions, TILE_Q, head_dim]
    device const float* m_partial [[buffer(1)]],  // [num_blocks * num_partitions, TILE_Q]
    device const float* l_partial [[buffer(2)]],  // [num_blocks * num_partitions, TILE_Q]
    device float* O_final         [[buffer(3)]],  // [seq_len, head_dim]
    constant AttentionParams& params [[buffer(4)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],  // (query_block, 0)
    uint tid     [[thread_index_in_threadgroup]]
) {
    const uint block_row = tg_pos.x;
    const uint seq_len   = params.seq_len;
    const uint D         = HEAD_DIM;
    const uint num_parts = params.num_partitions;

    // Each thread handles one (row, dim) pair within the query block.
    // Thread layout: tid = row * TILE_D + d_col
    const uint row   = tid / TILE_D;
    const uint d_col = tid % TILE_D;

    const uint global_row = block_row * TILE_Q + row;

    // Out-of-bounds: row beyond seq_len or d_col beyond head_dim
    if (row >= TILE_Q || global_row >= seq_len || d_col >= D) {
        return;
    }

    // ------------------------------------------------------------------
    // Step 1: Find global max across all partitions for this row
    // ------------------------------------------------------------------
    float m_global = -INFINITY;
    for (uint p = 0; p < num_parts; p++) {
        uint pb_idx  = block_row * num_parts + p;
        uint ml_base = pb_idx * TILE_Q + row;
        float m_p    = m_partial[ml_base];
        m_global     = max(m_global, m_p);
    }

    // ------------------------------------------------------------------
    // Step 2: Accumulate rescaled O and l across partitions
    // ------------------------------------------------------------------
    float o_total = 0.0f;
    float l_total = 0.0f;

    for (uint p = 0; p < num_parts; p++) {
        uint pb_idx  = block_row * num_parts + p;
        uint ml_base = pb_idx * TILE_Q + row;
        uint o_base  = pb_idx * TILE_Q * D + row * D + d_col;

        float m_p = m_partial[ml_base];
        float l_p = l_partial[ml_base];

        // Rescale this partition's contribution using log-sum-exp trick
        float correction = exp(m_p - m_global);
        float l_corrected = l_p * correction;
        float o_corrected = O_partial[o_base] * correction;

        l_total += l_corrected;
        o_total += o_corrected;
    }

    // ------------------------------------------------------------------
    // Step 3: Normalize and write final output
    // ------------------------------------------------------------------
    float inv_l = (l_total > 0.0f) ? (1.0f / l_total) : 0.0f;
    O_final[global_row * D + d_col] = o_total * inv_l;
}
