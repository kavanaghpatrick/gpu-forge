#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// PagedAttention V2 — Phase 1: Partition Kernel
//
// Each threadgroup handles one query block x one partition of KV pages.
// Pages are loaded via page_table indirection (logical -> physical mapping).
// Outputs partial attention (O_partial, m_partial, l_partial) for each
// partition, to be combined by the phase 2 reduce kernel.
//
// Memory budget (PAGE_SIZE=16, HEAD_DIM=64):
//   Q_tile:   16 * 64 * 4 = 4 KB
//   K_page:   16 * 64 * 4 = 4 KB
//   V_page:   16 * 64 * 4 = 4 KB
//   S_buf:    16 * 16 * 4 = 1 KB
//   Total:    ~13 KB  (well within 32 KB)
// ---------------------------------------------------------------------------

// Tile sizes — compile-time constants for threadgroup array dimensions.
// PAGE_SIZE=16 tokens per page, HEAD_DIM=64 dimensions.
#define TILE_Q    16     // query block size (BLOCK_R)
#define TILE_PAGE 16     // tokens per page (matches page_size)
#define TILE_D    64     // head dimension

// Function constants set by host at PSO compile time.
constant uint HEAD_DIM   [[function_constant(0)]];
constant uint PAGE_SIZE  [[function_constant(1)]];

kernel void paged_attention_partition(
    device const float* Q           [[buffer(0)]],   // [num_heads, seq_len, head_dim]
    device const float* KV_cache    [[buffer(1)]],   // [num_pages, 2, page_size, head_dim]
                                                      // layout: K pages then V pages interleaved per page
    device const uint*  page_table  [[buffer(2)]],   // [max_pages_per_seq] logical->physical
    device float*       O_partial   [[buffer(3)]],   // [num_partitions, num_heads, BLOCK_R, head_dim]
    device float*       m_partial   [[buffer(4)]],   // [num_partitions, num_heads, BLOCK_R]
    device float*       l_partial   [[buffer(5)]],   // [num_partitions, num_heads, BLOCK_R]
    constant AttentionParams& params [[buffer(6)]],
    uint2 tg_pos  [[threadgroup_position_in_grid]],  // (query_block, partition)
    uint  tid     [[thread_index_in_threadgroup]]
) {
    // ------------------------------------------------------------------
    // Decode grid position
    // ------------------------------------------------------------------
    const uint block_row   = tg_pos.x;   // which query block
    const uint partition   = tg_pos.y;   // which partition of KV pages

    const uint seq_len     = params.seq_len;
    const uint D           = HEAD_DIM;
    const uint ps          = PAGE_SIZE;
    const uint num_parts   = params.num_partitions;
    const uint scale_val   = params.head_dim;  // unused, use params.scale directly

    // For simplicity this prototype handles head=0 (single-head).
    // Multi-head support will come in the host code (dispatch grid includes head dim).
    const uint head        = 0;
    const uint head_offset = head * seq_len * D;

    // Query block bounds
    const uint q_start = block_row * TILE_Q;
    if (q_start >= seq_len) return;
    const uint q_count = min((uint)TILE_Q, seq_len - q_start);

    // ------------------------------------------------------------------
    // Compute partition page range
    // Total logical pages covering the context
    // ------------------------------------------------------------------
    const uint total_context   = params.max_context_len;
    const uint total_pages     = (total_context + ps - 1) / ps;
    const uint pages_per_part  = (total_pages + num_parts - 1) / num_parts;
    const uint part_page_start = partition * pages_per_part;
    const uint part_page_end   = min(part_page_start + pages_per_part, total_pages);

    if (part_page_start >= total_pages) return;

    // ------------------------------------------------------------------
    // Threadgroup memory
    // ------------------------------------------------------------------
    threadgroup float q_tile[TILE_Q * TILE_D];        // 4 KB
    threadgroup float k_page[TILE_PAGE * TILE_D];     // 4 KB
    threadgroup float v_page[TILE_PAGE * TILE_D];     // 4 KB
    threadgroup float s_buf[TILE_Q * TILE_PAGE];      // 1 KB
                                                       // Total: 13 KB

    const uint num_threads = 32;  // one simdgroup

    // ------------------------------------------------------------------
    // Cooperative load: Q_tile (TILE_Q x D) from global Q buffer
    // ------------------------------------------------------------------
    {
        const uint total_elems = TILE_Q * TILE_D;
        for (uint idx = tid; idx < total_elems; idx += num_threads) {
            uint r = idx / TILE_D;
            uint c = idx % TILE_D;
            uint global_row = q_start + r;
            q_tile[idx] = (r < q_count && c < D)
                ? Q[head_offset + global_row * D + c]
                : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------
    // Per-row online softmax state (registers)
    // Thread tid owns row tid (for tid < TILE_Q).
    // ------------------------------------------------------------------
    const uint my_row = tid;
    const bool owns_row = (my_row < q_count);

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_acc[TILE_D];
    for (uint d = 0; d < TILE_D; d++) {
        o_acc[d] = 0.0f;
    }

    // ------------------------------------------------------------------
    // Iterate over pages in this partition
    // ------------------------------------------------------------------
    for (uint pg = part_page_start; pg < part_page_end; pg++) {
        // Look up physical page from page table
        uint phys_page = page_table[pg];

        // KV_cache layout: [num_pages, 2, page_size, head_dim]
        // K offset for this page: phys_page * 2 * ps * D + 0
        // V offset for this page: phys_page * 2 * ps * D + ps * D
        uint kv_base = phys_page * 2 * ps * D;

        // Number of valid tokens in this page
        uint page_token_start = pg * ps;
        uint page_token_count = min(ps, total_context - page_token_start);

        // -- Cooperative load K_page --
        {
            const uint total_elems = TILE_PAGE * TILE_D;
            for (uint idx = tid; idx < total_elems; idx += num_threads) {
                uint r = idx / TILE_D;
                uint c = idx % TILE_D;
                k_page[idx] = (r < page_token_count && c < D)
                    ? KV_cache[kv_base + r * D + c]
                    : 0.0f;
            }
        }
        // -- Cooperative load V_page --
        {
            uint v_offset = kv_base + ps * D;
            const uint total_elems = TILE_PAGE * TILE_D;
            for (uint idx = tid; idx < total_elems; idx += num_threads) {
                uint r = idx / TILE_D;
                uint c = idx % TILE_D;
                v_page[idx] = (r < page_token_count && c < D)
                    ? KV_cache[v_offset + r * D + c]
                    : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Compute scores S = Q_tile * K_page^T (dot products) --
        // s_buf[qr][kc] = dot(Q_tile[qr], K_page[kc]) * scale
        {
            const uint total_scores = TILE_Q * TILE_PAGE;
            for (uint idx = tid; idx < total_scores; idx += num_threads) {
                uint qr = idx / TILE_PAGE;
                uint kc = idx % TILE_PAGE;

                float dot_val = 0.0f;
                if (qr < q_count && kc < page_token_count) {
                    for (uint d = 0; d < TILE_D; d++) {
                        dot_val += q_tile[qr * TILE_D + d] * k_page[kc * TILE_D + d];
                    }
                    dot_val *= params.scale;
                } else {
                    dot_val = -INFINITY;
                }
                s_buf[idx] = dot_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Online softmax + O accumulation (per row) --
        if (owns_row) {
            uint row = my_row;

            // 1. Find max score in this page for this row
            float m_block = -INFINITY;
            for (uint c = 0; c < page_token_count; c++) {
                m_block = max(m_block, s_buf[row * TILE_PAGE + c]);
            }

            // 2. Update running max
            float m_prev = m_i;
            m_i = max(m_prev, m_block);

            // 3. Rescale previous accumulator
            float scale_prev = exp(m_prev - m_i);
            l_i *= scale_prev;
            for (uint d = 0; d < TILE_D; d++) {
                o_acc[d] *= scale_prev;
            }

            // 4. Accumulate exp(score - m_i) * V
            for (uint c = 0; c < page_token_count; c++) {
                float p = exp(s_buf[row * TILE_PAGE + c] - m_i);
                l_i += p;
                for (uint d = 0; d < TILE_D; d++) {
                    o_acc[d] += p * v_page[c * TILE_D + d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ------------------------------------------------------------------
    // Write partial results for this partition
    //
    // O_partial: [num_partitions, num_heads, BLOCK_R, head_dim]
    // m_partial: [num_partitions, num_heads, BLOCK_R]
    // l_partial: [num_partitions, num_heads, BLOCK_R]
    //
    // For single-head prototype, num_heads=1, head dim is collapsed.
    // Index: partition * (TILE_Q * D) + row * D + d
    // ------------------------------------------------------------------
    if (owns_row) {
        uint row = my_row;
        uint out_base = (partition * TILE_Q + block_row * num_parts * TILE_Q) * D;
        // Simplified flat indexing: partition * (num_query_blocks * TILE_Q * D) would be complex.
        // Use: (block_row * num_parts + partition) as a linear partition-block index.
        uint pb_idx = block_row * num_parts + partition;
        uint o_base = pb_idx * TILE_Q * D + row * D;
        uint ml_base = pb_idx * TILE_Q + row;

        for (uint d = 0; d < D; d++) {
            O_partial[o_base + d] = o_acc[d];
        }
        m_partial[ml_base] = m_i;
        l_partial[ml_base] = l_i;
    }
}
