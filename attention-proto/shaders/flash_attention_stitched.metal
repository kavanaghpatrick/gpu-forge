#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// Flash Attention Stitched — Proto 2: Function Stitching Overhead Measurement
//
// Variant of Proto 1 kernel with function constant STITCH_MODE controlling
// how inner-loop operations are factored:
//   Mode 0 (monolithic): All operations inline in main kernel body
//   Mode 1 (always_inline): Factored into separate functions with [[always_inline]]
//   Mode 2 (no_inline): Factored into separate functions with [[noinline]]
//
// The Metal compiler eliminates dead branches based on function constants
// at PSO creation time, so only one mode's code path exists at runtime.
//
// Grid dispatch: (ceil(seq_len / BLOCK_R), num_heads, 1)
// Threadgroup size: (32, 1) — one simdgroup
// ---------------------------------------------------------------------------

// Tile sizes for threadgroup memory arrays (compile-time constants).
#define TILE_R  16
#define TILE_C  64
#define TILE_D  64

// Function constants for runtime configuration.
constant uint HEAD_DIM    [[function_constant(0)]];
constant uint BLOCK_R_FC  [[function_constant(1)]];
constant uint BLOCK_C_FC  [[function_constant(2)]];
constant uint STITCH_MODE [[function_constant(3)]];

// ---------------------------------------------------------------------------
// Mode 1: Always-inline helper functions
// The [[always_inline]] attribute guarantees the compiler inlines these,
// eliminating any function call overhead.
// ---------------------------------------------------------------------------

// Compute S = Q_tile * K_chunk^T using simdgroup_matrix, store into s_tile
__attribute__((always_inline))
void compute_scores_inline(
    threadgroup float* q_tile,
    threadgroup float* k_chunk,
    threadgroup float* s_tile,
    uint tid_in_tg
) {
    const uint num_threads = 32;

    // Zero out S_tile first
    for (uint idx = tid_in_tg; idx < TILE_R * TILE_C; idx += num_threads) {
        s_tile[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_tr = TILE_R / 8;   // 2 row tiles
    const uint num_tc = TILE_C / 8;   // 8 col tiles
    const uint num_tk = TILE_D / 8;   // 8 inner tiles

    for (uint tr = 0; tr < num_tr; tr++) {
        for (uint tc = 0; tc < num_tc; tc++) {
            simdgroup_float8x8 acc;
            acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint tk = 0; tk < num_tk; tk++) {
                simdgroup_float8x8 q_sg;
                simdgroup_load(q_sg, &q_tile[tr * 8 * TILE_D + tk * 8], TILE_D);

                simdgroup_float8x8 k_sg;
                simdgroup_load(k_sg, &k_chunk[tc * 8 * TILE_D + tk * 8], TILE_D, ulong2(0, 0), true);

                simdgroup_multiply_accumulate(acc, q_sg, k_sg, acc);
            }

            simdgroup_store(acc, &s_tile[tr * 8 * TILE_C + tc * 8], TILE_C);
        }
    }
}

// Scale scores and apply mask for invalid columns
__attribute__((always_inline))
void scale_scores_inline(
    threadgroup float* s_tile,
    float scale,
    uint k_count,
    uint tid_in_tg
) {
    const uint num_threads = 32;
    for (uint idx = tid_in_tg; idx < TILE_R * TILE_C; idx += num_threads) {
        uint col = idx % TILE_C;
        s_tile[idx] = (col < k_count) ? (s_tile[idx] * scale) : -INFINITY;
    }
}

// Online softmax + output accumulation for one row
__attribute__((always_inline))
void softmax_accumulate_inline(
    threadgroup float* s_tile,
    device const float* V,
    uint row,
    uint k_start,
    uint k_count,
    uint head_offset,
    uint D,
    thread float& m_i,
    thread float& l_i,
    thread float* o_acc
) {
    // 1. Find max in this block's scores for this row
    float m_block = -INFINITY;
    for (uint col = 0; col < k_count; col++) {
        m_block = max(m_block, s_tile[row * TILE_C + col]);
    }

    // 2. Update global max
    float m_prev = m_i;
    m_i = max(m_prev, m_block);

    // 3. Rescale previous accumulator
    float scale_prev = exp(m_prev - m_i);
    l_i *= scale_prev;
    for (uint d = 0; d < D; d++) {
        o_acc[d] *= scale_prev;
    }

    // 4. Compute P = exp(S - m_i), accumulate sum and output
    for (uint col = 0; col < k_count; col++) {
        float p = exp(s_tile[row * TILE_C + col] - m_i);
        l_i += p;

        uint v_row = k_start + col;
        for (uint d = 0; d < D; d++) {
            o_acc[d] += p * V[head_offset + v_row * D + d];
        }
    }
}


// ---------------------------------------------------------------------------
// Mode 2: Non-inline helper functions (compiler decides, but [[noinline]]
// forces a real function call). This tests worst-case function call overhead.
// ---------------------------------------------------------------------------

__attribute__((noinline))
void compute_scores_noinline(
    threadgroup float* q_tile,
    threadgroup float* k_chunk,
    threadgroup float* s_tile,
    uint tid_in_tg
) {
    const uint num_threads = 32;

    for (uint idx = tid_in_tg; idx < TILE_R * TILE_C; idx += num_threads) {
        s_tile[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_tr = TILE_R / 8;
    const uint num_tc = TILE_C / 8;
    const uint num_tk = TILE_D / 8;

    for (uint tr = 0; tr < num_tr; tr++) {
        for (uint tc = 0; tc < num_tc; tc++) {
            simdgroup_float8x8 acc;
            acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            for (uint tk = 0; tk < num_tk; tk++) {
                simdgroup_float8x8 q_sg;
                simdgroup_load(q_sg, &q_tile[tr * 8 * TILE_D + tk * 8], TILE_D);

                simdgroup_float8x8 k_sg;
                simdgroup_load(k_sg, &k_chunk[tc * 8 * TILE_D + tk * 8], TILE_D, ulong2(0, 0), true);

                simdgroup_multiply_accumulate(acc, q_sg, k_sg, acc);
            }

            simdgroup_store(acc, &s_tile[tr * 8 * TILE_C + tc * 8], TILE_C);
        }
    }
}

__attribute__((noinline))
void scale_scores_noinline(
    threadgroup float* s_tile,
    float scale,
    uint k_count,
    uint tid_in_tg
) {
    const uint num_threads = 32;
    for (uint idx = tid_in_tg; idx < TILE_R * TILE_C; idx += num_threads) {
        uint col = idx % TILE_C;
        s_tile[idx] = (col < k_count) ? (s_tile[idx] * scale) : -INFINITY;
    }
}

__attribute__((noinline))
void softmax_accumulate_noinline(
    threadgroup float* s_tile,
    device const float* V,
    uint row,
    uint k_start,
    uint k_count,
    uint head_offset,
    uint D,
    thread float& m_i,
    thread float& l_i,
    thread float* o_acc
) {
    float m_block = -INFINITY;
    for (uint col = 0; col < k_count; col++) {
        m_block = max(m_block, s_tile[row * TILE_C + col]);
    }

    float m_prev = m_i;
    m_i = max(m_prev, m_block);

    float scale_prev = exp(m_prev - m_i);
    l_i *= scale_prev;
    for (uint d = 0; d < D; d++) {
        o_acc[d] *= scale_prev;
    }

    for (uint col = 0; col < k_count; col++) {
        float p = exp(s_tile[row * TILE_C + col] - m_i);
        l_i += p;

        uint v_row = k_start + col;
        for (uint d = 0; d < D; d++) {
            o_acc[d] += p * V[head_offset + v_row * D + d];
        }
    }
}


// ---------------------------------------------------------------------------
// Main kernel: flash_attention_stitched
//
// STITCH_MODE function constant selects which code path to use:
//   0 = monolithic (all inline in kernel body, identical to Proto 1)
//   1 = always_inline functions (guaranteed inlined — should be identical perf)
//   2 = noinline functions (forced function calls — measures call overhead)
//
// The compiler eliminates dead branches at PSO compile time.
// ---------------------------------------------------------------------------
kernel void flash_attention_stitched(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device float*       O       [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint2 tg_pos    [[threadgroup_position_in_grid]],
    uint2 tid_v     [[thread_position_in_threadgroup]]
) {
    const uint tid_in_tg = tid_v.x;
    const uint block_row = tg_pos.x;
    const uint head      = tg_pos.y;
    const uint seq_len   = params.seq_len;
    const uint D         = HEAD_DIM;
    const uint Br        = BLOCK_R_FC;
    const uint Bc        = BLOCK_C_FC;

    const uint head_offset = head * seq_len * D;
    const uint q_start = block_row * Br;
    if (q_start >= seq_len) return;

    const uint q_count = min(Br, seq_len - q_start);
    const uint num_threads = 32;

    // ------------------------------------------------------------------
    // Threadgroup memory
    // ------------------------------------------------------------------
    threadgroup float q_tile[TILE_R * TILE_D];          // 4 KB
    threadgroup float k_chunk[TILE_C * TILE_D];         // 16 KB
    threadgroup float s_tile[TILE_R * TILE_C];          // 4 KB
                                                         // Total = 24 KB

    // ------------------------------------------------------------------
    // Cooperative load: Q_tile
    // ------------------------------------------------------------------
    {
        const uint total_q = TILE_R * TILE_D;
        for (uint idx = tid_in_tg; idx < total_q; idx += num_threads) {
            uint r = idx / TILE_D;
            uint c = idx % TILE_D;
            uint global_row = q_start + r;
            q_tile[idx] = (r < q_count && c < D) ?
                Q[head_offset + global_row * D + c] : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------
    // Per-row online softmax state (registers)
    // ------------------------------------------------------------------
    const uint my_row = tid_in_tg;
    const bool owns_row = (my_row < q_count);

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_acc[TILE_D];
    for (uint d = 0; d < TILE_D; d++) {
        o_acc[d] = 0.0f;
    }

    // ------------------------------------------------------------------
    // Outer loop: iterate over K/V blocks
    // ------------------------------------------------------------------
    const uint num_kv_blocks = (seq_len + Bc - 1) / Bc;

    for (uint j = 0; j < num_kv_blocks; j++) {
        const uint k_start = j * Bc;
        const uint k_count = min(Bc, seq_len - k_start);

        // -- Cooperative load K_chunk --
        {
            const uint total_k = TILE_C * TILE_D;
            for (uint idx = tid_in_tg; idx < total_k; idx += num_threads) {
                uint r = idx / TILE_D;
                uint c = idx % TILE_D;
                uint global_row = k_start + r;
                k_chunk[idx] = (global_row < seq_len && c < D) ?
                    K[head_offset + global_row * D + c] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ==============================================================
        // STITCH_MODE 0: Monolithic (all operations inline)
        // ==============================================================
        if (STITCH_MODE == 0) {
            // -- Compute S = Q_tile * K_chunk^T --
            {
                for (uint idx = tid_in_tg; idx < TILE_R * TILE_C; idx += num_threads) {
                    s_tile[idx] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint num_tr = TILE_R / 8;
                const uint num_tc = TILE_C / 8;
                const uint num_tk = TILE_D / 8;

                for (uint tr = 0; tr < num_tr; tr++) {
                    for (uint tc = 0; tc < num_tc; tc++) {
                        simdgroup_float8x8 acc;
                        acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

                        for (uint tk = 0; tk < num_tk; tk++) {
                            simdgroup_float8x8 q_sg;
                            simdgroup_load(q_sg, &q_tile[tr * 8 * TILE_D + tk * 8], TILE_D);

                            simdgroup_float8x8 k_sg;
                            simdgroup_load(k_sg, &k_chunk[tc * 8 * TILE_D + tk * 8], TILE_D, ulong2(0, 0), true);

                            simdgroup_multiply_accumulate(acc, q_sg, k_sg, acc);
                        }

                        simdgroup_store(acc, &s_tile[tr * 8 * TILE_C + tc * 8], TILE_C);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // -- Scale scores --
            for (uint idx = tid_in_tg; idx < TILE_R * TILE_C; idx += num_threads) {
                uint col = idx % TILE_C;
                s_tile[idx] = (col < k_count) ? (s_tile[idx] * params.scale) : -INFINITY;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // -- Online softmax + O accumulation --
            if (owns_row) {
                uint row = my_row;

                float m_block = -INFINITY;
                for (uint col = 0; col < k_count; col++) {
                    m_block = max(m_block, s_tile[row * TILE_C + col]);
                }

                float m_prev = m_i;
                m_i = max(m_prev, m_block);

                float scale_prev = exp(m_prev - m_i);
                l_i *= scale_prev;
                for (uint d = 0; d < D; d++) {
                    o_acc[d] *= scale_prev;
                }

                for (uint col = 0; col < k_count; col++) {
                    float p = exp(s_tile[row * TILE_C + col] - m_i);
                    l_i += p;

                    uint v_row = k_start + col;
                    for (uint d = 0; d < D; d++) {
                        o_acc[d] += p * V[head_offset + v_row * D + d];
                    }
                }
            }
        }

        // ==============================================================
        // STITCH_MODE 1: Always-inline functions
        // ==============================================================
        else if (STITCH_MODE == 1) {
            compute_scores_inline(q_tile, k_chunk, s_tile, tid_in_tg);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            scale_scores_inline(s_tile, params.scale, k_count, tid_in_tg);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (owns_row) {
                softmax_accumulate_inline(
                    s_tile, V, my_row, k_start, k_count,
                    head_offset, D, m_i, l_i, o_acc
                );
            }
        }

        // ==============================================================
        // STITCH_MODE 2: Non-inline functions (forced function calls)
        // ==============================================================
        else if (STITCH_MODE == 2) {
            compute_scores_noinline(q_tile, k_chunk, s_tile, tid_in_tg);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            scale_scores_noinline(s_tile, params.scale, k_count, tid_in_tg);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (owns_row) {
                softmax_accumulate_noinline(
                    s_tile, V, my_row, k_start, k_count,
                    head_offset, D, m_i, l_i, o_acc
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ------------------------------------------------------------------
    // Normalize and write output: O[row] = o_acc / l_i
    // ------------------------------------------------------------------
    if (owns_row) {
        uint global_row = q_start + my_row;
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (uint d = 0; d < D; d++) {
            O[head_offset + global_row * D + d] = o_acc[d] * inv_l;
        }
    }
}
