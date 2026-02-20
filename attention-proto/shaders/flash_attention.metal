#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// Flash Attention forward kernel (FlashAttention-2 algorithm)
//
// Each threadgroup computes one block of output rows (BLOCK_R query rows).
// Grid dispatch: (ceil(seq_len / BLOCK_R), num_heads, 1)
// Threadgroup size: (BLOCK_R * (BLOCK_C / 8), 1) — enough threads for simdgroup matmul
//
// Initial version: BLOCK_R=16, BLOCK_C=64, D=64 (hardcoded tile sizes for arrays)
// Memory: Q_tile=16*64*4=4KB, K_chunk=64*64*4=16KB, S_tile=16*64*4=4KB = 24KB < 32KB
//
// Uses simdgroup_matrix<float, 8, 8> for Q*K^T and P*V matmul operations.
// Online softmax with running max/sum for numerical stability.
// ---------------------------------------------------------------------------

// Tile sizes for threadgroup memory arrays (compile-time constants).
#define TILE_R  16
#define TILE_C  64
#define TILE_D  64

// Function constants for runtime tile configuration.
// Set by host via MTLFunctionConstantValues at PSO compile time.
constant uint HEAD_DIM [[function_constant(0)]];
constant uint BLOCK_R  [[function_constant(1)]];
constant uint BLOCK_C  [[function_constant(2)]];

// ALiBi (Attention with Linear Biases) function constant.
// When enabled, adds position-dependent linear bias to attention scores:
//   bias = -slope * abs(pos_q - pos_k)
// where slope = 1 / 2^((head_idx + 1) * 8 / num_heads)
constant bool ALIBI_ENABLED [[function_constant(4)]];

kernel void flash_attention(
    device const float* Q       [[buffer(0)]],      // [num_heads, seq_len, head_dim]
    device const float* K       [[buffer(1)]],      // [num_heads, seq_len, head_dim]
    device const float* V       [[buffer(2)]],      // [num_heads, seq_len, head_dim]
    device float*       O       [[buffer(3)]],      // [num_heads, seq_len, head_dim]
    constant AttentionParams& params [[buffer(4)]],
    uint2 tg_pos    [[threadgroup_position_in_grid]],
    uint2 tid_v     [[thread_position_in_threadgroup]]
) {
    // ------------------------------------------------------------------
    // Identify which query block and head this threadgroup processes
    // ------------------------------------------------------------------
    const uint tid_in_tg = tid_v.x;
    const uint block_row = tg_pos.x;                    // which BLOCK_R-row block
    const uint head      = tg_pos.y;                    // which attention head
    const uint seq_len   = params.seq_len;
    const uint D         = HEAD_DIM;
    const uint Br        = BLOCK_R;
    const uint Bc        = BLOCK_C;

    // Base offset for this head (layout: [num_heads, seq_len, head_dim])
    const uint head_offset = head * seq_len * D;

    // Starting query row for this threadgroup
    const uint q_start = block_row * Br;
    if (q_start >= seq_len) return;

    const uint q_count = min(Br, seq_len - q_start);

    // Number of threads available in this threadgroup
    // For BLOCK_R=16: we need at least 32 threads (one simdgroup) for matmul
    // Threadgroup size should be set to 32 (one simdgroup) by host.
    const uint num_threads = 32;  // one simdgroup

    // ------------------------------------------------------------------
    // Threadgroup memory
    // ------------------------------------------------------------------
    threadgroup float q_tile[TILE_R * TILE_D];          // 4 KB
    threadgroup float k_chunk[TILE_C * TILE_D];         // 16 KB
    threadgroup float s_tile[TILE_R * TILE_C];          // 4 KB
                                                        // Total = 24 KB

    // ------------------------------------------------------------------
    // Cooperative load: Q_tile (BLOCK_R x D) from global memory
    // 32 threads load 16*64 = 1024 elements → 32 elements per thread
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
    // Each of BLOCK_R=16 rows needs: m (running max), l (running sum), o_acc[D]
    // With 32 threads and 16 rows, each pair of threads shares a row.
    // For simplicity: thread i handles row (i % BLOCK_R).
    // ------------------------------------------------------------------
    // Each thread maintains state for its assigned row(s).
    // With 32 threads and 16 rows: threads 0-15 each own one row,
    // threads 16-31 help with loads but don't own rows.
    const uint my_row = tid_in_tg;  // only valid for tid < BLOCK_R
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

        // -- Cooperative load K_chunk (BLOCK_C x D) -----------------------
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

        // -- Compute S = Q_tile * K_chunk^T using simdgroup_matrix --------
        // S is BLOCK_R x BLOCK_C (16 x 64).
        // Decompose into 8x8 tiles: (BLOCK_R/8) x (BLOCK_C/8) = 2 x 8 tiles.
        // Each simdgroup computes one or more 8x8 output tiles.
        // With one simdgroup (32 threads), we iterate over all 2*8=16 tiles.
        //
        // For each output tile S[tr][tc] (8x8):
        //   S[tr][tc] = sum_k( Q_tile[tr][k] * K_chunk[tc][k]^T )
        //   where k iterates over D/8 = 8 inner tiles.
        {
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
                        // Load 8x8 tile of Q_tile starting at (tr*8, tk*8)
                        simdgroup_float8x8 q_sg;
                        simdgroup_load(q_sg, &q_tile[tr * 8 * TILE_D + tk * 8], TILE_D);

                        // Load 8x8 tile of K_chunk starting at (tc*8, tk*8)
                        // We want K^T, so we load K transposed:
                        // K_chunk layout is [BLOCK_C, D], K^T is [D, BLOCK_C]
                        // For S = Q * K^T: need K_chunk[tc*8..tc*8+8][tk*8..tk*8+8] transposed
                        simdgroup_float8x8 k_sg;
                        simdgroup_load(k_sg, &k_chunk[tc * 8 * TILE_D + tk * 8], TILE_D, ulong2(0, 0), true);

                        // acc += q_sg * k_sg^T
                        simdgroup_multiply_accumulate(acc, q_sg, k_sg, acc);
                    }

                    // Store 8x8 result tile into s_tile
                    simdgroup_store(acc, &s_tile[tr * 8 * TILE_C + tc * 8], TILE_C);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Scale scores by 1/sqrt(D) + optional ALiBi bias ---------------
        for (uint idx = tid_in_tg; idx < TILE_R * TILE_C; idx += num_threads) {
            uint row = idx / TILE_C;
            uint col = idx % TILE_C;
            if (col < k_count) {
                float score = s_tile[idx] * params.scale;
                // ALiBi: add position-dependent linear bias
                if (ALIBI_ENABLED) {
                    float slope = 1.0 / pow(2.0, float(head + 1) * 8.0 / float(params.num_heads));
                    uint q_pos = q_start + row;
                    uint k_pos = k_start + col;
                    score += -slope * abs(float(q_pos) - float(k_pos));
                }
                s_tile[idx] = score;
            } else {
                s_tile[idx] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Online softmax + O accumulation (per row, threads 0..BLOCK_R-1) --
        if (owns_row) {
            uint row = my_row;

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

                // O += p * V[k_start + col, :]
                uint v_row = k_start + col;
                for (uint d = 0; d < D; d++) {
                    o_acc[d] += p * V[head_offset + v_row * D + d];
                }
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
