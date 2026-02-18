// gemm.metal -- Tiled GEMM (General Matrix Multiply) kernel.
//
// Simple shared-memory tiled GEMM: each threadgroup loads 16x16 tiles
// of A and B into threadgroup memory, computes partial dot products,
// and accumulates into the output matrix C.
//
// C[M,N] = A[M,K] * B[K,N]

#include "types.h"
#include <metal_simdgroup_matrix>

#define TILE_SIZE 16

/// Naive tiled GEMM for FP32.
/// Each thread computes one element of C.
/// Threadgroup size: (TILE_SIZE, TILE_SIZE) = (16, 16).
/// Grid size: (N / TILE_SIZE, M / TILE_SIZE).
kernel void gemm_naive_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    // Bounds check for non-multiple-of-16 sizes
    if (gid.x >= params.N || gid.y >= params.M) {
        return;
    }

    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    uint numTiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load tile of A: row = gid.y, col = t*TILE_SIZE + tid.x
        uint a_col = t * TILE_SIZE + tid.x;
        if (gid.y < params.M && a_col < params.K) {
            tileA[tid.y][tid.x] = A[gid.y * params.K + a_col];
        } else {
            tileA[tid.y][tid.x] = 0.0f;
        }

        // Load tile of B: row = t*TILE_SIZE + tid.y, col = gid.x
        uint b_row = t * TILE_SIZE + tid.y;
        if (b_row < params.K && gid.x < params.N) {
            tileB[tid.y][tid.x] = B[b_row * params.N + gid.x];
        } else {
            tileB[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    C[gid.y * params.N + gid.x] = sum;
}


// ============================================================================
// gemm_simdgroup_f32 -- GEMM using simdgroup_matrix_multiply_accumulate
// ============================================================================
//
// 8 SIMD groups per TG (256 threads), arranged 2 rows x 4 cols.
// Each SIMD group computes TWO 8x8 output tiles (stacked vertically = 16x8).
// TG covers 32 rows x 32 cols of output C.
// Shared memory tiling: BK=32 tiles with bank-conflict padding.
//
// Dispatch: (ceil(N/32), ceil(M/32)) threadgroups of 256 threads.

#define GEMM_BM 32
#define GEMM_BN 32
#define GEMM_BK 32
#define GEMM_PAD 4

kernel void gemm_simdgroup_f32(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float*       C       [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint  tid_flat [[thread_index_in_threadgroup]],
    uint  simd_id  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float tileA[GEMM_BM][GEMM_BK + GEMM_PAD];
    threadgroup float tileB[GEMM_BK][GEMM_BN + GEMM_PAD];

    // 8 SIMD groups: 2 rows x 4 cols, each computes 16x8 (two 8x8 stacked)
    // Total: (2 x 16) x (4 x 8) = 32x32 output per TG
    uint simd_row = simd_id >> 2;   // 0..1
    uint simd_col = simd_id & 3;    // 0..3

    uint c_row = group_id.y * GEMM_BM + simd_row * 16;
    uint c_col = group_id.x * GEMM_BN + simd_col * 8;

    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    // Cooperative loading layout: 256 threads, 32 columns → 8 rows per batch
    uint load_col = tid_flat & 31;
    uint load_row_base = tid_flat >> 5;  // 0..7

    for (uint k = 0; k < K; k += GEMM_BK) {
        // Load A tile: 32 x 32 = 1024 floats, 4 batches of 8 rows
        for (uint r = 0; r < 4; r++) {
            uint row = load_row_base + r * 8;
            uint g_row = group_id.y * GEMM_BM + row;
            uint g_col = k + load_col;
            tileA[row][load_col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        // Load B tile: 32 x 32 = 1024 floats, 4 batches of 8 rows
        for (uint r = 0; r < 4; r++) {
            uint row = load_row_base + r * 8;
            uint g_row = k + row;
            uint g_col = group_id.x * GEMM_BN + load_col;
            tileB[row][load_col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4 MAC steps per BK=32 tile, 2 accumulators per SIMD group
        for (uint kk = 0; kk < GEMM_BK; kk += 8) {
            simdgroup_float8x8 a0, a1, b;
            simdgroup_load(a0, &tileA[simd_row * 16][kk], GEMM_BK + GEMM_PAD);
            simdgroup_load(a1, &tileA[simd_row * 16 + 8][kk], GEMM_BK + GEMM_PAD);
            simdgroup_load(b,  &tileB[kk][simd_col * 8], GEMM_BN + GEMM_PAD);

            simdgroup_multiply_accumulate(acc0, a0, b, acc0);
            simdgroup_multiply_accumulate(acc1, a1, b, acc1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store two 8x8 tiles
    if (c_row + 8 <= M && c_col + 8 <= N) {
        simdgroup_store(acc0, C + c_row * N + c_col, N);
    }
    if (c_row + 16 <= M && c_col + 8 <= N) {
        simdgroup_store(acc1, C + (c_row + 8) * N + c_col, N);
    }
}
