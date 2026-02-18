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
// 8 SIMD groups per TG (256 threads), arranged 4 rows x 2 cols.
// Each SIMD group computes one 8x8 output tile via simdgroup_float8x8.
// TG covers 32 rows x 16 cols of output C.
// Shared memory tiling with BK=16 K-dimension tiles for data reuse.
//
// Dispatch: (ceil(N/16), ceil(M/32)) threadgroups of 256 threads.

#define GEMM_BM 32
#define GEMM_BN 16
#define GEMM_BK 16
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

    // SIMD group layout: 4 rows x 2 cols → 32x16 output per TG
    uint simd_row = simd_id >> 1;
    uint simd_col = simd_id & 1;

    uint c_row = group_id.y * GEMM_BM + simd_row * 8;
    uint c_col = group_id.x * GEMM_BN + simd_col * 8;

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    for (uint k = 0; k < K; k += GEMM_BK) {
        // Cooperative load A tile: BM x BK = 32x16 = 512 floats, 256 threads load 2 each
        for (uint i = tid_flat; i < GEMM_BM * GEMM_BK; i += 256) {
            uint row = i / GEMM_BK;
            uint col = i % GEMM_BK;
            uint g_row = group_id.y * GEMM_BM + row;
            uint g_col = k + col;
            tileA[row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        // Cooperative load B tile: BK x BN = 16x16 = 256 floats, 256 threads load 1 each
        for (uint i = tid_flat; i < GEMM_BK * GEMM_BN; i += 256) {
            uint row = i / GEMM_BN;
            uint col = i % GEMM_BN;
            uint g_row = k + row;
            uint g_col = group_id.x * GEMM_BN + col;
            tileB[row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Two 8x8 multiply-accumulate steps per BK=16 tile
        for (uint kk = 0; kk < GEMM_BK; kk += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, &tileA[simd_row * 8][kk], GEMM_BK + GEMM_PAD);
            simdgroup_load(b, &tileB[kk][simd_col * 8], GEMM_BN + GEMM_PAD);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store 8x8 result to global memory
    if (c_row + 8 <= M && c_col + 8 <= N) {
        simdgroup_store(acc, C + c_row * N + c_col, N);
    }
}
