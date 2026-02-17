// gemm.metal -- Tiled GEMM (General Matrix Multiply) kernel.
//
// Simple shared-memory tiled GEMM: each threadgroup loads 16x16 tiles
// of A and B into threadgroup memory, computes partial dot products,
// and accumulates into the output matrix C.
//
// C[M,N] = A[M,K] * B[K,N]

#include "types.h"

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
