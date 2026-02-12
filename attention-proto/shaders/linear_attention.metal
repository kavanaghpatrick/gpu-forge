#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// FLA Linear Attention — chunk_h kernel
//
// Computes per-chunk hidden state delta:
//   delta_H[chunk] = K_chunk^T * V_chunk
//
// Where K_chunk is [CHUNK_SIZE, D] and V_chunk is [CHUNK_SIZE, D].
// The outer product K^T * V is (D x CHUNK_SIZE) * (CHUNK_SIZE x D) = D x D.
//
// Each threadgroup handles one chunk.
// Thread mapping: tid maps to one (i, j) element of the D x D output.
//   delta_H[i][j] = sum_{t=0}^{CHUNK_SIZE-1} K[chunk_start + t][i] * V[chunk_start + t][j]
//
// The cumulative H_c = sum(delta_H[0..c]) is computed on the host (CPU prefix sum
// over D x D matrices — simple addition, not a bottleneck for the prototype).
//
// Memory budget (CHUNK_SIZE=32, HEAD_DIM=64):
//   K_chunk: 32 * 64 * 4 = 8 KB
//   V_chunk: 32 * 64 * 4 = 8 KB
//   Total:   16 KB (within 32 KB limit)
// ---------------------------------------------------------------------------

// Compile-time tile sizes for threadgroup memory arrays.
// TILE_C = chunk size, TILE_D = head dimension.
// Keep chunk small (32) to stay within threadgroup memory budget.
#define TILE_C 32
#define TILE_D 64

// Function constants set by host at PSO compile time.
constant uint HEAD_DIM    [[function_constant(0)]];
constant uint CHUNK_SIZE  [[function_constant(4)]];

kernel void chunk_h(
    device const float* K       [[buffer(0)]],      // [seq_len, head_dim]
    device const float* V       [[buffer(1)]],      // [seq_len, head_dim]
    device float*       H_out   [[buffer(2)]],      // [num_chunks, head_dim, head_dim]
    constant AttentionParams& params [[buffer(3)]],
    uint tg_id  [[threadgroup_position_in_grid]],    // chunk index
    uint tid    [[thread_index_in_threadgroup]]
) {
    const uint D  = HEAD_DIM;
    const uint C  = CHUNK_SIZE;
    const uint chunk_start = tg_id * C;

    // Thread maps to one (i, j) element of the D x D delta_H matrix.
    // tid ranges from 0 to D*D - 1 (e.g., 0..4095 for D=64).
    const uint i = tid / D;  // row in delta_H
    const uint j = tid % D;  // col in delta_H

    // Guard: if D*D < threadgroup size, extra threads exit early.
    // Also guard if this thread's (i,j) is out of bounds for non-square or smaller D.
    if (i >= D || j >= D) return;

    // Threadgroup memory for K_chunk and V_chunk.
    // Loaded cooperatively by all threads in the threadgroup.
    threadgroup float K_chunk[TILE_C * TILE_D];  // [CHUNK_SIZE, HEAD_DIM]
    threadgroup float V_chunk[TILE_C * TILE_D];  // [CHUNK_SIZE, HEAD_DIM]

    // Cooperative load of K_chunk and V_chunk into threadgroup memory.
    // Total elements to load: CHUNK_SIZE * HEAD_DIM per array.
    // Each thread loads multiple elements (stride by threadgroup size).
    const uint tg_size = D * D;  // total threads in threadgroup
    const uint load_count = C * D;

    for (uint idx = tid; idx < load_count; idx += tg_size) {
        uint t = idx / D;  // token index within chunk
        uint d = idx % D;  // dimension index
        uint global_idx = (chunk_start + t) * D + d;

        // Bounds check: if chunk extends past seq_len, zero-fill
        if (chunk_start + t < params.seq_len) {
            K_chunk[t * TILE_D + d] = K[global_idx];
            V_chunk[t * TILE_D + d] = V[global_idx];
        } else {
            K_chunk[t * TILE_D + d] = 0.0f;
            V_chunk[t * TILE_D + d] = 0.0f;
        }
    }

    // Barrier: ensure all threads have finished loading before computing.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute delta_H[i][j] = sum_{t=0}^{C-1} K_chunk[t][i] * V_chunk[t][j]
    // This is a dot product over the chunk dimension.
    float acc = 0.0f;
    for (uint t = 0; t < C; t++) {
        acc += K_chunk[t * TILE_D + i] * V_chunk[t * TILE_D + j];
    }

    // Write result to global memory.
    // H_out layout: [num_chunks, D, D], row-major.
    const uint chunk_offset = tg_id * D * D;
    H_out[chunk_offset + i * D + j] = acc;
}
