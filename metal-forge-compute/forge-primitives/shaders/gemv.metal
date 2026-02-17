// gemv.metal -- General Matrix-Vector multiply kernel.
//
// y[M] = A[M,N] * x[N]
//
// Each thread computes one output element (one row dot product).
// Uses vectorized float4 loads for bandwidth optimization.
// Adapted from inference pipeline matvec pattern.

#include "types.h"

/// GEMV kernel for FP32.
/// Each thread computes: y[row] = dot(A[row, :], x[:])
/// Uses float4 vectorized loads for coalesced memory access.
/// Grid: 1D with total_threads = M (one thread per output row).
kernel void gemv_f32(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.M) {
        return;
    }

    uint row = gid;
    uint N = params.N;

    // Pointer to start of this row in A
    device const float* row_ptr = A + row * N;

    float sum = 0.0f;

    // Vectorized path: process 4 elements at a time
    uint vec4_count = N / 4;
    uint remainder = N % 4;

    device const float4* row_vec = (device const float4*)row_ptr;
    device const float4* x_vec = (device const float4*)x;

    for (uint i = 0; i < vec4_count; i++) {
        float4 a_val = row_vec[i];
        float4 x_val = x_vec[i];
        sum += dot(a_val, x_val);
    }

    // Handle remaining elements
    uint base = vec4_count * 4;
    for (uint i = 0; i < remainder; i++) {
        sum += row_ptr[base + i] * x[base + i];
    }

    y[row] = sum;
}
