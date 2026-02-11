#include <metal_stdlib>
#include "search_types.h"

using namespace metal;

/// Turbo search kernel: high-throughput variant that processes multiple
/// bytes per thread using SIMD operations for maximum memory bandwidth.
kernel void turbo_search_kernel(
    device const uchar*         input       [[buffer(0)]],
    device const SearchParams*  params      [[buffer(1)]],
    device atomic_uint*         match_count [[buffer(2)]],
    device GpuMatchResult*      results     [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Stub: will be replaced with turbo search from rust-experiment
    if (tid >= params->total_bytes) return;
}
