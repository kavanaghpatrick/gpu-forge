#include <metal_stdlib>
#include "search_types.h"

using namespace metal;

/// Batch search kernel: persistent kernel variant that processes
/// multiple files in a single dispatch, using file offset metadata
/// to partition work across the input buffer.
kernel void batch_search_kernel(
    device const uchar*         input       [[buffer(0)]],
    device const SearchParams*  params      [[buffer(1)]],
    device atomic_uint*         match_count [[buffer(2)]],
    device GpuMatchResult*      results     [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Stub: will be replaced with persistent batch search from rust-experiment
    if (tid >= params->total_bytes) return;
}
