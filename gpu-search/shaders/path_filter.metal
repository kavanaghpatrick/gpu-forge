#include <metal_stdlib>
#include "search_types.h"

using namespace metal;

/// Path filter kernel: filters the GPU-resident filesystem index
/// by matching path entries against a pattern. Used for filename search.
kernel void path_filter_kernel(
    device const GpuPathEntry*  entries     [[buffer(0)]],
    device const uchar*         pattern     [[buffer(1)]],
    device atomic_uint*         match_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Stub: will be replaced with path filter from rust-experiment
}
