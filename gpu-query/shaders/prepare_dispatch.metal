#include "types.h"

/// prepare_query_dispatch: compute threadgroup counts from filter match_count.
///
/// Single-thread kernel (dispatch 1 thread). Reads match_count from filter output,
/// computes threadgroup count = ceil(match_count / threads_per_threadgroup),
/// writes DispatchArgs for downstream indirect dispatch.
///
/// This eliminates the GPU→CPU→GPU round-trip for reading match_count between
/// filter and aggregate stages.
kernel void prepare_query_dispatch(
    device const uint* match_count [[buffer(0)]],       // from filter stage
    device DispatchArgs* dispatch_args [[buffer(1)]],    // output for indirect dispatch
    constant uint& threads_per_threadgroup [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // single-thread kernel

    uint count = match_count[0];
    uint tg_count = (count + threads_per_threadgroup - 1) / threads_per_threadgroup;

    dispatch_args->threadgroupsPerGridX = max(tg_count, 1u);
    dispatch_args->threadgroupsPerGridY = 1;
    dispatch_args->threadgroupsPerGridZ = 1;
}
