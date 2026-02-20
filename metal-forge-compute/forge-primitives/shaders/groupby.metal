// groupby.metal -- Sort-based group-by aggregate kernels.
//
// After keys are sorted via radix sort, two kernels complete the group-by:
//   1. groupby_boundary_detect: flag[i] = (key[i] != key[i-1]) -- marks group boundaries
//   2. groupby_segmented_reduce: within each group segment, compute sum/count/min/max
//
// The segmented reduce uses a simple sequential scan approach per thread:
// each thread finds its group start via the boundary flags, then accumulates
// over the segment. Since groups are contiguous after sort, this is efficient.

#include <metal_stdlib>
#include "types.h"
using namespace metal;

/// Detect group boundaries in sorted key array.
/// flag[0] = 1 (first element always starts a group).
/// flag[i] = (key[i] != key[i-1]) for i > 0.
kernel void groupby_boundary_detect(
    device const uint*   sorted_keys   [[buffer(0)]],
    device       uint*   flags         [[buffer(1)]],
    device const GroupByParams* params  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = params->element_count;
    if (gid >= n) return;

    if (gid == 0) {
        flags[gid] = 1;
    } else {
        flags[gid] = (sorted_keys[gid] != sorted_keys[gid - 1]) ? 1u : 0u;
    }
}

/// Segmented reduce: compute per-group aggregates (sum, count, min, max) of values.
///
/// Each thread handles one group. Thread gid processes the group_id == gid.
/// group_offsets[gid] = start index of group gid in the sorted array.
/// Group end = group_offsets[gid+1] (or element_count for the last group).
///
/// Outputs 4 buffers of num_groups each:
///   agg_sum[gid], agg_count[gid], agg_min[gid], agg_max[gid]
kernel void groupby_segmented_reduce(
    device const float*  values        [[buffer(0)]],
    device const uint*   group_offsets [[buffer(1)]],
    device       float*  agg_sum       [[buffer(2)]],
    device       uint*   agg_count     [[buffer(3)]],
    device       float*  agg_min       [[buffer(4)]],
    device       float*  agg_max       [[buffer(5)]],
    device const GroupByParams* params  [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint num_groups = params->num_groups;
    uint n = params->element_count;
    if (gid >= num_groups) return;

    uint start = group_offsets[gid];
    uint end = (gid + 1 < num_groups) ? group_offsets[gid + 1] : n;

    float sum_val = 0.0f;
    uint  count_val = 0;
    float min_val = HUGE_VALF;
    float max_val = -HUGE_VALF;

    for (uint i = start; i < end; i++) {
        float v = values[i];
        sum_val += v;
        count_val += 1;
        min_val = min(min_val, v);
        max_val = max(max_val, v);
    }

    agg_sum[gid]   = sum_val;
    agg_count[gid]  = count_val;
    agg_min[gid]    = min_val;
    agg_max[gid]    = max_val;
}
