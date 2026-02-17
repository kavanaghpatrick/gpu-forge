// compact_scan.metal -- Stream compaction kernels for forge-primitives.
//
// Implements scan-based stream compaction:
//   1. compact_flags: evaluate predicate (value > threshold), write 0/1 to flags array
//   2. (reuse scan_local + scan_partials + scan_add_offsets from scan.metal for prefix scan)
//   3. compact_scatter: if flags[i]==1, write input[i] to output[scan[i]]
//
// The full pipeline is 5 dispatches in 1 command buffer:
//   compact_flags -> scan_local -> scan_partials -> scan_add_offsets -> compact_scatter

#include "types.h"


// ============================================================================
// compact_flags -- Evaluate predicate and write 0/1 flags
// ============================================================================
//
// Buffer layout:
//   buffer(0): input data (uint array, N elements)
//   buffer(1): flags output (uint array, N elements) -- 0 or 1
//   buffer(2): CompactParams { element_count, threshold }
//
// Predicate: flags[i] = (input[i] > threshold) ? 1 : 0
//
// Dispatch: ceil(N / 256) threadgroups of 256 threads.

kernel void compact_flags(
    device const uint*       input           [[buffer(0)]],
    device uint*             flags           [[buffer(1)]],
    constant CompactParams&  params          [[buffer(2)]],
    uint gid                                 [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) return;
    flags[gid] = (input[gid] > params.threshold) ? 1u : 0u;
}


// ============================================================================
// compact_scatter -- Scatter selected elements to compacted output
// ============================================================================
//
// Buffer layout:
//   buffer(0): input data (uint array, N elements)
//   buffer(1): flags (uint array, N elements) -- 0 or 1 from compact_flags
//   buffer(2): scan (uint array, N elements) -- exclusive prefix scan of flags
//   buffer(3): output (uint array, up to N elements) -- compacted result
//   buffer(4): CompactParams { element_count, threshold }
//
// If flags[gid]==1, output[scan[gid]] = input[gid].
// The total output count = scan[N-1] + flags[N-1].
//
// Dispatch: ceil(N / 256) threadgroups of 256 threads.

kernel void compact_scatter(
    device const uint*       input           [[buffer(0)]],
    device const uint*       flags           [[buffer(1)]],
    device const uint*       scan            [[buffer(2)]],
    device uint*             output          [[buffer(3)]],
    constant CompactParams&  params          [[buffer(4)]],
    uint gid                                 [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) return;
    if (flags[gid] == 1u) {
        output[scan[gid]] = input[gid];
    }
}
