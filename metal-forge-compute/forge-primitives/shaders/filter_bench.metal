// filter_bench.metal -- Columnar filter kernel for benchmarking.
//
// Input: u32 column + threshold value
// Output: match count via atomic counter
// Uses simd_sum for efficient per-threadgroup match counting.
// 256 threads per threadgroup.

#include <metal_stdlib>
#include "types.h"
using namespace metal;

#define FILTER_THREADS_PER_TG 256

/// Columnar filter kernel: count elements greater than threshold.
///
/// Each thread evaluates one element, uses simd_sum to aggregate matches
/// within each SIMD group, then the first lane of each SIMD group atomically
/// adds to the global match count.
///
/// Buffers:
///   [0] input:   device const uint*    -- input column
///   [1] output:  device atomic_uint*   -- match count (single atomic uint)
///   [2] params:  constant FilterBenchParams& -- element_count + threshold
kernel void filter_count_gt(
    device const uint*         input   [[buffer(0)]],
    device atomic_uint*        output  [[buffer(1)]],
    constant FilterBenchParams& params  [[buffer(2)]],
    uint gid                           [[thread_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]]
) {
    // Out-of-bounds threads contribute 0 matches
    uint match_val = 0;
    if (gid < params.element_count) {
        match_val = (input[gid] > params.threshold) ? 1u : 0u;
    }

    // SIMD-level reduction: sum matches within SIMD group (32 threads)
    uint simd_matches = simd_sum(match_val);

    // First lane of each SIMD group atomically adds to global count
    if (simd_lane == 0 && simd_matches > 0) {
        atomic_fetch_add_explicit(output, simd_matches, memory_order_relaxed);
    }
}
