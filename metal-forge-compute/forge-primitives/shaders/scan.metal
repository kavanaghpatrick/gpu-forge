// scan.metal -- 3-pass exclusive prefix scan for forge-primitives.
//
// Implements reduce-then-scan approach (NOT decoupled lookback which deadlocks
// on Apple Silicon).
//
// 3-pass algorithm:
//   Pass 1 (scan_local): Each threadgroup of 256 threads processes 1024 elements
//     (4 per thread) using SIMD prefix sum. Writes threadgroup total to
//     partials[threadgroup_id].
//
//   Pass 2 (scan_partials): Single threadgroup scans the partials array.
//     Handles up to 1024 partials (256 threads * 4 elements/thread).
//
//   Pass 3 (scan_add_offsets): Each threadgroup adds partials[threadgroup_id]
//     to every element in its chunk.
//
// All kernels use 256 threads per threadgroup.
// Each thread handles 4 elements = 1024 elements per threadgroup.

#include "types.h"

#define SCAN_THREADS_PER_TG 256
#define SCAN_ELEMENTS_PER_TG 1024  // 4 elements per thread


// ============================================================================
// scan_local -- Per-threadgroup SIMD-based exclusive scan
// ============================================================================
//
// Buffer layout:
//   buffer(0): input data (uint array, N elements)
//   buffer(1): output data (uint array, N elements) -- exclusive prefix scan
//   buffer(2): partials (uint array, one per threadgroup) -- threadgroup total sums
//   buffer(3): ScanParams
//
// Dispatch: ceil(N / 1024) threadgroups of 256 threads.

kernel void scan_local(
    device const uint*       input           [[buffer(0)]],
    device uint*             output          [[buffer(1)]],
    device uint*             partials        [[buffer(2)]],
    constant ScanParams&     params          [[buffer(3)]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_idx                              [[threadgroup_position_in_grid]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_idx                            [[simdgroup_index_in_threadgroup]]
) {
    // Shared memory for cross-SIMD aggregation (256 threads / 32 = 8 SIMD groups)
    threadgroup uint simd_totals[8];

    // --- Step 1: Load 4 elements per thread ---
    uint base = tg_idx * SCAN_ELEMENTS_PER_TG + tid_in_tg * 4;
    uint4 vals = load_uint4_safe(input, base, params.element_count);

    // --- Step 2: Compute per-thread total ---
    uint thread_total = vals.x + vals.y + vals.z + vals.w;

    // --- Step 3: SIMD prefix sum for inter-thread offsets within SIMD group ---
    uint simd_prefix = simd_prefix_exclusive_sum(thread_total);

    // --- Step 4: Store SIMD group totals for cross-SIMD aggregation ---
    if (simd_lane == 31) {
        simd_totals[simd_idx] = simd_prefix + thread_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 5: Cross-SIMD prefix scan (only first SIMD group participates) ---
    if (tid_in_tg < 8) {
        simd_totals[tid_in_tg] = simd_prefix_exclusive_sum(simd_totals[tid_in_tg]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 6: Compute base offset for this thread ---
    uint base_offset = simd_prefix + simd_totals[simd_idx];

    // --- Step 7: Write 4 exclusive prefix outputs ---
    if (base     < params.element_count) output[base]     = base_offset;
    if (base + 1 < params.element_count) output[base + 1] = base_offset + vals.x;
    if (base + 2 < params.element_count) output[base + 2] = base_offset + vals.x + vals.y;
    if (base + 3 < params.element_count) output[base + 3] = base_offset + vals.x + vals.y + vals.z;

    // --- Step 8: Write threadgroup total to partials ---
    // Last thread (255) has the highest base_offset; its base_offset + thread_total
    // equals the total sum of all elements in this threadgroup.
    threadgroup uint tg_total_shared;
    if (tid_in_tg == SCAN_THREADS_PER_TG - 1) {
        tg_total_shared = base_offset + thread_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_in_tg == 0) {
        partials[tg_idx] = tg_total_shared;
    }
}


// ============================================================================
// scan_partials -- Scan the partials array (single threadgroup)
// ============================================================================
//
// Buffer layout:
//   buffer(0): partials (uint array) -- in-place exclusive scan
//   buffer(1): ScanParams (element_count = number of partials)
//
// Dispatch: 1 threadgroup of 256 threads.
// Handles up to 1024 partials (256 threads * 4 elements each).

kernel void scan_partials(
    device uint*             partials        [[buffer(0)]],
    constant ScanParams&     params          [[buffer(1)]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_idx                            [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint simd_totals[8];

    uint n = params.element_count;  // number of partials

    // --- Step 1: Load 4 elements per thread ---
    uint base = tid_in_tg * 4;
    uint4 vals;
    vals.x = (base     < n) ? partials[base]     : 0;
    vals.y = (base + 1 < n) ? partials[base + 1] : 0;
    vals.z = (base + 2 < n) ? partials[base + 2] : 0;
    vals.w = (base + 3 < n) ? partials[base + 3] : 0;

    // --- Step 2: Compute per-thread total ---
    uint thread_total = vals.x + vals.y + vals.z + vals.w;

    // --- Step 3: SIMD prefix sum ---
    uint simd_prefix = simd_prefix_exclusive_sum(thread_total);

    // --- Step 4: Store SIMD group totals ---
    if (simd_lane == 31) {
        simd_totals[simd_idx] = simd_prefix + thread_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 5: Cross-SIMD prefix scan ---
    if (tid_in_tg < 8) {
        simd_totals[tid_in_tg] = simd_prefix_exclusive_sum(simd_totals[tid_in_tg]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 6: Compute base offset ---
    uint base_offset = simd_prefix + simd_totals[simd_idx];

    // --- Step 7: Write 4 exclusive prefix outputs back to partials ---
    if (base     < n) partials[base]     = base_offset;
    if (base + 1 < n) partials[base + 1] = base_offset + vals.x;
    if (base + 2 < n) partials[base + 2] = base_offset + vals.x + vals.y;
    if (base + 3 < n) partials[base + 3] = base_offset + vals.x + vals.y + vals.z;
}


// ============================================================================
// scan_add_offsets -- Add partials prefix to each element
// ============================================================================
//
// Buffer layout:
//   buffer(0): output data (uint array) -- in-place update
//   buffer(1): partials (uint array, scanned) -- prefix to add
//   buffer(2): ScanParams (element_count = total elements)
//
// Dispatch: ceil(N / 1024) threadgroups of 256 threads.
// Each threadgroup adds partials[tg_idx] to its 1024 elements (4 per thread).

kernel void scan_add_offsets(
    device uint*             output          [[buffer(0)]],
    device const uint*       partials        [[buffer(1)]],
    constant ScanParams&     params          [[buffer(2)]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_idx                              [[threadgroup_position_in_grid]]
) {
    uint offset = partials[tg_idx];

    uint base = tg_idx * SCAN_ELEMENTS_PER_TG + tid_in_tg * 4;

    if (base     < params.element_count) output[base]     += offset;
    if (base + 1 < params.element_count) output[base + 1] += offset;
    if (base + 2 < params.element_count) output[base + 2] += offset;
    if (base + 3 < params.element_count) output[base + 3] += offset;
}
