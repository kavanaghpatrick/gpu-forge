// scan.metal -- 3-pass exclusive prefix scan for forge-primitives.
//
// Implements reduce-then-scan approach (NOT decoupled lookback which deadlocks
// on Apple Silicon).
//
// 3-pass algorithm:
//   Pass 1 (scan_local): Each threadgroup of 256 threads processes 512 elements
//     using Blelloch work-efficient scan in shared memory. Writes threadgroup
//     partial sum to partials[threadgroup_id].
//
//   Pass 2 (scan_partials): Single threadgroup scans the partials array.
//     For large arrays (> 512 partials), host does sequential scan.
//
//   Pass 3 (scan_add_offsets): Each threadgroup adds partials[threadgroup_id]
//     to every element in its chunk.
//
// All kernels use 256 threads per threadgroup.
// Each thread handles 2 elements = 512 elements per threadgroup.

#include "types.h"

#define SCAN_THREADS_PER_TG 256
#define SCAN_ELEMENTS_PER_TG 512  // 2 elements per thread


// ============================================================================
// scan_local -- Per-threadgroup Blelloch exclusive scan
// ============================================================================
//
// Buffer layout:
//   buffer(0): input data (uint array, N elements)
//   buffer(1): output data (uint array, N elements) -- exclusive prefix scan
//   buffer(2): partials (uint array, one per threadgroup) -- threadgroup total sums
//   buffer(3): ScanParams
//
// Dispatch: ceil(N / 512) threadgroups of 256 threads.

kernel void scan_local(
    device const uint*       input           [[buffer(0)]],
    device uint*             output          [[buffer(1)]],
    device uint*             partials        [[buffer(2)]],
    constant ScanParams&     params          [[buffer(3)]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_idx                              [[threadgroup_position_in_grid]]
) {
    threadgroup uint shared[SCAN_ELEMENTS_PER_TG];

    uint base = tg_idx * SCAN_ELEMENTS_PER_TG;
    uint idx0 = base + tid_in_tg * 2;
    uint idx1 = idx0 + 1;

    // Load 2 elements per thread into shared memory
    shared[tid_in_tg * 2]     = (idx0 < params.element_count) ? input[idx0] : 0;
    shared[tid_in_tg * 2 + 1] = (idx1 < params.element_count) ? input[idx1] : 0;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Up-sweep (reduce) phase ---
    for (uint stride = 1; stride < SCAN_ELEMENTS_PER_TG; stride <<= 1) {
        uint index = (tid_in_tg + 1) * (stride << 1) - 1;
        if (index < SCAN_ELEMENTS_PER_TG) {
            shared[index] += shared[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Save total sum and clear last element for exclusive scan
    if (tid_in_tg == 0) {
        uint total = shared[SCAN_ELEMENTS_PER_TG - 1];
        partials[tg_idx] = total;
        shared[SCAN_ELEMENTS_PER_TG - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Down-sweep (distribute) phase ---
    for (uint stride = SCAN_ELEMENTS_PER_TG >> 1; stride >= 1; stride >>= 1) {
        uint index = (tid_in_tg + 1) * (stride << 1) - 1;
        if (index < SCAN_ELEMENTS_PER_TG) {
            uint temp = shared[index - stride];
            shared[index - stride] = shared[index];
            shared[index] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write scanned results back to global memory
    if (idx0 < params.element_count) {
        output[idx0] = shared[tid_in_tg * 2];
    }
    if (idx1 < params.element_count) {
        output[idx1] = shared[tid_in_tg * 2 + 1];
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
// Handles up to 512 partials (256 threads * 2 elements each).

kernel void scan_partials(
    device uint*             partials        [[buffer(0)]],
    constant ScanParams&     params          [[buffer(1)]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared[SCAN_ELEMENTS_PER_TG];

    uint n = params.element_count;  // number of partials

    // Load partials into shared memory
    uint idx0 = tid_in_tg * 2;
    uint idx1 = idx0 + 1;
    shared[idx0] = (idx0 < n) ? partials[idx0] : 0;
    shared[idx1] = (idx1 < n) ? partials[idx1] : 0;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Round up n to next power of 2 for Blelloch
    uint padded_n = 1;
    while (padded_n < n) padded_n <<= 1;
    if (padded_n > SCAN_ELEMENTS_PER_TG) padded_n = SCAN_ELEMENTS_PER_TG;

    // --- Up-sweep ---
    for (uint stride = 1; stride < padded_n; stride <<= 1) {
        uint index = (tid_in_tg + 1) * (stride << 1) - 1;
        if (index < padded_n) {
            shared[index] += shared[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear last for exclusive scan
    if (tid_in_tg == 0) {
        shared[padded_n - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Down-sweep ---
    for (uint stride = padded_n >> 1; stride >= 1; stride >>= 1) {
        uint index = (tid_in_tg + 1) * (stride << 1) - 1;
        if (index < padded_n) {
            uint temp = shared[index - stride];
            shared[index - stride] = shared[index];
            shared[index] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back scanned partials
    if (idx0 < n) {
        partials[idx0] = shared[idx0];
    }
    if (idx1 < n) {
        partials[idx1] = shared[idx1];
    }
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
// Dispatch: ceil(N / 512) threadgroups of 256 threads.
// Each threadgroup adds partials[tg_idx] to its 512 elements.

kernel void scan_add_offsets(
    device uint*             output          [[buffer(0)]],
    device const uint*       partials        [[buffer(1)]],
    constant ScanParams&     params          [[buffer(2)]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_idx                              [[threadgroup_position_in_grid]]
) {
    uint offset = partials[tg_idx];

    uint base = tg_idx * SCAN_ELEMENTS_PER_TG;
    uint idx0 = base + tid_in_tg * 2;
    uint idx1 = idx0 + 1;

    if (idx0 < params.element_count) {
        output[idx0] += offset;
    }
    if (idx1 < params.element_count) {
        output[idx1] += offset;
    }
}
