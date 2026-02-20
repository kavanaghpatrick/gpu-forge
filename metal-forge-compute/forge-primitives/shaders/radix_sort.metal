// radix_sort.metal -- 4-bit radix sort kernels for forge-primitives.
//
// Implements a reduce-then-scan radix sort (NOT OneSweep which deadlocks
// on Apple Silicon).
//
// Algorithm (per pass, 8 passes total for u32, 4 bits per pass):
//   1. radix_histogram: Each threadgroup builds a 16-bin histogram of its
//      chunk using shared memory atomics. Writes per-TG histogram to global
//      array in DIGIT-MAJOR order: histogram[digit * num_tg + tg_id].
//
//   2. (Host dispatches existing scan kernels to prefix-scan the global
//      histogram array of size 16 * num_tg.)
//
//   3. radix_scatter: Each thread reads its key, extracts the 4-bit digit,
//      computes a STABLE local offset by counting elements before it with
//      the same digit (via shared memory), then writes to output using
//      the scanned histogram offset.
//
// Digit-major layout ensures that after scanning, all TG contributions for
// digit 0 are contiguous, then all for digit 1, etc. This gives correct
// global positions for a sorted radix pass.
//
// Double-buffer: host ping-pongs between keys_a and keys_b across passes.
// All kernels use 256 threads per threadgroup.

#include "types.h"

#define SORT_THREADS_PER_TG 256
#define RADIX_BITS 4
#define RADIX_BINS 16  // 2^4

// ============================================================================
// radix_histogram -- Per-threadgroup 16-bin histogram (digit-major output)
// ============================================================================
//
// Buffer layout:
//   buffer(0): input keys (uint array, N elements)
//   buffer(1): global histogram (uint array, 16 * num_tg) -- DIGIT-MAJOR
//   buffer(2): SortParams { element_count, bit_offset, num_threadgroups }
//
// Dispatch: ceil(N / 256) threadgroups of 256 threads.
// Each thread processes 1 element.
//
// Output layout: histogram[digit * num_tg + tg_id] = count of elements
// in threadgroup tg_id with the given digit.

kernel void radix_histogram(
    device const uint*       keys            [[buffer(0)]],
    device uint*             histogram       [[buffer(1)]],
    constant SortParams&     params          [[buffer(2)]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_idx                              [[threadgroup_position_in_grid]]
) {
    // Shared memory for local 16-bin histogram
    threadgroup atomic_uint local_hist[RADIX_BINS];

    uint num_tgs = params.num_threadgroups;

    // Initialize local histogram to zero
    if (tid_in_tg < RADIX_BINS) {
        atomic_store_explicit(&local_hist[tid_in_tg], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute this thread's global index
    uint gid = tg_idx * SORT_THREADS_PER_TG + tid_in_tg;

    // Accumulate digit into local histogram
    if (gid < params.element_count) {
        uint key = keys[gid];
        uint digit = (key >> params.bit_offset) & 0xF;
        atomic_fetch_add_explicit(&local_hist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write local histogram to global in DIGIT-MAJOR order:
    // histogram[digit * num_tg + tg_idx]
    if (tid_in_tg < RADIX_BINS) {
        uint count = atomic_load_explicit(&local_hist[tid_in_tg], memory_order_relaxed);
        histogram[tid_in_tg * num_tgs + tg_idx] = count;
    }
}


// ============================================================================
// radix_scatter -- STABLE scatter using shared-memory digit array
// ============================================================================
//
// Buffer layout:
//   buffer(0): input keys (uint array, N elements)
//   buffer(1): output keys (uint array, N elements)
//   buffer(2): scanned histogram (uint array, 16 * num_tg) -- DIGIT-MAJOR
//   buffer(3): SortParams { element_count, bit_offset, num_threadgroups }
//
// Dispatch: ceil(N / 256) threadgroups of 256 threads.
//
// Stability: Each thread stores its digit in shared memory. After a barrier,
// each thread counts how many threads with LOWER index have the same digit.
// This gives a deterministic, stable local offset.
//
// Global position = scanned_hist[digit * num_tg + tg_id] + local_offset

kernel void radix_scatter(
    device const uint*       keys_in         [[buffer(0)]],
    device uint*             keys_out        [[buffer(1)]],
    device const uint*       scanned_hist    [[buffer(2)]],
    constant SortParams&     params          [[buffer(3)]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_idx                              [[threadgroup_position_in_grid]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // Per-SIMD-group digit counts for cross-SIMD offset computation.
    // Layout: simd_digit_counts[simd_group][digit]
    constexpr uint NUM_SIMD_GROUPS = SORT_THREADS_PER_TG / 32; // 8
    threadgroup uint simd_digit_counts[NUM_SIMD_GROUPS][RADIX_BINS];

    uint num_tgs = params.num_threadgroups;
    uint gid = tg_idx * SORT_THREADS_PER_TG + tid_in_tg;
    bool valid = (gid < params.element_count);

    uint key = 0;
    uint my_digit = RADIX_BINS; // sentinel for invalid threads

    if (valid) {
        key = keys_in[gid];
        my_digit = (key >> params.bit_offset) & 0xF;
    }

    // Phase 1: Compute intra-SIMD rank using SIMD prefix sum.
    // For each of 16 digit values, compute prefix count within this SIMD group.
    uint my_rank_in_simd = 0;
    for (uint d = 0; d < RADIX_BINS; d++) {
        uint match = (my_digit == d) ? 1u : 0u;
        uint prefix = simd_prefix_exclusive_sum(match);
        uint count = simd_sum(match);

        // Record rank for the thread's own digit
        if (my_digit == d) {
            my_rank_in_simd = prefix;
        }

        // Last lane in SIMD group stores the digit count
        if (simd_lane == 31) {
            simd_digit_counts[simd_group_id][d] = count;
        }
    }

    // Synchronize: all SIMD groups must have written their counts
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute cross-SIMD offset.
    // Sum digit counts from all SIMD groups before this one.
    uint cross_offset = 0;
    if (valid) {
        for (uint s = 0; s < simd_group_id; s++) {
            cross_offset += simd_digit_counts[s][my_digit];
        }
    }

    // Phase 3: Global scatter.
    // local_pos = cross_offset + intra-SIMD rank
    // global_pos = scanned_hist[digit * num_tg + tg_idx] + local_pos
    if (valid) {
        uint local_pos = cross_offset + my_rank_in_simd;
        uint global_base = scanned_hist[my_digit * num_tgs + tg_idx];
        uint global_pos = global_base + local_pos;
        keys_out[global_pos] = key;
    }
}
