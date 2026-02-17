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
    uint tg_idx                              [[threadgroup_position_in_grid]]
) {
    // Shared memory to store each thread's digit for stable offset computation.
    threadgroup uchar shared_digits[SORT_THREADS_PER_TG];

    uint num_tgs = params.num_threadgroups;
    uint gid = tg_idx * SORT_THREADS_PER_TG + tid_in_tg;
    bool valid = (gid < params.element_count);

    uint key = 0;
    uint digit = RADIX_BINS; // sentinel for invalid threads

    if (valid) {
        key = keys_in[gid];
        digit = (key >> params.bit_offset) & 0xF;
    }

    // Store digit in shared memory
    shared_digits[tid_in_tg] = (uchar)digit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        // Count how many threads before this one have the same digit.
        // This gives a stable local offset.
        uint local_pos = 0;
        for (uint j = 0; j < tid_in_tg; j++) {
            if (shared_digits[j] == (uchar)digit) {
                local_pos++;
            }
        }

        // Global position = scanned prefix for (digit, tg_idx) + local offset
        // Digit-major layout: scanned_hist[digit * num_tg + tg_idx]
        uint global_base = scanned_hist[digit * num_tgs + tg_idx];
        uint global_pos = global_base + local_pos;
        keys_out[global_pos] = key;
    }
}
