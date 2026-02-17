// histogram.metal -- GPU histogram kernels for forge-primitives.
//
// histogram_256: shared-memory 256-bin histogram using threadgroup atomics.
//   Each threadgroup accumulates into local 256-bin histogram (1KB threadgroup memory),
//   then merges to global histogram via atomic_fetch_add.
//
// For 65536+ bins: threadgroup memory would exceed 32KB limit on Apple Silicon.
//   A tiled multi-pass approach would be needed. Noted as future enhancement.
//
// Dispatch: ceil(element_count / 256) threadgroups of 256 threads.

#include "types.h"

#define HISTOGRAM_TG_SIZE 256


// ============================================================================
// histogram_256 -- shared-memory 256-bin histogram
// ============================================================================
//
// Buffer layout:
//   buffer(0): input data (uint array, N elements)
//   buffer(1): output histogram (atomic_uint array, 256 bins)
//   buffer(2): HistogramParams
//
// Each threadgroup:
//   1. Initialize local_hist[256] to zero
//   2. Accumulate: bin = input[gid] % num_bins
//   3. Merge local_hist to global histogram via atomic add
//
// Threadgroup memory: 256 * sizeof(atomic_uint) = 1KB

kernel void histogram_256(
    device const uint*        input           [[buffer(0)]],
    device atomic_uint*       global_hist     [[buffer(1)]],
    constant HistogramParams& params          [[buffer(2)]],
    uint tid                                  [[thread_position_in_grid]],
    uint tid_in_tg                            [[thread_position_in_threadgroup]],
    uint tg_size                              [[threads_per_threadgroup]]
) {
    // --- Step 1: Initialize shared-memory histogram to zero ---
    threadgroup atomic_uint local_hist[256];

    for (uint i = tid_in_tg; i < 256; i += tg_size) {
        atomic_store_explicit(&local_hist[i], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 2: Accumulate into local histogram ---
    if (tid < params.element_count) {
        uint value = input[tid];
        uint bin = value % params.num_bins;
        atomic_fetch_add_explicit(&local_hist[bin], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 3: Merge local histogram to global ---
    for (uint i = tid_in_tg; i < params.num_bins; i += tg_size) {
        uint count = atomic_load_explicit(&local_hist[i], memory_order_relaxed);
        if (count > 0) {
            atomic_fetch_add_explicit(&global_hist[i], count, memory_order_relaxed);
        }
    }
}
