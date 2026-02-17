// histogram.metal -- GPU histogram kernels for forge-primitives.
//
// histogram_256: shared-memory 256-bin histogram using threadgroup atomics.
//   Each thread loads 4 elements via uint4 vectorized loads, uses bitmask
//   binning for power-of-2 bin counts (modulo fallback for non-power-of-2).
//   Accumulates into local 256-bin histogram (1KB threadgroup memory),
//   then merges to global histogram via atomic_fetch_add.
//
// For 65536+ bins: threadgroup memory would exceed 32KB limit on Apple Silicon.
//   A tiled multi-pass approach would be needed. Noted as future enhancement.
//
// Dispatch: ceil(element_count / (256 * 4)) threadgroups of 256 threads.

#include "types.h"

#define HISTOGRAM_TG_SIZE 256


// ============================================================================
// histogram_256 -- shared-memory 256-bin histogram with uint4 vectorized loads
// ============================================================================
//
// Buffer layout:
//   buffer(0): input data (uint array, N elements)
//   buffer(1): output histogram (atomic_uint array, 256 bins)
//   buffer(2): HistogramParams
//
// Each thread processes 4 elements via load_uint4_safe.
// Bitmask binning for power-of-2 bin counts (& mask instead of % modulo).
// Non-power-of-2 fallback uses modulo.
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

    for (int i = tid_in_tg; i < 256; i += tg_size) {
        atomic_store_explicit(&local_hist[i], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 2: Compute bitmask (power-of-2) or fallback to modulo ---
    uint num_bins = params.num_bins;
    bool is_power_of_2 = (num_bins & (num_bins - 1)) == 0;
    uint mask = num_bins - 1;

    // --- Step 3: Load 4 elements and accumulate into local histogram ---
    uint4 vals = load_uint4_safe(input, tid * 4, params.element_count);

    // Accumulate each element individually with bounds check
    uint base = tid * 4;
    if (base < params.element_count) {
        uint bin = is_power_of_2 ? (vals.x & mask) : (vals.x % num_bins);
        atomic_fetch_add_explicit(&local_hist[bin], 1, memory_order_relaxed);
    }
    if (base + 1 < params.element_count) {
        uint bin = is_power_of_2 ? (vals.y & mask) : (vals.y % num_bins);
        atomic_fetch_add_explicit(&local_hist[bin], 1, memory_order_relaxed);
    }
    if (base + 2 < params.element_count) {
        uint bin = is_power_of_2 ? (vals.z & mask) : (vals.z % num_bins);
        atomic_fetch_add_explicit(&local_hist[bin], 1, memory_order_relaxed);
    }
    if (base + 3 < params.element_count) {
        uint bin = is_power_of_2 ? (vals.w & mask) : (vals.w % num_bins);
        atomic_fetch_add_explicit(&local_hist[bin], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 4: Merge local histogram to global ---
    for (int i = tid_in_tg; i < (int)params.num_bins; i += tg_size) {
        uint count = atomic_load_explicit(&local_hist[i], memory_order_relaxed);
        if (count > 0) {
            atomic_fetch_add_explicit(&global_hist[i], count, memory_order_relaxed);
        }
    }
}
