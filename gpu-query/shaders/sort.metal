// sort.metal -- Placeholder radix sort kernels for ORDER BY.
//
// GPU radix sort is complex; these are stub implementations for the POC.
// Actual sorting is done CPU-side in the executor. These kernels will be
// GPU-accelerated in a future optimization pass.
//
// Algorithm (future): 4-bit radix sort, 16 passes for 64-bit keys.
// Three-phase per pass: histogram -> prefix scan -> scatter.

#include "types.h"

/// Placeholder: Compute per-digit histogram for radix sort.
///
/// Future implementation: Each thread inspects one key, extracts the 4-bit
/// digit at `params.bit_offset`, and atomically increments the corresponding
/// histogram bin. 16 bins per threadgroup, merged globally.
kernel void radix_sort_histogram(
    device const long* keys          [[buffer(0)]],
    device atomic_uint* histogram    [[buffer(1)]],
    device const SortParams& params  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Stub: no-op. CPU-side sort is used instead.
    if (tid >= params.element_count) return;
}

/// Placeholder: Exclusive prefix scan over histogram bins.
///
/// Future implementation: Blelloch-style work-efficient scan over the
/// 16-bin histogram to compute scatter offsets for each digit value.
kernel void radix_sort_scan(
    device uint* histogram           [[buffer(0)]],
    device uint* scan_output         [[buffer(1)]],
    device const SortParams& params  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Stub: no-op. CPU-side sort is used instead.
    if (tid >= 16) return;
}

/// Placeholder: Scatter keys to sorted positions using prefix scan offsets.
///
/// Future implementation: Each thread reads one key, computes its digit,
/// looks up the scatter offset from the scanned histogram, and writes
/// the key (and its original index) to the output position.
/// For DESC order, keys are XOR'd with 0xFFFFFFFFFFFFFFFF before sorting.
kernel void radix_sort_scatter(
    device const long* keys_in       [[buffer(0)]],
    device long* keys_out            [[buffer(1)]],
    device const uint* scan_offsets  [[buffer(2)]],
    device const SortParams& params  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Stub: no-op. CPU-side sort is used instead.
    if (tid >= params.element_count) return;
}
