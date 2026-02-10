// compact.metal -- Stream compaction: bitmask to dense row indices.
//
// Converts a 1-bit-per-row selection bitmask into a dense array of selected
// row indices. Used when downstream operations (ORDER BY, SELECT *) need
// actual row data, not just aggregates.
//
// Algorithm: each thread handles one bitmask word (32 rows).
// 1. Count set bits in this word (popcount)
// 2. Determine output position via exclusive prefix sum (atomic counter)
// 3. Write selected row indices to output array
//
// This is a simple atomic-based approach suitable for the POC. A work-efficient
// parallel prefix scan [KB #193] can be added later for 10M+ row performance.

#include "types.h"

// Buffer layout:
//   buffer(0): input bitmask (uint32 array, num_words elements)
//   buffer(1): output indices (uint32 array, up to total_rows elements)
//   buffer(2): output_count (atomic uint32, total selected rows written)
//   buffer(3): total_rows (uint32 constant, for bounds checking)
//   buffer(4): num_words (uint32 constant)

kernel void compact_selection(
    device const uint*   bitmask      [[buffer(0)]],
    device uint*         out_indices  [[buffer(1)]],
    device atomic_uint*  output_count [[buffer(2)]],
    constant uint&       total_rows   [[buffer(3)]],
    constant uint&       num_words    [[buffer(4)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= num_words) return;

    uint word = bitmask[tid];
    if (word == 0) return;

    uint bits = popcount(word);

    // Reserve output slots atomically
    uint base_out = atomic_fetch_add_explicit(output_count, bits, memory_order_relaxed);

    // Write selected row indices
    uint row_base = tid * 32;
    uint out_idx = base_out;

    // Extract set bits and write corresponding row indices
    while (word != 0) {
        // Find lowest set bit
        uint bit_pos = ctz(word);
        uint row = row_base + bit_pos;

        if (row < total_rows) {
            out_indices[out_idx] = row;
            out_idx++;
        }

        // Clear the lowest set bit
        word &= (word - 1);
    }
}
