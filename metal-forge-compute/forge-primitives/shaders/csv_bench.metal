// csv_bench.metal -- GPU CSV parsing kernels for forge-bench.
//
// Two-kernel approach for parsing raw CSV byte data:
//   1. csv_newline_detect: each thread checks one byte for '\n', writes 1/0 flag
//   2. csv_field_count: uses threadgroup-local atomic counting of commas and
//      newlines, then merges to global counters.
//
// This demonstrates GPU text/byte-stream processing -- a workload traditionally
// considered CPU-bound due to branch divergence.

#include "types.h"

// ============================================================================
// csv_newline_detect -- Per-byte newline flag generation
// ============================================================================
//
// Buffer layout:
//   buffer(0): input bytes (uchar array, byte_count elements)
//   buffer(1): output flags (uint array, byte_count elements) -- 1 if '\n', else 0
//   buffer(2): CsvBenchParams
//
// Dispatch: ceil(byte_count / 256) threadgroups of 256 threads.

kernel void csv_newline_detect(
    device const uchar*       input       [[buffer(0)]],
    device uint*              flags       [[buffer(1)]],
    constant CsvBenchParams&  params      [[buffer(2)]],
    uint gid                              [[thread_position_in_grid]]
) {
    if (gid >= params.byte_count) return;
    flags[gid] = (input[gid] == '\n') ? 1u : 0u;
}


// ============================================================================
// csv_field_count -- Count commas (fields) and newlines (rows) in CSV data
// ============================================================================
//
// Uses threadgroup-level atomic accumulation then global atomic merge.
// Each thread processes one byte, checking for ',' and '\n'.
//
// Buffer layout:
//   buffer(0): input bytes (uchar array, byte_count elements)
//   buffer(1): output counters (uint array, 2 elements: [0]=comma_count, [1]=newline_count)
//   buffer(2): CsvBenchParams
//
// Dispatch: ceil(byte_count / 256) threadgroups of 256 threads.

kernel void csv_field_count(
    device const uchar*       input       [[buffer(0)]],
    device atomic_uint*       counters    [[buffer(1)]],
    constant CsvBenchParams&  params      [[buffer(2)]],
    uint gid                              [[thread_position_in_grid]],
    uint tid_in_tg                        [[thread_position_in_threadgroup]]
) {
    // Threadgroup-local accumulators to reduce global atomic contention
    threadgroup atomic_uint local_commas;
    threadgroup atomic_uint local_newlines;

    if (tid_in_tg == 0) {
        atomic_store_explicit(&local_commas, 0, memory_order_relaxed);
        atomic_store_explicit(&local_newlines, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < params.byte_count) {
        uchar c = input[gid];
        if (c == ',') {
            atomic_fetch_add_explicit(&local_commas, 1, memory_order_relaxed);
        } else if (c == '\n') {
            atomic_fetch_add_explicit(&local_newlines, 1, memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 merges threadgroup totals into global counters
    if (tid_in_tg == 0) {
        uint commas = atomic_load_explicit(&local_commas, memory_order_relaxed);
        uint newlines = atomic_load_explicit(&local_newlines, memory_order_relaxed);
        if (commas > 0) {
            atomic_fetch_add_explicit(&counters[0], commas, memory_order_relaxed);
        }
        if (newlines > 0) {
            atomic_fetch_add_explicit(&counters[1], newlines, memory_order_relaxed);
        }
    }
}
