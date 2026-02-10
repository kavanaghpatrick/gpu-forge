// aggregate.metal -- GPU aggregation kernels (COUNT/SUM) with hierarchical reduction.
//
// 3-level reduction pattern [KB #188]:
//   Level 1: simd_sum across SIMD group (32 lanes on Apple Silicon)
//   Level 2: threadgroup reduction via shared memory + barrier
//   Level 3: global atomic accumulation
//
// Reads selection_mask (1-bit per row bitmask) from filter output to only
// aggregate selected rows. If all bits are set, acts as unfiltered aggregation.
//
// Note: Metal's simd_shuffle_down does NOT support long/int64. For 64-bit
// reductions we split into lo/hi uint halves, simd-reduce each, and recombine.

#include "types.h"

// Maximum threadgroup size for reduction. Must match host dispatch.
#define MAX_THREADS_PER_TG 256

// ---- Helper: 64-bit SIMD reduction via 32-bit halves ----
// Metal simd_shuffle_down only works on 32-bit types. We split the long into
// lo and hi uint, reduce each independently, then recombine.

static inline long simd_sum_int64(long value, uint simd_lane) {
    // Metal simd_shuffle_down only supports 32-bit types.
    // Split int64 into lo/hi uint halves, reduce with carry tracking.
    int2 val;
    val.x = static_cast<int>(static_cast<ulong>(value));         // lo
    val.y = static_cast<int>(static_cast<ulong>(value) >> 32);   // hi

    for (ushort offset = 16; offset > 0; offset >>= 1) {
        int2 other;
        other.x = simd_shuffle_down(val.x, offset);
        other.y = simd_shuffle_down(val.y, offset);

        // 64-bit add: lo + lo, if overflow -> carry into hi
        uint a_lo = static_cast<uint>(val.x);
        uint b_lo = static_cast<uint>(other.x);
        uint new_lo = a_lo + b_lo;
        uint carry = (new_lo < a_lo) ? 1u : 0u;

        val.x = static_cast<int>(new_lo);
        val.y = val.y + other.y + static_cast<int>(carry);
    }

    // Recombine
    ulong result = static_cast<ulong>(static_cast<uint>(val.x))
                 | (static_cast<ulong>(static_cast<uint>(val.y)) << 32);
    return static_cast<long>(result);
}


// ---- aggregate_count ----
//
// Counts the number of set bits in the selection_mask.
// Each thread handles one uint32 bitmask word -> popcount() for bits set.
//
// Buffer layout:
//   buffer(0): selection_mask (uint32 array, 1 bit per row)
//   buffer(1): output result (atomic uint)
//   buffer(2): AggParams { row_count, group_count, agg_function, _pad0 }
//
// Dispatch: one thread per bitmask word = ceil(row_count / 32) threads.

kernel void aggregate_count(
    device const uint*       selection_mask  [[buffer(0)]],
    device atomic_uint*      result          [[buffer(1)]],
    constant AggParams&      params          [[buffer(2)]],
    uint tid                                 [[thread_position_in_grid]],
    uint tg_idx                              [[threadgroup_position_in_grid]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_size                             [[threads_per_threadgroup]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint num_words = (params.row_count + 31) / 32;

    // --- Level 0: per-thread popcount ---
    uint local_count = 0;
    if (tid < num_words) {
        uint word = selection_mask[tid];

        // Mask off bits beyond row_count in the last word
        if (tid == num_words - 1) {
            uint valid_bits = params.row_count % 32;
            if (valid_bits != 0) {
                word &= (1u << valid_bits) - 1u;
            }
        }

        local_count = popcount(word);
    }

    // --- Level 1: SIMD reduction ---
    uint simd_total = simd_sum(local_count);

    // --- Level 2: threadgroup reduction via shared memory ---
    threadgroup uint tg_partials[MAX_THREADS_PER_TG / 32];  // one slot per simdgroup

    if (simd_lane == 0) {
        tg_partials[simd_group_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces all partial sums
    uint tg_total = 0;
    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        uint val = (simd_lane < num_simd_groups) ? tg_partials[simd_lane] : 0;
        tg_total = simd_sum(val);
    }

    // --- Level 3: global atomic ---
    if (tid_in_tg == 0 && tg_total > 0) {
        atomic_fetch_add_explicit(result, tg_total, memory_order_relaxed);
    }
}


// ---- aggregate_sum_int64 ----
//
// Computes SUM of int64 column values for selected rows (set bits in selection_mask).
// Each thread processes one row: checks its bit in selection_mask, adds value if set.
//
// Buffer layout:
//   buffer(0): column data (long* = int64)
//   buffer(1): selection_mask (uint32 array, 1 bit per row)
//   buffer(2): output result lo (atomic_uint, lower 32 bits of int64 result)
//   buffer(3): output result hi (atomic_uint, upper 32 bits of int64 result)
//   buffer(4): AggParams { row_count, group_count, agg_function, _pad0 }
//
// Dispatch: one thread per row = row_count threads.
// Uses dispatchThreadgroups (not dispatchThreads) so tg_size is exact.

kernel void aggregate_sum_int64(
    device const long*       column_data     [[buffer(0)]],
    device const uint*       selection_mask  [[buffer(1)]],
    device atomic_uint*      result_lo       [[buffer(2)]],
    device atomic_uint*      result_hi       [[buffer(3)]],
    constant AggParams&      params          [[buffer(4)]],
    uint tid                                 [[thread_position_in_grid]],
    uint tg_idx                              [[threadgroup_position_in_grid]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_size                             [[threads_per_threadgroup]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // --- Level 0: per-thread value ---
    long local_sum = 0;

    if (tid < params.row_count) {
        // Check if this row is selected in the bitmask
        uint word_idx = tid / 32;
        uint bit_idx = tid % 32;
        uint word = selection_mask[word_idx];
        bool selected = ((word >> bit_idx) & 1u) != 0;

        if (selected) {
            local_sum = column_data[tid];
        }
    }

    // --- Level 1: SIMD reduction (64-bit via int2 halves) ---
    long simd_total = simd_sum_int64(local_sum, simd_lane);

    // --- Level 2: threadgroup reduction via shared memory ---
    // Store as two uint arrays to avoid alignment issues with long.
    threadgroup uint tg_lo[MAX_THREADS_PER_TG / 32];
    threadgroup uint tg_hi[MAX_THREADS_PER_TG / 32];

    if (simd_lane == 0) {
        ulong val_u = static_cast<ulong>(simd_total);
        tg_lo[simd_group_id] = static_cast<uint>(val_u);
        tg_hi[simd_group_id] = static_cast<uint>(val_u >> 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces all partial sums
    long tg_total = 0;
    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        long val = 0;
        if (simd_lane < num_simd_groups) {
            ulong v = static_cast<ulong>(tg_lo[simd_lane])
                    | (static_cast<ulong>(tg_hi[simd_lane]) << 32);
            val = static_cast<long>(v);
        }
        tg_total = simd_sum_int64(val, simd_lane);
    }

    // --- Level 3: global atomic (split into lo/hi uint32) ---
    if (tid_in_tg == 0 && tg_total != 0) {
        ulong val_u = static_cast<ulong>(tg_total);
        uint lo = static_cast<uint>(val_u);
        uint hi = static_cast<uint>(val_u >> 32);

        // Add low part; if it wraps, carry into high part
        uint old_lo = atomic_fetch_add_explicit(result_lo, lo, memory_order_relaxed);
        ulong sum_lo = static_cast<ulong>(old_lo) + static_cast<ulong>(lo);
        uint carry = (sum_lo > 0xFFFFFFFF) ? 1u : 0u;

        atomic_fetch_add_explicit(result_hi, hi + carry, memory_order_relaxed);
    }
}
