// filter.metal -- GPU column filter kernel (WHERE clause).
//
// Uses Metal function constants for compile-time specialization of comparison
// operator, column type, and null-check behavior. This eliminates all runtime
// branching in the hot path, yielding ~84% instruction reduction [KB #210, #202].
//
// Output: 1-bit-per-row selection bitmask + atomic match_count.
// Bitmask layout: bit (row % 32) of word (row / 32) in uint32 array.

#include "types.h"

// ---- Function constants for compile-time specialization ----
// These are set via MTLFunctionConstantValues on the host side.
// The Metal compiler generates a specialized variant for each combination.

constant uint COMPARE_OP   [[function_constant(0)]];    // 0=EQ, 1=NE, 2=LT, 3=LE, 4=GT, 5=GE
constant uint COLUMN_TYPE  [[function_constant(1)]];    // 0=INT64, 1=FLOAT64
constant bool HAS_NULL_CHECK [[function_constant(2)]];  // true if null bitmap present

// ---- INT64 comparison (specialized at compile time) ----

static inline bool compare_int64(long value, long threshold) {
    if (COMPARE_OP == 0) return value == threshold;      // EQ
    if (COMPARE_OP == 1) return value != threshold;      // NE
    if (COMPARE_OP == 2) return value <  threshold;      // LT
    if (COMPARE_OP == 3) return value <= threshold;      // LE
    if (COMPARE_OP == 4) return value >  threshold;      // GT
    if (COMPARE_OP == 5) return value >= threshold;      // GE
    return false;
}

// ---- FLOAT comparison (specialized at compile time) ----
// Metal lacks double; we reinterpret the 64-bit float bits as a long,
// then convert to float for comparison. This loses precision but is the
// only option on Metal GPUs.

static inline bool compare_float(float value, long threshold_bits) {
    // Reinterpret the stored bits as a float value.
    // The host side stores the f64 value cast to f32 bits in the lower 32 bits.
    float threshold = as_type<float>(static_cast<int>(threshold_bits));

    if (COMPARE_OP == 0) return value == threshold;      // EQ
    if (COMPARE_OP == 1) return value != threshold;      // NE
    if (COMPARE_OP == 2) return value <  threshold;      // LT
    if (COMPARE_OP == 3) return value <= threshold;      // LE
    if (COMPARE_OP == 4) return value >  threshold;      // GT
    if (COMPARE_OP == 5) return value >= threshold;      // GE
    return false;
}

// ---- Column filter kernel ----
//
// Each thread processes one row. Writes a 1-bit selection bitmask and
// accumulates match_count using simd_sum + atomic_add for efficiency.
//
// Buffer layout:
//   buffer(0): column data (long* for INT64, float* for FLOAT64)
//   buffer(1): output bitmask (uint32 array, 1 bit per row)
//   buffer(2): output match_count (atomic uint32)
//   buffer(3): FilterParams (comparison threshold + metadata)
//   buffer(4): null bitmap (uint32 array, 1 bit per row) -- only if HAS_NULL_CHECK

kernel void column_filter(
    device const void*       column_data   [[buffer(0)]],
    device atomic_uint*      bitmask       [[buffer(1)]],
    device atomic_uint*      match_count   [[buffer(2)]],
    constant FilterParams&   params        [[buffer(3)]],
    device const uint*       null_bitmap   [[buffer(4)]],
    uint tid                               [[thread_position_in_grid]],
    uint simd_lane                         [[thread_index_in_simdgroup]]
) {
    uint row_count = params.row_count;
    if (tid >= row_count) {
        return;
    }

    // Check null bitmap if present
    if (HAS_NULL_CHECK) {
        uint word_idx = tid / 32;
        uint bit_idx = tid % 32;
        uint null_word = null_bitmap[word_idx];
        // Bit set = null -> skip this row (no match)
        if ((null_word >> bit_idx) & 1u) {
            return;
        }
    }

    // Evaluate the comparison based on column type (compile-time branch)
    bool match_result = false;

    if (COLUMN_TYPE == 0) {
        // INT64 column
        device const long* int_data = (device const long*)column_data;
        long value = int_data[tid];
        match_result = compare_int64(value, params.compare_value_int);
    } else if (COLUMN_TYPE == 1) {
        // FLOAT32 column (Metal uses float, not double)
        device const float* float_data = (device const float*)column_data;
        float value = float_data[tid];
        match_result = compare_float(value, params.compare_value_float_bits);
    }

    // Write to bitmask using atomic_or (multiple threads may write same uint32 word)
    if (match_result) {
        uint word_idx = tid / 32;
        uint bit_idx = tid % 32;
        atomic_fetch_or_explicit(&bitmask[word_idx], (1u << bit_idx), memory_order_relaxed);
    }

    // Count matches using simd_sum for efficient partial reduction,
    // then one atomic_add per simdgroup instead of per-thread.
    uint match_val = match_result ? 1u : 0u;
    uint simd_total = simd_sum(match_val);

    // Lane 0 of each simdgroup writes the partial sum
    if (simd_lane == 0 && simd_total > 0) {
        atomic_fetch_add_explicit(match_count, simd_total, memory_order_relaxed);
    }
}
