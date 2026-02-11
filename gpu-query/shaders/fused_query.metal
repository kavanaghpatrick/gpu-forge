// fused_query.metal -- Single-pass fused filter+aggregate+GROUP BY kernel.
//
// This kernel performs filtering, bucketing, aggregation, and output in a single
// pass over the data. Each thread processes one row: evaluate predicates, bucket
// into threadgroup-local accumulators, simd-reduce within simdgroup, merge across
// simdgroups, then atomically merge threadgroup results into the global output.
//
// Function constants (set via MTLFunctionConstantValues on the host):
//   FILTER_COUNT: number of active filter predicates (0..4)
//   AGG_COUNT:    number of active aggregate functions (0..5)
//   HAS_GROUP_BY: whether GROUP BY is active
//
// Buffer layout:
//   buffer(0): QueryParamsSlot  (512 bytes, contains filters, aggs, group_by)
//   buffer(1): column data      (contiguous binary columnar data)
//   buffer(2): ColumnMeta[]     (per-column metadata: offset, type, stride)
//   buffer(3): OutputBuffer     (result: group_keys, agg_results, ready_flag)
//
// Dispatch: one thread per row, threadgroup size = 256.

#include "autonomous_types.h"

// --- Threadgroup size ---
#define THREADGROUP_SIZE 256

// --- Sentinel values for MIN/MAX identity elements ---
#define INT64_MAX_VAL  0x7FFFFFFFFFFFFFFF
#define INT64_MIN_VAL  0x8000000000000000
#define FLOAT_MAX_VAL  3.402823466e+38f
#define FLOAT_MIN_VAL  (-3.402823466e+38f)

// --- Function constants for specialization ---
constant uint FILTER_COUNT [[function_constant(0)]];
constant uint AGG_COUNT    [[function_constant(1)]];
constant bool HAS_GROUP_BY [[function_constant(2)]];

// ============================================================================
// SIMD helper functions (64-bit reductions via 32-bit halves)
// Metal's simd_shuffle_down does NOT support long/int64.
// ============================================================================

static inline long fused_simd_sum_int64(long value, uint simd_lane) {
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

    ulong result = static_cast<ulong>(static_cast<uint>(val.x))
                 | (static_cast<ulong>(static_cast<uint>(val.y)) << 32);
    return static_cast<long>(result);
}

static inline long fused_simd_min_int64(long value, uint simd_lane) {
    int2 val;
    val.x = static_cast<int>(static_cast<ulong>(value));
    val.y = static_cast<int>(static_cast<ulong>(value) >> 32);

    for (ushort offset = 16; offset > 0; offset >>= 1) {
        int2 other;
        other.x = simd_shuffle_down(val.x, offset);
        other.y = simd_shuffle_down(val.y, offset);

        bool other_less = (other.y < val.y) ||
                          (other.y == val.y && static_cast<uint>(other.x) < static_cast<uint>(val.x));
        if (other_less) {
            val.x = other.x;
            val.y = other.y;
        }
    }

    ulong result = static_cast<ulong>(static_cast<uint>(val.x))
                 | (static_cast<ulong>(static_cast<uint>(val.y)) << 32);
    return static_cast<long>(result);
}

static inline long fused_simd_max_int64(long value, uint simd_lane) {
    int2 val;
    val.x = static_cast<int>(static_cast<ulong>(value));
    val.y = static_cast<int>(static_cast<ulong>(value) >> 32);

    for (ushort offset = 16; offset > 0; offset >>= 1) {
        int2 other;
        other.x = simd_shuffle_down(val.x, offset);
        other.y = simd_shuffle_down(val.y, offset);

        bool other_greater = (other.y > val.y) ||
                             (other.y == val.y && static_cast<uint>(other.x) > static_cast<uint>(val.x));
        if (other_greater) {
            val.x = other.x;
            val.y = other.y;
        }
    }

    ulong result = static_cast<ulong>(static_cast<uint>(val.x))
                 | (static_cast<ulong>(static_cast<uint>(val.y)) << 32);
    return static_cast<long>(result);
}

// ============================================================================
// Column data reading helpers
// ============================================================================

// Read an int64 value from the column data buffer using ColumnMeta
static inline long read_int64(device const char* data, device const ColumnMeta& meta, uint row) {
    device const long* col = (device const long*)(data + meta.offset);
    return col[row];
}

// Read a float32 value from the column data buffer using ColumnMeta
static inline float read_float32(device const char* data, device const ColumnMeta& meta, uint row) {
    device const float* col = (device const float*)(data + meta.offset);
    return col[row];
}

// Read a uint32 dictionary code from the column data buffer using ColumnMeta
static inline uint read_dict_u32(device const char* data, device const ColumnMeta& meta, uint row) {
    device const uint* col = (device const uint*)(data + meta.offset);
    return col[row];
}

// ============================================================================
// Filter evaluation
// ============================================================================

// Evaluate a single filter predicate against a row. Returns true if the row passes.
static inline bool evaluate_filter(
    device const char* data,
    device const ColumnMeta* columns,
    const FilterSpec filter,
    uint row
) {
    uint col_idx = filter.column_idx;
    uint col_type = filter.column_type;
    uint op = filter.compare_op;

    if (col_type == COLUMN_TYPE_INT64) {
        long value = read_int64(data, columns[col_idx], row);
        long threshold = filter.value_int;

        if (op == COMPARE_OP_EQ) return value == threshold;
        if (op == COMPARE_OP_NE) return value != threshold;
        if (op == COMPARE_OP_LT) return value <  threshold;
        if (op == COMPARE_OP_LE) return value <= threshold;
        if (op == COMPARE_OP_GT) return value >  threshold;
        if (op == COMPARE_OP_GE) return value >= threshold;
    } else if (col_type == COLUMN_TYPE_FLOAT32) {
        float value = read_float32(data, columns[col_idx], row);
        float threshold = as_type<float>(filter.value_float_bits);

        if (op == COMPARE_OP_EQ) return value == threshold;
        if (op == COMPARE_OP_NE) return value != threshold;
        if (op == COMPARE_OP_LT) return value <  threshold;
        if (op == COMPARE_OP_LE) return value <= threshold;
        if (op == COMPARE_OP_GT) return value >  threshold;
        if (op == COMPARE_OP_GE) return value >= threshold;
    } else if (col_type == COLUMN_TYPE_DICT_U32) {
        uint value = read_dict_u32(data, columns[col_idx], row);
        uint threshold = static_cast<uint>(filter.value_int);

        if (op == COMPARE_OP_EQ) return value == threshold;
        if (op == COMPARE_OP_NE) return value != threshold;
        if (op == COMPARE_OP_LT) return value <  threshold;
        if (op == COMPARE_OP_LE) return value <= threshold;
        if (op == COMPARE_OP_GT) return value >  threshold;
        if (op == COMPARE_OP_GE) return value >= threshold;
    }

    return false;
}

// ============================================================================
// Threadgroup-local accumulator for GROUP BY bucketing
// ============================================================================

struct GroupAccumulator {
    long  count;        // row count for this group
    long  sum_int;      // integer sum accumulator
    float sum_float;    // float sum accumulator
    long  min_int;      // integer min
    long  max_int;      // integer max
    float min_float;    // float min
    float max_float;    // float max
    long  group_key;    // the group key value
    uint  valid;        // 1 if this bucket has data
    uint  _pad[3];      // align to 64 bytes
};
// sizeof(GroupAccumulator) = 64 bytes
// threadgroup GroupAccumulator[64] = 4096 bytes (well under 32KB limit)

// ============================================================================
// Device atomic helpers for 64-bit values (split lo/hi)
// ============================================================================

// Atomically add a 64-bit value using two 32-bit atomics with carry propagation.
static inline void atomic_add_int64(
    device atomic_uint* lo_ptr,
    device atomic_uint* hi_ptr,
    long value
) {
    ulong val_u = static_cast<ulong>(value);
    uint lo = static_cast<uint>(val_u);
    uint hi = static_cast<uint>(val_u >> 32);

    uint old_lo = atomic_fetch_add_explicit(lo_ptr, lo, memory_order_relaxed);
    ulong sum_lo = static_cast<ulong>(old_lo) + static_cast<ulong>(lo);
    uint carry = (sum_lo > 0xFFFFFFFFUL) ? 1u : 0u;

    atomic_fetch_add_explicit(hi_ptr, hi + carry, memory_order_relaxed);
}

// Atomically update MIN of a 64-bit signed value via CAS loop on lo/hi uint pair.
// This stores the value as (lo, hi) in two adjacent atomic_uint slots.
// Uses a combined 64-bit comparison approach.
static inline void atomic_min_int64(
    device atomic_uint* lo_ptr,
    device atomic_uint* hi_ptr,
    long new_val
) {
    ulong new_u = static_cast<ulong>(new_val);
    uint new_lo = static_cast<uint>(new_u);
    uint new_hi = static_cast<uint>(new_u >> 32);

    // CAS loop on the hi part first for sign comparison
    for (int i = 0; i < 64; i++) {
        uint cur_hi = atomic_load_explicit(hi_ptr, memory_order_relaxed);
        uint cur_lo = atomic_load_explicit(lo_ptr, memory_order_relaxed);

        // Reconstruct current value as signed long
        long cur_val = static_cast<long>(
            static_cast<ulong>(cur_lo) | (static_cast<ulong>(cur_hi) << 32)
        );

        if (new_val >= cur_val) return;  // Current is already <= new, nothing to do

        // Try to swap hi first
        if (atomic_compare_exchange_weak_explicit(hi_ptr, &cur_hi, new_hi,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) {
            // Hi swapped successfully, now swap lo
            // (Not perfectly atomic across both words, but sufficient for convergence)
            atomic_store_explicit(lo_ptr, new_lo, memory_order_relaxed);
            return;
        }
        // CAS failed, retry
    }
}

// Atomically update MAX of a 64-bit signed value via CAS loop.
static inline void atomic_max_int64(
    device atomic_uint* lo_ptr,
    device atomic_uint* hi_ptr,
    long new_val
) {
    ulong new_u = static_cast<ulong>(new_val);
    uint new_lo = static_cast<uint>(new_u);
    uint new_hi = static_cast<uint>(new_u >> 32);

    for (int i = 0; i < 64; i++) {
        uint cur_hi = atomic_load_explicit(hi_ptr, memory_order_relaxed);
        uint cur_lo = atomic_load_explicit(lo_ptr, memory_order_relaxed);

        long cur_val = static_cast<long>(
            static_cast<ulong>(cur_lo) | (static_cast<ulong>(cur_hi) << 32)
        );

        if (new_val <= cur_val) return;  // Current is already >= new, nothing to do

        if (atomic_compare_exchange_weak_explicit(hi_ptr, &cur_hi, new_hi,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) {
            atomic_store_explicit(lo_ptr, new_lo, memory_order_relaxed);
            return;
        }
    }
}

// Atomically add a float value via CAS loop (reinterpret float as uint bits).
static inline void atomic_add_float(device atomic_uint* ptr, float value) {
    if (value == 0.0f) return;
    uint expected = atomic_load_explicit(ptr, memory_order_relaxed);
    for (int i = 0; i < 64; i++) {
        float cur = as_type<float>(expected);
        float desired = cur + value;
        uint desired_bits = as_type<uint>(desired);
        if (atomic_compare_exchange_weak_explicit(ptr, &expected, desired_bits,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) {
            return;
        }
    }
}

// Atomically update MIN of a float via CAS loop.
static inline void atomic_min_float(device atomic_uint* ptr, float value) {
    uint expected = atomic_load_explicit(ptr, memory_order_relaxed);
    for (int i = 0; i < 64; i++) {
        float cur = as_type<float>(expected);
        if (value >= cur) return;
        uint desired_bits = as_type<uint>(value);
        if (atomic_compare_exchange_weak_explicit(ptr, &expected, desired_bits,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) {
            return;
        }
    }
}

// Atomically update MAX of a float via CAS loop.
static inline void atomic_max_float(device atomic_uint* ptr, float value) {
    uint expected = atomic_load_explicit(ptr, memory_order_relaxed);
    for (int i = 0; i < 64; i++) {
        float cur = as_type<float>(expected);
        if (value <= cur) return;
        uint desired_bits = as_type<uint>(value);
        if (atomic_compare_exchange_weak_explicit(ptr, &expected, desired_bits,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) {
            return;
        }
    }
}

// ============================================================================
// Main fused query kernel
// ============================================================================

kernel void fused_query(
    device const QueryParamsSlot* params   [[buffer(0)]],
    device const char*            data     [[buffer(1)]],
    device const ColumnMeta*      columns  [[buffer(2)]],
    device OutputBuffer*          output   [[buffer(3)]],
    uint tid       [[thread_position_in_grid]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]]
) {
    // ---- Phase 0: Initialize threadgroup-local accumulators ----
    threadgroup GroupAccumulator accum[MAX_GROUPS];

    // Each thread initializes a subset of the accumulators
    uint groups_per_thread = (MAX_GROUPS + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    for (uint i = 0; i < groups_per_thread; i++) {
        uint g = lid * groups_per_thread + i;
        if (g < MAX_GROUPS) {
            accum[g].count     = 0;
            accum[g].sum_int   = 0;
            accum[g].sum_float = 0.0f;
            accum[g].min_int   = static_cast<long>(INT64_MAX_VAL);
            accum[g].max_int   = static_cast<long>(INT64_MIN_VAL);
            accum[g].min_float = FLOAT_MAX_VAL;
            accum[g].max_float = FLOAT_MIN_VAL;
            accum[g].group_key = 0;
            accum[g].valid     = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Read query parameters ----
    uint row_count = params->row_count;

    // ---- Phase 1: FILTER ----
    // Each thread processes one row. Out-of-range threads skip.
    bool row_passes = (tid < row_count);

    if (row_passes && FILTER_COUNT > 0) {
        // Evaluate all filter predicates (AND compound)
        if (FILTER_COUNT >= 1) {
            row_passes = row_passes && evaluate_filter(data, columns, params->filters[0], tid);
        }
        if (FILTER_COUNT >= 2) {
            row_passes = row_passes && evaluate_filter(data, columns, params->filters[1], tid);
        }
        if (FILTER_COUNT >= 3) {
            row_passes = row_passes && evaluate_filter(data, columns, params->filters[2], tid);
        }
        if (FILTER_COUNT >= 4) {
            row_passes = row_passes && evaluate_filter(data, columns, params->filters[3], tid);
        }
    }

    // ---- Phase 2: GROUP BY BUCKETING ----
    uint bucket = 0;
    if (row_passes && HAS_GROUP_BY) {
        uint group_col = params->group_by_col;
        uint group_type = columns[group_col].column_type;

        long group_val = 0;
        if (group_type == COLUMN_TYPE_INT64) {
            group_val = read_int64(data, columns[group_col], tid);
        } else if (group_type == COLUMN_TYPE_FLOAT32) {
            // Use float bits as integer for bucketing
            float fval = read_float32(data, columns[group_col], tid);
            group_val = static_cast<long>(as_type<uint>(fval));
        } else if (group_type == COLUMN_TYPE_DICT_U32) {
            group_val = static_cast<long>(read_dict_u32(data, columns[group_col], tid));
        }

        // Simple modular hash into MAX_GROUPS buckets
        // For dictionary codes (0..N), this gives direct indexing when N < MAX_GROUPS
        bucket = static_cast<uint>(
            (group_val >= 0 ? group_val : -group_val) % MAX_GROUPS
        );

        if (row_passes) {
            accum[bucket].group_key = group_val;
            accum[bucket].valid = 1;
        }
    } else if (row_passes) {
        // No GROUP BY: all rows go into bucket 0
        bucket = 0;
        accum[0].valid = 1;
    }

    // ---- Phase 3: AGGREGATE (threadgroup-local per-thread accumulation) ----
    // Each passing thread accumulates into its group's accumulator.
    // We do this per-thread first, then simd-reduce, then merge across simdgroups.

    // Per-thread local accumulators (to avoid excessive threadgroup atomics)
    long local_count = row_passes ? 1 : 0;

    // For each active aggregate, read the column value and accumulate
    // We store per-thread values and then reduce within simdgroup.

    // We need per-agg local values. Since AGG_COUNT <= 5, unroll manually.
    long  local_sum_int[MAX_AGGS];
    float local_sum_float[MAX_AGGS];
    long  local_min_int[MAX_AGGS];
    long  local_max_int[MAX_AGGS];
    float local_min_float[MAX_AGGS];
    float local_max_float[MAX_AGGS];

    for (uint a = 0; a < MAX_AGGS; a++) {
        local_sum_int[a]   = 0;
        local_sum_float[a] = 0.0f;
        local_min_int[a]   = static_cast<long>(INT64_MAX_VAL);
        local_max_int[a]   = static_cast<long>(INT64_MIN_VAL);
        local_min_float[a] = FLOAT_MAX_VAL;
        local_max_float[a] = FLOAT_MIN_VAL;
    }

    if (row_passes) {
        for (uint a = 0; a < AGG_COUNT; a++) {
            uint agg_func  = params->aggs[a].agg_func;
            uint agg_col   = params->aggs[a].column_idx;
            uint agg_type  = params->aggs[a].column_type;

            if (agg_func == AGG_FUNC_COUNT) {
                // COUNT doesn't need column data, just increment
                local_sum_int[a] = 1;
            } else if (agg_type == COLUMN_TYPE_INT64) {
                long val = read_int64(data, columns[agg_col], tid);
                if (agg_func == AGG_FUNC_SUM || agg_func == AGG_FUNC_AVG) {
                    local_sum_int[a] = val;
                }
                if (agg_func == AGG_FUNC_MIN) {
                    local_min_int[a] = val;
                }
                if (agg_func == AGG_FUNC_MAX) {
                    local_max_int[a] = val;
                }
            } else if (agg_type == COLUMN_TYPE_FLOAT32) {
                float val = read_float32(data, columns[agg_col], tid);
                if (agg_func == AGG_FUNC_SUM || agg_func == AGG_FUNC_AVG) {
                    local_sum_float[a] = val;
                }
                if (agg_func == AGG_FUNC_MIN) {
                    local_min_float[a] = val;
                }
                if (agg_func == AGG_FUNC_MAX) {
                    local_max_float[a] = val;
                }
            }
        }
    }

    // ---- SIMD reduction within each simdgroup ----
    long  simd_count = fused_simd_sum_int64(local_count, simd_lane);
    uint  simd_bucket = simd_broadcast_first(bucket);

    long  simd_sum_int_arr[MAX_AGGS];
    float simd_sum_float_arr[MAX_AGGS];
    long  simd_min_int_arr[MAX_AGGS];
    long  simd_max_int_arr[MAX_AGGS];
    float simd_min_float_arr[MAX_AGGS];
    float simd_max_float_arr[MAX_AGGS];

    for (uint a = 0; a < MAX_AGGS; a++) {
        simd_sum_int_arr[a]   = fused_simd_sum_int64(local_sum_int[a], simd_lane);
        simd_sum_float_arr[a] = simd_sum(local_sum_float[a]);
        simd_min_int_arr[a]   = fused_simd_min_int64(local_min_int[a], simd_lane);
        simd_max_int_arr[a]   = fused_simd_max_int64(local_max_int[a], simd_lane);
        simd_min_float_arr[a] = simd_min(local_min_float[a]);
        simd_max_float_arr[a] = simd_max(local_max_float[a]);
    }

    // ---- Merge simdgroup results into threadgroup accumulators ----
    // Lane 0 of each simdgroup writes to the threadgroup accumulators.
    // Since multiple simdgroups may target different buckets, we serialize
    // access using threadgroup_barrier between simdgroup writes.

    if (simd_lane == 0 && simd_count > 0) {
        // Use the bucket from the first lane (simd_broadcast_first)
        uint b = simd_bucket;

        // Atomic-like threadgroup accumulation (we use barriers to serialize)
        accum[b].count   += simd_count;
        accum[b].valid    = 1;

        for (uint a = 0; a < AGG_COUNT; a++) {
            uint agg_func = params->aggs[a].agg_func;

            if (agg_func == AGG_FUNC_COUNT) {
                accum[b].sum_int += simd_sum_int_arr[a];
            } else if (agg_func == AGG_FUNC_SUM || agg_func == AGG_FUNC_AVG) {
                accum[b].sum_int   += simd_sum_int_arr[a];
                accum[b].sum_float += simd_sum_float_arr[a];
            } else if (agg_func == AGG_FUNC_MIN) {
                if (simd_min_int_arr[a] < accum[b].min_int) {
                    accum[b].min_int = simd_min_int_arr[a];
                }
                if (simd_min_float_arr[a] < accum[b].min_float) {
                    accum[b].min_float = simd_min_float_arr[a];
                }
            } else if (agg_func == AGG_FUNC_MAX) {
                if (simd_max_int_arr[a] > accum[b].max_int) {
                    accum[b].max_int = simd_max_int_arr[a];
                }
                if (simd_max_float_arr[a] > accum[b].max_float) {
                    accum[b].max_float = simd_max_float_arr[a];
                }
            }
        }
    }

    // Wait for all simdgroups in this threadgroup to finish writing
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 4: GLOBAL REDUCTION (cross-threadgroup merge via device atomics) ----
    // Only thread 0 of each threadgroup merges its local accumulators into global output.
    // This serializes per-threadgroup but keeps global atomics to a minimum.

    if (lid == 0) {
        for (uint g = 0; g < MAX_GROUPS; g++) {
            if (accum[g].valid == 0) continue;

            // Write group key (last writer wins -- all threadgroups compute same key for same bucket)
            output->group_keys[g] = accum[g].group_key;

            // Merge each aggregate
            for (uint a = 0; a < AGG_COUNT; a++) {
                uint agg_func = params->aggs[a].agg_func;
                uint agg_type = params->aggs[a].column_type;

                // Compute byte offsets into the agg_results array
                // agg_results is [MAX_GROUP_SLOTS][MAX_AGGS] = [256][5] of AggResult (16 bytes each)
                // Access: output->agg_results[g][a]

                // Use pointer arithmetic for atomic access to AggResult fields
                // AggResult = { long value_int(8), float value_float(4), uint count(4) } = 16 bytes
                // Base offset of agg_results[g][a]:
                //   output_base + offsetof(OutputBuffer, agg_results) + (g * MAX_AGGS + a) * sizeof(AggResult)
                // offsetof(OutputBuffer, agg_results) = 2080
                // sizeof(AggResult) = 16
                uint agg_result_offset = 2080 + (g * MAX_AGGS + a) * 16;

                device char* out_base = (device char*)output;

                // AggResult.value_int at offset 0 within AggResult (8 bytes = 2 atomic_uint)
                device atomic_uint* val_int_lo = (device atomic_uint*)(out_base + agg_result_offset);
                device atomic_uint* val_int_hi = (device atomic_uint*)(out_base + agg_result_offset + 4);

                // AggResult.value_float at offset 8 within AggResult
                device atomic_uint* val_float = (device atomic_uint*)(out_base + agg_result_offset + 8);

                // AggResult.count at offset 12 within AggResult
                device atomic_uint* val_count = (device atomic_uint*)(out_base + agg_result_offset + 12);

                // Merge count (every aggregate tracks contributing row count)
                atomic_fetch_add_explicit(val_count, static_cast<uint>(accum[g].count), memory_order_relaxed);

                if (agg_func == AGG_FUNC_COUNT) {
                    // COUNT: accumulate count into value_int
                    atomic_add_int64(val_int_lo, val_int_hi, accum[g].count);
                } else if (agg_func == AGG_FUNC_SUM || agg_func == AGG_FUNC_AVG) {
                    if (agg_type == COLUMN_TYPE_INT64) {
                        atomic_add_int64(val_int_lo, val_int_hi, accum[g].sum_int);
                    } else {
                        atomic_add_float(val_float, accum[g].sum_float);
                    }
                } else if (agg_func == AGG_FUNC_MIN) {
                    if (agg_type == COLUMN_TYPE_INT64) {
                        atomic_min_int64(val_int_lo, val_int_hi, accum[g].min_int);
                    } else {
                        atomic_min_float(val_float, accum[g].min_float);
                    }
                } else if (agg_func == AGG_FUNC_MAX) {
                    if (agg_type == COLUMN_TYPE_INT64) {
                        atomic_max_int64(val_int_lo, val_int_hi, accum[g].max_int);
                    } else {
                        atomic_max_float(val_float, accum[g].max_float);
                    }
                }
            }
        }
    }

    // ---- Phase 5: OUTPUT (metadata + ready flag) ----
    // Only the very first thread (threadgroup 0, thread 0) sets result metadata.
    // This must happen AFTER all threadgroups have merged -- we use a
    // device atomic counter to detect when all threadgroups are done.
    //
    // For the AOT kernel with standard dispatch (waitUntilCompleted), the host
    // reads the output after the command buffer completes, so we can simply
    // set metadata from threadgroup 0. The ready_flag is set as a signal for
    // the persistent kernel path (Phase 4 of the spec).

    // Count completed threadgroups via atomic counter in error_code field (repurposed temporarily)
    device atomic_uint* tg_done_counter = (device atomic_uint*)&output->error_code;
    uint total_tgs = (row_count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    // Ensure all writes from this threadgroup are visible
    threadgroup_barrier(mem_flags::mem_device);

    if (lid == 0) {
        uint prev_done = atomic_fetch_add_explicit(tg_done_counter, 1u, memory_order_relaxed);

        // Last threadgroup to finish sets the output metadata
        if (prev_done + 1 == total_tgs) {
            // Count valid groups
            uint group_count = 0;
            if (HAS_GROUP_BY) {
                // Read back from output group_keys to count valid groups
                // Since we merged atomically, scan the device output
                for (uint g = 0; g < MAX_GROUPS; g++) {
                    // Check if any threadgroup wrote to this group by checking
                    // the count in agg_results[g][0].count (first agg's count)
                    if (AGG_COUNT > 0) {
                        uint agg_result_offset = 2080 + (g * MAX_AGGS + 0) * 16;
                        device char* out_base = (device char*)output;
                        device atomic_uint* cnt = (device atomic_uint*)(out_base + agg_result_offset + 12);
                        uint c = atomic_load_explicit(cnt, memory_order_relaxed);
                        if (c > 0) {
                            group_count++;
                        }
                    }
                }
            } else {
                group_count = 1;  // No GROUP BY means single result row
            }

            output->result_row_count = group_count;
            output->result_col_count = params->agg_count;
            output->sequence_id = params->sequence_id;
            output->error_code = 0;  // Reset from counter use

            // Memory fence to ensure all writes above are visible before ready_flag
            threadgroup_barrier(mem_flags::mem_device);

            // Set ready flag LAST -- this signals the host that results are ready
            output->ready_flag = 1;
        }
    }
}
