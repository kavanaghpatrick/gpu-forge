#include <metal_stdlib>
using namespace metal;

// =================================================================
// forge-filter: GPU filter + compact kernels
//
// 4 kernels:
//   1. filter_predicate_scan — fused predicate eval + local scan + partials
//   2. filter_scan_partials  — single-TG exclusive scan of partials array
//   3. filter_scatter        — re-eval + local scan + global prefix scatter
//   4. filter_atomic_scatter — unordered single-dispatch atomic scatter
//
// Supports 32-bit (uint/int/float) and 64-bit (ulong/long/double) types
// via IS_64BIT function constant. Predicate type eliminated at PSO
// compile time via PRED_TYPE function constant.
// =================================================================

// --- Defines ---

#define FILTER_THREADS       256u
#define FILTER_ELEMS_32      16u    // elements per thread (32-bit types)
#define FILTER_ELEMS_64      8u     // elements per thread (64-bit types)
#define FILTER_TILE_32       4096u  // FILTER_THREADS * FILTER_ELEMS_32
#define FILTER_TILE_64       2048u  // FILTER_THREADS * FILTER_ELEMS_64
#define FILTER_NUM_SGS       8u     // FILTER_THREADS / 32
#define SCAN_ELEMS_PER_THREAD 16u   // elements per thread in scan_partials

// --- Function Constants ---
// Specialized at PSO creation time — all branches eliminated by compiler.

constant uint PRED_TYPE   [[function_constant(0)]]; // 0=GT,1=LT,2=GE,3=LE,4=EQ,5=NE,6=BETWEEN,7=TRUE
constant bool IS_64BIT    [[function_constant(1)]]; // false=32-bit, true=64-bit
constant bool OUTPUT_IDX  [[function_constant(2)]]; // true=write indices to output
constant bool OUTPUT_VALS [[function_constant(3)]]; // true=write values to output
constant uint DATA_TYPE   [[function_constant(4)]]; // 0=uint/ulong, 1=int/long, 2=float/double

// Resolved with defaults (used when constants not set)
constant uint pred_type   = is_function_constant_defined(PRED_TYPE)   ? PRED_TYPE   : 0u;
constant bool is_64bit    = is_function_constant_defined(IS_64BIT)    ? IS_64BIT    : false;
constant bool output_idx  = is_function_constant_defined(OUTPUT_IDX)  ? OUTPUT_IDX  : false;
constant bool output_vals = is_function_constant_defined(OUTPUT_VALS) ? OUTPUT_VALS : true;
constant uint data_type   = is_function_constant_defined(DATA_TYPE)   ? DATA_TYPE   : 0u;

// --- Parameter Structs (must match Rust repr(C) layout) ---

struct FilterParams {
    uint element_count;
    uint num_tiles;
    uint lo_bits;
    uint hi_bits;
};

struct FilterParams64 {
    uint element_count;
    uint num_tiles;
    uint lo_lo;    // low 32 bits of lo threshold
    uint lo_hi;    // high 32 bits of lo threshold
    uint hi_lo;    // low 32 bits of hi threshold
    uint hi_hi;    // high 32 bits of hi threshold
    uint _pad0;
    uint _pad1;
};

// --- Predicate Evaluation ---
// All branches eliminated at PSO compile time via pred_type constant.

template<typename T>
inline bool evaluate_predicate(T val, T lo, T hi) {
    if (pred_type == 0u) return val > lo;              // GT
    if (pred_type == 1u) return val < lo;              // LT
    if (pred_type == 2u) return val >= lo;             // GE
    if (pred_type == 3u) return val <= lo;             // LE
    if (pred_type == 4u) return val == lo;             // EQ
    if (pred_type == 5u) return val != lo;             // NE
    if (pred_type == 6u) return val >= lo && val <= hi; // BETWEEN
    return true; // 7 = TRUE (passthrough)
}

// --- IEEE 754 f64 comparison via raw ulong bits ---
// Metal has no `double` type, so we compare f64 bit patterns directly.
// NaN handling: f64 NaN has exponent=0x7FF and nonzero mantissa.
// We follow IEEE 754: NaN compares unordered (returns false for <, >, <=, >=, ==).
// NaN != NaN returns true.

inline bool is_f64_nan(ulong bits) {
    // Exponent = bits[62:52] = 0x7FF, mantissa != 0
    ulong exp_mask = 0x7FF0000000000000ul;
    ulong mantissa_mask = 0x000FFFFFFFFFFFFFul;
    return (bits & exp_mask) == exp_mask && (bits & mantissa_mask) != 0ul;
}

// Compare two f64 values stored as raw ulong bits.
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
// For NaN: returns -2 (unordered)
inline int f64_compare(ulong a_bits, ulong b_bits) {
    if (is_f64_nan(a_bits) || is_f64_nan(b_bits)) return -2; // unordered

    bool a_neg = (a_bits >> 63u) != 0u;
    bool b_neg = (b_bits >> 63u) != 0u;

    // Handle negative zero == positive zero
    ulong a_abs = a_bits & 0x7FFFFFFFFFFFFFFFul;
    ulong b_abs = b_bits & 0x7FFFFFFFFFFFFFFFul;
    if (a_abs == 0ul && b_abs == 0ul) return 0; // -0.0 == +0.0

    if (a_neg != b_neg) {
        // Different signs: negative < positive
        return a_neg ? -1 : 1;
    }
    if (!a_neg) {
        // Both positive: larger bits = larger value
        return (a_bits > b_bits) ? 1 : ((a_bits < b_bits) ? -1 : 0);
    }
    // Both negative: larger bits = more negative = smaller value
    return (a_bits > b_bits) ? -1 : ((a_bits < b_bits) ? 1 : 0);
}

inline bool evaluate_predicate_f64(ulong val, ulong lo, ulong hi) {
    if (pred_type == 5u) {
        // NE: NaN != anything = true, also NaN != NaN = true
        if (is_f64_nan(val) || is_f64_nan(lo)) return true;
        // -0.0 == +0.0
        ulong v_abs = val & 0x7FFFFFFFFFFFFFFFul;
        ulong l_abs = lo & 0x7FFFFFFFFFFFFFFFul;
        if (v_abs == 0ul && l_abs == 0ul) return false;
        return val != lo;
    }

    int cmp_lo = f64_compare(val, lo);
    if (cmp_lo == -2) return false; // NaN → unordered → false for all ordered predicates

    if (pred_type == 0u) return cmp_lo > 0;   // GT
    if (pred_type == 1u) return cmp_lo < 0;   // LT
    if (pred_type == 2u) return cmp_lo >= 0;  // GE
    if (pred_type == 3u) return cmp_lo <= 0;  // LE
    if (pred_type == 4u) return cmp_lo == 0;  // EQ

    if (pred_type == 6u) {
        // BETWEEN: val >= lo && val <= hi
        if (cmp_lo < 0) return false;
        int cmp_hi = f64_compare(val, hi);
        if (cmp_hi == -2) return false;
        return cmp_hi <= 0;
    }
    return true; // 7 = TRUE
}

// =================================================================
// Kernel 1: filter_predicate_scan
//
// Evaluate predicate for each element, compute per-thread match count,
// SIMD prefix sum within each simdgroup, cross-SG aggregation,
// write TG total to partials[tg_idx].
//
// Dispatch: ceil(N / TILE_SIZE) threadgroups x 256 threads
// TG memory: 32 bytes (simd_totals[8])
// =================================================================

kernel void filter_predicate_scan(
    device const uint*       input        [[buffer(0)]],
    device uint*             partials     [[buffer(1)]],
    constant void*           params_raw   [[buffer(2)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint tg_idx    [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_idx  [[simdgroup_index_in_threadgroup]]
)
{
    threadgroup uint simd_totals[FILTER_NUM_SGS];

    uint thread_match_count = 0u;

    if (is_64bit) {
        // --- 64-bit path ---
        constant FilterParams64& params = *reinterpret_cast<constant FilterParams64*>(params_raw);
        uint n = params.element_count;
        uint base = tg_idx * FILTER_TILE_64;

        // Reconstruct 64-bit thresholds from split halves
        ulong lo_raw = ulong(params.lo_lo) | (ulong(params.lo_hi) << 32u);
        ulong hi_raw = ulong(params.hi_lo) | (ulong(params.hi_hi) << 32u);

        device const ulong* input64 = reinterpret_cast<device const ulong*>(input);

        for (uint e = 0u; e < FILTER_ELEMS_64; e++) {
            uint idx = base + e * FILTER_THREADS + lid;
            if (idx < n) {
                ulong raw = input64[idx];
                if (data_type == 1u) {
                    // Signed long comparison
                    thread_match_count += uint(evaluate_predicate(
                        as_type<long>(raw), as_type<long>(lo_raw), as_type<long>(hi_raw)));
                } else if (data_type == 2u) {
                    // f64 comparison via raw bit ordering (Metal has no double type)
                    thread_match_count += uint(evaluate_predicate_f64(raw, lo_raw, hi_raw));
                } else {
                    // Unsigned ulong comparison
                    thread_match_count += uint(evaluate_predicate(raw, lo_raw, hi_raw));
                }
            }
        }
    } else {
        // --- 32-bit path ---
        constant FilterParams& params = *reinterpret_cast<constant FilterParams*>(params_raw);
        uint n = params.element_count;
        uint base = tg_idx * FILTER_TILE_32;
        uint lo_u = params.lo_bits;
        uint hi_u = params.hi_bits;

        for (uint e = 0u; e < FILTER_ELEMS_32; e++) {
            uint idx = base + e * FILTER_THREADS + lid;
            if (idx < n) {
                uint raw = input[idx];
                if (data_type == 1u) {
                    // Signed int comparison
                    thread_match_count += uint(evaluate_predicate(
                        as_type<int>(raw), as_type<int>(lo_u), as_type<int>(hi_u)));
                } else if (data_type == 2u) {
                    // Float comparison (IEEE 754 — NaN comparisons follow standard)
                    thread_match_count += uint(evaluate_predicate(
                        as_type<float>(raw), as_type<float>(lo_u), as_type<float>(hi_u)));
                } else {
                    // Unsigned uint comparison
                    thread_match_count += uint(evaluate_predicate(raw, lo_u, hi_u));
                }
            }
        }
    }

    // --- SIMD prefix sum ---
    uint simd_prefix = simd_prefix_exclusive_sum(thread_match_count);

    // Last lane in each simdgroup writes total to shared memory
    if (simd_lane == 31u) {
        simd_totals[simd_idx] = simd_prefix + thread_match_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Cross-SG scan (first 8 threads) ---
    if (lid < FILTER_NUM_SGS) {
        simd_totals[lid] = simd_prefix_exclusive_sum(simd_totals[lid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Last thread writes TG total to partials ---
    if (lid == FILTER_THREADS - 1u) {
        uint tg_total = simd_prefix + thread_match_count + simd_totals[simd_idx];
        partials[tg_idx] = tg_total;
    }
}

// =================================================================
// Kernel 2: filter_scan_partials
//
// Single-threadgroup exclusive scan of the partials array.
// 256 threads x SCAN_ELEMS_PER_THREAD(16) = 4096 partials max.
// Writes exclusive prefix sums back to partials[].
// Writes grand total to count_out[0].
//
// Dispatch: 1 threadgroup x 256 threads
// TG memory: 32 bytes (simd_totals[8])
// =================================================================

kernel void filter_scan_partials(
    device uint*      partials   [[buffer(0)]],
    constant uint&    num_parts  [[buffer(1)]],
    device uint*      count_out  [[buffer(2)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_idx  [[simdgroup_index_in_threadgroup]]
)
{
    threadgroup uint simd_totals[FILTER_NUM_SGS];

    uint n = num_parts;

    // Each thread loads SCAN_ELEMS_PER_THREAD elements
    uint vals[SCAN_ELEMS_PER_THREAD];
    uint thread_total = 0u;
    for (uint i = 0u; i < SCAN_ELEMS_PER_THREAD; i++) {
        uint idx = lid * SCAN_ELEMS_PER_THREAD + i;
        uint v = (idx < n) ? partials[idx] : 0u;
        vals[i] = v;
        thread_total += v;
    }

    // SIMD prefix sum of per-thread totals
    uint simd_prefix = simd_prefix_exclusive_sum(thread_total);

    if (simd_lane == 31u) {
        simd_totals[simd_idx] = simd_prefix + thread_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG scan
    if (lid < FILTER_NUM_SGS) {
        simd_totals[lid] = simd_prefix_exclusive_sum(simd_totals[lid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute this thread's base offset
    uint base_offset = simd_prefix + simd_totals[simd_idx];

    // Write exclusive prefix sums back
    uint running = base_offset;
    for (uint i = 0u; i < SCAN_ELEMS_PER_THREAD; i++) {
        uint idx = lid * SCAN_ELEMS_PER_THREAD + i;
        if (idx < n) {
            uint orig = vals[i];
            partials[idx] = running;
            running += orig;
        }
    }

    // Last thread writes grand total to count_out[0]
    if (lid == FILTER_THREADS - 1u) {
        uint grand_total = base_offset + thread_total;
        count_out[0] = grand_total;
    }
}

// =================================================================
// Kernel 3: filter_scatter
//
// Re-evaluate predicate, recompute local scan, read global prefix
// from scanned partials[tg_idx], scatter matching elements to
// output_vals and/or output_idx (controlled by function constants).
//
// Dispatch: ceil(N / TILE_SIZE) threadgroups x 256 threads
// TG memory: 32 bytes (simd_totals[8])
// =================================================================

kernel void filter_scatter(
    device const uint*       input        [[buffer(0)]],
    device uint*             out_vals     [[buffer(1)]],
    device uint*             out_idx      [[buffer(2)]],
    device const uint*       partials     [[buffer(3)]],
    constant void*           params_raw   [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint tg_idx    [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_idx  [[simdgroup_index_in_threadgroup]]
)
{
    threadgroup uint simd_totals[FILTER_NUM_SGS];
    threadgroup uint round_total_tg; // separate from simd_totals to avoid aliasing

    // Read global prefix for this TG
    uint global_prefix = partials[tg_idx];

    // Running write offset — accumulates across element rounds
    uint running_offset = global_prefix;

    if (is_64bit) {
        // --- 64-bit scatter path (ordered, round-by-round) ---
        constant FilterParams64& params = *reinterpret_cast<constant FilterParams64*>(params_raw);
        uint n = params.element_count;
        uint base = tg_idx * FILTER_TILE_64;

        ulong lo_raw = ulong(params.lo_lo) | (ulong(params.lo_hi) << 32u);
        ulong hi_raw = ulong(params.hi_lo) | (ulong(params.hi_hi) << 32u);

        device const ulong* input64 = reinterpret_cast<device const ulong*>(input);
        device ulong* out_vals64 = reinterpret_cast<device ulong*>(out_vals);

        for (uint e = 0u; e < FILTER_ELEMS_64; e++) {
            uint idx = base + e * FILTER_THREADS + lid;
            bool valid = idx < n;
            ulong raw = valid ? input64[idx] : 0ul;
            bool match_flag;
            if (data_type == 1u) {
                match_flag = valid && evaluate_predicate(
                    as_type<long>(raw), as_type<long>(lo_raw), as_type<long>(hi_raw));
            } else if (data_type == 2u) {
                match_flag = valid && evaluate_predicate_f64(raw, lo_raw, hi_raw);
            } else {
                match_flag = valid && evaluate_predicate(raw, lo_raw, hi_raw);
            }
            uint flag = uint(match_flag);

            // TG-wide prefix sum of the per-element flag
            uint simd_prefix = simd_prefix_exclusive_sum(flag);
            if (simd_lane == 31u) {
                simd_totals[simd_idx] = simd_prefix + flag;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < FILTER_NUM_SGS) {
                simd_totals[lid] = simd_prefix_exclusive_sum(simd_totals[lid]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint write_pos = running_offset + simd_prefix + simd_totals[simd_idx];

            if (match_flag) {
                if (output_vals) out_vals64[write_pos] = raw;
                if (output_idx)  out_idx[write_pos] = idx;
            }

            // Last thread computes round total
            if (lid == FILTER_THREADS - 1u) {
                round_total_tg = write_pos + flag - running_offset;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            running_offset += round_total_tg;
        }
    } else {
        // --- 32-bit scatter path (ordered, round-by-round) ---
        constant FilterParams& params = *reinterpret_cast<constant FilterParams*>(params_raw);
        uint n = params.element_count;
        uint base = tg_idx * FILTER_TILE_32;
        uint lo_u = params.lo_bits;
        uint hi_u = params.hi_bits;

        for (uint e = 0u; e < FILTER_ELEMS_32; e++) {
            uint idx = base + e * FILTER_THREADS + lid;
            bool valid = idx < n;
            uint raw = valid ? input[idx] : 0u;
            bool match_flag;
            if (data_type == 1u) {
                match_flag = valid && evaluate_predicate(
                    as_type<int>(raw), as_type<int>(lo_u), as_type<int>(hi_u));
            } else if (data_type == 2u) {
                match_flag = valid && evaluate_predicate(
                    as_type<float>(raw), as_type<float>(lo_u), as_type<float>(hi_u));
            } else {
                match_flag = valid && evaluate_predicate(raw, lo_u, hi_u);
            }
            uint flag = uint(match_flag);

            // TG-wide prefix sum of the per-element flag
            uint simd_prefix = simd_prefix_exclusive_sum(flag);
            if (simd_lane == 31u) {
                simd_totals[simd_idx] = simd_prefix + flag;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < FILTER_NUM_SGS) {
                simd_totals[lid] = simd_prefix_exclusive_sum(simd_totals[lid]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint write_pos = running_offset + simd_prefix + simd_totals[simd_idx];

            if (match_flag) {
                if (output_vals) out_vals[write_pos] = raw;
                if (output_idx)  out_idx[write_pos] = idx;
            }

            // Last thread computes round total
            if (lid == FILTER_THREADS - 1u) {
                round_total_tg = write_pos + flag - running_offset;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            running_offset += round_total_tg;
        }
    }
}

// =================================================================
// Kernel 4: filter_atomic_scatter
//
// Single-dispatch unordered mode. Evaluate predicate, SIMD-aggregated
// atomic scatter. Output order is non-deterministic but correct set.
//
// Uses simd_sum + simd_prefix_exclusive_sum + atomic_fetch_add per SG
// lane 0 + simd_broadcast_first — only 1 atomic per 32 threads.
//
// Dispatch: ceil(N / TILE_SIZE) threadgroups x 256 threads
// TG memory: none (all via SIMD intrinsics + device atomic)
// =================================================================

kernel void filter_atomic_scatter(
    device const uint*       input        [[buffer(0)]],
    device uint*             out_vals     [[buffer(1)]],
    device uint*             out_idx      [[buffer(2)]],
    device atomic_uint*      counter      [[buffer(3)]],
    constant void*           params_raw   [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_idx  [[simdgroup_index_in_threadgroup]]
)
{
    if (is_64bit) {
        // --- 64-bit atomic scatter path ---
        constant FilterParams64& params = *reinterpret_cast<constant FilterParams64*>(params_raw);
        uint n = params.element_count;
        uint base = gid * FILTER_TILE_64;

        ulong lo_raw = ulong(params.lo_lo) | (ulong(params.lo_hi) << 32u);
        ulong hi_raw = ulong(params.hi_lo) | (ulong(params.hi_hi) << 32u);

        device const ulong* input64 = reinterpret_cast<device const ulong*>(input);
        device ulong* out_vals64 = reinterpret_cast<device ulong*>(out_vals);

        // Phase 1: Evaluate predicate, cache results
        uint local_count = 0u;
        bool flags[FILTER_ELEMS_64];
        ulong values[FILTER_ELEMS_64];
        uint idx_cache[FILTER_ELEMS_64];

        for (uint e = 0u; e < FILTER_ELEMS_64; e++) {
            uint idx = base + e * FILTER_THREADS + lid;
            bool valid = idx < n;
            ulong raw = valid ? input64[idx] : 0ul;
            bool match_v;
            if (data_type == 1u) {
                match_v = valid && evaluate_predicate(
                    as_type<long>(raw), as_type<long>(lo_raw), as_type<long>(hi_raw));
            } else if (data_type == 2u) {
                match_v = valid && evaluate_predicate_f64(raw, lo_raw, hi_raw);
            } else {
                match_v = valid && evaluate_predicate(raw, lo_raw, hi_raw);
            }
            flags[e] = match_v;
            values[e] = raw;
            idx_cache[e] = idx;
            local_count += uint(match_v);
        }

        // Phase 2: SIMD-aggregated atomic
        uint simd_total = simd_sum(local_count);
        uint simd_prefix = simd_prefix_exclusive_sum(local_count);
        uint simd_base = 0u;
        if (simd_lane == 0u && simd_total > 0u) {
            simd_base = atomic_fetch_add_explicit(counter, simd_total, memory_order_relaxed);
        }
        simd_base = simd_broadcast_first(simd_base);

        // Phase 3: Scatter
        uint write_pos = simd_base + simd_prefix;
        for (uint e = 0u; e < FILTER_ELEMS_64; e++) {
            if (flags[e]) {
                if (output_vals) out_vals64[write_pos] = values[e];
                if (output_idx)  out_idx[write_pos] = idx_cache[e];
                write_pos++;
            }
        }
    } else {
        // --- 32-bit atomic scatter path ---
        constant FilterParams& params = *reinterpret_cast<constant FilterParams*>(params_raw);
        uint n = params.element_count;
        uint base = gid * FILTER_TILE_32;
        uint lo_u = params.lo_bits;
        uint hi_u = params.hi_bits;

        // Phase 1: Evaluate predicate, cache results
        uint local_count = 0u;
        bool flags[FILTER_ELEMS_32];
        uint values[FILTER_ELEMS_32];
        uint idx_cache[FILTER_ELEMS_32];

        for (uint e = 0u; e < FILTER_ELEMS_32; e++) {
            uint idx = base + e * FILTER_THREADS + lid;
            bool valid = idx < n;
            uint raw = valid ? input[idx] : 0u;
            bool match_v;
            if (data_type == 1u) {
                match_v = valid && evaluate_predicate(
                    as_type<int>(raw), as_type<int>(lo_u), as_type<int>(hi_u));
            } else if (data_type == 2u) {
                match_v = valid && evaluate_predicate(
                    as_type<float>(raw), as_type<float>(lo_u), as_type<float>(hi_u));
            } else {
                match_v = valid && evaluate_predicate(raw, lo_u, hi_u);
            }
            flags[e] = match_v;
            values[e] = raw;
            idx_cache[e] = idx;
            local_count += uint(match_v);
        }

        // Phase 2: SIMD-aggregated atomic
        uint simd_total = simd_sum(local_count);
        uint simd_prefix = simd_prefix_exclusive_sum(local_count);
        uint simd_base = 0u;
        if (simd_lane == 0u && simd_total > 0u) {
            simd_base = atomic_fetch_add_explicit(counter, simd_total, memory_order_relaxed);
        }
        simd_base = simd_broadcast_first(simd_base);

        // Phase 3: Scatter
        uint write_pos = simd_base + simd_prefix;
        for (uint e = 0u; e < FILTER_ELEMS_32; e++) {
            if (flags[e]) {
                if (output_vals) out_vals[write_pos] = values[e];
                if (output_idx)  out_idx[write_pos]  = idx_cache[e];
                write_pos++;
            }
        }
    }
}
