#include <metal_stdlib>
using namespace metal;

// =================================================================
// forge-filter: GPU filter + compact kernels
//
// 7 kernels:
//   1. filter_predicate_scan      — fused predicate eval + local scan + partials
//   2. filter_scan_partials       — single-TG exclusive scan of partials array
//   3. filter_scatter             — re-eval + local scan + global prefix scatter
//   4. filter_atomic_scatter      — unordered single-dispatch atomic scatter
//   5. filter_bitmap_scan         — predicate eval + bitmap write + partials
//   6. filter_bitmap_scatter      — bitmap read + local scan + global prefix scatter
//   7. filter_multi_bitmap_scan   — multi-column predicate eval + bitmap + partials
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
constant bool HAS_NULLS   [[function_constant(5)]]; // true=check validity bitmap for NULLs

// --- Multi-Column Function Constants (indices 6-19) ---
constant uint N_COLUMNS    [[function_constant(6)]];   // number of active columns (1-4)
constant uint PRED_TYPE_A  [[function_constant(7)]];   // predicate type for column A
constant uint PRED_TYPE_B  [[function_constant(8)]];   // predicate type for column B
constant uint PRED_TYPE_C  [[function_constant(9)]];   // predicate type for column C
constant uint PRED_TYPE_D  [[function_constant(10)]];  // predicate type for column D
constant uint DATA_TYPE_A  [[function_constant(11)]];  // data type for column A
constant uint DATA_TYPE_B  [[function_constant(12)]];  // data type for column B
constant uint DATA_TYPE_C  [[function_constant(13)]];  // data type for column C
constant uint DATA_TYPE_D  [[function_constant(14)]];  // data type for column D
constant bool IS_64BIT_A   [[function_constant(15)]];  // 64-bit flag for column A
constant bool IS_64BIT_B   [[function_constant(16)]];  // 64-bit flag for column B
constant bool IS_64BIT_C   [[function_constant(17)]];  // 64-bit flag for column C
constant bool IS_64BIT_D   [[function_constant(18)]];  // 64-bit flag for column D
constant bool LOGIC_AND    [[function_constant(19)]];  // true=AND, false=OR

// Resolved with defaults (used when constants not set)
constant uint pred_type   = is_function_constant_defined(PRED_TYPE)   ? PRED_TYPE   : 0u;
constant bool is_64bit    = is_function_constant_defined(IS_64BIT)    ? IS_64BIT    : false;
constant bool output_idx  = is_function_constant_defined(OUTPUT_IDX)  ? OUTPUT_IDX  : false;
constant bool output_vals = is_function_constant_defined(OUTPUT_VALS) ? OUTPUT_VALS : true;
constant uint data_type   = is_function_constant_defined(DATA_TYPE)   ? DATA_TYPE   : 0u;
constant bool has_nulls   = is_function_constant_defined(HAS_NULLS)   ? HAS_NULLS   : false;

// Multi-column resolved with defaults
constant uint n_columns   = is_function_constant_defined(N_COLUMNS)   ? N_COLUMNS   : 1u;
constant uint pred_type_a = is_function_constant_defined(PRED_TYPE_A) ? PRED_TYPE_A : 0u;
constant uint pred_type_b = is_function_constant_defined(PRED_TYPE_B) ? PRED_TYPE_B : 0u;
constant uint pred_type_c = is_function_constant_defined(PRED_TYPE_C) ? PRED_TYPE_C : 0u;
constant uint pred_type_d = is_function_constant_defined(PRED_TYPE_D) ? PRED_TYPE_D : 0u;
constant uint data_type_a = is_function_constant_defined(DATA_TYPE_A) ? DATA_TYPE_A : 0u;
constant uint data_type_b = is_function_constant_defined(DATA_TYPE_B) ? DATA_TYPE_B : 0u;
constant uint data_type_c = is_function_constant_defined(DATA_TYPE_C) ? DATA_TYPE_C : 0u;
constant uint data_type_d = is_function_constant_defined(DATA_TYPE_D) ? DATA_TYPE_D : 0u;
constant bool is_64bit_a  = is_function_constant_defined(IS_64BIT_A)  ? IS_64BIT_A  : false;
constant bool is_64bit_b  = is_function_constant_defined(IS_64BIT_B)  ? IS_64BIT_B  : false;
constant bool is_64bit_c  = is_function_constant_defined(IS_64BIT_C)  ? IS_64BIT_C  : false;
constant bool is_64bit_d  = is_function_constant_defined(IS_64BIT_D)  ? IS_64BIT_D  : false;
constant bool logic_and   = is_function_constant_defined(LOGIC_AND)   ? LOGIC_AND   : true;

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

// Multi-column params: thresholds for up to 4 columns.
// Each column gets 4 uint fields (lo_lo, lo_hi, hi_lo, hi_hi).
// For 32-bit types, only lo_lo (=lo_bits) and hi_lo (=hi_bits) are used;
// lo_hi and hi_hi are zero padding.
struct MultiColumnParams {
    uint element_count;
    uint num_tiles;
    uint _pad0;
    uint _pad1;
    // Column A thresholds
    uint a_lo_lo;  uint a_lo_hi;  uint a_hi_lo;  uint a_hi_hi;
    // Column B thresholds
    uint b_lo_lo;  uint b_lo_hi;  uint b_hi_lo;  uint b_hi_hi;
    // Column C thresholds
    uint c_lo_lo;  uint c_lo_hi;  uint c_hi_lo;  uint c_hi_hi;
    // Column D thresholds
    uint d_lo_lo;  uint d_lo_hi;  uint d_hi_lo;  uint d_hi_hi;
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
// Kernel 2b: filter_add_block_offsets
//
// For hierarchical scan: adds a block-level prefix offset to each
// partial within a block. Each threadgroup handles one block of
// up to SCAN_BLOCK_SIZE (4096) partials.
//
// Dispatch: num_blocks threadgroups x 256 threads
// =================================================================

#define SCAN_BLOCK_SIZE 4096u  // FILTER_THREADS * SCAN_ELEMS_PER_THREAD

kernel void filter_add_block_offsets(
    device uint*        partials       [[buffer(0)]],
    device const uint*  block_offsets  [[buffer(1)]],
    constant uint&      total_parts    [[buffer(2)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint tg_idx    [[threadgroup_position_in_grid]]
)
{
    uint block_base = tg_idx * SCAN_BLOCK_SIZE;
    uint offset = block_offsets[tg_idx];

    for (uint i = 0u; i < SCAN_ELEMS_PER_THREAD; i++) {
        uint idx = block_base + lid * SCAN_ELEMS_PER_THREAD + i;
        if (idx < total_parts) {
            partials[idx] += offset;
        }
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

// --- Manual Ballot ---
// simd_ballot() returns opaque simd_vote on Metal — cannot extract
// bitmask. Build ballot manually via butterfly reduction: 5 uniform
// simd_shuffle_xor ops broadcast each lane's bit to all lanes.

inline uint manual_ballot(bool pred, uint lane) {
    uint val = pred ? (1u << lane) : 0u;
    val |= simd_shuffle_xor(val, 1u);
    val |= simd_shuffle_xor(val, 2u);
    val |= simd_shuffle_xor(val, 4u);
    val |= simd_shuffle_xor(val, 8u);
    val |= simd_shuffle_xor(val, 16u);
    return val;
}

// =================================================================
// Kernel 5: filter_bitmap_scan
//
// Evaluate predicate for each element, pack 32 predicate results per
// simdgroup into a bitmap word via manual ballot (butterfly reduction),
// write bitmap for downstream bitmap_scatter to use (avoids
// re-evaluating predicate). Also computes per-tile match count via
// SIMD prefix sum + cross-SG aggregation, writes tile total to
// partials[tg_idx].
//
// When has_nulls=true, reads validity bitmap and ANDs with predicate
// result before ballot (NULLs always excluded).
//
// Buffer layout:
//   buffer(0) = src data (const uint or const ulong)
//   buffer(1) = bitmap output (atomic_uint, ceil(N/32) words)
//   buffer(2) = partials output (one uint per threadgroup)
//   buffer(3) = validity bitmap (const uint, optional, Arrow LSB-first)
//   buffer(4) = params (FilterParams or FilterParams64)
//
// Dispatch: ceil(N / TILE_SIZE) threadgroups x 256 threads
// TG memory: 32 bytes (simd_totals[8])
// =================================================================

kernel void filter_bitmap_scan(
    device const uint*       input        [[buffer(0)]],
    device atomic_uint*      bitmap       [[buffer(1)]],
    device uint*             partials     [[buffer(2)]],
    device const uint*       validity     [[buffer(3)]],
    constant void*           params_raw   [[buffer(4)]],
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
            bool pred_result = false;
            if (idx < n) {
                ulong raw = input64[idx];
                if (data_type == 1u) {
                    pred_result = evaluate_predicate(
                        as_type<long>(raw), as_type<long>(lo_raw), as_type<long>(hi_raw));
                } else if (data_type == 2u) {
                    pred_result = evaluate_predicate_f64(raw, lo_raw, hi_raw);
                } else {
                    pred_result = evaluate_predicate(raw, lo_raw, hi_raw);
                }

                // NULL check: exclude NULLs by ANDing with validity bitmap
                if (has_nulls) {
                    uint validity_word = validity[idx / 32u];
                    bool is_valid = (validity_word >> (idx % 32u)) & 1u;
                    pred_result = pred_result && is_valid;
                }
            }

            // Manual ballot: butterfly reduction packs 32 results into one uint
            uint ballot_bits = manual_ballot(pred_result, simd_lane);

            // Lane 0 of each simdgroup writes ballot word to bitmap
            if (simd_lane == 0u) {
                uint bitmap_word_idx = (base + e * FILTER_THREADS + simd_idx * 32u) / 32u;
                atomic_store_explicit(&bitmap[bitmap_word_idx],
                                     ballot_bits, memory_order_relaxed);
            }

            // Each thread counts its own match
            thread_match_count += uint(pred_result);
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
            bool pred_result = false;
            if (idx < n) {
                uint raw = input[idx];
                if (data_type == 1u) {
                    pred_result = evaluate_predicate(
                        as_type<int>(raw), as_type<int>(lo_u), as_type<int>(hi_u));
                } else if (data_type == 2u) {
                    pred_result = evaluate_predicate(
                        as_type<float>(raw), as_type<float>(lo_u), as_type<float>(hi_u));
                } else {
                    pred_result = evaluate_predicate(raw, lo_u, hi_u);
                }

                // NULL check: exclude NULLs by ANDing with validity bitmap
                if (has_nulls) {
                    uint validity_word = validity[idx / 32u];
                    bool is_valid = (validity_word >> (idx % 32u)) & 1u;
                    pred_result = pred_result && is_valid;
                }
            }

            // Manual ballot: butterfly reduction packs 32 results into one uint
            uint ballot_bits = manual_ballot(pred_result, simd_lane);

            // Lane 0 of each simdgroup writes ballot word to bitmap
            if (simd_lane == 0u) {
                uint bitmap_word_idx = (base + e * FILTER_THREADS + simd_idx * 32u) / 32u;
                atomic_store_explicit(&bitmap[bitmap_word_idx],
                                     ballot_bits, memory_order_relaxed);
            }

            // Each thread counts its own match
            thread_match_count += uint(pred_result);
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
// Kernel 6: filter_bitmap_scatter
//
// Reads pre-computed bitmap from filter_bitmap_scan (no predicate
// re-evaluation). Extracts per-thread match flag from bitmap word,
// then does round-by-round SIMD prefix sum + cross-SG aggregation
// to compute ordered scatter positions. Reads global prefix from
// scanned partials[tg_idx].
//
// Buffer layout:
//   buffer(0) = src data (const uint or const ulong)
//   buffer(1) = bitmap (const uint, ceil(N/32) words, from bitmap_scan)
//   buffer(2) = out_vals (output values)
//   buffer(3) = out_idx  (output indices)
//   buffer(4) = partials (scanned, one uint per threadgroup)
//   buffer(5) = params   (FilterParams or FilterParams64)
//
// Dispatch: ceil(N / TILE_SIZE) threadgroups x 256 threads
// TG memory: 32 bytes (simd_totals[8])
// =================================================================

kernel void filter_bitmap_scatter(
    device const uint*       input        [[buffer(0)]],
    device const uint*       bitmap       [[buffer(1)]],
    device uint*             out_vals     [[buffer(2)]],
    device uint*             out_idx      [[buffer(3)]],
    device const uint*       partials     [[buffer(4)]],
    constant void*           params_raw   [[buffer(5)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint tg_idx    [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_idx  [[simdgroup_index_in_threadgroup]]
)
{
    threadgroup uint simd_totals[FILTER_NUM_SGS];
    threadgroup uint round_total_tg;

    // Read global prefix for this TG
    uint global_prefix = partials[tg_idx];

    // Running write offset — accumulates across element rounds
    uint running_offset = global_prefix;

    if (is_64bit) {
        // --- 64-bit scatter path (ordered, round-by-round) ---
        constant FilterParams64& params = *reinterpret_cast<constant FilterParams64*>(params_raw);
        uint n = params.element_count;
        uint base = tg_idx * FILTER_TILE_64;

        device const ulong* input64 = reinterpret_cast<device const ulong*>(input);
        device ulong* out_vals64 = reinterpret_cast<device ulong*>(out_vals);

        for (uint e = 0u; e < FILTER_ELEMS_64; e++) {
            uint idx = base + e * FILTER_THREADS + lid;
            bool valid = idx < n;

            // Read match flag from pre-computed bitmap
            bool match_flag = false;
            if (valid) {
                uint bitmap_word = bitmap[idx / 32u];
                match_flag = (bitmap_word >> (idx % 32u)) & 1u;
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
                ulong raw = input64[idx];
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

        for (uint e = 0u; e < FILTER_ELEMS_32; e++) {
            uint idx = base + e * FILTER_THREADS + lid;
            bool valid = idx < n;

            // Read match flag from pre-computed bitmap
            bool match_flag = false;
            if (valid) {
                uint bitmap_word = bitmap[idx / 32u];
                match_flag = (bitmap_word >> (idx % 32u)) & 1u;
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
                uint raw = input[idx];
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
// Multi-column predicate evaluation helpers
//
// Each column can have its own pred_type (GT/LT/GE/LE/EQ/NE/BETWEEN/TRUE),
// data_type (0=uint, 1=int, 2=float), and is_64bit flag.
// These helpers accept the pred/data type as runtime values but are
// called with function-constant-derived values, so the compiler
// eliminates dead branches at PSO specialization time.
// =================================================================

inline bool eval_column_32(
    device const uint* col,
    uint idx,
    uint lo_lo, uint hi_lo,
    uint p_type, uint d_type
) {
    uint raw = col[idx];
    if (d_type == 1u) {
        // Signed int
        int val = as_type<int>(raw);
        int lo  = as_type<int>(lo_lo);
        int hi  = as_type<int>(hi_lo);
        if (p_type == 0u) return val > lo;
        if (p_type == 1u) return val < lo;
        if (p_type == 2u) return val >= lo;
        if (p_type == 3u) return val <= lo;
        if (p_type == 4u) return val == lo;
        if (p_type == 5u) return val != lo;
        if (p_type == 6u) return val >= lo && val <= hi;
        return true;
    } else if (d_type == 2u) {
        // Float
        float val = as_type<float>(raw);
        float lo  = as_type<float>(lo_lo);
        float hi  = as_type<float>(hi_lo);
        if (p_type == 0u) return val > lo;
        if (p_type == 1u) return val < lo;
        if (p_type == 2u) return val >= lo;
        if (p_type == 3u) return val <= lo;
        if (p_type == 4u) return val == lo;
        if (p_type == 5u) return val != lo;
        if (p_type == 6u) return val >= lo && val <= hi;
        return true;
    } else {
        // Unsigned uint
        uint val = raw;
        if (p_type == 0u) return val > lo_lo;
        if (p_type == 1u) return val < lo_lo;
        if (p_type == 2u) return val >= lo_lo;
        if (p_type == 3u) return val <= lo_lo;
        if (p_type == 4u) return val == lo_lo;
        if (p_type == 5u) return val != lo_lo;
        if (p_type == 6u) return val >= lo_lo && val <= hi_lo;
        return true;
    }
}

inline bool eval_column_64(
    device const uint* col,
    uint idx,
    uint lo_lo, uint lo_hi, uint hi_lo, uint hi_hi,
    uint p_type, uint d_type
) {
    device const ulong* col64 = reinterpret_cast<device const ulong*>(col);
    ulong raw = col64[idx];
    ulong lo_raw = ulong(lo_lo) | (ulong(lo_hi) << 32u);
    ulong hi_raw = ulong(hi_lo) | (ulong(hi_hi) << 32u);

    if (d_type == 1u) {
        // Signed long
        long val = as_type<long>(raw);
        long lo  = as_type<long>(lo_raw);
        long hi  = as_type<long>(hi_raw);
        if (p_type == 0u) return val > lo;
        if (p_type == 1u) return val < lo;
        if (p_type == 2u) return val >= lo;
        if (p_type == 3u) return val <= lo;
        if (p_type == 4u) return val == lo;
        if (p_type == 5u) return val != lo;
        if (p_type == 6u) return val >= lo && val <= hi;
        return true;
    } else if (d_type == 2u) {
        // f64 via raw bits
        if (is_f64_nan(raw)) {
            return p_type == 5u; // NaN != anything = true
        }
        if (p_type == 5u) {
            if (is_f64_nan(lo_raw)) return true;
            ulong v_abs = raw & 0x7FFFFFFFFFFFFFFFul;
            ulong l_abs = lo_raw & 0x7FFFFFFFFFFFFFFFul;
            if (v_abs == 0ul && l_abs == 0ul) return false;
            return raw != lo_raw;
        }
        int cmp_lo = f64_compare(raw, lo_raw);
        if (cmp_lo == -2) return false;
        if (p_type == 0u) return cmp_lo > 0;
        if (p_type == 1u) return cmp_lo < 0;
        if (p_type == 2u) return cmp_lo >= 0;
        if (p_type == 3u) return cmp_lo <= 0;
        if (p_type == 4u) return cmp_lo == 0;
        if (p_type == 6u) {
            if (cmp_lo < 0) return false;
            int cmp_hi = f64_compare(raw, hi_raw);
            if (cmp_hi == -2) return false;
            return cmp_hi <= 0;
        }
        return true;
    } else {
        // Unsigned ulong
        if (p_type == 0u) return raw > lo_raw;
        if (p_type == 1u) return raw < lo_raw;
        if (p_type == 2u) return raw >= lo_raw;
        if (p_type == 3u) return raw <= lo_raw;
        if (p_type == 4u) return raw == lo_raw;
        if (p_type == 5u) return raw != lo_raw;
        if (p_type == 6u) return raw >= lo_raw && raw <= hi_raw;
        return true;
    }
}

// =================================================================
// Kernel 7: filter_multi_bitmap_scan
//
// Multi-column predicate evaluation + bitmap write + partials.
// Accepts up to 4 column data buffers. Evaluates each column's
// predicate independently, then combines results with AND or OR
// (controlled by LOGIC_AND function constant).
//
// Function constants control per-column configuration:
//   N_COLUMNS (6)          — number of active columns (1-4)
//   PRED_TYPE_A..D (7-10)  — predicate type per column
//   DATA_TYPE_A..D (11-14) — data type per column
//   IS_64BIT_A..D (15-18)  — 64-bit flag per column
//   LOGIC_AND (19)         — true=AND combine, false=OR combine
//
// All columns must have the same element count. Tile size is
// FILTER_TILE_32 (4096) — all columns indexed identically.
// For 64-bit columns, the column buffer holds ulong data but
// element indexing uses the same idx (column buffer is 2x larger).
//
// Buffer layout:
//   buffer(0) = column A data
//   buffer(1) = column B data
//   buffer(2) = column C data
//   buffer(3) = column D data
//   buffer(4) = bitmap output (atomic_uint)
//   buffer(5) = partials output (one uint per threadgroup)
//   buffer(6) = params (MultiColumnParams)
//
// Dispatch: ceil(N / FILTER_TILE_32) threadgroups x 256 threads
// TG memory: 32 bytes (simd_totals[8])
// =================================================================

kernel void filter_multi_bitmap_scan(
    device const uint*       col_a        [[buffer(0)]],
    device const uint*       col_b        [[buffer(1)]],
    device const uint*       col_c        [[buffer(2)]],
    device const uint*       col_d        [[buffer(3)]],
    device atomic_uint*      bitmap       [[buffer(4)]],
    device uint*             partials     [[buffer(5)]],
    constant MultiColumnParams& params    [[buffer(6)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint tg_idx    [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_idx  [[simdgroup_index_in_threadgroup]]
)
{
    threadgroup uint simd_totals[FILTER_NUM_SGS];

    uint n = params.element_count;
    uint base = tg_idx * FILTER_TILE_32;
    uint thread_match_count = 0u;

    for (uint e = 0u; e < FILTER_ELEMS_32; e++) {
        uint idx = base + e * FILTER_THREADS + lid;
        bool combined;

        if (idx < n) {
            // Initialize combined result based on logic mode
            // AND: start true, any false → false
            // OR:  start false, any true → true
            combined = logic_and;

            // --- Column A (always evaluated) ---
            {
                bool result_a;
                if (is_64bit_a) {
                    result_a = eval_column_64(col_a, idx,
                        params.a_lo_lo, params.a_lo_hi,
                        params.a_hi_lo, params.a_hi_hi,
                        pred_type_a, data_type_a);
                } else {
                    result_a = eval_column_32(col_a, idx,
                        params.a_lo_lo, params.a_hi_lo,
                        pred_type_a, data_type_a);
                }
                if (logic_and) combined = combined && result_a;
                else           combined = combined || result_a;
            }

            // --- Column B (if n_columns >= 2) ---
            if (n_columns >= 2u) {
                bool result_b;
                if (is_64bit_b) {
                    result_b = eval_column_64(col_b, idx,
                        params.b_lo_lo, params.b_lo_hi,
                        params.b_hi_lo, params.b_hi_hi,
                        pred_type_b, data_type_b);
                } else {
                    result_b = eval_column_32(col_b, idx,
                        params.b_lo_lo, params.b_hi_lo,
                        pred_type_b, data_type_b);
                }
                if (logic_and) combined = combined && result_b;
                else           combined = combined || result_b;
            }

            // --- Column C (if n_columns >= 3) ---
            if (n_columns >= 3u) {
                bool result_c;
                if (is_64bit_c) {
                    result_c = eval_column_64(col_c, idx,
                        params.c_lo_lo, params.c_lo_hi,
                        params.c_hi_lo, params.c_hi_hi,
                        pred_type_c, data_type_c);
                } else {
                    result_c = eval_column_32(col_c, idx,
                        params.c_lo_lo, params.c_hi_lo,
                        pred_type_c, data_type_c);
                }
                if (logic_and) combined = combined && result_c;
                else           combined = combined || result_c;
            }

            // --- Column D (if n_columns >= 4) ---
            if (n_columns >= 4u) {
                bool result_d;
                if (is_64bit_d) {
                    result_d = eval_column_64(col_d, idx,
                        params.d_lo_lo, params.d_lo_hi,
                        params.d_hi_lo, params.d_hi_hi,
                        pred_type_d, data_type_d);
                } else {
                    result_d = eval_column_32(col_d, idx,
                        params.d_lo_lo, params.d_hi_lo,
                        pred_type_d, data_type_d);
                }
                if (logic_and) combined = combined && result_d;
                else           combined = combined || result_d;
            }
        } else {
            combined = false;
        }

        // Manual ballot: butterfly reduction packs 32 results into one uint
        uint ballot_bits = manual_ballot(combined, simd_lane);

        // Lane 0 of each simdgroup writes ballot word to bitmap
        if (simd_lane == 0u) {
            uint bitmap_word_idx = (base + e * FILTER_THREADS + simd_idx * 32u) / 32u;
            atomic_store_explicit(&bitmap[bitmap_word_idx],
                                 ballot_bits, memory_order_relaxed);
        }

        // Each thread counts its own match
        thread_match_count += uint(combined);
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
