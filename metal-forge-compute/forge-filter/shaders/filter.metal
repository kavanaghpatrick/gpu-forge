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

// Resolved with defaults (used when constants not set)
constant uint pred_type   = is_function_constant_defined(PRED_TYPE)   ? PRED_TYPE   : 0u;
constant bool is_64bit    = is_function_constant_defined(IS_64BIT)    ? IS_64BIT    : false;
constant bool output_idx  = is_function_constant_defined(OUTPUT_IDX)  ? OUTPUT_IDX  : false;
constant bool output_vals = is_function_constant_defined(OUTPUT_VALS) ? OUTPUT_VALS : true;

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
                // Use as_type to reinterpret for signed/double comparisons
                // The predicate compares raw ulong bits — Rust side handles
                // the bit reinterpretation for signed types
                thread_match_count += uint(evaluate_predicate(raw, lo_raw, hi_raw));
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
                thread_match_count += uint(evaluate_predicate(raw, lo_u, hi_u));
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

    // Read global prefix for this TG
    uint global_prefix = partials[tg_idx];

    if (is_64bit) {
        // --- 64-bit scatter path ---
        constant FilterParams64& params = *reinterpret_cast<constant FilterParams64*>(params_raw);
        uint n = params.element_count;
        uint base = tg_idx * FILTER_TILE_64;

        ulong lo_raw = ulong(params.lo_lo) | (ulong(params.lo_hi) << 32u);
        ulong hi_raw = ulong(params.hi_lo) | (ulong(params.hi_hi) << 32u);

        device const ulong* input64 = reinterpret_cast<device const ulong*>(input);
        device ulong* out_vals64 = reinterpret_cast<device ulong*>(out_vals);

        // Evaluate predicate and cache results
        bool flags[FILTER_ELEMS_64];
        ulong values[FILTER_ELEMS_64];
        uint indices[FILTER_ELEMS_64];
        uint thread_match_count = 0u;

        for (uint e = 0u; e < FILTER_ELEMS_64; e++) {
            uint idx = base + e * FILTER_THREADS + lid;
            bool valid = idx < n;
            ulong raw = valid ? input64[idx] : 0ul;
            bool match = valid && evaluate_predicate(raw, lo_raw, hi_raw);
            flags[e] = match;
            values[e] = raw;
            indices[e] = idx;
            thread_match_count += uint(match);
        }

        // SIMD prefix sum
        uint simd_prefix = simd_prefix_exclusive_sum(thread_match_count);

        if (simd_lane == 31u) {
            simd_totals[simd_idx] = simd_prefix + thread_match_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < FILTER_NUM_SGS) {
            simd_totals[lid] = simd_prefix_exclusive_sum(simd_totals[lid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute write base position
        uint thread_base = global_prefix + simd_prefix + simd_totals[simd_idx];

        // Scatter matching elements
        uint write_pos = thread_base;
        for (uint e = 0u; e < FILTER_ELEMS_64; e++) {
            if (flags[e]) {
                if (output_vals) out_vals64[write_pos] = values[e];
                if (output_idx)  out_idx[write_pos] = indices[e];
                write_pos++;
            }
        }
    } else {
        // --- 32-bit scatter path ---
        constant FilterParams& params = *reinterpret_cast<constant FilterParams*>(params_raw);
        uint n = params.element_count;
        uint base = tg_idx * FILTER_TILE_32;
        uint lo_u = params.lo_bits;
        uint hi_u = params.hi_bits;

        // Evaluate predicate and cache results
        bool flags[FILTER_ELEMS_32];
        uint values[FILTER_ELEMS_32];
        uint indices[FILTER_ELEMS_32];
        uint thread_match_count = 0u;

        for (uint e = 0u; e < FILTER_ELEMS_32; e++) {
            uint idx = base + e * FILTER_THREADS + lid;
            bool valid = idx < n;
            uint raw = valid ? input[idx] : 0u;
            bool match = valid && evaluate_predicate(raw, lo_u, hi_u);
            flags[e] = match;
            values[e] = raw;
            indices[e] = idx;
            thread_match_count += uint(match);
        }

        // SIMD prefix sum
        uint simd_prefix = simd_prefix_exclusive_sum(thread_match_count);

        if (simd_lane == 31u) {
            simd_totals[simd_idx] = simd_prefix + thread_match_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < FILTER_NUM_SGS) {
            simd_totals[lid] = simd_prefix_exclusive_sum(simd_totals[lid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute write base position
        uint thread_base = global_prefix + simd_prefix + simd_totals[simd_idx];

        // Scatter matching elements
        uint write_pos = thread_base;
        for (uint e = 0u; e < FILTER_ELEMS_32; e++) {
            if (flags[e]) {
                if (output_vals) out_vals[write_pos] = values[e];
                if (output_idx)  out_idx[write_pos]  = indices[e];
                write_pos++;
            }
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
            bool match = valid && evaluate_predicate(raw, lo_raw, hi_raw);
            flags[e] = match;
            values[e] = raw;
            idx_cache[e] = idx;
            local_count += uint(match);
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
            bool match = valid && evaluate_predicate(raw, lo_u, hi_u);
            flags[e] = match;
            values[e] = raw;
            idx_cache[e] = idx;
            local_count += uint(match);
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
