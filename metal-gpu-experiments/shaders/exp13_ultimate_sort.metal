#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 13: Ultimate Radix Sort
//
// Combines all proven optimizations into a production-quality sort:
//   1. Fence-fixed single dispatch (atomic_thread_fence seq_cst device_scope)
//   2. SIMD rank (2.83x faster than serial rank)
//   3. Tile-status reuse (single section, cleared between passes)
//   4. u32, u64, and key-value pair support
//
// Scale test: 1M to 16M elements
// ══════════════════════════════════════════════════════════════════════

#define EXP13_NUM_BINS 256u
#define EXP13_NUM_SGS 8u

struct Exp13Params {
    uint element_count;
    uint num_tiles;
    uint num_tgs;
    uint num_passes;
};

// ── Global sync with device-scope fence (proven in exp12) ───────────
inline void exp13_sync(device atomic_uint* counter, uint num_tgs, uint lid)
{
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (lid == 0u) {
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        while (atomic_load_explicit(counter, memory_order_relaxed) < num_tgs) {}
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
}

// ══════════════════════════════════════════════════════════════════════
// MEGASORT BODY MACRO
//
// Single-dispatch megasort with:
//   - Fence-fixed cross-TG coherence
//   - SIMD rank (within-SG scan + cross-SG histogram)
//   - Tile-status reuse (ts_off=0 always, cleared between passes)
//   - Work-steal counter reuse (counters[0..1], cleared between passes)
//   - Barrier counters at counters[2..] (unique per barrier instance)
//
// Works for any integral element type (uint, ulong) via auto deduction.
// ══════════════════════════════════════════════════════════════════════

#define EXP13_MEGASORT_BODY \
    uint num_tiles   = params.num_tiles; \
    uint num_tgs     = params.num_tgs; \
    uint num_passes  = params.num_passes; \
    uint elem_count  = params.element_count; \
    uint ts_section  = num_tiles * EXP13_NUM_BINS; \
    \
    threadgroup atomic_uint local_hist[EXP13_NUM_BINS]; \
    threadgroup uint shared_digits[TILE_SIZE]; \
    threadgroup uint shared_tile_id; \
    threadgroup uint global_bin_start[EXP13_NUM_BINS]; \
    threadgroup uint sg_totals[8]; \
    threadgroup atomic_uint sg_hist[EXP13_NUM_SGS * EXP13_NUM_BINS]; \
    \
    uint barrier_idx = 2u; /* counters[0..1] = work-steal, [2..] = barriers */ \
    \
    for (uint pass = 0u; pass < num_passes; pass++) { \
        uint shift = pass * 8u; \
        \
        device atomic_uint* ws1 = &counters[0]; \
        device atomic_uint* ws2 = &counters[1]; \
        \
        auto src = (pass % 2u == 0u) ? buf_a : buf_b; \
        auto dst = (pass % 2u == 0u) ? buf_b : buf_a; \
        \
        /* ═══ Loop 1: Histogram + Decoupled Lookback ═══ */ \
        while (true) { \
            if (lid == 0u) { \
                shared_tile_id = atomic_fetch_add_explicit(ws1, 1u, memory_order_relaxed); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            uint tile_id = shared_tile_id; \
            if (tile_id >= num_tiles) break; \
            \
            atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed); \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            uint base = tile_id * TILE_SIZE; \
            bool valid = (base + lid) < elem_count; \
            \
            if (valid) { \
                auto val = src[base + lid]; \
                uint digit = (uint)((val >> shift) & 0xFFu); \
                atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            uint my_count = atomic_load_explicit(&local_hist[lid], memory_order_relaxed); \
            \
            uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT) | (my_count & VALUE_MASK); \
            atomic_store_explicit( \
                &tile_status[tile_id * EXP13_NUM_BINS + lid], \
                packed_agg, memory_order_relaxed); \
            \
            if (tile_id == 0u) { \
                uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (my_count & VALUE_MASK); \
                atomic_store_explicit(&tile_status[lid], packed_pfx, memory_order_relaxed); \
            } else { \
                uint running = 0u; \
                int lb = (int)tile_id - 1; \
                while (lb >= 0) { \
                    uint status = atomic_load_explicit( \
                        &tile_status[lb * EXP13_NUM_BINS + lid], memory_order_relaxed); \
                    uint flag  = status >> FLAG_SHIFT; \
                    uint value = status & VALUE_MASK; \
                    if (flag == FLAG_PREFIX)    { running += value; break; } \
                    else if (flag == FLAG_AGGREGATE) { running += value; lb--; } \
                } \
                uint inclusive = running + my_count; \
                uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK); \
                atomic_store_explicit( \
                    &tile_status[tile_id * EXP13_NUM_BINS + lid], \
                    packed_pfx, memory_order_relaxed); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        \
        /* ═══ Sync A: All tiles histogrammed ═══ */ \
        exp13_sync(&counters[barrier_idx], num_tgs, lid); \
        barrier_idx++; \
        \
        /* ═══ Compute global_bin_start ═══ */ \
        uint last_status = atomic_load_explicit( \
            &tile_status[(num_tiles - 1u) * EXP13_NUM_BINS + lid], memory_order_relaxed); \
        uint global_total_val = last_status & VALUE_MASK; \
        \
        uint simd_scan = simd_prefix_inclusive_sum(global_total_val); \
        if (simd_lane == 31u) sg_totals[simd_id] = simd_scan; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        if (simd_id == 0u && simd_lane < 8u) { \
            uint sg = sg_totals[simd_lane]; \
            uint sg_scan = simd_prefix_inclusive_sum(sg); \
            sg_totals[simd_lane] = sg_scan; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        uint inc_val = simd_scan; \
        if (simd_id > 0u) inc_val += sg_totals[simd_id - 1u]; \
        global_bin_start[lid] = inc_val - global_total_val; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        /* ═══ Loop 2: Scatter (SIMD Rank) ═══ */ \
        while (true) { \
            if (lid == 0u) { \
                shared_tile_id = atomic_fetch_add_explicit(ws2, 1u, memory_order_relaxed); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            uint tile_id = shared_tile_id; \
            if (tile_id >= num_tiles) break; \
            \
            uint base = tile_id * TILE_SIZE; \
            bool valid = (base + lid) < elem_count; \
            auto element = src[min(base + lid, elem_count - 1u)]; \
            uint digit = valid ? (uint)((element >> shift) & 0xFFu) : 0xFFFFu; \
            \
            for (uint i = lid; i < EXP13_NUM_SGS * EXP13_NUM_BINS; i += TILE_SIZE) { \
                atomic_store_explicit(&sg_hist[i], 0u, memory_order_relaxed); \
            } \
            shared_digits[lid] = digit; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            if (valid) { \
                atomic_fetch_add_explicit( \
                    &sg_hist[simd_id * EXP13_NUM_BINS + digit], 1u, memory_order_relaxed); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            if (valid) { \
                uint within_rank = 0u; \
                uint sg_base = simd_id * 32u; \
                for (uint j = 0u; j < simd_lane; j++) { \
                    within_rank += (shared_digits[sg_base + j] == digit) ? 1u : 0u; \
                } \
                \
                uint cross_rank = 0u; \
                for (uint s = 0u; s < simd_id; s++) { \
                    cross_rank += atomic_load_explicit( \
                        &sg_hist[s * EXP13_NUM_BINS + digit], memory_order_relaxed); \
                } \
                \
                uint rank = cross_rank + within_rank; \
                \
                uint bin_excl; \
                if (tile_id == 0u) { \
                    bin_excl = 0u; \
                } else { \
                    uint prev = atomic_load_explicit( \
                        &tile_status[(tile_id - 1u) * EXP13_NUM_BINS + digit], \
                        memory_order_relaxed); \
                    bin_excl = prev & VALUE_MASK; \
                } \
                \
                uint pos = global_bin_start[digit] + bin_excl + rank; \
                dst[pos] = element; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        \
        /* Between passes: clear tile_status + work-steal counters */ \
        if (pass < num_passes - 1u) { \
            exp13_sync(&counters[barrier_idx], num_tgs, lid); \
            barrier_idx++; \
            \
            /* Clear tile_status (single section, reused) */ \
            for (uint i = gid * TILE_SIZE + lid; i < ts_section; \
                 i += num_tgs * TILE_SIZE) { \
                atomic_store_explicit(&tile_status[i], 0u, memory_order_relaxed); \
            } \
            \
            /* Reset work-steal counters for next pass */ \
            if (gid == 0u && lid < 2u) { \
                atomic_store_explicit(&counters[lid], 0u, memory_order_relaxed); \
            } \
            \
            exp13_sync(&counters[barrier_idx], num_tgs, lid); \
            barrier_idx++; \
        } \
    }


// ═══════════════════════════════════════════════════════════════════
// u32 Sort — 4-pass, 8-bit radix
// ═══════════════════════════════════════════════════════════════════

kernel void exp13_sort_u32(
    device uint* buf_a               [[buffer(0)]],
    device uint* buf_b               [[buffer(1)]],
    device atomic_uint* tile_status  [[buffer(2)]],
    device atomic_uint* counters     [[buffer(3)]],
    constant Exp13Params& params     [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    EXP13_MEGASORT_BODY
}

// ═══════════════════════════════════════════════════════════════════
// u64 Sort — 8-pass, 8-bit radix
// ═══════════════════════════════════════════════════════════════════

kernel void exp13_sort_u64(
    device ulong* buf_a              [[buffer(0)]],
    device ulong* buf_b              [[buffer(1)]],
    device atomic_uint* tile_status  [[buffer(2)]],
    device atomic_uint* counters     [[buffer(3)]],
    constant Exp13Params& params     [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    EXP13_MEGASORT_BODY
}

// ═══════════════════════════════════════════════════════════════════
// Key-Value u32 Sort — 4-pass, sorts by key, carries value
// SoA layout: separate key and value buffers
// ═══════════════════════════════════════════════════════════════════

kernel void exp13_sort_kv32(
    device uint* key_a               [[buffer(0)]],
    device uint* key_b               [[buffer(1)]],
    device uint* val_a               [[buffer(2)]],
    device uint* val_b               [[buffer(3)]],
    device atomic_uint* tile_status  [[buffer(4)]],
    device atomic_uint* counters     [[buffer(5)]],
    constant Exp13Params& params     [[buffer(6)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint num_tiles   = params.num_tiles;
    uint num_tgs     = params.num_tgs;
    uint num_passes  = params.num_passes;
    uint elem_count  = params.element_count;
    uint ts_section  = num_tiles * EXP13_NUM_BINS;

    threadgroup atomic_uint local_hist[EXP13_NUM_BINS];
    threadgroup uint shared_digits[TILE_SIZE];
    threadgroup uint shared_tile_id;
    threadgroup uint global_bin_start[EXP13_NUM_BINS];
    threadgroup uint sg_totals[8];
    threadgroup atomic_uint sg_hist[EXP13_NUM_SGS * EXP13_NUM_BINS];

    uint barrier_idx = 2u;

    for (uint pass = 0u; pass < num_passes; pass++) {
        uint shift = pass * 8u;

        device atomic_uint* ws1 = &counters[0];
        device atomic_uint* ws2 = &counters[1];

        device uint* key_src = (pass % 2u == 0u) ? key_a : key_b;
        device uint* key_dst = (pass % 2u == 0u) ? key_b : key_a;
        device uint* val_src = (pass % 2u == 0u) ? val_a : val_b;
        device uint* val_dst = (pass % 2u == 0u) ? val_b : val_a;

        // ═══ Loop 1: Histogram + Decoupled Lookback (keys only) ═══
        while (true) {
            if (lid == 0u) {
                shared_tile_id = atomic_fetch_add_explicit(ws1, 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tile_id = shared_tile_id;
            if (tile_id >= num_tiles) break;

            atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint base = tile_id * TILE_SIZE;
            bool valid = (base + lid) < elem_count;

            if (valid) {
                uint key = key_src[base + lid];
                uint digit = (key >> shift) & 0xFFu;
                atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint my_count = atomic_load_explicit(&local_hist[lid], memory_order_relaxed);

            uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT) | (my_count & VALUE_MASK);
            atomic_store_explicit(
                &tile_status[tile_id * EXP13_NUM_BINS + lid],
                packed_agg, memory_order_relaxed);

            if (tile_id == 0u) {
                uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (my_count & VALUE_MASK);
                atomic_store_explicit(&tile_status[lid], packed_pfx, memory_order_relaxed);
            } else {
                uint running = 0u;
                int lb = (int)tile_id - 1;
                while (lb >= 0) {
                    uint status = atomic_load_explicit(
                        &tile_status[lb * EXP13_NUM_BINS + lid], memory_order_relaxed);
                    uint flag  = status >> FLAG_SHIFT;
                    uint value = status & VALUE_MASK;
                    if (flag == FLAG_PREFIX)    { running += value; break; }
                    else if (flag == FLAG_AGGREGATE) { running += value; lb--; }
                }
                uint inclusive = running + my_count;
                uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK);
                atomic_store_explicit(
                    &tile_status[tile_id * EXP13_NUM_BINS + lid],
                    packed_pfx, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        exp13_sync(&counters[barrier_idx], num_tgs, lid);
        barrier_idx++;

        // ═══ Compute global_bin_start ═══
        uint last_status = atomic_load_explicit(
            &tile_status[(num_tiles - 1u) * EXP13_NUM_BINS + lid], memory_order_relaxed);
        uint global_total_val = last_status & VALUE_MASK;

        uint simd_scan = simd_prefix_inclusive_sum(global_total_val);
        if (simd_lane == 31u) sg_totals[simd_id] = simd_scan;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u && simd_lane < 8u) {
            uint sg = sg_totals[simd_lane];
            uint sg_scan = simd_prefix_inclusive_sum(sg);
            sg_totals[simd_lane] = sg_scan;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint inc_val = simd_scan;
        if (simd_id > 0u) inc_val += sg_totals[simd_id - 1u];
        global_bin_start[lid] = inc_val - global_total_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ═══ Loop 2: Scatter keys + values (SIMD Rank) ═══
        while (true) {
            if (lid == 0u) {
                shared_tile_id = atomic_fetch_add_explicit(ws2, 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tile_id = shared_tile_id;
            if (tile_id >= num_tiles) break;

            uint base = tile_id * TILE_SIZE;
            bool valid = (base + lid) < elem_count;
            uint key_elem = 0u;
            uint val_elem = 0u;
            uint digit = 0xFFFFu;

            if (valid) {
                key_elem = key_src[base + lid];
                val_elem = val_src[base + lid];
                digit = (key_elem >> shift) & 0xFFu;
            }

            for (uint i = lid; i < EXP13_NUM_SGS * EXP13_NUM_BINS; i += TILE_SIZE) {
                atomic_store_explicit(&sg_hist[i], 0u, memory_order_relaxed);
            }
            shared_digits[lid] = digit;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (valid) {
                atomic_fetch_add_explicit(
                    &sg_hist[simd_id * EXP13_NUM_BINS + digit], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (valid) {
                uint within_rank = 0u;
                uint sg_base = simd_id * 32u;
                for (uint j = 0u; j < simd_lane; j++) {
                    within_rank += (shared_digits[sg_base + j] == digit) ? 1u : 0u;
                }

                uint cross_rank = 0u;
                for (uint s = 0u; s < simd_id; s++) {
                    cross_rank += atomic_load_explicit(
                        &sg_hist[s * EXP13_NUM_BINS + digit], memory_order_relaxed);
                }

                uint rank = cross_rank + within_rank;

                uint bin_excl;
                if (tile_id == 0u) {
                    bin_excl = 0u;
                } else {
                    uint prev = atomic_load_explicit(
                        &tile_status[(tile_id - 1u) * EXP13_NUM_BINS + digit],
                        memory_order_relaxed);
                    bin_excl = prev & VALUE_MASK;
                }

                uint pos = global_bin_start[digit] + bin_excl + rank;
                key_dst[pos] = key_elem;
                val_dst[pos] = val_elem;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Between passes
        if (pass < num_passes - 1u) {
            exp13_sync(&counters[barrier_idx], num_tgs, lid);
            barrier_idx++;

            for (uint i = gid * TILE_SIZE + lid; i < ts_section;
                 i += num_tgs * TILE_SIZE) {
                atomic_store_explicit(&tile_status[i], 0u, memory_order_relaxed);
            }

            if (gid == 0u && lid < 2u) {
                atomic_store_explicit(&counters[lid], 0u, memory_order_relaxed);
            }

            exp13_sync(&counters[barrier_idx], num_tgs, lid);
            barrier_idx++;
        }
    }
}
