#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 10: SIMD-Optimized Stable Rank
//
// The O(TILE_SIZE) serial stable_rank is the compute bottleneck in
// persistent radix sort — each thread loops over up to 255 predecessors.
//
// This experiment replaces it with a two-phase approach:
//   Phase 1: Within-simdgroup rank via simd_shuffle — O(31) max
//   Phase 2: Cross-simdgroup rank via shared histogram — O(7) reads
//   Total: O(38) vs O(255) — ~6.7x fewer operations
//
// Version A: Serial rank (baseline, identical to exp8_persistent_pass)
// Version B: SIMD rank (same algorithm, faster rank computation)
// ══════════════════════════════════════════════════════════════════════

#define EXP10_NUM_BINS 256u
#define EXP10_NUM_SGS 8u  // 256 threads / 32 per simdgroup

struct Exp10PersistParams {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint num_tgs;
    uint counter_base;
    uint ts_offset;
};

// ── Serial stable rank (baseline) ────────────────────────────────────
inline uint exp10_serial_rank(uint lid, uint digit,
                              threadgroup uint* shared_digits)
{
    uint rank = 0;
    for (uint i = 0; i < lid; i++) {
        rank += (shared_digits[i] == digit) ? 1u : 0u;
    }
    return rank;
}

// ── Global barrier for persistent kernels ────────────────────────────
inline void exp10_global_sync(device atomic_uint* counter,
                              uint num_tgs, uint lid)
{
    threadgroup_barrier(mem_flags::mem_device);
    if (lid == 0u) {
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        while (atomic_load_explicit(counter, memory_order_relaxed) < num_tgs) {}
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// ═══════════════════════════════════════════════════════════════════
// Version A: Serial rank (baseline)
// Identical to exp8_persistent_pass — copied here for clean A/B timing
// ═══════════════════════════════════════════════════════════════════

kernel void exp10_serial_pass(
    device const uint* src          [[buffer(0)]],
    device uint* dst                [[buffer(1)]],
    device atomic_uint* tile_status [[buffer(2)]],
    device atomic_uint* counters    [[buffer(3)]],
    constant Exp10PersistParams& params [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint num_tiles = params.num_tiles;
    uint num_tgs   = params.num_tgs;
    uint shift     = params.shift;
    uint ts_off    = params.ts_offset;

    device atomic_uint* ws1 = &counters[params.counter_base];
    device atomic_uint* ws2 = &counters[params.counter_base + 1u];
    device atomic_uint* barrier_ctr = &counters[params.counter_base + 2u];

    threadgroup atomic_uint local_hist[EXP10_NUM_BINS];
    threadgroup uint shared_digits[TILE_SIZE];
    threadgroup uint shared_tile_id;
    threadgroup uint global_bin_start[EXP10_NUM_BINS];
    threadgroup uint sg_totals[8];

    // ═══ Loop 1: Histogram + Decoupled Lookback ═══
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
        bool valid = (base + lid) < params.element_count;

        if (valid) {
            uint val = src[base + lid];
            uint digit = (val >> shift) & 0xFFu;
            atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint my_count = atomic_load_explicit(&local_hist[lid], memory_order_relaxed);

        uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT) | (my_count & VALUE_MASK);
        atomic_store_explicit(&tile_status[ts_off + tile_id * EXP10_NUM_BINS + lid],
                              packed_agg, memory_order_relaxed);

        if (tile_id == 0u) {
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (my_count & VALUE_MASK);
            atomic_store_explicit(&tile_status[ts_off + lid], packed_pfx, memory_order_relaxed);
        } else {
            uint running = 0u;
            int lb = (int)tile_id - 1;
            while (lb >= 0) {
                uint status = atomic_load_explicit(
                    &tile_status[ts_off + lb * EXP10_NUM_BINS + lid], memory_order_relaxed);
                uint flag  = status >> FLAG_SHIFT;
                uint value = status & VALUE_MASK;
                if (flag == FLAG_PREFIX)    { running += value; break; }
                else if (flag == FLAG_AGGREGATE) { running += value; lb--; }
            }
            uint inclusive = running + my_count;
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK);
            atomic_store_explicit(&tile_status[ts_off + tile_id * EXP10_NUM_BINS + lid],
                                  packed_pfx, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    exp10_global_sync(barrier_ctr, num_tgs, lid);

    // ═══ Compute global_bin_start ═══
    uint last_status = atomic_load_explicit(
        &tile_status[ts_off + (num_tiles - 1u) * EXP10_NUM_BINS + lid], memory_order_relaxed);
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

    uint inc = simd_scan;
    if (simd_id > 0u) inc += sg_totals[simd_id - 1u];
    global_bin_start[lid] = inc - global_total_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══ Loop 2: Scatter (SERIAL RANK) ═══
    while (true) {
        if (lid == 0u) {
            shared_tile_id = atomic_fetch_add_explicit(ws2, 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tile_id = shared_tile_id;
        if (tile_id >= num_tiles) break;

        uint base = tile_id * TILE_SIZE;
        bool valid = (base + lid) < params.element_count;
        uint element = 0u;
        uint digit = 0u;

        if (valid) {
            element = src[base + lid];
            digit = (element >> shift) & 0xFFu;
        }
        shared_digits[lid] = valid ? digit : 0xFFFFu;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            uint rank = exp10_serial_rank(lid, digit, shared_digits);

            uint bin_excl;
            if (tile_id == 0u) {
                bin_excl = 0u;
            } else {
                uint prev = atomic_load_explicit(
                    &tile_status[ts_off + (tile_id - 1u) * EXP10_NUM_BINS + digit],
                    memory_order_relaxed);
                bin_excl = prev & VALUE_MASK;
            }

            uint pos = global_bin_start[digit] + bin_excl + rank;
            dst[pos] = element;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Version B: SIMD-optimized rank
//
// Two-phase stable rank:
//   Phase 1 (within-simdgroup): scan shared_digits[] for my SG's
//     range only. Cost: O(simd_lane) <= O(31)
//   Phase 2 (cross-simdgroup): per-SG histogram in threadgroup memory.
//     Each thread reads histogram entries from earlier SGs.
//     Cost: O(simd_id) <= O(7)
//   Total: O(38) max vs O(255) for serial.
//
// NOTE: simd_shuffle was tried first but produces incorrect results
// on Apple M4 Pro — likely due to non-uniform lane IDs in the loop.
// shared_digits[] reads are equally fast and correct.
//
// Extra TG memory: sg_hist[8 * 256] atomic_uint = 8 KB
// Extra barriers: 1 (after histogram contribution, before read)
// ═══════════════════════════════════════════════════════════════════

kernel void exp10_simd_pass(
    device const uint* src          [[buffer(0)]],
    device uint* dst                [[buffer(1)]],
    device atomic_uint* tile_status [[buffer(2)]],
    device atomic_uint* counters    [[buffer(3)]],
    constant Exp10PersistParams& params [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint num_tiles = params.num_tiles;
    uint num_tgs   = params.num_tgs;
    uint shift     = params.shift;
    uint ts_off    = params.ts_offset;

    device atomic_uint* ws1 = &counters[params.counter_base];
    device atomic_uint* ws2 = &counters[params.counter_base + 1u];
    device atomic_uint* barrier_ctr = &counters[params.counter_base + 2u];

    threadgroup atomic_uint local_hist[EXP10_NUM_BINS];
    threadgroup uint shared_digits[TILE_SIZE];
    threadgroup uint shared_tile_id;
    threadgroup uint global_bin_start[EXP10_NUM_BINS];
    threadgroup uint sg_totals[8];
    // Extra TG memory for SIMD rank: per-simdgroup digit histogram
    threadgroup atomic_uint sg_hist[EXP10_NUM_SGS * EXP10_NUM_BINS];

    // ═══ Loop 1: Histogram + Decoupled Lookback (identical to serial) ═══
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
        bool valid = (base + lid) < params.element_count;

        if (valid) {
            uint val = src[base + lid];
            uint digit = (val >> shift) & 0xFFu;
            atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint my_count = atomic_load_explicit(&local_hist[lid], memory_order_relaxed);

        uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT) | (my_count & VALUE_MASK);
        atomic_store_explicit(&tile_status[ts_off + tile_id * EXP10_NUM_BINS + lid],
                              packed_agg, memory_order_relaxed);

        if (tile_id == 0u) {
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (my_count & VALUE_MASK);
            atomic_store_explicit(&tile_status[ts_off + lid], packed_pfx, memory_order_relaxed);
        } else {
            uint running = 0u;
            int lb = (int)tile_id - 1;
            while (lb >= 0) {
                uint status = atomic_load_explicit(
                    &tile_status[ts_off + lb * EXP10_NUM_BINS + lid], memory_order_relaxed);
                uint flag  = status >> FLAG_SHIFT;
                uint value = status & VALUE_MASK;
                if (flag == FLAG_PREFIX)    { running += value; break; }
                else if (flag == FLAG_AGGREGATE) { running += value; lb--; }
            }
            uint inclusive = running + my_count;
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK);
            atomic_store_explicit(&tile_status[ts_off + tile_id * EXP10_NUM_BINS + lid],
                                  packed_pfx, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    exp10_global_sync(barrier_ctr, num_tgs, lid);

    // ═══ Compute global_bin_start (identical to serial) ═══
    uint last_status = atomic_load_explicit(
        &tile_status[ts_off + (num_tiles - 1u) * EXP10_NUM_BINS + lid], memory_order_relaxed);
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

    uint inc = simd_scan;
    if (simd_id > 0u) inc += sg_totals[simd_id - 1u];
    global_bin_start[lid] = inc - global_total_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══ Loop 2: Scatter (SIMD RANK) ═══
    while (true) {
        if (lid == 0u) {
            shared_tile_id = atomic_fetch_add_explicit(ws2, 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tile_id = shared_tile_id;
        if (tile_id >= num_tiles) break;

        uint base = tile_id * TILE_SIZE;
        bool valid = (base + lid) < params.element_count;
        uint element = 0u;
        uint digit = 0u;

        if (valid) {
            element = src[base + lid];
            digit = (element >> shift) & 0xFFu;
        }

        // ── Clear sg_hist + store digits ──
        // 2048 entries / 256 threads = 8 entries per thread
        for (uint i = lid; i < EXP10_NUM_SGS * EXP10_NUM_BINS; i += TILE_SIZE) {
            atomic_store_explicit(&sg_hist[i], 0u, memory_order_relaxed);
        }
        shared_digits[lid] = valid ? digit : 0xFFFFu;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Contribute to per-simdgroup histogram ──
        if (valid) {
            atomic_fetch_add_explicit(
                &sg_hist[simd_id * EXP10_NUM_BINS + digit], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            // ── Phase 1: Within-simdgroup rank via shared_digits ──
            // Count threads in my SG with lower lane index and same digit
            // Uses shared_digits[] (already barrier'd) instead of simd_shuffle
            uint within_rank = 0u;
            uint sg_base = simd_id * 32u;
            for (uint j = 0u; j < simd_lane; j++) {
                within_rank += (shared_digits[sg_base + j] == digit) ? 1u : 0u;
            }

            // ── Phase 2: Cross-simdgroup rank from histogram ──
            // Sum counts from all earlier simdgroups for my digit
            uint cross_rank = 0u;
            for (uint s = 0u; s < simd_id; s++) {
                cross_rank += atomic_load_explicit(
                    &sg_hist[s * EXP10_NUM_BINS + digit], memory_order_relaxed);
            }

            uint rank = cross_rank + within_rank;

            // ── Scatter ──
            uint bin_excl;
            if (tile_id == 0u) {
                bin_excl = 0u;
            } else {
                uint prev = atomic_load_explicit(
                    &tile_status[ts_off + (tile_id - 1u) * EXP10_NUM_BINS + digit],
                    memory_order_relaxed);
                bin_excl = prev & VALUE_MASK;
            }

            uint pos = global_bin_start[digit] + bin_excl + rank;
            dst[pos] = element;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
