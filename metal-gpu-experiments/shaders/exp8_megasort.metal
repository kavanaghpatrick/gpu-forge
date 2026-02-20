#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 8: Single-Dispatch Megasort (1M elements)
//
// Removes exp7's 16K limitation using occupancy-bound dispatch:
// 64 persistent TGs loop over 4096 tiles via atomic work-stealing.
//
// Version A: 8 dispatches (2 per pass × 4 = 8 total)
//   Traditional: histogram dispatch → CPU prefix sum → scatter dispatch
//
// Version B: 4 persistent dispatches in 1 command buffer
//   Each compute encoder dispatches 64 persistent TGs that loop over
//   all tiles via work-stealing. Metal's inter-encoder barriers provide
//   cross-pass coherence (no cross-TG memory ordering issues).
//
// NOTE: A true single-dispatch version (all 4 passes in one kernel)
// was attempted but failed due to non-deterministic stale cache reads
// on Apple Silicon GPU when re-reading ping-pong buffers across passes
// within a single dispatch. threadgroup_barrier(mem_flags::mem_device)
// provides threadgroup-scoped ordering, which is insufficient for
// cross-TG read cache invalidation within a persistent kernel.
// ══════════════════════════════════════════════════════════════════════

#define EXP8_NUM_BINS 256u

struct Exp8PassParams {
    uint element_count;
    uint num_tiles;
    uint shift;
};

struct Exp8PersistParams {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint num_tgs;
    uint counter_base;  // index into counters array for this pass's 3 slots
    uint ts_offset;     // index into tile_status for this pass's section
};

// ── Stable rank: count threads before lid with same digit ────────────
inline uint exp8_stable_rank(uint lid, uint digit,
                             threadgroup uint* shared_digits)
{
    uint rank = 0;
    for (uint i = 0; i < lid; i++) {
        rank += (shared_digits[i] == digit) ? 1u : 0u;
    }
    return rank;
}

// ── Global barrier for persistent kernels ────────────────────────────
// Each barrier uses its own dedicated atomic counter (never reset).
// Safe: counter increments monotonically, each used exactly once.
inline void global_sync(device atomic_uint* counter,
                        uint num_tgs, uint lid)
{
    // Flush this TG's device writes before signaling arrival
    threadgroup_barrier(mem_flags::mem_device);
    if (lid == 0u) {
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        while (atomic_load_explicit(counter, memory_order_relaxed) < num_tgs) {}
    }
    // Ensure all threads wait + see other TGs' device writes
    threadgroup_barrier(mem_flags::mem_device);
}

// ═══════════════════════════════════════════════════════════════════
// Version A: Traditional multi-dispatch kernels
// ═══════════════════════════════════════════════════════════════════

kernel void exp8_histogram(
    device const uint* input          [[buffer(0)]],
    device uint* tile_histograms      [[buffer(1)]],
    constant Exp8PassParams& params   [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint local_hist[EXP8_NUM_BINS];
    atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < params.element_count) {
        uint digit = (input[tid] >> params.shift) & 0xFFu;
        atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tile_histograms[gid * EXP8_NUM_BINS + lid] =
        atomic_load_explicit(&local_hist[lid], memory_order_relaxed);
}

kernel void exp8_scatter(
    device const uint* input          [[buffer(0)]],
    device uint* output               [[buffer(1)]],
    device const uint* offsets        [[buffer(2)]],
    constant Exp8PassParams& params   [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup uint shared_digits[TILE_SIZE];

    uint element = 0u;
    uint digit = 0u;
    bool valid = (tid < params.element_count);

    if (valid) {
        element = input[tid];
        digit = (element >> params.shift) & 0xFFu;
    }
    shared_digits[lid] = valid ? digit : 0xFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        uint rank = exp8_stable_rank(lid, digit, shared_digits);
        output[offsets[gid * EXP8_NUM_BINS + digit] + rank] = element;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Version B: Persistent single-pass kernel with work-stealing
//
// 64 persistent TGs × 256 threads process all tiles for ONE pass.
// Host dispatches this kernel 4 times (once per radix pass) via
// separate compute encoders within a single command buffer.
//
// src/dst are regular device pointers (not atomic) — cross-pass
// coherence is guaranteed by Metal's inter-encoder barriers.
// tile_status uses atomics for the decoupled lookback protocol.
// ═══════════════════════════════════════════════════════════════════

kernel void exp8_persistent_pass(
    device const uint* src       [[buffer(0)]],
    device uint* dst             [[buffer(1)]],
    device atomic_uint* tile_status [[buffer(2)]],
    device atomic_uint* counters    [[buffer(3)]],
    constant Exp8PersistParams& params [[buffer(4)]],
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

    // Threadgroup scratch
    threadgroup atomic_uint local_hist[EXP8_NUM_BINS];
    threadgroup uint shared_digits[TILE_SIZE];
    threadgroup uint shared_tile_id;
    threadgroup uint global_bin_start[EXP8_NUM_BINS];
    threadgroup uint sg_totals[8];

    // ═══ Loop 1: Histogram + Decoupled Lookback ═══
    while (true) {
        // Grab next tile via atomic work-stealing
        if (lid == 0u) {
            shared_tile_id = atomic_fetch_add_explicit(
                ws1, 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tile_id = shared_tile_id;
        if (tile_id >= num_tiles) break;

        // Clear local histogram
        atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load element and accumulate histogram
        uint base = tile_id * TILE_SIZE;
        bool valid = (base + lid) < params.element_count;

        if (valid) {
            uint val = src[base + lid];
            uint digit = (val >> shift) & 0xFFu;
            atomic_fetch_add_explicit(&local_hist[digit], 1u,
                                      memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread lid handles lookback for bin lid
        uint my_count = atomic_load_explicit(&local_hist[lid],
                                              memory_order_relaxed);

        // Publish aggregate flag
        uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT)
                        | (my_count & VALUE_MASK);
        atomic_store_explicit(
            &tile_status[ts_off + tile_id * EXP8_NUM_BINS + lid],
            packed_agg, memory_order_relaxed);

        // Decoupled lookback
        if (tile_id == 0u) {
            // First tile: inclusive prefix = local count
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT)
                            | (my_count & VALUE_MASK);
            atomic_store_explicit(&tile_status[ts_off + lid],
                                  packed_pfx, memory_order_relaxed);
        } else {
            uint running = 0u;
            int lb = (int)tile_id - 1;
            while (lb >= 0) {
                uint status = atomic_load_explicit(
                    &tile_status[ts_off + lb * EXP8_NUM_BINS + lid],
                    memory_order_relaxed);
                uint flag  = status >> FLAG_SHIFT;
                uint value = status & VALUE_MASK;

                if (flag == FLAG_PREFIX) {
                    running += value;
                    break;
                } else if (flag == FLAG_AGGREGATE) {
                    running += value;
                    lb--;
                }
                // FLAG_NOT_READY: spin
            }
            uint inclusive = running + my_count;
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT)
                            | (inclusive & VALUE_MASK);
            atomic_store_explicit(
                &tile_status[ts_off + tile_id * EXP8_NUM_BINS + lid],
                packed_pfx, memory_order_relaxed);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ═══ Sync: All tiles histogrammed, all lookbacks complete ═══
    global_sync(barrier_ctr, num_tgs, lid);

    // ═══ Compute global_bin_start ═══
    // Last tile's inclusive prefix = global total for each bin
    uint last_status = atomic_load_explicit(
        &tile_status[ts_off + (num_tiles - 1u) * EXP8_NUM_BINS + lid],
        memory_order_relaxed);
    uint global_total_val = last_status & VALUE_MASK;

    // SIMD-based exclusive prefix sum over 256 bins
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

    // ═══ Loop 2: Scatter ═══
    while (true) {
        if (lid == 0u) {
            shared_tile_id = atomic_fetch_add_explicit(
                ws2, 1u, memory_order_relaxed);
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
            uint rank = exp8_stable_rank(lid, digit, shared_digits);

            // Exclusive prefix for this tile's bin:
            // = inclusive prefix of (tile_id - 1), or 0 for tile 0
            uint bin_excl;
            if (tile_id == 0u) {
                bin_excl = 0u;
            } else {
                uint prev = atomic_load_explicit(
                    &tile_status[ts_off + (tile_id - 1u) * EXP8_NUM_BINS + digit],
                    memory_order_relaxed);
                bin_excl = prev & VALUE_MASK;
            }

            uint pos = global_bin_start[digit] + bin_excl + rank;
            dst[pos] = element;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
