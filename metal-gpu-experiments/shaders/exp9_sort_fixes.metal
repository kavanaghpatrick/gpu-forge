#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 9: Single-Dispatch Coherence Fixes
//
// Exp8 discovered that all-4-passes-in-one-dispatch fails due to stale
// cache reads on ping-pong buffers. This experiment tests whether the
// `volatile` qualifier fixes the issue by preventing read caching.
//
// Version A (baseline): Multi-encoder approach from exp8 — 4 compute
//   encoders in 1 command buffer. Known correct.
//
// Version B: Single-dispatch with `device volatile uint*` for buf_a/buf_b.
//   volatile prevents compiler/hardware read caching, forcing every load
//   to go to device memory. tile_status/counters remain atomic.
//
// Version C: Single-dispatch with plain `device uint*` (non-atomic,
//   non-volatile). Relies on threadgroup_barrier(mem_flags::mem_device)
//   alone. Expected to reproduce exp8's stale-read bug.
// ══════════════════════════════════════════════════════════════════════

#define EXP9_NUM_BINS 256u

struct Exp9Params {
    uint element_count;
    uint num_tiles;
    uint num_tgs;
    uint num_passes;
};

struct Exp9PersistParams {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint num_tgs;
    uint counter_base;  // index into counters array for this pass's work-steal slots
    uint ts_offset;     // index into tile_status for this pass's section
};

// ── Stable rank: count threads before lid with same digit ────────────
inline uint exp9_stable_rank(uint lid, uint digit,
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
inline void exp9_global_sync(device atomic_uint* counter,
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
// Version A: Single-pass persistent kernel (for multi-encoder baseline)
//
// Identical to exp8_persistent_pass. 64 persistent TGs loop over
// all tiles via work-stealing for ONE radix pass. Host dispatches
// this 4 times (once per pass) via separate compute encoders.
// ═══════════════════════════════════════════════════════════════════

kernel void exp9_persistent_pass(
    device const uint* src          [[buffer(0)]],
    device uint* dst                [[buffer(1)]],
    device atomic_uint* tile_status [[buffer(2)]],
    device atomic_uint* counters    [[buffer(3)]],
    constant Exp9PersistParams& params [[buffer(4)]],
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
    threadgroup atomic_uint local_hist[EXP9_NUM_BINS];
    threadgroup uint shared_digits[TILE_SIZE];
    threadgroup uint shared_tile_id;
    threadgroup uint global_bin_start[EXP9_NUM_BINS];
    threadgroup uint sg_totals[8];

    // ═══ Loop 1: Histogram + Decoupled Lookback ═══
    while (true) {
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
            &tile_status[ts_off + tile_id * EXP9_NUM_BINS + lid],
            packed_agg, memory_order_relaxed);

        // Decoupled lookback
        if (tile_id == 0u) {
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT)
                            | (my_count & VALUE_MASK);
            atomic_store_explicit(&tile_status[ts_off + lid],
                                  packed_pfx, memory_order_relaxed);
        } else {
            uint running = 0u;
            int lb = (int)tile_id - 1;
            while (lb >= 0) {
                uint status = atomic_load_explicit(
                    &tile_status[ts_off + lb * EXP9_NUM_BINS + lid],
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
                &tile_status[ts_off + tile_id * EXP9_NUM_BINS + lid],
                packed_pfx, memory_order_relaxed);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ═══ Sync: All tiles histogrammed, all lookbacks complete ═══
    exp9_global_sync(barrier_ctr, num_tgs, lid);

    // ═══ Compute global_bin_start ═══
    uint last_status = atomic_load_explicit(
        &tile_status[ts_off + (num_tiles - 1u) * EXP9_NUM_BINS + lid],
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
            uint rank = exp9_stable_rank(lid, digit, shared_digits);

            uint bin_excl;
            if (tile_id == 0u) {
                bin_excl = 0u;
            } else {
                uint prev = atomic_load_explicit(
                    &tile_status[ts_off + (tile_id - 1u) * EXP9_NUM_BINS + digit],
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
// Version B: Single-dispatch megasort with `volatile` data buffers
//
// All 4 radix passes in one kernel dispatch.
// buf_a and buf_b are `device volatile uint*` — the volatile qualifier
// prevents the GPU from caching reads, forcing every load to go through
// to device memory. This should fix the stale-read bug that killed
// exp8's single-dispatch attempt.
//
// tile_status uses atomics for decoupled lookback protocol.
// counters layout:
//   [0..7]  = work-steal counters (pass*2 + {0,1})
//   [8..21] = barrier counters (each used once, never reset)
// ═══════════════════════════════════════════════════════════════════

kernel void exp9_megasort_volatile(
    device volatile uint* buf_a        [[buffer(0)]],
    device volatile uint* buf_b        [[buffer(1)]],
    device atomic_uint* tile_status    [[buffer(2)]],
    device atomic_uint* counters       [[buffer(3)]],
    constant Exp9Params& params        [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint num_tiles   = params.num_tiles;
    uint num_tgs     = params.num_tgs;
    uint num_passes  = params.num_passes;
    uint elem_count  = params.element_count;

    uint ts_section = num_tiles * EXP9_NUM_BINS;  // tile_status entries per pass

    // Threadgroup scratch (reused across all passes)
    threadgroup atomic_uint local_hist[EXP9_NUM_BINS];
    threadgroup uint shared_digits[TILE_SIZE];
    threadgroup uint shared_tile_id;
    threadgroup uint global_bin_start[EXP9_NUM_BINS];
    threadgroup uint sg_totals[8];

    // Barrier counter index — each global_sync uses a unique counter
    uint barrier_idx = 8u;  // counters[8..] are for barriers

    for (uint pass = 0u; pass < num_passes; pass++) {
        uint shift = pass * 8u;
        uint ts_off = pass * ts_section;

        // Work-steal counters for this pass
        device atomic_uint* ws1 = &counters[pass * 2u];
        device atomic_uint* ws2 = &counters[pass * 2u + 1u];

        // Ping-pong: even passes read a→write b, odd passes read b→write a
        device volatile uint* src = (pass % 2u == 0u) ? buf_a : buf_b;
        device volatile uint* dst = (pass % 2u == 0u) ? buf_b : buf_a;

        // ═══ Loop 1: Histogram + Decoupled Lookback ═══
        while (true) {
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
            bool valid = (base + lid) < elem_count;

            if (valid) {
                uint val = src[base + lid];  // volatile read
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
                &tile_status[ts_off + tile_id * EXP9_NUM_BINS + lid],
                packed_agg, memory_order_relaxed);

            // Decoupled lookback
            if (tile_id == 0u) {
                uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT)
                                | (my_count & VALUE_MASK);
                atomic_store_explicit(&tile_status[ts_off + lid],
                                      packed_pfx, memory_order_relaxed);
            } else {
                uint running = 0u;
                int lb = (int)tile_id - 1;
                while (lb >= 0) {
                    uint status = atomic_load_explicit(
                        &tile_status[ts_off + lb * EXP9_NUM_BINS + lid],
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
                    &tile_status[ts_off + tile_id * EXP9_NUM_BINS + lid],
                    packed_pfx, memory_order_relaxed);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ═══ Sync A: All tiles histogrammed ═══
        exp9_global_sync(&counters[barrier_idx], num_tgs, lid);
        barrier_idx++;

        // ═══ Compute global_bin_start ═══
        uint last_status = atomic_load_explicit(
            &tile_status[ts_off + (num_tiles - 1u) * EXP9_NUM_BINS + lid],
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

        uint inc_val = simd_scan;
        if (simd_id > 0u) inc_val += sg_totals[simd_id - 1u];
        global_bin_start[lid] = inc_val - global_total_val;
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
            bool valid = (base + lid) < elem_count;
            uint element = 0u;
            uint digit = 0u;

            if (valid) {
                element = src[base + lid];  // volatile read
                digit = (element >> shift) & 0xFFu;
            }
            shared_digits[lid] = valid ? digit : 0xFFFFu;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (valid) {
                uint rank = exp9_stable_rank(lid, digit, shared_digits);

                uint bin_excl;
                if (tile_id == 0u) {
                    bin_excl = 0u;
                } else {
                    uint prev = atomic_load_explicit(
                        &tile_status[ts_off + (tile_id - 1u) * EXP9_NUM_BINS + digit],
                        memory_order_relaxed);
                    bin_excl = prev & VALUE_MASK;
                }

                uint pos = global_bin_start[digit] + bin_excl + rank;
                dst[pos] = element;  // volatile write
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Between passes: sync + clear tile_status for next pass
        if (pass < num_passes - 1u) {
            // ═══ Sync B: All scatter complete ═══
            exp9_global_sync(&counters[barrier_idx], num_tgs, lid);
            barrier_idx++;

            // Clear tile_status for the NEXT pass cooperatively
            // Next pass's ts_off = (pass + 1) * ts_section
            uint next_ts_off = (pass + 1u) * ts_section;
            uint total_ts_entries = ts_section;  // num_tiles * EXP9_NUM_BINS
            for (uint i = gid * TILE_SIZE + lid; i < total_ts_entries;
                 i += num_tgs * TILE_SIZE) {
                atomic_store_explicit(&tile_status[next_ts_off + i],
                                      0u, memory_order_relaxed);
            }

            // Also clear next pass's work-steal counters
            if (gid == 0u && lid < 2u) {
                atomic_store_explicit(&counters[(pass + 1u) * 2u + lid],
                                      0u, memory_order_relaxed);
            }

            // ═══ Sync C: tile_status cleared ═══
            exp9_global_sync(&counters[barrier_idx], num_tgs, lid);
            barrier_idx++;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Version C: Single-dispatch megasort with plain (non-volatile) data buffers
//
// Same algorithm as Version B, but buf_a/buf_b are plain `device uint*`.
// No volatile, no atomics on data buffers — relies solely on
// threadgroup_barrier(mem_flags::mem_device) for cross-TG visibility.
//
// This is expected to FAIL with stale cache reads, reproducing exp8's bug.
// If it passes, it means the bug was something else entirely.
// ═══════════════════════════════════════════════════════════════════

kernel void exp9_megasort_plain(
    device uint* buf_a                 [[buffer(0)]],
    device uint* buf_b                 [[buffer(1)]],
    device atomic_uint* tile_status    [[buffer(2)]],
    device atomic_uint* counters       [[buffer(3)]],
    constant Exp9Params& params        [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint num_tiles   = params.num_tiles;
    uint num_tgs     = params.num_tgs;
    uint num_passes  = params.num_passes;
    uint elem_count  = params.element_count;

    uint ts_section = num_tiles * EXP9_NUM_BINS;

    // Threadgroup scratch (reused across all passes)
    threadgroup atomic_uint local_hist[EXP9_NUM_BINS];
    threadgroup uint shared_digits[TILE_SIZE];
    threadgroup uint shared_tile_id;
    threadgroup uint global_bin_start[EXP9_NUM_BINS];
    threadgroup uint sg_totals[8];

    uint barrier_idx = 8u;

    for (uint pass = 0u; pass < num_passes; pass++) {
        uint shift = pass * 8u;
        uint ts_off = pass * ts_section;

        device atomic_uint* ws1 = &counters[pass * 2u];
        device atomic_uint* ws2 = &counters[pass * 2u + 1u];

        // Ping-pong: even passes read a→write b, odd passes read b→write a
        device uint* src = (pass % 2u == 0u) ? buf_a : buf_b;
        device uint* dst = (pass % 2u == 0u) ? buf_b : buf_a;

        // ═══ Loop 1: Histogram + Decoupled Lookback ═══
        while (true) {
            if (lid == 0u) {
                shared_tile_id = atomic_fetch_add_explicit(
                    ws1, 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tile_id = shared_tile_id;
            if (tile_id >= num_tiles) break;

            atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint base = tile_id * TILE_SIZE;
            bool valid = (base + lid) < elem_count;

            if (valid) {
                uint val = src[base + lid];  // plain read — may be stale!
                uint digit = (val >> shift) & 0xFFu;
                atomic_fetch_add_explicit(&local_hist[digit], 1u,
                                          memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint my_count = atomic_load_explicit(&local_hist[lid],
                                                  memory_order_relaxed);

            uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT)
                            | (my_count & VALUE_MASK);
            atomic_store_explicit(
                &tile_status[ts_off + tile_id * EXP9_NUM_BINS + lid],
                packed_agg, memory_order_relaxed);

            if (tile_id == 0u) {
                uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT)
                                | (my_count & VALUE_MASK);
                atomic_store_explicit(&tile_status[ts_off + lid],
                                      packed_pfx, memory_order_relaxed);
            } else {
                uint running = 0u;
                int lb = (int)tile_id - 1;
                while (lb >= 0) {
                    uint status = atomic_load_explicit(
                        &tile_status[ts_off + lb * EXP9_NUM_BINS + lid],
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
                }
                uint inclusive = running + my_count;
                uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT)
                                | (inclusive & VALUE_MASK);
                atomic_store_explicit(
                    &tile_status[ts_off + tile_id * EXP9_NUM_BINS + lid],
                    packed_pfx, memory_order_relaxed);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ═══ Sync A ═══
        exp9_global_sync(&counters[barrier_idx], num_tgs, lid);
        barrier_idx++;

        // ═══ Compute global_bin_start ═══
        uint last_status = atomic_load_explicit(
            &tile_status[ts_off + (num_tiles - 1u) * EXP9_NUM_BINS + lid],
            memory_order_relaxed);
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
            bool valid = (base + lid) < elem_count;
            uint element = 0u;
            uint digit = 0u;

            if (valid) {
                element = src[base + lid];  // plain read — may be stale!
                digit = (element >> shift) & 0xFFu;
            }
            shared_digits[lid] = valid ? digit : 0xFFFFu;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (valid) {
                uint rank = exp9_stable_rank(lid, digit, shared_digits);

                uint bin_excl;
                if (tile_id == 0u) {
                    bin_excl = 0u;
                } else {
                    uint prev = atomic_load_explicit(
                        &tile_status[ts_off + (tile_id - 1u) * EXP9_NUM_BINS + digit],
                        memory_order_relaxed);
                    bin_excl = prev & VALUE_MASK;
                }

                uint pos = global_bin_start[digit] + bin_excl + rank;
                dst[pos] = element;  // plain write — may not be visible!
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Between passes: sync + clear
        if (pass < num_passes - 1u) {
            exp9_global_sync(&counters[barrier_idx], num_tgs, lid);
            barrier_idx++;

            uint next_ts_off = (pass + 1u) * ts_section;
            uint total_ts_entries = ts_section;
            for (uint i = gid * TILE_SIZE + lid; i < total_ts_entries;
                 i += num_tgs * TILE_SIZE) {
                atomic_store_explicit(&tile_status[next_ts_off + i],
                                      0u, memory_order_relaxed);
            }

            if (gid == 0u && lid < 2u) {
                atomic_store_explicit(&counters[(pass + 1u) * 2u + lid],
                                      0u, memory_order_relaxed);
            }

            exp9_global_sync(&counters[barrier_idx], num_tgs, lid);
            barrier_idx++;
        }
    }
}
