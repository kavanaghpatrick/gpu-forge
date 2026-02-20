#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 15: Onesweep-style Fused Radix Sort
//
// Core innovation: combined histogram (all passes in 1 dispatch) +
// fused partition kernel (load + histogram + lookback + rank + scatter).
// Reduces memory ops from 24n (V5) to 17n = 29% bandwidth reduction.
//
// Architecture: non-persistent dispatch (1 TG/tile), decoupled lookback
// with device-scope fence for cross-TG coherence (proven in exp12).
// 4-bit radix, 8 passes, 2048 elements/tile, 256 threads/TG.
// ═══════════════════════════════════════════════════════════════════

#define EXP15_NUM_BINS   16u
#define EXP15_NUM_SGS    8u
#define EXP15_TILE_SIZE  2048u
#define EXP15_ELEMS      8u
#define EXP15_NUM_PASSES 8u

struct Exp15Params {
    uint element_count;
    uint num_tiles;
    uint num_tgs;
    uint shift;
    uint pass;
};

// Ballot via butterfly XOR reduction (proven in exp14)
inline uint exp15_ballot(bool pred, uint simd_lane) {
    uint my_bit = pred ? (1u << simd_lane) : 0u;
    my_bit |= simd_shuffle_xor(my_bit, 1u);
    my_bit |= simd_shuffle_xor(my_bit, 2u);
    my_bit |= simd_shuffle_xor(my_bit, 4u);
    my_bit |= simd_shuffle_xor(my_bit, 8u);
    my_bit |= simd_shuffle_xor(my_bit, 16u);
    return my_bit;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Combined Histogram — reads ALL data once, computes
// histograms for ALL 8 passes simultaneously.
// Uses register-based private counting + SIMD butterfly reduction.
// Output: global_hist[pass * 16 + digit] = total count
// ═══════════════════════════════════════════════════════════════════

kernel void exp15_combined_histogram(
    device const uint*     src          [[buffer(0)]],
    device atomic_uint*    global_hist  [[buffer(1)]],
    constant Exp15Params&  params       [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint n = params.element_count;
    uint base = gid * EXP15_TILE_SIZE;

    // Load 8 elements into registers (one global memory read)
    uint keys[EXP15_ELEMS];
    bool valid[EXP15_ELEMS];
    for (uint e = 0u; e < EXP15_ELEMS; e++) {
        uint idx = base + e * 256u + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    // Per-SG accumulator in shared memory (reused per pass)
    threadgroup uint sg_counts[EXP15_NUM_SGS * EXP15_NUM_BINS]; // 512B

    // Process each pass: extract digits, reduce, atomically add to global
    for (uint p = 0u; p < EXP15_NUM_PASSES; p++) {
        // Private per-thread counters (16 bins)
        uint my_count[EXP15_NUM_BINS];
        for (uint d = 0u; d < EXP15_NUM_BINS; d++) my_count[d] = 0u;

        for (uint e = 0u; e < EXP15_ELEMS; e++) {
            if (valid[e]) {
                uint digit = (keys[e] >> (p * 4u)) & 0xFu;
                my_count[digit]++;
            }
        }

        // SIMD butterfly reduction per bin
        for (uint d = 0u; d < EXP15_NUM_BINS; d++) {
            uint val = my_count[d];
            val += simd_shuffle_xor(val, 1u);
            val += simd_shuffle_xor(val, 2u);
            val += simd_shuffle_xor(val, 4u);
            val += simd_shuffle_xor(val, 8u);
            val += simd_shuffle_xor(val, 16u);
            if (simd_lane == 0u) {
                sg_counts[simd_id * EXP15_NUM_BINS + d] = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduce across SGs (first 16 threads)
        if (lid < EXP15_NUM_BINS) {
            uint total = 0u;
            for (uint sg = 0u; sg < EXP15_NUM_SGS; sg++) {
                total += sg_counts[sg * EXP15_NUM_BINS + lid];
            }
            if (total > 0u) {
                atomic_fetch_add_explicit(&global_hist[p * EXP15_NUM_BINS + lid],
                                          total, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Global Prefix — exclusive prefix sum per pass
// Input/output: global_hist[pass * 16 + digit]
// ═══════════════════════════════════════════════════════════════════

kernel void exp15_global_prefix(
    device uint* global_hist [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // 8 SGs handle 8 passes, each SG does 16-bin exclusive prefix sum
    if (simd_id < EXP15_NUM_PASSES && simd_lane < EXP15_NUM_BINS) {
        uint p = simd_id;
        uint count = global_hist[p * EXP15_NUM_BINS + simd_lane];
        uint prefix = simd_prefix_exclusive_sum(count);
        global_hist[p * EXP15_NUM_BINS + simd_lane] = prefix;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Zero tile status + work counter between passes
// ═══════════════════════════════════════════════════════════════════

kernel void exp15_zero_status(
    device uint*           tile_status  [[buffer(0)]],
    device atomic_uint*    counters     [[buffer(1)]],
    constant Exp15Params&  params       [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total_entries = params.num_tiles * EXP15_NUM_BINS;
    if (tid < total_entries) {
        tile_status[tid] = 0u;
    }
    if (tid == 0u) {
        atomic_store_explicit(&counters[0], 0u, memory_order_relaxed);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: Fused Partition — non-persistent, one TG per tile
//
// Flow per tile: load → register histogram → publish AGGREGATE →
// decoupled lookback → publish PREFIX → per-SG atomic rank + scatter.
//
// Key insight: Metal compiler generates catastrophically bad code (~4000x)
// when simd_shuffle_xor and TG memory ops coexist in the same loop body.
// Solution: per-SG atomic_fetch_add ranking (no SIMD shuffles in scatter).
// Atomics on TG memory within a simdgroup serialize in lane order on
// Apple Silicon, giving stable ranking for LSD radix sort.
//
// Non-persistent dispatch: num_tiles TGs, tile_id = gid.
// Device-scope fence for cross-TG coherence (exp12 proven).
// ═══════════════════════════════════════════════════════════════════

kernel void exp15_partition(
    device const uint*     src          [[buffer(0)]],
    device uint*           dst          [[buffer(1)]],
    device atomic_uint*    tile_status  [[buffer(2)]],
    device atomic_uint*    counters     [[buffer(3)]],
    device const uint*     global_hist  [[buffer(4)]],
    constant Exp15Params&  params       [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n          = params.element_count;
    uint shift      = params.shift;
    uint pass       = params.pass;

    // ── Shared memory (~1.2 KB) ────────────────────────────────
    threadgroup uint sg_counts[EXP15_NUM_SGS * EXP15_NUM_BINS]; // 512B per-SG hist
    threadgroup uint sg_prefix[EXP15_NUM_SGS * EXP15_NUM_BINS]; // 512B cross-SG prefix
    threadgroup uint tile_hist[EXP15_NUM_BINS];          // 64B
    threadgroup uint exclusive_pfx[EXP15_NUM_BINS];      // 64B lookback result

    // Non-persistent: one TG per tile, tile_id = gid
    uint tile_id = gid;
    uint base = tile_id * EXP15_TILE_SIZE;
    {

        // ── Phase 1: Load 8 elements (SG-contiguous layout) ──
        // Keys in mk[], digits in md[], validity in mv[].
        uint mk[EXP15_ELEMS];
        uint md[EXP15_ELEMS];
        bool mv[EXP15_ELEMS];
        for (int e = 0; e < (int)EXP15_ELEMS; e++) {
            uint idx = base + simd_id * 256u + (uint)e * 32u + simd_lane;
            mv[e] = idx < n;
            mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
            md[e] = mv[e] ? ((mk[e] >> shift) & 0xFu) : 0xFu;
        }

        // ── Phase 2: Register-based per-SG histogram ──────────
        // Private per-thread counters → SIMD butterfly reduction
        // (replaces atomic histogram: eliminates atomics + 2 barriers)
        uint my_count[EXP15_NUM_BINS];
        for (uint d = 0u; d < EXP15_NUM_BINS; d++) my_count[d] = 0u;
        for (int e = 0; e < (int)EXP15_ELEMS; e++) {
            if (mv[e]) my_count[md[e]]++;
        }

        for (uint d = 0u; d < EXP15_NUM_BINS; d++) {
            uint val = my_count[d];
            val += simd_shuffle_xor(val, 1u);
            val += simd_shuffle_xor(val, 2u);
            val += simd_shuffle_xor(val, 4u);
            val += simd_shuffle_xor(val, 8u);
            val += simd_shuffle_xor(val, 16u);
            if (simd_lane == 0u) {
                sg_counts[simd_id * EXP15_NUM_BINS + d] = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // B2: sg_counts ready

        // ── Phase 2b: tile_hist + cross-SG exclusive prefix ───
        if (lid < EXP15_NUM_SGS * EXP15_NUM_BINS) {
            uint sg = lid / EXP15_NUM_BINS;
            uint d  = lid % EXP15_NUM_BINS;
            uint acc = 0u;
            for (uint s = 0u; s < sg; s++) acc += sg_counts[s * EXP15_NUM_BINS + d];
            sg_prefix[lid] = acc;
        }
        if (lid < EXP15_NUM_BINS) {
            uint total = 0u;
            for (uint sg = 0u; sg < EXP15_NUM_SGS; sg++) {
                total += sg_counts[sg * EXP15_NUM_BINS + lid];
            }
            tile_hist[lid] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // B3: tile_hist + sg_prefix ready

        // ── Phase 3: Publish AGGREGATE ────────────────────────
        if (lid < EXP15_NUM_BINS) {
            uint packed = (FLAG_AGGREGATE << FLAG_SHIFT)
                        | (tile_hist[lid] & VALUE_MASK);
            atomic_store_explicit(&tile_status[tile_id * EXP15_NUM_BINS + lid],
                                  packed, memory_order_relaxed);
        }
        atomic_thread_fence(mem_flags::mem_device,
                            memory_order_seq_cst, thread_scope_device);
        threadgroup_barrier(mem_flags::mem_threadgroup);  // B4: AGGREGATE published

        // ── Phase 4: Decoupled lookback (16 threads) ─────────
        if (lid < EXP15_NUM_BINS) {
            uint lk_running = 0u;
            if (tile_id > 0u) {
                int look = (int)tile_id - 1;
                while (look >= 0) {
                    atomic_thread_fence(mem_flags::mem_device,
                                        memory_order_seq_cst,
                                        thread_scope_device);
                    uint val = atomic_load_explicit(
                        &tile_status[(uint)look * EXP15_NUM_BINS + lid],
                        memory_order_relaxed);
                    uint flag  = val >> FLAG_SHIFT;
                    uint count = val & VALUE_MASK;

                    if (flag == FLAG_PREFIX) {
                        lk_running += count;
                        break;
                    } else if (flag == FLAG_AGGREGATE) {
                        lk_running += count;
                        look--;
                    }
                }
            }
            exclusive_pfx[lid] = lk_running;

            uint inclusive = lk_running + tile_hist[lid];
            uint packed = (FLAG_PREFIX << FLAG_SHIFT)
                        | (inclusive & VALUE_MASK);
            atomic_store_explicit(&tile_status[tile_id * EXP15_NUM_BINS + lid],
                                  packed, memory_order_relaxed);
        }
        atomic_thread_fence(mem_flags::mem_device,
                            memory_order_seq_cst, thread_scope_device);
        threadgroup_barrier(mem_flags::mem_threadgroup);  // B5: PREFIX + exclusive_pfx ready

        // (bin_start removed — not needed with per-SG atomic ranking)

        // ── Phase 6: Per-SG atomic ranking + scatter ──
        // Uses atomic_fetch_add on per-SG counters for within-SG ranking.
        // No SIMD shuffles in scatter loop — avoids Metal compiler pathology.
        // Atomics within a simdgroup on TG memory serialize in lane order
        // on Apple Silicon, giving stable (lane-ordered) ranking.
        // Cross-iteration ordering is guaranteed by sequential execution.
        threadgroup atomic_uint sg_rank_ctr[EXP15_NUM_SGS * EXP15_NUM_BINS];
        if (lid < EXP15_NUM_SGS * EXP15_NUM_BINS) {
            atomic_store_explicit(&sg_rank_ctr[lid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // B7: rank counters zeroed

        for (int e = 0; e < (int)EXP15_ELEMS; e++) {
            if (mv[e]) {
                uint d = md[e];
                uint within_sg = atomic_fetch_add_explicit(
                    &sg_rank_ctr[simd_id * EXP15_NUM_BINS + d],
                    1u, memory_order_relaxed);
                uint gp = global_hist[pass * EXP15_NUM_BINS + d]
                         + exclusive_pfx[d]
                         + sg_prefix[simd_id * EXP15_NUM_BINS + d]
                         + within_sg;
                dst[gp] = mk[e];
            }
        }
    }  // end scope
}
