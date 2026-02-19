#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 16: 8-Bit Radix Sort (256-bin, 4-pass)
//
// Core improvement over exp15: 8-bit digit (256 bins) with 4 passes
// instead of 4-bit (16 bins) with 8 passes. Halves bandwidth by
// halving pass count. 256-bin histogram uses per-SG atomic counting
// on TG memory (NOT private counters — 256 regs would spill).
//
// Architecture: non-persistent dispatch (1 TG/tile), decoupled lookback
// with device-scope fence for cross-TG coherence (proven in exp12).
// 8-bit radix, 4 passes, 2048 elements/tile, 256 threads/TG.
// ═══════════════════════════════════════════════════════════════════

#define EXP16_NUM_BINS   256u
#define EXP16_NUM_SGS    8u
#define EXP16_TILE_SIZE  2048u
#define EXP16_ELEMS      8u
#define EXP16_NUM_PASSES 4u
#define EXP16_THREADS    256u

struct Exp16Params {
    uint element_count;
    uint num_tiles;
    uint num_tgs;
    uint shift;
    uint pass;
};

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Combined Histogram — reads ALL data once, computes
// 256-bin histograms for ALL 4 passes simultaneously.
// Uses per-SG atomic histogram on TG memory (NOT private counters —
// 256 private counters would cause instant register spill).
// Output: global_hist[pass * 256 + digit] = total count
// ═══════════════════════════════════════════════════════════════════

kernel void exp16_combined_histogram(
    device const uint*     src          [[buffer(0)]],
    device atomic_uint*    global_hist  [[buffer(1)]],
    constant Exp16Params&  params       [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint n = params.element_count;
    uint base = gid * EXP16_TILE_SIZE;

    // Load 8 elements into registers (one global memory read)
    uint keys[EXP16_ELEMS];
    bool valid[EXP16_ELEMS];
    for (uint e = 0u; e < EXP16_ELEMS; e++) {
        uint idx = base + e * EXP16_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    // Per-SG accumulator in shared memory (reused per pass)
    threadgroup atomic_uint sg_counts[EXP16_NUM_SGS * EXP16_NUM_BINS]; // 8 KB

    // Process each pass: extract digits, per-SG atomic count, reduce, global add
    for (uint p = 0u; p < EXP16_NUM_PASSES; p++) {

        // Zero sg_counts (all 256 threads cooperate)
        for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
            atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-SG atomic histogram (no private counters, no SIMD shuffle)
        for (uint e = 0u; e < EXP16_ELEMS; e++) {
            if (valid[e]) {
                uint digit = (keys[e] >> (p * 8u)) & 0xFFu;
                atomic_fetch_add_explicit(
                    &sg_counts[simd_id * EXP16_NUM_BINS + digit],
                    1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduce across SGs: 256 threads handle one bin each
        {
            uint total = 0u;
            for (uint sg = 0u; sg < EXP16_NUM_SGS; sg++) {
                total += atomic_load_explicit(
                    &sg_counts[sg * EXP16_NUM_BINS + lid],
                    memory_order_relaxed);
            }
            if (total > 0u) {
                atomic_fetch_add_explicit(&global_hist[p * EXP16_NUM_BINS + lid],
                                          total, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Global Prefix — exclusive prefix sum per pass
// 256 bins per pass via 8 chunks of 32 with simd_prefix_exclusive_sum.
// 4 SGs (one per pass) process in parallel. Input/output: global_hist.
// ═══════════════════════════════════════════════════════════════════

kernel void exp16_global_prefix(
    device uint* global_hist [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // 4 SGs handle 4 passes, each SG does 256-bin exclusive prefix sum
    // via 8 serial chunks of 32 bins (one SIMD width each)
    if (simd_id < EXP16_NUM_PASSES) {
        uint p = simd_id;
        uint running = 0u;

        for (uint chunk = 0u; chunk < 8u; chunk++) {
            uint bin = chunk * 32u + simd_lane;
            uint val = global_hist[p * EXP16_NUM_BINS + bin];
            uint prefix = simd_prefix_exclusive_sum(val) + running;
            global_hist[p * EXP16_NUM_BINS + bin] = prefix;
            // Broadcast chunk total from lane 31 (uniform lane — safe)
            running += simd_shuffle(simd_prefix_exclusive_sum(val) + val, 31u);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Zero tile status + work counter between passes
// ═══════════════════════════════════════════════════════════════════

kernel void exp16_zero_status(
    device uint*           tile_status  [[buffer(0)]],
    device atomic_uint*    counters     [[buffer(1)]],
    constant Exp16Params&  params       [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total_entries = params.num_tiles * EXP16_NUM_BINS;
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
// Flow per tile: load → per-SG atomic histogram → cross-SG prefix →
// publish AGGREGATE → decoupled lookback → publish PREFIX →
// per-SG atomic rank + scatter.
//
// Key design: 256-bin histogram uses per-SG atomic_fetch_add on TG
// memory (not private counters). No SIMD shuffles in histogram or
// scatter loops — avoids Metal compiler pathology (~4000x slowdown).
//
// TG memory layout (18 KB total):
//   sg_hist_or_rank[8*256] = 8 KB  (sg_counts in P2, sg_rank_ctr in P5)
//   sg_prefix[8*256]       = 8 KB  (cross-SG prefix, live P2b-P5)
//   tile_hist[256]         = 1 KB  (tile totals, live P2b-P4)
//   exclusive_pfx[256]     = 1 KB  (lookback result, live P4-P5)
//
// Non-persistent dispatch: num_tiles TGs, tile_id = gid.
// Device-scope fence for cross-TG coherence (exp12 proven).
// ═══════════════════════════════════════════════════════════════════

kernel void exp16_partition(
    device const uint*     src          [[buffer(0)]],
    device uint*           dst          [[buffer(1)]],
    device atomic_uint*    tile_status  [[buffer(2)]],
    device atomic_uint*    counters     [[buffer(3)]],
    device const uint*     global_hist  [[buffer(4)]],
    constant Exp16Params&  params       [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n     = params.element_count;
    uint shift = params.shift;
    uint pass  = params.pass;

    // ── TG Memory (18 KB total) ──────────────────────────────────
    // sg_hist_or_rank: sg_counts in P2/P2b, sg_rank_ctr in P5
    // (non-overlapping lifetimes, declared as atomic_uint for P5)
    threadgroup atomic_uint sg_hist_or_rank[EXP16_NUM_SGS * EXP16_NUM_BINS]; // 8 KB
    threadgroup uint sg_prefix[EXP16_NUM_SGS * EXP16_NUM_BINS];             // 8 KB
    threadgroup uint tile_hist[EXP16_NUM_BINS];                              // 1 KB
    threadgroup uint exclusive_pfx[EXP16_NUM_BINS];                          // 1 KB

    // Non-persistent: one TG per tile, tile_id = gid
    uint tile_id = gid;
    uint base = tile_id * EXP16_TILE_SIZE;

    // ── Phase 1: Load 8 elements (SG-contiguous layout) ──────────
    uint mk[EXP16_ELEMS];
    uint md[EXP16_ELEMS];
    bool mv[EXP16_ELEMS];
    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        uint idx = base + simd_id * 256u + (uint)e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG atomic histogram on TG memory ────────────
    // Zero sg_hist_or_rank (all 256 threads cooperate)
    for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread: 8 atomic increments on its SG's histogram
    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP16_NUM_BINS + md[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // B2: sg_counts ready

    // ── Phase 2b: Tile histogram + cross-SG exclusive prefix ─────
    // 256 threads, each handles one bin (lid = bin index)
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP16_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * EXP16_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * EXP16_NUM_BINS + lid] = total;  // exclusive prefix
            total += c;
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // B3: tile_hist + sg_prefix ready

    // ── Phase 3: Publish AGGREGATE ───────────────────────────────
    // 256 threads publish 256 bins simultaneously
    {
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT)
                    | (tile_hist[lid] & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * EXP16_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);  // B4: AGGREGATE published

    // ── Phase 4: Decoupled lookback (ALL 256 threads, one per bin)
    {
        uint lk_running = 0u;
        if (tile_id > 0u) {
            int look = (int)tile_id - 1;
            while (look >= 0) {
                atomic_thread_fence(mem_flags::mem_device,
                                    memory_order_seq_cst,
                                    thread_scope_device);
                uint val = atomic_load_explicit(
                    &tile_status[(uint)look * EXP16_NUM_BINS + lid],
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
                // FLAG_NOT_READY: spin (implicit)
            }
        }
        exclusive_pfx[lid] = lk_running;

        // Publish PREFIX
        uint inclusive = lk_running + tile_hist[lid];
        uint packed = (FLAG_PREFIX << FLAG_SHIFT)
                    | (inclusive & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * EXP16_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);  // B5: PREFIX + exclusive_pfx ready

    // ── Phase 5: Per-SG atomic ranking + scatter ─────────────────
    // Reuse sg_hist_or_rank as sg_rank_ctr (non-overlapping lifetime
    // with Phase 2 sg_counts — sg_counts was last read in Phase 2b).
    // Zero ranking counters
    for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // B7: rank counters zeroed

    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        if (mv[e]) {
            uint d = md[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP16_NUM_BINS + d],
                1u, memory_order_relaxed);
            uint gp = global_hist[pass * EXP16_NUM_BINS + d]
                     + exclusive_pfx[d]
                     + sg_prefix[simd_id * EXP16_NUM_BINS + d]
                     + within_sg;
            dst[gp] = mk[e];
        }
    }
}
