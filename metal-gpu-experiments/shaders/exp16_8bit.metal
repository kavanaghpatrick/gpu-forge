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
// 8-bit radix, 4 passes, 4096 elements/tile, 256 threads/TG.
// ═══════════════════════════════════════════════════════════════════

#define EXP16_NUM_BINS   256u
#define EXP16_NUM_SGS    8u
#define EXP16_TILE_SIZE  4096u
#define EXP16_ELEMS      16u
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
// Diagnostic: Sequential copy (bandwidth ceiling)
// ═══════════════════════════════════════════════════════════════════
kernel void exp16_diag_copy(
    device const uint* src [[buffer(0)]],
    device uint*       dst [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    dst[tid] = src[tid];
}

// ═══════════════════════════════════════════════════════════════════
// Diagnostic: Random scatter (isolate scatter penalty)
// src[i] → dst[perm[i]], where perm is a random permutation
// ═══════════════════════════════════════════════════════════════════
kernel void exp16_diag_scatter(
    device const uint* src  [[buffer(0)]],
    device uint*       dst  [[buffer(1)]],
    device const uint* perm [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    dst[perm[tid]] = src[tid];
}

// ═══════════════════════════════════════════════════════════════════
// Diagnostic: Random gather (read from random positions, write sequentially)
// dst[i] = src[perm[i]]  — opposite of scatter
// ═══════════════════════════════════════════════════════════════════
kernel void exp16_diag_gather(
    device const uint* src  [[buffer(0)]],
    device uint*       dst  [[buffer(1)]],
    device const uint* perm [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    dst[tid] = src[perm[tid]];
}

// ═══════════════════════════════════════════════════════════════════
// Diagnostic: Blocked gather — read 32 consecutive elements (1 cache line)
// from random block positions. Simulates gather with spatial locality.
// Each thread reads from: random_block_start + (tid % 32)
// ═══════════════════════════════════════════════════════════════════
kernel void exp16_diag_gather_blocked(
    device const uint* src  [[buffer(0)]],
    device uint*       dst  [[buffer(1)]],
    device const uint* block_starts [[buffer(2)]],  // N/32 random offsets
    constant uint&     n    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    uint block_id = tid / 32u;
    uint lane = tid % 32u;
    uint base = block_starts[block_id];
    dst[tid] = src[base + lane];
}

// ═══════════════════════════════════════════════════════════════════
// Diagnostic: 256-bin structured scatter — simulate radix sort scatter
// where elements within a TG scatter to ~256 destination regions.
// Uses precomputed bin offsets + local rank to create a realistic
// radix sort scatter pattern (much less random than full permutation).
// ═══════════════════════════════════════════════════════════════════
kernel void exp16_diag_scatter_binned(
    device const uint* src     [[buffer(0)]],
    device uint*       dst     [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],  // per-element destination offset
    constant uint&     n       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    dst[offsets[tid]] = src[tid];
}

// ═══════════════════════════════════════════════════════════════════
// Diagnostic: Sequential merge of two sorted halves (merge bandwidth)
// Each TG merges a chunk — measures achievable merge throughput
// ═══════════════════════════════════════════════════════════════════
kernel void exp16_diag_merge_pair(
    device const uint* src  [[buffer(0)]],
    device uint*       dst  [[buffer(1)]],
    constant uint&     half_size [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    // Simple 2-way merge: each TG handles one pair of sorted runs
    uint pair_id = gid;
    uint left_start  = pair_id * 2u * half_size;
    uint right_start = left_start + half_size;
    uint dst_start   = left_start;
    uint left_end    = right_start;
    uint right_end   = right_start + half_size;

    // Thread 0 does the merge (sequential, measures bandwidth not parallelism)
    if (lid == 0u) {
        uint li = left_start, ri = right_start, di = dst_start;
        while (li < left_end && ri < right_end) {
            uint lv = src[li], rv = src[ri];
            if (lv <= rv) { dst[di++] = lv; li++; }
            else          { dst[di++] = rv; ri++; }
        }
        while (li < left_end) dst[di++] = src[li++];
        while (ri < right_end) dst[di++] = src[ri++];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Diagnostic: Bitonic sort of 4096 elements in TG memory
// Measures: can we fully sort a tile without any DRAM scatter?
// 256 threads × 16 elements each = 4096 elements
// ═══════════════════════════════════════════════════════════════════
kernel void exp16_diag_bitonic_tile(
    device const uint* src [[buffer(0)]],
    device uint*       dst [[buffer(1)]],
    constant uint&     element_count [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup uint tile[4096];

    uint base = gid * 4096u;

    // Load 16 elements per thread
    for (uint e = 0u; e < 16u; e++) {
        uint idx = base + lid * 16u + e;
        tile[lid * 16u + e] = (idx < element_count) ? src[idx] : 0xFFFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort network: log2(4096) = 12 stages
    for (uint k = 2u; k <= 4096u; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            // Each thread handles 16 compare-swaps per sub-pass
            for (uint e = 0u; e < 16u; e++) {
                uint i = lid * 16u + e;
                uint partner = i ^ j;
                if (partner > i && partner < 4096u) {
                    bool ascending = ((i & k) == 0u);
                    uint vi = tile[i], vp = tile[partner];
                    if ((ascending && vi > vp) || (!ascending && vi < vp)) {
                        tile[i] = vp;
                        tile[partner] = vi;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write sorted tile back (sequential — no scatter!)
    for (uint e = 0u; e < 16u; e++) {
        uint idx = base + lid * 16u + e;
        if (idx < element_count)
            dst[idx] = tile[lid * 16u + e];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Diagnostic: Partition WITHOUT scatter (load+hist+lookback only)
// Same as exp16_partition but writes nothing — measures non-scatter
// ═══════════════════════════════════════════════════════════════════
kernel void exp16_diag_noscat(
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

    threadgroup atomic_uint sg_hist_or_rank[EXP16_NUM_SGS * EXP16_NUM_BINS];
    threadgroup uint sg_prefix[EXP16_NUM_SGS * EXP16_NUM_BINS];
    threadgroup uint tile_hist[EXP16_NUM_BINS];
    threadgroup uint exclusive_pfx[EXP16_NUM_BINS];

    uint tile_id = gid;
    uint base = tile_id * EXP16_TILE_SIZE;

    // Phase 1: Load
    uint mk[EXP16_ELEMS];
    uint md[EXP16_ELEMS];
    bool mv[EXP16_ELEMS];
    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        uint idx = base + simd_id * (EXP16_ELEMS * 32u) + (uint)e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // Phase 2: Per-SG atomic histogram
    for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(&sg_hist_or_rank[simd_id * EXP16_NUM_BINS + md[e]], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2b: Tile histogram + cross-SG prefix
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP16_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(&sg_hist_or_rank[sg * EXP16_NUM_BINS + lid], memory_order_relaxed);
            sg_prefix[sg * EXP16_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Publish AGGREGATE
    {
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT) | (tile_hist[lid] & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * EXP16_NUM_BINS + lid], packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Decoupled lookback
    {
        uint lk_running = 0u;
        if (tile_id > 0u) {
            int look = (int)tile_id - 1;
            while (look >= 0) {
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                uint val = atomic_load_explicit(&tile_status[(uint)look * EXP16_NUM_BINS + lid], memory_order_relaxed);
                uint flag  = val >> FLAG_SHIFT;
                uint count = val & VALUE_MASK;
                if (flag == FLAG_PREFIX) { lk_running += count; break; }
                else if (flag == FLAG_AGGREGATE) { lk_running += count; look--; }
            }
        }
        exclusive_pfx[lid] = lk_running;
        uint inclusive = lk_running + tile_hist[lid];
        uint packed = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * EXP16_NUM_BINS + lid], packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 5: SKIP scatter — just sink registers to prevent dead code elimination
    uint sink = 0u;
    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        if (mv[e]) {
            sink += mk[e] + md[e] + exclusive_pfx[md[e]];
        }
    }
    // Write one value to prevent entire kernel from being optimized away
    if (lid == 0u && tile_id == 0u) {
        dst[0] = sink;
    }
}

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
        uint idx = base + simd_id * (EXP16_ELEMS * 32u) + (uint)e * 32u + simd_lane;
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

// ═══════════════════════════════════════════════════════════════════
// Kernel 4c: Partition v3 — Large tiles (8192 elem, 32/thread)
// Halves tile count vs 4096 → shallower lookback, less overhead.
// Same random scatter as v1 but fewer tiles = less lookback latency.
// TG memory: 18 KB (same as v1). Registers: 32 keys × 3 arrays = 96.
// ═══════════════════════════════════════════════════════════════════

#define V3_TILE_SIZE  8192u
#define V3_ELEMS      32u

kernel void exp16_partition_v3(
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

    threadgroup atomic_uint sg_hist_or_rank[EXP16_NUM_SGS * EXP16_NUM_BINS];
    threadgroup uint sg_prefix[EXP16_NUM_SGS * EXP16_NUM_BINS];
    threadgroup uint tile_hist[EXP16_NUM_BINS];
    threadgroup uint exclusive_pfx[EXP16_NUM_BINS];

    uint tile_id = gid;
    uint base = tile_id * V3_TILE_SIZE;

    // Phase 1: Load 32 elements per thread (SG-contiguous layout)
    uint mk[V3_ELEMS];
    uint md[V3_ELEMS];
    bool mv[V3_ELEMS];
    for (int e = 0; e < (int)V3_ELEMS; e++) {
        uint idx = base + simd_id * (V3_ELEMS * 32u) + (uint)e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // Phase 2: Per-SG atomic histogram
    for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V3_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP16_NUM_BINS + md[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2b: Tile histogram + cross-SG prefix
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP16_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * EXP16_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * EXP16_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Publish AGGREGATE
    {
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT)
                    | (tile_hist[lid] & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * EXP16_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Decoupled lookback
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
            }
        }
        exclusive_pfx[lid] = lk_running;

        uint inclusive = lk_running + tile_hist[lid];
        uint packed = (FLAG_PREFIX << FLAG_SHIFT)
                    | (inclusive & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * EXP16_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 5: Per-SG atomic ranking + scatter
    for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V3_ELEMS; e++) {
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

// ═══════════════════════════════════════════════════════════════════
// Kernel 4d: Partition v4 — Two-half TG reorder for coalesced scatter
//
// Keeps 4096-element tiles (16 elem/thread) but splits scatter into
// two phases through a 2048-element reorder buffer. Avoids the tile
// count increase that killed v2.
//
// Phase 5a: Rank all 4096 elements, store tile_local_rank + global_pos
// Phase 5b: Elements with rank < 2048 → tg_reorder, barrier, coalesced scatter
// Phase 5c: Elements with rank >= 2048 → tg_reorder, barrier, coalesced scatter
//
// TG memory (27 KB):
//   sg_hist_or_rank[8*256] = 8 KB  (P2 histogram, P5a rank)
//   sg_prefix[8*256]       = 8 KB  (cross-SG prefix P2b-P5a)
//   tile_hist[256]         = 1 KB
//   exclusive_pfx[256]     = 1 KB
//   tile_digit_pfx[256]    = 1 KB  (exclusive prefix of tile_hist)
//   tg_reorder[2048]       = 8 KB  (reorder buffer for coalesced scatter)
// ═══════════════════════════════════════════════════════════════════

kernel void exp16_partition_v4(
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

    threadgroup atomic_uint sg_hist_or_rank[EXP16_NUM_SGS * EXP16_NUM_BINS]; // 8 KB
    threadgroup uint sg_prefix[EXP16_NUM_SGS * EXP16_NUM_BINS];             // 8 KB
    threadgroup uint tile_hist[EXP16_NUM_BINS];                              // 1 KB
    threadgroup uint exclusive_pfx[EXP16_NUM_BINS];                          // 1 KB
    threadgroup uint tile_digit_pfx[EXP16_NUM_BINS];                         // 1 KB
    threadgroup uint tg_reorder[2048];                                       // 8 KB

    uint tile_id = gid;
    uint base = tile_id * EXP16_TILE_SIZE;
    uint tile_count = min(EXP16_TILE_SIZE, n - min(base, n));

    // ── Phase 1: Load (SG-contiguous) ──────────────────────────────
    uint mk[EXP16_ELEMS];
    uint md[EXP16_ELEMS];
    bool mv[EXP16_ELEMS];
    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        uint idx = base + simd_id * (EXP16_ELEMS * 32u) + (uint)e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG atomic histogram ───────────────────────────
    for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP16_NUM_BINS + md[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Tile histogram + cross-SG prefix ─────────────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP16_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * EXP16_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * EXP16_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2c: tile_digit_pfx = exclusive prefix of tile_hist ───
    if (simd_id == 0u) {
        uint running = 0u;
        for (uint chunk = 0u; chunk < 8u; chunk++) {
            uint bin = chunk * 32u + simd_lane;
            uint val = tile_hist[bin];
            uint prefix = simd_prefix_exclusive_sum(val) + running;
            tile_digit_pfx[bin] = prefix;
            running += simd_shuffle(simd_prefix_exclusive_sum(val) + val, 31u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Publish AGGREGATE ─────────────────────────────────
    {
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT)
                    | (tile_hist[lid] & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * EXP16_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Decoupled lookback ────────────────────────────────
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
            }
        }
        exclusive_pfx[lid] = lk_running;

        uint inclusive = lk_running + tile_hist[lid];
        uint packed = (FLAG_PREFIX << FLAG_SHIFT)
                    | (inclusive & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * EXP16_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5a: Rank all elements, store tile_local_rank ─────────
    for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint m_rank[EXP16_ELEMS];
    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        if (mv[e]) {
            uint d = md[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP16_NUM_BINS + d],
                1u, memory_order_relaxed);
            m_rank[e] = tile_digit_pfx[d]
                      + sg_prefix[simd_id * EXP16_NUM_BINS + d]
                      + within_sg;
        } else {
            m_rank[e] = 0xFFFFFFFFu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5b: First half — ranks [0, 2048) ────────────────────
    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        if (mv[e] && m_rank[e] < 2048u) {
            tg_reorder[m_rank[e]] = mk[e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Coalesced scatter: sequential read from reorder buffer
    {
        uint half_count = min(2048u, tile_count);
        for (int e = 0; e < 8; e++) {
            uint read_idx = (uint)e * EXP16_THREADS + lid;
            if (read_idx < half_count) {
                uint key = tg_reorder[read_idx];
                uint d = (key >> shift) & 0xFFu;
                uint within_tile = read_idx - tile_digit_pfx[d];
                uint gp = global_hist[pass * EXP16_NUM_BINS + d]
                         + exclusive_pfx[d]
                         + within_tile;
                dst[gp] = key;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5c: Second half — ranks [2048, 4096) ────────────────
    for (int e = 0; e < (int)EXP16_ELEMS; e++) {
        if (mv[e] && m_rank[e] >= 2048u && m_rank[e] < EXP16_TILE_SIZE) {
            tg_reorder[m_rank[e] - 2048u] = mk[e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Coalesced scatter for second half
    {
        uint second_start = min(2048u, tile_count);
        uint second_count = tile_count > 2048u ? tile_count - 2048u : 0u;
        for (int e = 0; e < 8; e++) {
            uint read_idx = (uint)e * EXP16_THREADS + lid;
            if (read_idx < second_count) {
                uint key = tg_reorder[read_idx];
                uint d = (key >> shift) & 0xFFu;
                uint within_tile = (read_idx + 2048u) - tile_digit_pfx[d];
                uint gp = global_hist[pass * EXP16_NUM_BINS + d]
                         + exclusive_pfx[d]
                         + within_tile;
                dst[gp] = key;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 11-BIT 3-PASS RADIX SORT (2048 bins, 3 passes = 11+11+10 = 32 bits)
//
// Reduces passes from 4→3 by using 11-bit radix width (2048 bins).
// Key differences from 8-bit:
//   - TG-wide atomic histogram (2048 bins too many for per-SG approach)
//   - Per-SG sequential ranking for sort stability
//   - Each thread handles 8 bins in lookback (2048/256)
//   - 16 KB TG memory (vs 18 KB for 8-bit)
//
// Same decoupled lookback + device-scope fence architecture.
// ═══════════════════════════════════════════════════════════════════

#define V5_MAX_BINS    2048u
#define V5_TILE_SIZE   4096u
#define V5_ELEMS       16u
#define V5_THREADS     256u
#define V5_NUM_SGS     8u
#define V5_BINS_PER_THREAD (V5_MAX_BINS / V5_THREADS)  // 8

struct V5Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint mask;       // 0x7FF for 11-bit, 0x3FF for 10-bit
    uint pass;
};

// ───────────────────────────────────────────────────────────────────
// Kernel: 3-pass combined histogram
// Reads data once, counts all 3 passes simultaneously.
// TG-wide atomic histogram: 3 × 2048 × 4B = 24 KB
// ───────────────────────────────────────────────────────────────────
kernel void exp16_3pass_histogram(
    device const uint*    src            [[buffer(0)]],
    device atomic_uint*   global_hist    [[buffer(1)]],
    constant uint&        element_count  [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint local_hist[3u * V5_MAX_BINS]; // 24 KB

    // Zero local histogram (6144 entries / 256 threads = 24 per thread)
    for (uint i = lid; i < 3u * V5_MAX_BINS; i += V5_THREADS)
        atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load + count all 3 passes
    uint base = gid * V5_TILE_SIZE;
    for (uint e = 0u; e < V5_ELEMS; e++) {
        uint idx = base + e * V5_THREADS + lid;
        if (idx < element_count) {
            uint key = src[idx];
            uint d0 = key & 0x7FFu;
            uint d1 = (key >> 11u) & 0x7FFu;
            uint d2 = (key >> 22u) & 0x3FFu;
            atomic_fetch_add_explicit(&local_hist[d0], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&local_hist[V5_MAX_BINS + d1], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&local_hist[2u * V5_MAX_BINS + d2], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to global
    for (uint i = lid; i < 3u * V5_MAX_BINS; i += V5_THREADS) {
        uint count = atomic_load_explicit(&local_hist[i], memory_order_relaxed);
        if (count > 0u)
            atomic_fetch_add_explicit(&global_hist[i], count, memory_order_relaxed);
    }
}

// ───────────────────────────────────────────────────────────────────
// Kernel: 3-pass global prefix sum
// 3 passes × 2048 bins, using 3 SGs with 64 chunks of 32 each.
// ───────────────────────────────────────────────────────────────────
kernel void exp16_3pass_prefix(
    device uint* global_hist [[buffer(0)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    if (simd_id < 3u) {
        uint p = simd_id;
        uint running = 0u;
        for (uint chunk = 0u; chunk < 64u; chunk++) {
            uint bin = chunk * 32u + simd_lane;
            uint val = global_hist[p * V5_MAX_BINS + bin];
            uint prefix = simd_prefix_exclusive_sum(val) + running;
            global_hist[p * V5_MAX_BINS + bin] = prefix;
            running += simd_shuffle(simd_prefix_exclusive_sum(val) + val, 31u);
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Kernel: 3-pass zero status buffer
// ───────────────────────────────────────────────────────────────────
kernel void exp16_3pass_zero(
    device uint*     status        [[buffer(0)]],
    constant uint&   total_entries [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < total_entries)
        status[tid] = 0u;
}

// ───────────────────────────────────────────────────────────────────
// Kernel: 3-pass partition
// TG-wide atomic histogram + per-SG sequential ranking for stability.
// 2048-bin lookback: each of 256 threads handles 8 bins.
// TG memory: tg_hist[2048] + tg_prefix[2048] = 16 KB
// ───────────────────────────────────────────────────────────────────
kernel void exp16_3pass_partition(
    device const uint*     src         [[buffer(0)]],
    device uint*           dst         [[buffer(1)]],
    device atomic_uint*    tile_status [[buffer(2)]],
    device const uint*     global_pfx  [[buffer(3)]],
    constant V5Params&     params      [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n     = params.element_count;
    uint shift = params.shift;
    uint mask  = params.mask;
    uint pass  = params.pass;

    threadgroup atomic_uint tg_hist[V5_MAX_BINS];  // 8 KB
    threadgroup uint tg_prefix[V5_MAX_BINS];        // 8 KB

    uint tile_id = gid;
    uint base = tile_id * V5_TILE_SIZE;

    // ── Phase 1: Load elements (SG-contiguous for stability) ──────
    uint mk[V5_ELEMS];
    uint md[V5_ELEMS];
    bool mv[V5_ELEMS];
    for (uint e = 0u; e < V5_ELEMS; e++) {
        uint idx = base + simd_id * (V5_ELEMS * 32u) + e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = (mk[e] >> shift) & mask;
    }

    // ── Phase 2: TG-wide atomic histogram (counting only) ─────────
    for (uint i = lid; i < V5_MAX_BINS; i += V5_THREADS)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < V5_ELEMS; e++) {
        if (mv[e])
            atomic_fetch_add_explicit(&tg_hist[md[e]], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Publish AGGREGATE (8 bins per thread) ────────────
    for (uint b = 0u; b < V5_BINS_PER_THREAD; b++) {
        uint bin = lid * V5_BINS_PER_THREAD + b;
        uint count = atomic_load_explicit(&tg_hist[bin], memory_order_relaxed);
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT) | (count & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * V5_MAX_BINS + bin],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Decoupled lookback (8 bins per thread) ───────────
    // NOTE: PREFIX values contain ONLY tile counts (no global prefix).
    // This avoids a race where tile N reads tile 0's AGGREGATE (Phase 3)
    // instead of PREFIX (Phase 4) — AGGREGATE has just count, PREFIX had
    // gp+count, causing lost global offsets. With tile-counts-only,
    // AGGREGATE and PREFIX values are compatible. Global prefix is added
    // separately during scatter.
    {
        uint lk_running = 0u;
        if (tile_id > 0u) {
            for (uint b = 0u; b < V5_BINS_PER_THREAD; b++) {
                uint bin = lid * V5_BINS_PER_THREAD + b;
                uint running = 0u;

                for (int pred = (int)tile_id - 1; pred >= 0; pred--) {
                    uint val;
                    while (true) {
                        atomic_thread_fence(mem_flags::mem_device,
                                            memory_order_seq_cst,
                                            thread_scope_device);
                        val = atomic_load_explicit(
                            &tile_status[(uint)pred * V5_MAX_BINS + bin],
                            memory_order_relaxed);
                        if ((val >> FLAG_SHIFT) != FLAG_NOT_READY) break;
                    }
                    uint flag = val >> FLAG_SHIFT;
                    running += val & VALUE_MASK;
                    if (flag == FLAG_PREFIX) break;
                }

                tg_prefix[bin] = running;

                // Publish our PREFIX (exclusive prefix + local count)
                uint count = atomic_load_explicit(&tg_hist[bin], memory_order_relaxed);
                uint packed = (FLAG_PREFIX << FLAG_SHIFT)
                            | ((running + count) & VALUE_MASK);
                atomic_store_explicit(&tile_status[tile_id * V5_MAX_BINS + bin],
                                      packed, memory_order_relaxed);
            }
        } else {
            // tile_id == 0: no predecessors, exclusive prefix = 0
            for (uint b = 0u; b < V5_BINS_PER_THREAD; b++) {
                uint bin = lid * V5_BINS_PER_THREAD + b;
                tg_prefix[bin] = 0u;
                uint count = atomic_load_explicit(&tg_hist[bin], memory_order_relaxed);
                uint packed = (FLAG_PREFIX << FLAG_SHIFT) | (count & VALUE_MASK);
                atomic_store_explicit(&tile_status[bin], packed, memory_order_relaxed);
            }
        }
        atomic_thread_fence(mem_flags::mem_device,
                            memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Per-SG sequential ranking + scatter ──────────────
    // Reset histogram counters for ranking
    for (uint i = lid; i < V5_MAX_BINS; i += V5_THREADS)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process SGs in order for sort stability:
    // SG 0 ranks first (lowest input positions → lowest ranks),
    // then SG 1, etc. Within each SG, lane-ordered atomics are stable.
    uint ranks[V5_ELEMS];
    for (uint sg = 0u; sg < V5_NUM_SGS; sg++) {
        if (simd_id == sg) {
            for (uint e = 0u; e < V5_ELEMS; e++) {
                if (mv[e]) {
                    ranks[e] = atomic_fetch_add_explicit(
                        &tg_hist[md[e]], 1u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scatter with computed ranks (global prefix added separately)
    for (uint e = 0u; e < V5_ELEMS; e++) {
        if (mv[e]) {
            uint dst_idx = global_pfx[pass * V5_MAX_BINS + md[e]]
                         + tg_prefix[md[e]] + ranks[e];
            dst[dst_idx] = mk[e];
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Kernel: 3-pass partition v2 — PARALLEL ranking (unstable)
//
// Key change from v1: removes serial SG ranking loop.
// v1 serialized 8 SGs for sort stability (only 12.5% utilization).
// v2 ranks all threads in parallel — unstable but correct for uint32.
// Also: processes lookback bins in interleaved order (thread T handles
// bins T, T+256, T+512, ...) for better load distribution.
// TG memory: tg_hist[2048] + tg_prefix[2048] = 16 KB (unchanged)
// ───────────────────────────────────────────────────────────────────
kernel void exp16_3pass_partition_v2(
    device const uint*     src         [[buffer(0)]],
    device uint*           dst         [[buffer(1)]],
    device atomic_uint*    tile_status [[buffer(2)]],
    device const uint*     global_pfx  [[buffer(3)]],
    constant V5Params&     params      [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n     = params.element_count;
    uint shift = params.shift;
    uint mask  = params.mask;
    uint pass  = params.pass;

    threadgroup atomic_uint tg_hist[V5_MAX_BINS];  // 8 KB
    threadgroup uint tg_prefix[V5_MAX_BINS];        // 8 KB

    uint tile_id = gid;
    uint base = tile_id * V5_TILE_SIZE;

    // ── Phase 1: Load elements ───────────────────────────────────────
    uint mk[V5_ELEMS];
    uint md[V5_ELEMS];
    bool mv[V5_ELEMS];
    for (uint e = 0u; e < V5_ELEMS; e++) {
        uint idx = base + simd_id * (V5_ELEMS * 32u) + e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = (mk[e] >> shift) & mask;
    }

    // ── Phase 2: TG-wide atomic histogram ────────────────────────────
    for (uint i = lid; i < V5_MAX_BINS; i += V5_THREADS)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < V5_ELEMS; e++) {
        if (mv[e])
            atomic_fetch_add_explicit(&tg_hist[md[e]], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Publish AGGREGATE (interleaved bin assignment) ──────
    // Thread T handles bins T, T+256, T+512, ... (stride=256)
    // Better spatial distribution than contiguous blocks
    for (uint b = 0u; b < V5_BINS_PER_THREAD; b++) {
        uint bin = lid + b * V5_THREADS;
        uint count = atomic_load_explicit(&tg_hist[bin], memory_order_relaxed);
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT) | (count & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * V5_MAX_BINS + bin],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Decoupled lookback (interleaved bins) ───────────────
    {
        if (tile_id > 0u) {
            for (uint b = 0u; b < V5_BINS_PER_THREAD; b++) {
                uint bin = lid + b * V5_THREADS;
                uint running = 0u;

                for (int pred = (int)tile_id - 1; pred >= 0; pred--) {
                    uint val;
                    while (true) {
                        atomic_thread_fence(mem_flags::mem_device,
                                            memory_order_seq_cst,
                                            thread_scope_device);
                        val = atomic_load_explicit(
                            &tile_status[(uint)pred * V5_MAX_BINS + bin],
                            memory_order_relaxed);
                        if ((val >> FLAG_SHIFT) != FLAG_NOT_READY) break;
                    }
                    uint flag = val >> FLAG_SHIFT;
                    running += val & VALUE_MASK;
                    if (flag == FLAG_PREFIX) break;
                }

                tg_prefix[bin] = running;

                uint count = atomic_load_explicit(&tg_hist[bin], memory_order_relaxed);
                uint packed = (FLAG_PREFIX << FLAG_SHIFT)
                            | ((running + count) & VALUE_MASK);
                atomic_store_explicit(&tile_status[tile_id * V5_MAX_BINS + bin],
                                      packed, memory_order_relaxed);
            }
        } else {
            for (uint b = 0u; b < V5_BINS_PER_THREAD; b++) {
                uint bin = lid + b * V5_THREADS;
                tg_prefix[bin] = 0u;
                uint count = atomic_load_explicit(&tg_hist[bin], memory_order_relaxed);
                uint packed = (FLAG_PREFIX << FLAG_SHIFT) | (count & VALUE_MASK);
                atomic_store_explicit(&tile_status[bin], packed, memory_order_relaxed);
            }
        }
        atomic_thread_fence(mem_flags::mem_device,
                            memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: PARALLEL ranking + scatter (unstable, all threads) ──
    // No serial SG loop — all 256 threads rank simultaneously.
    // Correct for uint32 sort (stability not needed for plain keys).
    for (uint i = lid; i < V5_MAX_BINS; i += V5_THREADS)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < V5_ELEMS; e++) {
        if (mv[e]) {
            uint d = md[e];
            uint rank = atomic_fetch_add_explicit(
                &tg_hist[d], 1u, memory_order_relaxed);
            uint dst_idx = global_pfx[pass * V5_MAX_BINS + d]
                         + tg_prefix[d] + rank;
            dst[dst_idx] = mk[e];
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Kernel: 3-pass partition v3 — LARGE TILES (8192 elements)
//
// Key insight: 2048-bin tile_status at 4096 tiles = 32 MB (DRAM).
// With 8192-element tiles: 1953 tiles × 2048 × 4B = 16 MB (SLC!)
// SLC lookback is ~3x faster than DRAM lookback.
// 32 elements/thread, serial SG ranking preserved for stability.
// TG memory: tg_hist[2048] + tg_prefix[2048] = 16 KB (unchanged)
// ───────────────────────────────────────────────────────────────────

#define V3P_TILE_SIZE  8192u
#define V3P_ELEMS      32u

kernel void exp16_3pass_partition_v3(
    device const uint*     src         [[buffer(0)]],
    device uint*           dst         [[buffer(1)]],
    device atomic_uint*    tile_status [[buffer(2)]],
    device const uint*     global_pfx  [[buffer(3)]],
    constant V5Params&     params      [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n     = params.element_count;
    uint shift = params.shift;
    uint mask  = params.mask;
    uint pass  = params.pass;

    threadgroup atomic_uint tg_hist[V5_MAX_BINS];  // 8 KB
    threadgroup uint tg_prefix[V5_MAX_BINS];        // 8 KB

    uint tile_id = gid;
    uint base = tile_id * V3P_TILE_SIZE;

    // ── Phase 1: Load 32 elements per thread (SG-contiguous) ─────────
    uint mk[V3P_ELEMS];
    uint md[V3P_ELEMS];
    bool mv[V3P_ELEMS];
    for (uint e = 0u; e < V3P_ELEMS; e++) {
        uint idx = base + simd_id * (V3P_ELEMS * 32u) + e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = (mk[e] >> shift) & mask;
    }

    // ── Phase 2: TG-wide atomic histogram ────────────────────────────
    for (uint i = lid; i < V5_MAX_BINS; i += V5_THREADS)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < V3P_ELEMS; e++) {
        if (mv[e])
            atomic_fetch_add_explicit(&tg_hist[md[e]], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Publish AGGREGATE (interleaved bins) ────────────────
    for (uint b = 0u; b < V5_BINS_PER_THREAD; b++) {
        uint bin = lid + b * V5_THREADS;
        uint count = atomic_load_explicit(&tg_hist[bin], memory_order_relaxed);
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT) | (count & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * V5_MAX_BINS + bin],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Decoupled lookback (interleaved bins) ───────────────
    {
        if (tile_id > 0u) {
            for (uint b = 0u; b < V5_BINS_PER_THREAD; b++) {
                uint bin = lid + b * V5_THREADS;
                uint running = 0u;

                for (int pred = (int)tile_id - 1; pred >= 0; pred--) {
                    uint val;
                    while (true) {
                        atomic_thread_fence(mem_flags::mem_device,
                                            memory_order_seq_cst,
                                            thread_scope_device);
                        val = atomic_load_explicit(
                            &tile_status[(uint)pred * V5_MAX_BINS + bin],
                            memory_order_relaxed);
                        if ((val >> FLAG_SHIFT) != FLAG_NOT_READY) break;
                    }
                    uint flag = val >> FLAG_SHIFT;
                    running += val & VALUE_MASK;
                    if (flag == FLAG_PREFIX) break;
                }

                tg_prefix[bin] = running;

                uint count = atomic_load_explicit(&tg_hist[bin], memory_order_relaxed);
                uint packed = (FLAG_PREFIX << FLAG_SHIFT)
                            | ((running + count) & VALUE_MASK);
                atomic_store_explicit(&tile_status[tile_id * V5_MAX_BINS + bin],
                                      packed, memory_order_relaxed);
            }
        } else {
            for (uint b = 0u; b < V5_BINS_PER_THREAD; b++) {
                uint bin = lid + b * V5_THREADS;
                tg_prefix[bin] = 0u;
                uint count = atomic_load_explicit(&tg_hist[bin], memory_order_relaxed);
                uint packed = (FLAG_PREFIX << FLAG_SHIFT) | (count & VALUE_MASK);
                atomic_store_explicit(&tile_status[bin], packed, memory_order_relaxed);
            }
        }
        atomic_thread_fence(mem_flags::mem_device,
                            memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Serial SG ranking + scatter (stable) ────────────────
    for (uint i = lid; i < V5_MAX_BINS; i += V5_THREADS)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint ranks[V3P_ELEMS];
    for (uint sg = 0u; sg < V5_NUM_SGS; sg++) {
        if (simd_id == sg) {
            for (uint e = 0u; e < V3P_ELEMS; e++) {
                if (mv[e]) {
                    ranks[e] = atomic_fetch_add_explicit(
                        &tg_hist[md[e]], 1u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint e = 0u; e < V3P_ELEMS; e++) {
        if (mv[e]) {
            uint dst_idx = global_pfx[pass * V5_MAX_BINS + md[e]]
                         + tg_prefix[md[e]] + ranks[e];
            dst[dst_idx] = mk[e];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4b: Partition v2 — TG memory reorder for coalesced scatter
//
// Same as exp16_partition but adds a reorder step in Phase 5:
// 1. Rank elements and write to TG reorder buffer at tile-local rank
// 2. Barrier
// 3. Read sequentially from reorder buffer → coalesced global scatter
//
// Uses 2048-element tiles (8 elem/thread) to fit 8KB reorder buffer
// within 32KB TG memory limit (27KB total).
//
// TG memory layout (27 KB):
//   sg_hist_or_rank[8*256] = 8 KB  (atomic histogram P2, rank P5a)
//   sg_prefix[8*256]       = 8 KB  (cross-SG prefix P2b-P5a)
//   tile_hist[256]         = 1 KB  (tile totals P2b-P3)
//   exclusive_pfx[256]     = 1 KB  (lookback result P4-P5c)
//   tile_digit_pfx[256]    = 1 KB  (exclusive prefix of tile_hist, P5a-P5c)
//   tg_reorder[2048]       = 8 KB  (digit-sorted keys for coalesced scatter)
// ═══════════════════════════════════════════════════════════════════

#define V2_TILE_SIZE  2048u
#define V2_ELEMS      8u

kernel void exp16_partition_v2(
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

    // ── TG Memory (27 KB) ──────────────────────────────────────────
    threadgroup atomic_uint sg_hist_or_rank[EXP16_NUM_SGS * EXP16_NUM_BINS]; // 8 KB
    threadgroup uint sg_prefix[EXP16_NUM_SGS * EXP16_NUM_BINS];             // 8 KB
    threadgroup uint tile_hist[EXP16_NUM_BINS];                              // 1 KB
    threadgroup uint exclusive_pfx[EXP16_NUM_BINS];                          // 1 KB
    threadgroup uint tile_digit_pfx[EXP16_NUM_BINS];                         // 1 KB
    threadgroup uint tg_reorder[V2_TILE_SIZE];                               // 8 KB

    uint tile_id = gid;
    uint base = tile_id * V2_TILE_SIZE;
    uint tile_count = min(V2_TILE_SIZE, n - min(base, n));

    // ── Phase 1: Load (SG-contiguous layout) ───────────────────────
    uint mk[V2_ELEMS];
    uint md[V2_ELEMS];
    bool mv[V2_ELEMS];
    for (int e = 0; e < (int)V2_ELEMS; e++) {
        uint idx = base + simd_id * (V2_ELEMS * 32u) + (uint)e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG atomic histogram ───────────────────────────
    for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V2_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP16_NUM_BINS + md[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Tile histogram + cross-SG prefix ─────────────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP16_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * EXP16_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * EXP16_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2c: Compute tile_digit_pfx (exclusive prefix of tile_hist) ──
    // Use SIMD prefix sum: 256 bins = 8 chunks of 32
    {
        // Only SG 0 computes this (single-SG sequential prefix)
        if (simd_id == 0u) {
            uint running = 0u;
            for (uint chunk = 0u; chunk < 8u; chunk++) {
                uint bin = chunk * 32u + simd_lane;
                uint val = tile_hist[bin];
                uint prefix = simd_prefix_exclusive_sum(val) + running;
                tile_digit_pfx[bin] = prefix;
                running += simd_shuffle(simd_prefix_exclusive_sum(val) + val, 31u);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Publish AGGREGATE ─────────────────────────────────
    {
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT)
                    | (tile_hist[lid] & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * EXP16_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Decoupled lookback (ALL 256 threads, one per bin) ─
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
            }
        }
        exclusive_pfx[lid] = lk_running;

        uint inclusive = lk_running + tile_hist[lid];
        uint packed = (FLAG_PREFIX << FLAG_SHIFT)
                    | (inclusive & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * EXP16_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5a: Rank + write to TG reorder buffer ────────────────
    // Zero ranking counters
    for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V2_ELEMS; e++) {
        if (mv[e]) {
            uint d = md[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP16_NUM_BINS + d],
                1u, memory_order_relaxed);
            uint tile_local_rank = tile_digit_pfx[d]
                                 + sg_prefix[simd_id * EXP16_NUM_BINS + d]
                                 + within_sg;
            tg_reorder[tile_local_rank] = mk[e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5b: Coalesced scatter from TG reorder buffer ─────────
    // Read sequentially: threads read consecutive elements from reorder
    // buffer. Since data is sorted by digit, adjacent reads are same
    // digit → adjacent writes to same global region → coalesced.
    for (int e = 0; e < (int)V2_ELEMS; e++) {
        uint read_idx = (uint)e * EXP16_THREADS + lid;
        if (read_idx < tile_count) {
            uint key = tg_reorder[read_idx];
            uint d = (key >> shift) & 0xFFu;
            uint within_tile = read_idx - tile_digit_pfx[d];
            uint gp = global_hist[pass * EXP16_NUM_BINS + d]
                     + exclusive_pfx[d]
                     + within_tile;
            dst[gp] = key;
        }
    }
}
