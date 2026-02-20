#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 25: Fence-Free Radix Sort
//
// Eliminates ALL device-scope fences from the radix sort partition.
// Instead of decoupled lookback (which needs seq_cst device fences),
// uses precomputed tile prefix sums from a separate dispatch.
//
// Per pass (4 dispatches):
//   1. Tile histogram: read current source → per-tile 256-bin counts
//   2. Tile prefix: serial scan across tiles per bin → tile offsets
//   3. Global prefix: convert global counts to exclusive prefix
//   4. Scatter: re-read source, rank within tile, use precomputed offsets
//
// The scatter kernel is identical to exp16_partition but with:
//   - Phase 3 (AGGREGATE publish): REMOVED
//   - Phase 4 (lookback): REPLACED by single tile_prefix read
//   - ALL atomic_thread_fence: REMOVED
// ═══════════════════════════════════════════════════════════════════

#define E25_NUM_BINS   256u
#define E25_NUM_SGS    8u
#define E25_TILE_SIZE  4096u
#define E25_ELEMS      16u
#define E25_THREADS    256u

struct E25Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint pass;
};

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Single-Pass Tile Histogram
// Reads keys from current source buffer, computes per-tile 256-bin histogram.
// Called once per pass with the current source buffer.
// Output: tile_hists[tile_id * 256 + bin]
// ═══════════════════════════════════════════════════════════════════

kernel void exp25_tile_histogram(
    device const uint*     src        [[buffer(0)]],
    device uint*           tile_hists [[buffer(1)]],
    constant E25Params&    params     [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n = params.element_count;
    uint shift = params.shift;
    uint tile_id = gid;
    uint base = tile_id * E25_TILE_SIZE;

    // Load keys (SG-contiguous layout, same as exp16)
    uint keys[E25_ELEMS];
    bool valid[E25_ELEMS];
    for (int e = 0; e < (int)E25_ELEMS; e++) {
        uint idx = base + simd_id * (E25_ELEMS * 32u) + (uint)e * 32u + (lid & 31u);
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    // Per-SG atomic histogram in TG memory
    threadgroup atomic_uint sg_counts[E25_NUM_SGS * E25_NUM_BINS]; // 8 KB

    // Zero
    for (uint i = lid; i < E25_NUM_SGS * E25_NUM_BINS; i += E25_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-SG atomic count
    for (int e = 0; e < (int)E25_ELEMS; e++) {
        if (valid[e]) {
            uint digit = (keys[e] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * E25_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SGs
    uint total = 0u;
    for (uint sg = 0u; sg < E25_NUM_SGS; sg++) {
        total += atomic_load_explicit(
            &sg_counts[sg * E25_NUM_BINS + lid],
            memory_order_relaxed);
    }

    // Write: tile_hists[tile_id * 256 + bin]
    tile_hists[tile_id * E25_NUM_BINS + lid] = total;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Tile Prefix Sum + Global Histogram
// Serial exclusive prefix sum across tiles for each bin.
// Dispatch: 256 TGs (one per bin), thread 0 does the work.
// Output:
//   tile_prefix[tile_id * 256 + bin] = sum of tile_hists for tiles < tile_id
//   global_hist[bin] = total across all tiles
// ═══════════════════════════════════════════════════════════════════

kernel void exp25_tile_prefix(
    device const uint*    tile_hists   [[buffer(0)]],
    device uint*          tile_prefix  [[buffer(1)]],
    device uint*          global_hist  [[buffer(2)]],
    constant E25Params&   params       [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint nt = params.num_tiles;
    uint bin = gid;  // one TG per bin

    if (bin >= E25_NUM_BINS) return;

    // Thread 0 does serial prefix (other threads idle)
    // 256 TGs run in parallel → all 256 bins processed simultaneously
    if (lid == 0u) {
        uint running = 0u;
        for (uint t = 0u; t < nt; t++) {
            uint count = tile_hists[t * E25_NUM_BINS + bin];
            tile_prefix[t * E25_NUM_BINS + bin] = running;
            running += count;
        }
        global_hist[bin] = running;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Global Prefix Sum (256 bins → exclusive prefix)
// Dispatch: 1 TG of 256 threads.
// ═══════════════════════════════════════════════════════════════════

kernel void exp25_global_prefix(
    device uint*  global_hist  [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    // 256-bin exclusive prefix sum: 8 chunks of 32 via simd_prefix
    threadgroup uint chunk_totals[8];

    uint chunk = lid / 32u;
    uint val = global_hist[lid];
    uint prefix = simd_prefix_exclusive_sum(val);

    if (simd_lane == 31u) {
        chunk_totals[chunk] = prefix + val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        uint running = 0u;
        for (uint c = 0u; c < 8u; c++) {
            uint ct = chunk_totals[c];
            chunk_totals[c] = running;
            running += ct;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    global_hist[lid] = prefix + chunk_totals[chunk];
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: Fence-Free Scatter
//
// Re-reads keys, recomputes within-tile histogram + rank, then scatters
// using precomputed tile_prefix (no lookback, no device fences).
//
// Identical to exp16_partition phases 1, 2, 2b, 5 — with phases 3-4
// replaced by a simple tile_prefix read.
// ═══════════════════════════════════════════════════════════════════

kernel void exp25_scatter(
    device const uint*     src          [[buffer(0)]],
    device uint*           dst          [[buffer(1)]],
    device const uint*     tile_prefix  [[buffer(2)]],
    device const uint*     global_hist  [[buffer(3)]],
    constant E25Params&    params       [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n     = params.element_count;
    uint shift = params.shift;
    uint tile_id = gid;
    uint base = tile_id * E25_TILE_SIZE;

    // ── TG Memory (18 KB — same as exp16) ─────────────────────────
    threadgroup atomic_uint sg_hist_or_rank[E25_NUM_SGS * E25_NUM_BINS]; // 8 KB
    threadgroup uint sg_prefix[E25_NUM_SGS * E25_NUM_BINS];             // 8 KB
    threadgroup uint exclusive_pfx[E25_NUM_BINS];                        // 1 KB

    // ── Phase 1: Load (same layout as exp16) ──────────────────────
    uint mk[E25_ELEMS];
    uint md[E25_ELEMS];
    bool mv[E25_ELEMS];
    for (int e = 0; e < (int)E25_ELEMS; e++) {
        uint idx = base + simd_id * (E25_ELEMS * 32u) + (uint)e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG atomic histogram ──────────────────────────
    for (uint i = lid; i < E25_NUM_SGS * E25_NUM_BINS; i += E25_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)E25_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * E25_NUM_BINS + md[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Cross-SG prefix ────────────────────────────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < E25_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * E25_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * E25_NUM_BINS + lid] = total;
            total += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Read precomputed tile prefix (NO LOOKBACK!) ──────
    exclusive_pfx[lid] = tile_prefix[tile_id * E25_NUM_BINS + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Per-SG atomic ranking + scatter ──────────────────
    for (uint i = lid; i < E25_NUM_SGS * E25_NUM_BINS; i += E25_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)E25_ELEMS; e++) {
        if (mv[e]) {
            uint d = md[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * E25_NUM_BINS + d],
                1u, memory_order_relaxed);
            uint gp = global_hist[d]
                     + exclusive_pfx[d]
                     + sg_prefix[simd_id * E25_NUM_BINS + d]
                     + within_sg;
            dst[gp] = mk[e];
        }
    }
}
