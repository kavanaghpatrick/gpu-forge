#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 17: MSD+LSD Hybrid Radix Sort (5000+ Mkeys/s target)
//
// Architecture: 1 MSD scatter (bits 24:31) → 256 SLC-resident buckets
// → 3 inner LSD passes per bucket at SLC speed.
// Single encoder, 14 dispatches, zero CPU readback.
// ═══════════════════════════════════════════════════════════════════

#define EXP17_NUM_BINS  256u
#define EXP17_TILE_SIZE 4096u
#define EXP17_ELEMS     16u
#define EXP17_THREADS   256u
#define EXP17_NUM_SGS   8u
#define EXP17_MAX_TPB   17u

// V2 tile size: 8192 elements/tile (32 per thread)
#define EXP17_TILE_SIZE_LARGE 8192u
#define EXP17_ELEMS_LARGE     32u
#define EXP17_MAX_TPB_V2      9u

struct Exp17Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint pass;
};

struct Exp17InnerParams {
    uint shift;
};

struct BucketDesc {
    uint offset;
    uint count;
    uint tile_count;
    uint tile_base;
};

// ═══════════════════════════════════════════════════════════════════
// Placeholder kernel — trivial copy so the file compiles
// ═══════════════════════════════════════════════════════════════════
kernel void exp17_placeholder(
    device const uint* src [[buffer(0)]],
    device uint*       dst [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    dst[tid] = src[tid];
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: MSD Histogram — single-pass, bits[24:31]
//
// Cloned from exp16_combined_histogram but removes the 4-pass loop.
// Reads ALL data once, computes 256-bin histogram for MSD byte only.
// Uses per-SG atomic histogram on TG memory.
// Output: global_hist[digit] = total count for digit 0..255
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_msd_histogram(
    device const uint*     src          [[buffer(0)]],
    device atomic_uint*    global_hist  [[buffer(1)]],
    constant Exp17Params&  params       [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint n = params.element_count;
    uint shift = params.shift;
    uint base = gid * EXP17_TILE_SIZE;

    // Load 16 elements into registers (one global memory read)
    uint keys[EXP17_ELEMS];
    bool valid[EXP17_ELEMS];
    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        uint idx = base + e * EXP17_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    // Per-SG accumulator in shared memory
    threadgroup atomic_uint sg_counts[EXP17_NUM_SGS * EXP17_NUM_BINS]; // 8 KB

    // Zero sg_counts (all 256 threads cooperate)
    for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-SG atomic histogram (single pass: bits[24:31])
    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        if (valid[e]) {
            uint digit = (keys[e] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * EXP17_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SGs: 256 threads handle one bin each
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP17_NUM_SGS; sg++) {
            total += atomic_load_explicit(
                &sg_counts[sg * EXP17_NUM_BINS + lid],
                memory_order_relaxed);
        }
        if (total > 0u) {
            atomic_fetch_add_explicit(&global_hist[lid],
                                      total, memory_order_relaxed);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Compute BucketDesc — derives offset/count/tile_count from
// raw histogram. Must run AFTER msd_histogram, BEFORE global_prefix
// (since prefix overwrites raw counts in-place).
//
// 1 TG, 256 threads. Thread lid handles bucket lid.
// Thread 0 does serial prefix sum for element offsets.
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_compute_bucket_descs(
    device const uint*    global_hist   [[buffer(0)]],
    device BucketDesc*    bucket_descs  [[buffer(1)]],
    constant uint&        tile_size     [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]])
{
    // Each thread reads the raw count for its bucket
    uint count = global_hist[lid];
    uint tile_count = (count + tile_size - 1u) / tile_size;

    // Store count and tile_count in TG memory for thread 0 prefix sum
    threadgroup uint tg_counts[EXP17_NUM_BINS];
    tg_counts[lid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 computes serial prefix sum for offsets
    threadgroup uint tg_offsets[EXP17_NUM_BINS];
    if (lid == 0u) {
        uint running = 0u;
        for (uint i = 0u; i < EXP17_NUM_BINS; i++) {
            tg_offsets[i] = running;
            running += tg_counts[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread writes its BucketDesc
    BucketDesc desc;
    desc.offset     = tg_offsets[lid];
    desc.count      = count;
    desc.tile_count = tile_count;
    desc.tile_base  = 0u;
    bucket_descs[lid] = desc;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: MSD Global Prefix — exclusive prefix sum, 256 bins
//
// Cloned from exp16_global_prefix but only 1 pass (SG 0 only).
// 256-bin prefix sum via 8 chunks of 32 with simd_prefix_exclusive_sum.
// Input/output: global_hist (in-place).
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// Kernel: Inner Zero — zero the tile_hists buffer between inner passes
//
// Simple 1D zero: if (tid < total_entries) tile_hists[tid] = 0;
// total_entries = 256 buckets * 17 tiles * 256 bins = 1,114,112
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_inner_zero(
    device uint*      tile_hists    [[buffer(0)]],
    constant uint&    total_entries [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < total_entries)
        tile_hists[tid] = 0u;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Inner Histogram — per-tile, all buckets in one dispatch
//
// Fixed dispatch: 4352 TGs (EXP17_MAX_TPB * 256 = 17 * 256).
// Arithmetic mapping:
//   bucket_id       = gid / EXP17_MAX_TPB
//   tile_in_bucket  = gid % EXP17_MAX_TPB
//
// Each TG processes one tile of one bucket. Reads BucketDesc for
// bucket boundaries, early-exits if tile exceeds bucket count.
// Same per-SG atomic histogram pattern as exp16 Phase 2.
//
// Output: tile_hists[bucket_id * MAX_TPB * 256 + tile_in_bucket * 256 + bin]
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_inner_histogram(
    device const uint*          data          [[buffer(0)]],
    device uint*                tile_hists    [[buffer(1)]],
    device const BucketDesc*    bucket_descs  [[buffer(2)]],
    constant Exp17InnerParams&  params        [[buffer(3)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // Arithmetic mapping: which bucket, which tile within that bucket
    uint bucket_id      = gid / EXP17_MAX_TPB;
    uint tile_in_bucket = gid % EXP17_MAX_TPB;

    // Read bucket descriptor
    BucketDesc desc = bucket_descs[bucket_id];

    // Early-exit if this tile is beyond the bucket's element count
    uint tile_start = tile_in_bucket * EXP17_TILE_SIZE;
    if (tile_start >= desc.count) return;

    uint bucket_count = desc.count;
    uint base = desc.offset + tile_start;
    uint shift = params.shift;

    // ── Phase 1: Load elements (SG-contiguous layout) ────────────
    uint keys[EXP17_ELEMS];
    bool valid[EXP17_ELEMS];
    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        uint local_idx = tile_start + simd_id * (EXP17_ELEMS * 32u) + e * 32u + simd_lane;
        valid[e] = local_idx < bucket_count;
        uint idx = base + simd_id * (EXP17_ELEMS * 32u) + e * 32u + simd_lane;
        keys[e] = valid[e] ? data[idx] : 0u;
    }

    // ── Phase 2: Per-SG atomic histogram on TG memory ────────────
    threadgroup atomic_uint sg_counts[EXP17_NUM_SGS * EXP17_NUM_BINS]; // 8 KB

    // Zero sg_counts (all 256 threads cooperate)
    for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-SG atomic histogram
    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        if (valid[e]) {
            uint digit = (keys[e] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * EXP17_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Reduce across SGs: 256 threads handle one bin each ───────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP17_NUM_SGS; sg++) {
            total += atomic_load_explicit(
                &sg_counts[sg * EXP17_NUM_BINS + lid],
                memory_order_relaxed);
        }
        // Write to tile_hists: [bucket_id * MAX_TPB * 256 + tile_in_bucket * 256 + bin]
        tile_hists[bucket_id * EXP17_MAX_TPB * EXP17_NUM_BINS
                   + tile_in_bucket * EXP17_NUM_BINS
                   + lid] = total;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Inner Scan+Scatter — serial prefix + rank + scatter
//
// Same dispatch geometry as inner_histogram (4352 TGs, arithmetic
// mapping). Replaces decoupled lookback with serial scan of
// tile_hists for each bin. Each thread handles one bin, scans
// tile_hists[bucket_id * MAX_TPB * 256 + t * 256 + lid] for
// t=0..tile_in_bucket-1 to compute exclusive prefix within bucket.
// Then does per-SG atomic rank + scatter to dst within bucket region.
//
// TG memory: 18KB total (same as exp16_partition).
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_inner_scan_scatter(
    device const uint*          src           [[buffer(0)]],
    device uint*                dst           [[buffer(1)]],
    device const uint*          tile_hists    [[buffer(2)]],
    device const BucketDesc*    bucket_descs  [[buffer(3)]],
    constant Exp17InnerParams&  params        [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // ── Arithmetic mapping: which bucket, which tile within bucket ──
    uint bucket_id      = gid / EXP17_MAX_TPB;
    uint tile_in_bucket = gid % EXP17_MAX_TPB;

    // Read bucket descriptor
    BucketDesc desc = bucket_descs[bucket_id];

    // Early-exit if this tile is beyond the bucket's element count
    uint tile_start = tile_in_bucket * EXP17_TILE_SIZE;
    if (tile_start >= desc.count) return;

    uint bucket_count = desc.count;
    uint base = desc.offset + tile_start;
    uint shift = params.shift;

    // ── TG Memory (20 KB total) ─────────────────────────────────────
    threadgroup atomic_uint sg_hist_or_rank[EXP17_NUM_SGS * EXP17_NUM_BINS]; // 8 KB
    threadgroup uint sg_prefix[EXP17_NUM_SGS * EXP17_NUM_BINS];              // 8 KB
    threadgroup uint tile_hist_local[EXP17_NUM_BINS];                         // 1 KB
    threadgroup uint exclusive_pfx[EXP17_NUM_BINS];                           // 1 KB
    threadgroup uint global_digit_pfx[EXP17_NUM_BINS];                        // 1 KB
    threadgroup uint chunk_totals[8];                                          // 32 B

    // ── Phase 1: Load elements (SG-contiguous layout) ───────────────
    uint keys[EXP17_ELEMS];
    uint digits[EXP17_ELEMS];
    bool valid[EXP17_ELEMS];
    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        uint local_idx = tile_start + simd_id * (EXP17_ELEMS * 32u) + e * 32u + simd_lane;
        valid[e] = local_idx < bucket_count;
        uint idx = base + simd_id * (EXP17_ELEMS * 32u) + e * 32u + simd_lane;
        keys[e] = valid[e] ? src[idx] : 0xFFFFFFFFu;
        digits[e] = valid[e] ? ((keys[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG atomic histogram on TG memory ───────────────
    for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        if (valid[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP17_NUM_BINS + digits[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Tile histogram + cross-SG exclusive prefix ────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP17_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * EXP17_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * EXP17_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist_local[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3a: Sum ALL tiles for each digit → total_for_digit ────
    // Thread lid scans tile_hists[bucket][t][lid] for t=0..tile_count-1
    {
        uint hist_base = bucket_id * EXP17_MAX_TPB * EXP17_NUM_BINS;
        uint total_for_digit = 0u;
        uint cross_tile = 0u;
        for (uint t = 0u; t < desc.tile_count; t++) {
            uint h = tile_hists[hist_base + t * EXP17_NUM_BINS + lid];
            total_for_digit += h;
            if (t < tile_in_bucket) {
                cross_tile += h;
            }
        }
        // Store cross-tile prefix (preceding tiles only) for Phase 4
        exclusive_pfx[lid] = cross_tile;
        // Store total for digit in tile_hist_local temporarily for prefix sum
        // (tile_hist_local was computed in Phase 2b but we can overwrite since
        //  Phase 2b data is already captured in sg_prefix)
        // Actually we still need tile_hist_local for... no, we only need sg_prefix
        // and exclusive_pfx for Phase 4. Let's reuse tile_hist_local for the digit totals.
        tile_hist_local[lid] = total_for_digit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3b: 256-bin exclusive prefix sum across digits ─────────
    // Gives global_digit_pfx[d] = sum of tile_hist_local[d'] for d' < d
    // Uses simd_prefix_exclusive_sum in 8 chunks of 32
    {
        uint chunk = lid / 32u;
        uint lane = lid % 32u;
        uint val = tile_hist_local[lid];
        uint prefix = simd_prefix_exclusive_sum(val);

        // Each chunk's total = last lane's prefix + value
        if (lane == 31u) {
            chunk_totals[chunk] = prefix + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 does serial prefix sum of chunk totals (only 8 values)
        if (lid == 0u) {
            uint running = 0u;
            for (uint c = 0u; c < 8u; c++) {
                uint ct = chunk_totals[c];
                chunk_totals[c] = running;
                running += ct;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        global_digit_pfx[lid] = prefix + chunk_totals[chunk];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Per-SG atomic rank + scatter ───────────────────────
    // Reuse sg_hist_or_rank as rank counters (non-overlapping lifetime)
    for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        if (valid[e]) {
            uint d = digits[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP17_NUM_BINS + d],
                1u, memory_order_relaxed);
            // dst_idx = bucket_start + global_digit_offset + cross_tile_prefix
            //         + cross_sg_prefix + within_sg_rank
            uint dst_idx = desc.offset
                         + global_digit_pfx[d]
                         + exclusive_pfx[d]
                         + sg_prefix[simd_id * EXP17_NUM_BINS + d]
                         + within_sg;
            dst[dst_idx] = keys[e];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: MSD Global Prefix — exclusive prefix sum, 256 bins
//
// Cloned from exp16_global_prefix but only 1 pass (SG 0 only).
// 256-bin prefix sum via 8 chunks of 32 with simd_prefix_exclusive_sum.
// Input/output: global_hist (in-place).
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_global_prefix(
    device uint* global_hist [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // Only SG 0 does work: 256-bin exclusive prefix sum
    if (simd_id == 0u) {
        uint running = 0u;

        for (uint chunk = 0u; chunk < 8u; chunk++) {
            uint bin = chunk * 32u + simd_lane;
            uint val = global_hist[bin];
            uint prefix = simd_prefix_exclusive_sum(val) + running;
            global_hist[bin] = prefix;
            // Broadcast chunk total from lane 31 (uniform lane — safe)
            running += simd_shuffle(simd_prefix_exclusive_sum(val) + val, 31u);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// V2 Kernels: 8192 elements/tile (32 per thread) — for tile size tuning
//
// Duplicates of inner_histogram and inner_scan_scatter but using
// EXP17_TILE_SIZE_LARGE (8192), EXP17_ELEMS_LARGE (32), EXP17_MAX_TPB_V2 (9).
// Halves tile count per bucket (~8 vs ~16), doubles work per TG.
// Higher register pressure (32 keys + 32 digits = 64 register slots).
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_inner_histogram_v2(
    device const uint*          data          [[buffer(0)]],
    device uint*                tile_hists    [[buffer(1)]],
    device const BucketDesc*    bucket_descs  [[buffer(2)]],
    constant Exp17InnerParams&  params        [[buffer(3)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // Arithmetic mapping with V2 max tiles per bucket
    uint bucket_id      = gid / EXP17_MAX_TPB_V2;
    uint tile_in_bucket = gid % EXP17_MAX_TPB_V2;

    BucketDesc desc = bucket_descs[bucket_id];

    uint tile_start = tile_in_bucket * EXP17_TILE_SIZE_LARGE;
    if (tile_start >= desc.count) return;

    uint bucket_count = desc.count;
    uint base = desc.offset + tile_start;
    uint shift = params.shift;

    // ── Phase 1: Load 32 elements per thread (SG-contiguous layout) ──
    uint keys[EXP17_ELEMS_LARGE];
    bool valid[EXP17_ELEMS_LARGE];
    for (uint e = 0u; e < EXP17_ELEMS_LARGE; e++) {
        uint local_idx = tile_start + simd_id * (EXP17_ELEMS_LARGE * 32u) + e * 32u + simd_lane;
        valid[e] = local_idx < bucket_count;
        uint idx = base + simd_id * (EXP17_ELEMS_LARGE * 32u) + e * 32u + simd_lane;
        keys[e] = valid[e] ? data[idx] : 0u;
    }

    // ── Phase 2: Per-SG atomic histogram on TG memory ────────────
    threadgroup atomic_uint sg_counts[EXP17_NUM_SGS * EXP17_NUM_BINS];

    for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP17_ELEMS_LARGE; e++) {
        if (valid[e]) {
            uint digit = (keys[e] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * EXP17_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Reduce across SGs ───────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP17_NUM_SGS; sg++) {
            total += atomic_load_explicit(
                &sg_counts[sg * EXP17_NUM_BINS + lid],
                memory_order_relaxed);
        }
        tile_hists[bucket_id * EXP17_MAX_TPB_V2 * EXP17_NUM_BINS
                   + tile_in_bucket * EXP17_NUM_BINS
                   + lid] = total;
    }
}

kernel void exp17_inner_scan_scatter_v2(
    device const uint*          src           [[buffer(0)]],
    device uint*                dst           [[buffer(1)]],
    device const uint*          tile_hists    [[buffer(2)]],
    device const BucketDesc*    bucket_descs  [[buffer(3)]],
    constant Exp17InnerParams&  params        [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint bucket_id      = gid / EXP17_MAX_TPB_V2;
    uint tile_in_bucket = gid % EXP17_MAX_TPB_V2;

    BucketDesc desc = bucket_descs[bucket_id];

    uint tile_start = tile_in_bucket * EXP17_TILE_SIZE_LARGE;
    if (tile_start >= desc.count) return;

    uint bucket_count = desc.count;
    uint base = desc.offset + tile_start;
    uint shift = params.shift;

    // ── TG Memory (~20 KB) ─────────────────────────────────────
    threadgroup atomic_uint sg_hist_or_rank[EXP17_NUM_SGS * EXP17_NUM_BINS];
    threadgroup uint sg_prefix[EXP17_NUM_SGS * EXP17_NUM_BINS];
    threadgroup uint tile_hist_local[EXP17_NUM_BINS];
    threadgroup uint exclusive_pfx[EXP17_NUM_BINS];
    threadgroup uint global_digit_pfx[EXP17_NUM_BINS];
    threadgroup uint chunk_totals[8];

    // ── Phase 1: Load 32 elements (SG-contiguous layout) ───────────
    uint keys[EXP17_ELEMS_LARGE];
    uint digits[EXP17_ELEMS_LARGE];
    bool valid[EXP17_ELEMS_LARGE];
    for (uint e = 0u; e < EXP17_ELEMS_LARGE; e++) {
        uint local_idx = tile_start + simd_id * (EXP17_ELEMS_LARGE * 32u) + e * 32u + simd_lane;
        valid[e] = local_idx < bucket_count;
        uint idx = base + simd_id * (EXP17_ELEMS_LARGE * 32u) + e * 32u + simd_lane;
        keys[e] = valid[e] ? src[idx] : 0xFFFFFFFFu;
        digits[e] = valid[e] ? ((keys[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG atomic histogram ───────────────
    for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP17_ELEMS_LARGE; e++) {
        if (valid[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP17_NUM_BINS + digits[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Tile histogram + cross-SG exclusive prefix ────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP17_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * EXP17_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * EXP17_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist_local[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3a: Sum ALL tiles for each digit → total_for_digit ────
    {
        uint hist_base = bucket_id * EXP17_MAX_TPB_V2 * EXP17_NUM_BINS;
        uint total_for_digit = 0u;
        uint cross_tile = 0u;
        for (uint t = 0u; t < desc.tile_count; t++) {
            uint h = tile_hists[hist_base + t * EXP17_NUM_BINS + lid];
            total_for_digit += h;
            if (t < tile_in_bucket) {
                cross_tile += h;
            }
        }
        exclusive_pfx[lid] = cross_tile;
        tile_hist_local[lid] = total_for_digit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3b: 256-bin exclusive prefix sum across digits ─────────
    {
        uint chunk = lid / 32u;
        uint lane = lid % 32u;
        uint val = tile_hist_local[lid];
        uint prefix = simd_prefix_exclusive_sum(val);

        if (lane == 31u) {
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

        global_digit_pfx[lid] = prefix + chunk_totals[chunk];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Per-SG atomic rank + scatter ───────────────────────
    for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP17_ELEMS_LARGE; e++) {
        if (valid[e]) {
            uint d = digits[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP17_NUM_BINS + d],
                1u, memory_order_relaxed);
            uint dst_idx = desc.offset
                         + global_digit_pfx[d]
                         + exclusive_pfx[d]
                         + sg_prefix[simd_id * EXP17_NUM_BINS + d]
                         + within_sg;
            dst[dst_idx] = keys[e];
        }
    }
}
