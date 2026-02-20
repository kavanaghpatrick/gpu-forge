#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 21: Pre-Sort Scatter (Stehle-Jacobsen Technique)
//
// KEY INSIGHT from KB #3403, #3548: Pre-sort tile by scatter digit
// in TG memory BEFORE global write. Converts random scatter into
// sequential burst writes. Write amplification drops from ~32x to ~1x.
//
// This experiment tests the pre-sort technique on both MSD scatter
// and inner LSD scatter to measure actual bandwidth improvement.
//
// Architecture: Same as exp17 (MSD + 3×LSD inner) but scatter
// kernels add a local counting sort step before global write.
// ═══════════════════════════════════════════════════════════════════

#define E21_NUM_BINS   256u
#define E21_TILE_SIZE  4096u
#define E21_THREADS    256u
#define E21_ELEMS      16u   // 4096 / 256
#define E21_NUM_SGS    8u
#define E21_MAX_TPB    17u   // max tiles per bucket (62.5K / 4096)

struct E21Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint pass;
};

struct E21InnerParams {
    uint shift;
    uint tg_offset;
};

struct E21BucketDesc {
    uint offset;
    uint count;
    uint tile_count;
    uint tile_base;
};

// ═══════════════════════════════════════════════════════════════════
// Kernel: MSD Histogram — identical to exp17 (no change needed)
// ═══════════════════════════════════════════════════════════════════

kernel void exp21_msd_histogram(
    device const uint*     src          [[buffer(0)]],
    device atomic_uint*    global_hist  [[buffer(1)]],
    constant E21Params&    params       [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint n = params.element_count;
    uint shift = params.shift;
    uint base = gid * E21_TILE_SIZE;

    uint keys[E21_ELEMS];
    bool valid[E21_ELEMS];
    for (uint e = 0u; e < E21_ELEMS; e++) {
        uint idx = base + e * E21_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    threadgroup atomic_uint sg_counts[E21_NUM_SGS * E21_NUM_BINS];
    for (uint i = lid; i < E21_NUM_SGS * E21_NUM_BINS; i += E21_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E21_ELEMS; e++) {
        if (valid[e]) {
            uint digit = (keys[e] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * E21_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = 0u;
    for (uint sg = 0u; sg < E21_NUM_SGS; sg++) {
        total += atomic_load_explicit(
            &sg_counts[sg * E21_NUM_BINS + lid], memory_order_relaxed);
    }
    if (total > 0u) {
        atomic_fetch_add_explicit(&global_hist[lid], total, memory_order_relaxed);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Compute BucketDesc — same as exp17
// ═══════════════════════════════════════════════════════════════════

kernel void exp21_compute_bucket_descs(
    device const uint*    global_hist   [[buffer(0)]],
    device E21BucketDesc* bucket_descs  [[buffer(1)]],
    constant uint&        tile_size     [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]])
{
    uint count = global_hist[lid];
    threadgroup uint tg_counts[E21_NUM_BINS];
    tg_counts[lid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint tg_offsets[E21_NUM_BINS];
    if (lid == 0u) {
        uint running = 0u;
        for (uint i = 0u; i < E21_NUM_BINS; i++) {
            tg_offsets[i] = running;
            running += tg_counts[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    E21BucketDesc desc;
    desc.offset     = tg_offsets[lid];
    desc.count      = count;
    desc.tile_count = (count + tile_size - 1u) / tile_size;
    desc.tile_base  = 0u;
    bucket_descs[lid] = desc;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Global Prefix — exclusive prefix sum on 256 bins
// ═══════════════════════════════════════════════════════════════════

kernel void exp21_global_prefix(
    device uint*  hist  [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup uint tg_vals[E21_NUM_BINS];
    tg_vals[lid] = hist[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 8-chunk prefix sum using simd_prefix_exclusive_sum
    threadgroup uint chunk_totals[8];
    uint chunk = lid / 32u;
    uint val = tg_vals[lid];
    uint prefix = simd_prefix_exclusive_sum(val);
    if ((lid % 32u) == 31u) {
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

    hist[lid] = prefix + chunk_totals[chunk];
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Inner Zero — zero tile_hists between passes
// ═══════════════════════════════════════════════════════════════════

kernel void exp21_inner_zero(
    device uint*     buf     [[buffer(0)]],
    constant uint&   count   [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) buf[tid] = 0u;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Inner Histogram — per-tile histograms (same as exp17)
// ═══════════════════════════════════════════════════════════════════

kernel void exp21_inner_histogram(
    device const uint*          src           [[buffer(0)]],
    device uint*                tile_hists    [[buffer(1)]],
    device const E21BucketDesc* bucket_descs  [[buffer(2)]],
    constant E21InnerParams&    params        [[buffer(3)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint effective_gid  = gid + params.tg_offset;
    uint bucket_id      = effective_gid / E21_MAX_TPB;
    uint tile_in_bucket = effective_gid % E21_MAX_TPB;

    E21BucketDesc desc = bucket_descs[bucket_id];
    uint tile_start = tile_in_bucket * E21_TILE_SIZE;
    if (tile_start >= desc.count) return;

    uint base = desc.offset + tile_start;
    uint shift = params.shift;

    threadgroup atomic_uint sg_counts[E21_NUM_SGS * E21_NUM_BINS];
    for (uint i = lid; i < E21_NUM_SGS * E21_NUM_BINS; i += E21_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E21_ELEMS; e++) {
        uint local_idx = tile_start + simd_id * (E21_ELEMS * 32u) + e * 32u + simd_lane;
        bool v = local_idx < desc.count;
        if (v) {
            uint idx = base + simd_id * (E21_ELEMS * 32u) + e * 32u + simd_lane;
            uint digit = (src[idx] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * E21_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = 0u;
    for (uint sg = 0u; sg < E21_NUM_SGS; sg++) {
        total += atomic_load_explicit(
            &sg_counts[sg * E21_NUM_BINS + lid], memory_order_relaxed);
    }

    uint hist_base = bucket_id * E21_MAX_TPB * E21_NUM_BINS;
    tile_hists[hist_base + tile_in_bucket * E21_NUM_BINS + lid] = total;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Inner Pre-Sort Scatter — THE KEY INNOVATION
//
// Same as exp17_inner_scan_scatter but adds Phase 4b: local counting
// sort by digit in TG memory, then Phase 5: coalesced sequential write.
//
// TG Memory layout (31.5 KB):
//   shared_keys[4096]       = 16 KB  (loaded data / sorted output)
//   shared_sorted[3584]     = 14 KB  (local sort destination)
//   sg_hist_or_rank[8*256]  = 8  KB  (per-SG histogram/rank — reused)
//   sg_prefix[8*256]        = 8  KB  (per-SG prefix — reused)
//   ...
// Actually, TG memory is tight. Let me use a 2-phase approach:
// Phase A: compute ranks + cross-tile prefix (like exp17)
// Phase B: write to shared_keys in digit-sorted order
// Phase C: coalesced write from shared_keys to global
//
// The overhead is: Phase B (TG memory shuffle) + Phase C (sequential write)
// vs exp17's: Phase 4 (random global write)
// ═══════════════════════════════════════════════════════════════════

kernel void exp21_inner_presort_scatter(
    device const uint*          src           [[buffer(0)]],
    device uint*                dst           [[buffer(1)]],
    device const uint*          tile_hists    [[buffer(2)]],
    device const E21BucketDesc* bucket_descs  [[buffer(3)]],
    constant E21InnerParams&    params        [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint effective_gid  = gid + params.tg_offset;
    uint bucket_id      = effective_gid / E21_MAX_TPB;
    uint tile_in_bucket = effective_gid % E21_MAX_TPB;

    E21BucketDesc desc = bucket_descs[bucket_id];
    uint tile_start = tile_in_bucket * E21_TILE_SIZE;
    if (tile_start >= desc.count) return;

    uint bucket_count = desc.count;
    uint base = desc.offset + tile_start;
    uint shift = params.shift;

    // ── TG Memory ────────────────────────────────────────────────
    // We reuse arrays across phases to stay within 32KB.
    // Phase 1-3: sg_hist(8KB) + sg_prefix(8KB) + tile_hist(1KB)
    //            + exclusive_pfx(1KB) + digit_pfx(1KB) + chunk_totals(32B)
    // Phase 4:   shared_keys(16KB) for pre-sorted data
    // Phase 5:   coalesced write from shared_keys
    //
    // Since sg_hist/sg_prefix are done before Phase 4, and shared_keys
    // overlaps them: we use a union. But MSL doesn't support unions in
    // threadgroup. Instead, reuse the 8KB+8KB=16KB from sg arrays
    // as shared_keys[4096].

    threadgroup atomic_uint sg_hist_or_rank[E21_NUM_SGS * E21_NUM_BINS]; // 8 KB
    threadgroup uint sg_prefix[E21_NUM_SGS * E21_NUM_BINS];              // 8 KB
    threadgroup uint tile_hist_local[E21_NUM_BINS];                       // 1 KB
    threadgroup uint exclusive_pfx[E21_NUM_BINS];                         // 1 KB
    threadgroup uint global_digit_pfx[E21_NUM_BINS];                      // 1 KB
    threadgroup uint chunk_totals[8];                                      // 32 B
    // For pre-sort: reuse sg_hist_or_rank + sg_prefix as shared_keys
    // (16KB = 4096 uint) — used AFTER Phase 3 is complete
    threadgroup uint* shared_keys = reinterpret_cast<threadgroup uint*>(&sg_hist_or_rank[0]);
    // NOTE: shared_keys aliases sg_hist_or_rank. Only use AFTER sg_hist_or_rank
    // is no longer needed (after Phase 3).

    // ── Phase 1: Load elements ───────────────────────────────────
    uint keys[E21_ELEMS];
    uint digits[E21_ELEMS];
    bool valid[E21_ELEMS];
    for (uint e = 0u; e < E21_ELEMS; e++) {
        uint local_idx = tile_start + simd_id * (E21_ELEMS * 32u) + e * 32u + simd_lane;
        valid[e] = local_idx < bucket_count;
        uint idx = base + simd_id * (E21_ELEMS * 32u) + e * 32u + simd_lane;
        keys[e] = valid[e] ? src[idx] : 0xFFFFFFFFu;
        digits[e] = valid[e] ? ((keys[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG atomic histogram ─────────────────────────
    for (uint i = lid; i < E21_NUM_SGS * E21_NUM_BINS; i += E21_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E21_ELEMS; e++) {
        if (valid[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * E21_NUM_BINS + digits[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Tile histogram + cross-SG prefix ───────────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < E21_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * E21_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * E21_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist_local[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3a: Cross-tile prefix from tile_hists ──────────────
    {
        uint hist_base = bucket_id * E21_MAX_TPB * E21_NUM_BINS;
        uint total_for_digit = 0u;
        uint cross_tile = 0u;
        for (uint t = 0u; t < desc.tile_count; t++) {
            uint h = tile_hists[hist_base + t * E21_NUM_BINS + lid];
            total_for_digit += h;
            if (t < tile_in_bucket) {
                cross_tile += h;
            }
        }
        exclusive_pfx[lid] = cross_tile;
        tile_hist_local[lid] = total_for_digit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3b: 256-bin exclusive prefix sum ───────────────────
    {
        uint chunk = lid / 32u;
        uint val = tile_hist_local[lid];
        uint prefix = simd_prefix_exclusive_sum(val);
        if ((lid % 32u) == 31u) {
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

    // ── Phase 4: Compute ranks (per-SG atomic) + STORE TO SHARED ─
    // Instead of writing directly to global (random scatter), we
    // write to shared_keys at the local rank position. This produces
    // a digit-sorted tile in TG memory.
    //
    // IMPORTANT: sg_hist_or_rank is reused as rank counters here,
    // and shared_keys aliases the SAME memory. We carefully separate:
    // - First: use sg_hist_or_rank[0..2048] as rank counters
    // - Then:  write to shared_keys[0..4096] (aliases sg_hist_or_rank)
    //
    // This won't work with aliasing! Need separate approach.
    // Instead: compute rank into registers, barrier, then write.

    // Save the global output positions per-element in registers
    uint dst_offsets[E21_ELEMS];
    uint local_ranks[E21_ELEMS];

    // Reset rank counters
    for (uint i = lid; i < E21_NUM_SGS * E21_NUM_BINS; i += E21_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E21_ELEMS; e++) {
        if (valid[e]) {
            uint d = digits[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * E21_NUM_BINS + d],
                1u, memory_order_relaxed);

            // Global output position
            dst_offsets[e] = desc.offset
                           + global_digit_pfx[d]
                           + exclusive_pfx[d]
                           + sg_prefix[simd_id * E21_NUM_BINS + d]
                           + within_sg;

            // Local rank within tile (for TG memory placement)
            // = digit_prefix_within_tile + cross_sg_prefix + within_sg
            // where digit_prefix_within_tile skips over all preceding digits
            // We can compute: position in digit-sorted tile order
            local_ranks[e] = global_digit_pfx[d]  // skip digits 0..d-1 (total within bucket)
                           - exclusive_pfx[lid]    // NO — this is per-thread, wrong!
                           ;
            // Actually the local rank is simpler. After sorting by digit, element
            // goes to position: (prefix_of_digit_d within THIS tile) + rank_within_digit.
            // The tile-local prefix for digit d = sum of tile_hist_local[0..d-1]
            // which is exactly global_digit_pfx[d] (computed in Phase 3b over tile totals).
            // Wait no — global_digit_pfx is computed over TOTAL counts across ALL tiles,
            // not just this tile. We need the tile-local prefix.
            //
            // SIMPLIFICATION: We don't need local_ranks at all!
            // Instead of sorting in TG memory, we can just sort the dst_offsets
            // and keys together, then write sequentially.
            //
            // OR even simpler: just write elements to shared_keys in a way that
            // groups same-digit elements together. Use the tile-local counting sort.
        } else {
            dst_offsets[e] = 0u;
            local_ranks[e] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4b: Write keys to shared memory in digit-sorted order ──
    // Now sg_hist_or_rank is done. We can use shared_keys (aliased memory).
    // We also need to store dst_offsets in shared memory to read them
    // back in sequential order.
    //
    // Problem: we have 4096 keys + 4096 dst_offsets = 32KB. That's all
    // of our TG memory. We don't have room for both.
    //
    // ALTERNATIVE APPROACH: Don't sort in TG memory. Instead, write to
    // global memory in groups by digit. Each digit's elements are
    // written in a burst.
    //
    // But that requires knowing which elements have which digit, which
    // means scanning all elements per digit — O(256 × 4096) = too slow.
    //
    // REVISED APPROACH: Use a two-phase write.
    // Phase A: Write all keys to shared_keys at rank position (digit-sorted)
    // Phase B: Write all dst_offsets to a second shared array at same rank
    // Phase C: Sequential write from shared arrays to global
    //
    // Need: shared_keys[4096] = 16KB, shared_offsets[4096] = 16KB = 32KB total.
    // That's exactly the TG limit but leaves no room for anything else.
    //
    // ACTUAL IMPLEMENTATION: Use sg_prefix (8KB) = shared_offsets[2048]
    // plus the freed sg_hist_or_rank (8KB) = shared_keys first half.
    // Process in TWO batches of 2048 elements if needed.
    //
    // SIMPLEST APPROACH: Just write dst_offsets to registers (already done),
    // and write keys to shared_keys at the ORIGINAL position, then
    // sort shared_keys by digit using a simple in-TG counting sort.
    // The counting sort reads shared_keys, counts digits, computes prefix,
    // and writes to output positions. All in TG memory.

    // ── SIMPLEST: Direct coalesced write approach ────────────────
    // We already computed dst_offsets[e] for each element.
    // We want consecutive threads to write to consecutive dst addresses.
    //
    // Step 1: Write (key, dst_offset) pairs to shared memory at thread-linear positions
    // Step 2: Sort shared memory by dst_offset (which groups same-digit elements)
    // Step 3: Sequential write: thread i writes shared_keys[i] to shared_offsets[i]
    //
    // Step 2 is a full counting sort on dst_offsets which is complex.
    //
    // ACTUALLY: The simplest approach that works:
    // We already have the local rank (within-tile position after digit sort).
    // Compute it properly: for each element, its position in the tile
    // after sorting by digit = prefix_of_this_digit_in_tile + rank_within_digit.
    //
    // prefix_of_digit_d_in_tile = sum of tile_hist_local[d'] for d' < d
    // But we overwrote tile_hist_local with total_for_digit!
    //
    // We need to recompute tile-local prefix. Or save it earlier.
    // Let me restructure: save tile_hist_local into global_digit_pfx BEFORE
    // overwriting with total_for_digit.

    // CONCLUSION: The aliasing and TG memory constraints make this approach
    // complex. Let me use a CLEAN implementation with explicit arrays.
    // Trade: slightly smaller tile (3072) to fit everything.

    // FOR NOW: Fall back to random scatter (same as exp17) as baseline.
    // The pre-sort variant will be in exp21_inner_presort_scatter_v2.
    for (uint e = 0u; e < E21_ELEMS; e++) {
        if (valid[e]) {
            dst[dst_offsets[e]] = keys[e];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Inner Pre-Sort Scatter V2 — MEMORY-POOLED IMPLEMENTATION
//
// Stehle-Jacobsen technique: pre-sort tile by digit in TG memory,
// then write to global in sequential bursts per digit.
//
// TG Memory Pool (19,488 bytes, well under 32KB limit):
//   tg_pool[0..2047]     = sg_hist (Phase 2-4a, as atomic_uint*)
//                        → shared_reorder[0..2047] (Phase 4b-5)
//   tg_pool[2048..4095]  = sg_pfx (Phase 2b-4a)
//                        → shared_reorder[2048..4095] (Phase 4b-5)
//   tg_pool[4096..4351]  = tile_pfx[256] (all phases)
//   tg_pool[4352..4607]  = exclusive_pfx[256] (Phase 3-5)
//   tg_pool[4608..4863]  = digit_pfx[256] (Phase 3b-5)
//   tg_pool[4864..4871]  = chunk_totals[8] (Phase 3)
//
// KEY DESIGN: Phase 4 splits into 4a (compute tile_ranks into
// registers) and 4b (write keys to shared_reorder). Between 4a
// and 4b, sg_hist and sg_pfx are dead → safely aliased.
// ═══════════════════════════════════════════════════════════════════

kernel void exp21_inner_presort_scatter_v2(
    device const uint*          src           [[buffer(0)]],
    device uint*                dst           [[buffer(1)]],
    device const uint*          tile_hists    [[buffer(2)]],
    device const E21BucketDesc* bucket_descs  [[buffer(3)]],
    constant E21InnerParams&    params        [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint effective_gid  = gid + params.tg_offset;
    uint bucket_id      = effective_gid / E21_MAX_TPB;
    uint tile_in_bucket = effective_gid % E21_MAX_TPB;

    E21BucketDesc desc = bucket_descs[bucket_id];
    uint tile_start = tile_in_bucket * E21_TILE_SIZE;
    if (tile_start >= desc.count) return;

    uint bucket_count = desc.count;
    uint base = desc.offset + tile_start;
    uint shift = params.shift;
    uint tile_elem_count = min(E21_TILE_SIZE, bucket_count - tile_start);

    // ── TG Memory Pool (19,488 bytes < 32KB) ─────────────────────
    threadgroup uint tg_pool[4872];

    // Phase 2-4a aliases (sg_hist/sg_pfx region: [0..4095])
    threadgroup atomic_uint* sg_hist = reinterpret_cast<threadgroup atomic_uint*>(&tg_pool[0]);
    threadgroup uint* sg_pfx         = &tg_pool[2048];

    // Persistent small arrays ([4096..4871])
    threadgroup uint* tile_pfx       = &tg_pool[4096];   // 256 entries
    threadgroup uint* exclusive_pfx  = &tg_pool[4352];   // 256 entries
    threadgroup uint* digit_pfx      = &tg_pool[4608];   // 256 entries
    threadgroup uint* chunk_totals   = &tg_pool[4864];   // 8 entries

    // ── Phase 1: Load elements into registers ────────────────────
    uint keys[E21_ELEMS];
    uint digits[E21_ELEMS];
    bool valid[E21_ELEMS];
    for (uint e = 0u; e < E21_ELEMS; e++) {
        uint local_idx = tile_start + simd_id * (E21_ELEMS * 32u) + e * 32u + simd_lane;
        valid[e] = local_idx < bucket_count;
        uint idx = base + simd_id * (E21_ELEMS * 32u) + e * 32u + simd_lane;
        keys[e] = valid[e] ? src[idx] : 0xFFFFFFFFu;
        digits[e] = valid[e] ? ((keys[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG atomic histogram ─────────────────────────
    for (uint i = lid; i < E21_NUM_SGS * E21_NUM_BINS; i += E21_THREADS) {
        atomic_store_explicit(&sg_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E21_ELEMS; e++) {
        if (valid[e]) {
            atomic_fetch_add_explicit(
                &sg_hist[simd_id * E21_NUM_BINS + digits[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Cross-SG prefix + tile-local histogram ─────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < E21_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist[sg * E21_NUM_BINS + lid], memory_order_relaxed);
            sg_pfx[sg * E21_NUM_BINS + lid] = total;
            total += c;
        }
        tile_pfx[lid] = total;  // tile-local count for digit lid
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3a: Cross-tile prefix from tile_hists ──────────────
    uint my_tile_count;
    {
        uint hist_base = bucket_id * E21_MAX_TPB * E21_NUM_BINS;
        uint total_for_digit = 0u;
        uint cross_tile = 0u;
        for (uint t = 0u; t < desc.tile_count; t++) {
            uint h = tile_hists[hist_base + t * E21_NUM_BINS + lid];
            total_for_digit += h;
            if (t < tile_in_bucket) {
                cross_tile += h;
            }
        }
        exclusive_pfx[lid] = cross_tile;

        // Save tile-local count before overwriting tile_pfx
        my_tile_count = tile_pfx[lid];
        tile_pfx[lid] = total_for_digit;  // temporarily: total across ALL tiles
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3b: 256-bin exclusive prefix sum → digit_pfx ───────
    {
        uint chunk = lid / 32u;
        uint val = tile_pfx[lid];  // total_for_digit
        uint prefix = simd_prefix_exclusive_sum(val);
        if ((lid % 32u) == 31u) {
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

        digit_pfx[lid] = prefix + chunk_totals[chunk];
    }

    // ── Phase 3c: Tile-local prefix sum → tile_pfx ───────────────
    // Restore tile_pfx to tile-local counts, then compute prefix
    tile_pfx[lid] = my_tile_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint chunk = lid / 32u;
        uint val = tile_pfx[lid];
        uint prefix = simd_prefix_exclusive_sum(val);
        if ((lid % 32u) == 31u) {
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

        tile_pfx[lid] = prefix + chunk_totals[chunk];  // tile_pfx[d] = exclusive prefix within tile
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4a: Compute tile_ranks into REGISTERS ──────────────
    // Uses sg_hist (atomics) and sg_pfx (read-only). After this phase
    // both are dead and their memory can be reused as shared_reorder.
    uint tile_ranks[E21_ELEMS];

    // Reset sg_hist as per-SG rank counters
    for (uint i = lid; i < E21_NUM_SGS * E21_NUM_BINS; i += E21_THREADS) {
        atomic_store_explicit(&sg_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E21_ELEMS; e++) {
        if (valid[e]) {
            uint d = digits[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist[simd_id * E21_NUM_BINS + d],
                1u, memory_order_relaxed);

            // tile_rank = (prefix of digit d within tile) + (cross-SG prefix) + (within-SG rank)
            tile_ranks[e] = tile_pfx[d]
                          + sg_pfx[simd_id * E21_NUM_BINS + d]
                          + within_sg;
        } else {
            tile_ranks[e] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4b: Write keys to shared memory in digit-sorted order ──
    // sg_hist and sg_pfx are DEAD. Reuse tg_pool[0..4095] as shared_reorder.
    threadgroup uint* shared_reorder = &tg_pool[0];  // 4096 entries = 16KB

    for (uint e = 0u; e < E21_ELEMS; e++) {
        if (valid[e]) {
            shared_reorder[tile_ranks[e]] = keys[e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Coalesced sequential write from shared to global ──
    // Thread lid handles digit lid. Each digit has ~16 elements (4096/256).
    // Elements for digit d are at shared_reorder[tile_pfx[d] .. tile_pfx[d]+count_d).
    // Global base = desc.offset + digit_pfx[d] + exclusive_pfx[d].
    // Each thread writes its ~16 elements SEQUENTIALLY to consecutive addresses.
    {
        uint d = lid;
        uint local_start = tile_pfx[d];
        uint local_count;
        if (d < 255u) {
            local_count = tile_pfx[d + 1u] - local_start;
        } else {
            local_count = tile_elem_count - local_start;
        }

        uint global_base = desc.offset + digit_pfx[d] + exclusive_pfx[d];

        for (uint i = 0u; i < local_count; i++) {
            dst[global_base + i] = shared_reorder[local_start + i];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Control — exp17-style random scatter (no pre-sort)
// For A/B comparison against pre-sort scatter.
// Identical to exp17_inner_scan_scatter.
// ═══════════════════════════════════════════════════════════════════

kernel void exp21_inner_random_scatter(
    device const uint*          src           [[buffer(0)]],
    device uint*                dst           [[buffer(1)]],
    device const uint*          tile_hists    [[buffer(2)]],
    device const E21BucketDesc* bucket_descs  [[buffer(3)]],
    constant E21InnerParams&    params        [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint effective_gid  = gid + params.tg_offset;
    uint bucket_id      = effective_gid / E21_MAX_TPB;
    uint tile_in_bucket = effective_gid % E21_MAX_TPB;

    E21BucketDesc desc = bucket_descs[bucket_id];
    uint tile_start = tile_in_bucket * E21_TILE_SIZE;
    if (tile_start >= desc.count) return;

    uint bucket_count = desc.count;
    uint base = desc.offset + tile_start;
    uint shift = params.shift;

    threadgroup atomic_uint sg_hist_or_rank[E21_NUM_SGS * E21_NUM_BINS];
    threadgroup uint sg_prefix[E21_NUM_SGS * E21_NUM_BINS];
    threadgroup uint tile_hist_local[E21_NUM_BINS];
    threadgroup uint exclusive_pfx[E21_NUM_BINS];
    threadgroup uint global_digit_pfx[E21_NUM_BINS];
    threadgroup uint chunk_totals[8];

    // Phase 1: Load
    uint keys[E21_ELEMS];
    uint digits[E21_ELEMS];
    bool valid[E21_ELEMS];
    for (uint e = 0u; e < E21_ELEMS; e++) {
        uint local_idx = tile_start + simd_id * (E21_ELEMS * 32u) + e * 32u + simd_lane;
        valid[e] = local_idx < bucket_count;
        uint idx = base + simd_id * (E21_ELEMS * 32u) + e * 32u + simd_lane;
        keys[e] = valid[e] ? src[idx] : 0xFFFFFFFFu;
        digits[e] = valid[e] ? ((keys[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // Phase 2: Histogram
    for (uint i = lid; i < E21_NUM_SGS * E21_NUM_BINS; i += E21_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E21_ELEMS; e++) {
        if (valid[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * E21_NUM_BINS + digits[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2b
    {
        uint total = 0u;
        for (uint sg = 0u; sg < E21_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * E21_NUM_BINS + lid], memory_order_relaxed);
            sg_prefix[sg * E21_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist_local[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3a: Cross-tile prefix
    {
        uint hist_base = bucket_id * E21_MAX_TPB * E21_NUM_BINS;
        uint total_for_digit = 0u;
        uint cross_tile = 0u;
        for (uint t = 0u; t < desc.tile_count; t++) {
            uint h = tile_hists[hist_base + t * E21_NUM_BINS + lid];
            total_for_digit += h;
            if (t < tile_in_bucket) {
                cross_tile += h;
            }
        }
        exclusive_pfx[lid] = cross_tile;
        tile_hist_local[lid] = total_for_digit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3b: Prefix sum
    {
        uint chunk = lid / 32u;
        uint val = tile_hist_local[lid];
        uint prefix = simd_prefix_exclusive_sum(val);
        if ((lid % 32u) == 31u) {
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

    // Phase 4: Random scatter (baseline — same as exp17)
    for (uint i = lid; i < E21_NUM_SGS * E21_NUM_BINS; i += E21_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E21_ELEMS; e++) {
        if (valid[e]) {
            uint d = digits[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * E21_NUM_BINS + d],
                1u, memory_order_relaxed);
            uint dst_idx = desc.offset
                         + global_digit_pfx[d]
                         + exclusive_pfx[d]
                         + sg_prefix[simd_id * E21_NUM_BINS + d]
                         + within_sg;
            dst[dst_idx] = keys[e];
        }
    }
}
