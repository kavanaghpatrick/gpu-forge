#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 22: Work-Queue Local Sort
//
// KB #3460 Approach 2: 1 MSD scatter + SLC-resident local sort
// Projected: 5290 Mkeys/s
//
// Architecture:
//   Phase 1: MSD scatter (bits 24-31) → 256 buckets (~62K each)
//            Uses exp16_partition (decoupled lookback, proven)
//   Phase 2: Per-bucket local sort via work queue
//            Each TG grabs a tile from any bucket, loads into TG mem,
//            does 3-pass 8-bit counting sort ENTIRELY in TG memory,
//            then writes sorted output sequentially to global.
//
// Why this is different from exp17:
//   exp17: 3 global passes → 3× read+write entire 16M array through DRAM
//   exp22: 3 LOCAL passes → 3× read+write 2048 elements in TG memory
//          Only 1 global read + 1 global write per tile (at SLC speed)
//
// TG Memory Pool (~18 KB):
//   keys_a[2048]     = 8 KB  (ping buffer)
//   keys_b[2048]     = 8 KB  (pong buffer)
//   hist_cnt[256]    = 1 KB  (atomic histogram / scatter counters)
//   digit_pfx[256]   = 1 KB  (exclusive prefix sum)
//   chunk_totals[8]  = 32 B  (inter-SG prefix)
//   Total: 4616 × 4 = 18,464 bytes < 32 KB ✓
//
// Each local pass:
//   1. Zero hist_cnt[256]
//   2. Histogram: each element → atomic_fetch_add on hist_cnt[digit]
//   3. Read histogram → compute exclusive prefix sum → digit_pfx
//   4. Initialize hist_cnt[d] = digit_pfx[d] (scatter starting positions)
//   5. Scatter: pos = atomic_fetch_add(hist_cnt[digit], 1), write keys_b[pos]
//   6. Swap keys_a ↔ keys_b
// ═══════════════════════════════════════════════════════════════════

#define E22_TILE       1024u
#define E22_THREADS    256u
#define E22_ELEMS      4u      // 1024 / 256
#define E22_NUM_SGS    8u
#define E22_NUM_BINS   256u
#define E22_MAX_TPB    62u     // ceil(62500 / 1024) — max tiles per bucket

struct E22Params {
    uint element_count;
    uint num_tiles;
    uint shift;         // MSD shift (24)
    uint pass;
};

struct E22InnerParams {
    uint total_inner_tiles;   // sum of all bucket tile_counts (upper bound)
};

struct E22BucketDesc {
    uint offset;        // start index in sorted-by-MSD array
    uint count;         // number of elements in this bucket
    uint tile_count;    // ceil(count / E22_TILE)
    uint tile_base;     // cumulative tile index (prefix sum of tile_counts)
};

// ═══════════════════════════════════════════════════════════════════
// Kernel: Compute BucketDescs with tile_base (cumulative prefix)
// ═══════════════════════════════════════════════════════════════════

kernel void exp22_compute_bucket_descs(
    device const uint*     global_hist   [[buffer(0)]],
    device E22BucketDesc*  bucket_descs  [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]])
{
    uint count = global_hist[lid];
    threadgroup uint tg_counts[E22_NUM_BINS];
    tg_counts[lid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute offsets (exclusive prefix sum) — serial on thread 0
    threadgroup uint tg_offsets[E22_NUM_BINS];
    threadgroup uint tg_tile_base[E22_NUM_BINS];
    if (lid == 0u) {
        uint running_offset = 0u;
        uint running_tiles = 0u;
        for (uint i = 0u; i < E22_NUM_BINS; i++) {
            tg_offsets[i] = running_offset;
            tg_tile_base[i] = running_tiles;
            running_offset += tg_counts[i];
            running_tiles += (tg_counts[i] + E22_TILE - 1u) / E22_TILE;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    E22BucketDesc desc;
    desc.offset     = tg_offsets[lid];
    desc.count      = count;
    desc.tile_count = (count + E22_TILE - 1u) / E22_TILE;
    desc.tile_base  = tg_tile_base[lid];
    bucket_descs[lid] = desc;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Local Sort — THE KEY INNOVATION
//
// Each TG processes one tile from one bucket.
// Dispatch: total_inner_tiles TGs (upper bound). Extra TGs early-exit.
//
// Performs 3-pass 8-bit counting sort ENTIRELY in TG memory.
// Only reads global once (load tile) and writes global once (store sorted).
//
// Simplified approach: 256 global atomic counters per digit (no per-SG split).
// For 2048-element tiles with 256 bins, contention is ~8 ops/counter — trivial.
// ═══════════════════════════════════════════════════════════════════

kernel void exp22_local_sort(
    device const uint*          src           [[buffer(0)]],
    device uint*                dst           [[buffer(1)]],
    device const E22BucketDesc* bucket_descs  [[buffer(2)]],
    constant E22InnerParams&    params        [[buffer(3)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    // ── Find which bucket and tile this TG handles ───────────────
    // Linear scan of 256 bucket_descs.tile_base (L1-cached after first TG)
    uint bucket_id = 0u;
    for (uint b = 1u; b < E22_NUM_BINS; b++) {
        if (bucket_descs[b].tile_base <= gid) {
            bucket_id = b;
        }
    }

    E22BucketDesc desc = bucket_descs[bucket_id];
    uint tile_in_bucket = gid - desc.tile_base;
    uint tile_start = tile_in_bucket * E22_TILE;
    if (tile_start >= desc.count) return;

    uint tile_elem_count = min(E22_TILE, desc.count - tile_start);
    uint base = desc.offset + tile_start;

    // ── TG Memory Pool (26,656 bytes < 32 KB) ─────────────────────
    // keys_a[1024]        = 4 KB  (ping buffer)
    // keys_b[1024]        = 4 KB  (pong buffer)
    // sg_cnt[8*256]       = 8 KB  (per-SG atomic histogram + accumulator)
    // sg_pfx[8*256]       = 8 KB  (cross-SG prefix, SEPARATE from sg_cnt)
    // digit_pfx[256]      = 1 KB  (exclusive prefix sum of digit totals)
    // chunk_totals[8]     = 32 B  (inter-SG prefix sum scratch)
    // scratch_digits[256] = 1 KB  (per-thread digit buffer for stable rank)
    threadgroup uint tg_pool[6664];

    threadgroup uint* keys_a              = &tg_pool[0];      // [0..1023]
    threadgroup uint* keys_b              = &tg_pool[1024];   // [1024..2047]
    threadgroup atomic_uint* sg_cnt       = reinterpret_cast<threadgroup atomic_uint*>(&tg_pool[2048]);
                                                               // [2048..4095]
    threadgroup uint* sg_pfx              = &tg_pool[4096];   // [4096..6143]
    threadgroup uint* digit_pfx           = &tg_pool[6144];   // [6144..6399]
    threadgroup uint* chunk_totals        = &tg_pool[6400];   // [6400..6407]
    threadgroup uint* scratch_digits      = &tg_pool[6408];   // [6408..6663]

    // ── Load tile from global into keys_a (only real elements) ───
    for (uint i = lid; i < tile_elem_count; i += E22_THREADS) {
        keys_a[i] = src[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3 local counting sort passes (bits 0-7, 8-15, 16-23) ────
    // Uses per-SG histogram + separate sg_pfx for STABLE counting sort.
    // Stability is required for LSD radix sort correctness across passes.
    uint local_shifts[3] = {0u, 8u, 16u};

    for (uint pass = 0u; pass < 3u; pass++) {
        uint shift = local_shifts[pass];

        // Step 1: Zero per-SG histogram (8×256 = 2048 entries)
        for (uint i = lid; i < E22_NUM_SGS * E22_NUM_BINS; i += E22_THREADS) {
            atomic_store_explicit(&sg_cnt[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: Per-SG histogram — each element counts into its SG's bin
        for (uint e = 0u; e < E22_ELEMS; e++) {
            uint idx = lid + e * E22_THREADS;
            if (idx < tile_elem_count) {
                uint digit = (keys_a[idx] >> shift) & 0xFFu;
                atomic_fetch_add_explicit(
                    &sg_cnt[simd_id * E22_NUM_BINS + digit],
                    1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Cross-SG reduction → digit totals, then exclusive prefix sum
        {
            // Each thread lid handles one digit: sum across 8 SGs
            uint total = 0u;
            for (uint sg = 0u; sg < E22_NUM_SGS; sg++) {
                total += atomic_load_explicit(
                    &sg_cnt[sg * E22_NUM_BINS + lid], memory_order_relaxed);
            }

            // 2-level SIMD exclusive prefix sum on 256 digit totals
            uint chunk = lid / 32u;
            uint prefix = simd_prefix_exclusive_sum(total);

            if ((lid % 32u) == 31u) {
                chunk_totals[chunk] = prefix + total;
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
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 4: Compute cross-SG prefix → sg_pfx (SEPARATE array, stable)
        // For each digit (lid), scan the SG counts to get the offset for each SG
        {
            uint running = 0u;
            for (uint sg = 0u; sg < E22_NUM_SGS; sg++) {
                uint c = atomic_load_explicit(
                    &sg_cnt[sg * E22_NUM_BINS + lid], memory_order_relaxed);
                sg_pfx[sg * E22_NUM_BINS + lid] = running;
                running += c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Steps 5-6: Per-e-iteration stable scatter
        //
        // PROBLEM: The per-SG prefix groups ALL of SG 0 before SG 1.
        // With E22_ELEMS > 1, element 256 (SG 0, e=1) gets placed before
        // element 32 (SG 1, e=0), violating LSD stability (32 < 256).
        //
        // FIX: Process each e iteration independently:
        //   - Rebuild per-SG histogram for just this e's elements
        //   - Compute fresh sg_pfx for this e iteration
        //   - Use deterministic intra-SG rank (shared digit buffer + scan)
        //   - Advance digit_pfx after each e iteration
        // This ensures e=0's elements (all SGs) come before e=1's elements.
        for (uint e = 0u; e < E22_ELEMS; e++) {
            uint idx = lid + e * E22_THREADS;
            bool active = (idx < tile_elem_count);

            // 6a: Build per-SG histogram for this e iteration only
            for (uint i = lid; i < E22_NUM_SGS * E22_NUM_BINS; i += E22_THREADS) {
                atomic_store_explicit(&sg_cnt[i], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint digit = 0xFFFFFFFFu;
            uint key = 0u;
            if (active) {
                key = keys_a[idx];
                digit = (key >> shift) & 0xFFu;
                atomic_fetch_add_explicit(
                    &sg_cnt[simd_id * E22_NUM_BINS + digit],
                    1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 6b: Compute cross-SG prefix for this e iteration
            {
                uint running = 0u;
                for (uint sg = 0u; sg < E22_NUM_SGS; sg++) {
                    uint c = atomic_load_explicit(
                        &sg_cnt[sg * E22_NUM_BINS + lid], memory_order_relaxed);
                    sg_pfx[sg * E22_NUM_BINS + lid] = running;
                    running += c;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 6c: Deterministic intra-SG rank via shared digit buffer
            scratch_digits[lid] = digit;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active) {
                uint intra_rank = 0u;
                uint sg_base = simd_id * 32u;
                for (uint j = 0u; j < simd_lane; j++) {
                    if (scratch_digits[sg_base + j] == digit) {
                        intra_rank++;
                    }
                }

                uint pos = digit_pfx[digit]
                         + sg_pfx[simd_id * E22_NUM_BINS + digit]
                         + intra_rank;
                keys_b[pos] = key;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 6d: Advance digit_pfx by this e iteration's per-digit totals
            {
                uint total = 0u;
                for (uint sg = 0u; sg < E22_NUM_SGS; sg++) {
                    total += atomic_load_explicit(
                        &sg_cnt[sg * E22_NUM_BINS + lid], memory_order_relaxed);
                }
                digit_pfx[lid] += total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Step 7: Swap keys_a ↔ keys_b for next pass
        threadgroup uint* tmp = keys_a;
        keys_a = keys_b;
        keys_b = tmp;
    }

    // After 3 passes (odd), result is in keys_a (which points to tg_pool[1024..2047])

    // ── Write sorted tile to global output (sequential) ──────────
    for (uint i = lid; i < tile_elem_count; i += E22_THREADS) {
        dst[base + i] = keys_a[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Global Prefix — 256-bin exclusive prefix sum
// ═══════════════════════════════════════════════════════════════════

kernel void exp22_global_prefix(
    device uint*  hist  [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup uint tg_vals[E22_NUM_BINS];
    tg_vals[lid] = hist[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
// Kernel: MSD Histogram — per-SG atomic histogram for top 8 bits
// ═══════════════════════════════════════════════════════════════════

kernel void exp22_msd_histogram(
    device const uint*     src          [[buffer(0)]],
    device atomic_uint*    global_hist  [[buffer(1)]],
    constant E22Params&    params       [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    // Use 4096-element tiles for MSD histogram (same as exp17)
    #define E22_MSD_TILE  4096u
    #define E22_MSD_ELEMS 16u

    uint n = params.element_count;
    uint shift = params.shift;
    uint base = gid * E22_MSD_TILE;

    uint keys[E22_MSD_ELEMS];
    bool valid[E22_MSD_ELEMS];
    for (uint e = 0u; e < E22_MSD_ELEMS; e++) {
        uint idx = base + e * E22_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    threadgroup atomic_uint sg_counts[E22_NUM_SGS * E22_NUM_BINS];
    for (uint i = lid; i < E22_NUM_SGS * E22_NUM_BINS; i += E22_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E22_MSD_ELEMS; e++) {
        if (valid[e]) {
            uint digit = (keys[e] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * E22_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = 0u;
    for (uint sg = 0u; sg < E22_NUM_SGS; sg++) {
        total += atomic_load_explicit(
            &sg_counts[sg * E22_NUM_BINS + lid], memory_order_relaxed);
    }
    if (total > 0u) {
        atomic_fetch_add_explicit(&global_hist[lid], total, memory_order_relaxed);
    }
}
