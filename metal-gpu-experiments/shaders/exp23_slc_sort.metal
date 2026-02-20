#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 23: SLC-Speed Per-Bucket Global LSD Sort
//
// Architecture:
//   Phase 1: MSD scatter (bits 24-31) → 256 buckets (~62K each, ~250KB)
//            Reuses exp16_partition (proven decoupled lookback)
//   Phase 2: 3 per-bucket LSD passes (bits 0-7, 8-15, 16-23)
//            Each pass: decoupled lookback WITHIN bucket bounds
//            ~250KB buckets stay SLC-resident (469 GB/s vs 245 DRAM)
//
// Key insight: KB #3460's "SLC-resident local sort" means global LSD
// passes operating on SLC-sized buckets, NOT TG-memory counting sort.
//
// Estimated: MSD 0.52ms + 3 inner passes @ SLC = ~1.5ms → ~10000 Mk/s
// ═══════════════════════════════════════════════════════════════════

#define E23_TILE_SIZE  4096u
#define E23_THREADS    256u
#define E23_ELEMS      16u
#define E23_NUM_SGS    8u
#define E23_NUM_BINS   256u

struct E23BucketDesc {
    uint offset;        // start index in MSD-sorted array
    uint count;         // number of elements in this bucket
    uint tile_count;    // ceil(count / E23_TILE_SIZE)
    uint tile_base;     // cumulative tile index (prefix sum of tile_counts)
};

struct E23InnerParams {
    uint total_inner_tiles;
    uint shift;
};

struct E23Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint pass;
};

// ═══════════════════════════════════════════════════════════════════
// Kernel: Compute BucketDescs with 4096-element tiles
// 256 threads = 256 bins, one per thread
// ═══════════════════════════════════════════════════════════════════

kernel void exp23_compute_bucket_descs(
    device const uint*     global_hist   [[buffer(0)]],
    device E23BucketDesc*  bucket_descs  [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]])
{
    uint count = global_hist[lid];
    threadgroup uint tg_counts[E23_NUM_BINS];
    tg_counts[lid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint tg_offsets[E23_NUM_BINS];
    threadgroup uint tg_tile_base[E23_NUM_BINS];
    if (lid == 0u) {
        uint running_offset = 0u;
        uint running_tiles = 0u;
        for (uint i = 0u; i < E23_NUM_BINS; i++) {
            tg_offsets[i] = running_offset;
            tg_tile_base[i] = running_tiles;
            running_offset += tg_counts[i];
            running_tiles += (tg_counts[i] + E23_TILE_SIZE - 1u) / E23_TILE_SIZE;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    E23BucketDesc desc;
    desc.offset     = tg_offsets[lid];
    desc.count      = count;
    desc.tile_count = (count + E23_TILE_SIZE - 1u) / E23_TILE_SIZE;
    desc.tile_base  = tg_tile_base[lid];
    bucket_descs[lid] = desc;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: MSD Histogram — same as exp22 (per-SG atomic, top 8 bits)
// ═══════════════════════════════════════════════════════════════════

kernel void exp23_msd_histogram(
    device const uint*     src          [[buffer(0)]],
    device atomic_uint*    global_hist  [[buffer(1)]],
    constant E23Params&    params       [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    #define E23_MSD_TILE  4096u
    #define E23_MSD_ELEMS 16u

    uint n = params.element_count;
    uint shift = params.shift;
    uint base = gid * E23_MSD_TILE;

    uint keys[E23_MSD_ELEMS];
    bool valid[E23_MSD_ELEMS];
    for (uint e = 0u; e < E23_MSD_ELEMS; e++) {
        uint idx = base + e * E23_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    threadgroup atomic_uint sg_counts[E23_NUM_SGS * E23_NUM_BINS];
    for (uint i = lid; i < E23_NUM_SGS * E23_NUM_BINS; i += E23_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E23_MSD_ELEMS; e++) {
        if (valid[e]) {
            uint digit = (keys[e] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * E23_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = 0u;
    for (uint sg = 0u; sg < E23_NUM_SGS; sg++) {
        total += atomic_load_explicit(
            &sg_counts[sg * E23_NUM_BINS + lid], memory_order_relaxed);
    }
    if (total > 0u) {
        atomic_fetch_add_explicit(&global_hist[lid], total, memory_order_relaxed);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Global Prefix — 256-bin exclusive prefix sum
// ═══════════════════════════════════════════════════════════════════

kernel void exp23_global_prefix(
    device uint*  hist  [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup uint tg_vals[E23_NUM_BINS];
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
// Kernel: Inner Histogram — per-bucket digit histogram
// 256 TGs (one per bucket), 256 threads each.
// Each TG stride-loops its bucket's elements to count digit occurrences.
// ═══════════════════════════════════════════════════════════════════

kernel void exp23_inner_histogram(
    device const uint*            src          [[buffer(0)]],
    device uint*                  bucket_hist  [[buffer(1)]],
    device const E23BucketDesc*   bucket_descs [[buffer(2)]],
    constant E23InnerParams&      params       [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    uint shift = params.shift;
    E23BucketDesc desc = bucket_descs[gid];
    uint offset = desc.offset;
    uint count = desc.count;

    threadgroup atomic_uint sg_counts[E23_NUM_SGS * E23_NUM_BINS];
    for (uint i = lid; i < E23_NUM_SGS * E23_NUM_BINS; i += E23_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pos = lid; pos < count; pos += E23_THREADS) {
        uint key = src[offset + pos];
        uint digit = (key >> shift) & 0xFFu;
        atomic_fetch_add_explicit(
            &sg_counts[simd_id * E23_NUM_BINS + digit],
            1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = 0u;
    for (uint sg = 0u; sg < E23_NUM_SGS; sg++) {
        total += atomic_load_explicit(
            &sg_counts[sg * E23_NUM_BINS + lid], memory_order_relaxed);
    }
    bucket_hist[gid * E23_NUM_BINS + lid] = total;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Inner Prefix — per-bucket exclusive prefix sum
// 256 TGs (one per bucket), 256 threads each.
// Converts each bucket's histogram to exclusive prefix sums in-place.
// ═══════════════════════════════════════════════════════════════════

kernel void exp23_inner_prefix(
    device uint*  bucket_hist  [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint base = gid * E23_NUM_BINS;
    threadgroup uint tg_vals[E23_NUM_BINS];
    tg_vals[lid] = bucket_hist[base + lid];
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

    bucket_hist[base + lid] = prefix + chunk_totals[chunk];
}

// ═══════════════════════════════════════════════════════════════════
// Kernel: Inner Partition — per-bucket decoupled lookback LSD
//
// Uses pre-computed bucket_hist (cross-digit exclusive prefix) from
// the separate histogram+prefix kernels above.
// ═══════════════════════════════════════════════════════════════════

kernel void exp23_inner_partition(
    device const uint*            src          [[buffer(0)]],
    device uint*                  dst          [[buffer(1)]],
    device atomic_uint*           tile_status  [[buffer(2)]],
    device const E23BucketDesc*   bucket_descs [[buffer(3)]],
    constant E23InnerParams&      params       [[buffer(4)]],
    device const uint*            bucket_hist  [[buffer(5)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint shift = params.shift;

    // ── Find which bucket this TG belongs to ──────────────────────
    // Binary search of 256 bucket_descs.tile_base
    uint bucket_id = 0u;
    for (uint b = 1u; b < E23_NUM_BINS; b++) {
        if (bucket_descs[b].tile_base <= gid) {
            bucket_id = b;
        }
    }

    E23BucketDesc desc = bucket_descs[bucket_id];
    uint tile_in_bucket = gid - desc.tile_base;
    if (tile_in_bucket >= desc.tile_count) return;

    uint n = desc.count;
    uint offset = desc.offset;
    uint base = offset + tile_in_bucket * E23_TILE_SIZE;
    uint end = offset + n;

    // ── TG Memory (20 KB total) ──────────────────────────────────
    threadgroup atomic_uint sg_hist_or_rank[E23_NUM_SGS * E23_NUM_BINS]; // 8 KB
    threadgroup uint sg_prefix[E23_NUM_SGS * E23_NUM_BINS];              // 8 KB
    threadgroup uint tile_hist[E23_NUM_BINS];                             // 1 KB
    threadgroup uint exclusive_pfx[E23_NUM_BINS];                         // 1 KB

    // ── Phase 1: Load (SG-contiguous layout) ─────────────────────
    uint mk[E23_ELEMS];
    uint md[E23_ELEMS];
    bool mv[E23_ELEMS];
    for (int e = 0; e < (int)E23_ELEMS; e++) {
        uint idx = base + simd_id * (E23_ELEMS * 32u) + (uint)e * 32u + simd_lane;
        mv[e] = idx < end;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG atomic histogram ─────────────────────────
    for (uint i = lid; i < E23_NUM_SGS * E23_NUM_BINS; i += E23_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)E23_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * E23_NUM_BINS + md[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Tile histogram + cross-SG prefix ──────────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < E23_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * E23_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * E23_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Publish AGGREGATE ───────────────────────────────
    {
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT)
                    | (tile_hist[lid] & VALUE_MASK);
        atomic_store_explicit(&tile_status[gid * E23_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Decoupled lookback (WITHIN bucket only) ─────────
    {
        uint lk_running = 0u;
        if (tile_in_bucket > 0u) {
            int look = (int)gid - 1;
            int bucket_start = (int)desc.tile_base;
            while (look >= bucket_start) {
                atomic_thread_fence(mem_flags::mem_device,
                                    memory_order_seq_cst,
                                    thread_scope_device);
                uint val = atomic_load_explicit(
                    &tile_status[(uint)look * E23_NUM_BINS + lid],
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
        atomic_store_explicit(&tile_status[gid * E23_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Per-SG atomic ranking + scatter ─────────────────
    // bucket_hist[bucket_id * 256 + d] = cross-digit exclusive prefix
    for (uint i = lid; i < E23_NUM_SGS * E23_NUM_BINS; i += E23_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)E23_ELEMS; e++) {
        if (mv[e]) {
            uint d = md[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * E23_NUM_BINS + d],
                1u, memory_order_relaxed);
            uint gp = offset
                     + bucket_hist[bucket_id * E23_NUM_BINS + d]
                     + exclusive_pfx[d]
                     + sg_prefix[simd_id * E23_NUM_BINS + d]
                     + within_sg;
            dst[gp] = mk[e];
        }
    }
}
