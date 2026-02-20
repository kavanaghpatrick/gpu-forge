#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 24: Batched Inner Passes for SLC Residency
//
// Architecture:
//   Phase 1: MSD scatter (bits 24-31) → 256 buckets (~250KB each)
//            Reuses exp16_partition + exp23 MSD histogram/prefix
//   Phase 2: 3 per-bucket LSD passes with BATCHED dispatch
//            ~90 buckets per batch → ~22MB working set < 24MB SLC
//            Each batch: histogram → prefix → zero → partition
//
// Key insight from multi-AI analysis: exp23 dispatched ALL 4096 TGs
// simultaneously, creating a 64MB working set that defeats SLC.
// Batching keeps the active working set under SLC capacity.
//
// Target: 5000+ Mkeys/s via SLC bandwidth advantage
// ═══════════════════════════════════════════════════════════════════

#define E24_TILE_SIZE  4096u
#define E24_THREADS    256u
#define E24_ELEMS      16u
#define E24_NUM_SGS    8u
#define E24_NUM_BINS   256u

// Reuse BucketDesc from exp23
struct E24BucketDesc {
    uint offset;
    uint count;
    uint tile_count;
    uint tile_base;
};

// Batch-aware params for inner kernels
struct E24BatchParams {
    uint shift;
    uint batch_start;   // first bucket index in this batch
    uint batch_count;   // number of buckets in this batch
    uint tile_offset;   // global tile index of first tile in batch
};

// MSD params (same as exp23)
struct E24Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint pass;
};

// ═══════════════════════════════════════════════════════════════════
// MSD kernels — identical to exp23 (just renamed)
// ═══════════════════════════════════════════════════════════════════

kernel void exp24_msd_histogram(
    device const uint*     src          [[buffer(0)]],
    device atomic_uint*    global_hist  [[buffer(1)]],
    constant E24Params&    params       [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    #define E24_MSD_TILE  4096u
    #define E24_MSD_ELEMS 16u

    uint n = params.element_count;
    uint shift = params.shift;
    uint base = gid * E24_MSD_TILE;

    uint keys[E24_MSD_ELEMS];
    bool valid[E24_MSD_ELEMS];
    for (uint e = 0u; e < E24_MSD_ELEMS; e++) {
        uint idx = base + e * E24_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    threadgroup atomic_uint sg_counts[E24_NUM_SGS * E24_NUM_BINS];
    for (uint i = lid; i < E24_NUM_SGS * E24_NUM_BINS; i += E24_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E24_MSD_ELEMS; e++) {
        if (valid[e]) {
            uint digit = (keys[e] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * E24_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = 0u;
    for (uint sg = 0u; sg < E24_NUM_SGS; sg++) {
        total += atomic_load_explicit(
            &sg_counts[sg * E24_NUM_BINS + lid], memory_order_relaxed);
    }
    if (total > 0u) {
        atomic_fetch_add_explicit(&global_hist[lid], total, memory_order_relaxed);
    }
}

kernel void exp24_compute_bucket_descs(
    device const uint*     global_hist   [[buffer(0)]],
    device E24BucketDesc*  bucket_descs  [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]])
{
    uint count = global_hist[lid];
    threadgroup uint tg_counts[E24_NUM_BINS];
    tg_counts[lid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint tg_offsets[E24_NUM_BINS];
    threadgroup uint tg_tile_base[E24_NUM_BINS];
    if (lid == 0u) {
        uint running_offset = 0u;
        uint running_tiles = 0u;
        for (uint i = 0u; i < E24_NUM_BINS; i++) {
            tg_offsets[i] = running_offset;
            tg_tile_base[i] = running_tiles;
            running_offset += tg_counts[i];
            running_tiles += (tg_counts[i] + E24_TILE_SIZE - 1u) / E24_TILE_SIZE;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    E24BucketDesc desc;
    desc.offset     = tg_offsets[lid];
    desc.count      = count;
    desc.tile_count = (count + E24_TILE_SIZE - 1u) / E24_TILE_SIZE;
    desc.tile_base  = tg_tile_base[lid];
    bucket_descs[lid] = desc;
}

kernel void exp24_global_prefix(
    device uint*  hist  [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup uint tg_vals[E24_NUM_BINS];
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
// BATCHED Inner Histogram — dispatch batch_count TGs per batch
// gid is 0..batch_count-1, actual bucket = gid + batch_start
// ═══════════════════════════════════════════════════════════════════

kernel void exp24_inner_histogram(
    device const uint*            src          [[buffer(0)]],
    device uint*                  bucket_hist  [[buffer(1)]],
    device const E24BucketDesc*   bucket_descs [[buffer(2)]],
    constant E24BatchParams&      params       [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    uint shift = params.shift;
    uint bucket_id = gid + params.batch_start;  // Map local gid to global bucket
    E24BucketDesc desc = bucket_descs[bucket_id];
    uint offset = desc.offset;
    uint count = desc.count;

    threadgroup atomic_uint sg_counts[E24_NUM_SGS * E24_NUM_BINS];
    for (uint i = lid; i < E24_NUM_SGS * E24_NUM_BINS; i += E24_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pos = lid; pos < count; pos += E24_THREADS) {
        uint key = src[offset + pos];
        uint digit = (key >> shift) & 0xFFu;
        atomic_fetch_add_explicit(
            &sg_counts[simd_id * E24_NUM_BINS + digit],
            1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = 0u;
    for (uint sg = 0u; sg < E24_NUM_SGS; sg++) {
        total += atomic_load_explicit(
            &sg_counts[sg * E24_NUM_BINS + lid], memory_order_relaxed);
    }
    bucket_hist[bucket_id * E24_NUM_BINS + lid] = total;
}

// ═══════════════════════════════════════════════════════════════════
// BATCHED Inner Prefix — dispatch batch_count TGs per batch
// ═══════════════════════════════════════════════════════════════════

kernel void exp24_inner_prefix(
    device uint*               bucket_hist  [[buffer(0)]],
    constant E24BatchParams&   params       [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint bucket_id = gid + params.batch_start;
    uint base = bucket_id * E24_NUM_BINS;
    threadgroup uint tg_vals[E24_NUM_BINS];
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
// BATCHED Inner Partition — dispatch batch_tiles TGs per batch
// gid is local (0..batch_tiles-1), effective_gid = gid + tile_offset
// Binary search limited to batch_start..batch_start+batch_count
// ═══════════════════════════════════════════════════════════════════

kernel void exp24_inner_partition(
    device const uint*            src          [[buffer(0)]],
    device uint*                  dst          [[buffer(1)]],
    device atomic_uint*           tile_status  [[buffer(2)]],
    device const E24BucketDesc*   bucket_descs [[buffer(3)]],
    constant E24BatchParams&      params       [[buffer(4)]],
    device const uint*            bucket_hist  [[buffer(5)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint shift = params.shift;
    uint effective_gid = gid + params.tile_offset;

    // ── Find which bucket this TG belongs to ──────────────────────
    // Search only within this batch's buckets
    uint bucket_id = params.batch_start;
    uint batch_end = params.batch_start + params.batch_count;
    for (uint b = params.batch_start + 1u; b < batch_end; b++) {
        if (bucket_descs[b].tile_base <= effective_gid) {
            bucket_id = b;
        }
    }

    E24BucketDesc desc = bucket_descs[bucket_id];
    uint tile_in_bucket = effective_gid - desc.tile_base;
    if (tile_in_bucket >= desc.tile_count) return;

    uint n = desc.count;
    uint offset = desc.offset;
    uint base = offset + tile_in_bucket * E24_TILE_SIZE;
    uint end = offset + n;

    // ── TG Memory (20 KB total) ──────────────────────────────────
    threadgroup atomic_uint sg_hist_or_rank[E24_NUM_SGS * E24_NUM_BINS];
    threadgroup uint sg_prefix[E24_NUM_SGS * E24_NUM_BINS];
    threadgroup uint tile_hist[E24_NUM_BINS];
    threadgroup uint exclusive_pfx[E24_NUM_BINS];

    // ── Phase 1: Load (SG-contiguous layout) ─────────────────────
    uint mk[E24_ELEMS];
    uint md[E24_ELEMS];
    bool mv[E24_ELEMS];
    for (int e = 0; e < (int)E24_ELEMS; e++) {
        uint idx = base + simd_id * (E24_ELEMS * 32u) + (uint)e * 32u + simd_lane;
        mv[e] = idx < end;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG atomic histogram ─────────────────────────
    for (uint i = lid; i < E24_NUM_SGS * E24_NUM_BINS; i += E24_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)E24_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * E24_NUM_BINS + md[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Tile histogram + cross-SG prefix ──────────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < E24_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * E24_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * E24_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Publish AGGREGATE (use effective_gid) ───────────
    {
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT)
                    | (tile_hist[lid] & VALUE_MASK);
        atomic_store_explicit(&tile_status[effective_gid * E24_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Decoupled lookback (WITHIN bucket only) ─────────
    {
        uint lk_running = 0u;
        if (tile_in_bucket > 0u) {
            int look = (int)effective_gid - 1;
            int bucket_start_tile = (int)desc.tile_base;
            while (look >= bucket_start_tile) {
                atomic_thread_fence(mem_flags::mem_device,
                                    memory_order_seq_cst,
                                    thread_scope_device);
                uint val = atomic_load_explicit(
                    &tile_status[(uint)look * E24_NUM_BINS + lid],
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
        atomic_store_explicit(&tile_status[effective_gid * E24_NUM_BINS + lid],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Per-SG atomic ranking + scatter ─────────────────
    for (uint i = lid; i < E24_NUM_SGS * E24_NUM_BINS; i += E24_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)E24_ELEMS; e++) {
        if (mv[e]) {
            uint d = md[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * E24_NUM_BINS + d],
                1u, memory_order_relaxed);
            uint gp = offset
                     + bucket_hist[bucket_id * E24_NUM_BINS + d]
                     + exclusive_pfx[d]
                     + sg_prefix[simd_id * E24_NUM_BINS + d]
                     + within_sg;
            dst[gp] = mk[e];
        }
    }
}
