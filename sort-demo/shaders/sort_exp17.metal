#include <metal_stdlib>
using namespace metal;
#include "types.h"

// Constants from exp17_hybrid.metal
#define EXP17_NUM_BINS  256u
#define EXP17_TILE_SIZE 4096u
#define EXP17_ELEMS     16u
#define EXP17_THREADS   256u
#define EXP17_NUM_SGS   8u
#define EXP17_MAX_TPB   17u

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: MSD Histogram — single-pass, bits[24:31]
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

    uint keys[EXP17_ELEMS];
    bool valid[EXP17_ELEMS];
    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        uint idx = base + e * EXP17_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    threadgroup atomic_uint sg_counts[EXP17_NUM_SGS * EXP17_NUM_BINS]; // 8 KB

    for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        if (valid[e]) {
            uint digit = (keys[e] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * EXP17_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
// Kernel 2: MSD Prep — prefix sum + bucket descs (1 TG, 256 threads)
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_msd_prep(
    device const uint*     global_hist  [[buffer(0)]],
    device uint*           counters     [[buffer(1)]],
    device BucketDesc*     bucket_descs [[buffer(2)]],
    constant uint&         tile_size    [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup uint prefix[EXP17_NUM_BINS];
    threadgroup uint running_offset;

    if (lid == 0u) {
        uint sum = 0u;
        for (uint i = 0u; i < EXP17_NUM_BINS; i++) {
            prefix[i] = sum;
            sum += global_hist[i];
        }
        running_offset = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint count = global_hist[lid];
    uint offset = prefix[lid];
    counters[lid] = offset;

    uint tc = (count + tile_size - 1u) / tile_size;
    bucket_descs[lid] = BucketDesc{offset, count, tc, 0u};
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: MSD Atomic Scatter — scatter elements by MSD digit
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_msd_atomic_scatter(
    device const uint*     src       [[buffer(0)]],
    device uint*           dst       [[buffer(1)]],
    device atomic_uint*    counters  [[buffer(2)]],
    constant Exp17Params&  params    [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n     = params.element_count;
    uint shift = params.shift;
    uint base  = gid * EXP17_TILE_SIZE;

    threadgroup atomic_uint sg_hist_or_rank[EXP17_NUM_SGS * EXP17_NUM_BINS]; // 8 KB
    threadgroup uint sg_prefix[EXP17_NUM_SGS * EXP17_NUM_BINS];             // 8 KB
    threadgroup uint tile_hist[EXP17_NUM_BINS];                              // 1 KB
    threadgroup uint tile_base[EXP17_NUM_BINS];                              // 1 KB

    uint mk[EXP17_ELEMS];
    uint md[EXP17_ELEMS];
    bool mv[EXP17_ELEMS];
    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        uint idx = base + e * EXP17_THREADS + lid;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP17_NUM_BINS + md[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP17_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * EXP17_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * EXP17_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        tile_base[lid] = atomic_fetch_add_explicit(
            &counters[lid], tile_hist[lid], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP17_ELEMS; e++) {
        if (mv[e]) {
            uint d = md[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * EXP17_NUM_BINS + d],
                1u, memory_order_relaxed);
            uint gp = tile_base[d]
                     + sg_prefix[simd_id * EXP17_NUM_BINS + d]
                     + within_sg;
            dst[gp] = mk[e];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: Inner Precompute Histograms — 3-pass per-bucket histograms
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_inner_precompute_hists(
    device const uint*        src          [[buffer(0)]],
    device uint*              inner_hists  [[buffer(1)]],
    device const BucketDesc*  bucket_descs [[buffer(2)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    BucketDesc desc = bucket_descs[gid];
    if (desc.count == 0u) return;

    threadgroup atomic_uint sg_c[3u * EXP17_NUM_SGS * EXP17_NUM_BINS];

    for (uint i = lid; i < 3u * EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
        atomic_store_explicit(&sg_c[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0u; t < desc.tile_count; t++) {
        for (uint e = 0u; e < EXP17_ELEMS; e++) {
            uint local_idx = t * EXP17_TILE_SIZE
                           + simd_id * (EXP17_ELEMS * 32u) + e * 32u + simd_lane;
            if (local_idx < desc.count) {
                uint val = src[desc.offset + local_idx];
                uint sg_base = simd_id * EXP17_NUM_BINS;
                atomic_fetch_add_explicit(
                    &sg_c[0u * EXP17_NUM_SGS * EXP17_NUM_BINS + sg_base + (val & 0xFFu)],
                    1u, memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &sg_c[1u * EXP17_NUM_SGS * EXP17_NUM_BINS + sg_base + ((val >> 8u) & 0xFFu)],
                    1u, memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &sg_c[2u * EXP17_NUM_SGS * EXP17_NUM_BINS + sg_base + ((val >> 16u) & 0xFFu)],
                    1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_base = gid * 3u * EXP17_NUM_BINS;
    for (uint p = 0u; p < 3u; p++) {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP17_NUM_SGS; sg++) {
            total += atomic_load_explicit(
                &sg_c[p * EXP17_NUM_SGS * EXP17_NUM_BINS + sg * EXP17_NUM_BINS + lid],
                memory_order_relaxed);
        }
        inner_hists[out_base + p * EXP17_NUM_BINS + lid] = total;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 5: Fused Inner Sort V3 — 3 passes in 1 dispatch
// ═══════════════════════════════════════════════════════════════════

kernel void exp17_inner_fused_v3(
    device uint*                buf_a         [[buffer(0)]],
    device uint*                buf_b         [[buffer(1)]],
    device const BucketDesc*    bucket_descs  [[buffer(2)]],
    device const uint*          inner_hists   [[buffer(3)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    BucketDesc desc = bucket_descs[gid];
    if (desc.count == 0u) return;

    uint tile_count = desc.tile_count;

    threadgroup atomic_uint sg_ctr[EXP17_NUM_SGS * EXP17_NUM_BINS]; // 8 KB
    threadgroup uint sg_pfx[EXP17_NUM_SGS * EXP17_NUM_BINS];        // 8 KB
    threadgroup uint bkt_hist[EXP17_NUM_BINS];                       // 1 KB
    threadgroup uint glb_pfx[EXP17_NUM_BINS];                        // 1 KB
    threadgroup uint run_pfx[EXP17_NUM_BINS];                        // 1 KB
    threadgroup uint chk_tot[8];                                      // 32 B

    for (uint pass = 0u; pass < 3u; pass++) {
        uint shift = pass * 8u;

        device uint* src = (pass % 2u == 0u) ? buf_b : buf_a;
        device uint* dst = (pass % 2u == 0u) ? buf_a : buf_b;

        uint hist_base = gid * 3u * EXP17_NUM_BINS + pass * EXP17_NUM_BINS;
        bkt_hist[lid] = inner_hists[hist_base + lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint chunk = lid / 32u;
            uint lane = lid % 32u;
            uint val = bkt_hist[lid];
            uint prefix = simd_prefix_exclusive_sum(val);

            if (lane == 31u) chk_tot[chunk] = prefix + val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lid == 0u) {
                uint running = 0u;
                for (uint c = 0u; c < 8u; c++) {
                    uint ct = chk_tot[c];
                    chk_tot[c] = running;
                    running += ct;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            glb_pfx[lid] = prefix + chk_tot[chunk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        run_pfx[lid] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0u; t < tile_count; t++) {
            uint tile_base = desc.offset + t * EXP17_TILE_SIZE;

            uint keys[EXP17_ELEMS];
            uint digits[EXP17_ELEMS];
            bool valid[EXP17_ELEMS];
            for (uint e = 0u; e < EXP17_ELEMS; e++) {
                uint local_idx = t * EXP17_TILE_SIZE
                               + simd_id * (EXP17_ELEMS * 32u) + e * 32u + simd_lane;
                valid[e] = local_idx < desc.count;
                uint idx = tile_base + simd_id * (EXP17_ELEMS * 32u) + e * 32u + simd_lane;
                keys[e] = valid[e] ? src[idx] : 0xFFFFFFFFu;
                digits[e] = valid[e] ? ((keys[e] >> shift) & 0xFFu) : 0xFFu;
            }

            for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
                atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint e = 0u; e < EXP17_ELEMS; e++) {
                if (valid[e]) {
                    atomic_fetch_add_explicit(
                        &sg_ctr[simd_id * EXP17_NUM_BINS + digits[e]],
                        1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint tile_digit_count;
            {
                uint total = 0u;
                for (uint sg = 0u; sg < EXP17_NUM_SGS; sg++) {
                    uint c = atomic_load_explicit(
                        &sg_ctr[sg * EXP17_NUM_BINS + lid],
                        memory_order_relaxed);
                    sg_pfx[sg * EXP17_NUM_BINS + lid] = total;
                    total += c;
                }
                tile_digit_count = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = lid; i < EXP17_NUM_SGS * EXP17_NUM_BINS; i += EXP17_THREADS) {
                atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint e = 0u; e < EXP17_ELEMS; e++) {
                if (valid[e]) {
                    uint d = digits[e];
                    uint within_sg = atomic_fetch_add_explicit(
                        &sg_ctr[simd_id * EXP17_NUM_BINS + d],
                        1u, memory_order_relaxed);
                    uint dst_idx = desc.offset
                                 + glb_pfx[d]
                                 + run_pfx[d]
                                 + sg_pfx[simd_id * EXP17_NUM_BINS + d]
                                 + within_sg;
                    dst[dst_idx] = keys[e];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            run_pfx[lid] += tile_digit_count;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup_barrier(mem_flags::mem_device);
    }
}
