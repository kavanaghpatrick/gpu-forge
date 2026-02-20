#include <metal_stdlib>
using namespace metal;
#include "types.h"

// Constants from exp16_8bit.metal
#define EXP16_NUM_BINS   256u
#define EXP16_NUM_SGS    8u
#define EXP16_TILE_SIZE  4096u
#define EXP16_ELEMS      16u
#define EXP16_NUM_PASSES 4u
#define EXP16_THREADS    256u

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Combined Histogram — 4-pass histogram in one read
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

    uint keys[EXP16_ELEMS];
    bool valid[EXP16_ELEMS];
    for (uint e = 0u; e < EXP16_ELEMS; e++) {
        uint idx = base + e * EXP16_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    threadgroup atomic_uint sg_counts[EXP16_NUM_SGS * EXP16_NUM_BINS]; // 8 KB

    for (uint p = 0u; p < EXP16_NUM_PASSES; p++) {

        for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
            atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint e = 0u; e < EXP16_ELEMS; e++) {
            if (valid[e]) {
                uint digit = (keys[e] >> (p * 8u)) & 0xFFu;
                atomic_fetch_add_explicit(
                    &sg_counts[simd_id * EXP16_NUM_BINS + digit],
                    1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
// ═══════════════════════════════════════════════════════════════════

kernel void exp16_global_prefix(
    device uint* global_hist [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    if (simd_id < EXP16_NUM_PASSES) {
        uint p = simd_id;
        uint running = 0u;

        for (uint chunk = 0u; chunk < 8u; chunk++) {
            uint bin = chunk * 32u + simd_lane;
            uint val = global_hist[p * EXP16_NUM_BINS + bin];
            uint prefix = simd_prefix_exclusive_sum(val) + running;
            global_hist[p * EXP16_NUM_BINS + bin] = prefix;
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
// Requires Metal 3.2 for atomic_thread_fence with thread_scope_device
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

    threadgroup atomic_uint sg_hist_or_rank[EXP16_NUM_SGS * EXP16_NUM_BINS]; // 8 KB
    threadgroup uint sg_prefix[EXP16_NUM_SGS * EXP16_NUM_BINS];             // 8 KB
    threadgroup uint tile_hist[EXP16_NUM_BINS];                              // 1 KB
    threadgroup uint exclusive_pfx[EXP16_NUM_BINS];                          // 1 KB

    uint tile_id = gid;
    uint base = tile_id * EXP16_TILE_SIZE;

    // Phase 1: Load elements
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

    // Phase 5: Per-SG ranking + scatter
    for (uint i = lid; i < EXP16_NUM_SGS * EXP16_NUM_BINS; i += EXP16_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
