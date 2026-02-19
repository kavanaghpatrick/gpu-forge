#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 12: Cross-TG Coherence Strategy Benchmark
//
// Exp9 proved volatile doesn't fix cross-TG stale reads. This experiment
// tests Metal 3.2's official coherence mechanisms:
//
// Version A: Multi-encoder baseline (4 encoders in 1 cmdbuf, known correct)
// Version B: Single dispatch + coherent(device) buffers + atomic_thread_fence
// Version C: Single dispatch + atomic_thread_fence only (no coherent)
// Version D: Single dispatch + coherent(device) buffers only (no fence)
//
// All versions use SIMD rank from exp10 for maximum throughput.
// ══════════════════════════════════════════════════════════════════════

#define EXP12_NUM_BINS 256u
#define EXP12_NUM_SGS 8u   // 256 threads / 32 per simdgroup

struct Exp12PassParams {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint num_tgs;
    uint counter_base;
    uint ts_offset;
};

struct Exp12Params {
    uint element_count;
    uint num_tiles;
    uint num_tgs;
    uint num_passes;
    uint mode;  // 0 = use device-scope fence, 1 = no fence
};

// ── Global sync WITHOUT device-scope fence ──────────────────────────
inline void exp12_sync(device atomic_uint* counter,
                       uint num_tgs, uint lid)
{
    threadgroup_barrier(mem_flags::mem_device);
    if (lid == 0u) {
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        while (atomic_load_explicit(counter, memory_order_relaxed) < num_tgs) {}
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// ── Global sync WITH device-scope fence (Metal 3.2+) ───────────────
// Adds atomic_thread_fence at device scope on both sides of the barrier
// to ensure scatter writes are visible across ALL threadgroups.
inline void exp12_sync_fenced(device atomic_uint* counter,
                              uint num_tgs, uint lid)
{
    // Flush this TG's device writes
    threadgroup_barrier(mem_flags::mem_device);
    // Device-scope fence: push writes to device coherence point
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);

    if (lid == 0u) {
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        while (atomic_load_explicit(counter, memory_order_relaxed) < num_tgs) {}
    }

    // Device-scope fence: pull other TGs' writes from coherence point
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
}

// ═══════════════════════════════════════════════════════════════════
// Version A: Per-pass kernel with SIMD rank (for multi-encoder dispatch)
// Copied from exp10_simd_pass with renamed params struct.
// ═══════════════════════════════════════════════════════════════════

kernel void exp12_pass(
    device const uint* src          [[buffer(0)]],
    device uint* dst                [[buffer(1)]],
    device atomic_uint* tile_status [[buffer(2)]],
    device atomic_uint* counters    [[buffer(3)]],
    constant Exp12PassParams& params [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint num_tiles = params.num_tiles;
    uint num_tgs   = params.num_tgs;
    uint shift     = params.shift;
    uint ts_off    = params.ts_offset;

    device atomic_uint* ws1 = &counters[params.counter_base];
    device atomic_uint* ws2 = &counters[params.counter_base + 1u];
    device atomic_uint* barrier_ctr = &counters[params.counter_base + 2u];

    threadgroup atomic_uint local_hist[EXP12_NUM_BINS];
    threadgroup uint shared_digits[TILE_SIZE];
    threadgroup uint shared_tile_id;
    threadgroup uint global_bin_start[EXP12_NUM_BINS];
    threadgroup uint sg_totals[8];
    threadgroup atomic_uint sg_hist[EXP12_NUM_SGS * EXP12_NUM_BINS];

    // ═══ Loop 1: Histogram + Decoupled Lookback ═══
    while (true) {
        if (lid == 0u) {
            shared_tile_id = atomic_fetch_add_explicit(ws1, 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tile_id = shared_tile_id;
        if (tile_id >= num_tiles) break;

        atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint base = tile_id * TILE_SIZE;
        bool valid = (base + lid) < params.element_count;

        if (valid) {
            uint val = src[base + lid];
            uint digit = (val >> shift) & 0xFFu;
            atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint my_count = atomic_load_explicit(&local_hist[lid], memory_order_relaxed);
        uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT) | (my_count & VALUE_MASK);
        atomic_store_explicit(&tile_status[ts_off + tile_id * EXP12_NUM_BINS + lid],
                              packed_agg, memory_order_relaxed);

        if (tile_id == 0u) {
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (my_count & VALUE_MASK);
            atomic_store_explicit(&tile_status[ts_off + lid], packed_pfx, memory_order_relaxed);
        } else {
            uint running = 0u;
            int lb = (int)tile_id - 1;
            while (lb >= 0) {
                uint status = atomic_load_explicit(
                    &tile_status[ts_off + lb * EXP12_NUM_BINS + lid], memory_order_relaxed);
                uint flag  = status >> FLAG_SHIFT;
                uint value = status & VALUE_MASK;
                if (flag == FLAG_PREFIX)    { running += value; break; }
                else if (flag == FLAG_AGGREGATE) { running += value; lb--; }
            }
            uint inclusive = running + my_count;
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK);
            atomic_store_explicit(&tile_status[ts_off + tile_id * EXP12_NUM_BINS + lid],
                                  packed_pfx, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    exp12_sync(barrier_ctr, num_tgs, lid);

    // ═══ Compute global_bin_start ═══
    uint last_status = atomic_load_explicit(
        &tile_status[ts_off + (num_tiles - 1u) * EXP12_NUM_BINS + lid], memory_order_relaxed);
    uint global_total_val = last_status & VALUE_MASK;

    uint simd_scan = simd_prefix_inclusive_sum(global_total_val);
    if (simd_lane == 31u) sg_totals[simd_id] = simd_scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u && simd_lane < 8u) {
        uint sg = sg_totals[simd_lane];
        uint sg_scan = simd_prefix_inclusive_sum(sg);
        sg_totals[simd_lane] = sg_scan;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint inc = simd_scan;
    if (simd_id > 0u) inc += sg_totals[simd_id - 1u];
    global_bin_start[lid] = inc - global_total_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══ Loop 2: Scatter (SIMD Rank) ═══
    while (true) {
        if (lid == 0u) {
            shared_tile_id = atomic_fetch_add_explicit(ws2, 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tile_id = shared_tile_id;
        if (tile_id >= num_tiles) break;

        uint base = tile_id * TILE_SIZE;
        bool valid = (base + lid) < params.element_count;
        uint element = 0u;
        uint digit = 0u;

        if (valid) {
            element = src[base + lid];
            digit = (element >> shift) & 0xFFu;
        }

        for (uint i = lid; i < EXP12_NUM_SGS * EXP12_NUM_BINS; i += TILE_SIZE) {
            atomic_store_explicit(&sg_hist[i], 0u, memory_order_relaxed);
        }
        shared_digits[lid] = valid ? digit : 0xFFFFu;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            atomic_fetch_add_explicit(
                &sg_hist[simd_id * EXP12_NUM_BINS + digit], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            uint within_rank = 0u;
            uint sg_base = simd_id * 32u;
            for (uint j = 0u; j < simd_lane; j++) {
                within_rank += (shared_digits[sg_base + j] == digit) ? 1u : 0u;
            }

            uint cross_rank = 0u;
            for (uint s = 0u; s < simd_id; s++) {
                cross_rank += atomic_load_explicit(
                    &sg_hist[s * EXP12_NUM_BINS + digit], memory_order_relaxed);
            }

            uint rank = cross_rank + within_rank;

            uint bin_excl;
            if (tile_id == 0u) {
                bin_excl = 0u;
            } else {
                uint prev = atomic_load_explicit(
                    &tile_status[ts_off + (tile_id - 1u) * EXP12_NUM_BINS + digit],
                    memory_order_relaxed);
                bin_excl = prev & VALUE_MASK;
            }

            uint pos = global_bin_start[digit] + bin_excl + rank;
            dst[pos] = element;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}


// ═══════════════════════════════════════════════════════════════════
// MEGASORT BODY TEMPLATE
//
// Generates single-dispatch megasort kernels. The body is identical
// across coherent/plain variants — only buffer types and fence
// behavior differ. Uses params.mode to select fence vs no-fence.
// ═══════════════════════════════════════════════════════════════════

// Preprocessor template: instantiate megasort kernel with given buffer type
// BUF_QUAL: buffer qualifier (e.g., "device" or "device coherent(device)")
// KERNEL_NAME: kernel function name
#define EXP12_MEGASORT_BODY \
    uint num_tiles   = params.num_tiles; \
    uint num_tgs     = params.num_tgs; \
    uint num_passes  = params.num_passes; \
    uint elem_count  = params.element_count; \
    bool use_fence   = (params.mode == 0u); \
    \
    uint ts_section = num_tiles * EXP12_NUM_BINS; \
    \
    threadgroup atomic_uint local_hist[EXP12_NUM_BINS]; \
    threadgroup uint shared_digits[TILE_SIZE]; \
    threadgroup uint shared_tile_id; \
    threadgroup uint global_bin_start[EXP12_NUM_BINS]; \
    threadgroup uint sg_totals[8]; \
    threadgroup atomic_uint sg_hist[EXP12_NUM_SGS * EXP12_NUM_BINS]; \
    \
    uint barrier_idx = 8u; \
    \
    for (uint pass = 0u; pass < num_passes; pass++) { \
        uint shift = pass * 8u; \
        uint ts_off = pass * ts_section; \
        \
        device atomic_uint* ws1 = &counters[pass * 2u]; \
        device atomic_uint* ws2 = &counters[pass * 2u + 1u]; \
        \
        /* Ping-pong: even=a→b, odd=b→a */ \
        auto src = (pass % 2u == 0u) ? buf_a : buf_b; \
        auto dst = (pass % 2u == 0u) ? buf_b : buf_a; \
        \
        /* ═══ Loop 1: Histogram + Decoupled Lookback ═══ */ \
        while (true) { \
            if (lid == 0u) { \
                shared_tile_id = atomic_fetch_add_explicit(ws1, 1u, memory_order_relaxed); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            uint tile_id = shared_tile_id; \
            if (tile_id >= num_tiles) break; \
            \
            atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed); \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            uint base = tile_id * TILE_SIZE; \
            bool valid = (base + lid) < elem_count; \
            \
            if (valid) { \
                uint val = src[base + lid]; \
                uint digit = (val >> shift) & 0xFFu; \
                atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            uint my_count = atomic_load_explicit(&local_hist[lid], memory_order_relaxed); \
            \
            uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT) | (my_count & VALUE_MASK); \
            atomic_store_explicit( \
                &tile_status[ts_off + tile_id * EXP12_NUM_BINS + lid], \
                packed_agg, memory_order_relaxed); \
            \
            if (tile_id == 0u) { \
                uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (my_count & VALUE_MASK); \
                atomic_store_explicit(&tile_status[ts_off + lid], packed_pfx, memory_order_relaxed); \
            } else { \
                uint running = 0u; \
                int lb = (int)tile_id - 1; \
                while (lb >= 0) { \
                    uint status = atomic_load_explicit( \
                        &tile_status[ts_off + lb * EXP12_NUM_BINS + lid], memory_order_relaxed); \
                    uint flag  = status >> FLAG_SHIFT; \
                    uint value = status & VALUE_MASK; \
                    if (flag == FLAG_PREFIX)    { running += value; break; } \
                    else if (flag == FLAG_AGGREGATE) { running += value; lb--; } \
                } \
                uint inclusive = running + my_count; \
                uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK); \
                atomic_store_explicit( \
                    &tile_status[ts_off + tile_id * EXP12_NUM_BINS + lid], \
                    packed_pfx, memory_order_relaxed); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        \
        /* ═══ Sync A: All tiles histogrammed ═══ */ \
        if (use_fence) { exp12_sync_fenced(&counters[barrier_idx], num_tgs, lid); } \
        else           { exp12_sync(&counters[barrier_idx], num_tgs, lid); } \
        barrier_idx++; \
        \
        /* ═══ Compute global_bin_start ═══ */ \
        uint last_status = atomic_load_explicit( \
            &tile_status[ts_off + (num_tiles - 1u) * EXP12_NUM_BINS + lid], memory_order_relaxed); \
        uint global_total_val = last_status & VALUE_MASK; \
        \
        uint simd_scan = simd_prefix_inclusive_sum(global_total_val); \
        if (simd_lane == 31u) sg_totals[simd_id] = simd_scan; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        if (simd_id == 0u && simd_lane < 8u) { \
            uint sg = sg_totals[simd_lane]; \
            uint sg_scan = simd_prefix_inclusive_sum(sg); \
            sg_totals[simd_lane] = sg_scan; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        uint inc_val = simd_scan; \
        if (simd_id > 0u) inc_val += sg_totals[simd_id - 1u]; \
        global_bin_start[lid] = inc_val - global_total_val; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        /* ═══ Loop 2: Scatter (SIMD Rank) ═══ */ \
        while (true) { \
            if (lid == 0u) { \
                shared_tile_id = atomic_fetch_add_explicit(ws2, 1u, memory_order_relaxed); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            uint tile_id = shared_tile_id; \
            if (tile_id >= num_tiles) break; \
            \
            uint base = tile_id * TILE_SIZE; \
            bool valid = (base + lid) < elem_count; \
            uint element = 0u; \
            uint digit = 0u; \
            \
            if (valid) { \
                element = src[base + lid]; \
                digit = (element >> shift) & 0xFFu; \
            } \
            \
            for (uint i = lid; i < EXP12_NUM_SGS * EXP12_NUM_BINS; i += TILE_SIZE) { \
                atomic_store_explicit(&sg_hist[i], 0u, memory_order_relaxed); \
            } \
            shared_digits[lid] = valid ? digit : 0xFFFFu; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            if (valid) { \
                atomic_fetch_add_explicit( \
                    &sg_hist[simd_id * EXP12_NUM_BINS + digit], 1u, memory_order_relaxed); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            if (valid) { \
                uint within_rank = 0u; \
                uint sg_base = simd_id * 32u; \
                for (uint j = 0u; j < simd_lane; j++) { \
                    within_rank += (shared_digits[sg_base + j] == digit) ? 1u : 0u; \
                } \
                \
                uint cross_rank = 0u; \
                for (uint s = 0u; s < simd_id; s++) { \
                    cross_rank += atomic_load_explicit( \
                        &sg_hist[s * EXP12_NUM_BINS + digit], memory_order_relaxed); \
                } \
                \
                uint rank = cross_rank + within_rank; \
                \
                uint bin_excl; \
                if (tile_id == 0u) { \
                    bin_excl = 0u; \
                } else { \
                    uint prev = atomic_load_explicit( \
                        &tile_status[ts_off + (tile_id - 1u) * EXP12_NUM_BINS + digit], \
                        memory_order_relaxed); \
                    bin_excl = prev & VALUE_MASK; \
                } \
                \
                uint pos = global_bin_start[digit] + bin_excl + rank; \
                dst[pos] = element; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        \
        /* Between passes: sync + clear */ \
        if (pass < num_passes - 1u) { \
            if (use_fence) { exp12_sync_fenced(&counters[barrier_idx], num_tgs, lid); } \
            else           { exp12_sync(&counters[barrier_idx], num_tgs, lid); } \
            barrier_idx++; \
            \
            uint next_ts_off = (pass + 1u) * ts_section; \
            for (uint i = gid * TILE_SIZE + lid; i < ts_section; \
                 i += num_tgs * TILE_SIZE) { \
                atomic_store_explicit(&tile_status[next_ts_off + i], 0u, memory_order_relaxed); \
            } \
            \
            if (gid == 0u && lid < 2u) { \
                atomic_store_explicit(&counters[(pass + 1u) * 2u + lid], 0u, memory_order_relaxed); \
            } \
            \
            if (use_fence) { exp12_sync_fenced(&counters[barrier_idx], num_tgs, lid); } \
            else           { exp12_sync(&counters[barrier_idx], num_tgs, lid); } \
            barrier_idx++; \
        } \
    }


// ═══════════════════════════════════════════════════════════════════
// Version B & D: Megasort with coherent(device) buffers
//
// coherent(device) makes non-atomic device writes visible across all
// threadgroups when synchronized with atomic_thread_fence at device scope.
//
// params.mode controls fence behavior:
//   mode=0: coherent + fence (Version B)
//   mode=1: coherent only, no fence (Version D)
// ═══════════════════════════════════════════════════════════════════

kernel void exp12_megasort_coherent(
    device coherent(device) uint* buf_a  [[buffer(0)]],
    device coherent(device) uint* buf_b  [[buffer(1)]],
    device atomic_uint* tile_status      [[buffer(2)]],
    device atomic_uint* counters         [[buffer(3)]],
    constant Exp12Params& params         [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    EXP12_MEGASORT_BODY
}


// ═══════════════════════════════════════════════════════════════════
// Version C: Megasort with plain buffers + fence only
//
// Tests whether atomic_thread_fence alone (without coherent qualifier)
// is sufficient for cross-TG visibility. Always uses fenced sync.
// ═══════════════════════════════════════════════════════════════════

kernel void exp12_megasort_fence(
    device uint* buf_a                   [[buffer(0)]],
    device uint* buf_b                   [[buffer(1)]],
    device atomic_uint* tile_status      [[buffer(2)]],
    device atomic_uint* counters         [[buffer(3)]],
    constant Exp12Params& params         [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    EXP12_MEGASORT_BODY
}
