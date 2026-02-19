#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 14: Multi-Dispatch Radix Sort (Phase 1)
//
// 3 kernels per pass: histogram → prefix_scan → scatter
// Key optimization: local sort in TG shared memory for coalesced scatter
// Phase 1: scalar loads, serial rank (TILE_SIZE=256, 1 elem/thread)
// ══════════════════════════════════════════════════════════════════════

#define EXP14_TILE_SIZE   256u
#define EXP14_NUM_BINS    256u
#define EXP14_NUM_SGS     8u

struct Exp14Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint pass;
};

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Histogram — one TG per tile
// Computes per-tile digit histogram, writes to tile_histograms[tile][bin]
// ═══════════════════════════════════════════════════════════════════

kernel void exp14_histogram(
    device const uint*     src              [[buffer(0)]],
    device uint*           tile_histograms  [[buffer(1)]],
    constant Exp14Params&  params           [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint elem_count = params.element_count;
    uint base = tile_id * EXP14_TILE_SIZE;

    // Local histogram via shared atomics
    threadgroup atomic_uint local_hist[EXP14_NUM_BINS];
    atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (base + lid < elem_count) {
        uint key = src[base + lid];
        uint digit = (key >> shift) & 0xFFu;
        atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write tile histogram to device memory
    tile_histograms[tile_id * EXP14_NUM_BINS + lid] =
        atomic_load_explicit(&local_hist[lid], memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Prefix Scan — 256 TGs (one per bin)
// All 256 threads cooperate: process tiles in chunks of 256.
// Parallel SIMD prefix sum within each chunk, serial across chunks.
// 16 chunks for 1M elements (3907 tiles) vs 3907 serial iterations.
// ═══════════════════════════════════════════════════════════════════

kernel void exp14_prefix_scan(
    device uint*           tile_histograms  [[buffer(0)]],
    device uint*           global_histogram [[buffer(1)]],
    constant Exp14Params&  params           [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint bin = gid;  // 256 TGs, one per bin
    uint num_tiles = params.num_tiles;

    threadgroup uint sg_totals[8];
    threadgroup uint chunk_total;

    uint running = 0u;

    for (uint chunk_start = 0u; chunk_start < num_tiles; chunk_start += EXP14_TILE_SIZE) {
        uint tile_idx = chunk_start + lid;
        uint val = (tile_idx < num_tiles)
            ? tile_histograms[tile_idx * EXP14_NUM_BINS + bin]
            : 0u;

        // Parallel exclusive prefix sum across 256 threads (2-level SIMD)
        uint simd_pfx = simd_prefix_exclusive_sum(val);

        if (simd_lane == 31u) {
            sg_totals[simd_id] = simd_pfx + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u && simd_lane < 8u) {
            uint t = sg_totals[simd_lane];
            uint sg_excl = simd_prefix_exclusive_sum(t);
            sg_totals[simd_lane] = sg_excl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint local_prefix = simd_pfx;
        if (simd_id > 0u) {
            local_prefix += sg_totals[simd_id];
        }
        uint global_prefix = running + local_prefix;

        // Write exclusive prefix back to tile_histograms
        if (tile_idx < num_tiles) {
            tile_histograms[tile_idx * EXP14_NUM_BINS + bin] = global_prefix;
        }

        // Last thread computes chunk total for next iteration
        if (lid == EXP14_TILE_SIZE - 1u) {
            chunk_total = global_prefix + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running = chunk_total;
    }

    // Thread 0 writes the global total
    if (lid == 0u) {
        global_histogram[bin] = running;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2b: Global Prefix — 1 TG of 256 threads
// Converts global_histogram from counts to exclusive prefix sum
// ═══════════════════════════════════════════════════════════════════

kernel void exp14_global_prefix(
    device uint*           global_histogram [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint val = global_histogram[lid];
    uint simd_pfx = simd_prefix_exclusive_sum(val);

    threadgroup uint sg_totals[8];
    if (simd_lane == 31u) {
        sg_totals[simd_id] = simd_pfx + val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u && simd_lane < 8u) {
        uint t = sg_totals[simd_lane];
        uint sg_excl = simd_prefix_exclusive_sum(t);
        sg_totals[simd_lane] = sg_excl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint result = simd_pfx;
    if (simd_id > 0u) {
        result += sg_totals[simd_id];
    }
    global_histogram[lid] = result;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Scatter — one TG per tile
// Local sort in shared memory → coalesced global writes
// Phase 1: serial O(32) rank (ballot in Phase 2)
// ═══════════════════════════════════════════════════════════════════

kernel void exp14_scatter(
    device const uint*     src              [[buffer(0)]],
    device uint*           dst              [[buffer(1)]],
    device const uint*     tile_histograms  [[buffer(2)]],
    device const uint*     global_histogram [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint elem_count = params.element_count;
    uint base = tile_id * EXP14_TILE_SIZE;

    // ── Shared memory ──────────────────────────────────────
    threadgroup atomic_uint sg_hist[EXP14_NUM_SGS * EXP14_NUM_BINS];  // 8 KB
    threadgroup uint shared_digits[EXP14_TILE_SIZE];                   // 1 KB
    threadgroup uint tile_hist[EXP14_NUM_BINS];                        // 1 KB
    threadgroup uint bin_start[EXP14_NUM_BINS];                        // 1 KB
    threadgroup uint local_keys[EXP14_TILE_SIZE];                      // 1 KB
    threadgroup uint local_offsets[EXP14_NUM_BINS];                    // 1 KB
    threadgroup uint sg_totals_pfx[8];                                 // 32 B
    // Total: ~13 KB

    // ── Phase 1: Load element + digit ──────────────────────
    bool valid = (base + lid) < elem_count;
    uint key = valid ? src[base + lid] : 0xFFFFFFFFu;
    uint digit = valid ? ((key >> shift) & 0xFFu) : 0xFFFFu;

    // Zero SG histograms
    for (uint i = lid; i < EXP14_NUM_SGS * EXP14_NUM_BINS; i += EXP14_TILE_SIZE) {
        atomic_store_explicit(&sg_hist[i], 0u, memory_order_relaxed);
    }
    shared_digits[lid] = digit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Build per-SG histogram ────────────────────
    if (valid) {
        atomic_fetch_add_explicit(
            &sg_hist[simd_id * EXP14_NUM_BINS + digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Reduce SG histograms → tile_hist ──────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP14_NUM_SGS; sg++) {
            total += atomic_load_explicit(
                &sg_hist[sg * EXP14_NUM_BINS + lid], memory_order_relaxed);
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Exclusive prefix sum → bin_start ──────────
    {
        uint val = tile_hist[lid];
        uint simd_pfx = simd_prefix_exclusive_sum(val);

        if (simd_lane == 31u) {
            sg_totals_pfx[simd_id] = simd_pfx + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u && simd_lane < 8u) {
            uint t = sg_totals_pfx[simd_lane];
            uint sg_excl = simd_prefix_exclusive_sum(t);
            sg_totals_pfx[simd_lane] = sg_excl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint offset = simd_pfx;
        if (simd_id > 0u) {
            offset += sg_totals_pfx[simd_id];
        }
        bin_start[lid] = offset;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Compute rank + local sort ─────────────────
    if (valid) {
        // Within-SG rank: serial O(32) scan
        uint within_sg_rank = 0u;
        uint sg_base = simd_id * 32u;
        for (uint j = 0u; j < simd_lane; j++) {
            within_sg_rank += (shared_digits[sg_base + j] == digit) ? 1u : 0u;
        }

        // Cross-SG offset
        uint cross_sg = 0u;
        for (uint s = 0u; s < simd_id; s++) {
            cross_sg += atomic_load_explicit(
                &sg_hist[s * EXP14_NUM_BINS + digit], memory_order_relaxed);
        }

        uint tile_rank = bin_start[digit] + cross_sg + within_sg_rank;
        local_keys[tile_rank] = key;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 6: Load global offsets ───────────────────────
    {
        uint tile_excl = tile_histograms[tile_id * EXP14_NUM_BINS + lid];
        uint global_excl = global_histogram[lid];
        local_offsets[lid] = global_excl + tile_excl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 7: Coalesced scatter to global memory ────────
    if (valid) {
        uint k = local_keys[lid];
        uint d = (k >> shift) & 0xFFu;
        uint within_bin_pos = lid - bin_start[d];
        uint global_pos = local_offsets[d] + within_bin_pos;
        dst[global_pos] = k;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3b: Direct Scatter — no local sort, random writes
// Diagnostic: isolate multi-dispatch overhead from local sort overhead
// ═══════════════════════════════════════════════════════════════════

kernel void exp14_scatter_direct(
    device const uint*     src              [[buffer(0)]],
    device uint*           dst              [[buffer(1)]],
    device const uint*     tile_histograms  [[buffer(2)]],
    device const uint*     global_histogram [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint elem_count = params.element_count;
    uint base = tile_id * EXP14_TILE_SIZE;

    threadgroup atomic_uint sg_hist[EXP14_NUM_SGS * EXP14_NUM_BINS];
    threadgroup uint shared_digits[EXP14_TILE_SIZE];

    bool valid = (base + lid) < elem_count;
    uint key = valid ? src[base + lid] : 0xFFFFFFFFu;
    uint digit = valid ? ((key >> shift) & 0xFFu) : 0xFFFFu;

    for (uint i = lid; i < EXP14_NUM_SGS * EXP14_NUM_BINS; i += EXP14_TILE_SIZE) {
        atomic_store_explicit(&sg_hist[i], 0u, memory_order_relaxed);
    }
    shared_digits[lid] = digit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        atomic_fetch_add_explicit(
            &sg_hist[simd_id * EXP14_NUM_BINS + digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        uint within_sg_rank = 0u;
        uint sg_base = simd_id * 32u;
        for (uint j = 0u; j < simd_lane; j++) {
            within_sg_rank += (shared_digits[sg_base + j] == digit) ? 1u : 0u;
        }

        uint cross_sg = 0u;
        for (uint s = 0u; s < simd_id; s++) {
            cross_sg += atomic_load_explicit(
                &sg_hist[s * EXP14_NUM_BINS + digit], memory_order_relaxed);
        }
        uint rank = cross_sg + within_sg_rank;

        uint tile_excl = tile_histograms[tile_id * EXP14_NUM_BINS + digit];
        uint global_excl = global_histogram[digit];
        dst[global_excl + tile_excl + rank] = key;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3c: Ballot-Rank Scatter — WLMS bit-decomposition via shuffle
// Replaces O(32) serial rank with O(log2(32))=5 shuffle-ballot ops
// Uses 8 ballots for 8-bit digit → match mask → popcount for rank
// ═══════════════════════════════════════════════════════════════════

// Compute ballot mask via butterfly shuffle-OR reduction (5 ops)
inline uint shuffle_ballot(bool pred, uint simd_lane) {
    uint my_bit = pred ? (1u << simd_lane) : 0u;
    my_bit |= simd_shuffle_xor(my_bit, 1u);
    my_bit |= simd_shuffle_xor(my_bit, 2u);
    my_bit |= simd_shuffle_xor(my_bit, 4u);
    my_bit |= simd_shuffle_xor(my_bit, 8u);
    my_bit |= simd_shuffle_xor(my_bit, 16u);
    return my_bit;
}

kernel void exp14_scatter_ballot(
    device const uint*     src              [[buffer(0)]],
    device uint*           dst              [[buffer(1)]],
    device const uint*     tile_histograms  [[buffer(2)]],
    device const uint*     global_histogram [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint elem_count = params.element_count;
    uint base = tile_id * EXP14_TILE_SIZE;

    // ── Shared memory ──────────────────────────────────────
    threadgroup uint sg_hist_ballot[EXP14_NUM_SGS * EXP14_NUM_BINS]; // 8 KB (non-atomic)
    threadgroup uint tile_hist[EXP14_NUM_BINS];                       // 1 KB
    threadgroup uint bin_start[EXP14_NUM_BINS];                       // 1 KB
    threadgroup uint local_keys[EXP14_TILE_SIZE];                     // 1 KB
    threadgroup uint local_offsets[EXP14_NUM_BINS];                   // 1 KB
    threadgroup uint sg_totals_pfx[8];                                // 32 B
    // Total: ~12 KB

    // ── Phase 1: Load element + digit ──────────────────────
    bool valid = (base + lid) < elem_count;
    uint key = valid ? src[base + lid] : 0xFFFFFFFFu;
    uint digit = valid ? ((key >> shift) & 0xFFu) : 0xFFu;

    // ── Phase 2: WLMS ballot rank + SG histogram ──────────
    // Build 8 ballot masks (one per bit of digit)
    uint bm0 = shuffle_ballot((digit >> 0) & 1, simd_lane);
    uint bm1 = shuffle_ballot((digit >> 1) & 1, simd_lane);
    uint bm2 = shuffle_ballot((digit >> 2) & 1, simd_lane);
    uint bm3 = shuffle_ballot((digit >> 3) & 1, simd_lane);
    uint bm4 = shuffle_ballot((digit >> 4) & 1, simd_lane);
    uint bm5 = shuffle_ballot((digit >> 5) & 1, simd_lane);
    uint bm6 = shuffle_ballot((digit >> 6) & 1, simd_lane);
    uint bm7 = shuffle_ballot((digit >> 7) & 1, simd_lane);

    // Compute match mask: lanes in this SG with same digit as me
    uint match = 0xFFFFFFFFu;
    match &= ((digit >> 0) & 1) ? bm0 : ~bm0;
    match &= ((digit >> 1) & 1) ? bm1 : ~bm1;
    match &= ((digit >> 2) & 1) ? bm2 : ~bm2;
    match &= ((digit >> 3) & 1) ? bm3 : ~bm3;
    match &= ((digit >> 4) & 1) ? bm4 : ~bm4;
    match &= ((digit >> 5) & 1) ? bm5 : ~bm5;
    match &= ((digit >> 6) & 1) ? bm6 : ~bm6;
    match &= ((digit >> 7) & 1) ? bm7 : ~bm7;

    // Mask out invalid lanes (digit=0xFF for invalid → mismatches real digits)
    // Actually, invalid threads have digit=0xFF which is a valid 8-bit value.
    // But invalid threads should not be counted. Mask out inactive lanes:
    uint active_mask = shuffle_ballot(valid, simd_lane);
    match &= active_mask;

    // Within-SG rank = popcount(matching lanes below me)
    uint below_me = (1u << simd_lane) - 1u;
    uint within_sg_rank = popcount(match & below_me);

    // SG histogram: popcount of match mask for each thread's digit
    // Thread 0 of each SG: compute histogram for all 256 bins from ballot masks
    // Alternative: each thread knows its own bin's count = popcount(match)
    // But we need ALL bin counts for this SG. Use shared memory atomics (same as before).

    // Zero SG histograms (non-atomic version, one thread per bin)
    for (uint i = lid; i < EXP14_NUM_SGS * EXP14_NUM_BINS; i += EXP14_TILE_SIZE) {
        sg_hist_ballot[i] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread writes its SG bin count via atomic (simplest correct approach)
    // Actually, just use non-atomic: each thread stores its own popcount
    // But multiple threads in same SG may have same digit → race condition.
    // Fall back to: lane 0 computes all 256 histogram bins from ballot masks.
    // That's 256 match computations × popcount = ~4K ops. Too slow.

    // Simpler: use the match mask. Each thread writes popcount(match) as the
    // count for its digit, but only if it's the first thread with that digit
    // in the SG (i.e., within_sg_rank == 0).
    if (valid && within_sg_rank == 0u) {
        sg_hist_ballot[simd_id * EXP14_NUM_BINS + digit] = popcount(match);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Reduce SG histograms → tile_hist ──────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < EXP14_NUM_SGS; sg++) {
            total += sg_hist_ballot[sg * EXP14_NUM_BINS + lid];
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Exclusive prefix sum → bin_start ──────────
    {
        uint val = tile_hist[lid];
        uint simd_pfx = simd_prefix_exclusive_sum(val);

        if (simd_lane == 31u) {
            sg_totals_pfx[simd_id] = simd_pfx + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u && simd_lane < 8u) {
            uint t = sg_totals_pfx[simd_lane];
            uint sg_excl = simd_prefix_exclusive_sum(t);
            sg_totals_pfx[simd_lane] = sg_excl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint offset = simd_pfx;
        if (simd_id > 0u) {
            offset += sg_totals_pfx[simd_id];
        }
        bin_start[lid] = offset;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Local sort using ballot rank ────────────
    if (valid) {
        // Cross-SG offset: count matching elements in lower SGs
        uint cross_sg = 0u;
        for (uint s = 0u; s < simd_id; s++) {
            cross_sg += sg_hist_ballot[s * EXP14_NUM_BINS + digit];
        }

        uint tile_rank = bin_start[digit] + cross_sg + within_sg_rank;
        local_keys[tile_rank] = key;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 6: Load global offsets ───────────────────────
    {
        uint tile_excl = tile_histograms[tile_id * EXP14_NUM_BINS + lid];
        uint global_excl = global_histogram[lid];
        local_offsets[lid] = global_excl + tile_excl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 7: Coalesced scatter to global memory ────────
    if (valid) {
        uint k = local_keys[lid];
        uint d = (k >> shift) & 0xFFu;
        uint within_bin_pos = lid - bin_start[d];
        uint global_pos = local_offsets[d] + within_bin_pos;
        dst[global_pos] = k;
    }
}

// V2/V3 shared constants
#define V2_TILE_SIZE    1024u
#define V2_ELEMS        4u

// ═══════════════════════════════════════════════════════════════════
// V3: FUSED HISTOGRAM + DECOUPLED LOOKBACK + BALLOT SCATTER
// Single dispatch per pass — eliminates histogram, prefix_scan, global_prefix
// Saves: duplicate src read, prefix_scan overhead, 3 dispatch barriers
// Uses atomic_thread_fence(seq_cst, device_scope) for cross-TG visibility
// ═══════════════════════════════════════════════════════════════════

// tile_status encoding: top 2 bits = flag, bottom 30 bits = count
#define FLAG_INVALID  0u
#define FLAG_LOCAL    1u
#define FLAG_PREFIX   2u
#define FLAG_BITS     30u
#define COUNT_MASK    ((1u << FLAG_BITS) - 1u)

kernel void exp14_sort_v3(
    device const uint*     src              [[buffer(0)]],
    device uint*           dst              [[buffer(1)]],
    device atomic_uint*    tile_status      [[buffer(2)]],
    constant Exp14Params&  params           [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V2_TILE_SIZE;

    // ── Shared memory (~16 KB) ─────────────────────────────
    threadgroup uint sg_hist[EXP14_NUM_SGS * EXP14_NUM_BINS];  // 8 KB
    threadgroup uint tile_hist[EXP14_NUM_BINS];                  // 1 KB
    threadgroup uint bin_start[EXP14_NUM_BINS];                  // 1 KB
    threadgroup uint bin_ctr[EXP14_NUM_BINS];                    // 1 KB
    threadgroup uint lk[V2_TILE_SIZE];                           // 4 KB
    threadgroup uint local_off[EXP14_NUM_BINS];                  // 1 KB
    threadgroup uint sg_tp[8];                                   // 32 B

    // ── Phase 1: Load 4 elements ──────────────────────────
    uint mk[V2_ELEMS];
    uint md[V2_ELEMS];
    bool mv[V2_ELEMS];
    for (uint e = 0u; e < V2_ELEMS; e++) {
        uint idx = base + e * EXP14_TILE_SIZE + lid;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Local histogram ─────────────────────────
    threadgroup atomic_uint ha[EXP14_NUM_BINS];
    atomic_store_explicit(&ha[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < V2_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(&ha[md[e]], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tile_hist[lid] = atomic_load_explicit(&ha[lid], memory_order_relaxed);

    // ── Phase 2b: Publish local histogram to tile_status ──
    {
        uint val = (FLAG_LOCAL << FLAG_BITS) | tile_hist[lid];
        atomic_store_explicit(&tile_status[tile_id * EXP14_NUM_BINS + lid],
                              val, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Decoupled lookback ─────────────────────
    // 256 threads, one per bin — each finds its bin's exclusive prefix
    {
        uint sum = 0u;
        if (tile_id > 0u) {
            int look = (int)tile_id - 1;
            uint spin = 0u;
            while (look >= 0) {
                uint val = atomic_load_explicit(
                    &tile_status[(uint)look * EXP14_NUM_BINS + lid],
                    memory_order_relaxed);
                uint flag = val >> FLAG_BITS;
                uint count = val & COUNT_MASK;

                if (flag == FLAG_PREFIX) {
                    sum += count;
                    break;
                } else if (flag == FLAG_LOCAL) {
                    sum += count;
                    look--;
                    spin = 0u;
                } else {
                    // FLAG_INVALID — spin with timeout to prevent deadlock
                    spin++;
                    if (spin > 1000000u) break;
                }
            }
        }
        local_off[lid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3b: Publish inclusive prefix ─────────────
    {
        uint inclusive = local_off[lid] + tile_hist[lid];
        uint val = (FLAG_PREFIX << FLAG_BITS) | inclusive;
        atomic_store_explicit(&tile_status[tile_id * EXP14_NUM_BINS + lid],
                              val, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);

    // NOTE: local_off[lid] only has within-bin exclusive prefix.
    // The global bin offset must come from a separate dispatch (see upsweep_v3).

    // ── Phase 4: Exclusive prefix sum → bin_start ────────
    {
        uint val = tile_hist[lid];
        uint sp = simd_prefix_exclusive_sum(val);

        if (simd_lane == 31u) {
            sg_tp[simd_id] = sp + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u && simd_lane < 8u) {
            uint t = sg_tp[simd_lane];
            uint se = simd_prefix_exclusive_sum(t);
            sg_tp[simd_lane] = se;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint off = sp;
        if (simd_id > 0u) {
            off += sg_tp[simd_id];
        }
        bin_start[lid] = off;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Round-by-round ballot rank + local sort ──
    bin_ctr[lid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint below_me = (1u << simd_lane) - 1u;

    for (uint e = 0u; e < V2_ELEMS; e++) {
        uint d_e = md[e];

        for (uint i = lid; i < EXP14_NUM_SGS * EXP14_NUM_BINS; i += EXP14_TILE_SIZE) {
            sg_hist[i] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint match_e = 0xFFFFFFFFu;
        for (uint b = 0u; b < 8u; b++) {
            uint bm = shuffle_ballot((d_e >> b) & 1u, simd_lane);
            match_e &= ((d_e >> b) & 1u) ? bm : ~bm;
        }
        uint active = shuffle_ballot(mv[e], simd_lane);
        match_e &= active;

        uint wsr = popcount(match_e & below_me);

        if (mv[e] && wsr == 0u) {
            sg_hist[simd_id * EXP14_NUM_BINS + d_e] = popcount(match_e);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint csg = 0u;
        if (mv[e]) {
            for (uint s = 0u; s < simd_id; s++) {
                csg += sg_hist[s * EXP14_NUM_BINS + d_e];
            }
        }

        if (mv[e]) {
            uint rank = bin_start[d_e] + bin_ctr[d_e] + csg + wsr;
            lk[rank] = mk[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint rt = 0u;
            for (uint s = 0u; s < EXP14_NUM_SGS; s++) {
                rt += sg_hist[s * EXP14_NUM_BINS + lid];
            }
            bin_ctr[lid] += rt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 6: Coalesced scatter ──────────────────────
    for (uint e = 0u; e < V2_ELEMS; e++) {
        uint li = e * EXP14_TILE_SIZE + lid;
        if (li < V2_TILE_SIZE && (base + li) < n) {
            uint k = lk[li];
            uint d = (k >> shift) & 0xFFu;
            uint wbp = li - bin_start[d];
            uint gp = local_off[d] + wbp;
            dst[gp] = k;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// V2 KERNELS: TILE_SIZE=1024, 4 elements/thread, ballot rank
// ═══════════════════════════════════════════════════════════════════

// ── V2 Histogram: 1024 elements/tile, 4 loads per thread ──────────

kernel void exp14_histogram_v2(
    device const uint*     src              [[buffer(0)]],
    device uint*           tile_histograms  [[buffer(1)]],
    constant Exp14Params&  params           [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint elem_count = params.element_count;
    uint base = tile_id * V2_TILE_SIZE;

    threadgroup atomic_uint local_hist[EXP14_NUM_BINS];
    atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < V2_ELEMS; e++) {
        uint idx = base + e * EXP14_TILE_SIZE + lid;
        if (idx < elem_count) {
            uint digit = (src[idx] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_histograms[tile_id * EXP14_NUM_BINS + lid] =
        atomic_load_explicit(&local_hist[lid], memory_order_relaxed);
}

// ── V2 Scatter: 1024 elements, ballot rank, coalesced writes ──────
//
// 4 sequential rounds of 256 elements each.
// Each round: ballot rank → SG histogram → cross-SG → place in local sort.
// Cumulative bin_counter tracks offset across rounds.

kernel void exp14_scatter_v2(
    device const uint*     src              [[buffer(0)]],
    device uint*           dst              [[buffer(1)]],
    device const uint*     tile_histograms  [[buffer(2)]],
    device const uint*     global_histogram [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint elem_count = params.element_count;
    uint base = tile_id * V2_TILE_SIZE;

    // ── Shared memory (~16 KB) ─────────────────────────────
    threadgroup uint sg_hist_v2[EXP14_NUM_SGS * EXP14_NUM_BINS]; // 8 KB per-round
    threadgroup uint tile_hist_v2[EXP14_NUM_BINS];                 // 1 KB
    threadgroup uint bin_start_v2[EXP14_NUM_BINS];                 // 1 KB
    threadgroup uint bin_ctr[EXP14_NUM_BINS];                      // 1 KB
    threadgroup uint lk[V2_TILE_SIZE];                             // 4 KB
    threadgroup uint local_off[EXP14_NUM_BINS];                    // 1 KB
    threadgroup uint sg_tp[8];                                     // 32 B

    // ── Phase 1: Load 4 elements ──────────────────────────
    uint mk[V2_ELEMS];
    uint md[V2_ELEMS];
    bool mv[V2_ELEMS];
    for (uint e = 0u; e < V2_ELEMS; e++) {
        uint idx = base + e * EXP14_TILE_SIZE + lid;
        mv[e] = idx < elem_count;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Build tile histogram ─────────────────────
    threadgroup atomic_uint ha[EXP14_NUM_BINS];
    atomic_store_explicit(&ha[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < V2_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(&ha[md[e]], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tile_hist_v2[lid] = atomic_load_explicit(&ha[lid], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Exclusive prefix sum → bin_start ────────
    {
        uint val = tile_hist_v2[lid];
        uint sp = simd_prefix_exclusive_sum(val);

        if (simd_lane == 31u) {
            sg_tp[simd_id] = sp + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u && simd_lane < 8u) {
            uint t = sg_tp[simd_lane];
            uint se = simd_prefix_exclusive_sum(t);
            sg_tp[simd_lane] = se;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint off = sp;
        if (simd_id > 0u) {
            off += sg_tp[simd_id];
        }
        bin_start_v2[lid] = off;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Round-by-round ballot rank + local sort ──
    bin_ctr[lid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint below_me = (1u << simd_lane) - 1u;

    for (uint e = 0u; e < V2_ELEMS; e++) {
        uint d_e = md[e];

        // Clear per-round SG histogram
        for (uint i = lid; i < EXP14_NUM_SGS * EXP14_NUM_BINS; i += EXP14_TILE_SIZE) {
            sg_hist_v2[i] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // WLMS ballot rank
        uint match_e = 0xFFFFFFFFu;
        for (uint b = 0u; b < 8u; b++) {
            uint bm = shuffle_ballot((d_e >> b) & 1u, simd_lane);
            match_e &= ((d_e >> b) & 1u) ? bm : ~bm;
        }
        uint active = shuffle_ballot(mv[e], simd_lane);
        match_e &= active;

        uint wsr = popcount(match_e & below_me);

        if (mv[e] && wsr == 0u) {
            sg_hist_v2[simd_id * EXP14_NUM_BINS + d_e] = popcount(match_e);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cross-SG offset
        uint csg = 0u;
        if (mv[e]) {
            for (uint s = 0u; s < simd_id; s++) {
                csg += sg_hist_v2[s * EXP14_NUM_BINS + d_e];
            }
        }

        // Place element
        if (mv[e]) {
            uint rank = bin_start_v2[d_e] + bin_ctr[d_e] + csg + wsr;
            lk[rank] = mk[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update cumulative counter
        {
            uint rt = 0u;
            for (uint s = 0u; s < EXP14_NUM_SGS; s++) {
                rt += sg_hist_v2[s * EXP14_NUM_BINS + lid];
            }
            bin_ctr[lid] += rt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 5: Load global offsets ───────────────────────
    {
        uint te = tile_histograms[tile_id * EXP14_NUM_BINS + lid];
        uint ge = global_histogram[lid];
        local_off[lid] = ge + te;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 6: Coalesced scatter (4 per thread) ──────────
    for (uint e = 0u; e < V2_ELEMS; e++) {
        uint li = e * EXP14_TILE_SIZE + lid;
        if (li < V2_TILE_SIZE && (base + li) < elem_count) {
            uint k = lk[li];
            uint d = (k >> shift) & 0xFFu;
            uint wbp = li - bin_start_v2[d];
            uint gp = local_off[d] + wbp;
            dst[gp] = k;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// V3 UPSWEEP: Fused histogram + decoupled lookback (replaces
// histogram_v2 + prefix_scan + global_prefix = 3 dispatches → 1)
//
// Writes results to tile_histograms[] and global_histogram[] in the
// same format that exp14_scatter_v2 and exp14_global_prefix expect.
// ═══════════════════════════════════════════════════════════════════

kernel void exp14_upsweep_v3(
    device const uint*     src              [[buffer(0)]],
    device uint*           tile_histograms  [[buffer(1)]],
    device uint*           global_histogram [[buffer(2)]],
    device atomic_uint*    tile_status      [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V2_TILE_SIZE;

    // ── Phase 1: Load 4 elements, compute histogram ──────
    threadgroup atomic_uint ha[EXP14_NUM_BINS];
    threadgroup uint hist[EXP14_NUM_BINS];

    atomic_store_explicit(&ha[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < V2_ELEMS; e++) {
        uint idx = base + e * EXP14_TILE_SIZE + lid;
        if (idx < n) {
            uint digit = (src[idx] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(&ha[digit], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    hist[lid] = atomic_load_explicit(&ha[lid], memory_order_relaxed);

    // ── Phase 2: Publish local histogram to tile_status ──
    {
        uint val = (FLAG_LOCAL << FLAG_BITS) | hist[lid];
        atomic_store_explicit(&tile_status[tile_id * EXP14_NUM_BINS + lid],
                              val, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Decoupled lookback ─────────────────────
    // 256 threads, one per bin — each finds its bin's exclusive prefix
    uint exclusive_prefix;
    {
        uint sum = 0u;
        if (tile_id > 0u) {
            int look = (int)tile_id - 1;
            uint spin = 0u;
            while (look >= 0) {
                uint val = atomic_load_explicit(
                    &tile_status[(uint)look * EXP14_NUM_BINS + lid],
                    memory_order_relaxed);
                uint flag = val >> FLAG_BITS;
                uint count = val & COUNT_MASK;

                if (flag == FLAG_PREFIX) {
                    sum += count;
                    break;
                } else if (flag == FLAG_LOCAL) {
                    sum += count;
                    look--;
                    spin = 0u;
                } else {
                    // FLAG_INVALID — spin with timeout to prevent deadlock
                    spin++;
                    if (spin > 1000000u) break;
                }
            }
        }
        exclusive_prefix = sum;
    }

    // ── Phase 4: Publish inclusive prefix ─────────────────
    {
        uint inclusive = exclusive_prefix + hist[lid];
        uint val = (FLAG_PREFIX << FLAG_BITS) | inclusive;
        atomic_store_explicit(&tile_status[tile_id * EXP14_NUM_BINS + lid],
                              val, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);

    // ── Phase 5: Write results ───────────────────────────
    // tile_histograms[tile_id * 256 + bin] = exclusive prefix for this tile's bin
    // (same format that exp14_scatter_v2 Phase 5 reads)
    tile_histograms[tile_id * EXP14_NUM_BINS + lid] = exclusive_prefix;

    // Last tile writes global totals (inclusive prefix = total count per bin)
    if (tile_id == params.num_tiles - 1u) {
        global_histogram[lid] = exclusive_prefix + hist[lid];
    }
}

// ═══════════════════════════════════════════════════════════════════
// V4 KERNELS: 4-BIT RADIX (16 bins, 8 passes)
//
// Key optimizations over V2 (8-bit, 256 bins, 4 passes):
// 1. 16 bins → zero bank conflicts (16 < 32 banks)
// 2. sg_hist shrinks 16x: 512B vs 8KB → trivial per-round clearing
// 3. 4 ballot masks vs 8 → 50% fewer SIMD shuffle ops
// 4. 100% cache utilization (64 elems/bin avg vs 4 for 8-bit)
// 5. int loop indices → compiler vectorization (Apple recommendation)
// 6. TG memory drops from ~17KB to ~5KB
// ═══════════════════════════════════════════════════════════════════

#define V4_NUM_BINS   16u
#define V4_NUM_SGS    8u

// V5 constants: 8 elements/thread
#define V5_TILE_SIZE  2048u
#define V5_ELEMS      8u

// V8 constants: 5-bit radix (32 bins, 7 passes)
#define V8_NUM_BINS   32u
#define V8_NUM_SGS    8u
#define V8_TILE_SIZE  2048u
#define V8_ELEMS      8u

// ── V4 Histogram: 4-bit digits, 1024 elements/tile ──────────────

kernel void exp14_histogram_v4(
    device const uint*     src              [[buffer(0)]],
    device uint*           tile_histograms  [[buffer(1)]],
    constant Exp14Params&  params           [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V2_TILE_SIZE;

    // Per-SG histograms: 32 threads/SG × 16 bins = 2:1 contention (vs 16:1 global)
    threadgroup atomic_uint sg_ha[V4_NUM_SGS * V4_NUM_BINS]; // 128 entries, 512B
    if (lid < V4_NUM_SGS * V4_NUM_BINS) {
        atomic_store_explicit(&sg_ha[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V2_ELEMS; e++) {
        uint idx = base + (uint)e * 256u + lid;
        if (idx < n) {
            uint digit = (src[idx] >> shift) & 0xFu;
            atomic_fetch_add_explicit(&sg_ha[simd_id * V4_NUM_BINS + digit],
                                      1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SGs
    if (lid < V4_NUM_BINS) {
        uint total = 0u;
        for (int sg = 0; sg < (int)V4_NUM_SGS; sg++) {
            total += atomic_load_explicit(&sg_ha[(uint)sg * V4_NUM_BINS + lid],
                                          memory_order_relaxed);
        }
        tile_histograms[tile_id * V4_NUM_BINS + lid] = total;
    }
}

// ── V4 Prefix Scan: 16 TGs (one per bin), scans tiles ──────────

kernel void exp14_prefix_scan_v4(
    device uint*           tile_histograms  [[buffer(0)]],
    device uint*           global_histogram [[buffer(1)]],
    constant Exp14Params&  params           [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint bin = gid;  // 16 TGs, one per bin
    uint num_tiles = params.num_tiles;

    threadgroup uint sg_totals[8];
    threadgroup uint chunk_total;

    uint running = 0u;

    for (uint chunk_start = 0u; chunk_start < num_tiles; chunk_start += 256u) {
        uint tile_idx = chunk_start + lid;
        uint val = (tile_idx < num_tiles)
            ? tile_histograms[tile_idx * V4_NUM_BINS + bin]
            : 0u;

        uint simd_pfx = simd_prefix_exclusive_sum(val);

        if (simd_lane == 31u) {
            sg_totals[simd_id] = simd_pfx + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u && simd_lane < 8u) {
            uint t = sg_totals[simd_lane];
            uint sg_excl = simd_prefix_exclusive_sum(t);
            sg_totals[simd_lane] = sg_excl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint local_prefix = simd_pfx;
        if (simd_id > 0u) {
            local_prefix += sg_totals[simd_id];
        }
        uint global_prefix = running + local_prefix;

        if (tile_idx < num_tiles) {
            tile_histograms[tile_idx * V4_NUM_BINS + bin] = global_prefix;
        }

        if (lid == 255u) {
            chunk_total = global_prefix + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running = chunk_total;
    }

    if (lid == 0u) {
        global_histogram[bin] = running;
    }
}

// ── V4 Global Prefix: 16-bin exclusive prefix sum ───────────────

kernel void exp14_global_prefix_v4(
    device uint*           global_histogram [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // 16 bins fits in a single simdgroup (lanes 0-15 have data, 16-31 = 0)
    uint val = (simd_id == 0u && simd_lane < V4_NUM_BINS)
        ? global_histogram[simd_lane] : 0u;
    uint pfx = simd_prefix_exclusive_sum(val);
    if (simd_id == 0u && simd_lane < V4_NUM_BINS) {
        global_histogram[simd_lane] = pfx;
    }
}

// ── V4 Scatter: 4-bit ballot rank, zero bank conflicts, int loops ──

kernel void exp14_scatter_v4(
    device const uint*     src              [[buffer(0)]],
    device uint*           dst              [[buffer(1)]],
    device const uint*     tile_histograms  [[buffer(2)]],
    device const uint*     global_histogram [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V2_TILE_SIZE;

    // ── Shared memory (~5 KB total) ───────────────────────
    threadgroup uint sg_hist[V4_NUM_SGS * V4_NUM_BINS]; // 512 B (zero bank conflicts!)
    threadgroup uint tile_hist[V4_NUM_BINS];             // 64 B
    threadgroup uint bin_start[V4_NUM_BINS];             // 64 B
    threadgroup uint bin_ctr[V4_NUM_BINS];               // 64 B
    threadgroup uint lk[V2_TILE_SIZE];                   // 4 KB
    threadgroup uint local_off[V4_NUM_BINS];             // 64 B
    // Total: ~5 KB vs ~17 KB for 8-bit!

    // ── Phase 1: Load 4 elements ──────────────────────────
    uint mk[V2_ELEMS];
    uint md[V2_ELEMS];
    bool mv[V2_ELEMS];
    for (int e = 0; e < (int)V2_ELEMS; e++) {
        uint idx = base + (uint)e * 256u + lid;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFu) : 0xFu;
    }

    // ── Phase 2: Build tile histogram (per-SG atomics, 2:1 contention) ──
    threadgroup atomic_uint ha[V4_NUM_SGS * V4_NUM_BINS]; // 512B
    if (lid < V4_NUM_SGS * V4_NUM_BINS) {
        atomic_store_explicit(&ha[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V2_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(&ha[simd_id * V4_NUM_BINS + md[e]],
                                      1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SGs
    if (lid < V4_NUM_BINS) {
        uint total = 0u;
        for (int sg = 0; sg < (int)V4_NUM_SGS; sg++) {
            total += atomic_load_explicit(&ha[(uint)sg * V4_NUM_BINS + lid],
                                          memory_order_relaxed);
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Exclusive prefix sum → bin_start ─────────
    // 16 bins — single simdgroup prefix sum (no cross-SG reduction!)
    {
        uint val = (simd_id == 0u && simd_lane < V4_NUM_BINS)
            ? tile_hist[simd_lane] : 0u;
        uint pfx = simd_prefix_exclusive_sum(val);
        if (simd_id == 0u && simd_lane < V4_NUM_BINS) {
            bin_start[simd_lane] = pfx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Round-by-round ballot rank + local sort ──
    if (lid < V4_NUM_BINS) {
        bin_ctr[lid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint below_me = (1u << simd_lane) - 1u;

    for (int e = 0; e < (int)V2_ELEMS; e++) {
        uint d_e = md[e];

        // Clear per-round SG histogram (128 entries / 256 threads = instant)
        if (lid < V4_NUM_SGS * V4_NUM_BINS) {
            sg_hist[lid] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // WLMS ballot rank — only 4 ballots for 4-bit digits!
        uint match_e = 0xFFFFFFFFu;
        for (int b = 0; b < 4; b++) {
            uint bm = shuffle_ballot((d_e >> (uint)b) & 1u, simd_lane);
            match_e &= ((d_e >> (uint)b) & 1u) ? bm : ~bm;
        }
        uint active = shuffle_ballot(mv[e], simd_lane);
        match_e &= active;

        uint wsr = popcount(match_e & below_me);

        // SG histogram — NO bank conflicts (16 bins < 32 banks)
        if (mv[e] && wsr == 0u) {
            sg_hist[simd_id * V4_NUM_BINS + d_e] = popcount(match_e);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cross-SG offset
        uint csg = 0u;
        if (mv[e]) {
            for (int s = 0; s < (int)simd_id; s++) {
                csg += sg_hist[(uint)s * V4_NUM_BINS + d_e];
            }
        }

        // Place element in local sort buffer
        if (mv[e]) {
            uint rank = bin_start[d_e] + bin_ctr[d_e] + csg + wsr;
            lk[rank] = mk[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update cumulative counter (only 16 threads needed)
        if (lid < V4_NUM_BINS) {
            uint rt = 0u;
            for (int s = 0; s < (int)V4_NUM_SGS; s++) {
                rt += sg_hist[(uint)s * V4_NUM_BINS + lid];
            }
            bin_ctr[lid] += rt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 5: Load global offsets ──────────────────────
    if (lid < V4_NUM_BINS) {
        uint te = tile_histograms[tile_id * V4_NUM_BINS + lid];
        uint ge = global_histogram[lid];
        local_off[lid] = ge + te;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 6: Coalesced scatter (4 per thread) ─────────
    for (int e = 0; e < (int)V2_ELEMS; e++) {
        uint li = (uint)e * 256u + lid;
        if (li < V2_TILE_SIZE && (base + li) < n) {
            uint k = lk[li];
            uint d = (k >> shift) & 0xFu;
            uint wbp = li - bin_start[d];
            uint gp = local_off[d] + wbp;
            dst[gp] = k;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// V5 KERNELS: 4-BIT RADIX, 8 ELEMENTS/THREAD (2048 elements/tile)
//
// Same 4-bit optimizations as V4, but doubles elements per thread:
// - TILE_SIZE=2048, 8 elements/thread
// - Half the number of tiles → less histogram/prefix overhead
// - 128 elements/bin avg → even better cache utilization
// - 8 rounds in scatter → more compute per dispatch
// ═══════════════════════════════════════════════════════════════════

kernel void exp14_histogram_v5(
    device const uint*     src              [[buffer(0)]],
    device uint*           tile_histograms  [[buffer(1)]],
    constant Exp14Params&  params           [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V5_TILE_SIZE;

    threadgroup atomic_uint sg_ha[V4_NUM_SGS * V4_NUM_BINS];
    if (lid < V4_NUM_SGS * V4_NUM_BINS) {
        atomic_store_explicit(&sg_ha[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V5_ELEMS; e++) {
        uint idx = base + (uint)e * 256u + lid;
        if (idx < n) {
            uint digit = (src[idx] >> shift) & 0xFu;
            atomic_fetch_add_explicit(&sg_ha[simd_id * V4_NUM_BINS + digit],
                                      1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < V4_NUM_BINS) {
        uint total = 0u;
        for (int sg = 0; sg < (int)V4_NUM_SGS; sg++) {
            total += atomic_load_explicit(&sg_ha[(uint)sg * V4_NUM_BINS + lid],
                                          memory_order_relaxed);
        }
        tile_histograms[tile_id * V4_NUM_BINS + lid] = total;
    }
}

kernel void exp14_scatter_v5(
    device const uint*     src              [[buffer(0)]],
    device uint*           dst              [[buffer(1)]],
    device const uint*     tile_histograms  [[buffer(2)]],
    device const uint*     global_histogram [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V5_TILE_SIZE;

    // ── Shared memory (~9 KB total) ───────────────────────
    threadgroup uint sg_hist[V4_NUM_SGS * V4_NUM_BINS]; // 512 B
    threadgroup uint tile_hist[V4_NUM_BINS];             // 64 B
    threadgroup uint bin_start[V4_NUM_BINS];             // 64 B
    threadgroup uint bin_ctr[V4_NUM_BINS];               // 64 B
    threadgroup uint lk[V5_TILE_SIZE];                   // 8 KB
    threadgroup uint local_off[V4_NUM_BINS];             // 64 B

    // ── Phase 1: Load 8 elements ──────────────────────────
    uint mk[V5_ELEMS];
    uint md[V5_ELEMS];
    bool mv[V5_ELEMS];
    for (int e = 0; e < (int)V5_ELEMS; e++) {
        uint idx = base + (uint)e * 256u + lid;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFu) : 0xFu;
    }

    // ── Phase 2: Build tile histogram (per-SG atomics) ────
    threadgroup atomic_uint ha[V4_NUM_SGS * V4_NUM_BINS];
    if (lid < V4_NUM_SGS * V4_NUM_BINS) {
        atomic_store_explicit(&ha[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V5_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(&ha[simd_id * V4_NUM_BINS + md[e]],
                                      1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < V4_NUM_BINS) {
        uint total = 0u;
        for (int sg = 0; sg < (int)V4_NUM_SGS; sg++) {
            total += atomic_load_explicit(&ha[(uint)sg * V4_NUM_BINS + lid],
                                          memory_order_relaxed);
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Exclusive prefix sum → bin_start ─────────
    {
        uint val = (simd_id == 0u && simd_lane < V4_NUM_BINS)
            ? tile_hist[simd_lane] : 0u;
        uint pfx = simd_prefix_exclusive_sum(val);
        if (simd_id == 0u && simd_lane < V4_NUM_BINS) {
            bin_start[simd_lane] = pfx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Round-by-round ballot rank + local sort ──
    if (lid < V4_NUM_BINS) {
        bin_ctr[lid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint below_me = (1u << simd_lane) - 1u;

    for (int e = 0; e < (int)V5_ELEMS; e++) {
        uint d_e = md[e];

        if (lid < V4_NUM_SGS * V4_NUM_BINS) {
            sg_hist[lid] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint match_e = 0xFFFFFFFFu;
        for (int b = 0; b < 4; b++) {
            uint bm = shuffle_ballot((d_e >> (uint)b) & 1u, simd_lane);
            match_e &= ((d_e >> (uint)b) & 1u) ? bm : ~bm;
        }
        uint active = shuffle_ballot(mv[e], simd_lane);
        match_e &= active;

        uint wsr = popcount(match_e & below_me);

        if (mv[e] && wsr == 0u) {
            sg_hist[simd_id * V4_NUM_BINS + d_e] = popcount(match_e);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint csg = 0u;
        if (mv[e]) {
            for (int s = 0; s < (int)simd_id; s++) {
                csg += sg_hist[(uint)s * V4_NUM_BINS + d_e];
            }
        }

        if (mv[e]) {
            uint rank = bin_start[d_e] + bin_ctr[d_e] + csg + wsr;
            lk[rank] = mk[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < V4_NUM_BINS) {
            uint rt = 0u;
            for (int s = 0; s < (int)V4_NUM_SGS; s++) {
                rt += sg_hist[(uint)s * V4_NUM_BINS + lid];
            }
            bin_ctr[lid] += rt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 5: Load global offsets ──────────────────────
    if (lid < V4_NUM_BINS) {
        uint te = tile_histograms[tile_id * V4_NUM_BINS + lid];
        uint ge = global_histogram[lid];
        local_off[lid] = ge + te;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 6: Coalesced scatter (8 per thread) ─────────
    for (int e = 0; e < (int)V5_ELEMS; e++) {
        uint li = (uint)e * 256u + lid;
        if (li < V5_TILE_SIZE && (base + li) < n) {
            uint k = lk[li];
            uint d = (k >> shift) & 0xFu;
            uint wbp = li - bin_start[d];
            uint gp = local_off[d] + wbp;
            dst[gp] = k;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// V6: FUSED 4-BIT RADIX SORT (single dispatch per pass)
//
// Combines histogram + decoupled lookback + local sort + scatter.
// Reads src only ONCE per pass (eliminates duplicate 64MB read).
// 4-bit radix (16 bins), 8 elements/thread (2048/tile).
//
// Key insight: lookback gives per-bin exclusive prefix, but scatter
// also needs global_bin_start[b] = total elements in bins 0..b-1.
// Solution: after lookback, spin-wait on last tile's inclusive prefix
// (= total count per bin), then compute global_bin_start via
// simd_prefix_exclusive_sum.
// ═══════════════════════════════════════════════════════════════════

kernel void exp14_sort_v6(
    device const uint*     src          [[buffer(0)]],
    device uint*           dst          [[buffer(1)]],
    device atomic_uint*    tile_status  [[buffer(2)]],
    constant Exp14Params&  params       [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint num_tiles = params.num_tiles;
    uint base = tile_id * V5_TILE_SIZE;

    // ── Shared memory (~9.5 KB) ──────────────────────────────
    threadgroup uint sg_hist[V4_NUM_SGS * V4_NUM_BINS];  // 512B
    threadgroup uint tile_hist[V4_NUM_BINS];               // 64B
    threadgroup uint bin_start[V4_NUM_BINS];               // 64B
    threadgroup uint bin_ctr[V4_NUM_BINS];                 // 64B
    threadgroup uint lk[V5_TILE_SIZE];                     // 8KB
    threadgroup uint local_off[V4_NUM_BINS];               // 64B
    threadgroup uint global_bin_off[V4_NUM_BINS];           // 64B

    // ── Phase 1: Load 8 elements ─────────────────────────────
    uint mk[V5_ELEMS];
    uint md[V5_ELEMS];
    bool mv[V5_ELEMS];
    for (int e = 0; e < (int)V5_ELEMS; e++) {
        uint idx = base + (uint)e * 256u + lid;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFu) : 0xFu;
    }

    // ── Phase 2: Per-SG histogram ────────────────────────────
    threadgroup atomic_uint ha[V4_NUM_SGS * V4_NUM_BINS];
    if (lid < V4_NUM_SGS * V4_NUM_BINS) {
        atomic_store_explicit(&ha[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V5_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(&ha[simd_id * V4_NUM_BINS + md[e]],
                                      1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SGs → 16-bin tile histogram
    if (lid < V4_NUM_BINS) {
        uint total = 0u;
        for (int sg = 0; sg < (int)V4_NUM_SGS; sg++) {
            total += atomic_load_explicit(&ha[(uint)sg * V4_NUM_BINS + lid],
                                          memory_order_relaxed);
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Publish local histogram to tile_status ──────
    if (lid < V4_NUM_BINS) {
        uint val = (FLAG_LOCAL << FLAG_BITS) | tile_hist[lid];
        atomic_store_explicit(&tile_status[tile_id * V4_NUM_BINS + lid],
                              val, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Decoupled lookback (16 threads only) ────────
    if (lid < V4_NUM_BINS) {
        uint sum = 0u;
        if (tile_id > 0u) {
            int look = (int)tile_id - 1;
            uint spin = 0u;
            while (look >= 0) {
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                uint val = atomic_load_explicit(
                    &tile_status[(uint)look * V4_NUM_BINS + lid],
                    memory_order_relaxed);
                uint flag = val >> FLAG_BITS;
                uint count = val & COUNT_MASK;

                if (flag == FLAG_PREFIX) {
                    sum += count;
                    break;
                } else if (flag == FLAG_LOCAL) {
                    sum += count;
                    look--;
                    spin = 0u;
                } else {
                    spin++;
                    if (spin > 2000000u) break;
                }
            }
        }
        local_off[lid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4b: Publish inclusive prefix ────────────────────
    if (lid < V4_NUM_BINS) {
        uint inclusive = local_off[lid] + tile_hist[lid];
        uint val = (FLAG_PREFIX << FLAG_BITS) | inclusive;
        atomic_store_explicit(&tile_status[tile_id * V4_NUM_BINS + lid],
                              val, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4c: Get global bin starts ──────────────────────
    // Spin-wait on last tile's inclusive prefix = total count per bin.
    // Then compute global_bin_off[b] = exclusive prefix sum of totals.
    if (lid < V4_NUM_BINS) {
        uint last_tile = num_tiles - 1u;
        uint total_count;
        if (tile_id == last_tile) {
            // We ARE the last tile — our inclusive prefix is the total
            total_count = local_off[lid] + tile_hist[lid];
        } else {
            // Spin-wait for last tile to publish FLAG_PREFIX
            uint spin = 0u;
            while (true) {
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                uint val = atomic_load_explicit(
                    &tile_status[last_tile * V4_NUM_BINS + lid],
                    memory_order_relaxed);
                uint flag = val >> FLAG_BITS;
                if (flag == FLAG_PREFIX) {
                    total_count = val & COUNT_MASK;
                    break;
                }
                spin++;
                if (spin > 4000000u) {
                    total_count = 0u;
                    break;
                }
            }
        }
        // Exclusive prefix sum of total_count across 16 bins
        // Only lanes 0..15 in simdgroup 0 participate
        global_bin_off[lid] = simd_prefix_exclusive_sum(total_count);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Exclusive prefix sum → bin_start ────────────
    if (lid < V4_NUM_BINS) {
        uint val = tile_hist[lid];
        uint sp = simd_prefix_exclusive_sum(val);
        bin_start[lid] = sp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 6: Ballot rank + local sort (8 rounds) ─────────
    if (lid < V4_NUM_BINS) {
        bin_ctr[lid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint below_me = (1u << simd_lane) - 1u;

    for (int e = 0; e < (int)V5_ELEMS; e++) {
        uint d_e = md[e];

        if (lid < V4_NUM_SGS * V4_NUM_BINS) {
            sg_hist[lid] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint match_e = 0xFFFFFFFFu;
        for (int b = 0; b < 4; b++) {
            uint bm = shuffle_ballot((d_e >> (uint)b) & 1u, simd_lane);
            match_e &= ((d_e >> (uint)b) & 1u) ? bm : ~bm;
        }
        uint active = shuffle_ballot(mv[e], simd_lane);
        match_e &= active;

        uint wsr = popcount(match_e & below_me);

        if (mv[e] && wsr == 0u) {
            sg_hist[simd_id * V4_NUM_BINS + d_e] = popcount(match_e);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint csg = 0u;
        if (mv[e]) {
            for (int s = 0; s < (int)simd_id; s++) {
                csg += sg_hist[(uint)s * V4_NUM_BINS + d_e];
            }
        }

        if (mv[e]) {
            uint rank = bin_start[d_e] + bin_ctr[d_e] + csg + wsr;
            lk[rank] = mk[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < V4_NUM_BINS) {
            uint rt = 0u;
            for (int s = 0; s < (int)V4_NUM_SGS; s++) {
                rt += sg_hist[(uint)s * V4_NUM_BINS + lid];
            }
            bin_ctr[lid] += rt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 7: Coalesced scatter (8 per thread) ────────────
    // gp = global_bin_off[d] + local_off[d] + within_bin_position
    for (int e = 0; e < (int)V5_ELEMS; e++) {
        uint li = (uint)e * 256u + lid;
        if (li < V5_TILE_SIZE && (base + li) < n) {
            uint k = lk[li];
            uint d = (k >> shift) & 0xFu;
            uint wbp = li - bin_start[d];
            uint gp = global_bin_off[d] + local_off[d] + wbp;
            dst[gp] = k;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// V7: 8-BIT RADIX, 4 PASSES (halves total bandwidth vs 4-bit/8-pass)
//
// Key insight: bandwidth-bound at 174 GB/s, cutting passes from 8→4
// saves 512MB of src reads. 256 threads = 256 bins = perfect lookback
// utilization (every thread does useful work).
//
// uint4 vectorized loads: 2 loads/thread × 256 threads = 2048 elems/tile
// Per-SG atomic histograms: 8 SGs × 256 bins = 8KB (fits in TG memory)
// 2-level SIMD prefix sum for 256-bin bin_start
// ═══════════════════════════════════════════════════════════════════

#define V7_TILE_SIZE  2048u
#define V7_ELEMS      8u
#define V7_NUM_BINS   256u
#define V7_NUM_SGS    8u

kernel void exp14_upsweep_v7(
    device const uint*     src              [[buffer(0)]],
    device uint*           tile_histograms  [[buffer(1)]],
    device uint*           global_histogram [[buffer(2)]],
    device atomic_uint*    tile_status      [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V7_TILE_SIZE;

    // ── Phase 1: Per-SG histogram via uint4 vectorized loads ────
    threadgroup atomic_uint ha[V7_NUM_SGS * V7_NUM_BINS]; // 8KB
    threadgroup uint hist[V7_NUM_BINS]; // 1KB

    // Clear per-SG histograms: 2048 entries / 256 threads = 8 per thread
    for (uint i = lid; i < V7_NUM_SGS * V7_NUM_BINS; i += 256u) {
        atomic_store_explicit(&ha[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // uint4 loads: 2 loads × 4 elements = 8 per thread, coalesced
    device const uint4* src4 = (device const uint4*)(src + base);
    uint4 v0 = (base + lid * 4u < n) ? src4[lid] : uint4(0xFFFFFFFFu);
    uint4 v1 = (base + 1024u + lid * 4u < n) ? src4[256u + lid] : uint4(0xFFFFFFFFu);

    // Extract digits and accumulate histogram
    uint d0 = (v0.x >> shift) & 0xFFu;
    uint d1 = (v0.y >> shift) & 0xFFu;
    uint d2 = (v0.z >> shift) & 0xFFu;
    uint d3 = (v0.w >> shift) & 0xFFu;
    uint d4 = (v1.x >> shift) & 0xFFu;
    uint d5 = (v1.y >> shift) & 0xFFu;
    uint d6 = (v1.z >> shift) & 0xFFu;
    uint d7 = (v1.w >> shift) & 0xFFu;

    uint sg_base = simd_id * V7_NUM_BINS;
    if (base + lid * 4u      < n) atomic_fetch_add_explicit(&ha[sg_base + d0], 1u, memory_order_relaxed);
    if (base + lid * 4u + 1u < n) atomic_fetch_add_explicit(&ha[sg_base + d1], 1u, memory_order_relaxed);
    if (base + lid * 4u + 2u < n) atomic_fetch_add_explicit(&ha[sg_base + d2], 1u, memory_order_relaxed);
    if (base + lid * 4u + 3u < n) atomic_fetch_add_explicit(&ha[sg_base + d3], 1u, memory_order_relaxed);
    if (base + 1024u + lid * 4u      < n) atomic_fetch_add_explicit(&ha[sg_base + d4], 1u, memory_order_relaxed);
    if (base + 1024u + lid * 4u + 1u < n) atomic_fetch_add_explicit(&ha[sg_base + d5], 1u, memory_order_relaxed);
    if (base + 1024u + lid * 4u + 2u < n) atomic_fetch_add_explicit(&ha[sg_base + d6], 1u, memory_order_relaxed);
    if (base + 1024u + lid * 4u + 3u < n) atomic_fetch_add_explicit(&ha[sg_base + d7], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SGs: each of 256 threads handles one bin
    {
        uint total = 0u;
        for (int sg = 0; sg < (int)V7_NUM_SGS; sg++) {
            total += atomic_load_explicit(&ha[(uint)sg * V7_NUM_BINS + lid],
                                          memory_order_relaxed);
        }
        hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Publish FLAG_LOCAL ─────────────────────────────
    {
        uint val = (FLAG_LOCAL << FLAG_BITS) | hist[lid];
        atomic_store_explicit(&tile_status[tile_id * V7_NUM_BINS + lid],
                              val, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Decoupled lookback (256 threads = 256 bins, full util) ─
    uint exclusive_prefix = 0u;
    {
        uint sum = 0u;
        if (tile_id > 0u) {
            int look = (int)tile_id - 1;
            uint spin = 0u;
            while (look >= 0) {
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                uint val = atomic_load_explicit(
                    &tile_status[(uint)look * V7_NUM_BINS + lid],
                    memory_order_relaxed);
                uint flag = val >> FLAG_BITS;
                uint count = val & COUNT_MASK;

                if (flag == FLAG_PREFIX) {
                    sum += count;
                    break;
                } else if (flag == FLAG_LOCAL) {
                    sum += count;
                    look--;
                    spin = 0u;
                } else {
                    spin++;
                    if (spin > 2000000u) break;
                }
            }
        }
        exclusive_prefix = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Publish FLAG_PREFIX ────────────────────────────
    {
        uint inclusive = exclusive_prefix + hist[lid];
        uint val = (FLAG_PREFIX << FLAG_BITS) | inclusive;
        atomic_store_explicit(&tile_status[tile_id * V7_NUM_BINS + lid],
                              val, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);

    // ── Phase 5: Write per-tile exclusive prefix ────────────────
    tile_histograms[tile_id * V7_NUM_BINS + lid] = exclusive_prefix;

    // ── Phase 6: Last tile computes global_bin_off ──────────────
    // 256 bins → 2-level SIMD prefix sum (8 simdgroups × 32 lanes)
    if (tile_id == params.num_tiles - 1u) {
        uint grand_total = exclusive_prefix + hist[lid];

        threadgroup uint sg_pfx[8];
        uint sp = simd_prefix_exclusive_sum(grand_total);

        if (simd_lane == 31u) {
            sg_pfx[simd_id] = sp + grand_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u && simd_lane < 8u) {
            uint t = sg_pfx[simd_lane];
            uint se = simd_prefix_exclusive_sum(t);
            sg_pfx[simd_lane] = se;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint bin_off = sp;
        if (simd_id > 0u) {
            bin_off += sg_pfx[simd_id];
        }
        global_histogram[lid] = bin_off;
    }
}

kernel void exp14_scatter_v7(
    device const uint*     src              [[buffer(0)]],
    device uint*           dst              [[buffer(1)]],
    device const uint*     tile_histograms  [[buffer(2)]],
    device const uint*     global_histogram [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V7_TILE_SIZE;

    // ── Shared memory (~20 KB) ──────────────────────────────────
    threadgroup uint sg_hist[V7_NUM_SGS * V7_NUM_BINS];   // 8 KB
    threadgroup uint tile_hist[V7_NUM_BINS];                // 1 KB
    threadgroup uint bin_start[V7_NUM_BINS];                // 1 KB
    threadgroup uint bin_ctr[V7_NUM_BINS];                  // 1 KB
    threadgroup uint lk[V7_TILE_SIZE];                      // 8 KB
    threadgroup uint local_off[V7_NUM_BINS];                // 1 KB
    threadgroup uint sg_tp[8];                              // 32 B

    // ── Phase 1: Strided loads (preserves element order for stability) ─
    uint mk[V7_ELEMS];
    bool mv[V7_ELEMS];
    uint md[V7_ELEMS];
    for (int e = 0; e < (int)V7_ELEMS; e++) {
        uint idx = base + (uint)e * 256u + lid;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ── Phase 2: Per-SG histogram ───────────────────────────────
    threadgroup atomic_uint ha[V7_NUM_SGS * V7_NUM_BINS]; // 8KB
    for (uint i = lid; i < V7_NUM_SGS * V7_NUM_BINS; i += 256u) {
        atomic_store_explicit(&ha[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V7_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(&ha[simd_id * V7_NUM_BINS + md[e]],
                                      1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce → tile_hist
    {
        uint total = 0u;
        for (int sg = 0; sg < (int)V7_NUM_SGS; sg++) {
            total += atomic_load_explicit(&ha[(uint)sg * V7_NUM_BINS + lid],
                                          memory_order_relaxed);
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: 2-level SIMD prefix sum → bin_start ────────────
    {
        uint val = tile_hist[lid];
        uint sp = simd_prefix_exclusive_sum(val);

        if (simd_lane == 31u) {
            sg_tp[simd_id] = sp + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u && simd_lane < 8u) {
            uint t = sg_tp[simd_lane];
            uint se = simd_prefix_exclusive_sum(t);
            sg_tp[simd_lane] = se;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint off = sp;
        if (simd_id > 0u) {
            off += sg_tp[simd_id];
        }
        bin_start[lid] = off;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Round-by-round ballot rank + local sort ────────
    bin_ctr[lid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint below_me = (1u << simd_lane) - 1u;

    for (int e = 0; e < (int)V7_ELEMS; e++) {
        uint d_e = md[e];

        // Clear per-round SG histogram
        for (uint i = lid; i < V7_NUM_SGS * V7_NUM_BINS; i += 256u) {
            sg_hist[i] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 8-bit WLMS ballot rank
        uint match_e = 0xFFFFFFFFu;
        for (int b = 0; b < 8; b++) {
            uint bm = shuffle_ballot((d_e >> (uint)b) & 1u, simd_lane);
            match_e &= ((d_e >> (uint)b) & 1u) ? bm : ~bm;
        }
        uint active = shuffle_ballot(mv[e], simd_lane);
        match_e &= active;

        uint wsr = popcount(match_e & below_me);

        if (mv[e] && wsr == 0u) {
            sg_hist[simd_id * V7_NUM_BINS + d_e] = popcount(match_e);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint csg = 0u;
        if (mv[e]) {
            for (int s = 0; s < (int)simd_id; s++) {
                csg += sg_hist[(uint)s * V7_NUM_BINS + d_e];
            }
        }

        if (mv[e]) {
            uint rank = bin_start[d_e] + bin_ctr[d_e] + csg + wsr;
            lk[rank] = mk[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update cumulative counter
        {
            uint rt = 0u;
            for (int s = 0; s < (int)V7_NUM_SGS; s++) {
                rt += sg_hist[(uint)s * V7_NUM_BINS + lid];
            }
            bin_ctr[lid] += rt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 5: Load global offsets ────────────────────────────
    {
        uint te = tile_histograms[tile_id * V7_NUM_BINS + lid];
        uint ge = global_histogram[lid];
        local_off[lid] = ge + te;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 6: Coalesced scatter (8 per thread) ───────────────
    for (int e = 0; e < (int)V7_ELEMS; e++) {
        uint li = (uint)e * 256u + lid;
        if (li < V7_TILE_SIZE && (base + li) < n) {
            uint k = lk[li];
            uint d = (k >> shift) & 0xFFu;
            uint wbp = li - bin_start[d];
            uint gp = local_off[d] + wbp;
            dst[gp] = k;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// V6-2DISP: Fused 4-bit upsweep (histogram + lookback + global prefix)
//
// Fixes V6 deadlock: V6 had ALL tiles spin-waiting on the last tile's
// inclusive prefix to compute global_bin_off. If num_tiles > max
// concurrent TGs, the last tile never starts → deadlock.
//
// Fix: 2 dispatches per pass (upsweep + scatter), 16 encoders in one
// cmd.commit(). No CPU waits. Eliminates 2 of 4 original dispatches.
// Last tile computes global_bin_off inline via simd_prefix_exclusive_sum.
// ═══════════════════════════════════════════════════════════════════

kernel void exp14_upsweep_v6(
    device const uint*     src              [[buffer(0)]],
    device uint*           tile_histograms  [[buffer(1)]],
    device uint*           global_histogram [[buffer(2)]],
    device atomic_uint*    tile_status      [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V5_TILE_SIZE;

    // ── Phase 1: Per-SG histogram (512B TG memory) ──────────────
    threadgroup atomic_uint ha[V4_NUM_SGS * V4_NUM_BINS];
    threadgroup uint hist[V4_NUM_BINS];

    if (lid < V4_NUM_SGS * V4_NUM_BINS) {
        atomic_store_explicit(&ha[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V5_ELEMS; e++) {
        uint idx = base + (uint)e * 256u + lid;
        if (idx < n) {
            uint digit = (src[idx] >> shift) & 0xFu;
            atomic_fetch_add_explicit(&ha[simd_id * V4_NUM_BINS + digit],
                                      1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < V4_NUM_BINS) {
        uint total = 0u;
        for (int sg = 0; sg < (int)V4_NUM_SGS; sg++) {
            total += atomic_load_explicit(&ha[(uint)sg * V4_NUM_BINS + lid],
                                          memory_order_relaxed);
        }
        hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Publish local histogram ────────────────────────
    if (lid < V4_NUM_BINS) {
        uint val = (FLAG_LOCAL << FLAG_BITS) | hist[lid];
        atomic_store_explicit(&tile_status[tile_id * V4_NUM_BINS + lid],
                              val, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Decoupled lookback (16 threads, one per bin) ───
    uint exclusive_prefix = 0u;
    if (lid < V4_NUM_BINS) {
        uint sum = 0u;
        if (tile_id > 0u) {
            int look = (int)tile_id - 1;
            uint spin = 0u;
            while (look >= 0) {
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                uint val = atomic_load_explicit(
                    &tile_status[(uint)look * V4_NUM_BINS + lid],
                    memory_order_relaxed);
                uint flag = val >> FLAG_BITS;
                uint count = val & COUNT_MASK;

                if (flag == FLAG_PREFIX) {
                    sum += count;
                    break;
                } else if (flag == FLAG_LOCAL) {
                    sum += count;
                    look--;
                    spin = 0u;
                } else {
                    spin++;
                    if (spin > 2000000u) break;
                }
            }
        }
        exclusive_prefix = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Publish inclusive prefix ────────────────────────
    if (lid < V4_NUM_BINS) {
        uint inclusive = exclusive_prefix + hist[lid];
        uint val = (FLAG_PREFIX << FLAG_BITS) | inclusive;
        atomic_store_explicit(&tile_status[tile_id * V4_NUM_BINS + lid],
                              val, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);

    // ── Phase 5: Write per-tile exclusive prefix ────────────────
    if (lid < V4_NUM_BINS) {
        tile_histograms[tile_id * V4_NUM_BINS + lid] = exclusive_prefix;
    }

    // ── Phase 6: Last tile computes global_bin_off ──────────────
    // 16 bins in one simdgroup — single simd_prefix_exclusive_sum
    if (tile_id == params.num_tiles - 1u && simd_id == 0u) {
        uint grand_total = (simd_lane < V4_NUM_BINS)
            ? (exclusive_prefix + hist[simd_lane])
            : 0u;
        uint bin_off = simd_prefix_exclusive_sum(grand_total);
        if (simd_lane < V4_NUM_BINS) {
            global_histogram[simd_lane] = bin_off;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// V8: 5-BIT RADIX SORT (32 bins, 7 passes)
//
// Same architecture as V5 (histogram + prefix_scan + global_prefix + scatter)
// but with 5-bit digits (32 bins) instead of 4-bit (16 bins).
// 7 passes × 5 bits = 35 bits covers all 32 bits (last pass: 2 bits).
// Saves 1 full pass vs V5's 8 passes = 12.5% less bandwidth.
// 32 bins fits in one SIMD prefix sum (SIMD width = 32).
// ═══════════════════════════════════════════════════════════════════

kernel void exp14_histogram_v8(
    device const uint*     src              [[buffer(0)]],
    device uint*           tile_histograms  [[buffer(1)]],
    constant Exp14Params&  params           [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V8_TILE_SIZE;

    threadgroup atomic_uint sg_ha[V8_NUM_SGS * V8_NUM_BINS]; // 256 entries, 1KB
    // All 256 threads clear all 256 entries — perfect 1:1
    atomic_store_explicit(&sg_ha[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V8_ELEMS; e++) {
        uint idx = base + (uint)e * 256u + lid;
        if (idx < n) {
            uint digit = (src[idx] >> shift) & 0x1Fu;
            atomic_fetch_add_explicit(&sg_ha[simd_id * V8_NUM_BINS + digit],
                                      1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < V8_NUM_BINS) {
        uint total = 0u;
        for (int sg = 0; sg < (int)V8_NUM_SGS; sg++) {
            total += atomic_load_explicit(&sg_ha[(uint)sg * V8_NUM_BINS + lid],
                                          memory_order_relaxed);
        }
        tile_histograms[tile_id * V8_NUM_BINS + lid] = total;
    }
}

// ── V8 Prefix Scan: 32 TGs (one per bin), scans tiles ──────────

kernel void exp14_prefix_scan_v8(
    device uint*           tile_histograms  [[buffer(0)]],
    device uint*           global_histogram [[buffer(1)]],
    constant Exp14Params&  params           [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint bin = gid;  // 32 TGs, one per bin
    uint num_tiles = params.num_tiles;

    threadgroup uint sg_totals[8];
    threadgroup uint chunk_total;

    uint running = 0u;

    for (uint chunk_start = 0u; chunk_start < num_tiles; chunk_start += 256u) {
        uint tile_idx = chunk_start + lid;
        uint val = (tile_idx < num_tiles)
            ? tile_histograms[tile_idx * V8_NUM_BINS + bin]
            : 0u;

        uint simd_pfx = simd_prefix_exclusive_sum(val);

        if (simd_lane == 31u) {
            sg_totals[simd_id] = simd_pfx + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u && simd_lane < 8u) {
            uint t = sg_totals[simd_lane];
            uint sg_excl = simd_prefix_exclusive_sum(t);
            sg_totals[simd_lane] = sg_excl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint local_prefix = simd_pfx;
        if (simd_id > 0u) {
            local_prefix += sg_totals[simd_id];
        }
        uint global_prefix = running + local_prefix;

        if (tile_idx < num_tiles) {
            tile_histograms[tile_idx * V8_NUM_BINS + bin] = global_prefix;
        }

        if (lid == 255u) {
            chunk_total = global_prefix + val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running = chunk_total;
    }

    if (lid == 0u) {
        global_histogram[bin] = running;
    }
}

// ── V8 Global Prefix: 32-bin exclusive prefix sum ───────────────

kernel void exp14_global_prefix_v8(
    device uint*           global_histogram [[buffer(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // 32 bins fits exactly in one simdgroup (all 32 lanes active)
    uint val = (simd_id == 0u) ? global_histogram[simd_lane] : 0u;
    uint pfx = simd_prefix_exclusive_sum(val);
    if (simd_id == 0u) {
        global_histogram[simd_lane] = pfx;
    }
}

// ── V8 Scatter: 5-bit ballot rank + local sort + coalesced scatter ─

kernel void exp14_scatter_v8(
    device const uint*     src              [[buffer(0)]],
    device uint*           dst              [[buffer(1)]],
    device const uint*     tile_histograms  [[buffer(2)]],
    device const uint*     global_histogram [[buffer(3)]],
    constant Exp14Params&  params           [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint tile_id = gid;
    uint shift = params.shift;
    uint n = params.element_count;
    uint base = tile_id * V8_TILE_SIZE;

    // ── Shared memory (~11 KB total) ───────────────────────
    threadgroup uint sg_hist[V8_NUM_SGS * V8_NUM_BINS]; // 1 KB
    threadgroup uint tile_hist[V8_NUM_BINS];             // 128 B
    threadgroup uint bin_start[V8_NUM_BINS];             // 128 B
    threadgroup uint bin_ctr[V8_NUM_BINS];               // 128 B
    threadgroup uint lk[V8_TILE_SIZE];                   // 8 KB
    threadgroup uint local_off[V8_NUM_BINS];             // 128 B

    // ── Phase 1: Load 8 elements (strided for stability) ──
    uint mk[V8_ELEMS];
    uint md[V8_ELEMS];
    bool mv[V8_ELEMS];
    for (int e = 0; e < (int)V8_ELEMS; e++) {
        uint idx = base + (uint)e * 256u + lid;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0x1Fu) : 0x1Fu;
    }

    // ── Phase 2: Build tile histogram (per-SG atomics) ────
    threadgroup atomic_uint ha[V8_NUM_SGS * V8_NUM_BINS]; // 256 entries
    // All 256 threads clear all 256 entries
    atomic_store_explicit(&ha[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = 0; e < (int)V8_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(&ha[simd_id * V8_NUM_BINS + md[e]],
                                      1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < V8_NUM_BINS) {
        uint total = 0u;
        for (int sg = 0; sg < (int)V8_NUM_SGS; sg++) {
            total += atomic_load_explicit(&ha[(uint)sg * V8_NUM_BINS + lid],
                                          memory_order_relaxed);
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Exclusive prefix sum → bin_start ─────────
    // 32 bins = exactly one simdgroup
    {
        uint val = (simd_id == 0u) ? tile_hist[simd_lane] : 0u;
        uint pfx = simd_prefix_exclusive_sum(val);
        if (simd_id == 0u) {
            bin_start[simd_lane] = pfx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Round-by-round ballot rank + local sort ──
    if (lid < V8_NUM_BINS) {
        bin_ctr[lid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint below_me = (1u << simd_lane) - 1u;

    for (int e = 0; e < (int)V8_ELEMS; e++) {
        uint d_e = md[e];

        // Clear sg_hist — all 256 threads for 256 entries
        sg_hist[lid] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5-bit ballot rank
        uint match_e = 0xFFFFFFFFu;
        for (int b = 0; b < 5; b++) {
            uint bm = shuffle_ballot((d_e >> (uint)b) & 1u, simd_lane);
            match_e &= ((d_e >> (uint)b) & 1u) ? bm : ~bm;
        }
        uint active = shuffle_ballot(mv[e], simd_lane);
        match_e &= active;

        uint wsr = popcount(match_e & below_me);

        if (mv[e] && wsr == 0u) {
            sg_hist[simd_id * V8_NUM_BINS + d_e] = popcount(match_e);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint csg = 0u;
        if (mv[e]) {
            for (int s = 0; s < (int)simd_id; s++) {
                csg += sg_hist[(uint)s * V8_NUM_BINS + d_e];
            }
        }

        if (mv[e]) {
            uint rank = bin_start[d_e] + bin_ctr[d_e] + csg + wsr;
            lk[rank] = mk[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < V8_NUM_BINS) {
            uint rt = 0u;
            for (int s = 0; s < (int)V8_NUM_SGS; s++) {
                rt += sg_hist[(uint)s * V8_NUM_BINS + lid];
            }
            bin_ctr[lid] += rt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 5: Load global offsets ──────────────────────
    if (lid < V8_NUM_BINS) {
        uint te = tile_histograms[tile_id * V8_NUM_BINS + lid];
        uint ge = global_histogram[lid];
        local_off[lid] = ge + te;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 6: Coalesced scatter (8 per thread) ─────────
    for (int e = 0; e < (int)V8_ELEMS; e++) {
        uint li = (uint)e * 256u + lid;
        if (li < V8_TILE_SIZE && (base + li) < n) {
            uint k = lk[li];
            uint d = (k >> shift) & 0x1Fu;
            uint wbp = li - bin_start[d];
            uint gp = local_off[d] + wbp;
            dst[gp] = k;
        }
    }
}
