#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 7: Single-Dispatch-Per-Pass Radix Sort
//
// FIRST persistent-kernel radix sort on Metal.
// Disproves Raph Levien's claim (KB #1495) that "Metal requires
// multi-dispatch tree reduction" for single-pass prefix sums.
//
// 8-bit radix (256 bins), 4 LSD passes for 32-bit keys.
// Version A: 2 dispatches per pass × 4 = 8 total (histogram + scatter)
// Version B: 1 persistent dispatch per pass × 4 = 4 total
//            (fuses histogram → 256 parallel lookbacks → scatter)
//
// Key insight: TILE_SIZE=256 = NUM_BINS, so each thread handles
// exactly one bin's decoupled lookback. Zero idle threads.
//
// ⚠️  DEADLOCK HAZARD: Phase 3 requires ALL threadgroups simultaneously
// resident. If NUM_TILES exceeds the GPU's concurrent TG capacity
// (~60 on M4 Pro), early tiles spin-wait for the last tile which
// never gets scheduled → GPU deadlock → frozen machine.
// Host code MUST cap N so NUM_TILES ≤ concurrent TG limit.
// For production use: occupancy-bound dispatch with tile loops.
// ══════════════════════════════════════════════════════════════════════

#define NUM_BINS 256u

struct RadixParams {
    uint element_count;
    uint num_tiles;
    uint shift;    // 0, 8, 16, or 24
};

// ── Stable rank helper ─────────────────────────────────────────────
// For LSD radix sort, each pass MUST be stable (preserve prior order).
// Computes: how many threads with lid' < lid have the same digit.
// O(TILE_SIZE) per thread — ~0.2us for 256 threads.

inline uint stable_rank_of(uint lid, uint digit,
                           threadgroup uint* shared_digits)
{
    uint rank = 0;
    for (uint i = 0; i < lid; i++) {
        rank += (shared_digits[i] == digit) ? 1u : 0u;
    }
    return rank;
}

// ── Version A: Multi-dispatch building blocks ──────────────────────

kernel void exp7_histogram(
    device const uint* input          [[buffer(0)]],
    device uint* tile_histograms      [[buffer(1)]],
    constant RadixParams& params      [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint local_hist[NUM_BINS];
    atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < params.element_count) {
        uint digit = (input[tid] >> params.shift) & 0xFFu;
        atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tile_histograms[gid * NUM_BINS + lid] =
        atomic_load_explicit(&local_hist[lid], memory_order_relaxed);
}

kernel void exp7_scatter(
    device const uint* input          [[buffer(0)]],
    device uint* output               [[buffer(1)]],
    device const uint* offsets        [[buffer(2)]],
    constant RadixParams& params      [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    // Store digits for stable rank computation
    threadgroup uint shared_digits[TILE_SIZE];

    uint element = 0u;
    uint digit = 0u;
    bool valid = (tid < params.element_count);

    if (valid) {
        element = input[tid];
        digit = (element >> params.shift) & 0xFFu;
    }
    shared_digits[lid] = valid ? digit : 0xFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        uint rank = stable_rank_of(lid, digit, shared_digits);
        output[offsets[gid * NUM_BINS + digit] + rank] = element;
    }
}

// ── Version B: Persistent single dispatch per pass ─────────────────

kernel void exp7_radix_persistent(
    device const uint* input          [[buffer(0)]],
    device uint* output               [[buffer(1)]],
    device atomic_uint* tile_status   [[buffer(2)]],
    constant RadixParams& params      [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint num_tiles = params.num_tiles;

    // ═══ Phase 1: Histogram + stable rank ═══
    threadgroup atomic_uint local_hist[NUM_BINS];
    threadgroup uint shared_digits[TILE_SIZE];

    atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint element = 0u;
    uint digit = 0u;
    bool valid = (tid < params.element_count);

    if (valid) {
        element = input[tid];
        digit = (element >> params.shift) & 0xFFu;
        atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed);
    }
    shared_digits[lid] = valid ? digit : 0xFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute stable rank while histogram settles
    uint rank = valid ? stable_rank_of(lid, digit, shared_digits) : 0u;

    // ═══ Phase 2: 256 parallel decoupled lookbacks ═══
    // Thread `lid` handles lookback for bin `lid`.
    // Decoupled lookback is safe even with non-concurrent tiles because
    // dependencies are strictly backward (tile N depends only on tiles 0..N-1).
    uint my_bin_count = atomic_load_explicit(&local_hist[lid], memory_order_relaxed);

    // Publish aggregate
    uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT) | (my_bin_count & VALUE_MASK);
    atomic_store_explicit(&tile_status[gid * NUM_BINS + lid],
                          packed_agg, memory_order_relaxed);

    threadgroup uint bin_exclusive[NUM_BINS];

    if (gid == 0u) {
        uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (my_bin_count & VALUE_MASK);
        atomic_store_explicit(&tile_status[lid], packed_pfx, memory_order_relaxed);
        bin_exclusive[lid] = 0u;
    } else {
        uint running = 0u;
        int lb = (int)gid - 1;
        while (lb >= 0) {
            uint status = atomic_load_explicit(
                &tile_status[lb * NUM_BINS + lid], memory_order_relaxed);
            uint flag  = status >> FLAG_SHIFT;
            uint value = status & VALUE_MASK;

            if (flag == FLAG_PREFIX) {
                running += value;
                break;
            } else if (flag == FLAG_AGGREGATE) {
                running += value;
                lb--;
            }
            // FLAG_NOT_READY: spin
        }
        bin_exclusive[lid] = running;

        uint inclusive = running + my_bin_count;
        uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK);
        atomic_store_explicit(&tile_status[gid * NUM_BINS + lid],
                              packed_pfx, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══ Phase 3: Get global totals from last tile's inclusive prefix ═══
    //
    // The last tile's inclusive prefix for each bin = global total for that bin.
    // We read it directly from tile_status (an atomic), avoiding non-atomic
    // device memory entirely. MSL only supports memory_order_relaxed, so we
    // CANNOT safely synchronize non-atomic cross-TG writes — this approach
    // sidesteps the problem completely.
    //
    // ⚠️  DEADLOCK HAZARD: Non-last tiles spin on the last tile's status.
    // This requires ALL tiles to be concurrently resident (the last tile's
    // lookback must be able to complete while other tiles spin).
    // Host code MUST cap NUM_TILES ≤ GPU's concurrent TG capacity.
    threadgroup uint global_total[NUM_BINS];

    if (gid == num_tiles - 1u) {
        // Last tile already has the totals from its own lookback
        global_total[lid] = bin_exclusive[lid] + my_bin_count;
    } else {
        // Spin until last tile publishes FLAG_PREFIX for our bin
        uint status;
        do {
            status = atomic_load_explicit(
                &tile_status[(num_tiles - 1u) * NUM_BINS + lid],
                memory_order_relaxed);
        } while ((status >> FLAG_SHIFT) != FLAG_PREFIX);
        // The inclusive prefix of the last tile IS the global total
        global_total[lid] = status & VALUE_MASK;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══ Phase 4: Exclusive prefix of global totals → bin starts ═══
    // global_total[b] = count of elements with digit b across all tiles.
    // global_bin_start[b] = sum of global_total[0..b-1] = where bin b starts.
    threadgroup uint global_bin_start[NUM_BINS];

    // SIMD-based exclusive prefix sum (same technique as exp4)
    uint scan_val = global_total[lid];
    uint simd_scan = simd_prefix_inclusive_sum(scan_val);

    threadgroup uint sg_totals[8];
    if (simd_lane == 31u)
        sg_totals[simd_id] = simd_scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u && simd_lane < 8u) {
        uint sg = sg_totals[simd_lane];
        uint sg_scan = simd_prefix_inclusive_sum(sg);
        sg_totals[simd_lane] = sg_scan;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint inc = simd_scan;
    if (simd_id > 0u) inc += sg_totals[simd_id - 1u];

    // inclusive → exclusive: subtract own value
    global_bin_start[lid] = inc - scan_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══ Phase 5: Scatter ═══
    if (valid) {
        uint pos = global_bin_start[digit] + bin_exclusive[digit] + rank;
        output[pos] = element;
    }
}
