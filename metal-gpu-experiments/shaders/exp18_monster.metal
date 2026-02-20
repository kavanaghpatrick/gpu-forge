#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 18: Monster 3-Pass Radix Sort (5000+ Mkeys/s target)
//
// Config: 11+11+10 bits (passes sort bits 0-10, 11-21, 22-31)
// Architecture: Onesweep-style decoupled lookback, non-persistent dispatch.
// Histogram: TG-wide atomics (contention ~2/bin at 2048 = same as 256-bin/SG).
// Scatter: Sequential SG scatter with SG-contiguous load layout.
// Within-SG atomics are lane-ordered on Apple Silicon → stable sort.
// ═══════════════════════════════════════════════════════════════════

#define EXP18_TILE_SIZE  4096u
#define EXP18_ELEMS      16u
#define EXP18_THREADS    256u

// Pass configurations
#define EXP18_BINS_P0    2048u   // 11-bit, bits 0-10
#define EXP18_BINS_P1    2048u   // 11-bit, bits 11-21
#define EXP18_BINS_P2    1024u   // 10-bit, bits 22-31
#define EXP18_MAX_BINS   2048u   // max across passes
#define EXP18_TOTAL_BINS (EXP18_BINS_P0 + EXP18_BINS_P1 + EXP18_BINS_P2)  // 5120

struct Exp18Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint num_bins;    // 1024 or 2048
    uint pass;        // 0, 1, 2
};

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Combined Histogram — reads ALL data once, computes
// histograms for all 3 passes. Uses TG-wide atomics (8 KB max).
// Processes passes sequentially to reuse TG memory.
// Output: global_hist[pass_offset + digit] = total count
// ═══════════════════════════════════════════════════════════════════

kernel void exp18_combined_histogram(
    device const uint*     src          [[buffer(0)]],
    device atomic_uint*    global_hist  [[buffer(1)]],
    constant uint&         element_count [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint n = element_count;
    uint base = gid * EXP18_TILE_SIZE;

    // Load elements once into registers
    uint keys[EXP18_ELEMS];
    bool valid[EXP18_ELEMS];
    for (uint e = 0u; e < EXP18_ELEMS; e++) {
        uint idx = base + e * EXP18_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    threadgroup atomic_uint tg_hist[EXP18_MAX_BINS]; // 8 KB

    // ── Pass 0: 2048 bins (bits 0-10) ────────────────────────────
    for (uint i = lid; i < EXP18_BINS_P0; i += EXP18_THREADS) {
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP18_ELEMS; e++) {
        if (valid[e]) {
            atomic_fetch_add_explicit(
                &tg_hist[keys[e] & 0x7FFu], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < EXP18_BINS_P0; i += EXP18_THREADS) {
        uint c = atomic_load_explicit(&tg_hist[i], memory_order_relaxed);
        if (c > 0u) {
            atomic_fetch_add_explicit(&global_hist[i], c, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Pass 1: 2048 bins (bits 11-21) ───────────────────────────
    for (uint i = lid; i < EXP18_BINS_P1; i += EXP18_THREADS) {
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP18_ELEMS; e++) {
        if (valid[e]) {
            atomic_fetch_add_explicit(
                &tg_hist[(keys[e] >> 11u) & 0x7FFu], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < EXP18_BINS_P1; i += EXP18_THREADS) {
        uint c = atomic_load_explicit(&tg_hist[i], memory_order_relaxed);
        if (c > 0u) {
            atomic_fetch_add_explicit(
                &global_hist[EXP18_BINS_P0 + i], c, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Pass 2: 1024 bins (bits 22-31) ───────────────────────────
    for (uint i = lid; i < EXP18_BINS_P2; i += EXP18_THREADS) {
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP18_ELEMS; e++) {
        if (valid[e]) {
            atomic_fetch_add_explicit(
                &tg_hist[(keys[e] >> 22u) & 0x3FFu], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < EXP18_BINS_P2; i += EXP18_THREADS) {
        uint c = atomic_load_explicit(&tg_hist[i], memory_order_relaxed);
        if (c > 0u) {
            atomic_fetch_add_explicit(
                &global_hist[EXP18_BINS_P0 + EXP18_BINS_P1 + i],
                c, memory_order_relaxed);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Global Prefix — exclusive prefix sum for one pass.
// Variable bin count (1024 or 2048). Uses simd_prefix_exclusive_sum
// in chunks of 32. Single SG does all work (serial across chunks).
// Input/output: global_hist[offset..offset+num_bins] (in-place).
// ═══════════════════════════════════════════════════════════════════

kernel void exp18_global_prefix(
    device uint*         global_hist  [[buffer(0)]],
    constant uint&       pass_offset  [[buffer(1)]],
    constant uint&       num_bins     [[buffer(2)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    if (simd_id > 0u) return;  // Only SG 0

    uint chunks = num_bins / 32u;  // 32 or 64
    uint running = 0u;

    for (uint chunk = 0u; chunk < chunks; chunk++) {
        uint bin = chunk * 32u + simd_lane;
        uint val = global_hist[pass_offset + bin];
        uint prefix = simd_prefix_exclusive_sum(val) + running;
        global_hist[pass_offset + bin] = prefix;
        running += simd_shuffle(simd_prefix_exclusive_sum(val) + val, 31u);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Zero tile_status between passes
// ═══════════════════════════════════════════════════════════════════

kernel void exp18_zero_status(
    device uint*           tile_status  [[buffer(0)]],
    constant uint&         total_entries [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < total_entries) {
        tile_status[tid] = 0u;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: Partition — fused histogram + lookback + scatter
//
// Parameterized for variable bin counts (1024 or 2048).
// Histogram: TG-wide atomics (~2 contention/bin at 2048).
// Lookback: Each thread handles bpt bins (interleaved assignment).
// Scatter: Sequential SG (one SG at a time, lane-ordered atomics).
// SG-contiguous load: SG i loads [base+i*512..base+i*512+511].
//
// TG memory: 3 × num_bins × 4B = 24 KB max (at 2048 bins).
// ═══════════════════════════════════════════════════════════════════

kernel void exp18_partition(
    device const uint*     src          [[buffer(0)]],
    device uint*           dst          [[buffer(1)]],
    device atomic_uint*    tile_status  [[buffer(2)]],
    device const uint*     global_hist  [[buffer(3)]],
    constant Exp18Params&  params       [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n         = params.element_count;
    uint shift     = params.shift;
    uint num_bins  = params.num_bins;
    uint mask      = num_bins - 1u;
    uint bpt       = num_bins / EXP18_THREADS;  // bins per thread: 4 or 8

    uint tile_id = gid;
    uint base = tile_id * EXP18_TILE_SIZE;

    // ── TG Memory (24 KB max) ────────────────────────────────────
    threadgroup atomic_uint tg_hist[EXP18_MAX_BINS];       // 8 KB
    threadgroup uint        tile_hist[EXP18_MAX_BINS];     // 8 KB
    threadgroup uint        exclusive_pfx[EXP18_MAX_BINS]; // 8 KB

    // ── Phase 1: Load elements (SG-contiguous layout) ────────────
    // Each SG gets a contiguous 512-element block for stable scatter.
    // SG 0: [base+0..base+511], SG 1: [base+512..base+1023], etc.
    // Within each SG, 32 lanes × 16 elements with stride-32 (coalesced).
    uint mk[EXP18_ELEMS];
    uint md[EXP18_ELEMS];
    bool mv[EXP18_ELEMS];
    for (uint e = 0u; e < EXP18_ELEMS; e++) {
        uint idx = base + simd_id * (EXP18_ELEMS * 32u) + e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & mask) : mask;
    }

    // ── Phase 2: TG-wide atomic histogram ────────────────────────
    for (uint i = lid; i < num_bins; i += EXP18_THREADS) {
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP18_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(
                &tg_hist[md[e]], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Copy histogram to non-atomic tile_hist ─────────
    for (uint b = 0u; b < bpt; b++) {
        uint bin = lid + b * EXP18_THREADS;
        tile_hist[bin] = atomic_load_explicit(
            &tg_hist[bin], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Publish AGGREGATE ───────────────────────────────
    for (uint b = 0u; b < bpt; b++) {
        uint bin = lid + b * EXP18_THREADS;
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT)
                    | (tile_hist[bin] & VALUE_MASK);
        atomic_store_explicit(
            &tile_status[tile_id * num_bins + bin],
            packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Decoupled lookback (interleaved multi-bin) ──────
    // Each thread handles bpt bins simultaneously using interleaved
    // lookback to avoid serializing on one bin's slow predecessor.
    {
        uint running[8];   // max bpt = 8
        int  look[8];
        bool done[8];

        for (uint b = 0u; b < bpt; b++) {
            running[b] = 0u;
            look[b] = (int)tile_id - 1;
            done[b] = (tile_id == 0u);
        }

        bool all_done = (tile_id == 0u);
        while (!all_done) {
            atomic_thread_fence(mem_flags::mem_device,
                                memory_order_seq_cst,
                                thread_scope_device);
            all_done = true;
            for (uint b = 0u; b < bpt; b++) {
                if (done[b]) continue;
                uint bin = lid + b * EXP18_THREADS;
                uint val = atomic_load_explicit(
                    &tile_status[(uint)look[b] * num_bins + bin],
                    memory_order_relaxed);
                uint flag  = val >> FLAG_SHIFT;
                uint count = val & VALUE_MASK;

                if (flag == FLAG_PREFIX) {
                    running[b] += count;
                    done[b] = true;
                } else if (flag == FLAG_AGGREGATE) {
                    running[b] += count;
                    look[b]--;
                    if (look[b] < 0) {
                        done[b] = true;
                    } else {
                        all_done = false;
                    }
                } else {
                    // FLAG_NOT_READY: keep spinning
                    all_done = false;
                }
            }
        }

        // Store results and publish PREFIX
        for (uint b = 0u; b < bpt; b++) {
            uint bin = lid + b * EXP18_THREADS;
            exclusive_pfx[bin] = running[b];
            uint inclusive = running[b] + tile_hist[bin];
            uint packed = (FLAG_PREFIX << FLAG_SHIFT)
                        | (inclusive & VALUE_MASK);
            atomic_store_explicit(
                &tile_status[tile_id * num_bins + bin],
                packed, memory_order_relaxed);
        }
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Sequential SG scatter (stable ranking) ──────────
    // SG-contiguous load means SG 0's elements precede SG 1's in
    // input order. We scatter one SG at a time; within-SG atomics
    // are lane-ordered on Apple Silicon → stable output.
    //
    // Reuse tile_hist as cross_sg_prefix (no longer needed after P4)
    // Reuse tg_hist as per-SG rank counter
    uint pass_offset = params.pass == 0u ? 0u
                     : params.pass == 1u ? EXP18_BINS_P0
                     : (EXP18_BINS_P0 + EXP18_BINS_P1);

    // Zero cross-SG prefix
    for (uint i = lid; i < num_bins; i += EXP18_THREADS) {
        tile_hist[i] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint sg = 0u; sg < 8u; sg++) {
        // Zero rank counters for this SG
        for (uint i = lid; i < num_bins; i += EXP18_THREADS) {
            atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Only threads in current SG scatter their elements
        if (simd_id == sg) {
            for (uint e = 0u; e < EXP18_ELEMS; e++) {
                if (mv[e]) {
                    uint d = md[e];
                    uint rank = atomic_fetch_add_explicit(
                        &tg_hist[d], 1u, memory_order_relaxed);
                    uint gp = global_hist[pass_offset + d]
                             + exclusive_pfx[d]
                             + tile_hist[d]
                             + rank;
                    dst[gp] = mk[e];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update cross-SG prefix (all threads cooperate)
        for (uint b = 0u; b < bpt; b++) {
            uint bin = lid + b * EXP18_THREADS;
            tile_hist[bin] += atomic_load_explicit(
                &tg_hist[bin], memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
