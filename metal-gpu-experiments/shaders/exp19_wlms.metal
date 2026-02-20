#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 19: WLMS 3-Pass Radix Sort (5000+ Mkeys/s target)
//
// Key innovation: WLMS (Warp-Level Multi-Split) ballot ranking
// enables ALL 4 SGs to scatter in parallel (vs sequential in exp18).
//
// Config: 11+11+10 bits (2048/2048/1024 bins)
// 128 threads (4 SGs), 32 elements/lane, 4096-element tiles.
// TG memory: 24KB (16KB _block + 8KB exclusive_pfx).
// ═══════════════════════════════════════════════════════════════════

#define EXP19_TILE_SIZE  8192u
#define EXP19_ELEMS      64u
#define EXP19_THREADS    128u
#define EXP19_NUM_SGS    4u
#define EXP19_MAX_BINS   2048u

#define EXP19_BINS_P0    2048u
#define EXP19_BINS_P1    2048u
#define EXP19_BINS_P2    1024u
#define EXP19_TOTAL_BINS (EXP19_BINS_P0 + EXP19_BINS_P1 + EXP19_BINS_P2)

// Manual ballot construction via butterfly reduction (simd_ballot returns
// opaque simd_vote on Metal — cannot extract bitmask). Uses 5 uniform
// simd_shuffle_xor ops to broadcast each lane's bit to all lanes.
inline uint manual_ballot(bool pred, uint lane) {
    uint val = pred ? (1u << lane) : 0u;
    val |= simd_shuffle_xor(val, 1u);
    val |= simd_shuffle_xor(val, 2u);
    val |= simd_shuffle_xor(val, 4u);
    val |= simd_shuffle_xor(val, 8u);
    val |= simd_shuffle_xor(val, 16u);
    return val;
}

struct Exp19Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint num_bins;
    uint pass;
};

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Combined Histogram — 128 threads, 32 elements/thread
// TG-wide atomics, processes all 3 passes sequentially.
// ═══════════════════════════════════════════════════════════════════

kernel void exp19_combined_histogram(
    device const uint*     src            [[buffer(0)]],
    device atomic_uint*    global_hist    [[buffer(1)]],
    constant uint&         element_count  [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint n = element_count;
    uint base = gid * EXP19_TILE_SIZE;

    uint keys[EXP19_ELEMS];
    bool valid[EXP19_ELEMS];
    for (uint e = 0u; e < EXP19_ELEMS; e++) {
        uint idx = base + e * EXP19_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    threadgroup atomic_uint tg_hist[EXP19_MAX_BINS]; // 8KB

    // ── Pass 0: 2048 bins (bits 0-10) ────────────────────────────
    for (uint i = lid; i < EXP19_BINS_P0; i += EXP19_THREADS)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP19_ELEMS; e++) {
        if (valid[e])
            atomic_fetch_add_explicit(&tg_hist[keys[e] & 0x7FFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < EXP19_BINS_P0; i += EXP19_THREADS) {
        uint c = atomic_load_explicit(&tg_hist[i], memory_order_relaxed);
        if (c > 0u)
            atomic_fetch_add_explicit(&global_hist[i], c, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Pass 1: 2048 bins (bits 11-21) ───────────────────────────
    for (uint i = lid; i < EXP19_BINS_P1; i += EXP19_THREADS)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP19_ELEMS; e++) {
        if (valid[e])
            atomic_fetch_add_explicit(&tg_hist[(keys[e] >> 11u) & 0x7FFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < EXP19_BINS_P1; i += EXP19_THREADS) {
        uint c = atomic_load_explicit(&tg_hist[i], memory_order_relaxed);
        if (c > 0u)
            atomic_fetch_add_explicit(&global_hist[EXP19_BINS_P0 + i], c, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Pass 2: 1024 bins (bits 22-31) ───────────────────────────
    for (uint i = lid; i < EXP19_BINS_P2; i += EXP19_THREADS)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP19_ELEMS; e++) {
        if (valid[e])
            atomic_fetch_add_explicit(&tg_hist[(keys[e] >> 22u) & 0x3FFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < EXP19_BINS_P2; i += EXP19_THREADS) {
        uint c = atomic_load_explicit(&tg_hist[i], memory_order_relaxed);
        if (c > 0u)
            atomic_fetch_add_explicit(&global_hist[EXP19_BINS_P0 + EXP19_BINS_P1 + i], c, memory_order_relaxed);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Global Prefix — exclusive prefix sum for one pass.
// SG 0 only, variable bin count (1024 or 2048).
// ═══════════════════════════════════════════════════════════════════

kernel void exp19_global_prefix(
    device uint*         global_hist  [[buffer(0)]],
    constant uint&       pass_offset  [[buffer(1)]],
    constant uint&       num_bins     [[buffer(2)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    if (simd_id > 0u) return;

    uint chunks = num_bins / 32u;
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

kernel void exp19_zero_status(
    device uint*       tile_status    [[buffer(0)]],
    constant uint&     total_entries  [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < total_entries)
        tile_status[tid] = 0u;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: WLMS Partition — the key innovation
//
// 128 threads (4 SGs), WLMS ballot ranking, all SGs scatter in parallel.
//
// TG memory layout (24KB):
//   _block[4096] as atomic_uint = 16KB
//     Phase 2: [0..num_bins) = TG-wide atomic histogram
//     Phase 5: recast as ushort* sg_running[4 * num_bins]
//   exclusive_pfx[2048] as uint = 8KB
//     Phase 4-5: lookback results
//
// WLMS algorithm: O(radix_bits) simd_ballot calls per element.
// Builds peer_mask identifying all lanes with same digit.
// Leader (lowest-index peer) updates running count in TG memory.
// ═══════════════════════════════════════════════════════════════════

kernel void exp19_partition(
    device const uint*     src          [[buffer(0)]],
    device uint*           dst          [[buffer(1)]],
    device atomic_uint*    tile_status  [[buffer(2)]],
    device const uint*     global_hist  [[buffer(3)]],
    constant Exp19Params&  params       [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n         = params.element_count;
    uint shift     = params.shift;
    uint num_bins  = params.num_bins;
    uint mask      = num_bins - 1u;
    uint bpt       = num_bins / EXP19_THREADS;  // bins per thread: 8 or 16
    uint radix_bits = 31u - clz(num_bins);      // 11 for 2048, 10 for 1024

    uint tile_id = gid;
    uint base = tile_id * EXP19_TILE_SIZE;

    // Pass offset for global_hist
    uint pass_offset = params.pass == 0u ? 0u
                     : params.pass == 1u ? EXP19_BINS_P0
                     : (EXP19_BINS_P0 + EXP19_BINS_P1);

    // ── TG Memory (24.5KB) ───────────────────────────────────────
    threadgroup atomic_uint _block[4096];    // 16KB: histogram then sg_running
    threadgroup uint exclusive_pfx[2048];    // 8KB: lookback results
    threadgroup uint _sg_digits[128];        // 512B: per-SG digit broadcast (4×32)

    // ── Phase 1: Load 32 elements (SG-contiguous) ────────────────
    // SG 0: [base+0..1023], SG 1: [base+1024..2047], etc.
    uint mk[EXP19_ELEMS];
    uint md[EXP19_ELEMS];
    bool mv[EXP19_ELEMS];
    for (uint e = 0u; e < EXP19_ELEMS; e++) {
        uint idx = base + simd_id * (EXP19_ELEMS * 32u) + e * 32u + simd_lane;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & mask) : mask;
    }

    // ── Phase 2: TG-wide atomic histogram ────────────────────────
    for (uint i = lid; i < num_bins; i += EXP19_THREADS)
        atomic_store_explicit(&_block[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < EXP19_ELEMS; e++) {
        if (mv[e])
            atomic_fetch_add_explicit(&_block[md[e]], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Read tile histogram into registers ─────────────
    uint my_tile_hist[16];  // max bpt = 16
    for (uint b = 0u; b < bpt; b++) {
        uint bin = lid + b * EXP19_THREADS;
        my_tile_hist[b] = atomic_load_explicit(&_block[bin], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Publish AGGREGATE ───────────────────────────────
    for (uint b = 0u; b < bpt; b++) {
        uint bin = lid + b * EXP19_THREADS;
        uint packed = (FLAG_AGGREGATE << FLAG_SHIFT) | (my_tile_hist[b] & VALUE_MASK);
        atomic_store_explicit(&tile_status[tile_id * num_bins + bin],
                              packed, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Decoupled lookback (interleaved multi-bin) ──────
    {
        uint running[16];
        int  look[16];
        bool done[16];

        for (uint b = 0u; b < bpt; b++) {
            running[b] = 0u;
            look[b] = (int)tile_id - 1;
            done[b] = (tile_id == 0u);
        }

        bool all_done = (tile_id == 0u);
        while (!all_done) {
            atomic_thread_fence(mem_flags::mem_device,
                                memory_order_seq_cst, thread_scope_device);
            all_done = true;
            for (uint b = 0u; b < bpt; b++) {
                if (done[b]) continue;
                uint bin = lid + b * EXP19_THREADS;
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
                    if (look[b] < 0) done[b] = true;
                    else all_done = false;
                } else {
                    all_done = false;  // FLAG_NOT_READY: spin
                }
            }
        }

        // Store exclusive_pfx and publish PREFIX
        for (uint b = 0u; b < bpt; b++) {
            uint bin = lid + b * EXP19_THREADS;
            exclusive_pfx[bin] = running[b];
            uint inclusive = running[b] + my_tile_hist[b];
            uint packed = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK);
            atomic_store_explicit(&tile_status[tile_id * num_bins + bin],
                                  packed, memory_order_relaxed);
        }
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: WLMS Ranking + Parallel Scatter ─────────────────
    // Reuse _block as sg_running (ushort view, 16KB = 8192 ushorts)
    threadgroup ushort* sg_running = (threadgroup ushort*)((threadgroup void*)&_block[0]);

    // Zero sg_running
    for (uint i = lid; i < EXP19_NUM_SGS * num_bins; i += EXP19_THREADS)
        sg_running[i] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5b: Shared-memory ranking (32 elements per lane) ───
    // Each SG broadcasts digits via _sg_digits, then each lane scans
    // 31 other lanes to compute rank. O(32) TG reads per element
    // (1 cycle each) vs manual_ballot's O(60) multi-cycle shuffles.
    uint ranks[EXP19_ELEMS];

    for (uint e = 0u; e < EXP19_ELEMS; e++) {
        uint digit = md[e];
        bool is_valid = mv[e];

        // Broadcast digit to SG-local shared memory
        // Invalid lanes write num_bins (> any valid digit 0..num_bins-1)
        _sg_digits[simd_id * 32u + simd_lane] = is_valid ? digit : num_bins;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (is_valid) {
            uint rank = 0u;
            uint peer_count = 0u;
            for (uint i = 0u; i < 32u; i++) {
                if (_sg_digits[simd_id * 32u + i] == digit) {
                    peer_count++;
                    if (i < simd_lane) rank++;
                }
            }
            uint running = (uint)sg_running[simd_id * num_bins + digit];
            ranks[e] = running + rank;
            if (rank == 0u) {
                sg_running[simd_id * num_bins + digit] =
                    (ushort)(running + peer_count);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5c: Cross-SG prefix + scatter ──────────────────────
    for (uint e = 0u; e < EXP19_ELEMS; e++) {
        if (!mv[e]) continue;
        uint digit = md[e];

        // Sum counts from preceding SGs
        uint cross_rank = 0u;
        for (uint s = 0u; s < simd_id; s++) {
            cross_rank += (uint)sg_running[s * num_bins + digit];
        }

        // Final global position
        uint gpos = global_hist[pass_offset + digit]
                  + exclusive_pfx[digit]
                  + cross_rank
                  + ranks[e];
        dst[gpos] = mk[e];
    }
}
