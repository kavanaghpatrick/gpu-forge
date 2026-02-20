#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ═══════════════════════════════════════════════════════════════════
// EXPERIMENT 26: 3-Pass Radix Sort (10 + 11 + 11 bits)
//
// Targets 5000+ Mkeys/s at 16M uint32 on Apple M4 Pro.
// 3-pass scatter ceiling: 5360 Mkeys/s (measured in exp17 Investigation P).
// 4-pass ceiling: 3746 Mkeys/s (cannot reach 5000).
//
// Per pass (4 dispatches):
//   1. Tile histogram — 256 threads, per-tile bin counts via TG atomics
//   2. Tile prefix    — in-place serial prefix across tiles per bin
//   3. Global prefix  — bin counts → exclusive prefix sums
//   4. Scatter        — 256 threads, stable ranking via SG-serialized dispatch
//
// Scatter stability design (256 threads, 8 SGs):
//   - tile_rank[MAX_BINS]: single running counter (8KB)
//   - For each iteration, process SGs 0..7 in order with barriers
//   - Within each SG: simd_broadcast gives deterministic lane-order ranking
//   - Cross-SG order guaranteed by serialized SG processing
//   - Only 16KB TG memory → high occupancy
//
// Total: 12 dispatches in a single compute encoder.
// Zero device-scope fences in any kernel.
//
// Pass layout:
//   Pass 0: 10-bit (1024 bins), shift= 0, mask=0x3FF
//   Pass 1: 11-bit (2048 bins), shift=10, mask=0x7FF
//   Pass 2: 11-bit (2048 bins), shift=21, mask=0x7FF
// ═══════════════════════════════════════════════════════════════════

#define E26_TILE_SIZE   4096u
#define E26_MAX_BINS    2048u
#define E26_TG_SIZE     256u
#define E26_ELEMS       16u     // E26_TILE_SIZE / E26_TG_SIZE
#define E26_NUM_SGS     8u

struct E26Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint num_bins;
    uint mask;
};

// ─── Kernel 1: Tile Histogram ───────────────────────────────────
kernel void exp26_tile_histogram(
    device const uint*   src        [[buffer(0)]],
    device uint*         tile_hists [[buffer(1)]],
    constant E26Params&  params     [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint n        = params.element_count;
    uint shift    = params.shift;
    uint mask_val = params.mask;
    uint num_bins = params.num_bins;
    uint tile_id  = gid;
    uint base     = tile_id * E26_TILE_SIZE;

    threadgroup atomic_uint hist[E26_MAX_BINS]; // 8 KB

    for (uint i = lid; i < num_bins; i += E26_TG_SIZE) {
        atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E26_ELEMS; e++) {
        uint idx = base + e * E26_TG_SIZE + lid;
        if (idx < n) {
            uint digit = (src[idx] >> shift) & mask_val;
            atomic_fetch_add_explicit(&hist[digit], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < num_bins; i += E26_TG_SIZE) {
        tile_hists[tile_id * num_bins + i] =
            atomic_load_explicit(&hist[i], memory_order_relaxed);
    }
}

// ─── Kernel 2: Tile Prefix Sum (in-place) + Global Histogram ───
kernel void exp26_tile_prefix(
    device uint*         tile_hists  [[buffer(0)]],
    device uint*         global_hist [[buffer(1)]],
    constant E26Params&  params      [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint nt       = params.num_tiles;
    uint num_bins = params.num_bins;
    uint bin      = gid;

    if (bin >= num_bins || lid != 0u) return;

    uint running = 0u;
    for (uint t = 0u; t < nt; t++) {
        uint addr  = t * num_bins + bin;
        uint count = tile_hists[addr];
        tile_hists[addr] = running;
        running += count;
    }
    global_hist[bin] = running;
}

// ─── Kernel 2b: Tile Prefix Sum PARALLEL (all 256 threads per bin) ──
//
// 256 threads cooperate per bin: each thread handles ~num_tiles/256 tiles.
// Phase 1: local partial sums → Phase 2: parallel prefix → Phase 3: write back.
// Reduces serial dependency chain from num_tiles to ~num_tiles/256 + log2(256).
//
kernel void exp26_tile_prefix_v2(
    device uint*         tile_hists  [[buffer(0)]],
    device uint*         global_hist [[buffer(1)]],
    constant E26Params&  params      [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint nt       = params.num_tiles;
    uint num_bins = params.num_bins;
    uint bin      = gid;

    if (bin >= num_bins) return;

    // ── Phase 1: each thread computes partial sum for its chunk ──
    uint tiles_per_thread = (nt + E26_TG_SIZE - 1u) / E26_TG_SIZE;
    uint my_start = lid * tiles_per_thread;
    uint my_end   = min(my_start + tiles_per_thread, nt);

    uint local_sum = 0u;
    for (uint t = my_start; t < my_end; t++) {
        local_sum += tile_hists[t * num_bins + bin];
    }

    // ── Phase 2: parallel exclusive prefix sum across threads ──
    threadgroup uint shared[E26_TG_SIZE];
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch-style up-sweep
    for (uint stride = 1u; stride < E26_TG_SIZE; stride *= 2u) {
        uint idx_s = (lid + 1u) * stride * 2u - 1u;
        if (idx_s < E26_TG_SIZE) {
            shared[idx_s] += shared[idx_s - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store total, clear last
    uint total = 0u;
    if (lid == 0u) {
        total = shared[E26_TG_SIZE - 1u];
        shared[E26_TG_SIZE - 1u] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint stride = E26_TG_SIZE / 2u; stride >= 1u; stride /= 2u) {
        uint idx_s = (lid + 1u) * stride * 2u - 1u;
        if (idx_s < E26_TG_SIZE) {
            uint tmp = shared[idx_s - stride];
            shared[idx_s - stride] = shared[idx_s];
            shared[idx_s] += tmp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint my_prefix = shared[lid]; // exclusive prefix for this thread's chunk

    // ── Phase 3: write back per-tile exclusive prefix ──
    uint running = my_prefix;
    for (uint t = my_start; t < my_end; t++) {
        uint addr  = t * num_bins + bin;
        uint count = tile_hists[addr];
        tile_hists[addr] = running;
        running += count;
    }

    // Write global histogram (total count for this bin)
    if (lid == 0u) {
        global_hist[bin] = total;
    }
}

// ─── Kernel 2c: FUSED Histogram + Decoupled Lookback Prefix ─────
//
// Single kernel per pass: computes histogram, publishes tile count,
// does lookback to find cross-tile prefix, computes combined offsets.
// Eliminates tile_hists buffer entirely! Only needs tile_status buffer.
//
// tile_status layout per tile per bin (packed):
//   bits 31-30: status (0=not started, 1=local, 2=prefix)
//   bits 29-0:  value (local count or inclusive prefix)
//
// Uses device-scope atomic_thread_fence (proven in exp12).
// Dispatch: num_tiles TGs × 256 threads.
//
// Output: tile_offsets[tile * num_bins + bin] = global exclusive prefix for this tile+bin
//
#define E26_STATUS_MASK   0xC0000000u
#define E26_VALUE_MASK    0x3FFFFFFFu
#define E26_STATUS_LOCAL  0x40000000u
#define E26_STATUS_PREFIX 0x80000000u

kernel void exp26_fused_hist_prefix(
    device const uint*   src          [[buffer(0)]],
    device atomic_uint*  tile_status  [[buffer(1)]],  // [num_tiles × num_bins]
    device uint*         tile_offsets [[buffer(2)]],   // output: exclusive prefix per tile per bin
    device uint*         global_hist  [[buffer(3)]],   // output: total count per bin
    constant E26Params&  params       [[buffer(4)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    uint n        = params.element_count;
    uint shift    = params.shift;
    uint mask_val = params.mask;
    uint num_bins = params.num_bins;
    uint tile_id  = gid;
    uint base     = tile_id * E26_TILE_SIZE;

    threadgroup atomic_uint hist[E26_MAX_BINS]; // 8 KB
    threadgroup uint offsets[E26_MAX_BINS];     // 8 KB — combined offset per bin

    // ── Phase 1: Compute tile histogram (same as standalone kernel) ──
    for (uint i = lid; i < num_bins; i += E26_TG_SIZE) {
        atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < E26_ELEMS; e++) {
        uint idx = base + e * E26_TG_SIZE + lid;
        if (idx < n) {
            uint digit = (src[idx] >> shift) & mask_val;
            atomic_fetch_add_explicit(&hist[digit], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Publish local counts to tile_status ──
    for (uint i = lid; i < num_bins; i += E26_TG_SIZE) {
        uint count = atomic_load_explicit(&hist[i], memory_order_relaxed);
        atomic_store_explicit(
            &tile_status[tile_id * num_bins + i],
            E26_STATUS_LOCAL | (count & E26_VALUE_MASK),
            memory_order_relaxed);
    }
    // Device-scope fence ensures all tile_status writes are visible to other TGs
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Decoupled lookback per bin ──
    // Each thread handles a subset of bins. For each bin, look back
    // through previous tiles to compute exclusive prefix.
    for (uint i = lid; i < num_bins; i += E26_TG_SIZE) {
        uint my_count = atomic_load_explicit(&hist[i], memory_order_relaxed);
        uint exclusive_prefix = 0u;

        if (tile_id == 0u) {
            // First tile: prefix is 0, publish inclusive prefix = local count
            exclusive_prefix = 0u;
        } else {
            // Look back through previous tiles
            int look = (int)tile_id - 1;
            while (look >= 0) {
                uint val = atomic_load_explicit(
                    &tile_status[(uint)look * num_bins + i],
                    memory_order_relaxed);
                // Spin until status is non-zero
                while ((val & E26_STATUS_MASK) == 0u) {
                    // Re-read with fence
                    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                    val = atomic_load_explicit(
                        &tile_status[(uint)look * num_bins + i],
                        memory_order_relaxed);
                }
                if ((val & E26_STATUS_MASK) == E26_STATUS_PREFIX) {
                    // Found a prefix — we're done
                    exclusive_prefix += (val & E26_VALUE_MASK);
                    break;
                } else {
                    // Found a local count — add it and keep looking
                    exclusive_prefix += (val & E26_VALUE_MASK);
                    look--;
                }
            }
        }

        // Publish our inclusive prefix (exclusive + local count)
        uint inclusive = exclusive_prefix + my_count;
        atomic_store_explicit(
            &tile_status[tile_id * num_bins + i],
            E26_STATUS_PREFIX | (inclusive & E26_VALUE_MASK),
            memory_order_relaxed);

        offsets[i] = exclusive_prefix;

        // Last tile writes global histogram
        if (tile_id == params.num_tiles - 1u) {
            global_hist[i] = inclusive;
        }
    }
    // Fence to ensure prefix publications are visible
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Store offsets for scatter to read ──
    for (uint i = lid; i < num_bins; i += E26_TG_SIZE) {
        tile_offsets[tile_id * num_bins + i] = offsets[i];
    }
}

// ─── Kernel 3: Global Prefix Sum ────────────────────────────────
kernel void exp26_global_prefix(
    device uint*         global_hist [[buffer(0)]],
    constant E26Params&  params      [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (lid != 0u) return;

    uint num_bins = params.num_bins;
    uint running  = 0u;
    for (uint i = 0u; i < num_bins; i++) {
        uint val       = global_hist[i];
        global_hist[i] = running;
        running       += val;
    }
}

// ─── Kernel 4a: Scatter UNSTABLE (streaming, atomic ranking) ────────
//
// All 256 threads scatter in parallel via atomic_fetch_add.
// Streams from source (no register caching) to handle large tiles.
// NOT STABLE — used only to measure raw scatter bandwidth ceiling.
//
// TG memory: tile_rank (8KB) + cached_prefix (8KB) = 16 KB
// ─────────────────────────────────────────────────────────────────────

kernel void exp26_scatter(
    device const uint*   src            [[buffer(0)]],
    device uint*         dst            [[buffer(1)]],
    device const uint*   tile_prefix    [[buffer(2)]],
    device const uint*   global_prefix  [[buffer(3)]],
    constant E26Params&  params         [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]])
{
    uint n        = params.element_count;
    uint shift    = params.shift;
    uint mask_val = params.mask;
    uint num_bins = params.num_bins;
    uint tile_id  = gid;
    uint base     = tile_id * E26_TILE_SIZE;

    threadgroup atomic_uint tile_rank[E26_MAX_BINS];      // 8 KB — running counter
    threadgroup uint        cached_prefix[E26_MAX_BINS];  // 8 KB — combined offsets

    // ── Cache combined prefix + zero tile_rank ──
    for (uint i = lid; i < num_bins; i += E26_TG_SIZE) {
        cached_prefix[i] = global_prefix[i]
                         + tile_prefix[tile_id * num_bins + i];
        atomic_store_explicit(&tile_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Main loop: stream from source, all threads scatter in parallel ──
    for (uint e = 0u; e < E26_ELEMS; e++) {
        uint idx = base + e * E26_TG_SIZE + lid;
        if (idx < n) {
            uint key = src[idx];
            uint d = (key >> shift) & mask_val;
            uint rank = atomic_fetch_add_explicit(
                &tile_rank[d], 1u, memory_order_relaxed);
            dst[cached_prefix[d] + rank] = key;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─── Kernel 4b: Scatter STABLE (streaming, SG-serialized ranking) ───
//
// Streams from source. Uses simd_broadcast for within-SG ranking and
// serialized SG processing for cross-SG ordering.
//
// TG memory: tile_rank (8KB) + cached_prefix (8KB) = 16 KB
// ─────────────────────────────────────────────────────────────────────

kernel void exp26_scatter_stable(
    device const uint*   src            [[buffer(0)]],
    device uint*         dst            [[buffer(1)]],
    device const uint*   tile_prefix    [[buffer(2)]],
    device const uint*   global_prefix  [[buffer(3)]],
    constant E26Params&  params         [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n        = params.element_count;
    uint shift    = params.shift;
    uint mask_val = params.mask;
    uint num_bins = params.num_bins;
    uint tile_id  = gid;
    uint base     = tile_id * E26_TILE_SIZE;

    threadgroup atomic_uint tile_rank[E26_MAX_BINS];      // 8 KB — running counter
    threadgroup uint        cached_prefix[E26_MAX_BINS];  // 8 KB — combined offsets

    // ── Cache combined prefix + zero tile_rank ──
    for (uint i = lid; i < num_bins; i += E26_TG_SIZE) {
        cached_prefix[i] = global_prefix[i]
                         + tile_prefix[tile_id * num_bins + i];
        atomic_store_explicit(&tile_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Main loop: stream from source, SG-serialized scatter ──
    for (uint e = 0u; e < E26_ELEMS; e++) {
        uint idx = base + e * E26_TG_SIZE + lid;
        bool is_valid = idx < n;
        uint key = is_valid ? src[idx] : 0xFFFFFFFFu;
        uint d = is_valid ? ((key >> shift) & mask_val) : 0xFFFFFFFFu;

        // Within-SG rank via simd_broadcast (32 comparisons)
        uint within_sg_rank = 0u;
        uint sg_count = 0u;
        if (is_valid) {
            for (uint lane = 0u; lane < 32u; lane++) {
                uint other_d = simd_broadcast(d, lane);
                if (other_d == d) {
                    sg_count++;
                    if (lane < simd_lane) within_sg_rank++;
                }
            }
        }

        // Process SGs in order: SG 0, barrier, SG 1, barrier, ...
        for (uint sg = 0u; sg < E26_NUM_SGS; sg++) {
            if (simd_id == sg && is_valid) {
                uint base_rank = atomic_load_explicit(
                    &tile_rank[d], memory_order_relaxed);
                uint rank = base_rank + within_sg_rank;
                dst[cached_prefix[d] + rank] = key;

                if (within_sg_rank == 0u) {
                    atomic_fetch_add_explicit(
                        &tile_rank[d], sg_count,
                        memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
