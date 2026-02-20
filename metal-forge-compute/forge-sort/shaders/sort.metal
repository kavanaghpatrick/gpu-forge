#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════
// forge-sort: MSD+fused-inner radix sort (4 dispatches)
//
// Extracted from exp17 Investigation W.
// Architecture: 1 MSD scatter (bits 24:31) → 256 buckets
// → 3 inner LSD passes per bucket at SLC speed.
// Single encoder, 4 dispatches, zero CPU readback.
// ═══════════════════════════════════════════════════════════════════

#define SORT_NUM_BINS  256u
#define SORT_TILE_SIZE 4096u
#define SORT_ELEMS     16u
#define SORT_THREADS   256u
#define SORT_NUM_SGS   8u
#define SORT_MAX_TPB   17u

struct SortParams {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint pass;
};

struct BucketDesc {
    uint offset;
    uint count;
    uint tile_count;
    uint tile_base;
};

struct InnerParams {
    uint start_shift;
    uint pass_count;
    uint batch_start;
};

// ═══════════════════════════════════════════════════════════════════
// Function constants — specialized at PSO creation time
// Defaults ensure existing kernels are unaffected when unset.
// ═══════════════════════════════════════════════════════════════════

constant bool HAS_VALUES [[function_constant(0)]];
constant bool IS_64BIT   [[function_constant(1)]];
constant uint TRANSFORM_MODE [[function_constant(2)]];
constant bool has_values = is_function_constant_defined(HAS_VALUES) ? HAS_VALUES : false;
constant bool is_64bit   = is_function_constant_defined(IS_64BIT)   ? IS_64BIT   : false;
constant uint transform_mode = is_function_constant_defined(TRANSFORM_MODE) ? TRANSFORM_MODE : 0u;

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: MSD Histogram — single-pass, bits[24:31]
//
// Reads ALL data once, computes 256-bin histogram for MSD byte only.
// Uses per-SG atomic histogram on TG memory.
// Output: global_hist[digit] = total count for digit 0..255
// ═══════════════════════════════════════════════════════════════════

kernel void sort_msd_histogram(
    device const uint*     src          [[buffer(0)]],
    device atomic_uint*    global_hist  [[buffer(1)]],
    constant SortParams&   params       [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint n = params.element_count;
    uint shift = params.shift;
    uint base = gid * SORT_TILE_SIZE;

    // Load 16 elements into registers (one global memory read)
    uint keys[SORT_ELEMS];
    bool valid[SORT_ELEMS];
    for (uint e = 0u; e < SORT_ELEMS; e++) {
        uint idx = base + e * SORT_THREADS + lid;
        valid[e] = idx < n;
        keys[e] = valid[e] ? src[idx] : 0u;
    }

    // Per-SG accumulator in shared memory
    threadgroup atomic_uint sg_counts[SORT_NUM_SGS * SORT_NUM_BINS]; // 8 KB

    // Zero sg_counts (all 256 threads cooperate)
    for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-SG atomic histogram (single pass: bits[24:31])
    for (uint e = 0u; e < SORT_ELEMS; e++) {
        if (valid[e]) {
            uint digit = (keys[e] >> shift) & 0xFFu;
            atomic_fetch_add_explicit(
                &sg_counts[simd_id * SORT_NUM_BINS + digit],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SGs: 256 threads handle one bin each
    {
        uint total = 0u;
        for (uint sg = 0u; sg < SORT_NUM_SGS; sg++) {
            total += atomic_load_explicit(
                &sg_counts[sg * SORT_NUM_BINS + lid],
                memory_order_relaxed);
        }
        if (total > 0u) {
            atomic_fetch_add_explicit(&global_hist[lid],
                                      total, memory_order_relaxed);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: MSD Prep — combined prefix sum + bucket descs
//
// Single dispatch, 1 TG, 256 threads.
// Input:  global_hist[256] = per-digit counts
// Output: counters[256]    = exclusive prefix (for atomic scatter)
//         bucket_descs[256] = offset/count/tile_count for inner sort
// ═══════════════════════════════════════════════════════════════════

kernel void sort_msd_prep(
    device const uint*     global_hist  [[buffer(0)]],
    device uint*           counters     [[buffer(1)]],
    device BucketDesc*     bucket_descs [[buffer(2)]],
    constant uint&         tile_size    [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]])
{
    // Thread 0 does serial prefix sum (256 values — trivial)
    threadgroup uint prefix[SORT_NUM_BINS];
    threadgroup uint running_offset;

    if (lid == 0u) {
        uint sum = 0u;
        for (uint i = 0u; i < SORT_NUM_BINS; i++) {
            prefix[i] = sum;
            sum += global_hist[i];
        }
        running_offset = sum;  // total element count (sanity)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All 256 threads write counters and bucket_descs in parallel
    uint count = global_hist[lid];
    uint offset = prefix[lid];
    counters[lid] = offset;  // non-atomic write (used as initial value)

    uint tc = (count + tile_size - 1u) / tile_size;
    bucket_descs[lid] = BucketDesc{offset, count, tc, 0u};
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Atomic MSD Scatter — replaces decoupled lookback
//
// Uses atomic_fetch_add on global counters initialized to
// exclusive_prefix[d]. Each tile's atomic returns its exact
// global position — zero spin-waiting.
//
// TG Memory: 18 KB
// ═══════════════════════════════════════════════════════════════════

kernel void sort_msd_atomic_scatter(
    device const uint*     src       [[buffer(0)]],
    device uint*           dst       [[buffer(1)]],
    device atomic_uint*    counters  [[buffer(2)]],
    constant SortParams&   params    [[buffer(3)]],
    device const uint*     src_vals  [[buffer(4)]],
    device uint*           dst_vals  [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n     = params.element_count;
    uint shift = params.shift;
    uint base  = gid * SORT_TILE_SIZE;

    // ── TG Memory (18 KB) ────────────────────────────────────────
    threadgroup atomic_uint sg_hist_or_rank[SORT_NUM_SGS * SORT_NUM_BINS]; // 8 KB
    threadgroup uint sg_prefix[SORT_NUM_SGS * SORT_NUM_BINS];             // 8 KB
    threadgroup uint tile_hist[SORT_NUM_BINS];                              // 1 KB
    threadgroup uint tile_base[SORT_NUM_BINS];                              // 1 KB

    // ── Phase 1: Load 16 elements ────────────────────────────────
    uint mk[SORT_ELEMS];
    uint md[SORT_ELEMS];
    bool mv[SORT_ELEMS];
    uint mv_vals[SORT_ELEMS];
    for (uint e = 0u; e < SORT_ELEMS; e++) {
        uint idx = base + e * SORT_THREADS + lid;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
        if (has_values) {
            mv_vals[e] = mv[e] ? src_vals[idx] : 0u;
        }
    }

    // ── Phase 2: Per-SG atomic histogram ─────────────────────────
    for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < SORT_ELEMS; e++) {
        if (mv[e]) {
            atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * SORT_NUM_BINS + md[e]],
                1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2b: Tile histogram + cross-SG prefix ───────────────
    {
        uint total = 0u;
        for (uint sg = 0u; sg < SORT_NUM_SGS; sg++) {
            uint c = atomic_load_explicit(
                &sg_hist_or_rank[sg * SORT_NUM_BINS + lid],
                memory_order_relaxed);
            sg_prefix[sg * SORT_NUM_BINS + lid] = total;
            total += c;
        }
        tile_hist[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Atomic fetch-add on global counters ─────────────
    {
        tile_base[lid] = atomic_fetch_add_explicit(
            &counters[lid], tile_hist[lid], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Per-SG ranking + scatter ────────────────────────
    for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
        atomic_store_explicit(&sg_hist_or_rank[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < SORT_ELEMS; e++) {
        if (mv[e]) {
            uint d = md[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * SORT_NUM_BINS + d],
                1u, memory_order_relaxed);
            uint gp = tile_base[d]
                     + sg_prefix[simd_id * SORT_NUM_BINS + d]
                     + within_sg;
            dst[gp] = mk[e];
            if (has_values) {
                dst_vals[gp] = mv_vals[e];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: Fused Inner Sort — 3-pass LSD per bucket
//
// Self-contained: computes own histograms for all 3 inner passes
// during the first scan. No separate precompute dispatch needed.
// 4 dispatches total instead of 5.
//
// Extra TG memory: 2 × 256 atomic_uint = 2KB (pass 1 & 2 histograms)
// Total TG memory: ~22KB (within 32KB limit)
// ═══════════════════════════════════════════════════════════════════

kernel void sort_inner_fused(
    device uint*                buf_a         [[buffer(0)]],
    device uint*                buf_b         [[buffer(1)]],
    device const BucketDesc*    bucket_descs  [[buffer(2)]],
    constant InnerParams&       inner_params  [[buffer(3)]],
    device uint*                vals_a        [[buffer(4)]],
    device uint*                vals_b        [[buffer(5)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    BucketDesc desc = bucket_descs[gid + inner_params.batch_start];
    if (desc.count == 0u) return;

    uint tile_count = desc.tile_count;

    // ═══ Shared memory (22 KB total) ═══
    threadgroup atomic_uint sg_ctr[SORT_NUM_SGS * SORT_NUM_BINS]; // 8 KB (scatter ranking)
    threadgroup uint sg_pfx[SORT_NUM_SGS * SORT_NUM_BINS];        // 8 KB (cross-SG prefix)
    threadgroup uint bkt_hist[SORT_NUM_BINS];                       // 1 KB (current pass histogram)
    threadgroup uint glb_pfx[SORT_NUM_BINS];                        // 1 KB (exclusive prefix sum)
    threadgroup uint run_pfx[SORT_NUM_BINS];                        // 1 KB (running tile prefix)
    threadgroup uint chk_tot[8];                                      // 32 B (prefix sum helper)
    // Self-computed histograms for passes 1 and 2 (accumulated during pass 0)
    threadgroup atomic_uint hist_p1[SORT_NUM_BINS];                  // 1 KB
    threadgroup atomic_uint hist_p2[SORT_NUM_BINS];                  // 1 KB

    // ═══ Zero all histogram accumulators ═══
    atomic_store_explicit(&hist_p1[lid], 0u, memory_order_relaxed);
    atomic_store_explicit(&hist_p2[lid], 0u, memory_order_relaxed);
    // Zero per-SG counters (used for pass 0 histogram accumulation)
    for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
        atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pass = 0u; pass < inner_params.pass_count; pass++) {
        uint shift = (inner_params.start_shift + pass) * 8u;

        // Alternate buffers: pass 0: b->a, pass 1: a->b, pass 2: b->a
        device uint* src = (pass % 2u == 0u) ? buf_b : buf_a;
        device uint* dst = (pass % 2u == 0u) ? buf_a : buf_b;

        // Values follow same ping-pong as keys
        device uint* src_vals = (pass % 2u == 0u) ? vals_b : vals_a;
        device uint* dst_vals = (pass % 2u == 0u) ? vals_a : vals_b;

        // ═══ Load histogram ═══
        if (pass == 0u) {
            // Pass 0: compute histogram via first scan through data
            bkt_hist[lid] = 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // First pass: read all tiles, compute histograms for ALL 3 passes
            for (uint t = 0u; t < tile_count; t++) {
                for (uint e = 0u; e < SORT_ELEMS; e++) {
                    uint local_idx = t * SORT_TILE_SIZE
                                   + simd_id * (SORT_ELEMS * 32u) + e * 32u + simd_lane;
                    if (local_idx < desc.count) {
                        uint val = src[desc.offset + local_idx];
                        uint d0 = val & 0xFFu;
                        uint d1 = (val >> 8u) & 0xFFu;
                        uint d2 = (val >> 16u) & 0xFFu;
                        // Accumulate pass 0 histogram using per-SG accumulation
                        atomic_fetch_add_explicit(&sg_ctr[simd_id * SORT_NUM_BINS + d0],
                                                  1u, memory_order_relaxed);
                        // Pass 1 & 2: direct TG atomic (256 bins, avg 1 collision — fast)
                        atomic_fetch_add_explicit(&hist_p1[d1], 1u, memory_order_relaxed);
                        atomic_fetch_add_explicit(&hist_p2[d2], 1u, memory_order_relaxed);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Reduce pass 0 per-SG histogram to per-bucket histogram
            {
                uint total = 0u;
                for (uint sg = 0u; sg < SORT_NUM_SGS; sg++) {
                    total += atomic_load_explicit(
                        &sg_ctr[sg * SORT_NUM_BINS + lid],
                        memory_order_relaxed);
                }
                bkt_hist[lid] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else if (pass == 1u) {
            bkt_hist[lid] = atomic_load_explicit(&hist_p1[lid], memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            bkt_hist[lid] = atomic_load_explicit(&hist_p2[lid], memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ═══ 256-bin exclusive prefix sum ═══
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

        // ═══ Serial scatter with running cross-tile prefix ═══
        run_pfx[lid] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Zero sg_ctr before scatter
        for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
            atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0u; t < tile_count; t++) {
            uint tile_base = desc.offset + t * SORT_TILE_SIZE;

            uint keys[SORT_ELEMS];
            uint vals[SORT_ELEMS];
            uint digits[SORT_ELEMS];
            bool valid[SORT_ELEMS];
            for (uint e = 0u; e < SORT_ELEMS; e++) {
                uint local_idx = t * SORT_TILE_SIZE
                               + simd_id * (SORT_ELEMS * 32u) + e * 32u + simd_lane;
                valid[e] = local_idx < desc.count;
                uint idx = tile_base + simd_id * (SORT_ELEMS * 32u) + e * 32u + simd_lane;
                keys[e] = valid[e] ? src[idx] : 0xFFFFFFFFu;
                digits[e] = valid[e] ? ((keys[e] >> shift) & 0xFFu) : 0xFFu;
                if (has_values) {
                    vals[e] = valid[e] ? src_vals[idx] : 0u;
                }
            }

            for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
                atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint e = 0u; e < SORT_ELEMS; e++) {
                if (valid[e]) {
                    atomic_fetch_add_explicit(
                        &sg_ctr[simd_id * SORT_NUM_BINS + digits[e]],
                        1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint tile_digit_count;
            {
                uint total = 0u;
                for (uint sg = 0u; sg < SORT_NUM_SGS; sg++) {
                    uint c = atomic_load_explicit(
                        &sg_ctr[sg * SORT_NUM_BINS + lid],
                        memory_order_relaxed);
                    sg_pfx[sg * SORT_NUM_BINS + lid] = total;
                    total += c;
                }
                tile_digit_count = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
                atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint e = 0u; e < SORT_ELEMS; e++) {
                if (valid[e]) {
                    uint d = digits[e];
                    uint within_sg = atomic_fetch_add_explicit(
                        &sg_ctr[simd_id * SORT_NUM_BINS + d],
                        1u, memory_order_relaxed);
                    uint dst_idx = desc.offset
                                 + glb_pfx[d]
                                 + run_pfx[d]
                                 + sg_pfx[simd_id * SORT_NUM_BINS + d]
                                 + within_sg;
                    dst[dst_idx] = keys[e];
                    if (has_values) {
                        dst_vals[dst_idx] = vals[e];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            run_pfx[lid] += tile_digit_count;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Device memory barrier: ensure scatter writes visible for next pass reads
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 5: 32-bit Key Transform — pre/post sort bit manipulation
//
// mode 0: XOR 0x80000000 (i32 sign flip, self-inverse)
// mode 1: FloatFlip forward (map float order → unsigned order)
// mode 2: IFloatFlip inverse (map unsigned order → float order)
//
// Simple 1D dispatch: 1 thread per element.
// ═══════════════════════════════════════════════════════════════════

kernel void sort_transform_32(
    device uint*       data  [[buffer(0)]],
    constant uint&     count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint v = data[gid];

    if (transform_mode == 0u) {
        // i32: flip sign bit (self-inverse)
        v ^= 0x80000000u;
    } else if (transform_mode == 1u) {
        // FloatFlip forward: negative (sign set) → flip all; positive → flip sign only
        v = (v & 0x80000000u) ? ~v : (v ^ 0x80000000u);
    } else if (transform_mode == 2u) {
        // IFloatFlip inverse: sign set (was positive) → flip sign; sign clear (was negative) → flip all
        v = (v & 0x80000000u) ? (v ^ 0x80000000u) : ~v;
    }

    data[gid] = v;
}
