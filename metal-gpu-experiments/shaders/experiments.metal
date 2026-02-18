#include <metal_stdlib>
using namespace metal;
#include "types.h"

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 1: Texture L1 Cache Doubling
//
// Hypothesis: Texture unit and shader core have SEPARATE L1 caches.
// Reading data through both paths simultaneously should yield ~2x
// effective L1 bandwidth vs. a single path.
//
// KB #3119: "Separate L1 caches for texture and buffer reads"
// ══════════════════════════════════════════════════════════════════════

kernel void exp1_buffer_only(
    device const float* data [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant ExpParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    float sum = 0.0f;
    uint n = params.element_count;
    uint passes = params.num_passes;
    for (uint p = 0; p < passes; p++) {
        sum += data[(tid + p * 31u) % n];
    }
    output[tid] = sum;
}

kernel void exp1_dual_read(
    device const float* data  [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant ExpParams& params [[buffer(2)]],
    texture2d<float, access::read> tex [[texture(0)]],
    uint tid [[thread_position_in_grid]])
{
    float sum_buf = 0.0f;
    float sum_tex = 0.0f;
    uint n = params.element_count;
    uint tex_w = tex.get_width();
    uint half_passes = params.num_passes / 2u;

    for (uint p = 0; p < half_passes; p++) {
        uint idx_b = (tid + p * 31u) % n;
        uint idx_t = (tid + p * 37u) % n;
        sum_buf += data[idx_b];
        sum_tex += tex.read(uint2(idx_t % tex_w, idx_t / tex_w)).x;
    }
    output[tid] = sum_buf + sum_tex;
}

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 2: Texture Bilinear Interpolation as Free ALU
//
// Hypothesis: Hardware bilinear filtering can evaluate activation-
// function LUTs for "free" while ALUs do other work.
//
// KB #2857: texture hardware interpolation
// ══════════════════════════════════════════════════════════════════════

kernel void exp2_manual_lerp(
    device const float* lut    [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant ExpParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;

    float x = input[tid];                   // value in [0, 1]
    float scaled = x * 255.0f;
    uint lo = (uint)clamp(scaled, 0.0f, 254.0f);
    uint hi = lo + 1u;
    float frac = scaled - (float)lo;

    output[tid] = mix(lut[lo], lut[hi], frac);
}

kernel void exp2_hw_bilinear(
    texture2d<float, access::sample> lut_tex [[texture(0)]],
    sampler lut_sampler                      [[sampler(0)]],
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant ExpParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;

    float x = input[tid];
    // Sample 1D LUT packed as 256×1 texture with bilinear filter
    output[tid] = lut_tex.sample(lut_sampler, float2(x, 0.5f)).x;
}

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 3: Vectorized Hash Probe (Batch-4 vs Scalar)
//
// Hypothesis: Loading 4 consecutive hash table slots per loop iteration
// hides pipeline latency and reduces branch overhead vs. scalar probing.
//
// Uses AoS layout: table[i] = {key, value} as uint2
// ══════════════════════════════════════════════════════════════════════

inline uint murmur3_finalize(uint key) {
    key ^= key >> 16;
    key *= 0x85ebca6bu;
    key ^= key >> 13;
    key *= 0xc2b2ae35u;
    key ^= key >> 16;
    return key;
}

#define EMPTY_SLOT 0xFFFFFFFFu

kernel void exp3_scalar_probe(
    device const uint2* table  [[buffer(0)]],   // {key, value} pairs
    device const uint* queries [[buffer(1)]],
    device uint* output        [[buffer(2)]],
    constant ExpParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;

    uint key  = queries[tid];
    uint cap  = params.num_passes;   // table capacity
    uint mask = cap - 1u;
    uint slot = murmur3_finalize(key) & mask;

    for (uint i = 0; i < 64u; i++) {
        uint s = (slot + i) & mask;
        uint2 entry = table[s];
        if (entry.x == key)       { output[tid] = entry.y; return; }
        if (entry.x == EMPTY_SLOT) break;
    }
    output[tid] = EMPTY_SLOT;
}

kernel void exp3_batch4_probe(
    device const uint2* table  [[buffer(0)]],
    device const uint* queries [[buffer(1)]],
    device uint* output        [[buffer(2)]],
    constant ExpParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;

    uint key  = queries[tid];
    uint cap  = params.num_passes;
    uint mask = cap - 1u;
    uint slot = murmur3_finalize(key) & mask;

    for (uint i = 0; i < 64u; i += 4u) {
        // Pre-load 4 consecutive entries
        uint2 e0 = table[(slot + i)      & mask];
        uint2 e1 = table[(slot + i + 1u) & mask];
        uint2 e2 = table[(slot + i + 2u) & mask];
        uint2 e3 = table[(slot + i + 3u) & mask];

        if (e0.x == key)       { output[tid] = e0.y; return; }
        if (e0.x == EMPTY_SLOT) break;
        if (e1.x == key)       { output[tid] = e1.y; return; }
        if (e1.x == EMPTY_SLOT) break;
        if (e2.x == key)       { output[tid] = e2.y; return; }
        if (e2.x == EMPTY_SLOT) break;
        if (e3.x == key)       { output[tid] = e3.y; return; }
        if (e3.x == EMPTY_SLOT) break;
    }
    output[tid] = EMPTY_SLOT;
}

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 4: Decoupled Lookback Prefix Sum
//
// FIRST Apple Silicon implementation.
// Single-dispatch inclusive prefix sum using cross-threadgroup
// relaxed atomics. Tests whether Apple Silicon's TSO-like memory
// model allows the decoupled lookback algorithm to work despite
// Metal's spec only guaranteeing relaxed ordering.
//
// KB #155/#156: coherent(device) in Metal 3.2
// KB #699: Decoupled Fallback prefix sum
// ══════════════════════════════════════════════════════════════════════

// --- Multi-pass version: 3 dispatches ---

kernel void exp4_reduce(
    device const uint* input   [[buffer(0)]],
    device uint* tile_sums     [[buffer(1)]],
    constant ExpParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup uint shared[TILE_SIZE];
    shared[lid] = (tid < params.element_count) ? input[tid] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = TILE_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0u) tile_sums[gid] = shared[0];
}

kernel void exp4_local_scan_and_add(
    device const uint* input       [[buffer(0)]],
    device uint* output            [[buffer(1)]],
    device const uint* tile_prefix [[buffer(2)]],
    constant ExpParams& params     [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup uint shared[TILE_SIZE];
    shared[lid] = (tid < params.element_count) ? input[tid] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan
    for (uint offset = 1u; offset < TILE_SIZE; offset <<= 1u) {
        uint to_add = (lid >= offset) ? shared[lid - offset] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[lid] += to_add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < params.element_count)
        output[tid] = shared[lid] + tile_prefix[gid];
}

// --- Single-pass decoupled lookback ---

kernel void exp4_decoupled_lookback(
    device const uint* input       [[buffer(0)]],
    device uint* output            [[buffer(1)]],
    device atomic_uint* tile_status [[buffer(2)]],
    constant ExpParams& params     [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // --- Local inclusive scan via SIMD prefix sums ---
    uint val = (tid < params.element_count) ? input[tid] : 0u;
    uint simd_scan = simd_prefix_inclusive_sum(val);

    threadgroup uint simd_totals[8];   // max 8 simdgroups × 32 = 256 threads
    if (simd_lane == 31u)
        simd_totals[simd_id] = simd_scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u && simd_lane < 8u) {
        uint sg = simd_totals[simd_lane];
        uint sg_scan = simd_prefix_inclusive_sum(sg);
        simd_totals[simd_lane] = sg_scan;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_scan = simd_scan;
    if (simd_id > 0u) local_scan += simd_totals[simd_id - 1u];

    uint tile_aggregate = simd_totals[7];

    // --- Publish aggregate & do lookback (thread 0 only) ---
    threadgroup uint tile_exclusive;

    if (lid == 0u) {
        // Publish our aggregate
        uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT) | (tile_aggregate & VALUE_MASK);
        atomic_store_explicit(&tile_status[gid], packed_agg, memory_order_relaxed);

        if (gid == 0u) {
            // First tile: publish prefix immediately
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (tile_aggregate & VALUE_MASK);
            atomic_store_explicit(&tile_status[0], packed_pfx, memory_order_relaxed);
            tile_exclusive = 0u;
        } else {
            uint running = 0u;
            int lookback = (int)gid - 1;
            while (lookback >= 0) {
                uint status = atomic_load_explicit(&tile_status[lookback], memory_order_relaxed);
                uint flag   = status >> FLAG_SHIFT;
                uint value  = status & VALUE_MASK;

                if (flag == FLAG_PREFIX) {
                    running += value;
                    break;
                } else if (flag == FLAG_AGGREGATE) {
                    running += value;
                    lookback--;
                }
                // else: FLAG_NOT_READY — spin
            }
            tile_exclusive = running;

            // Publish our inclusive prefix
            uint inclusive = running + tile_aggregate;
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK);
            atomic_store_explicit(&tile_status[gid], packed_pfx, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < params.element_count)
        output[tid] = local_scan + tile_exclusive;
}

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 5: Megakernel (Single Dispatch vs N Dispatches)
//
// Hypothesis: One dispatch doing N operations on data eliminates
// N-1 command buffer overheads + N-1 extra DRAM passes.
//
// KB #136: 641% regression from command buffer fragmentation
// KB #1687: Visible function table megakernel pattern
// ══════════════════════════════════════════════════════════════════════

// Individual kernels for multi-dispatch version
kernel void exp5_scale(
    device float* data         [[buffer(0)]],
    constant float& scalar     [[buffer(1)]],
    constant ExpParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;
    data[tid] *= scalar;
}

kernel void exp5_add(
    device float* data         [[buffer(0)]],
    constant float& scalar     [[buffer(1)]],
    constant ExpParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;
    data[tid] += scalar;
}

kernel void exp5_sqrt_op(
    device float* data         [[buffer(0)]],
    constant ExpParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;
    data[tid] = sqrt(abs(data[tid]));
}

kernel void exp5_negate(
    device float* data         [[buffer(0)]],
    constant ExpParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;
    data[tid] = -data[tid];
}

// Megakernel: all 10 operations in a single dispatch
kernel void exp5_megakernel(
    device float* data         [[buffer(0)]],
    constant float* scalars    [[buffer(1)]],   // 6 scalar params
    constant ExpParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;

    float v = data[tid];

    v *= scalars[0];        // scale
    v += scalars[1];        // add
    v  = sqrt(abs(v));      // sqrt
    v  = -v;                // negate
    v *= scalars[2];        // scale
    v += scalars[3];        // add
    v  = sqrt(abs(v));      // sqrt
    v  = -v;                // negate
    v *= scalars[4];        // scale
    v += scalars[5];        // add

    data[tid] = v;
}

// ══════════════════════════════════════════════════════════════════════
// EXPERIMENT 6: Single-Dispatch Stream Compaction (COMBINED)
//
// Combines Exp 4 (decoupled lookback) + Exp 5 (megakernel fusion).
// Traditional stream compaction needs 4 dispatches:
//   1. evaluate predicate  2. reduce tiles  3. scan tiles  4. scatter
// This does it in ONE dispatch: predicate → local scan → decoupled
// lookback for global offsets → scatter. Data never leaves registers
// between phases.
//
// Predicate: value % 3 == 0  (keeps ~33% of elements)
// ══════════════════════════════════════════════════════════════════════

// --- Multi-dispatch building blocks ---

kernel void exp6_predicate(
    device const uint* input   [[buffer(0)]],
    device uint* pred_out      [[buffer(1)]],
    constant ExpParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;
    pred_out[tid] = (input[tid] % 3u == 0u) ? 1u : 0u;
}

kernel void exp6_scatter(
    device const uint* input   [[buffer(0)]],
    device const uint* pred    [[buffer(1)]],
    device const uint* prefix  [[buffer(2)]],
    device uint* output        [[buffer(3)]],
    constant ExpParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.element_count) return;
    if (pred[tid] == 1u) {
        // prefix is inclusive, so output position = prefix[tid] - 1
        output[prefix[tid] - 1u] = input[tid];
    }
}

// --- Single-dispatch persistent kernel ---

kernel void exp6_compact_single(
    device const uint* input        [[buffer(0)]],
    device uint* output             [[buffer(1)]],
    device atomic_uint* tile_status [[buffer(2)]],
    device atomic_uint* total_count [[buffer(3)]],
    constant ExpParams& params      [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // ── Phase 1: Load + evaluate predicate (fused, no DRAM write) ──
    uint val  = (tid < params.element_count) ? input[tid] : 0u;
    uint pred = (tid < params.element_count && val % 3u == 0u) ? 1u : 0u;

    // ── Phase 2: Local inclusive prefix sum of predicates ──
    uint simd_scan = simd_prefix_inclusive_sum(pred);

    threadgroup uint simd_totals[8];
    if (simd_lane == 31u)
        simd_totals[simd_id] = simd_scan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u && simd_lane < 8u) {
        uint sg = simd_totals[simd_lane];
        uint sg_scan = simd_prefix_inclusive_sum(sg);
        simd_totals[simd_lane] = sg_scan;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_scan = simd_scan;
    if (simd_id > 0u) local_scan += simd_totals[simd_id - 1u];

    uint tile_count = simd_totals[7];  // how many pass in this tile

    // ── Phase 3: Decoupled lookback for global prefix ──
    threadgroup uint tile_exclusive;

    if (lid == 0u) {
        uint packed_agg = (FLAG_AGGREGATE << FLAG_SHIFT) | (tile_count & VALUE_MASK);
        atomic_store_explicit(&tile_status[gid], packed_agg, memory_order_relaxed);

        if (gid == 0u) {
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (tile_count & VALUE_MASK);
            atomic_store_explicit(&tile_status[0], packed_pfx, memory_order_relaxed);
            tile_exclusive = 0u;
        } else {
            uint running = 0u;
            int lookback = (int)gid - 1;
            while (lookback >= 0) {
                uint status = atomic_load_explicit(&tile_status[lookback], memory_order_relaxed);
                uint flag  = status >> FLAG_SHIFT;
                uint value = status & VALUE_MASK;

                if (flag == FLAG_PREFIX) {
                    running += value;
                    break;
                } else if (flag == FLAG_AGGREGATE) {
                    running += value;
                    lookback--;
                }
                // else: spin on FLAG_NOT_READY
            }
            tile_exclusive = running;

            uint inclusive = running + tile_count;
            uint packed_pfx = (FLAG_PREFIX << FLAG_SHIFT) | (inclusive & VALUE_MASK);
            atomic_store_explicit(&tile_status[gid], packed_pfx, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Scatter passing elements (fused, no intermediate DRAM) ──
    if (pred == 1u && tid < params.element_count) {
        uint local_offset = local_scan - 1u;   // inclusive → 0-based
        uint global_pos = tile_exclusive + local_offset;
        output[global_pos] = val;
    }

    // Last tile publishes total count
    uint num_tiles = (params.element_count + TILE_SIZE - 1u) / TILE_SIZE;
    if (gid == num_tiles - 1u && lid == 0u) {
        atomic_store_explicit(total_count, tile_exclusive + tile_count, memory_order_relaxed);
    }
}
