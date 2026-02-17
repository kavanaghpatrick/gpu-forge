// reduce.metal -- GPU reduction kernels for forge-primitives.
//
// 4 kernels implementing 3-level hierarchical reduction:
//   Level 1: SIMD intrinsic across simdgroup (32 lanes on Apple Silicon)
//   Level 2: Threadgroup shared memory reduction + barrier
//   Level 3: Global accumulation (atomic for sum, partials array for min/max)
//
// All kernels use 256 threads per threadgroup.
// Dispatch: ceil(element_count / 256) threadgroups.

#include "types.h"

#define MAX_THREADS_PER_TG 256


// ============================================================================
// reduce_sum_u32 -- 3-level u32 sum with atomic global accumulation
// ============================================================================
//
// Buffer layout:
//   buffer(0): input data (uint array)
//   buffer(1): output result (atomic_uint, single element)
//   buffer(2): ReduceParams
//
// Dispatch: ceil(element_count / 256) threadgroups of 256 threads.

kernel void reduce_sum_u32(
    device const uint*       input           [[buffer(0)]],
    device atomic_uint*      result          [[buffer(1)]],
    constant ReduceParams&   params          [[buffer(2)]],
    uint tid                                 [[thread_position_in_grid]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_size                             [[threads_per_threadgroup]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // --- Level 0: per-thread load ---
    uint local_val = 0;
    if (tid < params.element_count) {
        local_val = input[tid];
    }

    // --- Level 1: SIMD reduction ---
    uint simd_total = simd_sum(local_val);

    // --- Level 2: threadgroup reduction via shared memory ---
    threadgroup uint tg_partials[MAX_THREADS_PER_TG / 32];

    if (simd_lane == 0) {
        tg_partials[simd_group_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tg_total = 0;
    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        uint val = (simd_lane < num_simd_groups) ? tg_partials[simd_lane] : 0;
        tg_total = simd_sum(val);
    }

    // --- Level 3: global atomic accumulation ---
    if (tid_in_tg == 0 && tg_total > 0) {
        atomic_fetch_add_explicit(result, tg_total, memory_order_relaxed);
    }
}


// ============================================================================
// reduce_sum_f32 -- 3-level float sum with CAS loop atomic accumulation
// ============================================================================
//
// Metal has no native atomic_float add. We use a compare-and-swap loop
// reinterpreting float bits as uint for the atomic operation.
//
// Buffer layout:
//   buffer(0): input data (float array)
//   buffer(1): output result (atomic_uint, stores float bits)
//   buffer(2): ReduceParams
//
// Dispatch: ceil(element_count / 256) threadgroups of 256 threads.

kernel void reduce_sum_f32(
    device const float*      input           [[buffer(0)]],
    device atomic_uint*      result          [[buffer(1)]],
    constant ReduceParams&   params          [[buffer(2)]],
    uint tid                                 [[thread_position_in_grid]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_size                             [[threads_per_threadgroup]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // --- Level 0: per-thread load ---
    float local_val = 0.0f;
    if (tid < params.element_count) {
        local_val = input[tid];
    }

    // --- Level 1: SIMD reduction (float natively supported) ---
    float simd_total = simd_sum(local_val);

    // --- Level 2: threadgroup reduction via shared memory ---
    threadgroup float tg_partials[MAX_THREADS_PER_TG / 32];

    if (simd_lane == 0) {
        tg_partials[simd_group_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tg_total = 0.0f;
    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        float val = (simd_lane < num_simd_groups) ? tg_partials[simd_lane] : 0.0f;
        tg_total = simd_sum(val);
    }

    // --- Level 3: global atomic via CAS loop (float as uint bits) ---
    if (tid_in_tg == 0 && tg_total != 0.0f) {
        uint expected = atomic_load_explicit(result, memory_order_relaxed);
        for (int i = 0; i < 64; i++) {
            float cur = as_type<float>(expected);
            float desired = cur + tg_total;
            uint desired_bits = as_type<uint>(desired);
            if (atomic_compare_exchange_weak_explicit(result, &expected, desired_bits,
                                                      memory_order_relaxed,
                                                      memory_order_relaxed)) {
                break;
            }
        }
    }
}


// ============================================================================
// reduce_sum_u32_v2 -- Two-pass atomic-free u32 sum (pass 1)
// ============================================================================
//
// Each thread loads 4 elements via load_uint4_safe for 4x throughput.
// SIMD reduction + threadgroup reduction via shared memory.
// Writes per-threadgroup partial sum to partials[tg_idx] (NO atomics).
//
// Buffer layout:
//   buffer(0): input data (uint array)
//   buffer(1): output partials (uint array, one per threadgroup)
//   buffer(2): ReduceParams
//
// Dispatch: ceil(element_count / (256 * 4)) threadgroups of 256 threads.

kernel void reduce_sum_u32_v2(
    device const uint*       input           [[buffer(0)]],
    device uint*             partials        [[buffer(1)]],
    constant ReduceParams&   params          [[buffer(2)]],
    uint tid                                 [[thread_position_in_grid]],
    uint tg_idx                              [[threadgroup_position_in_grid]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_size                             [[threads_per_threadgroup]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // --- Level 0: per-thread vectorized load (4 elements) ---
    uint base_idx = tid * 4;
    uint4 vals = load_uint4_safe(input, base_idx, params.element_count);
    uint local_sum = vals.x + vals.y + vals.z + vals.w;

    // --- Level 1: SIMD reduction ---
    uint simd_total = simd_sum(local_sum);

    // --- Level 2: threadgroup reduction via shared memory ---
    threadgroup uint tg_partials[MAX_THREADS_PER_TG / 32];

    if (simd_lane == 0) {
        tg_partials[simd_group_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tg_total = 0;
    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        uint val = (simd_lane < num_simd_groups) ? tg_partials[simd_lane] : 0;
        tg_total = simd_sum(val);
    }

    // --- Level 3: write per-threadgroup partial (NO atomics) ---
    if (tid_in_tg == 0) {
        partials[tg_idx] = tg_total;
    }
}


// ============================================================================
// reduce_sum_partials -- Two-pass atomic-free u32 sum (pass 2)
// ============================================================================
//
// Single-threadgroup reduction of the partials array into final result.
// Same SIMD + threadgroup reduction pattern as v2.
// Each thread loads 4 partials via load_uint4_safe for consistency.
//
// Buffer layout:
//   buffer(0): partials from pass 1 (uint array)
//   buffer(1): output result (single uint)
//   buffer(2): ReduceParams (element_count = number of partials)
//
// Dispatch: 1 threadgroup of 256 threads.

kernel void reduce_sum_partials(
    device const uint*       partials        [[buffer(0)]],
    device uint*             result          [[buffer(1)]],
    constant ReduceParams&   params          [[buffer(2)]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_size                             [[threads_per_threadgroup]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // --- Level 0: per-thread vectorized load (4 partials) ---
    uint base_idx = tid_in_tg * 4;
    uint4 vals = load_uint4_safe(partials, base_idx, params.element_count);
    uint local_sum = vals.x + vals.y + vals.z + vals.w;

    // --- Level 1: SIMD reduction ---
    uint simd_total = simd_sum(local_sum);

    // --- Level 2: threadgroup reduction via shared memory ---
    threadgroup uint tg_partials[MAX_THREADS_PER_TG / 32];

    if (simd_lane == 0) {
        tg_partials[simd_group_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tg_total = 0;
    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        uint val = (simd_lane < num_simd_groups) ? tg_partials[simd_lane] : 0;
        tg_total = simd_sum(val);
    }

    // --- Write final result ---
    if (tid_in_tg == 0) {
        result[0] = tg_total;
    }
}


// ============================================================================
// reduce_min_u32 -- 3-level u32 min with per-threadgroup partials
// ============================================================================
//
// No atomic MIN for uint on Metal. Each threadgroup writes its partial
// minimum to partials[tg_idx]. Host does final reduction on CPU.
//
// Buffer layout:
//   buffer(0): input data (uint array)
//   buffer(1): output partials (uint array, one per threadgroup)
//   buffer(2): ReduceParams
//
// Dispatch: ceil(element_count / 256) threadgroups of 256 threads.

kernel void reduce_min_u32(
    device const uint*       input           [[buffer(0)]],
    device uint*             partials        [[buffer(1)]],
    constant ReduceParams&   params          [[buffer(2)]],
    uint tid                                 [[thread_position_in_grid]],
    uint tg_idx                              [[threadgroup_position_in_grid]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_size                             [[threads_per_threadgroup]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // --- Level 0: per-thread load (identity = UINT_MAX) ---
    uint local_val = 0xFFFFFFFF;
    if (tid < params.element_count) {
        local_val = input[tid];
    }

    // --- Level 1: SIMD reduction ---
    uint simd_result = simd_min(local_val);

    // --- Level 2: threadgroup reduction via shared memory ---
    threadgroup uint tg_partials[MAX_THREADS_PER_TG / 32];

    if (simd_lane == 0) {
        tg_partials[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tg_result = 0xFFFFFFFF;
    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        uint val = (simd_lane < num_simd_groups) ? tg_partials[simd_lane] : 0xFFFFFFFF;
        tg_result = simd_min(val);
    }

    // --- Level 3: write per-threadgroup partial to output array ---
    if (tid_in_tg == 0) {
        partials[tg_idx] = tg_result;
    }
}


// ============================================================================
// reduce_max_u32 -- 3-level u32 max with per-threadgroup partials
// ============================================================================
//
// Same pattern as reduce_min_u32 but with max reduction.
// Identity element = 0 (minimum possible uint).
//
// Buffer layout:
//   buffer(0): input data (uint array)
//   buffer(1): output partials (uint array, one per threadgroup)
//   buffer(2): ReduceParams
//
// Dispatch: ceil(element_count / 256) threadgroups of 256 threads.

kernel void reduce_max_u32(
    device const uint*       input           [[buffer(0)]],
    device uint*             partials        [[buffer(1)]],
    constant ReduceParams&   params          [[buffer(2)]],
    uint tid                                 [[thread_position_in_grid]],
    uint tg_idx                              [[threadgroup_position_in_grid]],
    uint tid_in_tg                           [[thread_position_in_threadgroup]],
    uint tg_size                             [[threads_per_threadgroup]],
    uint simd_lane                           [[thread_index_in_simdgroup]],
    uint simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // --- Level 0: per-thread load (identity = 0) ---
    uint local_val = 0;
    if (tid < params.element_count) {
        local_val = input[tid];
    }

    // --- Level 1: SIMD reduction ---
    uint simd_result = simd_max(local_val);

    // --- Level 2: threadgroup reduction via shared memory ---
    threadgroup uint tg_partials[MAX_THREADS_PER_TG / 32];

    if (simd_lane == 0) {
        tg_partials[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tg_result = 0;
    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        uint val = (simd_lane < num_simd_groups) ? tg_partials[simd_lane] : 0;
        tg_result = simd_max(val);
    }

    // --- Level 3: write per-threadgroup partial to output array ---
    if (tid_in_tg == 0) {
        partials[tg_idx] = tg_result;
    }
}
