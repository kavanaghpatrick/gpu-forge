#include "search_types.h"

#include <metal_simdgroup>

// ============================================================================
// BATCH SEARCH KERNEL
// ============================================================================
// Processes all pending work items in one dispatch.
// Each threadgroup processes one work item.

kernel void batch_search_kernel(
    device PersistentKernelControl* control [[buffer(0)]],
    device SearchWorkItem* work_queue [[buffer(1)]],
    device const uchar* data [[buffer(2)]],
    device const DataBufferDescriptor* data_descriptors [[buffer(3)]],
    device atomic_uint* match_counts [[buffer(4)]],
    constant uint& queue_size [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Each threadgroup handles one work item
    // Threadgroup ID determines which queue slot to process

    // Threadgroup-local storage
    threadgroup uint tg_pattern_len;
    threadgroup bool tg_case_sensitive;
    threadgroup uchar tg_pattern[MAX_PATTERN_LEN];
    threadgroup ulong tg_data_offset;
    threadgroup uint tg_data_size;
    threadgroup bool has_work;
    threadgroup uint work_idx;

    // Thread 0 checks if this threadgroup has work
    if (tid == 0) {
        // Heartbeat
        atomic_fetch_add_explicit(&control->heartbeat, 1, memory_order_relaxed);

        uint head = atomic_load_explicit(&control->head, memory_order_relaxed);
        uint tail = atomic_load_explicit(&control->tail, memory_order_relaxed);
        uint pending = tail - head;

        // This threadgroup handles work item at head + tgid
        if (tgid < pending) {
            work_idx = (head + tgid) % queue_size;
            uint status = atomic_load_explicit(&work_queue[work_idx].status, memory_order_relaxed);

            if (status == STATUS_READY) {
                // Claim this work item
                atomic_store_explicit(&work_queue[work_idx].status, STATUS_PROCESSING, memory_order_relaxed);

                // Copy work item data to threadgroup memory
                tg_pattern_len = work_queue[work_idx].pattern_len;
                tg_case_sensitive = work_queue[work_idx].case_sensitive != 0;
                for (uint i = 0; i < tg_pattern_len && i < MAX_PATTERN_LEN; i++) {
                    tg_pattern[i] = work_queue[work_idx].pattern[i];
                }

                // Get data buffer info
                uint buf_id = work_queue[work_idx].data_buffer_id;
                tg_data_offset = data_descriptors[buf_id].offset;
                tg_data_size = data_descriptors[buf_id].size;

                // Reset match count
                atomic_store_explicit(&match_counts[work_idx], 0, memory_order_relaxed);

                has_work = true;
            } else {
                has_work = false;
            }
        } else {
            has_work = false;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!has_work) {
        return;
    }

    // All threads participate in search
    uint total_bytes = tg_data_size;
    uint thread_count = tg_size;
    uint bytes_per_chunk = (total_bytes + thread_count - 1) / thread_count;
    bytes_per_chunk = max(bytes_per_chunk, (uint)BYTES_PER_THREAD);

    uint my_start = tid * bytes_per_chunk;
    uint my_end = min(my_start + bytes_per_chunk, total_bytes);

    // Search within my range
    uint local_match_count = 0;

    if (my_start < total_bytes && tg_pattern_len > 0) {
        uint search_end = (my_end >= tg_pattern_len) ? (my_end - tg_pattern_len + 1) : 0;

        for (uint pos = my_start; pos < search_end && local_match_count < MAX_MATCHES_PER_THREAD; pos++) {
            bool match = true;

            // Check pattern match
            for (uint j = 0; j < tg_pattern_len && match; j++) {
                uchar data_byte = data[tg_data_offset + pos + j];
                if (!char_eq_fast(data_byte, tg_pattern[j], tg_case_sensitive)) {
                    match = false;
                }
            }

            if (match) {
                local_match_count++;
            }
        }
    }

    // Aggregate results using SIMD reduction
    uint simd_total = simd_sum(local_match_count);

    // First lane of each SIMD group adds to global count
    if (simd_is_first()) {
        atomic_fetch_add_explicit(&match_counts[work_idx], simd_total, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Thread 0 finalizes the work item
    if (tid == 0) {
        uint total_matches = atomic_load_explicit(&match_counts[work_idx], memory_order_relaxed);
        atomic_store_explicit(&work_queue[work_idx].result_count, total_matches, memory_order_relaxed);
        atomic_store_explicit(&work_queue[work_idx].status, STATUS_DONE, memory_order_relaxed);
    }
}
