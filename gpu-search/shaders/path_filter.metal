#include "search_types.h"

#include <metal_simdgroup>

// ============================================================================
// PATH FILTER KERNEL
// ============================================================================
// Filters the GPU-resident filesystem index by matching path entries against
// a pattern. Each thread processes one GpuPathEntry. Supports both substring
// and fuzzy matching for filename search.
//
// Adapted from rust-experiment filesystem.rs fuzzy_search_kernel, operating
// on the GpuPathEntry (256B) struct from the gpu-search index.

// Check if haystack contains needle as substring (case-insensitive)
inline bool path_contains(
    thread const uchar* haystack, uint haystack_len,
    constant const uchar* needle, uint needle_len,
    bool case_sensitive
) {
    if (needle_len > haystack_len) return false;

    for (uint i = 0; i <= haystack_len - needle_len; i++) {
        bool match = true;
        for (uint j = 0; j < needle_len; j++) {
            if (!char_eq_fast(haystack[i + j], needle[j], case_sensitive)) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

// Extract filename start position (find last '/')
inline uint find_filename_start(thread const uchar* path, uint path_len) {
    for (int i = (int)path_len - 1; i >= 0; i--) {
        if (path[i] == '/') {
            return (uint)(i + 1);
        }
    }
    return 0;
}

kernel void path_filter_kernel(
    device const GpuPathEntry* entries [[buffer(0)]],
    constant uchar* pattern [[buffer(1)]],
    constant PathFilterParams& params [[buffer(2)]],
    device atomic_uint& match_count [[buffer(3)]],
    device uint* match_indices [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    if (gid >= params.entry_count) return;

    GpuPathEntry entry = entries[gid];
    uint path_len = entry.path_len;

    if (path_len == 0) return;

    bool case_sensitive = params.case_sensitive != 0;

    // Find filename portion of the path
    uint filename_start = find_filename_start(entry.path, path_len);
    uint filename_len = path_len - filename_start;

    // Check if pattern matches in filename (preferred) or full path
    bool matched = false;

    // First: try substring match in filename
    if (path_contains(entry.path + filename_start, filename_len,
                      pattern, params.pattern_len, case_sensitive)) {
        matched = true;
    }
    // Fallback: try substring match in full path
    else if (path_contains(entry.path, path_len,
                           pattern, params.pattern_len, case_sensitive)) {
        matched = true;
    }

    // SIMD reduction for efficient atomic usage
    uint local_count = matched ? 1u : 0u;
    uint simd_total = simd_sum(local_count);
    uint my_offset = simd_prefix_exclusive_sum(local_count);

    uint group_base = 0;
    if (simd_lane == 0 && simd_total > 0) {
        group_base = atomic_fetch_add_explicit(&match_count, simd_total, memory_order_relaxed);
    }
    group_base = simd_broadcast_first(group_base);

    if (matched) {
        uint global_idx = group_base + my_offset;
        if (global_idx < params.max_matches) {
            match_indices[global_idx] = gid;
        }
    }
}
