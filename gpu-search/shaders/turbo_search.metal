#include "search_types.h"

#include <metal_simdgroup>

// =============================================================================
// TURBO MODE KERNEL - MAXIMUM THROUGHPUT
// =============================================================================
// Defers line number calculation to CPU for 70+ GB/s throughput
// Returns byte offsets only - CPU calculates line numbers post-search

kernel void turbo_search_kernel(
    device const uchar4* data [[buffer(0)]],
    device const ChunkMetadata* metadata [[buffer(1)]],
    constant SearchParams& params [[buffer(2)]],
    constant uchar* pattern [[buffer(3)]],
    device MatchResult* matches [[buffer(4)]],
    device atomic_uint& match_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint byte_base = gid * BYTES_PER_THREAD;
    uint vec4_base = byte_base / 4;

    if (byte_base >= params.total_bytes) return;

    uint chunk_idx = byte_base / CHUNK_SIZE;
    uint offset_in_chunk = byte_base % CHUNK_SIZE;

    if (chunk_idx >= params.chunk_count) return;

    ChunkMetadata meta = metadata[chunk_idx];
    uint chunk_len = meta.chunk_length;

    if (offset_in_chunk >= chunk_len) return;

    // Load 64 bytes using vectorized loads
    uchar local_data[BYTES_PER_THREAD];
    uint valid_bytes = min((uint)BYTES_PER_THREAD, chunk_len - offset_in_chunk);

    #pragma unroll
    for (uint i = 0; i < 16; i++) {
        uint vec_idx = vec4_base + i;
        if (i * 4 < valid_bytes) {
            uchar4 v = data[vec_idx];
            local_data[i*4 + 0] = v.x;
            local_data[i*4 + 1] = v.y;
            local_data[i*4 + 2] = v.z;
            local_data[i*4 + 3] = v.w;
        }
    }

    // Fast search - no line number calculation!
    uint local_matches_pos[MAX_MATCHES_PER_THREAD];
    uint local_match_count = 0;

    bool case_sensitive = params.case_sensitive != 0;
    uint search_end = (valid_bytes >= params.pattern_len) ? (valid_bytes - params.pattern_len + 1) : 0;

    for (uint pos = 0; pos < search_end && local_match_count < MAX_MATCHES_PER_THREAD; pos++) {
        bool match = true;
        for (uint j = 0; j < params.pattern_len && match; j++) {
            if (!char_eq_fast(local_data[pos + j], pattern[j], case_sensitive)) {
                match = false;
            }
        }
        if (match) {
            local_matches_pos[local_match_count++] = pos;
        }
    }

    // SIMD reduction
    uint simd_total = simd_sum(local_match_count);
    uint my_offset = simd_prefix_exclusive_sum(local_match_count);

    uint group_base = 0;
    if (simd_lane == 0 && simd_total > 0) {
        group_base = atomic_fetch_add_explicit(&match_count, simd_total, memory_order_relaxed);
    }
    group_base = simd_broadcast_first(group_base);

    // Write minimal match info (CPU calculates line numbers)
    for (uint i = 0; i < local_match_count; i++) {
        uint global_idx = group_base + my_offset + i;
        if (global_idx < 10000) {
            uint local_pos = local_matches_pos[i];

            MatchResult result;
            result.file_index = meta.file_index;
            result.chunk_index = chunk_idx;
            result.line_number = 0;  // CPU will calculate
            result.column = offset_in_chunk + local_pos;  // Byte offset in chunk
            result.match_length = params.pattern_len;
            result.context_start = offset_in_chunk + local_pos;  // Raw byte offset
            result.context_len = min(valid_bytes - local_pos, (uint)MAX_CONTEXT);
            result._padding = 0;

            matches[global_idx] = result;
        }
    }
}
