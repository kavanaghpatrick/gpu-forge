#ifndef SEARCH_TYPES_H
#define SEARCH_TYPES_H

#include <metal_stdlib>
using namespace metal;

// Search parameters passed from CPU to GPU
struct SearchParams {
    uchar pattern[256];     // Search pattern bytes
    uint pattern_len;       // Length of the pattern
    uint total_bytes;       // Total bytes in the input buffer
    uint case_sensitive;    // 1 = case sensitive, 0 = case insensitive
    uint max_matches;       // Maximum number of matches to return
    uint file_count;        // Number of files in the batch
    uint reserved[2];       // Padding for alignment
};

// A single match result written by the GPU
struct GpuMatchResult {
    uint file_index;        // Index of the file containing the match
    uint byte_offset;       // Byte offset of the match in the file
    uint line_number;       // Line number (computed post-search or in kernel)
    uint column;            // Column offset within the line
    uint match_length;      // Length of the matched region
    uint context_start;     // Start offset of surrounding context
    uint context_len;       // Length of surrounding context
    uint reserved;          // Padding for alignment
};

// A filesystem path entry for the GPU-resident index (256 bytes)
struct GpuPathEntry {
    uchar path[224];        // UTF-8 encoded path bytes
    uint path_len;          // Actual length of the path
    uint flags;             // Bitflags: is_dir, is_hidden, is_symlink, etc.
    uint parent_idx;        // Index of parent directory in the index
    uint size_lo;           // File size (low 32 bits)
    uint size_hi;           // File size (high 32 bits)
    uint mtime;             // Last modification time (unix timestamp)
    uint reserved[2];       // Padding to reach 256 bytes total
};

#endif // SEARCH_TYPES_H
