#ifndef SEARCH_TYPES_H
#define SEARCH_TYPES_H

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Common Constants
// ============================================================================

#define CHUNK_SIZE 4096
#define MAX_PATTERN_LEN 64
#define MAX_CONTEXT 80
#define THREADGROUP_SIZE 256
#define BYTES_PER_THREAD 64
#define MAX_MATCHES_PER_THREAD 4

// ============================================================================
// Content Search Structures (content_search + turbo_search kernels)
// ============================================================================

// Metadata for each chunk of file data loaded into the GPU buffer
struct ChunkMetadata {
    uint file_index;
    uint chunk_index;
    ulong offset_in_file;
    uint chunk_length;
    uint flags;  // Bit 0: is_text, Bit 1: is_first, Bit 2: is_last
};

// Search parameters for chunked content search kernels
struct SearchParams {
    uint chunk_count;
    uint pattern_len;
    uint case_sensitive;
    uint total_bytes;  // Total bytes across all chunks
};

// Match result written by GPU search kernels
struct MatchResult {
    uint file_index;
    uint chunk_index;
    uint line_number;
    uint column;
    uint match_length;
    uint context_start;
    uint context_len;
    uint _padding;
};

// ============================================================================
// Batch/Persistent Search Structures (batch_search_kernel)
// ============================================================================

constant uint STATUS_EMPTY = 0;
constant uint STATUS_READY = 1;
constant uint STATUS_PROCESSING = 2;
constant uint STATUS_DONE = 3;

struct SearchWorkItem {
    uchar pattern[64];
    uint pattern_len;
    uint case_sensitive;
    uint data_buffer_id;
    atomic_uint status;
    atomic_uint result_count;
    uint _padding[2];
};

struct PersistentKernelControl {
    atomic_uint head;
    atomic_uint tail;
    atomic_uint shutdown;
    atomic_uint heartbeat;
};

struct DataBufferDescriptor {
    ulong offset;
    uint size;
    uint _padding;
};

// ============================================================================
// Path Filter Structures (path_filter_kernel)
// ============================================================================

// Path flag constants
#define PATH_FLAG_IS_DIR       (1u << 0)
#define PATH_FLAG_IS_HIDDEN    (1u << 1)
#define PATH_FLAG_IS_SYMLINK   (1u << 2)
#define PATH_FLAG_IS_EXECUTABLE (1u << 3)
#define PATH_FLAG_IS_DELETED   (1u << 4)

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

// Path filter parameters
struct PathFilterParams {
    uint entry_count;       // Number of entries in the index
    uint pattern_len;       // Length of the filter pattern
    uint case_sensitive;    // 1 = case sensitive, 0 = case insensitive
    uint max_matches;       // Maximum number of matches to return
};

// ============================================================================
// Shared Utility Functions
// ============================================================================

// Case-insensitive character compare
inline bool char_eq_fast(uchar a, uchar b, bool case_sensitive) {
    if (case_sensitive) return a == b;
    uchar a_lower = (a >= 'A' && a <= 'Z') ? a + 32 : a;
    uchar b_lower = (b >= 'A' && b <= 'Z') ? b + 32 : b;
    return a_lower == b_lower;
}

#endif // SEARCH_TYPES_H
