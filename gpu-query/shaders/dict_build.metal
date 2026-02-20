//
// dict_build.metal
// GPU dictionary building kernel (placeholder for future GPU acceleration).
//
// Currently, dictionary encoding is done CPU-side after CSV/JSON parsing.
// This kernel is a placeholder for future sort-based dedup on GPU:
// 1. Copy string offsets to scratch buffer
// 2. Radix sort by string content
// 3. Parallel unique detection (compare adjacent sorted entries)
// 4. Scatter unique values + build code mapping
//

#include <metal_stdlib>
#include "types.h"
using namespace metal;

/// Placeholder: build_dictionary kernel for future GPU-side dictionary construction.
///
/// For now, dictionary encoding is handled entirely on CPU after file parsing.
/// This kernel structure is ready for when GPU-side dict building is needed
/// for very large cardinality columns (>100K distinct values).
///
/// Expected bindings (future):
///   buffer(0): raw string data (bytes)
///   buffer(1): string offsets (uint per row)
///   buffer(2): output dict codes (uint per row)
///   buffer(3): output unique string offsets
///   buffer(4): params (row_count, max_distinct)
kernel void build_dictionary(
    device const uint*   string_offsets   [[buffer(0)]],
    device uint*         dict_codes       [[buffer(1)]],
    device atomic_uint*  unique_count     [[buffer(2)]],
    uint                 tid              [[thread_position_in_grid]]
) {
    // Placeholder: identity mapping (each row gets its own code)
    // Real implementation would do sort-based dedup
    dict_codes[tid] = tid;
}
