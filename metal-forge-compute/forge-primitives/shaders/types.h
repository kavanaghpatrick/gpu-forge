// types.h -- Shared parameter structs between Metal shaders and Rust host.
//
// All structs are 16-byte aligned to match #[repr(C)] Rust counterparts.

#ifndef FORGE_TYPES_H
#define FORGE_TYPES_H

#include <metal_stdlib>
using namespace metal;

/// Parameters for reduce kernels.
/// Matches Rust ReduceParams in types.rs.
struct ReduceParams {
    uint element_count;
    uint _pad[3];
};

/// Parameters for prefix scan kernels.
/// Matches Rust ScanParams in types.rs.
struct ScanParams {
    uint element_count;
    uint pass;
    uint _pad[2];
};

/// Parameters for histogram kernels.
/// Matches Rust HistogramParams in types.rs.
struct HistogramParams {
    uint element_count;
    uint num_bins;
    uint _pad[2];
};

/// Parameters for stream compaction kernels.
/// Matches Rust CompactParams in types.rs.
struct CompactParams {
    uint element_count;
    uint threshold;
    uint _pad[2];
};

/// Parameters for radix sort kernels.
/// Matches Rust SortParams in types.rs.
struct SortParams {
    uint element_count;
    uint bit_offset;
    uint num_threadgroups;
    uint _pad;
};

/// Parameters for filter benchmark kernels.
/// Matches Rust FilterBenchParams in types.rs.
struct FilterBenchParams {
    uint element_count;
    uint threshold;
    uint _pad[2];
};

/// Parameters for group-by aggregate kernels.
/// Matches Rust GroupByParams in types.rs.
struct GroupByParams {
    uint element_count;
    uint num_groups;
    uint _pad[2];
};

/// Parameters for GEMM (General Matrix Multiply) kernels.
/// Matches Rust GemmParams in types.rs.
struct GemmParams {
    uint M;
    uint N;
    uint K;
    uint _pad;
};

/// Parameters for spreadsheet formula kernels.
/// Matches Rust SpreadsheetParams in types.rs.
/// formula_type: 0 = SUM, 1 = AVERAGE, 2 = VLOOKUP
struct SpreadsheetParams {
    uint rows;
    uint cols;
    uint formula_type;
    uint _pad;
};

/// Parameters for time series analytics kernels.
/// Matches Rust TimeSeriesParams in types.rs.
/// op_type: 0 = moving average, 1 = VWAP, 2 = bollinger (moving avg for POC)
struct TimeSeriesParams {
    uint tick_count;
    uint window_size;
    uint op_type;
    uint _pad;
};

/// Parameters for hash join kernels.
/// Matches Rust HashJoinParams in types.rs.
struct HashJoinParams {
    uint build_count;
    uint probe_count;
    uint table_size;
    uint _pad;
};

/// Parameters for CSV bench kernels.
/// Matches Rust CsvBenchParams in types.rs.
struct CsvBenchParams {
    uint byte_count;
    uint _pad[3];
};

/// Vectorized load helpers with bounds checking.
/// Return zero-initialized vector, then conditionally load each of 4 elements.

inline uint4 load_uint4_safe(device const uint* data, uint base_idx, uint element_count) {
    uint4 result = uint4(0);
    if (base_idx     < element_count) result.x = data[base_idx];
    if (base_idx + 1 < element_count) result.y = data[base_idx + 1];
    if (base_idx + 2 < element_count) result.z = data[base_idx + 2];
    if (base_idx + 3 < element_count) result.w = data[base_idx + 3];
    return result;
}

inline float4 load_float4_safe(device const float* data, uint base_idx, uint element_count) {
    float4 result = float4(0.0);
    if (base_idx     < element_count) result.x = data[base_idx];
    if (base_idx + 1 < element_count) result.y = data[base_idx + 1];
    if (base_idx + 2 < element_count) result.z = data[base_idx + 2];
    if (base_idx + 3 < element_count) result.w = data[base_idx + 3];
    return result;
}

#endif // FORGE_TYPES_H
