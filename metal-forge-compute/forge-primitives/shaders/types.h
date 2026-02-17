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

/// Parameters for GEMM (General Matrix Multiply) kernels.
/// Matches Rust GemmParams in types.rs.
struct GemmParams {
    uint M;
    uint N;
    uint K;
    uint _pad;
};

#endif // FORGE_TYPES_H
