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

#endif // FORGE_TYPES_H
