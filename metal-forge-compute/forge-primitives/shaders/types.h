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

#endif // FORGE_TYPES_H
