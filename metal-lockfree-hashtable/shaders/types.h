#pragma once

#include <metal_stdlib>
using namespace metal;

/// Parameters for GPU hash table operations.
/// Matches Rust HashTableParams in table.rs.
struct HashTableParams {
    uint capacity;   // Table capacity (must be power-of-2)
    uint num_ops;    // Number of insert/lookup operations
};
