// scan_helpers.h -- Blelloch work-efficient threadgroup scan utility.
//
// Implements exclusive prefix scan within a threadgroup using shared memory.
// Up-sweep (reduce) phase followed by down-sweep (distribute) phase.
//
// ELEMENTS_PER_TG must be defined before including this header.
// Each thread processes 2 elements: thread i handles indices 2*i and 2*i+1.

#ifndef FORGE_SCAN_HELPERS_H
#define FORGE_SCAN_HELPERS_H

#include <metal_stdlib>
using namespace metal;

/// Perform Blelloch exclusive prefix scan on shared memory array.
///
/// @param shared   Threadgroup shared memory array of ELEMENTS_PER_TG elements.
/// @param tid      Thread index in threadgroup [0, ELEMENTS_PER_TG/2).
/// @param n        Number of elements to scan (must be power of 2 <= ELEMENTS_PER_TG).
///
/// After return, shared[i] contains the exclusive prefix sum of the original
/// shared[0..i], i.e. shared[0] = 0, shared[1] = orig[0], shared[2] = orig[0]+orig[1], etc.
///
/// Returns the total sum of all elements (stored at shared[n-1] before clear).
inline uint blelloch_scan_exclusive(
    threadgroup uint* shared,
    uint tid,
    uint n
) {
    // --- Up-sweep (reduce) phase ---
    // Build partial sums bottom-up in the tree.
    for (uint stride = 1; stride < n; stride <<= 1) {
        uint idx = (tid + 1) * (stride << 1) - 1;
        if (idx < n) {
            shared[idx] += shared[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Save total sum before clearing last element
    uint total = 0;
    if (tid == 0) {
        total = shared[n - 1];
        shared[n - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Broadcast total to all threads via shared memory
    // (We'll use the return value only from tid==0, but this is fine
    //  since only tid==0 uses it for partials.)

    // --- Down-sweep (distribute) phase ---
    // Distribute prefix sums top-down.
    for (uint stride = n >> 1; stride >= 1; stride >>= 1) {
        uint idx = (tid + 1) * (stride << 1) - 1;
        if (idx < n) {
            uint temp = shared[idx - stride];
            shared[idx - stride] = shared[idx];
            shared[idx] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return total;
}

#endif // FORGE_SCAN_HELPERS_H
