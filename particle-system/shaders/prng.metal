#ifndef PRNG_METAL
#define PRNG_METAL

#include <metal_stdlib>
using namespace metal;

/// PCG hash: deterministic pseudo-random number generator.
/// Maps any uint seed to a well-distributed uint output.
inline uint pcg_hash(uint seed) {
    seed = seed * 747796405u + 2891336453u;
    uint word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Generate a random float in [0, 1) from a uint seed.
inline float rand_float(uint seed) {
    return float(pcg_hash(seed)) / 4294967296.0;
}

#endif // PRNG_METAL
