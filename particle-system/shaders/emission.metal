// emission.metal — Particle emission compute kernel.
//
// Allocates particles from the dead list via atomic decrement, initializes SoA
// attributes (position, velocity, lifetime, color, size) using GPU PRNG, and
// pushes onto the alive list. Supports burst emission at an arbitrary world position.

#include "types.h"
#include "prng.metal"

/// HSV to RGB conversion.
/// h: hue [0,1), s: saturation [0,1], v: value [0,1]
inline float3 hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0 - abs(fmod(h * 6.0, 2.0) - 1.0));
    float m = v - c;
    float3 rgb;
    if (h < 1.0/6.0)      rgb = float3(c, x, 0.0);
    else if (h < 2.0/6.0) rgb = float3(x, c, 0.0);
    else if (h < 3.0/6.0) rgb = float3(0.0, c, x);
    else if (h < 4.0/6.0) rgb = float3(0.0, x, c);
    else if (h < 5.0/6.0) rgb = float3(x, 0.0, c);
    else                   rgb = float3(c, 0.0, x);
    return rgb + float3(m, m, m);
}

/// Emission kernel: allocate particles from the dead list and initialize them.
///
/// Each thread attempts to pop one index from the dead list via atomic decrement,
/// initializes the particle attributes (position, velocity, lifetime, color, size),
/// and pushes the index onto the alive list via atomic increment.
///
/// Buffer layout for dead_list / alive_list:
///   [0]:   atomic counter (uint)
///   [1-3]: padding (align to 16 bytes)
///   [4..]: particle indices
kernel void emission_kernel(
    constant Uniforms&     uniforms       [[buffer(0)]],
    device uint*           dead_list      [[buffer(1)]],
    device uint*           alive_list     [[buffer(2)]],
    device packed_float3*  positions      [[buffer(3)]],
    device packed_float3*  velocities     [[buffer(4)]],
    device half2*          lifetimes      [[buffer(5)]],
    device half4*          colors         [[buffer(6)]],
    device half*           sizes          [[buffer(7)]],
    uint                   tid            [[thread_position_in_grid]]
) {
    // Guard: only emit up to emission_count particles
    if (tid >= uniforms.emission_count) return;

    // Unique seed per thread per frame for PRNG
    uint seed = tid * 1099087573u + uniforms.frame_number * 2654435761u;

    // --- Allocate from dead list ---
    // Atomic decrement the dead list counter
    device atomic_uint* dead_counter = (device atomic_uint*)&dead_list[0];
    uint prev_count = atomic_fetch_sub_explicit(dead_counter, 1, memory_order_relaxed);

    // If prev_count was 0, the atomic_fetch_sub wrapped to UINT_MAX.
    // Detect both 0 and wrap-around: any value > pool_size means underflow.
    if (prev_count == 0u || prev_count > uniforms.pool_size) {
        atomic_fetch_add_explicit(dead_counter, 1, memory_order_relaxed);
        return;
    }

    // The allocated slot is at index (prev_count - 1) in the indices array
    // Indices start at offset COUNTER_HEADER_UINTS (4 uints = 16 bytes)
    uint dead_slot = prev_count - 1u;
    uint particle_idx = dead_list[COUNTER_HEADER_UINTS + dead_slot];

    // --- Initialize particle attributes ---

    // Burst vs normal emission: burst threads use burst_position as center with 2x velocity
    bool is_burst = (tid < uniforms.burst_count);
    float3 emitter_center = is_burst ? uniforms.burst_position : float3(0.0);
    float speed_multiplier = is_burst ? 2.0 : 1.0;

    // Position: emitter center + random offset in sphere (radius 0.5)
    seed = pcg_hash(seed);
    float theta = rand_float(seed) * 2.0 * M_PI_F;
    seed = pcg_hash(seed);
    float phi = acos(1.0 - 2.0 * rand_float(seed));
    seed = pcg_hash(seed);
    float r = pow(rand_float(seed), 1.0/3.0) * 0.5; // cube root for uniform volume
    float sin_phi = sin(phi);
    positions[particle_idx] = emitter_center + float3(
        r * sin_phi * cos(theta),
        r * sin_phi * sin(theta),
        r * cos(phi)
    );

    // Velocity: random direction * speed (1.0 - 3.0 range), multiplied for burst
    seed = pcg_hash(seed);
    float v_theta = rand_float(seed) * 2.0 * M_PI_F;
    seed = pcg_hash(seed);
    float v_phi = acos(1.0 - 2.0 * rand_float(seed));
    seed = pcg_hash(seed);
    float speed = (1.0 + rand_float(seed) * 2.0) * speed_multiplier; // [1.0, 3.0) * multiplier
    float v_sin_phi = sin(v_phi);
    velocities[particle_idx] = float3(
        speed * v_sin_phi * cos(v_theta),
        speed * v_sin_phi * sin(v_theta),
        speed * cos(v_phi)
    );

    // Lifetime: half2(age=0.0, max_age=random 1.0-5.0)
    seed = pcg_hash(seed);
    float max_age = 1.0 + rand_float(seed) * 4.0; // [1.0, 5.0)
    lifetimes[particle_idx] = half2(half(0.0), half(max_age));

    // Color: random hue (HSV -> RGB), full saturation and value, alpha=1.0
    seed = pcg_hash(seed);
    float hue = rand_float(seed);
    float3 rgb = hsv_to_rgb(hue, 1.0, 1.0);
    colors[particle_idx] = half4(half(rgb.x), half(rgb.y), half(rgb.z), half(1.0));

    // Size: random 0.01 - 0.05
    seed = pcg_hash(seed);
    float particle_size = 0.01 + rand_float(seed) * 0.04; // [0.01, 0.05)
    sizes[particle_idx] = half(particle_size);

    // --- Push onto alive list ---
    device atomic_uint* alive_counter = (device atomic_uint*)&alive_list[0];
    uint alive_slot = atomic_fetch_add_explicit(alive_counter, 1, memory_order_relaxed);
    alive_list[COUNTER_HEADER_UINTS + alive_slot] = particle_idx;
}
