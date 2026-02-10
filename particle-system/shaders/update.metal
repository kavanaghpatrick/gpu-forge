#include "types.h"

/// Physics update kernel: apply gravity, drag, semi-implicit Euler integration,
/// boundary bounce, lifetime aging, and dead/alive list management.
///
/// Reads from alive_list_a (current frame's alive particles from emission).
/// Writes survivors to alive_list_b (for rendering this frame).
/// Writes dead particles back to dead_list (for recycling).
///
/// POC: skips grid density reads (pressure = 0) and mouse attraction.
kernel void update_physics_kernel(
    constant Uniforms&     uniforms       [[buffer(0)]],
    device uint*           dead_list      [[buffer(1)]],
    device const uint*     alive_list_a   [[buffer(2)]],  // input: current alive
    device uint*           alive_list_b   [[buffer(3)]],  // output: survivors
    device float3*         positions      [[buffer(4)]],
    device float3*         velocities     [[buffer(5)]],
    device half2*          lifetimes      [[buffer(6)]],
    device half4*          colors         [[buffer(7)]],
    device half*           sizes          [[buffer(8)]],
    uint                   tid            [[thread_position_in_grid]]
) {
    // Guard: alive_list_a counter is at offset 0 (first uint of header)
    uint alive_count = alive_list_a[0];
    if (tid >= alive_count) return;

    // Read particle index from alive_list_a (skip counter header)
    uint particle_idx = alive_list_a[COUNTER_HEADER_UINTS + tid];

    float dt = uniforms.dt;

    // --- Read SoA data ---
    float3 pos = positions[particle_idx];
    float3 vel = velocities[particle_idx];
    half2 lt = lifetimes[particle_idx];
    float age = float(lt.x);
    float max_age = float(lt.y);

    // --- Apply gravity ---
    // uniforms.gravity is a single float (e.g. -9.8)
    // Apply as velocity.y += gravity * dt
    vel.y += uniforms.gravity * dt;

    // --- Apply drag ---
    float drag = uniforms.drag_coefficient;
    vel *= (1.0 - drag * dt);

    // --- Semi-implicit Euler integration ---
    pos += vel * dt;

    // --- Boundary soft-bounce ---
    // If |position.xyz| > 10.0, reflect velocity component, dampen 0.5
    if (pos.x > 10.0) { pos.x = 10.0; vel.x = -vel.x * 0.5; }
    if (pos.x < -10.0) { pos.x = -10.0; vel.x = -vel.x * 0.5; }
    if (pos.y > 10.0) { pos.y = 10.0; vel.y = -vel.y * 0.5; }
    if (pos.y < -10.0) { pos.y = -10.0; vel.y = -vel.y * 0.5; }
    if (pos.z > 10.0) { pos.z = 10.0; vel.z = -vel.z * 0.5; }
    if (pos.z < -10.0) { pos.z = -10.0; vel.z = -vel.z * 0.5; }

    // --- Update lifetime ---
    age += dt;

    // --- Write back SoA data ---
    positions[particle_idx] = pos;
    velocities[particle_idx] = vel;
    lifetimes[particle_idx] = half2(half(age), half(max_age));

    // --- Dead or alive? ---
    if (age >= max_age) {
        // Particle died: push index back onto dead list
        device atomic_uint* dead_counter = (device atomic_uint*)&dead_list[0];
        uint dead_slot = atomic_fetch_add_explicit(dead_counter, 1, memory_order_relaxed);
        dead_list[COUNTER_HEADER_UINTS + dead_slot] = particle_idx;
    } else {
        // Particle survived: push index onto alive_list_b
        device atomic_uint* alive_b_counter = (device atomic_uint*)&alive_list_b[0];
        uint alive_slot = atomic_fetch_add_explicit(alive_b_counter, 1, memory_order_relaxed);
        alive_list_b[COUNTER_HEADER_UINTS + alive_slot] = particle_idx;
    }
}
