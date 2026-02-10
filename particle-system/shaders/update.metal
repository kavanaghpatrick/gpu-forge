#include "types.h"

/// Grid dimension constant (must match grid.metal).
constant int GRID_DIM = 64;

/// Convert a world-space position to a grid cell coordinate.
inline int3 pos_to_cell(float3 pos, float3 grid_min, float3 grid_max) {
    float3 norm = (pos - grid_min) / (grid_max - grid_min);
    int3 cell = int3(norm * float(GRID_DIM));
    return clamp(cell, int3(0), int3(GRID_DIM - 1));
}

/// Convert a 3D grid cell coordinate to a linear index.
inline int cell_index(int3 cell) {
    return cell.z * GRID_DIM * GRID_DIM + cell.y * GRID_DIM + cell.x;
}

/// Physics update kernel: apply gravity, drag, pressure gradient from grid density,
/// semi-implicit Euler integration, boundary bounce, lifetime aging, and
/// dead/alive list management.
///
/// Reads from alive_list_a (current frame's alive particles from emission).
/// Writes survivors to alive_list_b (for rendering this frame).
/// Writes dead particles back to dead_list (for recycling).
/// Reads grid_density for pressure gradient force computation.
kernel void update_physics_kernel(
    constant Uniforms&     uniforms       [[buffer(0)]],
    device uint*           dead_list      [[buffer(1)]],
    device const uint*     alive_list_a   [[buffer(2)]],  // input: current alive
    device uint*           alive_list_b   [[buffer(3)]],  // output: survivors
    device packed_float3*  positions      [[buffer(4)]],
    device packed_float3*  velocities     [[buffer(5)]],
    device half2*          lifetimes      [[buffer(6)]],
    device half4*          colors         [[buffer(7)]],
    device half*           sizes          [[buffer(8)]],
    device const uint*     grid_density   [[buffer(9)]],
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

    // --- Pressure gradient force from grid density ---
    // Read 3x3x3 neighborhood around particle's cell and compute
    // approximate pressure gradient (density difference * direction).
    int3 center_cell = pos_to_cell(pos, uniforms.grid_bounds_min, uniforms.grid_bounds_max);
    uint center_density = grid_density[cell_index(center_cell)];

    float3 pressure_gradient = float3(0.0);
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int3 neighbor = center_cell + int3(dx, dy, dz);
                if (any(neighbor < int3(0)) || any(neighbor >= int3(GRID_DIM))) continue;
                uint neighbor_density = grid_density[cell_index(neighbor)];
                float diff = float(neighbor_density) - float(center_density);
                float3 dir = normalize(float3(dx, dy, dz));
                pressure_gradient += diff * dir;
            }
        }
    }
    vel += pressure_gradient * uniforms.interaction_strength * dt;

    // --- Mouse attraction force ---
    // Particles within attraction_radius are pulled toward the mouse world position.
    // Force magnitude follows inverse-square falloff, clamped to prevent explosion.
    float3 to_mouse = uniforms.mouse_world_pos - pos;
    float mouse_dist = length(to_mouse);
    if (mouse_dist < uniforms.mouse_attraction_radius && mouse_dist > 0.01) {
        float3 mouse_dir = to_mouse / mouse_dist;
        float force_mag = uniforms.mouse_attraction_strength / max(mouse_dist * mouse_dist, 0.01);
        force_mag = min(force_mag, 50.0); // clamp to prevent explosion
        vel += mouse_dir * force_mag * dt;
    }

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

    // --- Lifetime-based color alpha interpolation ---
    // t = 0 at birth, 1 at death; quadratic fade-out for smooth disappearance
    float t = age / max_age;
    colors[particle_idx].w = half(1.0 - t * t);

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
