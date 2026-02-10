#ifndef TYPES_H
#define TYPES_H

#include <metal_stdlib>
using namespace metal;

// Shared uniform data uploaded from CPU each frame.
// Layout must match Rust-side `Uniforms` struct exactly.
//
// MSL struct layout rules:
//   float4x4 = 64 bytes, 16-byte aligned
//   float3   = 16 bytes in struct (padded to 16), 16-byte aligned
//   float    = 4 bytes, 4-byte aligned
//   uint     = 4 bytes, 4-byte aligned
//
// Total size: 224 bytes
struct Uniforms {
    float4x4 view_matrix;           // offset 0   (64 bytes)
    float4x4 projection_matrix;     // offset 64  (64 bytes)
    float3   mouse_world_pos;       // offset 128 (16 bytes, float3 in struct)
    float    dt;                    // offset 144 (4 bytes)
    float    gravity;               // offset 148 (4 bytes)
    float    drag_coefficient;      // offset 152 (4 bytes)
    float    _pad0;                 // offset 156 (4 bytes, align to 16 for float3)
    float3   grid_bounds_min;       // offset 160 (16 bytes)
    float3   grid_bounds_max;       // offset 176 (16 bytes)
    uint     frame_number;          // offset 192 (4 bytes)
    float    particle_size_scale;   // offset 196 (4 bytes)
    uint     emission_count;        // offset 200 (4 bytes)
    uint     pool_size;             // offset 204 (4 bytes)
    float    interaction_strength;  // offset 208 (4 bytes)
    float    mouse_attraction_radius;   // offset 212 (4 bytes)
    float    mouse_attraction_strength; // offset 216 (4 bytes)
    float    _pad3;                 // offset 220 (4 bytes)
};                                  // total: 224 bytes (multiple of 16)

// Indirect draw arguments matching MTLDrawPrimitivesIndirectArguments.
struct DrawArgs {
    uint vertexCount;     // number of vertices per instance (4 for quad)
    uint instanceCount;   // number of alive particles to draw
    uint vertexStart;     // first vertex (0)
    uint baseInstance;    // first instance (0)
};

// Counter header for dead/alive list buffers.
// Layout: first 16 bytes of buffer.
//   offset 0:  count (uint, atomic counter)
//   offset 4:  12 bytes padding (align to 16)
// After header, uint indices start at byte offset 16 (index 4 in uint terms).
// Access pattern:
//   device atomic_uint* counter = (device atomic_uint*)list_buffer;
//   device uint* indices = list_buffer + 4;  // skip 4 uints (16 bytes)
constant uint COUNTER_HEADER_UINTS = 4;  // 16 bytes / 4 = 4 uints to skip

#endif // TYPES_H

