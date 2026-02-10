#include "types.h"

/// Vertex output / fragment input for particle billboard quads.
struct VertexOut {
    float4 position [[position]];
    half4  color;
};

/// Vertex shader: billboard quad from alive list + SoA particle data.
///
/// vertex_id 0-3 defines quad corners as triangle strip:
///   0: (-0.5, -0.5)   1: (+0.5, -0.5)
///   2: (-0.5, +0.5)   3: (+0.5, +0.5)
///
/// instance_id indexes into alive list to get particle index,
/// then reads position/color/size from SoA buffers.
vertex VertexOut vertex_main(
    uint                   vertex_id    [[vertex_id]],
    uint                   instance_id  [[instance_id]],
    device const uint*     alive_list   [[buffer(0)]],
    device const float3*   positions    [[buffer(1)]],
    device const half4*    colors       [[buffer(2)]],
    device const half*     sizes        [[buffer(3)]],
    constant Uniforms&     uniforms     [[buffer(4)]]
) {
    // Read particle index from alive list (skip 4-uint counter header)
    uint particle_idx = alive_list[COUNTER_HEADER_UINTS + instance_id];

    // Read SoA attributes
    float3 world_pos    = positions[particle_idx];
    half4  particle_col = colors[particle_idx];
    float  particle_sz  = float(sizes[particle_idx]);

    // Billboard quad offsets from vertex_id (triangle strip order)
    float2 quad_offsets[4] = {
        float2(-0.5, -0.5),
        float2(+0.5, -0.5),
        float2(-0.5, +0.5),
        float2(+0.5, +0.5)
    };
    float2 offset = quad_offsets[vertex_id];

    // Transform particle center to clip space
    float4x4 vp = uniforms.projection_matrix * uniforms.view_matrix;
    float4 clip_center = vp * float4(world_pos, 1.0);

    // Scale quad in clip space by particle size (projected)
    // Use a fixed screen-space-ish scaling: size relative to clip w
    float scale = particle_sz;
    clip_center.xy += offset * scale * clip_center.w;

    VertexOut out;
    out.position = clip_center;
    out.color    = particle_col;
    return out;
}

/// Fragment shader: pass through particle color.
fragment half4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}

/// Tiny compute kernel: copy alive list counter to indirect draw args instanceCount.
/// Dispatched as a single thread after emission, before render.
kernel void sync_indirect_args(
    device const uint*  alive_list    [[buffer(0)]],
    device DrawArgs*    indirect_args [[buffer(1)]],
    uint                tid           [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    // alive_list[0] is the atomic counter (first uint of counter header)
    indirect_args->instanceCount = alive_list[0];
    indirect_args->vertexCount = 4;
    indirect_args->vertexStart = 0;
    indirect_args->baseInstance = 0;
}
