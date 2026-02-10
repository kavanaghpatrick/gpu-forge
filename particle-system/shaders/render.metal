#include "types.h"

/// Vertex output / fragment input for particle billboard quads.
struct VertexOut {
    float4 position [[position]];
    half4  color;
};

/// Vertex shader: world-space billboard quad from alive list + SoA particle data.
///
/// vertex_id 0-3 defines quad corners as triangle strip:
///   0: (-0.5, -0.5)   1: (+0.5, -0.5)
///   2: (-0.5, +0.5)   3: (+0.5, +0.5)
///
/// instance_id indexes into alive list to get particle index,
/// then reads position/color/size/lifetime from SoA buffers.
///
/// Billboard approach: extract camera right/up from view matrix columns,
/// offset quad vertices in world space so quads always face camera.
/// Lifetime-based size shrink (30% at death) and alpha fade computed here.
vertex VertexOut vertex_main(
    uint                   vertex_id    [[vertex_id]],
    uint                   instance_id  [[instance_id]],
    device const uint*     alive_list   [[buffer(0)]],
    device const packed_float3* positions [[buffer(1)]],
    device const half4*    colors       [[buffer(2)]],
    device const half*     sizes        [[buffer(3)]],
    constant Uniforms&     uniforms     [[buffer(4)]],
    device const half2*    lifetimes    [[buffer(5)]]
) {
    // Read particle index from alive list (skip 4-uint counter header)
    uint particle_idx = alive_list[COUNTER_HEADER_UINTS + instance_id];

    // Read SoA attributes
    float3 particle_pos = positions[particle_idx];
    half4  particle_col = colors[particle_idx];
    float  base_size    = float(sizes[particle_idx]);

    // Read lifetime: half2(age, max_age)
    half2 lt = lifetimes[particle_idx];
    float age     = float(lt.x);
    float max_age = float(lt.y);
    float t = (max_age > 0.0) ? (age / max_age) : 0.0;  // 0 at birth, 1 at death

    // Lifetime-based size: shrink to 30% at death
    float display_size = base_size * (1.0 - t * 0.7);

    // Billboard quad offsets from vertex_id (triangle strip order)
    float2 quad_offsets[4] = {
        float2(-0.5, -0.5),
        float2(+0.5, -0.5),
        float2(-0.5, +0.5),
        float2(+0.5, +0.5)
    };
    float2 offset = quad_offsets[vertex_id];

    // Extract camera right and up vectors from view matrix columns
    // View matrix column 0 = right, column 1 = up (in world space, transposed)
    float3 cam_right = float3(uniforms.view_matrix[0][0],
                              uniforms.view_matrix[1][0],
                              uniforms.view_matrix[2][0]);
    float3 cam_up    = float3(uniforms.view_matrix[0][1],
                              uniforms.view_matrix[1][1],
                              uniforms.view_matrix[2][1]);

    // Offset quad vertex in world space for true billboard facing camera
    float3 world_pos = particle_pos
                     + (cam_right * offset.x + cam_up * offset.y) * display_size;

    // Transform to clip space
    float4 clip_pos = uniforms.projection_matrix * uniforms.view_matrix * float4(world_pos, 1.0);

    VertexOut out;
    out.position = clip_pos;
    // Pass color with lifetime-based alpha (already set in update kernel)
    out.color    = particle_col;
    return out;
}

/// Fragment shader: pass through particle color with alpha.
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
