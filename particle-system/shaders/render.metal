// render.metal — Particle rendering: billboard quad vertex/fragment shaders
// and sync_indirect_args compute kernel.
//
// Vertex shader constructs world-space billboard quads from SoA particle data,
// with lifetime-based size shrink and alpha fade. Fragment shader passes through
// color. sync_indirect_args copies alive count to indirect draw arguments.

#include "types.h"

/// Vertex output / fragment input for particle billboard quads.
/// Color uses half4 for bandwidth: [0,1] RGBA fits FP16 precision.
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
///
/// --- FP16 Buffer Reads ---
/// Colors (half4), sizes (half), and lifetimes (half2) are read as FP16
/// directly from SoA buffers. The vertex shader promotes size/lifetime to
/// float for world-space transform math (needs full precision for MVP),
/// but keeps color as half4 through to fragment output.
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

/// Tiny compute kernel: copy alive list counter to indirect draw args instanceCount,
/// and compute next-frame update/grid_populate dispatch threadgroup count.
/// Dispatched as a single thread after update, before render.
kernel void sync_indirect_args(
    device const uint*    alive_list    [[buffer(0)]],
    device DrawArgs*      indirect_args [[buffer(1)]],
    device DispatchArgs*  update_args   [[buffer(2)]],
    uint                  tid           [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // alive_list[0] is the atomic counter (first uint of counter header)
    uint alive_count = alive_list[0];

    // Write indirect draw arguments
    indirect_args->instanceCount = alive_count;
    indirect_args->vertexCount = 4;
    indirect_args->vertexStart = 0;
    indirect_args->baseInstance = 0;

    // Write next-frame update/grid_populate indirect dispatch arguments
    uint threadgroups = max((alive_count + 255) / 256, 1u);
    update_args->threadgroupsPerGridX = threadgroups;
    update_args->threadgroupsPerGridY = 1;
    update_args->threadgroupsPerGridZ = 1;
}
