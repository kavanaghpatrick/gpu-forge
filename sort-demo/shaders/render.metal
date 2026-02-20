#include <metal_stdlib>
using namespace metal;

struct VSOut {
    float4 position [[position]];
    float2 uv;
};

/// Fullscreen triangle from vertex_id (no vertex buffer).
/// 3 vertices: covers NDC [-1,1]x[-1,1].
vertex VSOut fullscreen_vertex(uint vid [[vertex_id]]) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };
    float2 uvs[3] = {
        float2(0.0, 1.0),   // bottom-left
        float2(2.0, 1.0),   // bottom-right (oversized)
        float2(0.0, -1.0)   // top-left (oversized)
    };
    VSOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

/// Sample heatmap texture and output to drawable.
fragment half4 fullscreen_fragment(
    VSOut                           in       [[stage_in]],
    texture2d<half, access::sample> heatmap  [[texture(0)]],
    sampler                         samp     [[sampler(0)]])
{
    return heatmap.sample(samp, in.uv);
}
