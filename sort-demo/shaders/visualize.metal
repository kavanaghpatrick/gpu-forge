#include <metal_stdlib>
using namespace metal;
#include "types.h"

/// PCG hash for deterministic pseudo-random fill.
/// Each thread produces one random uint32 from its gid.
kernel void gpu_random_fill(
    device uint*          data   [[buffer(0)]],
    constant DemoParams&  params [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.element_count) return;
    // PCG-XSH-RR 32-bit
    uint state = gid * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    data[gid] = (word >> 22u) ^ word;
}

/// Map sorted uint32 values to HSV rainbow heatmap.
/// Writes to texture at (gid % width, gid / width).
kernel void value_to_heatmap(
    device const uint*              data   [[buffer(0)]],
    texture2d<half, access::write>  out    [[texture(0)]],
    constant DemoParams&            params [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.element_count) return;

    float t = float(data[gid]) / float(params.max_value);  // 0..1

    // HSV rainbow: H = t * 300 degrees (red -> violet), S=1, V=1
    float h = t * 5.0;  // 0..5 (6 HSV sectors, skip last to avoid wrap)
    float f = h - floor(h);
    half3 color;
    int sector = int(h);
    half q = half(1.0 - f);
    half fh = half(f);
    switch (sector) {
        case 0: color = half3(1.0h, fh,   0.0h); break;
        case 1: color = half3(q,    1.0h, 0.0h); break;
        case 2: color = half3(0.0h, 1.0h, fh  ); break;
        case 3: color = half3(0.0h, q,    1.0h); break;
        case 4: color = half3(fh,   0.0h, 1.0h); break;
        default: color = half3(1.0h, 0.0h, q  ); break;
    }

    uint x = gid % params.texture_width;
    uint y = gid / params.texture_width;
    out.write(half4(color, 1.0h), uint2(x, y));
}

/// Bar chart visualization: each column = one element, height proportional to value.
kernel void value_to_barchart(
    device const uint*              data   [[buffer(0)]],
    texture2d<half, access::write>  out    [[texture(0)]],
    constant DemoParams&            params [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    // gid is 2D: x = column, y = row (dispatched as texture_width * texture_height)
    uint x = gid % params.texture_width;
    uint y = gid / params.texture_width;
    if (gid >= params.texture_width * params.texture_height) return;

    // Sample the element for this column
    uint elem_idx = x;
    if (elem_idx >= params.element_count) {
        out.write(half4(0.0h, 0.0h, 0.0h, 1.0h), uint2(x, y));
        return;
    }

    float val = float(data[elem_idx]) / float(params.max_value);  // 0..1
    float bar_top = (1.0 - val) * float(params.texture_height);   // inverted Y

    half4 color;
    if (float(y) >= bar_top) {
        // Below bar top: colored
        float ct = val;
        color = half4(half(ct), half(0.3), half(1.0 - ct), 1.0h);
    } else {
        // Above bar top: black
        color = half4(0.0h, 0.0h, 0.0h, 1.0h);
    }
    out.write(color, uint2(x, y));
}
