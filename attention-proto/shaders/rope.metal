#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embeddings) kernel
//
// Applies rotary embeddings to Q and K tensors in-place.
// Each thread handles one (token, dim_pair) element.
// Grid dispatch: (seq_len, head_dim / 2)
//
// RoPE rotation for each (token, pair):
//   angle = token / (theta_base ^ (2 * pair / head_dim))
//   [q0', q1'] = [q0 * cos(angle) - q1 * sin(angle),
//                  q0 * sin(angle) + q1 * cos(angle)]
//   Same rotation applied to K.
//
// Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
// ---------------------------------------------------------------------------

kernel void apply_rope(
    device float* q [[buffer(0)]],         // [seq_len, head_dim] - modified in-place
    device float* k [[buffer(1)]],         // [seq_len, head_dim] - modified in-place
    constant AttentionParams& params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]  // (token, dim_pair)
) {
    uint token = tid.x;
    uint pair = tid.y;
    if (token >= params.seq_len || pair >= params.head_dim / 2) return;

    float theta_base = 10000.0;
    float angle = float(token) / pow(theta_base, 2.0 * float(pair) / float(params.head_dim));
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint idx0 = token * params.head_dim + 2 * pair;
    uint idx1 = idx0 + 1;

    // Rotate Q
    float q0 = q[idx0], q1 = q[idx1];
    q[idx0] = q0 * cos_a - q1 * sin_a;
    q[idx1] = q0 * sin_a + q1 * cos_a;

    // Rotate K
    float k0 = k[idx0], k1 = k[idx1];
    k[idx0] = k0 * cos_a - k1 * sin_a;
    k[idx1] = k0 * sin_a + k1 * cos_a;
}
