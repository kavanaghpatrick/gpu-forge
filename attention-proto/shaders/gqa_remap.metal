#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// GQA (Grouped Query Attention) head remapping kernel
//
// Expands KV heads to match Q head count for standard attention dispatch.
// kv_head = q_head / group_size, where group_size = num_heads / num_kv_heads.
//
// Grid dispatch: (num_heads, seq_len, head_dim)
// Each thread copies one element from K_full[kv_head] to K_expanded[q_head].
//
// This measures buffer rebinding overhead vs inline index remapping.
// Production would fuse this into the attention kernel to avoid the copy.
// ---------------------------------------------------------------------------

kernel void gqa_remap(
    device const float* K_full [[buffer(0)]],   // [num_kv_heads, seq_len, head_dim]
    device float* K_expanded [[buffer(1)]],      // [num_heads, seq_len, head_dim]
    constant AttentionParams& params [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]        // (head, token, dim)
) {
    uint q_head = tid.x;
    uint token = tid.y;
    uint dim = tid.z;
    if (q_head >= params.num_heads || token >= params.seq_len || dim >= params.head_dim) return;

    uint group_size = params.num_heads / params.num_kv_heads;
    uint kv_head = q_head / group_size;

    uint src = (kv_head * params.seq_len + token) * params.head_dim + dim;
    uint dst = (q_head * params.seq_len + token) * params.head_dim + dim;
    K_expanded[dst] = K_full[src];
}
