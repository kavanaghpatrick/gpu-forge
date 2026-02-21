#include <metal_stdlib>
using namespace metal;

/// Gather u32 values: output[tid] = source[indices[tid]]
kernel void gather_u32(
    device const uint32_t* source   [[buffer(0)]],
    device const uint32_t* indices  [[buffer(1)]],
    device       uint32_t* output   [[buffer(2)]],
    constant     uint32_t& count    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    output[tid] = source[indices[tid]];
}

/// Gather u64 values: output[tid] = source[indices[tid]]
/// Indices are still u32 (element offsets, not byte offsets).
kernel void gather_u64(
    device const uint64_t* source   [[buffer(0)]],
    device const uint32_t* indices  [[buffer(1)]],
    device       uint64_t* output   [[buffer(2)]],
    constant     uint32_t& count    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    output[tid] = source[indices[tid]];
}
