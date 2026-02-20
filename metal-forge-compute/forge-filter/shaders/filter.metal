#include <metal_stdlib>
using namespace metal;

/// Stub kernel for filter predicate scan.
/// Will be fully implemented in Task 1.3.
kernel void filter_predicate_scan(
    device const uint* input [[buffer(0)]],
    device uint* partials [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // Stub — no-op
}
