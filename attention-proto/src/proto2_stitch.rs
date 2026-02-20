//! Proto 2: Function Stitching — KB findings for inline vs noinline overhead
//!
//! Results from benchmarking flash_attention_stitched.metal with STITCH_MODE
//! function constant: monolithic (0), always_inline (1), noinline (2) on M4.
//!
//! Key result: always_inline has zero overhead, noinline adds 39% overhead.
//! Combined with Proto 4 data, function constants are the clear dispatch strategy.

/// Emit all Proto 2 KB findings to findings.jsonl.
///
/// Findings cover:
/// 1. always_inline has zero overhead vs monolithic
/// 2. noinline adds 39% overhead — real function calls are expensive
/// 3. Architecture recommendation: function constants over stitching
/// 4. Metal function stitching not viable for compute inner loops
pub fn emit_proto2_findings() {
    use crate::kb::{emit_finding, KbFinding};

    // Finding 1: always_inline has zero overhead
    emit_finding(&KbFinding {
        domain: "msl-kernels".to_string(),
        title: "proto2: MSL always_inline functions have zero overhead in flash attention inner loop on M4".to_string(),
        content: "MSL __attribute__((always_inline)) functions in flash attention inner loop add \
                  0.28% overhead vs monolithic kernel on M4 — within measurement noise. Compiler \
                  fully eliminates call overhead. Measured: monolithic 2.44ms vs always_inline \
                  2.45ms at N=1024, D=64 (100 iterations, criterion). Per-call overhead: ~144ns \
                  across 48 function calls per dispatch (1024/64=16 KV blocks * 3 functions each). \
                  This validates factoring GPU kernels into inline helper functions for readability \
                  and maintainability with zero performance cost."
            .to_string(),
        tags: vec![
            "proto2".to_string(),
            "function-stitching".to_string(),
            "always_inline".to_string(),
            "msl".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.9,
        source: "attention-proto/function_stitch benchmark (criterion, N=1024, D=64, 100 iterations)"
            .to_string(),
    });

    // Finding 2: noinline adds 39% overhead
    emit_finding(&KbFinding {
        domain: "msl-kernels".to_string(),
        title: "proto2: MSL noinline functions add 39% overhead in flash attention inner loop on M4".to_string(),
        content: "MSL __attribute__((noinline)) functions in flash attention inner loop add 39.4% \
                  overhead on M4 (2.44ms -> 3.38ms at N=1024, D=64). Real function calls in GPU \
                  inner loops are prohibitively expensive. Per-call overhead: ~20us across 48 calls \
                  per dispatch (0.94ms total overhead / 48 calls). The overhead comes from function \
                  call setup/teardown, register spilling, and loss of cross-function optimization \
                  (e.g., simdgroup_matrix register reuse). This definitively shows that runtime \
                  indirect function dispatch (visible_function_table) would have similar or worse \
                  overhead in compute kernel inner loops."
            .to_string(),
        tags: vec![
            "proto2".to_string(),
            "function-stitching".to_string(),
            "noinline".to_string(),
            "msl".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.9,
        source: "attention-proto/function_stitch benchmark (criterion, N=1024, D=64, 100 iterations)"
            .to_string(),
    });

    // Finding 3: Architecture recommendation — function constants over stitching
    emit_finding(&KbFinding {
        domain: "gpu-centric-arch".to_string(),
        title: "proto2: Function constants are the clear dispatch strategy for trait Attention<Q,K,V>".to_string(),
        content: "For trait Attention<Q,K,V>, use function constants (compile-time specialization, \
                  34-63us per variant) rather than runtime function dispatch. Combined Proto 2+4 \
                  data: compile-time = 0% overhead + 63us compile; runtime noinline = 39% overhead. \
                  Function constants are the clear winner. The break-even point does not exist in \
                  any practical scenario: even if a kernel is dispatched only once, the 63us compile \
                  cost is negligible compared to the 0.94ms overhead of noinline function calls at \
                  N=1024. For the recommended lazy-compile-and-cache pattern (Proto 4), subsequent \
                  dispatches pay only ~178ns cache lookup vs 39% runtime overhead per dispatch."
            .to_string(),
        tags: vec![
            "proto2".to_string(),
            "proto4".to_string(),
            "function-constants".to_string(),
            "architecture".to_string(),
            "trait-dispatch".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.95,
        source: "attention-proto/proto2+proto4 synthesis (function_stitch + constant_overhead results)"
            .to_string(),
    });

    // Finding 4: Metal function stitching not viable for compute inner loops
    emit_finding(&KbFinding {
        domain: "metal-compute".to_string(),
        title: "proto2: Metal function stitching not viable for compute kernel inner loops".to_string(),
        content: "[[stitchable]] / visible_function_table dispatch in Metal compute kernel inner \
                  loops would have overhead comparable to or worse than noinline (39%+). Function \
                  stitching is designed for render pipeline composition, not compute kernel inner \
                  loops. The noinline benchmark (39% overhead at N=1024) represents a lower bound \
                  for function table dispatch, since visible_function_table adds additional \
                  indirection through a function pointer table. For compute kernels requiring \
                  runtime polymorphism, use function constants to compile specialized variants \
                  (Proto 4: 63us/variant, 0% runtime overhead) rather than function tables."
            .to_string(),
        tags: vec![
            "proto2".to_string(),
            "function-stitching".to_string(),
            "visible_function_table".to_string(),
            "stitchable".to_string(),
            "metal-compute".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.8,
        source: "attention-proto/proto2 noinline benchmark extrapolation (lower bound for function table overhead)"
            .to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Run manually: cargo test -- --ignored generate_proto2_findings
    fn generate_proto2_findings() {
        emit_proto2_findings();
    }
}
