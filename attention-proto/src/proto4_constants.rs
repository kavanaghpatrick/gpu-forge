//! Proto 4: Function Constants — compilation overhead and dispatch strategy findings
//!
//! Results from benchmarking MTLFunctionConstantValues compilation overhead,
//! MTLBinaryArchive load times, and PsoCache HashMap lookup performance on M4.

/// Emit all Proto 4 KB findings to findings.jsonl.
///
/// Findings cover:
/// 1. Function constant PSO compilation speed per variant
/// 2. Binary archive provides no speedup over cold compile
/// 3. PsoCache HashMap is optimal dispatch strategy
/// 4. Architecture recommendation: compile-time specialization via function constants
pub fn emit_proto4_findings() {
    use crate::kb::{emit_finding, KbFinding};

    // Finding 1: Function constant compilation speed on M4
    emit_finding(&KbFinding {
        domain: "metal-compute".to_string(),
        title: "proto4: Function constant PSO compilation speed on M4".to_string(),
        content: "Metal function constant PSO compilation takes 34-63us per variant on M4, \
                  enabling runtime specialization of 72+ attention variants in <5ms. \
                  Measured cold compile times: N=1 variant ~34us, N=10 ~34us/variant, \
                  N=50 ~37us/variant, N=100 ~43us/variant. Scaling is near-linear with \
                  slight per-variant increase at higher counts (cache pressure). \
                  72 variants (full combinatorial: 3 HEAD_DIM x 4 BLOCK_R x 3 BLOCK_C x 2 VARIANT) \
                  compile in ~4.5ms total (~63us/variant). This is well under the 50ms/variant \
                  threshold for runtime dispatch viability."
            .to_string(),
        tags: vec![
            "proto4".to_string(),
            "function-constants".to_string(),
            "pso-compilation".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.9,
        source: "attention-proto/constant_overhead benchmark (criterion, cold_compile group)"
            .to_string(),
    });

    // Finding 2: Binary archive provides no speedup
    emit_finding(&KbFinding {
        domain: "metal-compute".to_string(),
        title: "proto4: MTLBinaryArchive provides no speedup over cold compile on M4".to_string(),
        content: "MTLBinaryArchive load time (4.7ms for 72 variants) equals cold compilation \
                  time on M4. Archives add 82ms creation overhead with zero runtime benefit. \
                  Archive creation is dominated by addComputePipelineFunctionsWithDescriptor_error \
                  calls (~82ms for 72 variants), not serialization. Archive load (~4.7ms) provides \
                  no speedup vs fresh cold compile (~4.5ms). The M4 Metal compiler is fast enough \
                  that pre-compiled binary archives are unnecessary for attention kernel variant \
                  counts up to at least 100. Binary archives may still benefit workloads with \
                  thousands of variants or slower GPU generations."
            .to_string(),
        tags: vec![
            "proto4".to_string(),
            "binary-archive".to_string(),
            "pso-compilation".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.9,
        source: "attention-proto/constant_overhead benchmark (criterion, binary_archive group)"
            .to_string(),
    });

    // Finding 3: PsoCache HashMap is optimal dispatch strategy
    emit_finding(&KbFinding {
        domain: "gpu-perf".to_string(),
        title: "proto4: PsoCache HashMap lookup is 350x faster than cold compile on M4".to_string(),
        content: "PsoCache HashMap lookup at 178ns/variant is 350x faster than cold compilation \
                  (~63us/variant). For trait Attention<Q,K,V>, lazy compilation + cache is the \
                  recommended dispatch strategy. Total lookup time for 72 variants: ~12.8us \
                  (vs ~4.5ms cold compile). The HashMap key (PsoKey) serializes function constant \
                  values into a comparable type with negligible overhead. First access triggers \
                  cold compile (~63us), all subsequent accesses hit cache (~178ns). This validates \
                  the lazy-compile-and-cache pattern for runtime PSO management."
            .to_string(),
        tags: vec![
            "proto4".to_string(),
            "pso-cache".to_string(),
            "function-constants".to_string(),
            "dispatch-strategy".to_string(),
        ],
        confidence: 0.9,
        source: "attention-proto/constant_overhead benchmark (criterion, pso_cache group)"
            .to_string(),
    });

    // Finding 4: Architecture recommendation
    emit_finding(&KbFinding {
        domain: "gpu-centric-arch".to_string(),
        title: "proto4: Function constants enable compile-time specialization for trait Attention"
            .to_string(),
        content: "Function constant compilation is fast enough (<63us/variant on M4) that \
                  trait Attention<Q,K,V> can use compile-time specialization via function \
                  constants rather than runtime dispatch via function stitching. This simplifies \
                  the architecture significantly: each (HEAD_DIM, BLOCK_R, BLOCK_C, VARIANT) \
                  tuple maps to a unique PSO compiled on first use and cached in PsoCache HashMap. \
                  No need for MTLLinkedFunctions, MTLVisibleFunctionTable, or binary archives. \
                  Recommended pattern: define trait variants as function constant combinations, \
                  lazy-compile PSOs on first dispatch, cache with PsoKey. Total startup cost \
                  for 72 variants: <5ms. Per-dispatch cache hit: ~178ns."
            .to_string(),
        tags: vec![
            "proto4".to_string(),
            "function-constants".to_string(),
            "architecture".to_string(),
            "trait-dispatch".to_string(),
            "M4".to_string(),
        ],
        confidence: 0.85,
        source: "attention-proto/proto4 synthesis (cold_compile + binary_archive + pso_cache results)"
            .to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Run manually: cargo test -- --ignored generate_proto4_findings
    fn generate_proto4_findings() {
        emit_proto4_findings();
    }
}
