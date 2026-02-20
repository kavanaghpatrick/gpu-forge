---
spec: gpu-perf-phase2
phase: requirements
created: 2026-02-15
generated: auto
---

# Requirements: gpu-perf-phase2

## Summary

Optimize metal-attention GPU inference pipeline from 257 tok/s to 389+ tok/s on SmolLM-135M Q4_0 by eliminating CPU-GPU sync points (GPU argmax), reducing memory coherency overhead (StorageModePrivate), and reducing kernel dispatch overhead (fused RMSNorm+Matvec).

## User Stories

### US-1: GPU-side argmax eliminates logits readback
As a developer running GPU inference, I want the argmax computed on-GPU so that the 192KB logits readback is eliminated from the decode hot path.

**Acceptance Criteria**:
- AC-1.1: `forward_token()` returns `Result<u32, String>` (token ID) instead of `Result<Vec<f32>, String>` (logits) in the optimized path
- AC-1.2: A new `forward_token_with_logits()` preserves the current Vec<f32> return for sampling/debug
- AC-1.3: GPU argmax produces identical token IDs to CPU argmax for all test cases
- AC-1.4: Handles edge cases: NaN logits, inf/-inf, ties (deterministic lowest-index wins)

### US-2: StorageModePrivate reduces coherency overhead
As a developer, I want scratch buffers allocated with StorageModePrivate so that GPU-only buffers avoid CPU cache coherency overhead.

**Acceptance Criteria**:
- AC-2.1: All scratch buffers (scratch_q through scratch_residual) use StorageModePrivate in non-debug mode
- AC-2.2: logits_buf uses StorageModePrivate when GPU argmax is active
- AC-2.3: Debug path (`GPU_DEBUG=1`) continues to use StorageModeShared for readback
- AC-2.4: All 297 existing tests pass unchanged
- AC-2.5: hidden_a/hidden_b remain StorageModeShared (CPU embed_lookup writes to hidden_a)

### US-3: Fused RMSNorm+Matvec reduces dispatch overhead
As a developer, I want RMSNorm and the subsequent matvec fused into a single kernel so that intermediate buffer writes and kernel dispatch overhead are reduced.

**Acceptance Criteria**:
- AC-3.1: Fused kernel produces output within 1e-4 of separate rmsnorm_optimized + matvec_q4_0 for all dimensions used (576->576, 576->192, 576->1536, 1536->576, 576->49152)
- AC-3.2: Per-layer numerical drift through 30 layers does not exceed 1e-3 cumulative
- AC-3.3: Fused kernel handles both Q4_0 and F32 weight formats (lm_head may be F32)
- AC-3.4: Dispatch count per token reduced from ~420 to ~300

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | GPU argmax kernel: 2-stage parallel reduction over vocab_size (49152) floats, outputs u32 token ID + f32 max value | Must | US-1 |
| FR-2 | New `forward_token_greedy()` method returns u32 token ID directly | Must | US-1 |
| FR-3 | Preserve `forward_token()` returning Vec<f32> for sampling use cases | Must | US-1 |
| FR-4 | `alloc_buffer_private()` function in buffer.rs | Must | US-2 |
| FR-5 | GpuForwardPass scratch buffer allocation conditional on debug mode | Must | US-2 |
| FR-6 | `rmsnorm_matvec_q4_0` fused Metal kernel | Should | US-3 |
| FR-7 | `rmsnorm_matvec_f32` fused Metal kernel (for lm_head) | Should | US-3 |
| FR-8 | Fused kernel Rust dispatch wrapper with inline encode method | Should | US-3 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Decode throughput >= 389 tok/s on M4 with SmolLM-135M Q4_0 | Performance |
| NFR-2 | No regression in existing 297 tests | Correctness |
| NFR-3 | GPU_DEBUG=1 path must remain fully functional | Debuggability |
| NFR-4 | Memory usage increase < 1MB from new buffers | Resource |

## Out of Scope
- Async pipeline (addCompletedHandler + double-buffering) -- deferred per foreman analysis
- Prefill optimization (batch matmul for prompt tokens)
- Multi-model or multi-GPU support
- KV cache optimization (already uses GPU-side append)

## Dependencies
- Apple Family 7+ GPU (M1 and later) -- already validated at startup
- objc2-metal crate with MTLResourceOptions::StorageModePrivate
- Existing 297 passing tests as regression baseline
