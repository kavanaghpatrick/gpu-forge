---
spec: gpu-perf-phase3
phase: requirements
created: 2026-02-15
generated: auto
---

# Requirements: gpu-perf-phase3

## Summary

Implement multi-token batched forward pass (`forward_prompt`) for prefill to reach 1000+ tok/s on SmolLM-135M Q4_0 on Apple Silicon M4, leveraging SLC cache reuse via proven multi-token matvec kernels.

## User Stories

### US-1: Batch prefill forward pass
As a developer using the inference engine, I want to process an entire prompt in a single `forward_prompt()` call so that prefill runs at 1000+ tok/s instead of sequential single-token decode speed (462 tok/s).

**Acceptance Criteria**:
- AC-1.1: `forward_prompt(token_ids: &[u32])` method exists on `GpuForwardPass`
- AC-1.2: Returns the greedy argmax token ID of the last prompt token's logits
- AC-1.3: KV cache populated for all prompt tokens after return
- AC-1.4: Subsequent `forward_token_greedy()` calls work correctly (decode continues from position = prompt_len)
- AC-1.5: Prefill throughput >= 1000 tok/s for 32+ token prompts on M4

### US-2: Multi-token Metal kernels
As a GPU kernel developer, I want production multi-token matvec kernels so that weight matrices are read once and reused across all tokens in the batch via SLC cache.

**Acceptance Criteria**:
- AC-2.1: `multi_token_matvec_q4_0` kernel in production shader file (not bandwidth_test.metal)
- AC-2.2: `multi_token_matvec_q4_0_accumulate` variant with `output[tok * out_dim + row] += sum`
- AC-2.3: Both kernels handle arbitrary batch_size (1 to 2048)
- AC-2.4: Numerical correctness matches single-token matvec within float tolerance

### US-3: Prefill benchmarking
As a performance engineer, I want to benchmark prefill throughput so that I can measure and report tok/s for different prompt lengths.

**Acceptance Criteria**:
- AC-3.1: CLI benchmark reports prefill tok/s separately from decode tok/s
- AC-3.2: Supports configurable prompt lengths
- AC-3.3: Reports both prefill and decode metrics in JSON and human-readable formats

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Create `multi_token_matvec_q4_0.metal` production shader | Must | US-2 |
| FR-2 | Create `multi_token_matvec_q4_0_accumulate` variant | Must | US-2 |
| FR-3 | Add `forward_prompt()` method to `GpuForwardPass` | Must | US-1 |
| FR-4 | Allocate batch buffers dynamically based on prompt length | Must | US-1 |
| FR-5 | Layer-major processing: all tokens through layer L before L+1 | Must | US-1 |
| FR-6 | Per-token RMSNorm, RoPE, attention, SiLU within batch buffers | Must | US-1 |
| FR-7 | Batched multi-token matvec for QKV, O, gate, up, down projections | Must | US-1 |
| FR-8 | Sequential attention per-token with causal KV cache growth | Must | US-1 |
| FR-9 | Only compute lm_head logits for last token | Should | US-1 |
| FR-10 | Prewarm multi-token PSOs at init | Must | US-2 |
| FR-11 | Add prefill benchmark to CLI or bench subcommand | Should | US-3 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Prefill >= 1000 tok/s for 32+ token prompts on SmolLM-135M Q4_0 M4 | Performance |
| NFR-2 | No regression in single-token decode throughput (462+ tok/s) | Performance |
| NFR-3 | Numerical correctness: `forward_prompt` output matches sequential `forward_token` calls | Correctness |
| NFR-4 | Batch buffers freed or reused efficiently (no memory leak) | Memory |

## Out of Scope
- Multi-token attention (attention remains per-token due to causal constraint)
- Batched SiLU kernel (per-token SiLU is cheap, not worth batching)
- Batched RMSNorm kernel (per-token RMSNorm is cheap)
- Speculative decoding
- Continuous batching for multiple requests

## Dependencies
- Existing `multi_token_matvec_q4_0` kernel in `bandwidth_test.metal` (proven, copy to production)
- Existing `encode_matvec_q4_0_accumulate` helper (pattern for accumulate variant)
- Existing `GpuKVCacheSet` with `encode_kv_append()` for per-token KV population
