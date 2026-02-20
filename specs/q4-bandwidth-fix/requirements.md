---
spec: q4-bandwidth-fix
phase: requirements
created: 2026-02-15
generated: auto
---

# Requirements: q4-bandwidth-fix

## Summary

Fix the catastrophic Q4_0 memory access pattern in the Metal GPU inference pipeline. Replace 18-byte unaligned struct reads with aligned uint4 loads to increase read bandwidth from 1.5 GB/s to >50 GB/s, enabling >500 tok/s decode for SmolLM-135M.

## User Stories

### US-1: Achieve coalesced Q4_0 weight reads

As a GPU inference developer, I want the Q4_0 matvec kernel to use aligned 16-byte reads instead of 18-byte struct reads so that memory bandwidth utilization exceeds 25% of peak (>50 GB/s).

**Acceptance Criteria**:
- AC-1.1: New kernel `matvec_q4_0_v5_coalesced` reads weights via `device const uint4*` (16-byte aligned loads)
- AC-1.2: 7 consecutive blocks (126 bytes) are reconstructed from 8 x uint4 reads (128 bytes = 1 cache line)
- AC-1.3: Achieved Q4_0 read bandwidth exceeds 50 GB/s in standalone bandwidth test
- AC-1.4: Kernel produces results within 1e-2 of CPU reference for all SmolLM dimension pairs

### US-2: Multi-row threadgroup for reduced dispatch overhead

As a GPU inference developer, I want the new kernel to process multiple output rows per threadgroup so that dispatch count is reduced and GPU occupancy is improved.

**Acceptance Criteria**:
- AC-2.1: Threadgroup size is 256 threads (8 simdgroups)
- AC-2.2: Each threadgroup processes 8 output rows
- AC-2.3: Dispatch count for 576x576 projection drops from 576 to 72 threadgroups

### US-3: Integrate v5 kernel into forward pass

As a GPU inference developer, I want the forward pass to use the new coalesced kernel for all Q4_0 matvec operations so that end-to-end decode throughput exceeds 500 tok/s.

**Acceptance Criteria**:
- AC-3.1: `encode_matvec_q4_0` in `gpu_forward_pass.rs` dispatches v5 kernel with correct threadgroup geometry
- AC-3.2: All existing GPU correctness tests pass (gpu_correctness, fused_kernel_test, matvec unit tests)
- AC-3.3: Benchmark shows >500 tok/s decode throughput (>2x improvement from ~240 tok/s baseline)

### US-4: Update fused RMSNorm+Matvec kernel

As a GPU inference developer, I want the fused rmsnorm_matvec_q4_0 kernel to also use coalesced reads so that the fused path gets the same bandwidth improvement.

**Acceptance Criteria**:
- AC-4.1: Fused kernel uses same uint4 aligned read pattern as standalone v5
- AC-4.2: Fused kernel results match separate rmsnorm + matvec within 1e-4
- AC-4.3: Fused kernel numerical tests pass for all SmolLM dimension pairs

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Create matvec_q4_0_v5_coalesced.metal with uint4 aligned reads | Must | US-1 |
| FR-2 | Pack 7 Q4_0 blocks into 8 x uint4 reads (128 bytes = 1 cache line) | Must | US-1 |
| FR-3 | 256 threads per threadgroup, 8 rows per threadgroup | Must | US-2 |
| FR-4 | Cross-simdgroup reduction via threadgroup memory | Must | US-2 |
| FR-5 | Add Rust dispatch function for v5 kernel | Must | US-3 |
| FR-6 | Wire v5 into gpu_forward_pass.rs encode_matvec_q4_0 | Must | US-3 |
| FR-7 | Update rmsnorm_matvec_q4_0.metal with coalesced read pattern | Must | US-4 |
| FR-8 | Handle n_blocks_per_row not divisible by 7 (remainder blocks) | Must | US-1 |
| FR-9 | Prewarm v5 PSO in GpuForwardPass::from_gguf | Should | US-3 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | >50 GB/s Q4_0 read bandwidth (>25% of 194 GB/s peak) | Performance |
| NFR-2 | >500 tok/s decode throughput for SmolLM-135M | Performance |
| NFR-3 | Numerical results within 1e-2 of CPU reference | Correctness |
| NFR-4 | Works on Apple Family 7+ (M1 and later) | Compatibility |
| NFR-5 | No change to GGUF weight layout | Compatibility |

## Out of Scope

- Weight repacking / custom quantization formats
- Prefill (batch) kernel optimization (this is decode-only matvec)
- Q8_0 or other quantization format optimization
- Changes to non-Q4_0 kernels (matvec_f32, embedding, etc.)

## Dependencies

- Metal Shading Language uint4 type (16-byte aligned vector)
- Existing PsoCache/PsoKey infrastructure for PSO registration
- SmolLM-135M.Q4_0.gguf model file for integration tests and benchmarks
