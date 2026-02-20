---
spec: q4-bandwidth-fix
phase: research
created: 2026-02-15
generated: auto
---

# Research: q4-bandwidth-fix

## Executive Summary

The current `matvec_q4_0` Metal kernel achieves only 1.5 GB/s read bandwidth (0.8% of 194 GB/s peak) due to unaligned 18-byte `BlockQ4_0` struct reads. The fix is to replace struct-based reads with raw aligned `uint4` (16-byte) loads and reconstruct block data from cache-line-aligned reads. Combined with multi-row threadgroups and half-precision accumulation, this should achieve >50 GB/s and >500 tok/s decode.

## Codebase Analysis

### Existing Patterns

| File | Pattern | Relevance |
|------|---------|-----------|
| `shaders/matvec_q4_0.metal` | 32 threads/TG, 1 simdgroup, struct BlockQ4_0 read per thread | Current baseline (70 lines) |
| `shaders/rmsnorm_matvec_q4_0.metal` | Same struct read pattern + inline RMS computation | Must also be updated |
| `shaders/matvec_q4_0_v2_vec4.metal` | float4 vectorized inner loop, still 18-byte struct reads | Improved compute, same memory issue |
| `shaders/matvec_q4_0_v3_multirow.metal` | 256 threads, 8 simdgroups, threadgroup memory reduction | Multi-row pattern to follow |
| `shaders/matvec_q4_0_v4_simdgroup.metal` | simdgroup_matrix (overkill for matvec) | Abandoned approach |
| `shaders/bandwidth_test.metal` | float4/half4 coalesced reads for peak BW measurement | Diagnostic reference |
| `src/matvec_q4_0.rs` | dispatch_matvec_q4_0() with unit tests for all SmolLM dims | Tests to validate against |
| `gpu_forward_pass.rs` | encode_matvec_q4_0() / encode_fused_rmsnorm_matvec_q4_0() | Integration points to swap |
| `pipeline.rs` | PsoKey::simple("matvec_q4_0") pattern | PSO registration for new kernel |

### Dependencies

- `metal_attention_kernels`: buffer alloc, dispatch helpers, PsoCache/PsoKey
- `objc2_metal`: MTLSize, MTLComputeCommandEncoder bindings
- `half` crate: f16 encoding for test helpers
- Apple Metal Shading Language: uint4 type, simd_sum, threadgroup_barrier

### Constraints

- BlockQ4_0 = 18 bytes (NOT power of 2) -- root cause of unaligned reads
- Cache lines are 128 bytes on Apple GPU
- GGUF weight format is fixed (cannot repack weights)
- SmolLM-135M has n_blocks_per_row = 18 for most projections (576/32)
- Must maintain bit-identical results (or within 1e-2 for half precision)
- Apple Family 7+ (M1+) required for simd_sum

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | uint4 aligned reads are standard Metal practice; 7 blocks = 126 bytes fits 1 cache line |
| Effort Estimate | M | New kernel + fused variant + Rust dispatch + integration wiring + tests |
| Risk Level | Low | Correctness easy to validate with existing CPU reference tests |

### Key Insight: 7-Block Cache Line Packing

- 7 blocks x 18 bytes = 126 bytes
- 8 x uint4 = 128 bytes (exactly 1 cache line)
- Reading 8 aligned uint4s gives us 128 bytes containing 7 complete blocks + 2 padding bytes
- This transforms scattered 18-byte reads into a single coalesced cache line fetch
- For n_blocks_per_row=18: process 2 groups of 7 + 1 group of 4 remaining blocks

### Multi-Row Benefits

- Current: 1 row per threadgroup, 32 threads (1 simdgroup)
- Proposed: 8 rows per threadgroup, 256 threads (8 simdgroups)
- Reduces dispatch count 8x (576 -> 72 for hidden projections)
- Better GPU occupancy from more threads per threadgroup

## Recommendations

1. Implement `matvec_q4_0_v5_coalesced.metal` with uint4 aligned reads + 8-row multi-row
2. Wire v5 into forward pass via PSO name swap (minimal Rust changes)
3. Update fused `rmsnorm_matvec_q4_0` with same coalesced read pattern
4. Validate with existing unit tests + benchmark for BW measurement
