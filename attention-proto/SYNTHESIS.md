# Architecture Recommendations: trait Attention<Q,K,V> on Apple Silicon Metal

> Synthesized from 8 GPU kernel prototypes, 38 KB findings, 48 criterion benchmarks, and 34 passing tests on Apple M4.

## Executive Summary

Eight prototypes were built to empirically validate design decisions for implementing `trait Attention<Q,K,V>` on Apple Silicon Metal. The key architectural recommendations are:

1. **Hand-written MSL with simdgroup_matrix** is the only viable path for competitive attention throughput. CubeCL achieves only 58-70% throughput and cannot access simdgroup_matrix from user kernels.
2. **Function constants** are the clear dispatch strategy. Compile-time specialization costs 34-63us/variant with 0% runtime overhead, versus 39% overhead for runtime function dispatch. Use lazy-compile-and-cache via PsoCache (178ns/hit).
3. **Linear attention (FLA)** beats softmax attention at ALL tested sequence lengths (N>=256, D=64). GPU kernel time is near-constant ~35us vs flash's quadratic growth. This is a design-breaking finding.
4. **PagedAttention V2** is viable with ~9% overhead at page_size=16, D=64. The 32KB threadgroup memory limit is the binding constraint (max page_size=32 for D=64).
5. **All attention variants** (RoPE, ALiBi, GQA) have negligible overhead (<0.1% of base attention compute) and are amenable to function constant specialization.
6. **Burn extension trait** works without forking. Bridge overhead is 2-17us. Backend delegation is the only remaining mechanical task.

---

## 1. Optimal Tile Sizes per Head Dimension

**Source**: Proto 1 (Flash Attention), Proto 3 (PagedAttention), Proto 6 (Linear Attention)

### Flash Attention Tile Configuration

| D | Br | Bc | Threadgroup Memory | Status | TFLOPS (N=2048) |
|---|----|----|-------------------|--------|-----------------|
| 64 | 16 | 64 | 24 KB | **Recommended** | 0.16 |
| 64 | 32 | 64 | 32 KB | At limit | Not tested |
| 64 | 16 | 128 | 44 KB | Exceeds 32KB | N/A |
| 128 | 16 | 32 | 40 KB | Exceeds 32KB | N/A |

**Threadgroup memory formula**: `Q_tile(Br*D*4) + K_chunk(Bc*D*4) + S_tile(Br*Bc*4)`

**Recommendation**: Use `Br=16, Bc=64` as default for D=64. The 32KB threadgroup limit on Apple Silicon is the hard constraint. D=128 requires smaller tiles (Br=16, Bc=16 = 24KB) with reduced throughput. Function constants should select tile size at PSO compile time based on head dimension.

### PagedAttention Page Size Budget

| D | page_size | Threadgroup Memory | Viable? |
|---|-----------|-------------------|---------|
| 64 | 8 | 8.5 KB | Yes (23.5KB headroom) |
| 64 | 16 | 13 KB | **Recommended** (19KB headroom) |
| 64 | 32 | 22 KB | Yes (10KB headroom) |
| 64 | 64 | 40 KB | No (exceeds 32KB) |
| 128 | 8 | 14 KB | Yes |
| 128 | 16 | 22 KB | **Recommended** for D=128 |
| 128 | 32 | 38 KB | No (exceeds 32KB) |

### Linear Attention Chunk Size Budget

| D | chunk_size | Threadgroup Memory (chunk_h / chunk_o) | Viable? |
|---|-----------|---------------------------------------|---------|
| 64 | 32 | 16 KB / 24 KB | **Recommended** |
| 64 | 64 | 32 KB / 40 KB | chunk_h at limit, chunk_o exceeds |
| 128 | 16 | 16 KB / 24 KB | Recommended for D=128 |
| 128 | 32 | 32 KB / 40 KB | chunk_h at limit, chunk_o exceeds |

---

## 2. Dispatch Strategy: Function Constants Win Decisively

**Source**: Proto 2 (Function Stitching), Proto 4 (Function Constants)

### Empirical Comparison

| Strategy | Compile Cost | Runtime Overhead | Cache Hit |
|----------|-------------|-----------------|-----------|
| Function constants (compile-time) | 34-63us/variant | **0%** | 178ns |
| always_inline (factored code) | N/A | **0%** (0.28%, noise) | N/A |
| noinline (real function calls) | N/A | **+39%** | N/A |
| visible_function_table (estimated) | N/A | **>=39%** (lower bound) | N/A |
| Binary archive (pre-compiled) | 82ms create | 0% | 5.9ms load (no speedup) |

### Recommended Pattern

```
trait Attention<Q, K, V> {
    // Each combination of (HEAD_DIM, BLOCK_R, BLOCK_C, VARIANT_FLAGS)
    // maps to a unique PsoKey -> lazy-compiled PSO -> cached in HashMap

    // First dispatch: ~63us cold compile + kernel execution
    // Subsequent dispatches: ~178ns cache lookup + kernel execution
    // Runtime overhead: 0% (function constants are compile-time)
}
```

**Key findings**:
- M4 Metal compiler is fast enough that binary archives provide **zero speedup** over cold compile for <=100 variants. Skip binary archives entirely.
- PsoCache HashMap lookup (178ns) is 350x faster than cold compile (63us). Lazy-compile-and-cache is optimal.
- Factoring kernels into `always_inline` helper functions has **zero measurable overhead** -- safe to use for code organization.
- Real function calls (`noinline`) cost **~20us/call** on M4 GPU -- prohibitively expensive for inner loops.
- Function stitching (`visible_function_table`) is not viable for compute kernel inner loops.

---

## 3. PagedAttention Viability

**Source**: Proto 3 (PagedAttention V2)

### Results

| Metric | Value |
|--------|-------|
| Page table overhead (N=256-512) | ~9% vs contiguous |
| Two-pass dispatch (partition + reduce) | Correct, single command buffer |
| Max page_size (D=64, 32KB limit) | 32 |
| Recommended page_size (D=64) | 16 (13KB, 19KB headroom) |
| Correctness | atol=1e-3 vs CPU FP64 reference |
| Memory leaks | 0% growth over 100 iterations |

### Architecture Recommendation

PagedAttention V2 is **viable** for KV cache management on M4 Metal:
- ~9% overhead is acceptable for production KV cache management where fragmentation avoidance is critical
- The 32KB threadgroup memory constraint is the binding limit -- not compute overhead or page table indirection
- Use `page_size` as a function constant for compile-time specialization (Proto 4: <63us per variant)
- Two-phase dispatch (partition + reduce) works cleanly in a single command buffer with implicit Metal synchronization
- Log-sum-exp reduction handles multi-partition long-context scenarios

### Caveats

- Benchmark used scalar dot products (not simdgroup_matrix) in the paged kernel, so the 9% overhead reflects page table indirection cost in isolation
- At N=1024, paged was actually faster than contiguous flash due to kernel implementation differences -- production comparison needs matched implementations
- Multi-partition reduce validated for single partition only; long-context (N>1024) multi-partition needs further testing

---

## 4. Linear Attention Crossover Point (Design-Breaking Finding)

**Source**: Proto 6 (FLA Linear Attention)

### This is the most significant finding of the entire prototype investigation.

| Metric | Flash Attention | Linear Attention | Ratio |
|--------|----------------|-----------------|-------|
| N=256 wall-clock | 388us | 205us | **0.53x** |
| N=512 wall-clock | 761us | 231us | **0.30x** |
| N=1024 wall-clock | 2381us | 280us | **0.12x** |
| GPU kernel time (N=1024) | ~2400us | **~35us** | **0.015x** |
| Scaling | O(N^2 * D) | O(N * D^2) | Linear wins at all tested N |
| Complexity | Softmax + tiled matmul | Chunk recurrence | Simpler algorithm |
| Crossover point | -- | **Below N=256** | Linear is faster everywhere tested |

### Wall-Clock Breakdown (Linear Attention)

| Component | Time | Notes |
|-----------|------|-------|
| chunk_h GPU kernel | ~15us | K^T * V outer product per chunk |
| chunk_o GPU kernel | ~20us | Q * H_cumulative per chunk |
| CPU prefix sum | ~300-380us | Buffer readback + prefix sum + re-upload |
| **Total** | **~350us** | Dominated by CPU overhead |

### With GPU Prefix Sum (Projected)

| Component | Time |
|-----------|------|
| chunk_h GPU kernel | ~15us |
| GPU prefix sum | ~10-20us (estimated) |
| chunk_o GPU kernel | ~20us |
| **Total** | **~35-55us** |

This would make linear attention **50-70x faster** than flash attention at N=1024.

### Architecture Impact

This finding changes the trait hierarchy design. Instead of:
```
trait Attention<Q,K,V> {
    fn forward(...) -> O;  // always softmax flash attention
}
```

The design should be:
```
trait Attention<Q,K,V> {
    fn forward(...) -> O;
}

trait SoftmaxAttention<Q,K,V>: Attention<Q,K,V> {
    // Flash Attention kernel (O(N^2*D))
    // Use for: exact softmax semantics, short sequences, compatibility
}

trait LinearAttention<Q,K,V>: Attention<Q,K,V> {
    // FLA chunk-based kernel (O(N*D^2))
    // Use for: long context, streaming inference, D <= 128
}
```

**Critical caveat**: Linear attention approximates softmax attention. The quality/accuracy trade-off depends on the specific model architecture and training regime. This finding validates the performance advantage; model quality validation is a separate concern.

---

## 5. Variant Overhead Summary

**Source**: Proto 7 (RoPE/ALiBi/GQA)

### All variants have negligible overhead relative to attention compute

| Variant | Mechanism | Overhead | % of Base Attention |
|---------|-----------|----------|-------------------|
| RoPE | Standalone pre-processing kernel | ~10us/head | <0.01% |
| ALiBi | Fused via function constant | ~0us (within noise) | 0% |
| GQA (gs=1) | Standalone remap kernel (full copy) | ~183us total | 0.09% |
| GQA (gs=2) | Standalone remap kernel | ~114us total | 0.06% |
| GQA (gs=4) | Standalone remap kernel | ~78us total | 0.04% |
| GQA (gs=8) | Standalone remap kernel | ~74us total | 0.04% |

**Base attention**: 32 heads, N=2048, D=64 = ~205ms total GPU time

### Dispatch Strategy for Variants

| Variant | Strategy | Rationale |
|---------|----------|-----------|
| ALiBi | Function constant (bool) | Dead-code elimination when disabled, zero overhead when enabled |
| Causal mask | Function constant (bool) | Same pattern as ALiBi |
| RoPE | Standalone kernel, always applied | No branching needed, ~10us is negligible |
| GQA | Runtime parameter (group_size) | Pure memory copy, no kernel specialization needed |

### Correctness Tolerances (Regression Baselines)

| Variant | atol | rtol | Notes |
|---------|------|------|-------|
| Flash Attention (base) | 5e-3 | 1e-2 | FP32 online softmax vs FP64 reference |
| RoPE | 1e-4 | 1e-3 | Tight -- element-wise trig has good FP32 agreement |
| ALiBi | 5e-3 | 1e-2 | Dominated by flash attention softmax accumulation |
| GQA remap | 1e-6 | 1e-6 | Exact -- pure memory copy |
| Linear Attention | 1e-3 | 1e-2 | FP32 chunk accumulation |
| PagedAttention | 1e-3 | 1e-2 | Scalar dot products (no simdgroup_matrix) |

---

## 6. Ecosystem Tool Recommendations

**Source**: Proto 5 (CubeCL), Proto 8 (Burn Extension Trait)

### CubeCL: Not Viable for Attention Kernels

| Metric | CubeCL | Hand-written MSL |
|--------|--------|-----------------|
| TFLOPS (N=1024, D=64) | 0.063 | 0.103 |
| Throughput ratio | 58-70% | 100% (baseline) |
| simdgroup_matrix ops | 0 (inaccessible) | 9 per dispatch |
| Threadgroup memory accesses | 0 (dynamic uchar[]) | 24 per dispatch |
| Function constants | 0 (wgpu limitation) | 4 per kernel |
| AIR instruction lines | 122 (50 code) | 528 (383 code) |
| Dependency crates | ~350 | ~20 |

**Verdict**: CubeCL is architecturally sound but the wgpu abstraction prevents access to Metal-specific features essential for competitive attention throughput. Use CubeCL only for simple, non-performance-critical compute kernels or cross-platform targets. For `trait Attention<Q,K,V>`, hand-written MSL is mandatory.

### Burn Extension Trait: Viable Without Forking

| Metric | Value |
|--------|-------|
| Pattern | `trait AttentionBackend: Backend` supertrait |
| Orphan rule | Solved via newtype `MetalAttentionBackend<B>` |
| Bridge overhead | 2-17us (tensor copy dominated) |
| Dependency footprint | ~15 crates (vs CubeCL's ~350) |
| Code complexity | ~150 lines, zero unsafe blocks |
| Remaining work | Backend delegation (~7 op traits, hundreds of methods) |
| Compatible Burn version | 0.20.1 |

**Verdict**: The Burn extension trait pattern works. `AttentionBackend: Backend` with a bridge function dispatching to native Metal kernels is the recommended integration path. Backend delegation is the only remaining blocker -- it is mechanical (solvable with proc-macro or ambassador crate), not architectural.

### Recommended Ecosystem Architecture

```
                    Burn 0.20+
                       |
               AttentionBackend (supertrait)
                       |
           MetalAttentionBackend (newtype)
                       |
                 Bridge function
              (2-17us overhead)
                       |
            +---------+---------+
            |                   |
     SoftmaxAttention    LinearAttention
     (flash kernel)      (FLA kernels)
            |                   |
     hand-written MSL    hand-written MSL
     simdgroup_matrix    chunk_h + chunk_o
     function constants  function constants
            |                   |
         PsoCache            PsoCache
       (178ns/hit)         (178ns/hit)
```

---

## 7. Design-Breaking Constraints

### Constraint 1: 32KB Threadgroup Memory Limit

The 32KB per-threadgroup limit on Apple Silicon is the single most impactful hardware constraint. It dictates:
- **Flash attention tile sizes**: Br=16, Bc=64 for D=64 (24KB). D=128 requires halving Bc.
- **PagedAttention page sizes**: max page_size=32 for D=64, max page_size=16 for D=128
- **Linear attention chunk sizes**: max chunk_size=32 for D=64 (chunk_o uses 24KB)
- **Variant selection**: No impact (variants add negligible memory)

All tile/page/chunk sizes should be function constants, allowing runtime selection based on head dimension while respecting the 32KB budget.

### Constraint 2: Linear Attention Dominates at All Tested Sequence Lengths

Linear attention's O(N*D^2) scaling beats softmax O(N^2*D) decisively for D=64 at all tested N (256-1024). This means:
- The trait hierarchy MUST support both softmax and linear attention as first-class backends
- Default dispatch should prefer linear attention for D<=128, N>=256
- Softmax attention remains necessary for exact softmax semantics and model compatibility
- A GPU prefix sum kernel is the highest-priority optimization for linear attention (eliminates ~300us CPU overhead)

### Constraint 3: No Runtime Function Dispatch in Compute Inner Loops

The 39% overhead from `noinline` function calls eliminates runtime polymorphism inside GPU kernels. This means:
- All kernel variant selection must happen at PSO compile time via function constants
- The trait dispatch layer operates at the host code level, not inside Metal shaders
- Function constants are cheap enough (34-63us/variant, 0% runtime overhead) to compile hundreds of specialized kernel variants

### Constraint 4: CubeCL Cannot Access simdgroup_matrix

CubeCL's wgpu abstraction layer prevents using simdgroup_matrix, function constants, and explicit threadgroup memory -- all essential for competitive attention throughput. This means:
- Hand-written MSL is mandatory for attention kernels
- CubeCL cannot be used as a portable codegen backend for this workload
- The trait implementation must include MSL shader source directly (not generated)

---

## 8. Quantitative Summary

### Performance Baselines (M4 Apple Silicon, D=64)

| Kernel | N=256 | N=512 | N=1024 | N=2048 | Scaling |
|--------|-------|-------|--------|--------|---------|
| Flash Attention | 389us | 762us | 2.42ms | 6.50ms | O(N^2) |
| Linear Attention (wall-clock) | 205us | 231us | 280us | -- | O(N) |
| Linear Attention (GPU only) | ~35us | ~35us | ~35us | -- | ~constant |
| PagedAttention (page_size=16) | 438us | 1.31ms | 1.72ms | -- | O(N^2) + 9% |

### Infrastructure Costs

| Operation | Cost | Notes |
|-----------|------|-------|
| PSO cold compile | 34-63us/variant | Near-linear scaling to 100+ variants |
| PsoCache hit | 178ns | 350x faster than cold compile |
| Binary archive | No speedup | M4 compiler too fast for archives to help |
| RoPE pre-processing | ~10us/head | Standalone kernel, negligible |
| GQA remap | 73-184us total | Scales with output tensor size |
| ALiBi fused | 0us | Function constant dead-code elimination |
| Burn bridge | 2-17us | Tensor copy dominated, grows with data size |
| GPU dispatch overhead | ~100-200us | Amortized with multi-head batching |

### Resource Utilization

| Resource | Value |
|----------|-------|
| Flash throughput (N=2048) | 0.16 TFLOPS (4-10% of MFA) |
| Flash throughput (N=1024) | 0.11 TFLOPS |
| CubeCL throughput ratio | 58-70% of hand-written |
| Memory leaks | 0% (Rust RAII + Retained<>) |
| Steady-state GPU memory | 464 KB |
| KB findings generated | 38 (target: 40-60) |
| Tests passing | 34 (+ 6 ignored generator tests) |
| Benchmark measurements | 48 across 8 groups |

---

## 9. Recommended Implementation Roadmap

### Phase A: Core Trait + Linear Attention (Highest Impact)

1. Define `trait Attention<Q,K,V>` with `forward()` method
2. Implement `LinearAttention` backend using Proto 6 chunk_h/chunk_o kernels
3. **Add GPU prefix sum kernel** to eliminate ~300us CPU overhead (highest-priority optimization)
4. Validate with chunk_size=32 (D=64) and chunk_size=16 (D=128) via function constants

### Phase B: Softmax Flash Attention Backend

5. Implement `SoftmaxAttention` backend using Proto 1 flash kernel
6. Add multi-simdgroup support (current: 1 simdgroup = 4-10% MFA throughput)
7. Add async copy / prefetching for improved memory bandwidth utilization
8. Tile size selection via function constants: `(HEAD_DIM, BLOCK_R, BLOCK_C)` -> PsoKey

### Phase C: KV Cache + Variants

9. Implement PagedAttention V2 for KV cache management (page_size=16/D=64)
10. Fuse ALiBi via function constant (zero overhead pattern validated)
11. Add RoPE standalone pre-processing kernel (~10us/head)
12. Add GQA remap kernel with runtime group_size

### Phase D: Burn Integration

13. Implement `AttentionBackend: Backend` supertrait
14. Solve Backend delegation (proc-macro or ambassador)
15. Bridge function connecting Burn tensors to native Metal kernels

### Phase E: Optimization

16. Multi-simdgroup flash attention (target: >1 TFLOPS on M4)
17. GPU prefix scan for linear attention H_cumulative
18. Register-level optimizations and async copy
19. Multi-generation support (M1/M2/M3/M4 via Metal feature set detection)
20. Metal 4 cooperative tensor migration path (when available)

---

## Appendix: Methodology

- **Hardware**: Apple M4 (10-core GPU, 16GB unified memory)
- **Benchmarking**: criterion 0.5 with 20-50 samples per measurement, GPU warmup, sub-1% CV
- **GPU timing**: MTLCommandBuffer GPUStartTime/GPUEndTime (hardware timestamps)
- **Correctness**: All kernels validated against FP64 CPU reference implementations
- **Shader validation**: All tests run with MTL_SHADER_VALIDATION=1
- **Memory safety**: Zero memory leaks verified via currentAllocatedSize() over 100-1000 iterations
- **Metal version**: MSL 3.1 with simdgroup_matrix support
- **Rust toolchain**: Stable with objc2-metal 0.3 bindings
