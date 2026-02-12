---
spec: attention-proto
phase: research
created: 2026-02-12T08:30:00Z
generated: auto
---

# Research: attention-proto

## Executive Summary

Eight targeted prototypes will close ~30% knowledge gap for `trait Attention<Q,K,V>` on Apple Silicon Metal. Focus: empirical verification of Flash Attention tile sizes, function dispatch strategies, PagedAttention memory constraints, linear attention viability, and ecosystem tooling (CubeCL, Burn). Target: 40-60 fine-grained KB findings. Duration: 14 days. Hardware: M4-only.

## Codebase Analysis

### Existing Patterns (from gpu-query, particle-system)

- **Device initialization**: `gpu-query/src/gpu/device.rs` — MTLDevice + MTLCommandQueue singleton
- **Pipeline caching**: `gpu-query/src/gpu/pipeline.rs` — PsoCache maps function name + constants to compiled PSO
- **Compute dispatch**: `gpu-query/src/gpu/encode.rs` — dispatch_1d, alloc_buffer, read_buffer_slice
- **GPU timing**: `gpu-query/src/gpu/metrics.rs` — GpuTimer wall-clock, plus MTLCommandBuffer GPUStartTime/GPUEndTime
- **build.rs**: Compile .metal → .air → .metallib via xcrun metal
- **Type safety**: particle-system shaders/types.h + Rust #[repr(C)] with layout assertions
- **Integration tests**: gpu-query/tests/autonomous_integration.rs — CPU FP64 reference + GPU verification

### Dependencies to Leverage

- objc2-metal 0.3, objc2-foundation 0.3, block2 0.6 — Metal bindings (already in use)
- criterion 0.5 — statistical benchmarking
- CubeCL 0.4 (Proto 5), Burn 0.16 (Proto 8) — behind feature flags

### Constraints

- **32KB threadgroup memory** — M4 hardware limit, binding constraint for tile sizes
- **120 GB/s bandwidth, 4.4 TFLOPS FP32** — M4 theoretical peaks for efficiency calculations
- **0.12ms dispatch overhead** — problem sizes must be large enough (kernel time > 1.2ms)
- **simdgroup_matrix** — requires Metal 3.1, Apple GPU Family 9 (M3/M4)
- **Metal Shader Validation** — MTL_SHADER_VALIDATION=1 enabled for all tests, exclusive with GPU capture

## External References

### Metal Flash Attention State of the Art

Philip Turner's [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) is the reference. Key findings:
- Warped tile aspect ratios: 16-32 rows, 80-128 columns minimize register spilling
- simdgroup_matrix + simdgroup_async_copy (undocumented) for compute-memory overlap
- Intentional register spilling to threadgroup memory (32KB constraint)
- 1790 GINSTRS on M4, up to 20% faster inference vs non-Flash baselines

We write from scratch to learn internals, use MFA for validation.

### CubeCL/Burn Ecosystem

CubeCL compiles Rust #[cube] functions to Metal via wgpu/Naga (Rust IR → WGSL → MSL). [Burn Metal performance issues](https://github.com/tracel-ai/burn/issues/3463) show 0.03 tok/s vs 20 tok/s CPU — suggests MSL codegen does not leverage simdgroup_matrix. Proto 5 measures this gap empirically.

Burn's Backend trait [supports custom operations](https://github.com/tracel-ai/burn) — Proto 8 tests AttentionBackend extension trait viability.

### Flash Linear Attention (FLA)

[FLA repository](https://github.com/fla-org/flash-linear-attention) provides Triton chunk_h/chunk_o kernels. Linear attention replaces softmax with linear recurrence (O(N) vs O(N²)). Chunk-based parallelization fits D×D hidden state in memory. No existing Metal port — Proto 6 ports from Triton, requires algorithmic adaptation (CUDA warp shuffle → Metal simdgroup operations).

[TFLA paper](https://arxiv.org/abs/2503.14376) (NeurIPS 2025) shows linear attention outperforms Flash Attention at long contexts (N > 4096-8192).

### PagedAttention on Metal

[vLLM PagedAttention](https://docs.vllm.ai/en/stable/design/paged_attention/) partitions KV cache into fixed-size blocks. V2 uses two-phase reduce (partition + combine). [vllm-mlx](https://arxiv.org/html/2601.19139) shows 21-87% higher throughput vs llama.cpp. No existing Metal implementation — Proto 3 tests viability under 32KB threadgroup constraint.

### GPU Kernel Benchmarking Best Practices

[Jan.ai benchmarking guide](https://www.jan.ai/post/how-we-benchmark-kernels) identifies pitfalls:
- Hardware-specific testing mandatory (one GPU != another)
- Warm-up runs prime caches, trigger clock boost
- Correctness before performance ([KernelBench, Stanford](https://scalingintelligence.stanford.edu/blogs/kernelbench/) — optimized code has higher error rates)
- Metal dispatch overhead 0.12ms vs 0.001ms CUDA — problem sizes must be large

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| **Technical Viability** | High | All APIs tested exist, reference implementations available |
| **Effort Estimate** | M (14 days) | 8 prototypes with shared infrastructure, parallelizable after Proto 1 |
| **Risk Level** | Medium | simdgroup_matrix undocumented, 32KB constraint tight for PagedAttention V2, FLA port may be hard |
| **KB Value** | High | Fills 30% gap with empirical data on target hardware (M4) |
| **Architecture Impact** | Critical | Findings inform `trait Attention<Q,K,V>` dispatch strategy, memory layout, variant design |

### Specific Feasibility by Prototype

| # | Prototype | Feasibility | Key Constraint |
|---|-----------|------------|----------------|
| 1 | Flash Attention | High | Reuse MFA tile sizes as starting point, simdgroup_matrix well-tested on M4 |
| 2 | Function Stitching | Medium | [[stitchable]] in compute kernels undocumented, may not be supported |
| 3 | PagedAttention V2 | Medium | 32KB may force small page sizes (16-32 tokens), indirection overhead unknown |
| 4 | Function Constants | High | Standard Metal API, PsoCache pattern already exists |
| 5 | CubeCL MSL Quality | Medium | CubeCL Metal backend may crash or emit invalid MSL (feature-gated) |
| 6 | FLA chunk kernels | Medium-Low | Triton → MSL port requires algorithmic redesign, not direct translation |
| 7 | RoPE/ALiBi/GQA | High | Simple modifiers, low risk |
| 8 | Burn Extension Trait | High | Rust-only, no Metal code, newtype pattern should work |

## Recommendations

1. **Proto 1 (Flash Attention) is foundation** — all others depend on it. Start here, validate tile sizes, establish TFLOPS baseline.
2. **Proto 4 (Function Constants) early** — determines dispatch strategy (compile-time vs runtime). Run in Week 1 alongside Proto 1.
3. **Proto 6 (FLA) can start independently** — different kernel, can parallelize with Proto 1.
4. **Proto 5, 8 (ecosystem) defer to Week 3** — CubeCL/Burn integration may require debugging, non-blocking if they fail.
5. **If schedule compresses** — cut Proto 5 (CubeCL), defer Proto 8 (Burn trait). Core kernel findings (Protos 1-4, 6-7) are non-negotiable.
6. **Thermal management** — use desktop form factor (Mac Mini) or add cooldown periods between benchmarks. Laptop M4 throttles under sustained load.

## Next Steps

1. Generate requirements.md — user stories, acceptance criteria, per-prototype success metrics
2. Generate design.md — project structure, shared infrastructure, all 8 prototype MSL designs
3. Generate tasks.md — POC-first implementation tasks with verify commands
