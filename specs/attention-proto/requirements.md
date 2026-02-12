---
spec: attention-proto
phase: requirements
created: 2026-02-12T08:30:00Z
generated: auto
---

# Requirements: attention-proto

## Summary

Eight GPU kernel prototypes producing 40-60 empirical KB findings to close ~30% knowledge gap for `trait Attention<Q,K,V>` implementation on Apple Silicon M4. Each prototype measures one specific aspect: tile sizes, dispatch strategies, memory constraints, throughput comparisons, codegen quality.

## User Stories

### US-1: Flash Attention Tile Size Selection
As an attention kernel developer, I want empirical throughput data for simdgroup_matrix tile sizes on M4 so that I can select optimal tile geometry for production kernels.

**Acceptance Criteria**:
- AC-1.1: Proto 1 kernel implements tiled Flash Attention with simdgroup_matrix_multiply_accumulate
- AC-1.2: Tile sweep covers Br in {16, 32, 64}, Bc in {64, 80, 128}, D in {64, 128, 256}
- AC-1.3: Each config measures TFLOPS, bandwidth GB/s, correctness vs FP64 CPU reference (atol=5e-3, rtol=1e-2 for FP16)
- AC-1.4: At least 5 KB findings on tile performance (e.g., "tile=32x128, D=64, N=2048 achieves 2.1 TFLOPS, 85% ALU occupancy")
- AC-1.5: Comparison vs metal-flash-attention published M4 numbers (within 50% expected, ours is from-scratch)

### US-2: Function Dispatch Strategy
As an attention trait designer, I want to know whether function stitching or function constants should be used for trait variant dispatch so that I can design the trait API correctly.

**Acceptance Criteria**:
- AC-2.1: Proto 2 measures [[stitchable]] function overhead in attention inner loop vs static dispatch
- AC-2.2: Proto 4 measures PSO compilation time for 1/10/50/100/500 function constant variants
- AC-2.3: Proto 4 tests binary archive creation and load time for pre-compiled variants
- AC-2.4: At least 3 KB findings on stitching overhead (ns/call, break-even point)
- AC-2.5: At least 3 KB findings on compilation overhead (ms/variant, archive speedup)
- AC-2.6: Recommendation: use stitching if overhead < 1%, function constants if compilation < 50ms/variant, else pre-compile all variants

### US-3: PagedAttention Viability
As an inference framework developer, I want to know if PagedAttention V2 works within M4's 32KB threadgroup constraint so that I can decide on KV cache memory management strategy.

**Acceptance Criteria**:
- AC-3.1: Proto 3 implements two-phase PagedAttention V2 (partition + reduce)
- AC-3.2: Block table indirection correctly gathers scattered KV blocks (bit-exact correctness vs contiguous)
- AC-3.3: Threadgroup memory usage mapped at block sizes 8/16/32/64/128 tokens
- AC-3.4: Throughput overhead measured vs Proto 1 contiguous flash attention (expected 10-25%)
- AC-3.5: At least 5 KB findings on page size, memory budget, throughput overhead

### US-4: Linear Attention Performance
As an attention mechanism researcher, I want throughput data for Flash Linear Attention chunk kernels on M4 so that I can compare linear attention vs softmax attention for long-context scenarios.

**Acceptance Criteria**:
- AC-4.1: Proto 6 ports FLA chunk_h and chunk_o kernels to MSL
- AC-4.2: Correctness verified vs CPU linear attention reference (FP64, atol=1e-4, rtol=1e-3)
- AC-4.3: Throughput measured for chunk sizes 32/64/128/256, seq_len up to 65536
- AC-4.4: Crossover point identified where linear attention becomes faster than softmax attention (expected N > 4096-8192)
- AC-4.5: At least 5 KB findings on chunk size, throughput, crossover behavior

### US-5: Attention Variant Overhead
As a model developer, I want per-variant timing for RoPE, ALiBi, and GQA modifiers on M4 so that I can select appropriate position encodings and attention patterns for my model.

**Acceptance Criteria**:
- AC-5.1: Proto 7 implements RoPE (rotary position embedding), ALiBi (linear bias), GQA (grouped query)
- AC-5.2: Each variant verified correct vs CPU reference
- AC-5.3: Per-token overhead measured: standalone and fused into Proto 1 kernel
- AC-5.4: At least 4 KB findings on per-variant microsecond overhead, % of base attention time
- AC-5.5: Validation that all three can be function-constant-specialized variants of one base kernel

### US-6: CubeCL Code Generation Quality
As an ecosystem tool evaluator, I want to compare CubeCL-generated MSL quality vs hand-written MSL so that I can decide whether CubeCL is viable for non-critical-path attention kernels.

**Acceptance Criteria**:
- AC-6.1: Proto 5 generates attention matmul via CubeCL Metal backend
- AC-6.2: Side-by-side MSL diff: instruction count, simdgroup_matrix usage (yes/no), register pressure
- AC-6.3: Performance comparison: CubeCL TFLOPS vs Proto 1 hand-written TFLOPS
- AC-6.4: At least 3 KB findings on codegen quality gap (expected 30-60% slower)
- AC-6.5: Recommendation on CubeCL viability documented

### US-7: Burn Extension Trait Feasibility
As a Burn user, I want to know if `trait AttentionBackend: Backend` can be implemented without forking Burn so that I can extend Burn with custom attention kernels.

**Acceptance Criteria**:
- AC-7.1: Proto 8 defines `trait AttentionBackend: Backend` with flash_attention method
- AC-7.2: Newtype wrapper pattern tested for orphan rule compliance (`struct MetalAttentionBackend(CubeCL<MetalRuntime>)`)
- AC-7.3: Extension trait compiles, can dispatch Proto 1 kernel via FFI
- AC-7.4: Dispatch overhead measured (microseconds, Burn API → Metal kernel)
- AC-7.5: At least 2 KB findings on trait viability, required Burn version, code complexity assessment

### US-8: Shared Infrastructure
As a prototype developer, I want reusable Metal host code (device, pipeline, encode, timing) so that I can focus on kernel-specific logic rather than Metal boilerplate.

**Acceptance Criteria**:
- AC-8.1: GpuDevice singleton for MTLDevice, MTLCommandQueue, MTLLibrary
- AC-8.2: PsoCache for function constant specialization
- AC-8.3: Encode helpers: alloc_buffer, dispatch_1d, dispatch_2d, read_buffer_slice
- AC-8.4: Timing: GPUStartTime/GPUEndTime wall-clock + optional MTLCounterSampleBuffer (feature-gated)
- AC-8.5: KB output: emit_finding writes JSON-lines for kb add ingestion

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Flash Attention kernel with simdgroup_matrix on M4 | Must | US-1 |
| FR-2 | Tile size sweep: Br x Bc x D parameter space | Must | US-1 |
| FR-3 | Function stitching overhead measurement | Should | US-2 |
| FR-4 | Function constant compilation time for N variants | Must | US-2 |
| FR-5 | Binary archive pre-compilation for 72 variants | Should | US-2 |
| FR-6 | PagedAttention V2 with block table and two-phase reduce | Must | US-3 |
| FR-7 | Threadgroup memory budget mapping at each block size | Must | US-3 |
| FR-8 | FLA chunk_h and chunk_o kernel port to MSL | Should | US-4 |
| FR-9 | Linear attention throughput at seq_len 1K-64K | Should | US-4 |
| FR-10 | RoPE, ALiBi, GQA kernels with correctness verification | Must | US-5 |
| FR-11 | Per-variant overhead measurement (standalone + fused) | Must | US-5 |
| FR-12 | CubeCL matmul codegen, MSL diff, performance comparison | Should | US-6 |
| FR-13 | Burn AttentionBackend extension trait with newtype pattern | Should | US-7 |
| FR-14 | Shared GpuDevice, PsoCache, encode helpers, timing | Must | US-8 |
| FR-15 | CPU FP64 reference implementations for all kernels | Must | Correctness |
| FR-16 | Criterion benchmarks with GPU warmup protocol | Must | Performance |
| FR-17 | Metal Shader Validation enabled for all tests | Must | Quality |
| FR-18 | KB finding output (JSON-lines format) | Must | Knowledge capture |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | All kernels fit within 32KB threadgroup memory | Hardware Constraint |
| NFR-2 | Kernel runtime > 1.2ms to dominate 0.12ms dispatch overhead | Performance |
| NFR-3 | FP16 tolerance atol=5e-3, rtol=1e-2 vs FP64 reference | Correctness |
| NFR-4 | FP32 tolerance atol=1e-5, rtol=1e-4 vs FP64 reference | Correctness |
| NFR-5 | Benchmark coefficient of variation (CV) < 5% for High confidence findings | Reproducibility |
| NFR-6 | Criterion warmup 5s, sample_size 50, measurement_time 10s | Benchmarking |
| NFR-7 | GPU warmup: 4 throwaway dispatches before measurement | Benchmarking |
| NFR-8 | Metal 3.1 language version for simdgroup_matrix | API Requirement |
| NFR-9 | macOS 14+, M4 Apple Silicon (sole target) | Platform |
| NFR-10 | Serial test execution (--test-threads=1) to prevent GPU contention | Testing |
| NFR-11 | Memory leak check: <1% growth over 1000 iterations | Quality |
| NFR-12 | No watchdog timeout: kernel time < 500ms per dispatch | Quality |

## Out of Scope

- Full `trait Attention<Q,K,V>` implementation (this is the next phase after prototypes)
- Multi-GPU or distributed inference
- Neural Engine (ANE) offloading
- Training workloads (inference-only)
- Metal 4 cooperative tensor types (too new, SDK not available)
- M1/M2/M3 hardware support (M4-only)
- Model-specific optimizations (prototypes test generic patterns)
- Production error handling, logging, CLI interfaces

## Dependencies

- Existing codebases: gpu-query, particle-system (Metal patterns)
- objc2-metal 0.3, criterion 0.5 (required)
- CubeCL 0.4 (Proto 5 only, feature-gated)
- Burn 0.16 (Proto 8 only, feature-gated)
- GPU Forge KB (output target for findings)
- Self-hosted M4 Mac Mini for CI (GitHub Actions lacks Metal runners)

## Success Metrics

- **Minimum**: 20 new KB findings (1,124 → 1,144+)
- **Target**: 40-60 fine-grained findings (5-8 per prototype)
- **Stretch**: Finding that reveals design-breaking constraint before full trait implementation begins (highest value outcome)
- **Duration**: 14 working days
- **Confidence**: High confidence findings (CV < 5%) for architecture decisions
