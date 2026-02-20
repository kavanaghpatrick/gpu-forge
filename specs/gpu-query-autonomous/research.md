---
spec: gpu-query-autonomous
phase: research
created: 2026-02-11
generated: auto
---

# Research: gpu-query-autonomous

## Executive Summary

GPU-autonomous query execution is technically feasible on Apple Silicon. The architecture inverts the execution model -- GPU owns the query loop via pseudo-persistent kernel (completion-handler re-dispatch), CPU shrinks to parameter writer. Target: 36x latency reduction (36ms to <1ms) on 1M-row compound filter + GROUP BY. No shipping database uses this architecture; academic validation exists since 2012 but all production GPU databases use CPU-orchestrated dispatch.

## Market Context

### GPU Database Landscape (2025-2026)

| Engine | Architecture | Latency Model | Status |
|--------|-------------|---------------|--------|
| HeavyDB | CPU-orchestrated JIT | Multi-ms | Acquired by NVIDIA, abandoned |
| BlazingSQL | cuDF-based, CUDA-only | 10s of ms | Maintenance mode |
| Sirius | DuckDB extension, Substrait | 7x DuckDB (multi-ms) | Active, NVIDIA-backed |
| DuckDB | Vectorized pull-based CPU | Sub-second at scale | Production |
| cuDF/RAPIDS | DataFrame API, CUDA | Batch-oriented | Active |

**Key insight**: Every production GPU database uses CPU-orchestrated dispatch. No shipping system uses GPU-centric persistent kernel architecture. Apple Silicon UMA enables this where discrete GPU architectures cannot -- no PCIe bus latency floor, no separate memory spaces.

### Academic Validation

- Persistent threads programming model studied since 2012 for irregular GPU workloads
- Cooperative Kernels (2017): GPU multitasking with work-stealing queues
- Sirius fused operators: kernel fusion achieves 85% warp occupancy vs 32% for precompiled [KB #522]
- Proteus JIT: 2.8x speedup from query specialization

## Current Architecture Analysis

### Execution Model

```
CPU: parse SQL -> optimize -> plan -> encode Metal commands -> commit -> waitUntilCompleted -> readback
GPU:                                                          [filter] [aggregate] [sort]
```

### Bottleneck Breakdown (36ms total, 1M rows)

| Phase | Time | % | Owner |
|-------|------|---|-------|
| `waitUntilCompleted` synchronous wait | ~20ms | 55% | CPU blocked |
| GPU compute (filter + aggregate) | ~6ms | 17% | GPU |
| CPU-side sort | ~5ms | 14% | CPU |
| CPU overhead (parse, encode, readback) | ~5ms | 14% | CPU |

### What Autonomous Architecture Eliminates

- `waitUntilCompleted` entirely (GPU always running)
- Command buffer creation per query (pre-encoded)
- PSO binding per query (already bound or JIT-cached)
- Result readback copy (CPU reads unified memory pointer)
- CSV parsing at query time (pre-loaded binary columnar)

## Codebase Patterns Found

| Pattern | Location | Relevance |
|---------|----------|-----------|
| `#[repr(C)]` structs with MSL counterparts | `src/gpu/types.rs` + `shaders/types.h` | Same pattern for autonomous types |
| PSO caching via `HashMap<PsoKey, PSO>` | `src/gpu/pipeline.rs` | JIT cache uses same pattern |
| Function constant specialization | `shaders/filter.metal` lines 16-18 | AOT fallback kernel pattern |
| 3-level SIMD reduction | `shaders/aggregate.metal` | Reuse `simd_sum_int64` in fused kernel |
| `GpuDevice` for Metal init | `src/gpu/device.rs` | Autonomous executor takes same device reference |
| `QueryExecutor` with `waitUntilCompleted` | `src/gpu/executor.rs` | New executor avoids this entirely |
| `ColumnarBatch` storage | `src/storage/columnar.rs` | Binary loader extends this format |

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | All building blocks exist in Metal API; UMA enables the model |
| Metal Watchdog [KB #441] | Mitigated | Bounded 16ms time-slice + completion-handler re-dispatch |
| Lock-Free Work Queue | Medium risk | CPU-GPU memory ordering not well-defined [KB #154]; stress testing required |
| Fused Kernel Register Pressure | Medium risk | 64-group limit keeps threadgroup memory at 10KB (well under 32KB [KB #22]) |
| JIT Compilation | High viability | `newLibraryWithSource` available on macOS; ~1-2ms per unique plan structure |
| Effort Estimate | XL | 6 architectural pillars, new executor, JIT compiler, TUI integration |
| Risk Level | Medium | Main risks: memory ordering bugs, register pressure, watchdog timing |

## KB Findings Referenced

| KB ID | Finding | Impact |
|-------|---------|--------|
| #22 | Threadgroup memory 32KB API limit | GROUP BY hash table sizing: 64 groups max (10KB) |
| #89 | 16KB page alignment for makeBuffer(bytesNoCopy:) | Binary columnar buffer allocation |
| #145 | No public API to configure GPU timeout | Must use bounded time-slice |
| #149 | ICBs for GPU-encoded compute dispatches | Phase 2 optimization for GPU-driven dispatch |
| #151 | True persistent kernels NOT feasible on Metal | Forward-progress issue; use re-dispatch chain |
| #152 | Pseudo-persistent: bounded chunks + re-dispatch + MTLEvent | Primary architecture pattern |
| #154 | CPU-GPU memory ordering not well-defined in Metal | Acquire-Release on sequence_id required |
| #186 | No standalone memory fence in Metal | threadgroup_barrier always includes execution sync |
| #202 | Function constants: 84% instruction reduction | JIT and AOT specialization |
| #210 | Function constants superior to macros | Single master function, specialized at PSO creation |
| #283 | Excessive global atomics bottleneck | Threadgroup-local hash table for GROUP BY |
| #440 | Metal does not support persistent kernels | ICB dispatch chain is closest equivalent |
| #441 | Metal GPU command buffer timeout/watchdog | 16ms time-slice budget |
| #453 | ICBs closest equivalent to Work Graphs | GPU-driven conditional dispatch |
| #522 | Proteus JIT: 2.8x speedup | Validates JIT approach for query kernels |

## Recommendations

1. **Implement completion-handler re-dispatch for MVP** -- proven pattern, 0.1ms gap within budget; ICB optimization in Phase 2
2. **JIT primary with AOT fallback** -- generates optimal specialized kernels per query plan structure
3. **64-group GROUP BY limit** -- fits in 10KB threadgroup memory with safe margin; covers 95% of analytics queries
4. **Separate command queue** -- prevents autonomous re-dispatch chain from blocking fallback queries
5. **Per-type binary columnar buffers** -- matches existing ColumnarBatch layout for easier migration
