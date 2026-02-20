---
spec: gpu-filter-v2
phase: research
created: 2026-02-20
---

# Research: forge-filter v2 → polars-gpu-filter

## Executive Summary

forge-filter v0.1.0 achieves 10.1x over Polars (unordered) and 6.8x (ordered) at 16M u32 50% selectivity. Five improvements will close the ordered gap to 10x+, add multi-column filtering, Arrow interop, NULL support, and a Polars expression plugin.

The key insight: ordered mode's 848µs bottleneck comes from **double-reading source data** (once for predicate eval, once for scatter) and 3 separate GPU dispatches. Bitmap caching via `simd_ballot` + SLC residency eliminates the double-read. The target is 580µs ordered (10x vs Polars CPU at 5,800µs).

## Current State (v0.1.0)

### Architecture
- 3-dispatch pipeline: `filter_predicate_scan` → `filter_scan_partials` → `filter_scatter`
- 1-dispatch atomic scatter for unordered mode (574µs, already 10.1x)
- Function constants: PRED_TYPE, IS_64BIT, DATA_TYPE, OUTPUT_IDX, OUTPUT_VALS
- FilterKey sealed trait: u32, i32, f32, u64, i64, f64
- Hierarchical scan for >4096 partials (supports 64M+ elements)
- SIMD prefix sum (simd_prefix_exclusive_sum) + cross-SG aggregation

### Performance Baseline (M4 Pro, 16M u32)
| Mode | Time | vs Polars |
|------|------|-----------|
| Ordered 50% sel | 848µs | 6.8x |
| Unordered 50% sel | 574µs | 10.1x |
| Ordered 1% sel | 695µs | 8.3x |
| Ordered 99% sel | 960µs | 6.0x |

### Bottleneck Analysis
1. **Double-read**: Scatter kernel re-evaluates predicate (reads source data again)
2. **3 dispatches**: Each cmd.commit()+waitUntilCompleted() adds ~97µs overhead
3. **Round-by-round scatter**: 16 rounds with threadgroup_barrier each

## Feature 1: Ordered Mode 10x+ (Target: 580µs)

### Approach: Bitmap Caching

**KB #1354** [high]: Two-phase filter with bitmask — Phase 1 outputs bitmask via `atomic_or`, Phase 2 reads cached bitmap for scatter.

**KB #1560** [verified]: `simd_ballot` returns simd_vote (uint64_t wrapper). Bit N set if thread N has pred=true. Packs 32 predicate results per simdgroup — eliminates 32x atomic contention.

**KB #1291** [high]: SLC sizes: M1 Pro 24MB, M1 Max 48MB. A 2MB predicate bitmap (16M elements) fits entirely in SLC. Pass 3 scatter gets ~95% cache hit rate.

### Implementation
1. Pass 1: Evaluate predicate + `simd_ballot` → pack into bitmap[N/32], atomic count per tile
2. Pass 2: Hierarchical prefix sum on tile counts (same as current, ~10µs)
3. Pass 3: Read bitmap (SLC hit), read source data once, scatter to offset

### Projected Performance
- Eliminate double-read: saves ~200µs (one fewer 64MB read)
- Bitmap read from SLC: ~2MB at 469 GB/s = ~4µs
- Target: 580-650µs ordered (10x-9x vs Polars)

### Alternative: Decoupled Fallback Single-Pass
**KB #1628** [verified]: Decoupled Fallback (Smith, Levien, Owens, SPAA 2025) enables single-pass chained scan on Apple Silicon without forward progress guarantees. Near-identical to Decoupled Lookback perf on NVIDIA.

**KB #1655** [verified]: `atomic_thread_fence(mem_device, seq_cst, thread_scope_device)` enables correct single-dispatch persistent kernels on Apple Silicon.

Risk: Higher implementation complexity. Bitmap caching is simpler and likely sufficient.

**Recommendation**: Start with bitmap caching (3-dispatch). If insufficient, try Decoupled Fallback.

## Feature 2: Multi-Column Filter

### Approach: Fused Single-Dispatch Kernel

**KB #813** [verified]: GPU filter is two-pass stream compaction. Multi-column AND: evaluate all predicates in one pass, combine with bitwise AND.

Current cascade approach (N separate filters + AND): 3 × 848µs = 2,544µs for 3 columns.
Fused approach: Single kernel reads all N columns per thread, evaluates all predicates, outputs single bitmap.

### Design
- Function constants per column: PRED_TYPE_A..H, DATA_TYPE_A..H, THRESHOLD_A..H
- `LOGIC_AND` function constant: AND vs OR combination
- Max 8 columns per kernel (covers 99% of real queries)
- Same 3-dispatch pipeline, just the predicate evaluation is fused

### Projected Performance
- 3-column AND: ~944µs (vs 2,544µs cascade = 2.69x speedup)
- Amortizes dispatch overhead across all columns

## Feature 3: Arrow Zero-Copy Integration

### Approach: Always Copy

**KB #1281** [verified]: `makeBuffer(bytesNoCopy:)` requires page-aligned (16KB) addresses. Arrow uses 64-byte alignment — incompatible in the general case.

**KB #243** [verified]: `StorageModeShared` buffers: zero DMA copy on Apple Silicon. CPU/GPU share same physical pages.

**Decision**: Always copy Arrow buffer to page-aligned Metal buffer.
- Copy cost: ~0.5ms for 64MB at ~100 GB/s
- Simplifies implementation (single code path)
- Acceptable overhead relative to 1ms+ filter time

### API Design
```rust
// New methods on GpuFilter
fn filter_arrow(&mut self, array: &PrimitiveArray<T>, pred: &Predicate<T>) -> Result<PrimitiveArray<T>>
fn filter_arrow_multi(&mut self, batch: &RecordBatch, preds: &[ColumnPredicate]) -> Result<RecordBatch>
```

### Dependencies
- `arrow-rs` (arrow-array, arrow-buffer, arrow-schema) as optional dependency
- Feature flag: `arrow` in Cargo.toml

## Feature 4: NULL Bitmap Support

### Approach: Exclude NULLs (Polars Semantics)

**Decision**: NULLs never appear in output. Matches Polars `filter()` behavior.

### GPU Implementation
- Add `HAS_NULLS` function constant (bool)
- When HAS_NULLS=true, predicate becomes: `is_valid(element) && evaluate_predicate(element)`
- Validity bitmap: 1 bit per element, packed u32 words
- Read pattern: 32 threads → 1 cache line (perfect coalescing)
- `simd_ballot` on combined predicate → same bitmap caching pipeline

### Overhead
- ~5% for bitmap read + AND with validity
- Negligible when HAS_NULLS=false (branch eliminated at compile time)

## Feature 5: Polars Expression Plugin

### Architecture: polars-gpu-filter Crate

**Web research**: Polars expression plugins use `pyo3-polars` + `polars_expr` macro. Plugin receives `&[Series]`, returns `Series`. For filter: return `BooleanChunked` mask.

### Two-Crate Design
1. `forge-filter` v0.2.0 — Rust library with Arrow support (no Python dependency)
2. `polars-gpu-filter` — thin wrapper using pyo3-polars, depends on forge-filter

### Plugin API
```python
import polars as pl
from polars_gpu_filter import gpu_filter

df.filter(gpu_filter(pl.col("price") > 100))
df.filter(gpu_filter((pl.col("price") > 100) & (pl.col("qty") < 50)))
```

### Implementation
- Parse Polars expression → extract column refs + predicates
- Convert Series chunks to Arrow arrays
- Call forge-filter with Arrow API
- Return BooleanChunked from bitmap result

## GPU Knowledge Base Findings Summary

| KB ID | Domain | Finding | Confidence |
|-------|--------|---------|------------|
| #813 | gpu-centric-arch | Two-pass stream compaction pattern | verified |
| #830 | gpu-centric-arch | Atomic bump vs prefix sum allocation | verified |
| #1354 | msl-kernels | Two-phase filter with bitmask output | high |
| #1560 | simd-wave | simd_ballot for bitmap construction | verified |
| #1291 | unified-memory | SLC sizes, bitmap fits in cache | high |
| #1628 | gpu-centric-arch | Decoupled Fallback single-pass scan | verified |
| #1655 | msl-kernels | device-scope fence for cross-TG coherence | verified |
| #1660 | msl-kernels | threadgroup_barrier insufficient cross-TG | verified |
| #1281 | gpu-io | makeBuffer(bytesNoCopy:) page alignment | verified |
| #243 | gpu-io | StorageModeShared zero DMA copy | verified |
| #1293 | gpu-perf | M4 Pro 273 GB/s bandwidth | verified |
| #1643 | gpu-perf | Single dispatch >> serial dispatches | verified |
| #1166 | metal-compute | PSO compile 34-63µs per variant | high |
| #357 | simd-wave | Prefix sum for stream compaction | high |
| #56 | unified-memory | CPU/GPU virtual address constant offset | verified |
| #1376 | gpu-centric-arch | GPU index with atomic_or bitmask | high |

## Feasibility Assessment

| Feature | Feasibility | Risk | Effort |
|---------|------------|------|--------|
| Ordered 10x+ | HIGH | Medium (new kernel) | 3-5 days |
| Multi-column | HIGH | Low (function constants) | 2-3 days |
| Arrow compat | HIGH | Low (copy approach) | 2-3 days |
| NULL bitmap | HIGH | Low (simple AND) | 1-2 days |
| Polars plugin | MEDIUM | Medium (pyo3 FFI) | 3-5 days |

## Risks

1. **Ordered mode may plateau at 8-9x** if bitmap caching doesn't fully eliminate overhead
2. **Polars plugin FFI** — pyo3-polars API may change between versions
3. **Arrow dependency** — arrow-rs major version bumps could break API
4. **PSO variant explosion** — 8 columns × 8 operators × 7 types = many combinations

## Recommendations

1. **Start with bitmap caching** for ordered mode — simplest path to 10x
2. **Feature-flag Arrow** — keep forge-filter usable without arrow-rs dependency
3. **Lazy PSO** — compile on first use, don't pre-build all combinations
4. **Two-crate strategy** — forge-filter (Rust) + polars-gpu-filter (Python/Polars)
5. **Extend API** — no breaking changes from v0.1.0
