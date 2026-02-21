---
spec: gpu-filter-v2
phase: requirements
created: 2026-02-20
---

# Requirements: forge-filter v2 (polars-gpu-filter)

## Goal

Upgrade forge-filter from 6.8x to 10x+ ordered mode (vs Polars CPU), add multi-column GPU-side AND/OR, Arrow buffer integration, NULL bitmap handling, and a Polars expression plugin -- creating the first GPU-accelerated filter for Polars on Apple Silicon.

## User Stories

### US-1: 10x+ Ordered Filter Performance

**As a** Rust developer using forge-filter in a time-series pipeline
**I want to** filter 16M rows with ordered output in under 580us
**So that** downstream aggregation gets deterministic results at 10x+ over Polars CPU

**Acceptance Criteria:**
- [ ] AC-1.1: Ordered filter at 16M u32, 50% selectivity runs in <=580us on M4 Pro (10x vs Polars 5,800us)
- [ ] AC-1.2: Ordered filter at 16M u32, 1% selectivity runs in <=450us (>=12x vs Polars)
- [ ] AC-1.3: Output is bit-identical to current v0.1.0 3-dispatch ordered pipeline for all types and selectivities
- [ ] AC-1.4: All 6 numeric types (u32, i32, f32, u64, i64, f64) achieve >=10x ordered at 50% selectivity
- [ ] AC-1.5: Existing v0.1.0 API (`filter()`, `filter_u32()`, etc.) unchanged -- no breaking changes
- [ ] AC-1.6: 10,000 correctness iterations pass (bitmap-cached output vs v0.1 3-dispatch oracle) with zero mismatches

**Implementation Approach:** Bitmap caching via `simd_ballot` -- Pass 1 evaluates predicate and writes 1-bit/element bitmap (2MB for 16M, fits in SLC). Pass 3 reads cached bitmap instead of re-evaluating predicate against 64MB source. Eliminates double-read bottleneck.

### US-2: Multi-Column GPU Filter

**As a** Rust developer implementing SQL WHERE clauses
**I want to** evaluate `WHERE age > 30 AND salary > 50000` in a single GPU dispatch
**So that** multi-predicate queries run 2-3x faster than cascading separate filters

**Acceptance Criteria:**
- [ ] AC-2.1: `filter_mask_multi()` accepts N=1..8 column+predicate pairs with AND or OR logic
- [ ] AC-2.2: 2-column AND at 16M rows completes in <=900us (vs ~1,700us cascade baseline)
- [ ] AC-2.3: 3-column AND at 16M rows completes in <=1,000us (vs ~2,544us cascade)
- [ ] AC-2.4: Output `BooleanMask` is a compact u8 array, LSB-first bit ordering (matches Arrow validity bitmap layout), ceil(n/8) bytes
- [ ] AC-2.5: AND/OR result matches element-wise CPU reference for all type/predicate combinations
- [ ] AC-2.6: Mixed-type columns supported (e.g., u32 age + f64 salary in one call)
- [ ] AC-2.7: Returns `FilterError` (not panic) for N=0 or N>8

### US-3: Arrow Buffer Integration

**As a** Rust developer using Arrow/Polars DataFrames
**I want to** filter Arrow PrimitiveArrays without manual buffer allocation
**So that** I avoid boilerplate and minimize copy overhead

**Acceptance Criteria:**
- [ ] AC-3.1: `filter_arrow(&PrimitiveArray<T>, &Predicate<T>)` returns `PrimitiveArray<T>` with matching values
- [ ] AC-3.2: `filter_arrow_multi(&RecordBatch, &[ColumnPredicate])` returns `BooleanArray` mask
- [ ] AC-3.3: Arrow data copied to page-aligned Metal buffer internally (always-copy approach per user decision)
- [ ] AC-3.4: Copy overhead <=0.5ms for 64MB (16M u32) on M4 Pro
- [ ] AC-3.5: End-to-end Arrow-to-result at 16M rows <=1,100us (filter time + copy overhead)
- [ ] AC-3.6: Works for all 6 numeric Arrow types (UInt32, Int32, Float32, UInt64, Int64, Float64)
- [ ] AC-3.7: Arrow integration gated behind `arrow` feature flag in Cargo.toml
- [ ] AC-3.8: Nullable Arrow arrays pass validity bitmap to GPU kernel (see US-4)

### US-4: NULL Bitmap Support

**As a** developer filtering nullable columns
**I want to** have NULL rows automatically excluded from filter results
**So that** GPU results match Polars/SQL WHERE semantics exactly

**Acceptance Criteria:**
- [ ] AC-4.1: When filtering a nullable column, rows with NULL validity bit=0 are excluded from output (never match any predicate)
- [ ] AC-4.2: Result matches `polars_series.filter(polars_series > threshold)` for nullable Series
- [ ] AC-4.3: NULL bitmap check adds <=5% overhead vs non-null path at 16M rows
- [ ] AC-4.4: `HAS_NULLS` function constant gates bitmap read logic -- non-nullable columns pay zero overhead (branch eliminated at compile time)
- [ ] AC-4.5: Boundary correctness: NULLs at positions 0, 7, 8, 15, 31, 32 (byte/word boundaries) correctly excluded
- [ ] AC-4.6: All-NULL input returns empty result
- [ ] AC-4.7: No-NULL input (validity bitmap all 1s) produces identical output to non-nullable path
- [ ] AC-4.8: NaN and NULL treated independently -- NaN follows IEEE 754, NULL always excluded

### US-5: Polars Expression Plugin

**As a** Python data scientist on Apple Silicon
**I want to** run `df.filter(gpu_filter(pl.col("price") > 100))` in my Polars pipeline
**So that** I get 10x+ filter speedup without changing my workflow

**Acceptance Criteria:**
- [ ] AC-5.1: `pip install polars-gpu-filter` succeeds on macOS 14+ arm64 with no manual build steps
- [ ] AC-5.2: `df.filter(gpu_filter(pl.col("col") > threshold))` produces identical results to `df.filter(pl.col("col") > threshold)`
- [ ] AC-5.3: Plugin returns `BooleanChunked` mask via `#[polars_expr(output_type=Boolean)]` pattern
- [ ] AC-5.4: End-to-end latency at 16M rows <=850us (including Series extraction + GPU dispatch + mask construction)
- [ ] AC-5.5: Supports all Polars numeric dtypes: Int32, Int64, UInt32, UInt64, Float32, Float64
- [ ] AC-5.6: Graceful CPU fallback when no Metal GPU detected -- returns Polars-native expression result + warning log
- [ ] AC-5.7: Multi-column filter via `gpu_filter((pl.col("a") > 10) & (pl.col("b") < 50))` with GPU-side AND
- [ ] AC-5.8: Nullable Series handled correctly (NULLs excluded per US-4)
- [ ] AC-5.9: Two-crate architecture: `forge-filter` (Rust engine) + `polars-gpu-filter` (pyo3-polars wrapper)

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | Bitmap caching in ordered pipeline: Pass 1 writes 1-bit/element bitmap via `simd_ballot`, Pass 3 reads cached bitmap (SLC-resident) instead of re-evaluating predicate | P0 | AC-1.1 through AC-1.4: <=580us ordered at 16M 50% sel, bit-identical output |
| FR-2 | Preserve v0.1.0 3-dispatch pipeline as fallback for Metal <3.2 (M1 devices) | P0 | AC-1.5: existing API unchanged; M1 gets v0.1 performance levels |
| FR-3 | `filter_mask_multi()` API: accepts Vec of (FilterBuffer, Predicate) pairs + LogicOp::And/Or, returns BooleanMask | P0 | AC-2.1 through AC-2.5: <=900us 2-col AND at 16M |
| FR-4 | Fused multi-column Metal kernel with function constants per column (N_COLUMNS, PRED_TYPE_A..H, DATA_TYPE_A..H, LOGIC_AND) | P0 | AC-2.2, AC-2.3: single dispatch evaluates all columns |
| FR-5 | `filter_arrow()` and `filter_arrow_multi()` methods on GpuFilter accepting Arrow PrimitiveArray/RecordBatch | P0 | AC-3.1 through AC-3.6 |
| FR-6 | Arrow data always copied to page-aligned Metal buffer (no conditional zero-copy paths) | P0 | AC-3.3, AC-3.4: copy <=0.5ms for 64MB |
| FR-7 | `arrow` feature flag in Cargo.toml gating arrow-rs dependency | P0 | AC-3.7: forge-filter usable without arrow-rs |
| FR-8 | GPU-side validity bitmap evaluation: `HAS_NULLS` function constant, bitmap read as packed u32, bit extraction per element | P0 | AC-4.1 through AC-4.8 |
| FR-9 | `polars-gpu-filter` Python package with `pyo3-polars` expression plugin | P1 | AC-5.1 through AC-5.9 |
| FR-10 | Plugin expression parsing: extract column refs + predicates from Polars expression, call forge-filter Arrow API | P1 | AC-5.2, AC-5.7 |
| FR-11 | CPU fallback in plugin when Metal GPU unavailable | P1 | AC-5.6: log warning, return Polars-native result |
| FR-12 | Lazy PSO compilation: compile on first use, cache via MTLBinaryArchive; ~63us first-use overhead per variant | P1 | PSO variant count bounded; no startup latency spike |
| FR-13 | `BooleanMask` type: compact u8 array with LSB-first bit ordering, `to_arrow_boolean()`, `to_polars_boolean()` conversion methods | P1 | AC-2.4: matches Arrow layout |
| FR-14 | Version bump to forge-filter v0.2.0 (semver minor -- additive only, no breaking changes) | P0 | AC-1.5: all v0.1 public types and methods retained |
| FR-15 | Auto-threshold in Polars plugin: if N < 500K rows, route to CPU Polars automatically | P2 | GPU dispatch overhead (~100us) not wasted on small data |
| FR-16 | `FilterError::TooSmall` variant for raw Rust API when N below GPU-useful threshold | P2 | Explicit error for callers who should use CPU |
| FR-17 | maturin-based build: `polars-gpu-filter` distributed as macOS arm64 wheel on PyPI | P1 | AC-5.1: pip install works without Rust toolchain |

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Ordered filter latency (16M u32, 50% sel) | Wall time | <=580us on M4 Pro |
| NFR-2 | Unordered filter latency (16M u32, 50% sel) | Wall time | <=574us (no regression from v0.1) |
| NFR-3 | Multi-column 2-col AND latency (16M rows) | Wall time | <=900us on M4 Pro |
| NFR-4 | NULL bitmap overhead | % increase vs non-null | <=5% |
| NFR-5 | Arrow copy overhead (64MB) | Wall time | <=0.5ms |
| NFR-6 | Polars plugin end-to-end (16M rows) | Wall time | <=850us |
| NFR-7 | PSO first-use compile time | Wall time | <=63us per variant |
| NFR-8 | Backward compatibility | Semver | v0.2.0 minor bump, zero breaking changes |
| NFR-9 | Test coverage | Test count | >=120 correctness tests + 20 benchmark cases |
| NFR-10 | CI performance gate | Regression threshold | Block merge if ordered speedup drops below 9x |
| NFR-11 | GPU correctness non-determinism | Repeat count | All correctness tests run N=10 in CI |
| NFR-12 | Memory safety | Rust guarantees | No unsafe data races; Arrow lifetime enforcement via Rust borrow checker |
| NFR-13 | Large-N support | Max elements | 256M elements without error (hierarchical scan validated) |

## Glossary

- **Bitmap caching**: Storing predicate evaluation results as 1 bit per element in a GPU buffer, reusable across dispatch passes
- **BooleanMask**: Compact u8 array where each bit represents a row's match status (1=match, 0=no-match), LSB-first ordering
- **Decoupled Fallback**: Single-pass chained scan algorithm (Smith, Levien, Owens, SPAA 2025) enabling persistent kernels on Apple Silicon without forward progress guarantees
- **FilterKey**: Sealed trait for GPU-filterable numeric types (u32, i32, f32, u64, i64, f64)
- **Function constant**: Metal shader compile-time specialization value; enables dead-code elimination and type specialization at PSO creation
- **LogicOp**: AND or OR combination mode for multi-column predicate evaluation
- **PSO**: Pipeline State Object -- Metal's compiled GPU kernel configuration
- **Selectivity**: Fraction of input rows matching the predicate (e.g., 50% = half match)
- **SLC**: System-Level Cache -- Apple Silicon shared L2/L3 cache (24MB on M4 Pro); bitmap fits entirely in SLC
- **simd_ballot**: Metal SIMD intrinsic returning a 32-bit mask where bit N is set if thread N's predicate is true
- **Validity bitmap**: Arrow/Polars null indicator; 1 bit per element, 0=NULL, 1=valid, LSB-first packing

## Out of Scope

- String/bytes column filtering (requires GPU string ops -- separate project)
- Polars LazyFrame optimizer integration (predicate pushdown into query plan)
- Multi-GPU / M4 Ultra split dispatch
- Cross-platform backends (Vulkan, WGSL, CUDA)
- Custom aggregation (SUM/AVG/COUNT after filter -- caller's responsibility)
- Streaming/chunked filter for data exceeding unified memory
- INT8/FP16 types (v2 targets the 6 types already in v0.1)
- NULL propagation mode (only NULL exclusion per user decision)
- Decoupled Fallback single-pass ordered pipeline (deferred; bitmap caching is primary approach)
- `filter_nullable_propagate()` variant that keeps NULLs in output

## Dependencies

| Dependency | Type | Risk | Notes |
|------------|------|------|-------|
| Metal 3.2 (device-scope fence) | Technical | Medium | Required for bitmap caching kernel barriers; M1 falls back to v0.1 3-dispatch |
| `arrow-rs` (arrow-array, arrow-buffer, arrow-schema) | Crate | Low | Optional via `arrow` feature flag; pin minor version |
| `pyo3-polars` + `polars_expr` macro | Crate | Medium | Plugin ABI tied to Polars version; pin and test per Polars release |
| `maturin` | Build tool | Low | Compiles polars-gpu-filter Python wheel |
| Self-hosted M4 runner | Infrastructure | Medium | CI performance regression gate requires Apple Silicon hardware |
| forge-filter v0.1.0 test suite | Testing | None | All 47 existing tests must pass unchanged |

## Risks

| ID | Risk | Probability | Impact | Mitigation |
|----|------|-------------|--------|------------|
| R-1 | Bitmap caching insufficient for 10x (plateaus at 8-9x) | Medium | High | 10,000 iteration correctness + benchmark suite; Decoupled Fallback as future upgrade path |
| R-2 | PSO variant explosion (8 cols x 8 ops x 7 types) | Low | Medium | Lazy compilation; ~50 real-world PSOs in practice; MTLBinaryArchive persistent cache |
| R-3 | pyo3-polars ABI breaks between Polars versions | Medium | Medium | Pin version, release per-Polars-version wheels, CI test against latest Polars |
| R-4 | NULL bitmap bit ordering mismatch (LSB vs MSB) | Low | High | Exhaustive boundary tests at positions 0,7,8,15,31,32; MSL code review before merge |
| R-5 | Arrow buffer alignment failures for small arrays | Low | Low | Always-copy approach eliminates alignment concerns entirely |
| R-6 | M1 lacks Metal 3.2 features needed for bitmap caching | Low | Medium | Runtime Metal family detection; M1 falls back to v0.1 3-dispatch pipeline |
| R-7 | GPU thermal throttling causes benchmark flakiness | Medium | Low | CI uses active cooling; CV <5% abort threshold; 5 warmup iterations |

## Success Criteria

- Ordered filter >=10x over Polars CPU at 16M u32, 50% selectivity (<=580us on M4 Pro)
- No regression in unordered mode (remains >=10x)
- Multi-column 2-col AND >=1.9x faster than cascade approach
- All 47 existing v0.1 tests pass unchanged
- >=120 new correctness tests pass
- `pip install polars-gpu-filter` works on macOS arm64 without build tools
- Plugin produces identical results to Polars CPU `filter()` for all numeric types + nullable columns

## Unresolved Questions

1. **Metal 3.2 availability on M1**: Need to verify whether `simd_ballot` and bitmap caching kernels compile/run on M1 (Metal 3.1). If not, M1 fallback scope increases.
2. **Optimal bitmap packing granularity**: Research suggests `simd_ballot` packs 32 bits per simdgroup call. Need to confirm whether 4 ballot calls per thread (for 4 elements/thread) or 1 call per element is faster in practice.
3. **Polars expression parsing depth**: How complex can the expression tree be? `(col("a") > 10) & (col("b") < 50)` is straightforward, but nested OR-of-AND expressions need design work.
4. **arrow-rs version pinning**: Which arrow-rs major version to target? arrow 53.x is current; Polars uses arrow2 internally. May need to support both.
5. **Performance gate threshold stability**: CI benchmarks may vary 5-10% across runs. Need to determine appropriate slack in the >=10x gate (e.g., gate at 9.5x to avoid flaky failures).

## Next Steps

1. Approve requirements, then proceed to design phase
2. Design bitmap caching kernel architecture (Pass 1 bitmap write + Pass 3 bitmap read)
3. Design multi-column fused kernel function constant scheme
4. Design Arrow integration layer and feature flag structure
5. Design polars-gpu-filter crate layout and pyo3-polars wiring
