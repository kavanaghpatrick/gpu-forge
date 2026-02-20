# forge-sort v2: Multi-Type Support Overview

## Executive Summary

forge-sort v2 extends the GPU radix sort library from u32-only to support i32, f32, u64, i64, f64, argsort, and key-value pair sorting on Apple Silicon Metal 3.2. The design uses GPU-side bit transformations (FloatFlip/XOR) and Metal function constants to achieve multi-type support from a single shader source with zero performance impact on the existing u32 path. There is no Metal-native GPU sort library that supports signed integers, floats, 64-bit integers, or key-value pairs -- forge-sort v2 will be the first.

---

## Scope

**In v2**: sort_i32, sort_f32, sort_u64, sort_i64, sort_f64, argsort (all 6 types), sort_pairs (u32/i32/f32/u64/i64), SortBuffer\<T\> with sealed SortKey trait (6 types: u32/i32/f32/u64/i64/f64), SortError::LengthMismatch variant, PsoCache function constant extension in forge-primitives.

**Deferred**: descending sort variants, partial-bit sorting (begin_bit/end_bit), sort_pairs_f64, generic value types beyond u32, fused transform optimization (Phase 4), native key-value shader (Strategy A).

---

## Key Decisions

| # | Decision | Rationale | Source |
|---|----------|-----------|--------|
| 1 | Explicit typed methods (`sort_i32`, `sort_f32`) over generics | Matches v1 `sort_u32` pattern, discoverable via autocomplete, no trait-bound leakage | PM, UX |
| 2 | `SortBuffer<T>` with sealed `SortKey` trait (breaking: `SortBuffer` -> `SortBuffer<u32>`) | Compile-time type safety, prevents `SortBuffer<String>`, bump to 0.2.0 | UX |
| 3 | `total_cmp` semantics: `-0.0 < +0.0`, NaN at extremes | Matches Rust stdlib and FloatFlip natural behavior, no special-case code | PM Q1 |
| 4 | 32-bit values only for key-value pairs (`u32`) | Covers 90%+ use cases (index pattern), simpler, faster to ship | PM Q2 |
| 5 | Argsort as first-class API (not internal-only) | Foundation for Strategy B key-value sorting, independently useful | PM Q3 |
| 6 | Strategy B (argsort + gather) for key-value, not native KV shader | Lower risk, reuses proven sort kernels, Strategy A deferred to Phase 4 | PM, TECH |
| 7 | Metal function constants (not separate .metal files) | Single shader source, dead code elimination produces identical perf, Apple-recommended | TECH |
| 8 | All transforms on GPU (not CPU-side) | 0.3ms GPU vs 5ms CPU single-threaded, one code path for memcpy and zero-copy | TECH Q2 |
| 9 | InnerParams struct (not separate buffer bindings) for inner kernel | Cleaner single binding, breaking change acceptable since we control all callers | TECH Q1 |
| 10 | PsoCache extension in forge-primitives (not local) | Function constants are general-purpose Metal, available to all forge-compute crates | TECH Q3 |
| 11 | Two separate SortBuffers for sort_pairs (not SortPairsBuffer) | Reuses existing types, no new public type needed | UX Q2 |
| 12 | No sort_pairs_f64 (but yes sort_pairs_u64 and sort_pairs_i64) | f64 is niche + 8-pass + doubled bandwidth too costly. u64/i64 have concrete KV use cases (database IDs, timestamps) | UX Q3 |
| 13 | u64/i64 included in Phase 3 alongside f64 | Once IS_64BIT pipeline exists for f64, u64 (no transform) and i64 (sign bit XOR) are zero marginal cost | PM, TECH |

---

## Implementation Phases

| Phase | Scope | Effort | Risk | New Methods |
|-------|-------|--------|------|-------------|
| **1: i32/f32** | Bit transforms, SortKey trait, SortBuffer\<T\> | 3-4 days | Very Low | sort_i32, sort_f32, sort_i32_buffer, sort_f32_buffer, alloc_sort_buffer::\<T\> |
| **2: Argsort/KV** | Index init kernel, gather kernel, has_values function constant | 5-8 days | Medium | argsort_{u32,i32,f32}, sort_pairs_{u32,i32,f32} |
| **3: u64/i64/f64** | 64-bit shader variant (IS_64BIT), 8-pass pipeline, sort_transform_64 | 3-5 days | Medium | sort_u64, sort_i64, sort_f64, sort_{u64,i64,f64}_buffer, argsort_{u64,i64,f64}, sort_pairs_{u64,i64} |

---

## API Surface

| Method | Signature | Phase |
|--------|-----------|-------|
| `sort_i32` | `(&mut self, &mut [i32]) -> Result<()>` | 1 |
| `sort_f32` | `(&mut self, &mut [f32]) -> Result<()>` | 1 |
| `sort_i32_buffer` | `(&mut self, &SortBuffer<i32>) -> Result<()>` | 1 |
| `sort_f32_buffer` | `(&mut self, &SortBuffer<f32>) -> Result<()>` | 1 |
| `alloc_sort_buffer::<T>` | `(&self, usize) -> SortBuffer<T>` | 1 |
| `argsort_u32` | `(&mut self, &[u32]) -> Result<Vec<u32>>` | 2 |
| `argsort_i32` | `(&mut self, &[i32]) -> Result<Vec<u32>>` | 2 |
| `argsort_f32` | `(&mut self, &[f32]) -> Result<Vec<u32>>` | 2 |
| `sort_pairs_u32` | `(&mut self, &mut [u32], &mut [u32]) -> Result<()>` | 2 |
| `sort_pairs_i32` | `(&mut self, &mut [i32], &mut [u32]) -> Result<()>` | 2 |
| `sort_pairs_f32` | `(&mut self, &mut [f32], &mut [u32]) -> Result<()>` | 2 |
| `sort_u64` | `(&mut self, &mut [u64]) -> Result<()>` | 3 |
| `sort_i64` | `(&mut self, &mut [i64]) -> Result<()>` | 3 |
| `sort_f64` | `(&mut self, &mut [f64]) -> Result<()>` | 3 |
| `sort_u64_buffer` | `(&mut self, &SortBuffer<u64>) -> Result<()>` | 3 |
| `sort_i64_buffer` | `(&mut self, &SortBuffer<i64>) -> Result<()>` | 3 |
| `sort_f64_buffer` | `(&mut self, &SortBuffer<f64>) -> Result<()>` | 3 |
| `argsort_u64` | `(&mut self, &[u64]) -> Result<Vec<u32>>` | 3 |
| `argsort_i64` | `(&mut self, &[i64]) -> Result<Vec<u32>>` | 3 |
| `argsort_f64` | `(&mut self, &[f64]) -> Result<Vec<u32>>` | 3 |
| `sort_pairs_u64` | `(&mut self, &mut [u64], &mut [u32]) -> Result<()>` | 3 |
| `sort_pairs_i64` | `(&mut self, &mut [i64], &mut [u32]) -> Result<()>` | 3 |

**Breaking change**: `SortBuffer` becomes `SortBuffer<u32>`. Version bump to 0.2.0.

---

## Technical Architecture

**Bit Transformations**: i32 uses XOR `0x80000000` (self-inverse). f32/f64 use FloatFlip/IFloatFlip (sign-dependent XOR) producing `total_cmp` ordering. u64 requires no transform (simplest 64-bit type). i64 uses XOR `0x8000000000000000` (self-inverse, same concept as i32 but 64-bit). Implemented as GPU kernel `sort_transform_32`/`sort_transform_64` with `TRANSFORM_MODE` function constant (mode 0=i32/no-op, 1=i64 sign XOR/float forward, 2=float inverse).

**Function Constants**: 2 constants control all kernel variants: `HAS_VALUES [[function_constant(0)]]` and `IS_64BIT [[function_constant(1)]]`. Metal compiler eliminates dead branches, producing identical machine code to separate files. ~11 PSOs at init; 64-bit PSOs (shared by u64/i64/f64) compiled lazily on first 64-bit sort call.

**Argsort Pipeline** (single command buffer): [transform] -> init_indices -> histogram -> prep -> scatter(has_values=true) -> inner_fused(has_values=true) -> [inverse transform]. sort_pairs adds a final gather dispatch.

**64-bit Pipeline (u64/i64/f64)**: 8 radix passes via 1 MSD + 3 inner fused dispatches (3+3+1 passes). Half tile size (2048 vs 4096) for register budget. 6 dispatches total. u64 uses no transform (simplest path), i64 adds sign bit XOR (self-inverse), f64 adds FloatFlip/IFloatFlip. All three share the same IS_64BIT=true sort PSOs.

**New Kernels**: sort_transform_32 (~20 lines), sort_transform_64 (~15 lines), sort_init_indices (~6 lines), sort_gather_values (~8 lines). All added to existing `sort.metal`.

---

## Test Strategy

**Total**: ~155 new tests + ~63 existing = ~220 tests across 9 test files. Verification via oracle comparison (`sort_by(total_cmp)` for floats, natural `sort()` for integers), permutation validity, pair integrity multiset, and bit-exact float comparison (`to_bits()` equality, not `==`).

**Key coverage**: 19 IEEE 754 special values (NaN variants, Inf, denormals, signed zeros), i32 boundary values, argsort permutation validity, sort_pairs stability, f64 per-byte-position isolation, cross-type interleaving regression.

**Performance tests**: feature-gated (`--features perf-test`), seeded RNG for determinism, 30% headroom in thresholds.

---

## Performance Targets (16M elements, M4 Pro)

| Method | Path | Expected (Mk/s) | Threshold (Mk/s) |
|--------|------|:---:|:---:|
| sort_u32 | memcpy | ~2859 | >= 2000 |
| sort_u32 | zero-copy | ~5207 | >= 3600 |
| sort_i32 | memcpy | ~2800 | >= 1900 |
| sort_f32 | memcpy | ~2500 | >= 1750 |
| sort_f32 | zero-copy | ~4500 | >= 3100 |
| argsort_u32 | memcpy | ~2000 | >= 1400 |
| sort_pairs_f32 | memcpy | ~1500 | >= 1000 |
| sort_u64 | memcpy | ~850 | >= 600 |
| sort_u64 | zero-copy | ~1250 | >= 850 |
| sort_i64 | memcpy | ~800 | >= 550 |
| sort_i64 | zero-copy | ~1200 | >= 800 |
| sort_f64 | memcpy | ~800 | >= 550 |
| sort_f64 | zero-copy | ~1200 | >= 800 |

---

## File Changes

| File | Change |
|------|--------|
| `shaders/sort.metal` | Function constants, has_values/is_64bit branches, 4 new kernels, InnerParams struct (~120 lines added) |
| `src/lib.rs` | SortBuffer\<T\>, SortKey trait (6 types), 24+ methods, SortPipelineConfig, dispatch_sort_pipeline (~500 lines added) |
| `Cargo.toml` | Version 0.1.0 -> 0.2.0, add rand_chacha dev-dep |
| `tests/correctness.rs` | Update `SortBuffer` -> `SortBuffer<u32>` type annotations |
| `tests/correctness_{i32,f32,u64,i64,f64,argsort,sort_pairs}.rs` | New test files (~155 tests) |
| `tests/common/mod.rs` | Shared oracle functions, verification helpers, test data generators |
| `tests/{regression,performance}.rs` | v1 compat tests, feature-gated perf tests |
| `forge-primitives/src/pso_cache.rs` | FnConstant enum, get_or_create_specialized method |

---

## Module Roadmap

1. **SortKey trait + SortBuffer\<T\>** -- sealed trait, generic buffer, alloc_sort_buffer::\<T\>
2. **PsoCache function constants** -- extend forge-primitives with FnConstant + specialized PSO creation
3. **sort_transform_32 kernel** -- GPU bit transformation for i32/f32
4. **sort_i32 + sort_f32** -- memcpy and zero-copy paths using transform -> sort -> inverse transform
5. **sort_init_indices kernel** -- write [0..n] for argsort
6. **has_values branches in scatter/inner kernels** -- carry indices through sort pipeline
7. **argsort_{u32,i32,f32}** -- full argsort pipeline
8. **sort_gather_values kernel** -- rearrange values by sorted indices
9. **sort_pairs_{u32,i32,f32}** -- argsort + gather composition
10. **is_64bit branches + sort_transform_64** -- 64-bit element handling, 8-pass pipeline
11. **sort_u64 + sort_u64_buffer** -- baseline 64-bit type (no transform, validates IS_64BIT pipeline)
12. **sort_i64 + sort_i64_buffer** -- sign bit XOR (0x8000000000000000) via sort_transform_64 mode 1
13. **sort_f64 + argsort_f64** -- f64 sort with 3 inner fused dispatches
14. **argsort_{u64,i64} + sort_pairs_{u64,i64}** -- 64-bit argsort and key-value pairs

---

*Cross-references: [PM.md](PM.md) (business context, priority ordering, competitive positioning) | [UX.md](UX.md) (API design, naming conventions, migration path, documentation patterns) | [TECH.md](TECH.md) (shader architecture, function constants, dispatch logic, f64 pipeline) | [QA.md](QA.md) (test matrix, IEEE 754 edge cases, verification methods, acceptance criteria)*
