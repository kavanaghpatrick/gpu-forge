# forge-sort Test Suite Specification -- Overview

## Executive Summary

forge-sort is a public Rust crate providing GPU-accelerated radix sorting on Apple Silicon (31x faster than CPU `sort_unstable` at 16M elements). The current test suite has 21 tests (3 unit + 18 integration) covering the happy path but leaving critical algorithm internals -- tile boundaries, adversarial bit patterns, atomics determinism, and buffer reuse -- unexercised. This spec defines a target of ~63 integration tests + 6 property tests + 5 performance tests to achieve production-grade correctness confidence.

## Scope

- **Current**: 21 tests (3 unit in `lib.rs` + 18 integration in `tests/correctness.rs`)
- **Target**: 63 integration + 6 proptest properties (200 cases each) + 5 perf tests = **74 test functions**, ~66 active in default `cargo test`
- **New test categories**:
  - Tile boundary precision -- catches off-by-one in partial tile handling (the #1 GPU kernel bug class)
  - Adversarial bit patterns -- catches degenerate MSD/inner bucket distributions that random data never triggers
  - Determinism verification -- catches atomic ordering race conditions via multi-run identity checks
  - Buffer management -- catches stale data leakage from oversized buffer reuse
  - Property-based (proptest) -- catches edge cases via shrinkable random generation with 200 cases per property
  - Performance regression -- catches throughput regressions behind a feature gate

## Key Decisions

- **Single test file with modules** (UX): Keep all integration tests in `tests/correctness.rs` organized by `mod` blocks. Each `tests/*.rs` file compiles as a separate crate with separate shader compilation; modules avoid that cost while enabling `cargo test tile_boundary` filtering. (See UX.md, Decision 1)
- **Three test files by concern** (TECH): `correctness.rs` (deterministic), `property.rs` (proptest), `performance.rs` (feature-gated). Shared helpers in `tests/common/mod.rs`. (See TECH.md, Section 1)
- **Seeded randomness everywhere** (UX/QA): All random data uses `ChaCha8Rng` with explicit seeds. Failure messages include the seed for exact reproduction. `thread_rng()` is prohibited. (See UX.md, Decision 8)
- **Feature-gated perf tests** (UX/TECH): `#[cfg(feature = "perf-test")]` eliminates perf tests from default builds. `#[ignore]` reserved for 128M scale tests (correctness, just slow). (See UX.md, Decision 5)
- **Four verification methods** (QA): Method A (std sort), B (is-sorted), C (permutation/histogram), D (hardcoded). Tests use appropriate combinations by category. (See QA.md, Section 2)
- **Proptest with thread-local GpuSorter** (TECH): 200 cases per property, thread-local sorter amortizes PSO compilation. Proptest shrinking finds minimal failing inputs. (See TECH.md, Section 4)
- **Dual perf thresholds** (PM/TECH): Hard floor (absolute ms) catches catastrophic regression. Relative floor (Mk/s) catches gradual degradation. 30% headroom absorbs thermal throttle. (See TECH.md, Section 5)
- **Correctness tests are never flaky** (QA): Deterministic inputs produce deterministic outputs. A "flaky" correctness test is treated as a P0 race-condition bug. (See QA.md, Section 10)

## Test Categories (Priority Order)

| Category | Count | Priority | Risk Level | What It Catches |
|----------|-------|----------|------------|-----------------|
| Tile boundary precision | 10 + 1 sweep | P0 | Critical | Off-by-one in `idx < n`, partial tile data loss, `num_tiles` miscalculation |
| Adversarial bit patterns | 12-14 | P0 | Critical | Empty bucket paths, single-bucket overflow, atomic contention extremes, per-byte-position degenerate cases |
| Core correctness (random sizes) | 18 | P0 | Critical | General algorithm correctness at 1K through 64M |
| Edge cases | 8-9 | P0 | High | Empty/single/two-element handling, u32::MAX/MIN, API contract |
| Determinism | 3 | P0 | High | Atomic ordering races, non-deterministic scatter positioning |
| Buffer management | 6 | P1 | High | Stale data leakage from oversized buffers, reallocation chains |
| Property-based (proptest) | 6 functions | P1 | High | Sorted-output, permutation, idempotency across random inputs and strategies |
| Scale/stress | 5 (1 ignored) | P2 | Medium | Large tile counts, memory pressure, rapid buffer cycling |
| Performance regression | 5 (feature-gated) | P2 | Medium | Throughput floors, scaling linearity, cold-start penalty |
| Concurrency | 2 | P2 | Low | Multi-sorter interference, rapid sequential resource exhaustion |
| Large scale (ignored) | 2 | P2 | Low | 128M correctness under memory pressure |

## File Structure

From TECH.md, Section 1:

```
forge-sort/
  Cargo.toml                          # Add proptest, rand_chacha, perf-test feature
  src/lib.rs                          # Existing (3 unit tests stay)
  tests/
    correctness.rs                    # REPLACE -- ~50 deterministic tests in mod blocks
    property.rs                       # NEW -- 6 proptest properties, 200 cases each
    performance.rs                    # NEW -- 5 perf tests, gated by feature = "perf-test"
    common/
      mod.rs                          # Shared: verification harness, generators, perf helpers
```

New dev-dependencies: `rand = "0.8"`, `rand_chacha = "0.3"`, `proptest = "1.4"`, `rayon = "1.10"`.

## Verification Strategy

From QA.md, Section 2:

| Method | What It Does | Cost | Used By |
|--------|-------------|------|---------|
| **A: std sort comparison** | CPU `sort()` + exact equality | O(n log n) | All correctness tests at n <= 16M (definitive oracle) |
| **B: is-sorted check** | `windows(2).all(\|w\| w[0] <= w[1])` | O(n), zero alloc | Fast pre-check for perf tests and 32M+ scale tests |
| **C: permutation check** | XOR + SUM checksums, then full HashMap histogram | O(n) | Combined with B for 32M+ tests; independent cross-check for correctness |
| **D: hardcoded expected** | Exact known output for small inputs | O(n) compare | Edge cases (empty, single, two, three elements) |

Assignment: edge cases use D, correctness up to 16M uses A+C, scale 32M+ uses B+C (A only on failure), perf uses B only, determinism checks equality across runs.

## Implementation Phases

From PM.md, Priority Matrix:

| Phase | Categories | New Tests | Effort | Cumulative |
|-------|-----------|-----------|--------|------------|
| **Phase 1 (P0)** | Tile boundary + adversarial bit patterns | ~22 | 2-3 hours | ~22 new |
| **Phase 2 (P1)** | Determinism + buffer management + proptest | ~15 | 2-3 hours | ~37 new |
| **Phase 3 (P2+P3)** | Scale, edge cases, perf, concurrency | ~16 | 2-3 hours | ~53 new |

Total estimated effort: 6-9 hours. Suite goes from 21 to ~74 test functions.

## Module Roadmap

Ordered by implementation priority (risk-reduction per effort):

1. **`tests/common/mod.rs`** -- Shared infrastructure: `gpu_sort_and_verify()`, `gen_sort_verify()`, `verify_sorted()`, `verify_permutation()`, seeded RNG factory, adversarial pattern generators, tile boundary size calculator, perf measurement helpers
2. **`correctness.rs::tile_boundary`** -- 10 tests targeting TILE_SIZE=4096 alignment: exact, +1, -1, primes, SIMD-group boundaries, fused_grid width boundary
3. **`correctness.rs::bit_patterns`** -- 12-14 tests: single MSD bucket, two buckets, uniform 256, zero inner bytes, all-max, alternating 0/MAX, per-byte-position sweep, near-duplicates, few-unique, nearly-sorted
4. **`correctness.rs::determinism`** -- 3 tests: 10-run identity at 1M (random + adversarial), 100-seed battery
5. **`correctness.rs::buffer_reuse`** -- 6 tests: large-then-small, small-then-large, same-size different data, 50 reuses, growing sequence, shrinking sequence
6. **`tests/property.rs`** -- 6 proptest properties: sorted output, permutation, idempotent, same-MSD correctness, few-unique correctness, tile-boundary correctness
7. **`correctness.rs::edge_cases`** -- 8-9 tests: empty, single, two sorted/reversed, three, all-max, all-min, max+min mixed, pre-sorted, reverse-sorted
8. **`correctness.rs::scale`** -- 5 tests: 32M, 64M, 128M (ignored), rapid 1000x small, alternating sizes
9. **`tests/performance.rs`** -- 5 tests (feature-gated): 16M hard threshold, 16M throughput floor, 1M hard threshold, scaling linearity, cold start penalty
10. **`correctness.rs::concurrent`** -- 2 tests: two sorters alternating, 100 rapid sequential sorts

---

For full details, see:
- [PM.md](PM.md) -- Business context, risk assessment, 8-category test requirements, success metrics
- [TECH.md](TECH.md) -- File structure, code samples, proptest strategies, tile math, error injection
- [UX.md](UX.md) -- 10 DX decisions: naming, helpers, assertions, feature gating, CI workflow
- [QA.md](QA.md) -- 4 verification methods, per-kernel coverage matrix, acceptance criteria, flakiness policy, mutation testing
