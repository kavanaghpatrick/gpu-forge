---
id: benchmarks.BREAKDOWN
module: benchmarks
priority: 4
status: failing
version: 1
origin: spec-workflow
dependsOn:
  - executor-cache.BREAKDOWN
  - persistent-executor.BREAKDOWN
  - catalog-cache.BREAKDOWN
  - warm-cache.BREAKDOWN
tags: [performance, gpu-query, testing, ci]
testRequirements:
  unit:
    required: true
    pattern: "tests/**/*.rs"
---
# Performance Benchmarks + Regression Tests

## Context

The performance optimization work (modules 0-3) requires rigorous before/after measurement to validate the projected 7.3x improvement (367ms -> <50ms). The existing benchmark suite has 49 criterion benchmarks across 5 files but lacks:

1. Warm vs. cold query comparison benchmarks
2. 1M-row compound filter benchmarks at scale
3. Cache hit/miss micro-benchmarks for each caching layer
4. An end-to-end acceptance test asserting the <50ms target
5. A `--benchmark` CLI flag for ad-hoc timing without the full criterion harness

This module adds ~14 new criterion benchmarks, a performance regression CI gate, and the acceptance test that gates the PR merge.

## Acceptance Criteria

1. Pre-fix baselines are saved via `cargo bench -- --save-baseline pre_opt` before any performance changes land
2. New criterion benchmark groups added:
   - `compound_filter_latency`: `and_1m_rows`, `or_1m_rows`, `nested_and_or_1m` -- measures executor-cache fix impact
   - `executor_lifecycle`: `cold_start`, `warm_pso_5th_query`, `warm_100th_query` -- measures persistent-executor fix
   - `repeated_query_latency`: `same_query_x10`, `alternating_queries_x10` -- measures all caches combined
   - `schema_inference_cache`: `cold_infer`, `cached_infer` -- measures catalog-cache schema fix
   - `catalog_scan_cache`: `cold_scan`, `cached_scan` -- measures catalog-cache directory fix
   - `dictionary_cache`: `cold_build_varchar`, `cached_lookup_varchar` -- measures dictionary caching
3. All benchmarks use consistent configuration: `sample_size(20)`, `measurement_time(10s)`, `noise_threshold(0.05)`, `significance_level(0.05)`
4. Post-fix comparison via `cargo bench -- --baseline pre_opt` shows statistically significant improvement (not noise)
5. Acceptance test: `test_compound_filter_group_by_performance` -- generates 1M-row CSV, runs warm query, asserts `elapsed < 50ms`
6. `--benchmark` CLI flag runs the reference query (`SELECT region, COUNT(*), SUM(amount) FROM sales WHERE amount > 100 AND region = 'US' GROUP BY region`) with timing output and no TUI overhead
7. CI pipeline includes a `perf-regression` job that compares PR benchmarks against base branch and warns on regressions
8. Memory stability test: 100 queries on reused executor, heap delta < 5MB
9. All 742+ existing tests continue to pass alongside the new benchmarks

## Technical Notes

- **Reference**: OVERVIEW.md Module Roadmap priority 4; QA.md Sections 4 (Benchmark Design) and 7 (CI Pipeline Changes); PM.md Section 6 (Success Criteria)
- **Files to create**:
  - `gpu-query/benches/query_latency.rs` -- New benchmark file for `compound_filter_latency`, `executor_lifecycle`, `repeated_query_latency` groups (or extend existing bench file)
- **Files to modify**:
  - `gpu-query/benches/scan_throughput.rs` -- Add `schema_inference_cache`, `catalog_scan_cache`, `dictionary_cache` benchmark groups
  - `gpu-query/benches/filter_throughput.rs` -- Add `compound_1m/and_1m` and `compound_1m/or_1m` benchmarks
  - `gpu-query/src/cli/mod.rs` -- Add `--benchmark` clap flag; implement timed single-query execution with structured output
  - `gpu-query/.github/workflows/ci.yml` -- Add `perf-regression` CI job (compare against base branch)
  - `gpu-query/Cargo.toml` -- Add criterion dev-dependency if not present; add `[[bench]]` entries for new bench files
- **Reference query for benchmarks**:
  ```sql
  SELECT region, COUNT(*), SUM(amount) FROM sales
  WHERE amount > 100 AND region = 'US'
  GROUP BY region
  ```
  On a generated 1M-row CSV with 5 columns (2 INT64, 1 FLOAT64, 2 VARCHAR).
- **Benchmark data generation**: Use a deterministic seed for reproducible test data; generate CSV with known distributions for predictable result verification.
- **Test**: `cargo bench` (all benchmarks); `cargo test test_compound_filter_group_by_performance` (acceptance test); `cargo test --all-targets` (full regression)
- **Key latency targets**:
  | Scenario | Before | Target |
  |----------|--------|--------|
  | Compound AND 1M warm | ~240ms | <35ms |
  | E2E query 1M warm | 367ms | <50ms |
  | Simple filter 1M warm | ~120ms | <30ms |
  | Cold first query 1M | 367ms | <80ms |
