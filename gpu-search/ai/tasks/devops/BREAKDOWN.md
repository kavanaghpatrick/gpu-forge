---
id: devops.BREAKDOWN
module: devops
priority: 0
status: failing
version: 1
origin: spec-workflow
dependsOn: []
tags: [ci, benchmark, infrastructure]
testRequirements:
  unit:
    required: false
    pattern: "tests/devops/**/*.test.*"
---
# DevOps Breakdown

## Context

The gpu-search UI/UX overhaul introduces new test categories (unit, integration, property, visual snapshot), performance benchmarks (grouped scroll, incremental grouping), and a hard CI gate requiring <10% regression on benchmarks. The existing 5-stage CI pipeline needs extension to accommodate these new test types, and benchmark infrastructure must be established before implementation begins so all modules can validate against performance budgets from day one.

From QA.md Section 8: The CI pipeline should be extended with new stages for stale results tests, UI pipeline tests, and grouped scroll benchmarks. From QA.md Q5: Hard gate -- CI fails if any benchmark regresses >10%. From QA.md Q1: egui_kittest with wgpu snapshots should be added for visual regression testing.

## Acceptance Criteria

1. CI pipeline configuration updated with 6 test levels (fast-checks, UI pipeline, GPU integration, performance, stress, build) as specified in QA.md Section 8.1
2. `cargo bench -p gpu-search` runs successfully with Criterion framework configured for the new `benches/grouped_scroll.rs` benchmark file (file can be a placeholder with one trivial benchmark)
3. Benchmark baseline established: existing `search_throughput` and `search_latency` benchmarks run and produce stored baseline values for regression comparison
4. Hard benchmark gate configured: CI fails if any benchmark regresses >10% from baseline
5. `egui_kittest` dependency added to `Cargo.toml` dev-dependencies with `wgpu` feature enabled; a minimal snapshot test compiles and runs
6. `proptest` dependency added to `Cargo.toml` dev-dependencies; a trivial property test compiles and runs
7. Test naming convention documented: `test_{module}_{function}_{scenario}` for unit, `test_{pipeline}_{scenario}` for integration, `bench_{component}_{scenario}` for benchmarks
8. All existing tests still pass after infrastructure changes: `cargo test -p gpu-search` green

## Technical Notes

- Reference: [spec/OVERVIEW.md] Module Roadmap -- devops is priority 0, unblocks all other modules
- UX: No UX changes in this module
- Test: From QA.md Section 8.2 -- CI pipeline stages and Section 5 -- performance benchmarks
- Key files: `Cargo.toml`, CI configuration, `benches/grouped_scroll.rs` (new), test runner scripts
- The benchmark gate (QA Q5) uses Criterion's statistical t-test with confidence intervals
- egui_kittest setup (QA Q1) requires wgpu feature for snapshot rendering
- proptest setup (QA Q4) needed for prefix-sum/binary-search property tests in later modules
