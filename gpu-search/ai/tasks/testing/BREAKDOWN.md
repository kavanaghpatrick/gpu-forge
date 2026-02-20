---
id: testing.BREAKDOWN
module: testing
priority: 8
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops, data-model, stale-results, path-utils, verify-mode, grouped-results, highlighting, status-bar]
tags: [testing, qa, property-tests, benchmarks]
testRequirements:
  unit:
    required: true
    pattern: "tests/**/*.rs, benches/**/*.rs"
---
# Testing Breakdown

## Context

After all implementation modules are complete, this module ensures comprehensive test coverage across the entire overhaul. While each module writes its own unit tests during implementation, this module adds the remaining cross-cutting tests: integration test suites (`test_stale_results.rs`, `test_ui_pipeline.rs`), property-based tests for the prefix-sum/binary-search virtual scroll algorithm, performance benchmarks for grouped scroll and incremental grouping, visual snapshot tests via egui_kittest, and a full regression run validating all 256+ existing tests still pass.

From QA.md Sections 2-10, covering all test IDs not yet implemented by individual modules.

## Acceptance Criteria

1. Integration test file `tests/test_stale_results.rs` exists with tests I-STALE-1 through I-STALE-4, all passing
2. Integration test file `tests/test_ui_pipeline.rs` exists with tests I-PIPE-1 through I-PIPE-4, all passing
3. Three property tests implemented (proptest): prefix-sum monotonicity, binary search correctness, round-trip height->row->height (QA Q4)
4. Performance benchmark `benches/grouped_scroll.rs` implemented with Criterion benchmarks: `bench_rebuild_flat_row_model_100`, `bench_rebuild_flat_row_model_10k`, `bench_rebuild_flat_row_model_100k`, `bench_first_visible_row_binary_search`, `bench_recompute_groups_incremental_50`, `bench_recompute_groups_full_sort`
5. All benchmark results within targets: rebuild 10K rows <1ms, binary search <0.01ms, incremental grouping 50 matches <0.5ms
6. egui_kittest visual snapshot tests for 3 key layouts: empty results, grouped results with 3 groups, expanded selected row
7. All remaining edge case tests from QA.md Section 6 implemented: EC-4 (very long query), EC-11 (every result different file), EC-18 (path with spaces), EC-21 (/ as root), EC-24 (click group header), EC-28 (Cmd+C no selection), EC-34 (match_range.start > line length), EC-35 (UTF-8 at match_range boundary)
8. Full regression: `cargo test -p gpu-search` passes (all 256+ existing + 45+ new unit + 16+ new integration tests)
9. `cargo clippy -p gpu-search -- -D warnings` reports zero warnings
10. Old results_list tests cleaned up (dual suite migration complete per QA Q3): old flat-index tests removed, replaced by RowKind-based tests

## Technical Notes

- Reference: [spec/OVERVIEW.md] testing is priority 8, depends on all implementation modules
- UX: No direct UX changes; validates UX decisions through automated tests
- Test: From QA.md Sections 2-6 (comprehensive test plan), Section 10 (implementation priority phases 1-4)
- Property tests use `proptest` crate with `prop::collection::vec(24.0f32..=52.0, 1..1000)` for height arrays
- Stale results tests use synchronous harness (QA Q2) -- deterministic, zero flakiness
- egui_kittest snapshot tests (QA Q1) need wgpu feature; manage snapshot file size
- Benchmark targets from QA.md Section 5: frame time <5ms for 25 visible rows, memory <8MB for 10K results
- Performance benchmark noise: Criterion's statistical t-test with confidence intervals (QA Section 9.1)
- Test naming: `test_{module}_{function}_{scenario}` for unit, `test_{pipeline}_{scenario}` for integration (QA Section 8.4)
