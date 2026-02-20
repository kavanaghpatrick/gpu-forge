---
id: executor-cache.BREAKDOWN
module: executor-cache
priority: 0
status: failing
version: 1
origin: spec-workflow
dependsOn: []
tags: [performance, gpu-query, critical-bug]
testRequirements:
  unit:
    required: true
    pattern: "tests/**/*.rs"
---
# Fix Double-Scan Bug + Add Per-Query ScanCache

## Context

`GpuCompoundFilter` in `executor.rs:379-382` calls `resolve_input()` independently on both the left and right branches of compound `WHERE` clauses (e.g., `WHERE amount > 100 AND region = 'US'`). Each branch walks down to `GpuScan`, triggering a full `execute_scan()` that includes schema inference (2-5ms), mmap open (5ms), GPU CSV parse (25ms), and CPU dictionary build (40-80ms). For a 1M-row CSV, this doubles the entire scan cost -- accounting for ~120-150ms of the 367ms total latency.

This is the single highest-impact fix. A per-query `ScanCache` (`HashMap<String, ScanResult>`) eliminates the redundant scan transparently: the first `resolve_input` call populates the cache; the second call hits the cache. This also fixes triple/quadruple scans for deeply nested compounds like `(A > 100 AND B < 50) OR C = 0`.

Additionally, schema inference (`infer_schema_from_csv()` at executor.rs:3020-3028) and dictionary building (`build_csv_dictionaries()` at executor.rs:3109-3182) are called during every scan. Since the schema and dictionaries for a given file never change between queries (unless the file is modified), the ScanCache implicitly caches both.

## Acceptance Criteria

1. Compound filter queries (`WHERE a AND b`, `WHERE a OR b`) execute exactly ONE `execute_scan()` call per table, not two -- verified by scan-count instrumentation or by asserting cache hit on the second `resolve_input` call
2. All 12 existing compound filter integration tests pass with identical results (tests at `tests/gpu_filter.rs:578-747`)
3. A 1M-row compound filter + GROUP BY query completes in <40ms on a warm (second) execution, down from ~240ms
4. Nested compound filters (`(A > 100 AND B < 50) OR C = 0`) produce correct results with a single scan
5. Schema inference runs at most once per table per query execution (cached in ScanResult)
6. Dictionary building runs at most once per table per query execution (cached in ColumnarBatch within ScanResult)
7. The ScanCache is cleared between queries (per-query scope) -- cross-query caching is handled by the warm-cache module
8. No additional GPU buffer allocations compared to the single-filter path (zero allocation regression)
9. Cache key uses `table_name.to_ascii_lowercase()` for case-insensitive matching consistent with SQL semantics
10. Error handling: if a table is not found in the catalog, the error message is preserved unchanged

## Technical Notes

- **Reference**: OVERVIEW.md Module Roadmap priority 0; TECH.md Section 6 (Fix B1+B3+B4 merged); PM.md Section 3.2 BUG #1
- **Files to modify**:
  - `gpu-query/src/gpu/executor.rs` -- Add `scan_cache: HashMap<String, ScanResult>` field to `QueryExecutor`; rename current `execute_scan` to `execute_scan_uncached`; add `execute_scan_cached` wrapper; modify `resolve_input` to return table-name keys instead of owned `ScanResult`
  - `gpu-query/src/gpu/executor.rs` -- Update `GpuCompoundFilter` arm in `resolve_input` (lines 379-395)
  - `gpu-query/src/gpu/executor.rs` -- Update `GpuAggregate`, `GpuFilter`, and other `resolve_input` callers to use `scan_cache.get(&key)` for data access
- **Borrow checker strategy**: Return `String` table-name key from `resolve_input()` instead of owned `ScanResult`. Use `ensure_scan_cached()` for the mutable phase, then `scan_cache.get(&key)` for the immutable read phase. This cleanly separates mutation from access.
- **Test**: `cargo test --all-targets` (all 742+ tests); `cargo bench --bench filter_throughput -- "compound"` for performance; new `test_compound_filter_single_scan` unit test asserting scan count == 1
- **Risk**: Medium -- changes the core `resolve_input` return type and data flow. Must verify all downstream consumers (`execute_filter`, `execute_aggregate`, `execute_aggregate_grouped`) accept `&ScanResult` references from the cache instead of owned values.
