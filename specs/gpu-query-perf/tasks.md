---
spec: gpu-query-perf
phase: tasks
total_tasks: 22
created: 2026-02-11
generated: auto
---

# Tasks: gpu-query-perf

## Phase 1: Make It Work (POC) -- Fix Double-Scan + Add ScanCache

Focus: Eliminate the #1 bottleneck (double CSV scan in compound filters) by adding a per-executor scan cache. This single change fixes B1, B3, and B4 for within-query deduplication.

- [x] 1.1 Add scan_cache field to QueryExecutor
  - **Do**: Add `scan_cache: HashMap<String, ScanResult>` field to `QueryExecutor` struct at `src/gpu/executor.rs:254`. Initialize as `HashMap::new()` in `QueryExecutor::new()` at line 261. Add `use std::collections::HashMap;` if not already imported (it is, line 10). Ensure `ScanResult` derives or implements any needed traits -- it currently does not need `Clone` since it stays in the cache.
  - **Files**: `gpu-query/src/gpu/executor.rs`
  - **Done when**: `QueryExecutor` struct has `scan_cache` field; `new()` initializes it; project compiles
  - **Verify**: `cd gpu-query && cargo check 2>&1 | tail -5`
  - **Commit**: `feat(executor): add scan_cache field to QueryExecutor`
  - _Requirements: FR-1, FR-2_
  - _Design: Component 3 (ScanCache)_

- [x] 1.2 Add ensure_scan_cached and execute_scan_uncached methods
  - **Do**: Rename current `execute_scan` to `execute_scan_uncached` (private). Create new `ensure_scan_cached(&mut self, table: &str, catalog: &[TableEntry]) -> Result<String, String>` that: (1) lowercases table name as key, (2) checks `self.scan_cache.contains_key(&key)`, (3) if miss, calls `self.execute_scan_uncached(table, catalog)?` and inserts result into `self.scan_cache`, (4) returns the lowercase key. Do NOT add file stat validation yet -- just get basic caching working.
  - **Files**: `gpu-query/src/gpu/executor.rs`
  - **Done when**: `ensure_scan_cached` populates cache on miss, returns key on hit; compiles
  - **Verify**: `cd gpu-query && cargo check 2>&1 | tail -5`
  - **Commit**: `feat(executor): add scan cache with ensure_scan_cached method`
  - _Requirements: FR-1_
  - _Design: Component 3 (ScanCache)_

- [x] 1.3 Refactor resolve_input to return table key instead of owned ScanResult
  - **Do**: Change `resolve_input` return type from `Result<(ScanResult, Option<FilterResult>), String>` to `Result<(String, Option<FilterResult>), String>`. In each match arm: (a) `GpuScan`: call `self.ensure_scan_cached(table, catalog)?`, return `(key, None)`. (b) `GpuFilter`: get key from inner resolve_input or ensure_scan_cached, then `let scan = self.scan_cache.get(&key).unwrap();` and call `self.execute_filter(scan, ...)`. (c) `GpuCompoundFilter`: both `resolve_input(left)` and `resolve_input(right)` now naturally deduplicate via cache. (d) `GpuAggregate` and `GpuSort`: pass through. Update ALL callers of `resolve_input` in `execute()` to use the key pattern: `let scan = self.scan_cache.get(&key).unwrap();` for aggregate/sort operations. This is the core refactor that fixes the double-scan bug.
  - **Files**: `gpu-query/src/gpu/executor.rs`
  - **Done when**: `resolve_input` returns `(String, Option<FilterResult>)`; compound filters hit cache on second branch; all callers updated
  - **Verify**: `cd gpu-query && cargo test --lib 2>&1 | tail -5` (all 494 tests pass)
  - **Commit**: `fix(executor): eliminate double-scan in compound filters via scan cache`
  - _Requirements: FR-1, FR-9_
  - _Design: Component 3 (ScanCache), resolve_input Refactor_

- [x] 1.4 Add file stat validation to scan cache
  - **Do**: In `ensure_scan_cached`, before returning a cache hit, stat the file to check `(size, mtime)`. If either differs from when the entry was cached, remove the stale entry and re-scan. Store `(file_size: u64, file_modified: SystemTime)` alongside each `ScanResult` in the cache -- either wrap in a `CachedScan { result: ScanResult, file_size: u64, file_modified: SystemTime }` struct or add the fields to `ScanResult`. Use `std::fs::metadata(&entry.path)` to get current stats.
  - **Files**: `gpu-query/src/gpu/executor.rs`
  - **Done when**: Cache returns miss when file size or mtime changes; re-scans automatically
  - **Verify**: `cd gpu-query && cargo test --lib 2>&1 | tail -5`
  - **Commit**: `feat(executor): add file stat validation to scan cache`
  - _Requirements: FR-3_
  - _Design: Cache Invalidation Summary_

- [x] 1.5 Clear scan cache between query executions
  - **Do**: At the top of `QueryExecutor::execute()` (the public entry point), call `self.scan_cache.clear()`. This ensures per-query deduplication (fixes double-scan) while preventing stale data across queries. Cross-query caching will be added in Phase 3 when file validation is mature.
  - **Files**: `gpu-query/src/gpu/executor.rs`
  - **Done when**: Scan cache is cleared at start of each execute() call; within-query dedup still works
  - **Verify**: `cd gpu-query && cargo test --lib 2>&1 | tail -5`
  - **Commit**: `feat(executor): clear scan cache between queries for safety`
  - _Requirements: FR-1_
  - _Design: Component 3_

- [x] 1.6 POC Checkpoint -- Verify double-scan eliminated
  - **Do**: Run full test suite. Verify all 494 lib tests pass. Run GPU integration tests if available. Check that compound filter queries produce identical results. Optionally add a temporary `eprintln!` counter in `execute_scan_uncached` to verify it's called once per table per compound filter query (remove before commit).
  - **Files**: None (verification only)
  - **Done when**: All tests pass; compound filter calls execute_scan_uncached exactly once per table
  - **Verify**: `cd gpu-query && cargo test --lib 2>&1 | tail -5 && cargo test --test gpu_filter 2>&1 | tail -10`
  - **Commit**: `feat(executor): complete POC -- scan cache eliminates double-scan`
  - _Requirements: AC-2.1, AC-2.2, AC-2.3_

## Phase 2: Persistent Executor -- Move QueryExecutor to AppState

Focus: Eliminate per-query Metal device + PSO cache recreation by persisting the executor in TUI AppState.

- [x] 2.1 Create CatalogCache module
  - **Do**: Create new file `gpu-query/src/io/catalog_cache.rs` with `CatalogCache` struct as designed in TECH.md Section 4. Fields: `dir: PathBuf`, `entries: Vec<TableEntry>`, `fingerprints: HashMap<PathBuf, FileFingerprint>`, `dir_modified: Option<SystemTime>`. Methods: `new(dir)`, `get_or_refresh() -> io::Result<&[TableEntry]>`, `is_valid() -> io::Result<bool>`, `refresh() -> io::Result<()>`, `invalidate()`. Add `pub mod catalog_cache;` to `gpu-query/src/io/mod.rs`.
  - **Files**: `gpu-query/src/io/catalog_cache.rs` (create), `gpu-query/src/io/mod.rs` (modify)
  - **Done when**: CatalogCache struct compiles; `get_or_refresh` returns cached entries on unchanged dir
  - **Verify**: `cd gpu-query && cargo check 2>&1 | tail -5`
  - **Commit**: `feat(io): add CatalogCache with fingerprint-based invalidation`
  - _Requirements: FR-5_
  - _Design: Component 1 (CatalogCache)_

- [x] 2.2 Add executor and catalog_cache to AppState
  - **Do**: In `gpu-query/src/tui/app.rs`: (1) Add `use crate::gpu::executor::QueryExecutor;` and `use crate::io::catalog_cache::CatalogCache;`. (2) Add fields `pub executor: Option<QueryExecutor>` and `pub catalog_cache: CatalogCache` to `AppState`. (3) Initialize in `AppState::new()`: `executor: None`, `catalog_cache: CatalogCache::new(data_dir.clone())`. (4) Add method `pub fn get_or_init_executor(&mut self) -> Result<&mut QueryExecutor, String>` that lazily creates executor.
  - **Files**: `gpu-query/src/tui/app.rs`
  - **Done when**: AppState has executor + catalog_cache fields; get_or_init_executor works; compiles
  - **Verify**: `cd gpu-query && cargo check 2>&1 | tail -5`
  - **Commit**: `feat(tui): add persistent executor and catalog cache to AppState`
  - _Requirements: FR-4, FR-5_
  - _Design: Component 2 (Persistent QueryExecutor)_

- [x] 2.3 Replace scan_directory calls with CatalogCache in TUI
  - **Do**: (1) In `gpu-query/src/tui/ui.rs:345`: Replace `crate::io::catalog::scan_directory(&app.data_dir)` with `app.catalog_cache.get_or_refresh()`. Handle the `io::Result` -> `String` error conversion. Clone the entries to `Vec<TableEntry>` for ownership (catalog_cache returns `&[TableEntry]`). (2) In `gpu-query/src/tui/event.rs:215`: Same replacement for the DESCRIBE path. (3) In `gpu-query/src/tui/mod.rs:65`: Replace the initial `scan_directory` call with `app.catalog_cache.get_or_refresh()`.
  - **Files**: `gpu-query/src/tui/ui.rs`, `gpu-query/src/tui/event.rs`, `gpu-query/src/tui/mod.rs`
  - **Done when**: All 3 `scan_directory` calls in TUI replaced with `catalog_cache.get_or_refresh()`
  - **Verify**: `cd gpu-query && cargo check 2>&1 | tail -5`
  - **Commit**: `refactor(tui): use CatalogCache instead of per-query scan_directory`
  - _Requirements: FR-5_
  - _Design: Component 1 Integration Points_

- [x] 2.4 Replace QueryExecutor::new() calls with persistent executor
  - **Do**: (1) In `gpu-query/src/tui/ui.rs:375`: Replace `crate::gpu::executor::QueryExecutor::new()` with `app.get_or_init_executor()`. Adjust borrow patterns -- the executor is `&mut` borrowed from app, so ensure the executor borrow ends before `app.set_result()`. Pattern: extract executor via `app.executor.take()`, use it, put it back with `app.executor = Some(executor)`. (2) In `gpu-query/src/tui/event.rs:216`: Same replacement for the DESCRIBE executor creation.
  - **Files**: `gpu-query/src/tui/ui.rs`, `gpu-query/src/tui/event.rs`
  - **Done when**: QueryExecutor created once per TUI session, reused across queries
  - **Verify**: `cd gpu-query && cargo test --lib 2>&1 | tail -5`
  - **Commit**: `feat(tui): use persistent QueryExecutor from AppState`
  - _Requirements: FR-4, AC-3.1, AC-3.2_
  - _Design: Component 2 Integration Pattern_

- [x] 2.5 Phase 2 Checkpoint -- Persistent executor + cached catalog
  - **Do**: Run full test suite. Verify TUI compiles and executor persists (second query should skip Metal init). Run GPU integration tests. Verify CLI path is unaffected.
  - **Files**: None (verification only)
  - **Done when**: All tests pass; TUI uses persistent executor and cached catalog
  - **Verify**: `cd gpu-query && cargo test --lib 2>&1 | tail -5 && cargo test --test gpu_filter 2>&1 | tail -10`
  - **Commit**: `feat(tui): complete persistent executor + catalog cache integration`
  - _Requirements: AC-3.1, AC-3.2, AC-3.3, AC-4.1_

## Phase 3: Warm Cache -- Cross-Query Scan Persistence

Focus: Remove `scan_cache.clear()` from execute() to enable cross-query scan result caching. With file stat validation from 1.4, stale entries are automatically detected.

- [x] 3.1 Enable cross-query scan cache persistence
  - **Do**: Remove `self.scan_cache.clear()` from the top of `QueryExecutor::execute()` (added in task 1.5). The file stat validation from task 1.4 already handles invalidation. Add FIFO eviction: if `scan_cache.len() >= 8`, remove the first key from iteration order before inserting new entry. This enables warm-query performance where repeat queries on the same table skip the entire scan pipeline.
  - **Files**: `gpu-query/src/gpu/executor.rs`
  - **Done when**: Second query on same unchanged file hits scan cache; eviction works at 8 entries
  - **Verify**: `cd gpu-query && cargo test --lib 2>&1 | tail -5`
  - **Commit**: `feat(executor): enable cross-query scan cache with FIFO eviction`
  - _Requirements: FR-2, FR-6_
  - _Design: Component 3 (ScanCache)_

- [ ] 3.2 Add .refresh command to invalidate all caches
  - **Do**: In the dot-command handler (likely in `event.rs` or `ui.rs` where `.profile` is handled), add `.refresh` command that: (1) calls `app.catalog_cache.invalidate()`, (2) if executor exists, clears its scan_cache (`app.executor.as_mut().map(|e| e.scan_cache.clear())`), (3) sets status message "Caches refreshed." Need to add `pub fn clear_scan_cache(&mut self)` method to QueryExecutor if scan_cache is private.
  - **Files**: `gpu-query/src/tui/event.rs` or `gpu-query/src/tui/ui.rs`, `gpu-query/src/gpu/executor.rs`
  - **Done when**: `.refresh` command clears catalog cache and scan cache; status message shown
  - **Verify**: `cd gpu-query && cargo check 2>&1 | tail -5`
  - **Commit**: `feat(tui): add .refresh command to invalidate all caches`
  - _Requirements: FR-7, AC-4.4_
  - _Design: Cache Invalidation Summary_

- [ ] 3.3 Phase 3 Checkpoint -- Full warm-cache verification
  - **Do**: Run full test suite. Verify warm query path works (scan cache hit on repeat query). Verify file modification triggers re-scan. Verify `.refresh` clears caches.
  - **Files**: None (verification only)
  - **Done when**: All tests pass; warm cache provides expected speedup
  - **Verify**: `cd gpu-query && cargo test --lib 2>&1 | tail -5 && cargo test --test gpu_filter 2>&1 | tail -10`
  - **Commit**: `feat(executor): complete warm cache -- cross-query scan persistence verified`
  - _Requirements: AC-1.1, AC-1.2, AC-1.3, AC-5.1_

## Phase 4: Testing

Focus: Add targeted tests for cache correctness, invalidation, and performance benchmarks.

- [ ] 4.1 Unit tests for CatalogCache
  - **Do**: Add tests in `gpu-query/src/io/catalog_cache.rs` (mod tests): (1) `test_catalog_cache_hit_on_unchanged_dir` -- scan, verify second call returns same entries without re-scan. (2) `test_catalog_cache_miss_on_new_file` -- add file to temp dir, verify cache invalidates. (3) `test_catalog_cache_miss_on_modified_file` -- modify file mtime, verify invalidation. (4) `test_catalog_cache_invalidate` -- manual invalidation clears cache. Use `tempfile::TempDir` for test directories.
  - **Files**: `gpu-query/src/io/catalog_cache.rs`
  - **Done when**: 4 catalog cache unit tests pass
  - **Verify**: `cd gpu-query && cargo test catalog_cache 2>&1 | tail -10`
  - **Commit**: `test(io): add CatalogCache unit tests for hit/miss/invalidation`
  - _Requirements: AC-4.1, AC-4.3_

- [ ] 4.2 Unit tests for scan cache in executor
  - **Do**: Add tests in `gpu-query/src/gpu/executor.rs` (mod tests) or a new test file: (1) `test_scan_cache_hit` -- execute same table twice, verify scan_uncached called once. (2) `test_scan_cache_eviction` -- insert 9 entries, verify oldest evicted. (3) `test_compound_filter_single_scan` -- execute compound AND query, verify single scan via cache. (4) `test_scan_cache_invalidation_on_file_change` -- modify file between queries, verify re-scan. These tests require GPU (Metal device) so may need to be integration tests.
  - **Files**: `gpu-query/src/gpu/executor.rs` or `gpu-query/tests/gpu_cache.rs`
  - **Done when**: 4 scan cache tests pass
  - **Verify**: `cd gpu-query && cargo test scan_cache 2>&1 | tail -10`
  - **Commit**: `test(executor): add scan cache unit tests`
  - _Requirements: AC-2.1, AC-5.1, AC-5.3_

- [ ] 4.3 Performance benchmark for warm compound filter query
  - **Do**: Add benchmark to `gpu-query/benches/filter_throughput.rs` or create new `benches/query_latency.rs`. Benchmark: (1) Generate 1M-row CSV (5 INT64, 3 FLOAT64, 2 VARCHAR cols). (2) Create persistent QueryExecutor. (3) First iteration: cold scan. (4) Subsequent iterations: warm cache -- compound AND filter + GROUP BY. Use Criterion with `sample_size(20)`, `measurement_time(10s)`. Assert warm iteration < 50ms.
  - **Files**: `gpu-query/benches/filter_throughput.rs` or `gpu-query/benches/query_latency.rs`
  - **Done when**: Benchmark runs; warm 1M-row compound filter + GROUP BY < 50ms
  - **Verify**: `cd gpu-query && cargo bench --bench filter_throughput -- "compound" 2>&1 | tail -20`
  - **Commit**: `test(bench): add 1M-row warm compound filter benchmark`
  - _Requirements: NFR-1_

- [ ] 4.4 Regression test for all existing tests
  - **Do**: Run the full test suite including lib tests, GPU integration tests, and E2E golden tests. Verify all pass. Fix any failures introduced by the refactor.
  - **Files**: Various (fix any broken tests)
  - **Done when**: All 494+ lib tests, 155 GPU tests, 93 E2E tests pass
  - **Verify**: `cd gpu-query && cargo test --all-targets 2>&1 | tail -10`
  - **Commit**: `fix(tests): resolve any test regressions from cache refactor` (if needed)
  - _Requirements: NFR-3_

## Phase 5: Quality Gates

Focus: Final polish, lint, and PR preparation.

- [ ] 5.1 Clippy and formatting pass
  - **Do**: Run `cargo clippy --lib --tests -- -D warnings` and `cargo fmt --check`. Fix any warnings. Common issues: unused imports from refactored code, missing `#[allow(dead_code)]` on temporary helpers, borrow pattern suggestions.
  - **Files**: Various
  - **Done when**: Zero clippy warnings, formatting clean
  - **Verify**: `cd gpu-query && cargo fmt --check && cargo clippy --lib --tests -- -D warnings 2>&1 | tail -10`
  - **Commit**: `fix(lint): resolve clippy warnings and formatting issues`
  - _Requirements: NFR-7_

- [ ] 5.2 Final performance verification
  - **Do**: Run the benchmark suite and verify: (1) Warm 1M-row compound filter GROUP BY < 50ms. (2) All existing benchmarks show no regression (or improvement). (3) Run `cargo test --all-targets` one final time to confirm 100% pass rate.
  - **Files**: None (verification only)
  - **Done when**: All performance targets met; all tests green
  - **Verify**: `cd gpu-query && cargo test --all-targets 2>&1 | tail -5 && cargo bench --bench filter_throughput 2>&1 | tail -20`
  - **Commit**: None (verification only)
  - _Requirements: NFR-1, NFR-2, NFR-3_

- [ ] 5.3 Create PR and verify CI
  - **Do**: Push branch, create PR with `gh pr create`. PR title: "perf(executor): 7x query speedup via scan cache + persistent executor". PR body should include: before/after benchmark numbers, summary of 5 bottleneck fixes, test results. Use `gh pr checks --watch` to verify CI passes.
  - **Files**: None (git/GitHub operations)
  - **Done when**: PR created, CI green
  - **Verify**: `gh pr checks --watch`
  - **Commit**: None (PR creation)

## Notes

- **POC shortcuts taken**: Phase 1 clears scan cache between queries (1.5) for safety; cross-query caching enabled in Phase 3
- **Borrow checker strategy**: `resolve_input` returns `String` key, callers do `self.scan_cache.get(&key)` for immutable access after mutable cache population phase
- **Metal buffer safety**: `CachedScan` (or `ScanResult` in cache) must own `MmapFile` to keep `bytesNoCopy` buffers valid
- **CLI unaffected**: Only TUI code paths get persistent executor/cache; CLI `cli/mod.rs` continues with per-invocation executor
- **Production TODOs for Phase 2+**: Memory-aware eviction (track bytes not just entry count), configurable cache ceiling, warm/cold indicator in status bar
