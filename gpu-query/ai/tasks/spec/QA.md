# QA Strategy: gpu-query Performance Optimization (367ms -> <50ms)

## 1. Executive Summary

This document defines the QA strategy for validating 5 critical performance bottleneck fixes in the gpu-query execution engine. The fixes target a 7.3x latency reduction on 1M-row queries (367ms current -> <50ms target) on Apple Silicon M4.

**Scope of changes:**
- `src/gpu/executor.rs` (3962 lines) -- compound filter double-scan fix, executor caching, schema caching, dictionary caching
- `src/tui/app.rs` / `src/tui/event.rs` / `src/tui/ui.rs` -- executor/catalog reuse in TUI
- `src/io/catalog.rs` -- catalog caching layer

**Existing test baseline:**
- 494 lib unit tests (Linux-compatible, no GPU)
- 155 GPU integration tests (require Metal device)
- 93 E2E golden tests
- 49 criterion benchmarks across 5 files
- Fuzz targets for CSV/JSON parsers
- CI pipeline: 8 jobs (fmt, clippy, unit, GPU, shader-validation, bench, fuzz, gate)

---

## 2. Research-Informed Testing Principles

Based on current best practices in performance regression testing ([ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S0950584924002465), [Criterion.rs docs](https://bheisler.github.io/criterion.rs/book/), [Facebook cache consistency](https://engineering.fb.com/2022/06/08/core-infra/cache-made-consistent/)):

1. **Statistical rigor over point measurements** -- Criterion's statistical engine detects real changes vs. noise. All performance gates use p-value confidence, not single-run timings.
2. **Baseline comparison** -- Save pre-fix baselines with `--save-baseline pre_opt`, compare post-fix with `--baseline pre_opt`.
3. **Shift-left** -- Every bottleneck fix gets a targeted unit test and micro-benchmark, not just end-to-end.
4. **Cache mutation tracing** -- Inspired by Facebook's stateful tracing approach: log cache hits/misses/evictions during tests to verify correctness.
5. **Throughput AND latency** -- Measure both: latency for interactive UX, throughput for batch workloads.

---

## 3. Test Matrix: Per-Bottleneck Coverage

### Bottleneck 1: Double CSV Scan in Compound Filter (BUG)

**Root Cause:** `resolve_input()` at executor.rs:379-382 calls `self.resolve_input(left)` and `self.resolve_input(right)` independently, each triggering a full `execute_scan()` (mmap + GPU CSV parse + dictionary build). The left scan result is kept; the right scan result is discarded after extracting its filter bitmask.

| Test Type | Test Name | What It Validates | Location |
|-----------|-----------|-------------------|----------|
| Unit | `test_compound_filter_single_scan` | Assert `execute_scan` called exactly once for compound AND/OR | `src/gpu/executor.rs` (mod tests) |
| Unit | `test_compound_filter_bitmask_correctness_and` | Bitmask AND produces correct match_count on known data | `src/gpu/executor.rs` (mod tests) |
| Unit | `test_compound_filter_bitmask_correctness_or` | Bitmask OR produces correct match_count on known data | `src/gpu/executor.rs` (mod tests) |
| Integration | `test_compound_and_count` (existing) | E2E correctness preserved | `tests/gpu_filter.rs:578` |
| Integration | `test_compound_or_sum` (existing) | E2E correctness preserved | `tests/gpu_filter.rs:648` |
| Integration | `test_compound_same_column_and` (existing) | Range filter correctness | `tests/gpu_filter.rs:728` |
| Integration | `test_compound_same_column_or` (existing) | Exclusive ranges correctness | `tests/gpu_filter.rs:747` |
| Integration | `test_compound_filter_1m_rows` | **NEW**: 1M-row compound filter returns correct count/sum | `tests/gpu_filter.rs` |
| Benchmark | `filter_compound/and_predicate` (existing) | Latency improvement measured | `benches/filter_throughput.rs:111` |
| Benchmark | `filter_compound/or_predicate` (existing) | Latency improvement measured | `benches/filter_throughput.rs:121` |
| Benchmark | `compound_filter_1m/and` | **NEW**: 1M-row compound filter latency | `benches/filter_throughput.rs` |
| Benchmark | `compound_filter_1m/or` | **NEW**: 1M-row compound filter latency | `benches/filter_throughput.rs` |
| Regression | `test_nested_compound_filters` | **NEW**: `(A > 100 AND B < 50) OR C = 0` still correct | `tests/gpu_filter.rs` |

**Acceptance Criteria:**
- Compound filter latency at 1M rows: **<40ms** (was ~2x single filter due to double scan)
- All 12 existing compound filter tests pass with identical results
- Zero additional GPU buffer allocations compared to single filter path

---

### Bottleneck 2: QueryExecutor Recreated Per Query

**Root Cause:** In `tui/event.rs:216` and `tui/ui.rs:375`, `QueryExecutor::new()` is called for every query execution. This allocates a new `GpuDevice` (Metal device + command queue + library compilation) and an empty `PsoCache` each time, losing all compiled pipeline state objects.

| Test Type | Test Name | What It Validates | Location |
|-----------|-----------|-------------------|----------|
| Unit | `test_executor_pso_cache_persists` | **NEW**: PSO cache retains entries across multiple execute() calls | `src/gpu/executor.rs` (mod tests) |
| Unit | `test_executor_reuse_correctness` | **NEW**: 10 sequential queries on same executor produce correct results | `src/gpu/executor.rs` (mod tests) |
| Unit | `test_executor_reuse_different_tables` | **NEW**: Queries against different tables on reused executor | `src/gpu/executor.rs` (mod tests) |
| Integration | `test_pso_cache_reuse` (existing) | PSO cache grows and reuses correctly | `tests/gpu_filter.rs:278` |
| Integration | `test_executor_sequential_queries` | **NEW**: 50 queries sequentially, verify PSO cache size stabilizes | `tests/gpu_filter.rs` |
| Benchmark | `executor_reuse/cold_start` | **NEW**: First query on fresh executor (includes MTLLibrary compile) | `benches/query_latency.rs` |
| Benchmark | `executor_reuse/warm_pso` | **NEW**: Subsequent query reusing compiled PSOs | `benches/query_latency.rs` |
| Benchmark | `executor_reuse/warm_100_queries` | **NEW**: Average latency over 100 queries with reused executor | `benches/query_latency.rs` |
| Memory | `test_executor_memory_stable` | **NEW**: Heap delta < 1MB after 100 queries on reused executor | `tests/gpu_filter.rs` |

**Acceptance Criteria:**
- Second-query latency: **<25ms** (first query may be ~50ms due to PSO compilation)
- PSO cache stabilizes within 5 queries (no unbounded growth for fixed query set)
- No `MTLLibrary` recompilation after first query

---

### Bottleneck 3: CPU-Side VARCHAR Dictionary Building

**Root Cause:** `build_csv_dictionaries()` at executor.rs:437 reads the entire CSV file from disk via CPU to extract unique string values for each VARCHAR column. This is called on every scan, even for repeated queries on the same file. Additionally, `build_csv_dictionaries_for_chunk()` at executor.rs:2236 rebuilds per-chunk dictionaries in the batched path.

| Test Type | Test Name | What It Validates | Location |
|-----------|-----------|-------------------|----------|
| Unit | `test_dictionary_cache_hit` | **NEW**: Second query on same file returns cached dictionary | `src/storage/dictionary.rs` |
| Unit | `test_dictionary_cache_invalidation_on_modify` | **NEW**: File modification invalidates cached dictionary | `src/storage/dictionary.rs` |
| Unit | `test_dictionary_cache_different_files` | **NEW**: Separate cache entries for different files | `src/storage/dictionary.rs` |
| Unit | `test_dictionary_correctness_unchanged` | **NEW**: Cached dict produces identical encoding to fresh build | `src/storage/dictionary.rs` |
| Integration | `test_string_filter_eq_europe` (existing) | VARCHAR filter correctness preserved | `tests/gpu_filter.rs:447` |
| Integration | `test_string_filter_group_by_region` (existing) | GROUP BY VARCHAR correctness preserved | `tests/gpu_filter.rs:547` |
| Integration | `test_varchar_filter_large_cardinality` | **NEW**: 10K distinct strings, filter correctness | `tests/gpu_filter.rs` |
| Benchmark | `dictionary_build/10k_distinct` | **NEW**: Cold dict build for 10K distinct strings | `benches/scan_throughput.rs` |
| Benchmark | `dictionary_build/cached_hit` | **NEW**: Cached dict lookup (should be ~0ms) | `benches/scan_throughput.rs` |
| Benchmark | `varchar_filter/1m_rows` | **NEW**: VARCHAR filter on 1M rows with cached dict | `benches/filter_throughput.rs` |

**Acceptance Criteria:**
- Dictionary cache hit: **<0.1ms** overhead (was 15-30ms for CPU rebuild)
- Dictionary cache miss (first query): unchanged from current behavior
- Cache invalidation triggers within 1 file-system stat check
- Dictionary encoding produces byte-identical GPU buffer content whether cached or freshly built

---

### Bottleneck 4: Schema Inference Re-reads File Every Query

**Root Cause:** `infer_schema_from_csv()` at executor.rs:427 and :2194 re-reads and parses up to 100 rows of the CSV file on every query to determine column types. This is redundant for repeated queries on unchanged files.

| Test Type | Test Name | What It Validates | Location |
|-----------|-----------|-------------------|----------|
| Unit | `test_schema_cache_returns_same_schema` | **NEW**: Cached schema matches fresh inference | `src/gpu/executor.rs` (mod tests) |
| Unit | `test_schema_cache_invalidation_file_change` | **NEW**: Modified file triggers re-inference | `src/gpu/executor.rs` (mod tests) |
| Unit | `test_schema_cache_invalidation_file_delete` | **NEW**: Deleted file gracefully handled | `src/gpu/executor.rs` (mod tests) |
| Unit | `test_schema_cache_key_includes_path` | **NEW**: Different files get different cache entries | `src/gpu/executor.rs` (mod tests) |
| Integration | `test_infer_all_int_columns` (existing) | Schema inference correctness | `tests/gpu_schema.rs:41` |
| Integration | `test_infer_mixed_types_across_rows` (existing) | Type voting correctness | `tests/gpu_schema.rs:74` |
| Integration | `test_infer_nullable_from_empty_field` (existing) | Nullable detection | `tests/gpu_schema.rs:85` |
| Integration | `test_schema_cache_across_queries` | **NEW**: Run 3 queries, schema inferred only on first | `tests/gpu_schema.rs` |
| Benchmark | `schema_inference/cold_50_rows` | **NEW**: Fresh inference on 50-row file | `benches/scan_throughput.rs` |
| Benchmark | `schema_inference/cached_hit` | **NEW**: Cached schema return (should be <0.01ms) | `benches/scan_throughput.rs` |

**Acceptance Criteria:**
- Schema cache hit: **<0.01ms** (was ~2-5ms for file read + parse)
- Schema returned from cache is structurally identical (same DataType, nullable flags, column order)
- Cache key includes: file path + file size + modification timestamp
- File with changed mtime triggers full re-inference

---

### Bottleneck 5: Catalog Directory Re-scanned Every Query

**Root Cause:** `catalog::scan_directory()` at `tui/event.rs:215`, `tui/ui.rs:345`, and in every benchmark `run_query()` helper calls `std::fs::read_dir()` + format detection + CSV header parsing on every query. This is pure I/O waste for unchanged directories.

| Test Type | Test Name | What It Validates | Location |
|-----------|-----------|-------------------|----------|
| Unit | `test_catalog_cache_returns_same_entries` | **NEW**: Cached catalog matches fresh scan | `src/io/catalog.rs` |
| Unit | `test_catalog_cache_invalidation_new_file` | **NEW**: Adding a file invalidates catalog cache | `src/io/catalog.rs` |
| Unit | `test_catalog_cache_invalidation_delete_file` | **NEW**: Removing a file invalidates catalog cache | `src/io/catalog.rs` |
| Unit | `test_catalog_cache_invalidation_modify_file` | **NEW**: Modifying a file's content invalidates cache | `src/io/catalog.rs` |
| Unit | `test_catalog_cache_key_includes_dir_path` | **NEW**: Different directories get separate cache entries | `src/io/catalog.rs` |
| Integration | `test_scan_csv_files` (existing) | Catalog scan correctness | `src/io/catalog.rs:106` |
| Integration | `test_scan_mixed_formats` (existing) | Multi-format detection | `src/io/catalog.rs:133` |
| Integration | `test_catalog_cache_across_queries` | **NEW**: 10 queries, directory scanned only on first | `tests/e2e_csv.rs` |
| Benchmark | `catalog_scan/cold_10_files` | **NEW**: Fresh directory scan with 10 files | `benches/scan_throughput.rs` |
| Benchmark | `catalog_scan/cached_hit` | **NEW**: Cached catalog return (should be <0.05ms) | `benches/scan_throughput.rs` |

**Acceptance Criteria:**
- Catalog cache hit: **<0.05ms** (was ~1-3ms for readdir + format detect + header parse)
- Cache invalidation on directory mtime change
- Catalog entries from cache are identical (same table names, formats, paths, CSV metadata)
- Adding/removing files between queries triggers rescan

---

## 4. Benchmark Design: Before/After Measurement Protocol

### 4.1 Baseline Capture (Pre-Fix)

```bash
# Save pre-optimization baselines
cd gpu-query
cargo bench -- --save-baseline pre_opt

# Specifically capture the key latency numbers
cargo bench --bench query_latency -- --save-baseline pre_opt
cargo bench --bench filter_throughput -- --save-baseline pre_opt
cargo bench --bench scan_throughput -- --save-baseline pre_opt
```

### 4.2 Post-Fix Comparison

```bash
# Compare against saved baseline
cargo bench -- --baseline pre_opt

# Generate HTML reports
# Results in target/criterion/<group>/<bench>/report/index.html
```

### 4.3 New Benchmark Groups Required

**File: `benches/query_latency.rs`** -- Add the following benchmark groups:

| Group | Benchmarks | Purpose |
|-------|-----------|---------|
| `compound_filter_latency` | `and_1m_rows`, `or_1m_rows`, `nested_and_or_1m` | Measure fix #1 impact at scale |
| `executor_lifecycle` | `cold_start`, `warm_pso_5th_query`, `warm_100th_query` | Measure fix #2 amortized cost |
| `repeated_query_latency` | `same_query_x10`, `alternating_queries_x10` | Measure all caches combined |

**File: `benches/scan_throughput.rs`** -- Add:

| Group | Benchmarks | Purpose |
|-------|-----------|---------|
| `schema_inference_cache` | `cold_infer`, `cached_infer` | Measure fix #4 improvement |
| `catalog_scan_cache` | `cold_scan`, `cached_scan` | Measure fix #5 improvement |
| `dictionary_cache` | `cold_build_varchar`, `cached_lookup_varchar` | Measure fix #3 improvement |

**File: `benches/filter_throughput.rs`** -- Add:

| Group | Benchmarks | Purpose |
|-------|-----------|---------|
| `compound_1m` | `and_1m`, `or_1m` | 1M-row compound filter at scale |

### 4.4 Benchmark Configuration

All new benchmarks should use:
```rust
group.sample_size(20);           // Sufficient for statistical confidence
group.measurement_time(Duration::from_secs(10)); // Longer window for GPU variance
group.noise_threshold(0.05);     // 5% noise threshold for GPU jitter
group.significance_level(0.05);  // 95% confidence for regression detection
group.throughput(Throughput::Elements(row_count as u64));
```

### 4.5 Key Latency Targets (1M Rows)

| Scenario | Before (measured) | After (target) | Improvement |
|----------|-------------------|-----------------|-------------|
| Simple filter (WHERE x > 500) | ~120ms | <30ms | 4x |
| Compound AND filter | ~240ms | <35ms | 7x |
| Compound OR filter | ~240ms | <35ms | 7x |
| Aggregate with filter | ~150ms | <40ms | 3.7x |
| TPC-H Q1 adapted (100K rows) | ~50ms | <10ms | 5x |
| Full pipeline (filter+agg) | ~180ms | <45ms | 4x |
| **E2E query latency (1M rows)** | **367ms** | **<50ms** | **7.3x** |

---

## 5. Cache Correctness Testing Strategy

Caching introduces the most dangerous class of bugs: stale data returning silently correct-looking but wrong results. The following strategy is modeled on Facebook's cache consistency verification approach.

### 5.1 Invariant Tests (Must Never Break)

| Invariant | How Verified | Frequency |
|-----------|-------------|-----------|
| Cached schema == freshly inferred schema | Byte-compare `RuntimeSchema` after each cache hit | Every test run |
| Cached catalog == freshly scanned catalog | Compare `Vec<TableEntry>` after each cache hit | Every test run |
| Cached dictionary == freshly built dictionary | Compare encoded GPU buffer bytes | Every test run |
| Query result with caches == query result without caches | Run same query with cache enabled/disabled, diff results | Nightly |
| Cache never returns data for a different file | Parameterized test with 3+ files | Every test run |

### 5.2 Cache Invalidation Test Scenarios

| Scenario | Expected Behavior | Test Method |
|----------|-------------------|-------------|
| File content changes between queries | Re-infer schema, re-build dict, re-scan catalog | Modify file, verify cache miss |
| File deleted between queries | Error or graceful fallback | Delete file, run query, check error |
| New file added to directory | Catalog cache invalidated, new table visible | Add file, query new table |
| File renamed | Old table gone, new table appears | Rename, query both |
| File size changes but mtime doesn't | Detect via size check in cache key | Truncate file, force same mtime |
| Concurrent query during file write | No crash, eventually consistent result | Write file in thread, query simultaneously |
| Empty directory after files removed | Catalog returns empty, queries fail gracefully | Remove all files, query |

### 5.3 Cache Key Design Validation

The cache key for each layer must be tested for completeness:

```
Schema cache key:  (file_path, file_size, mtime_ns)
Dict cache key:    (file_path, file_size, mtime_ns, column_index)
Catalog cache key: (dir_path, dir_mtime_ns, entry_count)
```

Tests must verify:
- Identical files in different directories get separate cache entries
- File with same content but different mtime triggers re-cache
- File with same mtime but different size triggers re-cache (guards against fast-modify)

---

## 6. Risk Analysis

### 6.1 Memory Risks from Caching

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Unbounded schema cache growth | OOM on directory with 1000+ files | Low | LRU eviction or bounded HashMap (max 64 entries) |
| Dictionary cache holding large strings | 100MB+ for high-cardinality VARCHAR columns | Medium | Cache size limit (e.g., 256MB total), evict LRU |
| Executor PSO cache growth | One PSO per unique (op, type, nullable) combo = ~36 max | Very Low | Bounded by design (finite key space) |
| Catalog cache holding stale file handles | File descriptor leak if mmap references cached | Low | Cache stores metadata only, not mmap handles |
| Triple memory: cache + old + new during invalidation | 2x peak during cache rebuild | Medium | Drop old cache entry before building new one |

**Recommended tests:**

```rust
#[test]
fn test_schema_cache_bounded_size() {
    // Create 100 different CSV files
    // Run queries against each
    // Assert cache.len() <= MAX_CACHE_SIZE
    // Assert memory delta < 10MB
}

#[test]
fn test_dictionary_cache_eviction() {
    // Create CSV with 100K distinct VARCHAR values
    // Cache the dictionary
    // Assert cache memory < 256MB limit
    // Add another large-cardinality file
    // Assert first entry evicted
}
```

### 6.2 Correctness Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Compound filter returns wrong results after fix | Silent data corruption in query results | Medium | Run ALL 12 existing compound filter tests + new 1M-row tests |
| Shared scan in compound filter: bitmask buffer aliasing | GPU race condition if left/right filter write same buffer | Low | Allocate separate bitmask buffers per sub-filter |
| Stale schema after file ALTER (add column) | Wrong column count, index out of bounds | Medium | Schema cache invalidation on mtime change |
| PSO cache key collision | Wrong kernel dispatched | Very Low | Key is (op_name, type_code, nullable) triple, unit tested |
| Dictionary cache returns wrong encoding for modified file | String filter matches wrong rows | High if no invalidation | Invalidation test + byte-comparison test |

### 6.3 Concurrency Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| TUI query while file is being written | Partial read, corrupt parse | Low | File-level read lock or retry on parse error |
| Multiple TUI tabs sharing executor | PSO cache thread safety | Low (current TUI is single-threaded) | `Mutex<PsoCache>` if threading added later |
| Benchmark iteration reusing executor during Metal async | Command buffer submitted before previous completes | Very Low | `waitUntilCompleted` in current impl prevents this |

---

## 7. CI Pipeline Changes

### 7.1 New CI Job: Performance Regression Gate

Add to `.github/workflows/ci.yml`:

```yaml
perf-regression:
  name: Performance regression check
  if: github.event_name == 'pull_request'
  runs-on: macos-14
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    - uses: dtolnay/rust-toolchain@stable
    - uses: Swatinem/rust-cache@v2
      with:
        workspaces: gpu-query
    - name: Checkout base branch benchmarks
      run: |
        git checkout ${{ github.base_ref }}
        cargo bench -- --save-baseline base
        git checkout ${{ github.head_ref }}
    - name: Run benchmarks and compare
      run: |
        cargo bench -- --baseline base 2>&1 | tee bench-compare.txt
    - name: Check for regressions
      run: |
        if grep -q "regressed" bench-compare.txt; then
          echo "::warning::Performance regression detected!"
          grep "regressed" bench-compare.txt
        fi
    - name: Upload comparison
      uses: actions/upload-artifact@v4
      with:
        name: bench-comparison
        path: gpu-query/bench-compare.txt
```

### 7.2 Updated Benchmark Job (Main Branch)

The existing `benchmarks` job (ci.yml:96) should be updated to save named baselines for tracking over time:

```yaml
- run: cargo bench -- --save-baseline "main_$(date +%Y%m%d_%H%M%S)"
```

### 7.3 Test Execution Order

For the PR that implements these fixes, CI should run in this order:

1. **fmt + clippy** -- Fast syntax/lint check
2. **Unit tests** (`cargo test --lib`) -- Verify cache logic, no GPU needed
3. **GPU integration tests** (`cargo test --all-targets`) -- Verify correctness with Metal
4. **Shader validation** -- Metal shader validation enabled
5. **Performance regression** -- Compare benchmarks against base branch
6. **CI gate** -- All 5 above must pass

---

## 8. Test Implementation Priorities

### Phase 1: Correctness First (Before Any Performance Fix)

1. Run full existing test suite, capture baseline pass rate: 494 + 155 + 93 = 742 tests
2. Save criterion baselines: `cargo bench -- --save-baseline pre_opt`
3. Write the 12 existing compound filter tests into a "golden results" file for comparison

### Phase 2: Per-Fix Validation (With Each Fix)

| Fix Order | Fix | Critical New Tests | Existing Tests to Watch |
|-----------|-----|-------------------|------------------------|
| 1 | Double scan bug | `test_compound_filter_single_scan`, `compound_filter_1m` bench | All 12 compound filter tests |
| 2 | Executor reuse | `test_executor_pso_cache_persists`, `executor_reuse` bench group | All E2E tests (use fresh executor) |
| 3 | Dictionary cache | `test_dictionary_cache_hit`, `test_dictionary_cache_invalidation_on_modify` | All VARCHAR filter tests (6 tests) |
| 4 | Schema cache | `test_schema_cache_returns_same_schema`, `test_schema_cache_invalidation_file_change` | All 14 schema inference tests |
| 5 | Catalog cache | `test_catalog_cache_returns_same_entries`, `test_catalog_cache_invalidation_new_file` | All 8 catalog unit tests |

### Phase 3: Integration Validation (All Fixes Combined)

1. Run full test suite: all 742+ tests must pass
2. Run criterion comparison: `cargo bench -- --baseline pre_opt`
3. Verify E2E 1M-row query latency < 50ms
4. Run memory stability test: 1000 queries, heap growth < 50MB
5. Manual TUI smoke test: rapid query execution, verify responsive UI

---

## 9. Acceptance Criteria Summary

### Hard Requirements (PR Blocks Without These)

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| All existing tests pass | 742/742 green | `cargo test --all-targets` |
| E2E query latency (1M rows, warm) | < 50ms | `cargo bench --bench query_latency` |
| Compound filter latency (1M rows) | < 40ms | `cargo bench --bench filter_throughput` |
| No memory leak over 100 queries | Heap delta < 5MB | Custom test with allocation tracking |
| Cache correctness invariants | All pass | New unit tests (Section 5.1) |
| Cache invalidation on file change | All scenarios pass | New unit tests (Section 5.2) |
| Clippy clean | Zero warnings | `cargo clippy --lib --tests -- -D warnings` |

### Soft Requirements (Track but Don't Block)

| Criterion | Target | Notes |
|-----------|--------|-------|
| First-query (cold) latency | < 80ms | Includes PSO compilation |
| PSO cache stabilization | Within 5 queries | For a fixed query workload |
| Schema cache hit ratio | > 95% in TUI session | After initial scan |
| Benchmark noise | < 5% coefficient of variation | Across 20 samples |

---

## 10. Verification Commands

```bash
# Full correctness check
cargo test --all-targets 2>&1 | tail -5

# Performance baseline save (run BEFORE fixes)
cargo bench -- --save-baseline pre_opt

# Performance comparison (run AFTER fixes)
cargo bench -- --baseline pre_opt

# Specific benchmark groups for each fix
cargo bench --bench filter_throughput -- "compound"     # Fix 1
cargo bench --bench query_latency -- "executor_reuse"   # Fix 2
cargo bench --bench scan_throughput -- "dictionary"      # Fix 3
cargo bench --bench scan_throughput -- "schema"          # Fix 4
cargo bench --bench scan_throughput -- "catalog"         # Fix 5

# Memory check (manual, requires heap profiler)
MALLOC_CONF="stats_print:true" cargo test test_executor_memory_stable -- --nocapture

# Shader validation pass
MTL_SHADER_VALIDATION=1 cargo test --test gpu_filter --test gpu_aggregate

# CI-equivalent full validation
cargo fmt --check && \
  cargo clippy --lib --tests -- -D warnings && \
  cargo test --all-targets && \
  cargo bench -- --baseline pre_opt
```

---

## 11. New Test Count Summary

| Category | Existing | New | Total |
|----------|----------|-----|-------|
| Unit tests (cache logic) | 494 | ~18 | ~512 |
| GPU integration tests | 155 | ~6 | ~161 |
| E2E golden tests | 93 | 0 | 93 |
| Criterion benchmarks | 49 | ~14 | ~63 |
| Cache invariant tests | 0 | ~5 | 5 |
| Memory stability tests | 0 | ~2 | 2 |
| **Total** | **791** | **~45** | **~836** |

---

*QA Strategy v1.0 -- authored by QA Manager agent (agent-foreman:qa)*
*Target: gpu-query performance optimization PR*
