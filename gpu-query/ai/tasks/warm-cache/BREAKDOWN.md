---
id: warm-cache.BREAKDOWN
module: warm-cache
priority: 3
status: failing
version: 1
origin: spec-workflow
dependsOn:
  - executor-cache.BREAKDOWN
  - persistent-executor.BREAKDOWN
tags: [performance, gpu-query, cache]
testRequirements:
  unit:
    required: true
    pattern: "tests/**/*.rs"
---
# Cross-Query LRU ScanResult Cache

## Context

The executor-cache module (priority 0) adds a per-query `ScanCache` that eliminates redundant scans within a single query (fixing the double-scan bug). However, this cache is scoped to a single query execution -- it does not persist across queries.

For interactive TUI sessions where users run multiple queries against the same tables, re-scanning the CSV on every query still costs ~115ms (mmap + GPU parse + dictionary build). To reach the <30ms warm-query target, scan results must persist across queries.

This module promotes the per-query ScanCache to a cross-query LRU cache owned by the persistent `QueryExecutor`. It stores full `ScanResult` objects (mmap + ColumnarBatch with Metal GPU buffers + RuntimeSchema + dictionaries) keyed by `(table_name, file_size, file_mtime)`. A 512MB memory ceiling with FIFO eviction (8 entries max) bounds memory growth. The ceiling is user-configurable via a `--cache-limit` CLI flag.

With this cache, a warm repeat query skips the entire scan pipeline and goes directly to GPU filter + aggregate kernels (~25ms), achieving the projected 28-42ms warm-query latency.

## Acceptance Criteria

1. `ScanCache` persists across query executions within a TUI session (not just within a single query)
2. Second query on the same table returns a cache hit and skips `execute_scan()` entirely -- latency drops from ~115ms to <1ms for the scan phase
3. Cache is keyed by `(table_name, file_size, file_mtime)` -- automatic invalidation when the underlying file changes
4. FIFO eviction when cache exceeds 8 entries or 512MB total memory (whichever limit is hit first)
5. `--cache-limit <MB>` CLI flag allows users to configure the memory ceiling (default: 512MB)
6. `.cache clear` or `.refresh` dot command in TUI clears the entire ScanCache
7. Memory accounting: each `CachedScan` reports its `estimated_bytes()` (sum of Metal buffer sizes + dictionary HashMap overhead)
8. A warm 1M-row compound filter + GROUP BY query completes in <50ms (target: 28-42ms)
9. Metal buffer lifetime is correct: `MmapFile` outlives all `ColumnarBatch` Metal buffers that reference it via `bytesNoCopy`; both are dropped together when evicted
10. No GPU command buffer race: `waitUntilCompleted()` ensures no in-flight GPU work references evicted buffers
11. CLI one-shot mode does not use the warm cache (per-invocation executor has a fresh cache each time)
12. Cache statistics are available for status bar display: hit count, miss count, total cached bytes

## Technical Notes

- **Reference**: OVERVIEW.md Module Roadmap priority 3; TECH.md Section 6 (ScanCache design); PM.md Section 5 Risk Assessment (Memory Growth); QA.md Section 6.1 (Memory Risks)
- **Files to modify**:
  - `gpu-query/src/gpu/executor.rs` -- Promote `scan_cache: HashMap<String, ScanResult>` from per-query to persistent field on `QueryExecutor`; add `ScanCacheKey` struct with `(table_name, file_size, file_mtime)`; add `max_entries` and `max_bytes` limits; add `estimated_bytes()` method to `CachedScan`/`ScanResult`; add FIFO eviction logic
  - `gpu-query/src/gpu/executor.rs` -- Add `clear_scan_cache()` method; add `scan_cache_stats()` method returning `(hits, misses, cached_bytes)`
  - `gpu-query/src/tui/app.rs` -- Wire `.cache clear` dot command to `executor.clear_scan_cache()`
  - `gpu-query/src/cli/mod.rs` -- Add `--cache-limit <MB>` clap argument; pass to `QueryExecutor::new_with_config()`
  - `gpu-query/src/tui/ui.rs` -- Display cache stats in status bar when profile mode is on
- **Memory model**: Each cached 1M-row table with 5 INT64 + 3 FLOAT64 + 2 VARCHAR columns uses ~62MB. With 8 entries, worst case is ~496MB. The 512MB ceiling ensures this stays bounded.
- **Eviction strategy**: FIFO (remove oldest insertion). Simple, predictable, sufficient for typical 3-5 table workloads. Full LRU would require a linked list which adds complexity for minimal benefit at 8 entries.
- **Metal buffer safety**: `CachedScan` owns both `MmapFile` and `ColumnarBatch`. Rust's drop order (fields dropped in declaration order) means we must declare `batch` before `mmap` so Metal buffers are released before the mmap is unmapped. Alternatively, use `ManualDrop` or explicit drop ordering.
- **Test**: `cargo test --all-targets`; new `test_scan_cache_cross_query_hit`, `test_scan_cache_eviction`, `test_scan_cache_invalidation_on_file_change` tests; `cargo bench --bench query_latency -- "repeated_query"` for warm-path measurement
