# gpu-query Performance Optimization -- Spec Overview

## Executive Summary

gpu-query, a GPU-native local analytics engine for Apple Silicon, currently executes a 1M-row compound filter + GROUP BY query in 367ms -- over 7x above the <50ms competitive target. Five critical host-side bottlenecks (redundant I/O, missing caches, stateless per-query architecture) have been identified, all structurally independent and low-risk. The fixes introduce three caching layers (executor persistence, scan result caching, catalog caching) that collectively deliver a projected 7.3x latency reduction, bringing warm-query performance to 28-42ms and placing gpu-query below the human perceptual threshold.

## PM Summary

- **Double-scan bug (P0)**: `GpuCompoundFilter` calls `resolve_input()` independently on both branches, triggering two full CSV scans for every compound `WHERE` clause -- the single biggest bottleneck at ~120-150ms of waste
- **Stateless executor (P1)**: `QueryExecutor::new()` is called per query, recreating the Metal device, loading the metallib, and starting with an empty PSO cache every time -- costing 10-25ms before any data is touched
- **CPU dictionary rebuild (P1)**: `build_csv_dictionaries()` re-reads the entire CSV on CPU after the GPU has already parsed it, adding 40-80ms of redundant I/O that scales linearly with row count
- **Schema re-inference (P2)** and **catalog re-scan (P2)**: Both repeat deterministic work on every query (2-5ms + 1-3ms), compounding with the double-scan bug
- **Competitive context**: DuckDB achieves 30-80ms on CPU alone; if a GPU engine cannot match it, there is no credible product story

## UX Summary

- **Perceptual threshold**: Going from 367ms (noticeable delay) to <50ms (below the ~65ms perceptual floor) is a qualitative shift from "waits for query" to "instant response"
- **Persistent executor in AppState**: Initialize once at TUI startup with a brief "Initializing GPU..." splash; surface warm/cold status in the status bar to teach users the system improves
- **Status bar enhancement**: Expand from `"4 rows | 367.6ms"` to `"4 rows | 12.3ms (warm) | GPU 94%"` with optional profile mode breakdown
- **First-query experience**: The most psychologically important query -- with caching, cold-start drops from 400ms to ~195ms; warm repeats feel instant
- **Async execution (deferred)**: Keep synchronous for now per Q&A decision; sub-50ms latency makes the freeze imperceptible

## Tech Summary

- **Three caching layers**: (1) CatalogCache with (dir_mtime, file_size, file_mtime) fingerprinting, (2) Persistent QueryExecutor with lazy init in AppState, (3) ScanCache (HashMap<table_name, ScanResult>) that unifies fixes for double-scan, schema caching, and dictionary caching
- **Borrow checker strategy**: Return table-name keys from `resolve_input()` instead of owned `ScanResult` values; access cached data via `self.scan_cache.get(&key)` after mutation phase completes
- **Cache invalidation**: Per-file `(size, mtime)` stat check (~0.1ms per file); `.refresh` command for manual override; automatic miss on file change
- **Memory budget**: ~62MB per cached 1M-row table (5 INT64 + 3 FLOAT64 + 2 VARCHAR); 8-entry FIFO cap; 512MB configurable ceiling for warm-cache layer
- **Performance budget**: Cold first query ~195ms (1.88x improvement); warm repeat query 28-42ms (8.7-13.1x improvement); GPU kernel time is the floor at ~25ms

## QA Summary

- **Existing baseline**: 494 lib unit tests + 155 GPU integration tests + 93 E2E golden tests + 49 criterion benchmarks -- all must continue to pass
- **New tests**: ~45 new tests covering cache hit/miss/invalidation, compound filter single-scan verification, memory stability, and 1M-row performance assertions
- **Benchmark protocol**: Save pre-fix baselines with `--save-baseline pre_opt`, compare post-fix; statistical confidence via Criterion (p=0.05, 20 samples, 5% noise threshold)
- **Cache correctness invariants**: Cached schema/catalog/dictionary must be byte-identical to freshly computed versions; verified on every test run
- **Hard acceptance criterion**: Warm 1M-row compound filter + GROUP BY < 50ms; all 742+ existing tests green; heap delta < 5MB over 100 queries

## Q&A Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Per-query scan cache or targeted compound-filter fix? | Per-query scan cache (`HashMap<table_name, ScanResult>`) | More general -- also fixes triple-scans for deeply nested compounds |
| Cache ColumnarBatch across queries? | LRU cache with 512MB memory ceiling, user-configurable | Enables sub-30ms warm repeats; memory bounded |
| Cache invalidation strategy? | `mtime + size` stat check (~0.1ms per file) | Fast, automatic, no external dependencies |
| Async query execution? | Keep synchronous | Sub-50ms makes freeze imperceptible; avoids tokio complexity |
| ScanCache key design? | `(table_name, file_size, mtime)` | Automatic invalidation on file change |
| Executor lifetime (TUI)? | Persistent `QueryExecutor` in AppState with lazy init | Amortizes Metal device + PSO cache across all queries |
| CatalogCache design? | Fingerprinting with `(dir_mtime, per-file size+mtime)` | O(n) stat calls on hit (~0.1ms for 10 files) |
| Eviction policy? | FIFO, 8 entry max | Simple, sufficient for typical workloads |
| CLI behavior? | CLI stays per-invocation; TUI gets persistent state | CLI is one-shot; no state to persist |
| Async/tokio? | No async/tokio -- keep sync, optimize the hot path | Complexity not justified when target is <50ms |

## Module Roadmap

| Module | Priority | Description |
|--------|----------|-------------|
| executor-cache | 0 | Fix double-scan bug in GpuCompoundFilter + add per-query ScanCache |
| persistent-executor | 1 | Move QueryExecutor to AppState with lazy init; persist Metal device + PSO cache |
| catalog-cache | 2 | Add CatalogCache with (size, mtime) fingerprinting; cache RuntimeSchema per table |
| warm-cache | 3 | Cross-query LRU ColumnarBatch cache with 512MB ceiling and FIFO eviction |
| benchmarks | 4 | Criterion benchmarks for warm/cold queries; --benchmark CLI flag; acceptance tests |
