---
spec: gpu-query-perf
phase: requirements
created: 2026-02-11
generated: auto
---

# Requirements: gpu-query-perf

## Summary

Fix 5 critical performance bottlenecks in gpu-query's execution engine to reduce 1M-row compound filter GROUP BY latency from 367ms to <50ms on warm queries. Introduce three caching layers (scan cache, persistent executor, catalog cache) while maintaining 100% backward compatibility with all 742+ existing tests.

## User Stories

### US-1: Instant warm query response
As a data analyst using gpu-query TUI, I want repeated queries on the same data to return in <50ms so that my analytical flow feels instantaneous.

**Acceptance Criteria**:
- AC-1.1: Warm 1M-row `SELECT region, COUNT(*), SUM(amount) FROM sales WHERE amount > 100 AND region = 'US' GROUP BY region` completes in <50ms
- AC-1.2: Second execution of same query on unchanged file hits scan cache (no re-mmap, no re-parse, no dictionary rebuild)
- AC-1.3: PSO cache retains compiled Metal pipeline states across queries in TUI session

### US-2: Compound filter without double scan
As a user writing compound WHERE clauses, I want `AND`/`OR` filters to scan the table once so that compound queries are no slower than single-predicate queries.

**Acceptance Criteria**:
- AC-2.1: `WHERE a > X AND b < Y` executes exactly one `execute_scan` call (verified by scan cache hit count)
- AC-2.2: Deeply nested compounds like `(A > 100 AND B < 50) OR C = 0` also trigger single scan
- AC-2.3: All 12 existing compound filter tests produce identical results

### US-3: Persistent GPU executor in TUI
As a TUI user, I want the GPU device and pipeline cache to persist across queries so that I do not pay Metal initialization cost on every query.

**Acceptance Criteria**:
- AC-3.1: `QueryExecutor` created once per TUI session, stored in `AppState`
- AC-3.2: Second query skips `MTLCreateSystemDefaultDevice`, metallib load, PSO compilation
- AC-3.3: CLI one-shot mode remains unaffected (per-invocation executor)

### US-4: Cached catalog and schema
As a TUI user, I want the file catalog and inferred schemas to be cached so that directory scans and schema inference do not repeat on every query.

**Acceptance Criteria**:
- AC-4.1: `scan_directory()` called once per session (or on cache miss from file change)
- AC-4.2: Schema inference runs once per table per session (cached in ScanResult)
- AC-4.3: File modification (mtime/size change) automatically invalidates affected cache entries
- AC-4.4: `.refresh` command forces full cache invalidation

### US-5: Dictionary caching
As a user querying VARCHAR columns, I want dictionary encoding to be cached with scan results so that CPU-side CSV re-reads are eliminated on repeat queries.

**Acceptance Criteria**:
- AC-5.1: `build_csv_dictionaries()` called once per table (on first scan), not on cache hit
- AC-5.2: Dictionary correctness preserved -- encoded GPU buffer content byte-identical to fresh build
- AC-5.3: Dictionary cache invalidated when source file changes

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Per-query scan cache (`HashMap<String, ScanResult>`) deduplicates scans within a single query execution | Must | US-2 |
| FR-2 | Cross-query scan cache persists ScanResult across queries on the same table | Must | US-1, US-5 |
| FR-3 | Scan cache validates entries via `(file_size, file_mtime)` stat check before returning | Must | US-4 |
| FR-4 | Persistent `QueryExecutor` in `AppState` with lazy initialization | Must | US-3 |
| FR-5 | `CatalogCache` struct with `(dir_mtime, per-file size+mtime)` fingerprinting | Must | US-4 |
| FR-6 | FIFO eviction policy, 8-entry max, 512MB configurable memory ceiling | Should | US-1 |
| FR-7 | `.refresh` dot command invalidates all caches (catalog, scan, schema, dictionaries) | Should | US-4 |
| FR-8 | Scan cache owns both `MmapFile` and `ColumnarBatch` to ensure Metal buffer lifetime safety | Must | US-1 |
| FR-9 | `resolve_input` returns table key (String) instead of owned ScanResult; callers access via cache | Must | US-2 |
| FR-10 | CLI one-shot mode unaffected -- per-invocation executor, no persistent state | Must | US-3 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Warm 1M-row compound filter GROUP BY query < 50ms | Performance |
| NFR-2 | Cold first query < 200ms (vs 367ms before) | Performance |
| NFR-3 | All 494 lib tests + 155 GPU integration tests + 93 E2E golden tests pass | Compatibility |
| NFR-4 | Heap delta < 5MB over 100 repeated queries | Memory |
| NFR-5 | Cache stat check overhead < 0.2ms per query (for 10 cached tables) | Performance |
| NFR-6 | Zero new runtime dependencies (no tokio, no async) | Architecture |
| NFR-7 | Clippy clean with `-D warnings` | Code Quality |

## Out of Scope

- Async/tokio query execution (per Q&A decision: sub-50ms makes sync acceptable)
- Status bar UX enhancements (warm/cold indicator, GPU% display)
- Progress bar for large scans
- Query cancellation (Ctrl+C)
- Cross-query LRU with configurable memory ceiling (deferred to future; per-query + simple FIFO sufficient)
- CLI persistent state (CLI stays per-invocation)
- Filesystem watch (`notify` crate) for auto-invalidation

## Dependencies

- Existing `ScanResult` struct in executor.rs (must be made cache-friendly -- own mmap + batch + schema)
- `AppState` struct in tui/app.rs (must be extended with executor + catalog_cache fields)
- `TableEntry` struct in io/catalog.rs (unchanged; used as cache key source)
- Criterion benchmark infrastructure (49 existing benches, new ones added in Phase 4)
