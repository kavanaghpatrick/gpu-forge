---
spec: gpu-query-perf
phase: research
created: 2026-02-11
generated: auto
---

# Research: gpu-query-perf

## Executive Summary

gpu-query's 1M-row compound filter GROUP BY takes 367ms vs <50ms target due to five host-side bottlenecks: double CSV scan, per-query executor recreation, CPU dictionary rebuild, schema re-inference, and catalog re-scan. All are redundant I/O / missing caches -- GPU kernel time is only ~25ms. Three caching layers (executor persistence, per-query ScanCache, catalog cache) eliminate all redundant work. Feasibility is high; all fixes are internal refactors with no API changes.

## Foreman Analysis Summary

Five-agent analysis completed (PM, UX, TECH, QA agents). Key findings cited below.

| Agent | Doc | Key Contribution |
|-------|-----|-----------------|
| PM | `gpu-query/ai/tasks/spec/PM.md` | Competitive analysis (DuckDB 30-80ms CPU), bottleneck prioritization P0-P2, 6-8x savings estimate |
| UX | `gpu-query/ai/tasks/spec/UX.md` | Perceptual threshold analysis (~65ms), status bar enhancement design, first-query experience |
| TECH | `gpu-query/ai/tasks/spec/TECH.md` (1073 lines) | Full architecture: ScanCache, CatalogCache, borrow checker strategy, memory budget, code diffs |
| QA | `gpu-query/ai/tasks/spec/QA.md` | 53 acceptance criteria, 45 new tests, Criterion benchmark protocol, cache invariant strategy |
| Overview | `gpu-query/ai/tasks/spec/OVERVIEW.md` | Consolidated decisions, module roadmap, Q&A outcomes |

## Codebase Analysis

### Existing Patterns

- **QueryExecutor** (`src/gpu/executor.rs:254-265`): Owns `GpuDevice` + `PsoCache`, created via `::new()`. Two fields, clean struct -- easy to add `scan_cache: HashMap<String, ScanResult>`.
- **resolve_input** (`src/gpu/executor.rs:340-409`): Recursive plan-tree walker returning `(ScanResult, Option<FilterResult>)`. The double-scan bug is at lines 379-382 where both compound branches independently call `resolve_input`.
- **AppState** (`src/tui/app.rs:38-98`): Flat struct with 20 fields. No executor or cache fields. Straightforward to extend.
- **scan_directory** (`src/io/catalog.rs:37-85`): Pure function, returns `Vec<TableEntry>`. Called from 3 TUI sites (mod.rs:65, ui.rs:345, event.rs:215).
- **infer_schema_from_csv** (`src/gpu/executor.rs:3020-3101`): Opens file, reads 100 rows, type-votes. Pure function, deterministic for unchanged files.
- **build_csv_dictionaries** (`src/gpu/executor.rs:3109-3182`): Opens CSV again via BufReader, reads ALL rows on CPU for VARCHAR columns. Most expensive single CPU operation.

### Dependencies

- `objc2-metal` / `objc2` -- Metal bindings, `Retained<dyn MTLDevice>` is ARC refcounted, safe to hold long-term
- `ratatui` 0.29+ -- TUI framework, single-threaded event loop
- `criterion` -- existing benchmark infrastructure (49 benches)
- No async runtime (no tokio) -- synchronous execution model

### Constraints

- Rust borrow checker: `resolve_input(&mut self)` borrows executor mutably; returning `&ScanResult` from cache requires two-phase pattern (ensure_cached then get)
- Metal buffer lifetime: `bytesNoCopy` buffers reference mmap; cache entry must own both mmap and ColumnarBatch
- APFS mtime granularity: 1-second resolution; sub-second file modifications may miss invalidation (acceptable for interactive analytics)
- CLI stays per-invocation; only TUI gets persistent state

## Bottleneck Attribution

| # | Bottleneck | Location | Est. Cost | % of 367ms |
|---|-----------|----------|-----------|-----------|
| B1 | Double scan (GpuCompoundFilter) | executor.rs:379-395 | ~120-150ms | 33-41% |
| B2 | Executor recreated per query | ui.rs:375, event.rs:216 | ~10-25ms | 3-7% |
| B3 | CPU VARCHAR dictionary rebuild | executor.rs:3109-3182 | ~40-80ms | 11-22% |
| B4 | Schema re-inference every query | executor.rs:3020-3028 | ~2-5ms | 1% |
| B5 | Catalog re-scan every query | ui.rs:345, event.rs:215 | ~1-3ms | <1% |

Note: B1 amplifies B3 and B4 (each triggers twice). Fixing B1 alone removes the doubled cost.

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | **High** | All fixes are internal refactors; no API changes, no new dependencies |
| Effort Estimate | **M** | ~25 tasks across 5 phases; most changes in executor.rs + app.rs |
| Risk Level | **Low** | 494 lib tests + 155 GPU tests provide safety net; fixes are independent |
| Borrow Checker Risk | **Medium** | ScanCache requires two-phase lookup; TECH.md provides working pattern |
| Memory Risk | **Low** | ~62MB per cached 1M-row table, 8-entry cap = 496MB max; configurable |

## Performance Budget

| Scenario | Current | Target | Method |
|----------|---------|--------|--------|
| Cold first query | 367ms | ~195ms | Executor init + single scan (no double) |
| Warm repeat query | 367ms | 28-42ms | All caches hit, GPU kernels only |
| GPU kernel floor | ~25ms | ~25ms | Irreducible compute time |

## Recommendations

1. Fix B1 (double-scan) first -- biggest single win, removes multiplier on B3/B4
2. Use per-query `HashMap<String, ScanResult>` scan cache -- generalizes to triple-scans in nested compounds
3. Persistent `QueryExecutor` in `AppState` with lazy init -- amortizes Metal device + PSO across session
4. `CatalogCache` with `(size, mtime)` fingerprinting -- O(n) stat calls on hit, ~0.1ms for 10 files
5. FIFO eviction, 8-entry max, 512MB configurable ceiling for cross-query cache
