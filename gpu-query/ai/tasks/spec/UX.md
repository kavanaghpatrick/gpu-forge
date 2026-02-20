# UX/UI Analysis: gpu-query TUI Performance Optimization

**Agent**: agent-foreman:ux (UX/UI Designer)
**Date**: 2026-02-11
**Scope**: 5 critical performance bottlenecks affecting perceived latency in gpu-query TUI dashboard
**Target**: Reduce 1M-row query from 367ms to <50ms perceived response time

---

## 1. Research Foundation

### 1.1 Nielsen's Response Time Thresholds (Still the Gold Standard)

Jakob Nielsen's three response-time limits define the UX impact of each bottleneck:

| Threshold | User Perception | gpu-query Status |
|-----------|----------------|-----------------|
| **<100ms** | "Instantaneous" -- direct manipulation feel | **Target for warm queries** |
| **100ms-1s** | Noticeable delay, flow of thought preserved | Current 367ms sits here |
| **>1s** | Attention drift, progress indicator required | First-query cold start may hit this |

**Key insight**: A 2017 empirical study found the actual mean perceptual threshold is ~65ms (range: 34-137ms), not 100ms. This means our <50ms target is well-chosen -- it places gpu-query below the perceptual floor for most users.

### 1.2 Competitive Landscape: How Database CLIs Handle Latency

| Tool | Latency UX Pattern | What We Learn |
|------|-------------------|---------------|
| **DuckDB CLI** | Shows elapsed time after result; "Instant SQL" mode in UI auto-runs as you type | Users compare gpu-query directly to DuckDB's ~30-80ms; must match or beat |
| **pgcli** | Spinner during query execution; auto-completion provides "working" feedback | The feedback loop paper shows that perceived speed improves with responsiveness |
| **litecli** | Same pattern as pgcli; immediate prompt return | Lightweight CLI tools set sub-100ms as baseline expectation |
| **DuckDB Local UI** | Real-time result streaming as query executes | Sets new bar: query results should appear progressively |

### 1.3 Ratatui Async Patterns

The ratatui ecosystem provides established patterns for non-blocking query execution:
- `tokio::spawn` for background task execution with channel-based result delivery
- `select!` macro for concurrent event handling (user input + query completion)
- Render ticks continue during background work (animation stays smooth at 60fps)

The current gpu-query TUI executes queries **synchronously** in the event handler, which blocks the entire render loop during execution.

---

## 2. Bottleneck-by-Bottleneck UX Impact Analysis

### 2.1 Bottleneck #1: Double CSV Scan on Compound Filters

**Technical**: `WHERE a > X AND a < Y` scans the CSV twice -- once per predicate.

**UX Impact**: HIGH
- On a 1M-row CSV, each full scan is ~50-80ms of I/O-bound work
- Two scans = 100-160ms of pure waste
- This single bottleneck accounts for ~40% of the 367ms total
- Users writing range queries (the most common compound filter pattern) experience the worst performance -- exactly the users who should feel the product is fast

**Perceived Behavior**: User presses F5, sees "Executing..." for noticeably longer on range filters than on single-predicate filters. This inconsistency erodes confidence.

**UX Recommendation**: Beyond the engineering fix (fused scan), the status bar should show scan progress for large files: `Scanning sales.csv... 650K/1M rows`. Even if the fused scan completes in <50ms, the progress infrastructure is needed for files larger than the current test set.

### 2.2 Bottleneck #2: QueryExecutor Recreated Per Query (Cold Start)

**Technical**: `QueryExecutor::new()` in `execute_editor_query()` calls `GpuDevice::new()` (Metal device init + metallib load) and `PsoCache::new()` (empty) on every query execution.

**UX Impact**: CRITICAL for first-query experience, HIGH for all subsequent queries
- Metal device initialization: ~2-5ms (OS-level device enumeration)
- Metallib loading from disk: ~5-15ms (file I/O + Metal compiler front-end)
- PSO cache starts empty every time: first-time kernel compilation is ~10-30ms per PSO
- **Total cold-start tax per query: ~17-50ms** -- this alone could consume our entire <50ms budget

**Perceived Behavior**: Every single query feels like the "first query." Users never experience the satisfaction of warm-cache performance. The sparkline history in the GPU dashboard shows flat, consistently slow performance instead of the dramatic warm-up curve users expect from GPU-accelerated tools.

**UX Recommendation**:
1. **Warm executor on TUI startup**: Initialize `QueryExecutor` once and store in `AppState`. Show a brief "Initializing GPU..." splash during the 20-50ms startup cost.
2. **Surface warm/cold in status**: After fixing, show `"4 rows | 2.1ms (warm)"` to teach users the system gets faster. The current `is_warm` field in `QueryMetrics` already tracks this -- it just needs to flow to the status bar.
3. **Pre-warm common PSOs**: On startup, compile the 4-5 most common kernel variants (csv_parse, column_filter with INT64/FLOAT64, aggregate_count, aggregate_sum). Cost is ~50-100ms at startup but makes every subsequent query feel instant.

### 2.3 Bottleneck #3: CPU-Side Dictionary Building

**Technical**: `Dictionary::build()` collects unique strings into a HashSet, sorts them, and builds a reverse index HashMap -- all on CPU, inline with query execution.

**UX Impact**: MEDIUM (data-dependent)
- For low-cardinality columns (region, status): <1ms -- negligible
- For high-cardinality string columns (names, IDs with 10K+ distinct): can take 5-30ms
- The 10K threshold check means large string columns silently fall back to no encoding, but the user has no visibility into why performance changed

**Perceived Behavior**: Queries against string columns have unpredictable latency. The same `WHERE region = 'US'` pattern works fast on one table but slow on another, with no visible explanation.

**UX Recommendation**:
1. **Cache dictionaries per table+column**: Once built, store in the executor or a separate `DictionaryCache`. Dictionary changes only when the underlying file changes (detectable via file mtime).
2. **Show dictionary status in catalog panel**: Next to each string column, show `[dict: 42 values]` or `[raw: 15K+ distinct]` so users understand the encoding strategy before querying.
3. **Build dictionaries during catalog scan**: When the TUI first loads and scans the data directory, pre-build dictionaries for all string columns in the background. By the time the user types their first query, dictionaries are ready.

### 2.4 Bottleneck #4: Schema Inference Re-runs Every Query

**Technical**: `infer_schema_from_csv()` reads up to `SCHEMA_INFER_SAMPLE_ROWS` lines from the CSV, parsing each field to vote on column types (int vs float vs varchar). This runs on every query execution.

**UX Impact**: MEDIUM
- For typical CSVs (100 sample rows): ~2-5ms of redundant I/O and parsing
- For wide tables (20+ columns): up to 10ms per inference run
- Schema never changes between queries against the same file, so 100% of this work is wasted after the first query

**Perceived Behavior**: A constant 2-10ms tax on every query that provides zero user value. Users cannot perceive this individually, but it contributes to the cumulative "death by a thousand cuts" that makes 367ms instead of 50ms.

**UX Recommendation**:
1. **Cache schema per file path + mtime**: Store inferred schemas in a `HashMap<(PathBuf, SystemTime), RuntimeSchema>` on the executor or `AppState`. Invalidate only when file modification time changes.
2. **Show schema in catalog panel**: The catalog tree already shows column names -- enrich it with inferred types: `amount (FLOAT64)`, `region (VARCHAR)`. This surfaces the schema work to the user as a visible benefit, not hidden cost.
3. **Schema indicator in status bar**: After first query, show schema status: `Schema cached for 3 tables`. This builds user confidence that the system is learning their data.

### 2.5 Bottleneck #5: Catalog Re-scans Filesystem Every Query

**Technical**: `scan_directory()` in `execute_editor_query()` calls `std::fs::read_dir()`, iterates all files, detects formats, and parses CSV headers on every query execution.

**UX Impact**: LOW-MEDIUM (but constant)
- For small data directories (5-10 files): ~1-3ms
- For larger directories (50+ files): up to 10-20ms
- Combined with schema inference (#4), the catalog+schema overhead is 5-25ms per query -- meaningful against a 50ms budget

**Perceived Behavior**: The catalog panel in the TUI already shows the scanned tables from startup. Yet the system re-scans on every query, doing work the user can literally see was already done.

**UX Recommendation**:
1. **Scan once at startup, cache until invalidated**: The TUI's `run_dashboard()` already does an initial scan at line 65. Reuse that result for queries instead of re-scanning.
2. **Manual refresh command**: Add `.refresh` or `Ctrl+R` to re-scan the data directory on demand. Show a brief `"Refreshing catalog..."` status message.
3. **File watcher (future)**: Use `notify` crate to watch the data directory for changes and auto-refresh the catalog. Show a subtle indicator when new files are detected: `"New file detected: orders.csv -- press Ctrl+R to refresh"`.

---

## 3. Cumulative UX Impact Map

```
Current query execution timeline (1M rows, compound filter):

 |-- Catalog scan: 3ms --|-- Schema infer: 5ms --|-- SQL parse: <1ms --|
 |-- Dict build: 8ms --|-- CSV scan #1: 80ms --|-- CSV scan #2: 80ms --|
 |-- GPU filter: 2ms --|-- GPU aggregate: 1ms --|-- Result format: 3ms --|
 |========================== TOTAL: ~183ms (scan only) ==================|
 |+ Executor cold start: ~40ms                                          |
 |+ PSO compilation: ~30ms (first unique kernel)                        |
 |============================== WALL: ~250-370ms ======================|

Target query execution timeline (after all fixes):

 |-- (cached) --|-- (cached) --|-- SQL parse: <1ms --|
 |-- (cached) --|-- Fused CSV scan: 40ms --|
 |-- GPU filter: 2ms --|-- GPU aggregate: 1ms --|-- Result format: 3ms --|
 |========================== TOTAL: ~47ms ==============================|
 |+ Executor warm start: 0ms (persistent)                              |
 |+ PSO lookup: <0.1ms (cached)                                        |
 |============================== WALL: ~47ms ===========================|
```

**Projected speedup: 367ms -> ~47ms (7.8x improvement)**

The critical insight is that the **perceptual impact is nonlinear**: going from 367ms (noticeable delay, flow interrupted) to 47ms (below perceptual threshold) crosses Nielsen's 100ms boundary. Users will experience this not as "somewhat faster" but as a qualitative shift from "waits for query" to "instant response."

---

## 4. UX Improvement Recommendations

### 4.1 Async Query Execution (Priority: P0)

**Problem**: The current `execute_editor_query()` is synchronous and blocks the render loop.

**Current behavior (from `event.rs` line 76)**:
```rust
let _ = super::ui::execute_editor_query(app);
```

This call blocks the event loop. During execution, the TUI cannot render frames, respond to Ctrl+C, or show progress. The "Executing..." animation (line 298 in results.rs) never actually animates because frames are not being drawn.

**Recommendation**: Move query execution to a background `tokio::spawn` task:

1. Add `QueryState::Running` transition immediately when F5/Ctrl+Enter is pressed (already happens at line 341 of ui.rs)
2. Spawn the actual execution on a background thread/task
3. Communicate results back via a `tokio::sync::oneshot` or `std::sync::mpsc` channel
4. The render loop continues drawing the "Executing..." animation smoothly
5. On completion, transition to `QueryState::Complete` and update results

**UX benefit**: Even if query execution still takes 367ms, the TUI remains responsive. The animated dots in the results panel will actually animate. The user can press Ctrl+C to cancel. This is the single highest-impact UX change independent of performance fixes.

### 4.2 Persistent QueryExecutor in AppState (Priority: P0)

**Problem**: `QueryExecutor` is created fresh in `execute_editor_query()` every time.

**Recommendation**: Add `executor: Option<QueryExecutor>` to `AppState`. Initialize it once during `run_dashboard()` startup. This eliminates the cold-start tax entirely.

**Status bar integration**:
- During startup: `"Initializing GPU pipeline... (one-time)"`
- After first query: `"4 rows | 12.3ms (cold) | PSO cache: 3 kernels"`
- After subsequent queries: `"4 rows | 2.1ms (warm) | PSO cache: 5 kernels"`

This surfaces the performance improvement arc to users, building confidence in the GPU-accelerated approach.

### 4.3 Cached Catalog + Schema in AppState (Priority: P1)

**Problem**: Both catalog scan and schema inference repeat on every query.

**Recommendation**: Add to `AppState`:
```
cached_catalog: Option<Vec<TableEntry>>
cached_schemas: HashMap<String, RuntimeSchema>
catalog_mtime: Option<SystemTime>
```

On query execution, check if the data directory's modification time changed. If not, reuse cached catalog and schemas. If changed, re-scan and update the catalog panel simultaneously.

**Catalog panel enhancement**: Show cache status with a small indicator:
- Fresh: green dot or checkmark next to directory name
- Stale (file changed since last scan): yellow dot, `"[stale -- Ctrl+R to refresh]"`

### 4.4 Timing Breakdown in Status Bar (Priority: P1)

**Problem**: The status bar shows only total time: `"4 rows | 367.6ms"`. Users cannot tell if the bottleneck is I/O, GPU, or overhead.

**Recommendation**: Expand the timing display in stages:

**Default mode** (compact):
```
4 rows | 47.2ms (warm) | GPU 94% | ~312x vs CPU
```

**Profile mode** (`.profile on`, already exists):
```
4 rows | 47.2ms | scan: 40.1ms | filter: 2.3ms | agg: 0.8ms | overhead: 4.0ms
```

**Speedup display** (after optimization):
When the system detects a query is running faster than it would have with the old path, briefly flash a comparison:
```
4 rows | 47.2ms (was: ~367ms before caching) | 7.8x faster
```

This could be shown only for the first few queries after optimization, as a one-time educational moment.

### 4.5 First-Query vs. Subsequent-Query Experience (Priority: P1)

The first query is psychologically the most important -- it sets the user's mental model for the tool's performance.

**Current first-query experience**:
1. User types SQL (good -- editor is responsive)
2. User presses F5 (screen freezes for ~400ms)
3. Results appear with `"4 rows | 367.6ms"`
4. User thinks: "This is slower than DuckDB. Why use GPU?"

**Target first-query experience**:
1. User types SQL (editor is responsive + autocomplete suggests table names from cached catalog)
2. User presses F5
3. Results panel immediately shows `"Executing..."` with animated spinner (async)
4. Within ~50ms (below perception), results appear
5. Status bar shows: `"4 rows | 47.2ms (cold) | ~8x vs CPU est."`
6. User thinks: "Fast, and it says this is the cold run? It gets even faster?"

**Target subsequent-query experience**:
1. User modifies SQL and presses F5
2. Results appear before the user's finger lifts from the key
3. Status bar: `"4 rows | 12.3ms (warm) | ~26x vs CPU est. | PSO cache hit"`
4. GPU dashboard sparkline shows the performance improvement visually

### 4.6 Visual Progress for Large Scans (Priority: P2)

For queries that scan truly large datasets (10M+ rows, multi-GB files), even optimized execution may take >100ms. In these cases:

**Recommendation**: Add a progress bar to the results panel during execution:
```
 Results
 Scanning sales_10m.csv... [=========>        ] 45%  |  4.5M/10M rows
```

This requires chunked execution reporting (the batch system in `executor.rs` already processes in chunks via `BATCH_SIZE_BYTES`). After each chunk completes, send a progress update through the async channel.

### 4.7 Query Cancel Support (Priority: P2)

**Problem**: During synchronous execution, there is no way to cancel a running query. The TUI is completely frozen.

**Recommendation**: With async execution in place, add Ctrl+C (in-query, not global quit) or Esc to cancel:
- Set an `AtomicBool` cancellation flag that the executor checks between batches
- Show `"Query cancelled after 2.3s (processed 5M of 10M rows)"`
- This is especially important for accidental `SELECT * FROM huge_table` without LIMIT

### 4.8 Warm-Up Indicator During TUI Startup (Priority: P2)

**Recommendation**: During the 50-100ms startup window (executor init + catalog scan + PSO pre-warm), show a brief splash state:

```
 gpu-query  GPU-Native Data Analytics

 Initializing GPU pipeline...  [done]
 Scanning data directory...    [3 tables found]
 Pre-warming kernel cache...   [5 PSOs compiled]

 Ready. Type SQL and press F5 to execute.
```

This converts invisible startup cost into visible progress, setting user expectations correctly.

---

## 5. Priority Matrix

| # | Recommendation | UX Impact | Eng Effort | Priority |
|---|---------------|-----------|------------|----------|
| 4.1 | Async query execution | Critical | Medium | **P0** |
| 4.2 | Persistent QueryExecutor | Critical | Low | **P0** |
| 4.3 | Cached catalog + schema | High | Low | **P1** |
| 4.4 | Timing breakdown display | High | Low | **P1** |
| 4.5 | First-query experience design | High | Low | **P1** |
| 4.6 | Progress bar for large scans | Medium | Medium | **P2** |
| 4.7 | Query cancel (Ctrl+C) | Medium | Medium | **P2** |
| 4.8 | Startup warm-up indicator | Low | Low | **P2** |

---

## 6. Success Metrics

### Quantitative
- **P50 query latency (warm)**: <20ms (from ~367ms)
- **P50 query latency (cold/first)**: <50ms (from ~400ms+)
- **TUI frame rate during query**: 60fps constant (from 0fps during execution)
- **Time to first interactive result**: <100ms from F5 press

### Qualitative
- User perceives queries as "instant" (below 65ms perceptual threshold)
- GPU dashboard sparkline shows visible warm-up improvement curve
- Status bar communicates performance story (cold/warm, GPU vs CPU comparison)
- First-query experience sets positive expectations rather than negative comparison to DuckDB

---

## 7. Implementation Notes for Engineers

### Changes to `AppState` (app.rs)
New fields needed:
- `executor: Option<QueryExecutor>` -- persistent GPU executor
- `cached_catalog: Option<Vec<TableEntry>>` -- cached directory scan
- `cached_schemas: HashMap<String, RuntimeSchema>` -- cached inferred schemas
- `dict_cache: HashMap<(String, String), Dictionary>` -- per-table+column dictionary cache
- `query_cancel: Arc<AtomicBool>` -- cancellation flag for async execution
- `query_result_rx: Option<Receiver<QueryResult>>` -- async result channel

### Changes to `execute_editor_query()` (ui.rs)
The function should become a "dispatch" that:
1. Validates SQL (parse only -- fast)
2. Sends work to a background thread via channel
3. Returns immediately, leaving `query_state = Running`
4. The event loop checks for completed results on each tick

### Changes to `run_dashboard()` (mod.rs)
In the main event loop, add a check on each tick:
```
AppEvent::Tick => {
    app.tick();
    // Check for async query completion
    if let Some(rx) = &app.query_result_rx {
        if let Ok(result) = rx.try_recv() {
            app.set_result(result);
        }
    }
}
```

### Status Bar Enhancement (ui.rs)
The `render_status_bar()` function should conditionally show the timing breakdown, warm/cold indicator, and cache statistics. The existing `last_query_metrics` field already provides `is_warm` -- this just needs to flow to the status text.

---

## 8. Wire-Frame: Enhanced Status Bar

```
Before (current):
+--------------------------------------------------------------------+
| Query completed: 4 rows in 367.6ms | Editor | Tab: cycle | ...    |
+--------------------------------------------------------------------+

After (warm query, default mode):
+--------------------------------------------------------------------+
| 4 rows | 12.3ms (warm) | GPU 94% | ~26x vs CPU | Editor | ...    |
+--------------------------------------------------------------------+

After (cold query, first run):
+--------------------------------------------------------------------+
| 4 rows | 47.2ms (cold) | GPU 68% | ~8x vs CPU | Editor | ...     |
+--------------------------------------------------------------------+

After (profile mode on):
+--------------------------------------------------------------------+
| 4 rows | 47ms | scan:40ms flt:2ms agg:1ms oh:4ms | Editor | ...   |
+--------------------------------------------------------------------+

During execution (async):
+--------------------------------------------------------------------+
| Executing... sales.csv 650K/1M rows | Editor | Ctrl+C: cancel     |
+--------------------------------------------------------------------+
```

---

## 9. Interaction Flow: Optimized Query Lifecycle

```
                User presses F5
                       |
                       v
         +----- Parse SQL (sync, <1ms) -----+
         |                                    |
         |  Fail: show error immediately      |
         |  Pass: v                           |
         |                                    |
         +--- Set QueryState::Running --------+
         |  (render loop continues 60fps)     |
         |                                    |
         +--- Spawn background task ----------+
              |  1. Use cached catalog        |
              |  2. Use cached schema         |
              |  3. Use cached dictionaries   |
              |  4. Use persistent executor   |
              |  5. Execute fused scan+filter |
              |  6. Send result via channel   |
              v                               |
         Render loop ticks                    |
         - Animated "Executing..." dots       |
         - GPU dashboard still updates        |
         - User can type / cancel             |
              |                               |
              v                               |
         Result arrives on channel            |
         - Set QueryState::Complete           |
         - Update results panel               |
         - Update status bar with timing      |
         - Record metrics for sparkline       |
         - Flash speedup if notably faster    |
```

---

## 10. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Persistent executor holds Metal resources indefinitely | Low | Metal device + PSO memory is minimal (~2MB). Acceptable for TUI lifetime. |
| Cached schema becomes stale if file edited externally | Medium | Check file mtime before each query; invalidate on change. |
| Async execution complicates error handling | Medium | Use `Result<QueryResult, String>` through the channel; handle errors on receive. |
| Users expect sub-ms after seeing "warm" label | Low | Set expectations with "(warm)" label; document that GPU overhead has a floor. |
| Large file progress reporting adds complexity | Medium | Implement as P2; the P0/P1 fixes deliver the core <50ms target without progress. |
