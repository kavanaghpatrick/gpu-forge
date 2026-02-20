---
id: tui.BREAKDOWN
module: tui
priority: 3
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN, gpu-engine.BREAKDOWN]
tags: [gpu-autonomous]
testRequirements:
  unit:
    required: false
---
# TUI Module Breakdown

## Context

Sub-millisecond query execution transforms the TUI interaction model from "write-then-run" to "live data instrument." At 36ms, querying is an action (press F5). At <1ms, querying becomes perception -- results appear as fast as the user types. The TUI must evolve to support live mode (auto-execute on every valid keystroke), engine status display, warm-up progress, fallback UX, and non-blocking result polling.

The existing TUI is built around a synchronous execute-then-display cycle: F5 triggers `execute_editor_query()` which blocks on `waitUntilCompleted` then calls `set_result()`. The autonomous engine eliminates this cycle -- the event loop polls `ready_flag` on every frame tick and updates results when available.

## Acceptance Criteria

1. **AppState extensions**: New fields -- `autonomous_executor: Option<AutonomousExecutor>`, `live_mode: bool`, `engine_status: EngineStatus`, `warmup_progress: f32`, `last_autonomous_us: Option<u64>`, `autonomous_stats: AutonomousStats`, `sql_validity: SqlValidity`, `query_compatibility: QueryCompatibility`, `cached_plan: Option<PhysicalPlan>`
2. **EngineStatus enum**: Off, WarmingUp{table, progress_pct}, Compiling{plan_hash}, Live, Idle, Error(String) -- displayed in dashboard panel
3. **Live mode**: ON by default when engine ready (UX-Q1). Every keystroke triggers `update_sql_validity()` and, if SQL is valid + autonomous-compatible, immediate `submit_query()` (0ms debounce per UX-Q2). Toggle with Ctrl+L.
4. **Event loop changes**: Poll `autonomous_executor.poll_ready()` on every frame tick. Check warm-up progress via channel. Submit autonomous query when SQL validity changes to Valid + Autonomous.
5. **Engine status badge**: Rendered in dashboard panel (`dashboard.rs`) with states [OFF]/[WARMING]/[COMPILING]/[LIVE]/[IDLE]/[FALLBACK]/[ERROR] with color coding per UX.md Section 4.2
6. **Fallback UX**: When query uses ORDER BY or other unsupported patterns, badge shows [FALLBACK] with reason. Performance line shows "36.2ms | ORDER BY requires standard path" (UX-Q4).
7. **Results panel**: `[auto]` tag in title for autonomous results. Microsecond precision timing. Performance line format: `8 rows | 1M scanned | 0.42ms | autonomous | ~86x vs standard path`
8. **Warm-up flow**: TUI launches instantly. Background thread loads binary columnar data. Dashboard shows progress bar. Manual F5 queries use standard 36ms path during warm-up. Auto-transition to live mode when complete.
9. **SQL validity checking**: `update_sql_validity()` called on each editor keystroke in live mode. Attempts `parse_query()` + `plan()`. Updates `SqlValidity` (Empty/Incomplete/ParseError/Valid) and `QueryCompatibility` (Unknown/Autonomous/Fallback(reason)/Invalid).
10. **Keybindings**: Ctrl+L toggles live mode. F5 forces manual execution (bypasses live mode). Status bar updated to show live mode state.
11. **Rolling stats**: `AutonomousStats` struct tracks total_queries, fallback_queries, avg_latency_us, p99_latency_us, consecutive_sub_1ms.
12. ~15 unit tests for TUI state (EngineStatus default, SqlValidity transitions, QueryCompatibility checks, AutonomousStats recording)

## Technical Notes

- **Event loop polling**: The autonomous engine poll happens BEFORE input handling in the event loop. Since GPU responds in <1ms and TUI renders at 60fps (16ms), worst-case result display delay is one frame tick.
- **0ms debounce**: User decided every keystroke fires immediately (UX-Q2). This means on every editor keystroke: parse SQL -> if valid, check compatibility -> if autonomous, submit to work queue. CPU cost is ~0.3ms per keystroke (parse + plan).
- **Warm-up channel**: Background loading thread sends `(table_name, progress_f32)` via `std::sync::mpsc::channel`. TUI event loop calls `warmup_rx.try_recv()` on each tick.
- **Result freshness**: FRESH (results match current SQL), STALE (new query submitted, waiting), INVALID (SQL not parseable). Stale results rendered with dimmed theme color.
- **Dual executor**: AppState holds both `autonomous_executor: Option<AutonomousExecutor>` and existing `executor: Option<QueryExecutor>`. Live mode uses autonomous; F5 or fallback uses standard.
- **Existing patterns**: Follow `pub` field pattern on AppState (app.rs). Extend `QueryState` enum with `AutonomousSubmitted`. Modify `render_gpu_dashboard()` in dashboard.rs. Extend `build_performance_line()` in results.rs.
- Reference: UX.md Sections 3-10, TECH.md Section 9, QA.md Section 4.6
