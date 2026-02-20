---
id: global-root-index.BREAKDOWN
module: global-root-index
priority: 4
status: pending
version: 1
origin: foreman-spec
dependsOn: [gsix-v2-format.BREAKDOWN, mmap-gpu-pipeline.BREAKDOWN, fsevents-watcher.BREAKDOWN, incremental-updates.BREAKDOWN]
tags: [persistent-index, gpu-search, root-index, ux]
---
# Global Root Index — BREAKDOWN

## Context
The current architecture uses per-directory indexes that must be built on first access to each new root. This module switches to a single global index file (`~/.gpu-search/index/global.idx`) covering `/` with an expanded exclude list. It implements background initial build with progress reporting, integrates the index into the search orchestrator (preferring index over walk), and wires the `IndexState` enum into the status bar for at-a-glance index health.

References: TECH.md Section 5 (Global Root Index), PM.md Section 4 (Background initial scan, expanded exclude list), UX.md Sections 3-6 (Status indicators, first-run, startup, background updates).

## Tasks
1. **global-root-index.global-path** -- Implement `global_index_path() -> PathBuf` returning `~/.gpu-search/index/global.idx` and `global_cache_key() -> u32` returning the hash of `"/"`. Ensure the index directory is created on first use (`create_dir_all`). (est: 1h)

2. **global-root-index.expanded-excludes** -- Expand `DEFAULT_EXCLUDES` with system-root paths: Tier 1 absolute (`/System`, `/Library/Caches`, `/private/var`, `/private/tmp`, `/Volumes`, `/dev`, `/cores`), Tier 2 user-relative (`~/Library/Caches`, `~/.Trash`), Tier 3 basename (`.git`, `node_modules`, `target`, `__pycache__`, `vendor`, `dist`, `build`, `.cache`, `venv`, `.venv`, `.Spotlight-V100`, `.fseventsd`, `.DS_Store`, `.Trashes`, `.hg`, `.svn`, `.idea`, `.vscode`). Build the `ExcludeTrie` from these defaults. (est: 1h)

3. **global-root-index.config-file** -- Implement optional config loading from `~/.gpu-search/config.json`. Parse `exclude_dirs`, `exclude_absolute`, and `include_override` arrays. Merge with hardcoded defaults. If no config file exists, use defaults only. (est: 2h)

4. **global-root-index.background-initial-build** -- On first launch (no `global.idx`), spawn a background thread running `FilesystemScanner::scan("/")` with the expanded exclude list. Use an `AtomicUsize` progress counter incremented per file scanned. On completion, save the v2 index and create the initial `IndexSnapshot` in `IndexStore`. (est: 3h)

5. **global-root-index.progress-counter** -- Pass the `Arc<AtomicUsize>` progress counter to the scanner thread. The scanner increments it per discovered file. The UI reads it every frame during the BUILDING state to display `Indexing: 420K files`. (est: 1h)

6. **global-root-index.index-state-enum** -- Define `IndexState` enum in `src/ui/status_bar.rs` (or a shared types module): `Loading`, `Building { files_indexed: usize }`, `Ready { file_count: usize }`, `Updating`, `Stale { file_count: usize }`, `Error { message: String }`. Add `index_state: IndexState` field to `GpuSearchApp`. (est: 1h)

7. **global-root-index.status-bar-render** -- Update `StatusBar::render()` to prepend the index status as the first segment. Color-code by state: SUCCESS (#9ECE6A) for Ready, ACCENT (#E0AF68) for Building/Updating/Stale, ERROR (#F7768E) for Error. Format file count with K/M abbreviation (exact if <1K, K if 1K-999K, M if >=1M). (est: 2h)

8. **global-root-index.status-bar-transitions** -- Wire index state transitions into `app.rs`: set `Building` when background scan starts, update `files_indexed` from atomic counter each frame, transition to `Ready` when scan completes, set `Updating` when FSEvents trigger flush, set `Stale` when `is_stale()` returns true on startup, set `Error` on load/save failures. (est: 2h)

9. **global-root-index.request-repaint** -- Call `ctx.request_repaint()` during the `Building` state to keep the progress counter updating. Condition: `is_searching || matches!(index_state, IndexState::Building { .. })`. (est: 1h)

10. **global-root-index.orchestrator-integration** -- Update `SearchOrchestrator` to prefer the index over filesystem walk. On search: check `IndexStore::snapshot()`. If available and not stale, use the Metal buffer for GPU dispatch. If unavailable, fall back to `walk_and_filter()`. The existing streaming pipeline and blocking `search()` API must both work. (est: 3h)

11. **global-root-index.stale-detection** -- On launch, check `saved_at` against `DEFAULT_MAX_AGE`. If stale, show `Index: 1.3M files (stale)` in status bar, start FSEvents watcher which triggers background re-scan. Search uses the stale index (fast but possibly incomplete) while update runs. (est: 1h)

12. **global-root-index.index-daemon** -- Implement `IndexDaemon` in `src/index/daemon.rs` that orchestrates the full lifecycle: load or build index on startup, start FSEventsListener, start IndexWriter thread, handle shutdown. This is the top-level coordination point. (est: 3h)

13. **global-root-index.status-bar-tests** -- Write unit tests: status bar renders each `IndexState` variant correctly, file count formatting (exact/K/M), color mapping per state, segment priority on narrow displays (index status never hidden). (~6 tests) (est: 2h)

14. **global-root-index.integration-test** -- Write integration test: simulate first-run (no index file), verify background build starts, verify progress counter increments, verify index file created on completion, verify subsequent load is instant (<100ms). (est: 2h)

## Dependencies
- Requires: [gsix-v2-format (v2 format for save/load), mmap-gpu-pipeline (IndexStore + IndexSnapshot for orchestrator), fsevents-watcher (live update source), incremental-updates (IndexWriter for processing changes)]
- Enables: [integration (end-to-end pipeline ready), testing-reliability (all subsystems wired together for system-level tests)]

## Acceptance Criteria
1. Single `global.idx` file at `~/.gpu-search/index/` covers the entire boot volume (`/`)
2. Background initial build shows progressive file count in status bar without blocking search
3. `SearchOrchestrator` uses the index when available, falls back to walk when not
4. Status bar shows correct state (BUILDING/READY/UPDATING/STALE/ERROR) with appropriate colors
5. File count formatted with K/M abbreviation for compact display
6. Stale index detected on startup; triggers background update while serving stale results
7. All existing search tests continue to pass (blocking `search()` and streaming `search_streaming()`)
8. First-run experience: search works immediately via walk fallback during background build

## Technical References
- PM: ai/tasks/spec/PM.md -- Section 4 (Background initial scan, expanded exclude list), Phase 1 (Persistent Index), Section 7 (Data Flow)
- UX: ai/tasks/spec/UX.md -- Section 3 (Index Status Indicators), Section 4 (First-Run Experience), Section 5 (Background Update UX), Section 6 (Startup Experience), Section 8 (Status Bar Integration), Section 10 (First-Run Progress Detail)
- Tech: ai/tasks/spec/TECH.md -- Section 5 (Global Root Index: single index, rationale, first-run), Section 9 (Exclude List: tiers, runtime configuration)
- QA: ai/tasks/spec/QA.md -- Section 8 (Regression Tests: walk_and_filter fallback, search equivalence)
