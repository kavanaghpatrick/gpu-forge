---
id: integration.BREAKDOWN
module: integration
priority: 999
status: pending
version: 1
origin: foreman-spec
dependsOn: [gsix-v2-format.BREAKDOWN, mmap-gpu-pipeline.BREAKDOWN, fsevents-watcher.BREAKDOWN, incremental-updates.BREAKDOWN, global-root-index.BREAKDOWN, testing-reliability.BREAKDOWN]
tags: [persistent-index, gpu-search, integration, final-assembly]
---
# Integration — BREAKDOWN

## Context
This is the final assembly module that validates the end-to-end pipeline: app startup loads (or builds) the global index via mmap, creates a Metal `bytesNoCopy` GPU buffer, starts the FSEvents watcher, processes incremental updates, and serves sub-100ms searches. It covers production migration from v1 to v2, cleanup of old index files, performance validation against all success metrics, and the complete UX lifecycle from first-run through steady-state operation.

References: PM.md Section 7 (Technical Architecture Summary: all data flows), UX.md Section 12 (Visual Summary: state transitions), QA.md Section 14 (Acceptance Criteria).

## Tasks
1. **integration.startup-warm-path** -- Validate the warm startup path end-to-end: app launch with existing `global.idx` v2 -> mmap file (<1ms) -> validate v2 header -> create `bytesNoCopy` Metal buffer (<1ms) -> start FSEvents with stored `sinceWhen` -> search ready in <100ms total. Measure with `Instant::elapsed()`. (est: 2h)

2. **integration.startup-cold-path** -- Validate the cold startup path end-to-end: app launch with no index -> spawn background `FilesystemScanner::scan("/")` -> progressive file count in status bar -> search available immediately via `walk_and_filter()` -> on scan completion, save v2 index -> mmap + `bytesNoCopy` -> FSEvents with `kFSEventStreamEventIdSinceNow` -> transition to indexed search. (est: 3h)

3. **integration.v1-migration-production** -- Test the v1-to-v2 migration in a realistic scenario: create a v1 index file with real entries, launch the app, verify v1 is detected and full rebuild triggered, verify v2 `global.idx` produced, verify per-directory v1 files cleaned up (`.v2-migrated` marker created), verify subsequent launches use v2 instantly. (est: 2h)

4. **integration.live-update-cycle** -- Validate the full update cycle: index loaded and READY -> create a file on disk -> FSEvents delivers event -> IndexWriter processes Created -> flush writes new `global.idx` -> IndexStore swaps snapshot -> next search finds the new file. Measure total latency from `fs::write()` to searchable: must be <1s. (est: 3h)

5. **integration.search-during-build** -- Validate concurrent search during initial build: start a search via `walk_and_filter()` while the background index build is running. Verify search returns valid results, build completes successfully, and subsequent searches use the index. No panics, no data races. (est: 2h)

6. **integration.search-during-update** -- Validate concurrent search during incremental update: dispatch a GPU search while the IndexWriter is flushing. Verify the search uses the old snapshot (not the in-progress new one). After flush + swap, verify the next search sees the updated entries. (est: 2h)

7. **integration.stale-index-recovery** -- Test stale index on startup: save an index with `saved_at` 2 hours ago, launch the app. Verify: status bar shows `(stale)`, FSEvents watcher starts, background update triggered, stale index serves fast searches immediately, after update completes status transitions to READY. (est: 2h)

8. **integration.corrupt-index-recovery** -- Test corrupt index on startup: overwrite first 4 bytes of `global.idx` with zeros. Launch the app. Verify: corruption detected, status bar shows BUILDING, full rebuild started, search works via walk fallback during rebuild, rebuild produces valid v2 index, status transitions to READY. (est: 2h)

9. **integration.error-state-recovery** -- Test disk-full scenario: make the index directory read-only. Trigger a flush. Verify: save fails, status bar shows ERROR, search falls back to walk, after restoring write permissions and next flush, index is saved successfully and status transitions to READY. (est: 2h)

10. **integration.old-index-cleanup** -- Validate cleanup of old per-directory v1 index files: create multiple `.idx` files with different hash names in `~/.gpu-search/index/`. After successful v2 build, verify all non-`global.idx` files are deleted and `.v2-migrated` marker exists. Verify cleanup runs only once (idempotent). (est: 1h)

11. **integration.performance-validation** -- Run comprehensive performance validation against all PM success metrics: cold start <100ms (index exists), warm start <5ms, incremental update <1s, mmap load <50ms at 1M entries, `bytesNoCopy` <1ms, initial scan <60s, memory overhead <50MB for mmap pages at 1M entries. Document results. (est: 3h)

12. **integration.ux-state-transitions** -- Validate the complete UX state machine from UX.md Section 12: first-launch (BUILDING with progress) -> search during build (walk fallback) -> build complete (READY) -> FSEvent (UPDATING) -> update complete (READY) -> subsequent launch (instant READY). Verify status bar text and colors at each state. (est: 2h)

13. **integration.backward-compatibility** -- Verify backward compatibility: existing blocking `search()` API works unchanged, `SearchUpdate` streaming protocol in `search_streaming()` works, `GpuPathEntry` layout matches Metal shader, all 404 existing tests pass, `walk_and_filter()` fallback works when index is absent. (est: 2h)

14. **integration.full-regression-suite** -- Run the complete test suite: `cargo test` (all unit + integration), `cargo test --test test_index_proptest` with `PROPTEST_CASES=5000`, `cargo test --ignored` (scale tests), `cargo bench` (performance benchmarks), `cargo clippy -D warnings`. Document any regressions or failures. (est: 2h)

## Dependencies
- Requires: [All 6 previous modules (gsix-v2-format, mmap-gpu-pipeline, fsevents-watcher, incremental-updates, global-root-index, testing-reliability)]
- Enables: [Feature is ready for merge to main after integration passes]

## Acceptance Criteria
1. Warm startup (index exists): search ready in <100ms from app launch
2. Cold startup (no index): search works immediately via walk; index built in background in <60s
3. Incremental update: file change reflected in search results within <1s
4. v1-to-v2 migration: automatic, one-time, with cleanup of old files
5. Corrupt/stale/error states: graceful recovery with walk fallback, no crashes
6. All 404 existing tests + ~146 new tests pass
7. Performance benchmarks meet all targets from PM.md Section 3
8. Status bar correctly shows all `IndexState` variants with proper colors
9. Search results from indexed path match non-indexed path (equivalence verified)
10. No regressions in the existing streaming search pipeline
11. `cargo clippy -D warnings` clean; no new `unsafe` without `// SAFETY:` comments
12. CI pipeline green across all test categories

## Technical References
- PM: ai/tasks/spec/PM.md -- Section 3 (Success Metrics), Section 7 (Technical Architecture: all data flow diagrams)
- UX: ai/tasks/spec/UX.md -- Section 12 (Visual Summary: state transitions), Section 9 (Progressive Enhancement), Section 7 (Error States)
- Tech: ai/tasks/spec/TECH.md -- Section 1 (Architecture: component diagram, data flows), Section 10 (Migration), Appendix B (Source File Changes)
- QA: ai/tasks/spec/QA.md -- Section 14 (Acceptance Criteria: all 14 items), Section 12 (Test Execution Plan)
