---
id: fsevents-watcher.BREAKDOWN
module: fsevents-watcher
priority: 2
status: pending
version: 1
origin: foreman-spec
dependsOn: [gsix-v2-format.BREAKDOWN]
tags: [persistent-index, gpu-search, fsevents, macos]
---
# FSEvents Watcher — BREAKDOWN

## Context
The current `IndexWatcher` uses the `notify` crate with kqueue backend, which hits the 256 file descriptor limit when watching `/` recursively. The `notify` crate also abstracts away FSEvents event IDs, making it impossible to persist event IDs for cross-restart resume. This module replaces the notify-based watcher with direct macOS FSEvents integration via `fsevent-sys`, implementing event ID persistence, resume-from-stored-ID on startup, and efficient path filtering via an `ExcludeTrie`.

References: TECH.md Section 3 (FSEvents Integration), Section 9 (Exclude List), PM.md Section 4 (FSEvents watcher migration).

## Tasks
1. **fsevents-watcher.fsevent-sys-dependency** -- Add `fsevent-sys = "4"` and `core-foundation = "0.10"` to `Cargo.toml`. Verify the crate compiles on macOS ARM64 and provides access to `FSEventStreamCreate`, `FSEventStreamGetLatestEventId`, and relevant flag constants. (est: 1h)

2. **fsevents-watcher.fschange-enum** -- Define `FsChange` enum in a new `src/index/fsevents.rs` module: `Created(PathBuf)`, `Modified(PathBuf)`, `Deleted(PathBuf)`, `Renamed { old: PathBuf, new: PathBuf }`, `MustRescan(PathBuf)`, `HistoryDone`. Register module in `src/index/mod.rs`. (est: 1h)

3. **fsevents-watcher.exclude-trie** -- Implement `ExcludeTrie` in a new `src/index/exclude.rs`: absolute path prefix matching (Tier 1: `/System`, `/Volumes`, etc.), directory basename matching via `HashSet` (Tier 3: `.git`, `node_modules`, `target`, etc.), user-relative prefixes (Tier 2: `~/Library/Caches`, `~/.Trash`), and include overrides. Implement `should_exclude(&self, path: &[u8]) -> bool`. (est: 2h)

4. **fsevents-watcher.exclude-hash** -- Implement `compute_exclude_hash(excludes: &ExcludeTrie) -> u32` using `crc32fast`. Sort all patterns, hash with NUL separators. This hash is stored in the GSIX v2 header; a mismatch on startup triggers full rebuild. Add `crc32fast = "1"` to Cargo.toml if not already present. (est: 1h)

5. **fsevents-watcher.fsevents-listener** -- Implement `FSEventsListener` struct: holds a CFRunLoop thread handle, crossbeam sender for `FsChange` events, `Arc<AtomicU64>` for `last_event_id`, `Arc<ExcludeTrie>`, and shutdown `Arc<AtomicBool>`. Implement `new()` and `start()` methods. (est: 3h)

6. **fsevents-watcher.cfrunloop-thread** -- Spawn a dedicated thread that runs `CFRunLoopRun()`. Create the FSEvents stream with: paths `["/"]`, `sinceWhen` from stored event ID (or `kFSEventStreamEventIdSinceNow`), latency 0.5s, flags `kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagUseCFTypes | kFSEventStreamCreateFlagNoDefer`. Schedule on the thread's run loop. (est: 3h)

7. **fsevents-watcher.event-callback** -- Implement the FSEvents callback function that receives event batches. For each event: extract path and flags, filter through `ExcludeTrie::should_exclude()`, map FSEvents flags to `FsChange` variants (see TECH.md Section 3.5 flag mapping table), send via `change_tx` channel. Update `last_event_id` atomically after each batch. (est: 3h)

8. **fsevents-watcher.event-id-resume** -- On startup, read `last_fsevents_id` from the GSIX v2 header and pass as `sinceWhen` to `FSEventStreamCreate`. Handle edge cases: event ID regression (volume reformat) triggers full rebuild; journal truncation (stored ID older than oldest retained) triggers full rebuild. (est: 2h)

9. **fsevents-watcher.history-done** -- Handle `kFSEventStreamEventFlagHistoryDone` by sending `FsChange::HistoryDone` through the channel. This signals the IndexWriter that replay is complete and live monitoring has begun. Log the number of replayed changes. (est: 1h)

10. **fsevents-watcher.shutdown** -- Implement `FSEventsListener::stop()`: set shutdown flag, `FSEventStreamStop()`, `FSEventStreamInvalidate()`, `FSEventStreamRelease()`, stop the CFRunLoop, join the thread. Implement `Drop` to call `stop()`. (est: 2h)

11. **fsevents-watcher.deprecate-notify** -- Feature-gate the `notify` and `notify-debouncer-mini` crate dependencies behind `#[cfg(not(target_os = "macos"))]`. On macOS, the `FSEventsListener` is used unconditionally. Keep the old `IndexWatcher` code behind the same gate for potential future non-macOS use. (est: 1h)

12. **fsevents-watcher.exclude-trie-tests** -- Write unit tests for `ExcludeTrie`: absolute prefix matching, basename matching, include overrides take priority, empty trie excludes nothing, deeply nested excluded dir detected, non-excluded path passes. (~8 tests) (est: 2h)

13. **fsevents-watcher.event-id-tests** -- Write tests: event ID stored in header after processing events, event IDs increase monotonically, event ID survives restart (stop/load/verify), event ID = 0 triggers full scan, resume catches missed changes, no duplicate entries after resume. (~6 tests) (est: 2h)

14. **fsevents-watcher.coalescing-tests** -- Write tests: 100 rapid file creates trigger single rebuild, rapid edits coalesced, debounce window respected, burst-then-quiet settles within 3s. (~4 tests) (est: 2h)

## Dependencies
- Requires: [gsix-v2-format (needs `last_fsevents_id` and `exclude_hash` fields in v2 header)]
- Enables: [incremental-updates (receives FsChange events from FSEventsListener via crossbeam channel), global-root-index (FSEvents stream watches "/" for live updates)]

## Acceptance Criteria
1. FSEvents stream watches `/` recursively with no file descriptor limit (unlike kqueue's 256 FD limit)
2. Event ID is persisted in the GSIX v2 header and used for `sinceWhen` on restart
3. All events since the stored event ID are replayed on startup (verified by create-while-stopped test)
4. `ExcludeTrie::should_exclude()` correctly filters all Tier 1/2/3 patterns
5. `HistoryDone` event is detected and forwarded to signal replay completion
6. Watcher shuts down cleanly (no leaked threads, no leaked file descriptors)
7. Event ID regression (stored > current) is detected and triggers full rebuild
8. `notify` crate is feature-gated behind `cfg(not(target_os = "macos"))`

## Technical References
- PM: ai/tasks/spec/PM.md -- Section 4 (FSEvents watcher migration), Section 5 (Risks: FSEvents event flood, event ID regression, fsevent-stream maturity)
- UX: ai/tasks/spec/UX.md -- Section 5 (Background Update UX: subtle updating indicator)
- Tech: ai/tasks/spec/TECH.md -- Section 3 (FSEvents Integration: crate selection, listener design, event flag mapping, event ID persistence), Section 9 (Exclude List: tiers, ExcludeTrie, exclude hash)
- QA: ai/tasks/spec/QA.md -- Section 3 (FSEvents Tests: event ID persistence, resume, coalescing, rapid changes, permissions)
