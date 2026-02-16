---
id: incremental-updates.BREAKDOWN
module: incremental-updates
priority: 3
status: pending
version: 1
origin: foreman-spec
dependsOn: [gsix-v2-format.BREAKDOWN, fsevents-watcher.BREAKDOWN, mmap-gpu-pipeline.BREAKDOWN]
tags: [persistent-index, gpu-search, incremental, index-writer]
---
# Incremental Updates — BREAKDOWN

## Context
The current watcher performs a full filesystem re-scan on every event batch, which takes 10-30 seconds for system-root indexes. This module implements the `IndexWriter` that applies targeted insert/modify/delete/rename operations to the in-memory index using a `HashMap<path, index>` for O(1) lookup. Changes are flushed to disk periodically (1000 dirty entries or 30s timer) using atomic write, then the `IndexStore` swaps the new snapshot for lock-free reader access.

References: TECH.md Section 4 (Incremental Update Strategy), PM.md Section 4 (Incremental index update), PM.md Phase 3.

## Tasks
1. **incremental-updates.index-writer-struct** -- Define `IndexWriter` in a new `src/index/index_writer.rs`: `entries: Vec<GpuPathEntry>`, `path_index: HashMap<Box<[u8]>, usize>` for O(1) lookup, `dirty_count: usize`, `last_flush: Instant`, `excludes: Arc<ExcludeTrie>`. Initialize from an existing `IndexSnapshot` by building the HashMap from loaded entries. (est: 2h)

2. **incremental-updates.created-handler** -- Implement `IndexWriter::handle_created(path)`: check `excludes.should_exclude()`, `stat()` the path (skip if race-deleted), build `GpuPathEntry` from metadata, check `path_index` for existing (treat as Modified if found), otherwise append to `entries` and record in `path_index`. Increment `dirty_count`. (est: 2h)

3. **incremental-updates.modified-handler** -- Implement `IndexWriter::handle_modified(path)`: lookup in `path_index` (treat as Created if not found), `stat()` (treat as Deleted if stat fails), update `mtime`, `size_lo`, `size_hi`, `flags` in place at existing index. Increment `dirty_count`. (est: 1h)

4. **incremental-updates.deleted-handler** -- Implement `IndexWriter::handle_deleted(path)`: lookup in `path_index` (skip if not found), set `entries[idx].flags |= IS_DELETED` (tombstone), remove from `path_index`. Increment `dirty_count`. (est: 1h)

5. **incremental-updates.renamed-handler** -- Implement `IndexWriter::handle_renamed(old, new)`: process as `handle_deleted(old)` followed by `handle_created(new)`. This handles both the path change and any metadata changes. (est: 1h)

6. **incremental-updates.must-rescan-handler** -- Implement `IndexWriter::handle_must_rescan(subtree)`: walk the subtree using `FilesystemScanner::scan()`, collect the set of found paths. For each existing entry whose path starts with `subtree`: update if found, tombstone if not found. For each found path not in `path_index`: insert. This is the expensive path (~10-100ms for typical subtrees). (est: 3h)

7. **incremental-updates.event-dispatch** -- Implement `IndexWriter::process_event(change: FsChange)` that dispatches to the appropriate handler based on the `FsChange` variant. Handle `HistoryDone` by logging replayed change count and triggering immediate flush. (est: 1h)

8. **incremental-updates.flush-trigger** -- Implement flush condition checking: `dirty_count >= 1000`, `last_flush.elapsed() >= 30s`, `HistoryDone` received, or graceful shutdown signal. Call `flush()` when any condition is met. Wire into the event processing loop. (est: 1h)

9. **incremental-updates.compaction** -- Implement compaction within `flush()`: remove entries where `flags & IS_DELETED != 0`, shift remaining entries to close gaps, update `path_index` with new positions. Track tombstone count; trigger compaction when tombstone ratio > 20%. (est: 2h)

10. **incremental-updates.sort-entries** -- After compaction, if any entries were appended (unsorted), sort the entire `entries` array by path bytes. Set `FLAG_SORTED` in the header flags. Sorting 1M entries takes ~200ms -- acceptable during flush. (est: 1h)

11. **incremental-updates.atomic-write** -- Implement the flush write procedure: write v2 header (with updated `entry_count`, `entry_capacity`, `last_fsevents_id`, `saved_at`, recomputed `checksum`) + entries to `global.idx.tmp`, `fsync()`, then `rename()` to `global.idx`. (est: 2h)

12. **incremental-updates.snapshot-swap** -- After successful atomic write, create a new `IndexSnapshot` (mmap the new file, create Metal buffer), and call `IndexStore::swap()` to make it available to readers. Reset `dirty_count = 0` and `last_flush = Instant::now()`. (est: 2h)

13. **incremental-updates.writer-thread** -- Spawn the IndexWriter on a dedicated thread that receives `FsChange` events from the FSEventsListener via a crossbeam bounded channel. Process events in a loop, checking flush conditions after each event or batch. Handle shutdown via a shutdown channel or atomic flag. (est: 2h)

14. **incremental-updates.crud-tests** -- Write tests: file create appears in index, file modify updates mtime/size, file delete tombstones entry (IS_DELETED set), file rename removes old + adds new, directory create indexes all children, directory delete tombstones all descendants, nested file create has correct full path. (~8 tests) (est: 3h)

15. **incremental-updates.concurrent-tests** -- Write tests: concurrent read during rebuild (mmap survives rename), atomic rename preserves old mmap, search during incremental update returns valid results, multiple readers + one writer with no data races. (~4 tests) (est: 2h)

16. **incremental-updates.no-data-loss-tests** -- Write tests: 500 rapid file creates all appear in final index, no duplicate entries after multiple update cycles, update preserves 999 of 1000 unmodified entries. (~3 tests) (est: 2h)

## Dependencies
- Requires: [gsix-v2-format (IS_DELETED flag, v2 header fields for flush), fsevents-watcher (FsChange events as input), mmap-gpu-pipeline (IndexStore.swap() for snapshot replacement, IndexSnapshot for mmap + Metal buffer)]
- Enables: [global-root-index (incremental updates keep the global index fresh), integration (end-to-end pipeline with live updates)]

## Acceptance Criteria
1. Single file change reflected in searchable index within <1 second of the FSEvents event arriving
2. No full filesystem re-scan on watcher events (only affected paths processed)
3. HashMap lookup provides O(1) amortized event processing for Created/Modified/Deleted
4. Tombstoned entries have `IS_DELETED` flag set and are filtered by the GPU kernel
5. Compaction removes tombstones when ratio exceeds 20%, producing a clean contiguous entry array
6. Flush writes atomically (no partial `.idx` files on crash)
7. `IndexStore::swap()` makes new snapshot available to readers without blocking in-flight searches
8. 500 rapid file creates all appear in the final settled index (no data loss)

## Technical References
- PM: ai/tasks/spec/PM.md -- Section 4 (Incremental index update), Phase 3 (Incremental Update), Section 8 decision #2 (full rewrite vs delta)
- UX: ai/tasks/spec/UX.md -- Section 5 (Background Update UX: transient updating indicator)
- Tech: ai/tasks/spec/TECH.md -- Section 4 (Incremental Update Strategy: in-memory representation, event processing logic, flush strategy, amortized costs)
- QA: ai/tasks/spec/QA.md -- Section 4 (Incremental Update Tests: CRUD, concurrent read, no data loss)
