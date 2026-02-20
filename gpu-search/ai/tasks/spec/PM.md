# PM Analysis: Persistent GPU-Friendly File Index

**Author**: Product Manager Agent
**Date**: 2026-02-14
**Status**: Final
**Spec Phase**: Requirements Analysis (Persistent File Index)

---

## 1. Problem Statement

gpu-search currently suffers from three compounding issues that make root-level search unusable:

**29-second cold start from `/`.** When no cached index exists (or the cache is stale), `walk_directory()` in the orchestrator blocks while enumerating ~1.3M files from the system root. The streaming pipeline (`search_streaming`) mitigated this by overlapping walk with GPU dispatch, but users still wait tens of seconds before seeing first results because the producer thread must traverse the entire filesystem before meaningful coverage is achieved.

**1-hour staleness window.** The current `SharedIndexManager` uses `DEFAULT_MAX_AGE = 3600s` (defined in `src/index/shared_index.rs:36`). Any index older than one hour triggers a full rebuild via `is_stale()`. Files created or deleted within that window are invisible to search. For developers who create and switch branches frequently, an hour of staleness means the index rarely reflects reality.

**kqueue file descriptor exhaustion.** The `IndexWatcher` (in `src/index/watcher.rs`) uses the `notify` crate's `RecommendedWatcher` with `debouncer-mini` at 500ms. On macOS, `notify` uses kqueue for per-file watching, which hits the 256 open file descriptor limit at scale. Watching `/` recursively with kqueue is fundamentally impossible -- the system has hundreds of thousands of directories. The current watcher works for project-scoped roots but fails silently for system-wide indexing.

**Net effect:** Users who search from `/` (or any broad root) experience either a 29-second blocking stall, stale results, or a watcher that silently stops receiving events. The tool becomes unreliable for the most common power-user workflow: "find that file somewhere on my machine."

### Current Architecture

| Component | File | Role | Limitation |
|-----------|------|------|------------|
| `GpuPathEntry` | `src/gpu/types.rs` | 256B `#[repr(C)]` record matching Metal shader layout | Solid -- no changes needed |
| `GpuResidentIndex` | `src/index/gpu_index.rs` | CPU-side `Vec<GpuPathEntry>`, uploads via `newBufferWithBytes` | Copies data to GPU; no zero-copy path from mmap |
| `SharedIndexManager` | `src/index/shared_index.rs` | GSIX binary format (64B header + packed 256B entries), `fs::read()` load | Reads entire file into Vec; 1-hour staleness check; no event ID tracking |
| `MmapIndexCache` | `src/index/cache.rs` | mmap-based zero-copy loading of GSIX files | Works but `into_resident_index()` copies out of mmap into Vec; no `bytesNoCopy` Metal path |
| `GpuIndexLoader` | `src/index/gpu_loader.rs` | Orchestrates cache-hit vs scan-build-save-load | Falls back to full scan on cache miss; no incremental update |
| `FilesystemScanner` | `src/index/scanner.rs` | Parallel walk via `ignore` crate + rayon | Full re-scan every time; no delta awareness |
| `IndexWatcher` | `src/index/watcher.rs` | `notify` + `debouncer-mini` (500ms), full re-scan on any event | kqueue FD limit at 256; re-scans entire root on every single change event |
| `SearchOrchestrator` | `src/search/orchestrator.rs` | 3-stage streaming pipeline; walks filesystem per search | No index-first path; always walks from scratch |

### Industry Precedent

Research into how established tools solve this same problem confirms the architecture we need:

**plocate** builds a trigram inverted index over path strings stored as compressed posting lists in blocks. The entire database is memory-mapped. It finds 2 files out of 27 million entries in 0.008 milliseconds -- a 2,500x speedup over mlocate's linear scan (20ms for the same query). This proves that an mmap'd persistent index is the correct architecture for sub-millisecond path search at scale.

**mlocate** uses a merging `updatedb` strategy: it reuses the existing database to avoid re-reading directories that have not changed, reducing I/O. Its binary format uses a header followed by a directory tree structure. The key insight: mlocate proves that incremental updates (only re-reading changed directories) dramatically reduce update cost versus full re-scan.

**macOS Spotlight** uses FSEvents journal events with stored event IDs. Apple's patents describe a two-level inverted indexing table with a live index and annotated postings list, supporting incremental updates via delta files rather than complete rebuilds. When a file changes, the change is recorded in the volume's FSEvents database at `/.fseventsd/`, which triggers Spotlight's Metadata Server (`mds_stores`) to re-index only the affected files. Event IDs persist across reboots.

**Apple FSEvents API** operates at the kernel level with no file descriptor limit (unlike kqueue's 256). It provides persistent event storage on each volume's `/.fseventsd/` directory. Applications can resume watching from a stored event ID via the `sinceWhen` parameter to `FSEventStreamCreate`, receiving all events that occurred since that ID -- even across reboots. This is the exact primitive we need to replace kqueue-based watching.

---

## 2. User Stories

### Primary

**US-1: Instant search from root.**
As a developer, I want to open gpu-search and immediately search from `/` with results appearing in under 100ms, so I can find any file on my machine without waiting for a filesystem walk.

**US-2: Always-fresh results.**
As a developer, I want search results to reflect filesystem changes (file create, rename, delete) within 1 second of the change occurring, so I never see stale results or miss newly created files.

**US-3: Survive restarts.**
As a developer, I want the file index to persist across app restarts and load instantly from disk, so I never have to wait for a full rebuild after rebooting or restarting the app.

### Secondary

**US-4: Zero-copy GPU dispatch.**
As a performance-sensitive user, I want the index to load directly into a Metal GPU buffer via `makeBuffer(bytesNoCopy:)` without any memory copies, so GPU path-filter searches start with minimal latency and zero additional memory overhead.

**US-5: Ignore useless paths.**
As a developer, I want the index to automatically skip system directories (`/System`, `/Library/Caches`, `/private/var`), build artifacts (`target`, `dist`, `build`, `node_modules`), VCS internals (`.git`, `.hg`, `.svn`), caches (`__pycache__`, `.pytest_cache`, `.mypy_cache`), and trash (`.Trash`), so the index stays small and searches return only relevant results.

**US-6: Scale to 1M+ files.**
As a power user with large monorepos, multiple Node.js projects (each with deep `node_modules` trees), and Homebrew-managed toolchains, I want the index to handle over 1 million file entries without degraded search or update performance.

### Edge Cases

**US-7: Permission handling.**
As a user, I want the indexer to silently skip directories it cannot read (e.g., `/private`, SIP-protected system paths) without errors, hangs, or Full Disk Access permission prompts, so the index-build process is always non-interactive.

**US-8: Volume scoping.**
As a user with external drives or network mounts, I want the index to stay scoped to the boot volume by default and not traverse into `/Volumes/` mounts, so indexing time and index size are predictable and bounded.

---

## 3. Success Metrics

### Performance Targets

| Metric | Target | Current | Measurement Method |
|--------|--------|---------|-------------------|
| Cold start (index exists on disk) | < 100ms to search-ready state | ~29s (full walk from `/`) | `Instant::elapsed()` from app launch to `GpuLoadedIndex` available |
| Warm start (index already in memory) | < 5ms | N/A (no in-memory persistence across searches) | Timer around second `search()` call in same session |
| Incremental update latency | < 1s from file change to index updated | ~30s (full re-scan triggered by watcher event in `processor_loop`) | FSEvents event timestamp to `SharedIndexManager::save()` completion |
| Index file size at 1M entries | <= 260 MB (4KB header + 1M x 256B) | N/A at this scale | `stat ~/.gpu-search/index/*.idx` |
| mmap load time (1M entries, ~260MB) | < 50ms | ~10ms for 100K entries (per `gpu_loader.rs` target comment) | Timer around `MmapBuffer::from_file()` + header validation |
| GPU buffer creation (bytesNoCopy) | < 1ms | ~5ms (`newBufferWithBytes` copies data) | Timer around `makeBuffer(bytesNoCopy:length:options:deallocator:)` |
| Initial full scan (first-ever build from `/`) | < 60s | ~29s walk + build + save overhead | Timer around `FilesystemScanner::scan()` through save completion |
| Memory overhead (process RSS delta) | < 50 MB for 1M entries (mmap pages faulted on demand) | ~256 MB (full Vec loaded into heap) | `mach_task_basic_info` RSS before/after index load |

### Reliability Targets

| Metric | Target |
|--------|--------|
| FSEvents event delivery rate | 100% of non-excluded paths (FSEvents kernel guarantee) |
| Index corruption rate | 0 (atomic write-tmp-then-rename pattern, GSIX magic/version validation) |
| Permission-related crashes or hangs | 0 (all permission errors silently skipped, matching existing `FilesystemScanner` behavior) |
| Watcher uptime after `start()` | 100% (no FD exhaustion with FSEvents; no silent failures) |
| Event ID resume correctness | 100% (all events since stored ID replayed on restart) |

### Quality Gates

- All existing 253 unit tests and 152 integration tests continue to pass
- New tests for: FSEvents watcher creation/teardown, incremental update (add/remove/modify), mmap load with v2 header, `bytesNoCopy` GPU buffer creation, stored event ID resume, exclude-list filtering
- Benchmark gate: load 1M-entry index file in < 100ms on Apple Silicon (measured in CI)
- No regressions in existing streaming search pipeline (`search_streaming`) performance
- Backward compatibility: GSIX v1 files are detected and trigger a one-time full rebuild to v2

---

## 4. Scope

### In Scope

| Feature | Description | Priority |
|---------|-------------|----------|
| **GSIX v2 binary format** | Bump header version to 2. Add `last_fsevent_id: u64` for incremental resume, `entry_count_u64: u64` for future >4B entries, `flags: u64` for format metadata. Pad header to 4096 bytes (one memory page) so entries start at page boundary for `bytesNoCopy` alignment. Use existing 40-byte `_reserved` space in v1 header for new fields; extend to 4096 total with additional reserved padding. | P0 |
| **mmap-based instant load** | On startup, `GpuIndexLoader::try_load_from_cache()` memory-maps the GSIX v2 file. Already partially implemented in `cache.rs` (`MmapIndexCache::load_mmap`). Eliminate the `into_resident_index()` copy that currently transfers data from mmap to a heap-allocated Vec. | P0 |
| **Metal `bytesNoCopy` zero-copy buffer** | Replace `newBufferWithBytes` (which copies) with `makeBuffer(bytesNoCopy:length:options:deallocator:)` using the mmap'd region directly. The 4096-byte page-aligned header ensures entries start at a page boundary. The mmap buffer's lifetime must be tied to the Metal buffer's lifetime via the deallocator closure. | P0 |
| **FSEvents watcher migration** | Replace `notify` crate + kqueue backend with `fsevent-stream` or `fsevent` Rust crate for macOS FSEvents API. Create a recursive FSEvents stream for `/` with no FD limit. Store `last_fsevent_id` in GSIX v2 header. On startup, resume from stored event ID via `sinceWhen` parameter to receive all missed events. | P0 |
| **Incremental index update** | On FSEvents notification (directory-level events after 500ms debounce), re-scan only the affected directory (single `read_dir` call). Diff against existing entries for that directory: insert new entries, remove deleted entries, update modified entries (mtime/size). Persist updated index atomically. Replaces current behavior of full `FilesystemScanner::scan(root)` on every watcher event. | P0 |
| **Expanded exclude list** | Add to `DEFAULT_EXCLUDES` in `shared_index.rs`: `/System`, `/Library/Caches`, `/private/var`, `/private/tmp`, `/Volumes`, `.Trash`, `.fseventsd`, `.Spotlight-V100` (already present), `/cores`, `/dev`. Make exclude list configurable via a `~/.gpu-search/excludes.txt` file. | P0 |
| **Background initial scan** | First-ever scan runs in a background thread. UI shows progressive results as they arrive, reusing the existing streaming architecture from `search_streaming`. Save index to disk on completion. Subsequent launches load from disk instantly. | P1 |
| **Index compaction** | Deleted file entries are marked as tombstones (`path_len = 0`). GPU kernel already skips `path_len == 0` entries. When tombstone ratio exceeds 20% of total entries, trigger a compaction pass that rewrites the index without tombstones. | P1 |
| **Diagnostic commands** | Add `index stats` subcommand: entry count, file size, tombstone count, last event ID, index age, exclude count. Add `index rebuild` subcommand: force full re-scan and rebuild. | P2 |

### Out of Scope

| Feature | Reason |
|---------|--------|
| **Full-text content indexing** | The file index is for path enumeration only. Content search is handled by the GPU streaming pipeline in `SearchOrchestrator`. These are separate concerns. |
| **Network/remote drives** | `/Volumes` is explicitly excluded. Network filesystems have different consistency models and FSEvents may not function reliably on them. Future work if demand exists. |
| **Trigram inverted path index (plocate-style)** | Would provide sub-millisecond path filtering (0.008ms for 27M entries) but adds significant complexity (posting list compression, trigram extraction, index maintenance). The current 256B GpuPathEntry + Metal GPU kernel approach is fast enough for 1M entries (~2ms GPU dispatch). This is a potential future optimization if scale demands exceed 5M entries. |
| **Cross-volume event IDs** | FSEvents event IDs are per-volume. Multi-volume indexing requires per-volume streams with separate event ID tracking. Out of scope for boot-volume-only indexing. |
| **Spotlight integration** | Querying Spotlight's existing index could avoid building our own, but ties us to Apple's indexing schedule and non-GPU-compatible data format. We need 256B `#[repr(C)]` records for Metal shader layout. |
| **Windows/Linux support** | FSEvents is macOS-only. This entire feature is Apple Silicon + Metal specific by design. |
| **User-configurable root paths** | Always index from `/`. Users can narrow at search time via existing `SearchRequest.root`, filetype filters, and gitignore rules. |
| **Content-aware file type detection** | File type is determined by extension only, not by reading file headers (magic bytes). Matches existing `FilesystemScanner` behavior. |

### Compatibility Constraints

- The existing blocking `search()` API in `SearchOrchestrator` must continue to work unchanged for backward compatibility and tests
- The existing `SharedIndexManager` save/load API must remain backward-compatible: GSIX v1 files must be detected (magic + version check) and trigger a one-time full rebuild to v2
- The `GpuPathEntry` 256-byte `#[repr(C)]` layout must not change -- it matches the Metal shader definition in `search_types.h`
- The existing `SearchUpdate` streaming protocol used by `search_streaming()` must not break

---

## 5. Risks

### Technical Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **Permission errors scanning from `/`** | Medium | High | `FilesystemScanner` already silently skips unreadable entries (`Err(_) => continue` in `scan_recursive` and the parallel walker). Expanded exclude list preemptively avoids known SIP-protected paths (`/System`, `/private/var`, `/Library/Caches`). No Full Disk Access entitlement needed -- unreadable dirs are silently skipped. Validate with a test scan from `/` that completes without error. |
| **Large index file on disk** | Medium | Medium | At 1M entries x 256B + 4KB header = ~256 MB. Typical developer Mac has 256GB+ SSD. The file is written atomically (tmp + rename) so partial writes never corrupt. mmap means only touched pages consume physical RAM. Future optimization: compress the path field (paths share common prefixes), potentially reducing size by 40-60%. |
| **FSEvents event flood during large operations** | Medium | Medium | FSEvents coalesces events at the directory level, not per-file. A `git checkout` modifying 10K files produces ~100-200 directory events, not 10K. The 500ms debounce window (matching current `IndexWatcher` behavior) handles bursts. Incremental update processes only changed directories, not full re-scan. Worst case: a large `npm install` creates ~50K files in `node_modules` but `node_modules` is excluded. |
| **FSEvents event ID regression on volume reformat** | Low | Low | Event IDs are 64-bit unsigned integers; rollover is impossible in practice. Volume reformat resets the ID sequence. Mitigation: detect ID regression (current kernel event ID < stored `last_fsevent_id`) and trigger full rebuild. Also trigger rebuild if stored event ID is older than the FSEvents journal's oldest retained entry. |
| **mmap + bytesNoCopy page alignment** | Medium | Low | Metal's `makeBuffer(bytesNoCopy:)` requires page-aligned memory (4096 bytes on Apple Silicon). mmap returns page-aligned addresses. The GSIX v2 header is padded to exactly 4096 bytes, so entries begin at the next page boundary. If the file itself is not page-size-aligned in total length, the last partial page is still valid for mmap. Validate with a test that creates a bytesNoCopy buffer from mmap and reads entries in a Metal kernel. |
| **`fsevent-stream` crate maturity** | Medium | Medium | The crate provides Stream-based bindings supporting `kFSEventStreamEventIdSinceNow` and stored event IDs. If edge cases arise, fallback options: (a) the older `fsevent` crate (more established, on crates.io since 2016), (b) raw `CoreServices` FFI via `objc2`, (c) configure `notify` crate to force its FSEvents backend (it has one, but prefers kqueue on macOS by default for per-file granularity). |
| **Race: index file replaced during mmap read** | Medium | Low | Already handled by the atomic write pattern in `SharedIndexManager::save()`: writes to `.idx.tmp`, then `rename()` to `.idx`. The old inode persists until all mmap handles close. A reader with an existing mmap sees the old (consistent) data. New readers opening after the rename see the new data. No corruption possible. |
| **Index grows unbounded with file churn** | Low | Medium | Deleted files become tombstones (`path_len = 0`). Without compaction, tombstones accumulate. Mitigation: compaction pass (Phase 4) runs when tombstone ratio > 20%. Hard cap: reject new entries beyond 4M total (1 GB index file) and log a warning. |
| **bytesNoCopy deallocator lifetime** | Medium | Low | The mmap buffer backing a `bytesNoCopy` Metal buffer must not be unmapped while the GPU buffer exists. Solution: store the `MmapBuffer` inside `GpuLoadedIndex` (the `_mmap` field already exists for this purpose in `gpu_loader.rs:93`). The `MmapBuffer` is dropped only when the `GpuLoadedIndex` is dropped, which is after all GPU dispatches complete. |

### Product Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **First-launch UX: 60-second initial scan** | Medium | Show a progress indicator with discovered file count and estimated time remaining. Allow searching against the partial index during the scan (existing streaming architecture supports this). Cache the index immediately on completion so subsequent launches are instant. |
| **Disk space usage** | Low | 256 MB for 1M files is modest on modern Macs. Add `index stats` diagnostic showing size, entry count, and age. Add `index rebuild` for manual invalidation. Document expected disk usage in help text. |
| **Privacy concern: indexing all paths** | Low | Only path names are stored, not file contents. The exclude list skips sensitive system paths. The index file is stored in `~/.gpu-search/` with user-only permissions (file mode 0600). The index is local-only and never transmitted. |
| **Existing users: GSIX v1 to v2 migration** | Low | On first launch with the v2 code, `GpuIndexLoader` detects the v1 version in the header, discards the v1 file, and triggers a full rebuild to v2. Users experience one slower launch (~60s) then instant launches thereafter. Log a message explaining the one-time rebuild. |

---

## 6. Priority and Phasing

**Overall priority: P0 for usability.** System-root search is the flagship use case for gpu-search. Without a persistent index, every search from `/` costs 29 seconds. With it, searches start in <100ms. This is the single highest-impact improvement for user experience.

### Phase 1: Persistent Index with mmap Load and bytesNoCopy GPU Buffer (P0, ~3 days)

**Goal:** Index always exists on disk; loads instantly on startup; zero-copy into Metal GPU buffer.

**Work items:**
- Define GSIX v2 header: bump version to 2, add `last_fsevent_id: u64`, `entry_count_u64: u64`, `flags: u64` in the reserved space, pad total header to 4096 bytes
- Update `SharedIndexManager::save()` and `SharedIndexManager::load()` for v2 format
- Update `MmapIndexCache::load_mmap()` to validate v2 headers and handle v1-to-v2 migration (detect v1, return error signaling rebuild needed)
- Implement `bytesNoCopy` Metal buffer creation in `GpuIndexLoader`: after mmap, create Metal buffer with `makeBuffer(bytesNoCopy:length:options:deallocator:)` pointing at `mmap_ptr + 4096` (entries start after page-aligned header)
- Eliminate the `into_resident_index()` copy path in `GpuIndexLoader::try_load_from_cache()`
- Background initial scan on first launch: spawn `FilesystemScanner::scan("/")` in a thread, save v2 index on completion
- Expand `DEFAULT_EXCLUDES` with system paths: `/System`, `/Library/Caches`, `/private/var`, `/private/tmp`, `/Volumes`, `.Trash`, `/cores`, `/dev`

**Deliverables:** Cold start from cached index < 100ms. GPU buffer created via `bytesNoCopy` with zero memory copies. All existing tests pass. v1 indexes trigger clean rebuild.

### Phase 2: FSEvents Watcher Migration (P0, ~2 days)

**Goal:** Replace kqueue-based `notify` watcher with native FSEvents; no FD limit; event ID persistence.

**Work items:**
- Add `fsevent-stream` (or `fsevent`) crate dependency to `Cargo.toml`
- Implement `FsEventsWatcher` struct replacing `IndexWatcher` internals: creates a recursive FSEvents stream for `/` using `FSEventStreamCreate` with `kFSEventStreamCreateFlagFileEvents` for file-level granularity
- On startup, read `last_fsevent_id` from GSIX v2 header and pass as `sinceWhen` parameter to receive all missed events since last run
- On each index persist, store current FSEvents event ID in the v2 header
- Maintain 500ms debounce window (match current `IndexWatcher` behavior)
- Handle edge cases: event ID regression detection (volume reformat), journal truncation (event ID older than oldest retained event)
- Remove or gate the `notify` crate dependency behind a feature flag

**Deliverables:** Watcher works from `/` without FD exhaustion. FSEvents events resume correctly across app restarts. Event IDs stored in index file.

### Phase 3: Incremental Index Update (P0, ~3 days)

**Goal:** Apply deltas instead of full re-scan; sub-second update latency.

**Work items:**
- On FSEvents notification (after debounce), identify changed paths from event data (FSEvents provides directory-level or file-level paths depending on flags)
- For each changed directory: execute a single `std::fs::read_dir()` call to get current state
- Build a diff against existing entries for that directory prefix in the index: detect new files (add entry), deleted files (tombstone: set `path_len = 0`), modified files (update `mtime`, `size_lo`, `size_hi`)
- Apply delta to the in-memory index representation
- Persist updated index atomically: write to `.idx.tmp`, rename to `.idx`
- Update mmap reference: close old mmap, re-mmap new file, recreate `bytesNoCopy` Metal buffer pointing to new mmap
- Replace the current `processor_loop` in `watcher.rs` which calls `FilesystemScanner::scan(root)` (full re-scan) with the incremental delta logic

**Deliverables:** File changes reflected in searchable index within 1 second. No full filesystem re-scan on watcher events. Verified with test: create file, wait 1.5s, search finds it.

### Phase 4: Index Compaction and Scale Testing (P1, ~2 days)

**Goal:** Handle 1M+ entries reliably; prevent unbounded growth; prove performance at scale.

**Work items:**
- Implement tombstone tracking: count entries with `path_len == 0`
- Compaction trigger: when tombstone ratio > 20%, rewrite index excluding tombstoned entries
- Scale test harness: generate 1M synthetic `GpuPathEntry` records, save to GSIX v2 file, benchmark mmap load time, bytesNoCopy buffer creation, and GPU kernel dispatch
- Verify benchmarks: mmap load < 50ms, bytesNoCopy < 1ms, total cold start < 100ms at 1M entries
- Hard capacity cap: reject new entries beyond 4M total (1 GB index file), log warning
- Add `index stats` diagnostic: entry count, tombstone count, file size, last event ID, index age
- Add `index rebuild` manual rebuild command

**Deliverables:** Proven 1M-entry performance meeting all targets. No unbounded growth. Diagnostic tooling for debugging.

### Phase Dependencies

```
Phase 1 (Persistent Index + bytesNoCopy) ---------> Phase 3 (Incremental Update)
          |                                                    |
          v                                                    v
Phase 2 (FSEvents Watcher) --------------------------> Phase 4 (Compaction + Scale)
```

Phase 1 is the foundation (v2 format, mmap, bytesNoCopy). Phase 2 provides the event source (FSEvents with stored IDs). Phase 3 connects them (FSEvents events drive incremental index updates). Phase 4 hardens the system at scale.

---

## 7. Technical Architecture Summary

### Data Flow: Startup (Warm -- Index Exists on Disk)

```
App Launch
  |
  v
~/.gpu-search/index/<hash>.idx exists?
  |
  YES --> mmap file via MmapBuffer::from_file() ............... < 1ms
  |         |
  |         v
  |       Validate GSIX v2 header:
  |         - magic == 0x58495347 ("GSIX")
  |         - version == 2
  |         - root_hash matches expected
  |         |
  |         v
  |       Read last_fsevent_id from header .................... < 0.01ms
  |         |
  |         v
  |       makeBuffer(bytesNoCopy:
  |           mmap_ptr + 4096,               // entries after page-aligned header
  |           length: entry_count * 256,
  |           options: .storageModeShared,
  |           deallocator: { /* release mmap */ }
  |       ) ................................................... < 1ms
  |         |
  |         v
  |       Start FSEvents stream with sinceWhen: last_fsevent_id
  |         |
  |         v
  |       SEARCH READY ...................................... < 100ms total
  |
  NO --> Spawn background scan thread (see Cold Start flow)
```

### Data Flow: Startup (Cold -- First Launch, No Index)

```
App Launch (no index file)
  |
  v
Spawn background thread:
  FilesystemScanner::scan("/")
  with expanded excludes (/System, /Volumes, node_modules, .git, etc.)
  ~1.3M files -> ~800K after excludes -> ~60 seconds
  |
  v (progressive: batches sent via channel to UI)
Build Vec<GpuPathEntry> from scanned entries
  |
  v
Save GSIX v2 file: 4096B header + N * 256B entries
  |
  v
mmap saved file -> bytesNoCopy Metal buffer
  |
  v
Start FSEvents stream with sinceWhen: kFSEventStreamEventIdSinceNow
  |
  v
Store initial event ID in next index persist
  |
  v
SEARCH READY (progressive results available during scan)
```

### Data Flow: Incremental Update (Steady State)

```
FSEvents stream delivers directory-level event(s)
  |
  v
500ms debounce coalesce window
  |
  v
For each changed directory path:
  |
  v
read_dir(changed_dir) -> collect current files
  |
  v
Diff against existing index entries with that directory prefix:
  - New files:     insert GpuPathEntry (append to index)
  - Deleted files: tombstone existing entry (set path_len = 0)
  - Modified files: update mtime, size_lo, size_hi in existing entry
  |
  v
Persist: write .idx.tmp (atomic), rename to .idx
  |
  v
Re-mmap new file -> new bytesNoCopy Metal buffer
  |
  v
Store current FSEvents event ID in v2 header
  |
  v
INDEX UPDATED (< 1 second from file change)
```

### GSIX v2 Header Layout

```
Offset  Size    Field              Description
------  ------  -----------------  ------------------------------------------
0       4       magic              0x58495347 ("GSIX") -- unchanged from v1
4       4       version            2 (bumped from 1)
8       4       entry_count        u32 entry count -- unchanged from v1
12      4       root_hash          u32 hash of root path -- unchanged from v1
16      8       saved_at           u64 unix timestamp -- unchanged from v1
24      8       last_fsevent_id    u64 FSEvents event ID for resume (NEW)
32      8       entry_count_u64    u64 entry count, future >4B entries (NEW)
40      8       flags              u64 bitflags: 0x1=sorted, 0x2=has_tombstones (NEW)
48      4048    _reserved          Zero-padded to 4096 bytes total
------  ------  -----------------  ------------------------------------------
TOTAL   4096    (one memory page, page-aligned for bytesNoCopy)
```

The v1 header was 64 bytes. The v2 header expands to 4096 bytes (one 4KB memory page). The first 24 bytes are layout-compatible with v1. New fields occupy bytes 24-48 (previously `_reserved` in v1). Bytes 48-4095 are reserved for future use (zero-filled).

Entries start at byte offset 4096, which is page-aligned on both Intel (4KB pages) and Apple Silicon (16KB pages -- 4096 is aligned to 16KB as well since 16384 / 4096 = 4). This ensures `makeBuffer(bytesNoCopy:)` can use the entries region directly.

### Key Crate Dependencies

| Crate | Purpose | Status |
|-------|---------|--------|
| `memmap2` | mmap file I/O | Already in use via `MmapBuffer` wrapper in `src/io/mmap.rs` |
| `fsevent-stream` or `fsevent` | Native macOS FSEvents API bindings with event ID support | **NEW** -- replaces `notify` crate for system-wide watching |
| `objc2-metal` | `makeBuffer(bytesNoCopy:length:options:deallocator:)` | Already in `Cargo.toml` |
| `crossbeam-channel` | Bounded channels for watcher->processor events | Already in `Cargo.toml` |
| `ignore` | Parallel directory walk with gitignore rules | Already used by `FilesystemScanner` |
| `notify` + `notify-debouncer-mini` | Current kqueue-based watcher | **REMOVE or feature-gate** after FSEvents migration |
| `dirs` | Home directory resolution for `~/.gpu-search/` | Already in `Cargo.toml` |

---

## 8. Open Design Decisions

These are flagged for the design phase and are not blockers for PM approval:

**1. Header size: 4096 bytes vs. 64 bytes with kernel offset.**
Padding to 4096 bytes wastes ~4KB per index file but ensures trivial page-aligned `bytesNoCopy`. Alternative: keep 64-byte header and have the Metal kernel offset its buffer reads by 64 bytes. Recommendation: **pad to 4096** -- the cost is negligible (0.0015% of a 260MB file) and eliminates all alignment concerns.

**2. In-place delta update vs. full rewrite on each change.**
For small deltas (1-10 files), rewriting 260MB is I/O-heavy. Options:
- (a) Always rewrite: simple, atomic, correct. 260MB sequential write takes ~260ms on APFS SSD.
- (b) Append-only delta log: new/modified entries appended; compaction merges periodically. More complex but reduces write amplification.
- (c) In-place mmap write: `MAP_SHARED` + `mprotect(PROT_WRITE)` to patch entries in-place. Fastest but risks corruption if process crashes mid-write.
Recommendation: **start with (a)** for correctness and simplicity. Measure write latency. If >500ms at 1M entries, optimize to (b).

**3. `notify` crate removal vs. dual backend.**
The `notify` crate provides cross-platform support. If `fsevent-stream` proves reliable, `notify` can be removed to reduce the dependency tree. Alternative: keep `notify` behind a `cfg(not(target_os = "macos"))` gate for theoretical future Linux support.
Recommendation: **feature-gate `notify`** behind `#[cfg(not(target_os = "macos"))]`. Use `fsevent-stream` on macOS unconditionally.

**4. Tombstone representation.**
Options:
- (a) Set `path_len = 0`: simplest. GPU kernel already skips entries with `path_len == 0` (no valid path has zero length). No additional storage.
- (b) Separate tombstone bitmap: more structured but adds complexity and a second data structure.
- (c) Never tombstone -- always compact immediately on delete: simplest semantics but requires full rewrite on every delete.
Recommendation: **(a) `path_len = 0`** as tombstone. The GPU kernel naturally ignores these. Compact when tombstone ratio exceeds 20%.

**5. FSEvents granularity: directory-level vs. file-level.**
`kFSEventStreamCreateFlagFileEvents` enables file-level events but increases event volume significantly. Directory-level events require re-scanning the affected directory to determine what changed.
Recommendation: **use file-level events** (`kFSEventStreamCreateFlagFileEvents`). This avoids the `read_dir` diff step for most updates and provides exact paths for insert/delete/modify. The slightly higher event volume is handled by the 500ms debounce window.

---

## 9. Research Sources

- [plocate: a much faster locate](https://plocate.sesse.net/) -- trigram inverted index architecture, mmap database, 2500x speedup over mlocate
- [plocate command as faster locate alternative (Linuxiac)](https://linuxiac.com/plocate-command/) -- 0.008ms query performance for 27M entries
- [plocate: Much Faster locate (Linux Uprising Blog)](https://www.linuxuprising.com/2021/09/plocate-is-much-faster-locate-drop-in.html) -- compressed posting list blocks, io_uring optimization
- [Apple FSEvents Programming Guide](https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/FSEvents_ProgGuide/UsingtheFSEventsFramework/UsingtheFSEventsFramework.html) -- `sinceWhen` parameter, stored event IDs, `kFSEventStreamEventIdSinceNow`, persistence across reboots
- [FSEvents: How They Work and Why They Matter (Hexordia)](https://www.hexordia.com/blog/mac-forensics-analysis) -- kernel-level event database at `/.fseventsd/`, event structure
- [Deeper dive into Spotlight indexes (Eclectic Light)](https://eclecticlight.co/2025/07/30/a-deeper-dive-into-spotlight-indexes/) -- two-level inverted index with live deltas
- [Spotlight on search: How Spotlight works (Eclectic Light)](https://eclecticlight.co/2021/01/28/spotlight-on-search-how-spotlight-works/) -- FSEvents triggers mds_stores re-indexing, XPC pipeline
- [Watching macOS file systems: FSEvents and volume journals (Eclectic Light)](https://eclecticlight.co/2017/09/12/watching-macos-file-systems-fsevents-and-volume-journals/) -- per-volume event IDs, persistence semantics
- [fsevent-stream Rust crate (docs.rs)](https://docs.rs/fsevent-stream) -- Stream-based FSEvents API bindings, event ID support, `kFSEventStreamCreateFlagUseExtendedData`
- [fsevent Rust crate (lib.rs)](https://lib.rs/crates/fsevent) -- established FSEvents bindings since 2016
- [mlocate.db(5) man page](https://linux.die.net/man/5/mlocate.db) -- binary header format, directory tree structure, incremental updatedb
- [updatedb(8) man page](https://linux.die.net/man/8/updatedb) -- merging database reuse, avoiding re-reading unchanged directories
- [Metal makeBuffer(bytesNoCopy:) Apple Documentation](https://developer.apple.com/documentation/metal/mtldevice/1433382-makebuffer) -- zero-copy Metal buffer from existing memory, page-alignment requirement
- [Regular Expression Matching with a Trigram Index (Russ Cox)](https://swtch.com/~rsc/regexp/regexp4.html) -- trigram indexing theory and posting list design
- [Cloudflare mmap-sync (GitHub)](https://github.com/cloudflare/mmap-sync) -- concurrent data access via mmap with zero-copy deserialization
