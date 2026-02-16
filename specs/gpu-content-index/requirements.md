---
spec: gpu-content-index
phase: requirements
created: 2026-02-16
generated: auto
---

# Requirements: gpu-content-index

## Summary

Eliminate ALL disk reads during content search by building an in-memory content store at startup, enabling zero-disk-I/O GPU-accelerated search with sub-100ms response times. Phase 1 POC: brute-force GPU scan of in-memory content (no trigram index).

## User Stories

### US-1: Zero Disk I/O Content Search
As a developer searching a codebase, I want search queries answered entirely from memory so that every keystroke yields results in under 100ms with zero disk reads.

**Acceptance Criteria**:
- AC-1.1: After content store build completes, typing a 2+ character pattern yields results within 100ms
- AC-1.2: Zero `read()`, `open()`, or `pread()` syscalls occur on content files during search
- AC-1.3: Results include file path, line number, column, and match context (identical to current disk-based results)
- AC-1.4: GPU scans content store buffer directly via `bytesNoCopy` -- no data copying

### US-2: Background Content Store Build
As a developer launching the tool, I want the content store to build in the background so that I can begin searching immediately using the existing disk-based fallback.

**Acceptance Criteria**:
- AC-2.1: Content store builds on a background thread without blocking the UI
- AC-2.2: Search functions via disk-based path while content store builds (graceful fallback)
- AC-2.3: When content store is ready, subsequent searches automatically use the in-memory fast path
- AC-2.4: Build progress is trackable via an atomic counter (files indexed so far)

### US-3: Persistent Content Store (Warm Restart)
As a developer restarting the tool, I want the content store loaded from disk instantly so that I don't wait for a full filesystem re-read on every launch.

**Acceptance Criteria**:
- AC-3.1: Content store is serialized to a GCIX file on disk after build completes
- AC-3.2: On restart, GCIX file is mmap'd and available in under 5 seconds (no re-scanning)
- AC-3.3: GCIX file includes a magic number, version, and CRC32 for integrity validation
- AC-3.4: Stale GCIX files (version mismatch or corruption) trigger automatic rebuild

### US-4: Incremental Content Updates
As a developer editing files, I want changed files re-indexed automatically so that search results stay fresh without manual rebuild.

**Acceptance Criteria**:
- AC-4.1: FSEvents notifications trigger re-read of modified/created/deleted files
- AC-4.2: Changed file content is reflected in search results within 2 seconds
- AC-4.3: Incremental updates do not require full content store rebuild
- AC-4.4: Deleted files are removed from the content store

### US-5: Search Result Correctness
As a developer, I want in-memory search results to be identical to disk-based search results so that I can trust the new code path.

**Acceptance Criteria**:
- AC-5.1: For any pattern, in-memory results are a superset of disk-based results (zero false negatives)
- AC-5.2: GPU verification produces no false positives (byte-exact matching)
- AC-5.3: Line numbers and byte offsets match between in-memory and disk-based paths

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | ContentStore holds all text file contents in a contiguous, page-aligned, mmap'd buffer | Must | US-1 |
| FR-2 | ContentStore buffer is GPU-addressable via Metal `bytesNoCopy` (zero-copy) | Must | US-1 |
| FR-3 | FileContentMeta table tracks offset, length, path_id, content_hash, mtime per file | Must | US-1 |
| FR-4 | ContentDaemon builds content store on background thread using existing filesystem walker | Must | US-2 |
| FR-5 | ContentDaemon skips binary files (reuse BinaryDetector) and files > 100MB | Must | US-2 |
| FR-6 | ContentIndexStore provides lock-free `ArcSwap` access to current ContentSnapshot | Must | US-2 |
| FR-7 | SearchOrchestrator checks ContentIndexStore; if available, dispatches GPU from content store buffer instead of per-file `fs::read()` | Must | US-1 |
| FR-8 | ContentSearchEngine gains `search_with_buffer()` accepting external Metal buffer | Must | US-1 |
| FR-9 | GCIX file format: header (16KB page-aligned) + FileContentMeta table + content data | Must | US-3 |
| FR-10 | GCIX mmap on restart provides instant content store with bytesNoCopy GPU access | Must | US-3 |
| FR-11 | FSEvents listener feeds file changes to content store for incremental updates | Should | US-4 |
| FR-12 | Content store append-only during runtime; compaction deferred to Phase 2 | Should | US-4 |
| FR-13 | Existing disk-based search pipeline preserved as fallback when content store unavailable | Must | US-2 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Content search latency < 100ms after content store ready (any corpus < 10GB) | Performance |
| NFR-2 | Content store build time < 120s for 800K text files on M4 Pro | Performance |
| NFR-3 | Warm restart (GCIX mmap) < 5 seconds | Performance |
| NFR-4 | Memory usage: content store < 2x raw content size | Resource |
| NFR-5 | All 561 existing lib tests pass without modification | Regression |
| NFR-6 | Zero data races under concurrent read/write (arc-swap semantics) | Safety |
| NFR-7 | GPU Metal buffer lifetime never exceeds backing mmap lifetime | Safety |

## Out of Scope

- Trigram inverted index (Phase 2 -- not part of this POC)
- Compressed posting lists (Phase 4)
- Regex trigram extraction (Phase 4)
- UI changes beyond status bar text updates
- Memory budget management with LRU eviction (Phase 3)
- Case-insensitive trigram index
- Cross-platform support (macOS Apple Silicon only)

## Dependencies

- Existing `MmapBuffer` for page-aligned memory allocation
- Existing `IndexStore` pattern for lock-free snapshot store
- Existing `IndexDaemon` pattern for background build lifecycle
- Existing `ContentSearchEngine` for GPU dispatch (extended with `search_with_buffer`)
- Existing `FSEventsListener` for filesystem change notifications
- Existing `BinaryDetector` for binary file filtering during build
