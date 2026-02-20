# Persistent GPU-Friendly File Index — Spec Overview

## Date
2026-02-14

## Requirement Summary
gpu-search suffers from a 29-second cold start when searching from `/`, 1-hour index staleness, and kqueue file descriptor exhaustion at scale. This feature introduces a persistent GSIX v2 binary index loaded via mmap into a Metal `bytesNoCopy` GPU buffer, with FSEvents-driven incremental updates replacing the kqueue-based `notify` watcher. The result is sub-100ms cold start, sub-1s update latency, and reliable system-wide file search across 1M+ entries.

## PM Summary
- **Problem**: Root-level search blocks for ~29s (full filesystem walk), indexes go stale within 1 hour, and kqueue hits the 256 FD limit when watching `/` recursively
- **User stories**: Instant search from `/` (<100ms), always-fresh results (<1s update), survive app restarts, zero-copy GPU dispatch, ignore useless system/build paths, scale to 1M+ files
- **Success metrics**: Cold start <100ms (from 29s), warm start <5ms, incremental update <1s, index size <=260MB at 1M entries, GPU buffer creation <1ms via `bytesNoCopy`
- **Scope**: GSIX v2 format (page-aligned header), mmap instant load, Metal `bytesNoCopy`, FSEvents watcher migration, incremental index updates, expanded exclude list, background initial scan, index compaction
- **Out of scope**: Full-text content indexing, network drives, trigram inverted index, Windows/Linux, Spotlight integration
- **Phasing**: Phase 1 (persistent index + bytesNoCopy, ~3d) -> Phase 2 (FSEvents watcher, ~2d) -> Phase 3 (incremental update, ~3d) -> Phase 4 (compaction + scale testing, ~2d)
- **Risks**: Permission errors from `/` (mitigated by expanded excludes), 256MB index size (acceptable for modern Macs), FSEvents event flood (mitigated by coalescing + debounce), `fsevent-sys` crate maturity

## UX Summary
- **Index lifecycle states**: LOADING -> BUILDING -> READY -> UPDATING -> STALE -> ERROR, with well-defined transitions and fallback behavior at each state
- **Status bar as single source of truth**: Index status prepended as first segment (e.g., `1.3M files | 42 matches in 0.8ms | / | .rs`), color-coded by state (SUCCESS green for READY, ACCENT amber for BUILDING/UPDATING, ERROR red)
- **First-run experience**: Non-blocking background build with progressive file count (`Indexing: 420K files`); search works immediately via `walk_and_filter()` fallback; no time estimate shown (total unknown)
- **Background updates**: Subtle `Updating index` text in status bar during FSEvents-driven updates; no toast/notification on completion; file count silently increments
- **Error handling**: Corrupt index triggers silent rebuild; permissions silently skipped; disk full shows `Index: error` in status bar with walk fallback; watcher failure degrades to stale-but-usable
- **Number formatting**: Abbreviated K/M display (<1K exact, 1K-999K show K, >=1M show M) for compact status bar
- **Progressive enhancement**: Index is a performance optimization, not a requirement; search always works via walk fallback; no disabled states during build
- **Accessibility**: Color AND text for every state; WCAG AA contrast ratios verified for Tokyo Night theme

## Tech Summary
- **Architecture**: IndexDaemon (FSEvents + EventProcessor + IndexWriter) feeds IndexStore, which provides lock-free snapshots to SearchPipeline via `arc-swap`
- **GSIX v2 format**: 16KB page-aligned header (up from 64B) with `last_fsevents_id`, `exclude_hash`, `entry_capacity`, `flags`, CRC32 checksum; entries start at offset 16384 for Apple Silicon 16KB page alignment
- **`IS_DELETED` tombstone flag**: New bit flag on `GpuPathEntry.flags` (bit 4) for incremental deletes; GPU kernel checks `!(flags & IS_DELETED)` at negligible cost
- **FSEvents via `fsevent-sys`**: Direct CoreServices FFI for `sinceWhen` resume, `kFSEventStreamCreateFlagFileEvents` granularity, `HistoryDone` detection; replaces `notify` crate
- **Concurrency**: `arc-swap::ArcSwap<IndexSnapshot>` for lock-free reader access; writers produce new mmap + Metal buffer and swap atomically; no RwLock blocking
- **IndexWriter**: In-memory `Vec<GpuPathEntry>` + `HashMap<path, index>` for O(1) CRUD; flush on 1000 dirty entries or 30s timer; compact tombstones + sort + atomic write
- **Memory budget**: ~304MB at 1M entries (256MB mmap entries + 48MB writer HashMap); mmap pages reclaimable under pressure; UMA means GPU shares same physical pages
- **New dependencies**: `fsevent-sys = "4"`, `core-foundation = "0.10"`, `arc-swap = "1"`, `crc32fast = "1"`; removes `notify` and `notify-debouncer-mini`

## QA Summary
- **Test pyramid**: ~146 new tests (10 compile-time asserts, 80 unit, 25 integration, 15 property, 8 benchmarks, 8 stress/scale) bringing total to ~550
- **Key test areas**: GSIX v2 header validation, entry alignment verification, corrupt file handling (no panics on malformed input), FSEvents event ID persistence/resume, incremental CRUD operations, concurrent read-during-write safety, 1M+ entry scale, `bytesNoCopy` GPU pipeline
- **Property tests**: Format roundtrip fidelity, binary fuzzing (flip random bytes, truncate, random input -- never panic), cache key determinism
- **Acceptance criteria**: All 404 existing tests pass, mmap load <1ms at 100K entries, incremental update <1s, 1M entry scale passes without OOM, corrupt files produce `Err` not panics, indexed search results match non-indexed results, CI green
- **Performance benchmarks**: `criterion` benchmarks for mmap load (100K and 1M), save throughput, incremental update latency, GPU buffer creation
- **Regression guards**: `walk_and_filter()` fallback still works, blocking `search()` API unchanged, `GpuPathEntry` layout assertions, GPU-vs-CPU equivalence proptest

## Cross-Cutting Decisions
- **Header size**: 16KB (one Apple Silicon page) to ensure entry region page alignment for `bytesNoCopy` -- 0.006% overhead on a 260MB file
- **v1 to v2 migration**: No in-place migration; v1 detected by version field triggers one-time full rebuild to v2; old per-directory `.idx` files cleaned up after successful v2 build
- **Tombstone representation**: `path_len = 0` (PM spec) evolving to `IS_DELETED` flag bit (Tech spec) -- GPU kernel naturally skips both; compact when tombstone ratio >20%
- **FSEvents granularity**: File-level events (`kFSEventStreamCreateFlagFileEvents`) to avoid `read_dir` diff step; 500ms latency parameter for coalescing
- **Single global index**: One `global.idx` for `/` replaces per-directory indexes; eliminates cold start on project switching; search scopes at query time via path prefix
- **Flush strategy**: 1000 dirty entries OR 30s timer OR HistoryDone OR graceful shutdown -- balances write amplification against freshness
- **Performance targets**: Cold start <100ms, warm <5ms, incremental <1s, mmap load <50ms at 1M, `bytesNoCopy` <1ms

## Module Roadmap

1. **gsix-v2-format** (Priority 0) -- Foundation: v2 header layout, IS_DELETED flag, version detection, v1->v2 migration, format validation
2. **mmap-gpu-pipeline** (Priority 1) -- Zero-Copy: page-aligned mmap loading, Metal `bytesNoCopy` buffer creation, `arc-swap` atomic snapshots, IndexStore integration
3. **fsevents-watcher** (Priority 2) -- Live Updates: `fsevent-sys` FFI integration, event ID persistence/resume, ExcludeTrie filtering, FsChange event types, CFRunLoop thread
4. **incremental-updates** (Priority 3) -- Index Maintenance: IndexWriter with HashMap lookup, CRUD event processing, flush/compact strategy, atomic write persistence
5. **global-root-index** (Priority 4) -- Root Indexing: single `global.idx` for `/`, expanded exclude list, background initial build with progress, orchestrator integration, status bar UX
6. **testing-reliability** (Priority 5) -- Quality: ~146 new tests, property-based format fuzzing, benchmark suite, crash safety tests, graceful degradation
7. **integration** (Priority 999) -- Final Assembly: end-to-end pipeline validation, v1->v2 production migration, old index cleanup, performance validation against success metrics
