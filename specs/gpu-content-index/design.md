---
spec: gpu-content-index
phase: design
created: 2026-02-16
generated: auto
---

# Design: gpu-content-index

## Overview

Add a ContentStore subsystem that loads all text file contents into a contiguous, page-aligned mmap'd buffer at startup. The buffer is wrapped as a Metal buffer via `bytesNoCopy` for zero-copy GPU access. The SearchOrchestrator gains a fast-path that dispatches GPU content search directly from the content store, bypassing all per-file `fs::read()` calls. A GCIX persistent format enables instant warm restart via mmap.

## Architecture

```
+------------------------------------------------------------------+
|                     CONTENT INDEX SUBSYSTEM                      |
+------------------------------------------------------------------+
|                                                                  |
|  +---------------------+     +------------------------+          |
|  |    ContentStore      |     |   ContentIndexStore    |          |
|  |                      |     |   (ArcSwap<Option<     |          |
|  | mmap: MmapBuffer     |     |    ContentSnapshot>>)  |          |
|  | metal_buffer: MTL    |     +------------------------+          |
|  | files: Vec<Meta>     |              |                          |
|  | total_bytes: u64     |     +--------v-----------+              |
|  +----------+-----------+     | ContentSnapshot    |              |
|             |                 | store: ContentStore |              |
|             | GPU-addressable | metal_buf: MTL      |              |
|             | zero-copy       +--------------------+              |
|  +----------v-----------+                                         |
|  | ContentDaemon        |     +--------------------+              |
|  | (background builder) |---->| GCIX File (disk)   |              |
|  | FSEvents listener    |     | header + meta + data|              |
|  +----------------------+     +--------------------+              |
+------------------------------------------------------------------+
         |                              |
         v                              v
+------------------+          +---------------------+
| SearchOrchestrator|         | ContentSearchEngine |
| (fast-path check) |-------->| search_with_buffer()|
+------------------+          +---------------------+
```

## Components

### ContentStore
**Purpose**: Holds all indexed file contents in a contiguous, GPU-addressable buffer.
**File**: `gpu-search/src/index/content_store.rs` (NEW)
**Responsibilities**:
- Allocate page-aligned anonymous mmap (`MAP_ANON`) for build phase
- Track per-file metadata (offset, length, path_id, content_hash, mtime)
- Provide `content_for(file_id)` -> `&[u8]` accessor
- Provide `metal_buffer()` -> `&ProtocolObject<dyn MTLBuffer>` for GPU dispatch
- Provide `total_bytes()`, `file_count()` statistics

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct FileContentMeta {
    pub content_offset: u64,  // Byte offset into content buffer
    pub content_len: u32,     // Content length in bytes
    pub path_id: u32,         // Index into path table
    pub content_hash: u32,    // CRC32 of content
    pub mtime: u32,           // Last modification time (unix)
    pub trigram_count: u32,   // Reserved for Phase 2
    pub flags: u32,           // bit 0=is_text, bit 1=is_utf8
}
// 32 bytes per entry

pub struct ContentStore {
    mmap: MmapBuffer,                                      // Anonymous mmap (build) or file-backed (warm)
    metal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,  // GPU handle via bytesNoCopy
    files: Vec<FileContentMeta>,                            // Per-file metadata
    total_bytes: u64,                                       // Total content bytes
    capacity: usize,                                        // mmap region capacity
}
```

### ContentSnapshot
**Purpose**: Immutable snapshot bundling ContentStore + metadata, shareable via Arc.
**File**: `gpu-search/src/index/content_snapshot.rs` (NEW)
**Responsibilities**:
- Bundle ContentStore with path-to-file mapping
- Provide thread-safe read-only access to content and metadata
- Enforce drop order: Metal buffer dropped before mmap

Follows `IndexSnapshot` pattern from `src/index/snapshot.rs`.

### ContentIndexStore
**Purpose**: Lock-free atomic store for current ContentSnapshot.
**File**: `gpu-search/src/index/content_index_store.rs` (NEW)
**Responsibilities**:
- `ArcSwap<Option<ContentSnapshot>>` for wait-free readers
- `swap()` for atomic snapshot replacement
- `is_available()` check for fast-path decision

Follows `IndexStore` pattern from `src/index/store.rs`.

### ContentBuilder
**Purpose**: Builds ContentStore from filesystem in background thread.
**File**: `gpu-search/src/index/content_builder.rs` (NEW)
**Responsibilities**:
- Walk filesystem (reuse `FilesystemScanner` or `ignore` crate walker)
- Read text files, skip binary (reuse `BinaryDetector`), skip >100MB
- Allocate anonymous mmap, copy file contents sequentially
- Build `FileContentMeta` table with offsets, hashes, mtimes
- Create Metal buffer via `bytesNoCopy`
- Publish ContentSnapshot to ContentIndexStore
- Track progress via `Arc<AtomicUsize>`

### ContentDaemon
**Purpose**: Lifecycle coordinator for content index (build + FSEvents + persistence).
**File**: `gpu-search/src/index/content_daemon.rs` (NEW)
**Responsibilities**:
- On launch: check for GCIX file, mmap if valid, else trigger build
- Start background build thread via ContentBuilder
- Listen to FSEvents channel for incremental updates
- Save GCIX file after build completes
- Graceful shutdown

Follows `IndexDaemon` pattern from `src/index/daemon.rs`.

### GCIX File Format
**Purpose**: Persistent serialization of content store for warm restart.
**File**: `gpu-search/src/index/gcix.rs` (NEW)
**Responsibilities**:
- Write: header + FileContentMeta table + content data to file
- Read: validate header, mmap file, extract sections via pointer arithmetic
- Content data region starts at page-aligned offset for `bytesNoCopy`

```
GCIX v1 File Layout:
+--------------------------------------------------------------------+
| Header (16384 bytes, page-aligned)                                 |
|   magic: "GCIX", version: 1, file_count, content_bytes,           |
|   meta_offset, content_offset, root_hash, last_fsevents,          |
|   saved_at, header_crc32, [padding]                                |
+--------------------------------------------------------------------+
| FileContentMeta Table (32 bytes * file_count)                      |
+--------------------------------------------------------------------+
| [padding to page boundary]                                         |
+--------------------------------------------------------------------+
| Content Data (content_bytes, page-aligned start)                   |
|   [raw file contents, concatenated]                                |
+--------------------------------------------------------------------+
```

### Modified: ContentSearchEngine
**Purpose**: Add `search_with_buffer()` method for external buffer dispatch.
**File**: `gpu-search/src/search/content.rs` (MODIFY)
**Change**: New method accepts an external Metal buffer + chunk metadata. Sets external buffer at buffer(0) instead of internal `chunks_buffer`. No shader changes.

### Modified: SearchOrchestrator
**Purpose**: Add content store fast-path to `search_streaming_inner()`.
**File**: `gpu-search/src/search/orchestrator.rs` (MODIFY)
**Change**: Before disk-based pipeline, check if `ContentIndexStore` has a snapshot. If yes, dispatch GPU search directly from content store buffer. Fall back to disk path if content store unavailable.

## Data Flow

### Build Phase (One-Time or Warm Restart)

1. ContentDaemon checks for existing GCIX file
2. If valid GCIX: mmap file, create Metal buffer, publish ContentSnapshot
3. If no GCIX: spawn ContentBuilder background thread
4. ContentBuilder walks filesystem, reads text files, builds contiguous buffer
5. ContentBuilder creates Metal buffer via bytesNoCopy
6. ContentBuilder publishes ContentSnapshot to ContentIndexStore
7. ContentDaemon saves GCIX file for next restart

### Search Phase (Zero Disk I/O)

1. User types query -> SearchOrchestrator::search_streaming_inner()
2. Check `content_store.is_available()`
3. If available: load ContentSnapshot via arc-swap guard
4. For each file in content store: build ChunkMetadata pointing into content buffer
5. Call `engine.search_with_buffer(content_store.metal_buffer(), chunks, pattern)`
6. GPU kernel scans content buffer at buffer(0) -- same shader, different buffer source
7. Resolve matches: file_id -> FileContentMeta -> path, line number, column
8. Return ContentMatch entries

### Fallback Path (Content Store Not Ready)

1. `content_store.is_available()` returns false
2. Execute existing disk-based `search_streaming_inner()` pipeline
3. Walk files -> fs::read() -> load into chunks_buffer -> GPU dispatch

## Technical Decisions

| Decision | Options | Choice | Rationale |
|----------|---------|--------|-----------|
| Content buffer allocation | Vec<u8> vs anonymous mmap | Anonymous mmap (`MAP_ANON`) | Page-aligned for bytesNoCopy; growable; same pattern as MmapBuffer |
| Lock-free store | Mutex vs RwLock vs ArcSwap | ArcSwap | Proven pattern from IndexStore; wait-free reads |
| Persistence format | Custom binary vs serde | Custom binary (GCIX) | Must be mmap-friendly; follows GSIX v2 pattern |
| GPU dispatch change | Modify existing kernel vs new method | New `search_with_buffer()` method | Zero shader changes; backward compatible |
| Binary detection | New detector vs reuse BinaryDetector | Reuse BinaryDetector | Already handles extension + NUL heuristic |
| File size limit | Configurable vs fixed | Fixed 100MB (MAX_FILE_SIZE) | Matches existing streaming.rs constant |
| Trigram index | Include in Phase 1 vs defer | Defer to Phase 2 | Brute-force GPU scan is fast enough (<100ms for 10GB on M4) |
| Content data padding | Pad between files vs no padding | No padding | GPU kernel handles arbitrary offsets via ChunkMetadata |

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `gpu-search/src/index/content_store.rs` | Create | ContentStore struct, FileContentMeta, buffer management |
| `gpu-search/src/index/content_snapshot.rs` | Create | ContentSnapshot (immutable, Arc-shareable) |
| `gpu-search/src/index/content_index_store.rs` | Create | ContentIndexStore (ArcSwap wrapper) |
| `gpu-search/src/index/content_builder.rs` | Create | Background filesystem reader + buffer builder |
| `gpu-search/src/index/content_daemon.rs` | Create | Lifecycle coordinator (build + FSEvents + persist) |
| `gpu-search/src/index/gcix.rs` | Create | GCIX file format read/write |
| `gpu-search/src/index/mod.rs` | Modify | Add new module declarations |
| `gpu-search/src/search/content.rs` | Modify | Add `search_with_buffer()` method |
| `gpu-search/src/search/orchestrator.rs` | Modify | Add content store fast-path in search_streaming_inner |

## Error Handling

| Error | Handling | User Impact |
|-------|----------|-------------|
| Content store build fails (OOM) | Log error, continue with disk-based search | Transparent -- search works, just slower |
| GCIX file corrupt/version mismatch | Delete GCIX, trigger full rebuild | ~60s delay on restart, then normal |
| bytesNoCopy fails | Fall back to newBufferWithBytes (copy) | ~1ms/GB overhead, still much faster than disk |
| File deleted during build | Log warning, skip file, continue | One file missing from index |
| FSEvents event lost | Periodic staleness check via mtime | Brief stale results, self-healing |
| Metal command buffer error | Log, fall back to disk path for this search | One search uses disk, next retries GPU |

## Existing Patterns to Follow

- **IndexStore** (`src/index/store.rs`): ArcSwap pattern for ContentIndexStore -- identical API surface
- **IndexSnapshot** (`src/index/snapshot.rs`): Drop order (Metal buffer before mmap) for ContentSnapshot
- **IndexDaemon** (`src/index/daemon.rs`): Background builder + FSEvents + shutdown for ContentDaemon
- **MmapBuffer** (`src/io/mmap.rs`): `from_file()` for warm restart, `from_bytes()` for testing, `as_metal_buffer()` for GPU access
- **create_gpu_buffer** (`src/index/metal_buffer.rs`): bytesNoCopy creation pattern
- **BinaryDetector** (`src/search/binary.rs`): `should_skip()` during content ingestion
- **GSIX v2** (`src/index/gsix_v2.rs`): 16KB page-aligned header, CRC32 validation, magic + version
