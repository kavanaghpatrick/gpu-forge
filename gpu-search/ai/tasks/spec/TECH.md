# Technical Architecture: Persistent GPU-Friendly File Index

**Author**: Technical Architect Agent
**Date**: 2026-02-14
**Status**: Final Analysis
**Scope**: Global persistent file index with FSEvents incremental updates, GPU zero-copy pipeline

---

## 1. Architecture

### 1.1 Component Diagram

```
+-------------------------------------------------------------------+
|                        gpu-search process                         |
|                                                                   |
|  +---------------------+     +------------------------------+    |
|  |   IndexDaemon        |     |   SearchPipeline             |    |
|  |                     |     |                              |    |
|  |  FSEventsListener   |     |  IndexLoader (mmap)          |    |
|  |    |                |     |    |                          |    |
|  |    v                |     |    v                          |    |
|  |  EventProcessor     |     |  MmapBuffer (page-aligned)   |    |
|  |    |                |     |    |                          |    |
|  |    v                |     |    v                          |    |
|  |  IndexWriter -------+---->|  Metal bytesNoCopy           |    |
|  |    |                |     |    |                          |    |
|  |    v                |     |    v                          |    |
|  |  IndexStore (GSIX)  |     |  GPU path_filter kernel      |    |
|  +---------------------+     +------------------------------+    |
+-------------------------------------------------------------------+
                |
                v
   ~/.gpu-search/index/global.idx  (GSIX v2 file, single global index)
```

### 1.2 Data Flow

**Startup (cache hit)**:
1. `IndexLoader` opens `~/.gpu-search/index/global.idx`.
2. Reads the GSIX v2 header, validates magic/version/checksum.
3. `mmap(PROT_READ, MAP_PRIVATE)` the entire file. Pointer is page-aligned (16KB on Apple Silicon).
4. Entry region starts at file offset 16384 (one full page after header). This offset is itself page-aligned.
5. `MTLDevice::makeBuffer(bytesNoCopy:)` wraps the entry region pointer directly. No memcpy from disk to GPU.
6. `IndexDaemon` starts FSEvents stream with `sinceWhen = header.last_fsevents_id`. Historical events replay, then live monitoring begins.
7. Search is immediately available -- GPU buffer is ready before FSEvents finishes replaying history.

**Startup (cache miss / first run)**:
1. No `global.idx` found. `FilesystemScanner::scan("/")` performs a full parallel walk using the `ignore` crate.
2. Scan takes 10-30s for a typical macOS system (500K-1.5M files).
3. Entries are sorted by path, written to `global.idx` in GSIX v2 format.
4. `IndexDaemon` starts FSEvents with `sinceWhen = kFSEventStreamEventIdSinceNow`.
5. Subsequent launches use the stored event ID, avoiding full rescans.

**Incremental update (runtime)**:
1. FSEvents callback receives a batch of changed paths (debounced at 500ms by FSEvents latency parameter).
2. `EventProcessor` filters paths through `ExcludeTrie`, maps events to `FsChange` variants (Created/Modified/Deleted/Renamed/MustRescan).
3. `IndexWriter` applies changes to in-memory `Vec<GpuPathEntry>` + `HashMap<path, index>` for O(1) lookup.
4. Every 30s or 1000+ dirty entries, `IndexWriter` flushes: compact tombstoned entries, sort, write atomically, update `last_fsevents_id` in header.
5. `IndexStore` swaps the `Arc<IndexSnapshot>` atomically. In-flight search queries hold the old snapshot until they complete.

### 1.3 Key Invariants

- Exactly one global index file: `~/.gpu-search/index/global.idx`
- Index always covers `/` with the configured exclude list
- Entry region starts at 16KB page boundary (file offset 16384)
- Writers hold exclusive file lock (`flock(LOCK_EX)`) during flush; readers use mmap snapshots (copy-on-write semantics via `MAP_PRIVATE`)
- FSEvents event IDs are monotonically increasing and persisted in the GSIX v2 header
- Entries are sorted by path bytes for binary search lookups

---

## 2. GSIX v2 Format

### 2.1 Header (16384 bytes, padded to fill one Apple Silicon page)

The v1 header is 64 bytes with entries starting immediately after. This creates two problems: (a) the entry region at offset 64 is not page-aligned, preventing direct `bytesNoCopy` mapping of just the entries; (b) there is no space for FSEvents state or capacity fields.

GSIX v2 expands the header to occupy a full 16KB page. The entry region then begins at offset 16384, which is page-aligned by construction.

```
Offset  Size    Field                Type     Description
------  ------  -------------------  -------  -------------------------------------------
0       4       magic                u32      0x58495347 ("GSIX" LE) -- unchanged
4       4       version              u32      2 (was 1)
8       4       entry_count          u32      Number of live (non-tombstoned) entries
12      4       root_hash            u32      cache_key("/") -- fixed for global index
16      8       saved_at             u64      Unix timestamp of last flush
24      8       last_fsevents_id     u64      FSEventStreamEventId for resume on restart
32      4       exclude_hash         u32      CRC32 of sorted exclude list config
36      4       entry_capacity       u32      Total allocated entry slots (>= entry_count)
40      4       flags                u32      Bit 0: sorted, Bit 1: compacted
44      4       checksum             u32      CRC32 of header bytes [0..44)
48      16336   _reserved            [u8]     Zero-filled to pad header to 16384 bytes
```

Rust definition:

```rust
const HEADER_SIZE_V2: usize = 16384; // One Apple Silicon page

#[repr(C)]
struct GsixHeaderV2 {
    magic: u32,              // 0x58495347
    version: u32,            // 2
    entry_count: u32,
    root_hash: u32,
    saved_at: u64,
    last_fsevents_id: u64,
    exclude_hash: u32,
    entry_capacity: u32,
    flags: u32,
    checksum: u32,
    _reserved: [u8; 16336],
}

const _: () = assert!(std::mem::size_of::<GsixHeaderV2>() == 16384);
const _: () = assert!(HEADER_SIZE_V2 % crate::io::mmap::PAGE_SIZE == 0);
```

### 2.2 Entry Region

Immediately after the header at file offset 16384. Contains `entry_capacity` slots of 256 bytes each. Live entries occupy the first `entry_count` slots. Slots from `entry_count` to `entry_capacity - 1` are zeroed (available for growth without reallocation).

```
File Offset            Size              Content
-----------            ------            -------
16384                  256               GpuPathEntry[0]
16640                  256               GpuPathEntry[1]
...
16384 + (N-1)*256      256               GpuPathEntry[N-1]   (N = entry_count)
16384 + N*256          256               [zeroed slot]        (growth room)
...
16384 + (C-1)*256      256               [zeroed slot]        (C = entry_capacity)
```

### 2.3 GpuPathEntry (unchanged layout, new flag)

The `GpuPathEntry` struct remains 256 bytes with identical field layout. One new flag is added:

```rust
pub mod path_flags {
    pub const IS_DIR: u32        = 1 << 0;
    pub const IS_HIDDEN: u32     = 1 << 1;
    pub const IS_SYMLINK: u32    = 1 << 2;
    pub const IS_EXECUTABLE: u32 = 1 << 3;
    pub const IS_DELETED: u32    = 1 << 4;  // NEW: tombstone for incremental deletes
}
```

`IS_DELETED` marks entries awaiting compaction. The GPU `path_filter` kernel must check `!(flags & IS_DELETED)` before returning a match. This is a single bitwise AND in the shader -- negligible cost.

### 2.4 File Size Estimates

| Entries | Capacity (1.2x) | Header | Entry Data | Total File Size |
|---------|-----------------|--------|------------|-----------------|
| 500K    | 600K            | 16 KB  | 146 MB     | ~146 MB         |
| 1M      | 1.2M            | 16 KB  | 293 MB     | ~293 MB         |
| 1.5M    | 1.8M            | 16 KB  | 439 MB     | ~440 MB         |
| 2M      | 2.4M            | 16 KB  | 586 MB     | ~586 MB         |

### 2.5 Backward Compatibility (v1 to v2)

The `version` field at offset 4 distinguishes formats:
- `version == 1`: v1 format, 64-byte header, entries at offset 64.
- `version == 2`: v2 format, 16384-byte header, entries at offset 16384.

When `MmapIndexCache::load_mmap()` encounters version 1, it returns `CacheError::InvalidFormat`. The caller (`GpuIndexLoader::load_global()`) triggers a full rebuild producing a v2 file. No data migration is needed since `GpuPathEntry` layout is identical between versions.

Old per-directory v1 index files (`~/.gpu-search/index/<hash>.idx`) are left in place but unused. A one-time cleanup routine deletes them after the first successful v2 build.

---

## 3. FSEvents Integration

### 3.1 Why Replace notify

The current `IndexWatcher` (in `src/index/watcher.rs`) uses `notify` v7 with `notify-debouncer-mini`. On macOS, `notify::RecommendedWatcher` does use FSEvents as its backend. However, `notify` abstracts away the `FSEventStreamEventId`, making it impossible to:

1. **Persist the event ID** in the GSIX header for cross-restart resume.
2. **Request historical events** since a specific ID on next launch.
3. **Detect `HistoryDone`** to know when replay is complete.

These three capabilities are the entire value of FSEvents for a persistent index. Without them, every process restart requires a full rescan.

### 3.2 Crate Selection: fsevent-sys

Use `fsevent-sys` v4.x for raw FFI bindings to CoreServices `FSEventStream*` functions. This provides direct access to:

- `FSEventStreamCreate()` with `sinceWhen` parameter (the event ID)
- `FSEventStreamGetLatestEventId()` for persistence
- `kFSEventStreamCreateFlagFileEvents` for file-level granularity
- `kFSEventStreamEventFlagHistoryDone` for replay completion detection
- `kFSEventStreamEventFlagMustScanSubDirs` for coalesced rescan requests

The `fsevent-sys` crate has a deprecation notice suggesting `objc2-core-services` as a future replacement. Since `objc2-core-services` does not yet expose FSEvents stream APIs at feature parity, `fsevent-sys` is the pragmatic choice. Pin at `fsevent-sys = "4"` and plan migration when the objc2 ecosystem covers FSEvents.

Additionally, `core-foundation = "0.10"` is needed for `CFRunLoop` scheduling of the FSEvents stream.

### 3.3 FSEventsListener Design

```rust
pub struct FSEventsListener {
    /// CFRunLoop thread handle
    runloop_thread: Option<JoinHandle<()>>,
    /// Sender to dispatch FsChange events to the IndexWriter
    change_tx: crossbeam_channel::Sender<FsChange>,
    /// Last processed event ID (updated atomically from callback)
    last_event_id: Arc<AtomicU64>,
    /// Compiled exclude rules
    excludes: Arc<ExcludeTrie>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

/// Filesystem change events produced by FSEvents.
pub enum FsChange {
    /// File or directory created at path
    Created(PathBuf),
    /// File modified (content or metadata changed)
    Modified(PathBuf),
    /// File or directory removed
    Deleted(PathBuf),
    /// File renamed (old_path -> new_path)
    Renamed { old: PathBuf, new: PathBuf },
    /// FSEvents lost fine-grained tracking; must rescan this subtree
    MustRescan(PathBuf),
    /// All historical events since stored ID have been replayed
    HistoryDone,
}
```

### 3.4 Startup and Resume Sequence

1. Read `last_fsevents_id` from GSIX v2 header. If no index exists, use `kFSEventStreamEventIdSinceNow` (value `0xFFFFFFFFFFFFFFFF`).

2. Create `FSEventStreamRef`:
   ```
   paths: ["/"]
   sinceWhen: last_fsevents_id
   latency: 0.5 seconds
   flags: kFSEventStreamCreateFlagFileEvents
        | kFSEventStreamCreateFlagUseCFTypes
        | kFSEventStreamCreateFlagNoDefer
   ```

3. Spawn a dedicated thread running `CFRunLoopRun()`. Schedule the FSEvents stream on this thread's run loop.

4. FSEvents replays all events since `sinceWhen`:
   - Each event batch arrives in the callback with paths, flags, and event IDs.
   - The callback filters each path through `ExcludeTrie::should_exclude()`.
   - Non-excluded paths are mapped to `FsChange` variants based on `FSEventStreamEventFlags`.
   - Changes are sent via `change_tx` to the `IndexWriter` thread.

5. When `kFSEventStreamEventFlagHistoryDone` is received, send `FsChange::HistoryDone`. This tells the IndexWriter that replay is complete and live monitoring has begun.

6. After each callback batch, update `last_event_id` atomically:
   ```rust
   self.last_event_id.store(latest_id, Ordering::Release);
   ```

### 3.5 Event Flag Mapping

| FSEvents Flag | FsChange Variant | Notes |
|---------------|-----------------|-------|
| `ItemCreated` | `Created(path)` | New file or directory |
| `ItemModified` | `Modified(path)` | Content or metadata change |
| `ItemRemoved` | `Deleted(path)` | File or directory removed |
| `ItemRenamed` | `Renamed{old, new}` | Paired events with `ItemRenamed` flag |
| `MustScanSubDirs` | `MustRescan(path)` | Events were coalesced; subtree state is unknown |
| `HistoryDone` | `HistoryDone` | Replay complete, switching to live |
| `RootChanged` | `MustRescan("/")` | Watched root changed (unlikely for "/") |

### 3.6 Event ID Persistence

The `last_event_id` (an `AtomicU64`) is read during flush and written into `GsixHeaderV2.last_fsevents_id`. On unclean shutdown, at most 0.5 seconds of events are re-replayed on next launch (bounded by the FSEvents latency parameter). This is safe because event processing is idempotent: re-processing a Create for an already-indexed file results in a no-op stat check, and re-processing a Delete for an already-tombstoned entry is a no-op.

---

## 4. Incremental Update Strategy

### 4.1 In-Memory Representation

The `IndexWriter` maintains a mutable entry array and a path-to-index lookup map:

```rust
pub struct IndexWriter {
    /// Entry array, sorted by path bytes. Matches on-disk layout.
    entries: Vec<GpuPathEntry>,
    /// Path -> index in entries Vec for O(1) event processing.
    /// Key is the path bytes (avoiding PathBuf allocation).
    path_index: HashMap<Box<[u8]>, usize>,
    /// Number of mutations since last flush.
    dirty_count: usize,
    /// Timestamp of last flush.
    last_flush: Instant,
    /// Exclude trie (shared with FSEventsListener).
    excludes: Arc<ExcludeTrie>,
}
```

The `HashMap` provides O(1) amortized lookup for event processing. For 1M entries, the HashMap consumes approximately 48MB (key: ~80 bytes avg path + value: 8 bytes usize + HashMap overhead per entry ~24 bytes = ~112 bytes/entry). Combined with the 256MB entry array, total writer memory is ~304MB.

### 4.2 Event Processing Logic

For each `FsChange` received from the FSEvents listener:

**Created(path)**:
1. Check `excludes.should_exclude(&path)` -- skip if excluded.
2. `stat()` the path. If stat fails (race: file already deleted), skip.
3. Build a `GpuPathEntry` from stat metadata.
4. Check `path_index` for existing entry:
   - If found: treat as `Modified` (update in place).
   - If not found: append to `entries`, record in `path_index`, mark as unsorted.
5. Increment `dirty_count`.

**Modified(path)**:
1. Look up in `path_index`. If not found, treat as `Created`.
2. `stat()` the path. If stat fails, treat as `Deleted`.
3. Update `mtime`, `size_lo`, `size_hi`, `flags` in place at the existing index.
4. Increment `dirty_count`.

**Deleted(path)**:
1. Look up in `path_index`. If not found, skip (already deleted or never indexed).
2. Set `entries[idx].flags |= IS_DELETED` (tombstone).
3. Remove from `path_index`.
4. Increment `dirty_count`.

**Renamed { old, new }**:
1. Process as `Deleted(old)` followed by `Created(new)`.

**MustRescan(subtree)**:
1. Walk the subtree using `FilesystemScanner::scan()`.
2. Collect the set of paths found.
3. For each existing entry whose path starts with `subtree`:
   - If path is in the found set: update metadata (Modified).
   - If path is not in the found set: tombstone (Deleted).
4. For each found path not in `path_index`: insert (Created).
5. This is the expensive path. For a typical subtree (e.g., a project directory), this processes 1K-10K paths in 10-100ms.

**HistoryDone**:
1. Log the number of changes replayed.
2. Trigger an immediate flush (persists the caught-up state).

### 4.3 Flush Strategy

Flush when any condition is met:
- `dirty_count >= 1000`
- `last_flush.elapsed() >= 30 seconds`
- `FsChange::HistoryDone` received
- Graceful shutdown signal (SIGTERM / SIGINT)

Flush procedure:
1. **Compact**: Remove entries where `flags & IS_DELETED != 0`. Shift remaining entries to close gaps. Update `path_index` with new positions.
2. **Sort**: If any entries were appended (unsorted), sort the entire array by path bytes. Sorting 1M entries: ~200ms.
3. **Write atomically**: Write to `global.idx.tmp` (header + entries), `fsync()`, then `rename()` to `global.idx`. The rename is atomic on APFS.
4. **Update header**: Set `entry_count`, `entry_capacity = max(entry_count * 1.2, old_capacity)`, `last_fsevents_id`, `saved_at`, recompute `checksum`.
5. **Notify IndexStore**: After the new file is on disk, the IndexStore creates a new `IndexSnapshot` (mmap + Metal buffer) and swaps it atomically.
6. Reset `dirty_count = 0`, `last_flush = Instant::now()`.

### 4.4 Amortized Costs

For a typical development session (~100 file changes/minute):

| Operation | Per-Event Cost | Frequency | Total/Minute |
|-----------|---------------|-----------|-------------|
| Lookup (HashMap) | O(1), ~50ns | 100/min | ~5us |
| Update in-place | O(1), ~100ns | 80/min | ~8us |
| Append (unsorted) | O(1), ~200ns | 15/min | ~3us |
| Tombstone | O(1), ~50ns | 5/min | ~0.25us |
| Flush (sort + write) | O(n log n), ~500ms | 2/min | ~1s |

The flush is the dominant cost at ~500ms every 30 seconds. This runs on a background thread and does not block search queries (readers hold the old snapshot via Arc).

---

## 5. Global Root Index

### 5.1 Single Index for `/`

The system maintains exactly one index file: `~/.gpu-search/index/global.idx`. All searches, regardless of the user's working directory, use this global index. The search pipeline filters results to the user's scope (e.g., current project) at query time via the existing `GpuPathEntry.path` prefix check in the Metal kernel.

### 5.2 Rationale

| Factor | Per-Directory Indexes (v1) | Global Root Index (v2) |
|--------|---------------------------|----------------------|
| Cold start | Re-scan each new project | Zero -- already indexed |
| Cross-project search | Not possible | Native |
| FSEvents complexity | One stream per watched dir | One stream for `/` |
| Disk usage | N small files | One large file |
| Memory usage | Load/unload per dir | Fixed ~256-440MB |
| Update complexity | Per-dir staleness checks | Single FSEvents resume |

The primary driver is eliminating cold start latency. Switching to a new project directory with v1 requires a full rescan (10-30s for large repos). With the global index, every path on the system is already indexed.

### 5.3 Cache Key

```rust
pub fn global_cache_key() -> u32 {
    let mut hasher = DefaultHasher::new();
    "/".hash(&mut hasher);
    hasher.finish() as u32
}

pub fn global_index_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".gpu-search")
        .join("index")
        .join("global.idx")
}
```

The per-directory `cache_key()` and `index_path_for()` functions in `cache.rs` remain for backward compatibility but are no longer called by the main search flow.

### 5.4 First-Run Experience

On first launch (no `global.idx` exists):
1. Display "Building file index... (one-time setup)" in the status bar.
2. Run `FilesystemScanner::scan("/")` with the full exclude list.
3. Show a progress indicator based on directory count (approximate).
4. Expected duration: 10-30 seconds on a typical Mac.
5. After completion, all subsequent launches start in <1 second.

---

## 6. Page Alignment

### 6.1 Apple Silicon Page Size

Apple Silicon (M1 through M5) uses a 16KB (16384-byte) hardware page size. This is confirmed by:
- `sysctl hw.pagesize` returning 16384
- The existing `src/io/mmap.rs` defining `PAGE_SIZE = 16384` and `align_to_page()` correctly

The `MmapBuffer::from_file()` in the codebase already uses `libc::mmap()` with `MAP_PRIVATE`, which returns a page-aligned base address.

### 6.2 bytesNoCopy Alignment Requirements

`MTLDevice::makeBuffer(bytesNoCopy:length:options:deallocator:)` requires:
1. **Pointer**: Must be page-aligned (multiple of `vm_page_size` = 16384 on Apple Silicon).
2. **Length**: Must be a multiple of `vm_page_size`.
3. **Allocation**: Must be via `mmap()` or `vm_allocate()`. Standard `malloc()` does not guarantee page alignment.

Source: Apple Developer Forums confirm that `newBufferWithBytesNoCopy` fails with assertion `"pointer is not 4096 byte aligned"` on Intel (4KB pages) and would similarly fail with 16KB misalignment on Apple Silicon.

### 6.3 GSIX v2 Alignment Strategy

The v2 header is exactly 16384 bytes. Since:
- `mmap()` returns a page-aligned base address (call it `base`)
- `base` is at file offset 0
- The entry region is at file offset 16384 = `base + 16384`
- `base + 16384` is page-aligned because `base` is page-aligned and 16384 is a multiple of `PAGE_SIZE`

Therefore the entry region pointer is inherently page-aligned. No additional alignment logic is needed beyond using the 16KB header.

### 6.4 Metal Buffer Creation Strategies

**Strategy A: Full-file mmap with kernel offset (recommended)**:
```rust
let mmap = MmapBuffer::from_file(&global_index_path)?;
let buffer = mmap.as_metal_buffer(&device);
// GPU kernel accesses entries at buffer offset 16384
encoder.setBuffer(buffer, offset: HEADER_SIZE_V2 as u64, index: 0);
```

This is the simplest approach. The 16KB header is included in the Metal buffer but the kernel skips it via the offset parameter. The overhead of mapping an extra 16KB is negligible (<0.01% of a 256MB buffer).

**Strategy B: Offset mmap of entry region only**:
```rust
let fd = file.as_raw_fd();
let entry_len = align_to_page(entry_count * 256);
let ptr = libc::mmap(null_mut(), entry_len, PROT_READ, MAP_PRIVATE, fd, HEADER_SIZE_V2 as i64);
// ptr is page-aligned, entry data starts at ptr[0]
let buffer = device.newBufferWithBytesNoCopy(ptr, entry_len, options, None);
```

This avoids mapping the header but requires managing a second mmap region. More complex, minimal benefit.

**Decision**: Use Strategy A. The kernel offset approach is well-supported by Metal (every `setBuffer` call accepts an offset parameter) and avoids complexity.

### 6.5 Compile-Time and Runtime Verification

```rust
// Compile-time
const _: () = assert!(HEADER_SIZE_V2 == 16384);
const _: () = assert!(HEADER_SIZE_V2 % 16384 == 0);
const _: () = assert!(std::mem::size_of::<GpuPathEntry>() == 256);

// Runtime (in load path)
fn verify_alignment(mmap_ptr: *const u8) {
    let entry_ptr = unsafe { mmap_ptr.add(HEADER_SIZE_V2) };
    assert!(
        entry_ptr as usize % PAGE_SIZE == 0,
        "Entry region at offset {} is not page-aligned (ptr={:p})",
        HEADER_SIZE_V2, entry_ptr
    );
}
```

---

## 7. Concurrency

### 7.1 Problem

The `IndexDaemon` (writer) flushes the index periodically, while the `SearchPipeline` (reader) needs concurrent access. Readers access the mmap'd Metal buffer; writers produce a new file. These operations must not conflict.

### 7.2 Solution: Snapshot + Atomic Swap (Lock-Free Reads)

Use the `arc-swap` crate for lock-free atomic swapping of `Arc<IndexSnapshot>`:

```rust
use arc_swap::ArcSwap;

pub struct IndexSnapshot {
    /// mmap of the current GSIX v2 file (keeps mapping alive)
    mmap: MmapBuffer,
    /// Metal buffer wrapping the mmap entry region (zero-copy)
    metal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Number of live entries
    entry_count: usize,
    /// FSEvents ID at time of this snapshot
    fsevents_id: u64,
}

pub struct IndexStore {
    /// Current snapshot, swapped atomically on each flush
    current: ArcSwap<IndexSnapshot>,
    /// Path to global.idx
    path: PathBuf,
}

impl IndexStore {
    /// Readers call this. Returns an Arc snapshot -- wait-free, no locks.
    pub fn snapshot(&self) -> arc_swap::Guard<Arc<IndexSnapshot>> {
        self.current.load()
    }

    /// Writer calls this after flushing a new global.idx.
    pub fn swap(&self, new_snapshot: Arc<IndexSnapshot>) {
        self.current.store(new_snapshot);
        // Old snapshot stays alive until all reader Guards are dropped
    }
}
```

**Reader path** (search thread):
```rust
let snap = index_store.snapshot();
let buffer = &snap.metal_buffer;
let count = snap.entry_count;
// Use buffer for Metal dispatch. Snapshot is pinned for query duration.
// When snap goes out of scope, Arc ref count decrements.
```

**Writer path** (IndexWriter flush):
```rust
// 1. Write new global.idx.tmp, fsync, rename to global.idx
// 2. mmap the new file
let mmap = MmapBuffer::from_file(&global_index_path)?;
let metal_buffer = mmap.as_metal_buffer(&device);
let new_snap = Arc::new(IndexSnapshot {
    mmap,
    metal_buffer,
    entry_count: header.entry_count as usize,
    fsevents_id: header.last_fsevents_id,
});
// 3. Atomic swap -- readers still using old snapshot are unaffected
index_store.swap(new_snap);
```

### 7.3 Why Not Alternatives

**RwLock**: Blocks readers during writes. A flush takes 300-500ms for large indexes. This would cause search latency stalls of up to 500ms.

**Double buffering with manual swap**: Essentially what `ArcSwap` does, but with manual memory management and no protection against use-after-free. `ArcSwap` handles the reference counting correctly.

**Append-only WAL**: Would avoid full rewrites but the GPU kernel requires a contiguous array of 256B entries, not a log of mutations. Reconstructing the array from a WAL on each search query would negate the zero-copy benefit.

### 7.4 New Dependency

```toml
arc-swap = "1"  # Lock-free atomic Arc swap
```

---

## 8. Memory Budget

### 8.1 On-Disk and mmap Resident Set

| Entries | Entry Data (disk/mmap) | HashMap (writer only) | Total RSS |
|---------|----------------------|----------------------|-----------|
| 500K    | 128 MB               | ~24 MB               | ~152 MB   |
| 1M      | 256 MB               | ~48 MB               | ~304 MB   |
| 1.5M    | 384 MB               | ~72 MB               | ~456 MB   |
| 2M      | 512 MB               | ~96 MB               | ~608 MB   |

The HashMap is only present in the writer thread. Readers access the mmap'd entry data only (no HashMap).

### 8.2 Unified Memory Considerations

Apple Silicon uses Unified Memory Architecture (UMA). A Metal buffer created via `bytesNoCopy` shares the same physical pages as the mmap. There is no CPU copy + GPU copy -- the 256MB is counted once.

System memory by Apple Silicon generation:
- M1/M2 base: 8 GB -- index is 3.2% (256MB) to 4.8% (384MB)
- M1/M2 Pro: 16-32 GB -- index is 0.8-1.6%
- M3/M4 Pro: 18-48 GB
- M4 Max/Ultra: 64-192 GB

For 8GB systems, 256-384MB is significant but manageable because:
- mmap pages are reclaimable by macOS under memory pressure (the kernel can evict clean mmap pages and re-fault from disk on next access)
- `madvise(MADV_SEQUENTIAL)` during scan reduces working set
- The index is read-only during search (no dirty pages, no writeback cost)
- macOS memory compression (typical 2:1 ratio on path data) further reduces physical footprint

### 8.3 Metal Buffer Size Limits

Per Apple Metal Feature Set Tables, Apple Silicon GPUs support buffer sizes up to at least 50% of unified memory (minimum ~4GB on 8GB systems). A 512MB index is well within limits on all configurations.

### 8.4 Recommended Defaults

```rust
/// Default entry capacity (1.5M entries, ~384MB)
pub const DEFAULT_ENTRY_CAPACITY: u32 = 1_500_000;

/// Maximum entry capacity (4M entries, ~1GB)
pub const MAX_ENTRY_CAPACITY: u32 = 4_000_000;

/// Warning threshold -- log if entry count exceeds this
pub const ENTRY_COUNT_WARNING: u32 = 2_000_000;
```

The 1.5M default accommodates a typical developer Mac (500K-1.2M files after exclusions). The 4M maximum supports power users with extensive codebases or relaxed exclusion lists.

---

## 9. Exclude List

### 9.1 Default Exclusion Rules

Scanning from `/` requires aggressive exclusion to avoid indexing system directories, caches, and build artifacts. The exclude list is divided into tiers:

**Tier 1 -- Absolute path prefixes (system directories)**:
```
/System
/Library/Caches
/private/var
/private/tmp
/Volumes
/dev
```

**Tier 2 -- User-relative prefixes (expanded from `$HOME`)**:
```
~/Library/Caches
~/.Trash
~/Library/Application Support/*/Caches
```

**Tier 3 -- Directory name matches (anywhere in path)**:
```
.git
node_modules
target
__pycache__
vendor
dist
build
.cache
venv
.venv
.Spotlight-V100
.fseventsd
.DS_Store
.Trashes
.hg
.svn
.idea
.vscode
```

### 9.2 ExcludeTrie Implementation

```rust
pub struct ExcludeTrie {
    /// Absolute path prefixes (e.g., "/System", "/private/var")
    absolute_prefixes: Vec<Box<[u8]>>,
    /// Directory basename matches (e.g., "node_modules", ".git")
    dir_names: HashSet<Box<[u8]>>,
    /// Optional include overrides (whitelist within excluded subtrees)
    include_overrides: Vec<Box<[u8]>>,
}

impl ExcludeTrie {
    /// Check if a path should be excluded from indexing.
    /// Returns true if excluded, false if the path should be indexed.
    pub fn should_exclude(&self, path: &[u8]) -> bool {
        // Check include overrides first (whitelist takes priority)
        for override_prefix in &self.include_overrides {
            if path.starts_with(override_prefix) {
                return false;
            }
        }

        // Check absolute prefixes (fast, few entries)
        for prefix in &self.absolute_prefixes {
            if path.starts_with(prefix) {
                return true;
            }
        }

        // Check if any path component matches a dir_name
        for component in path.split(|&b| b == b'/') {
            if !component.is_empty() && self.dir_names.contains(component) {
                return true;
            }
        }

        false
    }
}
```

### 9.3 Performance

For 1M paths at ~80 bytes average:
- Absolute prefix check: ~6 prefix comparisons per path. First-byte mismatch short-circuits. Cost: <1ms for 1M paths.
- Dir name check: `HashSet` lookup per path component. Average 4 components per path = 4M hash lookups. O(1) each. Cost: ~5ms for 1M paths.
- Total exclude filtering: <10ms for 1M paths (invoked during scan, not during search).

During FSEvents processing, `should_exclude()` is called once per event. At 100 events/minute, cost is negligible.

### 9.4 Exclude Hash for Cache Invalidation

```rust
fn compute_exclude_hash(excludes: &ExcludeTrie) -> u32 {
    let mut all_patterns: Vec<&[u8]> = Vec::new();
    for p in &excludes.absolute_prefixes {
        all_patterns.push(p);
    }
    for d in &excludes.dir_names {
        all_patterns.push(d);
    }
    all_patterns.sort();

    let mut hasher = crc32fast::Hasher::new();
    for pattern in all_patterns {
        hasher.update(pattern);
        hasher.update(b"\0");
    }
    hasher.finalize()
}
```

The `exclude_hash` is stored in the GSIX v2 header. On startup, if the computed exclude hash differs from the stored one (user changed exclude configuration), the entire index is rebuilt.

### 9.5 Runtime Configuration

A configuration file at `~/.gpu-search/config.json` allows user customization:

```json
{
    "exclude_dirs": [".git", "node_modules", "target"],
    "exclude_absolute": ["/System", "/Volumes"],
    "include_override": ["/Volumes/Code"]
}
```

If no config file exists, the hardcoded defaults from Section 9.1 apply. The `include_override` array whitelists specific paths within otherwise excluded subtrees (e.g., allowing `/Volumes/Code` while excluding all other volumes).

---

## 10. Migration

### 10.1 Version Detection

```rust
fn detect_version(data: &[u8]) -> Result<u32, CacheError> {
    if data.len() < 8 {
        return Err(CacheError::InvalidFormat("File too short for header".into()));
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != INDEX_MAGIC {
        return Err(CacheError::InvalidFormat(format!(
            "Bad magic: 0x{:08X}", magic
        )));
    }
    Ok(u32::from_le_bytes(data[4..8].try_into().unwrap()))
}
```

### 10.2 v1 to v2 Upgrade Path

There is no in-place migration. The v2 format shifts the entry region from offset 64 to offset 16384, making all entry positions incompatible. The upgrade is a full rebuild:

1. On startup, `GpuIndexLoader::load_global()` checks for `global.idx`.
2. If found with `version == 2`: load normally (mmap + Metal buffer).
3. If found with `version == 1`: log warning, trigger full rebuild.
4. If not found: trigger full rebuild (first run).
5. Full rebuild: `FilesystemScanner::scan("/")` + write v2 `global.idx` + start FSEvents with `kFSEventStreamEventIdSinceNow`.

```rust
impl GpuIndexLoader {
    pub fn load_global(&self) -> Result<GpuLoadedIndex, GpuLoaderError> {
        let path = global_index_path();

        if path.exists() {
            let header_bytes = std::fs::read(&path)?;
            match detect_version(&header_bytes) {
                Ok(2) => {
                    // Validate checksum, load via mmap
                    return self.load_v2_mmap(&path);
                }
                Ok(1) => {
                    eprintln!("[IndexLoader] v1 index detected, rebuilding as v2");
                }
                Ok(v) => {
                    eprintln!("[IndexLoader] Unknown version {v}, rebuilding");
                }
                Err(e) => {
                    eprintln!("[IndexLoader] Corrupt index: {e}, rebuilding");
                }
            }
        }

        self.full_rebuild_from_root()
    }
}
```

### 10.3 Cleanup of v1 Files

After a successful v2 build, old per-directory v1 index files are cleaned up:

```rust
fn cleanup_v1_indexes(index_dir: &Path) {
    let marker = index_dir.join(".v2-migrated");
    if marker.exists() {
        return; // Already cleaned
    }
    if let Ok(entries) = std::fs::read_dir(index_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension() == Some("idx".as_ref())
                && path.file_stem() != Some("global".as_ref())
            {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    let _ = std::fs::write(&marker, "migrated to gsix v2\n");
}
```

---

## Appendix A: Dependency Changes

### Add to `[dependencies]`

```toml
fsevent-sys = "4"           # Raw macOS FSEvents FFI bindings (sinceWhen, event IDs)
core-foundation = "0.10"    # CFRunLoop for FSEvents stream scheduling
arc-swap = "1"              # Lock-free atomic Arc swap for reader/writer snapshots
crc32fast = "1"             # CRC32 for header checksum and exclude_hash
```

### Remove from `[dependencies]`

```toml
# notify = "7"              # Replaced by direct FSEvents via fsevent-sys
# notify-debouncer-mini = "0.5"  # No longer needed (FSEvents has native latency param)
```

Net dependency change: +4 added, -2 removed.

## Appendix B: Source File Changes

### New Files

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/index/fsevents.rs` | `FSEventsListener`, `FsChange` enum, CFRunLoop thread management | ~300 |
| `src/index/daemon.rs` | `IndexDaemon` orchestration, startup/shutdown, thread lifecycle | ~200 |
| `src/index/index_store.rs` | `IndexStore` with `ArcSwap<IndexSnapshot>`, snapshot/swap API | ~150 |
| `src/index/index_writer.rs` | `IndexWriter`, incremental update logic, flush/compact/sort | ~400 |
| `src/index/exclude.rs` | `ExcludeTrie`, default exclude list, `exclude_hash` computation | ~200 |

### Modified Files

| File | Changes |
|------|---------|
| `src/index/mod.rs` | Add modules: `fsevents`, `daemon`, `index_store`, `index_writer`, `exclude` |
| `src/index/shared_index.rs` | Add `GsixHeaderV2` struct, `HEADER_SIZE_V2 = 16384`, v2 write logic |
| `src/index/cache.rs` | Support v2 header parsing, entry offset at 16384, version detection |
| `src/index/gpu_loader.rs` | Add `load_global()` entry point, v1/v2 detection, full rebuild path |
| `src/gpu/types.rs` | Add `IS_DELETED = 1 << 4` to `path_flags` module |

### Unchanged Files

| File | Reason |
|------|--------|
| `src/index/scanner.rs` | Used for initial full scan from `/`. No API changes. |
| `src/index/gpu_index.rs` | `GpuResidentIndex` API unchanged. |
| `src/io/mmap.rs` | `MmapBuffer` already correct for 16KB page alignment. |
| `src/search/*` | Search pipeline uses `IndexSnapshot` buffer; no changes needed. |
| `src/ui/*` | UI consumes search results; unaffected by index changes. |

### Deprecated Files

| File | Replacement |
|------|------------|
| `src/index/watcher.rs` | Replaced by `fsevents.rs` + `daemon.rs`. Keep for one release cycle, then remove. |

## Appendix C: GPU Forge KB References

| KB ID | Topic | Relevance |
|-------|-------|-----------|
| #1264 | mmap zero-copy pipeline | Validates mmap -> Metal bytesNoCopy approach |
| #1281 | bytesNoCopy page alignment | Confirms 16KB alignment requirement on Apple Silicon |
| #1316 | Metal IO command buffers | Alternative async file loading (not used, mmap is simpler) |
| #1349 | SLC cache behavior | Shared storage mode buffers are cached in SLC on Apple Silicon |

## Appendix D: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| FSEvents drops events under heavy I/O load | Low | Medium | `MustScanSubDirs` flag triggers subtree rescan; periodic full-consistency check (weekly) |
| 256MB index causes memory pressure on 8GB M1 | Medium | Low | mmap pages reclaimable by kernel; add `madvise(MADV_FREE)` after search completes |
| Initial full scan of `/` takes >30s | High | Medium | Show progress indicator; scan is one-time per machine; subsequent starts <1s |
| `fsevent-sys` crate deprecated before `objc2-core-services` is ready | Low | Low | Pin version; FFI layer is thin (~50 lines) and could be vendored if needed |
| GSIX file corruption from unclean shutdown | Low | Medium | Atomic write (tmp + rename + fsync); CRC32 checksum in header; auto-rebuild on checksum mismatch |
| Paths >224 bytes silently dropped from index | Medium | Low | Log warning; 224B covers ~99.5% of real paths on macOS; consider overflow table in future version |
| HashMap memory overhead (48MB @ 1M entries) | Low | Low | Only present in writer thread; proportional to entry count; acceptable for the O(1) lookup benefit |
| rename() and MustRescan events cause path_index inconsistency | Medium | Medium | Renamed events processed atomically (delete old + create new); MustRescan does full subtree diff |

## Appendix E: Research Sources

- [fsevent-sys crate](https://docs.rs/fsevent-sys/) -- FSEvents FFI bindings for Rust
- [fsevent-stream crate](https://lib.rs/crates/fsevent-stream) -- Stream-based FSEvents API (evaluated, not selected)
- [Apple FSEvents Programming Guide](https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/FSEvents_ProgGuide/UsingtheFSEventsFramework/UsingtheFSEventsFramework.html) -- Official FSEvents documentation
- [Apple Developer Forums: bytesNoCopy alignment](https://developer.apple.com/forums/thread/8011) -- Page alignment requirements
- [Apple Developer: makeBuffer(bytesNoCopy:)](https://developer.apple.com/documentation/metal/mtldevice/1433382-newbufferwithbytesnocopy) -- Metal API reference
- [plocate](https://plocate.sesse.net/) -- Trigram inverted index for path search (architectural inspiration)
- [Apple Silicon page size discussion](https://github.com/flyinghead/flycast/issues/186) -- Confirms 16KB pages on M1+
- [Apple: Addressing architectural differences](https://developer.apple.com/documentation/apple-silicon/addressing-architectural-differences-in-your-macos-code) -- ARM64 memory model
