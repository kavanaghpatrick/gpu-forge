---
spec: gpu-search-persistent-index
phase: research
created: 2026-02-14
---

# Research: File System Index Persistence for GPU-Accelerated Search

**Findings Source**: Web search (2026), GPU Forge KB (601 findings), existing codebase analysis

## Executive Summary

Persistent file indexes for search tools follow 3 architectural patterns: (1) **Trigram inverted index** (plocate: 2500x faster than mlocate, 58% smaller), (2) **NTFS MFT direct read** (Everything search: instant on Windows), (3) **Spotlight-style database** (hidden .db files + FSEvents journal). For macOS Apple Silicon GPU search, **mmap + bytesNoCopy + FSEvents** delivers zero-copy GPU loading with incremental updates. SQLite FTS5 trigram indexes work (100x LIKE speedup) but add 3x storage overhead. Critical: **kqueue has file descriptor limits** (256 default) unsuitable for millions of files; FSEvents is mandatory for macOS at-scale watching.

## 1. FSEvents on macOS - Best Practices

| Aspect | Finding | Source |
|--------|---------|--------|
| **Backend choice** | FSEvents (NOT kqueue) for large-scale watching | Watchexec FSEvents limitations |
| **File-level granularity** | `kFSEventStreamCreateFlagFileEvents` (10.7+) for per-file events instead of directory-only | GPU Forge KB #1264, Apple FSEvents Guide |
| **Rust crate** | `notify` v7+ uses kqueue by default (hits 256 fd limit); `fsevent-stream` crate exposes file-level flags directly | notify crates.io, fsevent-stream docs |
| **kqueue limits** | Default ulimit 256 files → immediate failure watching 300+ files; FSEvents has no such limit | notify #596 issue |
| **Event coalescing** | Latency parameter (0-1000ms) trades responsiveness for CPU overhead; 500ms recommended | GPU Forge KB #1264 |
| **Incremental resume** | Store last FSEvent ID on disk → resume from checkpoint after app restart | GPU Forge KB #1264 |
| **Known issues** | Events are advisory only (coalescing, loss, out-of-order); periodic full re-scan required for correctness | Eclectic Light FSEvents analysis |

**Critical Insight**: Watchexec moved from FSEvents to kqueue in v5+ for development tools, but **reversed this for Watchexec 1.18+** because kqueue's file descriptor limits made it unusable. For indexing millions of files, FSEvents is the only viable option on macOS.

**Production Pattern** (from research):
```rust
// File-level granularity + no defer for low latency
kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagNoDefer
// Latency: 100ms for interactive tools, 500ms for background indexing
// Store last event ID for resume after restart
```

## 2. Index Formats - Performance & Trade-offs

| Tool | Format | Size (1M files) | Search Speed | Update Method |
|------|--------|----------------|--------------|---------------|
| **mlocate** | Custom binary (linear scan) | 1.1 GB | 20s | Full rebuild daily |
| **plocate** | Trigram inverted index + io_uring | 466 MB | 0.008s (2500x faster) | Full rebuild daily |
| **Everything (Windows)** | NTFS MFT direct read | N/A (OS index) | Instant | NTFS change journal |
| **macOS Spotlight** | Hidden .db + indexes in .Spotlight-V100/ | Varies | <1s | FSEvents fed to mdworker |
| **SQLite FTS5 trigram** | FTS table + trigram tokenizer | 3.7 GB (3x growth) | 10-30ms (100x LIKE) | INSERT on file change |

### plocate Architecture

From [plocate.sesse.net](https://plocate.sesse.net/):
- Trigram inverted index (all 3-byte combos in paths) → rapid candidate filtering
- io_uring async I/O (Linux 5.1+) reduces seek latency on HDDs
- Posting lists enable 2500x speedup over linear mlocate scan
- Index build: ~3min for 27M files, vacuum: 50s

### SQLite FTS5 Trigram Performance

From [Andrew Mara blog](https://andrewmara.com/blog/faster-sqlite-like-queries-using-fts5-trigram-indexes):
- 18.2M rows: 1.75s (no index) → 10-30ms (with FTS5 trigram)
- `detail='none'` saves 40% storage (2.8GB vs 3.7GB) without losing LIKE/GLOB speed
- Trade-off: 1.5GB overhead for 50-100x performance gain
- Best for 10MB-100GB read-only datasets

### Spotlight Internal Format

From [Eclectic Light analysis](https://eclecticlight.co/2025/07/30/a-deeper-dive-into-spotlight-indexes/):
- .Spotlight-V100/Store-V2/ contains .db files + indexes/maps/shadows
- Records inode + parent inode for each indexed path
- FSEvents notifies `mds` daemon → `mdworker` re-indexes changed files
- Near-real-time (1-2s latency after file modification)

### Recommendation for GPU Search

**Custom binary format** (GPU Forge KB #1281, #1328) with mmap + demand paging for zero-copy GPU access. Avoid SQLite overhead for simple path matching (no content indexing).

**Existing Implementation** (gpu-search/src/index/shared_index.rs):
```rust
// GSIX format: 64-byte header + packed GpuPathEntry array
const INDEX_MAGIC: u32 = 0x58495347; // "GSIX"
const INDEX_VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;

// Header: magic, version, entry_count, root_hash, saved_at timestamp
// Entries: 256 bytes each (224B path + metadata)
```

✅ **Already optimal** - no format changes needed.

## 3. Incremental Update Strategies

| Strategy | Mechanism | Latency | Accuracy | Overhead |
|----------|-----------|---------|----------|----------|
| **FSEvents journal** | Read event stream from /dev/fsevents | 100-500ms | Advisory (not guaranteed) | Low (kernel-native) |
| **NTFS Change Journal** | USN journal tracks all file changes | ~Instant | Guaranteed | Very low (OS built-in) |
| **Modification timestamps** | stat() mtime comparison | On-demand | 1-second granularity | Low (metadata only) |
| **inode tracking** | Store inode → path mapping, detect stale entries | On rebuild | Handles renames poorly | Medium (extra map) |
| **Content hashing** | SHA256 of file → skip unchanged | On-demand | Perfect | High (read entire file) |

### FSEvents Integration Details

From GPU Forge KB #1264:
- `kFSEventStreamCreateFlagFileEvents` enables file-level granularity (macOS 10.7+)
- `kFSEventStreamCreateFlagNoDefer` delivers events immediately when latency window expires
- Persistent event IDs via `FSEventsGetLastEventIdForDeviceBeforeTime()` enable resume after restart
- No performance degradation observed on 500GB+ filesystems

**Recommended Pattern**:
1. Initial index build (full walk with `ignore` crate)
2. Save index + last FSEvent ID to disk
3. On startup: mmap index, resume FSEvents from stored ID
4. On FSEvent: if file modified → update specific entry; if directory modified → re-scan subtree
5. Periodic full re-scan (hourly or daily) to catch missed events

### Current Implementation Gap

**Existing Code** (gpu-search/src/index/watcher.rs):
- Uses `notify` v7 with `RecommendedWatcher` (defaults to kqueue on macOS)
- 500ms debounce window via `notify-debouncer-mini`
- **Full re-scan on any change** (inefficient)
- No last event ID persistence

**Problems**:
1. kqueue hits 256 file descriptor limit → fails on large directories
2. Full re-scan wastes time for single-file changes
3. No resume capability after restart → always full rebuild

**Fix needed**: Switch to `fsevent-stream` crate with `kFSEventStreamCreateFlagFileEvents`.

## 4. Directory Exclusion Patterns

### Ripgrep/fd Standard Excludes

From [ripgrep GUIDE.md](https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md):

```gitignore
# Version control
.git/ .hg/ .svn/

# Build artifacts
target/ build/ dist/ out/ bin/ obj/

# Dependencies
node_modules/ vendor/ .cargo/ venv/ .venv/ env/

# Caches
.cache/ __pycache__/ .pytest_cache/ .mypy_cache/ .tox/

# IDE
.idea/ .vscode/ .vs/

# macOS
.DS_Store .Spotlight-V100/ .Trashes/ .fseventsd/

# Temporary
tmp/ temp/ .tmp/ *.swp *.swo
```

### Current Implementation

**Existing Code** (gpu-search/src/index/shared_index.rs:39-77):
```rust
pub const DEFAULT_EXCLUDES: &[&str] = &[
    // Version control
    ".git", ".hg", ".svn",
    // Build artifacts
    "target", "build", "dist", "out", "bin", "obj",
    // Dependencies
    "node_modules", "vendor", ".cargo", "venv", ".venv", "env",
    // Caches
    ".cache", "__pycache__", ".pytest_cache", ".mypy_cache", ".tox",
    // IDE
    ".idea", ".vscode", ".vs",
    // macOS
    ".DS_Store", ".Spotlight-V100", ".Trashes", ".fseventsd",
    // Temporary
    "tmp", "temp", ".tmp",
];
```

✅ **Already matches ripgrep best practices** - no changes needed.

**Scanner Integration** (gpu-search/src/index/scanner.rs):
- Uses `ignore` crate (ripgrep's library) for `.gitignore` parsing
- Respects `.git/info/exclude` and global gitignore
- Parallel directory traversal via `WalkBuilder`

✅ **Already optimal** - no changes needed.

## 5. Startup Performance - mmap vs Read

### Performance Comparison

From GPU Forge KB #1281, #1316, and ripgrep discussion:

| Aspect | mmap | read() into memory |
|--------|------|-------------------|
| **Initial load** | Instant (demand paging) | Full file read (slower) |
| **Memory usage** | OS page cache (shared) | Private heap allocation |
| **GPU access** | Zero-copy via bytesNoCopy | Requires CPU→GPU copy |
| **Persistence** | Pages stay in cache after process exit | Lost on exit |
| **Best for** | Single large index file | Many small files |

### Critical Trade-off

From GPU Forge KB #1316 (ripgrep discussion):
- **Many small files**: read() faster (avoid mmap overhead per file)
- **Single large file searched repeatedly**: mmap wins (amortized cost)
- **Existing gpu-search pattern**: Batches files into 64MB chunks → read() is correct
- **Index file pattern**: Single persistent index → mmap is correct

### Rust mmap Crates

From web research:
- `memmap2` (most popular): basic mmap/munmap wrappers
- `mmap-sync` (Cloudflare): wait-free concurrent reader-writer access
- `mmap-io`: async-ready with segment-based loading

### Current Implementation

**Existing Code** (gpu-search/src/io/mmap.rs):
```rust
pub struct MmapBuffer {
    ptr: *mut c_void,
    mapped_len: usize,  // Page-aligned (16KB on Apple Silicon)
    file_size: usize,
    _file: File,
}

impl MmapBuffer {
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        // mmap with MAP_PRIVATE + PROT_READ
        // madvise(MADV_WILLNEED) for prefetch
    }

    pub fn as_metal_buffer(&self, device: &MTLDevice) -> MTLBuffer {
        // Zero-copy via newBufferWithBytesNoCopy
        // Falls back to copy if bytesNoCopy fails
    }
}
```

**Cache Layer** (gpu-search/src/index/cache.rs):
```rust
pub struct MmapIndexCache {
    _mmap: MmapBuffer,
    entries_ptr: *const GpuPathEntry,
    entry_count: usize,
}

impl MmapIndexCache {
    pub fn load_mmap(path: &Path, expected_root_hash: Option<u32>) -> Result<Self> {
        // Validates header (magic, version, root hash)
        // Returns zero-copy slice into mmap region
    }

    pub fn entries(&self) -> &[GpuPathEntry] {
        // Direct slice from mmap - no copying
    }
}
```

✅ **Already optimal** - mmap + bytesNoCopy zero-copy pattern fully implemented.

### Startup Performance Target

Based on research:
- **Index mmap**: <1ms (demand paging, zero-copy to GPU)
- **FSEvents resume**: <10ms (read last event ID, create stream)
- **Total cold start**: <100ms (acceptable for TUI app)
- **Warm start** (index in page cache): <5ms

## Related Specs Discovery

Scanned all specs directories via glob:
- No existing specs found for persistent indexing
- No existing specs found for FSEvents integration
- This is **new functionality** not covered by prior work

Classification: **None** - this is greenfield functionality.

## Quality Commands

Checked gpu-search project (Cargo.toml + Makefile):

| Type | Command | Source |
|------|---------|--------|
| Lint | `cargo clippy --all-targets --all-features` | Rust default |
| TypeCheck | `cargo check --all-targets --all-features` | Rust default |
| Unit Test | `cargo test --lib` | Rust default |
| Integration Test | `cargo test --test '*'` | Rust default |
| Build | `cargo build --release` | Rust default |

**Local CI**: `cargo clippy --all-targets && cargo check && cargo test && cargo build --release`

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| **Technical Viability** | High | All patterns proven (plocate, Spotlight, gpu-search mmap) |
| **Effort Estimate** | M (2-3 days) | Index format exists, need FSEvents integration + save/load |
| **Risk Level** | Low | FSEvents is stable API, mmap already working |
| **Performance Impact** | High positive | Eliminates 29s walk on startup (instant mmap load) |
| **Maintainability** | High | Standard macOS patterns, well-documented |

## Recommendations for Requirements

1. **Index Format**: Extend existing `SharedIndexManager` binary format (GSIX header + GpuPathEntry array) with mmap-based loading via existing `MmapIndexCache` ✅ Already implemented

2. **FSEvents Integration**: Replace `notify` crate with `fsevent-stream` to avoid kqueue file descriptor limits; use `kFSEventStreamCreateFlagFileEvents` for file-level granularity ⚠️ **Requires implementation**

3. **Incremental Updates**: Store last FSEvent ID in index file; on file change, re-scan affected file only (not full directory) ⚠️ **Requires implementation**

4. **Persistence Location**: Keep existing `~/.gpu-search/index/<hash>.idx` path ✅ Already implemented

5. **Startup Sequence**:
   - mmap index file (instant via demand paging) ✅ Already implemented
   - Create Metal buffer via `bytesNoCopy` (zero-copy) ✅ Already implemented
   - Resume FSEvents from stored event ID ⚠️ **Requires implementation**
   - Background: check for stale entries (mtime comparison) ✅ Already implemented

6. **Exclusion Patterns**: Keep existing `DEFAULT_EXCLUDES` + `.gitignore` parsing ✅ Already optimal

7. **Testing Strategy**: Add integration tests for index save/load/staleness (✅ exists); add unit tests for FSEvents event handling ⚠️ **Requires implementation**

## Open Questions

- **Full re-scan frequency**: Hourly? Daily? User-configurable?
- **Index invalidation trigger**: On major macOS version upgrade? APFS snapshot restore?
- **Multiple root directories**: One index per root (current) or merged index?
- **Index size limits**: Cap at 1M files? Warn user if exceeded?
- **FSEvent ID storage**: In index header (requires rewrite on save) or separate sidecar file?

## Sources

### Web Research
- [plocate architecture](https://plocate.sesse.net/)
- [plocate vs mlocate comparison - Linux Uprising](https://www.linuxuprising.com/2021/09/plocate-is-much-faster-locate-drop-in.html)
- [Spotlight internals - Eclectic Light](https://eclecticlight.co/2021/01/28/spotlight-on-search-how-spotlight-works/)
- [Spotlight index deep dive - Eclectic Light](https://eclecticlight.co/2025/07/30/a-deeper-dive-into-spotlight-indexes/)
- [FSEvents limitations - Watchexec](https://watchexec.github.io/docs/macos-fsevents.html)
- [SQLite FTS5 trigram performance - Andrew Mara](https://andrewmara.com/blog/faster-sqlite-like-queries-using-fts5-trigram-indexes)
- [notify crate kqueue issues #596](https://github.com/notify-rs/notify/issues/596)
- [ripgrep mmap heuristic discussion #1769](https://github.com/BurntSushi/ripgrep/discussions/1769)
- [Apple FSEvents Programming Guide](https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/FSEvents_ProgGuide/)
- [notify crate documentation](https://docs.rs/notify/latest/notify/)
- [fsevent-stream crate](https://docs.rs/fsevent-stream)
- [mmap-sync - Cloudflare](https://github.com/cloudflare/mmap-sync)
- [Everything search - Wikipedia](https://en.wikipedia.org/wiki/Everything_(software))
- [mlocate.db format - Linux man page](https://linux.die.net/man/5/mlocate.db)
- [blocate SQLite alternative - GitHub](https://github.com/jboero/blocate)
- [Mapping Files Into Memory - Apple Developer](https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemAdvancedPT/MappingFilesIntoMemory/MappingFilesIntoMemory.html)
- [ripgrep GUIDE.md - GitHub](https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md)
- [Handling Large Files in Rust with mmap - Sling Academy](https://www.slingacademy.com/article/handling-large-files-in-rust-with-memory-mapping-mmap/)
- [mmap performance in Rust - Medium](https://medium.com/@FAANG/advanced-memory-mapping-in-rust-the-hidden-superpower-for-high-performance-systems-a47679aa205e)

### GPU Forge Knowledge Base
- #1264 (gpu-io): FSEvents incremental updates with file-level granularity
- #1281 (gpu-io): mmap-based filesystem index for GPU zero-copy loading
- #1316 (gpu-io): mmap vs read() trade-offs (ripgrep findings)
- #1328 (gpu-io): GPU-optimized index file format design
- #1349 (gpu-io): Metal IO command queue patterns
- #1266 (gpu-io): Triple buffering recommendation for streaming pipelines
- #1337 (metal-compute): MTLDevice multi-queue IO+compute overlap
- #1135 (gpu-io): MTLSharedEvent for cross-queue synchronization
- #1141 (gpu-io): Multiple command queues overlap compute with IO
- #1370 (metal-compute): Simultaneous execution from multiple command queues
- #1359 (gpu-io): Index invalidation and staleness detection strategy
- #1387 (gpu-io): Index build time expectations on macOS APFS

### Codebase
- `gpu-search/src/index/cache.rs`: MmapIndexCache (mmap-based loading)
- `gpu-search/src/index/shared_index.rs`: GSIX format + DEFAULT_EXCLUDES
- `gpu-search/src/index/watcher.rs`: FSEvents integration (notify v7)
- `gpu-search/src/index/scanner.rs`: FilesystemScanner with ignore crate
- `gpu-search/src/io/mmap.rs`: MmapBuffer with bytesNoCopy zero-copy
- `gpu-search/src/gpu/types.rs`: GpuPathEntry (256B repr(C) struct)
- `gpu-search/src/search/streaming.rs`: Streaming search pipeline
- `gpu-search/Cargo.toml`: Dependencies (notify 7, notify-debouncer-mini 0.5)
