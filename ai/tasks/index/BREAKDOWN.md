---
id: index.BREAKDOWN
module: index
priority: 4
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN, gpu-engine.BREAKDOWN]
tags: [gpu-search]
testRequirements:
  unit:
    required: true
---

# Index Module Breakdown

## Context

The filesystem indexer builds and maintains a GPU-resident index of file paths for fast path filtering. The index is cached at `~/.gpu-search/` and loaded via mmap + `newBufferNoCopy` for sub-1ms startup. The GPU path filter kernel runs against this index to scope searches to matching files before any content is loaded.

The index structure (`GpuPathEntry`, 256B per entry, cache-aligned) is ported from rust-experiment. The indexer uses `rayon` for parallel directory scanning. Incremental updates via filesystem watchers keep the index fresh without full rescans.

## Tasks

### T-050: Port index scanner from rust-experiment

Parallel filesystem scanner using `rayon`:
- Walk directory tree with `rayon`-parallelized directory enumeration
- Build `Vec<GpuPathEntry>` with path, flags (is_dir, is_symlink, is_hidden), parent_idx, size, mtime
- Respect .gitignore during scan (use `ignore` crate WalkBuilder)
- Skip symlinks by default (Q&A decision #17)
- Skip hidden directories (`.git/`, `.hg/`, etc.) by default

**Source**: rust-experiment `gpu_os/shared_index.rs` (scan logic), `gpu_os/gpu_index.rs` (entry format)
**Target**: `gpu-search/src/index/scanner.rs`
**Verify**: `cargo test -p gpu-search test_scanner` -- scan test directory, correct entry count, paths match

### T-051: Implement index persistence at ~/.gpu-search/

Save/load binary index to disk:
- Location: `~/.gpu-search/index/` directory
- Format: binary file with header (version, entry count, root path hash) + packed `GpuPathEntry` array
- Save after initial scan completes
- Load on subsequent launches: mmap file -> GPU buffer via `newBufferNoCopy`
- Invalidation: compare root directory mtime against cached mtime

**Target**: `gpu-search/src/index/cache.rs`
**Verify**: `cargo test -p gpu-search test_index_cache` -- save index, restart, load cached index, verify identical entries

### T-052: Implement incremental index updates

Keep index fresh without full rescan:
- Use `notify` crate for filesystem event watching (create, modify, delete, rename)
- On file create: add new `GpuPathEntry`, rebuild GPU buffer
- On file delete: mark entry as deleted (lazy removal)
- On file modify: update mtime and size
- On directory create/delete: add/remove subtree entries
- Batch updates: accumulate changes for 500ms, apply as single GPU buffer rebuild

**Target**: `gpu-search/src/index/watcher.rs`
**Dependency**: Add `notify = "7"` to Cargo.toml
**Verify**: `cargo test -p gpu-search test_incremental_update` -- create file while index active, new file appears in search results

### T-053: Implement GPU-resident index loading

Load index into Metal buffer for GPU path filter kernel:
- mmap cached index file from `~/.gpu-search/`
- `device.newBufferWithBytesNoCopy()` -> zero-copy GPU buffer
- If no cache: scan -> build -> save -> load

Loading should be <10ms for typical index sizes (100K entries = ~25MB).

**Target**: `gpu-search/src/index/gpu_loader.rs`
**Verify**: `cargo test -p gpu-search test_gpu_index_load` -- index loaded into GPU buffer, path filter kernel finds known files

### T-054: Define index entry format

`GpuPathEntry` struct (256 bytes, cache-aligned):

```rust
#[repr(C, align(256))]
pub struct GpuPathEntry {
    pub path: [u8; 224],      // Fixed-width path bytes (UTF-8)
    pub path_len: u16,        // Actual path length
    pub flags: u16,           // is_dir, is_symlink, is_hidden, is_executable
    pub parent_idx: u32,      // Index of parent directory entry
    pub size: u64,            // File size in bytes
    pub mtime: u64,           // Modification time (Unix epoch seconds)
    pub _reserved: [u8; 8],   // Future use (file type hash, encoding, etc.)
}
```

Compile-time assertions:
- `size_of::<GpuPathEntry>() == 256`
- Field offsets match `search_types.h` MSL definitions

**Target**: Already in `gpu-search/src/gpu/types.rs` (task T-013), but index module re-exports
**Verify**: `cargo test -p gpu-search test_gpu_path_entry_layout` -- size and offset assertions pass

### T-055: Implement first-run experience

On first launch (no cached index):
1. Show "Building index..." status in UI
2. Background thread scans filesystem with rayon
3. Progressive count update: "Indexing... 12,450 files found"
4. On completion: save cache, transition to "Ready" status
5. Search available immediately for already-scanned files (progressive indexing)

**Target**: Logic in `index/scanner.rs` + UI status updates via channel
**Verify**: Manual test -- delete `~/.gpu-search/`, launch app, observe indexing status and completion

## Acceptance Criteria

1. Parallel filesystem scan builds correct index using rayon + ignore crate
2. Index cached at `~/.gpu-search/index/` -- persists between sessions
3. Cached index loads in <10ms via mmap + newBufferNoCopy
4. Incremental updates: new/modified/deleted files reflected without full rescan
5. .gitignore respected during scan (uses ignore crate WalkBuilder)
6. Symlinks not followed by default
7. Hidden directories (`.git/`) excluded by default
8. `GpuPathEntry` is 256 bytes, cache-aligned, matches MSL definition
9. GPU path filter kernel successfully queries loaded index
10. First-run indexing shows progress in UI

## Technical Notes

- **ignore crate**: `ignore::WalkBuilder` combines directory walking with .gitignore parsing. Same library ripgrep uses. Handles nested .gitignore, global gitignore, and .ignore files.
- **notify crate**: `notify 7.x` provides cross-platform filesystem watching. On macOS, uses FSEvents (efficient, kernel-level).
- **Index size**: 100K files x 256B = ~25MB. Typical developer workspace. mmap is fast enough.
- **Path length limit**: 224 bytes covers most paths. Paths longer than 224 bytes are truncated (with flag set). This matches rust-experiment behavior.
- **Cache invalidation**: Simple mtime comparison of root directory. Full rescan if stale. Future: content hash for more precise invalidation.
- **GPU buffer rebuild**: When index changes, allocate new GPU buffer, copy updated entries, swap pointer. Old buffer deallocated after next frame completes.
- Reference: TECH.md Section 8, QA.md Section 6
