# QA Analysis: Persistent GPU-Friendly File Index

**Author**: QA Manager Agent
**Date**: 2026-02-14
**Status**: Final
**Feature**: Persistent mmap-based filesystem index with FSEvents incremental updates
**Upstream**: Requirement spec (system-root indexing, GSIX v2 format, 256B entries, FSEvents, 1M+ scale)

---

## 1. Test Strategy Overview

### 1.1 Scope

The persistent file index feature introduces six interconnected subsystems requiring testing at unit, integration, property, and system levels:

1. **GSIX v2 binary format** -- header layout, entry alignment, magic/version fields, FSEvents event ID storage
2. **Mmap-based persistence** -- page alignment, zero-copy semantics, load/save roundtrip
3. **FSEvents watcher** -- event delivery, debouncing, event ID persistence and resume
4. **Incremental updates** -- insert/modify/delete/rename propagation to the index
5. **GPU pipeline** -- bytesNoCopy buffer creation from mmap'd index, kernel dispatch correctness
6. **System-root indexing** -- full `/` scan with exclusion lists, 1M+ entry scale

### 1.2 Existing Test Infrastructure

The codebase already has a mature test foundation:

- **404 unit tests** (100% pass) in `mod tests` blocks across source files
- **Integration tests** in `tests/` directory (11 test files including `test_index.rs`, `test_proptest.rs`, `test_stress.rs`)
- **Property-based tests** using `proptest` (version 1, already in dev-dependencies)
- **Benchmarks** using `criterion` (version 0.5, already in dev-dependencies)
- **Tempfile fixtures** using `tempfile` (version 3, already in dev-dependencies)
- **Compile-time layout assertions** using `const _: () = assert!(...)` for struct sizes

Key existing test patterns to follow:

- `GpuPathEntry` size assertion at compile time: `const _: () = assert!(size_of::<GpuPathEntry>() == 256)` in `/Users/patrickkavanagh/gpu_kernel/gpu-search/src/gpu/types.rs`
- Atomic save via temp file + rename in `SharedIndexManager::save()` at `/Users/patrickkavanagh/gpu_kernel/gpu-search/src/index/shared_index.rs`
- Mmap buffer with Metal `bytesNoCopy` in `/Users/patrickkavanagh/gpu_kernel/gpu-search/src/io/mmap.rs`
- FSEvents watcher with 500ms debounce in `/Users/patrickkavanagh/gpu_kernel/gpu-search/src/index/watcher.rs`
- GPU-vs-CPU equivalence proptest in `/Users/patrickkavanagh/gpu_kernel/gpu-search/tests/test_proptest.rs`

### 1.3 Test Pyramid

| Level | New Tests (est.) | Purpose |
|-------|-----------------|---------|
| Compile-time asserts | ~10 | Layout guarantees (`const _: () = assert!(...)`) |
| Unit tests (`mod tests`) | ~80 | Per-function correctness in each module |
| Integration tests (`tests/`) | ~25 | Cross-module pipeline validation |
| Property tests (proptest) | ~15 | Format invariants, roundtrip fidelity, fuzz resistance |
| Performance benchmarks | ~8 | Latency budgets and throughput gates |
| Stress / scale tests | ~8 | 1M+ entries, memory pressure, concurrency |

**Total new tests: ~146** (added to existing 404 unit tests = ~550 total).

### 1.4 Test File Organization

Following existing conventions:

```
src/index/
  gpu_index.rs     -- existing 8 tests + ~12 new unit tests
  cache.rs         -- existing 8 tests + ~10 new unit tests
  shared_index.rs  -- existing 9 tests + ~8 new unit tests
  scanner.rs       -- existing 10 tests + ~6 new unit tests
  watcher.rs       -- existing 4 tests + ~15 new unit tests
  gpu_loader.rs    -- existing 4 tests + ~8 new unit tests

tests/
  test_index.rs                -- existing integration tests
  test_index_persistence.rs    -- NEW: roundtrip, crash recovery, format validation
  test_index_incremental.rs    -- NEW: FSEvents-driven update correctness
  test_index_scale.rs          -- NEW: 1M+ entries, large file, memory pressure
  test_index_gpu_pipeline.rs   -- NEW: mmap -> bytesNoCopy -> kernel dispatch
  test_index_proptest.rs       -- NEW: property-based format and invariant tests

benches/
  index_load.rs                -- NEW: mmap load latency, save throughput
```

### 1.5 No New Dependencies Required

All testing infrastructure is already available:
- `proptest = "1"` for property-based testing
- `criterion = { version = "0.5", features = ["html_reports"] }` for benchmarks
- `tempfile = "3"` for filesystem fixtures
- `notify = "7"` and `notify-debouncer-mini = "0.5"` for FSEvents integration

---

## 2. Index Format Tests

### 2.1 GSIX v2 Header Validation

The current GSIX v1 format (defined in `shared_index.rs`) uses a 64-byte header:

```
[0..4]   magic:       u32  "GSIX" (0x58495347 LE)
[4..8]   version:     u32  format version (currently 1, becoming 2)
[8..12]  entry_count: u32  number of GpuPathEntry records
[12..16] root_hash:   u32  hash of the root path
[16..24] saved_at:    u64  unix timestamp
[24..32] fsevent_id:  u64  last FSEvents event ID (NEW in v2)
[32..64] reserved:    [u8; 32]
```

**Compile-time assertion (existing, in `shared_index.rs`):**
```rust
const _: () = assert!(std::mem::size_of::<IndexHeader>() == 64);
```

**Unit tests to add in `shared_index.rs::mod tests`:**

| Test Name | Assertion |
|-----------|-----------|
| `test_header_magic_bytes` | First 4 bytes of saved file are `0x47, 0x53, 0x49, 0x58` ("GSIX" LE) |
| `test_header_version_v2` | Bytes [4..8] decode to version 2 after format upgrade |
| `test_header_entry_count_matches` | Bytes [8..12] decode to the exact number of entries written |
| `test_header_root_hash_matches` | Bytes [12..16] match `SharedIndexManager::cache_key(root)` |
| `test_header_saved_at_recent` | `saved_at` timestamp is within 5 seconds of `SystemTime::now()` |
| `test_header_fsevent_id_stored` | Bytes [24..32] store the last FSEvents event ID (v2 field, non-zero after watcher runs) |
| `test_header_reject_unknown_magic` | Loading file with magic `0xDEADBEEF` returns `InvalidFormat` |
| `test_header_reject_future_version` | Loading file with version 99 returns `InvalidFormat("unsupported version")` |
| `test_header_reject_truncated` | Loading file smaller than 64 bytes returns `InvalidFormat("too short")` |
| `test_header_v1_backward_compat` | v1 index files (without fsevent_id) either upgrade gracefully or return a clear error |

**Implementation note:** The existing `IndexHeader::from_bytes()` method at line 173 of `shared_index.rs` already validates magic and version. The v2 upgrade adds the `fsevent_id` field at offset [24..32], reducing the reserved region from 40 to 32 bytes. This is a backward-incompatible change; the version field must increment to 2.

### 2.2 Entry Alignment (256B)

**Existing compile-time assertions (in `gpu/types.rs` lines 163-169):**

```rust
const _: () = assert!(std::mem::size_of::<GpuPathEntry>() == 256);
const _: () = assert!(std::mem::size_of::<GpuMatchResult>() == 32);
const _: () = assert!(std::mem::size_of::<SearchParams>() == 284);
```

**Existing runtime offset tests (in `gpu/types.rs::tests::test_types_layout`):**

The existing `test_types_layout` test at line 181 verifies all field offsets match `search_types.h`. This test is critical and must continue passing after any format changes.

**Additional tests to add:**

| Test Name | Assertion |
|-----------|-----------|
| `test_entry_256b_stride_in_file` | For any saved index file, `file_size == 64 + entry_count * 256` exactly |
| `test_entry_alignment_in_mmap` | After mmap, each `GpuPathEntry` pointer satisfies 4-byte alignment |
| `test_entries_contiguous_no_padding` | `&entries[1] as *const _ - &entries[0] as *const _` equals exactly 256 bytes |
| `test_entry_array_as_raw_bytes_roundtrip` | Cast `&[GpuPathEntry]` to `&[u8]`, then back: byte-identical (validates `#[repr(C)]` layout) |

### 2.3 Page Alignment for mmap

The existing `MmapBuffer` in `src/io/mmap.rs` enforces Apple Silicon 16KB page alignment via `align_to_page()` (line 28). The `PAGE_SIZE` constant is 16384 (line 23).

**Existing tests (in `mmap.rs::tests`):**
- `test_align_to_page` validates the alignment function
- `test_mmap_buffer_basic` verifies `mapped_len() == PAGE_SIZE` for small files
- `test_mmap_buffer_multi_page` verifies multi-page mapping

**Additional tests to add:**

| Test Name | Assertion |
|-----------|-----------|
| `test_index_file_mmap_page_aligned` | After saving and mmap'ing an index, `mmap.mapped_len() % 16384 == 0` |
| `test_entries_start_at_offset_64` | The first `GpuPathEntry` starts at byte 64 (header size), which is 4-byte aligned for the entry's `align_of::<GpuPathEntry>() == 4` |
| `test_bytesnocopy_accepts_mmap_alignment` | `device.newBufferWithBytesNoCopy(...)` returns `Some(buffer)` (not `None`) for a page-aligned mmap region containing index data |
| `test_page_boundary_spanning_entries` | An index with entries that cross a 16KB page boundary loads correctly via mmap (no split-read issues) |

### 2.4 Corrupt File Handling

**Existing corruption tests:**
- `cache.rs::test_index_cache_corrupt_file` (line 421) -- garbage bytes return `InvalidFormat`
- `cache.rs::test_index_cache_wrong_root_hash` (line 439) -- hash mismatch detected
- `shared_index.rs::test_shared_index_corrupt_file` (line 684) -- text content returns `InvalidFormat`

**Additional tests to add:**

| Test Name | Assertion |
|-----------|-----------|
| `test_corrupt_entry_count_overflow` | Header with `entry_count = u32::MAX`: returns error (would require 1TB of data, not available) |
| `test_corrupt_zero_byte_file` | Empty file returns error via `MmapBuffer::from_file()` ("Cannot mmap empty file") |
| `test_corrupt_header_only_zero_entries` | 64-byte file with `entry_count=0`, valid magic/version: loads successfully as empty index |
| `test_corrupt_partial_entry` | File size = 64 + 128 (half an entry): returns `InvalidFormat("File too short")` |
| `test_corrupt_random_bytes_no_panic` | 1KB of random data via proptest: returns `Err`, never panics or triggers UB |
| `test_corrupt_wrong_endianness` | Big-endian magic bytes `0x47534958` at [0..4] instead of LE: returns `InvalidFormat` |
| `test_corrupt_trailing_garbage` | Valid header + valid entries + 100 bytes of garbage appended: loads successfully (ignores trailing bytes) |
| `test_corrupt_header_fields_zeroed` | All-zero 64-byte header: magic mismatch detected, returns `InvalidFormat` |

---

## 3. FSEvents Tests

### 3.1 Architecture Context

The `IndexWatcher` in `src/index/watcher.rs` uses `notify-debouncer-mini` with a 500ms debounce window. The processor loop (line 206) runs in a dedicated thread, re-scanning the entire root directory on every event batch and persisting via `SharedIndexManager::save()`.

For v2, the watcher must additionally:
1. Read the last FSEvents event ID from the GSIX v2 header on startup
2. Resume watching from that stored event ID (not from "now")
3. Store the latest event ID in the header on each save

### 3.2 Event ID Persistence

| Test Name | Assertion |
|-----------|-----------|
| `test_fsevent_id_stored_in_header` | After watcher processes at least one event batch, saved index header contains non-zero event ID at bytes [24..32] |
| `test_fsevent_id_increases_monotonically` | Save index after event batch A, then after event batch B: event ID in second save >= event ID in first save |
| `test_fsevent_id_survives_restart` | Stop watcher -> load index from disk -> verify event ID field is preserved from previous save |
| `test_fsevent_id_zero_triggers_full_scan` | An index with event ID = 0 in the header causes the watcher to perform a complete directory scan rather than an incremental update |

### 3.3 Resume from Stored Event ID

| Test Name | Assertion |
|-----------|-----------|
| `test_resume_catches_missed_changes` | Create a file while the watcher is stopped. Restart watcher with stored event ID from last save. After debounce settles: new file appears in the rebuilt index |
| `test_resume_no_duplicate_entries` | Stop watcher, restart with stored event ID: no duplicate entries for files already in the index |
| `test_resume_handles_large_event_id` | Event ID near `u64::MAX`: watcher resumes without panic or overflow |
| `test_resume_stale_id_fallback` | If the stored event ID is older than FSEvents log retention (or the system has been rebooted), the watcher falls back to a full scan gracefully |

**Research note:** FSEvents on macOS stores event data in `.fseventsd` on each volume. When a volume is properly unmounted, the database is updated. However, if the storage is disconnected unexpectedly, the database may be incomplete and events may be lost. The watcher must handle this by falling back to a full scan when the stored event ID is not recognized by the system. (Source: [Hexordia FSEvents analysis](https://www.hexordia.com/blog/mac-forensics-analysis), [FSEvents Wikipedia](https://en.wikipedia.org/wiki/FSEvents))

### 3.4 Event Coalescing

The existing debouncer uses `Duration::from_millis(500)` (line 121 of `watcher.rs`). FSEvents itself performs temporal coalescing: if two files in the same directory change within a short period, a single event may be delivered specifying the directory contains changes.

| Test Name | Assertion |
|-----------|-----------|
| `test_rapid_creates_single_rebuild` | Creating 100 files in <500ms triggers a single index rebuild (not 100 rebuilds). Verify by counting rebuilds via a counter or log analysis |
| `test_rapid_edits_coalesced` | Writing to the same file 50 times in <500ms coalesces to at most 2 rebuild cycles |
| `test_debounce_window_respected` | Events within 500ms are batched; events arriving after the window trigger a new batch |
| `test_burst_then_quiet_settles` | 1000 file operations then 2 seconds of silence: index settles to a final consistent state within 3 seconds |

### 3.5 Rapid File Changes

| Test Name | Assertion |
|-----------|-----------|
| `test_create_delete_same_file_within_debounce` | Create then immediately delete a file within the debounce window: after settle, file does not appear in index |
| `test_rename_chain` | Rename `a.rs` -> `b.rs` -> `c.rs`: after settle, only `c.rs` appears in index; `a.rs` and `b.rs` do not |
| `test_directory_rename_updates_all_paths` | Rename a directory containing 50 files: all 50 paths updated correctly in the next rebuild |
| `test_concurrent_create_and_search` | Creating files while a GPU search is running: no panic, search completes with consistent (possibly pre-change) results |

### 3.6 Permission Denied Events

| Test Name | Assertion |
|-----------|-----------|
| `test_permission_denied_dir_skipped_silently` | A directory with `chmod 000` encountered during scan: silently skipped, no error propagated. (This already works via `scan_recursive()` line 80-83 in `gpu_index.rs` and `WalkBuilder` in `scanner.rs`) |
| `test_permission_change_after_index` | File permissions changed to `000` after initial indexing: on next rebuild, file removed from index or flagged |
| `test_root_permission_denied` | Calling `watcher.start()` on a root with no read permission: returns `WatcherError::Io` (existing `test_watcher_invalid_path` covers a similar case) |

---

## 4. Incremental Update Tests

### 4.1 CRUD Operations

The current `processor_loop()` performs a full rescan on every event batch (line 231 of `watcher.rs`). True incremental updates (insert/delete/modify specific entries) are a future optimization. These tests verify correctness regardless of whether the implementation uses full rescan or targeted updates.

| Test Name | Assertion |
|-----------|-----------|
| `test_incremental_file_create` | New file in watched directory appears in index after debounce window + rebuild |
| `test_incremental_file_modify` | Modified file has updated `mtime` in its index entry (and possibly `size` if content length changed) |
| `test_incremental_file_delete` | Deleted file is removed from index (`entry_count` decreases by 1) |
| `test_incremental_file_rename` | Renamed file: old path absent from index, new path present, metadata preserved |
| `test_incremental_dir_create_with_files` | New directory with 5 files: all 5 files appear in index after rebuild |
| `test_incremental_dir_delete` | Deleted directory: all descendant file entries removed from index |
| `test_incremental_nested_create` | File created 5 levels deep: appears in index with correct full path |
| `test_incremental_symlink_create` | Symlink created: appears with `IS_SYMLINK` flag set (if `follow_symlinks=false`, target's children not indexed) |

**Implementation approach:** Each test creates a `TempDir`, starts an `IndexWatcher`, performs a filesystem mutation, waits for the debounce window (500ms) plus a settlement buffer (500ms), then loads the index from disk and verifies the expected state.

### 4.2 Concurrent Read During Update

| Test Name | Assertion |
|-----------|-----------|
| `test_concurrent_read_during_rebuild` | Thread A holds an mmap'd `MmapIndexCache` while thread B triggers a rebuild that atomic-renames the index file. Thread A's mmap remains valid (Unix semantics: mmap survives unlink/rename of underlying file) |
| `test_atomic_rename_preserves_old_mmap` | Mmap the index file, then `save()` a new index (which renames over the old file). The original mmap'd data remains accessible until dropped |
| `test_search_during_incremental_update` | Dispatch a GPU search via `SearchOrchestrator` while the watcher is rebuilding the index: search returns valid (possibly stale) results, no crash |
| `test_multiple_readers_one_writer` | 4 threads read entries from `MmapIndexCache` concurrently while 1 thread triggers saves. No data races, no panics. (The `unsafe impl Send + Sync for MmapIndexCache` at line 97-98 of `cache.rs` makes this claim; this test verifies it) |

### 4.3 No Data Loss

| Test Name | Assertion |
|-----------|-----------|
| `test_no_entry_loss_on_rapid_creates` | Create 500 files sequentially with small delays between each: final index after settle contains all 500 entries |
| `test_no_duplicate_entries_after_updates` | After multiple watcher rebuild cycles, `entry_count` matches the actual unique file count on disk (no duplicates) |
| `test_update_preserves_unmodified_entries` | In a 1000-entry index, modify 1 file: the other 999 entries are byte-identical in the rebuilt index |

---

## 5. Performance Tests

### 5.1 Index Load Time (mmap)

**Target: <1ms for mmap load of a cached index with 100K entries.**

The mmap syscall is O(1) -- it sets up page table entries without reading data. The cost is header validation (64 bytes of reads). No page faults occur until entries are actually accessed.

| Test/Bench | Measurement | Budget |
|------------|-------------|--------|
| `bench_mmap_load_100k` | `MmapIndexCache::load_mmap()` call to completion | <1ms |
| `bench_mmap_load_1m` | Same for 1M entries (256MB index file) | <5ms |
| `test_mmap_load_under_budget` | Assert `load_time < Duration::from_millis(10)` for 100K entries | Hard gate |
| `bench_mmap_first_entry_access` | Time from `load_mmap()` to accessing `entries()[0]` (includes first page fault) | <0.1ms |

**Benchmark file:** `benches/index_load.rs` using criterion.

### 5.2 Initial Build Time from `/`

The full system-root scan is inherently slow (1M+ files). This is informational, not a hard budget.

| Test/Bench | Measurement | Expectation |
|------------|-------------|-------------|
| `bench_scan_src_directory` | `FilesystemScanner::scan()` on `gpu-search/src/` (~40 files) | <100ms baseline |
| `bench_build_index_from_entries_100k` | `GpuResidentIndex::from_entries()` with 100K pre-built entries | <10ms |
| `bench_save_index_100k` | `SharedIndexManager::save()` for 100K entries (25.6MB write) | <100ms |
| `bench_save_index_1m` | `SharedIndexManager::save()` for 1M entries (256MB write) | <1s |

### 5.3 Incremental Update Latency

**Target: <1s from filesystem change to index updated on disk.**

| Test/Bench | Measurement | Budget |
|------------|-------------|--------|
| `test_incremental_latency_under_1s` | Timestamp delta from `fs::write()` call to new index file mtime on disk | <1s |
| `bench_rescan_10k_dir` | `FilesystemScanner::scan()` + `save()` on a 10K-file directory after adding 1 file | <500ms |

**Note:** The current architecture performs a full rescan on every event batch. For 1M entries, this will exceed the 1s budget. A true incremental update path (apply deltas to the existing index) would be needed to meet the budget at scale. The performance test documents the current behavior and establishes a baseline.

---

## 6. Scale Tests

### 6.1 Entry Count: 1M+

All scale tests use synthetic `GpuPathEntry` generation to avoid dependency on real filesystem state:

```rust
fn generate_synthetic_entries(count: usize) -> Vec<GpuPathEntry> {
    (0..count).map(|i| {
        let mut entry = GpuPathEntry::new();
        let path = format!("/synthetic/depth_{}/sub_{}/file_{:06}.rs",
                           i % 100, i % 1000, i);
        entry.set_path(path.as_bytes());
        entry.set_size((i * 1024) as u64);
        entry.mtime = 1700000000 + i as u32;
        entry
    }).collect()
}
```

| Test Name | Assertion |
|-----------|-----------|
| `test_scale_1m_entries_build` | `GpuResidentIndex::from_entries()` with 1M entries: completes without OOM, `entry_count() == 1_000_000` |
| `test_scale_1m_entries_save_load` | Save 1M entries (256MB file), load via `SharedIndexManager::load()`: `entry_count == 1_000_000` |
| `test_scale_1m_entries_mmap_load` | Save 1M entries, load via `MmapIndexCache::load_mmap()`: `entry_count == 1_000_000`, byte-identical to original |
| `test_scale_1m_entries_gpu_buffer` | Upload 1M entries to Metal GPU buffer via `to_gpu_buffer()`: `buffer.length() >= 256 * 1_000_000` |
| `test_scale_1m_find_by_name` | `find_by_name("file_000042.rs")` on 1M-entry index: returns exactly 1 result, completes in <100ms |

### 6.2 Large Index File: 256MB+

| Test Name | Assertion |
|-----------|-----------|
| `test_scale_large_file_mmap` | Mmap a 256MB index file: `mapped_len` is page-aligned (multiple of 16KB), `as_slice()` returns valid header + entries |
| `test_scale_large_file_bytesnocopy` | `MmapBuffer::as_metal_buffer()` with 256MB region: `newBufferWithBytesNoCopy` succeeds or `newBufferWithBytes` fallback succeeds |
| `test_scale_large_file_save_load_roundtrip` | Save 1M entries -> mmap load -> verify first, middle, and last entries are byte-identical to originals |

### 6.3 Memory Pressure Behavior

| Test Name | Assertion |
|-----------|-----------|
| `test_scale_mmap_under_memory_pressure` | With 3 concurrent mmap'd indexes (each 100K entries), all load correctly. Kernel may page out inactive regions but correctness is maintained |
| `test_scale_drop_releases_mapping` | After dropping `MmapIndexCache`, calling `MmapBuffer::drop()` triggers `munmap()`. Verify by loading a new index into the same address range (no double-mapping) |
| `test_scale_concurrent_large_indexes` | 3 separate 100K-entry indexes loaded simultaneously: all entry counts correct, no cross-contamination |

---

## 7. Edge Cases

### 7.1 Symlinks

| Test Name | Assertion |
|-----------|-----------|
| `test_edge_symlink_to_file` | Symlink to a regular file: indexed with `IS_SYMLINK` flag set. The scanner's `follow_symlinks=false` default (line 25 of `scanner.rs`) means the symlink itself is recorded, not the target |
| `test_edge_symlink_cycle` | Symlink creating a directory cycle (A -> B -> A): scanner terminates without infinite loop. The `ignore` crate's `WalkBuilder` handles this via cycle detection |
| `test_edge_broken_symlink` | Symlink pointing to a nonexistent target: silently skipped by `metadata()` error handling (line 147 of `scanner.rs`) |
| `test_edge_symlink_to_excluded_dir` | Symlink pointing into `node_modules/`: behavior depends on whether the symlink target is in the exclusion list. Document and test actual behavior |

### 7.2 Long Paths (>224 bytes)

The `GPU_PATH_MAX_LEN` constant is 224 bytes (line 76 of `types.rs`). Paths exceeding this are silently skipped in both `gpu_index.rs` (line 95) and `scanner.rs` (line 142).

| Test Name | Assertion |
|-----------|-----------|
| `test_edge_path_exactly_224_bytes` | Path with exactly 224 UTF-8 bytes: indexed successfully, `path_len == 224` |
| `test_edge_path_225_bytes` | Path with 225 bytes: silently skipped, entry does not appear in index |
| `test_edge_path_1024_bytes` | Very long path (deeply nested directories): skipped without error or panic |
| `test_edge_path_near_limit_roundtrip` | 224-byte path survives save -> mmap load roundtrip with byte-identical `path[0..224]` and `path_len == 224` |

### 7.3 Unicode Filenames

| Test Name | Assertion |
|-----------|-----------|
| `test_edge_unicode_ascii` | File named "cafe.rs": indexed, `path_len` equals byte count of UTF-8 path |
| `test_edge_unicode_multibyte_cjk` | File named with CJK characters (3-byte UTF-8 each): indexed if total path byte count <= 224 |
| `test_edge_unicode_emoji` | File named with emoji (4-byte UTF-8 codepoints): indexed if total path byte count <= 224 |
| `test_edge_unicode_normalization` | NFD vs NFC filenames (common on APFS vs HFS+): stored as-is from the filesystem, no normalization applied. Two identical-looking filenames in different normalization forms produce two separate index entries |
| `test_edge_unicode_roundtrip_fidelity` | Unicode path bytes are identical after save -> load -> mmap cycle (no encoding transformation) |

### 7.4 Special Characters

| Test Name | Assertion |
|-----------|-----------|
| `test_edge_space_in_filename` | File named "hello world.rs": indexed, path contains literal space character |
| `test_edge_newline_in_filename` | File with newline in name (valid on macOS): indexed or gracefully skipped, no panic |
| `test_edge_null_byte_path` | Path containing NUL byte: `set_path()` stores the bytes as-given; `to_str()` may fail. Index stores raw bytes regardless |
| `test_edge_backslash_in_name` | File named "back\\slash.rs": indexed on macOS (backslash is a valid filename character) |

### 7.5 Permission Denied Directories

| Test Name | Assertion |
|-----------|-----------|
| `test_edge_permission_denied_dir` | Directory with `chmod 000`: `scan_recursive()` returns `Ok(())` and continues (line 80-83 of `gpu_index.rs`) |
| `test_edge_permission_denied_file` | File with `chmod 000`: silently skipped by `metadata()` error handling |
| `test_edge_mixed_permissions` | Directory tree with some readable and some unreadable entries: only readable entries indexed, no error for unreadable ones |
| `test_edge_root_unreadable` | Completely unreadable root: `build_from_directory()` returns an empty index (0 entries) |

### 7.6 Disk Full During Save

| Test Name | Assertion |
|-----------|-----------|
| `test_edge_disk_full_preserves_old_index` | Simulate write failure during `save()` (e.g., using a restricted-size tmpfs or intercepting the temp file write). The existing `.idx` file must remain intact because the atomic rename never occurs |
| `test_edge_write_fail_no_partial_idx` | If `fs::File::create(&tmp_path)` or `write_all()` fails, the `.idx.tmp` file is not left behind (or if it is, it does not corrupt the next `load()`) |
| `test_edge_rename_fail_cleanup` | If `fs::rename(&tmp_path, &idx_path)` fails (e.g., cross-device), the `.idx.tmp` file exists but the original `.idx` is untouched |

---

## 8. Regression Tests

### 8.1 walk_and_filter Fallback

The streaming search pipeline (`search_streaming()` in `orchestrator.rs`) uses `walk_and_filter()` for file discovery. The index must not break this fallback path.

| Test Name | Assertion |
|-----------|-----------|
| `test_regression_walk_and_filter_still_works` | `search_streaming()` produces results for a known query in a test directory, without any index file existing on disk |
| `test_regression_blocking_search_still_works` | `orchestrator.search()` (non-streaming, blocking API) produces results without an index present |
| `test_regression_index_deleted_mid_search` | Delete the index file while a search is in progress: search completes using walk_and_filter, no crash |

### 8.2 Search Results Equivalence

This is a critical validation: indexed and non-indexed search must produce equivalent results.

| Test Name | Assertion |
|-----------|-----------|
| `test_regression_indexed_vs_unindexed_content_results` | For a known directory with 10 known files, search with the index and search without the index produce the same `ContentMatch` set (same file paths, line numbers, and match content) |
| `test_regression_filename_match_equivalence` | `find_by_name("main.rs")` via the index produces the same file list as `walk_and_filter()` filename matching |
| `test_regression_file_type_filter_equivalence` | `files_with_extension("rs")` via the index matches the set of `.rs` files found by runtime file type detection |
| `test_regression_hidden_file_handling` | Hidden files (starting with `.`) are excluded/included consistently between indexed and non-indexed paths, respecting the `skip_hidden` config |

### 8.3 Existing Test Suite Non-Regression

| Check | Method |
|-------|--------|
| All 404 existing tests pass | `cargo test` full suite after index feature merge |
| No new clippy warnings | `cargo clippy -p gpu-search -- -D warnings` |
| No new `unsafe` without `// SAFETY:` comment | Manual code review during PR |
| Compile-time assertions still hold | Build succeeds (assertions are checked at compile time) |
| Existing proptest properties hold | `cargo test --test test_proptest` passes at 1000 iterations |

---

## 9. GPU Pipeline Tests

### 9.1 bytesNoCopy Buffer Creation from mmap

The `MmapBuffer::as_metal_buffer()` method at line 188 of `mmap.rs` attempts `newBufferWithBytesNoCopy_length_options_deallocator` first, falling back to `newBufferWithBytes_length_options` if it returns `None`.

For the index use case, the mmap'd region is page-aligned (16KB), which satisfies the Metal requirement for `bytesNoCopy`. On Apple Silicon unified memory, this means the GPU accesses the index data with zero copies -- the mmap'd virtual address is directly GPU-accessible.

| Test Name | Assertion |
|-----------|-----------|
| `test_gpu_bytesnocopy_from_index_mmap` | Save an index to disk, mmap it via `MmapBuffer::from_file()`, create a Metal buffer via `as_metal_buffer()`: buffer is non-null and has valid length |
| `test_gpu_bytesnocopy_page_alignment_verified` | The mmap'd region pointer is 16KB-aligned (assert `mmap.as_ptr() as usize % 16384 == 0`). This is guaranteed by the kernel but worth verifying |
| `test_gpu_bytesnocopy_contents_match` | Metal buffer contents (read back via `buffer.contents()`) match mmap'd data byte-for-byte for the entry region |
| `test_gpu_bytesnocopy_length_covers_entries` | `buffer.length() >= header_size + entry_count * 256` |
| `test_gpu_bytesnocopy_lifetime_safety` | Document behavior when mmap is dropped while Metal buffer exists. Since the deallocator is `None` (line 203 of `mmap.rs`), the Metal buffer must not outlive the `MmapBuffer`. Test that the `_mmap` field in `GpuLoadedIndex` keeps the mapping alive |

### 9.2 Entry Count Validation

| Test Name | Assertion |
|-----------|-----------|
| `test_gpu_entry_count_matches_header` | `GpuLoadedIndex::entry_count()` matches the `entry_count` field parsed from the GSIX header |
| `test_gpu_entry_count_matches_buffer_size` | `buffer.length() / 256 >= entry_count` (buffer may be larger due to page alignment) |
| `test_gpu_zero_entries_valid_buffer` | Empty index (0 entries): `to_gpu_buffer()` produces a valid minimal buffer (existing test `test_gpu_index_empty` at line 567 of `gpu_index.rs` covers this) |
| `test_gpu_entry_count_overflow_rejected` | Index file claiming `u32::MAX` entries: rejected during `load_mmap()` before any GPU allocation attempt |

### 9.3 Search Results Match CPU Verification

Leveraging the established proptest pattern from `tests/test_proptest.rs` (which already validates GPU-vs-CPU consistency for content search):

| Test Name | Assertion |
|-----------|-----------|
| `test_gpu_index_search_matches_cpu` | For a known set of files, search via the index-loaded GPU buffer produces the same `ContentMatch` results as the CPU reference implementation |
| `test_gpu_index_no_false_positives` | Every GPU match from an indexed search has the search pattern at the reported `byte_offset` position in the source file content |
| `test_gpu_index_no_false_negatives_non_boundary` | For patterns that do not cross 64-byte GPU thread boundaries, all CPU matches are also found by the GPU search |
| `prop_gpu_index_search_consistency` | Proptest: random temp directory with random files + random search pattern. Indexed search and non-indexed search produce the same result count (within GPU boundary-crossing tolerance) |

---

## 10. Reliability Tests

### 10.1 Crash During Index Write (Atomic Rename)

The existing `SharedIndexManager::save()` at line 284 of `shared_index.rs` implements atomic write:

```rust
let tmp_path = idx_path.with_extension("idx.tmp");
fs::File::create(&tmp_path)?;
// ... write header + entries ...
fs::rename(&tmp_path, &idx_path)?;
```

This pattern ensures that readers either see the old complete index or the new complete index, never a partial write.

**Research note:** The `atomic-write-file` crate and `tempfile` crate's `persist()` method both use this rename pattern. The SquirrelFS project at Cornell demonstrated that Rust's type system can enforce correct crash-consistency ordering, and that `rename()` on APFS is atomic for same-volume operations. (Sources: [atomic-write-file crate](https://crates.io/crates/atomic-write-file), [SquirrelFS](https://arxiv.org/html/2406.09649v1))

| Test Name | Assertion |
|-----------|-----------|
| `test_reliability_atomic_rename_no_partial` | After `save()` completes, no `.idx.tmp` file exists. Only the final `.idx` file is present |
| `test_reliability_tmp_file_left_on_write_error` | Simulate a write error after creating `.idx.tmp`: the original `.idx` file is untouched; `.idx.tmp` may exist but is invalid |
| `test_reliability_concurrent_save_no_corruption` | Two threads call `save()` simultaneously with different entry sets: the final `.idx` file contains one complete valid index (no interleaving) |
| `test_reliability_rename_atomicity` | Save a new index while another thread is reading the old one via mmap: the reader sees either the old or new index, never a mix |

### 10.2 Corrupt Index Recovery

| Test Name | Assertion |
|-----------|-----------|
| `test_reliability_corrupt_triggers_rebuild` | Manually corrupt the index file (overwrite first 4 bytes with zeros). Call `GpuIndexLoader::load()`: detects corruption, falls through to `scan_build_save_load()`, produces a valid new index |
| `test_reliability_stale_triggers_rebuild` | Set the index `saved_at` timestamp to 2 hours ago. Call `loader.load()`: `is_stale()` returns true, triggers rebuild |
| `test_reliability_delete_and_rebuild` | `manager.delete(root)` then `loader.load(root)`: rebuilds cleanly with a fresh scan |
| `test_reliability_corrupt_recovery_full_pipeline` | Corrupt the index file with random bytes. Call `loader.load()` which internally: detects InvalidFormat -> scans filesystem -> saves new valid index -> returns `GpuLoadedIndex`. Verify the returned index has valid entries |

### 10.3 Graceful Degradation

| Test Name | Assertion |
|-----------|-----------|
| `test_reliability_missing_cache_dir_created` | First `save()` call on a fresh system: creates `~/.gpu-search/index/` directory (existing test `test_shared_index_index_dir_creation` at line 663 of `shared_index.rs` covers this) |
| `test_reliability_cache_dir_permission_denied` | If the index directory is unwritable: `save()` returns an error, but `search_streaming()` still works via walk_and_filter |
| `test_reliability_mmap_failure_fallback` | If `MmapBuffer::from_file()` fails: `GpuIndexLoader` falls through to `scan_build_save_load()` which uses `newBufferWithBytes` (copy path) instead of `bytesNoCopy` |
| `test_reliability_no_metal_device` | If `MTLCreateSystemDefaultDevice()` returns `None`: all index operations (build, save, load, scan) succeed without a GPU buffer. The `gpu_buffer` field remains `None` |
| `test_reliability_watcher_thread_panic` | If the processor thread panics: `IndexWatcher::drop()` calls `stop()` which calls `handle.join()`. The `let _ = handle.join()` at line 166 of `watcher.rs` silently absorbs the panic. Verify the app continues without live updates |
| `test_reliability_watcher_channel_disconnect` | If the event sender is dropped (e.g., `debouncer` dropped): `event_rx.recv_timeout()` returns `Disconnected`, processor loop exits cleanly (line 252-255 of `watcher.rs`) |

---

## 11. Property-Based Tests (Proptest)

These tests use the `proptest` crate to verify format invariants across randomly generated inputs. They follow the established pattern from `tests/test_proptest.rs` with `TestRunner` and explicit `Config { cases: N }`.

### 11.1 Format Roundtrip Properties

| Property | Generator | Assertion |
|----------|-----------|-----------|
| `prop_save_load_roundtrip` | Random `Vec<GpuPathEntry>` (1-10K entries with random paths, sizes, flags) | `save() -> load()` produces byte-identical entries for all 256 bytes of each entry |
| `prop_save_mmap_roundtrip` | Random `Vec<GpuPathEntry>` (1-10K entries) | `save() -> MmapIndexCache::load_mmap()` produces byte-identical entries |
| `prop_entry_size_invariant` | Random `GpuPathEntry` (arbitrary field values) | `size_of_val(&entry) == 256` always |
| `prop_path_len_bounded` | Random byte arrays (0-300 bytes) | After `set_path()`, `path_len <= 224` always, and `path_len == min(input.len(), 224)` |

### 11.2 Index Integrity Properties

| Property | Generator | Assertion |
|----------|-----------|-----------|
| `prop_file_size_matches_formula` | Random entry count (0-100K) | Saved index file size equals exactly `64 + entry_count * 256` |
| `prop_header_fields_consistent` | Random entry count + random root path | After `save()`, header magic is `0x58495347`, version matches expected, entry_count matches input length |
| `prop_cache_key_deterministic` | Random `PathBuf` values | `cache_key(path)` called twice produces identical u32 hash |
| `prop_cache_key_no_collisions_sample` | 1000 distinct random paths | All 1000 `cache_key()` values are unique (probabilistic; collision probability is ~1/2^32 per pair, ~1/8M for 1000 pairs) |

### 11.3 Binary Format Fuzzing

These tests ensure the parser never panics or triggers undefined behavior on malformed input. This is critical because `MmapIndexCache::load_mmap()` performs `unsafe` pointer arithmetic (line 187-189 of `cache.rs`).

**Research note:** The proptest and cargo-fuzz ecosystems support structured input fuzzing for binary format parsers. Proptest performs automatic test case shrinking to isolate minimal failing inputs. (Sources: [proptest GitHub](https://github.com/proptest-rs/proptest), [propfuzz](https://github.com/facebookarchive/propfuzz), [Rust Fuzz Book](https://rust-fuzz.github.io/book/afl/tutorial.html))

| Property | Generator | Assertion |
|----------|-----------|-----------|
| `prop_corrupt_bytes_no_panic` | Take a valid saved index file, flip 1-10 random bytes | `load_mmap()` returns `Err(CacheError)`, never panics. Run with `cases = 5000` |
| `prop_truncated_file_no_panic` | Take a valid saved index file, truncate at a random offset (1..file_size) | `load_mmap()` returns `Err`, never panics |
| `prop_random_bytes_no_panic` | Generate 64-8192 random bytes, write to a file | `load_mmap()` returns `Err`, never panics |
| `prop_extended_file_no_panic` | Take a valid saved index file, append 1-1000 random bytes | `load_mmap()` either succeeds (extra bytes ignored) or returns `Err`, never panics |

---

## 12. Test Execution Plan

### Phase 1: Format and Persistence (Week 1)

Priority: Section 2 (format), Section 10.1 (atomic rename), Section 11 (proptest)

- Add v2 header tests with FSEvents event ID field
- Add entry alignment verification tests
- Add corrupt file handling edge cases
- Add property-based roundtrip and fuzzing tests
- **Gate:** `cargo test --lib -- index` passes; all 404 existing + ~40 new tests green

### Phase 2: FSEvents and Incremental Updates (Week 2)

Priority: Section 3 (FSEvents), Section 4 (incremental), Section 10.2 (recovery)

- Add event ID persistence and resume tests
- Add event coalescing and rapid-change tests
- Add CRUD operation tests (create/modify/delete/rename)
- Add concurrent read-during-update tests
- **Gate:** `cargo test -- test_incremental test_fsevent` passes; ~25 new tests green

### Phase 3: Scale and Performance (Week 3)

Priority: Section 5 (performance), Section 6 (scale)

- Implement synthetic entry generator for 1M+ tests
- Add mmap load latency benchmarks
- Add save throughput benchmarks
- Add 1M entry build/save/load/GPU-upload tests
- **Gate:** All performance benchmarks within budget; scale tests pass without OOM

### Phase 4: GPU Pipeline, Edge Cases, and Regression (Week 4)

Priority: Section 7 (edge cases), Section 8 (regression), Section 9 (GPU pipeline), Section 10.3 (degradation)

- Add bytesNoCopy pipeline tests
- Add GPU-vs-CPU search equivalence tests for indexed path
- Add symlink, Unicode, permission, disk-full edge cases
- Add walk_and_filter fallback regression tests
- Add indexed-vs-unindexed result equivalence tests
- **Gate:** Full `cargo test` passes (404 existing + ~146 new = ~550 total); all regressions verified

### CI Integration

```yaml
# GitHub Actions workflow additions for index feature

- name: Run index unit tests
  run: cargo test --lib -- index cache shared_index scanner watcher gpu_loader

- name: Run index integration tests
  run: cargo test --test test_index_persistence --test test_index_incremental -- --nocapture

- name: Run index scale tests (slower, run separately)
  run: cargo test --test test_index_scale -- --nocapture --ignored
  # Scale tests are #[ignore]'d by default, run explicitly

- name: Run index property tests (higher iteration count)
  run: cargo test --test test_index_proptest -- --nocapture
  env:
    PROPTEST_CASES: 5000

- name: Run index GPU pipeline tests (requires Metal)
  run: cargo test --test test_index_gpu_pipeline -- --nocapture
  env:
    MTL_SHADER_VALIDATION: 1

- name: Run index benchmarks (regression check)
  run: cargo bench --bench index_load -- --warm-up-time 1
```

---

## 13. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| FSEvents event ID lost after unexpected reboot/volume detach | Medium | Medium | Fall back to full scan when stored event ID is unrecognized by the system. Test `test_resume_stale_id_fallback` covers this |
| Mmap region invalidated by concurrent file truncation | High | Low | Atomic rename in `save()` ensures readers with existing mmap see a complete index. The Unix kernel keeps the old inode data alive until all mmaps are unmapped. Test `test_atomic_rename_preserves_old_mmap` verifies this |
| 256MB GPU buffer allocation failure on low-memory systems | Medium | Low | `newBufferWithBytes` returns `None` on failure; `expect()` at line 232 of `gpu_index.rs` would panic. Should convert to graceful error. Test `test_scale_large_file_bytesnocopy` documents behavior |
| Hash collision in `cache_key()` causes wrong index loaded | Low | Very Low | `root_hash` field in header provides secondary validation. Collision probability per pair is ~1/2^32. Property test `prop_cache_key_no_collisions_sample` provides confidence |
| Unicode normalization differences between APFS and HFS+ volumes | Low | Low | Paths stored as raw bytes from the filesystem; no normalization applied. Test `test_edge_unicode_normalization` documents behavior |
| `GpuPathEntry` layout mismatch with Metal shader `search_types.h` | Critical | Very Low | Compile-time `const` assertions (line 163 of `types.rs`) and runtime offset tests (`test_types_layout`) already guard against this |
| Watcher thread deadlock on shutdown | Medium | Low | `recv_timeout(200ms)` + shutdown channel pattern (line 219 of `watcher.rs`) prevents indefinite blocking. Test `test_watcher_stop_and_restart` verifies clean shutdown |
| Full rescan on every event batch exceeds 1s budget at 1M+ entries | High | High | Current architecture does full rescan. A true incremental update path (apply targeted inserts/deletes) is needed for 1M+ scale. Performance test documents the gap; the watcher can be enhanced later without changing the index format |
| `unsafe` pointer arithmetic in mmap cache causes UB on corrupt files | Critical | Low | Header validation (magic, version, entry_count, file size check) gates all pointer access. Property fuzz tests `prop_corrupt_bytes_no_panic` and `prop_random_bytes_no_panic` verify no panics on malformed input |

---

## 14. Acceptance Criteria

The feature is ready for merge when all of the following are satisfied:

1. **All 404 existing tests pass** -- zero regressions confirmed by `cargo test`
2. **All ~146 new tests pass** -- across 6 new integration test files and in-module unit tests
3. **Compile-time assertions hold** -- `GpuPathEntry` is 256 bytes, `IndexHeader` is 64 bytes, `GpuMatchResult` is 32 bytes
4. **Mmap load latency <1ms** for 100K entries -- verified by `bench_mmap_load_100k`
5. **Incremental update latency <1s** for small directories -- verified by `test_incremental_latency_under_1s`
6. **1M entry scale validation** -- build, save, mmap load, and GPU upload all succeed without OOM
7. **Property tests pass at 5000 iterations** -- no format invariant violations, no panics on corrupt input
8. **Corrupt file recovery** -- corrupt/truncated/random-byte index files produce `Err`, not panics or UB
9. **Atomic write safety** -- simulated failures during `save()` leave no partial `.idx` files; old index preserved
10. **GPU pipeline equivalence** -- indexed search results match non-indexed search results for identical queries on the same directory
11. **Exclusion list correctness** -- system/build directories (`.git`, `node_modules`, `target`, `.Spotlight-V100`, etc.) excluded from the index, matching the `DEFAULT_EXCLUDES` list in `shared_index.rs`
12. **FSEvents resume works** -- watcher correctly resumes from stored event ID on restart
13. **CI green** -- all new test jobs pass in GitHub Actions
14. **No new `unsafe` without documented `// SAFETY:` comments** -- clippy clean with `-D warnings`

---

## 15. Research References

- [memmap2 crate](https://docs.rs/memmap2/latest/memmap2/struct.Mmap.html) -- Cross-platform Rust mmap API, used for comparison with the custom `MmapBuffer` implementation
- [mmap-sync (Cloudflare)](https://github.com/cloudflare/mmap-sync) -- Concurrent data access using memory-mapped files with wait-free synchronization
- [Hexordia FSEvents Analysis](https://www.hexordia.com/blog/mac-forensics-analysis) -- How macOS FSEvents work, event coalescing behavior, and data loss scenarios on volume detach
- [FSEvents Wikipedia](https://en.wikipedia.org/wiki/FSEvents) -- Overview of the macOS FSEvents API, event delivery semantics, and volume format reliability
- [Apple Metal bytesNoCopy](https://developer.apple.com/documentation/metal/mtldevice/1433382-makebuffer) -- Metal buffer creation from existing memory allocations; alignment requirements for zero-copy access
- [Metal Best Practices: Resource Options](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/ResourceOptions.html) -- Shared storage mode for unified memory on Apple Silicon
- [SquirrelFS (Cornell)](https://arxiv.org/html/2406.09649v1) -- Using the Rust type system to check file-system crash consistency; atomic rename semantics
- [atomic-write-file crate](https://crates.io/crates/atomic-write-file) -- Atomic file write pattern using temp file + rename
- [proptest GitHub](https://github.com/proptest-rs/proptest) -- Hypothesis-like property testing for Rust with automatic shrinking
- [rust-test-assembler](https://github.com/luser/rust-test-assembler) -- Building complex binary streams for parser testing
- [Rust Type Layout Reference](https://doc.rust-lang.org/reference/type-layout.html) -- repr(C) layout guarantees for FFI struct alignment
