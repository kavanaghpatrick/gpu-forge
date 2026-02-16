---
spec: persistent-gpu-index
phase: tasks
total_tasks: 120
created: 2026-02-14
---

# Tasks: Persistent GPU-Friendly File Index

## Phase 1: Foundation — GSIX v2 Format (Make It Work)

Focus: Define the v2 binary format, IS_DELETED flag, version detection, v1 migration. This is the POC foundation all other modules depend on.

- [x] 1.1 Define GsixHeaderV2 struct
  - **Do**:
    1. Create `src/index/gsix_v2.rs` with `#[repr(C)]` struct `GsixHeaderV2`: magic (u32), version (u32=2), entry_count (u32), root_hash (u32), saved_at (u64), last_fsevents_id (u64), exclude_hash (u32), entry_capacity (u32), flags (u32), checksum (u32), _reserved ([u8; 16336])
    2. Add compile-time assertion: `size_of::<GsixHeaderV2>() == 16384`
    3. Define constants: `HEADER_SIZE_V2: usize = 16384`, `INDEX_MAGIC: u32 = 0x58495347`, `INDEX_VERSION_V2: u32 = 2`
    4. Register module in `src/index/mod.rs`
  - **Files**: `src/index/gsix_v2.rs` (new), `src/index/mod.rs`
  - **Done when**: `size_of::<GsixHeaderV2>() == 16384` compiles, module registered
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): define GsixHeaderV2 16KB page-aligned struct`
  - _Requirements: PM Section 4 (GSIX v2 binary format), PM Section 7 (Header Layout)_
  - _Design: TECH.md Section 2 (GSIX v2 Format)_

- [x] 1.2 Implement header serialization with CRC32
  - **Do**:
    1. Add `crc32fast = "1"` to Cargo.toml
    2. Implement `GsixHeaderV2::to_bytes(&self) -> [u8; 16384]` — serialize all fields little-endian, compute CRC32 over bytes [0..44) and store at checksum offset
    3. Implement `GsixHeaderV2::from_bytes(buf: &[u8]) -> Result<Self, CacheError>` — validate magic, version, checksum; return `Err(CacheError::InvalidFormat)` on failure
    4. Add `GsixHeaderV2::new(entry_count: u32, root_hash: u32) -> Self` constructor
  - **Files**: `src/index/gsix_v2.rs`, `Cargo.toml`
  - **Done when**: `to_bytes()` / `from_bytes()` roundtrip produces identical header
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib gsix_v2 2>&1 | tail -10`
  - **Commit**: `feat(index): implement GsixHeaderV2 serialization with CRC32 checksum`
  - _Requirements: PM Section 4 (GSIX v2 binary format)_
  - _Design: TECH.md Section 2 (GSIX v2 Format: header layout)_

- [x] 1.3 Add IS_DELETED flag to Rust and Metal
  - **Do**:
    1. Add `pub const IS_DELETED: u32 = 1 << 4;` to `path_flags` module in `src/gpu/types.rs`
    2. Add `#define PATH_FLAG_IS_DELETED (1u << 4)` to `shaders/search_types.h` in the Path Filter section
    3. In `shaders/path_filter.metal`, add early return: `if (entry.flags & PATH_FLAG_IS_DELETED) return;` before any matching logic
  - **Files**: `src/gpu/types.rs`, `shaders/search_types.h`, `shaders/path_filter.metal`
  - **Done when**: IS_DELETED defined in both Rust and Metal, GPU kernel skips deleted entries
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): add IS_DELETED tombstone flag to Rust and Metal shader`
  - _Requirements: TECH.md Section 2 (IS_DELETED flag)_
  - _Design: TECH.md Section 2, QA.md Section 2_

- [x] 1.4 Define header flags field constants
  - **Do**:
    1. In `src/index/gsix_v2.rs`, define `pub const FLAG_SORTED: u32 = 0x1` and `pub const FLAG_COMPACTED: u32 = 0x2`
    2. Implement helper methods on `GsixHeaderV2`: `is_sorted()`, `is_compacted()`, `set_sorted()`, `set_compacted()`
  - **Files**: `src/index/gsix_v2.rs`
  - **Done when**: Flag constants and helpers compile
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): add FLAG_SORTED and FLAG_COMPACTED header flags`
  - _Requirements: TECH.md Section 2_
  - _Design: TECH.md Section 2 (GSIX v2 Format: flags field)_

- [x] 1.5 Implement save_v2 with atomic write
  - **Do**:
    1. Implement `pub fn save_v2(entries: &[GpuPathEntry], root_hash: u32, path: &Path, last_fsevents_id: u64, exclude_hash: u32) -> Result<PathBuf, CacheError>` in `src/index/gsix_v2.rs`
    2. Write 16KB v2 header + packed entries to `.idx.tmp`
    3. Call `fsync()` via `file.sync_all()`
    4. Atomic rename `.idx.tmp` -> `.idx`
    5. Compute and store checksum in header before write
  - **Files**: `src/index/gsix_v2.rs`
  - **Done when**: save_v2 writes valid v2 index file atomically
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib gsix_v2 2>&1 | tail -10`
  - **Commit**: `feat(index): implement save_v2 with atomic write and fsync`
  - _Requirements: PM Section 4 (GSIX v2 binary format)_
  - _Design: TECH.md Section 2 (entry region, backward compatibility)_

- [x] 1.6 Implement load_v2 with validation
  - **Do**:
    1. Implement `pub fn load_v2(path: &Path) -> Result<(GsixHeaderV2, Vec<GpuPathEntry>), CacheError>` in `src/index/gsix_v2.rs`
    2. Read file, validate 16KB v2 header (magic, version, checksum)
    3. Parse all new fields: last_fsevents_id, exclude_hash, entry_capacity, flags
    4. Return entries starting at offset 16384
    5. Validate entry_count * 256 + 16384 <= file_len
  - **Files**: `src/index/gsix_v2.rs`
  - **Done when**: load_v2 reads and validates v2 index, roundtrip with save_v2 produces identical entries
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib gsix_v2 2>&1 | tail -10`
  - **Commit**: `feat(index): implement load_v2 with full header validation`
  - _Requirements: PM Section 4_
  - _Design: TECH.md Section 2_

- [x] 1.7 Implement version detection and v1 migration
  - **Do**:
    1. Implement `pub fn detect_version(data: &[u8]) -> Result<u32, CacheError>` — read magic + version, return Ok(1) for v1, Ok(2) for v2, Err for unknown
    2. Implement v1 migration: when detect_version returns 1, log warning, delete old v1 file, return `CacheError::InvalidFormat("v1 index detected, rebuild required")`
    3. Implement `pub fn cleanup_v1_indexes(index_dir: &Path) -> std::io::Result<()>` — delete all `.idx` files whose stem != "global", create `.v2-migrated` marker
  - **Files**: `src/index/gsix_v2.rs`
  - **Done when**: v1 detected and signals rebuild, v2 loaded normally, cleanup removes old files
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib gsix_v2 2>&1 | tail -10`
  - **Commit**: `feat(index): add version detection and v1-to-v2 migration path`
  - _Requirements: PM Section 4, TECH.md Section 10 (Migration)_
  - _Design: TECH.md Section 2 (backward compatibility)_

- [ ] 1.8 [VERIFY] Quality checkpoint: cargo clippy and tests
  - **Do**: Run clippy with deny warnings and all gsix_v2 tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo clippy -- -D warnings 2>&1 | tail -10 && cargo test --lib gsix_v2 2>&1 | tail -10`
  - **Done when**: No clippy warnings, all gsix_v2 tests pass
  - **Commit**: `chore(index): pass quality checkpoint for gsix v2 format` (only if fixes needed)

- [x] 1.9 Write v2 format unit tests
  - **Do**:
    1. Write tests in `src/index/gsix_v2.rs` `#[cfg(test)] mod tests`:
       - `test_header_size_is_16384` — assert sizeof
       - `test_header_magic_bytes` — verify magic = 0x58495347
       - `test_header_version_is_2` — verify version field
       - `test_header_serialization_roundtrip` — to_bytes/from_bytes identity
       - `test_save_load_roundtrip` — save_v2 + load_v2 produces identical entries (all 256 bytes each)
       - `test_checksum_validation` — corrupt 1 bit, verify load fails
       - `test_detect_version_v1` — recognizes v1 magic+version
       - `test_detect_version_v2` — recognizes v2
       - `test_detect_version_unknown` — rejects garbage magic
       - `test_entry_count_matches` — saved entry_count matches written entries
  - **Files**: `src/index/gsix_v2.rs`
  - **Done when**: 10 unit tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib gsix_v2 -- --nocapture 2>&1 | tail -20`
  - **Commit**: `test(index): add v2 format unit tests for header and roundtrip`
  - _Requirements: QA.md Section 2 (Index Format Tests)_
  - _Design: QA.md Section 2_

- [x] 1.10 Write corrupt handling tests
  - **Do**:
    1. Add tests to `src/index/gsix_v2.rs`:
       - `test_corrupt_entry_count_overflow` — entry_count larger than file
       - `test_corrupt_zero_byte_file` — 0-byte file
       - `test_corrupt_header_only_zero_entries` — header + 0 entries
       - `test_corrupt_partial_entry` — header + partial entry bytes
       - `test_corrupt_random_bytes` — random 1024 bytes (no panic)
       - `test_corrupt_wrong_endianness` — big-endian magic
       - `test_corrupt_trailing_garbage` — valid file + extra bytes (still loads)
       - `test_corrupt_all_zeros` — all-zero header
  - **Files**: `src/index/gsix_v2.rs`
  - **Done when**: 8 corrupt handling tests pass, none panic
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib gsix_v2::tests::test_corrupt 2>&1 | tail -20`
  - **Commit**: `test(index): add corrupt file handling tests for v2 format`
  - _Requirements: QA.md Section 2 (corrupt handling)_
  - _Design: UX.md Section 7.2.1 (Index Corrupt error handling)_

- [ ] 1.11 [VERIFY] Phase 1 POC checkpoint
  - **Do**:
    1. Run all existing tests to verify no regressions
    2. Run all gsix_v2 tests
    3. Verify cargo clippy clean
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -5 && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass, clippy clean, v2 format fully functional
  - **Commit**: `feat(index): complete GSIX v2 format POC`

## Phase 2: Zero-Copy Pipeline (mmap-gpu-pipeline)

Focus: mmap v2 index files, create Metal bytesNoCopy buffers, implement arc-swap IndexStore.

- [x] 2.1 Update MmapIndexCache for v2 headers
  - **Do**:
    1. Update `MmapIndexCache::load_mmap()` in `src/index/cache.rs` to handle v2 headers
    2. Add `detect_version()` call: if v1, return `CacheError::InvalidFormat` (signals rebuild)
    3. For v2: validate header at [0..16384), read entry_count and v2 fields, calculate entry region at offset 16384
    4. Preserve backward-compatible behavior: old v1 files detected and rejected gracefully
  - **Files**: `src/index/cache.rs`
  - **Done when**: MmapIndexCache loads v2 files, rejects v1 files with clear error
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib cache 2>&1 | tail -10`
  - **Commit**: `feat(index): update MmapIndexCache to handle v2 16KB headers`
  - _Requirements: PM Section 4 (mmap-based instant load)_
  - _Design: TECH.md Section 6 (Page Alignment)_

- [x] 2.2 Implement Metal bytesNoCopy buffer creation
  - **Do**:
    1. Create `src/index/metal_buffer.rs` with function `create_gpu_buffer(device, mmap_ptr, offset, length) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>>`
    2. Use Strategy A: mmap full file, call `device.newBufferWithBytesNoCopy_length_options_deallocator_()` on full mmap region
    3. Set buffer offset to `HEADER_SIZE_V2` (16384) in GPU dispatch calls
    4. Fallback: if `bytesNoCopy` returns None, use `newBufferWithBytes` copy path
    5. Register module in `src/index/mod.rs`
  - **Files**: `src/index/metal_buffer.rs` (new), `src/index/mod.rs`
  - **Done when**: bytesNoCopy buffer creation succeeds on Apple Silicon, fallback works
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement Metal bytesNoCopy zero-copy buffer creation`
  - _Requirements: PM Section 4 (Metal bytesNoCopy zero-copy buffer)_
  - _Design: TECH.md Section 6 (strategies A vs B, alignment verification)_

- [x] 2.3 Define IndexSnapshot struct
  - **Do**:
    1. In `src/index/metal_buffer.rs` (or new `src/index/snapshot.rs`), define `IndexSnapshot`:
       - `mmap: MmapBuffer` (keeps mapping alive)
       - `metal_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>` (GPU buffer)
       - `entry_count: usize`
       - `fsevents_id: u64`
       - `header: GsixHeaderV2`
    2. Implement `Drop` to ensure proper cleanup: Metal buffer dropped before mmap
    3. Implement `IndexSnapshot::from_file(path, device) -> Result<Self, CacheError>`
    4. Add `entries(&self) -> &[GpuPathEntry]` accessor via mmap pointer arithmetic
  - **Files**: `src/index/snapshot.rs` (new) or `src/index/metal_buffer.rs`, `src/index/mod.rs`
  - **Done when**: IndexSnapshot holds mmap + Metal buffer with correct lifetime semantics
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): define IndexSnapshot struct with mmap-anchored Metal buffer`
  - _Requirements: TECH.md Section 7 (Concurrency)_
  - _Design: TECH.md Section 7 (ArcSwap snapshot pattern)_

- [x] 2.4 Implement IndexStore with arc-swap
  - **Do**:
    1. Add `arc-swap = "1"` to Cargo.toml
    2. Create `src/index/store.rs` with `IndexStore`:
       - Inner: `ArcSwap<Option<IndexSnapshot>>`
       - `snapshot(&self) -> arc_swap::Guard<Arc<Option<IndexSnapshot>>>` — lock-free reader
       - `swap(&self, new: IndexSnapshot)` — atomic writer swap
       - `is_available(&self) -> bool` — check if snapshot exists
    3. Register module in `src/index/mod.rs`
  - **Files**: `src/index/store.rs` (new), `src/index/mod.rs`, `Cargo.toml`
  - **Done when**: IndexStore compiles, provides lock-free reader/writer access
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement IndexStore with arc-swap for lock-free snapshots`
  - _Requirements: TECH.md Section 7 (Concurrency)_
  - _Design: TECH.md Section 7_

- [ ] 2.5 [VERIFY] Quality checkpoint: clippy + build + existing tests
  - **Do**: Run clippy, build, and all existing tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo clippy -- -D warnings 2>&1 | tail -10 && cargo test 2>&1 | tail -10`
  - **Done when**: No clippy warnings, all existing tests pass
  - **Commit**: `chore(index): pass quality checkpoint for mmap pipeline` (only if fixes needed)

- [x] 2.6 Eliminate copy path in GpuIndexLoader
  - **Do**:
    1. Modify `GpuIndexLoader::try_load_from_cache()` (or `try_index_producer` in orchestrator.rs) to use the new mmap->bytesNoCopy path
    2. When IndexStore has a valid snapshot, use it directly for path iteration instead of loading via `MmapIndexCache::into_resident_index()`
    3. Keep `into_resident_index()` as fallback for non-Metal contexts
  - **Files**: `src/search/orchestrator.rs`, `src/index/gpu_loader.rs`
  - **Done when**: Search pipeline uses mmap-backed snapshot instead of copy
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib orchestrator 2>&1 | tail -10`
  - **Commit**: `refactor(search): use mmap-backed IndexSnapshot instead of copy path`
  - _Requirements: PM Section 4_
  - _Design: TECH.md Section 1.2 (Data Flow)_

- [x] 2.7 Integrate IndexStore with try_index_producer
  - **Do**:
    1. Update `SearchOrchestrator::try_index_producer()` to check `IndexStore::snapshot()` first
    2. If valid snapshot available and not stale, iterate entries from snapshot into the channel
    3. Fall back to `walk_and_filter()` when no snapshot available
    4. Keep existing `MmapIndexCache` load path as secondary fallback
  - **Files**: `src/search/orchestrator.rs`
  - **Done when**: try_index_producer prefers IndexStore snapshot, falls back correctly
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib orchestrator 2>&1 | tail -10`
  - **Commit**: `feat(search): integrate IndexStore into search pipeline with fallback`
  - _Requirements: PM Section 4_
  - _Design: TECH.md Section 1.2_

- [x] 2.8 Add alignment verification assertions
  - **Do**:
    1. Add compile-time assertions: `HEADER_SIZE_V2 == 16384`, `HEADER_SIZE_V2 % 16384 == 0`, `size_of::<GpuPathEntry>() == 256`
    2. Add runtime assertion in load path: verify `(mmap_ptr as usize + HEADER_SIZE_V2) % page_size == 0` where page_size = 16384 on Apple Silicon
  - **Files**: `src/index/gsix_v2.rs`, `src/index/cache.rs` or `src/index/snapshot.rs`
  - **Done when**: Compile-time and runtime alignment assertions pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib 2>&1 | tail -5`
  - **Commit**: `feat(index): add compile-time and runtime alignment verification`
  - _Requirements: TECH.md Section 6 (Page Alignment)_
  - _Design: TECH.md Section 6_

- [x] 2.9 Write mmap pipeline unit tests
  - **Do**:
    1. Write tests in appropriate test modules:
       - `test_mmap_v2_page_aligned` — mmap of v2 file has entries at page boundary
       - `test_bytesnocopy_succeeds` — Metal buffer creation from page-aligned mmap
       - `test_metal_buffer_contents_match` — buffer data matches mmap byte-for-byte
       - `test_buffer_length_covers_entries` — buffer length = entry_count * 256
       - `test_index_store_snapshot_consistent` — snapshot returns consistent data
       - `test_index_store_swap_visible` — swap makes new data visible to new readers
       - `test_mmap_fallback_to_copy` — bytesNoCopy failure falls back to copy
  - **Files**: `src/index/snapshot.rs` or `src/index/store.rs` (tests), `tests/test_index.rs`
  - **Done when**: 7+ mmap pipeline tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test index_store 2>&1 | tail -15 && cargo test mmap_v2 2>&1 | tail -15`
  - **Commit**: `test(index): add mmap pipeline and IndexStore unit tests`
  - _Requirements: QA.md Section 5.1, Section 9_
  - _Design: TECH.md Section 6, Section 7_

- [ ] 2.10 [VERIFY] Phase 2 checkpoint: full test suite
  - **Do**: Run complete test suite, verify no regressions, clippy clean
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -10 && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass including new mmap tests, zero regressions
  - **Commit**: `feat(index): complete zero-copy mmap-to-GPU pipeline`

## Phase 3: Live Updates — FSEvents Watcher

Focus: Replace notify-based kqueue watcher with direct macOS FSEvents via fsevent-sys.

- [x] 3.1 Add fsevent-sys and core-foundation dependencies
  - **Do**:
    1. Add to Cargo.toml: `fsevent-sys = "4"`, `core-foundation = "0.10"`
    2. Verify compilation on macOS ARM64
  - **Files**: `Cargo.toml`
  - **Done when**: `cargo build` succeeds with new dependencies
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(deps): add fsevent-sys and core-foundation for FSEvents integration`
  - _Requirements: PM Section 4 (FSEvents watcher migration)_
  - _Design: TECH.md Section 3 (crate selection)_

- [x] 3.2 Define FsChange enum
  - **Do**:
    1. Create `src/index/fsevents.rs` with `FsChange` enum: `Created(PathBuf)`, `Modified(PathBuf)`, `Deleted(PathBuf)`, `Renamed { old: PathBuf, new: PathBuf }`, `MustRescan(PathBuf)`, `HistoryDone`
    2. Register in `src/index/mod.rs`
  - **Files**: `src/index/fsevents.rs` (new), `src/index/mod.rs`
  - **Done when**: FsChange enum compiles, module registered
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): define FsChange enum for filesystem event types`
  - _Requirements: TECH.md Section 3_
  - _Design: TECH.md Section 3.5 (flag mapping table)_

- [x] 3.3 Implement ExcludeTrie
  - **Do**:
    1. Create `src/index/exclude.rs` with `ExcludeTrie`:
       - Tier 1: absolute path prefix matching (`/System`, `/Volumes`, `/dev`, `/cores`, `/private/var`, `/private/tmp`)
       - Tier 2: user-relative prefixes (`~/Library/Caches`, `~/.Trash`)
       - Tier 3: directory basename via `HashSet` (`.git`, `node_modules`, `target`, `__pycache__`, `vendor`, `dist`, `build`, `.cache`, `venv`, `.venv`, `.Spotlight-V100`, `.fseventsd`, `.DS_Store`, `.Trashes`, `.hg`, `.svn`, `.idea`, `.vscode`)
       - Include overrides list
    2. Implement `should_exclude(&self, path: &[u8]) -> bool`
    3. Register in `src/index/mod.rs`
  - **Files**: `src/index/exclude.rs` (new), `src/index/mod.rs`
  - **Done when**: ExcludeTrie filters all 3 tiers correctly
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib exclude 2>&1 | tail -10`
  - **Commit**: `feat(index): implement ExcludeTrie with 3-tier path filtering`
  - _Requirements: TECH.md Section 9 (Exclude List)_
  - _Design: TECH.md Section 9 (tiers, ExcludeTrie)_

- [x] 3.4 Implement exclude hash computation
  - **Do**:
    1. In `src/index/exclude.rs`, implement `compute_exclude_hash(excludes: &ExcludeTrie) -> u32`
    2. Sort all patterns, hash with CRC32 using NUL separators
    3. This hash is stored in GSIX v2 header; mismatch on startup triggers rebuild
  - **Files**: `src/index/exclude.rs`
  - **Done when**: Deterministic exclude hash computable from ExcludeTrie
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib exclude 2>&1 | tail -10`
  - **Commit**: `feat(index): implement exclude hash for config change detection`
  - _Requirements: TECH.md Section 9_
  - _Design: TECH.md Section 9 (exclude hash)_

- [ ] 3.5 [VERIFY] Quality checkpoint: clippy + build
  - **Do**: Run clippy and build to catch issues early
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo clippy -- -D warnings 2>&1 | tail -10 && cargo test 2>&1 | tail -5`
  - **Done when**: No clippy warnings, build succeeds
  - **Commit**: `chore(index): pass quality checkpoint for FSEvents foundation` (only if fixes needed)

- [x] 3.6 Implement FSEventsListener struct
  - **Do**:
    1. In `src/index/fsevents.rs`, implement `FSEventsListener`:
       - Fields: CFRunLoop thread handle, `crossbeam::Sender<FsChange>`, `Arc<AtomicU64>` for last_event_id, `Arc<ExcludeTrie>`, shutdown `Arc<AtomicBool>`
       - `new(excludes, change_tx, last_event_id) -> Self`
       - `start(&self) -> Result<(), ...>` — spawns thread
  - **Files**: `src/index/fsevents.rs`
  - **Done when**: FSEventsListener struct compiles with all fields
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement FSEventsListener struct skeleton`
  - _Requirements: PM Section 4 (FSEvents watcher migration)_
  - _Design: TECH.md Section 3 (FSEvents Integration: listener design)_

- [x] 3.7 Implement CFRunLoop thread and stream creation
  - **Do**:
    1. Spawn dedicated thread running `CFRunLoopRun()`
    2. Create FSEvents stream with: paths `["/"]`, `sinceWhen` from stored event ID (or `kFSEventStreamEventIdSinceNow`), latency 0.5s
    3. Flags: `kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagUseCFTypes | kFSEventStreamCreateFlagNoDefer`
    4. Schedule on thread's run loop
  - **Files**: `src/index/fsevents.rs`
  - **Done when**: FSEvents stream created and running on dedicated thread
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement CFRunLoop thread with FSEvents stream`
  - _Requirements: PM Section 4_
  - _Design: TECH.md Section 3 (FSEvents Integration)_

- [x] 3.8 Implement FSEvents callback with event mapping
  - **Do**:
    1. Implement FSEvents callback function receiving event batches
    2. For each event: extract path + flags, filter via `ExcludeTrie::should_exclude()`
    3. Map FSEvents flags to `FsChange` variants per TECH.md Section 3.5 flag mapping table
    4. Send via `change_tx` channel
    5. Update `last_event_id` atomically after each batch
    6. Handle `kFSEventStreamEventFlagHistoryDone` -> `FsChange::HistoryDone`
  - **Files**: `src/index/fsevents.rs`
  - **Done when**: Events flow from FSEvents -> FsChange enum -> channel
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement FSEvents callback with flag-to-FsChange mapping`
  - _Requirements: TECH.md Section 3.5_
  - _Design: TECH.md Section 3 (event flag mapping)_

- [x] 3.9 Implement event ID resume from stored header
  - **Do**:
    1. On startup, read `last_fsevents_id` from GSIX v2 header
    2. Pass as `sinceWhen` to `FSEventStreamCreate`
    3. Handle edge cases: event ID regression (volume reformat) -> trigger full rebuild
    4. Handle journal truncation (stored ID older than oldest retained) -> trigger full rebuild
  - **Files**: `src/index/fsevents.rs`
  - **Done when**: FSEvents resumes from stored event ID, handles edge cases
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement FSEvents event ID resume from stored header`
  - _Requirements: TECH.md Section 3 (event ID persistence)_
  - _Design: TECH.md Section 3_

- [x] 3.10 Implement clean shutdown
  - **Do**:
    1. `FSEventsListener::stop()`: set shutdown flag, `FSEventStreamStop()`, `FSEventStreamInvalidate()`, `FSEventStreamRelease()`, stop CFRunLoop, join thread
    2. Implement `Drop` to call `stop()`
  - **Files**: `src/index/fsevents.rs`
  - **Done when**: Clean shutdown with no leaked threads or FDs
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement FSEventsListener clean shutdown`
  - _Requirements: QA.md Section 3_
  - _Design: TECH.md Section 3_

- [x] 3.11 Feature-gate notify crate
  - **Do**:
    1. Feature-gate `notify` and `notify-debouncer-mini` behind `#[cfg(not(target_os = "macos"))]` in Cargo.toml (use `[target.'cfg(not(target_os = "macos"))'.dependencies]`)
    2. Feature-gate old `IndexWatcher` code in `src/index/watcher.rs` behind same cfg
    3. On macOS, `FSEventsListener` is used unconditionally
  - **Files**: `Cargo.toml`, `src/index/watcher.rs`, `src/index/mod.rs`
  - **Done when**: notify not compiled on macOS, FSEventsListener used instead
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `refactor(index): feature-gate notify crate behind non-macOS cfg`
  - _Requirements: TECH.md Section 3_
  - _Design: TECH.md Section 3 (crate selection)_

- [x] 3.12 Write ExcludeTrie tests
  - **Do**:
    1. Write unit tests in `src/index/exclude.rs`:
       - `test_exclude_absolute_prefix` — /System excluded
       - `test_exclude_basename` — `.git` excluded
       - `test_include_override` — include overrides take priority
       - `test_empty_trie` — empty trie excludes nothing
       - `test_deeply_nested_excluded` — /foo/bar/.git/objects excluded
       - `test_non_excluded_passes` — /Users/dev/code passes
       - `test_user_relative_prefix` — ~/Library/Caches excluded
       - `test_exclude_hash_deterministic` — same config -> same hash
  - **Files**: `src/index/exclude.rs`
  - **Done when**: 8 ExcludeTrie tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib exclude 2>&1 | tail -15`
  - **Commit**: `test(index): add ExcludeTrie unit tests`
  - _Requirements: QA.md Section 3_
  - _Design: TECH.md Section 9_

- [x] 3.13 [VERIFY] Phase 3 checkpoint: full test suite
  - **Do**: Run complete test suite, verify no regressions
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -10 && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass, clippy clean, FSEvents integration functional
  - **Commit**: `feat(index): complete FSEvents watcher integration`

## Phase 4: Incremental Writer

Focus: IndexWriter with CRUD operations, flush strategy, compaction, snapshot swap.

- [x] 4.1 Define IndexWriter struct
  - **Do**:
    1. Create `src/index/index_writer.rs` with `IndexWriter`:
       - `entries: Vec<GpuPathEntry>`
       - `path_index: HashMap<Box<[u8]>, usize>` for O(1) lookup
       - `dirty_count: usize`
       - `last_flush: Instant`
       - `excludes: Arc<ExcludeTrie>`
       - `index_store: Arc<IndexStore>`
    2. Implement `from_snapshot(snapshot: &IndexSnapshot, excludes, store) -> Self` — build HashMap from loaded entries
    3. Register in `src/index/mod.rs`
  - **Files**: `src/index/index_writer.rs` (new), `src/index/mod.rs`
  - **Done when**: IndexWriter struct compiles, initializable from snapshot
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): define IndexWriter struct with HashMap lookup`
  - _Requirements: PM Phase 3 (Incremental Update)_
  - _Design: TECH.md Section 4 (Incremental Update Strategy)_

- [x] 4.2 Implement Created handler
  - **Do**:
    1. `IndexWriter::handle_created(path: &Path)`:
       - Check `excludes.should_exclude()`
       - `stat()` the path (skip if race-deleted)
       - Build `GpuPathEntry` from metadata
       - Check `path_index` for existing (treat as Modified if found)
       - Otherwise append to `entries` and record in `path_index`
       - Increment `dirty_count`
  - **Files**: `src/index/index_writer.rs`
  - **Done when**: Created handler adds entries correctly
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib index_writer 2>&1 | tail -10`
  - **Commit**: `feat(index): implement IndexWriter Created handler`
  - _Requirements: TECH.md Section 4_
  - _Design: TECH.md Section 4 (event processing logic)_

- [x] 4.3 Implement Modified and Deleted handlers
  - **Do**:
    1. `handle_modified(path)`: lookup in path_index (Created if not found), stat (Deleted if fails), update mtime/size/flags in place
    2. `handle_deleted(path)`: lookup in path_index (skip if not found), set `IS_DELETED` flag, remove from path_index
    3. Both increment dirty_count
  - **Files**: `src/index/index_writer.rs`
  - **Done when**: Modified updates in place, Deleted tombstones correctly
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib index_writer 2>&1 | tail -10`
  - **Commit**: `feat(index): implement Modified and Deleted handlers with tombstoning`
  - _Requirements: TECH.md Section 4_
  - _Design: TECH.md Section 4_

- [x] 4.4 Implement Renamed and MustRescan handlers
  - **Do**:
    1. `handle_renamed(old, new)`: `handle_deleted(old)` + `handle_created(new)`
    2. `handle_must_rescan(subtree)`: walk subtree via `FilesystemScanner`, collect found paths. For existing entries with subtree prefix: update if found, tombstone if not. For new found paths: insert
  - **Files**: `src/index/index_writer.rs`
  - **Done when**: Rename and rescan handlers functional
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib index_writer 2>&1 | tail -10`
  - **Commit**: `feat(index): implement Renamed and MustRescan handlers`
  - _Requirements: TECH.md Section 4_
  - _Design: TECH.md Section 4_

- [x] 4.5 [VERIFY] Quality checkpoint: clippy + tests
  - **Do**: Verify IndexWriter handlers compile and pass tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo clippy -- -D warnings 2>&1 | tail -10 && cargo test --lib index_writer 2>&1 | tail -10`
  - **Done when**: No clippy warnings, all index_writer tests pass
  - **Commit**: `chore(index): pass quality checkpoint for IndexWriter handlers` (only if fixes needed)

- [x] 4.6 Implement event dispatch and flush trigger
  - **Do**:
    1. `process_event(change: FsChange)`: dispatch to appropriate handler based on variant; handle `HistoryDone` by logging + immediate flush
    2. Flush condition check: `dirty_count >= 1000`, `last_flush.elapsed() >= 30s`, `HistoryDone` received, or shutdown signal
  - **Files**: `src/index/index_writer.rs`
  - **Done when**: Events dispatched, flush triggers on conditions
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib index_writer 2>&1 | tail -10`
  - **Commit**: `feat(index): implement event dispatch and flush trigger logic`
  - _Requirements: PM Section 4 (Incremental index update)_
  - _Design: TECH.md Section 4 (flush strategy)_

- [x] 4.7 Implement compaction
  - **Do**:
    1. In `flush()`: check tombstone ratio; if > 20%, compact
    2. Compaction: remove entries with `IS_DELETED`, shift remaining to close gaps, rebuild `path_index` with new positions
    3. After compaction, sort entries by path bytes if any were appended (unsorted)
    4. Set `FLAG_SORTED` and `FLAG_COMPACTED` in header flags
  - **Files**: `src/index/index_writer.rs`
  - **Done when**: Compaction removes tombstones and re-sorts
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib index_writer 2>&1 | tail -10`
  - **Commit**: `feat(index): implement tombstone compaction with 20% threshold`
  - _Requirements: PM Section 4 (Scope: index compaction)_
  - _Design: TECH.md Section 4 (amortized costs)_

- [x] 4.8 Implement atomic flush and snapshot swap
  - **Do**:
    1. `flush()`: write v2 header (updated entry_count, entry_capacity, last_fsevents_id, saved_at, checksum) + entries to `global.idx.tmp`, fsync, rename
    2. After successful write: create new IndexSnapshot (mmap new file, create Metal buffer), call `IndexStore::swap()`
    3. Reset dirty_count = 0, last_flush = Instant::now()
  - **Files**: `src/index/index_writer.rs`
  - **Done when**: Flush writes atomically, snapshot swap makes new data available
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib index_writer 2>&1 | tail -10`
  - **Commit**: `feat(index): implement atomic flush with snapshot swap`
  - _Requirements: PM Section 4_
  - _Design: TECH.md Section 4_

- [x] 4.9 Implement writer thread
  - **Do**:
    1. Spawn IndexWriter on dedicated thread receiving `FsChange` events via crossbeam bounded channel
    2. Process events in loop, check flush conditions after each event/batch
    3. Handle shutdown via channel disconnect or atomic flag
  - **Files**: `src/index/index_writer.rs`
  - **Done when**: Writer thread processes events and flushes periodically
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement IndexWriter thread with event loop`
  - _Requirements: TECH.md Section 4_
  - _Design: TECH.md Section 4_

- [x] 4.10 Write CRUD unit tests
  - **Do**:
    1. Tests in `src/index/index_writer.rs`:
       - `test_created_appears_in_index` — create file, verify in entries
       - `test_modified_updates_mtime` — modify file, verify mtime/size changed
       - `test_deleted_tombstones_entry` — delete, verify IS_DELETED flag set
       - `test_renamed_removes_old_adds_new` — rename removes old + adds new
       - `test_compaction_removes_tombstones` — tombstones removed after compaction
       - `test_flush_writes_valid_v2` — flush produces loadable v2 file
       - `test_dirty_count_triggers_flush` — 1000 changes triggers flush
       - `test_path_index_o1_lookup` — HashMap lookup is correct
  - **Files**: `src/index/index_writer.rs`
  - **Done when**: 8 CRUD tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib index_writer 2>&1 | tail -20`
  - **Commit**: `test(index): add IndexWriter CRUD unit tests`
  - _Requirements: QA.md Section 4 (Incremental Update Tests)_
  - _Design: QA.md Section 4_

- [x] 4.11 [VERIFY] Phase 4 checkpoint
  - **Do**: Full test suite, clippy
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -10 && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass, clippy clean
  - **Commit**: `feat(index): complete incremental update writer`

## Phase 5: Global Root Index and UI

Focus: Single global.idx for /, background initial build, status bar integration, orchestrator preference.

- [x] 5.1 Implement global index path helpers
  - **Do**:
    1. In `src/index/gsix_v2.rs` or new `src/index/global.rs`: `global_index_path() -> PathBuf` returning `~/.gpu-search/index/global.idx`
    2. `global_cache_key() -> u32` returning hash of `"/"`
    3. Ensure `create_dir_all` for index directory
  - **Files**: `src/index/gsix_v2.rs` or `src/index/global.rs`, `src/index/mod.rs`
  - **Done when**: Global path helpers compile and return correct paths
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib global 2>&1 | tail -10`
  - **Commit**: `feat(index): implement global index path helpers`
  - _Requirements: TECH.md Section 5 (Global Root Index)_
  - _Design: TECH.md Section 5_

- [x] 5.2 Expand DEFAULT_EXCLUDES with system-root paths
  - **Do**:
    1. In `src/index/exclude.rs`, create `default_excludes() -> ExcludeTrie` with expanded patterns:
       - Tier 1: `/System`, `/Library/Caches`, `/private/var`, `/private/tmp`, `/Volumes`, `/dev`, `/cores`
       - Tier 2: `~/Library/Caches`, `~/.Trash`
       - Tier 3: `.git`, `node_modules`, `target`, `__pycache__`, `vendor`, `dist`, `build`, `.cache`, `venv`, `.venv`, `.Spotlight-V100`, `.fseventsd`, `.DS_Store`, `.Trashes`, `.hg`, `.svn`, `.idea`, `.vscode`
  - **Files**: `src/index/exclude.rs`
  - **Done when**: Default excludes cover all system paths from TECH.md
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib exclude 2>&1 | tail -10`
  - **Commit**: `feat(index): expand default excludes with system-root paths`
  - _Requirements: PM Section 4 (expanded exclude list)_
  - _Design: TECH.md Section 9 (Exclude List: tiers)_

- [x] 5.3 Implement optional config file loading
  - **Do**:
    1. Implement loading from `~/.gpu-search/config.json`
    2. Parse `exclude_dirs`, `exclude_absolute`, `include_override` arrays
    3. Merge with hardcoded defaults
    4. If no config file, use defaults only
  - **Files**: `src/index/exclude.rs` or `src/index/global.rs`
  - **Done when**: Config file parsed and merged with defaults
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib exclude 2>&1 | tail -10`
  - **Commit**: `feat(index): implement optional config.json for exclude customization`
  - _Requirements: TECH.md Section 9 (runtime configuration)_
  - _Design: TECH.md Section 9_

- [x] 5.4 Implement background initial build
  - **Do**:
    1. On first launch (no `global.idx`), spawn background thread running `FilesystemScanner::scan("/")` with expanded excludes
    2. Use `Arc<AtomicUsize>` progress counter incremented per file scanned
    3. On completion, save v2 index and create initial IndexSnapshot in IndexStore
  - **Files**: `src/index/index_writer.rs` or `src/index/daemon.rs` (new)
  - **Done when**: Background build completes, saves v2 index, creates snapshot
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement background initial build with progress counter`
  - _Requirements: PM Section 4 (Background initial scan)_
  - _Design: TECH.md Section 5 (first-run)_

- [ ] 5.5 [VERIFY] Quality checkpoint
  - **Do**: Clippy + build + tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo clippy -- -D warnings 2>&1 | tail -10 && cargo test 2>&1 | tail -5`
  - **Done when**: Clean build, all tests pass
  - **Commit**: `chore(index): pass quality checkpoint for global index` (only if fixes needed)

- [x] 5.6 Define IndexState enum and wire into app
  - **Do**:
    1. Define `IndexState` enum in `src/ui/status_bar.rs` or shared types: `Loading`, `Building { files_indexed: usize }`, `Ready { file_count: usize }`, `Updating`, `Stale { file_count: usize }`, `Error { message: String }`
    2. Add `index_state: IndexState` field to `GpuSearchApp`
    3. Wire state transitions: Building on scan start, Ready on completion, Updating on FSEvents flush, Stale on staleness detection, Error on failures
  - **Files**: `src/ui/status_bar.rs`, `src/ui/app.rs`
  - **Done when**: IndexState transitions correctly in response to index lifecycle events
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(ui): define IndexState enum and wire into app lifecycle`
  - _Requirements: UX.md Section 3 (Index Status Indicators)_
  - _Design: UX.md Sections 3-6_

- [x] 5.7 Update StatusBar to render index state
  - **Do**:
    1. Update `StatusBar` to include `index_state: IndexState` field
    2. Prepend index status as first segment in `render()`
    3. Color-code: SUCCESS (#9ECE6A) for Ready, ACCENT (#E0AF68) for Building/Updating/Stale, ERROR (#F7768E) for Error
    4. Format file count: exact if <1K, K if 1K-999K, M if >=1M
    5. Building state: `Indexing: 420K files`
    6. Ready state: `1.3M files`
    7. Stale state: `1.3M files (stale)`
    8. Error state: `Index: error`
  - **Files**: `src/ui/status_bar.rs`
  - **Done when**: Status bar renders all IndexState variants with correct colors and formatting
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib status_bar 2>&1 | tail -15`
  - **Commit**: `feat(ui): render IndexState in status bar with color coding`
  - _Requirements: UX.md Section 8 (Status Bar Integration)_
  - _Design: UX.md Section 3, Section 10 (First-Run Progress Detail)_

- [x] 5.8 Update SearchOrchestrator to prefer index
  - **Do**:
    1. Update `SearchOrchestrator` to accept `Arc<IndexStore>`
    2. On search: check `IndexStore::snapshot()`. If available and not stale, use Metal buffer for GPU dispatch
    3. If unavailable, fall back to `walk_and_filter()`
    4. Both streaming and blocking search APIs must work
  - **Files**: `src/search/orchestrator.rs`
  - **Done when**: Orchestrator prefers index, falls back to walk
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib orchestrator 2>&1 | tail -10`
  - **Commit**: `feat(search): orchestrator prefers IndexStore snapshot over walk`
  - _Requirements: PM Section 4_
  - _Design: TECH.md Section 5_

- [x] 5.9 Implement stale detection
  - **Do**:
    1. On launch, check `saved_at` against `DEFAULT_MAX_AGE`
    2. If stale, set `IndexState::Stale`, start FSEvents watcher to trigger background update
    3. Stale index still serves search (fast but possibly incomplete) while update runs
  - **Files**: `src/index/gsix_v2.rs` or `src/index/daemon.rs`
  - **Done when**: Stale index detected, triggers update, still serves search
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement stale index detection with background update`
  - _Requirements: UX.md Section 6 (Startup Experience)_
  - _Design: TECH.md Section 5_

- [x] 5.10 Implement IndexDaemon coordination
  - **Do**:
    1. Create `src/index/daemon.rs` with `IndexDaemon`:
       - Orchestrates full lifecycle: load or build index on startup, start FSEventsListener, start IndexWriter thread, handle shutdown
       - `start(device, store) -> Self` — top-level entry point
       - `shutdown(&self)` — graceful teardown
    2. Register in `src/index/mod.rs`
  - **Files**: `src/index/daemon.rs` (new), `src/index/mod.rs`
  - **Done when**: IndexDaemon coordinates all index subsystems
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(index): implement IndexDaemon lifecycle coordinator`
  - _Requirements: TECH.md Section 5_
  - _Design: TECH.md Section 1 (Architecture)_

- [x] 5.11 Request repaint during Building state
  - **Do**:
    1. In `app.rs`, call `ctx.request_repaint()` when `index_state` is `Building` to keep progress counter updating each frame
    2. Condition: `is_searching || matches!(index_state, IndexState::Building { .. })`
  - **Files**: `src/ui/app.rs`
  - **Done when**: UI repaints during building state showing progressive count
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(ui): request repaint during index building for live progress`
  - _Requirements: UX.md Section 4 (First-Run Experience)_
  - _Design: UX.md Section 10_

- [x] 5.12 Write status bar tests
  - **Do**:
    1. Add tests to `src/ui/status_bar.rs`:
       - `test_index_state_ready_display` — "1.3M files"
       - `test_index_state_building_display` — "Indexing: 420K files"
       - `test_index_state_stale_display` — "1.3M files (stale)"
       - `test_index_state_error_display` — "Index: error"
       - `test_file_count_k_formatting` — 1500 -> "1.5K"
       - `test_file_count_m_formatting` — 1300000 -> "1.3M"
  - **Files**: `src/ui/status_bar.rs`
  - **Done when**: 6 status bar tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib status_bar 2>&1 | tail -15`
  - **Commit**: `test(ui): add IndexState status bar rendering tests`
  - _Requirements: UX.md Section 8_
  - _Design: UX.md Section 3_

- [ ] 5.13 [VERIFY] Phase 5 checkpoint: full test suite
  - **Do**: Run complete test suite, verify all tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -10 && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass, clippy clean, global index and UI integration complete
  - **Commit**: `feat(index): complete global root index with UI integration`

## Phase 6: Testing and Reliability

Focus: Comprehensive test suite -- property tests, benchmarks, scale tests, crash safety, edge cases, regressions.

- [x] 6.1 Implement synthetic entry generator
  - **Do**:
    1. Create shared test utility `generate_synthetic_entries(count: usize) -> Vec<GpuPathEntry>` in test helper module (e.g. in `tests/helpers/mod.rs` or `src/index/gsix_v2.rs` under `#[cfg(test)]`)
    2. Generate realistic paths with varied depths (1-8), extensions (.rs, .txt, .js, .py, .md), sizes, mtimes
  - **Files**: test helper module
  - **Done when**: Generator produces realistic entries for any count
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test synthetic_entries 2>&1 | tail -10`
  - **Commit**: `test(index): add synthetic GpuPathEntry generator for test/bench`
  - _Requirements: QA.md_
  - _Design: QA.md_

- [x] 6.2 Write property tests for format roundtrip
  - **Do**:
    1. Create `tests/test_index_proptest.rs` with proptest tests:
       - `prop_save_load_roundtrip` — random entries survive save/load (5000 iterations)
       - `prop_save_mmap_roundtrip` — random entries survive save/mmap
       - `prop_entry_size_invariant` — always 256B
       - `prop_path_len_bounded` — always <= 224
  - **Files**: `tests/test_index_proptest.rs` (new)
  - **Done when**: 4 property tests pass at 5000 iterations
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && PROPTEST_CASES=100 cargo test --test test_index_proptest 2>&1 | tail -15`
  - **Commit**: `test(index): add proptest format roundtrip property tests`
  - _Requirements: QA.md (Property tests)_
  - _Design: QA.md_

- [x] 6.3 Write property tests for binary fuzzing
  - **Do**:
    1. In `tests/test_index_proptest.rs`:
       - `prop_corrupt_bytes_no_panic` — flip random bytes in valid index, load returns Err
       - `prop_truncated_file_no_panic` — truncate at random offset
       - `prop_random_bytes_no_panic` — 64-8192 random bytes as input
       - `prop_extended_file_no_panic` — append random bytes to valid index
    2. All must never panic; run at 5000 iterations
  - **Files**: `tests/test_index_proptest.rs`
  - **Done when**: 4 fuzz property tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && PROPTEST_CASES=100 cargo test --test test_index_proptest corrupt 2>&1 | tail -15`
  - **Commit**: `test(index): add proptest binary fuzzing tests`
  - _Requirements: QA.md (Property tests)_
  - _Design: QA.md_

- [ ] 6.4 [VERIFY] Quality checkpoint
  - **Do**: All tests including proptests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -10 && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass
  - **Commit**: `chore(index): pass quality checkpoint for testing phase` (only if fixes needed)

- [x] 6.5 Create index load benchmarks
  - **Do**:
    1. Create `benches/index_load.rs` with criterion benchmarks:
       - `bench_mmap_load_100k` (target <1ms)
       - `bench_mmap_load_1m` (target <5ms)
       - `bench_mmap_first_entry_access` (target <0.1ms)
       - `bench_save_index_100k` (target <100ms)
       - `bench_save_index_1m` (target <1s)
    2. Add `[[bench]] name = "index_load"` to Cargo.toml
  - **Files**: `benches/index_load.rs` (new), `Cargo.toml`
  - **Done when**: 5 benchmarks run and produce results
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo bench --bench index_load 2>&1 | tail -20`
  - **Commit**: `test(index): add criterion benchmarks for index load/save`
  - _Requirements: PM Section 3 (Success Metrics)_
  - _Design: QA.md Section 5.1_

- [x] 6.6 Write scale tests (1M entries)
  - **Do**:
    1. Create `tests/test_index_scale.rs` with `#[ignore]` tests:
       - `test_scale_1m_build` — build 1M entry index
       - `test_scale_1m_save_load_roundtrip` — save/load 1M entries
       - `test_scale_1m_mmap_load` — mmap load 1M entries
       - `test_scale_1m_find_by_name` — find entry in 1M
    2. All must complete without OOM
  - **Files**: `tests/test_index_scale.rs` (new)
  - **Done when**: Scale tests pass (run with `--ignored`)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_index_scale -- --ignored --nocapture 2>&1 | tail -20`
  - **Commit**: `test(index): add 1M entry scale tests`
  - _Requirements: PM Section 3 (scale to 1M+ files)_
  - _Design: QA.md Section 6_

- [x] 6.7 Write crash safety tests
  - **Do**:
    1. In `tests/test_index_persistence.rs` (or extend `tests/test_index.rs`):
       - `test_atomic_rename_no_partial_file` — interrupted write leaves no .idx
       - `test_tmp_left_on_error_no_corrupt` — .idx.tmp from failed write doesn't corrupt next load
       - `test_concurrent_save_one_valid` — two threads writing produce valid file
       - `test_rename_atomicity` — reader sees old or new, never partial
  - **Files**: `tests/test_index.rs` or new file
  - **Done when**: 4 crash safety tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test atomic_rename 2>&1 | tail -10 && cargo test concurrent_save 2>&1 | tail -10`
  - **Commit**: `test(index): add crash safety and atomic write tests`
  - _Requirements: QA.md Section 7_
  - _Design: QA.md_

- [x] 6.8 Write graceful degradation tests
  - **Do**:
    1. Tests:
       - `test_missing_cache_dir_created` — auto-creates cache dir
       - `test_unwritable_dir_fallback_to_walk` — save fails but search works
       - `test_corrupt_index_triggers_rebuild` — corrupt file detected, rebuild
       - `test_walk_fallback_without_index` — walk_and_filter still works
       - `test_blocking_search_without_index` — blocking search() API works
       - `test_index_deleted_mid_search` — search completes via walk
  - **Files**: `tests/test_index.rs` or new test file
  - **Done when**: 6 graceful degradation tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test graceful 2>&1 | tail -15`
  - **Commit**: `test(index): add graceful degradation and fallback tests`
  - _Requirements: UX.md Section 9 (Progressive Enhancement)_
  - _Design: QA.md Section 8_

- [x] 6.9 Write edge case tests (paths, permissions, symlinks)
  - **Do**:
    1. Path edge cases:
       - `test_path_exactly_224_bytes` — indexed
       - `test_path_225_bytes` — skipped
       - `test_path_1024_bytes` — skipped no panic
       - `test_unicode_cjk_path` — CJK characters
       - `test_space_in_filename` — space handled
       - `test_null_byte_path_rejected` — no panic
    2. Permission edge cases:
       - `test_chmod_000_dir_skipped` — unreadable dir skipped
       - `test_mixed_permissions` — only readable files indexed
    3. Symlink edge cases:
       - `test_symlink_to_file` — IS_SYMLINK flag
       - `test_broken_symlink_skipped` — silently skipped
  - **Files**: various test files
  - **Done when**: 10 edge case tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test edge_case 2>&1 | tail -20`
  - **Commit**: `test(index): add path, permission, and symlink edge case tests`
  - _Requirements: QA.md (Edge Cases)_
  - _Design: QA.md_

- [x] 6.10 Write regression tests (walk fallback, search equivalence)
  - **Do**:
    1. Walk fallback regressions:
       - `test_walk_fallback_still_works` — walk_and_filter returns results without index
       - `test_blocking_search_api_works` — blocking search() API unchanged
    2. Search equivalence:
       - `test_indexed_vs_unindexed_results_match` — same query, same files found
       - `test_filename_match_equivalence` — filename matches identical
  - **Files**: `tests/test_index.rs` or new test file
  - **Done when**: 4 regression tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test regression 2>&1 | tail -15`
  - **Commit**: `test(index): add regression tests for walk fallback and search equivalence`
  - _Requirements: QA.md Section 8 (Regression Tests)_
  - _Design: QA.md Section 8_

- [ ] 6.11 [VERIFY] Phase 6 checkpoint: full suite
  - **Do**: Run all tests including new ones
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -10 && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass, clippy clean
  - **Commit**: `test(index): complete testing and reliability suite`

## Phase 7: Integration and End-to-End Validation

Focus: End-to-end pipeline validation, performance verification, v1 migration, UX state transitions.

- [x] 7.1 Validate warm startup path
  - **Do**:
    1. Write integration test: create a v2 index file, launch app (simulate startup sequence)
    2. Measure: mmap file (<1ms) -> validate header -> create bytesNoCopy (<1ms) -> start FSEvents with stored sinceWhen -> search ready
    3. Assert total startup <100ms
  - **Files**: `tests/test_index.rs` or integration test
  - **Done when**: Warm startup validated under 100ms
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test warm_startup 2>&1 | tail -10`
  - **Commit**: `test(index): validate warm startup path under 100ms`
  - _Requirements: PM Section 3 (cold start <100ms)_
  - _Design: PM Section 7 (Data Flow)_

- [x] 7.2 Validate cold startup path
  - **Do**:
    1. Write integration test: no index file, spawn background scanner
    2. Verify search available immediately via walk_and_filter fallback
    3. Verify background build completes and saves v2 index
    4. Verify subsequent launches use v2 instantly
  - **Files**: integration test
  - **Done when**: Cold startup validated with walk fallback + background build
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test cold_startup 2>&1 | tail -10`
  - **Commit**: `test(index): validate cold startup with background build`
  - _Requirements: PM Section 3_
  - _Design: TECH.md Section 5_

- [x] 7.3 Validate v1-to-v2 migration
  - **Do**:
    1. Create a v1 index file (use old format code)
    2. Verify v1 detected and rebuild triggered
    3. Verify v2 global.idx produced
    4. Verify per-directory v1 files cleaned up (.v2-migrated marker)
    5. Verify subsequent launches use v2
  - **Files**: integration test
  - **Done when**: v1 migration validated end-to-end
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test v1_migration 2>&1 | tail -10`
  - **Commit**: `test(index): validate v1-to-v2 migration end-to-end`
  - _Requirements: TECH.md Section 10 (Migration)_
  - _Design: TECH.md Section 10_

- [ ] 7.4 [VERIFY] Quality checkpoint
  - **Do**: Clippy + all tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo clippy -- -D warnings 2>&1 | tail -10 && cargo test 2>&1 | tail -10`
  - **Done when**: Clean
  - **Commit**: `chore(index): pass integration quality checkpoint` (only if fixes needed)

- [x] 7.5 Validate live update cycle
  - **Do**:
    1. Integration test: load index -> create file on disk -> FSEvents event -> IndexWriter processes -> flush -> IndexStore swap -> search finds new file
    2. Measure total latency from fs::write() to searchable: must be <1s
  - **Files**: integration test
  - **Done when**: Live update cycle under 1s
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test live_update 2>&1 | tail -10`
  - **Commit**: `test(index): validate live update cycle under 1 second`
  - _Requirements: PM Section 3 (incremental update <1s)_
  - _Design: TECH.md Section 4_

- [x] 7.6 Validate concurrent search during operations
  - **Do**:
    1. Test search during initial build: walk_and_filter while build running, verify results
    2. Test search during incremental update: GPU search while IndexWriter flushing, verify old snapshot used
    3. No panics, no data races
  - **Files**: integration test
  - **Done when**: Concurrent search validated
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test concurrent_search 2>&1 | tail -10`
  - **Commit**: `test(index): validate concurrent search during build and update`
  - _Requirements: QA.md Section 4_
  - _Design: TECH.md Section 7_

- [x] 7.7 Validate error recovery states
  - **Do**:
    1. Stale index: save with old saved_at, verify stale detected, verify search works on stale data
    2. Corrupt index: overwrite first bytes, verify rebuild triggered, verify walk fallback
    3. Disk-full: read-only dir, verify save fails, verify search via walk
  - **Files**: integration test
  - **Done when**: All error recovery paths validated
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test error_recovery 2>&1 | tail -10`
  - **Commit**: `test(index): validate error recovery for stale, corrupt, and disk-full`
  - _Requirements: UX.md Section 7 (Error States)_
  - _Design: UX.md Section 7_

- [x] 7.8 Performance validation against success metrics
  - **Do**:
    1. Run comprehensive performance checks:
       - Warm start <100ms
       - Mmap load <5ms at 1M entries
       - bytesNoCopy <1ms
       - Save <1s at 1M
    2. Document results in test output
  - **Files**: integration test or benchmark
  - **Done when**: Performance targets met
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo bench --bench index_load 2>&1 | tail -20`
  - **Commit**: `test(index): validate performance against PM success metrics`
  - _Requirements: PM Section 3 (Success Metrics)_
  - _Design: PM Section 3_

- [x] 7.9 Validate backward compatibility
  - **Do**:
    1. Verify blocking search() API works unchanged
    2. Verify SearchUpdate streaming protocol works
    3. Verify GpuPathEntry layout matches Metal shader (compile-time assertion)
    4. Verify all existing tests pass
    5. Verify walk_and_filter fallback works without index
  - **Files**: integration test
  - **Done when**: All backward compat confirmed
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -10`
  - **Commit**: `test(index): validate backward compatibility with existing APIs`
  - _Requirements: QA.md Section 8_
  - _Design: QA.md Section 8_

- [ ] 7.10 [VERIFY] Full local CI
  - **Do**: Run complete local CI suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo clippy -- -D warnings 2>&1 | tail -5 && cargo test 2>&1 | tail -5 && cargo build --release 2>&1 | tail -5`
  - **Done when**: Clippy clean, all tests pass, release build succeeds
  - **Commit**: `chore(index): pass full local CI suite`

## Phase 8: Quality Gates

- [x] 8.1 Local quality check
  - **Do**: Run ALL quality checks locally
  - **Verify**: All commands must pass:
    - `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo clippy -- -D warnings`
    - `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test`
    - `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build --release`
  - **Done when**: All commands pass with no errors
  - **Commit**: `fix(index): address lint/type issues` (if fixes needed)

- [x] 8.2 Create PR and verify CI
  - **Do**:
    1. Verify current branch is a feature branch: `git branch --show-current`
    2. If on default branch, STOP and alert user
    3. Push branch: `git push -u origin <branch-name>`
    4. Create PR: `gh pr create --title "feat(index): persistent GPU-friendly file index with FSEvents" --body "..."`
    5. Wait for CI
  - **Verify**: `gh pr checks --watch` — all checks must show passing
  - **Done when**: All CI checks green, PR ready for review
  - **If CI fails**: Read failure details with `gh pr checks`, fix, push, re-verify

## Phase 9: PR Lifecycle

- [x] 9.1 Monitor CI and fix failures
  - **Do**:
    1. Check CI status: `gh pr checks`
    2. If any failures: read logs, fix locally, commit, push
    3. Repeat until all green
  - **Verify**: `gh pr checks` shows all passing
  - **Done when**: CI fully green
  - **Commit**: `fix(index): address CI failures` (if needed)

- [x] 9.2 Address review comments
  - **Do**:
    1. Check for review comments: `gh api repos/{owner}/{repo}/pulls/{pr}/comments`
    2. Address each comment with code changes
    3. Push fixes
  - **Verify**: No unresolved review comments
  - **Done when**: All review feedback addressed
  - **Commit**: `fix(index): address review feedback`

- [ ] 9.3 [VERIFY] Final verification
  - **Do**:
    1. Run full test suite one more time
    2. Verify CI green
    3. Verify all acceptance criteria from PM.md:
       - Warm startup <100ms
       - Incremental update <1s
       - v1 migration works
       - Corrupt/stale recovery works
       - All existing tests pass
       - Clippy clean
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -5 && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: All acceptance criteria confirmed
  - **Commit**: None

- [ ] 9.4 [VERIFY] AC checklist
  - **Do**: Programmatically verify each acceptance criterion:
    1. AC-1: GsixHeaderV2 is 16384 bytes (compile-time assertion in code)
    2. AC-2: IS_DELETED flag defined in Rust and Metal
    3. AC-3: Save/load roundtrip byte-identical (unit tests pass)
    4. AC-4: v1 detected and triggers rebuild (unit test)
    5. AC-5: CRC32 catches corruption (unit test)
    6. AC-6: Corrupt inputs produce Err, not panics (proptest)
    7. AC-7: bytesNoCopy succeeds on Apple Silicon
    8. AC-8: IndexStore lock-free (arc-swap)
    9. AC-9: FSEvents watches / with no FD limit
    10. AC-10: Incremental update <1s
    11. AC-11: Walk fallback works without index
    12. AC-12: Status bar shows correct states
    13. AC-13: All existing tests pass
    14. AC-14: Clippy clean
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | grep -E "test result|FAILED" && cargo clippy -- -D warnings 2>&1 | tail -3`
  - **Done when**: All acceptance criteria confirmed via automated checks
  - **Commit**: None

## Notes

- **POC shortcuts taken (Phase 1)**:
  - No tests for Metal bytesNoCopy in Phase 1 (deferred to Phase 2)
  - v1 migration does full rebuild, no data migration
  - ExcludeTrie uses simple string matching, not compiled regex

- **Production TODOs**:
  - Benchmark mmap load at 1M entries
  - Test FSEvents event flood handling
  - Validate event ID regression detection
  - Memory pressure testing for 3+ concurrent mmap regions

- **Key dependencies between phases**:
  - Phase 2 requires Phase 1 (v2 header for page alignment)
  - Phase 3 requires Phase 1 (last_fsevents_id, exclude_hash fields)
  - Phase 4 requires Phases 1-3 (IS_DELETED, IndexStore, FsChange events)
  - Phase 5 requires Phases 1-4 (all subsystems)
  - Phase 6-7 require Phase 5 (full system assembled)

- **New Cargo dependencies**:
  - `crc32fast = "1"` (Phase 1)
  - `arc-swap = "1"` (Phase 2)
  - `fsevent-sys = "4"` (Phase 3)
  - `core-foundation = "0.10"` (Phase 3)

- **New source files**:
  - `src/index/gsix_v2.rs` — v2 header, save/load, version detection
  - `src/index/metal_buffer.rs` or `src/index/snapshot.rs` — IndexSnapshot, bytesNoCopy
  - `src/index/store.rs` — IndexStore (arc-swap)
  - `src/index/fsevents.rs` — FSEventsListener, FsChange
  - `src/index/exclude.rs` — ExcludeTrie
  - `src/index/index_writer.rs` — IndexWriter with CRUD
  - `src/index/daemon.rs` — IndexDaemon lifecycle coordinator
  - `src/index/global.rs` — global index path helpers (optional, could be in gsix_v2.rs)
