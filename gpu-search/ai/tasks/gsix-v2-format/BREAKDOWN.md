---
id: gsix-v2-format.BREAKDOWN
module: gsix-v2-format
priority: 0
status: pending
version: 1
origin: foreman-spec
dependsOn: []
tags: [persistent-index, gpu-search, binary-format]
---
# GSIX v2 Format — BREAKDOWN

## Context
The existing GSIX v1 format uses a 64-byte header with entries starting immediately after. This prevents page-aligned `bytesNoCopy` Metal buffer creation and lacks fields for FSEvents event ID tracking, exclude configuration hashing, and entry capacity management. This module defines the GSIX v2 binary format with a 16KB page-aligned header, adds the `IS_DELETED` tombstone flag to `GpuPathEntry`, implements version detection, and handles v1-to-v2 migration via full rebuild.

References: TECH.md Section 2 (GSIX v2 Format), PM.md Section 4 (GSIX v2 binary format), QA.md Section 2 (Index Format Tests).

## Tasks
1. **gsix-v2-format.header-struct** -- Define `GsixHeaderV2` as a `#[repr(C)]` struct: magic (u32), version (u32=2), entry_count (u32), root_hash (u32), saved_at (u64), last_fsevents_id (u64), exclude_hash (u32), entry_capacity (u32), flags (u32), checksum (u32), _reserved ([u8; 16336]). Add compile-time assertion: `size_of::<GsixHeaderV2>() == 16384`. Define `HEADER_SIZE_V2 = 16384` constant. (est: 1h)

2. **gsix-v2-format.header-serialization** -- Implement `GsixHeaderV2::to_bytes()` and `GsixHeaderV2::from_bytes()` for reading/writing the 16KB header. Include CRC32 checksum computation over bytes [0..44) using `crc32fast`. Validate magic, version, and checksum on load. (est: 2h)

3. **gsix-v2-format.is-deleted-flag** -- Add `IS_DELETED: u32 = 1 << 4` to the `path_flags` module in `src/gpu/types.rs`. Update the Metal shader `search_types.h` to define `PATH_FLAG_IS_DELETED = (1 << 4)` and add the `!(flags & IS_DELETED)` check to the `path_filter` kernel. (est: 1h)

4. **gsix-v2-format.flags-field** -- Define bitflag constants for the header `flags` field: `FLAG_SORTED = 0x1`, `FLAG_COMPACTED = 0x2`. Implement helper methods `is_sorted()`, `is_compacted()`, `set_sorted()`, `set_compacted()` on the header. (est: 1h)

5. **gsix-v2-format.save-v2** -- Update `SharedIndexManager::save()` (or create a new `IndexPersister::save_v2()`) to write the 16KB v2 header followed by entry data. Maintain atomic write pattern (write to `.idx.tmp`, `fsync()`, then `rename()`). Compute and store `exclude_hash` and `checksum`. (est: 2h)

6. **gsix-v2-format.load-v2** -- Update `SharedIndexManager::load()` (or create `IndexPersister::load_v2()`) to read and validate the 16KB v2 header. Parse all new fields (`last_fsevents_id`, `exclude_hash`, `entry_capacity`, `flags`). Verify checksum. Return entries starting at offset 16384. (est: 2h)

7. **gsix-v2-format.version-detection** -- Implement `detect_version(data: &[u8]) -> Result<u32, CacheError>` that reads magic and version fields. Return `Ok(1)` for v1, `Ok(2)` for v2, or `Err(InvalidFormat)` for unknown magic/version. (est: 1h)

8. **gsix-v2-format.v1-migration** -- When `detect_version()` returns v1, trigger a full rebuild: log a warning message, delete the old v1 file, return an error that signals the caller to perform a fresh scan. No data migration needed since `GpuPathEntry` layout is unchanged. (est: 1h)

9. **gsix-v2-format.v1-cleanup** -- Implement `cleanup_v1_indexes()` that deletes old per-directory v1 `.idx` files from `~/.gpu-search/index/` (any `.idx` file whose stem is not `global`). Create a `.v2-migrated` marker file to avoid re-running cleanup. (est: 1h)

10. **gsix-v2-format.unit-tests** -- Write unit tests: header magic bytes correctness, version field = 2, entry_count matches written entries, root_hash matches `cache_key("/")`, saved_at is recent, checksum validates, reject unknown magic, reject future version, reject truncated file, v1 backward compat detection. (~10 tests) (est: 2h)

11. **gsix-v2-format.corrupt-handling-tests** -- Write tests for corrupt file handling: entry_count overflow, zero-byte file, header-only with zero entries, partial entry, random bytes (no panic), wrong endianness, trailing garbage, all-zero header. (~8 tests) (est: 2h)

## Dependencies
- Requires: Nothing (this is the foundation module)
- Enables: [mmap-gpu-pipeline (needs v2 header for page alignment), incremental-updates (needs IS_DELETED flag and entry_capacity), fsevents-watcher (needs last_fsevents_id field), global-root-index (needs exclude_hash)]

## Acceptance Criteria
1. `GsixHeaderV2` struct is exactly 16384 bytes (compile-time assertion passes)
2. `IS_DELETED` flag (bit 4) is defined in both Rust `path_flags` and Metal `search_types.h`
3. Save/load roundtrip produces byte-identical entries for all 256 bytes of each entry
4. v1 index files are detected and trigger a rebuild signal (not a panic or silent data corruption)
5. CRC32 checksum detects single-bit corruption in header fields
6. All corrupt file inputs produce `Err(CacheError)`, never panics or undefined behavior
7. Old per-directory v1 index files are cleaned up after successful v2 migration

## Technical References
- PM: ai/tasks/spec/PM.md -- Section 4 (Scope: GSIX v2 binary format), Section 7 (GSIX v2 Header Layout)
- UX: ai/tasks/spec/UX.md -- Section 7.2.1 (Index Corrupt error handling)
- Tech: ai/tasks/spec/TECH.md -- Section 2 (GSIX v2 Format: header layout, entry region, backward compatibility), Section 10 (Migration)
- QA: ai/tasks/spec/QA.md -- Section 2 (Index Format Tests: header validation, entry alignment, corrupt handling)
