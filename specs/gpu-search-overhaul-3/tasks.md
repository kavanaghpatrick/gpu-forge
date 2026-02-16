---
spec: gpu-search-overhaul-3
phase: tasks
total_tasks: 15
created: 2026-02-16
generated: auto
---

# Tasks: gpu-search-overhaul-3

## Phase 1: Make It Work (POC)

Focus: Get all three fixes working end-to-end. Accept shortcuts, skip tests.

- [x] 1.1 Remove 100K chunk cap and add dynamic metadata buffer resizing
  - **Do**:
    1. In `ContentSearchEngine::new()` (content.rs:177): change `let max_chunks = (max_files * 10).min(100_000)` to `let max_chunks = max_files * 10` (remove the `.min(100_000)` cap)
    2. Add method `ensure_metadata_capacity(&mut self, needed: usize)` that reallocates `self.metadata_buffer` if `needed > self.max_chunks`, using `self.device.newBufferWithLength_options(needed * mem::size_of::<ChunkMetadata>(), options)`. Update `self.max_chunks = needed` after successful realloc
    3. In `search_zerocopy()` (content.rs:560-682): at the top of the method (after empty checks), call `self.ensure_metadata_capacity(chunk_metas.len())` to ensure the metadata buffer can hold all chunks
  - **Files**: `gpu-search/src/search/content.rs`
  - **Done when**: `ensure_metadata_capacity` exists, `.min(100_000)` is removed, `search_zerocopy()` calls it
  - **Verify**: `cargo build -p gpu-search`
  - **Commit**: `perf(content): remove 100K chunk cap, add dynamic metadata buffer`
  - _Requirements: FR-1, FR-2_
  - _Design: Component A_

- [ ] 1.2 Eliminate serial batch loop in search_zerocopy
  - **Do**:
    1. In `search_zerocopy()` (content.rs:587-682): replace the batch loop `for batch_start in (0..chunk_metas.len()).step_by(batch_size)` with a single-pass dispatch:
       - Write ALL chunk_metas to metadata_buffer at once (no batching)
       - Write params with full chunk_count and total_bytes = chunk_count * CHUNK_SIZE
       - Reset match_count to 0
       - Single `cmd.commit()` + `waitUntilCompleted()`
       - Single `collect_results()` call
    2. Remove the batch-local byte_offset translation loop (lines 665-671) -- not needed when all chunks are dispatched at once, since chunk_index is already global
    3. Keep the `all_results` Vec and truncation logic for max_results
  - **Files**: `gpu-search/src/search/content.rs`
  - **Done when**: `search_zerocopy()` has zero `step_by` loops, single dispatch path
  - **Verify**: `cargo build -p gpu-search`
  - **Commit**: `perf(content): single-dispatch zerocopy search, remove batch loop`
  - _Requirements: FR-3_
  - _Design: Component A_

- [ ] 1.3 Add file-relative line number computation to zerocopy kernel
  - **Do**:
    1. In `content_search_zerocopy_kernel` (content_search.metal:90-97): replace the local-window newline scan with a scan from file start:
       ```metal
       // Replace existing line_num computation (lines 90-97) with:
       uint file_line = 1;
       ulong match_abs = meta.buffer_offset + offset_in_chunk + local_pos;
       // Scan from file start to match position
       for (ulong s = meta.buffer_offset; s < match_abs; s++) {
           if (raw_data[s] == 0x0A) { file_line++; }
       }
       result.line_number = file_line;
       ```
    2. Also update `line_start` computation to find the start of the current line (last newline before match position) for correct column calculation
    3. Leave `content_search_kernel` and `turbo_search_kernel` unchanged (they use the padded path)
  - **Files**: `gpu-search/shaders/content_search.metal`
  - **Done when**: zerocopy kernel scans from `meta.buffer_offset` for line numbers
  - **Verify**: `cargo build -p gpu-search` (Metal shader compilation)
  - **Commit**: `perf(metal): compute file-relative line numbers in zerocopy kernel`
  - _Requirements: FR-4_
  - _Design: Component B_

- [ ] 1.4 Remove CPU resolve loop, use GPU line numbers directly
  - **Do**:
    1. In `search_from_content_store()` (orchestrator.rs:1491-1581): replace the `String::from_utf8_lossy` + `lines().collect()` loop with a new resolve function that:
       - Takes `file_content: &[u8]`, `byte_offset: usize`, `gpu_line_number: u32`, `pattern: &str`
       - Uses `byte_offset` to find line boundaries: scan backward from byte_offset for `\n` to find line_start, scan forward for `\n` to find line_end
       - Extract `line_content` as `&file_content[line_start..line_end]`
       - Extract 2 context lines before and 2 after by scanning for additional `\n` boundaries
       - Use `gpu_line_number` directly as the line number (trust GPU)
       - Find pattern match within the extracted line for `match_range`
    2. Add helper function `fn extract_line_context(content: &[u8], byte_offset: usize, pattern: &str, case_sensitive: bool) -> Option<(String, Vec<String>, Vec<String>, usize)>` that returns (line_content, context_before, context_after, match_col)
    3. Update the loop to use `m.line_number` from GPU match result (content.rs ContentMatch already has this field)
  - **Files**: `gpu-search/src/search/orchestrator.rs`
  - **Done when**: No `String::from_utf8_lossy` or `lines().collect()` in resolve loop, GPU line numbers used directly
  - **Verify**: `cargo build -p gpu-search`
  - **Commit**: `perf(orchestrator): replace CPU line resolve with GPU line numbers`
  - _Requirements: FR-5, FR-6_
  - _Design: Component B_

- [ ] 1.5 Add chunk_metadata field to ContentStore
  - **Do**:
    1. Add field `chunk_metadata: Option<Vec<ChunkMetadata>>` to `ContentStore` struct (content_store.rs:173)
    2. Add `use crate::search::content::ChunkMetadata;` import (already exists at bottom of file for build_chunk_metadata)
    3. Initialize to `None` in all constructors: `new()`, `with_capacity()`, `from_gcix_mmap()`
    4. Add getter: `pub fn chunk_metadata(&self) -> Option<&[ChunkMetadata]>`
    5. Add setter: `pub fn set_chunk_metadata(&mut self, chunks: Vec<ChunkMetadata>)`
  - **Files**: `gpu-search/src/index/content_store.rs`
  - **Done when**: ContentStore has chunk_metadata field with getter/setter
  - **Verify**: `cargo build -p gpu-search`
  - **Commit**: `feat(content-store): add chunk_metadata field for pre-built metadata`
  - _Requirements: FR-7_
  - _Design: Component C_

- [ ] 1.6 Persist chunk metadata in GCIX v3 format
  - **Do**:
    1. In gcix.rs: bump `GCIX_VERSION` to 3
    2. Add to `GcixHeader`: `chunks_offset: u64` and `chunks_bytes: u64` fields (before `header_crc32`)
    3. Update `to_bytes()` and `from_bytes()` to serialize/deserialize the new fields
    4. Update `CRC32_OFFSET` constant (now +16 bytes = 92)
    5. Update header size assertion and `_padding` array size
    6. In `save_gcix()`: after writing path table, call `build_chunk_metadata()` on the store, write the resulting `Vec<ChunkMetadata>` as raw bytes, record offset/bytes in header
    7. In `load_gcix()`: if version >= 3 and chunks_bytes > 0, read chunk metadata from the mmap at chunks_offset, parse as `&[ChunkMetadata]` slice, call `store.set_chunk_metadata(chunks.to_vec())`
    8. If version == 2: skip chunk metadata loading (will be rebuilt on first search)
  - **Files**: `gpu-search/src/index/gcix.rs`, `gpu-search/src/index/content_store.rs`
  - **Done when**: GCIX v3 saves and loads chunk metadata, v2 files still load without crash
  - **Verify**: `cargo build -p gpu-search`
  - **Commit**: `feat(gcix): v3 format with pre-built chunk metadata table`
  - _Requirements: FR-7, FR-8, FR-9_
  - _Design: Component C_

- [ ] 1.7 Use pre-built chunk metadata in orchestrator
  - **Do**:
    1. In `search_from_content_store()` (orchestrator.rs:1396-1398): replace `let chunk_metas = build_chunk_metadata(content_store)` with:
       ```rust
       let chunk_metas = match content_store.chunk_metadata() {
           Some(cached) => cached.to_vec(),
           None => build_chunk_metadata(content_store),
       };
       ```
    2. Log whether cached or rebuilt metadata was used (for debugging)
  - **Files**: `gpu-search/src/search/orchestrator.rs`
  - **Done when**: Orchestrator uses pre-built metadata when available, falls back to rebuild
  - **Verify**: `cargo build -p gpu-search`
  - **Commit**: `perf(orchestrator): use pre-built chunk metadata from content store`
  - _Requirements: FR-8_
  - _Design: Component C_

- [ ] 1.8 POC Checkpoint: compile and run unit tests
  - **Do**: Run full test suite to verify all changes compile and existing tests pass
  - **Done when**: All `cargo test -p gpu-search` tests pass
  - **Verify**: `cargo test -p gpu-search 2>&1 | tail -20`
  - **Commit**: `fix(gpu-search): address compilation issues from overhaul` (if needed)

## Phase 2: Refactoring

After POC validated, clean up code.

- [ ] 2.1 Clean up search_zerocopy and remove dead batch code
  - **Do**:
    1. Remove any remaining batch-related variables (`batch_size`, `batch_start`, `batch_end`) from `search_zerocopy()`
    2. Remove the old byte_offset translation loop that was batch-specific
    3. Simplify the method to a clean single-dispatch flow: write metadata -> write params -> dispatch -> collect
    4. Update doc comments to reflect single-dispatch behavior
    5. Remove `batch_size` variable usage: `let batch_size = self.max_chunks;`
  - **Files**: `gpu-search/src/search/content.rs`
  - **Done when**: `search_zerocopy()` is clean single-dispatch, no batch remnants
  - **Verify**: `cargo clippy -p gpu-search -- -D warnings`
  - **Commit**: `refactor(content): clean up search_zerocopy single-dispatch path`
  - _Design: Component A_

- [ ] 2.2 Add error handling for metadata buffer reallocation
  - **Do**:
    1. In `ensure_metadata_capacity()`: handle `None` from `newBufferWithLength_options` gracefully
    2. Return `Result<(), &'static str>` or log warning and keep old buffer
    3. In `search_zerocopy()`: if capacity insufficient and realloc fails, fall back to batch mode (keep old batch code behind a feature flag or conditional)
  - **Files**: `gpu-search/src/search/content.rs`
  - **Done when**: Buffer reallocation failure doesn't panic
  - **Verify**: `cargo clippy -p gpu-search -- -D warnings`
  - **Commit**: `refactor(content): add error handling for metadata buffer realloc`
  - _Design: Component A, Error Handling_

- [ ] 2.3 Extract line context helper into utility function
  - **Do**:
    1. Move the `extract_line_context()` helper from orchestrator.rs into a separate module or as an associated function
    2. Make it reusable by both `search_from_content_store()` and `search_streaming_inner()`
    3. Add proper doc comments explaining byte-offset-based line extraction
  - **Files**: `gpu-search/src/search/orchestrator.rs`
  - **Done when**: Line context extraction is a clean reusable function
  - **Verify**: `cargo clippy -p gpu-search -- -D warnings`
  - **Commit**: `refactor(orchestrator): extract line context helper`
  - _Design: Component B_

## Phase 3: Testing

- [ ] 3.1 Unit tests for single-dispatch search
  - **Do**:
    1. Add test in content.rs that creates >100K chunks and verifies single-dispatch finds all matches
    2. Test `ensure_metadata_capacity()` with various chunk counts
    3. Test that match results from single dispatch match CPU reference
  - **Files**: `gpu-search/src/search/content.rs`
  - **Done when**: Tests cover >100K chunk dispatch, metadata resize, result correctness
  - **Verify**: `cargo test -p gpu-search -- test_single_dispatch`
  - **Commit**: `test(content): add single-dispatch search tests`
  - _Requirements: AC-1.1, AC-1.2_

- [ ] 3.2 Unit tests for GPU line number accuracy
  - **Do**:
    1. Add test that loads multi-line files into ContentStore, searches via zerocopy kernel, verifies GPU line numbers match CPU newline count
    2. Test edge cases: match on first line, match on last line, match after empty lines, CRLF line endings
    3. Compare GPU line_number against `content.iter().take(byte_offset).filter(|&&b| b == b'\n').count() + 1`
  - **Files**: `gpu-search/src/search/content.rs`
  - **Done when**: GPU line numbers verified against CPU reference for all edge cases
  - **Verify**: `cargo test -p gpu-search -- test_gpu_line_number`
  - **Commit**: `test(content): add GPU line number accuracy tests`
  - _Requirements: AC-2.1, AC-2.3_

- [ ] 3.3 Unit tests for GCIX v3 chunk metadata persistence
  - **Do**:
    1. Add test in gcix.rs that saves a ContentStore to GCIX v3, reloads it, verifies chunk_metadata() returns the same data
    2. Test v2 backward compat: create a v2 GCIX file, load it, verify chunk_metadata() returns None
    3. Test round-trip: build_chunk_metadata -> save GCIX -> load GCIX -> compare
  - **Files**: `gpu-search/src/index/gcix.rs`
  - **Done when**: GCIX v3 chunk metadata round-trips correctly, v2 fallback works
  - **Verify**: `cargo test -p gpu-search -- test_gcix_v3`
  - **Commit**: `test(gcix): add v3 chunk metadata persistence tests`
  - _Requirements: AC-3.1, AC-3.2, AC-3.4_

## Phase 4: Quality Gates

- [ ] 4.1 Full quality check (clippy, test, build)
  - **Do**: Run all quality gates locally
  - **Verify**: `cargo clippy -p gpu-search -- -D warnings && cargo test -p gpu-search && cargo build -p gpu-search --release`
  - **Done when**: All three commands pass with zero warnings/errors
  - **Commit**: `fix(gpu-search): address lint/type issues` (if needed)

- [ ] 4.2 Create PR and verify CI
  - **Do**: Push branch, create PR with gh CLI, watch CI
  - **Verify**: `gh pr checks --watch` all green
  - **Done when**: PR created, CI green, ready for review

## Notes

- **POC shortcuts taken**: Error handling for metadata buffer realloc deferred to Phase 2. Context line extraction may be naive (byte scan) vs optimized (precomputed newline index).
- **Production TODOs**: Consider precomputing a newline offset table per file for O(log n) line lookups. Consider MAX_MATCHES increase from 10000 for very large codebases.
- **Key risk**: GPU kernel scanning from file start for every match may be slow for matches deep in 10MB+ files. Profile and consider threadgroup-level prefix sum optimization if needed.
