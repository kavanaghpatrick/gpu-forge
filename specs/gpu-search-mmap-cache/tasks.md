# Tasks: GPU Search mmap Chunk Cache

## Phase 1: Make It Work (POC)

Focus: Get cache save/load/mmap working end-to-end with unit tests. Skip app integration.

- [x] 1.1 Add libc dependency and chunk_cache module scaffold
  - **Do**:
    1. Add `libc = "0.2"` to `[dependencies]` in `/Users/patrickkavanagh/gpu-search-ui/Cargo.toml`
    2. Add `pub mod chunk_cache;` to `/Users/patrickkavanagh/gpu-search-ui/src/engine/mod.rs` (after existing `pub mod shader;`)
    3. Create `/Users/patrickkavanagh/gpu-search-ui/src/engine/chunk_cache.rs` with constants, `MmapChunkData` struct (with `Send`/`Sync` impls, `chunk_ptr()`, `chunk_data_len()`, `chunk_data_slice()`, `Drop` calling `munmap`), and stub `save_chunk_cache()` / `load_chunk_cache()` functions returning `None`
    4. Make `PathCache::default_cache_dir()` visibility `pub(crate)` in `/Users/patrickkavanagh/gpu-search-ui/src/engine/index.rs` (change `fn default_cache_dir()` to `pub(crate) fn default_cache_dir()`)
  - **Files**:
    - `/Users/patrickkavanagh/gpu-search-ui/Cargo.toml` (modify)
    - `/Users/patrickkavanagh/gpu-search-ui/src/engine/mod.rs` (modify)
    - `/Users/patrickkavanagh/gpu-search-ui/src/engine/chunk_cache.rs` (create)
    - `/Users/patrickkavanagh/gpu-search-ui/src/engine/index.rs` (modify)
  - **Done when**: `cargo check` passes with new module and libc dep
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo check 2>&1 | tail -5`
  - **Commit**: `feat(engine): add chunk_cache module scaffold with MmapChunkData struct`
  - _Requirements: FR-1, FR-3, FR-4, NFR-3_
  - _Design: Components 1-2_

- [x] 1.2 Implement save_chunk_cache()
  - **Do**:
    1. In `chunk_cache.rs`, implement `save_chunk_cache(root, chunk_data, chunk_count, path_count) -> Option<()>`
    2. Build 64-byte header: magic "GPSC" (4B), version u32 LE (4B), timestamp u64 LE (8B), chunk_count u32 LE (4B), path_count u32 LE (4B), chunk_data_len u64 LE (8B), root_hash 32B (first 32 bytes of root path, zero-padded)
    3. Write header + raw chunk_data bytes + zero-pad to 16KB boundary
    4. Use atomic temp-file + rename pattern (call `PathCache::default_cache_dir()` for path)
    5. Log via `eprintln!("[ChunkCache] saved ...")`
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/chunk_cache.rs` (modify)
  - **Done when**: `save_chunk_cache` writes valid cache file with correct header + padded data
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo check 2>&1 | tail -5`
  - **Commit**: `feat(chunk-cache): implement save_chunk_cache with atomic write`
  - _Requirements: FR-1, FR-3_
  - _Design: Component 3_

- [x] 1.3 Implement load_chunk_cache()
  - **Do**:
    1. In `chunk_cache.rs`, implement `load_chunk_cache(root: &Path) -> Option<MmapChunkData>`
    2. Open cache file, read 64-byte header, validate: magic, version, timestamp freshness (<1 hour), root path prefix match (32 bytes), file_len sanity
    3. `mmap(fd, file_len, PROT_READ|PROT_WRITE, MAP_PRIVATE)` — MAP_PRIVATE for copy-on-write safety
    4. Check `MAP_FAILED`, construct `MmapChunkData` with mmap_ptr, mmap_len, chunk_count, path_count
    5. Log via `eprintln!("[ChunkCache] loaded ...")`
    6. Return `Some(MmapChunkData)` on success, `None` on any failure
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/chunk_cache.rs` (modify)
  - **Done when**: `load_chunk_cache` opens, validates, mmaps, and returns `MmapChunkData`
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo check 2>&1 | tail -5`
  - **Commit**: `feat(chunk-cache): implement load_chunk_cache with mmap and validation`
  - _Requirements: FR-2, FR-4, FR-6, FR-8, NFR-5_
  - _Design: Component 4_

- [ ] 1.4 Add unit tests for cache round-trip and error cases
  - **Do**:
    1. Add `#[cfg(test)] mod tests` in `chunk_cache.rs`
    2. `test_save_load_round_trip`: Create fake chunk_data (100 chunks of 4096 bytes with path strings), save, load, verify `chunk_count`, `path_count`, `chunk_data_slice()` matches original
    3. `test_invalid_magic`: Write garbage to cache file, verify `load_chunk_cache` returns `None`
    4. `test_stale_cache`: Build valid header with timestamp 2 hours in past, verify rejection
    5. `test_wrong_root`: Save with root "/projects/alpha", load with root "/projects/beta", verify `None`
    6. `test_truncated_file`: Write only partial header (32 bytes), verify `None`
    7. `test_empty_chunks`: Save 0 chunks (header only + padding), verify load succeeds with chunk_count=0
    8. `test_page_alignment`: Save various chunk counts, verify file size is multiple of 16384
    9. Use `tempfile::tempdir()` for all tests. Add helper `save_to_dir`/`load_from_dir` test variants (or make save/load accept custom cache dir) to avoid polluting real cache
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/chunk_cache.rs` (modify)
  - **Done when**: All 8 unit tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test chunk_cache -- --nocapture 2>&1 | tail -20`
  - **Commit**: `test(chunk-cache): add unit tests for round-trip, corruption, staleness`
  - _Requirements: AC-2.1, AC-2.4, NFR-5_
  - _Design: Test Strategy_

- [ ] 1.5 [VERIFY] Quality checkpoint: cargo check && cargo clippy && cargo test
  - **Do**: Run quality commands, fix any warnings or errors
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo check 2>&1 | tail -3 && cargo clippy 2>&1 | tail -5 && cargo test 2>&1 | tail -10`
  - **Done when**: Zero errors, zero clippy warnings, all tests pass
  - **Commit**: `chore(chunk-cache): pass quality checkpoint` (only if fixes needed)

## Phase 2: Integration

Focus: Wire chunk_cache into app.rs index_thread and search_thread. Add load_from_cache to search engine.

- [ ] 2.1 Add load_from_cache() method to GpuContentSearch
  - **Do**:
    1. In `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`, add `pub fn load_from_cache(&mut self, mmap: &MmapChunkData) -> usize`
    2. Import `super::chunk_cache::MmapChunkData`
    3. Implementation: clear `file_paths`, push empty string (virtual file), set `current_chunk_count`, `total_bytes`
    4. Create GPU buffer via `self.device.new_buffer_with_bytes_no_copy(mmap.chunk_ptr(), chunk_len as u64, MTLResourceOptions::StorageModeShared, None)`
    5. Replace `self.chunks_buffer` with the new buffer
    6. Copy mmap slice to `self.chunk_data` Vec (CPU-side access for `search_paths()`)
    7. Rebuild uniform `ChunkMetadata` entries in `metadata_buffer`
    8. Return `chunk_count`
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (modify)
  - **Done when**: `cargo check` passes, method signature matches design
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo check 2>&1 | tail -5`
  - **Commit**: `feat(search): add load_from_cache() for mmap-backed GPU buffer`
  - _Requirements: FR-5, FR-9, FR-10, AC-3.1, AC-3.2_
  - _Design: Component 5_

- [ ] 2.2 Add PathBatch enum to bridge.rs and update channel types
  - **Do**:
    1. In `/Users/patrickkavanagh/gpu-search-ui/src/bridge.rs`, add enum: `pub enum PathBatch { Paths(Vec<PathBuf>), CachedChunks(MmapChunkData) }` (import MmapChunkData from engine::chunk_cache)
    2. Change `batch_tx`/`batch_rx` type from `Sender<Vec<PathBuf>>` / `Receiver<Vec<PathBuf>>` to `Sender<PathBatch>` / `Receiver<PathBatch>`
    3. Update `Bridge::new()` channel creation
    4. Update existing tests in bridge.rs to use `PathBatch::Paths(...)` wrapper
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/bridge.rs` (modify)
  - **Done when**: `cargo check` passes with new enum, existing bridge tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo check 2>&1 | tail -5 && cargo test bridge -- --nocapture 2>&1 | tail -10`
  - **Commit**: `feat(bridge): add PathBatch enum for mmap cache channel transport`
  - _Requirements: FR-2_
  - _Design: Components 6-7, Technical Decisions (Channel type)_

- [ ] 2.3 Integrate chunk cache into index_thread and search_thread
  - **Do**:
    1. In `/Users/patrickkavanagh/gpu-search-ui/src/app.rs`, modify `index_thread()`:
       - Before PathCache check, try `chunk_cache::load_chunk_cache(&root)`
       - If Some(mmap_data): send `PathBatch::CachedChunks(mmap_data)` via batch_tx, send `IndexUpdate::Complete`, return early
       - Wrap existing PathCache/walk `batch_tx.send(paths)` calls with `PathBatch::Paths(...)`
    2. Modify `search_thread()`:
       - Add `let mut mmap_holder: Option<MmapChunkData> = None;` to hold mmap alive
       - Change `batch_rx.try_recv()` handling to match on `PathBatch`:
         - `PathBatch::Paths(paths)`: existing behavior (extend all_paths)
         - `PathBatch::CachedChunks(mmap)`: create engine via `new_for_paths`, call `engine.load_from_cache(&mmap)`, set `paths_loaded = true`, store mmap in `mmap_holder`
       - After `load_paths()` succeeds (cold path), call `chunk_cache::save_chunk_cache(&root, &engine.chunk_data, engine.chunk_count(), path_count)`
       - Need to pass `root` into search_thread — add as parameter or capture
    3. Update imports in app.rs for chunk_cache, PathBatch
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/app.rs` (modify)
  - **Done when**: `cargo check` passes, warm start path compiles
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo check 2>&1 | tail -5`
  - **Commit**: `feat(app): integrate chunk cache into index and search threads`
  - _Requirements: FR-2, FR-5, FR-8, AC-1.1, AC-1.4, AC-2.3_
  - _Design: Components 6-7, Data Flow_

- [ ] 2.4 [VERIFY] Quality checkpoint: cargo check && cargo clippy && cargo test
  - **Do**: Run full quality suite, fix any issues introduced during integration
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo check 2>&1 | tail -3 && cargo clippy 2>&1 | tail -5 && cargo test 2>&1 | tail -10`
  - **Done when**: Zero errors, all existing tests pass, no clippy warnings
  - **Commit**: `chore(app): pass quality checkpoint after integration` (only if fixes needed)

- [ ] 2.5 POC Checkpoint: Build release and verify cold start saves cache
  - **Do**:
    1. Build release: `cargo build --release`
    2. Remove existing cache: `rm -f ~/.cache/gpu-search-ui/chunks.cache`
    3. Run app briefly (cold start): `timeout 10 ./target/release/gpu-search-ui ~/Documents 2>&1` (or similar short path)
    4. Verify cache file was created: `ls -la ~/.cache/gpu-search-ui/chunks.cache`
    5. Verify file starts with "GPSC" magic: `xxd -l 64 ~/.cache/gpu-search-ui/chunks.cache`
    6. Verify file size is 16KB-aligned: `stat -f %z ~/.cache/gpu-search-ui/chunks.cache` (check divisible by 16384)
    7. Re-run app (warm start): `timeout 5 ./target/release/gpu-search-ui ~/Documents 2>&1` — check stderr for "[ChunkCache] loaded" message
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build --release 2>&1 | tail -3 && ls -la ~/.cache/gpu-search-ui/chunks.cache 2>&1`
  - **Done when**: Cold start creates cache file, warm start loads it with "[ChunkCache] loaded" in stderr
  - **Commit**: `feat(chunk-cache): complete POC — cold save + warm load verified`
  - _Requirements: AC-1.1, AC-1.2, AC-1.3_

## Phase 3: Testing

Focus: Add integration tests verifying search correctness from cached data, and error paths.

- [ ] 3.1 Integration test: cache round-trip produces identical search results
  - **Do**:
    1. In chunk_cache.rs tests (or new test), create test that:
       - Creates 500+ fake paths, calls `GpuContentSearch::new_for_paths()`, `load_paths()`
       - Runs `search_paths("test", &opts)` — record results as baseline
       - Calls `save_chunk_cache()` with engine's chunk_data
       - Calls `load_chunk_cache()` to get `MmapChunkData`
       - Creates new engine, calls `load_from_cache(&mmap)`
       - Runs same `search_paths("test", &opts)` — compare results to baseline
    2. Verify: same match count, same file paths in results
    3. Note: This test requires Metal device — mark with `#[ignore]` if CI has no GPU, or gate with `Device::system_default().is_some()`
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/chunk_cache.rs` (modify)
  - **Done when**: Integration test passes, results identical
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test cache_search_identical -- --nocapture 2>&1 | tail -15`
  - **Commit**: `test(chunk-cache): add integration test verifying search result identity`
  - _Requirements: AC-1.2, AC-3.4_
  - _Design: Test Strategy_

- [ ] 3.2 Test fallback on corrupt/missing cache
  - **Do**:
    1. Add test `test_fallback_corrupt_cache`: write garbage to chunks.cache, verify load returns None
    2. Add test `test_fallback_missing_cache`: delete cache file, verify load returns None (not panic)
    3. Add test `test_fallback_partial_write`: write only header (no chunk data), verify None
    4. Verify these are covered by existing unit tests from 1.4, consolidate if duplicate
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/chunk_cache.rs` (modify)
  - **Done when**: All fallback tests pass, no panics on any corrupt input
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test chunk_cache -- --nocapture 2>&1 | tail -15`
  - **Commit**: `test(chunk-cache): add fallback and corruption tests`
  - _Requirements: AC-2.4, FR-8, NFR-5_
  - _Design: Error Handling, Test Strategy_

- [ ] 3.3 [VERIFY] Quality checkpoint: cargo check && cargo clippy && cargo test
  - **Do**: Run full quality suite including all new tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo check 2>&1 | tail -3 && cargo clippy 2>&1 | tail -5 && cargo test 2>&1 | tail -10`
  - **Done when**: All tests pass, zero warnings
  - **Commit**: `chore(chunk-cache): pass quality checkpoint after testing` (only if fixes needed)

## Phase 4: Quality Gates

- [ ] 4.1 [VERIFY] Full local CI: cargo check && cargo clippy && cargo test && cargo build --release
  - **Do**: Run complete local CI suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo check && cargo clippy -- -D warnings && cargo test && cargo build --release 2>&1 | tail -10`
  - **Done when**: Build succeeds, all tests pass, no warnings treated as errors
  - **Commit**: `chore(chunk-cache): pass local CI` (if fixes needed)

- [ ] 4.2 Create PR and verify CI
  - **Do**:
    1. Verify current branch is a feature branch: `git branch --show-current`
    2. If on default branch, STOP and alert user
    3. Push branch: `git push -u origin <branch-name>`
    4. Create PR: `gh pr create --title "feat(engine): add mmap chunk cache for instant warm start" --body "..."`
    5. Wait for CI: `gh pr checks --watch`
  - **Verify**: `gh pr checks` shows all green
  - **Done when**: All CI checks green, PR ready for review
  - **Commit**: None (PR creation only)

## Phase 5: PR Lifecycle

- [ ] 5.1 Monitor CI and fix failures
  - **Do**:
    1. Check CI status: `gh pr checks`
    2. If any check fails, read failure details, fix locally, push fix
    3. Re-verify: `gh pr checks --watch`
  - **Verify**: `gh pr checks` all passing
  - **Done when**: All CI checks green

- [ ] 5.2 [VERIFY] AC checklist
  - **Do**: Programmatically verify each acceptance criteria:
    - AC-1.1: Warm start <100ms — verify `[ChunkCache] loaded` message appears in stderr within first 100ms (grep stderr of timed run)
    - AC-1.2: Search results identical — confirmed by integration test `cache_search_identical`
    - AC-2.1: Root mtime invalidation — confirmed by `test_wrong_root` unit test
    - AC-2.3: Fallback to walk — confirmed by `test_fallback_*` tests
    - AC-2.4: Corrupt cache graceful — confirmed by `test_invalid_magic`, `test_truncated_file` tests
    - AC-3.1: bytesNoCopy — grep for `new_buffer_with_bytes_no_copy` in search.rs
    - AC-3.2: StorageModeShared — grep for `StorageModeShared` in search.rs load_from_cache
    - AC-3.3: munmap on drop — grep for `munmap` in chunk_cache.rs Drop impl
    - AC-3.4: Identical search results — confirmed by integration test
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test chunk_cache -- --nocapture 2>&1 | tail -5 && grep -c 'new_buffer_with_bytes_no_copy' src/engine/search.rs && grep -c 'munmap' src/engine/chunk_cache.rs && grep -c 'StorageModeShared' src/engine/search.rs`
  - **Done when**: All acceptance criteria confirmed met via automated checks
  - **Commit**: None

## Notes

- **POC shortcuts taken**: None significant — this is a small feature (~150 lines)
- **Production TODOs**:
  - AC-1.3 (status bar "Loaded N paths from cache"): Requires UI status update from search_thread on cache load. Can be added via `SearchUpdate::Loading` message. Low priority cosmetic.
  - AC-2.2 (readdir count delta >5%): Design lists as Medium priority. Skipped for simplicity. File count is stored in header and can be validated later.
  - FR-7 (quick readdir count): Deferred — adds complexity for marginal invalidation improvement.
- **Key risk**: `new_buffer_with_bytes_no_copy` with `None` deallocator requires MmapChunkData to outlive GPU buffer. `mmap_holder` in search_thread owns lifetime. Verify drop order.
- **Test caveat**: GPU integration tests require Metal device. Tests gated with `Device::system_default()` check. CI without GPU skips these.
- **Cache location**: `~/.cache/gpu-search-ui/chunks.cache` (same dir as existing `paths.bin` from PathCache)
