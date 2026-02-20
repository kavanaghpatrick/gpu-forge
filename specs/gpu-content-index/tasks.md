---
spec: gpu-content-index
phase: tasks
total_tasks: 27
created: 2026-02-16
generated: auto
---

# Tasks: gpu-content-index

## Phase 1: Content Store (Make It Work)

Focus: ContentStore struct, FileContentMeta, anonymous mmap buffer, Metal bytesNoCopy wrapping.

- [x] 1.1 Create FileContentMeta and ContentStore structs
  - **Do**: Create `gpu-search/src/index/content_store.rs`. Define `FileContentMeta` as `#[repr(C)]` 32-byte struct (content_offset: u64, content_len: u32, path_id: u32, content_hash: u32, mtime: u32, trigram_count: u32, flags: u32). Define `ContentStore` struct with fields: `buffer: Vec<u8>` (content data), `files: Vec<FileContentMeta>` (metadata table), `total_bytes: u64`, `file_count: u32`. Add static assert for `size_of::<FileContentMeta>() == 32`. Add basic constructors: `ContentStore::new()` (empty), `ContentStore::with_capacity(estimated_bytes: usize)`. Add `insert(&mut self, content: &[u8], path_id: u32, content_hash: u32, mtime: u32) -> u32` that appends content to buffer, creates FileContentMeta entry, returns file_id. Add `content_for(&self, file_id: u32) -> Option<&[u8]>` accessor. Add `total_bytes()`, `file_count()` getters. Add module declaration in `src/index/mod.rs`.
  - **Files**: `gpu-search/src/index/content_store.rs`, `gpu-search/src/index/mod.rs`
  - **Done when**: `FileContentMeta` is 32 bytes, `ContentStore::insert` + `content_for` roundtrip works, module compiles
  - **Verify**: `cargo test --lib -p gpu-search content_store`
  - **Commit**: `feat(content-index): add ContentStore and FileContentMeta structs`
  - _Requirements: FR-1, FR-3_
  - _Design: ContentStore_

- [x] 1.2 Add anonymous mmap allocation for content buffer
  - **Do**: Replace `Vec<u8>` backing with anonymous mmap (`MAP_ANON`). Add `ContentStoreBuilder` that: (1) allocates page-aligned anonymous mmap via `libc::mmap(NULL, capacity, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, -1, 0)`, (2) copies file contents into the mmap region sequentially, (3) tracks write cursor for append. Add `finalize(self, device: &MTLDevice) -> ContentStore` that makes the mmap read-only via `mprotect(PROT_READ)` and creates Metal buffer via `bytesNoCopy`. Ensure the mmap pointer is page-aligned (16KB on Apple Silicon). Add `Drop` impl that calls `munmap`. Use `MmapBuffer::from_bytes` pattern as reference.
  - **Files**: `gpu-search/src/index/content_store.rs`
  - **Done when**: ContentStoreBuilder allocates anonymous mmap, appends content, finalizes with bytesNoCopy Metal buffer. Test verifies page alignment and Metal buffer creation.
  - **Verify**: `cargo test --lib -p gpu-search content_store`
  - **Commit**: `feat(content-index): anonymous mmap allocation for content buffer`
  - _Requirements: FR-1, FR-2_
  - _Design: ContentStore_

- [x] 1.3 Add Metal bytesNoCopy buffer wrapping
  - **Do**: In `ContentStore`, add `metal_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>` field. In `ContentStoreBuilder::finalize()`, call `device.newBufferWithBytesNoCopy_length_options_deallocator(ptr, aligned_len, StorageModeShared, None)`. If bytesNoCopy fails, fall back to `newBufferWithBytes` (copy path). Add `metal_buffer(&self) -> Option<&ProtocolObject<dyn MTLBuffer>>` accessor. Add test: create ContentStore, verify `metal_buffer()` returns Some, verify Metal buffer length >= total_bytes, verify Metal buffer contents match mmap'd data byte-for-byte (unsafe pointer comparison).
  - **Files**: `gpu-search/src/index/content_store.rs`
  - **Done when**: Metal buffer wraps content store mmap via bytesNoCopy. Test confirms GPU buffer contents match source data.
  - **Verify**: `cargo test --lib -p gpu-search content_store -- metal`
  - **Commit**: `feat(content-index): Metal bytesNoCopy buffer wrapping for GPU access`
  - _Requirements: FR-2_
  - _Design: ContentStore_

- [x] 1.4 Create ContentSnapshot and ContentIndexStore
  - **Do**: Create `gpu-search/src/index/content_snapshot.rs` with `ContentSnapshot` struct: fields `content_store: ContentStore`, `build_timestamp: u64`, `file_count: u32`. Implement `Send + Sync` (unsafe, same safety argument as IndexSnapshot). Create `gpu-search/src/index/content_index_store.rs` with `ContentIndexStore` struct wrapping `ArcSwap<Option<ContentSnapshot>>`. Add methods: `new()`, `snapshot() -> Guard<Arc<Option<ContentSnapshot>>>`, `swap(new: ContentSnapshot)`, `is_available() -> bool`. Follow IndexStore pattern exactly. Add module declarations in `src/index/mod.rs`. Add tests: store initially empty, swap makes new snapshot visible, old guard still sees old data.
  - **Files**: `gpu-search/src/index/content_snapshot.rs`, `gpu-search/src/index/content_index_store.rs`, `gpu-search/src/index/mod.rs`
  - **Done when**: ContentIndexStore compiles, tests for empty/swap/guard visibility pass
  - **Verify**: `cargo test --lib -p gpu-search content_index_store`
  - **Commit**: `feat(content-index): ContentSnapshot and ContentIndexStore with arc-swap`
  - _Requirements: FR-6_
  - _Design: ContentSnapshot, ContentIndexStore_

- [x] 1.5 Phase 1 checkpoint -- ContentStore end-to-end
  - **Do**: Write an integration test that: (1) creates a ContentStoreBuilder, (2) inserts 100 files with known content, (3) finalizes to ContentStore with Metal buffer, (4) verifies `content_for()` returns exact bytes for each file, (5) verifies Metal buffer is valid and page-aligned, (6) wraps in ContentSnapshot, stores in ContentIndexStore, (7) reads back via arc-swap guard, (8) verifies all content still accessible through the snapshot.
  - **Files**: `gpu-search/src/index/content_store.rs` (test module)
  - **Done when**: End-to-end test passes: build -> store -> snapshot -> arc-swap -> read back
  - **Verify**: `cargo test --lib -p gpu-search content_store -- checkpoint`
  - **Commit**: `feat(content-index): complete Phase 1 content store`
  - _Requirements: FR-1, FR-2, FR-6_

## Phase 2: Content Builder (Background Build)

Focus: Background filesystem walker + reader that populates ContentStore.

- [x] 2.1 Create ContentBuilder struct
  - **Do**: Create `gpu-search/src/index/content_builder.rs`. Define `ContentBuilder` with: `store: Arc<ContentIndexStore>`, `excludes: Arc<ExcludeTrie>`, `progress: Arc<AtomicUsize>`, `device: Retained<ProtocolObject<dyn MTLDevice>>`. Add `build(&self, root: &Path) -> Result<ContentSnapshot, BuildError>` method that: (1) walks `root` using `ignore::WalkBuilder` (same as existing scanner), (2) filters via BinaryDetector, (3) skips files > 100MB, (4) reads each file via `std::fs::read()`, (5) appends to ContentStoreBuilder, (6) increments progress counter, (7) finalizes and returns ContentSnapshot. Use `rayon::par_bridge()` for parallel file reads where possible.
  - **Files**: `gpu-search/src/index/content_builder.rs`, `gpu-search/src/index/mod.rs`
  - **Done when**: ContentBuilder builds ContentStore from a directory. Test with tempdir of 50 files.
  - **Verify**: `cargo test --lib -p gpu-search content_builder`
  - **Commit**: `feat(content-index): ContentBuilder for background filesystem reading`
  - _Requirements: FR-4, FR-5_
  - _Design: ContentBuilder_

- [x] 2.2 Add progress tracking and binary detection
  - **Do**: In ContentBuilder::build(), integrate `BinaryDetector::new()` to skip binary files. Track: files_scanned (total walked), files_indexed (text files added to store), bytes_indexed (total content bytes). Expose via `Arc<AtomicUsize>` progress counter. Add file size check: skip if size == 0 or size > MAX_FILE_SIZE (100MB, from streaming.rs). Add error handling: log and skip files that fail to read (permission denied, etc). Test: create tempdir with mix of text + binary files, verify only text files appear in content store.
  - **Files**: `gpu-search/src/index/content_builder.rs`
  - **Done when**: Binary files skipped, progress tracked, oversized files excluded
  - **Verify**: `cargo test --lib -p gpu-search content_builder -- binary`
  - **Commit**: `feat(content-index): binary detection and progress tracking in builder`
  - _Requirements: FR-5_
  - _Design: ContentBuilder_

- [x] 2.3 Create ContentDaemon lifecycle coordinator
  - **Do**: Create `gpu-search/src/index/content_daemon.rs`. Define `ContentDaemon` with: `store: Arc<ContentIndexStore>`, `builder_thread: Option<JoinHandle<()>>`, `progress: Arc<AtomicUsize>`. Add `start(store, device, excludes) -> Self` that spawns ContentBuilder on a background thread. On build completion, builder publishes ContentSnapshot to store via `store.swap()`. Add `shutdown(&mut self)` that joins the builder thread. Add `progress(&self) -> usize` accessor. Follow IndexDaemon pattern. Add module declaration in `src/index/mod.rs`.
  - **Files**: `gpu-search/src/index/content_daemon.rs`, `gpu-search/src/index/mod.rs`
  - **Done when**: ContentDaemon spawns background build, publishes snapshot on completion, shuts down cleanly
  - **Verify**: `cargo test --lib -p gpu-search content_daemon`
  - **Commit**: `feat(content-index): ContentDaemon background build lifecycle`
  - _Requirements: FR-4_
  - _Design: ContentDaemon_

- [x] 2.4 Phase 2 checkpoint -- background build works
  - **Do**: Integration test: (1) create tempdir with 200 files, (2) start ContentDaemon, (3) poll progress until complete, (4) verify ContentIndexStore is_available() returns true, (5) verify content_for() returns correct bytes for several files, (6) shutdown daemon cleanly.
  - **Files**: `gpu-search/src/index/content_daemon.rs` (test module)
  - **Done when**: Full background build lifecycle works end-to-end
  - **Verify**: `cargo test --lib -p gpu-search content_daemon -- checkpoint`
  - **Commit**: `feat(content-index): complete Phase 2 background builder`
  - _Requirements: FR-4, FR-5, FR-6_

## Phase 3: Search Integration (Zero Disk I/O)

Focus: Wire ContentStore into search pipeline, bypass fs::read, GPU dispatch from store.

- [x] 3.1 Add search_with_buffer to ContentSearchEngine
  - **Do**: In `gpu-search/src/search/content.rs`, add a new public method `search_with_buffer(&mut self, content_buffer: &ProtocolObject<dyn MTLBuffer>, chunk_metas: &[ChunkMetadata], pattern: &[u8], options: &SearchOptions) -> Vec<ContentMatch>`. This method: (1) writes pattern to pattern_buffer, (2) writes search params to params_buffer, (3) writes chunk_metas to metadata_buffer, (4) resets match_count to 0, (5) dispatches GPU compute with `content_buffer` at buffer(0) instead of `self.chunks_buffer`, (6) reads back matches. Make `ChunkMetadata` pub so content store integration can construct it. Add test: create a ContentStore with known content, call search_with_buffer, verify correct matches found.
  - **Files**: `gpu-search/src/search/content.rs`
  - **Done when**: search_with_buffer dispatches GPU search using external buffer, returns correct matches
  - **Verify**: `cargo test --lib -p gpu-search content -- search_with_buffer`
  - **Commit**: `feat(content-index): add search_with_buffer to ContentSearchEngine`
  - _Requirements: FR-8_
  - _Design: Modified ContentSearchEngine_

- [x] 3.2 Build ChunkMetadata from ContentStore
  - **Do**: Add `fn build_chunk_metadata(store: &ContentStore) -> Vec<ChunkMetadata>` utility function (in content_store.rs or a new helpers module). For each file in the content store, create ChunkMetadata entries: split file content into CHUNK_SIZE (4096 byte) chunks, set file_index to file_id, set offset_in_file for each chunk, set chunk_length, set flags (is_text=1, is_first, is_last). This maps the contiguous content buffer into the chunk-based format the GPU kernel expects. Add test: build metadata for 10 files of varying sizes, verify chunk count and offsets are correct.
  - **Files**: `gpu-search/src/index/content_store.rs`
  - **Done when**: ChunkMetadata generation from ContentStore works correctly with CHUNK_SIZE splitting
  - **Verify**: `cargo test --lib -p gpu-search content_store -- chunk_metadata`
  - **Commit**: `feat(content-index): ChunkMetadata builder from ContentStore`
  - _Requirements: FR-7_
  - _Design: Data Flow_

- [x] 3.3 Add content store fast-path to SearchOrchestrator
  - **Do**: In `gpu-search/src/search/orchestrator.rs`, add `content_store: Option<Arc<ContentIndexStore>>` field to `SearchOrchestrator`. Add constructor `with_content_store(device, pso_cache, index_store, content_store)`. In `search_streaming_inner()`, before the existing disk-based pipeline, add: `if let Some(cs) = &self.content_store { if cs.is_available() { return self.search_from_content_store(cs, &request, update_tx, session); } }`. Implement `search_from_content_store()` that: (1) loads ContentSnapshot via arc-swap guard, (2) builds ChunkMetadata from store, (3) calls engine.search_with_buffer(), (4) resolves matches (file_id -> path via FileContentMeta + path index), (5) sends results via update_tx channel, (6) returns SearchResponse. Filename matching logic stays the same (use path index).
  - **Files**: `gpu-search/src/search/orchestrator.rs`
  - **Done when**: Orchestrator checks content store availability, dispatches GPU from content buffer when available, falls back to disk when not
  - **Verify**: `cargo test --lib -p gpu-search orchestrator`
  - **Commit**: `feat(content-index): content store fast-path in SearchOrchestrator`
  - _Requirements: FR-7, FR-13_
  - _Design: Modified SearchOrchestrator_

- [x] 3.4 Phase 3 checkpoint -- zero disk I/O search
  - **Do**: Integration test: (1) create tempdir with 100 files containing known patterns, (2) build ContentStore and ContentIndexStore, (3) create SearchOrchestrator with content store, (4) search for known pattern, (5) verify results match expected file paths and line numbers, (6) verify results are identical to disk-based search path (A/B oracle test). Optionally: track that no fs::read() calls were made during search (use a wrapper or counter).
  - **Files**: `gpu-search/src/search/orchestrator.rs` (test module) or `gpu-search/tests/test_content_search.rs`
  - **Done when**: Content store search returns correct results, matching disk-based path
  - **Verify**: `cargo test --lib -p gpu-search orchestrator -- content_store`
  - **Commit**: `feat(content-index): complete Phase 3 zero disk I/O search`
  - _Requirements: FR-7, FR-8, AC-1.1, AC-1.2, AC-5.1_

## Phase 4: Persistence (GCIX Format)

Focus: Save ContentStore to disk, mmap on restart for instant warm start.

- [x] 4.1 Define GCIX header and write function
  - **Do**: Create `gpu-search/src/index/gcix.rs`. Define `GcixHeader` as `#[repr(C)]` struct: magic [u8; 4] = "GCIX", version: u32 = 1, file_count: u32, content_bytes: u64, meta_offset: u64, content_offset: u64, root_hash: u64, last_fsevents: u64, saved_at: u64, header_crc32: u32, padding to 16384 bytes. Add `pub fn save_gcix(store: &ContentStore, path: &Path, root_hash: u64, fsevents_id: u64) -> io::Result<()>` that writes: (1) header at offset 0, (2) FileContentMeta table at meta_offset = 16384, (3) padding to next page boundary, (4) content data at content_offset (page-aligned). Content_offset must be page-aligned for bytesNoCopy on mmap reload. Add module declaration in mod.rs.
  - **Files**: `gpu-search/src/index/gcix.rs`, `gpu-search/src/index/mod.rs`
  - **Done when**: save_gcix writes valid GCIX file. Test: save, read back raw bytes, verify header magic and content data.
  - **Verify**: `cargo test --lib -p gpu-search gcix -- save`
  - **Commit**: `feat(content-index): GCIX file format header and save function`
  - _Requirements: FR-9_
  - _Design: GCIX File Format_

- [x] 4.2 Add GCIX load function with mmap
  - **Do**: In `gpu-search/src/index/gcix.rs`, add `pub fn load_gcix(path: &Path, device: Option<&MTLDevice>) -> Result<ContentSnapshot, CacheError>`. Implementation: (1) `MmapBuffer::from_file(path)`, (2) validate header (magic, version, CRC32), (3) extract FileContentMeta table via pointer arithmetic, (4) create Metal buffer from content region via bytesNoCopy (content_offset is page-aligned), (5) construct ContentStore and ContentSnapshot. Verify content_offset is page-aligned. Test: save_gcix then load_gcix, verify roundtrip: all file content accessible, metal buffer valid, file_count matches.
  - **Files**: `gpu-search/src/index/gcix.rs`
  - **Done when**: save -> load roundtrip preserves all content and metadata. Metal buffer created via bytesNoCopy from mmap'd content region.
  - **Verify**: `cargo test --lib -p gpu-search gcix -- load`
  - **Commit**: `feat(content-index): GCIX load with mmap and bytesNoCopy`
  - _Requirements: FR-10, AC-3.2_
  - _Design: GCIX File Format_

- [x] 4.3 Wire GCIX into ContentDaemon
  - **Do**: In ContentDaemon::start(), before spawning builder thread: check for existing GCIX file at `~/.gpu-search/index/global.gcix`. If exists, try `load_gcix()`. If valid, publish snapshot to ContentIndexStore and skip build. If load fails (corrupt, version mismatch), delete file and proceed with build. After build completes, call `save_gcix()` to persist. Add `gcix_path()` helper using `dirs::data_dir()` or `~/.gpu-search/index/`. Test: build -> save -> simulate restart by loading -> verify content accessible without rebuild.
  - **Files**: `gpu-search/src/index/content_daemon.rs`
  - **Done when**: ContentDaemon loads GCIX on restart, skips build if valid, saves after build
  - **Verify**: `cargo test --lib -p gpu-search content_daemon -- gcix`
  - **Commit**: `feat(content-index): GCIX persistence in ContentDaemon lifecycle`
  - _Requirements: FR-9, FR-10, AC-3.1, AC-3.4_
  - _Design: ContentDaemon_

## Phase 5: Incremental Updates (FSEvents)

Focus: FSEvents integration to update ContentStore when files change.

- [x] 5.1 Add file update method to ContentStore
  - **Do**: Add `ContentStore::update_file(&mut self, file_id: u32, new_content: &[u8], new_hash: u32, new_mtime: u32)` method. Strategy: append new content to end of buffer (append-only, old content becomes dead space). Update FileContentMeta entry with new offset, length, hash, mtime. Track dead_bytes counter for future compaction. Add `ContentStore::remove_file(&mut self, file_id: u32)` that marks entry flags with deleted bit. Add `ContentStore::add_file(&mut self, content: &[u8], path_id: u32, hash: u32, mtime: u32) -> u32` for new files. Test: insert file, update content, verify content_for returns new content.
  - **Files**: `gpu-search/src/index/content_store.rs`
  - **Done when**: Update/remove/add work correctly. Old content accessible by offset, new content returned by content_for.
  - **Verify**: `cargo test --lib -p gpu-search content_store -- update`
  - **Commit**: `feat(content-index): incremental update methods for ContentStore`
  - _Requirements: FR-12_
  - _Design: ContentStore_

- [x] 5.2 Wire FSEvents changes to content store updates
  - **Do**: In ContentDaemon, subscribe to the same `crossbeam_channel::Receiver<FsChange>` that IndexWriter uses (or create a second receiver from a broadcast). For each `FsChange::Modify(path)` or `FsChange::Create(path)`: (1) check if file is text (BinaryDetector), (2) read file content, (3) compute CRC32 hash, (4) call content_store.update_file() or add_file(), (5) create new ContentSnapshot, (6) swap into ContentIndexStore. For `FsChange::Delete(path)`: call remove_file(). For `FsChange::MustRescan`: trigger full rebuild. Debounce rapid changes (coalesce changes within 500ms window).
  - **Files**: `gpu-search/src/index/content_daemon.rs`
  - **Done when**: File modifications trigger content store updates. Test: modify file in watched directory, verify search returns new content.
  - **Verify**: `cargo test --lib -p gpu-search content_daemon -- fsevents`
  - **Commit**: `feat(content-index): FSEvents integration for incremental updates`
  - _Requirements: FR-11, AC-4.1, AC-4.2, AC-4.3, AC-4.4_
  - _Design: ContentDaemon_

- [x] 5.3 Phase 5 checkpoint -- live updates work
  - **Do**: Integration test: (1) build content store from tempdir, (2) create a new file, send FsChange::Create, verify file appears in search, (3) modify a file, send FsChange::Modify, verify search returns new content, (4) delete a file, send FsChange::Delete, verify file gone from search. Verify all operations complete within 2 seconds.
  - **Files**: `gpu-search/src/index/content_daemon.rs` (test module)
  - **Done when**: Create/modify/delete all reflected in content store and search results
  - **Verify**: `cargo test --lib -p gpu-search content_daemon -- incremental_checkpoint`
  - **Commit**: `feat(content-index): complete Phase 5 incremental updates`
  - _Requirements: FR-11, FR-12, AC-4.1 through AC-4.5_

## Phase 6: Testing

Focus: Comprehensive correctness, regression, and performance tests.

- [x] 6.1 Unit tests for ContentStore
  - **Do**: In content_store.rs test module, add: (1) `test_insert_retrieve_roundtrip` -- insert 10 files, verify byte-exact retrieval, (2) `test_empty_file` -- insert empty content, verify no panic, (3) `test_large_file` -- insert 10MB content, verify roundtrip, (4) `test_binary_content` -- insert all 256 byte values, verify roundtrip, (5) `test_file_content_meta_size` -- assert size_of == 32, (6) `test_metal_buffer_valid` -- verify Metal buffer contents match source, (7) `test_page_alignment` -- verify mmap pointer is 16KB aligned.
  - **Files**: `gpu-search/src/index/content_store.rs`
  - **Done when**: All 7 unit tests pass
  - **Verify**: `cargo test --lib -p gpu-search content_store`
  - **Commit**: `test(content-index): unit tests for ContentStore`
  - _Requirements: AC-5.2_

- [x] 6.2 Unit tests for GCIX format
  - **Do**: In gcix.rs test module, add: (1) `test_save_load_roundtrip` -- save 50 files, load back, verify all content, (2) `test_header_validation` -- corrupt magic, verify CacheError, (3) `test_version_mismatch` -- wrong version, verify CacheError, (4) `test_crc32_validation` -- corrupt CRC, verify detection, (5) `test_content_region_page_aligned` -- verify content_offset is multiple of 16384, (6) `test_metal_buffer_from_gcix` -- load with device, verify bytesNoCopy succeeds.
  - **Files**: `gpu-search/src/index/gcix.rs`
  - **Done when**: All 6 GCIX tests pass
  - **Verify**: `cargo test --lib -p gpu-search gcix`
  - **Commit**: `test(content-index): unit tests for GCIX format`
  - _Requirements: AC-3.3, AC-3.4_

- [x] 6.3 Integration test: A/B oracle (in-memory vs disk)
  - **Do**: Create `gpu-search/tests/test_content_vs_disk.rs`. Generate 200-file tempdir with known patterns. Run both paths: (1) disk-based: SearchOrchestrator without content store, (2) in-memory: SearchOrchestrator with content store. For 20 different patterns, verify result sets are identical (same file paths, same line numbers). This is the definitive correctness gate.
  - **Files**: `gpu-search/tests/test_content_vs_disk.rs`
  - **Done when**: All 20 pattern comparisons produce identical results between disk and memory paths
  - **Verify**: `cargo test --test test_content_vs_disk -p gpu-search`
  - **Commit**: `test(content-index): A/B oracle test comparing in-memory vs disk search`
  - _Requirements: AC-5.1, AC-5.2, AC-5.3_

- [x] 6.4 Benchmark: content store search vs disk-based
  - **Do**: Create `gpu-search/benches/content_search.rs` using criterion. Generate 1000-file corpus (~5MB total). Benchmark: (1) `disk_search` -- existing streaming pipeline with fs::read, (2) `content_store_search` -- content store fast-path. Report throughput (MB/s) and latency (us). Add bench entry in Cargo.toml. Target: content_store_search latency < 10% of disk_search latency.
  - **Files**: `gpu-search/benches/content_search.rs`, `gpu-search/Cargo.toml`
  - **Done when**: Benchmark runs, content store path significantly faster than disk path
  - **Verify**: `cargo bench --bench content_search -p gpu-search`
  - **Commit**: `perf(content-index): benchmark content store vs disk search`
  - _Requirements: NFR-1_

## Phase 7: Quality Gates

Focus: Full test suite, linting, CI verification.

- [x] 7.1 Verify all existing tests pass
  - **Do**: Run the full existing test suite to confirm no regressions. Run: `cargo test --lib -p gpu-search` (all lib tests), `cargo test --test test_search_accuracy -p gpu-search`, `cargo test --test test_false_positives -p gpu-search`, `cargo test --test test_gpu_cpu_consistency -p gpu-search`. All must pass with zero failures.
  - **Files**: (none -- verification only)
  - **Done when**: All existing tests pass (561+ lib tests, all integration tests)
  - **Verify**: `cargo test --lib -p gpu-search && cargo test --test test_search_accuracy --test test_false_positives --test test_gpu_cpu_consistency -p gpu-search`
  - **Commit**: `fix(content-index): address lint/type issues` (if any fixes needed)
  - _Requirements: NFR-5_

- [x] 7.2 Run clippy and fix warnings
  - **Do**: Run `cargo clippy -p gpu-search -- -W warnings`. Fix any new warnings introduced by content index code. Common issues: unused imports, unused variables, missing docs on public items.
  - **Files**: All new files in `gpu-search/src/index/`
  - **Done when**: `cargo clippy -p gpu-search` reports zero warnings for new code
  - **Verify**: `cargo clippy -p gpu-search -- -W warnings`
  - **Commit**: `fix(content-index): address clippy warnings`

- [x] 7.3 VF: Verification Final
  - **Do**: Run the complete verification suite: (1) `cargo test --lib -p gpu-search` -- all unit tests, (2) `cargo test --test test_content_vs_disk -p gpu-search` -- A/B oracle, (3) `cargo bench --bench content_search -p gpu-search -- --test` -- benchmark compiles, (4) verify content store search returns results for a known pattern. Document results in .progress.md.
  - **Files**: `specs/gpu-content-index/.progress.md`
  - **Done when**: All tests pass, benchmark runs, content store search confirmed working
  - **Verify**: `cargo test -p gpu-search && cargo bench --bench content_search -p gpu-search -- --test`
  - **Commit**: `docs(content-index): verification final results`

- [x] 7.4 Create PR and verify CI (skipped: changes are part of feat/gpu-query branch)
  - **Do**: Push branch, create PR with `gh pr create`. PR description should summarize: ContentStore subsystem, zero disk I/O search, GCIX persistence, FSEvents integration, A/B oracle test. Wait for CI checks.
  - **Verify**: `gh pr checks --watch`
  - **Done when**: PR created, CI green
  - **Commit**: (none -- PR creation only)

## Notes

- **POC shortcuts taken**: Phase 1-3 use Vec<u8> initially then upgrade to anonymous mmap. No trigram index (Phase 2 of TECH.md). No compaction of dead space. No memory budget enforcement.
- **Production TODOs**: Trigram inverted index (Phase 2 of TECH.md), compressed postings, memory cap with LRU, compaction, adaptive debounce.
- **Key insight**: Existing Metal shader (`content_search_kernel`) requires ZERO changes. Only buffer(0) source changes from per-batch chunks_buffer to content store buffer.
