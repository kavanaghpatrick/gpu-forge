---
spec: gpu-search-ui-fix
phase: tasks
total_tasks: 16
created: 2026-02-17
generated: auto
---

# Tasks: gpu-search-ui-fix

## Phase 1: Make It Work (POC)

Focus: Fix GPU kernel context extraction, verify path prefix, add cache persistence. Skip tests initially.

- [x] 1.1 Add GPU-side newline scanning to turbo_search_kernel
  - **Do**: In `/Users/patrickkavanagh/gpu-search-ui/src/engine/shader.rs`, modify the turbo_search_kernel match-writing loop (lines 309-326). For each match, cast `data` to `device const uchar*`, scan backward from the match position to find `\n` or chunk start (`chunk_idx * CHUNK_SIZE`), scan forward from match end to find `\n` or chunk end (`chunk_idx * CHUNK_SIZE + chunk_len`). Write `context_start = line_start - chunk_idx * CHUNK_SIZE` (offset within chunk) and `context_len = min(line_end - line_start, MAX_CONTEXT)`. Keep existing `column` field as byte offset in chunk for backward compatibility.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/shader.rs`
  - **Done when**: turbo_search_kernel writes newline-bounded context_start and context_len per match
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build 2>&1 | tail -5` (shader compiles)
  - **Commit**: `fix(shader): add GPU-side newline scanning to turbo_search_kernel`
  - _Requirements: FR-1_
  - _Design: Component A_

- [x] 1.2 Update search_paths() to use GPU-provided context offsets
  - **Do**: In `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`, replace the CPU backward/forward newline scan in `search_paths()` (lines 401-412) with direct slice using GPU-provided `m.context_start` and `m.context_len`. Calculate `abs_start = chunk_idx * CHUNK_SIZE + m.context_start as usize`, `abs_end = (abs_start + m.context_len as usize).min(self.chunk_data.len())`. Slice `&self.chunk_data[abs_start..abs_end]` and convert to string. Remove the backward/forward while loops.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`
  - **Done when**: search_paths() uses GPU offsets instead of CPU scan; existing tests still pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test test_load_paths_and_search_basic test_chunk_data_integrity test_many_paths_search -- --nocapture 2>&1 | tail -20`
  - **Commit**: `fix(search): use GPU-provided context offsets in search_paths()`
  - _Requirements: FR-2_
  - _Design: Component B_

- [x] 1.3 Verify absolute path preservation with diagnostic test
  - **Do**: In `/Users/patrickkavanagh/gpu-search-ui/tests/test_path_search.rs`, add a test `test_absolute_path_prefix` that loads paths like `/usr/share/vim/vim91/plugin/matchit.vim` and `/Users/patrickkavanagh/Documents/test.txt`, searches for `vim91`, and asserts that the matched `file_path` starts with `/usr/share/`. Also verify no result starts without `/`. If test fails, debug and fix `load_paths()` or the extraction logic.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/tests/test_path_search.rs`
  - **Done when**: Test passes confirming absolute paths are preserved
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test test_absolute_path_prefix -- --nocapture 2>&1 | tail -15`
  - **Commit**: `test(search): add absolute path prefix verification test`
  - _Requirements: FR-3, AC-2.1_
  - _Design: Component B_

- [ ] 1.4 Add PathCache save/load to index.rs
  - **Do**: In `/Users/patrickkavanagh/gpu-search-ui/src/engine/index.rs`, add a `PathCache` struct with `save(root: &Path, paths: &[PathBuf])` and `load(root: &Path) -> Option<Vec<PathBuf>>` methods. Cache file at `~/.cache/gpu-search-ui/paths.bin`. Format: 8-byte magic `GPUSRCH\0`, 4-byte version, 8-byte unix timestamp, 4-byte root_path_len + root_path bytes, 4-byte file_count, then per file: 4-byte path_len + path bytes. `load()` returns None if file missing, corrupt, wrong root, or age > 3600s.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/index.rs`
  - **Done when**: `PathCache::save()` and `PathCache::load()` compile and unit tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test test_path_cache -- --nocapture 2>&1 | tail -10`
  - **Commit**: `feat(index): add PathCache persistence for path index`
  - _Requirements: FR-4, FR-5_
  - _Design: Component C_

- [ ] 1.5 Integrate PathCache into app.rs index_thread
  - **Do**: In `/Users/patrickkavanagh/gpu-search-ui/src/app.rs`, at the start of `index_thread()`, call `PathCache::load(&root)`. If it returns `Some(paths)`, send them via `batch_tx` and `index_tx.send(IndexUpdate::Complete)`, then return early. After the existing walk loop completes, call `PathCache::save(&root, &all_collected_paths)` where `all_collected_paths` is built by collecting all sent batches. Import `PathCache` from `crate::engine::index`.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/app.rs`
  - **Done when**: App loads cached index on second launch; first launch still walks and saves cache
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(app): use PathCache for instant index loading on launch`
  - _Requirements: FR-4, AC-3.1, AC-3.2_
  - _Design: Component C_

- [ ] 1.6 POC Checkpoint
  - **Do**: Run full test suite to verify all fixes work end-to-end. Run profiling test to confirm CPU extract time reduction.
  - **Done when**: All existing tests pass and search_paths CPU extract time reduced
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test 2>&1 | tail -20 && cargo test --release test_profile_gpu_vs_cpu -- --nocapture --ignored 2>&1 | tail -15`
  - **Commit**: `fix(search): complete POC — GPU context extraction + index persistence`

## Phase 2: Refactoring

After POC validated, clean up code.

- [ ] 2.1 Clean up turbo_search_kernel newline scan
  - **Do**: Extract the newline scanning into an inline MSL function `find_line_bounds()` that takes `device const uchar* data, uint match_pos, uint chunk_start, uint chunk_end` and returns `(uint line_start, uint line_end)`. Use this in turbo_search_kernel. Consider also updating content_search_kernel to use the same function for consistency.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/shader.rs`
  - **Done when**: Newline scanning is a reusable inline function; tests still pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test 2>&1 | tail -10`
  - **Commit**: `refactor(shader): extract find_line_bounds() inline function`
  - _Design: Component A_

- [ ] 2.2 Add error handling to PathCache
  - **Do**: Add proper error logging to PathCache::save() (log if cache dir creation fails, if write fails). Add validation to load() (check magic bytes, version, CRC or at least file_count sanity). Handle edge cases: empty path list, root path with spaces, very large caches.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/index.rs`
  - **Done when**: PathCache handles all error cases gracefully with eprintln diagnostics
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test test_path_cache -- --nocapture 2>&1 | tail -10`
  - **Commit**: `refactor(index): add error handling and validation to PathCache`
  - _Design: Component C, Error Handling_

## Phase 3: Testing

- [ ] 3.1 Create comprehensive GPU correctness test suite
  - **Do**: Create `/Users/patrickkavanagh/gpu-search-ui/tests/test_gpu_correctness.rs` with tests: (1) `test_context_no_truncation` — load paths > 64 bytes, verify full path returned. (2) `test_chunk_boundary_path` — create paths that land exactly at 4096-byte boundary, verify correct extraction. (3) `test_long_paths` — paths 200-400 bytes. (4) `test_unicode_paths` — paths with accented characters, CJK. (5) `test_case_insensitive_search` — mixed case paths, case_sensitive=false. (6) `test_absolute_prefix_preserved` — verify leading `/` on all results.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/tests/test_gpu_correctness.rs`
  - **Done when**: All 6+ tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test --test test_gpu_correctness -- --nocapture 2>&1 | tail -20`
  - **Commit**: `test(search): add comprehensive GPU correctness test suite`
  - _Requirements: FR-6, AC-4.1 through AC-4.5_
  - _Design: Component D_

- [ ] 3.2 Add performance regression test
  - **Do**: In `/Users/patrickkavanagh/gpu-search-ui/tests/test_gpu_correctness.rs`, add `test_perf_500k_under_10ms` that generates 500K paths, loads them, runs `search_paths("friendship")` 10 times, asserts average < 10ms. Mark with `#[ignore]` so it only runs explicitly. Also add `test_perf_improvement` that compares GPU-context search vs manual CPU scan (by timing search_paths with 50K max_results and verifying < 5ms at 1M paths).
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/tests/test_gpu_correctness.rs`
  - **Done when**: Performance tests pass and show improvement
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test --release --test test_gpu_correctness test_perf_500k_under_10ms -- --nocapture --ignored 2>&1 | tail -10`
  - **Commit**: `test(search): add performance regression gate`
  - _Requirements: AC-4.6, NFR-1_

- [ ] 3.3 Add PathCache unit tests
  - **Do**: In `/Users/patrickkavanagh/gpu-search-ui/src/engine/index.rs`, add tests: `test_path_cache_round_trip` (save + load returns same paths), `test_path_cache_stale` (set timestamp to 2 hours ago, verify load returns None), `test_path_cache_wrong_root` (save with root A, load with root B returns None), `test_path_cache_corrupt` (write garbage, verify load returns None).
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/index.rs`
  - **Done when**: All 4 cache tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test test_path_cache -- --nocapture 2>&1 | tail -15`
  - **Commit**: `test(index): add PathCache persistence tests`
  - _Requirements: AC-3.3, AC-3.4_

## Phase 4: Quality Gates

- [ ] 4.1 Run full test suite
  - **Do**: Run all tests (unit, integration, correctness) and verify 100% pass. Run clippy for lint warnings. Run profile tests to confirm performance improvement.
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test 2>&1 | tail -30 && cargo clippy 2>&1 | tail -10`
  - **Done when**: All tests pass, no clippy warnings
  - **Commit**: `fix(search): address lint/type issues` (if needed)

- [ ] 4.2 Run performance benchmark comparison
  - **Do**: Run criterion benchmarks before and after to quantify improvement. Compare search_latency at 500K and 1M scales. Verify throughput >= 30 GB/s.
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo bench --bench gpu_search_bench -- --sample-size 50 2>&1 | tail -30`
  - **Done when**: Benchmarks show measurable improvement in search latency

- [ ] 4.3 VF — Verify all 3 bugs fixed
  - **Do**: Run diagnostic tests that specifically verify: (1) No 64-byte truncation in context extraction — search for a string in a path > 100 bytes, verify full path returned. (2) Absolute paths preserved — verify results start with `/`. (3) Cache persistence — run app twice, verify second launch uses cache. Compare against BEFORE state captured in .progress.md.
  - **Verify**: `cd /Users/patrickkavanagh/gpu-search-ui && cargo test --test test_gpu_correctness -- --nocapture 2>&1 | tail -30`
  - **Done when**: All 3 bugs verified fixed

- [ ] 4.4 Create PR
  - **Do**: Push branch, create PR with gh CLI. Title: "fix: GPU context extraction, path prefix, index persistence". Body includes bug descriptions, performance numbers, and test summary.
  - **Verify**: `gh pr checks --watch` all green
  - **Done when**: PR created and CI passes

## Notes

- **POC shortcuts**: PathCache uses simple binary format without CRC; no background cache refresh
- **Production TODOs**: Add background re-scan after cache load to catch new files; add CRC to cache format
- **Key insight**: turbo_search_kernel reads device buffer for newline scan (not local_data which is only 128 bytes)
- **All code at**: `/Users/patrickkavanagh/gpu-search-ui/` — NOT in gpu_kernel/
