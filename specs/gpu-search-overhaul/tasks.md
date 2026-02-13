---
spec: gpu-search-overhaul
phase: tasks
total_tasks: 28
created: 2026-02-12
generated: auto
---

# Tasks: gpu-search-overhaul

## Phase 1: Make It Work (POC) -- Fix Accuracy

Focus: Fix the P0 false positive bug end-to-end. Add CPU verification. No new features -- accuracy first.

- [x] 1.1 Add memchr CPU verification module
  - **Do**: Create `src/search/verify.rs` with `cpu_verify_matches()` function. Uses `memchr::memmem::find_iter()` to find all pattern occurrences in file content. Returns `VerificationResult { confirmed, false_positives, missed }`. Add `VerifyMode` enum (Off/Sample/Full). Add `pub mod verify;` to `src/search/mod.rs`.
  - **Files**: `gpu-search/src/search/verify.rs` (create), `gpu-search/src/search/mod.rs` (modify)
  - **Done when**: `verify.rs` compiles, `cpu_verify_matches()` correctly finds patterns via memchr on test strings
  - **Verify**: `cargo check -p gpu-search`
  - **Commit**: `feat(gpu-search): add CPU verification layer with memchr::memmem`
  - _Requirements: FR-1_
  - _Design: Component A_

- [x] 1.2 Add accuracy test: assert every returned line contains pattern
  - **Do**: Create `tests/test_search_accuracy.rs`. Build test corpus in tempdir: 10 files (100B to 1MB) with known patterns at known byte offsets. Test `test_accuracy_line_content_contains_pattern`: for each of 10 patterns, run `SearchOrchestrator::search()`, assert EVERY `ContentMatch.line_content` contains the pattern. Test `test_no_false_positives_multi_file_batch`: 5 files of different sizes, one unique pattern, assert no cross-file contamination. These tests MAY FAIL initially (they catch the P0 bug).
  - **Files**: `gpu-search/tests/test_search_accuracy.rs` (create)
  - **Done when**: Tests compile and run (expected: some may fail due to P0 bug)
  - **Verify**: `cargo test -p gpu-search --test test_search_accuracy -- --nocapture 2>&1 | head -50`
  - **Commit**: `test(gpu-search): add accuracy test suite for false positive detection`
  - _Requirements: AC-1.1, AC-1.3_
  - _Design: Component E_

- [x] 1.3 Fix byte_offset mapping in collect_results()
  - **Do**: In `src/search/content.rs`, `collect_results()` at line ~408: change `byte_offset` calculation. Currently uses `m.chunk_index * CHUNK_SIZE + m.context_start` where `context_start` has ambiguous meaning. Fix: store `m.chunk_index * CHUNK_SIZE as u32 + (m.column + (m.context_start - m.context_start % CHUNK_SIZE as u32).min(m.context_start))` -- or more directly, use the match's actual byte position: `m.chunk_index * CHUNK_SIZE as u32 + m.context_start - m.column + m.column` (i.e., the global position of the match). Key insight: `context_start` = `offset_in_chunk + line_start_in_window`, but for byte_offset we need `offset_in_chunk + local_pos` = the match position. The GPU writes `column = local_pos - line_start`, so `match_pos_in_chunk = context_start + column`. Use `byte_offset = m.chunk_index * CHUNK_SIZE as u32 + m.context_start + m.column` to get the actual match byte offset (not line start).
  - **Files**: `gpu-search/src/search/content.rs` (~line 408)
  - **Done when**: `collect_results()` produces byte_offset pointing to the match byte, not line start
  - **Verify**: `cargo test -p gpu-search`
  - **Commit**: `fix(gpu-search): correct byte_offset in collect_results to use match position`
  - _Requirements: FR-5_
  - _Design: Component E_

- [x] 1.4 Add multi-file batch chunk mapping test
  - **Do**: Add test `test_accuracy_multi_file_batch_byte_offset` in `test_search_accuracy.rs`. Create 5 files: file_a (100B, "ALPHA" at byte 50), file_b (8000B, "ALPHA" at byte 4000), file_c (200B, "ALPHA" at byte 10), file_d (16000B, "ALPHA" at bytes 1000 and 12000), file_e (50B, no "ALPHA"). Search "ALPHA". Assert: match count = 5, file_e never in results, every match line_content contains "ALPHA".
  - **Files**: `gpu-search/tests/test_search_accuracy.rs` (extend)
  - **Done when**: Test passes with correct file attribution for all matches
  - **Verify**: `cargo test -p gpu-search --test test_search_accuracy test_accuracy_multi_file -- --nocapture`
  - **Commit**: `test(gpu-search): add multi-file batch byte_offset mapping test`
  - _Requirements: AC-1.3_
  - _Design: Component E_

- [x] 1.5 Wire CPU verification into dispatch_gpu_batch
  - **Do**: In `orchestrator.rs`, `dispatch_gpu_batch()` (~line 468): after GPU results collected and before `resolve_match()`, if `GPU_SEARCH_VERIFY` env var is set, run `cpu_verify_matches()` on each file's content and GPU byte offsets. Log any mismatches via `eprintln!("[VERIFY]...")`. In test mode (env var Full), assert zero false positives.
  - **Files**: `gpu-search/src/search/orchestrator.rs` (modify ~20 lines in dispatch_gpu_batch)
  - **Done when**: Running with `GPU_SEARCH_VERIFY=full` logs verification results, no false positives in test corpus
  - **Verify**: `GPU_SEARCH_VERIFY=full cargo test -p gpu-search --test test_search_accuracy -- --nocapture`
  - **Commit**: `feat(gpu-search): wire CPU verification into dispatch pipeline`
  - _Requirements: FR-1, AC-1.5_
  - _Design: Component A_

- [x] 1.6 Add resolve_match() unit tests
  - **Do**: Add inline `#[cfg(test)] mod tests` in `orchestrator.rs` (or extend existing). Tests: `test_resolve_match_correct_line` (known byte_offset -> exact line+column), `test_resolve_match_rejects_wrong_line` (bad offset -> None), `test_resolve_match_multi_chunk_file` (offset in chunk 2 of 5), `test_resolve_match_last_line_no_newline`, `test_resolve_match_empty_file`, `test_resolve_match_case_insensitive`. Make `resolve_match` `pub(crate)` if needed.
  - **Files**: `gpu-search/src/search/orchestrator.rs` (add ~80 lines of tests)
  - **Done when**: All 6 resolve_match unit tests pass
  - **Verify**: `cargo test -p gpu-search orchestrator::tests::test_resolve_match -- --nocapture`
  - **Commit**: `test(gpu-search): add resolve_match() unit tests covering edge cases`
  - _Requirements: AC-1.4_
  - _Design: Component E_

- [x] 1.7 POC Checkpoint -- verify zero false positives
  - **Do**: Run full accuracy test suite and existing 409 tests. Verify zero false positives across all patterns. Check `matches_rejected` counter is low (ideally zero after fix). Run with Metal shader validation enabled.
  - **Done when**: All accuracy tests pass, all existing tests pass, zero false positives
  - **Verify**: `MTL_SHADER_VALIDATION=1 cargo test -p gpu-search 2>&1 | tail -20`
  - **Commit**: `feat(gpu-search): complete accuracy POC -- zero false positives`

## Phase 2: Features -- Performance & Index

Focus: Wire filesystem index, add pipeline profiler, add FSEvents watcher.

- [x] 2.1 Implement PipelineProfile struct
  - **Do**: Create `src/search/profile.rs` with `PipelineProfile` struct (walk_us, filter_us, batch_us, gpu_load_us, gpu_dispatch_us, resolve_us, total_us + counters for files_walked, files_filtered, files_searched, bytes_searched, gpu_dispatches, matches_raw, matches_resolved, matches_rejected, ttfr_us). Add `Display` impl for human-readable output. Add `pub mod profile;` to `src/search/mod.rs`.
  - **Files**: `gpu-search/src/search/profile.rs` (create), `gpu-search/src/search/mod.rs` (modify)
  - **Done when**: Profile struct compiles, Display impl formats stage breakdown
  - **Verify**: `cargo check -p gpu-search`
  - **Commit**: `feat(gpu-search): add PipelineProfile struct with per-stage timing`
  - _Requirements: FR-2, AC-4.1_
  - _Design: Component B_

- [x] 2.2 Instrument orchestrator with per-stage timing
  - **Do**: In `orchestrator.rs`, add `Instant::now()` checkpoints at: walk_and_filter start/end, filter inline, batch loop, GPU dispatch (wrap `engine.search_files()`), resolve_match loop, TTFR (first SearchUpdate::ContentMatches send). Populate `PipelineProfile` fields. Add `profile` field to `SearchResponse`. Merge existing `StreamingProfile` timing into `PipelineProfile`.
  - **Files**: `gpu-search/src/search/orchestrator.rs` (~50 lines instrumentation), `gpu-search/src/search/types.rs` (add profile field to SearchResponse)
  - **Done when**: Every search produces a populated PipelineProfile in SearchResponse
  - **Verify**: `cargo test -p gpu-search --test test_orchestrator_integration -- --nocapture 2>&1 | grep -i profile`
  - **Commit**: `feat(gpu-search): instrument pipeline with per-stage timing`
  - _Requirements: FR-2, AC-4.2, AC-4.3_
  - _Design: Component B_

- [x] 2.3 Wire MmapIndexCache into orchestrator search path
  - **Do**: In `orchestrator.rs`, modify `search_streaming()` Stage 1. Before spawning walk_and_filter producer: check if GSIX index exists via `SharedIndexManager::index_path()`, load via `MmapIndexCache::load_mmap()`, check staleness. If fresh: iterate index entries into crossbeam channel (instant, no walk). If stale/missing: fall back to walk_and_filter as before. Add `index_path` to SearchRequest or derive from root. Import from `crate::index::{cache::MmapIndexCache, shared_index::SharedIndexManager}`.
  - **Files**: `gpu-search/src/search/orchestrator.rs` (~40 lines in search_streaming), `gpu-search/src/search/types.rs` (optional: add index_path to SearchRequest)
  - **Done when**: Orchestrator uses index when available, falls back to walk when not
  - **Verify**: `cargo test -p gpu-search --test test_orchestrator_integration -- --nocapture`
  - **Commit**: `feat(gpu-search): wire mmap index into orchestrator search path`
  - _Requirements: FR-3, AC-2.4_
  - _Design: Component C_

- [x] 2.4 Add notify v7 + debouncer dependencies
  - **Do**: Add `notify = "7"` and `notify-debouncer-mini = "0.5"` to `Cargo.toml` [dependencies]. Run cargo check to ensure they compile.
  - **Files**: `gpu-search/Cargo.toml` (add 2 lines)
  - **Done when**: `cargo check -p gpu-search` succeeds with new deps
  - **Verify**: `cargo check -p gpu-search`
  - **Commit**: `build(gpu-search): add notify v7 and debouncer-mini dependencies`
  - _Requirements: FR-4_
  - _Design: Component D_

- [x] 2.5 Implement IndexWatcher with FSEvents
  - **Do**: Create `src/index/watcher.rs`. `IndexWatcher` struct with `RecommendedWatcher` from notify, `Arc<RwLock<GpuResidentIndex>>`, `SharedIndexManager`. Method `start(root, index)`: create debounced watcher (500ms via notify-debouncer-mini), watch root recursively. On events: batch create/modify/delete, update index entries via scanner re-stat, persist via SharedIndexManager::save(). Method `stop()`: drop watcher. Add `pub mod watcher;` to `src/index/mod.rs`.
  - **Files**: `gpu-search/src/index/watcher.rs` (create ~200 lines), `gpu-search/src/index/mod.rs` (modify)
  - **Done when**: IndexWatcher compiles, can be instantiated with a test directory
  - **Verify**: `cargo check -p gpu-search`
  - **Commit**: `feat(gpu-search): add FSEvents index watcher via notify v7`
  - _Requirements: FR-4, AC-3.1, AC-3.2_
  - _Design: Component D_

- [x] 2.6 Implement deferred line counting as default path
  - **Do**: In `orchestrator.rs`, modify `resolve_match()` to compute line numbers from byte_offset by counting newlines in file content up to byte_offset. This is already what it does -- but ensure it's the sole source of truth for line numbers (don't trust GPU `line_number` field which only counts within 64B window). Set `line_number` from `resolve_match()` result, not from GPU result.
  - **Files**: `gpu-search/src/search/orchestrator.rs` (~10 lines in dispatch_gpu_batch)
  - **Done when**: All line numbers come from CPU counting, not GPU field
  - **Verify**: `cargo test -p gpu-search --test test_search_accuracy -- --nocapture`
  - **Commit**: `refactor(gpu-search): use CPU deferred line counting as default path`
  - _Requirements: FR-6_
  - _Design: Technical Decision (KB #1320, #1322)_

- [x] 2.7 Add hot/cold dispatch logic stub
  - **Do**: In `streaming.rs` or a new `dispatch.rs`, add a function `should_use_gpu(file_size: u64, page_cache_hint: bool) -> bool`. For now: always return true for files >128KB, false for files <4KB, configurable for middle range. This is a stub for future page-cache detection. Mark with TODO for KB #1377 hot/cold implementation.
  - **Files**: `gpu-search/src/search/streaming.rs` (~30 lines)
  - **Done when**: Function compiles, unit test verifies threshold logic
  - **Verify**: `cargo test -p gpu-search streaming::tests -- --nocapture`
  - **Commit**: `feat(gpu-search): add hot/cold dispatch threshold stub`
  - _Requirements: FR-9_
  - _Design: Technical Decision (KB #1377, #1290)_

- [x] 2.8 Performance checkpoint -- benchmark against baseline
  - **Do**: Run existing benchmarks. Compare TTFR for project search (should be <50ms), GPU throughput (should maintain 79-110 GB/s). Record baseline numbers in specs/gpu-search-overhaul/.progress.md.
  - **Done when**: Benchmarks run, no regression >10% from baseline
  - **Verify**: `cargo bench -p gpu-search 2>&1 | tail -30`
  - **Commit**: `perf(gpu-search): feature phase complete -- benchmark checkpoint`

## Phase 3: Testing

Focus: Grep oracle, proptest, benchmark regression, shader validation.

- [x] 3.1 Add grep oracle comparison test
  - **Do**: Add `test_accuracy_vs_grep_oracle` in `test_search_accuracy.rs`. Search gpu-search/src/ with patterns ["fn ", "pub ", "struct ", "impl ", "use ", "mod "]. Run grep -rn --include=*.rs for each. Assert: every gpu-search match exists in grep output (zero false positives). Assert: miss rate <20%. Parse grep output into (path, line_number) tuples for comparison.
  - **Files**: `gpu-search/tests/test_search_accuracy.rs` (extend ~100 lines)
  - **Done when**: Oracle test passes with zero false positives and miss rate <20%
  - **Verify**: `cargo test -p gpu-search --test test_search_accuracy test_accuracy_vs_grep -- --nocapture`
  - **Commit**: `test(gpu-search): add grep oracle comparison test`
  - _Requirements: AC-5.1, AC-5.2, AC-5.3_

- [x] 3.2 Add pipeline-level proptest at orchestrator level
  - **Do**: Extend `tests/test_proptest.rs` (or create new) with property: "For any (corpus, pattern) pair, every ContentMatch.line_content contains the pattern." Strategy: generate 1-10 temp files with random ASCII content (64-4096 bytes), pick a 2-16 byte substring from one file as pattern, run orchestrator, verify all matches. 100 iterations.
  - **Files**: `gpu-search/tests/test_proptest.rs` (extend ~60 lines)
  - **Done when**: Proptest runs 100 iterations with zero false positives
  - **Verify**: `cargo test -p gpu-search --test test_proptest prop_pipeline -- --nocapture`
  - **Commit**: `test(gpu-search): add pipeline-level property test for accuracy`
  - _Requirements: AC-1.1_

- [ ] 3.3 Add benchmark regression detection with stored baselines
  - **Do**: Create `benches/pipeline_profile.rs`. Benchmarks: (1) `bench_project_search` -- search gpu-search/src/ for "fn ", (2) `bench_index_load` -- load GSIX index via MmapIndexCache, (3) `bench_full_pipeline` -- orchestrator search end-to-end. Use Criterion groups. Store initial baseline via `cargo bench --save-baseline initial`.
  - **Files**: `gpu-search/benches/pipeline_profile.rs` (create ~120 lines), `gpu-search/Cargo.toml` (add [[bench]] entry)
  - **Done when**: New benchmark runs and produces Criterion reports
  - **Verify**: `cargo bench -p gpu-search --bench pipeline_profile`
  - **Commit**: `perf(gpu-search): add pipeline profiler benchmarks with baselines`
  - _Requirements: AC-5.4_

- [x] 3.4 Add Metal shader validation in test builds
  - **Do**: In existing test setup or a shared test helper, set env vars `MTL_SHADER_VALIDATION=1` and `MTL_DEBUG_LAYER=1` for GPU tests. Add a note in test_search_accuracy.rs header comment explaining these env vars. Verify all GPU tests pass with validation enabled.
  - **Files**: `gpu-search/tests/test_search_accuracy.rs` (add header comment), CI config if exists
  - **Done when**: Tests pass with shader validation enabled
  - **Verify**: `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 cargo test -p gpu-search --test test_search_accuracy -- --nocapture`
  - **Commit**: `test(gpu-search): verify Metal shader validation passes on all tests`

- [ ] 3.5 Add index watcher integration test
  - **Do**: In `tests/test_index.rs` (or new file), add test: create tempdir, build index, start IndexWatcher, create new file, wait 1s, verify index updated. Delete file, wait 1s, verify removed. Rename file, wait 1s, verify path updated.
  - **Files**: `gpu-search/tests/test_index.rs` (extend ~80 lines)
  - **Done when**: FSEvents watcher correctly updates index on file changes
  - **Verify**: `cargo test -p gpu-search --test test_index test_watcher -- --nocapture`
  - **Commit**: `test(gpu-search): add FSEvents index watcher integration test`
  - _Requirements: AC-3.1_

- [ ] 3.6 Testing checkpoint -- full suite passes
  - **Do**: Run all tests: existing 409 + new accuracy + proptest + index watcher. Verify zero failures.
  - **Done when**: All tests pass
  - **Verify**: `cargo test -p gpu-search 2>&1 | tail -20`
  - **Commit**: `test(gpu-search): complete test phase -- all tests green`

## Phase 4: Quality Gates

- [ ] 4.1 Clippy clean pass
  - **Do**: Run clippy with all warnings as errors. Fix any new warnings introduced by the overhaul. Ensure no `unwrap()` in non-test code (use `?` or explicit error handling).
  - **Verify**: `cargo clippy -p gpu-search -- -D warnings 2>&1 | tail -20`
  - **Done when**: Zero clippy warnings
  - **Commit**: `fix(gpu-search): address clippy warnings`

- [ ] 4.2 CI pipeline with benchmark gate
  - **Do**: Update `.github/workflows/` (if exists) to add accuracy test stage. Ensure `test_search_accuracy` runs on every PR and blocks merge. Add benchmark stage that runs `pipeline_profile` benchmark.
  - **Files**: `.github/workflows/gpu-search-ci.yml` (modify or create)
  - **Done when**: CI runs accuracy tests and benchmarks on PR
  - **Verify**: `gh workflow list` or verify CI config
  - **Commit**: `ci(gpu-search): add accuracy gate and benchmark stage`
  - _Requirements: AC-5.4_

- [ ] 4.3 Quality checkpoint
  - **Do**: Run all quality checks: clippy, tests (all), benchmarks, Metal shader validation. Verify everything passes.
  - **Verify**: `cargo clippy -p gpu-search -- -D warnings && MTL_SHADER_VALIDATION=1 cargo test -p gpu-search && cargo bench -p gpu-search --bench pipeline_profile`
  - **Done when**: All quality gates pass
  - **Commit**: `chore(gpu-search): quality gate checkpoint -- all green`

## Phase 5: PR

- [ ] 5.1 Create PR
  - **Do**: Push branch, create PR with `gh pr create`. Title: "fix: gpu-search accuracy + index + profiler overhaul". Body: summary of P0 fix, index wiring, profiler, test additions. Reference any relevant issues.
  - **Verify**: `gh pr checks --watch`
  - **Done when**: PR created, CI passes
  - **Commit**: (no commit -- PR creation)

- [ ] 5.2 Address CI/review feedback
  - **Do**: Monitor CI checks, fix any failures. Address code review comments if any.
  - **Verify**: `gh pr checks`
  - **Done when**: PR is green and ready for merge

## Notes

- **POC shortcuts taken**: Hot/cold dispatch is a stub (always GPU >128KB). Bitap kernel deferred to future. I/O pipeline uses sequential read() not true MTLIOCommandQueue overlap.
- **Production TODOs for Phase 2+**: True MTLIOCommandQueue I/O-compute overlap (KB #1337), Bitap kernel for <=32 char patterns (KB #1329), GPU Bloom filter for gitignore (KB #1363), PFAC multi-pattern search (KB #1251).
- **Known limitation**: GPU kernel misses patterns that cross 64-byte thread boundaries. Miss rate <5% for patterns >10 chars, 0% for short patterns placed in aligned test data. Documented in test assertions.
