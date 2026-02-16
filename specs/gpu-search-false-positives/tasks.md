---
spec: gpu-search-false-positives
phase: tasks
total_tasks: 16
created: 2026-02-15
generated: auto
---

# Tasks: gpu-search-false-positives

## Phase 1: Make It Work (POC) -- Reproduce & Instrument

Focus: Prove the bug exists programmatically. Create test harness, add instrumentation, reproduce false positives.

- [x] 1.1 Create integration test harness for false positive detection
  - **Do**: Create `tests/test_false_positives.rs` with GPU init boilerplate and helper functions. Create temp files with unique content per file (e.g., file_0 contains "UNIQUE_ALPHA_0", file_1 contains "UNIQUE_BETA_1", etc.). Write a test that searches for "UNIQUE_ALPHA" and verifies ONLY file_0 matches. Then immediately search for "UNIQUE_BETA" and verify ONLY file_1 matches. Use `StreamingSearchEngine` directly (not orchestrator) to isolate the GPU pipeline.
  - **Files**: `gpu-search/tests/test_false_positives.rs`
  - **Done when**: Test file compiles and runs, exercises `StreamingSearchEngine::search_files()` with multiple sequential queries against temp files
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_false_positives -- --nocapture 2>&1 | head -50`
  - **Commit**: `test(search): add false positive detection test harness`
  - _Requirements: FR-6, AC-4.1_
  - _Design: Component C_

- [x] 1.2 Add buffer state inspection to ContentSearchEngine
  - **Do**: Add `#[cfg(test)]` public methods to `ContentSearchEngine` in content.rs: `pub fn inspect_metadata_buffer(&self) -> Vec<u8>` (returns raw bytes of metadata_buffer up to max_chunks * sizeof(ChunkMetadata)), `pub fn inspect_match_count(&self) -> u32` (reads match_count_buffer value), and `pub fn max_chunks(&self) -> usize` (returns max_chunks). These enable tests to verify buffer state after reset().
  - **Files**: `gpu-search/src/search/content.rs`
  - **Done when**: `#[cfg(test)]` methods compile and are accessible from integration tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib -- content::tests --nocapture 2>&1 | tail -5`
  - **Commit**: `feat(search): add test-only buffer inspection to ContentSearchEngine`
  - _Requirements: FR-1, AC-4.2_
  - _Design: Component A_

- [x] 1.3 Write stale buffer detection test
  - **Do**: In `test_false_positives.rs`, add a test that: (1) creates a `ContentSearchEngine` directly, (2) loads content for 10 files with pattern "AAAA", (3) searches for "AAAA" -- confirm matches, (4) calls `reset()`, (5) loads content for 2 files with pattern "BBBB" (fewer files than before), (6) inspects `metadata_buffer` bytes at positions for chunks 2-9 -- verify they still contain stale data (non-zero `file_index` or `chunk_length`), (7) searches for "AAAA" -- if matches > 0, the stale buffer caused a false positive. This proves the bug.
  - **Files**: `gpu-search/tests/test_false_positives.rs`
  - **Done when**: Test demonstrates stale buffer data exists after `reset()` and/or produces false positives
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_false_positives test_stale_buffer_after_reset -- --nocapture 2>&1`
  - **Commit**: `test(search): prove stale GPU buffer data after reset`
  - _Requirements: FR-1, AC-4.2_
  - _Design: Component A_

- [x] 1.4 Write rapid query change simulation test
  - **Do**: In `test_false_positives.rs`, add a test that simulates rapid query changes: create 50 temp files where files 0-9 contain "kolbey", files 10-19 contain "patrick", files 20-49 contain generic filler. Then execute this sequence on a single `StreamingSearchEngine`: (1) search "patrick" -> collect results, (2) search "kolbey" -> collect results, (3) search "patrick" again -> collect results. For each search, validate every result against CPU reference (`cpu_streaming_search`). Assert zero false positives across all three queries. Check that result count is deterministic (search 1 == search 3).
  - **Files**: `gpu-search/tests/test_false_positives.rs`
  - **Done when**: Test exercises rapid sequential queries and validates each against CPU reference
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_false_positives test_rapid_query_change -- --nocapture 2>&1`
  - **Commit**: `test(search): add rapid query change simulation`
  - _Requirements: FR-3, AC-2.1, AC-4.1_
  - _Design: Component C_

- [x] 1.5 POC Checkpoint -- reproduce false positives
  - **Do**: Run all test_false_positives tests. If stale buffer test (1.3) shows false positives, document the exact reproduction. If it doesn't reproduce at the ContentSearchEngine level, try at the StreamingSearchEngine level with many sub-batches (>200 files per query to trigger sub-batching). Document findings in `.progress.md`.
  - **Files**: `specs/gpu-search-false-positives/.progress.md`
  - **Done when**: False positive reproduction is documented OR conclusive evidence that the stale buffer hypothesis is incorrect
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_false_positives -- --nocapture 2>&1`
  - **Commit**: `docs(spec): POC checkpoint - false positive reproduction status`
  - _Requirements: FR-1_

## Phase 2: Fix the Root Cause

- [x] 2.1 Fix ContentSearchEngine::reset() to zero GPU buffers
  - **Do**: Modify `reset()` in content.rs to zero `metadata_buffer` (up to `max_chunks * sizeof(ChunkMetadata)` = max_chunks * 24 bytes) and zero `match_count_buffer` (4 bytes). Use `unsafe { std::ptr::write_bytes(ptr, 0, len) }` matching existing patterns. Also zero the `matches_buffer` (up to `MAX_MATCHES * sizeof(GpuMatchResult)` = 10000 * 32 = 320KB) for defense-in-depth. Do NOT zero `chunks_buffer` (up to 400MB) -- it's already zero-padded per-chunk in `load_content()`.
  - **Files**: `gpu-search/src/search/content.rs`
  - **Done when**: `reset()` zeros metadata_buffer, match_count_buffer, and matches_buffer. All 556 existing tests still pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib -- --test-threads=1 2>&1 | tail -5`
  - **Commit**: `fix(search): zero GPU metadata and match buffers in reset()`
  - _Requirements: FR-1_
  - _Design: Component A_

- [x] 2.2 Verify stale buffer test now passes
  - **Do**: Re-run the stale buffer detection test from 1.3. After the fix, inspecting metadata_buffer after reset() should show all zeros for chunks beyond current_chunk_count. The false positive test should now report 0 false positives. If any test still fails, investigate further (byte_offset chain or kernel bounds).
  - **Files**: `gpu-search/tests/test_false_positives.rs` (may need assertion adjustments)
  - **Done when**: Stale buffer test passes, rapid query change test reports 0 false positives
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_false_positives -- --nocapture 2>&1`
  - **Commit**: `fix(search): verify stale buffer fix eliminates false positives`
  - _Requirements: FR-1, AC-1.4_
  - _Design: Component A_

- [x] 2.3 Audit and fix byte_offset calculation chain
  - **Do**: Trace byte_offset from GPU kernel through collect_results (content.rs:408), through streaming.rs:410 file_byte_offset calculation, through resolve_match (orchestrator.rs:1315). Add a debug assertion in resolve_match that the pattern is found at exactly `byte_offset % line_len` within the line (not just anywhere in the line). If the assertion fails, fix the byte_offset arithmetic. Key formula: GPU `byte_offset = chunk_index * 4096 + context_start + column`. Streaming.rs `file_byte_offset = byte_offset - start_chunk * 4096`. These should yield the file-relative byte position of the match.
  - **Files**: `gpu-search/src/search/content.rs`, `gpu-search/src/search/streaming.rs`, `gpu-search/src/search/orchestrator.rs`
  - **Done when**: byte_offset chain is validated end-to-end. Any arithmetic errors are fixed. Debug assertions pass for all test cases.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib -- --test-threads=1 2>&1 | tail -5 && cargo test --test test_false_positives -- --nocapture 2>&1 | tail -10`
  - **Commit**: `fix(search): validate byte_offset calculation chain`
  - _Requirements: FR-2, FR-5, AC-3.3_
  - _Design: Component B_

- [x] 2.4 Add match_range validation to resolve_match
  - **Do**: In `resolve_match()` (orchestrator.rs), after computing `match_col`, add an assertion that `line_content[match_col..match_col+pattern.len()]` case-insensitively equals the pattern. If this fails, the match_range is corrupt. Also ensure `match_range` (returned to ContentMatch) is `match_col..match_col+pattern.len()` which is already the case but add a debug_assert for CI coverage.
  - **Files**: `gpu-search/src/search/orchestrator.rs`
  - **Done when**: resolve_match validates match_range content before returning. All tests pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --lib -- --test-threads=1 2>&1 | tail -5`
  - **Commit**: `fix(search): add match_range content validation in resolve_match`
  - _Requirements: FR-4, FR-5, AC-3.1, AC-3.2_
  - _Design: Component B_

## Phase 3: Comprehensive Test Suite

- [x] 3.1 Add match_range accuracy tests
  - **Do**: In `test_false_positives.rs`, add tests that create files with known patterns at known positions, search via `SearchOrchestrator::search()` (blocking path), and validate every `ContentMatch.match_range` against the expected position. Use `cpu_verify_matches` from verify.rs as ground truth. Test cases: (a) single match per file, (b) multiple matches per file, (c) match at start of line, (d) match at end of line, (e) match spanning near 64-byte thread boundary.
  - **Files**: `gpu-search/tests/test_false_positives.rs`
  - **Done when**: 5+ match_range accuracy test cases pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_false_positives match_range -- --nocapture 2>&1`
  - **Commit**: `test(search): add match_range accuracy validation tests`
  - _Requirements: FR-5, AC-3.1, AC-3.2, AC-4.3_

- [x] 3.2 Add concurrent search cancellation tests
  - **Do**: In `test_false_positives.rs`, add a test that simulates search cancellation: (1) create a large set of temp files (500+), (2) start search on a background thread via `SearchOrchestrator::search_streaming()`, (3) after 50ms, cancel the session and start a new search with a different pattern, (4) verify the new search returns correct results (no contamination from the cancelled search). Use `SearchSession`, `CancellationHandle`, `SearchGeneration` from cancel.rs.
  - **Files**: `gpu-search/tests/test_false_positives.rs`
  - **Done when**: Cancellation test passes, new search results are correct
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_false_positives test_search_cancellation -- --nocapture 2>&1`
  - **Commit**: `test(search): add concurrent search cancellation tests`
  - _Requirements: FR-3, AC-2.4, AC-4.4_

- [x] 3.3 Add large file chunk boundary tests
  - **Do**: In `test_false_positives.rs`, add tests for files that span multiple 4KB chunks: (a) create a file exactly 8192 bytes with a pattern at byte 4090 (spanning chunk boundary), (b) create a file of 16384 bytes with patterns at bytes 0, 4095, 4096, 8191, 8192 (chunk boundaries), (c) verify match count against CPU reference, (d) verify byte_offsets are file-relative (not global). Use `ContentSearchEngine` directly for precise control.
  - **Files**: `gpu-search/tests/test_false_positives.rs`
  - **Done when**: Chunk boundary tests pass, matches at boundaries are correctly detected or gracefully missed (documented boundary limitation)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_false_positives chunk_boundary -- --nocapture 2>&1`
  - **Commit**: `test(search): add chunk boundary tests for multi-chunk files`
  - _Requirements: FR-2, AC-4.5_

- [x] 3.4 Add deterministic repeat search test
  - **Do**: In `test_false_positives.rs`, add a test that runs the same search 10 times on the same file set and asserts identical results each time. This catches any non-determinism from stale buffers, race conditions, or atomic ordering issues. Compare full result vectors (file paths, line numbers, match_ranges) not just counts.
  - **Files**: `gpu-search/tests/test_false_positives.rs`
  - **Done when**: 10 identical searches produce identical results
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_false_positives test_deterministic_repeat -- --nocapture 2>&1`
  - **Commit**: `test(search): add deterministic repeat search test`
  - _Requirements: AC-1.2, AC-4.1_

- [x] 3.5 Add CPU verification integration test with GPU_SEARCH_VERIFY=full
  - **Do**: In `test_false_positives.rs`, add a test that sets `GPU_SEARCH_VERIFY=full` (env var), then runs a search via `SearchOrchestrator::search()` against a directory with known content. The verify layer in orchestrator.rs will panic on any false positives (line 300-305). Test should complete without panic. Also test with intentionally corrupted byte_offsets to verify the panic fires.
  - **Files**: `gpu-search/tests/test_false_positives.rs`
  - **Done when**: Test passes with VERIFY=full (no false positives) and correctly panics when false positives are injected
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_false_positives test_verify_full_mode -- --nocapture 2>&1`
  - **Commit**: `test(search): add GPU_SEARCH_VERIFY=full integration test`
  - _Requirements: AC-1.4, AC-4.6_

## Phase 4: Quality Gates

- [x] 4.1 Run full test suite and fix any regressions
  - **Do**: Run `cargo test` (all unit + integration tests) and `cargo clippy`. Fix any failures or warnings introduced by the changes. Ensure all 556 original tests still pass plus the new false positive tests.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -10 && cargo clippy 2>&1 | tail -10`
  - **Done when**: All tests pass, clippy clean
  - **Commit**: `fix(search): address lint/type issues` (if needed)

- [x] 4.2 Verify fix with manual app test
  - **Do**: Build and run the gpu-search app. Search for "kolbey" -- should return 0 matches (or correct matches if the pattern exists). Search for "fn " -- should return consistent results on repeat. Type rapidly changing patterns and verify no false positives appear. Document manual verification in `.progress.md`.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo build --release 2>&1 | tail -5`
  - **Done when**: App builds, manual testing confirms no false positives
  - **Commit**: `docs(spec): manual verification of false positive fix`

- [x] 4.3 Create PR and verify CI
  - **Do**: Push branch, create PR with `gh pr create`. PR description should summarize the false positive root cause, the fix (buffer zeroing), and the test suite added. Use `gh pr checks --watch` to verify CI passes.
  - **Verify**: `gh pr checks --watch`
  - **Done when**: PR created, CI green
  - **Commit**: N/A (PR creation, no commit)

## Notes

- **POC shortcuts taken**: None significant -- this is a bug fix, not a feature POC
- **Production TODOs**:
  - Consider zeroing `chunks_buffer` selectively (only the region beyond current_chunk_count) for maximum defense-in-depth
  - Consider adding `GPU_SEARCH_VERIFY=full` as a CI-only flag (not default in production due to perf overhead)
  - The 64-byte thread boundary miss issue (matches spanning boundaries) is a known kernel limitation and is NOT addressed in this spec
