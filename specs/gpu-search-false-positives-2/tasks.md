---
spec: gpu-search-false-positives-2
phase: tasks
total_tasks: 15
created: 2026-02-16
generated: auto
---

# Tasks: gpu-search-false-positives-2

## Phase 1: Make It Work (POC)

Focus: Build orchestrator-level integration tests that reproduce false positives, then fix the root cause.

- [x] 1.1 Create orchestrator-level false positive test file with test infrastructure
  - **Do**: Create `gpu-search/tests/test_fp_orchestrator.rs` with helpers: (1) `create_test_directory()` that creates 20 temp files -- 5 containing "kolbey", 5 containing "Patrick Kavanagh", 10 containing generic filler. Each file ~4KB+ (above cold threshold). (2) `create_orchestrator()` helper that initializes GpuDevice + PsoCache + SearchOrchestrator. (3) `validate_content_matches()` helper that for each ContentMatch reads the file and asserts line_content contains the search pattern.
  - **Files**: `gpu-search/tests/test_fp_orchestrator.rs`
  - **Done when**: File compiles, helpers work, can create orchestrator and test directory
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_fp_orchestrator -- --nocapture 2>&1 | head -20`
  - **Commit**: `test(gpu-search): add orchestrator-level false positive test infrastructure`
  - _Requirements: FR-3_
  - _Design: Component A_

- [x] 1.2 Add basic false positive detection test
  - **Do**: Add test `test_orchestrator_no_false_positives` that: (1) creates test directory with known files, (2) searches for "kolbey" using `orchestrator.search()`, (3) for EVERY ContentMatch, reads the file and verifies `file_content.to_lowercase().contains("kolbey")`, (4) also validates `cm.line_content.to_lowercase().contains("kolbey")`, (5) asserts zero false positives. Also add test for "patrick" pattern with same validation.
  - **Files**: `gpu-search/tests/test_fp_orchestrator.rs`
  - **Done when**: Tests pass -- every returned ContentMatch genuinely contains the searched pattern
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_fp_orchestrator test_orchestrator_no_false_positives -- --nocapture`
  - **Commit**: `test(gpu-search): add orchestrator-level false positive detection tests`
  - _Requirements: FR-1, AC-1.1, AC-1.3_
  - _Design: Component A_

- [x] 1.3 Add match_range validation test
  - **Do**: Add test `test_match_range_integrity` that for every ContentMatch from an orchestrator search: (1) asserts `match_range.start < match_range.end`, (2) asserts `match_range.end <= line_content.len()`, (3) asserts `line_content[match_range.clone()].to_lowercase() == pattern.to_lowercase()`. Test with both "kolbey" and "patrick" patterns.
  - **Files**: `gpu-search/tests/test_fp_orchestrator.rs`
  - **Done when**: All match_range properties hold for every ContentMatch
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_fp_orchestrator test_match_range_integrity -- --nocapture`
  - **Commit**: `test(gpu-search): add match_range integrity validation`
  - _Requirements: FR-5, AC-3.1, AC-3.2, AC-3.3_
  - _Design: Component C_

- [x] 1.4 Add rapid query change simulation test (orchestrator level)
  - **Do**: Add test `test_rapid_query_change_orchestrator` that: (1) creates test directory, (2) creates orchestrator + crossbeam channels (cmd_tx/cmd_rx, update_tx/update_rx) mimicking app.rs orchestrator_thread pattern, (3) dispatches search for "patrick" via channel with SearchSession gen=1, (4) after 10ms, cancels and dispatches "kolbey" with gen=2, (5) after 10ms, cancels and dispatches "patrick" with gen=3, (6) collects all StampedUpdate messages, (7) filters to final generation only, (8) validates all surviving ContentMatches are genuine (file contains pattern). Use `GPU_SEARCH_VERIFY=full` env var.
  - **Files**: `gpu-search/tests/test_fp_orchestrator.rs`
  - **Done when**: Rapid cancel/restart produces zero false positives in final generation
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && GPU_SEARCH_VERIFY=full cargo test --test test_fp_orchestrator test_rapid_query_change -- --nocapture`
  - **Commit**: `test(gpu-search): add rapid query change simulation test`
  - _Requirements: FR-4, AC-2.1, AC-2.2, AC-2.3_
  - _Design: Component D_

- [x] 1.5 Add character-by-character typing simulation test
  - **Do**: Add test `test_typing_simulation` that: (1) creates test directory with "kolbey" files and "Patrick Kavanagh" files, (2) simulates typing "kolbey" character by character: dispatches searches for "k", "ko", "kol", "kolb", "kolbe", "kolbey" with 5ms gaps, cancelling previous each time, (3) waits for final search to complete, (4) validates ALL ContentMatches in final results: every matched file must actually contain "kolbey", (5) specifically asserts 0 matches from "Patrick Kavanagh" files.
  - **Files**: `gpu-search/tests/test_fp_orchestrator.rs`
  - **Done when**: Character-by-character typing produces correct final results with zero false positives
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && GPU_SEARCH_VERIFY=full cargo test --test test_fp_orchestrator test_typing_simulation -- --nocapture`
  - **Commit**: `test(gpu-search): add character-by-character typing simulation`
  - _Requirements: FR-4, AC-2.1_
  - _Design: Component D_

- [x] 1.6 POC Checkpoint: Diagnose root cause from test instrumentation
  - **Do**: Run all orchestrator tests with `GPU_SEARCH_VERIFY=full` and `RUST_LOG=debug`. Analyze output to identify: (1) Are false positives originating from GPU dispatch or resolve_match? (2) Are stale-generation results leaking through? (3) Is byte_offset mapping incorrect? Document findings in `.progress.md`. If tests all pass with zero false positives, the bug may only manifest with real filesystem searches from `/` -- document this and move to Phase 2 fixes.
  - **Done when**: Root cause identified or confirmed that orchestrator-level tests pass (meaning bug is in app.rs client-side logic)
  - **Verify**: All tests pass, findings documented
  - **Commit**: `docs(spec): document false positive root cause analysis`
  - _Requirements: FR-1, FR-2_

## Phase 2: Fix Root Cause

Based on diagnostic findings, fix the actual false positive sources.

- [x] 2.1 Harden resolve_match byte_offset validation
  - **Do**: In `resolve_match()` (orchestrator.rs:1488), after finding the line at byte_offset and searching for the pattern: (1) compute `expected_col = byte_offset - cumulative`, (2) if `match_col` from find() differs from `expected_col` by more than `pattern.len()`, log a warning with file path, byte_offset, expected_col, match_col, (3) verify the content at byte_offset actually starts with the pattern bytes (case-insensitive) by checking `content[byte_offset..].starts_with(pattern_bytes)`, (4) if the content at byte_offset does NOT match the pattern, this is a GPU false positive -- still accept the find() result (the pattern IS in the line) but log the discrepancy. This maintains correctness while providing diagnostic data.
  - **Files**: `gpu-search/src/search/orchestrator.rs`
  - **Done when**: resolve_match logs byte_offset discrepancies without breaking existing functionality
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test -- --nocapture 2>&1 | grep -c "byte_offset discrepancy" || echo "no discrepancies"`
  - **Commit**: `fix(gpu-search): add byte_offset validation in resolve_match`
  - _Requirements: FR-2, FR-6_
  - _Design: Component B_

- [x] 2.2 Add content-level false positive rejection in dispatch_gpu_batch_profiled
  - **Do**: In `dispatch_gpu_batch_profiled()` (orchestrator.rs:1190-1207), after `resolve_match()` returns a ContentMatch, add a FINAL validation gate: read the file at `m.file_path`, verify the file content (not just the resolved line) contains the pattern. If the FILE does not contain the pattern at all, reject the match entirely with a warning log. This catches the case where byte_offset maps to the wrong file entirely.
  - **Files**: `gpu-search/src/search/orchestrator.rs`
  - **Done when**: False positives from wrong-file byte_offset mapping are rejected
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && GPU_SEARCH_VERIFY=full cargo test --test test_fp_orchestrator -- --nocapture`
  - **Commit**: `fix(gpu-search): add file-level content validation in GPU batch dispatch`
  - _Requirements: FR-1, AC-1.3_
  - _Design: Component B_

- [x] 2.3 Fix client-side refinement filter match_range update
  - **Do**: In `dispatch_search()` (app.rs:372-376), the refinement filter updates match_range using `find()` on the lowercased line. This can produce incorrect match_range if the line contains the pattern at a different position than originally matched. Fix: (1) when updating match_range, also verify the extracted text `line_content[new_range.clone()]` case-insensitively matches the pattern, (2) if the pattern isn't found in the line at all, remove the match from content_matches (it's a false positive from refinement).
  - **Files**: `gpu-search/src/ui/app.rs`
  - **Done when**: Refinement filtering produces correct match_ranges and removes non-matching entries
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test -- --nocapture`
  - **Commit**: `fix(gpu-search): harden client-side refinement filter match_range`
  - _Requirements: FR-5, AC-3.1_
  - _Design: Component C_

- [x] 2.4 Add drain guard to prevent stale results in poll_updates
  - **Do**: In `dispatch_search()` (app.rs:430), the current code drains the update_rx channel before sending the new search command. But there's a race: after drain and before the new search produces output, stale results from the old search (already in-flight on the orchestrator thread) may arrive. Fix: after drain, also assert that `pending_result_clear` is set, ensuring the first valid update for the new generation clears old results. Verify the generation guard comparison in `poll_updates()` uses strict equality (already does: `stamped.generation != self.search_generation.current_id()`).
  - **Files**: `gpu-search/src/ui/app.rs`
  - **Done when**: No stale results can leak through poll_updates after a new search is dispatched
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test -- --nocapture`
  - **Commit**: `fix(gpu-search): harden stale result drain guard in dispatch_search`
  - _Requirements: FR-1, AC-2.2_

## Phase 3: Testing

- [x] 3.1 Add comprehensive regression test: kolbey-in-kavanagh scenario
  - **Do**: Add test `test_kolbey_kavanagh_regression` to `test_fp_orchestrator.rs` that exactly reproduces the reported bug: (1) create files containing "Patrick Kavanagh" text (names like `kavanagh_bio.txt`), (2) create files containing "kolbey" text, (3) create many filler files, (4) search for "kolbey", (5) assert ZERO matches from kavanagh files, (6) assert all matches are from kolbey files, (7) validate match_range for every result. Run with `GPU_SEARCH_VERIFY=full`.
  - **Files**: `gpu-search/tests/test_fp_orchestrator.rs`
  - **Done when**: Regression test passes, specifically validating the reported scenario
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && GPU_SEARCH_VERIFY=full cargo test --test test_fp_orchestrator test_kolbey_kavanagh_regression -- --nocapture`
  - **Commit**: `test(gpu-search): add kolbey-kavanagh false positive regression test`
  - _Requirements: FR-8, AC-1.3_

- [x] 3.2 Add stress test: 100 rapid queries with content validation
  - **Do**: Add test `test_100_rapid_queries_no_false_positives` to `test_fp_orchestrator.rs`: (1) create test directory with 50 files (25 with unique patterns, 25 filler), (2) run 100 sequential searches alternating between different patterns, (3) for each search result, validate every ContentMatch file actually contains the pattern, (4) track total false positives across all 100 queries, (5) assert total == 0.
  - **Files**: `gpu-search/tests/test_fp_orchestrator.rs`
  - **Done when**: 100 sequential queries produce zero false positives
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_fp_orchestrator test_100_rapid -- --nocapture`
  - **Commit**: `test(gpu-search): add 100-query stress test for false positives`
  - _Requirements: FR-8_

- [x] 3.3 Add match_range corruption stress test
  - **Do**: Add test `test_match_range_never_corrupted` to `test_fp_orchestrator.rs`: (1) create files with known patterns at known positions, (2) run 20 searches with different patterns (alternating short "fn", medium "kolbey", long "Patrick Kavanagh"), (3) for every ContentMatch, validate: match_range in bounds, extracted text matches pattern, match_range.start < match_range.end.
  - **Files**: `gpu-search/tests/test_fp_orchestrator.rs`
  - **Done when**: 20 varied searches produce zero match_range corruptions
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test --test test_fp_orchestrator test_match_range_never -- --nocapture`
  - **Commit**: `test(gpu-search): add match_range corruption stress test`
  - _Requirements: FR-5, AC-3.1_

## Phase 4: Quality Gates

- [x] 4.1 Run all existing tests to verify no regressions
  - **Do**: Run the complete test suite: unit tests, existing integration tests (test_false_positives, test_stale_results, test_orchestrator_integration, test_gpu_cpu_consistency, test_stress), and the new test_fp_orchestrator tests. All must pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && cargo test 2>&1 | tail -5`
  - **Done when**: All tests pass, zero failures
  - **Commit**: `fix(gpu-search): address any lint/type issues` (if needed)

- [x] 4.2 Run with GPU_SEARCH_VERIFY=full to validate fix
  - **Do**: Run the key false positive tests with full CPU verification enabled: `GPU_SEARCH_VERIFY=full cargo test --test test_fp_orchestrator -- --nocapture`. The verify layer will panic on any false positive. Also run `GPU_SEARCH_VERIFY=full cargo test --test test_false_positives -- --nocapture` to confirm existing tests still pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-search && GPU_SEARCH_VERIFY=full cargo test --test test_fp_orchestrator --test test_false_positives -- --nocapture 2>&1 | tail -10`
  - **Done when**: Zero false positives in full verification mode
  - **Commit**: No commit needed (verification only)

- [ ] 4.3 Create PR and verify CI
  - **Do**: Push branch, create PR with gh CLI summarizing: (1) false positive root cause, (2) fix approach, (3) new test coverage. Target main branch.
  - **Verify**: `gh pr checks --watch` all green
  - **Done when**: PR created, CI green, ready for review
  - **Commit**: No commit (PR creation only)

## Notes

- **POC shortcuts**: Phase 1 tests may create temp directories each time (not reused across tests). Acceptable for correctness testing.
- **Production TODOs**: If the root cause is byte_offset mapping in streaming.rs, a deeper fix in the GPU pipeline may be needed in a follow-up.
- **Existing coverage**: test_false_positives.rs already has 6 tests at the StreamingSearchEngine level -- those pass, meaning the bug is above that layer.
- **Key files**: orchestrator.rs (resolve_match, dispatch_gpu_batch_profiled), app.rs (dispatch_search, poll_updates), verify.rs (cpu_verify_matches)
