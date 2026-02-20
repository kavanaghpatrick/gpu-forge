---
id: testing-reliability.BREAKDOWN
module: testing-reliability
priority: 5
status: pending
version: 1
origin: foreman-spec
dependsOn: [gsix-v2-format.BREAKDOWN, mmap-gpu-pipeline.BREAKDOWN, fsevents-watcher.BREAKDOWN, incremental-updates.BREAKDOWN, global-root-index.BREAKDOWN]
tags: [persistent-index, gpu-search, testing, quality]
---
# Testing and Reliability — BREAKDOWN

## Context
The persistent index feature introduces ~146 new tests across all modules. This module covers cross-cutting test infrastructure, property-based tests for format fuzzing, benchmark suites for performance gates, crash safety validation, graceful degradation tests, and edge case coverage that does not belong to a single module. It also ensures all 404 existing tests continue to pass without regression.

References: QA.md (all sections), PM.md Section 3 (Success Metrics), TECH.md Appendix D (Risk Assessment).

## Tasks
1. **testing-reliability.synthetic-entry-generator** -- Implement `generate_synthetic_entries(count: usize) -> Vec<GpuPathEntry>` helper in a shared test utilities module. Generate realistic paths with varied depths, extensions, sizes, and mtimes. Used by scale tests and benchmarks across all modules. (est: 1h)

2. **testing-reliability.proptest-format-roundtrip** -- Write property tests in `tests/test_index_proptest.rs`: `prop_save_load_roundtrip` (random entries survive save/load), `prop_save_mmap_roundtrip` (random entries survive save/mmap), `prop_entry_size_invariant` (always 256B), `prop_path_len_bounded` (always <=224). Run at 5000 iterations. (est: 2h)

3. **testing-reliability.proptest-integrity** -- Write property tests: `prop_file_size_matches_formula` (file size == header + entry_count * 256), `prop_header_fields_consistent` (magic/version/entry_count correct), `prop_cache_key_deterministic` (same path -> same hash), `prop_cache_key_no_collisions_sample` (1000 distinct paths -> 1000 distinct hashes). (est: 2h)

4. **testing-reliability.proptest-binary-fuzzing** -- Write fuzz property tests: `prop_corrupt_bytes_no_panic` (flip random bytes in valid index, load returns Err), `prop_truncated_file_no_panic` (truncate at random offset), `prop_random_bytes_no_panic` (64-8192 random bytes), `prop_extended_file_no_panic` (append random bytes to valid index). All must never panic. Run at 5000 iterations. (est: 3h)

5. **testing-reliability.bench-mmap-load** -- Create `benches/index_load.rs` with criterion benchmarks: `bench_mmap_load_100k` (<1ms target), `bench_mmap_load_1m` (<5ms target), `bench_mmap_first_entry_access` (<0.1ms), `bench_save_index_100k` (<100ms), `bench_save_index_1m` (<1s). (est: 2h)

6. **testing-reliability.bench-incremental** -- Add criterion benchmarks: `bench_rescan_10k_dir` (<500ms), incremental update latency measurement (filesystem change to index updated). Establish baseline for current full-rescan architecture vs future incremental path. (est: 2h)

7. **testing-reliability.scale-1m-tests** -- Write scale tests in `tests/test_index_scale.rs` (marked `#[ignore]` for CI separate runs): 1M entry build, 1M entry save/load roundtrip, 1M entry mmap load, 1M entry GPU buffer upload, 1M entry find-by-name. All must complete without OOM. (est: 3h)

8. **testing-reliability.scale-memory-pressure** -- Write tests: 3 concurrent mmap'd indexes all load correctly, drop releases mapping, concurrent large indexes have no cross-contamination. Verify mmap pages are reclaimable under memory pressure. (est: 2h)

9. **testing-reliability.crash-safety-tests** -- Write tests in `tests/test_index_persistence.rs`: atomic rename leaves no partial `.idx` file, `.idx.tmp` left on write error does not corrupt next load, concurrent save produces one valid index (no interleaving), rename atomicity (reader sees old or new, never a mix). (~4 tests) (est: 2h)

10. **testing-reliability.corrupt-recovery-tests** -- Write tests: corrupt index triggers rebuild, stale index triggers rebuild, delete-and-rebuild works cleanly, corrupt random bytes -> full pipeline recovery (detect + scan + save + return valid index). (~4 tests) (est: 2h)

11. **testing-reliability.graceful-degradation-tests** -- Write tests: missing cache dir created automatically, unwritable cache dir -> save fails but search works via walk, mmap failure -> fallback to copy path, no Metal device -> all operations succeed without GPU buffer, watcher thread panic absorbed, watcher channel disconnect -> clean exit. (~6 tests) (est: 3h)

12. **testing-reliability.edge-case-symlinks** -- Write tests: symlink to file (IS_SYMLINK flag), symlink cycle (no infinite loop), broken symlink (silently skipped), symlink to excluded dir. (~4 tests) (est: 1h)

13. **testing-reliability.edge-case-paths** -- Write tests: path exactly 224 bytes (indexed), path 225 bytes (skipped), path 1024 bytes (skipped no panic), path near-limit roundtrip fidelity, unicode multibyte CJK, unicode emoji, unicode normalization (NFD vs NFC), space in filename, newline in filename, null byte path, backslash in name. (~11 tests) (est: 2h)

14. **testing-reliability.edge-case-permissions** -- Write tests: chmod 000 directory skipped, chmod 000 file skipped, mixed permissions (only readable indexed), root unreadable -> empty index, disk full preserves old index, write fail no partial, rename fail cleanup. (~7 tests) (est: 2h)

15. **testing-reliability.regression-walk-fallback** -- Write regression tests: `walk_and_filter()` still works without index, blocking `search()` API works without index, index deleted mid-search -> search completes via walk. (~3 tests) (est: 1h)

16. **testing-reliability.regression-search-equivalence** -- Write regression tests: indexed vs unindexed content results match, filename match equivalence, file type filter equivalence, hidden file handling consistent between paths. (~4 tests) (est: 2h)

17. **testing-reliability.gpu-pipeline-equivalence** -- Write GPU pipeline tests in `tests/test_index_gpu_pipeline.rs`: indexed GPU search matches CPU verification, no false positives (pattern at reported offset), no false negatives for non-boundary patterns, proptest random directory + random pattern consistency. (~4 tests) (est: 2h)

18. **testing-reliability.ci-integration** -- Update GitHub Actions workflow to run: index unit tests, index integration tests, scale tests (separate job, `--ignored`), proptest with `PROPTEST_CASES=5000`, GPU pipeline tests with `MTL_SHADER_VALIDATION=1`, benchmark regression check. (est: 1h)

## Dependencies
- Requires: [All previous modules must be implemented so tests have code to exercise]
- Enables: [integration (test suite provides confidence for final assembly)]

## Acceptance Criteria
1. All 404 existing tests pass with zero regressions
2. ~146 new tests pass across unit, integration, property, and scale levels
3. Property tests pass at 5000 iterations with no panics on malformed input
4. Benchmark gates met: mmap load <1ms at 100K, save <1s at 1M
5. 1M entry scale tests pass without OOM
6. Crash safety: no partial `.idx` files, atomic rename preserves consistency
7. Graceful degradation: search works via walk fallback when index is unavailable/corrupt/error
8. Indexed search results match non-indexed results for identical queries
9. CI pipeline runs all test categories with appropriate configuration
10. No new `unsafe` without documented `// SAFETY:` comments; clippy clean with `-D warnings`

## Technical References
- PM: ai/tasks/spec/PM.md -- Section 3 (Success Metrics: performance and reliability targets)
- UX: ai/tasks/spec/UX.md -- Section 9 (Progressive Enhancement: fallback behavior), Section 7 (Error States)
- Tech: ai/tasks/spec/TECH.md -- Appendix D (Risk Assessment)
- QA: ai/tasks/spec/QA.md -- All sections (test pyramid, format tests, FSEvents tests, incremental tests, performance tests, scale tests, edge cases, regression tests, GPU pipeline tests, reliability tests, proptest, acceptance criteria)
