---
spec: gpu-search-dedup
phase: tasks
total_tasks: 16
created: 2026-02-18T00:00:00Z
generated: auto
---

# Tasks: gpu-search-dedup

## Phase 1: Make It Work (POC)

Focus: Validate FNV-1a hashing + atomic CAS dedup works end-to-end. Accept hardcoded 100K table. Skip error handling.

- [x] 1.1 Add dedup_kernel MSL source to shader.rs
  - **Do**: Append `dedup_kernel` to PATH_SEARCH_SHADER. Copy hashtable.metal kernel patterns (atomic_compare_exchange_weak_explicit). Implement FNV-1a hash loop (6 ops/byte). Scan backward/forward for path newline bounds. Linear probing (64 max probes).
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/shader.rs` (add ~40 lines before closing of PATH_SEARCH_SHADER)
  - **Done when**: dedup_kernel compiles without errors, accepts chunks_buffer, matches, hash_table, unique_flags, match_count, params
  - **Verify**: `cargo build --release` in gpu-search-ui succeeds, shader compilation returns no errors
  - **Commit**: `feat(engine): add dedup_kernel MSL compute shader`
  - _Requirements: FR-2, FR-3, FR-4_
  - _Design: Component DedupdKernel_

- [x] 1.2 Add hash table buffer + dedup pipeline to GpuContentSearch
  - **Do**: In `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`, add fields: `dedup_pipeline: ComputePipelineState`, `hash_table_buffer: Buffer`, `unique_flags_buffer: Buffer`. In `new_for_paths()`, allocate buffers (100K uint32 for hash table, 100K uint8 for flags). Compile dedup_kernel from path_library.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (struct GpuContentSearch, impl GpuContentSearch::new_for_paths)
  - **Done when**: GpuContentSearch has dedup_pipeline + buffers initialized, no compiler errors
  - **Verify**: `cargo build --release` succeeds, no undefined reference errors
  - **Commit**: `feat(engine): allocate hash table buffers and dedup pipeline`
  - _Requirements: FR-1_
  - _Design: Component GpuContentSearch Integration_

- [x] 1.3 Implement dedup() method to dispatch kernel
  - **Do**: Add `fn dedup(&self, match_count: u32) -> u64` to GpuContentSearch impl. Create command buffer + encoder. Set compute pipeline (dedup_pipeline). Bind buffers: chunks_buffer(0), path_matches_buffer(1), hash_table_buffer(2), match_count_buffer(3), unique_flags_buffer(4), params_buffer(5). Dispatch threads: total_threads = match_count, 256 per threadgroup. Encode, commit, wait_until_completed(). Return GPU time.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (new method dedup)
  - **Done when**: dedup() method compiles and can be called after path_search
  - **Verify**: `cargo build --release` succeeds
  - **Commit**: `feat(engine): implement dedup kernel dispatch method`
  - _Requirements: FR-3, FR-6, FR-8_
  - _Design: Component GpuContentSearch Integration_

- [x] 1.4 Integrate dedup into search_paths()
  - **Do**: In `search_paths()` method, after path_search_kernel dispatch + wait_until_completed (line 460), add: (1) Clear hash_table_buffer with 0xFF (unsafe memset). (2) Call dedup(match_count). (3) Read unique_flags_buffer. (4) Filter matches by unique_flags in result extraction loop. Accept all 50K unique paths in POC (no filtering, will add in Phase 2).
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (method search_paths, lines 462-532)
  - **Done when**: search_paths() calls dedup() and returns ContentMatch vector (POC: unfiltered)
  - **Verify**: No compiler errors, test POC with 50K path search
  - **Commit**: `feat(engine): integrate dedup into search_paths pipeline`
  - _Requirements: FR-1, FR-4, FR-7_
  - _Design: Data Flow_

- [x] 1.5 Add CPU filter in app.rs search_thread()
  - **Do**: In `/Users/patrickkavanagh/gpu-search-ui/src/app.rs` search_thread(), replace lines 341-356 (HashSet::insert loop) with new filter that uses unique_flags from GPU. Read unique_flags_buffer from engine, zip with raw_matches, filter by flag==1.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/app.rs` (search_thread, lines 341-356)
  - **Done when**: filter removes all duplicates (can verify with test data)
  - **Verify**: Manual test: search for path pattern that has duplicates, verify only unique results returned
  - **Commit**: `feat(app): replace CPU HashSet dedup with GPU-filtered results`
  - _Requirements: FR-4, AC-1.4_
  - _Design: Component CPU Filter_

- [x] 1.X POC Checkpoint
  - **Do**: Run full integration test. Search 50K paths with duplicates (e.g., symlinks). Verify results are deduplicated, no false negatives.
  - **Done when**: 50K path search returns unique paths only, <1.5s total time (path_search + dedup GPU + CPU filter)
  - **Verify**: Manual test with test_data/paths_50k.txt (generated from filesystem walk with duplicates)
  - **Commit**: `feat(engine): complete dedup POC`

## Phase 2: Refactoring

After POC validated, clean up and add error handling.

- [ ] 2.1 Extract dedup helper function + error handling
  - **Do**: Create `fn dedup_with_error(&self, match_count: u32) -> Result<u64, String>`. Add checks: hash_table_buffer null, match_count overflow, GPU allocation failure. Add FNV-1a hash test function (unit test for collision rate).
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`
  - **Done when**: dedup_with_error() handles error cases gracefully, logs to error callback
  - **Verify**: Unit tests pass: test_fnv1a_hash, test_dedup_collision_rate
  - **Commit**: `refactor(engine): add dedup error handling and validation`
  - _Design: Error Handling_

- [ ] 2.2 Add hash table pre-clearing optimization
  - **Do**: Move hash_table clear into dedup() dispatch (zero-copy via GPU kernel initial write). Use first threadgroup to atomic_store 0xFF across table in parallel (faster than CPU memset). Measure timing improvement.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`
  - **Done when**: hash table clears in <100µs (GPU kernel) vs CPU memset
  - **Verify**: Profiling shows clear_us < 100µs in SearchProfile
  - **Commit**: `refactor(engine): move hash table clear to GPU kernel`
  - _Design: Optimization Opportunities_

- [ ] 2.3 Extend SearchProfile with dedup metrics
  - **Do**: Add `dedup_us: u64`, `dedup_table_size: u32`, `dedup_collisions: u32` to SearchProfile. Update dedup() to return struct. Print in SearchProfile::print().
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs`
  - **Done when**: SearchProfile includes dedup timings and collision count
  - **Verify**: `cargo build --release` succeeds, profiling output shows dedup_us
  - **Commit**: `feat(engine): add dedup metrics to SearchProfile`
  - _Requirements: FR-8_

- [ ] 2.4 Validate dedup correctness with unit tests
  - **Do**: Add test in search.rs: `test_dedup_no_duplicates()` — load paths with known duplicates, verify all duplicates marked with unique_flags==0. Test FNV-1a collision rate at 50K scale.
  - **Files**: `/Users/patrickkavanagh/gpu-search-ui/src/engine/search.rs` (tests module)
  - **Done when**: All dedup tests pass (100% correctness on synthetic data)
  - **Verify**: `cargo test --lib engine::search` passes
  - **Commit**: `test(engine): add comprehensive dedup tests`
  - _Requirements: AC-1.5_

## Phase 3: Testing

Comprehensive testing with various match counts, edge cases.

- [ ] 3.1 Integration test: dedup with 10K matches
  - **Do**: Create test data with 10K unique paths + 5K duplicates (15K total). Load into GPU, verify 10K returned after dedup.
  - **Files**: `tests/test_dedup.rs` (new file)
  - **Done when**: Test passes: 10K unique paths correctly identified
  - **Verify**: `cargo test --test test_dedup` passes
  - **Commit**: `test: add integration test for dedup with 10K matches`

- [ ] 3.2 Integration test: dedup with 50K matches (peak load)
  - **Do**: Create test with 50K unique + 25K duplicates (75K total, table may overflow). Verify graceful handling: unique entries inserted, overflow entries marked duplicate.
  - **Files**: `tests/test_dedup.rs`
  - **Done when**: Test passes at 50K scale with <1.5ms GPU time
  - **Verify**: `cargo test --test test_dedup -- --ignored` passes
  - **Commit**: `test: add peak load test for dedup at 50K matches`

- [ ] 3.3 Benchmark: dedup latency vs match count
  - **Do**: Benchmark dedup GPU time at 1K, 5K, 10K, 25K, 50K matches. Verify linear scaling (should be O(n) for n matches, constant time per match ~20µs at 256 threads).
  - **Files**: `benches/dedup_bench.rs` (new file using Criterion)
  - **Done when**: Benchmark data shows <1ms for 50K matches, latency scales linearly
  - **Verify**: `cargo bench --bench dedup_bench` completes
  - **Commit**: `perf: add dedup latency benchmark`

- [ ] 3.4 Edge case test: empty matches (0 results)
  - **Do**: Test dedup with 0 matches (match_count==0). Should return immediately, no GPU work.
  - **Files**: `tests/test_dedup.rs`
  - **Done when**: Test passes with 0 GPU time
  - **Verify**: `cargo test --test test_dedup` passes
  - **Commit**: `test: add edge case for zero matches`

- [ ] 3.5 Edge case test: all duplicates
  - **Do**: Test dedup with 50K matches all pointing to same path. Verify first marked unique (1), rest marked duplicate (0).
  - **Files**: `tests/test_dedup.rs`
  - **Done when**: Test passes: exactly 1 unique, 49,999 duplicates
  - **Verify**: `cargo test --test test_dedup` passes
  - **Commit**: `test: add edge case for all duplicate paths`

## Phase 4: Quality Gates

- [ ] 4.1 Local quality check
  - **Do**: Run all checks locally: `cargo build --release`, `cargo test --all`, `cargo clippy -- -D warnings`, `cargo fmt --check`.
  - **Done when**: All commands pass with zero errors, zero warnings
  - **Verify**: All CI checks green
  - **Commit**: `fix(engine): address clippy lints and format issues` (if needed)

- [ ] 4.2 Code review + type safety
  - **Do**: Manual review of dedup_kernel MSL + Rust unsafe pointers. Verify: (1) No buffer overruns in path scanning (check bounds), (2) No use-after-free (hash_table_buffer lifetime), (3) Atomic CAS memory ordering correct (relaxed sufficient).
  - **Done when**: Code review complete, no safety issues found
  - **Verify**: Self-review checklist passes

- [ ] 4.3 Performance validation
  - **Do**: Measure end-to-end search time with dedup vs without. Target: 50K search <1s (path_search + dedup + CPU filter all <1ms GPU, rest is latency overhead).
  - **Done when**: 50K search shows >3500x speedup in dedup phase (3.6s CPU → <1ms GPU)
  - **Verify**: SearchProfile output shows dedup_us < 1000

- [ ] 4.4 Create PR and CI verification
  - **Do**: Push branch, create PR with `gh pr create`. Watch CI: type check, clippy, tests, benchmarks all pass. Squash commits before merge.
  - **Done when**: CI green, all checks pass, code ready for merge
  - **Verify**: `gh pr checks --watch` all green
  - **Commit**: (squashed) `feat(engine): GPU-side lock-free hash table dedup for path search`

## Notes

- **POC shortcuts taken**:
  - Accept hardcoded 100K hash table capacity (no dynamic sizing)
  - No fancy error recovery (duplicate all on table full)
  - FNV-1a collision rate not validated in POC (will add in Phase 2)
  - CPU filter is linear scan, not optimized

- **Production TODOs**:
  - Make hash table capacity configurable (based on max_results option)
  - Add GPU-side fallback to CPU HashSet if table fills
  - Implement SIMD 4-way parallel FNV-1a hashing (10-20% speedup)
  - Consider Bloom filter pre-pass if dedup rate is 90%+ (avoid hash table for mostly unique searches)
  - Profile on M3, M4, M4 Max variants to validate <1ms claim across hardware
