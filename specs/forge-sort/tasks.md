# Tasks: forge-sort

Extraction of exp17 Investigation W radix sort into `forge-sort` crate within metal-forge-compute workspace. Proven algorithm, 4 kernels, ~120 lines Rust dispatch logic. Pure packaging.

## Phase 1: Make It Work (POC)

Focus: Get the crate building, shaders compiling, sort running correctly at 16M.

- [ ] 1.1 Create forge-sort crate scaffold and workspace integration
  - **Do**:
    1. Add `"forge-sort"` to `metal-forge-compute/Cargo.toml` workspace members
    2. Create `metal-forge-compute/forge-sort/Cargo.toml` with deps: forge-primitives (path), objc2 0.6, objc2-metal 0.3, objc2-foundation 0.3, thiserror 2. Dev-deps: rand 0.8
    3. Create `metal-forge-compute/forge-sort/src/lib.rs` with minimal `pub struct GpuSorter;` placeholder
    4. Create `metal-forge-compute/forge-sort/shaders/` directory
  - **Files**:
    - `metal-forge-compute/Cargo.toml` (modify)
    - `metal-forge-compute/forge-sort/Cargo.toml` (create)
    - `metal-forge-compute/forge-sort/src/lib.rs` (create)
  - **Done when**: `cargo check -p forge-sort` passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check -p forge-sort`
  - **Commit**: `feat(forge-sort): scaffold crate and workspace integration`
  - _Requirements: AC-6.1, AC-6.2, AC-6.3_
  - _Design: File Structure, Cargo.toml_

- [ ] 1.2 Create build.rs for Metal shader compilation
  - **Do**:
    1. Create `metal-forge-compute/forge-sort/build.rs` based on `metal-gpu-experiments/build.rs`
    2. Key changes from experiments build.rs:
       - Keep `-std=metal3.2` flag (required for atomic_thread_fence)
       - Add `println!("cargo:rustc-env=SORT_METALLIB_PATH={}", metallib_path.display())` to embed path
       - Compile `shaders/sort.metal` -> `sort.air` -> `shaders.metallib`
    3. Create placeholder `metal-forge-compute/forge-sort/shaders/sort.metal` with a trivial kernel so build.rs succeeds
  - **Files**:
    - `metal-forge-compute/forge-sort/build.rs` (create)
    - `metal-forge-compute/forge-sort/shaders/sort.metal` (create, placeholder)
  - **Done when**: `cargo build -p forge-sort` compiles shader and embeds metallib path
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-sort 2>&1 | grep -E "Built shaders.metallib|SORT_METALLIB_PATH"`
  - **Commit**: `feat(forge-sort): add build.rs for Metal shader compilation`
  - _Requirements: AC-3.7, FR-10_
  - _Design: build.rs section_

- [ ] 1.3 Extract and rename 4 Metal kernels into sort.metal
  - **Do**:
    1. Extract from `metal-gpu-experiments/shaders/exp17_hybrid.metal`:
       - `exp17_msd_histogram` (line 64-119) -> `sort_msd_histogram`
       - `exp17_msd_atomic_scatter` (line 2186-2274) -> `sort_msd_atomic_scatter`
       - `exp17_msd_prep` (line 2287-2315) -> `sort_msd_prep`
       - `exp17_inner_fused_v4` (line 1976-2171) -> `sort_inner_fused`
    2. Extract struct definitions: `Exp17Params` -> `SortParams`, `BucketDesc` (keep name)
    3. Rename all constants: `EXP17_NUM_BINS` -> `SORT_NUM_BINS`, `EXP17_TILE_SIZE` -> `SORT_TILE_SIZE`, `EXP17_ELEMS` -> `SORT_ELEMS`, `EXP17_THREADS` -> `SORT_THREADS`, `EXP17_NUM_SGS` -> `SORT_NUM_SGS`, `EXP17_MAX_TPB` -> `SORT_MAX_TPB`
    4. Remove `#include "types.h"` — file is self-contained (only uses metal_stdlib)
    5. Do NOT extract: large variants, fused_scatter, inner v1/v2/v3, placeholder, bitonic, etc.
    6. Do NOT change any kernel logic — rename only
  - **Files**:
    - `metal-forge-compute/forge-sort/shaders/sort.metal` (replace placeholder)
  - **Done when**: `cargo build -p forge-sort` compiles all 4 kernels without error
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-sort 2>&1 | grep "Built shaders.metallib"`
  - **Commit**: `feat(forge-sort): extract and rename 4 Metal kernels from exp17`
  - _Requirements: AC-3.1, AC-3.2, AC-3.3, AC-3.4, AC-3.5, AC-3.6, AC-3.8_
  - _Design: sort.metal, Rename map_

- [ ] 1.4 [VERIFY] Quality checkpoint: shader compilation
  - **Do**: Verify shader compiles, crate builds, no warnings
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-sort 2>&1`
  - **Done when**: Build succeeds, metallib generated
  - **Commit**: `chore(forge-sort): pass quality checkpoint` (only if fixes needed)

- [ ] 1.5 Implement GpuSorter struct with sort_u32 dispatch logic
  - **Do**:
    1. In `forge-sort/src/lib.rs`, implement:
       - `SortError` enum with DeviceNotFound, ShaderCompilation(String), GpuExecution(String) via thiserror
       - Private `#[repr(C)] SortParams { element_count: u32, num_tiles: u32, shift: u32, pass: u32 }`
       - Private `#[repr(C)] BucketDesc { offset: u32, count: u32, tile_count: u32, tile_base: u32 }`
       - `GpuSorter` struct with MetalContext, PsoCache, 5 Option buffer fields, data_buf_capacity
       - `GpuSorter::new()`: create MetalContext, load metallib via `env!("SORT_METALLIB_PATH")`, compile 4 PSOs
       - `GpuSorter::sort_u32(&mut self, data: &mut [u32])`: early return for n<=1, ensure_buffers, memcpy host->GPU, zero msd_hist, create cmd_buf+encoder, 4 dispatches, commit+wait, memcpy GPU->host
    2. Port dispatch logic from `bench_investigation_w_at_size()` (exp17_hybrid.rs line 2186-2299):
       - Dispatch 1: sort_msd_histogram — buf_a[0], msd_hist[1], params[2]. Grid: num_tiles TGs x 256
       - Dispatch 2: sort_msd_prep — msd_hist[0], counters[1], bucket_descs[2], tile_size[3]. Grid: 1 TG x 256
       - Dispatch 3: sort_msd_atomic_scatter — buf_a[0], buf_b[1], counters[2], params[3]. Grid: num_tiles TGs x 256
       - Dispatch 4: sort_inner_fused — buf_a[0], buf_b[1], bucket_descs[2], batch_start=0[3]. Grid: 256 TGs x 256
    3. `ensure_buffers(n)`: allocate buf_a/buf_b if None or n*4 > capacity. Metadata buffers: 256*4, 256*4, 256*16 bytes
    4. Constants: `TILE_SIZE: usize = 4096`, `THREADS_PER_TG: usize = 256`
  - **Files**:
    - `metal-forge-compute/forge-sort/src/lib.rs` (rewrite)
  - **Done when**: `cargo build -p forge-sort` succeeds
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-sort`
  - **Commit**: `feat(forge-sort): implement GpuSorter with 4-dispatch sort_u32`
  - _Requirements: AC-1.1, AC-1.2, AC-1.3, AC-1.4, AC-1.5, AC-4.1, AC-4.2, AC-4.3, AC-4.4, AC-5.3, AC-5.4, AC-6.5, FR-1 through FR-9_
  - _Design: GpuSorter, Data Flow, Buffer lifecycle_

- [ ] 1.6 Add basic correctness test at 16M
  - **Do**:
    1. Create `metal-forge-compute/forge-sort/tests/correctness.rs`
    2. Add `sort_and_verify(n)` helper: generate random u32 vec, clone+sort for expected, create GpuSorter, sort_u32, assert_eq
    3. Add `test_sort_16m` test calling `sort_and_verify(16_000_000)`
    4. Add `test_empty` and `test_single` trivial tests
  - **Files**:
    - `metal-forge-compute/forge-sort/tests/correctness.rs` (create)
  - **Done when**: `cargo test -p forge-sort` passes all 3 tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort -- --nocapture 2>&1`
  - **Commit**: `feat(forge-sort): add basic correctness test at 16M`
  - _Requirements: AC-2.1_
  - _Design: Test Strategy_

- [ ] 1.7 [VERIFY] POC checkpoint: sort works end-to-end
  - **Do**: Run full test suite, verify 16M sort produces correct output
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort -- --nocapture 2>&1 | grep -E "test result|ok|FAILED"`
  - **Done when**: All tests pass, 16M random sort matches CPU sort
  - **Commit**: `feat(forge-sort): complete POC`

## Phase 2: Refactoring

After POC validated, clean up code structure.

- [ ] 2.1 Add comprehensive edge case and multi-size tests
  - **Do**:
    1. Add to `tests/correctness.rs`:
       - Size tests: 1K, 4K, 16K, 64K, 256K, 1M, 4M, 32M (in addition to existing 16M)
       - Edge cases: all-zeros (16M), all-same (0xDEADBEEF, 16M), pre-sorted asc (16M), pre-sorted desc (16M), non-tile-aligned (4097), sub-tile (100)
    2. Use `sort_and_verify_data(data: Vec<u32>)` helper for edge cases
  - **Files**:
    - `metal-forge-compute/forge-sort/tests/correctness.rs` (modify)
  - **Done when**: All 16 tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort -- --nocapture 2>&1 | grep -c "test .* ok"`
  - **Commit**: `test(forge-sort): add multi-size and edge case correctness tests`
  - _Requirements: AC-2.1 through AC-2.7_
  - _Design: Test Strategy, Integration Tests_

- [ ] 2.2 Clean up error handling and public API surface
  - **Do**:
    1. Review `lib.rs`: ensure all `unwrap()` calls have proper error propagation via `?`
    2. Verify `SortError` variants cover all failure modes: device not found, shader compile, metallib load, PSO creation, cmd buffer, GPU execution
    3. Add doc comments to `GpuSorter::new()` and `sort_u32()`
    4. Ensure only `GpuSorter`, `SortError`, and `sort_u32` are public. `SortParams`, `BucketDesc` stay private
  - **Files**:
    - `metal-forge-compute/forge-sort/src/lib.rs` (modify)
  - **Done when**: Public API is `GpuSorter::new() -> Result<Self, SortError>` and `sort_u32(&mut self, &mut [u32]) -> Result<(), SortError>` only
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo doc -p forge-sort --no-deps 2>&1 | grep -E "error|warning" | head -5`
  - **Commit**: `refactor(forge-sort): clean up error handling and public API`
  - _Requirements: AC-1.1, AC-1.3, FR-9_
  - _Design: Error Handling_

- [ ] 2.3 [VERIFY] Quality checkpoint: build + test + doc
  - **Do**: Run build, all tests, doc generation
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-sort && cargo test -p forge-sort && cargo doc -p forge-sort --no-deps 2>&1 | grep -c "error"`
  - **Done when**: Build clean, all tests pass, zero doc errors
  - **Commit**: `chore(forge-sort): pass quality checkpoint` (only if fixes needed)

## Phase 3: Testing

- [ ] 3.1 Add struct layout unit tests
  - **Do**:
    1. Add `#[cfg(test)]` module in `lib.rs` with:
       - `test_sort_params_size`: `assert_eq!(size_of::<SortParams>(), 16)`
       - `test_bucket_desc_size`: `assert_eq!(size_of::<BucketDesc>(), 16)`
       - `test_sort_error_display`: verify each SortError variant has meaningful Display output
  - **Files**:
    - `metal-forge-compute/forge-sort/src/lib.rs` (modify)
  - **Done when**: Unit tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --lib`
  - **Commit**: `test(forge-sort): add struct layout and error display unit tests`
  - _Requirements: AC-6.5_
  - _Design: Unit Tests_

- [ ] 3.2 Add performance sanity test
  - **Do**:
    1. Add `test_sort_16m_perf_sanity` to `tests/correctness.rs`
    2. Sort 16M random u32, assert total wall time < 50ms (very generous bound -- actual is ~3ms GPU)
    3. This is NOT a perf benchmark -- just a sanity check to detect catastrophic regressions
  - **Files**:
    - `metal-forge-compute/forge-sort/tests/correctness.rs` (modify)
  - **Done when**: Perf sanity test passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort -- test_sort_16m_perf_sanity --nocapture`
  - **Commit**: `test(forge-sort): add performance sanity test`
  - _Requirements: AC-5.1_
  - _Design: Performance Tests_

- [ ] 3.3 Add buffer reuse test
  - **Do**:
    1. Add `test_buffer_reuse` to `tests/correctness.rs`
    2. Create GpuSorter, sort 16M, then sort 16M again, then sort 1M -- all should succeed and produce correct results
    3. This validates grow-only buffer pool behavior
  - **Files**:
    - `metal-forge-compute/forge-sort/tests/correctness.rs` (modify)
  - **Done when**: Buffer reuse test passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort -- test_buffer_reuse --nocapture`
  - **Commit**: `test(forge-sort): add buffer reuse test`
  - _Requirements: AC-4.1, AC-4.2, AC-4.3, AC-4.4_
  - _Design: Buffer lifecycle_

- [ ] 3.4 [VERIFY] Quality checkpoint: full test suite
  - **Do**: Run entire test suite, verify all pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort 2>&1 | tail -5`
  - **Done when**: All tests pass (expected: ~20 tests)
  - **Commit**: `chore(forge-sort): pass quality checkpoint` (only if fixes needed)

## Phase 4: Quality Gates

- [ ] 4.1 Local quality check
  - **Do**: Run all quality checks locally
  - **Verify**: All commands must pass:
    - `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-sort`
    - `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort`
    - `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo clippy -p forge-sort -- -D warnings 2>&1`
    - `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo doc -p forge-sort --no-deps 2>&1`
  - **Done when**: All commands pass with no errors or warnings
  - **Commit**: `fix(forge-sort): address lint/type issues` (if fixes needed)

- [ ] 4.2 Create PR and verify CI
  - **Do**:
    1. Verify current branch: `git branch --show-current` (should be `feat/gpu-query` or a feature branch)
    2. Stage forge-sort files: `git add metal-forge-compute/forge-sort/ metal-forge-compute/Cargo.toml metal-forge-compute/Cargo.lock`
    3. Commit all changes
    4. Push branch: `git push -u origin <branch>`
    5. Create PR: `gh pr create --title "feat(forge-sort): GPU radix sort library" --body "..."`
  - **Verify**: `gh pr checks --watch` (wait for CI completion), all checks green
  - **Done when**: PR created, CI passing
  - **If CI fails**: Read failure details, fix, push, re-verify

## Phase 5: PR Lifecycle

- [ ] 5.1 Monitor CI and fix failures
  - **Do**:
    1. Check CI status: `gh pr checks`
    2. If failures, read logs, fix locally, push
    3. Repeat until green
  - **Verify**: `gh pr checks` shows all passing
  - **Done when**: CI green

- [ ] 5.2 [VERIFY] Full local CI simulation
  - **Do**: Run complete local quality suite
  - **Verify**:
    - `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-sort && cargo test -p forge-sort && cargo clippy -p forge-sort -- -D warnings`
  - **Done when**: All pass
  - **Commit**: None (or fix commit if needed)

- [ ] 5.3 [VERIFY] AC checklist
  - **Do**: Programmatically verify all acceptance criteria:
    1. AC-1.1: `cargo test -p forge-sort --lib -- test_sort_error_display` (GpuSorter::new exists, returns Result)
    2. AC-1.2/1.3: `cargo test -p forge-sort -- test_sort_16m` (sort_u32 works, returns Result)
    3. AC-1.4/1.5: `cargo test -p forge-sort -- test_empty test_single`
    4. AC-2.1-2.7: `cargo test -p forge-sort -- test_sort_ test_all_ test_pre_ test_reverse test_non_tile test_sub_tile`
    5. AC-3.1-3.8: `grep -c "sort_msd_histogram\|sort_msd_prep\|sort_msd_atomic_scatter\|sort_inner_fused\|SORT_NUM_BINS\|SortParams" metal-forge-compute/forge-sort/shaders/sort.metal`
    6. AC-4.1-4.4: `cargo test -p forge-sort -- test_buffer_reuse`
    7. AC-5.3: `grep -c "dispatchThreadgroups" metal-forge-compute/forge-sort/src/lib.rs` (should be 4)
    8. AC-6.1-6.5: `cargo build -p forge-sort && grep "forge-sort" metal-forge-compute/Cargo.toml`
  - **Verify**: All grep/test commands produce expected output
  - **Done when**: All ACs confirmed met
  - **Commit**: None

## Notes

- **POC shortcuts taken**: None significant -- this is extraction of proven code
- **Production TODOs**:
  - Benchmark harness integration with forge-bench (out of scope per requirements)
  - Key-value pair sorting (out of scope for v1)
  - Async/non-blocking API (out of scope)
- **Key source locations**:
  - Metal kernels: `metal-gpu-experiments/shaders/exp17_hybrid.metal` lines 64-119, 1976-2171, 2186-2274, 2287-2315
  - Rust dispatch: `metal-gpu-experiments/src/exp17_hybrid.rs` lines 2186-2299 (`bench_investigation_w_at_size`)
  - Build pattern: `metal-gpu-experiments/build.rs` (has `-std=metal3.2` already)
  - Forge primitives: `metal-forge-compute/forge-primitives/src/` (MetalContext, PsoCache, alloc_buffer, read_buffer_slice)
- **Metallib loading**: Embedded via `env!("SORT_METALLIB_PATH")` to avoid ambiguity with forge-primitives `shaders.metallib`
- **Buffer constants**: msd_hist=1024B, counters=1024B, bucket_descs=4096B (256 x 16B BucketDesc)
