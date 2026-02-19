---
spec: gpu-sort-5000
phase: tasks
total_tasks: 20
created: 2026-02-19
---

# Tasks: GPU Radix Sort 5000+ Mkeys/s (MSD+LSD Hybrid, Experiment 17)

## Execution Context

- **Testing depth**: Minimal -- POC only, correctness verified inline per exp16 pattern (no separate test suite)
- **Deployment approach**: N/A -- standalone experiment, no deployment
- **Execution priority**: Ship fast -- POC first, prove the concept works, polish later
- **Build**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments`
- **Run**: `cargo run --release -p metal-gpu-experiments`
- **Codebase**: `metal-gpu-experiments/` crate, Experiment 17

## Phase 1: Make It Work (POC)

Focus: Get Phase 0 SLC benchmark running, then MSD scatter, then inner sort, then end-to-end. Skip polish, accept hardcoded values, inline correctness checks only.

- [x] 1.1 Scaffold exp17 module with Phase 0 SLC scatter benchmark
  - **Do**:
    1. Create `metal-gpu-experiments/shaders/exp17_hybrid.metal` with minimal header: includes, defines for `EXP17_NUM_BINS 256`, `EXP17_TILE_SIZE 4096`, `EXP17_ELEMS 16`, `EXP17_THREADS 256`, `EXP17_NUM_SGS 8`, `EXP17_MAX_TPB 17`. Define `Exp17Params` struct (element_count, num_tiles, shift, pass), `Exp17InnerParams` struct (shift), and `BucketDesc` struct (offset, count, tile_count, tile_base). Add a placeholder kernel `exp17_placeholder` (trivial copy) so the file compiles.
    2. Create `metal-gpu-experiments/src/exp17_hybrid.rs` with `pub fn run(ctx: &MetalContext)`. Port the SLC scatter benchmark from `exp16_8bit.rs` scatter diagnostic section: reuse `exp16_diag_scatter_binned` PSO. Run at 5 sizes: 62500, 250000, 1000000, 4000000, 16000000. For each size, generate random uint32 data, compute 256-bin scatter offsets on CPU (same pattern as exp16 lines 1200-1224), dispatch `exp16_diag_scatter_binned` with 5 warmup + 20 timed iterations, print p5/p50/p95 timing + GB/s. Include go/no-go gate: if 250K scatter < 80 GB/s, print "ABORT: SLC scatter too slow" and return early.
    3. Modify `metal-gpu-experiments/src/main.rs`: add `mod exp17_hybrid;`, comment out `exp16_8bit::run(&ctx);`, add `exp17_hybrid::run(&ctx);`.
    4. Copy `percentile()` and `print_stats()` helpers into exp17_hybrid.rs (or make them pub in exp16 and reuse -- simpler to copy for POC).
  - **Files**: `metal-gpu-experiments/shaders/exp17_hybrid.metal`, `metal-gpu-experiments/src/exp17_hybrid.rs`, `metal-gpu-experiments/src/main.rs`
  - **Done when**: `cargo run --release -p metal-gpu-experiments` prints SLC scatter bandwidth at all 5 sizes and either prints go/no-go result or aborts
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `feat(exp17): scaffold exp17 with Phase 0 SLC scatter benchmark`
  - _Requirements: FR-1, FR-2, AC-1.1 through AC-1.5_
  - _Design: Phase 0 SLC Scatter Bandwidth Benchmark_

- [x] 1.2 Implement MSD histogram kernel (1-pass, bits 24:31)
  - **Do**:
    1. In `exp17_hybrid.metal`, implement `exp17_msd_histogram` -- clone from `exp16_combined_histogram` but single-pass only (removes the `for (uint p = 0; p < NUM_PASSES; p++)` loop). Key changes: read bits[24:31] only (shift=24, mask=0xFF), write to `global_hist[lid]` (not `global_hist[p*256+lid]`). Same per-SG atomic histogram pattern (TG memory `sg_counts[8*256]`, zero, atomic_fetch_add, reduce across SGs, atomic_fetch_add to global).
    2. Implement `exp17_global_prefix` -- clone from `exp16_global_prefix` but only 1 pass (only SG 0 does work, 256-bin prefix sum via 8 chunks of 32 with `simd_prefix_exclusive_sum`).
    3. In `exp17_hybrid.rs`, add `bench_msd_histogram()`: allocate buf_input (16M random u32), buf_msd_hist (256*4 bytes, zeroed), set Exp17Params (element_count=16M, num_tiles=3907, shift=24, pass=0). Dispatch msd_histogram with 3907 TGs, then global_prefix with 1 TG. Read back histogram, verify sum == N. Print histogram stats (min/max/avg bucket size).
  - **Files**: `metal-gpu-experiments/shaders/exp17_hybrid.metal`, `metal-gpu-experiments/src/exp17_hybrid.rs`
  - **Done when**: MSD histogram produces 256 bins that sum to N=16M, prefix sums are monotonically increasing
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `feat(exp17): add MSD histogram kernel (1-pass bits 24:31)`
  - _Requirements: FR-3, FR-4, AC-2.1_
  - _Design: Phase 1 MSD Scatter, exp17_msd_histogram_

- [x] 1.3 Implement GPU-side BucketDesc computation kernel
  - **Do**:
    1. In `exp17_hybrid.metal`, implement `exp17_compute_bucket_descs` per design: 1 TG, 256 threads. Each thread reads `global_hist[lid]` (raw count), computes `tile_count = ceil(count/4096)`. Thread 0 serial prefix sum for offsets. Writes `BucketDesc{offset, count, tile_count, tile_base=0}` to output buffer. Must run AFTER msd_histogram but BEFORE global_prefix (since prefix destroys raw counts).
    2. In `exp17_hybrid.rs`, add `#[repr(C)] struct BucketDesc { offset: u32, count: u32, tile_count: u32, tile_base: u32 }`. Allocate `buf_bucket_descs` (256*16 bytes). After histogram dispatch, dispatch `compute_bucket_descs` with 1 TG. Read back and verify: sum of counts == N, offsets are monotonically increasing, offsets[last] + counts[last] == N.
  - **Files**: `metal-gpu-experiments/shaders/exp17_hybrid.metal`, `metal-gpu-experiments/src/exp17_hybrid.rs`
  - **Done when**: BucketDesc[256] read back with correct offsets and counts (verified sum=N)
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `feat(exp17): add GPU-side BucketDesc computation kernel`
  - _Requirements: FR-4, AC-2.3_
  - _Design: exp17_compute_bucket_descs_

- [x] 1.4 Implement MSD scatter pass (reuse exp16_partition)
  - **Do**:
    1. In `exp17_hybrid.rs`, implement `run_msd_scatter()`: encode the full MSD pipeline in a single command buffer with single encoder and PSO switching (per design -- 1 CB, 1 encoder, PSO switches between dispatches):
       - `setPSO(msd_histogram)` -> dispatch 3907 TGs
       - `setPSO(compute_bucket_descs)` -> dispatch 1 TG (reads raw histogram before prefix destroys it)
       - `setPSO(global_prefix)` -> dispatch 1 TG
       - `setPSO(zero_status)` -> dispatch 3907 TGs (reuse `exp16_zero_status`)
       - `setPSO(msd_scatter)` -> dispatch 3907 TGs (reuse `exp16_partition` with shift=24, pass=0)
       - endEncoding(), commit(), waitUntilCompleted()
    2. Note: For MSD scatter with exp16_partition, `global_hist` must have prefix sums at offset `pass*256 = 0*256 = 0`. The global_prefix kernel writes to `global_hist[0..255]`. Set `params.pass = 0` for exp16_partition so it reads `global_hist[0*256+d]`.
    3. Buffers needed: buf_input -> buf_a (copy), buf_a (src) -> buf_b (MSD output), buf_msd_hist (256*4, for 1 pass only -- but exp16_partition reads global_hist at pass*256 offset, so allocate 4*256*4 to be safe and only populate pass 0 bins), buf_tile_status (num_tiles*256*4), buf_counters (4), buf_bucket_descs (256*16).
    4. Correctness check: read buf_b, for each element verify `(element >> 24) & 0xFF` matches the bucket it's in (use bucket_descs offsets to determine bucket boundaries). Print mismatch count.
  - **Files**: `metal-gpu-experiments/src/exp17_hybrid.rs`
  - **Done when**: MSD scatter at 16M elements produces 256 correctly-partitioned buckets, all elements in correct bucket by bits[24:31], 0 mismatches
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `feat(exp17): implement MSD scatter pass with single-encoder pipeline`
  - _Requirements: FR-3, AC-2.1, AC-2.2, AC-2.5_
  - _Design: Phase 1 MSD Scatter, Command Structure_

- [x] 1.5 [VERIFY] Quality checkpoint: build compiles, MSD scatter correctness
  - **Do**: Build with shader rebuild, verify the binary compiles and runs without panics. If any compilation errors, fix them.
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5`
  - **Done when**: Build succeeds with exit code 0
  - **Commit**: `chore(exp17): pass quality checkpoint` (only if fixes needed)

- [x] 1.6 Implement inner histogram kernel (per-tile, all buckets in one dispatch)
  - **Do**:
    1. In `exp17_hybrid.metal`, implement `exp17_inner_histogram`:
       - Fixed dispatch: 4352 TGs (17 tiles x 256 buckets). Arithmetic mapping: `bucket_id = gid / EXP17_MAX_TPB`, `tile_in_bucket = gid % EXP17_MAX_TPB`.
       - Read BucketDesc for bucket_id. Early-exit if `tile_in_bucket * EXP17_TILE_SIZE >= desc.count`.
       - Load 4096 elements from `data[desc.offset + tile_in_bucket * EXP17_TILE_SIZE + ...]` using SG-contiguous layout (same as exp16 Phase 1).
       - Extract digit: `(key >> params.shift) & 0xFF`.
       - Per-SG atomic histogram on TG memory (same pattern as exp16 Phase 2).
       - Reduce across SGs, write to `tile_hists[bucket_id * EXP17_MAX_TPB * 256 + tile_in_bucket * 256 + lid]`.
    2. Implement `exp17_inner_zero`:
       - Simple kernel: `if (tid < total_entries) tile_hists[tid] = 0;`
       - `total_entries = 256 * EXP17_MAX_TPB * 256 = 1,114,112`
  - **Files**: `metal-gpu-experiments/shaders/exp17_hybrid.metal`
  - **Done when**: Inner histogram and zero kernels compile in Metal shader
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `feat(exp17): add inner histogram and zero kernels`
  - _Requirements: FR-5, FR-6, AC-3.2, AC-3.3_
  - _Design: exp17_inner_histogram, exp17_inner_zero_

- [x] 1.7 Implement inner scan+scatter kernel (serial prefix + rank + scatter)
  - **Do**:
    1. In `exp17_hybrid.metal`, implement `exp17_inner_scan_scatter`:
       - Same dispatch geometry as inner_histogram (4352 TGs, arithmetic mapping).
       - TG memory: `sg_hist_or_rank[8*256]` (8KB), `sg_prefix[8*256]` (8KB), `tile_hist_local[256]` (1KB), `exclusive_pfx[256]` (1KB). Total 18KB.
       - Phase 1: Load 4096 elements from src[desc.offset + tile * 4096 + ...]. Extract digits.
       - Phase 2: Per-SG atomic histogram on TG memory (same as exp16 P2).
       - Phase 2b: Cross-SG prefix (same as exp16 P2b). Store tile_hist_local[lid].
       - Phase 3 (SERIAL SCAN -- replaces decoupled lookback): Thread lid scans `tile_hists[bucket_id * MAX_TPB * 256 + t * 256 + lid]` for t=0..tile_in_bucket-1. Running sum = exclusive prefix for this tile within bucket. Store to `exclusive_pfx[lid]`.
       - Phase 4: Per-SG atomic rank + scatter (same as exp16 P5). Destination = `desc.offset + exclusive_pfx[d] + sg_prefix[sg*256+d] + within_sg`. Write to dst[destination].
       - Note: NO global_hist offset -- inner sort writes within bucket region only.
  - **Files**: `metal-gpu-experiments/shaders/exp17_hybrid.metal`
  - **Done when**: Inner scan+scatter kernel compiles in Metal shader
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `feat(exp17): add inner scan+scatter kernel with serial prefix`
  - _Requirements: FR-5, FR-6, FR-7, AC-3.1, AC-3.4_
  - _Design: exp17_inner_scan_scatter, Serial scan prefix_

- [x] 1.8 Wire up inner sort in Rust host + end-to-end correctness
  - **Do**:
    1. In `exp17_hybrid.rs`, implement `run_hybrid_sort()`:
       - Allocate all buffers per design: buf_a (64MB), buf_b (64MB), buf_msd_hist (4*256*4 for exp16_partition compat), buf_tile_status (3907*256*4), buf_counters (4), buf_bucket_descs (256*16), buf_tile_hists (256*17*256*4 = ~4.5MB).
       - Copy random input to buf_a.
       - Single command buffer, single encoder, 14 dispatches:
         a. `setPSO(msd_histogram)` -> dispatch(3907, 256)
         b. `setPSO(compute_bucket_descs)` -> dispatch(1, 256)
         c. `setPSO(global_prefix)` -> dispatch(1, 256)
         d. `setPSO(zero_status)` -> dispatch(3907, 256)
         e. `setPSO(msd_scatter/exp16_partition)` -> dispatch(3907, 256) -- src=buf_a, dst=buf_b, shift=24, pass=0
         f. For each inner pass (shift=0, 8, 16):
            - `setPSO(inner_zero)` -> dispatch(ceil(256*17*256/256), 256)
            - `setPSO(inner_histogram)` -> dispatch(4352, 256) -- reads from current src buffer
            - `setPSO(inner_scan_scatter)` -> dispatch(4352, 256) -- reads src, writes dst
            - Ping-pong: pass 0 src=buf_b dst=buf_a, pass 1 src=buf_a dst=buf_b, pass 2 src=buf_b dst=buf_a
         g. endEncoding(), commit(), waitUntilCompleted()
       - Result after 3 inner passes (odd count) is in buf_a.
       - For inner passes, set buffer bindings: src buffer, dst buffer, tile_hists, bucket_descs, Exp17InnerParams{shift}.
    2. Correctness check at 16M: read buf_a, compare with CPU `expected.sort()`. Print mismatch count + first 5 mismatches.
    3. Also verify per-bucket correctness: for each bucket, check elements within bucket range are sorted.
  - **Files**: `metal-gpu-experiments/src/exp17_hybrid.rs`
  - **Done when**: End-to-end hybrid sort of 16M random uint32 produces correct sorted output (0 mismatches vs CPU sort)
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `feat(exp17): wire up end-to-end hybrid sort with correctness check`
  - _Requirements: FR-5, FR-8, FR-9, AC-4.1, AC-4.6_
  - _Design: Command Structure, Buffer Layout, Data Flow_

- [x] 1.9 [VERIFY] Quality checkpoint: full build + run
  - **Do**: Full shader rebuild + build. Run the binary and verify it completes without panics and produces correct output.
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5`
  - **Done when**: Build succeeds with exit code 0
  - **Commit**: `chore(exp17): pass quality checkpoint` (only if fixes needed)

- [x] 1.10 Add benchmark loop + per-phase timing + multi-size correctness
  - **Do**:
    1. In `exp17_hybrid.rs`, add benchmark: 5 warmup + 50 timed iterations of the full hybrid pipeline at 16M. Compute p5/p50/p95 timing and Mkeys/s using `percentile()` and `print_stats()`.
    2. Add per-phase timing: run MSD phase and inner phases in separate command buffers to measure individual GPU times. Print breakdown: MSD histogram time, MSD scatter time (includes prefix+zero), inner pass 0/1/2 time, dispatch overhead estimate (total - sum of phases).
    3. Add correctness checks at 1M, 4M, 16M (AC-4.6). For 1M and 4M, adjust num_tiles/dispatch geometries accordingly.
    4. Print comparison vs baseline: hybrid p50 Mkeys/s vs known 3003 Mkeys/s baseline.
  - **Files**: `metal-gpu-experiments/src/exp17_hybrid.rs`
  - **Done when**: Binary prints benchmark results at 1M/4M/16M with correctness, per-phase timing breakdown, and comparison vs baseline
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `feat(exp17): add benchmark loop with per-phase timing and multi-size correctness`
  - _Requirements: FR-8, FR-9, AC-4.1 through AC-4.6_
  - _Design: Benchmark Methodology, Performance Budget_

- [x] 1.11 POC Checkpoint
  - **Do**: Run the full experiment end-to-end. Verify Phase 0 SLC benchmark runs, MSD scatter correct, inner sort correct, end-to-end benchmark prints results. Capture output. This is the POC validation -- the hybrid sort compiles, runs, and produces correct results. Performance target may or may not be met.
  - **Done when**: `cargo run --release -p metal-gpu-experiments` completes without error and prints all benchmark results
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo run --release -p metal-gpu-experiments 2>&1 | grep -E '(Mkeys|scatter|ABORT|FAIL|ok|mismatch)' | head -30`
  - **Commit**: `feat(exp17): complete POC -- MSD+LSD hybrid sort`
  - _Requirements: All P0 FRs_

## Phase 2: Refactoring + Polish

After POC validated, clean up code and add fallback analysis.

- [x] 2.1 Add fallback analysis if target not met
  - **Do**:
    1. In `exp17_hybrid.rs`, after benchmark results, add fallback analysis section (FR-10):
       - Print per-phase bandwidth utilization: actual GB/s vs theoretical (245 for MSD DRAM, 469 for inner SLC).
       - Identify bottleneck phase: which phase uses the most time relative to its theoretical minimum.
       - Compute theoretical ceiling from measured per-phase bandwidths.
       - Print comparison table: baseline 3003 Mkeys/s vs hybrid measured.
    2. Only print this section if p50 < 5000 Mkeys/s (AC-5.1 trigger).
  - **Files**: `metal-gpu-experiments/src/exp17_hybrid.rs`
  - **Done when**: If target not met, detailed fallback analysis printed with per-phase BW utilization
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `feat(exp17): add fallback bandwidth analysis`
  - _Requirements: FR-10, AC-5.1 through AC-5.4_
  - _Design: Critical Path Optimization_

- [ ] 2.2 Refactor run() for clean output formatting
  - **Do**:
    1. Structure the `run()` function output to match exp16 style:
       - Section header: "Experiment 17: MSD+LSD Hybrid Radix Sort"
       - Phase 0: SLC scatter benchmark results
       - Phase 1+2: End-to-end sort results at multiple sizes
       - Per-phase timing breakdown
       - Fallback analysis (if applicable)
       - Summary comparison vs baseline
    2. Extract helper functions: `bench_slc_scatter()`, `bench_hybrid()`, `check_correctness()`.
    3. Ensure all buffers are properly sized for variable N (not just 16M).
  - **Files**: `metal-gpu-experiments/src/exp17_hybrid.rs`
  - **Done when**: Output is clean, well-formatted, matches exp16 style
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `refactor(exp17): clean output formatting and extract helpers`
  - _Design: Existing Patterns to Follow_

- [ ] 2.3 [VERIFY] Quality checkpoint: build + correctness
  - **Do**: Full rebuild, run, verify no regressions in correctness at all sizes.
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5`
  - **Done when**: Build succeeds, no correctness regressions
  - **Commit**: `chore(exp17): pass quality checkpoint` (only if fixes needed)

## Phase 3: Optimize (if needed)

Only needed if POC shows < 5000 Mkeys/s and bottleneck is identified.

- [ ] 3.1 Tune inner sort tile size (4096 vs 8192)
  - **Do**:
    1. If per-phase timing shows inner passes are the bottleneck, try 8192 elements/tile (32 per thread, halves tile count to ~8 per bucket).
    2. Add `EXP17_TILE_SIZE_LARGE 8192`, `EXP17_ELEMS_LARGE 32` defines. Duplicate inner_histogram and inner_scan_scatter as `_v2` variants using 8192 tiles.
    3. Benchmark both tile sizes, print comparison.
    4. If 8192 is faster, switch default. If not, keep 4096.
  - **Files**: `metal-gpu-experiments/shaders/exp17_hybrid.metal`, `metal-gpu-experiments/src/exp17_hybrid.rs`
  - **Done when**: Both tile sizes benchmarked, winner identified
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` exits 0
  - **Commit**: `feat(exp17): tune inner sort tile size (4096 vs 8192)`
  - _Requirements: FR-12_
  - _Design: Technical Decisions, tile size_

- [ ] 3.2 [VERIFY] Quality checkpoint after optimization
  - **Do**: Full rebuild, run, verify correctness still holds at all sizes with any optimization changes.
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5`
  - **Done when**: Build succeeds, correctness preserved
  - **Commit**: `chore(exp17): pass quality checkpoint` (only if fixes needed)

## Phase 4: Quality Gates

- [ ] 4.1 [VERIFY] Full local CI: build + run
  - **Do**: Run complete build from scratch with shader rebuild. Verify binary produces correct output at all test sizes.
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release -p metal-gpu-experiments 2>&1 | tail -5` -- must exit 0
  - **Done when**: Build succeeds, all correctness checks pass
  - **Commit**: `chore(exp17): pass local CI` (if fixes needed)

- [ ] 4.2 Create PR and verify CI
  - **Do**:
    1. Verify current branch: `git branch --show-current` (expect `feat/gpu-query` or a feature branch)
    2. Stage new/modified files: `metal-gpu-experiments/shaders/exp17_hybrid.metal`, `metal-gpu-experiments/src/exp17_hybrid.rs`, `metal-gpu-experiments/src/main.rs`
    3. Create PR: `gh pr create --title "feat(exp17): MSD+LSD hybrid radix sort targeting 5000+ Mkeys/s" --body "..."`
    4. Verify CI: `gh pr checks --watch`
  - **Verify**: `gh pr checks` shows all green
  - **Done when**: PR created, CI passes
  - **If CI fails**: Read failure, fix locally, push, re-verify

## Phase 5: PR Lifecycle

- [ ] 5.1 Monitor CI and fix failures
  - **Do**: If CI fails after PR creation, read failure details with `gh pr checks`, fix issues locally, commit and push. Repeat until green.
  - **Verify**: `gh pr checks` shows all passing
  - **Done when**: All CI checks green

- [ ] 5.2 [VERIFY] AC checklist
  - **Do**: Verify all acceptance criteria are met:
    - AC-1.1: Scatter bandwidth at 5 sizes (grep output for 5 size labels)
    - AC-1.5: Go/no-go documented in output
    - AC-2.1: Bucket counts sum to N (verified in correctness check)
    - AC-2.2: Elements in correct buckets (verified in MSD correctness)
    - AC-3.1: Per-bucket sorted (verified in end-to-end correctness)
    - AC-3.2: Single dispatch for all buckets (code inspection: 4352 TGs)
    - AC-4.1: End-to-end correctness at 16M
    - AC-4.2: 50 runs + 5 warmup
    - AC-4.3: p5/p50/p95 printed
    - AC-4.4: Per-phase breakdown printed
    - AC-4.6: Correctness at 1M, 4M, 16M
    - AC-5.1-5.4: Fallback analysis if target not met
  - **Verify**: `rm -rf target/release/build/metal-gpu-experiments-* && cargo run --release -p metal-gpu-experiments 2>&1 | grep -cE '(Mkeys|scatter|ok|FAIL|bucket|phase)'` returns > 10 matching lines
  - **Done when**: All ACs confirmed met via output inspection
  - **Commit**: None

## Notes

- **POC shortcuts taken**:
  - `percentile()` and `print_stats()` copied into exp17 (not extracted to shared module)
  - Hardcoded 4352 TGs for inner sort (17*256) -- correct for uniform random at 16M only
  - buf_msd_hist allocated as 4*256*4 (4 passes worth) for exp16_partition compatibility even though only pass 0 is used
  - No automated test suite -- correctness checked inline at runtime per exp16 pattern

- **Production TODOs**:
  - Extract shared timing/stats helpers to common module
  - Support non-uniform distributions (would need dynamic MAX_TPB)
  - Consider TG reorder for inner scatter if bandwidth utilization < 75%
  - Consider 4-bit MSD fallback (FR-13) if 8-bit MSD scatter proves slow

- **Key design decisions in POC**:
  - Single encoder with PSO switching (14 dispatches, ~14us overhead) -- NOT separate encoders
  - GPU-computed BucketDesc eliminates CPU readback between phases
  - exp16_partition reused directly for MSD scatter (shift=24, pass=0)
  - Serial scan for inner prefix (~16 cycles/thread) instead of decoupled lookback
  - Fixed 4352 TG dispatch with arithmetic mapping (gid/17, gid%17) instead of lookup table
