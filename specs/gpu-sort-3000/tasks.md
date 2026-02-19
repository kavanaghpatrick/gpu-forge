---
spec: gpu-sort-3000
phase: tasks
total_tasks: 20
created: 2026-02-19
---

# Tasks: 8-Bit Radix Sort (3000+ Mkeys/s)

## Phase 1: Make It Work (POC) — 8-Bit Radix @ 2048 Tiles

Focus: Get 8-bit/4-pass radix sort working with same tile size as exp15. Validate 256-bin histogram, lookback, ranking. Benchmark vs exp15 baseline.

- [x] 1.1 Create exp16 Metal shader with 4 kernels (2048 tile, 256 bins, 4 passes)
  - **Do**:
    1. Create `metal-gpu-experiments/shaders/exp16_8bit.metal`
    2. Define constants: `EXP16_NUM_BINS=256`, `EXP16_NUM_SGS=8`, `EXP16_TILE_SIZE=2048`, `EXP16_ELEMS=8`, `EXP16_NUM_PASSES=4`, `EXP16_THREADS=256`
    3. Define `Exp16Params` struct (element_count, num_tiles, num_tgs, shift, pass)
    4. Implement `exp16_combined_histogram`: per-SG atomic histogram on TG memory (NOT private counters), loop over 4 passes, reduce across SGs, atomic add to global_hist[p*256+bin]
    5. Implement `exp16_global_prefix`: 4 SGs (one per pass), each does 8 chunks of 32-bin prefix sum via `simd_prefix_exclusive_sum` + running total, broadcast chunk total from lane 31
    6. Implement `exp16_zero_status`: zero tile_status[tid] and counters[0], identical pattern to exp15
    7. Implement `exp16_partition` with 5 phases:
       - P1: Load 8 elems, extract 8-bit digits (`& 0xFF`), invalid = 0xFF
       - P2: Per-SG atomic histogram (`atomic_fetch_add` on sg_counts[simd_id*256+d])
       - P2b: Cross-SG prefix (256 threads, each handles one bin, serial loop over 8 SGs)
       - P3: Publish AGGREGATE (256 threads, one per bin), device-scope fence
       - P4: Decoupled lookback (ALL 256 threads participate, one per bin)
       - P5: Per-SG atomic ranking + scatter (reuse sg_counts memory for sg_rank_ctr)
    8. TG memory layout: sg_hist_or_rank[8*256]=8KB, sg_prefix[8*256]=8KB, tile_hist[256]=1KB, exclusive_pfx[256]=1KB = 18KB total
    9. Include `#include "types.h"` for FLAG_AGGREGATE, FLAG_PREFIX, FLAG_SHIFT, VALUE_MASK
  - **Files**: `metal-gpu-experiments/shaders/exp16_8bit.metal`
  - **Done when**: Shader compiles without errors via Metal 3.2 compiler
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release 2>&1 | tail -5`
  - **Commit**: `feat(exp16): add 8-bit radix sort Metal shader with 256-bin histogram`
  - _Requirements: FR-1, FR-2, FR-5, FR-6, FR-8, FR-9, FR-10, FR-12_
  - _Design: K1-K4, TG Memory Layout, Partition Kernel Phase Flow_

- [ ] 1.2 Create exp16 Rust host with bench and correctness verification
  - **Do**:
    1. Create `metal-gpu-experiments/src/exp16_8bit.rs`
    2. Define constants: NUM_BINS=256, TILE_SIZE=2048, NUM_PASSES=4, THREADS_PER_TG=256, WARMUP=5, RUNS=50
    3. Define `Exp16Params` struct (#[repr(C)]): element_count, num_tiles, num_tgs, shift, pass (all u32)
    4. Copy `percentile()` and `print_stats()` from exp15 (or make them pub in exp15 and import — prefer copy for isolation)
    5. Implement `bench_8bit()` following exp15's `bench_onesweep()` pattern:
       - Allocate: buf_input, buf_a, buf_b, buf_global_hist (4*256*4B), buf_tile_status (num_tiles*256*4B), buf_counters (4B)
       - Encode: hist(buf_a) -> prefix(global_hist) -> 4x(zero + partition with ping-pong)
       - Zero global_hist CPU-side before encoding
       - Shift: pass*8 (not pass*4)
       - Hist dispatches num_tiles TGs; prefix dispatches 1 TG; zero dispatches ceil(num_tiles*256/256) TGs; partition dispatches num_tiles TGs
       - Correctness: compare buf_a result with CPU sorted expected
       - Benchmark: 5 warmup + 50 timed runs, return sorted times + correct flag
    6. Implement `run()`:
       - Create 4 PSOs: exp16_combined_histogram, exp16_global_prefix, exp16_zero_status, exp16_partition
       - Run at 1M, 2M, 4M, 8M, 16M random uint32
       - Print stats per size
    7. Add `mod exp16_8bit;` to main.rs
    8. Add `exp16_8bit::run(&ctx);` call in main() (comment out exp15 call or keep both)
  - **Files**: `metal-gpu-experiments/src/exp16_8bit.rs`, `metal-gpu-experiments/src/main.rs`
  - **Done when**: Binary builds, runs correctness check at 1M, prints results
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release && ./target/release/metal-gpu-experiments 2>&1 | grep -E "(exp16|Exp.*16|8bit|8-bit|Mkeys|FAIL|ok)" | head -20`
  - **Commit**: `feat(exp16): add Rust host for 8-bit radix sort with correctness verification`
  - _Requirements: FR-11, FR-14, AC-1.5, AC-5.1_
  - _Design: Rust Host Design, Encoder Sequence, Buffer Allocations_

- [ ] 1.3 Debug and fix correctness at 1M elements
  - **Do**:
    1. Run exp16 and check correctness output
    2. If FAIL: examine first mismatched indices to identify which phase is broken
    3. Common bugs to check:
       - Histogram: per-SG atomic contention producing wrong counts (compare GPU histogram vs CPU)
       - Global prefix: 8-chunk serial approach off-by-one (broadcast from lane 31 must be prefix+val, not just prefix)
       - Partition: sg_prefix computation (cross-SG exclusive prefix for each bin)
       - Partition: global position formula: `global_hist[pass*256+d] + exclusive_pfx[d] + sg_prefix[simd_id*256+d] + within_sg`
       - Partition: TG memory reuse — ensure sg_rank_ctr zeroed AFTER lookback completes
       - Partition: 256-thread lookback — ensure ALL 256 threads participate (not gated by `lid < NUM_BINS` since NUM_BINS=256=all threads)
    4. Add debug prints if needed (temporary): dump histogram, prefix sums, first 100 output elements
    5. Fix all issues until 1M correctness passes
  - **Files**: `metal-gpu-experiments/shaders/exp16_8bit.metal`, `metal-gpu-experiments/src/exp16_8bit.rs`
  - **Done when**: `ok` printed for 1M correctness check
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release && ./target/release/metal-gpu-experiments 2>&1 | grep -E "(1M|FAIL|ok|mismatch)"` — must show "ok" for 1M, no FAIL
  - **Commit**: `fix(exp16): correct 8-bit radix sort at 1M elements`
  - _Requirements: AC-1.5, AC-5.1_

- [ ] 1.4 [VERIFY] Quality checkpoint: build + correctness at all sizes
  - **Do**:
    1. Run exp16 at all 5 sizes (1M, 2M, 4M, 8M, 16M)
    2. Verify all show "ok"
    3. Check for any compiler warnings
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && cargo build --release 2>&1 | grep -i warning; ./target/release/metal-gpu-experiments 2>&1 | grep -E "(FAIL|ok)" | head -10` — all must show "ok", no FAIL
  - **Done when**: All 5 sizes correct, no compiler warnings
  - **Commit**: `chore(exp16): pass quality checkpoint — all sizes correct` (only if fixes needed)

- [ ] 1.5 Benchmark Step 1 and compare vs exp15
  - **Do**:
    1. Enable both exp15 and exp16 in main.rs (uncomment exp15 call)
    2. Run binary, capture full output
    3. Compare p50 Mkeys/s at 16M: exp16 should be >= 1.3x exp15
    4. Record baseline numbers in commit message for future reference
    5. If exp16 is slower than exp15, investigate:
       - 256-bin atomic histogram overhead (more atomics per element)
       - Tile status buffer SLC misses (3.81MB vs 122KB)
       - Lookback with 256 threads all spinning (vs 16 in exp15)
  - **Files**: `metal-gpu-experiments/src/main.rs`
  - **Done when**: Benchmark numbers printed for both exp15 and exp16 side-by-side
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && cargo build --release && ./target/release/metal-gpu-experiments 2>&1 | grep -E "16M.*Mkeys"` — should show 2 lines (exp15 + exp16), exp16 >= 2400 Mkeys/s
  - **Commit**: `feat(exp16): benchmark 8-bit radix sort step 1 — X Mkeys/s (Y.Zx over exp15)`
  - _Requirements: AC-1.6, AC-6.2, AC-6.4_
  - _Design: Incremental Build Strategy Step 1_

## Phase 2: Feature Complete — 4096 Tiles + Adversarial Inputs

Focus: Increase tile size to 4096 (16 elem/thread), add adversarial input testing, add per-pass timing.

- [ ] 2.1 Increase tile size to 4096 elements (16 elem/thread)
  - **Do**:
    1. In exp16_8bit.metal: change `EXP16_TILE_SIZE` to 4096, `EXP16_ELEMS` to 16
    2. Update load loop indexing: `idx = base + simd_id * 512u + (uint)e * 32u + simd_lane` (512 = 16 elems * 32 lanes)
    3. In exp16_8bit.rs: change `TILE_SIZE` to 4096
    4. Verify num_tiles at 16M = 3906 (16M / 4096)
    5. Update tile_status buffer allocation: num_tiles * 256 * 4
    6. Run correctness check at all sizes
    7. If timing variance spikes >200% (register spill indicator), revert to 2048
  - **Files**: `metal-gpu-experiments/shaders/exp16_8bit.metal`, `metal-gpu-experiments/src/exp16_8bit.rs`
  - **Done when**: All sizes pass correctness with TILE_SIZE=4096
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release && ./target/release/metal-gpu-experiments 2>&1 | grep -E "(FAIL|ok|Mkeys)" | head -10` — all "ok", throughput numbers visible
  - **Commit**: `feat(exp16): increase tile size to 4096 elements (16 elem/thread)`
  - _Requirements: FR-3, AC-2.1, AC-2.2, AC-2.3, AC-2.4_
  - _Design: Incremental Build Strategy Step 2_

- [ ] 2.2 [VERIFY] Quality checkpoint: correctness at 4096 tile size
  - **Do**:
    1. Run full benchmark at 1M, 2M, 4M, 8M, 16M
    2. Verify all pass correctness
    3. Check p95/p5 spread — if >50% at any size, register spill likely
    4. Compare 4096-tile throughput vs 2048-tile (from 1.5 commit message)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && ./target/release/metal-gpu-experiments 2>&1 | grep -E "(FAIL|ok|spread)" | head -10` — all "ok", spread < 50%
  - **Done when**: All sizes correct, no register spill symptoms
  - **Commit**: `chore(exp16): pass quality checkpoint — 4096 tile correctness` (only if fixes needed)

- [ ] 2.3 Add adversarial input tests
  - **Do**:
    1. Add adversarial test cases to `run()` in exp16_8bit.rs:
       - All zeros (vec![0u32; 16M])
       - All ones (vec![0xFFFFFFFFu32; 16M])
       - Already sorted (0..16M as u32)
       - Reverse sorted ((0..16M).rev())
    2. For each: generate input, CPU sort as reference, run GPU sort, verify match
    3. Print "ok" / "FAIL" per adversarial input
    4. Run at 16M only (these are edge-case tests, not benchmarks)
  - **Files**: `metal-gpu-experiments/src/exp16_8bit.rs`
  - **Done when**: All 4 adversarial inputs pass at 16M
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && cargo build --release && ./target/release/metal-gpu-experiments 2>&1 | grep -E "(zeros|ones|sorted|reverse|FAIL|ok)" | head -10` — all show "ok"
  - **Commit**: `test(exp16): add adversarial input tests (zeros, ones, sorted, reverse)`
  - _Requirements: AC-5.2, AC-5.3, AC-5.4, FR-13_
  - _Design: Edge Cases, Test Strategy_

- [ ] 2.4 Add exp15 vs exp16 comparison output
  - **Do**:
    1. In main.rs, ensure both exp15 and exp16 run
    2. In exp16_8bit.rs, add a comparison section at end of run():
       - Run exp15 at 16M (call exp15's bench function or re-run its PSOs)
       - OR: just run both experiments and let user compare console output
       - Print explicit "Speedup: exp16/exp15 = X.XXx" line
    3. Ensure exp15 module is accessible (may need `pub` on exp15's bench_onesweep or duplicate benchmark call)
  - **Files**: `metal-gpu-experiments/src/exp16_8bit.rs`, `metal-gpu-experiments/src/main.rs`, possibly `metal-gpu-experiments/src/exp15_onesweep.rs`
  - **Done when**: Console output shows side-by-side Mkeys/s comparison with speedup ratio
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && cargo build --release && ./target/release/metal-gpu-experiments 2>&1 | grep -iE "(speedup|exp15.*exp16|comparison)" | head -5`
  - **Commit**: `feat(exp16): add exp15 vs exp16 performance comparison`
  - _Requirements: AC-6.4, FR-16_

- [ ] 2.5 [VERIFY] Quality checkpoint: full feature verification
  - **Do**:
    1. Full build with no warnings
    2. All random sizes correct (1M-16M)
    3. All adversarial inputs correct
    4. Comparison output visible
    5. Verify TG memory usage <= 32KB (check shader compilation — Metal compiler would error if exceeded)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && cargo build --release 2>&1 | grep -c warning && ./target/release/metal-gpu-experiments 2>&1 | grep -c FAIL` — both should output 0
  - **Done when**: Zero warnings, zero failures
  - **Commit**: `chore(exp16): pass quality checkpoint — feature complete` (only if fixes needed)

## Phase 3: Optimize — Block-Wise Scatter (Conditional)

Focus: Only implement if Phase 2 benchmarks show scatter is a bottleneck. Otherwise skip to Phase 4.

- [ ] 3.1 Analyze scatter bottleneck via per-pass timing
  - **Do**:
    1. Add per-pass timing to exp16: encode each pass in separate command buffer (temporarily) to measure individual pass GPU time via `gpu_elapsed_ms()`
    2. OR: instrument with Metal counters (MTLCounterSampleBuffer) if available
    3. Simplest approach: time total of 4 passes, then time histogram+prefix separately
       - Run histogram+prefix alone in one cmd buffer, measure
       - Run full sort in one cmd buffer, measure
       - Partition time per pass ~ (total - hist_prefix) / 4
    4. Within partition: estimate scatter fraction by comparing with a "scatter-disabled" variant (write nothing in Phase 5, just compute positions — measure time delta)
    5. Print per-pass breakdown
  - **Files**: `metal-gpu-experiments/src/exp16_8bit.rs`
  - **Done when**: Per-pass timing visible. Decision made: if scatter > 20% of pass time, proceed to 3.2. Otherwise mark 3.2 as skipped.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && cargo build --release && ./target/release/metal-gpu-experiments 2>&1 | grep -iE "(per.pass|breakdown|scatter|histogram|timing)" | head -10`
  - **Commit**: `feat(exp16): add per-pass timing breakdown for bottleneck analysis`
  - _Requirements: AC-6.3, FR-15_
  - _Design: Performance Considerations_

- [ ] 3.2 Implement block-wise scatter (Option A: half-tile reorder) — CONDITIONAL
  - **Do**:
    1. **Skip this task if 3.1 shows scatter < 20% of pass time**
    2. If proceeding: add `local_keys[2048]` (8KB) TG memory to partition kernel
    3. Replace Phase 5 scatter with 2 sub-passes:
       - Sub-pass 1: rank elements 0-7 into local_keys, barrier, coalesced scatter from local_keys
       - Sub-pass 2: rank elements 8-15 into local_keys, barrier, coalesced scatter from local_keys
    4. Total TG memory: 18KB + 8KB + 1KB (bin_start) = 27KB (fits 32KB)
    5. Verify correctness at all sizes
    6. Benchmark: compare random-scatter vs block-scatter at 16M
    7. If block-scatter overhead > 5% vs random, revert
  - **Files**: `metal-gpu-experiments/shaders/exp16_8bit.metal`
  - **Done when**: Block-wise scatter correct and within 5% overhead target, OR skipped per 3.1 analysis
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release && ./target/release/metal-gpu-experiments 2>&1 | grep -E "(FAIL|ok|Mkeys)" | head -10`
  - **Commit**: `feat(exp16): add block-wise scatter for coalesced writes` OR `chore(exp16): skip block-wise scatter — scatter not bottleneck`
  - _Requirements: FR-7, AC-3.1, AC-3.2, AC-3.3, AC-3.4_
  - _Design: Block-Wise Scatter, Option A_

## Phase 4: Quality Gates

- [ ] 4.1 [VERIFY] Full local CI: build + all correctness checks
  - **Do**:
    1. Clean build from scratch
    2. Run all experiments (exp15 + exp16)
    3. Verify all correctness checks pass
    4. Verify no compiler warnings
    5. Verify benchmark stability (p95/p5 spread < 15% at 16M)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && rm -rf target/release/build/metal-gpu-experiments-* && cargo build --release 2>&1 | grep -c warning && ./target/release/metal-gpu-experiments 2>&1 | grep -c FAIL` — both 0
  - **Done when**: Clean build, zero failures, stable benchmarks
  - **Commit**: `chore(exp16): pass local CI` (if fixes needed)

- [ ] 4.2 Final benchmark: record definitive numbers
  - **Do**:
    1. Close all other apps to minimize interference
    2. Run binary 3 times, take best p50 from each run
    3. Record final numbers:
       - exp15 @ 16M: X Mkeys/s
       - exp16 @ 16M: Y Mkeys/s
       - Speedup: Y/X = Z.ZZx
    4. Record per-size results (1M, 2M, 4M, 8M, 16M)
    5. Record adversarial results
    6. Update .progress.md with final results
  - **Files**: `specs/gpu-sort-3000/.progress.md`
  - **Done when**: Definitive benchmark numbers recorded
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && ./target/release/metal-gpu-experiments 2>&1 | grep -E "16M.*Mkeys"`
  - **Commit**: `docs(exp16): record final benchmark results`
  - _Requirements: AC-6.1, AC-6.2, NFR-1, NFR-7_

- [ ] 4.3 Create PR and verify CI
  - **Do**:
    1. Verify current branch: `git branch --show-current` (should be feat/gpu-query or feature branch)
    2. If on default branch, STOP and alert user
    3. Stage files: exp16_8bit.metal, exp16_8bit.rs, main.rs changes
    4. Push branch: `git push -u origin <branch-name>`
    5. Create PR: `gh pr create --title "feat(exp16): 8-bit radix sort achieving X Mkeys/s" --body "..."`
    6. Include benchmark results in PR body
  - **Verify**: `gh pr checks` or `gh pr checks --watch` — all checks passing
  - **Done when**: PR created, CI green
  - **Commit**: None (PR creation only)

## Phase 5: PR Lifecycle

- [ ] 5.1 Monitor CI and fix failures
  - **Do**:
    1. Watch CI: `gh pr checks --watch`
    2. If failures: read details, fix locally, push
    3. Re-verify: `gh pr checks --watch`
  - **Verify**: `gh pr checks` — all green
  - **Done when**: All CI checks pass

- [ ] 5.2 Address review comments
  - **Do**:
    1. Check for review comments: `gh pr view --comments`
    2. Address each comment with code changes
    3. Push fixes
    4. Re-run correctness + benchmark to ensure no regression
  - **Verify**: `gh pr view --comments` — all comments resolved
  - **Done when**: All review comments addressed, CI still green

- [ ] 5.3 [VERIFY] AC checklist
  - **Do**:
    1. Read requirements.md, verify each AC programmatically:
       - AC-1.1: grep shader for 256-bin histogram (`EXP16_NUM_BINS.*256`)
       - AC-1.2: grep shader for 4 passes (`EXP16_NUM_PASSES.*4`)
       - AC-1.3: grep shader for `atomic_fetch_add` in histogram phase
       - AC-1.4: count TG memory arrays, verify <= 32KB
       - AC-1.5: run binary, verify "ok" at 16M
       - AC-1.6: run binary, check exp16 >= 1.3x exp15
       - AC-2.1: grep for `EXP16_ELEMS.*16` (after Phase 2)
       - AC-2.2: verify num_tiles formula = n/4096
       - AC-5.1: run binary, all sizes "ok"
       - AC-5.2: run binary, adversarial inputs "ok"
       - AC-6.1: grep for p5/p50/p95 in output
       - AC-6.2: grep for Mkeys/s at 16M
    2. Document pass/fail for each AC
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && ./target/release/metal-gpu-experiments 2>&1 | grep -c FAIL` — must be 0
  - **Done when**: All acceptance criteria confirmed met
  - **Commit**: None

## Notes

- **POC shortcuts taken (Phase 1)**:
  - Start with 2048 tile size (same as exp15) to isolate 8-bit radix changes
  - Copy percentile/print_stats instead of making shared module
  - No per-pass timing in Phase 1 (added in Phase 3)
  - No adversarial inputs in Phase 1 (added in Phase 2)

- **Production TODOs (Phase 2+)**:
  - Increase to 4096 tile size (16 elem/thread)
  - Add adversarial input testing
  - Add exp15 comparison output
  - Per-pass timing breakdown

- **Key risks**:
  - Register spill at 16 elem/thread (4096 tiles) — fallback is 2048 tiles
  - 256-bin atomic histogram may be slower than expected — fallback is exp15's butterfly approach adapted
  - Tile status buffer (3.81MB) may cause SLC misses — monitor lookback latency

- **Build command**: Always `rm -rf target/release/build/metal-gpu-experiments-*` before `cargo build --release` when .metal files change

- **Shader compilation**: build.rs already compiles with `-std=metal3.2` and links all .metal into shaders.metallib
