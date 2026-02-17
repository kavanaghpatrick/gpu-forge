---
spec: gpu-compute-experiments
phase: tasks
total_tasks: 52
created: 2026-02-17
---

# Tasks: metal-forge-compute Experiment Suite

## Phase 1: Make It Work (POC) -- Infrastructure + Reduce End-to-End

Focus: Workspace compiles, one experiment (reduce) runs end-to-end producing correct results + table output. Validates Metal dispatch, timing, data gen, and output pipeline.

- [x] 1.1 Create workspace and forge-primitives crate skeleton
  - **Do**:
    1. Create `metal-forge-compute/Cargo.toml` workspace with members `forge-bench`, `forge-primitives`
    2. Create `forge-primitives/Cargo.toml` with deps: objc2 0.6, objc2-metal 0.3, objc2-foundation 0.3, block2 0.6
    3. Create `forge-primitives/src/lib.rs` re-exporting all modules
    4. Create `forge-primitives/src/metal_ctx.rs` -- MetalContext struct: device + queue + library (follow `gpu-query/src/gpu/device.rs` pattern with `MTLCreateSystemDefaultDevice`, `newCommandQueue`, `newLibraryWithFile`)
    5. Create `forge-primitives/src/pso_cache.rs` -- PsoCache with `get_or_create()` (adapt from `gpu-query/src/gpu/pipeline.rs`, simplify: no function constants needed for reduce)
    6. Create `forge-primitives/src/buffer_pool.rs` -- BufferPool with 16KB page-aligned alloc/recycle/peak tracking
    7. Create `forge-primitives/src/dispatch.rs` -- `dispatch_1d`, `dispatch_threads_1d`, `alloc_buffer`, `alloc_buffer_with_data`, `read_buffer`, `read_buffer_slice` (copy patterns from `gpu-query/src/gpu/encode.rs`)
    8. Create `forge-primitives/src/types.rs` -- `#[repr(C)] ReduceParams { element_count: u32, _pad: [u32; 3] }` with layout test
    9. Create `forge-primitives/src/hardware.rs` -- detect chip name via `device.name()`, lookup bandwidth table (M4 Pro=273, M4 Max=546, M4=120)
    10. Create `forge-primitives/src/timing.rs` -- `BenchTimer::start()` / `.stop() -> f64` (wall-clock ms)
  - **Files**: `metal-forge-compute/Cargo.toml`, `forge-primitives/Cargo.toml`, `forge-primitives/src/lib.rs`, `forge-primitives/src/metal_ctx.rs`, `forge-primitives/src/pso_cache.rs`, `forge-primitives/src/buffer_pool.rs`, `forge-primitives/src/dispatch.rs`, `forge-primitives/src/types.rs`, `forge-primitives/src/hardware.rs`, `forge-primitives/src/timing.rs`
  - **Done when**: `cargo check -p forge-primitives` passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check -p forge-primitives`
  - **Commit**: `feat(forge-primitives): create library crate with Metal context, PSO cache, buffer pool`
  - _Requirements: FR-16, NFR-2, NFR-4, NFR-6_
  - _Design: forge-primitives module structure, Buffer Allocation, Timing Infrastructure_

- [ ] 1.2 Metal build.rs + reduce shader + types.h
  - **Do**:
    1. Create `forge-primitives/build.rs` -- xcrun metal -> .air, xcrun metallib -> .metallib (copy from `gpu-query/build.rs`, handle stub case for empty shaders dir)
    2. Create `forge-primitives/shaders/types.h` with `ReduceParams` struct matching Rust types.rs
    3. Create `forge-primitives/shaders/reduce.metal` with 4 kernels:
       - `reduce_sum_u32`: 3-level (simd_sum -> threadgroup shared mem -> atomic_fetch_add)
       - `reduce_sum_f32`: simd_sum(float) -> threadgroup -> CAS loop atomic (pattern from aggregate_sum_float)
       - `reduce_min_u32`: simd_min -> threadgroup -> partials array (CPU final reduction)
       - `reduce_max_u32`: simd_max -> threadgroup -> partials array
       - All use 256 threads/threadgroup, #define MAX_THREADS_PER_TG 256
  - **Files**: `forge-primitives/build.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/reduce.metal`
  - **Done when**: `cargo build -p forge-primitives` compiles shaders to .metallib
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-primitives 2>&1 | grep "Built shaders.metallib"`
  - **Commit**: `feat(forge-primitives): add build.rs Metal compilation and reduce kernel`
  - _Requirements: FR-3_
  - _Design: Reduce kernel architecture, build.rs metal compilation_

- [ ] 1.3 Create forge-bench crate with CLI, config, stats, data_gen
  - **Do**:
    1. Create `forge-bench/Cargo.toml` with deps: forge-primitives (path), clap 4 (derive), serde 1 (derive), serde_json 1, toml 0.8, comfy-table 7, colored 2, indicatif 0.17, rayon 1, rand 0.8, chrono 0.4
    2. Create `forge-bench/src/cli.rs` -- clap derive: `ForgeArgs` with subcommands `Run { experiments, sizes, runs, warmup, profile, json_file, csv_file }` and `List`
    3. Create `forge-bench/src/config.rs` -- profiles (quick: 1M/3/1, standard: 10M/10/3, thorough: 100M/30/3), TOML loading, size parser (1M=1000000, 100K=100000)
    4. Create `forge-bench/src/stats.rs` -- `compute_stats(samples: &[f64]) -> Stats { mean, median, min, max, stddev, cv_percent }` with IQR outlier detection
    5. Create `forge-bench/src/data_gen.rs` -- `DataGenerator::new(seed)` with `uniform_u32(count)`, `uniform_f32(count)` using `StdRng::seed_from_u64`
    6. Create `forge-bench/src/main.rs` -- parse CLI, dispatch to harness (stub)
  - **Files**: `forge-bench/Cargo.toml`, `forge-bench/src/main.rs`, `forge-bench/src/cli.rs`, `forge-bench/src/config.rs`, `forge-bench/src/stats.rs`, `forge-bench/src/data_gen.rs`
  - **Done when**: `cargo check -p forge-bench` passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check -p forge-bench`
  - **Commit**: `feat(forge-bench): scaffold CLI, config profiles, stats, data generation`
  - _Requirements: FR-16, NFR-1, NFR-5, NFR-7_
  - _Design: forge-bench module structure, Data Generation, CLI_

- [ ] 1.4 Experiment trait + reduce experiment + CPU baseline
  - **Do**:
    1. Create `forge-bench/src/experiments/mod.rs` -- define `Experiment` trait:
       ```rust
       pub trait Experiment {
           fn name(&self) -> &str;
           fn description(&self) -> &str;
           fn supported_sizes(&self) -> Vec<usize>;
           fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator);
           fn run_gpu(&mut self, ctx: &MetalContext) -> f64; // ms
           fn run_cpu(&mut self) -> f64; // ms
           fn validate(&self) -> Result<(), String>;
           fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64>;
       }
       ```
       Plus `pub fn all_experiments() -> Vec<Box<dyn Experiment>>` registry
    2. Create `forge-bench/src/experiments/reduce.rs` -- implements Experiment for ReduceExperiment:
       - `setup`: generate uniform_u32/f32 data, create Metal buffers, create output buffers (atomic for sum, partials array for min/max)
       - `run_gpu`: create command buffer, encode reduce kernel, commit, waitUntilCompleted, read back result. Time with BenchTimer
       - `run_cpu`: rayon `par_iter().sum()` / `par_iter().min()` / `par_iter().max()`
       - `validate`: compare GPU vs CPU results (exact for u32, 1e-3 relative for f32)
       - `metrics`: GB/s = (N * 4 bytes) / (elapsed_ms / 1000) / 1e9, bandwidth_utilization = GB/s / hardware_bw
    3. Create `forge-bench/src/cpu_baselines/mod.rs` + `forge-bench/src/cpu_baselines/rayon_reduce.rs`
  - **Files**: `forge-bench/src/experiments/mod.rs`, `forge-bench/src/experiments/reduce.rs`, `forge-bench/src/cpu_baselines/mod.rs`, `forge-bench/src/cpu_baselines/rayon_reduce.rs`
  - **Done when**: `cargo check -p forge-bench` passes with experiment module
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check -p forge-bench`
  - **Commit**: `feat(forge-bench): add Experiment trait, reduce experiment, rayon CPU baseline`
  - _Requirements: FR-3, AC-1.3, AC-1.6, AC-1.7, NFR-3_
  - _Design: Experiment trait, Reduce kernel, CPU Baseline Implementation_

- [ ] 1.5 [VERIFY] Quality checkpoint: cargo build workspace
  - **Do**: Build entire workspace, fix any compile errors
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build 2>&1 | tail -5`
  - **Done when**: Workspace builds with no errors
  - **Commit**: `chore(forge): pass quality checkpoint` (only if fixes needed)

- [ ] 1.6 Harness measurement loop + table output
  - **Do**:
    1. Create `forge-bench/src/harness.rs` -- `run_experiment(exp, config, ctx) -> Vec<DataPoint>`:
       - For each size: setup -> validate() -> warmup loop (discard) -> measured loop (GPU + CPU) -> compute_stats -> DataPoint
       - DataPoint: { experiment, size, gpu_stats, cpu_stats, speedup, metrics }
    2. Create `forge-bench/src/output/mod.rs`, `forge-bench/src/output/table.rs` -- render comfy-table with columns: Size | GPU (ms) | CPU (ms) | Speedup | GB/s | BW% | CV%
    3. Create `forge-bench/src/output/json.rs` -- serialize Vec<DataPoint> to JSON with hardware info header
    4. Create `forge-bench/src/output/progress.rs` -- indicatif progress bar during measurement
    5. Wire main.rs: parse CLI -> select experiments -> create MetalContext -> run harness -> output table/json
  - **Files**: `forge-bench/src/harness.rs`, `forge-bench/src/output/mod.rs`, `forge-bench/src/output/table.rs`, `forge-bench/src/output/json.rs`, `forge-bench/src/output/progress.rs`, `forge-bench/src/main.rs` (update)
  - **Done when**: `forge-bench reduce --sizes 1M` runs and prints table
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- reduce --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge-bench): add measurement harness, table output, JSON export`
  - _Requirements: FR-16, NFR-1, NFR-7_
  - _Design: Data Flow, Harness measurement loop_

- [ ] 1.7 POC Checkpoint: reduce experiment end-to-end
  - **Do**:
    1. Run reduce at 1M with quick profile, verify table prints correctly
    2. Run reduce at 10M to see real speedup numbers
    3. Verify JSON output with --json-file flag
    4. Confirm GPU result matches CPU (validate passes)
    5. Confirm CV < 5% (NFR-1)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- reduce --sizes 1M,10M --runs 5 --warmup 2 --json-file /tmp/reduce_poc.json && cat /tmp/reduce_poc.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Experiments: {len(d)}'); [print(f'{r[\"size\"]}: GPU={r[\"gpu_stats\"][\"mean_ms\"]:.2f}ms CPU={r[\"cpu_stats\"][\"mean_ms\"]:.2f}ms Speedup={r[\"speedup\"]:.1f}x') for r in d]"`
  - **Done when**: Reduce produces correct results at 1M and 10M with speedup > 1x at 10M
  - **Commit**: `feat(forge-bench): complete POC -- reduce experiment validated end-to-end`
  - _Requirements: FR-3, FR-16, AC-1.3, AC-1.6, AC-1.7_
  - _Design: Implementation Steps 1-10_

## Phase 2: Foundation Primitives (scan, compact, sort, histogram)

Focus: Complete all P0 primitive kernels. Each follows the pattern: write shader -> implement experiment -> add CPU baseline -> verify correctness.

- [ ] 2.1 Prefix scan kernel + experiment
  - **Do**:
    1. Add `ScanParams` to `forge-primitives/src/types.rs` and `shaders/types.h`: `{ element_count: u32, pass: u32, _pad: [u32; 2] }`
    2. Create `forge-primitives/shaders/scan_helpers.h` -- Blelloch threadgroup scan utility
    3. Create `forge-primitives/shaders/scan.metal` with 3 kernels:
       - `scan_local`: each threadgroup scans its chunk (512 elements, Blelloch), writes partial sum to `partials[]`
       - `scan_partials`: single threadgroup scans partials array
       - `scan_add_offsets`: each threadgroup adds its prefix from `partials[]` to every element
    4. Create `forge-bench/src/experiments/scan.rs` -- ScanExperiment:
       - `run_gpu`: encode all 3 passes in single command buffer. Read back exclusive prefix scan
       - `run_cpu`: sequential `iter::scan` accumulator
       - `validate`: exact match for u32
       - `metrics`: GB/s = (N * 4 * 2 [read+write]) / elapsed
    5. Create `forge-bench/src/cpu_baselines/sequential.rs` -- sequential scan, histogram baselines
    6. Register scan in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/scan_helpers.h`, `forge-primitives/shaders/scan.metal`, `forge-bench/src/experiments/scan.rs`, `forge-bench/src/cpu_baselines/sequential.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench scan --sizes 1M --runs 3` produces correct results
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- scan --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add 3-pass prefix scan kernel and experiment`
  - _Requirements: FR-2, AC-1.2_
  - _Design: Prefix Scan 3-pass, Implementation Step 11_

- [ ] 2.2 Stream compaction kernel + experiment
  - **Do**:
    1. Add `CompactParams` to types: `{ element_count: u32, threshold: u32, _pad: [u32; 2] }`
    2. Create `forge-primitives/shaders/compact_scan.metal` with 2 kernels:
       - `compact_flags`: evaluate predicate (value > threshold), write 0/1 to flags array
       - `compact_scatter`: if flags[i]==1, write input[i] to output[scan[i]]
       - Reuse scan kernels from scan.metal for the prefix scan step
    3. Create `forge-bench/src/experiments/compact.rs` -- CompactExperiment:
       - `setup`: generate data, set threshold for 10%/50%/90% selectivity
       - `run_gpu`: flags -> scan -> scatter (3 command buffer dispatches in 1 cmdbuf)
       - `run_cpu`: rayon `par_iter().filter().collect()`
       - `validate`: exact set equality (sort both, compare)
       - `metrics`: elements/sec, selectivity rate
    4. Create `forge-bench/src/cpu_baselines/rayon_filter.rs` -- parallel filter baseline
    5. Register compact in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/compact_scan.metal`, `forge-bench/src/experiments/compact.rs`, `forge-bench/src/cpu_baselines/rayon_filter.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench compact --sizes 1M --runs 3` produces correct results at 10%/50%/90% selectivity
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- compact --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add scan-based stream compaction kernel and experiment`
  - _Requirements: FR-5, AC-1.5_
  - _Design: Stream Compaction, Implementation Step 12_

- [ ] 2.3 [VERIFY] Quality checkpoint: cargo build + scan/compact correctness
  - **Do**: Build workspace, verify scan and compact produce correct output
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build 2>&1 | tail -3 && cargo run -p forge-bench -- scan --sizes 1M --runs 1 --warmup 0 && cargo run -p forge-bench -- compact --sizes 1M --runs 1 --warmup 0`
  - **Done when**: All experiments pass validation, no compile errors
  - **Commit**: `chore(forge): pass quality checkpoint` (only if fixes needed)

- [ ] 2.4 Radix sort kernel + experiment
  - **Do**:
    1. Ensure SortParams already in types: `{ element_count: u32, bit_offset: u32, descending: u32, _pad: u32 }`
    2. Create `forge-primitives/shaders/radix_sort.metal` with 2 kernels:
       - `radix_histogram`: per-threadgroup 16-bin histogram via shared memory atomics. Write local histograms to global array [num_tg x 16]
       - `radix_scatter`: read key, compute 4-bit digit, lookup scatter position from scanned histogram, write to output buffer
       - 8 passes for u32 (4 bits per pass). Double-buffer ping-pong
       - f32 support: XOR sign bit transformation for radix-sortable order
    3. Create `forge-bench/src/experiments/sort.rs` -- SortExperiment:
       - `run_gpu`: for each of 8 passes: encode histogram -> scan histogram -> scatter (all in 1 cmdbuf). Ping-pong buffers
       - `run_cpu`: `std::sort_unstable` + rayon `par_sort_unstable`
       - `validate`: exact element equality with sorted reference
       - `metrics`: elements/sec = N / elapsed_sec
    4. Create `forge-bench/src/cpu_baselines/rayon_sort.rs` -- parallel sort baseline
    5. Register sort in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/radix_sort.metal`, `forge-bench/src/experiments/sort.rs`, `forge-bench/src/cpu_baselines/rayon_sort.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench sort --sizes 1M --runs 3` sorts correctly, elements/sec reported
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- sort --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add reduce-then-scan radix sort kernel and experiment`
  - _Requirements: FR-1, AC-1.1, AC-1.6_
  - _Design: Radix Sort, Implementation Step 13_

- [ ] 2.5 Histogram kernel + experiment
  - **Do**:
    1. Add `HistogramParams` to types: `{ element_count: u32, num_bins: u32, _pad: [u32; 2] }`
    2. Create `forge-primitives/shaders/histogram.metal`:
       - `histogram_256`: shared-memory 256-bin histogram. 1KB threadgroup memory. Init -> accumulate -> merge to global
       - `histogram_65536`: tiled approach -- each simdgroup maintains 256 bins, tile across 256 sub-ranges. If threadgroup memory exceeds 32KB, fall back to multiple passes
    3. Create `forge-bench/src/experiments/histogram.rs` -- HistogramExperiment:
       - `run_gpu`: dispatch histogram kernel, read global histogram buffer
       - `run_cpu`: sequential Vec<u32> bin accumulation
       - `validate`: exact histogram bin equality
       - `metrics`: elements/sec, bins/sec
    4. Register histogram in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/histogram.metal`, `forge-bench/src/experiments/histogram.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench histogram --sizes 1M --runs 3` produces correct 256-bin and 65536-bin histograms
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- histogram --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add shared-memory histogram kernel and experiment`
  - _Requirements: FR-4, AC-1.4_
  - _Design: Histogram, Implementation Step 14_

- [ ] 2.6 [VERIFY] Quality checkpoint: all 5 foundation primitives
  - **Do**: Run all 5 experiments (reduce, scan, compact, sort, histogram) at 1M, verify all pass validation
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build && for exp in reduce scan compact sort histogram; do echo "=== $exp ===" && cargo run -p forge-bench -- $exp --sizes 1M --runs 1 --warmup 0 || exit 1; done`
  - **Done when**: All 5 experiments produce correct results
  - **Commit**: `chore(forge): pass quality checkpoint -- all foundation primitives validated`

- [ ] 2.7 Output: CSV, summary, roofline
  - **Do**:
    1. Create `forge-bench/src/output/csv.rs` -- write DataPoint vec to CSV file
    2. Create `forge-bench/src/output/summary.rs` -- print summary table: experiment | size | speedup | verdict (>5x=STRONG, 2-5x=SOLID, 1-2x=MARGINAL, <1x=SLOWER)
    3. Create `forge-bench/src/output/roofline.rs` -- ASCII roofline diagram showing bandwidth utilization % for each experiment
    4. Add `--csv-file` flag to CLI
    5. Wire all outputs in main.rs
  - **Files**: `forge-bench/src/output/csv.rs`, `forge-bench/src/output/summary.rs`, `forge-bench/src/output/roofline.rs`, `forge-bench/src/cli.rs`, `forge-bench/src/main.rs`
  - **Done when**: Running all experiments produces summary + CSV + JSON
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- reduce --sizes 1M --runs 3 --warmup 1 --json-file /tmp/test.json --csv-file /tmp/test.csv && test -f /tmp/test.json && test -f /tmp/test.csv`
  - **Commit**: `feat(forge-bench): add CSV export, summary verdicts, ASCII roofline output`
  - _Requirements: FR-16, FR-17_
  - _Design: Output module, Implementation Step 15_

- [ ] 2.8 Run foundation primitives at 10M benchmark
  - **Do**:
    1. Run all 5 primitives at 1M, 10M sizes with standard profile (10 runs, 3 warmup)
    2. Save results to `results/phase1_foundation.json`
    3. Check crossover points: where GPU first beats CPU
    4. Verify reduce bandwidth utilization > 50%
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && mkdir -p results && cargo run -p forge-bench -- reduce scan compact sort histogram --sizes 1M,10M --runs 10 --warmup 3 --json-file results/phase1_foundation.json && python3 -c "import json; d=json.load(open('results/phase1_foundation.json')); [print(f'{r[\"experiment\"]}@{r[\"size\"]}: {r[\"speedup\"]:.1f}x') for r in d]"`
  - **Done when**: All 5 experiments have data at 1M and 10M, speedup numbers reported
  - **Commit**: `perf(forge): benchmark foundation primitives at 1M and 10M`
  - _Requirements: AC-1.1, AC-1.2, AC-1.3, AC-1.4, AC-1.5, AC-1.6_
  - _Design: Phase 1 Foundation Primitives_

## Phase 3: Query Operations (filter, group-by, GEMM, pipeline, DuckDB)

Focus: Relational operations and competitive benchmarks. Validate product direction.

- [ ] 3.1 Filter kernel + experiment
  - **Do**:
    1. Add `FilterBenchParams` to types: `{ element_count: u32, threshold: u32, selectivity_pct: u32, _pad: u32 }`
    2. Create `forge-primitives/shaders/filter_bench.metal` -- columnar filter kernel adapted from gpu-query/shaders/filter.metal:
       - Input: u32 column, threshold value
       - Output: bitmask (1 bit per element)
       - Uses simd_sum for efficient match counting
       - Function constant for compare op (GT/LT/EQ)
    3. Create `forge-bench/src/experiments/filter.rs` -- FilterExperiment:
       - `setup`: generate data with target selectivity (1%, 10%, 50%, 90%)
       - `run_gpu`: filter kernel + read bitmask + popcount for match count
       - `run_cpu`: rayon `par_iter().filter().count()`
       - `validate`: bitmask match count equals CPU count
       - `metrics`: rows/sec, selectivity
    4. Register filter in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/filter_bench.metal`, `forge-bench/src/experiments/filter.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench filter --sizes 1M --runs 3` works at all selectivity rates
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- filter --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add columnar filter kernel and experiment`
  - _Requirements: FR-8, AC-3.1_
  - _Design: Filter kernel, Implementation Step 16_

- [ ] 3.2 Group-by aggregate kernel + experiment
  - **Do**:
    1. Add `GroupByParams` to types: `{ element_count: u32, num_groups: u32, _pad: [u32; 2] }`
    2. Create `forge-primitives/shaders/groupby.metal`:
       - `groupby_boundary_detect`: compare adjacent sorted keys, flag[i] = (key[i] != key[i-1])
       - `groupby_segmented_reduce`: within each segment (group), compute sum/count/min/max of value column
    3. Create `forge-bench/src/experiments/groupby.rs` -- GroupByExperiment:
       - `setup`: generate key column with target cardinality (10, 1K, 100K, 1M groups), value column
       - `run_gpu`: sort keys -> boundary detect -> segmented reduce (all 1 cmdbuf, reuse sort)
       - `run_cpu`: HashMap<u32, (sum, count, min, max)>
       - `validate`: per-group aggregates match within 1e-3
    4. Create `forge-bench/src/cpu_baselines/hashmap_ops.rs` -- HashMap group-by baseline
    5. Register groupby in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/groupby.metal`, `forge-bench/src/experiments/groupby.rs`, `forge-bench/src/cpu_baselines/hashmap_ops.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench groupby --sizes 1M --runs 3` produces correct per-group aggregates
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- groupby --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add sort-based group-by aggregate kernel and experiment`
  - _Requirements: FR-9, AC-3.2_
  - _Design: Group-By Aggregate, Implementation Step 16_

- [ ] 3.3 [VERIFY] Quality checkpoint: filter + groupby correctness
  - **Do**: Build workspace, verify filter and groupby produce correct output
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build && cargo run -p forge-bench -- filter --sizes 1M --runs 1 --warmup 0 && cargo run -p forge-bench -- groupby --sizes 1M --runs 1 --warmup 0`
  - **Done when**: Both experiments pass validation, no compile errors
  - **Commit**: `chore(forge): pass quality checkpoint` (only if fixes needed)

- [ ] 3.4 GEMM kernel + experiment
  - **Do**:
    1. Add `GemmParams` to types: `{ M: u32, N: u32, K: u32, _pad: u32 }`
    2. Create `forge-primitives/shaders/gemm.metal`:
       - `gemm_simdgroup_f32`: simdgroup_matrix 8x8 tiled GEMM for FP32
       - `gemm_simdgroup_f16`: simdgroup_matrix<half, 8, 8> for FP16
       - Each simdgroup computes one 8x8 output tile, iterating K in steps of 8
       - Threadgroup tiling: 32x32 tiles with shared memory staging for 1024+ sizes
    3. Create `forge-bench/src/experiments/gemm.rs` -- GemmExperiment:
       - `setup`: generate random FP32/FP16 matrices at 256x256, 1024x1024, 4096x4096
       - `run_gpu`: dispatch 2D threadgroups, read back C matrix
       - `run_cpu`: Accelerate cblas_sgemm via FFI
       - `validate`: relative error < 1e-4 (FP32), < 1e-2 (FP16)
       - `metrics`: GFLOPS = 2*M*N*K / elapsed_sec / 1e9, % of theoretical peak
    4. Create `forge-bench/src/cpu_baselines/accelerate.rs` -- Accelerate BLAS FFI (cblas_sgemm, cblas_sgemv)
    5. Register gemm in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/gemm.metal`, `forge-bench/src/experiments/gemm.rs`, `forge-bench/src/cpu_baselines/accelerate.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench gemm --sizes 256,1024` produces GFLOPS numbers
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- gemm --sizes 256,1024 --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add simdgroup_matrix GEMM kernel and experiment`
  - _Requirements: FR-6, AC-2.1, AC-2.3, AC-2.4_
  - _Design: GEMM kernel, Implementation Step 16_

- [ ] 3.5 End-to-end pipeline experiment
  - **Do**:
    1. Create `forge-bench/src/experiments/pipeline.rs` -- PipelineExperiment:
       - `setup`: generate columnar data (key col u32 + value col f32) at 10M/100M rows
       - `run_gpu`: single command buffer encoding: filter -> compact -> sort (group keys) -> segmented reduce -> sort (result) -> topK (truncate). All in 1 cmdbuf
       - `run_cpu`: idiomatic Rust: `iter().filter().map()` -> HashMap group-by -> sort by value -> take(K)
       - `validate`: top-K results match
       - `metrics`: total wall-clock ms, per-stage breakdown (% in filter, sort, groupby)
    2. Register pipeline in experiments/mod.rs
  - **Files**: `forge-bench/src/experiments/pipeline.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench pipeline --sizes 1M --runs 3` completes and reports per-stage breakdown
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- pipeline --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add end-to-end analytical pipeline experiment`
  - _Requirements: FR-14, AC-5.1, AC-5.2, AC-5.4, AC-5.5_
  - _Design: End-to-End Pipeline, Implementation Step 17_

- [ ] 3.6 DuckDB comparison experiment
  - **Do**:
    1. Create `forge-bench/src/experiments/duckdb.rs` -- DuckDbExperiment:
       - `setup`: generate same columnar data as pipeline, write to Parquet file using `parquet` crate (or CSV as fallback)
       - `run_gpu`: same pipeline as pipeline experiment
       - `run_cpu`: shell out to `duckdb` CLI: `SELECT key, SUM(value) FROM read_parquet('...') WHERE value > threshold GROUP BY key ORDER BY 2 DESC LIMIT K`
       - Time DuckDB via `Command::new("duckdb")` wall-clock
       - `validate`: DuckDB result matches GPU top-K
       - `metrics`: gpu_ms, duckdb_ms, ratio
    2. Create `forge-bench/src/cpu_baselines/duckdb_runner.rs` -- DuckDB CLI wrapper
    3. Register duckdb in experiments/mod.rs
    4. Handle DuckDB not installed: skip with warning
  - **Files**: `forge-bench/src/experiments/duckdb.rs`, `forge-bench/src/cpu_baselines/duckdb_runner.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench duckdb --sizes 1M --runs 3` runs (or gracefully skips if duckdb not installed)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- duckdb --sizes 1M --runs 3 --warmup 1 || echo "DuckDB experiment ran (may skip if duckdb not installed)"`
  - **Commit**: `feat(forge): add DuckDB comparison experiment`
  - _Requirements: FR-15, AC-5.3_
  - _Design: DuckDB Comparison, Implementation Step 17_

- [ ] 3.7 [VERIFY] Quality checkpoint: all Phase 2 experiments
  - **Do**: Build workspace, run all Phase 2 experiments at 1M
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build && for exp in filter groupby gemm pipeline; do echo "=== $exp ===" && cargo run -p forge-bench -- $exp --sizes 1M --runs 1 --warmup 0 || exit 1; done`
  - **Done when**: All experiments produce correct results
  - **Commit**: `chore(forge): pass quality checkpoint -- query operations validated`

- [ ] 3.8 Run Phase 2 query operations at 10M benchmark
  - **Do**:
    1. Run filter, groupby, gemm, pipeline at 1M, 10M with standard profile
    2. Run duckdb comparison at 1M, 10M (if duckdb installed)
    3. Save results to `results/phase2_query_ops.json`
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- filter groupby gemm pipeline --sizes 1M,10M --runs 10 --warmup 3 --json-file results/phase2_query_ops.json && python3 -c "import json; d=json.load(open('results/phase2_query_ops.json')); [print(f'{r[\"experiment\"]}@{r[\"size\"]}: {r[\"speedup\"]:.1f}x') for r in d]"`
  - **Done when**: Phase 2 speedup numbers collected
  - **Commit**: `perf(forge): benchmark query operations at 1M and 10M`
  - _Requirements: AC-3.1, AC-3.2, AC-2.1, AC-5.1_
  - _Design: Phase 2 Query Operations_

## Phase 4: Consumer & Completion (spreadsheet, timeseries, JSON, hash join, GEMV)

Focus: Product exploration experiments + remaining kernels.

- [ ] 4.1 Spreadsheet formulas kernel + experiment
  - **Do**:
    1. Add `SpreadsheetParams` to types: `{ rows: u32, cols: u32, formula_type: u32, _pad: u32 }`
    2. Create `forge-primitives/shaders/spreadsheet.metal`:
       - `spreadsheet_sum`: column SUM -- reuse reduce pattern per column range
       - `spreadsheet_average`: SUM/COUNT
       - `spreadsheet_vlookup`: binary search in sorted lookup column, each thread searches for its key
    3. Create `forge-bench/src/experiments/spreadsheet.rs` -- SpreadsheetExperiment:
       - `setup`: generate grid of 1M+ cells (1000 rows x 1000 cols), sorted lookup table
       - `run_gpu`: dispatch formula kernel
       - `run_cpu`: sequential loop over cells
       - `validate`: results match within f32 epsilon
       - `metrics`: cells/sec, wall-clock latency
    4. Register spreadsheet in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/spreadsheet.metal`, `forge-bench/src/experiments/spreadsheet.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench spreadsheet --sizes 1M --runs 3` reports cells/sec
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- spreadsheet --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add spreadsheet formula eval kernel and experiment`
  - _Requirements: FR-11, AC-4.1, AC-4.4_
  - _Design: Spreadsheet Formulas, Implementation Step 18_

- [ ] 4.2 Time series analytics kernel + experiment
  - **Do**:
    1. Add `TimeSeriesParams` to types: `{ tick_count: u32, window_size: u32, op_type: u32, _pad: u32 }`
    2. Create `forge-primitives/shaders/timeseries.metal`:
       - `timeseries_moving_avg`: sliding window sum via shared memory (threadgroup caches window elements)
       - `timeseries_vwap`: sum(price*volume) / sum(volume) per window
       - `timeseries_bollinger`: moving mean + moving stddev (Welford online)
    3. Create `forge-bench/src/experiments/timeseries.rs` -- TimeSeriesExperiment:
       - `setup`: generate sorted timestamps + prices + volumes (DataGenerator::time_series)
       - `run_gpu`: dispatch timeseries kernel
       - `run_cpu`: sequential sliding window
       - `validate`: relative error < 1e-3
       - `metrics`: ticks/sec, wall-clock latency
    4. Add `time_series()` generator to data_gen.rs
    5. Register timeseries in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/timeseries.metal`, `forge-bench/src/experiments/timeseries.rs`, `forge-bench/src/data_gen.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench timeseries --sizes 1M --runs 3` reports ticks/sec
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- timeseries --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add time series analytics kernel and experiment`
  - _Requirements: FR-12, AC-4.2, AC-4.4_
  - _Design: Time Series, Implementation Step 18_

- [ ] 4.3 [VERIFY] Quality checkpoint: spreadsheet + timeseries
  - **Do**: Build workspace, verify new experiments pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build && cargo run -p forge-bench -- spreadsheet --sizes 1M --runs 1 --warmup 0 && cargo run -p forge-bench -- timeseries --sizes 1M --runs 1 --warmup 0`
  - **Done when**: Both experiments pass validation
  - **Commit**: `chore(forge): pass quality checkpoint` (only if fixes needed)

- [ ] 4.4 JSON/CSV parsing experiment
  - **Do**:
    1. Create `forge-bench/src/experiments/json_parse.rs` -- JsonParseExperiment:
       - Reuse existing `gpu-query/shaders/csv_parse.metal` and `json_parse.metal` patterns
       - For this experiment, implement a simplified GPU CSV parser in `shaders/csv_bench.metal`:
         - Parallel newline detection via scan
         - Per-field extraction kernel
       - `run_gpu`: dispatch newline scan -> field extraction
       - `run_cpu`: csv crate Reader for CSV, serde_json for JSON
       - `validate`: parsed field count matches
       - `metrics`: MB/s throughput, rows/sec
    2. Add `json_records()` and `csv_records()` generators to data_gen.rs
    3. Add csv crate to forge-bench dependencies
    4. Register json_parse in experiments/mod.rs
  - **Files**: `forge-primitives/shaders/csv_bench.metal`, `forge-bench/src/experiments/json_parse.rs`, `forge-bench/src/data_gen.rs`, `forge-bench/Cargo.toml`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench json_parse --sizes 1M --runs 3` reports MB/s
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- json_parse --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add JSON/CSV parsing experiment`
  - _Requirements: FR-13, AC-4.3_
  - _Design: JSON/CSV Parsing, Implementation Step 18-19_

- [ ] 4.5 Hash join kernel + experiment
  - **Do**:
    1. Add `HashJoinParams` to types: `{ build_count: u32, probe_count: u32, table_size: u32, _pad: u32 }`
    2. Create `forge-primitives/shaders/hash_join.metal`:
       - `hash_join_build`: hash build keys, insert into open-addressing hash table with atomic CAS
       - `hash_join_probe`: each thread hashes its probe key, linear-probes hash table, writes match pairs
    3. Create `forge-bench/src/experiments/hash_join.rs` -- HashJoinExperiment:
       - `setup`: generate build table (1M) and probe table (1M, 10M) with varying join selectivity
       - `run_gpu`: build -> probe (2 dispatches in 1 cmdbuf)
       - `run_cpu`: HashMap build + probe
       - `validate`: join result set matches
       - `metrics`: joins/sec, build_ms, probe_ms
    4. Register hash_join in experiments/mod.rs
  - **Files**: `forge-primitives/src/types.rs`, `forge-primitives/shaders/types.h`, `forge-primitives/shaders/hash_join.metal`, `forge-bench/src/experiments/hash_join.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench hash_join --sizes 1M --runs 3` reports joins/sec
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- hash_join --sizes 1M --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add hash join kernel and experiment`
  - _Requirements: FR-10, AC-3.3_
  - _Design: Hash Join, Implementation Step 18_

- [ ] 4.6 GEMV kernel + experiment
  - **Do**:
    1. Create `forge-primitives/shaders/gemv.metal`:
       - `gemv_f32`: each thread computes one output element via dot product of matrix row with input vector
       - Vectorized loads (float4) for bandwidth optimization
       - Adapted from inference pipeline matvec_f32.metal
    2. Create `forge-bench/src/experiments/gemv.rs` -- GemvExperiment:
       - `setup`: generate matrix (768x768, 2048x768, etc.) + input vector
       - `run_gpu`: dispatch GEMV kernel
       - `run_cpu`: Accelerate cblas_sgemv via FFI (reuse accelerate.rs)
       - `validate`: relative error < 1e-4
       - `metrics`: GB/s = (M*N*4 + M*4 + N*4) / elapsed, % of bandwidth peak
    3. Register gemv in experiments/mod.rs
  - **Files**: `forge-primitives/shaders/gemv.metal`, `forge-bench/src/experiments/gemv.rs`, `forge-bench/src/experiments/mod.rs`
  - **Done when**: `forge-bench gemv --sizes 768 --runs 3` reports GB/s
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- gemv --sizes 768 --runs 3 --warmup 1`
  - **Commit**: `feat(forge): add GEMV kernel and experiment`
  - _Requirements: FR-7, AC-2.2, AC-2.5_
  - _Design: GEMV kernel, Implementation Step 18-19_

- [ ] 4.7 [VERIFY] Quality checkpoint: all 16 experiments compile and run
  - **Do**: Build workspace, run every experiment at 1M with 1 run
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build && for exp in reduce scan compact sort histogram filter groupby gemm gemv pipeline duckdb spreadsheet timeseries json_parse hash_join; do echo "=== $exp ===" && cargo run -p forge-bench -- $exp --sizes 1M --runs 1 --warmup 0 2>&1 | tail -3 || echo "SKIPPED: $exp"; done`
  - **Done when**: All 15 non-DuckDB experiments pass; DuckDB may skip if not installed
  - **Commit**: `chore(forge): pass quality checkpoint -- all 16 experiments validated`

- [ ] 4.8 Add `all` command to run full suite
  - **Do**:
    1. Add `All` subcommand to CLI that runs every experiment
    2. Add `--profile` flag (quick/standard/thorough) that sets sizes/runs/warmup
    3. Add suite-level summary output after all experiments complete
    4. Add crossover-point analysis: for each experiment, find smallest N where GPU > CPU
  - **Files**: `forge-bench/src/cli.rs`, `forge-bench/src/main.rs`, `forge-bench/src/output/summary.rs`
  - **Done when**: `forge-bench all --profile quick` runs all experiments
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- all --profile quick 2>&1 | tail -20`
  - **Commit**: `feat(forge-bench): add 'all' command with profile support and crossover analysis`
  - _Requirements: FR-16, FR-17_
  - _Design: CLI, Summary_

## Phase 5: Run All Experiments + Collect Data

Focus: Execute full benchmark suite, collect machine-readable results.

- [ ] 5.1 Run Phase 3 consumer experiments at 1M/10M
  - **Do**:
    1. Run spreadsheet, timeseries, json_parse, hash_join, gemv at 1M, 10M
    2. Save results to `results/phase3_consumer.json`
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run -p forge-bench -- spreadsheet timeseries json_parse hash_join gemv --sizes 1M,10M --runs 10 --warmup 3 --json-file results/phase3_consumer.json && python3 -c "import json; d=json.load(open('results/phase3_consumer.json')); [print(f'{r[\"experiment\"]}@{r[\"size\"]}: {r[\"speedup\"]:.1f}x') for r in d]"`
  - **Done when**: All Phase 3 speedup numbers collected
  - **Commit**: `perf(forge): benchmark consumer experiments at 1M and 10M`
  - _Requirements: AC-4.1, AC-4.2, AC-4.3, AC-2.2, AC-3.3_

- [ ] 5.2 Full suite run with standard profile
  - **Do**:
    1. Run `forge-bench all --profile standard --json-file results/full_run.json`
    2. This runs all 16 experiments at standard sizes (10M) with 10 measured runs + 3 warmup
    3. Also generate CSV: `--csv-file results/full_run.csv`
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- all --profile standard --json-file results/full_run.json --csv-file results/full_run.csv && python3 -c "import json; d=json.load(open('results/full_run.json')); exps=set(r['experiment'] for r in d); print(f'Experiments run: {len(exps)}/{16}'); [print(f'{r[\"experiment\"]}@{r[\"size\"]}: GPU={r[\"gpu_stats\"][\"mean_ms\"]:.2f}ms Speedup={r[\"speedup\"]:.1f}x') for r in sorted(d, key=lambda x: x['experiment'])]"`
  - **Done when**: results/full_run.json contains data for all 16 experiments, CV < 5% for most
  - **Commit**: `perf(forge): full benchmark suite run with standard profile`
  - _Requirements: All ACs_

- [ ] 5.3 Run 100M element experiments for primitives
  - **Do**:
    1. Run reduce, scan, compact, sort, histogram at 100M elements (thorough: 30 runs)
    2. This validates GPU advantage at scale and captures roofline behavior
    3. Save to `results/primitives_100M.json`
    4. Monitor memory usage (should be ~400MB per experiment for 100M u32)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- reduce scan compact sort histogram --sizes 100M --runs 10 --warmup 3 --json-file results/primitives_100M.json && python3 -c "import json; d=json.load(open('results/primitives_100M.json')); [print(f'{r[\"experiment\"]}@100M: GPU={r[\"gpu_stats\"][\"mean_ms\"]:.1f}ms CPU={r[\"cpu_stats\"][\"mean_ms\"]:.1f}ms Speedup={r[\"speedup\"]:.1f}x') for r in d]"`
  - **Done when**: 100M speedup numbers collected for all 5 primitives
  - **Commit**: `perf(forge): benchmark primitives at 100M elements`
  - _Requirements: AC-1.1, AC-1.2, AC-1.3, AC-1.4, AC-1.5_

## Phase 6: Testing

- [ ] 6.1 Unit tests for forge-primitives
  - **Do**:
    1. Add tests to `forge-primitives/src/types.rs`: layout tests for all param structs (size_of, offset_of)
    2. Add tests to `forge-primitives/src/buffer_pool.rs`: page_align correctness, alloc/recycle
    3. Add tests to `forge-primitives/src/hardware.rs`: chip detection returns non-empty string
    4. Add tests to `forge-primitives/src/timing.rs`: timer measures > 0ms
    5. Add tests to `forge-primitives/src/pso_cache.rs`: key equality/inequality
  - **Files**: All forge-primitives src files (add `#[cfg(test)] mod tests` blocks)
  - **Done when**: `cargo test -p forge-primitives` passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-primitives`
  - **Commit**: `test(forge-primitives): add unit tests for types, buffer pool, hardware, timing`
  - _Design: Test Strategy unit tests_

- [ ] 6.2 Unit tests for forge-bench (stats, config, data_gen)
  - **Do**:
    1. Add tests to `forge-bench/src/stats.rs`: CV calculation, median, outlier detection
    2. Add tests to `forge-bench/src/config.rs`: size parser (1M, 100K, 1_000_000), profile loading
    3. Add tests to `forge-bench/src/data_gen.rs`: reproducibility (same seed = same data), correct count
  - **Files**: All forge-bench utility src files
  - **Done when**: `cargo test -p forge-bench` passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-bench`
  - **Commit**: `test(forge-bench): add unit tests for stats, config, data generation`
  - _Design: Test Strategy unit tests_

- [ ] 6.3 [VERIFY] Quality checkpoint: all tests pass
  - **Do**: Run full test suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test --workspace`
  - **Done when**: All tests pass
  - **Commit**: `chore(forge): pass quality checkpoint` (only if fixes needed)

- [ ] 6.4 Integration tests for GPU kernel correctness
  - **Do**:
    1. Create `forge-bench/tests/test_gpu_correctness.rs`:
       - `test_reduce_gpu_correctness_1m`: run reduce at 1M, assert GPU sum == CPU sum
       - `test_scan_gpu_correctness_1m`: run scan at 1M, assert GPU scan == sequential scan
       - `test_sort_gpu_correctness_1m`: run sort at 1M, assert GPU sorted == std::sort
       - `test_compact_gpu_correctness_1m`: run compact at 1M, assert set equality
       - `test_histogram_gpu_correctness_1m`: run histogram at 1M, assert bin counts match
    2. Each test creates MetalContext, generates data with fixed seed, runs GPU, compares to CPU reference
  - **Files**: `forge-bench/tests/test_gpu_correctness.rs`
  - **Done when**: `cargo test -p forge-bench --test test_gpu_correctness` passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-bench --test test_gpu_correctness`
  - **Commit**: `test(forge-bench): add GPU kernel correctness integration tests`
  - _Design: Test Strategy integration tests_

- [ ] 6.5 Reproducibility tests
  - **Do**:
    1. Create `forge-bench/tests/test_reproducibility.rs`:
       - `test_data_gen_deterministic`: same seed produces identical data
       - `test_reduce_reproducible`: run reduce twice with same seed, results match
       - `test_benchmark_cv`: run reduce 10 times, assert CV < 10% (relaxed for CI)
  - **Files**: `forge-bench/tests/test_reproducibility.rs`
  - **Done when**: `cargo test -p forge-bench --test test_reproducibility` passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-bench --test test_reproducibility`
  - **Commit**: `test(forge-bench): add reproducibility and statistical tests`
  - _Requirements: NFR-1, NFR-5_
  - _Design: Test Strategy reproducibility tests_

## Phase 7: Quality Gates

- [ ] 7.1 [VERIFY] Full local CI: cargo build + cargo test + cargo clippy
  - **Do**: Run complete local CI suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --workspace && cargo test --workspace && cargo clippy --workspace -- -D warnings 2>&1 | tail -5`
  - **Done when**: Build succeeds, all tests pass, no clippy warnings
  - **Commit**: `fix(forge): address clippy and type issues` (if fixes needed)

- [ ] 7.2 Create PR and verify CI
  - **Do**:
    1. Verify current branch is feat/gpu-query or create feature branch
    2. Stage all metal-forge-compute files
    3. Push branch: `git push -u origin <branch>`
    4. Create PR: `gh pr create --title "feat(forge): metal-forge-compute experiment suite" --body "..."`
  - **Verify**: `gh pr checks --watch` (wait for CI)
  - **Done when**: PR created, CI green
  - **Commit**: None (PR creation)

## Phase 8: PR Lifecycle

- [ ] 8.1 Monitor CI and fix failures
  - **Do**:
    1. Check CI status: `gh pr checks`
    2. If failures: read logs, fix issues, push fixes
    3. Re-check until all green
  - **Verify**: `gh pr checks` shows all passing
  - **Done when**: All CI checks green

- [ ] 8.2 Address review comments
  - **Do**:
    1. Check for review comments: `gh pr view --comments`
    2. Address each comment with code changes
    3. Push fixes, verify CI still green
  - **Verify**: `gh pr view --comments` shows all resolved
  - **Done when**: All review comments addressed

- [ ] 8.3 [VERIFY] AC checklist
  - **Do**: Verify all acceptance criteria from requirements.md are met:
    1. AC-1.1: Radix sort measured at 1M/10M/100M -- check results/full_run.json for sort entries
    2. AC-1.2: Prefix scan GB/s reported -- check scan entries
    3. AC-1.3: Reduce GB/s reported -- check reduce entries
    4. AC-1.4: Histogram 256/65536 bins -- check histogram entries
    5. AC-1.5: Compact at 10%/50%/90% selectivity -- check compact entries
    6. AC-1.6: Crossover points reported -- check summary output
    7. AC-1.7: Dispatch overhead included -- wall-clock timing by design
    8. AC-2.1: GEMM at 256/1024/4096 -- check gemm entries
    9. AC-2.2: GEMV for inference shapes -- check gemv entries
    10. AC-2.3: Accelerate baseline -- check cpu_stats in GEMM/GEMV
    11. AC-3.1: Filter selectivity sweep -- check filter entries
    12. AC-3.2: Group-by cardinality sweep -- check groupby entries
    13. AC-3.3: Hash join sizes -- check hash_join entries
    14. AC-4.1: Spreadsheet 1M cells -- check spreadsheet entries
    15. AC-4.2: Time series 1M ticks -- check timeseries entries
    16. AC-4.3: JSON/CSV throughput -- check json_parse entries
    17. AC-5.1: Pipeline 10M/100M -- check pipeline entries
    18. AC-5.3: DuckDB comparison -- check duckdb entries
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && python3 -c "import json; d=json.load(open('results/full_run.json')); exps=set(r['experiment'] for r in d); required={'reduce','scan','compact','sort','histogram','filter','groupby','gemm','gemv','pipeline','spreadsheet','timeseries','json_parse','hash_join'}; missing=required-exps; print(f'Found: {len(exps)} Missing: {missing if missing else \"none\"}'); assert not missing, f'Missing experiments: {missing}'"`
  - **Done when**: All 14+ experiments have results data
  - **Commit**: None

## Notes

- **POC shortcuts taken** (Phase 1):
  - Only reduce experiment; no other kernels
  - Hardcoded 256 threadgroup size (no auto-tuning)
  - No TOML config file loading (CLI args only)
  - No CSV output or roofline diagram
  - No error handling for Metal failures beyond panic
  - DuckDB comparison deferred to Phase 3

- **Production TODOs** (addressed in Phase 2+):
  - All 16 kernels implemented
  - Full output formats (JSON, CSV, roofline, summary)
  - Crossover-point analysis
  - Accelerate BLAS baselines for GEMM/GEMV
  - DuckDB CLI integration
  - 100M element support with memory pressure handling

- **Key risk areas**:
  - Radix sort is most complex kernel (8 passes x 3 dispatches = 24 dispatches in 1 cmdbuf)
  - 65536-bin histogram may hit threadgroup memory limits (32KB on Apple Silicon)
  - Hash join expected to show GPU < CPU for random access patterns
  - DuckDB comparison depends on duckdb being installed via Homebrew

- **Dependencies between tasks**:
  - Sort (2.4) depends on scan (2.1) -- scan kernels reused for radix sort histogram prefix sum
  - Compact (2.2) depends on scan (2.1) -- scan-based compaction
  - GroupBy (3.2) depends on sort (2.4) -- sort-based group-by
  - Pipeline (3.5) depends on filter (3.1), compact (2.2), sort (2.4), groupby (3.2)
  - DuckDB (3.6) depends on pipeline (3.5) -- same query, same data
