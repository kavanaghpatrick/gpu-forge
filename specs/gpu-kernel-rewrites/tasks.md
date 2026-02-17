---
spec: gpu-kernel-rewrites
phase: tasks
total_tasks: 38
created: 2026-02-17
---

# Tasks: GPU Kernel Rewrites

## Phase 1: Make It Work (POC)

Focus: Infrastructure (GPU timer, PSO hints, vec helpers) + first 3 kernel rewrites (reduce, scan, histogram) to prove the approach end-to-end. Skip tests, accept shortcuts.

### Infrastructure

- [x] 1.1 Add GpuTimer to timing.rs and export from lib.rs
  - **Do**:
    1. Add `GpuTimer` struct to `forge-primitives/src/timing.rs` with static method `elapsed_ms(&ProtocolObject<dyn MTLCommandBuffer>) -> Option<f64>` using `GPUStartTime()`/`GPUEndTime()`
    2. Add `use objc2::runtime::ProtocolObject; use objc2_metal::MTLCommandBuffer;` imports
    3. Implementation: call `cmd_buf.GPUStartTime()` and `cmd_buf.GPUEndTime()`, return `None` if either is 0.0, else `Some((end - start) * 1000.0)`
    4. Export `GpuTimer` from `forge-primitives/src/lib.rs` via `pub use timing::GpuTimer;`
  - **Files**: `metal-forge-compute/forge-primitives/src/timing.rs`, `metal-forge-compute/forge-primitives/src/lib.rs`
  - **Done when**: `GpuTimer` compiles and is importable from forge-primitives
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release -p forge-primitives 2>&1 | tail -3`
  - **Commit**: `feat(forge-primitives): add GpuTimer using GPUStartTime/GPUEndTime`
  - _Requirements: FR-1, AC-1.1_
  - _Design: Component 1 - GPU Timer Infrastructure_

- [x] 1.2 Upgrade PsoCache to descriptor-based PSO with occupancy hints
  - **Do**:
    1. In `forge-primitives/src/pso_cache.rs`, add imports: `use objc2_metal::{MTLComputePipelineDescriptor, MTLPipelineOption};`
    2. Replace `compile_pso` body (lines 63-78): create `MTLComputePipelineDescriptor::new()`, call `descriptor.setComputeFunction(Some(&function))`, `descriptor.setMaxTotalThreadsPerThreadgroup(256)`, `unsafe { descriptor.setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true) }`
    3. Use `device.newComputePipelineStateWithDescriptor_options_reflection_error(&descriptor, MTLPipelineOption::MTLPipelineOptionNone, None)` instead of `newComputePipelineStateWithFunction_error`
  - **Files**: `metal-forge-compute/forge-primitives/src/pso_cache.rs`
  - **Done when**: All PSOs compile with occupancy hints; existing tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test --release -p forge-primitives 2>&1 | tail -5`
  - **Commit**: `feat(forge-primitives): add PSO occupancy hints via descriptor`
  - _Requirements: FR-3, AC-2.1, AC-2.2_
  - _Design: Component 2 - PSO Descriptor Upgrade_

- [x] 1.3 Add vectorized load helpers to types.h
  - **Do**:
    1. Add `load_uint4_safe(device const uint*, uint base_idx, uint element_count) -> uint4` inline function before `#endif`
    2. Add `load_float4_safe(device const float*, uint base_idx, uint element_count) -> float4` inline function
    3. Both return zero-initialized vector, then conditionally load each of 4 elements with bounds check
  - **Files**: `metal-forge-compute/forge-primitives/shaders/types.h`
  - **Done when**: types.h compiles (triggers recompile of all shaders via build.rs rerun-if-changed)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release -p forge-primitives 2>&1 | tail -3`
  - **Commit**: `feat(forge-primitives): add vectorized load helpers to types.h`
  - _Requirements: FR-8, FR-19, AC-4.1 through AC-4.6_
  - _Design: Component 4 - Vectorized Load Macros_

- [x] 1.4 [VERIFY] Quality checkpoint: cargo build + cargo test
  - **Do**: Run build and all tests to verify infrastructure changes don't break anything
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release 2>&1 | tail -3 && cargo test --release 2>&1 | grep "test result"`
  - **Done when**: Zero build warnings, all 96 tests pass
  - **Commit**: `chore(forge): pass quality checkpoint` (only if fixes needed)

### Reduce Kernel Rewrite (simplest, validates approach)

- [x] 1.5 Rewrite reduce.metal with two-pass atomic-free kernel
  - **Do**:
    1. Add `reduce_sum_u32_v2` kernel to `reduce.metal`: each thread processes 4 elements via `load_uint4_safe`, SIMD reduction via `simd_sum`, threadgroup reduction via shared memory, writes per-TG partial to `partials[tg_idx]` (NO atomics)
    2. Add `reduce_sum_partials` kernel: single-TG reduction of partials array into final result, same SIMD+TG pattern
    3. Keep existing `reduce_sum_u32` kernel intact for now (avoid breaking existing code)
    4. Use `int` loop indices where applicable per FR-19
  - **Files**: `metal-forge-compute/forge-primitives/shaders/reduce.metal`
  - **Done when**: Both new kernels compile without errors
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release -p forge-primitives 2>&1 | tail -3`
  - **Commit**: `feat(reduce): add two-pass atomic-free reduce kernel`
  - _Requirements: FR-9, FR-8, AC-5.1, AC-5.2, AC-5.3_
  - _Design: Kernel 1 - Reduce Two-Pass Atomic-Free_

- [x] 1.6 Update reduce.rs to use two-pass dispatch + GpuTimer
  - **Do**:
    1. Add `partials_buffer` field to `ReduceExperiment` struct (sized `ceil(N / 1024) * sizeof(u32)`)
    2. Add `partials_params_buffer` field for the second pass params
    3. In `setup()`: allocate partials buffer, pre-warm `reduce_sum_u32_v2` and `reduce_sum_partials` PSOs
    4. In `run_gpu()`: replace BenchTimer with GpuTimer. Create single cmd_buf. Pass 1: dispatch `reduce_sum_u32_v2` with `ceil(N / (256*4))` TGs. Pass 2: dispatch `reduce_sum_partials` with 1 TG, input=partials, output=result. For >256 partials (>262K elements): chain a 3rd level. Commit, waitUntilCompleted, return `GpuTimer::elapsed_ms(&cmd_buf).unwrap_or(0.0)`
    5. Update imports: replace `BenchTimer` with `GpuTimer` (keep BenchTimer for run_cpu)
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/reduce.rs`
  - **Done when**: Reduce benchmark runs and validate() passes at 10M
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- reduce --sizes 10M --runs 1 --warmup 1 2>&1 | tail -10`
  - **Commit**: `feat(reduce): two-pass dispatch with GpuTimer`
  - _Requirements: FR-2, FR-9, AC-1.2, AC-5.4, AC-5.5_
  - _Design: Kernel 1 dispatch, Component 1 integration_

### Scan Kernel Rewrite (feeds sort, compact, groupby)

- [x] 1.7 Rewrite scan.metal with SIMD prefix scan
  - **Do**:
    1. Replace `scan_local` kernel body: each thread loads 4 elements via `load_uint4_safe`, computes local sum. Use `simd_prefix_exclusive_sum(thread_sum)` for intra-SIMD prefix. Cross-SIMD aggregation via `threadgroup uint simd_totals[8]` and second `simd_prefix_exclusive_sum`. Write 4 output elements with per-element prefix. Write TG total to partials.
    2. Update `SCAN_ELEMENTS_PER_TG` from 512 to 1024 (256 threads * 4 elements)
    3. Update `scan_partials` to use SIMD prefix pattern (handles up to 1024 partials now with 4 elements/thread)
    4. Update `scan_add_offsets` to add offset to 4 elements per thread instead of 2
    5. Keep `int` loop indices per FR-19
  - **Files**: `metal-forge-compute/forge-primitives/shaders/scan.metal`
  - **Done when**: All 3 scan kernels compile
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release -p forge-primitives 2>&1 | tail -3`
  - **Commit**: `feat(scan): SIMD prefix scan replacing Blelloch`
  - _Requirements: FR-11, FR-8, AC-7.1, AC-7.2_
  - _Design: Kernel 3 - Scan SIMD Prefix Replacement_

- [x] 1.8 Update scan.rs for new dispatch grid + GpuTimer
  - **Do**:
    1. Change `ELEMENTS_PER_TG` from 512 to 1024
    2. Change `MAX_GPU_PARTIALS` from 512 to 1024 (scan_partials now handles 4 elem/thread)
    3. Replace BenchTimer with GpuTimer in `run_gpu()`: call `GpuTimer::elapsed_ms(&cmd_buf)` after commit+waitUntilCompleted
    4. Update `num_threadgroups` calculation: `size.div_ceil(1024)` instead of `size.div_ceil(512)`
    5. Update partials buffer sizing accordingly
    6. Import GpuTimer, keep BenchTimer for CPU
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/scan.rs`
  - **Done when**: Scan benchmark runs and validate() passes at 10M (exact match)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- scan --sizes 10M --runs 1 --warmup 1 2>&1 | tail -10`
  - **Commit**: `feat(scan): update dispatch for 4-elem/thread + GpuTimer`
  - _Requirements: FR-2, FR-11, AC-7.3, AC-7.4_
  - _Design: Kernel 3 dispatch adjustment_

### Histogram Kernel Rewrite (quick win, validates bitmask)

- [x] 1.9 Rewrite histogram.metal with uint4 loads + bitmask binning
  - **Do**:
    1. Replace `histogram_256` kernel: each thread loads 4 elements via `load_uint4_safe(input, tid * 4, params.element_count)`. Use bitmask binning: `uint mask = params.num_bins - 1; uint bin = vals.x & mask;`. Accumulate each of 4 elements individually to `local_hist[bin]` via threadgroup atomic (TG atomics are cheap in shared memory). Keep TG-to-global merge unchanged.
    2. Add guard for non-power-of-2 bins: `uint mask = (num_bins & (num_bins - 1)) == 0 ? (num_bins - 1) : num_bins;` and use modulo as fallback
  - **Files**: `metal-forge-compute/forge-primitives/shaders/histogram.metal`
  - **Done when**: histogram_256 compiles with vectorized loads
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release -p forge-primitives 2>&1 | tail -3`
  - **Commit**: `feat(histogram): uint4 vectorized loads + bitmask binning`
  - _Requirements: FR-14, FR-8, AC-10.1, AC-10.2_
  - _Design: Kernel 2 - Histogram SIMD Pre-Aggregation + Bitmask_

- [x] 1.10 Update histogram.rs dispatch for 4x elements/thread + GpuTimer
  - **Do**:
    1. In `run_gpu()`: replace BenchTimer with GpuTimer
    2. Change dispatch size: `dispatch_1d` with `self.size / 4` (or `self.size.div_ceil(4)`) total threads since each thread handles 4 elements
    3. Import GpuTimer, keep BenchTimer for CPU
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/histogram.rs`
  - **Done when**: Histogram benchmark runs and validate() passes at 10M (exact bin match)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- histogram --sizes 10M --runs 1 --warmup 1 2>&1 | tail -10`
  - **Commit**: `feat(histogram): 4x dispatch + GpuTimer`
  - _Requirements: FR-2, FR-14, AC-10.4, AC-10.5_
  - _Design: Kernel 2 dispatch adjustment_

- [x] 1.11 [VERIFY] POC checkpoint: 3 kernels pass validate() at 10M
  - **Do**: Run reduce, scan, histogram at 10M to verify correctness and measure initial speedups
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- reduce --sizes 10M --runs 3 --warmup 1 2>&1 | grep -E "speedup|PASS|FAIL" && cargo run --release -p forge-bench -- scan --sizes 10M --runs 3 --warmup 1 2>&1 | grep -E "speedup|PASS|FAIL" && cargo run --release -p forge-bench -- histogram --sizes 10M --runs 3 --warmup 1 2>&1 | grep -E "speedup|PASS|FAIL"`
  - **Done when**: All 3 kernels pass validation at 10M; GPU timing operational
  - **Commit**: `feat(forge): POC validated - reduce, scan, histogram rewritten`

## Phase 2: Feature Complete (Remaining Kernel Rewrites)

### Sort Rewrite (depends on scan)

- [x] 2.1 Rewrite radix_scatter in radix_sort.metal with SIMD prefix rank
  - **Do**:
    1. Replace quadratic loop (lines 134-138) in `radix_scatter` kernel with SIMD prefix approach: for each of 16 digit values, compute `simd_prefix_exclusive_sum(digit == d ? 1u : 0u)` for intra-SIMD rank and `simd_sum()` for count. Record rank for matching digit. Store per-SIMD-group digit counts to `threadgroup uint simd_digit_counts[8][RADIX_BINS]`. Cross-SIMD offset by summing counts from earlier SIMD groups. Global scatter: `keys_out[global_base + local_pos] = key`.
    2. Add SIMD attributes to kernel params: `simd_lane [[thread_index_in_simdgroup]]`, `simd_group_id [[simdgroup_index_in_threadgroup]]`
  - **Files**: `metal-forge-compute/forge-primitives/shaders/radix_sort.metal`
  - **Done when**: radix_scatter compiles with SIMD prefix rank
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release -p forge-primitives 2>&1 | tail -3`
  - **Commit**: `feat(sort): SIMD prefix rank replacing quadratic scatter`
  - _Requirements: FR-10, AC-6.1, AC-6.2_
  - _Design: Kernel 4 - Radix Sort Scatter SIMD Prefix_

- [x] 2.2 Update sort.rs: single command buffer + blit fill + GpuTimer
  - **Do**:
    1. Replace per-pass command buffer creation with single cmd_buf for all 8 passes
    2. Replace CPU histogram zeroing loop with `blitCommandEncoder().fillBuffer_range_value()` before each pass's histogram encoder
    3. Pre-allocate 8 SortParams buffers in `setup()` instead of per-pass `alloc_buffer_with_data`
    4. Pre-allocate 8 ScanParams buffers in `setup()` similarly
    5. Replace BenchTimer with GpuTimer after single commit+waitUntilCompleted
    6. Update scan dispatch to use new ELEMENTS_PER_TG=1024 (matching scan rewrite)
    7. Update MAX_GPU_PARTIALS to 1024
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/sort.rs`
  - **Done when**: Sort runs with single cmdbuf, validate() passes at 10M (element-by-element exact match vs CPU sort_unstable)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- sort --sizes 10M --runs 1 --warmup 1 2>&1 | tail -10`
  - **Commit**: `feat(sort): single command buffer + SIMD scatter + GpuTimer`
  - _Requirements: FR-4, FR-2, AC-3.1, AC-6.3, AC-6.4, AC-6.5_
  - _Design: Component 3 - Sort single cmdbuf, Kernel 4 dispatch_

- [ ] 2.3 [VERIFY] Quality checkpoint: build + test + sort validates
  - **Do**: Verify sort correctness and overall build health
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release 2>&1 | tail -3 && cargo test --release 2>&1 | grep "test result" && cargo run --release -p forge-bench -- sort --sizes 10M --runs 1 --warmup 1 2>&1 | grep -E "speedup|PASS|FAIL"`
  - **Done when**: Build clean, tests pass, sort validates at 10M
  - **Commit**: `chore(forge): pass quality checkpoint` (only if fixes needed)

### Spreadsheet Rewrite (dramatic improvement)

- [ ] 2.4 Rewrite spreadsheet.metal with 2D row-parallel dispatch
  - **Do**:
    1. Add `spreadsheet_sum_v2` kernel: 2D dispatch (X=columns, Y=row chunks). Each thread sums `SS_ROWS_PER_CHUNK=64` rows for its column, writes to `partials[row_chunk * cols + col]`. Define `SS_ROWS_PER_CHUNK 64`.
    2. Add `spreadsheet_sum_reduce` kernel: 1D dispatch over columns. Each thread sums all row-chunk partials for one column into final output.
    3. Similarly add `spreadsheet_avg_v2` and `spreadsheet_avg_reduce` (divide by rows after summing).
    4. Keep existing VLOOKUP kernel (no rewrite needed).
  - **Files**: `metal-forge-compute/forge-primitives/shaders/spreadsheet.metal`
  - **Done when**: New spreadsheet kernels compile
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release -p forge-primitives 2>&1 | tail -3`
  - **Commit**: `feat(spreadsheet): 2D row-parallel coalesced kernels`
  - _Requirements: FR-12, AC-8.1, AC-8.2_
  - _Design: Kernel 5 - Spreadsheet Row-Parallel Coalesced_

- [ ] 2.5 Update spreadsheet.rs: 2D dispatch + partials buffer + single cmdbuf + GpuTimer
  - **Do**:
    1. Add `partials_buffer` field for row-chunk partial sums
    2. In `setup()`: allocate partials buffer sized `ceil(rows/64) * cols * sizeof(f32)`, pre-warm new PSOs
    3. In `run_gpu()`: single cmd_buf. Encoder 1: `spreadsheet_sum_v2` with 2D grid (ceil(cols/16), ceil(rows/64)) and TG (16,16). Encoder 2: `spreadsheet_sum_reduce` with 1D dispatch over cols. Encoder 3: average (sum_reduce output / rows). Encoder 4: vlookup (unchanged).
    4. Replace BenchTimer with GpuTimer
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/spreadsheet.rs`
  - **Done when**: Spreadsheet validates at 10M (f32 tolerance <= 1e-3)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- spreadsheet --sizes 10M --runs 1 --warmup 1 2>&1 | tail -10`
  - **Commit**: `feat(spreadsheet): 2D dispatch + partials + single cmdbuf + GpuTimer`
  - _Requirements: FR-7, FR-12, FR-2, AC-3.4, AC-8.3, AC-8.4, AC-8.5_
  - _Design: Kernel 5 dispatch, Component 3_

### GEMM Rewrite (simdgroup_matrix)

- [ ] 2.6 Add simdgroup_matrix GEMM kernel to gemm.metal
  - **Do**:
    1. Add `#include <metal_simdgroup_matrix>` and `#include <metal_simdgroup>` headers
    2. Add `gemm_simdgroup_f32` kernel: TG=(16,16)=256 threads=8 SIMD groups. Each SIMD group computes one 8x8 output tile via `simdgroup_matrix<float,8,8>`. 2x2 arrangement of SIMD groups covers 16x16 output tile. Shared memory with `GEMM_PAD=4` between rows for bank conflict avoidance. Loop over K tiles: cooperative load A,B tiles into shared memory, `simdgroup_load`, `simdgroup_multiply_accumulate`, `simdgroup_store` result to C.
    3. Keep existing `gemm_naive_f32` kernel
  - **Files**: `metal-forge-compute/forge-primitives/shaders/gemm.metal`
  - **Done when**: gemm_simdgroup_f32 compiles
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release -p forge-primitives 2>&1 | tail -3`
  - **Commit**: `feat(gemm): simdgroup_matrix_multiply_accumulate kernel`
  - _Requirements: FR-13, AC-9.1, AC-9.2, AC-9.3_
  - _Design: Kernel 6 - GEMM simdgroup_matrix_

- [ ] 2.7 Update gemm.rs to use new kernel + GpuTimer
  - **Do**:
    1. Change PSO name from `gemm_naive_f32` to `gemm_simdgroup_f32` in setup() and run_gpu()
    2. Update dispatch: TG=(16,16), grid=(ceil(N/16), ceil(M/16)) threadgroups
    3. Replace BenchTimer with GpuTimer
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/gemm.rs`
  - **Done when**: GEMM validates at 1024x1024 (f32 tolerance <= 1e-3)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- gemm --sizes 1M --runs 1 --warmup 1 2>&1 | tail -10`
  - **Commit**: `feat(gemm): simdgroup dispatch + GpuTimer`
  - _Requirements: FR-13, FR-2, AC-9.4, AC-9.5_
  - _Design: Kernel 6 dispatch_

- [ ] 2.8 [VERIFY] Quality checkpoint: build + test + validate spreadsheet + gemm
  - **Do**: Full build, test suite, and verify recent kernels
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release 2>&1 | tail -3 && cargo test --release 2>&1 | grep "test result" && cargo run --release -p forge-bench -- spreadsheet --sizes 10M --runs 1 --warmup 1 2>&1 | grep -E "speedup|PASS|FAIL" && cargo run --release -p forge-bench -- gemm --sizes 1M --runs 1 --warmup 1 2>&1 | grep -E "speedup|PASS|FAIL"`
  - **Done when**: Build clean, tests pass, spreadsheet + gemm validate
  - **Commit**: `chore(forge): pass quality checkpoint` (only if fixes needed)

### Filter + Timeseries Rewrites (independent, vectorization)

- [ ] 2.9 Rewrite filter_bench.metal with uint4 vectorized loads + SIMD reduce
  - **Do**:
    1. Replace `filter_count_gt` kernel: each thread loads `uint4` via `load_uint4_safe(input, gid * 4, params.element_count)`. Evaluate predicate on each component. Sum match count (0-4). Use `simd_sum(match_val)` to aggregate within SIMD group. Only lane 0 does `atomic_fetch_add_explicit` on output.
    2. Add `simd_lane [[thread_index_in_simdgroup]]` to kernel params
  - **Files**: `metal-forge-compute/forge-primitives/shaders/filter_bench.metal`
  - **Done when**: filter_count_gt compiles with vectorized loads
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release -p forge-primitives 2>&1 | tail -3`
  - **Commit**: `feat(filter): uint4 vectorized loads + SIMD reduce`
  - _Requirements: FR-15, AC-11.1_
  - _Design: Kernel 7 - Filter Vectorized Loads_

- [ ] 2.10 Update filter.rs dispatch + GpuTimer
  - **Do**:
    1. Change dispatch total_threads from `self.size` to `self.size.div_ceil(4)` (each thread handles 4 elements)
    2. Replace BenchTimer with GpuTimer
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/filter.rs`
  - **Done when**: Filter validates at 10M (exact match count)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- filter --sizes 10M --runs 1 --warmup 1 2>&1 | tail -10`
  - **Commit**: `feat(filter): 4x dispatch + GpuTimer`
  - _Requirements: FR-15, FR-2, AC-11.2, AC-11.3_
  - _Design: Kernel 7 dispatch_

- [ ] 2.11 Rewrite timeseries.metal with float4 window reads
  - **Do**:
    1. Modify `timeseries_moving_avg` kernel: in the window summation loop, process 4 elements at a time via `float4(prices[i], prices[i+1], prices[i+2], prices[i+3])` with `sum += chunk.x + chunk.y + chunk.z + chunk.w`. Scalar remainder loop for last 0-3 elements.
  - **Files**: `metal-forge-compute/forge-primitives/shaders/timeseries.metal`
  - **Done when**: Timeseries kernel compiles with float4 reads
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release -p forge-primitives 2>&1 | tail -3`
  - **Commit**: `feat(timeseries): float4 vectorized window reads`
  - _Requirements: FR-16, AC-12.1_
  - _Design: Kernel 8 - Timeseries Float4 Window Reads_

- [ ] 2.12 Update timeseries.rs with GpuTimer
  - **Do**:
    1. Replace BenchTimer with GpuTimer in run_gpu()
    2. No dispatch change needed (each thread still handles one output element; vectorization is within the window read)
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/timeseries.rs`
  - **Done when**: Timeseries validates at 10M (f32 tolerance <= 1e-4)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- timeseries --sizes 10M --runs 1 --warmup 1 2>&1 | tail -10`
  - **Commit**: `feat(timeseries): GpuTimer integration`
  - _Requirements: FR-2, AC-12.2, AC-12.3_
  - _Design: Kernel 8 dispatch_

### JSON Parse + GEMV Scale-Up (lowest priority)

- [ ] 2.13 Update json_parse.rs: add 10M to sizes + GpuTimer
  - **Do**:
    1. Add `10_000_000` to `supported_sizes()` vec
    2. Replace BenchTimer with GpuTimer in run_gpu()
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/json_parse.rs`
  - **Done when**: JSON parse runs at 10M without crash
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- json_parse --sizes 10M --runs 1 --warmup 1 2>&1 | tail -10`
  - **Commit**: `feat(json_parse): scale to 10M + GpuTimer`
  - _Requirements: FR-17, FR-2, AC-13.1_
  - _Design: Kernel 9 - JSON Parse Scale-Up_

- [ ] 2.14 Update gemv.rs: add 4096 to sizes + GpuTimer
  - **Do**:
    1. Add `4096` to `supported_sizes()` vec
    2. Replace BenchTimer with GpuTimer in run_gpu()
    3. Ensure buffer allocation handles 4096x4096 = 16.7M elements
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/gemv.rs`
  - **Done when**: GEMV runs at 4096 without crash
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- gemv --sizes 4096 --runs 1 --warmup 1 2>&1 | tail -10`
  - **Commit**: `feat(gemv): scale to 4096 + GpuTimer`
  - _Requirements: FR-18, FR-2, AC-14.1_
  - _Design: Kernel 10 - GEMV Scale-Up_

### Remaining GpuTimer Integration (composite + hash_join)

- [ ] 2.15 Add GpuTimer to compact, groupby, pipeline, duckdb, hash_join experiments
  - **Do**:
    1. In each of `compact.rs`, `groupby.rs`, `pipeline.rs`, `duckdb.rs`, `hash_join.rs`: replace BenchTimer with GpuTimer in `run_gpu()`. Pattern: remove `let timer = BenchTimer::start();` and `timer.stop()`, add `GpuTimer::elapsed_ms(&cmd_buf).unwrap_or(0.0)` after commit+waitUntilCompleted.
    2. Update imports: add `GpuTimer` to use statement, keep `BenchTimer` for `run_cpu()`
    3. For duckdb.rs: also consolidate multiple cmd_buf commits into single command buffer where possible (single cmdbuf for filter+scan+scatter pipeline)
    4. Update compact.rs constants: `ELEMENTS_PER_TG` from 512 to 1024 and `MAX_GPU_PARTIALS` from 512 to 1024 to match scan rewrite
  - **Files**: `metal-forge-compute/forge-bench/src/experiments/compact.rs`, `metal-forge-compute/forge-bench/src/experiments/groupby.rs`, `metal-forge-compute/forge-bench/src/experiments/pipeline.rs`, `metal-forge-compute/forge-bench/src/experiments/duckdb.rs`, `metal-forge-compute/forge-bench/src/experiments/hash_join.rs`
  - **Done when**: All 5 experiments compile with GpuTimer
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release 2>&1 | tail -3`
  - **Commit**: `feat(forge-bench): add GpuTimer to composite + hash_join experiments`
  - _Requirements: FR-2, FR-5, FR-6, AC-1.2, AC-3.2, AC-3.3, AC-3.5_
  - _Design: Component 1 integration, Component 3_

- [ ] 2.16 [VERIFY] Quality checkpoint: full build + test + all 15 experiments compile
  - **Do**: Verify everything builds and tests pass after all Phase 2 changes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release 2>&1 | tail -3 && cargo test --release 2>&1 | grep "test result"`
  - **Done when**: Clean build, all tests pass
  - **Commit**: `chore(forge): pass quality checkpoint` (only if fixes needed)

## Phase 3: Testing (Per-Kernel Correctness + Benchmark Validation)

- [ ] 3.1 Validate all 15 experiments at 10M elements
  - **Do**:
    1. Run each of the 15 experiments with `--sizes 10M --runs 1 --warmup 1` and verify validate() passes
    2. For GEMM: use `--sizes 1M` (1024x1024 matrix)
    3. For GEMV: use `--sizes 4096`
    4. For hash_join: use `--sizes 10M`
    5. Document any failures for immediate investigation
  - **Files**: None (verification only)
  - **Done when**: All 15 experiments pass validate() at their target sizes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && for exp in reduce histogram scan compact sort filter groupby gemm timeseries hash_join json_parse pipeline duckdb spreadsheet; do echo "=== $exp ===" && cargo run --release -p forge-bench -- $exp --sizes 10M --runs 1 --warmup 1 2>&1 | grep -E "speedup|PASS|FAIL|Error|panic"; done && cargo run --release -p forge-bench -- gemv --sizes 4096 --runs 1 --warmup 1 2>&1 | grep -E "speedup|PASS|FAIL|Error|panic"`
  - **Commit**: `test(forge): validate all 15 experiments at 10M`
  - _Requirements: FR-20, AC-15.6, NFR-3, NFR-4_
  - _Design: Test Strategy - Unit Tests_

- [ ] 3.2 Run full benchmark suite with standard profile (10 runs, 3 warmup)
  - **Do**:
    1. Run each P0 kernel with standard profile: `--sizes 10M --runs 10 --warmup 3`
    2. Record speedup values for each kernel
    3. Verify each kernel meets minimum speedup target from requirements
    4. If any kernel misses target, investigate and fix (tuning TG size, buffer sizes, etc.)
  - **Files**: None (benchmark verification)
  - **Done when**: All P0 kernels meet minimum speedup targets
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && for exp in reduce histogram scan sort spreadsheet gemm compact groupby; do echo "=== $exp ===" && cargo run --release -p forge-bench -- $exp --sizes 10M --runs 10 --warmup 3 2>&1 | grep -E "speedup|GPU.*ms|CPU.*ms"; done`
  - **Commit**: `perf(forge): benchmark results meet P0 targets`
  - _Requirements: NFR-1, NFR-2, NFR-5_
  - _Design: Test Strategy - Performance Validation_

- [ ] 3.3 [VERIFY] Quality checkpoint: build + test + validate
  - **Do**: Full quality check after testing phase
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build --release 2>&1 | tail -3 && cargo test --release 2>&1 | grep "test result"`
  - **Done when**: Clean build, all tests pass
  - **Commit**: `chore(forge): pass quality checkpoint` (only if fixes needed)

## Phase 4: Quality Gates

- [ ] 4.1 [VERIFY] Full local CI: build + test + fmt check
  - **Do**: Run complete local quality suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo fmt --check 2>&1 | tail -5 && cargo build --release 2>&1 | tail -3 && cargo test --release 2>&1 | grep "test result"`
  - **Done when**: Build succeeds, all tests pass, code formatted
  - **Commit**: `chore(forge): pass local CI` (if fixes needed)

- [ ] 4.2 Create PR and verify CI
  - **Do**:
    1. Verify current branch is `feat/gpu-query` (feature branch): `git branch --show-current`
    2. Stage all modified files: `git add -A`
    3. Push branch: `git push -u origin feat/gpu-query`
    4. Create PR: `gh pr create --title "feat(forge): rewrite 15 GPU kernels to best-in-class Apple Silicon" --body "..."`
    5. Wait for CI: `gh pr checks --watch`
  - **Verify**: `gh pr checks` shows all green
  - **Done when**: All CI checks pass, PR ready for review
  - **Commit**: None (PR creation, not code change)

## Phase 5: PR Lifecycle

- [ ] 5.1 Monitor CI and fix failures
  - **Do**:
    1. Check CI status: `gh pr checks`
    2. If failures: read failure details, fix locally, push
    3. Re-verify: `gh pr checks --watch`
  - **Verify**: `gh pr checks` all passing
  - **Done when**: CI green
  - **Commit**: `fix(forge): address CI failures` (if needed)

- [ ] 5.2 [VERIFY] AC checklist verification
  - **Do**: Verify each acceptance criterion from requirements.md is met:
    1. AC-1.1: GpuTimer exists in timing.rs using GPUStartTime/GPUEndTime
    2. AC-1.2: All 15 experiments use GpuTimer in run_gpu()
    3. AC-2.1-2.3: PSO uses descriptor with occupancy hints
    4. AC-3.1: Sort uses single cmdbuf (verify no per-pass commit)
    5. AC-5.1-5.5: Reduce is two-pass atomic-free, speedup >= 5.0x
    6. AC-6.1-6.5: Sort uses simd_prefix, speedup >= 10.0x
    7. AC-7.1-7.4: Scan uses simd_prefix, speedup >= 3.0x
    8. AC-8.1-8.5: Spreadsheet 2D dispatch, speedup >= 3.0x
    9. AC-9.1-9.5: GEMM uses simdgroup_matrix, speedup >= 1.5x
    10. AC-10.1-10.5: Histogram bitmask+vectorized, speedup >= 3.0x
    11. AC-11.1-11.3: Filter uint4, speedup >= 6.0x
    12. AC-12.1-12.3: Timeseries float4, speedup >= 3.0x
    13. AC-15.1-15.6: Composite kernels inherit improvements
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && grep -c "GpuTimer" forge-bench/src/experiments/*.rs && grep -c "MTLComputePipelineDescriptor" forge-primitives/src/pso_cache.rs && grep -c "simd_prefix_exclusive_sum" forge-primitives/shaders/scan.metal forge-primitives/shaders/radix_sort.metal && grep -c "simdgroup_matrix" forge-primitives/shaders/gemm.metal && grep -c "load_uint4_safe" forge-primitives/shaders/reduce.metal forge-primitives/shaders/histogram.metal forge-primitives/shaders/filter_bench.metal`
  - **Done when**: All ACs verified via code inspection + benchmark results
  - **Commit**: None

- [ ] 5.3 Final benchmark sweep at 10M with all 15 kernels
  - **Do**: Run standard profile across all kernels, capture final speedup numbers
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo run --release -p forge-bench -- all --profile standard 2>&1 | tail -60`
  - **Done when**: All 15 kernels show GPU > 1.0x at 10M, P0 kernels hit minimum targets
  - **Commit**: `docs(forge): final benchmark sweep results`
  - _Requirements: NFR-1, NFR-2_

## Notes

- **POC shortcuts taken**:
  - Phase 1 keeps old kernel variants (reduce_sum_u32, gemm_naive_f32) intact rather than deleting
  - Reduce 3-level chain for >256 partials may use a simplified approach initially
  - DuckDB single-cmdbuf consolidation deferred to task 2.15 (grouped with GpuTimer integration)
  - GEMV may not hit 1.0x at 4096 if AMX dominates -- document as known limitation

- **Production TODOs**:
  - Delete legacy kernel variants after all tests pass (Phase 2 cleanup)
  - StorageModePrivate optimization for read-only GPU buffers (deferred per requirements out-of-scope)
  - Scan 3-level GPU-only scan for 100M+ elements (currently CPU fallback for >1024 partials)
  - Consider multi-element/thread for JSON parse shader (not just Rust-side scale-up)

- **Critical dependency chain**:
  - GpuTimer (1.1) -> all GpuTimer integrations (1.6, 1.8, 1.10, 2.2, 2.5, 2.7, 2.10, 2.12-2.15)
  - types.h vec helpers (1.3) -> reduce, histogram, filter, scan shader rewrites
  - scan.metal (1.7) -> sort.rs dispatch (2.2) + compact.rs constants (2.15)
  - All kernel rewrites -> validation sweep (3.1) -> benchmark suite (3.2)
