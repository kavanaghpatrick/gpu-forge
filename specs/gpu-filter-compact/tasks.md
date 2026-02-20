---
spec: gpu-filter-compact
phase: tasks
created: 2026-02-20
---

# Tasks: GPU Filter + Compact (forge-filter)

## Phase 1: Make It Work (POC)

Focus: Get a u32 > threshold filter working end-to-end with the 3-dispatch pipeline. Skip compound predicates, index output, unordered mode, 64-bit types. Prove GPU bandwidth advantage over CPU.

### Task 1.1: Scaffold forge-filter crate [x]

- **Do**:
  1. Create `metal-forge-compute/forge-filter/` directory structure
  2. Create `Cargo.toml` mirroring forge-sort: name=`forge-filter`, version=`0.1.0`, same deps (objc2 0.6, objc2-metal 0.3, objc2-foundation 0.3, thiserror 2), dev-deps (rand 0.8, rand_chacha 0.3, criterion 0.5)
  3. Create `build.rs` identical to forge-sort but for `shaders/filter.metal` -> `filter.air` -> `filter.metallib`, env var `FILTER_METALLIB_PATH`
  4. Create `src/metal_helpers.rs` — copy verbatim from `forge-sort/src/metal_helpers.rs` (PsoCache, FnConstant, alloc_buffer, init_device_and_queue)
  5. Create `src/lib.rs` stub with `mod metal_helpers;` and empty `pub struct GpuFilter`
  6. Create `shaders/filter.metal` stub with `#include <metal_stdlib>` and empty kernel signature for `filter_predicate_scan`
  7. Add `"forge-filter"` to `metal-forge-compute/Cargo.toml` workspace members
- **Files**: `metal-forge-compute/forge-filter/Cargo.toml`, `metal-forge-compute/forge-filter/build.rs`, `metal-forge-compute/forge-filter/src/lib.rs`, `metal-forge-compute/forge-filter/src/metal_helpers.rs`, `metal-forge-compute/forge-filter/shaders/filter.metal`, `metal-forge-compute/Cargo.toml`
- **Done when**: `cargo build -p forge-filter` succeeds with no errors
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-filter`
- **Commit**: `feat(forge-filter): scaffold crate with build.rs, metal_helpers, shader stub`
- _Requirements: NFR-8, NFR-9_
- _Design: Crate Structure, Build System_

### Task 1.2: Implement FilterKey trait, Predicate enum, FilterParams, FilterError [x]

- **Do**:
  1. In `src/lib.rs`, add sealed `FilterKey` trait with `KEY_SIZE`, `IS_64BIT`, `to_bits()` — implement for u32, i32, f32, u64, i64, f64 (exact code in design.md Rust API section)
  2. Add `Predicate<T: FilterKey>` enum: Gt, Lt, Ge, Le, Eq, Ne, Between(T,T), And(Vec), Or(Vec)
  3. Add `Predicate::pred_type_id(&self) -> u32` method returning 0-7 for function constant
  4. Add `Predicate::to_bits(&self) -> (u64, u64)` returning (lo_bits, hi_bits) for Metal params
  5. Add `FilterError` enum with DeviceNotFound, ShaderCompilation, GpuExecution, EmptyInput, InvalidPredicate variants (use thiserror)
  6. Add `#[repr(C)] FilterParams` struct: element_count(u32), num_tiles(u32), lo_bits(u32), hi_bits(u32)
  7. Add `#[repr(C)] FilterParams64` struct: element_count(u32), num_tiles(u32), lo_lo(u32), lo_hi(u32), hi_lo(u32), hi_hi(u32), _pad([u32;2])
  8. Add constants: `FILTER_THREADS=256`, `FILTER_TILE_32=4096`, `FILTER_TILE_64=2048`
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
- **Done when**: `cargo build -p forge-filter` succeeds, all types/enums compile
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-filter`
- **Commit**: `feat(forge-filter): add FilterKey trait, Predicate enum, FilterParams, FilterError`
- _Requirements: FR-1, FR-2, FR-3_
- _Design: FilterKey Trait, Predicate Enum, FilterParams Struct_

### Task 1.3: Write filter.metal — all 4 GPU kernels [x]

- **Do**:
  1. Write `shaders/filter.metal` with defines: FILTER_THREADS=256, FILTER_ELEMS_32=16, FILTER_ELEMS_64=8, FILTER_TILE_32=4096, FILTER_TILE_64=2048, FILTER_NUM_SGS=8, SCAN_ELEMS_PER_THREAD=16
  2. Add function constants: PRED_TYPE(index 0, uint), IS_64BIT(index 1, bool), OUTPUT_IDX(index 2, bool), OUTPUT_VALS(index 3, bool) with defaults via `is_function_constant_defined`
  3. Add FilterParams and FilterParams64 structs matching Rust repr(C) layout
  4. Add `evaluate_predicate<T>(val, lo, hi)` template — 8 pred_type branches (GT=0, LT=1, GE=2, LE=3, EQ=4, NE=5, BETWEEN=6, TRUE=7), all eliminated at PSO compile time
  5. Write `filter_predicate_scan` kernel: evaluate predicate for ELEMS elements/thread, SIMD prefix sum, cross-SG aggregation via simd_totals[8], write TG total to partials[tg_idx]. Handle both 32-bit and 64-bit paths (via is_64bit constant). Use striped layout: `idx = base + e * FILTER_THREADS + lid`
  6. Write `filter_scan_partials` kernel: single-TG scan of partials array. 256 threads x SCAN_ELEMS_PER_THREAD(16) = 4096 partials max. SIMD prefix sum + cross-SG aggregation. Write exclusive prefix sums back. Write grand total to count_out[0] from last thread.
  7. Write `filter_scatter` kernel: re-evaluate predicate, recompute local scan, read global_prefix from scanned partials[tg_idx], scatter matching elements to output_vals and/or output_idx (controlled by OUTPUT_VALS/OUTPUT_IDX function constants). Same striped element layout.
  8. Write `filter_atomic_scatter` kernel: single-dispatch unordered mode. Evaluate predicate, SIMD-aggregated atomic (`simd_sum` + `simd_prefix_exclusive_sum` + `atomic_fetch_add` per SG lane 0 + `simd_broadcast_first`), scatter to output.
  9. Both 32-bit path (uint, as_type<int/float>) and 64-bit path (ulong, as_type<long/double>) in each kernel, selected by is_64bit constant.
- **Files**: `metal-forge-compute/forge-filter/shaders/filter.metal`
- **Done when**: `cargo build -p forge-filter` succeeds (shader compiles via build.rs)
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-filter 2>&1 | grep -E "(error|warning|Built)"`
- **Commit**: `feat(forge-filter): implement 4 Metal kernels — predicate_scan, scan_partials, scatter, atomic_scatter`
- _Requirements: FR-1, FR-2, FR-5, FR-9, FR-12_
- _Design: Metal Shader Design (all 4 kernels), Function Constants_

### Task 1.4: Implement FilterBuffer<T> and FilterResult<T> [x]

- **Do**:
  1. In `src/lib.rs`, add `FilterBuffer<T: FilterKey>` struct: buffer (Retained MTLBuffer), len, capacity, PhantomData. Mirror SortBuffer<T> exactly.
  2. Implement: `len()`, `is_empty()`, `capacity()`, `as_slice()` (unsafe ptr cast), `as_mut_slice()` (unsafe ptr cast over capacity), `set_len()` (assert <= capacity), `copy_from_slice()` (copy + set_len), `metal_buffer()` (return &buffer)
  3. Add `FilterResult<T: FilterKey>` struct: count(usize), values_buf(Option<Retained<MTLBuffer>>), indices_buf(Option<Retained<MTLBuffer>>), capacity(usize), PhantomData
  4. Implement: `len()`, `is_empty()`, `as_slice()` (unsafe ptr from values_buf, count elements), `indices()` -> Option<&[u32]> (from indices_buf), `to_vec()`, `metal_buffer()` -> Option<&MTLBuffer>
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
- **Done when**: `cargo build -p forge-filter` succeeds
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-filter`
- **Commit**: `feat(forge-filter): add FilterBuffer<T> and FilterResult<T> types`
- _Requirements: FR-6, FR-13, AC-4.1 through AC-4.5_
- _Design: FilterBuffer, FilterResult_

### Task 1.5: Implement GpuFilter::new() and 3-dispatch filter pipeline [x]

- **Do**:
  1. Add `GpuFilter` struct: device, queue, library, pso_cache, scratch buffers (buf_partials, buf_output, buf_output_idx, buf_count), scratch_capacity
  2. Implement `GpuFilter::new()`: init_device_and_queue(), load metallib via `env!("FILTER_METALLIB_PATH")`, pre-compile 15 PSOs (7 pred types x filter_predicate_scan + filter_scatter with IS_64BIT=false, plus filter_scan_partials)
  3. Implement `alloc_filter_buffer<T>(&self, capacity)` -> FilterBuffer<T>
  4. Implement private `ensure_scratch_buffers(&mut self, n: usize)` — grow-only allocation for partials (num_tiles * 4), output_vals (n * elem_size), output_idx (n * 4), count_buf (4 bytes)
  5. Implement `fn dispatch_filter<T: FilterKey>(&mut self, input_buf: &MTLBuffer, n: usize, pred: &Predicate<T>, output_vals: bool, output_idx: bool) -> Result<usize, FilterError>`:
     - Compute num_tiles = ceil(n / tile_size)
     - Build FilterParams from pred.to_bits() and n
     - Get PSOs for pred_type + IS_64BIT from cache
     - Create command buffer + single compute encoder
     - Encode dispatch 1: filter_predicate_scan (num_tiles TGs x 256 threads)
     - Encode dispatch 2: filter_scan_partials (1 TG x 256 threads), pass partials + num_tiles + count_buf
     - Encode dispatch 3: filter_scatter (num_tiles TGs x 256 threads), pass input, output_vals, output_idx, partials, count_buf, params
     - endEncoding, commit, waitUntilCompleted
     - Check status != Error
     - Read count from count_buf[0] (CPU readback from shared memory)
     - Return count
  6. Implement `pub fn filter<T: FilterKey>(&mut self, buf: &FilterBuffer<T>, pred: &Predicate<T>) -> Result<FilterResult<T>, FilterError>`:
     - Call ensure_scratch_buffers
     - Call dispatch_filter with output_vals=true, output_idx=false
     - Return FilterResult with count, values_buf, None indices
  7. Implement `pub fn filter_u32(&mut self, data: &[u32], pred: &Predicate<u32>) -> Result<Vec<u32>, FilterError>`:
     - Alloc FilterBuffer, copy_from_slice, call filter(), return to_vec()
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
- **Done when**: `cargo build -p forge-filter` succeeds
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-filter`
- **Commit**: `feat(forge-filter): implement GpuFilter with 3-dispatch ordered filter pipeline`
- _Requirements: FR-1, FR-5, FR-10, FR-11, FR-12, AC-1.1, AC-7.1_
- _Design: GpuFilter, Data Flow, PSO Specialization Matrix_

### Task 1.6: POC smoke test — filter u32 > threshold [x]

- **Do**:
  1. Add `#[cfg(test)] mod tests` in `src/lib.rs`
  2. Write `test_filter_u32_gt_basic`: create GpuFilter, generate 1M u32s (0..1_000_000), filter with Predicate::Gt(500_000), compare result to CPU reference (`data.iter().filter(|&&x| x > 500_000).copied().collect()`), assert_eq on lengths and contents
  3. Write `test_filter_u32_gt_16m`: same but 16M elements, Predicate::Gt(8_000_000), verify length matches CPU reference (don't compare all elements for speed — compare first 100 + last 100 + length)
  4. Write `test_filter_u32_empty_result`: filter 1M elements with Predicate::Gt(2_000_000) (none match), verify result.len() == 0
  5. Write `test_filter_u32_all_match`: filter 1M elements with Predicate::Gt(0) (all but 0 match), verify result.len() == 999_999
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
- **Done when**: All 4 tests pass
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter -- --nocapture 2>&1 | tail -20`
- **Commit**: `feat(forge-filter): POC verified — u32 Gt filter matches CPU reference at 1M and 16M`
- _Requirements: AC-1.1, AC-1.4, FR-5_
- _Design: Data Flow_

### Task 1.7: [VERIFY] Quality checkpoint: build + test + clippy [x]

- **Do**: Run quality commands, fix any issues
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-filter && cargo test -p forge-filter && cargo clippy -p forge-filter -- -D warnings`
- **Done when**: All commands exit 0
- **Commit**: `chore(forge-filter): pass quality checkpoint` (only if fixes needed)

## Phase 2: Core Features

Focus: Complete the API — all 6 types, all predicates, index output, unordered mode, convenience methods.

### Task 2.1: Add all 6 comparison operators for u32 [x]

- **Do**:
  1. Implement `filter_u32` convenience method if not already done (should exist from 1.5)
  2. Add tests for all 6 operators on u32 at 1M: `test_filter_u32_lt`, `test_filter_u32_ge`, `test_filter_u32_le`, `test_filter_u32_eq`, `test_filter_u32_ne`. Each generates data, applies GPU filter, compares to CPU reference.
  3. Fix any predicate evaluation bugs found in the Metal shader or Rust pred_type_id mapping
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
- **Done when**: 6 operator tests pass for u32 (Gt from 1.6 + 5 new)
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter -- test_filter_u32 --nocapture 2>&1 | tail -30`
- **Commit**: `feat(forge-filter): all 6 comparison operators verified for u32`
- _Requirements: FR-1, AC-1.2_
- _Design: Predicate Evaluation_

### Task 2.2: Add all 6 numeric types with convenience methods

- **Do**:
  1. Implement convenience methods: `filter_i32`, `filter_f32`, `filter_u64`, `filter_i64`, `filter_f64` — each takes `&[T]` + `&Predicate<T>`, returns `Result<Vec<T>>`
  2. Ensure `dispatch_filter` handles IS_64BIT=true for u64/i64/f64 — uses FilterParams64, FILTER_TILE_64, passes correct buffer sizes (N * 8)
  3. For f32: verify NaN handling (NaN > x = false, NaN == NaN = false, NaN != NaN = true) — the Metal `as_type<float>` comparison follows IEEE 754 naturally
  4. Add one test per type with Gt predicate at 1M: `test_filter_i32_gt`, `test_filter_f32_gt`, `test_filter_u64_gt`, `test_filter_i64_gt`, `test_filter_f64_gt`
  5. Add `test_filter_f32_nan` — insert NaN values, verify they are excluded by Gt, included by Ne
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`, `metal-forge-compute/forge-filter/shaders/filter.metal`
- **Done when**: All 6 type tests + NaN test pass
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter -- --nocapture 2>&1 | tail -30`
- **Commit**: `feat(forge-filter): support all 6 numeric types — u32/i32/f32/u64/i64/f64`
- _Requirements: FR-2, AC-1.3, FR-17, AC-7.2_
- _Design: FilterKey trait, Type Support_

### Task 2.3: BETWEEN range predicate

- **Do**:
  1. Ensure Predicate::Between(lo, hi) maps to pred_type=6, to_bits returns (lo.to_bits(), hi.to_bits())
  2. Verify Metal shader `evaluate_predicate` with pred_type==6 evaluates `val >= lo && val <= hi`
  3. Add tests: `test_filter_u32_between` (1M, between(250_000, 750_000) ~50% match), `test_filter_f32_between`, `test_filter_between_inclusive` (verify both endpoints included), `test_filter_between_eq` (Between(x,x) same as Eq(x)), `test_filter_between_inverted` (Between(hi,lo) returns empty)
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
- **Done when**: All BETWEEN tests pass
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter -- between --nocapture 2>&1 | tail -20`
- **Commit**: `feat(forge-filter): BETWEEN range predicate with inclusive endpoints`
- _Requirements: FR-3, AC-2.1 through AC-2.4_
- _Design: Predicate Evaluation (pred_type 6)_

### Task 2.4: [VERIFY] Quality checkpoint: build + all tests + clippy

- **Do**: Run quality commands, fix any issues
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-filter --release && cargo test -p forge-filter && cargo clippy -p forge-filter -- -D warnings`
- **Done when**: All commands exit 0
- **Commit**: `chore(forge-filter): pass quality checkpoint` (only if fixes needed)

### Task 2.5: Index output mode (filter_indices, filter_with_indices)

- **Do**:
  1. Implement `pub fn filter_indices<T>(&mut self, buf: &FilterBuffer<T>, pred: &Predicate<T>) -> Result<FilterResult<T>>` — calls dispatch_filter with output_vals=false, output_idx=true
  2. Implement `pub fn filter_with_indices<T>(&mut self, buf: &FilterBuffer<T>, pred: &Predicate<T>) -> Result<FilterResult<T>>` — calls dispatch_filter with output_vals=true, output_idx=true
  3. In dispatch_filter, pass OUTPUT_IDX and OUTPUT_VALS as function constants (indices 2, 3) to scatter PSO. Ensure correct PSO cache key includes these constants.
  4. Ensure index scatter writes global element index (`base + e * FILTER_THREADS + lid`) as u32 to output_idx buffer
  5. Add tests: `test_filter_indices_ascending` (verify indices sorted ascending), `test_filter_with_indices` (verify both values and indices correct, same length), `test_filter_indices_match_values` (gather original data at indices, compare to values output)
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
- **Done when**: All index output tests pass
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter -- indices --nocapture 2>&1 | tail -20`
- **Commit**: `feat(forge-filter): index output mode — filter_indices and filter_with_indices`
- _Requirements: FR-7, FR-8, AC-5.1 through AC-5.4_
- _Design: FilterResult indices(), Kernel 3 OUTPUT_IDX constant_

### Task 2.6: Unordered atomic scatter mode

- **Do**:
  1. Implement `pub fn filter_unordered<T>(&mut self, buf: &FilterBuffer<T>, pred: &Predicate<T>) -> Result<FilterResult<T>>`:
     - Ensure scratch counter buffer (4 bytes) exists, zero it before dispatch
     - Get PSO for `filter_atomic_scatter` with correct PRED_TYPE + IS_64BIT constants
     - Single dispatch: num_tiles TGs x 256 threads
     - Read count from counter buffer after completion
     - Return FilterResult
  2. Add tests: `test_filter_unordered_set_eq` (compare sorted(unordered result) == sorted(ordered result)), `test_filter_unordered_count` (verify same count as ordered)
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
- **Done when**: Unordered tests pass, same element set as ordered mode
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter -- unordered --nocapture 2>&1 | tail -20`
- **Commit**: `feat(forge-filter): unordered atomic scatter mode — single dispatch for aggregation queries`
- _Requirements: FR-9, AC-6.1 through AC-6.4_
- _Design: Kernel 4 (filter_atomic_scatter), Unordered Mode_

### Task 2.7: Compound predicates (AND/OR)

- **Do**:
  1. For AND/OR of simple predicates on a single column: implement multi-pass approach in `dispatch_filter`:
     - AND: run ordered filter for first predicate, then filter the output with the second predicate. Alternative (faster): fuse by evaluating all sub-predicates in a single pass. For V1, use the simpler multi-pass approach.
     - OR: similarly multi-pass (filter for each, union results). For V1, use bit-mask composition.
  2. Alternative (recommended in design): Detect common patterns. `And([Gt(lo), Lt(hi)])` -> convert to BETWEEN. `And([Ge(lo), Le(hi)])` -> BETWEEN. For general AND/OR, evaluate each sub-predicate into a bool mask buffer, combine with bitwise AND/OR, then compact using mask.
  3. Implement `pred.flatten()` or `pred.simplify()` that detects BETWEEN pattern.
  4. For general compounds: add a `filter_compound` internal method that allocates a u32 mask buffer, runs predicate_scan for each sub-predicate (writing 0/1 flags), combines masks, then does a separate scan+scatter on the combined mask.
  5. Add tests: `test_compound_and` (And([Gt(100), Lt(900)]) on 0..1000), `test_compound_or` (Or([Lt(100), Gt(900)])), `test_compound_nested` (And([Or([Lt(100), Gt(900)]), Ne(950)])), `test_compound_and_as_between` (verify And([Ge(lo), Le(hi)]) matches Between(lo,hi))
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
- **Done when**: Compound predicate tests pass, including nested
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter -- compound --nocapture 2>&1 | tail -20`
- **Commit**: `feat(forge-filter): compound predicates — AND/OR with BETWEEN optimization`
- _Requirements: FR-4, AC-3.1 through AC-3.5_
- _Design: Compound predicate implementation_

### Task 2.8: [VERIFY] Quality checkpoint: full test suite + clippy

- **Do**: Run all tests and clippy
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter && cargo clippy -p forge-filter -- -D warnings`
- **Done when**: All commands exit 0
- **Commit**: `chore(forge-filter): pass quality checkpoint` (only if fixes needed)

## Phase 3: Testing

Focus: Comprehensive correctness tests, property-based tests, edge cases.

### Task 3.1: Exhaustive type x operator matrix tests (42 combos)

- **Do**:
  1. Write a parameterized test helper `fn test_filter_correctness<T: FilterKey + PartialOrd + std::fmt::Debug>(data: &[T], pred: &Predicate<T>, filter: &mut GpuFilter)` that compares GPU output to CPU reference for each combo
  2. Generate test data for each type: u32 (random 0..1M), i32 (random -500K..500K), f32 (random 0.0..1.0), u64 (random 0..1M as u64), i64 (random -500K..500K as i64), f64 (random 0.0..1.0 as f64)
  3. Write tests for all 42 combos: 6 types x 7 predicates (Gt, Lt, Ge, Le, Eq, Ne, Between). Each at 100K elements for speed (full 42 combo suite should finish < 60s).
  4. Use threshold values that give ~50% selectivity for meaningful testing
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs` (tests module)
- **Done when**: All 42 tests pass
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter -- test_filter_correctness --nocapture 2>&1 | tail -50`
- **Commit**: `test(forge-filter): exhaustive 42-combo type x operator correctness matrix`
- _Requirements: AC-1.5, FR-1, FR-2, FR-3_
- _Design: Test Strategy — Unit Tests_

### Task 3.2: Edge case tests

- **Do**:
  1. `test_filter_empty_input`: verify FilterError::EmptyInput or Ok with empty result for 0 elements
  2. `test_filter_single_element_match`: 1 element that matches
  3. `test_filter_single_element_no_match`: 1 element that doesn't match
  4. `test_filter_all_same_value`: all elements identical, test Eq (all match) and Gt (none match)
  5. `test_filter_max_min_values`: u32::MAX, u32::MIN as thresholds
  6. `test_filter_f32_nan`: NaN values excluded by all predicates except Ne
  7. `test_filter_f32_inf`: +/- infinity handling
  8. `test_filter_between_inverted`: Between(hi, lo) returns empty
  9. `test_filter_between_eq`: Between(x, x) matches Eq(x) exactly
  10. `test_filter_preserves_order`: verify output[i] < output[i+1] index-wise for ascending input
  11. `test_filter_buffer_zero_copy`: alloc FilterBuffer, write via as_mut_slice, filter, read result.as_slice — no intermediate Vec allocations
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs` (tests module)
- **Done when**: All edge case tests pass
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter -- test_filter_ --nocapture 2>&1 | tail -30`
- **Commit**: `test(forge-filter): edge case tests — NaN, inf, empty, single element, order preservation`
- _Requirements: FR-17, AC-1.4, AC-4.3_
- _Design: Edge Cases_

### Task 3.3: Large input tests (16M, 64M)

- **Do**:
  1. `test_filter_u32_16m`: 16M random u32, Gt(threshold at 50%), verify count matches CPU reference, spot-check first/last 1000 elements
  2. `test_filter_u32_64m`: 64M random u32, Gt(threshold at 50%), verify count only (full comparison too slow). This also validates the hierarchical scan path if >4096 tiles.
  3. `test_filter_f32_16m`: 16M random f32, Lt(0.5), verify count
  4. `test_filter_u64_16m`: 16M random u64, Gt(threshold), verify count — tests 64-bit path at scale
  5. Print timing info (wall clock) for each test to give early perf signal
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs` (tests module)
- **Done when**: All large tests pass with correct counts
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter --release -- test_filter_u32_16m test_filter_u32_64m test_filter_f32_16m test_filter_u64_16m --nocapture 2>&1 | tail -30`
- **Commit**: `test(forge-filter): large input tests at 16M and 64M elements`
- _Requirements: NFR-1, NFR-2_
- _Design: Hierarchical Scan_

### Task 3.4: [VERIFY] Quality checkpoint: full test suite + clippy + check

- **Do**: Run complete quality suite
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check --workspace && cargo clippy -p forge-filter -- -D warnings && cargo test -p forge-filter`
- **Done when**: All commands exit 0
- **Commit**: `chore(forge-filter): pass quality checkpoint` (only if fixes needed)

## Phase 4: Polish + Benchmarks

Focus: Doc comments, benchmarks, README, publish preparation.

### Task 4.1: Add doc comments to all public items

- **Do**:
  1. Add module-level rustdoc to `lib.rs` (mirror forge-sort's: description, supported types, quick start example showing 5-line usage, requirements, license)
  2. Doc comments on: `GpuFilter` (struct + all pub methods), `FilterBuffer<T>` (struct + all methods), `FilterResult<T>` (struct + all methods), `Predicate<T>` (enum + all variants), `FilterKey` (trait), `FilterError` (enum + all variants)
  3. Add `#![warn(missing_docs)]` to lib.rs
  4. Include code examples in GpuFilter::filter and filter_u32 docs
  5. Add f64 performance warning in FilterKey docs for f64 impl (FR-15)
- **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
- **Done when**: `cargo doc -p forge-filter --no-deps` succeeds with no missing_docs warnings
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && RUSTDOCFLAGS="-D warnings" cargo doc -p forge-filter --no-deps 2>&1 | tail -10`
- **Commit**: `docs(forge-filter): complete rustdoc for all public items`
- _Requirements: NFR-10, FR-15, NFR-7_

### Task 4.2: Criterion benchmarks

- **Do**:
  1. Create `metal-forge-compute/forge-filter/benches/filter_benchmark.rs`
  2. Add Criterion benchmark groups:
     - `filter_u32_gt`: 1M, 4M, 16M at 50% selectivity
     - `filter_u32_selectivity`: 16M at 1%, 10%, 50%, 90%, 99% selectivity
     - `filter_f32_between`: 16M at 50% selectivity
     - `filter_u64_gt`: 16M at 50% selectivity
     - `filter_u32_unordered_vs_ordered`: 16M at 50% (compare both modes)
     - `filter_u32_indices`: 16M at 50% (index-only mode)
  3. Each benchmark: create GpuFilter once (setup), pre-allocate FilterBuffer, run filter in measured loop. Report iterations and mean time.
  4. Add comment documenting Polars baselines: u32 16M = 5.8ms (2780 Mrows/s), u32 4M = 0.89ms (4489 Mrows/s)
  5. Ensure `[[bench]] name = "filter_benchmark" harness = false` is in Cargo.toml
- **Files**: `metal-forge-compute/forge-filter/benches/filter_benchmark.rs`, `metal-forge-compute/forge-filter/Cargo.toml`
- **Done when**: `cargo bench -p forge-filter` runs and produces timing output
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo bench -p forge-filter -- --quick 2>&1 | grep -E "(filter_|time:)" | head -20`
- **Commit**: `feat(forge-filter): Criterion benchmarks — u32/f32/u64, selectivity sweep, unordered comparison`
- _Requirements: FR-14, AC-8.1 through AC-8.5_
- _Design: Benchmarks (Criterion)_

### Task 4.3: [VERIFY] Full local CI: build + test + clippy + doc + bench

- **Do**: Run complete local CI suite
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-filter --release && cargo test -p forge-filter && cargo clippy -p forge-filter -- -D warnings && RUSTDOCFLAGS="-D warnings" cargo doc -p forge-filter --no-deps && cargo bench -p forge-filter -- --quick 2>&1 | tail -5`
- **Done when**: All commands pass
- **Commit**: `chore(forge-filter): pass full local CI` (only if fixes needed)

### Task 4.4: Verify 10x+ speedup over Polars

- **Do**:
  1. Run `cargo bench -p forge-filter -- filter_u32_gt/16M` and capture mean time
  2. Compare against documented Polars baseline (5.8ms at 16M)
  3. Calculate speedup ratio. Target: >= 10x at low selectivity, >= 7x at 50%
  4. If < 7x at 50%, profile and optimize: check if scan_partials is bottleneck, verify single command buffer, check for unnecessary barriers
  5. Document actual measured numbers in a comment in the benchmark file
- **Files**: `metal-forge-compute/forge-filter/benches/filter_benchmark.rs`
- **Done when**: Benchmark results documented, 10x+ achieved at <=25% selectivity
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo bench -p forge-filter -- "filter_u32_gt/16M" 2>&1 | grep "time:" | head -5`
- **Commit**: `feat(forge-filter): verified 10x+ speedup over Polars at 16M`
- _Requirements: NFR-1, NFR-2, AC-8.5_
- _Design: Performance Model_

### Task 4.5: [VERIFY] AC checklist verification

- **Do**: Programmatically verify each acceptance criterion:
  1. AC-1.1 through AC-1.5: `cargo test -p forge-filter -- test_filter_correctness` (42 combos)
  2. AC-2.1 through AC-2.4: `cargo test -p forge-filter -- between`
  3. AC-3.1 through AC-3.5: `cargo test -p forge-filter -- compound`
  4. AC-4.1 through AC-4.5: `cargo test -p forge-filter -- zero_copy`
  5. AC-5.1 through AC-5.4: `cargo test -p forge-filter -- indices`
  6. AC-6.1 through AC-6.4: `cargo test -p forge-filter -- unordered`
  7. AC-7.1 through AC-7.4: `cargo test -p forge-filter -- convenience` (filter_u32 etc.)
  8. AC-8.1 through AC-8.5: `cargo bench -p forge-filter -- --quick`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter 2>&1 | tail -5`
- **Done when**: All acceptance criteria confirmed met via test results
- **Commit**: None

## Phase 5: PR Lifecycle

### Task 5.1: Create PR and verify CI

- **Do**:
  1. Verify on feature branch: `git branch --show-current`
  2. Stage all forge-filter files + workspace Cargo.toml update
  3. Push branch: `git push -u origin $(git branch --show-current)`
  4. Create PR: `gh pr create --title "feat(forge-filter): GPU filter+compact for numeric columns" --body "..."`
  5. PR body: Summary of forge-filter crate, 10x+ over Polars claim, test matrix coverage, benchmark results
- **Verify**: `gh pr checks` shows all green (or no CI configured — document)
- **Done when**: PR created and pushed
- **Commit**: None (PR creation, not a code commit)

### Task 5.2: Address review feedback

- **Do**: Monitor `gh pr view` for review comments, address each
- **Verify**: `gh pr checks` remains green after fixes
- **Done when**: PR approved or no outstanding review comments
- **Commit**: As needed per feedback

### Task 5.3: Publish to crates.io

- **Do**:
  1. Verify Cargo.toml metadata complete: name, version, description, license, repository, homepage, documentation, keywords, categories, include
  2. Run `cargo package -p forge-filter --list` to verify included files
  3. Run `cargo publish -p forge-filter --dry-run` to verify publishability
  4. If dry-run passes: `cargo publish -p forge-filter`
- **Verify**: `cargo publish -p forge-filter --dry-run` exits 0
- **Done when**: Package published (or dry-run verified if awaiting PR merge)
- **Commit**: None (publish action)

## Notes

- **POC shortcuts taken**: Phase 1 only tests u32 Gt. 64-bit types, other operators, compound predicates, index output, unordered mode all deferred to Phase 2.
- **Production TODOs**:
  - Hierarchical scan for >16.7M elements (64M test in 3.3 will reveal if this is needed — it is, 64M/4096=15625 tiles > 4096 max)
  - Auto-detect BETWEEN from And([Ge, Le]) pattern
  - `filter_into()` for caller-provided output buffers
  - Multi-column compound predicates (deferred to future spec)
- **Scan partials limit**: 256 threads x 16 elements = 4096 partials max. For 32-bit: 4096 * 4096 = 16.7M elements. For 64-bit: 4096 * 2048 = 8.4M. For >16.7M, need hierarchical scan (2 extra dispatches). Must be implemented before 64M test in 3.3.
- **PSO count**: 15 pre-compiled at init + lazy compilation for 64-bit variants and index/unordered modes
- **Forge-sort reference files**: `metal-forge-compute/forge-sort/src/lib.rs` (GpuSorter, SortBuffer, SortKey, SortError patterns), `metal-forge-compute/forge-sort/src/metal_helpers.rs` (PsoCache, FnConstant, alloc_buffer), `metal-forge-compute/forge-sort/build.rs` (shader build), `metal-forge-compute/forge-sort/Cargo.toml` (crate metadata)
