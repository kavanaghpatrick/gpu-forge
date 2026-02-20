---
spec: gpu-filter-v2
phase: tasks
total_tasks: 34
created: 2026-02-20
---

# Tasks: forge-filter v2 — Bitmap-Cached Ordered Mode + Multi-Column + Arrow + NULL

## Phase 1: Make It Work (POC)

Focus: Bitmap-cached ordered mode working end-to-end with u32, benchmarked at >=10x vs Polars.

- [x] 1.1 Add `filter_bitmap_scan` kernel to filter.metal
  - **Do**:
    1. Add `HAS_NULLS` function constant (index 5) with default false
    2. Add `filter_bitmap_scan` kernel that:
       - Evaluates predicate for each element (same logic as `filter_predicate_scan`)
       - Uses `simd_ballot` to pack 32 predicate results into one uint per simdgroup
       - Lane 0 of each simdgroup writes ballot word to `bitmap[word_idx]` via `atomic_store_explicit` (relaxed)
       - Computes per-thread match count via `popcount` of ballot within lane 0, or direct `pred_result ? 1 : 0` for all lanes
       - SIMD prefix sum + cross-SG aggregation same as existing `filter_predicate_scan`
       - Writes tile total to `partials[tg_idx]`
    3. Buffer layout: `buffer(0)=src, buffer(1)=bitmap (atomic_uint), buffer(2)=partials, buffer(3)=validity (optional), buffer(4)=params`
    4. Support both 32-bit and 64-bit paths via IS_64BIT function constant
    5. When `has_nulls`, read validity bitmap and AND with predicate result before ballot
  - **Files**: `metal-forge-compute/forge-filter/shaders/filter.metal`
  - **Done when**: Kernel compiles with `xcrun metal -std=metal3.2 -c shaders/filter.metal`
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(forge-filter): add filter_bitmap_scan kernel with simd_ballot bitmap`
  - _Requirements: FR-1, AC-1.1, AC-1.3_
  - _Design: Section 1 — New Metal Kernels_

- [ ] 1.2 Add `filter_bitmap_scatter` kernel to filter.metal
  - **Do**:
    1. Add `filter_bitmap_scatter` kernel that:
       - Reads cached bitmap word `bitmap[idx / 32]`, extracts bit `(idx % 32)`
       - No predicate re-evaluation needed
       - Uses SIMD prefix sum for local scatter offset (same round-by-round pattern as `filter_scatter`)
       - Reads global prefix from scanned `partials[tg_idx]`
       - Writes matching values to output (and optionally indices)
    2. Buffer layout: `buffer(0)=src, buffer(1)=bitmap (const uint), buffer(2)=out_vals, buffer(3)=out_idx, buffer(4)=partials, buffer(5)=params`
    3. Use same function constants: IS_64BIT, OUTPUT_IDX, OUTPUT_VALS, DATA_TYPE
    4. Support both 32-bit and 64-bit scatter
  - **Files**: `metal-forge-compute/forge-filter/shaders/filter.metal`
  - **Done when**: Both new kernels compile; existing 4 kernels still present
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(forge-filter): add filter_bitmap_scatter kernel`
  - _Requirements: FR-1, AC-1.3_
  - _Design: Section 1 — filter_bitmap_scatter_

- [ ] 1.3 Wire bitmap pipeline into Rust dispatch path
  - **Do**:
    1. Add `buf_bitmap: Option<Retained<...>>` scratch buffer to GpuFilter struct
    2. In `ensure_scratch_buffers`, allocate bitmap buffer: `ceil(n / 32) * 4` bytes (one u32 per 32 elements)
    3. Add `dispatch_filter_bitmap<T>()` method that:
       - Dispatch 1: `filter_bitmap_scan` (predicate + bitmap write + tile counts)
       - Dispatch 2: `encode_scan_partials` (unchanged — reuse existing)
       - Dispatch 3: `filter_bitmap_scatter` (bitmap read + scatter)
       - All 3 dispatches within single command buffer + encoder
    4. PSO creation for new kernels: lazy via `pso_cache.get_or_create_specialized()`
    5. New function constant bindings: same PRED_TYPE(0), IS_64BIT(1), OUTPUT_IDX(2), OUTPUT_VALS(3), DATA_TYPE(4)
    6. Modify `filter()` method to call `dispatch_filter_bitmap()` instead of `dispatch_filter()` for ordered mode
    7. Keep `dispatch_filter()` intact as fallback
  - **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
  - **Done when**: `filter()` method uses bitmap pipeline internally; all 47 existing tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test 2>&1 | tail -5`
  - **Commit**: `feat(forge-filter): wire bitmap-cached pipeline for ordered mode`
  - _Requirements: FR-1, FR-2, AC-1.3, AC-1.5_
  - _Design: Section 3 — GpuFilter Methods_

- [ ] 1.4 [VERIFY] Quality checkpoint: all v0.1 tests pass with bitmap pipeline
  - **Do**:
    1. Run full test suite — all 47 tests must pass
    2. Run `cargo clippy -- -D warnings`
    3. If any failures, fix bitmap kernel/dispatch bugs
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test 2>&1 | grep "test result" && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: 47 tests pass, zero clippy warnings
  - **Commit**: `chore(forge-filter): pass quality checkpoint after bitmap pipeline` (only if fixes needed)

- [ ] 1.5 Add bitmap-cached ordered benchmark and measure speedup
  - **Do**:
    1. Add `filter_u32_bitmap_ordered` benchmark group to `filter_benchmark.rs`
    2. Benchmark: 16M u32, 50% selectivity, ordered mode (now uses bitmap internally)
    3. Add selectivity sweep: 1%, 10%, 50%, 90%, 99% at 16M
    4. Run benchmark, record times in comments at top of file
    5. Compare vs Polars baseline (5,800us) — target <=580us for 10x
    6. If >=10x not achieved, investigate and tune (SLC residency, tile size, bitmap packing)
  - **Files**: `metal-forge-compute/forge-filter/benches/filter_benchmark.rs`
  - **Done when**: Benchmark runs; ordered mode is measurably faster than v0.1 baseline (848us)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo bench --bench filter_benchmark -- filter_u32_gt 2>&1 | grep -E "(time|ordered)" | head -10`
  - **Commit**: `feat(forge-filter): benchmark bitmap-cached ordered mode`
  - _Requirements: AC-1.1, AC-1.2, NFR-1_
  - _Design: Performance Projections_

- [ ] 1.6 Add 10,000-iteration correctness oracle test
  - **Do**:
    1. Add test `test_bitmap_correctness_10k_iterations` that:
       - For 10,000 iterations with ChaCha8Rng(seed=42+i):
       - Generate 100K random u32 values
       - Run GPU filter ordered mode (bitmap-cached)
       - Run CPU reference filter
       - Assert exact match (count + every element)
    2. Add test `test_bitmap_vs_v1_oracle_all_types` that:
       - For each type (u32, i32, f32, u64, i64, f64):
       - 1M elements, multiple predicates (Gt, Lt, Between)
       - Assert GPU output bit-identical to CPU reference
  - **Files**: `metal-forge-compute/forge-filter/src/lib.rs` (in `#[cfg(test)]` module)
  - **Done when**: Oracle test passes with zero mismatches across all 10K iterations
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test test_bitmap_correctness -- --nocapture 2>&1 | tail -10`
  - **Commit**: `test(forge-filter): add 10K-iteration bitmap correctness oracle`
  - _Requirements: AC-1.3, AC-1.6_

- [ ] 1.7 POC Checkpoint: bitmap-cached ordered mode validated
  - **Do**:
    1. Run full test suite (47 existing + new bitmap tests)
    2. Run benchmark and record ordered mode speedup ratio
    3. Verify bitmap pipeline is the active path for `filter()`
    4. Document POC results in benchmark file comments
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test 2>&1 | grep "test result" && cargo bench --bench filter_benchmark -- filter_u32_gt/ordered/16M 2>&1 | grep "time:"  | head -5`
  - **Done when**: All tests pass, benchmark shows measurable improvement
  - **Commit**: `feat(forge-filter): complete bitmap-cached ordered mode POC`
  - _Requirements: AC-1.1 through AC-1.6_

## Phase 2: Core Features

### 2A: BooleanMask Type + Multi-Column Filter

- [ ] 2.1 Add BooleanMask type and filter_mask method
  - **Do**:
    1. Add `BooleanMask` struct to lib.rs:
       - `buffer: Retained<ProtocolObject<dyn MTLBuffer>>` (packed u32 words)
       - `len: usize` (number of elements/bits)
       - `count: usize` (number of true bits)
    2. Implement `BooleanMask::to_vec() -> Vec<bool>`, `BooleanMask::len()`, `BooleanMask::count()`
    3. Add `GpuFilter::filter_mask<T>()` that runs bitmap_scan + scan_partials, returns BooleanMask
       - Reuses dispatch_filter_bitmap's first 2 passes (no scatter)
       - Reads count from partials scan total
    4. Add `GpuFilter::gather<T>()` that takes data + BooleanMask, runs scatter pass only
    5. Export BooleanMask in public API
  - **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
  - **Done when**: `filter_mask()` returns correct BooleanMask; `gather()` with that mask produces same output as `filter()`
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test test_filter_mask 2>&1 | tail -5`
  - **Commit**: `feat(forge-filter): add BooleanMask type with filter_mask and gather`
  - _Requirements: FR-3, FR-13, AC-2.4_
  - _Design: Section 2 — BooleanMask_

- [ ] 2.2 Add multi-column predicate bitmap kernel
  - **Do**:
    1. Add `filter_multi_bitmap_scan` kernel to filter.metal that:
       - Accepts up to 4 column buffers via buffer bindings (buffer 0-3)
       - Function constants: N_COLUMNS (uint, index 6), PRED_TYPE_A..D (indices 7-10), DATA_TYPE_A..D (indices 11-14), IS_64BIT_A..D (indices 15-18), LOGIC_AND (bool, index 19)
       - Each thread evaluates predicates for all columns, combines with AND or OR
       - Uses simd_ballot on combined result, writes bitmap + partials
    2. Add params struct for multi-column: thresholds for each column packed in buffer
    3. Keep kernel complexity bounded: 4 columns max in single kernel
  - **Files**: `metal-forge-compute/forge-filter/shaders/filter.metal`
  - **Done when**: Multi-column kernel compiles
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(forge-filter): add multi-column predicate bitmap kernel`
  - _Requirements: FR-3, FR-4_
  - _Design: Section 1 — Multi-Column Pipeline_

- [ ] 2.3 Wire multi-column Rust API
  - **Do**:
    1. Add `LogicOp` enum: `And`, `Or`
    2. Add `GpuFilter::filter_multi_mask()` method accepting vec of (data_slice, predicate) pairs + LogicOp
    3. Dispatch multi_bitmap_scan + scan_partials, return BooleanMask
    4. Validate N=1..4 columns, return `FilterError::InvalidPredicate` for N=0 or N>4
    5. Handle mixed types by dispatching separate bitmap_scan per column, then AND/OR bitmaps on GPU
       (Alternative: fused kernel. Start with separate-scan approach for correctness, optimize later)
    6. Add unit tests: 2-col AND u32, 2-col OR, mixed types, edge cases (all-match, zero-match)
  - **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
  - **Done when**: `filter_multi_mask()` returns correct BooleanMask for 2-column AND case vs CPU reference
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test test_multi 2>&1 | tail -10`
  - **Commit**: `feat(forge-filter): add filter_multi_mask API for multi-column filter`
  - _Requirements: FR-3, AC-2.1, AC-2.2, AC-2.5, AC-2.6, AC-2.7_
  - _Design: Section 3 — filter_multi_mask_

- [ ] 2.4 [VERIFY] Quality checkpoint: multi-column correctness + all v0.1 tests
  - **Do**: Run full test suite + clippy
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test 2>&1 | grep "test result" && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass, zero clippy warnings
  - **Commit**: `chore(forge-filter): pass quality checkpoint after multi-column` (only if fixes needed)

### 2B: NULL Bitmap Support

- [ ] 2.5 Add NULL bitmap handling to bitmap kernels
  - **Do**:
    1. Ensure `HAS_NULLS` function constant (index 5) already defined from task 1.1
    2. In `filter_bitmap_scan`: when `has_nulls=true`, read validity bitmap from buffer(3), extract bit for each element, AND with predicate result before ballot
    3. Validity bitmap format: packed u32, LSB-first (Arrow convention): bit_idx = idx % 32, word_idx = idx / 32, valid = (word >> bit_idx) & 1
    4. Add Rust-side: `filter_nullable()` method on GpuFilter:
       - Accepts data + predicate + validity bitmap (as `&[u8]`)
       - Copies validity bitmap to Metal buffer (page-aligned)
       - Sets HAS_NULLS=true function constant on PSO
       - Returns FilterResult (NULLs excluded)
    5. Add `filter_mask_nullable()` that returns BooleanMask (NULLs excluded)
    6. Add tests:
       - Every 1000th element NULL, verify excluded
       - All-NULL returns empty
       - No-NULL identical to non-nullable path
       - NULL at boundaries (0, 7, 8, 15, 31, 32)
  - **Files**: `metal-forge-compute/forge-filter/shaders/filter.metal`, `metal-forge-compute/forge-filter/src/lib.rs`
  - **Done when**: Nullable filter excludes NULLs correctly; boundary tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test test_null 2>&1 | tail -10`
  - **Commit**: `feat(forge-filter): add NULL bitmap support via HAS_NULLS function constant`
  - _Requirements: FR-8, AC-4.1 through AC-4.8_
  - _Design: Section 1 — HAS_NULLS constant_

- [ ] 2.6 [VERIFY] Quality checkpoint: NULL + multi-column + v0.1 tests
  - **Do**: Run full test suite + clippy
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test 2>&1 | grep "test result" && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass, zero clippy warnings
  - **Commit**: `chore(forge-filter): pass quality checkpoint after NULL support` (only if fixes needed)

### 2C: Arrow Integration

- [ ] 2.7 Add arrow feature flag and ArrowFilterKey trait
  - **Do**:
    1. Add to Cargo.toml:
       ```toml
       [features]
       default = []
       arrow = ["dep:arrow-array", "dep:arrow-buffer", "dep:arrow-schema"]

       [dependencies]
       arrow-array = { version = "54", optional = true }
       arrow-buffer = { version = "54", optional = true }
       arrow-schema = { version = "54", optional = true }
       ```
    2. Create `src/arrow.rs` with `#[cfg(feature = "arrow")]`
    3. Define `ArrowFilterKey` trait mapping FilterKey types to Arrow primitive types:
       - u32 -> UInt32Type, i32 -> Int32Type, f32 -> Float32Type
       - u64 -> UInt64Type, i64 -> Int64Type, f64 -> Float64Type
    4. Add `mod arrow;` to lib.rs behind `#[cfg(feature = "arrow")]`
    5. Re-export Arrow types from lib.rs
  - **Files**: `metal-forge-compute/forge-filter/Cargo.toml`, `metal-forge-compute/forge-filter/src/arrow.rs`, `metal-forge-compute/forge-filter/src/lib.rs`
  - **Done when**: `cargo build --features arrow` succeeds; `cargo build` (no arrow) still works
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo build --features arrow 2>&1 | tail -5 && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(forge-filter): add arrow feature flag and ArrowFilterKey trait`
  - _Requirements: FR-5, FR-7, AC-3.7_
  - _Design: Section 4 — Arrow Feature Flag_

- [ ] 2.8 Implement filter_arrow and filter_arrow_nullable
  - **Do**:
    1. In `src/arrow.rs`, implement `GpuFilter::filter_arrow<T: ArrowFilterKey>()`:
       - Extract values buffer from `PrimitiveArray<T::ArrowType>`
       - Copy to page-aligned Metal buffer via `alloc_filter_buffer` + `copy_from_slice`
       - Run `filter()` on the Metal buffer
       - Construct output `PrimitiveArray` from result
    2. Implement `GpuFilter::filter_arrow_nullable<T>()`:
       - Extract validity bitmap from Arrow array's null buffer
       - Convert to packed u32 format (Arrow uses LSB-first u8, need u32 words)
       - Call `filter_nullable()` with validity bitmap
       - Return `PrimitiveArray` with no null buffer (NULLs excluded from output)
    3. Handle array offset: `array.offset()` must be accounted for when copying data
    4. Add tests:
       - `test_filter_arrow_u32_basic`: PrimitiveArray<UInt32Type> filter matches filter_u32
       - `test_filter_arrow_all_types`: all 6 types
       - `test_filter_arrow_nullable_basic`: NULLs excluded
       - `test_filter_arrow_sliced`: array with non-zero offset
       - `test_filter_arrow_empty`: 0-length array
  - **Files**: `metal-forge-compute/forge-filter/src/arrow.rs`
  - **Done when**: All Arrow tests pass; `filter_arrow` output matches `filter_u32` for identical data
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test --features arrow test_filter_arrow 2>&1 | tail -10`
  - **Commit**: `feat(forge-filter): implement filter_arrow and filter_arrow_nullable`
  - _Requirements: FR-5, FR-6, AC-3.1 through AC-3.6, AC-3.8_
  - _Design: Section 3 — Arrow integration_

- [ ] 2.9 [VERIFY] Quality checkpoint: Arrow + NULL + multi-column + v0.1
  - **Do**: Run full test suite with and without arrow feature + clippy
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test 2>&1 | grep "test result" && cargo test --features arrow 2>&1 | grep "test result" && cargo clippy --features arrow -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass both ways, zero clippy warnings
  - **Commit**: `chore(forge-filter): pass quality checkpoint after Arrow integration` (only if fixes needed)

### 2D: API Cleanup and Version Bump

- [ ] 2.10 Version bump to 0.2.0 and API audit
  - **Do**:
    1. Bump version in Cargo.toml: `version = "0.2.0"`
    2. Audit all new public types are documented (`#![warn(missing_docs)]` already set)
    3. Add doc comments to: BooleanMask, LogicOp, filter_mask, filter_multi_mask, gather, filter_nullable, filter_mask_nullable
    4. Add `#[cfg(feature = "arrow")]` doc comments to arrow types
    5. Verify `cargo doc --no-deps --features arrow` produces clean output
    6. Verify no breaking changes: all v0.1 public types still exported (GpuFilter, FilterBuffer, FilterResult, Predicate, FilterKey, FilterError, FilterParams, FilterParams64)
  - **Files**: `metal-forge-compute/forge-filter/Cargo.toml`, `metal-forge-compute/forge-filter/src/lib.rs`, `metal-forge-compute/forge-filter/src/arrow.rs`
  - **Done when**: `cargo doc --no-deps --features arrow` succeeds with no warnings; all 47 v0.1 tests still pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo doc --no-deps --features arrow 2>&1 | grep -E "(warning|error)" | head -10 && cargo test 2>&1 | grep "test result"`
  - **Commit**: `feat(forge-filter): bump to v0.2.0 with full API documentation`
  - _Requirements: FR-14, AC-1.5, NFR-8_

## Phase 3: Testing

- [ ] 3.1 Comprehensive ordered mode correctness tests
  - **Do**:
    1. Add `test_bitmap_ordered_all_types_all_preds` test:
       - For each type in (u32, i32, f32, u64, i64, f64):
       - For each predicate (Gt, Lt, Ge, Le, Eq, Ne, Between):
       - 100K elements, ChaCha8Rng seed 42
       - Assert GPU == CPU reference (full comparison)
    2. Add `test_bitmap_selectivity_sweep` test:
       - 16M u32, selectivities 0%, 1%, 10%, 50%, 90%, 99%, 100%
       - Assert count and first/last 100 elements match CPU reference
    3. Add `test_bitmap_indices_ascending_16m`:
       - filter_with_indices at 16M, verify indices strictly ascending
    4. Add `test_bitmap_tile_boundary`:
       - Data of size exactly TILE_SIZE, TILE_SIZE+1, TILE_SIZE-1
       - Verify correctness at tile boundaries
  - **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
  - **Done when**: All new correctness tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test test_bitmap 2>&1 | grep -E "(test|ok|FAILED)" | tail -20`
  - **Commit**: `test(forge-filter): add comprehensive bitmap ordered mode tests`
  - _Requirements: AC-1.3, AC-1.4, AC-1.6_

- [ ] 3.2 Multi-column filter test matrix
  - **Do**:
    1. Add parameterized test covering:
       - N_cols in [2, 3, 4]
       - Type combos: (u32,u32), (i32,f32), (u64,f64), (u32,i32,f32)
       - Pred combos: (Gt,Lt), (Between,Ne), (Eq,Ge)
       - Logic: AND and OR
       - 100K rows each, ChaCha8Rng seed 42
    2. Add edge case tests:
       - Single column via multi API (should == single-column filter)
       - All columns all-match
       - All columns zero-match
       - Mixed: col A high selectivity, col B low selectivity
    3. All tests compare GPU result against CPU element-wise evaluation
  - **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
  - **Done when**: All multi-column test matrix tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test test_multi 2>&1 | grep -E "(test|ok|FAILED)" | tail -20`
  - **Commit**: `test(forge-filter): add multi-column filter test matrix`
  - _Requirements: AC-2.1 through AC-2.7_

- [ ] 3.3 NULL bitmap edge case tests
  - **Do**:
    1. Add tests:
       - NULL at position 0 (first element)
       - NULL at position N-1 (last element)
       - NULL at tile boundaries (4095, 4096, 4097, 8191, 8192)
       - Alternating NULL/valid pattern
       - 1% NULL density at 1M rows
       - NaN + NULL interaction: f32 with both NaN and NULL values
       - NaN follows IEEE 754, NULL always excluded, verify independent treatment
    2. Each test compares GPU result vs CPU reference that respects both NaN and NULL semantics
  - **Files**: `metal-forge-compute/forge-filter/src/lib.rs`
  - **Done when**: All NULL edge case tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test test_null 2>&1 | grep -E "(test|ok|FAILED)" | tail -20`
  - **Commit**: `test(forge-filter): add NULL bitmap edge case tests`
  - _Requirements: AC-4.1 through AC-4.8_

- [ ] 3.4 [VERIFY] Quality checkpoint: full test suite
  - **Do**: Run complete test suite including arrow feature
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test 2>&1 | grep "test result" && cargo test --features arrow 2>&1 | grep "test result" && cargo clippy --features arrow -- -D warnings 2>&1 | tail -5`
  - **Done when**: All tests pass, zero clippy warnings
  - **Commit**: `chore(forge-filter): pass quality checkpoint after full test suite` (only if fixes needed)

- [ ] 3.5 Arrow integration tests
  - **Do**:
    1. Add tests behind `#[cfg(feature = "arrow")]`:
       - `test_filter_arrow_matches_filter_u32`: identical data, verify output matches
       - `test_filter_arrow_all_types`: UInt32, Int32, Float32, UInt64, Int64, Float64
       - `test_filter_arrow_nullable_exclusion`: NULLs excluded
       - `test_filter_arrow_nullable_all_null`: empty result
       - `test_filter_arrow_nullable_no_null`: identical to non-nullable
       - `test_filter_arrow_sliced_offset`: array.slice() with non-zero offset
       - `test_filter_arrow_empty`: 0-length array
       - `test_filter_arrow_copy_overhead`: verify copy takes <=0.5ms for 64MB
    2. Each test compares with CPU reference or equivalent non-arrow API call
  - **Files**: `metal-forge-compute/forge-filter/src/arrow.rs`
  - **Done when**: All Arrow tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test --features arrow test_filter_arrow 2>&1 | grep -E "(test|ok|FAILED)" | tail -20`
  - **Commit**: `test(forge-filter): add comprehensive Arrow integration tests`
  - _Requirements: AC-3.1 through AC-3.8_

- [ ] 3.6 Property tests with proptest
  - **Do**:
    1. Add `proptest` to dev-dependencies in Cargo.toml
    2. Add property tests:
       - `prop_filter_output_sorted`: for any u32 data + simple predicate, output is sorted ascending
       - `prop_filter_output_subset`: all output values exist in input
       - `prop_filter_count_matches_cpu`: GPU count == CPU count for any data + pred
       - `prop_multi_and_subset_of_single`: multi-column AND result is subset of each single-column result
       - `prop_indices_in_bounds`: all returned indices < input length
       - `prop_indices_ascending`: indices strictly ascending
    3. Use ProptestConfig with 1000 cases per test, fixed seed for CI reproducibility
  - **Files**: `metal-forge-compute/forge-filter/Cargo.toml`, `metal-forge-compute/forge-filter/src/lib.rs`
  - **Done when**: All property tests pass 1000 cases
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test prop_ 2>&1 | grep -E "(test|ok|FAILED)" | tail -20`
  - **Commit**: `test(forge-filter): add proptest property-based tests`
  - _Requirements: NFR-9_

- [ ] 3.7 [VERIFY] Quality checkpoint: all tests including proptest
  - **Do**: Run complete test suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test --features arrow 2>&1 | grep "test result"`
  - **Done when**: All tests pass (target: >=120 total tests)
  - **Commit**: `chore(forge-filter): pass quality checkpoint — full test suite` (only if fixes needed)

## Phase 4: Quality Gates

- [ ] 4.1 Update benchmarks with all v2 features
  - **Do**:
    1. Add benchmark groups:
       - `filter_bitmap_ordered`: 16M u32 at 1%, 50%, 99% selectivity (bitmap pipeline)
       - `filter_bitmap_all_types`: 16M each of u32, i32, f32, u64, i64, f64 at 50% sel
       - `filter_multi_2col_and`: 16M rows, 2 columns, 50% per-column selectivity
       - `filter_multi_3col_and`: 16M rows, 3 columns
       - `filter_nullable_overhead`: 16M u32 with and without NULL bitmap (measure overhead %)
       - `filter_arrow_e2e`: 16M u32 Arrow array, end-to-end including copy
    2. Update comments with measured results
    3. Print Polars comparison ratios
  - **Files**: `metal-forge-compute/forge-filter/benches/filter_benchmark.rs`
  - **Done when**: All benchmarks run; ordered mode speedup documented
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo bench --bench filter_benchmark 2>&1 | grep "time:" | head -20`
  - **Commit**: `bench(forge-filter): add v2 benchmarks for bitmap, multi-column, NULL, Arrow`
  - _Requirements: NFR-1 through NFR-6_

- [ ] 4.2 Update README.md with v2 features and benchmarks
  - **Do**:
    1. Update README with:
       - v0.2.0 features list
       - Updated benchmark table with bitmap-cached ordered results
       - Multi-column filter usage example
       - Arrow integration usage example (behind feature flag)
       - NULL handling example
       - API additions section
    2. Update quick start example if needed
    3. Ensure all code examples compile (test with `cargo test --doc`)
  - **Files**: `metal-forge-compute/forge-filter/README.md`
  - **Done when**: README reflects v0.2 features; doc tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test --doc 2>&1 | grep "test result"`
  - **Commit**: `docs(forge-filter): update README with v0.2 features and benchmarks`
  - _Requirements: NFR-8_

- [ ] 4.3 [VERIFY] Full local CI: test + clippy + doc + build
  - **Do**: Run complete local CI suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test --features arrow 2>&1 | grep "test result" && cargo clippy --features arrow -- -D warnings 2>&1 | tail -3 && cargo doc --no-deps --features arrow 2>&1 | grep -c "warning" && cargo build --release --features arrow 2>&1 | tail -3`
  - **Done when**: All commands pass with zero errors/warnings
  - **Commit**: `chore(forge-filter): pass full local CI` (only if fixes needed)

- [ ] 4.4 Create PR and verify CI
  - **Do**:
    1. Verify current branch is a feature branch: `git branch --show-current`
    2. If on default branch, STOP and alert user
    3. Push branch: `git push -u origin <branch-name>`
    4. Create PR: `gh pr create --title "feat(forge-filter): v0.2.0 — bitmap-cached ordered mode + multi-column + Arrow + NULL" --body "..."`
    5. Wait for CI
  - **Verify**: `gh pr checks --watch` (wait for CI completion)
  - **Done when**: All CI checks green, PR ready for review
  - **Commit**: None (PR creation, not code change)

## Phase 5: PR Lifecycle + Publish

- [ ] 5.1 [VERIFY] CI pipeline passes
  - **Do**: Verify GitHub Actions/CI passes after push
  - **Verify**: `gh pr checks` shows all green
  - **Done when**: CI pipeline passes
  - **Commit**: None

- [ ] 5.2 Address review feedback (if any)
  - **Do**:
    1. Read PR review comments: `gh pr view --comments`
    2. Address each comment with code changes
    3. Push fixes
    4. Re-verify CI
  - **Verify**: `gh pr checks` after push shows all green
  - **Done when**: All review comments resolved
  - **Commit**: `fix(forge-filter): address PR review feedback` (if applicable)

- [ ] 5.3 Verify cargo publish dry-run
  - **Do**:
    1. Run `cargo publish --dry-run --features arrow` in forge-filter directory
    2. Verify no errors (missing files, license issues, etc.)
    3. Check `cargo package --list` includes all needed files
    4. Verify `include` field in Cargo.toml covers: src/**, shaders/**, build.rs, README.md, LICENSE
  - **Files**: `metal-forge-compute/forge-filter/Cargo.toml`
  - **Done when**: `cargo publish --dry-run` succeeds
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo publish --dry-run 2>&1 | tail -10`
  - **Commit**: `chore(forge-filter): fix packaging for crates.io` (only if fixes needed)

- [ ] 5.4 [VERIFY] AC checklist verification
  - **Do**: Programmatically verify each acceptance criterion is satisfied:
    1. AC-1.1: Ordered filter 16M u32 50% sel <=580us — check benchmark output
    2. AC-1.3: Output bit-identical to v0.1 — oracle test passes
    3. AC-1.5: v0.1 API unchanged — grep for all public methods still present
    4. AC-1.6: 10K correctness iterations — test passes
    5. AC-2.1: filter_multi_mask exists — grep public API
    6. AC-2.4: BooleanMask type exists — grep public API
    7. AC-3.1: filter_arrow exists — grep behind feature flag
    8. AC-3.7: arrow feature flag — grep Cargo.toml
    9. AC-4.1: NULL exclusion — test passes
    10. All 47 v0.1 tests still pass — cargo test
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-filter && cargo test 2>&1 | grep "test result" && grep -c "pub fn filter_mask\|pub fn filter_multi_mask\|pub fn gather\|pub fn filter_nullable\|pub struct BooleanMask\|pub enum LogicOp" src/lib.rs`
  - **Done when**: All acceptance criteria confirmed met via automated checks
  - **Commit**: None

## Notes

- **POC shortcuts taken**:
  - Multi-column kernel may use separate-scan-then-AND approach instead of fused kernel (simpler, correct)
  - Arrow integration always copies (no conditional zero-copy path)
  - Polars expression plugin (US-5) deferred to separate spec — not in forge-filter v0.2.0 scope
  - Multi-column capped at 4 (not 8 from requirements) — covers 99% of queries

- **Production TODOs**:
  - Fused multi-column kernel (single dispatch evaluating all columns) for maximum performance
  - MTLBinaryArchive PSO caching for persistent compilation cache
  - Performance regression CI gate (hard block at >=9.5x)
  - Decoupled Fallback single-pass pipeline if bitmap caching insufficient
  - polars-gpu-filter crate (US-5) as separate follow-on spec

- **Key risk**: Bitmap caching may plateau at 8-9x instead of 10x. If so, need to investigate SLC residency, tune tile sizes, or explore Decoupled Fallback approach.

- **Build note**: `rm -rf target/release/build/forge-filter-*` may be needed after changing .metal files if build.rs caching misses the change.
