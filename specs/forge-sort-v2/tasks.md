# Tasks: forge-sort v2 Multi-Type GPU Radix Sort

Total tasks: 35

## Phase 1: Make It Work (POC) — i32/f32 Sorting End-to-End

Focus: Get sort_i32 and sort_f32 working with GPU-side transforms. Skip argsort/KV/64-bit. Accept monolithic dispatch. Prove transform correctness.

- [x] 1.1 Extend PsoCache with function constant support
  - **Do**:
    1. Add `FnConstant` enum to `forge-primitives/src/pso_cache.rs`: `Bool(bool)`, `U32(u32)`
    2. Add `get_or_create_specialized()` method that takes `&[(usize, FnConstant)]`
    3. Build cache key as `"fn_name:idx=val:idx=val"` string
    4. Build `MTLFunctionConstantValues` using `setConstantValue_type_atIndex`
    5. Use `newFunctionWithName_constantValues_error` for function creation
    6. Use `MTLComputePipelineDescriptor` with `maxTotalThreadsPerThreadgroup=256` + `threadGroupSizeIsMultipleOfThreadExecutionWidth=true` (match existing `compile_pso`)
    7. Add imports: `MTLFunctionConstantValues`, `MTLDataType`, `NonNull`
    8. Add unit tests for cache key generation
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-primitives/src/pso_cache.rs` (modify, +70 lines)
  - **Done when**: `get_or_create_specialized()` compiles. Unit tests for key format pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-primitives --release`
  - **Commit**: `feat(forge-primitives): add function constant support to PsoCache`
  - _Requirements: FR-9_
  - _Design: PsoCache Extension_

- [ ] 1.2 Add SortKey sealed trait and make SortBuffer generic
  - **Do**:
    1. Add `mod private` with `Sealed` trait + impls for u32/i32/f32/u64/i64/f64
    2. Add `pub trait SortKey: private::Sealed + Copy + 'static` with consts: `KEY_SIZE`, `NEEDS_TRANSFORM`, `IS_64BIT`, `TRANSFORM_MODE_FORWARD`, `TRANSFORM_MODE_INVERSE`
    3. Impl `SortKey` for all 6 types per design table
    4. Convert `SortBuffer` to `SortBuffer<T: SortKey>` with `PhantomData<T>`
    5. Make `as_slice() -> &[T]`, `as_mut_slice() -> &mut [T]` generic (using `T::KEY_SIZE`)
    6. Update `copy_from_slice`, `copy_to_slice`, `set_len` for generic T
    7. Add `SortError::LengthMismatch { keys: usize, values: usize }` variant
    8. Update `alloc_sort_buffer` to `alloc_sort_buffer::<T>(capacity)` returning `SortBuffer<T>`
    9. Update `sort_buffer` signature to `sort_buffer(&mut self, buf: &SortBuffer<u32>)` (u32-specific for now)
    10. Update `sort_u32` to work with `SortBuffer<u32>` internally
    11. Update inline unit tests (size checks, error display)
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/src/lib.rs` (modify, +120 lines)
  - **Done when**: `SortBuffer<u32>` compiles. Existing sort_u32 and sort_buffer still work. `SortBuffer<String>` is compile error.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release`
  - **Commit**: `feat(forge-sort): add SortKey trait and generic SortBuffer<T>`
  - _Requirements: FR-14, AC-6.1, AC-6.2, AC-6.3, AC-6.4, AC-6.5, AC-6.6_
  - _Design: SortKey Trait, SortBuffer\<T\>_

- [ ] 1.3 Update existing tests for SortBuffer\<u32\> annotation
  - **Do**:
    1. Update `tests/correctness.rs`: change `alloc_sort_buffer(n)` calls to `alloc_sort_buffer::<u32>(n)` if signature changed
    2. If `sort_buffer` now requires `&SortBuffer<u32>`, no test changes needed (type inference)
    3. Run all 25 existing tests to verify no regressions
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/correctness.rs` (modify, ~5 lines)
  - **Done when**: All 25 existing tests pass unchanged or with minimal annotation changes.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- --test-threads=1 2>&1 | tail -5`
  - **Commit**: `fix(forge-sort): update tests for SortBuffer<u32> annotation`
  - _Requirements: AC-7.1, AC-7.2, AC-7.4_
  - _Design: Backward-Compatible u32 Sorting_

- [ ] V1 [VERIFY] Quality checkpoint: cargo check + existing tests
  - **Do**: Run type check and all existing tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check --workspace && cargo test -p forge-sort --release -- --test-threads=1 2>&1 | tail -5`
  - **Done when**: Zero compiler errors, all 25 tests pass
  - **Commit**: `chore(forge-sort): pass quality checkpoint` (only if fixes needed)

- [ ] 1.4 Add sort_transform_32 kernel + function constant declarations to sort.metal
  - **Do**:
    1. Add function constant declarations at top of sort.metal (after struct defs):
       ```metal
       constant bool HAS_VALUES [[function_constant(0)]];
       constant bool IS_64BIT   [[function_constant(1)]];
       constant uint TRANSFORM_MODE [[function_constant(2)]];
       constant bool has_values = is_function_constant_defined(HAS_VALUES) ? HAS_VALUES : false;
       constant bool is_64bit   = is_function_constant_defined(IS_64BIT)   ? IS_64BIT   : false;
       constant uint transform_mode = is_function_constant_defined(TRANSFORM_MODE) ? TRANSFORM_MODE : 0u;
       ```
    2. Add `sort_transform_32` kernel (~20 lines):
       - mode 0: XOR 0x80000000 (i32, self-inverse)
       - mode 1: FloatFlip (positive: as-is, negative: flip all bits)
       - mode 2: IFloatFlip (inverse of FloatFlip)
       - Takes `device uint* data`, `constant uint& count` as params
       - Simple 1D dispatch: each thread handles 1 element
    3. Keep existing 4 kernels unchanged (function constants have defaults)
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/shaders/sort.metal` (modify, +35 lines)
  - **Done when**: Shader compiles. Existing sort_u32 works (function constants default to false/0).
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- test_sort_1m --test-threads=1`
  - **Commit**: `feat(forge-sort): add sort_transform_32 kernel and function constants`
  - _Requirements: FR-1, FR-3_
  - _Design: Function Constants, sort_transform_32_

- [ ] 1.5 Implement sort_i32 and sort_i32_buffer methods
  - **Do**:
    1. Pre-compile transform PSOs at `GpuSorter::new()`:
       - `sort_transform_32` with TRANSFORM_MODE=0 (i32 XOR)
       - `sort_transform_32` with TRANSFORM_MODE=1 (FloatFlip forward)
       - `sort_transform_32` with TRANSFORM_MODE=2 (IFloatFlip inverse)
    2. Add `sort_i32(&mut self, data: &mut [i32]) -> Result<(), SortError>`:
       - Early return for n <= 1
       - `ensure_buffers(n)` (reuse u32 path -- same 4-byte elements)
       - memcpy `data` as bytes into `buf_a`
       - Dispatch `sort_transform_32` mode=0 on `buf_a` (forward)
       - Dispatch existing 4-dispatch sort pipeline (reuse `dispatch_sort`)
       - Dispatch `sort_transform_32` mode=0 on `buf_a` (inverse = same for XOR)
       - memcpy result back
    3. Add `sort_i32_buffer(&mut self, buf: &SortBuffer<i32>) -> Result<(), SortError>`:
       - Same as sort_buffer but with transform dispatches wrapping the sort
    4. Helper method: `dispatch_transform_32(&self, encoder, buf, n, mode)` to encode one transform dispatch
    5. Refactor `dispatch_sort` to accept an encoder (instead of creating its own cmd buffer) so transforms + sort share one command buffer. Or: create a new `dispatch_sort_with_transforms` that wraps the existing pattern.
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/src/lib.rs` (modify, +100 lines)
  - **Done when**: `sort_i32` correctly sorts `[i32::MIN, 0, -1, 1, i32::MAX]` ascending.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- test_sort_i32 --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(forge-sort): implement sort_i32 and sort_i32_buffer`
  - _Requirements: AC-1.1, AC-1.2, AC-1.3, AC-1.4_
  - _Design: 32-bit Key-Only Sort, dispatch_sort_pipeline_

- [ ] 1.6 Implement sort_f32 and sort_f32_buffer methods
  - **Do**:
    1. Add `sort_f32(&mut self, data: &mut [f32]) -> Result<(), SortError>`:
       - Same pattern as sort_i32 but mode=1 (FloatFlip) forward, mode=2 (IFloatFlip) inverse
    2. Add `sort_f32_buffer(&mut self, buf: &SortBuffer<f32>) -> Result<(), SortError>`:
       - Zero-copy with transform dispatches
    3. FloatFlip correctness: for negative floats, flip ALL bits. For positive, flip only sign bit.
       This maps float ordering to unsigned integer ordering.
    4. IFloatFlip: reverse of FloatFlip
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/src/lib.rs` (modify, +60 lines)
  - **Done when**: `sort_f32` sorts `[-NaN, -Inf, -1.0, -0.0, 0.0, 1.0, Inf, NaN]` in total_cmp order.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- test_sort_f32 --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(forge-sort): implement sort_f32 and sort_f32_buffer`
  - _Requirements: AC-2.1, AC-2.2, AC-2.5, AC-2.6_
  - _Design: 32-bit Key-Only Sort_

- [ ] 1.7 Add i32/f32 correctness tests
  - **Do**:
    1. Create `tests/correctness_i32.rs` with ~15 tests:
       - `test_sort_i32_boundaries`: [MIN, MAX, 0, -1, 1] sorted correctly
       - `test_sort_i32_all_negative`: all negative values sorted ascending
       - `test_sort_i32_mixed_1m`: 1M random i32, oracle `[i32]::sort()`
       - `test_sort_i32_16m`: 16M random i32
       - `test_sort_i32_presorted`, `test_sort_i32_reverse`
       - `test_sort_i32_empty`, `test_sort_i32_single`
       - `test_sort_i32_buffer_basic`, `test_sort_i32_buffer_16m`
       - `test_sort_i32_xor_self_inverse`: apply transform twice = identity
       - `test_sort_i32_non_tile_aligned`
    2. Create `tests/correctness_f32.rs` with ~18 tests:
       - `test_sort_f32_ieee754_special`: all 19 IEEE 754 special values
       - `test_sort_f32_total_cmp_order`: [-NaN < -Inf < -1.0 < -0.0 < 0.0 < 1.0 < Inf < NaN]
       - `test_sort_f32_neg_zero_before_pos_zero`: distinct positions
       - `test_sort_f32_nan_variants`: different NaN payloads sort to distinct positions
       - `test_sort_f32_denormals`: subnormals between 0 and smallest normal
       - `test_sort_f32_bit_exact`: compare via `to_bits()` against `sort_by(f32::total_cmp)`
       - `test_sort_f32_random_1m`, `test_sort_f32_random_16m`
       - `test_sort_f32_all_nan`, `test_sort_f32_all_inf`
       - `test_sort_f32_buffer_basic`, `test_sort_f32_buffer_16m`
       - `test_sort_f32_floatflip_roundtrip`: forward+inverse = identity for sampled patterns
       - `test_sort_f32_empty`, `test_sort_f32_single`
    3. Use `rand_chacha` with fixed seed for deterministic random data. Add `rand_chacha = "0.3"` to dev-deps.
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/correctness_i32.rs` (create, ~120 lines)
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/correctness_f32.rs` (create, ~160 lines)
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/Cargo.toml` (modify: add rand_chacha dev-dep)
  - **Done when**: All i32 and f32 tests pass. Bit-exact match with Rust oracle.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- correctness_i32 correctness_f32 --test-threads=1 2>&1 | tail -10`
  - **Commit**: `test(forge-sort): add i32/f32 correctness tests`
  - _Requirements: AC-1.1, AC-1.2, AC-2.1, AC-2.2, AC-2.3, AC-2.4_
  - _Design: Test Strategy_

- [ ] V2 [VERIFY] Quality checkpoint: full test suite
  - **Do**: Run full workspace check + all forge-sort tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check --workspace && cargo test -p forge-sort --release -- --test-threads=1 2>&1 | tail -10`
  - **Done when**: Zero errors, all tests pass (25 existing + ~33 new i32/f32)
  - **Commit**: `chore(forge-sort): pass quality checkpoint` (only if fixes needed)

- [ ] 1.8 POC Checkpoint: verify i32/f32 sort end-to-end
  - **Do**:
    1. Run the full i32 + f32 test suites
    2. Run existing u32 tests to confirm no regression
    3. Verify sort_i32 handles i32::MIN/MAX boundary correctly
    4. Verify sort_f32 handles -0.0 < +0.0 and NaN ordering
    5. Verify sort_u32 performance not regressed (existing perf sanity test)
  - **Done when**: All tests pass. sort_i32 and sort_f32 produce correct results.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- --test-threads=1 2>&1 | grep -E "test result|FAILED"`
  - **Commit**: `feat(forge-sort): complete Phase 1 POC - i32/f32 sorting`
  - _Requirements: US-1, US-2, US-7_

## Phase 2: Argsort + Key-Value Pairs

Focus: Add value tracking through sort pipeline. sort_init_indices, HAS_VALUES branches in scatter+inner, argsort, sort_pairs.

- [ ] 2.1 Add InnerParams struct and refactor sort_inner_fused binding
  - **Do**:
    1. Add `InnerParams` struct to sort.metal: `{start_shift: uint, pass_count: uint, batch_start: uint}`
    2. Change `sort_inner_fused` buffer(3) from `constant uint& batch_start` to `constant InnerParams& inner_params`
    3. Update kernel to read `inner_params.batch_start`, `inner_params.start_shift`, `inner_params.pass_count`
    4. For now: pass_count always 3, start_shift always 0 (same as current behavior)
    5. Replace `pass * 8u` with `(inner_params.start_shift + pass) * 8u`
    6. Replace `pass < 3u` with `pass < inner_params.pass_count`
    7. Add `#[repr(C)] InnerParams` struct in lib.rs (Rust side)
    8. Update `dispatch_sort()` to send InnerParams struct via setBytes instead of bare u32
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/shaders/sort.metal` (modify, ~10 lines)
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/src/lib.rs` (modify, ~20 lines)
  - **Done when**: All existing tests pass with InnerParams. No behavioral change.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- --test-threads=1 2>&1 | tail -5`
  - **Commit**: `refactor(forge-sort): replace batch_start with InnerParams struct`
  - _Requirements: FR-8_
  - _Design: InnerParams Struct_

- [ ] 2.2 Add HAS_VALUES branches to sort_msd_atomic_scatter
  - **Do**:
    1. Add value buffer bindings to scatter kernel: `device const uint* src_vals [[buffer(4)]]`, `device uint* dst_vals [[buffer(5)]]`
    2. Guard with `if (has_values)` for all value operations
    3. In Phase 1 (load): `mv_vals[e] = mv[e] ? src_vals[idx] : 0u;`
    4. In Phase 4 (scatter): `if (has_values) { dst_vals[gp] = mv_vals[e]; }`
    5. Values loaded into register array `uint mv_vals[SORT_ELEMS]`
    6. Existing kernels (HAS_VALUES=false default) unaffected -- dead code elimination
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/shaders/sort.metal` (modify, +25 lines)
  - **Done when**: Shader compiles. Existing u32 tests pass (HAS_VALUES defaults false).
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- test_sort_1m --test-threads=1`
  - **Commit**: `feat(forge-sort): add HAS_VALUES value tracking to MSD scatter`
  - _Requirements: FR-4_
  - _Design: sort_msd_atomic_scatter HAS_VALUES_

- [ ] 2.3 Add HAS_VALUES branches to sort_inner_fused
  - **Do**:
    1. Add value buffer bindings: `device uint* vals_a [[buffer(4)]]`, `device uint* vals_b [[buffer(5)]]`
    2. Guard all value ops with `if (has_values)`
    3. In tile load: `uint vals[SORT_ELEMS]; vals[e] = src_vals[idx];`
    4. In scatter: `dst_vals[dst_idx] = vals[e];`
    5. src_vals/dst_vals alternate with src/dst each pass (same ping-pong as keys)
    6. Values follow same buffer alternation: pass 0: vals_b->vals_a, pass 1: vals_a->vals_b, pass 2: vals_b->vals_a
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/shaders/sort.metal` (modify, +35 lines)
  - **Done when**: Shader compiles. Existing u32 tests pass (HAS_VALUES defaults false).
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- test_sort_16m --test-threads=1`
  - **Commit**: `feat(forge-sort): add HAS_VALUES value tracking to inner fused sort`
  - _Requirements: FR-5_
  - _Design: sort_inner_fused HAS_VALUES_

- [ ] V3 [VERIFY] Quality checkpoint: shader changes safe
  - **Do**: Full test suite to verify HAS_VALUES branches don't break default path
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check --workspace && cargo test -p forge-sort --release -- --test-threads=1 2>&1 | tail -10`
  - **Done when**: All tests pass (25 existing + i32/f32 tests from Phase 1)
  - **Commit**: `chore(forge-sort): pass quality checkpoint` (only if fixes needed)

- [ ] 2.4 Add sort_init_indices and sort_gather_values kernels
  - **Do**:
    1. Add `sort_init_indices` kernel to sort.metal (~6 lines):
       ```metal
       kernel void sort_init_indices(
           device uint* indices [[buffer(0)]],
           constant uint& count [[buffer(1)]],
           uint gid [[thread_position_in_grid]])
       { if (gid < count) indices[gid] = gid; }
       ```
    2. Add `sort_gather_values` kernel to sort.metal (~8 lines):
       ```metal
       kernel void sort_gather_values(
           device const uint* sorted_indices [[buffer(0)]],
           device const uint* original_vals  [[buffer(1)]],
           device uint*       gathered_vals   [[buffer(2)]],
           constant uint&     count           [[buffer(3)]],
           uint gid [[thread_position_in_grid]])
       { if (gid < count) gathered_vals[gid] = original_vals[sorted_indices[gid]]; }
       ```
    3. Pre-compile PSOs for these 2 new kernels at GpuSorter::new()
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/shaders/sort.metal` (modify, +16 lines)
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/src/lib.rs` (modify, +5 lines for PSO init)
  - **Done when**: New kernels compile. Existing tests pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- test_sort_1m --test-threads=1`
  - **Commit**: `feat(forge-sort): add sort_init_indices and sort_gather_values kernels`
  - _Requirements: FR-6, FR-7_
  - _Design: New Kernels_

- [ ] 2.5 Implement argsort methods and lazy value buffer allocation
  - **Do**:
    1. Add `buf_vals_a`, `buf_vals_b` as `Option<Retained<...>>` fields to GpuSorter
    2. Add `vals_buf_capacity: usize` field
    3. Add `ensure_buffers_with_values(&mut self, n: usize)` method
    4. Pre-compile HAS_VALUES=true PSOs for scatter and inner fused at init
    5. Create `dispatch_sort_pipeline()` method that handles:
       - Optional pre-sort transform dispatch
       - Optional sort_init_indices dispatch
       - sort_msd_histogram (key-only, no change)
       - sort_msd_prep (no change)
       - sort_msd_atomic_scatter (with or without values)
       - sort_inner_fused (with or without values)
       - Optional post-sort inverse transform dispatch
       - All in single command buffer / single encoder
    6. Implement `argsort_u32(&mut self, data: &[u32]) -> Result<Vec<u32>, SortError>`:
       - ensure_buffers_with_values(n)
       - memcpy data to buf_a
       - dispatch sort_init_indices to buf_vals_a
       - dispatch sort pipeline with HAS_VALUES=true PSOs
       - memcpy buf_vals_a (sorted indices) to Vec<u32>
       - Input data NOT modified
    7. Implement `argsort_i32`, `argsort_f32` (same pattern + transforms)
    8. Handle edge cases: empty -> Ok(vec![]), single -> Ok(vec![0])
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/src/lib.rs` (modify, +200 lines)
  - **Done when**: `argsort_u32([3,1,2])` returns `[1,2,0]`. Input unchanged.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- argsort --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(forge-sort): implement argsort_u32, argsort_i32, argsort_f32`
  - _Requirements: AC-3.1, AC-3.2, AC-3.3, AC-3.4, AC-3.5, FR-12_
  - _Design: dispatch_sort_pipeline, Buffer Management_

- [ ] 2.6 Implement sort_pairs methods
  - **Do**:
    1. Add `buf_orig_vals` as `Option<Retained<...>>` field to GpuSorter
    2. Add `ensure_buffers_with_values_and_orig(&mut self, n: usize)` method
    3. Implement `sort_pairs_u32(&mut self, keys: &mut [u32], values: &mut [u32]) -> Result<(), SortError>`:
       - Check `keys.len() == values.len()`, return `SortError::LengthMismatch` if not
       - memcpy keys to buf_a, values to buf_orig_vals
       - dispatch sort_init_indices to buf_vals_a (indices)
       - dispatch sort pipeline with HAS_VALUES=true (sorts keys, carries indices)
       - dispatch sort_gather_values: rearrange buf_orig_vals by sorted indices -> buf_vals_b
       - memcpy buf_a -> keys (sorted), buf_vals_b -> values (rearranged)
    4. Implement `sort_pairs_i32`, `sort_pairs_f32` (same + transforms)
    5. All use Strategy B (argsort + gather)
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/src/lib.rs` (modify, +120 lines)
  - **Done when**: `sort_pairs_u32([3,1,2], [30,10,20])` -> keys=[1,2,3], values=[10,20,30].
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- sort_pairs --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(forge-sort): implement sort_pairs_u32, sort_pairs_i32, sort_pairs_f32`
  - _Requirements: AC-4.1, AC-4.2, AC-4.3, AC-4.4, AC-4.6, FR-16, FR-18_
  - _Design: sort_pairs Strategy B_

- [ ] 2.7 Add argsort and sort_pairs correctness tests
  - **Do**:
    1. Create `tests/correctness_argsort.rs` with ~15 tests:
       - `test_argsort_u32_basic`: small known input, verify permutation
       - `test_argsort_u32_permutation_valid`: every index 0..n appears exactly once
       - `test_argsort_u32_sorted_order`: data[indices[i]] <= data[indices[i+1]]
       - `test_argsort_u32_input_unmodified`: original slice unchanged
       - `test_argsort_u32_stable`: equal keys preserve relative index order
       - `test_argsort_u32_1m`, `test_argsort_u32_16m`
       - `test_argsort_i32_basic`, `test_argsort_i32_boundaries`
       - `test_argsort_f32_basic`, `test_argsort_f32_nan`
       - `test_argsort_empty`, `test_argsort_single`
       - `test_argsort_u32_all_same`: equal keys, indices should be 0..n
    2. Create `tests/correctness_sort_pairs.rs` with ~15 tests:
       - `test_sort_pairs_u32_basic`: known input, verify key+value pairing
       - `test_sort_pairs_u32_multiset_preserved`: same set of (k,v) pairs before/after
       - `test_sort_pairs_u32_stable`: equal keys preserve value order
       - `test_sort_pairs_i32_basic`, `test_sort_pairs_f32_basic`
       - `test_sort_pairs_f32_nan_with_values`
       - `test_sort_pairs_length_mismatch`: verify SortError::LengthMismatch
       - `test_sort_pairs_u32_1m`, `test_sort_pairs_f32_16m`
       - `test_sort_pairs_empty`, `test_sort_pairs_single`
       - `test_sort_pairs_u32_all_same`
    3. Create `tests/common/mod.rs` with shared helpers:
       - `verify_permutation(indices: &[u32], n: usize) -> bool`
       - `verify_sorted_by_indices<T: Ord>(data: &[T], indices: &[u32]) -> bool`
       - `verify_pairs_preserved<K, V>(orig_keys, orig_vals, sorted_keys, sorted_vals) -> bool`
       - `seeded_rng(seed: u64) -> ChaCha8Rng`
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/correctness_argsort.rs` (create, ~120 lines)
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/correctness_sort_pairs.rs` (create, ~120 lines)
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/common/mod.rs` (create, ~60 lines)
  - **Done when**: All argsort and sort_pairs tests pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- correctness_argsort correctness_sort_pairs --test-threads=1 2>&1 | tail -10`
  - **Commit**: `test(forge-sort): add argsort and sort_pairs correctness tests`
  - _Requirements: AC-3.1 through AC-3.5, AC-4.1 through AC-4.5_
  - _Design: Test Strategy_

- [ ] V4 [VERIFY] Quality checkpoint: full test suite after Phase 2
  - **Do**: Run all tests to verify argsort/KV + existing paths all work
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check --workspace && cargo test -p forge-sort --release -- --test-threads=1 2>&1 | grep -E "test result|FAILED"`
  - **Done when**: All tests pass (25 existing + i32/f32 + argsort + sort_pairs)
  - **Commit**: `chore(forge-sort): pass quality checkpoint` (only if fixes needed)

## Phase 3: 64-bit Pipeline

Focus: IS_64BIT branches in sort kernels, 8-pass sort via 3 inner fused dispatches, u64/i64/f64.

- [ ] 3.1 Add IS_64BIT branches to sort kernels
  - **Do**:
    1. In `sort_msd_histogram`: add `if (is_64bit)` branch:
       - Use `device const ulong*` cast for key reads
       - Use `ELEMS_PER_THREAD` (8 for 64-bit) instead of hardcoded SORT_ELEMS
       - Compute `EFFECTIVE_TILE_SIZE` based on `is_64bit`
       - Shift on bits[56:63] for 64-bit
    2. In `sort_msd_atomic_scatter`: add `if (is_64bit)` branch:
       - ulong keys in registers, 8 elements/thread
       - Scatter to `device ulong* dst` (reinterpret_cast)
       - HAS_VALUES branches still use uint for values
    3. In `sort_inner_fused`: add `if (is_64bit)` branch:
       - ulong keys, 8 elements/thread, 2048-element tiles
       - Use `inner_params.start_shift` and `inner_params.pass_count` for variable passes
       - Same histogram + scatter logic but with ulong keys
       - Values still uint (argsort indices are always u32)
    4. Use conditional compilation: `using key_t = select_type<is_64bit, ulong, uint>::type;`
       Or simpler: `if (is_64bit) { ... ulong path ... } else { ... uint path ... }`
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/shaders/sort.metal` (modify, +80 lines)
  - **Done when**: Shader compiles. Existing 32-bit tests pass (IS_64BIT defaults false).
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- test_sort_1m test_sort_16m --test-threads=1`
  - **Commit**: `feat(forge-sort): add IS_64BIT branches to all sort kernels`
  - _Requirements: FR-3, FR-11_
  - _Design: Kernel Modifications Summary_

- [ ] 3.2 Add sort_transform_64 kernel
  - **Do**:
    1. Add `sort_transform_64` kernel to sort.metal (~15 lines):
       - mode 1 (i64/f64 forward): FloatFlip64 -- if sign bit set, flip all 64 bits; else flip only sign bit
         (Mathematically same as XOR 0x8000000000000000 for i64, full FloatFlip for f64)
       - mode 2 (f64 inverse): IFloatFlip64 -- reverse of FloatFlip64
       - Takes `device ulong* data`, `constant uint& count`
    2. Pre-compile PSOs: sort_transform_64 mode=1, mode=2 (lazy -- part of 64-bit lazy init)
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/shaders/sort.metal` (modify, +20 lines)
  - **Done when**: Shader compiles. Transform roundtrip is identity for sampled i64/f64 values.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- test_sort_1m --test-threads=1`
  - **Commit**: `feat(forge-sort): add sort_transform_64 kernel for i64/f64`
  - _Requirements: FR-2_
  - _Design: sort_transform_64_

- [ ] 3.3 Implement 64-bit dispatch pipeline and sort_u64 method
  - **Do**:
    1. Add lazy 64-bit PSO compilation: on first 64-bit sort call, compile IS_64BIT=true PSOs for histogram, scatter, inner (with and without HAS_VALUES)
    2. 64-bit buffer management: buf_a/buf_b need `n * 8` bytes for 64-bit keys
    3. Add `ensure_buffers_64(&mut self, n: usize)` or modify `ensure_buffers` to accept element size
    4. Extend `dispatch_sort_pipeline()` for 64-bit:
       - MSD histogram IS_64BIT=true (shift=56, bits[56:63])
       - MSD prep (unchanged)
       - MSD scatter IS_64BIT=true
       - 3 inner fused dispatches:
         - InnerParams{start_shift: 32, pass_count: 3, batch_start: 0}  (bytes 4,5,6)
         - InnerParams{start_shift: 8,  pass_count: 3, batch_start: 0}  (bytes 1,2,3)
         - InnerParams{start_shift: 0,  pass_count: 1, batch_start: 0}  (byte 0)
       - Buffer ping-pong: dispatch 1 reads buf_b->buf_a, dispatch 2 reads buf_a->buf_b, dispatch 3 reads buf_b->buf_a
       - Final output in buf_a (same as 32-bit)
    5. Implement `sort_u64(&mut self, data: &mut [u64]) -> Result<(), SortError>`:
       - No transform needed for u64
       - Use 64-bit pipeline dispatch
       - memcpy 8-byte elements
    6. Implement `sort_u64_buffer(&mut self, buf: &SortBuffer<u64>) -> Result<(), SortError>`
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/src/lib.rs` (modify, +150 lines)
  - **Done when**: `sort_u64` correctly sorts `[u64::MAX, 0, 1, u64::MAX-1]`.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- test_sort_u64 --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(forge-sort): implement 64-bit sort pipeline and sort_u64`
  - _Requirements: AC-5.1, AC-5.4, AC-5.5, FR-10, FR-11_
  - _Design: 64-bit Sort Pipeline, PSO Compilation Strategy_

- [ ] V5 [VERIFY] Quality checkpoint: 64-bit pipeline
  - **Do**: Verify 64-bit pipeline + all existing 32-bit tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check --workspace && cargo test -p forge-sort --release -- --test-threads=1 2>&1 | grep -E "test result|FAILED"`
  - **Done when**: All tests pass
  - **Commit**: `chore(forge-sort): pass quality checkpoint` (only if fixes needed)

- [ ] 3.4 Implement sort_i64, sort_f64 and buffer variants
  - **Do**:
    1. Implement `sort_i64(&mut self, data: &mut [i64])`:
       - Lazy compile sort_transform_64 mode=1 PSO
       - Transform forward (mode=1 is self-inverse for i64)
       - 64-bit sort pipeline
       - Transform inverse (mode=1 again -- self-inverse)
    2. Implement `sort_f64(&mut self, data: &mut [f64])`:
       - Transform forward mode=1 (FloatFlip64)
       - 64-bit sort pipeline
       - Transform inverse mode=2 (IFloatFlip64)
    3. Buffer variants: `sort_i64_buffer`, `sort_f64_buffer`
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/src/lib.rs` (modify, +80 lines)
  - **Done when**: `sort_i64([MIN, -1, 0, 1, MAX])` and `sort_f64([-NaN, -Inf, -0.0, 0.0, Inf, NaN])` correct.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- "test_sort_i64|test_sort_f64" --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(forge-sort): implement sort_i64, sort_f64 and buffer variants`
  - _Requirements: AC-5.2, AC-5.3, AC-5.5_
  - _Design: 64-bit Sort Pipeline_

- [ ] 3.5 Implement 64-bit argsort and sort_pairs
  - **Do**:
    1. `argsort_u64`, `argsort_i64`, `argsort_f64`:
       - 64-bit sort pipeline with HAS_VALUES=true + IS_64BIT=true PSOs
       - Values are u32 indices (always 4 bytes)
       - Lazy compile 64-bit KV PSOs on first use
    2. `sort_pairs_u64`, `sort_pairs_i64`:
       - Strategy B: argsort + gather
       - No `sort_pairs_f64` (out of scope)
    3. 64-bit KV buffer management: key buffers 8 bytes, value buffers 4 bytes
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/src/lib.rs` (modify, +100 lines)
  - **Done when**: `argsort_u64([30,10,20])` returns `[1,2,0]`. `sort_pairs_i64` preserves pairs.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- "argsort_u64|argsort_i64|argsort_f64|sort_pairs_u64|sort_pairs_i64" --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(forge-sort): implement 64-bit argsort and sort_pairs`
  - _Requirements: AC-5.6, AC-5.7_
  - _Design: 64-bit Argsort + KV_

- [ ] 3.6 Add 64-bit correctness tests
  - **Do**:
    1. Create `tests/correctness_u64.rs` (~12 tests):
       - `test_sort_u64_boundaries`: [0, 1, MAX-1, MAX]
       - `test_sort_u64_byte_boundary_values`: values that isolate each of 8 bytes
       - `test_sort_u64_random_1m`, `test_sort_u64_random_16m`
       - `test_sort_u64_buffer_basic`, `test_sort_u64_buffer_16m`
       - `test_sort_u64_presorted`, `test_sort_u64_reverse`
       - `test_sort_u64_empty`, `test_sort_u64_single`
       - `test_sort_u64_non_tile_aligned`
       - `test_sort_u64_powers_of_two`
    2. Create `tests/correctness_i64.rs` (~12 tests):
       - `test_sort_i64_boundaries`: [MIN, -1, 0, 1, MAX]
       - `test_sort_i64_sign_boundary`: values around 0
       - `test_sort_i64_all_negative`, `test_sort_i64_mixed_1m`
       - `test_sort_i64_xor_self_inverse`: transform twice = identity
       - `test_sort_i64_buffer_basic`
       - `test_sort_i64_random_16m`
       - `test_sort_i64_empty`, `test_sort_i64_single`
       - `test_argsort_i64_basic`, `test_sort_pairs_i64_basic`
    3. Create `tests/correctness_f64.rs` (~15 tests):
       - `test_sort_f64_total_cmp_order`: full NaN/Inf/zero ordering
       - `test_sort_f64_ieee754_special`: 19 IEEE 754 f64 special values
       - `test_sort_f64_neg_zero_before_pos_zero`
       - `test_sort_f64_nan_variants`: different NaN payloads
       - `test_sort_f64_denormals`
       - `test_sort_f64_bit_exact`: compare via `to_bits()` against `sort_by(f64::total_cmp)`
       - `test_sort_f64_random_1m`, `test_sort_f64_random_16m`
       - `test_sort_f64_buffer_basic`, `test_sort_f64_buffer_16m`
       - `test_sort_f64_floatflip_roundtrip`
       - `test_sort_f64_empty`, `test_sort_f64_single`
       - `test_argsort_f64_basic`, `test_sort_f64_per_byte_isolation`
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/correctness_u64.rs` (create, ~100 lines)
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/correctness_i64.rs` (create, ~100 lines)
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/correctness_f64.rs` (create, ~130 lines)
  - **Done when**: All 64-bit tests pass. Oracle match for all types.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- "correctness_u64|correctness_i64|correctness_f64" --test-threads=1 2>&1 | tail -10`
  - **Commit**: `test(forge-sort): add u64/i64/f64 correctness tests`
  - _Requirements: AC-5.1 through AC-5.9_
  - _Design: Test Strategy_

- [ ] V6 [VERIFY] Quality checkpoint: complete 64-bit pipeline
  - **Do**: Full test suite verification
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check --workspace && cargo test -p forge-sort --release -- --test-threads=1 2>&1 | grep -E "test result|FAILED"`
  - **Done when**: All tests pass across all types
  - **Commit**: `chore(forge-sort): pass quality checkpoint` (only if fixes needed)

## Phase 4: Testing + Regression + Performance

Focus: Comprehensive test coverage, regression tests, performance validation.

- [ ] 4.1 Add regression tests and cross-type interleaving tests
  - **Do**:
    1. Create `tests/regression.rs` (~7 tests):
       - `test_v1_compat_sort_u32_unchanged`: sort_u32 behavior identical to v1
       - `test_v1_compat_sort_buffer_u32`: sort_buffer with SortBuffer<u32>
       - `test_cross_type_interleave`: sort_u32 -> sort_f32 -> sort_i32 -> sort_u32 on same GpuSorter, all correct
       - `test_cross_type_interleave_with_argsort`: mix sort + argsort calls
       - `test_cross_type_interleave_64bit`: mix 32-bit and 64-bit sorts
       - `test_pso_cache_isolation`: different types use different PSOs
       - `test_sort_u32_perf_no_regression`: 16M sort < 50ms (existing bound)
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/regression.rs` (create, ~80 lines)
  - **Done when**: All regression tests pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release -- regression --test-threads=1 2>&1 | tail -10`
  - **Commit**: `test(forge-sort): add regression and cross-type interleaving tests`
  - _Requirements: AC-7.1 through AC-7.5_
  - _Design: Regression Tests_

- [ ] 4.2 Add performance tests (feature-gated)
  - **Do**:
    1. Add `perf-test` feature to Cargo.toml: `[features] perf-test = []`
    2. Create `tests/performance.rs` with `#[cfg(feature = "perf-test")]` on all tests
    3. Test pattern: seeded RNG, warmup run, 10 measured runs, take p50
    4. Performance thresholds per requirements (30% headroom):
       - sort_u32 memcpy >= 2000 Mk/s, zero-copy >= 3600 Mk/s
       - sort_i32 memcpy >= 1900 Mk/s
       - sort_f32 memcpy >= 1750 Mk/s, zero-copy >= 3100 Mk/s
       - argsort_u32 >= 1400 Mk/s
       - sort_pairs_f32 >= 1000 Mk/s
       - sort_u64 memcpy >= 600 Mk/s, zero-copy >= 850 Mk/s
       - sort_i64 memcpy >= 550 Mk/s
       - sort_f64 memcpy >= 550 Mk/s
    5. All tests use 16M elements, seeded via rand_chacha
    6. Print actual throughput for debugging
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/Cargo.toml` (modify: add feature)
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/tests/performance.rs` (create, ~120 lines)
  - **Done when**: Performance tests pass with `--features perf-test`.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort --release --features perf-test -- performance --test-threads=1 2>&1 | tail -20`
  - **Commit**: `test(forge-sort): add feature-gated performance tests`
  - _Requirements: NFR-1 through NFR-11_
  - _Design: Performance Targets_

- [ ] 4.3 Bump version to 0.2.0
  - **Do**:
    1. Update `forge-sort/Cargo.toml` version from "0.1.0" to "0.2.0"
    2. Verify all tests still pass (version change is just metadata)
  - **Files**:
    - `/Users/patrickkavanagh/gpu_kernel/metal-forge-compute/forge-sort/Cargo.toml` (modify: version)
  - **Done when**: Cargo.toml shows version = "0.2.0".
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && grep 'version = "0.2.0"' forge-sort/Cargo.toml && cargo test -p forge-sort --release -- test_sort_1m --test-threads=1`
  - **Commit**: `chore(forge-sort): bump version to 0.2.0`
  - _Requirements: FR-17_

- [ ] V7 [VERIFY] Quality checkpoint: all tests pass
  - **Do**: Complete test suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo check --workspace && cargo test -p forge-sort --release -- --test-threads=1 2>&1 | grep -E "test result|FAILED"`
  - **Done when**: All tests pass
  - **Commit**: `chore(forge-sort): pass quality checkpoint` (only if fixes needed)

## Phase 5: Quality Gates + PR

- [ ] 5.1 [VERIFY] Full local CI: check + test + build
  - **Do**: Run complete local CI suite
  - **Verify**: All commands must pass:
    ```
    cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && \
    cargo check --workspace && \
    cargo test -p forge-primitives --release && \
    cargo test -p forge-sort --release -- --test-threads=1 && \
    cargo test -p forge-sort --release --features perf-test -- performance --test-threads=1 && \
    cargo build --workspace --release
    ```
  - **Done when**: All commands pass with no errors
  - **Commit**: `fix(forge-sort): address lint/type issues` (if fixes needed)

- [ ] 5.2 Create PR and verify CI
  - **Do**:
    1. Verify current branch is a feature branch: `git branch --show-current`
    2. If on default branch, STOP and alert user
    3. Push branch: `git push -u origin <branch-name>`
    4. Create PR with summary:
       - Title: "feat(forge-sort): v2 multi-type GPU radix sort"
       - Body: list new types (i32/f32/u64/i64/f64), argsort, sort_pairs, test count, perf numbers
    5. If gh CLI unavailable, provide URL for manual PR creation
  - **Verify**: `gh pr checks --watch` or `gh pr checks` (poll)
  - **Done when**: PR created, CI checks green
  - **If CI fails**: Read failures, fix locally, push fixes, re-verify

- [ ] 5.3 [VERIFY] CI pipeline passes
  - **Do**: Verify GitHub Actions/CI passes after push
  - **Verify**: `gh pr checks` shows all green
  - **Done when**: CI pipeline passes
  - **Commit**: None

- [ ] 5.4 [VERIFY] AC checklist
  - **Do**: Programmatically verify each acceptance criterion:
    1. AC-1.1: Run `cargo test -p forge-sort --release -- test_sort_i32_boundaries`
    2. AC-1.2: Run `cargo test -p forge-sort --release -- test_sort_i32_mixed`
    3. AC-2.1: Run `cargo test -p forge-sort --release -- test_sort_f32_total_cmp_order`
    4. AC-2.2: Run `cargo test -p forge-sort --release -- test_sort_f32_bit_exact`
    5. AC-2.3: Run `cargo test -p forge-sort --release -- test_sort_f32_ieee754_special`
    6. AC-2.4: Run `cargo test -p forge-sort --release -- test_sort_f32_neg_zero`
    7. AC-3.1: Run `cargo test -p forge-sort --release -- test_argsort_u32_permutation`
    8. AC-3.3: Run `cargo test -p forge-sort --release -- test_argsort_u32_input_unmodified`
    9. AC-3.5: Run `cargo test -p forge-sort --release -- test_argsort_u32_stable`
    10. AC-4.2: Run `cargo test -p forge-sort --release -- test_sort_pairs_u32_multiset`
    11. AC-4.4: Run `cargo test -p forge-sort --release -- test_sort_pairs_length_mismatch`
    12. AC-5.1: Run `cargo test -p forge-sort --release -- test_sort_u64_boundaries`
    13. AC-6.1: `grep "pub trait SortKey" forge-sort/src/lib.rs`
    14. AC-6.6: `grep 'version = "0.2.0"' forge-sort/Cargo.toml`
    15. AC-7.4: Run `cargo test -p forge-sort --release -- correctness --test-threads=1` (all v1 tests in correctness.rs)
  - **Verify**: All commands exit 0
  - **Done when**: All acceptance criteria confirmed met via automated checks
  - **Commit**: None

## Notes

- **POC shortcuts taken (Phase 1)**:
  - dispatch_sort extended with transforms rather than full SortPipelineConfig refactor
  - No argsort/KV/64-bit in Phase 1
  - Monolithic lib.rs (may exceed 800 lines by Phase 2 -- extract if needed)

- **Production TODOs (deferred)**:
  - Strategy A native KV shader (higher perf than Strategy B argsort+gather)
  - Fused transform optimization (embed FloatFlip in sort kernels to eliminate 2 dispatches)
  - Descending sort variants
  - File split: extract pipeline.rs and types.rs from lib.rs if it exceeds ~1000 lines
  - Performance regression CI (automated benchmark comparison)

- **Key implementation risks**:
  - 64-bit buffer ping-pong tracking is the most error-prone part -- 3 inner dispatches with alternating buf_a/buf_b arguments
  - HAS_VALUES in sort_inner_fused has complex value ping-pong logic matching key ping-pong
  - Function constants + MTLFunctionConstantValues in objc2-metal: well-proven in gpu-query and attention-proto codebases
