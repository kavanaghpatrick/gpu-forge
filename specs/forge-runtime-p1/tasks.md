---
spec: forge-runtime-p1
phase: tasks
total_tasks: 14
created: 2026-02-21
generated: auto
---

# Tasks: forge-runtime-p1

## Phase 1: Make It Work (POC)

Focus: Prove shared context + buffer flow + single-CB pipeline works end-to-end.

- [ ] 1.1 Create forge-runtime crate skeleton with ForgeContext
  - **Do**:
    1. Add `"forge-runtime"` to workspace members in `metal-forge-compute/Cargo.toml`
    2. Create `forge-runtime/Cargo.toml` with deps: objc2 0.6, objc2-metal 0.3, objc2-foundation 0.3, thiserror 2, forge-sort (path), forge-filter (path)
    3. Create `forge-runtime/src/lib.rs` with module declarations
    4. Create `forge-runtime/src/context.rs` with ForgeContext struct: device + queue + alloc_buffer() method
    5. ForgeContext::new() calls MTLCreateSystemDefaultDevice + newCommandQueue
    6. Add device() and queue() accessor methods returning Retained (clone)
  - **Files**: `metal-forge-compute/Cargo.toml`, `forge-runtime/Cargo.toml`, `forge-runtime/src/lib.rs`, `forge-runtime/src/context.rs`
  - **Done when**: `cargo build -p forge-runtime` succeeds, ForgeContext::new() compiles
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-runtime`
  - **Commit**: `feat(forge-runtime): create crate with ForgeContext`
  - _Requirements: FR-1_
  - _Design: ForgeContext_

- [ ] 1.2 Add with_context() to GpuSorter
  - **Do**:
    1. In `forge-sort/src/lib.rs`, add `pub fn with_context(device: Retained<ProtocolObject<dyn MTLDevice>>, queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Result<Self, SortError>`
    2. Implementation: use provided device/queue instead of calling init_device_and_queue(), load metallib + pre-compile PSOs same as new()
    3. Refactor new() to call with_context() internally (create device/queue then delegate)
  - **Files**: `forge-sort/src/lib.rs`
  - **Done when**: `GpuSorter::with_context()` compiles, `GpuSorter::new()` still works, all sort tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort`
  - **Commit**: `feat(forge-sort): add with_context() constructor for shared device/queue`
  - _Requirements: FR-2, FR-11_
  - _Design: with_context() Constructors_

- [ ] 1.3 Add with_context() to GpuFilter
  - **Do**:
    1. In `forge-filter/src/lib.rs`, add `pub fn with_context(device: Retained<ProtocolObject<dyn MTLDevice>>, queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Result<Self, FilterError>`
    2. Implementation: same pattern as GpuSorter -- use provided device/queue, load metallib, pre-compile PSOs
    3. Refactor new() to call with_context() internally
  - **Files**: `forge-filter/src/lib.rs`
  - **Done when**: `GpuFilter::with_context()` compiles, `GpuFilter::new()` still works, all filter tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter`
  - **Commit**: `feat(forge-filter): add with_context() constructor for shared device/queue`
  - _Requirements: FR-3, FR-11_
  - _Design: with_context() Constructors_

- [ ] 1.4 Add from_raw_parts / into_raw_parts to SortBuffer and FilterBuffer
  - **Do**:
    1. Add `SortBuffer::from_raw_parts(buffer, len, capacity) -> Self` to forge-sort/src/lib.rs
    2. Add `SortBuffer::into_raw_parts(self) -> (Retained<MTLBuffer>, usize, usize)` to forge-sort/src/lib.rs
    3. Add `FilterBuffer::from_raw_parts(buffer, len, capacity) -> Self` to forge-filter/src/lib.rs
    4. Add `FilterBuffer::into_raw_parts(self) -> (Retained<MTLBuffer>, usize, usize)` to forge-filter/src/lib.rs
    5. Add `FilterResult::take_values_buffer(self) -> Option<(Retained<MTLBuffer>, usize, usize)>` to forge-filter/src/lib.rs
    6. Add `FilterResult::take_indices_buffer(self) -> Option<(Retained<MTLBuffer>, usize)>` to forge-filter/src/lib.rs
  - **Files**: `forge-sort/src/lib.rs`, `forge-filter/src/lib.rs`
  - **Done when**: All new methods compile, existing tests still pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort && cargo test -p forge-filter`
  - **Commit**: `feat(forge-sort,forge-filter): add from_raw_parts/into_raw_parts for buffer interop`
  - _Requirements: FR-12_
  - _Design: Buffer Conversions_

- [ ] 1.5 Add ForgeBuffer trait and conversion functions
  - **Do**:
    1. Create `forge-runtime/src/buffer.rs` with `ForgeBuffer<T>` trait: metal_buffer(), len(), capacity(), is_empty()
    2. Add conversion functions: `filter_result_to_sort_buffer<T>(FilterResult<T>) -> SortBuffer<T>` using take_values_buffer + from_raw_parts
    3. Add `sort_buffer_to_filter_buffer<T>(SortBuffer<T>) -> FilterBuffer<T>` using into_raw_parts + from_raw_parts
    4. Export from lib.rs
  - **Files**: `forge-runtime/src/buffer.rs`, `forge-runtime/src/lib.rs`
  - **Done when**: `cargo build -p forge-runtime` succeeds with buffer module
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-runtime`
  - **Commit**: `feat(forge-runtime): add ForgeBuffer trait and buffer conversion functions`
  - _Requirements: FR-4, FR-5, FR-6_
  - _Design: ForgeBuffer<T> Trait, Buffer Conversions_

- [ ] 1.6 POC: filter-then-sort with shared context (no pipeline builder yet)
  - **Do**:
    1. Create `forge-runtime/tests/poc_test.rs`
    2. Test: create ForgeContext, create GpuSorter::with_context + GpuFilter::with_context
    3. Filter 100K random u32 with Predicate::Gt(50_000)
    4. Convert FilterResult to SortBuffer using conversion function
    5. Sort the result
    6. Verify output is sorted and all values > 50_000
    7. Each primitive still creates its own command buffer (no pipeline yet)
  - **Files**: `forge-runtime/tests/poc_test.rs`
  - **Done when**: Test passes -- filter output flows into sort, result is correct
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-runtime -- poc`
  - **Commit**: `test(forge-runtime): POC proving filter→sort with shared context`
  - _Requirements: AC-1.2, AC-1.3, AC-2.2_
  - _Design: Data Flow_

- [ ] 1.7 POC Checkpoint
  - **Do**: Verify shared context + buffer conversion works end-to-end
  - **Done when**: POC test passes, all existing forge-sort + forge-filter tests still pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort && cargo test -p forge-filter && cargo test -p forge-runtime`
  - **Commit**: `feat(forge-runtime): complete POC — shared context + buffer interop`

## Phase 2: Pipeline Builder + Gather Kernel

After POC validated, build the pipeline and gather kernel.

- [ ] 2.1 Create GPU Gather kernel and build.rs
  - **Do**:
    1. Create `forge-runtime/shaders/gather.metal` with gather_u32 and gather_u64 kernels
    2. Create `forge-runtime/build.rs` that compiles gather.metal via `xcrun metal` + `xcrun metallib`, sets `GATHER_METALLIB_PATH` env
    3. Copy build.rs pattern from forge-sort/build.rs or forge-filter/build.rs
    4. Create `forge-runtime/src/gather.rs` with encode_gather() that loads metallib, creates PSO, encodes dispatch
    5. Test gather standalone: create indices [2,0,1,3], source [10,20,30,40], verify output [30,10,20,40]
  - **Files**: `forge-runtime/shaders/gather.metal`, `forge-runtime/build.rs`, `forge-runtime/src/gather.rs`
  - **Done when**: `cargo test -p forge-runtime -- gather` passes with correct permutation
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-runtime -- gather`
  - **Commit**: `feat(forge-runtime): add GPU gather kernel (u32 + u64)`
  - _Requirements: FR-9, FR-10_
  - _Design: GPU Gather Kernel_

- [ ] 2.2 Add encode_sort() public method to GpuSorter
  - **Do**:
    1. In forge-sort/src/lib.rs, add `pub fn encode_sort(&mut self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, buf: &SortBuffer<u32>) -> Result<(), SortError>`
    2. Implementation: ensure scratch buffers, zero MSD histogram, call existing `encode_sort_pipeline()` with the external encoder
    3. This method does NOT create its own CB/encoder -- uses the one passed in
    4. Add unit test that creates external CB+encoder, calls encode_sort, commits, verifies sort is correct
  - **Files**: `forge-sort/src/lib.rs`
  - **Done when**: encode_sort() works with externally-provided encoder, all existing tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-sort`
  - **Commit**: `feat(forge-sort): add encode_sort() for external encoder pipeline integration`
  - _Requirements: FR-8_
  - _Design: Pipeline Builder_

- [ ] 2.3 Add encode_filter() public method to GpuFilter
  - **Do**:
    1. In forge-filter/src/lib.rs, add `pub fn encode_filter<T: FilterKey>(&mut self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, input_buf: &ProtocolObject<dyn MTLBuffer>, n: usize, pred: &Predicate<T>) -> Result<FilterResult<T>, FilterError>`
    2. Implementation: ensure scratch buffers, encode predicate_scan + scan_partials + scatter onto the provided encoder
    3. Does NOT create own CB -- uses the passed encoder. Does NOT commit.
    4. Returns FilterResult with count read AFTER the pipeline commits (caller must commit first)
    5. Alternative: return a PendingFilterResult that resolves after commit
  - **Files**: `forge-filter/src/lib.rs`
  - **Done when**: encode_filter() compiles, existing tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-filter`
  - **Commit**: `feat(forge-filter): add encode_filter() for external encoder pipeline integration`
  - _Requirements: FR-8_
  - _Design: Pipeline Builder_

- [ ] 2.4 Build Pipeline struct and execute method
  - **Do**:
    1. Create `forge-runtime/src/pipeline.rs` with Pipeline struct
    2. Pipeline::new(ctx: &ForgeContext) -> creates CB + encoder from ctx.queue
    3. Add pipeline.filter() that calls GpuFilter::encode_filter()
    4. Add pipeline.sort() that calls GpuSorter::encode_sort()
    5. Add pipeline.gather() that calls encode_gather()
    6. Add pipeline.execute() that does endEncoding + commit + waitUntilCompleted
    7. Handle the deferred count read for filter results (read from count buffer after execute)
  - **Files**: `forge-runtime/src/pipeline.rs`, `forge-runtime/src/lib.rs`
  - **Done when**: Pipeline compiles and basic filter->sort pipeline works via Pipeline API
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo build -p forge-runtime`
  - **Commit**: `feat(forge-runtime): add Pipeline builder for single-CB multi-dispatch`
  - _Requirements: FR-7, FR-8_
  - _Design: Pipeline Builder_

## Phase 3: Testing

- [ ] 3.1 Integration tests for full pipeline
  - **Do**:
    1. Create `forge-runtime/tests/pipeline_tests.rs`
    2. Test filter->sort: 1M random u32, filter Gt(500K), sort result, verify sorted + all > 500K
    3. Test filter->sort->gather: filter, sort, gather top 1000 elements
    4. Test sort->gather: sort 100K elements, gather first 100 by index
    5. Test filter with zero matches: pipeline handles empty intermediate buffers
    6. Test each numeric type: u32, i32, f32 at minimum
    7. Verify identical results to standalone sequential execution
  - **Files**: `forge-runtime/tests/pipeline_tests.rs`
  - **Done when**: All integration tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test -p forge-runtime`
  - **Commit**: `test(forge-runtime): integration tests for pipeline compositions`
  - _Requirements: AC-3.1 through AC-3.5, AC-4.4_
  - _Design: Data Flow_

- [ ] 3.2 Backward compatibility tests
  - **Do**:
    1. Run full forge-sort test suite -- must pass unchanged
    2. Run full forge-filter test suite -- must pass unchanged
    3. Verify no public API changes that break semver (no removed types/methods)
    4. Add a test in forge-runtime that uses both ::new() (standalone) and ::with_context() and verifies identical results
  - **Files**: `forge-runtime/tests/compat_test.rs`
  - **Done when**: All existing tests pass, compat test passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test --workspace`
  - **Commit**: `test(forge-runtime): backward compatibility verification`
  - _Requirements: AC-5.1 through AC-5.4_
  - _Design: Backward compatibility_

## Phase 4: Quality Gates

- [ ] 4.1 Local quality check
  - **Do**: Run all quality checks locally
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-forge-compute && cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo doc --workspace --no-deps`
  - **Done when**: All commands pass with zero warnings
  - **Commit**: `fix(forge-runtime): address lint/type issues` (if needed)

- [ ] 4.2 Create PR and verify CI
  - **Do**: Push branch, create PR with gh CLI
  - **Verify**: `gh pr checks --watch` all green
  - **Done when**: PR ready for review

## Notes

- **POC shortcuts taken**: POC test (1.6) uses separate command buffers per primitive -- pipeline builder (2.4) upgrades to single CB
- **Production TODOs**: encode_filter() count read is deferred until after CB commit -- Pipeline.execute() must handle this
- **Key risk**: encode_filter's 3-dispatch pipeline needs count buffer zeroed by CPU before encoding. In pipeline mode, must zero before encoding, not before commit. The existing pattern (CPU write_bytes before encoding) works since CPU writes to StorageModeShared are visible to GPU.
