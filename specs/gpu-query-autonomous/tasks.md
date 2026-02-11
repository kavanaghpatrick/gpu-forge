---
spec: gpu-query-autonomous
phase: tasks
total_tasks: 44
created: 2026-02-11
generated: auto
---

# Tasks: gpu-query-autonomous

## Phase 1: Foundation (Types + Shared Headers + Binary Columnar Loader)

Focus: Establish the shared data formats between Rust and MSL. Load data into GPU-resident buffers. No GPU execution yet.

- [x] 1.1 Create autonomous module structure
  - **Do**: Create `src/gpu/autonomous/mod.rs` with `pub mod types; pub mod work_queue; pub mod loader; pub mod executor; pub mod jit;`. Add `pub mod autonomous;` to `src/gpu/mod.rs`. Ensure `cargo build` compiles.
  - **Files**: `gpu-query/src/gpu/autonomous/mod.rs`, `gpu-query/src/gpu/mod.rs`
  - **Done when**: `cargo build` succeeds with empty sub-modules
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(autonomous): scaffold autonomous module structure`
  - _Requirements: FR-1_
  - _Design: Component 6_

- [x] 1.2 Define #[repr(C)] shared types with layout tests
  - **Do**: Create `src/gpu/autonomous/types.rs` with `FilterSpec` (48B), `AggSpec` (16B), `QueryParamsSlot` (512B), `ColumnMeta` (32B), `OutputBuffer` (~22KB), `AggResult` (16B). Follow exact struct definitions from design.md. Add comprehensive layout tests: size assertions (`assert_eq!(std::mem::size_of::<FilterSpec>(), 48)`), alignment assertions, offset assertions using `std::mem::offset_of!`, and non-zero round-trip tests. Follow existing pattern in `src/gpu/types.rs`.
  - **Files**: `gpu-query/src/gpu/autonomous/types.rs`
  - **Done when**: All size/alignment/offset tests pass. ~35 tests.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::types -- --test-threads=1`
  - **Commit**: `feat(autonomous): define #[repr(C)] shared types with layout tests`
  - _Requirements: FR-2, FR-3, FR-5_
  - _Design: Shared Struct Definitions_

- [x] 1.3 Create MSL shared type header
  - **Do**: Create `shaders/autonomous_types.h` with MSL counterparts of all Rust types: `FilterSpec`, `AggSpec`, `QueryParamsSlot`, `ColumnMeta`, `OutputBuffer`, `AggResult`. Byte-identical layouts. Include `#pragma once` and `#include <metal_stdlib>`. Add `#define MAX_GROUPS 64`, `#define MAX_AGGS 5`, `#define MAX_FILTERS 4`.
  - **Files**: `gpu-query/shaders/autonomous_types.h`
  - **Done when**: Header compiles when included by a .metal file (verified via build.rs)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(autonomous): add MSL shared type definitions header`
  - _Requirements: FR-3_
  - _Design: Shared Struct Definitions_

- [x] 1.4 Implement triple-buffered work queue
  - **Do**: Create `src/gpu/autonomous/work_queue.rs`. Implement `WorkQueue` struct wrapping a Metal buffer (3 x 512B = 1536B, StorageModeShared). Methods: `new(device) -> Self` (allocate buffer), `write_params(params: &QueryParamsSlot)` (write to current slot, bump sequence_id with Release ordering, advance write_idx mod 3), `read_latest_sequence_id() -> u32` (for CPU-side debug). Add unit tests: buffer size, shared mode, write_idx cycles 0->1->2->0, sequence_id monotonic, slot population correctness.
  - **Files**: `gpu-query/src/gpu/autonomous/work_queue.rs`
  - **Done when**: ~12 unit tests pass. Work queue allocates and writes correctly.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::work_queue -- --test-threads=1`
  - **Commit**: `feat(autonomous): implement triple-buffered work queue`
  - _Requirements: FR-2_
  - _Design: Component 2_

- [x] 1.5 Implement binary columnar data loader
  - **Do**: Create `src/gpu/autonomous/loader.rs`. Implement `BinaryColumnarLoader` with method `load_table(device, table_name, schema, batch, progress_tx) -> Result<ResidentTable, String>`. Convert `ColumnarBatch` columns to contiguous Metal buffers: INT64 as `i64[]`, FLOAT64 as `f32[]` (downcast), VARCHAR as `u32[]` (dictionary codes). Allocate page-aligned (16KB) Metal buffers with StorageModeShared. Build `ColumnMeta` array. Store result in `ResidentTable { data_buffer, column_metas, column_meta_buffer, row_count, schema, dictionaries }`. Report progress via channel.
  - **Files**: `gpu-query/src/gpu/autonomous/loader.rs`
  - **Done when**: Loads 1K-row test data into Metal buffers, round-trips correctly. ~14 tests.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::loader -- --test-threads=1`
  - **Commit**: `feat(autonomous): implement binary columnar data loader`
  - _Requirements: FR-4_
  - _Design: Component 5_

- [x] 1.6 Foundation checkpoint
  - **Do**: Verify all foundation components work together: types compile, MSL header included, work queue allocates, loader converts data. Run full regression to confirm no breakage.
  - **Done when**: All existing tests pass + all new foundation tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test -- --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(autonomous): complete foundation phase`
  - _Requirements: FR-1 through FR-5_

## Phase 2: Fused Kernel (Single-Pass Filter+Aggregate+GROUP BY)

Focus: Prove the fused kernel works with standard CPU-orchestrated dispatch (before persistent kernel). Validate correctness.

- [x] 2.1 Create AOT fused query Metal shader
  - **Do**: Create `shaders/fused_query.metal`. Implement `fused_query` kernel that: (1) reads `QueryParamsSlot` from buffer(0), (2) reads column data from buffer(1) via `ColumnMeta` from buffer(2), (3) evaluates up to 4 filter predicates (AND compound), (4) buckets passing rows into threadgroup-local `GroupAccumulator[64]` hash table, (5) performs simd reductions for COUNT/SUM/MIN/MAX/AVG, (6) merges threadgroup partials to global `OutputBuffer` via device atomics, (7) sets ready_flag. Use function constants for specialization: `FILTER_COUNT`, `AGG_COUNT`, `HAS_GROUP_BY`. Include `autonomous_types.h`. Include helper functions from `aggregate.metal` pattern: `simd_sum_int64`, `simd_min_int64`, `simd_max_int64`.
  - **Files**: `gpu-query/shaders/fused_query.metal`
  - **Done when**: Shader compiles via build.rs without errors
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(autonomous): implement AOT fused query Metal shader`
  - _Requirements: FR-3, FR-7, FR-8, FR-9_
  - _Design: Component 3_

- [x] 2.2 Add fused kernel PSO compilation to pipeline
  - **Do**: Add `compile_fused_pso(device, filter_count, agg_count, has_group_by) -> PSO` function in a new section of `src/gpu/autonomous/executor.rs` (or a helper module). Use `MTLFunctionConstantValues` for `FILTER_COUNT`, `AGG_COUNT`, `HAS_GROUP_BY`. Cache compiled PSOs in `HashMap<(u32,u32,bool), PSO>`. Follow existing pattern in `src/gpu/pipeline.rs::compile_pso()`.
  - **Files**: `gpu-query/src/gpu/autonomous/executor.rs`
  - **Done when**: PSO compiles for headline query pattern (2 filters, 2 aggs, group_by=true)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::executor::test_fused_pso_compilation -- --test-threads=1`
  - **Commit**: `feat(autonomous): add fused kernel PSO compilation`
  - _Requirements: FR-3_
  - _Design: Component 3_

- [x] 2.3 Implement one-shot fused kernel dispatch
  - **Do**: In `executor.rs`, add `execute_fused_oneshot(device, pso, params_slot, resident_table) -> OutputBuffer`. This is a STANDARD Metal dispatch (not persistent yet): create command buffer, create compute encoder, set buffers (work queue slot, data buffer, column meta, output buffer, ready flag), dispatch threadgroups (row_count / 256), commit, `waitUntilCompleted`. Read output buffer. This proves the fused kernel correctness before adding persistent complexity.
  - **Files**: `gpu-query/src/gpu/autonomous/executor.rs`
  - **Done when**: `SELECT COUNT(*) FROM sales` returns correct count via fused kernel one-shot dispatch
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::executor::test_fused_oneshot_count -- --test-threads=1`
  - **Commit**: `feat(autonomous): implement one-shot fused kernel dispatch`
  - _Requirements: FR-3_
  - _Design: Component 3_

- [x] 2.4 Test fused kernel with compound filters
  - **Do**: Add integration tests for fused kernel with various query patterns: (1) COUNT(*) no filter, (2) SUM(amount), (3) MIN/MAX, (4) AVG, (5) single filter GT, (6) compound AND filter (2 predicates), (7) GROUP BY without filter, (8) compound filter + GROUP BY + multi-agg (headline query). For each test, compare fused kernel result against known-correct values computed on CPU. Use deterministic test data (amount = (i*7+13)%1000, region = i%5).
  - **Files**: `gpu-query/tests/autonomous_integration.rs` (new file)
  - **Done when**: All 8 query patterns return correct results via one-shot fused dispatch
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test --test autonomous_integration -- --test-threads=1`
  - **Commit**: `test(autonomous): verify fused kernel correctness across query patterns`
  - _Requirements: FR-3, FR-7, FR-8, FR-9_
  - _Design: Component 3_

- [x] 2.5 Add parity tests (fused vs standard executor)
  - **Do**: For each supported query pattern, run the same query via the existing `QueryExecutor` (36ms path) and the new fused kernel one-shot dispatch. Assert results match. Integer operations (COUNT, MIN, MAX, integer SUM) must be exact. Float SUM/AVG use relative tolerance 1e-5. Use 100K-row deterministic test data.
  - **Files**: `gpu-query/tests/autonomous_integration.rs`
  - **Done when**: Parity verified for all query patterns. ~10 parity tests pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test --test autonomous_integration parity -- --test-threads=1`
  - **Commit**: `test(autonomous): add fused vs standard executor parity tests`
  - _Requirements: FR-3_

- [x] 2.6 Fused kernel checkpoint
  - **Do**: Verify fused kernel produces correct results for all supported patterns. Run full regression.
  - **Done when**: All tests pass (existing + new). Fused kernel correctness proven.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test -- --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(autonomous): complete fused kernel phase (POC validated)`

## Phase 3: JIT Compiler (Metal Source Generation + Runtime Compilation)

Focus: Generate specialized Metal shader source from query plan, compile at runtime, cache PSOs.

- [x] 3.1 Implement plan structure hashing
  - **Do**: Create `src/gpu/autonomous/jit.rs`. Implement `plan_structure_hash(plan: &PhysicalPlan) -> u64` using `DefaultHasher`. Hash captures: plan node types (GpuAggregate, GpuFilter, GpuCompoundFilter, GpuScan), aggregate functions, compare ops, column references, group_by columns. Does NOT hash literal values (thresholds). Add tests: deterministic (same plan -> same hash), structural equality (different literals -> same hash), different structure -> different hash, filter op matters, agg func matters, group_by matters.
  - **Files**: `gpu-query/src/gpu/autonomous/jit.rs`
  - **Done when**: ~8 plan hash unit tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::jit::test_plan_hash -- --test-threads=1`
  - **Commit**: `feat(autonomous): implement plan structure hashing for JIT cache`
  - _Requirements: FR-6_
  - _Design: Component 4_

- [x] 3.2 Implement Metal source generator
  - **Do**: In `jit.rs`, implement `JitCompiler::generate_metal_source(plan, schema) -> String`. Template engine emits specialized MSL: (1) header with includes, (2) inline filter functions per predicate (baked comparison operator, no branches), (3) inline aggregate accumulation per function, (4) main `fused_query` kernel with exact operations inlined. The generated kernel reads params from `QueryParamsSlot` for literal values but has specialized code structure.
  - **Files**: `gpu-query/src/gpu/autonomous/jit.rs`
  - **Done when**: Generated MSL for headline query is syntactically valid (contains expected patterns). ~10 source generation tests pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::jit::test_generate -- --test-threads=1`
  - **Commit**: `feat(autonomous): implement Metal source generator for JIT compilation`
  - _Requirements: FR-6_
  - _Design: Component 4_

- [x] 3.3 Implement runtime compilation and PSO cache
  - **Do**: In `jit.rs`, implement `JitCompiler` struct with `device`, `cache: HashMap<u64, CompiledPlan>`. Method `compile(plan, schema) -> Result<&CompiledPlan, String>`: check cache, on miss generate source, call `device.newLibraryWithSource_options_error()`, create PSO via `device.newComputePipelineStateWithFunction_error()`, insert into cache. `CompiledPlan { pso, plan_hash, source_len }`. Add tests: compiles valid MSL, creates PSO, cache hit returns same PSO, cache miss compiles new, invalid source returns Err.
  - **Files**: `gpu-query/src/gpu/autonomous/jit.rs`
  - **Done when**: JIT compiles and caches PSO for headline query pattern. ~6 tests pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::jit::test_compile -- --test-threads=1`
  - **Commit**: `feat(autonomous): implement JIT runtime compilation with PSO cache`
  - _Requirements: FR-6_
  - _Design: Component 4_

- [x] 3.4 Wire JIT compiler into fused dispatch
  - **Do**: Modify `execute_fused_oneshot` to accept a JIT-compiled PSO instead of the AOT PSO. Add `execute_jit_oneshot(jit_compiler, plan, schema, resident_table) -> OutputBuffer` that: (1) compiles via JIT, (2) dispatches the JIT PSO, (3) reads output. Run parity tests with JIT PSO vs AOT PSO -- results must match.
  - **Files**: `gpu-query/src/gpu/autonomous/executor.rs`, `gpu-query/tests/autonomous_integration.rs`
  - **Done when**: JIT-compiled kernel produces identical results to AOT kernel for all test patterns
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test --test autonomous_integration jit -- --test-threads=1`
  - **Commit**: `feat(autonomous): wire JIT compiler into fused kernel dispatch`
  - _Requirements: FR-6_
  - _Design: Component 4_

- [x] 3.5 JIT checkpoint
  - **Do**: Verify JIT compilation, caching, and dispatch work end-to-end. Run full regression.
  - **Done when**: JIT produces correct results, cache hit is <0.01ms, all tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test -- --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(autonomous): complete JIT compiler phase`

## Phase 4: Persistent Kernel (Work Queue + Re-Dispatch Chain + Autonomous Executor)

Focus: Prove GPU autonomy. Eliminate per-query command buffer creation. This is the core architectural inversion.

- [x] 4.1 Implement re-dispatch chain
  - **Do**: In `executor.rs`, implement the re-dispatch chain: `dispatch_slice()` creates a command buffer, encodes the fused kernel dispatch, commits, and registers a completion handler that calls `dispatch_slice()` again (immediate re-dispatch). Add `enqueue()` for next command buffer to hide gap [KB #152]. Track re-dispatch count. Add idle detection: if no new queries for 500ms, completion handler does NOT re-dispatch; sets state to Idle.
  - **Files**: `gpu-query/src/gpu/autonomous/executor.rs`
  - **Done when**: Re-dispatch chain runs for 10 seconds without GPU watchdog kill. Idle timeout works.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::executor::test_redispatch_chain -- --test-threads=1`
  - **Commit**: `feat(autonomous): implement completion-handler re-dispatch chain`
  - _Requirements: FR-1_
  - _Design: Component 1_

- [x] 4.2 Implement MTLSharedEvent idle/wake
  - **Do**: Create MTLSharedEvent on the device. On idle: completion handler skips re-dispatch, sets state=Idle. On new query submission when idle: signal MTLSharedEvent, dispatch new slice, set state=Active. Add test: submit query -> idle timeout -> submit again -> wake from idle -> correct result.
  - **Files**: `gpu-query/src/gpu/autonomous/executor.rs`, `gpu-query/src/gpu/device.rs`
  - **Done when**: Idle/wake cycle works correctly
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::executor::test_idle_wake -- --test-threads=1`
  - **Commit**: `feat(autonomous): implement MTLSharedEvent idle/wake cycle`
  - _Requirements: FR-1_
  - _Design: Component 1_

- [x] 4.3 Build AutonomousExecutor struct
  - **Do**: Implement full `AutonomousExecutor` struct combining all components: `device`, `command_queue` (separate), `work_queue` (WorkQueue), `output_buffer`, `control_buffer`, `jit_compiler` (JitCompiler), `resident_tables` (HashMap), `state` (AtomicU8 for EngineState), `shared_event`, `stats`. Lifecycle methods: `new()`, `load_table()`, `submit_query()` (JIT lookup + write work queue + wake if idle), `poll_ready()` (atomic load on output_buffer.ready_flag), `read_result()` (read unified memory + reset flag), `shutdown()`.
  - **Files**: `gpu-query/src/gpu/autonomous/executor.rs`
  - **Done when**: Full lifecycle works: new -> load_table -> submit_query -> poll_ready -> read_result -> shutdown
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::executor::test_full_lifecycle -- --test-threads=1`
  - **Commit**: `feat(autonomous): build complete AutonomousExecutor struct`
  - _Requirements: FR-1 through FR-6_
  - _Design: Component 6_

- [x] 4.4 End-to-end autonomous query test
  - **Do**: Integration test that: (1) creates AutonomousExecutor, (2) loads 100K-row deterministic data, (3) submits headline query via submit_query(), (4) polls ready_flag, (5) reads result, (6) verifies correctness against known values. This is the first truly autonomous query -- no `waitUntilCompleted`, no per-query command buffer.
  - **Files**: `gpu-query/tests/autonomous_integration.rs`
  - **Done when**: Autonomous headline query returns correct result without `waitUntilCompleted`
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test --test autonomous_integration test_autonomous_headline -- --test-threads=1`
  - **Commit**: `feat(autonomous): verify end-to-end autonomous query execution`
  - _Requirements: FR-1 through FR-6_

- [x] 4.5 Test 1000 sequential queries without restart
  - **Do**: Submit 1000 queries with varying parameters (different thresholds) to the same AutonomousExecutor. Verify each returns correct results. No re-initialization between queries.
  - **Files**: `gpu-query/tests/autonomous_integration.rs`
  - **Done when**: 1000 consecutive queries all return correct results
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test --test autonomous_integration test_1000_queries -- --test-threads=1`
  - **Commit**: `test(autonomous): verify 1000 consecutive autonomous queries`
  - _Requirements: AC-2.5_

- [x] 4.6 Autonomous kernel checkpoint
  - **Do**: GPU autonomy proven. Verify all tests pass (existing + new). Full regression.
  - **Done when**: Autonomous executor processes queries via persistent kernel. No `waitUntilCompleted` in hot path.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test -- --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(autonomous): complete persistent kernel phase (GPU autonomy achieved)`

## Phase 5: TUI Integration (Live Mode + Engine Status + Fallback)

Focus: Wire autonomous executor into the TUI. Enable live mode. Add engine status display.

- [x] 5.1 Add autonomous state fields to AppState
  - **Do**: In `src/tui/app.rs`, add fields: `autonomous_executor: Option<AutonomousExecutor>`, `live_mode: bool` (default false), `engine_status: EngineStatus` (enum Off/WarmingUp/Compiling/Live/Idle/Fallback/Error), `warmup_progress: f32`, `last_autonomous_us: Option<u64>`, `autonomous_stats: AutonomousStats` (struct with total_queries, fallback_queries, avg_latency_us, p99_latency_us, consecutive_sub_1ms), `sql_validity: SqlValidity` (enum Empty/Incomplete/ParseError/Valid), `query_compatibility: QueryCompatibility` (enum Unknown/Autonomous/Fallback/Invalid), `cached_plan: Option<PhysicalPlan>`. Also add `QueryState::AutonomousSubmitted` variant.
  - **Files**: `gpu-query/src/tui/app.rs`
  - **Done when**: AppState compiles with new fields. Existing TUI tests still pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build && cargo test -- --test-threads=1 2>&1 | tail -5`
  - **Commit**: `feat(autonomous): add TUI state fields for autonomous engine`
  - _Requirements: UX-1 through UX-10_
  - _Design: File Structure (Existing Files to Modify)_

- [x] 5.2 Implement autonomous compatibility check
  - **Do**: Add `check_autonomous_compatibility(plan: &PhysicalPlan) -> QueryCompatibility` function. Returns Autonomous for supported patterns (scan, filter, compound filter, aggregate with <=1 GROUP BY column, cardinality <=64). Returns Fallback with reason string for ORDER BY, multi-column GROUP BY, etc. Add `update_sql_validity(app)` and `update_query_compatibility(app)` functions.
  - **Files**: `gpu-query/src/tui/app.rs` or `gpu-query/src/gpu/autonomous/executor.rs`
  - **Done when**: Correctly classifies various query patterns. ~10 unit tests.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::test_compatibility -- --test-threads=1`
  - **Commit**: `feat(autonomous): implement query compatibility checking`
  - _Requirements: FR-10, UX-9_

- [x] 5.3 Integrate autonomous executor into event loop
  - **Do**: Modify `src/tui/mod.rs` event loop: (1) Poll `autonomous_executor.poll_ready()` on every tick -- if ready, read result, update app state. (2) In live mode with valid autonomous-compatible SQL, call `submit_query()` on every keystroke (0ms debounce). (3) For fallback queries in live mode, use existing `execute_editor_query()`. (4) Check warm-up progress via channel.
  - **Files**: `gpu-query/src/tui/mod.rs`
  - **Done when**: Autonomous queries execute via event loop polling. Live mode works.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build && cargo test -- --test-threads=1 2>&1 | tail -5`
  - **Commit**: `feat(autonomous): integrate autonomous executor into TUI event loop`
  - _Requirements: UX-1, UX-2_
  - _Design: Event Loop Integration_

- [x] 5.4 Add Ctrl+L live mode toggle
  - **Do**: In `src/tui/event.rs`, add handler for Ctrl+L: toggle `app.live_mode`. When toggling ON, show status "Live mode ON". When toggling OFF, show "Live mode OFF. Press F5 to execute." Also add keystroke handling: in live mode, after each editor keystroke, call `update_sql_validity()` and `update_query_compatibility()`.
  - **Files**: `gpu-query/src/tui/event.rs`
  - **Done when**: Ctrl+L toggles live mode. Keystrokes trigger validity checking.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build`
  - **Commit**: `feat(autonomous): add Ctrl+L live mode toggle`
  - _Requirements: UX-1_

- [x] 5.5 Add engine status badge to dashboard
  - **Do**: In `src/tui/dashboard.rs`, add an "ENGINE" section after existing GPU metrics. Display: engine status badge ([LIVE]/[WARMING]/[IDLE]/[FALLBACK]/[OFF]/[ERROR]), last latency, average latency, queries processed count, JIT cache stats (plans compiled, misses). Follow existing dashboard rendering patterns.
  - **Files**: `gpu-query/src/tui/dashboard.rs`
  - **Done when**: Dashboard shows engine status section with correct badge
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build`
  - **Commit**: `feat(autonomous): add engine status badge to GPU dashboard`
  - _Requirements: UX-3, UX-10_

- [x] 5.6 Update results panel for autonomous queries
  - **Do**: In `src/tui/results.rs`, modify `build_performance_line()`: (1) Add `[auto]` tag to results title when result came from autonomous path. (2) Show microsecond precision for sub-1ms queries ("0.42ms (420us)"). (3) When falling back, show reason: "36.2ms (ORDER BY requires standard path)". (4) Add `| autonomous` or `| standard path` suffix. Follow existing performance line pattern.
  - **Files**: `gpu-query/src/tui/results.rs`
  - **Done when**: Results panel correctly labels autonomous vs standard results
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build`
  - **Commit**: `feat(autonomous): update results panel for autonomous query display`
  - _Requirements: UX-4, UX-5_

- [x] 5.7 Implement background warm-up with progress
  - **Do**: When TUI starts with a data directory, spawn background thread to: (1) scan for tables, (2) load each via `AutonomousExecutor::load_table()`, (3) send progress updates via mpsc channel, (4) on completion, set `engine_status = EngineStatus::Live`, `live_mode = true`. TUI shows progress in dashboard. If user presses F5 before warm-up complete, fall back to standard executor with message.
  - **Files**: `gpu-query/src/tui/mod.rs`, `gpu-query/src/tui/app.rs`
  - **Done when**: TUI launches instantly, background loads data, transitions to LIVE
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build`
  - **Commit**: `feat(autonomous): implement background warm-up with progress display`
  - _Requirements: UX-6_

- [ ] 5.8 Wire fallback path
  - **Do**: When `check_autonomous_compatibility()` returns Fallback, route query to existing `QueryExecutor`. Set `engine_status = EngineStatus::Fallback`. Display fallback reason in performance line. After query completes, restore engine_status to Live if autonomous executor is still warm.
  - **Files**: `gpu-query/src/tui/ui.rs`, `gpu-query/src/tui/mod.rs`
  - **Done when**: ORDER BY queries seamlessly fall back to standard path with clear status
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build`
  - **Commit**: `feat(autonomous): wire fallback path for unsupported queries`
  - _Requirements: FR-10_

- [ ] 5.9 TUI integration checkpoint
  - **Do**: Full TUI works with autonomous engine: live mode, fallback, status display, warm-up. Run full regression.
  - **Done when**: TUI functional with both autonomous and standard paths. All tests pass.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test -- --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(autonomous): complete TUI integration phase`

## Phase 6: Testing (Unit + Integration + Benchmarks + Stress)

Focus: Comprehensive test coverage. Benchmark validation. Stress testing for stability.

- [ ] 6.1 Add comprehensive unit tests for work queue concurrency
  - **Do**: Expand `work_queue.rs` tests: (1) wrap-around after 100+ writes, (2) concurrent write/read simulation (different slots), (3) sequence_id written last (verify Release ordering), (4) stale sequence detection. Optionally add loom tests behind `#[cfg(feature = "loom")]` feature flag for exhaustive interleaving checks.
  - **Files**: `gpu-query/src/gpu/autonomous/work_queue.rs`
  - **Done when**: ~15 work queue tests pass. No torn read scenarios.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::work_queue -- --test-threads=1`
  - **Commit**: `test(autonomous): comprehensive work queue concurrency tests`
  - _Requirements: AC-3.4_

- [ ] 6.2 Add comprehensive unit tests for JIT compiler
  - **Do**: Expand `jit.rs` tests: (1) source generation for each query pattern (count, sum, filter, compound, groupby, multi-agg), (2) source includes expected patterns (atomic_fetch_add for COUNT, simd_sum_int64 for SUM, comparison operator), (3) source does NOT include unnecessary code (no hash table when no GROUP BY), (4) compile invalid source returns Err.
  - **Files**: `gpu-query/src/gpu/autonomous/jit.rs`
  - **Done when**: ~20 JIT tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test autonomous::jit -- --test-threads=1`
  - **Commit**: `test(autonomous): comprehensive JIT compiler tests`
  - _Requirements: FR-6_

- [ ] 6.3 Add edge case integration tests
  - **Do**: Add tests for: (1) empty table (0 rows) -> count=0, no crash, (2) single row -> correct scalar, (3) 257 rows (2 threadgroups boundary) -> correct cross-threadgroup reduction, (4) all identical values -> correct, (5) negative values -> sign preserved, (6) 64 distinct groups (max GROUP BY) -> all 64 present, (7) 65 groups -> falls back to standard. (8) NULL handling tests if null bitmaps implemented.
  - **Files**: `gpu-query/tests/autonomous_integration.rs`
  - **Done when**: All edge case tests pass. ~12 tests.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test --test autonomous_integration edge -- --test-threads=1`
  - **Commit**: `test(autonomous): add edge case integration tests`

- [ ] 6.4 Add fallback integration tests
  - **Do**: Test fallback scenarios: (1) ORDER BY -> standard executor with reason, (2) GROUP BY cardinality > 64 -> standard, (3) multi-column GROUP BY -> standard, (4) query before warm-up -> standard with message, (5) after warm-up completes -> autonomous.
  - **Files**: `gpu-query/tests/autonomous_integration.rs`
  - **Done when**: All fallback tests pass. ~7 tests.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test --test autonomous_integration fallback -- --test-threads=1`
  - **Commit**: `test(autonomous): add fallback integration tests`
  - _Requirements: FR-10_

- [ ] 6.5 Create Criterion benchmark suite
  - **Do**: Create `benches/autonomous_latency.rs`. Benchmarks: (1) BM-01: COUNT(*) 1M rows, (2) BM-02: SUM + filter 1M rows, (3) BM-03: headline query (compound filter + GROUP BY) 1M rows -- target p50 <1ms p99 <2ms, (4) BM-04: autonomous vs standard comparison, (5) BM-05: parameter handoff latency, (6) BM-06: JIT compile (cache miss), (7) BM-07: JIT cache hit, (8) BM-08: data loading 1M rows, (9) BM-09: poll_ready latency, (10) BM-10: read_result latency. Use 100 samples, 30s measurement time, 5s warm-up for BM-03.
  - **Files**: `gpu-query/benches/autonomous_latency.rs`, `gpu-query/Cargo.toml` (add criterion bench target)
  - **Done when**: All benchmarks run. BM-03 p50 < 1ms.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo bench -- autonomous 2>&1 | grep -E "time:|mean"`
  - **Commit**: `bench(autonomous): add Criterion benchmark suite`
  - _Requirements: NFR-1, NFR-2_

- [ ] 6.6 Add stress tests
  - **Do**: Create `tests/autonomous_stress.rs` with `#[ignore]` tests: (1) `stress_memory_leak`: run 100K queries, verify Metal allocated size growth < 1%. (2) `stress_watchdog_survival`: continuous queries for 2 minutes, zero watchdog errors. (3) `stress_concurrent_submit_poll`: one thread submits at max rate, main thread polls, verify all received results correct. All tests use `--ignored` flag.
  - **Files**: `gpu-query/tests/autonomous_stress.rs`
  - **Done when**: All stress tests pass (run with `--ignored`)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test --test autonomous_stress -- --ignored --test-threads=1 --nocapture 2>&1 | tail -20`
  - **Commit**: `test(autonomous): add stress tests (memory leak, watchdog, concurrency)`
  - _Requirements: NFR-8_

- [ ] 6.7 Testing checkpoint
  - **Do**: All tests pass. Benchmarks validate <1ms target. Stress tests clean.
  - **Done when**: Full test suite green. Benchmark results documented.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test -- --test-threads=1 && cargo bench -- autonomous 2>&1 | grep "time:"`
  - **Commit**: `test(autonomous): complete testing phase`

## Phase 7: Quality Gates & PR

Focus: Final quality checks, documentation, PR creation.

- [ ] 7.1 Run clippy and fix warnings
  - **Do**: Run `cargo clippy -- -D warnings` on the entire workspace. Fix all warnings in autonomous modules. Ensure no warnings in modified existing files.
  - **Files**: All autonomous source files
  - **Done when**: `cargo clippy -- -D warnings` exits 0
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo clippy -- -D warnings 2>&1 | tail -5`
  - **Commit**: `fix(autonomous): address clippy warnings`

- [ ] 7.2 Run full regression suite
  - **Do**: Run all existing tests plus all new tests. Verify zero failures. Run benchmarks to confirm no regression in existing query paths.
  - **Done when**: All tests pass (existing 807 + new ~200)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo test -- --test-threads=1 2>&1 | grep -E "test result|FAILED"`
  - **Commit**: `fix(autonomous): address any regression issues` (if needed)

- [ ] 7.3 Verify zero `waitUntilCompleted` in autonomous hot path
  - **Do**: Code audit: grep for `waitUntilCompleted` in `src/gpu/autonomous/`. Must return 0 matches (the one-shot test helper is in test code only, not hot path). Grep for `commandBuffer()` calls in `submit_query` path -- must be 0.
  - **Done when**: Zero blocking calls in autonomous hot path
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && grep -rn "waitUntilCompleted" src/gpu/autonomous/ && echo "FAIL: found blocking calls" || echo "PASS: no blocking calls"`
  - **Commit**: No commit needed (verification only)
  - _Requirements: AC-1.4, AC-2.2_

- [ ] 7.4 Update build.rs for new Metal shaders
  - **Do**: Ensure `build.rs` compiles `fused_query.metal` and `check_work.metal` (if used). Verify both new .metal files are included in the metallib. Check that `autonomous_types.h` is accessible via include path.
  - **Files**: `gpu-query/build.rs`
  - **Done when**: `cargo build --release` compiles all shaders
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/gpu-query && cargo build --release 2>&1 | tail -5`
  - **Commit**: `build(autonomous): update build.rs for new Metal shaders`

- [ ] 7.5 Create PR and verify CI
  - **Do**: Push branch, create PR with gh CLI. Title: "feat: GPU-autonomous query engine (<1ms warm latency)". Body includes summary of architecture, benchmark results, test counts, and link to spec. Run `gh pr checks --watch`.
  - **Verify**: `gh pr checks --watch`
  - **Done when**: PR created, all CI checks green
  - **Commit**: No commit (PR creation)

## Notes

- **POC shortcuts taken**: Phase 2 uses `waitUntilCompleted` for one-shot testing of fused kernel correctness before persistent kernel is built. This is intentional -- prove correctness first, then add autonomy.
- **Production TODOs for Phase 2 features**: ICB-based GPU-driven dispatch (reduces gap from 0.1ms to near-zero), GROUP BY >64 groups (two-level hash table), string filter predicates, binary columnar disk cache.
- **Test data**: All tests use deterministic data generators (amount = (i*7+13)%1000, region = i%5) so expected results are pre-computable.
- **Metal validation**: Run stress tests with `METAL_DEVICE_WRAPPER_TYPE=1 MTL_SHADER_VALIDATION=1` for API correctness validation.
- **Thread safety**: All Metal tests must run `--test-threads=1` to avoid GPU device contention.
