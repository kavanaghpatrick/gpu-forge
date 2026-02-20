---
id: persistent-executor.BREAKDOWN
module: persistent-executor
priority: 1
status: failing
version: 1
origin: spec-workflow
dependsOn: []
tags: [performance, gpu-query, metal]
testRequirements:
  unit:
    required: true
    pattern: "tests/**/*.rs"
---
# Persistent QueryExecutor in AppState

## Context

Every query execution path in the TUI (`tui/ui.rs:375`, `tui/event.rs:216`) and CLI (`cli/mod.rs:254`) calls `QueryExecutor::new()`, which internally calls `GpuDevice::new()`. This performs:

1. `MTLCreateSystemDefaultDevice()` -- GPU hardware enumeration and driver init (~1-5ms)
2. `GpuDevice::find_metallib()` -- filesystem walk to locate `shaders.metallib` (~5ms)
3. `device.newLibraryWithFile_error()` -- metallib load + Metal compiler front-end (~5-15ms)
4. `PsoCache::new()` -- starts with empty HashMap, forcing recompilation of all function-constant-specialized PSOs on the next query (~10-30ms for 3-4 variants)

Total per-query cold-start tax: ~20-50ms, which is 40-100% of the <50ms target budget. Apple's Metal Best Practices Guide explicitly states: "Create only one MTLDevice object per GPU and reuse it."

The fix moves `QueryExecutor` into `AppState` with lazy initialization. The Metal device, command queue, library, and PSO cache persist across queries for the TUI session lifetime. The CLI one-shot mode remains per-invocation (no state to persist).

## Acceptance Criteria

1. `QueryExecutor` is created exactly once per TUI session and stored in `AppState.executor: Option<QueryExecutor>`
2. `get_or_init_executor()` lazily initializes on first use and returns `&mut QueryExecutor` on subsequent calls
3. PSO cache entries persist across queries -- second query on the same kernel variant shows 0ms PSO compilation (verified by PSO cache size assertion)
4. All 494 lib unit tests + 155 GPU integration tests + 93 E2E golden tests pass unchanged
5. Second-query latency is at least 15ms faster than first-query latency (Metal init + PSO compilation amortized)
6. CLI one-shot mode (`cli/mod.rs`) continues to create a fresh `QueryExecutor` per invocation -- no behavioral change for CLI users
7. Memory overhead of persistent executor is <10MB (Metal device + queue + library + PSO cache)
8. No resource leaks: Metal objects (`Retained<dyn MTLDevice>`, `Retained<dyn MTLComputePipelineState>`) are properly released when `AppState` is dropped
9. Borrow checker is satisfied: `app.get_or_init_executor()` and `app.set_result()` do not create conflicting mutable borrows (use take/replace pattern or split-borrow strategy)

## Technical Notes

- **Reference**: OVERVIEW.md Module Roadmap priority 1; TECH.md Section 5 (Fix B2); PM.md Section 3.2 BUG #2; UX.md Section 4.2
- **Files to modify**:
  - `gpu-query/src/tui/app.rs` -- Add `pub executor: Option<QueryExecutor>` field; add `get_or_init_executor(&mut self) -> Result<&mut QueryExecutor, String>` method; initialize to `None` in `AppState::new()`
  - `gpu-query/src/tui/ui.rs:375` -- Replace `QueryExecutor::new()` with `app.get_or_init_executor()`; handle borrow split between executor use and result storage
  - `gpu-query/src/tui/event.rs:216` -- Same replacement for DESCRIBE path
  - `gpu-query/src/cli/mod.rs:254` -- No change (CLI stays per-invocation)
- **Borrow checker strategy**: The current pattern `let mut executor = QueryExecutor::new()?; let result = executor.execute(...); app.set_result(result);` works because `executor` is a local. With the persistent executor in `app`, use the take/replace pattern: `let mut executor = app.executor.take().unwrap(); ... app.executor = Some(executor);` or restructure to separate the executor borrow from the result-set borrow.
- **Thread safety**: `PsoCache` uses `HashMap` which is `!Sync`. The TUI is single-threaded, so this is safe. Document for future multi-threaded use: wrap in `Mutex` if needed.
- **Test**: `cargo test --all-targets`; new `test_executor_pso_cache_persists` and `test_executor_reuse_correctness` unit tests; `cargo bench --bench query_latency -- "executor_reuse"` for performance measurement
