---
spec: forge-runtime-p1
phase: research
created: 2026-02-21
generated: auto
---

# Research: forge-runtime-p1

## Executive Summary

ForgeRuntime Phase 1 is highly feasible. The codebase already has all primitives (sort, filter) working independently with identical Metal boilerplate. The refactor extracts shared device/queue/PSO into a common `ForgeContext`, adds a `ForgeBuffer<T>` trait for zero-copy buffer flow, a pipeline builder for single-command-buffer multi-dispatch, and a trivial gather kernel. Estimated ~350 LOC net new code.

## Codebase Analysis

### Existing Patterns

| Pattern | Location | Notes |
|---------|----------|-------|
| `init_device_and_queue()` duplicated | `forge-sort/src/metal_helpers.rs:206`, `forge-filter/src/metal_helpers.rs:206` | Identical code creating `MTLCreateSystemDefaultDevice()` + `newCommandQueue()` |
| `PsoCache` duplicated | Both `metal_helpers.rs` files (~100 LOC each) | HashMap<String, Retained<PSO>>, get_or_create + specialized variants |
| `FnConstant` duplicated | Both `metal_helpers.rs` files | Bool(bool), U32(u32) |
| `alloc_buffer()` duplicated | Both `metal_helpers.rs` + `forge-primitives/src/dispatch.rs` | StorageModeShared allocation |
| `MetalContext` in forge-primitives | `forge-primitives/src/metal_ctx.rs` | Has device + queue + optional library, but not used by forge-sort/filter |
| `PsoCache` in forge-primitives | `forge-primitives/src/pso_cache.rs` | Identical to inlined copies |
| `BufferPool` in forge-primitives | `forge-primitives/src/buffer_pool.rs` | Page-aligned recycling pool, not used by sort/filter yet |
| `encode_sort_pipeline()` | `forge-sort/src/lib.rs:453` | Already takes encoder as param (not CB) |
| `encode_scan_partials()` | `forge-filter/src/lib.rs:555` | Already encodes onto existing encoder |
| `SortBuffer<T>` | `forge-sort/src/lib.rs:160` | `buffer + len + capacity + PhantomData<T>` |
| `FilterBuffer<T>` | `forge-filter/src/lib.rs:2659` | Identical layout: `buffer + len + capacity + PhantomData<T>` |
| `FilterResult<T>` | `forge-filter/src/lib.rs:2741` | `count + values_buf + indices_buf + capacity + PhantomData<T>` |
| `metal_buffer()` accessor | Both SortBuffer and FilterBuffer | Returns `&ProtocolObject<dyn MTLBuffer>` |

### Dependencies

| Dependency | Version | Used By |
|-----------|---------|---------|
| objc2 | 0.6 | All crates |
| objc2-metal | 0.3 | All crates |
| objc2-foundation | 0.3 | All crates |
| thiserror | 2 | forge-sort, forge-filter |

### Constraints

1. **Sealed traits**: Both `SortKey` and `FilterKey` are sealed (private::Sealed). Cannot add impls from outside. `ForgeBuffer<T>` must use a different trait bound or be generic over the concrete buffer types.
2. **Per-crate metallib paths**: Each crate has its own `build.rs` that compiles `.metal` shaders and sets `env!("SORT_METALLIB_PATH")` / `env!("FILTER_METALLIB_PATH")`. The runtime crate needs to load both metallibs or accept pre-loaded libraries.
3. **PsoCache borrows library**: `get_or_create(&mut self, library: &MTLLibrary, ...)` takes library as param. A shared PsoCache works across different libraries already.
4. **Scratch buffer ownership**: GpuSorter owns buf_a/buf_b scratch buffers; GpuFilter owns buf_partials/buf_output. These stay owned by each primitive; only data buffers flow between stages.
5. **objc2 Retained semantics**: `Retained<ProtocolObject<dyn MTLBuffer>>` is the ownership type. References use `&ProtocolObject<dyn MTLBuffer>`. Buffer conversion requires transferring or cloning the Retained wrapper.

### Key Code Path Analysis

**GpuSorter::new()** (lib.rs:561):
- Calls `init_device_and_queue()` - creates device+queue
- Loads metallib from `env!("SORT_METALLIB_PATH")`
- Pre-compiles ~10 PSOs into PsoCache
- Allocates no scratch buffers (lazy)

**GpuFilter::new()** (lib.rs:430):
- Calls `init_device_and_queue()` - creates SEPARATE device+queue
- Loads metallib from `env!("FILTER_METALLIB_PATH")`
- Pre-compiles ~15 PSOs into PsoCache
- Allocates no scratch buffers (lazy)

**dispatch_sort()** (lib.rs:250):
- Creates own command buffer + encoder
- Encodes 4 dispatches
- Commits + waitUntilCompleted

**encode_sort_pipeline()** (lib.rs:453):
- Takes encoder as param (no CB creation)
- Encodes same 4 dispatches
- Does NOT commit -- caller controls lifecycle

This means forge-sort already has the "encode onto external encoder" pattern ready for pipeline integration.

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | All patterns exist; mostly extraction + wiring |
| Effort Estimate | S (small) | ~350 LOC new, ~50 LOC refactor existing |
| Risk Level | Low | No new GPU kernels except trivial gather; existing tests validate primitives |

## Recommendations

1. Place `forge-runtime` crate in workspace; depends on forge-sort + forge-filter as path deps
2. `ForgeContext` holds `device + queue + buffer_pool`; does NOT hold PsoCache (each primitive keeps own PSOs since they load different metallibs)
3. `ForgeBuffer<T>` is a trait with `metal_buffer() + len() + capacity()`; impl for SortBuffer, FilterBuffer, FilterResult
4. Buffer conversions use `from_raw_parts(buffer, len, capacity)` constructors -- zero copy, just wraps the Retained<MTLBuffer>
5. Pipeline builder creates one CB + one encoder, calls `encode_*` methods on primitives, commits once
6. Gather kernel: ~20 LOC Metal, ~80 LOC Rust wrapper. Simplest possible kernel.
