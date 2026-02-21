---
spec: forge-runtime-p1
phase: requirements
created: 2026-02-21
generated: auto
---

# Requirements: forge-runtime-p1

## Summary

Build `forge-runtime` crate providing shared GPU context, buffer interop trait, pipeline builder, and gather kernel. Refactor forge-sort/forge-filter to accept injected context while preserving standalone `::new()` APIs.

## User Stories

### US-1: Shared GPU context across primitives

As a developer composing GPU operations, I want a single ForgeContext so that sort and filter share one Metal device/queue instead of creating duplicates.

**Acceptance Criteria**:
- AC-1.1: `ForgeContext::new()` creates device + queue + buffer_pool once
- AC-1.2: `GpuSorter::with_context(&ForgeContext)` uses shared device/queue
- AC-1.3: `GpuFilter::with_context(&ForgeContext)` uses shared device/queue
- AC-1.4: Existing `GpuSorter::new()` and `GpuFilter::new()` still work standalone
- AC-1.5: Both primitives produce identical results whether standalone or with shared context

### US-2: Zero-copy buffer flow between primitives

As a developer chaining filter then sort, I want filter output to flow directly into sort without CPU buffer copies.

**Acceptance Criteria**:
- AC-2.1: `ForgeBuffer<T>` trait exposes `metal_buffer()`, `len()`, `capacity()`
- AC-2.2: `FilterResult<T>` can convert to `SortBuffer<T>` via `From<>` (zero memcpy)
- AC-2.3: `SortBuffer<T>` can convert to `FilterBuffer<T>` via `From<>` (zero memcpy)
- AC-2.4: Conversion transfers Retained<MTLBuffer> ownership, no allocation

### US-3: Single-command-buffer pipeline

As a developer building multi-step GPU pipelines, I want all dispatches in one command buffer so that total overhead is ~1us/dispatch instead of ~97us/CB.

**Acceptance Criteria**:
- AC-3.1: `Pipeline::new(&ForgeContext)` creates builder with one CB + one encoder
- AC-3.2: `pipeline.filter()` encodes filter dispatches onto shared encoder
- AC-3.3: `pipeline.sort()` encodes sort dispatches onto shared encoder
- AC-3.4: `pipeline.gather()` encodes gather dispatch onto shared encoder
- AC-3.5: `pipeline.execute()` commits once and returns result
- AC-3.6: Total overhead for 10-dispatch pipeline < 20us

### US-4: GPU Gather kernel

As a developer, I want a gather operation that takes indices + source buffer and produces a permuted output buffer on GPU without CPU intervention.

**Acceptance Criteria**:
- AC-4.1: `gather.metal` kernel: `dst[gid] = src[indices[gid]]`
- AC-4.2: Rust wrapper encodes gather onto existing encoder
- AC-4.3: Works for u32, i32, f32, u64, i64, f64 element types
- AC-4.4: Correctness: output matches CPU gather for random index permutations

### US-5: Backward compatibility

As a user of forge-sort or forge-filter, I want existing code to compile unchanged after this refactor.

**Acceptance Criteria**:
- AC-5.1: All existing forge-sort tests pass
- AC-5.2: All existing forge-filter tests pass
- AC-5.3: Public API of forge-sort unchanged (SortKey, SortBuffer, GpuSorter, SortError)
- AC-5.4: Public API of forge-filter unchanged (FilterKey, FilterBuffer, FilterResult, GpuFilter, FilterError, Predicate)

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | ForgeContext struct: device + queue + buffer_pool | Must | US-1 |
| FR-2 | GpuSorter::with_context() constructor | Must | US-1 |
| FR-3 | GpuFilter::with_context() constructor | Must | US-1 |
| FR-4 | ForgeBuffer<T> trait: metal_buffer + len + capacity | Must | US-2 |
| FR-5 | From<FilterResult<T>> for SortBuffer<T> | Must | US-2 |
| FR-6 | From<SortBuffer<T>> for FilterBuffer<T> | Must | US-2 |
| FR-7 | Pipeline builder: new, filter, sort, gather, execute | Must | US-3 |
| FR-8 | Single CB + single encoder for all pipeline stages | Must | US-3 |
| FR-9 | gather.metal kernel (32-bit + 64-bit variants via function constants) | Must | US-4 |
| FR-10 | Gather Rust wrapper with encode_gather() | Must | US-4 |
| FR-11 | Standalone ::new() preserved on GpuSorter and GpuFilter | Must | US-5 |
| FR-12 | from_raw_parts() constructors on SortBuffer and FilterBuffer | Should | US-2 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Zero CPU buffer copies between pipeline stages (UMA zero-copy) | Performance |
| NFR-2 | < 20us total dispatch overhead for 10-dispatch pipeline | Performance |
| NFR-3 | ~350 LOC total net new code | Scope |
| NFR-4 | No new external dependencies beyond objc2 ecosystem | Dependencies |

## Out of Scope

- Megakernel interpreter (Phase 2)
- Arrow integration
- Multi-command-buffer chaining for long pipelines
- Hash table integration (GpuHashTable)
- Benchmark crate changes
- Pipeline optimizer / instruction compiler

## Dependencies

- forge-sort (path dependency)
- forge-filter (path dependency)
- objc2 0.6, objc2-metal 0.3, objc2-foundation 0.3
- thiserror 2
