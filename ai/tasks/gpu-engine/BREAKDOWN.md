---
id: gpu-engine.BREAKDOWN
module: gpu-engine
priority: 1
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN]
tags: [gpu-search]
testRequirements:
  unit:
    required: true
---

# GPU Engine Module Breakdown

## Context

Port ~8,000 lines of proven GPU search code from `rust-experiment/` (deprecated `metal 0.33`) to `objc2-metal 0.3`. The core algorithms (MSL shader code, streaming pipeline logic, index structures) remain UNCHANGED. Only Rust-side Metal API calls change. This module delivers the GPU compute backbone: content search kernels (55-80 GB/s), MTLIOCommandQueue batch file I/O, mmap zero-copy buffers, and the GPU-resident filesystem index.

Additionally, adopt gpu-query patterns: triple-buffered work queue, completion-handler re-dispatch chain, MTLSharedEvent for idle/wake power management.

## Tasks

### T-010: Port gpu/device.rs -- Metal device initialization

Port from gpu-query `gpu/device.rs` (already objc2-metal). Provides:
- `MTLCreateSystemDefaultDevice()` initialization
- Device capability queries (max threadgroup size, max buffer length)
- Shared device handle for both compute and UI verification

**Source**: `gpu_kernel/gpu-query/src/gpu/device.rs`
**Target**: `gpu-search/src/gpu/device.rs`
**Verify**: `cargo test -p gpu-search test_device_init` -- device created, name contains "Apple"

### T-011: Port gpu/pipeline.rs -- PSO cache

Port from gpu-query `gpu/pipeline.rs`. HashMap-based PSO cache for search kernels.
- Load `shaders.metallib` from build output
- Create `MTLComputePipelineState` for each kernel function
- Cache by kernel name

**Source**: `gpu_kernel/gpu-query/src/gpu/pipeline.rs`
**Target**: `gpu-search/src/gpu/pipeline.rs`
**Verify**: `cargo test -p gpu-search test_pso_cache` -- PSOs created for all 4 search kernels

### T-012: Extract and port Metal shaders from rust-experiment

Extract MSL shader source from inline Rust strings into standalone `.metal` files:

| Source File | Inline Constant | Target .metal File |
|------------|----------------|-------------------|
| `rust-experiment/src/gpu_os/content_search.rs:95` | `CONTENT_SEARCH_SHADER` | `shaders/content_search.metal` |
| `rust-experiment/src/gpu_os/content_search.rs:276` | turbo_search_kernel | `shaders/turbo_search.metal` |
| `rust-experiment/src/gpu_os/persistent_search.rs:173` | `PERSISTENT_SEARCH_SHADER` | `shaders/batch_search.metal` |
| `rust-experiment/src/gpu_os/filesystem.rs` | path filter MSL | `shaders/path_filter.metal` |

Also create `shaders/search_types.h` with shared `#[repr(C)]` type definitions matching Rust-side structs.

**Verify**: `cargo build -p gpu-search` compiles all shaders without errors. `xcrun -sdk macosx metal -c shaders/content_search.metal` succeeds individually.

### T-013: Create gpu/types.rs -- shared repr(C) types

Define Rust-side `#[repr(C)]` structs matching MSL types in `search_types.h`:

- `SearchParams`: pattern bytes, pattern length, flags
- `SearchResult`: match offset, file index, line number
- `GpuPathEntry`: 256B cache-aligned entry (path, flags, parent_idx, size, mtime)

Include compile-time layout assertions (offset + size checks) per gpu-query pattern.

**Source**: rust-experiment `gpu_os/gpu_index.rs` (GpuPathEntry), `gpu_os/content_search.rs` (SearchParams)
**Target**: `gpu-search/src/gpu/types.rs`
**Verify**: `cargo test -p gpu-search test_types_layout` -- all offset/size assertions pass

### T-014: Port io/mmap.rs -- zero-copy mmap buffers

Port from `rust-experiment/src/gpu_os/mmap_buffer.rs` (566 lines). Only API call changes:
- `Device` -> `ProtocolObject<dyn MTLDevice>`
- `device.new_buffer_with_bytes_no_copy()` -> `device.newBufferWithBytesNoCopy_length_options_deallocator()`
- mmap logic unchanged

**Source**: `rust-experiment/src/gpu_os/mmap_buffer.rs`
**Target**: `gpu-search/src/io/mmap.rs`
**Verify**: `cargo test -p gpu-search test_mmap_buffer` -- mmap file, create Metal buffer, read contents back

### T-015: Port io/gpu_io.rs -- MTLIOCommandQueue

Port from `rust-experiment/src/gpu_os/gpu_io.rs` (524 lines). **Major change**: replace all raw `msg_send!` calls with native objc2-metal `MTLIOCommandQueue` bindings.

Key translation:
```
// OLD: msg_send![class!(MTLIOCommandQueueDescriptor), new]
// NEW: MTLIOCommandQueueDescriptor::new()
```

Enable objc2-metal feature flag `MTLIOCommandQueue`.

**Source**: `rust-experiment/src/gpu_os/gpu_io.rs`
**Target**: `gpu-search/src/io/gpu_io.rs`
**Verify**: `cargo test -p gpu-search test_io_command_queue` -- create IO queue, load file to GPU buffer

### T-016: Port io/batch.rs -- batch file loading

Port from `rust-experiment/src/gpu_os/batch_io.rs` (514 lines). Port `BatchLoadResult` and `BatchLoadHandle` types.

**Source**: `rust-experiment/src/gpu_os/batch_io.rs`
**Target**: `gpu-search/src/io/batch.rs`
**Verify**: `cargo test -p gpu-search test_batch_load` -- load 100 files via MTLIOCommandQueue, verify contents

### T-017: Port search/content.rs -- content search orchestration

Port from `rust-experiment/src/gpu_os/content_search.rs` (1660 lines). Content search dispatch logic:
- Buffer allocation for input/output/match positions
- Kernel dispatch with correct threadgroup sizing (256 threads, 64 bytes/thread)
- Result collection from GPU output buffers
- Both standard and turbo (deferred line numbers) modes

**Source**: `rust-experiment/src/gpu_os/content_search.rs`
**Target**: `gpu-search/src/search/content.rs`
**Verify**: `cargo test -p gpu-search test_content_search` -- search for known pattern in test file, correct match count and positions

### T-018: Port search/streaming.rs -- streaming pipeline

Port from `rust-experiment/src/gpu_os/streaming_search.rs` (1071 lines). Quad-buffered streaming:
- 4 x 64MB StreamChunks
- I/O + search overlap pipeline
- AtomicBool per chunk for ready signaling

**Source**: `rust-experiment/src/gpu_os/streaming_search.rs`
**Target**: `gpu-search/src/search/streaming.rs`
**Verify**: `cargo test -p gpu-search test_streaming_search` -- stream search through >64MB test data, correct results

### T-019: Port index/gpu_index.rs -- GPU-resident path index

Port from `rust-experiment/src/gpu_os/gpu_index.rs` (867 lines). `GpuPathEntry` (256B), `GpuResidentIndex` struct.

**Source**: `rust-experiment/src/gpu_os/gpu_index.rs`
**Target**: `gpu-search/src/index/gpu_index.rs`
**Verify**: `cargo test -p gpu-search test_gpu_index` -- build index from test directory, load into GPU buffer

### T-020: Port index/shared_index.rs -- shared index manager

Port from `rust-experiment/src/gpu_os/shared_index.rs` (921 lines). `GpuFilesystemIndex` struct, cache at `~/.gpu-search/` (changed from `~/.gpu_os/`).

**Source**: `rust-experiment/src/gpu_os/shared_index.rs`
**Target**: `gpu-search/src/index/shared_index.rs`
**Verify**: `cargo test -p gpu-search test_shared_index` -- save index to disk, reload, verify identical

### T-021: Adopt gpu-query work queue pattern

Adapt gpu-query's `WorkQueue` for search requests. Triple-buffered (3 x `SearchRequestSlot`) in StorageModeShared. CPU writes search params, bumps sequence_id with Release ordering.

**Source**: `gpu_kernel/gpu-query/src/gpu/autonomous/work_queue.rs`
**Target**: `gpu-search/src/gpu/work_queue.rs`
**Verify**: `cargo test -p gpu-search test_work_queue` -- write/read cycle, atomic ordering correct

### T-022: Implement SearchExecutor with re-dispatch chain

Build `SearchExecutor` following gpu-query's `executor.rs` pattern:
- Completion-handler re-dispatch chain for persistent search
- MTLSharedEvent for idle/wake
- Separate MTLCommandQueue for search compute

**Source**: `gpu_kernel/gpu-query/src/gpu/autonomous/executor.rs`
**Target**: `gpu-search/src/gpu/executor.rs`
**Verify**: `cargo test -p gpu-search test_executor_lifecycle` -- start, submit search, get results, idle, wake

## Acceptance Criteria

1. All ~8,000 lines of rust-experiment GPU code ported to objc2-metal 0.3
2. MSL shader algorithms UNCHANGED -- extracted into standalone .metal files
3. Content search achieves 55-80 GB/s on 100MB test data (Criterion benchmark)
4. MTLIOCommandQueue loads 100 files without raw `msg_send!` -- all native objc2-metal
5. GPU-CPU dual verification: every search result matches CPU reference implementation
6. `#[repr(C)]` type layout assertions pass for all GPU-shared types
7. Streaming pipeline correctly overlaps I/O with compute (verified by timing)
8. Work queue triple-buffer protocol correct under concurrent access
9. SearchExecutor re-dispatch chain survives 60 seconds without watchdog errors
10. All Metal API calls use objc2-metal 0.3 -- zero raw `msg_send!` calls in codebase

## Technical Notes

- **Port order matters**: Start with `device.rs` and `types.rs` (no dependencies), then `mmap.rs` (simple), then `gpu_io.rs` (complex msg_send! replacement), then `batch.rs`, then search modules that depend on I/O.
- **API translation table**: See TECH.md Section 9.2 for comprehensive metal 0.33 -> objc2-metal 0.3 mapping.
- **MTLIOCommandQueue feature flag**: Requires `objc2-metal` feature `MTLIOCommandQueue`. If specific methods are missing from bindings, fall back to `msg_send!` for those methods only and file upstream issue.
- **Search kernel correctness**: Use GPU-CPU dual verification pattern from QA.md Section 3. Every GPU search test compares against `memchr` or manual CPU search.
- **Cache location change**: rust-experiment uses `~/.gpu_os/`. gpu-search uses `~/.gpu-search/`. Update all path references during port.
- **Streaming chunk size**: 64MB per chunk, 4 chunks. Fits in L2 cache on M4 Max.
- Reference: TECH.md Sections 3-9, QA.md Sections 3-4, PM.md US-1 through US-6
