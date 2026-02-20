---
id: storage.BREAKDOWN
module: storage
priority: 1
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN]
tags: [gpu-autonomous]
testRequirements:
  unit:
    required: false
---
# Storage Module Breakdown

## Context

The autonomous engine requires pre-loaded binary columnar data in GPU-resident Metal buffers for sub-millisecond query execution. Currently, gpu-query parses CSV/Parquet at query time via the scan cache. The autonomous path needs data converted from the existing `ColumnarBatch` format into page-aligned Metal buffers at startup, persisting across all queries. This eliminates the ~15ms CSV parsing overhead that accounts for a significant portion of the current 36ms latency.

This module also defines the shared `#[repr(C)]` types used by both Rust and Metal for the work queue, output buffer, and column metadata -- the "contract" between CPU and GPU.

## Acceptance Criteria

1. `autonomous/types.rs` defines `QueryParamsSlot` (512B), `FilterSpec` (48B), `AggSpec` (16B), `OutputBuffer` (~22KB), `ColumnMeta` (32B), `AggResult` (16B) with `#[repr(C)]` and layout tests verifying exact byte sizes and field offsets -- per TECH.md Section 4.2
2. `shaders/autonomous_types.h` defines MSL counterparts byte-identical to Rust structs -- verified by size assertions in Metal shader compilation
3. `autonomous/loader.rs` loads data from existing `ColumnarBatch` into page-aligned (16KB) Metal buffers (StorageModeShared) -- per TECH.md Section 7.3
4. Binary columnar buffers use separate per-type buffers matching existing `ColumnarBatch` layout (decision TECH-Q5)
5. 1M rows loaded in < 500ms from existing in-memory `ColumnarBatch` (NFR-3)
6. GPU memory for 1M rows < 100MB (NFR-4), target ~33MB for 5 columns
7. Column data starts at 16-byte aligned boundaries within buffers
8. Background loading with progress callback (0.0 -> 1.0) for TUI warm-up display
9. ~50 unit tests pass for struct layouts (size, alignment, offsets, round-trip) following existing `src/gpu/types.rs` pattern
10. ~16 unit tests pass for loader (column types, page alignment, round-trip, empty table, single row, progress callback)

## Technical Notes

- **Struct layout tests**: Follow exact pattern from existing `src/gpu/types.rs` -- use `std::mem::{size_of, align_of}` and `std::mem::offset_of!` macro (Rust 1.82+)
- **Page alignment**: `(total_bytes + 16383) & !16383` for 16KB ARM64 page alignment [KB #89]. Required for potential future `makeBuffer(bytesNoCopy:)` optimization.
- **Column alignment**: Each column starts at 16-byte boundary within buffer for optimal GPU access patterns
- **StorageModeShared**: All buffers use `MTLResourceOptions::StorageModeShared` for unified memory (UMA) -- CPU and GPU share physical memory
- **Separate per-type buffers**: INT64 columns in one buffer, FLOAT32 in another, DICT_U32 in another -- matching existing `ColumnarBatch` pattern for easier migration (TECH-Q5 decision)
- **ColumnMeta**: Stores byte offset, type code (0=INT64, 1=FLOAT32, 2=DICT_U32), stride, null_offset, row_count for each column. Passed to GPU as a Metal buffer.
- **ResidentTable struct**: Holds data_buffer, column_metas, column_meta_buffer, row_count, schema, dictionaries -- per TECH.md Section 8.1
- Reference: PM.md US-5 (pre-loaded binary data), TECH.md Sections 4.2, 7, 8.1-8.2, QA.md Section 4.1-4.4
