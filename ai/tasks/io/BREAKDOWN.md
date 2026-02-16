---
id: io.BREAKDOWN
module: io
priority: 1
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN]
tags: [gpu-query]
testRequirements:
  unit:
    required: false
    pattern: "tests/io/**/*.test.*"
---
# I/O -- BREAKDOWN

## Context

The I/O layer is the foundation of gpu-query's performance advantage. On Apple Silicon, mmap'd file data resides in unified memory accessible by both CPU and GPU without any DMA transfer. The zero-copy pattern (mmap + `makeBuffer(bytesNoCopy:)`) wraps file-backed pages as Metal buffers with 0.0004ms overhead -- a ~1,464,600x advantage over CUDA's PCIe transfer for mixed access patterns [KB #248]. This module implements the file discovery, mmap wrapping, page alignment, pre-warming, and fallback paths described in TECH.md Section 2.

## Scope

### Rust Files to Create/Implement

- `gpu-query/src/io/mod.rs` -- Module root, public API exports
- `gpu-query/src/io/mmap.rs` -- `MmapFile` struct: open() with mmap + PROT_READ + MAP_SHARED, 16KB page alignment [KB #89, #223], `makeBuffer(bytesNoCopy:)` wrapping, `madvise(MADV_WILLNEED, MADV_SEQUENTIAL)` pre-warming [KB #226], Drop impl for munmap, fallback path (copy to Metal shared buffer)
- `gpu-query/src/io/format_detect.rs` -- File format detection: magic bytes for Parquet (PAR1), JSON ('{' or '['), CSV (heuristic: delimiter frequency analysis), file extension fallback
- `gpu-query/src/io/catalog.rs` -- Directory scanner: recursive file discovery, table registry (path -> detected format + schema), auto-naming (filename without extension as table name)
- `gpu-query/src/io/csv.rs` -- CSV metadata reader: header parsing, delimiter detection (comma/tab/pipe), row count estimation (file_size / avg_row_length)
- `gpu-query/src/io/json.rs` -- JSON/NDJSON metadata reader: detect NDJSON vs regular JSON, sample first N records for schema inference
- `gpu-query/src/io/parquet.rs` -- Parquet metadata reader: read footer via `parquet` crate, extract schema, row group info, column chunk offsets for selective mmap

### Tests to Write

- mmap page alignment: verify pointer is 16KB aligned on ARM64 [KB #89]
- mmap length rounding: verify aligned_len is multiple of page size
- Zero-copy buffer contents match: mmap bytes == Metal buffer bytes (same physical pages)
- Format detection: correct identification of CSV, JSON, Parquet from magic bytes
- Fallback path: copy-to-shared-buffer works when bytesNoCopy is disabled
- Directory scanning: discovers all supported file types recursively
- Parquet metadata: reads schema and row group info correctly

## Acceptance Criteria

1. `MmapFile::open()` successfully mmap's a file and wraps it as a Metal buffer with `bytesNoCopy`
2. mmap pointer is always 16KB page-aligned (assert in debug builds)
3. `madvise(MADV_WILLNEED)` is called on file open for pre-warming
4. Fallback to copy-based Metal buffer works when `force_buffer_copy` flag is set
5. Format detection correctly identifies Parquet (PAR1 magic), CSV, and JSON files
6. Directory scanner discovers and catalogs all supported file types with row count estimates
7. All unit tests pass: `cargo test -p gpu-query --lib io`

## References

- PM: Section 5 (Technical Feasibility -- zero-copy file access), Section 9 Decision 4 (File Access)
- UX: Section 2.1 (First Contact -- auto-scan), Section 9.2 (GPU-specific errors -- mmap failures)
- TECH: Section 2 (Zero-Copy I/O Pipeline -- full implementation with code), Section 6 (Memory Management)
- QA: Section 8.4 (mmap failure testing), Section 8.5 (zero-copy buffer validation)
- KB: #73 (storage modes), #89 (page alignment), #102 (UMA bandwidth), #223 (bytesNoCopy requirements), #224 (mmap+bytesNoCopy), #226 (madvise), #248 (zero-copy overhead), #250 (memory-to-storage gap), #595 (TLB optimization)

## Technical Notes

- Pointer MUST be 16KB page-aligned (ARM64 hardware page size); only mmap, vm_allocate, or posix_memalign provide suitable alignment -- standard malloc CANNOT be used
- `makeBuffer(bytesNoCopy:)` requires `.storageModeShared` and the deallocator should be None (munmap handled by Rust Drop)
- TLB optimization: reuse mmap regions across queries (don't munmap between queries) to avoid TLB shootdowns [KB #595] -- 28% improvement potential
- Parquet metadata is read on CPU via the `parquet` crate; only column data bytes are mmap'd for GPU access
- The MmapFile must track both the original file_size (for accurate byte counts) and the aligned_len (for Metal buffer)
