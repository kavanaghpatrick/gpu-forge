---
id: mmap-gpu-pipeline.BREAKDOWN
module: mmap-gpu-pipeline
priority: 1
status: pending
version: 1
origin: foreman-spec
dependsOn: [gsix-v2-format.BREAKDOWN]
tags: [persistent-index, gpu-search, zero-copy, metal]
---
# Mmap GPU Pipeline — BREAKDOWN

## Context
The current index loading path reads the GSIX file into a `Vec<GpuPathEntry>`, then copies it into a Metal GPU buffer via `newBufferWithBytes`. This double-copy wastes time and memory. With the GSIX v2 format placing entries at a 16KB page-aligned offset, we can mmap the file and pass the pointer directly to Metal's `makeBuffer(bytesNoCopy:)` for true zero-copy GPU access. This module implements the mmap-to-GPU pipeline, the `IndexSnapshot` struct, and the `arc-swap`-based `IndexStore` for lock-free concurrent reader/writer access.

References: TECH.md Section 1.2 (Data Flow), Section 6 (Page Alignment), Section 7 (Concurrency), PM.md Section 4 (mmap-based instant load, bytesNoCopy).

## Tasks
1. **mmap-gpu-pipeline.update-mmap-cache** -- Update `MmapIndexCache::load_mmap()` in `src/index/cache.rs` to handle v2 headers: validate header at offset [0..16384), read `entry_count` and other v2 fields, calculate entry region at offset 16384. Return error for v1 format (signals rebuild). (est: 2h)

2. **mmap-gpu-pipeline.bytesnocopy-buffer** -- Implement Metal `bytesNoCopy` buffer creation using Strategy A (full-file mmap with kernel offset). After mmap, call `device.makeBuffer(bytesNoCopy:length:options:deallocator:)` on the full mmap region. Use `encoder.setBuffer(buffer, offset: HEADER_SIZE_V2, index: 0)` to skip the header in GPU dispatches. Add fallback to `newBufferWithBytes` if `bytesNoCopy` returns `None`. (est: 3h)

3. **mmap-gpu-pipeline.index-snapshot** -- Define `IndexSnapshot` struct containing: `mmap: MmapBuffer`, `metal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>`, `entry_count: usize`, `fsevents_id: u64`. The mmap lifetime anchors the Metal buffer's memory. Implement `Drop` to ensure proper cleanup order. (est: 2h)

4. **mmap-gpu-pipeline.index-store** -- Implement `IndexStore` using `arc_swap::ArcSwap<IndexSnapshot>`. Provide `snapshot()` for lock-free reader access (returns `Guard<Arc<IndexSnapshot>>`) and `swap()` for writer updates. Old snapshots stay alive until all reader guards are dropped. (est: 2h)

5. **mmap-gpu-pipeline.arc-swap-dependency** -- Add `arc-swap = "1"` to `Cargo.toml`. Wire `IndexStore` into the application lifecycle: create on startup, pass `Arc<IndexStore>` to search pipeline and index writer. (est: 1h)

6. **mmap-gpu-pipeline.eliminate-copy-path** -- Modify `GpuIndexLoader::try_load_from_cache()` to use the new mmap->bytesNoCopy path instead of the `into_resident_index()` copy. The `GpuLoadedIndex` struct should hold the `IndexSnapshot` (which owns the mmap) rather than a separate `Vec<GpuPathEntry>`. (est: 2h)

7. **mmap-gpu-pipeline.try-index-producer** -- Integrate with the existing `try_index_producer` in the search pipeline. When the `IndexStore` has a valid snapshot, use it directly for GPU dispatch instead of walking the filesystem. Fall back to `walk_and_filter()` when no snapshot is available. (est: 2h)

8. **mmap-gpu-pipeline.alignment-verification** -- Add compile-time assertions: `HEADER_SIZE_V2 == 16384`, `HEADER_SIZE_V2 % 16384 == 0`, `size_of::<GpuPathEntry>() == 256`. Add runtime assertion in the load path: `(mmap_ptr + HEADER_SIZE_V2) as usize % PAGE_SIZE == 0`. (est: 1h)

9. **mmap-gpu-pipeline.unit-tests** -- Write tests: mmap of v2 index file is page-aligned, `bytesNoCopy` succeeds on page-aligned mmap, Metal buffer contents match mmap data byte-for-byte, buffer length covers all entries, `IndexStore::snapshot()` returns consistent data, `IndexStore::swap()` updates visible to new readers while old readers retain old data. (~10 tests) (est: 3h)

10. **mmap-gpu-pipeline.lifetime-safety-tests** -- Write tests verifying: mmap is not dropped while Metal buffer exists (`_mmap` field in snapshot keeps it alive), `IndexSnapshot::drop()` releases mmap after Metal buffer, concurrent readers during swap see consistent snapshots, multiple readers + one writer with no data races. (~5 tests) (est: 2h)

## Dependencies
- Requires: [gsix-v2-format (v2 header layout, 16KB page alignment, entry offset at 16384)]
- Enables: [incremental-updates (needs IndexStore.swap() for atomic snapshot replacement), global-root-index (needs IndexStore for orchestrator integration), integration (end-to-end pipeline)]

## Acceptance Criteria
1. mmap load of a 1M-entry v2 index file completes in <5ms (no data copy, just page table setup)
2. Metal `bytesNoCopy` buffer creation succeeds on Apple Silicon (pointer is page-aligned)
3. `IndexStore::snapshot()` is wait-free with no locks (verified by concurrent reader test)
4. Old snapshots remain valid after `IndexStore::swap()` until all reader guards are dropped
5. The `into_resident_index()` copy path is no longer used in the main search flow
6. Fallback to `newBufferWithBytes` works correctly when `bytesNoCopy` is unavailable
7. All existing search tests pass with the new mmap-backed index path

## Technical References
- PM: ai/tasks/spec/PM.md -- Section 4 (mmap-based instant load, Metal bytesNoCopy zero-copy buffer)
- UX: ai/tasks/spec/UX.md -- Section 6 (Startup Experience: instant ready state from mmap)
- Tech: ai/tasks/spec/TECH.md -- Section 6 (Page Alignment: strategies A vs B, alignment verification), Section 7 (Concurrency: ArcSwap snapshot pattern)
- QA: ai/tasks/spec/QA.md -- Section 5.1 (Index Load Time benchmarks), Section 9 (GPU Pipeline Tests: bytesNoCopy creation, entry count validation)
