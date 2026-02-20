---
spec: gpu-search-overhaul-3
phase: research
created: 2026-02-16
generated: auto
---

# Research: gpu-search-overhaul-3

## Executive Summary

gpu-search is 25-34x slower than GPUripgrep despite using nearly identical Metal kernels. Three root causes identified through empirical profiling. All three are fixable with targeted code changes -- no architectural redesign needed.

## Codebase Analysis

### Root Cause 1: 100K Chunk Cap (content.rs:177)

```rust
let max_chunks = (max_files * 10).min(100_000); // Cap at 100K chunks (~400MB)
```

- 6.39M chunks / 100K = 64 serial GPU dispatches
- Each dispatch: commit() + waitUntilCompleted() = ~100ms overhead
- Serial batching in `search_zerocopy()` (content.rs:591): `for batch_start in (0..chunk_metas.len()).step_by(batch_size)`
- GPUripgrep dispatches ALL data in one call = 1 commit + 1 wait
- Throughput: 3.2 GB/s (64 batches) vs 79-110 GB/s (1 batch)

**Key insight**: Metal can dispatch millions of threadgroups. The cap is artificial. The metadata buffer is the bottleneck -- it's sized to `max_chunks * 32 bytes`. For 6.39M chunks we need ~204MB metadata buffer which is fine on M4 Pro (36GB unified memory).

### Root Cause 2: CPU Line Number Resolution (orchestrator.rs:1491-1581)

```rust
let text = String::from_utf8_lossy(file_content);  // alloc + copy
let lines: Vec<&str> = text.lines().collect();       // alloc + split
// then linear scan for target_line...
```

Per match: `String::from_utf8_lossy(entire_file)` + `text.lines().collect()` = ~1.9ms/match. For 1880 matches = 3.6s total.

The GPU kernel already counts newlines within its 64-byte window (`content_search_zerocopy_kernel`, line 90-97). But this only counts within the local thread window, not from file start.

**Fix**: Count newlines from `buffer_offset` (file start in contiguous buffer) to `buffer_offset + offset_in_chunk + local_pos`. The kernel has access to `raw_data` and `meta.buffer_offset` -- just scan backwards to file start.

### Root Cause 3: Chunk Metadata Rebuild (content_store.rs:916-954)

```rust
pub fn build_chunk_metadata(store: &ContentStore) -> Vec<ChunkMetadata> {
    // Iterates all files, divides each into 4KB chunks, builds Vec
}
```

Called on every keystroke via `orchestrator.rs:1397`. 6.39M chunks * 32 bytes = ~204MB allocation + computation. Takes ~20ms per call. Completely deterministic given the same ContentStore -- should be computed once at index time and persisted.

GCIX v2 format (gcix.rs) already stores header + meta table + content + path table. Adding a chunk metadata table after the path table is straightforward -- just need new offset/bytes fields in the header.

### Dependencies

- `objc2-metal`: Metal buffer allocation, command queue, PSO dispatch
- `crossbeam_channel`: Streaming pipeline channels (not affected)
- GCIX format: Binary format with versioned header, CRC32 integrity
- `search_types.h`: Shared GPU/CPU struct definitions for MatchResult, ChunkMetadata

### Constraints

- GpuMatchResult and Metal MatchResult must stay in sync (32 bytes each)
- ChunkMetadata is 32 bytes repr(C) -- adding fields would break layout
- GCIX version bump needed (v2 -> v3) for backward compat detection
- Metal max buffer size: 256GB on Apple Silicon (no practical limit)
- MAX_MATCHES constant (10000) -- may need increase for large codebases

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | All changes are straightforward Rust + Metal |
| Effort Estimate | M | ~5-10 LOC for Fix 1, ~50+90 LOC for Fix 2, ~30-50 LOC for Fix 3 |
| Risk Level | Low | Fixes are independent; can validate each separately |

## Recommendations

1. Fix 1 (chunk cap) first -- largest impact, smallest change
2. Fix 2 (GPU line numbers) second -- eliminates CPU bottleneck
3. Fix 3 (pre-built metadata) third -- eliminates per-keystroke overhead
4. Add benchmark task to validate end-to-end <200ms target
