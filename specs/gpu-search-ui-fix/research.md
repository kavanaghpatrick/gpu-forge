---
spec: gpu-search-ui-fix
phase: research
created: 2026-02-17
generated: auto
---

# Research: gpu-search-ui-fix

## Executive Summary

Three bugs in gpu-search-ui filename search and one missing capability. All are fixable with targeted changes to `shader.rs`, `search.rs`, and `app.rs`. GPU kernel modification (Bug 1) eliminates 70-79% of search latency. Bug 2 is likely a non-issue after code review (paths already preserved). Bug 3 requires new persistence module. Test gap is straightforward to close.

## Codebase Analysis

### Architecture

| Component | File | Role |
|-----------|------|------|
| MSL kernels | `/gpu-search-ui/src/engine/shader.rs` | 4 Metal compute kernels (~540 lines MSL) |
| Search engine | `/gpu-search-ui/src/engine/search.rs` | `GpuContentSearch` — load, search, extract |
| App + threads | `/gpu-search-ui/src/app.rs` | `index_thread` walks FS, `search_thread` dispatches GPU |
| Channel bridge | `/gpu-search-ui/src/bridge.rs` | crossbeam bounded channels between threads |
| File indexer | `/gpu-search-ui/src/engine/index.rs` | `FileIndex::scan_directory()` — not used by app.rs |

### Existing Patterns

- **Chunk packing**: `load_paths()` in search.rs packs paths newline-delimited into 4KB chunks, pads with `\n`
- **Turbo kernel**: `turbo_search_kernel` writes `(chunk_index, byte_offset_in_chunk)` to MatchResult; CPU scans backward/forward to `\n`
- **Content kernel**: `content_search_kernel` already does GPU-side newline scanning for context — proves the approach works
- **Buffer sizing**: `total_path_bytes / 4096 + 2048` (fixed formula in app.rs)
- **Test patterns**: GPU engine tests in `test_path_search.rs` use `GpuContentSearch::new_for_paths()` directly

### Bug 1 Analysis: CPU context extraction bottleneck

**Profiling data** (1M paths, 102.9 MB):
| Query | GPU-only (ms) | Total (ms) | CPU extract % |
|-------|--------------|------------|---------------|
| friendship (50K results) | 3.08 | 10.45 | 70% |
| file (50K results) | 2.14 | 7.90 | 73% |
| a (50K results) | 1.58 | 7.63 | 79% |

The `turbo_search_kernel` writes raw byte offsets. CPU in `search_paths()` lines 401-412 scans backward/forward to `\n` per match. With 50K matches this is 5-8ms of CPU work.

**Fix approach**: Modify turbo_search_kernel to scan backward to `\n` (or chunk start) and forward to `\n` (or chunk end) per match, writing `context_start` (absolute offset from chunk start) and `context_len`. CPU then does `&chunk_data[abs_offset..abs_offset+len]` — O(1) per match. The `content_search_kernel` already demonstrates this pattern (lines 178-201 of shader.rs).

**Risk**: Thread's local_data is only 64+64 bytes. Backward scan must use device memory (chunks_buffer), not local_data. This is fine — device reads are fast for scattered small scans on Apple Silicon unified memory.

### Bug 2 Analysis: Paths without absolute prefix

After careful code review, `load_paths()` does: `p.to_string_lossy()` → bytes → chunk_data. `PathBuf::to_string_lossy()` preserves the full path including leading `/`. The backward scan in `search_paths()` starts at `abs_offset` and scans through `chunk_data` (which is contiguous across all chunks). The only risk is if `abs_offset` calculation `chunk_idx * CHUNK_SIZE + byte_offset_in_chunk` overflows or if `chunk_data[line_start - 1] != b'\n'` terminates too early.

**Hypothesis**: The backward scan `while line_start > 0 && self.chunk_data[line_start - 1] != b'\n'` scans ALL of chunk_data backward, which works fine since chunk_data is contiguous. Paths should be preserved. Need a targeted test to verify.

### Bug 3 Analysis: Index rebuilds from scratch

`index_thread()` in app.rs uses `ignore::WalkBuilder` to walk from root on every launch. At 4.7M files this takes seconds.

**Fix approach**: Persist `Vec<PathBuf>` to `~/.cache/gpu-search-ui/paths.bin` with metadata (root dir, file count, timestamp). On launch, load cache if fresh (< 1 hour or same file count), else rescan. Format: header + length-prefixed path strings.

**Dependencies**: Only needs `std::fs` — no new crates.

### Test Gap Analysis

Existing tests in `test_path_search.rs` cover:
- Basic load + search (5 paths, 1K paths, 200K, 1M)
- Multi-word search
- Real filesystem search

Missing:
- Chunk boundary path verification
- Very long paths (> 200 bytes)
- Unicode paths
- Full path preservation (absolute prefix)
- Context extraction completeness (no truncation)
- Case sensitivity toggle
- Performance regression gate
- Index persistence round-trip

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | content_search_kernel already does GPU-side newline scan |
| Effort Estimate | M | 3 files changed, ~200 lines new code, ~50 lines modified |
| Risk Level | Low | All changes are isolated; existing API preserved |

## Recommendations

1. Fix Bug 1 first — highest performance impact (70-79% CPU time eliminated)
2. Bug 2 may be a non-issue — verify with targeted test before changing code
3. Bug 3 is independent — can be done in parallel
4. Add test suite early to catch regressions during Bug 1 kernel changes
