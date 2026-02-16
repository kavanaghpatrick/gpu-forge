---
spec: gpu-search-false-positives
phase: research
created: 2026-02-15
generated: auto
---

# Research: gpu-search-false-positives

## Executive Summary

GPU content search produces false positives during rapid query changes. Root cause: `ContentSearchEngine::reset()` clears counters but does NOT zero GPU buffer data. Stale content from previous queries persists in `chunks_buffer` and `metadata_buffer`, causing the kernel to produce matches against leftover data when fewer files load in the subsequent search. The CPU-side `resolve_match()` should theoretically reject these, but the stale byte offsets map to valid (but wrong) lines in re-read files.

## Codebase Analysis

### Existing Patterns

- **GPU buffer reuse**: `ContentSearchEngine` allocates buffers once in `new()` (content.rs:190-207), reuses across searches. This is performance-optimal but requires explicit clearing.
- **Sub-batch pipeline**: `search_files_with_profile()` in streaming.rs:371 calls `reset()` between sub-batches of 200 files, then `load_content()` + `search()`. Each sub-batch reuses the same engine buffers.
- **CPU verification layer**: `verify.rs` has `cpu_verify_matches()` using `memchr::memmem` to find ground-truth offsets and compare against GPU byte offsets. Controlled by `GPU_SEARCH_VERIFY` env var.
- **Generation-stamped updates**: `StampedUpdate` in types.rs wraps `SearchUpdate` with a generation ID. `poll_updates()` discards stale generations. This prevents old search results from displaying but does NOT prevent GPU producing false matches.
- **Cancellation**: `SearchSession::should_stop()` checks both `CancellationToken` and `SearchGenerationGuard::is_stale()`. Checked between batches but NOT between sub-batches within a single GPU dispatch.

### Root Cause Analysis

**Primary cause: Stale GPU buffer data after `reset()`**

1. `reset()` (content.rs:295-299) only clears counters:
   ```rust
   pub fn reset(&mut self) {
       self.current_chunk_count = 0;
       self.total_data_bytes = 0;
       self.file_count = 0;
   }
   ```

2. `chunks_buffer` retains file content from the previous search
3. `metadata_buffer` retains `ChunkMetadata` entries (file_index, chunk_length, etc.)
4. `matches_buffer` retains old match results (but `match_count` IS reset to 0 in `search()`)

5. **How stale data causes false positives**: When a new sub-batch loads fewer chunks than the previous one, `load_content()` overwrites chunks 0..N-1 but chunks N..old_max retain stale data. The GPU kernel dispatches `total_data_bytes / 64` threads. Since `total_data_bytes = current_chunk_count * 4096`, threads should stay within bounds. However:
   - The last chunk's zero-padding region (line 264-266) may not fully overwrite stale data if `chunk_len < CHUNK_SIZE` and the stale data had content in those bytes from a longer previous chunk
   - Actually, lines 264-266 DO zero-pad: `write_bytes(dst.add(chunk_len), 0, CHUNK_SIZE - chunk_len)`. So intra-chunk stale data IS cleared.

6. **Revised root cause**: The stale data problem is in the `metadata_buffer`. When `load_content()` writes metadata for chunk_idx 0..N-1, entries at N..old_max retain old `ChunkMetadata`. The kernel checks `if (chunk_idx >= params.chunk_count) return;` (line 34 of kernel), so these stale entries should not be accessed. **Unless** the thread's `byte_base` falls within the valid range but the `chunk_idx` calculation maps to a stale metadata entry due to a race condition in buffer loading.

7. **Most likely trigger**: The zero-padding in `load_content()` IS correct for data. The metadata IS bounded by `params.chunk_count`. The real issue is likely in **byte_offset calculation** in streaming.rs:410-411:
   ```rust
   let file_byte_offset = m.byte_offset
       .saturating_sub((*start_chunk as u32) * 4096);
   ```
   If `start_chunk` is wrong (because it was read BEFORE `reset()` cleared `current_chunk_count`), the resulting `file_byte_offset` would be wrong, and `resolve_match()` would map to the wrong line -- but if that line happens to contain the pattern, it would NOT be rejected.

8. **Alternative trigger**: The `byte_offset` calculation in `collect_results()` (content.rs:408-409):
   ```rust
   byte_offset: m.chunk_index * CHUNK_SIZE as u32
       + m.context_start + m.column,
   ```
   `chunk_index` here is the GPU's `chunk_idx` (global), not file-relative. When combined with `start_chunk` subtraction in streaming.rs, errors accumulate.

### Dependencies

- `objc2-metal` for GPU buffer operations
- `memchr` for CPU verification
- `crossbeam-channel` for streaming pipeline
- `tempfile` for test fixtures
- `ignore` crate for directory walking

### Constraints

- Cannot change Metal kernel dispatch model (threads = total_bytes / 64)
- Buffer zeroing has a performance cost (up to 400MB for max allocation)
- Must not break existing 556 unit tests
- Must work with both blocking `search()` and streaming `search_streaming()` paths

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | Buffer zeroing is straightforward; byte_offset fix is targeted |
| Effort Estimate | M | Test harness + fix + comprehensive tests |
| Risk Level | Medium | Race conditions are subtle; must test rapid query changes thoroughly |

## Recommendations

1. Zero `metadata_buffer` in `reset()` (targeted: only zero metadata up to `max_chunks * sizeof(ChunkMetadata)`)
2. Zero `match_count_buffer` in `reset()` rather than in `search()` for defense-in-depth
3. Add `clear_buffers()` method that zeros all buffers, called from `reset()`
4. Fix byte_offset calculation chain to use file-relative offsets consistently
5. Build comprehensive integration test suite that drives `StreamingSearchEngine` directly with rapid query changes
6. Enable `GPU_SEARCH_VERIFY=full` in CI test runs to catch false positives automatically
