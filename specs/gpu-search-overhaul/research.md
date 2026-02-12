---
spec: gpu-search-overhaul
phase: research
created: 2026-02-12
generated: auto
---

# Research: gpu-search-overhaul

## Executive Summary

gpu-search has a working GPU-accelerated search pipeline (79-110 GB/s on M4 Pro) but suffers from a P0 accuracy bug where `resolve_match()` maps GPU byte offsets to wrong files in multi-file batches, producing false positives. Secondary issues: 33s time-to-first-result from `/` (99.8% I/O), unwired filesystem index (GSIX format exists but orchestrator never calls it), missing pipeline profiler, and critical test gaps. KB research confirms read() into pre-allocated MTLBuffers beats mmap for multi-file scanning (KB #1343), quad-buffered 64MB chunks saturate NVMe (KB #1274/#1374), and deferred line counting avoids Metal's prefix-sum limitations (KB #1320/#1322).

## Codebase Analysis

### Existing Architecture

| Component | File | Status |
|-----------|------|--------|
| Search orchestrator | `src/search/orchestrator.rs` | Working, houses P0 bug in `resolve_match()` |
| GPU content engine | `src/search/content.rs` | Working, `collect_results()` has byte_offset ambiguity |
| Streaming pipeline | `src/search/streaming.rs` | Quad-buffered, sequential I/O (POC) |
| Metal executor | `src/gpu/executor.rs` | MTLSharedEvent idle/wake, re-dispatch chain |
| Content search kernel | `shaders/content_search.metal` | Correct byte matching, line counting in 64B window |
| Turbo search kernel | `shaders/turbo_search.metal` | Deferred line counting to CPU |
| GSIX index cache | `src/index/cache.rs` | mmap-based zero-copy, 64B header + 256B entries |
| Index scanner | `src/index/scanner.rs` | Parallel scan to `Vec<GpuPathEntry>` |
| Shared index manager | `src/index/shared_index.rs` | Atomic write (temp+rename), staleness check |
| mmap buffer | `src/io/mmap.rs` | 16KB alignment, madvise, Metal bytesNoCopy |

### P0 Bug Root Cause Chain

1. **GPU kernel** (correct): Writes `context_start = offset_in_chunk + line_start_in_64B_window` -- the chunk-relative byte position of the line start. But `line_start` only reflects newlines within the local 64-byte thread window, not from chunk start.

2. **`collect_results()`** (content.rs:404-411): Computes `byte_offset = chunk_index * 4096 + context_start`. Uses engine-global `chunk_index` from GPU, producing an engine-global byte offset.

3. **`search_files_with_profile()`** (streaming.rs:410-411): Converts to file-relative via `saturating_sub(start_chunk * 4096)`. Arithmetic is correct IF `context_start` is accurate -- but it isn't when matches occur deep in a chunk (>64B from chunk start).

4. **`resolve_match()`** (orchestrator.rs:734-792): Re-reads file, walks to `byte_offset` line. If wrong line, `find(pattern)` returns `None` and match is rejected. This safety net catches most false positives but also discards valid matches whose byte offsets are slightly wrong.

### Dependencies (Existing)

| Crate | Version | Usage |
|-------|---------|-------|
| `memchr` | 2 | SIMD byte search -- ideal for CPU verification |
| `crossbeam-channel` | 0.5 | Pipeline channels |
| `ignore` | 0.4 | Walk + gitignore |
| `objc2-metal` | 0.3 | Metal API bindings |
| `tempfile` | 3 (dev) | Test fixtures |
| `criterion` | 0.5 (dev) | Benchmarks |
| `proptest` | 1 (dev) | Property-based testing |

### Dependencies (New Required)

| Crate | Version | Purpose |
|-------|---------|---------|
| `notify` | 7 | FSEvents-based file watcher |
| `notify-debouncer-mini` | 0.5 | 500ms event batching |

## KB Findings Analysis

### I/O Strategy (KB #1343, #1274, #1374, #1377)

- **read() beats mmap for multi-file scanning**: mmap kernel overhead on macOS (ripgrep disables it), per-file mapping cost. read() into fixed 64MB MTLBuffer pool allows explicit scheduling.
- **mmap IS correct for index file**: Single large file loaded once, random access pattern.
- **Quad buffering absorbs SSD variability**: 4x64MB = 256MB (3.1% of 8GB min). GPU processes 64MB in ~0.6ms, SSD delivers in ~22ms.
- **SSD bottleneck on M4 base**: ~2.9 GB/s throughput cap regardless of GPU capability.

### GPU String Matching (KB #1329, #1263, #1251)

- **Bitap for patterns <=32 chars**: NFA state in single `uint32`, compute-bound shift+OR. Ideal for common search patterns.
- **HybridSA**: 4-233x speedup over prior GPU engines (KB #1263).
- **PFAC**: One-thread-per-byte for multi-pattern future enhancement (KB #1251).

### Line Counting (KB #1322, #1320)

- **Deferred line counting**: GPU records byte offsets only, CPU extracts line numbers. Avoids Metal's inability to do decoupled lookback prefix sum.
- `turbo_search.metal` already implements this pattern -- should become the default path.

### GPU Bloom Filters for Gitignore (KB #1363, #1386)

- 50 patterns = 63-byte Bloom filter fits GPU register file. 11.35x speedup.
- Double hashing: `h_i = h1 + i*h2`, only 2 hash computations.

### I/O-Compute Overlap (KB #1337, #1135)

- Separate `MTLIOCommandQueue` + `MTLCommandQueue` + `MTLSharedEvent` for automatic overlap.
- I/O fills buffer N while GPU processes buffer N-1 on Apple Silicon unified memory.

### Index Design (KB #1388, #1328, #1376)

- **256B AoS entries**: 224B inline paths (>99% fit inline). Already implemented in GSIX format.
- **SoA auxiliary arrays**: After main entries for optimized field-level GPU scans.
- **Two-phase GPU filtering**: Phase 1 scans SoA (coalesced), Phase 2 loads full 256B for candidates.

### Hot/Cold Dispatch (KB #1377, #1290, #1298)

- GPU wins over CPU for files >128KB. Unified memory drops crossover dramatically.
- Cold data (not in page cache): CPU NEON search is near SSD bandwidth -- GPU adds no benefit.
- Hot data (page cache): GPU exploits full 120+ GB/s bandwidth.

## Benchmark Baseline

| Scope | Files | Streaming Time | Blocking Time |
|-------|-------|----------------|---------------|
| src/ | 39 | 1-15ms | -- |
| project/ | 66 | 3-23ms | 524-1052ms |
| home/ | 193K | 29-31s | -- |
| root/ | 700K+ | 33.9s | -- |
| GPU compute | -- | <50ms total | -- |
| Cancellation | -- | 105ms | -- |

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| P0 Fix (accuracy) | **High** | Bug is in CPU-side byte_offset interpretation, not kernel. memchr verification + fix = 1-2 days |
| Pipeline Profiler | **High** | Instant-based timing, no new deps. PipelineProfile struct = 1 day |
| Index Wiring | **High** | GSIX format + MmapIndexCache + SharedIndexManager all exist. Channel-swap producer = 2-3 days |
| notify v7 Watcher | **Medium** | FSEvents well-supported but edge cases (rename, rapid changes). Debouncer mitigates = 2-3 days |
| Bitap Kernel | **Medium** | New Metal kernel, tested pattern. Complexity in host-side dispatch routing = 2-3 days |
| Deferred Line Counting | **High** | turbo_search.metal already does this. Make it default path = 1 day |
| Quad I/O Pipeline | **Medium** | Streaming.rs has structure, need true MTLIOCommandQueue overlap = 3-4 days |
| **Overall** | **High** | 15-20 engineering days total |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| byte_offset fix is more complex than expected | Medium | High | Add CPU verification layer first, fix iteratively |
| FSEvents edge cases (rename, move) | Medium | Medium | Polling fallback, 1-hour staleness cap |
| Index memory (700K * 256B = 175MB) | Low | Medium | mmap with MADV_SEQUENTIAL, cap at 500K |
| Bitap kernel correctness | Low | Medium | Extensive proptest comparison with CPU reference |
| Breaking existing 409 tests | Medium | Medium | Add new tests first, then fix, verify all pass |

## Recommendations

1. Fix P0 accuracy bug first -- add memchr CPU verification, then fix byte_offset chain
2. Wire existing GSIX index before building new features -- instant TTFR improvement
3. Use turbo_search deferred line counting as default path
4. Add PipelineProfile instrumentation to measure before/after
5. Bitap kernel and I/O overlap are Phase 2 enhancements after accuracy + index
