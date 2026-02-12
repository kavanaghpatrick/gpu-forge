---
spec: gpu-search-overhaul
phase: requirements
created: 2026-02-12
generated: auto
---

# Requirements: gpu-search-overhaul

## Summary

Fix gpu-search's P0 false positive bug (GPU byte_offset maps to wrong file in multi-file batches), eliminate 33s time-to-first-result by wiring the existing GSIX filesystem index with FSEvents change detection, add per-stage pipeline profiler, and close critical test coverage gaps -- all while maintaining existing 79-110 GB/s GPU throughput.

## User Stories

### US-1: Accurate Search Results
As a developer, I want every content match to show the correct file and line containing my search pattern so that I can trust gpu-search results and open the right file.

**Acceptance Criteria**:
- AC-1.1: Every `ContentMatch.line_content` contains the search pattern (case-adjusted)
- AC-1.2: Every `ContentMatch.path` is a file that contains the pattern when read from disk
- AC-1.3: Zero false positives across a 10-pattern test corpus including multi-file batches
- AC-1.4: `resolve_match()` rejects mismatched byte offsets and logs rejections for debugging
- AC-1.5: CPU verification via `memchr::memmem` confirms GPU results in test mode

### US-2: Fast Time-to-First-Result
As a developer, I want search results to appear within 500ms for previously-searched directories so that gpu-search feels instant like Spotlight.

**Acceptance Criteria**:
- AC-2.1: Index-backed searches deliver first result in <500ms for ~/  (previously indexed)
- AC-2.2: Cold searches (no index) deliver first result in <3s from project root
- AC-2.3: Index loads in <100ms via mmap from GSIX binary format
- AC-2.4: Producer thread feeds from index instead of walk when index is fresh

### US-3: Always-Fresh Index
As a developer, I want the filesystem index to auto-update when files change so that search results reflect my latest code without manual rebuilds.

**Acceptance Criteria**:
- AC-3.1: notify v7 FSEvents watcher detects file create/modify/delete within 500ms
- AC-3.2: Index updates are debounced (500ms batch window) to prevent churn during builds
- AC-3.3: Stale index (>1 hour) falls back to walk-based search automatically
- AC-3.4: Watcher is zero-cost when idle (FSEvents journal-based, no polling)

### US-4: Pipeline Visibility
As a developer working on gpu-search, I want per-stage timing breakdown for every search so that I can identify and fix performance bottlenecks.

**Acceptance Criteria**:
- AC-4.1: `PipelineProfile` captures walk, filter, batch, GPU load, GPU dispatch, resolve, and total timing
- AC-4.2: Profile includes counters: files_walked, files_searched, bytes_searched, matches_raw, matches_resolved, matches_rejected
- AC-4.3: Time-to-first-result (TTFR) measured and reported
- AC-4.4: Profile attached to `SearchResponse`

### US-5: Search Quality Verification
As a developer, I want automated tests comparing gpu-search results against `grep -rn` so that accuracy regressions are caught before merge.

**Acceptance Criteria**:
- AC-5.1: Grep oracle test covers at least 6 patterns on gpu-search/src/
- AC-5.2: Zero false positives (gpu-search never returns matches grep doesn't)
- AC-5.3: Miss rate <20% (boundary misses acceptable, documented)
- AC-5.4: Accuracy tests run on every PR and block merge on failure

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | CPU verification layer using `memchr::memmem` validates GPU match byte offsets against file content | Must | US-1, AC-1.5 |
| FR-2 | `PipelineProfile` struct with per-stage timing (walk/filter/batch/GPU-load/GPU-dispatch/resolve/total) and counters | Must | US-4, AC-4.1 |
| FR-3 | Index-first search path: orchestrator queries `MmapIndexCache` for file list when index is fresh, skipping directory walk | Must | US-2, AC-2.4 |
| FR-4 | `notify` v7 FSEvents watcher with `notify-debouncer-mini` (500ms) for incremental index updates | Must | US-3, AC-3.1 |
| FR-5 | Fix `collect_results()` byte_offset calculation to use match position instead of ambiguous `context_start` | Must | US-1, AC-1.1 |
| FR-6 | Deferred line counting: GPU records byte offsets only, CPU extracts line numbers (turbo_search pattern as default) | Should | US-1 |
| FR-7 | Quad-buffered I/O pipeline with `read()` into 64MB MTLBuffer pool (not mmap per-file) | Should | US-2 |
| FR-8 | Bitap kernel for patterns <=32 chars (NFA state in single `uint32`) | Could | Research |
| FR-9 | Hot/cold dispatch: route to CPU NEON for cold data (not in page cache), GPU for hot data | Could | Research |

## Non-Functional Requirements

| ID | Requirement | Category | Source |
|----|-------------|----------|--------|
| NFR-1 | 100% search accuracy: no false positives in any content match | Correctness | US-1, AC-1.3 |
| NFR-2 | Time-to-first-result <500ms for indexed home directory searches | Performance | US-2, AC-2.1 |
| NFR-3 | Index load <100ms via mmap for GSIX binary format | Performance | US-2, AC-2.3 |
| NFR-4 | `read()` into MTLBuffer pool for multi-file scanning (not mmap per-file), per KB #1343 | Performance | Research |
| NFR-5 | No regression in GPU throughput (maintain 79-110 GB/s on M4 Pro) | Performance | Baseline |
| NFR-6 | Memory: quad-buffer pool = 256MB (4x64MB), index mmap ~175MB for 700K files | Resource | Research |
| NFR-7 | All existing 409 tests (256 unit + 153 integration) continue to pass | Compatibility | Existing |

## Out of Scope

- Regex support (literal string search only)
- Content indexing (trigram/n-gram) -- only path enumeration index
- Cross-platform (macOS Apple Silicon only)
- Semantic/fuzzy search
- UI redesign (only progressive loading improvements)
- New GPU kernels beyond Bitap (existing 4 kernels are correct)
- Multi-pattern search (PFAC for future)

## Dependencies

- `memchr` 2 (already in Cargo.toml) -- CPU verification
- `notify` 7 (new) -- FSEvents watcher
- `notify-debouncer-mini` 0.5 (new) -- event batching
- Existing GSIX index infrastructure in `src/index/` (cache, scanner, shared_index, gpu_index, gpu_loader)
- Existing `turbo_search.metal` deferred line counting kernel
