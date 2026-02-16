---
spec: gpu-search-overhaul-3
phase: requirements
created: 2026-02-16
generated: auto
---

# Requirements: gpu-search-overhaul-3

## Summary

Fix 3 root causes of gpu-search being 25-34x slower than GPUripgrep: remove 100K chunk cap for single-dispatch GPU search, move line number resolution to GPU kernel, and pre-build chunk metadata at index time.

## User Stories

### US-1: Single-dispatch GPU search

As a developer searching a 21GB codebase, I want all 6.39M chunks dispatched in one GPU call so that search completes in <200ms instead of 6.6s.

**Acceptance Criteria**:
- AC-1.1: `search_zerocopy()` dispatches all chunks in a single `cmd.commit()` + `waitUntilCompleted()`
- AC-1.2: No serial batch loop in `search_zerocopy()` -- metadata buffer sized to actual chunk count
- AC-1.3: GPU throughput >= 50 GB/s (up from 3.2 GB/s with batching)

### US-2: GPU-computed line numbers

As a developer viewing search results, I want accurate line numbers computed by the GPU kernel so that the 3.6s CPU resolve loop is eliminated.

**Acceptance Criteria**:
- AC-2.1: GPU kernel counts newlines from file start to match offset (not just 64-byte window)
- AC-2.2: CPU resolve loop in orchestrator.rs removed -- line numbers read directly from GPU output
- AC-2.3: Line numbers match CPU reference implementation for all test cases
- AC-2.4: Context lines (before/after) still populated for UI display

### US-3: Pre-built chunk metadata in GCIX

As a developer typing search queries, I want chunk metadata loaded from GCIX at startup so that no metadata rebuild happens per keystroke.

**Acceptance Criteria**:
- AC-3.1: `build_chunk_metadata()` called once at GCIX save time, stored in GCIX file
- AC-3.2: `load_gcix()` loads pre-built chunk metadata into ContentStore
- AC-3.3: `search_from_content_store()` uses pre-built metadata instead of calling `build_chunk_metadata()`
- AC-3.4: GCIX format version bumped to v3 with backward-compatible v2 fallback

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Remove 100K cap in ContentSearchEngine::new() | Must | US-1 |
| FR-2 | Size metadata buffer to actual chunk count, not fixed cap | Must | US-1 |
| FR-3 | Dispatch all chunks in single GPU call (no batch loop) | Must | US-1 |
| FR-4 | Add file-relative line number computation to zerocopy kernel | Must | US-2 |
| FR-5 | Remove CPU resolve loop (orchestrator.rs lines 1491-1581) | Must | US-2 |
| FR-6 | Populate context_before/context_after from in-memory content | Should | US-2 |
| FR-7 | Persist chunk metadata in GCIX format | Must | US-3 |
| FR-8 | Load pre-built chunk metadata on GCIX reload | Must | US-3 |
| FR-9 | Bump GCIX version to v3 with v2 fallback | Must | US-3 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Full search of 21GB codebase for "kolbey" completes in <200ms | Performance |
| NFR-2 | GPU throughput >= 50 GB/s for single dispatch | Performance |
| NFR-3 | All existing tests pass after changes | Compatibility |
| NFR-4 | No increase in GCIX file size by more than 5% | Storage |

## Out of Scope

- Kernel algorithm changes (search logic is already optimal)
- UI changes (result rendering unchanged)
- Multi-GPU support
- Streaming pipeline changes (search_streaming_inner disk-based path)

## Dependencies

- Metal compute shader compilation (search_types.h, content_search.metal)
- GCIX format compatibility (existing v2 files must still load or trigger rebuild)
- ContentStore Metal buffer lifecycle (drop order invariants)
