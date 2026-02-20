---
spec: gpu-search-false-positives
phase: requirements
created: 2026-02-15
generated: auto
---

# Requirements: gpu-search-false-positives

## Summary

Fix GPU content search false positives caused by stale buffer data and byte_offset calculation errors. Build a comprehensive integration test suite that simulates rapid query changes to catch false positives, stale results, and match_range corruption.

## User Stories

### US-1: Zero false positives in content search results
As a user, I want content search results to only show files that actually contain my search pattern so that I can trust the results.

**Acceptance Criteria**:
- AC-1.1: Searching for a unique pattern returns only files that contain that pattern (CPU-verified)
- AC-1.2: Repeating the same search immediately produces identical results (deterministic)
- AC-1.3: Searching for a pattern that exists in 0 files returns 0 content matches
- AC-1.4: False positive rate is 0% when verified against CPU ground truth

### US-2: Correct results during rapid query changes
As a user, I want correct results when I type quickly, changing my search pattern mid-search, so that results reflect my current query, not a previous one.

**Acceptance Criteria**:
- AC-2.1: Typing "kol" then changing to "pat" then to "kolbey" returns correct results for each query
- AC-2.2: No results from a previous query leak into the current query's results
- AC-2.3: Generation guards properly discard stale results
- AC-2.4: Cancellation of old searches stops GPU processing promptly

### US-3: Accurate match_range highlighting
As a user, I want the highlighted text in search results to exactly match my search pattern so that I can see where the match is.

**Acceptance Criteria**:
- AC-3.1: `match_range` in `ContentMatch` corresponds to the exact byte range of the pattern in `line_content`
- AC-3.2: `line_content[match_range]` equals the search pattern (case-insensitive comparison when case_sensitive=false)
- AC-3.3: byte_offset from GPU correctly maps to the file-relative position

### US-4: Comprehensive test suite for search correctness
As a developer, I want an automated test suite that catches false positives, stale results, and match_range corruption so that regressions are caught in CI.

**Acceptance Criteria**:
- AC-4.1: Test suite includes rapid query change simulation (>=3 sequential queries)
- AC-4.2: Test suite includes stale buffer detection (verify buffer state after reset)
- AC-4.3: Test suite includes match_range accuracy validation against CPU ground truth
- AC-4.4: Test suite includes concurrent search cancellation tests
- AC-4.5: Test suite includes large file boundary tests (files spanning multiple 4KB chunks)
- AC-4.6: All tests pass in CI with `GPU_SEARCH_VERIFY=full`

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | `ContentSearchEngine::reset()` must zero GPU buffer data (metadata + match buffers) | Must | US-1 |
| FR-2 | `byte_offset` calculation must produce file-relative offsets that correctly map to line content | Must | US-1, US-3 |
| FR-3 | Rapid sequential searches must produce correct results for each query independently | Must | US-2 |
| FR-4 | `resolve_match()` must reject all GPU matches where pattern is not found at the byte offset | Must | US-1 |
| FR-5 | `match_range` must exactly correspond to pattern position within `line_content` | Must | US-3 |
| FR-6 | Integration test suite must cover rapid query changes, stale buffers, match_range accuracy, cancellation, and chunk boundaries | Must | US-4 |
| FR-7 | Existing 556 unit tests must continue to pass with zero regressions | Must | All |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Buffer zeroing in `reset()` must add <1ms overhead per search dispatch | Performance |
| NFR-2 | Test suite must complete in <30 seconds (excluding large file tests) | Performance |
| NFR-3 | No new unsafe code beyond what exists in content.rs buffer management | Safety |

## Out of Scope

- Rewriting the Metal kernel dispatch model
- True GPU I/O overlap (Phase 2 of streaming architecture)
- Boyer-Moore or other advanced pattern matching in the GPU kernel
- UI-level integration tests (egui testing)

## Dependencies

- `tempfile` crate (already in dev-dependencies)
- `memchr` crate (already in dependencies)
- Metal GPU device (tests require Apple Silicon)
