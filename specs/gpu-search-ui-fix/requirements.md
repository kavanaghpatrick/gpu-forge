---
spec: gpu-search-ui-fix
phase: requirements
created: 2026-02-17
generated: auto
---

# Requirements: gpu-search-ui-fix

## Summary

Fix 3 bugs in gpu-search-ui filename search (GPU context truncation, path prefix loss, index rebuild on launch) and add comprehensive automated test suite for GPU search correctness.

## User Stories

### US-1: Full path context extraction
As a user searching filenames, I want the full path (newline-to-newline) returned for each match so that I can see the complete filename and open it.

**Acceptance Criteria**:
- AC-1.1: Searching "Mutual_Fund" returns the complete path including filename extension, not truncated at 64 bytes
- AC-1.2: CPU context extraction time reduced by >= 60% compared to current backward/forward scan
- AC-1.3: GPU kernel writes context_start (line start offset) and context_len (full line length) per match
- AC-1.4: `search_paths()` API signature unchanged — callers unaffected

### US-2: Absolute path preservation
As a user, I want search results to show absolute paths (e.g. `/usr/share/vim/vim91/...`) so I can open files directly.

**Acceptance Criteria**:
- AC-2.1: All paths loaded via `load_paths()` round-trip through GPU search with leading `/` preserved
- AC-2.2: No path starts with `share/`, `Users/`, or other relative-looking prefix in results
- AC-2.3: Paths spanning chunk boundaries are extracted correctly

### US-3: Persistent path index
As a user, I want the file index to load instantly on app launch instead of re-walking the filesystem every time.

**Acceptance Criteria**:
- AC-3.1: Path index persisted to `~/.cache/gpu-search-ui/paths.bin` after first scan
- AC-3.2: Subsequent launches load cached index in < 500ms (vs seconds for FS walk)
- AC-3.3: Cache invalidated when root dir changes or cache is > 1 hour old
- AC-3.4: Falls back to full FS walk if cache is missing, corrupt, or stale

### US-4: Automated GPU search test suite
As a developer, I want comprehensive automated tests that verify GPU search correctness without GUI interaction.

**Acceptance Criteria**:
- AC-4.1: Path round-trip test: load N paths, search, verify all expected matches found with full paths
- AC-4.2: Chunk boundary test: path positioned exactly at 4KB boundary is correctly extracted
- AC-4.3: Long path test: paths > 200 bytes are fully preserved
- AC-4.4: Unicode path test: paths with non-ASCII characters work correctly
- AC-4.5: Case sensitivity test: case-insensitive search finds mixed-case paths
- AC-4.6: Performance regression test: search completes in < 10ms at 500K paths

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | turbo_search_kernel scans backward/forward to newline per match, writes context_start + context_len | Must | US-1 |
| FR-2 | search_paths() uses GPU-provided context offsets instead of CPU newline scan | Must | US-1 |
| FR-3 | load_paths() preserves absolute path prefix in chunk_data | Must | US-2 |
| FR-4 | Path index serialized to ~/.cache/gpu-search-ui/paths.bin | Should | US-3 |
| FR-5 | Cache header includes root path, file count, timestamp for staleness check | Should | US-3 |
| FR-6 | Test suite covers round-trip, boundaries, long paths, unicode, case, perf | Must | US-4 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | search_paths() at 1M paths completes in < 5ms (from ~10ms currently) | Performance |
| NFR-2 | GPU kernel change does not reduce throughput below 30 GB/s | Performance |
| NFR-3 | Cache file size < 500MB for 5M paths | Storage |
| NFR-4 | All tests runnable with `cargo test` (no GUI dependency) | Testability |

## Out of Scope

- Changing the eframe UI rendering code
- Modifying the content_search_kernel (only turbo kernel for path search)
- Adding new GPU kernels beyond modifying existing turbo_search_kernel
- Cross-platform support (macOS Apple Silicon only)
- Changing the search_paths() public API signature

## Dependencies

- `metal` crate v0.33 (existing)
- `ignore` crate v0.4 (existing)
- `crossbeam-channel` v0.5 (existing)
- No new external crates required
