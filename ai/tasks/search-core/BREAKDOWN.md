---
id: search-core.BREAKDOWN
module: search-core
priority: 2
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN, gpu-engine.BREAKDOWN]
tags: [gpu-search]
testRequirements:
  unit:
    required: true
---

# Search Core Module Breakdown

## Context

The search-core module is the orchestration layer between the GPU engine and the UI. It receives search requests from the UI (via the work queue), coordinates GPU path filtering and content search, collects and ranks results, and delivers them back to the UI via unified memory output buffers. It also handles .gitignore filtering, binary file detection, file type filtering, and search cancellation.

This module does NOT own the GPU kernels (gpu-engine does) or the UI (ui module does). It owns the search pipeline logic: what to search, in what order, and how to present results.

## Tasks

### T-030: Implement SearchOrchestrator

Top-level coordinator that manages the full search pipeline:

```
SearchRequest -> path filter -> file selection -> batch I/O -> content search -> result collection -> SearchResponse
```

Key responsibilities:
- Accept `SearchRequest` from UI (pattern, search root, filters)
- Dispatch to GPU path filter kernel on filesystem index
- Select files matching filters (.gitignore, file type, size limits)
- Queue selected files for batch I/O (MTLIOCommandQueue)
- Dispatch content search kernel on loaded files
- Collect match results into `SearchResponse`
- Send response back to UI via output buffer or channel

**Target**: `gpu-search/src/search/orchestrator.rs` (new file, not ported)
**Verify**: `cargo test -p gpu-search test_orchestrator` -- search for pattern in test directory, correct files and match counts returned

### T-031: Implement .gitignore filtering

Follow ripgrep convention: respect .gitignore by default (Q&A decision #5).

- Parse `.gitignore` files at each directory level
- Support standard gitignore patterns: globs, negation (`!`), directory markers (`/`)
- Use `ignore` crate (same as ripgrep) for correct .gitignore semantics
- Apply before GPU search (filter file list, not search results)

**Target**: `gpu-search/src/search/ignore.rs`
**Dependency**: Add `ignore = "0.4"` to Cargo.toml
**Verify**: `cargo test -p gpu-search test_gitignore` -- files matching .gitignore patterns excluded from search results

### T-032: Implement binary file detection

Follow ripgrep convention: skip binary files by default (Q&A decision #16).

- NUL byte (`\x00`) heuristic: read first 8KB, skip if NUL found
- Skip known binary extensions: `.exe`, `.o`, `.dylib`, `.metallib`, images, audio, video
- Configurable: `--binary` flag to include binary files

**Target**: `gpu-search/src/search/binary.rs`
**Verify**: `cargo test -p gpu-search test_binary_detection` -- binary files skipped, text files searched

### T-033: Implement file type filtering

Support filtering by file extension:
- Built-in type definitions: `rs` -> `*.rs`, `metal` -> `*.metal`, `py` -> `*.py`, etc.
- Custom extension patterns via filter pills in UI
- Multiple types can be active simultaneously (OR logic)

**Target**: `gpu-search/src/search/filetype.rs`
**Verify**: `cargo test -p gpu-search test_filetype_filter` -- only matching extensions returned

### T-034: Implement result ranking and deduplication

Rank search results for display:
1. Filename matches sorted by path length (shorter = more relevant)
2. Content matches sorted by: exact word match > partial match > path depth
3. Deduplicate: if filename matches, don't also show it in content matches section
4. Cap results: 10K maximum (Q&A, ripgrep convention for reasonable display)

**Target**: `gpu-search/src/search/ranking.rs`
**Verify**: `cargo test -p gpu-search test_result_ranking` -- results ordered by relevance score

### T-035: Implement search cancellation

When user types a new character, cancel in-flight search and start new one:
- Cancel previous GPU dispatch if still running
- Drop partial results from cancelled search
- Track search generation ID to discard stale results

**Target**: `gpu-search/src/search/cancel.rs`
**Verify**: `cargo test -p gpu-search test_search_cancel` -- rapid sequential searches, only latest results returned

### T-036: Implement progressive result delivery

Two-wave result delivery per UX spec:
1. **Wave 1 (0-2ms)**: Filename/path matches from GPU index
2. **Wave 2 (2-5ms)**: Content matches from GPU content search

Use channels (`std::sync::mpsc` or `crossbeam`) to stream results to UI:
- `SearchUpdate::FileMatches(Vec<FileMatch>)` -- Wave 1
- `SearchUpdate::ContentMatches(Vec<ContentMatch>)` -- Wave 2
- `SearchUpdate::Complete(stats)` -- final timing and counts

**Target**: `gpu-search/src/search/channel.rs`
**Verify**: `cargo test -p gpu-search test_progressive_results` -- Wave 1 arrives before Wave 2

### T-037: Define SearchRequest and SearchResponse types

Public API types for the search pipeline:

```rust
pub struct SearchRequest {
    pub pattern: String,
    pub root: PathBuf,
    pub file_types: Vec<String>,
    pub case_sensitive: bool,
    pub respect_gitignore: bool,
    pub include_binary: bool,
    pub max_results: usize,
}

pub struct FileMatch {
    pub path: PathBuf,
    pub score: f32,
}

pub struct ContentMatch {
    pub path: PathBuf,
    pub line_number: u32,
    pub line_content: String,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
    pub match_range: Range<usize>,
}

pub struct SearchResponse {
    pub file_matches: Vec<FileMatch>,
    pub content_matches: Vec<ContentMatch>,
    pub total_files_searched: usize,
    pub total_matches: usize,
    pub elapsed: Duration,
}
```

**Target**: `gpu-search/src/search/types.rs`
**Verify**: `cargo check -p gpu-search` -- types compile and are used by orchestrator

## Acceptance Criteria

1. `SearchOrchestrator` coordinates full pipeline: request -> filter -> I/O -> search -> response
2. .gitignore filtering works correctly using `ignore` crate (matches ripgrep behavior)
3. Binary files detected and skipped by default (NUL byte heuristic)
4. File type filtering supports common extensions (rs, py, js, ts, metal, md, toml, etc.)
5. Results ranked by relevance (filename match > content match, shorter paths preferred)
6. Search cancellation works: new keystroke cancels previous search, no stale results displayed
7. Progressive delivery: filename matches arrive before content matches
8. Max 10K results returned (cap for UI performance)
9. Search timing reported in `SearchResponse.elapsed` (for status bar display)
10. All search operations are async (never block the egui render thread)

## Technical Notes

- **ignore crate**: The `ignore` crate (by BurntSushi, same author as ripgrep) handles .gitignore, .ignore, global gitignore, and nested overrides. It's the gold standard for gitignore compatibility.
- **Async search**: The orchestrator runs on a background thread. UI submits requests via channel, receives results via channel. Never block the egui frame loop.
- **Content match context**: GPU kernel returns byte offsets. CPU post-processes to extract line content and +/- 1 context lines. This is the "turbo" mode approach from TECH.md Section 6.3.
- **Cancellation strategy**: Use an `AtomicBool` cancellation flag per search. GPU kernel checks flag between chunks. If set, kernel exits early.
- **Memory budget**: Search results are small (paths + line content). 10K results ~= 5-10MB. Well within memory budget.
- Reference: TECH.md Sections 2, 6, 10; UX.md Sections 3, 4; QA.md Sections 4, 6
