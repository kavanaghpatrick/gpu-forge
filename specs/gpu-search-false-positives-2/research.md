---
spec: gpu-search-false-positives-2
phase: research
created: 2026-02-16
generated: auto
---

# Research: gpu-search-false-positives-2

## Executive Summary

GPU content search produces false positive matches: searching "kolbey" returns ~9 matches in "Patrick Kavanagh" files that do NOT contain "kolbey". Root cause analysis identifies 3 likely sources across the streaming pipeline, orchestrator resolve step, and client-side refinement filter. Existing test infrastructure (test_false_positives.rs, verify.rs) covers GPU-level buffer isolation but NOT the full orchestrator+UI pipeline path where the bug manifests.

## Codebase Analysis

### Existing Defenses (Already Implemented)
- `ContentSearchEngine::reset()` (content.rs:299) -- zeros metadata_buffer, match_count_buffer, matches_buffer
- `StreamingSearchEngine::search_files()` (streaming.rs:376) -- calls reset() per sub-batch inside autoreleasepool
- `cpu_verify_matches()` (verify.rs:63) -- compares GPU byte_offsets against memchr ground truth
- `resolve_match()` (orchestrator.rs:1488) -- re-reads file, finds pattern in line, rejects if not found
- `StampedUpdate` generation guard (types.rs:108) -- discards stale-gen updates in poll_updates()
- Existing test_false_positives.rs -- 6 tests covering StreamingSearchEngine-level isolation

### Pipeline Architecture (3 layers where false positives can originate)

**Layer 1: GPU dispatch** (content.rs + streaming.rs)
- ContentSearchEngine dispatches Metal compute kernel
- StreamingMatch carries `file_path`, `byte_offset`, `match_length`
- `byte_offset` = `m.byte_offset.saturating_sub(start_chunk * 4096)` (streaming.rs:414)
- Risk: if `start_chunk` tracking is wrong, byte_offset points to wrong position

**Layer 2: Orchestrator resolve** (orchestrator.rs:1098-1213)
- `dispatch_gpu_batch_profiled()` calls `engine.search_files_with_profile()`
- For each StreamingMatch, calls `resolve_match(path, byte_offset, pattern, case_sensitive)`
- `resolve_match` re-reads the file, finds which line byte_offset falls on, then searches for pattern in that line
- **KEY**: If pattern is found ANYWHERE in the line (not just at byte_offset), the match is accepted
- This means: GPU points at wrong byte_offset due to stale data, but if the LINE happens to contain the pattern somewhere, the false positive passes through

**Layer 3: Client-side refinement** (app.rs:345-413)
- When new query extends old (e.g., "pa" -> "pat" -> "patr" -> "patri" -> "patric" -> "patrick")
- Filters existing content_matches by `cm.line_content.to_lowercase().contains(&query_lower)`
- Then updates `cm.match_range` using `cm.line_content.to_lowercase().find(&query_lower)`
- **CRITICAL BUG PATH**: If user searched "fa" first (which matches many files), then switches to "kolbey", the refinement filter is NOT applied (different query, not an extension). But if the user was typing "k" -> "ko" -> "kol" -> "kolb" -> "kolbe" -> "kolbey", each step applies refinement to the previous results. If "k" returned matches in "Patrick Kavanagh" files (line_content contains "k"), then refinement for "ko" would filter to lines containing "ko" -- which would filter those out. So the refinement path is likely NOT the root cause.

### Key Finding: resolve_match() False Positive Amplifier

The `resolve_match()` function (orchestrator.rs:1530-1536) searches for the pattern anywhere in the line, not at the specific byte_offset. If a GPU byte_offset is wrong (pointing to wrong file/line), but the target line happens to contain the pattern, the match passes through:

```rust
let match_col = if case_sensitive {
    line_content.find(pattern)?
} else {
    line_content.to_lowercase().find(&pattern.to_lowercase())?
};
```

This could explain how "Patrick Kavanagh" files match "kolbey": if a stale byte_offset points to the right file but wrong line, and some other line in that file happens to contain something matching the case-insensitive pattern.

### Constraints
- Must test on real Apple Silicon (Metal compute required)
- Existing test_false_positives.rs tests pass -- the bug is not in the StreamingSearchEngine layer
- The bug manifests at the orchestrator level (dispatch_gpu_batch_profiled + resolve_match)
- tempfile crate available in dev-dependencies
- crossbeam-channel for orchestrator channel testing

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | Existing verify.rs + test infrastructure covers GPU layer; need orchestrator-level tests |
| Effort Estimate | M | ~15 tasks across 5 phases |
| Risk Level | Medium | Root cause may be in byte_offset mapping rather than buffer contamination |

## Recommendations
1. Build orchestrator-level integration tests that exercise the full pipeline path (not just StreamingSearchEngine)
2. Add byte_offset validation: verify extracted text at byte_offset matches the pattern
3. Instrument resolve_match to log when GPU byte_offset disagrees with find() position
4. Test rapid cancel/restart cycles with the orchestrator (not just StreamingSearchEngine)
5. Fix resolve_match to validate byte_offset consistency, not just line-level pattern presence
