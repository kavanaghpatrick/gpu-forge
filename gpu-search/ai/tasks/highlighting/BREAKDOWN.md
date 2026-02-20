---
id: highlighting.BREAKDOWN
module: highlighting
priority: 6
status: failing
version: 1
origin: spec-workflow
dependsOn: [data-model]
tags: [rendering, syntax, highlighting]
testRequirements:
  unit:
    required: true
    pattern: "src/ui/highlight.rs::tests, src/ui/results_list.rs::tests"
---
# Highlighting Breakdown

## Context

Match highlighting in gpu-search has two interconnected problems. First, `render_highlighted_text()` (`results_list.rs:396-485`) performs case-insensitive query string re-search to find highlight positions, ignoring the GPU-reported `match_range: Range<usize>` on `ContentMatch`. This causes wrong highlights when the query appears multiple times in a line. Second, the `SyntaxHighlighter` in `highlight.rs` has a complete implementation (330+ lines, Tokyo Night theme, span splitting, `apply_match_overlay()`) that is entirely unused -- the `_highlighter` field on `GpuSearchApp` is prefixed with underscore.

The fix connects the existing syntax highlighter to the results renderer, adds a new `apply_match_range_overlay()` function that uses the GPU's byte range directly (instead of re-searching for the query), and adds `render_styled_spans()` to paint the resulting styled spans with correct syntax coloring and amber match overlay.

From PM P1-3, PM P2-2, UX.md Section 5, TECH.md Section 6, TECH Q1.

## Acceptance Criteria

1. New public function `apply_match_range_overlay(spans: Vec<StyledSpan>, match_range: Range<usize>) -> Vec<StyledSpan>` added to `src/ui/highlight.rs`
2. Internal helper `split_spans_at_ranges(spans, ranges)` extracted from existing `apply_match_overlay()` and shared between both functions
3. `apply_match_range_overlay` splits syntax spans at `match_range` boundaries and applies amber bold + background overlay to the matching portion
4. New function `render_styled_spans(painter, pos, spans, font_size) -> f32` added to `src/ui/results_list.rs` that paints spans with correct foreground color and optional background rect
5. `_highlighter` field renamed to `highlighter` on `GpuSearchApp` (removing unused prefix)
6. `SyntaxHighlighter` passed as `&mut SyntaxHighlighter` parameter to `results_list.show()` (TECH Q1 decision)
7. Content match rendering pipeline: `highlighter.highlight_line()` -> `apply_match_range_overlay(spans, cm.match_range)` -> `render_styled_spans(painter, pos, spans, 12.0)`
8. File match highlighting unchanged: continues using query-based highlighting for filename portion only
9. Match highlight visual: ACCENT (#E0AF68) foreground bold, ACCENT at 20% alpha background, 2px corner radius (UX.md Section 5.5)
10. `match_range` clamped to line length before applying overlay (prevents panic on truncated lines)
11. Empty `match_range` (0..0) results in no overlay applied
12. Unit test U-MRO-1 passes: basic match_range overlay at known position
13. Unit test U-MRO-2 passes: match_range at line start
14. Unit test U-MRO-5 passes: match_range crosses syntax span boundary (span split correctly)
15. Unit test U-MRO-6 passes: empty range produces no overlay
16. Unit test U-MRO-7 passes: match_range beyond line length is clamped (no panic)
17. All 14 existing highlight tests still pass (regression)
18. All existing tests pass: `cargo test -p gpu-search`

## Technical Notes

- Reference: [spec/OVERVIEW.md] highlighting is priority 6, depends on data-model (needs ContentMatch.match_range)
- UX: From UX.md Section 5 -- proposed approach: match_range + syntax highlighting; Section 5.3 -- render_styled_spans signature
- Test: From QA.md Section 2.9 (U-MRO-1..8) -- match range overlay tests
- Fresh syntect parse state per line (TECH TD-5): call `reset_cache(ext)` before each `highlight_line()`. Single-line highlighting is 95%+ accurate
- Performance: ~50-80us per `highlight_line()` call on M4. For ~20 visible rows: ~1-1.6ms total (within 2ms budget)
- Risk: match_range byte offsets invalid after line truncation (TECH.md Section 13.2). Mitigation: apply overlay BEFORE truncation, then truncate styled spans
- Risk: SyntaxHighlighter borrow conflict (TECH.md Section 13.1). Resolved by passing as `&mut` parameter
- The existing `apply_match_overlay()` (query-based) is NOT removed -- it is still used by file match filename highlighting
