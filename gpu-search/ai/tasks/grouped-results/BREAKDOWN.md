---
id: grouped-results.BREAKDOWN
module: grouped-results
priority: 5
status: failing
version: 1
origin: spec-workflow
dependsOn: [data-model, path-utils]
tags: [layout, virtual-scroll, rendering, core]
testRequirements:
  unit:
    required: true
    pattern: "src/ui/results_list.rs::tests, src/ui/app.rs::tests"
---
# Grouped Results Breakdown

## Context

This is the largest module in the overhaul -- a near-complete rewrite of `results_list.rs`. Currently, content matches appear in a single flat list with the full path repeated for every match. The overhaul groups content matches by file (file header row with filename + match count, indented match rows beneath), introduces a flattened virtual scroll model with heterogeneous row heights via `show_viewport()`, and adds selected-row expansion from 24px to 52px showing context lines.

The existing `show_file_matches()` and `show_content_matches()` using `show_rows()` with fixed heights are replaced by a unified rendering pipeline that iterates over a `FlatRowModel` of `RowKind` entries, rendering each visible row based on its type. Keyboard navigation skips section headers and group headers. Groups are sorted by match count descending on `SearchUpdate::Complete`.

From UX.md Section 6, TECH.md Sections 4.1-4.4, TECH.md Section 11, PM P1-2, PM P1-4, PM P1-5, PM Q3.

## Acceptance Criteria

1. `recompute_groups()` method on `GpuSearchApp` incrementally builds `content_groups: Vec<ContentGroup>` from `content_matches` using a `HashMap<PathBuf, usize>` for O(1) insertion
2. `rebuild_flat_row_model()` constructs `FlatRowModel` from file_matches + content_groups, with correct RowKind sequence and cumulative heights
3. `results_list.show()` replaced with `show_viewport()` rendering that binary-searches `cum_heights` for first visible row and renders only visible rows (~25)
4. `render_group_header()` implemented: 28px height, 6px colored dot, bold filename, match count, abbreviated directory path
5. `render_match_row()` implemented: 24px compact with `:line_number` + content; 52px expanded with context_before + match + context_after
6. `render_file_row()` updated: abbreviated path via `path_utils::abbreviate_path()`, 6px colored dot
7. Section headers show counts: "FILENAME MATCHES (N)" and "CONTENT MATCHES (N in M files)"
8. Keyboard navigation (`select_next`/`select_prev`) skips `SectionHeader` and `GroupHeader` entries -- only `FileMatchRow` and `MatchRow` are selectable
9. Tab jumps from filename section to first content match; Shift+Tab jumps back
10. Selected `MatchRow` expands from 24px to 52px; previous selection collapses back to 24px
11. `rebuild_flat_row_model()` called on selection change to update cum_heights
12. Groups sorted by match count descending on `SearchUpdate::Complete`; scroll preserves selected item identity after sort
13. Incremental grouping: new ContentMatches batches insert into existing groups or create new groups; groups appear in arrival order during streaming
14. 6px colored dot rendered before filename in group headers and file match rows using `theme::extension_dot_color()`
15. Unit tests pass: U-GRP-1..10 (grouping), U-ROW-4..9 (flattening with mixed content), U-PFX-1..8 (prefix-sum heights)
16. Integration tests pass: I-NAV-1..7 (navigation skipping headers), I-EXP-1..5 (row expansion/collapse)
17. Integration tests pass: I-PIPE-1..4 (search -> group -> display pipeline)
18. Performance: `rebuild_flat_row_model()` <1ms for 10K rows; frame time <5ms for 25 visible rows
19. Existing results_list tests maintained during migration (dual suites per QA Q3), then cleaned up
20. All existing tests pass: `cargo test -p gpu-search`

## Technical Notes

- Reference: [spec/OVERVIEW.md] grouped-results is priority 5, depends on data-model and path-utils. This is the XL effort module (~4hr for show_viewport rewrite alone)
- UX: From UX.md Sections 3.2-3.4, 6.1-6.4 -- detailed wireframes for group header, match row, selected expansion, progressive grouping during streaming
- Test: From QA.md Sections 2.3 (grouping), 2.4 (flattening), 2.5 (prefix-sum), 3.1 (pipeline), 3.3 (navigation), 3.4 (expansion)
- Risk: results_list.rs rewrite is the highest-risk change (TECH.md Section 13.1). Mitigation: implement new RowKind selection first with unit tests, then migrate rendering
- Risk: show_viewport() scroll position reset when total height changes (TECH.md Section 13.2). Mitigation: save/restore scroll offset
- Risk: virtual scroll jank from frequent rebuilds (QA Section 9.1). Mitigation: debounce rebuild to max once per 100ms during streaming
- The `SyntaxHighlighter` is passed as `&mut` parameter to rendering methods (TECH Q1)
- Double-click on match row opens file (UX Section 7.3)
- Click on group header selects first match in that group (UX Section 7.3)
