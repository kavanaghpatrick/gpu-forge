---
id: data-model.BREAKDOWN
module: data-model
priority: 1
status: failing
version: 1
origin: spec-workflow
dependsOn: []
tags: [types, data-model, core]
testRequirements:
  unit:
    required: true
    pattern: "src/search/types.rs::tests, src/ui/results_list.rs::tests"
---
# Data Model Breakdown

## Context

The UI/UX overhaul requires three new core data types that multiple downstream modules depend on. `StampedUpdate` wraps `SearchUpdate` with a generation ID for stale result filtering (used by stale-results module). `ContentGroup` groups content matches by file path for the grouped display (used by grouped-results module). `RowKind` defines the flattened virtual scroll model with heterogeneous row types and heights (used by grouped-results module). These types must be defined first as they form the foundation for the generation guard, grouping logic, and virtual scroll rendering.

From TECH.md Section 3.1: StampedUpdate is a wrapper struct with `generation: u64` and `update: SearchUpdate`. From TECH.md Section 4.1: ContentGroup struct with path, dir_display, filename, extension, match_indices. From TECH.md Section 4.3: RowKind enum with SectionHeader, FileMatchRow, GroupHeader, MatchRow variants. From TECH.md Section 4.3: Height constants SECTION_HEADER_HEIGHT=24, GROUP_HEADER_HEIGHT=28, FILE_ROW_HEIGHT=28, MATCH_ROW_COMPACT=24, MATCH_ROW_EXPANDED=52.

## Acceptance Criteria

1. `StampedUpdate` struct added to `src/search/types.rs` with `generation: u64` and `update: SearchUpdate` fields, deriving `Debug` and `Clone`
2. `ContentGroup` struct added (in `src/ui/results_list.rs` or new grouping module) with fields: `path: PathBuf`, `dir_display: String`, `filename: String`, `extension: String`, `match_indices: Vec<usize>`
3. `RowKind` enum added with variants: `SectionHeader(SectionType)`, `FileMatchRow(usize)`, `GroupHeader(usize)`, `MatchRow { group_idx: usize, local_idx: usize }`
4. `SectionType` enum added with `FileMatches` and `ContentMatches` variants
5. Height constants defined: `SECTION_HEADER_HEIGHT: f32 = 24.0`, `GROUP_HEADER_HEIGHT: f32 = 28.0`, `FILE_ROW_HEIGHT: f32 = 28.0`, `MATCH_ROW_COMPACT: f32 = 24.0`, `MATCH_ROW_EXPANDED: f32 = 52.0`
6. `FlatRowModel` struct added with `rows: Vec<RowKind>`, `cum_heights: Vec<f32>`, `total_height: f32`
7. `FlatRowModel::first_visible_row(viewport_top: f32) -> usize` method implemented using `partition_point()` on cum_heights
8. Unit test `test_stamped_update_construction` passes (QA U-GEN-1)
9. Unit tests for RowKind flattening pass: U-ROW-1 (empty), U-ROW-2 (file matches only), U-ROW-3 (content groups only)
10. All existing tests still pass: `cargo test -p gpu-search`

## Technical Notes

- Reference: [spec/OVERVIEW.md] data-model is priority 1, no dependencies, blocks stale-results and grouped-results
- UX: From UX.md Section 6.2 -- flattened virtual list model with pre-known heights per row type
- Test: From QA.md Section 2.1 (U-GEN-1), Section 2.4 (U-ROW-1..3) -- type construction and basic flattening tests
- StampedUpdate wraps existing SearchUpdate without modifying it (TECH TD-2)
- RowKind and FlatRowModel are UI-side types, not part of the search protocol
- The channel type change (Receiver<SearchUpdate> -> Receiver<StampedUpdate>) is deferred to stale-results module
