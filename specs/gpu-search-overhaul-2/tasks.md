---
spec: gpu-search-overhaul-2
phase: tasks
total_tasks: 28
created: 2026-02-13
generated: auto
---

# Tasks: gpu-search-overhaul-2

## Phase 1: Make It Work (POC)

Focus: Fix the P0 stale results bug and prove the generation-stamped architecture works. Add the core data model types. Accept minimal tests, no grouped rendering yet.

- [x] 1.1 Add StampedUpdate struct and change channel types
  - **Do**:
    1. In `src/search/types.rs`, add `StampedUpdate` struct with `generation: u64` and `update: SearchUpdate` fields. Derive Debug, Clone.
    2. In `src/search/orchestrator.rs`, change `search_streaming` signature from `tx: &Sender<SearchUpdate>` to `tx: &Sender<StampedUpdate>`. Wrap every `tx.send(SearchUpdate::...)` call with `StampedUpdate { generation: session.guard.generation_id(), update: ... }`.
    3. In `src/ui/app.rs`, change `update_rx: Receiver<SearchUpdate>` to `Receiver<StampedUpdate>`. Change channel creation in `new()` and `default()` to use `StampedUpdate`. Update `orchestrator_thread()` signature.
  - **Files**: `gpu-search/src/search/types.rs`, `gpu-search/src/search/orchestrator.rs`, `gpu-search/src/ui/app.rs`
  - **Done when**: Code compiles with `cargo check -p gpu-search`. Channel carries `StampedUpdate`.
  - **Verify**: `cargo check -p gpu-search`
  - **Commit**: `feat(gpu-search): add StampedUpdate wrapper for generation-stamped search messages`
  - _Requirements: FR-1_
  - _Design: Component A_

- [x] 1.2 Add generation guard to poll_updates()
  - **Do**:
    1. In `poll_updates()` in `app.rs`, unwrap `StampedUpdate`: `while let Ok(stamped) = self.update_rx.try_recv()`. Add guard: `if stamped.generation != self.search_generation.current_id() { continue; }`. Then match on `stamped.update`.
    2. Add `update_status_from_displayed()` method that derives status bar count from `self.file_matches.len() + self.content_matches.len()`. Call it after every poll loop iteration.
    3. Ensure `dispatch_search()` still clears results and drains channel before sending new command.
  - **Files**: `gpu-search/src/ui/app.rs`
  - **Done when**: `poll_updates()` discards stale generations. Status bar always matches displayed count.
  - **Verify**: `cargo check -p gpu-search`
  - **Commit**: `fix(gpu-search): add generation guard to poll_updates, fix stale results race condition`
  - _Requirements: FR-1, FR-18_
  - _Design: Component A_

- [x] 1.3 Change debounce from 30ms to 100ms
  - **Do**:
    1. In `src/ui/search_bar.rs`, change `const DEFAULT_DEBOUNCE_MS: u64 = 30;` to `100`.
    2. Update the test `test_default_debounce_is_30ms` to assert 100ms and rename to `test_default_debounce_is_100ms`.
  - **Files**: `gpu-search/src/ui/search_bar.rs`
  - **Done when**: Default debounce is 100ms. Updated test passes.
  - **Verify**: `cargo test -p gpu-search search_bar -- --nocapture`
  - **Commit**: `fix(gpu-search): increase search debounce from 30ms to 100ms`
  - _Requirements: FR-2_
  - _Design: Section 8_

- [x] 1.4 Change VerifyMode default to Sample with adaptive effective()
  - **Do**:
    1. In `src/search/verify.rs`, change default in `from_env()` from `VerifyMode::Off` to `VerifyMode::Sample`.
    2. Add `pub fn effective(self, result_count: usize) -> VerifyMode` method: returns `Full` when `self == Sample && result_count < 100`, otherwise returns `self`.
    3. In `src/search/orchestrator.rs`, find where verification is conditionally applied and use `mode.effective(results.len())` to determine effective mode.
    4. Update existing test `test_verify_mode_from_env` (or equivalent) to expect `Sample` as default.
  - **Files**: `gpu-search/src/search/verify.rs`, `gpu-search/src/search/orchestrator.rs`
  - **Done when**: Default verify mode is Sample. `effective()` upgrades to Full below 100 results.
  - **Verify**: `cargo test -p gpu-search verify -- --nocapture`
  - **Commit**: `feat(gpu-search): enable adaptive CPU verification (Sample default, Full when <100 results)`
  - _Requirements: FR-10_
  - _Design: Component F_

- [x] 1.5 Wire OpenFile, OpenInEditor, CopyPath actions
  - **Do**:
    1. In `app.rs` `handle_key_action()`, replace the deferred block for `OpenFile | OpenInEditor | CopyPath` with actual implementations.
    2. For `OpenFile`: get selected result path from `results_list`, call `actions::open_file(path)`.
    3. For `OpenInEditor`: get path + line number, call `actions::open_in_editor(path, line)`.
    4. For `CopyPath`: get path, set `ctx.output_mut(|o| o.copied_text = path.display().to_string())`.
    5. Add helper methods on `ResultsList`: `get_selected_path(&self, file_matches, content_matches) -> Option<&Path>` and `get_selected_line(...) -> Option<u32>` that use `selected_index` to look up the right match.
  - **Files**: `gpu-search/src/ui/app.rs`, `gpu-search/src/ui/results_list.rs`
  - **Done when**: Enter opens file, Cmd+Enter opens in editor, Cmd+C copies path.
  - **Verify**: `cargo check -p gpu-search`
  - **Commit**: `feat(gpu-search): wire OpenFile, OpenInEditor, CopyPath keybind actions`
  - _Requirements: FR-11_
  - _Design: Section 7.2_

- [x] 1.6 POC Checkpoint: verify stale results fix works end-to-end
  - **Do**:
    1. Run `cargo test -p gpu-search` -- all existing tests pass (with debounce + verify default updates).
    2. Run the app: `cargo run -p gpu-search -- --root ~/gpu_kernel/gpu-search`.
    3. Type "orchestrator" rapidly char-by-char. Verify: no stale prefix results visible; status bar count matches displayed results.
    4. Test Cmd+C copies path, Enter opens file.
  - **Files**: None (verification only)
  - **Done when**: Stale results bug is fixed. All existing tests pass. Actions work.
  - **Verify**: `cargo test -p gpu-search`
  - **Commit**: `feat(gpu-search): complete POC -- stale results fix verified`

## Phase 2: Features

Focus: Path abbreviation, grouped results, match highlighting, status bar improvements. This is the core layout rewrite.

- [x] 2.1 Create path_utils.rs module with abbreviate_path()
  - **Do**:
    1. Create `gpu-search/src/ui/path_utils.rs` with: `pub fn abbreviate_path(path: &Path, search_root: &Path) -> (String, String)` returning (dir_display, filename).
    2. Implementation: (a) try `path.parent().strip_prefix(search_root)` for relative, (b) try `$HOME` substitution via `std::env::var_os("HOME")` cached in `OnceLock<Option<PathBuf>>`, (c) middle-truncate if dir > 50 chars.
    3. Add `fn middle_truncate(s: &str, max_len: usize) -> String`.
    4. Add `pub mod path_utils;` to `src/ui/mod.rs`.
    5. Apply `abbreviate_path()` in `render_file_row()` for file match path display.
    6. Apply to `StatusBar` for root path display (`~` substitution).
  - **Files**: `gpu-search/src/ui/path_utils.rs` (create), `gpu-search/src/ui/mod.rs`, `gpu-search/src/ui/results_list.rs`, `gpu-search/src/ui/status_bar.rs`
  - **Done when**: File match rows show abbreviated paths. Status bar root uses `~`. New module has basic tests.
  - **Verify**: `cargo test -p gpu-search path_utils -- --nocapture`
  - **Commit**: `feat(gpu-search): add path abbreviation with relative-to-root and ~ substitution`
  - _Requirements: FR-3_
  - _Design: Component D_

- [x] 2.2 Add extension_dot_color() to theme.rs
  - **Do**:
    1. In `src/ui/theme.rs`, add `pub fn extension_dot_color(ext: &str) -> Color32` mapping: rs -> purple (#BB9AF7), py -> green (#9ECE6A), js/jsx -> amber (#E0AF68), ts/tsx -> blue (#7AA2F7), md/txt -> cyan (#2AC3DE), toml/yaml/json -> blue, sh/bash/zsh -> green, c/cpp/h -> blue, go -> cyan, swift -> amber, html -> red (#F7768E), default -> muted (#565F89).
    2. Add unit tests for known extensions and default fallback.
  - **Files**: `gpu-search/src/ui/theme.rs`
  - **Done when**: `extension_dot_color()` returns correct colors for all documented extensions.
  - **Verify**: `cargo test -p gpu-search theme -- --nocapture`
  - **Commit**: `feat(gpu-search): add extension-to-color mapping for file type dots`
  - _Requirements: FR-12_
  - _Design: Component C, Section 10_

- [x] 2.3 Add apply_match_range_overlay() to highlight.rs
  - **Do**:
    1. In `src/ui/highlight.rs`, extract the span-splitting logic from existing `apply_match_overlay()` into a shared `fn split_spans_at_ranges(spans: Vec<StyledSpan>, ranges: &[Range<usize>]) -> Vec<StyledSpan>`.
    2. Add `pub fn apply_match_range_overlay(spans: Vec<StyledSpan>, match_range: Range<usize>) -> Vec<StyledSpan>` that calls `split_spans_at_ranges` with `vec![match_range]`.
    3. Clamp `match_range.end` to `min(end, total_span_text_len)`. If `start >= total_len`, return spans unchanged.
    4. Refactor existing `apply_match_overlay()` to also use `split_spans_at_ranges` internally.
    5. Add unit tests: basic overlay, at-start, at-end, full-line, empty range, clamped range.
  - **Files**: `gpu-search/src/ui/highlight.rs`
  - **Done when**: `apply_match_range_overlay()` correctly splits spans at GPU match_range boundaries. All existing highlight tests still pass.
  - **Verify**: `cargo test -p gpu-search highlight -- --nocapture`
  - **Commit**: `feat(gpu-search): add apply_match_range_overlay for precise GPU match highlighting`
  - _Requirements: FR-7_
  - _Design: Component E_

- [x] 2.4 Add ContentGroup struct and incremental grouping logic
  - **Do**:
    1. In `src/ui/results_list.rs` (or new `src/ui/grouping.rs`), add `ContentGroup` struct: `path: PathBuf, dir_display: String, filename: String, extension: String, match_indices: Vec<usize>`.
    2. In `app.rs`, add fields: `content_groups: Vec<ContentGroup>`, `group_index_map: HashMap<PathBuf, usize>`, `last_grouped_index: usize`.
    3. Add `fn recompute_groups(&mut self)`: iterate from `last_grouped_index..content_matches.len()`, for each match either insert into existing group via `group_index_map` or create new group.
    4. Call `recompute_groups()` after every `ContentMatches` and `Complete` update in `poll_updates()`.
    5. On `Complete`, sort `content_groups` by `match_indices.len()` descending (most matches first).
    6. Clear groups in `dispatch_search()` when starting a new search.
  - **Files**: `gpu-search/src/ui/app.rs`, `gpu-search/src/ui/results_list.rs`
  - **Done when**: `content_groups` correctly reflect grouped content matches. Incremental grouping works. Groups sorted on Complete.
  - **Verify**: `cargo check -p gpu-search`
  - **Commit**: `feat(gpu-search): add ContentGroup with incremental HashMap-based grouping`
  - _Requirements: FR-4, FR-17_
  - _Design: Component B_

- [x] 2.5 Add RowKind enum, FlatRowModel, and rebuild logic
  - **Do**:
    1. In `src/ui/results_list.rs`, add `RowKind` enum: `SectionHeader(SectionType)`, `FileMatchRow(usize)`, `GroupHeader(usize)`, `MatchRow { group_idx: usize, local_idx: usize }`.
    2. Add `SectionType` enum: `FileMatches`, `ContentMatches`.
    3. Add height constants: `SECTION_HEADER_HEIGHT = 24.0`, `GROUP_HEADER_HEIGHT = 28.0`, `MATCH_ROW_COMPACT = 24.0`, `MATCH_ROW_EXPANDED = 52.0`.
    4. Add `FlatRowModel` struct with `rows: Vec<RowKind>`, `cum_heights: Vec<f32>`, `total_height: f32`.
    5. Implement `rebuild_flat_row_model()` method that constructs the flat row list from `file_matches` + `content_groups`, computing prefix-sum heights. Selected `MatchRow` gets `MATCH_ROW_EXPANDED`.
    6. Implement `first_visible_row(viewport_top: f32) -> usize` using `cum_heights.partition_point()`.
    7. Implement `is_selectable(idx)` returning true only for `FileMatchRow` and `MatchRow`.
    8. Implement `next_selectable_row(from)` and `prev_selectable_row(from)` that skip headers.
  - **Files**: `gpu-search/src/ui/results_list.rs`
  - **Done when**: FlatRowModel correctly represents grouped results with prefix-sum heights. Navigation skips headers.
  - **Verify**: `cargo check -p gpu-search`
  - **Commit**: `feat(gpu-search): add RowKind flat row model with prefix-sum virtual scroll`
  - _Requirements: FR-5, FR-15_
  - _Design: Component C_

- [ ] 2.6 Replace show_rows() with show_viewport() grouped rendering
  - **Do**:
    1. In `results_list.rs`, replace `show_file_matches()` and `show_content_matches()` with a unified `show()` method using `ScrollArea::vertical().show_viewport()`.
    2. Inside `show_viewport`, use `first_visible_row(viewport.top())` to find start row. Iterate forward rendering rows until y-position exceeds viewport bottom.
    3. For each visible row, match on `RowKind` and call the appropriate render function: `render_section_header()`, `render_file_row()` (updated), `render_group_header()` (new), `render_match_row()` (new, replaces `render_content_row()`).
    4. `render_group_header()`: 28px row with 6px colored dot + bold filename + " -- N matches" muted + abbreviated dir path.
    5. `render_match_row()`: 24px compact with `:line_number` in ACCENT + line content. Selected: 52px with context_before + match + context_after.
    6. Add `render_styled_spans()` function that paints `Vec<StyledSpan>` from the highlighter.
    7. Pass `&mut SyntaxHighlighter` as parameter to `show()`. Rename `_highlighter` to `highlighter` in `app.rs`.
    8. Wire `highlighter.highlight_line()` + `apply_match_range_overlay()` + `render_styled_spans()` for match rows.
    9. Update `show()` call in `app.rs` to pass `&mut self.highlighter`.
    10. Set `inner_content_size` on `ScrollArea` to `[0.0, self.flat_row_model.total_height]` so egui knows total scroll extent.
  - **Files**: `gpu-search/src/ui/results_list.rs`, `gpu-search/src/ui/app.rs`
  - **Done when**: Results display as grouped-by-file with section headers, group headers, file type dots, match rows with syntax + match_range highlighting. Virtual scroll works with variable heights.
  - **Verify**: `cargo run -p gpu-search -- --root ~/gpu_kernel/gpu-search` (visual verification)
  - **Commit**: `feat(gpu-search): replace flat results with grouped show_viewport rendering`
  - _Requirements: FR-5, FR-6, FR-8, FR-9, FR-13, FR-16_
  - _Design: Components B, C, E_

- [ ] 2.7 Update status bar with searching state and live count
  - **Do**:
    1. In `src/ui/status_bar.rs`, add `pub is_searching: bool` and `pub search_start: Option<Instant>` fields.
    2. Update render: when `is_searching`, show "Searching... | N matches | X.Xs | ~root | filters". When not searching, show "N matches in X.Xms | ~root | filters".
    3. Use `abbreviate_path` for root display (or inline `$HOME` substitution for the single root path).
    4. In `app.rs`, set `status_bar.is_searching = true` in `dispatch_search()` and `false` on `Complete`.
    5. Set `status_bar.search_start = Some(Instant::now())` in `dispatch_search()`.
    6. Ensure `update_status_from_displayed()` updates count every frame.
  - **Files**: `gpu-search/src/ui/status_bar.rs`, `gpu-search/src/ui/app.rs`
  - **Done when**: Status bar shows live progress during search and final counts after.
  - **Verify**: `cargo run -p gpu-search -- --root ~/gpu_kernel/gpu-search` (visual verification)
  - **Commit**: `feat(gpu-search): add live searching status with match count and elapsed time`
  - _Requirements: FR-14, FR-18_
  - _Design: Section 9_

- [ ] 2.8 Add hover tooltip and Tab section jump
  - **Do**:
    1. In `render_file_row()` and `render_group_header()`, add `response.on_hover_text(full_absolute_path)` to show full path on hover.
    2. In `keybinds.rs` or `results_list.rs`, add Tab handling: jump from filename section to first content match, Shift+Tab jumps back.
    3. Ensure `scroll_to_selected` works after section jump by computing selected row's y-position from `cum_heights`.
  - **Files**: `gpu-search/src/ui/results_list.rs`, `gpu-search/src/ui/keybinds.rs`
  - **Done when**: Hover on file/group header shows full path. Tab jumps between sections.
  - **Verify**: `cargo run -p gpu-search -- --root ~/gpu_kernel/gpu-search` (visual verification)
  - **Commit**: `feat(gpu-search): add path hover tooltip and Tab section navigation`
  - _Requirements: FR-19, FR-20_
  - _Design: Section 7.1_

## Phase 3: Testing

Focus: Comprehensive unit tests, integration tests, property tests, and visual snapshots per QA.md strategy.

- [ ] 3.1 Unit tests for StampedUpdate generation filtering
  - **Do**:
    1. In `src/search/types.rs` or `src/ui/app.rs` tests module, add tests U-GEN-1 through U-GEN-7.
    2. Create a test helper that mimics `poll_updates()` logic: takes a vec of `StampedUpdate`, a current_gen, and returns the filtered results.
    3. Test: current gen accepted, stale gen discarded, future gen discarded, rapid advance, Complete only applied if current gen, u64::MAX wrap.
  - **Files**: `gpu-search/src/search/types.rs` or `gpu-search/src/ui/app.rs`
  - **Done when**: 7 generation guard unit tests pass.
  - **Verify**: `cargo test -p gpu-search stamped -- --nocapture`
  - **Commit**: `test(gpu-search): add unit tests for StampedUpdate generation filtering`
  - _Requirements: AC-1.1, AC-1.3_
  - _Design: Component A_

- [ ] 3.2 Unit tests for path abbreviation
  - **Do**:
    1. In `src/ui/path_utils.rs` tests module, add tests U-PATH-1 through U-PATH-12.
    2. Cover: relative to root, root-level file, home substitution, deep path, outside home, middle truncation, no parent, unicode, empty filename, $HOME unset.
    3. Use `std::env::set_var("HOME", ...)` in tests that need controlled $HOME (wrap with `#[serial]` if needed).
  - **Files**: `gpu-search/src/ui/path_utils.rs`
  - **Done when**: 12 path abbreviation tests pass.
  - **Verify**: `cargo test -p gpu-search path_utils -- --nocapture`
  - **Commit**: `test(gpu-search): add comprehensive path abbreviation unit tests`
  - _Requirements: AC-2.1, AC-2.2, AC-2.3_
  - _Design: Component D_

- [ ] 3.3 Unit tests for ContentGroup building and sorting
  - **Do**:
    1. Add tests U-GRP-1 through U-GRP-10 testing incremental grouping logic.
    2. Cover: single file/match, single file/multiple matches, multiple files, incremental add, sort by count desc, stable sort, empty input, dir_display, extension extraction, idempotent recompute.
  - **Files**: `gpu-search/src/ui/app.rs` or `gpu-search/src/ui/results_list.rs`
  - **Done when**: 10 grouping tests pass.
  - **Verify**: `cargo test -p gpu-search group -- --nocapture`
  - **Commit**: `test(gpu-search): add unit tests for ContentGroup incremental grouping`
  - _Requirements: AC-3.1, AC-3.3_
  - _Design: Component B_

- [ ] 3.4 Unit tests for RowKind flattening and prefix-sum heights
  - **Do**:
    1. Add tests U-ROW-1 through U-ROW-9 for flat row model construction.
    2. Add tests U-PFX-1 through U-PFX-8 for prefix-sum invariants and binary search.
    3. Add 3 proptest property tests: (a) `cum_heights` monotonically increasing, (b) `partition_point` correctness for all positions, (c) round-trip height->row->height.
  - **Files**: `gpu-search/src/ui/results_list.rs`
  - **Done when**: 17 tests (9 row + 8 prefix-sum including 3 property tests) pass.
  - **Verify**: `cargo test -p gpu-search results_list -- --nocapture`
  - **Commit**: `test(gpu-search): add RowKind flattening and prefix-sum property tests`
  - _Requirements: NFR-2_
  - _Design: Component C_

- [ ] 3.5 Unit tests for adaptive VerifyMode
  - **Do**:
    1. Add tests U-VFY-1 through U-VFY-10 in `verify.rs` tests module.
    2. Cover: default is Sample, env off/full, effective() upgrades below 100, stays at 100, stays above, Full ignores count, Off ignores count, boundary 99, zero.
  - **Files**: `gpu-search/src/search/verify.rs`
  - **Done when**: 10 verify mode tests pass.
  - **Verify**: `cargo test -p gpu-search verify -- --nocapture`
  - **Commit**: `test(gpu-search): add adaptive VerifyMode unit tests`
  - _Requirements: AC-5.1, AC-5.2, AC-5.3_
  - _Design: Component F_

- [ ] 3.6 Unit tests for match_range overlay
  - **Do**:
    1. Add tests U-MRO-1 through U-MRO-8 in `highlight.rs` tests module.
    2. Cover: basic overlay, at-start, at-end, full line, span boundary split, empty range, clamped to line, compare match_range vs query search results.
  - **Files**: `gpu-search/src/ui/highlight.rs`
  - **Done when**: 8 match_range overlay tests pass. All 14 existing highlight tests still pass.
  - **Verify**: `cargo test -p gpu-search highlight -- --nocapture`
  - **Commit**: `test(gpu-search): add match_range overlay unit tests`
  - _Requirements: AC-4.1_
  - _Design: Component E_

- [ ] 3.7 Integration tests for stale result filtering
  - **Do**:
    1. Create `gpu-search/tests/test_stale_results.rs`.
    2. Implement synchronous test harness: construct `StampedUpdate` messages manually (no real channels), call poll helper directly.
    3. Add I-STALE-1: rapid dispatch (5 generations), only last gen results survive.
    4. Add I-STALE-2: manually inject stale gen, verify discarded.
    5. Add I-STALE-3: verify status bar count == displayed count invariant.
    6. Add I-STALE-4: simulate drain + late arrival with old gen.
  - **Files**: `gpu-search/tests/test_stale_results.rs` (create)
  - **Done when**: 4 integration tests pass, all deterministic (no timing dependence).
  - **Verify**: `cargo test -p gpu-search --test test_stale_results -- --nocapture`
  - **Commit**: `test(gpu-search): add deterministic stale result filtering integration tests`
  - _Requirements: AC-1.1, AC-1.2, AC-1.3_
  - _Design: Component A_

- [ ] 3.8 Integration tests for navigation and row expansion
  - **Do**:
    1. In `results_list.rs` tests module, add I-NAV-1 through I-NAV-7 for header-skipping navigation.
    2. Add I-EXP-1 through I-EXP-5 for selected row expansion/collapse and height changes.
    3. Cover: next skips section header, next skips group header, prev skips headers, wrap-around, Tab section jump, empty results, only-headers edge case.
  - **Files**: `gpu-search/src/ui/results_list.rs`
  - **Done when**: 12 navigation + expansion tests pass.
  - **Verify**: `cargo test -p gpu-search results_list -- --nocapture`
  - **Commit**: `test(gpu-search): add keyboard navigation and row expansion integration tests`
  - _Requirements: AC-3.5, AC-6.4_
  - _Design: Component C_

- [ ] 3.9 Unit tests for extension dot colors and status bar
  - **Do**:
    1. Add tests U-DOT-1 through U-DOT-5 in `theme.rs` tests: rs=purple, py=green, unknown=muted, all known non-muted, case sensitivity.
    2. Add tests U-SB-1 through U-SB-4 in `status_bar.rs` tests: count from displayed, searching state, root abbreviation, zero matches.
    3. Add test U-DEB-1 for 100ms debounce default (rename from 30ms test).
  - **Files**: `gpu-search/src/ui/theme.rs`, `gpu-search/src/ui/status_bar.rs`, `gpu-search/src/ui/search_bar.rs`
  - **Done when**: 10 tests pass across theme, status_bar, and search_bar modules.
  - **Verify**: `cargo test -p gpu-search theme status_bar search_bar -- --nocapture`
  - **Commit**: `test(gpu-search): add extension color, status bar, and debounce unit tests`
  - _Requirements: AC-3.4, AC-7.1, AC-7.3_

## Phase 4: Quality Gates

Focus: CI pipeline, benchmark hard gate, clippy, full regression.

- [ ] 4.1 Run full regression test suite
  - **Do**:
    1. Run `cargo test -p gpu-search` -- all existing + new tests must pass.
    2. Run `cargo clippy -p gpu-search -- -D warnings` -- zero warnings.
    3. Fix any clippy or test failures.
    4. Verify the 2 documented test updates: debounce 30->100, verify default Off->Sample.
  - **Files**: Any files with issues
  - **Done when**: All tests pass. Zero clippy warnings.
  - **Verify**: `cargo test -p gpu-search && cargo clippy -p gpu-search -- -D warnings`
  - **Commit**: `fix(gpu-search): address lint/type issues` (if needed)
  - _Requirements: NFR-5, NFR-6_

- [ ] 4.2 Add grouped scroll benchmark
  - **Do**:
    1. Create `gpu-search/benches/grouped_scroll.rs` with Criterion benchmarks.
    2. `bench_rebuild_flat_row_model_100`: 10 groups, 10 matches each.
    3. `bench_rebuild_flat_row_model_10k`: 500 groups, 20 matches each.
    4. `bench_first_visible_row_binary_search`: 10K rows, binary search.
    5. `bench_recompute_groups_incremental_50`: add 50 matches to existing 1000.
    6. Add `[[bench]] name = "grouped_scroll" harness = false` to Cargo.toml.
  - **Files**: `gpu-search/benches/grouped_scroll.rs` (create), `gpu-search/Cargo.toml`
  - **Done when**: Benchmarks run and produce results. `rebuild_flat_row_model` < 1ms for 10K rows.
  - **Verify**: `cargo bench -p gpu-search -- grouped_scroll`
  - **Commit**: `perf(gpu-search): add grouped scroll performance benchmarks`
  - _Requirements: NFR-1, NFR-2, NFR-7_

- [ ] 4.3 Visual verification of all 5 issues
  - **Do**:
    1. Run `cargo run -p gpu-search -- --root ~/gpu_kernel/gpu-search`.
    2. **Issue 1 (Stale)**: Type "orchestrator" rapidly -- no stale results, status bar matches.
    3. **Issue 2 (Formatting)**: Paths show `src/search/...` not `/Users/.../src/search/...`. Hover shows full path.
    4. **Issue 3 (Highlighting)**: Match text has amber overlay on syntax-colored background at exact match position.
    5. **Issue 4 (Verification)**: Search with no env var set -- results are verified (Sample mode active).
    6. **Issue 5 (Grouping)**: Content matches grouped by file with colored dots, headers show match counts, 24px compact / 52px selected, keyboard skips headers.
  - **Files**: None (manual verification)
  - **Done when**: All 5 issues visually confirmed fixed.
  - **Verify**: Manual visual test
  - **Commit**: No commit (verification only)

## Phase 5: PR

Focus: Push branch, create PR, verify CI green.

- [ ] 5.1 Create PR and verify CI
  - **Do**:
    1. Ensure all changes are committed on `feat/gpu-query` branch (or create new branch if needed).
    2. Push branch to remote: `git push -u origin feat/gpu-query`.
    3. Create PR with `gh pr create` targeting `main`.
    4. PR title: "feat(gpu-search): comprehensive UI/UX overhaul (stale results fix, grouped results, match highlighting)"
    5. PR body: summary of 5 issues fixed, link to spec, test plan results.
    6. Wait for CI: `gh pr checks --watch`.
  - **Files**: None (git/GitHub operations)
  - **Done when**: PR created. CI green. Ready for review.
  - **Verify**: `gh pr checks --watch`
  - **Commit**: No commit (PR operations)

## Notes

- **POC shortcuts taken**: Phase 1 wires actions and generation guard without grouped rendering. The flat results list still works -- generation guard is the critical P0 fix.
- **Production TODOs**: Phase 2 is the core rewrite of `results_list.rs`. Dual test suites maintained during migration (old flat-index tests + new RowKind tests). Old tests deleted in a cleanup after Phase 3 verifies new tests.
- **Foreman spec reference**: All task details derived from gpu-search/ai/tasks/spec/{PM,UX,TECH,QA}.md (4 agent analyses + 18 Q&A decisions).
- **egui_kittest**: QA-Q1 approved visual snapshot testing. Can be added as a Phase 4 enhancement if time permits, but not blocking for the initial PR.
