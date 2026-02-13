---
spec: gpu-search-overhaul-2
phase: requirements
created: 2026-02-13
generated: auto
---

# Requirements: gpu-search-overhaul-2

## Summary

Fix the P0 stale results race condition, enable adaptive CPU verification, redesign the results list with grouped-by-file display, path abbreviation, precise match_range highlighting with syntax coloring, and wire existing but unconnected infrastructure (actions, syntax highlighter).

## User Stories

### US-1: Accurate Search Results (P0)

As a developer searching my codebase, I want search results to only show matches for my current query so that I can trust every result displayed.

**Acceptance Criteria**:
- AC-1.1: After typing "kolbey", only results containing "kolbey" are displayed (zero stale prefix results)
- AC-1.2: Status bar match count always equals the number of displayed results
- AC-1.3: Rapid typing (10 chars/sec) produces only final-query results via generation-stamped updates
- AC-1.4: Debounce interval is 100ms (below 200ms perception threshold)

### US-2: Readable File Paths

As a developer scanning results, I want abbreviated file paths so that I can quickly identify files without parsing long absolute paths.

**Acceptance Criteria**:
- AC-2.1: Paths under search root display as relative (e.g., `src/search/orchestrator.rs`)
- AC-2.2: Paths under home dir display with `~` substitution (e.g., `~/Library/...`)
- AC-2.3: Long directory paths (>50 chars) are middle-truncated with `...`
- AC-2.4: Full absolute path shown on hover tooltip and Cmd+C copy

### US-3: Grouped Content Results

As a developer reviewing search results, I want content matches grouped by file so that I can see all matches in a file together with a match count.

**Acceptance Criteria**:
- AC-3.1: Content matches are grouped by file with a header showing filename + match count
- AC-3.2: Groups are always expanded (no collapsible state)
- AC-3.3: Groups sorted by match count descending after search completes
- AC-3.4: 6px colored dot before filename indicates file type (purple .rs, green .py, amber .js, etc.)
- AC-3.5: Compact match rows (24px) expand to 52px on selection showing context before/after

### US-4: Precise Match Highlighting

As a developer evaluating results, I want the exact GPU match range highlighted with syntax coloring so that I can instantly see why a result matched.

**Acceptance Criteria**:
- AC-4.1: Content match highlighting uses `ContentMatch.match_range` (not query re-search)
- AC-4.2: Match lines rendered with syntect syntax coloring + amber match overlay
- AC-4.3: `SyntaxHighlighter` is connected to results renderer (no longer `_highlighter`)
- AC-4.4: File match rows continue to highlight query substring in filename

### US-5: GPU False Positive Filtering

As a developer, I want CPU verification enabled by default so that GPU false positives are filtered before reaching the UI.

**Acceptance Criteria**:
- AC-5.1: Default `VerifyMode` is `Sample` (was `Off`)
- AC-5.2: Auto-upgrades to `Full` when result count < 100
- AC-5.3: `GPU_SEARCH_VERIFY=off` still disables verification
- AC-5.4: Verification overhead < 10% of search time

### US-6: Result Actions

As a developer, I want to open files and copy paths from search results using keyboard shortcuts.

**Acceptance Criteria**:
- AC-6.1: Enter opens selected file via macOS `open` command
- AC-6.2: Cmd+Enter opens in `$EDITOR` at the matched line number
- AC-6.3: Cmd+C copies full absolute path to clipboard
- AC-6.4: Keyboard navigation (Up/Down) skips group headers and section headers

### US-7: Live Search Status

As a developer, I want the status bar to show live search progress so that I know the search is active.

**Acceptance Criteria**:
- AC-7.1: During search: "Searching... | N matches | X.Xs | ~/root | filters"
- AC-7.2: After search: "N matches in X.Xms | root | filters"
- AC-7.3: Status bar root path uses `~` substitution
- AC-7.4: Match count derived from displayed data (never from SearchResponse.total_matches)

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Add `StampedUpdate` wrapper with `generation: u64` field; `poll_updates()` discards stale generations | Must | US-1, PM-Q1 |
| FR-2 | Increase debounce from 30ms to 100ms | Must | US-1, PM-Q4 |
| FR-3 | Add `abbreviate_path()` in new `path_utils.rs`: relative-to-root, `~` substitution, middle truncation | Must | US-2, TECH-Q4 |
| FR-4 | Add `ContentGroup` struct with incremental grouping via `HashMap<PathBuf, usize>` | Must | US-3, TECH-TD3 |
| FR-5 | Add `RowKind` enum + `FlatRowModel` with prefix-sum `cum_heights` for virtual scroll | Must | US-3, TECH-TD1 |
| FR-6 | Replace `show_rows()` with `show_viewport()` rendering using binary search for first visible row | Must | US-3, TECH-TD1 |
| FR-7 | Add `apply_match_range_overlay()` to `highlight.rs` using `ContentMatch.match_range` | Must | US-4, UX-Defect3 |
| FR-8 | Add `render_styled_spans()` to render syntect-colored + match-overlaid spans | Must | US-4 |
| FR-9 | Pass `&mut SyntaxHighlighter` to `results_list.show()` | Must | US-4, TECH-Q1 |
| FR-10 | Change `VerifyMode` default from `Off` to `Sample`; add `effective()` method | Must | US-5, PM-Q2 |
| FR-11 | Wire `OpenFile`, `OpenInEditor`, `CopyPath` in `handle_key_action()` | Should | US-6 |
| FR-12 | Add `extension_dot_color()` to `theme.rs` mapping extensions to Tokyo Night colors | Should | US-3, UX-Q2 |
| FR-13 | Render 6px colored dot before filenames in file rows and group headers | Should | US-3, UX-Q2 |
| FR-14 | Update `StatusBar` with `is_searching` state, live count from displayed data, `~` path | Should | US-7 |
| FR-15 | Keyboard navigation skips `SectionHeader` and `GroupHeader` rows | Must | US-6, UX-Q3 |
| FR-16 | Selected `MatchRow` expands from 24px to 52px showing context_before + match + context_after | Must | US-3, UX-Q1 |
| FR-17 | Sort `content_groups` by match count descending on `SearchUpdate::Complete` | Should | US-3, PM-P1-5 |
| FR-18 | Update `status_bar.update()` call after every `poll_updates()` cycle (not just on Complete) | Must | US-7, PM-P0-3 |
| FR-19 | Add hover tooltip with full absolute path on group headers and file rows | Should | UX-4.3 |
| FR-20 | Tab/Shift+Tab jumps between filename and content sections | Should | UX-7.1 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Rendering path < 5ms per frame with 10K grouped results | Performance |
| NFR-2 | `rebuild_flat_row_model()` < 1ms for 10K rows | Performance |
| NFR-3 | Memory overhead < 2MB for UI grouping structures at 10K results | Memory |
| NFR-4 | Verification overhead < 10% of search time in Sample mode | Performance |
| NFR-5 | All existing 256 unit + 153 integration tests pass (2 with documented updates) | Regression |
| NFR-6 | `cargo clippy -p gpu-search -- -D warnings` zero warnings | Quality |
| NFR-7 | CI benchmark gate: fail if >10% regression | Quality |

## Out of Scope

- GPU kernel/shader changes
- New search algorithms or fuzzy matching
- Network/remote search
- Custom themes or theme switching
- Collapsible group sections (always-expanded for now)
- Preview pane (window too narrow at 720px)
- Multi-select, drag-and-drop, tree view
- Accessibility beyond current WCAG AA
- Search history or bookmarks

## Dependencies

- Existing `eframe` 0.31 `show_viewport()` API
- Existing `syntect` 5 integration in `highlight.rs`
- Existing `crossbeam-channel` 0.5 for `StampedUpdate`
- Existing `SearchGeneration` in `cancel.rs` for generation IDs
- Existing `actions.rs` functions (open_file, open_in_editor)
- `proptest` 1 (dev-dependency, already present) for property tests
- `egui_kittest` (new dev-dependency) for visual snapshot tests
