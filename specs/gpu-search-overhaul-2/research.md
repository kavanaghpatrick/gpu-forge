---
spec: gpu-search-overhaul-2
phase: research
created: 2026-02-13
generated: auto
---

# Research: gpu-search-overhaul-2

## Executive Summary

Comprehensive UI/UX overhaul of gpu-search addressing 5 issues: a P0 stale results race condition, result formatting with path abbreviation, match highlighting via GPU match_range, adaptive CPU verification, and grouped-by-file results with virtual scroll. Feasibility is high -- all required infrastructure exists in the codebase, and the foreman-spec Deep Mode analysis (PM, UX, TECH, QA agents with 18 consolidated Q&A decisions) provides a complete technical blueprint.

## Codebase Analysis

### Existing Patterns

| Pattern | File | Relevance |
|---------|------|-----------|
| `SearchUpdate` enum (FileMatches, ContentMatches, Complete) | `src/search/types.rs` | Wrap with `StampedUpdate` for generation stamping |
| `SearchGeneration` + `CancellationHandle` + `SearchSession` | `src/search/cancel.rs` | Generation tracking exists; need to propagate to update messages |
| Crossbeam `Sender<SearchUpdate>` / `Receiver<SearchUpdate>` | `src/ui/app.rs:74,131` | Change to `StampedUpdate` channel type |
| `SyntaxHighlighter` with `apply_match_overlay()` | `src/ui/highlight.rs` | Complete but unused (`_highlighter` prefix). Wire to results renderer |
| `ContentMatch.match_range: Range<usize>` | `src/search/types.rs:68` | GPU match range exists but ignored by renderer |
| `show_rows()` uniform-height virtual scroll | `src/ui/results_list.rs:173+` | Replace with `show_viewport()` + prefix-sum heights |
| `VerifyMode::Off` default | `src/search/verify.rs:28` | Change to `Sample` with adaptive `effective()` |
| `DEFAULT_DEBOUNCE_MS: u64 = 30` | `src/ui/search_bar.rs` | Increase to 100ms |
| `open_file()` / `open_in_editor()` | `src/ui/actions.rs` | Built but unconnected; wire to keybinds |
| Tokyo Night theme constants | `src/ui/theme.rs` | Add `extension_dot_color()` function |
| `fm.path.display().to_string()` raw paths | `src/ui/results_list.rs:266,338` | Replace with `abbreviate_path()` |
| `render_highlighted_text()` query re-search | `src/ui/results_list.rs:396-485` | Replace with `render_styled_spans()` using match_range |

### Dependencies (Already in Cargo.toml)

- `eframe` 0.31 -- egui framework, has `show_viewport()` API
- `syntect` 5 -- syntax highlighting (integrated in `highlight.rs`)
- `crossbeam-channel` 0.5 -- thread-safe channels
- `memchr` 2 -- SIMD verification (`memmem`)
- `dirs` 6 -- already present; but `$HOME` env preferred for path abbreviation (per TECH-Q2)
- `proptest` 1 -- dev-dependency for property tests

### Constraints

- macOS Apple Silicon only (Metal compute shaders)
- egui `show_rows()` requires uniform height -- must switch to `show_viewport()` for mixed row heights
- `SyntaxHighlighter` needs `&mut self` -- pass as parameter to avoid borrow conflicts (TECH-Q1)
- Single-line syntect parsing per match row (95%+ accurate, avoids cross-file state caching)
- 16.6ms frame budget (60fps), rendering path must stay < 5ms

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | **High** | All infrastructure exists; changes are additive or rewiring |
| Effort Estimate | **L** (6-8 days) | `results_list.rs` near-complete rewrite is the dominant cost |
| Risk Level | **Medium** | `results_list.rs` rewrite + test migration is highest risk; mitigated by dual test suites |

### Risk Breakdown

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `results_list.rs` rewrite breaks 11 existing tests | High | High | Dual test suites during migration |
| Variable-height virtual scroll jank | Medium | High | Debounce rebuild; benchmark 10K rows |
| `SyntaxHighlighter` borrow conflict | High | Medium | Pass `&mut` parameter (TECH-Q1 decision) |
| `match_range` byte offsets invalid after truncation | Medium | Medium | Clamp range; apply overlay before truncation |
| VerifyMode default change breaks existing test | Certain | Low | Update `test_verify_mode_from_env` assertion |

## Key Foreman-Spec Decisions (Summary)

18 consolidated Q&A decisions from PM, UX, TECH, QA agents:

| Area | Key Decisions |
|------|--------------|
| PM | Generation-stamped updates; Adaptive verify (Sample/Full < 100); 100ms debounce; $EDITOR fallback |
| UX | 6px colored dot (not badge); context only on selected row (24/52px); skip headers in nav; status bar only for progress |
| TECH | `show_viewport()` + prefix-sum; `StampedUpdate` wrapper; incremental HashMap grouping; `$HOME` env; fresh parse per line |
| QA | egui_kittest snapshots; synchronous stale-results harness; dual test suites; 3 property tests; hard 10% benchmark gate |

## Recommendations

1. Implement P0 stale results fix first (StampedUpdate + generation guard) -- unblocks all other work
2. Build data model layer (ContentGroup, RowKind, FlatRowModel) before touching rendering
3. Keep dual test suites during results_list.rs migration
4. Add path_utils.rs as shared module (used by both results_list and status_bar)
5. Wire existing actions.rs/highlight.rs infrastructure before writing new code
