---
id: ui.BREAKDOWN
module: ui
priority: 3
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN, search-core.BREAKDOWN]
tags: [gpu-search]
testRequirements:
  unit:
    required: true
---

# UI Module Breakdown

## Context

Build the egui floating panel UI for gpu-search. This is a Spotlight-style floating window with a search input, real-time results list (filename matches + content matches with syntax highlighting), keyboard-first navigation, filter pills, and a status bar. Uses eframe (egui + winit + wgpu with Metal backend) for rendering. Dark theme only for v1 (Tokyo Night color scheme).

The UI communicates with the search-core module via channels: submits `SearchRequest` on input change (with 30ms debounce), receives progressive `SearchUpdate` messages (filename matches first, then content matches).

## Tasks

### T-040: Implement ui/app.rs -- egui Application state

Main application struct implementing `eframe::App`:

```rust
pub struct GpuSearchApp {
    search_input: String,
    file_matches: Vec<FileMatch>,
    content_matches: Vec<ContentMatch>,
    selected_index: usize,
    search_tx: Sender<SearchRequest>,
    result_rx: Receiver<SearchUpdate>,
    debounce_timer: Option<Instant>,
    search_stats: SearchStats,
    active_filters: Vec<FilterPill>,
}
```

- `update()` method: check for new results, render UI, handle input
- `eframe::run_native()` launch with window configuration
- Floating panel window: 720px wide, dynamic height (120-800px)
- Centered horizontally, 20% from top

**Target**: `gpu-search/src/ui/app.rs`
**Verify**: `cargo build -p gpu-search` -- binary launches, shows empty search window

### T-041: Implement ui/search_bar.rs -- search input with debounce

Search input widget with:
- Single-line `egui::TextEdit` with placeholder "Search files and content..."
- 30ms debounce: start timer on keystroke, fire search when timer expires or new keystroke
- Cancel in-flight search on new input
- Search icon (left) and filter toggle button (right)
- Minimum query length: 1 char for filename, 2 chars for content search

**Target**: `gpu-search/src/ui/search_bar.rs`
**Verify**: `cargo test -p gpu-search test_debounce_logic` -- debounce fires at correct interval

### T-042: Implement ui/results_list.rs -- scrollable results

Two-section results display:

**Filename Matches Section:**
- Header: "FILENAME MATCHES (N)" with count
- Each result: filename (highlighted match) + parent path (dimmed)
- Selected item: background highlight + left accent border

**Content Matches Section:**
- Header: "CONTENT MATCHES (N)"
- Each result: file:line header + 3 lines (1 before, match line, 1 after)
- Match line: syntax highlighted with query match in bold amber/gold
- Context lines: dimmed (50% opacity)
- Line numbers in gutter

Virtual scroll: render only visible items. Cap at 10K results with "N more..." indicator.

**Target**: `gpu-search/src/ui/results_list.rs`
**Verify**: `cargo test -p gpu-search test_results_rendering` -- results display correctly with mock data

### T-043: Implement ui/theme.rs -- Tokyo Night dark theme

Dark theme color constants per UX.md Section 7.1:

```rust
pub const BG_BASE: Color32 = Color32::from_rgb(0x1A, 0x1B, 0x26);      // #1A1B26
pub const BG_SURFACE: Color32 = Color32::from_rgb(0x24, 0x26, 0x3A);   // #24263A
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xC0, 0xCA, 0xF5); // #C0CAF5
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x56, 0x5F, 0x89);   // #565F89
pub const ACCENT: Color32 = Color32::from_rgb(0xE0, 0xAF, 0x68);       // #E0AF68 (match highlight)
pub const ERROR: Color32 = Color32::from_rgb(0xF7, 0x76, 0x8E);        // #F7768E
pub const SUCCESS: Color32 = Color32::from_rgb(0x9E, 0xCE, 0x6A);      // #9ECE6A
pub const BORDER: Color32 = Color32::from_rgb(0x3B, 0x3E, 0x52);       // #3B3E52
```

Apply to egui `Visuals`:
- Window background, widget backgrounds, text colors
- Selection highlight using ACCENT color
- All text WCAG 2.1 AA contrast (4.5:1 minimum)

**Target**: `gpu-search/src/ui/theme.rs`
**Verify**: Visual inspection -- theme applied correctly, no pure black/white

### T-044: Implement ui/keybinds.rs -- keyboard shortcuts

Keyboard-first navigation per UX.md Section 6.1:

| Action | Key | Handler |
|--------|-----|---------|
| Navigate results | Up/Down | Move `selected_index`, scroll into view |
| Open file at match | Enter | Launch file in default editor |
| Open in $EDITOR | Cmd+Enter | Launch file at line number in $EDITOR |
| Close panel | Escape | `frame.close()` or hide window |
| Clear search | Cmd+Backspace | Clear input, clear results |
| Copy file path | Cmd+C | Copy selected result path to clipboard |
| Cycle filters | Tab | Focus moves through filter pills |

**Target**: `gpu-search/src/ui/keybinds.rs`
**Verify**: `cargo test -p gpu-search test_keybind_actions` -- key events produce correct actions

### T-045: Implement syntax highlighting with syntect

Syntax-highlighted code lines in content match results:
- Use `syntect` with Sublime Text syntax definitions
- Custom dark theme matching Tokyo Night colors (4-5 syntax colors max)
- Query match highlighting: bold + amber background (overrides syntax colors)
- Cache: per-file parse state cached every 1000 lines
- Only highlight visible context lines (not entire file)

**Target**: `gpu-search/src/ui/highlight.rs`
**Verify**: `cargo test -p gpu-search test_syntax_highlight` -- Rust code correctly highlighted with expected colors

### T-046: Implement filter UI (pills/chips)

Filter pills below search bar when active:

```
[.rs] [.metal] [x target/]
```

- File extension filter: click "Filters" or type `ext:rs` in search bar
- Exclude directory: type `!target/` or via Filters menu
- Case sensitive toggle: Cmd+I or prefix `Cc`
- Each pill is dismissible (click X)
- Filters applied before GPU search dispatch

**Target**: `gpu-search/src/ui/filters.rs`
**Verify**: `cargo test -p gpu-search test_filter_pills` -- filter pills add/remove correctly, affect search request

### T-047: Implement status bar

Bottom bar showing ambient information:

```
12 files | 847 matches | 0.8ms GPU | ~/gpu_kernel | .rs .metal
```

- Match count (files and total matches)
- Search timing (from SearchResponse.elapsed)
- Active search root
- Active file type filters
- Update on every SearchUpdate

**Target**: `gpu-search/src/ui/status_bar.rs`
**Verify**: `cargo test -p gpu-search test_status_bar` -- displays correct stats from mock SearchResponse

### T-048: Implement panel animations

Subtle animations per UX.md Section 7.4:

| Animation | Duration | Easing |
|-----------|----------|--------|
| Panel appear | 150ms | ease-out |
| Panel dismiss | 100ms | ease-in |
| Results fade-in | 80ms | ease-out |
| Selection highlight | 60ms | linear |

Use egui's animation support (`ctx.animate_value_with_time()`). No animation on search input text changes.

**Target**: Integrated into `app.rs` and `results_list.rs`
**Verify**: Visual inspection -- animations are smooth, not jarring

## Acceptance Criteria

1. Floating panel appears with search input, results list, and status bar
2. Search-as-you-type: typing triggers search after 30ms debounce
3. Progressive results: filename matches appear first, content matches follow
4. Keyboard navigation: Up/Down moves selection, Enter opens file, Escape dismisses
5. Syntax highlighting on content match lines (via syntect, 4-5 colors)
6. Query match highlighted with bold + amber background
7. Tokyo Night dark theme applied to all UI elements
8. Filter pills for file type and directory exclusion
9. Virtual scroll handles 10K+ results at 60fps
10. Status bar shows match count, timing, active root, active filters
11. Panel dimensions: 720px wide, 120-800px dynamic height
12. All text meets WCAG 2.1 AA contrast (4.5:1 ratio)

## Technical Notes

- **egui immediate mode**: Entire UI redraws every frame. State is in `GpuSearchApp` struct. No retained widget trees.
- **Font selection**: Try SF Pro + SF Mono (system fonts). Fall back to bundled JetBrains Mono + Inter if system fonts unavailable from egui.
- **Virtual scroll**: egui's `ScrollArea` with `show_rows()` for lazy rendering. Only render visible items.
- **Channel integration**: `result_rx.try_recv()` in `update()` loop -- non-blocking check for new results every frame.
- **Global hotkey**: Deferred to integration module. For now, panel shows on app launch.
- **Screenshot tests**: Use `egui_kittest` with `snapshot` feature for visual regression testing (QA.md Section 9).
- **wgpu Metal backend**: eframe automatically selects Metal backend on macOS. No manual configuration needed.
- Reference: UX.md Sections 2-7, QA.md Section 9, TECH.md Section 4
