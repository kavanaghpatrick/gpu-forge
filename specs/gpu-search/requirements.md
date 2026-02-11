---
spec: gpu-search
phase: requirements
created: 2026-02-11
generated: auto
---

# Requirements: gpu-search

## Summary

Standalone GPU-accelerated filesystem search tool at `gpu_kernel/gpu-search/`. Spotlight-style floating panel with search-as-you-type (<5ms cached), 55-80 GB/s Metal compute throughput, filename AND content search, .gitignore respected, fixed string matching for v1.

## User Stories

### US-1: Search-as-you-type with GPU acceleration
As a developer, I want to type a search query and see results updating in real-time (<5ms for cached files) so that search feels instantaneous.

**Acceptance Criteria**:
- AC-1.1: Results appear within 5ms of last keystroke for cached/indexed files
- AC-1.2: 30ms debounce prevents redundant GPU dispatches during fast typing
- AC-1.3: New keystroke cancels in-flight search, starts fresh search
- AC-1.4: Progressive results: filename matches first (~0-2ms), content matches second (~2-5ms)

### US-2: Floating search panel (Spotlight-style)
As a developer, I want a floating panel that appears on Cmd+Shift+F and disappears on Escape so that search integrates into my workflow without context switching.

**Acceptance Criteria**:
- AC-2.1: Panel appears on global hotkey Cmd+Shift+F from any application
- AC-2.2: Panel persists until Escape (Alfred model for iterative exploration)
- AC-2.3: Panel dimensions: 720px wide, 120-800px dynamic height
- AC-2.4: Centered horizontally, offset 20% from top
- AC-2.5: 150ms appear animation, 100ms dismiss animation

### US-3: Content search with syntax highlighting
As a developer, I want content match results with syntax-highlighted context lines so that I can understand matches in context without opening the file.

**Acceptance Criteria**:
- AC-3.1: Content matches show match line + 1 context line above and below
- AC-3.2: Syntax highlighting via syntect (4-5 colors, Tokyo Night theme)
- AC-3.3: Query match highlighted with bold + amber background (#E0AF68)
- AC-3.4: Line numbers shown in gutter, dimmed
- AC-3.5: Language badge shown per result (e.g., [Rust], [Python])

### US-4: Filename and path search
As a developer, I want to search by filename/path in addition to content so that I can quickly navigate to files by name.

**Acceptance Criteria**:
- AC-4.1: Filename matches shown in separate section above content matches
- AC-4.2: Query substring highlighted in filename with accent color
- AC-4.3: Parent path shown dimmed next to filename
- AC-4.4: Filename search from 1 character, content search from 2+ characters

### US-5: Keyboard-first navigation
As a developer, I want full keyboard control of search results so that I never need to reach for the mouse.

**Acceptance Criteria**:
- AC-5.1: Up/Down arrows navigate results, scroll into view
- AC-5.2: Enter opens file in default application
- AC-5.3: Cmd+Enter opens file at line number in $EDITOR
- AC-5.4: Escape dismisses panel
- AC-5.5: Cmd+C copies selected result file path
- AC-5.6: Tab cycles through filter pills

### US-6: File type and ignore filtering
As a developer, I want to filter search by file type and respect .gitignore so that results are relevant and exclude build artifacts.

**Acceptance Criteria**:
- AC-6.1: .gitignore patterns respected by default (like ripgrep)
- AC-6.2: File type filter via pills (e.g., `.rs`, `.metal`) or `ext:rs` syntax
- AC-6.3: Directory exclusion via `!target/` syntax
- AC-6.4: Binary files skipped by default (NUL byte heuristic)
- AC-6.5: Symlinks not followed by default

### US-7: GPU-accelerated content search pipeline
As a developer, I want content search to run at 55-80 GB/s on Apple Silicon so that even large codebases return results in milliseconds.

**Acceptance Criteria**:
- AC-7.1: Vectorized uchar4 Metal kernel achieves 55-80 GB/s throughput
- AC-7.2: MTLIOCommandQueue batch loads 10K files in <30ms
- AC-7.3: Streaming pipeline overlaps I/O with compute (quad-buffered)
- AC-7.4: First search (cold, 10K files) completes in <200ms
- AC-7.5: GPU-CPU dual verification confirms identical results

### US-8: Open results in editor
As a developer, I want to open a search result directly in my editor at the matching line so that I can jump straight to the relevant code.

**Acceptance Criteria**:
- AC-8.1: Enter opens file in default application
- AC-8.2: Cmd+Enter opens in $EDITOR at line number
- AC-8.3: Supports VS Code (`code --goto`), Vim (`+line`), Sublime (`file:line`)
- AC-8.4: Falls back to `open <file>` if editor unknown

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Vectorized uchar4 Metal content search kernel | Must | US-7 |
| FR-2 | MTLIOCommandQueue GPU-direct batch file loading | Must | US-7 |
| FR-3 | Streaming quad-buffered I/O + search pipeline | Must | US-7 |
| FR-4 | GPU-resident filesystem index (256B/entry, mmap) | Must | US-4, US-7 |
| FR-5 | Triple-buffered work queue (CPU->GPU handoff) | Must | US-1, US-7 |
| FR-6 | Completion-handler re-dispatch chain (persistent search) | Must | US-7 |
| FR-7 | egui floating panel with search input | Must | US-2 |
| FR-8 | Search-as-you-type with 30ms debounce | Must | US-1 |
| FR-9 | Progressive result delivery (filename then content) | Must | US-1 |
| FR-10 | Syntax highlighting on content matches (syntect) | Must | US-3 |
| FR-11 | Keyboard navigation (Up/Down/Enter/Escape) | Must | US-5 |
| FR-12 | .gitignore filtering (ignore crate) | Must | US-6 |
| FR-13 | Binary file detection and skip (NUL byte) | Must | US-6 |
| FR-14 | File type filtering (extension-based pills) | Should | US-6 |
| FR-15 | Search cancellation on new input | Should | US-1 |
| FR-16 | Global hotkey (Cmd+Shift+F) | Should | US-2 |
| FR-17 | Open-in-editor action (Cmd+Enter) | Should | US-8 |
| FR-18 | Virtual scroll for 10K+ results | Should | US-1 |
| FR-19 | Dark theme (Tokyo Night colors) | Must | US-2 |
| FR-20 | Status bar (match count, timing, search root) | Should | US-1 |
| FR-21 | Filter pills UI (file type, exclude directory) | Should | US-6 |
| FR-22 | Index persistence at ~/.gpu-search/ | Should | US-4 |
| FR-23 | Incremental index updates (notify crate) | Could | US-4 |
| FR-24 | MTLSharedEvent idle/wake power management | Could | US-7 |

## Non-Functional Requirements

| ID | Requirement | Category | Target |
|----|-------------|----------|--------|
| NFR-1 | Cached incremental search latency | Performance | <5ms p50 |
| NFR-2 | Content search throughput | Performance | 55-80 GB/s |
| NFR-3 | Cold first search (10K files) | Performance | <200ms |
| NFR-4 | Batch I/O (10K files via MTLIOCommandQueue) | Performance | <30ms |
| NFR-5 | Index load from cache (mmap) | Performance | <10ms |
| NFR-6 | UI frame time (60fps) | Performance | <16ms p99 |
| NFR-7 | GPU memory budget | Resource | <500MB for 100K files |
| NFR-8 | GPU allocation growth (10K searches) | Stability | <1% |
| NFR-9 | Watchdog survival (continuous operation) | Stability | >60 seconds zero errors |
| NFR-10 | Text contrast ratio | Accessibility | WCAG 2.1 AA (4.5:1) |
| NFR-11 | Search result correctness (GPU vs CPU) | Correctness | 100% match |
| NFR-12 | Benchmark regression threshold | Quality | <10% per commit |

## Out of Scope

- Regex search (Phase 2 enhancement)
- Light theme (dark-only for v1)
- Full-window mode with file preview pane (v2)
- Cross-platform (macOS + Apple Silicon only)
- Multi-GPU support
- Network/remote filesystem search
- Natural language / fuzzy search
- File content indexing (rely on GPU raw speed)

## Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| rust-experiment GPU code (~8K lines) | Internal, port source | Complete -- proven at 55-80 GB/s |
| gpu-query autonomous patterns | Internal, reference | Complete -- triple-buffer, re-dispatch, MTLSharedEvent |
| objc2-metal 0.3 | External crate | Stable, published |
| eframe 0.31 | External crate | Stable, published |
| syntect 5 | External crate | Stable, published |
| ignore 0.4 | External crate | Stable (ripgrep's crate) |
| Apple Silicon hardware | Hardware | Required (M1+) |
| macOS 14+ | OS | Required for Metal 3 features |
