# gpu-search: Feature Complete

GPU-accelerated filesystem search tool for Apple Silicon.

## Core Features

- **GPU Content Search**: Vectorized uchar4 pattern matching via Metal compute kernels (content_search, turbo_search, batch_search, path_filter). Standard and turbo modes with SIMD prefix sum.
- **Filename Search**: GPU-accelerated path filtering with substring matching and SIMD reduction.
- **Streaming Pipeline**: Quad-buffered (4 x 64MB) streaming search with I/O-compute overlap.
- **Search-as-you-type**: 30ms debounce, 1-char minimum for filename, 2-char for content search.
- **Search Cancellation**: AtomicBool per-search cancellation with generation tracking to discard stale results.

## GPU Engine

- **Metal Device**: objc2-metal 0.3.2 bindings, Apple Silicon capability queries.
- **PSO Cache**: Pre-compiled pipeline state objects for all 4 search kernels.
- **Triple-buffered Work Queue**: CPU-GPU communication with atomic sequence IDs.
- **SearchExecutor**: Background dispatch thread with MTLSharedEvent idle/wake signaling.
- **MTLIOCommandQueue**: Native batch file loading (64 files per command buffer).
- **Zero-copy mmap**: Page-aligned memory-mapped buffers for Metal.

## Filesystem Index

- **Parallel Scanner**: rayon + ignore crate WalkBuilder for multi-threaded directory traversal.
- **GPU-resident Index**: GpuPathEntry (256B) array uploaded to Metal buffer.
- **Binary Persistence**: 64-byte header + packed entry array at ~/.gpu-search/index/.
- **mmap Cache Loading**: Zero-copy index reload via memory-mapped files.
- **Cache Invalidation**: Directory mtime-based staleness detection.

## UI (egui/eframe)

- **Floating Panel**: Borderless 720x400 window (Spotlight-style).
- **Tokyo Night Dark Theme**: WCAG 2.1 AA contrast ratios, no pure black/white.
- **Search Bar**: TextEdit with debounce timer, search icon, filter toggle.
- **Results List**: Virtual scroll (show_rows) for 10K+ items, two sections (filename/content matches).
- **Syntax Highlighting**: syntect with custom Tokyo Night theme, query match overlay (bold amber).
- **Keyboard Navigation**: Up/Down navigate, Enter opens file, Cmd+Enter opens in editor, Escape dismisses, Cmd+Backspace clears.
- **Status Bar**: Match count, elapsed time, search root, active filter count.
- **Filter Pills**: File extension (.rs), exclude directory (!target/), case sensitive toggle. Dismissible.
- **Open-in-Editor**: Detects VS Code, Vim/Neovim, Sublime, Emacs from $VISUAL/$EDITOR/PATH.

## Search Pipeline

- **.gitignore Filtering**: ignore crate with nested .gitignore support, globs, negation.
- **Binary Detection**: NUL byte heuristic (first 8KB) + known binary extension list.
- **Result Ranking**: Word boundary > partial match, path length tiebreak.
- **Progressive Delivery**: Wave 1 (FileMatches) -> Wave 2 (ContentMatches) -> Complete.
- **Error Handling**: 9 error variants with recovery actions (Fatal, FallbackToCpu, SkipFile, etc.).

## Stats

- 249 unit tests passing
- ~8,000 lines of Rust + Metal shader code
- 4 Metal compute kernels
- 6 modules: gpu, search, io, index, ui, error
