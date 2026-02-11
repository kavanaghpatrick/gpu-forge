---
spec: gpu-search
phase: tasks
total_tasks: 52
created: 2026-02-11
generated: auto
---

# Tasks: gpu-search

## Phase 1: Make It Work (POC)

Focus: Project scaffold + core GPU engine + basic content search working end-to-end. Skip tests, accept hardcoded values. Validate the port works and search returns correct results.

- [x] 1.1 Create project directory structure
  - **Do**: Create `gpu_kernel/gpu-search/` with full directory layout: `src/{gpu,search,io,index,ui}/mod.rs`, `shaders/`, `tests/`, `benches/`. Create empty mod.rs in each module directory.
  - **Files**: `gpu-search/Cargo.toml`, `gpu-search/src/main.rs`, `gpu-search/src/lib.rs`, `gpu-search/src/gpu/mod.rs`, `gpu-search/src/search/mod.rs`, `gpu-search/src/io/mod.rs`, `gpu-search/src/index/mod.rs`, `gpu-search/src/ui/mod.rs`, `gpu-search/shaders/`, `gpu-search/tests/`, `gpu-search/benches/`
  - **Done when**: `ls -R gpu_kernel/gpu-search/src/` shows all module directories with mod.rs files
  - **Verify**: `ls -R gpu_kernel/gpu-search/src/`
  - **Commit**: `feat(gpu-search): create project directory structure`
  - _Requirements: FR-1 through FR-24_
  - _Design: File Structure_

- [x] 1.2 Create Cargo.toml with all dependencies
  - **Do**: Write Cargo.toml with package metadata, all dependencies (objc2-metal 0.3 with feature flags, eframe 0.31, syntect 5, rayon, ignore, dirs, memchr, libc, serde/serde_json, block2, dispatch2, objc2, objc2-foundation), dev-dependencies (tempfile, criterion, proptest), and bench targets (search_throughput, io_latency).
  - **Files**: `gpu-search/Cargo.toml`
  - **Done when**: `cargo check` resolves all dependencies (may fail on missing src, that's ok)
  - **Verify**: `cd gpu_kernel/gpu-search && cargo check 2>&1 | head -20`
  - **Commit**: `feat(gpu-search): add Cargo.toml with all dependencies`
  - _Requirements: FR-1, FR-7, FR-10, FR-12_
  - _Design: Dependency Plan_

- [x] 1.3 Create build.rs for Metal shader compilation
  - **Do**: Port gpu-query's `build.rs` pattern. Discover `.metal` files in `shaders/`, compile each to `.air` via `xcrun -sdk macosx metal -c -I shaders/`, link all `.air` into `shaders.metallib`, copy to OUT_DIR. Add `cargo:rerun-if-changed` for each .metal and .h file.
  - **Files**: `gpu-search/build.rs`
  - **Done when**: `cargo build` compiles Metal shaders without errors
  - **Verify**: `cd gpu_kernel/gpu-search && cargo build 2>&1 | grep -i metal`
  - **Commit**: `feat(gpu-search): add build.rs for Metal shader compilation`
  - _Requirements: FR-1_
  - _Design: GPU Device Layer_

- [x] 1.4 Create placeholder Metal shader files
  - **Do**: Create stub `.metal` files with `#include <metal_stdlib>` and empty kernel functions. Create `search_types.h` with basic shared type definitions (SearchParams, GpuMatchResult, GpuPathEntry stubs).
  - **Files**: `gpu-search/shaders/search_types.h`, `gpu-search/shaders/content_search.metal`, `gpu-search/shaders/turbo_search.metal`, `gpu-search/shaders/batch_search.metal`, `gpu-search/shaders/path_filter.metal`
  - **Done when**: `cargo build` compiles all shaders
  - **Verify**: `cd gpu_kernel/gpu-search && cargo build`
  - **Commit**: `feat(gpu-search): add placeholder Metal shader files`
  - _Requirements: FR-1_
  - _Design: GPU Device Layer_

- [x] 1.5 Create stub main.rs and lib.rs
  - **Do**: `main.rs`: minimal entry point that prints "gpu-search" and exits (eframe launch comes later). `lib.rs`: declare modules `pub mod gpu; pub mod search; pub mod io; pub mod index; pub mod ui;`. Each module's `mod.rs`: empty or with `// TODO` comments.
  - **Files**: `gpu-search/src/main.rs`, `gpu-search/src/lib.rs`
  - **Done when**: `cargo build` produces a binary
  - **Verify**: `cd gpu_kernel/gpu-search && cargo build && cargo run`
  - **Commit**: `feat(gpu-search): add stub main.rs and lib.rs`
  - _Requirements: FR-7_
  - _Design: Components_

- [x] 1.6 Port gpu/device.rs -- Metal device initialization
  - **Do**: Copy gpu-query `gpu/device.rs` pattern. `MTLCreateSystemDefaultDevice()` init, device capability queries (max threadgroup size, max buffer length), device name validation. Export `GpuDevice` struct with `Retained<ProtocolObject<dyn MTLDevice>>`.
  - **Files**: `gpu-search/src/gpu/device.rs`, update `gpu-search/src/gpu/mod.rs`
  - **Done when**: Device initializes and reports Apple GPU name
  - **Verify**: `cd gpu_kernel/gpu-search && cargo run` prints device name
  - **Commit**: `feat(gpu-search): port Metal device initialization`
  - _Requirements: FR-1, FR-5_
  - _Design: GPU Device Layer_

- [x] 1.7 Create gpu/types.rs -- shared repr(C) types
  - **Do**: Define `#[repr(C)]` structs matching `search_types.h`: SearchParams (pattern bytes, pattern_len, case_sensitive, total_bytes), GpuMatchResult (file_index, line_number, column, match_length, context), GpuPathEntry (256B: path[224], path_len, flags, parent_idx, size, mtime). Add compile-time `assert!(std::mem::size_of::<GpuPathEntry>() == 256)` and offset assertions.
  - **Files**: `gpu-search/src/gpu/types.rs`, update `gpu-search/src/gpu/mod.rs`
  - **Done when**: Type layout assertions compile and pass
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_types_layout`
  - **Commit**: `feat(gpu-search): add shared repr(C) GPU types with layout assertions`
  - _Requirements: FR-1, FR-4_
  - _Design: GPU Device Layer, types.rs_

- [x] 1.8 Port gpu/pipeline.rs -- PSO cache
  - **Do**: Port gpu-query `pipeline.rs`. Load `shaders.metallib` from build output via `device.newLibraryWithURL()`. HashMap PSO cache: function name -> `MTLComputePipelineState`. Create PSOs for content_search, turbo_search, batch_search, path_filter kernels.
  - **Files**: `gpu-search/src/gpu/pipeline.rs`, update `gpu-search/src/gpu/mod.rs`
  - **Done when**: PSOs created for all 4 search kernels without error
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_pso_cache`
  - **Commit**: `feat(gpu-search): port PSO cache for search kernels`
  - _Requirements: FR-1_
  - _Design: GPU Device Layer, pipeline.rs_

- [x] 1.9 Extract Metal shaders from rust-experiment
  - **Do**: Extract MSL source from inline Rust string constants: `CONTENT_SEARCH_SHADER` from `content_search.rs:95`, turbo kernel from `content_search.rs`, `PERSISTENT_SEARCH_SHADER` from `persistent_search.rs:173`, path filter from `filesystem.rs`. Place into standalone `.metal` files. Update `search_types.h` with actual struct definitions matching the Rust types. Remove the `{{APP_SHADER_HEADER}}` template and replace with actual includes.
  - **Files**: `gpu-search/shaders/content_search.metal`, `gpu-search/shaders/turbo_search.metal`, `gpu-search/shaders/batch_search.metal`, `gpu-search/shaders/path_filter.metal`, `gpu-search/shaders/search_types.h`
  - **Done when**: `cargo build` compiles all extracted shaders, PSOs load successfully
  - **Verify**: `cd gpu_kernel/gpu-search && cargo build && cargo test test_pso_cache`
  - **Commit**: `feat(gpu-search): extract Metal shaders from rust-experiment`
  - _Requirements: FR-1_
  - _Design: Kernel Architecture_

- [x] 1.10 Port io/mmap.rs -- zero-copy mmap buffers
  - **Do**: Port `rust-experiment/src/gpu_os/mmap_buffer.rs` (566 lines). Change `Device` -> `ProtocolObject<dyn MTLDevice>`, `device.new_buffer_with_bytes_no_copy()` -> `device.newBufferWithBytesNoCopy_length_options_deallocator()`. Preserve mmap logic (libc::mmap, munmap). Test: mmap a file, create Metal buffer, read contents back.
  - **Files**: `gpu-search/src/io/mmap.rs`, update `gpu-search/src/io/mod.rs`
  - **Done when**: File mmapped into Metal buffer, contents readable
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_mmap_buffer`
  - **Commit**: `feat(gpu-search): port zero-copy mmap buffer to objc2-metal`
  - _Requirements: FR-4_
  - _Design: File I/O, mmap.rs_

- [ ] 1.11 Port io/gpu_io.rs -- MTLIOCommandQueue
  - **Do**: Port `rust-experiment/src/gpu_os/gpu_io.rs` (524 lines). **Major**: Replace ALL raw `msg_send!` with native objc2-metal `MTLIOCommandQueue` bindings. Enable `MTLIOCommandQueue` feature flag. Create IO command queue, load files to GPU buffers. If specific methods missing from bindings, use `msg_send!` for those only and document.
  - **Files**: `gpu-search/src/io/gpu_io.rs`, update `gpu-search/src/io/mod.rs`
  - **Done when**: IO queue created, file loaded to GPU buffer without raw msg_send! (except documented gaps)
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_io_command_queue`
  - **Commit**: `feat(gpu-search): port MTLIOCommandQueue to native objc2-metal bindings`
  - _Requirements: FR-2_
  - _Design: File I/O, gpu_io.rs_

- [ ] 1.12 Port io/batch.rs -- batch file loading
  - **Do**: Port `rust-experiment/src/gpu_os/batch_io.rs` (514 lines). Port `BatchLoadResult` and `BatchLoadHandle` types to objc2-metal. Batch multiple files into single MTLIOCommandBuffer for parallel loading.
  - **Files**: `gpu-search/src/io/batch.rs`, update `gpu-search/src/io/mod.rs`
  - **Done when**: 100 files loaded via MTLIOCommandQueue, contents verified
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_batch_load`
  - **Commit**: `feat(gpu-search): port batch file loading to objc2-metal`
  - _Requirements: FR-2_
  - _Design: File I/O, batch.rs_

- [ ] 1.13 Port search/content.rs -- content search orchestration
  - **Do**: Port `rust-experiment/src/gpu_os/content_search.rs` (1660 lines). Content search dispatch logic: buffer allocation for input/output/match positions, kernel dispatch with threadgroup sizing (256 threads, 64 bytes/thread), result collection. Both standard and turbo modes. Test with known pattern in known content, verify match count and positions against CPU reference.
  - **Files**: `gpu-search/src/search/content.rs`, update `gpu-search/src/search/mod.rs`
  - **Done when**: GPU search for known pattern returns correct match count matching CPU reference
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_content_search`
  - **Commit**: `feat(gpu-search): port vectorized content search kernel dispatch`
  - _Requirements: FR-1, NFR-2_
  - _Design: Search Engine, content.rs_

- [ ] 1.14 Port search/streaming.rs -- streaming pipeline
  - **Do**: Port `rust-experiment/src/gpu_os/streaming_search.rs` (1071 lines). Quad-buffered streaming: 4 x 64MB StreamChunks, I/O + search overlap pipeline, AtomicBool per chunk for ready signaling. Test with >64MB test data.
  - **Files**: `gpu-search/src/search/streaming.rs`, update `gpu-search/src/search/mod.rs`
  - **Done when**: Stream search through >64MB data returns correct results
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_streaming_search`
  - **Commit**: `feat(gpu-search): port quad-buffered streaming search pipeline`
  - _Requirements: FR-3, NFR-2_
  - _Design: Search Engine, streaming.rs_

- [ ] 1.15 Define search API types
  - **Do**: Create SearchRequest (pattern, root, file_types, case_sensitive, respect_gitignore, include_binary, max_results), FileMatch (path, score), ContentMatch (path, line_number, line_content, context_before, context_after, match_range), SearchResponse (file_matches, content_matches, total_files_searched, total_matches, elapsed).
  - **Files**: `gpu-search/src/search/types.rs`, update `gpu-search/src/search/mod.rs`
  - **Done when**: Types compile and are usable from other modules
  - **Verify**: `cd gpu_kernel/gpu-search && cargo check`
  - **Commit**: `feat(gpu-search): define search API types`
  - _Requirements: FR-8, FR-9_
  - _Design: Search Engine, types.rs_

- [ ] 1.16 POC Checkpoint: end-to-end GPU search
  - **Do**: Wire together: load a test directory of files via batch I/O, search for a known pattern via content search kernel, collect results, print to stdout. Verify GPU results match CPU `grep` output. This validates the entire GPU pipeline works after the port.
  - **Files**: `gpu-search/src/main.rs` (temporary POC wiring)
  - **Done when**: `cargo run -- "pattern" ./test-dir` prints correct search results matching `grep -rn "pattern" ./test-dir`
  - **Verify**: `cd gpu_kernel/gpu-search && cargo run -- "test_pattern" ./src/ 2>&1 | head -20`
  - **Commit**: `feat(gpu-search): complete POC -- end-to-end GPU search working`
  - _Requirements: FR-1, FR-2, FR-3, NFR-2, NFR-11_
  - _Design: Data Flow_

## Phase 2: Feature Complete

After POC validated, build out full feature set: search orchestrator, UI, index, filters.

- [ ] 2.1 Port index/gpu_index.rs -- GPU-resident path index
  - **Do**: Port `rust-experiment/src/gpu_os/gpu_index.rs` (867 lines). GpuPathEntry (256B), GpuResidentIndex struct. Port to objc2-metal types.
  - **Files**: `gpu-search/src/index/gpu_index.rs`, update `gpu-search/src/index/mod.rs`
  - **Done when**: Index built from test directory, loaded into GPU buffer
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_gpu_index`
  - **Commit**: `feat(gpu-search): port GPU-resident filesystem index`
  - _Requirements: FR-4_
  - _Design: Filesystem Index_

- [ ] 2.2 Port index/shared_index.rs -- shared index manager
  - **Do**: Port `rust-experiment/src/gpu_os/shared_index.rs` (921 lines). Change cache path from `~/.gpu_os/` to `~/.gpu-search/`. Port GpuFilesystemIndex to objc2-metal types.
  - **Files**: `gpu-search/src/index/shared_index.rs`, update `gpu-search/src/index/mod.rs`
  - **Done when**: Index saved to ~/.gpu-search/, reloaded identically
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_shared_index`
  - **Commit**: `feat(gpu-search): port shared index manager with ~/.gpu-search/ cache`
  - _Requirements: FR-4, FR-22_
  - _Design: Filesystem Index_

- [ ] 2.3 Implement index/scanner.rs -- parallel filesystem scanner
  - **Do**: New module using rayon + ignore crate WalkBuilder. Parallel directory scan building `Vec<GpuPathEntry>`. Respect .gitignore, skip symlinks by default, skip hidden directories (.git/, .hg/).
  - **Files**: `gpu-search/src/index/scanner.rs`, update `gpu-search/src/index/mod.rs`
  - **Done when**: Scan test directory, correct entry count, paths match filesystem
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_scanner`
  - **Commit**: `feat(gpu-search): implement parallel filesystem scanner with rayon + ignore`
  - _Requirements: FR-4, FR-12_
  - _Design: Filesystem Index, scanner.rs_

- [ ] 2.4 Implement index/cache.rs -- binary index persistence
  - **Do**: Save/load binary index at `~/.gpu-search/index/`. Header with version, entry count, root path hash. Pack GpuPathEntry array. mmap for loading.
  - **Files**: `gpu-search/src/index/cache.rs`, update `gpu-search/src/index/mod.rs`
  - **Done when**: Save index, restart, load cached, verify identical entries
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_index_cache`
  - **Commit**: `feat(gpu-search): implement binary index persistence`
  - _Requirements: FR-22_
  - _Design: Filesystem Index, cache.rs_

- [ ] 2.5 Implement index/gpu_loader.rs -- GPU buffer loading
  - **Do**: mmap cached index -> `newBufferWithBytesNoCopy` -> GPU buffer. If no cache: scan -> build -> save -> load. Target <10ms for 100K entries.
  - **Files**: `gpu-search/src/index/gpu_loader.rs`, update `gpu-search/src/index/mod.rs`
  - **Done when**: Index in GPU buffer, path filter kernel queries it successfully
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_gpu_index_load`
  - **Commit**: `feat(gpu-search): implement GPU-resident index loading`
  - _Requirements: FR-4, NFR-5_
  - _Design: Filesystem Index, gpu_loader.rs_

- [ ] 2.6 Adopt gpu-query work queue pattern
  - **Do**: Adapt gpu-query `WorkQueue` for search requests. Triple-buffered (3 x SearchRequestSlot) in StorageModeShared. CPU writes search params with Release ordering, GPU reads with Acquire. Atomic sequence_id for new-request detection.
  - **Files**: `gpu-search/src/gpu/work_queue.rs`, update `gpu-search/src/gpu/mod.rs`
  - **Done when**: Write/read cycle works, atomic ordering correct
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_work_queue`
  - **Commit**: `feat(gpu-search): adopt triple-buffered work queue from gpu-query`
  - _Requirements: FR-5_
  - _Design: GPU Device Layer, work_queue.rs_

- [ ] 2.7 Implement SearchExecutor with re-dispatch chain
  - **Do**: Build SearchExecutor following gpu-query executor.rs. Completion-handler re-dispatch chain for persistent search. MTLSharedEvent for idle/wake. Separate MTLCommandQueue for search compute.
  - **Files**: `gpu-search/src/gpu/executor.rs`, update `gpu-search/src/gpu/mod.rs`
  - **Done when**: Start, submit search, get results, idle, wake all work
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_executor_lifecycle`
  - **Commit**: `feat(gpu-search): implement SearchExecutor with re-dispatch chain`
  - _Requirements: FR-6, NFR-9_
  - _Design: GPU Device Layer, executor.rs_

- [ ] 2.8 Implement search orchestrator
  - **Do**: Top-level SearchOrchestrator coordinating full pipeline: accept SearchRequest from channel, apply .gitignore + binary + filetype filters, dispatch GPU path filter, queue files for batch I/O, dispatch content search, collect results into SearchResponse, send back via channel.
  - **Files**: `gpu-search/src/search/orchestrator.rs`, update `gpu-search/src/search/mod.rs`
  - **Done when**: Orchestrator returns correct results for test directory search
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_orchestrator`
  - **Commit**: `feat(gpu-search): implement search orchestrator pipeline`
  - _Requirements: FR-8, FR-9, FR-15_
  - _Design: Search Engine, orchestrator.rs_

- [ ] 2.9 Implement .gitignore filtering
  - **Do**: Use `ignore` crate (ripgrep's). Parse .gitignore at each directory level, support globs, negation, directory markers. Apply before GPU search. Add `ignore = "0.4"` to Cargo.toml.
  - **Files**: `gpu-search/src/search/ignore.rs`, update `gpu-search/src/search/mod.rs`, `gpu-search/Cargo.toml`
  - **Done when**: Files matching .gitignore excluded from results
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_gitignore`
  - **Commit**: `feat(gpu-search): implement .gitignore filtering via ignore crate`
  - _Requirements: FR-12_
  - _Design: Search Engine, ignore.rs_

- [ ] 2.10 Implement binary file detection
  - **Do**: NUL byte heuristic: read first 8KB, skip if NUL found. Skip known binary extensions (.exe, .o, .dylib, .metallib, images, audio, video). Configurable `include_binary` flag.
  - **Files**: `gpu-search/src/search/binary.rs`, update `gpu-search/src/search/mod.rs`
  - **Done when**: Binary files skipped, text files searched
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_binary_detection`
  - **Commit**: `feat(gpu-search): implement binary file detection`
  - _Requirements: FR-13_
  - _Design: Search Engine, binary.rs_

- [ ] 2.11 Implement result ranking and progressive delivery
  - **Do**: Rank: filename matches by path length (shorter = more relevant), content matches by exact word > partial match > path depth. Cap at 10K results. Progressive delivery via channels: Wave 1 (FileMatches), Wave 2 (ContentMatches), Complete (stats).
  - **Files**: `gpu-search/src/search/ranking.rs`, `gpu-search/src/search/channel.rs`, update `gpu-search/src/search/mod.rs`
  - **Done when**: Results ranked by relevance, filename matches arrive before content matches
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_result_ranking && cargo test test_progressive_results`
  - **Commit**: `feat(gpu-search): implement result ranking and progressive delivery`
  - _Requirements: FR-9, FR-18_
  - _Design: Search Engine, ranking.rs, channel.rs_

- [ ] 2.12 Implement ui/app.rs -- egui application
  - **Do**: GpuSearchApp struct implementing eframe::App. State: search_input, file_matches, content_matches, selected_index, search_tx/result_rx channels, debounce_timer. `eframe::run_native()` with window config: 720px wide, dynamic height, centered, decorated=false for floating panel.
  - **Files**: `gpu-search/src/ui/app.rs`, update `gpu-search/src/ui/mod.rs`, update `gpu-search/src/main.rs`
  - **Done when**: Binary launches and shows empty search window
  - **Verify**: `cd gpu_kernel/gpu-search && cargo build && cargo run`
  - **Commit**: `feat(gpu-search): implement egui application shell`
  - _Requirements: FR-7_
  - _Design: UI, app.rs_

- [ ] 2.13 Implement ui/theme.rs -- Tokyo Night dark theme
  - **Do**: Define color constants: BG_BASE (#1A1B26), BG_SURFACE (#24263A), TEXT_PRIMARY (#C0CAF5), TEXT_MUTED (#565F89), ACCENT (#E0AF68), ERROR (#F7768E), SUCCESS (#9ECE6A), BORDER (#3B3E52). Apply to egui Visuals (window bg, widget bg, text colors, selection).
  - **Files**: `gpu-search/src/ui/theme.rs`, update `gpu-search/src/ui/mod.rs`
  - **Done when**: Theme applied, no pure black/white, WCAG 2.1 AA contrast
  - **Verify**: Visual inspection after `cargo run`
  - **Commit**: `feat(gpu-search): implement Tokyo Night dark theme`
  - _Requirements: FR-19, NFR-10_
  - _Design: UI, theme.rs_

- [ ] 2.14 Implement ui/search_bar.rs -- search input with debounce
  - **Do**: Single-line `egui::TextEdit` with placeholder "Search files and content...". 30ms debounce timer: start on keystroke, fire search when timer expires. Cancel in-flight on new input. Search icon left, filter toggle right. Min query: 1 char filename, 2 chars content.
  - **Files**: `gpu-search/src/ui/search_bar.rs`, update `gpu-search/src/ui/mod.rs`
  - **Done when**: Typing triggers search after 30ms debounce
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_debounce_logic`
  - **Commit**: `feat(gpu-search): implement search bar with 30ms debounce`
  - _Requirements: FR-8, AC-1.2_
  - _Design: UI, search_bar.rs_

- [ ] 2.15 Implement ui/results_list.rs -- scrollable results
  - **Do**: Two sections: FILENAME MATCHES (N) with path + highlighted match, CONTENT MATCHES (N) with file:line header + 3 context lines. Selected item: background highlight + left accent border. Virtual scroll via egui ScrollArea + show_rows() for 10K+ results.
  - **Files**: `gpu-search/src/ui/results_list.rs`, update `gpu-search/src/ui/mod.rs`
  - **Done when**: Results display with mock data, virtual scroll handles 10K items
  - **Verify**: `cd gpu_kernel/gpu-search && cargo run` with mock results
  - **Commit**: `feat(gpu-search): implement scrollable results list with virtual scroll`
  - _Requirements: FR-9, FR-18_
  - _Design: UI, results_list.rs_

- [ ] 2.16 Implement ui/keybinds.rs -- keyboard shortcuts
  - **Do**: Up/Down navigate results (move selected_index, scroll into view), Enter opens file, Cmd+Enter opens in $EDITOR, Escape dismisses, Cmd+Backspace clears search, Cmd+C copies path, Tab cycles filters.
  - **Files**: `gpu-search/src/ui/keybinds.rs`, update `gpu-search/src/ui/mod.rs`
  - **Done when**: All keyboard shortcuts produce correct actions
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_keybind_actions`
  - **Commit**: `feat(gpu-search): implement keyboard shortcuts`
  - _Requirements: FR-11_
  - _Design: UI, keybinds.rs_

- [ ] 2.17 Implement ui/highlight.rs -- syntax highlighting
  - **Do**: syntect integration with custom Tokyo Night theme (4-5 syntax colors). Query match highlight: bold + amber background overriding syntax colors. Cache per-file parse state. Only highlight visible context lines.
  - **Files**: `gpu-search/src/ui/highlight.rs`, update `gpu-search/src/ui/mod.rs`
  - **Done when**: Rust code correctly highlighted with Tokyo Night colors, query matches in amber
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_syntax_highlight`
  - **Commit**: `feat(gpu-search): implement syntax highlighting with syntect`
  - _Requirements: FR-10_
  - _Design: UI, highlight.rs_

- [ ] 2.18 Implement ui/status_bar.rs and ui/filters.rs
  - **Do**: Status bar: match count, timing, search root, active filters. Filter pills: file extension (`.rs`), exclude dir (`!target/`), case sensitive (Cmd+I). Dismissible pills. Filters applied before GPU dispatch.
  - **Files**: `gpu-search/src/ui/status_bar.rs`, `gpu-search/src/ui/filters.rs`, update `gpu-search/src/ui/mod.rs`
  - **Done when**: Status bar shows stats, filter pills add/remove correctly
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_status_bar && cargo test test_filter_pills`
  - **Commit**: `feat(gpu-search): implement status bar and filter pills`
  - _Requirements: FR-14, FR-20, FR-21_
  - _Design: UI, status_bar.rs, filters.rs_

- [ ] 2.19 Wire GPU engine -> orchestrator -> UI
  - **Do**: Connect subsystems: GpuSearchApp creates SearchOrchestrator with GPU device. Orchestrator spawns background thread with own MTLCommandQueue. UI sends SearchRequest via crossbeam channel. Orchestrator sends SearchUpdate back. UI polls with try_recv in update() loop. Add `crossbeam-channel = "0.5"` to Cargo.toml.
  - **Files**: `gpu-search/src/main.rs`, `gpu-search/src/ui/app.rs`, `gpu-search/Cargo.toml`
  - **Done when**: Type query, see GPU-powered results in UI
  - **Verify**: `cd gpu_kernel/gpu-search && cargo run` -- type query, results appear
  - **Commit**: `feat(gpu-search): wire GPU engine to search orchestrator to UI`
  - _Requirements: FR-1 through FR-11_
  - _Design: Data Flow_

- [ ] 2.20 Implement open-in-editor action
  - **Do**: Enter opens file via `open <path>`. Cmd+Enter opens in $EDITOR at line. Support VS Code (`code --goto file:line`), Vim (`nvim +line file`), Sublime (`subl file:line`). Detect editor from $VISUAL, $EDITOR, or PATH. Default `open`.
  - **Files**: `gpu-search/src/ui/actions.rs`, update `gpu-search/src/ui/mod.rs`
  - **Done when**: Correct editor command generated for each editor type
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_editor_command`
  - **Commit**: `feat(gpu-search): implement open-in-editor action`
  - _Requirements: FR-17_
  - _Design: UI, actions.rs_

- [ ] 2.21 Implement search cancellation
  - **Do**: AtomicBool cancellation flag per search. GPU kernel checks flag between chunks. New keystroke sets flag and starts fresh search. Track generation ID to discard stale results.
  - **Files**: `gpu-search/src/search/cancel.rs`, update `gpu-search/src/search/mod.rs`
  - **Done when**: Rapid sequential searches, only latest results returned
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_search_cancel`
  - **Commit**: `feat(gpu-search): implement search cancellation`
  - _Requirements: FR-15_
  - _Design: Search Engine, cancel.rs_

- [ ] 2.22 Implement error handling
  - **Do**: Centralized error types in `error.rs`. GPU device unavailable: show error. MTLIOCommandQueue failure: fall back to CPU reads. Watchdog timeout: restart chain. Invalid UTF-8: skip file. Permission denied: skip, log. Out of GPU memory: reduce batch size.
  - **Files**: `gpu-search/src/error.rs`, update `gpu-search/src/lib.rs`
  - **Done when**: Each failure mode handled gracefully, no panics
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test test_error_recovery`
  - **Commit**: `feat(gpu-search): implement comprehensive error handling`
  - _Design: Error Handling_

- [ ] 2.23 Feature Complete Checkpoint
  - **Do**: Full app working: floating panel, search-as-you-type, GPU content search, filename search, syntax highlighting, keyboard navigation, .gitignore filtering, status bar.
  - **Done when**: Feature can be demonstrated end-to-end
  - **Verify**: Manual test of full workflow: launch -> type query -> navigate results -> open file -> dismiss
  - **Commit**: `feat(gpu-search): feature complete`

## Phase 3: Testing

Unit tests, integration tests, GPU kernel correctness, filesystem edge cases.

- [ ] 3.1 GPU-CPU dual verification tests
  - **Do**: Test every GPU search against CPU reference (memchr/manual). Matrix: literal single/multiple/overlapping, case insensitive, Unicode UTF-8 multibyte, binary NUL detection, boundary crossing (match spans buffer boundary), empty files, large files (>100MB).
  - **Files**: `gpu-search/tests/test_gpu_cpu_consistency.rs`
  - **Done when**: All GPU results match CPU reference across full test matrix
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test --test test_gpu_cpu_consistency`
  - **Commit**: `test(gpu-search): add GPU-CPU dual verification tests`
  - _Requirements: NFR-11_
  - _Design: Test Strategy_

- [ ] 3.2 GPU memory safety tests
  - **Do**: Test with Metal validation enabled. OOB reads (non-aligned file sizes), buffer overrun (more matches than slots), atomic race conditions, watchdog timeout on large single dispatch.
  - **Files**: `gpu-search/tests/test_gpu_memory.rs`
  - **Done when**: Zero Metal validation errors with `MTL_SHADER_VALIDATION=1`
  - **Verify**: `METAL_DEVICE_WRAPPER_TYPE=1 MTL_SHADER_VALIDATION=1 cargo test -p gpu-search --test test_gpu_memory`
  - **Commit**: `test(gpu-search): add GPU memory safety tests`
  - _Design: Test Strategy_

- [ ] 3.3 Filesystem edge case tests
  - **Do**: Test matrix from QA.md: empty files, single-byte, >4GB, NUL-only, broken/circular symlinks, permission denied, Unicode filenames (CJK, emoji, mixed NFC/NFD), .gitignore patterns, hidden directories, deeply nested dirs (100+).
  - **Files**: `gpu-search/tests/test_filesystem_edge_cases.rs`
  - **Done when**: All edge cases handled gracefully (skip or process correctly, no crash)
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test --test test_filesystem_edge_cases`
  - **Commit**: `test(gpu-search): add filesystem edge case tests`
  - _Requirements: FR-12, FR-13_
  - _Design: Test Strategy_

- [ ] 3.4 Search orchestrator integration tests
  - **Do**: Full pipeline tests: search for pattern in test directory with various filter combinations. Test progressive delivery (Wave 1 before Wave 2). Test cancellation. Test empty results. Test error recovery.
  - **Files**: `gpu-search/tests/test_orchestrator_integration.rs`
  - **Done when**: All orchestrator paths tested with real GPU
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test --test test_orchestrator_integration`
  - **Commit**: `test(gpu-search): add search orchestrator integration tests`
  - _Requirements: FR-8, FR-9, FR-15_
  - _Design: Search Engine_

- [ ] 3.5 Index tests
  - **Do**: Test filesystem scanner (correct entry count, paths match), index persistence (save/load cycle), incremental updates (create/delete file reflected), GPU buffer loading (<10ms for 100K entries).
  - **Files**: `gpu-search/tests/test_index.rs`
  - **Done when**: Index lifecycle fully tested
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test --test test_index`
  - **Commit**: `test(gpu-search): add filesystem index tests`
  - _Requirements: FR-4, FR-22, NFR-5_
  - _Design: Filesystem Index_

- [ ] 3.6 Property-based tests with proptest
  - **Do**: GPU matches CPU for arbitrary pattern/content pairs. Properties: count match, positions monotonic, no false positives, no false negatives, idempotent.
  - **Files**: `gpu-search/tests/test_proptest.rs`
  - **Done when**: 10,000 proptest iterations pass
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test --test test_proptest`
  - **Commit**: `test(gpu-search): add property-based tests with proptest`
  - _Requirements: NFR-11_
  - _Design: Test Strategy_

- [ ] 3.7 Stress tests
  - **Do**: Memory leak (10K searches, <1% GPU allocation growth), watchdog survival (60s continuous, zero errors), sustained UI responsiveness (p99 <16ms during large search), concurrent filesystem changes (no crash).
  - **Files**: `gpu-search/tests/test_stress.rs`
  - **Done when**: All stress tests pass (run with `--ignored --test-threads=1`)
  - **Verify**: `cd gpu_kernel/gpu-search && cargo test --test test_stress -- --ignored --test-threads=1`
  - **Commit**: `test(gpu-search): add stress tests`
  - _Requirements: NFR-8, NFR-9_
  - _Design: Test Strategy_

## Phase 4: Quality Gates

Benchmarks, regression detection, CI pipeline, documentation.

- [ ] 4.1 Implement search throughput benchmark
  - **Do**: Criterion benchmark: raw GPU content search at 1MB, 10MB, 100MB synthetic data. Report GB/s. Target: 55-80 GB/s. Sample size 20.
  - **Files**: `gpu-search/benches/search_throughput.rs`
  - **Done when**: `cargo bench -- search_throughput` reports GB/s within expected range
  - **Verify**: `cd gpu_kernel/gpu-search && cargo bench -- search_throughput`
  - **Commit**: `perf(gpu-search): add search throughput benchmark`
  - _Requirements: NFR-2_
  - _Design: Test Strategy_

- [ ] 4.2 Implement I/O latency benchmark
  - **Do**: Criterion benchmark: MTLIOCommandQueue batch loading at 100, 1K, 10K files. Compare against sequential CPU reads. Target: 10K files <30ms.
  - **Files**: `gpu-search/benches/io_latency.rs`
  - **Done when**: MTLIOCommandQueue consistently faster than CPU reads
  - **Verify**: `cd gpu_kernel/gpu-search && cargo bench -- io_latency`
  - **Commit**: `perf(gpu-search): add I/O latency benchmark`
  - _Requirements: NFR-4_
  - _Design: Test Strategy_

- [ ] 4.3 Implement end-to-end search latency benchmark
  - **Do**: Criterion benchmark: full pipeline SearchRequest -> results. Cached incremental <5ms p50, cold first search <200ms, index load <10ms.
  - **Files**: `gpu-search/benches/search_latency.rs`
  - **Done when**: Cached search latency <5ms p50
  - **Verify**: `cd gpu_kernel/gpu-search && cargo bench -- search_latency`
  - **Commit**: `perf(gpu-search): add end-to-end search latency benchmark`
  - _Requirements: NFR-1, NFR-3_
  - _Design: Test Strategy_

- [ ] 4.4 Implement ripgrep comparison benchmark
  - **Do**: Side-by-side Criterion benchmark against ripgrep on same corpus (~100MB). Target: gpu-search 4-7x faster.
  - **Files**: `gpu-search/benches/search_throughput.rs` (additional group)
  - **Done when**: gpu-search consistently faster than ripgrep
  - **Verify**: `cd gpu_kernel/gpu-search && cargo bench -- vs_ripgrep`
  - **Commit**: `perf(gpu-search): add ripgrep comparison benchmark`
  - _Requirements: NFR-2_
  - _Design: Test Strategy_

- [ ] 4.5 Implement regression detection script
  - **Do**: Python script: parse Criterion JSON, compare against baseline, fail on >10% regression. `check_bench_regression.py bench.json --threshold 10`.
  - **Files**: `gpu-search/scripts/check_bench_regression.py`
  - **Done when**: Script detects >10% regression in mock data
  - **Verify**: `cd gpu_kernel/gpu-search && python3 scripts/check_bench_regression.py test_bench.json --threshold 10`
  - **Commit**: `ci(gpu-search): add benchmark regression detection script`
  - _Requirements: NFR-12_
  - _Design: Test Strategy_

- [ ] 4.6 Create CI pipeline
  - **Do**: GitHub Actions workflow at `.github/workflows/gpu-search-ci.yml`. 5 stages: fast checks (clippy, check, test --lib), GPU integration (test gpu_*), performance (bench), stress (nightly, --ignored), UI screenshots.
  - **Files**: `.github/workflows/gpu-search-ci.yml`
  - **Done when**: CI YAML valid, pipeline stages defined
  - **Verify**: Inspect `.github/workflows/gpu-search-ci.yml`
  - **Commit**: `ci(gpu-search): add GitHub Actions CI pipeline`
  - _Design: Test Strategy_

- [ ] 4.7 Local quality check
  - **Do**: Run all quality checks: `cargo clippy -p gpu-search -- -D warnings && cargo test -p gpu-search && cargo build -p gpu-search --release`
  - **Verify**: All commands pass with zero warnings/errors
  - **Done when**: Clean clippy, all tests pass, release build succeeds
  - **Commit**: `fix(gpu-search): address lint/type issues` (if needed)

- [ ] 4.8 Create .gitignore for gpu-search
  - **Do**: Ignore build artifacts: `/target/`, `*.metallib`, `*.air`, `criterion/`.
  - **Files**: `gpu-search/.gitignore`
  - **Done when**: `cat gpu-search/.gitignore` shows expected patterns
  - **Verify**: `cat gpu_kernel/gpu-search/.gitignore`
  - **Commit**: `chore(gpu-search): add .gitignore`

## Phase 5: PR Lifecycle

- [ ] 5.1 Create PR and verify CI
  - **Do**: Push branch, create PR with gh CLI. Description: GPU-accelerated filesystem search tool, ~8K line port from rust-experiment, egui UI, 55-80 GB/s throughput.
  - **Verify**: `gh pr checks --watch` all green
  - **Done when**: PR ready for review, CI passing
  - **Commit**: N/A (PR creation)

- [ ] 5.2 Verification Final (VF)
  - **Do**: Verify all acceptance criteria met: search-as-you-type <5ms cached, 55-80 GB/s throughput, GPU-CPU consistency, .gitignore filtering, keyboard navigation, syntax highlighting, dark theme.
  - **Done when**: All NFRs and FRs verified
  - **Verify**: Run full test suite + benchmarks + manual demo

## Notes

- **POC shortcuts**: Phase 1 uses hardcoded test data, stdout output, no UI. UI and full orchestrator come in Phase 2.
- **Port order**: device -> types -> mmap (simplest) -> gpu_io (complex) -> batch -> content_search -> streaming
- **MSL unchanged**: Metal shader algorithms are NOT modified during port. Only Rust host API calls change.
- **Production TODOs for Phase 2**: Full UI, orchestrator, filters, index, error handling, search cancellation
- **GPU-CPU verification required**: Every search test must compare GPU output against CPU reference implementation
