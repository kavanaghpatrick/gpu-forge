---
spec: gpu-query
phase: tasks
total_tasks: 40
created: 2026-02-10
generated: auto
---

# Tasks: gpu-query

## Phase 1: Make It Work (POC)

Focus: Prove "SELECT count(*) FROM file.csv WHERE col > 100" works on GPU end-to-end. Skip TUI, accept hardcoded values, no JSON yet.

- [x] 1.1 Scaffold project and build.rs
  - **Do**: Create `gpu-query/` as Cargo workspace member alongside particle-system. Set up Cargo.toml with all dependencies (objc2-metal, sqlparser, parquet, ratatui, crossterm, clap, libc, block2, dispatch2). Create `shaders/` directory with `types.h` defining shared structs (FilterParams, AggParams, SortParams, CsvParseParams, DispatchArgs, ColumnSchema). Create `build.rs` copying pattern from particle-system/build.rs (xcrun metal -c -> metallib). Create minimal `src/main.rs` that compiles.
  - **Files**: `gpu-query/Cargo.toml`, `gpu-query/build.rs`, `gpu-query/src/main.rs`, `gpu-query/src/lib.rs`, `gpu-query/shaders/types.h`
  - **Done when**: `cargo build` succeeds, shaders.metallib generated in OUT_DIR
  - **Verify**: `cargo build 2>&1 | grep "Built shaders.metallib"`
  - **Commit**: `feat(gpu-query): scaffold project with build.rs and shared types`
  - _Requirements: FR-38_
  - _Design: File Structure, build.rs_

- [x] 1.2 Metal device init and GPU types
  - **Do**: Create `src/gpu/device.rs` with Metal device creation, command queue, library loading (reuse particle-system/gpu.rs pattern). Create `src/gpu/types.rs` with `#[repr(C)]` structs: FilterParams, AggParams, CsvParseParams, DispatchArgs, ColumnSchema. Add byte-level layout tests for all structs. Create `src/gpu/mod.rs`.
  - **Files**: `gpu-query/src/gpu/mod.rs`, `gpu-query/src/gpu/device.rs`, `gpu-query/src/gpu/types.rs`
  - **Done when**: Metal device initializes, metallib loads, all struct layout tests pass
  - **Verify**: `cargo test --lib gpu::types`
  - **Commit**: `feat(gpu-query): Metal device init and repr(C) GPU types`
  - _Requirements: FR-38, NFR-12_
  - _Design: GPU Kernel Architecture, Existing Patterns_

- [x] 1.3 mmap + zero-copy Metal buffer
  - **Do**: Create `src/io/mmap.rs` with MmapFile struct: open file, mmap with MAP_SHARED+PROT_READ, round to 16KB page alignment [KB #89], madvise(MADV_WILLNEED), makeBuffer(bytesNoCopy:) with StorageModeShared. Add fallback copy path. Create `src/io/mod.rs`.
  - **Files**: `gpu-query/src/io/mod.rs`, `gpu-query/src/io/mmap.rs`
  - **Done when**: Can mmap a file and verify Metal buffer contents match file bytes; page alignment test passes
  - **Verify**: `cargo test --lib io::mmap`
  - **Commit**: `feat(gpu-query): zero-copy mmap + Metal buffer wrapping`
  - _Requirements: FR-2, FR-36, FR-37_
  - _Design: Zero-Copy I/O Pipeline_

- [x] 1.4 CSV metadata reader (CPU-side)
  - **Do**: Create `src/io/csv.rs` with CPU-side header parsing: read first line for column names, detect delimiter (comma/tab/pipe), count columns. Create `src/io/format_detect.rs` with magic bytes detection (Parquet PAR1 header, JSON { or [, else CSV). Create `src/io/catalog.rs` with directory scanner.
  - **Files**: `gpu-query/src/io/csv.rs`, `gpu-query/src/io/format_detect.rs`, `gpu-query/src/io/catalog.rs`
  - **Done when**: Can scan a directory, detect CSV files, parse headers, return table metadata
  - **Verify**: `cargo test --lib io::csv && cargo test --lib io::format_detect`
  - **Commit**: `feat(gpu-query): CSV metadata reader and format detection`
  - _Requirements: FR-1_
  - _Design: Components, File System Layer_

- [x] 1.5 GPU CSV parser kernel (newline detection + field extraction)
  - **Do**: Create `shaders/csv_parse.metal` with two kernels: `csv_detect_newlines` (parallel scan for '\n', output row_offsets via atomic counter) and `csv_parse_fields` (per-row field extraction, type coercion to INT64/FLOAT64, write to SoA columns). Create `src/gpu/encode.rs` with encode helpers. Create `src/storage/mod.rs`, `src/storage/columnar.rs` for SoA buffer allocation, `src/storage/schema.rs` for runtime schema.
  - **Files**: `gpu-query/shaders/csv_parse.metal`, `gpu-query/src/gpu/encode.rs`, `gpu-query/src/storage/mod.rs`, `gpu-query/src/storage/columnar.rs`, `gpu-query/src/storage/schema.rs`
  - **Done when**: Can parse a simple CSV (no quoting) into columnar SoA buffers on GPU; verify column values match expected
  - **Verify**: `cargo test --test gpu_csv` (create tests/gpu_csv.rs with small test CSV)
  - **Commit**: `feat(gpu-query): GPU CSV parser kernel with SoA output`
  - _Requirements: FR-3, FR-15_
  - _Design: Kernel 1: CSV Parser_

- [x] 1.6 GPU column filter kernel (WHERE clause)
  - **Do**: Create `shaders/filter.metal` with `column_filter` kernel using function constants for COMPARE_OP, COLUMN_TYPE, HAS_NULL_CHECK. Output: selection bitmask (1-bit per row) + match_count via simd_sum + atomic. Create `src/gpu/pipeline.rs` with PSO creation using MTLFunctionConstantValues and PSO cache.
  - **Files**: `gpu-query/shaders/filter.metal`, `gpu-query/src/gpu/pipeline.rs`
  - **Done when**: Can filter INT64 column with GT/LT/EQ operators; match_count correct; PSO cache works
  - **Verify**: `cargo test --test gpu_filter` (create tests/gpu_filter.rs)
  - **Commit**: `feat(gpu-query): GPU filter kernel with function constant specialization`
  - _Requirements: FR-10, FR-11, FR-17_
  - _Design: Kernel 4: Column Filter_

- [x] 1.7 GPU aggregation kernel (COUNT/SUM)
  - **Do**: Create `shaders/aggregate.metal` with `aggregate_sum_int64` and `aggregate_count` kernels using 3-level hierarchical reduction (SIMD -> threadgroup -> global atomic). Read selection_mask from filter output.
  - **Files**: `gpu-query/shaders/aggregate.metal`
  - **Done when**: SUM and COUNT produce correct results for filtered data; verified against CPU reference
  - **Verify**: `cargo test --test gpu_aggregate`
  - **Commit**: `feat(gpu-query): GPU aggregation kernel with hierarchical reduction`
  - _Requirements: FR-12_
  - _Design: Kernel 5: Aggregation_

- [x] 1.8 SQL parser integration
  - **Do**: Create `src/sql/mod.rs`, `src/sql/parser.rs` wrapping sqlparser-rs for the MVP SQL subset. Create `src/sql/types.rs` with DataType enum and expression types. Create `src/sql/logical_plan.rs` with plan nodes (Scan, Filter, Aggregate, Sort, Limit). Create `src/sql/physical_plan.rs` mapping logical plan to kernel dispatch graph.
  - **Files**: `gpu-query/src/sql/mod.rs`, `gpu-query/src/sql/parser.rs`, `gpu-query/src/sql/types.rs`, `gpu-query/src/sql/logical_plan.rs`, `gpu-query/src/sql/physical_plan.rs`
  - **Done when**: Can parse `SELECT count(*) FROM t WHERE col > 100` to a physical plan; unit tests for parse + plan
  - **Verify**: `cargo test --lib sql`
  - **Commit**: `feat(gpu-query): SQL parser and query planner integration`
  - _Requirements: FR-7, FR-8, FR-9_
  - _Design: Query Compiler_

- [x] 1.9 GPU execution engine (end-to-end query)
  - **Do**: Create `src/gpu/executor.rs` connecting SQL physical plan to Metal kernel dispatch. Encode command buffer: CSV parse -> filter -> aggregate. Use waitUntilCompleted for synchronous result. Wire together: main.rs takes directory path + SQL string via CLI args, runs query, prints result.
  - **Files**: `gpu-query/src/gpu/executor.rs`, update `gpu-query/src/main.rs`
  - **Done when**: `gpu-query ./test-data/ -e "SELECT count(*) FROM test WHERE amount > 100"` prints correct count
  - **Verify**: Manual: create test CSV, run query, verify count matches `wc -l` + grep equivalent
  - **Commit**: `feat(gpu-query): end-to-end GPU query execution`
  - _Requirements: FR-2, FR-3, FR-7, FR-11, FR-12_
  - _Design: Data Flow, Persistent Engine_

- [x] 1.10 POC Checkpoint
  - **Do**: Verify full pipeline: SQL -> parse -> plan -> GPU CSV parse -> GPU filter -> GPU aggregate -> result. Create a 1M-row test CSV, run `SELECT count(*), sum(amount) FROM sales WHERE amount > 500` and verify correctness against CPU reference. Document POC results.
  - **Files**: None (verification only)
  - **Done when**: Query returns correct results on 1M-row dataset; pipeline proven end-to-end
  - **Verify**: `gpu-query ./test-data/ -e "SELECT count(*), sum(amount) FROM sales WHERE amount > 500"`
  - **Commit**: `feat(gpu-query): complete POC -- end-to-end GPU query pipeline`
  - _Requirements: AC-2.1, AC-2.2, AC-2.4_
  - _Design: Data Flow_

## Phase 2: Core Engine

After POC validated, build the full query engine with all formats and operators.

- [x] 2.1 Parquet reader (CPU metadata + GPU decode)
  - **Do**: Create `src/io/parquet.rs` reading Parquet footer, schema, row group descriptors via `parquet` crate. mmap column chunk byte ranges. Create `shaders/parquet_decode.metal` with `parquet_decode_plain_int64`, `parquet_decode_plain_double`, `parquet_decode_dictionary` kernels. Column pruning: only load queried columns.
  - **Files**: `gpu-query/src/io/parquet.rs`, `gpu-query/shaders/parquet_decode.metal`
  - **Done when**: Can query Parquet files with SELECT/WHERE/aggregate; column pruning reduces I/O
  - **Verify**: `cargo test --test gpu_parquet`
  - **Commit**: `feat(gpu-query): Parquet reader with GPU column decoding`
  - _Requirements: FR-5, AC-6.1, AC-6.2, AC-6.3_
  - _Design: Kernel 3: Parquet Decoder_

- [x] 2.2 JSON parser kernel (NDJSON)
  - **Do**: Create `src/io/json.rs` for NDJSON metadata (detect fields from first record). Create `shaders/json_parse.metal` with `json_structural_index` (parallel structural char detection) and `json_extract_columns` (field extraction to SoA). Target NDJSON only (one record per line).
  - **Files**: `gpu-query/src/io/json.rs`, `gpu-query/shaders/json_parse.metal`
  - **Done when**: Can query NDJSON files with SELECT/WHERE/aggregate
  - **Verify**: `cargo test --test gpu_json`
  - **Commit**: `feat(gpu-query): GPU NDJSON parser with structural indexing`
  - _Requirements: FR-4, AC-8.1, AC-8.2, AC-8.3_
  - _Design: Kernel 2: JSON Parser_

- [x] 2.3 Full aggregation kernel (SUM, AVG, MIN, MAX, GROUP BY)
  - **Do**: Extend `shaders/aggregate.metal` with `aggregate_sum_double`, `aggregate_min_max` (function constant: min vs max), `aggregate_grouped` (threadgroup hash table for GROUP BY, 256-bucket local + global merge). Extend physical planner to map GROUP BY + multiple aggregates.
  - **Files**: `gpu-query/shaders/aggregate.metal`, update `src/sql/physical_plan.rs`
  - **Done when**: All 5 aggregate functions work with GROUP BY; verified against CPU oracle
  - **Verify**: `cargo test --test gpu_aggregate`
  - **Commit**: `feat(gpu-query): full aggregation with GROUP BY and all aggregate functions`
  - _Requirements: FR-12, AC-2.3, AC-2.4_
  - _Design: Kernel 5: Aggregation_

- [x] 2.4 Radix sort kernel (ORDER BY)
  - **Do**: Create `shaders/sort.metal` with `radix_sort_histogram`, `radix_sort_scan` (exclusive prefix scan), `radix_sort_scatter`. 4-bit radix sort, 16 passes for 64-bit keys. Support ASC/DESC via key transformation. Wire into physical planner for ORDER BY.
  - **Files**: `gpu-query/shaders/sort.metal`, update `src/sql/physical_plan.rs`, `src/gpu/executor.rs`
  - **Done when**: ORDER BY ASC/DESC correct for INT64 and FLOAT64; matches CPU sort
  - **Verify**: `cargo test --test gpu_sort`
  - **Commit**: `feat(gpu-query): GPU radix sort kernel for ORDER BY`
  - _Requirements: FR-13, AC-2.5_
  - _Design: Kernel 6: Radix Sort_

- [x] 2.5 Indirect dispatch and prepare_dispatch kernel
  - **Do**: Create `shaders/prepare_dispatch.metal` with `prepare_query_dispatch` kernel: read match_count from filter, compute threadgroup counts for downstream kernels. Wire into executor to eliminate CPU readback of intermediate counts. Same pattern as particle-system.
  - **Files**: `gpu-query/shaders/prepare_dispatch.metal`, update `src/gpu/executor.rs`
  - **Done when**: Query pipeline runs without CPU readback between stages; indirect dispatch correct
  - **Verify**: `cargo test --test gpu_dispatch`
  - **Commit**: `feat(gpu-query): indirect dispatch for GPU-autonomous query pipeline`
  - _Requirements: FR-14_
  - _Design: Kernel 7: Prepare Dispatch_

- [x] 2.6 Schema inference kernel
  - **Do**: Create `shaders/schema_infer.metal` with `infer_schema` kernel (type voting from sample data). Create `src/storage/null_bitmap.rs` for null bitmap ops. Integrate with CSV/JSON readers for auto-type detection.
  - **Files**: `gpu-query/shaders/schema_infer.metal`, `gpu-query/src/storage/null_bitmap.rs`
  - **Done when**: GPU infers correct types for CSV columns (INT64, FLOAT64, VARCHAR) from first 10K rows
  - **Verify**: `cargo test --test gpu_schema`
  - **Commit**: `feat(gpu-query): GPU schema inference with type voting`
  - _Requirements: FR-6_
  - _Design: Kernel 8: Schema Inference_

- [x] 2.7 Dictionary encoding for strings
  - **Do**: Create `shaders/dict_build.metal` with `build_dictionary` kernel (sort-based dedup). Create `src/storage/dictionary.rs` with adaptive encoding: dict if <10K distinct, offset+data otherwise. Wire into filter kernel for string equality (compare dict indices).
  - **Files**: `gpu-query/shaders/dict_build.metal`, `gpu-query/src/storage/dictionary.rs`
  - **Done when**: String columns dictionary-encoded; WHERE region = 'Europe' works via integer comparison
  - **Verify**: `cargo test --test gpu_filter -- test_string_filter`
  - **Commit**: `feat(gpu-query): adaptive dictionary encoding for string columns`
  - _Requirements: FR-16_
  - _Design: Columnar Storage Engine, String storage_

- [x] 2.8 Compound predicates (AND/OR) and stream compaction
  - **Do**: Add `compound_filter_and` and `compound_filter_or` to `shaders/filter.metal` (bitwise AND/OR of selection masks). Create `shaders/compact.metal` with `compact_selection` kernel (bitmask to dense row indices via prefix scan [KB #193]). Wire into executor for multi-predicate WHERE.
  - **Files**: `gpu-query/shaders/filter.metal`, `gpu-query/shaders/compact.metal`, update `src/gpu/executor.rs`
  - **Done when**: `WHERE col1 > 100 AND col2 < 200` works correctly
  - **Verify**: `cargo test --test gpu_filter -- test_compound`
  - **Commit**: `feat(gpu-query): compound predicates and stream compaction`
  - _Requirements: FR-31, AC-2.8_
  - _Design: Kernel 4: Column Filter_

- [x] 2.9 Batched query execution for large files
  - **Do**: In `src/gpu/executor.rs`, add batched execution path: if estimated scan > 1GB, split into multiple command buffers (1GB chunks) with pre-enqueue ordering [KB #152]. Merge partial results (partial sums, partial sorts). Auto-detect batch threshold.
  - **Files**: update `gpu-query/src/gpu/executor.rs`
  - **Done when**: Queries on >1GB files complete without GPU watchdog timeout
  - **Verify**: Generate 2GB test CSV, run aggregation query, verify correct result
  - **Commit**: `feat(gpu-query): batched query execution for GPU watchdog safety`
  - _Requirements: FR-18_
  - _Design: Persistent Engine, Command buffer strategy_

- [x] 2.10 Query optimizer (column pruning + predicate pushdown)
  - **Do**: Create `src/sql/optimizer.rs` with logical plan transformations: (a) column pruning -- only load columns referenced in SELECT/WHERE/GROUP BY/ORDER BY, (b) predicate pushdown -- move WHERE filters before aggregation, (c) constant folding. Wire into planner pipeline.
  - **Files**: `gpu-query/src/sql/optimizer.rs`
  - **Done when**: `SELECT sum(amount) FROM wide_table WHERE region = 'EU'` only loads amount + region columns
  - **Verify**: `cargo test --lib sql::optimizer`
  - **Commit**: `feat(gpu-query): query optimizer with column pruning and predicate pushdown`
  - _Requirements: FR-8_
  - _Design: Query Compiler, Logical planner_

- [x] 2.11 Non-interactive CLI mode
  - **Do**: Create `src/cli/mod.rs`, `src/cli/args.rs` with clap derive for all flags (-e, -f, -o, --format, --no-gpu, --profile, --dashboard, --cold, --theme). Implement non-interactive execution path: parse args -> run query -> format output -> exit. Support pipe input.
  - **Files**: `gpu-query/src/cli/mod.rs`, `gpu-query/src/cli/args.rs`, update `src/main.rs`
  - **Done when**: `gpu-query ./data/ -e "SELECT ..." --format csv > output.csv` works
  - **Verify**: `cargo test --test e2e_cli`
  - **Commit**: `feat(gpu-query): non-interactive CLI mode with format output`
  - _Requirements: FR-26, AC-5.1, AC-5.2, AC-5.3, AC-5.4, AC-5.5_
  - _Design: CLI / Script Mode_

## Phase 3: Dashboard TUI

- [x] 3.1 ratatui setup with gradient rendering
  - **Do**: Create `src/tui/mod.rs`, `src/tui/app.rs` (AppState model), `src/tui/event.rs` (event loop with crossterm), `src/tui/gradient.rs` (gradient color interpolation for true-color), `src/tui/themes.rs` (gradient palette definitions: thermal GPU bar, glow throughput). Set up 60fps render loop.
  - **Files**: `gpu-query/src/tui/mod.rs`, `gpu-query/src/tui/app.rs`, `gpu-query/src/tui/event.rs`, `gpu-query/src/tui/gradient.rs`, `gpu-query/src/tui/themes.rs`
  - **Done when**: TUI renders empty dashboard with gradient-colored title bar; event loop runs at 60fps
  - **Verify**: `cargo run -- ./test-data/ --dashboard` shows colored TUI frame
  - **Commit**: `feat(gpu-query): ratatui TUI setup with gradient rendering system`
  - _Requirements: FR-19, FR-20_
  - _Design: TUI Rendering Architecture_

- [x] 3.2 Query editor with syntax highlighting and autocomplete
  - **Do**: Create `src/tui/editor.rs` with multi-line SQL editor (tui-textarea or custom). SQL syntax highlighting: keywords (blue), identifiers (green), literals (yellow), numbers (magenta). Create `src/tui/autocomplete.rs` with tab-complete popup: table names, column names + types + cardinality, SQL keywords, functions. Fuzzy matching.
  - **Files**: `gpu-query/src/tui/editor.rs`, `gpu-query/src/tui/autocomplete.rs`
  - **Done when**: Can type SQL with syntax highlighting; Tab shows autocomplete popup with column types
  - **Verify**: Manual: type `SELECT re` + Tab shows `region [VARCHAR, 8 distinct]`
  - **Commit**: `feat(gpu-query): query editor with syntax highlighting and rich autocomplete`
  - _Requirements: FR-21, FR-22, AC-4.4, AC-4.5, AC-4.6_
  - _Design: TUI Rendering Architecture_

- [x] 3.3 Results table with streaming pagination
  - **Do**: Create `src/tui/results.rs` with scrollable table widget. Number formatting (thousands separators, right-aligned numerics). NULL rendering (dim gray). Column auto-width. Pagination for >1000 rows (Space: next page). Performance line below results.
  - **Files**: `gpu-query/src/tui/results.rs`
  - **Done when**: Query results display in formatted table; can scroll through large result sets
  - **Verify**: Manual: run query with 10K results, scroll with arrow keys
  - **Commit**: `feat(gpu-query): results table with streaming pagination`
  - _Requirements: FR-34, AC-4.7_
  - _Design: TUI Rendering Architecture_

- [x] 3.4 GPU status dashboard and metrics
  - **Do**: Create `src/gpu/metrics.rs` collecting GPU metrics (Metal Counters API [KB #258] for utilization and timestamps, `device.currentAllocatedSize` for memory). Create `src/tui/dashboard.rs` with GPU status panel: utilization bar (gradient green->yellow->red), scan throughput, memory used, sparkline history (ratatui Sparkline widget).
  - **Files**: `gpu-query/src/gpu/metrics.rs`, `gpu-query/src/tui/dashboard.rs`
  - **Done when**: GPU utilization bar and sparkline update after each query; metrics accurate
  - **Verify**: Manual: run query, verify utilization bar matches Instruments trace
  - **Commit**: `feat(gpu-query): GPU status dashboard with gradient metrics`
  - _Requirements: FR-23, AC-3.1, AC-4.2, AC-4.3_
  - _Design: TUI Rendering Architecture, GPU Metrics_

- [x] 3.5 Data catalog tree view
  - **Do**: Create `src/tui/catalog.rs` with tree view widget: directory > table (with format badge) > columns (with types). Navigation: arrow keys or j/k. Enter on table = DESCRIBE. Row count badges per table. Integrate with catalog module.
  - **Files**: `gpu-query/src/tui/catalog.rs`
  - **Done when**: Left panel shows file tree with columns and types; clicking table shows DESCRIBE
  - **Verify**: Manual: launch with multi-file directory, navigate catalog, select table
  - **Commit**: `feat(gpu-query): data catalog tree view with column browser`
  - _Requirements: FR-30, AC-4.1_
  - _Design: TUI Rendering Architecture_

- [x] 3.6 CPU comparison and per-query performance line
  - **Do**: Implement CPU comparison estimate: `(bytes_processed / 6.5 GB_per_sec) / actual_gpu_time` for the `~Nx vs CPU` display. Add to every query result line. Implement `.benchmark` command running actual CPU-side query (Rust iterators) for real comparison. Implement warm/cold timing display.
  - **Files**: update `gpu-query/src/gpu/metrics.rs`, `gpu-query/src/tui/results.rs`
  - **Done when**: Every query shows `142M rows | 8.4 GB | 2.3ms | GPU 94% | ~312x vs CPU`
  - **Verify**: Manual: run query, verify CPU comparison is plausible
  - **Commit**: `feat(gpu-query): CPU comparison estimate and performance line on every query`
  - _Requirements: FR-24, FR-25, FR-33, AC-3.1, AC-3.2, AC-3.3_
  - _Design: TUI Rendering Architecture_

- [x] 3.7 Full dashboard layout with responsive panels
  - **Do**: Create `src/tui/ui.rs` with three-panel layout composition. Panel focus cycling (Ctrl+1/2/3 or Tab). Responsive: >=120 cols = full three-panel, 80-119 = two-panel, <80 = minimal REPL. Connect all widgets. Wire Ctrl+Enter to execute query from editor.
  - **Files**: `gpu-query/src/tui/ui.rs`, update `src/main.rs`
  - **Done when**: Full dashboard launches as default; all panels connected; query execution from editor works
  - **Verify**: Manual: launch, type query, Ctrl+Enter, see results + GPU metrics
  - **Commit**: `feat(gpu-query): full dashboard TUI with responsive panel layout`
  - _Requirements: FR-19, AC-4.1, AC-4.8_
  - _Design: TUI Rendering Architecture_

- [x] 3.8 Dot commands and query history
  - **Do**: Create `src/cli/commands.rs` with dot command handler: .tables, .schema, .describe, .gpu, .profile, .benchmark, .timer, .comparison, .format, .save, .history, .clear, .help, .quit. Implement query history persistence to `~/.config/gpu-query/history`. Create `src/tui/` config loading from `~/.config/gpu-query/config.toml`.
  - **Files**: `gpu-query/src/cli/commands.rs`, update `src/tui/event.rs`
  - **Done when**: All dot commands work in TUI; history persists across sessions
  - **Verify**: `.tables` shows loaded tables; `.history` shows past queries; restart confirms persistence
  - **Commit**: `feat(gpu-query): dot commands and persistent query history`
  - _Requirements: FR-27, FR-28, FR-29_
  - _Design: Dot Commands_

- [x] 3.9 Profile mode with kernel timeline
  - **Do**: Implement `.profile on` mode that shows per-kernel timing after each query: Parse, Plan, mmap warm, GPU Scan, GPU Filter, GPU Agg, GPU Sort, Transfer, Format, Total. Use Metal timestamp counters [KB #258] for GPU-side timing. Proportional ASCII bar widths.
  - **Files**: update `gpu-query/src/gpu/metrics.rs`, `gpu-query/src/tui/results.rs`
  - **Done when**: `.profile on` then query shows kernel timeline with proportional bars
  - **Verify**: Manual: run profiled query, verify GPU kernel times sum to total GPU time
  - **Commit**: `feat(gpu-query): profile mode with per-kernel pipeline timeline`
  - _Requirements: FR-35, AC-3.5_
  - _Design: GPU Metrics_

- [x] 3.10 DESCRIBE and GPU-parallel column statistics
  - **Do**: Implement DESCRIBE command that runs GPU kernels to compute per-column: null%, distinct count, min, max, sample value. Reuse aggregation kernels for min/max. Use dictionary build for cardinality. Display in formatted table.
  - **Files**: update `gpu-query/src/gpu/executor.rs`, `src/cli/commands.rs`
  - **Done when**: `DESCRIBE sales` shows column stats computed on GPU in <100ms for 1M rows
  - **Verify**: `cargo test --test e2e_csv -- test_describe`
  - **Commit**: `feat(gpu-query): GPU-parallel DESCRIBE with column statistics`
  - _Requirements: FR-32, AC-2.7_
  - _Design: Query Compiler_

## Phase 4: Testing & Quality

- [ ] 4.1 Struct layout tests for all GPU types
  - **Do**: Ensure all `#[repr(C)]` structs in `src/gpu/types.rs` have layout tests: size_of, align_of, offset_of for every field. Cover: FilterParams, AggParams, SortParams, CsvParseParams, JsonParseParams, ParquetChunkParams, ColumnSchema, DispatchArgs, InferParams.
  - **Files**: `gpu-query/src/gpu/types.rs` (tests module)
  - **Done when**: ~20 layout tests pass, catching any Rust-MSL struct mismatch
  - **Verify**: `cargo test --lib gpu::types`
  - **Commit**: `test(gpu-query): comprehensive struct layout tests for all GPU types`
  - _Requirements: NFR-12_
  - _Design: Existing Patterns (particle-system/types.rs)_

- [ ] 4.2 SQL parser unit tests
  - **Do**: Write ~50 unit tests for SQL parser integration: valid queries (SELECT, WHERE, GROUP BY, ORDER BY, LIMIT, all aggregates), invalid SQL (parse errors), edge cases (empty WHERE, multiple GROUP BY columns, nested expressions).
  - **Files**: `gpu-query/src/sql/parser.rs` (tests module)
  - **Done when**: ~50 parser tests pass covering the MVP SQL subset
  - **Verify**: `cargo test --lib sql::parser`
  - **Commit**: `test(gpu-query): SQL parser unit tests for MVP subset`
  - _Requirements: NFR-13_
  - _Design: Query Compiler_

- [ ] 4.3 GPU kernel integration tests
  - **Do**: Create test harness (GpuTestContext with device, queue, library, pre-compiled PSOs). Write GPU integration tests: CSV parser (~20), filter kernel (~20 covering all operators and types), aggregation (~20 covering all functions + GROUP BY), sort (~15 covering all edge cases). Use `waitUntilCompleted` for synchronous assertions.
  - **Files**: `gpu-query/tests/gpu_csv.rs`, `gpu-query/tests/gpu_filter.rs`, `gpu-query/tests/gpu_aggregate.rs`, `gpu-query/tests/gpu_sort.rs`
  - **Done when**: ~75 GPU integration tests pass on Metal device
  - **Verify**: `MTL_SHADER_VALIDATION=1 cargo test --test gpu_filter --test gpu_aggregate --test gpu_sort`
  - **Commit**: `test(gpu-query): GPU kernel integration tests with Metal shader validation`
  - _Requirements: NFR-13_
  - _Design: GPU Kernel Architecture_

- [ ] 4.4 End-to-end SQL query tests with golden files
  - **Do**: Create golden file test framework: for each test query, store expected CSV output in `tests/golden/`. Test runner executes SQL against test data files, compares output against golden file (exact for integers, tolerance for floats). Cover: ~25 CSV queries, ~25 Parquet queries, ~20 JSON queries, ~10 cross-format consistency, ~10 CLI mode, ~15 error handling.
  - **Files**: `gpu-query/tests/e2e_csv.rs`, `gpu-query/tests/e2e_parquet.rs`, `gpu-query/tests/e2e_json.rs`, `gpu-query/tests/e2e_cross.rs`, `gpu-query/tests/e2e_cli.rs`, `gpu-query/tests/e2e_errors.rs`, `gpu-query/tests/golden/` directory
  - **Done when**: ~105 E2E tests pass against golden files
  - **Verify**: `cargo test --test e2e_csv --test e2e_parquet --test e2e_json --test e2e_errors`
  - **Commit**: `test(gpu-query): end-to-end SQL tests with golden file oracle`
  - _Requirements: NFR-13_
  - _Design: Data Flow_

- [ ] 4.5 Performance benchmarks (criterion.rs)
  - **Do**: Create benchmark suite with criterion.rs: scan throughput (per format at 1MB/100MB/1GB), filter throughput (various selectivities), aggregation throughput (various group counts), sort throughput (various sizes), end-to-end query latency (representative SQL). Add TPC-H Q1 adapted for gpu-query.
  - **Files**: `gpu-query/benches/scan_throughput.rs`, `gpu-query/benches/filter_throughput.rs`, `gpu-query/benches/aggregate_throughput.rs`, `gpu-query/benches/sort_throughput.rs`, `gpu-query/benches/query_latency.rs`
  - **Done when**: ~30 benchmarks run with statistical rigor; baseline established
  - **Verify**: `cargo bench --bench scan_throughput -- --save-baseline v0.1.0`
  - **Commit**: `test(gpu-query): criterion.rs performance benchmark suite`
  - _Requirements: NFR-1, NFR-2, NFR-11_
  - _Design: Performance Model_

- [ ] 4.6 Fuzz targets for CSV and JSON parsers
  - **Do**: Create `fuzz/` directory with cargo-fuzz targets: `fuzz_csv_parser` (arbitrary bytes -> CPU row detection, compare GPU if <1MB), `fuzz_json_parser` (arbitrary bytes -> structural indexing). Seed corpus with edge-case files.
  - **Files**: `gpu-query/fuzz/fuzz_targets/fuzz_csv_parser.rs`, `gpu-query/fuzz/fuzz_targets/fuzz_json_parser.rs`, `gpu-query/fuzz/Cargo.toml`
  - **Done when**: Fuzz targets build and run for 5 minutes each without crashes
  - **Verify**: `cargo fuzz run fuzz_csv_parser -- -max_total_time=300`
  - **Commit**: `test(gpu-query): fuzz targets for CSV and JSON parsers`
  - _Requirements: NFR-13_
  - _Design: Error Handling_

- [ ] 4.7 CI/CD pipeline setup
  - **Do**: Create `.github/workflows/ci.yml` with: (a) unit tests on ubuntu-latest (no GPU), (b) GPU tests on macos-latest (Apple Silicon), (c) Metal Shader Validation on shader changes, (d) benchmarks on main only (save baseline), (e) fuzz nightly. Two-tier performance gate: 15% blocks merge, 5% warns.
  - **Files**: `gpu-query/.github/workflows/ci.yml`
  - **Done when**: CI pipeline runs on push; unit tests pass on Linux, GPU tests pass on macOS
  - **Verify**: Push branch, verify all CI checks green
  - **Commit**: `ci(gpu-query): CI/CD pipeline with GPU testing and performance gates`
  - _Requirements: NFR-11, NFR-13_
  - _Design: Quality Strategy_

## Phase 5: PR Lifecycle

- [ ] 5.1 Local quality check
  - **Do**: Run all quality checks locally: `cargo test --all-targets`, `cargo clippy -- -D warnings`, `cargo fmt --check`, `MTL_SHADER_VALIDATION=1 cargo test --test gpu_filter --test gpu_aggregate`. Fix any issues. Run benchmarks and verify no regressions > 15%.
  - **Files**: None (verification only)
  - **Done when**: All checks pass locally; no clippy warnings; no perf regressions
  - **Verify**: `cargo test --all-targets && cargo clippy -- -D warnings && cargo fmt --check`
  - **Commit**: `fix(gpu-query): address lint/type issues` (if needed)

- [ ] 5.2 Create PR and verify CI
  - **Do**: Push branch to remote. Create PR with `gh pr create` including summary of all features, architecture decisions, test counts, and performance data. Monitor CI with `gh pr checks --watch`. Address any CI failures.
  - **Files**: None
  - **Done when**: PR created, all CI checks green, ready for review
  - **Verify**: `gh pr checks --watch` all green
  - **Commit**: N/A (PR creation)

- [ ] 5.3 VF: Final verification
  - **Do**: Run the complete test suite one final time. Verify: (a) all ~410 tests pass, (b) 1M-row query completes in <10ms warm, (c) TUI dashboard launches in <3 seconds, (d) all three formats (CSV, Parquet, JSON) queryable. Document final performance numbers.
  - **Files**: None (verification only)
  - **Done when**: All acceptance criteria verified; performance targets met
  - **Verify**: Full test suite + manual TUI verification
  - **Commit**: N/A (verification)

## Notes

- **POC shortcuts taken**: No TUI in Phase 1 (CLI output only), no JSON parser, no sort kernel, no dictionary encoding, synchronous execution (waitUntilCompleted)
- **Production TODOs in Phase 2**: Async execution for TUI, batched dispatches, full aggregate set, string support, multiple file formats
- **Key risk mitigation**: mmap+bytesNoCopy tested early in task 1.3; GPU watchdog handled by batching in task 2.9
- **Particle-system patterns reused**: build.rs (1.1), device init (1.2), buffer allocation (1.3), encode helpers (1.5), indirect dispatch (2.5), struct layout tests (4.1)
