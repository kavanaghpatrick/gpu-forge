---
spec: gpu-query
phase: requirements
created: 2026-02-10
generated: auto
---

# Requirements: gpu-query

## Summary

GPU-native local data analytics engine for Apple Silicon. Users point at a directory of CSV/Parquet/JSON files and query them with SQL in milliseconds using Metal compute. Full TUI dashboard with gradient rendering, GPU metrics, and CPU comparison on every query.

## User Stories

### US-1: Zero-Setup File Query
As a data engineer, I want to point gpu-query at a directory and immediately query files with SQL, so that I can explore data without import steps or schema definitions.

**Acceptance Criteria**:
- AC-1.1: `gpu-query ~/data/` opens TUI, scans directory, shows tables in <3 seconds
- AC-1.2: Auto-detects file formats (CSV, Parquet, JSON) by magic bytes + extension
- AC-1.3: Auto-infers column names and types from file headers/metadata
- AC-1.4: Multiple paths supported: `gpu-query ~/data/ ~/logs/`

### US-2: SQL Query Execution on GPU
As a data engineer, I want to write SQL queries (SELECT, WHERE, GROUP BY, ORDER BY, LIMIT, aggregates) that execute entirely on the GPU, so that I get results 10-100x faster than CPU engines.

**Acceptance Criteria**:
- AC-2.1: SELECT with column list and * supported
- AC-2.2: WHERE with comparison operators (=, <, >, <=, >=, !=), IS NULL, IS NOT NULL, BETWEEN, IN
- AC-2.3: GROUP BY with one or more columns
- AC-2.4: Aggregate functions: COUNT(*), COUNT(col), SUM, AVG, MIN, MAX
- AC-2.5: ORDER BY with ASC/DESC on one or more columns
- AC-2.6: LIMIT N clause
- AC-2.7: DESCRIBE table shows column stats (type, null%, cardinality, min/max, sample)
- AC-2.8: Compound predicates with AND/OR

### US-3: Performance Visibility
As a data engineer, I want to see GPU utilization, scan throughput, wall-clock time, and CPU comparison after every query, so that I can understand and demonstrate the GPU speed advantage.

**Acceptance Criteria**:
- AC-3.1: Every query result shows: rows scanned, GB processed, wall-clock time (ms), GPU utilization %, effective throughput (GB/s)
- AC-3.2: CPU comparison estimate shown on every query (`~Nx vs CPU`)
- AC-3.3: Warm vs cold query distinction displayed transparently
- AC-3.4: `.benchmark` command runs same query on CPU and GPU, shows side-by-side comparison
- AC-3.5: `.profile` mode shows per-kernel pipeline timeline (Parse, Filter, Agg, Sort times)

### US-4: TUI Dashboard
As a data engineer, I want a full-featured TUI dashboard with data catalog, query editor, results viewer, and GPU status panel, so that I have an interactive analytics environment in my terminal.

**Acceptance Criteria**:
- AC-4.1: Three-panel layout: Data Catalog (left) | Query Editor (top-right) | Results (bottom-right)
- AC-4.2: GPU Status panel in left panel bottom with utilization bar, throughput, memory, sparkline history
- AC-4.3: Gradient color rendering (thermal GPU bars, glow effects on throughput numbers)
- AC-4.4: Syntax highlighting for SQL (keywords, identifiers, literals, operators)
- AC-4.5: Rich autocomplete: table names, column names with types + cardinality, SQL keywords, functions
- AC-4.6: Multi-line query editor with Ctrl+Enter to execute
- AC-4.7: Scrollable results table with pagination for large result sets
- AC-4.8: Responsive layout adapts to terminal width (120+ = full, 80-119 = two-panel, <80 = minimal)
- AC-4.9: Works in iTerm2, Terminal.app, Alacritty, WezTerm, tmux

### US-5: Non-Interactive/Script Mode
As a data engineer, I want to run queries non-interactively via CLI flags and pipe output, so that I can integrate gpu-query into scripts and pipelines.

**Acceptance Criteria**:
- AC-5.1: `-e "SQL"` executes query and exits
- AC-5.2: `-f query.sql` executes SQL from file
- AC-5.3: `--format csv/json/table` controls output format
- AC-5.4: Pipe input: `echo "SQL" | gpu-query ./data/`
- AC-5.5: `-o output.csv` writes results to file

### US-6: Parquet File Support
As a data engineer, I want to query Parquet files with column pruning and row group skipping, so that only relevant data is loaded from disk.

**Acceptance Criteria**:
- AC-6.1: Reads Parquet metadata (schema, row groups) via `parquet` crate
- AC-6.2: Column pruning: only queried columns loaded
- AC-6.3: PLAIN and DICTIONARY encoding decoded on GPU
- AC-6.4: Null bitmap correctly handled for nullable columns

### US-7: CSV File Support
As a data engineer, I want to query CSV files with GPU-parallel parsing, so that large CSV files are processed at GPU bandwidth speeds.

**Acceptance Criteria**:
- AC-7.1: GPU-parallel row boundary detection
- AC-7.2: GPU-parallel field extraction and type coercion
- AC-7.3: Header row detected and used for column names
- AC-7.4: Configurable delimiter (default comma)
- AC-7.5: Quoted fields handled correctly

### US-8: JSON File Support
As a data engineer, I want to query NDJSON files with GPU-parallel structural indexing, so that JSON data is queryable at GPU speeds.

**Acceptance Criteria**:
- AC-8.1: NDJSON (newline-delimited JSON) format supported
- AC-8.2: GPU-parallel structural character detection
- AC-8.3: Column extraction from top-level JSON fields
- AC-8.4: Type coercion for numeric, string, boolean, null JSON values

### US-9: Error Handling
As a data engineer, I want clear error messages for SQL errors, missing tables, type mismatches, and GPU failures, so that I can fix issues quickly.

**Acceptance Criteria**:
- AC-9.1: SQL parse errors show caret pointing to error position with suggestion
- AC-9.2: Table not found lists available tables
- AC-9.3: Column not found shows fuzzy-match suggestion
- AC-9.4: GPU watchdog timeout auto-retries with batched dispatches
- AC-9.5: Malformed file rows skipped with count reported
- AC-9.6: Permission denied shows file path

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Directory scanning with format auto-detection (magic bytes + extension) | Must | US-1 |
| FR-2 | mmap + makeBuffer(bytesNoCopy:) zero-copy file loading [KB #224] | Must | US-1, US-2 |
| FR-3 | GPU-parallel CSV parser (newline detection + field extraction) | Must | US-7 |
| FR-4 | GPU-parallel JSON structural indexer (NDJSON) | Must | US-8 |
| FR-5 | Parquet metadata reader + GPU column decoder (PLAIN, DICTIONARY) | Must | US-6 |
| FR-6 | GPU schema inference kernel (type voting from sample data) | Must | US-1 |
| FR-7 | SQL parser via sqlparser-rs (SELECT, WHERE, GROUP BY, ORDER BY, LIMIT) | Must | US-2 |
| FR-8 | Custom logical planner with column pruning and predicate pushdown | Must | US-2 |
| FR-9 | Physical planner mapping logical ops to Metal kernel graph | Must | US-2 |
| FR-10 | Function constant specialization for query-specific kernel variants [KB #210] | Must | US-2 |
| FR-11 | GPU filter kernel with function-constant comparison operators | Must | US-2 |
| FR-12 | GPU aggregation kernel with 3-level hierarchical reduction [KB #188] | Must | US-2 |
| FR-13 | GPU radix sort kernel for ORDER BY and GROUP BY [KB #388] | Must | US-2 |
| FR-14 | Indirect dispatch for variable-size query outputs [KB #277] | Must | US-2 |
| FR-15 | Columnar SoA storage with per-column null bitmaps (1-bit, Arrow-compatible) | Must | US-2 |
| FR-16 | Adaptive dictionary encoding for strings (<10K distinct: dict, else offset+data) | Must | US-2 |
| FR-17 | PSO cache keyed by (kernel_name, function_constants) | Must | US-2 |
| FR-18 | Batched query execution for >1GB scans (GPU watchdog safety) [KB #441] | Must | US-2 |
| FR-19 | ratatui TUI dashboard with three-panel layout | Must | US-4 |
| FR-20 | Gradient color rendering system (thermal bars, glow effects) | Must | US-4 |
| FR-21 | SQL syntax highlighting in query editor | Must | US-4 |
| FR-22 | Rich autocomplete (tables, columns with types + cardinality, keywords, functions) | Must | US-4 |
| FR-23 | GPU status panel (utilization bar, throughput, memory, sparkline history) | Must | US-3 |
| FR-24 | Per-query performance line (rows, GB, ms, GPU%, GB/s, CPU comparison) | Must | US-3 |
| FR-25 | CPU comparison estimate on every query | Must | US-3 |
| FR-26 | CLI non-interactive mode (-e, -f, --format, -o) | Must | US-5 |
| FR-27 | Dot commands (.tables, .schema, .describe, .gpu, .profile, .benchmark) | Must | US-3, US-4 |
| FR-28 | Query history persistence (~/.config/gpu-query/history) | Should | US-4 |
| FR-29 | Configuration file (~/.config/gpu-query/config.toml) | Should | US-4 |
| FR-30 | Data catalog tree view (directory > table > columns with types) | Must | US-4 |
| FR-31 | Compound filter predicates (AND/OR) via chained bitmask operations | Must | US-2 |
| FR-32 | DESCRIBE command with GPU-computed column statistics | Should | US-2 |
| FR-33 | Warm/cold query timing transparency | Should | US-3 |
| FR-34 | Streaming results pagination for large result sets | Should | US-4 |
| FR-35 | Profile mode with per-kernel pipeline timeline | Should | US-3 |
| FR-36 | madvise(MADV_WILLNEED) pre-warming for cold queries [KB #226] | Should | US-2 |
| FR-37 | Fallback copy path when bytesNoCopy fails | Should | US-2 |
| FR-38 | build.rs AOT Metal shader compilation [KB #159] | Must | US-2 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Warm query latency <5ms for 1B rows simple filter (M4 Pro) | Performance |
| NFR-2 | Warm scan throughput >=80 GB/s (M4 base), >=400 GB/s (M4 Max) | Performance |
| NFR-3 | CPU utilization <1% during query execution (0.3% target) | Performance |
| NFR-4 | GPU utilization >80% during query execution | Performance |
| NFR-5 | Startup time (zero-to-query) <3 seconds | Performance |
| NFR-6 | Single binary, zero runtime dependencies | Deployment |
| NFR-7 | Apple Silicon only (M1-M5), Metal 3 baseline | Compatibility |
| NFR-8 | Terminal support: iTerm2, Terminal.app, Alacritty, WezTerm, tmux, SSH | Compatibility |
| NFR-9 | Graceful degradation: true-color -> 256-color -> 16-color -> monochrome | Compatibility |
| NFR-10 | Memory overhead <2x file size for columnar index | Resource |
| NFR-11 | 15% throughput regression blocks merge; 5% triggers warning | Quality |
| NFR-12 | All #[repr(C)] structs have byte-level layout tests | Quality |
| NFR-13 | ~410 tests: ~160 unit (CPU), ~110 GPU integration, ~105 E2E, ~30 perf, ~5 fuzz | Quality |

## Hardware-Tiered Performance Targets

| Hardware | Unified Memory | Target Dataset | Warm Scan | E2E Filter+Agg |
|----------|---------------|----------------|-----------|----------------|
| M4 base (10 GPU cores) | 16 GB | 100M rows (~3 GB) | >=80 GB/s | <10ms |
| M4 Pro (20 cores) | 24-48 GB | 1B rows (~30 GB) | >=120 GB/s | <7ms |
| M4 Max (40 cores) | 36-128 GB | 10B rows (~300 GB) | >=400 GB/s | <3ms |

## Out of Scope

- Server mode, networking, multi-user
- Write/mutation of source files (INSERT/UPDATE/DELETE)
- JOINs, subqueries, CTEs, window functions (Phase 2 via DataFusion)
- Export to file (Phase 2)
- Embeddable library API (Phase 2)
- Cross-platform (Vulkan port for Linux/Windows)
- Python/Node bindings
- Persistent storage or caching between sessions
- GPU-to-GPU cluster support

## Dependencies

- `objc2-metal` 0.3 -- Rust Metal bindings
- `sqlparser` 0.55 -- SQL parser (used by DataFusion)
- `parquet` 55.0 -- Apache Parquet reader (metadata only)
- `ratatui` 0.29 -- TUI framework
- `crossterm` 0.28 -- Terminal backend
- `clap` 4 -- CLI argument parsing
- `libc` 0.2 -- mmap, madvise
- `block2` 0.6 -- Objective-C blocks for Metal callbacks
- `dispatch2` 0.3 -- Grand Central Dispatch
