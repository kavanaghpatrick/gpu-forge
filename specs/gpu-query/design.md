---
spec: gpu-query
phase: design
created: 2026-02-10
generated: auto
---

# Design: gpu-query

## Overview

GPU-first analytics engine where CPU orchestrates zero-copy data ingestion and query planning (<2ms), while the GPU executes all data processing through a pipeline of 8+ Metal compute kernels. Exploits Apple Silicon UMA to wrap mmap'd files as Metal buffers without DMA transfer [KB #73, #248].

## System Architecture

```
+=========================================================================+
|                            gpu-query Process                             |
|                                                                          |
|  +---------------------------+     +----------------------------------+  |
|  |     TUI Layer (ratatui)   |     |    CLI / Script Mode             |  |
|  |  +--------+ +-----------+ |     |  -e "SQL" | -f query.sql        |  |
|  |  | Editor | | Results   | |     |  --format csv/json/parquet      |  |
|  |  +--------+ +-----------+ |     +----------------------------------+  |
|  |  | Catalog| | GPU Dash  | |                   |                       |
|  |  +--------+ +-----------+ |                   |                       |
|  +-------------|-------------+                   |                       |
|                | SQL string                      | SQL string            |
|  +-------------v---------------------------------v--------------------+  |
|  |                  Query Compiler (CPU, <2ms)                        |  |
|  |  sqlparser-rs --> Logical Plan --> Physical Plan --> Kernel Graph  |  |
|  |  [column pruning] [predicate pushdown] [type inference]           |  |
|  +-------------------------------|------------------------------------+  |
|                                  | KernelGraph + FunctionConstants      |
|  +-------------------------------v------------------------------------+  |
|  |              GPU Execution Engine (Metal Compute)                  |  |
|  |                                                                    |  |
|  |  Command Buffer (single per query, batched for >1GB)              |  |
|  |  +----------+  +---------+  +-----------+  +-------+  +--------+ |  |
|  |  | Parse    |->| Filter  |->| Aggregate |->| Sort  |->| Output | |  |
|  |  | Kernel   |  | Kernel  |  |  Kernel   |  | Kernel|  | Kernel | |  |
|  |  +----------+  +---------+  +-----------+  +-------+  +--------+ |  |
|  |       ^                                                           |  |
|  |       | zero-copy (bytesNoCopy)                                   |  |
|  |  +----+-------------------------------------------------------+  |  |
|  |  |          Columnar Buffer Pool (SoA, GPU-shared)             |  |  |
|  |  |  [INT64] [FLOAT64] [VARCHAR dict] [BOOL] [DATE] [NULL bmp] |  |  |
|  |  +----+-------------------------------------------------------+  |  |
|  +-------|------------------------------------------------------------+  |
|          | mmap (demand-paged, MADV_WILLNEED)                            |
|  +-------v------------------------------------------------------------+  |
|  |                   File System Layer                                |  |
|  |  mmap registry --> page-aligned buffers --> Metal bytesNoCopy      |  |
|  |  Format detection: magic bytes + extension                         |  |
|  +--------------------------------------------------------------------+  |
|                                                                          |
+============================|=============================================+
                             | mmap (zero-copy)
                +------------v-----------+
                |    Local Files (SSD)    |
                |  *.parquet  *.csv       |
                |  *.json  *.log  *.txt   |
                +-------------------------+
```

## Data Flow

```
1. SQL text --(CPU)--> sqlparser-rs AST              [<0.1ms]
2. AST --(CPU)--> Logical Plan (column prune, etc.)  [<0.1ms]
3. Logical Plan --(CPU)--> Physical Plan             [<0.1ms]
4. Physical Plan --(CPU)--> Metal PSO selection      [<0.1ms, cached]
5. Encode kernel graph into command buffer           [<0.2ms]
6. GPU executes kernel pipeline                      [0.5-50ms]
7. Results in shared memory (zero-copy read by CPU)  [0.0ms transfer]
8. TUI formats and displays results                  [<1ms]
```

## Components

### Zero-Copy I/O Pipeline

**Purpose**: Map local files directly into GPU-accessible memory with zero data copying.

```
open() --> mmap(MAP_SHARED, PROT_READ) --> madvise(MADV_WILLNEED) --> makeBuffer(bytesNoCopy:)
```

**Key implementation** (adapted from TECH.md):

```rust
pub struct MmapFile {
    fd: RawFd,
    ptr: *mut c_void,       // 16KB page-aligned mmap pointer [KB #89]
    len: usize,             // file size (rounded to page boundary)
    metal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}
```

- Pointer MUST be 16KB page-aligned (ARM64 hardware page size) [KB #89, #223]
- `StorageModeShared` for CPU+GPU access on UMA [KB #73]
- Fallback: copy to Metal-allocated shared buffer if bytesNoCopy fails (~1ms/GB)
- TLB optimization: reuse mmap regions across queries [KB #595]

### Query Compiler

**Purpose**: Transform SQL text into Metal compute kernel dispatch plan.

**Pipeline**: sqlparser-rs AST -> Logical Plan -> Physical Plan -> PSO selection -> Command Buffer

**Logical planner operations**:
- Column pruning: only load columns referenced in query
- Predicate pushdown: push WHERE filters close to scan
- Type checking: validate column types vs operations

**Physical planner**:
- Map logical operators to Metal kernel variants
- Generate MTLFunctionConstantValues per query [KB #210, #202]
- PSO cache: `HashMap<(KernelName, FunctionConstantHash), MTLComputePipelineState>` [KB #159]

**MVP SQL subset**:
```sql
SELECT col1, col2, agg(col3) FROM table
WHERE predicate [AND|OR predicate]*
GROUP BY col1, col2
ORDER BY col1 [ASC|DESC]
LIMIT n
-- Aggregates: COUNT(*), COUNT(col), SUM, AVG, MIN, MAX
-- Predicates: =, <, >, <=, >=, !=, IS NULL, IS NOT NULL, BETWEEN, IN
-- DESCRIBE table, .tables, .schema, .gpu
```

### GPU Kernel Architecture

All data processing runs as Metal compute kernels. Tile-based execution model adapted from Crystal+ [KB #311, #470] for Apple Silicon SIMD-32.

#### Kernel 1: CSV Parser

**Algorithm** (3-phase, adapted from nvParse):

1. **Newline Detection**: Each thread scans 4KB chunk for `\n`, accounting for quoted strings. Output: `row_offsets[]` via prefix scan [KB #193].
2. **Field Boundary Detection**: Each thread processes one row, scanning for delimiters. Output: `field_offsets[row][col]`.
3. **Type Coercion + Column Write**: Each thread converts one field to typed value, writes to SoA columns.

```metal
kernel void csv_parse_newlines(
    device const char*       file_data      [[buffer(0)]],
    device uint*             row_offsets    [[buffer(1)]],
    device atomic_uint*      row_count      [[buffer(2)]],
    constant CsvParseParams& params         [[buffer(3)]],
    uint                     tid            [[thread_position_in_grid]]
);

kernel void csv_parse_fields(
    device const char*       file_data      [[buffer(0)]],
    device const uint*       row_offsets    [[buffer(1)]],
    device int64_t*          col_int64      [[buffer(2)]],
    device float*            col_float64    [[buffer(3)]],
    device uint*             col_str_offsets[[buffer(4)]],
    device char*             col_str_data   [[buffer(5)]],
    constant CsvParseParams& params         [[buffer(6)]],
    constant ColumnSchema*   schema         [[buffer(7)]],
    uint                     tid            [[thread_position_in_grid]]
);
```

#### Kernel 2: JSON Parser

**Algorithm** (adapted from GpJSON [KB #472]):

1. **Structural Index**: Parallel scan for `{ } [ ] , : "`, build bitmask, handle escape sequences.
2. **Token Classification**: Classify tokens (key, string, number, bool, null), build key-value offsets.
3. **Column Extraction**: Each thread extracts one field from one record to SoA columns.

```metal
kernel void json_structural_index(
    device const char*        file_data       [[buffer(0)]],
    device uint*              structural_bitmask [[buffer(1)]],
    device atomic_uint*       struct_count    [[buffer(2)]],
    constant JsonParseParams& params          [[buffer(3)]],
    uint                      tid             [[thread_position_in_grid]]
);

kernel void json_extract_columns(
    device const char*        file_data       [[buffer(0)]],
    device const uint*        structural_bitmask [[buffer(1)]],
    device const uint*        row_offsets     [[buffer(2)]],
    device void*              output_columns  [[buffer(3)]],
    constant JsonParseParams& params          [[buffer(4)]],
    constant ColumnSchema*    target_paths    [[buffer(5)]],
    uint                      tid             [[thread_position_in_grid]]
);
```

#### Kernel 3: Parquet Decoder

CPU reads Parquet metadata (footer, row groups, column chunk offsets) via `parquet` crate. GPU decodes compressed column data.

```metal
kernel void parquet_decode_plain_int64(
    device const char*           compressed_data [[buffer(0)]],
    device int64_t*              output_column   [[buffer(1)]],
    device uint8_t*              null_bitmap     [[buffer(2)]],
    constant ParquetChunkParams& params          [[buffer(3)]],
    uint                         tid             [[thread_position_in_grid]]
);

kernel void parquet_decode_dictionary(
    device const char*           dict_page       [[buffer(0)]],
    device const uint*           index_page      [[buffer(1)]],
    device void*                 output_column   [[buffer(2)]],
    constant ParquetChunkParams& params          [[buffer(3)]],
    uint                         tid             [[thread_position_in_grid]]
);
```

Note: MVP decompresses Parquet on CPU (parquet crate handles SNAPPY/ZSTD). GPU reads decompressed columns via mmap. GPU decompression kernels in Phase 2.

#### Kernel 4: Column Filter (WHERE)

Function-constant-specialized predicate evaluation [KB #210, #202]:

```metal
constant uint COMPARE_OP   [[function_constant(0)]];  // 0=EQ,1=LT,2=GT,3=LE,4=GE,5=NE
constant uint COLUMN_TYPE  [[function_constant(1)]];   // 0=INT64,1=FLOAT64,2=VARCHAR,3=BOOL
constant bool HAS_NULL_CHECK [[function_constant(2)]];

kernel void column_filter(
    device const void*      column_data     [[buffer(0)]],
    device const uint8_t*   null_bitmap     [[buffer(1)]],
    device uint*            selection_mask  [[buffer(2)]],   // 1-bit per row
    device atomic_uint*     match_count     [[buffer(3)]],
    constant FilterParams&  params          [[buffer(4)]],
    uint                    tid             [[thread_position_in_grid]],
    uint                    simd_lane       [[thread_index_in_simdgroup]]
);
```

Function constants eliminate branches at compile time -- 84% instruction reduction [KB #202]. Compound predicates: AND = bitwise AND of two selection masks; OR = bitwise OR.

#### Kernel 5: Aggregation (GROUP BY + Aggregates)

Three-level hierarchical reduction [KB #188, #328]:

```
Level 1: SIMD-group reduction (simd_sum/min/max, 32 threads -> 1 partial)
         No barrier needed (lockstep) [KB #336]
Level 2: Threadgroup reduction via threadgroup memory
         threadgroup_barrier(mem_flags::mem_threadgroup) [KB #185]
Level 3: Global atomic update [KB #283]
```

```metal
kernel void aggregate_sum_int64(
    device const int64_t*   column_data      [[buffer(0)]],
    device const uint*      selection_mask   [[buffer(1)]],
    device atomic_long*     global_sum       [[buffer(2)]],
    device atomic_uint*     global_count     [[buffer(3)]],
    constant AggParams&     params           [[buffer(4)]],
    uint                    tid              [[thread_position_in_grid]],
    uint                    simd_lane        [[thread_index_in_simdgroup]],
    uint                    simd_id          [[simdgroup_index_in_threadgroup]],
    uint                    tg_size          [[threads_per_threadgroup]]
);
```

**GROUP BY**: Low cardinality (<256 groups) uses threadgroup-local hash table of 256 buckets. High cardinality uses radix sort + scan.

#### Kernel 6: Radix Sort (ORDER BY)

4-bit radix sort following Linebender/FidelityFX pattern [KB #388]:

```
For each 4-bit digit (16 passes for 64-bit keys):
  1. Local histogram: threadgroup digit frequencies via simdgroup ballot
  2. Global prefix scan: exclusive scan via simd_prefix_exclusive_sum [KB #193]
  3. Scatter: each element written to sorted position
```

Performance: ~3B elements/sec on M1 Max [KB #388]. 100M rows: ~33ms. 1B rows: ~330ms.

```metal
kernel void radix_sort_histogram(
    device const uint64_t* keys       [[buffer(0)]],
    device uint*           histograms [[buffer(1)]],
    constant SortParams&   params     [[buffer(2)]],
    uint                   tid        [[thread_position_in_grid]],
    uint                   simd_lane  [[thread_index_in_simdgroup]],
    threadgroup uint*      local_hist [[threadgroup(0)]]
);

kernel void radix_sort_scatter(
    device const uint64_t* keys_in       [[buffer(0)]],
    device uint64_t*       keys_out      [[buffer(1)]],
    device const uint*     values_in     [[buffer(2)]],
    device uint*           values_out    [[buffer(3)]],
    device const uint*     global_offsets[[buffer(4)]],
    constant SortParams&   params        [[buffer(5)]],
    uint                   tid           [[thread_position_in_grid]]
);
```

#### Kernel 7: Prepare Dispatch (Indirect Dispatch Sizing)

GPU-side computation of dispatch arguments for downstream kernels [KB #277]. Same pattern as particle-system `prepare_dispatch.metal`.

```metal
kernel void prepare_query_dispatch(
    device const atomic_uint* match_count  [[buffer(0)]],
    device DispatchArgs*      agg_dispatch [[buffer(1)]],
    device DispatchArgs*      sort_dispatch[[buffer(2)]],
    uint                      tid          [[thread_position_in_grid]]
);
```

#### Kernel 8: Schema Inference

GPU-parallel type detection from sample data (first 10,000 rows):

```metal
kernel void infer_schema(
    device const char*      file_data    [[buffer(0)]],
    device const uint*      field_offsets[[buffer(1)]],
    device atomic_uint*     type_votes   [[buffer(2)]],  // [col][type] matrix
    constant InferParams&   params       [[buffer(3)]],
    uint                    tid          [[thread_position_in_grid]]
);
```

CPU reads vote matrix, picks majority type per column with promotion rules (INT->FLOAT if mixed).

### Columnar Storage Engine

**Type system**:

| SQL Type | GPU Type | Bytes/Value | Null Handling |
|----------|---------|-------------|--------------|
| BIGINT/INT64 | int64_t | 8 | Bitmap (1-bit) |
| INTEGER/INT32 | int32_t | 4 | Bitmap |
| DOUBLE/FLOAT64 | double | 8 | Bitmap |
| VARCHAR | dict_idx (uint32) | 4 + variable heap | Bitmap |
| BOOLEAN | uint8_t (packed) | 1 bit | Bitmap |
| DATE | int32_t (days since epoch) | 4 | Bitmap |

**SoA Layout** (matching GPU 128-byte cache line access [KB #266]):
```
Table "sales" (1M rows):
  Buffer 0: order_id    [int64_t x 1M]   =  8 MB
  Buffer 1: customer_id [int32_t x 1M]   =  4 MB
  Buffer 2: region      [uint32_t x 1M]  =  4 MB  (dictionary index)
  Buffer 3: amount      [double x 1M]    =  8 MB
  Buffer 4: null_bitmaps[uint8_t x N]    = ~625 KB
  Buffer 5: region_dict [char x var]     = ~100 B  (8 unique strings)
  Total: ~24.6 MB for 1M rows (vs ~300 MB raw CSV)
```

**String storage**: Adaptive dictionary encoding. Dictionary for low-cardinality (<10K distinct values) -- GROUP BY on strings = GROUP BY on uint32 indices. Offset+data buffer for high-cardinality columns.

**Null bitmap**: 1 bit per row, packed into uint8_t arrays (Arrow-compatible). Function constants eliminate null check for NOT NULL columns [KB #202].

### Memory Management

**Buffer pool architecture**:
- MmapRegistry: path -> MmapFile, refcount, persistent across queries
- ColumnBuffers: table.col -> typed SoA buffer
- TempBuffers: scratch pool for filter/sort intermediates, recycled (grow-only)

**Memory budget**:

| Component | 1M rows | 100M rows | 1B rows |
|-----------|---------|-----------|---------|
| mmap virtual | ~300 MB | ~30 GB | ~300 GB |
| Columnar (parsed) | ~28 MB | ~2.8 GB | ~28 GB |
| Temp buffers | ~16 MB | ~1.6 GB | ~16 GB |
| **Total resident** | **~344 MB** | **~34 GB** | **memory-limited** |

**Pressure handling**: Column pruning -> row group streaming -> mmap eviction (OS) -> user warning.

**GPU memory monitoring**: `device.currentAllocatedSize()` vs `device.recommendedMaxWorkingSetSize()`. Warn >70%, streaming mode >90%.

### Persistent Engine Architecture

**Session lifecycle**:

```
Startup:
  1. Create Metal device + command queue (one-time)
  2. Pre-compile base PSOs (parse, filter, agg, sort kernels) [KB #159]
  3. Scan directory, mmap all files (lazy: no page faults yet)
  4. GPU-parallel DESCRIBE on all tables (schema + cardinality)
  5. Enter TUI event loop

Per-Query:
  1. Parse SQL (CPU, <0.1ms)
  2. Plan (CPU, <0.1ms)
  3. Lookup/create PSO for function constants (<1ms or cached)
  4. Encode command buffer (CPU, <0.2ms)
  5. GPU executes (0.5-50ms)
  6. Completion handler updates TUI (<1ms)

Between Queries:
  - Metal device stays resident
  - PSO cache stays warm
  - mmap regions stay mapped (OS page cache retains data)
  - Column buffers LRU eviction
```

**Command buffer strategy** [KB #276, #152]:
- One command buffer per query: all kernels encoded sequentially, implicit barriers
- Batching for >1GB: split into 1GB chunks, pre-enqueue ordering [KB #152]
- Async via completion handlers for TUI mode

### TUI Rendering Architecture

**Technology**: ratatui 0.29+ (Rust, 60fps immediate-mode) + crossterm backend

**MVC pattern**:
- **Model**: AppState (query_text, results, catalog, gpu_metrics, mode)
- **View**: Widget tree (rendered each frame via ratatui)
- **Controller**: Event loop processing keyboard/mouse/GPU-callback events

**Dashboard layout** (default, 120+ cols):
```
+-------------------+---------------------------------------------------+
|  DATA CATALOG     |  QUERY EDITOR                               [1/3]|
|  > dir/           |  SELECT region, SUM(amount)...                    |
|    table [type]   |  [Tab: autocomplete] [Ctrl+Enter: run]            |
|    col   type     +---------------------------------------------------+
|                   |  RESULTS                          10 rows | 2.3ms |
|  GPU STATUS       |  col1 | col2 | col3                               |
| [||||||||  ] 94%  |  val  | val  | val                                |
| Scan: 3,652 GB/s  |                                                   |
| Mem:  8.4/24 GB   |  142M rows | 8.4 GB | 2.3ms | GPU 94% | ~312x   |
+-------------------+---------------------------------------------------+
```

**Gradient rendering system** (true-color via `Style::fg(Color::Rgb(r, g, b))`):
- GPU utilization bar: green->yellow->red thermal gradient
- Throughput numbers: "GPU green" glow for >100 GB/s [KB #477]
- Sparkline history for last 20 queries

**Color degradation**: TrueColor -> 256-color -> Basic16 -> Monochrome (NO_COLOR support)

**TUI-GPU synchronization**: TUI never blocks on GPU. Completion handler sends result via channel. TUI renders "Executing..." spinner during query execution.

## Technical Decisions

| Decision | Options | Choice | Rationale |
|----------|---------|--------|-----------|
| Dispatch model | Persistent kernels vs batched | Batched dispatches | Metal has no persistent kernels [KB #440]; watchdog kills long shaders [KB #441] |
| Memory layout | Row-major vs columnar SoA | Columnar SoA | Matches GPU cache line access [KB #266]; standard for GPU databases [KB #311] |
| Aggregation | Naive atomic vs hierarchical | 3-level hierarchical | SIMD->TG->global reduces atomic contention 256x [KB #188, #283] |
| String storage | Always-dict vs threshold | Adaptive (<10K dict) | Optimizes both low and high cardinality; two code paths via function constants |
| Query specialization | JIT vs function constants | Function constants [KB #210] | 84% instruction reduction; proven in MLX/llama.cpp; no runtime compilation |
| SQL parser | Custom vs sqlparser-rs | sqlparser-rs | Battle-tested, used by DataFusion; custom minimal planner for MVP |
| Null handling | Sentinel vs bitmap | 1-bit bitmap (Arrow) | Arrow compatibility for DataFusion Phase 2; trivial overhead with function constants |
| Metal API | Metal 3 vs Metal 4 | Metal 3 baseline | Maximum M1+ compatibility; Metal 4 opt-in Phase 2 [KB #442] |
| Shader compilation | Runtime vs AOT | AOT via build.rs | Proven in particle-system; zero startup compilation; function constants at runtime |
| Result buffering | Triple vs double | Double buffer | Queries are user-initiated (not 60fps continuous); double buffer suffices |

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `gpu-query/Cargo.toml` | Create | Package with Metal, SQL, TUI, Parquet deps |
| `gpu-query/build.rs` | Create | AOT Metal shader compilation (reuse particle-system pattern) |
| `gpu-query/src/main.rs` | Create | Entry point, CLI arg parsing (clap) |
| `gpu-query/src/lib.rs` | Create | Public library interface |
| `gpu-query/src/cli/args.rs` | Create | Argument parsing (clap derive) |
| `gpu-query/src/cli/commands.rs` | Create | Dot commands (.tables, .schema, etc.) |
| `gpu-query/src/sql/parser.rs` | Create | sqlparser-rs integration |
| `gpu-query/src/sql/logical_plan.rs` | Create | Logical plan nodes (Scan, Filter, Agg, Sort, Limit) |
| `gpu-query/src/sql/physical_plan.rs` | Create | Physical plan (kernel graph) |
| `gpu-query/src/sql/optimizer.rs` | Create | Column pruning, predicate pushdown |
| `gpu-query/src/sql/types.rs` | Create | DataType enum, schema definitions |
| `gpu-query/src/sql/expressions.rs` | Create | Expression tree (BinaryOp, Literal, ColumnRef) |
| `gpu-query/src/gpu/device.rs` | Create | Metal device + command queue init |
| `gpu-query/src/gpu/pipeline.rs` | Create | PSO creation + caching + function constants |
| `gpu-query/src/gpu/executor.rs` | Create | Query execution engine (encode + submit) |
| `gpu-query/src/gpu/buffers.rs` | Create | Buffer pool + allocation + recycling |
| `gpu-query/src/gpu/metrics.rs` | Create | GPU metrics collection (Metal Counters) [KB #258] |
| `gpu-query/src/gpu/encode.rs` | Create | Compute encoder helpers (per-kernel) |
| `gpu-query/src/gpu/types.rs` | Create | GPU-side types (#[repr(C)] structs) |
| `gpu-query/src/io/mmap.rs` | Create | mmap + bytesNoCopy wrapper [KB #224] |
| `gpu-query/src/io/format_detect.rs` | Create | File format detection (magic bytes) |
| `gpu-query/src/io/csv.rs` | Create | CSV metadata reader (header, delimiter) |
| `gpu-query/src/io/json.rs` | Create | JSON/NDJSON metadata reader |
| `gpu-query/src/io/parquet.rs` | Create | Parquet metadata reader (footer, schema) |
| `gpu-query/src/io/catalog.rs` | Create | Directory scanner, table registry |
| `gpu-query/src/storage/columnar.rs` | Create | Columnar SoA buffer management |
| `gpu-query/src/storage/dictionary.rs` | Create | Dictionary encoding for strings |
| `gpu-query/src/storage/null_bitmap.rs` | Create | Null bitmap operations |
| `gpu-query/src/storage/schema.rs` | Create | Runtime schema (column names + types) |
| `gpu-query/src/tui/app.rs` | Create | Application state (Model) |
| `gpu-query/src/tui/event.rs` | Create | Event loop + async GPU callbacks |
| `gpu-query/src/tui/ui.rs` | Create | Layout + rendering (View) |
| `gpu-query/src/tui/editor.rs` | Create | Query editor widget |
| `gpu-query/src/tui/results.rs` | Create | Results table widget |
| `gpu-query/src/tui/catalog.rs` | Create | Data catalog tree widget |
| `gpu-query/src/tui/dashboard.rs` | Create | GPU metrics dashboard widget |
| `gpu-query/src/tui/gradient.rs` | Create | Gradient color rendering system |
| `gpu-query/src/tui/themes.rs` | Create | Theme definitions (gradient palettes) |
| `gpu-query/src/tui/autocomplete.rs` | Create | Tab-complete (tables, columns, keywords) |
| `gpu-query/shaders/types.h` | Create | Shared type definitions (Rust <-> MSL) |
| `gpu-query/shaders/csv_parse.metal` | Create | CSV tokenization + field extraction |
| `gpu-query/shaders/json_parse.metal` | Create | JSON structural indexing + extraction |
| `gpu-query/shaders/parquet_decode.metal` | Create | Parquet column chunk decoding |
| `gpu-query/shaders/filter.metal` | Create | Predicate evaluation (function constants) |
| `gpu-query/shaders/aggregate.metal` | Create | Hierarchical reduction |
| `gpu-query/shaders/sort.metal` | Create | Radix sort (histogram, scan, scatter) |
| `gpu-query/shaders/prepare_dispatch.metal` | Create | Indirect dispatch argument computation |
| `gpu-query/shaders/schema_infer.metal` | Create | Type inference from sample data |
| `gpu-query/shaders/dict_build.metal` | Create | Dictionary construction for strings |
| `gpu-query/shaders/compact.metal` | Create | Stream compaction (selection -> dense) |

## Error Handling

| Error | Handling | User Impact |
|-------|----------|-------------|
| SQL syntax error | sqlparser-rs error with position | Caret pointing to error + suggestion |
| Table not found | List available tables | "Available tables: ..." |
| Column not found | Fuzzy-match suggestion | "Did you mean: region?" |
| Type mismatch | Show column type | "Column 'name' is VARCHAR, cannot compare with >" |
| GPU watchdog timeout | Auto-retry with batched dispatches [KB #441] | "Splitting into N batches..." |
| mmap permission denied | Show file path | "Permission denied: /path" |
| bytesNoCopy failure | Fallback to copy mode | "Using buffered mode (slightly slower)" |
| Malformed CSV rows | Skip + count | "Skipped 42 malformed rows (0.003%)" |
| Out of memory | Column pruning warning + streaming | "Exceeds memory. Use SELECT specific_columns" |
| Corrupt Parquet | Graceful error | "Invalid Parquet format: /path" |

## Performance Model (Roofline) [KB #476, #477]

| Kernel | Op Intensity | Bottleneck | M4 Throughput | M4 Max |
|--------|-------------|-----------|--------------|--------|
| Scan (read columns) | ~0.1 FLOPS/byte | Memory BW | 100 GB/s | 546 GB/s |
| Filter (predicate) | ~1 FLOP/byte | Memory BW | 100 GB/s | 546 GB/s |
| Aggregation (SUM) | ~0.5 FLOPS/byte | Memory BW | 100 GB/s | 546 GB/s |
| Sort (radix) | ~4 FLOPS/byte | Memory BW | 100 GB/s | 546 GB/s |
| JSON parse | ~10 FLOPS/byte | Compute | 2.9 TFLOPS | ~14 TFLOPS |

**Expected warm query latency**:

| Query Pattern | 1M rows | 100M rows | 1B rows | 10B rows |
|--------------|---------|-----------|---------|----------|
| SELECT COUNT(*) | <0.1ms | 0.8ms | 8ms | 80ms |
| Filter + scan | 0.1ms | 1ms | 10ms | 100ms |
| Filter + agg + GROUP BY | 0.2ms | 2ms | 20ms | 200ms |
| ORDER BY + LIMIT | 0.5ms | 33ms | 330ms | 3.3s |

## Existing Patterns to Follow

- **particle-system/build.rs**: AOT Metal shader compilation via xcrun [lines 1-98]
- **particle-system/src/types.rs**: `#[repr(C)]` structs with byte-level layout tests [lines 27-385]
- **particle-system/src/buffers.rs**: alloc_buffer helper, StorageModeShared, buffer_ptr [lines 73-92]
- **particle-system/src/encode.rs**: Per-pass encoder pattern (create encoder, set PSO, bind buffers, dispatch, end) [lines 25-55]
- **particle-system/src/gpu.rs**: Device init, find_metallib, PSO creation from named functions [lines 36-184]
- **particle-system/src/types.rs**: DispatchArgs for indirect dispatch [lines 157-170]
