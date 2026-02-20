---
spec: gpu-query-autonomous
phase: design
created: 2026-02-11
generated: auto
---

# Design: gpu-query-autonomous

## Overview

Invert execution model: GPU runs pseudo-persistent polling kernel that owns the query loop; CPU shrinks from orchestrator to parameter writer. New `AutonomousExecutor` sits alongside existing `QueryExecutor`, selected at plan time based on pattern compatibility. Six pillars: persistent kernel (completion-handler re-dispatch), lock-free triple-buffered work queue, fused single-pass kernel, JIT Metal shader compilation, pre-loaded binary columnar data, zero-readback unified memory output.

## Architecture

```
+=========================================================================+
|                         gpu-query Process                                |
|                                                                          |
|  +---------------------------+                                           |
|  |     TUI Layer (ratatui)   |                                           |
|  |  Editor -> parse -> plan -+---> AutonomousExecutor (new, <1ms)        |
|  |  F5/Ctrl+Enter -----------+---> QueryExecutor (existing, 36ms)        |
|  |  Poll ready_flag (tick) --+<--- Unified Memory Output Buffer          |
|  +---------------------------+                                           |
|                                                                          |
|  +=========================== AutonomousExecutor =======================+|
|  |                                                                      ||
|  |  +------------------+     +-----------------------+                  ||
|  |  | JIT Compiler     |     | Work Queue (Triple)   |                  ||
|  |  | PhysicalPlan --> |     | [Slot A] [Slot B] [C] |                  ||
|  |  | Metal source --> |     | write_idx (CPU atomic) |                  ||
|  |  | compile PSO --> |     | read_idx  (GPU atomic) |                  ||
|  |  | cache PSO        |     +-----------+-----------+                  ||
|  |  +------------------+                 |                              ||
|  |                                       v                              ||
|  |  +----------------------------------------------------+             ||
|  |  |        Persistent Kernel (GPU, re-dispatch chain)   |             ||
|  |  |                                                     |             ||
|  |  |  loop (bounded 16ms time-slice) {                   |             ||
|  |  |    poll work_queue[read_idx].sequence_id             |             ||
|  |  |    if (new_query) {                                 |             ||
|  |  |      fused_filter_aggregate_groupby()               |             ||
|  |  |      write output_buffer                            |             ||
|  |  |      set ready_flag                                 |             ||
|  |  |    }                                                |             ||
|  |  |  }                                                  |             ||
|  |  |  -> completion handler re-dispatches immediately    |             ||
|  |  +----------------------------------------------------+             ||
|  |                              |                                       ||
|  |  +---------------------------v--------------------------+            ||
|  |  |          Binary Columnar Data (GPU-Resident)         |            ||
|  |  |  INT64[] | FLOAT32[] | DictIdx[] | NullBitmaps[]     |            ||
|  |  |  16KB page-aligned, StorageModeShared [KB #89]       |            ||
|  |  +------------------------------------------------------+            ||
|  |                              |                                       ||
|  |  +---------------------------v--------------------------+            ||
|  |  |         Output Buffer (Unified Memory)               |            ||
|  |  |  ready_flag (atomic) | result_rows | latency_ns      |            ||
|  |  |  CPU reads via pointer -- zero waitUntilCompleted     |            ||
|  |  +------------------------------------------------------+            ||
|  +======================================================================+|
+=========================================================================+
```

### Data Flow (Warm Autonomous Query)

```
1. User keystroke        -> parse SQL (CPU, ~0.2ms)
2. Parse succeeds        -> build PhysicalPlan (CPU, ~0.1ms)
3. Plan check            -> autonomous-compatible? (CPU, ~0.01ms)
4. JIT cache lookup      -> PSO exists? (CPU, ~0.01ms)
   4a. Cache miss         -> generate Metal source -> compile PSO (~1ms, one-time)
   4b. Cache hit          -> use existing PSO
5. Write QueryParams     -> work_queue[write_idx] (CPU, ~0.01ms)
6. Flip write_idx        -> atomic increment (CPU, ~0.001ms)
7. GPU polls read_idx    -> sees new sequence_id (GPU, 0-0.1ms polling gap)
8. Fused kernel executes -> filter+aggregate+GROUP BY (GPU, ~0.3-0.7ms)
9. Write output_buffer   -> results + ready_flag (GPU, ~0.01ms)
10. TUI tick polls flag  -> reads result (CPU, 0-16ms frame tick)

Total CPU: ~0.35ms | Total GPU: ~0.5ms | Total wall-clock: <1ms
```

## Components

### Component 1: Persistent Kernel (Re-Dispatch Chain)

**Purpose**: Eliminate per-query command buffer creation and `waitUntilCompleted`.

**Design**: Bounded 16ms time-slice kernels with CPU-side re-dispatch via completion handlers [KB #152]. True infinite kernels NOT feasible on Metal due to GPU watchdog [KB #441, #151].

```
Time ───────────────────────────────────────────────>
GPU:  [kernel_0 ~~~~16ms~~~~] [kernel_1 ~~~~16ms~~~~] [kernel_2 ...]
                              ^                       ^
CPU:  commit(cb_0)           complete(cb_0)          complete(cb_1)
                             -> commit(cb_1)         -> commit(cb_2)
                             enqueue(cb_2)           enqueue(cb_3)
Gap: ~0.05-0.1ms between command buffers
```

**Idle Strategy**: Hybrid -- active polling when queries flowing, MTLSharedEvent-driven wake when idle (>500ms no queries). Zero idle power cost.

```rust
impl AutonomousExecutor {
    fn on_command_buffer_complete(&self) {
        if self.has_pending_queries() {
            self.dispatch_next_slice();  // Active: immediate re-dispatch
        } else {
            self.state.store(EngineState::Idle, Ordering::Release);
            // Next submit_query() call signals MTLSharedEvent
        }
    }
}
```

### Component 2: Work Queue (Triple-Buffered)

**Purpose**: Lock-free zero-copy CPU->GPU parameter handoff.

**Layout**: 3 x 512B QueryParamsSlot in unified memory (StorageModeShared).

```
Work Queue Memory Layout (3 x QueryParamsSlot):
+================================================================+
| Offset | Field                  | Size   | Description          |
+--------+------------------------+--------+----------------------+
|      0 | slot[0].sequence_id    |    4B  | Monotonic query ID   |
|      4 | slot[0].query_hash     |    8B  | JIT PSO cache key    |
|     12 | slot[0].filter_count   |    4B  | Number of predicates |
|     16 | slot[0].filters[0..3]  | 4x48B  | Up to 4 filter specs |
|    208 | slot[0].agg_count      |    4B  | Number of agg funcs  |
|    212 | slot[0].aggs[0..4]     | 5x16B  | Up to 5 agg specs    |
|    292 | slot[0].group_by_col   |    4B  | GROUP BY column idx  |
|    296 | slot[0].has_group_by   |    4B  | 0 or 1               |
|    300 | slot[0].row_count      |    4B  | Total rows in table  |
|    304 | slot[0]._padding       |  208B  | Pad to 512B boundary |
|    512 | slot[1]...             |  512B  | Second slot          |
|   1024 | slot[2]...             |  512B  | Third slot           |
+================================================================+
Total: 1536B (3 x 512B)
```

**Atomic Protocol**:
1. CPU writes all fields EXCEPT sequence_id
2. CPU memory barrier (Release)
3. CPU writes sequence_id (atomic store, Release)
4. GPU loads sequence_id (atomic load, Acquire) -- if new, reads all other fields safely

### Component 3: Fused Query Kernel

**Purpose**: Single-pass filter+aggregate+GROUP BY over data -- one memory read per row.

**Structure**:
```
Phase 1: FILTER -- evaluate all predicates for this row (early exit if fails)
Phase 2: GROUP BY BUCKETING -- read group column, hash to bucket (0..63)
Phase 3: AGGREGATE -- threadgroup-local accumulators with simd reductions
Phase 4: GLOBAL REDUCTION -- cross-threadgroup merge via device atomics
```

**Threadgroup Memory**: 64 groups x 160B/group = 10KB (well under 32KB limit [KB #22]).

**Register Pressure Mitigation**:
- JIT generates exact kernel needed (dead code elimination)
- Function constants for AOT fallback [KB #202, #210]
- MAX_GROUPS=64, MAX_AGGS=5 -- compiler unrolls known-bound loops
- Fallback to 2-pass if register pressure >80%

### Component 4: JIT Compiler

**Purpose**: Generate specialized Metal shaders from query plan. Eliminates combinatorial explosion of function-constant PSO variants.

**Pipeline**:
```
PhysicalPlan -> plan_structure_hash() -> PSO cache lookup
  -> cache hit: return cached PSO (~0.01ms)
  -> cache miss: generate_metal_source() -> newLibraryWithSource() -> newComputePipelineState()
     -> cache insertion (~1-2ms one-time)
```

**Key Design**: Hash captures plan STRUCTURE, not literal values. Same structure with different WHERE thresholds shares the same PSO. Literals passed via work queue QueryParamsSlot.

**Fallback**: AOT fused kernel with ~20 function constants if JIT compilation fails.

### Component 5: Binary Columnar Loader

**Purpose**: Pre-load CSV/Parquet data into GPU-resident Metal buffers at startup.

**Buffer Layout**: Separate per-type buffers matching existing ColumnarBatch pattern (int_buffer, float_buffer, dict_buffer). Each column starts at 16-byte boundary; total buffer is 16KB page-aligned [KB #89].

**Loading**: Background thread converts ColumnarBatch to Metal buffers. Progress reported via channel. TUI remains responsive during loading.

### Component 6: AutonomousExecutor

**Purpose**: Central coordinator. Manages lifecycle: Off -> WarmingUp -> Idle -> Active -> Idle cycle.

**Key Methods**:
- `new(device, command_queue)` -- allocate buffers, create MTLSharedEvent
- `load_table(table, schema, batch, progress_tx)` -- background data load
- `submit_query(plan, schema)` -- JIT lookup + write work queue + wake if idle
- `poll_ready() -> bool` -- non-blocking atomic load on ready_flag
- `read_result() -> QueryResult` -- read from unified memory, reset flag
- `shutdown()` -- stop re-dispatch chain

## Shared Struct Definitions (#[repr(C)] + MSL)

### FilterSpec (48 bytes)

```rust
#[repr(C)]
pub struct FilterSpec {
    pub column_idx: u32,         // offset 0
    pub compare_op: u32,         // offset 4  (0=EQ,1=NE,2=LT,3=LE,4=GT,5=GE)
    pub column_type: u32,        // offset 8  (0=INT64, 1=FLOAT32)
    pub _pad0: u32,              // offset 12
    pub value_int: i64,          // offset 16
    pub value_float_bits: u32,   // offset 24
    pub _pad1: u32,              // offset 28
    pub has_null_check: u32,     // offset 32
    pub _pad2: [u32; 3],         // offset 36
}
```

### AggSpec (16 bytes)

```rust
#[repr(C)]
pub struct AggSpec {
    pub agg_func: u32,           // 0=COUNT,1=SUM,2=AVG,3=MIN,4=MAX
    pub column_idx: u32,
    pub column_type: u32,
    pub _pad0: u32,
}
```

### QueryParamsSlot (512 bytes)

```rust
#[repr(C)]
pub struct QueryParamsSlot {
    pub sequence_id: u32,
    pub _pad_seq: u32,
    pub query_hash: u64,
    pub filter_count: u32,
    pub _pad_fc: u32,
    pub filters: [FilterSpec; 4],
    pub agg_count: u32,
    pub _pad_ac: u32,
    pub aggs: [AggSpec; 5],
    pub group_by_col: u32,
    pub has_group_by: u32,
    pub row_count: u32,
    pub _padding: [u8; 208],     // pad to 512B total
}
```

### ColumnMeta (32 bytes)

```rust
#[repr(C)]
pub struct ColumnMeta {
    pub offset: u64,
    pub column_type: u32,        // 0=INT64, 1=FLOAT32, 2=DICT_U32
    pub stride: u32,
    pub null_offset: u64,
    pub row_count: u32,
    pub _pad: u32,
}
```

### OutputBuffer (~22KB)

```rust
#[repr(C)]
pub struct OutputBuffer {
    pub ready_flag: u32,         // offset 0: 0=pending, 1=ready
    pub sequence_id: u32,        // offset 4: echoed from query
    pub latency_ns: u64,         // offset 8: GPU-measured time
    pub result_row_count: u32,   // offset 16
    pub result_col_count: u32,   // offset 20
    pub error_code: u32,         // offset 24
    pub _pad: u32,               // offset 28
    pub group_keys: [i64; 256],  // offset 32
    pub agg_results: [[AggResult; 5]; 256],  // offset 2080
}

#[repr(C)]
pub struct AggResult {
    pub value_int: i64,          // offset 0
    pub value_float: f32,        // offset 8
    pub count: u32,              // offset 12
}
// sizeof = 16 bytes
```

## Technical Decisions

| Decision | Options | Choice | Rationale |
|----------|---------|--------|-----------|
| Persistent kernel model | True persistent / Re-dispatch / ICB-driven | Completion-handler re-dispatch (MVP) | True persistent risks deadlock [KB #151]. Re-dispatch proven, 0.1ms gap within budget. ICB Phase 2 |
| Work queue protocol | Ring buffer / Triple buffer / Single slot | Triple buffer (3 slots) | Lock-free, no torn reads. Proven pattern [Apple triple-buffering] |
| Fused kernel strategy | Function constants only / JIT / Pre-compiled | JIT primary + AOT fallback | JIT gives 84% instruction reduction [KB #202]. AOT catches edge cases |
| GROUP BY limit | 64 / 128 / 256 groups | 64 groups | 10KB threadgroup memory, safe under 32KB [KB #22]. Covers 95% analytics |
| Memory ordering | Relaxed / Acquire-Release / SeqCst | Acquire-Release on sequence_id | Minimal overhead. CPU Release + GPU Acquire ensures visibility [KB #154] |
| Debounce | 0ms / 50ms / Adaptive | 0ms per user decision | Every keystroke fires. GPU work trivial at <1ms |
| Buffer alignment | 4B / 16B / 16KB page | 16B column start, 16KB total buffer | Avoids unaligned access, page-aligned for makeBuffer [KB #89] |
| JIT cache key | Full plan hash / Structure-only | Structure-only hash | Same structure, different literals shares PSO |
| Output buffer | Dynamic / Fixed 256 groups | Fixed 256 groups x 5 aggs | Matches MAX_GROUPS, avoids dynamic GPU allocation |
| Command queue | Shared / Separate | Separate for autonomous | Prevents re-dispatch chain from blocking fallback queries |
| Idle detection | Timer / Event-driven | 500ms timeout + MTLSharedEvent wake | Zero idle power + instant wake |
| Binary columnar layout | Single buffer / Per-type / Per-column | Per-type (matching ColumnarBatch) | Easier migration from existing codebase |
| Float tolerance | Exact / 1e-5 / 1e-3 | Split: exact int, 1e-5 float | Catches real errors, allows GPU reduction order differences |

## File Structure

### New Files to Create

| File | Purpose |
|------|---------|
| `src/gpu/autonomous/mod.rs` | Module root |
| `src/gpu/autonomous/types.rs` | `QueryParamsSlot`, `FilterSpec`, `AggSpec`, `OutputBuffer`, `ColumnMeta`, `AggResult` with layout tests |
| `src/gpu/autonomous/executor.rs` | `AutonomousExecutor` struct + lifecycle + poll_ready + read_result |
| `src/gpu/autonomous/jit.rs` | JIT Metal source generator + compiler + PSO cache |
| `src/gpu/autonomous/loader.rs` | Background binary columnar data loader |
| `src/gpu/autonomous/work_queue.rs` | Triple-buffer work queue protocol |
| `shaders/autonomous_types.h` | MSL-side shared type definitions |
| `shaders/fused_query.metal` | AOT fused kernel (function constant version, fallback) |
| `shaders/check_work.metal` | Work queue check kernel (for re-dispatch chain) |
| `benches/autonomous_latency.rs` | Criterion benchmarks: autonomous vs orchestrated |
| `tests/autonomous_integration.rs` | Integration tests: correctness, parity, fallback |
| `tests/autonomous_stress.rs` | Stress tests: endurance, memory leak, watchdog |

### Existing Files to Modify

| File | Change |
|------|--------|
| `src/gpu/mod.rs` | Add `pub mod autonomous;` |
| `src/tui/app.rs` | Add `AutonomousExecutor`, `EngineStatus`, `SqlValidity`, `QueryCompatibility`, `AutonomousStats` fields |
| `src/tui/event.rs` | Add Ctrl+L handler, live-mode keystroke logic |
| `src/tui/mod.rs` (event loop) | Add autonomous polling, live-mode submission, warm-up progress check |
| `src/tui/dashboard.rs` | Add engine status section (LIVE/WARMING/IDLE/FALLBACK badge) |
| `src/tui/results.rs` | Add `[auto]` tag, microsecond precision, autonomous vs standard comparison |
| `src/tui/ui.rs` | Wire autonomous executor to query execution path |
| `src/gpu/device.rs` | Add `create_shared_event()` helper |
| `build.rs` | Compile new .metal files |

## Error Handling

| Error | Detection | Handling | User Impact |
|-------|-----------|----------|-------------|
| GPU watchdog kills command buffer | `MTLCommandBufferError` in completion handler | Log, re-dispatch kernel chain | Brief [ERROR] flash, auto-recovers ~0.2ms |
| JIT compilation failure | `newLibraryWithSource` error | Fall back to AOT; if AOT fails, fall back to QueryExecutor | [FALLBACK] badge |
| Work queue torn read | Sequence ID mismatch / output corruption | Reset work queue indices, re-submit | Transparent retry |
| GROUP BY cardinality >64 | Plan check before submission | Route to QueryExecutor | [FALLBACK] with reason |
| Out of unified memory | `newBufferWithLength` nil | LRU eviction of resident tables, retry | [WARMING] re-loads reduced |
| Background loader panic | `JoinHandle` returns Err | Standard executor remains | Engine stays [OFF] |

## Existing Patterns to Follow

| Pattern | Location | How We Follow It |
|---------|----------|------------------|
| `#[repr(C)]` + MSL counterparts + offset tests | `src/gpu/types.rs` + `shaders/types.h` | `autonomous_types.rs` + `autonomous_types.h` |
| PSO caching via HashMap | `src/gpu/pipeline.rs` | `HashMap<u64, CompiledPlan>` in JIT |
| Function constant specialization | `shaders/filter.metal` L16-18 | AOT fallback kernel |
| 3-level SIMD reduction (simd_sum_int64) | `shaders/aggregate.metal` | Reuse in fused kernel |
| `GpuDevice` reference | `src/gpu/device.rs` | AutonomousExecutor takes same ref |
| `QueryExecutor` API shape | `src/gpu/executor.rs` | Similar `new()` + `submit_query()` shape |
| `ColumnarBatch` per-type storage | `src/storage/columnar.rs` | Binary loader matches same format |
| `encode::alloc_buffer()` | `src/gpu/encode.rs` | Page-aligned buffer allocation |
