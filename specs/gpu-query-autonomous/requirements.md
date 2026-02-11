---
spec: gpu-query-autonomous
phase: requirements
created: 2026-02-11
generated: auto
---

# Requirements: gpu-query-autonomous

## Summary

Build a GPU-autonomous query engine achieving sub-1ms warm query latency on 1M rows by inverting the execution model: GPU owns the query loop via persistent kernel, CPU writes parameters to shared memory. Six architectural pillars: persistent kernel, lock-free work queue, fused single-pass kernel, JIT shader compilation, pre-loaded binary columnar data, zero-readback output.

## User Stories

### US-1: Sub-Millisecond Warm Query

As a data engineer running iterative queries on cached data, I want compound filter + GROUP BY queries to complete in <1ms so that query latency is imperceptible and I can explore data at the speed of thought.

**Acceptance Criteria**:
- AC-1.1: `SELECT region, count(*), sum(amount) FROM sales WHERE amount > 500 AND region < 3 GROUP BY region` on 1M rows completes in <1ms (warm, data pre-loaded)
- AC-1.2: Measured via criterion benchmark, p50 < 1ms, p99 < 2ms
- AC-1.3: CPU utilization during query < 0.5%
- AC-1.4: No `waitUntilCompleted` in the hot path

### US-2: Persistent GPU Execution Loop

As a system architect, I want the GPU to own the execution loop via a continuously-running compute kernel so that per-query Metal command buffer overhead is eliminated.

**Acceptance Criteria**:
- AC-2.1: Pseudo-persistent kernel runs via completion-handler re-dispatch chain, polling a work queue
- AC-2.2: No `commandBuffer()` or `computeCommandEncoder()` calls per query
- AC-2.3: GPU utilization > 10% while idle (polling cost), > 90% during query execution
- AC-2.4: Kernel survives Metal GPU watchdog via bounded 16ms time-slice + re-dispatch
- AC-2.5: Kernel processes at least 1000 queries without re-initialization

### US-3: Zero-Copy Parameter Handoff

As a performance engineer, I want query parameters passed to GPU via lock-free triple-buffered unified memory so that there is zero synchronization overhead between CPU and GPU.

**Acceptance Criteria**:
- AC-3.1: Triple-buffered work queue allocated in unified memory (StorageModeShared)
- AC-3.2: CPU writes query params to buffer[write_index] without locks or barriers
- AC-3.3: GPU reads from buffer[read_index] without blocking
- AC-3.4: Atomic index swap ensures no torn reads
- AC-3.5: Parameter handoff latency < 0.01ms (measured)

### US-4: Fused Single-Pass Query Kernel

As a GPU compute engineer, I want filter + aggregate + GROUP BY to execute in a single kernel pass so that data is read from memory exactly once per query.

**Acceptance Criteria**:
- AC-4.1: Single kernel function handles filter predicate evaluation, aggregate accumulation, and group-by bucketing
- AC-4.2: Each row is read from memory exactly once (no multi-pass)
- AC-4.3: Threadgroup-local hash tables for GROUP BY with <= 64 groups
- AC-4.4: Simdgroup reductions for aggregate accumulation (SUM, COUNT, MIN, MAX, AVG)
- AC-4.5: Performance within 80% of theoretical memory bandwidth limit

### US-5: Pre-Loaded Binary Columnar Data

As a data engineer, I want data loaded into GPU-resident Metal buffers at startup so that query execution never touches CSV/JSON parsing.

**Acceptance Criteria**:
- AC-5.1: Binary columnar format stored in Metal buffers (INT64, FLOAT32, dictionary-encoded strings)
- AC-5.2: Data loaded once at table registration, persists across queries
- AC-5.3: 1M rows loaded in < 500ms from CSV source
- AC-5.4: No CSV/JSON parsing during query execution path
- AC-5.5: Memory layout is 16KB page-aligned for optimal GPU access [KB #89]

### US-6: Zero-Readback Results

As a TUI developer, I want query results written directly to a unified memory output buffer so that the CPU can read results without any GPU synchronization.

**Acceptance Criteria**:
- AC-6.1: Output buffer allocated in unified memory (StorageModeShared)
- AC-6.2: GPU writes result rows + ready flag to output buffer
- AC-6.3: CPU polls ready flag and reads result data -- no `waitUntilCompleted`
- AC-6.4: Result format is directly renderable (no CPU-side format conversion for numeric types)
- AC-6.5: Output buffer supports up to 256 result groups (GROUP BY cardinality limit)

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Persistent compute kernel with work queue polling via completion-handler re-dispatch chain | Must | US-2 |
| FR-2 | Lock-free triple-buffered work queue (CPU->GPU) with atomic sequence_id protocol | Must | US-3 |
| FR-3 | Fused filter+aggregate+GROUP BY kernel in single pass | Must | US-4 |
| FR-4 | Pre-loaded binary columnar data in Metal buffers, 16KB page-aligned | Must | US-5 |
| FR-5 | Unified memory output buffer (GPU->CPU) with ready_flag polling | Must | US-6 |
| FR-6 | JIT Metal shader compilation from query plan with AOT fallback | Must | US-1 |
| FR-7 | Support compound filters (AND/OR, up to 4 predicates) | Must | US-1 |
| FR-8 | Support 5 aggregate functions (COUNT, SUM, AVG, MIN, MAX) | Must | US-4 |
| FR-9 | Support GROUP BY on integer/dictionary columns (cardinality <= 64) | Must | US-4 |
| FR-10 | Graceful fallback to current QueryExecutor for unsupported queries (ORDER BY, JOINs, subqueries, >64 groups) | Should | US-1, US-2 |
| FR-11 | Separate MTLCommandQueue for autonomous executor | Should | US-2 |

## UX Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| UX-1 | Live mode ON by default when autonomous engine is warm; toggle with Ctrl+L | Must | UX Q&A #1 |
| UX-2 | 0ms debounce -- every valid keystroke fires the autonomous engine immediately | Must | UX Q&A #2 |
| UX-3 | Engine status badge in dashboard: [LIVE], [WARMING], [COMPILING], [IDLE], [FALLBACK], [ERROR], [OFF] | Must | UX Section 4 |
| UX-4 | Sub-millisecond latency display with microsecond precision (e.g., "0.42ms (420us)") | Should | UX Section 4.3 |
| UX-5 | Fallback performance line shows reason (e.g., "36ms (ORDER BY requires standard path)") | Should | UX Q&A #4 |
| UX-6 | Background data loading -- TUI launches instantly, warm-up progress shown | Must | UX Section 3.4 |
| UX-7 | Brief column-header flash (1 frame) on result update in live mode | Nice | UX Q&A #3 |
| UX-8 | SQL validity checking on keystroke (SqlValidity enum: Empty, Incomplete, Valid, ParseError) | Should | UX Section 9.2 |
| UX-9 | Autonomous compatibility check (QueryCompatibility enum: Autonomous, Fallback, Unknown) | Should | UX Section 9.2 |
| UX-10 | Rolling autonomous stats (avg latency, p99, consecutive sub-1ms streak, total/fallback counts) | Nice | UX Section 6.3 |

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Warm query latency (1M rows, compound filter + GROUP BY) | p50 wall-clock | < 1ms |
| NFR-2 | Warm query latency (1M rows, compound filter + GROUP BY) | p99 wall-clock | < 2ms |
| NFR-3 | Cold-to-warm transition (1M rows CSV -> binary columnar) | Wall-clock | < 500ms |
| NFR-4 | GPU memory overhead per 1M rows | Bytes | < 100MB |
| NFR-5 | CPU utilization during persistent kernel idle | % | < 2% |
| NFR-6 | CPU utilization during query execution | % | < 1% |
| NFR-7 | GPU polling power overhead (idle) | Watts delta | < 1W vs baseline |
| NFR-8 | Kernel stability (continuous operation) | Hours | > 8 hours without restart |
| NFR-9 | Parameter handoff latency | Measured | < 0.01ms |
| NFR-10 | Result availability latency (after GPU writes) | Measured | < 0.05ms |

## Out of Scope

- ORDER BY in autonomous path (requires sort kernel; use current CPU-side sort via fallback)
- JOINs (not supported in current engine; orthogonal feature)
- String filter predicates (LIKE, regex) -- Phase 2
- GROUP BY on string columns directly -- Phase 2
- GROUP BY cardinality > 64 -- Phase 2
- Multiple concurrent queries (single-query work queue for MVP)
- Binary columnar cache persistence to disk -- Phase 2
- Metal 4 work graphs integration -- hardware-dependent future optimization
- Streaming/incremental queries -- Phase 3
- Chaos monkey / fault injection testing -- limited deliberate injection only
- Welcome overlay or onboarding UI

## Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| Existing gpu-query engine (filter, aggregate, compact shaders) | Internal | Complete -- provides baseline and fallback path |
| objc2-metal Rust bindings v0.3 | External crate | Stable |
| Metal compute shader compiler (macOS 14+) | System | Required |
| Apple Silicon UMA (M1+) | Hardware | Required |
| Existing `ColumnarBatch` storage format | Internal | Complete -- extend for binary columnar |
| criterion 0.5+ | Dev dependency | Benchmarking |
| loom 0.7+ | Optional dev dependency (feature flag) | Concurrency testing |
