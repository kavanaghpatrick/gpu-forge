# Requirements: metal-forge-compute Experiment Suite

## Goal

Validate where Apple Silicon GPU compute delivers real-world speedups over CPU baselines, producing hard numbers that determine whether to invest in (a) a Rust GPU primitives library, (b) a GPU-accelerated consumer product, or (c) both. M4 Pro target hardware: 273 GB/s bandwidth, 20 GPU cores.

## Business Case

### Market Gap

No Rust library exposes GPU primitives (sort, scan, reduce) for Apple Silicon. Every GPU analytics engine is CUDA-only. MLX is Python-first, ML-focused. wgpu doesn't expose Metal 4 features. This is a greenfield opportunity -- but only if the speedups justify the engineering cost.

### Why Experiments First

We already achieved 462 tok/s GPU decode (beating llama.cpp 389 tok/s) on SmolLM-135M. But inference is bandwidth-bound with predictable roofline behavior. General compute workloads (sort, join, group-by) have irregular memory access, CPU cache advantages, and lower arithmetic intensity. We need data before committing 6+ months of library development.

### Decision Stakes

- **Library path**: 6-12 months, serving other Rust developers. Needs 3-10x speedups on primitives.
- **Product path**: 3-6 months, shipping a consumer app (spreadsheet, analytics tool). Needs end-to-end pipeline wins, not just kernel-level.
- **Neither**: If GPU only wins on embarrassingly parallel workloads with >10M elements, the addressable market is too narrow.

## User Stories

### US-1: GPU Primitives Validation (Library Decision)

**As a** library architect evaluating the metal-forge-compute investment
**I want to** measure GPU vs CPU throughput for sort, scan, reduce, histogram, and stream compaction at multiple data sizes
**So that** I know which primitives deliver >3x speedup and at what minimum data size the GPU crossover occurs.

**Acceptance Criteria:**
- [ ] AC-1.1: Radix sort measured for 1M, 10M, 100M u32/f32 elements; elements/sec and speedup vs `std::sort` reported
- [ ] AC-1.2: Prefix scan measured for 1M-100M elements; GB/s achieved vs theoretical 273 GB/s ceiling reported
- [ ] AC-1.3: Parallel reduce (sum/min/max) measured for 1M-100M elements; GB/s achieved reported
- [ ] AC-1.4: Histogram measured for 256-bin and 65536-bin at 1M-100M elements; speedup vs CPU reported
- [ ] AC-1.5: Stream compaction measured at 10%, 50%, 90% selectivity rates for 1M-100M elements
- [ ] AC-1.6: Each experiment reports GPU crossover point (minimum N where GPU > CPU)
- [ ] AC-1.7: All measurements include GPU dispatch overhead (not just kernel time)

### US-2: Linear Algebra Validation (GEMM/GEMV Roofline)

**As a** performance engineer
**I want to** benchmark simdgroup_matrix GEMM and bandwidth-bound GEMV against Accelerate framework
**So that** I understand where Metal 4 matrix hardware matches or exceeds Apple's own optimized CPU/ANE paths.

**Acceptance Criteria:**
- [ ] AC-2.1: GEMM benchmarked at 256x256, 1024x1024, 4096x4096 for FP16 and FP32
- [ ] AC-2.2: GEMV benchmarked for shapes matching inference workloads (768x768, 2048x768, etc.)
- [ ] AC-2.3: Accelerate vDSP/BLAS used as CPU baseline (not naive loops)
- [ ] AC-2.4: GFLOPS and % of theoretical peak reported for each size
- [ ] AC-2.5: Existing matvec kernels from gpu-inference-pipeline reused where applicable

### US-3: Relational Operations Validation (Database Decision)

**As a** product strategist evaluating GPU analytics
**I want to** measure GPU filter, group-by aggregate, and hash join against CPU implementations
**So that** I determine if a GPU query engine on Apple Silicon can compete with DuckDB.

**Acceptance Criteria:**
- [ ] AC-3.1: Filter measured at 1%, 10%, 50%, 90% selectivity for 1M-100M rows on columnar int64/f64 data
- [ ] AC-3.2: Group-by aggregate measured with cardinality 10, 1K, 100K, 1M groups on 1M-100M rows
- [ ] AC-3.3: Hash join measured for varying table sizes (1M x 1M, 10M x 1M, 10M x 10M) and join selectivity
- [ ] AC-3.4: Existing `filter.metal`, `compact.metal`, `aggregate.metal` kernels from gpu-query reused as starting point
- [ ] AC-3.5: All measurements include data transfer time (if any) and buffer allocation

### US-4: Consumer Workload Validation (Product Decision)

**As a** product manager evaluating consumer app viability
**I want to** measure GPU speedup on spreadsheet formulas, time-series analytics, and data parsing
**So that** I know if "GPU-accelerated spreadsheet" or "GPU trading analytics" is a viable product pitch.

**Acceptance Criteria:**
- [ ] AC-4.1: Spreadsheet batch formula eval (SUM, AVERAGE, VLOOKUP-equivalent) measured on 1M+ cells
- [ ] AC-4.2: Time series ops (moving average, VWAP, Bollinger bands) measured on 1M-100M tick records
- [ ] AC-4.3: JSON and CSV parsing throughput measured vs serde_json and csv crate
- [ ] AC-4.4: Each reports "user-perceptible" latency (wall clock ms) not just throughput

### US-5: End-to-End Pipeline (Competitive Validation)

**As a** decision maker
**I want to** run a complete analytical query (read -> filter -> group-by -> aggregate -> sort -> top-K) on GPU vs CPU vs DuckDB
**So that** I have a single number ("our GPU pipeline is Nx faster/slower than DuckDB on M4 Pro") for the go/no-go decision.

**Acceptance Criteria:**
- [ ] AC-5.1: Full pipeline timed end-to-end on 10M and 100M row datasets
- [ ] AC-5.2: Rust CPU baseline uses idiomatic iterators + hashmap (not naive)
- [ ] AC-5.3: DuckDB comparison uses same query, same data, same hardware
- [ ] AC-5.4: Reports breakdown by stage (% time in filter, group-by, sort, etc.)
- [ ] AC-5.5: Reports total wall-clock time including data loading

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | GPU radix sort kernel (4-bit, 16 passes) for u32/f32 | P0 | 1M/10M/100M elements benchmarked, elements/sec reported |
| FR-2 | Decoupled lookback prefix scan kernel | P0 | GB/s at 1M-100M elements, comparison to theoretical bandwidth |
| FR-3 | Hierarchical parallel reduce (simd_sum + threadgroup + global) | P0 | Sum/min/max at 1M-100M, GB/s reported |
| FR-4 | Shared-memory histogram kernel (256-bin, 65536-bin) | P1 | Speedup vs CPU at multiple data sizes |
| FR-5 | Stream compaction kernel (prefix scan + scatter) | P0 | 10%/50%/90% selectivity at 1M-100M elements |
| FR-6 | simdgroup_matrix 8x8 tiled GEMM kernel (FP16, FP32) | P1 | GFLOPS at 256/1024/4096 square, vs Accelerate |
| FR-7 | GEMV kernel (reuse existing matvec) | P2 | GB/s vs Accelerate for inference-relevant shapes |
| FR-8 | GPU columnar filter (extend existing filter.metal) | P0 | Selectivity sweep on 1M-100M rows |
| FR-9 | GPU sort-based group-by aggregate | P1 | Cardinality sweep on 1M-100M rows |
| FR-10 | GPU hash join kernel | P2 | Various table sizes and selectivities |
| FR-11 | GPU batch spreadsheet formula evaluation | P2 | SUM/AVG/VLOOKUP on 1M+ cells |
| FR-12 | GPU time-series analytics (MA, VWAP, Bollinger) | P2 | 1M-100M tick records |
| FR-13 | GPU JSON/CSV parsing | P2 | Throughput vs serde_json/csv |
| FR-14 | End-to-end GPU analytical pipeline | P1 | Full query on 10M/100M rows, wall-clock vs CPU |
| FR-15 | DuckDB comparison benchmark | P1 | Same queries, same data, same hardware |
| FR-16 | Unified benchmark harness with JSON output | P0 | All experiments produce machine-readable results |
| FR-17 | Crossover-point analysis for each primitive | P0 | Report minimum N where GPU beats CPU |

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Measurement accuracy | Coefficient of variation | < 5% across 10 runs per data point |
| NFR-2 | Include dispatch overhead | Wall-clock time | Measure total time including buffer alloc, encode, commit, waitUntilCompleted |
| NFR-3 | Fair CPU baseline | Implementation quality | Use std::sort, rayon parallel iterators, Accelerate BLAS -- not naive loops |
| NFR-4 | Memory reporting | Peak GPU memory | Report MTLDevice.currentAllocatedSize for each experiment |
| NFR-5 | Reproducibility | Determinism | Fixed random seeds, documented data generation |
| NFR-6 | Hardware reporting | System info | Log M4 Pro core count, memory bandwidth, Metal feature set |
| NFR-7 | Warm-up protocol | Consistency | 3 warm-up iterations discarded before 10 measured iterations |

## Experiment Priority & Sequencing

### Phase 1: Foundation Primitives (P0) -- Week 1-2

**Rationale**: Sort, scan, reduce, compact are building blocks for everything else. If these don't achieve >3x over CPU at 10M+ elements, the library thesis fails early.

| # | Experiment | Signal | Kill Criterion |
|---|-----------|--------|----------------|
| 1 | Parallel Reduce | Roofline baseline -- simplest kernel, should hit near-bandwidth | < 50% bandwidth utilization |
| 2 | Prefix Scan | Required by sort and compact | < 2x vs CPU sequential scan |
| 3 | Stream Compaction | Required by filter pipeline | < 2x vs CPU filter+collect |
| 4 | Radix Sort | Highest complexity, highest value | < 2x vs std::sort at 10M |
| 5 | Benchmark harness | Infrastructure | N/A |

**Why this order**: Reduce validates our measurement methodology and Metal dispatch overhead. Scan and compaction are dependencies for sort. Sort is the capstone that validates the full primitive stack.

### Phase 2: Query Operations (P1) -- Week 2-3

**Rationale**: Relational ops validate the product direction. Even if primitives are fast, irregular memory access in joins and group-by may negate GPU advantage.

| # | Experiment | Signal | Kill Criterion |
|---|-----------|--------|----------------|
| 6 | Filter (extend existing) | Leverages proven filter.metal | < 1.5x vs CPU at 10M rows |
| 7 | Group-By Aggregate | Tests sort+reduce composition | < 1.5x vs CPU hashmap at 10M |
| 8 | Histogram | Validates shared memory atomics | < 2x vs CPU at 10M |
| 9 | GEMM (simdgroup_matrix) | Validates Metal 4 matrix hardware | < 50% of Accelerate at 1024x1024 |
| 10 | End-to-end pipeline vs CPU | Composition test | GPU pipeline slower than CPU pipeline |
| 11 | DuckDB comparison | Market positioning | > 2x slower than DuckDB on same query |

### Phase 3: Product Exploration (P2) -- Week 3-4

**Rationale**: Only worth pursuing if Phase 1-2 show consistent 3x+ wins. These experiments validate specific product pitches.

| # | Experiment | Signal | Kill Criterion |
|---|-----------|--------|----------------|
| 12 | Spreadsheet formulas | Consumer product signal | No perceptible speedup on 1M cells |
| 13 | Time series analytics | Finance product signal | < 2x vs optimized CPU |
| 14 | JSON/CSV parsing | Data ingestion signal | < 2x vs serde_json/csv |
| 15 | GEMV (reuse existing) | Inference library signal | < 80% of Accelerate bandwidth |
| 16 | Hash join | Complex relational signal | < 1x vs CPU (likely outcome) |

## Success Criteria & Decision Framework

### Speedup Thresholds

| Speedup | Interpretation | Action |
|---------|---------------|--------|
| < 1x (GPU slower) | GPU overhead dominates at this scale | Document crossover point, skip for small N |
| 1-2x | Marginal -- not worth GPU complexity | CPU-only unless latency-hiding matters |
| 2-5x | Solid win -- justifies GPU path for large N | Include in library, document minimum N |
| 5-10x | Strong win -- core library primitive | Prioritize, optimize further |
| > 10x | Dominant -- build product around this | Centerpiece capability |

### Decision Matrix

| Phase 1 Result | Phase 2 Result | Decision |
|----------------|----------------|----------|
| Sort/scan/reduce all >5x at 10M | Filter/group-by >3x at 10M | **Ship library + build product** |
| Sort/scan/reduce all >5x at 10M | Relational ops <2x | **Ship primitives library only** (Layer 1) |
| Sort >3x but scan/reduce <2x | N/A | **Ship sort only**, not full library |
| All primitives <2x at 10M | N/A | **Kill library thesis**. Pivot to ML/inference only |
| Primitives >3x | Pipeline < DuckDB by >5x | **Library yes, product no** (DuckDB too strong) |
| Primitives >3x | Pipeline within 2x of DuckDB | **Library + product** (competitive positioning) |

### Key Numbers That Matter

| Metric | Target | Validates |
|--------|--------|-----------|
| Radix sort elements/sec | > 1B elem/s (stretch: 3B) | Library Layer 1 thesis |
| Reduce bandwidth utilization | > 70% of 273 GB/s | Metal dispatch overhead is manageable |
| Filter throughput at 10% selectivity | > 5x CPU on 10M rows | GPU query engine viability |
| Group-by 1K groups on 10M rows | > 3x CPU hashmap | Analytics product viability |
| End-to-end pipeline vs DuckDB | Within 2x | Competitive product positioning |
| GPU crossover point (typical) | < 100K elements | GPU useful for interactive workloads |

## Risk Assessment

### R1: GPU Dispatch Overhead Kills Small-N Performance

**Likelihood**: High. Metal command buffer encode + commit + wait adds 10-50us minimum.
**Impact**: GPU crossover may not occur until 100K-1M elements, limiting consumer app use cases.
**Mitigation**: Measure dispatch overhead explicitly. Use indirect command buffers and persistent command queues. Batch multiple operations per command buffer (proven in inference pipeline: 91 -> 1 command buffer).

### R2: Irregular Memory Access Negates GPU Bandwidth

**Likelihood**: Medium. Hash joins and hash-based group-by have random scatter/gather patterns. CPU L1/L2 cache (192KB/16MB on M4 Pro) may outperform GPU SLC (48MB shared) for small working sets.
**Impact**: Relational ops may show < 1x speedup, killing the database product path.
**Mitigation**: Sort-based alternatives (sort then scan-based group-by) use sequential access. Measure both hash-based and sort-based approaches.

### R3: Accelerate BLAS Already Optimal for GEMM

**Likelihood**: High. Apple's Accelerate uses AMX (matrix coprocessor) for large GEMM, which may exceed GPU Metal throughput.
**Impact**: GEMM experiment may show GPU is slower than AMX, but this is expected -- AMX exists specifically for this.
**Mitigation**: Frame GEMM as "what can Metal 4 simdgroup_matrix achieve" not "can we beat Apple's own silicon". The win is on GPU-resident data that avoids CPU transfer.

### R4: DuckDB Comparison Is Unflattering

**Likelihood**: Medium-High. DuckDB is extremely optimized, vectorized, with 10+ years of query optimization.
**Impact**: "5x slower than DuckDB" is a bad headline for investors/users.
**Mitigation**: (1) Frame as "first GPU implementation vs mature CPU engine" -- trajectory matters. (2) Measure at 100M+ rows where bandwidth advantage grows. (3) DuckDB is single-threaded per query in many workloads; GPU parallelism has different scaling.

### R5: JSON/CSV GPU Parsing Has Branchy Control Flow

**Likelihood**: High. Parsing is control-flow heavy (state machines, escaping, quoting). GPU SIMT model penalizes divergent branches.
**Impact**: GPU parsing likely < 1x for complex formats (nested JSON). May work for simple CSV.
**Mitigation**: Test both simple CSV (no quoting) and complex JSON separately. Consider GPU for validation/filtering only, not full parsing.

## Glossary

- **Crossover point**: Minimum element count N where GPU throughput exceeds CPU
- **Roofline**: Theoretical maximum throughput given bandwidth (273 GB/s) and compute (M4 Pro 20 cores)
- **SLC**: System-Level Cache. 48MB on M4 Pro, shared between CPU and GPU. Enables cache reuse across dispatches
- **simdgroup_matrix**: Metal 4 API for 8x8 matrix multiply in hardware. Analogous to CUDA tensor cores
- **Dispatch overhead**: Time to encode, commit, and synchronize a Metal command buffer (typically 10-50us)
- **Bandwidth utilization**: Actual GB/s achieved divided by theoretical peak (273 GB/s). >70% is excellent
- **Selectivity**: Fraction of rows passing a filter predicate. Low selectivity (1%) = few matches; high (90%) = most pass
- **Cardinality**: Number of distinct group keys in group-by operations
- **Stream compaction**: Removing unselected elements to produce a dense output array
- **Decoupled lookback scan**: GPU prefix scan algorithm that avoids global barriers by having each threadgroup look back at predecessor status flags

## Out of Scope

- Multi-GPU dispatch (single M4 Pro GPU only)
- Network/distributed operations
- Comparison with CUDA implementations (no NVIDIA hardware available)
- ANE (Apple Neural Engine) benchmarking
- iOS/iPadOS targets (macOS only)
- Kernel auto-tuning (fixed thread counts per experiment)
- Production-quality error handling (benchmark code only)
- FP64 operations (Metal has no native FP64)
- Graph algorithms (BFS, PageRank -- too specialized for initial validation)
- Sparse matrix operations (future experiment if dense results are positive)

## Dependencies

| Dependency | Status | Impact |
|------------|--------|--------|
| objc2-metal Rust bindings | Available (used in inference pipeline) | Required for all experiments |
| Metal 4 SDK (Xcode 16+) | Installed | Required for simdgroup_matrix |
| Accelerate framework | System default | Required for BLAS baselines |
| DuckDB CLI or C API | Installable via Homebrew | Required for Experiment #15 |
| gpu-query shaders (filter, compact, aggregate) | Exist in repo | Reuse for relational experiments |
| gpu-inference matvec kernels | Exist in repo | Reuse for GEMV experiment |
| Criterion benchmark crate | In Cargo.toml | Timing infrastructure |
| rayon crate | Standard | Multi-threaded CPU baselines |

## Unresolved Questions

1. **Data generation strategy**: Synthetic uniform random, or realistic distributions (Zipf for cardinality, skewed for selectivity)? Realistic distributions change GPU behavior significantly.
2. **Cold vs warm measurement**: Should we report cold (first dispatch, empty SLC) or warm (SLC populated)? Both matter for different use cases.
3. **Accelerate BLAS vs manual CPU SIMD**: For GEMM baseline, is Accelerate-only sufficient, or should we also test manual NEON SIMD to isolate AMX contribution?
4. **100M element memory**: 100M f32 = 400MB. With input + output + scratch buffers, some experiments may approach M4 Pro's working set limits. Cap at 100M or push to 1B?
5. **DuckDB version pinning**: DuckDB performance varies significantly between versions. Pin to latest stable or match a specific benchmark from their published results?

## Next Steps

1. Approve this requirements document
2. Design benchmark harness architecture (output format, warm-up protocol, statistical analysis)
3. Implement Phase 1 experiments (reduce, scan, compact, sort) in priority order
4. Run Phase 1, analyze results, make go/no-go decision on Phase 2
5. If Phase 1 passes: implement and run Phase 2 relational experiments
6. Final decision memo: library vs product vs both vs neither
