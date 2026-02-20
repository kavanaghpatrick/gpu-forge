---
spec: gpu-query
phase: research
created: 2026-02-10
generated: auto
---

# Research: gpu-query

## Executive Summary

gpu-query is a GPU-native local data analytics engine for Apple Silicon that delivers BigQuery-class query performance on local files with zero data copying. Technical feasibility is HIGH -- the core zero-copy pattern (mmap + `makeBuffer(bytesNoCopy:)`) is validated by MLX and llama.cpp, GPU database research (Crystal+, GpJSON, Sirius) confirms 10-100x over CPU engines, and the particle-system codebase proves the Rust+Metal stack at scale. No competitor exists for Apple Silicon GPU analytics -- the entire CUDA GPU database ecosystem (cuDF, HeavyDB, BlazingSQL) cannot run on Macs.

## Competitive Landscape

### Direct Competitors

| Engine | Execution | Apple Silicon | Zero-Copy | GPU Accel | SQL | Scale |
|--------|-----------|---------------|-----------|-----------|-----|-------|
| **DuckDB** | CPU (NEON SIMD) | Yes | mmap (CPU) | None | Full | Sub-second TPC-H SF100 |
| **ClickHouse Local** | CPU | Yes | Copy-on-import | None | Full | p95 ~11ms server |
| **RAPIDS cuDF** | GPU (CUDA) | **No** | GPUDirect | CUDA-only | DataFrame | 150x over pandas |
| **DataFusion** | CPU (Arrow) | Yes | mmap (CPU) | None | Full | Comparable to DuckDB |
| **gpu-query** | GPU (Metal) | **Native** | mmap+Metal | Metal | SQL subset | Target: 10-100x over DuckDB |

### Key Differentiator: UMA Zero-Copy

Apple Silicon UMA eliminates the PCIe transfer bottleneck that plagues discrete GPU databases. `makeBuffer(bytesNoCopy:)` wraps CPU memory as a Metal buffer with 0.0004ms overhead vs CUDA's 585.84ms for ~1GB [KB #248]. This is a 1,464,600x advantage for mixed CPU-GPU data access patterns.

### Academic Validation

- **Crystal+**: 1.97x over HeavyDB, 17.66x over TQP on TPC-H using tile-based execution [KB #470]
- **GpJSON**: 2.9x over parallel CPU JSON parsers, 6-8x over NVIDIA RAPIDS on A100 [KB #472]
- **Sirius** (DuckDB+cuDF hybrid): 7.2x cost-efficiency over CPU on ClickBench [KB #470]
- **GPU Sort**: ~3B elements/sec on M1 Max, 50-103x speedup [KB #321, #388]
- **GPUfs**: 7x faster file search than 8-core CPU using GPU filesystem access [KB #339]

### Market Opportunity

- GPU database market: $1.2-1.4B in 2025, 12% CAGR to $3.7B by 2035
- Apple Silicon installed base: 100M+ Macs with underutilized GPUs
- DuckDB trajectory: 25M monthly PyPI downloads, 1.4% to 3.3% Stack Overflow adoption in one year
- Zero Metal-compute analytics tools exist -- first mover wins Apple ecosystem

## Technical Feasibility

### Core Architecture Validation

| Component | KB Evidence | Feasibility |
|-----------|-------------|-------------|
| **Zero-copy I/O** | mmap + `makeBuffer(bytesNoCopy:)` [KB #224]. 16KB page alignment required [KB #89, #223] | Validated (MLX, llama.cpp) |
| **GPU memory bandwidth** | M4: 100 GB/s STREAM [KB #477]. M4 Max: 546 GB/s | Sufficient for 10GB in ~100ms |
| **GPU JSON parsing** | GpJSON: 2.9x over CPU [KB #472]. cuJSON outperforms simdjson [KB #330] | Validated (academic) |
| **GPU string matching** | HybridSA: 40x over CPU BNDM [KB #347] | Validated (academic) |
| **GPU radix sort** | ~3B elements/sec on M1 Max [KB #388]. 50-103x over CPU [KB #321] | Validated (benchmarked) |
| **Columnar GPU processing** | Crystal+, HeavyDB, BlazingSQL all columnar + tile-based [KB #311] | Validated (production) |
| **Function constant specialization** | 84% instruction reduction [KB #202, #210] | Validated (MLX, llama.cpp) |
| **Hierarchical reduction** | SIMD -> threadgroup -> global [KB #188, #328] | Validated (particle-system) |
| **Indirect dispatch** | Eliminates CPU-GPU sync [KB #277] | Validated (particle-system) |
| **No persistent kernels** | Metal watchdog kills long shaders [KB #440, #441] | Constraint -- use batched dispatches |

### SSD Bandwidth Reality Check

| Chip | SSD Sequential Read | GPU Memory Bandwidth | Ratio |
|------|-------------------|---------------------|-------|
| M4 base | ~3.2 GB/s | 100 GB/s | 31x |
| M4 Pro | ~6.7 GB/s | 150 GB/s | 22x |
| M4 Max | ~7.5 GB/s | 400 GB/s | 53x |

Cold queries hit SSD bottleneck. Warm queries (page cache resident) run at GPU memory bandwidth. Architecture uses `madvise(MADV_WILLNEED)` for pre-warming [KB #226] and column pruning to minimize I/O.

### Roofline Analysis [KB #476, #477]

Ridge point for M4: ~24 FLOPS/byte. All analytics kernels (scan, filter, aggregate, sort) are well below this -- **memory-bandwidth-bound**. GPU advantage: massive bandwidth (100-546 GB/s) vs CPU (~60 GB/s), giving 1.7-9x from bandwidth alone plus parallelism gains.

### Existing Codebase Leverage

Reusable from `particle-system/`:

| Component | Reuse Level |
|-----------|-------------|
| objc2-metal Rust bindings | Direct -- same API layer |
| build.rs (AOT shader compilation) | Direct -- same pattern |
| SoA buffer allocation pattern | Direct -- columnar buffers are SoA |
| Indirect dispatch (DispatchArgs) | Direct -- variable-size query results |
| GPU-side atomic counters | Direct -- row counting, aggregation |
| Command buffer encoding helpers | Direct -- same dispatch pattern |
| `#[repr(C)]` struct layout tests | Direct -- same validation pattern |

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | **High** | Core patterns validated by MLX, llama.cpp, and particle-system |
| Effort Estimate | **XL** | 8-12 week MVP: 8 GPU kernels, SQL planner, TUI dashboard |
| Risk Level | **Medium** | mmap+Metal undocumented (HIGH), GPU parser complexity (HIGH), watchdog (MEDIUM) |

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| mmap+Metal buffer not officially supported | HIGH | Fallback copy path (~1ms/GB); MLX + llama.cpp depend on same pattern [KB #224] |
| GPU CSV/JSON parsing complexity | HIGH | Simple CSV + NDJSON first; CPU fallback for complex cases; GpJSON algorithm [KB #472] |
| Metal GPU watchdog kills long queries | MEDIUM | Batched execution (1GB chunks) with pre-enqueue ordering [KB #441, #152] |
| SSD bandwidth on cold queries | MEDIUM | madvise pre-warming [KB #226]; Parquet column pruning; transparent warm/cold UX |
| PSO compilation latency | MEDIUM | Pre-compile common variants at startup; async creation; PSO cache [KB #159] |
| String operations on GPU | MEDIUM | Dictionary encoding reduces to integer ops; CPU fallback for complex patterns |
| Single-platform lock-in | LOW | Apple Silicon is fastest-growing compute platform; Metal-first is competitive moat |

## Recommendations

1. **Start with Parquet + CSV**: Parquet is columnar-native (trivial GPU loading), CSV is ubiquitous. JSON adds 3-4 weeks but user confirmed all three for MVP.
2. **Reuse particle-system patterns**: build.rs, buffer allocation, indirect dispatch, struct layout tests transfer directly.
3. **Function constants over JIT**: Use MTLFunctionConstantValues for query specialization -- proven pattern, avoids runtime compilation [KB #210].
4. **Metal 3 baseline**: Maximum M1+ compatibility. Metal 4 unified encoder as Phase 2 opt-in [KB #442].
5. **sqlparser-rs for SQL parsing**: Battle-tested, used by DataFusion. Custom minimal planner for MVP; DataFusion integration Phase 2.
6. **Tiered performance targets**: 100M rows M4 base, 1B M4 Pro, 10B M4 Max. Honest about warm vs cold.

## GPU Forge KB Findings Referenced

KB #22, #57, #73, #89, #102, #134, #137, #145, #151, #152, #159, #185, #188, #193, #202, #210, #217, #223, #224, #226, #248, #250, #258, #262, #264, #266, #269, #276, #277, #278, #282, #283, #311, #315, #321, #328, #330, #336, #339, #340, #343, #347, #388, #393, #396, #440, #441, #442, #457, #463, #464, #470, #471, #472, #476, #477, #497, #522, #567, #574, #579, #592, #595
