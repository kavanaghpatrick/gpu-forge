# OVERVIEW: metal-forge-compute Experiment Suite

## Executive Summary

A suite of 16 GPU compute benchmark experiments on Apple Silicon M4 Pro (273 GB/s, 20 GPU cores) to empirically determine where Metal GPU compute delivers real speedups over CPU. Results drive a go/no-go decision on investing in a Rust GPU primitives library and/or GPU-accelerated consumer product.

## Why Experiments

We proved GPU inference works (462 tok/s, beating llama.cpp 389). But inference is bandwidth-bound with predictable roofline. General compute (sort, join, group-by) has irregular memory access and CPU cache advantages. We need data before committing 6+ months.

## Decision Framework

| Phase 1 Result | Phase 2 Result | Decision |
|----------------|----------------|----------|
| Primitives all >5x at 10M | Relational >3x | Ship library + build product |
| Primitives all >5x at 10M | Relational <2x | Ship primitives library only |
| Primitives >3x but mixed | N/A | Ship selective primitives |
| All primitives <2x at 10M | N/A | Kill library thesis |

## Agent Summaries

### PM (Requirements)
- 5 user stories: Primitives validation, LinAlg roofline, Relational ops, Consumer workloads, E2E pipeline
- 17 functional requirements with priorities (P0/P1/P2)
- 7 non-functional requirements (CV<5%, dispatch overhead included, fair baselines, reproducibility)
- 3-phase sequencing: P0 foundation primitives → P1 query ops → P2 consumer exploration
- Kill criteria per experiment with explicit thresholds

### UX (Developer Experience)
- `forge-bench` CLI binary with clap derive (matches gpu-query patterns)
- Preset profiles: quick (1M/3 runs/~30s), standard (10M/10 runs/~5min), thorough (100M/30 runs/~20min)
- Structured JSON output schema with hardware info, per-iteration data, statistical summaries, verdicts
- TOML configuration with per-experiment overrides
- `Experiment` trait with 9 methods for extensibility
- ASCII roofline diagrams + Python plotting scripts
- Automatic go/no-go verdict display mapped to PM decision matrix

### Tech (Architecture)
- Workspace: `forge-bench` (binary) + `forge-primitives` (library seed)
- ~65 new files across experiments, shaders, infrastructure, tests
- Critical algorithm choices:
  - Scan: Reduce-then-scan 3-pass (decoupled lookback DEADLOCKS on Apple Silicon)
  - Sort: Reduce-then-scan radix (OneSweep DEADLOCKS)
  - Group-by: Sort-based (not hash — sequential access favors GPU)
  - Histogram: Shared-memory (not global atomics)
- Single command buffer per experiment (lesson from inference: 91→1 cmdbuf)
- Page-aligned buffers for zero-copy (makeBuffer bytesNoCopy)
- PSO cache pattern reused from gpu-query
- Reuses existing shaders: filter.metal, compact.metal, aggregate.metal, matvec kernels

### QA (Quality)
- ~145 tests: ~80 unit + ~50 integration + ~10 reproducibility + ~5 E2E
- Per-kernel correctness: GPU vs CPU reference with epsilon bounds
- Statistical validation: CV<5%, outlier detection, thermal throttling detection
- All 17 FRs + 32 ACs + 7 NFRs mapped to specific test cases
- Kill criteria automated: each experiment auto-evaluates against PM thresholds
- CI: GitHub Actions workflow with artifact storage and regression detection

## Module Roadmap

| Phase | Experiments | Duration | Gate |
|-------|------------|----------|------|
| 1. Foundation | reduce, scan, compact, sort, harness | Week 1-2 | All >2x at 10M |
| 2. Query Ops | filter, group-by, histogram, GEMM, pipeline, DuckDB | Week 2-3 | Relational >1.5x |
| 3. Consumer | spreadsheet, timeseries, JSON/CSV, GEMV, hash join | Week 3-4 | Phase 1-2 passed |
