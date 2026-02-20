---
id: bench.BREAKDOWN
module: bench
priority: 5
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN, gpu-engine.BREAKDOWN, search-core.BREAKDOWN]
tags: [gpu-search]
testRequirements:
  unit:
    required: false
---

# Bench Module Breakdown

## Context

Criterion benchmarks validating gpu-search performance targets: 55-80 GB/s content search throughput, <5ms search-as-you-type latency (cached files), and comparison against ripgrep as the baseline competitor. The 10% regression threshold (Q&A decision #14) is enforced as a CI gate.

These benchmarks serve dual purpose: (1) validate that the objc2-metal port preserves rust-experiment's proven performance, and (2) provide ongoing regression detection as the codebase evolves.

## Tasks

### T-060: Implement search throughput benchmark

Criterion benchmark measuring raw GPU content search throughput in GB/s:

```rust
fn bench_search_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_throughput");
    group.sample_size(20);

    for &size_mb in &[1, 10, 100] {
        let data = generate_test_content(size_mb * 1_048_576);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("literal", size_mb),
            &data,
            |b, data| { b.iter(|| gpu_search(data, "pattern")); },
        );
    }
    group.finish();
}
```

Sizes: 1MB, 10MB, 100MB, 1GB synthetic content.
Target: 55-80 GB/s (matching rust-experiment proven benchmarks).

**Target**: `gpu-search/benches/search_throughput.rs`
**Verify**: `cargo bench -p gpu-search -- search_throughput` -- reports GB/s within expected range

### T-061: Implement I/O latency benchmark

Benchmark MTLIOCommandQueue batch file loading:

| Scenario | File Count | Target |
|----------|-----------|--------|
| Small batch | 100 files | <5ms |
| Medium batch | 1,000 files | <15ms |
| Large batch | 10,000 files | <30ms |

Compare against sequential CPU `read()` calls.

**Target**: `gpu-search/benches/io_latency.rs`
**Verify**: `cargo bench -p gpu-search -- io_latency` -- MTLIOCommandQueue consistently faster than CPU reads

### T-062: Implement end-to-end search latency benchmark

Measure full pipeline: SearchRequest -> results visible.

| Scenario | Data | Target |
|----------|------|--------|
| Cached incremental search | Already-indexed files, pattern change | <5ms |
| Cold first search | 10K files, no cache | <200ms |
| Index load | 100K entries from cache | <10ms |

**Target**: `gpu-search/benches/search_latency.rs` (new benchmark file)
**Verify**: `cargo bench -p gpu-search -- search_latency` -- cached search <5ms p50

### T-063: Implement ripgrep comparison benchmark

Side-by-side comparison against ripgrep on same corpus:

```rust
fn bench_vs_ripgrep(c: &mut Criterion) {
    let corpus = Path::new("test_data/medium_project"); // ~100MB
    let pattern = "struct";

    let mut group = c.benchmark_group("vs_ripgrep");

    group.bench_function("gpu-search", |b| {
        b.iter(|| gpu_search_directory(corpus, pattern));
    });

    group.bench_function("ripgrep", |b| {
        b.iter(|| {
            Command::new("rg")
                .args(&["--count", pattern])
                .current_dir(corpus)
                .output()
        });
    });

    group.finish();
}
```

Target: gpu-search 4-7x faster than ripgrep (matching rust-experiment benchmarks).

**Target**: `gpu-search/benches/search_throughput.rs` (additional benchmark group)
**Verify**: `cargo bench -p gpu-search -- vs_ripgrep` -- gpu-search consistently faster

### T-064: Implement index load benchmark

Benchmark filesystem index operations:

| Operation | Target |
|-----------|--------|
| Full scan (10K files) | <2s |
| Full scan (100K files) | <10s |
| Cache load (mmap) | <10ms |
| GPU buffer creation | <5ms |

**Target**: `gpu-search/benches/index_load.rs`
**Verify**: `cargo bench -p gpu-search -- index_load` -- cache load <10ms

### T-065: Implement regression detection script

Python or shell script for CI:
- Parse Criterion JSON output (`--output-format json`)
- Compare against saved baseline
- Fail if any benchmark regresses >10% (Q&A decision #14)
- Generate human-readable summary of changes

```bash
cargo bench -p gpu-search -- --output-format json > bench.json
python3 scripts/check_bench_regression.py bench.json --threshold 10
```

**Target**: `scripts/check_bench_regression.py`
**Verify**: Script correctly detects >10% regression in mock JSON data

### T-066: Create deterministic test data generator

Generate reproducible test content for benchmarks:
- `generate_test_content(size_bytes)`: Lorem ipsum with injected patterns at known intervals
- `generate_test_directory(file_count, avg_file_size)`: Create temp directory with synthetic source files
- Deterministic (seeded RNG) for reproducible benchmarks across runs

**Target**: `gpu-search/benches/common/mod.rs` or `gpu-search/src/test_utils.rs`
**Verify**: Two runs of generator produce identical output

## Acceptance Criteria

1. Content search throughput: 55-80 GB/s on 100MB synthetic data (Criterion)
2. I/O latency: MTLIOCommandQueue loads 10K files in <30ms
3. End-to-end cached search: <5ms p50 for incremental search
4. Ripgrep comparison: gpu-search 4-7x faster on same corpus
5. Index cache load: <10ms via mmap
6. Regression detection: CI script fails on >10% regression
7. All benchmarks use deterministic test data (reproducible)
8. Sample size: 20+ per benchmark for statistical confidence
9. Benchmark reports include throughput (GB/s) where applicable
10. `cargo bench -p gpu-search` runs all benchmarks without errors

## Technical Notes

- **Warm-up**: Criterion handles warm-up automatically. For GPU benchmarks, first iteration may include shader compilation -- this is expected and filtered by Criterion's warm-up phase.
- **Thermal throttling**: Apple Silicon throttles under sustained load. Criterion's statistical analysis handles this (outlier detection). Report p50 not mean.
- **Test corpus**: Use `tempfile` for ephemeral test directories. For ripgrep comparison, use a checked-in corpus (e.g., subset of a public repo).
- **Baseline management**: Use `cargo bench -- --save-baseline main` on main branch. Compare with `--baseline main` on PRs.
- **CI integration**: Self-hosted Apple Silicon runner required. Benchmarks run as separate CI stage (Stage 3 in QA pipeline).
- Reference: PM.md Section 8, QA.md Sections 5.1-5.4, TECH.md Section 10
