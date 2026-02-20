# UX/DX Design: metal-forge-compute Experiment Suite

## Overview

This document specifies the developer experience for running, configuring, visualizing, and interpreting GPU compute benchmark experiments on Apple Silicon M4 Pro. The target audience is internal GPU compute developers making library and product investment decisions based on PM.md requirements.

The experiment suite is a standalone Rust binary (`forge-bench`) with a TOML configuration layer, structured JSON output, and terminal-based result presentation. It is not a library -- it is an opinionated benchmark harness purpose-built for the 16 experiments defined in PM.md.

---

## 1. CLI Interface Design

### 1.1 Binary and Invocation

The binary is `forge-bench`, built as a `[[bin]]` target in the `metal-forge-compute` workspace. It follows the same `clap` derive pattern established in `gpu-query` (see `gpu-query/src/cli/args.rs`).

```
forge-bench [OPTIONS] [EXPERIMENT...]
```

When invoked with no arguments, it runs all Phase 1 experiments at the "standard" profile. When invoked with experiment names, it runs only those experiments.

### 1.2 Command-Line Arguments

```
USAGE:
    forge-bench [OPTIONS] [EXPERIMENT...]

ARGS:
    [EXPERIMENT...]    Experiments to run (default: all Phase 1)
                       Values: reduce, scan, compact, sort, histogram,
                               filter, groupby, gemm, gemv, pipeline,
                               duckdb, spreadsheet, timeseries,
                               json-parse, hash-join
                       Groups: phase1, phase2, phase3, all

OPTIONS:
    -p, --profile <PROFILE>    Preset profile [default: standard]
                               Values: quick, standard, thorough, custom
    -s, --size <SIZE>...       Element counts to test [default: from profile]
                               Examples: 1M, 10M, 100M, 1_000_000
    -r, --runs <N>             Measured iterations per data point [default: 10]
    -w, --warmup <N>           Warm-up iterations (discarded) [default: 3]
        --json                 Emit JSON to stdout (suppresses table output)
        --json-file <PATH>     Write JSON results to file
        --csv-file <PATH>      Write CSV results to file
        --append               Append to existing JSON/CSV file (for regression tracking)
        --no-cpu               Skip CPU baseline measurements
        --no-color             Disable colored terminal output
        --cold                 Cold execution: no warm-up, measure first dispatch
        --seed <N>             Random seed for data generation [default: 42]
    -v, --verbose              Print per-iteration timings and kernel details
    -q, --quiet                Minimal output: just the summary table
        --list                 List available experiments and exit
        --hardware             Print detected hardware info and exit
    -c, --config <PATH>        Path to TOML config file [default: forge-bench.toml]
    -h, --help                 Print help
    -V, --version              Print version
```

### 1.3 Example Invocations

```bash
# Run all Phase 1 experiments at standard profile
forge-bench phase1

# Quick smoke test of reduce kernel only
forge-bench reduce --profile quick

# Thorough sort benchmark with JSON output
forge-bench sort --profile thorough --json-file results/sort_thorough.json

# Custom sizes for filter experiment
forge-bench filter --size 1M 10M 50M 100M --runs 20

# Full competitive comparison
forge-bench pipeline duckdb --profile thorough --json-file results/competitive.json

# List everything available
forge-bench --list

# Check hardware detection
forge-bench --hardware
```

### 1.4 Progress Reporting

Long benchmarks (especially `--profile thorough` at 100M elements with 30 runs) can take minutes per experiment. The harness uses `indicatif` for real-time progress reporting to stderr (so stdout remains clean for `--json`).

**Progress display during execution:**

```
metal-forge-compute v0.1.0 | M4 Pro 20-core GPU | 273 GB/s | macOS 15.2

Phase 1: Foundation Primitives
==============================

[1/5] Reduce (sum, u32)
  Generating data... 100M u32 elements (400 MB)          done [0.3s]
  GPU warm-up... 3 iterations                             done [0.1s]
  GPU measured... ████████████████████████████████ 10/10   done [2.1s]
  CPU warm-up... 3 iterations                             done [0.2s]
  CPU measured... ████████████████████████░░░░░░░░  7/10   [1.4s]
```

**Key design decisions for progress:**

- Each experiment shows a deterministic step sequence: generate, warm-up, measure GPU, warm-up, measure CPU.
- The progress bar shows `completed/total` iterations, not estimated time (GPU dispatch timing is too variable for ETA).
- Data generation time is shown but excluded from benchmark measurements.
- All progress goes to stderr. Only `--json` structured output goes to stdout.
- When `--quiet` is set, progress bars are suppressed; only the final summary table prints.

### 1.5 Terminal Table Output

After each experiment completes, results appear immediately (not batched until the end). This gives developers useful data even if they Ctrl-C midway through a long suite.

**Per-experiment result table:**

```
Reduce (sum, u32)
+-----------+----------+----------+---------+---------+--------+--------+
| Elements  | GPU ms   | CPU ms   | Speedup | GPU GB/s| %Peak  | Verdict|
+-----------+----------+----------+---------+---------+--------+--------+
|     1,000 |    0.052 |    0.001 |   0.02x |    0.08 |  0.0%  |  SKIP  |
|    10,000 |    0.054 |    0.008 |   0.15x |    0.74 |  0.3%  |  SKIP  |
|   100,000 |    0.061 |    0.072 |   1.18x |    6.56 |  2.4%  |  WEAK  |
| 1,000,000 |    0.098 |    0.641 |   6.54x |   40.8  | 14.9%  |  PASS  |
|10,000,000 |    0.582 |    6.12  |  10.51x |   68.7  | 25.2%  |  PASS  |
|100,000,000|    5.41  |   62.3   |  11.51x |   73.9  | 27.1%  |  PASS  |
+-----------+----------+----------+---------+---------+--------+--------+
  Crossover: ~63,000 elements (GPU matches CPU)
  Peak bandwidth: 73.9 GB/s (27.1% of 273 GB/s theoretical)
  Kill criterion: > 50% bandwidth utilization -- INVESTIGATING (27.1% < 50%)
```

**Verdict column color coding** (when terminal supports color):

| Verdict | Color  | Meaning |
|---------|--------|---------|
| PASS    | Green  | Speedup exceeds PM.md threshold for this experiment |
| WEAK    | Yellow | GPU wins but below threshold (1-2x) |
| SKIP    | Gray   | GPU slower than CPU at this size (below crossover) |
| FAIL    | Red    | Misses kill criterion from PM.md |

### 1.6 Suite Summary Table

After all experiments complete, a compact summary prints:

```
Suite Summary: Phase 1 Foundation Primitives
+------------------+----------+-----------+---------+--------+---------+
| Experiment       | Best GPU | Crossover | Max     | %Peak  | Status  |
|                  | Speedup  | Point     | GB/s    | BW     |         |
+------------------+----------+-----------+---------+--------+---------+
| Reduce (sum)     |  11.51x  |    63K    |  73.9   | 27.1%  |  INVEST |
| Prefix Scan      |   5.23x  |   250K    |  58.2   | 21.3%  |  PASS   |
| Stream Compact   |   3.87x  |   500K    |  45.1   | 16.5%  |  PASS   |
| Radix Sort       |   7.12x  |   100K    |  31.4   | 11.5%  |  PASS   |
| Histogram (256)  |   4.55x  |   200K    |  52.3   | 19.2%  |  PASS   |
+------------------+----------+-----------+---------+--------+---------+

Decision signal: 4/5 experiments PASS kill criteria
Recommendation: PROCEED to Phase 2 (see PM.md Decision Matrix)
```

**Status column values:**

| Status       | Meaning |
|--------------|---------|
| INVEST       | > 10x speedup -- centerpiece capability, optimize further |
| PASS         | Exceeds kill criterion threshold from PM.md |
| MARGINAL     | Meets minimum but below target |
| FAIL         | Below kill criterion -- document and move on |

---

## 2. Results Visualization

### 2.1 JSON Output Schema

All results are emitted in a structured JSON format compatible with Bencher Metric Format patterns but extended for GPU-specific metrics. The schema is designed for machine consumption by CI, dashboards, and regression detection scripts.

```jsonc
{
  "version": "1.0.0",
  "suite": "metal-forge-compute",
  "timestamp": "2026-02-17T14:30:00Z",
  "hardware": {
    "chip": "Apple M4 Pro",
    "gpu_cores": 20,
    "cpu_cores_performance": 12,
    "cpu_cores_efficiency": 4,
    "memory_gb": 48,
    "bandwidth_theoretical_gbps": 273.0,
    "metal_family": "apple9",
    "metal_gpu_family": "metal3",
    "os_version": "macOS 15.2",
    "driver_version": "Metal 4.0"
  },
  "config": {
    "profile": "standard",
    "warmup_iterations": 3,
    "measured_iterations": 10,
    "seed": 42,
    "cold_start": false
  },
  "experiments": [
    {
      "name": "reduce_sum_u32",
      "category": "primitives",
      "phase": 1,
      "priority": "P0",
      "kill_criterion": "< 50% bandwidth utilization",
      "data_points": [
        {
          "element_count": 1000000,
          "element_type": "u32",
          "data_bytes": 4000000,
          "gpu": {
            "mean_ms": 0.098,
            "median_ms": 0.095,
            "std_dev_ms": 0.004,
            "min_ms": 0.091,
            "max_ms": 0.107,
            "cv_percent": 4.1,
            "throughput_elements_per_sec": 10204081632,
            "bandwidth_gbps": 40.8,
            "bandwidth_utilization_percent": 14.9,
            "peak_gpu_memory_bytes": 8388608,
            "dispatch_overhead_us": 12.3,
            "iterations": [0.095, 0.091, 0.098, 0.107, 0.093, 0.096, 0.101, 0.098, 0.095, 0.094]
          },
          "cpu": {
            "mean_ms": 0.641,
            "median_ms": 0.638,
            "std_dev_ms": 0.012,
            "min_ms": 0.625,
            "max_ms": 0.667,
            "cv_percent": 1.9,
            "throughput_elements_per_sec": 1560062402,
            "method": "rayon_parallel_sum",
            "threads": 12
          },
          "comparison": {
            "speedup": 6.54,
            "verdict": "PASS",
            "gpu_wins": true
          }
        }
        // ... additional data points for 10M, 100M
      ],
      "analysis": {
        "crossover_elements": 63000,
        "max_speedup": 11.51,
        "max_bandwidth_gbps": 73.9,
        "max_bandwidth_utilization_percent": 27.1,
        "kill_criterion_met": false,
        "kill_criterion_value": 27.1,
        "overall_verdict": "INVESTIGATING",
        "notes": "Bandwidth utilization 27.1% is below 50% target. Likely bottleneck: dispatch overhead amortization. Consider fused multi-pass reduce."
      }
    }
  ],
  "summary": {
    "total_experiments": 5,
    "passed": 4,
    "failed": 1,
    "marginal": 0,
    "recommendation": "PROCEED",
    "decision_matrix_row": "Sort/scan/reduce >5x at 10M",
    "wall_clock_total_seconds": 342.7
  }
}
```

### 2.2 CSV Output Format

For quick spreadsheet import and time-series tracking:

```csv
timestamp,experiment,element_count,element_type,gpu_mean_ms,cpu_mean_ms,speedup,gpu_gbps,bw_util_pct,verdict
2026-02-17T14:30:00Z,reduce_sum_u32,1000000,u32,0.098,0.641,6.54,40.8,14.9,PASS
2026-02-17T14:30:00Z,reduce_sum_u32,10000000,u32,0.582,6.12,10.51,68.7,25.2,PASS
```

### 2.3 Terminal Roofline Diagram

A simplified ASCII roofline model prints after bandwidth-bound experiments (reduce, scan, GEMV) to visually show where each data point lands relative to the theoretical ceiling:

```
Roofline: Reduce (sum, u32) -- M4 Pro 273 GB/s

  273 |------------------------------------------------ peak bandwidth
      |
  200 |
      |
  150 |
      |
  100 |                                          *  100M
      |                                  * 10M
   75 |
      |
   50 |                   * 1M
      |
   25 |
      |         * 100K
    0 +--+--------+--------+--------+--------+--------+---->
      1K   10K   100K    1M     10M    100M
  GB/s                  Element Count

  Bandwidth utilization: 27.1% at 100M (room for optimization)
```

This is intentionally simple -- a quick visual sanity check, not a publication-quality chart. For real roofline analysis, use the gnuplot/matplotlib export described below.

### 2.4 Chart Generation Commands

The harness does not generate charts directly (no Python dependency in the Rust binary). Instead, it prints suggested commands after JSON export:

```
Results written to results/phase1_standard.json

Generate charts:
  # Speedup vs element count (all experiments)
  python3 scripts/plot_speedup.py results/phase1_standard.json -o charts/speedup.png

  # Bandwidth roofline (reduce, scan, GEMV)
  python3 scripts/plot_roofline.py results/phase1_standard.json -o charts/roofline.png

  # Comparison dashboard (GPU vs CPU across all experiments)
  python3 scripts/plot_dashboard.py results/phase1_standard.json -o charts/dashboard.png

  # gnuplot one-liner (speedup bars)
  gnuplot -e "set terminal png; set output 'speedup.png'; \
    plot 'results/phase1_standard.csv' using 4:6 with linespoints title 'GPU Speedup'"
```

The `scripts/` directory ships with three Python plotting scripts that consume the JSON output:

**`scripts/plot_speedup.py`** -- Log-log plot of speedup vs element count for each experiment. Horizontal lines at 2x, 5x, 10x thresholds from PM.md. Crossover points marked.

**`scripts/plot_roofline.py`** -- Classic roofline model with arithmetic intensity on X-axis, throughput on Y-axis. M4 Pro bandwidth ceiling at 273 GB/s. Each data point labeled with experiment name and element count.

**`scripts/plot_dashboard.py`** -- 2x3 grid of subplots: one per Phase 1 experiment. Each subplot shows GPU vs CPU time on dual Y-axis with speedup bars. Kill criterion line overlaid.

All scripts require only `matplotlib` and `numpy` (standard data science stack). No exotic dependencies.

### 2.5 Regression Comparison

When `--append` is used, results accumulate in the same JSON file with different timestamps. A comparison script detects regressions:

```bash
# Compare today's run against baseline
python3 scripts/compare_runs.py results/history.json \
    --baseline "2026-02-15T10:00:00Z" \
    --current  "2026-02-17T14:30:00Z" \
    --threshold 5%
```

Output:

```
Regression Report: 2026-02-15 vs 2026-02-17
+------------------+-----------+-----------+--------+---------+
| Experiment       | Baseline  | Current   | Delta  | Status  |
+------------------+-----------+-----------+--------+---------+
| reduce_sum_u32   |  11.51x   |  11.23x   | -2.4%  |   OK    |
| prefix_scan      |   5.23x   |   4.89x   | -6.5%  |  REGR   |
| stream_compact   |   3.87x   |   3.91x   | +1.0%  |   OK    |
| radix_sort       |   7.12x   |   7.08x   | -0.6%  |   OK    |
+------------------+-----------+-----------+--------+---------+
1 regression detected (threshold: 5%)
```

---

## 3. Experiment Configuration

### 3.1 TOML Configuration File

The harness reads `forge-bench.toml` from the working directory (overridable with `--config`). CLI flags override TOML values. TOML uses the `toml` crate for serde deserialization, consistent with Cargo ecosystem conventions.

```toml
# forge-bench.toml -- Experiment suite configuration
# All values here are defaults; CLI flags override.

[hardware]
# Auto-detected if omitted. Set manually for cross-machine comparison.
# bandwidth_theoretical_gbps = 273.0
# gpu_cores = 20

[defaults]
profile = "standard"
seed = 42
warmup_iterations = 3
measured_iterations = 10
cold_start = false
output_dir = "results"

# ── Preset Profiles ──────────────────────────────────────

[profiles.quick]
description = "Fast smoke test for development iteration"
sizes = ["1M"]
warmup_iterations = 1
measured_iterations = 3
# ~30 seconds for full Phase 1

[profiles.standard]
description = "Default profile for reliable measurements"
sizes = ["1M", "10M", "100M"]
warmup_iterations = 3
measured_iterations = 10
# ~5 minutes for full Phase 1

[profiles.thorough]
description = "Publication-quality measurements"
sizes = ["100K", "1M", "10M", "100M"]
warmup_iterations = 5
measured_iterations = 30
# ~20 minutes for full Phase 1

[profiles.ci]
description = "CI pipeline: quick with JSON output"
sizes = ["1M", "10M"]
warmup_iterations = 2
measured_iterations = 5
json_output = true

# ── Per-Experiment Overrides ─────────────────────────────

[experiments.reduce]
enabled = true
phase = 1
priority = "P0"
variants = ["sum_u32", "sum_f32", "min_u32", "max_u32"]
kill_criterion = "< 50% bandwidth utilization"
cpu_baseline = "rayon_parallel"
# Override sizes for this experiment only:
# sizes = ["10M", "100M"]

[experiments.scan]
enabled = true
phase = 1
priority = "P0"
variants = ["inclusive_u32", "exclusive_u32"]
kill_criterion = "< 2x vs CPU sequential scan"
cpu_baseline = "sequential_scan"

[experiments.compact]
enabled = true
phase = 1
priority = "P0"
selectivity_rates = [0.1, 0.5, 0.9]
kill_criterion = "< 2x vs CPU filter+collect"
cpu_baseline = "rayon_filter_collect"

[experiments.sort]
enabled = true
phase = 1
priority = "P0"
variants = ["u32", "f32"]
kill_criterion = "< 2x vs std::sort at 10M"
cpu_baseline = "std_sort_unstable"

[experiments.histogram]
enabled = true
phase = 2
priority = "P1"
bin_counts = [256, 65536]
kill_criterion = "< 2x vs CPU at 10M"
cpu_baseline = "sequential_histogram"

[experiments.filter]
enabled = true
phase = 2
priority = "P0"
selectivity_rates = [0.01, 0.1, 0.5, 0.9]
data_types = ["int64", "f64"]
kill_criterion = "< 1.5x vs CPU at 10M rows"
cpu_baseline = "rayon_filter"

[experiments.groupby]
enabled = true
phase = 2
priority = "P1"
group_cardinalities = [10, 1000, 100_000, 1_000_000]
kill_criterion = "< 1.5x vs CPU hashmap at 10M"
cpu_baseline = "hashmap_groupby"

[experiments.gemm]
enabled = true
phase = 2
priority = "P1"
matrix_sizes = [[256, 256], [1024, 1024], [4096, 4096]]
dtypes = ["f16", "f32"]
kill_criterion = "< 50% of Accelerate at 1024x1024"
cpu_baseline = "accelerate_blas"

[experiments.gemv]
enabled = true
phase = 3
priority = "P2"
shapes = [[768, 768], [2048, 768], [4096, 2048]]
kill_criterion = "< 80% of Accelerate bandwidth"
cpu_baseline = "accelerate_blas"

[experiments.pipeline]
enabled = true
phase = 2
priority = "P1"
query = "SELECT region, SUM(amount) FROM data WHERE amount > 100 GROUP BY region ORDER BY SUM(amount) DESC LIMIT 10"
kill_criterion = "GPU pipeline slower than CPU pipeline"
cpu_baseline = "idiomatic_rust_iterators"

[experiments.duckdb]
enabled = true
phase = 2
priority = "P1"
duckdb_path = "duckdb"  # PATH lookup or absolute path
kill_criterion = "> 2x slower than DuckDB on same query"

[experiments.spreadsheet]
enabled = true
phase = 3
priority = "P2"
formulas = ["sum", "average", "vlookup"]
kill_criterion = "No perceptible speedup on 1M cells"

[experiments.timeseries]
enabled = true
phase = 3
priority = "P2"
operations = ["moving_average", "vwap", "bollinger"]
window_size = 20
kill_criterion = "< 2x vs optimized CPU"

[experiments.json_parse]
enabled = true
phase = 3
priority = "P2"
formats = ["simple_csv", "complex_json", "ndjson"]
kill_criterion = "< 2x vs serde_json/csv"
cpu_baseline = "serde"

[experiments.hash_join]
enabled = true
phase = 3
priority = "P2"
table_sizes = [[1_000_000, 1_000_000], [10_000_000, 1_000_000], [10_000_000, 10_000_000]]
kill_criterion = "< 1x vs CPU (likely outcome)"
cpu_baseline = "hashmap_join"
```

### 3.2 Hardware Detection and Auto-Configuration

On startup, the harness queries Metal device properties and adjusts configuration:

```rust
// Pseudocode for hardware detection
struct HardwareInfo {
    chip_name: String,          // "Apple M4 Pro"
    gpu_core_count: u32,        // 20
    max_threads_per_threadgroup: u32, // 1024
    max_buffer_length: u64,     // bytes
    unified_memory_gb: u32,     // 48
    metal_family: String,       // "apple9"
    bandwidth_theoretical: f64, // 273.0 GB/s (lookup table by chip)
}
```

**Bandwidth lookup table** (since Metal does not expose theoretical bandwidth):

| Chip | Bandwidth (GB/s) | GPU Cores |
|------|-------------------|-----------|
| M4 | 120 | 10 |
| M4 Pro | 273 | 20 |
| M4 Max | 546 | 40 |
| M4 Ultra | 819 | 80 |
| M3 | 100 | 10 |
| M3 Pro | 150 | 18 |
| M3 Max | 400 | 40 |
| M2 | 100 | 10 |
| M2 Pro | 200 | 19 |
| M2 Max | 400 | 38 |
| M2 Ultra | 800 | 76 |
| M1 | 68.25 | 8 |
| M1 Pro | 200 | 16 |
| M1 Max | 400 | 32 |
| M1 Ultra | 800 | 64 |

**Auto-configuration rules:**

1. If 100M elements exceeds 80% of available unified memory (accounting for input + output + scratch), cap at 50M and warn.
2. If GPU core count < 16, use half the threadgroup count for occupancy tuning.
3. If `duckdb` binary is not found on PATH, disable the `duckdb` experiment and warn (do not fail the suite).
4. Hardware info is embedded in every JSON output file for reproducibility (NFR-6).

### 3.3 Size Notation Parser

The CLI and TOML config accept human-readable size notation:

| Input | Parsed Value |
|-------|-------------|
| `1K` | 1,000 |
| `10K` | 10,000 |
| `100K` | 100,000 |
| `1M` | 1,000,000 |
| `10M` | 10,000,000 |
| `100M` | 100,000,000 |
| `1B` | 1,000,000,000 |
| `1_000_000` | 1,000,000 |
| `1000000` | 1,000,000 |

These are element counts, not byte counts. The harness calculates byte counts from element type (`u32` = 4 bytes, `f32` = 4 bytes, `f16` = 2 bytes, `i64` = 8 bytes, `f64` = 8 bytes).

---

## 4. Developer Workflow

### 4.1 Adding a New Experiment

Every experiment is a Rust module implementing a trait. The pattern follows the existing `criterion` benchmark structure in `gpu-query/benches/` but adds GPU/CPU comparison and structured output.

**Step 1: Create the experiment module.**

```
src/experiments/
  mod.rs              # Experiment trait + registry
  reduce.rs           # Existing
  scan.rs             # Existing
  compact.rs          # Existing
  sort.rs             # Existing
  my_new_experiment.rs  # <-- new file
```

**Step 2: Implement the `Experiment` trait.**

```rust
/// Trait that every experiment must implement.
pub trait Experiment: Send + Sync {
    /// Unique name used in CLI and JSON output (e.g., "reduce_sum_u32").
    fn name(&self) -> &str;

    /// Human-readable display name (e.g., "Reduce (sum, u32)").
    fn display_name(&self) -> &str;

    /// Phase from PM.md (1, 2, or 3).
    fn phase(&self) -> u8;

    /// Priority from PM.md (P0, P1, P2).
    fn priority(&self) -> Priority;

    /// Kill criterion string from PM.md.
    fn kill_criterion(&self) -> &str;

    /// Data sizes this experiment supports.
    /// May differ from profile defaults (e.g., GEMM uses matrix dimensions).
    fn supported_sizes(&self) -> Vec<usize>;

    /// Generate test data for a given element count.
    /// Returns (gpu_buffers, cpu_data) for fair comparison.
    fn generate_data(&self, size: usize, seed: u64) -> ExperimentData;

    /// Run the GPU kernel. Returns timing in seconds.
    /// Must include buffer allocation, encode, commit, waitUntilCompleted.
    fn run_gpu(&self, data: &ExperimentData) -> GpuResult;

    /// Run the CPU baseline. Returns timing in seconds.
    fn run_cpu(&self, data: &ExperimentData) -> CpuResult;

    /// Validate that GPU and CPU produce the same result.
    /// Called once before measurement to catch correctness bugs.
    fn validate(&self, data: &ExperimentData) -> Result<(), String>;

    /// Compute experiment-specific metrics (e.g., bandwidth, GFLOPS).
    fn compute_metrics(&self, data: &ExperimentData, gpu: &GpuResult, cpu: &CpuResult) -> Metrics;
}
```

**Step 3: Register the experiment.**

In `src/experiments/mod.rs`:

```rust
pub fn all_experiments() -> Vec<Box<dyn Experiment>> {
    vec![
        Box::new(reduce::ReduceExperiment::new()),
        Box::new(scan::ScanExperiment::new()),
        // ... existing experiments ...
        Box::new(my_new_experiment::MyNewExperiment::new()),  // <-- add here
    ]
}
```

**Step 4: Add TOML configuration.**

Add a `[experiments.my_new_experiment]` section to `forge-bench.toml` with relevant parameters. The experiment reads its config through the standard `ExperimentConfig` deserialization path.

**Step 5: Add the experiment name to the CLI parser.**

Update the `EXPERIMENT` value list in the clap argument definition.

**Step 6: Run and verify.**

```bash
# Run just the new experiment at quick profile
forge-bench my_new_experiment --profile quick -v

# Validate correctness (runs validate() then one measured iteration)
forge-bench my_new_experiment --runs 1 -v
```

### 4.2 Adding a New Variant to an Existing Experiment

Variants are sub-configurations of an experiment (e.g., `reduce` has `sum_u32`, `sum_f32`, `min_u32`, `max_u32`). To add a variant:

1. Add the variant name to the `variants` array in the TOML config.
2. Handle the new variant in the experiment's `run_gpu()` and `run_cpu()` methods.
3. If the variant requires a new Metal kernel, add the `.metal` file and register it in the shader compilation step.

```toml
# Before
[experiments.reduce]
variants = ["sum_u32", "sum_f32", "min_u32", "max_u32"]

# After
[experiments.reduce]
variants = ["sum_u32", "sum_f32", "min_u32", "max_u32", "sum_f16"]
```

```bash
# Run only the new variant
forge-bench reduce --variant sum_f16 --profile quick
```

### 4.3 Regression Detection Workflow

For ongoing development, the recommended workflow is:

```bash
# 1. Establish baseline (once, on clean system)
forge-bench phase1 --profile thorough --json-file results/baseline.json

# 2. After kernel changes, run comparison
forge-bench phase1 --profile standard --json-file results/current.json

# 3. Check for regressions
python3 scripts/compare_runs.py \
    --baseline results/baseline.json \
    --current results/current.json \
    --threshold 5%

# 4. If regression detected, bisect with per-iteration verbose output
forge-bench reduce --profile quick -v 2>&1 | tee results/debug_reduce.log
```

**CI integration** (GitHub Actions or local pre-push hook):

```yaml
# .github/workflows/bench.yml (conceptual)
bench:
  runs-on: [self-hosted, macos, arm64]
  steps:
    - uses: actions/checkout@v4
    - run: cargo build --release -p forge-bench
    - run: ./target/release/forge-bench phase1 --profile ci --json-file bench.json
    - run: python3 scripts/compare_runs.py --baseline bench_baseline.json --current bench.json --threshold 10%
```

### 4.4 Project Directory Structure

```
metal-forge-compute/
  Cargo.toml
  forge-bench.toml              # Default experiment configuration
  src/
    main.rs                     # CLI entry point (clap parse -> dispatch)
    cli.rs                      # Argument definitions and parsing
    config.rs                   # TOML config loading and merging with CLI
    hardware.rs                 # Metal device detection and bandwidth lookup
    harness.rs                  # Measurement loop: warmup, measure, stats
    output/
      mod.rs
      table.rs                  # Terminal table formatting
      json.rs                   # JSON serialization
      csv.rs                    # CSV serialization
      roofline.rs               # ASCII roofline diagram
      summary.rs                # Suite summary and decision signal
      progress.rs               # indicatif progress bar management
    experiments/
      mod.rs                    # Experiment trait + registry
      reduce.rs
      scan.rs
      compact.rs
      sort.rs
      histogram.rs
      filter.rs
      groupby.rs
      gemm.rs
      gemv.rs
      pipeline.rs
      duckdb.rs
      spreadsheet.rs
      timeseries.rs
      json_parse.rs
      hash_join.rs
    data_gen.rs                 # Synthetic data generation (deterministic)
    stats.rs                    # Statistical analysis (mean, median, CV, outlier detection)
    metal_helpers.rs            # Shared Metal boilerplate (device, queue, buffer alloc)
  shaders/
    reduce.metal
    scan.metal
    compact.metal
    sort.metal
    histogram.metal
    filter.metal
    groupby.metal
    gemm.metal
    gemv.metal                  # Reused from gpu-inference-pipeline
  scripts/
    plot_speedup.py             # Speedup vs element count chart
    plot_roofline.py            # Roofline model chart
    plot_dashboard.py           # Multi-experiment comparison dashboard
    compare_runs.py             # Regression detection between JSON files
  results/                      # gitignored, benchmark output directory
    .gitkeep
```

---

## 5. Result Interpretation Guidance

### 5.1 Automatic Crossover-Point Detection

The harness computes the GPU/CPU crossover point for every experiment using linear interpolation between measured data points:

```
Crossover analysis: Reduce (sum, u32)

  GPU < CPU (skip zone)         GPU > CPU (advantage zone)
  ←─────────────────────────────|──────────────────────────→
  1K     10K     100K          63K          1M     10M    100M
                                ^
                          crossover: ~63,000 elements

  At crossover: GPU = 0.058ms, CPU = 0.059ms
  Below 63K:  CPU is faster (use CPU path in production)
  Above 63K:  GPU is faster (use GPU path in production)
  At 10M:     GPU is 10.51x faster
```

**How crossover is computed:**

1. For each pair of adjacent data points where the speedup crosses 1.0x, perform linear interpolation on log-scale element counts.
2. If GPU never beats CPU across all measured sizes, report "No crossover found in measured range (GPU slower at all sizes)."
3. If GPU always beats CPU (even at the smallest size), report "GPU faster at all measured sizes. Crossover < [smallest_size]."

### 5.2 Bandwidth Utilization Display

For bandwidth-bound experiments (reduce, scan, compact, GEMV), the harness displays achieved bandwidth as a percentage of theoretical peak:

```
Bandwidth Analysis: Reduce (sum, u32) at 100M elements

  Theoretical peak:  273.0 GB/s (M4 Pro memory bandwidth)
  Achieved:           73.9 GB/s
  Utilization:        27.1%

  Interpretation:
    < 30%   Low utilization. Likely bottlenecked by dispatch overhead,
            threadgroup synchronization, or insufficient occupancy.
    30-50%  Moderate. Typical for first-pass kernels. Optimization
            opportunities in threadgroup sizing, memory coalescing.
    50-70%  Good. Approaching practical limits for non-trivial kernels
            with reduction patterns.
    70-90%  Excellent. Near-optimal for streaming workloads.
    > 90%   Exceptional. Only achievable for pure memcpy-like patterns.

  This result: LOW -- investigate dispatch overhead and occupancy.
```

For compute-bound experiments (GEMM), the metric is GFLOPS and percentage of theoretical peak compute:

```
Compute Analysis: GEMM (FP16, 1024x1024)

  Theoretical peak:  ~16.6 TFLOPS (M4 Pro, FP16, estimated)
  Achieved:           8.3 TFLOPS
  Utilization:        50.0%

  vs Accelerate:      9.1 TFLOPS (91.2% utilization via AMX)
  Ratio:              0.91x (Metal achieves 91% of Accelerate)
```

### 5.3 Kill Criterion Evaluation

Each experiment in PM.md has a specific kill criterion. The harness automatically evaluates these and produces an unambiguous verdict:

```
Kill Criterion Evaluation
=========================

Experiment            Criterion                              Value     Status
────────────────────  ─────────────────────────────────────  ────────  ──────
Reduce                < 50% bandwidth utilization            27.1%     FAIL
Prefix Scan           < 2x vs CPU sequential scan            5.23x     PASS
Stream Compaction     < 2x vs CPU filter+collect             3.87x     PASS
Radix Sort            < 2x vs std::sort at 10M              7.12x     PASS
Filter (10M)          < 1.5x vs CPU at 10M rows             3.21x     PASS
Group-By (10M)        < 1.5x vs CPU hashmap at 10M          1.89x     PASS
Histogram             < 2x vs CPU at 10M                    4.55x     PASS
GEMM (1024)           < 50% of Accelerate at 1024x1024      91%       PASS
Pipeline              GPU pipeline slower than CPU pipeline  2.34x     PASS
DuckDB                > 2x slower than DuckDB               0.87x     PASS
────────────────────────────────────────────────────────────────────────────
                                                          9/10 PASS   1 FAIL

Notes:
  - Reduce FAIL: 27.1% bandwidth utilization is below 50% target.
    However, absolute speedup is 11.51x. The kill criterion may be
    too aggressive for reduction patterns with threadgroup sync overhead.
    Recommendation: PROCEED with caveat -- optimize reduce kernel.
```

### 5.4 Go/No-Go Decision Signal

After Phase 1 completes, the harness maps results to the PM.md Decision Matrix and outputs a concrete recommendation:

```
╔══════════════════════════════════════════════════════════════════╗
║                     DECISION SIGNAL: Phase 1                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Sort:     7.12x at 10M  ✓  (threshold: 5x)                    ║
║  Scan:     5.23x at 10M  ✓  (threshold: 5x)                    ║
║  Reduce:  11.51x at 10M  ✓  (threshold: 5x)                    ║
║                                                                  ║
║  Phase 1 result: "Sort/scan/reduce all >5x at 10M"             ║
║                                                                  ║
║  Decision Matrix lookup:                                         ║
║    IF Phase 2 shows filter/group-by >3x at 10M:                ║
║      → SHIP LIBRARY + BUILD PRODUCT                             ║
║    IF Phase 2 shows relational ops <2x:                         ║
║      → SHIP PRIMITIVES LIBRARY ONLY                             ║
║                                                                  ║
║  Recommendation: PROCEED TO PHASE 2                             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

If any kill criterion triggers an experiment failure:

```
╔══════════════════════════════════════════════════════════════════╗
║                  DECISION SIGNAL: Phase 1                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Sort:     1.3x at 10M  ✗  (threshold: 2x)   KILL TRIGGERED   ║
║  Scan:     1.8x at 10M  ✗  (threshold: 2x)   KILL TRIGGERED   ║
║  Reduce:   3.2x at 10M  ✓  (threshold: n/a)                    ║
║                                                                  ║
║  Phase 1 result: "All primitives <2x at 10M"                   ║
║                                                                  ║
║  Decision Matrix lookup:                                         ║
║    → KILL LIBRARY THESIS. Pivot to ML/inference only.           ║
║                                                                  ║
║  Recommendation: STOP. Do not proceed to Phase 2.               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### 5.5 Summary Report Generation

After a complete suite run, the harness generates a markdown summary file alongside the JSON output:

```bash
forge-bench all --profile thorough --json-file results/full_run.json
# Automatically generates: results/full_run_summary.md
```

The summary file contains:

1. **Executive summary** -- one-paragraph go/no-go recommendation.
2. **Hardware context** -- detected chip, bandwidth, core count.
3. **Results table** -- all experiments, all sizes, verdicts.
4. **Crossover table** -- minimum N for GPU advantage per experiment.
5. **Bandwidth utilization table** -- achieved vs peak for bandwidth-bound experiments.
6. **Kill criterion evaluation** -- pass/fail per experiment.
7. **Decision matrix mapping** -- which row of the PM.md decision matrix applies.
8. **Anomaly notes** -- any measurements with CV > 5% (NFR-1 violation), any unexpected results.

This markdown file is designed to be copy-pasted directly into a decision document or Slack message for stakeholder review.

---

## 6. Measurement Protocol

This section specifies the exact measurement methodology to satisfy NFR-1 through NFR-7 from PM.md.

### 6.1 Warm-Up Protocol (NFR-7)

```
For each (experiment, element_count) data point:
  1. Generate data (excluded from timing)
  2. Allocate GPU buffers and copy data (excluded from timing)
  3. Run `warmup_iterations` GPU dispatches (discarded, not recorded)
  4. Run `measured_iterations` GPU dispatches (recorded)
  5. Run `warmup_iterations` CPU executions (discarded)
  6. Run `measured_iterations` CPU executions (recorded)
```

Warm-up ensures the SLC is populated and the Metal driver has compiled the pipeline state. Step 3 before step 4, and step 5 before step 6, prevents cold-start bias.

When `--cold` is specified, warm-up steps 3 and 5 are skipped, and only iteration 1 is reported (measures first-dispatch latency).

### 6.2 Timing Boundaries (NFR-2)

GPU timing includes the full dispatch cycle:

```rust
let start = Instant::now();
// Buffer allocation is OUTSIDE timing (pre-allocated in data gen)
command_buffer.encode(|encoder| { /* set buffers, dispatch */ });
command_buffer.commit();
command_buffer.wait_until_completed();  // Synchronous wait
let elapsed = start.elapsed();
```

This deliberately includes encode + commit + GPU execution + synchronization overhead, matching real-world usage. Kernel-only time (from Metal GPU timestamps) is recorded separately in verbose mode but is not the primary metric.

### 6.3 Statistical Analysis (NFR-1)

For each data point, the harness computes:

| Metric | Definition |
|--------|-----------|
| Mean | Arithmetic mean of measured iterations |
| Median | Middle value (robust to outliers) |
| Std Dev | Standard deviation |
| CV% | Coefficient of variation (std_dev / mean * 100) |
| Min | Fastest iteration |
| Max | Slowest iteration |

If CV > 5% for any data point, the harness prints a warning:

```
WARNING: reduce_sum_u32 at 100K has CV=7.2% (target: <5%)
  Consider increasing --runs or checking for thermal throttling.
  Individual timings: [0.061, 0.059, 0.058, 0.091, 0.060, ...]
                                              ^^^^^ outlier
```

Outlier detection uses the 1.5*IQR rule. Outliers are flagged but not removed from statistics (removal would bias results).

### 6.4 Memory Reporting (NFR-4)

After each GPU experiment, the harness queries `MTLDevice.currentAllocatedSize` and reports peak GPU memory:

```
Memory: 8.0 MB allocated (input: 4.0 MB + output: 4.0 MB + scratch: 0 MB)
```

For experiments approaching memory limits (100M elements with multiple buffers), the harness warns if allocation exceeds 80% of unified memory.

---

## 7. Crate Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `clap` | 4.x | CLI argument parsing (derive mode) |
| `toml` | 0.8.x | TOML config file parsing |
| `serde` | 1.x | Serialization framework |
| `serde_json` | 1.x | JSON output |
| `indicatif` | 0.17.x | Progress bars and spinners |
| `comfy-table` | 7.x | Terminal table formatting |
| `colored` | 2.x | ANSI color output for verdicts |
| `objc2-metal` | 0.3 | Metal GPU API bindings |
| `objc2-foundation` | 0.3 | Foundation framework bindings |
| `block2` | 0.6 | Objective-C block support |
| `rayon` | 1.x | Multi-threaded CPU baselines |
| `rand` | 0.8.x | Deterministic data generation |
| `csv` | 1.x | CSV output writing |
| `chrono` | 0.4.x | Timestamp formatting |
| `sysinfo` | 0.32.x | System/hardware detection |

No dependency on `criterion` -- the harness has its own measurement loop because criterion's model (statistical regression detection, adaptive iteration count) is not suited for GPU dispatch timing where we need fixed iteration counts, explicit warm-up control, and dual GPU/CPU measurement per data point.

---

## 8. Error Handling and Edge Cases

### 8.1 Metal Device Not Found

```
ERROR: No Metal GPU device found.
  This benchmark suite requires Apple Silicon with Metal support.
  Detected system: [system info]

  If running in a VM or CI without GPU access, use --no-gpu for CPU-only baselines.
```

### 8.2 Experiment Disabled or Unavailable

```
WARNING: Experiment 'duckdb' skipped: duckdb binary not found on PATH.
  Install with: brew install duckdb
  Or set path in forge-bench.toml: [experiments.duckdb] duckdb_path = "/path/to/duckdb"
```

### 8.3 Memory Exhaustion

```
WARNING: Experiment 'sort' at 100M elements requires ~1.6 GB GPU memory.
  Available unified memory: 48 GB (current allocation: 0.4 GB).
  Proceeding, but results may be affected by memory pressure.
```

If allocation fails:

```
ERROR: Metal buffer allocation failed for sort at 100M elements.
  Requested: 1.6 GB, Available: insufficient.
  Skipping this data point. Try --size 50M or smaller.
```

### 8.4 Thermal Throttling Detection

If median iteration time increases by > 20% compared to the first measured iteration within a single data point, the harness flags potential thermal throttling:

```
WARNING: Possible thermal throttling detected in reduce_sum_u32 at 100M.
  First iteration: 5.21ms, Last iteration: 6.83ms (+31%)
  Consider: (1) waiting between experiments, (2) checking Activity Monitor,
  (3) reducing --runs to avoid sustained GPU load.
```

---

## 9. Accessibility and Portability

### 9.1 No-Color Mode

All colored output respects the `NO_COLOR` environment variable (per no-color.org convention) and the `--no-color` flag. Verdicts are shown as text labels (PASS/FAIL/WEAK/SKIP) regardless of color support.

### 9.2 Machine-Readable First

The `--json` and `--json-file` flags are the primary integration surface. Terminal tables are for human consumption during development. Any automated tooling (CI, dashboards, regression detection) should consume JSON.

### 9.3 Cross-Machine Comparison

JSON output includes full hardware info. The `scripts/compare_runs.py` script can compare results from different machines when both JSON files are provided:

```bash
python3 scripts/compare_runs.py \
    --file1 results/m4_pro.json --label1 "M4 Pro" \
    --file2 results/m2_max.json --label2 "M2 Max" \
    --normalize-bandwidth  # Normalize by theoretical peak
```

This enables "is M4 Pro better than M2 Max for GPU compute?" comparisons that account for different bandwidth ceilings.
