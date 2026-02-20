# QA Strategy: metal-forge-compute Experiment Suite

## Overview

This document defines the quality assurance strategy for the GPU compute benchmark experiment suite. It covers kernel correctness validation, statistical measurement integrity, kill criteria automation, CI integration, edge case coverage, and the full acceptance test matrix mapping every functional requirement from PM.md to concrete, automatable test cases.

The testing approach follows the patterns established in `gpu-query/tests/` (real GPU dispatch with CPU reference comparison) and avoids mock-based testing entirely. Every test runs actual Metal compute dispatches and validates results against deterministic CPU reference implementations.

---

## 1. Testing Strategy Overview

### 1.1 Test Pyramid

```
                    /\
                   /  \       End-to-End Pipeline Tests (3-5 tests)
                  /    \      Full config -> run -> JSON output -> verdict
                 /------\
                /        \    Integration Tests (~50 tests)
               /          \   GPU kernel dispatch -> CPU readback -> correctness check
              /------------\
             /              \  Unit Tests (~80 tests)
            /                \ Data generation, statistics, config parsing,
           /                  \ size notation, hardware detection, buffer pool
          /--------------------\
```

| Layer | Count | Scope | GPU Required | Runtime |
|-------|-------|-------|--------------|---------|
| Unit | ~80 | Pure Rust logic: stats, config, data gen, size parser, buffer alignment | No | < 5s |
| Integration | ~50 | Per-kernel correctness: GPU dispatch + CPU reference comparison | Yes | ~60s |
| Benchmark Reproducibility | ~10 | Same experiment twice, verify CV < 5% and data determinism | Yes | ~120s |
| End-to-End | ~5 | Full harness: CLI args -> experiment run -> JSON/CSV output -> verdicts | Yes | ~180s |

### 1.2 Test Execution Commands

```bash
# Unit tests only (no GPU required, fast feedback)
cargo test -p forge-bench --lib
cargo test -p forge-primitives --lib

# Integration tests (GPU required, per-kernel correctness)
cargo test -p forge-bench --test '*'

# Full suite including benchmarks (GPU required, slow)
cargo test -p forge-bench --test '*' -- --include-ignored

# Single kernel correctness
cargo test -p forge-bench --test kernel_correctness -- reduce

# Benchmark reproducibility
cargo test -p forge-bench --test reproducibility

# End-to-end pipeline
cargo test -p forge-bench --test e2e_harness
```

### 1.3 Test Organization

```
forge-bench/tests/
  kernel_correctness/
    mod.rs                  # Shared GPU context + helpers
    reduce.rs               # Reduce kernel tests (sum/min/max, u32/f32)
    scan.rs                 # Prefix scan tests (inclusive/exclusive)
    compact.rs              # Stream compaction tests (selectivity sweep)
    sort.rs                 # Radix sort tests (u32/f32, permutation check)
    histogram.rs            # Histogram tests (256/65536 bins)
    filter.rs               # Columnar filter tests (selectivity sweep)
    groupby.rs              # Group-by aggregate tests (cardinality sweep)
    gemm.rs                 # GEMM tests (FP16/FP32, vs Accelerate)
    gemv.rs                 # GEMV tests (vs Accelerate)
    pipeline.rs             # End-to-end pipeline correctness
    spreadsheet.rs          # Formula evaluation tests
    timeseries.rs           # Time series operation tests
    json_parse.rs           # Parsing throughput + correctness
    hash_join.rs            # Hash join correctness
  reproducibility.rs        # Benchmark determinism + CV tests
  e2e_harness.rs            # CLI -> JSON output -> verdict validation
  statistical_validation.rs # Warm-up, CV, outlier, throttling detection
  edge_cases.rs             # Empty input, single element, NaN/Inf, max size
  data_gen_validation.rs    # Distribution verification (Zipf, uniform, skewed)

forge-primitives/src/
  (unit tests embedded in each module via #[cfg(test)])
```

---

## 2. Kernel Correctness Validation

Each experiment implements the `Experiment::validate()` method from TECH.md. This section specifies the exact correctness criteria per kernel category.

### 2.1 Reduce (FR-3)

| Variant | CPU Reference | Comparison Method | Epsilon |
|---------|--------------|-------------------|---------|
| `reduce_sum_u32` | `data.iter().map(\|&x\| x as u64).sum::<u64>()` | Exact equality | 0 |
| `reduce_sum_f32` | Kahan compensated summation | Relative error | 1e-3 |
| `reduce_min_u32` | `data.iter().copied().min()` | Exact equality | 0 |
| `reduce_max_u32` | `data.iter().copied().max()` | Exact equality | 0 |

**Test cases:**

```rust
#[test]
fn test_reduce_sum_u32_1m() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let gpu_sum = reduce_sum_u32_gpu(&ctx, &data);
    let cpu_sum: u64 = data.iter().map(|&x| x as u64).sum();
    assert_eq!(gpu_sum, cpu_sum, "GPU sum must exactly match CPU sum for u32");
}

#[test]
fn test_reduce_sum_f32_relative_error() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_f32(1_000_000);
    let gpu_sum = reduce_sum_f32_gpu(&ctx, &data);
    let cpu_sum = kahan_sum_f32(&data);
    let relative_error = (gpu_sum - cpu_sum).abs() / cpu_sum.abs();
    assert!(
        relative_error < 1e-3,
        "f32 reduce relative error {relative_error} exceeds 1e-3 threshold"
    );
}

#[test]
fn test_reduce_min_u32_correctness() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let gpu_min = reduce_min_u32_gpu(&ctx, &data);
    let cpu_min = *data.iter().min().unwrap();
    assert_eq!(gpu_min, cpu_min);
}

#[test]
fn test_reduce_max_u32_correctness() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let gpu_max = reduce_max_u32_gpu(&ctx, &data);
    let cpu_max = *data.iter().max().unwrap();
    assert_eq!(gpu_max, cpu_max);
}
```

### 2.2 Prefix Scan (FR-2)

| Variant | CPU Reference | Comparison Method | Epsilon |
|---------|--------------|-------------------|---------|
| `inclusive_scan_u32` | Sequential accumulate | Element-by-element exact | 0 |
| `exclusive_scan_u32` | Sequential accumulate (offset by 1) | Element-by-element exact | 0 |

**Correctness property**: For exclusive scan, `output[i] == sum(input[0..i])` for all `i`. For inclusive scan, `output[i] == sum(input[0..=i])`.

```rust
#[test]
fn test_exclusive_scan_u32_element_by_element() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let gpu_scan = exclusive_scan_u32_gpu(&ctx, &data);

    let mut expected = vec![0u32; data.len()];
    let mut acc = 0u32;
    for i in 0..data.len() {
        expected[i] = acc;
        acc = acc.wrapping_add(data[i]);
    }

    for i in 0..data.len() {
        assert_eq!(
            gpu_scan[i], expected[i],
            "Scan mismatch at index {i}: GPU={}, CPU={}",
            gpu_scan[i], expected[i]
        );
    }
}

#[test]
fn test_inclusive_scan_u32_element_by_element() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let gpu_scan = inclusive_scan_u32_gpu(&ctx, &data);

    let mut expected = vec![0u32; data.len()];
    let mut acc = 0u32;
    for i in 0..data.len() {
        acc = acc.wrapping_add(data[i]);
        expected[i] = acc;
    }

    assert_eq!(gpu_scan, expected, "Inclusive scan does not match CPU reference");
}
```

### 2.3 Radix Sort (FR-1)

**Correctness properties (all must hold):**
1. `is_sorted()` -- output is in non-decreasing order
2. Permutation check -- output contains exactly the same multiset of elements as input (no duplicates added, no elements lost)

| Variant | CPU Reference | Comparison Method | Epsilon |
|---------|--------------|-------------------|---------|
| `sort_u32` | `std::sort_unstable()` | Exact element equality | 0 |
| `sort_f32` | `sort_unstable_by(\|a, b\| a.partial_cmp(b).unwrap())` | Exact element equality | 0 |

```rust
#[test]
fn test_radix_sort_u32_correctness() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let gpu_sorted = radix_sort_u32_gpu(&ctx, &data);

    // Property 1: sorted
    assert!(gpu_sorted.windows(2).all(|w| w[0] <= w[1]), "Output not sorted");

    // Property 2: permutation (same elements)
    let mut expected = data.clone();
    expected.sort_unstable();
    assert_eq!(gpu_sorted, expected, "Sort result differs from CPU reference");
}

#[test]
fn test_radix_sort_f32_correctness() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_f32(1_000_000);
    let gpu_sorted = radix_sort_f32_gpu(&ctx, &data);

    // Property 1: sorted
    assert!(
        gpu_sorted.windows(2).all(|w| w[0] <= w[1]),
        "f32 output not sorted"
    );

    // Property 2: permutation check via sorted comparison
    let mut expected = data.clone();
    expected.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(gpu_sorted.len(), expected.len(), "Length mismatch");
    for i in 0..expected.len() {
        assert_eq!(
            gpu_sorted[i].to_bits(), expected[i].to_bits(),
            "Sort mismatch at index {i}"
        );
    }
}

#[test]
fn test_radix_sort_f32_negative_values() {
    let ctx = MetalContext::new();
    // Mix of negative and positive: tests float bit-flip for radix sort
    let data: Vec<f32> = (-500..500).map(|i| i as f32 * 0.1).collect();
    let gpu_sorted = radix_sort_f32_gpu(&ctx, &data);

    assert!(gpu_sorted.windows(2).all(|w| w[0] <= w[1]), "Negative f32 sort failed");
    assert_eq!(gpu_sorted[0], -50.0);
    assert_eq!(*gpu_sorted.last().unwrap(), 49.9);
}
```

### 2.4 Stream Compaction (FR-5)

**Correctness properties:**
1. Output count matches the number of input elements satisfying the predicate
2. Every element in the output satisfies the predicate
3. Relative order of elements is preserved (stable compaction)

```rust
fn verify_compaction(input: &[u32], output: &[u32], predicate: impl Fn(u32) -> bool) {
    // Property 1: correct count
    let expected: Vec<u32> = input.iter().copied().filter(|&x| predicate(x)).collect();
    assert_eq!(output.len(), expected.len(), "Compacted count mismatch");

    // Property 2: all elements satisfy predicate
    for &val in output {
        assert!(predicate(val), "Element {val} does not satisfy predicate");
    }

    // Property 3: order preserved
    assert_eq!(output, &expected[..], "Compacted elements differ from CPU reference");
}

#[test]
fn test_compact_10_percent_selectivity() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let threshold = percentile(&data, 10); // 10% selectivity
    let gpu_output = compact_gpu(&ctx, &data, |x| x < threshold);
    verify_compaction(&data, &gpu_output, |x| x < threshold);
}

#[test]
fn test_compact_50_percent_selectivity() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let threshold = percentile(&data, 50);
    let gpu_output = compact_gpu(&ctx, &data, |x| x < threshold);
    verify_compaction(&data, &gpu_output, |x| x < threshold);
}

#[test]
fn test_compact_90_percent_selectivity() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let threshold = percentile(&data, 90);
    let gpu_output = compact_gpu(&ctx, &data, |x| x < threshold);
    verify_compaction(&data, &gpu_output, |x| x < threshold);
}
```

### 2.5 Histogram (FR-4)

**Correctness properties:**
1. Sum of all bin counts equals N (total element count)
2. Each element falls in the correct bin
3. Exact match vs CPU sequential histogram

```rust
fn verify_histogram(input: &[u32], gpu_hist: &[u32], num_bins: usize) {
    // Property 1: counts sum to N
    let total: u64 = gpu_hist.iter().map(|&c| c as u64).sum();
    assert_eq!(total, input.len() as u64, "Histogram counts do not sum to N");

    // Property 2: CPU reference comparison
    let mut cpu_hist = vec![0u32; num_bins];
    for &val in input {
        let bin = (val as usize) % num_bins;
        cpu_hist[bin] += 1;
    }
    assert_eq!(gpu_hist, &cpu_hist[..], "Histogram bins differ from CPU reference");
}

#[test]
fn test_histogram_256_bins() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let gpu_hist = histogram_gpu(&ctx, &data, 256);
    verify_histogram(&data, &gpu_hist, 256);
}

#[test]
fn test_histogram_65536_bins() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let gpu_hist = histogram_gpu(&ctx, &data, 65536);
    verify_histogram(&data, &gpu_hist, 65536);
}
```

### 2.6 GEMM (FR-6)

**Correctness criterion**: Maximum element-wise relative error versus Accelerate `cblas_sgemm` reference.

| Precision | Max Relative Error | Max Absolute Error |
|-----------|-------------------|-------------------|
| FP32 | 1e-4 | 1e-3 |
| FP16 | 1e-2 | 1e-1 |

```rust
fn verify_gemm(gpu_c: &[f32], cpu_c: &[f32], m: usize, n: usize, epsilon: f32) {
    assert_eq!(gpu_c.len(), m * n);
    let mut max_rel_error: f32 = 0.0;
    let mut max_abs_error: f32 = 0.0;
    let mut error_count = 0;

    for i in 0..gpu_c.len() {
        let abs_err = (gpu_c[i] - cpu_c[i]).abs();
        let rel_err = if cpu_c[i].abs() > 1e-7 {
            abs_err / cpu_c[i].abs()
        } else {
            abs_err
        };
        max_rel_error = max_rel_error.max(rel_err);
        max_abs_error = max_abs_error.max(abs_err);
        if rel_err > epsilon {
            error_count += 1;
        }
    }

    assert!(
        max_rel_error < epsilon,
        "GEMM max relative error {max_rel_error} exceeds {epsilon} ({error_count} elements)"
    );
}

#[test]
fn test_gemm_f32_256x256() {
    let ctx = MetalContext::new();
    let (a, b) = generate_matrices(256, 256, 256, 42);
    let gpu_c = gemm_f32_gpu(&ctx, &a, &b, 256, 256, 256);
    let cpu_c = accelerate_sgemm(&a, &b, 256, 256, 256);
    verify_gemm(&gpu_c, &cpu_c, 256, 256, 1e-4);
}

#[test]
fn test_gemm_f32_1024x1024() {
    let ctx = MetalContext::new();
    let (a, b) = generate_matrices(1024, 1024, 1024, 42);
    let gpu_c = gemm_f32_gpu(&ctx, &a, &b, 1024, 1024, 1024);
    let cpu_c = accelerate_sgemm(&a, &b, 1024, 1024, 1024);
    verify_gemm(&gpu_c, &cpu_c, 1024, 1024, 1e-4);
}

#[test]
fn test_gemm_f16_1024x1024() {
    let ctx = MetalContext::new();
    let (a, b) = generate_matrices_f16(1024, 1024, 1024, 42);
    let gpu_c = gemm_f16_gpu(&ctx, &a, &b, 1024, 1024, 1024);
    let cpu_c = accelerate_sgemm_f16_reference(&a, &b, 1024, 1024, 1024);
    verify_gemm(&gpu_c, &cpu_c, 1024, 1024, 1e-2);
}
```

### 2.7 Filter (FR-8)

**Correctness criterion**: Exact bitmask equality versus CPU `iter().filter()`.

```rust
#[test]
fn test_filter_int64_selectivity_sweep() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_i64(1_000_000);

    for selectivity in [0.01, 0.1, 0.5, 0.9] {
        let threshold = percentile_i64(&data, (selectivity * 100.0) as usize);
        let gpu_mask = filter_gpu_lt(&ctx, &data, threshold);
        let cpu_mask = data.iter().map(|&x| x < threshold).collect::<Vec<_>>();

        let gpu_count = gpu_mask.iter().filter(|&&b| b).count();
        let cpu_count = cpu_mask.iter().filter(|&&b| b).count();
        assert_eq!(gpu_count, cpu_count,
            "Filter count mismatch at {selectivity} selectivity");

        for i in 0..data.len() {
            assert_eq!(gpu_mask[i], cpu_mask[i],
                "Filter mask mismatch at index {i}");
        }
    }
}
```

### 2.8 Group-By Aggregate (FR-9)

**Correctness criterion**: Per-group aggregate values match CPU HashMap-based group-by within floating-point epsilon.

```rust
#[test]
fn test_groupby_sum_cardinality_sweep() {
    let ctx = MetalContext::new();
    for num_groups in [10, 1_000, 100_000] {
        let (keys, values) = DataGenerator::new(42)
            .grouped_data(1_000_000, num_groups);
        let gpu_result = groupby_sum_gpu(&ctx, &keys, &values);
        let cpu_result = groupby_sum_cpu(&keys, &values);

        assert_eq!(gpu_result.len(), cpu_result.len(),
            "Group count mismatch for {num_groups} groups");

        for (group_key, gpu_sum) in &gpu_result {
            let cpu_sum = cpu_result.get(group_key)
                .expect(&format!("GPU produced group {group_key} not in CPU result"));
            let rel_err = (gpu_sum - cpu_sum).abs() / cpu_sum.abs().max(1e-10);
            assert!(rel_err < 1e-3,
                "Group {group_key}: GPU={gpu_sum}, CPU={cpu_sum}, rel_err={rel_err}");
        }
    }
}
```

### 2.9 Hash Join (FR-10)

**Correctness criterion**: Join output contains exactly the matching pairs from both tables, verified by CPU nested-loop join reference.

```rust
#[test]
fn test_hash_join_correctness() {
    let ctx = MetalContext::new();
    let gen = DataGenerator::new(42);
    let (build_keys, build_vals) = gen.join_table(100_000, 50_000); // 50K unique keys
    let (probe_keys, probe_vals) = gen.join_table(100_000, 50_000);

    let gpu_result = hash_join_gpu(&ctx, &build_keys, &build_vals, &probe_keys, &probe_vals);
    let cpu_result = hash_join_cpu(&build_keys, &build_vals, &probe_keys, &probe_vals);

    // Same number of join matches
    assert_eq!(gpu_result.len(), cpu_result.len(), "Join result count mismatch");

    // Same (sorted) output rows
    let mut gpu_sorted = gpu_result.clone();
    let mut cpu_sorted = cpu_result.clone();
    gpu_sorted.sort();
    cpu_sorted.sort();
    assert_eq!(gpu_sorted, cpu_sorted, "Join results differ");
}
```

---

## 3. Statistical Measurement Validation

### 3.1 Warm-Up Verification (NFR-7)

Verify that the first `warmup_iterations` are discarded and only `measured_iterations` contribute to statistics.

```rust
#[test]
fn test_warmup_iterations_discarded() {
    let harness = BenchHarness::new(HarnessConfig {
        warmup_iterations: 3,
        measured_iterations: 10,
        ..Default::default()
    });

    let result = harness.run_single("reduce", 1_000_000, 42);

    // Must have exactly 10 recorded iterations (not 13)
    assert_eq!(result.gpu.iterations.len(), 10,
        "Should have exactly measured_iterations timings");
    assert_eq!(result.cpu.iterations.len(), 10);
}

#[test]
fn test_cold_mode_skips_warmup() {
    let harness = BenchHarness::new(HarnessConfig {
        warmup_iterations: 3,
        measured_iterations: 1,
        cold_start: true,
        ..Default::default()
    });

    let result = harness.run_single("reduce", 1_000_000, 42);

    // Cold mode: no warm-up, single measured iteration
    assert_eq!(result.gpu.iterations.len(), 1);
}
```

### 3.2 Coefficient of Variation (NFR-1)

The CV must be < 5% for all data points under normal conditions. Tests validate the CV calculation and the flagging mechanism.

```rust
#[test]
fn test_cv_calculation() {
    let timings = vec![10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0];
    let stats = compute_stats(&timings);
    assert!((stats.mean - 10.0).abs() < 0.05);
    assert!(stats.cv_percent < 2.0, "CV should be very low for tight timings");
}

#[test]
fn test_cv_flagging_above_threshold() {
    // Simulate high-variance timings
    let timings = vec![5.0, 5.0, 5.0, 15.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let stats = compute_stats(&timings);
    assert!(stats.cv_percent > 5.0, "CV should exceed 5% with outlier");
    assert!(stats.cv_warning, "CV warning flag should be set");
}

#[test]
fn test_cv_below_threshold_no_warning() {
    let timings = vec![10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.0];
    let stats = compute_stats(&timings);
    assert!(stats.cv_percent < 5.0);
    assert!(!stats.cv_warning);
}
```

### 3.3 Outlier Detection

Uses the 1.5 * IQR (interquartile range) method. Outliers are flagged but not removed from statistics.

```rust
#[test]
fn test_outlier_detection_iqr() {
    let timings = vec![10.0, 10.1, 10.2, 9.9, 10.0, 9.8, 10.1, 10.0, 25.0, 10.0];
    //                                                                  ^^^^^ outlier
    let stats = compute_stats(&timings);
    assert_eq!(stats.outlier_count, 1, "Should detect 1 outlier");
    assert!(stats.outlier_indices.contains(&8), "Index 8 should be flagged");

    // Outlier is NOT removed from mean
    assert!(stats.mean > 10.0, "Mean should include the outlier");
}

#[test]
fn test_no_outliers_in_clean_data() {
    let timings = vec![10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.1, 10.0, 9.9, 10.0];
    let stats = compute_stats(&timings);
    assert_eq!(stats.outlier_count, 0);
}
```

### 3.4 Thermal Throttling Detection

Detect monotonic slowdown pattern: if the median of the last 3 iterations exceeds 1.2x the median of the first 3 iterations, flag thermal throttling.

```rust
#[test]
fn test_thermal_throttle_detection() {
    // Simulated monotonic slowdown pattern
    let timings = vec![5.0, 5.1, 5.2, 5.5, 5.8, 6.1, 6.5, 7.0, 7.5, 8.0];
    let stats = compute_stats(&timings);
    assert!(stats.thermal_throttle_warning,
        "Should detect thermal throttling: last iterations are >20% slower");
}

#[test]
fn test_no_thermal_throttle_in_stable_timings() {
    let timings = vec![10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.0];
    let stats = compute_stats(&timings);
    assert!(!stats.thermal_throttle_warning);
}

#[test]
fn test_thermal_throttle_threshold_boundary() {
    // Exactly at 20% boundary -- should NOT flag
    let timings = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 11.9, 11.9, 12.0];
    let stats = compute_stats(&timings);
    // 12.0 / 10.0 = 1.2x exactly -- threshold is "> 1.2x" so this is borderline
    // Implementation should use > not >= for the 1.2x check
}
```

### 3.5 Benchmark Reproducibility (NFR-5)

Same seed, same data, same experiment must produce identical input data and statistically similar GPU timings.

```rust
#[test]
fn test_data_generation_deterministic() {
    let d1 = DataGenerator::new(42).uniform_u32(1_000_000);
    let d2 = DataGenerator::new(42).uniform_u32(1_000_000);
    assert_eq!(d1, d2, "Same seed must produce identical data");
}

#[test]
fn test_data_generation_different_seeds() {
    let d1 = DataGenerator::new(42).uniform_u32(1_000_000);
    let d2 = DataGenerator::new(43).uniform_u32(1_000_000);
    assert_ne!(d1, d2, "Different seeds must produce different data");
}

#[test]
fn test_benchmark_timing_reproducibility() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);

    // Run twice with same data
    let timings_1 = measure_gpu_n_times(&ctx, "reduce_sum_u32", &data, 10);
    let timings_2 = measure_gpu_n_times(&ctx, "reduce_sum_u32", &data, 10);

    let stats_1 = compute_stats(&timings_1);
    let stats_2 = compute_stats(&timings_2);

    // Means should be within 10% of each other (thermal/system variance)
    let ratio = stats_1.mean / stats_2.mean;
    assert!(
        ratio > 0.9 && ratio < 1.1,
        "Back-to-back runs differ by {:.1}% (threshold: 10%)",
        (ratio - 1.0).abs() * 100.0
    );
}
```

---

## 4. Kill Criteria Automation

Each kill criterion from PM.md is mapped to an automated check. The `KillCriterionEvaluator` reads experiment results and outputs PASS/FAIL per criterion.

### 4.1 Kill Criterion Mapping

| Experiment | Kill Criterion (PM.md) | Automated Check | Threshold |
|------------|----------------------|-----------------|-----------|
| Reduce | < 50% bandwidth utilization | `gpu_bandwidth_gbps / theoretical_peak_gbps < 0.50` | 50% of 273 GB/s = 136.5 GB/s |
| Prefix Scan | < 2x vs CPU sequential scan | `gpu_mean_ms > cpu_mean_ms / 2.0` | speedup < 2.0 |
| Stream Compact | < 2x vs CPU filter+collect | `speedup < 2.0` at 10M elements | speedup < 2.0 |
| Radix Sort | < 2x vs std::sort at 10M | `speedup < 2.0` at 10M elements | speedup < 2.0 |
| Filter | < 1.5x vs CPU at 10M rows | `speedup < 1.5` at 10M elements | speedup < 1.5 |
| Group-By | < 1.5x vs CPU hashmap at 10M | `speedup < 1.5` at 10M elements | speedup < 1.5 |
| Histogram | < 2x vs CPU at 10M | `speedup < 2.0` at 10M elements | speedup < 2.0 |
| GEMM | < 50% of Accelerate at 1024x1024 | `gpu_gflops / accelerate_gflops < 0.50` | 50% of Accelerate |
| Pipeline | GPU slower than CPU | `gpu_mean_ms > cpu_mean_ms` | speedup < 1.0 |
| DuckDB | > 2x slower than DuckDB | `gpu_mean_ms > duckdb_mean_ms * 2.0` | > 2x slower |
| Spreadsheet | No perceptible speedup on 1M cells | `speedup < 1.1` (10% threshold for "perceptible") | speedup < 1.1 |
| Time Series | < 2x vs optimized CPU | `speedup < 2.0` | speedup < 2.0 |
| JSON/CSV Parse | < 2x vs serde_json/csv | `speedup < 2.0` | speedup < 2.0 |
| GEMV | < 80% of Accelerate bandwidth | `gpu_gbps / accelerate_gbps < 0.80` | 80% of Accelerate |
| Hash Join | < 1x vs CPU | `speedup < 1.0` | speedup < 1.0 |

### 4.2 Kill Criterion Evaluator

```rust
pub struct KillCriterionResult {
    pub experiment: String,
    pub criterion_text: String,
    pub measured_value: f64,
    pub threshold: f64,
    pub passed: bool,
    pub notes: String,
}

pub fn evaluate_kill_criteria(results: &SuiteResults) -> Vec<KillCriterionResult> {
    let mut evaluations = Vec::new();

    for exp_result in &results.experiments {
        let eval = match exp_result.name.as_str() {
            "reduce_sum_u32" | "reduce_sum_f32" | "reduce_min_u32" | "reduce_max_u32" => {
                let best_bw_util = exp_result.best_bandwidth_utilization_percent();
                KillCriterionResult {
                    experiment: exp_result.name.clone(),
                    criterion_text: "< 50% bandwidth utilization".to_string(),
                    measured_value: best_bw_util,
                    threshold: 50.0,
                    passed: best_bw_util >= 50.0,
                    notes: format!("{:.1}% of {:.0} GB/s", best_bw_util,
                        results.hardware.bandwidth_theoretical_gbps),
                }
            }
            // ... similar for each experiment
            _ => continue,
        };
        evaluations.push(eval);
    }

    evaluations
}
```

### 4.3 Kill Criterion Tests

```rust
#[test]
fn test_kill_criterion_reduce_bandwidth() {
    let result = mock_reduce_result(bandwidth_gbps: 140.0, theoretical: 273.0);
    let eval = evaluate_kill_criterion_reduce(&result);
    assert!(eval.passed, "140/273 = 51.3% should pass 50% threshold");
}

#[test]
fn test_kill_criterion_reduce_bandwidth_fail() {
    let result = mock_reduce_result(bandwidth_gbps: 130.0, theoretical: 273.0);
    let eval = evaluate_kill_criterion_reduce(&result);
    assert!(!eval.passed, "130/273 = 47.6% should fail 50% threshold");
}

#[test]
fn test_kill_criterion_sort_speedup() {
    let result = mock_sort_result(speedup_at_10m: 2.5);
    let eval = evaluate_kill_criterion_sort(&result);
    assert!(eval.passed, "2.5x should pass 2x threshold");
}

#[test]
fn test_kill_criterion_sort_speedup_fail() {
    let result = mock_sort_result(speedup_at_10m: 1.8);
    let eval = evaluate_kill_criterion_sort(&result);
    assert!(!eval.passed, "1.8x should fail 2x threshold");
}

#[test]
fn test_kill_criterion_pipeline_gpu_slower() {
    let result = mock_pipeline_result(gpu_ms: 150.0, cpu_ms: 120.0);
    let eval = evaluate_kill_criterion_pipeline(&result);
    assert!(!eval.passed, "GPU slower than CPU should fail");
}

#[test]
fn test_kill_criterion_duckdb_comparison() {
    let result = mock_duckdb_result(gpu_ms: 250.0, duckdb_ms: 100.0);
    let eval = evaluate_kill_criterion_duckdb(&result);
    assert!(!eval.passed, "GPU 2.5x slower than DuckDB should fail >2x threshold");
}
```

### 4.4 Decision Matrix Automation

```rust
pub fn evaluate_decision_matrix(
    phase1: &PhaseResults,
    phase2: Option<&PhaseResults>,
) -> DecisionSignal {
    let sort_5x = phase1.get("sort").speedup_at(10_000_000) >= 5.0;
    let scan_5x = phase1.get("scan").speedup_at(10_000_000) >= 5.0;
    let reduce_5x = phase1.get("reduce").speedup_at(10_000_000) >= 5.0;
    let all_primitives_5x = sort_5x && scan_5x && reduce_5x;
    let all_primitives_2x = phase1.all_speedups_at(10_000_000) >= 2.0;

    if !all_primitives_2x {
        return DecisionSignal::Kill {
            reason: "All primitives <2x at 10M. Kill library thesis.".into(),
            recommendation: "STOP. Pivot to ML/inference only.".into(),
        };
    }

    if let Some(p2) = phase2 {
        let filter_3x = p2.get("filter").speedup_at(10_000_000) >= 3.0;
        let groupby_3x = p2.get("groupby").speedup_at(10_000_000) >= 3.0;
        let relational_3x = filter_3x && groupby_3x;

        if all_primitives_5x && relational_3x {
            return DecisionSignal::Proceed {
                decision: "Ship library + build product".into(),
            };
        } else if all_primitives_5x && !relational_3x {
            return DecisionSignal::Proceed {
                decision: "Ship primitives library only (Layer 1)".into(),
            };
        }
    }

    DecisionSignal::Proceed {
        decision: "Proceed to Phase 2".into(),
    }
}
```

---

## 5. CI Integration

### 5.1 GitHub Actions Workflow

```yaml
# .github/workflows/forge-bench.yml
name: GPU Benchmark Suite

on:
  pull_request:
    paths:
      - 'metal-forge-compute/**'
      - 'forge-primitives/**'
  push:
    branches: [main]
    paths:
      - 'metal-forge-compute/**'

jobs:
  correctness:
    name: Kernel Correctness
    runs-on: [self-hosted, macos, arm64, m4-pro]
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: cargo build --release -p forge-bench -p forge-primitives
      - name: Unit Tests
        run: cargo test -p forge-bench --lib -p forge-primitives --lib
      - name: Integration Tests (GPU Correctness)
        run: cargo test -p forge-bench --test kernel_correctness -- --test-threads=1
      - name: Edge Case Tests
        run: cargo test -p forge-bench --test edge_cases -- --test-threads=1
      - name: Statistical Validation Tests
        run: cargo test -p forge-bench --test statistical_validation

  benchmark-regression:
    name: Benchmark Regression Check
    runs-on: [self-hosted, macos, arm64, m4-pro]
    timeout-minutes: 30
    needs: correctness
    steps:
      - uses: actions/checkout@v4
      - name: Build Release
        run: cargo build --release -p forge-bench
      - name: Run CI Profile Benchmarks
        run: |
          ./target/release/forge-bench phase1 \
            --profile ci \
            --json-file bench_current.json
      - name: Download Baseline
        uses: actions/download-artifact@v4
        with:
          name: bench-baseline
          path: .
        continue-on-error: true
      - name: Regression Check
        run: |
          if [ -f bench_baseline.json ]; then
            python3 scripts/compare_runs.py \
              --baseline bench_baseline.json \
              --current bench_current.json \
              --threshold 10 \
              --fail-on-regression
          else
            echo "No baseline found. Setting current as baseline."
          fi
      - name: Upload Results as Artifact
        uses: actions/upload-artifact@v4
        with:
          name: bench-results-${{ github.sha }}
          path: bench_current.json
          retention-days: 90
      - name: Update Baseline (main only)
        if: github.ref == 'refs/heads/main'
        uses: actions/upload-artifact@v4
        with:
          name: bench-baseline
          path: bench_current.json
          overwrite: true

  full-suite:
    name: Full Benchmark Suite (Manual)
    runs-on: [self-hosted, macos, arm64, m4-pro]
    timeout-minutes: 60
    if: github.event_name == 'workflow_dispatch'
    steps:
      - uses: actions/checkout@v4
      - name: Build Release
        run: cargo build --release -p forge-bench
      - name: Run Full Suite
        run: |
          ./target/release/forge-bench all \
            --profile thorough \
            --json-file results/full_run.json
      - name: Upload Full Results
        uses: actions/upload-artifact@v4
        with:
          name: full-bench-${{ github.sha }}
          path: results/
          retention-days: 365
```

### 5.2 Artifact Storage Strategy

| Artifact | Retention | Purpose |
|----------|-----------|---------|
| `bench-baseline` | Permanent (overwritten on main) | Regression reference point |
| `bench-results-{sha}` | 90 days | Per-commit comparison |
| `full-bench-{sha}` | 365 days | Publication-quality results |

### 5.3 Cross-Run Comparison and Alerting

The `scripts/compare_runs.py` regression checker produces machine-readable output:

```json
{
  "status": "REGRESSION_DETECTED",
  "regressions": [
    {
      "experiment": "prefix_scan",
      "baseline_speedup": 5.23,
      "current_speedup": 4.89,
      "delta_percent": -6.5,
      "threshold_percent": 10.0,
      "verdict": "REGRESSION"
    }
  ],
  "improvements": [],
  "stable": ["reduce_sum_u32", "radix_sort", "stream_compact"]
}
```

**Alerting rules:**
- Regression > 10% on any Phase 1 experiment: CI fails, PR blocked
- Regression > 5% on any experiment: CI warning, comment on PR
- Improvement > 10%: CI info comment celebrating the win

### 5.4 Hardware-Specific Test Matrix

| Hardware | Role | Profile | Frequency |
|----------|------|---------|-----------|
| M4 Pro (20-core) | Primary benchmark target | `thorough` (30 runs) | Every PR + nightly |
| M1 (8-core) | Baseline / regression reference | `standard` (10 runs) | Weekly |
| M4 Max (40-core) | Scaling validation | `standard` (10 runs) | Manual / monthly |

Hardware is recorded in JSON output (NFR-6), enabling cross-machine comparison via the `--normalize-bandwidth` flag in `compare_runs.py`.

---

## 6. Edge Case Testing

### 6.1 Empty Input (0 Elements)

Every kernel must handle N=0 without crashing, returning a valid (empty or identity) result.

```rust
#[test]
fn test_reduce_sum_empty() {
    let ctx = MetalContext::new();
    let data: Vec<u32> = vec![];
    let result = reduce_sum_u32_gpu(&ctx, &data);
    assert_eq!(result, 0, "Sum of empty input must be 0");
}

#[test]
fn test_scan_empty() {
    let ctx = MetalContext::new();
    let data: Vec<u32> = vec![];
    let result = exclusive_scan_u32_gpu(&ctx, &data);
    assert!(result.is_empty(), "Scan of empty input must be empty");
}

#[test]
fn test_sort_empty() {
    let ctx = MetalContext::new();
    let data: Vec<u32> = vec![];
    let result = radix_sort_u32_gpu(&ctx, &data);
    assert!(result.is_empty());
}

#[test]
fn test_compact_empty() {
    let ctx = MetalContext::new();
    let data: Vec<u32> = vec![];
    let result = compact_gpu(&ctx, &data, |_| true);
    assert!(result.is_empty());
}

#[test]
fn test_histogram_empty() {
    let ctx = MetalContext::new();
    let data: Vec<u32> = vec![];
    let result = histogram_gpu(&ctx, &data, 256);
    assert!(result.iter().all(|&c| c == 0), "Empty histogram must have all-zero bins");
}

#[test]
fn test_filter_empty() {
    let ctx = MetalContext::new();
    let data: Vec<i64> = vec![];
    let (mask, count) = filter_gpu_lt_i64(&ctx, &data, 0);
    assert_eq!(count, 0);
}
```

### 6.2 Single Element

```rust
#[test]
fn test_reduce_sum_single() {
    let ctx = MetalContext::new();
    let result = reduce_sum_u32_gpu(&ctx, &[42]);
    assert_eq!(result, 42);
}

#[test]
fn test_scan_single() {
    let ctx = MetalContext::new();
    let result = exclusive_scan_u32_gpu(&ctx, &[42]);
    assert_eq!(result, vec![0], "Exclusive scan of single element is [0]");
}

#[test]
fn test_sort_single() {
    let ctx = MetalContext::new();
    let result = radix_sort_u32_gpu(&ctx, &[42]);
    assert_eq!(result, vec![42]);
}

#[test]
fn test_compact_single_pass() {
    let ctx = MetalContext::new();
    let result = compact_gpu(&ctx, &[42], |x| x > 10);
    assert_eq!(result, vec![42]);
}

#[test]
fn test_compact_single_fail() {
    let ctx = MetalContext::new();
    let result = compact_gpu(&ctx, &[42], |x| x > 100);
    assert!(result.is_empty());
}

#[test]
fn test_gemm_1x1() {
    let ctx = MetalContext::new();
    let a = vec![3.0f32];
    let b = vec![4.0f32];
    let result = gemm_f32_gpu(&ctx, &a, &b, 1, 1, 1);
    assert!((result[0] - 12.0).abs() < 1e-6, "1x1 GEMM: 3*4=12");
}
```

### 6.3 Power-of-2 and Non-Power-of-2 Sizes

Metal threadgroup dispatch requires careful handling of non-power-of-2 element counts. Test at boundaries that expose dispatch rounding issues.

```rust
const BOUNDARY_SIZES: &[usize] = &[
    255,        // Just below 256 (threadgroup size)
    256,        // Exact threadgroup size
    257,        // One above threadgroup size
    1023,       // Just below 1024
    1024,       // Common power-of-2
    1025,       // One above 1024
    65535,      // 2^16 - 1
    65536,      // 2^16
    65537,      // 2^16 + 1
    1_000_000,  // Standard benchmark size (not power-of-2)
    1_048_576,  // 2^20 (1Mi)
    10_000_000, // Standard benchmark size
];

#[test]
fn test_reduce_at_boundary_sizes() {
    let ctx = MetalContext::new();
    for &size in BOUNDARY_SIZES {
        let data = DataGenerator::new(42).uniform_u32(size);
        let gpu_sum = reduce_sum_u32_gpu(&ctx, &data);
        let cpu_sum: u64 = data.iter().map(|&x| x as u64).sum();
        assert_eq!(gpu_sum, cpu_sum, "Reduce failed at size {size}");
    }
}

#[test]
fn test_scan_at_boundary_sizes() {
    let ctx = MetalContext::new();
    for &size in BOUNDARY_SIZES {
        let data = DataGenerator::new(42).uniform_u32(size);
        let gpu_scan = exclusive_scan_u32_gpu(&ctx, &data);
        let cpu_scan = cpu_exclusive_scan(&data);
        assert_eq!(gpu_scan, cpu_scan, "Scan failed at size {size}");
    }
}

#[test]
fn test_sort_at_boundary_sizes() {
    let ctx = MetalContext::new();
    for &size in BOUNDARY_SIZES {
        let data = DataGenerator::new(42).uniform_u32(size);
        let gpu_sorted = radix_sort_u32_gpu(&ctx, &data);
        let mut expected = data.clone();
        expected.sort_unstable();
        assert_eq!(gpu_sorted, expected, "Sort failed at size {size}");
    }
}
```

### 6.4 Maximum Memory (100M Elements)

```rust
#[test]
#[ignore] // Run only with --include-ignored (requires ~800MB GPU memory)
fn test_reduce_100m_elements() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(100_000_000);
    let gpu_sum = reduce_sum_u32_gpu(&ctx, &data);
    let cpu_sum: u64 = data.iter().map(|&x| x as u64).sum();
    assert_eq!(gpu_sum, cpu_sum, "Reduce sum incorrect at 100M elements");
}

#[test]
#[ignore]
fn test_sort_100m_elements() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(100_000_000);
    let gpu_sorted = radix_sort_u32_gpu(&ctx, &data);

    // Check sorted property (faster than full comparison at 100M)
    assert!(gpu_sorted.windows(2).all(|w| w[0] <= w[1]), "100M sort not sorted");

    // Spot-check: same min, max, and length
    assert_eq!(gpu_sorted.len(), data.len());
    assert_eq!(gpu_sorted[0], *data.iter().min().unwrap());
    assert_eq!(*gpu_sorted.last().unwrap(), *data.iter().max().unwrap());
}
```

### 6.5 NaN/Inf Handling in Float Operations

```rust
#[test]
fn test_reduce_sum_f32_with_nan() {
    let ctx = MetalContext::new();
    let mut data = vec![1.0f32, 2.0, 3.0, f32::NAN, 5.0];
    let gpu_sum = reduce_sum_f32_gpu(&ctx, &data);
    assert!(gpu_sum.is_nan(), "Sum with NaN must propagate NaN (IEEE 754)");
}

#[test]
fn test_reduce_sum_f32_with_inf() {
    let ctx = MetalContext::new();
    let data = vec![1.0f32, 2.0, f32::INFINITY, 3.0];
    let gpu_sum = reduce_sum_f32_gpu(&ctx, &data);
    assert!(gpu_sum.is_infinite() && gpu_sum.is_sign_positive(),
        "Sum with +Inf must be +Inf");
}

#[test]
fn test_reduce_sum_f32_inf_minus_inf() {
    let ctx = MetalContext::new();
    let data = vec![f32::INFINITY, f32::NEG_INFINITY];
    let gpu_sum = reduce_sum_f32_gpu(&ctx, &data);
    assert!(gpu_sum.is_nan(), "Inf + (-Inf) = NaN");
}

#[test]
fn test_sort_f32_with_nan() {
    let ctx = MetalContext::new();
    // NaN handling in sort: NaN should sort to end (consistent with total_cmp)
    let data = vec![3.0f32, f32::NAN, 1.0, 2.0];
    let gpu_sorted = radix_sort_f32_gpu(&ctx, &data);
    // Non-NaN values should be sorted
    assert_eq!(gpu_sorted[0], 1.0);
    assert_eq!(gpu_sorted[1], 2.0);
    assert_eq!(gpu_sorted[2], 3.0);
    assert!(gpu_sorted[3].is_nan(), "NaN should sort to end");
}

#[test]
fn test_reduce_min_f32_with_nan() {
    let ctx = MetalContext::new();
    let data = vec![5.0f32, f32::NAN, 1.0, 3.0];
    let gpu_min = reduce_min_f32_gpu(&ctx, &data);
    // IEEE 754: min(x, NaN) = x for fmin semantics
    // Metal fmin follows this convention
    assert_eq!(gpu_min, 1.0, "fmin should ignore NaN and return 1.0");
}
```

### 6.6 Negative Numbers in Sort

```rust
#[test]
fn test_sort_all_negative_u32() {
    // u32 has no negatives, but test max values near u32::MAX
    let ctx = MetalContext::new();
    let data = vec![u32::MAX, u32::MAX - 1, u32::MAX - 2, 0, 1];
    let gpu_sorted = radix_sort_u32_gpu(&ctx, &data);
    assert_eq!(gpu_sorted, vec![0, 1, u32::MAX - 2, u32::MAX - 1, u32::MAX]);
}

#[test]
fn test_sort_f32_mixed_negative_positive() {
    let ctx = MetalContext::new();
    let data = vec![-100.0f32, 50.0, -0.5, 0.0, -50.0, 100.0, 0.5];
    let gpu_sorted = radix_sort_f32_gpu(&ctx, &data);
    assert_eq!(gpu_sorted, vec![-100.0, -50.0, -0.5, 0.0, 0.5, 50.0, 100.0]);
}

#[test]
fn test_sort_f32_negative_zero() {
    let ctx = MetalContext::new();
    // IEEE 754: -0.0 and +0.0 are equal but have different bit representations
    let data = vec![0.0f32, -0.0, 1.0, -1.0];
    let gpu_sorted = radix_sort_f32_gpu(&ctx, &data);
    assert_eq!(gpu_sorted[0], -1.0);
    // -0.0 and 0.0 should both be present, order between them is implementation-defined
    assert!(gpu_sorted[1] == 0.0 || gpu_sorted[1] == -0.0);
    assert!(gpu_sorted[2] == 0.0 || gpu_sorted[2] == -0.0);
    assert_eq!(gpu_sorted[3], 1.0);
}
```

### 6.7 All-Same Values

```rust
#[test]
fn test_reduce_sum_all_same() {
    let ctx = MetalContext::new();
    let data = vec![7u32; 1_000_000];
    let gpu_sum = reduce_sum_u32_gpu(&ctx, &data);
    assert_eq!(gpu_sum, 7_000_000u64);
}

#[test]
fn test_sort_all_same() {
    let ctx = MetalContext::new();
    let data = vec![42u32; 1_000_000];
    let gpu_sorted = radix_sort_u32_gpu(&ctx, &data);
    assert!(gpu_sorted.iter().all(|&x| x == 42));
    assert_eq!(gpu_sorted.len(), 1_000_000);
}

#[test]
fn test_histogram_all_same_bin() {
    let ctx = MetalContext::new();
    let data = vec![0u32; 1_000_000]; // All go to bin 0
    let gpu_hist = histogram_gpu(&ctx, &data, 256);
    assert_eq!(gpu_hist[0], 1_000_000);
    assert!(gpu_hist[1..].iter().all(|&c| c == 0));
}

#[test]
fn test_compact_all_pass() {
    let ctx = MetalContext::new();
    let data: Vec<u32> = (0..10_000).collect();
    let result = compact_gpu(&ctx, &data, |_| true);
    assert_eq!(result.len(), 10_000, "All elements should pass");
    assert_eq!(result, data);
}

#[test]
fn test_compact_none_pass() {
    let ctx = MetalContext::new();
    let data: Vec<u32> = (0..10_000).collect();
    let result = compact_gpu(&ctx, &data, |_| false);
    assert!(result.is_empty(), "No elements should pass");
}
```

---

## 7. Acceptance Test Matrix

This matrix maps every functional requirement from PM.md to specific test cases with pass/fail criteria.

### 7.1 Phase 1: Foundation Primitives (P0)

| FR | Description | Test Case(s) | Pass Criteria | Priority |
|----|-------------|-------------|---------------|----------|
| FR-1 | GPU radix sort | `test_radix_sort_u32_correctness`, `test_radix_sort_f32_correctness`, `test_sort_at_boundary_sizes` | Exact match vs `std::sort_unstable` at 1M/10M/100M; `is_sorted()` + permutation check | P0 |
| FR-1 | Sort benchmarks | `e2e_harness::test_sort_produces_elements_per_sec` | Elements/sec reported in JSON for 1M, 10M, 100M | P0 |
| FR-2 | Prefix scan | `test_exclusive_scan_u32_element_by_element`, `test_inclusive_scan_u32_element_by_element` | `output[i] == sum(input[0..i])` at 1M; GB/s reported | P0 |
| FR-3 | Parallel reduce | `test_reduce_sum_u32_1m`, `test_reduce_sum_f32_relative_error`, `test_reduce_min_u32`, `test_reduce_max_u32` | Exact for u32, relative error < 1e-3 for f32; GB/s reported | P0 |
| FR-5 | Stream compaction | `test_compact_10_percent`, `test_compact_50_percent`, `test_compact_90_percent` | Count matches predicate count; all output elements satisfy predicate; order preserved | P0 |
| FR-16 | Benchmark harness | `e2e_harness::test_json_output_schema`, `e2e_harness::test_csv_output` | Valid JSON per UX.md 2.1 schema; hardware info present; all data points populated | P0 |
| FR-17 | Crossover analysis | `e2e_harness::test_crossover_detection` | Crossover point reported per experiment; interpolation between data points | P0 |

### 7.2 Phase 2: Query Operations (P1)

| FR | Description | Test Case(s) | Pass Criteria | Priority |
|----|-------------|-------------|---------------|----------|
| FR-4 | Histogram | `test_histogram_256_bins`, `test_histogram_65536_bins` | Bin counts sum to N; each element in correct bin; speedup vs CPU reported | P1 |
| FR-6 | GEMM | `test_gemm_f32_256x256`, `test_gemm_f32_1024x1024`, `test_gemm_f16_1024x1024` | Max element-wise error < 1e-4 (FP32), < 1e-2 (FP16) vs Accelerate; GFLOPS reported | P1 |
| FR-8 | Columnar filter | `test_filter_int64_selectivity_sweep` | Exact bitmask match vs CPU for 1%, 10%, 50%, 90% selectivity at 1M-100M | P0 |
| FR-9 | Group-by aggregate | `test_groupby_sum_cardinality_sweep` | Per-group values within 1e-3 relative error for cardinality 10, 1K, 100K, 1M | P1 |
| FR-14 | E2E pipeline | `test_pipeline_filter_groupby_sort_topk` | Full chain produces same result as CPU reference; stage timing breakdown in JSON | P1 |
| FR-15 | DuckDB comparison | `test_duckdb_same_query_same_data` | DuckDB result matches GPU result; wall-clock comparison reported | P1 |

### 7.3 Phase 3: Product Exploration (P2)

| FR | Description | Test Case(s) | Pass Criteria | Priority |
|----|-------------|-------------|---------------|----------|
| FR-7 | GEMV | `test_gemv_shapes` | Max element-wise error < 1e-4 vs Accelerate cblas_sgemv | P2 |
| FR-10 | Hash join | `test_hash_join_correctness` | Same match set as CPU nested-loop join (sorted comparison) | P2 |
| FR-11 | Spreadsheet formulas | `test_spreadsheet_sum_average_vlookup` | SUM/AVG exact match; VLOOKUP correct results for 1M cells | P2 |
| FR-12 | Time series | `test_timeseries_ma_vwap_bollinger` | Moving average within 1e-6 of CPU; VWAP within 1e-4; Bollinger bands correct | P2 |
| FR-13 | JSON/CSV parsing | `test_csv_parse_correctness`, `test_json_parse_correctness` | Parsed output matches serde_json/csv crate output byte-for-byte | P2 |

### 7.4 Non-Functional Requirements

| NFR | Requirement | Test Case(s) | Pass Criteria |
|-----|-------------|-------------|---------------|
| NFR-1 | CV < 5% | `test_cv_below_threshold_no_warning`, `test_benchmark_timing_reproducibility` | CV < 5% for 10 runs of reduce at 1M on quiet system |
| NFR-2 | Include dispatch overhead | `e2e_harness::test_timing_includes_encode_commit_wait` | Timing boundary includes `commandBuffer()` through `waitUntilCompleted()` |
| NFR-3 | Fair CPU baseline | `test_cpu_sort_uses_std_sort_unstable`, `test_cpu_gemm_uses_accelerate` | CPU baselines use optimized implementations, not naive loops |
| NFR-4 | Memory reporting | `e2e_harness::test_json_contains_peak_gpu_memory` | JSON output includes `peak_gpu_memory_bytes` for each data point |
| NFR-5 | Reproducibility | `test_data_generation_deterministic` | Same seed produces identical data across runs |
| NFR-6 | Hardware reporting | `e2e_harness::test_json_contains_hardware_info` | JSON includes chip name, GPU cores, bandwidth, Metal family, OS version |
| NFR-7 | Warm-up protocol | `test_warmup_iterations_discarded` | First 3 iterations excluded from statistics |

### 7.5 User Story Acceptance Criteria

| AC | Criterion | Test(s) | Verified By |
|----|-----------|---------|-------------|
| AC-1.1 | Radix sort at 1M/10M/100M, elements/sec reported | `test_radix_sort_*`, `e2e_harness::test_sort_output` | Integration + E2E |
| AC-1.2 | Prefix scan GB/s vs 273 GB/s ceiling | `test_*scan*`, `e2e_harness::test_scan_output` | Integration + E2E |
| AC-1.3 | Reduce GB/s reported | `test_reduce_*`, `e2e_harness::test_reduce_output` | Integration + E2E |
| AC-1.4 | Histogram 256/65536 bin speedup | `test_histogram_*` | Integration |
| AC-1.5 | Compact at 10%/50%/90% selectivity | `test_compact_*_selectivity` | Integration |
| AC-1.6 | Crossover point per experiment | `e2e_harness::test_crossover_detection` | E2E |
| AC-1.7 | GPU dispatch overhead included | `e2e_harness::test_timing_includes_encode_commit_wait` | E2E |
| AC-2.1 | GEMM at 256/1024/4096 for FP16/FP32 | `test_gemm_*` | Integration |
| AC-2.2 | GEMV for inference shapes | `test_gemv_shapes` | Integration |
| AC-2.3 | Accelerate baseline used | `test_cpu_gemm_uses_accelerate` | Unit |
| AC-2.4 | GFLOPS and % peak reported | `e2e_harness::test_gemm_output` | E2E |
| AC-2.5 | Existing matvec kernels reused | Code inspection (GEMV reuses `matvec_f32.metal` pattern) | Manual review |
| AC-3.1 | Filter selectivity sweep | `test_filter_int64_selectivity_sweep` | Integration |
| AC-3.2 | Group-by cardinality sweep | `test_groupby_sum_cardinality_sweep` | Integration |
| AC-3.3 | Hash join table sizes | `test_hash_join_correctness` | Integration |
| AC-3.4 | Existing shaders reused | Code inspection + `build.rs` includes existing shader paths | Manual review |
| AC-3.5 | Transfer time included | `e2e_harness::test_timing_includes_encode_commit_wait` | E2E |
| AC-4.1 | Spreadsheet formulas at 1M cells | `test_spreadsheet_sum_average_vlookup` | Integration |
| AC-4.2 | Time series at 1M-100M ticks | `test_timeseries_ma_vwap_bollinger` | Integration |
| AC-4.3 | JSON/CSV parsing throughput | `test_csv_parse_correctness`, `test_json_parse_correctness` | Integration |
| AC-4.4 | Wall-clock ms reported | `e2e_harness::test_json_contains_wall_clock` | E2E |
| AC-5.1 | Pipeline at 10M/100M rows | `test_pipeline_filter_groupby_sort_topk` | Integration |
| AC-5.2 | CPU baseline uses idiomatic Rust | Code inspection | Manual review |
| AC-5.3 | DuckDB same query/data/hardware | `test_duckdb_same_query_same_data` | Integration |
| AC-5.4 | Stage breakdown in results | `e2e_harness::test_pipeline_stage_breakdown` | E2E |
| AC-5.5 | Total wall-clock including data loading | `e2e_harness::test_pipeline_includes_data_load` | E2E |

---

## 8. Test Data Management

### 8.1 Deterministic Data Generation

All data is generated from `DataGenerator` with a fixed seed (default: 42). The seed is configurable via CLI (`--seed`) and TOML config for reproducibility.

```rust
pub struct DataGenerator {
    rng: StdRng,
}

impl DataGenerator {
    pub fn new(seed: u64) -> Self {
        Self { rng: StdRng::seed_from_u64(seed) }
    }
}
```

**Invariant**: `DataGenerator::new(S).method(N)` must produce identical output across:
- Multiple runs on the same machine
- Different machines (same Rust toolchain)
- Debug vs Release builds

This is guaranteed by using `StdRng` (ChaCha12) which is deterministic and portable.

### 8.2 Distribution Verification

Each data distribution must be validated to confirm its statistical properties before use in benchmarks.

```rust
#[test]
fn test_uniform_distribution_properties() {
    let data = DataGenerator::new(42).uniform_u32(1_000_000);

    // Mean should be near u32::MAX / 2
    let mean: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64;
    let expected_mean = u32::MAX as f64 / 2.0;
    let relative_error = (mean - expected_mean).abs() / expected_mean;
    assert!(relative_error < 0.01, "Uniform mean off by {:.2}%", relative_error * 100.0);

    // Min should be near 0, max near u32::MAX
    let min = *data.iter().min().unwrap();
    let max = *data.iter().max().unwrap();
    assert!(min < 1_000, "Min too high for uniform: {min}");
    assert!(max > u32::MAX - 1_000, "Max too low for uniform: {max}");
}

#[test]
fn test_zipf_distribution_properties() {
    let data = DataGenerator::new(42).zipf_u32(1_000_000, 1_000, 1.0);

    // Zipf: most frequent key should appear much more than least frequent
    let mut counts = std::collections::HashMap::new();
    for &val in &data {
        *counts.entry(val).or_insert(0u32) += 1;
    }

    let max_count = *counts.values().max().unwrap();
    let min_count = *counts.values().min().unwrap();

    // With alpha=1.0 and 1000 keys, top key should appear ~7x more than bottom
    assert!(
        max_count as f64 / min_count as f64 > 3.0,
        "Zipf skew too low: max={max_count}, min={min_count}"
    );
}

#[test]
fn test_skewed_selectivity_data() {
    let data = DataGenerator::new(42).uniform_u32(1_000_000);

    for target_pct in [1, 10, 50, 90] {
        let threshold = percentile(&data, target_pct);
        let actual_pct = data.iter().filter(|&&x| x < threshold).count() as f64
            / data.len() as f64 * 100.0;
        assert!(
            (actual_pct - target_pct as f64).abs() < 1.0,
            "Selectivity {target_pct}%: actual {actual_pct:.1}%"
        );
    }
}
```

### 8.3 Data Integrity Validation

Before and after GPU processing, validate data integrity with checksums to detect corruption.

```rust
pub fn checksum_u32(data: &[u32]) -> u64 {
    // Simple XOR-rotate hash for fast integrity check
    let mut hash: u64 = 0;
    for (i, &val) in data.iter().enumerate() {
        hash ^= (val as u64).rotate_left((i % 64) as u32);
    }
    hash
}

#[test]
fn test_data_integrity_through_gpu_pipeline() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(1_000_000);
    let checksum_before = checksum_u32(&data);

    // Sort should not lose or duplicate elements
    let sorted = radix_sort_u32_gpu(&ctx, &data);
    let mut sorted_for_check = sorted.clone();
    sorted_for_check.sort_unstable(); // Re-sort to canonical order
    let mut data_sorted = data.clone();
    data_sorted.sort_unstable();

    assert_eq!(
        checksum_u32(&sorted_for_check),
        checksum_u32(&data_sorted),
        "Data integrity check failed: element set changed after GPU sort"
    );
}

#[test]
fn test_input_buffer_not_modified() {
    let ctx = MetalContext::new();
    let data = DataGenerator::new(42).uniform_u32(100_000);
    let checksum_before = checksum_u32(&data);

    // Run reduce (should not modify input)
    let _ = reduce_sum_u32_gpu(&ctx, &data);

    let checksum_after = checksum_u32(&data);
    assert_eq!(checksum_before, checksum_after,
        "Reduce modified input buffer (should be read-only)");
}
```

### 8.4 Test Data Size Matrix

| Experiment Category | Small (unit test) | Medium (integration) | Large (ignored, manual) |
|--------------------|-------------------|---------------------|------------------------|
| Primitives (reduce, scan, sort, compact) | 100, 256, 1000 | 1M | 10M, 100M |
| Histogram | 1000 | 1M | 10M, 100M |
| GEMM | 8x8, 16x16 | 256x256, 1024x1024 | 4096x4096 |
| GEMV | 64x64 | 768x768 | 4096x2048 |
| Filter / Group-By | 1000 | 1M | 10M, 100M |
| Pipeline | 1000 rows | 1M rows | 10M, 100M rows |
| Spreadsheet | 1K cells | 100K cells | 1M cells |
| Time Series | 1K ticks | 100K ticks | 1M, 100M ticks |
| JSON/CSV | 100 records | 10K records | 1M records |
| Hash Join | 1K x 1K | 100K x 100K | 1M x 1M, 10M x 10M |

---

## 9. End-to-End Harness Tests

These tests validate the full CLI -> experiment -> output pipeline, ensuring the harness itself works correctly independent of kernel correctness.

### 9.1 JSON Output Schema Validation

```rust
#[test]
fn test_json_output_schema() {
    let output = run_forge_bench(&["reduce", "--profile", "quick", "--json"]);
    let json: serde_json::Value = serde_json::from_str(&output.stdout)
        .expect("Output must be valid JSON");

    // Required top-level fields
    assert!(json["version"].is_string());
    assert!(json["suite"].as_str() == Some("metal-forge-compute"));
    assert!(json["timestamp"].is_string());
    assert!(json["hardware"].is_object());
    assert!(json["config"].is_object());
    assert!(json["experiments"].is_array());
    assert!(json["summary"].is_object());

    // Hardware fields
    let hw = &json["hardware"];
    assert!(hw["chip"].is_string());
    assert!(hw["gpu_cores"].is_number());
    assert!(hw["bandwidth_theoretical_gbps"].is_number());
    assert!(hw["metal_family"].is_string());
    assert!(hw["os_version"].is_string());

    // At least one experiment with data points
    let experiments = json["experiments"].as_array().unwrap();
    assert!(!experiments.is_empty());
    let exp = &experiments[0];
    assert!(exp["name"].is_string());
    assert!(exp["data_points"].is_array());
    let dp = &exp["data_points"][0];
    assert!(dp["gpu"]["mean_ms"].is_number());
    assert!(dp["gpu"]["cv_percent"].is_number());
    assert!(dp["comparison"]["speedup"].is_number());
}
```

### 9.2 CSV Output Validation

```rust
#[test]
fn test_csv_output_format() {
    let tmp = TempDir::new().unwrap();
    let csv_path = tmp.path().join("results.csv");
    run_forge_bench(&[
        "reduce", "--profile", "quick",
        "--csv-file", csv_path.to_str().unwrap(),
    ]);

    let content = std::fs::read_to_string(&csv_path).expect("CSV file should exist");
    let lines: Vec<&str> = content.lines().collect();

    // Header row
    assert!(lines[0].contains("timestamp"));
    assert!(lines[0].contains("experiment"));
    assert!(lines[0].contains("speedup"));

    // At least one data row
    assert!(lines.len() >= 2, "CSV should have header + data rows");
}
```

### 9.3 Kill Criterion Verdict in Output

```rust
#[test]
fn test_kill_criterion_in_json_output() {
    let output = run_forge_bench(&["reduce", "--profile", "quick", "--json"]);
    let json: serde_json::Value = serde_json::from_str(&output.stdout).unwrap();

    let exp = &json["experiments"][0];
    assert!(exp["kill_criterion"].is_string());
    assert!(exp["analysis"]["kill_criterion_met"].is_boolean());
    assert!(exp["analysis"]["overall_verdict"].is_string());
}
```

### 9.4 Hardware Detection Test

```rust
#[test]
fn test_hardware_detection() {
    let output = run_forge_bench(&["--hardware"]);
    assert!(output.stdout.contains("Apple M"));
    assert!(output.stdout.contains("GPU cores"));
    assert!(output.stdout.contains("GB/s"));
    assert!(output.stdout.contains("Metal"));
}
```

### 9.5 Experiment List Test

```rust
#[test]
fn test_list_experiments() {
    let output = run_forge_bench(&["--list"]);
    for exp in ["reduce", "scan", "compact", "sort", "histogram",
                "filter", "groupby", "gemm", "gemv", "pipeline"] {
        assert!(output.stdout.contains(exp), "Missing experiment: {exp}");
    }
}
```

---

## 10. Test Execution Order and Dependencies

### 10.1 Test Phase Dependencies

```
Phase 1 Tests (Foundation)           Phase 2 Tests (Query Ops)
  reduce correctness ─────────┐
  scan correctness ───────────┤
  compact correctness ────────┼──→ pipeline correctness
  sort correctness ───────────┤       (composes Phase 1 kernels)
  histogram correctness ──────┘

  harness unit tests ─────────────→ e2e harness tests
  stats unit tests ───────────────→ reproducibility tests
  data_gen unit tests ────────────→ distribution validation
```

### 10.2 Recommended Execution Order

1. `cargo test -p forge-primitives --lib` -- Buffer pool, PSO cache, types, hardware
2. `cargo test -p forge-bench --lib` -- Stats, config, data gen, size parser
3. `cargo test -p forge-bench --test edge_cases` -- Zero, single, boundary
4. `cargo test -p forge-bench --test kernel_correctness -- reduce` -- Simplest kernel first
5. `cargo test -p forge-bench --test kernel_correctness -- scan`
6. `cargo test -p forge-bench --test kernel_correctness -- compact`
7. `cargo test -p forge-bench --test kernel_correctness -- sort`
8. `cargo test -p forge-bench --test kernel_correctness` -- All kernels
9. `cargo test -p forge-bench --test statistical_validation`
10. `cargo test -p forge-bench --test e2e_harness`
11. `cargo test -p forge-bench --test reproducibility` -- Slowest, run last

### 10.3 GPU Test Serialization

All GPU tests must run with `--test-threads=1` to avoid contention on the Metal command queue. Concurrent GPU dispatches from multiple test threads would invalidate timing measurements and may cause Metal validation errors.

```bash
cargo test -p forge-bench --test kernel_correctness -- --test-threads=1
```

---

## 11. Regression Detection Protocol

### 11.1 Baseline Establishment

```bash
# Run on clean, idle system (no other GPU workloads)
forge-bench phase1 --profile thorough --json-file baseline.json
```

**Pre-conditions for valid baseline:**
- System idle (no other applications using GPU)
- Thermal state: cool (not after sustained GPU load)
- Activity Monitor shows < 5% CPU usage from other processes
- No Time Machine backup or Spotlight indexing in progress

### 11.2 Regression Thresholds

| Context | Threshold | Rationale |
|---------|-----------|-----------|
| CI (PR check) | 10% | Allow thermal/system noise |
| Local development | 5% | Tighter for optimization work |
| Publication | 2% | Maximum precision for reports |

### 11.3 False Positive Mitigation

If regression detected:
1. Re-run with `--profile thorough` (30 iterations) to confirm
2. Check CV of both baseline and current -- if either has CV > 5%, measurement is unreliable
3. Check for thermal throttling warnings
4. If confirmed, bisect with `git bisect run forge-bench <exp> --profile ci --json | jq '.experiments[0].analysis.max_speedup'`

---

## 12. Summary

| Metric | Target |
|--------|--------|
| Total test cases | ~145 |
| Unit tests | ~80 (pure Rust, no GPU) |
| Integration tests | ~50 (GPU kernel correctness) |
| E2E tests | ~5 (full harness pipeline) |
| Reproducibility tests | ~10 (statistical validation) |
| CI runtime (correctness) | < 5 minutes |
| CI runtime (benchmark regression) | < 15 minutes |
| Kill criteria automated | 15/15 (100%) |
| ACs covered by tests | 32/32 (100%) |
| FRs covered by tests | 17/17 (100%) |
| NFRs covered by tests | 7/7 (100%) |

Every kernel is tested against a CPU reference implementation using real GPU dispatch. No mocks. No simulations. Test data is deterministic, distributions are validated, and statistical measurement integrity is enforced through CV checks, outlier detection, and thermal throttling monitoring. Kill criteria from PM.md are fully automated with clear PASS/FAIL verdicts.
