# QA Strategy: forge-sort Test Suite

**Date**: 2026-02-20
**Analyst**: QA Manager (Claude Opus 4.6)
**System**: forge-sort (Rust + objc2-metal bindings)
**Platform**: macOS Apple Silicon M4 Pro (273 GB/s BW, 20 GPU cores, 48 MB SLC, Metal 3.2)
**Scope**: Complete quality strategy for the forge-sort radix sort library test suite, covering u32 keys-only sorting at scales from 0 to 128M elements

---

## 1. Test Quality Criteria

A test is valuable when it satisfies all of the following:

### 1.1 What Makes a Test Valuable

| Criterion | Definition | Anti-pattern |
|-----------|-----------|-------------|
| **Targeted** | Tests exactly one failure mode of one kernel phase | Testing "sort works" without isolating which byte position, tile boundary, or kernel phase is exercised |
| **Deterministic** | Same input always produces the same pass/fail verdict | Using `thread_rng()` without a seed -- failures cannot be reproduced |
| **Independent** | Does not depend on execution order or state from other tests | Sharing a `GpuSorter` instance across tests without reinitializing |
| **Fast** | Completes in under 2 seconds for CI tests, under 10 seconds for stress tests | Sorting 128M elements in a test that runs on every PR |
| **Diagnostic** | On failure, prints the seed, size, first-diff index, and which verification check failed | `assert_eq!(actual, expected)` on a 16M-element vector with no context |
| **Non-circular** | Verification does not re-implement the code under test | Comparing GPU output against another GPU sort instead of an independent CPU reference |
| **Adversarial** | Designed to trigger known failure modes, not just confirm the happy path | Only testing random uniform data, which exercises the average case |

### 1.2 Diagnostic Failure Output Standard

Every test must produce failure output containing:

```
FAIL: test_msd_single_bucket_16k
  Size: 16384
  Seed: 0xDEADBEEF
  Sorted: false (first unsorted pair at index 4091: 0xFF00ABCD > 0xFF00ABCC)
  Permutation: true (XOR match, SUM match)
  Histogram: SKIPPED (sorted check failed first)
  First 10 mismatches: [4091, 4092, 4095, 4096, ...]
```

This standard is enforced by routing all correctness tests through a single `sort_and_verify` harness function.

---

## 2. Verification Methods

The test suite employs four distinct verification methods. Each has different cost/precision tradeoffs. Tests use the appropriate combination depending on their category.

### 2.1 Method A: Compare Against std Sort (Reference Comparison)

```rust
fn verify_vs_std(input: &[u32], gpu_output: &[u32]) -> bool {
    let mut expected = input.to_vec();
    expected.sort();
    gpu_output == expected.as_slice()
}
```

**Precision**: Exact. Catches every possible error including dropped/duplicated elements, wrong order, and data corruption.
**Cost**: O(n log n) for the CPU sort. At 16M elements, `sort()` takes ~160ms. At 128M, ~1.5 seconds.
**When to use**: All correctness tests at sizes up to 16M. The definitive oracle.
**Limitation**: Circular only if the CPU sort itself is wrong (vanishingly unlikely for `std::sort`). At 64M+ the CPU sort dominates test runtime, so we use Method B+C as a fast pre-check and only run Method A on failure.

### 2.2 Method B: Is-Sorted Check

```rust
fn is_sorted(data: &[u32]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}
```

**Precision**: Catches ordering errors. Does NOT catch dropped elements, duplicated elements, or data corruption that happens to produce a sorted sequence.
**Cost**: O(n), single pass, no allocation.
**When to use**: As a fast pre-check in performance tests and large-scale tests. Never as the sole verification.

### 2.3 Method C: Permutation Check (Histogram Preservation)

```rust
fn verify_permutation(input: &[u32], output: &[u32]) -> bool {
    if input.len() != output.len() { return false; }
    // Fast pre-check: XOR and SUM checksums
    let in_xor = input.iter().fold(0u32, |a, &v| a ^ v);
    let out_xor = output.iter().fold(0u32, |a, &v| a ^ v);
    let in_sum: u64 = input.iter().map(|&v| v as u64).sum();
    let out_sum: u64 = output.iter().map(|&v| v as u64).sum();
    if in_xor != out_xor || in_sum != out_sum { return false; }
    // Full histogram check (HashMap-based)
    let mut hist = std::collections::HashMap::new();
    for &v in input { *hist.entry(v).or_insert(0i64) += 1; }
    for &v in output { *hist.entry(v).or_insert(0i64) -= 1; }
    hist.values().all(|&c| c == 0)
}
```

**Precision**: Catches dropped/duplicated elements. The XOR+SUM pre-check is O(n) with near-zero false-negative rate. The full HashMap check is definitive but O(n) with allocation.
**Cost**: XOR+SUM: O(n), zero allocation. Full histogram: O(n), HashMap allocation.
**When to use**: Combined with Method B. Method B + C together prove the output is a sorted permutation of the input, which is the full correctness contract.
**Why not just Method A**: At 64M+ elements, Method A requires 1.5 seconds of CPU sort. Method B+C together take ~100ms. For large-scale tests, B+C is the primary verification with Method A reserved for failure diagnosis.

### 2.4 Method D: Reference Comparison Against Known Output

For deterministic inputs with known correct outputs (e.g., `[3, 1, 2]` must produce `[1, 2, 3]`), hardcode the expected output.

**Precision**: Exact for that specific input.
**Cost**: O(n) comparison, zero CPU sort cost.
**When to use**: Small edge case tests (empty, single, two elements, hardcoded bit patterns). Also used to validate that the verification methods themselves are correct.

### 2.5 Verification Assignment Per Category

| Test Category | Primary Method | Secondary Method | Rationale |
|--------------|---------------|-----------------|-----------|
| Edge cases (n <= 100) | D (hardcoded) | A (std sort) | Small enough for exact comparison, hardcoded for documentation value |
| Correctness (n <= 16M) | A (std sort) | C (permutation) | Full oracle, permutation as independent cross-check |
| Correctness (n > 16M) | B + C | A on failure only | CPU sort too slow for primary path |
| Tile boundary | A (std sort) | -- | Exact sizes matter; full oracle required |
| Bit patterns | A (std sort) | C (permutation) | Adversarial inputs; need full oracle |
| Determinism | Equality across runs | -- | Not about correctness of a single sort but consistency across multiple |
| Performance | B (is-sorted) | -- | Speed matters; correctness is validated by other tests |
| Buffer reuse | A (std sort) | -- | Testing that old data does not leak; need full comparison |
| Stress | A (every Nth) + B (every) | C on failure | Balance thoroughness with runtime |

---

## 3. Coverage Matrix: Tests to Algorithm Internals

The forge-sort algorithm has 4 GPU kernels executed in a single command buffer with a single compute encoder. Each kernel has distinct failure modes that require targeted tests.

### 3.1 Kernel 1: sort_msd_histogram

**Purpose**: Reads all input data, computes 256-bin histogram for MSD byte (bits 24:31).
**Internal structure**: Per-SG atomic histogram on TG memory, cross-SG reduction, global atomic accumulation.

| Failure Mode | Triggered By | Test That Covers It |
|-------------|-------------|-------------------|
| Partial tile: `idx < n` check wrong | n not divisible by TILE_SIZE | `test_tile_4095`, `test_tile_4097`, `test_prime_4099` |
| Invalid elements counted (padding) | Last tile has fewer than 4096 valid elements | `test_tile_4096_plus_1`, `test_100_elements` |
| Per-SG atomic contention | All elements have same MSD byte | `test_msd_single_bucket_1m`, `test_all_same_1m` |
| Cross-SG reduction wrong | 8 SGs must sum correctly | `test_uniform_256_msd_buckets` |
| Global atomic overflow | Large n: counts exceed 16M per bucket | `test_all_same_64m` (ignored) |
| Zero histogram from empty | n = 0 | `test_empty` |

### 3.2 Kernel 2: sort_msd_prep

**Purpose**: Serial prefix sum (thread 0) + parallel bucket descriptor writes.
**Internal structure**: Thread 0 computes exclusive prefix of 256 bins. All 256 threads write counters and BucketDesc.

| Failure Mode | Triggered By | Test That Covers It |
|-------------|-------------|-------------------|
| Prefix sum overflow at large n | Total element count approaches u32::MAX | `test_64m_random` (ignored) |
| Bucket count = 0 for empty bins | Only a few MSD bins populated | `test_msd_two_buckets`, `test_msd_single_bucket_1m` |
| tile_count computation wrong | Bucket with count not divisible by TILE_SIZE | `test_16k_random` (MSD buckets have non-aligned counts) |
| BucketDesc.offset wrong | Prefix sum error | `test_uniform_256_msd_buckets` (all 256 buckets populated) |

### 3.3 Kernel 3: sort_msd_atomic_scatter

**Purpose**: Scatter elements from buf_a to buf_b based on MSD byte, using atomic_fetch_add for positioning.
**Internal structure**: Load -> per-SG histogram -> tile histogram + cross-SG prefix -> global atomic fetch-add -> per-SG ranking + scatter.

| Failure Mode | Triggered By | Test That Covers It |
|-------------|-------------|-------------------|
| Atomic scatter race condition | High contention: single MSD bucket | `test_msd_single_bucket_1m`, `test_all_zeros_1m` |
| Wrong scatter position: sg_prefix error | Multiple SGs contribute to same bucket | `test_1m_random` (statistical coverage) |
| Partial tile: invalid elements scattered | Last tile partial | `test_tile_4097`, `test_prime_8191` |
| dst[gp] write out of bounds | tile_base + sg_prefix + within_sg exceeds bucket allocation | `test_uniform_256_msd_buckets` (tight bucket boundaries) |
| Invalid keys in padding (0xFFFFFFFF) leak | Padding elements with `mv[e]=false` should not scatter | `test_tile_4095`, `test_100_elements` |
| Elements dropped | Atomic rank collision (should not happen with relaxed atomics on Apple Silicon, but could on other hardware) | `test_permutation_1m` (histogram check) |

### 3.4 Kernel 4: sort_inner_fused

**Purpose**: 3-pass LSD sort per MSD bucket. Self-computes histograms for all 3 inner passes during pass 0 scan.
**Internal structure**: 256 TGs process 256 MSD buckets. Each TG runs 3 scatter passes with buffer ping-pong (b->a, a->b, b->a). Final result in buf_a.

| Failure Mode | Triggered By | Test That Covers It |
|-------------|-------------|-------------------|
| Empty bucket: `desc.count == 0` early-exit | MSD byte not present in input | `test_msd_two_buckets` (254 empty buckets) |
| Multi-tile inner sort: `tile_count > 1` | MSD bucket with > 4096 elements | `test_1m_random`, `test_16m_random` |
| Single-tile inner sort | MSD bucket with <= 4096 elements | `test_4096_random`, `test_100_elements` |
| Buffer ping-pong direction wrong | 3 passes: b->a, a->b, b->a. If one is reversed, data is read from stale buffer | `test_reverse_sorted_1m` (sequential bit patterns exercise all passes) |
| hist_p1/hist_p2 accumulation wrong | Self-computed histograms for passes 1 and 2 | `test_only_lsb_varies`, `test_only_middle_byte_varies` |
| Cross-tile run_pfx accumulation wrong | Bucket spans multiple tiles | `test_msd_single_bucket_16k` (single bucket gets 16K = 4 tiles) |
| `threadgroup_barrier(mem_flags::mem_device)` insufficient | Next pass reads stale data from device memory | `test_16m_random` (large buckets stress device memory coherence) |
| Global prefix sum (256-bin, 8-chunk) wrong | Inner pass histogram does not sum to bucket count | `test_bit_24_boundary` |

### 3.5 Cross-Kernel Interactions

| Failure Mode | Triggered By | Test That Covers It |
|-------------|-------------|-------------------|
| MSD scatter output not aligned with BucketDesc offsets | Mismatch between kernel 2 (prep) and kernel 3 (scatter) | `test_uniform_256_msd_buckets` |
| Inner sort reads wrong region of buf_b | BucketDesc.offset from kernel 2 does not match actual scatter positions from kernel 3 | `test_1m_random`, `test_4m_random` |
| Implicit barrier between dispatches insufficient | Single encoder sequential dispatches should have implicit barriers (Metal spec) | `test_16m_random` (large data exercises memory subsystem) |
| Final result in wrong buffer | 3 inner passes end in buf_a. If the count changes, result could be in buf_b | `test_all_same_1m` (trivial inner sort), `test_16m_random` (full inner sort) |

---

## 4. Acceptance Criteria Per Test Category

### 4.1 Correctness Tests

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| All tests pass | 100% (zero failures) | `cargo test --release` |
| Verification method | Method A (std sort comparison) for all sizes up to 16M; Method B+C for 32M+ | Enforced by test harness |
| Permutation check passes independently | XOR, SUM, and full histogram all match | `verify_permutation()` |
| First-diff index reported on failure | Failure message includes index of first mismatch | Enforced by `sort_and_verify()` helper |

### 4.2 Tile Boundary Tests

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Every size in the boundary set passes | 100% | 9 specific sizes tested (see section 5) |
| Both Method A and Method C pass | Sorted AND permutation of input | Dual verification |
| No test takes longer than 1 second | Performance constraint | Sizes are all <= 1M elements |

### 4.3 Adversarial Bit Pattern Tests

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| All 10 bit patterns pass at 1M | 100% | Method A verification |
| At least 5 bit patterns pass at 16M | 100% | Method A verification |
| Single-MSD-bucket test passes at 16K and 1M | 100% | Critical path for kernel 3 atomic contention |

### 4.4 Determinism Tests

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| 10 runs of same input produce identical output | 10/10 identical | Byte-for-byte comparison |
| Tested at 3 different sizes (1K, 1M, 4M) | All pass | |
| Tested with 3 different distributions (random, sorted, all-same) | All pass | |

### 4.5 Scale and Stress Tests

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| 32M and 64M pass Method B+C | Sorted permutation | |
| 128M passes Method B+C | Sorted permutation | `#[ignore]` test |
| Buffer reuse: large-then-small produces correct result | No data leakage | Method A |
| 50 sequential sorts all correct | 50/50 | Method A on each |

### 4.6 Performance Tests (behind `perf` feature flag)

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| **Hard floor (16M random)** | > 2000 Mk/s (p50) | 5 warmup + 20 timed runs |
| **Hard floor (1M random)** | > 1500 Mk/s (p50) | 5 warmup + 20 timed runs |
| **Relative regression** | p50 does not drop > 15% below saved baseline | Baseline file comparison |
| **Variance** | p95/p5 spread < 20% | If exceeded: warning, not failure (thermal) |
| **Cold start** | First sort < 5x median of subsequent sorts | Separate measurement |
| **Scaling linearity** | 16M time < 6x of 4M time | Sub-linear scaling check |

### 4.7 Error Handling Tests

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Empty input returns Ok, no dispatch | No crash, no GPU work | |
| Single element returns Ok, unchanged | `[42]` -> `[42]` | Method D |
| `GpuSorter::new()` succeeds | Returns `Ok(GpuSorter)` | |
| Error variant display strings are correct | Match expected messages | String comparison |

### 4.8 Concurrency Tests

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Two GpuSorter instances sort correctly in sequence | Both outputs correct | Method A on both |
| 100 sequential sorts on one GpuSorter | All correct | Method A on every 10th, Method B on all |

---

## 5. Complete Test Inventory

### 5.1 Correctness Tests (18 tests)

| # | Test Name | Size | Distribution | Kernel Focus |
|---|-----------|------|-------------|-------------|
| C01 | `test_1k_random` | 1,000 | Uniform random | Sub-tile, all kernels |
| C02 | `test_4k_random` | 4,000 | Uniform random | ~1 tile, all kernels |
| C03 | `test_16k_random` | 16,000 | Uniform random | ~4 tiles, inner multi-tile |
| C04 | `test_64k_random` | 64,000 | Uniform random | ~16 tiles |
| C05 | `test_256k_random` | 256,000 | Uniform random | ~63 tiles |
| C06 | `test_1m_random` | 1,000,000 | Uniform random | ~244 tiles, SLC-resident |
| C07 | `test_4m_random` | 4,000,000 | Uniform random | ~977 tiles, near SLC boundary |
| C08 | `test_16m_random` | 16,000,000 | Uniform random | ~3907 tiles, exceeds SLC |
| C09 | `test_all_zeros_1m` | 1,000,000 | All zeros | Single MSD bucket, max atomic contention |
| C10 | `test_all_same_1m` | 1,000,000 | All 0xDEADBEEF | Single MSD bucket, specific bin |
| C11 | `test_pre_sorted_1m` | 1,000,000 | 0..1M ascending | Sequential bytes across all passes |
| C12 | `test_reverse_sorted_1m` | 1,000,000 | 1M..0 descending | Maximum displacement per element |
| C13 | `test_two_values_1m` | 1,000,000 | Alternating 0/u32::MAX | 2 MSD buckets, extreme skew |
| C14 | `test_few_unique_4_values_1m` | 1,000,000 | Random from {0, 1, 2, 3} | 4 active bins in LSB pass only |
| C15 | `test_few_unique_256_values_1m` | 1,000,000 | Random from 0..256 | Exactly 1 value per MSD bin (all in bin 0), 256 inner bins |
| C16 | `test_near_duplicates_1m` | 1,000,000 | Pairs of (v, v+1) | Adjacent values stress ranking |
| C17 | `test_32m_random` | 32,000,000 | Uniform random | 2x current max; Method B+C |
| C18 | `test_64m_random` | 64,000,000 | Uniform random | Max normal size; Method B+C |

### 5.2 Tile Boundary Tests (9 tests)

| # | Test Name | Size | Rationale |
|---|-----------|------|-----------|
| B01 | `test_tile_exact_4096` | 4,096 | Exactly 1 full tile; no partial tile |
| B02 | `test_tile_4097` | 4,097 | 1 valid element in second tile |
| B03 | `test_tile_4095` | 4,095 | 1 invalid element in first tile (255 threads valid at position 15, last thread invalid) |
| B04 | `test_tile_8192` | 8,192 | Exactly 2 full tiles |
| B05 | `test_tile_plus_threads_4352` | 4,352 | TILE_SIZE + THREADS_PER_TG = partial tile of exactly 256 elements |
| B06 | `test_tile_plus_one_thread_4097_alt` | 4,097 | Same as B02 but with adversarial bit pattern (all 0xFF MSD) |
| B07 | `test_prime_4099` | 4,099 | Prime, no alignment to any power-of-two |
| B08 | `test_prime_8191` | 8,191 | Prime, just under 2 tiles |
| B09 | `test_prime_16381` | 16,381 | Prime, just under 4 tiles |

### 5.3 Adversarial Bit Pattern Tests (10 tests)

| # | Test Name | Size | Pattern | Kernel Targeted |
|---|-----------|------|---------|----------------|
| A01 | `test_msd_single_bucket_1m` | 1,000,000 | All `0xAA_XX_XX_XX` (random lower 24 bits) | K3 scatter: 1 bucket gets everything |
| A02 | `test_msd_two_buckets_1m` | 1,000,000 | Alternating `0x00_*` and `0xFF_*` | K3 scatter: extreme 2-bucket skew |
| A03 | `test_uniform_256_msd_buckets_1m` | 1,000,000 | `(i % 256) << 24 \| random` | K2 prep: all bucket descriptors populated |
| A04 | `test_identical_inner_bytes_1m` | 1,000,000 | `XX_00_00_00` (random MSD, zero lower 24) | K4 inner: passes 0-2 are single-bin |
| A05 | `test_all_bits_set_1m` | 1,000,000 | `0xFFFFFFFF` repeated | All kernels: bin 255 at every pass |
| A06 | `test_bit_24_boundary_1m` | 1,000,000 | Mix of `0x00FFFFFF` and `0x01000000` | K3: adjacent MSD bins; K4: max inner spread |
| A07 | `test_only_lsb_varies_1m` | 1,000,000 | `0x42_42_42_XX` (random lowest byte) | K4 pass 0: real work; passes 1-2 single-bin |
| A08 | `test_only_middle_byte_varies_1m` | 1,000,000 | `0x42_XX_42_42` (random byte 2) | K4 pass 2: real work; passes 0,1 single-bin |
| A09 | `test_power_of_two_values_1m` | 1,000,000 | Random choice from {1, 2, 4, 8, ..., 2^31} | Sparse bit patterns; most bytes are 0 |
| A10 | `test_sequential_high_bytes_1m` | 1,000,000 | `(i % 256) << 24` (sequential MSD, zero inner) | K3: uniform scatter; K4: trivial inner |

### 5.4 Determinism Tests (3 tests)

| # | Test Name | Size | Runs | Distribution |
|---|-----------|------|------|-------------|
| D01 | `test_determinism_1k_10runs` | 1,000 | 10 | Seeded random |
| D02 | `test_determinism_1m_10runs` | 1,000,000 | 10 | Seeded random |
| D03 | `test_determinism_4m_10runs` | 4,000,000 | 10 | Seeded random |

### 5.5 Buffer Management Tests (6 tests)

| # | Test Name | Scenario | What It Verifies |
|---|-----------|----------|-----------------|
| R01 | `test_reuse_large_then_small` | Sort 16M, then sort 100 | Old data does not leak from oversized buffer |
| R02 | `test_reuse_small_then_large` | Sort 100, then sort 16M | Reallocation works correctly |
| R03 | `test_reuse_same_size_different_data` | Sort 1M of X, then 1M of Y | Previous results do not affect new sort |
| R04 | `test_reuse_50_sorts` | 50 sequential sorts, different sizes and data | No memory accumulation or corruption |
| R05 | `test_reuse_growing_sequence` | Sort 1K, 2K, 4K, 8K, ..., 1M | Progressive reallocation chain |
| R06 | `test_reuse_shrinking_sequence` | Sort 1M, 512K, 256K, ..., 1K | Buffer oversized but correct |

### 5.6 Edge Case Tests (8 tests)

| # | Test Name | Input | Expected Output |
|---|-----------|-------|----------------|
| E01 | `test_empty` | `[]` | `[]`, Ok returned |
| E02 | `test_single_element` | `[42]` | `[42]` |
| E03 | `test_two_sorted` | `[1, 2]` | `[1, 2]` |
| E04 | `test_two_reversed` | `[2, 1]` | `[1, 2]` |
| E05 | `test_all_max_u32` | `[0xFFFFFFFF; 10000]` | All equal, sorted |
| E06 | `test_all_min_u32` | `[0; 10000]` | All zero, sorted |
| E07 | `test_max_and_min_mixed` | `[MAX, 0, MAX, 0, ...]` | `[0, 0, ..., MAX, MAX]` |
| E08 | `test_three_elements` | `[3, 1, 2]` | `[1, 2, 3]` |

### 5.7 Performance Tests (5 tests, behind `perf` feature flag)

| # | Test Name | Size | Metric | Hard Floor |
|---|-----------|------|--------|-----------|
| P01 | `test_perf_1m` | 1,000,000 | Mk/s at p50 | > 1500 Mk/s |
| P02 | `test_perf_4m` | 4,000,000 | Mk/s at p50 | > 2000 Mk/s |
| P03 | `test_perf_16m` | 16,000,000 | Mk/s at p50 | > 2000 Mk/s |
| P04 | `test_perf_scaling_linearity` | 4M vs 16M | Time ratio | 16M < 6x of 4M |
| P05 | `test_perf_cold_start` | 16,000,000 | First vs median | First < 5x median |

### 5.8 Concurrency Tests (2 tests)

| # | Test Name | Scenario |
|---|-----------|----------|
| N01 | `test_two_sorters_sequential` | Create 2 GpuSorter instances, sort alternately, verify both correct |
| N02 | `test_rapid_sequential_100` | One GpuSorter, 100 sorts in tight loop, verify every 10th |

### 5.9 Large Scale Tests (2 tests, `#[ignore]`)

| # | Test Name | Size | Method |
|---|-----------|------|--------|
| L01 | `test_128m_random` | 128,000,000 | Method B+C (sorted + permutation) |
| L02 | `test_128m_all_zeros` | 128,000,000 | Method B (sorted; permutation trivially true) |

### 5.10 Test Count Summary

| Category | Count | Priority |
|----------|-------|----------|
| Correctness | 18 | P0 |
| Tile boundary | 9 | P0 |
| Adversarial bit patterns | 10 | P0 |
| Determinism | 3 | P0 |
| Buffer management | 6 | P1 |
| Edge cases | 8 | P0 |
| Performance | 5 | P2 (feature-gated) |
| Concurrency | 2 | P2 |
| Large scale (#[ignore]) | 2 | P2 |
| **Total** | **63** | |

Current test count: 21 (18 integration + 3 unit). Target: 63 integration + 3 existing unit = **66 total tests**.

---

## 6. Test Data Generation Strategy

### 6.1 Deterministic Seeds

All random test data uses seeded RNGs from `rand::rngs::StdRng`. Non-deterministic `thread_rng()` is prohibited.

| Context | Seed | Rationale |
|---------|------|-----------|
| Correctness tests | `0xDEAD_BEEF` | Fixed across all correctness tests for consistency |
| Determinism tests | `0xCAFE_BABE` | Must be reproducible to verify 10 runs match |
| Buffer reuse tests | `iteration_index` | Each iteration uses its own seed for variety while staying reproducible |
| Performance tests | `0xBEEF_CAFE` | Fixed seed so benchmarks are comparable across runs |
| Tile boundary tests | `0xFACE_FEED` | Fixed |

### 6.2 Generator Helper

```rust
fn generate_data(n: usize, seed: u64) -> Vec<u32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen()).collect()
}

fn generate_patterned(n: usize, f: impl Fn(usize) -> u32) -> Vec<u32> {
    (0..n).map(f).collect()
}
```

### 6.3 Failure Seed Reporting

When any test fails, the failure message includes the seed used to generate the input. This allows exact reproduction:

```
FAIL: test_1m_random (seed=0xDEADBEEF, n=1000000)
  Reproduce: cargo test test_1m_random -- --nocapture
```

For tests that sweep parameters (like determinism tests), the seed and iteration index are both reported.

---

## 7. Regression Test Strategy

When a bug is found and fixed, the following protocol ensures it does not recur.

### 7.1 Bug-to-Test Protocol

1. **Reproduce**: Write the minimal input that triggers the bug. Record the exact size, seed, and distribution.
2. **Classify**: Identify which kernel phase and failure mode the bug belongs to (use section 3 coverage matrix).
3. **Add a named regression test**: Name format `test_regression_ISSUE_SHORT_DESC` (e.g., `test_regression_42_partial_tile_scatter_oob`).
4. **Place in regression test file**: All regression tests go in `tests/regressions.rs`, separate from the main correctness suite, so they are never accidentally pruned.
5. **Document**: Each regression test has a comment with:
   - The bug description
   - The date it was found
   - The commit that fixed it
   - Which kernel/phase was affected
6. **Verify the test fails before the fix**: Check out the pre-fix commit and run the regression test to confirm it fails. This validates the test is actually testing the bug.

### 7.2 Regression Test Template

```rust
/// Regression test for issue #42: partial tile in MSD scatter wrote
/// one element beyond the bucket boundary when the last tile had
/// exactly 1 valid element and that element mapped to the last bin.
/// Found: 2026-02-20, Fixed: commit abc123
/// Kernel: sort_msd_atomic_scatter (phase 4, scatter)
#[test]
fn test_regression_42_partial_tile_last_bin() {
    let mut data: Vec<u32> = vec![0xFF000001; 4096]; // Full first tile
    data.push(0xFF000002); // Single element in second tile, same MSD bin
    sort_and_verify_data(data);
}
```

### 7.3 Regression Test Coverage Rule

No bug fix may be merged without an accompanying regression test. The regression test must fail on the unfixed code and pass on the fixed code. This is verified by the reviewer, not by CI (since CI only runs against the current code).

---

## 8. Pass/Fail Criteria for the Overall Suite

### 8.1 Merge Gate (PR CI)

The full test suite has a three-tier verdict:

| Verdict | Condition | Action |
|---------|-----------|--------|
| **PASS** | All P0 tests pass. All P1 tests pass. Perf tests pass (if `perf` feature enabled). No `#[ignore]` tests failed. | Merge allowed. |
| **WARN** | All P0 tests pass. One or more P1 tests fail. Perf variance > 20%. | Merge allowed with reviewer acknowledgement. Warning logged. |
| **FAIL** | Any P0 test fails. OR any test panics/crashes. OR any test hangs > 30 seconds. | Merge blocked. |

### 8.2 Release Gate

Before any crate publish (`cargo publish`):

| Requirement | Check |
|-------------|-------|
| All 63 integration tests pass | `cargo test --release` |
| All 3 unit tests pass | `cargo test --release --lib` |
| `#[ignore]` tests pass (128M) | `cargo test --release -- --ignored` |
| Performance above hard floors | `cargo test --release --features perf` |
| Metal validation clean | `MTL_SHADER_VALIDATION=1 cargo test --release` |
| Zero `clippy` warnings | `cargo clippy -- -D warnings` |

### 8.3 Single Test Failure Impact

| Test Category | Single Failure Impact |
|--------------|---------------------|
| Correctness (C01-C18) | Blocks merge. Data corruption risk. |
| Tile boundary (B01-B09) | Blocks merge. Off-by-one is the #1 GPU kernel bug class. |
| Adversarial bit pattern (A01-A10) | Blocks merge. Exercises degenerate paths. |
| Determinism (D01-D03) | Blocks merge. Non-determinism indicates race condition. |
| Buffer management (R01-R06) | Blocks merge. Data leakage risk. |
| Edge cases (E01-E08) | Blocks merge. API contract violation. |
| Performance (P01-P05) | Warning only unless below hard floor. |
| Concurrency (N01-N02) | Warning only. |
| Large scale (L01-L02) | Warning only (these are `#[ignore]`). |

---

## 9. Risk-Based Test Prioritization (Execution Order)

Tests are ordered by the probability and severity of the bugs they detect. When running the full suite, this ordering ensures the most critical failures are surfaced first, minimizing developer wait time.

### 9.1 Execution Order

```
Phase 1: Fast critical checks (~2 seconds)
  E01 test_empty
  E02 test_single_element
  E03 test_two_sorted
  E04 test_two_reversed
  E08 test_three_elements

Phase 2: Tile boundary precision (~5 seconds)
  B03 test_tile_4095
  B01 test_tile_exact_4096
  B02 test_tile_4097
  B04 test_tile_8192
  B05 test_tile_plus_threads_4352
  B06 test_tile_plus_one_thread_4097_alt
  B07 test_prime_4099
  B08 test_prime_8191
  B09 test_prime_16381

Phase 3: Adversarial bit patterns (~10 seconds)
  A05 test_all_bits_set_1m
  A01 test_msd_single_bucket_1m
  A02 test_msd_two_buckets_1m
  A04 test_identical_inner_bytes_1m
  A07 test_only_lsb_varies_1m
  A08 test_only_middle_byte_varies_1m
  A03 test_uniform_256_msd_buckets_1m
  A06 test_bit_24_boundary_1m
  A09 test_power_of_two_values_1m
  A10 test_sequential_high_bytes_1m

Phase 4: Core correctness at scale (~15 seconds)
  C01-C08 (random at increasing sizes)
  C09-C16 (distributions at 1M)

Phase 5: Determinism (~10 seconds)
  D01-D03

Phase 6: Buffer management (~5 seconds)
  R01-R06

Phase 7: Remaining edge cases (~2 seconds)
  E05-E07

Phase 8: Scale tests (~10 seconds)
  C17 test_32m_random
  C18 test_64m_random

Phase 9: Concurrency (~3 seconds)
  N01-N02

Phase 10: Performance (feature-gated, ~30 seconds)
  P01-P05

Phase 11: Large scale (ignored, ~20 seconds)
  L01-L02
```

### 9.2 Rationale

- **Edge cases first**: If `test_empty` crashes, nothing else matters. These take < 100ms total.
- **Tile boundaries second**: Off-by-one in `idx < n` or `num_tiles` computation is the highest-probability GPU kernel bug. Finding it early saves debugging time on downstream failures.
- **Bit patterns third**: These exercise degenerate paths (empty buckets, single-bucket overflow, all-same values) that random data rarely triggers. High probability of finding real bugs.
- **Scale last**: A bug that only manifests at 32M+ is rare compared to a bug at any size. Scale tests are expensive (CPU reference sort at 32M takes ~500ms).
- **Performance last**: Performance tests are the slowest and least likely to indicate a correctness bug. They are feature-gated so they do not slow down the default test run.

---

## 10. Flakiness Management

GPU tests are inherently more prone to flakiness than CPU-only tests due to non-deterministic thread scheduling, thermal throttling, and shared hardware resources. The following policies manage this.

### 10.1 Sources of Flakiness

| Source | Symptom | Frequency | Severity |
|--------|---------|-----------|----------|
| **Thermal throttle** | Performance test fails (below hard floor) even though correctness is fine | Common under sustained load | Low (correctness unaffected) |
| **GPU scheduling jitter** | Timing variance > 20% between runs | Occasional | Low |
| **Atomic ordering non-determinism** | Different output on different runs for same input | Should be impossible if algorithm is correct | Critical (indicates a real bug) |
| **OS preemption** | Wall-clock time spike on a single run | Occasional | Low |
| **Metal driver bug** | Random crash or incorrect output | Rare | Critical |

### 10.2 Flakiness Policy

| Rule | Rationale |
|------|-----------|
| **Correctness tests are NEVER flaky** | If a correctness test fails even once, it is a bug until proven otherwise. Inputs are deterministic. Outputs must be deterministic. A "flaky" correctness test means the algorithm has a race condition. |
| **Performance tests allow 1 retry** | If a performance test fails, retry once after a 5-second cooldown. If it fails again, report as a true failure. The cooldown allows thermal recovery. |
| **Performance tests use GPU timestamps, not wall-clock** | GPU timestamps (`MTLCommandBuffer.GPUStartTime`/`GPUEndTime`) are immune to OS scheduling jitter. Wall-clock is used only for overall sanity (e.g., "sort completed in < 50ms"). |
| **Performance hard floors have 30% headroom** | The hard floor of 2000 Mk/s at 16M is 30% below the typical 2859 Mk/s. This absorbs thermal throttle without false-failing. |
| **`#[ignore]` tests are never gating** | The 128M tests are expensive and may fail due to memory pressure on CI runners. They run in nightly or pre-release pipelines, not on every PR. |
| **No quarantine** | Tests are either reliable (and must pass) or feature-gated (and opt-in). No test is permanently quarantined. A quarantined test is either fixed or deleted. |

### 10.3 Thermal Throttle Detection

Performance tests include a thermal throttle detector:

```rust
fn check_thermal_throttle(times_ms: &[f64]) -> bool {
    let p5 = percentile(times_ms, 5.0);
    let p95 = percentile(times_ms, 95.0);
    let spread = (p95 - p5) / p5;
    if spread > 0.30 {
        eprintln!("WARNING: Thermal throttle suspected. Spread={:.0}% (p5={:.2}ms, p95={:.2}ms)",
            spread * 100.0, p5, p95);
        eprintln!("         Consider: close other GPU apps, wait 60s for cooldown, re-run.");
        true
    } else {
        false
    }
}
```

When thermal throttle is detected:
- Performance test logs a warning but does NOT automatically fail.
- The test evaluates against the p5 time (best case, least-throttled) instead of p50.
- If even p5 is below the hard floor, the test fails (genuine regression).

### 10.4 Determinism Enforcement

To verify that the GPU sort is truly deterministic (not just "usually correct"):

```rust
#[test]
fn test_determinism_1m_10runs() {
    let mut sorter = GpuSorter::new().unwrap();
    let input = generate_data(1_000_000, 0xCAFE_BABE);
    let mut first_output: Option<Vec<u32>> = None;

    for run in 0..10 {
        let mut data = input.clone();
        sorter.sort_u32(&mut data).unwrap();

        match &first_output {
            None => first_output = Some(data),
            Some(expected) => {
                assert_eq!(
                    &data, expected,
                    "Non-deterministic output on run {}. First diff at index {}",
                    run,
                    data.iter().zip(expected.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(data.len())
                );
            }
        }
    }
}
```

If this test ever fails, it means the atomics or scatter ranking has a race condition. This is treated as a P0 bug regardless of how rare it appears.

---

## 11. Mutation Testing Applicability

### 11.1 Assessment

[cargo-mutants](https://mutants.rs/) is a mutation testing tool for Rust that injects faults (e.g., replacing `+` with `-`, changing `<` to `<=`, removing function bodies) and checks whether existing tests catch them.

**Applicability to forge-sort**:

| Component | Mutation Testing Value | Rationale |
|-----------|----------------------|-----------|
| **lib.rs (Rust host code)** | HIGH | 263 lines of Rust with buffer management, bounds computation (`n.div_ceil(TILE_SIZE)`), error handling, and memcpy logic. cargo-mutants can mutate these and verify tests catch the mutations. |
| **sort.metal (GPU shader)** | NONE | cargo-mutants cannot mutate Metal shader code. The shader is compiled by `xcrun metal` in build.rs, outside Rust's compilation model. |
| **Verification helpers** | MEDIUM | Mutating the test helpers themselves can validate that the helpers are correct. If a mutation to `verify_permutation()` is not caught, it means the helper could mask bugs. |

### 11.2 Recommended cargo-mutants Configuration

```bash
# Run mutation testing on the host-side Rust code only
cargo mutants --package forge-sort --timeout 120
```

Expected mutant categories in `lib.rs`:

| Mutation | What It Tests |
|----------|--------------|
| Replace `n <= 1` with `n <= 0` in `sort_u32()` | `test_single_element` must catch this |
| Replace `n.div_ceil(TILE_SIZE)` with `n / TILE_SIZE` | Tile boundary tests must catch this |
| Remove `std::ptr::write_bytes` (histogram zeroing) | Correctness tests must catch this |
| Replace `buf_a` with `buf_b` in final readback | All correctness tests must catch this |
| Remove `cmd.waitUntilCompleted()` | Test reads uninitialized data; correctness tests catch |
| Replace `data_bytes > self.data_buf_capacity` with `>=` | Buffer reuse tests must catch this |

### 11.3 Mutation Score Target

**Target: > 90% mutation kill rate** on `lib.rs` mutants. The 10% allowance is for mutations in error paths that are difficult to trigger in tests (e.g., `MTLCommandBuffer` creation failure on functional hardware).

### 11.4 GPU-Specific Mutation Testing Limitation

Research from [Zhu et al. (ICST 2020)](https://azaidman.github.io/publications/zhuICST2020.pdf) found that GPU-specific mutation operators (e.g., changing `threadgroup_barrier` to a no-op, swapping `memory_order_relaxed` with `memory_order_seq_cst`, changing buffer indices) are more effective at finding GPU bugs than conventional operators. However, no tool currently supports Metal Shading Language mutations.

**Workaround**: Manual mutation testing of the shader. The developer introduces one fault at a time into `sort.metal`, rebuilds, and verifies at least one test fails. Priority mutations:

| Manual Mutation | Expected Failing Test |
|----------------|---------------------|
| Remove `threadgroup_barrier(mem_flags::mem_threadgroup)` in K1 | Any correctness test > 256 elements |
| Remove `threadgroup_barrier(mem_flags::mem_device)` in K4 | `test_16m_random` (cross-tile coherence) |
| Change `shift` from 24 to 16 in K1 | Any test with varied MSD bytes |
| Change `0xFFu` to `0xFu` in digit extraction | Any correctness test |
| Replace `atomic_fetch_add` with `atomic_load` in K3 scatter | All correctness tests (all elements scatter to position 0) |
| Change `pass * 8u` to `pass * 4u` in K4 shift | Any correctness test |
| Swap `buf_b`/`buf_a` in K4 pass direction | Any correctness test |

---

## 12. Test Harness Implementation

### 12.1 Common Verification Harness

All correctness tests route through a single harness function. This ensures consistent verification, diagnostic output, and failure reporting.

```rust
fn sort_and_verify_data(data: Vec<u32>) {
    let n = data.len();
    let mut expected = data.clone();
    expected.sort();

    let mut actual = data.clone();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u32(&mut actual).unwrap();

    // Check 1: sorted order
    let sorted = actual.windows(2).all(|w| w[0] <= w[1]);

    // Check 2: permutation (XOR + SUM fast check)
    let in_xor = data.iter().fold(0u32, |a, &v| a ^ v);
    let out_xor = actual.iter().fold(0u32, |a, &v| a ^ v);
    let in_sum: u64 = data.iter().map(|&v| v as u64).sum();
    let out_sum: u64 = actual.iter().map(|&v| v as u64).sum();
    let perm = in_xor == out_xor && in_sum == out_sum;

    // Check 3: exact match against CPU reference
    let exact = actual == expected;

    if !sorted || !perm || !exact {
        let first_diff = actual.iter().zip(expected.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(n);
        panic!(
            "Sort verification failed at n={}\n\
             Sorted: {}\n\
             Permutation (XOR+SUM): {}\n\
             Exact match vs std::sort: {}\n\
             First diff index: {}\n\
             actual[{}]={:#010X} vs expected[{}]={:#010X}",
            n, sorted, perm, exact,
            first_diff,
            first_diff, actual.get(first_diff).copied().unwrap_or(0),
            first_diff, expected.get(first_diff).copied().unwrap_or(0),
        );
    }
}
```

### 12.2 Large-Scale Verification Harness

For 32M+ elements where CPU reference sort is expensive:

```rust
fn sort_and_verify_large(data: Vec<u32>) {
    let n = data.len();
    let mut actual = data.clone();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u32(&mut actual).unwrap();

    // Check 1: sorted
    let sorted = actual.windows(2).all(|w| w[0] <= w[1]);
    assert!(sorted, "Large-scale sort not sorted at n={}", n);

    // Check 2: permutation via XOR + SUM
    let in_xor = data.iter().fold(0u32, |a, &v| a ^ v);
    let out_xor = actual.iter().fold(0u32, |a, &v| a ^ v);
    let in_sum: u64 = data.iter().map(|&v| v as u64).sum();
    let out_sum: u64 = actual.iter().map(|&v| v as u64).sum();
    assert_eq!(in_xor, out_xor, "XOR checksum mismatch at n={} (elements dropped/duplicated)", n);
    assert_eq!(in_sum, out_sum, "SUM checksum mismatch at n={} (elements dropped/duplicated)", n);
}
```

### 12.3 Seeded Data Generation

```rust
fn generate_seeded(n: usize, seed: u64) -> Vec<u32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen()).collect()
}
```

---

## 13. CI Pipeline Integration

### 13.1 Per-PR Gate (~30 seconds)

```bash
# All correctness, boundary, bit-pattern, determinism, buffer, edge case tests
cargo test --release --test correctness
```

This runs all 63 non-ignored, non-feature-gated tests. Expected runtime: ~30 seconds on M4 Pro. The bottleneck is CPU reference sorting at 64M (~800ms).

### 13.2 Nightly Extended (~2 minutes)

```bash
# Standard suite + ignored large-scale tests
cargo test --release --test correctness -- --include-ignored

# Performance tests (if feature flag wired up)
cargo test --release --test correctness --features perf

# Metal validation layer (catches GPU memory errors)
MTL_SHADER_VALIDATION=1 cargo test --release --test correctness
```

### 13.3 Pre-Release (~5 minutes)

```bash
# Everything from nightly, plus:
cargo test --release --workspace
cargo clippy -- -D warnings

# Mutation testing on host code
cargo mutants --package forge-sort --timeout 120
```

---

## 14. Research Sources

### GPU Sort Verification
- [AMD GPUOpen: Boosting GPU Radix Sort](https://gpuopen.com/learn/boosting_gpu_radix_sort/) -- Distribution testing methodology for radix sort
- [Performance Evaluation of GPU-based Parallel Sorting Algorithms (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12867261/) -- Comprehensive evaluation framework for GPU sorts
- [Wikipedia: Radix Sort](https://en.wikipedia.org/wiki/Radix_sort) -- Correctness proof by induction, stability requirements
- [Verification of Counting Sort and Radix Sort (Heerink 2018)](http://jurriaan.creativecode.org/wp-content/uploads/2018/10/chapter19.pdf) -- Formal correctness proof: permutation property + sorted property

### Permutation and Histogram Preservation
- [Generic Approach to Verification of the Permutation Property of Sorting Algorithms (Springer 2020)](https://link.springer.com/chapter/10.1007/978-3-030-63461-2_14) -- Formal verification of sort permutation invariant
- [How to argue the correctness of radix sort (Quora)](https://www.quora.com/How-can-I-argue-the-correctness-of-radix-sort) -- Inductive correctness argument

### Mutation Testing
- [cargo-mutants](https://mutants.rs/) -- Rust mutation testing tool
- [Massively Parallel, Highly Efficient, but What About the Test Suite Quality? (ICST 2020)](https://azaidman.github.io/publications/zhuICST2020.pdf) -- GPU-specific mutation operators, test adequacy for GPU code
- [Automated Test Generation for OpenCL Kernels Using Fuzzing (ACM 2020)](https://dl.acm.org/doi/10.1145/3366428.3380768) -- Mutation-based fuzzing of GPU kernel inputs

### GPU Test Reliability
- [Strategies for Handling Flaky Tests (RWX)](https://www.rwx.com/blog/strategies-for-handling-flaky-tests) -- Quarantine, retry, and triage strategies
- [Automated testing for GPU kernels (Edinburgh 2020)](https://era.ed.ac.uk/items/16b53370-37ac-4f6d-91ce-0d52ef071805) -- CLTestCheck: coverage metrics for GPU kernels
- [Cornell: Verifying GPU Kernels by Test Amplification](https://www.cs.cornell.edu/~lerner/papers/verifying_gpu_kernels_by_test_amplification.pdf) -- Test adequacy beyond functional correctness

### Metal-Specific
- [Apple WWDC: Optimize Metal Performance](https://developer.apple.com/videos/play/wwdc2020/10632/) -- Metal validation layer, GPU profiling
- [NVIDIA CUB DeviceRadixSort](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh) -- Reference API and correctness contract
