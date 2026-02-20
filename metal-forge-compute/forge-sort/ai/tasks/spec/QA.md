# QA Strategy: forge-sort v2 Multi-Type Support

**Date**: 2026-02-20
**Analyst**: QA Manager (Claude Opus 4.6)
**System**: forge-sort v2 (Rust + objc2-metal, Apple Silicon Metal 3.2)
**Platform**: macOS Apple Silicon M4 Pro (273 GB/s BW, 20 GPU cores, 48 MB SLC, Metal 3.2)
**Scope**: Comprehensive test strategy for adding i32, f32, f64, argsort, and key-value pair sorting to forge-sort GPU radix sort library

---

## Executive Summary

This document defines the comprehensive test strategy for forge-sort v2, which extends the GPU radix sort library from u32-only to support **i32, f32, u64, i64, f64, argsort, and key-value pair sorting** on Apple Silicon Metal 3.2. The strategy covers 14 test categories, ~155 new test cases across 3 phases, 7 verification methods, and performance acceptance criteria per type.

The central testing challenge is that each new type introduces a **transformation layer** (bit flips) between the user's domain and the sort kernel. Bugs in this layer are silent -- data sorts "correctly" as unsigned integers but maps back to wrong values in the original type's ordering. The test strategy therefore emphasizes **oracle-based verification** (comparing against Rust's stdlib sort with `total_cmp`) and **edge case saturation** (every IEEE 754 special value, every signed integer boundary) over statistical sampling.

### Test Budget Summary

| Phase | Types | New Tests | Cumulative |
|-------|-------|-----------|------------|
| Phase 1 (i32 + f32) | sort_i32, sort_f32, sort_i32_buffer, sort_f32_buffer | ~50 | ~50 |
| Phase 2 (argsort + key-value) | argsort_*, sort_pairs_* | ~50 | ~100 |
| Phase 3 (u64 + i64 + f64) | sort_u64, sort_i64, sort_f64, buffer/argsort/pairs variants | ~50 | ~150 |
| Regression | v1 u32 sort unchanged | ~5 | ~155 |

Combined with the existing v1 test suite (~63 tests per OVERVIEW.md), the total reaches **~220 test functions**.

---

## 1. Test Categories and Priorities

### 1.1 Category Hierarchy

| Priority | Category | Risk | Description | Phase |
|----------|----------|------|-------------|-------|
| **P0** | Bit transformation correctness | Critical | FloatFlip/IFloatFlip and i32 XOR produce correct sort order for ALL representable values | 1, 3 |
| **P0** | IEEE 754 edge cases | Critical | NaN, Inf, -Inf, -0.0, +0.0, denormals, max/min normals sort to correct positions | 1, 3 |
| **P0** | Signed integer boundaries | Critical | i32::MIN, i32::MAX, -1, 0, 1 sort correctly; two's complement edge cases | 1 |
| **P0** | Argsort permutation validity | Critical | Result is a valid permutation of 0..n; applying it yields sorted order | 2 |
| **P0** | Key-value integrity | Critical | Every (key, value) pair in output corresponds to exactly one pair in input; no duplication or loss | 2 |
| **P1** | Sort stability | High | Equal keys preserve original relative order (radix sort guarantee); critical for key-value and multi-column sort | 2 |
| **P1** | Cross-type consistency | High | sort_u32 results are bit-identical to v1; new types do not regress existing paths | 1 |
| **P1** | SortBuffer\<T\> type safety | High | Generic SortBuffer works correctly for each T; sealed trait prevents unsupported types at compile time | 1 |
| **P0** | Unsigned 64-bit boundaries | Critical | u64::MIN (0), u64::MAX, large values sort correctly; baseline 64-bit pipeline validation | 3 |
| **P0** | Signed 64-bit boundaries | Critical | i64::MIN, i64::MAX, -1, 0, 1 sort correctly; 64-bit sign bit XOR transformation | 3 |
| **P1** | f64 multi-pass correctness | High | 8 radix passes (1 MSD + 7 inner in 3 fused groups) produce correct output; buffer ping-pong tracking is correct | 3 |
| **P2** | Error handling | Medium | LengthMismatch error for sort_pairs; empty/single-element edge cases for all new methods | 2 |
| **P2** | Performance thresholds | Medium | Each type meets throughput floor at 16M elements | 1, 2, 3 |
| **P2** | Buffer management | Medium | Lazy allocation of value/index buffers; grow-only strategy; no stale data leakage | 2 |

### 1.2 Category-to-Phase Mapping

```
Phase 1 (i32 + f32):
  - Bit transformation: i32 XOR, f32 FloatFlip/IFloatFlip
  - IEEE 754 edge cases: f32 full matrix
  - Signed integer boundaries: i32 full matrix
  - Cross-type regression: v1 u32 unchanged
  - SortBuffer<T>: generic buffer for i32, f32
  - Performance: i32, f32 throughput floors

Phase 2 (argsort + key-value):
  - Argsort permutation validity: u32, i32, f32
  - Key-value integrity: u32, i32, f32
  - Sort stability: duplicate keys + values
  - Error handling: LengthMismatch, empty/single
  - Buffer management: lazy value buffer allocation

Phase 3 (u64 + i64 + f64):
  - u64 boundaries: u64::MIN, u64::MAX, large values, random distributions
  - i64 boundaries: i64::MIN, i64::MAX, -1, 0, 1, sign boundary, random distributions
  - i64 transform: XOR sign bit 0x8000000000000000 (self-inverse)
  - Bit transformation: f64 FloatFlip/IFloatFlip
  - IEEE 754 edge cases: f64 full matrix
  - f64 multi-pass: 8-pass pipeline correctness
  - argsort_{u64,i64,f64}: permutation validity
  - sort_pairs_{u64,i64}: key-value integrity with 64-bit keys
  - Performance: u64, i64, f64 throughput floors
```

---

## 2. Verification Methods

### 2.1 Method Definitions

| Method | Name | Implementation | Cost | When to Use |
|--------|------|---------------|------|-------------|
| **A** | Oracle comparison | Sort same data with Rust stdlib, assert bit-exact equality | O(n log n) | All correctness tests at n <= 16M |
| **B** | Sorted-order check | `windows(2).all(\|w\| w[0] <= w[1])` for the type's natural ordering | O(n) | Fast check for perf tests, scale tests, supplementary |
| **C** | Permutation check | XOR + SUM checksums, then full HashMap histogram | O(n) | All tests -- catches data loss/duplication |
| **D** | Hardcoded expected | Small inputs with known correct output | O(1) | Edge cases, documentation examples |
| **E** | Permutation validity | For argsort: verify result contains each index 0..n exactly once | O(n) | All argsort tests |
| **F** | Gather verification | For argsort: `data[indices[i]] <= data[indices[i+1]]` for all i | O(n) | All argsort tests (combined with E) |
| **G** | Pair integrity | For sort_pairs: multiset of (key, value) pairs is preserved | O(n) | All key-value tests |

### 2.2 Oracle Functions per Type

The oracle is the ground truth that GPU sort results are compared against. Each type uses its natural Rust ordering:

```rust
// Oracle for i32
fn oracle_i32(data: &[i32]) -> Vec<i32> {
    let mut expected = data.to_vec();
    expected.sort();  // i32 implements Ord naturally
    expected
}

// Oracle for f32 (total_cmp ordering -- matches FloatFlip behavior)
fn oracle_f32(data: &[f32]) -> Vec<f32> {
    let mut expected = data.to_vec();
    expected.sort_by(f32::total_cmp);
    expected
}

// Oracle for f64 (total_cmp ordering)
fn oracle_f64(data: &[f64]) -> Vec<f64> {
    let mut expected = data.to_vec();
    expected.sort_by(f64::total_cmp);
    expected
}

// Oracle for argsort (any type -- returns stable sorted indices)
fn oracle_argsort_f32(data: &[f32]) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..data.len() as u32).collect();
    indices.sort_by(|&a, &b| data[a as usize].total_cmp(&data[b as usize]));
    indices
}
```

### 2.3 Verification Assignments by Category

| Category | Methods Used | Rationale |
|----------|-------------|-----------|
| Bit transformation correctness | A + C | Oracle catches wrong ordering; permutation catches data loss from transform bugs |
| IEEE 754 edge cases | A + D | Oracle for full correctness; hardcoded for specific value positions |
| Signed integer boundaries | A + D | Same rationale as IEEE 754 |
| Argsort permutation validity | E + F + A | E verifies valid permutation; F verifies sorted order; A compares full index sequence against oracle |
| Key-value integrity | G + A (keys) | G checks pair preservation; A verifies key ordering |
| Sort stability | Custom | Assign unique sequence numbers to equal keys; verify sequence order preserved |
| Cross-type regression | A + C | Existing u32 tests unchanged and still pass |
| Performance | B only | Sorted check sufficient; oracle too slow for repeated benchmarks |

### 2.4 Float Comparison Strategy

**f32/f64 equality**: Because FloatFlip/IFloatFlip are bijections that preserve bit patterns, GPU sort output should be **bit-exact** with the oracle. We compare using `to_bits()` equality, NOT `==` (which treats NaN != NaN and -0.0 == +0.0):

```rust
fn assert_f32_sorted_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(), e.to_bits(),
            "Mismatch at index {}: actual={} (bits={:#010x}), expected={} (bits={:#010x})",
            i, a, a.to_bits(), e, e.to_bits()
        );
    }
}
```

This catches the two critical float pitfalls:
1. **`-0.0` vs `+0.0`**: `(-0.0f32 == 0.0f32)` is `true`, but `(-0.0f32).to_bits() != (0.0f32).to_bits()`. We must distinguish them.
2. **`NaN`**: `(f32::NAN == f32::NAN)` is `false`. Bit comparison catches NaN placement correctly.

---

## 3. IEEE 754 Edge Case Test Matrix

### 3.1 f32 Special Values (19 canonical values)

Every test that claims "f32 correctness" must handle these values. The matrix shows the bit representation, the FloatFlip result, expected sort position, and risk level.

| # | Value | Bit Pattern | FloatFlip Result | Expected Position | Risk |
|---|-------|-------------|-----------------|-------------------|------|
| 1 | Negative quiet NaN (non-canonical) | `0xFFC00001` | `0x003FFFFE` | First (smallest) | High |
| 2 | Negative quiet NaN (canonical) | `0xFFC00000` | `0x003FFFFF` | After #1 | High |
| 3 | Negative signaling NaN | `0xFF800001` | `0x007FFFFE` | After negative quiet NaN | Medium |
| 4 | `-Inf` | `0xFF800000` | `0x007FFFFF` | After all negative NaN | High |
| 5 | `-f32::MAX` (-3.4028235e38) | `0xFF7FFFFF` | `0x00800000` | After -Inf | Medium |
| 6 | `-1.0` | `0xBF800000` | `0x407FFFFF` | Middle negative range | Low |
| 7 | `-f32::MIN_POSITIVE` (-1.1754944e-38) | `0x80800000` | `0x7F7FFFFF` | Smallest negative normal | Medium |
| 8 | Negative denormal boundary | `0x807FFFFF` | `0x7F800000` | Boundary negative normal/denormal | **Critical** |
| 9 | Negative denormal (smallest magnitude) | `0x80000001` | `0x7FFFFFFE` | Largest negative denormal | High |
| 10 | `-0.0` | `0x80000000` | `0x7FFFFFFF` | Just before +0.0 | **Critical** |
| 11 | `+0.0` | `0x00000000` | `0x80000000` | Just after -0.0 | **Critical** |
| 12 | Positive denormal (smallest) | `0x00000001` | `0x80000001` | First positive denormal | High |
| 13 | `f32::MIN_POSITIVE` (1.1754944e-38) | `0x00800000` | `0x80800000` | Boundary positive denormal/normal | Medium |
| 14 | `1.0` | `0x3F800000` | `0xBF800000` | Middle positive range | Low |
| 15 | `f32::MAX` (3.4028235e38) | `0x7F7FFFFF` | `0xFF7FFFFF` | Largest finite positive | Medium |
| 16 | `+Inf` | `0x7F800000` | `0xFF800000` | After all finite positives | High |
| 17 | Positive signaling NaN | `0x7F800001` | `0xFF800001` | After +Inf | Medium |
| 18 | Positive quiet NaN (canonical) | `0x7FC00000` | `0xFFC00000` | After signaling NaN | High |
| 19 | Positive quiet NaN (non-canonical) | `0x7FC00001` | `0xFFC00001` | Last (largest) | High |

### 3.2 f32 Composite Test Vectors

These test vectors combine special values to catch specific transformation bugs:

| Test Name | Input | Expected Order | What It Catches |
|-----------|-------|---------------|-----------------|
| `f32_zeros_ordering` | `[+0.0, -0.0, +0.0, -0.0]` | `[-0.0, -0.0, +0.0, +0.0]` | FloatFlip distinguishes sign-bit of zero |
| `f32_inf_and_nan` | `[NaN, Inf, -Inf, -NaN, 0.0]` | `[-NaN, -Inf, 0.0, Inf, NaN]` | NaN sorts to extremes, not middle |
| `f32_denormal_boundary` | `[-MIN_POS, -5e-45, -0.0, +0.0, 5e-45, MIN_POS]` | exact order | Normal-denormal boundary is continuous |
| `f32_all_nan` | `[NaN; 1000]` (mixed payloads) | all NaN, sorted by bit pattern | Transform handles all-NaN degenerate input |
| `f32_nan_among_normals` | `[NaN, 1.0, NaN, 2.0]` | `[1.0, 2.0, NaN, NaN]` | NaN does not corrupt normal values |
| `f32_negative_special` | `[-Inf, -MAX, -1.0, -MIN_POS, -5e-45, -0.0]` | exact input order (already sorted) | Full negative range is monotonic after FloatFlip |
| `f32_positive_special` | `[+0.0, 5e-45, MIN_POS, 1.0, MAX, Inf]` | exact input order (already sorted) | Full positive range is monotonic after FloatFlip |
| `f32_mixed_comprehensive` | All 19 special values from matrix + 81 random | Oracle comparison | Complete coverage in one test |

### 3.3 f64 Special Values

f64 uses the same FloatFlip algorithm with 64-bit width. The test matrix mirrors f32 with f64 equivalents:

| Value | f64 Constant | Risk |
|-------|-------------|------|
| Negative quiet NaN | `f64::from_bits(0xFFF8_0000_0000_0001)` | High |
| Negative signaling NaN | `f64::from_bits(0xFFF0_0000_0000_0001)` | Medium |
| `-Inf` | `f64::NEG_INFINITY` | High |
| `-f64::MAX` | `-f64::MAX` | Medium |
| `-1.0` | `-1.0f64` | Low |
| `-f64::MIN_POSITIVE` | `-f64::MIN_POSITIVE` | Medium |
| Negative denormal | `f64::from_bits(0x8000_0000_0000_0001)` | High |
| `-0.0` | `-0.0f64` | **Critical** |
| `+0.0` | `0.0f64` | **Critical** |
| Positive denormal | `f64::from_bits(0x0000_0000_0000_0001)` | High |
| `f64::MIN_POSITIVE` | `f64::MIN_POSITIVE` | Medium |
| `1.0` | `1.0f64` | Low |
| `f64::MAX` | `f64::MAX` | Medium |
| `+Inf` | `f64::INFINITY` | High |
| Positive signaling NaN | `f64::from_bits(0x7FF0_0000_0000_0001)` | Medium |
| Positive quiet NaN | `f64::NAN` | High |

**f64-specific additional tests**:
- **8-pass boundary**: All elements identical except byte 0 (tests that inner pass 0 alone correctly sorts the LSB)
- **Cross-pass consistency**: Elements that differ only in bytes 4-7 (tests MSD and first inner fused group) vs elements that differ only in bytes 0-3 (tests second and third inner fused groups)
- **Register pressure**: 16M random f64 (tests that 64-bit registers do not spill under full load)

---

## 4. Signed Integer (i32) Test Cases

### 4.1 Boundary Value Tests

| Test Name | Input | Expected Output | What It Catches |
|-----------|-------|----------------|-----------------|
| `i32_min_max` | `[i32::MAX, i32::MIN, 0, -1, 1]` | `[i32::MIN, -1, 0, 1, i32::MAX]` | XOR 0x80000000 maps MIN/MAX correctly |
| `i32_around_zero` | `[-1, 0, 1]` | `[-1, 0, 1]` | Sign bit transition at zero boundary |
| `i32_two_complement_edge` | `[i32::MIN, i32::MIN + 1, i32::MAX - 1, i32::MAX]` | same order (already sorted) | Adjacent values at extremes |
| `i32_all_negative` | `[-100, -1, -50, -1000, -2]` | `[-1000, -100, -50, -2, -1]` | Negative-only sort preserves magnitude ordering |
| `i32_negative_boundary` | `[-1, -2, -128, -129, -256, -257]` | sorted ascending | Values that cross byte boundaries in XOR'd representation |
| `i32_sign_boundary_dense` | `[-2, -1, 0, 1, 2]` | `[-2, -1, 0, 1, 2]` | Dense coverage around sign transition |
| `i32_powers_of_two` | `[1, 2, 4, ..., 2^30, -1, -2, -4, ..., -2^31]` | all negatives ascending, then all positives ascending | Power-of-two boundaries in XOR'd space |

### 4.2 Distribution Tests

| Test Name | N | Distribution | Verification |
|-----------|---|-------------|-------------|
| `i32_random_1m` | 1M | `rng.gen::<i32>()` (full range) | Oracle (A) + Permutation (C) |
| `i32_random_16m` | 16M | `rng.gen::<i32>()` (full range) | Oracle (A) |
| `i32_all_same_negative` | 1M | `[-42; 1M]` | All elements unchanged |
| `i32_pre_sorted` | 1M | `-500_000..500_000` | Oracle (A) |
| `i32_reverse_sorted` | 1M | `(500_000..-500_000).rev()` | Oracle (A) |
| `i32_narrow_range` | 1M | `rng.gen_range(-10..10)` | Oracle (A) -- many duplicates, tests stability |
| `i32_half_negative` | 1M | 500K in `[-1M, 0)` + 500K in `[0, 1M)` | Oracle (A) -- tests sign boundary distribution |

### 4.3 SortBuffer\<i32\> Tests

| Test Name | Description | Verification |
|-----------|-------------|-------------|
| `i32_buffer_basic` | Copy 1M random i32 to SortBuffer\<i32\>, sort, verify | Oracle (A) |
| `i32_buffer_direct_write` | Write i32 values via as_mut_slice(), sort, verify | Hardcoded (D) |
| `i32_buffer_special_values` | Buffer containing MIN, MAX, 0, -1, 1 | Hardcoded (D) |
| `i32_buffer_reuse` | Sort 5 different datasets through same buffer | Oracle (A) each time |

### 4.4 i32 Transform Roundtrip Tests

| Test Name | Description |
|-----------|-------------|
| `i32_transform_roundtrip` | Apply XOR 0x80000000 twice to all boundary values; verify identity |
| `i32_transform_monotonicity` | For pairs where `a < b` as i32: verify `(a ^ 0x80000000) < (b ^ 0x80000000)` as u32 |

---

## 4.5 Unsigned 64-bit (u64) Test Cases

### 4.5.1 Boundary Value Tests

| Test Name | Input | Expected Output | What It Catches |
|-----------|-------|----------------|-----------------|
| `u64_min_max` | `[u64::MAX, 0, 1, u64::MAX - 1]` | `[0, 1, u64::MAX - 1, u64::MAX]` | 64-bit pipeline handles full range |
| `u64_large_values` | `[u64::MAX, u64::MAX/2, u64::MAX/4, 0]` | `[0, u64::MAX/4, u64::MAX/2, u64::MAX]` | High bits sort correctly across 8 passes |
| `u64_byte_boundaries` | `[0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFF]` | sorted ascending | Values crossing byte boundaries in 8-pass pipeline |
| `u64_powers_of_two` | `[1, 2, 4, ..., 2^63]` | sorted ascending | Each radix pass handles power-of-two transitions |
| `u64_all_same` | `[42u64; 1M]` | All elements unchanged | Degenerate: all buckets empty except one |

### 4.5.2 Distribution Tests

| Test Name | N | Distribution | Verification |
|-----------|---|-------------|-------------|
| `u64_random_1m` | 1M | `rng.gen::<u64>()` (full range) | Oracle (A) + Permutation (C) |
| `u64_random_16m` | 16M | `rng.gen::<u64>()` (full range) | Oracle (A) |
| `u64_pre_sorted` | 1M | `0..1_000_000u64` | Oracle (A) |
| `u64_reverse_sorted` | 1M | `(0..1_000_000u64).rev()` | Oracle (A) |
| `u64_narrow_range` | 1M | `rng.gen_range(0..1000u64)` | Oracle (A) -- many duplicates |
| `u64_high_bits_only` | 1M | Values with only bits 56-63 set | Tests MSD pass in isolation |
| `u64_low_bits_only` | 1M | Values with only bits 0-7 set | Tests final inner pass in isolation |

### 4.5.3 SortBuffer\<u64\> and Argsort Tests

| Test Name | Description | Verification |
|-----------|-------------|-------------|
| `u64_buffer_basic` | Copy 1M random u64 to SortBuffer\<u64\>, sort, verify | Oracle (A) |
| `u64_buffer_reuse` | Sort 5 different u64 datasets through same buffer | Oracle (A) each time |
| `argsort_u64_basic` | `[30, 10, 20]` as u64 | D: indices = `[1, 2, 0]` |
| `argsort_u64_1m` | 1M random u64 | E + F + A |
| `sort_pairs_u64_basic` | `keys=[30,10,20], vals=[100,200,300]` as u64/u32 | D: keys=[10,20,30], vals=[200,300,100] |
| `sort_pairs_u64_1m` | 1M random u64 keys, sequential u32 values | G + A(keys) |

---

## 4.6 Signed 64-bit (i64) Test Cases

### 4.6.1 Boundary Value Tests

| Test Name | Input | Expected Output | What It Catches |
|-----------|-------|----------------|-----------------|
| `i64_min_max` | `[i64::MAX, i64::MIN, 0, -1, 1]` | `[i64::MIN, -1, 0, 1, i64::MAX]` | XOR 0x8000000000000000 maps MIN/MAX correctly for 64-bit |
| `i64_around_zero` | `[-1i64, 0, 1]` | `[-1, 0, 1]` | Sign bit transition at zero boundary (64-bit) |
| `i64_two_complement_edge` | `[i64::MIN, i64::MIN + 1, i64::MAX - 1, i64::MAX]` | same order (already sorted) | Adjacent values at 64-bit extremes |
| `i64_all_negative` | `[-100, -1, -50, -1000, -2]` as i64 | `[-1000, -100, -50, -2, -1]` | Negative-only sort preserves magnitude ordering (64-bit) |
| `i64_sign_boundary_dense` | `[-2i64, -1, 0, 1, 2]` | `[-2, -1, 0, 1, 2]` | Dense coverage around sign transition (64-bit) |
| `i64_large_magnitude` | `[i64::MIN/2, i64::MAX/2, -1_000_000_000_000, 1_000_000_000_000]` | sorted ascending | Large magnitudes beyond i32 range |
| `i64_byte_boundaries` | `[-0x100, -0xFF, -1, 0, 0xFF, 0x100]` as i64 | sorted ascending | Values crossing byte boundaries in XOR'd 64-bit representation |

### 4.6.2 Distribution Tests

| Test Name | N | Distribution | Verification |
|-----------|---|-------------|-------------|
| `i64_random_1m` | 1M | `rng.gen::<i64>()` (full range) | Oracle (A) + Permutation (C) |
| `i64_random_16m` | 16M | `rng.gen::<i64>()` (full range) | Oracle (A) |
| `i64_all_same_negative` | 1M | `[-42i64; 1M]` | All elements unchanged |
| `i64_pre_sorted` | 1M | `-500_000i64..500_000` | Oracle (A) |
| `i64_reverse_sorted` | 1M | `(500_000i64..-500_000).rev()` | Oracle (A) |
| `i64_narrow_range` | 1M | `rng.gen_range(-10i64..10)` | Oracle (A) -- many duplicates, tests stability |
| `i64_half_negative` | 1M | 500K in `[-1M, 0)` + 500K in `[0, 1M)` as i64 | Oracle (A) -- tests sign boundary distribution |

### 4.6.3 SortBuffer\<i64\>, Argsort, and sort_pairs Tests

| Test Name | Description | Verification |
|-----------|-------------|-------------|
| `i64_buffer_basic` | Copy 1M random i64 to SortBuffer\<i64\>, sort, verify | Oracle (A) |
| `i64_buffer_special_values` | Buffer containing MIN, MAX, 0, -1, 1 | Hardcoded (D) |
| `i64_buffer_reuse` | Sort 5 different i64 datasets through same buffer | Oracle (A) each time |
| `argsort_i64_basic` | `[5, -3, 0, -1, 8]` as i64 | D: indices = `[1, 3, 2, 0, 4]` |
| `argsort_i64_1m` | 1M random i64 | E + F + A |
| `sort_pairs_i64_basic` | `keys=[5,-3,0,-1], vals=[a,b,c,d]` as i64/u32 | D: keys=[-3,-1,0,5], vals=[b,d,c,a] |
| `sort_pairs_i64_1m` | 1M random i64 keys, sequential u32 values | G + A(keys) |

### 4.6.4 i64 Transform Roundtrip Tests

| Test Name | Description |
|-----------|-------------|
| `i64_transform_roundtrip` | Apply XOR 0x8000000000000000 twice to all boundary values; verify identity |
| `i64_transform_monotonicity` | For pairs where `a < b` as i64: verify `(a ^ 0x8000000000000000) < (b ^ 0x8000000000000000)` as u64 |

---

## 5. Floating-Point (f32) Test Cases

### 5.1 Edge Case Tests

Every test from Section 3.2 (f32 Composite Test Vectors) becomes a named test function. Additionally:

| Test Name | Input | Expected | Verification |
|-----------|-------|----------|-------------|
| `f32_total_cmp_semantics` | `[-0.0, +0.0, -Inf, Inf, NaN, -NaN, 1.0, -1.0]` | `[-NaN, -Inf, -1.0, -0.0, +0.0, 1.0, Inf, NaN]` | Hardcoded (D) + bit-exact comparison |
| `f32_all_special_values` | All 19 values from IEEE 754 matrix (Section 3.1) | Sorted by FloatFlip order | Hardcoded (D) |
| `f32_nan_payloads` | 10 distinct NaN bit patterns (both signs) | Sorted by bit pattern after FloatFlip | Oracle (A) with `sort_by(f32::total_cmp)` |
| `f32_denormals_only` | 1000 positive + 1000 negative denormals | Oracle (A) | Denormal handling under pressure |
| `f32_one_nan_among_normals` | `[1.0, NaN, 2.0, 3.0]` | `[1.0, 2.0, 3.0, NaN]` | Single NaN placement |
| `f32_all_nan` | `[NaN; 10000]` (same bits) | All identical | Degenerate: single MSD bucket, all NaN |
| `f32_all_negative_nan` | 1000 negative NaN with varied payloads | Sorted by bit pattern | Negative NaN ordering |
| `f32_inf_density` | 5000 Inf + 5000 -Inf + 5000 normal | Oracle (A) | Heavy Inf concentration |

### 5.2 Distribution Tests

| Test Name | N | Distribution | Verification |
|-----------|---|-------------|-------------|
| `f32_random_uniform_1m` | 1M | `rng.gen::<f32>()` (0.0 to 1.0) | Oracle (A) |
| `f32_random_full_range_1m` | 1M | `f32::from_bits(rng.gen::<u32>())` (all bit patterns incl. NaN/Inf/denormal) | Oracle (A) |
| `f32_random_16m` | 16M | `rng.gen::<f32>()` | Oracle (A) |
| `f32_gaussian` | 1M | Normal distribution, mean=0, std=1000 | Oracle (A) |
| `f32_all_negative` | 1M | `-(rng.gen::<f32>() * 1e6)` | Oracle (A) |
| `f32_clustered` | 1M | 90% in [0.0, 1.0], 10% in [1e30, 1e38] | Oracle (A) -- extreme range variation |
| `f32_pre_sorted` | 1M | Already sorted float sequence | Oracle (A) |
| `f32_nearly_sorted` | 1M | Sorted with 1% random swaps | Oracle (A) |

### 5.3 SortBuffer\<f32\> Tests

| Test Name | Description | Verification |
|-----------|-------------|-------------|
| `f32_buffer_basic` | Copy 1M random f32, sort, verify | Oracle (A) |
| `f32_buffer_with_specials` | Buffer with NaN, Inf, -0.0, denormals mixed in | Oracle (A) + bit-exact |
| `f32_buffer_16m` | 16M f32 zero-copy sort | Oracle (A) |
| `f32_buffer_reuse` | Sort 5 different f32 datasets through same buffer | Oracle (A) each time |

### 5.4 Transform Roundtrip Tests

These tests verify that FloatFlip followed by IFloatFlip is the identity function, independent of the sort:

| Test Name | Description |
|-----------|-------------|
| `f32_transform_roundtrip_special` | Apply FloatFlip then IFloatFlip to all 19 special values; verify bit-exact identity |
| `f32_transform_roundtrip_random` | Apply FloatFlip then IFloatFlip to 1M random f32; verify bit-exact identity |
| `f32_transform_monotonicity` | Apply FloatFlip to pairs where `a < b` (total_cmp); verify `FloatFlip(a) < FloatFlip(b)` as u32 |

**Implementation note**: These tests can run on CPU (applying the transform formulas in Rust) or on GPU (dispatching sort_transform_32 twice). Both should be tested. The CPU version catches algorithm bugs; the GPU version catches shader compilation or dispatch bugs.

---

## 6. Argsort Test Cases

### 6.1 Core Correctness

| Test Name | Type | N | Input | Verification |
|-----------|------|---|-------|-------------|
| `argsort_u32_basic` | u32 | 3 | `[30, 10, 20]` | D: indices = `[1, 2, 0]` |
| `argsort_u32_1m` | u32 | 1M | Random | E + F + A |
| `argsort_u32_16m` | u32 | 16M | Random | E + F |
| `argsort_i32_basic` | i32 | 5 | `[5, -3, 0, -1, 8]` | D: indices = `[1, 3, 2, 0, 4]` |
| `argsort_i32_1m` | i32 | 1M | Random signed | E + F + A |
| `argsort_f32_basic` | f32 | 4 | `[3.14, -2.7, 0.0, 1.0]` | D: indices = `[1, 2, 3, 0]` |
| `argsort_f32_1m` | f32 | 1M | Random | E + F + A |
| `argsort_f32_with_nan` | f32 | 100 | Mix of normals + NaN + Inf | E + F + A |
| `argsort_f32_with_negzero` | f32 | 10 | `[0.0, -0.0, 1.0, -1.0, -0.0, 0.0]` | E + F + verify -0.0 indices before +0.0 indices |

### 6.2 Permutation Validity Verification

These tests focus specifically on verifying that argsort returns a valid permutation:

```rust
fn verify_argsort_permutation(indices: &[u32], n: usize) {
    // E: Every index appears exactly once
    assert_eq!(indices.len(), n, "argsort returned wrong number of indices");
    let mut seen = vec![false; n];
    for &idx in indices {
        assert!((idx as usize) < n, "index {} out of range [0, {})", idx, n);
        assert!(!seen[idx as usize], "duplicate index {}", idx);
        seen[idx as usize] = true;
    }
}

fn verify_argsort_sorted_order<T>(
    data: &[T],
    indices: &[u32],
    cmp: fn(&T, &T) -> std::cmp::Ordering,
) {
    for i in 1..indices.len() {
        let a = &data[indices[i - 1] as usize];
        let b = &data[indices[i] as usize];
        assert!(
            cmp(a, b) != std::cmp::Ordering::Greater,
            "argsort not sorted at position {}: data[{}] > data[{}]",
            i, indices[i - 1], indices[i]
        );
    }
}
```

### 6.3 Argsort Does Not Modify Input

| Test Name | Description |
|-----------|-------------|
| `argsort_input_unmodified_u32` | Clone input before argsort, verify input unchanged after |
| `argsort_input_unmodified_f32` | Same for f32 with NaN values (bit-exact comparison) |

### 6.4 Argsort Edge Cases

| Test Name | Input | Expected |
|-----------|-------|----------|
| `argsort_empty` | `[]` | `[]` |
| `argsort_single` | `[42.0]` | `[0]` |
| `argsort_two_sorted` | `[1.0, 2.0]` | `[0, 1]` |
| `argsort_two_reversed` | `[2.0, 1.0]` | `[1, 0]` |
| `argsort_all_equal` | `[7.0; 1000]` | Any valid permutation (all indices 0..1000, stable: `[0, 1, ..., 999]`) |
| `argsort_stability` | `[1.0, 2.0, 1.0, 2.0, 1.0]` | `[0, 2, 4, 1, 3]` (stable: equal keys preserve original order) |

---

## 7. Key-Value Pair (sort_pairs) Test Cases

### 7.1 Core Correctness

| Test Name | Key Type | N | Description | Verification |
|-----------|----------|---|-------------|-------------|
| `sort_pairs_u32_basic` | u32 | 3 | `keys=[30,10,20], vals=[100,200,300]` | D: keys=[10,20,30], vals=[200,300,100] |
| `sort_pairs_u32_1m` | u32 | 1M | Random keys, sequential values 0..1M | G + A(keys) |
| `sort_pairs_i32_basic` | i32 | 4 | `keys=[5,-3,0,-1], vals=[a,b,c,d]` | D: keys=[-3,-1,0,5], vals=[b,d,c,a] |
| `sort_pairs_i32_1m` | i32 | 1M | Random signed keys, sequential values | G + A(keys) |
| `sort_pairs_f32_basic` | f32 | 3 | `keys=[5.0,1.0,3.0], vals=[100,200,300]` | D: keys=[1.0,3.0,5.0], vals=[200,300,100] |
| `sort_pairs_f32_1m` | f32 | 1M | Random f32 keys, sequential values | G + A(keys) |
| `sort_pairs_f32_with_nan` | f32 | 100 | Mix of normals + NaN keys, tracked values | G + A(keys) |
| `sort_pairs_f32_16m` | f32 | 16M | Random keys, random values | G (permutation) + B (keys sorted) |

### 7.2 Pair Integrity Verification

The critical invariant: sorting must not lose, duplicate, or misassociate any (key, value) pair.

```rust
fn verify_pair_integrity_f32(
    original_keys: &[f32], original_values: &[u32],
    sorted_keys: &[f32], sorted_values: &[u32],
) {
    assert_eq!(original_keys.len(), sorted_keys.len());
    assert_eq!(original_values.len(), sorted_values.len());

    // Use u32 bit representations for f32 keys (handles NaN and -0.0)
    let mut input_pairs: HashMap<(u32, u32), usize> = HashMap::new();
    for (k, &v) in original_keys.iter().zip(original_values.iter()) {
        *input_pairs.entry((k.to_bits(), v)).or_default() += 1;
    }

    let mut output_pairs: HashMap<(u32, u32), usize> = HashMap::new();
    for (k, &v) in sorted_keys.iter().zip(sorted_values.iter()) {
        *output_pairs.entry((k.to_bits(), v)).or_default() += 1;
    }

    assert_eq!(input_pairs, output_pairs,
        "Pair integrity violation: input and output multisets differ");
}
```

### 7.3 Stability Tests

Radix sort is inherently stable. This must be verified for key-value pairs:

| Test Name | Description | Verification |
|-----------|-------------|-------------|
| `sort_pairs_stability_u32` | Keys: `[3, 1, 3, 1, 3]`, Values: `[0, 1, 2, 3, 4]`. After sort: keys=`[1,1,3,3,3]`, values=`[1,3,0,2,4]` | Hardcoded (D) |
| `sort_pairs_stability_f32` | Keys: `[1.0, 2.0, 1.0, 2.0]`, Values: `[10, 20, 30, 40]`. After sort: keys=`[1.0,1.0,2.0,2.0]`, values=`[10,30,20,40]` | Hardcoded (D) |
| `sort_pairs_stability_1m` | 1M elements with only 10 distinct keys, values=0..1M | Within each key group, values must be in ascending order (original insertion order) |
| `sort_pairs_negzero_stability` | Keys: `[-0.0, 0.0, -0.0, 0.0]`, Values: `[0, 1, 2, 3]`. After sort: keys=`[-0.0, -0.0, 0.0, 0.0]`, values=`[0, 2, 1, 3]` | Hardcoded (D), bit-exact key comparison |

### 7.4 Error Handling Tests

| Test Name | Input | Expected Result |
|-----------|-------|-----------------|
| `sort_pairs_length_mismatch` | keys.len()=100, values.len()=99 | `Err(SortError::LengthMismatch { keys: 100, values: 99 })` |
| `sort_pairs_empty` | keys=[], values=[] | `Ok(())`, both unchanged |
| `sort_pairs_single` | keys=[42.0], values=[7] | `Ok(())`, both unchanged |

### 7.5 sort_pairs Strategy B Verification

Since sort_pairs uses argsort + gather internally (Strategy B), we verify the gather step:

| Test Name | Description |
|-----------|-------------|
| `sort_pairs_gather_correctness` | After sort_pairs, verify `sorted_values[i] == original_values[argsort_result[i]]` for a known argsort |
| `sort_pairs_random_values` | Values are random u32 (not sequential) -- verifies gather handles non-sequential patterns |
| `sort_pairs_values_all_same` | All values identical -- verifies gather handles degenerate case |
| `sort_pairs_values_max` | All values = u32::MAX -- verifies no corruption of extreme value bits |

---

## 8. f64 Test Cases

### 8.1 Core Correctness (Mirrors f32 Structure)

| Test Name | N | Description | Verification |
|-----------|---|-------------|-------------|
| `f64_total_cmp_semantics` | 8 | Same special values as f32 version | D (hardcoded) |
| `f64_all_special_values` | 16 | All values from Section 3.3 matrix | D + bit-exact |
| `f64_zeros_ordering` | 4 | `[+0.0, -0.0, +0.0, -0.0]` | D: `[-0.0, -0.0, +0.0, +0.0]` |
| `f64_denormals` | 1000 | Mixed positive and negative denormals | Oracle (A) |
| `f64_nan_payloads` | 100 | Varied NaN bit patterns | Oracle (A) |
| `f64_random_1m` | 1M | `rng.gen::<f64>()` | Oracle (A) |
| `f64_random_16m` | 16M | `rng.gen::<f64>()` | Oracle (A) |
| `f64_full_bit_range` | 1M | `f64::from_bits(rng.gen::<u64>())` | Oracle (A) |

### 8.2 Multi-Pass Specific Tests

f64 uses 8 radix passes (1 MSD + 7 inner across 3 fused dispatches). These tests specifically target multi-pass correctness:

| Test Name | Description | What It Catches |
|-----------|-------------|-----------------|
| `f64_differ_only_in_lsb` | All elements identical except least significant byte | Final inner pass (byte 0) is the only discriminator |
| `f64_differ_only_in_msb` | All elements identical except most significant byte | MSD scatter is the only discriminator |
| `f64_differ_in_middle_bytes` | Elements differ only in bytes 3-4 | Tests boundary between first and second inner fused groups |
| `f64_pass_boundary_sweep` | For each byte position 0-7: create data where only that byte varies | Verifies each individual pass contributes correctly |
| `f64_all_same_except_one` | 16M identical elements + 1 different | Degenerate bucket structure across 8 passes |
| `f64_buffer_pingpong` | 16M random f64 | Verifies buf_a/buf_b tracking across 6 dispatches is correct |

### 8.3 SortBuffer\<f64\> and Argsort Tests

| Test Name | Description |
|-----------|-------------|
| `f64_buffer_basic` | 1M random f64 via SortBuffer\<f64\> |
| `f64_buffer_with_specials` | Buffer with f64 NaN, Inf, -0.0, denormals |
| `f64_argsort_basic` | argsort_f64 with small input, verify permutation |
| `f64_argsort_1m` | argsort_f64 with 1M random f64 |

---

## 9. Regression Test Plan for v1 Compatibility

### 9.1 Invariant: sort_u32 Results Are Bit-Identical to v1

The v2 changes (SortBuffer becoming generic, PsoCache extension, new function constant PSOs) must NOT affect the u32-only code path.

| Test Name | Description | v1 Baseline |
|-----------|-------------|-------------|
| `regression_u32_sort_unchanged` | Run all existing 18 integration tests from correctness.rs | Must produce identical results |
| `regression_u32_buffer_unchanged` | All 7 existing SortBuffer tests | Must produce identical results |
| `regression_u32_perf_no_regression` | sort_u32 16M throughput | Must meet or exceed 2000 Mk/s (memcpy) |
| `regression_u32_perf_zc_no_regression` | sort_buffer 16M throughput | Must meet or exceed 3600 Mk/s (zero-copy) |
| `regression_u32_interleaved_with_new_types` | Interleave sort_u32, sort_i32, sort_f32 on same sorter | u32 results unchanged by intervening typed sorts |

### 9.2 Regression Strategy

All existing tests in `tests/correctness.rs` remain **unchanged in logic**. The only change is that `SortBuffer` references become `SortBuffer<u32>` per the UX.md breaking change. The test *expected results* must not change.

**Execution**: Run the existing test suite (`cargo test` on the v2 branch) before adding any new tests. All existing tests must pass as-is (modulo the `SortBuffer` -> `SortBuffer<u32>` type annotation).

### 9.3 PSO Cache Isolation

The new specialized PSOs must not interfere with the unspecialized PSOs used by sort_u32:

| Test Name | Description |
|-----------|-------------|
| `regression_pso_cache_isolation` | Create sorter, call sort_f32 (creates transform PSOs), then call sort_u32. Verify sort_u32 still uses the original unspecialized PSOs and produces correct results. |
| `regression_sort_order_independence` | sort_u32 -> sort_f32 -> sort_u32 -> sort_i32 -> sort_u32. Each sort_u32 call produces correct results. |

---

## 10. Performance Acceptance Criteria

### 10.1 Throughput Floors per Type

All thresholds measured at 16M elements on Apple M4 Pro. Thresholds include 30% headroom below expected performance to absorb thermal throttle variation.

| Method | Path | Expected (Mk/s) | Threshold (Mk/s) | Basis |
|--------|------|-----------------|-------------------|-------|
| `sort_u32` | memcpy | ~2859 | >= 2000 | Existing baseline (no regression) |
| `sort_u32` (buffer) | zero-copy | ~5207 | >= 3600 | Existing baseline (no regression) |
| `sort_i32` | memcpy | ~2800 | >= 1900 | Same as u32 (negligible transform overhead) |
| `sort_i32` (buffer) | zero-copy | ~4200 | >= 2900 | ~19% overhead for GPU transforms |
| `sort_f32` | memcpy | ~2500 | >= 1750 | ~10% transform overhead |
| `sort_f32` (buffer) | zero-copy | ~4500 | >= 3100 | ~15% for GPU-side FloatFlip |
| `argsort_u32` | memcpy | ~2000 | >= 1400 | ~25-30% overhead (init indices + KV sort) |
| `argsort_f32` | memcpy | ~1700 | >= 1200 | ~35-40% overhead (transform + KV sort) |
| `sort_pairs_u32` | memcpy | ~1800 | >= 1200 | ~35-40% overhead (argsort + gather) |
| `sort_pairs_f32` | memcpy | ~1500 | >= 1000 | ~45-50% overhead (transform + argsort + gather) |
| `sort_u64` | memcpy | ~850 | >= 600 | 8 passes + 8-byte elements, no transform |
| `sort_u64` (buffer) | zero-copy | ~1250 | >= 850 | Baseline 64-bit (no transform overhead) |
| `sort_i64` | memcpy | ~800 | >= 550 | 8 passes + 8-byte elements, sign bit XOR |
| `sort_i64` (buffer) | zero-copy | ~1200 | >= 800 | Same as f64 (similar transform cost) |
| `sort_f64` | memcpy | ~800 | >= 550 | 8 passes + 8-byte elements |
| `sort_f64` (buffer) | zero-copy | ~1200 | >= 800 | 4x slower than u32 zero-copy |
| `argsort_u64` | memcpy | ~600 | >= 400 | u64 argsort (KV + 8 passes) |
| `argsort_i64` | memcpy | ~550 | >= 380 | i64 argsort (KV + 8 passes + transform) |
| `sort_pairs_u64` | memcpy | ~500 | >= 350 | u64 KV pairs (argsort + gather, 64-bit keys) |
| `sort_pairs_i64` | memcpy | ~480 | >= 330 | i64 KV pairs (argsort + gather + transform) |

### 10.2 Performance Test Structure

Performance tests are **feature-gated** to prevent CI flakiness:

```rust
#[cfg(feature = "perf-test")]
mod performance {
    use super::*;

    #[test]
    fn perf_sort_i32_16m() {
        let mut sorter = GpuSorter::new().unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data: Vec<i32> = (0..16_000_000).map(|_| rng.gen()).collect();

        // Warmup
        for _ in 0..3 {
            let mut copy = data.clone();
            sorter.sort_i32(&mut copy).unwrap();
        }

        // Measure
        let mut times = Vec::new();
        for _ in 0..10 {
            let mut copy = data.clone();
            let start = Instant::now();
            sorter.sort_i32(&mut copy).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = times[times.len() / 2];
        let mkeys = 16_000_000.0 / p50 / 1000.0;

        assert!(
            mkeys >= 1900.0,
            "sort_i32 16M: {:.0} Mk/s below threshold 1900 Mk/s (p50={:.3}ms)",
            mkeys, p50
        );
    }
}
```

### 10.3 Transform Overhead Measurement

Separate tests that measure transform kernel overhead in isolation:

| Test | Measurement | Threshold |
|------|------------|-----------|
| `perf_transform_f32_16m` | Time for FloatFlip + IFloatFlip dispatches alone (no sort) | < 1.0ms total |
| `perf_transform_f64_16m` | Time for f64 FloatFlip + IFloatFlip dispatches alone | < 2.0ms total |
| `perf_init_indices_16m` | Time for sort_init_indices dispatch alone | < 0.5ms |
| `perf_gather_16m` | Time for sort_gather_values dispatch alone | < 1.5ms |

These help diagnose whether performance issues originate in the transform/utility kernels or in the sort pipeline itself.

---

## 11. Test File Organization

### 11.1 File Structure

```
forge-sort/
  Cargo.toml                          # Add: rand_chacha = "0.3" (dev-dep)
  src/lib.rs                          # Unit tests for struct sizes, SortKey trait
  tests/
    correctness.rs                    # PRESERVE -- v1 u32 tests (update SortBuffer type only)
    correctness_i32.rs                # NEW -- i32 correctness tests (~20 tests)
    correctness_f32.rs                # NEW -- f32 correctness + IEEE 754 edge cases (~25 tests)
    correctness_u64.rs                # NEW -- u64 correctness (baseline 64-bit) (~15 tests)
    correctness_i64.rs                # NEW -- i64 correctness + sign boundary (~15 tests)
    correctness_f64.rs                # NEW -- f64 correctness + multi-pass tests (~20 tests)
    correctness_argsort.rs            # NEW -- argsort permutation + correctness (~20 tests)
    correctness_sort_pairs.rs         # NEW -- key-value integrity + stability (~20 tests)
    regression.rs                     # NEW -- v1 compatibility + cross-type isolation (~7 tests)
    performance.rs                    # NEW -- feature-gated throughput tests (~15 tests)
    common/
      mod.rs                          # Shared: oracle functions, verification helpers,
                                      #         seeded RNG factory, test vector generators
```

### 11.2 Rationale for Separate Files

Unlike the v1 test strategy (OVERVIEW.md) which recommended a single `correctness.rs` with modules, v2 splits by type because:

1. **Each type has distinct oracle/verification logic** -- f32 needs bit-exact comparison, i32 uses natural Ord, argsort has permutation checks
2. **Test count is large** (~130+ new tests) -- a single file becomes unwieldy
3. **Parallel compilation** -- separate .rs files compile as separate crates, enabling parallel test compilation
4. **Selective execution** -- `cargo test --test correctness_f32` runs only f32 tests, useful during Phase 1 development
5. **Independent failures** -- a compile error in f64 tests does not block running f32 tests

### 11.3 Shared Test Helpers (`tests/common/mod.rs`)

```rust
pub mod common {
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;
    use std::collections::HashMap;

    /// Create a seeded RNG for reproducible tests.
    pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    // ── Oracle Functions ──────────────────────────────────────

    pub fn oracle_sort_i32(data: &[i32]) -> Vec<i32> {
        let mut v = data.to_vec();
        v.sort();
        v
    }

    pub fn oracle_sort_f32(data: &[f32]) -> Vec<f32> {
        let mut v = data.to_vec();
        v.sort_by(f32::total_cmp);
        v
    }

    pub fn oracle_sort_u64(data: &[u64]) -> Vec<u64> {
        let mut v = data.to_vec();
        v.sort();
        v
    }

    pub fn oracle_sort_i64(data: &[i64]) -> Vec<i64> {
        let mut v = data.to_vec();
        v.sort();
        v
    }

    pub fn oracle_sort_f64(data: &[f64]) -> Vec<f64> {
        let mut v = data.to_vec();
        v.sort_by(f64::total_cmp);
        v
    }

    pub fn oracle_argsort_f32(data: &[f32]) -> Vec<u32> {
        let mut indices: Vec<u32> = (0..data.len() as u32).collect();
        indices.sort_by(|&a, &b| {
            data[a as usize].total_cmp(&data[b as usize])
        });
        indices
    }

    // ── Verification Helpers ──────────────────────────────────

    /// Bit-exact f32 comparison (handles NaN and -0.0).
    pub fn assert_f32_eq(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(), e.to_bits(),
                "Mismatch at index {}: actual={} (bits={:#010x}), expected={} (bits={:#010x})",
                i, a, a.to_bits(), e, e.to_bits()
            );
        }
    }

    /// Bit-exact f64 comparison.
    pub fn assert_f64_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(), e.to_bits(),
                "Mismatch at index {}: actual={} (bits={:#018x}), expected={} (bits={:#018x})",
                i, a, a.to_bits(), e, e.to_bits()
            );
        }
    }

    /// Verify argsort result is a valid permutation and yields sorted order.
    pub fn verify_argsort<T: Copy>(
        data: &[T],
        indices: &[u32],
        cmp: fn(&T, &T) -> std::cmp::Ordering,
    ) {
        let n = data.len();
        assert_eq!(indices.len(), n, "argsort returned wrong count");

        // Permutation validity
        let mut seen = vec![false; n];
        for &idx in indices {
            let i = idx as usize;
            assert!(i < n, "index {} out of range", idx);
            assert!(!seen[i], "duplicate index {}", idx);
            seen[i] = true;
        }

        // Sorted order
        for i in 1..n {
            let a = &data[indices[i - 1] as usize];
            let b = &data[indices[i] as usize];
            assert!(
                cmp(a, b) != std::cmp::Ordering::Greater,
                "not sorted at position {}: data[{}] > data[{}]",
                i, indices[i - 1], indices[i]
            );
        }
    }

    /// Verify sort stability: for equal keys, values must be in
    /// ascending order of their original position.
    pub fn verify_stability_f32(sorted_keys: &[f32], sorted_values: &[u32]) {
        for i in 1..sorted_keys.len() {
            if sorted_keys[i - 1].to_bits() == sorted_keys[i].to_bits() {
                assert!(
                    sorted_values[i - 1] < sorted_values[i],
                    "stability violation at index {}: equal keys (bits={:#010x}) \
                     but value {} >= {}",
                    i, sorted_keys[i].to_bits(),
                    sorted_values[i - 1], sorted_values[i]
                );
            }
        }
    }

    // ── Test Data Generators ──────────────────────────────────

    pub fn gen_f32_special_values() -> Vec<f32> {
        vec![
            f32::from_bits(0xFFC00001), // negative quiet NaN (non-canonical)
            f32::from_bits(0xFFC00000), // negative quiet NaN (canonical)
            f32::from_bits(0xFF800001), // negative signaling NaN
            f32::NEG_INFINITY,
            -f32::MAX,
            -1.0,
            -f32::MIN_POSITIVE,
            f32::from_bits(0x807FFFFF), // negative denormal boundary
            f32::from_bits(0x80000001), // smallest negative denormal
            -0.0_f32,
            0.0_f32,
            f32::from_bits(0x00000001), // smallest positive denormal
            f32::MIN_POSITIVE,
            1.0,
            f32::MAX,
            f32::INFINITY,
            f32::from_bits(0x7F800001), // positive signaling NaN
            f32::from_bits(0x7FC00000), // positive quiet NaN (canonical)
            f32::from_bits(0x7FC00001), // positive quiet NaN (non-canonical)
        ]
    }

    pub fn gen_f64_special_values() -> Vec<f64> {
        vec![
            f64::from_bits(0xFFF8_0000_0000_0001), // negative quiet NaN
            f64::from_bits(0xFFF0_0000_0000_0001), // negative signaling NaN
            f64::NEG_INFINITY,
            -f64::MAX,
            -1.0,
            -f64::MIN_POSITIVE,
            f64::from_bits(0x8000_0000_0000_0001), // smallest negative denormal
            -0.0_f64,
            0.0_f64,
            f64::from_bits(0x0000_0000_0000_0001), // smallest positive denormal
            f64::MIN_POSITIVE,
            1.0,
            f64::MAX,
            f64::INFINITY,
            f64::from_bits(0x7FF0_0000_0000_0001), // positive signaling NaN
            f64::NAN,
        ]
    }

    pub fn gen_u64_boundary_values() -> Vec<u64> {
        vec![
            0,
            1,
            0xFF,
            0xFFFF,
            0xFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFFFF,
            0xFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFF,
            u64::MAX / 2,
            u64::MAX - 1,
            u64::MAX,
        ]
    }

    pub fn gen_i64_boundary_values() -> Vec<i64> {
        vec![
            i64::MIN,
            i64::MIN + 1,
            -1_000_000_000_000,
            -1_000_000,
            -256,
            -128,
            -1,
            0,
            1,
            128,
            256,
            1_000_000,
            1_000_000_000_000,
            i64::MAX - 1,
            i64::MAX,
        ]
    }

    pub fn gen_i32_boundary_values() -> Vec<i32> {
        vec![
            i32::MIN,
            i32::MIN + 1,
            -1_000_000,
            -256,
            -128,
            -1,
            0,
            1,
            128,
            256,
            1_000_000,
            i32::MAX - 1,
            i32::MAX,
        ]
    }
}
```

### 11.4 Phase-to-File Mapping

| Phase | Files Created/Modified | Test Count |
|-------|----------------------|------------|
| Phase 1 | `correctness_i32.rs`, `correctness_f32.rs`, `common/mod.rs`, update `correctness.rs` type annotation | ~50 |
| Phase 2 | `correctness_argsort.rs`, `correctness_sort_pairs.rs`, `regression.rs` | ~50 |
| Phase 3 | `correctness_u64.rs`, `correctness_i64.rs`, `correctness_f64.rs`, extend `performance.rs` | ~50 |
| All phases | `performance.rs` (feature-gated, incremental) | ~5 per phase |

---

## 12. Mutation Testing Guidance

To validate that the test suite catches real bugs, the following deliberate mutations should be introduced and verified to fail at least one test.

### 12.1 Transform Mutations

| Mutation | File | Expected Failing Tests |
|----------|------|----------------------|
| Change i32 XOR mask from `0x80000000` to `0x7FFFFFFF` | sort.metal (sort_transform_32) | All i32 tests |
| Swap FloatFlip forward and inverse modes (mode 1 <-> mode 2) | lib.rs (TRANSFORM_MODE constants) | All f32 tests |
| Remove FloatFlip post-sort inverse transform dispatch | lib.rs (dispatch pipeline) | All f32 tests (data returns as u32 bit soup) |
| Use `0x80000000` mask for both f32 positive and negative | sort.metal (sort_transform_32) | f32 negative value tests |
| Off-by-one in f64 shift (use 55 instead of 56 for MSD) | lib.rs (config_f64) | f64 tests |
| Remove transform for zero-copy path but keep for memcpy | lib.rs (sort_f32_buffer) | f32 buffer tests only |

### 12.2 Argsort Mutations

| Mutation | Expected Failing Tests |
|----------|----------------------|
| Initialize indices to `[0, 0, 0, ..., 0]` instead of `[0, 1, 2, ..., n-1]` | All argsort permutation validity tests |
| Skip the sort_init_indices dispatch entirely | All argsort tests (uninitialized index buffer) |
| Swap buf_vals_a and buf_vals_b in the final readback | All argsort tests (read wrong buffer) |
| Read keys from buf_b instead of buf_a after pipeline | All sort tests including argsort |

### 12.3 Key-Value Mutations

| Mutation | Expected Failing Tests |
|----------|----------------------|
| Omit value scatter in sort_msd_atomic_scatter (`has_values` branch deleted) | All sort_pairs and argsort tests |
| Scatter values to key positions (write to dst instead of dst_vals) | Pair integrity tests |
| Skip gather dispatch in sort_pairs | sort_pairs value ordering tests |
| Gather from sorted buffer instead of original values buffer | Pair integrity tests (values become key duplicates) |

### 12.4 u64/i64-Specific Mutations

| Mutation | Expected Failing Tests |
|----------|----------------------|
| Skip IS_64BIT=true for u64 sort (use 32-bit PSO) | All u64 tests (only lower 32 bits sorted) |
| Use 0x80000000 (32-bit) instead of 0x8000000000000000 (64-bit) for i64 XOR mask | All i64 tests with negative values |
| Apply transform to u64 (should have none) | u64 sort order incorrect for values with high bit set |
| Use TRANSFORM_MODE=2 (IFloatFlip) instead of mode=1 for i64 forward transform | i64 negative value ordering incorrect |

### 12.5 f64-Specific Mutations

| Mutation | Expected Failing Tests |
|----------|----------------------|
| Use 32-bit tile size (4096) instead of 2048 for 64-bit keys | f64 correctness tests (TG memory overflow or wrong indexing) |
| Wrong buffer swap between inner fused dispatches 4 and 5 | f64 tests with varied middle bytes |
| Use 4 inner passes instead of 7 (skip byte positions 0-2) | f64 tests where low bytes differ |

---

## 13. Test Execution Strategy

### 13.1 Local Development

```bash
# Run all correctness tests (default cargo test)
cargo test --release

# Run only f32 tests during Phase 1 development
cargo test --release --test correctness_f32

# Run only argsort tests during Phase 2 development
cargo test --release --test correctness_argsort

# Run performance tests (requires feature flag)
cargo test --release --features perf-test --test performance

# Run regression tests explicitly
cargo test --release --test regression
```

### 13.2 CI Pipeline

```yaml
# Correctness (every PR)
- cargo test --release

# Performance (nightly or manual trigger, not every PR)
- cargo test --release --features perf-test --test performance
```

Performance tests are excluded from default CI because GPU thermal state varies between runs. They run nightly on a dedicated M4 Pro machine.

### 13.3 Test Ordering Within Each Phase

1. **Hardcoded edge cases (D)** -- fastest to write, catches obvious transform bugs
2. **Small random (A)** -- catches algorithm bugs with oracle comparison at N=1K
3. **Medium random (A + C)** -- catches tile boundary / batching bugs at N=1M
4. **Large random (A or B)** -- catches scale-dependent bugs at N=16M
5. **Performance (B)** -- last, after correctness is established

---

## 14. Acceptance Criteria Summary

### 14.1 Phase 1 Gate (i32 + f32)

- [ ] All 21 existing v1 tests pass with `SortBuffer<u32>` type annotation
- [ ] 20+ i32 tests pass (boundaries, distributions, buffer, transform roundtrip)
- [ ] 25+ f32 tests pass (IEEE 754 edge cases, distributions, buffer, transform roundtrip)
- [ ] f32 bit-exact match with `sort_by(f32::total_cmp)` for all test inputs
- [ ] `-0.0` sorts before `+0.0` in all f32 tests
- [ ] NaN sorts after `+Inf` (positive NaN) and before `-Inf` (negative NaN)
- [ ] Transform roundtrip tests pass (FloatFlip -> IFloatFlip = identity)
- [ ] i32 transform roundtrip tests pass (XOR 0x80000000 is self-inverse)
- [ ] sort_u32 performance unchanged (>= 2000 Mk/s memcpy, >= 3600 Mk/s zero-copy)
- [ ] sort_i32 performance >= 1900 Mk/s (memcpy @ 16M)
- [ ] sort_f32 performance >= 1750 Mk/s (memcpy @ 16M)

### 14.2 Phase 2 Gate (argsort + key-value)

- [ ] All Phase 1 tests still pass
- [ ] 15+ argsort tests pass (u32, i32, f32 variants)
- [ ] Every argsort result is a valid permutation of 0..n
- [ ] Argsort does not modify input data
- [ ] 20+ sort_pairs tests pass (u32, i32, f32 variants)
- [ ] Every sort_pairs call preserves the (key, value) pair multiset
- [ ] Sort stability verified: equal keys preserve original relative order
- [ ] LengthMismatch error returned for mismatched key/value lengths
- [ ] Empty and single-element edge cases pass for all argsort and sort_pairs variants
- [ ] argsort_u32 performance >= 1400 Mk/s (memcpy @ 16M)
- [ ] sort_pairs_f32 performance >= 1000 Mk/s (memcpy @ 16M)

### 14.3 Phase 3 Gate (u64 + i64 + f64)

- [ ] All Phase 1 + Phase 2 tests still pass
- [ ] 15+ u64 tests pass (boundary values, random distributions, buffer, argsort, sort_pairs)
- [ ] u64 exact match with `[u64]::sort()` for all test inputs
- [ ] 15+ i64 tests pass (boundary values, sign boundary, random, buffer, argsort, sort_pairs)
- [ ] i64 exact match with `[i64]::sort()` for all test inputs
- [ ] i64 transform roundtrip: XOR 0x8000000000000000 twice = identity
- [ ] i64::MIN sorts before i64::MAX; -1 sorts before 0
- [ ] 20+ f64 tests pass (IEEE 754 edge cases, multi-pass, buffer)
- [ ] f64 bit-exact match with `sort_by(f64::total_cmp)` for all test inputs
- [ ] Multi-pass specific tests pass (byte-position isolation, buffer ping-pong)
- [ ] argsort_{u64,i64,f64} return valid permutations
- [ ] sort_pairs_{u64,i64} preserve key-value pair integrity
- [ ] sort_u64 performance >= 600 Mk/s (memcpy @ 16M)
- [ ] sort_u64_buffer performance >= 850 Mk/s (zero-copy @ 16M)
- [ ] sort_i64 performance >= 550 Mk/s (memcpy @ 16M)
- [ ] sort_i64_buffer performance >= 800 Mk/s (zero-copy @ 16M)
- [ ] sort_f64 performance >= 550 Mk/s (memcpy @ 16M)
- [ ] sort_f64_buffer performance >= 800 Mk/s (zero-copy @ 16M)

### 14.4 Overall Quality Gate

- [ ] Total new test count >= 150
- [ ] Zero flaky tests (correctness tests use seeded RNG, are fully deterministic)
- [ ] All mutation tests from Section 12 fail at least one test when applied
- [ ] Cross-type interleaving works (sort_u32, sort_f32, sort_i32 on same GpuSorter)
- [ ] No memory leaks (lazy buffer allocation, grow-only, no unbounded growth)
- [ ] Correctness test is NEVER flaky: a failure is treated as a P0 bug

---

## 15. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| FloatFlip has a subtle bug for a specific NaN payload range | Low | Data corruption (silent) | Full IEEE 754 edge case matrix (Section 3) with bit-exact verification |
| f64 buffer ping-pong tracking is wrong for one of the 6 dispatches | Medium | Sorted data in wrong buffer, appears as scrambled output | Dedicated per-byte-position tests (Section 8.2) catch individual pass errors |
| argsort index initialization dispatch silently fails or is skipped | Low | All indices zero, sort appears to "work" but permutation is invalid | Permutation validity check (every index appears exactly once) catches this immediately |
| Gather kernel reads from wrong buffer | Medium | Values associated with wrong keys | Pair integrity multiset check catches any key-value misassociation |
| Performance regresses for existing sort_u32 due to PsoCache changes | Low | Throughput drops below threshold | Dedicated regression perf test with absolute threshold |
| i32 XOR mask applied twice (pre + post) cancels out, appears to sort unsigned | Very Low | i32 sorts as u32 (wrong order for negatives) | Boundary value tests with i32::MIN, i32::MAX, -1 catch this in first test |
| f32 transform applied to wrong buffer in zero-copy path | Medium | SortBuffer\<f32\> contents corrupted | SortBuffer\<f32\> specific tests with special values catch this |
| sort_pairs gather writes to input buffer instead of output buffer | Low | Original values overwritten | Pair integrity check fails if values are corrupted |
| Function constant specialization produces wrong PSO for KV path | Medium | has_values=true PSO differs from expected | Argsort and sort_pairs tests exercise the KV PSO path; keys-only tests exercise the non-KV path |
| 64-bit element loading in scatter kernel misaligns | Medium | Garbled 64-bit keys | f64/u64/i64 random tests at 1M/16M catch any misalignment (oracle comparison) |
| u64 accidentally routed through 32-bit PSO | Low | Only lower 32 bits sorted | u64 boundary tests with values > 2^32 catch this immediately |
| i64 XOR mask uses 32-bit constant instead of 64-bit | Low | Negative i64 values sort incorrectly | i64 boundary tests with i64::MIN, i64::MAX catch this immediately |
| sort_pairs_u64/i64 value buffer size mismatch (8-byte keys vs 4-byte values) | Medium | Buffer overrun or wrong scatter positions | sort_pairs correctness tests with known expected output catch misalignment |

---

## References

- [CUB DeviceRadixSort API](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html) -- Production GPU sort reference for multi-type support and NaN handling
- [CUB Issue #293: Radix Sort Bitwise Transformations](https://github.com/NVIDIA/cub/issues/293) -- CUB's signed/float transformation documentation
- [Stereopsis: Radix Sort Tricks (FloatFlip/IFloatFlip)](https://stereopsis.com/radix.html) -- Canonical float-to-sortable-uint transformation
- [FloatRadixSort](https://github.com/lshamis/FloatRadixSort) -- Reference float radix sort implementation with test cases
- [eloj/radix-sorting](https://github.com/eloj/radix-sorting) -- Radix sorting ground-up implementation with float support
- [Rust f32::total_cmp](https://doc.rust-lang.org/std/primitive.f32.html#method.total_cmp) -- IEEE 754 total ordering in Rust (our oracle standard)
- [IEEE 754 - Wikipedia](https://en.wikipedia.org/wiki/IEEE_754) -- IEEE 754 standard reference for special values
- [Radix Sort Revisited (Codercorner)](https://codercorner.com/RadixSortRevisited.htm) -- Float radix sort with NaN handling discussion
- [GPU Gems 2: Improved GPU Sorting](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting) -- GPU sorting key-value pair handling
- [AMD FidelityFX Parallel Sort](https://gpuopen.com/fidelityfx-parallel-sort/) -- AMD GPU sort with key-value pair support
- [Sorting with GPUs: A Survey (2017)](https://arxiv.org/pdf/1709.02520) -- Comprehensive GPU sorting literature review and verification methodology

## Questions & Answers

### Q1: Should f32 all-bits random test be in default suite or feature-gated?
**Answer**: Include in default suite. It's the most thorough f32 test and GPU sorts are fast enough that it won't slow CI.
**Impact**: f32_random_full_range_1m runs by default, exercises all 2^32 bit pattern categories including NaN/Inf/denormals.

### Q2: Should stability tests cover NaN mantissa payload ordering?
**Answer**: Yes, test NaN payload stability. Multiple NaN values with same sign bit but different mantissa bits must maintain relative order.
**Impact**: Adds test_f32_nan_payload_stability test case. Catches subtle FloatFlip bugs where mantissa bits are incorrectly handled.

### Q3: Should QA include compile-fail test for excluded sort_pairs_f64?
**Answer**: No, skip. That's API documentation scope, not QA.
**Impact**: No compile-fail tests. f64 key-value exclusion is documented in UX.md and README.
