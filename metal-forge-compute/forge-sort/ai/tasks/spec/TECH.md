# Technical Design: forge-sort Comprehensive Test Suite

## 1. File Structure and Module Layout

### 1.1 Directory Tree

```
forge-sort/
  Cargo.toml                          # Add proptest, features
  src/
    lib.rs                            # Add #[cfg(test)] helpers + constants re-export
  tests/
    correctness.rs                    # REPLACE — all deterministic correctness tests
    property.rs                       # NEW — proptest-based property verification
    performance.rs                    # NEW — perf regression (gated: feature = "perf-test")
    common/
      mod.rs                          # Shared test helpers, generators, verification
```

### 1.2 Rationale

Rust integration tests live in `tests/` and each file compiles as a separate crate.
We split into three files by concern:

- **correctness.rs** — deterministic tests covering PM categories 1-6, plus concurrency.
  Every test has a known input and a single correct output. ~55 tests.
- **property.rs** — proptest-powered randomized tests covering PM category 3
  (determinism) and cross-cutting property verification. ~8 test functions generating
  thousands of cases.
- **performance.rs** — wall-clock regression tests gated behind
  `#[cfg(feature = "perf-test")]` covering PM category 7. ~5 tests. Not run in default
  `cargo test`.

The `common/` module is a shared utility crate available to all three test files via
`mod common;`.

### 1.3 Cargo.toml Changes

```toml
[features]
perf-test = []

[dev-dependencies]
rand = "0.8"
rand_chacha = "0.3"        # Deterministic seeded RNG
rayon = "1.10"
proptest = "1.4"
```

The `perf-test` feature has no effect on library code. It exists purely to gate test
compilation:

```bash
# Run correctness + property tests (CI default)
cargo test

# Run everything including perf regression
cargo test --features perf-test
```

---

## 2. Shared Test Infrastructure (`tests/common/mod.rs`)

### 2.1 Core Verification Function

```rust
use forge_sort::GpuSorter;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Sort `data` on GPU and assert it matches CPU reference sort.
/// Returns (actual, expected) for further inspection if needed.
pub fn gpu_sort_and_verify(sorter: &mut GpuSorter, data: &mut Vec<u32>) {
    let mut expected = data.clone();
    expected.sort_unstable();

    sorter.sort_u32(data).unwrap();

    if *data != expected {
        // Find first mismatch for diagnostic
        let idx = data.iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(data.len());
        panic!(
            "Sort mismatch at n={}. First diff at index {}: \
             got 0x{:08X}, expected 0x{:08X}",
            data.len(), idx,
            data.get(idx).copied().unwrap_or(0),
            expected.get(idx).copied().unwrap_or(0),
        );
    }
}

/// Convenience: generate + sort + verify.
pub fn gen_sort_verify(sorter: &mut GpuSorter, data: Vec<u32>) {
    let mut data = data;
    gpu_sort_and_verify(sorter, &mut data);
}
```

### 2.2 Seeded RNG Factory

All randomized tests use `ChaCha8Rng` with explicit seeds so failures are reproducible:

```rust
/// Create a deterministic RNG from a test-specific seed.
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Generate `n` random u32 values with a specific seed.
pub fn random_data(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = seeded_rng(seed);
    (0..n).map(|_| rng.gen()).collect()
}
```

### 2.3 Adversarial Pattern Generators

Each generator targets a specific kernel or byte position.

```rust
/// All elements share the same MSD byte (bits 24:31).
/// Exercises: 1 of 256 MSD buckets gets everything, 255 empty.
pub fn same_msd_byte(n: usize, msd_byte: u8, seed: u64) -> Vec<u32> {
    let mut rng = seeded_rng(seed);
    (0..n)
        .map(|_| {
            let lower: u32 = rng.gen::<u32>() & 0x00FF_FFFF;
            ((msd_byte as u32) << 24) | lower
        })
        .collect()
}

/// Exactly 2 MSD buckets populated, alternating.
/// Exercises: extreme skew, 2 large buckets, 254 empty.
pub fn two_msd_buckets(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = seeded_rng(seed);
    (0..n)
        .map(|i| {
            let msd: u32 = if i % 2 == 0 { 0x00 } else { 0xFF };
            let lower: u32 = rng.gen::<u32>() & 0x00FF_FFFF;
            (msd << 24) | lower
        })
        .collect()
}

/// Uniform distribution across all 256 MSD buckets.
/// Exercises: max bucket count per bin, even atomic contention.
pub fn uniform_msd(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = seeded_rng(seed);
    (0..n)
        .map(|i| {
            let msd: u32 = (i as u32) % 256;
            let lower: u32 = rng.gen::<u32>() & 0x00FF_FFFF;
            (msd << 24) | lower
        })
        .collect()
}

/// All inner bytes identical (XX_00_00_00 pattern).
/// Exercises: inner fused sort sees single-bin-gets-everything.
pub fn zero_inner_bytes(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = seeded_rng(seed);
    (0..n)
        .map(|_| {
            let msd: u32 = rng.gen::<u32>() & 0xFF;
            msd << 24 // lower 24 bits all zero
        })
        .collect()
}

/// All identical values. Every byte-position pass sees one bin.
pub fn all_identical(n: usize, value: u32) -> Vec<u32> {
    vec![value; n]
}

/// Power-of-two values. Sparse bit patterns stress rank computation.
pub fn powers_of_two(n: usize) -> Vec<u32> {
    (0..n).map(|i| 1u32 << (i as u32 % 32)).collect()
}

/// Alternating 0 and u32::MAX.
pub fn alternating_zero_max(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| if i % 2 == 0 { 0u32 } else { u32::MAX })
        .collect()
}

/// Near-duplicate pairs: [v, v+1, v, v+1, ...] with random v.
/// Stresses ranking correctness at adjacent values.
pub fn near_duplicates(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = seeded_rng(seed);
    let v: u32 = rng.gen::<u32>() & 0xFFFF_FFFE; // room for +1
    (0..n).map(|i| if i % 2 == 0 { v } else { v + 1 }).collect()
}

/// Sequential high bytes: 0x00_*, 0x01_*, ..., 0xFF_*
/// Every MSD bucket gets exactly n/256 elements.
pub fn sequential_msd(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = seeded_rng(seed);
    let per_bucket = n / 256;
    let remainder = n % 256;
    let mut data = Vec::with_capacity(n);
    for msd in 0u32..256 {
        let count = per_bucket + if (msd as usize) < remainder { 1 } else { 0 };
        for _ in 0..count {
            let lower: u32 = rng.gen::<u32>() & 0x00FF_FFFF;
            data.push((msd << 24) | lower);
        }
    }
    data
}

/// Bit-24 boundary: half values at 0x00FFFFFF, half at 0x01000000.
/// Adjacent MSD bins, maximum inner spread for one bin.
pub fn bit24_boundary(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            if i % 2 == 0 { 0x00FF_FFFFu32 } else { 0x0100_0000u32 }
        })
        .collect()
}

/// Few unique values (k distinct values, repeated).
pub fn few_unique(n: usize, k: usize, seed: u64) -> Vec<u32> {
    let mut rng = seeded_rng(seed);
    let uniques: Vec<u32> = (0..k).map(|_| rng.gen()).collect();
    (0..n).map(|i| uniques[i % k]).collect()
}

/// Nearly sorted: sorted array with `frac` fraction of elements randomly swapped.
pub fn nearly_sorted(n: usize, frac: f64, seed: u64) -> Vec<u32> {
    let mut data: Vec<u32> = (0..n as u32).collect();
    let mut rng = seeded_rng(seed);
    let swaps = (n as f64 * frac) as usize;
    for _ in 0..swaps {
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        data.swap(i, j);
    }
    data
}
```

### 2.4 Tile Boundary Size Calculator

Formulaic generation of sizes that hit exact tile boundaries:

```rust
pub const TILE_SIZE: usize = 4096;
pub const THREADS_PER_TG: usize = 256;
pub const ELEMS_PER_THREAD: usize = 16;  // SORT_ELEMS
pub const NUM_BINS: usize = 256;

/// Return a vector of interesting sizes for tile-boundary testing.
/// Each size targets a specific partial-tile or alignment condition.
pub fn tile_boundary_sizes() -> Vec<(usize, &'static str)> {
    vec![
        // Exact tile multiples
        (TILE_SIZE, "exactly 1 tile"),
        (TILE_SIZE * 2, "exactly 2 tiles"),
        (TILE_SIZE * 256, "exactly 256 tiles (fused_grid width)"),
        // Tile + small remainder
        (TILE_SIZE + 1, "1 tile + 1 element (minimal partial tile)"),
        (TILE_SIZE * 2 + 1, "2 tiles + 1 element"),
        (TILE_SIZE + THREADS_PER_TG, "1 tile + 1 TG worth of elements"),
        (TILE_SIZE * 256 + 1, "256 tiles + 1 (beyond fused_grid width)"),
        // Tile - small deficit
        (TILE_SIZE - 1, "1 tile - 1 (4095: last thread has 15 valid elems)"),
        (TILE_SIZE * 2 - 1, "2 tiles - 1"),
        // Primes (no alignment to any power-of-two)
        (4099, "prime near 1 tile"),
        (8191, "Mersenne prime near 2 tiles"),
        (16381, "prime near 4 tiles"),
        // SIMD-group boundaries within a tile
        (TILE_SIZE + 32, "1 tile + 1 SIMD group"),
        (TILE_SIZE + 31, "1 tile + 31 (partial SIMD in partial tile)"),
        // Sub-tile sizes
        (1, "single element (GPU early return)"),
        (2, "two elements"),
        (THREADS_PER_TG, "exactly 1 TG width (256)"),
        (THREADS_PER_TG - 1, "255: 1 TG width - 1"),
        (THREADS_PER_TG + 1, "257: 1 TG width + 1"),
    ]
}
```

### 2.5 Performance Measurement Helpers

```rust
use std::time::Instant;

pub struct PerfResult {
    pub median_ms: f64,
    pub p5_ms: f64,
    pub p95_ms: f64,
    pub mkeys_per_sec: f64,
}

/// Run `sort_u32` multiple times, discard warmup, return statistics.
pub fn measure_sort_perf(
    sorter: &mut GpuSorter,
    data: &[u32],
    warmup: usize,
    runs: usize,
) -> PerfResult {
    let n = data.len();
    let mut times = Vec::with_capacity(runs);

    for i in 0..(warmup + runs) {
        let mut copy = data.to_vec();
        let start = Instant::now();
        sorter.sort_u32(&mut copy).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms);
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |pct: f64| -> f64 {
        let idx = (pct / 100.0 * (times.len() - 1) as f64).round() as usize;
        times[idx.min(times.len() - 1)]
    };

    let median = p(50.0);
    PerfResult {
        median_ms: median,
        p5_ms: p(5.0),
        p95_ms: p(95.0),
        mkeys_per_sec: n as f64 / median / 1e3,
    }
}
```

---

## 3. Correctness Tests (`tests/correctness.rs`)

### 3.1 Module Structure

```rust
mod common;

use common::*;
use forge_sort::GpuSorter;

// Each category gets its own module for organization.
// All tests share a single GpuSorter where possible to test buffer reuse,
// but some tests intentionally create fresh sorters.
```

### 3.2 Category 1: Tile Boundary Precision (P0, 10 tests)

These are the highest-priority tests. Off-by-one in the `idx < n` check within
`sort_msd_histogram`, `sort_msd_atomic_scatter`, or `sort_inner_fused` will cause
silent data loss at tile boundaries.

**Tile boundary math**: given `n` elements and `TILE_SIZE = 4096`:
- `num_tiles = ceil(n / 4096)`
- Last tile has `n - (num_tiles - 1) * 4096` valid elements
- Within the last tile, thread `t` handles elements at offsets `t, t+256, t+512, ...`
  up to `t + 15*256 = t + 3840`. Thread `t` has `ceil((valid_in_last_tile - t) / 256)`
  valid elements (clamped to 0..16).
- Critical: when `n % TILE_SIZE == 1`, only thread 0 in the last tile has a single
  valid element. The other 255 threads must all hit `valid[e] = false` for all 16 slots.
- When `n % TILE_SIZE == 0`, there is no partial tile at all.

```rust
#[test]
fn tile_exact_one() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(TILE_SIZE, 100));
}

#[test]
fn tile_exact_two() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(TILE_SIZE * 2, 101));
}

#[test]
fn tile_exact_256() {
    // Matches fused_grid dispatch width (256 TGs in inner sort)
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(TILE_SIZE * 256, 102));
}

#[test]
fn tile_plus_one() {
    // Minimal partial tile: 1 valid element in tile 2
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(TILE_SIZE + 1, 103));
}

#[test]
fn tile_minus_one() {
    // 4095 elements: thread 255 has only 15 valid (not 16)
    // because element 255 + 15*256 = 4095 is valid but
    // element 255 + 15*256 = 4095 (0-indexed) is the last one.
    // Actually: 4095 elements means indices 0..4094.
    // Thread 255: indices 255, 511, 767, ..., 255+15*256=4095 — INVALID (>= 4095)
    // So thread 255 has 15 valid elements, not 16.
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(TILE_SIZE - 1, 104));
}

#[test]
fn tile_plus_tg_width() {
    // 4096 + 256: partial tile has exactly 256 elements (1 per thread, 1 element each)
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(TILE_SIZE + THREADS_PER_TG, 105));
}

#[test]
fn tile_256_plus_one() {
    // Beyond the fused_grid dispatch width of 256 TGs
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(TILE_SIZE * 256 + 1, 106));
}

#[test]
fn tile_prime_4099() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(4099, 107));
}

#[test]
fn tile_prime_8191() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(8191, 108));
}

#[test]
fn tile_prime_16381() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(16381, 109));
}
```

**Parametric sweep** (additional coverage without 10 more named tests):

```rust
#[test]
fn tile_boundary_sweep() {
    let mut s = GpuSorter::new().unwrap();
    for (size, label) in tile_boundary_sizes() {
        if size <= 4_000_000 {  // keep sweep fast
            let data = random_data(size, size as u64);
            gen_sort_verify(&mut s, data);
            // If it fails, the panic message from gpu_sort_and_verify
            // includes n, so we know which size broke.
        }
    }
}
```

### 3.3 Category 2: Adversarial Bit Patterns (P0, 12 tests)

Each test targets a specific kernel path.

**MSD scatter paths** (kernel 3): The atomic scatter writes `dst[tile_base[d] + sg_prefix[...] + within_sg]`. When one digit `d` gets all `n` elements, `tile_base[d]` must equal 0 and `sg_prefix` must accumulate correctly across all tiles.

**Inner fused paths** (kernel 4): When all inner bytes are identical (e.g., `0x00`), the inner sort's pass-0 histogram has `bkt_hist[0] = count` and `bkt_hist[1..255] = 0`. The prefix sum is trivially `[0, count, count, ...]`. Every scatter writes to `glb_pfx[0] + run_pfx[0] + rank`, effectively a no-scatter (identity permutation). This must preserve data.

```rust
#[test]
fn pattern_same_msd_byte() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, same_msd_byte(1_000_000, 0xAA, 200));
}

#[test]
fn pattern_same_msd_byte_small() {
    // Small enough to fit in 1 tile for the single bucket
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, same_msd_byte(100, 0x42, 201));
}

#[test]
fn pattern_two_msd_buckets() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, two_msd_buckets(1_000_000, 202));
}

#[test]
fn pattern_uniform_msd() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, uniform_msd(1_000_000, 203));
}

#[test]
fn pattern_zero_inner_bytes() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, zero_inner_bytes(1_000_000, 204));
}

#[test]
fn pattern_all_max() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, all_identical(1_000_000, 0xFFFF_FFFF));
}

#[test]
fn pattern_all_zero() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, all_identical(1_000_000, 0));
}

#[test]
fn pattern_alternating_zero_max() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, alternating_zero_max(1_000_000));
}

#[test]
fn pattern_powers_of_two() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, powers_of_two(1_000_000));
}

#[test]
fn pattern_near_duplicates() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, near_duplicates(1_000_000, 205));
}

#[test]
fn pattern_sequential_msd() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, sequential_msd(1_000_000, 206));
}

#[test]
fn pattern_bit24_boundary() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, bit24_boundary(1_000_000));
}

#[test]
fn pattern_few_unique_3() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, few_unique(1_000_000, 3, 207));
}

#[test]
fn pattern_nearly_sorted() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, nearly_sorted(1_000_000, 0.01, 208));
}
```

### 3.4 Category 3: Determinism Verification (P1, 3 tests)

GPU atomics have non-deterministic ordering. We verify the sort output is always
the unique correct sorted sequence, not just "some sorted permutation."

```rust
#[test]
fn determinism_10_runs_random() {
    let mut s = GpuSorter::new().unwrap();
    let template = random_data(1_000_000, 300);
    let mut reference: Option<Vec<u32>> = None;

    for _ in 0..10 {
        let mut data = template.clone();
        s.sort_u32(&mut data).unwrap();

        match &reference {
            None => reference = Some(data),
            Some(ref_data) => {
                assert_eq!(
                    &data, ref_data,
                    "Non-deterministic output detected across runs"
                );
            }
        }
    }
}

#[test]
fn determinism_10_runs_adversarial() {
    // Use same_msd_byte which causes extreme atomic contention
    let mut s = GpuSorter::new().unwrap();
    let template = same_msd_byte(1_000_000, 0x42, 301);
    let mut reference: Option<Vec<u32>> = None;

    for _ in 0..10 {
        let mut data = template.clone();
        s.sort_u32(&mut data).unwrap();

        match &reference {
            None => reference = Some(data),
            Some(ref_data) => {
                assert_eq!(
                    &data, ref_data,
                    "Non-deterministic output on adversarial pattern"
                );
            }
        }
    }
}

#[test]
fn determinism_seeded_battery() {
    // 100 different seeded random arrays, all must sort correctly
    let mut s = GpuSorter::new().unwrap();
    for seed in 0u64..100 {
        let data = random_data(50_000, seed + 1000);
        gen_sort_verify(&mut s, data);
    }
}
```

### 3.5 Category 4: Scale and Stress (P2, 5 tests)

Large sizes increase `num_tiles`, which increases atomic contention on global counters.
At 64M elements: `num_tiles = ceil(64_000_000 / 4096) = 15_625 tiles`.

The 128M test is `#[ignore]` because it takes ~20 seconds (CPU reference sort dominates)
and allocates 512 MB.

```rust
#[test]
fn scale_32m() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(32_000_000, 400));
}

#[test]
fn scale_64m() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(64_000_000, 401));
}

#[test]
#[ignore]  // 512 MB allocation, ~20s for CPU reference sort
fn scale_128m() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(128_000_000, 402));
}

#[test]
fn stress_rapid_small_sorts() {
    // 1000 sorts of 1K — exercises buffer reuse under rapid cycling
    let mut s = GpuSorter::new().unwrap();
    for seed in 0u64..1000 {
        let data = random_data(1_000, seed + 5000);
        gen_sort_verify(&mut s, data);
    }
}

#[test]
fn stress_alternating_sizes() {
    // Buffer capacity grows to 16M then is underutilized at 1K
    let mut s = GpuSorter::new().unwrap();
    for i in 0..5 {
        gen_sort_verify(&mut s, random_data(16_000_000, 600 + i * 2));
        gen_sort_verify(&mut s, random_data(1_000, 601 + i * 2));
    }
}
```

**How `#[ignore]` works**: Rust's test harness skips `#[ignore]` tests by default.
To include them:

```bash
# Run only ignored tests
cargo test -- --ignored

# Run all tests including ignored
cargo test -- --include-ignored
```

The 128M test is `#[ignore]` rather than feature-gated because it is a correctness
test (not a performance test) — it just takes a long time. Feature gates are reserved
for performance tests where the threshold values are hardware-dependent.

### 3.6 Category 5: Buffer Management and Reuse (P1, 6 tests)

The `ensure_buffers()` method only reallocates when `data_bytes > self.data_buf_capacity`.
After sorting 16M elements, sorting 100 elements reuses the 64 MB buffer. The `n`
parameter passed to the kernel controls bounds — but leftover data from a previous sort
sits in the buffer beyond index `n`. If any kernel reads beyond `n` without checking
`valid[e]`, that stale data could corrupt results.

```rust
#[test]
fn buffer_large_then_small() {
    // Sort 16M, then sort 100.
    // The 100-element sort must not be affected by stale 16M data in the buffer.
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(16_000_000, 500));
    gen_sort_verify(&mut s, random_data(100, 501));
}

#[test]
fn buffer_small_then_large() {
    // Reallocation from 100-element buffer to 16M.
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(100, 502));
    gen_sort_verify(&mut s, random_data(16_000_000, 503));
}

#[test]
fn buffer_same_size_different_data() {
    // Sort 1M of random A, then 1M of random B.
    // Previous sort result must not leak.
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, random_data(1_000_000, 504));
    gen_sort_verify(&mut s, random_data(1_000_000, 505));
}

#[test]
fn buffer_many_reuses() {
    // 50 sorts at the same size
    let mut s = GpuSorter::new().unwrap();
    for seed in 0u64..50 {
        gen_sort_verify(&mut s, random_data(100_000, 600 + seed));
    }
}

#[test]
fn buffer_growing_sequence() {
    // 1K -> 2K -> 4K -> ... -> 16M
    // Each triggers reallocation since size doubles.
    let mut s = GpuSorter::new().unwrap();
    let mut size = 1_000;
    let mut seed = 700u64;
    while size <= 16_000_000 {
        gen_sort_verify(&mut s, random_data(size, seed));
        size *= 2;
        seed += 1;
    }
}

#[test]
fn buffer_shrinking_sequence() {
    // 16M -> 8M -> 4M -> ... -> 1K
    // No reallocation after first sort; all reuse oversized buffer.
    let mut s = GpuSorter::new().unwrap();
    let mut size = 16_000_000;
    let mut seed = 800u64;
    while size >= 1_000 {
        gen_sort_verify(&mut s, random_data(size, seed));
        size /= 2;
        seed += 1;
    }
}
```

### 3.7 Category 6: Error Handling and Edge Cases (P2, 8 tests)

```rust
#[test]
fn edge_empty() {
    let mut s = GpuSorter::new().unwrap();
    let mut data: Vec<u32> = vec![];
    s.sort_u32(&mut data).unwrap();
    assert!(data.is_empty());
}

#[test]
fn edge_single() {
    let mut s = GpuSorter::new().unwrap();
    let mut data = vec![42u32];
    s.sort_u32(&mut data).unwrap();
    assert_eq!(data, vec![42]);
}

#[test]
fn edge_two_sorted() {
    let mut s = GpuSorter::new().unwrap();
    let mut data = vec![1u32, 2];
    s.sort_u32(&mut data).unwrap();
    assert_eq!(data, vec![1, 2]);
}

#[test]
fn edge_two_reversed() {
    let mut s = GpuSorter::new().unwrap();
    let mut data = vec![2u32, 1];
    s.sort_u32(&mut data).unwrap();
    assert_eq!(data, vec![1, 2]);
}

#[test]
fn edge_all_max() {
    let mut s = GpuSorter::new().unwrap();
    let mut data = vec![u32::MAX; 10_000];
    s.sort_u32(&mut data).unwrap();
    assert!(data.iter().all(|&v| v == u32::MAX));
}

#[test]
fn edge_all_min() {
    let mut s = GpuSorter::new().unwrap();
    let mut data = vec![0u32; 10_000];
    s.sort_u32(&mut data).unwrap();
    assert!(data.iter().all(|&v| v == 0));
}

#[test]
fn edge_max_and_min() {
    let mut s = GpuSorter::new().unwrap();
    let mut data = vec![u32::MAX, 0, u32::MAX, 0, u32::MAX, 0];
    s.sort_u32(&mut data).unwrap();
    assert_eq!(data, vec![0, 0, 0, u32::MAX, u32::MAX, u32::MAX]);
}

#[test]
fn edge_pre_sorted() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, (0..100_000u32).collect());
}

#[test]
fn edge_reverse_sorted() {
    let mut s = GpuSorter::new().unwrap();
    gen_sort_verify(&mut s, (0..100_000u32).rev().collect());
}
```

### 3.8 Category 8: Concurrent Usage (P3, 2 tests)

```rust
#[test]
fn concurrent_two_sorters_sequential() {
    // Two independent GpuSorter instances, used alternately
    let mut s1 = GpuSorter::new().unwrap();
    let mut s2 = GpuSorter::new().unwrap();

    for seed in 0u64..10 {
        gen_sort_verify(&mut s1, random_data(100_000, 900 + seed * 2));
        gen_sort_verify(&mut s2, random_data(100_000, 901 + seed * 2));
    }
}

#[test]
fn concurrent_rapid_sequential() {
    // 100 sorts in tight loop from one sorter
    let mut s = GpuSorter::new().unwrap();
    for seed in 0u64..100 {
        let mut data = random_data(10_000, 1000 + seed);
        gpu_sort_and_verify(&mut s, &mut data);
    }
}
```

### 3.9 Correctness Test Count Summary

| Category | Tests | Status |
|----------|-------|--------|
| 1. Tile Boundary | 10 + 1 sweep | New |
| 2. Adversarial Patterns | 14 | New |
| 3. Determinism | 3 | New |
| 4. Scale/Stress | 5 (1 ignored) | New |
| 5. Buffer Management | 6 | New (replaces 1) |
| 6. Edge Cases | 9 | New (replaces 5) |
| 8. Concurrency | 2 | New |
| **Total** | **~50** | |

---

## 4. Property-Based Tests (`tests/property.rs`)

### 4.1 Proptest Configuration

```rust
mod common;

use common::*;
use forge_sort::GpuSorter;
use proptest::prelude::*;
use std::cell::RefCell;

// Thread-local GpuSorter to avoid creating one per proptest case.
// proptest runs cases sequentially within a single test function,
// so this is safe.
thread_local! {
    static SORTER: RefCell<GpuSorter> = RefCell::new(GpuSorter::new().unwrap());
}

fn with_sorter<F: FnOnce(&mut GpuSorter)>(f: F) {
    SORTER.with(|s| f(&mut s.borrow_mut()));
}
```

### 4.2 Custom Strategies

```rust
/// Strategy: random Vec<u32> of size 0..max_n
fn arb_u32_vec(max_n: usize) -> impl Strategy<Value = Vec<u32>> {
    prop::collection::vec(any::<u32>(), 0..max_n)
}

/// Strategy: Vec<u32> where all elements share the same MSD byte
fn arb_same_msd(max_n: usize) -> impl Strategy<Value = Vec<u32>> {
    (any::<u8>(), prop::collection::vec(0u32..0x00FF_FFFF, 1..max_n))
        .prop_map(|(msd, lowers)| {
            lowers.into_iter()
                .map(|low| ((msd as u32) << 24) | low)
                .collect()
        })
}

/// Strategy: Vec<u32> with exactly k unique values
fn arb_few_unique(max_n: usize, max_k: usize) -> impl Strategy<Value = Vec<u32>> {
    (1..max_k + 1)
        .prop_flat_map(move |k| {
            (
                prop::collection::vec(any::<u32>(), k..=k),
                prop::collection::vec(0..k, 1..max_n),
            )
        })
        .prop_map(|(uniques, indices)| {
            indices.into_iter().map(|i| uniques[i]).collect()
        })
}

/// Strategy: tile-boundary aligned sizes with random data
fn arb_tile_boundary() -> impl Strategy<Value = Vec<u32>> {
    prop_oneof![
        Just(TILE_SIZE),
        Just(TILE_SIZE + 1),
        Just(TILE_SIZE - 1),
        Just(TILE_SIZE * 2),
        Just(TILE_SIZE * 2 + 1),
        Just(TILE_SIZE * 2 - 1),
        Just(TILE_SIZE + THREADS_PER_TG),
        Just(TILE_SIZE + 1),
        Just(THREADS_PER_TG),
        Just(THREADS_PER_TG + 1),
        Just(THREADS_PER_TG - 1),
        Just(1),
        Just(2),
        Just(3),
    ]
    .prop_flat_map(|n| prop::collection::vec(any::<u32>(), n..=n))
}
```

### 4.3 Properties to Verify

The sorting function must satisfy these properties for any input:

1. **Output is sorted**: `output[i] <= output[i+1]` for all `i`.
2. **Output is a permutation**: same multiset of elements as input. Verified by sorting both CPU-side and comparing.
3. **Length preserved**: `output.len() == input.len()`.
4. **Idempotent**: sorting an already-sorted array produces the same array.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_sorted_output(mut data in arb_u32_vec(100_000)) {
        with_sorter(|s| {
            s.sort_u32(&mut data).unwrap();
            // Property 1: output is sorted
            for i in 1..data.len() {
                prop_assert!(
                    data[i - 1] <= data[i],
                    "Not sorted at index {}: {} > {}",
                    i, data[i - 1], data[i]
                );
            }
        });
    }

    #[test]
    fn prop_permutation(data in arb_u32_vec(100_000)) {
        with_sorter(|s| {
            let mut actual = data.clone();
            s.sort_u32(&mut actual).unwrap();
            // Property 2+3: same elements, same length
            let mut expected = data;
            expected.sort_unstable();
            prop_assert_eq!(actual, expected);
        });
    }

    #[test]
    fn prop_idempotent(data in arb_u32_vec(50_000)) {
        with_sorter(|s| {
            let mut first = data.clone();
            s.sort_u32(&mut first).unwrap();
            let snapshot = first.clone();
            // Property 4: sorting again yields same result
            s.sort_u32(&mut first).unwrap();
            prop_assert_eq!(first, snapshot);
        });
    }

    #[test]
    fn prop_same_msd_correctness(mut data in arb_same_msd(50_000)) {
        with_sorter(|s| {
            let mut expected = data.clone();
            expected.sort_unstable();
            s.sort_u32(&mut data).unwrap();
            prop_assert_eq!(data, expected);
        });
    }

    #[test]
    fn prop_few_unique_correctness(mut data in arb_few_unique(50_000, 10)) {
        with_sorter(|s| {
            let mut expected = data.clone();
            expected.sort_unstable();
            s.sort_u32(&mut data).unwrap();
            prop_assert_eq!(data, expected);
        });
    }

    #[test]
    fn prop_tile_boundary_correctness(mut data in arb_tile_boundary()) {
        with_sorter(|s| {
            let mut expected = data.clone();
            expected.sort_unstable();
            s.sort_u32(&mut data).unwrap();
            prop_assert_eq!(data, expected);
        });
    }
}
```

### 4.4 Proptest Configuration Rationale

- **200 cases per property** (not the default 256): Each case invokes the GPU, which
  takes ~0.5-5ms. 200 cases * 6 properties * ~2ms average = ~2.4 seconds total. Fast
  enough for CI.
- **max_n = 100,000**: Large enough to exercise multi-tile behavior (24+ tiles) without
  making CPU reference sorts slow.
- **Thread-local `GpuSorter`**: proptest generates and runs cases sequentially within one
  `#[test]` function. Creating a new `GpuSorter` per case would add ~5ms PSO compilation
  overhead * 200 = 1 second wasted. The thread-local approach amortizes init to once.
- **Shrinking**: proptest automatically shrinks failing inputs to minimal reproduction cases.
  For a Vec<u32>, this means finding the shortest vector that triggers the bug — very
  valuable for GPU sort failures which typically depend on specific element counts or values.

### 4.5 Proptest Failure Persistence

Proptest writes failing seeds to `proptest-regressions/` files. These should be committed
to the repo so regressions stay caught:

```
# .gitignore — do NOT ignore proptest regressions
# proptest-regressions/  <-- leave this out
```

---

## 5. Performance Regression Tests (`tests/performance.rs`)

### 5.1 Feature Gate Mechanism

The entire file is gated:

```rust
#![cfg(feature = "perf-test")]

mod common;

use common::*;
use forge_sort::GpuSorter;
```

This means `cargo test` compiles and runs zero performance tests. Only
`cargo test --features perf-test` enables them. This is critical because:

1. Performance thresholds are hardware-specific (M4 Pro baseline numbers).
2. CI runners may thermal-throttle, causing false failures.
3. Performance tests need warmup and multiple runs, making them slow.

### 5.2 Threshold Design

Two threshold types, as specified in PM Q&A:

**Hard thresholds**: absolute millisecond bounds. If the sort takes longer than this,
something is fundamentally broken (e.g., sort is running on CPU, kernel is not launching).

**Relative thresholds**: Mk/s floor. Catches 2x regressions while allowing normal
machine-to-machine variance.

```rust
// Hard thresholds (generous — 3x current best)
const HARD_16M_MS: f64 = 18.0;      // Currently ~5.6ms, fail at 18ms
const HARD_1M_MS: f64 = 6.0;        // Currently ~1.5ms, fail at 6ms

// Relative thresholds (Mk/s floor — catches 2x regression)
const FLOOR_16M_MKEYS: f64 = 2000.0;  // Currently ~2800, fail below 2000
const FLOOR_1M_MKEYS: f64 = 500.0;    // Currently ~650, fail below 500

const WARMUP: usize = 5;
const RUNS: usize = 15;
```

### 5.3 Performance Tests

```rust
#[test]
fn perf_16m_hard_threshold() {
    let data = random_data(16_000_000, 10000);
    let mut s = GpuSorter::new().unwrap();
    let result = measure_sort_perf(&mut s, &data, WARMUP, RUNS);

    assert!(
        result.median_ms < HARD_16M_MS,
        "16M sort took {:.2}ms median (hard limit: {}ms). \
         Throughput: {:.0} Mk/s",
        result.median_ms, HARD_16M_MS, result.mkeys_per_sec
    );
}

#[test]
fn perf_16m_throughput_floor() {
    let data = random_data(16_000_000, 10001);
    let mut s = GpuSorter::new().unwrap();
    let result = measure_sort_perf(&mut s, &data, WARMUP, RUNS);

    assert!(
        result.mkeys_per_sec > FLOOR_16M_MKEYS,
        "16M throughput {:.0} Mk/s below floor {} Mk/s. \
         Median: {:.2}ms",
        result.mkeys_per_sec, FLOOR_16M_MKEYS, result.median_ms
    );
}

#[test]
fn perf_1m_hard_threshold() {
    let data = random_data(1_000_000, 10002);
    let mut s = GpuSorter::new().unwrap();
    let result = measure_sort_perf(&mut s, &data, WARMUP, RUNS);

    assert!(
        result.median_ms < HARD_1M_MS,
        "1M sort took {:.2}ms median (hard limit: {}ms)",
        result.median_ms, HARD_1M_MS
    );
}

#[test]
fn perf_scaling_linearity() {
    let data_4m = random_data(4_000_000, 10003);
    let data_16m = random_data(16_000_000, 10004);
    let mut s = GpuSorter::new().unwrap();

    let r4 = measure_sort_perf(&mut s, &data_4m, WARMUP, RUNS);
    let r16 = measure_sort_perf(&mut s, &data_16m, WARMUP, RUNS);

    let ratio = r16.median_ms / r4.median_ms;
    // 4x data should take less than 6x time (sublinear due to fixed overhead)
    assert!(
        ratio < 6.0,
        "Scaling ratio {:.2}x (16M/4M) exceeds 6.0x. \
         4M: {:.2}ms, 16M: {:.2}ms",
        ratio, r4.median_ms, r16.median_ms
    );
}

#[test]
fn perf_cold_start_penalty() {
    let data = random_data(1_000_000, 10005);
    let mut s = GpuSorter::new().unwrap();

    // Cold sort (first ever)
    let mut cold_data = data.clone();
    let cold_start = std::time::Instant::now();
    s.sort_u32(&mut cold_data).unwrap();
    let cold_ms = cold_start.elapsed().as_secs_f64() * 1000.0;

    // Warm sort (buffers allocated, GPU warmed)
    let warm = measure_sort_perf(&mut s, &data, 3, 10);

    let penalty = cold_ms / warm.median_ms;
    assert!(
        penalty < 5.0,
        "Cold start penalty {:.1}x exceeds 5x. \
         Cold: {:.2}ms, Warm median: {:.2}ms",
        penalty, cold_ms, warm.median_ms
    );
}
```

---

## 6. Adversarial Bit Pattern Generation: Technical Deep Dive

### 6.1 Per-Byte-Position Targeting

The 4-dispatch architecture partitions a 32-bit key across byte positions:

| Pass | Byte Position | Bits | Kernel |
|------|---------------|------|--------|
| MSD | 3 (MSB) | 24:31 | sort_msd_histogram + sort_msd_atomic_scatter |
| Inner 0 | 0 (LSB) | 0:7 | sort_inner_fused (pass=0) |
| Inner 1 | 1 | 8:15 | sort_inner_fused (pass=1) |
| Inner 2 | 2 | 16:23 | sort_inner_fused (pass=2) |

For each byte position, we need inputs that create:

1. **All-same**: every element has the same value at that byte. Tests the
   single-bucket-gets-everything path.
2. **Two-value**: binary split. Tests extreme skew.
3. **All-different**: 256 unique values at that byte. Tests maximum bin spread.

Constructing these programmatically:

```rust
/// Generate data where byte `byte_pos` (0=LSB, 3=MSB) is constant
/// and all other bytes are random.
pub fn constant_byte(n: usize, byte_pos: u8, value: u8, seed: u64) -> Vec<u32> {
    let mut rng = seeded_rng(seed);
    let mask = !(0xFFu32 << (byte_pos * 8));  // clear target byte
    let fixed = (value as u32) << (byte_pos * 8);  // set target byte
    (0..n)
        .map(|_| (rng.gen::<u32>() & mask) | fixed)
        .collect()
}

/// Generate data targeting byte `byte_pos` with only 2 distinct values.
pub fn two_value_byte(n: usize, byte_pos: u8, v0: u8, v1: u8, seed: u64) -> Vec<u32> {
    let mut rng = seeded_rng(seed);
    let mask = !(0xFFu32 << (byte_pos * 8));
    (0..n)
        .map(|i| {
            let target = if i % 2 == 0 { v0 } else { v1 };
            let fixed = (target as u32) << (byte_pos * 8);
            (rng.gen::<u32>() & mask) | fixed
        })
        .collect()
}
```

### 6.2 Degenerate Distribution Matrix

The complete adversarial test matrix for 4 byte positions x 3 distribution types
= 12 specific patterns. The tests in Section 3.3 cover the most critical subset
(MSD-focused). A parametric sweep covers the rest:

```rust
#[test]
fn adversarial_byte_position_sweep() {
    let mut s = GpuSorter::new().unwrap();
    let n = 100_000;

    for byte_pos in 0u8..4 {
        // All-same at this byte position
        gen_sort_verify(&mut s, constant_byte(n, byte_pos, 0x42, byte_pos as u64 * 100));

        // Two-value at this byte position (0x00 and 0xFF)
        gen_sort_verify(&mut s, two_value_byte(
            n, byte_pos, 0x00, 0xFF, byte_pos as u64 * 100 + 1
        ));

        // 256 distinct values at this byte position
        // (use sequential_msd adjusted for byte position)
        let mask = !(0xFFu32 << (byte_pos * 8));
        let mut rng = seeded_rng(byte_pos as u64 * 100 + 2);
        let data: Vec<u32> = (0..n)
            .map(|i| {
                let target = (i as u32 % 256) << (byte_pos * 8);
                (rng.gen::<u32>() & mask) | target
            })
            .collect();
        gen_sort_verify(&mut s, data);
    }
}
```

---

## 7. Error Injection Testing

### 7.1 Approach

GPU error injection is extremely limited compared to CPU testing. We cannot corrupt
shader execution mid-flight. The viable approaches are:

**7.1.1 Input Boundary Stress**

Force the kernel to handle the most extreme valid inputs. These are not "errors" but
inputs that are one step away from triggering bugs:

- `n = 2`: smallest non-trivial sort. MSD histogram has 1-2 tiles. Inner fused has
  buckets with 0-2 elements.
- `n = TILE_SIZE * 65535`: near-maximum `num_tiles`. The `gid` value reaches 65534,
  which is near the u16 range (though Metal uses u32 for `threadgroup_position_in_grid`).

**7.1.2 Pre-Corruption Detection**

Sort the same data twice. If the GPU has memory corruption (e.g., from a previous failed
sort or buffer overflow), the two results will differ:

```rust
#[test]
fn error_double_sort_consistency() {
    let mut s = GpuSorter::new().unwrap();
    let template = random_data(1_000_000, 9000);

    let mut run1 = template.clone();
    s.sort_u32(&mut run1).unwrap();

    let mut run2 = template.clone();
    s.sort_u32(&mut run2).unwrap();

    assert_eq!(run1, run2, "Double sort produced different results — \
        possible GPU memory corruption");
}
```

**7.1.3 GpuSorter Creation Validation**

The library should gracefully handle environments where Metal is available:

```rust
#[test]
fn error_sorter_creation() {
    // On any machine with Metal (all Apple Silicon), this must succeed.
    let sorter = GpuSorter::new();
    assert!(sorter.is_ok(), "GpuSorter::new() failed: {:?}", sorter.err());
}
```

**7.1.4 Post-Sort Invariant Checking**

Rather than injecting errors, we verify invariants that would catch hidden errors:

```rust
/// Verify that the output is a valid sorted permutation of the input.
/// This catches: lost elements, duplicated elements, corrupted values.
pub fn verify_permutation(input: &[u32], output: &[u32]) {
    assert_eq!(input.len(), output.len(), "Length mismatch");

    // Histogram comparison: O(n) space, O(n) time
    use std::collections::HashMap;
    let mut freq: HashMap<u32, i64> = HashMap::new();
    for &v in input {
        *freq.entry(v).or_insert(0) += 1;
    }
    for &v in output {
        *freq.entry(v).or_insert(0) -= 1;
    }
    for (&val, &count) in &freq {
        assert_eq!(
            count, 0,
            "Element 0x{:08X}: input has {} more than output",
            val, count
        );
    }
}
```

This is used in the proptest `prop_permutation` test and can optionally be added to
`gpu_sort_and_verify` for maximum safety (at the cost of O(n) HashMap allocation per
verification). Decision: use the cheaper `sort_unstable()` comparison in the default
helper, reserve `verify_permutation` for targeted error-detection tests.

---

## 8. Test Execution Order and GPU State Considerations

### 8.1 Rust Test Runner Behavior

Rust's default test runner executes tests **in parallel** (one thread per test) with the
number of threads equal to the number of logical CPUs. For GPU tests, this means:

- Multiple `GpuSorter` instances may submit command buffers simultaneously.
- Metal handles this correctly — each `GpuSorter` has its own `MTLCommandQueue`, and
  Metal serializes command buffers within a queue.
- Cross-queue scheduling is handled by the Metal driver.

**No special ordering is needed.** Each test creates its own `GpuSorter` and operates
independently. We deliberately do NOT use `lazy_static` or a shared global sorter for
integration tests because:

1. It would serialize all tests (since `sort_u32` takes `&mut self`).
2. Parallel execution is a better test of the library's real-world usage.
3. Buffer reuse tests explicitly need their own sorter to control reuse sequencing.

### 8.2 Thermal Throttle Mitigation

Apple Silicon GPUs throttle under sustained load. This affects performance tests
but NOT correctness tests (the sort still produces correct results, just slower).

Mitigations:

1. **Feature gate** (`perf-test`): performance tests are opt-in, never run in CI by
   default.
2. **Generous thresholds**: hard limits are 3x current best, so mild throttle does not
   cause false failures.
3. **Warmup runs**: 5 warmup iterations bring the GPU to steady-state thermal before
   measurement begins.
4. **Median, not mean**: the p50 statistic is resistant to individual outlier runs caused
   by thermal transients.

### 8.3 Test Execution Time Budget

| Test File | Test Count | Estimated Time |
|-----------|-----------|---------------|
| correctness.rs | ~50 | ~15s (parallel, dominated by 32M/64M CPU sorts) |
| property.rs | 6 functions * 200 cases | ~3s (small sizes, GPU fast) |
| performance.rs | 5 | ~8s (15 measured runs each, but only with --features) |
| **Total (default)** | **~56** | **~18s** |
| **Total (with perf)** | **~61** | **~26s** |

Well within the PM's 60-second budget.

### 8.4 Parallelism and Thread Count

For correctness tests, the default parallelism (= CPU core count) is fine. Each test
allocates its own GPU buffers. On an M4 Pro with 12 CPU cores, up to 12 tests run
simultaneously, each submitting to independent command queues.

If GPU memory becomes a concern at high parallelism (12 tests * 64M * 4 bytes = 3 GB),
limit threads:

```bash
cargo test -- --test-threads=4
```

But this is unlikely to be needed given the 48 GB unified memory on the M4 Pro.

---

## 9. Tile Boundary Calculations: Detailed Analysis

### 9.1 Kernel 1+3 (MSD histogram + scatter): Tile Structure

```
TILE_SIZE = 4096 elements
THREADS = 256 per TG
ELEMS_PER_THREAD = 16

Grid: num_tiles = ceil(n / 4096) TGs
Each TG processes one tile:
  Thread t handles elements at indices:
    base + 0*256 + t = base + t
    base + 1*256 + t = base + 256 + t
    ...
    base + 15*256 + t = base + 3840 + t

  For the LAST tile (gid = num_tiles - 1):
    base = (num_tiles - 1) * 4096
    valid_in_last = n - base
    Thread t's element e is valid iff: base + e*256 + t < n
    Equivalently: e*256 + t < valid_in_last
    Thread t has valid_count = ceil((valid_in_last - t) / 256) elements
                             = max(0, (valid_in_last - t + 255) / 256)
    Clamped to [0, 16].
```

**Critical boundary: `n % 4096 == 1`**

Only thread 0, element 0 is valid. Threads 1-255 have zero valid elements.
The histogram must count exactly 1 element. The scatter must write exactly 1 element
to the correct position. If any thread mistakenly processes a stale/padding value,
it corrupts the output.

**Critical boundary: `n % 4096 == 256`**

Threads 0-255 each have exactly 1 valid element (element 0). No thread processes
element 1 through element 15 in the last tile.

### 9.2 Kernel 4 (Inner Fused): Bucket Tile Structure

The inner fused sort operates per-bucket. Each bucket has `count` elements starting
at `offset`. Within a bucket:

```
tile_count = ceil(count / 4096)
Bucket with count=0: early return (line 257 of sort.metal)
Bucket with count=1: 1 tile, 1 valid element, 255 threads idle
Bucket with count=4096: 1 full tile
Bucket with count=4097: 2 tiles, second has 1 valid element
```

The fused_grid dispatches 256 TGs (one per MSD bucket). Each TG processes all tiles
for its bucket serially in a `for (t = 0; t < tile_count; t++)` loop. This means
the number of tiles per bucket is unbounded (limited only by bucket count).

**Worst case for inner sort**: all `n` elements land in 1 MSD bucket (all same MSD byte).
That bucket has `tile_count = ceil(n / 4096)` tiles, processed by a single TG of 256
threads. At `n = 64M`, that is 15,625 tiles processed sequentially by one TG. This is
the test `pattern_same_msd_byte`.

### 9.3 Size Formulas for Targeted Testing

```rust
/// Sizes that exercise specific partial-tile conditions.
fn diagnostic_sizes() -> Vec<(usize, &'static str)> {
    let t = TILE_SIZE;          // 4096
    let tg = THREADS_PER_TG;   // 256
    let sg = 32usize;           // SIMD group width on Apple Silicon

    vec![
        // Partial tile: exactly 1 element in last tile
        (t + 1, "1 valid in partial tile"),
        (t * 2 + 1, "1 valid in last of 3 tiles"),
        // Partial tile: exactly 1 SIMD group in last tile
        (t + sg, "32 valid in partial tile"),
        (t + sg - 1, "31 valid (partial SIMD group)"),
        // Partial tile: exactly 1 TG-width in last tile
        (t + tg, "256 valid in partial (1 elem per thread)"),
        // Partial tile: max valid minus 1
        (t * 2 - 1, "4095 valid in second tile"),
        // Inner sort: bucket size hits tile boundaries
        // With n=4096 and all same MSD byte: 1 full bucket tile
        // With n=4097 and all same MSD byte: 2 bucket tiles, 1 valid in second
        // These are tested via pattern_same_msd_byte_small
    ]
}
```

---

## 10. Summary of Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RNG | ChaCha8Rng (seeded) | Reproducible across platforms, fast |
| Property testing | proptest 1.4 | Shrinking, persistence, rich strategies |
| Perf gating | `#[cfg(feature = "perf-test")]` | Avoids CI flakes from thermal/hardware variance |
| Large test gating | `#[ignore]` | Correctness (not perf) — just slow |
| Shared sorter | Per-test (integration), thread-local (proptest) | Isolation for integration, efficiency for proptest |
| Verification method | CPU `sort_unstable()` comparison | O(n log n) but simple, correct, and fast enough |
| Test parallelism | Default (= CPU cores) | Each test has independent GPU state |
| Threshold style | Both hard (ms) and relative (Mk/s) | Hard catches catastrophic regression, relative catches gradual |
| Perf warmup | 5 runs | Sufficient for GPU steady-state on M4 Pro |
| Perf measurement | 15 runs, p50 | Median resists outliers from thermal transients |
| Max normal size | 64M | 256 MB allocation, ~8s CPU sort — tolerable |
| Max ignored size | 128M | 512 MB, ~20s — opt-in only |
| proptest cases | 200 per property | ~2.4s total GPU time, sufficient statistical coverage |
| Error injection | Not viable for GPU | Use invariant checking + double-sort consistency instead |
