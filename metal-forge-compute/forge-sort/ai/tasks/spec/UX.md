# Developer Experience Analysis: forge-sort Test Suite

## Current State Assessment

The existing test suite in `tests/correctness.rs` has 21 integration tests and 3 unit tests in `src/lib.rs`. The current DX has both strengths and significant weaknesses.

### What Works Today

1. **Single-file simplicity**: A new contributor can open one file and see every test. There is no maze of test directories to navigate.
2. **Helper functions exist**: `sort_and_verify()` and `sort_and_verify_data()` eliminate boilerplate for the common case. This is the right pattern.
3. **The assertion message in `sort_and_verify_data` is good**: It reports `n` and the first differing index. This is actionable — a developer knows exactly where to look.

### What Fails Today

1. **Flat test names are unscannable**: `test_sort_1k`, `test_sort_4k`, `test_sort_16k` — when one fails in a list of 66 tests, the developer must read every name to find the category. There is no visual hierarchy in `cargo test` output.

2. **No category grouping**: All 21 tests appear in a single flat list. A developer cannot run "just the boundary tests" or "just the bit pattern tests." With 66 tests this becomes a wall of text.

3. **One-liner size tests sacrifice readability for density**:
   ```rust
   #[test] fn test_sort_1k()   { sort_and_verify(1_000); }
   #[test] fn test_sort_4k()   { sort_and_verify(4_000); }
   ```
   This saves vertical space but makes grep-for-test-name harder and prevents per-test attributes (`#[ignore]`, `#[cfg(feature)]`).

4. **The perf test has a magic number**: `50ms` with no explanation of where it came from. The comment says "expect ~3ms GPU" but the threshold is 16x looser. There is no relative threshold and no way to distinguish "GPU regression" from "memcpy regression."

5. **No test for error paths**: Only the `Display` impl is tested. No test verifies what happens when the GPU is unavailable, when the command buffer fails, or when you pass adversarial data sizes.

6. **`GpuSorter::new()` is called per-test**: This compiles shaders and creates the Metal context every time. At 21 tests this is tolerable. At 66 tests, this adds ~2-4 seconds of pure overhead. There is no shared fixture.

7. **Buffer reuse test has no assertion messages**: The three `assert_eq!` calls in `test_buffer_reuse` give no context on which phase (first sort, reuse, or shrink) failed.

---

## Decisions

### Decision 1: File Organization — Single File With Modules

**Choice**: Keep all integration tests in a single file `tests/correctness.rs`, organized with `mod` blocks. Do NOT split into multiple files.

**Rationale**: Each file in `tests/` compiles as a separate crate. For a GPU library, this means separate shader compilation and Metal context initialization per file. A single file with modules gives the same organizational benefits (filtering, grouping) without the compilation cost. Additionally, `cargo test` output already prefixes test names with the module path, so `cargo test tile_boundary` filters to just that category.

**Structure**:
```
tests/correctness.rs
  mod helpers         -- shared helpers, NOT a test module
  mod tile_boundary   -- P0: tile alignment tests
  mod bit_patterns    -- P0: adversarial bit pattern tests
  mod determinism     -- P1: multi-run reproducibility
  mod buffer_reuse    -- P1: buffer lifecycle tests
  mod scale           -- P2: large input stress tests
  mod edge_cases      -- P2: empty, single, extremes
  mod perf            -- P3: performance regression (feature-gated)
  mod concurrent      -- P3: multi-instance tests
```

**How cargo test output looks**:
```
test tile_boundary::exact_one_tile ... ok
test tile_boundary::one_tile_plus_one ... ok
test tile_boundary::one_tile_minus_one ... ok
test bit_patterns::all_same_msd_byte ... ok
test bit_patterns::two_msd_buckets_only ... ok
test determinism::ten_runs_identical_16m ... ok
test perf::throughput_floor_16m ... ok    (only with --features perf-tests)
```

A developer can immediately see which category failed. Running `cargo test tile_boundary` executes only that group.

### Decision 2: Test Naming Convention — Category::Behavior Without "test_" Prefix

**Choice**: Use `mod_name::descriptive_behavior` naming. Drop the `test_` prefix since the `#[test]` attribute already marks them. Names describe what is being verified, not the method being called.

**Current** (bad for scanning):
```
test_sort_1k
test_sort_4k
test_non_tile_aligned
test_all_zeros
```

**Proposed** (scannable, self-documenting):
```
tile_boundary::exact_one_tile_4096
tile_boundary::partial_tile_one_element_4097
tile_boundary::partial_tile_max_minus_one_4095
bit_patterns::single_msd_bucket_all_0xAA
bit_patterns::two_msd_buckets_00_and_FF
scale::random_32m
edge_cases::empty_slice
edge_cases::single_element
```

**Rules**:
- Module name is the category (what kind of test)
- Function name is the specific scenario (what makes this test unique)
- Include the size in the name when size IS the test (e.g., `exact_one_tile_4096`)
- Do not include the size when it is incidental (e.g., `single_msd_bucket_all_0xAA` — the 1M size is an implementation detail)
- Use underscores between words, not camelCase
- Avoid abbreviations except universally understood ones (MSD, LSD, u32, 1m, 16m)

### Decision 3: Helper Function Design — A Layered Verification API

**Choice**: Three layers of helpers, from high-level to diagnostic.

**Layer 1 — The one-liner** (for tests where you just need "sort this and check it"):
```rust
fn assert_sorts_correctly(data: Vec<u32>);
```
Creates a sorter, sorts, verifies against CPU reference. Used by 80% of tests. Includes actionable failure messages by default.

**Layer 2 — Sorter-reusing** (for tests that need to control the sorter lifecycle):
```rust
fn assert_sorts_correctly_with(sorter: &mut GpuSorter, data: Vec<u32>);
```
Same verification, but the caller provides the sorter. Used by buffer reuse tests, concurrent tests, and any test that sorts multiple arrays.

**Layer 3 — Diagnostic assertion** (the underlying verification with rich error messages):
```rust
fn verify_sorted(actual: &[u32], expected: &[u32], context: &str);
```
Called by layers 1 and 2. Produces detailed diagnostics on failure. The `context` parameter is a human-readable label that appears in the error message (e.g., "16M random, run 3 of 10").

**Additional helpers**:
```rust
fn gen_random_seeded(n: usize, seed: u64) -> Vec<u32>;  // Reproducible
fn gen_pattern(n: usize, f: impl Fn(usize) -> u32) -> Vec<u32>;  // Custom patterns
fn shared_sorter() -> GpuSorter;  // Lazy-initialized, one per thread
```

The `gen_random_seeded` helper is essential: when a randomized test fails, the seed appears in the error message, and the developer can reproduce the exact failure. Never use `thread_rng()` without a seed in a test that might fail.

### Decision 4: Assertion Messages — Five-Line Diagnostic on Failure

**Choice**: Every sort verification failure prints exactly five lines of actionable information.

**Current** (minimal):
```
Sort mismatch at n=4097. First diff at index 4096
```

**Proposed**:
```
SORT MISMATCH [tile_boundary::partial_tile_one_element_4097]
  context: 4097 elements, partial tile (1 element in last tile)
  n=4097, first diff at index 4096
  actual[4096]=0x00000003, expected[4096]=0x00000001
  actual[4095..4097]=[0x00000001, 0x00000003], expected[4095..4097]=[0x00000001, 0x00000001]
```

**What each line tells the developer**:
1. **Category + test name**: Which test, without scrolling up
2. **Context**: What this test is exercising, in plain English
3. **Location**: Where the first difference is (index-level)
4. **Values**: What was there vs what should be there (hex for bit patterns)
5. **Window**: A few elements around the diff for pattern recognition

**Implementation note**: The `verify_sorted` helper builds this message. The `context` string comes from the test function. Hex formatting is used because this is a radix sort — byte-level inspection matters. Decimal values would hide which radix pass failed.

### Decision 5: Performance Tests — Feature-Gated With Dual Thresholds

**Choice**: Perf tests live in `mod perf` and require `--features perf-tests` to compile. Each test has both a hard threshold (absolute wall-clock) and a relative threshold (scaling ratio).

**Cargo.toml addition**:
```toml
[features]
perf-tests = []
```

**Why feature-gated, not `#[ignore]`**: `#[ignore]` tests still compile, which means they still import timing infrastructure and increase binary size. Feature-gating with `#[cfg(feature = "perf-tests")]` eliminates them entirely from normal builds. More importantly, `cargo test` gives a clean "24 passed" instead of "24 passed, 5 ignored" — the developer does not wonder whether the ignored tests matter.

**Dual threshold example**:
```rust
#[cfg(feature = "perf-tests")]
mod perf {
    #[test]
    fn throughput_floor_16m() {
        // Hard threshold: must complete in < 10ms
        assert!(elapsed_ms < 10.0,
            "16M sort took {:.2}ms (limit: 10ms). \
             Throughput: {:.0} Mk/s (floor: 2000 Mk/s)", elapsed_ms, mkeys);

        // Relative threshold: 16M should be < 6x of 4M (sub-linear scaling)
        assert!(ratio_16m_to_4m < 6.0,
            "Scaling regression: 16M/4M ratio is {:.2}x (limit: 6x)", ratio);
    }
}
```

**Running perf tests**:
```bash
cargo test --features perf-tests -- perf    # Just perf tests
cargo test                                   # Everything except perf
```

### Decision 6: Scale Tests — `#[ignore]` for 128M, Inline for Up to 64M

**Choice**: Tests up to 64M run normally. The 128M test is `#[ignore]` because it allocates 512MB and takes significant time.

```rust
mod scale {
    #[test]
    fn random_32m() { /* runs normally */ }

    #[test]
    fn random_64m() { /* runs normally */ }

    #[test]
    #[ignore] // 512MB allocation, ~20s with CPU reference sort
    fn random_128m() { /* runs with cargo test -- --ignored */ }
}
```

**Why 64M is NOT ignored**: 256MB is well within Apple Silicon unified memory. The GPU sort itself takes <20ms. The bottleneck is the CPU reference `sort()` for verification (~2-3 seconds at 64M). This is acceptable for a correctness test suite. 128M pushes the CPU reference sort to ~6-8 seconds and the allocation to 512MB, which warrants opt-in.

### Decision 7: Sorter Initialization — Shared Per-Module, Not Per-Test

**Choice**: Each test module that runs multiple sorts shares a single `GpuSorter` instance via a module-level helper that creates one lazily.

**Problem**: `GpuSorter::new()` compiles Metal shaders. At 66 tests, that is 66 shader compilations. Even at ~50ms each, this adds 3.3 seconds of pure overhead.

**Solution**: Tests within a module that do not specifically test sorter creation/destruction share a sorter:

```rust
mod tile_boundary {
    use super::helpers::*;

    #[test]
    fn exact_one_tile_4096() {
        let mut sorter = GpuSorter::new().unwrap();
        assert_sorts_correctly_with(&mut sorter, gen_random_seeded(4096, 1));
    }
}
```

**Note**: Because `cargo test` runs tests in parallel across threads, and `GpuSorter` requires `&mut self`, each test still creates its own sorter. This is intentional — it avoids needing `Mutex<GpuSorter>` which would serialize all tests. The 66 shader compilations are parallelized by the test runner, so actual wall-clock overhead is modest (only as slow as the slowest thread's compilation). For tests that specifically verify sorter reuse (buffer management category), the test explicitly creates one sorter and calls it multiple times.

**Exception**: The `concurrent` module creates multiple sorters deliberately. The `buffer_reuse` module creates exactly one sorter and reuses it across multiple sorts within a single test function.

### Decision 8: Seeded Randomness — Every Random Test Is Reproducible

**Choice**: All tests that use random data use a fixed seed. The seed value is the test's "identity" — it appears in the function name comment and in the failure message.

**Current** (non-reproducible):
```rust
fn sort_and_verify(n: usize) {
    let mut rng = rand::thread_rng();  // Different every run
```

**Proposed**:
```rust
fn gen_random_seeded(n: usize, seed: u64) -> Vec<u32> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen()).collect()
}
```

**Seed assignment convention**: Each test uses a unique seed. Seeds are sequential within a module (tile_boundary: 100-199, bit_patterns: 200-299, etc.). This avoids accidental seed collisions and makes it trivial to reproduce any failure:

```
SORT MISMATCH [tile_boundary::exact_one_tile_4096]
  seed: 101, n=4096, first diff at index 3999
  ...
```

The developer runs `cargo test exact_one_tile_4096` and gets the same failure every time.

**The seeded random battery test** (in `determinism` module) runs 100 seeds and reports which specific seeds failed:
```
Seeded random battery: 2/100 seeds failed: [42, 77]
  Seed 42: n=10000, first diff at index 8192
  Seed 77: n=10000, first diff at index 4096
```

### Decision 9: CI vs Local Experience — Same Command, Different Depth

**Choice**: `cargo test` runs the fast, essential suite (~55 tests, <30 seconds). CI adds two extra commands for deeper coverage.

**Local development** (the only command a developer needs to remember):
```bash
cargo test
```
Runs: tile_boundary, bit_patterns, determinism, buffer_reuse, edge_cases, scale (up to 64M), concurrent. Does NOT run perf tests or 128M test.

**CI pipeline** (three commands):
```bash
cargo test                                    # Fast correctness (~55 tests, <30s)
cargo test -- --ignored                       # 128M scale test (~10s)
cargo test --features perf-tests -- perf      # Performance regression (~15s)
```

**Why three commands, not one**: A developer who runs `cargo test` and sees all green has high confidence. The CI adds the slow and environment-sensitive tests separately so that a thermal throttle on the CI runner does not block a merge that only changes documentation.

**Test output format**: Use the default `cargo test` output (not `--format json` or `--format terse`). The default output is well-understood, shows only failing test details, and produces a clean summary line:

```
test result: ok. 55 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out
```

### Decision 10: Test Discovery — A New Developer Can Understand the Suite in 60 Seconds

**Choice**: The test file opens with a structural comment that serves as a table of contents.

```rust
//! forge-sort correctness tests
//!
//! Organization (run a category with `cargo test <name>`):
//!   tile_boundary  — TILE_SIZE=4096 alignment edge cases
//!   bit_patterns   — adversarial byte distributions targeting each radix pass
//!   determinism    — multi-run reproducibility and seeded random battery
//!   buffer_reuse   — GpuSorter buffer lifecycle (grow, shrink, reuse)
//!   scale          — 32M/64M inputs (128M behind #[ignore])
//!   edge_cases     — empty, single, two-element, all-same, extremes
//!   perf           — performance thresholds (behind --features perf-tests)
//!   concurrent     — multiple GpuSorter instances
//!
//! Quick reference:
//!   cargo test                              # all fast tests (~55, <30s)
//!   cargo test tile_boundary                # just tile boundary tests
//!   cargo test -- --ignored                 # include 128M scale test
//!   cargo test --features perf-tests        # include performance tests

mod helpers { ... }
mod tile_boundary { ... }
...
```

This header answers the three questions every new developer asks:
1. "What categories of tests exist?" (the list)
2. "How do I run a subset?" (the quick reference)
3. "Why are some tests not running?" (feature gate and `#[ignore]` explanation)

---

## Complete Module Design

### Module: `helpers`

```rust
mod helpers {
    use forge_sort::GpuSorter;
    use rand::SeedableRng;
    use rand::Rng;

    /// Sort `data` on GPU and verify against CPU reference sort.
    /// Panics with detailed diagnostics on mismatch.
    pub fn assert_sorts_correctly(data: Vec<u32>) {
        let mut sorter = GpuSorter::new().unwrap();
        assert_sorts_correctly_with(&mut sorter, data);
    }

    /// Sort `data` using the provided sorter and verify.
    pub fn assert_sorts_correctly_with(sorter: &mut GpuSorter, data: Vec<u32>) {
        let mut actual = data.clone();
        let mut expected = data;
        expected.sort();
        sorter.sort_u32(&mut actual).unwrap();
        verify_sorted(&actual, &expected, "");
    }

    /// Sort `data` using the provided sorter and verify, with context label.
    pub fn assert_sorts_correctly_ctx(
        sorter: &mut GpuSorter,
        data: Vec<u32>,
        context: &str,
    ) {
        let mut actual = data.clone();
        let mut expected = data;
        expected.sort();
        sorter.sort_u32(&mut actual).unwrap();
        verify_sorted(&actual, &expected, context);
    }

    /// Core verification. Compares actual vs expected with rich diagnostics.
    pub fn verify_sorted(actual: &[u32], expected: &[u32], context: &str) {
        assert_eq!(actual.len(), expected.len(),
            "Length mismatch: actual={}, expected={} [{}]",
            actual.len(), expected.len(), context);

        if let Some(idx) = actual.iter().zip(expected.iter()).position(|(a, b)| a != b) {
            let window_start = idx.saturating_sub(2);
            let window_end = (idx + 3).min(actual.len());
            panic!(
                "\nSORT MISMATCH{}\n  \
                 n={}, first diff at index {}\n  \
                 actual[{}]=0x{:08X}, expected[{}]=0x{:08X}\n  \
                 actual[{}..{}]={:08X?}\n  \
                 expected[{}..{}]={:08X?}",
                if context.is_empty() { String::new() } else { format!(" [{}]", context) },
                actual.len(), idx,
                idx, actual[idx], idx, expected[idx],
                window_start, window_end, &actual[window_start..window_end],
                window_start, window_end, &expected[window_start..window_end],
            );
        }
    }

    /// Generate n random u32 values from a fixed seed.
    pub fn gen_random_seeded(n: usize, seed: u64) -> Vec<u32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen()).collect()
    }

    /// Generate n values from a pattern function.
    pub fn gen_pattern(n: usize, f: impl Fn(usize) -> u32) -> Vec<u32> {
        (0..n).map(f).collect()
    }
}
```

**Design rationale**:
- Three layers match three usage patterns (one-liner, reuse, diagnostic)
- `context` parameter is a free-form string, not a structured type — this keeps tests readable
- Hex formatting is deliberate: `0x00FF0100` immediately shows which byte position has the issue, mapping directly to which radix pass (MSD: byte 3, inner pass 1: byte 2, inner pass 2: byte 1, inner pass 3: byte 0)
- Window of 5 elements around the diff helps pattern recognition ("is it off by one index?" "are adjacent elements swapped?")

### Module: `tile_boundary` (9 tests)

```
tile_boundary::exact_one_tile_4096
tile_boundary::partial_one_extra_4097
tile_boundary::partial_max_minus_one_4095
tile_boundary::exact_two_tiles_8192
tile_boundary::partial_tile_one_simd_4352    (4096 + 256)
tile_boundary::partial_tile_one_thread_4097  (4096 + 1 thread)
tile_boundary::prime_4099
tile_boundary::prime_8191
tile_boundary::prime_16381
```

Each test uses `gen_random_seeded` with a unique seed, and `assert_sorts_correctly`.

### Module: `bit_patterns` (10 tests)

```
bit_patterns::single_msd_bucket_all_0xAA
bit_patterns::two_msd_buckets_00_FF
bit_patterns::uniform_256_msd_buckets
bit_patterns::identical_inner_bytes_XX000000
bit_patterns::all_bits_set_0xFFFFFFFF
bit_patterns::bit24_boundary_adjacent
bit_patterns::sequential_high_bytes
bit_patterns::power_of_two_values
bit_patterns::alternating_zero_max
bit_patterns::near_duplicates
```

Each test uses `gen_pattern` with a lambda that constructs the specific bit pattern. The test name encodes the pattern, not the assertion.

### Module: `determinism` (3 tests)

```
determinism::ten_runs_identical_1m
determinism::ten_runs_identical_16m
determinism::seeded_random_battery_100
```

The battery test iterates 100 seeds and collects ALL failures before panicking, so the developer sees the complete failure landscape, not just the first seed that broke.

### Module: `buffer_reuse` (5 tests)

```
buffer_reuse::large_then_small
buffer_reuse::small_then_large
buffer_reuse::same_size_different_data
buffer_reuse::fifty_reuses
buffer_reuse::growing_sequence
```

All tests in this module create a single `GpuSorter` and call `sort_u32` multiple times. Each `assert_sorts_correctly_ctx` call includes a context string like `"phase 2: sort 100K after 1M"`.

### Module: `scale` (4 tests)

```
scale::random_32m
scale::random_64m
scale::random_128m          (#[ignore])
scale::rapid_small_1000x
```

### Module: `edge_cases` (8 tests)

```
edge_cases::empty_slice
edge_cases::single_element
edge_cases::two_sorted
edge_cases::two_reversed
edge_cases::all_u32_max
edge_cases::all_u32_min
edge_cases::max_and_min_interleaved
edge_cases::sorter_creation
```

### Module: `perf` (4 tests, feature-gated)

```
perf::wall_clock_16m_under_10ms
perf::wall_clock_1m_under_2ms
perf::throughput_floor_2000_mkps
perf::scaling_4m_to_16m_sublinear
```

Perf test names encode the threshold so that the test list itself is a performance contract.

### Module: `concurrent` (2 tests)

```
concurrent::two_sorters_alternating
concurrent::rapid_sequential_100
```

---

## Test Count Summary

| Module | Tests | Gating |
|--------|-------|--------|
| tile_boundary | 9 | Normal |
| bit_patterns | 10 | Normal |
| determinism | 3 | Normal |
| buffer_reuse | 5 | Normal |
| scale | 4 | 1 `#[ignore]` |
| edge_cases | 8 | Normal |
| perf | 4 | `#[cfg(feature = "perf-tests")]` |
| concurrent | 2 | Normal |
| **Total integration** | **45 new** | |
| Unit tests (lib.rs) | 3 existing | Normal |
| **Grand total** | **~48** | |

Combined with the existing tests that will be reorganized into the new modules, the total will be approximately 48 tests (some existing tests merge into the new categories rather than being additive). The PM target of ~66 included keeping the old tests as-is and adding 45 new ones; the reorganized approach is more efficient because it eliminates redundancy (the old `test_sort_1k` through `test_sort_16m` are subsumed by the new tile_boundary, scale, and seeded battery tests).

If the PM requires exactly 66 tests, the seeded random battery can be split into individual test functions (one per size class), and the tile boundary module can be expanded with additional prime sizes. But test count is a vanity metric — the goal is coverage of failure modes, not a number.

---

## Dev Dependencies Addition

```toml
[dev-dependencies]
rand = "0.8"
rayon = "1.10"
```

The `rand` crate with `StdRng` and `SeedableRng` is already available via `rand = "0.8"`. No new dependencies are required for the test reorganization.

---

## Research Sources

- [Test Organization - The Rust Programming Language](https://doc.rust-lang.org/book/ch11-03-test-organization.html)
- [How to organize your Rust tests - LogRocket Blog](https://blog.logrocket.com/how-to-organize-rust-tests/)
- [Controlling How Tests Are Run - The Rust Programming Language](https://doc.rust-lang.org/book/ch11-02-running-tests.html)
- [Everything you need to know about testing in Rust - Shuttle](https://www.shuttle.dev/blog/2024/03/21/testing-in-rust)
- [Complete Guide To Testing Code In Rust - Zero To Mastery](https://zerotomastery.io/blog/complete-guide-to-testing-code-in-rust/)
- [Best Practices for Structuring Your Rust Tests - Moldstud](https://moldstud.com/articles/p-best-practices-for-structuring-your-rust-tests-a-comprehensive-guide)
- [How To Structure Unit Tests in Rust - Better Programming](https://betterprogramming.pub/how-to-structure-unit-tests-in-rust-cc4945536a32)
- [assert_eq in std - Rust](https://doc.rust-lang.org/std/macro.assert_eq.html)
- [Improve assert_eq failure message formatting - Rust RFC #1864](https://github.com/rust-lang/rfcs/issues/1864)
- [cargo test - The Cargo Book](https://doc.rust-lang.org/cargo/commands/cargo-test.html)
- [Features - The Cargo Book](https://doc.rust-lang.org/cargo/reference/features.html)
