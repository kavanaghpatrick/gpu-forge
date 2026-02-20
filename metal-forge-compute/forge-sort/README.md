# forge-sort

The fastest GPU radix sort on Apple Silicon. **31x faster** than `sort_unstable()`, **8x faster** than Rayon parallel sort, and **10x faster** than the next best published Apple Silicon implementation.

```rust
use forge_sort::GpuSorter;

let mut sorter = GpuSorter::new()?;
sorter.sort_u32(&mut data)?;
```

## Benchmarks

Measured on Apple M4 Pro (20-core GPU, 48GB unified memory). All numbers are median of 10 runs on random `u32` data.

| Size | `sort_unstable` | `rayon par_sort` | **forge-sort** | vs CPU | vs Rayon |
|-----:|----------------:|-----------------:|---------------:|-------:|---------:|
| 100K | 1.19 ms / 84 Mk/s | 0.44 ms / 226 Mk/s | **0.78 ms / 128 Mk/s** | 1.5x | 0.6x |
| 1M | 9.01 ms / 111 Mk/s | 2.69 ms / 372 Mk/s | **0.98 ms / 1,019 Mk/s** | 9.2x | 2.7x |
| 4M | 39.57 ms / 101 Mk/s | 10.46 ms / 382 Mk/s | **2.09 ms / 1,915 Mk/s** | 18.9x | 5.0x |
| 16M | 172.15 ms / 93 Mk/s | 44.75 ms / 358 Mk/s | **5.60 ms / 2,859 Mk/s** | 30.8x | 8.0x |
| 32M | 358.39 ms / 89 Mk/s | 90.10 ms / 355 Mk/s | **11.45 ms / 2,794 Mk/s** | 31.3x | 7.9x |

GPU wins above ~200K elements. Sweet spot is 4M-16M where data fits in the System Level Cache.

### vs other implementations

| Implementation | Platform | 1M Mk/s | 16M Mk/s |
|---------------|----------|--------:|----------:|
| **forge-sort** | **M4 Pro (Metal)** | **1,019** | **2,859** |
| VSort | M4 Pro (CPU+Metal) | 99 | - |
| VkRadixSort | RTX 3070 (Vulkan) | 53 | - |
| MPS | Apple Silicon | *no sort function* | *no sort function* |

Metal Performance Shaders does not include a sort primitive. forge-sort fills that gap.

## Algorithm

MSD+fused-inner 8-bit radix sort executed in **4 GPU dispatches** within a single command encoder:

1. **MSD Histogram** -- count byte[24:31] distribution across all tiles
2. **MSD Prep** -- exclusive prefix sum + bucket descriptors
3. **Atomic MSD Scatter** -- scatter to 256 buckets via `atomic_fetch_add` (no spin-wait)
4. **Fused Inner Sort** -- 3-pass LSD (bytes 0-2) per bucket, self-contained histograms

All dispatches share one encoder with implicit device memory barriers between them. The inner sort kernel computes its own histograms for all 3 passes during its first scan, eliminating a separate precompute dispatch.

### Why it's fast

- **4 total passes** over data (1 MSD + 3 inner LSD) vs 4-8 in traditional implementations
- **Atomic scatter** replaces decoupled lookback -- zero spin-waiting, no tile status buffers
- **Fused inner kernel** -- 3 LSD passes execute within a single dispatch, keeping data in SLC
- **~22 KB threadgroup memory** -- fits M4 Pro's 32 KB budget with room to spare
- **Per-simdgroup atomic histograms** -- 8 parallel accumulators reduce TG memory contention by 8x

## Requirements

- macOS with Apple Silicon (M1 or later)
- Metal 3.2 support (for `atomic_thread_fence` with device scope)
- Rust 1.70+
- Xcode Command Line Tools (for `xcrun metal` shader compiler)

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
forge-sort = { path = "path/to/metal-forge-compute/forge-sort" }
```

```rust
use forge_sort::{GpuSorter, SortError};

fn main() -> Result<(), SortError> {
    let mut sorter = GpuSorter::new()?;

    let mut data: Vec<u32> = vec![5, 3, 8, 1, 9, 2, 7, 4, 6, 0];
    sorter.sort_u32(&mut data)?;
    assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    // GpuSorter reuses GPU buffers across calls (grow-only, never shrink)
    let mut more_data: Vec<u32> = (0..16_000_000).rev().collect();
    sorter.sort_u32(&mut more_data)?;

    Ok(())
}
```

## API

```rust
pub struct GpuSorter { /* ... */ }

impl GpuSorter {
    /// Create a new sorter. Initializes Metal device, compiles 4 GPU kernels.
    pub fn new() -> Result<Self, SortError>;

    /// Sort a u32 slice in-place on GPU.
    /// Empty and single-element inputs return immediately.
    /// GPU buffers are allocated on first call and reused across subsequent calls.
    pub fn sort_u32(&mut self, data: &mut [u32]) -> Result<(), SortError>;
}
```

## Running benchmarks

```bash
cargo bench -p forge-sort
```

## Running tests

```bash
cargo test -p forge-sort                    # all 21 tests
cargo test -p forge-sort --release          # faster (0.3s vs 12s)
cargo test -p forge-sort -- --nocapture     # see output
```

Tests cover 8 sizes (100 to 16M), edge cases (empty, single, all-same, pre-sorted, reverse, non-aligned), buffer reuse, and performance sanity.

## Architecture

```
forge-sort/
  build.rs              # compiles sort.metal with -std=metal3.2
  shaders/sort.metal    # 4 Metal compute kernels (~350 lines)
  src/lib.rs            # GpuSorter struct (~290 lines)
  tests/correctness.rs  # 18 integration tests
  benches/sort_benchmark.rs
```

Part of the `metal-forge-compute` workspace. Depends on `forge-primitives` for Metal device initialization, buffer allocation, and PSO caching.

## Memory usage

| Elements | GPU buffers | Total |
|---------:|------------:|------:|
| 1M | 2 x 4 MB + 6 KB | ~8 MB |
| 16M | 2 x 64 MB + 6 KB | ~128 MB |
| 32M | 2 x 128 MB + 6 KB | ~256 MB |

Buffers are allocated via Apple Silicon unified memory (shared CPU/GPU). They grow on demand and are never shrunk, so repeated calls with the same or smaller sizes have zero allocation overhead.

## Limitations

- **u32 keys only** -- no key-value pairs, no signed integers, no floats (planned for v2)
- **Synchronous** -- `sort_u32()` blocks until the GPU completes
- **Apple Silicon only** -- requires Metal 3.2 (M1+)
- **Wall-clock includes memcpy** -- input is copied to GPU buffer and result copied back; for maximum throughput, a future `sort_buffer()` API could accept pre-allocated Metal buffers

## License

MIT
