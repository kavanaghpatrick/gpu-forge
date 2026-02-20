# forge-sort

The fastest GPU radix sort on Apple Silicon. **29x faster** than `sort_unstable()`, **8x faster** than Rayon parallel sort, and **8x faster** than the next best published Apple Silicon implementation.

```rust
use forge_sort::GpuSorter;

let mut sorter = GpuSorter::new()?;
sorter.sort_u32(&mut data)?;
```

## Benchmarks

Measured on Apple M4 Pro (20-core GPU, 48GB unified memory). All numbers are median of 10 runs on random data.

### sort_u32 (includes memcpy)

| Size | `sort_unstable` | `rayon par_sort` | **forge-sort** | vs CPU | vs Rayon |
|-----:|----------------:|-----------------:|---------------:|-------:|---------:|
| 100K | 0.95 ms / 105 Mk/s | 0.41 ms / 245 Mk/s | **0.80 ms / 125 Mk/s** | 1.2x | 0.5x |
| 1M | 8.82 ms / 113 Mk/s | 2.67 ms / 374 Mk/s | **1.20 ms / 836 Mk/s** | 7.4x | 2.2x |
| 4M | 38.57 ms / 104 Mk/s | 11.35 ms / 353 Mk/s | **1.77 ms / 2,255 Mk/s** | 21.7x | 6.4x |
| 16M | 168.19 ms / 95 Mk/s | 44.08 ms / 363 Mk/s | **5.75 ms / 2,784 Mk/s** | 29.3x | 7.7x |

### sort_buffer (zero-copy, pure GPU speed)

When data is already in a Metal buffer, skip the memcpy for up to **1.8x more throughput**:

| Size | sort_u32 (w/ memcpy) | sort_buffer (zero-copy) | Speedup |
|-----:|---------------------:|------------------------:|--------:|
| 100K | 428 Mk/s | 428 Mk/s | 1.00x |
| 1M | 2,510 Mk/s | 3,432 Mk/s | 1.37x |
| 4M | 3,097 Mk/s | **5,242 Mk/s** | **1.69x** |
| 16M | 2,693 Mk/s | **4,131 Mk/s** | 1.53x |

GPU wins above ~200K elements. Sweet spot is 4M where data fits in the System Level Cache.

### Multi-type performance (all 6 types @ 16M elements)

| Type | sort (w/ memcpy) | sort_buffer (zero-copy) | argsort | sort_pairs |
|------|----------------:|------------------------:|--------:|-----------:|
| **u32** | **2,802 Mk/s** | **4,131 Mk/s** | 1,147 Mk/s | 882 Mk/s |
| **i32** | 2,208 Mk/s | 3,255 Mk/s | 1,101 Mk/s | 804 Mk/s |
| **f32** | 2,188 Mk/s | 3,286 Mk/s | 1,118 Mk/s | 800 Mk/s |
| **u64** | 918 Mk/s | 1,174 Mk/s | 536 Mk/s | 450 Mk/s |
| **i64** | 811 Mk/s | 1,034 Mk/s | 490 Mk/s | 424 Mk/s |
| **f64** | 813 Mk/s | 1,025 Mk/s | 503 Mk/s | - |

32-bit types sort at ~2,200-2,800 Mk/s. 64-bit types sort at ~800-900 Mk/s (2x data, ~3x slower due to halved tile size + extra passes). Float types match integer performance with uniform bit distribution. Narrow-range floats (e.g. all values in [-1000, 1000]) are slower due to MSD bucket imbalance.

### vs other implementations

| Implementation | Platform | 1M Mk/s | 16M Mk/s |
|---------------|----------|--------:|----------:|
| **forge-sort** (zero-copy) | **M4 Pro (Metal)** | **3,432** | **4,131** |
| **forge-sort** (sort_u32) | **M4 Pro (Metal)** | **836** | **2,784** |
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

### Simple (sort_u32)

Copies data to GPU buffer and back. Easiest to use:

```rust
use forge_sort::{GpuSorter, SortError};

fn main() -> Result<(), SortError> {
    let mut sorter = GpuSorter::new()?;

    let mut data: Vec<u32> = vec![5, 3, 8, 1, 9, 2, 7, 4, 6, 0];
    sorter.sort_u32(&mut data)?;
    assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    Ok(())
}
```

### Zero-copy (sort_buffer)

Write directly into GPU-visible memory. No memcpy, maximum throughput:

```rust
use forge_sort::{GpuSorter, SortError};

fn main() -> Result<(), SortError> {
    let mut sorter = GpuSorter::new()?;

    // Allocate a buffer in unified memory (shared CPU/GPU)
    let mut buf = sorter.alloc_sort_buffer(16_000_000);

    // Write data directly into GPU-visible memory
    let slice = buf.as_mut_slice();
    for i in 0..16_000_000 {
        slice[i] = (16_000_000 - i) as u32;
    }
    buf.set_len(16_000_000);

    // Sort in-place — zero memcpy
    sorter.sort_buffer(&buf)?;

    // Read sorted results directly from GPU memory
    assert_eq!(buf.as_slice()[0], 1);

    Ok(())
}
```

## API

```rust
pub struct GpuSorter { /* ... */ }
pub struct SortBuffer<T> { /* ... */ }

impl GpuSorter {
    pub fn new() -> Result<Self, SortError>;

    // Sort slices in-place (copies data to/from GPU)
    pub fn sort_u32(&mut self, data: &mut [u32]) -> Result<(), SortError>;
    pub fn sort_i32(&mut self, data: &mut [i32]) -> Result<(), SortError>;
    pub fn sort_f32(&mut self, data: &mut [f32]) -> Result<(), SortError>;
    pub fn sort_u64(&mut self, data: &mut [u64]) -> Result<(), SortError>;
    pub fn sort_i64(&mut self, data: &mut [i64]) -> Result<(), SortError>;
    pub fn sort_f64(&mut self, data: &mut [f64]) -> Result<(), SortError>;

    // Zero-copy sort (data stays in GPU memory)
    pub fn sort_buffer(&mut self, buf: &SortBuffer<u32>) -> Result<(), SortError>;
    pub fn sort_f32_buffer(&mut self, buf: &SortBuffer<f32>) -> Result<(), SortError>;
    pub fn sort_i32_buffer(&mut self, buf: &SortBuffer<i32>) -> Result<(), SortError>;
    pub fn sort_u64_buffer(&mut self, buf: &SortBuffer<u64>) -> Result<(), SortError>;
    pub fn sort_i64_buffer(&mut self, buf: &SortBuffer<i64>) -> Result<(), SortError>;
    pub fn sort_f64_buffer(&mut self, buf: &SortBuffer<f64>) -> Result<(), SortError>;

    // Argsort -- returns index permutation
    pub fn argsort_u32(&mut self, data: &[u32]) -> Result<Vec<u32>, SortError>;
    pub fn argsort_i32(&mut self, data: &[i32]) -> Result<Vec<u32>, SortError>;
    pub fn argsort_f32(&mut self, data: &[f32]) -> Result<Vec<u32>, SortError>;
    pub fn argsort_u64(&mut self, data: &[u64]) -> Result<Vec<u32>, SortError>;
    pub fn argsort_i64(&mut self, data: &[i64]) -> Result<Vec<u32>, SortError>;
    pub fn argsort_f64(&mut self, data: &[f64]) -> Result<Vec<u32>, SortError>;

    // Key-value pair sort -- co-sorts values by key order
    pub fn sort_pairs_u32(&mut self, keys: &mut [u32], values: &mut [u32]) -> Result<(), SortError>;
    pub fn sort_pairs_i32(&mut self, keys: &mut [i32], values: &mut [u32]) -> Result<(), SortError>;
    pub fn sort_pairs_f32(&mut self, keys: &mut [f32], values: &mut [u32]) -> Result<(), SortError>;
    pub fn sort_pairs_u64(&mut self, keys: &mut [u64], values: &mut [u32]) -> Result<(), SortError>;
    pub fn sort_pairs_i64(&mut self, keys: &mut [i64], values: &mut [u32]) -> Result<(), SortError>;

    // Buffer allocation
    pub fn alloc_sort_buffer<T>(&self, capacity: usize) -> SortBuffer<T>;
}
```

## Running benchmarks

```bash
cargo bench -p forge-sort
```

## Running tests

```bash
cargo test -p forge-sort -- --test-threads=1      # all 165 tests (serial for GPU)
cargo test -p forge-sort --release -- --test-threads=1  # faster
cargo test -p forge-sort -- --nocapture --test-threads=1  # see output
```

165 tests covering all 6 data types (u32/i32/f32/u64/i64/f64), argsort, sort_pairs, 8 sizes (100 to 16M), edge cases (empty, single, all-same, pre-sorted, reverse, non-aligned), buffer reuse, zero-copy SortBuffer API, interleaved usage, and performance sanity.

## Architecture

```
forge-sort/
  build.rs              # compiles sort.metal with -std=metal3.2
  shaders/sort.metal    # 4 Metal compute kernels (~870 lines)
  src/lib.rs            # GpuSorter + SortBuffer (~3,300 lines)
  tests/correctness.rs  # 165 integration tests
  benches/sort_benchmark.rs
```

Part of the `metal-forge-compute` workspace. Depends on `forge-primitives` for Metal device initialization, buffer allocation, and PSO caching.

## Memory usage

| Elements | GPU buffers | Total |
|---------:|------------:|------:|
| 1M | 2 x 4 MB + 6 KB | ~8 MB |
| 16M | 2 x 64 MB + 6 KB | ~128 MB |
| 32M | 2 x 128 MB + 6 KB | ~256 MB |

Buffers are allocated via Apple Silicon unified memory (shared CPU/GPU). They grow on demand and are never shrunk, so repeated calls with the same or smaller sizes have zero allocation overhead. `sort_buffer()` only allocates scratch space (1 data-sized buffer + 6 KB metadata).

## Limitations

- **Synchronous** -- all sort methods block until the GPU completes
- **Apple Silicon only** -- requires Metal 3.2 (M1+)
- **u32 values for sort_pairs** -- value array is always u32 (use argsort for arbitrary reordering)

## License

MIT
