# forge-filter

GPU filter+compact for Apple Silicon. **6-10x faster** than Polars on numeric WHERE clauses, using Metal compute shaders with bitmap-cached scan and SIMD ballot packing.

```rust
use forge_filter::{GpuFilter, Predicate};

let mut filter = GpuFilter::new()?;
let data: Vec<u32> = (0..16_000_000).collect();
let result = filter.filter_u32(&data, &Predicate::Gt(8_000_000))?;
```

## Benchmarks (v0.2.0)

Measured on Apple M4 Pro (20-core GPU, 48GB unified memory). Polars baseline: 5.8ms @ 16M u32.

### Single-column filter @ 16M elements

| Mode | 50% sel. | vs Polars |
|------|----------|-----------|
| Ordered (u32) | 936 us | **6.2x** |
| Ordered (f32) | 920 us | **6.3x** |
| Ordered (u64) | 1604 us | **3.6x** |
| Unordered (u32) | 548 us | **10.6x** |

### Selectivity sweep (ordered, 16M u32)

| Selectivity | Time | vs Polars |
|-------------|------|-----------|
| 1% | 697 us | **8.3x** |
| 10% | 729 us | **8.0x** |
| 50% | 882 us | **6.6x** |
| 90% | 998 us | **5.8x** |
| 99% | 1021 us | **5.7x** |

### Multi-column AND @ 16M u32, 50% per-column

| Columns | Mask | Mask + Gather | vs Polars (N * 5.8ms) |
|---------|------|---------------|----------------------|
| 2-col | 1410 us | 1899 us | **3.1x** |
| 3-col | 1990 us | 2469 us | **2.4x** |

Cost per additional column: ~580 us (one bitmap scan dispatch).

### Other features @ 16M u32, 50% sel

| Feature | Time | Notes |
|---------|------|-------|
| NULL bitmap | 1319 us | +38.5% over non-nullable (validity buffer read) |
| Arrow end-to-end | 5782 us | ~1.0x Polars (copy-dominated; use FilterBuffer for max perf) |

## What's New in v0.2.0

- **Bitmap-cached ordered scan** -- SIMD ballot packing caches predicate results as 1-bit-per-element bitmap; scatter reads bitmap instead of re-evaluating predicate
- **Multi-column filter** -- `filter_multi_mask()` evaluates up to 4 columns with AND/OR logic, producing a combined `BooleanMask`
- **NULL bitmap support** -- `filter_nullable()` and `filter_mask_nullable()` exclude NULL elements via Arrow-compatible validity bitmaps
- **Arrow integration** -- `filter_arrow()` and `filter_arrow_nullable()` accept `PrimitiveArray<T>` directly (behind `arrow` feature flag)
- **BooleanMask type** -- first-class bitmap output for multi-step pipelines (`filter_mask()` + `gather()`)

## Features

- **6 numeric types**: u32, i32, f32, u64, i64, f64
- **7 predicates**: `>`, `<`, `>=`, `<=`, `==`, `!=`, `BETWEEN`
- **Compound predicates**: AND/OR with automatic BETWEEN optimization
- **Multi-column filter**: up to 4 columns with AND/OR logic (v0.2)
- **NULL handling**: validity bitmap support, NULLs always excluded (v0.2)
- **Arrow integration**: `PrimitiveArray` input/output behind feature flag (v0.2)
- **Index output**: get matching row indices for multi-column gather
- **Unordered mode**: ~50% faster via atomic scatter (for aggregation queries)
- **Zero-copy**: `FilterBuffer<T>` API for GPU-resident data pipelines
- **BooleanMask**: reusable bitmap output for mask-then-gather pipelines (v0.2)

## Requirements

- macOS with Apple Silicon (M1 or later)
- Metal 3.2 support
- Rust 1.70+
- Xcode Command Line Tools (for `xcrun metal` shader compiler)

## Usage

```toml
[dependencies]
forge-filter = "0.2"

# Optional: Arrow integration
forge-filter = { version = "0.2", features = ["arrow"] }
```

### Simple (slice in, Vec out)

```no_run
use forge_filter::{GpuFilter, Predicate};

let mut filter = GpuFilter::new().unwrap();
let data: Vec<u32> = (0..1_000_000).collect();
let result = filter.filter_u32(&data, &Predicate::Gt(500_000)).unwrap();
assert_eq!(result.len(), 499_999);
```

### Zero-copy (FilterBuffer)

```no_run
use forge_filter::{GpuFilter, Predicate};

let mut filter = GpuFilter::new().unwrap();
let data: Vec<u32> = (0..1_000_000).collect();
let mut buf = filter.alloc_filter_buffer::<u32>(1_000_000);
buf.copy_from_slice(&data);
let result = filter.filter(&buf, &Predicate::Between(200_000, 800_000)).unwrap();
let filtered = result.as_slice();
```

### Index output

```no_run
use forge_filter::{GpuFilter, Predicate};

let mut filter = GpuFilter::new().unwrap();
let data: Vec<u32> = (0..1_000_000).collect();
let mut buf = filter.alloc_filter_buffer::<u32>(1_000_000);
buf.copy_from_slice(&data);
let result = filter.filter_indices(&buf, &Predicate::Lt(100)).unwrap();
let indices: &[u32] = result.indices().unwrap();
```

### Unordered (faster for aggregation)

```no_run
use forge_filter::{GpuFilter, Predicate};

let mut filter = GpuFilter::new().unwrap();
let data: Vec<u32> = (0..1_000_000).collect();
let mut buf = filter.alloc_filter_buffer::<u32>(1_000_000);
buf.copy_from_slice(&data);
let result = filter.filter_unordered(&buf, &Predicate::Gt(0)).unwrap();
// Same elements as ordered, but in arbitrary order -- ~50% faster
```

### Multi-column filter (v0.2)

Use `filter_multi_mask()` to evaluate predicates across multiple columns and combine them with AND or OR logic.

```no_run
use forge_filter::{GpuFilter, BooleanMask, LogicOp, Predicate};

let mut filter = GpuFilter::new().unwrap();

// Two columns of data
let ages: Vec<u32> = (0..100_000).map(|i| 18 + (i % 80)).collect();
let salaries: Vec<u32> = (0..100_000).map(|i| 30_000 + i * 10).collect();

// Find rows where age > 30 AND salary > 50000
let mask: BooleanMask = filter.filter_multi_mask(
    &[&ages, &salaries],
    &[&Predicate::Gt(30u32), &Predicate::Gt(50_000u32)],
    LogicOp::And,
).unwrap();

// Use the mask to gather matching values from any column
let mut age_buf = filter.alloc_filter_buffer::<u32>(ages.len());
age_buf.copy_from_slice(&ages);
let matching_ages = filter.gather(&age_buf, &mask).unwrap();
```

### NULL handling (v0.2)

Filter columns with NULL values using Arrow-compatible validity bitmaps (packed bits, LSB-first). NULLs are always excluded from results.

```no_run
use forge_filter::{GpuFilter, Predicate};

let mut filter = GpuFilter::new().unwrap();
let data: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];

// Validity bitmap: 1 = valid, 0 = NULL. Packed LSB-first.
// 0b11110101 = elements 0,2,4,5,6,7 valid; elements 1,3 are NULL
let validity: Vec<u8> = vec![0b11110101];

let result = filter.filter_nullable(&data, &Predicate::Gt(25u32), &validity).unwrap();
// Returns [30, 50, 60, 70, 80] -- elements 1 (20) and 3 (40) excluded as NULL
```

### Arrow integration (v0.2, feature = "arrow")

Filter Arrow `PrimitiveArray` types directly. Requires the `arrow` feature.

```toml
[dependencies]
forge-filter = { version = "0.2", features = ["arrow"] }
```

```ignore
use forge_filter::{GpuFilter, ArrowFilterKey, Predicate};
use arrow_array::{UInt32Array, Array};

let mut filter = GpuFilter::new().unwrap();
let array = UInt32Array::from(vec![10, 20, 30, 40, 50]);
let result: UInt32Array = filter.filter_arrow::<u32>(&array, &Predicate::Gt(25)).unwrap();
assert_eq!(result.len(), 3); // [30, 40, 50]

// Nullable arrays -- NULLs excluded from output
let nullable = UInt32Array::from(vec![Some(10), None, Some(30), None, Some(50)]);
let result: UInt32Array = filter.filter_arrow_nullable::<u32>(&nullable, &Predicate::Gt(15)).unwrap();
assert_eq!(result.len(), 2); // [30, 50]
```

## API Reference (v0.2)

### Core types (v0.1)

| Type | Description |
|------|-------------|
| `GpuFilter` | Main entry point -- owns Metal device, command queue, PSO cache |
| `FilterBuffer<T>` | GPU-resident data buffer (page-aligned unified memory) |
| `FilterResult<T>` | Filtered output with optional indices |
| `Predicate<T>` | Comparison predicate (Gt, Lt, Ge, Le, Eq, Ne, Between, And, Or) |
| `FilterKey` | Sealed trait for GPU-filterable types (u32/i32/f32/u64/i64/f64) |
| `FilterError` | Error type |

### New in v0.2

| Type | Description |
|------|-------------|
| `BooleanMask` | Packed bitmap result from `filter_mask()` / `filter_multi_mask()` |
| `LogicOp` | `And` or `Or` -- combine multi-column predicates |
| `ArrowFilterKey` | Sealed trait mapping FilterKey to Arrow types (behind `arrow` feature) |

### New methods on `GpuFilter`

| Method | Description |
|--------|-------------|
| `filter_mask()` | Evaluate predicate, return `BooleanMask` (no scatter) |
| `gather()` | Scatter data using a precomputed `BooleanMask` |
| `filter_multi_mask()` | Multi-column predicate evaluation (up to 4 columns) |
| `filter_nullable()` | Filter with NULL exclusion via validity bitmap |
| `filter_mask_nullable()` | Like `filter_mask()` but excludes NULLs |
| `filter_arrow()` | Filter an Arrow `PrimitiveArray` (feature = "arrow") |
| `filter_arrow_nullable()` | Filter nullable Arrow array (feature = "arrow") |

## Algorithm

Bitmap-cached 3-dispatch pipeline within a single Metal command encoder:

1. **Bitmap Scan** -- evaluate predicate per element, SIMD ballot pack results into 1-bit-per-element bitmap, SIMD prefix sum, write tile totals
2. **Scan Partials** -- exclusive prefix sum of tile totals (hierarchical for >16M elements)
3. **Bitmap Scatter** -- read cached bitmap (no predicate re-evaluation), compute global write positions, scatter to output

Unordered mode uses a single dispatch with SIMD-aggregated atomics.

Multi-column mode runs one bitmap scan per column, combines bitmaps with AND/OR, then runs a single scatter pass.

## License

**Dual-licensed.**

- **Open source**: [AGPL-3.0](LICENSE) -- free for open-source projects that comply with AGPL terms.
- **Commercial**: Proprietary license available for closed-source / commercial use. Contact [kavanagh.patrick@gmail.com](mailto:kavanagh.patrick@gmail.com) for pricing.
