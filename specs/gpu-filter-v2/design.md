---
spec: gpu-filter-v2
phase: design
created: 2026-02-20
---

# Technical Design: forge-filter v2

## Overview

Extend forge-filter v0.1.0 with 5 features, maintaining full backward compatibility (v0.2.0 minor bump). The ordered mode bottleneck (6.8x → 10x target) is solved by bitmap caching that eliminates the double-read in scatter.

## Architecture

### Current Pipeline (v0.1 — preserved)

```
Input → [1] predicate_scan → partials[] → [2] scan_partials → [3] scatter → Output
                                                                    ↑
                                                            re-evaluates predicate
                                                            (DOUBLE READ)
```

### v0.2 Bitmap-Cached Pipeline (new default for ordered mode)

```
Input → [1] predicate_bitmap_scan → bitmap[] + partials[]
                                         ↓
                            [2] scan_partials (unchanged)
                                         ↓
                  [3] bitmap_scatter ← reads bitmap (SLC cache hit)
                                    ← reads input ONCE
                                    → Output
```

Key change: Pass 1 writes a 1-bit-per-element bitmap via `simd_ballot` + `atomic_or`. Pass 3 reads the cached bitmap (2MB for 16M elements, fits in SLC) instead of re-evaluating the predicate. Eliminates one full read of source data.

### Multi-Column Pipeline

```
Col_A, Col_B, ..., Col_N → [1] multi_predicate_bitmap_scan → bitmap[] + partials[]
                                         ↓
                            [2] scan_partials (unchanged)
                                         ↓
                  [3] multi_scatter ← reads bitmap
                                   ← reads EACH column ONCE
                                   → Output_A, Output_B, ..., Output_N
```

## Component Design

### 1. New Metal Kernels (shaders/filter.metal)

#### `filter_bitmap_scan` — replaces `filter_predicate_scan` for ordered mode

```metal
// Function constants (same as existing + new HAS_NULLS)
constant bool HAS_NULLS [[function_constant(5)]];  // NEW: gate validity bitmap check
constant bool has_nulls = is_function_constant_defined(HAS_NULLS) ? HAS_NULLS : false;

kernel void filter_bitmap_scan(
    device const void*   src        [[buffer(0)]],
    device atomic_uint*  bitmap     [[buffer(1)]],  // NEW: 1 bit/element, packed u32
    device uint*         partials   [[buffer(2)]],
    device const uint*   validity   [[buffer(3)]],  // NEW: Arrow null bitmap (optional)
    constant FilterParams& params   [[buffer(4)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    // Each thread evaluates predicate for FILTER_ELEMS elements
    // Use simd_ballot to pack 32 results per simdgroup
    // Write packed word to bitmap via atomic_or
    // Use simd_prefix_exclusive_sum for tile count
    // Write tile count to partials[tg_id]

    threadgroup uint simd_totals[FILTER_NUM_SGS];
    uint local_count = 0;

    uint base = tg_id * tile_size;
    for (uint r = 0; r < elems_per_thread; r++) {
        uint idx = base + r * FILTER_THREADS + tid;
        if (idx >= params.element_count) break;

        bool pred_result;
        if (is_64bit) {
            // 64-bit path
            ulong val = ((device const ulong*)src)[idx];
            pred_result = evaluate_predicate_64(val, params);
        } else {
            uint val = ((device const uint*)src)[idx];
            pred_result = evaluate_predicate(val, params);
        }

        // NULL check: if has_nulls && !is_valid, force false
        if (has_nulls) {
            uint word = validity[idx / 32];
            bool valid = (word >> (idx % 32)) & 1;
            pred_result = pred_result && valid;
        }

        // simd_ballot packs 32 predicate results into one uint
        simd_vote vote = simd_ballot(pred_result);
        uint ballot_bits = uint(vote); // lower 32 bits

        // Lane 0 writes ballot word to bitmap
        if (lane_id == 0) {
            uint bitmap_idx = (base + r * FILTER_THREADS + sg_id * 32) / 32;
            atomic_store_explicit((device atomic_uint*)&bitmap[bitmap_idx],
                                 ballot_bits, memory_order_relaxed);
        }

        local_count += popcount(ballot_bits) * (lane_id == 0 ? 1 : 0);
        // Actually: each thread tracks its own bit
        local_count += pred_result ? 1 : 0;
    }

    // SIMD prefix sum + cross-SG aggregation (same pattern as v0.1)
    uint sg_sum = simd_sum(local_count);
    if (lane_id == 0) simd_totals[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute tile total
    if (tid == 0) {
        uint total = 0;
        for (uint s = 0; s < FILTER_NUM_SGS; s++) total += simd_totals[s];
        partials[tg_id] = total;
    }
}
```

#### `filter_bitmap_scatter` — replaces `filter_scatter` for ordered mode

```metal
kernel void filter_bitmap_scatter(
    device const void*  src        [[buffer(0)]],
    device const uint*  bitmap     [[buffer(1)]],  // read cached bitmap
    device const uint*  partials   [[buffer(2)]],
    device void*        output     [[buffer(3)]],
    device uint*        output_idx [[buffer(4)]],  // optional
    constant FilterParams& params  [[buffer(5)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    // Read bitmap word → check bit for this element → scatter if set
    // No predicate re-evaluation needed!
    // Use prefix sum on bitmap bits for local offset
    // Global offset from partials (already computed by scan_partials)

    uint tile_offset = partials[tg_id]; // exclusive prefix from scan_partials
    threadgroup uint simd_offsets[FILTER_NUM_SGS];

    uint base = tg_id * tile_size;
    uint running_offset = 0;

    for (uint r = 0; r < elems_per_thread; r++) {
        uint idx = base + r * FILTER_THREADS + tid;
        if (idx >= params.element_count) break;

        // Read cached bitmap (SLC hit ~95%)
        uint bitmap_word = bitmap[idx / 32];
        bool matches = (bitmap_word >> (idx % 32)) & 1;

        // SIMD prefix sum for write position
        uint local_prefix = simd_prefix_exclusive_sum(matches ? 1u : 0u);
        uint sg_total = simd_sum(matches ? 1u : 0u);

        if (lane_id == 0) simd_offsets[sg_id] = sg_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cross-SG prefix
        uint sg_prefix = 0;
        if (tid == 0) {
            for (uint s = 1; s < FILTER_NUM_SGS; s++) {
                uint tmp = simd_offsets[s];
                simd_offsets[s] = sg_prefix + simd_offsets[s-1];
                // ... (standard cross-SG prefix pattern)
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (matches) {
            uint write_pos = tile_offset + running_offset +
                           (sg_id > 0 ? simd_offsets[sg_id] : 0) + local_prefix;
            // Write value
            if (is_64bit) {
                ((device ulong*)output)[write_pos] = ((device const ulong*)src)[idx];
            } else {
                ((device uint*)output)[write_pos] = ((device const uint*)src)[idx];
            }
            if (output_idx) {
                ((device uint*)output_idx)[write_pos] = idx;
            }
        }

        // Track running offset for next round
        running_offset += simd_offsets[FILTER_NUM_SGS - 1] +
                          /* last SG total */ sg_total_last;
    }
}
```

### 2. New Rust Types

#### BooleanMask — GPU-resident bitmap result

```rust
/// GPU-resident boolean mask from a filter operation.
///
/// Represents 1 bit per element, packed into u32 words.
/// Can be reused across multiple gather/scatter operations.
pub struct BooleanMask {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,      // number of elements (bits)
    count: usize,    // number of true bits (matching elements)
}

impl BooleanMask {
    /// Number of elements in the mask.
    pub fn len(&self) -> usize { self.len }
    /// Number of matching (true) elements.
    pub fn count(&self) -> usize { self.count }
    /// Read mask bits back to CPU as Vec<bool>.
    pub fn to_vec(&self) -> Vec<bool> { ... }
}
```

#### ColumnPredicate — multi-column filter specification

```rust
/// Predicate specification for one column in a multi-column filter.
pub struct ColumnPredicate<T: FilterKey> {
    /// Column data (reference to FilterBuffer or slice).
    pub data: ColumnRef<T>,
    /// Predicate to apply.
    pub predicate: Predicate<T>,
}

/// Reference to column data for multi-column filter.
pub enum ColumnRef<'a, T: FilterKey> {
    Slice(&'a [T]),
    Buffer(&'a FilterBuffer<T>),
}
```

### 3. New GpuFilter Methods

```rust
impl GpuFilter {
    // --- v0.2 bitmap-cached ordered mode (replaces internal pipeline) ---
    // No new public API needed — existing filter_* methods use bitmap internally

    // --- Multi-column filter ---

    /// Filter multiple columns with AND-combined predicates.
    /// Returns a BooleanMask that can be used with `gather()`.
    pub fn filter_multi_mask<T: FilterKey>(
        &mut self,
        columns: &[(&[T], &Predicate<T>)],
    ) -> Result<BooleanMask, FilterError>;

    /// Gather elements from a column using a BooleanMask.
    pub fn gather<T: FilterKey>(
        &mut self,
        data: &[T],
        mask: &BooleanMask,
    ) -> Result<FilterResult<T>, FilterError>;

    // --- Arrow integration (behind `arrow` feature) ---

    #[cfg(feature = "arrow")]
    /// Filter an Arrow PrimitiveArray, returning filtered array.
    pub fn filter_arrow<T: ArrowFilterKey>(
        &mut self,
        array: &PrimitiveArray<T::ArrowType>,
        pred: &Predicate<T>,
    ) -> Result<PrimitiveArray<T::ArrowType>, FilterError>;

    #[cfg(feature = "arrow")]
    /// Filter nullable Arrow array, excluding NULLs from output.
    pub fn filter_arrow_nullable<T: ArrowFilterKey>(
        &mut self,
        array: &PrimitiveArray<T::ArrowType>,
        pred: &Predicate<T>,
    ) -> Result<PrimitiveArray<T::ArrowType>, FilterError>;
}
```

### 4. Arrow Feature Flag

```toml
# Cargo.toml additions
[features]
default = []
arrow = ["dep:arrow-array", "dep:arrow-buffer", "dep:arrow-schema"]

[dependencies]
arrow-array = { version = "54", optional = true }
arrow-buffer = { version = "54", optional = true }
arrow-schema = { version = "54", optional = true }
```

### 5. File Structure

| File | Action | Description |
|------|--------|-------------|
| `shaders/filter.metal` | Modify | Add `filter_bitmap_scan`, `filter_bitmap_scatter`, HAS_NULLS constant |
| `src/lib.rs` | Modify | Add BooleanMask, gather(), filter_multi_mask(), internal bitmap pipeline |
| `src/arrow.rs` | Create | Arrow integration behind feature flag |
| `src/multi_column.rs` | Create | Multi-column filter logic |
| `Cargo.toml` | Modify | Add arrow feature, bump to 0.2.0 |
| `README.md` | Modify | Update benchmarks, add multi-column + Arrow examples |
| `benches/filter_benchmark.rs` | Modify | Add bitmap-cached and multi-column benchmarks |

### 6. polars-gpu-filter Crate (separate, interface only)

```
polars-gpu-filter/
├── Cargo.toml      # depends on forge-filter with arrow feature
├── src/
│   └── lib.rs      # pyo3-polars expression plugin
└── python/
    └── polars_gpu_filter/__init__.py
```

The plugin converts Polars Series → Arrow arrays → forge-filter calls → BooleanChunked.
This is a separate crate and NOT part of the forge-filter v0.2 scope. Design only.

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Ordered mode approach | Bitmap caching (3-dispatch) | Simpler than Decoupled Fallback, eliminates double-read |
| Bitmap storage | `atomic_store` packed u32 words | `simd_ballot` gives 32 bits per simdgroup naturally |
| NULL handling | `HAS_NULLS` function constant | Zero overhead when no NULLs (branch eliminated at compile time) |
| Arrow buffer transfer | Always copy via `newBufferWithBytes` | 16KB alignment mismatch, 0.5ms copy acceptable |
| Multi-column max | 4 columns per kernel | Function constants PRED_TYPE_A..D cover 99% of queries |
| PSO caching | Lazy compilation | 63µs per variant, no startup bloat |
| Backward compat | Extend API, no breaking changes | Existing users unaffected |
| Arrow dependency | Optional via feature flag | Core crate stays dependency-free |

## Performance Projections

| Workload | v0.1 | v0.2 Target | Improvement |
|----------|------|-------------|-------------|
| 16M u32 ordered 50% sel | 848µs | ~580µs | 1.46x (10x vs Polars) |
| 16M u32 unordered | 574µs | 574µs | unchanged |
| 3-col AND ordered | 2,544µs | ~944µs | 2.69x |
| 16M nullable ordered | N/A | ~610µs | — |

## GPU KB References

- **#1560** simd_ballot for bitmap construction
- **#1354** Two-phase filter with bitmask
- **#1291** SLC sizes, bitmap fits in cache
- **#813** Two-pass stream compaction
- **#1628** Decoupled Fallback (backup approach)
- **#1655** device-scope fence for cross-TG coherence
- **#887** Function constants: 84% instruction reduction
- **#1293** M4 Pro 273 GB/s bandwidth
