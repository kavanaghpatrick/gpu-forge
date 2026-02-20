# UX Design: forge-sort v2 Multi-Type API

## Executive Summary

This document defines the developer-facing API for forge-sort v2, which adds signed integer sorting (i32), floating-point sorting (f32/f64), key-value pair sorting, and argsort to the existing u32-only GPU radix sort library. The design prioritizes three principles in order: **zero performance overhead** from the type system, **compile-time type safety** so users cannot accidentally sort i32 data through a u32 path, and **API familiarity** for Rust developers coming from `[T]::sort()`, rayon, or rdst.

The central API decision is **explicit typed methods** (`sort_i32`, `sort_f32`) over generics (`sort::<T>`) or traits. This matches the existing `sort_u32` pattern, avoids trait-bound complexity leaking GPU implementation details, and aligns with CUB's proven API design. A sealed `SortKey` trait provides the generic `SortBuffer<T>` while keeping the method-level API concrete and discoverable.

---

## Research: Rust Sorting API Patterns

### Standard Library

The Rust standard library provides sorting on slices via inherent methods:

```rust
// Direct sort (requires Ord)
slice.sort();
slice.sort_unstable();

// Key-based sort
slice.sort_by_key(|x| x.field);
slice.sort_unstable_by_key(|x| x.field);

// Custom comparator
slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
```

Key observations:
- Methods are on `[T]` directly, not behind traits
- No `argsort` -- Rust developers use the enumerate-sort-extract pattern: `indices.sort_by_key(|&i| &data[i])`
- `f32` has no `Ord` impl, requiring `sort_by(f32::total_cmp)` or `sort_unstable_by(f32::total_cmp)`
- Naming is verb-first: `sort_unstable_by_key`, not `unstable_sort_by_key`

### Rayon (Parallel Sort)

[Rayon](https://docs.rs/rayon/latest/rayon/slice/trait.ParallelSliceMut.html) extends slices via the `ParallelSliceMut` trait:

```rust
use rayon::prelude::*;
slice.par_sort();
slice.par_sort_unstable();
slice.par_sort_by_key(|x| x.field);
```

Key observations:
- Prefix-based disambiguation: `par_sort` vs `sort`
- Same method structure as stdlib, just with `par_` prefix
- No separate type-specific methods -- uses generics throughout

### rdst (Radix Sort)

[rdst](https://crates.io/crates/rdst) uses a trait-based approach:

```rust
use rdst::RadixSort;
vec.radix_sort_unstable();  // works for u32, i32, f32, f64, etc.
```

Key observations:
- Single method handles all types via the `RadixKey` trait
- `RadixKey` requires `LEVELS: usize` (byte count) and `get_level(level) -> u8`
- Users implement `RadixKey` for custom types
- Builder pattern for tuning: `vec.radix_sort_builder().with_low_mem_tuner().sort()`

### radsort

[radsort](https://crates.io/crates/radsort) uses free functions with a sealed `Key` trait:

```rust
radsort::sort(&mut data);           // any Key type
radsort::sort_by_key(&mut items, |x| x.score);
radsort::sort_by_cached_key(&mut items, |x| x.expensive_key());
```

Key observations:
- Free functions, not methods
- Sealed `Key` trait -- users cannot implement it for custom types
- Supports signed integers, floats, bools, chars out of the box
- `sort_by_key` is the argsort primitive (sort items by extracted key)
- Clean, minimal API surface

### wgpu_sort (GPU Sort)

[wgpu_sort](https://docs.rs/wgpu_sort) is the closest ecosystem comparator:

```rust
let sorter = GPUSorter::new(&device, subgroup_size);
let sort_buffers = sorter.create_sort_buffers(&device, NonZeroU32::new(n).unwrap());
// upload keys and values
sorter.sort(&mut encoder, &queue, &sort_buffers, None);
```

Key observations:
- Always key-value pairs (no keys-only variant)
- No type parameterization -- always u32 keys, u32 values
- Separate `SortBuffers` type holds keys + values
- Async-compatible (takes encoder, does not block)

---

## API Design Decisions

### Decision 1: Explicit Typed Methods (not generics, not traits)

**Chosen**: `sort_i32()`, `sort_f32()`, `argsort_u32()`, etc.

**Rejected alternatives**:

| Approach | Example | Why Rejected |
|----------|---------|-------------|
| Generic method | `sort::<i32>(&mut data)` | Requires `SortKey` trait bound on `T`, which leaks GPU details (byte transformations, buffer layout). Turbofish syntax is unfamiliar for sort operations. Error messages reference trait bounds instead of "i32 not supported". |
| Trait on slice | `data.gpu_sort()` | Orphan rules prevent implementing foreign traits on foreign types (`[i32]`). A newtype wrapper adds friction. Extension trait pollutes autocomplete with GPU methods on every slice. |
| Trait on GpuSorter | `sorter.sort(&mut data)` where `T: GpuSortable` | Overloaded `sort` method with different behavior per type is confusing. Which `sort` handles `&mut [u32]` vs `&SortBuffer`? Method resolution becomes complex. |
| Free functions | `forge_sort::sort(&mut data)` | Cannot access sorter state (Metal device, PSO cache, scratch buffers). Would need a global singleton or per-call initialization. |

**Rationale**:
1. **Matches v1**: `sort_u32` already exists. `sort_i32`/`sort_f32` are the natural extension.
2. **Discoverable**: IDE autocomplete on `sorter.sort_` shows all available types immediately.
3. **Compile-time safety**: Passing `&mut [f32]` to `sort_i32` is a type error, not a runtime error.
4. **No trait complexity**: Users never see `where T: SortKey` bounds. No orphan rule issues.
5. **CUB precedent**: NVIDIA CUB uses `SortKeys<KeyT>` with explicit type -- the most successful GPU sort API in production.
6. **Stable API surface**: Adding `sort_u64` later does not change any existing signatures or trait bounds.

### Decision 2: SortBuffer Gets a Generic Type Parameter

**Chosen**: `SortBuffer<T>` with sealed `SortKey` trait, replacing the untyped `SortBuffer`.

```rust
// v1 (current)
pub struct SortBuffer { /* untyped, always u32 */ }

// v2 (proposed)
pub struct SortBuffer<T: SortKey> { /* typed */ }
```

The `SortKey` trait is **sealed** (cannot be implemented outside forge-sort):

```rust
mod private { pub trait Sealed {} }

pub trait SortKey: private::Sealed + Copy + 'static {
    // Not visible to users -- sealed trait, no user-implementable methods
}

impl private::Sealed for u32 {}
impl private::Sealed for i32 {}
impl private::Sealed for f32 {}
impl private::Sealed for f64 {}

impl SortKey for u32 {}
impl SortKey for i32 {}
impl SortKey for f32 {}
impl SortKey for f64 {}
```

**Why sealed**: Users should not be able to implement `SortKey` for arbitrary types. The GPU kernels only support specific bit layouts. An unsealed trait would invite `impl SortKey for MyStruct` which cannot work.

**Why generic instead of `SortBufferI32`, `SortBufferF32`**: Reduces API surface from 4+ buffer types to 1 generic type. `SortBuffer<f32>` is self-documenting. The generic parameter carries zero runtime cost (it only affects which `as_slice` / `as_mut_slice` return type is used, via `PhantomData`).

**Migration**: `SortBuffer` (unqualified) becomes `SortBuffer<u32>`. Since forge-sort is pre-1.0, this breaking change is acceptable (see Migration Path section).

### Decision 3: Argsort as a First-Class Method

**Chosen**: `argsort_*` methods return `Vec<u32>` (indices).

```rust
let indices: Vec<u32> = sorter.argsort_f32(&data)?;
// data is NOT modified
// indices[0] is the index of the smallest element in data
// data[indices[i]] <= data[indices[i+1]] for all i
```

**Rationale**:
1. PM decision: argsort is first-class (Q&A #3).
2. argsort is the building block for key-value sorting via Strategy B (sort keys+indices, then gather).
3. Numpy/scipy users expect `argsort` as a named concept. It is the standard term.
4. Returning `Vec<u32>` (not modifying input) is correct semantics -- argsort observes data, does not mutate it.
5. The zero-copy variant (`argsort_buffer`) returns a `SortBuffer<u32>` of indices.

### Decision 4: Key-Value Pairs Use Separate Keys/Values Slices

**Chosen**: `sort_pairs_*` takes `(&mut [K], &mut [V])` where `V` is `u32`.

```rust
sorter.sort_pairs_f32(&mut keys, &mut values)?;
// keys is sorted, values are rearranged to match
```

**Why `(&mut [K], &mut [V])` not `&mut [(K, V)]`**:
- GPU memory layout is Structure of Arrays (SoA), not Array of Structures (AoS). Keys and values are in separate buffers on the GPU.
- AoS would require splitting into SoA for the GPU and merging back, adding overhead and complexity.
- SoA matches the internal implementation directly.
- CUB uses separate key/value pointers for the same reason.

**Why `V` is always `u32`**:
- PM decision: 32-bit values only (Q&A #2).
- Covers the dominant `(key, index)` use case. The index pattern (`u32` indices into another array) handles arbitrary value types: sort keys with indices, then gather values by index.
- Expanding to generic `V: Copy + 'static` where `size_of::<V>() == 4` can come in v2.1 if needed, but u32 covers 90%+ of cases.

### Decision 5: Error Handling Unchanged

**Chosen**: All new methods return `Result<_, SortError>`. `SortError` gains one new variant.

```rust
#[derive(Debug, thiserror::Error)]
pub enum SortError {
    #[error("no Metal GPU device found")]
    DeviceNotFound,

    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("GPU execution failed: {0}")]
    GpuExecution(String),

    #[error("mismatched lengths: keys ({keys}) != values ({values})")]
    LengthMismatch { keys: usize, values: usize },
}
```

The new `LengthMismatch` variant is for `sort_pairs_*` when key and value slices have different lengths. This is a programmer error (not a GPU error), but returning `Result` is more ergonomic than panicking -- it lets callers use `?` consistently.

**Considered and rejected**: Separate error types per method family. Adds complexity with no benefit -- all methods share the same failure modes (no device, shader fail, GPU fail), and `LengthMismatch` is harmless as an unused variant for non-pairs methods.

### Decision 6: Naming Convention for Method Families

The naming follows this pattern:

```
{verb}_{type}                -- sort keys only (memcpy path)
{verb}_buffer                -- sort buffer in-place (zero-copy, u32 compat)
{verb}_{type}_buffer         -- sort typed buffer (zero-copy, new)
argsort_{type}               -- return sorted indices (memcpy path)
sort_pairs_{type}            -- sort key-value pairs (memcpy path)
alloc_sort_buffer::<T>       -- allocate typed buffer (zero-copy)
```

**Why not `sort_buffer_i32`?** The type comes before `buffer` because the type qualifies the *sort*, not the buffer. You are sorting i32 values, and the buffer is the mechanism. Reading left to right: "sort i32, from a buffer." This also groups by type in IDE autocomplete: `sort_f32`, `sort_f32_buffer` appear adjacent.

**Why `argsort_*` returns `Vec<u32>` not `&[u32]`?** Argsort creates new data (the index permutation). It cannot return a reference to data that did not exist before the call. The internal GPU buffer holding the indices is ephemeral scratch space. `Vec<u32>` is the correct return type for owned, newly-created data.

---

## Complete API Specification

### GpuSorter Methods

```rust
impl GpuSorter {
    // ── Construction ────────────────────────────────────────────

    /// Create a new GPU sorter. Initializes Metal device and compiles sort kernels.
    pub fn new() -> Result<Self, SortError>;

    // ── Key-Only Sorting (memcpy path) ─────────────────────────

    /// Sort u32 slice in-place on GPU.
    /// Copies data to GPU buffer, sorts, copies back.
    pub fn sort_u32(&mut self, data: &mut [u32]) -> Result<(), SortError>;

    /// Sort i32 slice in-place on GPU.
    /// Negative values sort before positive. i32::MIN is the smallest value.
    pub fn sort_i32(&mut self, data: &mut [i32]) -> Result<(), SortError>;

    /// Sort f32 slice in-place on GPU.
    /// Uses IEEE 754 total ordering: -NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN.
    /// All NaN values sort to the extremes (negative NaN before -Inf, positive NaN after +Inf).
    pub fn sort_f32(&mut self, data: &mut [f32]) -> Result<(), SortError>;

    /// Sort u64 slice in-place on GPU.
    /// Requires 8 radix passes (vs 4 for 32-bit types). No bit transformation needed.
    pub fn sort_u64(&mut self, data: &mut [u64]) -> Result<(), SortError>;

    /// Sort i64 slice in-place on GPU.
    /// Negative values sort before positive. i64::MIN is the smallest value.
    /// Uses XOR sign bit (0x8000000000000000) transformation, same concept as i32.
    /// Requires 8 radix passes (vs 4 for 32-bit types).
    pub fn sort_i64(&mut self, data: &mut [i64]) -> Result<(), SortError>;

    /// Sort f64 slice in-place on GPU.
    /// Same ordering semantics as sort_f32. Requires 8 radix passes (vs 4 for 32-bit types).
    pub fn sort_f64(&mut self, data: &mut [f64]) -> Result<(), SortError>;

    // ── Key-Only Sorting (zero-copy path) ──────────────────────

    /// Sort a SortBuffer<u32> in-place. Zero memcpy.
    pub fn sort_buffer(&mut self, buf: &SortBuffer<u32>) -> Result<(), SortError>;

    /// Sort a SortBuffer<i32> in-place. Zero memcpy.
    /// GPU-side bit transformation (no CPU round-trip).
    pub fn sort_i32_buffer(&mut self, buf: &SortBuffer<i32>) -> Result<(), SortError>;

    /// Sort a SortBuffer<f32> in-place. Zero memcpy.
    /// GPU-side FloatFlip transformation (no CPU round-trip).
    pub fn sort_f32_buffer(&mut self, buf: &SortBuffer<f32>) -> Result<(), SortError>;

    /// Sort a SortBuffer<u64> in-place. Zero memcpy.
    /// No bit transformation needed (unsigned integers sort naturally).
    pub fn sort_u64_buffer(&mut self, buf: &SortBuffer<u64>) -> Result<(), SortError>;

    /// Sort a SortBuffer<i64> in-place. Zero memcpy.
    /// GPU-side sign bit XOR transformation (no CPU round-trip).
    pub fn sort_i64_buffer(&mut self, buf: &SortBuffer<i64>) -> Result<(), SortError>;

    /// Sort a SortBuffer<f64> in-place. Zero memcpy.
    pub fn sort_f64_buffer(&mut self, buf: &SortBuffer<f64>) -> Result<(), SortError>;

    // ── Argsort (return sorted indices) ────────────────────────

    /// Return indices that would sort the u32 slice.
    /// The input slice is NOT modified.
    /// result[i] is the index of the i-th smallest element in data.
    pub fn argsort_u32(&mut self, data: &[u32]) -> Result<Vec<u32>, SortError>;

    /// Return indices that would sort the i32 slice.
    pub fn argsort_i32(&mut self, data: &[i32]) -> Result<Vec<u32>, SortError>;

    /// Return indices that would sort the f32 slice.
    pub fn argsort_f32(&mut self, data: &[f32]) -> Result<Vec<u32>, SortError>;

    /// Return indices that would sort the u64 slice.
    pub fn argsort_u64(&mut self, data: &[u64]) -> Result<Vec<u32>, SortError>;

    /// Return indices that would sort the i64 slice.
    pub fn argsort_i64(&mut self, data: &[i64]) -> Result<Vec<u32>, SortError>;

    /// Return indices that would sort the f64 slice.
    pub fn argsort_f64(&mut self, data: &[f64]) -> Result<Vec<u32>, SortError>;

    // ── Key-Value Pair Sorting (memcpy path) ───────────────────

    /// Sort key-value pairs in-place. Keys are sorted, values are rearranged to match.
    /// Keys and values must have the same length.
    pub fn sort_pairs_u32(
        &mut self, keys: &mut [u32], values: &mut [u32],
    ) -> Result<(), SortError>;

    /// Sort i32 key-value pairs in-place.
    pub fn sort_pairs_i32(
        &mut self, keys: &mut [i32], values: &mut [u32],
    ) -> Result<(), SortError>;

    /// Sort f32 key-value pairs in-place.
    pub fn sort_pairs_f32(
        &mut self, keys: &mut [f32], values: &mut [u32],
    ) -> Result<(), SortError>;

    /// Sort u64 key-value pairs in-place. 64-bit keys with 32-bit values.
    pub fn sort_pairs_u64(
        &mut self, keys: &mut [u64], values: &mut [u32],
    ) -> Result<(), SortError>;

    /// Sort i64 key-value pairs in-place. 64-bit signed keys with 32-bit values.
    pub fn sort_pairs_i64(
        &mut self, keys: &mut [i64], values: &mut [u32],
    ) -> Result<(), SortError>;

    // ── Buffer Allocation ──────────────────────────────────────

    /// Allocate a typed GPU buffer for zero-copy sorting.
    /// The buffer uses unified memory (StorageModeShared).
    pub fn alloc_sort_buffer<T: SortKey>(&self, capacity: usize) -> SortBuffer<T>;
}
```

### Full Method Table

| Method | Input | Output | Phase |
|--------|-------|--------|-------|
| `sort_u32(&mut [u32])` | mutable slice | in-place | **existing** |
| `sort_buffer(&SortBuffer<u32>)` | typed buffer | in-place | **existing** (signature changes) |
| `sort_i32(&mut [i32])` | mutable slice | in-place | Phase 1 |
| `sort_f32(&mut [f32])` | mutable slice | in-place | Phase 1 |
| `sort_u64(&mut [u64])` | mutable slice | in-place | Phase 3 |
| `sort_i64(&mut [i64])` | mutable slice | in-place | Phase 3 |
| `sort_f64(&mut [f64])` | mutable slice | in-place | Phase 3 |
| `sort_i32_buffer(&SortBuffer<i32>)` | typed buffer | in-place | Phase 1 |
| `sort_f32_buffer(&SortBuffer<f32>)` | typed buffer | in-place | Phase 1 |
| `sort_u64_buffer(&SortBuffer<u64>)` | typed buffer | in-place | Phase 3 |
| `sort_i64_buffer(&SortBuffer<i64>)` | typed buffer | in-place | Phase 3 |
| `sort_f64_buffer(&SortBuffer<f64>)` | typed buffer | in-place | Phase 3 |
| `argsort_u32(&[u32])` | immutable slice | `Vec<u32>` | Phase 2 |
| `argsort_i32(&[i32])` | immutable slice | `Vec<u32>` | Phase 2 |
| `argsort_f32(&[f32])` | immutable slice | `Vec<u32>` | Phase 2 |
| `argsort_u64(&[u64])` | immutable slice | `Vec<u32>` | Phase 3 |
| `argsort_i64(&[i64])` | immutable slice | `Vec<u32>` | Phase 3 |
| `argsort_f64(&[f64])` | immutable slice | `Vec<u32>` | Phase 3 |
| `sort_pairs_u32(&mut [u32], &mut [u32])` | key + value slices | in-place both | Phase 2 |
| `sort_pairs_i32(&mut [i32], &mut [u32])` | key + value slices | in-place both | Phase 2 |
| `sort_pairs_f32(&mut [f32], &mut [u32])` | key + value slices | in-place both | Phase 2 |
| `sort_pairs_u64(&mut [u64], &mut [u32])` | key + value slices | in-place both | Phase 3 |
| `sort_pairs_i64(&mut [i64], &mut [u32])` | key + value slices | in-place both | Phase 3 |
| `alloc_sort_buffer::<T>(usize)` | capacity | `SortBuffer<T>` | Phase 1 |

### SortBuffer<T> Methods

```rust
/// A typed Metal buffer for zero-copy GPU sorting.
///
/// Created via [`GpuSorter::alloc_sort_buffer`]. Uses unified memory
/// (StorageModeShared) -- CPU reads/writes go directly to GPU-visible pages.
///
/// # Type Safety
///
/// `SortBuffer<f32>` can only be sorted via `sort_f32_buffer()`.
/// Attempting to pass it to `sort_buffer()` (which expects `SortBuffer<u32>`)
/// is a compile error.
pub struct SortBuffer<T: SortKey> {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    capacity: usize,
    _marker: PhantomData<T>,
}

impl<T: SortKey> SortBuffer<T> {
    /// Number of elements currently in this buffer.
    pub fn len(&self) -> usize;

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool;

    /// Capacity in elements of type T.
    pub fn capacity(&self) -> usize;

    /// Get a mutable slice to write data directly into GPU-visible memory.
    /// Write your data here, then call set_len() to mark how many elements are valid.
    pub fn as_mut_slice(&mut self) -> &mut [T];

    /// Get a slice to read sorted results directly from GPU-visible memory.
    pub fn as_slice(&self) -> &[T];

    /// Set the number of valid elements. Must be <= capacity.
    ///
    /// # Panics
    /// Panics if len > capacity.
    pub fn set_len(&mut self, len: usize);

    /// Copy data from a slice into the buffer. Sets len automatically.
    ///
    /// # Panics
    /// Panics if data.len() > capacity.
    pub fn copy_from_slice(&mut self, data: &[T]);

    /// Copy sorted results out to a slice.
    pub fn copy_to_slice(&self, dest: &mut [T]);

    /// Access the underlying Metal buffer (for advanced use / pipeline integration).
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer>;
}
```

### SortKey Trait (Sealed)

```rust
/// Marker trait for types that can be GPU-sorted.
///
/// This trait is **sealed** -- it cannot be implemented outside forge-sort.
/// Supported types: u32, i32, f32, u64, i64, f64.
///
/// You do not need to interact with this trait directly. It exists only
/// to constrain SortBuffer<T> and alloc_sort_buffer::<T>() to
/// supported types.
pub trait SortKey: private::Sealed + Copy + 'static {}
```

The private module pattern:

```rust
mod private {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
    impl Sealed for u64 {}
    impl Sealed for i64 {}
    impl Sealed for f64 {}
}
```

### SortError Enum

```rust
#[derive(Debug, thiserror::Error)]
pub enum SortError {
    #[error("no Metal GPU device found")]
    DeviceNotFound,

    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("GPU execution failed: {0}")]
    GpuExecution(String),

    #[error("mismatched lengths: keys ({keys}) != values ({values})")]
    LengthMismatch { keys: usize, values: usize },
}
```

---

## Error Message Design

### Compile-Time Errors (Type System)

The typed API prevents the most dangerous class of errors at compile time:

```rust
// COMPILE ERROR: expected `&mut [u32]`, found `&mut [f32]`
let mut data = vec![1.0f32, 2.0, 3.0];
sorter.sort_u32(&mut data)?;  // won't compile

// COMPILE ERROR: expected `&SortBuffer<u32>`, found `&SortBuffer<f32>`
let buf: SortBuffer<f32> = sorter.alloc_sort_buffer(1000);
sorter.sort_buffer(&buf)?;  // won't compile -- use sort_f32_buffer()

// COMPILE ERROR: trait bound `String: SortKey` is not satisfied
let buf: SortBuffer<String> = sorter.alloc_sort_buffer(1000);  // won't compile
```

These produce standard Rust type errors. No custom error messages needed -- the compiler's output is clear.

### Runtime Errors

| Scenario | Error | Message |
|----------|-------|---------|
| No Metal GPU | `SortError::DeviceNotFound` | `"no Metal GPU device found"` |
| Shader compile fail | `SortError::ShaderCompilation(detail)` | `"shader compilation failed: {detail}"` |
| Command buffer error | `SortError::GpuExecution(detail)` | `"GPU execution failed: {detail}"` |
| Key/value length mismatch | `SortError::LengthMismatch { keys, values }` | `"mismatched lengths: keys (1000) != values (999)"` |

### Panic Conditions (Programmer Errors)

| Scenario | Behavior | Message |
|----------|----------|---------|
| `set_len(len > capacity)` | panic | `"len 2000 exceeds capacity 1000"` |
| `copy_from_slice(data.len() > capacity)` | panic | `"data len 2000 exceeds capacity 1000"` |

These match the existing v1 panic behavior exactly. They are assertion failures for violated preconditions, not recoverable errors.

---

## Documentation Patterns

### Module-Level Documentation

```rust
//! # forge-sort
//!
//! GPU-accelerated radix sort for Apple Silicon. 31x faster than `sort_unstable()`.
//!
//! ## Supported Types
//!
//! | Type | Method | Zero-Copy | Argsort | Key-Value |
//! |------|--------|-----------|---------|-----------|
//! | `u32` | `sort_u32` | `sort_buffer` | `argsort_u32` | `sort_pairs_u32` |
//! | `i32` | `sort_i32` | `sort_i32_buffer` | `argsort_i32` | `sort_pairs_i32` |
//! | `f32` | `sort_f32` | `sort_f32_buffer` | `argsort_f32` | `sort_pairs_f32` |
//! | `u64` | `sort_u64` | `sort_u64_buffer` | `argsort_u64` | `sort_pairs_u64` |
//! | `i64` | `sort_i64` | `sort_i64_buffer` | `argsort_i64` | `sort_pairs_i64` |
//! | `f64` | `sort_f64` | `sort_f64_buffer` | `argsort_f64` | -- |
//!
//! ## Quick Start
//!
//! ```rust
//! use forge_sort::GpuSorter;
//!
//! let mut sorter = GpuSorter::new()?;
//!
//! // Sort integers
//! let mut ints = vec![5i32, -3, 8, -1, 0];
//! sorter.sort_i32(&mut ints)?;
//! assert_eq!(ints, vec![-3, -1, 0, 5, 8]);
//!
//! // Sort floats
//! let mut floats = vec![3.14f32, -2.7, 0.0, 1.0, -0.0];
//! sorter.sort_f32(&mut floats)?;
//! // -2.7, -0.0, 0.0, 1.0, 3.14  (total_cmp ordering: -0.0 < +0.0)
//!
//! // Get sorted indices without modifying data
//! let data = vec![30u32, 10, 20];
//! let indices = sorter.argsort_u32(&data)?;
//! assert_eq!(indices, vec![1, 2, 0]);  // data[1]=10 < data[2]=20 < data[0]=30
//!
//! // Sort key-value pairs (e.g., depth + instance ID for rendering)
//! let mut depths = vec![5.0f32, 1.0, 3.0];
//! let mut instance_ids = vec![100u32, 200, 300];
//! sorter.sort_pairs_f32(&mut depths, &mut instance_ids)?;
//! assert_eq!(depths, vec![1.0, 3.0, 5.0]);
//! assert_eq!(instance_ids, vec![200, 300, 100]);
//! # Ok::<(), forge_sort::SortError>(())
//! ```
//!
//! ## Float Ordering
//!
//! Float sorting uses IEEE 754 `total_cmp` semantics:
//! - `-NaN < -Inf < -1.0 < -0.0 < +0.0 < 1.0 < +Inf < +NaN`
//! - `-0.0` sorts before `+0.0` (they are distinct values)
//! - All NaN payloads are preserved; different NaN bit patterns sort to distinct positions
//! - This matches [`f32::total_cmp`](https://doc.rust-lang.org/std/primitive.f32.html#method.total_cmp)
//!
//! ## Stability
//!
//! Radix sort is inherently **stable**: equal keys preserve their original relative order.
//! This is guaranteed for all sort methods.
```

### Per-Method Documentation Template

Each method follows this structure: one-sentence summary, ordering semantics (if applicable), performance path note, example, performance numbers, errors.

**sort_f32 example**:

```rust
/// Sort f32 slice in-place on GPU.
///
/// Uses IEEE 754 total ordering: `-NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN`.
/// The sort is stable (equal values preserve relative order).
///
/// Copies data to an internal GPU buffer, sorts, and copies back. For zero-copy
/// sorting, use [`alloc_sort_buffer::<f32>`](Self::alloc_sort_buffer) +
/// [`sort_f32_buffer`](Self::sort_f32_buffer).
///
/// # Examples
///
/// ```rust
/// # use forge_sort::GpuSorter;
/// let mut sorter = GpuSorter::new()?;
/// let mut data = vec![3.14f32, -2.7, 0.0, f32::NAN, -0.0, f32::INFINITY];
/// sorter.sort_f32(&mut data)?;
/// assert_eq!(data[0], -2.7);
/// assert_eq!(data[1].to_bits(), (-0.0f32).to_bits());  // -0.0
/// assert_eq!(data[2].to_bits(), 0.0f32.to_bits());      // +0.0
/// assert_eq!(data[3], 3.14);
/// assert_eq!(data[4], f32::INFINITY);
/// assert!(data[5].is_nan());
/// # Ok::<(), forge_sort::SortError>(())
/// ```
///
/// # Performance
///
/// ~10% slower than `sort_u32` due to bit transformation overhead.
/// At 16M elements: ~2500 Mk/s (memcpy path), ~4500 Mk/s (zero-copy).
///
/// # Errors
///
/// Returns [`SortError::GpuExecution`] if the GPU command buffer fails.
pub fn sort_f32(&mut self, data: &mut [f32]) -> Result<(), SortError>;
```

**argsort_f32 example**:

```rust
/// Return indices that would sort the f32 slice in ascending order.
///
/// The input data is **not modified**. Returns a `Vec<u32>` where `result[i]`
/// is the index of the i-th smallest element in `data`.
///
/// This is the GPU equivalent of NumPy's `argsort`. Useful for:
/// - Sorting multiple arrays by the same key (sort once, gather many)
/// - Building sorted indices for database-style operations
/// - Any case where you need the permutation, not the sorted data
///
/// # Examples
///
/// ```rust
/// # use forge_sort::GpuSorter;
/// let mut sorter = GpuSorter::new()?;
///
/// let scores = vec![3.5f32, 1.2, 2.8];
/// let order = sorter.argsort_f32(&scores)?;
/// assert_eq!(order, vec![1, 2, 0]);  // 1.2 < 2.8 < 3.5
///
/// // Use indices to reorder associated data
/// let names = vec!["Alice", "Bob", "Carol"];
/// let sorted_names: Vec<&str> = order.iter().map(|&i| names[i as usize]).collect();
/// assert_eq!(sorted_names, vec!["Bob", "Carol", "Alice"]);
/// # Ok::<(), forge_sort::SortError>(())
/// ```
///
/// # Relationship to sort_pairs
///
/// `argsort` is the primitive that `sort_pairs` is built on internally.
/// If you need both sorted keys and rearranged values, prefer `sort_pairs_f32`
/// which does both in one call.
pub fn argsort_f32(&mut self, data: &[f32]) -> Result<Vec<u32>, SortError>;
```

**sort_pairs_f32 example**:

```rust
/// Sort f32 key-value pairs in-place on GPU.
///
/// Keys are sorted in ascending order. Values (u32) are rearranged to maintain
/// the key-value association. The sort is **stable**: equal keys preserve their
/// original relative order among associated values.
///
/// # Examples
///
/// ```rust
/// # use forge_sort::GpuSorter;
/// let mut sorter = GpuSorter::new()?;
///
/// // Sort particles by depth for back-to-front rendering
/// let mut depths = vec![5.0f32, 1.0, 3.0, 1.0];
/// let mut ids    = vec![  100,   200,  300,  400];
/// sorter.sort_pairs_f32(&mut depths, &mut ids)?;
///
/// assert_eq!(depths, vec![1.0, 1.0, 3.0, 5.0]);
/// assert_eq!(ids,    vec![200, 400, 300, 100]);  // stable: 200 before 400
/// # Ok::<(), forge_sort::SortError>(())
/// ```
///
/// # Errors
///
/// Returns [`SortError::LengthMismatch`] if `keys.len() != values.len()`.
/// Returns [`SortError::GpuExecution`] if the GPU command buffer fails.
///
/// # Performance
///
/// ~50% slower than key-only sorting due to doubled memory bandwidth.
/// At 16M elements: ~1400 Mk/s (memcpy path).
pub fn sort_pairs_f32(
    &mut self,
    keys: &mut [f32],
    values: &mut [u32],
) -> Result<(), SortError>;
```

### sort_i32 Documentation

```rust
/// Sort i32 slice in-place on GPU.
///
/// Sorts in ascending signed integer order: `i32::MIN` (-2,147,483,648) is
/// the smallest value, `i32::MAX` (2,147,483,647) is the largest.
///
/// Copies data to an internal GPU buffer, sorts, and copies back. For zero-copy
/// sorting, use [`alloc_sort_buffer::<i32>`](Self::alloc_sort_buffer) +
/// [`sort_i32_buffer`](Self::sort_i32_buffer).
///
/// # Examples
///
/// ```rust
/// # use forge_sort::GpuSorter;
/// let mut sorter = GpuSorter::new()?;
/// let mut data = vec![5i32, -3, 0, i32::MIN, i32::MAX, -1];
/// sorter.sort_i32(&mut data)?;
/// assert_eq!(data, vec![i32::MIN, -3, -1, 0, 5, i32::MAX]);
/// # Ok::<(), forge_sort::SortError>(())
/// ```
///
/// # Performance
///
/// Same throughput as `sort_u32` -- the sign bit transformation is negligible.
/// At 16M elements: ~2800 Mk/s (memcpy path), ~4200 Mk/s (zero-copy).
///
/// # Errors
///
/// Returns [`SortError::GpuExecution`] if the GPU command buffer fails.
pub fn sort_i32(&mut self, data: &mut [i32]) -> Result<(), SortError>;
```

---

## Migration Path: v1 to v2

### Breaking Changes

1. **`SortBuffer` becomes `SortBuffer<u32>`**: The untyped `SortBuffer` gains a type parameter. This is a breaking change for any code that names the type explicitly.

2. **`alloc_sort_buffer` becomes generic**: The signature changes from `fn alloc_sort_buffer(&self, capacity: usize) -> SortBuffer` to `fn alloc_sort_buffer<T: SortKey>(&self, capacity: usize) -> SortBuffer<T>`. Callers that do not annotate the type must either add a turbofish or let the type be inferred from usage.

### Why No Backward-Compatibility Alias

Since forge-sort is a pre-1.0 crate (`version = "0.1.0"`), the semver expectation is that breaking changes are acceptable between 0.x releases. Adding a deprecated type alias (`type UntypedSortBuffer = SortBuffer<u32>`) creates unnecessary complexity for a crate with very few external users at this stage.

**Recommendation**: Simply change `SortBuffer` to `SortBuffer<u32>`. Bump the version to `0.2.0`.

### Migration Examples

```rust
// v1 code:
let mut buf = sorter.alloc_sort_buffer(1_000_000);
buf.copy_from_slice(&data);
sorter.sort_buffer(&buf)?;

// v2 code (option A: explicit type annotation):
let mut buf: SortBuffer<u32> = sorter.alloc_sort_buffer(1_000_000);
buf.copy_from_slice(&data);
sorter.sort_buffer(&buf)?;

// v2 code (option B: type inference from usage -- works if data is &[u32]):
let mut buf = sorter.alloc_sort_buffer(1_000_000);
buf.copy_from_slice(&data);   // data: &[u32] -> infers SortBuffer<u32>
sorter.sort_buffer(&buf)?;
```

In most cases, type inference from `copy_from_slice(&[u32])` or `sort_buffer()` will resolve the generic parameter without any code changes. The migration cost is near-zero.

### Non-Breaking Additions

All new methods (`sort_i32`, `sort_f32`, `argsort_*`, `sort_pairs_*`) are purely additive. They do not change existing method signatures.

---

## Usage Examples by Use Case

### Gaussian Splatting (Primary Target Market)

```rust
use forge_sort::GpuSorter;

fn sort_splats_by_depth(
    depths: &mut [f32],       // camera-space Z distances
    splat_ids: &mut [u32],    // indices into splat attribute arrays
    sorter: &mut GpuSorter,
) -> Result<(), forge_sort::SortError> {
    sorter.sort_pairs_f32(depths, splat_ids)?;
    // splat_ids[0] is now the nearest splat (smallest depth)
    // Use splat_ids to index into position, color, opacity, etc. buffers
    Ok(())
}
```

### Database Index Building

```rust
use forge_sort::GpuSorter;

fn build_sorted_index(
    column: &[i32],           // database column values
    sorter: &mut GpuSorter,
) -> Result<Vec<u32>, forge_sort::SortError> {
    // argsort gives row indices sorted by column value
    let row_order = sorter.argsort_i32(column)?;
    Ok(row_order)
}
```

### Zero-Copy Float Pipeline

```rust
use forge_sort::{GpuSorter, SortBuffer, SortKey};

fn gpu_float_pipeline(sorter: &mut GpuSorter) -> Result<(), forge_sort::SortError> {
    let mut buf: SortBuffer<f32> = sorter.alloc_sort_buffer(4_000_000);

    // Write simulation results directly into GPU memory
    let slice = buf.as_mut_slice();
    for i in 0..4_000_000 {
        slice[i] = (i as f32).sin();  // example: sin wave values
    }
    buf.set_len(4_000_000);

    // Sort in-place -- zero memcpy
    sorter.sort_f32_buffer(&buf)?;

    // Read median directly from GPU memory
    let median = buf.as_slice()[2_000_000];
    println!("Median sin value: {median}");

    Ok(())
}
```

### Multi-Column Sort (argsort composition)

```rust
use forge_sort::GpuSorter;

fn sort_by_two_columns(
    primary: &[f32],       // sort by this first
    secondary: &[i32],     // break ties with this
    sorter: &mut GpuSorter,
) -> Result<Vec<u32>, forge_sort::SortError> {
    // Stable radix sort: sort secondary first, then primary
    // (stable sort on primary preserves secondary ordering within ties)
    let mut indices = sorter.argsort_i32(secondary)?;

    // Gather primary values in secondary-sorted order
    let mut primary_reordered: Vec<f32> = indices.iter()
        .map(|&i| primary[i as usize])
        .collect();

    // Sort primary, carrying indices along
    sorter.sort_pairs_f32(&mut primary_reordered, &mut indices)?;

    Ok(indices)
}
```

---

## API Surface Summary

### v1 (current): 5 public items

```
GpuSorter::new()
GpuSorter::sort_u32()
GpuSorter::alloc_sort_buffer()
GpuSorter::sort_buffer()
SortBuffer  (+ 8 methods)
SortError   (3 variants)
```

### v2 (proposed): 30 public items

```
GpuSorter::new()                    (unchanged)
GpuSorter::sort_u32()               (unchanged)
GpuSorter::sort_i32()               (new)
GpuSorter::sort_f32()               (new)
GpuSorter::sort_u64()               (new, Phase 3)
GpuSorter::sort_i64()               (new, Phase 3)
GpuSorter::sort_f64()               (new, Phase 3)
GpuSorter::sort_buffer()            (signature: SortBuffer -> SortBuffer<u32>)
GpuSorter::sort_i32_buffer()        (new)
GpuSorter::sort_f32_buffer()        (new)
GpuSorter::sort_u64_buffer()        (new, Phase 3)
GpuSorter::sort_i64_buffer()        (new, Phase 3)
GpuSorter::sort_f64_buffer()        (new, Phase 3)
GpuSorter::argsort_u32()            (new)
GpuSorter::argsort_i32()            (new)
GpuSorter::argsort_f32()            (new)
GpuSorter::argsort_u64()            (new, Phase 3)
GpuSorter::argsort_i64()            (new, Phase 3)
GpuSorter::argsort_f64()            (new, Phase 3)
GpuSorter::sort_pairs_u32()         (new)
GpuSorter::sort_pairs_i32()         (new)
GpuSorter::sort_pairs_f32()         (new)
GpuSorter::sort_pairs_u64()         (new, Phase 3)
GpuSorter::sort_pairs_i64()         (new, Phase 3)
GpuSorter::alloc_sort_buffer::<T>() (now generic)
SortBuffer<T>  (+ 8 methods)        (now generic)
SortKey trait  (sealed)              (new)
SortError      (4 variants)          (+1 variant)
```

### Phased Delivery

| Phase | New Methods | Complexity |
|-------|------------|------------|
| Phase 1 (P0+P1) | `sort_i32`, `sort_f32`, `sort_i32_buffer`, `sort_f32_buffer`, `SortBuffer<T>`, `SortKey` | Low -- bit transforms only, no shader changes |
| Phase 2 (P2) | `argsort_*` (u32/i32/f32), `sort_pairs_*` (u32/i32/f32) | Medium -- new gather kernel, index tracking |
| Phase 3 (P3) | `sort_u64`, `sort_i64`, `sort_f64`, `sort_{u64,i64,f64}_buffer`, `argsort_{u64,i64,f64}`, `sort_pairs_{u64,i64}` | Medium -- 64-bit shader variant, 8 passes. u64/i64 are free once f64 pipeline exists. |

---

## Naming Convention Rationale

### Why `sort_f32` not `sort_float` or `sort_real`

- Rust convention uses concrete type names (`f32`, `i32`), not abstract names (`float`, `int`).
- `sort_float` is ambiguous between `f32` and `f64`.
- The Rust standard library uses `f32::total_cmp`, not `float::total_cmp`.
- CUB uses template parameters (`float`, `double`) but Rust does not have that style.

### Why `sort_pairs` not `sort_kv` or `sort_by_key`

- `sort_by_key` in Rust stdlib means "extract a sort key from each element via a closure." That is NOT what this does. This sorts explicit key/value arrays.
- `sort_kv` is abbreviation-heavy and unclear to newcomers.
- `sort_pairs` is the CUB/Thrust convention (`SortPairs`) and clearly describes the operation: sorting pairs of (key, value).

### Why `argsort` not `sort_indices` or `indirect_sort`

- `argsort` is the universally recognized name from NumPy, SciPy, Julia, MATLAB, and PyTorch.
- Rust developers working with GPU data are likely to come from Python/ML backgrounds where `argsort` is the standard term.
- `sort_indices` could be confused with "sort a slice of indices."

### Why `sort_i32_buffer` not `sort_buffer_i32`

- The type qualifier applies to the sort operation, not the buffer: you are sorting i32, from a buffer.
- Grouping in IDE autocomplete: `sort_f32`, `sort_f32_buffer` appear adjacent, showing both paths for the same type.
- Reading left to right follows English: "sort i32 buffer" vs "sort buffer i32."

---

## Rejected Design Alternatives

### Alternative A: Single `sort()` Method with Trait Dispatch

```rust
// REJECTED
sorter.sort(&mut data)?;  // where data: &mut [T], T: GpuSortable
```

**Why rejected**: Method resolution becomes ambiguous when multiple `sort` overloads exist (`&mut [T]` vs `&SortBuffer<T>`). IDE autocomplete shows a single `sort` method with complex trait bounds instead of the full menu of options. Error messages reference `GpuSortable` trait bounds that users never implemented. And most critically: the `sort` name collides with the stdlib `[T]::sort()` in developer mental models, causing confusion about which sort is being called.

### Alternative B: Builder Pattern

```rust
// REJECTED
sorter.sort()
    .keys(&mut data)
    .key_type::<f32>()
    .values(&mut vals)
    .execute()?;
```

**Why rejected**: Over-engineered for a library with 4 types. Adds allocation (builder struct) and runtime type checking where static dispatch suffices. The builder pattern is appropriate for complex configuration (rdst's tuner selection) but not for type selection -- that is what Rust's type system does at zero cost.

### Alternative C: Enum-Based Type Selection

```rust
// REJECTED
sorter.sort(&mut data, SortType::F32)?;
```

**Why rejected**: Runtime type checking instead of compile-time. Users can pass `SortType::F32` with `&mut [u32]` data and get a runtime error instead of a compile error. Loses all type safety benefits.

### Alternative D: Separate Sorter Per Type

```rust
// REJECTED
let u32_sorter = GpuSorter::<u32>::new()?;
let f32_sorter = GpuSorter::<f32>::new()?;
```

**Why rejected**: Forces users to create multiple sorter instances, each with its own Metal device/queue/PSO cache. Wastes GPU resources. Users who sort multiple types (e.g., depth sort + index sort in the same frame) would need 2+ sorters. The current single-sorter design is correct -- one device, one queue, multiple sort methods.

### Alternative E: Generic SortBuffer Without Sealed Trait

```rust
// REJECTED
pub struct SortBuffer<T: Copy + 'static> { ... }
```

**Why rejected**: Allows `SortBuffer<String>`, `SortBuffer<Vec<u8>>`, `SortBuffer<[f32; 4]>` -- none of which can be GPU-sorted. The sealed trait constrains the generic parameter to exactly the types the GPU supports. Without the seal, users get a runtime error ("unsupported type") instead of a compile error ("trait bound not satisfied").

---

## References

- [rdst -- Rust radix sort crate](https://crates.io/crates/rdst) -- Trait-based (`RadixKey`) approach for custom types
- [radsort -- Rust scalar radix sort](https://crates.io/crates/radsort) -- Free-function API with sealed `Key` trait
- [voracious_radix_sort -- Rust radix sort](https://docs.rs/voracious_radix_sort) -- `Radixable` trait with `Dispatcher` pattern
- [rayon `ParallelSliceMut`](https://docs.rs/rayon/latest/rayon/slice/trait.ParallelSliceMut.html) -- `par_sort_by_key` signature pattern
- [wgpu_sort -- WebGPU radix sort](https://docs.rs/wgpu_sort) -- GPU sort API with separate key/value buffers
- [CUB DeviceRadixSort](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html) -- Production GPU sort API reference
- [Rust `f32::total_cmp`](https://doc.rust-lang.org/std/primitive.f32.html#method.total_cmp) -- IEEE 754 total ordering in Rust
- [Rust argsort pattern (community)](https://gist.github.com/mbhall88/80cd054410f960cea0c451b8b0edae71) -- enumerate-sort-extract idiom
- [Rust internals: argsort feature request](https://internals.rust-lang.org/t/feature-request-methods-for-sorting-reordering-with-indices/15568)

## Questions & Answers

### Q1: Should there be a zero-copy argsort_buffer variant?
**Answer**: Yes, include argsort_u32_buffer, argsort_i32_buffer, argsort_f32_buffer variants that return SortBuffer<u32> indices in GPU memory.
**Impact**: Enables GPU pipelines where indices feed directly into gather kernels. Adds 4 methods but completes the zero-copy story.

### Q2: Should sort_pairs use a dedicated SortPairsBuffer or two separate SortBuffers?
**Answer**: Two separate SortBuffers. Reuses existing types, no new public type needed.
**Impact**: sort_pairs_u32_buffer(&SortBuffer<u32>, &SortBuffer<u32>) signature. Users manage two buffers but the API stays simpler.

### Q3: Should f64 key-value pairs be included?
**Answer**: No, exclude sort_pairs_f64. f64 is niche + 8-pass sort + doubled bandwidth is too costly.
**Impact**: f64 is keys-only (sort_f64, argsort_f64). Reduces API surface and testing burden. However, sort_pairs_u64 and sort_pairs_i64 ARE included — database IDs and timestamps are concrete key-value use cases that justify the bandwidth cost.
