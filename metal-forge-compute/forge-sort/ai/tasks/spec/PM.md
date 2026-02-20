# Product Manager Analysis: forge-sort Multi-Type Support

## Executive Summary

forge-sort currently sorts `u32` keys only. This analysis covers extending the library to support **signed integers (i32)**, **floating-point numbers (f32/f64)**, and **key-value pair sorting** — the three capabilities needed to move forge-sort from a technical demo to a general-purpose GPU sort primitive on Apple Silicon.

The recommendation is to ship these features in priority order: **(1) i32, (2) f32, (3) key-value pairs, (4) f64**. Items 1 and 2 are low-risk bit transformations that unlock the majority of use cases at near-zero performance cost. Key-value pairs require shader changes and double the bandwidth. f64 requires a 5th radix pass.

---

## Research Findings

### How Production GPU Sort Libraries Handle Multi-Type Support

#### NVIDIA CUB DeviceRadixSort

CUB is the de facto standard for GPU sorting. Its API provides four method families, each templated on key and value types ([CUB DeviceRadixSort API](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html)):

- `SortKeys` / `SortKeysDescending` — keys only
- `SortPairs` / `SortPairsDescending` — key-value pairs

CUB handles signed and floating-point types transparently via **bitwise transformations applied before sorting and reversed when writing to the final output buffer** ([CUB Issue #293: Document Radix Sort Bitwise Transformations](https://github.com/NVIDIA/cub/issues/293)):

| Key Type | Pre-Sort Transformation | Post-Sort Reversal |
|----------|------------------------|-------------------|
| Unsigned integers | None | None |
| Signed integers | Flip sign bit (XOR `0x80000000`) | Flip sign bit back |
| Positive floats | Flip sign bit | Flip sign bit back |
| Negative floats | Flip all bits | Flip all bits back |

CUB also supports an optional `[begin_bit, end_bit)` range to sort on a subset of key bits, which reduces the number of radix passes and improves performance for keys with limited dynamic range.

Key-value pairs use a **DoubleBuffer** structure that tracks which of two buffers is "current" after sorting. The API does NOT sort in-place — it ping-pongs between two buffers, and the "current" pointer is updated to indicate where the sorted output lives.

#### Thrust sort_by_key

Thrust's `sort_by_key` sorts key-value pairs on GPU. Performance analysis reveals a critical insight: **key-value sorting has significantly worse occupancy than key-only sorting** due to doubled register pressure ([Thrust sort_by_key Issue #717](https://github.com/NVIDIA/cccl/issues/717)). In benchmarks with ~1M pairs, struct values took 10ms vs 5.8ms for integer values. Sorting keys with indices and then gathering values was marginally faster (9.5ms) than direct key-value sorting.

This suggests a "sort keys + indices, then gather" approach can be competitive with native key-value sorting, especially when values are large structs.

#### VkRadixSort (Vulkan)

VkRadixSort provides a Vulkan GLSL implementation of GPU radix sort with support for key-value pair sorting ([VkRadixSort GitHub](https://github.com/MircoWerner/VkRadixSort)). It focuses on 32-bit keys and uses a 4-bit radix (16 bins per pass, 8 passes), which is simpler but slower than 8-bit approaches.

#### wgpu_sort (WebGPU/Rust)

The closest Rust ecosystem comparator. wgpu_sort currently supports **only 32-bit key-value pairs** and interprets keys as unsigned integers ([wgpu_sort docs](https://docs.rs/wgpu_sort)). It notes that non-negative floats can be sorted by reinterpreting as unsigned integers, but NaN/Inf produce unexpected results. The implementation is derived from Google Fuchsia's Vulkan radix sort.

#### AMD FidelityFX Parallel Sort

AMD's GPU sort implementation sorts key-value pairs and uses a 4-bit radix with 4 passes for 16-bit payloads ([AMD GPUOpen FidelityFX Parallel Sort](https://gpuopen.com/fidelityfx-parallel-sort/)).

### Float Bit Transformation (Definitive Algorithm)

The canonical float-to-sortable-integer transformation from [Stereopsis Radix Tricks](https://stereopsis.com/radix.html):

```c
// Pre-sort: float bits -> sortable uint
uint32 FloatFlip(uint32 f) {
    uint32 mask = -int32(f >> 31) | 0x80000000;
    return f ^ mask;
}

// Post-sort: sortable uint -> float bits
uint32 IFloatFlip(uint32 f) {
    uint32 mask = ((f >> 31) - 1) | 0x80000000;
    return f ^ mask;
}
```

**Logic**: Always flip the sign bit. If the original sign bit was set (negative float), flip ALL other bits too. This converts IEEE 754 signed-magnitude representation into a monotonically increasing unsigned integer sequence. The transformation is a bijection — every float maps to a unique uint32 and back.

**NaN handling**: IEEE 754 NaN values have exponent bits all-set and non-zero mantissa. After FloatFlip, NaN values sort to the end (after +Infinity). This is the standard behavior in CUB and is acceptable for most use cases. Explicit NaN-to-end behavior should be documented.

### Key-Value Bandwidth Impact

Sorting key-value pairs doubles the memory bandwidth per element (8 bytes vs 4 bytes). Since GPU radix sort is memory-bandwidth bound, this translates to approximately a **2x slowdown** in throughput ([Stehle & Satish: Memory Bandwidth-Efficient Hybrid Radix Sort](https://arxiv.org/pdf/1611.01137)). The AMD Onesweep extension with circular buffers partially mitigates this by reducing global memory traffic ([AMD GPUOpen: Boosting GPU Radix Sort](https://gpuopen.com/learn/boosting_gpu_radix_sort/)).

For forge-sort specifically, with current u32 throughput of ~5200 Mk/s zero-copy, key-value pair throughput would be expected at ~2600-3000 Mk/s — still far above CPU sort speeds.

---

## Business Context and Use Cases

### Use Case Analysis by Type

#### Signed Integer Sorting (i32)

| Use Case | Description | Demand |
|----------|-------------|--------|
| Database operations | Sorting signed columns (ages, balances, temperatures, deltas) | **High** — most real-world integer data is signed |
| Array indexing with offsets | Sorting relative positions, coordinate offsets | High |
| Signal processing | Sorting signed audio/sensor samples | Medium |
| Game physics | Sorting spatial hash keys (signed coordinates) | Medium |

**Market reality**: Most integer data in the wild is signed. A sort library that only handles unsigned integers forces users to manually offset their data (`val + i32::MIN as u32`), which is error-prone and inconvenient. This is the single most common user frustration with unsigned-only sort libraries.

#### Floating-Point Sorting (f32)

| Use Case | Description | Demand |
|----------|-------------|--------|
| 3D rendering / depth sorting | Sort particles, transparent objects, splats by camera distance | **Critical** — the #1 GPU sort use case in graphics |
| Gaussian splatting | Sort 3DGS splats by depth for alpha compositing | Critical — the hottest GPU sort workload in 2024-2025 |
| ML preprocessing | Sort embeddings, activations, loss values for analysis | High |
| Scientific computing | Sort simulation results (temperatures, pressures, velocities) | High |
| Financial data | Sort prices, returns, risk metrics | Medium |
| Search ranking | Sort search results by relevance score | Medium |

**Market reality**: f32 sorting is arguably the most commercially important type. The explosion of 3D Gaussian Splatting (3DGS) has created massive demand for real-time GPU depth sorting of millions of splats. wgpu_sort was built specifically for this use case ([wgpu_sort crates.io](https://crates.io/crates/wgpu_sort)). An Apple Silicon-native f32 sort at 5000+ Mk/s would be the fastest available option for Metal-based 3DGS renderers.

#### Floating-Point Sorting (f64)

| Use Case | Description | Demand |
|----------|-------------|--------|
| Scientific computing | Double-precision simulation results | Medium |
| Financial data | High-precision monetary calculations | Medium |
| Geospatial | GPS coordinates, distances | Low-Medium |

**Market reality**: f64 demand is lower than f32. Most GPU workloads use f32 or f16. f64 requires 8 bytes per key and 5 radix passes (40 bits for 8-bit radix), making it the most expensive type to support. However, it is expected by users who see f32 support.

#### Unsigned 64-bit Integer Sorting (u64)

| Use Case | Description | Demand |
|----------|-------------|--------|
| Database primary keys | Sorting u64 auto-increment IDs, UUIDs mapped to u64 | **High** — u64 is the standard key type in modern databases |
| Timestamps | Sorting Unix timestamps (nanosecond precision), chrono epochs | **High** — temporal data is ubiquitous |
| Hash values | Sorting hash digests truncated to 64 bits, FNV/SipHash outputs | Medium |
| File sizes / offsets | Sorting large file byte offsets, memory addresses | Medium |
| Network identifiers | Sorting IPv6 address components, MAC addresses as u64 | Low-Medium |

**Market reality**: u64 is the baseline 64-bit type — it requires zero bit transformation (simplest 64-bit type). Once the f64 8-pass pipeline exists (IS_64BIT function constant, ulong key storage), u64 support is essentially free. Database and timestamp sorting are high-demand use cases that justify inclusion alongside f64 in Phase 3.

#### Signed 64-bit Integer Sorting (i64)

| Use Case | Description | Demand |
|----------|-------------|--------|
| Financial data | Sorting monetary amounts in cents/basis points (avoids floating-point) | **High** — finance uses i64 to avoid float rounding |
| Temporal deltas | Sorting time differences, durations (can be negative) | Medium |
| Database signed columns | Sorting 64-bit signed columns (large counters, sequence numbers) | Medium |
| Scientific measurements | Sorting high-precision integer-encoded measurements | Low-Medium |

**Market reality**: i64 uses the same 8-pass pipeline as f64/u64 with a simple sign bit XOR (`0x8000000000000000`), identical in concept to the i32 transform but 64-bit. Once u64 works, i64 is trivially derived. Financial applications that use i64 to avoid floating-point rounding errors represent a concrete, high-value use case.

#### Key-Value Pair Sorting

| Use Case | Description | Demand |
|----------|-------------|--------|
| Depth-sorted rendering | Sort (depth, instance_id) for draw order | **Critical** — cannot render correctly without values |
| Database index building | Sort (key, row_id) for B-tree construction | **High** |
| Histogram / reduce-by-key | Sort (bin, value) then scan for boundaries | High |
| Spatial hashing | Sort (cell_hash, particle_id) for neighbor queries | High |
| Search index | Sort (score, doc_id) for ranked retrieval | Medium |
| Graph algorithms | Sort (edge_weight, edge_id) for MST | Medium |

**Market reality**: Key-value sorting is what makes a sort library usable in practice. Sorting raw values without tracking which element went where is only useful for verification and statistics. Nearly every production use case requires associating a payload (index, ID, pointer) with each sorted key. CUB, Thrust, VkRadixSort, and wgpu_sort all provide key-value sorting as a primary API.

---

## Priority Ordering

### Recommended Implementation Order

| Priority | Feature | Effort | Risk | Unlock Value | Rationale |
|----------|---------|--------|------|-------------|-----------|
| **P0** | `sort_i32` | 1-2 days | Very Low | High | Pure bit transformation (XOR sign bit). Zero shader changes needed — transform on CPU before/after GPU sort. Unlocks all signed integer use cases. |
| **P1** | `sort_f32` | 1-2 days | Low | Very High | Slightly more complex bit transformation (sign-dependent XOR). Same zero-shader-change approach. Unlocks the massive 3DGS and graphics market. |
| **P2** | Key-value pairs (`sort_pairs_u32`, etc.) | 5-8 days | Medium | Very High | Requires shader modifications (load/scatter values alongside keys). Doubles bandwidth. But unlocks nearly all production use cases. |
| **P3** | `sort_f64`, `sort_u64`, `sort_i64` | 3-5 days | Medium | Medium | Requires 64-bit data handling in shaders (2x register pressure) and 8 radix passes. u64/i64 are essentially free once the f64 pipeline exists — u64 needs no transform, i64 needs only sign bit XOR. |

### Rationale for This Order

1. **i32 and f32 first**: These are "free" features — the GPU shader does not need to change at all. The transformation happens on the CPU side (or in a trivial pre/post-processing kernel). The existing u32 sort handles the actual sorting. This means i32 and f32 support can ship with **zero risk to the existing u32 sort correctness or performance**.

2. **Key-value pairs before f64**: Key-value pairs unlock far more use cases than f64. They are also architecturally necessary for the rendering/splatting use cases that drive f32 demand. Without key-value pairs, f32 sorting is only useful for statistics (median, percentile), not for rendering.

3. **f64/u64/i64 last**: The 64-bit types share the same IS_64BIT pipeline infrastructure (8 passes, ulong key storage). f64 requires FloatFlip, u64 requires no transform (simplest 64-bit type), and i64 requires only sign bit XOR (same as i32 but 64-bit). Once the 64-bit shader variant is built for f64, u64 and i64 are essentially zero marginal cost. All three ship together in Phase 3.

---

## Technical Approach Recommendation

### i32 and f32: CPU-Side Bit Transformation (Zero Shader Changes)

The fastest path to i32 and f32 support requires **no shader modifications**:

```
sort_i32(data: &mut [i32]):
  1. Reinterpret as &mut [u32] (same bits)
  2. XOR each element with 0x80000000 (flip sign bit)
  3. Call sort_u32()
  4. XOR each element with 0x80000000 (flip sign bit back)

sort_f32(data: &mut [f32]):
  1. Reinterpret as &mut [u32] (same bits)
  2. Apply FloatFlip to each element
  3. Call sort_u32()
  4. Apply IFloatFlip to each element
```

For the zero-copy `SortBuffer` API, the transformation should happen **in a GPU kernel** (a trivial 1-dispatch pre/post-process) to avoid CPU-GPU memory round-trips. This is a 5-line kernel:

```metal
kernel void transform_f32_pre(device uint* data, uint gid [[thread_position_in_grid]]) {
    uint f = data[gid];
    uint mask = -int(f >> 31) | 0x80000000;
    data[gid] = f ^ mask;
}
```

**Performance impact**: The transformation kernels are trivially memory-bandwidth-bound. At 200 GB/s bandwidth, transforming 16M elements (64 MB) takes ~0.3ms. Against the 3.1ms sort time, this is ~10% overhead. Acceptable.

**Alternative**: Embed the transformation directly in the sort shader via a `key_transform` parameter. This avoids the extra dispatch but complicates the shader and couples type awareness into every kernel. CUB does this; it is more efficient but higher-risk. Recommend starting with separate transform dispatches and optimizing later if the 10% overhead matters.

### Key-Value Pairs: Shader Modifications Required

Key-value pair sorting requires changes to all 4 kernels:

1. **sort_msd_histogram**: No change needed (only reads keys)
2. **sort_msd_prep**: No change needed (only processes histograms)
3. **sort_msd_atomic_scatter**: Must scatter values alongside keys (`dst_vals[gp] = vals[idx]`)
4. **sort_inner_fused**: Must load, track, and scatter values in every pass

The shader changes are straightforward but double the register pressure and TG memory usage:
- 16 key registers + 16 value registers = 32 registers per thread for element data
- TG memory for value reordering if local scatter is used

**Two implementation strategies**:

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **A: Native key-value shader** | Carry values alongside keys in all 4 kernels | Optimal bandwidth (1 pass reads keys+values) | 4 shader variants, doubled register pressure, potential spilling |
| **B: Sort keys+indices, then gather** | Sort `(key, original_index)` pairs, then gather values by index | Reuses existing shader with minor change, simpler | Extra gather pass, 50% more memory (keys + indices + values) |

**Recommendation**: Strategy B for initial release, Strategy A as optimization. Strategy B is proven (Thrust benchmarks show it is competitive), lower risk, and reuses the existing shader. Strategy A can be added later for users who need maximum throughput.

### f64: Requires Significant Shader Work

f64 keys are 8 bytes. Current shaders assume 4-byte elements throughout. Changes needed:

- All `uint` key variables become `ulong` (or `uint2` for register efficiency)
- TILE_SIZE may need to halve (4096 * 8 = 32KB, approaching TG limits)
- 5 radix passes instead of 4 (40 bits for 8-bit radix, or 5 full bytes)
- MSD scatter operates on bits[32:39] instead of bits[24:31]
- Inner fused sort handles 4 inner passes instead of 3

This is essentially a variant of the entire shader, not a small modification. Recommend deferring until there is concrete user demand.

---

## API Design Recommendations

### Principle: Type-Safe, Familiar, Zero-Surprise

Users should be able to sort any supported type without thinking about bit transformations. The API should feel like Rust's standard `sort()` — you call it, it works.

### Recommended API (User Perspective)

```rust
// ── Keys Only ──────────────────────────────────────────────────
// Existing
sorter.sort_u32(&mut data)?;           // &mut [u32]
sorter.sort_buffer(&buf)?;             // SortBuffer (zero-copy u32)

// New: signed integers
sorter.sort_i32(&mut data)?;           // &mut [i32]

// New: floating point
sorter.sort_f32(&mut data)?;           // &mut [f32]

// Future: 64-bit types
sorter.sort_u64(&mut data)?;           // &mut [u64]
sorter.sort_i64(&mut data)?;           // &mut [i64]
sorter.sort_f64(&mut data)?;           // &mut [f64]

// ── Key-Value Pairs ────────────────────────────────────────────
// Sort keys, rearrange values to match
sorter.sort_pairs_u32(&mut keys, &mut values)?;   // (&mut [u32], &mut [V])
sorter.sort_pairs_i32(&mut keys, &mut values)?;    // (&mut [i32], &mut [V])
sorter.sort_pairs_f32(&mut keys, &mut values)?;    // (&mut [f32], &mut [V])

// Zero-copy key-value buffer
let kv_buf = sorter.alloc_sort_pairs_buffer::<V>(capacity);
sorter.sort_pairs_buffer(&kv_buf)?;

// ── Utilities ──────────────────────────────────────────────────
// Sort and return permutation indices (useful for multi-column sort)
let indices: Vec<u32> = sorter.argsort_u32(&data)?;
let indices: Vec<u32> = sorter.argsort_u64(&data)?;
let indices: Vec<u32> = sorter.argsort_i64(&data)?;
let indices: Vec<u32> = sorter.argsort_f32(&data)?;
```

### API Design Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Separate methods vs generic | **Separate methods** (`sort_i32`, `sort_f32`) | Explicit, no trait complexity, matches CUB's API style. Generic `sort<T>()` requires trait bounds that leak GPU implementation details. |
| Value type constraint | **`V: Copy + 'static + Sized`** with `size_of::<V>() == 4` initially | Start with 32-bit values (u32, i32, f32, enum indices). Expand to 64-bit and arbitrary structs later. |
| Descending sort | **Defer** | Add `sort_u32_descending()` etc. later. Can be trivially implemented by XOR-ing all key bits (bitwise NOT) before sorting and reversing after. |
| Partial-bit sorting | **Defer** | CUB's `begin_bit`/`end_bit` optimization is powerful but adds API complexity. Add when a user needs it. |
| NaN handling | **Document: NaN sorts after +Inf** | Matches IEEE 754 totalOrder and CUB behavior. Do not silently filter NaN. |
| Stability guarantee | **Document: sort is stable** | Radix sort is inherently stable. This is valuable for key-value pairs and should be promised in the API docs. |
| In-place vs out-of-place | **In-place for `sort_*` methods, ping-pong internally** | Matches current API. Users see in-place behavior. Internal buffer ping-pong is an implementation detail. |

### SortBuffer Extensions for Typed Access

```rust
// Typed SortBuffer variants
pub struct SortBufferI32 { /* wraps SortBuffer with i32 accessors */ }
pub struct SortBufferF32 { /* wraps SortBuffer with f32 accessors */ }

// Or: generic typed wrapper
pub struct TypedSortBuffer<T: SortKey> { inner: SortBuffer, _phantom: PhantomData<T> }

// Usage:
let buf: TypedSortBuffer<f32> = sorter.alloc_typed_sort_buffer(1_000_000);
buf.as_mut_slice()[0] = 3.14;  // &mut [f32]
sorter.sort_typed_buffer(&buf)?;
```

**Recommendation**: Start with simple wrapper structs (`SortBufferF32`). Add the generic `TypedSortBuffer<T>` later if the pattern proves useful. Avoid premature generalization.

---

## Risk Assessment

### Feature-Level Risks

| Feature | Risk | Likelihood | Impact | Mitigation |
|---------|------|-----------|--------|-----------|
| i32 sort | Transformation off-by-one (e.g., `i32::MIN` maps wrong) | Very Low | Data corruption | Test with `i32::MIN`, `i32::MAX`, 0, -1, 1 explicitly |
| f32 sort | NaN/Inf handling surprise | Low | User confusion | Document behavior. Test with NaN, +Inf, -Inf, -0.0, +0.0 |
| f32 sort | Denormalized floats sort incorrectly | Very Low | Silent wrong output | FloatFlip handles denormals correctly (proven algorithm) |
| f32 sort | Performance regression from transform overhead | Low | 10% slowdown | Benchmark. If unacceptable, fuse transform into sort kernels |
| Key-value pairs | Shader register spilling from doubled data | Medium | 30-50% perf loss | Monitor register usage. Fall back to sort+gather if spilling |
| Key-value pairs | TG memory overflow (doubled element storage) | Medium | Correctness failure | Halve TILE_SIZE for KV variant, or use sort+gather strategy |
| Key-value pairs | Buffer management complexity (keys + values + scratch) | Medium | API confusion | Keep buffer management internal. User sees `sort_pairs(&mut k, &mut v)` |
| f64 sort | Apple Silicon f64 performance is poor | High | Disappointing throughput | Document that f64 is 3-5x slower than f32. Consider CPU fallback for small N |
| f64 sort | 8-pass sort exceeds reasonable latency | Medium | User frustration | 8 passes for 64-bit types is inherent to 8-bit radix. Profile carefully. |
| u64 sort | None beyond general 64-bit pipeline risk | Very Low | N/A | u64 uses no transform — the simplest 64-bit type. If f64 works, u64 works. |
| i64 sort | Sign bit XOR off-by-one for 64-bit mask | Very Low | Data corruption | Test with i64::MIN, i64::MAX, -1, 0, 1 explicitly. Same XOR pattern proven for i32. |
| All types | Increased maintenance burden (4+ shader variants) | Medium | Technical debt | Use Metal function constants or preprocessor macros to generate variants from one source |

### Architecture-Level Risks

| Risk | Description | Mitigation |
|------|-------------|-----------|
| Shader variant explosion | 6 types * 2 modes (keys/KV) * 4 kernels = 48 PSOs | Use Metal function constants (`[[function_constant(0)]]`) to compile variants from one shader source. 64-bit types share IS_64BIT PSOs. |
| API surface growth | 20+ public methods instead of 3 | Group by trait or module. Keep top-level API clean. |
| Test matrix explosion | 6 types * 28 existing tests = 168+ tests | Use test macros/generics. Parameterize existing tests over types. |
| Breaking change risk | Adding type support may tempt API redesign | Keep `sort_u32` and `sort_buffer` unchanged. New methods only. |

---

## Success Metrics

### Correctness Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| i32 sort correctness | 100% match with `[i32].sort()` for all test cases | Comparison against Rust stdlib sort |
| f32 sort correctness | 100% match with `[f32].sort_by(f32::total_cmp)` for all test cases | Comparison against Rust stdlib with total ordering |
| f32 NaN handling | NaN values sort after +Inf, NaN == NaN in sort order | Explicit NaN position test |
| f32 special values | -0.0 sorts before +0.0 (or equal — document choice) | Explicit special-value test |
| Key-value stability | Values preserve relative order for equal keys | Explicit stability test with duplicate keys |
| Key-value integrity | Every input (key, value) pair appears exactly once in output | Bijection test: sorted output is a permutation of input |

### Performance Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| i32 sort throughput @ 16M | >= 2800 Mk/s end-to-end | Same as u32 (transformation is negligible) |
| f32 sort throughput @ 16M | >= 2500 Mk/s end-to-end | Allow 10% overhead for FloatFlip transform |
| f32 zero-copy throughput @ 8M | >= 4500 Mk/s | Allow 15% for GPU-side transform dispatch |
| KV u32 sort throughput @ 16M | >= 1400 Mk/s end-to-end | 2x bandwidth = ~2x slowdown |
| KV u32 zero-copy throughput @ 8M | >= 2500 Mk/s | Same ratio as keys-only |
| f64 sort throughput @ 16M | >= 800 Mk/s | 8 passes + 8-byte elements |
| u64 sort throughput @ 16M | >= 800 Mk/s | 8 passes + 8-byte elements, no transform (baseline 64-bit) |
| i64 sort throughput @ 16M | >= 750 Mk/s | 8 passes + 8-byte elements, sign bit XOR transform |
| Transform kernel overhead | < 0.5ms @ 16M | Should be pure bandwidth-bound |

### Adoption Metrics

| Metric | Target | Timeframe |
|--------|--------|-----------|
| API completeness | All 6 types (u32/i32/f32/u64/i64/f64) + KV pairs | 3-4 weeks |
| Test coverage | 130+ tests (28 current * 6 types + KV-specific) | Alongside implementation |
| Documentation | Complete rustdoc with examples for each type | Alongside implementation |
| Benchmark suite | Cover all types + KV pairs | Alongside implementation |

---

## Competitive Positioning

| Library | Platform | u32 | i32 | f32 | u64 | i64 | f64 | KV Pairs | Mk/s @ 16M |
|---------|----------|-----|-----|-----|-----|-----|-----|----------|-------------|
| **forge-sort (current)** | Metal/Apple Silicon | Yes | No | No | No | No | No | No | 2859 (e2e) / 5207 (zc) |
| **forge-sort (target)** | Metal/Apple Silicon | Yes | Yes | Yes | Yes | Yes | Yes | Yes | 2800+ (e2e) |
| CUB DeviceRadixSort | CUDA/NVIDIA | Yes | Yes | Yes | Yes | Yes | Yes | Yes | ~29,400 (A100) |
| wgpu_sort | WebGPU/Cross-platform | Yes | No | Partial | No | No | No | Yes (32-bit only) | ~53 (RTX 3070) |
| VkRadixSort | Vulkan/Cross-platform | Yes | No | No | No | No | No | Yes | ~53 (RTX 3070) |
| VSort | Metal/Apple Silicon | Yes | No | No | No | No | No | No | 99 |
| MPS | Metal/Apple Silicon | — | — | — | — | — | — | — | No sort API |

**Key insight**: There is NO Metal-native GPU sort library that supports signed integers, floats, 64-bit integers, or key-value pairs. forge-sort has the opportunity to be the first and only option for these use cases on Apple Silicon.

---

## Implementation Phases

### Phase 1: Signed and Float Key Sorting (P0+P1, 3-4 days)

**Scope**: `sort_i32()`, `sort_f32()`, zero-copy variants, typed `SortBuffer` wrappers

**Deliverables**:
- CPU-side bit transformation for `sort_i32` and `sort_f32`
- GPU transform kernel for zero-copy `SortBuffer` path
- `SortBufferI32` and `SortBufferF32` typed wrappers
- 40+ new tests (existing test matrix * 2 new types + special-value tests)
- Updated benchmarks

**Risk**: Very low. No shader changes to proven sort kernels.

### Phase 2: Key-Value Pair Sorting (P2, 5-8 days)

**Scope**: `sort_pairs_u32()`, `sort_pairs_i32()`, `sort_pairs_f32()`, zero-copy KV buffer

**Deliverables**:
- Strategy B implementation (sort keys+indices, then gather)
- `SortPairsBuffer` for zero-copy KV sorting
- 30+ new KV-specific tests (stability, integrity, performance)
- KV benchmark suite

**Risk**: Medium. New gather kernel, doubled memory allocation, potential for subtle index bugs.

### Phase 3: 64-bit Support (P3, 3-5 days)

**Scope**: `sort_f64()`, `sort_u64()`, `sort_i64()`, zero-copy variants, argsort, key-value pairs for u64/i64

**Deliverables**:
- 64-bit shader variant with IS_64BIT function constant
- 8-pass sort pipeline (1 MSD + 7 inner across 3 fused dispatches)
- `sort_u64()`, `sort_i64()`, `sort_f64()` — memcpy and zero-copy paths
- `argsort_u64()`, `argsort_i64()`, `argsort_f64()` — all 64-bit argsort variants
- `sort_pairs_u64()`, `sort_pairs_i64()` — key-value pairs with 64-bit keys + 32-bit values
- Zero-copy buffer variants: `sort_u64_buffer()`, `sort_i64_buffer()`, `sort_f64_buffer()`, `argsort_u64_buffer()`, `argsort_i64_buffer()`
- u64 uses TRANSFORM_MODE=0 (no transform — the baseline 64-bit type)
- i64 uses TRANSFORM_MODE=1 (XOR sign bit `0x8000000000000000`, same concept as i32 but 64-bit)
- f64 uses FloatFlip/IFloatFlip (64-bit width)
- 30+ tests covering all three 64-bit types
- 64-bit benchmarks

**Risk**: Medium. Significant shader changes, potential register pressure issues. However, u64 and i64 add zero marginal shader complexity once f64 works — they differ only in the transform mode constant.

### Phase 4: Optimization (Optional, 3-5 days)

**Scope**: Fused transform kernels, native KV shaders (Strategy A), descending sort

**Deliverables**:
- Fuse FloatFlip/IFloatFlip into sort kernels (eliminate transform dispatch)
- Native key-value scatter kernel (eliminate gather pass)
- `sort_*_descending()` variants
- Performance parity between types (eliminate transform overhead)

---

## Open Questions

These questions do not need to be resolved before implementation begins, but should be addressed during Phase 1:

1. **-0.0 vs +0.0 ordering**: IEEE 754 defines -0.0 == +0.0, but FloatFlip maps them to different unsigned integers, so -0.0 will sort before +0.0. This matches `f32::total_cmp()` behavior. Should we document this as intentional (matching Rust's `total_cmp`) or add special handling to treat them as equal?

2. **NaN payload preservation**: Different NaN payloads (signaling vs quiet, different mantissa bits) will sort to different positions. Should we document this, or normalize all NaN to a single value before sorting?

3. **Key-value value size**: Start with 32-bit values only (`u32` values), or support arbitrary `Copy` types from day one? 32-bit is simpler and covers the common `(key, index)` pattern. Arbitrary sizes add complexity.

4. **Argsort as a first-class API**: Should `argsort()` (return indices rather than sorting in-place) be a top-level method? It is the natural primitive for key-value sorting via Strategy B and useful independently.

---

## References

- [CUB DeviceRadixSort API](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html) — NVIDIA's reference GPU sort implementation
- [CUB Issue #293: Radix Sort Bitwise Transformations](https://github.com/NVIDIA/cub/issues/293) — Documentation of signed/float transformations
- [Stereopsis: Radix Sort Tricks](https://stereopsis.com/radix.html) — Canonical FloatFlip/IFloatFlip algorithm
- [Thrust sort_by_key Performance Issue #717](https://github.com/NVIDIA/cccl/issues/717) — KV sorting occupancy analysis
- [Stehle & Satish: Memory Bandwidth-Efficient Hybrid Radix Sort](https://arxiv.org/pdf/1611.01137) — Bandwidth analysis for KV pairs
- [AMD GPUOpen: Boosting GPU Radix Sort](https://gpuopen.com/learn/boosting_gpu_radix_sort/) — Onesweep circular buffer optimization
- [AMD FidelityFX Parallel Sort](https://gpuopen.com/fidelityfx-parallel-sort/) — AMD's KV sort implementation
- [VkRadixSort](https://github.com/MircoWerner/VkRadixSort) — Vulkan GLSL radix sort reference
- [wgpu_sort](https://docs.rs/wgpu_sort) — Rust WebGPU radix sort (closest ecosystem comparator)
- [NVIDIA GPU Gems: Improved GPU Sorting](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting) — GPU sort use cases survey
- [Sorting with GPUs: A Survey (2017)](https://arxiv.org/pdf/1709.02520) — Comprehensive GPU sorting literature review

## Questions & Answers

### Q1: Should sort_f32 treat -0.0 and +0.0 as distinct?
**Answer**: Yes, use `total_cmp` semantics (-0.0 < +0.0). The bit transformation naturally produces this ordering.
**Impact**: Simplifies implementation — no special-case handling needed.

### Q2: Should key-value pairs initially support only 32-bit values?
**Answer**: Yes, 32-bit values only (u32 index pattern). Simpler, faster to ship.
**Impact**: Key-value buffer is 8 bytes/element (4 key + 4 value). Covers 90%+ of use cases.

### Q3: Should argsort() be a first-class API method?
**Answer**: Yes, include argsort as first-class. It's the primitive for Strategy B key-value sorting and independently useful.
**Impact**: argsort becomes the foundation — sort_pairs can be built on top of it.

### Q4: Should descending sort variants be included?
**Answer**: Defer. Trivial to add later (XOR all bits), keeps initial scope focused.
**Impact**: Reduces API surface in v2.0, can add in v2.1.
