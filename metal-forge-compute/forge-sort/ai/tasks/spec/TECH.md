# Technical Architecture: forge-sort v2 Multi-Type Support

## Executive Summary

This document specifies the implementation architecture for extending forge-sort from u32-only to support i32, f32, u64, i64, f64, argsort, and key-value pair sorting on Apple Silicon Metal 3.2. The design achieves multi-type support through **three mechanisms operating at different layers**:

1. **GPU-side bit transformation kernels** (2 new trivial kernels) for i32/f32/f64 pre/post-sort conversion
2. **Metal function constants** for compile-time specialization of the 4 existing sort kernels (key-only vs key-value, 32-bit vs 64-bit element width)
3. **Rust-side `SortBuffer<T>` with sealed `SortKey` trait** for type-safe zero-copy buffers

The approach minimizes shader duplication (0 new sort kernel source files), preserves the proven 4-dispatch pipeline, and adds at most 2 extra dispatches per sort call (transform pre + transform post). f64 extends the pipeline to 8 radix passes across 6 dispatches.

---

## 1. Shader Architecture

### 1.1 Design Decision: Function Constants (Not Separate Kernel Files)

The central shader design question is how to handle multiple types across the 4 sort kernels. Three approaches were evaluated:

| Approach | Description | PSOs | Maintenance | Performance |
|----------|-------------|------|-------------|-------------|
| **A: Separate .metal files** | `sort_u32.metal`, `sort_kv.metal`, `sort_f64.metal` | 12+ | High (3 copies of same algorithm) | Optimal (no branches) |
| **B: Preprocessor macros** | `#define KEY_TYPE uint` with multiple compilations | 12+ | Medium (one source, but build.rs complexity) | Optimal (no branches) |
| **C: Function constants** | `constant bool HAS_VALUES [[function_constant(0)]]` | 8-12 | **Low** (one source, runtime specialization) | **Optimal** (dead code eliminated at PSO creation) |

**Chosen: C (Function Constants)**. Metal function constants allow declaring shader parameters that are resolved at pipeline state creation time. The Metal compiler eliminates dead branches, producing the same optimal code as separate files -- but from a single source. This is the [Apple-recommended approach](https://developer.apple.com/documentation/metal/using-function-specialization-to-build-pipeline-variants) for kernel variants.

Key advantages:
- **One shader source file** (`sort.metal`) handles all type combinations
- **Dead code elimination**: `if (has_values)` branches are compiled away when `HAS_VALUES=false`, producing identical machine code to the current u32-only kernels
- **No build.rs complexity**: One .air + one .metallib. Variants are created at PSO creation time in Rust via `MTLFunctionConstantValues`
- **PsoCache integration**: Cache key becomes `"sort_msd_scatter:kv=true"` or similar, mapping to the same MTLFunction with different constant values

### 1.2 Function Constant Definitions

```metal
// ═══════════════════════════════════════════════════════════════
// Function constants for type specialization
// ═══════════════════════════════════════════════════════════════

// Whether this sort carries values alongside keys (argsort/key-value mode)
constant bool HAS_VALUES [[function_constant(0)]];

// Whether keys are 64-bit (f64/u64 mode). When true, each "element"
// is 8 bytes and we use ulong for key storage.
constant bool IS_64BIT [[function_constant(1)]];

// Compile-time defaults (used when constants are not set at PSO creation)
constant bool has_values = is_function_constant_defined(HAS_VALUES) ? HAS_VALUES : false;
constant bool is_64bit   = is_function_constant_defined(IS_64BIT)   ? IS_64BIT   : false;
```

Only **2 function constants** are needed because:
- **Type transformations (i32/f32/f64 bit flips)** are handled by separate transform kernels, not by the sort kernels themselves. The sort kernels always sort unsigned bit patterns.
- **32-bit vs 64-bit** affects element size, register count, and tile geometry. This is the only structural difference in the sort algorithm.
- **Keys-only vs key-value** affects whether values are loaded, carried, and scattered alongside keys.

This gives 4 combinations (`{has_values, is_64bit}` x `{false, true}`) applied to 4 kernels, but in practice:
- `sort_msd_histogram` and `sort_msd_prep` only need `IS_64BIT` (histogram ignores values)
- `sort_msd_atomic_scatter` and `sort_inner_fused` need both

Pre-compiled PSO count: ~10-12 for all current use cases (see Section 5.4).

### 1.3 Transform Kernels (New)

Two new trivial kernels handle bit transformations. These are bandwidth-bound single-dispatch operations that cost ~0.3ms at 16M elements (negligible vs the 3.1ms sort).

```metal
// ═══════════════════════════════════════════════════════════════
// Bit transformation kernels for signed/float types
// ═══════════════════════════════════════════════════════════════

// Function constant: selects transformation mode
//   0 = i32 (XOR sign bit -- self-inverse)
//   1 = float forward (FloatFlip)
//   2 = float inverse (IFloatFlip)
constant uint TRANSFORM_MODE [[function_constant(2)]];
constant uint transform_mode = is_function_constant_defined(TRANSFORM_MODE)
                                ? TRANSFORM_MODE : 0u;

kernel void sort_transform_32(
    device uint* data [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint f = data[gid];

    if (transform_mode == 0u) {
        // i32: XOR sign bit to convert two's complement to unsigned order
        data[gid] = f ^ 0x80000000u;
    } else if (transform_mode == 1u) {
        // f32 forward (FloatFlip): positive flips sign bit, negative flips all
        uint mask = as_type<uint>(-as_type<int>(f >> 31u)) | 0x80000000u;
        data[gid] = f ^ mask;
    } else {
        // f32 inverse (IFloatFlip): reverse the FloatFlip transformation
        uint mask = ((f >> 31u) - 1u) | 0x80000000u;
        data[gid] = f ^ mask;
    }
}

kernel void sort_transform_64(
    device ulong* data [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ulong f = data[gid];

    if (transform_mode == 0u) {
        // u64: no transform needed (this mode is never dispatched for u64,
        // but included for completeness). Identity operation.
        // In practice, sort_u64 skips the transform dispatch entirely.
    } else if (transform_mode == 1u) {
        // i64 OR f64 forward:
        // i64: XOR sign bit to convert two's complement to unsigned order
        //      (self-inverse, same concept as i32 but 64-bit)
        // f64: FloatFlip — same algorithm, 64-bit width
        // Note: i64 uses a simplified path (just XOR 0x8000000000000000),
        // but shares mode 1 since the host-side dispatches the correct variant.
        // For i64: data[gid] = f ^ 0x8000000000000000uL; (self-inverse)
        // For f64: full FloatFlip as below
        ulong mask = as_type<ulong>(-as_type<long>(f >> 63uL)) | 0x8000000000000000uL;
        data[gid] = f ^ mask;
    } else {
        // f64 inverse (IFloatFlip)
        ulong mask = ((f >> 63uL) - 1uL) | 0x8000000000000000uL;
        data[gid] = f ^ mask;
    }
}

// Note on i64 vs f64 transform dispatch:
// i64 uses TRANSFORM_MODE=1 with the sort_transform_64 kernel, BUT the i64
// XOR sign bit (0x8000000000000000) is self-inverse, so the same mode 1
// dispatch works for both forward and inverse. The FloatFlip formula
// (mask = -(long)(f>>63) | 0x8000...) when applied to a two's complement
// integer where only the sign bit matters, produces exactly the XOR sign bit
// result for both positive (mask=0x8000...) and negative (mask=0xFFFF...)
// integers. This is mathematically equivalent to the simple XOR for signed
// integers. See Section 2.1 for the proof.
```

**Why separate transform kernels instead of fusing into sort kernels?**

1. **Simplicity**: The sort kernels remain type-agnostic. They sort unsigned bit patterns. The transform layer is a clean separation of concerns.
2. **Reuse**: The same sort kernels work for u32, i32, f32 (all 32-bit) and u64, f64 (all 64-bit). No type-specific branching inside the hot loop.
3. **Performance cost is negligible**: At 200 GB/s bandwidth, 16M * 4 bytes = 64 MB takes ~0.3ms per transform pass. Two transform dispatches (pre + post) add ~0.6ms to a 3.1ms sort = ~19% overhead. For the zero-copy path, these dispatches are part of the same command buffer -- no CPU round-trip.
4. **Fusing is a future optimization**: If 19% overhead proves unacceptable, transforms can be fused into the MSD histogram (pre-transform during first load) and the final inner scatter (inverse-transform during last write). This is a Phase 4 optimization, not a correctness change.

**Why not CPU-side transforms for the memcpy path?**

For `sort_i32(&mut [i32])`, the data is already being memcpy'd to GPU. We could transform on the CPU before the memcpy. However:
- CPU transform at 16M elements takes ~5ms (single-threaded) or ~1ms (rayon parallel). The GPU kernel takes ~0.3ms.
- The GPU transform runs in the same command buffer as the sort, so there is zero dispatch overhead beyond the kernel itself.
- Keeping all transforms on GPU means one code path for both memcpy and zero-copy APIs.

**Decision**: All transforms happen on GPU, in the same command buffer as the sort dispatches. This is the simplest and most consistent approach.

### 1.4 Key-Value Support in Sort Kernels

When `has_values` is true, the scatter and inner kernels carry a parallel `values` buffer. The changes are surgical:

**sort_msd_histogram**: No change needed. Histogram only reads keys, never values.

**sort_msd_prep**: No change needed. Prefix sum only processes histogram counts.

**sort_msd_atomic_scatter**: Load values alongside keys, scatter both to output positions.

```metal
kernel void sort_msd_atomic_scatter(
    device const uint*     src       [[buffer(0)]],
    device uint*           dst       [[buffer(1)]],
    device atomic_uint*    counters  [[buffer(2)]],
    constant SortParams&   params    [[buffer(3)]],
    // Conditionally used -- only when has_values is true
    device const uint*     src_vals  [[buffer(4)]],
    device uint*           dst_vals  [[buffer(5)]],
    /* ... thread indices ... */)
{
    // ... existing Phase 1: Load keys ...
    uint mk[SORT_ELEMS];
    uint mv_vals[SORT_ELEMS];  // Only used when has_values -- eliminated by compiler otherwise
    for (uint e = 0u; e < SORT_ELEMS; e++) {
        uint idx = base + e * SORT_THREADS + lid;
        mv[e] = idx < n;
        mk[e] = mv[e] ? src[idx] : 0xFFFFFFFFu;
        if (has_values) {
            mv_vals[e] = mv[e] ? src_vals[idx] : 0u;
        }
        md[e] = mv[e] ? ((mk[e] >> shift) & 0xFFu) : 0xFFu;
    }

    // ... existing Phases 2-3 (histogram, prefix, atomic fetch-add): unchanged ...

    // Phase 4 scatter: write both key and value
    for (uint e = 0u; e < SORT_ELEMS; e++) {
        if (mv[e]) {
            uint d = md[e];
            uint within_sg = atomic_fetch_add_explicit(
                &sg_hist_or_rank[simd_id * SORT_NUM_BINS + d],
                1u, memory_order_relaxed);
            uint gp = tile_base[d]
                     + sg_prefix[simd_id * SORT_NUM_BINS + d]
                     + within_sg;
            dst[gp] = mk[e];
            if (has_values) {
                dst_vals[gp] = mv_vals[e];
            }
        }
    }
}
```

**Register pressure analysis**: Adding `mv_vals[16]` consumes 16 extra 32-bit registers per thread. The current kernel uses ~48 registers (16 keys + 16 digits + 16 validity + misc). Adding 16 values brings it to ~64. Apple M4 Pro has 96 32-bit registers per thread, so 64 registers is within budget. No register spilling expected.

**sort_inner_fused**: Same pattern. Values are loaded, carried through all 3 inner passes, and scattered alongside keys in each pass. Values are stored in per-thread registers (`mv_vals[SORT_ELEMS]`), NOT in threadgroup memory. The inner fused kernel already uses ~22KB of 32KB available TG memory; adding values to TG memory would overflow. Keeping values in registers avoids this entirely.

The value buffer ping-pong within the inner fused kernel follows the same src/dst alternation as keys:
- Pass 0: `src_vals -> dst_vals` (buf_vals_b -> buf_vals_a)
- Pass 1: `dst_vals -> src_vals` (buf_vals_a -> buf_vals_b)
- Pass 2: `src_vals -> dst_vals` (buf_vals_b -> buf_vals_a)

### 1.5 Index Initialization Kernel (New, for Argsort)

Argsort needs an initial index buffer `[0, 1, 2, ..., n-1]`:

```metal
kernel void sort_init_indices(
    device uint* indices [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    indices[gid] = gid;
}
```

Cost: ~0.15ms at 16M elements (write-only, 64MB at 400+ GB/s).

### 1.6 Gather Kernel (New, for sort_pairs)

After argsort produces sorted indices, `sort_pairs` rearranges the original values by those indices:

```metal
kernel void sort_gather_values(
    device const uint* sorted_indices    [[buffer(0)]],
    device const uint* original_values   [[buffer(1)]],
    device uint*       gathered_values   [[buffer(2)]],
    constant uint&     count             [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint idx = sorted_indices[gid];
    gathered_values[gid] = original_values[idx];
}
```

Cost: ~0.6ms at 16M elements (1 read of indices + 1 random-access read of values + 1 sequential write = ~192MB effective bandwidth at ~200 GB/s for the random reads).

Note: The gather kernel uses random-access reads on `original_values[idx]`, which have poor cache locality. This is the inherent cost of the Strategy B (argsort + gather) approach. Strategy A (native key-value sort kernels carrying values through all passes) avoids this random-access gather, but requires more complex shader modifications. We implement Strategy B first, with Strategy A available as a Phase 4 optimization.

### 1.7 Complete New Kernel Summary

| Kernel | Lines | Purpose | Dispatch Cost @ 16M |
|--------|-------|---------|-------------------|
| `sort_transform_32` | ~20 | i32/f32 bit flip | ~0.3ms |
| `sort_transform_64` | ~15 | f64 bit flip | ~0.6ms |
| `sort_init_indices` | ~6 | Write [0,1,...,n-1] | ~0.15ms |
| `sort_gather_values` | ~8 | Rearrange values by indices | ~0.6ms |

---

## 2. Bit Transformation Implementation

### 2.1 IEEE 754 Float Transformations

The canonical [FloatFlip algorithm](https://stereopsis.com/radix.html) converts IEEE 754 floats to a sortable unsigned integer representation. This is the same transformation [documented by NVIDIA CUB](https://github.com/NVIDIA/cub/issues/293).

#### f32 (32-bit)

```
IEEE 754 f32 layout: [sign(1)][exponent(8)][mantissa(23)]

Pre-sort (FloatFlip):
  If sign bit = 0 (positive): XOR with 0x80000000 (flip sign bit only)
  If sign bit = 1 (negative): XOR with 0xFFFFFFFF (flip all bits)
  Combined: mask = -(int)(f >> 31) | 0x80000000; result = f ^ mask;

Post-sort (IFloatFlip):
  If current bit 31 = 0 (was negative): XOR with 0xFFFFFFFF
  If current bit 31 = 1 (was positive): XOR with 0x80000000
  Combined: mask = ((f >> 31) - 1) | 0x80000000; result = f ^ mask;
```

**Why this works**: IEEE 754 positive floats already sort correctly as unsigned integers (larger exponent = larger value, larger mantissa = larger value when exponent is equal). The sign bit is inverted so positives sort after negatives. Negative floats have inverted magnitude ordering (more negative = larger unsigned representation), so flipping ALL bits corrects this inversion.

**Resulting total order**: `-NaN < -Inf < -max < ... < -min < -0.0 < +0.0 < +min < ... < +max < +Inf < +NaN`

This matches [Rust's `f32::total_cmp()`](https://doc.rust-lang.org/std/primitive.f32.html#method.total_cmp) semantics exactly, and aligns with the [IEEE 754 totalOrder predicate](https://en.wikipedia.org/wiki/IEEE_754).

#### f64 (64-bit)

Identical algorithm, 64-bit width:
```
mask = -(long)(f >> 63) | 0x8000000000000000L
result = f ^ mask
```

#### i32 (32-bit signed integer)

Simpler than floats -- two's complement to unsigned:
```
result = i32_bits XOR 0x80000000
```

This flips the sign bit, converting the two's complement range `[-2^31, 2^31-1]` to the unsigned range `[0, 2^32-1]` while preserving order. The transformation is its own inverse (XOR twice = identity).

#### u64 (64-bit unsigned integer)

No transformation needed. u64 values sort correctly as raw unsigned bit patterns. The sort pipeline uses IS_64BIT=true with no transform dispatch. This is the simplest 64-bit type -- the baseline for validating the 64-bit pipeline.

#### i64 (64-bit signed integer)

Identical to i32, but 64-bit:
```
result = i64_bits XOR 0x8000000000000000
```

This flips the sign bit, converting the two's complement range `[-2^63, 2^63-1]` to the unsigned range `[0, 2^64-1]` while preserving order. The transformation is its own inverse (XOR twice = identity).

**Why i64 can share f64's transform_mode=1**: The FloatFlip formula `mask = -(long)(f>>63) | 0x8000000000000000` applied to a two's complement integer produces:
- For non-negative i64 (sign bit 0): `mask = -(0) | 0x8000... = 0x8000...`, so `result = f XOR 0x8000...` (flip sign bit)
- For negative i64 (sign bit 1): `mask = -(1) | 0x8000... = 0xFFFF... | 0x8000... = 0xFFFF...`, so `result = f XOR 0xFFFF...` (flip all bits)

For two's complement integers, flipping all bits of a negative number and flipping only the sign bit of a positive number produces the same unsigned ordering as the simple sign-bit XOR. This is because two's complement negatives are already in reverse magnitude order (more negative = larger bit pattern), and flipping all bits reverses this into correct ascending order. The self-inverse property is preserved: applying the same transform twice returns the original value.

### 2.2 NaN Handling

After FloatFlip:
- Positive NaN (sign=0, exp=all-ones, mantissa!=0) maps to values above +Inf
- Negative NaN (sign=1, exp=all-ones, mantissa!=0) maps to values below -Inf
- Different NaN payloads (quiet vs signaling, different mantissa bits) sort to different positions
- This is standard CUB behavior and matches `total_cmp`

**Documentation commitment**: "NaN values sort to the extremes: negative NaN before -Inf, positive NaN after +Inf. Different NaN bit patterns sort to distinct positions. This matches `f32::total_cmp()` / `f64::total_cmp()`."

### 2.3 Where Transforms Execute

| API Method | Transform Location | Pipeline |
|-----------|-------------------|----------|
| `sort_u32(&mut [u32])` | None | Copy -> sort -> copy back |
| `sort_i32(&mut [i32])` | GPU | Copy -> transform -> sort -> inverse transform -> copy back |
| `sort_f32(&mut [f32])` | GPU | Copy -> FloatFlip -> sort -> IFloatFlip -> copy back |
| `sort_u64(&mut [u64])` | None | Copy -> sort(64-bit) -> copy back |
| `sort_i64(&mut [i64])` | GPU | Copy -> transform64 -> sort(64-bit) -> inverse transform64 -> copy back |
| `sort_f64(&mut [f64])` | GPU | Copy -> FloatFlip64 -> sort(64-bit) -> IFloatFlip64 -> copy back |
| `sort_i32_buffer(buf)` | GPU | Transform in-place -> sort -> inverse transform in-place |
| `sort_u64_buffer(buf)` | None | sort(64-bit) in-place (no transform needed) |
| `sort_i64_buffer(buf)` | GPU | Transform64 in-place -> sort(64-bit) -> inverse transform64 in-place |
| `argsort_f32(&[f32])` | GPU | Copy -> FloatFlip -> sort+indices -> IFloatFlip -> read indices |
| `argsort_u64(&[u64])` | None | Copy -> sort(64-bit)+indices -> read indices |
| `argsort_i64(&[i64])` | GPU | Copy -> transform64 -> sort(64-bit)+indices -> inverse transform64 -> read indices |

**Critical subtlety for argsort with transforms**: When doing argsort on f32, the pipeline is:
1. Copy keys to GPU buffer, apply FloatFlip
2. Initialize index buffer [0, 1, 2, ..., n-1]
3. Sort keys+indices together (key-value sort with `has_values=true`)
4. Apply IFloatFlip to restore original float values in sorted order
5. Return sorted indices

The indices are valid because the FloatFlip transform is position-preserving -- element at position `i` stays at position `i` after transform, just with different bit patterns. The sort moves elements (and their paired indices) to sorted positions. The indices thus correctly map back to original positions in the user's input array.

---

## 3. Argsort Pipeline Design

### 3.1 Dispatch Sequence

Argsort for 32-bit types (u32/i32/f32). Single command buffer, single encoder:

```
Dispatch 1 [optional]: sort_transform_32      -- i32/f32 only (pre-sort bit flip)
Dispatch 2:            sort_init_indices       -- write [0, 1, ..., n-1] to values buffer
Dispatch 3:            sort_msd_histogram      -- histogram of (transformed) keys
Dispatch 4:            sort_msd_prep           -- prefix sum + bucket descriptors
Dispatch 5:            sort_msd_atomic_scatter -- scatter keys + indices (has_values=true PSO)
Dispatch 6:            sort_inner_fused        -- 3-pass inner sort carrying indices
Dispatch 7 [optional]: sort_transform_32      -- i32/f32 only (post-sort inverse transform)
```

Total dispatches: 5 (u32 argsort) or 7 (i32/f32 argsort). Same command buffer, implicit barriers.

### 3.2 Buffer Requirements

For argsort, the GpuSorter needs:

| Buffer | Size | Purpose | When Allocated |
|--------|------|---------|---------------|
| `buf_a` | n * 4 | Keys (primary) | Always (existing) |
| `buf_b` | n * 4 | Keys (scratch) | Always (existing) |
| `buf_vals_a` | n * 4 | Values/indices (primary) | **First argsort/KV call** |
| `buf_vals_b` | n * 4 | Values/indices (scratch) | **First argsort/KV call** |
| `buf_msd_hist` | 256 * 4 | MSD histogram | Always (existing) |
| `buf_counters` | 256 * 4 | Atomic counters | Always (existing) |
| `buf_bucket_descs` | 256 * 16 | Bucket descriptors | Always (existing) |

New memory: 2 * n * 4 bytes for the value/index ping-pong buffers. At 16M elements: 128 MB additional.

**Buffer ping-pong for key-value sort (tracing through all passes)**:

| Step | Keys | Values/Indices |
|------|------|---------------|
| After MSD scatter (buf_a -> buf_b) | buf_b | buf_vals_b |
| After inner pass 0 (b -> a) | buf_a | buf_vals_a |
| After inner pass 1 (a -> b) | buf_b | buf_vals_b |
| After inner pass 2 (b -> a) | **buf_a** | **buf_vals_a** |

Final output: keys in `buf_a`, sorted indices in `buf_vals_a`. This matches the existing u32-only pipeline where keys end up in `buf_a`.

### 3.3 Argsort for SortBuffer (Zero-Copy)

For `argsort_*_buffer(&SortBuffer<T>) -> SortBuffer<u32>`, the user's input must NOT be modified. But the sort pipeline modifies `buf_a` in-place.

**Solution**: Copy the user's SortBuffer contents to internal `buf_a` first, run the full sort pipeline, and return only the indices from `buf_vals_a`.

```
1. Allocate buf_a, buf_b (scratch for keys)
2. Copy user's SortBuffer -> buf_a (GPU blit or copy kernel, ~0.15ms @ 16M)
3. [Optional transform on buf_a]
4. Init indices in buf_vals_a
5. Sort buf_a + buf_vals_a (4 sort dispatches)
6. [Optional inverse transform on buf_a -- not strictly needed since we discard keys]
7. Wrap buf_vals_a as SortBuffer<u32> and return it
```

The inverse transform in step 6 is actually unnecessary for argsort_buffer -- we only return indices, not keys. We can skip it, saving one dispatch.

**Optimization**: For `argsort_u32_buffer` (no transform needed), we can use the user's SortBuffer directly as the MSD histogram source (read-only), then copy to `buf_a` only for the scatter (which writes). Actually, all 4 sort dispatches read from and write to buf_a/buf_b, so we must copy upfront. The copy is ~0.15ms at 16M -- negligible.

---

## 4. Key-Value Pair Pipeline (sort_pairs)

### 4.1 Strategy B: Argsort + Gather

As recommended in the PM analysis and confirmed in UX Q&A, we implement Strategy B (argsort + gather) for the initial release:

```
sort_pairs_f32(&mut keys, &mut values):
  Buffers needed:
    buf_a:         keys (sorted in-place)
    buf_b:         keys scratch
    buf_vals_a:    indices (sorted alongside keys)
    buf_vals_b:    indices scratch
    buf_orig_vals: copy of original values (for gather)

  Command buffer (single encoder):
    1. [CPU: Copy keys to buf_a, values to buf_orig_vals]
    2. sort_transform_32 on buf_a (FloatFlip)
    3. sort_init_indices on buf_vals_a
    4. sort_msd_histogram (buf_a)
    5. sort_msd_prep
    6. sort_msd_atomic_scatter (buf_a -> buf_b, buf_vals_a -> buf_vals_b)
    7. sort_inner_fused (buf_a <-> buf_b, buf_vals_a <-> buf_vals_b)
    8. sort_transform_32 inverse on buf_a (IFloatFlip)
    9. sort_gather_values (buf_vals_a indices, buf_orig_vals -> buf_vals_b gathered)

  After GPU completes:
    Copy buf_a -> keys slice (sorted keys)
    Copy buf_vals_b -> values slice (gathered values)
```

### 4.2 Memory Requirements for sort_pairs

| Buffer | Size | Purpose |
|--------|------|---------|
| `buf_a` | n * 4 | Keys (primary) |
| `buf_b` | n * 4 | Keys (scratch) |
| `buf_vals_a` | n * 4 | Indices (primary) |
| `buf_vals_b` | n * 4 | Indices (scratch) / gathered values output |
| `buf_orig_vals` | n * 4 | Original values (for gather) |
| Fixed metadata | ~6 KB | Histogram, counters, bucket descs |

Total: **5 * n * 4 bytes + 6 KB**. At 16M elements: **320 MB**.

Compared to key-only sort (2 * n * 4 = 128 MB), key-value adds 192 MB (3 extra n-sized buffers).

**Future optimization (Strategy A)**: Carrying values natively through all sort passes eliminates `buf_orig_vals` and the gather dispatch. Memory drops to 4 * n * 4 = 256 MB. This is a Phase 4 optimization.

### 4.3 Zero-Copy sort_pairs with Two SortBuffers

As decided in UX Q&A #2, sort_pairs uses two separate SortBuffers:

```rust
pub fn sort_pairs_u32_buffer(
    &mut self,
    keys: &SortBuffer<u32>,
    values: &SortBuffer<u32>,
) -> Result<(), SortError>;
```

For zero-copy key-value sorting:
1. User's `keys.buffer` serves as `buf_a` (sorted in-place)
2. Copy `values.buffer` contents to `buf_orig_vals` (internal scratch, for gather source)
3. Initialize `buf_vals_a` with [0, 1, ..., n-1]
4. Sort keys + indices (4 sort dispatches, keys buffer modified in-place)
5. Gather: `values.buffer[i] = buf_orig_vals[sorted_indices[i]]`

After sorting, `keys` SortBuffer contains sorted keys and `values` SortBuffer contains reordered values. Zero-copy for keys; one internal copy needed for the values gather source.

---

## 5. SortBuffer\<T\> and SortKey Trait

### 5.1 SortKey Trait (Sealed)

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

/// Marker trait for types that can be GPU-sorted by forge-sort.
///
/// Sealed: cannot be implemented outside this crate.
/// Supported types: `u32`, `i32`, `f32`, `u64`, `i64`, `f64`.
pub trait SortKey: private::Sealed + Copy + 'static {
    /// Size of one key in bytes (4 for 32-bit types, 8 for f64).
    const KEY_SIZE: usize;

    /// Whether this type requires a pre-sort bit transformation.
    const NEEDS_TRANSFORM: bool;

    /// Whether this type is 64-bit.
    const IS_64BIT: bool;

    /// Transform mode for the GPU kernel (0=i32, 1=float forward, 2=float inverse).
    /// Only meaningful when NEEDS_TRANSFORM is true.
    const TRANSFORM_MODE_FORWARD: u32;
    const TRANSFORM_MODE_INVERSE: u32;
}

impl SortKey for u32 {
    const KEY_SIZE: usize = 4;
    const NEEDS_TRANSFORM: bool = false;
    const IS_64BIT: bool = false;
    const TRANSFORM_MODE_FORWARD: u32 = 0;
    const TRANSFORM_MODE_INVERSE: u32 = 0;
}

impl SortKey for i32 {
    const KEY_SIZE: usize = 4;
    const NEEDS_TRANSFORM: bool = true;
    const IS_64BIT: bool = false;
    const TRANSFORM_MODE_FORWARD: u32 = 0; // XOR 0x80000000 (self-inverse)
    const TRANSFORM_MODE_INVERSE: u32 = 0; // same operation reverses itself
}

impl SortKey for f32 {
    const KEY_SIZE: usize = 4;
    const NEEDS_TRANSFORM: bool = true;
    const IS_64BIT: bool = false;
    const TRANSFORM_MODE_FORWARD: u32 = 1; // FloatFlip
    const TRANSFORM_MODE_INVERSE: u32 = 2; // IFloatFlip
}

impl SortKey for u64 {
    const KEY_SIZE: usize = 8;
    const NEEDS_TRANSFORM: bool = false;
    const IS_64BIT: bool = true;
    const TRANSFORM_MODE_FORWARD: u32 = 0; // No transform needed
    const TRANSFORM_MODE_INVERSE: u32 = 0; // No transform needed
}

impl SortKey for i64 {
    const KEY_SIZE: usize = 8;
    const NEEDS_TRANSFORM: bool = true;
    const IS_64BIT: bool = true;
    const TRANSFORM_MODE_FORWARD: u32 = 1; // XOR sign bit 0x8000000000000000 (shares f64's mode 1)
    const TRANSFORM_MODE_INVERSE: u32 = 1; // Self-inverse: same XOR reverses the transformation
}

impl SortKey for f64 {
    const KEY_SIZE: usize = 8;
    const NEEDS_TRANSFORM: bool = true;
    const IS_64BIT: bool = true;
    const TRANSFORM_MODE_FORWARD: u32 = 1; // FloatFlip 64-bit
    const TRANSFORM_MODE_INVERSE: u32 = 2; // IFloatFlip 64-bit
}
```

### 5.2 SortBuffer\<T\> Implementation

```rust
use std::marker::PhantomData;

pub struct SortBuffer<T: SortKey> {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    capacity: usize,
    _marker: PhantomData<T>,
}

impl<T: SortKey> SortBuffer<T> {
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn capacity(&self) -> usize { self.capacity }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.contents().as_ptr() as *mut T,
                self.capacity,
            )
        }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().as_ptr() as *const T,
                self.len,
            )
        }
    }

    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.capacity, "len {} exceeds capacity {}", len, self.capacity);
        self.len = len;
    }

    pub fn copy_from_slice(&mut self, data: &[T]) {
        assert!(data.len() <= self.capacity,
            "data len {} exceeds capacity {}", data.len(), self.capacity);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.buffer.contents().as_ptr() as *mut u8,
                data.len() * T::KEY_SIZE,
            );
        }
        self.len = data.len();
    }

    pub fn copy_to_slice(&self, dest: &mut [T]) {
        let n = self.len.min(dest.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.contents().as_ptr() as *const u8,
                dest.as_mut_ptr() as *mut u8,
                n * T::KEY_SIZE,
            );
        }
    }

    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}
```

### 5.3 GpuSorter Internal State Changes

```rust
pub struct GpuSorter {
    // Existing fields (unchanged)
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pso_cache: PsoCache,
    buf_a: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_b: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_msd_hist: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_counters: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_bucket_descs: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    data_buf_capacity: usize,

    // New: value/index ping-pong buffers (lazy -- allocated on first KV/argsort call)
    buf_vals_a: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_vals_b: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    vals_buf_capacity: usize,

    // New: original values buffer for sort_pairs gather (lazy)
    buf_orig_vals: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    orig_vals_capacity: usize,
}
```

Value buffers are **lazy** -- only allocated when a key-value or argsort method is first called. Pure key-only sorting never allocates them. This means `sort_u32` performance and memory usage are identical to v1.

### 5.4 PsoCache Extension for Function Constants

The current `PsoCache` in forge-primitives does not support function constants (it uses `newFunctionWithName` which takes no constants). We need to extend it.

**Option A**: Extend `PsoCache` in forge-primitives with a `get_or_create_specialized` method.

**Option B**: Create a local PSO cache in forge-sort that handles function constants.

**Chosen: Option A** (extend forge-primitives). The extension is backward-compatible -- existing `get_or_create` continues to work unchanged. The new method is additive.

```rust
// Added to PsoCache in forge-primitives/src/pso_cache.rs

/// Function constant value for PSO specialization.
#[derive(Debug, Clone, PartialEq)]
pub enum FnConstant {
    Bool(bool),
    U32(u32),
}

impl PsoCache {
    /// Get or create a PSO specialized with function constant values.
    ///
    /// Cache key includes the function name AND constant values, so different
    /// specializations of the same function are cached as separate PSOs.
    pub fn get_or_create_specialized(
        &mut self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
        constants: &[(usize, FnConstant)],  // (index, value) pairs
    ) -> &ProtocolObject<dyn MTLComputePipelineState> {
        // Build a cache key that encodes the specialization
        let cache_key = if constants.is_empty() {
            function_name.to_string()
        } else {
            let mut key = function_name.to_string();
            for (idx, val) in constants {
                match val {
                    FnConstant::Bool(b) => key.push_str(&format!(":{}={}", idx, b)),
                    FnConstant::U32(v) => key.push_str(&format!(":{}={}", idx, v)),
                }
            }
            key
        };

        if !self.cache.contains_key(&cache_key) {
            let pso = Self::compile_pso_specialized(library, function_name, constants);
            self.cache.insert(cache_key.clone(), pso);
        }
        &self.cache[&cache_key]
    }

    fn compile_pso_specialized(
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
        constants: &[(usize, FnConstant)],
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        // Create MTLFunctionConstantValues and set each constant
        let fc_values = MTLFunctionConstantValues::new();
        for (idx, val) in constants {
            match val {
                FnConstant::Bool(b) => {
                    let mut v = *b;
                    unsafe {
                        fc_values.setConstantValue_type_atIndex(
                            NonNull::new(&mut v as *mut bool as *mut _).unwrap(),
                            MTLDataType::Bool,
                            *idx,
                        );
                    }
                }
                FnConstant::U32(u) => {
                    let mut v = *u;
                    unsafe {
                        fc_values.setConstantValue_type_atIndex(
                            NonNull::new(&mut v as *mut u32 as *mut _).unwrap(),
                            MTLDataType::UInt,
                            *idx,
                        );
                    }
                }
            }
        }

        // Create function with constants
        let fn_name = NSString::from_str(function_name);
        let function = library
            .newFunctionWithName_constantValues_error(&fn_name, &fc_values)
            .unwrap_or_else(|e| panic!(
                "Failed to create specialized function '{}': {:?}", function_name, e
            ));

        // Create PSO from specialized function
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setMaxTotalThreadsPerThreadgroup(256);
        unsafe { descriptor.setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true); }

        let device = library.device();
        device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor, MTLPipelineOption::None, None,
            )
            .unwrap_or_else(|e| panic!(
                "Failed to create PSO for '{}' with constants: {:?}", function_name, e
            ))
    }
}
```

**Pre-compilation strategy at `GpuSorter::new()`**:

```rust
// Existing PSOs (no function constants -- backward compatible)
"sort_msd_histogram"       // keys-only, 32-bit (default)
"sort_msd_prep"            // no specialization needed
"sort_msd_atomic_scatter"  // keys-only, 32-bit (default)
"sort_inner_fused"         // keys-only, 32-bit (default)

// New PSOs for argsort/key-value (has_values=true, 32-bit)
"sort_msd_atomic_scatter:0=true"  // HAS_VALUES=true
"sort_inner_fused:0=true"         // HAS_VALUES=true

// Transform PSOs
"sort_transform_32:2=0"   // mode 0 = i32
"sort_transform_32:2=1"   // mode 1 = f32 forward
"sort_transform_32:2=2"   // mode 2 = f32 inverse

// Utility PSOs
"sort_init_indices"        // no specialization
"sort_gather_values"       // no specialization
```

Total at init: ~11 PSOs. The 64-bit PSOs (4 additional: histogram, scatter, inner, transform_64) are compiled lazily on first u64/i64/f64 call to avoid penalizing startup time. Since u64, i64, and f64 all share the same IS_64BIT=true sort PSOs, no additional PSOs are needed per 64-bit type -- only the transform_64 PSO varies by TRANSFORM_MODE.

---

## 6. Dispatch Logic

### 6.1 Sort Pipeline Configuration

```rust
/// Configuration for a sort pipeline dispatch.
struct SortPipelineConfig {
    has_values: bool,
    is_64bit: bool,
    /// (forward_mode, inverse_mode). None for u32.
    transform: Option<(u32, u32)>,
    /// Shift for MSD byte extraction (24 for 32-bit, 56 for 64-bit).
    msd_shift: u32,
    /// Inner fused dispatch parameters: [(start_shift, pass_count), ...]
    /// 32-bit: [(0, 3)]. 64-bit: [(32, 3), (8, 3), (0, 1)]
    inner_dispatches: Vec<(u32, u32)>,
}
```

### 6.2 Per-Type Configuration

```rust
fn config_u32_keys_only() -> SortPipelineConfig {
    SortPipelineConfig {
        has_values: false, is_64bit: false,
        transform: None,
        msd_shift: 24,
        inner_dispatches: vec![(0, 3)],
    }
}

fn config_i32_keys_only() -> SortPipelineConfig {
    SortPipelineConfig {
        has_values: false, is_64bit: false,
        transform: Some((0, 0)),  // i32: mode 0 is self-inverse
        msd_shift: 24,
        inner_dispatches: vec![(0, 3)],
    }
}

fn config_f32_keys_only() -> SortPipelineConfig {
    SortPipelineConfig {
        has_values: false, is_64bit: false,
        transform: Some((1, 2)),  // FloatFlip forward=1, inverse=2
        msd_shift: 24,
        inner_dispatches: vec![(0, 3)],
    }
}

fn config_u64_keys_only() -> SortPipelineConfig {
    SortPipelineConfig {
        has_values: false, is_64bit: true,
        transform: None,           // u64: no transform needed (baseline 64-bit type)
        msd_shift: 56,
        inner_dispatches: vec![(32, 3), (8, 3), (0, 1)],
    }
}

fn config_i64_keys_only() -> SortPipelineConfig {
    SortPipelineConfig {
        has_values: false, is_64bit: true,
        transform: Some((1, 1)),   // i64: mode 1 is self-inverse (XOR sign bit)
        msd_shift: 56,
        inner_dispatches: vec![(32, 3), (8, 3), (0, 1)],
    }
}

fn config_f64_keys_only() -> SortPipelineConfig {
    SortPipelineConfig {
        has_values: false, is_64bit: true,
        transform: Some((1, 2)),   // f64: FloatFlip forward=1, IFloatFlip inverse=2
        msd_shift: 56,
        inner_dispatches: vec![(32, 3), (8, 3), (0, 1)],
    }
}

// Argsort variants: same configs but with has_values: true
fn config_argsort_f32() -> SortPipelineConfig {
    SortPipelineConfig {
        has_values: true, is_64bit: false,
        transform: Some((1, 2)),
        msd_shift: 24,
        inner_dispatches: vec![(0, 3)],
    }
}

fn config_argsort_u64() -> SortPipelineConfig {
    SortPipelineConfig {
        has_values: true, is_64bit: true,
        transform: None,           // u64: no transform
        msd_shift: 56,
        inner_dispatches: vec![(32, 3), (8, 3), (0, 1)],
    }
}

fn config_argsort_i64() -> SortPipelineConfig {
    SortPipelineConfig {
        has_values: true, is_64bit: true,
        transform: Some((1, 1)),   // i64: self-inverse sign bit XOR
        msd_shift: 56,
        inner_dispatches: vec![(32, 3), (8, 3), (0, 1)],
    }
}
```

### 6.3 Unified Pipeline Dispatcher

The current `dispatch_sort` function is refactored into a general-purpose pipeline:

```rust
fn dispatch_sort_pipeline(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    library: &ProtocolObject<dyn MTLLibrary>,
    pso_cache: &mut PsoCache,
    config: &SortPipelineConfig,
    buf_a: &ProtocolObject<dyn MTLBuffer>,    // keys primary
    buf_b: &ProtocolObject<dyn MTLBuffer>,    // keys scratch
    buf_vals_a: Option<&ProtocolObject<dyn MTLBuffer>>,  // values primary
    buf_vals_b: Option<&ProtocolObject<dyn MTLBuffer>>,  // values scratch
    buf_msd_hist: &ProtocolObject<dyn MTLBuffer>,
    buf_counters: &ProtocolObject<dyn MTLBuffer>,
    buf_bucket_descs: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) -> Result<(), SortError> {
    let num_tiles = n.div_ceil(if config.is_64bit { 2048 } else { 4096 });

    let cmd = queue.commandBuffer().ok_or_else(|| ...)?;
    let enc = cmd.computeCommandEncoder().ok_or_else(|| ...)?;

    // Step 1: Optional pre-sort transform
    if let Some((fwd_mode, _)) = config.transform {
        encode_transform(&enc, pso_cache, library, buf_a, n, fwd_mode, config.is_64bit);
    }

    // Step 2: Optional index initialization
    if config.has_values {
        encode_init_indices(&enc, pso_cache, library, buf_vals_a.unwrap(), n);
    }

    // Step 3: MSD histogram
    encode_msd_histogram(&enc, pso_cache, library, buf_a, buf_msd_hist,
                         n, num_tiles, config.msd_shift, config.is_64bit);

    // Step 4: MSD prep
    encode_msd_prep(&enc, pso_cache, library, buf_msd_hist,
                    buf_counters, buf_bucket_descs);

    // Step 5: MSD atomic scatter
    encode_msd_scatter(&enc, pso_cache, library,
                       buf_a, buf_b, buf_counters,
                       buf_vals_a, buf_vals_b,
                       n, num_tiles, config.msd_shift,
                       config.has_values, config.is_64bit);

    // Step 6: Inner fused dispatches (1 for 32-bit, 3 for 64-bit)
    let mut keys_in_a = false;  // after MSD scatter, keys are in buf_b
    for (start_shift, pass_count) in &config.inner_dispatches {
        let (src_keys, dst_keys) = if keys_in_a { (buf_a, buf_b) } else { (buf_b, buf_a) };
        let (src_vals, dst_vals) = if keys_in_a {
            (buf_vals_a, buf_vals_b)
        } else {
            (buf_vals_b, buf_vals_a)
        };

        encode_inner_fused(&enc, pso_cache, library,
                           dst_keys, src_keys,  // note: inner kernel takes (buf_a, buf_b)
                           dst_vals, src_vals,   // where pass 0 reads from buf_b
                           buf_bucket_descs,
                           *start_shift, *pass_count,
                           config.has_values, config.is_64bit);

        // After pass_count passes, data lands in buf_a or buf_b
        // Pass count determines final buffer: odd count -> swapped, even -> same
        if *pass_count % 2 == 1 {
            keys_in_a = !keys_in_a;
        }
        // 3 passes: starts in src (buf_b after MSD), passes through a->b->a, ends in a
        // Wait: inner fused alternates b->a, a->b, b->a for 3 passes.
        // After 3 passes starting from buf_b: ends in buf_a. keys_in_a = true.
        // After 1 pass starting from buf_a: reads buf_b (NO -- depends on dispatch args)
    }

    // Step 7: Optional post-sort inverse transform
    // Keys should be in buf_a after all dispatches
    if let Some((_, inv_mode)) = config.transform {
        let final_buf = if keys_in_a { buf_a } else { buf_b };
        encode_transform(&enc, pso_cache, library, final_buf, n, inv_mode, config.is_64bit);
    }

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    // Check for GPU errors
    if cmd.status() == MTLCommandBufferStatus::Error {
        return Err(SortError::GpuExecution(format!("{:?}", cmd.error())));
    }
    Ok(())
}
```

**Note on inner fused buffer arguments**: The current `sort_inner_fused` kernel takes `(buf_a, buf_b)` and internally alternates: pass 0 reads from `buf_b` (the MSD scatter output), pass 1 reads from `buf_a`, pass 2 reads from `buf_b`. After 3 passes, the final write is to `buf_a`.

For f64 with multiple inner dispatches, we must swap the buffer arguments between dispatches to account for where data ends up after each set of passes. The dispatch logic handles this by tracking which buffer holds the current data.

### 6.4 Method Implementations (Examples)

```rust
impl GpuSorter {
    pub fn sort_i32(&mut self, data: &mut [i32]) -> Result<(), SortError> {
        let n = data.len();
        if n <= 1 { return Ok(()); }
        self.ensure_buffers(n);

        // Reinterpret i32 as u32 for memcpy (same bit representation)
        let data_u32 = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u32, n)
        };

        // Copy to buf_a, zero histogram
        unsafe { /* memcpy to buf_a, zero buf_msd_hist */ }

        dispatch_sort_pipeline(
            &self.queue, &self.library, &mut self.pso_cache,
            &config_i32_keys_only(),
            self.buf_a.as_ref().unwrap(),
            self.buf_b.as_ref().unwrap(),
            None, None,  // no value buffers
            /* ... scratch buffers ... */
            n,
        )?;

        // Copy result back from buf_a
        unsafe { /* memcpy from buf_a to data_u32 */ }
        Ok(())
    }

    pub fn argsort_f32(&mut self, data: &[f32]) -> Result<Vec<u32>, SortError> {
        let n = data.len();
        if n <= 1 { return Ok((0..n as u32).collect()); }
        self.ensure_buffers_with_values(n);

        // Copy keys to buf_a
        let data_u32 = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u32, n)
        };
        unsafe { /* memcpy to buf_a, zero buf_msd_hist */ }

        dispatch_sort_pipeline(
            &self.queue, &self.library, &mut self.pso_cache,
            &config_argsort_f32(),
            self.buf_a.as_ref().unwrap(),
            self.buf_b.as_ref().unwrap(),
            Some(self.buf_vals_a.as_ref().unwrap()),
            Some(self.buf_vals_b.as_ref().unwrap()),
            /* ... scratch buffers ... */
            n,
        )?;

        // Read sorted indices from buf_vals_a
        let mut indices = vec![0u32; n];
        unsafe { /* memcpy from buf_vals_a to indices */ }
        Ok(indices)
    }

    pub fn sort_pairs_f32(
        &mut self, keys: &mut [f32], values: &mut [u32],
    ) -> Result<(), SortError> {
        let n = keys.len();
        if n != values.len() {
            return Err(SortError::LengthMismatch { keys: n, values: values.len() });
        }
        if n <= 1 { return Ok(()); }
        self.ensure_buffers_with_values_and_orig(n);

        // Copy keys and values to GPU
        unsafe {
            /* memcpy keys to buf_a */
            /* memcpy values to buf_orig_vals */
            /* zero buf_msd_hist */
        }

        // Sort keys + indices
        dispatch_sort_pipeline(
            /* ... config_argsort_f32() ... */
        )?;

        // Gather values using sorted indices
        dispatch_gather_values(
            &self.queue, &self.library, &mut self.pso_cache,
            self.buf_vals_a.as_ref().unwrap(),   // sorted indices
            self.buf_orig_vals.as_ref().unwrap(), // original values
            self.buf_vals_b.as_ref().unwrap(),    // gathered values output
            n,
        )?;

        // Copy results back
        let keys_u32 = unsafe {
            std::slice::from_raw_parts_mut(keys.as_mut_ptr() as *mut u32, n)
        };
        unsafe {
            /* memcpy buf_a -> keys_u32 (sorted keys) */
            /* memcpy buf_vals_b -> values (gathered values) */
        }
        Ok(())
    }
}
```

---

## 7. f64 Pipeline Design (Detailed)

### 7.1 Architecture Overview

f64 keys are 64 bits = 8 bytes. With an 8-bit radix, a full sort requires 8 passes. Our MSD+inner architecture distributes these across 6 dispatches:

| Dispatch | Kernel | Byte Position | Bits | Description |
|----------|--------|--------------|------|-------------|
| 1 | sort_msd_histogram (64-bit PSO) | byte 7 | 56:63 | Histogram of MSB |
| 2 | sort_msd_prep | -- | -- | Prefix sum + bucket descriptors |
| 3 | sort_msd_atomic_scatter (64-bit PSO) | byte 7 | 56:63 | Scatter by MSB |
| 4 | sort_inner_fused (64-bit, 3-pass) | bytes 4,5,6 | 32:55 | First inner group |
| 5 | sort_inner_fused (64-bit, 3-pass) | bytes 1,2,3 | 8:31 | Second inner group |
| 6 | sort_inner_fused (64-bit, 1-pass) | byte 0 | 0:7 | Final inner pass |

Total: 6 dispatches vs 4 for 32-bit. The inner fused kernel is parameterized to handle variable pass counts via buffer constants (not function constants -- see 7.3).

### 7.2 Kernel Modifications for 64-bit

When `is_64bit` is true via function constant, the sort kernels adapt:

**Element loading**: Pointer is reinterpreted as `ulong*`, registers hold `ulong` keys.

```metal
kernel void sort_msd_histogram(
    device const uint* src [[buffer(0)]],  // always uint* -- reinterpreted for 64-bit
    /* ... */)
{
    if (is_64bit) {
        device const ulong* src64 = (device const ulong*)src;
        // Load 8 elements per thread (half of 32-bit's 16)
        ulong keys64[8];
        bool valid[8];
        uint base = gid * 2048u;  // effective tile size = 8 * 256
        for (uint e = 0u; e < 8u; e++) {
            uint idx = base + e * SORT_THREADS + lid;
            valid[e] = idx < params.element_count;
            keys64[e] = valid[e] ? src64[idx] : 0uL;
        }
        // Histogram: extract byte at params.shift
        // sg_counts logic identical, just digit = uint((keys64[e] >> shift) & 0xFFuL)
    } else {
        // Existing 32-bit code path (unchanged, eliminated by compiler for 64-bit PSO)
    }
}
```

Since `is_64bit` is a function constant, the Metal compiler **eliminates the unused branch entirely**. The 64-bit PSO contains only the 64-bit code path. The 32-bit PSO contains only the 32-bit code path. Zero runtime branching, zero code size overhead.

### 7.3 Inner Fused Kernel Parameterization

The inner fused kernel needs `start_shift` and `pass_count` to vary per dispatch. These are passed as **buffer constants** (not function constants), since they change between dispatches within the same sort call.

```metal
struct InnerParams {
    uint start_shift;   // first byte position * 8 (0, 8, 16, 24, 32, ...)
    uint pass_count;    // number of passes (1 or 3)
    uint batch_start;   // existing field (TG offset for batched dispatch)
};

kernel void sort_inner_fused(
    device uint*                buf_a         [[buffer(0)]],
    device uint*                buf_b         [[buffer(1)]],
    device const BucketDesc*    bucket_descs  [[buffer(2)]],
    constant InnerParams&       inner_params  [[buffer(3)]],
    /* ... value buffers at buffer(4), buffer(5) when has_values ... */
    /* ... thread indices ... */)
{
    // Replace hardcoded "3u" with inner_params.pass_count
    for (uint pass = 0u; pass < inner_params.pass_count; pass++) {
        uint shift = inner_params.start_shift + pass * 8u;
        // ... rest of pass logic unchanged ...
    }
}
```

This replaces the current `constant uint& batch_start [[buffer(3)]]` with a struct that includes `batch_start` plus the new fields. The `InnerParams` struct is `#[repr(C)]` on the Rust side.

### 7.4 64-bit Tile Size and Register Budget

For 64-bit keys, each element is 8 bytes. Loading 16 elements per thread (like 32-bit) would require 128 bytes of registers for keys alone, exceeding the practical register budget.

**Solution**: Halve elements per thread for 64-bit.

```metal
constant uint ELEMS_PER_THREAD = is_64bit ? 8u : 16u;
constant uint EFFECTIVE_TILE_SIZE = ELEMS_PER_THREAD * SORT_THREADS;
// 64-bit: 8 * 256 = 2048 elements/tile
// 32-bit: 16 * 256 = 4096 elements/tile
```

Register usage for 64-bit:
- `ulong keys64[8]` = 16 x 32-bit registers (each ulong = 2 registers)
- `uint digits[8]` = 8 registers
- `bool valid[8]` = 8 registers (likely packed)
- Misc (counters, indices, temporaries) = ~8 registers
- Total: ~40 registers. Well within Apple M4's 96-register budget.

For 64-bit key-value: add `uint vals[8]` = 8 more registers. Total ~48. Still safe.

### 7.5 64-bit Buffer Ping-Pong

After MSD scatter, data is in `buf_b`. The 3 inner fused dispatches must track which buffer has current data:

| Inner Dispatch | Pass Count | Input Buffer | Output Buffer |
|---------------|-----------|-------------|--------------|
| Dispatch 4 (bytes 4,5,6) | 3 | buf_b | buf_a (3 passes: b->a, a->b, b->a) |
| Dispatch 5 (bytes 1,2,3) | 3 | buf_a | buf_b (need to swap args so pass 0 reads from buf_a) |
| Dispatch 6 (byte 0) | 1 | buf_b | buf_a (1 pass: b->a) |

Final output: **buf_a**. Same as 32-bit.

**Buffer argument swapping**: The inner fused kernel's pass 0 always reads from the second buffer argument (`buf_b` in the kernel's parameter list). To make it read from the buffer that actually holds data, we swap the Rust-side arguments:

- Dispatch 4: pass `(buf_a, buf_b)` -- kernel reads from buf_b (correct: MSD output is in buf_b)
- Dispatch 5: pass `(buf_b, buf_a)` -- kernel reads from buf_a (correct: dispatch 4 output is in buf_a)
- Dispatch 6: pass `(buf_a, buf_b)` -- kernel reads from buf_b (correct: dispatch 5 output is in buf_b)

### 7.6 f64 Performance Projection

| Metric | 32-bit (measured) | 64-bit (projected) | Notes |
|--------|------------------|-------------------|-------|
| Bytes per element | 4 | 8 | 2x data width |
| Total passes | 4 | 8 | 2x passes |
| Elements per tile | 4096 | 2048 | Halved for register budget |
| Bandwidth per sort | ~32n bytes | ~128n bytes | 4x total I/O |
| Sort time @ 16M | ~3.1ms | ~12-13ms | 4x bandwidth |
| Transform overhead | 0ms (u32) | ~0.6ms | 2 transform dispatches |
| Total time @ 16M | ~3.1ms | ~13ms | |
| Throughput (Mk/s ZC) | ~5200 | ~1200 | |
| Throughput (Mk/s E2E) | ~2800 | ~700-800 | E2E includes memcpy of 8-byte elements |

This meets the PM target of >= 800 Mk/s for f64.

**u64 and i64 performance projections**: Identical pipeline to f64 (same IS_64BIT path, same 8 passes, same tile size). The only difference is transform overhead:
- **u64**: No transform dispatches. Expected ~1250 Mk/s zero-copy, ~850 Mk/s end-to-end. Meets PM target >= 800 Mk/s.
- **i64**: 2 transform dispatches (sort_transform_64, self-inverse). Expected ~1200 Mk/s zero-copy, ~800 Mk/s end-to-end. Meets PM target >= 750 Mk/s.

---

## 8. Memory Layout for Key-Value Pairs

### 8.1 Structure of Arrays (SoA)

Keys and values are stored in **separate GPU buffers**, not interleaved (AoS):

```
buf_a:      [key_0][key_1][key_2]...[key_n-1]     (n * KEY_SIZE bytes)
buf_vals_a: [val_0][val_1][val_2]...[val_n-1]     (n * 4 bytes, always u32)
```

**Why SoA, not AoS**:
1. The histogram and prefix sum kernels only access keys -- SoA avoids loading unused value bytes, saving 50% bandwidth in those passes
2. Memory coalescing is optimal when all threads in a SIMD group access consecutive addresses of the same type
3. The scatter kernel writes keys and values as two separate coalesced streams
4. This matches CUB's internal layout and the [bandwidth analysis by Stehle & Satish](https://arxiv.org/pdf/1611.01137)

### 8.2 Buffer Allocation Growth Strategy

All scratch buffers follow the existing grow-only strategy:

```rust
fn ensure_buffers_with_values(&mut self, n: usize) {
    self.ensure_buffers(n);  // existing: buf_a, buf_b, metadata

    let data_bytes = n * 4;  // values are always u32
    if self.buf_vals_a.is_none() || data_bytes > self.vals_buf_capacity {
        self.buf_vals_a = Some(alloc_buffer(&self.device, data_bytes));
        self.buf_vals_b = Some(alloc_buffer(&self.device, data_bytes));
        self.vals_buf_capacity = data_bytes;
    }
}

fn ensure_buffers_with_values_and_orig(&mut self, n: usize) {
    self.ensure_buffers_with_values(n);

    let data_bytes = n * 4;
    if self.buf_orig_vals.is_none() || data_bytes > self.orig_vals_capacity {
        self.buf_orig_vals = Some(alloc_buffer(&self.device, data_bytes));
        self.orig_vals_capacity = data_bytes;
    }
}
```

---

## 9. Performance Implications

### 9.1 Overhead Summary by Feature

| Feature | Extra Dispatches | Sort Kernel Overhead | Memory Overhead | Projected vs u32 |
|---------|-----------------|---------------------|----------------|------------------|
| i32 keys | +2 (transform) | 0% (same PSO) | 0% | ~19% slower E2E |
| f32 keys | +2 (transform) | 0% (same PSO) | 0% | ~19% slower E2E |
| argsort u32 | +1 (init indices) | ~10-15% (KV PSO) | +2n*4 bytes | ~25-30% slower |
| argsort f32 | +3 (transform x2 + init) | ~10-15% (KV PSO) | +2n*4 bytes | ~35-40% slower |
| sort_pairs u32 | +2 (init + gather) | ~10-15% (KV PSO) | +3n*4 bytes | ~35-40% slower |
| sort_pairs f32 | +4 (transform x2 + init + gather) | ~10-15% (KV PSO) | +3n*4 bytes | ~45-50% slower |
| u64 keys | +2 (6 total vs 4, no transforms) | ~300% (8 passes) | 2x per buffer | ~4x slower |
| i64 keys | +4 (6 total vs 4, + 2 transforms) | ~300% (8 passes) | 2x per buffer | ~4x slower |
| f64 keys | +4 (6 total vs 4, + 2 transforms) | ~300% (8 passes) | 2x per buffer | ~4x slower |

### 9.2 Phase 4 Optimization Roadmap

1. **Fuse transforms into sort kernels** (~15% improvement for i32/f32):
   Apply FloatFlip during the MSD histogram's element load. Apply IFloatFlip during the last inner scatter's element write. Eliminates 2 dispatches. Requires `TRANSFORM_MODE` function constant added to sort kernels.

2. **Native key-value scatter (Strategy A)** (~15% improvement for KV, saves n*4 bytes):
   Carry values through all sort passes natively. Eliminates gather dispatch and `buf_orig_vals`. More complex shader changes but reduces total dispatches and memory.

3. **Partial-bit sorting** (variable improvement):
   Skip passes where all elements have the same byte value. For f32 data with limited range, the exponent byte may be uniform -- skip that pass entirely. CUB's `begin_bit`/`end_bit` approach.

4. **f64 MSD on 2 bytes** (reduces f64 from 8 to 7 passes):
   Use a 16-bit MSD pass (65536 bins) to sort 2 bytes at once. Requires 65536-bin histogram in global memory (not TG memory). May not be worth the complexity for the niche f64 market.

---

## 10. File Structure

### 10.1 Modified Files

| File | Changes |
|------|---------|
| `shaders/sort.metal` | Add function constant declarations, `if (has_values)` branches in scatter/inner kernels, `if (is_64bit)` branches for 64-bit element handling, 4 new trivial kernels, `InnerParams` struct |
| `src/lib.rs` | `SortBuffer<T>`, `SortKey` trait (sealed, 6 types: u32/i32/f32/u64/i64/f64), 24+ new public methods, `SortPipelineConfig`, `dispatch_sort_pipeline`, buffer management extensions, `SortError::LengthMismatch` |
| `Cargo.toml` | Version bump to 0.2.0 (breaking: `SortBuffer` -> `SortBuffer<u32>`) |
| `tests/correctness.rs` | Extend with i32/f32/f64/argsort/sort_pairs tests |

### 10.2 Modified Files in forge-primitives

| File | Changes |
|------|---------|
| `forge-primitives/src/pso_cache.rs` | Add `FnConstant` enum, `get_or_create_specialized` method, `compile_pso_specialized` helper |

### 10.3 New Files

None. All changes fit within existing files. The 4 new shader kernels (~50 lines total) are added to `sort.metal`. The Rust-side additions (~400 lines) go in `lib.rs`.

If `lib.rs` grows beyond ~800 lines during Phase 2 (argsort + KV), consider extracting:
- `src/pipeline.rs` -- `SortPipelineConfig`, `dispatch_sort_pipeline`
- `src/types.rs` -- `SortKey`, `SortBuffer<T>`, `SortError`

But defer splitting until it's clearly needed.

### 10.4 Shader Code Organization Within sort.metal

```
sort.metal (after v2):
  Lines 1-30:    #includes, #defines, function constant declarations
  Lines 31-50:   Struct definitions (SortParams, BucketDesc, InnerParams)
  Lines 51-100:  Kernel 1: sort_msd_histogram (+ is_64bit branch)
  Lines 101-140: Kernel 2: sort_msd_prep (unchanged)
  Lines 141-250: Kernel 3: sort_msd_atomic_scatter (+ has_values/is_64bit branches)
  Lines 251-460: Kernel 4: sort_inner_fused (+ has_values/is_64bit branches, InnerParams)
  Lines 461-500: Kernel 5: sort_transform_32 (NEW)
  Lines 501-530: Kernel 6: sort_transform_64 (NEW)
  Lines 531-545: Kernel 7: sort_init_indices (NEW)
  Lines 546-560: Kernel 8: sort_gather_values (NEW)
```

Estimated total: ~560 lines (up from 439).

### 10.5 build.rs Changes

**None required.** The build script already compiles `shaders/sort.metal` with `-std=metal3.2` and produces a single `.metallib`. All function constant specialization happens at PSO creation time in Rust, not at shader compile time. The same `.metallib` serves all PSO variants.

---

## 11. Implementation Plan by Phase

### Phase 1: i32 + f32 (3-4 days)

**Shader changes** (~30 lines added):
- Add function constant declarations (5 lines)
- Add `sort_transform_32` kernel (20 lines)
- Add `InnerParams` struct change (5 lines)

**Rust changes** (~250 lines added/changed):
- `SortKey` trait + sealed module (~50 lines)
- Convert `SortBuffer` to `SortBuffer<T>` (~40 lines changed)
- Add `sort_i32`, `sort_f32` methods (~80 lines)
- Add `sort_i32_buffer`, `sort_f32_buffer` methods (~60 lines)
- Add generic `alloc_sort_buffer::<T>` (~10 lines)
- Add `SortError::LengthMismatch` (~5 lines)
- Extend PsoCache for function constants (~60 lines in forge-primitives)
- Transform dispatch helper (~30 lines)

**Tests** (~40 new tests):
- i32: boundary values (MIN, MAX, 0, -1, +1), random 1M/16M, all-negative, mixed
- f32: NaN, Inf, -Inf, -0.0, +0.0, denormals, random 1M/16M, all-negative, mixed
- Verify against `[i32]::sort()` and `[f32]::sort_by(f32::total_cmp)`

### Phase 2: Argsort + Key-Value Pairs (5-8 days)

**Shader changes** (~60 lines added):
- Add `has_values` branches to `sort_msd_atomic_scatter` (~20 lines)
- Add `has_values` branches to `sort_inner_fused` (~30 lines)
- Add `sort_init_indices` kernel (6 lines)
- Add `sort_gather_values` kernel (8 lines)

**Rust changes** (~350 lines added):
- Add `buf_vals_a/b`, `buf_orig_vals` to GpuSorter (~15 lines)
- Add `ensure_buffers_with_values[_and_orig]` (~30 lines)
- Refactor `dispatch_sort` into `dispatch_sort_pipeline` (~100 lines)
- Add `argsort_u32/i32/f32` methods (~120 lines)
- Add `argsort_u32/i32/f32_buffer` methods (~90 lines)
- Add `sort_pairs_u32/i32/f32` methods (~150 lines)

**Tests** (~50 new tests):
- argsort: correctness (indices produce sorted order), stability (equal keys preserve order)
- sort_pairs: key-value integrity (bijection), stability, length mismatch error
- Cross-type: argsort_f32 with NaN, sort_pairs_i32 with negative keys

### Phase 3: 64-bit types — u64, i64, f64 (3-5 days)

**Shader changes** (~170 lines added):
- Add `is_64bit` branches to all 4 sort kernels (~40 lines each)
- Add `sort_transform_64` kernel (15 lines) — handles i64 (mode 1, self-inverse) and f64 (modes 1/2)
- u64 requires zero shader changes beyond the IS_64BIT function constant (no transform dispatch)

**Rust changes** (~200 lines added):
- Add `sort_u64`, `sort_u64_buffer` methods (simplest — no transform, just IS_64BIT pipeline)
- Add `sort_i64`, `sort_i64_buffer` methods (sign bit XOR via sort_transform_64)
- Add `sort_f64`, `sort_f64_buffer` methods (FloatFlip via sort_transform_64)
- Add `argsort_u64`, `argsort_i64`, `argsort_f64` methods and buffer variants
- Add `sort_pairs_u64`, `sort_pairs_i64` methods (64-bit keys + 32-bit values)
- Add SortKey impls for u64 (NEEDS_TRANSFORM=false) and i64 (TRANSFORM_MODE_FORWARD=1, self-inverse)
- Add 64-bit buffer management (ulong elements = 8 bytes per key)
- Add pipeline configs: config_u64 (no transform), config_i64 (self-inverse), config_f64 (FloatFlip)

**Tests** (~30+ new tests):
- u64: boundary values (0, u64::MAX, large values), random 1M/16M, already sorted
- i64: boundary values (i64::MIN, i64::MAX, -1, 0, 1), sign boundary, random 1M/16M
- f64: NaN, Inf, denormals, large magnitude, precision edge cases
- Verify u64 against `[u64]::sort()`, i64 against `[i64]::sort()`, f64 against `[f64]::sort_by(f64::total_cmp)`
- argsort and sort_pairs variants for u64/i64

---

## 12. References

- [Stereopsis: Radix Sort Tricks (FloatFlip/IFloatFlip)](https://stereopsis.com/radix.html) -- Canonical float-to-sortable-uint transformation
- [CUB Issue #293: Document Radix Sort Bitwise Transformations](https://github.com/NVIDIA/cub/issues/293) -- CUB's signed/float transformation documentation
- [Apple: Using function specialization to build pipeline variants](https://developer.apple.com/documentation/metal/using-function-specialization-to-build-pipeline-variants) -- Metal function constants documentation
- [Metal Shading Language Specification v4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) -- MSL function constant syntax and semantics
- [Onesweep: A Faster LSD Radix Sort for GPUs (Adinets & Merrill)](https://arxiv.org/pdf/2206.01784) -- State-of-the-art GPU radix sort architecture
- [Stehle & Satish: Memory Bandwidth-Efficient Hybrid Radix Sort](https://arxiv.org/pdf/1611.01137) -- Key-value bandwidth analysis
- [AMD GPUOpen: Introduction to GPU Radix Sort](https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf) -- 64-bit radix sort pass analysis
- [XiSort: Deterministic Sorting via IEEE-754 Total Ordering](https://arxiv.org/html/2505.11927v1) -- IEEE 754 totalOrder for sorting
- [NVIDIA CUB DeviceRadixSort API](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html) -- Production GPU sort API reference
- [Rust f32::total_cmp](https://doc.rust-lang.org/std/primitive.f32.html#method.total_cmp) -- IEEE 754 total ordering in Rust
- [WWDC 2016 Session 605: What's New in Metal](https://asciiwwdc.com/2016/sessions/605) -- Function specialization introduction
- [GitHub: FloatRadixSort](https://github.com/lshamis/FloatRadixSort) -- Reference float radix sort implementation
- [IEEE 754 - Wikipedia](https://en.wikipedia.org/wiki/IEEE_754) -- IEEE 754 standard reference

## Questions & Answers

### Q1: InnerParams struct vs additional buffer bindings?
**Answer**: Use InnerParams struct. Cleaner single binding. Breaking the kernel signature is fine — we control all callers.
**Impact**: sort_inner_fused gets `constant InnerParams& params [[buffer(3)]]` with {start_shift, pass_count, batch_start}. Existing u32 sort path updated to use the struct too.

### Q2: Gather dispatch inside or outside main sort command buffer?
**Answer**: Inside same command buffer/encoder. Saves ~97us overhead, maintains the single-encoder pattern proven optimal in experiments.
**Impact**: dispatch_sort_pipeline passes additional buffer refs for gather. All dispatches (transform + sort + gather) in one encoder.

### Q3: PsoCache extension in forge-primitives or local to forge-sort?
**Answer**: Extend forge-primitives. Function constants are a general-purpose Metal feature.
**Impact**: forge-primitives/src/pso_cache.rs gets FnConstant enum and get_or_create_specialized(). Available to all forge-compute crates.
