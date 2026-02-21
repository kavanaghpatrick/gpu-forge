---
spec: forge-runtime-p1
phase: design
created: 2026-02-21
generated: auto
---

# Design: forge-runtime-p1

## Overview

New `forge-runtime` crate in the `metal-forge-compute` workspace providing ForgeContext (shared Metal state), ForgeBuffer trait (zero-copy buffer interop), Pipeline builder (single CB multi-dispatch), and GPU Gather kernel. Existing primitives gain `with_context()` constructors while preserving standalone `::new()`.

## Architecture

```
forge-runtime (new)
├── ForgeContext ─────────── shared device + queue + buffer_pool
├── ForgeBuffer<T> trait ─── metal_buffer() + len() + capacity()
├── Pipeline builder ─────── single CB + encoder, chain operations
└── Gather kernel ────────── indices + source → permuted output

forge-sort (modified)
├── GpuSorter::with_context() ── accepts shared device/queue/library
├── GpuSorter::encode_sort()   ── encode onto external encoder (exists as encode_sort_pipeline)
└── SortBuffer::from_raw_parts() ── wrap existing MTLBuffer

forge-filter (modified)
├── GpuFilter::with_context() ── accepts shared device/queue/library
├── GpuFilter::encode_filter() ── encode onto external encoder
└── FilterBuffer::from_raw_parts() ── wrap existing MTLBuffer
```

## Components

### ForgeContext

**Purpose**: Single source of truth for Metal device, queue, and buffer pool. Injectable into all primitives.

**Fields**:
```rust
pub struct ForgeContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    buffer_pool: BufferPool,
}
```

**Responsibilities**:
- Create device + queue once via `MTLCreateSystemDefaultDevice()`
- Provide `&ProtocolObject<dyn MTLDevice>` and `&ProtocolObject<dyn MTLCommandQueue>` to primitives
- Buffer pool for page-aligned recycling (reuse existing `forge_primitives::BufferPool`)
- `alloc_buffer(size)` convenience method

**Key Decision**: ForgeContext does NOT hold PsoCache or MTLLibrary. Each primitive (GpuSorter, GpuFilter) loads its own metallib and manages its own PSOs. Reason: metallib paths are crate-specific, compiled by per-crate `build.rs`. Sharing would require runtime metallib path injection, adding complexity for no benefit.

### ForgeBuffer<T> Trait

**Purpose**: Common interface for any GPU-resident typed buffer. Enables zero-copy flow between primitives.

```rust
pub trait ForgeBuffer<T> {
    fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer>;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}
```

**Implementations**:
- `impl<T: SortKey> ForgeBuffer<T> for SortBuffer<T>` (in forge-sort)
- `impl<T: FilterKey> ForgeBuffer<T> for FilterBuffer<T>` (in forge-filter)
- `impl<T: FilterKey> ForgeBuffer<T> for FilterResult<T>` (in forge-filter, returns values_buf)

**Design Decision**: The trait is defined in forge-runtime but implemented in each primitive crate. This avoids orphan rule issues. forge-sort and forge-filter will have forge-runtime as an optional dependency behind a `runtime` feature flag.

**Alternative considered**: Define trait in forge-runtime, have forge-runtime depend on forge-sort/filter and impl there. Rejected: creates circular dependency potential and couples the runtime to specific primitive versions.

**Chosen approach**: ForgeBuffer trait lives in a new `forge-buffer` module within forge-runtime. SortBuffer and FilterBuffer get `from_raw_parts()` constructors. Conversions implemented in forge-runtime since it depends on both.

### Buffer Conversions

Zero-copy conversions by transferring `Retained<MTLBuffer>` ownership:

```rust
// In forge-sort: expose constructor
impl<T: SortKey> SortBuffer<T> {
    pub fn from_raw_parts(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        len: usize,
        capacity: usize,
    ) -> Self { ... }

    pub fn into_raw_parts(self) -> (Retained<ProtocolObject<dyn MTLBuffer>>, usize, usize) { ... }
}

// In forge-filter: expose constructor
impl<T: FilterKey> FilterBuffer<T> {
    pub fn from_raw_parts(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        len: usize,
        capacity: usize,
    ) -> Self { ... }

    pub fn into_raw_parts(self) -> (Retained<ProtocolObject<dyn MTLBuffer>>, usize, usize) { ... }
}

// In forge-filter: expose take on FilterResult
impl<T: FilterKey> FilterResult<T> {
    pub fn take_values_buffer(self) -> Option<(Retained<ProtocolObject<dyn MTLBuffer>>, usize, usize)> { ... }
    pub fn take_indices_buffer(self) -> Option<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)> { ... }
}
```

Conversions in forge-runtime:
```rust
// FilterResult → SortBuffer (for types that are both SortKey + FilterKey)
pub fn filter_result_to_sort_buffer<T>(result: FilterResult<T>) -> SortBuffer<T>
where T: SortKey + FilterKey
{
    let (buf, len, cap) = result.take_values_buffer().expect("no values buffer");
    SortBuffer::from_raw_parts(buf, len, cap)
}
```

**Note**: Cannot use `From<>` impl directly in forge-runtime for types defined in forge-sort/filter (orphan rules). Instead, provide free functions or extension methods. Alternatively, the `From<>` impls can live in forge-sort/filter behind a feature gate.

### Pipeline Builder

**Purpose**: Build and execute multi-primitive GPU pipelines with single command buffer.

```rust
pub struct Pipeline<'ctx> {
    ctx: &'ctx ForgeContext,
    // The pipeline accumulates operations and executes them
    // via a single command buffer + encoder
}

impl<'ctx> Pipeline<'ctx> {
    pub fn new(ctx: &'ctx ForgeContext) -> Self;
    pub fn execute(self) -> Result<(), PipelineError>;
}
```

**Design**: The Pipeline creates one `MTLCommandBuffer` + one `MTLComputeCommandEncoder` at construction. Operations encode dispatches onto this encoder. `execute()` calls `endEncoding()` + `commit()` + `waitUntilCompleted()`.

**Pipeline stages need mutable access to primitives** (for PSO cache, scratch buffers). The Pipeline holds references to pre-configured GpuSorter/GpuFilter instances:

```rust
let ctx = ForgeContext::new()?;
let mut sorter = GpuSorter::with_context(&ctx)?;
let mut filter = GpuFilter::with_context(&ctx)?;

// Encode filter onto pipeline's encoder
let filter_result = filter.encode_filter(&encoder, &input, &predicate)?;

// Convert result to sort buffer (zero-copy)
let sort_buf = filter_result_to_sort_buffer(filter_result);

// Encode sort onto same encoder
sorter.encode_sort(&encoder, &sort_buf)?;

// Single commit
encoder.endEncoding();
cmd_buf.commit();
cmd_buf.waitUntilCompleted();
```

**Simplified Pipeline API** wraps this pattern:

```rust
let result = Pipeline::new(&ctx)
    .with_sorter(&mut sorter)
    .with_filter(&mut filter)
    .filter(&input_buf, &Predicate::Gt(100))?
    .sort()?
    .execute()?;
```

### GPU Gather Kernel

**Purpose**: Index-based permutation on GPU. Takes source buffer + indices buffer, writes `dst[i] = src[indices[i]]`.

**Metal shader** (`shaders/gather.metal`):
```metal
#include <metal_stdlib>
using namespace metal;

constant bool IS_64BIT [[function_constant(0)]];

kernel void gather_u32(
    device const uint* src     [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device uint* dst           [[buffer(2)]],
    constant uint& count       [[buffer(3)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid < count) {
        dst[gid] = src[indices[gid]];
    }
}

kernel void gather_u64(
    device const ulong* src    [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device ulong* dst          [[buffer(2)]],
    constant uint& count       [[buffer(3)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid < count) {
        dst[gid] = src[indices[gid]];
    }
}
```

Two kernels (32-bit, 64-bit) rather than function constants for the element type, matching existing forge-sort/filter patterns.

**Rust wrapper**: `encode_gather()` function that encodes the dispatch onto an existing encoder.

### with_context() Constructors

**GpuSorter::with_context()**:
```rust
impl GpuSorter {
    pub fn with_context(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Result<Self, SortError> {
        // Clone device/queue (Retained is ref-counted)
        // Load metallib using same env! path
        // Pre-compile PSOs as in ::new()
        // Return with shared device/queue
    }
}
```

Takes `Retained<>` (cloned from ForgeContext) since `Retained` is cheap to clone (ARC increment). This avoids lifetime coupling between ForgeContext and primitives.

**GpuFilter::with_context()**: Same pattern.

## Data Flow

1. User creates `ForgeContext` (one device + queue)
2. User creates `GpuSorter::with_context()` and `GpuFilter::with_context()` sharing the context
3. User creates input `FilterBuffer<T>` from context
4. Pipeline creates CB + encoder from context's queue
5. Filter encodes 3 dispatches onto encoder -> produces `FilterResult<T>` (values_buf + indices_buf)
6. `FilterResult<T>` converted to `SortBuffer<T>` (zero-copy Retained transfer)
7. Sort encodes 4 dispatches onto same encoder
8. Gather encodes 1 dispatch onto same encoder (if needed)
9. `execute()` commits CB -> GPU executes all 8+ dispatches with implicit barriers
10. CPU reads result from UMA buffer (zero-copy)

## Technical Decisions

| Decision | Options | Choice | Rationale |
|----------|---------|--------|-----------|
| PsoCache location | Shared in ForgeContext vs per-primitive | Per-primitive | Each crate loads different metallib; shared PSO cache would need multi-library support |
| ForgeBuffer trait location | forge-runtime vs forge-primitives | forge-runtime | forge-runtime is the new integration crate; forge-primitives is for benchmarks |
| Buffer conversion mechanism | From<> trait vs free functions | Free functions in forge-runtime | Orphan rules prevent From<> impls across crate boundaries |
| with_context parameter type | &ForgeContext vs (device, queue) | (device, queue) as Retained | Avoids lifetime coupling; Retained clone is cheap (ARC) |
| Gather kernel variants | Function constants vs separate kernels | Separate kernels (gather_u32, gather_u64) | Matches forge-sort/filter pattern; simpler |
| Pipeline ownership model | Owns primitives vs borrows | Borrows (&mut GpuSorter, etc.) | Primitives own scratch buffers; pipeline is transient |

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `metal-forge-compute/Cargo.toml` | Modify | Add forge-runtime to workspace members |
| `metal-forge-compute/forge-runtime/Cargo.toml` | Create | New crate: depends on forge-sort, forge-filter, objc2-* |
| `metal-forge-compute/forge-runtime/src/lib.rs` | Create | Re-exports: ForgeContext, ForgeBuffer, Pipeline, gather |
| `metal-forge-compute/forge-runtime/src/context.rs` | Create | ForgeContext struct (~60 LOC) |
| `metal-forge-compute/forge-runtime/src/buffer.rs` | Create | ForgeBuffer trait + conversion functions (~50 LOC) |
| `metal-forge-compute/forge-runtime/src/pipeline.rs` | Create | Pipeline builder (~120 LOC) |
| `metal-forge-compute/forge-runtime/src/gather.rs` | Create | Gather kernel wrapper (~80 LOC) |
| `metal-forge-compute/forge-runtime/shaders/gather.metal` | Create | Gather GPU kernel (~30 LOC) |
| `metal-forge-compute/forge-runtime/build.rs` | Create | Compile gather.metal to metallib |
| `metal-forge-compute/forge-sort/src/lib.rs` | Modify | Add with_context(), from_raw_parts(), into_raw_parts() on SortBuffer |
| `metal-forge-compute/forge-filter/src/lib.rs` | Modify | Add with_context(), from_raw_parts(), into_raw_parts() on FilterBuffer, take_values_buffer() on FilterResult |
| `metal-forge-compute/forge-runtime/tests/pipeline_tests.rs` | Create | Integration tests |

## Error Handling

| Error | Handling | User Impact |
|-------|----------|-------------|
| Device not found | `ForgeContext::new()` returns `ForgeError::DeviceNotFound` | Same as existing primitives |
| Metallib load failure | `with_context()` returns primitive-specific error | Shader not compiled |
| Command buffer failure | `Pipeline::execute()` returns `PipelineError::GpuExecution` | GPU execution error |
| Buffer capacity mismatch | Conversion functions panic on invalid capacity | Programming error (debug assert) |
| Empty pipeline | `execute()` returns Ok(()) immediately | No-op is valid |

## Existing Patterns to Follow

- **Shader compilation**: `build.rs` uses `xcrun metal` + `xcrun metallib` (see forge-sort/build.rs, forge-filter/build.rs)
- **Buffer allocation**: `StorageModeShared` everywhere (UMA zero-copy)
- **Encoder pattern**: `queue.commandBuffer()` -> `cmd.computeCommandEncoder()` -> encode dispatches -> `enc.endEncoding()` -> `cmd.commit()` -> `cmd.waitUntilCompleted()`
- **PSO pre-compilation**: Done in constructor, lazy for less-common variants
- **Error types**: `thiserror` derive macros, `#[error("...")]` format strings
- **Threadgroup size**: 256 threads, `MTLSize { width: 256, height: 1, depth: 1 }`
