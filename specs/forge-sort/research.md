# Research: forge-sort

## Goal

Production GPU radix sort library extracted from exp17 Investigation W. MSD+fused-inner 4-dispatch architecture achieving 5431 Mk/s @ 16M on M4 Pro. New crate in metal-forge-compute workspace using forge-primitives.

## Algorithm (Proven — No Research Needed)

Investigation W from exp17: 4 dispatches, single encoder, single command buffer.

| Dispatch | Kernel | Grid | Purpose |
|----------|--------|------|---------|
| 1 | `sort_msd_histogram` | N/4096 TGs × 256 | Count byte 24:31 distribution |
| 2 | `sort_msd_prep` | 1 TG × 256 | Exclusive prefix + bucket descriptors |
| 3 | `sort_msd_atomic_scatter` | N/4096 TGs × 256 | Atomic scatter to 256 buckets |
| 4 | `sort_inner_fused` | 256 TGs × 256 | 3-pass LSD per bucket (self-contained) |

**Performance** (measured, M4 Pro):
- 1M: 1688, 2M: 4354, 4M: 7662, 8M: 6315, 16M: 5431, 32M: 4752 Mk/s
- All sizes correct, all pass verification

## Source Files to Extract

| Source | Lines | Target |
|--------|-------|--------|
| `metal-gpu-experiments/shaders/exp17_hybrid.metal` (4 kernels) | ~400 | `forge-sort/shaders/sort.metal` |
| `metal-gpu-experiments/src/exp17_hybrid.rs` (bench_investigation_w_at_size) | ~120 | `forge-sort/src/lib.rs` |
| `metal-gpu-experiments/build.rs` | ~60 | `forge-sort/build.rs` |

### Kernel Rename Map

| exp17 Name | Production Name |
|------------|-----------------|
| `exp17_msd_histogram` | `sort_msd_histogram` |
| `exp17_msd_prep` | `sort_msd_prep` |
| `exp17_msd_atomic_scatter` | `sort_msd_atomic_scatter` |
| `exp17_inner_fused_v4` | `sort_inner_fused` |
| `Exp17Params` | `SortParams` |
| `BucketDesc` | `BucketDesc` (keep) |

## forge-primitives Compatibility

**Available and sufficient:**
- `MetalContext` — device + queue + library loading
- `PsoCache` — compile and cache PSOs by name
- `alloc_buffer()` / `alloc_buffer_with_data()` — buffer allocation
- `read_buffer_slice()` — readback
- `GpuTimer` — GPU-side timing

**Gaps requiring direct Metal API calls:**
1. **No `command_buffer()`** — use `ctx.queue.commandBuffer()` directly
2. **No `setBytes` helper** — call `enc.setBytes_length_atIndex()` directly
3. **`library` is `Option<>`** — use `ctx.library()` which panics if not loaded
4. **build.rs needs `-std=metal3.2`** — for atomic_thread_fence in inner sort

These gaps are trivial — 3-4 lines of direct objc2-metal calls each.

## Buffer Requirements

| Buffer | Size @ 16M | Purpose |
|--------|-----------|---------|
| buf_a | 64 MB | Working buffer (input copied here) |
| buf_b | 64 MB | Working buffer (ping-pong) |
| buf_msd_hist | 1 KB | MSD histogram (256 × u32) |
| buf_counters | 1 KB | Atomic scatter counters (256 × u32) |
| buf_bucket_descs | 4 KB | Bucket metadata (256 × BucketDesc) |
| **Total** | **~128 MB** | |

Buffers are reusable across calls. Grow on demand, never shrink.

## API Design

```rust
pub struct GpuSorter { /* device, queue, psos, buffers */ }

impl GpuSorter {
    pub fn new() -> Result<Self, SortError>;
    pub fn sort_u32(&mut self, data: &mut [u32]) -> Result<(), SortError>;
}

pub enum SortError {
    DeviceNotFound,
    ShaderCompilation(String),
    EmptyInput,
    GpuExecution(String),
}
```

## Build System

Copy `metal-gpu-experiments/build.rs` with one change: add `-std=metal3.2` flag.
The build.rs pattern works for library crates — forge-primitives already uses it.

## Testing Strategy

- Correctness: compare against `data.sort()` at 1K, 64K, 1M, 4M, 16M
- Edge cases: empty, single element, all-same values, pre-sorted, reverse-sorted
- Performance sanity: assert 16M completes in < 10ms

## Risks

1. **PsoCache hardcodes maxTotalThreadsPerThreadgroup=256** — this is exactly what our kernels need, no issue
2. **metallib path finding** — forge-primitives searches build/ directory, should work for workspace members
3. **Large buffer allocation** — 128MB for 16M elements, Apple Silicon unified memory handles this fine

## Recommendations

1. Keep it simple: one struct, one method, minimal API surface
2. Extract kernels verbatim from exp17, only rename prefixes
3. Grow buffers lazily on first call / when size increases
4. No key-value sort in v1 — just u32 keys (add later if needed)
