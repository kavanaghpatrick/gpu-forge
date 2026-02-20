---
spec: forge-sort
phase: requirements
created: 2026-02-20
---

# Requirements: forge-sort

## Goal

Extract exp17 Investigation W radix sort into `forge-sort` crate within the metal-forge-compute workspace. Production-quality API, proven 4-dispatch architecture (5431 Mk/s @ 16M on M4 Pro), reusable buffers, proper error handling. No algorithm changes.

## User Stories

### US-1: Sort u32 Slices via GpuSorter

**As a** library consumer
**I want to** call `GpuSorter::sort_u32(&mut self, data: &mut [u32])` to sort data on GPU
**So that** I get a simple, safe API without managing Metal resources

**Acceptance Criteria:**
- [ ] AC-1.1: `GpuSorter::new()` returns `Result<Self, SortError>`, initializes device, queue, compiles 4 PSOs
- [ ] AC-1.2: `sort_u32()` sorts in-place via host memcpy to GPU buffer, GPU sort, memcpy back
- [ ] AC-1.3: `sort_u32()` returns `Result<(), SortError>` with meaningful error variants
- [ ] AC-1.4: Empty slice `sort_u32(&mut [])` returns `Ok(())` immediately (no GPU dispatch)
- [ ] AC-1.5: Single-element slice returns `Ok(())` without GPU dispatch

### US-2: Correct Sorting at All Sizes

**As a** library consumer
**I want** sorted output to match `data.sort()` for any valid input
**So that** I can trust the GPU sort as a drop-in replacement

**Acceptance Criteria:**
- [ ] AC-2.1: Correct at 1K, 4K, 16K, 64K, 256K, 1M, 4M, 16M, 32M random u32
- [ ] AC-2.2: Correct for all-zeros input (16M)
- [ ] AC-2.3: Correct for all-same-value input (e.g., all 0xDEADBEEF)
- [ ] AC-2.4: Correct for pre-sorted ascending input (16M)
- [ ] AC-2.5: Correct for pre-sorted descending input (16M)
- [ ] AC-2.6: Correct for input size not divisible by 4096 (tile size)
- [ ] AC-2.7: Correct for sizes below one tile (n < 4096)

### US-3: Extract and Rename 4 Metal Kernels

**As a** crate maintainer
**I want** 4 kernels extracted from exp17_hybrid.metal into `forge-sort/shaders/sort.metal`
**So that** the crate is self-contained with clean naming

**Acceptance Criteria:**
- [ ] AC-3.1: `exp17_msd_histogram` renamed to `sort_msd_histogram`
- [ ] AC-3.2: `exp17_msd_prep` renamed to `sort_msd_prep`
- [ ] AC-3.3: `exp17_msd_atomic_scatter` renamed to `sort_msd_atomic_scatter`
- [ ] AC-3.4: `exp17_inner_fused_v4` renamed to `sort_inner_fused`
- [ ] AC-3.5: Struct names: `Exp17Params` -> `SortParams`, `BucketDesc` kept as-is
- [ ] AC-3.6: Shader constants renamed from `EXP17_*` to `SORT_*`
- [ ] AC-3.7: `build.rs` compiles `sort.metal` with `-std=metal3.2` flag
- [ ] AC-3.8: No runtime or correctness changes to kernel logic

### US-4: Reusable Buffer Pool

**As a** library consumer calling sort repeatedly
**I want** GPU buffers to persist and grow across calls
**So that** repeated sorts avoid re-allocation overhead

**Acceptance Criteria:**
- [ ] AC-4.1: First `sort_u32()` allocates 5 buffers (buf_a, buf_b, msd_hist, counters, bucket_descs)
- [ ] AC-4.2: Second `sort_u32()` with same or smaller size reuses existing buffers (zero allocations)
- [ ] AC-4.3: Larger subsequent call grows buffers (buf_a, buf_b) to new size; never shrinks
- [ ] AC-4.4: Metadata buffers (msd_hist=1KB, counters=1KB, bucket_descs=4KB) allocated once, reused always

### US-5: Performance Matching Experiment Results

**As a** library consumer
**I want** sort throughput matching exp17 Investigation W benchmarks
**So that** extraction doesn't introduce performance regressions

**Acceptance Criteria:**
- [ ] AC-5.1: >= 5000 Mk/s @ 16M random u32 (median of 50 runs after 5 warmup)
- [ ] AC-5.2: >= 4000 Mk/s @ 32M random u32
- [ ] AC-5.3: 4-dispatch execution: histogram, prep, scatter, fused-inner (single encoder, single command buffer)
- [ ] AC-5.4: No CPU readback between dispatches

### US-6: Workspace Integration

**As a** crate maintainer
**I want** forge-sort as a workspace member of metal-forge-compute
**So that** it shares deps and integrates with forge-bench

**Acceptance Criteria:**
- [ ] AC-6.1: `Cargo.toml` at `metal-forge-compute/forge-sort/` with `forge-primitives` dependency
- [ ] AC-6.2: Workspace `Cargo.toml` includes `"forge-sort"` in members list
- [ ] AC-6.3: `cargo build -p forge-sort` succeeds
- [ ] AC-6.4: `cargo test -p forge-sort` passes all tests
- [ ] AC-6.5: `SortParams` and `BucketDesc` structs defined in forge-sort (NOT in forge-primitives types.rs)

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | `GpuSorter::new()` initializes device, queue, PSO cache, compiles 4 kernels | High | `new()` returns Ok, PSO cache has 4 entries |
| FR-2 | `sort_u32()` dispatches 4 kernels in single encoder | High | GPU timeline shows 4 dispatches, no encoder boundary |
| FR-3 | Dispatch 1: histogram counts MSD byte (24:31) across N/4096 TGs | High | `buf_msd_hist` sums to N |
| FR-4 | Dispatch 2: prep computes exclusive prefix sum + BucketDesc (1 TG) | High | `counters[i]` = prefix_sum(hist[0..i]) |
| FR-5 | Dispatch 3: atomic scatter distributes to 256 buckets | High | Elements partitioned by MSD byte |
| FR-6 | Dispatch 4: fused inner sort — 3 LSD passes per bucket (256 TGs) | High | Each bucket internally sorted on bits 0:23 |
| FR-7 | Result copied back to caller's `&mut [u32]` after GPU completion | High | `data` is sorted after `sort_u32()` returns |
| FR-8 | `buf_a`/`buf_b` grow to `max(current, n*4)` bytes on each call | Medium | No allocation when n <= previous max |
| FR-9 | `SortError` enum with DeviceNotFound, ShaderCompilation, GpuExecution, EmptyInput | High | Each variant has descriptive context |
| FR-10 | `build.rs` compiles `shaders/sort.metal` to `sort.metallib` | High | metallib found at runtime by forge-primitives path logic |

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Sort throughput @ 16M | Mk/s (median, 50 runs) | >= 5000 |
| NFR-2 | Sort throughput @ 32M | Mk/s (median, 50 runs) | >= 4000 |
| NFR-3 | GPU memory @ 16M | bytes | ~128 MB (2x data + 6KB metadata) |
| NFR-4 | API latency overhead | us per sort call | < 100 (host memcpy + dispatch overhead) |
| NFR-5 | Crate compile time | seconds | < 30 (shader compilation included) |

## Glossary

- **MSD**: Most Significant Digit — the first radix pass sorts by bits 24:31
- **LSD**: Least Significant Digit — inner passes sort by bits 0:7, 8:15, 16:23
- **Fused inner**: Single kernel that performs all 3 LSD passes per bucket with TG barriers
- **Atomic scatter**: Each tile atomically increments per-bin counters for write positions (no lookback)
- **BucketDesc**: Struct {offset, count, tile_count, tile_base} describing each MSD bucket
- **Tile**: 4096 elements processed by one threadgroup (256 threads x 16 elements each)
- **PSO**: Pipeline State Object — compiled Metal compute kernel
- **Mk/s**: Millions of keys sorted per second (N / time_ms / 1000)

## Out of Scope

- Key-value pair sorting (u32 keys only in v1)
- Key types other than u32 (no f32, u64, i32)
- Descending sort order
- Partial/top-K sort
- Multi-GPU / distributed sort
- Async/non-blocking API
- Benchmark harness integration with forge-bench (separate task)
- New algorithm optimizations beyond Investigation W

## Dependencies

- `forge-primitives` crate (MetalContext, PsoCache, BufferPool, alloc helpers)
- `objc2-metal` 0.3 (direct API calls for command buffer, setBytes)
- Metal 3.2 runtime (for `atomic_thread_fence` in fused inner kernel)
- Apple Silicon GPU (M1 or later)

## Success Criteria

- `cargo test -p forge-sort` passes: correctness at 9 sizes + 5 edge cases
- `sort_u32()` at 16M achieves >= 5000 Mk/s (no regression from exp17 W)
- Library is usable with 3 lines: `let mut sorter = GpuSorter::new()?; sorter.sort_u32(&mut data)?;`
- Zero `unsafe` in public API (all unsafety internal)

## Unresolved Questions

- Should `GpuSorter` use forge-primitives `MetalContext` directly or own its own device/queue?
  - Recommendation: own its own (simpler API, no lifetime coupling)
- Should `GpuSorter` use forge-primitives `BufferPool` or inline buffer management?
  - Recommendation: inline (only 5 buffers, pool overhead not justified)
- Should metallib path resolution use forge-primitives logic or custom?
  - Recommendation: copy pattern from forge-primitives `MetalContext::try_load_metallib()`

## Next Steps

1. Approve requirements
2. Design phase: module layout, buffer lifecycle, build.rs details
3. Tasks: extract kernels, write Rust host code, tests, benchmarks
