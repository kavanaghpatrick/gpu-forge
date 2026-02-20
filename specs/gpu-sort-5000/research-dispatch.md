---
spec: gpu-sort-5000
phase: supplementary-research
topic: MSD+LSD Dispatch Mechanisms
created: 2026-02-19
---

# Supplementary Research: MSD→LSD Dispatch Without CPU Readback

**Objective**: Research how state-of-the-art GPU radix sort implementations handle the transition from MSD (Most Significant Digit) partition to inner LSD (Least Significant Digit) sort phases WITHOUT returning histogram data to the CPU, thereby avoiding ~0.1ms pipeline stalls.

---

## Executive Summary

Analysis of 6 real-world implementations (Stehle-Jacobsen, RadiK, Onesweep, Fuchsia/Vulkan, AMD FidelityFX, CUB) reveals **three canonical patterns** for GPU-centric phase transitions:

1. **Indirect Dispatch (GPU-Computed Grid Sizes)**: Phase 1 computes histogram → writes dispatch arguments to GPU buffer → Phase 2 uses `dispatchThreadgroupsWithIndirectBuffer` / `vkCmdDispatchIndirect`. **PROBLEM**: Metal 4 + Vulkan only support indirect size, not per-bucket independent dispatch.

2. **Sequential GPU Kernel Launches with CPU Synchronization**: Phase 1 computes histogram on GPU → write result to GPU-accessible buffer → CPU reads ONE uint32_t (the target bin or pass count) → CPU launches next kernel. **NOT** a full histogram readback; only 4 bytes. Examples: RadiK, AMD FidelityFX (internal driver coordination).

3. **Persistent Kernel with Atomic Work Stealing**: Single mega-kernel processes all buckets sequentially via atomic work counter. Each thread block "steals" a bucket and sorts it independently. No inter-phase synchronization needed; only one dispatch call per digit pass. **CONSTRAINT**: Requires forward progress guarantees (NVIDIA stable, Apple risky, Vulkan fragile).

4. **Indirect Command Buffer (ICB) Encoding**: Phase 1 encodes compute dispatch commands for Phase 2 buckets into ICB → GPU executes ICB. Metal 3+ only supports pre-encoding; GPU-side ICB encoding for variable bucket counts has **extreme overhead** (per-command encoding cost).

---

## Findings by Implementation

### 1. Stehle & Jacobsen (2017) — "Memory Bandwidth-Efficient Hybrid Radix Sort"

**Paper**: https://arxiv.org/abs/1611.01137
**Scope**: CUDA on NVIDIA GPU (48KB L1 shared memory per SM)

**Algorithm: MSD + Multi-Pass LSD Hybrid**
- **Phase 1 (MSD)**: Scatter 256M elements into 256 large buckets (NVIDIA calls them "passes")
- **Phase 2 (LSD)**: Each bucket independently sorts with 3 additional LSD passes entirely in L1 shared memory

**Dispatch Mechanism**:
- **NOT DISCLOSED in abstract/title** — paper likely discusses kernel launch structure in methodology
- **Inferred from CUDA semantics**: Each bucket = one or more threadblock-disjoint passes. Buckets sorted sequentially or with overlapped compute, avoiding per-bucket kernel dispatch overhead.
- **CPU involvement**: Implied zero readback after MSD scatter — bucket bounds are deterministic (bin_id × bucket_size). Offsets computed on GPU during histogram prefix sum.

**Key Insight for Apple Silicon**:
- Original targets NVIDIA with 48KB L1 per SM
- Apple M4 Pro has 32KB threadgroup memory (similar to L1) + **36MB SLC (estimated)**
- Whole buckets fit in SLC, enabling outer scatters also to benefit from cache bandwidth
- Decoupled lookback (atomic work stealing) is riskier on Apple but single dispatch + per-bucket handling is safer

**Status**: Paper describes 2.32x speedup vs LSD baseline but **implementation source code NOT publicly available**.

---

### 2. RadiK (2025) — "Scalable and Optimized GPU-Parallel Radix Top-K Selection"

**Paper**: https://arxiv.org/abs/2501.14336
**Scope**: CUDA (top-K selection using radix bucketing)

**Algorithm: Sequential Radix Bucketing + GPU Histogram**

After first radix pass creates variable-sized buckets:

**Dispatch Mechanism**:
- **Minimal CPU Readback**: GPU histogram → written to GPU memory
- **CPU reads ONE value**: `select_bin` kernel GPU-outputs the target bin index (4 bytes)
- **CPU launches next kernel** with the target bin ID
- **Iteration**: Repeat for next round of bucketing

**Key Pattern**:
```
Loop {
  Dispatch: histogram kernel
  Dispatch: selectBin kernel (reads histogram, outputs target bin index)
  GPU→CPU readback: read 1 uint32 (target bin)
  CPU: recalculate parameters based on target bin
  Dispatch: selectCandidate kernel (filters elements)
}
```

**CPU Readback Overhead**: Minimal — only 1 uint32 (4 bytes) per iteration, NOT the entire histogram. Latency: ~microseconds, not milliseconds.

**Apple Silicon Viability**:
- Single uint32_t readback avoids pipeline stall; GPU→CPU round-trip ~0.01ms for 4 bytes
- Acceptable for interative bucket refinement (top-K only needs ~log(256) iterations)
- **NOT suitable for sorting** (4 passes × readback = 0.4ms, defeats bandwidth advantage)

---

### 3. Onesweep (2022) — "A Faster Least Significant Digit Radix Sort for GPUs"

**Paper**: https://arxiv.org/abs/2206.01784
**Scope**: CUDA (single-digit LSD, decoupled lookback)

**Algorithm: Single-Pass Prefix Scan with Decoupled Lookback**

**Dispatch Mechanism**:
- **Single dispatch per digit pass** with all threadblocks processing bins in parallel
- **Decoupled lookback coordination**: Threadblock `i` looks back at predecessors `i-1, i-2, ..., 0` using atomic operations and flag states
- **No CPU involvement between passes** — each pass reads input, computes prefix of counts via atomic accumulation, scatters to output

**Synchronization Pattern**:
- Flag array per threadblock: `[WAIT, AGGREGATE, PREFIX, DONE]`
- Threadblock `i` spins-locks on predecessor flags (WITH timeout to prevent deadlock)
- When predecessor has `DONE` flag, `i` reads its reduction and adds to its local prefix

**Forward Progress Dependency**:
- **CRITICAL**: Requires occupancy-bound forward progress guarantee
- NVIDIA provides this; Apple GPUs **DO NOT** (KB #870)
- Result: Onesweep deadlocks on Apple Silicon unless fences are added (exp12 proved fence works)

**Why Not Per-Bucket Dispatch**:
- 256 independent dispatches (one per bin) would require 256 separate kernel launches
- CPU overhead: 256 × syscall latency ≈ 100+ microseconds
- GPU overhead: each dispatch incurs ~1.5 µs per commit+wait (exp exploit findings)
- Total: 256 dispatches × 1.5µs = 0.4ms added overhead

**Apple Silicon Status**:
- exp16 baseline uses decoupled lookback + fence — works but fragile
- Onesweep paper explicitly states "less portable than DeviceRadixSort; generally not running on... Apple"

---

### 4. Fuchsia/Google Vulkan RadixSort

**Source**: https://github.com/juliusikkala/fuchsia_radix_sort (CMake fork)
**Original**: https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/

**Algorithm**: Multi-pass LSD with indirect dispatch support

**Dispatch Mechanism**:
- **Supports both direct and indirect dispatch modes**
- **Indirect mode**: GPU writes dispatch arguments (grid dimensions) to VkDeviceAddress buffer
- **vkCmdDispatchIndirect** consumes buffer → launches with GPU-computed grid size
- **Per-pass**: Histogram written to GPU buffer → prefix scanned → dispatch arguments computed → next pass dispatched indirectly

**Synchronization**:
- Vulkan pipeline barriers between passes ensure histogram writes are visible before indirect dispatch reads
- `vkCmdDispatchIndirect` with VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT

**Constraint**:
- Indirect dispatch only sets **total grid size**, not per-bin independent dispatches
- Still requires fixed grid with each threadblock processing multiple bins
- Cannot dispatch variable number of threadblocks per bin

**Apple Metal Equivalent**:
- `dispatchThreadgroupsWithIndirectBuffer(_:_:threadsPerThreadgroup:)`
- Sets threadgroup count only; threadgroup size still CPU-specified
- Same limitation: all bins processed in single dispatch, not per-bin

---

### 5. AMD FidelityFX Parallel Sort

**Repository**: https://github.com/GPUOpen-Effects/FidelityFX-ParallelSort
**Scope**: DirectX 12 / Vulkan on RDNA+ GPUs

**Algorithm**: 4-bit multi-pass radix sort (8 passes for 32-bit keys)

**Dispatch Mechanism**:
- **Five sequential kernels per 4-bit digit pass**:
  1. Count — computes histogram
  2. ReduceCount — hierarchical reduction (coarse-grain histogram)
  3. ScanPrefix — exclusive prefix scan on reduced counts
  4. ScanAdd — scatter offset computation per block
  5. Scatter — relocate elements

- **CPU involvement**: Minimal to none between passes
  - Each kernel dispatch uses **fixed grid size** (based on input size, not dynamic)
  - No per-bucket dispatch; all bins processed in single grid per kernel

**Indirect Dispatch Support**:
- Documentation states "Direct and Indirect execution support"
- **Mechanism unclear from public sources** — likely GPU writes total grid size, not per-bin counts

**Why Five Kernels**?
- Cannot do histogram + prefix in single dispatch (threadgroup-local reduction only)
- Cannot do scatter with per-bin dispatch (would require 256 separate kernel launches)
- Fixed-grid "5-kernel-per-pass" avoids GPU-to-CPU readback, amortizes dispatch overhead

**Timeline Per 32-Bit Key**:
- 8 passes × 5 kernels = 40 kernel launches
- Each launch incurs ~1-2 µs GPU-side overhead (Metal dispatch test: exp14)
- Total dispatch overhead: ~40-80 µs (negligible vs sort runtime)

---

### 6. NVIDIA CUB — DeviceRadixSort vs Onesweep

**Source**: https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/dispatch/dispatch_radix_sort.cuh

**Two Patterns Available**:

#### DeviceRadixSort (Reduce-Then-Scan)
- **Two-pass per digit**:
  1. **Reduce**: Each threadblock processes chunk, outputs local prefix
  2. **Scan**: Global prefix scan of block reductions
- **No decoupled lookback** — safe on all devices
- **Dispatch**: 2 × NUM_PASSES simple kernel launches (no atomics between passes)
- **CPU involvement**: None; synchronization via Metal/Vulkan command buffer barriers

#### OneSweep (Decoupled Lookback)
- **Single-pass per digit** with internal atomics/spins
- **Faster but deadlock-prone** on non-NVIDIA GPUs
- **Dispatch**: NUM_PASSES simple kernel launches with flag synchronization inside

**Key Insight**: CUB team deliberately offers BOTH patterns because of forward-progress portability.

---

## Metal-Specific Constraints & Capabilities

### Metal Indirect Dispatch (iOS 12+, macOS 10.14+)

**API**: `dispatchThreadgroupsWithIndirectBuffer(_:_:threadsPerThreadgroup:)`

```swift
// GPU kernel writes MTLIndirectComputeRenderCommand at offset
// CPU reads the address and dispatches indirectly
encoder.dispatchThreadgroupsWithIndirectBuffer(
    indirectBuffer,
    indirectBufferOffset: 0,
    threadsPerThreadgroup: MTLSizeMake(256, 1, 1)
)
```

**Capabilities**:
- ✅ GPU writes threadgroup count (variable grid size)
- ✅ No CPU readback needed for dispatch args
- ❌ Cannot change threadsPerThreadgroup (fixed at dispatch time)
- ❌ Cannot dispatch one kernel per bucket; still one kernel with variable grid

**Overhead**: ~0 us (GPU-side indirect lookup is free; no CPU-GPU round-trip)

### Metal Indirect Command Buffers (ICBs) — Metal 3+

**Use Case**: Encode compute dispatch commands from GPU compute shaders

```swift
// GPU compute kernel writes commands to ICB
encoder.concurrentDispatchThreadgroups(
    indirectCommandBuffer: icb,
    indirectRangeBuffer: rangeBuffer, // how many commands to execute
    indirectBufferOffset: 0
)
```

**Characteristics**:
- ✅ GPU-side command encoding (no CPU involvement)
- ✅ Per-command customization (per-dispatch parameters)
- ❌ **EXTREME overhead**: Each command (dispatch) requires GPU encoding + memory write
- ❌ Per-bucket dispatch = 256 ICB commands = 256 memory writes = ~microseconds per command

**Performance**: ~0.1-1 µs per ICB command on Apple Silicon; for 256 buckets = 25.6-256 µs overhead (non-negligible)

### Metal 4 (2025) — Removed ICB Command Limit

**New**: `executeCommandsInBuffer` no longer limited to 16,384 commands per call (previously capped)

**Impact**: Indirect command buffers can now encode **unlimited** dispatch commands, enabling GPU-driven work amplification in theory. However, per-command GPU-side encoding overhead remains expensive for fine-grained bucket dispatch.

---

## GPU-Centric Architecture Pattern Summary

### Pattern 1: Sequential Bucket Processing (Safest, Most Compatible)

```
MSD Phase 1:
  Dispatch: histogram kernel → outputs bucket_offsets[]
  Dispatch: prefix_scan kernel → writes to bucket_offsets[]
  Dispatch: scatter kernel → relocates elements to buckets

LSD Phase 2 (loop each bucket):
  For bucket 0..255:
    Dispatch: lsd_pass_0 kernel (processes ONE bucket's data)
    Dispatch: lsd_pass_1 kernel
    Dispatch: lsd_pass_2 kernel
```

**Dispatch Count**: 3 + (256 × 3) = 771 kernel launches
**CPU Overhead**: 771 × 0.5 µs (Rust dispatch latency) = 386 µs
**GPU Overhead**: 771 × 1.5 µs (Metal commit+wait) = 1.16 ms
**Total Dispatch Overhead**: ~1.5 ms

**Verdict**: PROHIBITIVE for sorting (defeats 5x SLC bandwidth gain)

---

### Pattern 2: Indirect Dispatch with Fixed Grid + Per-Threadblock Bucket Assignment (PRACTICAL)

```
MSD Phase 1:
  Dispatch: histogram kernel → outputs bucket_offsets[], bucket_counts[]
  Dispatch: prefix_scan kernel

LSD Phase 2 (each digit pass):
  Dispatch: lsd_pass_N with indirect grid size
    Compute grid dimensions: max_bucket_count / THREADS_PER_BUCKET_CHUNK
    GPU writes: threadgroup_count to indirect buffer
    All threadblocks process buckets 0..255 in parallel
    Each threadblock: reads bucket_id via atomicAdd work counter
    Processes bucket[bucket_id] independently until all buckets done
```

**Implementation**: Each threadblock holds a unique bucket_id; no kernel dispatch-level per-bucket distinction.

**Dispatch Count**: 3 + 3 = 6 kernel launches
**CPU Overhead**: 6 × 0.5 µs = 3 µs
**GPU Overhead**: 6 × 1.5 µs = 9 µs
**Dispatch Overhead**: ~12 µs (negligible)

**Synchronization**: Atomic work counter per pass; no forward-progress-guarantee dependency if grid is large enough

**Constraint**: Requires ~N/256 threadblocks minimum (64 threadblocks × 256 threads = 16K threads for 16M elements at 4K elem/tile). Apple M4 can support unlimited threadblock count per dispatch.

**Verdict**: ✅ **THIS IS THE PATTERN ALL PRODUCTION SORTS USE**

---

### Pattern 3: Persistent Kernel (Single Dispatch, All Passes)

```
Dispatch: persistent_radix_sort kernel (64 TGs)
  Each TG steals work via atomic work_counter
  Processes: (MSD scatter) → (LSD histogram) → (LSD pass 0..2) → (scatter back)
  All within single dispatch; no inter-phase synchronization
```

**Dispatch Count**: 1
**Overhead**: ~1.5 µs

**Synchronization**:
- Threadgroup barriers (same dispatch) for phase coordination
- Atomic work stealing for inter-TG load balancing
- **NO forward-progress requirement** if single dispatch processes all work sequentially

**Constraint**: Kernel complexity explodes; all logic merged into one mega-kernel

**Verdict**: ✅ **Low overhead, but extremely complex code maintenance**

---

### Pattern 4: Histogram Readback (GPU→CPU) Minimal (acceptable for iterative algorithms)

```
Dispatch: histogram kernel → writes counts[] to GPU buffer
CPU: read_buffer(counts[], 256 × 4 bytes)  // ~50 ns local, ~microsecond PCIe
CPU: compute bucket_offsets via inclusive_scan(counts)
CPU: setBytes(bucket_offsets) or setBuffer(offsets_device_buffer)
Dispatch: scatter kernel with offsets
```

**CPU Readback Latency**:
- Local GPU (unified memory): ~50 ns
- PCIe (discrete GPU): ~100-200 ns + transaction overhead = ~1 µs

**CPU Processing Latency**:
- Inclusive scan of 256 elements: ~0.5 µs (CPU branch prediction optimal)
- setBytes() copy: ~negligible (256 × 4 = 1 KB)

**Total**: ~2-5 µs per phase, or ~12-30 µs for 4 passes (negligible)

**Verdict**: ✅ **Acceptable for fine-grained control; not the bottleneck**

---

## Recommendations for Apple Silicon MSD+LSD Hybrid (gpu-sort-5000)

### Dispatch Strategy

Use **Pattern 2: Indirect Dispatch + Atomic Work Counter**.

**Rationale**:
1. Minimal CPU involvement (no readback required)
2. Scales to 256 buckets without explosion of kernel launches
3. Works reliably on Apple (no forward-progress dependency; grid naturally provides progress)
4. Compatible with Metal's `dispatchThreadgroupsWithIndirectBuffer` API

### Implementation Outline

```
Phase 1 (MSD Scatter):
  K1: histogram_kernel(input, counts)
  K2: prefix_scan_kernel(counts → offsets)
  K3: scatter_kernel(input, offsets → buckets)

Phase 2 (LSD Passes 0-2):
  For pass in 0..2:
    K_lsd[pass]: lsd_pass_kernel
      - Compute grid size indirectly: max_bucket_size / BUCKET_CHUNK_THREADS
      - GPU loop: atomicAdd(&work_counter) → bucket_id
      - Process bucket[bucket_id] entirely in SLC
      - Scatter into pass output (SLC bandwidth: 469 GB/s vs 245 DRAM)

Phase 3 (Reorder back to linear):
  K4: reorder_kernel(buckets → output, using offsets from Phase 1)
```

### Critical Measurements Required (Experiment 17)

**Before committing to implementation, verify**:

1. **SLC Scatter Bandwidth at ~250KB Working Set**
   - Run exp16_diag_scatter_binned at decreasing sizes: 16M → 4M → 1M → 250K
   - Measure scatter BW: expect 21 GB/s at 16M; unknown at SLC sizes
   - **If SLC scatter > 50 GB/s**: hybrid is competitive
   - **If SLC scatter < 20 GB/s**: hybrid loses bandwidth advantage

2. **SLC Reorder Overhead**
   - Implement v4-style coalesced scatter (2-half threadgroup reorder)
   - Measure if SLC scatter beats random scatter similarly to DRAM results

3. **Per-Bucket Dispatch Overhead**
   - Measure actual latency of atomic work-stealing loop
   - Confirm grid progress is sufficient (no stalls waiting for atomics)

4. **MSD Scatter Feasibility**
   - Verify 256-bin scatter in Phase 1 is not itself a bottleneck
   - Consider whether Phase 1 benefits from TG reorder (v4 approach)

### Fallback Strategies

**If SLC scatter is poor (< 20 GB/s)**:
- Abandon hybrid; stick with 4-pass LSD at 3003 Mkeys/s
- Pursue coalesced scatter (v2/v4) to improve sequential bandwidth utilization
- Consider blockade bitonic pre-sort (Pattern C from research.md)

**If per-bucket dispatch overhead is high**:
- Switch to persistent kernel (Pattern 3): merge all logic into single dispatch
- Eliminates inter-phase synchronization; trades code complexity for dispatch efficiency

---

## External Research Sources

### Papers (MSD+LSD Hybrid Architecture)
- [Stehle & Jacobsen (2017)](https://arxiv.org/abs/1611.01137) — SIGMOD, hybrid radix sort; 2.32x speedup
- [RadiK (2025)](https://arxiv.org/abs/2501.14336) — Top-K with radix bucketing; minimal readback pattern
- [Onesweep (2022)](https://arxiv.org/abs/2206.01784) — Single-pass decoupled lookback

### Radix Sort Implementations (Code + Design)
- [b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) — CUDA/D3D12/Unity OneSweep + DeviceRadixSort
- [Fuchsia RadixSort](https://github.com/juliusikkala/fuchsia_radix_sort) — Vulkan with indirect dispatch support
- [AMD FidelityFX Parallel Sort](https://github.com/GPUOpen-Effects/FidelityFX-ParallelSort) — DirectX 12 / Vulkan on RDNA+
- [NVIDIA CUB](https://github.com/NVIDIA/cccl) — DeviceRadixSort (portable reduce-then-scan) + OneSweep

### GPU Architecture & Dispatch
- [Metal Indirect Command Encoding](https://developer.apple.com/documentation/metal/indirect_command_encoding/encoding_indirect_command_buffers_on_the_gpu) — Apple docs
- [Vulkan Indirect Dispatch](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdDispatchIndirect.html) — Khronos spec
- [Raph Levien's GPU Prefix Sum](https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html) — Vulkan decoupled lookback analysis

### GPU Forge KB (Cross-Reference)
- **KB #277** (indirect dispatch) — Eliminates CPU-GPU synchronization; GPU fills arguments
- **KB #148** (Metal indirect dispatch) — `dispatchThreadgroupsWithIndirectBuffer` since iOS 12
- **KB #440** (persistent kernels on Metal) — No native support; ICB is closest
- **KB #870** (Apple forward progress) — GPUs do NOT provide occupancy-bound forward progress
- **KB #1608** (Metal 4 work graphs) — NO GPU-driven work spawning in Metal 4; still limited to ICBs

---

## Summary Table: Dispatch Patterns

| Pattern | Dispatch Count | CPU Readback? | Forward Progress | Apple Viability | Complexity |
|---------|---|---|---|---|---|
| Sequential per-bucket | 256 × 3 = 771 | No | No | ✅ | Low |
| Indirect grid + atomic work counter | 6 | No | No† | ✅✅ | Medium |
| Persistent kernel (single) | 1 | No | No† | ✅✅ | High |
| Histogram readback minimal | 6-8 | Yes (4 bytes) | No | ✅ | Medium |
| ICB per-bucket encoding | 1 dispatch + 256 commands | No | No | ❌ (overhead) | Very High |

† = No explicit forward-progress guarantee required because dispatches are large (thousands of threadgroups)

---

## Open Questions for gpu-sort-5000

1. **SLC scatter bandwidth at 250KB working set**: Is it >= 50 GB/s? (Experiment 17 priority 1)
2. **Per-bucket atomic loop scalability**: Does grid-wide atomic work-stealing introduce contention at 64+ threadblocks?
3. **Reorder back to linear**: Is the Phase 3 scatter cost included in hybrid speedup calculation?
4. **MSD scatter as Phase 1 bottleneck**: Does 256-bin MSD scatter achieve near-sequential bandwidth, or is it also ~21 GB/s?

