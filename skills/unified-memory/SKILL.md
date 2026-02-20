---
name: unified-memory
layer: 0
description: >
  This skill should be used when the conversation involves Apple Silicon unified memory architecture,
  CPU-GPU shared memory, memory storage modes, zero-copy buffers, SLC (System Level Cache) behavior,
  memory bandwidth, cache coherency, virtual address translation, memory compression, memory pressure,
  IOSurface sharing, or framework interop through shared buffers. Trigger keywords: MTLStorageMode,
  makeBuffer(bytesNoCopy:), shared memory, private storage, managed storage, zero-copy, SLC cache,
  cache coherency, unified memory, LPDDR5X, memory bandwidth, page fault, DART, UAT, IOSurface,
  MTLHazardTrackingMode, MTLPurgeableState, memory limits, recommendedMaxWorkingSetSize, Apple Fabric,
  constant address offset, CPU-GPU transfer, bandwidth QoS.
skills: []
cross_references:
  - gpu-silicon
  - metal-compute
  - gpu-io
  - gpu-perf
---

# Unified Memory Knowledge

This skill provides expertise in Apple Silicon's unified memory architecture. Use when questions involve how CPU and GPU share physical memory, storage mode selection, zero-copy buffer patterns, SLC cache behavior, memory bandwidth characteristics, virtual address translation, or cross-framework memory sharing.

## Domain Overview

Apple Silicon's defining architectural feature is unified memory: CPU, GPU, Neural Engine, and media engines all share a single pool of LPDDR5X SDRAM accessed through the System Level Cache (SLC). Unlike discrete GPU systems where data must be copied across a PCIe bus between CPU RAM and GPU VRAM, Apple's design enables true zero-copy sharing. A buffer allocated once in unified memory is directly accessible by any on-chip processor without physical data movement. This eliminates the single largest bottleneck in traditional heterogeneous computing.

The memory subsystem is built around LPDDR5X SDRAM soldered directly to the package substrate with memory bus widths that scale by chip tier: 128-bit (base), 256-bit (Pro), 512-bit (Max), 1024-bit (Ultra). M4 base delivers 120 GB/s with LPDDR5X-7500, while M5 upgrades to LPDDR5X-9600 for 153.6 GB/s on the base chip. Real-world GPU memory bandwidth efficiency is remarkably high, with measured STREAM results showing 90-100% of theoretical bandwidth utilization. CPU and GPU can achieve comparable bandwidth independently, but compete for the same physical bandwidth when active simultaneously.

The SLC serves as a shared last-level cache between all on-chip processors. SLC sizes range from 8 MB (base chips) to 128 MB (M4 Max), with 128-byte cache lines and a pseudo-random replacement policy. The SLC is exclusive with respect to CPU caches (data in SLC is not duplicated in CPU L2) but inclusive with respect to GPU caches. SLC bandwidth is approximately 2x DRAM bandwidth per GPU core, and SRAM reads consume single-digit picojoules versus hundreds of picojoules for DRAM reads, making SLC hits dramatically more energy-efficient.

Cache coherency between CPU and GPU is maintained automatically by hardware on Apple Silicon. No explicit cache flushes or invalidations are needed for shared memory access. Measured coherency latencies show L1/L2 hits at ~27ns (~80 GPU ticks), SLC hits at ~100ns (~300 ticks), and DRAM accesses at ~180ns (~540 ticks). The Apple Fabric on-chip interconnect connects all processors, media engines, and memory controllers, routing traffic with QoS-based bandwidth allocation at the thread level.

Virtual address translation on the GPU uses the UAT (Unified Address Translator) with ARM64-identical page tables, managed by the DART (Device Address Resolution Table) IOMMU. A critical architectural detail: CPU and GPU virtual addresses are always offset by a constant value for the same physical page, enabling pointer arithmetic across address spaces. GPU page faults are handled as stop-the-world events by the GPU firmware, making them extremely expensive and worth avoiding through proper resource residency management.

## Key Knowledge Areas

The unified-memory skill contains 104 findings across these domains:

- **Storage Modes**: MTLStorageModeShared vs .private vs .managed behavior on UMA, lossless compression for private textures, storage mode selection guidelines (Findings #73-77)
- **SLC Architecture**: SLC sizes per chip generation (M1-M5), exclusive/inclusive policy, 128-byte cache lines, pseudo-random replacement, set indexing behavior (Findings #84-88, #96-98, #122)
- **SLC-GPU Interaction**: Tile memory SLC residency, SLC as effective GPU L3, SLC bandwidth per core, threadgroup memory as on-chip SRAM (Findings #111-114)
- **Cache Coherency**: Automatic hardware coherency, hybrid SLC inclusiveness, coherency latencies, coherent(device) MSL attribute (Findings #54-55, #125, #131)
- **Memory Controllers**: Bus widths and bandwidth per chip tier (M4/M5 families), LPDDR5X speeds, channel configurations (Findings #60-63, #71)
- **Zero-Copy Buffers**: makeBuffer(bytesNoCopy:) requirements, 16KB page alignment, IOSurface sharing, CVPixelBuffer-to-Metal pipeline (Findings #89-94)
- **Memory Bandwidth**: STREAM benchmarks, bandwidth partitioning between CPU and GPU, QoS-based scheduling, measured efficiency (Findings #101-104)
- **Memory Limits**: 75% GPU allocation limit, recommendedMaxWorkingSetSize, maxBufferLength, memory pressure behavior (Findings #68, #105-109)
- **Virtual Addressing**: UAT page tables, DART IOMMU, constant CPU-GPU address offset, GPU page fault cost (Findings #56, #126-127, #130)
- **Memory Packaging**: On-package LPDDR5X, interleaving across controllers, in-line ECC (Findings #69-72)
- **Hazard Tracking**: MTLHazardTrackingModeUntracked, purgeable states, manual dependency management (Findings #78-79)
- **Hardware Compression**: Lossless and lossy memory compression, virtual memory compression under pressure (Findings #57, #108)
- **Framework Interop**: IOSurface cross-process sharing, MTLSharedTextureHandle, Core Image to Metal rendering (Findings #91-94)
- **Memory Ordering**: TSO on CPU cores, memory ordering guarantees across CPU-GPU boundary (Finding #129)
- **Neural Accelerator Memory**: M5 per-core neural accelerators accessing unified memory via SLC (Finding #53)
- **Academic Research**: GPU-driven virtual memory (GPUVM), SVM prefetching/thrashing, GMLake virtual memory stitching, unified physical memory characterization on AMD MI300A (Findings #482-492, #505-525)
- **GPU Memory Model**: Weakly-ordered memory model, relaxed atomics only (no acquire/release/seq_cst), TSO vs ARM weak ordering (8.94% performance difference), no didModifyRange needed on Apple Silicon (Findings #632-#635, #637)
- **Device-Scope Coherency**: MSL 3.2 coherent(device) qualifier, GPU memory barriers depth (threadgroup_barrier/simdgroup_barrier mem_flags), device-scope coherency details (Finding #636)
- **Metal 4 Memory Management**: MTL4CommandAllocator decouples command memory from buffers, MTLResidencySet as exclusive residency mechanism, placement sparse resources, no auto-retain, no auto-hazard tracking (Findings #639-#643)
- **SLC Architecture Depth**: EXAM paper findings on SLC bit indexing (lowest 13 bits excluded), 128-byte cache line discrepancy with sysctl, M3 Pro bandwidth reduction and M4 Pro restoration (Findings #648-#649)
- **Physical Memory Architecture**: TechInsights M4 Max die analysis, LPDDR5X interleaving patterns, in-line ECC (Finding #650)
- **Hardware Compression Limitation**: Lossless compression applies to textures only, NOT general compute buffers; macOS VM compression is separate mechanism (Findings #651-#652)
- **Virtual Address Translation Depth**: 4-level UAT page tables (L0-L3), TTBR0/TTBR1 split, PTE format divergence from standard ARM64, DART vs UAT separation (DART=IOMMU, UAT=GPU-specific) (Findings #653-#656)
- **GPU TLB**: Estimated ~2048 entries covering ~32 MB, TLB miss penalty, 16KB page size enables larger VIPT L1 cache (Finding #657)
- **Zero-Copy Pipeline Depth**: CVPixelBufferPool -> CVMetalTextureCache full pipeline, alignment requirements, Ingonyama 1GB array benchmark showing near-zero overhead (Findings #658-#662, #709)

## How to Query Knowledge

Use the portable KB CLI to search unified-memory findings:

```bash
# List all unified-memory findings
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill unified-memory

# Search specific topics
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "storage mode"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "SLC cache"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "zero-copy"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "bandwidth"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "page fault"

# Get detailed finding with sources
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <finding-id>

# Check statistics
${CLAUDE_PLUGIN_ROOT}/scripts/kb stats
```

The `search` command uses BM25-weighted FTS5 ranking, prioritizing claim text over evidence. The `skill` command returns all 104 unified-memory findings in table format.

## Common Patterns & Quick Answers

**Q: What storage mode should I use for CPU-GPU shared buffers on Apple Silicon?**
A: On Apple Silicon (UMA), both .shared and .private allocate from the same physical unified memory. Use .shared when CPU needs direct access (e.g., uploading vertex data, reading back results). Use .private for GPU-only resources to get automatic lossless bandwidth compression and optimal tiling. The .managed mode exists on macOS but is unnecessary on UMA — it behaves identically to .shared. (Findings #73-77)

**Q: How big is the SLC and how does it behave?**
A: SLC sizes: M1/M2/M4 base = 8 MB, M1/M2 Pro = 24 MB, M3 Pro = 12 MB (reduced), M3 Max = 48 MB, M4 Max = 128 MB, M5 base = 8 MB. Cache line = 128 bytes. SLC is exclusive wrt CPU caches (not duplicated in CPU L2) but inclusive wrt GPU caches. Pseudo-random replacement policy. SLC SRAM reads cost single-digit picojoules vs hundreds for DRAM reads. (Findings #84-88, #96-98, #122)

**Q: Do I need to flush caches when sharing memory between CPU and GPU?**
A: No. All memory is coherent between CPU and GPU on Apple Silicon without explicit cache flushes or invalidations. The hardware maintains coherency automatically through the SLC. Coherency latencies: L1/L2 ~27ns, SLC ~100ns, DRAM ~180ns. (Findings #54-55, #125)

**Q: How do I create a zero-copy Metal buffer from existing memory?**
A: Use makeBuffer(bytesNoCopy:length:options:deallocator:). The pointer must be page-aligned (16KB / 16384 bytes on ARM64) and length must be a multiple of page size. This wraps existing memory as an MTLBuffer without copying. For cross-process sharing, use IOSurface or MTLSharedTextureHandle. (Findings #89-92)

**Q: What is the maximum GPU memory allocation?**
A: GPU can use approximately 75% of total system unified memory (e.g., 36 GB on a 48 GB machine). Check device.recommendedMaxWorkingSetSize for the runtime limit. device.maxBufferLength returns maximum single buffer size. Exceeding limits triggers memory pressure: system compresses pages and may evict Metal allocations. (Findings #68, #105-109)

**Q: What memory bandwidth can I expect on Apple Silicon?**
A: STREAM benchmarks: M4 CPU = 103 GB/s, M4 GPU = 100 GB/s (theoretical 120 GB/s). M1 Pro measured 180/200 GB/s (90%), M2 Pro measured 202/200 GB/s (>100% due to compression). CPU and GPU achieve comparable bandwidth independently but share the same physical bandwidth. (Findings #101-103)

**Q: How does the CPU-GPU virtual address offset work?**
A: CPU and GPU virtual addresses for the same physical page differ by a constant offset (discovered by Philip Turner's metal-usm project). This enables pointer arithmetic across address spaces: CPU pointers embedded in GPU data structures can be translated by adding the offset. GPU uses UAT (Unified Address Translator) with ARM64-identical page tables. (Findings #56, #81, #126)

**Q: What happens during a GPU page fault?**
A: GPU page faults on Apple Silicon are handled as stop-the-world events by the GPU firmware — all GPU execution stalls until the fault is resolved. This makes page faults extremely expensive. Ensure all resources are properly made resident before GPU access. In Metal 4, residency sets are the exclusive mechanism for managing resource residency. (Finding #127)

**Q: How does Apple Fabric route memory traffic?**
A: Apple Fabric is the on-chip interconnect connecting CPU, GPU, Neural Engine, media engines, and memory controllers. It routes all memory traffic with QoS-based bandwidth allocation. Thread-level QoS properties guide scheduler decisions for bandwidth partitioning. (Findings #104, #115)

**Q: How does M5 Neural Accelerator access memory?**
A: M5 Neural Accelerators (one per GPU core) operate on data from unified memory via the SLC, same as regular GPU compute. No separate memory pool — they share the unified memory hierarchy with full coherency. (Finding #53)

**Q: What is the GPU memory ordering model on Apple Silicon?**
A: Apple GPUs use a weakly-ordered memory model with relaxed atomics only — no acquire/release/seq_cst ordering. CPU cores use TSO (Total Store Order) while GPU uses ARM-style weak ordering. The TSO vs weak ordering gap shows an 8.94% performance difference. No explicit didModifyRange is needed for shared storage on Apple Silicon UMA. (Findings #632, #633, #637)

**Q: What changed in Metal 4 memory management?**
A: Metal 4 introduces: (1) MTL4CommandAllocator decouples command memory from buffer lifecycle, (2) MTLResidencySet is now the EXCLUSIVE mechanism for GPU residency — setBuffer/setTexture no longer auto-retain, (3) Placement sparse resources for fine-grained memory control, (4) No automatic hazard tracking — explicit barriers required. (Findings #639-#643)

**Q: How does UAT page table translation work in detail?**
A: Apple GPU uses 4-level page tables (L0-L3) with TTBR0/TTBR1 split for user/kernel spaces. PTE format diverges from standard ARM64. The DART (Device Address Resolution Table) is the IOMMU for general device memory mapping, while UAT (Unified Address Translator) is GPU-specific with constant CPU-GPU virtual address offset. (Findings #653-#656)

**Q: What is the difference between DART and UAT?**
A: DART is Apple's IOMMU for general device address resolution across all I/O devices. UAT is the GPU-specific address translator with ARM64-identical page tables. DART provides system-wide address translation; UAT provides the GPU's own virtual address space with the constant offset from CPU virtual addresses that enables pointer sharing. (Finding #656)

**Q: Does hardware compression apply to compute buffers?**
A: No. Apple Silicon's lossless hardware compression applies to textures only (private storage mode), NOT to general compute buffers. This is a common misconception. The measured >100% bandwidth efficiency on some chips comes from texture compression, not buffer compression. macOS virtual memory compression is a separate OS-level mechanism. (Findings #651, #652)

## Cross-Skill Integration

**With gpu-silicon**: Unified memory's SLC is the bridge between CPU and GPU cache hierarchies. GPU core architecture (tile memory, register file) determines how on-chip memory interacts with the SLC and DRAM. Memory hierarchy understanding requires both skills.

**With metal-compute**: Storage mode selection directly affects command buffer synchronization requirements. Shared buffers with atomic operations enable CPU-GPU ring buffer patterns. Metal 4 residency sets change how resources are made GPU-resident.

**With gpu-io**: GPU I/O patterns (SSD-to-GPU streaming, mmap, Metal Fast Resource Loading) build on unified memory's zero-copy foundation. Understanding storage modes and page alignment is prerequisite for efficient I/O.

**With gpu-perf**: Memory bandwidth is often the primary bottleneck in GPU compute. Profiling memory access patterns, SLC hit rates, and bandwidth utilization requires understanding the unified memory hierarchy. Storage mode choice directly impacts achievable bandwidth (private mode enables compression).

**With msl-kernels**: Kernel memory access patterns (coalesced vs scattered, threadgroup memory usage) interact with the unified memory hierarchy. Understanding SLC behavior and cache line sizes informs optimal memory access patterns in MSL code.

## Advanced Topics

**Memory Compression**: Apple Silicon implements both lossless and lossy hardware memory compression. Lossless compression is automatically applied to private storage mode textures, improving effective bandwidth. macOS also uses virtual memory compression under memory pressure, compressing inactive pages before swapping to disk. (Findings #57, #108)

**Cross-Process Sharing**: IOSurface is the kernel-managed mechanism for cross-process GPU buffer sharing with residency tracking. MTLSharedTextureHandle provides a simpler alternative for texture sharing. Complete zero-copy pipeline: CVPixelBuffer backed by IOSurface can be wrapped as MTLTexture for GPU processing without any data copies. (Findings #91-93)

**DART and Address Translation**: Apple Silicon uses DART (Device Address Resolution Table) as its IOMMU for GPU memory mapping. The GPU's UAT uses ARM64-identical page tables with unusual set indexing in the SLC. This architecture enables the constant CPU-GPU virtual address offset that powers novel programming patterns like direct pointer sharing. (Findings #126, #130, #97)

**Academic Research on Unified Memory**: Recent academic work characterizes unified physical memory on AMD MI300A APU, showing applications using unified memory path achieved up to 6x higher bandwidth than discrete-like paths. GPU-driven virtual memory (GPUVM) research eliminates CPU/OS from the page fault critical path. GMLake's virtual memory stitching defragments GPU memory by fusing non-contiguous physical blocks into contiguous virtual ranges. (Findings #482-492, #505, #523, #525)

## Investigation Guidelines

When investigating unified memory topics:

1. **Check existing findings first**: `${CLAUDE_PLUGIN_ROOT}/scripts/kb search "<topic>"`
2. **Focus on Apple sources**: WWDC sessions, Apple documentation, developer forums
3. **Cross-reference with benchmarks**: Memory bandwidth and latency claims need empirical validation
4. **Note chip generation**: SLC sizes, bandwidth, and features vary significantly across M1-M5
5. **Consider both CPU and GPU perspectives**: Unified memory behavior differs based on which processor is accessing

Use `/gpu-forge:investigate unified-memory "<specific-topic>"` to research new areas and automatically store findings in the knowledge base.

## References

- Layer 0 references: See `${CLAUDE_PLUGIN_ROOT}/skills/unified-memory/references/` for exported finding lists by topic
- Apple Metal Programming Guide: Resource storage modes, memory management
- WWDC sessions: Discover Metal 4, Metal Best Practices, Optimize GPU Memory
- Philip Turner's metal-usm: CPU-GPU constant address offset discovery
- Philip Turner's metal-benchmarks: Bandwidth and latency measurements
- Chips and Cheese: SLC architecture analysis, memory controller deep dives

## Finding ID Reference

Key findings by topic area (investigation #22):
- **GPU memory model & ordering**: #632, #633, #634, #635, #637
- **Device-scope coherency**: #636
- **Synchronization primitives**: #638
- **Metal 4 memory management**: #639, #640, #641, #642, #643
- **SLC architecture depth**: #644, #645, #646, #647, #648, #649
- **Physical memory architecture**: #650
- **Hardware compression**: #651, #652
- **Virtual address translation (UAT)**: #653, #654, #655
- **DART vs UAT**: #656
- **GPU TLB**: #657
- **Zero-copy pipeline depth**: #658, #659, #660, #661, #662
- **Memory limits & pressure**: #663
- **Zero-copy transfer benchmark**: #709

Use `kb detail <id>` to retrieve full finding details including source URLs and confidence levels.

## Version History

- 2026-02-10: Updated with 28 new findings from investigation #22 (104 total)
- 2026-02-09: Initial skill creation with 76 findings from knowledge base
