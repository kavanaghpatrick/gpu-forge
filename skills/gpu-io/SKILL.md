---
name: gpu-io
layer: 2
description: >
  This skill should be used when the user asks about GPU I/O, Metal IO, MTLIOCommandQueue, mmap GPU access, fast resource loading, SSD-to-GPU streaming, makeBuffer bytesNoCopy, direct storage, GPU filesystem, BaM, SCADA, GPU-initiated I/O, sparse textures, virtual texture streaming, write-back patterns, network-to-GPU zero-copy, RDMA GPU data transfer, Apple SSD architecture, NVMe GPU access, or storage-compute integration on Apple Silicon.
related_skills:
  - unified-memory
  - metal-compute
  - gpu-perf
  - gpu-centric-arch
---

# GPU I/O and Storage Access

GPU I/O on Apple Silicon covers the full path from persistent storage to GPU compute buffers, including Metal Fast Resource Loading, memory-mapped file access, SSD architecture, streaming patterns, write-back paths, GPU-initiated I/O research, and network-to-GPU data transfer. Apple's unified memory architecture creates unique opportunities for zero-copy I/O that discrete GPU systems cannot match.

The storage-to-GPU pipeline on Apple Silicon differs fundamentally from discrete GPU architectures. There is no PCIe bus separating host and device memory. Instead, the GPU shares the same physical DRAM through a unified address space, enabling patterns like mmap-to-GPU zero-copy that bypass traditional DMA transfers entirely. Metal Fast Resource Loading (MTLIOCommandQueue) provides an optimized async path for bulk data streaming, while mmap combined with makeBuffer(bytesNoCopy:) enables demand-paged GPU access to file-backed memory.

This skill covers Metal I/O APIs, SSD hardware characteristics, streaming patterns for real-time workloads, write-back paths from GPU to storage, GPU-initiated I/O research from academia, network-to-GPU transfers via RDMA, and benchmark data on I/O performance. The knowledge base contains 38 verified and high-confidence findings from Apple documentation, WWDC sessions, academic papers, reverse engineering, and benchmark studies.

## Domain Overview

**Metal Fast Resource Loading**: MTLIOCommandQueue provides an async I/O pipeline purpose-built for streaming assets from SSD to GPU buffers and textures. Created via MTLIOCommandQueueDescriptor with three properties: maxCommandBufferCount, maxCommandsInFlight, and priority (normal/high/low). Metal I/O supports five compression methods: LZ4 (fastest decompression, ~1 byte overhead), LZFSE (Apple proprietary, good ratio), LZ4+Huffman, LZBITMAP (fast random access), and ZLIB (highest ratio but slowest). Metal Fast Resource Loading integrates with sparse textures for virtual texture streaming -- 16KB tile granularity with GPU-driven demand paging via access counters. Metal 4 adds placement sparse resources that allocate without backing memory, deferring physical allocation to demand time.

**Memory-Mapped GPU Access**: makeBuffer(bytesNoCopy:length:options:deallocator:) wraps existing contiguous memory as a Metal buffer without copying. Combining mmap(file) with makeBuffer(bytesNoCopy:) and .storageModeShared creates a zero-copy path from file to GPU compute. Apple GPU uses UAT (Unified Address Translator) -- essentially ARM64 MMU with identical page table format -- enabling GPU to directly walk CPU page tables. madvise() hints affect page cache for mmap'd GPU buffers: MADV_SEQUENTIAL enables readahead prefetching, MADV_WILLNEED triggers pre-fault. Philip Turner's metal-usm demonstrates CPU-GPU pointer translation via constant-offset virtual address space.

**SSD Architecture**: Apple's storage controller (AppleANS3NVMeController) is an ARM64 coprocessor running its own firmware, managing wear leveling, encryption, and garbage collection. SSD bandwidth scales with NAND chip count: M4 base (512GB, 2 NAND) ~2.9/3.3 GB/s read/write, M4 Pro (1TB, 4 NAND) ~4.5/3.8 GB/s, M4 Max (2-8TB) ~7.4 GB/s read. Every Apple Silicon Mac encrypts all SSD data at hardware level using AES-256-XTS with per-file keys. Apple SSDs use TLC 3D NAND with SLC write cache -- during burst writes, SLC cache absorbs at full speed; when exhausted, write speed drops to TLC direct-write rates.

**Streaming Patterns**: Triple buffering is the canonical Metal streaming pattern: 3 pre-allocated buffers rotated via semaphore, allowing CPU to prepare frame N+2 while GPU executes frame N. Metal sparse textures provide GPU-driven demand paging with 16KB tile granularity, memory mapped without backing, and access counters for feedback. For ML inference, vllm-mlx implements production continuous batching on Apple Silicon with dynamic buffer management. MLX arrays live in shared unified memory -- operations directed to CPU or GPU without explicit data transfers.

**Write-Back Patterns**: GPU compute results written to storage follow a CPU-mediated pattern: GPU writes to unified memory buffer, CPU issues write() or pwrite() syscall, kernel handles page cache and block layer. Apple Silicon's write-back path is uniquely efficient: GPU writes to unified memory are immediately visible to CPU without explicit synchronization beyond command buffer completion. For continuous GPU-compute-to-file output, triple-buffer pipeline: Frame N uses buffer A for GPU write, buffer B for CPU file write, buffer C for next GPU dispatch.

**GPU-Initiated I/O**: BaM (ASPLOS 2023) enables GPU threads to directly submit NVMe I/O by mapping SSD doorbell registers into GPU address space. BaM dramatically outperforms GPUDirect Storage for fine-grained (<32KB) random access patterns. AGILE (SC 2025) introduces asynchronous GPU-initiated NVMe I/O, solving BaM's synchronous stall problem. NVIDIA SCADA implements complete NVMe driver inside GPU. GPUfs (ASPLOS 2013) was first POSIX-like file API for GPU kernels. LithOS (SOSP 2025) is the first step toward a GPU OS with fine-grained scheduling. True GPU-initiated I/O on Apple Silicon would require mapping NVMe queue registers into GPU-accessible memory, which is not currently exposed.

**Network-to-GPU Transfer**: macOS Tahoe 26.2 enables RDMA over Thunderbolt 5: 80 Gbps max bandwidth, 5-9 microsecond latency. MLX provides 4 distributed backends: MPI, Ring (TCP sockets), JACCL (RDMA over Thunderbolt 5), and Gloo (cross-platform). Network-to-GPU zero-copy on Apple Silicon: receive data into page-aligned buffer then wrap with makeBuffer(bytesNoCopy:). RDMA cluster scaling: HPL single node 1.3 TFLOPS to 4-node 3.7 TFLOPS (~70% efficiency).

**I/O Benchmarks**: Memory-to-storage bandwidth gap on Apple Silicon is 15-70x: M1 60 vs ~3 GB/s (20x), M4 Max 546 vs ~7.4 GB/s (74x). No published head-to-head benchmark exists for MTLIOCommandQueue vs mmap vs standard read(). GPU DMA engine compute-communication overlap achieves only 21% of ideal speedup due to contention. Programmed I/O outperforms DMA on coherent interconnects like CXL 3.0. Fast page recycling eliminates TLB shootdown bottleneck in mmap. GPU zero-copy memory access combined with late materialization enables data analytics beyond memory capacity.

## Key Knowledge Areas

The knowledge base contains **38 verified and high-confidence findings** covering:

- **Metal Fast Resource Loading** (4 findings): MTLIOCommandQueue API, compression methods, sparse texture integration, Metal 4 placement resources
- **Memory-Mapped GPU Access** (5 findings): makeBuffer(bytesNoCopy:), mmap zero-copy path, UAT page table sharing, madvise hints, metal-usm pointer translation
- **SSD Architecture** (4 findings): AppleANS3 controller, bandwidth scaling with NAND count, hardware encryption, SLC write cache behavior
- **Streaming Patterns** (4 findings): Triple buffering, sparse texture demand paging, continuous batching (vllm-mlx), MLX unified memory arrays
- **Write-Back Patterns** (3 findings): CPU-mediated write-back, unified memory coherency, triple-buffer file output pipeline
- **GPU-Initiated I/O** (7 findings): BaM, AGILE, SCADA, GPUfs, LithOS, Apple Silicon feasibility analysis
- **Network I/O** (4 findings): RDMA over Thunderbolt 5, MLX distributed backends, zero-copy network receive, cluster scaling
- **I/O Benchmarks** (7 findings): Memory-storage bandwidth gap, DMA overlap efficiency, programmed I/O on coherent interconnects, TLB shootdown, zero-copy analytics

## How to Query

Use the knowledge base CLI to retrieve GPU I/O findings:

```bash
# Get all gpu-io findings
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill gpu-io

# Search by I/O topic
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "MTLIOCommandQueue"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "mmap GPU"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "SSD bandwidth"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "fast resource loading"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "GPU-initiated I/O"

# Get specific finding details
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <id>

# Search across all skills for I/O-related topics
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "zero-copy"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "streaming"
```

Query strategies by problem type:
- **Asset Streaming**: Search "MTLIOCommandQueue", "fast resource loading", "sparse textures", "compression"
- **File-to-GPU Access**: Search "mmap", "bytesNoCopy", "makeBuffer", "zero-copy"
- **Storage Performance**: Search "SSD bandwidth", "NAND", "write cache", "storage controller"
- **Write-Back**: Search "write-back", "GPU to file", "triple buffer"
- **GPU Filesystem**: Search "BaM", "GPUfs", "LithOS", "GPU-initiated I/O", "SCADA"
- **Network I/O**: Search "RDMA", "Thunderbolt", "distributed", "network GPU"
- **Benchmarks**: Search "bandwidth gap", "MTLIOCommandQueue benchmark", "DMA overlap"

## Common Patterns & Quick Answers

**Q: How do I stream assets from SSD to GPU on Apple Silicon?**
A: Use MTLIOCommandQueue for async bulk loading (#219). It supports five compression methods -- LZ4 is fastest for decompression (#220). For virtual textures, integrate with Metal sparse textures for GPU-driven demand paging at 16KB tile granularity (#221). Metal 4 adds placement sparse resources that defer physical allocation (#222).

**Q: How do I get zero-copy file access from GPU?**
A: Combine mmap(file) with makeBuffer(bytesNoCopy:length:options:deallocator:) using .storageModeShared (#223, #224). Apple GPU uses UAT with identical page table format to ARM64 MMU, so GPU can directly walk CPU page tables (#225). Use madvise(MADV_SEQUENTIAL) for readahead and madvise(MADV_WILLNEED) for pre-faulting (#226).

**Q: What are Apple SSD bandwidth numbers?**
A: SSD bandwidth scales with NAND chip count: M4 base ~2.9 GB/s read, M4 Pro ~4.5 GB/s, M4 Max ~7.4 GB/s (#229). The memory-to-storage bandwidth gap is 15-70x depending on chip (#250). Every Mac encrypts SSD data with AES-256-XTS at hardware level (#230). TLC NAND uses SLC write cache for burst writes (#231).

**Q: How do I write GPU results back to storage?**
A: GPU writes to unified memory buffer, then CPU issues write() syscall (#243). On Apple Silicon, GPU writes are immediately visible to CPU after command buffer completion -- no explicit transfer needed (#244). For continuous output, use triple-buffer pipeline: one buffer for GPU write, one for CPU file write, one for next GPU dispatch (#245).

**Q: What is GPU-initiated I/O and does it work on Apple Silicon?**
A: BaM (ASPLOS 2023) maps SSD doorbell registers into GPU address space for direct NVMe submission (#236). It outperforms GPUDirect Storage for fine-grained random access (#237). AGILE adds async GPU-initiated NVMe I/O (#238). On Apple Silicon, true GPU-initiated I/O would require mapping NVMe queue registers into GPU-accessible memory, which is not currently exposed (#242).

**Q: How do I stream data from network to GPU?**
A: macOS Tahoe 26.2 enables RDMA over Thunderbolt 5 at 80 Gbps with 5-9 microsecond latency (#246). MLX provides 4 distributed backends including JACCL for RDMA (#247). For zero-copy: receive into page-aligned buffer, wrap with makeBuffer(bytesNoCopy:) (#248). RDMA cluster scaling achieves ~70% efficiency at 4 nodes (#249).

**Q: What are the best streaming patterns for real-time GPU workloads?**
A: Triple buffering with semaphore rotation is canonical -- 3 buffers allow CPU/GPU overlap (#232). Sparse textures provide GPU-driven demand paging at 16KB tile granularity (#233). MLX arrays in unified memory avoid explicit transfers entirely (#235). For ML inference, vllm-mlx implements continuous batching with dynamic buffer management (#234).

**Q: What does the I/O performance landscape look like?**
A: GPU DMA compute-communication overlap achieves only 21% of ideal speedup (#493). On coherent interconnects, programmed I/O can outperform DMA (#512). Fast page recycling eliminates TLB shootdown bottleneck in mmap (#595). GPU zero-copy with late materialization enables analytics beyond memory capacity (#592). No head-to-head benchmark exists for MTLIOCommandQueue vs mmap vs read() (#251).

## Cross-References

**Related Skills**:
- **unified-memory** (Layer 0): UMA architecture, storage modes, SLC cache, CPU-GPU coherency -- foundational for understanding zero-copy I/O
- **metal-compute** (Layer 1): Metal API, command buffers, buffer allocation, compute pipelines -- the API layer for I/O integration
- **gpu-perf** (Layer 2): Bandwidth optimization, profiling tools, dispatch strategies -- diagnosing I/O bottlenecks
- **gpu-centric-arch** (Layer 4): GPU-centric OS, persistent kernels, GPU databases -- advanced I/O architectures like BaM and LithOS

**I/O Integration Workflow**:
1. Identify data source: SSD file, network stream, or in-memory
2. Choose access pattern: MTLIOCommandQueue (bulk async), mmap+bytesNoCopy (demand-paged), standard read (simple)
3. Configure buffering: triple buffer for streaming, sparse textures for virtual texturing
4. Optimize: madvise hints for mmap, compression for MTLIOCommandQueue, page-aligned buffers for network
5. Write-back: CPU-mediated via unified memory, triple-buffer for continuous output

**Common I/O Paths**:
- **Bulk asset loading** -> MTLIOCommandQueue with LZ4 compression (metal-compute for command encoding)
- **Large file compute** -> mmap + makeBuffer(bytesNoCopy:) (unified-memory for storage modes)
- **Real-time streaming** -> Triple buffer with semaphore (gpu-perf for dispatch optimization)
- **Network ingest** -> RDMA + page-aligned bytesNoCopy (gpu-distributed for cluster setup)
- **GPU filesystem** -> Research: BaM, LithOS patterns (gpu-centric-arch for OS-level design)

## Investigation Prompts

Suggested investigation topics to expand this skill:

- "MTLIOCommandQueue vs mmap vs read() benchmark on M4 Max"
- "Apple SSD firmware behavior under GPU-driven random 4KB reads"
- "Sparse texture access counter API and feedback-driven streaming"
- "Metal 4 placement sparse resources allocation patterns"
- "mmap page fault overhead for GPU-accessed file-backed buffers"
- "Network-to-GPU zero-copy with RDMA on Thunderbolt 5 clusters"
- "Apple ANS3 NVMe controller queue depth and command latency"
- "GPU-initiated I/O feasibility on Apple Silicon via IOKit"
- "Continuous GPU-to-disk write throughput with triple buffering"
- "CXL memory expansion for Apple Silicon GPU workloads"

## Notes

- **Layer 2 dependencies**: Requires unified-memory (storage modes) and metal-compute (API basics)
- **Apple-unique advantage**: Unified memory eliminates PCIe DMA transfers, enabling mmap-to-GPU zero-copy
- **Bandwidth gap**: Memory bandwidth outpaces storage by 15-70x -- I/O is often the true bottleneck
- **No GPU-initiated I/O on Apple**: NVMe queue registers not exposed to GPU address space (unlike BaM on NVIDIA)
- **Compression trade-offs**: LZ4 fastest decompression, ZLIB best ratio -- choose based on I/O vs compute bound
- **SSD variance**: Bandwidth scales with NAND count -- base models have significantly lower I/O than Pro/Max
- **RDMA availability**: Only macOS Tahoe 26.2+ with Thunderbolt 5 hardware
- **Benchmark gap**: No published MTLIOCommandQueue vs mmap comparison -- opportunity for original research
- **Write-back simplicity**: Unified memory makes GPU-to-file writes trivially simple vs discrete GPU architectures
- **Academic frontier**: BaM, AGILE, LithOS represent cutting-edge GPU I/O research not yet available on Apple
