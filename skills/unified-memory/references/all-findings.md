# unified-memory — All Findings (75)

## Finding 128: Chips and Cheese found that M1 "is unable to keep CPU to GPU transfers on-die" or experiences very high latency for small transfers. Despite sharing the SLC, cross-domain (CPU->GPU) cache transfers do not appear to hit the SLC efficiently for small working sets. This suggests the SLC may serve each domain independently rather than enabling direct CPU-GPU cache sharing.
**Confidence**: high
**Source**: Chips and Cheese - iGPU Cache Setups Compared
**Evidence**: Chips and Cheese iGPU cache setups article: "M1 is unable to keep CPU to GPU transfers on-die" and notes "very high latency" for small cross-domain transfers.
**Tags**: CPU-GPU-transfer,SLC,latency,cross-domain,cache

## Finding 525: On cache-coherent CPU-GPU systems (Grace Hopper C2C: 450 GB/s bidirectional), careful memory placement is critical — transparent access across unified address space hides significant performance differences based on where data physically resides (CPU vs GPU DRAM)
**Confidence**: high
**Source**: Understanding Data Movement in Tightly Coupled Heterogeneous Systems: A Case Study with the Grace Hopper Superchip
**Evidence**: Characterizes intra- and inter-node memory operations on GH200 at Swiss NSSC Alps supercomputer. The unified address space provides transparent access but physical location determines actual bandwidth and latency. Highlights tradeoffs between placement convenience and performance.
**Tags**: data-movement,memory-placement,grace-hopper,cache-coherent,numa-effects,bandwidth

## Finding 99: GPU per-core caches: L1 Data=8 KB, Instruction Cache=12 KB, Threadgroup/Shared Memory=~60 KB. L2 cache is shared across all GPU cores (varies by variant: 768KB-1.5MB). SLC serves as effective L3 for GPU.
**Confidence**: verified
**Source**: Philip Turner - metal-benchmarks
**Evidence**: Philip Turner metal-benchmarks: L1D=8KB, IC=12KB, threadgroup memory ~60KB per core. L2 varies: M1=768KB, M2=~1.5MB. Register file ~208KB per core. Chips and Cheese confirms L1=8KB, L2=1MB shared.
**Tags**: gpu-cache,l1,l2,slc,threadgroup-memory,register-file

## Finding 100: GPU SLC bandwidth: ~15.4-19.8 bytes/cycle per core (M1 variants). GPU RAM bandwidth: ~7.7-9.9 bytes/cycle per core. On-core data bandwidth: 64 bytes/cycle. On-GPU data bandwidth: ~32 bytes/cycle.
**Confidence**: high
**Source**: Philip Turner - metal-benchmarks
**Evidence**: Philip Turner metal-benchmarks provides per-core bandwidth measurements across cache levels. These show ~2x SLC bandwidth vs DRAM, consistent with SLC being a significant bandwidth amplifier.
**Tags**: gpu-bandwidth,slc-bandwidth,dram-bandwidth,per-core

## Finding 505: GMLake introduces Virtual Memory Stitching (VMS) -- fusing non-contiguous physical memory blocks into contiguous virtual address ranges via GPU virtual memory page table manipulation. Reduces GPU memory usage by avg 9.2GB (up to 25GB) and fragmentation by 15-33% across 8 LLM models on A100. Operates transparently without modifying DL frameworks. This is the GPU equivalent of OS-level memory compaction.
**Confidence**: high
**Source**: GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching
**Evidence**: ASPLOS 2024 paper. Uses low-level CUDA virtual memory management APIs (cuMemCreate, cuMemMap, cuMemAddressReserve) to stitch non-contiguous physical allocations into contiguous virtual ranges. Key insight: GPU caching allocators (PyTorch) create severe fragmentation because they never return memory to the OS allocator.
**Tags**: memory-defragmentation,virtual-memory-stitching,page-table,memory-allocator,ASPLOS

## Finding 523: Virtual memory stitching (fusing non-contiguous physical memory blocks into contiguous virtual address ranges) reduces GPU memory fragmentation by 15-33% and saves average 9.2 GB (up to 25 GB) on A100 GPUs during LLM training
**Confidence**: verified
**Source**: GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching
**Evidence**: GMLake leverages low-level GPU virtual memory management APIs to combine non-contiguous memory blocks via virtual address mapping. The VMS mechanism avoids physical memory compaction/copying. Transparent to existing models. Tested on 8 LLMs. Open-sourced.
**Tags**: memory-fragmentation,virtual-memory,defragmentation,llm-training,memory-allocator

## Finding 482: GPU threads can manage their own page faults and memory migration without CPU/OS involvement, achieving up to 4x performance over traditional UVM for latency-bound applications
**Confidence**: high
**Source**: GPUVM: GPU-driven Unified Virtual Memory
**Evidence**: GPUVM enables on-demand paging entirely on the GPU side using RDMA-capable network devices. By removing CPU from the page fault handling path, it eliminates the round-trip latency of GPU->CPU interrupt->OS handler->migration->GPU resume. This is directly relevant to Apple Silicon where the GPU and CPU share the same physical memory — a GPU-driven page fault handler could be implemented entirely in tile memory or threadgroup memory, avoiding the OS overhead entirely.
**Tags**: unified-memory,page-faults,gpu-driven,virtual-memory,zero-copy

## Finding 488: GPUVM eliminates CPU/OS from the virtual memory critical path by using RDMA-capable network devices to construct GPU-driven page management. GPU threads themselves handle memory management and page migration via on-demand paging, achieving up to 4x performance over traditional UVM for latency-bound applications. This GPU-driven approach removes CPU bottleneck from address translation entirely.
**Confidence**: high
**Source**: GPUVM: GPU-driven Unified Virtual Memory
**Evidence**: Proposes GPU-driven unified virtual memory where GPU threads perform their own memory management. Uses RDMA for direct GPU-to-memory communication without CPU intervention. Evaluated against NVIDIA UVM baseline showing 4x improvement for latency-bound workloads.
**Tags**: virtual-memory,gpu-driven,RDMA,page-migration,address-translation,memory-management

## Finding 130: Apple Silicon uses DART (Device Address Resolution Table) as its IOMMU for GPU memory protection. DART supports 16KB minimum page size (matching ARM64 pages), provides per-device address space isolation, and is configured by the GPU firmware (ASC). The GPU UAT page tables are managed through DART, which translates GPU virtual addresses to physical addresses. M1 DART has been reverse-engineered by Asahi Linux team.
**Confidence**: high
**Source**: Asahi Linux - Apple GPU (AGX) Documentation
**Evidence**: Asahi Linux documentation on DART IOMMU and AGX GPU subsystem. Confirmed 16KiB minimum page size.
**Tags**: DART,IOMMU,memory-protection,address-translation,16KB

## Finding 91: IOSurface is a kernel-managed image buffer with GPU residency tracking that enables zero-copy sharing across processes and frameworks. To share: (1) create IOSurface with appropriate parameters, (2) pass via IOSurfaceCreateMachPort() or IOSurfaceCreateXPCObject(), (3) receiving process creates Metal texture via device.makeTexture(descriptor:iosurface:plane:). No data is copied when sharing across processes. IOSurface also enables zero-copy bridging between Core Image, Metal, Core Video, and vImage. Clean up mach ports with mach_port_deallocate() on both sides.
**Confidence**: verified
**Source**: Apple Developer: IOSurface Framework
**Evidence**: Apple Developer documentation for IOSurface framework. Russ Bishop blog "Cross-process Rendering" details the workflow. Apple Developer Forums thread confirms makeTexture(descriptor:iosurface:plane:) for Metal texture creation from IOSurface.
**Tags**: IOSurface,cross-process,zero-copy,mach-port,XPC

## Finding 92: MTLSharedTextureHandle provides a simpler alternative to IOSurface for cross-process Metal texture sharing via XPC/NSXPCConnection. Usage: (1) create shared texture with device.makeSharedTexture(descriptor:), (2) get handle via texture.makeSharedTextureHandle(), (3) send handle over XPC connection, (4) receiver calls device.makeSharedTexture(handle:) on the SAME GPU device that created the original. Available since macOS 10.14, now also on iOS/tvOS. Useful for XPC render services where rendering happens in a helper process.
**Confidence**: verified
**Source**: Apple Developer: MTLSharedTextureHandle
**Evidence**: Metal by Example "What's New in Metal 2019" describes MTLSharedTextureHandle as enabling texture sharing across processes via XPC connections. Apple Developer documentation for makeSharedTexture(handle:) and makeSharedTexture(descriptor:) confirm the API.
**Tags**: shared-texture,cross-process,XPC,zero-copy

## Finding 484: On Grace Hopper with integrated CPU-GPU system page table and cache-coherent NVLink-C2C, system-allocated memory matches or outperforms explicit GPU memory management for most HPC use cases while dramatically simplifying code
**Confidence**: high
**Source**: Harnessing Integrated CPU-GPU System Memory for HPC: a first look into Grace Hopper
**Evidence**: First comprehensive characterization of unified page table on Grace Hopper. Tested 6 applications including quantum computing simulator. System-allocated memory benefits most use cases with minimal porting. First-touch behavior, page table entry setup, and migration patterns documented.
**Tags**: unified-memory,system-page-table,grace-hopper,cache-coherent,first-touch

## Finding 65: M5 cache hierarchy: P-cores 192KB I-cache + 128KB D-cache L1, E-cores 128KB I-cache + 64KB D-cache L1. P-core cluster 64MB L2, E-core cluster 6MB L2. Each core cluster has 8MB SLC (significant increase). Memory bandwidth: 153.6 GB/s (base M5), 205 GB/s reported by some sources. M5 shares structural similarity with A19 Pro, adding core counts and increased bandwidth.
**Confidence**: medium
**Source**: Creative Strategies - M5: Cache and Tensors
**Evidence**: Multiple specification sources. Creative Strategies analysis. 9to5Mac confirms 30% bandwidth increase over M4.
**Tags**: m5,cache,l1,l2,slc,bandwidth,specifications

## Finding 84: SLC sizes for M1 family: M1=8 MB, M1 Pro=24 MB, M1 Max=48 MB, M1 Ultra=96 MB. SLC scales roughly 3x from base to Pro, 2x from Pro to Max, 2x from Max to Ultra.
**Confidence**: high
**Source**: Wikipedia - Apple M1
**Evidence**: Wikipedia M1 page and NamuWiki comprehensive table confirm: M1=8MB, M1 Pro=24MB, M1 Max=48MB. M1 Ultra=96MB (2x M1 Max via UltraFusion). Multiple corroborating sources including AnandTech review.
**Tags**: slc,m1,cache-hierarchy,system-level-cache

## Finding 85: SLC sizes for M2 family: M2=8 MB, M2 Pro=24 MB, M2 Max=48 MB, M2 Ultra=96 MB. Identical SLC sizing to M1 family.
**Confidence**: high
**Source**: NamuWiki - Apple Silicon M Series
**Evidence**: NamuWiki comprehensive table confirms identical SLC sizes to M1 generation: M2=8MB, M2 Pro=24MB, M2 Max=48MB, M2 Ultra=96MB.
**Tags**: slc,m2,cache-hierarchy,system-level-cache

## Finding 86: SLC sizes for M3 family: M3=8 MB, M3 Pro=12 MB (reduced from 24MB in M2 Pro), M3 Max=48 MB. M3 Ultra=96 MB (likely, via UltraFusion 2x Max).
**Confidence**: high
**Source**: NamuWiki - Apple Silicon M Series
**Evidence**: NamuWiki table shows M3=8MB, M3 Pro=12MB, M3 Max=48MB. RealWorldTech forum tests show M3 Max with latency plateau from 32-96 MB consistent with SLC. M3 Pro reduction from 24MB to 12MB mirrors its reduced memory bus (256-bit to 192-bit).
**Tags**: slc,m3,cache-hierarchy,system-level-cache

## Finding 87: SLC size for M4 base: 8 MB confirmed. M4 Pro and M4 Max SLC sizes not officially published by Apple. RealWorldTech forum projects M4 Max at ~96 MiB SLC based on die analysis, but this is speculative.
**Confidence**: medium
**Source**: Low End Mac - M4 chip specs + RealWorldTech Forum
**Evidence**: Low End Mac and multiple spec databases confirm M4 base at 8 MB SLC. M4 Pro shows "- MB SLC" (unpublished). RealWorldTech forum: "Dividing the 96 MiB SLC into 48 banks" for projected M4 Max architecture, but this is a proposal/estimate, not measurement.
**Tags**: slc,m4,cache-hierarchy,system-level-cache

## Finding 88: M5 base SLC: 8 MB (same as M4 base). However, M5 increased L2 cache per CPU cluster significantly: P-core L2=64 MB (vs M4 16 MB), E-core L2=6 MB (vs M4 4 MB). Creative Strategies confirms "more SLC among the entire SoC" in M5.
**Confidence**: high
**Source**: Low End Mac - M5 MacBook Pro specs
**Evidence**: Low End Mac M5 specs: 8MB SLC for base M5. Multiple sources confirm L2 P-core = 64 MB, L2 E-core = 6 MB. Creative Strategies article: M5 has "more L2 cache per CPU cluster, more SLC among the entire SoC" vs M4.
**Tags**: slc,m5,cache-hierarchy,system-level-cache,l2-cache

## Finding 96: Apple SLC is EXCLUSIVE with respect to CPU caches but INCLUSIVE with respect to GPU caches. Data evicted from CPU L2 spills into SLC. GPU shares direct SLC access. This is a unique hybrid inclusiveness policy.
**Confidence**: verified
**Source**: EXAM: Exploiting Exclusive SLC in Apple M-Series (Xu et al., ACM ASIACCS 2025)
**Evidence**: EXAM paper (arXiv:2504.13385): "exclusive with respect to the CPU cache" but "inclusive with respect to the GPU cache." CPU L2 evictions are "spilled over to the SLC." This differs from Intel inclusive LLC design.
**Tags**: slc,exclusive-cache,inclusive-cache,gpu-cache,cpu-cache,cache-policy

## Finding 97: SLC cache line size is 128 bytes (same as L2). SLC uses unusual set indexing: excludes lowest 13 bits of physical address, uses bits from 14th position and above. This differs from typical cache configurations.
**Confidence**: verified
**Source**: EXAM: Exploiting Exclusive SLC in Apple M-Series (Xu et al., ACM ASIACCS 2025)
**Evidence**: EXAM paper: "system-level cache line size is 128 bytes, the same as L2 caches" and "excludes the lowest 13 bits of the physical address for indexing and uses bits from the 14th position and above."
**Tags**: slc,cache-line,set-indexing,128-bytes

## Finding 98: SLC uses a pseudo-random replacement policy. The replacement is independent of access order, unlike typical LRU or pseudo-LRU policies used in CPU caches.
**Confidence**: verified
**Source**: EXAM: Exploiting Exclusive SLC in Apple M-Series (Xu et al., ACM ASIACCS 2025)
**Evidence**: EXAM paper: "The SLC replacement policy is independent of access order, indicating a pseudo-random policy."
**Tags**: slc,replacement-policy,pseudo-random,cache-policy

## Finding 122: SRAM reads (SLC) consume "generally single digit pJ" vs DRAM reads at "hundreds of pJ." This 10-100x energy advantage is why Apple invests heavily in SLC: each SLC hit saves significant power vs DRAM access.
**Confidence**: high
**Source**: Creative Strategies - M5 Apple Silicon: Its All About the Cache And Tensors
**Evidence**: Creative Strategies M5 analysis: SRAM reads "generally single digit pJ" versus DRAM reads "hundreds of pJ." Apple deliberately chose larger SLC over smaller die size, "valuing efficiency above cost."
**Tags**: slc,energy-efficiency,sram,dram,power

## Finding 111: On Apple Silicon, tile memory (used for render targets during fragment shading) serves as threadgroup memory during compute kernel execution. Same physical on-chip SRAM serves dual purpose. Tile memory allocated via MTLStorageModeMemoryless is ephemeral per pass.
**Confidence**: verified
**Source**: Apple Developer Forums - GPU Hardware and Metal concerning Tile Memory
**Evidence**: Apple developer forums (thread 668073) and WWDC sessions confirm: "On M1, tiles which are used to store render target data during fragment shader executions, are used as threadgroup memory when a compute kernel executes." Metal tech talk: tile shading operates directly on imageblock and stores results to threadgroup memory backed by tile memory.
**Tags**: tile-memory,threadgroup-memory,compute-kernel,tbdr

## Finding 112: SLC serves as an effective L3 cache for the GPU. Unlike NVIDIA GPUs which have no equivalent layer between L2 and DRAM, Apple adds SLC between GPU L2 and main memory. Because SLC is inclusive of GPU caches, GPU data benefits from SLC persistence.
**Confidence**: high
**Source**: Apple Silicon Metal vs NVIDIA CUDA (Shashank Shekhar)
**Evidence**: Shashank Shekhar blog: "SLC represents a substantial layer between L2 and main memory - a layer absent in NVIDIA GPUs." EXAM paper confirms SLC is inclusive of GPU caches. Philip Turner: "massive SLC, memory bandwidth, and incredibly low latency of the L2 makes up for less data cache."
**Tags**: slc,gpu-l3,nvidia-comparison,cache-hierarchy

## Finding 113: SLC bandwidth is approximately 2x DRAM bandwidth per GPU core (15.4-19.8 B/cycle SLC vs 7.7-9.9 B/cycle DRAM on M1). For compute workloads with working set <SLC size (8-96MB), SLC provides significant bandwidth amplification over DRAM-limited performance.
**Confidence**: high
**Source**: Philip Turner - metal-benchmarks
**Evidence**: Philip Turner metal-benchmarks measured per-core bandwidth ratios. The 2x amplification means compute kernels with good locality (working set fits in SLC) effectively see 2x the memory bandwidth vs kernels that miss SLC entirely.
**Tags**: slc-hit-rate,bandwidth-amplification,compute-workloads

## Finding 114: Threadgroup memory (backed by tile SRAM, ~60KB per core) is purely on-chip and does NOT go through the SLC or DRAM path. It has the lowest latency of any GPU memory. Device memory reads go through L1->L2->SLC->DRAM path.
**Confidence**: verified
**Source**: WWDC20: Optimize Metal Performance for Apple silicon
**Evidence**: WWDC20 session 10632 (Optimize Metal Performance for Apple silicon): copying device memory to threadgroup memory as "software cache" can be SLOWER on Apple Silicon due to unified memory and effective L2/SLC. Direct device buffer reads may be faster because L2+SLC already cache hot data.
**Tags**: threadgroup-memory,tile-sram,device-memory,slc-bypass

## Finding 483: Aggressive prefetching in GPU shared virtual memory causes excessive thrashing and performance degradation when memory is oversubscribed; fine-grained access pattern analysis reveals the prefetcher migrates far more data than needed
**Confidence**: high
**Source**: Shared Virtual Memory: Its Design and Performance Implications for Diverse Applications
**Evidence**: First comprehensive study of AMD GPU SVM technologies. Discovered that SVM employs aggressive prefetching that works well when GPU memory is not oversubscribed but causes severe thrashing otherwise. Proposes algorithm-level and hardware modifications to mitigate.
**Tags**: unified-memory,prefetching,thrashing,oversubscription,svm

## Finding 597: Selectively compressing read-only GPU pages (Split Linearly Compressed Pages scheme) provides 53-60% performance improvement under 125-150% memory oversubscription by increasing effective memory size without costly page remapping
**Confidence**: high
**Source**: Selective Memory Compression for GPU Memory Oversubscription Management
**Evidence**: SMC selectively compresses read-only pages using SLCP (Split Linearly Compressed Pages) line packing scheme. SLCP minimizes unused space, improves compressibility, and reduces extra memory accesses for fetching compressed data. Only compresses read-only pages to avoid write-back complexity.
**Tags**: memory-compression,oversubscription,read-only-pages,gpu-memory,compression

## Finding 492: On AMD MI300A APU with unified physical memory, applications using the unified memory model can match or outperform explicitly managed memory while reducing memory costs by up to 44%
**Confidence**: high
**Source**: Dissecting CPU-GPU Unified Physical Memory on AMD MI300A APUs
**Evidence**: First comprehensive characterization of UPM architecture on MI300A. Examined latency, bandwidth, cache management, and developed porting strategies. Key finding: the unified memory model eliminates redundant copies and reduces memory footprint dramatically. 6 applications tested with detailed performance analysis.
**Tags**: unified-memory,upm,mi300a,apu,memory-cost-reduction

## Finding 104: Apple uses QoS (Quality of Service) properties at the thread level to guide scheduling decisions between P-cores and E-cores. QoS semantics help the scheduler decide when/where to execute tasks. For GPU bandwidth, macOS allocates dynamically based on demand - no fixed partitioning.
**Confidence**: high
**Source**: WWDC20: Explore the new system architecture of Apple silicon Macs
**Evidence**: WWDC20 session 10686: "Set QoS on all work items - QoS properties are an indication to macOS of how work should be prioritized." Grand Central Dispatch uses QoS to route to appropriate cores. Memory bandwidth arbitration mechanism is not publicly documented.
**Tags**: qos,quality-of-service,scheduling,bandwidth-arbitration

## Finding 101: STREAM benchmark measured bandwidth: M4 CPU=103 GB/s, M4 GPU=100 GB/s (theoretical peak 120 GB/s, ~85% utilization). M3 CPU=92 GB/s, GPU=92 GB/s. M2 CPU=78 GB/s, GPU=91 GB/s. M1 CPU=59 GB/s, GPU=60 GB/s.
**Confidence**: verified
**Source**: Apple vs. Oranges: Evaluating Apple Silicon M-Series for HPC (arXiv:2502.05317)
**Evidence**: arXiv:2502.05317 (Apple vs. Oranges: HPC on Apple Silicon): STREAM benchmark measurements showing "All chips achieve approximately 85% of theoretical peak bandwidth." M4 approaches 120 GB/s theoretical peak.
**Tags**: stream-benchmark,bandwidth,m4,m3,m2,m1,hpc

## Finding 102: On base Apple Silicon chips, CPU and GPU achieve comparable memory bandwidth individually (M4: CPU=103 GB/s vs GPU=100 GB/s). There is no fixed partitioning - bandwidth is dynamically shared through the fabric.
**Confidence**: high
**Source**: Apple vs. Oranges (arXiv:2502.05317) + HN Discussion
**Evidence**: STREAM benchmark data (arXiv:2502.05317) shows near-identical CPU and GPU bandwidth on base chips. HN discussion: "Apple architecture uses true unified memory where CPU and GPU access the same physical DRAM pool." No evidence of static bandwidth partitioning.
**Tags**: bandwidth-partitioning,cpu-gpu,dynamic-allocation,fabric

## Finding 103: M1 Max achieved only ~50% of theoretical CPU bandwidth in real workloads (AnandTech benchmarks). GPU utilization even lower in 3D applications. Real bandwidth is significantly below theoretical peak in complex workloads.
**Confidence**: medium
**Source**: Hacker News - M4 Max memory discussion
**Evidence**: HN discussion citing AnandTech: "benchmarks showed the M1 Max could only achieve approximately 50% of theoretical CPU bandwidth in real workloads, with GPU utilization even lower in 3D applications."
**Tags**: bandwidth-utilization,real-world,m1-max

## Finding 54: Apple Silicon SLC uses HYBRID inclusiveness policy: inclusive wrt GPU cache but exclusive wrt CPU cache. When GPU accesses data first, subsequent CPU access hits SLC. CPU evictions do NOT necessarily remain in SLC. SLC uses 128-byte cache lines, indexes from physical address bit 14+, pseudo-random replacement. Hardware coherency ensures GPU reads shared data directly from CPU caches; GPU writes automatically invalidate corresponding CPU cache lines. No explicit cache flushing ever required.
**Confidence**: verified
**Source**: EXAM: Exploiting Exclusive SLC in Apple M-Series SoCs
**Evidence**: EXAM paper (ACM AsiaCCS 2025) reverse-engineered SLC: hybrid inclusive/exclusive policy confirmed via systematic experiments. Asahi Linux team: "We haven't used a single cache management instruction yet and everything still works."
**Tags**: slc,cache-coherency,inclusive,exclusive,hybrid,cpu-gpu

## Finding 123: M4 CPU cache hierarchy: P-core L1I=192KB, L1D=128KB (per core); P-core shared L2=16MB; E-core L1I=128KB, L1D=64KB (per core); E-core shared L2=4MB; SLC=8MB. M4 GPU: L1D=8KB, IC=12KB (per core), shared L2=~1-1.5MB, SLC=8MB shared with CPU.
**Confidence**: high
**Source**: Low End Mac M4 specs + Philip Turner metal-benchmarks
**Evidence**: Low End Mac M4 specs confirm CPU caches. Philip Turner metal-benchmarks confirms GPU L1/L2/IC sizes. SLC 8MB confirmed across multiple sources.
**Tags**: cache-hierarchy,m4,l1,l2,slc,complete-picture

## Finding 124: M5 CPU cache hierarchy: P-core cluster L2=64 MB (massive 4x increase from M4 16 MB), E-core cluster L2=6 MB (1.5x increase from M4 4 MB), SLC=8 MB base. Total on-chip cache dramatically increased.
**Confidence**: high
**Source**: Low End Mac - M5 MacBook Pro specs
**Evidence**: Low End Mac M5 specs and multiple sources confirm: P-core L2 jumped to 64MB (from M4 16MB), E-core L2 to 6MB (from M4 4MB). Creative Strategies confirms "more L2 per cluster, more SLC throughout the SoC."
**Tags**: cache-hierarchy,m5,l2,slc,p-core,e-core

## Finding 55: Measured CPU-GPU shared memory latencies on Apple Silicon: L1/L2 hit ~27ns (~80 GPU ticks), SLC hit ~234ns (~220 CPU ticks), DRAM miss ~342+ns (~430 ticks). SLC hit latency represents the coherency cost when data moves between CPU and GPU domains. Global memory atomic operations: ~58.57ns on M2 Pro. M5 memory bandwidth increased to 153.6 GB/s (30% over M4). RAM access can be as low as 8.8ns for cached hits.
**Confidence**: high
**Source**: Chips and Cheese - iGPU Cache Setups Including M1
**Evidence**: Chips and Cheese M2 Pro benchmarks for GPU latencies. EXAM paper for SLC characterization. Apple newsroom for M5 bandwidth.
**Tags**: latency,coherency,slc,bandwidth,nanoseconds,cache-hierarchy

## Finding 131: Metal Shading Language provides [[coherent(device)]] attribute for buffer/texture access in shaders to guarantee coherency across SIMD groups/threadgroups within a device. This ensures writes from one threadgroup are visible to others without additional barriers. Available since Metal 3.2 (Apple Family 9+). Default buffer access IS coherent at device scope on Apple Silicon due to hardware coherency, but explicit [[coherent(device)]] makes the intent clear and may prevent compiler reordering optimizations.
**Confidence**: medium
**Source**: Metal Shading Language Specification
**Evidence**: Apple Metal Shading Language Specification v3.2, memory model section. Apple Silicon hardware coherency means the attribute primarily affects compiler behavior rather than generating cache flushes.
**Tags**: coherent-device,MSL,buffer-attribute,memory-model,Metal3.2

## Finding 115: Apple Fabric is the on-chip interconnect connecting CPU, GPU, Neural Engine, media engines, and memory controllers. It contains ~12 ARM cores running their own firmware (resembling cut-down E-cores). Fabric handles memory arbitration, coherence, and I/O routing.
**Confidence**: medium
**Source**: Eclectic Light Company - Explainer: chipsets and Fabric
**Evidence**: Eclectic Light Company: Fabric "likely contains as many as a dozen ARM cores running their own firmware" resembling "a cut-down version of the Efficiency (Icestorm) cores." Fabric replaces traditional northbridge/southbridge chipset functions. Supports firmware updates via macOS.
**Tags**: apple-fabric,interconnect,memory-arbiter,coherence

## Finding 93: Complete zero-copy pipeline: CVPixelBuffer backed by IOSurface can be wrapped as Metal textures via CVMetalTextureCacheCreateTextureFromImage(). Steps: (1) create CVPixelBufferPool with IOSurface backing, (2) get pixel buffer from pool, (3) create CVMetalTextureCache, (4) create Metal texture per plane (e.g., R8Unorm for luma, RG8Unorm for chroma in biplanar YCbCr), (5) render into Metal textures which updates underlying IOSurface, (6) pass same CVPixelBuffer to Media Engine/VideoToolbox for zero-copy encoding. The CVPixelBuffer, Metal textures, and media engine all share the same physical memory through IOSurface.
**Confidence**: verified
**Source**: WWDC21: Create image processing apps powered by Apple silicon
**Evidence**: WWDC21 session 10153 provides complete code example of zero-copy HEVC encoding pipeline using CVPixelBufferPool with IOSurface backing, CVMetalTextureCacheCreateTextureFromImage for texture creation, and direct handoff to media engine. States: "CVPixelBuffer and Metal textures are backed by the same IOSurface in system memory."
**Tags**: CVPixelBuffer,CVMetalTextureCache,IOSurface,zero-copy,VideoToolbox,interop

## Finding 94: Core Image can render directly into Metal textures via CIContext.render(_:to:commandBuffer:bounds:colorSpace:), and CIImage can be initialized from MTLTexture via CIImage(mtlTexture:options:). For zero-copy: use IOSurface-backed textures so Core Image and Metal share the same memory. CVMetalTextureCache enables creating Metal textures from CVPixelBuffers without copying. CoreGraphics can share backing memory with Metal by creating a CGContext backed by the same memory as an MTLBuffer (both referencing the same vm_allocate'd page-aligned memory).
**Confidence**: high
**Source**: Apple Developer: CIImage init(mtlTexture:)
**Evidence**: Apple Developer documentation for CIImage.init(mtlTexture:options:) and CIContext render methods. WWDC21 session 10153 demonstrates IOSurface-backed pipeline. MetalPetal open-source project demonstrates CVMetalTextureCache and IOSurface patterns for zero-copy processing.
**Tags**: Core-Image,Metal,interop,zero-copy,CIContext,CIImage

## Finding 57: Apple Silicon implements BOTH lossless and lossy hardware memory compression. Lossless compression introduced with A12 Bionic (2018), transparent to software. Lossy compression added in A15 Bionic: provides 50% memory savings for textures with minimal visual loss, enabled via MTLTextureCompressionType.lossy on MTLTextureDescriptor. GPU memory usage is data-dependent (zero-heavy data compresses better). For compute workloads, lossless compression provides effective bandwidth amplification — actual useful bandwidth exceeds raw DRAM bandwidth for compressible data.
**Confidence**: high
**Source**: Metal by Example - Understanding Metal Enhancements in A15
**Evidence**: Metal by Example (A15 article) documents lossy compression API. EXAM paper demonstrates data-dependent GPU memory usage (white vs black iframes). A12/A14/A15 progression documented across multiple Apple sources.
**Tags**: compression,lossless,lossy,bandwidth-amplification,texture,hardware

## Finding 78: MTLHazardTrackingModeUntracked disables Metal automatic dependency tracking for a resource, placing responsibility on the developer to prevent read/write hazards manually using MTLFence, MTLEvent, MTLSharedEvent, or memory barriers. Default for heap-allocated resources is untracked. The main benefit is eliminating "false sharing" where tracked resources on a heap cause GPU work to serialize even when sub-resources are independent. Apple engineer warning: "this API requires care as all tracking is done at the granularity of the heap." MTLFence is the lowest-overhead synchronization primitive for single-queue scenarios.
**Confidence**: verified
**Source**: WWDC22: Go bindless with Metal 3
**Evidence**: WWDC22 "Go bindless with Metal 3" session details: When heap resources are tracked, Metal conservatively schedules work, causing false sharing that "reduces GPU parallelism and increases execution time." Untracked mode with manual fences provides fine-grained control. MTLFence for single queue, MTLEvent for multi-queue, MTLSharedEvent for cross-device/CPU. Memory barriers within a pass should AVOID post-fragment stage on Apple GPUs (very expensive). Metal by Example 2019 notes Apple engineer warning about heap-granularity tracking.
**Tags**: hazard-tracking,untracked,MTLFence,MTLEvent,synchronization,performance

## Finding 105: GPU can use up to approximately 75% of total system unified memory (e.g., 36 GB from a 48 GB system). This is a macOS-imposed limit, not a hardware constraint.
**Confidence**: medium
**Source**: MacRumors Forums - M4 Pro vs M4 Max GPU memory
**Evidence**: Community reports and MacRumors forum discussion: "GPU can only take up to 75% of system RAM, so it would take 36GB from a 48GB system." This is controlled by macOS memory management.
**Tags**: gpu-memory,allocation-limit,75-percent,macos

## Finding 110: On Apple Silicon, unified memory serves as both system RAM and GPU "VRAM" simultaneously. Frame buffers, shaders, textures, and render targets all occupy system RAM, meaning "RAM is pulling double-duty as RAM and VRAM." This enables much larger GPU working sets than traditional discrete GPUs (e.g., Mac Studio with 192GB unified memory provides far more usable GPU memory than any consumer discrete GPU). However, lower-memory Macs experience greater pressure from GPU workloads competing with system needs. The CPU and GPU share the same physical memory controllers and bandwidth.
**Confidence**: high
**Source**: macOS Memory Management - Gregg Ant
**Evidence**: Gregg Ant macOS memory management blog: "Frame buffers, shaders, textures, and render targets occupy RAM, thus the RAM is now pulling double-duty as the RAM and VRAM." Also notes Mac Studio 192GB as example of massive GPU-accessible memory. XDA Developers article on unified memory architecture confirms shared bandwidth.
**Tags**: unified-memory,VRAM,GPU-memory,shared-bandwidth,Mac-Studio

## Finding 125: All memory is coherent between CPU and GPU on Apple Silicon without explicit cache flushing. Asahi Linux team confirms: "We have not used a single cache management instruction yet and everything still works." No PCIe bus overhead - CPU/GPU access same physical DRAM.
**Confidence**: verified
**Source**: Asahi Linux - Tales of the M1 GPU
**Evidence**: Asahi Linux blog (Tales of the M1 GPU), WWDC20 session 10686 confirms unified memory with no copy overhead. The coherence is maintained by hardware in the fabric.
**Tags**: coherence,cache-flush,unified-memory,zero-copy

## Finding 108: macOS uses two types of memory compression relevant to GPU compute: (1) Virtual Memory Compression (software, since macOS Mavericks): compresses inactive process memory pages to free RAM, transparent to applications. (2) GPU Lossless Bandwidth Compression (hardware): reduces memory bandwidth (not storage size) when reading textures/render targets. Available since A12 Bionic. Private textures get this automatically; shared/managed textures need optimizeContentsForGPUAccess(). A15 Bionic added Lossy Compression providing 50% memory savings with minimal visual loss. These are distinct systems: VM compression reduces physical memory footprint, GPU compression reduces bandwidth consumed.
**Confidence**: high
**Source**: macOS Memory Management - Gregg Ant
**Evidence**: Gregg Ant blog on macOS memory management confirms VM compression since Mavericks. WWDC21 session 10153 details lossless bandwidth compression for GPU. Metal by Example "Understanding Metal Enhancements in A15 Bionic GPU" discusses lossy compression. Apple documentation for optimizeContentsForGPUAccess confirms the API.
**Tags**: memory-compression,VM-compression,lossless-compression,lossy-compression,bandwidth

## Finding 60: M4 base chip has 128-bit memory bus with 2 channels (64-bit per channel) using LPDDR5X-7500 (3750 MHz), delivering 120 GB/s bandwidth.
**Confidence**: high
**Source**: Low End Mac - M4 chip specs
**Evidence**: Low End Mac spec sheet lists: Memory Bus Width: 128-Bit, Total channels: 2, Bit per channel: 64-Bit, LPDDR5X-7500 (3750 MHz), Bandwidth: 120 GB/s. Confirmed by Apple official specs.
**Tags**: memory-bus,lpddr5x,m4,channels,bandwidth

## Finding 61: M4 Pro has 256-bit memory bus using LPDDR5X-8533 (4266 MHz), delivering approximately 273 GB/s bandwidth. Supports up to 64 GB.
**Confidence**: high
**Source**: Low End Mac - M4 Pro chip specs
**Evidence**: Low End Mac spec sheet lists: Memory Bus Width: 256-Bit, LPDDR5X-8533 (4266 MHz), ~273 GB/s bandwidth. Channel count not explicitly listed but community reports suggest 4 channels. Apple official bandwidth confirmed.
**Tags**: memory-bus,lpddr5x,m4-pro,channels,bandwidth

## Finding 62: M4 Max has 384-bit (binned) or 512-bit (full) memory bus with 3 or 4 channels (128-bit per channel) using LPDDR5X-8533, delivering 410 GB/s or 546 GB/s. Supports up to 128 GB.
**Confidence**: high
**Source**: Low End Mac - M4 Max chip specs
**Evidence**: Low End Mac spec sheet: Memory Bus Width: 384-Bit or 512-Bit, Total channels: 3 or 4, Bit per channel: 128-Bit, LPDDR5X-8533 (4266 MHz). 32-core GPU variant = 384-bit/410 GB/s, 40-core GPU variant = 512-bit/546 GB/s.
**Tags**: memory-bus,lpddr5x,m4-max,channels,bandwidth

## Finding 63: M5 uses LPDDR5X-9600 (faster than M4 LPDDR5X-7500), delivering 153.6 GB/s bandwidth with 128-bit bus. Supports up to 32 GB unified memory.
**Confidence**: verified
**Source**: Apple Support - MacBook Pro M5 Tech Specs
**Evidence**: Multiple sources confirm: LPDDR5X-9600 memory, 153.6 GB/s bandwidth (nearly 30% increase over M4 120 GB/s). Same 128-bit bus width as M4 base, bandwidth gain comes from higher clock speed.
**Tags**: memory-bus,lpddr5x,m5,bandwidth

## Finding 71: Memory bus width scales consistently: base=128-bit, Pro=256-bit, Max=512-bit, Ultra=1024-bit across all generations (M1 through M4). This is the primary mechanism for bandwidth scaling between chip variants.
**Confidence**: high
**Source**: NamuWiki - Apple Silicon M Series
**Evidence**: NamuWiki comprehensive table confirms: M1/M2/M3/M4 base=128-bit, M1/M2 Pro=256-bit, M1/M2/M3 Max=512-bit, M1/M2/M3 Ultra=1024-bit. M3 Pro is exception at 192-bit. M4 Pro=256-bit (back to standard).
**Tags**: memory-bus,scaling-pattern,pro,max,ultra

## Finding 69: Apple uses 16-bit per LPDDR5X controller/channel. Each controller accesses up to 4 GB. M4 base has 8 x 16-bit controllers composing 128-bit total bus. Data is interleaved 16 bits at a time across channels.
**Confidence**: high
**Source**: AnandTech Forums - Apple Silicon SoC thread
**Evidence**: AnandTech forum discussion: Apple uses "8 x16 LPDDR5 chips" per 128-bit package. Cache line size is 128 bytes. Minimum burst for 16-bit channel at LPDDR5 speeds is 32 beats = 64 bytes per channel. Wikipedia M4: "Each LPDDR5 memory controller contains a 16-bit memory channel and can access up to 4 GB."
**Tags**: lpddr5x,interleaving,memory-controller,cache-line

## Finding 68: On Apple Silicon, the GPU cannot use more than approximately 75% of total system memory. Example: 128GB RAM Mac can only use ~96GB for GPU tasks. This limit exists because the OS, CPU tasks, and other system components need reserved memory. The recommendedMaxWorkingSetSize property on MTLDevice reports this limit. For RDMA clusters, each node contributes up to 75% of its memory to the distributed pool.
**Confidence**: high
**Source**: Apple Community - GPU VRAM Discussion
**Evidence**: Apple Community forums, practical reports from developers. metal-usm project references recommendedMaxWorkingSetSize for heap sizing.
**Tags**: memory-limit,vram,75-percent,working-set,unified-memory

## Finding 106: Metal device.recommendedMaxWorkingSetSize returns the maximum GPU memory allocation that won't degrade performance. On Apple Silicon, this defaults to approximately 65-75% of total physical memory. Specific values: 128GB system → ~96GB GPU, 64GB → ~48GB, 32GB → ~21-24GB. This is a deliberate macOS Metal driver design choice to reserve memory for OS/CPU operations. The limit can be overridden via: sudo sysctl iogpu.wired_limit_mb=<MB> (macOS Sonoma+). A companion parameter iogpu.wired_lwm_mb sets the low-water mark for reclamation (default 20%). Safe practice: leave at least 8GB for system operations.
**Confidence**: high
**Source**: Apple Silicon limitations with local LLM - Greg Stencel
**Evidence**: Greg Stencel blog on Apple Silicon LLM limitations provides specific values and explains the 75% cap. osxdaily.com guide details iogpu.wired_limit_mb syntax. GitHub gist by havenwood shows default 8GB system reservation. Apple Developer Forums thread 732035 discusses the property. llama.cpp discussion #2182 confirms practical limits.
**Tags**: memory-limit,recommendedMaxWorkingSetSize,iogpu,wired-limit,GPU-allocation

## Finding 107: MTLDevice.maxBufferLength returns the maximum single buffer allocation size in bytes. This should be queried at runtime rather than hardcoded, as it varies by device. On Apple Silicon Macs with large unified memory (e.g., M4 Max 128GB), this value can be very large. The actual usable amount is further constrained by recommendedMaxWorkingSetSize and physical memory pressure. Maximum texture buffer width is limited by maxBufferLength divided by pixel size.
**Confidence**: verified
**Source**: Apple Developer: maxBufferLength
**Evidence**: Apple Developer documentation for MTLDevice.maxBufferLength: "The largest amount of memory, in bytes, that a GPU device can allocate to a buffer instance." Metal Feature Set Tables PDF contains per-family specifications. The property must be queried at runtime for accurate values.
**Tags**: maxBufferLength,buffer-size,memory-limit

## Finding 79: MTLPurgeableState has four states: keepCurrent (query without changing), nonVolatile (normal, cannot be purged), volatile (can be purged under memory pressure, contents become undefined), empty (already purged/discarded). Use setPurgeableState(.volatile) to hint that a resource can be reclaimed. Before re-reading a volatile resource, check setPurgeableState(.nonVolatile) return value - if it returns .empty, contents were purged. Known issue: Metal heap allocations may not always be released back to macOS even when marked volatile, as reported in developer forums.
**Confidence**: high
**Source**: Apple Developer: MTLPurgeableState
**Evidence**: Apple Developer documentation for MTLPurgeableState and setPurgeableState(_:) describe the four states. Developer forum thread (812368) reports Metal RHI memory leak where heap allocations are not released. The setPurgeableState API returns the old state, allowing detection of purged resources.
**Tags**: purgeable-state,memory-pressure,volatile,memory-management

## Finding 129: Apple M1 implements TSO (Total Store Ordering) for CPU cores as documented in academic analysis. For GPU-CPU memory ordering, Metal provides explicit synchronization primitives: MTLFence (within queue), MTLEvent (across queues), MTLSharedEvent (across devices/CPU). No documented hardware memory ordering guarantees exist across CPU-GPU boundary beyond what Metal APIs provide. All coherency is maintained at cache level but ordering requires API-level sync.
**Confidence**: high
**Source**: Analyzing the memory ordering models of the Apple M1
**Evidence**: ScienceDirect paper "Analyzing the memory ordering models of the Apple M1" confirms TSO for CPU. Metal documentation requires explicit sync for cross-domain ordering.
**Tags**: memory-ordering,TSO,MTLFence,MTLEvent,synchronization

## Finding 70: Apple places LPDDR5X SDRAM directly on the package substrate (not PoP or memory-down on logic board). Uses bespoke 64 and 128-Gbit x128 packages with at least 8 dies. This enables extreme channel density impossible with board-level implementations.
**Confidence**: high
**Source**: AnandTech Forums - How did Apple achieve high bandwidth
**Evidence**: AnandTech forum: Apple uses "bespoke 64 and 128-Gbit x128 packages with at least 8 dies, and places the SDRAM on the package substrate rather than using traditional PoP or memory-down configurations to handle that many channels."
**Tags**: package-substrate,lpddr5x,memory-packaging

## Finding 109: When Metal allocations approach or exceed physical memory limits on Apple Silicon: (1) partial offloading occurs - frameworks like llama.cpp load what fits on GPU and move remainder to CPU, causing significant performance degradation since CPU matrix ops are much slower, (2) macOS virtual memory (swap) can be triggered causing severe slowdowns, (3) the system can generate out-of-memory errors. GPU memory on Apple Silicon IS part of the virtual memory system and CAN be swapped, unlike discrete GPU VRAM which is not swappable. Memory Pressure indicator: green = OK, yellow = caution, red = performance suffering. Metal purgeableState volatile resources may be reclaimed first.
**Confidence**: high
**Source**: Apple Silicon limitations with local LLM - Greg Stencel
**Evidence**: Greg Stencel blog details partial offloading and swap behavior. Gregg Ant blog explains macOS swap uses SSD and is "generally transparent" with fast SSDs. Apple community forums confirm GPU memory sharing with system RAM means GPU allocations compete with all other system memory usage.
**Tags**: memory-pressure,swap,GPU-memory,offloading,performance

## Finding 72: Apple Silicon uses in-line ECC where error correction code is stored alongside data in the same DRAM device, not side-band ECC. This is standard for LPDDR5/LPDDR5X.
**Confidence**: medium
**Source**: AnandTech Forums - Apple Silicon SoC thread
**Evidence**: AnandTech forum discussion confirms in-line ECC implementation. Eclectic Light Company notes: "It is not known whether the memory controllers in Apple silicon chips are capable of managing ECC memory" (for external/full ECC), implying the in-line ECC is handled by DRAM itself.
**Tags**: ecc,lpddr5x,reliability

## Finding 53: M5 Neural Accelerators (one per GPU core) operate on data from unified memory via Metal 4 Tensor APIs. Optimal tile size is 32x32 (2KB per FP16 matrix). Data bandwidth is the PRIMARY bottleneck — requires 16 bytes/cycle/partition to saturate accelerators, exceeding typical memory interface. Supported formats: FP16 (with FP16 or FP32 accumulator) and INT8 (INT32 output). No BF16 or FP8 support. Per-core: ~1024 FLOPS/cycle FP16, ~2048 OPS/cycle INT8.
**Confidence**: verified
**Source**: Zakharko - Investigating GPU Neural Accelerators on Apple M5
**Evidence**: Independent benchmark (Zakharko) measured optimal 32x32 tile size. Patent WO2025071810 describes "32-wide 4-way hardware dot product datapath". Estimated M5 Max (40 cores, 1750MHz): ~70 TFLOPS FP16, ~130 TOPS INT8.
**Tags**: m5,neural-accelerator,tensor,bandwidth,memory-bottleneck,unified-memory

## Finding 127: GPU page faults on Apple Silicon are handled as "stop-the-world" events by the GPU firmware (ASC). If a GPU shader accesses an unmapped page, the firmware halts GPU execution. Recovery from GPU faults requires full machine reboot in severe cases. The macOS kernel driver does NOT communicate with GPU hardware directly - all fault handling goes through the ASC firmware via shared memory data structures.
**Confidence**: high
**Source**: Asahi Linux - Tales of the M1 GPU
**Evidence**: Asahi Linux: "GPU faults seem to be quite poorly handled as a stop-the-world process." Also: "If firmware crashes, the only way to recover is to fully reboot." The firmware-mediated architecture means no direct kernel-GPU fault handling path.
**Tags**: page-fault,firmware,ASC,fault-recovery,GPU-halt

## Finding 81: The constant CPU-GPU virtual address offset enables novel programming patterns: (1) SYCL USM-style shared pointers with zero-cost GPU-side address translation, (2) CPU pointers captured in GPU kernel lambdas get translated at encoding time, (3) indirect pointer chasing works because offset is uniform across allocations, (4) enables GPU-accessible linked data structures (lists, trees) with CPU pointers, (5) CPU malloc'd memory can be made GPU-accessible via heap mapping. Trade-offs: 1 cycle penalty per access for shared pointers, higher for system allocations.
**Confidence**: high
**Source**: Philip Turner - metal-usm Project
**Evidence**: metal-usm project documents: "At runtime, the shader tests the upper 16 bits of any GPU pointer. If empty, it adds the difference between the MTLHeap's GPU and CPU base address." Also: address translation can happen during encoding, becoming "zero-cost inside the shader."
**Tags**: constant-offset,programming-patterns,usm,sycl,pointer-translation,gpu-data-structures

## Finding 508: First comprehensive study of AMD Shared Virtual Memory (SVM) reveals aggressive prefetching strategy for demand paging that works well under normal conditions but causes excessive thrashing under memory oversubscription. SVM prefetching interacts pathologically with eviction policy, creating a feedback loop where prefetched pages are immediately evicted. Proposes algorithm-level and design-level solutions. Key distinction from NVIDIA UVM: SVM has fundamentally different prefetch/evict policies.
**Confidence**: high
**Source**: Shared Virtual Memory: Its Design and Performance Implications for Diverse Applications
**Evidence**: ICS 2024 paper. Tested on AMD GPUs deployed in supercomputers (Frontier-class). Quantitative analysis shows SVM prefetching grabs up to 64 pages per fault, which is efficient when memory is available but catastrophic under oversubscription. The prefetch-evict feedback loop is a novel finding not previously documented.
**Tags**: shared-virtual-memory,demand-paging,prefetching,oversubscription,thrashing,AMD

## Finding 73: On Apple Silicon (UMA), .shared and .private storage modes both allocate from the same physical unified memory pool. The key difference is access permissions: .shared allows CPU+GPU access, .private allows GPU-only access. For compute workloads, .private textures get automatic lossless bandwidth compression, while .shared buffers do not. Apple documentation states .shared is the default recommended choice for Apple GPUs, and .private should only be used when the CPU never accesses the resource.
**Confidence**: verified
**Source**: Apple Developer: Choosing a resource storage mode for Apple GPUs
**Evidence**: Apple documentation "Choosing a resource storage mode for Apple GPUs" explicitly states: "Use shared for resources that both the CPU and GPU access. Use private for resources that only the GPU accesses." The Metal Best Practices Guide notes that for small frequently-changing data, .shared is preferred because "the overhead of copying data to video memory may be more expensive than the overhead of the GPU accessing system memory directly." On UMA, both modes access the same physical DRAM.
**Tags**: storage-mode,shared,private,UMA,compute

## Finding 74: Private textures on Apple Silicon get automatic lossless bandwidth compression applied by the GPU hardware. Shared and managed textures can also get compression, but only after explicitly calling optimizeContentsForGPUAccess() on a blit command encoder. This compression reduces memory bandwidth consumption (not storage size), which is the primary performance advantage of .private over .shared on UMA systems. Linear textures backed by MTLBuffer cannot use lossless compression at all.
**Confidence**: verified
**Source**: WWDC21: Create image processing apps powered by Apple silicon
**Evidence**: WWDC21 session 10153 "Create image processing apps powered by Apple silicon" states: "Render command encoder enables a unique Apple GPU feature: lossless bandwidth compression for textures and render targets." Also: "For non-private textures, call optimizeContentsForGPUAccess() to stay on the fastest path." Lossless compression cannot be used with already-compressed formats, PixelFormatView flag, or linear textures backed by MTLBuffer.
**Tags**: storage-mode,private,lossless-compression,bandwidth,texture

## Finding 75: The M4 achieves approximately 100 GB/s measured GPU memory bandwidth via STREAM benchmarks, reaching ~85% of the theoretical peak 120 GB/s. The HPC benchmark paper used MTLResourceStorageModeShared for all page-aligned buffers, achieving 2.9 FP32 TFLOPS peak GPU compute performance. The paper did not compare shared vs private modes, but the high bandwidth utilization (85% of peak) with .shared mode suggests that on UMA, the storage mode does not significantly limit raw bandwidth for compute workloads.
**Confidence**: verified
**Source**: arXiv:2502.05317 - Apple vs. Oranges: Evaluating Apple Silicon for HPC
**Evidence**: arXiv:2502.05317 "Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC" measured 100 GB/s on M4 with STREAM benchmark, ~85% of theoretical 120 GB/s peak. All benchmarks used MTLResourceStorageModeShared with page-aligned buffers. FP32 throughput peaked at 2.9 TFLOPS. Note: this paper only benchmarked M4 base, not M4 Pro/Max variants.
**Tags**: storage-mode,shared,bandwidth,M4,benchmark,STREAM

## Finding 76: On Apple Silicon Macs, .managed storage mode is technically supported (macOS API compatibility) but behaves as a single-copy allocation since there is no discrete VRAM to synchronize with. Apple documentation states: "On GPUs without discrete memory, managed resources have only a single memory allocation accessible to both the CPU and GPU." The didModifyRange() and synchronizeResource() calls become effectively no-ops on Apple Silicon but should still be called for correctness and forward compatibility. Apple recommends always targeting a discrete memory model even on UMA systems.
**Confidence**: high
**Source**: Apple Developer: Synchronizing a managed resource in macOS
**Evidence**: Apple documentation "Synchronizing a managed resource in macOS" and Metal Best Practices Guide both confirm that macOS Metal apps should target a discrete memory model. The statement "On GPUs without discrete memory, managed resources have only a single memory allocation" comes from Apple docs. The didModifyRange() call notifies Metal about modified ranges. On UMA (Apple Silicon), this is internally optimized since there is no second copy to synchronize.
**Tags**: storage-mode,managed,UMA,synchronization,Apple-Silicon

## Finding 77: On Intel Macs with discrete GPUs, .managed mode maintains two physical copies: one in system RAM (CPU-accessible) and one in VRAM (GPU-accessible). After CPU writes, didModifyRange(range) must be called to synchronize only the modified range to VRAM. After GPU writes, a blit encoder synchronizeResource() call is needed to copy GPU results back to system RAM. .shared mode on Intel Macs is buffers-only (textures cannot use .shared) and stores data only in system RAM. .private stores data only in VRAM.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Resource Options
**Evidence**: Metal Best Practices Guide "Resource Options" section provides detailed guidelines: "Managed mode defines a synchronized memory pair for a resource, with one copy in system memory and another in video memory." It specifies didModifyRange: for CPU writes and synchronizeResource: for GPU writes. Also notes that shared mode is "buffers only" on macOS (Intel) and textures cannot use shared mode.
**Tags**: storage-mode,managed,Intel,discrete-GPU,synchronization

## Finding 490: First comprehensive characterization of Unified Physical Memory (UPM) on AMD MI300A APUs shows that UPM-based unified memory model matches or outperforms explicitly managed memory while reducing memory costs by up to 44%. Detailed analysis of memory latency, bandwidth, coherence overhead, TLB management, and Infinity Cache utilization. Page fault handling and allocation overhead are key bottlenecks.
**Confidence**: high
**Source**: Dissecting CPU-GPU Unified Physical Memory on AMD MI300A APUs
**Evidence**: Evaluated on El Capitan supercomputer MI300A APUs. Six applications tested. UPM avoids duplicate allocations (CPU+GPU copies), saving up to 44% memory. Key finding: unified model can match explicit management performance when system software is properly optimized. TLB and page fault overhead are the main remaining bottlenecks.
**Tags**: unified-memory,APU,MI300A,page-faults,TLB,coherence,memory-cost

## Finding 126: Apple GPU uses UAT (Unified Address Translator) with ARM64-identical page tables. 40-bit GPU virtual addresses (sign-extended to 64-bit), fixed 16KB page size, up to 16 GPU contexts with separate kernel/user address space splits. The macOS kernel driver configures GPU page table base pointer identically to its own ARM64 TTBR.
**Confidence**: verified
**Source**: Asahi Linux - Apple GPU (AGX) Documentation
**Evidence**: Asahi Linux AGX documentation: GPU MMU called UAT uses ARM64 page table format. Firmware configures GPU TTBR to share page tables with its own ARM64 address space. 40-bit VA space with 16KB pages.
**Tags**: UAT,page-tables,ARM64,virtual-memory,address-translation

## Finding 56: On Apple Silicon, CPU and GPU virtual addresses are always offset by a CONSTANT integer, enabling trivial translation. The metal-usm project exploits this: allocate large virtual memory range, find base CPU address and base GPU VA, delta between any allocation's CPU and GPU address matches. Enables: (1) CPU pointers usable from GPU with simple offset, (2) USM (Unified Shared Memory) SYCL-style programming, (3) zero-cost address translation during command encoding. Offset changes when heap reallocates but stays constant within a session.
**Confidence**: verified
**Source**: Philip Turner - metal-usm: CPU Pointers from Apple GPU
**Evidence**: Philip Turner metal-usm project: extensive experiments confirmed constant offset. Can scale from near-zero to almost all device RAM. Both shared and device USM pointers occur in same address space. 128 heaps is practical maximum before overhead increases >50%.
**Tags**: virtual-addressing,constant-offset,usm,cpu-gpu,zero-copy,programming-patterns

## Finding 89: makeBuffer(bytesNoCopy:length:options:deallocator:) creates an MTLBuffer that wraps existing memory without copying. Requirements: (1) pointer must be page-aligned, (2) length must be a multiple of the page size, (3) on Apple Silicon the hardware page size is 16KB (16384 bytes) not 4KB, (4) standard malloc cannot be used - must use vm_allocate, mmap, or posix_memalign with page alignment. On Apple Silicon ARM64, getpagesize() returns 16384. The page size should be queried dynamically via vm_page_size or getpagesize(), never hardcoded. The deallocator block is called when Metal is done with the buffer.
**Confidence**: verified
**Source**: Apple Developer: makeBuffer(bytesNoCopy:)
**Evidence**: Apple Developer documentation for makeBuffer(bytesNoCopy:) states pointer must be page-aligned. Kodeco forums show assertion error "pointer is not 4096 byte aligned" on Intel. Apple Silicon uses 16KB pages as confirmed by hw.pagesize sysctl output and Apple open-source kernel headers (xnu vm_param.h). Multiple sources confirm vm_allocate or mmap required instead of malloc.
**Tags**: zero-copy,bytesNoCopy,page-alignment,16KB,vm_allocate,mmap

## Finding 90: Apple Silicon ARM64 uses a 16KB (16384 byte) page size, unlike Intel x86_64 which uses 4KB (4096 byte) pages. This affects Metal bytesNoCopy alignment requirements: on Apple Silicon, buffers must be aligned to 16KB boundaries and sized in 16KB multiples. The M1 DART IOMMU has a minimum page size of 16KiB. Code should use vm_page_size (dynamic) rather than hardcoding PAGE_SIZE. This also means the minimum overhead for bytesNoCopy buffers is 16KB of alignment waste.
**Confidence**: high
**Source**: Confirmed Apple Silicon 16KB page size via sysctl
**Evidence**: Hacker News discussion with confirmed sysctl output: "hw.pagesize: 16384" on M1. Box64 project notes ARM64 Apple Silicon 16K page size requirement. K3s issue #7335 confirms 16K page kernel. Apple kernel source (xnu) defines ARM64 page parameters dynamically.
**Tags**: zero-copy,page-size,16KB,ARM64,Apple-Silicon,alignment

