# gpu-io — All Findings (38)

## Finding 595: TLB shootdowns are a significant bottleneck in memory-mapped I/O (previously misattributed to other components); fast page recycling (reusing same physical pages without TLB flush) achieves up to 28% improvement in real applications and 92% in microbenchmarks
**Confidence**: high
**Source**: Skip TLB flushes for reused pages within mmaps
**Evidence**: Proposes MAP_FPR flag for mmap() that instructs the kernel to reuse physical pages within the same process without triggering TLB shootdowns. Only triggers shootdowns when pages exit the process. No hardware modifications needed. Works within existing Linux virtual memory system.
**Tags**: mmap,tlb-shootdown,page-recycling,zero-copy,virtual-memory,io

## Finding 493: Concurrent computation and communication on GPUs achieves only 21% of ideal speedup due to compute/memory interference; offloading communication to GPU DMA engines via ConCCL closes the gap to 72% of ideal speedup with up to 1.67x speedup
**Confidence**: high
**Source**: Optimizing ML Concurrent Computation and Communication with GPU DMA Engines
**Evidence**: Evaluated C3 (concurrent computation and communication) performance on GPU-based ML systems. Found severe interference between concurrent kernels. Schedule prioritization improved to 42% of ideal. GPU DMA engine offloading (ConCCL) achieved 72% of ideal by freeing compute units entirely from communication tasks.
**Tags**: dma,compute-communication-overlap,ml-training,kernel-interference,bandwidth

## Finding 512: On cache-coherent interconnects like CXL 3.0, programmed I/O (CPU loads/stores to device) can deliver superior latency and comparable throughput to DMA, challenging the conventional wisdom that DMA is always better for device communication
**Confidence**: high
**Source**: Rethinking Programmed I/O for Fast Devices, Cheap Cores, and Coherent Interconnects
**Evidence**: Demonstrates that with fast cores and coherent interconnects, the overhead of setting up DMA descriptors, waiting for completion interrupts, and cache maintenance exceeds the cost of direct load/store access. Shows three applications: RPC-style accelerator invocation, streaming dataflow offloading, and serverless network interface. Coherence-based approach matches or beats DMA on real hardware.
**Tags**: programmed-io,dma,cache-coherent,cxl,interconnect,latency,zero-copy

## Finding 592: GPU zero-copy memory access combined with late materialization techniques enables data analytics workloads exceeding GPU memory capacity, achieving 5.7x average speedup over state-of-the-art GPU baseline without requiring GPU memory caching
**Confidence**: high
**Source**: Vortex: Overcoming Memory Capacity Limitations in GPU-Accelerated Large-Scale Data Analytics
**Evidence**: Vortex handles large-scale analytics workloads exceeding GPU memory. Key innovations: optimized IO primitives leveraging multiple PCIe links, programming model separating kernel development from IO scheduling, late materialization via zero-copy. 2.5x better price-performance vs CPU-based DuckDB.
**Tags**: zero-copy,data-analytics,out-of-core,late-materialization,gpu-memory-limit

## Finding 236: BaM (ASPLOS 2023) enables GPU threads to directly submit NVMe I/O by mapping SSD doorbell registers into GPU address space via GPUDirect Async. Submission/completion queues in GPU HBM. GPU-side software cache with clock replacement, 4KB cache lines, warp coalescing via __match_any_sync. Achieves 45.8M random read IOPS, 10.6M write IOPS with ten Optane SSDs, 22.9 GB/s random read (90% PCIe Gen4 x16 peak).
**Confidence**: verified
**Source**: BaM System Architecture (ASPLOS 2023)
**Evidence**: BaM paper: 'leverages GPUDirect Async to map NVMe SSD doorbells to CUDA address space.' '45.8M random read IOPs with ten Optane SSDs.' 'BaM achieves 25GBps using four SSDs while GDS only reaches 23.6% of PCIe bandwidth at 4KB.'
**Tags**: bam,gpu-initiated-io,nvme,asplos-2023,iops,software-cache

## Finding 237: BaM dramatically outperforms GPUDirect Storage for fine-grained (<32KB) random access. GDS requires 32KB I/O blocks to saturate PCIe due to CPU control path overhead per request. BaM efficient at 4KB -- critical for pointer-chasing workloads (graph traversal, recommendation systems, GNNs). BFS graph traversal from disk: 1.0x speed with 21.7x hardware cost reduction by eliminating need for full-graph DRAM.
**Confidence**: verified
**Source**: BaM System Architecture (ASPLOS 2023)
**Evidence**: BaM: 'GDS requires 32KB I/O blocks, whereas BaM handles smaller granularities through GPU-native software stack.' CPU initiation unsuitable due to 'high CPU-GPU synchronization overheads, I/O traffic amplification, long CPU processing latencies.'
**Tags**: bam,gpudirect-storage,fine-grained,random-access,pointer-chasing,graph-traversal

## Finding 238: AGILE (SC 2025) introduces asynchronous GPU-initiated NVMe I/O, solving BaM''s synchronous stalling. GPU threads submit NVMe commands and continue computation. Lightweight GPU daemon polls completion queues non-blocking. Achieves 1.88x speedup by overlapping compute and I/O, 1.75x over BaM on DLRM inference. Share Table with MOESI-inspired protocol prevents deadlocks. Cache: four states (INVALID, BUSY, READY, MODIFIED).
**Confidence**: verified
**Source**: AGILE: Async GPU-SSD Integration (SC 2025)
**Evidence**: AGILE paper: 'overlaps computation and I/O, improving performance by up to 1.88x.' 'Compared to BaM on DLRM, AGILE achieves 1.75x speedup.' GPU daemon 'continuously polls CQE in non-blocking fashion.'
**Tags**: agile,async-io,gpu-ssd,sc-2025,computation-overlap,moesi,share-table

## Finding 239: NVIDIA SCADA implements complete NVMe driver inside GPU, taking full ownership of NVMe block devices using load/store operations. Eliminates CPU from both data AND control paths. At SC25: 44 Micron 9650 PCIe Gen6 SSDs via Broadcom Gen6 switches to 3 H100 GPUs achieved 230M 512B random read IOPS. Evolution: Gen1 (CPU bounce buffers) -> Gen2 (GDS: direct DMA, CPU control) -> Gen3 (SCADA: GPU controls everything).
**Confidence**: high
**Source**: SC25 Performance Breakthrough: 230M IOPS (Micron)
**Evidence**: Micron SC25: '230 million 512B random read IOPS with SOL benchmark SCADA workload.' Blocks and Files: 'SCADA takes control path away from CPU as well.'
**Tags**: scada,nvidia,nvme-driver,gpu-initiated,230m-iops,gen6,evolution

## Finding 240: GPUfs (ASPLOS 2013) was first POSIX-like file API for GPU kernels: gopen, gread/gwrite (position-specified like pread/pwrite), gclose, gmmap, gfsync. All warp threads must invoke same GPUfs call with same arguments. GPU-as-client, CPU-as-server architecture with GPU-CPU RPC. Weak consistency model (sync-to-open semantics like AFS). GPU-side buffer cache extends host buffer cache.
**Confidence**: verified
**Source**: GPUfs: Integrating a File System with GPUs (ACM TOCS 2014)
**Evidence**: GPUfs paper: 'gopen/gclose in namespace of single threadblock, gread/gwrite always supply explicit file offsets.' 'All application threads in a warp must invoke the same GPUfs call with same arguments.'
**Tags**: gpufs,posix,gpu-filesystem,file-api,buffer-cache,asplos-2013

## Finding 241: LithOS (SOSP 2025) is first step toward a GPU OS: TPC Scheduler for fine-grained spatial scheduling on disjoint SM sets with TPC stealing, transparent kernel atomization (~500us quanta to reduce head-of-line blocking), per-(kernel,shape) micro-models. Reduces tail latencies 13x vs MPS, 12x vs Orion. Critically does NOT manage storage/I/O -- explicitly does not support swapping to CPU.
**Confidence**: verified
**Source**: LithOS: An Operating System for Efficient ML on GPUs (SOSP 2025)
**Evidence**: LithOS paper: '500us quantum for kernel slicing.' 'Reduces tail latencies 13x vs. MPS and 12x vs. Orion.' 'LithOS does not manage memory by allowing for swapping to the CPU.'
**Tags**: lithos,gpu-os,tpc-scheduler,kernel-atomization,sosp-2025,no-storage

## Finding 242: True GPU-initiated I/O on Apple Silicon would require: (1) mapping NVMe queue registers into GPU address space (feasible -- SSD controller on-die via Apple Fabric), (2) new MSL intrinsics for I/O commands, (3) GPU-side NVMe driver or command formatter, (4) GPU-side completion handling, (5) filesystem metadata management. Apple''s integrated architecture (GPU, memory, NVMe on-die) makes this more natural than discrete GPU systems, but NO current Metal API supports any form of GPU-initiated storage.
**Confidence**: verified
**Source**: MTLIOCommandQueue - Apple Developer Documentation
**Evidence**: Metal shaders can only access MTLBuffer, MTLTexture, threadgroup/simdgroup memory, device/constant address spaces. 'There is no API in MSL for a GPU thread to open a file, issue an NVMe command, trigger DMA, or initiate any I/O operation.'
**Tags**: apple-silicon,metal,gpu-initiated-io,feasibility,nvme,gap

## Finding 250: Memory-to-storage bandwidth gap on Apple Silicon is 15-70x: M1: 60 vs ~3 GB/s (20x), M4: 100 vs ~3-7 GB/s (14-33x), M4 Max: 546 vs ~7.9 GB/s (69x), M5: 153.6 vs ~6.3 GB/s (24x). GPU compute overwhelmingly memory-bound, not storage-bound. Data must be resident in unified memory for full bandwidth. Double/triple buffering essential to pre-stage data.
**Confidence**: verified
**Source**: Apple vs. Oranges: Evaluating Apple Silicon M-Series SoCs for HPC
**Evidence**: GPU bandwidth from arXiv:2502.05317 STREAM benchmark (~85% theoretical peak). SSD from Blackmagic tests. 'SSD cannot saturate GPU compute; buffering to pre-stage data in memory is essential.'
**Tags**: bandwidth-gap,memory-bandwidth,ssd-bandwidth,bottleneck,gpu-compute,roofline

## Finding 251: No published head-to-head benchmark exists for MTLIOCommandQueue vs mmap vs standard I/O. Qualitative comparison: Standard I/O (read/pread): CPU-mediated, blocks thread, ~3-7 GB/s. mmap+bytesNoCopy: zero-copy after page fault, lazy loading, good for random access. MTLIOCommandQueue: async, non-blocking, GPU-synchronized, inline decompression, priority queues. WWDC22 demo showed MTLIOCommandQueue dramatically outperforming pread for texture tile streaming.
**Confidence**: high
**Source**: Load resources faster with Metal 3 - WWDC22
**Evidence**: WWDC22 showed side-by-side comparison with visible reduction in low-res artifact duration. 'Key advantage is not raw throughput (limited by same SSD) but async overlap with GPU compute and priority-based scheduling.'
**Tags**: mtlio,mmap,pread,comparison,async-io,streaming,no-formal-benchmark

## Finding 252: MLX benchmarks (arXiv:2510.18921): BERT-base batch-1 inference: M1 14.48ms, M2 Max 4.92ms, NVIDIA A10 35.37ms. Apple Silicon wins on small batches due to unified memory eliminating transfer overhead. M2 Max 4.92ms is ~7x faster than A10''s 35.37ms at batch-1. NVIDIA wins on large batches from raw compute. For streaming network data with small batches, Apple''s unified memory provides lower latency.
**Confidence**: verified
**Source**: Benchmarking On-Device Machine Learning on Apple Silicon with MLX
**Evidence**: arXiv:2510.18921: 'M1 14.48ms, M2 Max 4.92ms, A10 35.37ms.' M2 Max sublinear scaling: '8.02ms to 70.48ms for 1x to 32x batch (9x time for 32x work).'
**Tags**: mlx,bert,inference-latency,m1,m2-max,nvidia-comparison,small-batch

## Finding 219: MTLIOCommandQueue is created via MTLIOCommandQueueDescriptor with three properties: type (concurrent or serial), priority (high/normal/low), and maxCommandBufferCount. MTLIOCommandBuffer supports three load operations: loadTexture (file to texture), loadBuffer (file to buffer), loadBytes (file to CPU memory). All read from MTLIOFileHandle. The API is strictly read-only -- no write/store capability exists.
**Confidence**: verified
**Source**: MTLIOCommandQueueDescriptor - Apple Developer Documentation
**Evidence**: Apple documentation for MTLIOCommandQueueDescriptor. WWDC22 'Load resources faster with Metal 3' confirms three load types. No write-back mentioned in any session.
**Tags**: metal-io,mtliocommandqueue,api,read-only,load-texture,load-buffer

## Finding 220: Metal I/O supports five compression methods: LZ4 (fastest decompression, ~1 byte/cycle), LZFSE (Apple-optimized, 2-3x faster than zlib with comparable ratio), Zlib (standard DEFLATE), LZBITMAP (bitmap-optimized), and LZMA (highest ratio). Assets compressed offline via MTLIOCreateCompressionContext with configurable chunk size (default 64KB). During load, Metal translates byte offsets to compressed chunks, decompresses inline, and writes directly to destination Metal resource.
**Confidence**: verified
**Source**: Load resources faster with Metal 3 - WWDC22
**Evidence**: WWDC22 session: 'Metal 3 performs inline decompression by translating offsets to chunks.' Apple LZFSE docs: 'decompresses 2-3x faster than zlib.' LZ4 docs: 'multiple GB/s per core.'
**Tags**: metal-io,compression,lz4,lzfse,zlib,lzbitmap,lzma,inline-decompression

## Finding 221: Metal Fast Resource Loading integrates with sparse textures for virtual texture streaming. Sparse textures with 16KB tile size used as destination; Metal I/O loads tiles on demand from compressed pack files. GPU access counters (feedback buffer) track which tiles are needed but not resident, driving the streaming engine. Enables rendering textures far larger than memory with quality scaling to memory budget.
**Confidence**: verified
**Source**: Streaming large images with Metal sparse textures - Apple Developer Documentation
**Evidence**: WWDC22: 'uses sparse textures with a tile size of 16 kilobytes.' Apple docs: 'Sparse textures let you stream more visible detail with the same memory budget.'
**Tags**: metal-io,sparse-texture,virtual-texturing,tile-streaming,16kb-tiles,feedback

## Finding 222: Metal 4 adds complementary features: placement sparse resources (allocate without storage pages initially, provide pages on-demand from placement heap), unified MTL4ComputeCommandEncoder (compute+blit+acceleration in one encoder), and residency sets (configured once at startup, support background thread updates for streaming). These complement MTLIOCommandQueue for streaming scenarios. Automatic hazard tracking removed -- manual synchronization required.
**Confidence**: verified
**Source**: Discover Metal 4 - WWDC25
**Evidence**: WWDC25 'Discover Metal 4': placement sparse resources, residency sets, unified encoder. No replacement of MTLIOCommandQueue announced.
**Tags**: metal-4,placement-sparse,residency-sets,unified-encoder,streaming

## Finding 223: makeBuffer(bytesNoCopy:length:options:deallocator:) wraps existing contiguous memory as MTLBuffer without copying. On Apple Silicon with .storageModeShared, CPU and GPU access same physical pages. Pointer MUST be page-aligned (4096-byte minimum assertion, but ARM64 hardware uses 16KB pages). Only vm_allocate, mmap, or posix_memalign provide suitable alignment -- standard malloc cannot be used.
**Confidence**: verified
**Source**: makeBuffer(bytesNoCopy:) - Apple Developer Documentation
**Evidence**: Apple docs: 'creates a buffer that wraps an existing contiguous memory allocation.' Kodeco forums: crash 'pointer is not 4096 byte aligned.' Apple XNU: ARM64 uses 16KB pages. Developer forums: 'One cannot use malloc for newBufferWithBytesNoCopy.'
**Tags**: makeBuffer,bytesNoCopy,zero-copy,page-alignment,4096,16kb,vm_allocate,mmap

## Finding 224: Combining mmap(file) with makeBuffer(bytesNoCopy:) and .storageModeShared creates file-backed Metal buffer where GPU reads trigger page faults loading data from SSD on demand. Pattern: mmap(nil, fileSize, PROT_READ, MAP_SHARED, fd, 0) -> makeBuffer(bytesNoCopy:) -> GPU reads. This is technically functional but NOT officially documented or supported by Apple.
**Confidence**: high
**Source**: Loading models with mmap - MLX GitHub Discussion
**Evidence**: MLX discussion #615 confirms prototype: 'enhanced MLX metal allocator to use mmap and create metal buffers via newBufferWithBytesNoCopy for loading from safetensors.' Research notes: 'Apple does not officially document or support using mmap''d file regions directly as GPU-accessible memory.'
**Tags**: mmap,file-backed,demand-paging,page-fault,zero-copy,gpu-buffer,unsupported

## Finding 225: Apple GPU uses UAT (Unified Address Translator) -- essentially ARM64 MMU with identical page tables. 40-bit virtual addresses (sign-extended to 64-bit), 16KB pages, 4-level page table hierarchy. GPU firmware (ARM64 ASC coprocessor) literally uses the same page table base pointer as the GPU MMU. All memory is fully coherent between CPU and GPU -- 'we haven''t used a single cache management instruction yet and everything still works.'
**Confidence**: verified
**Source**: Tales of the M1 GPU - Asahi Linux
**Evidence**: Asahi Linux docs: 'GPU VAs are 40 bits, top bit sign-extended.' 'UAT is essentially the ARM64 MMU using identical page tables.' 'All memory is coherent as far as we can tell.' GPU faults are 'stop-the-world' process where macOS dumps GPU MMIO registers.
**Tags**: uat,page-tables,virtual-address,cache-coherency,arm64-mmu,agx,40-bit

## Finding 226: madvise() hints affect page cache for mmap''d GPU buffers: MADV_SEQUENTIAL enables read-ahead + early page freeing, MADV_WILLNEED triggers pre-loading (reduces page faults ~110 to ~6), MADV_DONTNEED marks for reuse. However, kernel page cache eviction algorithms poorly match GPU inference patterns -- I/O stalls compound throughout computational graphs without explicit control over swap timing.
**Confidence**: high
**Source**: Should use mmap for model loading - llama.cpp Issue #91
**Evidence**: llama.cpp issue #91: 'madvise() hints like MADV_SEQUENTIAL and MADV_WILLNEED to guide kernel paging behavior.' MLX mmap prototype: 'Kernel page cache eviction algorithms poorly match inference patterns.'
**Tags**: madvise,madv_sequential,madv_willneed,page-cache,prefetching,eviction

## Finding 227: Philip Turner''s metal-usm demonstrates CPU-GPU pointer translation via constant integer offset on Apple Silicon. CPU and GPU virtual addresses for same physical memory differ by fixed constant. Default mode: 1 cycle penalty per memory access. All USM allocations must come from pre-declared Metal heaps (useHeaps mandatory, ~128 max). Heap size typically 256 MB. Critical: calling useResource on every heap is mandatory or 'the GPU will freeze and force restart.'
**Confidence**: verified
**Source**: metal-usm: Access CPU pointers from inside the Apple GPU
**Evidence**: metal-usm README: 'CPU and GPU addresses will always be off by a constant integer.' '1 cycle penalty for every memory access, except from first 30 pointers captured as lambda arguments.'
**Tags**: metal-usm,pointer-translation,constant-offset,gpu-va,cpu-va

## Finding 246: macOS Tahoe 26.2 enables RDMA over Thunderbolt 5: 80 Gbps max bandwidth, 5-9 microsecond latency (vs ~300us TCP/IP). Uses InfiniBand Verbs (ibverbs) API. Requires TB5 hardware (M4 Pro/Max, M3 Ultra+). Enabled via Recovery Mode (rdma_ctl enable). Max 4 Macs in cluster. Jeff Geerling tested 4 Mac Studios (1.5TB unified memory): Qwen3 235B at 31.9 tok/s (Exo RDMA) vs 15.2 tok/s (llama.cpp TCP).
**Confidence**: high
**Source**: 1.5 TB of VRAM on Mac Studio - RDMA over Thunderbolt 5
**Evidence**: Jeff Geerling: '1.5 TB of VRAM on Mac Studio - RDMA over TB5.' Stabilise.io: '80 Gbps, 5-9 microseconds.' DeepSeek V3.1 (671B): 32.5 tok/s RDMA vs 14.6 tok/s TCP.
**Tags**: rdma,thunderbolt-5,mac-cluster,benchmark,latency,exo

## Finding 247: MLX provides 4 distributed backends: MPI, Ring (TCP sockets), JACCL (RDMA over TB), NCCL. JACCL achieves order-of-magnitude lower latency than Ring backend. Uses ibverbs via dynamic library loading. Supports all_sum, all_gather, send/recv. Merged Dec 2025 (PR #2808). Known issue: send/recv can SIGBUS under asymmetric timing; all_sum preferred. MLX_METAL_FAST_SYNCH=1 optimizes GPU-CPU sync for low-latency.
**Confidence**: verified
**Source**: Distributed Communication - MLX Documentation
**Evidence**: MLX distributed docs: JACCL 'latency order of magnitude lower than Ring backend.' PR #2808: 30 commits. 'mlx.distributed_config --backend jaccl --over thunderbolt --auto-setup' for configuration.
**Tags**: mlx,jaccl,rdma,distributed,all-reduce,thunderbolt,ibverbs

## Finding 248: Network-to-GPU zero-copy on Apple Silicon: receive data into page-aligned buffer (via mmap/vm_allocate), wrap with makeBuffer(bytesNoCopy:) -- Metal buffer without copying. Ingonyama metal-poc: 0.0004ms memory transfer overhead (0.0009% of total) on M1 Pro for ~1GB arrays vs CUDA 585.84ms (97.77%) on RTX 4080 -- 12.9x advantage for mixed CPU-GPU workflows. Apple''s DART (IOMMU) with default-deny policy controls all DMA access.
**Confidence**: verified
**Source**: metal-poc: Zero cost memory transfer between CPU and GPU in Metal
**Evidence**: metal-poc: 'Zero cost memory transfer between CPU and GPU in Metal.' WWDC20: 'DART starts with default-deny policy.' NIC path: DMA through DART -> unified memory -> already GPU-accessible.
**Tags**: zero-copy,makeBuffer,bytesNoCopy,dart,iommu,network,dma

## Finding 249: RDMA cluster scaling: HPL single node 1.3 TFLOPS -> 4-node 3.7 TFLOPS (~70% efficiency). LLM inference scales better: Qwen3 235B from 19.5 tok/s (1 node) to 31.9 tok/s (4 nodes). Non-RDMA TCP shows degradation: 20.4 -> 15.2 tok/s. Cluster cost ~$40,000 for 4 Mac Studios vs claimed ~$770,000 NVIDIA equivalent. Primarily inference play -- Macs lag for training.
**Confidence**: high
**Source**: Apple's RDMA Revolution - Stabilise.io
**Evidence**: Jeff Geerling: HPL 4-node 3.7 TFLOPS. Qwen3 235B 31.9 tok/s vs TCP 15.2 tok/s. Stabilise.io: '~$730,000 savings vs NVIDIA.' 'primarily inference, not training.'
**Tags**: rdma,scaling,hpl,benchmark,cluster,inference,cost-comparison

## Finding 228: Apple''s storage controller (AppleANS3NVMeController) is an ARM64 coprocessor running RTKit with ~800KB firmware loaded by iBoot. Fully integrated into SoC die -- NOT on the NAND module. Connected via Apple Fabric (not standard PCIe). Separate S5E NAND SSD Controller (~700KB firmware) manages raw NAND interface. NAND module contains only flash chips + power circuits.
**Confidence**: verified
**Source**: Introduction to Apple Silicon - Asahi Linux Documentation
**Evidence**: Asahi Linux: ANS is NVMe controller with RTKit firmware. Thomas Kaiser: 'AppleANS3NVMeController.' Jeff Geerling: 'card that''s user-replaceable is just flash chips, while storage controller is part of the SoC.' Asahi lists both ANS and S5E as separate firmware components.
**Tags**: ans3,nvme-controller,rtkit,arm64-coprocessor,apple-fabric,s5e,soc-integrated

## Finding 229: SSD bandwidth scales with NAND chip count: M4 base (512GB, 2 NAND) ~2.9/3.3 GB/s R/W; M4 Pro (2TB, 4 NAND) ~6.7/7.5 GB/s; M4 Max (4TB) ~7.3/8.0 GB/s; M5 base (256GB) ~6.3/6.1 GB/s (2.5x faster than M4 base due to PCIe 4.0 upgrade). MacBook Air uses PCIe Gen 3; Pro uses Gen 4. M1 ANS3 supports NVMe 1.10, max I/O segment 1MB, command pool 64 commands.
**Confidence**: high
**Source**: M5 MacBook Pro SSD 2.5x faster - Tom's Hardware
**Evidence**: Multiple Blackmagic benchmarks. Tom''s Hardware M5: '6,323 MB/s read, 6,068 MB/s write.' Thomas Kaiser M1: 'IOMaximumSegmentByteCountRead 1,048,576 bytes, command pool 64.' MacRumors: 'Air has PCIe gen 3, Pro has gen 4.'
**Tags**: ssd-bandwidth,m4,m4-pro,m4-max,m5,nand-count,pcie-gen3,pcie-gen4

## Finding 230: Every Apple Silicon Mac encrypts all internal SSD data at hardware level using AES-256 in XTS mode. AES crypto engine sits in DMA path between NAND and memory, within Secure Enclave''s storage controller. Zero measurable performance overhead. Enabling FileVault merely adds password protection to existing hardware encryption keys -- no re-encryption occurs. Keys never leave Secure Enclave.
**Confidence**: verified
**Source**: Volume encryption with FileVault in macOS - Apple Support
**Evidence**: Apple Security Guide: 'encrypted internal storage directly connected to Secure Enclave leverage its hardware security.' Eclectic Light: 'enabling FileVault incurs no penalty.' Apple Support: 'AES-256 in XTS mode.'
**Tags**: aes-256,xts,encryption,secure-enclave,dma-path,zero-overhead,filevault

## Finding 231: Apple Silicon SSDs use TLC 3D NAND with SLC write cache. During burst writes, SLC cache provides peak speed (e.g., 7.5 GB/s M4 Pro). When cache exhausted after continuous writes, speed drops to native TLC rate. Sustained sequential reads more consistent. M1 ATTO showed ~10 GB/s with small files vs 3.1 GB/s with 10GB files (cache + controller queue effects). Metal I/O operates through APFS -- no filesystem bypass.
**Confidence**: high
**Source**: How fast is the SSD inside an M1 Mac? - Eclectic Light
**Evidence**: Eclectic Light: 'Most TLC SSDs use caches, typically static cache up to 16GB.' 'Nearly 10 GB/s at 64 MiB vs 3.1 GB/s with 10GB files.' Apple docs: MTLIOFileHandle operates through file system. APFS: 'copy-on-write design that uses I/O coalescing.'
**Tags**: slc-cache,tlc,burst-vs-sustained,apfs,no-filesystem-bypass,io-coalescing

## Finding 232: Triple buffering is the canonical Metal streaming pattern: 3 pre-allocated buffers in ring with dispatch_semaphore(3). Each frame: CPU waits on semaphore, writes buffer[n%3], encodes commands, GPU signals on completion. Two buffers can cause stalls from command buffer transfer latency; three balances parallelism, memory, and latency. Pattern works for both render and compute streaming.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Triple Buffering
**Evidence**: Apple Metal Best Practices Guide provides complete code with dispatch_semaphore_create(kMaxInflightBuffers), ring buffer index, completion handler signaling. 'Triple buffering accommodates command buffer transfer time.'
**Tags**: triple-buffering,ring-buffer,dispatch-semaphore,synchronization,inflight-buffers

## Finding 233: Metal sparse textures provide GPU-driven demand paging. Memory mapped in 16KB tile units. Textures start with no mapped tiles; tiles mapped/unmapped from sparse heaps on demand. GPU access counters track which tiles are sampled but not loaded, driving streaming with very low overhead. Combined with MTLIOCommandQueue, streams tile data directly from SSD. Metal 4 adds placement sparse resources for explicit page-level control.
**Confidence**: verified
**Source**: Streaming large images with Metal sparse textures - Apple Developer Documentation
**Evidence**: Apple docs: 'Metal sparse textures allow rendering high-resolution image without allocating memory for entire image.' 16KB tile matches GPU page size. Metal 4: 'Placement Sparse Resources allocate without storage pages initially.'
**Tags**: sparse-textures,demand-paging,access-counters,16kb-tiles,metal-4,placement-sparse

## Finding 234: vllm-mlx implements production continuous batching on Apple Silicon: dynamically groups concurrent requests, admits new at token boundaries, completed exit without blocking others. Content-based prefix caching achieves 5.8x TTFT speedup; image caching 28x (21.7s to <1s). Up to 525 tok/s on Qwen3-0.6B (M4 Max), 21-87% higher throughput than llama.cpp, 3.7x scaling at 16 concurrent requests.
**Confidence**: verified
**Source**: Native LLM and MLLM Inference at Scale on Apple Silicon
**Evidence**: arXiv:2601.19139v2: '4.3x aggregate throughput at 16 concurrent requests' and '21-87% higher throughput than llama.cpp across models (0.6B to 30B).' Content caching: '28x speedup, reducing latency from 21.7 seconds to under 1 second.'
**Tags**: vllm-mlx,continuous-batching,kv-cache,streaming-inference,throughput,prefix-caching

## Finding 235: MLX arrays live in shared unified memory -- operations directed to CPU or GPU without data transfers. Lazy evaluation enables operation fusion and reduces allocation overhead. Evaluating after cast to float16 reduces peak memory ~33%. Demonstrated ~2x speedup on M1 Max from running GPU matmul concurrently with CPU exponential operations. set_memory_limit() and clear_cache() APIs manage pressure programmatically.
**Confidence**: verified
**Source**: Unified Memory - MLX Documentation
**Evidence**: MLX docs: 'arrays in MLX live in shared memory, operations can be performed on any supported device without transferring data.' 'evaluating after cast reduces peak memory by nearly a third.' '~2x speedup on M1 Max from CPU-GPU parallel.'
**Tags**: mlx,unified-memory,lazy-evaluation,zero-copy,operation-fusion,cpu-gpu-parallel

## Finding 243: GPU compute results written to storage via CPU-mediated pattern: (1) GPU writes to MTLBuffer in .storageModeShared (unified memory), (2) addCompletedHandler or waitUntilCompleted signals CPU, (3) CPU accesses same physical memory (zero-copy on Apple Silicon), (4) CPU writes to file via POSIX APIs. Key: step 3 is free on Apple Silicon -- no DMA copy. On discrete GPUs, managed mode requires explicit synchronizeResource: blit after GPU writes.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Resource Options
**Evidence**: Metal Best Practices: shared mode CPU+GPU access same memory. 'After encoding GPU write, encode blit that includes synchronizeResource:' (discrete GPUs only). Handlers 'invoked on undefined thread, should complete quickly.'
**Tags**: write-back,shared-buffer,cpu-mediated,unified-memory,completion-handler

## Finding 244: Apple Silicon''s write-back path is uniquely efficient: GPU writes to unified memory (nanoseconds, no bus crossing) -> CPU accesses same physical memory (no copy) -> filesystem -> on-die SSD controller (no PCIe). Entire path stays within SoC. On discrete GPUs: GPU VRAM -> PCIe -> CPU RAM -> filesystem -> SSD (at least one PCIe crossing). The only bottleneck on Apple Silicon is SSD write throughput (3-7 GB/s). Memory bandwidth is 20-100x higher than SSD throughput.
**Confidence**: verified
**Source**: Explore the New System Architecture of Apple Silicon Macs - WWDC20
**Evidence**: WWDC20 Apple Silicon architecture session. GPU-to-CPU handoff: 'no DMA transfer, no copy, just pointer passing.' SSD is clear bottleneck: 'memory bandwidth 120-546 GB/s vs SSD 3-7 GB/s.'
**Tags**: apple-silicon,unified-memory,write-back-efficiency,zero-copy,ssd-bottleneck

## Finding 245: For continuous GPU-compute-to-file output, triple-buffer pipeline: Frame N uses GPU computing into buffer A while CPU writes buffer C to disk and SSD loads into buffer B. MTLSharedEvent synchronizes: GPU signals after compute, CPU waits before accessing buffer, CPU signals after write to release buffer. Theoretical throughput = max(GPU_compute_time, file_write_time). BaM also supports GPU-initiated write-back via dirty cache lines with explicit flush.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Triple Buffering
**Evidence**: Metal Best Practices triple buffering guide. BaM: 'Write-back semantics with explicit flush operations.' AGILE cache: 'four states including MODIFIED for dirty tracking.'
**Tags**: triple-buffering,pipeline,write-back,mtlsharedevent,bam-writeback

