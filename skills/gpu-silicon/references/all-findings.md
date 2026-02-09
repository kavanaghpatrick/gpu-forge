# gpu-silicon — All Findings (57)

## Finding 548: On-chip memory controllers with lossless block compression (LZ4/ZSTD) of model weights and KV cache achieve 25.2% memory footprint reduction for weights and 46.9% for KV cache; hardware prototype delivers 8 TB/s throughput at 4 GHz with under 3.8 mm2 area
**Confidence**: high
**Source**: Reimagining Memory Access for LLM Inference: Compression-Aware Memory Controller Design
**Evidence**: Proposes enhancing memory controllers in AI accelerators with compression logic. Lossless compression applied transparently to model weights and KV cache. Dynamic quantization scales compression ratio with context. LLM-aware memory configuration improves bit-level accessibility and compressibility. Hardware prototype in 7nm technology.
**Tags**: memory-compression,memory-controller,bandwidth-amplification,kv-cache,llm-inference,hardware

## Finding 541: SCM-aware DRAM cache bypass that considers multi-dimensional access characteristics achieves up to 12.5x speedup and 89.3% energy reduction vs HBM alone; repurposing L2 cache space for DRAM cache tags (Configurable Tag Cache) reduces probe traffic by 91-93%
**Confidence**: high
**Source**: Bandwidth-Effective DRAM Cache for GPUs with Storage-Class Memory
**Evidence**: Proposes a DRAM cache paired with Storage-Class Memory for GPUs. Three innovations: (1) SCM-aware bypass policy considering multi-dimensional access characteristics, (2) Configurable Tag Cache repurposing L2 for DRAM cache tags, (3) AMIL organization co-locating all tags in a single DRAM column. Also includes SCM throttling and SLC/MLC mode adaptation.
**Tags**: dram-cache,slc,mlc,cache-bypass,tag-cache,storage-class-memory,energy-efficiency

## Finding 537: LATPC exploits VPN (Virtual Page Number) regularity across threads within a warp to coalesce TLB misses at the warp instruction level. Compresses multiple TLB miss requests into fewer MSHR (Miss Status Holding Register) entries, reducing contention. Achieves 1.47x geometric mean speedup over baseline without any TLB prefetching. Key insight: GPU warps have highly regular memory access patterns where threads access consecutive pages, making TLB miss coalescing natural.
**Confidence**: high
**Source**: LATPC: Accelerating GPU Address Translation Using Locality-Aware TLB Prefetching and MSHR Compression
**Evidence**: MICRO 2025 paper. Evaluated on 24 GPU workloads. The regularity of VPNs across warp threads is a direct consequence of coalesced memory access patterns that GPU programmers already optimize for. LATPC turns this regularity into TLB-level optimization automatically. MSHR compression prevents the TLB miss queue from becoming the bottleneck.
**Tags**: TLB,MSHR,prefetching,coalescing,warp-level,address-translation,MICRO

## Finding 487: NVIDIA GPUs use a software-based dependence management mechanism (compiler-guided) rather than hardware scoreboards for instruction scheduling. Descriptors are a new memory reference encoding using two operands: a uniform register for semantics + an operand for address. Despite being claimed as Hopper innovations, Ampere already uses them. Register file caches and stream buffer instruction prefetchers are key hidden components. This compiler-hardware co-design achieves 18.24% lower MAPE vs previous simulators.
**Confidence**: high
**Source**: Analyzing Modern NVIDIA GPU cores
**Evidence**: Reverse-engineered RTX A6000 (Ampere) microarchitecture. Key findings: (1) software-based dependence management outperforms hardware scoreboards, (2) register file has an associated cache structure, (3) instruction prefetcher uses stream buffers, (4) descriptor-based memory references predate their official introduction. Achieves 13.98% MAPE vs real hardware.
**Tags**: gpu-microarchitecture,instruction-scheduling,register-file,descriptors,reverse-engineering,nvidia

## Finding 520: GhOST is a minimal out-of-order execution technique for GPUs that leverages the decode stage existing pool of decoded instructions to select instructions for OoO execution with almost no additional hardware (only 0.007 area increase). Unlike CPU OoO which needs reorder buffers and complex rename logic, GPU OoO can reuse the warp scheduler existing instruction pool. Key finding: memory instruction latency variability is the main cause of GPU stalls even with massive multithreading.
**Confidence**: high
**Source**: GhOST: a GPU Out-of-Order Scheduling Technique for Stall Reduction
**Evidence**: ISCA 2024 paper. Evaluated on NVIDIA-like architecture. Does NOT slow down any benchmark while providing speedup. The minimal hardware approach is key -- GPUs already have decoded instruction pools in the warp scheduler, GhOST just allows reordering within them. This is fundamentally different from CPU OoO because GPU has hundreds of in-flight warps providing the instruction pool naturally.
**Tags**: out-of-order,instruction-scheduling,warp-scheduler,stall-reduction,ISCA,microarchitecture

## Finding 552: FS-HPT replaces traditional multi-level Radix Page Tables with a Fixed-Size Hashed Page Table for GPU address translation. Unlike radix trees that require sequential level-by-level walks, hash tables provide O(1) expected lookup. FS-HPT uses a fixed-size hash table that never grows (evicts rarely-used PTEs instead), a step table for fast lookup, and a victim buffer for evicted entries. Results in significantly fewer memory references per page table walk than radix page tables.
**Confidence**: high
**Source**: Rethinking Page Table Structure for Fast Address Translation in GPUs: A Fixed-Size Hashed Page Table
**Evidence**: PACT 2024 paper. Key innovation: conventional GPU page tables require 4 sequential memory accesses (one per level). FS-HPT replaces this with a single hash lookup + potential collision resolution. The fixed-size constraint prevents the common problem of hash table resizing causing latency spikes. Step table acts like a cuckoo hashing secondary location.
**Tags**: page-table,hash-table,address-translation,TLB-miss,PACT,O(1)-lookup

## Finding 554: Marching Page Walks redesigns GPU page walkers to handle multiple walk requests simultaneously in batches rather than one-at-a-time. In GPU workloads, thousands of threads concurrently access many pages, but each page walker handles only one walk. The queueing latency from this mismatch is the major bottleneck for GPU page table walks. Batching exploits the fact that concurrent walks often share intermediate page table nodes, enabling amortized memory accesses across multiple walks.
**Confidence**: high
**Source**: Marching Page Walks: Batching and Concurrent Page Table Walks for Enhancing GPU Throughput
**Evidence**: HPCA 2025 paper. The key observation: when 1000+ threads simultaneously need address translation, many of them share the same level-1 and level-2 page table entries. A batched walker can fetch one shared intermediate node and resolve multiple translations simultaneously. This is a GPU-specific optimization that does not apply to CPUs (which have few concurrent walks).
**Tags**: page-table-walk,batching,concurrent,throughput,HPCA,address-translation,GPU-specific

## Finding 14: M1 family (G13, TSMC N5): M1=7-8 cores/2.3-2.6 TFLOPS/68GB/s, M1 Pro=14-16/4.6-5.3/200GB/s, M1 Max=24-32/7.8-10.4/400GB/s, M1 Ultra=48-64/15.9-21.2/800GB/s. SLC: 8/24/48/96 MB.
**Confidence**: high
**Source**: Philip Turner - metal-benchmarks
**Evidence**: TFLOPS formula: cores x 128 ALUs x 2 ops/FMA x clock_GHz. For M1 8-core: 8x128x2x1.278 = ~2.6 TFLOPS. Confirmed by metal-benchmarks.
**Tags**: m1,m1-pro,m1-max,m1-ultra,tflops,bandwidth,slc,specs

## Finding 15: M2 family (G14, TSMC N5P): M2=8-10 cores/2.9-3.6 TFLOPS/100GB/s, M2 Pro=16-19/5.7-6.8/200GB/s, M2 Max=30-38/10.7-13.6/400GB/s, M2 Ultra=60-76/21.3-27.0/800GB/s. SLC: 8/24/48/96 MB.
**Confidence**: high
**Source**: Apple Newsroom - M2 Ultra
**Evidence**: GPU clock ~1.398 GHz. For M2 10-core: 10x128x2x1.398 = ~3.6 TFLOPS. M2 Ultra 76-core is the peak G14 chip.
**Tags**: m2,m2-pro,m2-max,m2-ultra,tflops,bandwidth,slc,specs

## Finding 16: M3 family (G16, TSMC N3B): M3=8-10 cores/3.55-4.1 TFLOPS/100GB/s, M3 Pro=14-18/5.7-7.4/150GB/s, M3 Max=30-40/12.3-16.4/300-400GB/s, M3 Ultra=60-80/24.6-32.8/800GB/s.
**Confidence**: high
**Source**: Apple Newsroom - M3 Ultra
**Evidence**: M3 Ultra 80-core (32.8 TFLOPS) announced March 2025 is the most powerful single-chip Apple GPU. M3 Pro SLC dropped to 12MB (from 24MB on M1/M2 Pro).
**Tags**: m3,m3-pro,m3-max,m3-ultra,tflops,bandwidth,specs

## Finding 17: M4 family (G16, TSMC N3E): M4=8-10 cores/3.5-4.4 TFLOPS/120GB/s, M4 Pro=16-20/7.4-9.2/273GB/s, M4 Max=32-40/14.7-18.4/410-546GB/s. No M4 Ultra exists. Max unified memory: 128GB on M4 Max.
**Confidence**: verified
**Source**: Apple Newsroom - M4 Pro and M4 Max
**Evidence**: M4 Max GPU clock reaches 1.8 GHz (base 500 MHz). Memory bandwidth increased significantly: 546 GB/s on M4 Max 40-core vs 400 GB/s on M3 Max 40-core.
**Tags**: m4,m4-pro,m4-max,tflops,bandwidth,specs,no-ultra

## Finding 18: M5 base chip (Apple10, TSMC 3nm): 10-core GPU, 153.6 GB/s memory bandwidth (30% over M4), 10-core CPU. 30% faster GPU than M4, 2.5x faster than M1. 45% ray tracing uplift vs M4. 3rd-gen ray tracing engine.
**Confidence**: verified
**Source**: Apple Newsroom - M5
**Evidence**: Announced October 15, 2025. Available in MacBook Pro 14", iPad Pro, Apple Vision Pro. M5 Pro and M5 Max expected spring 2026 with modular chiplet packaging (TSMC SoIC-mH).
**Tags**: m5,apple10,specs,bandwidth,ray-tracing

## Finding 35: M5 Ultra (leaked, expected mid-2026): up to 32 CPU cores, 80-84 GPU cores, ~1100 GB/s bandwidth, ~240B transistors, estimated 600-800 TOPS AI, ~190W. Found in macOS beta code.
**Confidence**: low
**Source**: Gadget Hacks - M5 Ultra 80 GPU cores
**Evidence**: If confirmed, M5 Ultra with 80+ GPU cores x Neural Accelerators would be the most powerful Apple Silicon for AI inference. Ultra uses die-to-die stitching like M1/M2 Ultra.
**Tags**: m5-ultra,leaked,specs,gpu-cores,bandwidth,tops

## Finding 3: Each Apple GPU shader core contains 128 ALUs organized with 4 schedulers per core. Each scheduler dispatches one instruction from one SIMD group (32 threads) per cycle.
**Confidence**: high
**Source**: Philip Turner - metal-benchmarks
**Evidence**: Apple describes each GPU core as having 128 ALUs. The 4 schedulers share pipelines. FP16 operations achieve double FP32 throughput because more 16-bit ALUs exist than 32-bit ALUs.
**Tags**: core,alu,scheduler,simd,architecture

## Finding 4: Apple GPU is fully scalar at all bit sizes (16-bit and 32-bit). Unlike some GPUs that vectorize 16-bit, Apple GPU is scalar throughout but superscalar with more FP16 ALUs than FP32 ALUs.
**Confidence**: verified
**Source**: Alyssa Rosenzweig - Dissecting the M1 GPU
**Evidence**: The hardware is superscalar with more 16-bit ALUs than 32-bit ALUs. Data type conversion between 16-bit and 32-bit is free, suggesting conversion hardware sits in the standard data path.
**Tags**: scalar,fp16,fp32,alu,architecture

## Finding 5: Each SIMD group has 32 threads with: per-group program counter, stack pointer, 32-bit execution mask, up to 128 GPRs (r0-r127), and 256 uniform registers (u0-u255).
**Confidence**: verified
**Source**: Dougall Johnson - G13 Architecture Reference
**Evidence**: SIMD groups are the fundamental execution unit. GPRs are 32-bit but accessible as 16-bit halves (r0l/r0h) or 64-bit pairs. Uniform registers are shared across all threads in the group.
**Tags**: simd,registers,gpr,uniform,architecture

## Finding 25: Apple GPU has a register cache for recently-used GPRs with ISA-level cache/discard hints. Register dependency causes 0.84-cycle penalty for 32-bit, 0.56-cycle for 16-bit. This makes F16/I16 significantly faster than F32/I32 at low occupancy.
**Confidence**: high
**Source**: metal-benchmarks README
**Evidence**: At minimum occupancy: F32 dependent FMA = 11.3 cycles vs F16 = 3.9 cycles. The register cache bandwidth limitation prevents 4 schedulers from always achieving 4 IPC.
**Tags**: register-cache,dependency,fp16,fp32,performance,latency

## Finding 26: Apple GPU prioritizes SIMD shuffle bandwidth (256 B/cycle) over threadgroup memory bandwidth. simdgroup_matrix instructions map to existing FP32 ALU pipelines without dedicated tensor hardware (pre-M5). This is opposite to NVIDIA which relies heavily on shared memory.
**Confidence**: high
**Source**: Philip Turner - metal-benchmarks
**Evidence**: Apple's design encourages SIMD shuffle and matrix operations over shared memory for inter-thread communication. simdgroup_matrix FFMA throughput: ~102 ops/core-cycle for both F16 and F32.
**Tags**: simd-shuffle,threadgroup,simdgroup-matrix,design-philosophy

## Finding 30: G16 (M3/M4) enhanced ALU parallelism: FP32, FP16, and integer pipelines can execute in parallel across different SIMD groups to a greater degree than G13/G14. Delivers up to 2x ALU performance.
**Confidence**: verified
**Source**: Apple Developer - GPU advancements in M3 and A17 Pro
**Evidence**: Before Family 9, the pipelines were more constrained in overlap. The improvement comes from both Dynamic Caching (better occupancy) and wider pipeline co-issue.
**Tags**: g16,alu,fp32,fp16,integer,parallel,m3,m4

## Finding 36: Apple GPU core has 128 ALUs organized as 16 Execution Units with 8 ALUs per EU. Two pipeline categories: (1) FP32/Integer/Conditional pipeline (4 instances, 512 ALUs total) and (2) Integer/Complex Math pipeline (128 ALUs, shared SFU with 32 SFUs per core).
**Confidence**: high
**Source**: metal-benchmarks README
**Evidence**: SFUs accept one instruction per cycle but appear as 4-cycle throughput because shared among 4 schedulers. The 4 schedulers share a single 4-stage pipeline with different instruction types consuming different stages.
**Tags**: eu,alu,pipeline,sfu,core-architecture

## Finding 37: Pre-M5 simdgroup_matrix maps to existing FP32 ALU pipelines without dedicated tensor hardware. FFMA throughput: ~102 ops/core-cycle for both F16 and F32. No F64 support. Reduces register pressure and improves ALU utilization.
**Confidence**: high
**Source**: Philip Turner - metal-benchmarks
**Evidence**: This is Apple's tensor-core-equivalent before M5 added real Neural Accelerators. The instruction uses existing ALUs cooperatively across a SIMD group rather than dedicated matrix hardware.
**Tags**: simdgroup-matrix,tensor,fp16,fp32,pre-m5,alu

## Finding 38: Performance degrades significantly when shader executable exceeds ~12 KB (instruction cache size) at 92% occupancy. Apple's 12 KB icache is smaller than AMD (32 KB) and NVIDIA equivalents. Practical concern for complex compute kernels.
**Confidence**: high
**Source**: Philip Turner - metal-benchmarks
**Evidence**: This means highly complex kernels (e.g., large unrolled loops, heavy branching) can suffer icache misses. Keep kernel code compact.
**Tags**: instruction-cache,12kb,performance,kernel-size

## Finding 39: Divergent branches use execution mask stack tracked in register r0l. if_*cmp/else_*cmp/while_*cmp instructions manipulate the mask. Inactive threads' writes do not commit. SIMD shuffle can read from inactive threads (unusual).
**Confidence**: verified
**Source**: Dougall Johnson - G13 Architecture Reference
**Evidence**: jmp_exec_none/jmp_exec_any allow conditional branching based on mask state. Hardware handles reconvergence automatically via the mask stack.
**Tags**: divergence,execution-mask,branching,simd,r0l

## Finding 10: M3/M4 (Apple Family 9, G16) introduced Dynamic Caching: register file, L1 cache, threadgroup memory, and stack share a unified SRAM pool dynamically allocated at runtime. Register allocation is no longer static.
**Confidence**: verified
**Source**: Apple Developer - GPU advancements in M3 and A17 Pro
**Evidence**: Previously registers were statically allocated for a shader's entire lifetime. Now the hardware dynamically monitors and adjusts occupancy to prevent spilling. This is an industry first for GPU register management.
**Tags**: m3,m4,dynamic-caching,registers,unified-sram,family9

## Finding 27: Apple GPU is firmware-mediated: an ARM64 ASC coprocessor running RTKit firmware brokers ALL GPU operations. The macOS kernel driver never talks directly to GPU hardware. 12 work channels (4 groups x 3 types: TA/3D/CP).
**Confidence**: verified
**Source**: Asahi Linux - Tales of the M1 GPU
**Evidence**: Firmware interface uses ~1000-field shared memory structures. Two command processors: VDM (vertex/tiling) and CDM (compute). Firmware handles scheduling, power management, preemption, fault recovery.
**Tags**: firmware,asc,rtkit,cdm,vdm,command-processor

## Finding 553: NDPage introduces two key optimizations for page tables in memory-constrained processors: (1) L1 cache bypass for Page Table Entries (PTEs) that prevents cache pollution from irregular PTE accesses, and (2) a flattened page table that merges the last two levels, cutting accesses while maintaining 4KB page flexibility. Improves performance 14.3% (1-core) to 30.5% (8-core). Key insight: PTE access patterns are highly irregular and pollute data caches.
**Confidence**: medium
**Source**: NDPage: Efficient Address Translation for Near-Data Processing Architectures via Tailored Page Table
**Evidence**: arXiv 2025 paper. The flattened design merges level 3 and level 4 of the page table into a single level, reducing walks from 4 to 3 memory accesses. PTE bypass prevents small, scattered PTE reads from evicting useful data from L1. The benefit scales with core count because more cores means more concurrent page walks competing for cache space.
**Tags**: page-table,cache-bypass,flattened-page-table,near-data-processing,address-translation

## Finding 7: Key instruction latencies: FP16 FMA=1 cycle throughput/~2.2 latency, FP32 FMA=2 cycles throughput/~2.2 latency, RECIP=6 cycles, RSQRT=8 cycles, SIN/COS=10 cycles, INT32 multiply=32 cycles (very slow).
**Confidence**: high
**Source**: metal-benchmarks README
**Evidence**: EXP2/LOG2 take 4 cycles via complex pipeline. Register dependency adds 0.84-cycle penalty for 32-bit, 0.56 for 16-bit. At minimum occupancy: F32 dependent FMA = 11.3 cycles, F16 = 3.9 cycles.
**Tags**: latency,throughput,fp32,fp16,int32,performance

## Finding 9: Per-core on-chip memory: register file ~208KB, instruction cache 12KB, L1 data cache 8KB, texture cache ~24KB, shared/threadgroup memory ~60KB. On-core data bandwidth 64 B/cycle.
**Confidence**: high
**Source**: Philip Turner - metal-benchmarks
**Evidence**: L1 data cache (8KB) is very small vs AMD (16KB) and NVIDIA. Compensated by low-latency L2 and high SIMD shuffle bandwidth (256 B/cycle). 12KB instruction cache is also small - performance degrades for shaders >12KB.
**Tags**: memory,cache,l1,register-file,threadgroup,per-core

## Finding 20: GPU memory latencies (M2 Pro, measured): L1/L2 hit ~27ns (~80 GPU ticks), SLC hit ~234ns (~220 ticks), DRAM ~342+ns (~430 ticks). Atomic on global memory: ~58.57ns. Cache line size: 128 bytes for L2 and SLC.
**Confidence**: high
**Source**: Chips and Cheese - M2 Pro iGPU
**Evidence**: SLC is exclusive wrt CPU caches, inclusive wrt GPU caches. Replacement policy is pseudo-random (not LRU). GPU hides latency through massive thread parallelism.
**Tags**: latency,cache,slc,dram,memory-hierarchy,m2-pro

## Finding 21: SLC (System Level Cache) sizes: M1/M2/M3/M4 base=8MB, M1/M2 Pro=24MB, M3 Pro=12MB (reduced!), M1/M2/M3 Max=48MB, Ultra variants=96MB. SLC is shared by CPU, GPU, NPU, and all SoC components.
**Confidence**: high
**Source**: EXAM paper - SLC characterization
**Evidence**: M3 Pro breaking the 24MB pattern at 12MB is notable. M4 Pro/Max SLC not publicly confirmed but estimated at ~24/48MB. A19 Pro SLC is 32MB (confirmed).
**Tags**: slc,cache,memory,m1,m2,m3,m4

## Finding 22: On Apple GPUs, threadgroup memory is physically backed by the same on-chip tile memory SRAM used for TBDR rendering. 32KB API limit per threadgroup, ~60KB hardware capacity per core. Tile memory NOT persistent across compute dispatches.
**Confidence**: verified
**Source**: WWDC20: Optimize Metal Performance for Apple silicon
**Evidence**: During compute, TBDR tile memory is repurposed as threadgroup memory. Apple intentionally provides lower threadgroup memory bandwidth than competitors, instead investing in SIMD shuffle (256 B/cycle).
**Tags**: threadgroup,tile-memory,tbdr,sram,compute

## Finding 23: Apple GPU shader core can hoist constant code and prefetch constant data once per draw/dispatch. No explicit software prefetch instructions exposed through Metal. Uniform registers (u0-u255) preload buffer pointers.
**Confidence**: medium
**Source**: WWDC20: Optimize Metal Performance for Apple silicon
**Evidence**: Using Metal 'constant' address space allows hardware to preload data. No documentation on prefetch depth, stream detection, or stride prefetch for non-constant data. This is a major unknown.
**Tags**: prefetch,constant,uniform-registers,memory

## Finding 24: Measured GPU DRAM bandwidth efficiency: M1=60/68GB/s (90%), M2=91/100GB/s (91%), M3=92/100GB/s (92%), M4=100/120GB/s (83%). GPU achieves 83-92% of theoretical peak bandwidth.
**Confidence**: verified
**Source**: Apple vs Oranges: HPC on Apple Silicon
**Evidence**: From STREAM benchmark on GPU. M4 shows lower efficiency (83%) possibly due to increased clock/bandwidth mismatch. Pro/Max variants not tested in this paper.
**Tags**: bandwidth,dram,efficiency,m1,m2,m3,m4,measured

## Finding 28: All memory is coherent between CPU and GPU on Apple Silicon - no explicit cache flushing required. Asahi team confirms: "We haven't used a single cache management instruction yet and everything still works."
**Confidence**: verified
**Source**: Asahi Linux - Tales of the M1 GPU
**Evidence**: GPU MMU (UAT - Unified Address Translator) uses ARM64-identical page tables. 40-bit GPU virtual addresses, 16K page size, up to 16 GPU contexts.
**Tags**: coherence,cpu-gpu,uat,mmu,unified-memory

## Finding 40: GPU cores have identical I/O bus width to CPU cores (32 bytes/cycle). GPU has no per-core bandwidth advantage over CPU for memory-bound tasks. GPU advantage comes from compute parallelism, not per-core memory bandwidth.
**Confidence**: medium
**Source**: Philip Turner - metal-benchmarks
**Evidence**: This is a significant architectural distinction from discrete GPUs where GPU memory bandwidth per SM far exceeds CPU core bandwidth. On Apple Silicon, both access the same unified memory at the same per-core rate.
**Tags**: bandwidth,cpu-gpu,io-bus,per-core,unified-memory

## Finding 41: Apple GPU memory coalescing: 32 threads accessing consecutive 4-byte values = single 128-byte transaction (perfect coalescing). device_load can unpack various formats (8/16/32-bit, rgb10a2, unorm8) during load. FP16 data: 2B x 32 threads = 64 bytes, more efficient.
**Confidence**: medium
**Source**: Dougall Johnson - G13 Architecture Reference
**Evidence**: Strided/scattered access = multiple cache line fetches. Apple uses scalar arch with vectorized I/O. Hardware rounds unaligned addresses down. Threadgroup memory accesses should be aligned to 16 bytes.
**Tags**: coalescing,cache-line,128-byte,memory-access,device-load

## Finding 42: M2 Pro GPU L2 cache delivers over 1 TB/s bandwidth (measured by Chips and Cheese). L2 latency is notably lower than AMD equivalent, compensating for the smaller 8KB L1. Texture cache ~24KB per core (separate from data L1).
**Confidence**: high
**Source**: Chips and Cheese - M2 Pro iGPU
**Evidence**: The low L1 size (8KB) is a deliberate power efficiency trade. Apple compensates with extremely low-latency L2. L1 and L2 show similar latency (~27ns/~80 ticks) on M2 Pro.
**Tags**: l2,bandwidth,1tb,m2-pro,texture-cache,latency

## Finding 43: SLC (System Level Cache) properties: 128-byte cache line, exclusive wrt CPU caches, inclusive wrt GPU caches, pseudo-random replacement (not LRU), uses physical address bits from position 14+. Atomic ops on global memory: 58.57ns on M2 Pro.
**Confidence**: verified
**Source**: EXAM paper - SLC characterization
**Evidence**: SLC indexing skips lowest 13 address bits. Discrimination thresholds: 160 ticks for L2/SLC boundary, 300 ticks for SLC/DRAM boundary. SLC is shared by CPU, GPU, NPU, media engine, and all SoC components.
**Tags**: slc,cache-properties,exclusive,inclusive,replacement,atomics

## Finding 44: Copying device memory to threadgroup memory as "software cache" (common CUDA pattern) can be SLOWER on Apple Silicon. Due to unified memory and effective L2 cache, reading directly from device/constant buffers may be faster on Apple Family 9+.
**Confidence**: verified
**Source**: WWDC20: Optimize Metal Performance for Apple silicon
**Evidence**: Apple WWDC guidance recommends trying direct reads first. The unified memory architecture means the CPU/GPU don't have separate address spaces, so the extra copy to threadgroup memory adds overhead without the benefit it provides on discrete GPUs.
**Tags**: threadgroup,cuda-pattern,software-cache,optimization,unified-memory

## Finding 120: Philip Turner metal-benchmarks documents Apple GPU per-core memory specs: On-core data bandwidth 64B/cycle, SIMD shuffle bandwidth 256B/cycle (much higher than competitors), shared memory ~60KB per core, instruction cache 12KB per core, data cache 8KB per core, register file ~208KB per core. System-level last-level cache to RAM ratio: ~7.7-9.9 B/cycle for LPDDR5 chips. On-GPU data transfer rate ~32B/cycle. Critical insight: "GPU cores have no advantage over the CPU on bandwidth-bound tasks since its I/O bus width is the same (32 bytes)." GPU advantage comes from computation, not raw bandwidth.
**Confidence**: high
**Source**: Philip Turner: metal-benchmarks
**Evidence**: Philip Turner metal-benchmarks repository README provides per-core specifications across Apple GPU generations. The statement about GPU I/O bus width equaling CPU is significant for understanding when .private vs .shared matters.
**Tags**: memory-bandwidth,per-core,register-file,cache,SIMD-shuffle

## Finding 19: M5 embeds a dedicated Neural Accelerator in each GPU core. Delivers 4x peak GPU AI compute vs M4 and 6x vs M1. Doubles FP16 throughput. Programmable via Metal 4 Tensor APIs.
**Confidence**: verified
**Source**: Apple ML Research - Exploring LLMs with MLX and M5
**Evidence**: This is the biggest architectural change since G16. Neural accelerators perform tensor/matrix operations directly in the GPU pipeline. Developers access them through Metal 4's cooperative tensor APIs and MTLTensor.
**Tags**: m5,neural-accelerator,tensor,metal4,ai,fp16

## Finding 31: M5/A19 Neural Accelerator per-core specs: 128 matrix FMAs per partition per clock. Supports FP16 inputs with FP16/FP32 accumulators, INT8 with INT32 accumulators. Optimal tile size 32x32. NO bfloat16, NO FP8/FP4, NO sparsity support.
**Confidence**: high
**Source**: Taras Zakharko - Investigating GPU Neural Accelerators on A19/M5
**Evidence**: On A19 (5 GPU cores, ~1460 MHz): FP16 matmul ~7.4 TFLOPS, INT8 ~13.4 TOPS, regular FP32 SIMD 1.88 TFLOPS. Matrix transpose at zero cost (per patent WO2025071810). Comparable to NVIDIA Turing tensor cores. Multi-SIMD-group mode showed performance degradation.
**Tags**: m5,a19,neural-accelerator,tensor,fp16,int8,matmul,benchmark

## Finding 32: M5 Neural Accelerators are NOT directly exposed in MSL. Accessed through Metal Performance Primitives (MPP), Metal Tensor APIs (Metal 4), and Shader ML. MLX uses TensorOps and MPP. Higher-level than CUDA's direct tensor core access.
**Confidence**: verified
**Source**: WWDC25 - Combine Metal 4 ML and graphics
**Evidence**: Developers program via Tensor APIs in Metal 4 and Shader ML (embed neural networks directly in shaders). Thread-level, SIMD-group, and multi-SIMD-group execution scopes available.
**Tags**: m5,neural-accelerator,metal4,mpp,shader-ml,api,programming

## Finding 33: M5 LLM inference via MLX: time-to-first-token 3.3-4x faster than M4 (compute-bound, uses Neural Accelerators). Token generation only 1.19-1.27x faster (bandwidth-limited at 153.6 vs 120 GB/s). Qwen 14B 4-bit: 4.06x TTFT speedup.
**Confidence**: verified
**Source**: Apple ML Research - Exploring LLMs with MLX and M5
**Evidence**: TTFT improvements directly exploit Neural Accelerators. Token generation is memory-bandwidth-bound and only sees the ~28% bandwidth improvement. M5 Max (with presumably ~546+ GB/s) will be far more impactful for serving LLMs.
**Tags**: m5,mlx,llm,inference,ttft,bandwidth,neural-accelerator

## Finding 546: Neural graphics applications have a 55.5x performance gap vs desired 4K@60fps on current GPUs. Input encoding and MLP kernels consume ~60% of execution time. A dedicated Neural Graphics Processing Cluster (NGPC) with specialized engines achieves 58x end-to-end improvement, enabling 4K@30fps NeRF and 8K@120fps for other neural graphics.
**Confidence**: high
**Source**: Hardware Acceleration of Neural Graphics
**Evidence**: Comprehensive analysis of neural graphics workloads including NeRF, neural SDF, neural images, neural volumes. Bottleneck analysis: MLP and encoding kernels dominate across all workloads. NGPC adds specialized hash-encoding engine and MLP engine to existing GPU core design. Scalable architecture evaluated across multiple neural graphics applications.
**Tags**: neural-graphics,hardware-accelerator,NGPC,MLP-engine,hash-encoding,NeRF,performance-analysis

## Finding 1: Apple GPU generations map to chips: G13 (A14/M1), G14 (A15/A16/M2), G15 (cancelled), G16 (A17Pro/M3/M4). G15 was cancelled due to thermal issues during A16 development.
**Confidence**: medium
**Source**: NamuWiki Apple Microarchitecture
**Evidence**: G15 was under development for A16 Bionic but cancelled. A16 reused G14. The architecture that succeeded was designated G16, appearing first in A17 Pro and M3. Both M3 and M4 share G16/Apple Family 9.
**Tags**: g13,g14,g15,g16,generations,architecture

## Finding 2: M5 introduces Apple GPU Family 10 (Apple10) architecture with per-core Neural Accelerators. This is a new GPU generation succeeding G16/Apple9.
**Confidence**: verified
**Source**: Apple Newsroom - M5 Announcement
**Evidence**: Apple10 doubles FP16 throughput and introduces per-core neural accelerators for tensor/matrix operations. Programmable via Metal 4 Tensor APIs. 4x peak GPU AI compute vs M4.
**Tags**: m5,apple10,neural-accelerator,metal4,generation

## Finding 11: G14 (M2 family) vs G13 (M1 family): ~8% higher GPU clock (~1.398 GHz vs ~1.296 GHz), LPDDR5 vs LPDDR4X, slightly larger L2 (~1.5MB vs 768KB on base). No new hardware features - no ray tracing, no mesh shading.
**Confidence**: high
**Source**: Philip Turner - metal-benchmarks
**Evidence**: The per-core TFLOPS increase from G13 to G14 is entirely from clock speed. A15 and A16 both used G14 with minimal architectural change between them. Almost no architectural change within G14 generation.
**Tags**: g13,g14,m1,m2,clock-speed,evolution

## Finding 12: G16 (M3/M4) introduced 3 major GPU architecture changes: (1) Dynamic Caching for unified on-chip memory, (2) Hardware-accelerated ray tracing with dedicated intersection units, (3) Hardware-accelerated mesh shading.
**Confidence**: verified
**Source**: Apple Newsroom - M3 Announcement
**Evidence**: G16 is a major redesign. Dynamic Caching alone provides up to 2x ALU performance by allowing FP32, FP16, and integer pipelines to execute in greater parallelism. G16 first appeared in A17 Pro, then M3 and M4 families.
**Tags**: g16,m3,m4,dynamic-caching,ray-tracing,mesh-shading,family9

## Finding 13: M4 uses same G16 ISA as M3 (both Apple Family 9). M4 improvements: 2x faster ray tracing engine, higher GPU clocks (up to 1.8 GHz on Max vs ~1.4 GHz on M3 Max), TSMC N3E vs N3B process.
**Confidence**: verified
**Source**: Apple Newsroom - M4 Pro and M4 Max
**Evidence**: M4 is NOT a new GPU architecture generation. Per-core TFLOPS increased from ~0.41 (M3) to ~0.46 (M4) mainly from clock speed. No M4 Ultra was produced - Apple skipped it.
**Tags**: m4,m3,g16,ray-tracing,n3e,same-isa

## Finding 29: M5 Pro and M5 Max rumored to use TSMC SoIC-mH modular chiplet packaging with separate CPU and GPU blocks. Expected spring 2026. M5 Max may surpass M3 Ultra performance.
**Confidence**: low
**Source**: Apple Newsroom + rumors
**Evidence**: This would be a major packaging change from monolithic dies. M5 Pro expected to match or exceed M4 Max multi-core performance. 30-40% more powerful GPU than M4-series predecessors estimated.
**Tags**: m5-pro,m5-max,soic,chiplet,modular,packaging

## Finding 34: M5 Pro/Max reportedly use TSMC SoIC-mH (2.5D chiplet) with separate CPU and GPU silicon blocks on common interposer. Departure from M1-M4 monolithic dies. M5 Ultra may use full 3D-stacked SoIC.
**Confidence**: medium
**Source**: SiliconAngle - M5 N3P and 2.5D packaging
**Evidence**: CPU and GPU on separate dies allows independent optimization and more flexible configurations. Supply chain leaks suggest this, not Apple-confirmed. M5 Pro/Max expected spring 2026.
**Tags**: m5-pro,m5-max,m5-ultra,soic,chiplet,2.5d,packaging

## Finding 6: Apple GPU register file is ~208 KB per core. At 104 registers: max 1024 threads. Between 104-256 registers: decreases in 64-thread steps. At 256 registers: minimum 384 threads.
**Confidence**: verified
**Source**: Alyssa Rosenzweig - Dissecting the M1 GPU Part III
**Evidence**: Using 256 half-word registers, the machine still supports 384 threads. To support 1024 threads at 104 registers requires exactly 208 KiB. Register blocks are allocated in granular chunks (believed 8 registers).
**Tags**: registers,occupancy,register-file,208kb

## Finding 535: Heliostat repurposes underutilized ray tracing accelerators (RTAs) to accelerate GPU page table walks by exploiting operational similarities between ray-BVH traversal and page table tree walking. Achieves 1.93x speedup over baseline GPU MMU while using only 1.53% of the area and 5.8% of the power of equivalent dedicated hardware. Heliostat+ adds further 1.23x improvement. The key insight: RT cores are tree traversal engines, and page tables are trees.
**Confidence**: high
**Source**: Heliostat: Harnessing Ray Tracing Accelerators for Page Table Walks
**Evidence**: ISCA 2025 paper. RTAs are idle during compute-heavy phases. Page table walk is tree traversal (like BVH). By encoding page table levels as BVH nodes, existing RT hardware can perform parallel page walks. Outperforms 128-PTW baseline while using fraction of the silicon.
**Tags**: ray-tracing,page-table-walk,address-translation,hardware-repurposing,BVH,ISCA

## Finding 8: Apple GPU supports 3 dispatch modes: single-dispatch from 3 SIMDs (rare, 16-bit only), dual-dispatch from 2 SIMDs (preferred at low occupancy), quad-dispatch from 1 SIMD (high-ILP kernels). Heritage from PowerVR.
**Confidence**: high
**Source**: metal-benchmarks README
**Evidence**: Dual-dispatching is preferred at low occupancy and required to fully utilize FP16/I16. The complex pipeline runs one 32-wide instruction/simd every 4 cycles. Before M1, F32 could only execute at 2 IPC.
**Tags**: scheduling,dispatch,dual-dispatch,powervr,simd

## Finding 536: Avatar achieves 90.3% speculation accuracy for GPU address translation by monitoring physical page contiguity. CAST (Contiguity-Aware Speculative Translation) predicts PA from VA based on observed contiguous page mappings. CAVA (In-Cache Validation) embeds page mapping info in each 32B sector of cache lines for rapid validation of speculated addresses. Combined, improves GPU performance by 37.2% average by hiding TLB miss latency.
**Confidence**: high
**Source**: A Case for Speculative Address Translation with Rapid Validation for GPUs
**Evidence**: MICRO 2024 paper. Two-component system: CAST monitors contiguous physical page allocations (common in GPU workloads because bulk allocation) and extrapolates. CAVA validates by storing mapping metadata in unused cache line bits. If speculation is wrong, fall back to normal page walk. 90% accuracy means only 10% pay the full walk latency.
**Tags**: speculative-translation,TLB,contiguity,cache-validation,MICRO,address-translation

## Finding 519: Revelator uses OS-level tiered hash-based allocation to create predictable VA-to-PA mappings, enabling hardware to speculatively predict physical addresses BEFORE translation completes. Achieves 27% speedup in native settings with only 0.01% area overhead and 0.02% power overhead. Key innovation: instead of complex hardware TLB hierarchies, make the OS allocator cooperate by placing pages predictably so simple hash functions can guess PA from VA.
**Confidence**: high
**Source**: Revelator: Rapid Data Fetching via OS-Driven Hash-based Speculative Address Translation
**Evidence**: MICRO 2025 paper. Hardware-OS cooperative scheme. Tiered allocation: first try hash-based placement, fall back to conventional if needed. Prediction accuracy high enough to justify speculative data fetch. Outperforms Avatar (prior speculative translation) by 5%. Reduces energy by 9%.
**Tags**: address-translation,speculative,hash-allocation,TLB,OS-cooperative,page-table

