---
name: gpu-centric-arch
layer: 4
description: >
  This skill should be used when the user asks about GPU-centric computing paradigms, reverse offloading, persistent kernels, GPU databases, GPU filesystems, GPU operating systems, GPU scheduling, megakernels, task graph execution, GPU-driven rendering, or treating the GPU as the primary compute engine. Trigger keywords: "GPU-centric", "reverse offloading", "persistent kernels", "GPU database", "GPU filesystem", "LithOS", "GPU OS", "GPU-as-CPU", "GPU scheduling", "megakernel", "GPU-driven", "task graph", "work graphs", "GPU packet processing", "GPU data processing", "roofline model", "GPU memory allocator", "GPU font rendering", "persistent threads", "indirect command buffers", "kernel atomization".
related_skills:
  - gpu-silicon
  - unified-memory
  - metal-compute
  - gpu-perf
  - gpu-io
  - gpu-distributed
---

# GPU-Centric Architecture

GPU-centric architecture inverts the traditional CPU-centric computing model: the GPU becomes the primary compute engine orchestrating work, while the CPU serves as a helper for tasks GPUs handle poorly. This paradigm shift is driven by the widening gap between GPU and CPU throughput, the inefficiency of CPU-mediated GPU dispatch, and the emergence of workloads (ML inference, data processing, rendering) where GPUs dominate execution time.

On Apple Silicon, unified memory architecture (UMA) uniquely enables GPU-centric patterns. Zero-copy data sharing, pointer passing between CPU and GPU, and the absence of PCIe transfer overhead make GPU-as-primary-compute more practical than on discrete GPU systems. Metal's indirect command buffers, Metal 4's unified encoders, and MPSGraph's multi-device execution provide the API building blocks.

This skill covers the full GPU-centric stack: persistent kernel architectures for long-running GPU programs, megakernel compilation for eliminating dispatch overhead, GPU-side scheduling and OS abstractions, GPU data processing (databases, JSON parsing, sorting), GPU filesystem and networking access, task graph execution for GPU-driven work scheduling, and real-world architectures like llama.cpp and MLX that demonstrate these patterns on Apple Silicon.

## Domain Overview

**Persistent Kernel Architecture**: Persistent threads (PT) programming dispatches a single kernel that fully occupies the GPU and processes work from shared queues indefinitely, eliminating per-kernel launch overhead (#291, #438). Work distribution uses distributed task-stealing/donation patterns with lock-free queues (#439). Metal does not natively support persistent kernels due to a GPU command buffer timeout/watchdog that kills long-running shaders (#440, #441). Metal 3+ indirect command buffers and Metal 4 unified compute encoders provide partial workarounds by consolidating dispatches (#442, #301). DX12 Work Graphs (2024) represent the state of the art: shader threads spawn new workloads dynamically without CPU intervention, achieving 3.35x mean speedup for SpMV (#294, #296, #352, #452).

**Megakernel Architecture**: Hazy Research (Stanford, May 2025) fused an entire Llama-1B forward pass (~100 ops) into a single GPU kernel, eliminating all inter-kernel launch overhead and memory round-trips (#447). This extended to Llama-70B with tensor parallelism across H100 GPUs (#448). Luminal compiler compiles entire ML models into single megakernels via expression graph optimization (#449). However, NVIDIA's 2013 "Megakernels Considered Harmful" showed monolithic kernels suffer from register pressure, branch divergence, and poor occupancy (#450). GPU dispatch overhead benchmarks: CUDA kernel launch costs 2.1us via streams, 1.3us via CUDA Graphs (#451). The tradeoff is between dispatch overhead elimination and occupancy loss from register pressure.

**GPU Scheduling**: GPREEMPT (ATC'25) achieves within 40 microsecond preemption latency using timeslice-based GPU multitasking (#443, #298). GPU spatial multitasking divides SMs into disjoint subsets per task; SMK (simultaneous multikernel) shares SMs across tasks (#444). XSched (OSDI'25) provides preemptive scheduling across GPUs, NPUs, ASICs, and FPGAs via a unified abstraction (#445). Cooperative Kernels (FSE'17) extend GPU programming for blocking algorithms using yield/wait primitives (#446). On heterogeneous SoCs with CPU/GPU/NPU, operator-accelerator affinity and asymmetric scheduling optimize LLM agent workloads (#494). GPU+NPU cooperative inference on mobile SoCs achieves better throughput than either alone (#511).

**GPU-Centric OS**: LithOS (SOSP'25, CMU) is a GPU OS with TPC Scheduler for spatial scheduling at SM granularity and kernel atomization for transparent preemption (#459, #510). LithOS reduces tail latencies 13x vs NVIDIA MPS, 12x vs Orion for inference stacks (#460). gpu_ext (2025) treats the GPU driver as a programmable OS subsystem using eBPF, enabling custom scheduling policies without kernel modifications (#461). GPUs must evolve from singletasking to multitasking with OS-like resource management as hardware grows (#550).

**GPU Data Processing**: GPU databases use columnar storage, JIT compilation, and tile-based execution (#311). Crystal+ achieves 1.97x over HeavyDB on SSB, 17.66x over TQP (#470). RAPIDS cuDF achieves 150x over pandas with zero code changes via GPUDirect Storage (#315, #471). GPU sort algorithms achieve 50-103x speedup: radix sort fastest for numeric types (#321). GpJSON (VLDB 2025) achieves 2.9x over state-of-art parallel JSON parsers (#472). GPU encryption achieves 878.6 Gbps AES throughput, 2.56x faster than best CPU (#334).

**GPU Filesystem & Networking**: GPUfs (ASPLOS 2013) provides POSIX-like filesystem API for GPU programs, 7x faster than CPU-mediated I/O (#339). BaM enables GPU-initiated direct SSD access without CPU, 25x performance increase (#343). GPU packet processing achieves 580 Gbps for IP lookups via NVIDIA DOCA GPUNetIO (#325). DPDK GPUdev enables persistent kernel-based packet processing with GPU kernel busy-wait patterns (#473). GPU string matching via HybridSA (OOPSLA 2024) enables efficient multi-pattern regex on GPU (#347).

**Task Graph Execution**: DX12 Work Graphs enable GPU-side DAG execution where shaders spawn new workloads and scheduler manages execution (#352, #452). Metal Indirect Command Buffers (ICBs) are Apple's closest equivalent, enabling GPU-driven command generation with 39-44% speedup (#453, #301). MPSGraph builds, compiles, and executes DAGs across GPU, CPU, and Neural Engine (#454, #356). Kitsune enables dataflow execution on GPUs via software-only task queue and fine-grained synchronization (#489).

**GPU-Driven Rendering**: Metal 3 mesh shaders with Object Shaders process abstract work collections and can amplify/cull geometry on GPU (#467). Metal 4 introduces long-lived command buffers with command allocators and unified encoders (#468). M2 has hardware acceleration for Nanite-style 32-bit buffer atomics (#469). Nanite uses GPU-driven virtualized geometry with 128-triangle clusters and hierarchical LOD (#304).

**GPU Memory Allocators**: SlabAlloc achieves 600M allocations/sec on GPU, 37x faster than Halloc, using warp-cooperative allocation (#455). DynaSOAr (ECOOP 2019) is a fully-parallel lock-free GPU memory allocator (#456). GPU lock-free allocator patterns: warp-synchronous allocation with one thread per warp doing the actual alloc (#457). Parallel prefix sum (scan) is fundamental for GPU dynamic allocation (#458).

**Roofline Model**: Apple Silicon GPU roofline: performance bound = min(bandwidth*I, peak_compute) where I is arithmetic intensity (#476). STREAM benchmarks: M1=60GB/s, M2=91GB/s, M3=92GB/s, M4=100GB/s achieved bandwidth (#477). LLM inference: prompt processing (batched matmul) is compute-bound; token generation is memory-bandwidth-bound (#478).

**Real-World Architectures**: llama.cpp Metal backend organizes kernels into 4 categories: quantization dequantization, matrix multiplication, element-wise operations, and specialized operations (#462). Uses 3 buffer allocation strategies: Shared, Private, and Mapped (#463). Two-level pipeline naming with base function + instance names (#464). On M4 (10 GPU cores): 221 tok/s prompt processing, 24 tok/s text generation (#465). Operation fusion combines normalization+scale, addition chains, and broadcast operations (#466). MLX: GPU-first lazy evaluation framework with unified memory and deferred computation (#379).

**Reverse Offloading**: HEFT algorithm and prediction-based schedulers optimize CPU/GPU task placement (#360). Reverse offloading: dynamic reassignment from GPU back to CPU when hardware cache contention makes CPU more efficient (#365). UMA eliminates GPU-centric design barriers: zero-copy data sharing, pointer passing between CPU and GPU (#368).

## Key Knowledge Areas

The knowledge base contains **70+ findings** covering:

- **Persistent Kernel Architecture** (8 findings): Persistent threads, work queues, Metal timeout constraints, Metal 4 unified encoders, DX12 Work Graphs, SpMV speedups (#291, #294, #296, #298, #438-#442)
- **Megakernel Architecture** (5 findings): Llama-1B/70B megakernels, Luminal compiler, register pressure tradeoffs, dispatch overhead benchmarks (#447-#451)
- **GPU Scheduling** (6 findings): GPREEMPT preemption, spatial multitasking, XSched, Cooperative Kernels, heterogeneous SoC scheduling (#443-#446, #494, #511)
- **GPU-Centric OS** (4 findings): LithOS kernel atomization, tail latency improvements, gpu_ext eBPF scheduling, singletasking limitations (#459-#461, #510, #550)
- **GPU Data Processing** (6 findings): GPU databases, cuDF, GPU sorting, GpJSON, GPU encryption (#311, #315, #321, #334, #470-#472)
- **GPU Filesystem & Networking** (5 findings): GPUfs, BaM direct SSD access, GPU packet processing, DPDK GPUdev, string matching (#325, #330, #339, #343, #347, #473)
- **Task Graph Execution** (5 findings): DX12 Work Graphs, Metal ICBs, MPSGraph, Kitsune dataflow (#352, #356, #452-#454, #489)
- **GPU-Driven Rendering** (4 findings): Metal mesh shaders, Metal 4 command model, Nanite, hardware atomics (#301, #304, #467-#469)
- **GPU Memory Allocators** (4 findings): SlabAlloc, DynaSOAr, warp-synchronous patterns, prefix sum allocation (#455-#458)
- **Roofline Model** (3 findings): Apple Silicon roofline, STREAM benchmarks, LLM inference bottlenecks (#476-#478)
- **Real-World Architectures** (7 findings): llama.cpp Metal backend, MLX framework, buffer strategies, operation fusion (#370, #376, #379, #462-#466)
- **Reverse Offloading & CPU/GPU Distribution** (3 findings): HEFT scheduling, reverse offloading, UMA advantages (#360, #365, #368)
- **GPU Font Rendering** (2 findings): Slug algorithm, Vello 177 fps on M1 Max (#474, #475)

## How to Query

Use the knowledge base CLI to retrieve gpu-centric-arch findings:

```bash
# Get all gpu-centric-arch findings
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill gpu-centric-arch

# Search by topic
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "persistent kernel"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "megakernel"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "GPU scheduling"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "LithOS"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "GPU database"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "GPU filesystem"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "work graphs"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "llama.cpp Metal"

# Get specific finding details
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <id>

# Search across all skills
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "GPU-centric"
```

Query strategies by problem type:
- **Persistent compute**: Search "persistent kernel", "persistent threads", "work queue"
- **Dispatch overhead**: Search "megakernel", "kernel fusion", "dispatch overhead"
- **Multi-tenant GPU**: Search "GPU scheduling", "preemption", "spatial multitasking", "LithOS"
- **Data processing**: Search "GPU database", "cuDF", "GPU sort", "JSON parsing"
- **Storage access**: Search "GPU filesystem", "GPUfs", "BaM", "direct SSD"
- **Network processing**: Search "GPU packet", "GPUNetIO", "DPDK"
- **Work orchestration**: Search "work graphs", "task graph", "indirect command buffer", "MPSGraph"
- **Apple Silicon specifics**: Search "Metal ICB", "Metal 4", "llama.cpp Metal", "MLX"
- **Performance modeling**: Search "roofline", "STREAM benchmark", "arithmetic intensity"

## Common Patterns & Quick Answers

**Q: What is GPU-centric computing and why does it matter?**
A: GPU-centric computing treats the GPU as the primary compute engine, with the CPU as a helper for tasks GPUs handle poorly. This eliminates CPU-mediated dispatch overhead, reduces memory round-trips, and keeps data resident on the GPU. On Apple Silicon, UMA makes this especially practical: zero-copy data sharing and pointer passing between CPU and GPU remove the PCIe bottleneck that plagues discrete GPU systems (#368). Real-world examples include llama.cpp achieving 221 tok/s prompt processing on M4 (#465) and megakernels fusing entire model forward passes (#447).

**Q: How do persistent kernels work, and can I use them on Metal?**
A: Persistent threads dispatch a single kernel that fully occupies the GPU and processes work from shared queues indefinitely, eliminating per-kernel launch overhead (#291, #438). Work uses distributed task-stealing with lock-free queues (#439). Metal does NOT natively support persistent kernels — a GPU command buffer timeout/watchdog kills long-running shaders (#440, #441). Workarounds: Metal 3+ indirect command buffers for GPU-driven dispatch (#301, #453), Metal 4 unified compute encoders for consolidating dispatches (#442, #468), and MPSGraph for automatic DAG execution (#454).

**Q: What are megakernels and when should I use them?**
A: Megakernels fuse many operations into a single GPU kernel. Hazy Research fused entire Llama-1B forward pass (~100 ops) into one kernel, eliminating all inter-kernel overhead (#447), extended to Llama-70B with tensor parallelism (#448). Luminal compiles entire ML models into single megakernels (#449). However, megakernels risk register pressure, branch divergence, and poor occupancy (#450). They're most beneficial when dispatch overhead dominates (CUDA: 2.1us per launch via streams, #451). For Apple Silicon, MLX's mx.compile provides automatic kernel fusion (#379).

**Q: How does GPU scheduling work for multi-tenant workloads?**
A: GPREEMPT achieves within 40us preemption latency using timeslice-based multitasking (#443). Spatial multitasking divides SMs per task; SMK shares SMs across tasks (#444). LithOS (SOSP'25) provides kernel atomization for transparent preemption and TPC scheduling at SM granularity, reducing tail latencies 13x vs NVIDIA MPS (#459, #460, #510). gpu_ext uses eBPF for custom scheduling policies without kernel modifications (#461). XSched provides a unified preemptive scheduling abstraction across GPUs, NPUs, and FPGAs (#445).

**Q: What GPU data processing capabilities exist?**
A: GPU databases use columnar storage and JIT compilation (#311). Crystal+ achieves 1.97x over HeavyDB (#470). RAPIDS cuDF gives 150x speedup over pandas with zero code changes (#315, #471). GPU sorting achieves 50-103x speedup with radix sort fastest for numerics (#321). GpJSON parses JSON 2.9x faster than best parallel CPU parsers (#472). GPU AES encryption hits 878.6 Gbps, 2.56x faster than CPU (#334). For Apple Silicon, Metal compute shaders can implement these patterns using threadgroup memory and SIMD operations.

**Q: Can GPUs access filesystems and network directly?**
A: GPUfs provides POSIX-like filesystem API for GPU programs, 7x faster than CPU-mediated I/O (#339). BaM enables GPU-initiated direct SSD access without CPU involvement, 25x performance gain (#343). For networking, GPU packet processing achieves 580 Gbps for IP lookups (#325). DPDK GPUdev enables persistent kernel-based packet processing (#473). On Apple Silicon, Metal Fast Resource Loading (MTLIOCommandQueue) provides async SSD-to-GPU streaming, and mmap + makeBuffer(bytesNoCopy:) enables zero-copy file-mapped GPU access — see gpu-io skill.

**Q: How do task graphs enable GPU-driven work scheduling?**
A: DX12 Work Graphs let shader threads spawn new workloads dynamically without CPU intervention (#352, #452), achieving 3.35x speedup for SpMV (#296). Metal Indirect Command Buffers (ICBs) are Apple's closest equivalent, enabling GPU-driven command generation with 39-44% speedup (#453, #301). MPSGraph executes DAGs across GPU, CPU, and Neural Engine automatically (#454, #356). Kitsune enables dataflow execution via software-only task queues (#489).

**Q: How does llama.cpp implement GPU-centric inference on Apple Silicon?**
A: llama.cpp Metal backend organizes kernels into 4 categories: quantization dequantization, matrix multiplication, element-wise, and specialized (#462). Uses 3 buffer strategies: Shared (CPU+GPU), Private (GPU-only), Mapped (large datasets) (#463, #370). Operation fusion combines normalization+scale, addition chains, and broadcast ops (#466, #376). On M4 (10 GPU cores): 221 tok/s prompt processing, 24 tok/s text generation (#465). MLX takes a different approach: GPU-first lazy evaluation with deferred computation and automatic kernel fusion (#379).

**Q: What is the roofline model for Apple Silicon GPUs?**
A: Performance bound = min(bandwidth * arithmetic_intensity, peak_compute) (#476). STREAM measured bandwidths: M1=60GB/s, M2=91GB/s, M3=92GB/s, M4=100GB/s (#477). LLM inference has two regimes: prompt processing (batched matmul) is compute-bound; token generation is memory-bandwidth-bound (#478). Use arithmetic intensity to determine whether to optimize for compute throughput or memory bandwidth.

**Q: What is reverse offloading?**
A: Reverse offloading dynamically reassigns work from GPU back to CPU when hardware cache contention makes CPU execution more efficient (#365). HEFT algorithm and prediction-based schedulers optimize CPU/GPU task placement for heterogeneous workloads (#360). On Apple Silicon UMA, the cost of this dynamic reassignment is near-zero since both CPU and GPU share the same physical memory (#368).

## Cross-References

**Related Skills**:
- **gpu-silicon** (Layer 0): Hardware architecture — SM/core counts, SIMD width, register file, cache hierarchy that constrain GPU-centric designs
- **unified-memory** (Layer 0): UMA fundamentals — zero-copy sharing, cache coherency, pointer passing that enable GPU-centric patterns on Apple Silicon
- **metal-compute** (Layer 1): Metal API — compute pipelines, command encoding, buffer management, indirect command buffers for GPU-driven dispatch
- **gpu-perf** (Layer 2): Performance engineering — profiling, occupancy, bandwidth optimization, kernel fusion techniques
- **gpu-io** (Layer 2): Fast Resource Loading, SSD-to-GPU streaming, mmap integration for GPU filesystem access on Apple Silicon
- **gpu-distributed** (Layer 3): Multi-GPU and cluster computing — RDMA over Thunderbolt 5, distributed scheduling for GPU-centric clusters

**GPU-Centric Design Workflow**:
1. Model workload with roofline analysis to identify compute vs memory bottleneck (this skill)
2. Design persistent kernel or megakernel architecture to minimize dispatch overhead (this skill)
3. Use indirect command buffers or task graphs for GPU-driven work scheduling (this skill + metal-compute)
4. Optimize memory access patterns and occupancy (gpu-perf)
5. Enable direct storage/network access where possible (gpu-io)
6. Scale across devices with distributed scheduling (gpu-distributed)

**Key Architecture Decisions**:
- **Persistent vs megakernel**: Persistent kernels for continuous work processing; megakernels for fusing a known computation graph
- **GPU-driven vs CPU-driven dispatch**: Use ICBs/Work Graphs when dispatch count is data-dependent or high-frequency
- **Direct I/O vs CPU-mediated**: Use GPU filesystem/network access for streaming workloads where CPU becomes the bottleneck
- **Spatial vs temporal multitasking**: Spatial partitioning (LithOS) for latency-sensitive multi-tenant; temporal (GPREEMPT) for fairness

## Investigation Prompts

Suggested investigation topics to expand this skill:

- "Metal 4 long-lived command buffers for persistent compute patterns on Apple Silicon"
- "GPU-centric LLM serving on Apple Silicon: llama.cpp vs MLX architecture comparison"
- "Apple Silicon GPU roofline model construction with Metal performance counters"
- "GPU-driven rendering pipeline on Metal: mesh shaders, ICBs, visibility buffer"
- "Kernel atomization feasibility on Apple GPU architecture (LithOS on Metal)"
- "GPU database query execution on Apple Silicon via Metal compute shaders"
- "Work Graph equivalents on Metal: ICBs, object shaders, and argument buffers"
- "GPU memory allocator implementation on Metal using threadgroup memory"
- "Reverse offloading patterns on Apple Silicon UMA: when CPU beats GPU"
- "GPU-centric packet processing on macOS: network extension + Metal compute"

## Notes

- **Layer 4 dependencies**: Requires gpu-silicon (hardware), unified-memory (UMA), metal-compute (API), gpu-perf (optimization), gpu-io (storage), gpu-distributed (multi-device)
- **Apple Silicon advantages**: UMA eliminates PCIe bottleneck, making GPU-centric patterns more practical than discrete GPU systems
- **Metal limitations**: No native persistent kernel support due to GPU watchdog timeout; ICBs and Metal 4 unified encoders provide workarounds
- **Research frontier**: LithOS, Work Graphs, megakernels, and GPU OS abstractions are active research areas (2024-2025)
- **Practical implementations**: llama.cpp and MLX demonstrate production GPU-centric patterns on Apple Silicon today
- **Roofline awareness**: Always model workload arithmetic intensity before choosing optimization strategy
- **Multi-tenant future**: GPU scheduling and OS abstractions will become critical as GPUs run diverse concurrent workloads
- **Cross-platform gaps**: Many GPU-centric techniques (Work Graphs, GPUfs, BaM) exist only on NVIDIA/AMD; Metal equivalents are approximations
