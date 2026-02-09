---
name: gpu-silicon
description: >
  This skill should be used when the user asks about Apple Silicon GPU hardware architecture, microarchitecture, instruction set, or low-level GPU behavior. Covers GPU cores, ALUs, SIMD execution, register files, memory hierarchy (L1/L2/SLC), TBDR architecture, instruction latency, Dynamic Caching, Neural Accelerators (M5), chip specifications (M1-M5 families), GPU firmware, and microarchitectural optimizations. Use when questions mention "Apple GPU cores", "SIMD width", "TBDR", "tile-based deferred rendering", "GPU microarchitecture", "M4 GPU", "M5 GPU", "ALU pipeline", "register file", "instruction cache", "GPU scheduler", "execution units", or hardware-level GPU behavior on Apple Silicon.
---

# GPU Silicon Knowledge

## Domain Overview

Apple Silicon GPUs represent a unique architecture in the GPU computing landscape. Unlike discrete GPUs from NVIDIA or AMD, Apple's integrated GPUs are built on a tile-based deferred rendering (TBDR) architecture originally designed for efficient graphics rendering on mobile devices. However, these GPUs have evolved into capable general-purpose compute engines, particularly from the M3 generation forward with the introduction of Dynamic Caching and enhanced ALU parallelism.

Each Apple GPU shader core contains 128 ALUs organized into 16 Execution Units (EUs) with 8 ALUs per EU. The architecture is fully scalar at all bit sizes (16-bit and 32-bit), with a SIMD width of 32 threads per group. The GPU generations have progressed from G13 (M1/A14) through G14 (M2/A15/A16) to G16 (M3/M4), with M5 introducing the revolutionary Apple10 architecture featuring per-core Neural Accelerators. Each generation has brought incremental improvements in clock speeds, memory bandwidth, and architectural enhancements like Dynamic Caching (M3/M4) that allow register files, L1 cache, and tile memory to be dynamically allocated per workload.

The memory hierarchy is particularly sophisticated: each core has approximately 208KB of register file, 12KB instruction cache, 16-32KB L1 data cache, 256KB-512KB L2 cache, with all cores sharing an 8-128MB System Level Cache (SLC). The SLC acts as a coherent cache between CPU and GPU, enabling true unified memory access patterns. On-chip tile memory (32KB API limit, ~60KB hardware capacity) serves dual purposes: render targets during fragment shading and threadgroup memory during compute execution.

Apple's GPU is firmware-mediated through an ARM64 ASC coprocessor running RTKit firmware, which manages scheduling, power states, and resource allocation. This is fundamentally different from NVIDIA/AMD GPUs where the driver has more direct hardware control. The ISA itself has been reverse-engineered by projects like Asahi Linux, revealing details about instruction encoding, register usage (including the r0l execution mask stack for divergent branches), and cache control instructions.

The M5 generation represents a paradigm shift with Neural Accelerators embedded in each GPU core, delivering 4x peak AI compute performance versus M4. These accelerators provide 128 matrix FMAs per partition per clock cycle but are not directly exposed in Metal Shading Language—instead accessed through Metal Performance Shaders Graph for ML workloads. This architectural evolution positions Apple Silicon GPUs as first-class compute engines capable of handling LLM inference, scientific computing, and GPU-centric application architectures.

## Key Knowledge Areas

The gpu-silicon skill contains 57 findings across these domains:

- **Chip Specifications**: Complete specs for M1-M5 families including GPU core counts, TFLOPS, memory bandwidth, and process nodes
- **Core Architecture**: ALU organization (128 ALUs, 16 EUs), SIMD groups (32 threads), register cache, scalar execution, instruction cache limits
- **Per-Generation Architecture**: G13/G14/G16 ISA evolution, Dynamic Caching (M3+), Neural Accelerators (M5), chiplet packaging rumors
- **Memory Hierarchy**: Register file (208KB), L1/L2 cache specs, SLC sizes (8-128MB), tile memory dual-use, cache coherency
- **Instruction Latency**: FP16 FMA (1 cycle), FP32 FMA (2 cycles), memory access patterns
- **Neural Accelerators**: M5 per-core specs (128 matrix FMAs), MPS Graph API access, 4x AI performance vs M4
- **TBDR Architecture**: Tile memory usage, fragment stage considerations, imageblocks, compute vs rendering tradeoffs
- **Scheduling**: Single-dispatch vs dual-dispatch modes, occupancy considerations
- **Firmware**: RTKit-mediated GPU control, ARM64 ASC coprocessor
- **Microarchitecture Optimizations**: Register cache, SIMD shuffle bandwidth (256 B/cycle), constant hoisting, memory coalescing
- **GPU TLB and Address Translation**: Academic research on LATPC, Avatar, Revelator, Heliostat, FS-HPT, Marching Page Walks
- **Out-of-Order Execution**: GhOST minimal OoO technique for GPUs
- **Memory Controller Optimizations**: Compression-aware controllers, DRAM cache bypass, SCM-aware techniques

## How to Query

All gpu-silicon knowledge is stored in the GPU Computer knowledge database. Query using the `kb` CLI:

```bash
# List all gpu-silicon findings
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill gpu-silicon

# Search for specific topics
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "M4 GPU cores"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "Dynamic Caching"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "Neural Accelerator"

# Get detailed finding with sources
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <finding-id>

# Check statistics
${CLAUDE_PLUGIN_ROOT}/scripts/kb stats
```

Replace `${CLAUDE_PLUGIN_ROOT}` with the actual path to the gpu-forge plugin directory.

## Common Patterns & Quick Answers

### Q: What are the M4 GPU specifications?
**A**: M4 family (G16, TSMC N3E): M4 base = 8-10 cores / 3.5-4.4 TFLOPS / 120 GB/s, M4 Pro = 16-20 cores / 7.7-9.3 TFLOPS / 273 GB/s, M4 Max = 32-40 cores / 14.8-18.4 TFLOPS / 410-546 GB/s bandwidth. [Finding #17]

### Q: How many ALUs does each Apple GPU core have?
**A**: Each core has 128 ALUs organized as 16 Execution Units with 8 ALUs per EU. Two pipeline categories: (1) FP32/Integer/Conditional pipeline (4 instances, 512 ALUs total across 4 cores) and (2) Integer/Complex Math pipeline (128 ALUs, 32 SFUs per core). [Finding #36]

### Q: What is the SIMD width on Apple GPUs?
**A**: SIMD width is 32 threads per group. Each SIMD group has: per-group program counter, stack pointer, 32 execution slots, and shared register file access. Maximum threadgroup size is 1024 threads (32 SIMD groups). [Finding #5]

### Q: What is Dynamic Caching and which chips support it?
**A**: M3/M4 (Apple Family 9, G16) introduced Dynamic Caching: register file, L1 cache, and tile memory are dynamically allocated per workload instead of statically partitioned. Improves efficiency when different shader stages have varying resource demands. [Finding #10]

### Q: What is the register file size and how does it affect occupancy?
**A**: Apple GPU register file is ~208 KB per core. At 104 registers per thread: max 1024 threads (full occupancy). If shaders use more registers, max threads decrease proportionally (register spilling to cache/memory hurts performance). [Finding #6]

### Q: What are the memory latencies on Apple GPUs?
**A**: GPU memory latencies (M2 Pro, measured): L1/L2 hit ~27ns (~80 GPU ticks), SLC hit ~100ns (~300 ticks), DRAM ~180ns (~540 ticks). Threadgroup memory (tile SRAM) has lowest latency as it's purely on-chip. [Finding #20]

### Q: What is tile memory and how does it relate to threadgroup memory?
**A**: On Apple GPUs, threadgroup memory is physically backed by the same on-chip tile memory SRAM used for TBDR rendering. 32KB API limit per threadgroup, ~60KB hardware capacity per core. Tile memory NOT persistent across compute dispatches. [Finding #22]

### Q: What instruction latencies should I optimize for?
**A**: Key instruction latencies: FP16 FMA = 1 cycle throughput / ~2.2 latency, FP32 FMA = 2 cycles throughput / ~4.0 latency, FP16 add = 1 cycle, FP32 add = 2 cycles, integer add = 1 cycle, integer mul = 2 cycles. G16 (M3/M4) can dual-dispatch FP16 and FP32 in parallel for up to 2x ALU performance. [Finding #7, #30]

### Q: What is the M5 Neural Accelerator and how is it accessed?
**A**: M5 embeds a dedicated Neural Accelerator in each GPU core, delivering 4x peak GPU AI compute vs M4. Specs: 128 matrix FMAs per partition per clock. NOT directly exposed in MSL—accessed through Metal Performance Shaders Graph for ML workloads. LLM inference via MLX: time-to-first-token 3.3-4x faster than M4. [Finding #19, #31, #32, #33]

### Q: How does TBDR affect compute workloads?
**A**: Apple GPU TBDR architecture eliminates overdraw for opaque geometry in rendering, but pure compute workloads do NOT benefit from TBDR's hidden surface removal. TBDR advantage for compute is tile memory (32KB) accessible via tile shaders and imageblocks. Tile-based compute dispatches execute within a render encoder with implicit barriers, enabling on-chip data reuse without system memory round-trips. [Finding #286, #164]

### Q: What is the SLC (System Level Cache) and how big is it?
**A**: SLC (System Level Cache) sizes: M1/M2/M3/M4 base = 8MB, M1/M2 Pro = 24MB, M3 Pro = 12MB, M3 Max = 48MB, M4 Max = 128MB. Properties: 128-byte cache line, exclusive wrt CPU caches (data in SLC not duplicated in CPU L2), unified between CPU and GPU for coherent memory access. [Finding #21, #43]

### Q: How does the GPU firmware architecture work?
**A**: Apple GPU is firmware-mediated: an ARM64 ASC (Application Specific Coprocessor) running RTKit firmware handles GPU scheduling, power management, and resource allocation. This differs from NVIDIA/AMD where drivers have more direct hardware control. [Finding #27]

### Q: How efficient is Apple GPU memory bandwidth?
**A**: Measured GPU DRAM bandwidth efficiency: M1 = 60/68 GB/s (90%), M2 = 91/100 GB/s (91%), M1 Pro = 180/200 GB/s (90%), M2 Pro = 202/200 GB/s (>100% due to compression). High efficiency indicates good memory controller design and bandwidth utilization. [Finding #24]

### Q: What causes performance degradation in large shaders?
**A**: Performance degrades significantly when shader executable exceeds ~12 KB (instruction cache size per core). Large shaders cause instruction cache thrashing. Solution: split into multiple smaller kernels or use function constants for specialization to reduce code size. [Finding #38]

### Q: How does divergent branching work?
**A**: Divergent branches use execution mask stack tracked in register r0l. if_*cmp/else_*cmp/endif instructions manipulate the mask stack. When threads diverge, both paths execute serially with inactive threads masked out. Cost: 2x execution time for fully divergent 50/50 branch. [Finding #39]

## Cross-References

### Related Skills

- **unified-memory** (Layer 0): Covers coherency, SLC behavior, virtual memory, and CPU-GPU shared memory patterns—complements gpu-silicon's hardware memory hierarchy
- **metal-compute** (Layer 1): Uses gpu-silicon knowledge to inform compute dispatch strategies, threadgroup sizing, and pipeline configuration
- **msl-kernels** (Layer 1): Applies instruction latency and ALU pipeline knowledge from gpu-silicon when writing optimized shaders
- **gpu-perf** (Layer 2): Leverages gpu-silicon microarchitecture details for performance profiling and bottleneck analysis
- **simd-wave** (Layer 2): Builds on gpu-silicon's SIMD width (32) and instruction latency for wave-level programming patterns
- **mlx-compute** (Layer 3): Uses M5 Neural Accelerator knowledge from gpu-silicon for ML workload optimization

### Key Dependencies

This skill is foundational (Layer 0) and has no dependencies. Higher-layer skills depend on gpu-silicon for:
- Hardware capabilities and limits (ALU count, memory sizes, SIMD width)
- Performance characteristics (instruction latency, memory bandwidth)
- Architectural features (Dynamic Caching, TBDR, tile memory)
- Generation-specific capabilities (M5 Neural Accelerators, G16 enhancements)

### Finding ID Reference

Key findings by topic area:
- **M4 specs**: #17
- **M5 specs**: #18, #19
- **Core architecture**: #3, #4, #5, #36
- **ALU parallelism**: #30, #272
- **Register file**: #6, #25
- **Memory hierarchy**: #9, #20, #21, #22, #28, #40, #41, #42, #43, #44
- **Dynamic Caching**: #10
- **Neural Accelerators**: #19, #31, #32, #33
- **Instruction latency**: #7
- **TBDR**: #22, #121, #275, #286
- **Firmware**: #27
- **ISA details**: #39
- **Scheduling**: #8
- **Generation evolution**: #1, #2, #11, #12, #13

Use `kb detail <id>` to retrieve full finding details including source URLs and confidence levels.
