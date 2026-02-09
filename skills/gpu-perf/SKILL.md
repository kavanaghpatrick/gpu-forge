---
name: gpu-perf
layer: 2
description: >
  This skill should be used when the user asks about GPU performance optimization, profiling, occupancy tuning, memory bandwidth, ALU utilization, kernel optimization, or Metal performance analysis. Trigger keywords: "GPU profiling", "occupancy", "Metal System Trace", "Xcode GPU profiler", "bandwidth bound", "ALU bound", "GPU performance", "register pressure", "threadgroup memory", "cache optimization", "memory access patterns", "kernel fusion", "dispatch optimization", "performance counters", "GPU bottleneck", "SIMD utilization", "thread divergence", "atomic operations", "fast-math", "Dynamic Caching", "benchmark methodology", "performance anti-patterns".
related_skills:
  - gpu-silicon
  - metal-compute
  - msl-kernels
  - simd-wave
  - gpu-io
---

# GPU Performance Engineering

GPU performance engineering on Apple Silicon requires understanding hardware architecture, memory hierarchy, execution model, and profiling tools. Unlike NVIDIA/AMD GPUs, Apple's unified memory architecture, tile-based deferred rendering (TBDR), and small L1 caches create unique optimization patterns.

Performance engineering spans multiple layers: profiling tools for diagnosis, occupancy optimization for thread utilization, memory access patterns for bandwidth efficiency, kernel fusion for latency reduction, and dispatch strategies for CPU-GPU coordination. Apple Silicon GPUs have distinctive characteristics: no severe penalty for uncoalesced access, small 8KB L1 caches, 128-byte cache lines, 32KB threadgroup memory with 32 banks, and ~208KB register file per core.

This skill covers the full performance optimization workflow: identifying bottlenecks via Metal System Trace and Xcode GPU profiler, tuning occupancy through register pressure and threadgroup size, optimizing memory access patterns and bandwidth, fusing kernels to reduce overhead, improving dispatch strategies, and avoiding common anti-patterns. The knowledge base contains verified findings from WWDC sessions, reverse engineering, academic papers, and benchmark studies.

## Domain Overview

**Profiling Tools**: Metal System Trace in Instruments provides visual timeline with separate tracks for command buffer submission, GPU execution, memory transfers, and synchronization events. The tool exposes 150+ performance counters organized into categories: Performance Limiters (memory bandwidth, ALU utilization, occupancy), Execution Statistics (active threads, stalls, divergence), Memory System (cache hits, bandwidth), and Shader Metrics (instruction counts, register usage). Xcode GPU profiler adds three key metrics: Utilization (% GPU resources in use), Limiter (primary bottleneck), and Cost (time breakdown per shader stage).

Shader Cost Graph (Xcode 15+, Apple Family 9+) visualizes per-function costs with dependency tracking. For compute-only workloads without presentDrawable, GPU capture requires MTLCaptureManager programmatic capture. Apple Silicon TBDR GPUs only support MTLCounterSamplingPoint.atStageBoundary, not per-draw sampling. Metal Counters API provides timestamp, statistic, and trace counter sets.

**Occupancy Optimization**: GPU occupancy measures the percentage of total thread capacity utilized. Low occupancy (16-25%) indicates wasted parallelism from excessive register usage, large threadgroup memory allocation, or suboptimal threadgroup dimensions. Apple GPU cores have ~208 KiB register file, and using ≤104 registers allows 2 SIMDgroups resident simultaneously for better latency hiding.

Apple GPU cores have 4 schedulers, each dispatching one instruction from one SIMDgroup per cycle. Using 16-bit types (half, ushort) instead of 32-bit uses 2x fewer registers, doubling occupancy potential. On Apple Family 9 GPUs (M3/M4), Dynamic Caching dynamically allocates/deallocates tile memory and render target backing storage, and threadgroup/device/constant memory types share the same physical memory pool. Divergent execution (threads taking different branches) costs ~70 cycles per divergence.

**Memory Access Patterns**: Apple GPU cache line size is 128 bytes (consistent across generations). For 32-wide SIMDgroups, strided access with stride=4 (float4) achieves perfect 128-byte alignment. Apple GPU L1 data cache is only 8 KB per core (vs AMD 16-32 KB, NVIDIA 128 KB), making cache blocking less critical. Apple GPUs show no severe penalty for uncoalesced memory access, unlike NVIDIA/AMD architectures.

Apple M1 GPU threadgroup memory has 32 independent banks with classical stride pattern: accessing elements with stride=32 causes bank conflicts. Use signed integer types (int) for array indexing instead of unsigned (uint) — unsigned comparisons generate extra instructions on Apple GPUs. On Apple Silicon unified memory architecture (UMA), the primary performance win is eliminating unnecessary blits and copies between CPU/GPU memory spaces.

**Kernel Optimization**: Fast-math mode provides 50%+ performance gain and is ON by default in Metal. Apple Family 9 GPUs can dual-dispatch FP16 and FP32 instructions in parallel across SIMD lanes, enabling mixed-precision optimization. Kernel fusion is most beneficial for memory-bound workloads, eliminating redundant memory round-trips. MLX mx.compile fuses GPU kernel launches by tracing computational graphs. On Apple TBDR GPUs, tile shaders and programmable blending merge multiple rendering passes into single tile operations.

**Dispatch Optimization**: Apple recommends 1-2 command buffers per frame with triple buffering to minimize CPU-GPU sync overhead. Indirect dispatch (MTLDispatchThreadgroupsIndirectArguments) eliminates CPU-GPU round-trips for data-dependent dispatch sizes. Metal 4 unified MTL4ComputeCommandEncoder handles compute dispatches, blits, and synchronization in a single encoder. waitUntilCompleted() is a major anti-pattern that blocks CPU thread creating GPU pipeline bubbles.

**Performance Anti-Patterns**: Excessive global atomics bottleneck scales with core count — especially problematic on 40-core M4 Max. Use simdgroup reduction primitives instead. NVIDIA GPUs get 2-4x FP16 acceleration via Tensor Cores, but Apple MPS gets almost no speedup from FP16 (unified scalar/vector pipeline). Fanless Apple Silicon devices (MacBook Air, iPad) throttle significantly under sustained GPU load. Apple GPU TBDR architecture eliminates overdraw for opaque geometry in rendering, but compute workloads don't benefit.

**Benchmark Methodology**: arXiv:2502.05317 establishes reproducible Apple Silicon GPU benchmarking methodology: thermal stabilization, CPU/GPU frequency monitoring, statistical significance testing, and fair comparison baselines. M4 Max 40-core GPU achieves 14.1 TFLOPS FP32 theoretical at 1.38 GHz with 546 GB/s memory bandwidth. AppleNumericalComputing benchmarks 16 algorithm categories on M1: FFT, sorting, matrix operations, reductions, scans, etc.

**Advanced Scheduling**: GPU context switching takes 50-750 microseconds. Driver-level preemptive priority scheduling enables microsecond-scale preemption improving SLO attainment by 2-10x. For LLM serving, GPU context switching incurs stall time several times longer than compute time for small batches. Low-priority GPU tasks can execute during inter-kernel idle time of high-priority tasks. Small GPU kernels in RL simulation and dynamic neural networks fail to saturate GPU hardware.

**Kernel Fusion Techniques**: Fine-grained kernel fusion over-decomposes computation and communication into fine-grained operations and fuses them using dataflow analysis. Coarse-grained kernel-level overlap leaves substantial slack when slowest tile stalls all others. GPU memory allocation optimization via STAlloc combines offline planning with online allocation by exploiting spatio-temporal reuse patterns. Proteus achieves up to 2.8x speedup over AOT compilation for GPU kernels by JIT-compiling based on runtime shapes and configurations.

## Key Knowledge Areas

The knowledge base contains **49 verified and high-confidence findings** covering:

- **Profiling Tools** (6 findings): Metal System Trace, 150+ performance counters, Xcode GPU profiler metrics, Shader Cost Graph, MTLCaptureManager, Metal Counters API
- **Occupancy Optimization** (7 findings): Register file size, scheduler architecture, 16-bit type optimization, Dynamic Caching, divergence costs, threadgroup memory banking
- **Memory Access Patterns** (5 findings): 128-byte cache lines, 8KB L1 cache, coalescing behavior, threadgroup memory banks, signed vs unsigned indexing
- **Kernel Optimization** (3 findings): Fast-math defaults, FP16/FP32 dual-dispatch, kernel fusion patterns
- **Dispatch Optimization** (4 findings): Command buffer pooling, indirect dispatch, unified encoders, synchronization anti-patterns
- **Performance Anti-Patterns** (5 findings): Global atomics, FP16 on Apple vs NVIDIA, thermal throttling, TBDR rendering vs compute
- **Benchmark Methodology** (3 findings): Reproducible benchmarking, M4 Max specs, AppleNumericalComputing test suite
- **Bandwidth Optimization** (2 findings): Reducing memory traffic, eliminating blits on UMA
- **Advanced Scheduling** (8 findings): Context switching overhead, preemptive scheduling, LLM serving, inter-kernel idle time, concurrent execution, JIT compilation, memory allocation, kernel fusion
- **Specialized Techniques** (6 findings): Work-stealing, compute-communication overlap, DISTWAR differentiable rendering atomics

## How to Query

Use the knowledge base CLI to retrieve performance findings:

```bash
# Get all gpu-perf findings
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill gpu-perf

# Search by performance topic
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "occupancy"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "memory bandwidth"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "kernel fusion"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "profiling tools"

# Get specific finding details
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <id>

# Search across all skills
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "Metal System Trace"
```

Query strategies by problem type:
- **Profiling**: Search "Metal System Trace", "performance counters", "GPU profiler"
- **Low Utilization**: Search "occupancy", "register pressure", "threadgroup size"
- **Memory Bound**: Search "bandwidth", "cache", "memory access patterns", "coalescing"
- **Launch Overhead**: Search "dispatch", "command buffer", "indirect dispatch", "kernel fusion"
- **Slow Atomics**: Search "atomic operations", "simdgroup reduction"
- **Thermal Issues**: Search "throttling", "thermal", "sustained load"

## Common Patterns & Quick Answers

**Q: How do I identify GPU performance bottlenecks on Apple Silicon?**
A: Use Metal System Trace in Instruments for visual timeline analysis (#253), then Xcode GPU profiler for three key metrics: Utilization, Limiter, and Cost (#282). Apple GPUs expose 150+ performance counters organized into Performance Limiters, Execution Statistics, Memory System, and Shader Metrics categories (#254). For compute workloads without presentDrawable, use MTLCaptureManager programmatic capture (#255).

**Q: How do I improve GPU occupancy?**
A: Apple GPU cores have ~208 KiB register file; using ≤104 registers allows 2 SIMDgroups resident simultaneously (#259). Use 16-bit types (half, ushort) instead of 32-bit to use 2x fewer registers and double occupancy (#263). Avoid divergent branches which cost ~70 cycles per divergence (#265). On Apple Family 9 GPUs, Dynamic Caching automatically manages tile memory and threadgroup/device/constant memory share the same pool (#261, #262).

**Q: How do I optimize memory bandwidth?**
A: Align memory access to 128-byte cache lines for 32-wide SIMDgroups (#266). Apple GPUs show no severe penalty for uncoalesced access unlike NVIDIA (#268), but coalescing still helps. On unified memory architecture, eliminate unnecessary blits and copies between CPU/GPU (#119). For threadgroup memory, avoid stride-32 access patterns due to 32-bank conflicts (#269). Use signed int for indexing instead of unsigned uint (#270).

**Q: What are the most impactful kernel optimizations?**
A: Fast-math mode provides 50%+ performance gain and is ON by default in Metal (#271). Apple Family 9 GPUs can dual-dispatch FP16 and FP32 instructions in parallel (#272). Fuse memory-bound kernels to eliminate redundant memory round-trips (#273, #274). On TBDR GPUs, use tile shaders and programmable blending to merge rendering passes (#275). Use MLX mx.compile for automatic kernel fusion (#274).

**Q: How do I optimize dispatch overhead?**
A: Use 1-2 command buffers per frame with triple buffering (#276). Use indirect dispatch (MTLDispatchThreadgroupsIndirectArguments) to eliminate CPU-GPU round-trips for data-dependent sizes (#277). Metal 4 unified MTL4ComputeCommandEncoder handles compute, blits, and sync in one encoder (#278). Never use waitUntilCompleted() which blocks CPU and creates pipeline bubbles (#279).

**Q: What are common performance anti-patterns?**
A: Excessive global atomics bottleneck scales with core count on 40-core M4 Max; use simdgroup reductions instead (#283). Apple MPS gets almost no FP16 speedup unlike NVIDIA Tensor Cores (#284). Fanless devices (MacBook Air, iPad) throttle significantly under sustained GPU load (#285). TBDR eliminates overdraw for rendering but doesn't help compute workloads (#286).

**Q: How do I benchmark GPU performance reproducibly?**
A: Follow arXiv:2502.05317 methodology: thermal stabilization, frequency monitoring, statistical significance testing (#280). M4 Max 40-core GPU: 14.1 TFLOPS FP32 theoretical at 1.38 GHz, 546 GB/s bandwidth (#281). Use AppleNumericalComputing benchmark suite for 16 algorithm categories (#287). Xcode GPU profiler provides Utilization, Limiter, and Cost metrics (#282).

**Q: What advanced scheduling techniques are available?**
A: GPU context switching takes 50-750 microseconds; driver-level preemptive priority scheduling improves SLO attainment by 2-10x (#499, #500). For LLM serving, context switching stall time exceeds compute time for small batches (#590). Low-priority tasks can execute during inter-kernel idle time of high-priority tasks (#540). Small kernels from RL/dynamic NNs fail to saturate hardware; use concurrent kernel execution (#524).

**Q: How do I optimize memory-bound workloads with kernel fusion?**
A: Fine-grained kernel fusion over-decomposes computation/communication and fuses via dataflow analysis (#539). Coarse-grained overlap leaves slack when slowest tile stalls (#591). GPU memory allocation optimization (STAlloc) exploits spatio-temporal reuse (#538). Proteus JIT compilation achieves 2.8x speedup over AOT by compiling based on runtime shapes (#522).

**Q: What about distributed GPU workloads?**
A: Adaptive asynchronous work-stealing gathers node capability information for heterogeneous CPU-GPU clusters (#596). See gpu-distributed skill for RDMA over Thunderbolt 5 and cluster programming.

## Cross-References

**Related Skills**:
- **gpu-silicon** (Layer 0): Hardware architecture, SIMD width, register file, cache hierarchy, scheduler details
- **metal-compute** (Layer 1): Metal API usage, compute pipelines, command encoding, buffer management
- **msl-kernels** (Layer 1): Metal Shading Language syntax, thread/threadgroup indexing, memory qualifiers
- **simd-wave** (Layer 2): SIMD intrinsics, simdgroup operations, warp-level primitives, subgroup algorithms
- **gpu-io** (Layer 2): Fast Resource Loading, SSD-to-GPU streaming, mmap integration, memory-mapped I/O
- **unified-memory** (Layer 0): UMA architecture, CPU-GPU memory sharing, cache coherency, zero-copy patterns

**Performance Workflow**:
1. Profile with Metal System Trace / Xcode GPU profiler (this skill)
2. Identify bottleneck: occupancy, bandwidth, ALU, dispatch overhead
3. Apply optimization: register reduction, coalescing, kernel fusion, indirect dispatch
4. Measure with performance counters and Shader Cost Graph
5. Iterate until hitting hardware limits or meeting performance targets

**Common Optimization Paths**:
- **Low occupancy** → Check register usage, threadgroup size, divergence (gpu-silicon for hardware limits)
- **Memory bound** → Optimize access patterns, eliminate copies, use threadgroup memory (unified-memory for UMA patterns)
- **Dispatch overhead** → Fuse kernels, use indirect dispatch, batch work (metal-compute for command encoding)
- **Atomic bottleneck** → Use simdgroup reductions, histogram techniques (simd-wave for primitives)
- **I/O bound** → Use Fast Resource Loading, async streaming (gpu-io for Metal I/O APIs)

## Verification Commands

When implementing performance optimizations:

```bash
# Profile with Metal System Trace
# Instruments GUI: Product > Profile > Metal System Trace

# Capture GPU workload programmatically
xcrun metal -c kernel.metal -o kernel.air
xcrun metallib kernel.air -o kernel.metallib

# Check profiling output structure
# Metal System Trace saves .trace file with:
# - Command Buffer Submission timeline
# - GPU Execution timeline
# - Memory Transfer timeline
# - Performance counter samples

# Verify occupancy (example computation)
# Max occupancy = (Register file size) / (Registers per thread × Threads per threadgroup)
# 208 KiB / (104 × 32) = ~64 threadgroups can be resident

# Check cache line alignment (32 threads × 4 bytes = 128 bytes aligned)
```

## Investigation Prompts

Suggested investigation topics to expand this skill:

- "Apple GPU Dynamic Caching implementation details and memory pooling behavior"
- "Metal 4 unified encoder performance impact vs separate compute/blit encoders"
- "Threadgroup memory bank conflict patterns on M4/M5 GPUs"
- "Apple GPU instruction-level parallelism and dual-issue capabilities"
- "Performance counter interpretation for memory vs ALU bottleneck diagnosis"
- "Kernel fusion heuristics: when does fusion hurt performance?"
- "Apple GPU divergence cost: warp-synchronous vs divergent execution overhead"
- "SIMD reduction patterns vs atomic operations: performance crossover points"
- "Thermal throttling behavior across MacBook Air, Pro, Studio, Mac mini"
- "GPU JIT compilation on Apple Silicon: Swift Metal, MLX, PyTorch"

## Notes

- **Layer 2 dependencies**: Requires gpu-silicon (hardware specs) and metal-compute (API basics)
- **Profiling setup**: Metal System Trace and Xcode GPU profiler require macOS development environment
- **Hardware variations**: Performance characteristics differ across M1/M2/M3/M4 families
- **Thermal management**: Fanless devices (MacBook Air, iPad) require different optimization strategies
- **UMA advantages**: Unified memory architecture enables unique optimization patterns not possible on discrete GPUs
- **TBDR implications**: Tile-based architecture benefits rendering workloads but not compute-only kernels
- **Metal 4 benefits**: Unified encoders, Dynamic Caching, tensor operations in shaders (Apple Family 9+)
- **Benchmark reproducibility**: Thermal state, CPU/GPU frequency, and background processes affect measurements
- **Academic research**: Recent papers focus on GPU scheduling, kernel fusion, memory allocation, JIT compilation
- **Cross-platform differences**: Apple GPU optimization patterns differ significantly from NVIDIA/AMD strategies
