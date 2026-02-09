# gpu-perf — All Findings (49)

## Finding 596: Adaptive asynchronous work-stealing that gathers node capability information for intelligent victim selection achieves 10.1% performance improvement over conventional implementations, eliminating need for extra communication threads
**Confidence**: medium
**Source**: Adaptive Asynchronous Work-Stealing for distributed load-balancing in heterogeneous systems
**Evidence**: Proposes work-stealing approach for heterogeneous supercomputer nodes where CPU and GPU have different capabilities. Uses runtime node profiling to select optimal steal targets. Asynchronous design avoids dedicated communication threads, reducing resource overhead.
**Tags**: work-stealing,load-balancing,heterogeneous,adaptive,distributed

## Finding 591: Coarse-grained kernel-level overlap leaves substantial slack when slowest tile stretches the communication tail; AutoOverlap compiler enables automatic fine-grained overlap INSIDE a single fused kernel, achieving average 1.3x and up to 4.7x speedup on multi-GPU workloads
**Confidence**: high
**Source**: AutoOverlap: Enabling Fine-Grained Overlap of Computation and Communication with Chunk-Based Scheduling
**Evidence**: AutoOverlap is a source-to-source compiler on Triton that introduces a communication chunk abstraction to decouple communication granularity from kernel structure. Eliminates extra kernel launches, avoids device-wide synchronizations at kernel boundaries, and handles the communication tail problem that coarse-grained approaches miss.
**Tags**: kernel-fusion,communication-overlap,compiler,triton,fine-grained,automatic

## Finding 524: Small GPU kernels in RL simulation and dynamic neural networks fail to saturate compute resources; lightweight runtime dependency detection with sliding-window scheduling (analogous to CPU out-of-order execution) achieves up to 2.19x speedup through concurrent kernel execution
**Confidence**: high
**Source**: ACS: Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
**Evidence**: ACS detects inter-kernel dependencies at runtime and schedules independent kernels concurrently on the GPU. Software-only implementation (ACS-SW) and hardware-software cooperative (ACS-HW) variants. Average 1.56x speedup on deep RL and dynamic DNN workloads. The sliding window approach is inspired by out-of-order processor scheduling.
**Tags**: concurrent-kernels,kernel-scheduling,deep-rl,dynamic-dnn,out-of-order,gpu-utilization

## Finding 539: Over-decomposing computation and communication into fine-grained operations and fusing them into a single kernel can hide up to 96% of communication latency, achieving 1.24x training speedup on 128 GPUs and 1.66x inference speedup
**Confidence**: high
**Source**: FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion
**Evidence**: Flux decomposes coarse-grained operations into much finer-grained ones and fuses GEMM with communication into a single kernel. Supports both training and inference. The key insight is that medium-grained decomposition (prior work) still leaves significant slack between tiles. Fine-grained fusion eliminates kernel launch overhead and device-wide synchronizations at kernel boundaries.
**Tags**: kernel-fusion,communication-overlap,fine-grained,gemm,training,inference

## Finding 522: Proteus achieves up to 2.8x speedup over AOT compilation for GPU kernels by JIT-compiling LLVM IR at runtime. Two-phase approach: AOT phase extracts IR and annotates specialization points, runtime phase specializes kernels using actual parameter values. Key insight: many GPU kernel parameters are runtime-constant but unknown at compile time (grid sizes, data shapes, flags), and specializing on these enables optimizations (dead code elimination, constant propagation) impossible for AOT.
**Confidence**: high
**Source**: Proteus: Portable Runtime Optimization of GPU Kernel Execution with Just-in-Time Compilation
**Evidence**: CGO 2025 paper from LLNL. Uses LLVM IR as common denominator across CUDA and HIP. Background JIT compilation means first execution uses AOT code, subsequent executions use specialized version. Outperforms NVIDIA Jitify by 1.23x average because Proteus operates at LLVM IR level (more optimization opportunities) rather than PTX level.
**Tags**: JIT-compilation,LLVM-IR,runtime-specialization,AOT,kernel-optimization,CGO

## Finding 590: GPU context switching for LLM serving incurs stall time several times longer than single inference iteration; KV cache memory discontinuity is the primary bottleneck; FastSwitch achieves 1.4-11.2x speedup on tail TTFT and TBT metrics
**Confidence**: high
**Source**: FastSwitch: Optimizing Context Switching Efficiency in Fairness-aware Large Language Model Serving
**Evidence**: Identifies three sources of context switch overhead: inadequate I/O utilization during KV cache swapping, GPU idleness during swap operations, and unnecessary data transmission in multi-turn conversations. KV cache discontinuity in memory causes fragmented I/O patterns that underutilize PCIe bandwidth.
**Tags**: context-switching,kv-cache,llm-serving,memory-discontinuity,fairness,latency

## Finding 538: STAlloc combines offline planning with online allocation by exploiting spatio-temporal regularities in GPU memory allocation patterns. DL training has PREDICTABLE allocation patterns across iterations, enabling offline planning of optimal placement. Reduces fragmentation by 85.1% average (up to 100%) and improves throughput up to 32.5%. Key insight: DL frameworks use online allocators that ignore tensor lifespans, but lifespans are deterministic and repeatable across training iterations.
**Confidence**: high
**Source**: STAlloc: Enhancing Memory Efficiency in Large-Scale Model Training with Spatio-Temporal Planning
**Evidence**: EuroSys 2026 paper. Built as pluggable PyTorch allocator. Offline phase: profile one training iteration to learn allocation/deallocation pattern. Online phase: use plan for placement, with fallback for dynamic allocations (MoE routing). The spatio-temporal regularity means the allocator can pre-plan where every tensor goes before the iteration starts.
**Tags**: memory-allocator,fragmentation,spatio-temporal,offline-planning,PyTorch,EuroSys

## Finding 499: GPU context switching takes 50-750 microseconds; driver-level preemptive priority scheduling achieves up to 40% higher schedulability with minimal code changes (one-line macros at GPU segment boundaries)
**Confidence**: high
**Source**: GCAPS: GPU Context-Aware Preemptive Priority-based Scheduling for Real-Time Tasks
**Evidence**: GCAPS enables preemption based on task priorities by controlling GPU context scheduling at the device driver level. Tested on NVIDIA Jetson embedded platforms. The 50-750 microsecond context switch range is determined by GPU cache size and memory access latency. Response time analysis accounts for segment-level task overlaps.
**Tags**: preemption,context-switch,scheduling,real-time,latency,priority

## Finding 540: Low-priority GPU tasks can execute during inter-kernel idle time of high-priority tasks, improving high-priority JCT by 1.32-16.41x compared to GPU sharing mode, with measurement overhead under 5%
**Confidence**: high
**Source**: FIKIT: Priority-Based Real-time GPU Multi-tasking Scheduling with Kernel Identification
**Evidence**: FIKIT uses kernel identification to detect inter-kernel gaps in high-priority task streams and schedules low-priority work in those gaps. Multiple priority queues enable immediate preemption — low-priority tasks pause instantly when high-priority work arrives. Fine-grained kernel measurement enables accurate gap prediction.
**Tags**: multi-tasking,priority-scheduling,kernel-identification,inter-kernel-gaps,gpu-sharing

## Finding 500: Microsecond-scale GPU preemption on closed-source GPUs improves SLO attainment by 9.7x with less than 1% performance degradation for high-priority tasks, while low-priority throughput increases 2.4x via idle time harvesting
**Confidence**: high
**Source**: Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale
**Evidence**: Hummingbird enables microsecond-scale preemption on closed-source GPUs. High-priority tasks suffer less than 1% performance drop when sharing GPU with low-priority tasks. Low-priority tasks gain 2.4x throughput over temporal-sharing by harvesting idle GPU time slices between high-priority kernel launches.
**Tags**: preemption,microsecond,slo,gpu-sharing,idle-harvesting,scheduling

## Finding 118: WWDC21 image processing session demonstrated 62% reduction in device memory traffic by leveraging Apple Silicon tile memory. For 4K FP32 processing: Before optimization: 2.16 GB total bandwidth (load 540MB + filter 1.62GB + store 540MB). After optimization with merged render passes: 810 MB total (load 270MB + store 540MB, intermediates stay in tile memory). Also saved 270 MB memory by eliminating intermediate buffers. Key techniques: (1) use MTLRenderCommandEncoder instead of compute for per-pixel ops, (2) merge render passes for tile memory persistence, (3) set .dontCare load/store actions for transients, (4) use .memoryless storage for ephemeral textures saving hundreds of MB for 6K/8K images.
**Confidence**: verified
**Source**: WWDC21: Create image processing apps powered by Apple silicon
**Evidence**: WWDC21 session 10153 provides exact bandwidth calculations for 4K FP32 pipeline: "Before: 2.16 GB, After: 810 MB (62% reduction)." Fragment shader chaining within single render pass keeps data in tile memory without device memory round-trips. Memoryless textures eliminate allocation for transient attachments.
**Tags**: bandwidth,tile-memory,render-pass-merging,memoryless,optimization

## Finding 119: On Apple Silicon UMA, the primary performance win is eliminating unnecessary blit copies that were required for discrete GPU architectures. Since CPU and GPU share the same physical memory, explicit blit copies between "system" and "video" memory are wasteful and should be removed. Apple GPU frame capture tool can identify these unnecessary blits. The WWDC20 session "Optimize Metal Performance for Apple Silicon Macs" specifically recommends: avoid separate clear passes (fold into render passes), never separately store/reload MSAA data for resolve, and minimize load/store traffic which consumes the majority of system bandwidth.
**Confidence**: verified
**Source**: WWDC20: Optimize Metal Performance for Apple silicon Macs
**Evidence**: WWDC20 session 10632 "Optimize Metal Performance for Apple Silicon Macs" and WWDC21 session 10153 both emphasize eliminating unnecessary copies. GPU frame capture tool identifies wasted bandwidth. The unified memory architecture makes many traditional copy patterns unnecessary.
**Tags**: bandwidth,blit,UMA,unnecessary-copies,optimization

## Finding 280: arXiv:2502.05317 establishes reproducible Apple Silicon GPU benchmarking methodology: post-reboot idle state, caffeinate to prevent sleep, 5 trial repetitions, page-aligned (16384-byte) buffers, powermetrics-based power measurement.
**Confidence**: verified
**Source**: Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC Performance and Efficiency
**Evidence**: Tests on M1-M4 base chips. Matrix sizes 32-16384. GPU: naive Metal, tiled Metal, MPS. Power via powermetrics with SIGINFO sampling. Published Feb 2025.
**Tags**: reproducible-benchmarking,methodology,power-measurement,HPC,thermal

## Finding 281: M4 Max 40-core GPU: 14.1 TFLOPS FP32 theoretical at 1.38 GHz, 546 GB/s bandwidth, 48 MB SLC, 128 GB max memory. LLM token generation is bandwidth-bound: M4 24 tok/s (120 GB/s), M4 Pro ~50 tok/s (273 GB/s), M4 Max ~83 tok/s (546 GB/s).
**Confidence**: high
**Source**: AppleGPUInfo - Print all known information about Apple GPU chips
**Evidence**: AppleGPUInfo: M4 Max specs. llama.cpp discussion #4167: strong correlation between bandwidth and text-generation throughput. Prompt processing (compute-bound) scales more aggressively with cores.
**Tags**: M4-Max,TFLOPS,bandwidth-bound,LLM,token-generation,specifications

## Finding 282: Xcode Metal GPU profiler provides three key metrics: Utilization (% GPU resource used), Limiter (% time bottlenecked by resource), Occupancy (active/max threads). Low occupancy + low limiters = inefficiency. Low occupancy + high limiters = acceptable.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Apple Tech Talk
**Evidence**: Tech Talk 10580: high utilization + high limiter = bottleneck. Low utilization + low limiter = inefficiency. Xcode 15+ adds shader cost graphs, heat maps, execution history for Family 9+.
**Tags**: profiling,Xcode,utilization,limiter,occupancy,methodology

## Finding 534: DISTWAR primitive achieves 2.44x average (up to 5.7x) speedup on differentiable rendering gradient computation by performing warp-level atomic reduction in SM registers, exploiting intra-warp locality in atomic updates. Distributes atomic computation between SM sub-cores and L2 ROP units.
**Confidence**: medium
**Source**: DISTWAR: Fast Differentiable Rendering on Raster-based Rendering Pipelines
**Evidence**: Gradient computation in differentiable rendering (3DGS, NVDiffRec, Pulsar) bottlenecked by massive atomic operations to same memory locations. DISTWAR leverages intra-warp locality to perform register-level reduction before issuing atomics. Evaluated on RTX 4090 across major differentiable rendering applications.
**Tags**: DISTWAR,differentiable-rendering,atomics,warp-reduction,3DGS,gradient-computation

## Finding 276: Apple recommends 1-2 command buffers per frame with triple buffering (dispatch_semaphore(3)) to overlap CPU/GPU work. Batch multiple compute encoders per command buffer before commit(). Small submissions create more waiting than working.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Command Buffers
**Evidence**: Metal Best Practices Guide: target 1-2 command buffers. Triple buffering: 3 dynamic buffers, semaphore-based rotation. Tech Talk 10580: batch encoders before commit(). Empty compute encoder never beats 2.5ms on A9.
**Tags**: command-buffer,triple-buffering,semaphore,dispatch,CPU-GPU-overlap,batching

## Finding 277: Indirect dispatch (MTLDispatchThreadgroupsIndirectArguments) eliminates CPU-GPU synchronization when dispatch arguments are GPU-generated. CPU issues dispatch immediately; GPU fills arguments and executes without CPU intervention.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Indirect Buffers
**Evidence**: Metal Best Practices Guide: indirect buffers eliminate GPU→CPU→GPU synchronization roundtrip. Most beneficial when grid dimensions computed by prior GPU pass (compaction, culling).
**Tags**: indirect-dispatch,GPU-driven,synchronization,CPU-GPU,dispatch-overhead

## Finding 278: Metal 4 unified MTL4ComputeCommandEncoder handles compute dispatches, blits, and acceleration structure builds in single encoder. Non-dependent commands run concurrently by default; pass barriers provide explicit serialization. Command buffers decoupled from queues enabling parallel encoding.
**Confidence**: verified
**Source**: Discover Metal 4 - WWDC25
**Evidence**: WWDC25/205: Metal 4 consolidates encoders. Concurrent execution by default with low-overhead barrier API. MTL4CommandBuffer created from device (not queue), enabling parallel encoding across threads.
**Tags**: Metal-4,unified-encoder,MTL4ComputeCommandEncoder,concurrent-dispatch,pass-barrier

## Finding 279: waitUntilCompleted() is a major anti-pattern that blocks CPU thread creating GPU bubbles. Correct patterns: (1) addCompletedHandler callbacks, (2) semaphore-based triple buffering, (3) multiple command queues from multiple CPU threads.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Apple Tech Talk
**Evidence**: Tech Talk 10580: ''Blocking with waitUntilCompleted is a terrible use of a CPU.'' Use completion handlers or semaphores. Applications on M1 may underperform on M1 Pro/Max because more cores need proportionally more submitted work.
**Tags**: anti-pattern,waitUntilCompleted,triple-buffering,command-buffer,GPU-bubbles

## Finding 273: Kernel fusion is most beneficial for memory-bound workloads, eliminating redundant global memory reads/writes. Fusing compute-bound kernels gives diminishing/negative returns due to increased register pressure reducing occupancy.
**Confidence**: high
**Source**: Kernel Fusion in GPU Computing
**Evidence**: Research: fusing memory-bound sequences preferred. Compute-bound fusion increases on-chip memory demands, may limit occupancy. Excessive registers from fusion spill to global memory, negating bandwidth savings.
**Tags**: kernel-fusion,memory-bound,compute-bound,register-pressure,occupancy

## Finding 274: MLX mx.compile fuses GPU kernel launches by tracing computational graphs: merges scalars, removes no-ops, fuses compatible primitives (unary, binary, ternary, broadcast). MLX custom metal_kernel() achieved 8x speedup for grid_sample and 40x for backward pass via SIMD reduction before atomics.
**Confidence**: verified
**Source**: Custom Metal Kernels - MLX Documentation
**Evidence**: MLX docs: mx.compile traces execution symbolically, optimizes graphs. grid_sample: 55.7ms→6.7ms (8x) on M1 Max. Backward: 676.4ms→16.7ms (40x) using simd_sum() before atomic writes.
**Tags**: MLX,kernel-fusion,custom-kernel,metal_kernel,SIMD-reduction,atomics

## Finding 275: On Apple TBDR GPUs, tile shaders and programmable blending merge multiple render passes into one, keeping data in ~60KB on-chip tile memory. For pure compute, equivalent fusion must be done manually via single-kernel approaches.
**Confidence**: verified
**Source**: Optimize Metal Performance for Apple Silicon Macs - WWDC20
**Evidence**: WWDC20: before tile shading, compute mid-render required store+reload from device memory. Tile shaders operate exclusively in tile memory via imageblocks. Programmable blending gives fragment shaders direct tile memory access.
**Tags**: tile-shading,TBDR,programmable-blending,kernel-fusion,tile-memory,imageblock

## Finding 271: Fast-math mode provides 50%+ performance gain and is ON by default in Metal. Allows compiler reassociation, fused multiply-add, relaxed NaN/infinity semantics. Disabling fast-math partially recoverable with explicit fma() built-in.
**Confidence**: verified
**Source**: Advanced Metal Shader Optimization - WWDC16
**Evidence**: WWDC16 session 606: fast-math delivers 50%+ gain. Compiler may increase precision via FMA instructions. Will NOT introduce new NaNs. Use metal::precise:: namespace for critical operations.
**Tags**: fast-math,precise-math,MSL,shader-optimization,compiler

## Finding 272: Apple Family 9 GPUs can dual-dispatch FP16 and FP32 instructions in parallel across different ALU pipelines, achieving up to 2x ALU performance from mixed-precision instruction streams.
**Confidence**: verified
**Source**: Explore GPU advancements in M3 and A17 Pro - Apple Tech Talks
**Evidence**: Tech Talk 111375: M3/A17 Pro FP16, FP32, and integer can execute in parallel. Up to 2x ALU performance. Requires instructions from multiple concurrent SIMD groups.
**Tags**: dual-dispatch,mixed-precision,Apple-Family-9,M3,M4,ALU-pipeline

## Finding 266: Apple GPU cache line size is 128 bytes (consistent across generations). For 32-wide SIMD accessing 4-byte values = 128 bytes = exactly one cache line fetch. SoC-level cache line also 128 bytes (CPU L1 uses 64-byte lines but requests expand to 128 at SLC/DRAM).
**Confidence**: high
**Source**: metal-benchmarks: Apple GPU microarchitecture
**Evidence**: metal-benchmarks: 128-byte cache line. RealWorldTech: CPU line 64, SoC line 128 — 64B request generates 128B request to SLC/DRAM.
**Tags**: cache-line,128-bytes,memory-coalescing,SLC,DRAM

## Finding 267: Apple GPU L1 data cache is only 8 KB per core (vs AMD 16-32 KB, NVIDIA 128 KB). Compensated by large L2 (~1 MB M1, ~3 MB M2 Pro), 8 MB SLC, and separate dedicated L1 caches for textures vs buffers. Moving data from MTLBuffer to MTLTexture effectively doubles L1 cache capacity.
**Confidence**: verified
**Source**: Optimize Metal apps and games with GPU counters - WWDC20
**Evidence**: Chips and Cheese: 8KB L1 with AMD-comparable latency. WWDC20/10603: separate L1 caches for texture and buffer reads. Moving to textures increases effective cache space. Textures use Morton/Z-order for spatial locality.
**Tags**: L1-cache,texture-cache,buffer-cache,8KB,Morton-order,cache-optimization

## Finding 268: Apple GPUs show no severe penalty for uncoalesced memory access, unlike NVIDIA/AMD. However, coalesced access still preferred for maximum throughput. Also: explicit loop unrolling provides no benefit in Metal kernels.
**Confidence**: high
**Source**: AppleNumericalComputing - Numerical Algorithms on Apple M1
**Evidence**: AppleNumericalComputing benchmarks on M1: no severe penalty for uncoalesced loads/stores from device memory. Loop unrolling factors >4 caused errors with no benefit.
**Tags**: memory-coalescing,uncoalesced-access,loop-unrolling,M1

## Finding 269: Apple M1 GPU threadgroup memory has 32 independent banks with classical stride penalties. M3 GPU shows anomalous non-classical banking behavior that doesn''t match standard GPU shared memory models — likely due to Dynamic Caching using same cache hierarchy.
**Confidence**: high
**Source**: The mystery of Apple M3 on-chip shared memory
**Evidence**: TechBoards benchmarks: M1 classical 32-bank shared memory with predictable stride penalties. M3 shows unpredictable non-classical behavior for stride patterns.
**Tags**: threadgroup-memory,bank-conflicts,M1,M3,shared-memory,stride-access

## Finding 270: Use signed integer types (int) for array indexing instead of unsigned (uint) — unsigned wrapping semantics disable vectorized loads. Batch adjacent memory accesses for compiler vectorization. Reorder struct fields to place co-accessed fields adjacently.
**Confidence**: verified
**Source**: Optimize Metal Performance for Apple silicon Macs - WWDC20
**Evidence**: WWDC20/10632: signed type indicates no wrapping — compiler CAN vectorize. Use float2/float4 for explicit vector loads. SoA layout enables full coalescing; AoS wastes ~66% bandwidth when accessing single component.
**Tags**: vectorized-loads,signed-indexing,SoA,AoS,compiler-optimization,struct-layout

## Finding 259: Apple GPU cores have ~208 KiB register file per core. Using <=104 registers allows 1024 threads/threadgroup (full occupancy); 256 registers reduces max to 384 threads. ALU utilization maxes out at 24 SIMD groups/core (768 threads) — the minimum viable occupancy.
**Confidence**: verified
**Source**: Dissecting the Apple M1 GPU, part III
**Evidence**: Rosenzweig reverse engineering: 104*2*1024=208KiB confirms register file size. metal-benchmarks: ALU utilization maxes at 24 simds/core, Apple prefers spilling to device memory over reducing below this.
**Tags**: register-file,occupancy,208KiB,SIMD-groups,M1,M2

## Finding 260: Apple GPU cores have 4 schedulers, each dispatching one instruction from one SIMD group (32 threads) per cycle. FP16 FADD saturates at 8 simds/core (2/scheduler), not 16. Register dependency penalty: FP32 0.84 cycles (1.84 total), FP16 0.56 cycles (1.56 total).
**Confidence**: high
**Source**: metal-benchmarks: Apple GPU microarchitecture
**Evidence**: metal-benchmarks: 4 schedulers, 128 ALUs per core. At low occupancy (4 simds), FP32 latency rises to 6.60 cycles; at high occupancy drops to 1.84 cycles.
**Tags**: scheduler,SIMD-groups,ALU,instruction-dispatch,register-dependency,latency

## Finding 261: On Apple Family 9 GPUs (M3/M4), Dynamic Caching dynamically allocates/deallocates register memory from L1 cache over shader lifetime. Maximum register usage no longer dictates SIMD group count. Shader core auto-adjusts occupancy to prevent cache thrashing.
**Confidence**: verified
**Source**: Explore GPU advancements in M3 and A17 Pro - Apple Tech Talks
**Evidence**: Tech Talk 111375: registers, threadgroup, tile, stack memory types assigned dynamically from L1 cache. Register file is now a cache allowing spills. No code changes required for benefits.
**Tags**: dynamic-caching,Apple-Family-9,M3,M4,register-allocation,occupancy

## Finding 262: On Apple Family 9 GPUs, threadgroup/device/constant memory types share the same cache hierarchy. If working set fits in cache, device/constant buffer access can match threadgroup memory performance — copying to threadgroup memory may hurt performance.
**Confidence**: verified
**Source**: Learn performance best practices for Metal shaders - Apple Tech Talks
**Evidence**: Tech Talk 111373: On Family 9, threadgroup memory may be less beneficial as software-managed cache. Direct device/constant reads can be faster. Profile to determine.
**Tags**: Apple-Family-9,threadgroup-memory,cache-hierarchy,M3,M4,device-memory

## Finding 263: Using 16-bit types (half, ushort) instead of 32-bit uses 2x fewer registers, gives 2x bandwidth, and type conversion between half↔float is FREE. Use ushort for thread_position_in_threadgroup. FP16 ALU runs at double rate vs FP32 full rate.
**Confidence**: verified
**Source**: Optimize Metal Performance for Apple silicon Macs - WWDC20
**Evidence**: WWDC20/10632 and WWDC20/10603: Apple GPUs optimized for 16-bit. FP16 double rate, FP32 full rate, INT32 half rate. Free type conversion. Use ''h'' suffix on literals (1.0h).
**Tags**: FP16,register-pressure,occupancy,data-types,double-rate,half-rate

## Finding 264: GPU occupancy is percentage of total thread capacity utilized. Low occupancy (16-37%) warrants investigation only when ALU and memory limiter counters are also low. Main causes: register pressure and threadgroup memory exhaustion. Set maxThreadsPerThreadgroup on pipeline descriptor for compiler optimization.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Apple Tech Talks
**Evidence**: Tech Talk 10580: occupancy must correlate with limiters. Register blocks mean small reductions may not improve occupancy until block boundary crossed. maxThreadsPerThreadgroup enables compiler opts.
**Tags**: occupancy,profiling,register-pressure,threadgroup-memory,GPU-counters

## Finding 265: Apple GPU divergent execution (threads take different branches) costs ~70 cycles vs ~40 cycles for coherent (all same path) — a 75% penalty. Sorting work for branch coherence within SIMD groups is worthwhile.
**Confidence**: verified
**Source**: Optimize Metal apps and games with GPU counters - WWDC20
**Evidence**: WWDC20 session 10603: Coherent 40 cycles, divergent 70 cycles. Both paths executed when divergent.
**Tags**: branch-divergence,ALU,SIMD,coherent-execution,performance-penalty

## Finding 283: Excessive global atomics bottleneck scales with core count — especially problematic on Pro/Max variants. Use threadgroup atomics and SIMD-level reductions (simd_sum) before global atomic writes.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Apple Tech Talk
**Evidence**: Tech Talk 10580: excessive global atomics identified as new bottleneck on M1 Pro/Max with increased cores/bandwidth. Prefer threadgroup atomics. MLX uses simd_sum() before atomic updates for 40x speedup.
**Tags**: anti-pattern,atomics,global-atomics,threadgroup-atomics,SIMD-reduction

## Finding 284: NVIDIA GPUs get 2-4x FP16 acceleration via Tensor Cores, but Apple MPS gets almost no FP16 benefit and MLX only 20-30% improvement. Apple FP16 advantage comes from register savings and bandwidth, not dedicated tensor hardware.
**Confidence**: high
**Source**: Apple Silicon vs NVIDIA CUDA: AI Comparison 2025
**Evidence**: Analysis: FlashAttention, bitsandbytes, TensorRT have no Metal equivalents. ResNet-50 training: RTX 4090 ~15s/epoch vs M4 Max 45-50s (3x slower). Inference gap smaller due to unified memory.
**Tags**: FP16,NVIDIA-vs-Apple,Tensor-Cores,porting,software-ecosystem

## Finding 285: Fanless Apple Silicon devices (MacBook Air, iPad) throttle significantly under sustained GPU compute. Desktop devices (Mac Mini, Mac Studio) with active cooling maintain more consistent performance. For reproducible benchmarks, use desktop with active cooling.
**Confidence**: verified
**Source**: Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC Performance and Efficiency
**Evidence**: arXiv:2502.05317: laptops show lower power dissipation suggesting thermal limiting. M4 Mac Mini (active cooling) showed more sustained performance vs MacBook Air (fanless). Post-reboot idle state recommended for benchmarks.
**Tags**: thermal-throttling,sustained-workload,cooling,form-factor,benchmarking

## Finding 286: Apple GPU TBDR architecture eliminates overdraw for opaque geometry in rendering, but pure compute workloads do NOT benefit from TBDR''s hidden surface removal. TBDR advantage for compute is tile memory (32KB) accessible via tile shaders and imageblocks.
**Confidence**: high
**Source**: GPU Hardware and Metal concerning Tile Memory - Apple Developer Forums
**Evidence**: Apple Forums: TBDR performs HSR before pixel shading. For compute: tile shaders and imageblocks allow mixing compute with rendering using on-chip tile memory. Pure compute behaves more like IMR.
**Tags**: TBDR,tile-memory,compute-vs-graphics,IMR,architecture

## Finding 287: AppleNumericalComputing benchmarks 16 algorithm categories on M1: FFT, sorting, convolution, prefix-scan, N-body, sparse/dense matrix ops, Cholesky, conjugate gradient, Jacobi/Gauss-Seidel. Found uncoalesced access minimal penalty and explicit loop unrolling no benefit.
**Confidence**: high
**Source**: AppleNumericalComputing - Numerical Algorithms on Apple M1
**Evidence**: GitHub: ShoYamanishi/AppleNumericalComputing. Each section compares C++ baseline, BLAS/vDSP, NEON+multithreading, and custom Metal GPU kernels on M1 Mac Mini.
**Tags**: FFT,sort,convolution,reduction,scan,application-benchmark,numerical-computing

## Finding 253: Metal System Trace in Instruments provides visual timeline with separate tracks for CPU, GPU command buffers, compute encoders, and memory allocations. Expanding encoder tracks reveals a Shader Timeline showing which shaders run at sampled times during execution.
**Confidence**: verified
**Source**: Discover Metal debugging, profiling, and asset creation tools - WWDC21
**Evidence**: WWDC21 session 10157: Metal System Trace captures over-time behavior. Expanding GPU track shows command buffers color-coded by frame. Shader Timeline (opt-in under recording options) shows per-shader GPU time estimates via sampling in a waterfall visualization.
**Tags**: metal-system-trace,instruments,gpu-timeline,shader-timeline,compute-profiling

## Finding 254: Apple GPUs expose 150+ performance counters organized into: Performance Limiters (ALU, Texture Read/Write, Buffer Read/Write, Tile Memory, GPU LLC), Memory Bandwidth, Occupancy (overall/compute/vertex/fragment), and Hidden Surface Removal. ALU limiter measures FP16 at double rate, FP32 full rate, INT32 half rate.
**Confidence**: verified
**Source**: Optimize Metal apps and games with GPU counters - WWDC20
**Evidence**: WWDC20 session 10603: counter groups include Performance Limiters as primary bottleneck indicators, Memory Bandwidth, Occupancy for thread utilization, HSR for rendering. Each limiter maps to specific GPU hardware stage.
**Tags**: gpu-counters,performance-limiters,alu,texture,occupancy,bandwidth

## Finding 255: For compute-only workloads (no presentDrawable), GPU capture requires MTLCaptureScope (beginScope/endScope) or programmatic MTLCaptureManager with MTLCaptureDescriptor outputting .gputrace files. Requires Info.plist key MetalCaptureEnabled=true.
**Confidence**: verified
**Source**: Capturing a Metal workload in Xcode - Apple Developer Documentation
**Evidence**: Apple docs: Frame capture scoped by presentDrawable. Compute-only needs device-level MTLCaptureScope or MTLCaptureManager programmatic capture. Outputs .gputrace files openable in Xcode for full debugger/profiler analysis.
**Tags**: gpu-capture,compute-only,MTLCaptureScope,MTLCaptureManager,programmatic-capture

## Finding 256: Shader Cost Graph (Xcode 15+, Apple Family 9+) visualizes shader function costs as a flame graph with per-line source annotations showing GPU instruction counts and cost breakdown by category (ALU, Memory, Synchronization). Synchronization category indicates memory latency stalls.
**Confidence**: verified
**Source**: Discover new Metal profiling tools for M3 and A17 Pro - Apple Tech Talk
**Evidence**: Tech Talk 111374: Shader Cost Graph shows most expensive functions as flame graph. Per-line annotations with instruction categories. Synchronization cost = threads stalling waiting for data.
**Tags**: shader-cost-graph,per-line-profiling,flame-graph,instruction-count,xcode-15

## Finding 257: Only MTLCounterSamplingPoint.atStageBoundary is supported on Apple Silicon TBDR GPUs. atDispatchBoundary, atDrawBoundary, atBlitBoundary all return false. However, sampleCounters() on MTLComputeCommandEncoder can still insert sample points between dispatches.
**Confidence**: verified
**Source**: MTLCounterSamplingPoint - Apple Developer Documentation
**Evidence**: WebGPU issue #1956 and Apple docs: TBDR architecture processes all vertex/fragment work in parallel within a pass, making draw-level boundaries meaningless. Stage boundary = start/end of compute/blit passes. sampleCounters() provides finer granularity.
**Tags**: counter-sampling,atStageBoundary,TBDR,Apple-Silicon,sampleCounters

## Finding 258: Metal Counters API provides three common counter sets: timestamp (GPU timestamps aligned to mach_absolute_time on Apple Silicon), stageUtilization (hardware utilization), statistic (computeKernelInvocations count). Create MTLCounterSampleBuffer, attach to pass descriptor, resolve with resolveCounterRange().
**Confidence**: verified
**Source**: Explore Live GPU Profiling with Metal Counters - Apple Tech Talk
**Evidence**: Tech Talk 10001: timestamp counters pre-aligned to mach_absolute_time on Apple Silicon (unlike Intel/AMD). Implementation: create sample buffer, attach via sampleBufferAttachments, resolve after execution.
**Tags**: MTLCounterSet,timestamp,stageUtilization,statistic,programmatic-counters

## Finding 121: On Apple GPUs, memory barriers AFTER fragment stage are extremely expensive and should be avoided. Metal validation will report an error if you attempt this. Instead, end the render command encoder and use MTLFence for synchronization between passes. Memory barriers are fast for vertex stages and concurrent compute dispatches. This is an Apple GPU-specific consideration due to the TBDR architecture where fragment execution is heavily deferred and out-of-order. Using a barrier after fragment forces a full tile flush which defeats TBDR benefits.
**Confidence**: verified
**Source**: WWDC22: Go bindless with Metal 3
**Evidence**: WWDC22 "Go bindless with Metal 3" explicitly warns: "AVOID [memory barriers] after fragment stage on Apple GPUs (very expensive, validation error)." Recommends ending encoder and using Fence instead. This is a direct consequence of Apple's tile-based deferred rendering architecture.
**Tags**: memory-barrier,fragment,TBDR,fence,Apple-GPU,synchronization

