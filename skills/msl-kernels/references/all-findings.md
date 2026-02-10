# msl-kernels — All Findings (95)

## Finding 721: 16-bit ops have 2x throughput of 32-bit on Apple GPU: FADD16/FFMA16 achieve 1,1 throughput cycles (~2.17 latency) while FADD32/FFMA32 take 2,1 throughput cycles (~2.21 latency). IADD16 = 1,1 throughput; IMUL16 = 4,4 cycles. Register dependency: 16-bit chains incur 0.56-cycle penalty vs 0.84-cycle for 32-bit. The ~208KB register file with ~3072 max threads means 16-bit types directly halve register consumption, potentially doubling occupancy.
**Confidence**: verified
**Source**: metal-benchmarks - Apple GPU microarchitecture
**Evidence**: metal-benchmarks: "FADD16, FFMA16: 1,1 throughput; FADD32, FFMA32: 2,1 throughput." "16-bit register dependencies: 0.56-cycle penalty vs 0.84-cycle for 32-bit."
**Tags**: 16-bit,32-bit,throughput,alu,register-pressure,occupancy,performance

## Finding 745: When simdgroup_matrix was introduced on A14, Apple reported: 37% average improvement for general matrix multiplication, 36% for CNN convolutions, 22% for full ML networks (Inception V3) — all over A13 (which lacked simdgroup_matrix). Basic 16x16 matmul pattern: 2 simdgroups, each computing 8x16 strip, K=16 loop with 2 iterations. Pattern: for k in 0..K step 8: simdgroup_load from threadgroup, simdgroup_multiply_accumulate, simdgroup_store.
**Confidence**: verified
**Source**: Discover Metal enhancements for A14 Bionic - Apple Tech Talk
**Evidence**: Apple Tech Talk "Discover Metal enhancements for A14 Bionic": "General Matrix Multiplication: 37% improvement. CNN Convolutions: 36%. Inception V3: 22%."
**Tags**: simdgroup_matrix,A14,37-percent,matmul,CNN,introduction

## Finding 726: Standard float atomic add workaround in MSL: (1) atomic_load_explicit as uint, (2) as_type<float> reinterpret, (3) float operation, (4) as_type<uint> back, (5) atomic_compare_exchange_weak_explicit, (6) loop until success. Apple's "Modern Rendering with Metal" example validates this by reinterpreting threadgroup variables between atomic_uint* and float*. Alternative: fixed-point scaling (multiply by INT_MAX, use atomic_fetch_add on int, divide back).
**Confidence**: verified
**Source**: How to do atomic operations on real types - Apple Developer Forums
**Evidence**: Apple Forums thread 69703 and gpuweb#2377 confirm Apple's own code uses the reinterpret pattern. Fixed-point alternative: "multiply by large number, atomic_fetch_add_explicit on int, divide at end."
**Tags**: atomics,float,cas-loop,compare-exchange,as_type,workaround,pattern

## Finding 743: At ISA level, Apple G13 (M1) implements ballot via icmp_ballot (integer comparison) and fcmp_ballot (float comparison) instructions. Each produces 32-bit result where each bit indicates which threads satisfied the condition, broadcast to all active threads. Quad-scope variants: icmp_quad_ballot, fcmp_quad_ballot for 4-thread quad groups. Execution mask managed via stack: pop_exec, if_icmp, while_icmp, jmp_exec_any, jmp_exec_none — per-thread activity counters.
**Confidence**: verified
**Source**: Apple G13 GPU Architecture Reference (Dougall Johnson)
**Evidence**: Dougall Johnson G13 docs: "icmp_ballot: Compares integer values across active threads, producing 32-bit result." Execution mask: "pop_exec: Decrements per-thread counter in r0l; threads inactive when counter reaches zero."
**Tags**: G13,ISA,ballot,icmp_ballot,fcmp_ballot,execution-mask

## Finding 722: Apple M1 (G13) GPU has classical threadgroup memory with 32 independent banks. Stride-2 access hits every second bank; stride-4 hits every fourth; stride-32 causes all threads to access same bank. Performance degradation proportional to bank conflicts, matching standard GPU banking pattern.
**Confidence**: high
**Source**: The mystery of Apple M3 on-chip shared memory
**Evidence**: Microbenchmark study: "this is a classical shared memory with 32 independent banks." Using stride of 2 hits only every second bank so performance goes down. Penalty similar up to stride 32.
**Tags**: m1,threadgroup-memory,bank-conflicts,32-banks,simd,performance

## Finding 729: On Apple Family 9 GPUs (M3/A17 Pro+), threadgroup memory as software cache may be counterproductive. Dynamic Caching shares cache hierarchy with device/constant memory. If working set fits in cache, reading directly from device/constant buffers avoids copy latency. Apple: "if threadgroup memory is primarily a software-managed cache, it may be more performant to read directly from buffers instead."
**Confidence**: verified
**Source**: Learn performance best practices for Metal shaders - Apple Tech Talks
**Evidence**: Apple Tech Talk 111373: "threadgroup, device, constant memory types are using the same cache hierarchy." "Operate directly with device or constant buffers to avoid latencies involved with copying to threadgroup memory."
**Tags**: threadgroup-memory,dynamic-caching,m3,a17-pro,apple-family-9,cache-hierarchy

## Finding 723: M3 (G15) GPU threadgroup memory exhibits fundamentally different banking from M1's classical 32-bank model. Store-only kernels show bank conflicts at strides 8, 16, 24, 32 but no penalty for even strides; load-accumulate shows penalty for even strides yet stride-32 performs well; additional semi-cyclical effects. Dynamic Caching has changed the underlying memory organization — standard padding techniques may need re-evaluation.
**Confidence**: high
**Source**: The mystery of Apple M3 on-chip shared memory
**Evidence**: Researcher described M3 results as "a hot mess" with inconsistent patterns. "I have no idea how this memory is organised in the background."
**Tags**: m3,threadgroup-memory,bank-conflicts,dynamic-caching,non-classical

## Finding 737: MLX Steel decomposes GEMM into hierarchical tiles: block-level (BM x BN x BK, typically 32x32x16 or 64x64x16), distributed across WM x WN SIMD groups (typically 2x2 = 4 simdgroups = 128 threads). Each SIMD group handles TM x TN 8x8 tiles where TM=BM/(8*WM), TN=BN/(8*WN). For BM=BN=64: TM=TN=4, meaning 16 accumulator simdgroup_matrix objects per SIMD group. Inner K loop steps by 8 (kFragSize). Serpentine iteration order improves register reuse.
**Confidence**: verified
**Source**: MLX Steel GEMM MMA Implementation (mma.h)
**Evidence**: MLX mma.h: TM = BM/(kFragSize*WM), TN = BN/(kFragSize*WN). Inner loop: for kk in 0..BK step kFragSize: simdgroup_barrier, Atile.load, Btile.load, tile_matmad. Serpentine: n_serp = (m%2) ? (N-1-n) : n.
**Tags**: mlx,steel,gemm,tiling,simdgroup_matrix,BM,BN,BK,serpentine

## Finding 738: MLX Steel adds explicit threadgroup memory padding for bank conflict avoidance: tgp_padding = 16/sizeof(T) — 4 elements for float32, 8 for float16/bfloat16. Non-transposed A allocation: BM * (BK + padding). Transposed A: BK * (BM + padding). Padding shifts successive rows so different SIMD lanes accessing same column but different rows hit different banks.
**Confidence**: verified
**Source**: MLX Steel GEMM Kernel Header (gemm.h)
**Evidence**: MLX gemm.h: STEEL_CONST short tgp_padding_a = 16/sizeof(T); tgp_mem_size_a = transpose_a ? BK*(BM+tgp_padding_a) : BM*(BK+tgp_padding_a).
**Tags**: mlx,steel,threadgroup-memory,padding,bank-conflicts,simdgroup_matrix

## Finding 730: Complete MSL atomic functions (all memory_order_relaxed only): atomic_store_explicit, atomic_load_explicit, atomic_exchange_explicit, atomic_compare_exchange_weak_explicit, atomic_fetch_{add,sub,and,or,xor,min,max}_explicit. Work on atomic_int/atomic_uint in device and threadgroup address spaces. _explicit suffix mandatory. MSL requires explicit atomic types — cannot perform atomics on regular int/uint (unlike HLSL/GLSL/SPIR-V untyped atomics).
**Confidence**: verified
**Source**: Atomics proposal - gpuweb/gpuweb
**Evidence**: gpuweb#1360: "Metal 1.0 supports: atomic_store/load/exchange/compare_exchange_weak, atomic_fetch_{and,or,add,max,min,sub,xor}_explicit." MSL spec ch02: atomic types "restricted solely to Metal atomic functions," a "subset of C++14 atomic and synchronization functions."
**Tags**: atomics,function-list,c++14,explicit,complete-reference

## Finding 724: MSL atomic functions exclusively support memory_order_relaxed — no acquire, release, or seq_cst available. Significant departure from C++ atomics and other GPU APIs (GLSL/SPIR-V/HLSL). Atomics guarantee only atomicity and modification order consistency — no synchronization with non-atomic memory ops. Thread synchronization must use threadgroup_barrier() instead.
**Confidence**: verified
**Source**: Atomics proposal - gpuweb/gpuweb
**Evidence**: WebGPU atomics proposal (gpuweb#1360): "Atomic functions only support memory_order_relaxed (no synchronization with other memory operations)." "The biggest difference from SPIR-V is that atomic types are required."
**Tags**: atomics,memory-order,relaxed-only,synchronization,msl-spec,limitation

## Finding 749: MTLCompileOptions.mathMode (Metal 3) replaces Boolean fastMathEnabled with 3 values: safe (strict IEEE 754), relaxed (aggressive optimizations honoring INF/NaN — no signed zeros, allows reciprocal/reassociation, FP contract fast), fast (most aggressive, may violate IEEE 754 INF/NaN). Companion mathFloatingPointFunctions controls FP32 precision separately. preserveInvariance ensures consistent position calculations.
**Confidence**: verified
**Source**: MTLMathMode - Apple Developer Documentation
**Evidence**: Apple docs: "mathMode allows selecting precision more granularly, replacing fastMathEnabled." MTLMathMode.relaxed: "aggressive, potentially lossy assumptions while honoring Inf/NaN."
**Tags**: MTLCompileOptions,mathMode,fastMathEnabled,IEEE754,precision,compiler

## Finding 750: MTLLibraryOptimizationLevel: default (runtime performance) or size (binary size). maxTotalThreadsPerThreadgroup on pipeline descriptor (or [[max_total_threads_per_threadgroup(N)]] MSL attribute) lets compiler spill registers more efficiently by knowing max thread count at compile time — reducing register pressure, improving occupancy. Apple recommends setting when possible for better generated code.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Tech Talks
**Evidence**: Tech Talk 10580: "compiler can spill registers more efficiently when max thread count known at pipeline creation time. Enable via maxThreadsPerThreadgroup or max_total_threads_per_threadgroup attribute."
**Tags**: MTLCompileOptions,optimizationLevel,maxTotalThreadsPerThreadgroup,occupancy

## Finding 754: llama.cpp benchmarks: Metal 4 tensor_ops on M5 showed ~2.4x speedup over simdgroup_matrix on M4 Max (Llama 8B Q4_0: 540 vs 247 t/s prompt processing). Optimal: 128x64x64 tiles, 4 simdgroups, tensor_inline mode. Device-to-SRAM direct is faster than staging through threadgroup memory. Limitations: PSO creation 1-2s per pipeline with tensor ops; K-dimension splitting needed for K>=4096.
**Confidence**: verified
**Source**: metal: initial Metal4 tensor API support - llama.cpp
**Evidence**: llama.cpp PR #16634: "M5: Llama 8B Q4_0 ~2.4x (540 vs 247 t/s). Qwen3 0.6B F16 ~1.6x (4936 vs 3073 t/s)." "128x64x64 tile best. Device->SRAM faster than device->threadgroup. PSO 1-2s compilation."
**Tags**: metal4,tensor_ops,M5,performance,2.4x,llama-cpp,neural-accelerator

## Finding 753: Metal 4 Metal Performance Primitives (MPP) introduces tensor_ops::matmul2d. Requires #include <metal_tensor> and <MetalPerformancePrimitives/MPP.h>. matmul2d_descriptor(M, N, K, left_transpose, right_transpose, reduced_precision). Three execution modes: execution_thread (divergent control flow), execution_simdgroups<N> (N cooperating simdgroups), execution_threadgroup. Tensors use tensor_inline for direct device memory access. K must be multiple of 32 in macOS 26.1; dynamic_length_v<int> works around this.
**Confidence**: verified
**Source**: example_matmul_metal4 - Metal 4 tensor matmul example
**Evidence**: liuliu/example_matmul_metal4: "constexpr auto desc = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, false, false, mode::multiply); matmul2d<desc, execution_simdgroups<4>> op;"
**Tags**: metal4,MPP,tensor_ops,matmul2d,execution_simdgroups,cooperative-tensors

## Finding 728: Apple GPU deliberately trades slower inter-SIMD threadgroup communication for faster intra-SIMD shuffle: 256 bytes/cycle SIMD shuffle bandwidth — exactly 2x AMD/NVIDIA (128 B/cycle). Threadgroup memory capacity ~60KB per core (vs 64-100KB competitors). Design favors simd_shuffle-based algorithms over threadgroup memory staging. Bank size and shared BW/cycle remain "TBD" in benchmarks.
**Confidence**: verified
**Source**: metal-benchmarks: Apple GPU microarchitecture
**Evidence**: metal-benchmarks: "SIMD Shuffle BW/Cycle: 256 B" for Apple 7/8 vs 128 B for all competitors. "Apple reduces threadgroup memory bandwidth, prioritizing fast intra-SIMD communication for power efficiency."
**Tags**: simd-shuffle,threadgroup-memory,bandwidth,256-bytes,vs-nvidia,vs-amd

## Finding 178: The device address space is for read/write GPU buffers with no size restriction. Maps to DRAM via unified memory. Cache hierarchy: L1 (8KB/core) -> L2 (768KB-1MB) -> SLC (8-96MB) -> LPDDR. Per-core I/O bandwidth is 32 B/cycle. Cache line size 128 bytes. Use device when data differs per thread, size is dynamic, or writes are needed. Dynamic indexing REQUIRES device address space.
**Confidence**: verified
**Source**: Philip Turner's metal-benchmarks - Apple GPU microarchitecture
**Evidence**: Philip Turner's metal-benchmarks: L1=8KB, L2=768KB(M1), SLC=8MB(M1). Cache line 128 bytes. WWDC16 session 606: buffer accessed by vertex ID/thread position must be device. Per-core I/O 32 B/cycle identical to CPU core.
**Tags**: device,address-space,bandwidth,cache-hierarchy,performance,cache-line-128

## Finding 179: The constant address space is for read-only data, max 64 KB per function argument (Mac1/2 family). Optimized for uniform data (same value accessed by every thread). On Apple GPUs, constant data can be preloaded into special constant registers faster than L1 cache. Preloading requires: using references (not pointers), statically bounded access, and constant address space. Must NOT be used for per-thread varying access.
**Confidence**: verified
**Source**: Learn performance best practices for Metal shaders (Apple Tech Talk)
**Evidence**: WWDC16 session 606: 'data can be put into special constant registers that are even faster for the ALU to access.' Tech talk recommends constant for 'Read-only objects with fixed size, data constant across all threads.' 64 KB limit from Metal Feature Set Tables.
**Tags**: constant,address-space,preloading,caching,uniform-data,64kb-limit

## Finding 180: The thread address space (private) maps to the GPU register file (~208 KB per core). Each thread has up to 256 half-word (16-bit) registers. Occupancy decreases from max 1024 threads/core at <=104 registers to 384 threads/core at 240-256 registers, in increments of 64 threads. Apple prefers register spilling to device memory over reducing occupancy. ALU utilization maxes out at 24 SIMDs/core (lowest occupancy).
**Confidence**: verified
**Source**: Dissecting the Apple M1 GPU, part III - Alyssa Rosenzweig
**Evidence**: Alyssa Rosenzweig's Asahi GPU reverse engineering: 256 half-word registers per thread with occupancy tables. Philip Turner: 208 KB register file per core. M1 has ~4.875 MiB total register file.
**Tags**: thread,registers,occupancy,register-spill,private-memory,208kb

## Finding 181: Dynamically indexed non-constant arrays in thread (private) address space cause ~30% performance loss due to register spills. Use [[max_total_threads_per_threadgroup(N)]] attribute (MSL 2.1+) to hint compiler about max threadgroup size, allowing more efficient register allocation and potentially avoiding spills.
**Confidence**: verified
**Source**: WWDC16 - Advanced Metal Shader Optimization
**Evidence**: WWDC16 session 606: 'Dynamically indexed non-constant stack arrays are catastrophic -- causing 30% performance losses.' metal-benchmarks: compiler spills more efficiently when max thread count is known.
**Tags**: register-spill,private-arrays,dynamic-indexing,max-threads,occupancy

## Finding 182: Metal 3.2 introduced coherent(device) buffer qualifier and memory_coherence_device texture qualifier. These make memory operations visible across ALL threadgroups (not just within a threadgroup) when synchronized with atomic_thread_fence(thread_scope_device) and device-scope atomics. Without coherent(device), cross-threadgroup memory visibility is not guaranteed even with atomics.
**Confidence**: verified
**Source**: SPIRV-Cross Issue #2473 - coherent(device) in MSL
**Evidence**: SPIRV-Cross issue #2473 documents MSL syntax: 'coherent(device) device float* buffer'. Apple Developer Forums confirms Metal 3.2+ support. MoltenVK confirms mapping from SPIR-V's GloballyCoherent.
**Tags**: coherent,device-coherence,metal-3.2,atomics,synchronization,cross-threadgroup

## Finding 183: threadgroup_imageblock address space accesses on-chip tile memory in Apple's TBDR architecture with 2D image-oriented access (width, height, pixel depth). Distinct from threadgroup. On M1, tile memory used for render targets during fragment shading is reused as threadgroup memory during compute. Total imageblock + threadgroup allocation cannot exceed max tile memory limit.
**Confidence**: verified
**Source**: Metal 2 on A11 - Imageblocks (Apple Tech Talk)
**Evidence**: Apple WWDC tech talks: imageblocks are '2D data structure in Tile Memory with width, height and pixel depth.' Memory sharing constraint documented.
**Tags**: threadgroup-imageblock,tile-memory,tbdr,address-space,imageblock

## Finding 202: Branch-free programming via function constants: Larian Studios achieved 84% instruction reduction, 90% branch reduction, 25% register reduction through function constant specialization. Metal compiler folds constant booleans, eliminates unreachable code and unused control flow at pipeline creation time. Performance equal to macro-specialized variants with single master function in .metallib.
**Confidence**: verified
**Source**: Optimize GPU renderers with Metal - WWDC23
**Evidence**: WWDC23 'Optimize GPU renderers with Metal' shows Larian results. Apple tech talk: 'Metal folds constant Booleans, eliminates unreachable code, and removes unused control flow.'
**Tags**: branch-free,function-constants,specialization,register-pressure,divergence

## Finding 203: Visible functions (Metal's function pointers) in compute enable indirect dispatch via MTLVisibleFunctionTable. Significant performance costs: indirect calls prevent full optimization around call site, can cause SIMD divergence and serialization. function_groups attribute for statically-linked functions gives Metal optimization hints. Introduced in WWDC20.
**Confidence**: verified
**Source**: Get to know Metal function pointers - WWDC20
**Evidence**: WWDC20 'Get to know Metal function pointers' introduces visible attribute. Shader performance tech talk: 'indirect function calls can prevent Metal from fully optimizing the shader, especially around the call site.'
**Tags**: visible-functions,function-pointers,indirect-dispatch,divergence

## Finding 204: #pragma unroll (or MLX_MTL_PRAGMA_UNROLL) forces loop unrolling. Nuanced effect on Apple GPUs: improves large workloads (reduces branch overhead, enables vectorized loads) but degrades small/medium inputs (increased register pressure and code size). When loop indices known at compile time, compiler can unroll and optimize away any spill.
**Confidence**: high
**Source**: Metal MSM v2: Exploring MSM Acceleration on Apple GPUs
**Evidence**: MLX conv.metal uses MLX_MTL_PRAGMA_UNROLL. llama.cpp uses FOR_UNROLL macro. Metal MSM v2: 'loop unrolling shows improvements for larger input sizes (>2^20) but makes small & medium slower.'
**Tags**: loop-unrolling,pragma,register-pressure,performance,mlx

## Finding 205: Kernel fusion is the most impactful optimization for ML on Metal. AI-generated fused kernels achieved median 1.35x speedup across 215 PyTorch modules. Fused add+LayerNorm: 42% average speedup. Gimlet Labs: 1.87x on M4 Max via kernel fusion. Metal 4 tensor_ops enable matmul+activation fusion in single shader.
**Confidence**: high
**Source**: Speeding up PyTorch inference with AI-generated Metal kernels
**Evidence**: Gimlet Labs M4 Max benchmarks: 1.87x on KernelBench v0. WWDC24: 'operations fused into a single optimized Metal shader have no memory overhead internally.' Metal 4 neural material example fuses texture sampling+matmul+activation.
**Tags**: kernel-fusion,dispatch-overhead,performance,metal4,ml

## Finding 206: Register pressure management: (1) 16-bit types halve register usage, (2) reduce stack arrays/structs, (3) avoid dynamic indexing into static arrays, (4) function constants eliminate dead code paths, (5) [[max_total_threads_per_threadgroup(N)]] helps compiler spill efficiently. Instruction cache is only 12 KB per core -- -Os (optimize for size) can improve runtime for large kernels via fewer I-cache misses.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Apple Tech Talk
**Evidence**: Metal Compute on MacBook Pro tech talk details each technique. Philip Turner: I-cache 12 KB per core. WWDC22: Blender render improved up to 1.6x with -Os. Set via MTLCompileOptions.optimizationLevel = .size.
**Tags**: register-pressure,instruction-cache,optimize-size,half,max-threads

## Finding 207: ThunderKittens port to Metal (ThunderMittens, Nov 2024): M2 Pro has ~200GB/s bandwidth and ~6.5 TFLOPS. Bandwidth-to-compute ratio 2.5x lower than RTX 4090, making shared memory less crucial -- ALUs can be kept active loading directly from HBM into registers. GEMM implementation ~9% faster than MLX in 11 lines. 8x8 tiles optimal (not 16x16 as on NVIDIA). Swizzling disabled: ALU ops too precious for address computation.
**Confidence**: high
**Source**: ThunderMittens For Your ThunderKittens - Hazy Research
**Evidence**: ThunderMittens blog (Hazy Research, Nov 2024): 'bandwidth-to-compute ratio is 2.5x lower than RTX 4090.' 'ALUs can be kept active by directly loading values from HBM into registers.' GEMM ~9% faster than MLX.
**Tags**: thunderkittens,tiling,simdgroup-matrix,occupancy,mlx,gemm

## Finding 184: MSL atomic types: atomic_int, atomic_uint (Metal 1.0), atomic_bool and atomic<T> where T is int/uint/bool (Metal 2.4). NO native atomic<float> or atomic<half>. Metal requires explicit atomic<T> type declarations -- unlike HLSL/SPIR-V which support untyped atomics on regular addresses. Float atomics must be emulated via CAS loops.
**Confidence**: verified
**Source**: Metal Shading Language Specification Version 4
**Evidence**: gpuweb issue #1360: 'Metal provides dedicated atomic types supporting only 32-bit integer atomics.' gpuweb #2377: 'On metal, in order to do an atomic operation you need to have that explicit type.' Reinterpreting regular types as atomic is unsupported.
**Tags**: atomic-types,atomic-int,atomic-uint,typed-atomics,float-atomics

## Finding 185: threadgroup_barrier() mem_flags: mem_none (execution barrier only, no fence), mem_device (orders device memory ops), mem_threadgroup (orders threadgroup memory ops), mem_texture (texture read_write, Metal 1.2+), mem_threadgroup_imageblock (imageblock memory). Flags combinable. All barriers implement acquire-release semantics. simdgroup_barrier() is effectively a no-op on Apple Silicon since all 32 threads execute in lockstep.
**Confidence**: verified
**Source**: WebGPU Barriers Proposal - Metal Barrier Model
**Evidence**: gpuweb barriers proposal #1374: 'All barriers implement acquire-release semantics.' Apple Developer Forums: 'memflags tell the compiler which caches need to be flushed.' gpuweb #4437: simdgroup_barrier maps to MSL but GPU lockstep provides implicit synchronization.
**Tags**: threadgroup-barrier,mem-flags,acquire-release,simdgroup-barrier,lockstep

## Finding 186: Metal does NOT provide a standalone memory fence without execution synchronization. threadgroup_barrier() always combines execution sync AND memory ordering. An internal __metal_atomic_fence intrinsic exists but is NOT exposed publicly. This is a significant limitation vs Vulkan's memoryBarrierBuffer. Filed as Apple Feedback FB8196691.
**Confidence**: verified
**Source**: MoltenVK: memoryBarrierBuffer != threadgroup_barrier(mem_device)
**Evidence**: MoltenVK issue #973: 'Metal lacks a pure memory barrier without execution synchronization.' Contributors found __metal_atomic_fence compiler builtins exist but remain unexposed.
**Tags**: memory-barrier,fence,execution-barrier,limitation,moltenvk

## Finding 187: 64-bit atomics: Apple added hardware UInt64 min/max (only) starting with Apple8 family (M2). Full 64-bit CAS/add/sub NOT available in hardware. Added for Unreal Engine 5 Nanite. Metal 3.1 introduced texture atomics for int, uint, ulong on 1D/2D/3D textures. M1 lacks hardware image atomics entirely -- Asahi driver emulates via buffer atomics on calculated pixel addresses.
**Confidence**: high
**Source**: Philip Turner's Metal Benchmarks
**Evidence**: Philip Turner: 'Apple may have added atomic UInt64 min/max precisely to get Nanite running on M2.' MoltenVK #1939: Metal 3.1 texture atomics. Alyssa Rosenzweig: 'M1 lacks hardware instructions for image atomics.'
**Tags**: 64-bit-atomics,ulong,nanite,texture-atomics,metal-3.1,m1,m2

## Finding 188: Apple's recommended hierarchical reduction: (1) SIMD-group level -- simd_sum() within each 32-thread group (stays in registers), (2) Threadgroup level -- one thread per SIMD writes to threadgroup memory (scales with cores), (3) Global level -- one thread per threadgroup atomic adds to device memory. Global atomics get WORSE with more GPU cores (serialize at memory controller). Threadgroup atomics scale linearly (per-core tile memory).
**Confidence**: verified
**Source**: WWDC22: Scale Compute Workloads Across Apple GPUs
**Evidence**: WWDC22 session 10159: 'Use SIMD-group operations first, threadgroup atomics next, global atomics as last resort.' 'Increasing numbers of GPU cores leads to more contention.' 'Threadgroup atomics are fulfilled by per-core threadgroup memory.'
**Tags**: hierarchical-reduction,simd-sum,threadgroup-atomics,global-atomics,contention

## Finding 189: Serial dispatch mode provides implicit memory coherency -- all subsequent dispatches see all memory writes from prior dispatches without explicit barriers. Concurrent dispatch mode (MTLDispatchTypeConcurrent) removes this guarantee and requires explicit barriers.
**Confidence**: verified
**Source**: Apple Developer Docs: Tailor Apps for Apple GPUs and TBDR
**Evidence**: Apple developer documentation: 'When you dispatch a compute kernel in default serial mode, Metal guarantees that all subsequent dispatches see all the memory writes.'
**Tags**: serial-dispatch,memory-coherency,implicit-barrier,concurrent-dispatch

## Finding 190: Append buffer lock-free pattern: embed atomic_uint counter at buffer start (16-byte aligned), use atomic_fetch_add_explicit(counter, 1, memory_order_relaxed) to atomically allocate indices. Each thread gets unique index via return value. memory_order_relaxed guarantees atomicity of the increment even without ordering guarantees on surrounding memory.
**Confidence**: high
**Source**: Implementing an Append Buffer in Metal/MSL
**Evidence**: natillum.com article provides complete implementation with atomic_uint counter and append() method. 'memory_order_relaxed makes no guarantees on ordering but does guarantee atomicity.'
**Tags**: append-buffer,lock-free,atomic-fetch-add,counter,pattern

## Finding 191: llama.cpp Metal kernels contain zero atomic operations. Uses threadgroup_barrier(mem_flags::mem_threadgroup) + SIMD intrinsics (simd_sum, simd_max, simd_prefix_inclusive_sum) for all reductions. MLX uses simd_sum() pre-reduction then atomic_fetch_add_explicit for scatter/gradient accumulation. atomic_outputs=True in MLX automatically declares outputs as device atomic<float>.
**Confidence**: verified
**Source**: llama.cpp Metal Kernel Source - ggml-metal.metal
**Evidence**: Direct inspection of ggml-metal.metal: only barriers and SIMD reductions. MLX custom kernel docs: 'atomic_outputs=True and init_value parameters enable safe concurrent updates.'
**Tags**: llama-cpp,mlx,no-atomics,barrier-reduction,simd-sum,atomic-outputs

## Finding 521: Tawa is a compiler that automatically transforms tile-based GPU programs into warp-specialized code using a novel IR abstraction called "asynchronous references" (aref). Arefs express warp-level producer-consumer communication without exposing hardware details. Operates on unmodified Triton programs, performs task-aware partitioning across warp groups, and applies multi-granularity pipelining. Achieves 1.1x over cuBLAS GEMM and matches hand-optimized FlashAttention-3 on H100.
**Confidence**: high
**Source**: Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References
**Evidence**: CGO 2026 paper from NVIDIA. Key contribution: the aref IR abstraction decouples what data is communicated from how and when it moves. This enables automatic pipelining decisions that previously required expert manual optimization. Works by partitioning programs into producer and consumer warp groups with async data movement between them.
**Tags**: warp-specialization,compiler,asynchronous-references,IR,pipelining,Triton,CGO

## Finding 733: Standard technique for avoiding bank conflicts with 32 banks: pad shared memory arrays by 1 element per row. For TILE_DIM x TILE_DIM array, declare as threadgroup float tile[TILE_DIM][TILE_DIM + 1]. Offsets each row by one bank so column-major access hits 32 different banks. Directly applicable on M1 (classical 32-bank model). On M3+ with non-classical banking, standard padding may need re-evaluation.
**Confidence**: high
**Source**: Bank Conflicts in Shared Memory
**Evidence**: ianbarber.blog: "With PAD as 1 (TILE_DIM as 32) we have 32x33 or 132 bytes, offsetting writes ensuring each thread gets its own bank." TechBoards confirms M1 classical 32 banks, M3 changed behavior.
**Tags**: threadgroup-memory,bank-conflicts,padding,avoidance,optimization,m1,m3

## Finding 711: Hardware bfloat16 support (ARM FEAT_BF16) present on M2+ but absent on M1 (hw.optional.arm.FEAT_BF16: 0 vs 1). On M5, Neural Accelerators support FP16 (~7.4 TFLOPS) and INT8 (~13.4 TOPS) but do NOT support bfloat16 — a notable gap. Unclear whether first-gen NA hardware lacks bfloat support or it's not yet exposed in Metal.
**Confidence**: verified
**Source**: Investigating the GPU Neural Accelerators on Apple A19/M5
**Evidence**: Taras Zakharko benchmark: "a notable omission is the bfloat16 format — it is unclear whether the first-generation Neural Accelerator hardware lacks dedicated support."
**Tags**: bfloat16,m2,m5,neural-accelerator,hardware-support,FEAT_BF16

## Finding 710: bfloat (brain floating-point) introduced in MSL 3.1 (WWDC 2023, macOS Sonoma). 16-bit: 1 sign, 8 exponent, 7 mantissa bits — a truncated float32. Functions primarily as a STORAGE format: arithmetic requires explicit conversion to float or half, then back. Wider exponent range (same as float32) provides better gradient overflow/underflow protection for ML training than half's narrower range.
**Confidence**: verified
**Source**: Optimize machine learning for Metal apps - WWDC23
**Evidence**: WWDC23 session 10050: "bfloat16 is a 16-bit floating point format for deep learning comprised of 1 sign bit, 8 exponent bits, and 7 mantissa bits." HN confirms: "vectors of bfloat are a storage format only; you're supposed to explicitly convert them to a wider type to do arithmetic."
**Tags**: bfloat,bfloat16,msl-types,ml-training,storage-format,MSL-3.1

## Finding 719: MSL common functions (metal_common header): clamp(x, minval, maxval) = min(max(x, minval), maxval), undefined if minval > maxval. mix(x, y, a) = x + (y-x)*a for linear interpolation. saturate(x) = clamp(x, 0, 1) as free instruction modifier. step(edge, x) = 0.0 if x < edge else 1.0. smoothstep(edge0, edge1, x) = cubic Hermite interpolation after clamping. Essential for compute: clamp for bounds checking, mix for blending without branching.
**Confidence**: verified
**Source**: MSL Specification Chapter 5 - Metal Standard Library
**Evidence**: MSL specification chapter 5: "Common functions in metal_common: clamp, step, mix, smoothstep, saturate." Image processing examples show clamp for bounds: clamp(g.x - 1, 0, w - 1) for out-of-bounds prevention.
**Tags**: clamp,mix,step,smoothstep,saturate,common-functions

## Finding 208: Metal compiler pipeline: (1) MSL source -> AIR (Apple Intermediate Representation) via modified clang/LLVM, (2) AIR stored in .metallib files, (3) AIR -> GPU binary at pipeline state creation (JIT on device). Step 1 can be offline (Xcode) or runtime (newLibraryWithSource:). Step 3 can be offline via Binary Archives (Metal 3+) to eliminate runtime JIT.
**Confidence**: verified
**Source**: WWDC22 - Target and optimize GPU binaries with Metal 3
**Evidence**: WWDC22 'Target and optimize GPU binaries with Metal 3': full pipeline. Binary archives loaded via newBinaryArchiveWithDescriptor:. Metal Pipelines Script (JSON) for offline generation.
**Tags**: compiler,air,pipeline,binary-archives,offline-compilation,metallib

## Finding 209: Advanced compilation workflows: (1) Dynamic Libraries -- precompiled reusable shader code (Apple Family 7+), (2) Function Pointers -- GPU calls to unknown functions, AIR mode (best perf) or binary mode (fastest compile), (3) Function Stitching -- generates AIR directly from computation graphs skipping MSL frontend ([[stitchable]]), (4) Private Linked Functions -- static linking for maximum optimization.
**Confidence**: verified
**Source**: WWDC21 - Discover compilation workflows in Metal
**Evidence**: WWDC21 'Discover compilation workflows in Metal' with API examples. Dynamic libraries use newDynamicLibrary/serializeToURL. Function stitching uses MTLFunctionStitchingGraph.
**Tags**: compilation,dynamic-libraries,function-pointers,stitching,optimization

## Finding 210: Function constants are superior to preprocessor macros for kernel variants. Single master function compiled to AIR, specialized at runtime via MTLFunctionConstantValues. Compiler folds constants, removes unreachable/dead code. Minimal storage (one master in .metallib) with performance equal to macro variants. Optional constants via is_function_constant_defined mirror #ifdef.
**Confidence**: verified
**Source**: Learn performance best practices for Metal shaders (Apple Tech Talk)
**Evidence**: Apple tech talk: 'Metal folds constant Booleans, eliminates unreachable code, removes unused control flow.' MLX uses function constants extensively. llama.cpp uses function constants for kernel configuration.
**Tags**: function-constants,specialization,kernel-variants,dead-code-elimination

## Finding 211: Apple GPU ALU instruction latencies (M1/M2): FADD/FMUL/FFMA (16/32-bit) = 2.16-2.21 cycles. RECIP32 = 6.5 cycles, RSQRT32 = 8.99 cycles, SIN32/COS32 = ~27 cycles. Precise transcendentals 1.6-2.4x throughput penalty vs fast approximations. Four independent schedulers each dispatch one instruction/cycle from one SIMD.
**Confidence**: verified
**Source**: Philip Turner's metal-benchmarks - Apple GPU microarchitecture
**Evidence**: Philip Turner's metal-benchmarks exact measurements. Key: basic FP16 and FP32 ops have nearly identical latency (~2.2 cycles) but FP16 uses half register bandwidth.
**Tags**: alu-latency,instruction-throughput,dispatch,scheduler,transcendentals

## Finding 732: Metal supports dynamic threadgroup allocation at encoding time via setThreadgroupMemoryLength(_:index:) on MTLComputeCommandEncoder. Kernel declares dynamically-sized buffers with [[threadgroup(n)]] pointer parameters. Multiple buffers via different indices. Total of dynamic + static (queryable via staticThreadgroupMemoryLength) must not exceed maxThreadgroupMemoryLength.
**Confidence**: verified
**Source**: setThreadgroupMemoryLength - Apple Developer Documentation
**Evidence**: Apple documentation for setThreadgroupMemoryLength(_:index:). gpuweb#2024: MSL uses [[threadgroup(n)]] pointers, sizes at "encoding time" — more flexible than WebGPU which fixes sizes at pipeline creation.
**Tags**: threadgroup-memory,dynamic-allocation,setThreadgroupMemoryLength,encoding-time

## Finding 714: fast_math is ON by default in Metal (Xcode "Enable Fast Math" = YES). Provides 50%+ performance gain over -fno-fast-math. Assumptions: no NaNs, no INFs, no signed zeros, allow reciprocal/reassociation, FP contract fast. Does NOT decrease intermediate precision, does NOT introduce new NaNs. metal::precise and metal::fast namespaces provide per-function granularity — e.g., metal::precise::sin() gives full precision even with global fast-math.
**Confidence**: verified
**Source**: Advanced Metal Shader Optimization - WWDC16
**Evidence**: WWDC16 session 606: "Fast-math is on by default... can give 50% performance gain or more." "Does NOT decrease intermediate precision" and "Will NOT introduce new NaNs." MSL spec confirms metal::precise namespace for fine-grained control.
**Tags**: fast-math,precise,compiler-flags,performance,precision,namespaces

## Finding 717: On Apple A8+ and all M-series GPUs, saturate(x), -x (negate), and abs(x) are FREE instruction modifiers encoded in ALU instruction bits — zero additional cycles. The Apple GPU ISA encodes source modifiers for abs (modifier & 0b01) and negate (modifier & 0b10), plus a saturate bit on the destination. Always prefer saturate(x) over manual clamp(x, 0.0, 1.0).
**Confidence**: verified
**Source**: Apple G13 GPU Architecture Reference
**Evidence**: WWDC16 session 606: "These operations are free: float negate = -value; float absolute = abs(value); float saturated = saturate(value)." Dougall Johnson G13 docs: "Absolute Value (modifier & 0b01), Negate (modifier & 0b10), Saturation (S flag)."
**Tags**: saturate,negate,abs,free-modifier,instruction-encoding,zero-cost

## Finding 752: When SIMD group threads invoke different functions via function pointers, execution serializes — worst case 32 threads calling 32 functions = 32x slowdown. Mitigation: coherence reordering — write function indices/params to threadgroup memory, sort to group same-function threads, invoke in sorted order. Transforms worst-case serialization into full SIMD utilization. Function stitching (MTLFunctionStitchingGraph) generates functions from computation graphs at AIR level, skipping Metal frontend.
**Confidence**: verified
**Source**: Get to know Metal function pointers - WWDC20
**Evidence**: WWDC20: coherence reordering pseudocode: "1. Write params and function indices to threadgroup memory. 2. Sort indices. 3. Invoke sorted. 4. Read results. Transforms serialization into full SIMD utilization." Function stitching: "Generate directly to AIR."
**Tags**: visible-functions,SIMD-divergence,coherence-reordering,function-stitching

## Finding 727: Apple explicitly identifies global (device memory) atomics as a primary performance bottleneck, especially on higher-core chips (M1 Pro/Max). With increasing GPU cores and bandwidth, bottleneck has shifted from ALU/bandwidth to global atomics. Apple recommends: "minimize atomic operations, or use techniques built around thread-group atomics instead." SIMD-scoped reductions (simd_sum, simd_max) should be used before any atomics.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Apple Tech Talks
**Evidence**: Apple Tech Talk 10580: "primary bottlenecks have shifted from ALU or memory bandwidth to other areas. One of those is global atomics." "Moderate use of atomics won't be a problem."
**Tags**: atomics,performance,global-atomics,bottleneck,optimization,m1-pro

## Finding 720: Apple A8+ GPUs have 16-bit register units as native width. Key implications: (1) char/uchar do NOT save registers — no native 8-bit arithmetic, emulated with 16-bit, wastes ALU. (2) ushort for thread IDs — thread counts rarely exceed 65535, making all ID arithmetic faster and lower power. (3) half literal suffix required — "half result = input * 2.0f" promotes to float; use 2.0h. (4) uint for indexing prevents vectorized loads — compiler cannot assume overflow-free arithmetic.
**Confidence**: verified
**Source**: Advanced Metal Shader Optimization - WWDC16
**Evidence**: WWDC16/606: "char has no native 8-bit arithmetic on A8+, only use if absolutely necessary." "For global thread IDs, you can usually use ushort." "half result = input * 2.0f creates float operation; use 2.0h." "uint indexing disables vectorized loads."
**Tags**: integer-types,uchar,ushort,uint,register-width,16-bit,literal-suffix

## Finding 716: MSL precision guarantees (ULP): float32 without fast-math: basic arithmetic correctly rounded; sin/cos 4-6 ULP; pow 16 ULP. With fast-math: tolerances relax (acos 5 ULP vs 4, pow stays 16 ULP, NaN/INF behavior undefined). Half-precision (float16): arithmetic correctly rounded, transcendentals only 1 ULP (much tighter than float32's 4-6 ULP), suggesting dedicated half-precision function units.
**Confidence**: verified
**Source**: MSL Specification Chapter 7 - Numerical Compliance
**Evidence**: MSL specification chapter 7: Table 36 (Standard) lists per-function ULP. Table 37 (Fast Math) shows relaxed tolerances. Table 38 specifies half-precision: "Most arithmetic operations are correctly rounded. Transcendental functions achieve <=1 ulp."
**Tags**: precision,ulp,fast-math,sin,cos,pow,half-precision,numerical-accuracy

## Finding 212: MSL 4.0 introduces <metal_tensor> header with native tensor types. Declared as tensor<device half, dextents<int, 2>> for device memory tensors. Local tensors from arrays: auto t = tensor(array, extents<int, WIDTH, 1>()). MTLTensor API: rank, extents, dataType, usage flags (ML/Compute/Render combinable). Created from MTLDevice (optimized layout) or MTLBuffer (explicit strides).
**Confidence**: verified
**Source**: Combine Metal 4 machine learning and graphics - WWDC25
**Evidence**: WWDC25 session 262: '#include <metal_tensor>' and 'tensor<device half, dextents<int, 2>> weights [[buffer(0)]]'. MTLTensorDescriptor with rank, extents, dataType. Tensors from device get optimized internal layout.
**Tags**: metal-4,mtl-tensor,metal-tensor,msl-4.0,multi-dimensional

## Finding 213: Metal 4 cooperative tensors replace simdgroup_matrix. Partition storage among participating threads in local registers. Current limitation: cooperative tensors cannot be used as inputs for matrix multiplication directly. llama.cpp uses tile sizes 128x64x32 with Metal 4 tensor API. On M5: ~4x prefill speedup, 3.65x tokens/sec for Qwen 8B vs simdgroup_matrix. Disabled by default on M1-M3 due to performance considerations.
**Confidence**: high
**Source**: llama.cpp PR #16634: initial Metal4 tensor API support
**Evidence**: llama.cpp PR #16634: 'initial Metal4 tensor API support.' M5 benchmarks: gpt-oss 20B MXFP4 MoE: 846.69 vs 415.45 t/s for pp512. Apple neural accelerators benchmark: 'cooperative tensors currently cannot be used as inputs for matrix multiplication.'
**Tags**: cooperative-tensor,simdgroup-matrix,metal-4,m5,llama-cpp,neural-accelerator

## Finding 214: M5 GPU neural accelerators: ~128 matrix FMAs per compute partition per cycle, ~1024 FP16 FLOPS/core/cycle. Projected M5 Max (40 cores, ~1750 MHz): ~70 TFLOPS FP16 matrix, ~130 TFLOPS INT8. MLX benchmarks: 3.52-3.62x time-to-first-token speedup over M4 for LLMs (Qwen 8B BF16: 3.62x). Sub-10s TTFT for dense 14B models. Bandwidth may be limiting factor (~93 GB/s needed per core).
**Confidence**: verified
**Source**: Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU
**Evidence**: Apple ML Research: Qwen 1.7B: 3.57x, Qwen 8B BF16: 3.62x, Qwen 30B MoE: 3.52x TTFT speedup. Neural accelerators benchmark: '~70 TFLOPS FP16, ~130 TFLOPS INT8 projected M5 Max.' Bandwidth constraint: '93.44 GB/s inbound per GPU core.'
**Tags**: m5,neural-accelerators,tflops,fp16,int8,bandwidth-limited,mlx

## Finding 215: Metal 4 unified compute encoder consolidates compute, blit, and acceleration structure operations. Commands run concurrently by default. Pass barriers provide explicit stage-to-stage sync only where data dependencies require. Automatic hazard tracking removed for all MTL4 pipelines -- developers must perform manual synchronization. MSL kernel code unchanged; dispatch model enables better pipelining.
**Confidence**: verified
**Source**: Discover Metal 4 - WWDC25
**Evidence**: WWDC25 session 205: unified compute encoder 'manages blit and acceleration structure in addition to compute.' MoltenVK #2560: 'Automatic hazard tracking removed for all MTL4 pipelines/command encoders.'
**Tags**: unified-encoder,barrier,concurrent,manual-hazard-tracking,metal-4

## Finding 216: Metal shader logging via MTLLogState (from MTLLogStateDescriptor). Environment variable MTL_LOG_LEVEL=MTLLogLevelDebug captures shader logs for system Console. Available since Metal 3, continues in Metal 4. Metal 4 PSO creation for tensor ops takes 1-2 seconds per PSO. bfloat16 tensor API missing from MetalPerformancePrimitives v1.0 (macOS 26.0), expected in 26.1+.
**Confidence**: verified
**Source**: Logging shader debug messages - Apple Developer Documentation
**Evidence**: Apple Developer Docs: MTLLogState for shader logging. llama.cpp PR #16634: 'Takes 1-2 seconds per PSO containing tensor operations.' 'MetalPerformancePrimitives v1.0 lacks bfloat tensor API support.'
**Tags**: logging,mtllogstate,debug,pso-creation,bfloat16,metal-4

## Finding 173: MSL is based on C++14 with these features NOT supported: lambda expressions (prior to Metal 3.2), dynamic_cast, RTTI, new/delete operators, noexcept, goto, register storage, thread_local, virtual functions, derived classes, exception handling, and the C++ Standard Library. Metal 3.2+ added lambda support. MSL 4.0 upgrades base to C++17.
**Confidence**: verified
**Source**: Metal Shading Language Specification Version 4
**Evidence**: The MSL specification explicitly lists these restrictions. MSL replaces the C++ Standard Library with its Metal Standard Library. These restrictions exist because GPUs lack hardware for dynamic dispatch, heap allocation, and exception unwinding.
**Tags**: msl,c++14,c++17,restrictions,language-features

## Finding 174: MSL half is IEEE 754 binary16 (10-bit mantissa, 5-bit exponent, range ~6.1e-5 to 65504). bfloat (MSL 3.1+) is brain float (7-bit mantissa, 8-bit exponent) with same dynamic range as float32 but less precision. Type conversions between half and float are free (zero performance cost) on Apple Silicon.
**Confidence**: verified
**Source**: Learn performance best practices for Metal shaders (Apple Tech Talk)
**Evidence**: WWDC tech talk confirms 'Type conversions between half/float are free (no performance cost).' bfloat uses 8-bit exponent like float32, giving range ~1.2e-38 to 3.4e+38.
**Tags**: half,bfloat,precision,types,ml,performance

## Finding 175: Using ushort/half (16-bit) types reduces register pressure and improves occupancy. At low occupancy, F16 has ~0.56 cycle latency penalty vs F32's ~0.84 cycle per register dependency. Apple Silicon ALU has native 16-bit register units. FP16 FFMA has 1-cycle throughput vs FP32 at 2-cycle. Apple Family 9 (M3+) can issue FP32 and FP16 simultaneously, delivering up to 2x ALU performance.
**Confidence**: verified
**Source**: Philip Turner's metal-benchmarks - Apple GPU microarchitecture
**Evidence**: Philip Turner's metal-benchmarks: F16 FMA latency 2.16-2.18 cycles vs F32 at 2.20-2.21 cycles. Apple tech talk: 'Use half and short for arithmetic wherever you can. Energy wise, half is cheaper than float.' Apple Family 9 tech talk confirms simultaneous FP32+FP16 dispatch.
**Tags**: half,ushort,precision,throughput,register-pressure,occupancy,apple-family-9

## Finding 176: MSL supports C++ templates for non-entry-point functions. Kernel entry points cannot be directly templated -- macro-based instantiation generates typed kernel variants. MLX uses instantiate macros for float/half/bfloat16_t. The MLX custom kernel API uses template=[('T', dtype)] for compile-time specialization.
**Confidence**: verified
**Source**: MLX Metal kernel source - conv.metal
**Evidence**: MLX conv.metal uses macros like instantiate_naive_unfold_nd_dims(name, itype). llama.cpp uses the same pattern in ggml-metal.metal.
**Tags**: templates,kernel-instantiation,macros,type-generic,mlx

## Finding 177: Avoid uint for device memory offsets -- use signed int instead. Apple GPU has dedicated addressing circuitry for signed offsets but requires emulation for unsigned. Also avoid char/uchar (8-bit) for arithmetic -- native operation width is 16-bit, so 8-bit requires emulation. Use signed int loop variables to enable vectorized loads (uint wrapping behavior prevents vectorization).
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Apple Tech Talk
**Evidence**: WWDC16 session 606: 'Avoid uint offsets for device memory; use signed int instead.' 'Avoid char (8-bit) since native operations are 16-bit.' Metal Compute on MacBook Pro tech talk: 'wrapping behavior of uint disables vectorized loads.'
**Tags**: uint,signed-int,addressing,char,8-bit,performance-pitfall,vectorization

## Finding 725: MSL does not support native atomic operations on float or half types — even in MSL 4.0 / Metal 4. Supported atomic types: atomic_int, atomic_uint (Metal 1.0), atomic_bool and atomic<T> for int/uint/bool (MSL 2.0). Metal 3.1 texture atomics support only int, uint, ulong. Developers must use CAS loop workarounds with as_type bit reinterpretation.
**Confidence**: verified
**Source**: Atomics proposal - gpuweb/gpuweb
**Evidence**: gpuweb#1360: Metal provides only "atomic_int" and "atomic_uint." Apple Forums thread 69703: "no native support for atomic operations on halfs or floats." MoltenVK #1939: Metal 3.1 texture atomics only "int, uint, ulong color types."
**Tags**: atomics,float,half,limitations,cas-loop,workaround

## Finding 712: float3 has size/alignment of 16 bytes (padded to float4), while packed_float3 is 12 bytes/4-byte aligned with no padding. Using float3 for concurrent loading into dense shared arrays causes data races — kernel assumes 12-byte copies but float3 reads/writes 16 bytes. Real bugs in Apache TVM (PR #7830) and Gaussian splatting CUDA-to-Metal ports. Fix: use packed_float3 or raw float arrays.
**Confidence**: verified
**Source**: Fix Metal accuracy problem caused by dtype3 vectors usage
**Evidence**: TVM PR #7830: "Using float3 for loading data concurrently into dense array shared between threads can lead to data race, as float3 has size/alignment equal to 16 bytes while kernel assumes 12 bytes."
**Tags**: packed-vector,float3,packed_float3,alignment,data-race,correctness

## Finding 713: Packed vector types (packed_float2, packed_float3, packed_half4) guarantee contiguous memory layout without padding, matching PyTorch tensors and CPU structs. Recommended pattern: packed types in buffer structs for CPU-GPU data layout, construct aligned types (float3, half4) inside kernel for computation. Packed types have scalar alignment (4 bytes for float), aligned vectors use power-of-two (float3/float4 = 16 bytes).
**Confidence**: high
**Source**: Things April 2024: Gaussian splatting, Metal vs. CUDA
**Evidence**: Apple Forums thread 64057: "A packed type guarantees all floats are next to each other in adjacent memory locations." WWDC16 session 606 demonstrates constructing float3 from packed_float3 inside shader.
**Tags**: packed-vector,alignment,cpu-gpu-sharing,pytorch,data-layout

## Finding 217: Production MSL kernel pattern (MLX/llama.cpp): const device T* inputs, device T* outputs, const constant ParamStruct* uniforms, threadgroup T shared[], function constants for config, template functions with macro-stamped entry points, simd_sum() for reductions, threadgroup_barrier(mem_flags::mem_threadgroup) for sync. llama.cpp Q4_0 kernel on M2 Max: 292 GB/s out of 300 GB/s theoretical.
**Confidence**: verified
**Source**: MLX Metal kernel source - conv.metal
**Evidence**: MLX conv.metal: const device T* in, device T* out, const constant MLXConvParams<N>*, threadgroup T ins[], constant int [[function_constant(00)]]. llama.cpp: 292 GB/s from 3.8GB Q4_0 7B model bandwidth measurement.
**Tags**: mlx,llama-cpp,kernel-patterns,best-practices,production-code,bandwidth

## Finding 218: Target occupancy 1,000-2,000 concurrent threads per shader core for complex compute kernels. Apple GPUs support up to 3,072 threads/core depending on chip. Higher occupancy hides latency from barriers and memory operations. Per-core I/O bandwidth is only 32 B/cycle (identical to CPU core) -- GPU advantage comes from many cores (10-40), not per-core bandwidth.
**Confidence**: verified
**Source**: WWDC22: Scale Compute Workloads Across Apple GPUs
**Evidence**: WWDC22 session 10159: 'Target occupancy: 1K to 2K concurrent threads per shader core for complex kernels.' metal-benchmarks: 'GPU cores identical to CPU cores in I/O bandwidth, only differ in compute power.'
**Tags**: occupancy,threads-per-core,bandwidth,cpu-vs-gpu,roofline

## Finding 718: MSL select(a, b, c) returns b when c is true, a when false. Compiles to Apple GPU's native FCMPSEL (compare-and-select) instruction: ~4.74 cycles float32, ~2.17 cycles float16. WWDC16 warns against manual branchless tricks: "A8+ GPUs have very fast select instructions... compiler can't see through this cleverness." Ternary operator (condition ? a : b) compiles to the same instruction.
**Confidence**: verified
**Source**: Advanced Metal Shader Optimization - WWDC16
**Evidence**: G13 docs: FCMPSEL "if cc.compare(A, B): D = X; else: D = Y". metal-benchmarks: FCMPSEL32: 1,1 throughput, ~4.74 latency. FCMPSEL16: ~2.17 latency. WWDC16/606: "A8 and later GPUs have very fast select instructions."
**Tags**: select,FCMPSEL,branchless,conditional,ternary,utility-function

## Finding 507: ShaDiv generates test programs with carefully designed control and data flow divergence patterns to stress GPU shader compiler back-ends (the stage after IR optimization that generates machine code). Found 14 bugs across Intel, NVIDIA, AMD, and ARM shader compilers, with 12 in the back-end specifically. Achieves 25% coverage increase in back-end components and finds 4x more back-end bugs than existing tools. Key finding: SIMT divergence handling is the most bug-prone area of shader compilers.
**Confidence**: high
**Source**: Divergence-Aware Testing of Graphics Shader Compiler Back-Ends
**Evidence**: PLDI 2025 paper. Uses divergence and liveness analysis tailored to SIMT execution model to guide test generation. Programs with diverse divergence characteristics stress back-end register allocation, instruction scheduling, and reconvergence optimizations that are unique to GPU compilers.
**Tags**: shader-compiler,divergence,SIMT,back-end,register-allocation,testing,PLDI

## Finding 506: DarthShader is the first fuzzer combining IR-based mutators with AST-based mutators for GPU shader compilers. Found 39 bugs across Chrome, Firefox, and Safari (15 CVEs in Chrome alone). Key insight: shader compilation involves multiple IR translation layers (WGSL -> SPIR-V -> backend IR -> machine code), and each translation boundary is a bug-rich attack surface. The IR-based mutation approach is more effective at finding deep compiler bugs than AST-only approaches.
**Confidence**: high
**Source**: DarthShader: Fuzzing WebGPU Shader Translators & Compilers
**Evidence**: Tested against Tint (Chrome/Dawn), Naga (Firefox/wgpu), and WebKit shader compilers. The hybrid IR+AST mutation strategy discovers bugs that purely syntactic fuzzers miss because IR mutations can create semantically valid but structurally unusual programs that stress compiler optimization passes.
**Tags**: shader-compiler,fuzzing,WebGPU,SPIR-V,IR,security,compiler-bugs

## Finding 192: SIMD-group reduction functions (A14/Apple family 7+): simd_sum, simd_product, simd_minimum, simd_maximum, simd_and, simd_or, simd_xor. All work on float/int scalar/vector types (bitwise only on integers). Results broadcast to all active threads. Inactive threads skipped. Single instruction execution across all 32 threads.
**Confidence**: verified
**Source**: Discover Metal enhancements for A14 Bionic (Apple Tech Talk)
**Evidence**: Apple A14 tech talk: 'Metal now provides SIMD scope reduction instructions' listing all functions. 'All operations work on floating point and integer scalar and vector types (except bitwise operations, which work only on integer types).'
**Tags**: simd-reduction,simd-sum,simd-min,simd-max,simd-product,a14

## Finding 193: SIMD prefix scan functions: simd_prefix_exclusive_sum(T) and simd_prefix_inclusive_sum(T) compute running sums across SIMD lanes. On M1, maximum scan rate ~4.8 billion 32-bit values/sec (limited by 58 GB/s bandwidth with reduce-then-scan strategy). Exclusive and inclusive variants compile to nearly identical machine code.
**Confidence**: high
**Source**: Efficient Parallel Prefix Sum in Metal for Apple M1
**Evidence**: M1 scan benchmarks: raking and Blelloch algorithms achieved just over 18 GB/sec at 128M elements.
**Tags**: simd-prefix-sum,scan,parallel-reduction,m1,performance

## Finding 194: SIMD shuffle functions: simd_shuffle(data, index) -- arbitrary lane access; simd_shuffle_up/down(data, delta) -- shift; simd_shuffle_xor(data, mask) -- butterfly pattern; simd_shuffle_rotate_up/down(data, delta) -- circular. A15 added simd_shuffle_and_fill_up/down(data, fill_data, delta) with optional modulo argument for user-specified vector widths. At ISA level, shuffle operates on 4-thread quads within the 32-thread SIMD group.
**Confidence**: verified
**Source**: Discover advances in Metal for A15 Bionic (Apple Tech Talk)
**Evidence**: Apple A15 tech talk: 'new SIMD shuffle and fill instructions have optional modulo argument.' Dougall Johnson G13 reference: shuffle reads indices from operand B using 3-bit mask per quad.
**Tags**: simd-shuffle,simd-permute,data-movement,a15,shuffle-and-fill,quad

## Finding 195: Apple GPU divergent thread handling uses an execution mask stack in register r0l. Flow control instructions (if_icmp, else_icmp, while_icmp, pop_exec) manipulate this stack. When thread's depth counter reaches zero, it becomes inactive. The 32-bit exec_mask is the only known mechanism for thread masking, exclusively manipulated by execution mask stack instructions. All 32 threads execute same instruction stream; divergence managed through conditional register updates.
**Confidence**: verified
**Source**: Apple G13 GPU Architecture Reference (Dougall Johnson)
**Evidence**: G13 reference: 'pop_exec decrements r0l by n; deactivates threads when value reaches zero. if_icmp/if_fcmp: conditional branching; pushes new mask level.'
**Tags**: divergence,execution-mask,control-flow,isa,hardware,r0l

## Finding 734: simd_ballot(bool expr) returns a simd_vote object (not raw integer like CUDA's __ballot_sync). simd_vote wraps __METAL_VOTE_T__ and provides .all() and .any() methods. Available when __HAVE_SIMDGROUP_BALLOT__ defined (Apple GPU family 4+, A11+). simd_active_threads_mask() returns simd_vote of active threads (Metal equivalent of CUDA __activemask). Companion: simd_all(bool) and simd_any(bool) return simple bool for unanimous/existential voting.
**Confidence**: verified
**Source**: Metal Standard Library Header - metal_simdgroup
**Evidence**: From Metal standard library header metal_simdgroup: simd_ballot calls __metal_simd_ballot(expr, vote_t(0)). simd_active_threads_mask calls __metal_simd_active_threads_mask(vote_t(0)). simd_vote has explicit conversion to/from vote_t.
**Tags**: simd_ballot,simd_vote,simd_active_threads_mask,simd_all,simd_any,A11

## Finding 735: simd_is_helper_thread() returns bool, gated by __HAVE_SIMDGROUP_REDUCTION__. In fragment shaders, helper threads are invocations for pixels outside the primitive whose writes are discarded and atomics produce undefined results. In compute kernels, always returns false — compute dispatches have no helper invocations. Exists in compute for API uniformity only.
**Confidence**: verified
**Source**: Metal Standard Library Header - metal_simdgroup
**Evidence**: Metal header: simd_is_helper_thread calls __metal_simd_is_helper_thread(). Apple docs: helper threads are fragment function invocations near primitive edges that produce no output.
**Tags**: simd_is_helper_thread,fragment,compute,helper-invocation

## Finding 742: simd_shuffle_and_fill_down/up(data, filling_data, delta, [modulo]) combines shuffle with data injection from a second source. Fill variants replace shifted-out positions with filling_data (unlike plain shuffle which retains originals). Optional modulo splits 32-thread SIMD into smaller logical vectors. Apple demonstrated 84% reduction in texture samples for 5x5 edge detection convolution. Use cases: sliding-window convolutions, image processing, ML inference with spatial locality.
**Confidence**: verified
**Source**: Discover advances in Metal for A15 Bionic - Apple Tech Talk
**Evidence**: Apple Tech Talk "Discover advances in Metal for A15 Bionic": "84% reduction in number of samples per SIMD group" for 5x5 convolution. Header has two overloads each with optional modulo defaulting to __metal_get_simdgroup_size().
**Tags**: simd_shuffle_and_fill,A15,convolution,sliding-window,84-percent,modulo

## Finding 740: simd_shuffle_xor(T data, ushort mask) exchanges values between threads whose lane IDs differ by XOR with mask. Butterfly reduction: offsets 1,2,4,8,16 pair threads for combining — 5 steps for full 32-thread reduction. Complete shuffle variants: simd_shuffle (arbitrary), simd_shuffle_down/up (delta), simd_shuffle_xor (mask), simd_shuffle_rotate_down/up (__HAVE_SIMDGROUP_SHUFFLE_ROTATE__), simd_shuffle_and_fill_down/up (__HAVE_SIMDGROUP_SHUFFLE_AND_FILL__) with optional modulo parameter.
**Confidence**: verified
**Source**: Metal Standard Library Header - metal_simdgroup
**Evidence**: Metal header: simd_shuffle_xor calls __metal_simd_shuffle_xor(data, mask). GitHub gist: for (uint offset=simd_size/2; offset>0; offset/=2) val += simd_shuffle_down(val, offset).
**Tags**: simd_shuffle_xor,butterfly-reduction,complete-shuffle-list,fill-variants

## Finding 196: simdgroup_matrix<T, Rows, Cols> only supports 8x8 dimensions. Supported types: half, float, bfloat. Each 8x8 matrix has 64 elements distributed across 32 SIMD lanes (2 elements per thread, arranged as 1 row x 2 cols). Thread element coordinates computed from simd_lane_id. Introduced with A14 Bionic (Apple family 7).
**Confidence**: verified
**Source**: MLX STEEL GEMM MMA Implementation
**Evidence**: MLX mma.h: 'kFragRows = 8 and kFragCols = 8 are the only supported dimensions.' kElemsPerFrag = 64/32 = 2. Thread mapping: qid = simd_lane_id/4; fm = (qid&4) + ((simd_lane_id/2)%4); fn = (qid&2)*2 + (simd_lane_id%2)*2.
**Tags**: simdgroup-matrix,8x8,element-types,thread-distribution,half,float,bfloat

## Finding 197: simdgroup_multiply_accumulate(D, A, B, C) performs D = A*B+C for 8x8 matrices. Not a dedicated tensor core -- instead decreases register pressure and improves ALU utilization from ~25% to ~80% in existing FP32 pipelines. On A14: 37% GEMM improvement, 36% CNN improvement, 22% Inception V3 training improvement over A13.
**Confidence**: verified
**Source**: Discover Metal enhancements for A14 Bionic (Apple Tech Talk)
**Evidence**: Apple A14 tech talk: 'General matrix multiplication: 37% average improvement over A13.' metal-benchmarks: 'Apple''s tensor core is simdgroup_matrix, which decreases register pressure and improves ALU utilization in existing FP32 pipelines.'
**Tags**: simdgroup-multiply-accumulate,performance,a14,alu-utilization,not-tensor-core

## Finding 198: FP16 simdgroup_matrix has ~2x effective throughput over FP32 due to halved register usage, not raw ALU speed difference. On M1+, FP16 FFMA=102.5 and FP32 FFMA=101.7 (nearly equal raw throughput), but FP16 uses half register space. GEMV operations are memory-bound and do NOT benefit from simdgroup_matrix. Metal FlashAttention achieves 83-86% ALU utilization on M1 Max.
**Confidence**: high
**Source**: Philip Turner's metal-benchmarks
**Evidence**: metal-benchmarks: FP16 FFMA=102.5, FP32=101.7. Philip Turner MLX issue #171: 'Matrix-vector operations are memory bandwidth bound and would not benefit from SIMD-group matmul.' metal-flash-attention: '4400 gigainstructions/sec on M1 Max (83% utilization).'
**Tags**: fp16-vs-fp32,throughput,register-pressure,gemv,flashattention

## Finding 199: simdgroup_async_copy is an undocumented hardware feature (A14+) that overlaps compute and memory load instructions. Metal FlashAttention leverages this for 43-120% faster performance across Apple devices by prefetching the next tile while computing the current one.
**Confidence**: high
**Source**: Integrating Metal FlashAttention (Draw Things Engineering)
**Evidence**: Draw Things engineering blog: 'leverages the simdgroup_async_copy API (since A14), an undocumented hardware feature that overlaps compute and load instructions.'
**Tags**: simdgroup-async-copy,undocumented,overlap-compute-load,flashattention,prefetch

## Finding 739: MLX Steel handles non-8-aligned K dimensions via K_aligned_ flag. Last iteration uses load_safe() with clamped tile dimensions: lbk = remaining K elements. load_safe performs bounds-checked loading that zero-fills elements outside valid matrix region, ensuring correct simdgroup_multiply_accumulate results. Functionally equivalent to zero-padding inputs but done lazily at load time. M/N edges similarly use clamped tgp_bm/tgp_bn.
**Confidence**: verified
**Source**: MLX Steel GEMM Kernel Header (gemm.h)
**Evidence**: MLX gemm.h: if (!K_aligned_) { tile_dims_A_last = transpose_a ? short2(tgp_bm, lbk) : short2(lbk, tgp_bm); loader_a.load_safe(tile_dims_A_last); }
**Tags**: mlx,steel,edge-handling,non-aligned,K-dimension,load_safe,zero-padding

## Finding 744: simdgroup_matrix<T,8,8> internal storage is vec<T,64> distributed across 32 threads — each thread holds (8*8)/32 = 2 elements, accessed via thread_elements(). Per-thread coordinate mapping (from MLX): qid=simd_lane_id/4; fm=(qid&4)+((simd_lane_id/2)%4); fn=(qid&2)*2+(simd_lane_id%2)*2. Diagonal init: __metal_simdgroup_matrix_8x8_init_diag(value). Filled init: make_filled_simdgroup_matrix<T,8,8>(value).
**Confidence**: verified
**Source**: Metal Standard Library Header - metal_simdgroup_matrix
**Evidence**: metal_simdgroup_matrix header: typedef vec<T, Cols*Rows> type in _simdgroup_matrix_storage_type. thread_elements() returns reference to storage. MLX mma.h: kElemsPerFrag = (kFragRows*kFragCols)/32 = 2.
**Tags**: simdgroup_matrix,storage,thread_elements,2-elements-per-thread,register-layout

## Finding 736: simdgroup_load/store accept device and threadgroup pointers with: elements_per_row (stride in elements, default=Cols=8), matrix_origin (ulong2 offset within larger matrix), transpose_matrix (bool). No explicit alignment enforced at API level. Default bounds check mode is __METAL_SIMDGROUP_LOAD_STORE_BOUNDS_CHECK_NONE__ — out-of-bounds access is undefined behavior. Internal calls: __metal_simdgroup_matrix_8x8_load/store.
**Confidence**: verified
**Source**: Metal Standard Library Header - metal_simdgroup_matrix
**Evidence**: From metal_simdgroup_matrix header: _simdgroup_load_impl uses BOUNDS_CHECK_NONE mode. Store accesses via a.thread_elements(). Both device and threadgroup overloads have identical signatures.
**Tags**: simdgroup_matrix,load,store,elements_per_row,matrix_origin,alignment

## Finding 741: On Apple 7/8 (M1/A15+), simdgroup_matrix achieves nearly identical RAW throughput for FP32 (101.7 Matrix FFMA/core-cycle) and FP16 (102.5). Apple reuses FP32 ALU pipelines rather than separate matrix hardware. Earlier gens: A11-A13 FP32=43.6, FP16=83.7. For M1 at 1.278 GHz with 8 cores: peak ~2617 GFLOPS for both — 100% of advertised FP32. Note: FP16 still has EFFECTIVE throughput advantage via halved register pressure → higher occupancy.
**Confidence**: verified
**Source**: metal-benchmarks: Apple GPU microarchitecture
**Evidence**: metal-benchmarks: "Matrix FFMA16=102.5, Matrix FFMA32=101.7 for A15+/M1+. Matrix FFMA32=43.6 for A11-A13." "Apple's tensor core is simdgroup_matrix which decreases register pressure and improves ALU utilization in existing FP32 pipelines."
**Tags**: simdgroup_matrix,throughput,FFMA,FP32,FP16,M1,A15,raw-vs-effective

## Finding 748: MSL texture access qualifiers for compute: access::read (integer coordinate .read() only), access::write (.write() only), access::read_write (both), access::sample (filtered normalized coordinates via .sample() with sampler). Access qualifier restricts callable methods, giving compiler optimization info. .read() = nearest-neighbor integer coords; .sample() = bilinear/trilinear with normalized coords.
**Confidence**: high
**Source**: Fundamentals of Image Processing in Metal - Metal by Example
**Evidence**: Metal by Example: "access template parameter describes access to texture data: read for reading only; write for destination; read_write for both." "We restrict the set of functions callable on these parameters."
**Tags**: texture,access-qualifier,read,write,read_write,sample,compute

## Finding 747: Apple Silicon performs automatic lossless compression on GPU-private textures, reducing bandwidth. Shared/managed textures: explicitly compress via optimizeContentsForGPUAccess() on blit encoder. Requires usage shaderRead or renderTarget. Decompression automatic and transparent. Combined with ASTC/BC lossy compression (4:1 to 36:1 ratios) for additional bandwidth savings.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Tech Talks
**Evidence**: Tech Talk 10580: "Apple Silicon can perform lossless compression of a texture to further reduce memory bandwidth." "Private textures compressed by default; shared/managed textures via optimizeContentsForGPUAccess." "BC and ASTC have compression ratios from 4:1 through 36:1."
**Tags**: texture,compression,lossless,bandwidth,ASTC,optimizeContentsForGPUAccess

## Finding 746: Metal automatically twiddles (Morton-order / Z-order) texture data on upload, reordering texels for optimal random 2D access patterns. Improves cache efficiency for spatial locality compared to linearly-addressed buffers. Transparent to kernel — no shader code changes needed. Combined with separate L1 cache, textures can significantly outperform buffers for 2D access patterns.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Tech Talks
**Evidence**: Tech Talk 10580: "Texture data can be twiddled, and Metal will do this automatically on upload. Twiddling means texels are ordered more optimally for random access pattern and can help improve cache efficiency, giving another performance gain over a regular buffer."
**Tags**: texture,twiddling,morton-order,spatial-locality,cache,compute

## Finding 200: Textures have a dedicated L1 cache separate from buffer L1 cache, effectively doubling on-chip cache capacity when both used simultaneously. Textures benefit from lossless bandwidth compression (enabled by default for GPU-private textures) and automatic spatial locality optimization via texture twidling. Buffers get none of these hardware optimizations.
**Confidence**: verified
**Source**: Create image processing apps powered by Apple silicon - WWDC21
**Evidence**: Apple Tech Talk 'Metal Compute on MacBook Pro': textures have dedicated separate cache. WWDC21 session 10153: automatic lossless bandwidth compression for textures, direct tile memory access, buffers thrash cache hierarchy.
**Tags**: texture,buffer,cache,compression,bandwidth,2d-data

## Finding 201: ASTC texture compression can store LLM weights at ~3.6 bits/weight (6x6 block HDR-ch mode) with Apple GPU's fixed-function ASTC hardware decoder decompressing 'for free' during texture sampling -- zero ALU cycles. Provides 4-5x memory-and-bandwidth savings with less than 1 percentage point loss on MMLU. Available since A7 chip.
**Confidence**: high
**Source**: MLX Issue #2418: ASTC weight compression + hardware decoding
**Evidence**: MLX issue #2418 proposes encoding model weights as ASTC textures. 'ASTC decoder is already baked into the texture-sampling path on all Apple Silicon.' 'Metal/MPS can sample .astc textures with one line of shader code.'
**Tags**: astc,texture-compression,llm,quantization,hardware-decode,bandwidth

## Finding 708: Metal threadgroup_barrier combines execution and memory synchronization inseparably — ALL threads in a threadgroup must execute the barrier or the kernel deadlocks. Cannot be placed in divergent code paths (e.g., inside if-statement where only some threads enter). Unlike Vulkan memoryBarrierBuffer which is a pure memory barrier. Intel and Apple GPUs deadlock; AMD GPUs sometimes continue (undefined behavior).
**Confidence**: verified
**Source**: memoryBarrierBuffer != threadgroup_barrier(mem_device) - MoltenVK
**Evidence**: MoltenVK issue #973: kernel void with threadgroup_barrier inside conditional — threads not reaching barrier hang those that did. Proposes VK_KHR_portability_subset feature bit to document incompatibility. Critical for ring buffer implementations that might use barriers conditionally.
**Tags**: threadgroup-barrier,deadlock,divergent-code,synchronization,Metal-vs-Vulkan

## Finding 731: Maximum threadgroup memory per threadgroup (MTLDevice.maxThreadgroupMemoryLength): 32,768 bytes (32KB) on all Apple Silicon Macs (M1+, Apple Family 7+). Older mobile (A7-A10X, Family 1-3): 16,384 bytes (16KB). High usage directly reduces occupancy — the only way to increase occupancy is to reduce shared memory. M3+ can dynamically allocate more on-chip memory if registers/tile underutilized, but 32KB API limit remains.
**Confidence**: verified
**Source**: maxThreadgroupMemoryLength - Apple Developer Documentation
**Evidence**: oscarbg/metal2caps: 16384 for iPad Air 1 (A7). 32768 confirmed for M-series. Tech Talk 10580: "With high thread-group memory usage, the only way to increase occupancy is to reduce shared memory used."
**Tags**: threadgroup-memory,size-limits,occupancy,32kb,16kb

## Finding 715: Apple GPU transcendental instruction latency: EXP2/LOG2 ~4 cycles (16-bit and 32-bit). RSQRT ~8 cycles. RECIP ~6 cycles (precise: ~10.5 cycles). SIN/COS ~14 cycles throughput, 23-27 cycle adjusted latency. Basic ALU (FADD/FMUL/FFMA) = 1-2 cycles. Transcendentals are 2-7x slower than basic arithmetic, with sin/cos most expensive.
**Confidence**: verified
**Source**: metal-benchmarks - Apple GPU microarchitecture
**Evidence**: metal-benchmarks: EXP2_32: 4-cycle throughput, ~4.31 latency. RSQRT32: 8-cycle throughput, ~8.99 latency. RECIP16: 6-cycle, ~6.50. SIN32: 14.28-cycle throughput, ~23.04-27.35. Precise RECIP32: 10.46-cycle. FADD32/FMUL32: 2,1 throughput, ~2.20 latency.
**Tags**: instruction-latency,transcendental,exp,log,rsqrt,sin,cos,alu-cycles

## Finding 751: Metal visible function tables support 3 compilation models: (1) Single/AIR — statically linked, maximum LTO optimization, best runtime, largest binary, longest compile. (2) Separate/Binary — precompiled, faster pipeline creation, runtime call overhead, shared across pipelines. (3) Incremental — add new binary functions without pipeline recreation. [[function_groups("name")]] at call sites helps compiler optimize. Binary archives cache compiled functions.
**Confidence**: verified
**Source**: Get to know Metal function pointers - WWDC20
**Evidence**: WWDC20-10013: "separately compiled pipeline has some runtime overhead calling binary functions. Fully specialized single-compilation offers best performance." WWDC21-10229: "AIR: slower compile, better runtime. Binary: faster compile, optimized dispatch."
**Tags**: visible-functions,function-table,compilation-model,binary-archive

