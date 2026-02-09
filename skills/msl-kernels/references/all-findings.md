# msl-kernels — All Findings (49)

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

