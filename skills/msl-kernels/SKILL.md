---
name: msl-kernels
layer: 1
description: >
  This skill should be used when the conversation involves Metal Shading Language (MSL),
  writing or optimizing compute kernels, address spaces (device, constant, threadgroup, thread),
  function constants, atomics and synchronization barriers, simdgroup matrix operations,
  SIMD intrinsics, texture and imageblock access, kernel compilation pipelines, register pressure,
  or Metal 4 MSL features like cooperative tensors. Trigger keywords: MSL, Metal Shading Language,
  kernel, address space, threadgroup, device, constant, function constants, atomics,
  simdgroup, simdgroup_matrix, compute kernel, half precision, AIR, visible functions.
skills:
  - gpu-silicon
  - metal-compute
cross_references:
  - simd-wave
  - gpu-perf
  - metal4-api
---

# MSL Kernels

This skill provides expertise in the Metal Shading Language (MSL) for writing high-performance compute kernels on Apple Silicon. It covers the full MSL language surface relevant to general-purpose GPU computing: address spaces, data types, synchronization primitives, SIMD-group operations, matrix intrinsics, compilation pipelines, and Metal 4 language extensions.

## Domain Overview

Metal Shading Language is a C++14-based language with GPU-specific extensions for programming Apple GPUs. For compute workloads, MSL provides a rich set of features including typed address spaces that map to different levels of the GPU memory hierarchy, SIMD-group intrinsics for warp-level communication, matrix multiply-accumulate operations, atomic operations for inter-thread coordination, and function constants for compile-time kernel specialization.

MSL address spaces are central to performance. The `device` address space provides read/write access to GPU buffers with no size restriction. The `constant` address space is read-only with a 64 KB per-argument limit but benefits from dedicated cache hardware. The `threadgroup` address space maps to on-chip shared memory (~32 KB per threadgroup). The `thread` (private) address space maps directly to the GPU register file (~208 KB per core). Misusing address spaces -- for example, dynamically indexing thread-local arrays -- causes register spilling to device memory with severe performance penalties.

MSL's type system favors 16-bit computation. The `half` type is IEEE 754 binary16 with native hardware support, and using `half`/`ushort` reduces register pressure, improving occupancy. Apple GPUs have dedicated hardware for FP16 operations at roughly 2x the throughput of FP32 in many workloads. Conversely, unsigned integer math (`uint`) should be avoided for memory offsets due to dedicated hardware for signed integer multiply-add on Apple GPUs.

Kernel compilation follows a multi-stage pipeline: MSL source compiles to AIR (Apple Intermediate Representation), then AIR compiles to GPU-native binary at pipeline state creation time. Function constants enable single-source kernels with compile-time specialization, avoiding the combinatorial explosion of preprocessor-based variants. Metal 4 introduces native tensor types and cooperative tensor operations that leverage M5's neural accelerators directly from MSL.

## Key Knowledge Areas

The knowledge base contains 95 findings across these MSL kernel topics:

- **MSL Language Fundamentals**: C++14 base with restrictions (no lambdas, exceptions, virtual, RTTI), template support for non-entry functions, entry point specialization via [[kernel]] attribute (Findings #173, #176)
- **Address Spaces**: device (read/write, unbounded), constant (read-only, 64KB limit), threadgroup (~32KB shared), thread (register file ~208KB/core), threadgroup_imageblock (tile memory in TBDR) (Findings #178-183)
- **Data Types & Precision**: half (IEEE binary16, ~6.1e-5 to 65504), 16-bit types halve register pressure, signed int preferred over uint for offsets (Findings #174, #175, #177)
- **Atomics & Synchronization**: atomic_int/uint (Metal 1.0), atomic<T> (Metal 3.1), threadgroup_barrier with mem_flags, no standalone memory fence, 64-bit atomic min/max on Apple8+ (Findings #184-191)
- **SIMD-Group Intrinsics**: Reductions (simd_sum, simd_min, etc.), prefix scans, shuffle operations, divergent thread execution mask handling (Findings #192-195)
- **simdgroup_matrix**: 8x8 only, half/float/bfloat types, multiply_accumulate D=A*B+C, FP16 ~2x throughput, undocumented simdgroup_async_copy (Findings #196-199)
- **Texture & Imageblock**: Separate L1 cache from buffers, ASTC compression for LLM weights at ~3.6 bits/weight (Findings #200-201)
- **Advanced Kernel Patterns**: Function constants for branch-free code (84% instruction reduction), visible functions, pragma unroll, kernel fusion, register pressure management (Findings #202-207)
- **Compilation Pipeline**: MSL -> AIR -> GPU binary, dynamic libraries, function constants vs macros, ALU instruction latencies (Findings #208-211)
- **Metal 4 MSL**: Native tensor types via <metal_tensor>, cooperative tensors replacing simdgroup_matrix, M5 neural accelerator integration, unified encoder from MSL perspective, shader logging (Findings #212-216)
- **Production Patterns**: MLX/llama.cpp kernel structure, target occupancy 1000-2000 threads/core (Findings #217-218)
- **Data Types & Precision Depth**: bfloat type (MSL 3.1, 1+8+7 format, M2+ FEAT_BF16 hardware), packed vector types (float3 vs packed_float3 padding/data race), integer type optimization (char/uchar do NOT save registers), 16-bit vs 32-bit ALU throughput (2x ratio) (Findings #710-#713, #718, #720-#721)
- **Math Functions & Free Modifiers**: fast_math ON by default (50%+ perf gain), free instruction modifiers (saturate, negate, abs are zero-cost), select function as native FCMPSEL instruction, transcendental latency (EXP2/LOG2 ~4 cycles, RSQRT ~8, RECIP ~6), ULP precision guarantees (Findings #714-#719)
- **Threadgroup Memory Banking**: M1 has 32 independent banks with stride-2 conflicts, M3 banking is fundamentally different, bank conflict avoidance via +1 padding per row, 32KB limit all devices, dynamic threadgroup memory via setThreadgroupMemoryLength (Findings #722-#723, #728, #731-#733)
- **Atomics Limitations**: Relaxed ordering only (no acquire/release/seq_cst even in MSL 4.0), no native float atomics, CAS loop as standard workaround for float atomics, global atomics identified by Apple as primary performance bottleneck, complete atomic function list (Findings #724-#727, #730)
- **SIMD Ballot & Vote**: simd_ballot returns simd_vote object (not raw integer like CUDA), simd_is_helper_thread returns false in compute context, G13 ISA ballot instructions (icmp_ballot, fcmp_ballot) (Findings #734-#735, #743)
- **SIMD Shuffle Patterns**: simd_shuffle_xor for butterfly reduction (exchange by XOR mask), simd_shuffle_and_fill for sliding window with data injection, SIMD shuffle vs threadgroup memory design tradeoff (256 bytes/cycle intra-SIMD vs slower inter-SIMD) (Findings #728, #740, #742)
- **simdgroup_matrix Internals**: Internal storage is vec<T,64> across 32 threads, load/store API (elements_per_row, transpose_matrix parameters), FP32 throughput matches FP16 (identical RAW throughput), A14 introduction showed 37% average improvement, MLX Steel GEMM tiling (BM x BN x BK hierarchy), edge handling (K_aligned_ flag, load_safe()) (Findings #736-#741, #744-#745)
- **Texture Compute Access**: Morton-order automatic twiddling for spatial locality, lossless compression for GPU-private textures, access qualifiers (read/write/read_write/sample) (Findings #746-#748)
- **Compilation Options**: MTLCompileOptions mathMode 3-value enum (safe/relaxed/fast replacing boolean), occupancy hints (optimization level, maxTotalThreadsPerThreadgroup), visible function compilation models (3 models: Single/AIR, stitched, dynamic), function pointer SIMD divergence (serializes, worst case 32 paths) (Findings #749-#752)
- **Metal 4 tensor_ops**: #include <metal_tensor>, matmul2d API, 2.4x benchmark vs simdgroup_matrix on M4 Max, M3 Dynamic Caching threadgroup impact (software cache may be counterproductive) (Findings #729, #753-#754)

## How to Query Knowledge

Use the portable KB CLI to search MSL kernel findings:

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "address space"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "simdgroup_matrix"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "function constants"
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill msl-kernels
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <finding-id>
```

The `search` command uses BM25-weighted FTS5 ranking, prioritizing claim text over evidence. The `skill` command returns all 95 msl-kernels findings in table format.

## Common Patterns & Quick Answers

**Q: What are the MSL address spaces and their performance characteristics?**
A: Four main spaces: `device` (read/write buffers, no size limit), `constant` (read-only, 64KB/arg, cached), `threadgroup` (shared on-chip ~32KB), `thread` (register file ~208KB/core). Also `threadgroup_imageblock` for TBDR tile memory. Choose based on access pattern and data size. (Findings #178-183, verified)

**Q: Should I use half or float for compute kernels?**
A: Prefer `half` (FP16) when precision allows. It halves register pressure, improving occupancy, and Apple GPUs have ~2x effective FP16 throughput. Range is ~6.1e-5 to 65504 with 10-bit mantissa. (Findings #174-175, #198, verified)

**Q: How do atomics and barriers work in MSL?**
A: Use `atomic_int`/`atomic_uint` (Metal 1.0) or `atomic<T>` (Metal 3.1). `threadgroup_barrier()` takes mem_flags: `mem_none` (execution only), `mem_threadgroup` (threadgroup fence), `mem_device` (device fence). Metal has NO standalone memory fence without execution barrier. For reductions, use hierarchical pattern: SIMD-level -> threadgroup -> device atomics. (Findings #184-188, verified)

**Q: What is the simdgroup_matrix API?**
A: `simdgroup_matrix<T, Rows, Cols>` supports 8x8 dimensions only. Types: half, float, bfloat (Apple9+). `simdgroup_multiply_accumulate(D, A, B, C)` computes D=A*B+C. FP16 matrices have ~2x throughput over FP32 due to halved register usage. No int8 support. (Findings #196-198, verified)

**Q: How do function constants improve kernel performance?**
A: Function constants enable compile-time specialization from a single MSL source. The compiler dead-strips unused branches entirely. Larian Studios achieved 84% instruction reduction using this pattern. Superior to preprocessor macros because one source compiles to many variants without rebuilding. (Findings #202, #210, verified)

**Q: What MSL features does Metal 4 add?**
A: Metal 4 MSL introduces `<metal_tensor>` header with native tensor types declared as `tensor<T, extents<Rows, Cols>>`. Cooperative tensors replace simdgroup_matrix with partitioned storage across SIMD lanes. On M5, these map to neural accelerator hardware (~128 matrix FMAs/partition/cycle, ~16x over pure ALU). Also adds shader logging via MTLLogState. (Findings #212-216, verified/high)

**Q: What does the MSL compilation pipeline look like?**
A: Three stages: (1) MSL source -> AIR (Apple Intermediate Representation) at build time or runtime, (2) AIR -> GPU binary at pipeline state creation, (3) GPU binary cached for reuse. Advanced workflows include dynamic libraries (precompiled reusable shaders), binary archives (serialized PSO cache), and stitched functions (link-time composition). (Findings #208-209, verified)

**Q: How do SIMD shuffle operations work?**
A: `simd_shuffle(data, index)` reads any lane by index. `simd_shuffle_up/down(data, delta)` shifts by constant offset. `simd_shuffle_xor(data, mask)` uses XOR on lane index (butterfly pattern). `simd_shuffle_rotate_up/down` wraps around SIMD width. All operate within the 32-thread SIMD group. (Finding #194, verified)

**Q: What's the recommended kernel structure for production code?**
A: Follow MLX/llama.cpp pattern: `const device T*` inputs, `device T*` outputs, `constant Params&` uniforms (under 64KB), `threadgroup T*` scratch. Target 1000-2000 concurrent threads per shader core for complex compute kernels. (Findings #217-218, verified)

**Q: Why avoid uint for memory offsets in MSL?**
A: Apple GPU has dedicated hardware for signed integer multiply-add but not for unsigned. Using `int` instead of `uint` for buffer offsets enables more efficient address calculation. (Finding #177, verified)

**Q: How does llama.cpp handle synchronization in Metal kernels?**
A: llama.cpp Metal kernels use zero atomic operations. They rely on `threadgroup_barrier(mem_threadgroup)` for intra-threadgroup sync and structure dispatches to avoid inter-threadgroup communication entirely. (Finding #191, verified)

**Q: Is fast_math on by default in MSL?**
A: Yes. fast_math is ON by default in Metal and provides 50%+ performance gain for math-heavy kernels. It enables fused multiply-add and relaxed precision. Metal 4 replaces the boolean flag with MTLCompileOptions.mathMode (safe/relaxed/fast 3-value enum). Disable with `-fno-fast-math` when exact IEEE 754 compliance is needed. (Finding #714, #749)

**Q: Do char/uchar save registers in MSL?**
A: No. On Apple GPUs, char/uchar do NOT reduce register usage — they still occupy full 32-bit register slots. Only half/ushort (16-bit) types provide 2x register savings. Apple GPUs have 2x ALU throughput for 16-bit operations vs 32-bit. (Findings #720, #721)

**Q: What are free instruction modifiers in MSL?**
A: saturate(), negate (-x), and abs() are zero-cost "free" modifiers encoded in the instruction operand bits — they don't consume additional ALU cycles. The select() function maps to a native FCMPSEL compare-and-select instruction. Use these instead of manual clamp/branch patterns. (Finding #717)

**Q: How does threadgroup memory banking differ across Apple GPU generations?**
A: M1 has 32 independent banks. Stride-2 access patterns cause bank conflicts on M1. M3 uses a fundamentally different banking scheme (details not fully documented). Avoid bank conflicts by padding shared arrays by +1 element per row. All devices share the 32KB threadgroup memory limit. Use setThreadgroupMemoryLength for dynamic allocation at encode time. (Findings #722, #723, #731-#733)

**Q: Why can't I use acquire/release ordering in MSL?**
A: MSL supports only relaxed atomic ordering — no acquire, release, or sequential consistency, even in MSL 4.0. This is unique among major GPU APIs. For ordering guarantees, use threadgroup_barrier(mem_device) within a threadgroup, or MTLSharedEvent/command buffer boundaries across threadgroups. (Finding #724)

**Q: How do I implement float atomics in MSL?**
A: There are no native float atomics in MSL (even in 4.0). Use the standard CAS (compare-and-swap) loop workaround: atomically load the old value, compute the new value, attempt atomic_compare_exchange_weak in a loop until successful. Apple identifies global atomics as the primary performance bottleneck — minimize atomic contention by using hierarchical reduction (SIMD → threadgroup → device). (Findings #725, #726, #727)

**Q: What is simd_ballot and how does it differ from CUDA?**
A: MSL's simd_ballot() returns a simd_vote object (NOT a raw integer like CUDA's __ballot_sync()). The simd_vote type provides .all(), .any(), and .none() member functions. In compute shaders, simd_is_helper_thread() always returns false. At the G13 ISA level, ballot is implemented as icmp_ballot and fcmp_ballot instructions. (Findings #734, #735, #743)

**Q: What is the internal storage layout of simdgroup_matrix?**
A: simdgroup_matrix stores data as vec<T,64> distributed across 32 threads (each thread holds 2 elements for 8x8 matrix). The load/store API uses elements_per_row and transpose_matrix parameters. FP32 simdgroup_matrix throughput matches FP16 (identical RAW throughput) — the register savings of FP16 help occupancy, not raw multiply speed. A14's introduction of simdgroup_matrix showed 37% average improvement. (Findings #744, #741, #745)

**Q: How does M3 Dynamic Caching affect threadgroup memory as software cache?**
A: On M3+ with Dynamic Caching, using threadgroup memory as a software-managed cache may be counterproductive because the hardware already dynamically allocates L1 cache. The manual caching adds threadgroup_barrier overhead without benefit if the hardware cache already covers the access pattern. Profile both approaches. (Finding #729)

**Q: What is the Metal 4 tensor_ops API?**
A: Metal 4 adds `#include <metal_tensor>` with matmul2d() for tensor multiplication. Benchmarks show 2.4x speedup vs simdgroup_matrix on M4 Max. On M5, tensor_ops maps to per-core neural accelerator hardware. This is the successor to simdgroup_matrix for matrix operations. (Findings #753, #754)

## Understanding MSL Address Spaces

The MSL address space model maps directly to Apple GPU memory hierarchy:

1. **thread (private)**: Maps to GPU register file (~208 KB per core). Fastest access but limited. Dynamically indexed arrays in thread space cause register spilling to device memory -- a severe performance cliff. Prefer constant-indexed local variables.

2. **threadgroup (shared)**: On-chip memory shared within a threadgroup (~32 KB). Used for inter-thread communication, tiling, and reduction patterns. Accessed via `threadgroup_barrier()` synchronization.

3. **constant**: Read-only, 64 KB per function argument. Backed by dedicated constant cache hardware. Ideal for kernel parameters, lookup tables, and uniforms.

4. **device**: Main GPU memory (unified with CPU). No size restriction. Read/write. Subject to L1/L2 cache behavior. `coherent(device)` qualifier (Metal 3.2+) enables CPU-GPU coherent access.

5. **threadgroup_imageblock**: Tile memory in Apple's TBDR architecture. Provides access to the on-chip framebuffer during render passes. Can be used for compute-within-render patterns.

## SIMD-Group Programming

Apple GPUs execute 32 threads per SIMD group. MSL provides rich SIMD-group intrinsics:

**Reductions**: `simd_sum`, `simd_product`, `simd_min`, `simd_max`, `simd_and`, `simd_or`, `simd_xor` -- all complete in hardware without threadgroup memory. Available on Apple family 7+ (A14+).

**Prefix Scans**: `simd_prefix_exclusive_sum(T)` and `simd_prefix_inclusive_sum(T)` provide parallel prefix operations across the SIMD group.

**Shuffles**: `simd_shuffle(data, lane)` for arbitrary access, `simd_shuffle_up/down` for shift patterns, `simd_shuffle_xor` for butterfly reductions, `simd_shuffle_rotate_up/down` for circular patterns.

**Divergence Handling**: Apple GPU uses an execution mask stack in register r0l. Divergent branches don't cause warp splits -- instead, inactive threads are masked. All 32 threads remain resident regardless of divergence.

**Matrix Operations**: `simdgroup_matrix<T, 8, 8>` leverages matrix multiply hardware. `simdgroup_multiply_accumulate(D, A, B, C)` for D=A*B+C. Undocumented `simdgroup_async_copy` overlaps compute with memory transfers.

## Metal 4 MSL Extensions

Metal 4 introduces significant MSL language additions:

**Native Tensors**: `#include <metal_tensor>` provides `tensor<T, extents<R, C>>` types. Tensors are first-class objects in MSL 4.0, enabling direct tensor arithmetic in shaders.

**Cooperative Tensors**: Replace `simdgroup_matrix` with partitioned storage across SIMD lanes. Multiple SIMD groups cooperate on larger matrix operations. On M5 hardware, cooperative tensor operations map to neural accelerator units achieving ~128 matrix FMAs per compute partition per cycle (~16x over ALU-only path).

**Shader Logging**: `MTLLogState` (created from `MTLLogStateDescriptor`) enables printf-style debugging from shaders. Controlled via `MTL_SHADER_LOGGING_ENABLED=1` environment variable.

**Unified Encoder Integration**: MSL kernels dispatched from Metal 4's unified compute encoder can interleave with blit and acceleration operations without encoder boundaries.

## Advanced Optimization Techniques

**Function Constants for Branch Elimination**: Declare `constant bool HAS_BIAS [[function_constant(0)]]` and use `if (HAS_BIAS)` in kernel body. Compiler completely eliminates dead branches at PSO creation time. Larian Studios reduced instructions by 84% with this technique.

**Register Pressure Management**: (1) Use 16-bit types to halve register usage, (2) reduce live variable scope, (3) avoid dynamically indexed thread-local arrays. Register spilling is the #1 performance killer on Apple GPUs.

**Kernel Fusion**: Combine multiple operations into single kernels to eliminate intermediate buffer writes. This is the most impactful optimization for ML workloads on Metal. AI-generated fused kernels (e.g., via custom `mx.fast.metal_kernel()`) can outperform separate dispatches significantly.

**Loop Unrolling**: `#pragma unroll` (or `MLX_MTL_PRAGMA_UNROLL`) forces full unrolling. Effect is nuanced -- helps for small known-bound loops but increases register pressure for large loops.

**Hierarchical Reductions**: SIMD-level first (`simd_sum()`), then threadgroup-level via shared memory and barriers, then device-level via atomics. Minimizes expensive atomic operations.

## Cross-Skill Integration

**With metal-compute**: Metal compute pipeline executes MSL kernels. Understand dispatch dimensions (threads/threadgroup/grid) and how they map to kernel `[[thread_position_in_grid]]` attributes. Pipeline state creation compiles AIR to GPU binary.

**With simd-wave**: SIMD-group intrinsics in MSL are the language-level interface to hardware SIMD execution. Wave-level programming patterns (reductions, scans, shuffles) are expressed through MSL SIMD functions.

**With gpu-perf**: Register pressure, occupancy, and memory access patterns in MSL directly determine kernel performance. Profiling with Xcode GPU tools reveals which MSL constructs cause bottlenecks.

**With gpu-silicon**: Apple GPU hardware architecture (ALU latencies, cache hierarchy, TBDR design) determines which MSL patterns perform well. Register file size, SIMD width, and threadgroup memory limits are hardware constraints.

**With metal4-api**: Metal 4 API features (unified encoders, residency sets, command allocators) affect how MSL kernels are dispatched and what resources they can access.

## Investigation Guidelines

When investigating MSL kernel topics:

1. **Check existing findings first**: `${CLAUDE_PLUGIN_ROOT}/scripts/kb search "<topic>"`
2. **Focus on Apple sources**: MSL specification, WWDC sessions, Apple sample code
3. **Cross-reference with production code**: MLX and llama.cpp Metal kernels are the best real-world MSL references
4. **Test on hardware**: MSL behavior varies across GPU families -- test on target Apple GPU family
5. **Note MSL version**: Features differ between MSL 2.x, 3.x, and 4.x

Use `/gpu-forge:investigate msl-kernels "<specific-topic>"` to research new areas and automatically store findings in the knowledge base.

## References

- Layer 1 references: See `${CLAUDE_PLUGIN_ROOT}/skills/msl-kernels/references/` for exported finding lists by topic
- Metal Shading Language Specification (Apple): Address spaces, types, intrinsics
- WWDC Metal sessions: Shader optimization, function constants, Metal 4 MSL
- MLX source: Production MSL kernel patterns (github.com/ml-explore/mlx)
- llama.cpp Metal backend: Real-world compute kernel implementations
- Philip Turner's metal-benchmarks: ALU latencies, register pressure measurements

## Finding ID Reference

Key findings by topic area (investigation #24):
- **Threadgroup barrier**: #708
- **bfloat type**: #710, #711
- **Packed vector types**: #712, #713
- **fast_math & math precision**: #714, #715, #716
- **Free instruction modifiers**: #717
- **Integer type optimization**: #718, #720, #721
- **Common utility functions**: #719
- **Threadgroup memory banking**: #722, #723, #731, #732, #733
- **Atomics limitations**: #724, #725, #726, #727, #730
- **SIMD shuffle design**: #728
- **M3 Dynamic Caching threadgroup**: #729
- **simd_ballot & vote**: #734, #735, #743
- **simdgroup_matrix internals**: #736, #737, #738, #739, #744
- **simd_shuffle_xor butterfly**: #740
- **simdgroup_matrix throughput**: #741, #745
- **simd_shuffle_and_fill**: #742
- **Texture compute access**: #746, #747, #748
- **Compilation options**: #749, #750
- **Visible function models**: #751, #752
- **Metal 4 tensor_ops**: #753, #754

Use `kb detail <id>` to retrieve full finding details including source URLs and confidence levels.

## Version History

- 2026-02-10: Updated with 46 new findings from investigation #24 (95 total)
- 2026-02-09: Initial skill creation with 49 findings from knowledge base
