---
name: metal4-api
layer: 3
description: >
  This skill should be used when the conversation involves Metal 4 API architecture,
  MTLTensor, cooperative tensors, unified encoder, MTL4CommandBundle, ML-in-shaders,
  next-gen Metal, residency sets, Metal 4 API, MTL4Compiler, MTL4CommandAllocator,
  MTL4ComputeCommandEncoder, MTL4MachineLearningCommandEncoder, flexible pipeline state,
  argument tables, placement sparse resources, Metal Performance Primitives, MSL 4.0,
  matmul2d, Shader ML, AOT pipeline compilation, Metal 4 barriers, producer/consumer
  barriers, Metal 3/4 coexistence, phased adoption, MTL4CounterHeap, GPU work graphs,
  neural rendering on GPU, in-shader ML inference, or Metal 4 debugging.
skills:
  - metal-compute
  - msl-kernels
cross_references:
  - gpu-perf
  - mlx-compute
  - gpu-silicon
---

# Metal 4 API

This skill provides expertise in Apple's Metal 4 API, a ground-up redesign introduced at WWDC 2025. Metal 4 modernizes the GPU programming model with explicit resource management, unified encoders, native tensor types, and ML-in-shader capabilities. It also covers the academic research on neural rendering and in-shader ML inference that motivates these API designs.

## Domain Overview

Metal 4 represents the most significant evolution of Apple's GPU API since Metal's introduction. Announced at WWDC 2025 and shipping with macOS 26 Tahoe and iOS 26, Metal 4 aligns Apple's command model with Vulkan/DX12 conventions while introducing first-class ML primitives that leverage the M5's in-GPU Neural Accelerators.

The core architectural shift is from implicit to explicit resource and command management. Metal 4 removes automatic hazard tracking for all MTL4 pipelines and encoders, requiring developers to manually synchronize with producer/consumer barriers. Command buffers become long-lived reusable objects rather than fire-and-forget transients. MTL4CommandAllocator decouples command memory from buffer lifecycle, enabling efficient memory reuse. Residency sets become the exclusive mechanism for making resources GPU-resident — setBuffer/setTexture no longer implicitly manage residency.

Metal 4 introduces MTLTensor as a first-class multi-dimensional data container optimized for ML workloads. Cooperative tensors in MSL 4.0 (via `<metal_tensor>`) provide native tensor operations at thread, simdgroup, and threadgroup scopes. Metal Performance Primitives (MPP) offers high-performance shader-embeddable kernels for matmul2d and convolution2d operations. The MTL4MachineLearningCommandEncoder executes neural networks directly on the GPU timeline, while Shader ML enables ML inference within fragment and compute shaders without device memory round-trips.

MSL 4.0 upgrades the shading language to C++17 (from C++14), adding lambda expressions, and moves most new types into the MTL4 namespace. The compiler is separated into a dedicated MTL4Compiler object for better compilation control, with AOT pipeline compilation enabling ahead-of-time shader optimization via a harvest-bake-load workflow.

Metal 3 and Metal 4 coexist via MTLDevice extensions, enabling gradual migration. Apple recommends a phased adoption path: Phase 1 (compilation), Phase 2 (encoding), Phase 3 (resources). The Game Porting Toolkit 3 adds experimental Metal 4 support.

## Key Knowledge Areas

The knowledge base contains 67 findings across these Metal 4 topics:

- **Command Model** (7 findings): MTL4CommandAllocator, reusable command buffers, unified encoder consolidation, explicit barriers, resource lifecycle changes, Vulkan/DX12 alignment, MTL4 type inventory
- **Resource Management** (7 findings): Residency sets (mandatory), placement sparse resources, argument tables (GPU virtual addresses), texture view pools, resource lifecycle (no implicit retention), residency efficiency patterns
- **Tensor & ML** (11 findings): MTLTensor API, cooperative tensors, matmul2d descriptor/data types/tile dimensions/execution scopes, Metal Performance Primitives, ML command encoder, Shader ML
- **Compilation** (5 findings): MTL4Compiler object, AOT pipeline compilation (harvest-bake-load), flexible pipeline state, function descriptors, MSL 4.0 language
- **Rendering** (5 findings): Drawable presentation (4-step explicit pattern), early/late submission, MetalFX frame interpolation, MetalFX denoised upscaler, intersection function buffers
- **Compatibility** (5 findings): Metal 3/4 coexistence, phased adoption, device compatibility (M1-M5), Game Porting Toolkit 3, M5 Neural Accelerator
- **Debugging** (3 findings): ML debugging in Xcode 26, counter heaps, shader debug logging
- **Neural Rendering Research** (24 findings): In-shader ML inference, neural materials/textures/BRDFs, neural appearance models, neural deferred shading, mobile neural rendering, GPU work graphs, differentiable rendering

## How to Query Knowledge

Use the portable KB CLI to search Metal 4 findings:

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "MTLTensor"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "residency sets"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "cooperative tensors matmul"
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill metal4-api
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <finding-id>
```

The `search` command uses BM25-weighted FTS5 ranking, prioritizing claim text over evidence. The `skill` command returns all 67 metal4-api findings in table format.

## Common Patterns & Quick Answers

**Q: What changed about command submission in Metal 4?**
A: Metal 4 makes command buffers long-lived and reusable (previously fire-and-forget). MTL4CommandAllocator manages memory backing separately. Automatic hazard tracking is removed — use explicit producer/consumer barriers. (Findings #45, #50, #52, #64)

**Q: How do residency sets work in Metal 4?**
A: Residency sets are now the ONLY way to make resources GPU-resident. Workflow: create set, add allocations, request residency, commit to command buffer. setBuffer/setTexture no longer manage residency. Separate sets by usage pattern for efficiency. (Findings #46, #83, #116)

**Q: What is MTLTensor and how do I use it?**
A: MTLTensor is a first-class multi-dimensional data container for ML workloads with rank, extents, and data type. In MSL 4.0, `<metal_tensor>` provides four storage modes: tensor_handle (device), tensor_threadgroup, tensor_slice (register tile), tensor_fragment (SIMD-distributed). Works across M1-M5. (Findings #556, #557, #585)

**Q: How does matmul2d work with cooperative tensors?**
A: matmul2d_descriptor is a constexpr template with M, N, K dimensions and transpose flags. Three execution scopes: execution_thread (single thread), simdgroup (SIMD-level), threadgroup (cooperative). Supports mixed-precision (half x int8 -> float). Optimal tiles: 64x32x64 general, 128x64x64 for pipelining. (Findings #558, #559, #560, #561)

**Q: How does Shader ML work?**
A: Shader ML runs ML inference directly within fragment/compute shaders without device memory round-trips. Load model via MTL4MachineLearningCommandEncoder, then call inference functions inline in shader code. M5 Neural Accelerators in each GPU core provide 4x AI compute vs M4. (Findings #563, #562, #565)

**Q: What is the MTL4Compiler and AOT pipeline compilation?**
A: MTL4Compiler is a dedicated shader compilation object (separated from MTLDevice). AOT pipeline workflow: (1) HARVEST — capture pipeline descriptors in a serializer, (2) BAKE — compile offline into MTL4PipelineDataSet, (3) LOAD — instantiate at runtime with zero compilation. Flexible pipelines use sentinel values for deferred specialization. (Findings #566, #567, #568)

**Q: How do Metal 3 and Metal 4 coexist?**
A: Both APIs coexist via MTLDevice extensions. You can mix MTLCommandQueue with MTL4CommandQueue using shared MTLEvent/MTLSharedEvent for synchronization. Apple recommends phased adoption: Phase 1 (compilation), Phase 2 (encoding), Phase 3 (resources). (Findings #570, #571)

**Q: How does Metal 4 handle barriers and fences?**
A: Metal 4 uses explicit barriers: producer barriers block subsequent passes until writes complete, consumer barriers wait for prior writes. Fences are retained alongside barriers via updateFence:afterEncoderStages: and waitForFence:beforeEncoderStages:. This replaces Metal 3's automatic hazard tracking. (Findings #50, #581)

**Q: What are placement sparse resources?**
A: Buffers and textures allocated WITHOUT storage pages. Pages come from placement heaps and are mapped on demand. Enables virtual memory-like GPU resource management and efficient memory sharing between resources. (Findings #47, #117)

**Q: What is the neural rendering connection to Metal 4?**
A: Metal 4's tensor and Shader ML features are designed for the emerging paradigm of neural rendering — replacing hand-crafted graphics algorithms with compact neural networks running inline in shaders. Research shows neural materials, neural BRDFs, and neural radiance caching running entirely in compute/fragment shaders. (Findings #514, #515, #563, #600)

## Metal 4 Architecture Diagram

The Metal 4 command model follows this structure:

1. **MTL4Compiler** — Compile shaders independently from device, with AOT support
2. **MTL4CommandAllocator** — Manage command memory backing
3. **MTL4CommandQueue** — Submit reusable command buffers
4. **MTL4CommandBuffer** — Long-lived, reusable (not transient)
5. **MTL4ComputeCommandEncoder** — Unified compute/blit/acceleration encoder
6. **MTL4MachineLearningCommandEncoder** — Neural network execution on GPU timeline
7. **MTL4RenderCommandEncoder** — Unified render encoder
8. **MTLResidencySet** — Exclusive mechanism for resource residency
9. **MTL4ArgumentTable** — Explicit resource binding via GPU virtual addresses

Key types also include: MTL4PipelineDataSet, MTL4CounterHeap, MTLTextureViewPool, MTL4LibraryFunctionDescriptor. (Finding #582)

## Tensor Operations Pipeline

Metal 4 cooperative tensor workflow for compute kernels:

1. **Declare descriptor**: `matmul2d_descriptor<M, N, K, transpose_left, transpose_right>`
2. **Choose execution scope**: thread (divergent data), simdgroup (general ML), threadgroup (maximum throughput)
3. **Load tensor tiles**: From device memory via tensor_handle or from threadgroup via tensor_threadgroup
4. **Execute operation**: matmul2d accumulates C += A * B with chosen precision
5. **Store results**: Write tensor_slice back to device memory

Mixed-precision support: half x half -> half/float, half x int8 -> half/float, int8 x int8 -> int32, float x float -> float. MPP provides optimized implementations. (Findings #557-561, #564)

## Phased Migration Guide

Apple's recommended migration path from Metal 3 to Metal 4:

**Phase 1 — Compilation**: Adopt MTL4Compiler for shader compilation, flexible pipeline states for faster compilation, AOT harvesting for zero-runtime compilation cost. Lowest risk, immediate performance benefit.

**Phase 2 — Encoding**: Move to unified MTL4ComputeCommandEncoder and MTL4RenderCommandEncoder. Add explicit barriers. Adopt reusable command buffers with MTL4CommandAllocator. Moderate complexity.

**Phase 3 — Resources**: Switch to MTLResidencySet for all resource residency. Adopt MTL4ArgumentTable for explicit binding. Use placement sparse resources for advanced memory management. Highest complexity, largest performance gains. (Finding #571)

## Cross-Skill Integration

**With metal-compute**: Metal 4 builds on Metal 3 compute fundamentals. Understanding command buffer lifecycle, encoder patterns, and synchronization from metal-compute is prerequisite knowledge. Metal 4 unified encoder replaces the separate compute/blit encoder model.

**With msl-kernels**: MSL 4.0 extends the shading language with cooperative tensors, C++17 features, and the `<metal_tensor>` header. Kernel authors need both MSL fundamentals (address spaces, atomics, SIMD) and Metal 4 tensor operations.

**With gpu-perf**: Metal 4's explicit resource management enables fine-grained performance control. Residency set strategies, barrier placement, and AOT compilation directly impact GPU utilization. Counter heaps replace counter sample buffers for profiling.

**With mlx-compute**: MLX's custom Metal kernels can leverage Metal 4 tensor operations for improved matmul performance. The MPP library provides drop-in high-performance primitives.

**With gpu-silicon**: M5's Neural Accelerators in each GPU core are the hardware foundation for Shader ML and cooperative tensor performance. Understanding GPU core architecture informs optimal tile dimensions and execution scopes.

## Investigation Guidelines

When investigating Metal 4 topics:

1. **Check existing findings first**: `${CLAUDE_PLUGIN_ROOT}/scripts/kb search "<topic>"`
2. **Distinguish Metal 3 vs Metal 4**: Many patterns changed fundamentally — explicit vs implicit, reusable vs transient, barriers vs hazard tracking
3. **Note hardware requirements**: Tensor APIs work M1-M5, but Neural Accelerators are M5 only (M1-M4 fall back to ALU execution)
4. **Cross-reference with Vulkan/DX12**: Metal 4 closely mirrors modern explicit APIs — Vulkan concepts often translate directly
5. **Check phased adoption context**: Features belong to Phase 1 (compilation), Phase 2 (encoding), or Phase 3 (resources)

Use `/gpu-forge:investigate metal4-api "<specific-topic>"` to research new areas and automatically store findings in the knowledge base.

## References

- Layer 3 references: See `${CLAUDE_PLUGIN_ROOT}/skills/metal4-api/references/` for exported finding lists by topic
- WWDC 2025: "Discover Metal 4", "Explore GPU programming with Metal 4", "Enhance your Metal apps and games"
- Apple Developer Documentation: Metal 4 framework, MSL 4.0 specification
- Metal Performance Primitives: tensor operations, matmul2d, convolution2d
- Xcode 26 Metal Debugger: ML debugging, dependency viewer, tensor inspector

## Version History

- 2026-02-09: Initial skill creation with 67 findings from knowledge base
