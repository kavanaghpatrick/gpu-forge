---
name: metal-compute
layer: 1
description: >
  This skill should be used when the conversation involves Metal compute pipeline architecture,
  command buffer lifecycle, compute command encoders, dispatch patterns, pipeline state management,
  argument buffers, resource heaps, indirect command buffers, synchronization primitives, or
  Metal 4 unified encoder features. Trigger keywords: MTLComputePipelineState, MTLCommandQueue,
  MTLComputeCommandEncoder, dispatchThreadgroups, indirect dispatch, hazard tracking, memory barriers,
  command buffer batching, GPU timeout, Metal 4 barriers.
skills:
  - gpu-silicon
  - unified-memory
cross_references:
  - msl-kernels
  - gpu-perf
  - metal4-api
---

# Metal Compute Pipeline

This skill provides expertise in Metal's compute pipeline architecture on Apple Silicon. It covers the full command submission path from CPU-side encoding through GPU execution, focusing on performance-critical aspects like command buffer management, encoder patterns, synchronization, and Metal 4 enhancements.

## Domain Overview

The Metal compute pipeline on Apple Silicon is a sophisticated command submission architecture that bridges CPU and GPU execution. At its core, the pipeline follows a encode-commit-execute model: CPU code uses command encoders to write GPU commands into command buffers, commits those buffers to a command queue, and the GPU asynchronously executes the encoded work. This design enables overlapped CPU-GPU execution and supports both serial and concurrent dispatch patterns.

Command buffer lifecycle is central to Metal compute performance. Each buffer is a transient container for a sequence of GPU commands. Apple's recommended pattern is to submit one command buffer per frame (maximum two) with all compute passes encoded as sequential encoders within that buffer. This minimizes submission overhead while maximizing GPU utilization. Splitting work into many small command buffers introduces measurable overhead — empty kernel round-trips take ~2.5ms minimum due to power cycling and submission latency.

Metal provides multiple synchronization primitives for managing dependencies between compute passes. Fences synchronize across encoder boundaries within a command buffer, memory barriers enforce write-read ordering within a compute encoder, and events coordinate across command buffers. The framework offers automatic hazard tracking (MTLHazardTrackingMode.default) which infers dependencies, or untracked mode for manual control when performance is critical.

Metal 4 introduces significant architectural changes including unified encoders that combine compute, blit, and acceleration operations, producer/consumer barriers for fine-grained synchronization, command allocators that decouple command memory from buffers, and residency sets as the exclusive mechanism for making resources GPU-resident. These changes optimize for modern ML workloads with tensor operations.

## Key Knowledge Areas

The knowledge base contains 40 findings across these Metal compute topics:

- **Command Submission Architecture**: Full encode-commit-execute path, submission overhead (~2.5ms minimum), GPU power cycling behavior
- **Command Buffer Patterns**: Batching strategies (1-2 buffers/frame), multi-pass compute within single buffer, serial vs concurrent queue dispatch
- **Encoder Management**: Command encoder lifecycle (transient single-use), multi-pass sequential encoders, tile-based compute dispatches
- **Pipeline State**: Compute pipeline creation and caching (~1ms uncached), occupancy hints (maxTotalThreadsPerThreadgroup), threadGroupSizeIsMultipleOfThreadExecutionWidth optimization
- **Dispatch Patterns**: Direct threadgroup dispatch, indirect dispatch from GPU buffers, dispatch size validation
- **Indirect Command Buffers**: GPU-encoded compute dispatches, ICB structure and encoding, limitations (separate encoder passes still required)
- **Synchronization**: Three-level sync model (fences, barriers, events), automatic vs manual hazard tracking, coherent(device) qualifier behavior
- **Resource Management**: Argument buffers, resource heaps, dependency tracking, residency management
- **Metal 4 Features**: Unified encoders, producer/consumer barriers, command allocators, residency sets, compiler object separation
- **Error Handling**: GPU timeout errors (IOAF code 2), watchdog behavior (time-based not iteration-based), completion handler patterns
- **MLX Integration**: How MLX creates compute pipelines, mx.fast.metal_kernel() instantiation behavior
- **Multi-Queue Submission**: When multiple MTLCommandQueues help (hiding bubbles, async upload/compute/readback)
- **Language Bindings**: metal-cpp zero overhead, PyObjC bridge costs, Rust metal-rs deprecation
- **Atomic Patterns**: CPU-GPU shared buffer ring buffers, memory ordering guarantees, coherent device buffers

## How to Query Knowledge

Use the portable KB CLI to search Metal compute findings:

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "command buffer"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "dispatch threadgroups"
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill metal-compute
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <finding-id>
```

The `search` command uses BM25-weighted FTS5 ranking, prioritizing claim text over evidence. The `skill` command returns all 40 metal-compute findings in table format.

## Common Patterns & Quick Answers

**Q: What's the recommended command buffer submission pattern for compute workloads?**
A: Submit one command buffer per frame (max two) with all compute passes as sequential encoders within that buffer. Splitting into many small buffers adds overhead. (Finding #137, verified, Apple docs)

**Q: What's the minimum round-trip time for Metal compute on Apple Silicon?**
A: ~2.5ms minimum due to GPU power cycling. Empty kernel encode + commit + waitUntilCompleted never beats this threshold for small workloads. (Finding #135, verified, WWDC session)

**Q: How does Metal handle GPU command buffer timeouts?**
A: IOAF error code 2 indicates GPU Timeout Error. The watchdog is time-based (not iteration-count), and Metal does NOT expose public API to configure or extend the timeout. (Findings #144-146, high confidence)

**Q: Should I use one MTLCommandQueue or multiple queues for compute?**
A: Single queue executes buffers serially. Multiple queues help when: (1) hiding GPU bubbles by keeping queues full, (2) async upload while compute runs, (3) overlapping readback with next dispatch. Most compute workloads do fine with one queue. (Finding #140, verified, WWDC)

**Q: When should I use indirect command buffers (ICBs)?**
A: ICBs enable GPU-encoded compute dispatches, useful when dispatch parameters are GPU-computed. However, ICB-based kernel chaining still requires separate encoder passes (endEncoding + new encoder), limiting perf gains. (Findings #149-150, verified)

**Q: What's the difference between fences, barriers, and events?**
A: Three sync levels: MTLFence synchronizes across encoders within a command buffer, memoryBarrier(scope:) enforces ordering within a compute encoder, MTLEvent coordinates across command buffers or with CPU. (Findings #141-142, verified)

**Q: How expensive is compute pipeline state creation?**
A: ~1ms for a typical kernel when not cached. Metal caches PSOs automatically. Use pipeline state at init time to avoid frame hitches. (Finding #159, verified)

**Q: What does Metal 4 change about command submission?**
A: Metal 4 introduces MTL4ComputeCommandEncoder (unified compute/blit/acceleration), producer/consumer barriers (replacing fences), command allocators (decouple memory from buffers), and residency sets (now the ONLY way to make resources resident). (Findings #166-170, verified)

**Q: Can I run persistent kernels that loop indefinitely on Metal?**
A: No, true persistent kernels are not feasible due to GPU watchdog timeouts. Recommended pattern is bounded kernel chaining with iteration limits and periodic returns to CPU. (Findings #151-152, high/verified)

**Q: How does MLX create Metal compute pipelines?**
A: MLX calls device->newComputePipelineState() for simple kernels. mx.fast.metal_kernel() creates a NEW Metal library per kernel instantiation. (Findings #171-172, verified)

## Understanding Command Buffer Lifecycle

The full Metal command submission path on Apple Silicon:

1. **CPU Encoding**: Create MTLCommandBuffer from MTLCommandQueue, create MTLComputeCommandEncoder from buffer, set pipeline state and resources, encode dispatch commands
2. **Commit**: Call commandBuffer.commit() to enqueue for GPU execution
3. **GPU Execution**: Metal driver submits to GPU firmware, work queued to hardware, threadgroups execute on GPU cores
4. **Completion**: GPU signals completion, completion handlers fire on undefined thread

Critical performance implications:
- Command encoders are "very inexpensive to create" (Apple docs) — transient single-use objects
- Hazard tracking default mode automatically infers dependencies between resources
- Empty kernel dispatch overhead is ~2.5ms minimum even on Apple Silicon
- Multiple compute passes should be sequential encoders in ONE buffer, not many buffers

## Metal 4 Architecture Changes

Metal 4 represents a ground-up redesign focused on ML workloads:

**Unified Encoder**: MTL4ComputeCommandEncoder replaces separate compute/blit encoders, combining all operations in one object.

**Barrier Model**: Producer barriers block subsequent passes until writes complete. Consumer barriers wait for prior writes. Replaces old fence model.

**Command Allocators**: MTL4CommandAllocator decouples command memory allocation from command buffer lifecycle, enabling better memory reuse.

**Residency Sets**: Now the EXCLUSIVE mechanism for making resources resident. setBuffer/setTexture do NOT make resources resident in Metal 4.

**Compiler Separation**: Shader compilation moves from MTLDevice to dedicated MTL4Compiler object, enabling better compilation control.

These changes optimize for tensor-heavy workloads with many small dispatches and frequent resource access.

## Cross-Skill Integration

**With msl-kernels**: Metal compute pipeline executes MSL kernels — understand dispatch dimensions (threads/threadgroup/grid) and how they map to kernel [[thread_position_in_grid]] attributes.

**With gpu-perf**: Command buffer batching, encoder reuse, and sync primitive choice directly impact occupancy and GPU utilization. Performance profiling requires understanding command submission overhead.

**With gpu-silicon**: Apple GPU's TBDR architecture influences how Metal schedules compute work, especially tile-based compute dispatches that run within render encoders.

**With unified-memory**: Shared buffer storage modes affect CPU-GPU synchronization requirements. Atomic ring buffers between CPU-GPU require understanding Metal's memory ordering guarantees.

**With metal4-api**: Metal 4 introduces native tensor types (MTLTensor) and cooperative tensor operations that integrate with the unified encoder model.

## Advanced Topics

**Persistent Kernel Patterns**: While true persistent kernels (infinite loops) hit watchdog timeouts, pseudo-persistent patterns use bounded kernel chaining with iteration limits. Submit compute pass → GPU updates dispatch params → encode next pass with indirect dispatch.

**Atomic Ring Buffers**: MTLStorageModeShared enables direct CPU-GPU access to unified memory. CPU writes, GPU reads via atomic loads. Memory ordering is NOT well-defined in Apple docs — use explicit memory fences when ordering matters.

**Multi-Queue Strategies**: Single queue serializes all work. Multiple queues enable parallelism: upload queue fills buffers while compute queue runs kernels while readback queue copies results. Requires careful fence coordination.

**Language Binding Overhead**: metal-cpp has zero measurable overhead vs Objective-C due to inline functions. PyObjC adds bridge call overhead. Rust's metal-rs is deprecated as of Sep 2024 — use metal-cpp or Objective-C.

## Investigation Guidelines

When investigating Metal compute topics:

1. **Check existing findings first**: `${CLAUDE_PLUGIN_ROOT}/scripts/kb search "<topic>"`
2. **Focus on Apple sources**: WWDC sessions, Apple documentation, metal-cpp samples
3. **Validate with code**: Metal behavior varies across OS versions — test on target macOS/iOS
4. **Cross-reference with gpu-perf**: Performance claims need profiling validation
5. **Note Metal version**: Features differ significantly between Metal 2.x, 3.x, and 4.x

Use `/gpu-forge:investigate metal-compute "<specific-topic>"` to research new areas and automatically store findings in the knowledge base.

## References

- Layer 3 references: See `${CLAUDE_PLUGIN_ROOT}/skills/metal-compute/references/` for exported finding lists by topic
- Apple Metal Programming Guide: Command encoding, pipeline states, synchronization
- WWDC Metal sessions: Pipeline best practices, Metal 4 architecture
- metal-cpp: C++ binding headers and samples
- Philip Turner's metal-benchmarks: Empirical overhead measurements

## Version History

- 2026-02-09: Initial skill creation with 40 findings from knowledge base
