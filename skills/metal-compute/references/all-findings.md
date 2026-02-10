# metal-compute — All Findings (84)

## Finding 690: The AGX firmware communicates timeout events via EventMsg::Timeout (discriminant 4) including counter, event_slot, and unk_8 fields. This is architecturally distinct from EventMsg::Fault (discriminant 0). On timeout, driver checks for accompanying fault info; if present, treated as fault; otherwise pure timeout. Full enum: Fault=0, Flag=1, Unk2=2, Unk3=3, Timeout=4.
**Confidence**: verified
**Source**: Asahi Linux AGX Driver - fw/channels.rs event types
**Evidence**: Asahi fw/channels.rs defines EventMsg enum. The gpu.rs handler for timeout logs "GPU timeout nya~!!!!!" and calls get_fault_info() then recover().
**Tags**: timeout,firmware,AGX,asahi,event-channel,fault,recovery

## Finding 688: Asahi Linux reverse engineering reveals the AGX firmware receives two compute timeout parameters during initialization: cl_context_switch_timeout_ms=40 and cl_kill_timeout_ms=50. These control compute workload context switching and termination at firmware level. These are NOT the same as the macOS-level GPU watchdog that kills entire command buffers (which operates on a longer timescale of seconds).
**Confidence**: verified
**Source**: Asahi Linux AGX Driver - initdata.rs
**Evidence**: Asahi initdata.rs: cl_context_switch_timeout_ms: 40 and cl_kill_timeout_ms: 50. The "cl" prefix likely = "compute layer." A separate cdm_context_store_latency_threshold exists for compute dispatch manager context store timing.
**Tags**: timeout,firmware,AGX,asahi,compute,context-switch,preemption

## Finding 705: For CPU-GPU ring buffer correctness on Apple Silicon, memory ordering is NOT guaranteed by atomics alone (MSL only supports relaxed). The reliable mechanism is Metal command buffer commit/completion boundaries: CPU writes data, commits CB, GPU reads in subsequent CB. For continuous streaming, Apple recommends triple buffering with dispatch_semaphore(3), not atomic-based polling. A GPU kernel polling an atomic flag set by CPU has no guarantee preceding non-atomic writes are visible.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Triple Buffering
**Evidence**: Metal Best Practices Guide documents triple buffering as canonical streaming pattern. gfx-rs found intermittent transfer failures with MTLStorageModeShared coherency. Since MSL atomics are relaxed-only, command buffer boundaries provide the implicit memory barrier.
**Tags**: ring-buffer,CPU-GPU,memory-ordering,triple-buffering,command-buffer,synchronization

## Finding 689: The Asahi Linux DRM GPU scheduler is configured with a 100,000ms (100 second) timeout for GPU jobs. This is the Linux-side timeout for detecting incomplete GPU work, distinct from firmware-level timeouts. This suggests macOS likely has a similar kernel-level timeout — developers report the GPU watchdog killing work "after a few seconds" which aligns with an OS-level timeout in the 5-60 second range.
**Confidence**: verified
**Source**: Asahi Linux AGX Driver - queue/mod.rs
**Evidence**: Asahi queue/mod.rs: sched::Scheduler::new(dev.as_ref(), 3, WQ_SIZE, 0, 100000, ...) where 100000 is timeout in ms. The timed_out handler logs "Job timed out on the DRM scheduler" and returns NoDevice status.
**Tags**: timeout,DRM,scheduler,asahi,100-seconds,job-timeout

## Finding 686: Enhanced Command Buffer Errors (errorOptions = .encoderExecutionStatus) provide per-encoder diagnostics with negligible overhead, suitable for production. Error userInfo contains MTLCommandBufferEncoderInfoErrorKey -> array of MTLCommandBufferEncoderInfo with label and errorState. MTLCommandEncoderErrorState: Unknown(0), Completed(1), Affected(2), Pending(3), Faulted(4). Faulted identifies the causal encoder.
**Confidence**: verified
**Source**: Debug GPU-side errors in Metal - WWDC20
**Evidence**: WWDC20 describes as "low overhead" suitable for shipping in production, unlike Shader Validation. Flutter/Impeller enables this on iOS 14+/macOS 11+ and iterates encoder info in completion handlers for hierarchical diagnostics.
**Tags**: error-handling,encoderExecutionStatus,Enhanced-Command-Buffer-Errors,production

## Finding 144: IOAF error codes: code 2 = GPU Timeout Error (command buffer took too long), code 3 = GPU Hang (possible infinite loop), code 4 = execution aborted due to prior error, code 5 = discarded (victim of GPU error/recovery). Apple deliberately does not publicly document these codes.
**Confidence**: high
**Source**: Metal FAQ - Sealed Abstract
**Evidence**: Metal FAQ (sealedabstract.com) and developer forum reports. Apple treats IOAF codes as driver/kernel-level issues. If you encounter them, file a bug report.
**Tags**: IOAF,error-codes,timeout,GPU-hang

## Finding 681: Asahi Linux reverse engineering reveals the AGX firmware classifies GPU errors as: Timeout (ETIMEDOUT), Fault/MMU violation (EIO), Killed/victim of concurrent error (ECANCELED), NoDevice/GPU crashed (ENODEV). When timeout or fault occurs, firmware attempts recovery. Rarely, firmware itself can lock up during recovery, requiring full system restart.
**Confidence**: verified
**Source**: Asahi Linux DRM AGX driver - workqueue.rs
**Evidence**: Asahi workqueue.rs defines WorkError: Timeout, Fault(with fault info), Killed, ChannelError, NoDevice, Unknown. Error conversion: Timeout->ETIMEDOUT, Fault->EIO, Unknown->ENODATA, Killed->ECANCELED, NoDevice->ENODEV.
**Tags**: GPU-hang,timeout,recovery,Asahi,AGX,firmware,fault

## Finding 145: Metal does NOT expose a public API to configure or extend the GPU command buffer execution timeout. There is no equivalent of CUDA's TdrDelay registry setting. The timeout is enforced at the driver/firmware level (IOKit/IOGPU). The only workaround is splitting long-running work into smaller command buffers.
**Confidence**: high
**Source**: MTLCommandBuffer - Apple Developer Documentation
**Evidence**: Extensive search of Apple documentation and Metal API surface reveals no public method to set timeout duration. The watchdog timeout appears to be time-based (seconds range) rather than iteration-count-based.
**Tags**: timeout,watchdog,limitation,no-api

## Finding 146: The GPU watchdog timeout is time-based, not iteration-count-based. The system revokes GPU access after multiple timeout occurrences. MoltenVK developers confirmed the timeout is in the seconds range (not milliseconds) but exact values are undocumented.
**Confidence**: high
**Source**: Metal error: Execution of command buffer was aborted - MoltenVK
**Evidence**: MoltenVK issue #602: 'the system appears to revoke GPU access after multiple timeout occurrences rather than at a specific time threshold.' Splitting work across multiple sub-grids is the recommended workaround.
**Tags**: timeout,watchdog,MoltenVK,seconds

## Finding 674: For fully GPU-autonomous multi-stage compute DAGs, combine indirect dispatch with ICBs: Stage 1 writes results AND generates dispatch parameters into a buffer. Stage 2 uses dispatchThreadgroups(indirectBuffer:) to read GPU-determined parameters. Stage 1 can also encode compute_commands into an ICB that Stage 2 executes. CPU only encodes initial dispatch; all subsequent stages are GPU-determined, within a single command buffer submission.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Indirect Buffers
**Evidence**: dispatchThreadgroups(indirectBuffer:indirectBufferOffset:threadsPerThreadgroup:) reads MTLDispatchThreadgroupsIndirectArguments (3x uint32 for grid dimensions) from a GPU buffer. Combined with ICBs, enables N-stage chains where each stage's workload is determined by the previous stage.
**Tags**: multi-stage,compute-DAG,indirect-dispatch,GPU-autonomous,indirect-command-buffer

## Finding 667: Metal enables fully GPU-driven multi-stage pipelines where the GPU decides what to compute/render next with zero CPU-GPU synchronization during execution. WWDC19 demo showed a 5-pass pipeline: compute(occluder cull) -> render(occluders) -> compute(process occlusion) -> compute(main cull + LOD + ICB encode) -> render(scene), all without CPU intervention.
**Confidence**: verified
**Source**: Modern Rendering with Metal - WWDC19
**Evidence**: Each compute thread reads from scene argument buffer, performs frustum culling, calculates LOD, retrieves mesh/material data, and encodes a draw command into an ICB slot. Atomic operations on indirect range buffer pack valid commands. WWDC19 Bistro demo: 2.8M polygons, ~8000 draw calls across 4 views, completely GPU-driven.
**Tags**: GPU-driven,indirect-command-buffer,compute-pipeline,multi-stage,autonomous

## Finding 171: MLX creates compute pipelines via device->newComputePipelineState() for simple kernels, with dual-level caching (library-level + kernel-level hash lookup). MLX does NOT set threadGroupSizeIsMultipleOfThreadExecutionWidth or maxTotalThreadsPerThreadgroup on descriptors. For custom Metal kernels, MLX caches based on source+template hash.
**Confidence**: verified
**Source**: MLX Metal backend device.cpp - GitHub
**Evidence**: Direct reading of mlx/backend/metal/device.cpp: get_kernel_ method uses hash_name as cache key. Compile options disable fast math. Metal language version selected dynamically (3.1, 3.2, or 4.0 based on OS).
**Tags**: MLX,pipeline-caching,reference-implementation

## Finding 172: MLX mx.fast.metal_kernel() creates a new Metal library per kernel instantiation — kernels should be instantiated once and reused. grid_sample achieved 8x speedup (55.7ms → 6.7ms forward) and 40x backward (676.4ms → 16.7ms) on M1 Max via custom Metal kernels.
**Confidence**: verified
**Source**: Custom Metal Kernels - MLX Documentation
**Evidence**: MLX docs: 'Every time you make a kernel, a new Metal library is created and possibly JIT compiled.' Cache uses hash of source + template args. MLX_METAL_JIT flag enables JIT-only compilation.
**Tags**: MLX,custom-kernels,JIT,8x-speedup

## Finding 703: MSL atomic functions (atomic_fetch_add_explicit, atomic_store_explicit, atomic_load_explicit) ONLY support memory_order_relaxed — atomicity but NO synchronization or ordering guarantees with other memory operations. This is unlike CUDA, HLSL, or Vulkan/SPIR-V which support acquire/release and sequential consistency. This means CPU-GPU ring buffers using only atomic load/store cannot rely on ordering for data visibility.
**Confidence**: verified
**Source**: Atomics proposal - WebGPU
**Evidence**: WebGPU atomics proposal (gpuweb issue #1360) confirms: "Atomic functions only support memory_order_relaxed." Myles C. Maxfield (Apple) confirmed Metal requires explicit atomic types and only relaxed ordering.
**Tags**: atomics,memory-ordering,relaxed-only,ring-buffer,CPU-GPU,MSL,limitation

## Finding 697: MTLBinaryArchive supports compute pipelines via addComputePipelineFunctions(with:). Workflow: (1) create empty archive via device.makeBinaryArchive(descriptor:), (2) call binaryArchive.addComputePipelineFunctions(with: desc), (3) serialize via binaryArchive.serialize(to: url), (4) reload by setting descriptor.url. When creating pipeline states, set pipelineDescriptor.binaryArchives = [archive] — framework searches linearly and returns immediately if found, skipping MTLCompilerService.
**Confidence**: verified
**Source**: Build GPU binaries with Metal - WWDC20
**Evidence**: WWDC20 session 10615 "Build GPU binaries with Metal" shows complete workflow. Archives store render, compute, and tile render pipeline functions in a single container.
**Tags**: binary-archive,pipeline-caching,compute,MTLBinaryArchive

## Finding 685: MTLCommandBufferError defines 10 codes: None(0), Internal(1), Timeout(2), PageFault(3), AccessRevoked(4, formerly Blacklisted), NotPermitted(7), OutOfMemory(8), InvalidResource(9), Memoryless(10), DeviceRemoved(11), StackOverflow(12). AccessRevoked means process blocked from GPU due to too many prior errors. DeviceRemoved only applies to eGPUs — cannot occur on Apple Silicon integrated GPU. StackOverflow = too many stack frames.
**Confidence**: verified
**Source**: objc2-metal Rust bindings - MTLCommandBuffer.rs
**Evidence**: objc2-metal Rust bindings list all codes with exact values. Codes 5 and 6 are unused/skipped. Note: IOAF error codes and MTLCommandBufferError codes are DIFFERENT error reporting layers.
**Tags**: error-handling,MTLCommandBufferError,error-codes,AccessRevoked,StackOverflow

## Finding 693: MTLEvent (Metal 2.1, macOS 10.14) provides GPU-side synchronization between command buffers without CPU intervention. encodeWaitForEvent pauses a buffer's subsequent passes until event value is reached; encodeSignalEvent updates the value. This enables chaining command buffers where subsequent ones wait for prior completion, all on GPU without CPU round-trips — the key mechanism for reducing re-dispatch overhead in persistent kernel patterns.
**Confidence**: verified
**Source**: encodeWaitForEvent Apple Developer Documentation
**Evidence**: Apple docs: encodeWaitForEvent "pauses the GPU from running the buffer's subsequent passes until the event equals or exceeds a value." MoltenVK used MTLEvent for Vulkan semaphores: "MTLFence could be used within a single queue, but for cross-queue an MTLEvent is required."
**Tags**: MTLEvent,GPU-synchronization,kernel-chaining,dispatch-overhead,no-CPU

## Finding 677: MTLSharedEvent enables cross-process GPU synchronization via MTLSharedEventHandle (conforms to NSSecureCoding). Process A calls makeSharedEventHandle(), sends via XPC, Process B calls newSharedEvent(handle:) to recreate. Both use counter-based signaling with monotonically increasing uint64 values via encodeSignalEvent/encodeWaitForEvent.
**Confidence**: verified
**Source**: MTLSharedEventHandle - Apple Developer Documentation
**Evidence**: MTLDevice has newSharedEvent() and newSharedEventWithHandle:(). MTLSharedEventHandle conforms to NSSecureCoding for XPC. Available since macOS 10.14 / iOS 12. wgpu PR #6610 implemented this for cross-process fence sharing between Firefox Gecko and wgpu on macOS.
**Tags**: synchronization,cross-process,MTLSharedEvent,XPC,NSSecureCoding

## Finding 692: MTLSharedEvent with waitUntilSignaledValue achieves less than 50 microseconds of scheduling/waiting overhead for GPU-CPU synchronization, significantly outperforming waitUntilCompleted. Demonstrated by Anukari audio synthesis: command buffer double-buffering encodes next CB on CPU while previous executes on GPU, with MTLSharedEvent signaling completion.
**Confidence**: high
**Source**: Huge macOS performance improvements - Anukari Devlog
**Evidence**: Anukari developer: "< 50us of scheduling/waiting overhead" using MTLSharedEvent. waitUntilCompleted appeared similar but MTLSharedEvent delivered substantially lower latency. Dynamic kernel parameters written to device memory after encoding but before MTLSharedEvent signals execution.
**Tags**: MTLSharedEvent,dispatch-latency,re-dispatch,synchronization,overhead,microseconds

## Finding 678: MTLSharedEvent uses counter-based (timeline) signaling where signaledValue is a monotonically increasing uint64. GPU work signals to a specific value; other work waits until value equals or exceeds target. CPU-side notification uses MTLSharedEventListener with a dispatch queue and notify(_:atValue:block:). MoltenVK uses MTLSharedEvent to implement Vulkan timeline semaphores (VK_SEMAPHORE_TYPE_TIMELINE).
**Confidence**: verified
**Source**: SharedEvent in metal - Rust crate documentation
**Evidence**: SharedEvent exposes: signaled_value() -> u64, set_signaled_value(u64), notify(listener, value, block). MTLSharedEventListener is initialized with a DispatchQueue providing execution context for notification blocks.
**Tags**: synchronization,MTLSharedEvent,timeline-semaphore,counter-based

## Finding 679: MTLSharedEvent is significantly heavier than MTLEvent. MoltenVK chose device-side MTLEvent over MTLSharedEvent for Vulkan semaphores because MTLSharedEvent's host-signaling capability "could have performance implications that may not be acceptable." MTLEvent is lightweight single-device; MTLSharedEvent adds cross-process, cross-device, and CPU-signaling at higher overhead.
**Confidence**: verified
**Source**: MVKSemaphore: Use MTLEvent for device-side synchronization - MoltenVK PR #591
**Evidence**: MoltenVK PR #591 explicitly rejected MTLSharedEvent for common sync path due to performance concerns. Uses MTLEvent for device-side cross-queue sync, MTLSharedEvent only for Vulkan timeline semaphore export/import (VK_EXT_metal_objects).
**Tags**: synchronization,MTLEvent,MTLSharedEvent,performance,MoltenVK

## Finding 701: Metal 3 introduced offline compilation moving GPU binary generation entirely to build time. Workflow: (1) generate JSON pipelines script describing descriptors, (2) run "metal shaders.metal -N descriptors.mtlp-json -o archive.metallib" at build time, (3) load archive at runtime. Eliminates all runtime shader compilation, making PSO creation a lightweight lookup.
**Confidence**: verified
**Source**: Target and optimize GPU binaries with Metal 3 - WWDC22
**Evidence**: WWDC22 10102: pipelines script can be generated manually or harvested at runtime via addComputePipelineFunctions + serializeToURL, then extracted via "metal-source -flatbuffers=json harvested.metallib -o descriptors.mtlp-json". For Metal libraries use "metal-tt shaders.metallib descriptors.mtlp-json -o archive.metallib".
**Tags**: binary-archive,offline-compilation,Metal-3,build-time,pipeline-caching

## Finding 675: Metal 4 replaces argument buffers with MTL4ArgumentTable for resource binding. For bindless compute, the argument table needs just one buffer binding. Resources bind via gpuAddress (buffers) and gpuResourceID (textures). Residency is managed exclusively through MTL4ResidencySet. Residency sets can attach to a command queue once, automatically applying to all command buffers. Control Ultimate Edition reported significant overhead reductions.
**Confidence**: high
**Source**: WWDC 2025 - Discover Metal 4 - DEV Community
**Evidence**: MTL4ArgumentTable stores binding points with sizes based on bind point requirements. Residency sets use addAllocation/addAllocations + commit(). Background thread updates supported for streaming.
**Tags**: metal-4,argument-table,bindless,residency-sets,compute,WWDC25

## Finding 167: Metal 4 introduces two barrier types: Producer Barriers (block subsequent passes until current stages finish) and Consumer Barriers (block current pass stages until earlier passes finish). Three compute stages: dispatch, blit, acceleration_structure. This replaces implicit barrier-at-encoder-boundary from Metal 3.
**Confidence**: verified
**Source**: Synchronizing passes with producer barriers - Apple Developer
**Evidence**: Apple docs: 'Producer barriers block GPU stages in subsequent passes from running until stages in a pass, and earlier passes, finish.' WWDC 2025 example: dispatch-to-fragment barrier between compute texture write and render pass read.
**Tags**: Metal-4,producer-barrier,consumer-barrier,stages

## Finding 168: MTL4CommandAllocator decouples command memory from command buffers. Command buffers become long-lived reusable objects; allocators manage backing memory. Pattern: pool of allocators at init, reset() when GPU completes, beginCommandBuffer(allocator:) to encode. This mirrors Vulkan's VkCommandPool and eliminates dynamic allocation during encoding.
**Confidence**: verified
**Source**: Discover Metal 4 - WWDC25
**Evidence**: Metal by Example and WWDC 2025 session 205. Key workflow: retrieve allocator → reset() → beginCommandBuffer(allocator:) → encode → commit → wait for GPU before reusing allocator.
**Tags**: Metal-4,command-allocator,long-lived,VkCommandPool

## Finding 170: Metal 4 moves shader compilation from MTLDevice to a dedicated MTL4Compiler object with QoS-based thread prioritization. Flexible pipeline states allow reusing compiled Metal IR across pipelines with different configurations, reducing total compilation time. Argument tables (MTL4ArgumentTable) replace per-resource setBuffer/setTexture calls with bindless gpuResourceID/gpuAddress binding.
**Confidence**: verified
**Source**: Discover Metal 4 - WWDC25
**Evidence**: WWDC 2025 session 205: MTL4Compiler inherits QoS from requesting thread. Flexible pipelines reuse common IR. Argument tables use a single buffer per object. This aligns Metal 4 with modern GPU API patterns (Vulkan descriptor sets).
**Tags**: Metal-4,MTL4Compiler,argument-tables,bindless

## Finding 166: Metal 4 MTL4ComputeCommandEncoder unifies compute dispatch, blit, and acceleration structure building in a single encoder. Without additional synchronization, these commands run concurrently by default. This replaces separate MTLBlitCommandEncoder and MTLAccelerationStructureCommandEncoder for compute workloads.
**Confidence**: verified
**Source**: Discover Metal 4 - WWDC25
**Evidence**: WWDC 2025 session 205: 'All compute operations, kernel dispatches, blits, and acceleration structure builds can now be encoded in a single compute encoder, and without any additional synchronization, these commands run concurrently.'
**Tags**: Metal-4,unified-encoder,concurrent-default

## Finding 169: In Metal 4, residency sets are the ONLY way to make resources resident on GPU. Set up once at startup, add to command queue, all command buffers auto-include resources. llama.cpp PR #11427 measured ~250ms faster inference on M2 Ultra after implementing residency sets — the OS was reclaiming GPU memory after 1 second of inactivity.
**Confidence**: verified
**Source**: metal: use residency sets - llama.cpp GitHub
**Evidence**: Apple docs and WWDC 2025 confirm mandatory residency sets. llama.cpp implementation: MTLResidencySet created with initial capacity, addAllocation() stages resources, commit() makes resident, addResidencySet() on queue for auto-inclusion.
**Tags**: Metal-4,residency-sets,llama-cpp,250ms

## Finding 682: Metal Shader Validation (MTL_SHADER_VALIDATION=1) instruments GPU shaders to detect: (1) out-of-bounds device/constant memory access, (2) out-of-bounds threadgroup memory access, (3) null texture usage. Invalid operations are PREVENTED (not just logged). Diagnostics include encoder label, function name, file URL, line/column, GPU stack backtrace. Cannot detect: timeouts/infinite loops, invalid residency, or other UB.
**Confidence**: verified
**Source**: Debug GPU-side errors in Metal - WWDC20
**Evidence**: WWDC20 "Debug GPU-side errors in Metal" (10616). Invalid reads are zero-filled by default (MTL_SHADER_VALIDATION_FAIL_MODE=zerofill). System log: log stream --predicate "subsystem = 'com.apple.Metal' and category = 'GPUDebug'".
**Tags**: validation,shader-validation,debugging,out-of-bounds,null-texture

## Finding 691: The Metal shader compiler treats infinite loops as undefined behavior and can optimize them to no-ops. MSL is based on C++14 which requires forward progress. An empty while(true){} may compile to nothing — kernel appears to complete instantly. This is distinct from loops with high but finite iteration counts, which DO execute and can trigger the GPU watchdog. The wgpu project added a volatile bool workaround which caused 200 FPS to 75 FPS regression on M1 Pro.
**Confidence**: verified
**Source**: [Metal/MSL] Workaround for infinite loops optimized out causes performance regression - wgpu
**Evidence**: wgpu issue #6518: Metal compiler optimizes away infinite loops. Workaround volatile bool caused severe perf regression. wgpu issue #6546 proposed iteration limits (2^64-1) instead, reasoning GPU cannot execute enough cycles to hit it in a human lifetime.
**Tags**: compiler,infinite-loop,undefined-behavior,MSL,optimization,wgpu

## Finding 694: The Metal shader compiler aggressively unrolls loops accessing stack/threadgroup memory to eliminate dynamic indexing. For high iteration counts, this causes extremely long compilation or failure — a compile-time iteration ceiling separate from the runtime watchdog. Control with #pragma clang loop unroll(full) or -fno-unroll-loops. Metal 3 offline binary archive compilation is recommended for complex kernels to avoid runtime compilation timeout.
**Confidence**: high
**Source**: Target and optimize GPU binaries with Metal 3 - WWDC22
**Evidence**: Apple forums: "the compiler aggressively unrolls any loop accessing the stack." WWDC sessions: "Offline compilation is most likely to benefit large programs with deep call paths and loops, where inlining and unrolling are common."
**Tags**: compiler,loop-unrolling,compilation-time,binary-archive,offline-compilation

## Finding 704: Metal is the only major GPU API lacking device-scoped memory/execution barriers. HLSL has DeviceMemoryBarrier() since D3D11 (2009); Vulkan/SPIR-V supports them natively. Metal threadgroup_barrier with mem_device provides threadgroup-scoped ordering only, preventing inter-threadgroup patterns like single-pass prefix sum (decoupled look-back) that match memcpy throughput on other APIs. MSL 3.2 partially addresses this with coherent(device) + atomic_thread_fence with thread_scope_device.
**Confidence**: high
**Source**: A note on Metal shader converter - Raph Levien
**Evidence**: Raph Levien (raphlinus) June 2023: "Metal is the only one that does not support it." Blocks efficient single-pass prefix sum (SAM technique), lock-free data structures, multi-threadgroup coordination. vello/piet-gpu abandoned single-pass prefix sum on Metal. Tree-reduction alternatives achieve only 2/3 of memcpy throughput.
**Tags**: device-scope-barrier,memory-ordering,limitation,cross-threadgroup,prefix-sum

## Finding 683: Metal provides 18+ environment variables for fine-grained validation. Key: MTL_DEBUG_LAYER (API validation), MTL_SHADER_VALIDATION (GPU validation), MTL_SHADER_VALIDATION_FAIL_MODE (zerofill/allow), MTL_SHADER_VALIDATION_RESOURCE_USAGE (missing useResource/useHeap), MTL_SHADER_VALIDATION_STACK_OVERFLOW (recursive calls). Per-pipeline control via ENABLE_PIPELINES/DISABLE_PIPELINES using pipeline labels. Visual debugging: VALIDATE_LOAD_ACTIONS=1 replaces DontCare with fuchsia; VALIDATE_STORE_ACTIONS=1 writes checkerboard pattern.
**Confidence**: verified
**Source**: METALVALIDATION(1) man page
**Evidence**: MetalValidation(1) man page documents all variables. MTL_DEBUG_LAYER_ERROR_MODE defaults to assert, can be nslog or ignore. MTL_SHADER_VALIDATION_DUMP_PIPELINES=1 logs pipeline UIDs for selective validation. MTL_SHADER_VALIDATION_REPORT_TO_STDERR redirects to stderr.
**Tags**: validation,environment-variables,debugging,per-pipeline,visual-debugging

## Finding 666: Residency management is mandatory for all indirectly-accessed resources in argument buffers. Accessing a non-resident resource causes GPU restarts and command buffer failures. For compute encoders, use computeEncoder.useResource(resource, usage:). The useHeap() API reduces overhead to a single call for all resources on a heap.
**Confidence**: verified
**Source**: Explore bindless rendering in Metal - WWDC21
**Evidence**: Metal 3 added enhanced shader validation detecting missing resource residency during execution, reporting shader function name, file/line, resource label, size, and residency status. Batch residency via [encoder useHeap:heap] replaces individual useResource calls.
**Tags**: residency,argument-buffers,useResource,useHeap,compute,validation

## Finding 664: Metal 3 (WWDC22) eliminated MTLArgumentEncoder for Tier 2 argument buffers. Applications directly write GPU addresses (uint64_t via gpuAddress for buffers, MTLResourceID via gpuResourceID for textures) into C structs. A compute kernel receives one argument buffer pointer and navigates the entire resource graph via pointer chasing, eliminating per-dispatch binding overhead entirely.
**Confidence**: verified
**Source**: Go bindless with Metal 3 - WWDC22
**Evidence**: Host writes buffer.gpuAddress + offset as uint64_t; shader reads as constant T*. Shared header uses #if __METAL_VERSION__ to define CONSTANT_PTR(x) as either constant x* (shader) or uint64_t (CPU). For bindless compute, the argument table needs just one buffer binding: constant Scene& scene [[buffer(0)]] with scene.meshes[geometry_id].normals[0].
**Tags**: argument-buffers,bindless,metal-3,compute,gpuAddress,tier-2

## Finding 665: Argument buffer Tier 2 is required for bindless patterns and is available on Apple6 GPU family (A13+) and Mac2 GPU family. All Apple Silicon Macs support Tier 2. Tier 1 is limited to a single argument buffer; Tier 2 supports unbounded arrays with dynamic GPU-side indexing — up to 500,000 separate buffers or textures per draw/dispatch.
**Confidence**: verified
**Source**: Explore bindless rendering in Metal - WWDC21
**Evidence**: Tier 1 limits: 64 buffers, 128 textures, 16 samplers per stage. Tier 2 virtually removes slot limits. Key differentiator: Tier 2 can use arrays of argument buffers indexed dynamically in shaders; Tier 1 cannot.
**Tags**: argument-buffers,tier-1,tier-2,bindless,resource-limits,compute

## Finding 153: On Apple Silicon unified memory, MTLStorageModeShared buffers are directly accessible by both CPU and GPU at the same physical address. A ring buffer using atomic read/write indices enables CPU-GPU producer-consumer communication without copies. Only 32-bit integer atomics are natively supported on all Apple GPUs; FP32 atomics are emulated; 64-bit atomics require M2+.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Triple Buffering
**Evidence**: Metal Best Practices: 'For small-sized data that changes frequently, choose Shared mode.' Philip Turner: 'Apple hardware lacks native FP32 atomics (metal::atomic<float> is emulated).' M2 series added native 64-bit atomic hardware.
**Tags**: ring-buffer,atomic,unified-memory,shared-mode

## Finding 154: Metal CPU-GPU memory ordering for shared buffers is NOT well-defined in Apple documentation. The gfx-rs project found intermittent transfer failures with MTLStorageModeShared coherency. A CPU-GPU atomic ring buffer may need explicit synchronization beyond just atomics.
**Confidence**: high
**Source**: Coherent memory on Metal - gfx-rs GitHub
**Evidence**: GitHub gfx-rs/gfx#2069: 'The coherency guarantees of MTLStorageModeShared for buffers are not clearly specified anywhere in Metal documentation, and intermittent transfer failures occur.' They moved away from relying on Shared mode coherency.
**Tags**: coherency,shared-memory,undefined-behavior,limitation

## Finding 698: Apple benchmarked MTLBinaryArchive on Fortnite (11,000+ PSOs, 1,700 in archive) on 6-core 3GHz Mac mini 32GB: first-compile was 1 minute 26 seconds; with binary archive, 3 seconds — a 28x speedup. The 1,700 PSOs represent the subset compiled during a test session out of 11,000+ total shader variants.
**Confidence**: verified
**Source**: Build GPU binaries with Metal - WWDC20
**Evidence**: WWDC20 session 10615 case study. Fortnite compiled needed PSOs at load time to minimize hitching.
**Tags**: binary-archive,benchmark,pipeline-creation-time,Fortnite,performance

## Finding 699: In Metal 2.3 (WWDC20), binary archives required SAME GPU AND SAME OS BUILD; any mismatch caused fallback to runtime compilation. In Metal 3 (WWDC22), Apple introduced offline compilation and forward compatibility: Metal now "gracefully upgrades" archives during OS updates or at app install time, asynchronously in the background, ensuring forward compatibility with future OS versions.
**Confidence**: verified
**Source**: Target and optimize GPU binaries with Metal 3 - WWDC22
**Evidence**: WWDC20: "only requirement is same GPU and same OS build." WWDC22: "Metal gracefully upgrades your binary archives during OS updates or at app install time, asynchronously in the background." Major evolution in archive durability.
**Tags**: binary-archive,invalidation,compatibility,OS-update,Metal-3,offline-compilation

## Finding 707: MTLBinaryArchive serialization/deserialization requires file-based URLs only — data URLs and in-memory approaches fail with MTLBinaryArchiveErrorInvalidFile. Apple engineers confirmed: "Currently this has to be a local file. Annoying but inexpensive to bounce through a temporary file." MoltenVK currently caches MSL source code only (not compiled binaries), with ~15ms per pipeline variant compilation on Metal.
**Confidence**: verified
**Source**: Explore use of Metal Binary Archives for Vulkan pipeline caching - MoltenVK
**Evidence**: MoltenVK issue #1765 tested NSURLProtocol custom schemes and data URLs, both failing. MoltenVK discussion #1789 reports ~15ms per pipeline variant.
**Tags**: binary-archive,MoltenVK,file-constraint,pipeline-caching

## Finding 706: Binary archives are memory-mapped when loaded, reserving virtual address space. Pipelines from archives do NOT count against active app memory (unlike Metal Shader Cache). Archives can be released after pipeline states are created. Best practice: divide archives by usage pattern and release unused ones. Binary archives provide explicit lifecycle control that the automatic Metal Shader Cache does not.
**Confidence**: verified
**Source**: Build GPU binaries with Metal - WWDC20
**Evidence**: WWDC20 10615: "Your binary archive file is memory mapped when loaded, meaning a virtual memory range is reserved...and released when the archive is released." Distinct from automatic shader cache, provides explicit control to collect, organize, and release compiled pipelines.
**Tags**: binary-archive,memory-management,memory-mapped,virtual-memory,lifecycle

## Finding 700: MTLPipelineOption.failOnBinaryArchiveMiss causes pipeline state creation to return nil instead of falling back to runtime compilation when pipeline not found in archives. Enables deterministic behavior: apps detect missing pipelines explicitly rather than incurring unpredictable compilation pauses. Useful for validating all needed pipelines are pre-cached.
**Confidence**: verified
**Source**: Build GPU binaries with Metal - WWDC20
**Evidence**: WWDC20 10615 and WWDC21 10229 both document this. If binaryArchives set and pipeline found: returns immediately. Not found + failOnBinaryArchiveMiss: returns nil. Not found without flag: falls back to MTLCompilerService runtime compilation.
**Tags**: binary-archive,pipeline-caching,failOnBinaryArchiveMiss,deterministic

## Finding 702: Flutter's binary archive prototype showed disappointing gains. The bottleneck was shader compilation (SKSL to MSL to MTLLibrary translation), NOT PSO creation from pre-compiled libraries. Caching binary archives did not measurably reduce GrMtlPipelineStateBuilder::finalize time. Critical insight: binary archives only help when GPU binary compilation is the bottleneck, not when shader source translation dominates.
**Confidence**: verified
**Source**: Flutter Metal Binary Archive Prototype PR #23914
**Evidence**: Flutter engine PR #23914 by chinmaygarde. Archives serialized as flutter_engine_<version>_<skia_version>_<index>.metallib. Testing showed primary jank source was shader library construction.
**Tags**: binary-archive,flutter,practical-challenges,shader-compilation

## Finding 155: Metal 3.2 introduced coherent(device) qualifier enabling cross-threadgroup visibility when combined with atomic_thread_fence(memory_order_seq_cst, thread_scope_device). Before 3.2, Metal had NO device-scope atomic barriers — threadgroup_barrier(mem_flags::mem_device) actually operated at threadgroup scope only.
**Confidence**: high
**Source**: A note on Metal shader converter - Raph Levien
**Evidence**: Raph Levien: 'threadgroup_barrier(mem_flags::mem_device) actually operates at threadgroup scope, not device scope... there is simply no way to run decoupled look-back on Metal.' SPIRV-Cross issue #2473 confirms coherent(device) in MSL Spec v3.2.
**Tags**: coherent-device,Metal-3.2,device-scope,decoupled-lookback

## Finding 156: coherent(device) does NOT eliminate explicit fences. It enables cross-threadgroup visibility but requires explicit atomic_thread_fence(memory_order_seq_cst, thread_scope_device). The qualifier is necessary but not sufficient — fences are still required for correctness.
**Confidence**: high
**Source**: Support for globally coherent buffers - MoltenVK GitHub
**Evidence**: SPIRV-Cross #2473: 'Beyond the qualifier, proper synchronization requires explicit atomic_thread_fence calls with thread_scope_device and sequential consistency ordering.' MoltenVK #2497 confirms the same.
**Tags**: coherent-device,atomic-thread-fence,not-automatic

## Finding 136: Splitting 50 compute passes into 50 separate command buffers vs keeping them in 1 command buffer causes a 641% GPU-side regression on M1 and 423% on A15. The Metal driver coalesces sequential compute encoders within the same command buffer into a single GPU execution unit; this optimization does NOT occur across command buffer boundaries.
**Confidence**: verified
**Source**: Streaming implementations and indirect draws/dispatches - WebGPU
**Evidence**: WebGPU benchmark (gpuweb/gpuweb#2189) measured: M1 1 buffer=3.7ms vs 50 buffers=23.7ms (641%). A15 1 buffer=1.3ms vs 50 buffers=5.5ms (423%). wgpu profiling by kvark confirmed driver coalesces encoders from same buffer on GPU side; creating a new encoder per command buffer is 'heavy' on CPU side.
**Tags**: batching,regression,coalescing,M1,A15

## Finding 137: Apple's best practice: submit one command buffer per frame (max two). Each command buffer should contain multiple encoders. Batch as many dispatches as possible into a single compute encoder. Only split into multiple command buffers when CPU needs GPU results before encoding more work, or for triple-buffering.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Command Buffers
**Evidence**: Metal Best Practices Guide: 'Submit the fewest possible command buffers per frame without underutilizing the GPU. Preferred: submit one command buffer per frame. Acceptable: one to two maximum.' Tech Talk: 'Batch more encoders together into each command buffer before making that call to commit.'
**Tags**: best-practice,batching,command-buffer

## Finding 687: MTLCommandBuffer tracks 6 status values: NotEnqueued(0), Enqueued(1), Committed(2), Scheduled(3), Completed(4), Error(5). When Error, the error property has NSError in MTLCommandBufferError domain. Best practice: (1) always use addCompletedHandler to check status/error, (2) enable .encoderExecutionStatus for production, (3) use commandBuffer.logs for shader diagnostics in dev. The ONLY documented recovery from fatal GPU error is to destroy and recreate the MTLCommandQueue.
**Confidence**: verified
**Source**: Metal Framework MTLCommandBuffer.h header
**Evidence**: MTLCommandBuffer.h defines status progression: NotEnqueued->Enqueued->Committed->Scheduled->Completed|Error. Apple Developer Forums: "The only way to recover is mark a failure state, then destroy and recreate MTLCommandQueue next frame."
**Tags**: error-handling,MTLCommandBufferStatus,lifecycle,recovery,best-practices

## Finding 133: The full Metal command submission path on Apple Silicon: CPU encodes commands via MTLCommandEncoder → commit() on MTLCommandBuffer → Metal driver populates shared memory work items → driver sends doorbell (0x83000000000002 for compute) to ASC coprocessor via RTKit mailbox → ASC firmware reads work queue ring buffer and interprets microsequence → GPU hardware executes compute shader → stamp objects increment, event IDs asserted back to CPU.
**Confidence**: verified
**Source**: Asahi Linux GPU Driver - GitHub
**Evidence**: Asahi Linux driver source (drivers/gpu/drm/asahi/) reveals: channel.rs handles firmware communication, queue/compute.rs builds compute jobs with microsequences (StartCompute→Timestamp→WaitForIdle→Timestamp→FinalizeCompute→RetireStamp), workqueue.rs manages DRM scheduler integration. The ASC coprocessor is an ARM64 processor running Apple's RTKit firmware.
**Tags**: command-submission,ASC,firmware,microsequence,Asahi

## Finding 134: Empty Metal compute kernel round-trip (encode + commit + waitUntilCompleted) takes approximately 120 microseconds on M2 Max. Standard firmware scheduling latency is ~300 microseconds. Philip Turner measured a record-low CPU-GPU atomic bypass latency of ~4 microseconds but found it extremely unreliable (often jumping to ~2000 microseconds).
**Confidence**: high
**Source**: metal-benchmarks - Philip Turner GitHub
**Evidence**: Apple Developer Forums reports ~120us empty kernel round-trip. Philip Turner's metal-benchmarks measured 4us best-case via CPU-GPU atomics but described it as 'very finicky.' OpenMM Metal plugin showed ~6us dispatch overhead in Perfetto traces.
**Tags**: latency,overhead,microseconds,M2-Max

## Finding 135: For small workloads, Metal compute never beats ~2.5ms due to GPU power cycling. If submitting many short jobs with small breaks, the GPU goes to sleep and takes a long time to wake, causing 2-4x performance loss even on large ML workloads.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Apple Tech Talks
**Evidence**: Apple Developer Forums thread 'Metal Compute Never beats 2.5ms' and Metal Compute Tech Talk: 'If you are submitting a lot of short jobs with small breaks in between, the GPU will go to sleep and take a very long time to come back.'
**Tags**: power-cycling,latency,overhead,small-workloads

## Finding 143: addCompletedHandler runs on an undefined thread and should complete quickly — expensive work must be deferred. waitUntilCompleted blocks the calling thread. For pipelined workloads, use addCompletedHandler with semaphores for triple-buffering rather than blocking waits.
**Confidence**: verified
**Source**: Command Organization and Execution Model - Metal Programming Guide
**Evidence**: Metal Programming Guide: 'Scheduled and completed handlers are invoked in execution order on an undefined thread. Any code you execute in these handlers should complete quickly; if expensive or blocking work needs to be done, defer that work to another thread.'
**Tags**: completion-handler,waitUntilCompleted,triple-buffering

## Finding 165: Command encoders are transient single-use objects that are 'very inexpensive to allocate and deallocate.' They cannot be reused after endEncoding. The Metal driver internally coalesces sequential compute encoders within the same command buffer, making multiple encoder creation within one buffer nearly free on GPU side.
**Confidence**: verified
**Source**: Command Organization and Execution Model - Metal Programming Guide
**Evidence**: Metal Programming Guide: 'Command buffer and command encoder objects are transient and designed for a single use. They are very inexpensive to allocate and deallocate.' Confirmed by wgpu profiling (kvark): driver coalesces encoders from same buffer.
**Tags**: encoder,overhead,transient,coalescing

## Finding 668: Metal 3 extended indirect command buffer support to include compute dispatches (not just render commands). A compute kernel can encode compute_command dispatches into an ICB, which another compute pass executes. This enables GPU-driven compute-to-compute chains where one kernel dynamically generates workload for the next, all without CPU involvement.
**Confidence**: verified
**Source**: Modern Rendering with Metal - WWDC19
**Evidence**: WWDC content states: Metal 3 now supports encoding compute dispatches, allowing you to build your compute dispatches on the GPU too. Compute ICBs can be built once and reused, saving CPU cycles. Example: culling kernel encodes per-patch tessellation factor compute dispatches into ICB.
**Tags**: GPU-driven,compute-dispatch,indirect-command-buffer,metal-3,compute-to-compute

## Finding 671: The documented ICB maximum is 16,384 commands, but Apple's own sample code creates ICBs with 65,536 draw calls running without validation errors. The practical limit is governed by maxBufferLength and per-command memory footprint, not a hard 16,384 cap. For exceeding limits, execute multiple ICBs via separate executeCommandsInBuffer() calls.
**Confidence**: verified
**Source**: MultiDrawIndirect and Metal - Tellusim Technologies
**Evidence**: Tellusim reported the 16,384 limit. WebGPU issue #2612 analyzed: maxBufferLength >= 40 + maxCommandCount * ((2 + 2 * maxVertexBufferBindCount + maxFragmentBufferBindCount) * 8). Apple sample "Encoding Indirect Command Buffers on the GPU" uses 65,536 commands.
**Tags**: indirect-command-buffer,maxCommandCount,16384,limits,ICB

## Finding 672: On Apple Silicon (M1), ICBs perform better for many small dispatches but worse for fewer large ones. Small draws (4/2 primitives): ICB 48M ops vs CPU loop 20M (2.4x faster). Large draws (>200 primitives): ICB 1.05B vs loop 1.49B (1.4x slower). ICBs gave 39% overall improvement in GravityMark benchmark. The crossover means ICBs are most beneficial for GPU-driven culling patterns with many small work items.
**Confidence**: verified
**Source**: MultiDrawIndirect and Metal - Tellusim Technologies
**Evidence**: Tellusim GravityMark benchmark on M1. For comparison, AMD Radeon Vega 56 was 18% slower with ICBs but gained CPU availability.
**Tags**: indirect-command-buffer,performance,M1,Apple-Silicon,benchmark

## Finding 673: ICBs support two reuse patterns: (1) encode-once-execute-many — commands built once, executed repeatedly across frames without re-encoding (saves CPU+GPU), and (2) per-frame re-encoding — resetWithRange() called before compute kernels re-populate each frame (needed for dynamic workloads like culling). The atomic counter in the indirect range buffer is also reset each frame for pattern 2.
**Confidence**: verified
**Source**: MTLIndirectCommandBuffer - Apple Developer Documentation
**Evidence**: resetWithRange() available since iOS 12 / macOS 10.14. ICB is a persistent GPU resource created via device.makeIndirectCommandBuffer(descriptor:maxCommandCount:options:).
**Tags**: indirect-command-buffer,resetWithRange,reuse,patterns,compute

## Finding 669: Using ICBs for GPU-side encoding eliminates CPU-GPU synchronization entirely. WWDC22 recommends: "Using indirect command buffer, you can move encoding of the next batch directly on the GPU, avoiding any need for synchronization." Concurrent dispatch (MTLDispatchTypeConcurrent) processing 3 images in parallel showed 70% faster performance vs sequential.
**Confidence**: verified
**Source**: Scale compute workloads across Apple GPUs - WWDC22
**Evidence**: Solutions hierarchy for reducing sync overhead: (1) use MTLSharedEvents instead of CPU waits, (2) pipeline by encoding multiple batches ahead, (3) use ICBs for GPU-side encoding, (4) use concurrent dispatches to interleave independent work.
**Tags**: GPU-driven,synchronization,indirect-command-buffer,concurrent-dispatch,pipelining

## Finding 149: Metal Indirect Command Buffers (ICBs) enable GPU-encoded compute dispatches. A compute kernel fills an ICB, then executeCommandsInBuffer runs the commands. Range indirection via executeCommands(in:indirectBuffer:indirectBufferOffset:) allows GPU-controlled selection of which commands and how many to execute. Limit: 16,384 commands per ICB.
**Confidence**: verified
**Source**: MTLIndirectCommandBuffer - Apple Developer Documentation
**Evidence**: Metal 3 (2019) added executeCommands(in:indirectBuffer:) on compute encoders. M1 showed 39% improvement over CPU looping with ICBs. ICBs are reusable across frames.
**Tags**: ICB,indirect-command-buffer,GPU-driven,16384-limit

## Finding 150: ICB-based kernel chaining still requires separate encoder passes (endEncoding + new encoder). The GPU cannot autonomously create new passes — each must be pre-encoded by CPU. The GPU controls WHAT executes within each pass but not the pass structure itself.
**Confidence**: verified
**Source**: Command Organization and Execution Model - Metal Programming Guide
**Evidence**: Metal execution model: only one encoder active per command buffer at a time, endEncoding() must be called before creating a new encoder. ICB pattern: Pass 1 (compute) writes ICB → Pass 2 (compute) executes ICB. Both pre-encoded by CPU.
**Tags**: ICB,limitation,encoder,execution-model

## Finding 670: MTLIndirectComputeCommand supports: setComputePipelineState(), setKernelBuffer(offset:at:), concurrentDispatchThreadgroups(threadsPerThreadgroup:), concurrentDispatchThreads(threadsPerThreadgroup:), setThreadgroupMemoryLength(), setBarrier(), clearBarrier(), and reset(). The ICB descriptor uses maxKernelBufferBindCount for max kernel buffers per command.
**Confidence**: verified
**Source**: MTLIndirectComputeCommand - Apple Developer Documentation
**Evidence**: MTLIndirectCommandBufferDescriptor properties for compute: commandTypes (.concurrentDispatchThreads or .concurrentDispatchThreadgroups), inheritBuffers, inheritPipelineState, maxKernelBufferBindCount. When inheritPipelineState=false, each command must call setComputePipelineState() individually.
**Tags**: indirect-command-buffer,compute,MTLIndirectComputeCommand,API

## Finding 148: dispatchThreadgroups(indirectBuffer:indirectBufferOffset:threadgroupsPerGrid:) reads the number of threadgroups from a GPU-accessible buffer, enabling GPU-generated grid sizes. The threadgroup size is still CPU-specified. Available since iOS 12/macOS 10.14.
**Confidence**: verified
**Source**: MTLComputeCommandEncoder - Apple Developer Documentation
**Evidence**: Apple Metal documentation. The indirect buffer contains three uint32 values (threadgroupsPerGrid x, y, z). A compute kernel can write these values and the next dispatch reads them.
**Tags**: indirect-dispatch,GPU-driven,grid-size

## Finding 696: The ~25-30M iteration ceiling observed on M4 is best explained by the GPU watchdog's wall-clock time limit intersecting with per-iteration execution time, NOT a hardcoded iteration counter. Evidence: (1) firmware timeout is time-based per Asahi, (2) Metal compiler does NOT statically analyze loop counts for termination, (3) developers report "a few seconds" not fixed iteration count, (4) complex loop bodies hit the ceiling with fewer iterations. Three-tier timeout architecture: firmware (40-50ms per preemption), OS-level (5-60s), DRM scheduler (100s).
**Confidence**: medium
**Source**: Synthesis: Asahi GPU driver, WWDC 2020, wgpu issues, developer forums
**Evidence**: Synthesis: Asahi firmware cl_kill_timeout_ms=50 for compute kill. DRM scheduler timeout 100s. Developer forums report ~5 seconds before recovery. wgpu confirms Metal treats infinite loops as UB (optimized away) not detected at compile time. WWDC 2020 categorizes timeouts as runtime events.
**Tags**: timeout,iteration-ceiling,root-cause,watchdog,wall-clock,M4,empirical

## Finding 160: metal-cpp has zero measurable overhead vs Objective-C due to inlining of C++ function calls. However, it breaks ARC (Automatic Reference Counting), requiring manual retain/release. It registers ~1100 selectors at init time. C++ objects are NOT eligible for ARC.
**Confidence**: verified
**Source**: Getting started with Metal-cpp - Apple Developer
**Evidence**: Apple: 'No measurable overhead compared to calling Metal Objective-C headers, due to inlining of C++ function calls.' HN discussion revealed eager selector registration and lack of RAII wrappers despite being C++.
**Tags**: metal-cpp,zero-overhead,ARC,manual-memory

## Finding 161: Python→Metal paths: (1) PyObjC provides 1:1 bindings with bridge call overhead, (2) metalcompute (py-metal-compute) is a C extension achieving 2.53 TFLOPS on M1 with 27x speedup over CPU for 1.2M sine calcs (4ms vs 107ms), (3) MLX for ML workloads. metalcompute last released v0.2.9_r2 Jan 2025.
**Confidence**: high
**Source**: py-metal-compute - GitHub
**Evidence**: PyObjC (pyobjc-framework-Metal, Sep 2024) auto-generated from headers. metalcompute (GitHub: baldand/py-metal-compute, 88 stars) is C extension with NumPy buffer protocol support. MLX provides mx.fast.metal_kernel() for custom kernels.
**Tags**: Python,PyObjC,metalcompute,MLX

## Finding 162: Rust→Metal: metal-rs (v0.33.0, 693 stars) is officially deprecated as of Sep 2024. The replacement is objc2-metal (v0.3.2, ~1.16M downloads/month, 2109 dependents), which is auto-generated from Metal headers. wgpu is migrating from metal-rs to objc2-metal.
**Confidence**: verified
**Source**: Recommend objc2-metal instead of metal - GitHub
**Evidence**: GitHub issue #339 on metal-rs explains deprecation. objc2-metal provides Retained<T> for safe reference counting and complete Metal API coverage. LambdaClass demonstrated Metal FFT from Rust using the older metal-rs.
**Tags**: Rust,objc2-metal,metal-rs,deprecated

## Finding 695: macOS implements GPU recovery producing ".gpuRestart" kernel reports when GPU is reset after hang/timeout. Sequence: GPU hang detected -> gpuRestart event -> GPU reset attempted -> if recovery fails, watchdog timeouts on WindowServer follow (40-120 seconds) -> kernel panic and reboot. The firmware's agx_recovery task handles resets; "GPU faults seem to be quite poorly handled as a stop-the-world process."
**Confidence**: high
**Source**: Apple GPU (AGX) - Asahi Linux Documentation
**Evidence**: Asahi documentation: firmware runs agx_recovery task, "GPU faults are quite poorly handled as stop-the-world." Apple Community forums document: gpuRestart -> watchdog timeouts -> kernel panic sequence.
**Tags**: macOS,GPU-recovery,gpuRestart,watchdog,firmware,AGX,kernel-panic

## Finding 163: Multiple compute passes should be sequential compute encoders within a single command buffer. Each encoder can contain multiple kernel dispatches with different pipelines and bindings. Between encoders, Metal inserts implicit barriers — no explicit sync needed.
**Confidence**: verified
**Source**: Command Organization and Execution Model - Metal Programming Guide
**Evidence**: Metal Programming Guide: 'Only a single command encoder can be active at any point in time for a given command buffer.' Tech Talk: 'Encode multiple kernel dispatches to same encoder. Change kernel between dispatches.'
**Tags**: multi-pass,encoder,implicit-barrier

## Finding 164: Tile-based compute dispatches (since A11) execute within a render encoder with implicit barriers against fragment shading. Enables on-chip data reuse (G-Buffer → Tile Compute → Lighting) without system memory round-trips, keeping data in tile memory.
**Confidence**: verified
**Source**: Optimize Metal Performance for Apple silicon Macs - WWDC20
**Evidence**: WWDC 2020: 'Starting with A11 Bionic, Apple GPUs support tile-based compute dispatches. Tile dispatches introduce implicit barriers against fragment stage. Access to imageblocks, threadgroup memory, device memory.'
**Tags**: tile-compute,on-chip,TBDR,imageblock

## Finding 140: Multiple MTLCommandQueues help for compute when: (1) hiding GPU bubbles by keeping a second queue ready while the first completes, (2) overlapping compute with IO (MTLIOCommandQueue), (3) running independent compute streams. Cross-queue sync requires MTLEvent or MTLSharedEvent, not MTLFence.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Apple Tech Talks
**Evidence**: Tech Talk: 'To hide idle GPU time, consider using multiple CPU threads working on multiple pieces of work and keep the GPU busy either by creating multiple command buffers, or by creating multiple command queues.' Separate queues can run at different priorities.
**Tags**: multi-queue,parallelism,MTLEvent,IO

## Finding 151: True persistent kernels (kernels that run indefinitely) are NOT feasible on Metal due to the GPU watchdog timeout AND failures of forward progress. Apple GPUs do not guarantee that all dispatched threadgroups make progress concurrently, so spin-wait patterns across threadgroups may deadlock.
**Confidence**: high
**Source**: Prefix sum on portable compute shaders - Raph Levien
**Evidence**: Raph Levien: 'Apple GPUs exhibit failures of forward progress.' Metal Compute Tech Talk recommends CPU-driven re-dispatch pattern instead. The combination of watchdog timeout + scheduling constraints makes CUDA-style persistent threads impossible.
**Tags**: persistent-kernel,forward-progress,deadlock,limitation

## Finding 152: The recommended Metal pattern for pseudo-persistent compute is bounded kernel chunks with CPU re-dispatch, using MTLEvent/MTLSharedEvent for efficient chaining. Pre-enqueue multiple command buffers to hide CPU-GPU gaps. Use enqueue() to reserve execution order, then commit() when ready.
**Confidence**: verified
**Source**: Metal Compute on MacBook Pro - Apple Tech Talks
**Evidence**: Metal Tech Talk: 'Consider using multiple CPU threads working on multiple pieces of work and keep the GPU busy, either by creating multiple command buffers, or by creating multiple command queues.' MTLEvent enables GPU-GPU chaining within a device; MTLSharedEvent enables cross-process and CPU-GPU signaling.
**Tags**: persistent-kernel,re-dispatch,MTLEvent,chaining

## Finding 159: Compute pipeline state creation takes ~1ms for a typical kernel when not cached. Metal maintains an automatic filesystem cache making subsequent creations near-instant. Binary archives (MTLBinaryArchive) eliminate first-launch compilation entirely. Best practice: create all pipeline states asynchronously at app initialization using dispatch groups.
**Confidence**: verified
**Source**: Metal Best Practices Guide: Pipelines
**Evidence**: WWDC 2015 session 610 showed ~1ms CPU time for mid-frame compilation. WWDC 2021 session 10229 detailed binary archive workflow. Metal Best Practices Guide provides dispatch_group_t pattern for parallel async pipeline creation.
**Tags**: pipeline-creation,binary-archive,1ms,caching

## Finding 157: maxTotalThreadsPerThreadgroup on MTLComputePipelineDescriptor is an OCCUPANCY HINT to the compiler, not just a limit. It changes register allocation strategy. The Blender Cycles uber shader went from 58ms to 12.5ms (4.6x improvement) after occupancy tuning. The optimal value is NOT always the hardware maximum.
**Confidence**: verified
**Source**: Optimize GPU renderers with Metal - WWDC23
**Evidence**: WWDC 2023 session 10127: tuning maxTotalThreadsPerThreadgroup changes spill/register usage tradeoffs. At the sweet spot, spill increases slightly but more threadgroups run concurrently. Recommended: use GPU Debugger to measure spill per configuration.
**Tags**: occupancy,maxTotalThreadsPerThreadgroup,register-allocation,4.6x

## Finding 158: threadGroupSizeIsMultipleOfThreadExecutionWidth is a low-overhead optimization hint. When true, it enables a driver fast path by informing the compiler that dispatches will always use threadgroup sizes that are multiples of the SIMD width (32 on Apple GPUs). Asahi Linux shows this maps to a single hardware flag bit.
**Confidence**: verified
**Source**: MTLComputePipelineDescriptor - Apple Developer Documentation
**Evidence**: Asahi Linux reverse engineering: the flag maps directly to a control flag in the compute pipeline descriptor. gfx-rs/gfx#1998 confirmed measurable performance improvement on Intel GPUs; Apple GPU benefit is smaller but free to enable.
**Tags**: threadGroupSizeIsMultipleOfThreadExecutionWidth,optimization,SIMD-width

## Finding 147: Metal provides two hazard tracking modes: MTLHazardTrackingMode.default (automatic - Metal tracks read/write dependencies and inserts sync primitives) and .untracked (developer manually manages via fences/events/barriers). Automatic tracking can cause false dependencies that serialize work unnecessarily.
**Confidence**: verified
**Source**: MTLHazardTrackingMode - Apple Developer Documentation
**Evidence**: WWDC 2020: 'Unnecessary dependencies serialize workloads. Fragment Output → Vertex Input = Creates GPU idle bubbles.' Solution: 'Use separate Metal resources to eliminate false dependencies' or 'Mark resources as untracked to control synchronization manually.'
**Tags**: hazard-tracking,false-dependencies,untracked,performance

## Finding 138: A single MTLCommandQueue executes command buffers serially in enqueue order. There is no 'concurrent queue' concept in Metal. Concurrency is achieved via multiple queues or MTLDispatchType.concurrent within a single encoder. Concurrent dispatch type allows multiple dispatches to execute simultaneously within one encoder.
**Confidence**: verified
**Source**: Command Organization and Execution Model - Metal Programming Guide
**Evidence**: Metal Programming Guide: 'All command buffers sent to a single queue are guaranteed to execute in the order in which the command buffers were enqueued.' Created via makeComputeCommandEncoder(dispatchType: .concurrent).
**Tags**: command-queue,serial,concurrent,dispatch-type

## Finding 139: On Apple GPUs, the case where concurrent dispatches improve overall compute performance is rare. The GPU hardware already efficiently distributes work across cores from serial dispatches. Concurrent dispatch mainly helps when individual dispatches are too small to saturate the GPU.
**Confidence**: high
**Source**: Coherency, synchronization, scheduling - Apple Developer Forums
**Evidence**: Apple Developer Forums: 'In Apple GPUs, the case where concurrent dispatches improve overall compute performance is rare.' Apple's TBDR architecture already handles work distribution efficiently across GPU cores.
**Tags**: concurrent,performance,Apple-GPU,TBDR

## Finding 684: Shader Validation has HIGH performance and memory overhead — pipelines take longer to compile and ALL Metal commands go through validation. When enabled, maxTotalThreadsPerThreadgroup and threadExecutionWidth may return DIFFERENT values. Does NOT support binary function pointers or dynamic linking. On MTLGPUFamilyMac1/Apple5 and older, global memory access of pointers from argument buffers is NOT checked. Since Xcode 13, supports ICBs and dynamic libraries.
**Confidence**: verified
**Source**: Debug GPU-side errors in Metal - WWDC20
**Evidence**: WWDC20 session 10616 warns about changed maxTotalThreadsPerThreadgroup values. ICB support enabled via MTL_SHADER_VALIDATION_GPUOPT_ENABLE_INDIRECT_COMMAND_BUFFERS=1.
**Tags**: validation,shader-validation,performance-overhead,limitations,threadgroup

## Finding 680: waitUntilCompleted blocks CPU until GPU finishes AND all completion handlers execute. Apple recommends MTLSharedEvent over waitUntilCompleted: "MTLSharedEvents have lower overhead and can help reduce timeline gaps." The recommended pattern for resource management is DispatchSemaphore with addCompletedHandler rather than blocking.
**Confidence**: verified
**Source**: Scale compute workloads across Apple GPUs - WWDC22
**Evidence**: WWDC22 "Scale compute workloads across Apple GPUs" explicitly recommends SharedEvents over waitUntilCompleted. Completion handlers fire "immediately after GPU finishes" on Metal's internal serial dispatch queue (com.Metal.CompletionQueueDispatch). Handlers should "perform quickly" — expensive work deferred to another thread.
**Tags**: synchronization,waitUntilCompleted,completedHandler,MTLSharedEvent,overhead

## Finding 141: Metal provides three sync levels: MTLFence synchronizes across passes (encoders) within the SAME command buffer. MTLEvent synchronizes across different command buffers within the same queue. MTLSharedEvent synchronizes across different queues or between CPU and GPU. MTLSharedEvent also enables cross-process synchronization.
**Confidence**: verified
**Source**: Resource synchronization - Apple Developer Documentation
**Evidence**: Apple documentation: 'MTLFence synchronizes access to one or more resources across different render and compute passes within a single command queue.' MTLEvent for cross-buffer sync. MTLSharedEvent for cross-queue and CPU-GPU.
**Tags**: MTLFence,MTLEvent,MTLSharedEvent,synchronization

## Finding 142: memoryBarrier(scope:) on MTLComputeCommandEncoder enforces write-read ordering within a SINGLE concurrent encoder. Between different encoders, Metal automatically inserts barriers. Explicit barriers are only needed within a concurrent dispatch encoder where dispatches may overlap.
**Confidence**: verified
**Source**: memoryBarrier(scope:) - Apple Developer Documentation
**Evidence**: Metal docs: memoryBarrier(scope:) 'Creates a memory barrier that enforces the order of write and read operations for specific resource types.' Available since macOS 10.14. Between encoders (endEncoding → new encoder), ordering is implicit.
**Tags**: memory-barrier,concurrent,synchronization

## Finding 676: Creating command buffers with commandBufferWithUnretainedReferences yields 2% CPU reduction by eliminating ARC overhead. Combined with untracked heap resources (hazardTracking = .untracked) and manual fences, this eliminates false sharing where Metal conservatively serializes access to heap sub-resources, improving GPU parallelism for concurrent compute dispatches.
**Confidence**: verified
**Source**: Go bindless with Metal 3 - WWDC22
**Evidence**: WWDC22 benchmarked: unretainedReferences = 2% CPU savings. Untracked heaps solve false sharing: heap subresources appear as single resource to Metal, causing conservative scheduling. Solution: hazardTracking .untracked + MTLFence for ordering. Memory barriers after fragment stage are very high cost (similar to render pass split); Apple GPUs disable them entirely.
**Tags**: unretained-references,untracked-heaps,false-sharing,compute,performance,fences

