# metal-compute — All Findings (40)

## Finding 144: IOAF error codes: code 2 = GPU Timeout Error (command buffer took too long), code 3 = GPU Hang (possible infinite loop), code 4 = execution aborted due to prior error, code 5 = discarded (victim of GPU error/recovery). Apple deliberately does not publicly document these codes.
**Confidence**: high
**Source**: Metal FAQ - Sealed Abstract
**Evidence**: Metal FAQ (sealedabstract.com) and developer forum reports. Apple treats IOAF codes as driver/kernel-level issues. If you encounter them, file a bug report.
**Tags**: IOAF,error-codes,timeout,GPU-hang

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

## Finding 148: dispatchThreadgroups(indirectBuffer:indirectBufferOffset:threadgroupsPerGrid:) reads the number of threadgroups from a GPU-accessible buffer, enabling GPU-generated grid sizes. The threadgroup size is still CPU-specified. Available since iOS 12/macOS 10.14.
**Confidence**: verified
**Source**: MTLComputeCommandEncoder - Apple Developer Documentation
**Evidence**: Apple Metal documentation. The indirect buffer contains three uint32 values (threadgroupsPerGrid x, y, z). A compute kernel can write these values and the next dispatch reads them.
**Tags**: indirect-dispatch,GPU-driven,grid-size

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

