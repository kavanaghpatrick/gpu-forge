# mlx-compute — All Findings (49)

## Finding 338: ASTC texture decompression trick (proposed Issue #328): encode LLM weights as 6x6 ASTC texture blocks in HDR-ch mode. Apple GPU fixed-function ASTC decoder (since A7 chip) decompresses 'for free' during texture-sampling. ~3.6 bits/weight, 4-5x bandwidth reduction, virtually zero extra latency, <1pp MMLU accuracy loss.
**Confidence**: high
**Source**: medium
**Evidence**: Draft API: astc.encode_weights() for offline compression, astc.load_astc_weights() for runtime loading with CPU fallback. Feature remains community request (opened July 2025, no PR merged). Metal/MPS can sample .astc textures with one line of shader code. LoRA adaptation can recover accuracy loss from ASTC compression.
**Tags**: 

## Finding 342: MLX quantization: mx.quantize(w, bits, group_size) supports 2/3/4/5/6/8 bits with group_size 32/64/128. Affine mode: scale+bias per group. MXFP4 mode: microscaling FP4 format. mx.quantized_matmul() performs matmul with quantized matrix. mx.dequantize() reconstructs approximate values from quantized representation.
**Confidence**: verified
**Source**: high
**Evidence**: mx.nn.quantize() quantizes model modules. QLoRA: NormalFloat4 quantization of base model + LoRA adapters. Quantization example: quantized_weight, scales, biases = mx.quantize(weight, bits=4, group_size=32). Smaller group_size=32 gives better accuracy with more metadata, larger group_size=128 less metadata slightly lower accuracy.
**Tags**: 

## Finding 345: MLX interop: NumPy conversion via np.array(mx_arr) (copy) or np.array(mx_arr, copy=False) (zero-copy view). float64 auto-converted to float32. bfloat16 needs manual cast. PyTorch: torch.tensor(memoryview(mx_arr)) but experimental/breaks for multi-dim (cast to NumPy first advised). JAX: direct jnp.array(mx_arr).
**Confidence**: verified
**Source**: high
**Evidence**: Memory views don't propagate to MLX's automatic differentiation. DLPack protocol supported for cross-framework exchange. TensorFlow: tf.constant(memoryview(mx_arr)). Key tradeoff: default copying prevents gradient inconsistencies; zero-copy views optimize memory but risk gradient computation when external modifications occur.
**Tags**: 

## Finding 348: Extending MLX with C++ primitives: inherit from Primitive base class, implement eval_cpu(), eval_gpu(), jvp(), vjp(), vmap(), name(), is_equivalent(). Operations are front-end functions; Primitives define evaluation rules. GPU impl uses Metal kernels with instantiate_kernel() macro.
**Confidence**: verified
**Source**: high
**Evidence**: Build system: CMakeLists.txt with mlx_build_metallib() for Metal kernels. Python binding via nanobind (NB_MODULE). Setup uses mlx.extension.CMakeExtension and mlx.extension.CMakeBuild. Type dispatch via templates (float32, float16, bfloat16, complex64). All operations receive StreamOrDevice parameter for scheduling.
**Tags**: 

## Finding 351: mx.fast optimized operations: rms_norm (accumulates in higher precision, don't upcast/downcast around it), layer_norm, rope (rotary position embedding), scaled_dot_product_attention (softmax in float32 regardless of input precision, supports GQA/MQA without pre-tiling k/v, causal mask type).
**Confidence**: high
**Source**: high
**Evidence**: Performance tips from Awni Hannun: evaluate at iteration boundaries not continuously. Avoid frequent .item() calls in loops (use .tolist()). Python scalars preserve dtype (my_fp16 * 2.0 stays fp16). mx.addmm(a,b,c) > a@b+c. Prefer broadcasting over concatenation. Monitor recompilation with shapeless=True (easy mistakes). Profile GPU utilization with mactop.
**Tags**: 

## Finding 378: MLX supported dtypes: float16, bfloat16, float32, complex64, plus integer types. Type promotion follows NumPy rules. Python scalars preserve dtype of array operand (my_fp16 * 2.0 stays fp16, but my_fp16 * mx.array(2.0) upcasts to float32). bfloat16 needs manual cast for NumPy interop.
**Confidence**: high
**Source**: medium
**Evidence**: Mixed precision in MLX: no automatic mixed precision (AMP) like PyTorch. Manual dtype management. mx.fast.rms_norm and layer_norm accumulate in higher precision internally. scaled_dot_product_attention does softmax in float32 regardless of input. Quantization provides pseudo-mixed-precision: 4-bit weights with float16/float32 activations.
**Tags**: 

## Finding 297: mx.compile() fuses multiple GPU kernel launches into a single kernel. Three phases: 1) Graph building (traces function with placeholder inputs), 2) Optimization (identifies fusible ops, merges them), 3) Code generation (generates and compiles optimized kernels). Element-wise operations are primary fusion candidates.
**Confidence**: verified
**Source**: high
**Evidence**: GELU benchmark on M1 Max: uncompiled 15.5ms → compiled 3.1ms = ~5x speedup. Benefits both small overhead-bound and large bandwidth-bound arrays by eliminating intermediate memory operations. Compilation cached automatically; subsequent calls with identical input shapes/types skip recompilation.
**Tags**: 

## Finding 299: Recompilation triggers: input shapes/dims change, input dtypes change, number of inputs changes. shapeless=True avoids recompilation on variable shapes but disables shape-dependent conditionals (risky). Compiled functions must be pure (no side effects). Use inputs=/outputs= params to capture external state changes.
**Confidence**: verified
**Source**: high
**Evidence**: Decorator usage: @mx.compile or @partial(mx.compile, inputs=state, outputs=state). Composable with transforms: mx.compile(mx.grad(mx.exp)). Debug with mx.disable_compile() or MLX_DISABLE_COMPILE env var. Training pattern: compile step function with model.state and optimizer.state as inputs/outputs. Include mx.random.state for stochastic layers.
**Tags**: 

## Finding 300: MLX lazy evaluation scheduler: eval_impl() performs DFS dependency analysis to count dependencies, then BFS execution planning to build execution tape respecting dependencies and memory limits. Graph decouples building from execution, enabling transformations and optimizations before computing.
**Confidence**: verified
**Source**: high
**Evidence**: Key optimization: operation fusion reduces kernel launch overhead. MLX does NOT currently perform explicit dead code elimination or common subexpression elimination as separate passes. The main optimization path is through mx.compile() kernel fusion. Lazy evaluation also means you only pay for computations actually used.
**Tags**: 

## Finding 406: MLX added dynamic broadcasting support for shapeless compile/export via BroadcastAxes primitive with ignore_axes parameter. Dynamic reshape support also added, critical for LLM attention layers with varying sequence lengths. Works with mx.export_function for exporting compiled functions.
**Confidence**: verified
**Source**: Dynamic broadcasting for shapeless compile/export
**Evidence**: GitHub PR #1722 (BroadcastAxes) and PR #1657 (dynamic reshape)
**Tags**: compile,shapeless,dynamic_broadcasting,export,BroadcastAxes

## Finding 407: Accidental type upcasting is a common performance trap. Using my_fp16_array * mx.array(2.0) promotes to float32 since mx.array(2.0) is float32. Using my_fp16_array * 2.0 preserves float16 because Python scalars are weakly typed. MLX fast operations handle precision internally so explicit upcasting is unnecessary.
**Confidence**: high
**Source**: Writing Fast MLX - Awni Hannun
**Evidence**: Writing Fast MLX guide by Awni Hannun (MLX core developer)
**Tags**: type_promotion,float16,performance,best_practices

## Finding 293: mx.fast.metal_kernel(name, input_names, output_names, source, ensure_row_contiguous=True, atomic_outputs=False, init_value=None) creates custom Metal kernels from Python. Write only kernel body; framework auto-generates function signature with buffers, template instantiation, and Metal attributes.
**Confidence**: verified
**Source**: high
**Evidence**: For each input 'a': provides const device T* a, plus a_shape, a_strides, a_ndim if referenced. Output arrays always row-contiguous. Utils from mlx/backend/metal/kernels/utils.h auto-included (e.g., elem_to_loc()). Supports Metal attributes: thread_position_in_grid, thread_index_in_simdgroup, etc. Build kernel once, reuse many times to reduce JIT overhead.
**Tags**: 

## Finding 295: metal_kernel() execution args: inputs (list of mx.array), template (type params like [('T', mx.float32)] supporting Dtype/int/bool), grid (x,y,z total thread count), threadgroup (x,y,z threads per group), output_shapes, output_dtypes. verbose=True prints generated Metal code.
**Confidence**: verified
**Source**: high
**Evidence**: Key features: ensure_row_contiguous=True auto-copies non-contiguous inputs. atomic_outputs=True enables simultaneous updates from multiple threadgroups. init_value initializes all outputs before execution (useful for accumulation). SIMD operations like simd_sum() work for reductions. Compatible with mx.custom_function decorator for defining VJPs.
**Tags**: 

## Finding 404: Setting atomic_outputs=True enables concurrent thread writes but using init_value=0 causes up to 3x performance regression due to unnecessary zero-initialization overhead. After removing init_value=0 from an FDTD solver, performance improved from 11.4s to 3.31s on M3 Max, actually 30% faster than py-metal-compute (4.6s).
**Confidence**: verified
**Source**: Low performance of mx.fast.metal_kernel with repeated kernel calls
**Evidence**: GitHub issue #1828 with maintainer diagnosis and benchmarks
**Tags**: custom_kernel,atomic_outputs,init_value,performance,regression

## Finding 405: Every time a new metal_kernel is created, a new Metal library is JIT compiled. To reduce overhead, the kernel should be built once and reused. The class-based design was chosen over single-function approach specifically for caching. Kernel hashing is based on source code and template arguments.
**Confidence**: verified
**Source**: Custom Metal Kernels from Python by barronalex
**Evidence**: MLX documentation and PR #1325 discussion confirming caching rationale
**Tags**: custom_kernel,jit,compilation,caching,performance

## Finding 312: MLX supports 4 distributed backends: MPI (full-featured), Ring (TCP socket-based, always available), JACCL (low-latency RDMA over Thunderbolt, macOS M3 Ultra+), NCCL (GPU-optimized for CUDA). Primitives: all_sum(), all_gather(), send(), recv(), recv_like(). Operations become no-ops in single-process mode.
**Confidence**: verified
**Source**: high
**Evidence**: Launch via mlx.launch: -n 4 script.py (local), --hosts ip1,ip2 (remote), --backend jaccl --hostfile config.json. Manual env vars: Ring uses MLX_RANK/MLX_HOSTFILE, JACCL uses MLX_RANK/MLX_JACCL_COORDINATOR/MLX_IBV_DEVICES. Ring backend uses sequential topology (rank i to i-1/i+1). mlx.distributed_config auto-discovers Thunderbolt connections.
**Tags**: 

## Finding 317: JACCL backend: RDMA over Thunderbolt 5 available since macOS 26.2 (Dec 12, 2025). Achieves communication latency an order of magnitude lower than ring backend. Real-world latencies 5-9 microseconds. Thunderbolt 5 bandwidth up to 80Gbps (2x TB4). Requires fully connected mesh topology (every node direct-connected).
**Confidence**: high
**Source**: high
**Evidence**: Setup: Enable RDMA via recovery mode (rdma_ctl enable). JACCL currently supports only full mesh, ring/bandwidth optimization planned. Jeff Geerling tested 4 Mac Studios: 1.5TB total VRAM, 3.7 TFLOPS HPL (FP64). Kimi K2 Thinking (1T params) across 4x M3 Ultra at ~30 tok/s. Exo 1.0 is primary RDMA clustering tool. Limitation: no Thunderbolt switches exist, max ~4-5 Macs.
**Tags**: 

## Finding 319: Data parallelism in MLX: mlx.nn.average_gradients() efficiently aggregates gradients across nodes. Batch communications rather than many small messages for performance. Ring backend benefits over MPI even for Ethernet but main purpose is Thunderbolt rings for higher bandwidth. JACCL latency-optimized for full mesh.
**Confidence**: high
**Source**: high
**Evidence**: macOS 26.2 RDMA transforms consumer Macs into enterprise AI infrastructure. Awni Hannun demo: Kimi K2 Thinking 1T param model across 4x M3 Ultra with 2TB total RAM at 15-18 tok/s. Total cluster cost under K vs K+ equivalent NVIDIA infrastructure. RDMA latency drop from 300us to <50us.
**Tags**: 

## Finding 331: MLX FFT support: full suite of GPU-accelerated FFT operations. 1D: fft/ifft, rfft/irfft. 2D: fft2/ifft2, rfft2/irfft2. N-D: fftn/ifftn, rfftn/irfftn. Parameters: n (transform size), axis (which axis). FFT-based convolution not yet a dedicated op but building blocks available.
**Confidence**: verified
**Source**: high
**Evidence**: MLX as general-purpose compute: NumPy-like API for numerical simulations, scientific computing, and ML. Operations: sort, scatter, gather (with performance improvements in PR #1541), unique, repeat, concatenate. Broadcasting preferred over concatenation. Matrix ops: x @ W.T faster than x @ W for vector-matrix, mx.addmm(a,b,c) faster than a@b+c.
**Tags**: 

## Finding 335: MLX CUDA backend support added July 2025: code written for MLX can run on NVIDIA GPUs. Prototype on Ubuntu 22.04/CUDA 11.6. Core ops supported: matmul, softmax, reduction, sorting, indexing. First version on PyPI (mlx-cuda). Write-once-run-anywhere: prototype on Mac, deploy on NVIDIA clusters.
**Confidence**: high
**Source**: high
**Evidence**: AMD ROCm support proposed (Issue #2556). CUDA backend enables researchers to prototype locally on Mac using MLX then deploy on large-scale NVIDIA GPU clusters. MLX becoming a cross-platform array framework rather than Apple-only, though Apple Silicon remains the primary optimization target.
**Tags**: 

## Finding 362: MLX unified memory model: arrays live in shared memory accessible by CPU and GPU. No explicit data transfers needed. Operations can target any supported device via stream parameter. Key advantage for GPU compute: KV cache (grows linearly with context, can be tens of GB) stays in place, no device-to-device transfer.
**Confidence**: high
**Source**: high
**Evidence**: MLX function transformations: mx.grad() for automatic differentiation (nestable for higher-order derivatives), mx.vmap() for vectorized mapping, mx.compile() for kernel fusion. All composable: mx.compile(mx.grad(fn)). MLX is NOT just ML - it's a general-purpose numerical computing framework that happens to excel at ML.
**Tags**: 

## Finding 408: MLX provides comprehensive general-purpose compute: mlx.core.fft (full suite), mlx.core.linalg (inv, norm, cholesky, qr, svd, eig, solve), array manipulation, reductions, bitwise ops. Apple WWDC 2025 explicitly positions MLX for numerical simulations and scientific computing alongside ML.
**Confidence**: verified
**Source**: MLX 0.30.6 documentation
**Evidence**: MLX documentation index and WWDC 2025 positioning statement
**Tags**: general_compute,fft,linalg,scientific_computing

## Finding 409: MLX sort operation achieves exceptional performance beating all tested CUDA GPUs including RTX 4090 on M2 Max, M2 Ultra, and M3 Max, making MLX competitive for non-ML data processing workloads.
**Confidence**: high
**Source**: MLX Benchmark - Apple MLX operations on all Apple Silicon chips
**Evidence**: mlx-benchmark repository results comparing Apple Silicon vs CUDA GPUs
**Tags**: sort,performance,benchmark,cuda_comparison

## Finding 410: MLX layout optimization tips: vector-matrix x @ W.T faster than x @ W; matrix-vector W @ x faster than W.T @ x; use mx.addmm for fused a @ b + c; prefer mx.repeat() over concatenation; use mx.take_along_axis over fancy indexing for gather operations.
**Confidence**: high
**Source**: Writing Fast MLX - Awni Hannun
**Evidence**: Writing Fast MLX guide by Awni Hannun
**Tags**: layout,performance,matmul,optimization

## Finding 411: MLX has added CUDA support (cuda_kernel alongside metal_kernel in mx.fast), extending beyond Apple Silicon. MLX is evolving from Apple-only toward cross-platform array framework though Metal remains the primary target.
**Confidence**: verified
**Source**: MLX 0.30.6 documentation
**Evidence**: MLX 0.30.6 documentation lists mlx.core.cuda as submodule alongside mlx.core.metal
**Tags**: cuda,cross_platform,metal

## Finding 307: M5 GPU has dedicated Neural Accelerators (TensorOps) for matrix multiplication. MLX accesses them via Metal 4 TensorOps and Metal Performance Primitives framework. TTFT speedups vs M4: Qwen 1.7B 3.57x, Qwen 8B BF16 3.62x, Qwen 14B 4-bit 4.06x, Qwen 30B MoE 4-bit 3.52x, GPT-OSS 20B MXFP4 3.33x.
**Confidence**: verified
**Source**: high
**Evidence**: Token generation (memory-bandwidth-bound) shows 1.19-1.27x speedup (limited by bandwidth not compute). Image generation (FLUX-dev-4bit) 3.8x faster. M5 memory bandwidth 153GB/s vs M4 120GB/s (28% improvement). MLX maintains unified memory arch - operations run on CPU or GPU without moving memory.
**Tags**: 

## Finding 326: MLX Swift provides full ML capabilities via Swift Package Manager for macOS, iOS, iPadOS, and visionOS. Consistent naming conventions across Python and Swift APIs. Example apps: MNISTTrainer (iOS+macOS LeNet training), LLMEval (iOS+macOS LLM inference), MLXChatExample (iOS+macOS chat with LLMs+VLMs).
**Confidence**: verified
**Source**: high
**Evidence**: iOS Simulators don't support Metal features needed for mlx-swift - requires physical device with max RAM. Swift API uses __call__ not forward for layers. MLX Swift has C API layer connecting Swift with C++ libraries. Phi-3 on VisionPro: 22.25 tokens/second fully offline. Swift API provides prototyping-to-production workflow across all Apple platforms.
**Tags**: 

## Finding 327: MLX-Outil: tool calling library using MLX Swift across iOS, macOS, visionOS. MLX Swift Examples provides multi-model runtime and integration examples. mlx-swift available via Swift Package Index. MLX built on top of Metal, works across macOS, iOS, iPadOS, visionOS.
**Confidence**: verified
**Source**: high
**Evidence**: CoreML interop: models must be converted to CoreML format via coremltools. MLX offers C API to bridge Swift and C++ for high-performance integration. Key difference from Python: Swift API allows on-device deployment without Python runtime overhead. MLX Swift mirrors Python API design for cross-language consistency.
**Tags**: 

## Finding 288: eval_gpu() dispatch: Each MLX Primitive subclass implements eval_gpu() containing Metal-specific logic. The dispatch uses virtual methods on the Primitive base class. eval_gpu() determines the kernel, sets inputs, resolves grid dimensions, and dispatches to GPU via Metal compute encoder.
**Confidence**: high
**Source**: high
**Evidence**: Metal backend uses DeviceStream abstraction wrapping MTL::CommandQueue (one per stream index). Command encoding: 1) Retrieve/create CommandEncoder for target stream, 2) Set compute pipeline state (select kernel), 3) Bind input/output buffers and scalar parameters, 4) Configure thread dispatch dimensions, 5) Insert synchronization fences for cross-stream dependencies. Fence-based synchronization enables GPU-driven sync without CPU stalls.
**Tags**: 

## Finding 289: MLX kernel compilation uses dual delivery: pre-compiled .metallib (kernels compiled offline from .metal to .air to .metallib) and JIT compilation (MLX_METAL_JIT CMake flag). JIT minimizes library size by runtime-compiling kernels on first use. JIT incurs cold-start cost (100ms-few seconds) but Metal kernel cache persists across reboots.
**Confidence**: verified
**Source**: high
**Evidence**: Default library: core pre-compiled kernels bundled as mlx.metallib. Kernel lookup queries library for compiled function objects. JIT fallback: if JIT enabled, Metal compiler dynamically generates kernels not in default library. mlx.metallib must be at same directory as executable linked to libmlx.a, or METAL_PATH preprocessor constant must point to it. Build with MLX_METAL_JIT=ON, CMAKE_BUILD_TYPE=MinSizeRel for smaller binaries.
**Tags**: 

## Finding 290: MLX command buffer batching: Multiple operations encoded into single command buffer before submission. Lazy evaluation builds computation graph, scheduler uses eval_impl() with DFS dependency analysis (count deps) then BFS execution planning (build execution tape respecting deps and memory limits). Only evaluated when mx.eval() called.
**Confidence**: high
**Source**: high
**Evidence**: Command buffers are submitted per-stream via MTL::CommandQueue. Each stream maintains its own queue enabling pipelined execution and overlapping command submission. Cross-stream dependencies handled by fence-based synchronization. mx.async_eval() pipelines graph construction with computation for latency-sensitive code.
**Tags**: 

## Finding 292: MLX memory management: allocator::Buffer uses shared_ptr with deleter for automatic cleanup. allocator::malloc() allocates device-specific memory (Metal buffers). Arrays freed when reference count reaches zero. Memory challenges: intermediate activations released during computation but mechanisms not fully transparent.
**Confidence**: verified
**Source**: high
**Evidence**: Memory functions: mx.get_active_memory(), mx.get_peak_memory(), mx.get_cache_memory(), mx.set_memory_limit(), mx.set_cache_limit(), mx.set_wired_limit(), mx.clear_cache(). Tips: cast weights to lower precision before evaluating (reduces peak memory ~1/3), use del to release temporary references before mx.eval(), pass file paths as strings to mx.load() not file handles.
**Tags**: 

## Finding 358: Metal debugging in MLX: mx.metal.start_capture(path) initiates Metal capture session for GPU profiling, mx.metal.stop_capture() terminates it. mx.metal.is_available() checks Metal availability. mx.metal.device_info() returns device information. Capture files can be analyzed in Xcode GPU debugger.
**Confidence**: verified
**Source**: high
**Evidence**: Additional memory functions: mx.reset_peak_memory() clears peak stats, mx.device_count() returns available devices, mx.default_device()/mx.set_default_device() for device management. Profiling tip: check GPU utilization first with mactop - non-100% utilization suggests CPU bottlenecks in data loading/preprocessing.
**Tags**: 

## Finding 373: MLX kernel directory: mlx/backend/metal/kernels/ contains specialized Metal shader implementations. Key kernels: scaled_dot_product_attention.metal, sdpa_vector.h, conv.metal, plus normalization (LayerNorm, RMSNorm), RoPE, reduction, binary/unary ops. Compiled via CMakeLists.txt into .metallib format.
**Confidence**: high
**Source**: high
**Evidence**: MLX layered architecture: 1) Python API (mlx.core, mlx.nn, mlx.core.fast), 2) Python-C++ bindings (Nanobind), 3) C++ Core (array class, ops, primitives, transforms), 4) Backend abstraction (eval_cpu/eval_gpu), 5) Hardware backends (Metal, CUDA, CPU). Metal uses Device singleton per GPU. CPU uses Accelerate (macOS) or OpenBLAS (Linux).
**Tags**: 

## Finding 374: MLX array is a DAG node storing: shape, dtype, primitive reference, input arrays, data buffer, status flag. Three states: unscheduled (created not evaluated), scheduled (in execution queue), available (computation complete with data in memory). Follows NumPy broadcasting rules.
**Confidence**: high
**Source**: high
**Evidence**: Type system: automatic type promotion for Python scalars, NumPy-compatible indexing via ArrayAt system. mx.core.fast module provides optimized primitives inheriting from fast::Custom with specialized VJP rules: RMSNorm, LayerNorm, scaled_dot_product_attention (intelligent kernel selection), RoPE. Custom gradient implementations avoid intermediate allocations.
**Tags**: 

## Finding 394: MLX GPU dispatch follows a virtual method pattern where each Primitive subclass implements eval_gpu(). The core dispatch in eval.cpp creates a scoped memory pool, gets the command buffer for the array's stream via d.get_command_buffer(s.index), then calls arr.primitive().eval_gpu(). Input references are preserved during tracing to prevent donated buffers from being released prematurely.
**Confidence**: verified
**Source**: MLX Metal Backend eval.cpp
**Evidence**: Direct source code reading of mlx/backend/metal/eval.cpp from ml-explore/mlx main branch
**Tags**: eval_gpu,dispatch,command_buffer,metal_backend,primitives

## Finding 395: Binary operations use a BINARY_GPU macro delegating to binary_op_gpu(). Kernel names are constructed dynamically based on shape: prefixes ss/sv/vs/vv/g plus dimensionality suffix. Large data adds 2 suffix, multiple work items per thread adds n. Thread group size is fixed at 1024 threads for Metal compliance.
**Confidence**: verified
**Source**: MLX Metal Backend binary.cpp
**Evidence**: Direct source code reading of mlx/backend/metal/binary.cpp
**Tags**: binary_ops,kernel_naming,dispatch,broadcasting,metal_backend

## Finding 396: MLX command buffer batching uses device-tier-specific thresholds checked by command_buffer_needs_commit(): phone/iPad (p) = 20 ops/40 MB, base GPU (g) = 40 ops/40 MB, studio/Max (s) = 50 ops/50 MB, Ultra (d) = 50 ops/50 MB. Overridable via MLX_MAX_OPS_PER_BUFFER and MLX_MAX_MB_PER_BUFFER environment variables.
**Confidence**: verified
**Source**: MLX Metal Device Implementation
**Evidence**: Direct source code reading of mlx/backend/metal/device.cpp showing threshold constants and env var overrides
**Tags**: command_buffer,batching,flush_strategy,device_tiers,metal_backend,performance

## Finding 397: MLX eval_impl() uses two-phase evaluation: Phase 1 (DFS) walks the graph backward counting dependencies; Phase 2 (BFS with width limiting, default max width 20 via MLX_BFS_MAX_WIDTH) processes arrays breadth-first, switching to DFS when width exceeds limit. MAX_ACTIVE_TASKS=10, scheduler waits when active tasks exceed limit or memory pressure is high.
**Confidence**: verified
**Source**: MLX Transforms - eval_impl
**Evidence**: Direct source code reading of mlx/transforms.cpp and mlx/utils.h
**Tags**: eval_impl,scheduler,DFS,BFS,dependency_analysis,memory_pressure,lazy_evaluation

## Finding 398: MLX MetalAllocator caching strategy: size rounding to page boundaries for large allocs, cache lookup before new allocation, small buffer optimization (<small_size_) from 1MB Metal heap with shared storage, GC triggers at 95% of max_recommended_working_set_size, block_limit = min(1.5x max_recommended, 0.95x total_memory). Thread-safe via std::unique_lock with ResidencySet tracking.
**Confidence**: verified
**Source**: MLX Metal Allocator Implementation
**Evidence**: Direct source code reading of mlx/backend/metal/allocator.cpp and allocator.h
**Tags**: memory_allocator,buffer_cache,garbage_collection,metal_backend,unified_memory,residency_set

## Finding 399: MLX achieves memory reuse during evaluation through reference counting and detachment. After eval_impl calls eval on each intermediate array, arr.detach() decrements input reference counts. When refcount reaches zero, the buffer is freed back to the cache pool. Peak memory for a 10-layer model is approximately 3.8x a single activation tensor (not 10x).
**Confidence**: verified
**Source**: Memory reusing / garbage collection mechanism during a single eval
**Evidence**: GitHub discussion #912 with MLX maintainer responses confirming the mechanism
**Tags**: memory_reuse,reference_counting,garbage_collection,detach,eval_impl

## Finding 400: MLX kernel library at mlx/backend/metal/kernels/ contains ~24 primary .metal files organized as: core compute (arange, binary, conv, copy, fft, gemv, quantized, reduce, softmax, sort, unary), steel subdirectory (gemm/, conv/, attn/, utils/ for high-performance tiled implementations). Build pipeline: .metal -> .air (xcrun metal) -> mlx.metallib (xcrun metallib) with -fno-fast-math.
**Confidence**: verified
**Source**: MLX Metal Kernels Directory
**Evidence**: Direct reading of GitHub directory listing and mlx/backend/metal/kernels/CMakeLists.txt
**Tags**: kernel_library,metal_shaders,steel,build_pipeline,metallib

## Finding 401: MLX Steel GEMM kernels use simdgroup_multiply_accumulate with 6 block configs (64x64/BK16/WG2x2 through 64x32/BK8/WG4x1), each generating 4 transpose variants x 4 dtypes. Padding=16/sizeof(T) prevents bank conflicts. Dispatch selects GEMV for vectors, Split-K for small M*N with large K, NAX Split-K for newer architectures, or regular GEMM. Complex dtype uses Karatsuba multiplication.
**Confidence**: verified
**Source**: MLX Steel GEMM Kernel Implementation
**Evidence**: Direct source code reading of mlx/backend/metal/kernels/steel/gemm/gemm.h, mma.h, and matmul.cpp
**Tags**: steel,gemm,simdgroup_matrix,tiled_matmul,kernel_selection,split_k

## Finding 402: MLX supports two kernel delivery modes: pre-compiled (.metal -> .air -> mlx.metallib loaded at startup) and JIT compilation (Device::build_library_ compiles Metal source strings at runtime with fastMathEnabled=false). Two-level caching: library-level by name using shared_mutex, and kernel-level within each library for compiled pipeline states.
**Confidence**: verified
**Source**: MLX Metal Device - Library Loading and JIT Compilation
**Evidence**: Direct source code reading of mlx/backend/metal/device.cpp showing both newLibrary calls and caching maps
**Tags**: jit_compilation,metallib,kernel_caching,metal_kernel,pipeline_state

## Finding 403: MLX integrates Metal Residency Sets for GPU memory management. Non-heap buffer allocations are inserted into a ResidencySet. Requires GPUFamilyApple6 (not GPUFamilyMetal3 as initially implemented, corrected in Issue #1855). Residency sets keep frequently-used GPU buffers resident in fast memory, reducing page-in latency.
**Confidence**: verified
**Source**: Metal ResidencySet requires GPUFamilyApple6
**Evidence**: Source code in allocator.cpp showing residency_set_ member; GitHub Issue #1855 documenting the GPU family fix
**Tags**: residency_set,memory_management,gpu_family,metal_backend

## Finding 95: MLX arrays are backed by Metal buffers on Apple Silicon. Interop paths: (1) mx.fast.metal_kernel() allows custom Metal kernels to operate directly on MLX array backing buffers, (2) DLPack protocol (__dlpack__, __dlpack_device__) enables zero-copy sharing with other frameworks like PyTorch/JAX, (3) MLX metal module provides is_available(), device_info(), start_capture()/stop_capture() for Metal debugging. The vLLM-Metal project demonstrates "true zero-copy operations through Apple Silicon unified memory architecture" for inference. Custom Metal kernels receive MTLBuffer pointers to MLX array data directly.
**Confidence**: high
**Source**: MLX Documentation: Metal module
**Evidence**: MLX documentation for mlx.core.metal module and mlx.core.fast.metal_kernel. GitHub issue ml-explore/mlx#1159 discusses DLPack device support. vLLM-Metal project states zero-copy through unified memory.
**Tags**: MLX,Metal,interop,zero-copy,DLPack,custom-kernel

## Finding 302: MLX stream model: operations target specific devices via stream parameter (mx.gpu, mx.cpu). Stream struct = {Device device, int index}. Metal backend maps stream index to DeviceStream (wraps MTL::CommandQueue). Multiple streams enable pipelined execution and overlapping command submission.
**Confidence**: verified
**Source**: high
**Evidence**: CPU-GPU concurrency: unified memory means CPU and GPU can work on same data simultaneously without explicit transfers. mx.async_eval() pipelines graph construction with computation. Limitation: concurrent multi-model inference from separate threads crashes due to thread-safety issues in Metal backend (Issue #3078). Multi-stream concurrent inference still evolving.
**Tags**: 

## Finding 305: MLX continuous batching scales to 4.3x aggregate throughput at 16 concurrent requests (vllm-mlx). Single-stream: Qwen3-0.6B 525.5 tok/s, at 16 concurrent: 1,642 tok/s. Scheduler dynamically groups requests, new requests join at token boundaries, completed requests exit without blocking others.
**Confidence**: verified
**Source**: high
**Evidence**: vllm-mlx achieves 21-87% higher throughput than llama.cpp due to: zero-copy tensor ops (unified memory), lazy evaluation (operation fusion), native quantization (efficient dequantization kernels). Text prefix caching: 5.8x TTFT speedup via SHA-256 hashing of prompt tokens. Multimodal caching: 28x speedup on repeated image queries.
**Tags**: 

## Finding 355: vllm-mlx: native Apple Silicon GPU acceleration for vLLM using MLX + mlx-lm. OpenAI-compatible API. Continuous batching, prefix caching, multimodal support. M4 Max throughput: Qwen3-0.6B 525.5 tok/s single-stream, 1642 tok/s at 16 concurrent (3.7x scaling).
**Confidence**: verified
**Source**: high
**Evidence**: vs llama.cpp: 21-87% higher throughput (llama.cpp processes sequentially, no continuous batching). vs mlx-lm: adds continuous batching + OpenAI API + multimodal caching. vs vLLM-metal: adds content-based vision caching. Memory: LRU eviction with configurable limits (default 512MB). All benchmarks use 4-bit quantization.
**Tags**: 

