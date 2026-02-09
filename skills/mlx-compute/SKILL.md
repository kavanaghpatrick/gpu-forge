---
name: mlx-compute
layer: 3
description: >
  This skill should be used when the user asks about MLX framework programming, custom Metal kernels via mx.fast.metal_kernel(), lazy evaluation, MLX streams, mx.compile kernel fusion, mx.array operations, MLX distributed computing, MLX memory management, MLX Swift API, MLX quantization, or unified memory exploitation through MLX. Trigger keywords: "MLX", "mx.fast", "metal_kernel", "lazy evaluation", "streams", "mx.compile", "custom kernel", "MLX distributed", "mx.array", "unified memory MLX", "mx.fast.metal_kernel", "mlx-lm", "mx.eval", "mx.quantize", "Steel GEMM", "MLX backend", "vllm-mlx", "cuda_kernel", "mx.fast.rms_norm", "bfloat16 MLX".
related_skills:
  - metal-compute
  - msl-kernels
  - gpu-perf
  - gpu-distributed
  - unified-memory
---

# MLX Compute

MLX is Apple's open-source machine learning framework designed for Apple Silicon GPU-centric computing. Unlike PyTorch or TensorFlow, MLX is built from the ground up for unified memory architecture — arrays live in shared memory accessible by both CPU and GPU without explicit transfers. This eliminates the host-device copy overhead that dominates GPU programming on discrete GPU systems.

MLX uses lazy evaluation: operations build a computation graph (DAG) rather than executing immediately. When results are needed, `mx.eval()` triggers a two-phase evaluation — Phase 1 walks the graph backward via DFS to find all needed computations, Phase 2 evaluates in topological order with automatic memory reuse through reference counting. The `mx.compile()` function traces these graphs and fuses multiple GPU kernel launches into single kernels, reducing Metal command buffer overhead.

The framework provides `mx.fast.metal_kernel()` for writing custom Metal shaders that integrate directly into the MLX computation graph. This enables hand-optimized GPU kernels for operations not covered by built-in primitives. MLX also supports distributed computing via MPI, Ring (TCP), and JACCL (RDMA over Thunderbolt 5) backends, enabling multi-Mac GPU clusters.

MLX's Metal backend uses a sophisticated kernel library with ~24 primary .metal files, pre-compiled .metallib delivery, and JIT compilation for custom kernels. The Steel GEMM kernels use `simdgroup_multiply_accumulate` with 6 block configurations for matrix multiplication. Memory management uses a caching allocator with page-aligned rounding, residency sets, and automatic buffer lifecycle management.

## Domain Overview

**Lazy Evaluation & Compilation**: MLX arrays are DAG nodes storing shape, dtype, primitive reference, and input arrays. Operations are not executed until `mx.eval()` is called. The evaluation scheduler performs DFS dependency analysis to compute minimal required subgraph (#300). `mx.compile()` fuses GPU kernel launches in three phases: trace, optimize, emit (#297). Recompilation is triggered when input shapes/dims change, dtypes change, or the number of inputs changes (#299). Accidental type upcasting (e.g., `fp16_array * mx.array(2.0)`) silently converts to float32, degrading performance (#407).

**Custom Metal Kernels**: `mx.fast.metal_kernel()` accepts name, input_names, output_names, and MSL source code (#293). Execution takes inputs, template type parameters, grid/threadgroup dimensions, and output shapes/dtypes (#295). Setting `atomic_outputs=True` enables concurrent thread writes, but `init_value` only works for specific types (#404). Every new `metal_kernel()` call JIT-compiles a new Metal library — cache kernel objects to avoid recompilation overhead (#405).

**Metal Backend Internals**: Each MLX Primitive subclass implements `eval_gpu()` containing Metal kernel dispatch logic (#288, #394). Binary operations use a `BINARY_GPU` macro delegating to `binary_op_gpu()` with kernel naming conventions like `b<op><type>` (#395). Command buffer batching uses device-tier-specific thresholds (#396). Kernel delivery uses dual modes: pre-compiled .metallib and JIT-compiled from source (#289, #402). The MetalAllocator uses page-boundary rounding for large buffers and a caching strategy for reuse (#398). Residency sets manage GPU memory visibility (#403).

**Streams & Concurrency**: Operations target specific devices via stream parameter — `mx.gpu` or `mx.cpu` (#302). Streams enable concurrent CPU and GPU execution. Continuous batching for LLM serving scales to 4.3x aggregate throughput at 16 concurrent requests (#305).

**Distributed Computing**: MLX supports 4 backends: MPI (full-featured), Ring (TCP socket-based), and JACCL (RDMA over Thunderbolt 5, available since macOS 26.2) (#312, #317). Data parallelism uses `mlx.nn.average_gradients()` for gradient aggregation (#319).

**Advanced Patterns**: Quantization via `mx.quantize(w, bits, group_size)` supports 2/3/4/5/6/8 bits (#342). NumPy interop via `np.array(mx_arr)` for copy or `np.array(mx_arr, copy=False)` for zero-copy (#345). Custom C++ primitives inherit from Primitive base class (#348). `mx.fast` optimized operations include `rms_norm`, `layer_norm`, `rope`, `scaled_dot_product_attention` (#351). Supported dtypes: float16, bfloat16, float32, complex64, plus integer types (#378).

**General-Purpose Compute**: Full FFT suite (1D, 2D, nD, real FFT) on GPU (#331, #408). Sorting achieves exceptional performance beating tested CUDA GPUs (#409). Layout optimization: `x @ W.T` faster than `x @ W` for vector-matrix (#410). CUDA backend support added July 2025 for cross-platform code (#335, #411).

**M5 Neural Accelerator**: M5 GPU has dedicated Neural Accelerators (TensorOps) for matrix multiplication, providing 4x AI compute vs M4 (#307).

## Key Knowledge Areas

The knowledge base contains **49 findings** covering:

- **Metal Backend Internals** (18 findings): eval_gpu() dispatch, BINARY_GPU macro, command buffer batching, kernel delivery modes, MetalAllocator caching, residency sets, eval_impl() two-phase evaluation, memory reuse, kernel library structure, Steel GEMM
- **Custom Metal Kernel API** (4 findings): mx.fast.metal_kernel() signature, execution parameters, atomic outputs, JIT compilation caching
- **Compilation & Optimization** (5 findings): mx.compile() fusion, recompilation triggers, lazy evaluation scheduler, dynamic broadcasting, type upcasting traps
- **Stream & Concurrency** (2 findings): Stream model, continuous batching throughput
- **Distributed Computing** (3 findings): Backend types, JACCL RDMA, data parallelism patterns
- **Advanced Patterns** (6 findings): Quantization, NumPy interop, C++ primitives, mx.fast operations, supported dtypes, ASTC texture trick
- **General-Purpose Compute** (6 findings): FFT, sorting, layout optimization, CUDA backend, comprehensive compute ops
- **M5 Neural Accelerator** (1 finding): TensorOps in GPU cores
- **MLX Swift API** (2 findings): Swift Package Manager, MLX-Outil tool calling
- **Metal Interop** (1 finding): Metal buffer backing, interop paths
- **vllm-mlx** (1 finding): Native Apple Silicon GPU acceleration for vLLM

## How to Query

Use the knowledge base CLI to retrieve MLX findings:

```bash
# Get all mlx-compute findings
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill mlx-compute

# Search by MLX topic
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "metal_kernel"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "mx.compile"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "lazy evaluation"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "MLX distributed"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "Steel GEMM"

# Get specific finding details
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <id>

# Search across all skills
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "MLX memory"
```

Query strategies by problem type:
- **Custom kernels**: Search "metal_kernel", "atomic_outputs", "JIT compile"
- **Performance**: Search "mx.compile", "kernel fusion", "type upcasting", "layout"
- **Memory**: Search "MetalAllocator", "residency", "buffer caching", "unified memory"
- **Distributed**: Search "MPI", "JACCL", "Ring", "average_gradients"
- **Backend details**: Search "eval_gpu", "command buffer", "Steel GEMM", "kernel library"
- **Quantization**: Search "mx.quantize", "bits", "group_size"

## Common Patterns & Quick Answers

**Q: How do I write a custom Metal kernel in MLX?**
A: Use `mx.fast.metal_kernel(name, input_names, output_names, source)` (#293). Pass inputs as list of mx.array, template for type params, grid/threadgroup dims, output_shapes and output_dtypes (#295). Cache the kernel object — each new `metal_kernel()` call JIT-compiles a new Metal library (#405). For concurrent writes, set `atomic_outputs=True` (#404).

**Q: How does mx.compile() improve performance?**
A: `mx.compile()` fuses multiple GPU kernel launches into a single kernel through three phases: trace, optimize, emit (#297). It eliminates intermediate buffer allocations and reduces Metal command buffer overhead. Recompilation triggers: input shapes/dims change, dtypes change, number of inputs changes (#299). Dynamic broadcasting is supported via BroadcastDynamic for shapeless compile/export (#406).

**Q: How does lazy evaluation work in MLX?**
A: MLX arrays are DAG nodes; operations build a graph rather than executing immediately. When `mx.eval()` is called, Phase 1 (DFS) walks the graph backward to find all needed computations, Phase 2 evaluates in topological order (#300, #397). Memory reuse happens automatically through reference counting and detach operations (#399). Command buffer batching uses device-tier-specific thresholds (#396).

**Q: How do I avoid common MLX performance traps?**
A: Watch for accidental type upcasting — `my_fp16_array * mx.array(2.0)` silently converts to float32 (#407). Use `x @ W.T` instead of `x @ W` for vector-matrix operations (#410). Use `mx.fast` optimized operations (rms_norm, layer_norm, rope, sdpa) instead of manual implementations (#351). Cache metal_kernel objects to avoid JIT recompilation (#405).

**Q: How do I use MLX for distributed GPU computing?**
A: MLX supports MPI (full-featured), Ring (TCP socket-based), and JACCL (RDMA over Thunderbolt 5, macOS 26.2+) backends (#312, #317). Use `mlx.nn.average_gradients()` for efficient gradient aggregation in data parallelism (#319). JACCL enables direct GPU-to-GPU memory access across Macs.

**Q: How does MLX interact with Metal buffers?**
A: MLX arrays are backed by Metal buffers on Apple Silicon. Interop paths: mx.fast.metal_kernel() for custom GPU code, or direct buffer access for Metal integration (#95). The MetalAllocator uses page-boundary rounding and a caching strategy for buffer reuse (#398). Residency sets manage GPU memory visibility for non-heap buffers (#403).

**Q: What general-purpose compute does MLX support?**
A: Full FFT suite (1D, 2D, nD, real/inverse), linalg, random, sorting, scatter/gather (#331, #408). MLX sort beats tested CUDA GPUs in performance (#409). Comprehensive compute: fft, linalg, random, sorting, scan, cumsum, conv, pool, scatter/gather, topk (#408). CUDA backend enables cross-platform code (#335, #411).

**Q: How does MLX handle quantization?**
A: `mx.quantize(w, bits, group_size)` supports 2/3/4/5/6/8 bit quantization with configurable group sizes (#342). MLX-LM provides high-level APIs for model quantization and serving. Quantized models use specialized dequantization kernels fused with matrix operations.

**Q: What about MLX on M5 with Neural Accelerators?**
A: M5 GPU has dedicated Neural Accelerators (TensorOps) in each GPU core for matrix multiplication, providing 4x AI compute vs M4 (#307). MLX operations like matmul can potentially leverage TensorOps for acceleration. This is a fundamental hardware upgrade for ML workloads.

## Cross-References

**Related Skills**:
- **metal-compute** (Layer 1): Metal API fundamentals that MLX builds upon — command buffers, compute pipelines, buffer management
- **msl-kernels** (Layer 1): Metal Shading Language syntax needed for mx.fast.metal_kernel() custom kernels
- **gpu-perf** (Layer 2): Performance optimization patterns, profiling tools, occupancy tuning for MLX Metal kernels
- **gpu-distributed** (Layer 3): RDMA over Thunderbolt 5, multi-Mac cluster setup that MLX distributed backends leverage
- **unified-memory** (Layer 0): UMA architecture that enables MLX's zero-copy CPU/GPU memory model

**MLX Development Workflow**:
1. Prototype with built-in MLX operations and mx.compile() (this skill)
2. Profile with Metal System Trace to identify bottlenecks (gpu-perf)
3. Write custom Metal kernels via mx.fast.metal_kernel() for hot paths (this skill + msl-kernels)
4. Optimize kernel occupancy and memory access patterns (gpu-perf + unified-memory)
5. Scale across multiple Macs via MLX distributed backends (gpu-distributed)

**Common Integration Paths**:
- **MLX + Custom Kernels** → msl-kernels for MSL syntax, gpu-perf for optimization
- **MLX + Metal Direct** → metal-compute for pipeline setup, unified-memory for buffer sharing
- **MLX + Multi-Mac** → gpu-distributed for RDMA/Thunderbolt, this skill for MLX distributed API
- **MLX + M5 Hardware** → gpu-silicon for TensorOps architecture, this skill for MLX exploitation

## Investigation Prompts

Suggested investigation topics to expand this skill:

- "MLX Steel GEMM kernel implementation details and block configuration tuning"
- "mx.compile trace-optimize-emit pipeline internals and fusion heuristics"
- "MLX MetalAllocator buffer pool sizing and cache eviction policy"
- "JACCL RDMA backend performance characteristics vs MPI for MLX distributed"
- "MLX custom kernel performance comparison: metal_kernel vs native primitive"
- "M5 TensorOps integration in MLX: automatic vs explicit exploitation"
- "MLX continuous batching implementation for production LLM serving"
- "MLX CUDA backend parity: which operations differ between Metal and CUDA paths"
- "MLX memory pressure handling and eviction strategies on memory-constrained devices"
- "MLX quantization kernel fusion: dequantize-matmul patterns and performance"

## Notes

- **Layer 3 dependencies**: Requires metal-compute (Metal API), msl-kernels (MSL syntax), and builds on gpu-perf, gpu-distributed, unified-memory
- **Lazy evaluation gotcha**: Operations don't execute until mx.eval() — print(array) triggers eval, which can cause unexpected synchronization
- **Type upcasting trap**: Mixing float16 arrays with float32 scalars silently upcasts — use typed scalars or explicit casting
- **Kernel caching**: Each mx.fast.metal_kernel() call JIT-compiles — store kernel objects as module-level variables
- **CUDA portability**: Since July 2025, MLX code can run on NVIDIA GPUs via cuda_kernel alongside metal_kernel
- **M5 acceleration**: Neural Accelerators in M5 GPU cores provide 4x AI compute — MLX operations may automatically benefit
- **Distributed backends**: JACCL (Thunderbolt 5 RDMA) available since macOS 26.2; MPI requires separate installation
- **MLX Swift**: Full ML capabilities available via Swift Package Manager for iOS, macOS, visionOS integration
- **Production serving**: vllm-mlx provides native Apple Silicon GPU acceleration for vLLM workloads
