# simd-wave — All Findings (56)

## Finding 392: Quadgroup operations (quad_shuffle, quad_broadcast) operate on 4-thread subsets for 2D image processing — a SIMD-group function with execution width of 4, useful for reducing texture reads
**Confidence**: verified
**Source**: Discover advances in Metal for A15 Bionic
**Evidence**: A quad-group function is a SIMD-group function with execution width of 4. Introduced with A15 Bionic. Use case: reducing texture reads in fragment shaders. Each quad corresponds to a 2x2 pixel block. Quad operations include quad_shuffle, quad_broadcast_first, quad_sum, quad_min, quad_max. Available on all Apple GPUs from A13/M1 onwards.
**Tags**: quadgroup,quad_shuffle,2D-patterns,A15,texture

## Finding 393: Cross-SIMD-group communication requires threadgroup memory + threadgroup_barrier; raking pattern writes per-SIMD results then single SIMD group reduces them
**Confidence**: verified
**Source**: Advanced Metal Shader Optimization - WWDC16
**Evidence**: Cross-SIMD communication pattern: each SIMD group computes partial result (e.g., via simd_sum), writes to threadgroup memory indexed by simdgroup_index, issues threadgroup_barrier(mem_threadgroup), then designated SIMD group reads all partial results and computes final value. This raking strategy minimizes barriers and threadgroup memory usage.
**Tags**: cross-SIMD,threadgroup-memory,raking,barrier

## Finding 382: Apple GPUs use register-based execution mask stack (r0l) for divergence — each thread tracks nesting depth as 16-bit counter, unlike NVIDIA's centralized hardware reconvergence stack
**Confidence**: verified
**Source**: Apple G13 GPU Architecture Reference
**Evidence**: The Apple G13 GPU uses r0l (low 16 bits of register r0) to track execution mask stack. Value in r0l indicates how many pop operations needed to re-enable inactive thread, or zero if active. Instructions: pop_exec, if_*cmp, else_*cmp, while_*cmp. Contrasts with NVIDIA's (PC, active_mask, reconvergence_PC) tuple stack.
**Tags**: divergence,predication,execution-mask,r0l,ISA

## Finding 383: Apple GPU if_icmp/if_fcmp instructions increment r0l for failing threads; only threads with r0l==0 remain active — pure predication without branch prediction
**Confidence**: verified
**Source**: Apple G13 GPU Architecture Reference
**Evidence**: if_icmp/if_fcmp compare operands with condition code. Parameter n specifies nesting depth. Failing threads get r0l incremented. pop_exec decrements r0l by n (clamping to zero). else_icmp handles else-branch. jmp_exec_any/jmp_exec_none allow branching based on active thread count.
**Tags**: divergence,if_icmp,pop_exec,flow-control

## Finding 384: FCMPSEL (conditional select) executes at 1-cycle throughput for both 16-bit and 32-bit, making branchless select nearly free — preferred over divergent branches for short conditionals
**Confidence**: verified
**Source**: metal-benchmarks: Apple GPU microarchitecture
**Evidence**: metal-benchmarks: FCMPSEL16 has 1 cycle throughput ~2.17 cycle latency; FCMPSEL32 has 1 cycle throughput ~4.74 cycle latency. Performs compare-and-select in single operation. Metal compiler emits FCMPSEL for ternary expressions. For 1-3 instruction conditionals, avoids if_icmp/pop_exec overhead entirely.
**Tags**: FCMPSEL,branchless,select,ALU-pipeline

## Finding 385: Xcode 15 Thread Divergence Heat Maps for M3/A17 Pro visualize per-SIMD-group divergence, enabling developers to identify hotspots and inspect full execution history
**Confidence**: verified
**Source**: Analyzing Apple GPU performance with performance heat maps
**Evidence**: Performance heat maps visualize GPU thread divergence in SIMD groups. Developers can select a SIMD group to view entire execution history including function calls, loop iterations, and active threads. Instruction Count Heat Map shows exactly how many instructions executed per pixel/SIMD group.
**Tags**: divergence,profiling,Xcode,heat-map,M3

## Finding 381: Metal 4 introduces cooperative tensor types that extend simdgroup_matrix for SIMD-wide matrix multiplication. Uses Metal Performance Primitives (MPP) with tensor_ops::matmul2d for inline shader ML inference.
**Confidence**: verified
**Source**: high
**Evidence**: Metal 4 (MSL 4.0) adds #include <metal_tensor> and #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>. Tensors are declared as tensor<device half, dextents<int, 2>> in shader arguments. Matrix multiply uses: constexpr tensor_ops::matmul2d_descriptor desc(M, N, K, leftTranspose, rightTranspose, reducedPrecision); tensor_ops::matmul2d<desc, execution_thread> op; op.run(inputTensor, weights, output). The inner-product dimension K must be a multiple of 32, otherwise use dynamic_length_v<int> for correctness. Three execution groups: execution_thread (divergent), simdgroup (uniform within SIMD), threadgroup (uniform within threadgroup).
**Tags**: 

## Finding 390: Metal 4 cooperative tensors use execution_simdgroups<N> to distribute matmul work across N SIMD groups within a threadgroup — demonstrated with matmul2d and native tensor types
**Confidence**: verified
**Source**: Example of using tensor op for matmul in Metal 4
**Evidence**: liuliu/example_matmul_metal4 shader code: matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp. Uses tensor<device half, dextents<int32_t, 2>, tensor_inline> for native tensor types. matmul2d_descriptor(M, N, K, ...) configures tile sizes. Includes multiply and multiply_accumulate modes. Headers: metal_tensor and MetalPerformancePrimitives.h.
**Tags**: cooperative-tensors,Metal-4,matmul2d,execution_simdgroups,tensor

## Finding 391: Metal 4 tensor matmul optimal tile size is 128x64x64 with 4 SIMD groups; static slicing outperforms dynamic; fully unrolling K dimension is important for performance
**Confidence**: verified
**Source**: Example of using tensor op for matmul in Metal 4 - WORKLOG
**Evidence**: WORKLOG.md from example_matmul_metal4: (1) static slice faster than dynamic slice with cooperative tensor accumulate, (2) dynamic extents faster than static extents unexpectedly, (3) 128x64x64 with 4 simdgroups ideal for Neural Accelerators, (4) fully unroll K important, (5) Split K solves stream-K issues.
**Tags**: cooperative-tensors,Metal-4,tile-size,performance,matmul

## Finding 328: Multi-level SIMD reduction pattern: (1) each thread reduces its local values, (2) simd_sum/simd_max across 32-thread SIMD group, (3) write per-SIMD results to threadgroup memory, (4) final SIMD group reduces across all SIMD groups with another simd_sum
**Confidence**: high
**Source**: high
**Evidence**: The optimum Metal reduction pattern uses a raking strategy: each SIMD group reads a subregion and computes its sum using simd_sum. Results are written to threadgroup memory. The last SIMD group then executes simd_sum again for the final result. This approach decreases the number of threadgroup barriers and threadgroup memory usage compared to tree-based reductions. The raking factor (values per thread) equals threads_per_threadgroup / execution_width.
**Tags**: 

## Finding 332: Metal built-in SIMD reductions: simd_sum, simd_product, simd_min/simd_minimum, simd_max/simd_maximum, simd_or, simd_and, simd_xor - all operate across 32 active lanes and broadcast result
**Confidence**: verified
**Source**: high
**Evidence**: Metal provides built-in reductions: simd_sum/simd_product for arithmetic, simd_minimum/simd_maximum for comparisons, and simd_or/simd_and/simd_xor for bitwise operations (integer types only). These are hardware-accelerated single-instruction reductions across all active threads in the SIMD group. The result is broadcast to all threads. Manual shuffle-based reductions using simd_shuffle_down with offsets 16,8,4,2,1 can achieve equivalent results but built-ins are preferred for clarity and potential hardware optimization.
**Tags**: 

## Finding 336: For threadgroups fitting within a single SIMD (<=32 threads), use simdgroup_barrier instead of threadgroup_barrier - it is significantly faster and avoids threadgroup memory overhead entirely
**Confidence**: verified
**Source**: high
**Evidence**: If your thread group fits within a single SIMD, the regular thread group barrier function is unnecessary. Using the SIMD group barrier function is often faster than trying to use a larger thread group with thread group barriers. This is because simdgroup_barrier is essentially free (threads already execute in lockstep) while threadgroup_barrier requires actual hardware synchronization. For reductions of 32 or fewer elements, a pure SIMD approach with no threadgroup memory is optimal.
**Tags**: 

## Finding 340: Atomic-free reduction technique: each threadgroup reduces to a single value, writes to output buffer indexed by threadgroup_position_in_grid, then a second dispatch reduces the partial results
**Confidence**: high
**Source**: high
**Evidence**: The atomic-free approach avoids contention: Phase 1 dispatches N/threadgroup_size threadgroups, each producing one partial result via simd_sum cascade. Phase 2 dispatches a single threadgroup to reduce partial results. This two-pass approach avoids atomic operations entirely and achieves near-peak memory bandwidth. On M1, the raking reduction achieved ~18 GB/sec at 128M elements.
**Tags**: 

## Finding 344: Metal provides built-in SIMD prefix scan functions: simd_prefix_inclusive_sum(T data), simd_prefix_exclusive_sum(T data), and corresponding product variants - these operate within the 32-thread SIMD group
**Confidence**: verified
**Source**: high
**Evidence**: Metal offers hardware-accelerated prefix scan within SIMD groups. simd_prefix_inclusive_sum computes running sum including the current thread value. simd_prefix_exclusive_sum computes running sum excluding the current thread (shifted right by one with identity element). These are single-instruction operations on Apple GPUs, much faster than manual shuffle-based implementations. For scans beyond 32 elements, a multi-level approach combining SIMD scan + threadgroup memory + device-level scan is required.
**Tags**: 

## Finding 349: Three main prefix sum algorithms for Metal: Hillis-Steele (step-efficient, maps well to SIMD execution), Blelloch (work-efficient, two-phase up-sweep/down-sweep), and raking (hybrid with serial accumulation per thread)
**Confidence**: high
**Source**: high
**Evidence**: Hillis-Steele is based on the Kogge-Stone adder and maps naturally onto 32-wide SIMD groups due to its step-efficiency - each step halves the stride. Blelloch scan uses an up-sweep (tree reduction building partial sums) followed by a down-sweep (converting sums to prefix scan). The raking approach combines serial accumulation per thread with SIMD-level scan. On M1 at 128M elements, both raking and Blelloch achieve ~18 GB/sec, compared to 22 GB/sec for vectorized CPU code. For SIMD-level scan, Hillis-Steele is preferred due to lower latency.
**Tags**: 

## Finding 353: Multi-level scan pattern for large arrays: (1) SIMD-level scan within each 32-thread group, (2) write per-SIMD totals to threadgroup memory, (3) scan the totals, (4) add back scanned totals to each SIMD block, (5) repeat at device level across threadgroups
**Confidence**: high
**Source**: high
**Evidence**: For arrays larger than 32 elements, the scan must cascade across levels. Within a threadgroup of 1024 threads: each of 32 SIMD groups performs simd_prefix_inclusive_sum on 32 elements. The 32 SIMD totals go to threadgroup memory. A single SIMD scans these 32 totals. Each SIMD group adds the appropriate prefix to its local scan. For device-level scans beyond one threadgroup, a third level captures per-threadgroup totals and scans them in a subsequent dispatch, then adds back to all elements. This enables scanning millions of elements.
**Tags**: 

## Finding 357: Prefix sum applications on GPU: stream compaction (filter elements satisfying predicate), histogram building, radix sort (counting sort per digit), parallel allocation, and sparse matrix construction
**Confidence**: high
**Source**: high
**Evidence**: Stream compaction: compute predicate mask, prefix sum of mask gives output indices, scatter matching elements. For radix sort: count occurrences of each digit value (histogram), prefix sum of histogram gives scatter offsets, reorder elements. The enumerate operation in radix sort is equivalent to stream compaction. Onesweep radix sort uses decoupled look-back to combine prefix scan and reordering in a single pass, reducing global memory operations. These patterns all require efficient multi-level prefix sum as the foundational primitive.
**Tags**: 

## Finding 310: Apple GPU SIMD width is 32 threads per simdgroup, with 6 core shuffle variants: simd_shuffle(data,lane), simd_shuffle_xor(data,mask), simd_shuffle_up(data,delta), simd_shuffle_down(data,delta), simd_shuffle_and_fill_up(data,fill,delta), simd_shuffle_and_fill_down(data,fill,delta)
**Confidence**: verified
**Source**: high
**Evidence**: The G13 architecture has 32 threads per SIMD-group. simd_shuffle uses a 6-byte encoding with operands A (source value), B (shuffle index), and destination D. The shuffle index in B must be less than 32. Shuffle operations work on a per-quad basis internally - quad values are computed by ORing B across all threads in a quad. The shuffle_and_fill variants (introduced with A15 Bionic) add a fill buffer to populate lanes that would otherwise receive undefined values during directional shifts. All variants also have quad_* equivalents for 4-thread quad groups.
**Tags**: 

## Finding 314: Apple GPU has 256 bytes/cycle SIMD shuffle bandwidth per GPU core - 2x more than AMD RDNA or NVIDIA, representing industry-leading shuffle performance
**Confidence**: verified
**Source**: high
**Evidence**: Each GPU core sustains 256 bytes of SIMD shuffle bandwidth per cycle, significantly exceeding the 128 bytes typical on competing architectures (AMD RDNA, NVIDIA). Apple invested in industry-leading SIMD shuffle bandwidth and matrix instructions as a strategy to reduce the amount of threadgroup memory bandwidth needed. This 2x advantage enables efficient thread-level communication without resorting to shared memory.
**Tags**: 

## Finding 318: simd_shuffle_and_fill_up/down have an optional modulo argument that splits the SIMD group into smaller virtual vectors - e.g. modulo=8 splits 32-wide SIMD into four 8-wide vectors
**Confidence**: verified
**Source**: high
**Evidence**: The shuffle and fill instructions provide a fill buffer to update lanes that would otherwise receive unshifted values. With a modulo of eight, the SIMD group is effectively split into four vectors. This is critical for sliding-window operations like convolution - a 5x5 convolution kernel can reduce texture samples by 84% by shuffling sampled texel values from adjacent lanes as the window slides. These were introduced with A15 Bionic (Apple family 8).
**Tags**: 

## Finding 320: simd_shuffle_xor enables butterfly communication patterns for parallel reductions and FFT - each thread swaps values with another thread whose lane ID differs by XOR with the given offset
**Confidence**: high
**Source**: high
**Evidence**: shuffle_xor creates a butterfly communication pattern where threads exchange data based on XOR patterns of their indices. For all-reduce within a SIMD group, loop with decreasing offsets from simdWidth/2 down to 1: val = op(val, simd_shuffle_xor(val, offset)). This gives O(log n) complexity. The pattern is fundamental to FFT, bitonic sort, and parallel reductions. Unlike shared memory, shuffle operations achieve near-zero latency data exchange using the SIMD groups simultaneous execution.
**Tags**: 

## Finding 323: simd_shuffle can read from inactive (masked-out) threads - important for divergent code paths where you still need data from disabled lanes
**Confidence**: verified
**Source**: medium
**Evidence**: Execution masking generally prevents reading or writing values from inactive threads, however SIMD shuffle instructions can read from inactive threads in some cases. This is architecturally significant because it means shuffle operations can access data across branch boundaries, enabling communication patterns that would otherwise require shared memory synchronization.
**Tags**: 

## Finding 386: Metal Performance Shaders lacks reduce, scan, and radix sort primitives — developers must implement custom GPU sorting
**Confidence**: high
**Source**: Missing Reduce, Scan, Radix Sort? - Apple Developer Forums
**Evidence**: Apple Developer Forums thread highlights MPS missing fundamental parallel computing primitives that are standard in CUDA (CUB library). As of 2025, Apple still has not added these. MPS focuses on neural networks, image processing, and ray tracing.
**Tags**: MPS,radix-sort,reduce,scan,missing-primitives

## Finding 387: Apple Silicon GPUs lack forward progress guarantees between threadgroups — OneSweep radix sort deadlocks; DeviceRadixSort (reduce-then-scan) is safe
**Confidence**: verified
**Source**: GPUSorting - State of the art GPU sorting algorithms
**Evidence**: GPUSorting repo warns OneSweep tends to run on anything not mobile or Apple. OneSweep relies on chained-scan-with-decoupled-lookback requiring forward progress. Apple's GPU scheduler can deprioritize threadgroups indefinitely. DeviceRadixSort uses reduce-then-scan with independent dispatches — no inter-threadgroup sync needed.
**Tags**: forward-progress,OneSweep,DeviceRadixSort,deadlock,radix-sort

## Finding 388: Metal-native radix sort achieves ~3 billion elements/sec on M1 Max vs ~1 billion for WebGPU equivalent — 3x speedup from simdgroup operations
**Confidence**: high
**Source**: Sorting - Linebender GPU wiki
**Evidence**: Linebender GPU sorting wiki: WebGPU hybrid algorithm achieved ~1B elements/sec on M1 Max. Metal experiments on same hardware reached ~3B elements/sec using actual subgroup ballot operations and simdgroup_barrier. Uses 4-bit digits with tree reduction for histogram scan, patterned after AMD FidelityFX.
**Tags**: radix-sort,Metal,WebGPU,M1-Max,benchmark

## Finding 389: Bitonic sort within 32-thread SIMD group uses simd_shuffle for register-level compare-and-swap — 5 stages, no threadgroup memory needed, completable in ~15-20 cycles
**Confidence**: verified
**Source**: metal-benchmarks: Apple GPU microarchitecture
**Evidence**: For 32-element sort: 5 stages (log2(32)=5) of bitonic merge. Each thread holds one element, computes partner via XOR of lane index with step distance, shuffles to get partner value, keeps min or max. At Apple's 256 B/cycle shuffle bandwidth, each step completes in under 1 cycle.
**Tags**: bitonic-sort,simd-shuffle,register-sort,SIMD-group

## Finding 479: Cypress introduces task-based programming model with sequential semantics for tensor core programming. Tasks operate on tensors without communication or synchronization. Compiler automates data movement and async computation. Achieves 0.88x-1.06x cuBLAS on GEMM and 0.80x-0.98x FlashAttention while eliminating explicit data movement from application code.
**Confidence**: high
**Source**: Task-Based Tensor Computations on Modern GPUs
**Evidence**: Programs are connected to target machines via mapping specifications defining task placement and tensor memory materialization. The compiler transforms Cypress programs into competitive CUDA implementations. Key insight: separating WHAT to compute (tasks on tensors) from HOW to execute (mapping to hardware) enables portable high-performance tensor code.
**Tags**: tensor-core,programming-model,abstraction,warp-specialization,compiler,async

## Finding 480: ThunderKittens establishes 16x16 matrix tiles as the universal GPU programming primitive across three hierarchy levels: (1) warp-level 16x16 tiles with PyTorch-like ops, (2) thread-block-level templates for async overlap, (3) grid-level memory optimization. Matches cuBLAS/FlashAttention-3 on GEMM+attention, outperforms by 10-40% on attention backwards, 8x on state space models, 14x on linear attention.
**Confidence**: high
**Source**: ThunderKittens: Simple, Fast, and Adorable AI Kernels
**Evidence**: Framework from Stanford (Spector, Arora, Singhal, Fu, Re). Key insight: the 16x16 tile is the natural unit of GPU computation because it maps to register files, shared memory banks, and tensor core operand sizes simultaneously. By making tiles first-class, both the programmer and compiler reason at the right abstraction level.
**Tags**: tensor-core,tile-programming,16x16,warp-level,abstraction,framework

## Finding 485: TileLang decouples scheduling space (thread binding, layout, tensorize, pipeline) from dataflow using customization annotations and primitives. Core patterns: GEMM, COPY, ATOMIC, REDUCE expressed as tile operators. Achieves 1.36x over FlashAttention-3, 1.41x over Triton, 1.70x over PyTorch on attention workloads. Works across hardware platforms via unified block-and-thread paradigm.
**Confidence**: high
**Source**: TileLang: A Composable Tiled Programming Model for AI Systems
**Evidence**: From Microsoft Research + Peking University. Key insight: separating WHAT tiles do (dataflow via tile operators) from HOW they execute (scheduling via annotations) creates a composable system where each dimension of optimization can be tuned independently. The tile operator vocabulary (GEMM, COPY, ATOMIC, REDUCE) captures nearly all AI kernel patterns.
**Tags**: tile-programming,scheduling,dataflow,composable,cross-platform,compiler

## Finding 481: HipKittens ports ThunderKittens tile abstractions from NVIDIA to AMD GPUs. Core finding: tile and bulk compute interfaces carry over across vendors, BUT memory access patterns, compute/memory scheduling, and thread block ordering within chiplet architecture differ fundamentally. Competes with AMD hand-optimized assembly for GEMMs and attention. Outperforms compiler baselines by 1.2-2.4x.
**Confidence**: high
**Source**: HipKittens: Fast and Furious AMD Kernels
**Evidence**: Validated on CDNA3 and CDNA4 platforms. Key insight: the tile abstraction IS portable across GPU vendors, but the scheduling and memory access strategies are NOT. This means a cross-vendor tensor programming model needs to separate the tile computation semantics from the vendor-specific scheduling policies.
**Tags**: cross-vendor,AMD,tile-programming,portability,tensor-core,CDNA

## Finding 593: Libra: systematic framework for synergistic CUDA core + tensor core computation in sparse matrix multiply. 2D-aware workload distribution finds optimal task mapping between tensor cores (high throughput, structured) and CUDA/shader cores (flexible, unstructured). Hybrid load balancing + occupancy-aware scheduling. 1.77x over FlashSparse, 2.9x over DGL, outperforms 12 baselines on H100/RTX 4090.
**Confidence**: high
**Source**: Libra: Unleashing GPU Heterogeneity for High-Performance Sparse Matrix Multiplication
**Evidence**: Key insight: sparse workloads have heterogeneous tile densities — some tiles are dense enough for tensor cores, others are too sparse. Instead of choosing one path, Libra assigns tiles to the APPROPRIATE core type within the SAME kernel, achieving cooperative heterogeneous execution.
**Tags**: heterogeneous-cores,sparse-matrix,workload-distribution,tensor-core,CUDA-core,cooperative

## Finding 486: Tilus is a tile-level GPGPU programming language supporting arbitrary low-precision data types from 1 to 8 bits. Compiles through automatic vectorization and instruction selection. Achieves 1.75x over Triton, 2.61x over Ladder, 1.29x over QuantLLM, 1.03x over Marlin. A single parameterized program template efficiently supports the full range of quantization types.
**Confidence**: high
**Source**: Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation
**Evidence**: Key insight: low-precision (sub-8-bit) computation can be expressed as tile-level operations with automatic vectorization handling the bit-packing complexity. The programmer writes at the tile level; the compiler handles packing INT4/INT2/INT1 values into registers and selecting appropriate hardware instructions.
**Tags**: low-precision,quantization,tile-programming,INT4,INT2,compiler,LLM-serving

## Finding 530: Ozaki Scheme II emulates FP64 matrix multiplication using Chinese Remainder Theorem and INT8 tensor cores. Achieves 7.4-9.8 TFLOPS on RTX 4090 and 56.6-80.2 TFLOPS on GH200 — EXCEEDING native FP64 hardware performance. Also achieves 2.3x speedup for quadruple-precision emulation on CPUs. Controls number of matrix multiplications to tune accuracy vs speed.
**Confidence**: high
**Source**: Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique
**Evidence**: Key insight: the Chinese Remainder Theorem provides an algebraically exact decomposition of high-precision arithmetic into low-precision components. When combined with INT8 tensor cores (which are exact, unlike floating-point), you get FP64 GEMM that is both FASTER than native FP64 hardware AND mathematically equivalent.
**Tags**: FP64-emulation,Ozaki-scheme,INT8,Chinese-Remainder-Theorem,scientific-computing,precision

## Finding 531: DGEMM without FP64 Arithmetic: demonstrates FP64 matrix multiply using FP8 tensor cores + integer-emulated FP64 arithmetic. Eliminates hardware FP64 dependency entirely. Uses blocking strategies to optimize FP16 implementations. Motivated by industry trend of AI processors having no/slow FP64 hardware — shows double-precision scientific computing is still achievable on AI-focused chips.
**Confidence**: high
**Source**: DGEMM without FP64 Arithmetic - Using FP64 Emulation and FP8 Tensor Cores with Ozaki Scheme
**Evidence**: Key insight: as GPU vendors focus on AI-optimized low-precision (FP8/FP4), scientific computing users face declining FP64 performance per dollar. This paper proves that FP64 GEMM can be FULLY emulated on pure AI hardware with competitive performance, making AI-focused GPUs dual-purpose for scientific workloads.
**Tags**: FP64-emulation,FP8,scientific-computing,AI-hardware-reuse,Ozaki-scheme

## Finding 542: Automatic Dynamic Precision (ADP): GPU-resident framework for emulated FP64 GEMM via low-precision tensor cores. Introduces Exponent Span Capacity (ESC) metric to determine optimal decomposition parameters. Uses unsigned integer slicing to eliminate redundant sign bits. Achieves up to 2.3x speedup over native FP64 on Blackwell GB200 and 13.2x on RTX Pro 6000. Includes exception handling and native FP64 fallback.
**Confidence**: high
**Source**: Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores Through Extensions of the Ozaki Scheme
**Evidence**: From NVIDIA Research. Key insight: the ESC metric eliminates guesswork about how many slices/splits are needed for FP64 accuracy — it's computed from the input data's exponent range, making the decomposition ADAPTIVE rather than worst-case. The unsigned integer slicing is a novel optimization that previous Ozaki-style methods missed.
**Tags**: FP64-emulation,adaptive-precision,ESC,Ozaki-scheme,NVIDIA-research

## Finding 496: Formal SMT model of tensor cores across Volta, Turing, and Ampere reveals: (1) NVIDIA GPUs do NOT use round-to-zero accumulation as previously claimed in literature, (2) 5-term accumulator requires additional carry-out bits for accuracy, (3) a newer mixed-precision algorithm designed to be more accurate than an older one is actually LESS accurate for certain inputs. Generated test inputs that reveal behavioral differences between GPU generations.
**Confidence**: high
**Source**: An SMT Formalization of Mixed-Precision Matrix Multiplication: Modeling Three Generations of Tensor Cores
**Evidence**: From University of Utah. Published at NASA Formal Methods 2025. Key insight: tensor core arithmetic behavior is NOT well-documented by vendors and previous academic papers have gotten it WRONG. Formal verification reveals subtle numerical properties that empirical testing misses.
**Tags**: formal-verification,mixed-precision,numerical-accuracy,SMT,tensor-core-behavior

## Finding 598: Bipolar-INT data format enables arbitrary precision matrix multiplication at bit level on tensor cores. Facilitates parallel computing, supports symmetric quantization. Includes matrix preprocessing method and data recovery-oriented memory management using shared memory. Achieves 2.4x over CUTLASS for matrix multiplication and 6.7x inference acceleration for LLMs.
**Confidence**: high
**Source**: Efficient Arbitrary Precision Acceleration for Large Language Models on GPU Tensor Cores
**Evidence**: Accepted at ASP-DAC 2025. Key insight: by introducing a bipolar integer representation, arbitrary-precision arithmetic can be decomposed into a sequence of low-precision tensor core operations where each operation processes a few bits of the operand. The data recovery-oriented memory management strategically uses fast shared memory to reassemble results.
**Tags**: arbitrary-precision,bipolar-INT,quantization,bit-level,LLM-acceleration

## Finding 361: simdgroup_matrix is Apple GPU's tensor core equivalent - only 8x8 size supported. Types: simdgroup_float8x8, simdgroup_half8x8, simdgroup_bfloat8x8. Each 8x8=64 elements distributed across 32 threads (2 elements per thread via thread_elements())
**Confidence**: verified
**Source**: high
**Evidence**: simdgroup_matrix<T, 8, 8> is the only supported matrix size in Metal. The 64 elements of an 8x8 matrix are distributed across 32 threads in a SIMD group, with each thread holding exactly 2 elements accessible via thread_elements()[0] and thread_elements()[1]. Supported element types include float (32-bit), half (16-bit), and bfloat16. Unlike NVIDIA tensor cores which are dedicated hardware, Apple's simdgroup_matrix decreases register pressure and improves ALU utilization in existing FP32 pipelines.
**Tags**: 

## Finding 364: simdgroup_matrix operations: simdgroup_load(matrix, ptr, stride), simdgroup_store(matrix, ptr, stride), simdgroup_multiply(C, A, B), simdgroup_multiply_accumulate(D, A, B, C) where D = A*B + C
**Confidence**: verified
**Source**: high
**Evidence**: Load/store operations transfer between memory (device or threadgroup) and the distributed register representation. simdgroup_multiply computes C = A * B for 8x8 matrices. simdgroup_multiply_accumulate computes D = A * B + C, enabling FMA chains for GEMM tiling. For GEMM: tile the output into 8x8 blocks, each computed by one SIMD group. Loop over K dimension: load 8x8 tiles of A and B, accumulate via simdgroup_multiply_accumulate. Store result back. The stride parameter in load/store enables accessing sub-matrices of larger arrays.
**Tags**: 

## Finding 367: simdgroup_matrix has 3-cycle latency for FP16 FMA and 6-cycle latency for FP32 FMA. It reduces register pressure vs manual 8x8 multiply (which would need 64+ registers per matrix)
**Confidence**: verified
**Source**: high
**Evidence**: Benchmarked latencies: FFMA16 and FADD32/FMUL32 at 3 cycles, FFMA32 at 6 cycles. The simdgroup_matrix instruction decreases register pressure because 2 elements per thread (total 64 elements across 32 threads) only requires 2 registers per matrix, vs needing 64+ registers for explicit element storage. This also improves ALU utilization. Half-precision primarily remains useful to further decrease register pressure and bandwidth, since M1/A15 made FP32 just as fast as FP16 for arithmetic.
**Tags**: 

## Finding 371: MLX uses simdgroup_matrix<float,8,8> for Winograd convolution transforms - pattern: G*g*Gt and Bt*(O_mat*B) where G,Gt,B,Bt are 8x8 transform matrices and g is input tile
**Confidence**: verified
**Source**: high
**Evidence**: In MLX conv.metal: weight transform uses simdgroup_matrix<float,8,8> G, Gt to compute g_out = (G * g) * Gt. Output transform uses B, Bt to compute (Bt * (O_mat * B)). The SIMD lane position within the matrix is calculated as: qid = simd_lane_id/4, sm = (qid&4) + (simd_lane_id/2)%4, sn = (qid&2)*2 + (simd_lane_id%2)*2. Threadgroup memory stores weights as Ws[BO][R][R][BC], inputs as Is[A][A][BC], outputs as Os[M][M][BO], with threadgroup_barrier synchronization between loads and computes.
**Tags**: 

## Finding 375: Optimal GEMM tiling with simdgroup_matrix: assign each SIMD group an 8x8 output tile. Use threadgroup memory to stage A and B tiles. Chain simdgroup_multiply_accumulate over K dimension. Multiple SIMD groups per threadgroup process adjacent output tiles.
**Confidence**: verified
**Source**: high
**Evidence**: For a large GEMM (MxN output, K inner dimension): partition output into 8x8 tiles, each assigned to one SIMD group. Threadgroup loads BK-wide strips of A and B into threadgroup memory. Each SIMD group loops K/8 times, loading 8x8 sub-tiles and calling simdgroup_multiply_accumulate(D, A_tile, B_tile, D) to accumulate. Multiple SIMD groups within a threadgroup (e.g., 4 SIMD groups = 128 threads) process a 2x2 grid of 8x8 tiles. After the K loop completes, each SIMD group stores its 8x8 result via simdgroup_store.
**Tags**: 

## Finding 594: Fused3S: first fused 3S (SDDMM + softmax + SpMM) algorithm maximizing tensor core utilization while minimizing data movement. Introduces Binary Sparse Block (BSB) format to map sparse matrices onto tensor cores. 1.6-16.3x speedup on H100, 1.5-14x on A30 over state-of-art. Accelerates Graph Transformer inference 1.05-5.36x end-to-end.
**Confidence**: high
**Source**: Fused3S: Fast Sparse Attention on Tensor Cores
**Evidence**: Key insight: sparse attention decomposes into three operations (SDDMM, softmax, SpMM) that are traditionally executed as separate kernels with intermediate materialization. Fusing all three with a binary sparse block format eliminates intermediate storage and keeps data in tensor core registers across all three operations.
**Tags**: sparse-attention,SDDMM,SpMM,kernel-fusion,graph-transformer,tensor-core

## Finding 589: Acc-SpMM: high-performance sparse matrix-matrix multiplication on tensor cores using: data-affinity-based reordering (groups rows with similar nonzero patterns), memory-efficient compressed format, high-throughput pipeline, and adaptive sparsity-aware load balancing. 2.52x avg (5.11x max) over cuSPARSE on RTX 4090, 1.91x on A800, 1.58x on H100.
**Confidence**: high
**Source**: Acc-SpMM: Accelerating General-purpose Sparse Matrix-Matrix Multiplication with GPU Tensor Cores
**Evidence**: Published at PPoPP 2025. Key insight: the main bottleneck in sparse tensor core usage is load balancing — some tiles are mostly zeros while others are dense. The adaptive sparsity-aware balancing assigns work based on actual nonzero density per tile, not just tile count. Data-affinity reordering groups similar rows to maximize tensor core utilization.
**Tags**: sparse-matrix,tensor-core,load-balancing,SpMM,data-affinity,PPoPP

## Finding 509: SIMD² proposes generalized matrix operations using semiring-like structures — replacing multiply-add with arbitrary operations like add-minimum, multiply-max, etc. Accelerates 8 additional matrix operation types beyond standard GEMM. Achieves up to 38.59x speedup (avg 10.63x) over optimized CUDA on 8 non-ML applications. Requires only 5% additional chip area on existing tensor core hardware.
**Confidence**: high
**Source**: SIMD²: A Generalized Matrix Instruction Set for Accelerating Tensor Computation beyond GEMM
**Evidence**: Published at ISCA 2022. Applications include: shortest path (tropical semiring: add-min), graph matching, pattern matching, probabilistic inference. Key insight: tensor core hardware is ALMOST capable of general semiring operations — the multiply-accumulate datapath just needs the operation to be configurable. This means matrix accelerators can power graph algorithms, not just linear algebra.
**Tags**: semiring,non-ML,graph-algorithm,tensor-core-generalization,hardware,ISA

## Finding 513: SparStencil: first system to retarget SPARSE tensor cores (2:4 structured sparsity) for scientific stencil computations. Uses Adaptive Layout Morphing to restructure stencil patterns into staircase-aligned sparse matrices via flatten-and-crush pipeline. Formulates structured sparsity conversion as graph matching to ensure 2:4 sparsity compatibility. Up to 7.1x speedup (3.1x avg) over state-of-the-art across 79 stencil kernels.
**Confidence**: high
**Source**: SparStencil: Retargeting Sparse Tensor Cores to Scientific Stencil Computations via Structured Sparsity Transformation
**Evidence**: Accepted to SC 2025. Key insight: scientific stencil patterns (Laplacian, diffusion, etc.) have INHERENT sparsity that can be restructured to match hardware sparse tensor core formats. This is a completely novel use of sparse tensor cores for non-ML workloads.
**Tags**: stencil,sparse-tensor-core,scientific-computing,physics,2:4-sparsity,non-ML

## Finding 517: SPIDER: first system to turn sparsity from stencil-to-matrix transformation into an optimization opportunity using Sparse Tensor Cores. Combines preprocessing kernel matrices via strided swapping with dynamic input row-swapping. Zero runtime overhead. Outperforms cuDNN by 6.20x and state-of-the-art stencil methods by 2.00x on average.
**Confidence**: high
**Source**: SPIDER: Unleashing Sparse Tensor Cores for Stencil Computation via Strided Swapping
**Evidence**: Key insight: when stencil computations are expressed as matrix operations, the resulting matrices have a PREDICTABLE sparsity pattern. SPIDER exploits this predictability by pre-computing the swapping pattern at compile time, then applying it at runtime with zero overhead using sparse tensor core's native structured sparsity support.
**Tags**: stencil,sparse-tensor-core,strided-swapping,zero-overhead,scientific-computing

## Finding 518: HadaCore optimizes Fast Walsh-Hadamard Transform (FWHT) for tensor cores. Achieves same asymptotic complexity as original FWHT while leveraging hardware matrix multiply. Peak speedups of 3.5x (A100) and 3.6x (H100) using FP16/BF16. Maintains numerical accuracy comparable to FP32 implementations. Applied to Llama3 inference with quantized attention (Hadamard rotation for quantization-friendly representations).
**Confidence**: high
**Source**: HadaCore: Tensor Core Accelerated Hadamard Transform Kernel
**Evidence**: Key insight: the Walsh-Hadamard transform CAN be decomposed into a sequence of structured matrix multiplications that map to tensor cores. This is not obvious because FWHT is traditionally implemented as a butterfly-style divide-and-conquer algorithm. By reformulating it as tiled matrix multiplies, the tensor cores provide acceleration even though FWHT is not a traditional GEMM.
**Tags**: hadamard-transform,signal-processing,non-ML,tensor-core,quantization,butterfly

## Finding 526: Parallel scan (prefix sum) algorithm expressed entirely in the Tensor Core Unit computational model. For (s²,l)-TCU model with n inputs and p tensor core units: depth at most 2*floor(log_s(n)), performing O(n/s²) matrix multiplications. Scan enables: radix sort, quicksort, lexical analysis, stream compaction, polynomial evaluation — all via tensor cores.
**Confidence**: medium
**Source**: A Parallel Scan Algorithm in the Tensor Core Unit Model
**Evidence**: Key insight: prefix sum, which underlies many parallel algorithms (sort, compact, filter), can be expressed as a chain of matrix multiplications in a specially constructed form. This means ANY algorithm that reduces to prefix sum can potentially be accelerated by tensor cores, vastly expanding their applicability beyond GEMM.
**Tags**: prefix-sum,scan,sort,non-GEMM,tensor-core-theory,parallel-algorithm

## Finding 543: BLEST: BFS on tensor cores via Binarised Virtual Slice Sets (BVSS) for warp-level load balancing and batched SpMSpV multiplication. Reformulates pull-based BFS around bitmap-oriented structure to match tensor core operand formats. Achieves 3.58x over BerryBees, 4.64x over Gunrock, 4.9x over GSWITCH on real-world graphs. Uses kernel fusion and lazy vertex updates to minimize synchronization.
**Confidence**: high
**Source**: BLEST: Blazingly Efficient BFS using Tensor Cores
**Evidence**: Key insight: BFS can be expressed as sparse matrix-sparse vector multiplication (SpMSpV), which CAN be mapped to tensor cores if the sparse data is encoded as binary matrices. BVSS provides a compact binary representation that naturally maps to the integer matrix multiply capabilities of tensor cores.
**Tags**: BFS,graph-traversal,tensor-core,sparse-matrix,bitmap,non-ML

## Finding 547: Tensor-Core Beamformer: generic signal processing library using tensor cores for beamforming (combining signals from multiple sensors). Supports 16-bit and 1-bit precision. 16-bit achieves over 600 TeraOps/s on AMD MI300X. 1-bit mode breaks 3 PetaOps/s on A100. Cross-vendor: works on both NVIDIA and AMD. Applications: medical ultrasound imaging and radio astronomy.
**Confidence**: high
**Source**: The Tensor-Core Beamformer: A High-Speed Signal-Processing Library for Multidisciplinary Use
**Evidence**: Published at IPDPS 2025. Key insight: beamforming (delay-and-sum) is fundamentally a weighted matrix multiplication where sensor signals form one matrix and steering vectors form another. Tensor cores naturally accelerate this. The 1-bit mode is especially clever — binary beamforming for radio astronomy can use INT8 tensor cores with 0/1 values.
**Tags**: signal-processing,beamforming,medical-imaging,radio-astronomy,cross-vendor,non-ML

## Finding 551: Image processing pipelines (1D/2D convolution, upsampling, downsampling, resampling, recursive filtering, DCT denoising) are linear transformations over matrices in disguise and can be lowered to tensor cores. Uses equality saturation-based tensor instruction selector in Halide. Supports BOTH CPU and GPU tensor accelerators. A downsampling routine achieves 6.1x speedup on RTX 4070 via tensor cores.
**Confidence**: high
**Source**: Pushing Tensor Accelerators Beyond MatMul in a User-Schedulable Language
**Evidence**: Accepted to CGO 2026. Key insight: many image/signal processing operations that programmers implement as element-wise loops are actually matrix multiplications when viewed correctly. The equality saturation approach automatically discovers these equivalences and rewrites the computation to use tensor cores, without the programmer needing to know linear algebra.
**Tags**: image-processing,Halide,compiler,rewriting,non-ML,convolution,DCT,equality-saturation

## Finding 497: Software models emulating inner product behavior of matrix multipliers in V100, A100, H100, and B200 GPUs across 8-, 16-, and 19-bit floating point formats. Key finding: matrix multipliers targeted at AI are NOT compliant with IEEE 754, with different vendors offering different numerical features. This leads to non-reproducible results across GPU generations at the MMA instruction level.
**Confidence**: high
**Source**: Accurate Models of NVIDIA Tensor Cores
**Evidence**: Key properties modeled: rounding behaviour, accumulator width, normalization points, extra carry bits. Covers most input formats of interest: FP8, FP16, TF32/19-bit. The models can be configured per-GPU or with user-defined numerical features. Validates via randomized testing against actual hardware.
**Tags**: numerical-accuracy,tensor-core-model,IEEE754,reproducibility,mixed-precision

## Finding 501: MMA-Sim: first bit-accurate reference model covering TEN GPU architectures (8 NVIDIA + 2 AMD). Derived 9 distinct arithmetic algorithms for floating-point matrix multiplication across these platforms. Reveals UNDOCUMENTED behaviors affecting DNN training stability — identifies potential sources of significant computational errors in widely-used accelerators.
**Confidence**: high
**Source**: MMA-Sim: Bit-Accurate Reference Model of Tensor Cores and Matrix Cores
**Evidence**: Systematic testing combining targeted and randomized approaches. Confirmed bitwise equivalence with actual hardware. Key insight: there are at least 9 DIFFERENT arithmetic algorithms used across GPU vendors/generations for matrix multiply — the same GEMM produces different results on different hardware at the bit level.
**Tags**: bit-accurate,reference-model,cross-vendor,AMD,NVIDIA,numerical-behavior,reproducibility

## Finding 491: Tawa introduces asynchronous references (aref) as IR abstraction for warp-level communication without exposing hardware details. Automatically generates warp-specialized code from tile-based programs, assigning producer-consumer roles and managing dataflow pipelines. Achieves 1.1x over cuBLAS GEMM and 1.2x over Triton on attention, matching hand-optimized CUTLASS FlashAttention-3 on H100.
**Confidence**: high
**Source**: Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References
**Evidence**: From NVIDIA Research (Vinod Grover) + Cornell. Accepted to CGO 2026. Key insight: the 'asynchronous reference' abstraction captures the essential pattern of producer-consumer dataflow between specialized warps without requiring the programmer to manage barriers, shared memory allocation, or pipeline stages explicitly.
**Tags**: warp-specialization,async,compiler,producer-consumer,dataflow,aref

## Finding 495: Twill formulates software pipelining and warp specialization as a joint optimization problem solvable by off-the-shelf constraint solvers. First system to automatically derive OPTIMAL (not heuristic) SWP+WS schedules. Successfully reproduced and verified expert-developed schedules for FlashAttention on Hopper and Blackwell. Approach is heuristic-free, architecture-extensible, and guarantees optimality.
**Confidence**: high
**Source**: Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs
**Evidence**: From Stanford/NVIDIA (same group as Cypress). Key insight: the combinatorial explosion of software pipelining + warp specialization choices can be expressed as a constraint satisfaction problem, eliminating human heuristics. This means the optimal schedule is PROVABLY found, not approximated.
**Tags**: software-pipelining,warp-specialization,constraint-solver,optimal,compiler

