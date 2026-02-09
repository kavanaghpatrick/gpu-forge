---
name: simd-wave
description: >
  This skill should be used when the user asks about SIMD-level or wave-level GPU programming on Apple Silicon. Covers simdgroup operations, SIMD shuffle variants (simd_shuffle, simd_shuffle_xor, simd_shuffle_up, simd_shuffle_down, simd_shuffle_and_fill_up, simd_shuffle_and_fill_down), SIMD reductions (simd_sum, simd_product, simd_min, simd_max, simd_or, simd_and, simd_xor), prefix scans (simd_prefix_inclusive_sum, simd_prefix_exclusive_sum), simdgroup_matrix operations, cooperative tensors, ballot operations, quadgroup operations (quad_shuffle, quad_broadcast), parallel reduction patterns, prefix sum algorithms, sorting networks (bitonic sort, radix sort), divergence-aware programming, warp-level primitives, and wave-level optimization. Use when questions mention "SIMD", "simdgroup", "wave", "simd_shuffle", "simd_sum", "simd_prefix_exclusive_sum", "simdgroup_matrix", "reduction", "scan", "ballot", "warp", "quadgroup", "simdgroup_barrier", "butterfly pattern", "bitonic sort", "radix sort", "cooperative tensor", or wave-level GPU programming patterns.
---

# SIMD/Wave-Level Programming Knowledge

## Domain Overview

SIMD (Single Instruction, Multiple Data) programming on Apple GPUs operates at the level of 32-thread simdgroups -- the fundamental unit of execution. Every thread dispatched to an Apple GPU core executes in lockstep with 31 other threads in its simdgroup, sharing a program counter and execution mask. Understanding and exploiting this wave-level parallelism is critical for achieving peak GPU performance, as simdgroup operations bypass threadgroup memory entirely and communicate through the register file at extremely high bandwidth (256 bytes/cycle per core -- 2x more than AMD RDNA or NVIDIA).

Apple's Metal Shading Language provides a rich set of SIMD intrinsics organized into several categories: shuffle operations for arbitrary data exchange between lanes, reduction operations for computing aggregate values across all 32 threads, prefix scan operations for cumulative computations, and simdgroup_matrix operations for 8x8 matrix math that serves as Apple's equivalent to NVIDIA tensor cores. Metal 4 extends this further with cooperative tensor types that distribute matrix work across multiple SIMD groups.

Beyond the built-in intrinsics, effective wave-level programming requires understanding divergence handling (Apple's register-based execution mask stack), communication patterns between simdgroups (requiring threadgroup memory), quadgroup operations for 2D locality in image processing, and algorithmic patterns like parallel reductions, prefix sums, bitonic sorting networks, and radix sort. The absence of forward progress guarantees between threadgroups on Apple GPUs is a critical constraint that affects algorithm design -- certain lock-free patterns that work on NVIDIA GPUs will deadlock on Apple Silicon.

Research into tensor core programming beyond ML workloads has expanded rapidly, with simdgroup_matrix and cooperative tensors enabling non-ML applications: stencil computations, signal processing, graph traversal, image processing pipelines, and generalized semiring operations. Mixed-precision emulation techniques allow FP64-equivalent results from lower-precision tensor operations, opening scientific computing pathways on Apple GPUs.

## Key Knowledge Areas

The simd-wave skill contains 55 findings across these domains:

- **SIMD Shuffle Operations** (5 findings): Core shuffle variants, bandwidth characteristics, fill variants with modulo, XOR butterfly patterns, reading from inactive threads
- **Parallel Reduction Patterns** (4 findings): Multi-level SIMD reduction, built-in reduction functions, simdgroup_barrier optimization, atomic-free reduction technique
- **Prefix Sum Patterns** (4 findings): Built-in prefix scan functions, Hillis-Steele/Blelloch/Brent-Kung algorithms, multi-level scan for large arrays, GPU applications of prefix sum
- **simdgroup_matrix Deep Dive** (5 findings): 8x8 matrix operations, load/store/multiply API, FP16/FP32 latency, MLX Winograd usage, optimal GEMM tiling
- **Metal 4 Cooperative Tensors** (3 findings): Extended simdgroup_matrix, execution_simdgroups<N>, optimal tile sizes
- **Divergence-Aware Programming** (4 findings): Register-based execution mask (r0l), if_icmp/if_fcmp instructions, branchless FCMPSEL, Xcode divergence heat maps
- **Communication Patterns** (2 findings): Quadgroup operations for 2D image processing, cross-SIMD-group communication via threadgroup memory
- **Sorting Networks** (4 findings): MPS missing sort primitives, no forward progress guarantees, Metal-native radix sort performance, bitonic sort with simd_shuffle
- **Cooperative Tensor Programming** (3 findings): Cypress task-based model, ThunderKittens 16x16 tiles, TileLang scheduling
- **Tensor Core Non-ML Applications** (8 findings): Generalized semiring operations (SIMD-squared), sparse stencils (SparStencil/SPIDER), Walsh-Hadamard transform, BFS on tensor cores, beamforming, image processing, prefix sum on tensor cores
- **Mixed-Precision Emulation** (3 findings): Ozaki Scheme II (FP64 via INT8), DGEMM without FP64, Automatic Dynamic Precision
- **Warp Specialization** (2 findings): Tawa async references, Twill joint optimization
- **Other Tensor Research** (8 findings): Cross-vendor abstractions, low-precision types, numerical models, formalization, sparse operations

## How to Query

All simd-wave knowledge is stored in the GPU Computer knowledge database. Query using the `kb` CLI:

```bash
# List all simd-wave findings
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill simd-wave

# Search for specific topics
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "simd_shuffle"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "simdgroup_matrix"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "parallel reduction"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "prefix sum"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "cooperative tensor"

# Get detailed finding with sources
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <finding-id>

# Check statistics
${CLAUDE_PLUGIN_ROOT}/scripts/kb stats
```

Replace `${CLAUDE_PLUGIN_ROOT}` with the actual path to the gpu-forge plugin directory.

## Common Patterns & Quick Answers

### Q: What SIMD shuffle variants are available on Apple GPUs?
**A**: Six core variants: `simd_shuffle(data, lane)` for arbitrary lane read, `simd_shuffle_xor(data, mask)` for butterfly patterns, `simd_shuffle_up(data, delta)` and `simd_shuffle_down(data, delta)` for shift patterns, `simd_shuffle_and_fill_up(data, fill, delta)` and `simd_shuffle_and_fill_down(data, fill, delta)` with optional modulo argument for virtual sub-vectors. Bandwidth: 256 bytes/cycle per core, 2x AMD/NVIDIA. [Finding #310, #314]

### Q: How to implement a parallel reduction?
**A**: Multi-level pattern: (1) each thread reduces local values, (2) `simd_sum`/`simd_max` across 32-thread SIMD group, (3) write per-SIMD result to threadgroup memory, (4) final SIMD reduces those values. Built-in reductions: `simd_sum`, `simd_product`, `simd_min`/`simd_minimum`, `simd_max`/`simd_maximum`, `simd_or`, `simd_and`, `simd_xor`. For small threadgroups (<=32 threads), use `simdgroup_barrier` instead of `threadgroup_barrier`. [Finding #328, #332, #336]

### Q: What is simdgroup_matrix and how does it compare to NVIDIA tensor cores?
**A**: `simdgroup_matrix` is Apple GPU's tensor core equivalent -- only 8x8 size supported. Types: `simdgroup_float8x8`, `simdgroup_half8x8`, `simdgroup_bfloat8x8`. Operations: `simdgroup_load`, `simdgroup_store`, `simdgroup_multiply_accumulate`. Latency: 3-cycle FP16 FMA, 6-cycle FP32 FMA. Reduces register pressure vs manual SIMD operations. [Finding #361, #364, #367]

### Q: How to implement prefix sum (scan) on Apple GPUs?
**A**: Metal provides `simd_prefix_inclusive_sum(data)` and `simd_prefix_exclusive_sum(data)` for within-SIMD scans. For large arrays, use multi-level pattern: (1) SIMD-level scan within each 32-thread group, (2) write per-SIMD totals to threadgroup memory, (3) scan the totals, (4) add scanned totals back. Three algorithms: Hillis-Steele (step-efficient, maps to SIMD), Blelloch (work-efficient), Brent-Kung (hybrid). Applications: stream compaction, histogram building, radix sort, memory allocation. [Finding #344, #349, #353, #357]

### Q: How does divergence work on Apple GPUs?
**A**: Apple GPUs use a register-based execution mask stack tracked in register r0l. Each thread tracks nesting depth as a 16-bit counter. `if_icmp`/`if_fcmp` instructions increment r0l for failing threads; only threads with r0l==0 remain active. Both paths execute serially with inactive threads masked out. Use `FCMPSEL` (conditional select) at 1-cycle throughput for branchless alternatives. Xcode 15 provides Thread Divergence Heat Maps for M3/A17 Pro. [Finding #382, #383, #384, #385]

### Q: How to sort data on Apple GPUs?
**A**: Metal Performance Shaders lacks built-in reduce, scan, and radix sort. For small data within a SIMD group, use bitonic sort with `simd_shuffle` for register-level compare-and-swap (5 stages for 32 elements, no threadgroup memory needed). For large arrays, implement DeviceRadixSort (Metal-native achieves ~3 billion elements/sec on M1 Max). Critical constraint: Apple GPUs lack forward progress guarantees between threadgroups -- OneSweep-style algorithms deadlock. [Finding #386, #387, #388, #389]

### Q: What are Metal 4 cooperative tensors?
**A**: Metal 4 extends `simdgroup_matrix` with cooperative tensor types that distribute matrix work across multiple SIMD groups using `execution_simdgroups<N>`. Optimal tile size is 128x64x64 with 4 SIMD groups. Static slicing outperforms dynamic; fully unrolled loops with compile-time constants yield best performance. [Finding #381, #390, #391]

### Q: How do threads in different SIMD groups communicate?
**A**: Cross-SIMD-group communication requires threadgroup memory + `threadgroup_barrier()`. Raking pattern: each SIMD writes per-SIMD results to shared threadgroup memory, barrier, then one SIMD reads and processes all results. Quadgroup operations (`quad_shuffle`, `quad_broadcast`) operate on 4-thread subsets within a SIMD group for 2D image processing locality. [Finding #392, #393]

### Q: What is the optimal GEMM tiling strategy with simdgroup_matrix?
**A**: Assign each SIMD group an 8x8 output tile. Stage A and B matrix tiles in threadgroup memory for coalesced loads. Each SIMD group performs multiply-accumulate on its 8x8 tile. MLX uses this pattern with `simdgroup_matrix<float,8,8>` for Winograd convolution transforms. [Finding #375, #371]

### Q: Can simd_shuffle read from inactive (masked-out) threads?
**A**: Yes. `simd_shuffle` can read from inactive threads -- important for divergent code paths where data from masked-out threads is still needed. This is a key difference from some other GPU architectures. [Finding #323]

### Q: What non-ML workloads can use simdgroup_matrix/tensor operations?
**A**: Research has demonstrated tensor core usage for: generalized semiring matrix operations (SIMD-squared), sparse stencil computations (SparStencil, SPIDER), Fast Walsh-Hadamard Transform (HadaCore), BFS graph traversal (BLEST), beamforming signal processing, image processing pipelines (convolution, resampling, DCT denoising), and parallel prefix sum. Mixed-precision emulation enables FP64-equivalent scientific computing on FP8/INT8 tensor hardware. [Finding #509, #513, #517, #518, #526, #543, #547, #551, #530, #531]

## Advanced Topics

### Atomic-Free Reduction
Avoid expensive atomics by having each threadgroup reduce to a single value, write to an output buffer indexed by `threadgroup_position_in_grid`, then launch a second pass to reduce the partial results. Scales well for very large inputs. [Finding #340]

### Butterfly Communication with simd_shuffle_xor
`simd_shuffle_xor(data, mask)` enables butterfly communication patterns essential for parallel reductions and FFT. Each thread swaps values with its XOR partner, enabling log2(N) reduction steps without threadgroup memory. [Finding #320]

### Virtual Sub-Vectors with Fill Variants
`simd_shuffle_and_fill_up/down` with modulo argument splits the SIMD group into smaller virtual vectors -- useful for algorithms requiring sub-SIMD-width data exchange patterns. [Finding #318]

### Forward Progress Constraints
Apple Silicon GPUs lack forward progress guarantees between threadgroups. Algorithms relying on inter-threadgroup synchronization (OneSweep radix sort, certain lock-free data structures) will deadlock. Design algorithms to be self-contained within each threadgroup dispatch. [Finding #387]

## Cross-References

### Related Skills

- **msl-kernels** (Layer 1): Provides MSL language context (address spaces, atomics, function constants) that complements simd-wave's algorithmic patterns
- **gpu-perf** (Layer 2): Uses simd-wave knowledge to diagnose SIMD occupancy issues, divergence bottlenecks, and reduction efficiency
- **gpu-silicon** (Layer 0): Foundational hardware knowledge -- SIMD width (32), ALU organization (128 ALUs/core), shuffle bandwidth (256 B/cycle), register file size
- **metal-compute** (Layer 1): Dispatch and pipeline configuration that determines how simdgroups are scheduled and threadgroups are sized
- **metal4-api** (Layer 3): Metal 4 cooperative tensors extend simd-wave's simdgroup_matrix patterns to multi-SIMD-group operations
- **mlx-compute** (Layer 3): MLX framework uses simdgroup_matrix internally for custom Metal kernels (Winograd convolutions, GEMM)

### Key Dependencies

This skill depends on:
- **gpu-silicon** for hardware SIMD width, ALU counts, and shuffle bandwidth specs
- **metal-compute** for compute pipeline and dispatch configuration

Higher-layer skills depend on simd-wave for:
- Efficient reduction and scan patterns (gpu-perf optimization)
- simdgroup_matrix tiling strategies (mlx-compute, metal4-api)
- Divergence-aware programming techniques (msl-kernels)

### Finding ID Reference

Key findings by topic area:
- **SIMD Shuffle**: #310, #314, #318, #320, #323
- **Reductions**: #328, #332, #336, #340
- **Prefix Scans**: #344, #349, #353, #357
- **simdgroup_matrix**: #361, #364, #367, #371, #375
- **Metal 4 Cooperative Tensors**: #381, #390, #391
- **Divergence**: #382, #383, #384, #385
- **Communication**: #392, #393
- **Sorting**: #386, #387, #388, #389
- **Cooperative Tensor Research**: #479, #480, #485
- **Tensor Non-ML**: #509, #513, #517, #518, #526, #543, #547, #551
- **Mixed-Precision**: #530, #531, #542
- **Warp Specialization**: #491, #495
- **Other Research**: #481, #486, #496, #497, #501, #593, #594, #589, #598

Use `${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <id>` to retrieve full finding details including source URLs and confidence levels.
