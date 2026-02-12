---
spec: attention-proto
phase: tasks
total_tasks: 43
created: 2026-02-12T08:30:00Z
generated: auto
---

# Tasks: attention-proto

## Phase 1: Foundation (Shared Infrastructure)

Focus: Establish project scaffolding, build system, shared Metal host code. All prototypes depend on this.

- [x] 1.1 Create project structure and Cargo manifest
  - **Do**: Create `attention-proto/` directory, Cargo.toml with workspace member, features (cubecl, burn-ext, gpu-counters), dev-dependencies (criterion)
  - **Files**: `attention-proto/Cargo.toml`, `attention-proto/src/lib.rs`
  - **Done when**: `cargo check` passes, all features resolve
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/attention-proto && cargo check --all-features`
  - **Commit**: `feat(attention-proto): create project structure`
  - _Requirements: FR-14_
  - _Design: Project Structure_

- [x] 1.2 Implement build.rs for Metal shader compilation
  - **Do**: Copy particle-system/build.rs pattern, set -std=metal3.1 for simdgroup_matrix, -O2 for release, compile shaders/*.metal → .air → .metallib
  - **Files**: `attention-proto/build.rs`, `attention-proto/shaders/types.h` (empty placeholder)
  - **Done when**: Build script compiles, rerun-if-changed triggers on .metal files
  - **Verify**: `cargo build` (no shaders yet, so no compile errors expected)
  - **Commit**: `build(attention-proto): add Metal shader build system`
  - _Requirements: FR-14_
  - _Design: Build System_

- [x] 1.3 Implement GpuDevice singleton
  - **Do**: Adapt gpu-query/src/gpu/device.rs, use OnceLock for thread-safe singleton, find_metallib() searches OUT_DIR and target/debug, implement Drop for cleanup
  - **Files**: `attention-proto/src/device.rs`
  - **Done when**: GpuDevice::shared() returns MTLDevice + MTLCommandQueue + MTLLibrary
  - **Verify**: `cargo test --lib device::tests::test_device_init`
  - **Commit**: `feat(attention-proto): implement GpuDevice singleton`
  - _Requirements: FR-14, AC-8.1_
  - _Design: Shared Infrastructure_

- [x] 1.4 Implement PsoCache with function constants
  - **Do**: Adapt gpu-query/src/gpu/pipeline.rs, serialize function constant values to PsoKey, implement get_or_compile, add binary archive support (save/load)
  - **Files**: `attention-proto/src/pipeline.rs`
  - **Done when**: PsoCache::get_or_compile returns cached or fresh-compiled PSO
  - **Verify**: `cargo test --lib pipeline::tests::test_pso_cache`
  - **Commit**: `feat(attention-proto): implement PSO cache with function constants`
  - _Requirements: FR-14, AC-8.2_
  - _Design: Shared Infrastructure_

- [x] 1.5 Implement compute encoder helpers
  - **Do**: Adapt gpu-query/src/gpu/encode.rs, add alloc_buffer, alloc_buffer_with_data, dispatch_1d, dispatch_2d (new, needed for attention), read_buffer, read_buffer_slice
  - **Files**: `attention-proto/src/encode.rs`
  - **Done when**: All helper functions compile, type-safe buffer allocation and dispatch
  - **Verify**: `cargo test --lib encode::tests::test_buffer_roundtrip`
  - **Commit**: `feat(attention-proto): implement compute encoder helpers`
  - _Requirements: FR-14, AC-8.3_
  - _Design: Shared Infrastructure_

- [x] 1.6 Implement GPU timing infrastructure
  - **Do**: Create benchmark_kernel_gpu_time (GPUStartTime/GPUEndTime), gpu_warmup (4 dispatches), optional MTLCounterSampleBuffer behind gpu-counters feature
  - **Files**: `attention-proto/src/timing.rs`
  - **Done when**: benchmark_kernel_gpu_time returns f64 seconds, warmup completes without error
  - **Verify**: `cargo test --lib timing::tests::test_gpu_warmup`
  - **Commit**: `feat(attention-proto): implement GPU timing infrastructure`
  - _Requirements: FR-14, AC-8.4, NFR-6, NFR-7_
  - _Design: Shared Infrastructure_

- [x] 1.7 Implement KB finding output
  - **Do**: Define KbFinding struct (domain, title, content, tags, confidence, source), emit_finding writes JSON-lines to findings.jsonl
  - **Files**: `attention-proto/src/kb.rs`
  - **Done when**: emit_finding appends valid JSON to findings.jsonl
  - **Verify**: `cargo test --lib kb::tests::test_emit_finding`
  - **Commit**: `feat(attention-proto): implement KB finding output`
  - _Requirements: FR-18, AC-8.5_
  - _Design: Shared Infrastructure_

- [x] 1.8 Define AttentionParams #[repr(C)] type
  - **Do**: Create types.rs with AttentionParams struct matching shaders/types.h, add offset_of! layout assertions, include all fields for 8 prototypes
  - **Files**: `attention-proto/src/types.rs`, `attention-proto/shaders/types.h`
  - **Done when**: Layout assertions pass, 64-byte struct with explicit padding
  - **Verify**: `cargo test --lib types::tests::test_attention_params_layout`
  - **Commit**: `feat(attention-proto): define AttentionParams type with layout assertions`
  - _Requirements: FR-14_
  - _Design: Shared Infrastructure, types.h_

- [x] 1.9 Foundation checkpoint
  - **Do**: Verify all shared infrastructure compiles, tests pass, ready for prototype implementation
  - **Done when**: `cargo test --lib` passes, no clippy warnings
  - **Verify**: `cargo clippy -- -D warnings && cargo test --lib`
  - **Commit**: `feat(attention-proto): complete shared infrastructure`
  - _Requirements: FR-14_

## Phase 2: Proto 1 — Flash Attention (Critical Path)

Focus: Validate core hypothesis. Hand-written simdgroup_matrix tiled attention kernel.

- [x] 2.1 Implement naive CPU attention reference (FP64)
  - **Do**: In proto1_flash.rs, implement cpu_attention_f64: naive O(N²) scaled dot-product attention with FP64 accumulation, safe softmax (max subtraction), output FP32
  - **Files**: `attention-proto/src/proto1_flash.rs`
  - **Done when**: CPU reference produces correct attention output for small inputs (N=4, D=8)
  - **Verify**: Manual test with known Q/K/V (e.g., identity Q, uniform K) verifies output
  - **Commit**: `feat(proto1): implement CPU FP64 attention reference`
  - _Requirements: FR-15, AC-1.3_
  - _Design: Proto 1_

- [x] 2.2 Write basic flash_attention.metal kernel
  - **Do**: Implement kernel with function constants HEAD_DIM, BLOCK_R, BLOCK_C. Threadgroup memory for Q_tile, K_chunk. Online softmax with running max/sum. simdgroup_matrix for matmul. Start with simplest tile (16x64, D=64)
  - **Files**: `attention-proto/shaders/flash_attention.metal`
  - **Done when**: Kernel compiles with xcrun metal -std=metal3.1
  - **Verify**: `cargo build --release` (build.rs compiles shader)
  - **Commit**: `feat(proto1): implement basic Flash Attention kernel with simdgroup_matrix`
  - _Requirements: FR-1, AC-1.1_
  - _Design: Proto 1 MSL Design_

- [x] 2.3 Implement proto1_flash host code and correctness test
  - **Do**: Allocate Q/K/V buffers, upload data, compile PSO with function constants, dispatch 2D (ceil(N/Br), heads), read back, assert_allclose vs CPU reference (atol=5e-3, rtol=1e-2 for FP16)
  - **Files**: `attention-proto/src/proto1_flash.rs`, `attention-proto/tests/integration.rs`
  - **Done when**: Correctness test passes for N=256, D=64, single head
  - **Verify**: `MTL_SHADER_VALIDATION=1 cargo test --release -- proto1::test_flash_correctness --test-threads=1`
  - **Commit**: `feat(proto1): implement host code and correctness test`
  - _Requirements: AC-1.3, NFR-3, NFR-17_
  - _Design: Proto 1 Host Code_

- [x] 2.4 Implement tile size sweep benchmark
  - **Do**: Criterion benchmark group for D × N × tile_config. Configurations: Br in {16,32,64}, Bc in {64,128}, D in {64,128}. Compute TFLOPS = (4*N²*D / time) / 1e12. GPU warmup before measurement.
  - **Files**: `attention-proto/benches/flash_attention.rs`
  - **Done when**: Benchmark runs, produces TFLOPS numbers, CV < 10%
  - **Verify**: `cargo bench --bench flash_attention -- --output-format json > bench.json`
  - **Commit**: `feat(proto1): add tile size sweep benchmark`
  - _Requirements: FR-2, AC-1.2, NFR-5_
  - _Design: Proto 1 Benchmark_

- [x] 2.5 Generate KB findings for Proto 1
  - **Do**: Parse bench.json, emit 5+ findings (one per optimal config per D), include TFLOPS, tile size, occupancy estimate, comparison vs MFA M4 published numbers
  - **Files**: Update findings.jsonl via kb.rs
  - **Done when**: At least 5 findings emitted with confidence >= 0.8 (CV < 5%)
  - **Verify**: `cat /Users/patrickkavanagh/gpu_kernel/attention-proto/findings.jsonl | grep proto1 | wc -l` >= 5
  - **Commit**: `docs(proto1): add KB findings for Flash Attention tile sizes`
  - _Requirements: AC-1.4, AC-1.5_

- [x] 2.6 Proto 1 checkpoint
  - **Do**: Verify Proto 1 complete: correctness test passes, benchmark runs, findings generated
  - **Done when**: All Proto 1 tasks done, baseline TFLOPS established
  - **Verify**: `MTL_SHADER_VALIDATION=1 cargo test --release -- proto1 --test-threads=1 && cargo bench --bench flash_attention`
  - **Commit**: `feat(proto1): complete Flash Attention prototype`
  - _Requirements: US-1_

## Phase 3: Proto 4 — Function Constants (Determines Dispatch Strategy)

Focus: Measure compilation overhead. Must run early to inform Proto 2 (stitching).

- [x] 3.1 Implement PSO compilation benchmark
  - **Do**: Benchmark cold compilation of N variants (N=1/10/50/100/500) with different function constant values. Measure time from MTLFunctionConstantValues to compiled PSO. Use Proto 1 kernel.
  - **Files**: `attention-proto/benches/constant_overhead.rs`
  - **Done when**: Benchmark measures compilation time per variant count
  - **Verify**: `cargo bench --bench constant_overhead -- cold_compile`
  - **Commit**: `feat(proto4): add function constant compilation benchmark`
  - _Requirements: FR-4, AC-2.2_
  - _Design: Proto 4_

- [x] 3.2 Implement binary archive benchmark
  - **Do**: Measure archive creation time for 72 variants (all combinations of HEAD_DIM, BLOCK_R, BLOCK_C, VARIANT). Then measure PSO load time from archive vs fresh compilation. Compare speedup.
  - **Files**: `attention-proto/benches/constant_overhead.rs`
  - **Done when**: Benchmark measures archive build time and per-PSO load time
  - **Verify**: `cargo bench --bench constant_overhead -- binary_archive`
  - **Commit**: `feat(proto4): add binary archive compilation benchmark`
  - _Requirements: FR-5, AC-2.3_
  - _Design: Proto 4_

- [x] 3.3 Generate KB findings for Proto 4
  - **Do**: Emit 3+ findings: ms/variant cold compile, archive build time for 72 variants, archive load speedup factor, recommendation (use constants if < 50ms/variant, else pre-compile)
  - **Files**: Update findings.jsonl
  - **Done when**: At least 3 findings emitted
  - **Verify**: `cat findings.jsonl | grep proto4 | wc -l` >= 3
  - **Commit**: `docs(proto4): add KB findings for function constant overhead`
  - _Requirements: AC-2.4, AC-2.5, AC-2.6_

## Phase 4: Parallel Prototypes (Architecture-Shaping)

Focus: Run Protos 2, 3, 6, 7 in parallel after Proto 1 and Proto 4 establish baselines.

### Proto 2: Function Stitching

- [x] 4.1 Implement flash_attention_stitched.metal
  - **Do**: Create variant of Proto 1 kernel with function constant STITCH_MODE (0=monolithic, 1=inline, 2=function_table). Factor score compute, softmax, accumulation into separate functions with different linkage.
  - **Files**: `attention-proto/shaders/flash_attention_stitched.metal`
  - **Done when**: Kernel compiles for all 3 STITCH_MODE values
  - **Verify**: `cargo build --release`
  - **Commit**: `feat(proto2): implement stitchable function variants`
  - _Requirements: FR-3, AC-2.1_
  - _Design: Proto 2_

- [x] 4.2 Implement function stitching benchmark
  - **Do**: Compile 3 PSOs (STITCH_MODE 0/1/2), run identical workload (N=2048, D=128), measure GPU time, compare. 100 iterations each for statistical significance.
  - **Files**: `attention-proto/benches/function_stitch.rs`
  - **Done when**: Benchmark reports ns/call overhead for stitching vs monolithic
  - **Verify**: `cargo bench --bench function_stitch`
  - **Commit**: `feat(proto2): add function stitching overhead benchmark`
  - _Requirements: AC-2.1_
  - _Design: Proto 2_

- [x] 4.3 Generate KB findings for Proto 2
  - **Do**: Emit 3+ findings: inline overhead (expected <1%), function_table overhead (expected 5-15%), break-even point, recommendation for trait dispatch
  - **Files**: Update findings.jsonl
  - **Done when**: At least 3 findings emitted
  - **Verify**: `cat findings.jsonl | grep proto2 | wc -l` >= 3
  - **Commit**: `docs(proto2): add KB findings for function stitching overhead`
  - _Requirements: AC-2.4, US-2_

### Proto 3: PagedAttention V2

- [x] 4.4 Implement paged_attention.metal (phase 1 kernel)
  - **Do**: Kernel takes page_table buffer, iterates partitions, loads KV from physical page offsets, computes partial attention outputs + running max/sum. Threadgroup memory: 2 pages (16KB) + Q_tile (8KB) = 24KB fits in 32KB.
  - **Files**: `attention-proto/shaders/paged_attention.metal`
  - **Done when**: Phase 1 kernel compiles
  - **Verify**: `cargo build --release`
  - **Commit**: `feat(proto3): implement PagedAttention V2 partition kernel`
  - _Requirements: FR-6, AC-3.1_
  - _Design: Proto 3 MSL Design (Phase 1)_

- [x] 4.5 Implement paged_reduce.metal (phase 2 kernel)
  - **Do**: Kernel combines partial outputs from all partitions using log-sum-exp reduction trick, produces final O
  - **Files**: `attention-proto/shaders/paged_reduce.metal`
  - **Done when**: Phase 2 kernel compiles
  - **Verify**: `cargo build --release`
  - **Commit**: `feat(proto3): implement PagedAttention V2 reduce kernel`
  - _Requirements: FR-6, AC-3.1_
  - _Design: Proto 3 MSL Design (Phase 2)_

- [x] 4.6 Implement proto3_paged host code and correctness test
  - **Do**: Allocate page pool, construct fragmented page_table, two-pass dispatch (partition then reduce), validate bit-exact vs Proto 1 contiguous attention
  - **Files**: `attention-proto/src/proto3_paged.rs`, `attention-proto/tests/integration.rs`
  - **Done when**: Correctness test passes (atol=0, rtol=0 for bit-exact)
  - **Verify**: `MTL_SHADER_VALIDATION=1 cargo test --release -- proto3::test_paged_correctness --test-threads=1`
  - **Commit**: `feat(proto3): implement PagedAttention host code and correctness test`
  - _Requirements: AC-3.2, NFR-17_
  - _Design: Proto 3 Host Code_

- [x] 4.7 Implement threadgroup memory budget test
  - **Do**: Test varying block sizes (8/16/32/64/128 tokens), measure threadgroup memory usage (page_size * D * 4 bytes * 2 pages + Q_tile), verify <= 32KB
  - **Files**: `attention-proto/tests/integration.rs`
  - **Done when**: Test documents memory usage at each block size, fails if exceeds 32KB
  - **Verify**: `cargo test --release -- proto3::test_threadgroup_budget`
  - **Commit**: `test(proto3): add threadgroup memory budget test`
  - _Requirements: FR-7, AC-3.3, NFR-1_
  - _Design: Proto 3_

- [x] 4.8 Implement PagedAttention benchmark
  - **Do**: Benchmark context lengths 1K/4K/8K/16K, page sizes 8/16/32 tokens. Measure throughput vs Proto 1 contiguous. Report overhead %.
  - **Files**: `attention-proto/benches/paged_attention.rs`
  - **Done when**: Benchmark runs, reports overhead 10-25%
  - **Verify**: `cargo bench --bench paged_attention`
  - **Commit**: `feat(proto3): add PagedAttention throughput benchmark`
  - _Requirements: AC-3.4_
  - _Design: Proto 3 Benchmark_

- [x] 4.9 Generate KB findings for Proto 3
  - **Do**: Emit 5+ findings: optimal page size for M4, threadgroup memory map, overhead vs contiguous, two-phase viability, recommendation
  - **Files**: Update findings.jsonl
  - **Done when**: At least 5 findings emitted
  - **Verify**: `cat findings.jsonl | grep proto3 | wc -l` >= 5
  - **Commit**: `docs(proto3): add KB findings for PagedAttention V2`
  - _Requirements: AC-3.5, US-3_

### Proto 6: FLA Linear Attention

- [x] 4.10 Implement CPU linear attention reference (FP64)
  - **Do**: In proto6_fla.rs, implement cpu_linear_attention_f64: chunk-based recurrence, H = sum(K^T * V) over chunks, O = Q * H. FP64 accumulation.
  - **Files**: `attention-proto/src/proto6_fla.rs`
  - **Done when**: CPU reference produces correct output for small inputs
  - **Verify**: Manual test with known inputs
  - **Commit**: `feat(proto6): implement CPU FP64 linear attention reference`
  - _Requirements: FR-15_
  - _Design: Proto 6_

- [x] 4.11 Implement linear_attention.metal (chunk_h kernel)
  - **Do**: Kernel computes H_chunk = K_chunk^T * V_chunk (D×D outer product) using simdgroup_matrix. One threadgroup per chunk.
  - **Files**: `attention-proto/shaders/linear_attention.metal`
  - **Done when**: chunk_h kernel compiles
  - **Verify**: `cargo build --release`
  - **Commit**: `feat(proto6): implement FLA chunk_h kernel`
  - _Requirements: FR-8, AC-4.1_
  - _Design: Proto 6 MSL Design (chunk_h)_

- [x] 4.12 Implement linear_attention.metal (chunk_o kernel)
  - **Do**: Kernel computes O_chunk = Q_chunk * H_chunk (C×D = C×D * D×D matmul). Plus intra-chunk causal attention for tokens within chunk.
  - **Files**: `attention-proto/shaders/linear_attention.metal`
  - **Done when**: chunk_o kernel compiles
  - **Verify**: `cargo build --release`
  - **Commit**: `feat(proto6): implement FLA chunk_o kernel`
  - _Requirements: FR-8, AC-4.1_
  - _Design: Proto 6 MSL Design (chunk_o)_

- [x] 4.13 Implement proto6_fla host code and correctness test
  - **Do**: Two-pass dispatch (chunk_h then chunk_o), validate vs CPU linear attention reference (atol=1e-4, rtol=1e-3)
  - **Files**: `attention-proto/src/proto6_fla.rs`, `attention-proto/tests/integration.rs`
  - **Done when**: Correctness test passes for N=512, D=64, chunk=64
  - **Verify**: `MTL_SHADER_VALIDATION=1 cargo test --release -- proto6::test_fla_correctness --test-threads=1`
  - **Commit**: `feat(proto6): implement FLA host code and correctness test`
  - _Requirements: AC-4.2, NFR-4, NFR-17_
  - _Design: Proto 6 Host Code_

- [x] 4.14 Implement linear attention benchmark
  - **Do**: Benchmark seq lengths 1K/4K/16K/64K (long context), chunk sizes 32/64/128/256, D=64/128. Compare TFLOPS vs Proto 1 softmax attention. Report crossover point.
  - **Files**: `attention-proto/benches/linear_attention.rs`
  - **Done when**: Benchmark runs, identifies crossover seq_len where linear becomes faster
  - **Verify**: `cargo bench --bench linear_attention`
  - **Commit**: `feat(proto6): add FLA linear attention benchmark`
  - _Requirements: FR-9, AC-4.3_
  - _Design: Proto 6 Benchmark_

- [x] 4.15 Generate KB findings for Proto 6
  - **Do**: Emit 5+ findings: optimal chunk size, TFLOPS at each seq_len, crossover point vs softmax attention, memory bandwidth utilization, recommendation
  - **Files**: Update findings.jsonl
  - **Done when**: At least 5 findings emitted
  - **Verify**: `cat findings.jsonl | grep proto6 | wc -l` >= 5
  - **Commit**: `docs(proto6): add KB findings for FLA linear attention`
  - _Requirements: AC-4.4, AC-4.5, US-4_

### Proto 7: RoPE/ALiBi/GQA Variants

- [x] 4.16 Implement rope.metal kernel
  - **Do**: Standalone kernel applying rotary embeddings to Q and K tensors. Each thread handles one (token, dim_pair). Sincos lookup, rotation matrix.
  - **Files**: `attention-proto/shaders/rope.metal`
  - **Done when**: RoPE kernel compiles
  - **Verify**: `cargo build --release`
  - **Commit**: `feat(proto7): implement RoPE kernel`
  - _Requirements: FR-10_
  - _Design: Proto 7 MSL Design (RoPE)_

- [x] 4.17 Implement alibi.metal (fused into Proto 1)
  - **Do**: Modify Proto 1 kernel to add ALiBi bias: bias = -slope * abs(pos_q - pos_k) to score. Function constant ALIBI_ENABLED.
  - **Files**: `attention-proto/shaders/flash_attention.metal` (add ALiBi variant)
  - **Done when**: ALiBi variant compiles
  - **Verify**: `cargo build --release`
  - **Commit**: `feat(proto7): implement ALiBi bias variant`
  - _Requirements: FR-10_
  - _Design: Proto 7 MSL Design (ALiBi)_

- [x] 4.18 Implement gqa_remap.metal kernel
  - **Do**: Index remapping kernel or inline remapping in attention kernel. kv_head = q_head / group_size. Measure buffer rebinding overhead.
  - **Files**: `attention-proto/shaders/gqa_remap.metal`
  - **Done when**: GQA kernel compiles
  - **Verify**: `cargo build --release`
  - **Commit**: `feat(proto7): implement GQA head remapping`
  - _Requirements: FR-10_
  - _Design: Proto 7 MSL Design (GQA)_

- [x] 4.19 Implement proto7_variants host code and correctness tests
  - **Do**: For each variant (RoPE, ALiBi, GQA), implement CPU reference, run GPU kernel, validate vs reference (atol=1e-4, rtol=1e-3)
  - **Files**: `attention-proto/src/proto7_variants.rs`, `attention-proto/tests/integration.rs`
  - **Done when**: Correctness tests pass for all 3 variants
  - **Verify**: `MTL_SHADER_VALIDATION=1 cargo test --release -- proto7::test_rope_correctness proto7::test_alibi_correctness proto7::test_gqa_correctness --test-threads=1`
  - **Commit**: `feat(proto7): implement variant host code and correctness tests`
  - _Requirements: AC-5.1, AC-5.2, NFR-17_
  - _Design: Proto 7 Host Code_

- [x] 4.20 Implement variant overhead benchmark
  - **Do**: Measure per-variant microseconds: standalone (RoPE only, ALiBi only, GQA only) and fused into Proto 1 base attention. N=2048, D=128, heads=32. GQA group sizes 1/2/4/8.
  - **Files**: `attention-proto/benches/variant_overhead.rs`
  - **Done when**: Benchmark reports µs overhead per variant, % of base attention time
  - **Verify**: `cargo bench --bench variant_overhead`
  - **Commit**: `feat(proto7): add variant overhead benchmark`
  - _Requirements: FR-11, AC-5.3_
  - _Design: Proto 7 Benchmark_

- [ ] 4.21 Generate KB findings for Proto 7
  - **Do**: Emit 4+ findings: RoPE overhead (expected 2-5%), ALiBi overhead (<1%), GQA overhead (<1%), validation that all can be function-constant-specialized
  - **Files**: Update findings.jsonl
  - **Done when**: At least 4 findings emitted
  - **Verify**: `cat findings.jsonl | grep proto7 | wc -l` >= 4
  - **Commit**: `docs(proto7): add KB findings for variant overhead`
  - _Requirements: AC-5.4, AC-5.5, US-5_

## Phase 5: Ecosystem Prototypes (Defer to Week 3)

Focus: CubeCL and Burn integration. Non-blocking if they fail.

### Proto 5: CubeCL MSL Quality

- [ ] 5.1 Implement CubeCL matmul kernel (feature-gated)
  - **Do**: Behind `cubecl` feature, write simplified attention matmul in CubeCL #[cube] syntax, compile via Metal backend, extract generated MSL source
  - **Files**: `attention-proto/src/proto5_cubecl.rs` (feature-gated)
  - **Done when**: CubeCL kernel compiles, MSL source extracted
  - **Verify**: `cargo build --release --features cubecl`
  - **Commit**: `feat(proto5): implement CubeCL matmul for MSL quality comparison`
  - _Requirements: FR-12, AC-6.1_
  - _Design: Proto 5_

- [ ] 5.2 Analyze CubeCL vs hand-written MSL
  - **Do**: Diff generated MSL vs Proto 1 MSL: instruction count (xcrun metal -S), grep for simdgroup_matrix, register usage from GPU profiler, threadgroup memory
  - **Files**: Document in proto5_cubecl.rs comments
  - **Done when**: Analysis complete, documented
  - **Verify**: Manual inspection
  - **Commit**: `feat(proto5): analyze CubeCL-generated MSL quality`
  - _Requirements: AC-6.2_
  - _Design: Proto 5 Analysis_

- [ ] 5.3 Implement CubeCL comparison benchmark
  - **Do**: Side-by-side benchmark: CubeCL-generated kernel vs Proto 1 hand-written. Same N/D/tile. Report TFLOPS ratio.
  - **Files**: `attention-proto/benches/cubecl_comparison.rs` (feature-gated)
  - **Done when**: Benchmark runs, reports TFLOPS ratio (expected 30-60% slower for CubeCL)
  - **Verify**: `cargo bench --bench cubecl_comparison --features cubecl`
  - **Commit**: `feat(proto5): add CubeCL vs hand-written benchmark`
  - _Requirements: AC-6.3_
  - _Design: Proto 5_

- [ ] 5.4 Generate KB findings for Proto 5
  - **Do**: Emit 3+ findings: instruction count delta, simdgroup_matrix usage (no), TFLOPS ratio, recommendation on CubeCL viability
  - **Files**: Update findings.jsonl
  - **Done when**: At least 3 findings emitted
  - **Verify**: `cat findings.jsonl | grep proto5 | wc -l` >= 3
  - **Commit**: `docs(proto5): add KB findings for CubeCL MSL quality`
  - _Requirements: AC-6.4, AC-6.5, US-6_

### Proto 8: Burn Extension Trait

- [ ] 5.5 Define AttentionBackend trait (feature-gated)
  - **Do**: Behind `burn-ext` feature, define `trait AttentionBackend: Backend` with flash_attention method. Use newtype wrapper `MetalAttentionBackend(CubeCL<MetalRuntime>)` to avoid orphan rule.
  - **Files**: `attention-proto/src/proto8_burn.rs` (feature-gated)
  - **Done when**: Trait compiles, newtype pattern compiles
  - **Verify**: `cargo check --features burn-ext`
  - **Commit**: `feat(proto8): define AttentionBackend extension trait`
  - _Requirements: FR-13, AC-7.1, AC-7.2_
  - _Design: Proto 8_

- [ ] 5.6 Implement AttentionBackend for MetalAttentionBackend
  - **Do**: Implement flash_attention method by calling Proto 1 kernel via FFI, convert Burn tensors to Metal buffers, dispatch, wrap output back into Burn tensor
  - **Files**: `attention-proto/src/proto8_burn.rs`
  - **Done when**: Implementation compiles, can dispatch Proto 1 kernel
  - **Verify**: `cargo build --release --features burn-ext`
  - **Commit**: `feat(proto8): implement AttentionBackend for Metal backend`
  - _Requirements: AC-7.3_
  - _Design: Proto 8_

- [ ] 5.7 Implement Burn extension trait test and benchmark
  - **Do**: Test: verify trait method dispatches correctly. Benchmark: measure dispatch overhead (Burn API → Metal kernel) vs direct Proto 1.
  - **Files**: `attention-proto/tests/integration.rs`, `attention-proto/benches/burn_extension.rs` (feature-gated)
  - **Done when**: Test passes, benchmark reports overhead 1-5 µs
  - **Verify**: `cargo test --release --features burn-ext -- proto8::test_trait_dispatches && cargo bench --bench burn_extension --features burn-ext`
  - **Commit**: `feat(proto8): add Burn extension trait test and benchmark`
  - _Requirements: AC-7.3, AC-7.4_
  - _Design: Proto 8_

- [ ] 5.8 Generate KB findings for Proto 8
  - **Do**: Emit 2+ findings: trait viability (yes/no), dispatch overhead, required Burn version, code complexity (lines of boilerplate, unsafe blocks)
  - **Files**: Update findings.jsonl
  - **Done when**: At least 2 findings emitted
  - **Verify**: `cat findings.jsonl | grep proto8 | wc -l` >= 2
  - **Commit**: `docs(proto8): add KB findings for Burn extension trait`
  - _Requirements: AC-7.5, US-7_

## Phase 6: Quality Gates & Synthesis

Focus: Validate all prototypes, consolidate findings, prepare for KB ingestion.

- [ ] 6.1 Run full correctness test suite
  - **Do**: Run all integration tests with Metal Shader Validation enabled, serial execution (--test-threads=1)
  - **Files**: All tests
  - **Done when**: All tests pass
  - **Verify**: `MTL_SHADER_VALIDATION=1 cargo test --release -- --test-threads=1`
  - **Commit**: `test(all): verify all prototypes pass correctness tests`
  - _Requirements: NFR-17, NFR-10_

- [ ] 6.2 Run full benchmark suite
  - **Do**: Run all criterion benchmarks, generate JSON output
  - **Files**: All benchmarks
  - **Done when**: All benchmarks complete, bench.json generated
  - **Verify**: `cargo bench -- --output-format json > /Users/patrickkavanagh/gpu_kernel/attention-proto/bench.json`
  - **Commit**: `bench(all): run full benchmark suite`
  - _Requirements: FR-16_

- [ ] 6.3 Validate KB findings quality
  - **Do**: Parse findings.jsonl, verify: (1) count >= 20 (minimum), (2) each finding has confidence >= 0.5, (3) CV < 5% for High confidence, (4) all required prototypes have >= 2 findings each
  - **Files**: Script: `attention-proto/scripts/validate_findings.py`
  - **Done when**: Validation passes, count >= 40 (target)
  - **Verify**: `python3 /Users/patrickkavanagh/gpu_kernel/attention-proto/scripts/validate_findings.py findings.jsonl`
  - **Commit**: `test(kb): validate KB findings quality and count`
  - _Requirements: Success Metrics (40-60 findings)_

- [ ] 6.4 Memory leak stress test
  - **Do**: Run leak detection test for each prototype (1000 iterations), verify currentAllocatedSize growth < 1%
  - **Files**: `attention-proto/tests/integration.rs`
  - **Done when**: Leak tests pass for all prototypes
  - **Verify**: `cargo test --release -- --ignored test_no_metal_memory_leak --test-threads=1`
  - **Commit**: `test(all): verify no Metal memory leaks`
  - _Requirements: NFR-11_

- [ ] 6.5 Synthesize findings into architecture recommendations
  - **Do**: Write synthesis document analyzing all findings, identifying: (1) optimal tile sizes per D, (2) recommended dispatch strategy (constants vs stitching), (3) PagedAttention viability, (4) linear attention crossover point, (5) variant overhead summary, (6) ecosystem tool recommendations (CubeCL, Burn)
  - **Files**: `attention-proto/SYNTHESIS.md`
  - **Done when**: Document complete, addresses all 8 prototypes
  - **Verify**: Manual review
  - **Commit**: `docs(synthesis): add architecture recommendations from prototype findings`
  - _Requirements: Success Metrics (stretch goal: design-breaking constraint)_

- [ ] 6.6 Ingest findings into GPU Forge KB
  - **Do**: Run kb add for each finding in findings.jsonl, verify KB count increases from 1,124 to 1,164+ (assuming 40 findings)
  - **Files**: findings.jsonl → GPU Forge KB
  - **Done when**: All findings ingested, KB query returns new attention-proto findings
  - **Verify**: `cat findings.jsonl | while read line; do echo "$line" | gpu-forge:knowledge add; done && gpu-forge:knowledge query "attention M4 tile size"`
  - **Commit**: `docs(kb): ingest all prototype findings into GPU Forge`
  - _Requirements: FR-18, Success Metrics_

- [ ] 6.7 Create PR with all prototypes and findings
  - **Do**: Push branch attention-proto-impl, create PR with gh CLI, description includes: (1) summary of 8 prototypes, (2) KB findings count, (3) key recommendations, (4) link to SYNTHESIS.md
  - **Files**: All attention-proto files
  - **Done when**: PR created, CI passes (if self-hosted M4 runner available)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel && git add attention-proto && git commit -m "feat(attention-proto): complete 8 prototypes with 40+ KB findings" && git push -u origin attention-proto-impl && gh pr create --title "feat: attention-proto — 8 GPU kernel prototypes for trait Attention<Q,K,V>" --body "$(cat attention-proto/SYNTHESIS.md)"`
  - **Commit**: N/A (PR creation)
  - _Requirements: All US, all FR_

## Notes

**POC shortcuts taken** (Phase 1-5):
- CPU references use naive O(N²) algorithms (acceptable for correctness validation)
- Error handling is basic (panic on Metal errors, no retry logic)
- No production logging or telemetry
- Single precision (FP16/FP32) only, no FP64 GPU kernels
- M4-only target, no multi-generation GPU support
- Feature flags for CubeCL/Burn may fail without blocking other prototypes

**Production TODOs** (if prototypes → production trait):
- Robust error handling with Result types
- Structured logging (tracing crate)
- Support M1/M2/M3 via Metal feature set detection
- Dynamic tile size selection based on hardware capabilities
- Async dispatch with command buffer pooling
- Production-quality documentation and examples
- Comprehensive test matrix (all edge cases in QA.md)
- CI/CD pipeline with self-hosted M4 runner
- Metal 4 cooperative tensor migration path
