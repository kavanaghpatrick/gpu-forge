---
spec: gpu-perf-phase2
phase: tasks
total_tasks: 14
created: 2026-02-15
generated: auto
---

# Tasks: gpu-perf-phase2

## Phase 1: Make It Work (POC)

Focus: GPU argmax + Private buffers working end-to-end. Skip fused kernel, accept hardcoded values.

- [x] 1.1 Implement GPU argmax Metal kernel
  - **Do**: Create `argmax.metal` with 2-stage parallel reduction. Stage 1: `argmax_reduce` -- 256 threads/group, each processes 4 elements, threadgroup shared memory reduction to 1 (max_val, max_idx) pair per group. Stage 2: `argmax_final` -- single threadgroup reduces partial results. Output struct: `{ uint32_t token_id; float max_val; }`.
  - **Files**: `crates/metal-attention-kernels/shaders/argmax.metal`
  - **Done when**: Metal shader compiles as part of metallib build
  - **Verify**: `cargo build -p metal-attention-kernels`
  - **Commit**: `feat(kernels): add GPU argmax 2-stage parallel reduction kernel`
  - _Requirements: FR-1_
  - _Design: Component 1_

- [x] 1.2 Add alloc_buffer_private and argmax Rust dispatch
  - **Do**: Add `alloc_buffer_private()` to `buffer.rs` using `MTLResourceOptions::StorageModePrivate`. Add `encode_argmax()` method to `GpuForwardPass` that encodes both stages into an existing compute encoder. Allocate 3 new buffers in `from_gguf()`: `argmax_partial_vals` (192B Private), `argmax_partial_idxs` (192B Private), `argmax_result` (8B Shared). Prewarm `argmax_reduce` and `argmax_final` PSOs.
  - **Files**: `crates/metal-attention-kernels/src/buffer.rs`, `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: `encode_argmax()` method exists and compiles
  - **Verify**: `cargo build -p metal-attention`
  - **Commit**: `feat(gpu): add alloc_buffer_private and encode_argmax dispatch`
  - _Requirements: FR-2, FR-4_
  - _Design: Component 1, Component 2_

- [x] 1.3 Add forward_token_greedy and wire into CLI
  - **Do**: Add `forward_token_greedy(&mut self, token_id: u32) -> Result<u32, String>` that encodes the full forward pass + argmax into a single command buffer, reads back only the 8-byte argmax_result. Keep `forward_token()` unchanged (returns Vec<f32>). Update `run_inference_gpu()` and `run_bench_gpu()` in `main.rs` to use `forward_token_greedy()`.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`, `src/main.rs`, `crates/metal-attention/src/lib.rs`
  - **Done when**: `cargo run -- run --gpu -m <model> -p "Hello"` uses GPU argmax path
  - **Verify**: `cargo run -- run --gpu -m ~/models/smollm-135m-q4_0.gguf -p "The meaning of life is" -n 20` produces coherent text
  - **Commit**: `feat(gpu): add forward_token_greedy with GPU-side argmax`
  - _Requirements: FR-2, FR-3_
  - _Design: Component 1_

- [x] 1.4 Switch scratch buffers to StorageModePrivate
  - **Do**: In `from_gguf()`, check `GPU_DEBUG` env var. If not set, allocate scratch_q through scratch_residual and logits_buf with `alloc_buffer_private()`. hidden_a stays Shared (CPU write). hidden_b can go Private (only GPU reads/writes -- rmsnorm output, matvec input). If `GPU_DEBUG` is set, keep all Shared.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: Non-debug path uses Private buffers, debug path unchanged
  - **Verify**: `cargo run -- bench --gpu -m ~/models/smollm-135m-q4_0.gguf --seq-lengths 1 --gen-length 64 --iterations 3` shows tok/s improvement. `GPU_DEBUG=1 cargo run -- run --gpu -m ~/models/smollm-135m-q4_0.gguf -p "Test" -n 5` still works.
  - **Commit**: `perf(gpu): switch scratch buffers to StorageModePrivate`
  - _Requirements: FR-5_
  - _Design: Component 2_

- [x] 1.5 POC Checkpoint -- benchmark argmax + Private
  - **Do**: Run GPU bench with argmax + Private buffers. Compare against 257 tok/s baseline. Document results in `.progress.md`.
  - **Done when**: Benchmark shows measurable improvement and output is correct
  - **Verify**: `cargo run -- bench --gpu -m ~/models/smollm-135m-q4_0.gguf --seq-lengths 1 --gen-length 64 --iterations 5`
  - **Commit**: `perf(gpu): POC checkpoint - argmax + private buffers`

## Phase 2: Fused Kernel

After argmax+Private validated, implement the fused RMSNorm+Matvec kernel.

- [x] 2.1 Implement fused RMSNorm+Matvec Q4_0 Metal kernel
  - **Do**: Create `rmsnorm_matvec_q4_0.metal`. Each threadgroup: (1) compute RMS via simd_sum across input, (2) compute dot product of dequant(weight_row) * (input/rms * norm_weight). 32 threads per threadgroup, 1 threadgroup per output row. Buffer bindings: input(0), norm_weight(1), matvec_weight_q4_0(2), output(3), out_dim(4), in_dim(5), eps(6).
  - **Files**: `crates/metal-attention-kernels/shaders/rmsnorm_matvec_q4_0.metal`
  - **Done when**: Shader compiles in metallib
  - **Verify**: `cargo build -p metal-attention-kernels`
  - **Commit**: `feat(kernels): add fused rmsnorm+matvec Q4_0 kernel`
  - _Requirements: FR-6_
  - _Design: Component 3_

- [x] 2.2 Implement fused RMSNorm+Matvec F32 Metal kernel
  - **Do**: Create `rmsnorm_matvec_f32.metal`. Same structure as Q4_0 variant but with F32 weight access instead of block dequantization. For lm_head with tied embeddings.
  - **Files**: `crates/metal-attention-kernels/shaders/rmsnorm_matvec_f32.metal`
  - **Done when**: Shader compiles in metallib
  - **Verify**: `cargo build -p metal-attention-kernels`
  - **Commit**: `feat(kernels): add fused rmsnorm+matvec F32 kernel`
  - _Requirements: FR-7_
  - _Design: Component 3_

- [x] 2.3 Wire fused kernel into forward pass
  - **Do**: Add `encode_fused_rmsnorm_matvec_q4_0()` and `encode_fused_rmsnorm_matvec_f32()` methods. Replace paired rmsnorm+matvec calls in the layer loop (attention Q/K/V/O and FFN gate/up/down = 7 per layer, 210 total). Keep separate rmsnorm+matvec for final lm_head (out_dim=49152). Prewarm new PSOs.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: forward_token uses fused kernels for all per-layer projections
  - **Verify**: `cargo run -- run --gpu -m ~/models/smollm-135m-q4_0.gguf -p "The meaning of life is" -n 20` produces same output as before
  - **Commit**: `perf(gpu): wire fused rmsnorm+matvec into forward pass`
  - _Requirements: FR-8_
  - _Design: Component 3_

- [x] 2.4 Benchmark full optimization stack
  - **Do**: Run GPU bench with all three optimizations. Compare against 257 tok/s baseline and POC checkpoint. Document final numbers in `.progress.md`.
  - **Done when**: Benchmark shows combined improvement toward 389+ target
  - **Verify**: `cargo run -- bench --gpu -m ~/models/smollm-135m-q4_0.gguf --seq-lengths 1 --gen-length 64 --iterations 5`
  - **Commit**: `perf(gpu): benchmark full optimization stack`

## Phase 3: Testing

- [x] 3.1 Unit tests for GPU argmax kernel
  - **Do**: Add test in `gpu_forward_pass.rs` or new `tests/argmax_test.rs`: (1) known vector argmax matches CPU, (2) all-same values returns index 0, (3) max at last index, (4) NaN handling (NaN not selected), (5) inf/-inf handling, (6) vocab_size=49152 random vector matches CPU argmax. Property test: `assert!(result_token_id < vocab_size)`.
  - **Files**: `crates/metal-attention/tests/argmax_test.rs` or inline
  - **Done when**: All argmax tests pass
  - **Verify**: `cargo test -p metal-attention argmax`
  - **Commit**: `test(gpu): add GPU argmax kernel unit tests`
  - _Requirements: AC-1.3, AC-1.4_

- [x] 3.2 Numerical validation for fused kernel
  - **Do**: Add tests comparing fused vs separate rmsnorm+matvec for all SmolLM dimensions: (576->576), (576->192), (576->1536), (1536->576). Tolerance: 1e-4 per element. Also test 30-layer accumulated drift: run full forward pass with fused vs separate and compare final logits (tolerance 1e-3).
  - **Files**: `crates/metal-attention/tests/fused_kernel_test.rs` or inline
  - **Done when**: All numerical validation tests pass within tolerance
  - **Verify**: `cargo test -p metal-attention fused`
  - **Commit**: `test(gpu): add fused kernel numerical validation`
  - _Requirements: AC-3.1, AC-3.2_

- [x] 3.3 Run full regression suite
  - **Do**: Run all 297 existing tests. Verify no regressions.
  - **Done when**: All 297 tests pass
  - **Verify**: `cargo test --workspace`
  - **Commit**: `test(gpu): verify no regression in 297 existing tests` (if fixes needed)
  - _Requirements: AC-2.4, NFR-2_

## Phase 4: Quality Gates

- [x] 4.1 Local quality check
  - **Do**: Run clippy, formatting, and all tests. Fix any warnings or errors.
  - **Verify**: `cargo clippy --workspace -- -D warnings && cargo fmt --check && cargo test --workspace`
  - **Done when**: All commands pass cleanly
  - **Commit**: `fix(gpu): address clippy and formatting issues` (if needed)

- [x] 4.2 Create PR and verify CI
  - **Do**: Push branch, create PR with performance results summary. Include before/after tok/s numbers.
  - **Verify**: `gh pr checks --watch` all green
  - **Done when**: PR ready for review with benchmark data
  - **Commit**: n/a (PR creation, not a commit)

## Notes

- **POC shortcuts taken**: Argmax hardcodes 256 threads/group and 4 elements/thread. Could be tuned later.
- **Production TODOs**: Consider argmax with temperature sampling (top-k/top-p on GPU). Consider async pipeline if still below 389 tok/s.
- **Risk**: Fused kernel numerical drift is the highest-risk item. If drift exceeds tolerance, can ship argmax + Private alone for ~40-80 tok/s improvement.
