---
spec: gpu-perf-phase3
phase: tasks
total_tasks: 17
created: 2026-02-15
generated: auto
---

# Tasks: gpu-perf-phase3

## Phase 1: Make It Work (POC)

Focus: Get multi-token prefill working end-to-end. Skip tests, accept hardcoded values.

- [ ] 1.1 Create multi_token_matvec_q4_0.metal production shader
  - **Do**: Create new shader file with two kernels copied from `bandwidth_test.metal:225-281`. Kernel 1: `multi_token_matvec_q4_0` (exact copy from bandwidth_test.metal). Kernel 2: `multi_token_matvec_q4_0_accumulate` (same structure but `output[tok * out_dim + row] += sum` instead of `= sum`). Both kernels: 256 threads, 8 simdgroups, 8 rows/TG, buffer bindings weight[0] input[1] output[2] out_dim[3] in_dim[4] batch_size[5].
  - **Files**: `crates/metal-attention-kernels/shaders/multi_token_matvec_q4_0.metal`
  - **Done when**: New shader file compiles with `cargo build --release`
  - **Verify**: `cargo build --release` succeeds without shader compilation errors
  - **Commit**: `feat(shader): add multi-token matvec Q4_0 production kernels`
  - _Requirements: FR-1, FR-2_
  - _Design: Component 1_

- [ ] 1.2 Prewarm multi-token PSOs in from_gguf()
  - **Do**: In `gpu_forward_pass.rs:from_gguf()`, add `PsoKey::simple("multi_token_matvec_q4_0")` and `PsoKey::simple("multi_token_matvec_q4_0_accumulate")` to the `pso_keys` vec (around line 237). Run `cargo clean -p metal-attention-kernels` first if shader not picked up.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: Both PSOs prewarmed at startup without errors
  - **Verify**: `cargo build --release` succeeds, no PSO prewarm warnings
  - **Commit**: `feat(pso): prewarm multi-token matvec PSOs`
  - _Requirements: FR-10_
  - _Design: Component 4_

- [ ] 1.3 Add encode_multi_token_matvec_q4_0 helper
  - **Do**: Add method `encode_multi_token_matvec_q4_0(&self, encoder, weight_buf, input_buf, output_buf, out_dim, in_dim, batch_size)` to GpuForwardPass. Follow exact pattern of `encode_matvec_q4_0` but use PSO `"multi_token_matvec_q4_0"`, add `batch_size` as buffer(5), keep 256-thread/8-row dispatch geometry with `grid = ceil(out_dim/8)`.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: Helper compiles and follows existing encode pattern
  - **Verify**: `cargo build --release` succeeds
  - **Commit**: `feat(encode): add multi-token matvec Q4_0 dispatch helper`
  - _Requirements: FR-7_
  - _Design: Component 4_

- [ ] 1.4 Add encode_multi_token_matvec_q4_0_accumulate helper
  - **Do**: Same as 1.3 but for `"multi_token_matvec_q4_0_accumulate"` PSO. Identical buffer bindings. The accumulate variant reads-then-writes output buffer.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: Helper compiles
  - **Verify**: `cargo build --release` succeeds
  - **Commit**: `feat(encode): add multi-token matvec Q4_0 accumulate dispatch helper`
  - _Requirements: FR-7_
  - _Design: Component 4_

- [ ] 1.5 Implement forward_prompt() — batch buffer allocation and embedding
  - **Do**: Add `pub fn forward_prompt(&mut self, token_ids: &[u32]) -> Result<u32, String>` to GpuForwardPass. Step 1: Validate token_ids (not empty, within vocab range, position + len <= 2048). Step 2: Calculate batch_size = token_ids.len(). Step 3: Allocate batch buffers — batch_hidden_a (Shared, needs CPU write for embed), batch_hidden_b/batch_q/batch_k/batch_v/batch_attn_out/batch_gate/batch_up/batch_silu (Private). Step 4: CPU embed_lookup loop — for each token, memcpy embedding row into batch_hidden_a at offset `tok * hidden_size * 4`. Step 5: Create command buffer + compute encoder. Leave layer loop empty for now, just add final norm + lm_head + argmax on last token (copy last token hidden to single-token hidden_a buffer, reuse existing final logic).
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: Method compiles, allocates buffers, does embeddings, produces output (wrong output OK since layer loop empty)
  - **Verify**: `cargo build --release` succeeds
  - **Commit**: `feat(prefill): scaffold forward_prompt with batch buffer allocation`
  - _Requirements: FR-3, FR-4_
  - _Design: Component 2, Component 3_

- [ ] 1.6 Implement forward_prompt() — layer loop with batched matvec
  - **Do**: Fill in the layer loop body. For each layer: (a) Per-token attn RMSNorm: loop `0..batch_size`, call `encode_rmsnorm` with input=batch_hidden_a offset `tok*H*4`, output=batch_hidden_b offset `tok*H*4`. (b) Batched Q/K/V: call `encode_multi_token_matvec_q4_0` three times for Q, K, V weights with batch_hidden_b as input, batch_q/k/v as output. (c) Per-token RoPE: loop `0..batch_size`, call `encode_rope` on batch_q offset `tok*Q*4` and batch_k offset `tok*KV*4` with position = `self.position + tok`. (d) Per-token KV append + attention: loop `0..batch_size`, call `kv_caches.cache_mut(layer).encode_kv_append` with batch_k/v at offset, then `encode_decode_attention` with batch_q/attn_out at offset, kv_len = initial_pos + tok + 1. (e) Batched O-proj accumulate: `encode_multi_token_matvec_q4_0_accumulate` with batch_attn_out -> batch_hidden_a. (f) Per-token FFN RMSNorm: same loop as (a) but with ffn_norm. (g) Batched gate/up: two `encode_multi_token_matvec_q4_0` calls. (h) Per-token SiLU: loop, call `encode_ffn_silu` with offsets on batch_gate/up/silu. (i) Batched down accumulate: `encode_multi_token_matvec_q4_0_accumulate` with batch_silu -> batch_hidden_a.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: Full layer loop compiles. Note: per-token ops use buffer offsets via `set_buffer(encoder, buf, offset_bytes, index)`.
  - **Verify**: `cargo build --release` succeeds
  - **Commit**: `feat(prefill): implement layer loop with batched + per-token ops`
  - _Requirements: FR-5, FR-6, FR-7, FR-8_
  - _Design: Component 3_

- [ ] 1.7 Implement forward_prompt() — final logits and position update
  - **Do**: After layer loop: (a) Copy last token's hidden state from batch_hidden_a (offset `(batch_size-1)*H*4`) to single-token hidden_a buffer (for final norm/lm_head reuse). Or better: use buffer offset directly. (b) Encode rmsnorm on last token's hidden -> hidden_b (single token). (c) Encode lm_head matvec (Q8_0/F32/Q4_0 depending on weight type) from hidden_b -> logits_buf. (d) Encode argmax on logits_buf. (e) End encoding, commit, wait, validate. (f) Read 4-byte argmax result. (g) `self.position += batch_size`. (h) Return Ok(token_id).
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: `forward_prompt()` is complete end-to-end
  - **Verify**: `cargo build --release` succeeds
  - **Commit**: `feat(prefill): complete forward_prompt with final logits and argmax`
  - _Requirements: FR-9_
  - _Design: Component 3_

- [ ] 1.8 Wire forward_prompt into CLI run_inference_gpu
  - **Do**: In `src/main.rs:run_inference_gpu()`, replace the sequential prefill loop (`for &tok in &prompt_tokens[..len-1] { gpu.forward_token(tok)?; }` + `gpu.forward_token_greedy(last)`) with a single `gpu.forward_prompt(&prompt_tokens)?` call. Keep the decode loop unchanged.
  - **Files**: `src/main.rs`
  - **Done when**: `cargo run --release -- run --gpu -m models/SmolLM-135M.Q4_0.gguf -p "Hello world"` produces output
  - **Verify**: Run with a short prompt, verify output text is coherent and no crash/NaN
  - **Commit**: `feat(cli): use forward_prompt for GPU prefill`
  - _Requirements: FR-11_
  - _Design: Component 5_

- [ ] 1.9 POC Checkpoint — verify correctness and measure prefill
  - **Do**: Run `cargo run --release -- run --gpu -m models/SmolLM-135M.Q4_0.gguf -p "The quick brown fox" -n 32` and compare output to sequential prefill. Verify: (1) Output text matches or is very similar. (2) No NaN in any buffer. (3) Measure prefill tok/s from stderr timing output.
  - **Done when**: forward_prompt produces correct output and reports timing
  - **Verify**: Compare output tokens to sequential path; prefill tok/s reported in stderr
  - **Commit**: `feat(prefill): POC multi-token prefill verified`

## Phase 2: Refactoring

After POC validated, clean up code.

- [ ] 2.1 Add per-token RMSNorm helper with buffer offset
  - **Do**: Extract per-token RMSNorm loop into helper: `encode_rmsnorm_batched(&self, encoder, input_buf, weight_buf, output_buf, batch_size)` that loops internally and sets buffer offsets. Reduces code duplication in forward_prompt layer loop.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: forward_prompt uses the new helper, code is cleaner
  - **Verify**: `cargo build --release` succeeds, output unchanged
  - **Commit**: `refactor(rmsnorm): extract batched per-token RMSNorm helper`
  - _Design: Component 4_

- [ ] 2.2 Add per-token SiLU helper with buffer offset
  - **Do**: Extract per-token SiLU loop into helper: `encode_ffn_silu_batched(&self, encoder, batch_gate, batch_up, batch_silu, batch_size)` that loops and sets buffer offsets for each token.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: forward_prompt uses the new helper
  - **Verify**: `cargo build --release` succeeds, output unchanged
  - **Commit**: `refactor(silu): extract batched per-token SiLU helper`
  - _Design: Component 4_

- [ ] 2.3 Improve batch buffer allocation with reuse
  - **Do**: Add optional fields `batch_bufs: Option<BatchBuffers>` to GpuForwardPass struct. On first `forward_prompt` call, allocate and store. On subsequent calls, reuse if batch_size <= allocated size, else reallocate. This avoids per-call allocation overhead.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: Second `forward_prompt` call reuses buffers without new allocation
  - **Verify**: `cargo build --release` succeeds
  - **Commit**: `refactor(buffers): cache batch buffers for reuse across forward_prompt calls`
  - _Design: Component 2_

## Phase 3: Testing

- [ ] 3.1 Add correctness test: forward_prompt vs sequential forward_token
  - **Do**: In `tests/e2e.rs` or new test file, create test that runs SmolLM-135M with a 32-token prompt via both (1) sequential `forward_token` x32 + argmax and (2) `forward_prompt(&tokens)`. Assert both produce the same output token ID.
  - **Files**: `tests/e2e.rs` (or new `tests/prefill_correctness.rs`)
  - **Done when**: Test passes, proving forward_prompt matches sequential path
  - **Verify**: `cargo test --release --test e2e -- prefill` (or equivalent)
  - **Commit**: `test(prefill): verify forward_prompt matches sequential forward_token`
  - _Requirements: NFR-3_

- [ ] 3.2 Add prefill benchmark to inference.rs
  - **Do**: In `benches/inference.rs`, add `bench_gpu_prefill` function. Load SmolLM-135M, generate synthetic 64-token prompt, benchmark `forward_prompt(&tokens)` with criterion. Report throughput as `Elements(64)`.
  - **Files**: `benches/inference.rs`
  - **Done when**: `cargo bench --bench inference -- prefill` reports tok/s
  - **Verify**: Benchmark runs and reports >= 1000 tok/s
  - **Commit**: `bench(prefill): add multi-token prefill benchmark`
  - _Requirements: NFR-1_

- [ ] 3.3 Run full test suite
  - **Do**: Run `cargo test --workspace` to verify existing tests still pass (no regression from forward_prompt addition)
  - **Files**: All test files
  - **Done when**: All tests pass
  - **Verify**: `cargo test --workspace` exit code 0
  - **Commit**: `test(suite): verify no regression from forward_prompt`
  - _Requirements: NFR-2_

## Phase 4: Quality Gates

- [ ] 4.1 Local quality check
  - **Do**: Run `cargo clippy --workspace --all-targets` and `cargo fmt --all -- --check`. Fix any issues.
  - **Verify**: Both commands exit 0
  - **Done when**: No clippy warnings, no format issues
  - **Commit**: `fix(quality): address clippy and fmt issues` (if needed)

- [ ] 4.2 Create PR with benchmark results
  - **Do**: Push branch, create PR with `gh pr create`. Include in body: (1) prefill tok/s benchmark result, (2) decode tok/s to show no regression, (3) correctness test result, (4) summary of the three changes (kernels, forward_prompt, CLI).
  - **Verify**: `gh pr checks --watch` all green
  - **Done when**: PR created and CI passing
  - **Commit**: N/A (PR creation)

## Notes

**POC shortcuts taken**:
- Per-call batch buffer allocation (refactored in 2.3)
- No batched RMSNorm/SiLU kernels (per-token is cheap enough)
- Single command buffer for entire prefill (may need splitting for very long prompts)

**Key implementation details**:
- `set_buffer(encoder, buf, offset_bytes, index)` is the mechanism for per-token buffer slicing within batch buffers
- KV cache append is sequential due to causal constraint (each token needs previous K/V)
- Only the LAST token gets lm_head + argmax (all other tokens only need hidden state for next layer)
- `cargo clean -p metal-attention-kernels` needed when adding new .metal shader files
