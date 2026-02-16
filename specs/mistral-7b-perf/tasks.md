---
spec: mistral-7b-perf
phase: tasks
total_tasks: 24
created: 2026-02-16
generated: auto
---

# Tasks: mistral-7b-perf

## Phase 1: Q6_K Kernel (POC)

Focus: Native Q6_K matvec kernel + integration. Proves Q6_K works, eliminates 404 MB/token lm_head bandwidth.

- [x] 1.1 Create matvec_q6_k.metal shader
  - **Do**: Create `crates/metal-attention-kernels/shaders/matvec_q6_k.metal`. Define `BlockQ6_K` struct (ql[128], qh[64], scales[16], d as half). Implement `matvec_q6_k` kernel with 256 threads (8 simdgroups, ROWS_PER_TG=8). Inner loop: stride over Q6_K blocks per row, process two 128-element chunks, reconstruct 6-bit values from ql+qh, multiply by scale*d, float4 vectorized dot products, simd_sum reduction. Buffer binding: weight=0, input=1, output=2, out_dim=3, in_dim=4. Follow `matvec_q8_0.metal` structure exactly.
  - **Files**: `crates/metal-attention-kernels/shaders/matvec_q6_k.metal`
  - **Done when**: File compiles via `cargo clean -p metal-attention-kernels && cargo build -p metal-attention-kernels`
  - **Verify**: `cargo build -p metal-attention-kernels 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `feat(kernels): add Q6_K native matvec Metal shader`
  - _Requirements: FR-1_
  - _Design: Component 1_

- [x] 1.2 Create multi_token_matvec_q6_k.metal shader
  - **Do**: Create batched variant following `multi_token_matvec_q4_0.metal`. Add batch_size param at buffer(5). Outer loop over tokens, inner loop over Q6_K blocks. Same dequant math as 1.1. Two variants: overwrite and accumulate. Both reuse weight data from SLC across tokens.
  - **Files**: `crates/metal-attention-kernels/shaders/multi_token_matvec_q6_k.metal`
  - **Done when**: File compiles alongside matvec_q6_k.metal
  - **Verify**: `cargo build -p metal-attention-kernels 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `feat(kernels): add batched Q6_K matvec for prefill and speculative verify`
  - _Requirements: FR-2_
  - _Design: Component 1_

- [x] 1.3 Add lm_head_q6k field to GpuWeightStore
  - **Do**: In `gpu_weight_store.rs`: (1) Add `lm_head_q6k: Option<Retained<ProtocolObject<dyn MTLBuffer>>>` field. (2) Add `lm_head_q6k()` accessor returning `Option<&ProtocolObject<dyn MTLBuffer>>`. (3) In `from_gguf()`, when `output.weight` is Q6_K, create raw Q6_K buffer via `make_weight_buffer()` and store in `lm_head_q6k`. Keep existing F32 dequant path for lm_head fallback. Update struct constructor.
  - **Files**: `crates/metal-attention/src/gpu_weight_store.rs`
  - **Done when**: `cargo build -p metal-attention` succeeds; lm_head_q6k populated when loading Mistral-7B
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `feat(weights): store Q6_K lm_head as native quantized buffer`
  - _Requirements: FR-3_
  - _Design: Component 1_

- [x] 1.4 Add encode_matvec_q6_k to GpuForwardPass
  - **Do**: In `gpu_forward_pass.rs`: (1) Add `encode_matvec_q6_k()` method following `encode_matvec_q8_0()` pattern -- set_buffer for weight/input/output/out_dim/in_dim, dispatch with threadgroups=ceil(out_dim/8). (2) Add `encode_multi_token_matvec_q6_k()` for batched variant with batch_size param. (3) Add PSO prewarm for `matvec_q6_k` and `multi_token_matvec_q6_k` in `from_gguf()`. (4) Update lm_head dispatch chain in `forward_token()`: add Q6_K check before Q8_0 check. (5) Update lm_head dispatch in `forward_prompt()` similarly.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: `cargo build -p metal-attention` succeeds; forward_token with Mistral-7B uses Q6_K kernel (visible in GPU_DEBUG=1 output)
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `feat(inference): integrate Q6_K matvec kernel for lm_head dispatch`
  - _Requirements: FR-7_
  - _Design: Component 1_

- [x] 1.5 Q6_K kernel correctness test
  - **Do**: Add test `test_matvec_q6_k_gpu_vs_cpu` in `tests/gpu_correctness.rs`. Create synthetic Q6_K blocks with known values, run through GPU kernel, compare against Rust `dequantize_q6_k_to_f32` + F32 dot product. Test dimensions: (256, 256) small case and (4096, 32000) Mistral lm_head dims. Tolerance: atol=5e-2 for matvec output (quantization error).
  - **Files**: `crates/metal-attention/tests/gpu_correctness.rs`
  - **Done when**: Test passes with `cargo test --test gpu_correctness -- test_matvec_q6_k --test-threads=1`
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test gpu_correctness -- test_matvec_q6_k --test-threads=1 2>&1 | tail -5`
  - **Commit**: `test(q6k): add GPU vs CPU correctness test for Q6_K matvec`
  - _Requirements: AC-1.1_
  - _Design: Component 1_

- [ ] 1.6 POC Checkpoint: Q6_K end-to-end validation
  - **Do**: Run Mistral-7B inference with Q6_K kernel active. Verify: (1) lm_head buffer is 108 MB (not 512 MB), (2) forward_token produces coherent text, (3) no crash or NaN. Log tok/s to confirm improvement.
  - **Files**: None (integration test)
  - **Done when**: Mistral-7B generates coherent text with Q6_K lm_head; buffer size assertion passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test gpu_correctness -- test_mistral --ignored --test-threads=1 2>&1 | tail -10`
  - **Commit**: `feat(q6k): complete Q6_K kernel POC -- verified on Mistral-7B`

## Phase 2: Page-Aligned Weight Buffers

Focus: WeightBuffer refactor enabling zero-copy for all GGUF tensors via page-offset trick.

- [ ] 2.1 Define WeightBuffer struct
  - **Do**: In `gpu_weight_store.rs`: (1) Define `pub struct WeightBuffer { pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>, pub offset: usize }`. (2) Add `impl WeightBuffer` with `new(buffer, offset)` and `zero_offset(buffer)` constructors. (3) Replace `Retained<ProtocolObject<dyn MTLBuffer>>` in `AttnProjBuffers`, `FfnBuffers` with `WeightBuffer`. (4) Update all field accesses in the file. Keep `lm_head`, `lm_head_q8`, `lm_head_q6k` as separate refactor for now.
  - **Files**: `crates/metal-attention/src/gpu_weight_store.rs`
  - **Done when**: `cargo build -p metal-attention` succeeds with WeightBuffer fields
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `refactor(weights): introduce WeightBuffer struct for page-aligned zero-copy`
  - _Requirements: FR-4_
  - _Design: Component 2_

- [ ] 2.2 Implement make_weight_buffer_aligned
  - **Do**: In `gpu_weight_store.rs`: (1) Create `make_weight_buffer_aligned(device, data, name, page_size, mmap_end)` that rounds pointer down to page boundary, computes offset, creates `newBufferWithBytesNoCopy` for the padded region, returns `WeightBuffer{buffer, offset}`. (2) Add mmap end address parameter from GgufFile. (3) Replace `make_weight_buffer()` calls with `make_weight_buffer_aligned()` for all quantized tensors. (4) Fallback to copy with offset=0 when padded region exceeds mmap bounds. (5) Track zero-copy vs copy counts for logging.
  - **Files**: `crates/metal-attention/src/gpu_weight_store.rs`
  - **Done when**: `cargo build -p metal-attention` succeeds; GPU_DEBUG=1 shows zero-copy for most tensors
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `feat(weights): page-aligned zero-copy buffer allocation with offset tracking`
  - _Requirements: FR-4, AC-2.1_
  - _Design: Component 2_

- [ ] 2.3 Propagate WeightBuffer offset through encode methods
  - **Do**: In `gpu_forward_pass.rs`: Update ALL `encode_matvec_*` methods to accept `&WeightBuffer` and pass `weight.offset` to `set_buffer()`. Update all call sites: `encode_matvec_q4_0`, `encode_matvec_q8_0`, `encode_matvec_f32`, `encode_matvec_q6_k`, and all multi-token variants. Each `set_buffer(encoder, &weight.buffer, weight.offset, idx)` call.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: All encode methods use WeightBuffer; `cargo build -p metal-attention` succeeds
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `refactor(inference): propagate WeightBuffer offset through all dispatch calls`
  - _Requirements: FR-5, AC-2.2_
  - _Design: Component 2_

- [ ] 2.4 Page-aligned correctness test
  - **Do**: Add test in `tests/gpu_correctness.rs`: load SmolLM with page-aligned path, run forward_token, compare output with the existing (copy-based) path. Must be bit-identical since the same bytes are read.
  - **Files**: `crates/metal-attention/tests/gpu_correctness.rs`
  - **Done when**: Test passes confirming zero-copy = copy-based output
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test gpu_correctness -- test_page_aligned --ignored --test-threads=1 2>&1 | tail -5`
  - **Commit**: `test(weights): verify page-aligned zero-copy produces identical output`
  - _Requirements: AC-2.3_
  - _Design: Component 2_

## Phase 3: Batch Prefill for Mistral-7B GQA

Focus: Extend forward_prompt() to work for Mistral-7B dimensions. Add Q6_K lm_head path.

- [ ] 3.1 Test forward_prompt on Mistral-7B
  - **Do**: Add integration test `test_mistral_forward_prompt` in `tests/gpu_correctness.rs`. Load Mistral-7B, call `forward_prompt()` with a 32-token prompt, verify it returns a valid token. If it fails, diagnose and fix the issue (likely Q6_K lm_head dispatch or buffer sizing).
  - **Files**: `crates/metal-attention/tests/gpu_correctness.rs`
  - **Done when**: `forward_prompt()` produces valid output for Mistral-7B 32-token prompt
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test gpu_correctness -- test_mistral_forward_prompt --ignored --test-threads=1 2>&1 | tail -10`
  - **Commit**: `test(prefill): verify batch prefill works for Mistral-7B GQA`
  - _Requirements: FR-6, AC-3.2_
  - _Design: Component 3_

- [ ] 3.2 Validate batch vs sequential consistency for GQA
  - **Do**: Add test `test_mistral_batch_vs_sequential`: run 32-token prompt through forward_prompt() (batched) and through 32 sequential forward_token() calls. Compare final argmax token. Must match for greedy decode.
  - **Files**: `crates/metal-attention/tests/gpu_correctness.rs`
  - **Done when**: Batched and sequential produce identical output token
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test gpu_correctness -- test_mistral_batch_vs_sequential --ignored --test-threads=1 2>&1 | tail -10`
  - **Commit**: `test(prefill): validate batch prefill matches sequential for Mistral-7B GQA`
  - _Requirements: AC-3.3, AC-3.4_
  - _Design: Component 3_

- [ ] 3.3 Add multi_token_matvec_q6_k dispatch in forward_prompt
  - **Do**: In `forward_prompt()`, update the lm_head dispatch to use `encode_multi_token_matvec_q6_k()` when `lm_head_q6k` is present. This applies to the final lm_head computation on the last token's hidden state. For speculative verify (forward_prompt_logits), this will apply to ALL token positions.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: forward_prompt uses Q6_K batched kernel for lm_head; `cargo build` succeeds
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `feat(prefill): use batched Q6_K matvec for lm_head in forward_prompt`
  - _Requirements: FR-7, AC-3.5_
  - _Design: Component 3_

- [ ] 3.4 Batch prefill benchmark
  - **Do**: Add benchmark `bench_prefill_mistral_128tok` in `benches/inference.rs`. Load Mistral-7B, time forward_prompt() with 128-token prompt. Report tok/s. Target: >= 100 tok/s.
  - **Files**: `crates/metal-attention/benches/inference.rs`
  - **Done when**: Benchmark runs and reports tok/s
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo bench -- bench_prefill_mistral 2>&1 | tail -5`
  - **Commit**: `bench(prefill): add Mistral-7B batch prefill throughput benchmark`
  - _Requirements: AC-3.1_
  - _Design: Component 3_

## Phase 4: Speculative Decoding Scaffolding

Focus: SpeculativeDecoder struct, KV cache rollback, forward_prompt_logits, CLI flags.

- [ ] 4.1 Add KV cache truncate and rollback_to
  - **Do**: (1) In `gpu_kv_cache.rs`: add `pub fn truncate(&mut self, new_len: usize)` to `GpuKVCache` -- assert new_len <= self.len, set self.len = new_len. Add `pub fn truncate_all(&mut self, new_len: usize)` to `GpuKVCacheSet` -- iterate caches, call truncate. (2) In `gpu_forward_pass.rs`: add `pub fn rollback_to(&mut self, position: usize)` -- assert position <= self.position, set self.position = position, call kv_caches.truncate_all(position). (3) Add `pub fn position(&self) -> usize` accessor.
  - **Files**: `crates/metal-attention/src/gpu_kv_cache.rs`, `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: `cargo build -p metal-attention` succeeds; rollback_to compiles
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `feat(kv-cache): add truncate and rollback_to for speculative decoding`
  - _Requirements: FR-8, FR-9_
  - _Design: Component 4_

- [ ] 4.2 Add forward_prompt_logits method
  - **Do**: In `gpu_forward_pass.rs`: add `pub fn forward_prompt_logits(&mut self, token_ids: &[u32]) -> Result<Vec<Vec<f32>>, String>`. Same layer-by-layer processing as `forward_prompt()`, but after all layers, run final RMSNorm + lm_head for EACH token position (not just last). Use multi_token_matvec for lm_head to compute all logits in one dispatch. Read back logits buffer for each position.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: Method compiles and returns logits for all positions
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `feat(inference): add forward_prompt_logits for speculative verification`
  - _Requirements: FR-10, AC-4.4_
  - _Design: Component 4_

- [ ] 4.3 Make sampling helpers pub(crate)
  - **Do**: In `sampling.rs`: make `softmax()` function `pub(crate)`. Add `pub(crate) fn sample_from_probs(probs: &[f32], rng: &mut SimpleRng) -> u32` that samples a token from a probability distribution using the rng. Update `lib.rs` re-exports if needed.
  - **Files**: `crates/metal-attention/src/sampling.rs`, `crates/metal-attention/src/lib.rs`
  - **Done when**: `cargo build -p metal-attention` succeeds; softmax and sample_from_probs accessible from speculative module
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `refactor(sampling): expose softmax and sample_from_probs as pub(crate)`
  - _Requirements: FR-11_
  - _Design: Component 4_

- [ ] 4.4 Create speculative.rs module
  - **Do**: Create `crates/metal-attention/src/speculative.rs` with `SpeculativeDecoder` struct. Implement: (1) `new(draft_path, target_path, n_draft)` -- loads two GpuForwardPass instances. (2) `generate(prompt, max_tokens, callback)` -- prefill both models, then loop: draft N tokens with draft model, verify with target `forward_prompt_logits`, accept/reject (greedy: accept if argmax matches), rollback on rejection, yield accepted tokens via callback. (3) `reset()` -- reset both models. (4) Internal `speculation_round()` method. Add `pub mod speculative;` to `lib.rs`.
  - **Files**: `crates/metal-attention/src/speculative.rs`, `crates/metal-attention/src/lib.rs`
  - **Done when**: `cargo build -p metal-attention` succeeds; SpeculativeDecoder struct compiles
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `feat(speculative): add SpeculativeDecoder struct with draft-verify loop`
  - _Requirements: FR-11, AC-4.1_
  - _Design: Component 4_

- [ ] 4.5 Add --draft CLI flags to inference
  - **Do**: In `src/inference.rs` or `src/main.rs` (wherever CLI args are parsed): add `--draft <PATH>` (Option<PathBuf>) and `--draft-tokens N` (usize, default 8) flags. When `--draft` is provided, create `SpeculativeDecoder` instead of single `GpuForwardPass`. Route generation through `speculative_decoder.generate()`. Print speculative stats at end (acceptance rate, effective tok/s).
  - **Files**: `crates/metal-attention/src/inference.rs` or `src/main.rs`
  - **Done when**: `--draft` flag accepted; speculative decode invoked when provided
  - **Verify**: `cargo build 2>&1 | grep -i error; echo "exit: $?"`
  - **Commit**: `feat(cli): add --draft and --draft-tokens flags for speculative decoding`
  - _Requirements: FR-12, AC-4.5_
  - _Design: Component 4_

- [ ] 4.6 KV cache rollback correctness test
  - **Do**: Add test `test_kv_cache_rollback` in `tests/gpu_correctness.rs`: (1) Process 5 tokens through SmolLM forward_token. (2) Capture logits at position 5. (3) Rollback to position 3. (4) Re-process tokens 4, 5. (5) Compare logits at position 5 -- must match. This validates rollback + re-append produces same results.
  - **Files**: `crates/metal-attention/tests/gpu_correctness.rs`
  - **Done when**: Rollback test passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test gpu_correctness -- test_kv_cache_rollback --ignored --test-threads=1 2>&1 | tail -5`
  - **Commit**: `test(kv-cache): verify rollback + re-append produces identical logits`
  - _Requirements: AC-4.3_
  - _Design: Component 4_

- [ ] 4.7 Speculative decode greedy correctness test
  - **Do**: Add test `test_speculative_greedy_matches_target` in `tests/speculative_correctness.rs`: (1) Load SmolLM as both draft and target (same model, to guarantee high acceptance). (2) Generate 20 tokens with speculative decode (greedy). (3) Generate 20 tokens with target-only greedy. (4) Compare token sequences -- must be identical. This validates the accept/reject logic without needing Mistral-7B.
  - **Files**: `crates/metal-attention/tests/speculative_correctness.rs`
  - **Done when**: Test passes with identical output
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test speculative_correctness -- test_speculative_greedy --ignored --test-threads=1 2>&1 | tail -10`
  - **Commit**: `test(speculative): verify greedy speculative matches target-only output`
  - _Requirements: AC-4.2_
  - _Design: Component 4_

## Phase 5: Testing and Quality Gates

Focus: Full test suite, benchmarks, clippy, fmt, PR.

- [ ] 5.1 Q6_K kernel edge case tests
  - **Do**: Add tests in `tests/gpu_correctness.rs`: (1) `test_q6_k_zero_scale` -- d=0 produces all zeros. (2) `test_q6_k_max_values` -- all quants at max (63). (3) `test_q6_k_single_row` -- out_dim=1. (4) `test_q6_k_finiteness` -- no NaN/Inf in output. Use synthetic Q6_K block data.
  - **Files**: `crates/metal-attention/tests/gpu_correctness.rs`
  - **Done when**: All edge case tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test gpu_correctness -- test_q6_k --test-threads=1 2>&1 | tail -10`
  - **Commit**: `test(q6k): add edge case tests for zero scale, max values, single row`
  - _Requirements: AC-1.1_

- [ ] 5.2 Run full existing test suite
  - **Do**: Run all 324 existing tests to verify no regressions from any changes. Fix any failures.
  - **Files**: None (test run)
  - **Done when**: All existing tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --workspace -- --test-threads=1 2>&1 | tail -20`
  - **Commit**: `fix(tests): address any regressions from performance optimizations` (if needed)
  - _Requirements: NFR-7_

- [ ] 5.3 Decode throughput benchmark
  - **Do**: Add/update benchmark `bench_decode_mistral_100tok` in `benches/inference.rs`. Measure Mistral-7B decode throughput over 100 tokens. Report tok/s baseline and with Q6_K. Compare against 42 tok/s baseline.
  - **Files**: `crates/metal-attention/benches/inference.rs`
  - **Done when**: Benchmark reports tok/s, confirms >= 47 with Q6_K
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo bench -- bench_decode_mistral 2>&1 | tail -10`
  - **Commit**: `bench(decode): add Mistral-7B decode throughput benchmark with Q6_K`
  - _Requirements: NFR-1_

- [ ] 5.4 Clippy and fmt
  - **Do**: Run `cargo clippy --workspace -- -D warnings` and `cargo fmt --all -- --check`. Fix all warnings and formatting issues.
  - **Files**: Various
  - **Done when**: Both commands pass with zero warnings/errors
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clippy --workspace -- -D warnings 2>&1 | tail -5 && cargo fmt --all -- --check 2>&1 | tail -5`
  - **Commit**: `fix(lint): address clippy warnings and format all files`

- [ ] 5.5 Create PR and verify CI
  - **Do**: Push branch, create PR with `gh pr create`. Title: "perf: close 42->100+ tok/s gap on Mistral-7B Q4_0". Body: summary of 4 optimizations, benchmark results, test results. Run `gh pr checks --watch` to verify CI passes.
  - **Files**: None (PR creation)
  - **Done when**: PR created, all CI checks green
  - **Verify**: `gh pr checks --watch`
  - **Commit**: N/A (PR creation)

## Notes

- **POC shortcuts taken**: Speculative decode is scaffolding only (greedy, fixed N=8, no adaptive draft length). Draft model tokenizer mismatch not resolved.
- **Production TODOs**: Temperature>0 speculative sampling, adaptive N, draft model compatibility checking, BufferStats tracking for DX output.
- **Dependencies**: Tasks 1.1-1.4 must complete before 3.3. Task 4.1 must complete before 4.4.
- **Model files required**: SmolLM-135M Q4_0 (87 MB) for unit tests, Mistral-7B Q4_0 (3.8 GB) for integration tests. Tests requiring Mistral-7B use `#[ignore]`.
