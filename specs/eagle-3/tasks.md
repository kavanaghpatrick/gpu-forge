---
spec: eagle-3
phase: tasks
total_tasks: 28
created: 2026-02-17
---

# Tasks: EAGLE-3 Speculative Decoding

## Phase 1: Make It Work (POC)

Focus: Validate end-to-end EAGLE-3 infrastructure with random weights. Prove hidden state capture, draft head forward, and chain verification all integrate correctly. Skip SafeTensors parsing, skip tests, accept hardcoded layer indices.

- [x] 1.1 Add EagleCaptureBuffers and hidden state tap to GpuForwardPass
  - **Do**:
    1. Add `EagleCaptureBuffers` struct to `gpu_forward_pass.rs`: three `Retained<ProtocolObject<dyn MTLBuffer>>` (feat_low, feat_mid, feat_high), three layer indices, and `enabled: bool`
    2. Add `eagle_capture: Option<EagleCaptureBuffers>` field to `GpuForwardPass` struct (initialized to `None`)
    3. Add `pub fn enable_eagle_capture(&mut self, low: usize, mid: usize, high: usize)` method that allocates 3 buffers of `hidden_size * sizeof(f32)` bytes each and sets layer indices
    4. In `forward_token()` layer loop (after line 495, the FFN down+residual dispatch), insert: `if let Some(ref capture) = self.eagle_capture { if layer_idx == capture.layer_low { self.encode_buffer_copy(&encoder, &self.hidden_a, &capture.feat_low, self.hidden_size); } ... }` for all 3 layers
    5. Add public accessors: `eagle_capture_low()`, `eagle_capture_mid()`, `eagle_capture_high()` returning `Option<&ProtocolObject<dyn MTLBuffer>>`
    6. Add accessor `pub fn hidden_size(&self) -> usize`, `pub fn vocab_size(&self) -> usize`, `pub fn num_kv_heads(&self) -> usize`, `pub fn head_dim(&self) -> usize`, `pub fn num_heads(&self) -> usize`, `pub fn intermediate_size(&self) -> usize`, `pub fn rope_theta(&self) -> f32`, `pub fn rms_norm_eps(&self) -> f32`, `pub fn num_layers(&self) -> usize`
    7. Verify zero overhead: when `eagle_capture` is `None`, no code is executed in the layer loop
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: `enable_eagle_capture(0, 16, 31)` compiles and buffer_copy dispatches are inserted into forward_token layer loop
  - **Verify**: `cargo build -p metal-attention 2>&1 | tail -5`
  - **Commit**: `feat(eagle): add hidden state capture buffers to GpuForwardPass`
  - _Requirements: FR-1, AC-1.1, AC-1.2, AC-1.3, AC-1.4_
  - _Design: Component 1 - Hidden State Capture_

- [x] 1.2 Create concat_buffers Metal shader
  - **Do**:
    1. Create `concat_buffers.metal` in shaders directory
    2. Implement two kernels: `concat_buffers_2` (2 inputs -> 1 output, dims a+b) and `concat_buffers_3` (3 inputs -> 1 output, dims a+b+c)
    3. Each kernel reads from input buffers and writes sequentially to output: `if gid < dim_a { out[gid] = a[gid]; } else if gid < dim_a + dim_b { out[gid] = b[gid - dim_a]; }` etc.
    4. Run `cargo clean -p metal-attention-kernels` to force shader recompilation
    5. Add `concat_buffers_2` and `concat_buffers_3` PSO prewarming to `GpuForwardPass::from_gguf()` or eagle head constructor
  - **Files**: `crates/metal-attention-kernels/shaders/concat_buffers.metal`
  - **Done when**: New .metal file compiles into metallib via build.rs auto-discovery
  - **Verify**: `cargo clean -p metal-attention-kernels && cargo build -p metal-attention-kernels 2>&1 | tail -5`
  - **Commit**: `feat(eagle): add concat_buffers Metal shader for feature fusion`
  - _Requirements: FR-7_
  - _Design: Component 5 - concat_buffers Metal kernel_

- [x] 1.3 Create eagle_head.rs with EagleHead struct and random-weight initialization
  - **Do**:
    1. Create `eagle_head.rs` in `crates/metal-attention/src/`
    2. Define `EagleHead` struct with fields: `fc_fuse_weight: WeightBuffer` [4096,12288], `fc_concat_weight: WeightBuffer` [4096,8192], `decoder_attn_norm`, `decoder_attn: AttnProjBuffers`, `decoder_ffn_norm`, `decoder_ffn: FfnBuffers`, `eagle_kv_cache: GpuKVCache`, plus scratch buffers (hidden_a/b, q/k/v/attn_out/gate/up/silu, fused_buf [12288], combined_buf [8192], logits_buf, argmax bufs)
    3. Add `PsoCache` field for kernel pipeline states
    4. Implement `EagleHead::new_random(device, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, vocab_size)` that allocates all buffers with random F32 data (using simple LCG PRNG seeded with 42)
    5. The FC layers use F32 weights for POC (Q4_0 deferred to Phase 2)
    6. Eagle KV cache: `GpuKVCache::new(device, 64, kv_dim)` -- small, reset per round
    7. Add `pub fn reset_kv_cache(&mut self)` that calls truncate on eagle_kv_cache to 0
    8. Prewarm all needed PSOs: `rmsnorm_optimized`, `matvec_f32_v2`, `rope_apply_dual`, `decode_attention_v2`, `ffn_silu`, `buffer_copy`, `concat_buffers_3`, `concat_buffers_2`, `argmax_reduce`, `argmax_final`, `kv_cache_copy`
  - **Files**: `crates/metal-attention/src/eagle_head.rs`
  - **Done when**: `EagleHead::new_random()` compiles and allocates all buffers
  - **Verify**: `cargo build -p metal-attention 2>&1 | tail -5`
  - **Commit**: `feat(eagle): create EagleHead struct with random-weight init`
  - _Requirements: FR-2, FR-3, FR-4, FR-8, AC-2.1, AC-2.5_
  - _Design: Component 2 - EAGLE Head_

- [x] 1.4 Implement forward_draft_token GPU pipeline in EagleHead
  - **Do**:
    1. Implement `pub fn forward_draft_token(&mut self, feat_low, feat_mid, feat_high, prev_token, target_embed_buf, target_lm_head, target_lm_head_is_f32, target_lm_head_q6k, target_lm_head_q8) -> Result<u32, String>`
    2. Pipeline steps (all in single command buffer + encoder):
       a. concat_buffers_3: feat_low + feat_mid + feat_high -> fused_buf (12288-dim)
       b. matvec_f32_v2: fc_fuse_weight * fused_buf -> hidden_a (4096-dim)
       c. CPU embed lookup: copy prev_token embedding from target_embed_buf to scratch (4096-dim). Write to combined_buf at offset 0.
       d. End first encoder, commit+wait. Write prev_token embedding bytes to combined_buf offset 0, copy hidden_a to combined_buf offset 4096*4.
       e. Alternative: use concat_buffers_2 on GPU: hidden_a + embed_scratch -> combined_buf (8192-dim)
       f. matvec_f32_v2: fc_concat_weight * combined_buf -> hidden_b (4096-dim)
       g. Decoder layer: rmsnorm -> Q/K/V matvec -> RoPE -> KV cache append -> attention -> O proj + residual -> FFN rmsnorm -> gate/up -> silu -> down + residual
       h. Final norm -> lm_head matvec (reuse target's lm_head buffer) -> logits
       i. argmax_reduce + argmax_final -> token_id
    3. Use the same encode_* helper pattern as GpuForwardPass (or call equivalent dispatches directly)
    4. Eagle KV cache position increments each call; tracks its own RoPE position
    5. Return the argmax token ID
  - **Files**: `crates/metal-attention/src/eagle_head.rs`
  - **Done when**: `forward_draft_token` compiles and encodes the full pipeline
  - **Verify**: `cargo build -p metal-attention 2>&1 | tail -5`
  - **Commit**: `feat(eagle): implement forward_draft_token GPU pipeline`
  - _Requirements: FR-2, FR-3, FR-4, AC-2.2, AC-2.3_
  - _Design: Component 3 - EAGLE Draft Forward Pass_

- [x] 1.5 [VERIFY] Quality checkpoint: cargo build + cargo clippy
  - **Do**: Run build and clippy to catch compile errors and warnings from tasks 1.1-1.4
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build --workspace 2>&1 | tail -10 && cargo clippy --workspace -- -D warnings 2>&1 | tail -10`
  - **Done when**: Zero compile errors, zero clippy warnings
  - **Commit**: `chore(eagle): pass quality checkpoint` (only if fixes needed)

- [x] 1.6 Create eagle.rs with EagleDecoder and chain speculation loop
  - **Do**:
    1. Create `eagle.rs` in `crates/metal-attention/src/`
    2. Define `EagleDecoder` struct: `target: GpuForwardPass` (with eagle_capture enabled), `eagle_head: EagleHead`, `n_draft: usize` (default 6)
    3. Implement `EagleDecoder::new_random(target_path: &Path, n_draft: usize)` that:
       a. Loads target via `GpuForwardPass::from_gguf(target_path)`
       b. Calls `target.enable_eagle_capture(0, 16, 31)` (Mistral-7B: layers 0, 16, 31)
       c. Creates `EagleHead::new_random(...)` using target's dimensions
    4. Implement `pub fn generate(&mut self, prompt: &[u32], max_tokens: usize, callback: impl FnMut(u32)) -> Result<(Vec<u32>, SpecStats), String>`:
       a. Prefill target with `forward_prompt(prompt)` -- captures hidden states at layers 0/16/31
       b. Use first token, enter speculation loop
       c. Call `speculation_round()` until max_tokens reached
    5. Implement `fn speculation_round(&mut self, last_token, stats, callback) -> Result<Vec<u32>, String>`:
       a. Phase 1: Draft N tokens via `eagle_head.forward_draft_token()` loop, reading captured features
       b. Phase 2: Build verify_input = [last_token, d1..dn], call `target.forward_prompt_logits()`
       c. Phase 3: Greedy accept/reject (same logic as existing SpeculativeDecoder)
       d. Phase 4: Rollback target KV on rejection, reset eagle KV cache
    6. Reuse SpecStats from speculative.rs (or import it)
  - **Files**: `crates/metal-attention/src/eagle.rs`
  - **Done when**: `EagleDecoder` compiles with full generate + speculation_round
  - **Verify**: `cargo build -p metal-attention 2>&1 | tail -5`
  - **Commit**: `feat(eagle): create EagleDecoder with chain speculation loop`
  - _Requirements: FR-5, AC-3.1, AC-3.3_
  - _Design: Component 6 - EagleDecoder Integration_

- [x] 1.7 Register eagle modules in lib.rs
  - **Do**:
    1. Add `pub mod eagle;` and `pub mod eagle_head;` to `crates/metal-attention/src/lib.rs`
    2. Add re-exports: `pub use eagle::EagleDecoder;` and `pub use eagle_head::EagleHead;`
  - **Files**: `crates/metal-attention/src/lib.rs`
  - **Done when**: Both modules publicly accessible from the crate
  - **Verify**: `cargo build -p metal-attention 2>&1 | tail -5`
  - **Commit**: `feat(eagle): register eagle modules in lib.rs`
  - _Design: File Structure_

- [x] 1.8 [VERIFY] Quality checkpoint: full workspace build + clippy
  - **Do**: Compile entire workspace including new eagle modules, run clippy
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build --workspace 2>&1 | tail -10 && cargo clippy --workspace -- -D warnings 2>&1 | tail -10`
  - **Done when**: Zero compile errors, zero clippy warnings across workspace
  - **Commit**: `chore(eagle): pass quality checkpoint` (only if fixes needed)

- [x] 1.9 POC end-to-end test: run EagleDecoder with random weights on Mistral-7B
  - **Do**:
    1. Create a minimal integration test in `crates/metal-attention/tests/eagle_test.rs`
    2. Test `test_eagle_random_weights_runs`: constructs `EagleDecoder::new_random()` with Mistral-7B path, calls `generate(prompt, 20, callback)`, asserts no crash and returns 20 tokens
    3. Random weights will give ~0% acceptance (all draft tokens rejected), but infrastructure must not panic
    4. Print acceptance rate from SpecStats -- expect near 0% but non-crash
    5. Mark test with `#[ignore]` (requires Mistral-7B model file)
  - **Files**: `crates/metal-attention/tests/eagle_test.rs`
  - **Done when**: Test passes (20 tokens generated without crash)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test eagle_test -- test_eagle_random_weights_runs --ignored --test-threads=1 2>&1 | tail -20`
  - **Commit**: `feat(eagle): POC complete -- EagleDecoder runs end-to-end with random weights`
  - _Requirements: FR-8, AC-2.4_
  - _Design: POC validation_

## Phase 2: Weight Loading (SafeTensors + Real EAGLE Weights)

Focus: Load real EAGLE-3 draft head weights from SafeTensors format. Handle tensor name mapping and dimension validation. Enable Q4_0 quantized weights for the FC and decoder layers.

- [x] 2.1 Create eagle_weights.rs with SafeTensors parser
  - **Do**:
    1. Create `eagle_weights.rs` in `crates/metal-attention/src/`
    2. Implement minimal SafeTensors parser:
       a. Read first 8 bytes as u64 little-endian `header_size`
       b. Read `header_size` bytes as UTF-8 JSON string
       c. Parse JSON with `serde_json` to extract tensor metadata: name -> { dtype, shape, data_offsets: [start, end] }
       d. Tensor data starts at offset `8 + header_size` in the file
       e. Use `memmap2` to mmap the file
    3. Define `SafeTensorsFile` struct with `header: HashMap<String, TensorInfo>`, `data_offset: usize`, `mmap: memmap2::Mmap`
    4. Implement `SafeTensorsFile::open(path) -> Result<Self, String>` with validation
    5. Implement `fn get_tensor_data(&self, name: &str) -> Result<&[u8], String>` returning the raw byte slice for a tensor
    6. Implement `fn get_tensor_shape(&self, name: &str) -> Result<Vec<usize>, String>`
  - **Files**: `crates/metal-attention/src/eagle_weights.rs`
  - **Done when**: SafeTensors parser can open a file and extract tensor data by name
  - **Verify**: `cargo build -p metal-attention 2>&1 | tail -5`
  - **Commit**: `feat(eagle): add SafeTensors parser for EAGLE weight loading`
  - _Requirements: FR-6, AC-4.1_
  - _Design: Component 7 - Weight Loading_

- [x] 2.2 Add EagleWeightStore with tensor name mapping
  - **Do**:
    1. In `eagle_weights.rs`, define `EagleWeightStore` struct mirroring EagleHead fields: `fc_fuse_weight: WeightBuffer`, `fc_concat_weight: WeightBuffer`, `decoder_attn_norm`, `decoder_attn: AttnProjBuffers`, `decoder_ffn_norm`, `decoder_ffn: FfnBuffers`
    2. Implement tensor name mapping table (from design):
       - `fc.weight` -> fc_fuse [4096, 12288]
       - `fc2.weight` -> fc_concat [4096, 8192]
       - `layers.0.self_attn.{q,k,v,o}_proj.weight` -> decoder attn
       - `layers.0.mlp.{gate,up,down}_proj.weight` -> decoder FFN
       - `layers.0.input_layernorm.weight` -> attn norm
       - `layers.0.post_attention_layernorm.weight` -> ffn norm
    3. Implement `EagleWeightStore::from_safetensors(path, device, target_hidden_size)`:
       a. Open SafeTensors file
       b. For each expected tensor, look up by name, validate shape
       c. Create Metal buffers from tensor data (F32 weights -> alloc_buffer_with_data)
       d. For Q4_0/quantized weights if dtype indicates quantization, handle appropriately
    4. Add dimension validation: assert fc_fuse shape[1] == 3 * target_hidden_size, fc_concat shape[1] == 2 * target_hidden_size
    5. Return clear error messages for missing/mismatched tensors
  - **Files**: `crates/metal-attention/src/eagle_weights.rs`
  - **Done when**: EagleWeightStore loads all EAGLE head tensors with dimension validation
  - **Verify**: `cargo build -p metal-attention 2>&1 | tail -5`
  - **Commit**: `feat(eagle): add EagleWeightStore with SafeTensors tensor mapping`
  - _Requirements: FR-6, AC-4.2, AC-4.3, AC-4.4_
  - _Design: Component 7 - Weight Loading, Tensor name mapping table_

- [x] 2.3 Add EagleHead::from_weights() constructor loading real weights
  - **Do**:
    1. In `eagle_head.rs`, add `pub fn from_weights(weight_store: EagleWeightStore, device, target: &GpuForwardPass) -> Result<Self, String>`
    2. Move weight buffers from EagleWeightStore into EagleHead fields
    3. Allocate scratch buffers using target model dimensions
    4. Share target's lm_head and embed buffers (no duplication)
    5. Initialize eagle KV cache
    6. Update `EagleDecoder` to support `new(target_path, eagle_weights_path, n_draft)` that uses `from_weights` path
  - **Files**: `crates/metal-attention/src/eagle_head.rs`, `crates/metal-attention/src/eagle.rs`
  - **Done when**: `EagleDecoder::new()` loads real EAGLE weights from SafeTensors file
  - **Verify**: `cargo build -p metal-attention 2>&1 | tail -5`
  - **Commit**: `feat(eagle): load real EAGLE weights from SafeTensors`
  - _Requirements: FR-6, AC-2.1, AC-2.4, AC-2.5_
  - _Design: Component 2 weight loading path_

- [x] 2.4 Register eagle_weights module in lib.rs
  - **Do**:
    1. Add `pub mod eagle_weights;` to `crates/metal-attention/src/lib.rs`
  - **Files**: `crates/metal-attention/src/lib.rs`
  - **Done when**: Module publicly accessible
  - **Verify**: `cargo build -p metal-attention 2>&1 | tail -5`
  - **Commit**: `feat(eagle): register eagle_weights module`
  - _Design: File Structure_

- [x] 2.5 [VERIFY] Quality checkpoint: build + clippy + existing tests
  - **Do**: Full workspace build, clippy, and run existing unit tests to check for regressions
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build --workspace 2>&1 | tail -5 && cargo clippy --workspace -- -D warnings 2>&1 | tail -5 && cargo test --workspace --lib 2>&1 | tail -10`
  - **Done when**: Zero compile errors, zero clippy warnings, all existing unit tests pass
  - **Commit**: `chore(eagle): pass quality checkpoint` (only if fixes needed)

## Phase 3: Testing

Focus: Correctness tests for EAGLE head forward pass, hidden state capture, and end-to-end chain verification. Benchmark EAGLE vs baseline decode.

- [x] 3.1 Unit test: hidden state capture produces valid GPU buffers
  - **Do**:
    1. In `eagle_test.rs`, add `test_hidden_state_capture`:
       a. Load Mistral-7B, enable_eagle_capture(0, 16, 31)
       b. Call forward_token with a token
       c. Read back feat_low, feat_mid, feat_high via `read_buffer_slice`
       d. Assert: no NaN/Inf values, non-zero (different from initial buffer state)
       e. Assert: all 3 buffers have `hidden_size` f32 elements
    2. Add `test_hidden_state_capture_zero_overhead`:
       a. Load Mistral-7B WITHOUT eagle capture
       b. Time 100 forward_token calls
       c. Enable eagle capture, time 100 forward_token calls
       d. Assert: overhead < 0.5ms per token (3 buffer_copy dispatches of 16KB each)
    3. Mark both tests `#[ignore]` (requires Mistral-7B)
  - **Files**: `crates/metal-attention/tests/eagle_test.rs`
  - **Done when**: Both tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test eagle_test -- test_hidden_state_capture --ignored --test-threads=1 2>&1 | tail -20`
  - **Commit**: `test(eagle): add hidden state capture correctness tests`
  - _Requirements: AC-1.2, AC-1.3, AC-1.4_
  - _Design: Test Strategy - Unit Tests_

- [x] 3.2 Unit test: EagleHead forward_draft_token produces valid token IDs
  - **Do**:
    1. In `eagle_test.rs`, add `test_eagle_head_produces_valid_tokens`:
       a. Create `EagleHead::new_random()` with Mistral-7B dimensions
       b. Allocate 3 fake feature buffers with random F32 data (4096 elements each)
       c. Call `forward_draft_token()` 6 times
       d. Assert: each returned token_id < vocab_size (32000 for Mistral)
       e. Assert: no panics or GPU command buffer errors
    2. Add `test_eagle_head_kv_cache_reset`:
       a. Create EagleHead, run 6 draft tokens
       b. Call `reset_kv_cache()`
       c. Run 6 more draft tokens
       d. Assert: no crash (KV cache properly reset)
  - **Files**: `crates/metal-attention/tests/eagle_test.rs`
  - **Done when**: Both tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test eagle_test -- test_eagle_head --ignored --test-threads=1 2>&1 | tail -20`
  - **Commit**: `test(eagle): add EagleHead forward pass correctness tests`
  - _Requirements: AC-2.2, AC-2.4_
  - _Design: Test Strategy - Unit Tests_

- [x] 3.3 [VERIFY] Quality checkpoint: all tests pass
  - **Do**: Run all workspace unit tests and eagle integration tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --workspace --lib 2>&1 | tail -10 && cargo test --test eagle_test -- --ignored --test-threads=1 2>&1 | tail -10`
  - **Done when**: All tests pass
  - **Commit**: `chore(eagle): pass quality checkpoint` (only if fixes needed)

- [x] 3.4 Integration test: EagleDecoder greedy output correctness
  - **Do**:
    1. In `eagle_test.rs`, add `test_eagle_greedy_matches_target_only`:
       a. Generate 50 tokens with target-only greedy decode (standard GpuForwardPass)
       b. Generate 50 tokens with EagleDecoder (random weights, chain=6)
       c. With random weights, acceptance rate ~0%, so every round: target's token accepted at position 0
       d. Assert: output tokens are identical to target-only decode
       e. This validates that the verification + rollback logic preserves correctness
    2. Print SpecStats: rounds, acceptance rate, draft_accepted/tokens_drafted
    3. Mark `#[ignore]`
  - **Files**: `crates/metal-attention/tests/eagle_test.rs`
  - **Done when**: 50-token output matches target-only greedy decode exactly
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test eagle_test -- test_eagle_greedy_matches --ignored --test-threads=1 2>&1 | tail -20`
  - **Commit**: `test(eagle): verify EagleDecoder greedy output matches target-only decode`
  - _Requirements: AC-3.1, AC-3.5, NFR-5_
  - _Design: Test Strategy - Integration Tests_

- [x] 3.5 Benchmark: EAGLE decode vs baseline
  - **Do**:
    1. In `benches/inference.rs`, add benchmark group `eagle_decode`:
       a. `bench_eagle_random_decode_100tok`: EagleDecoder with random weights, 100 tokens. Reports effective tok/s.
       b. `bench_baseline_decode_100tok`: Target-only decode, 100 tokens (if not already present). Reports tok/s.
    2. Use criterion: sample_size(10), measurement_time(30s)
    3. Both benchmarks use same prompt `&[1, 2, 3, 4, 5]`
    4. Random weights will show overhead (no speedup), but validates benchmark infrastructure
  - **Files**: `benches/inference.rs`
  - **Done when**: Both benchmarks run and report tok/s numbers
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo bench -- eagle_decode 2>&1 | tail -20`
  - **Commit**: `bench(eagle): add EAGLE decode throughput benchmark`
  - _Requirements: NFR-4_
  - _Design: Test Strategy - Benchmarks_

## Phase 4: Quality Gates

- [ ] 4.1 [VERIFY] Full local CI: build + clippy + fmt + tests
  - **Do**: Run complete local quality suite
  - **Verify**: All commands must pass:
    ```
    cd /Users/patrickkavanagh/gpu_kernel/metal-attention
    cargo fmt --check 2>&1 | tail -5
    cargo clippy --workspace -- -D warnings 2>&1 | tail -10
    cargo build --workspace 2>&1 | tail -5
    cargo test --workspace --lib 2>&1 | tail -10
    cargo test --test eagle_test -- --ignored --test-threads=1 2>&1 | tail -20
    ```
  - **Done when**: All commands pass with zero errors
  - **Commit**: `fix(eagle): address lint/type/format issues` (if fixes needed)

- [ ] 4.2 Create PR and verify CI
  - **Do**:
    1. Verify current branch is a feature branch: `git branch --show-current`
    2. If on default branch, STOP and alert user
    3. Stage all new and modified files: eagle.rs, eagle_head.rs, eagle_weights.rs, gpu_forward_pass.rs, lib.rs, concat_buffers.metal, eagle_test.rs, inference.rs (bench)
    4. Push branch: `git push -u origin <branch-name>`
    5. Create PR: `gh pr create --title "feat(eagle): EAGLE-3 speculative decoding POC" --body "..."`
  - **Verify**: `gh pr checks --watch` (wait for CI completion, all checks green)
  - **Done when**: PR created, all CI checks green
  - **If CI fails**: Read failure, fix locally, push, re-verify

## Phase 5: PR Lifecycle

- [ ] 5.1 Monitor CI and fix failures
  - **Do**:
    1. Check PR CI status: `gh pr checks`
    2. If any check fails, read logs: `gh pr checks 2>&1`
    3. Fix issues locally (lint, test failures, build errors)
    4. Commit fix and push: `git push`
    5. Re-verify: `gh pr checks --watch`
  - **Verify**: `gh pr checks` shows all green
  - **Done when**: All CI checks passing

- [ ] 5.2 Address review comments
  - **Do**:
    1. Check for review comments: `gh api repos/{owner}/{repo}/pulls/{pr_number}/comments`
    2. Address each comment with code changes
    3. Push fixes
    4. Respond to comments via `gh api`
  - **Verify**: No unresolved review comments, CI still green
  - **Done when**: All review comments addressed

- [ ] 5.3 [VERIFY] Full local CI (post-review fixes)
  - **Do**: Re-run full quality suite after any review-driven changes
  - **Verify**:
    ```
    cd /Users/patrickkavanagh/gpu_kernel/metal-attention
    cargo fmt --check
    cargo clippy --workspace -- -D warnings
    cargo test --workspace --lib
    cargo test --test eagle_test -- --ignored --test-threads=1
    ```
  - **Done when**: All commands pass
  - **Commit**: None (should already be clean)

- [ ] 5.4 [VERIFY] AC checklist
  - **Do**: Read requirements.md, verify each acceptance criterion is satisfied:
    - AC-1.1: `enable_eagle_capture()` exists -> grep for fn signature
    - AC-1.2: captured hidden states available as GPU buffers -> grep for `eagle_capture_low`
    - AC-1.3: extraction overhead < 0.5ms -> verified by test
    - AC-1.4: zero overhead when disabled -> `Option<>` guard in layer loop
    - AC-2.1: EagleHead loads FC weights -> grep for `fc_fuse_weight`, `fc_concat_weight`
    - AC-2.2: forward_draft_token returns token ID -> grep for fn signature
    - AC-2.4: random weights produce valid tokens -> verified by test
    - AC-3.1: greedy output matches target-only -> verified by test
    - AC-4.1: SafeTensors parser -> grep for `SafeTensorsFile`
    - AC-4.2: tensor name mapping -> grep for `fc.weight`, `layers.0`
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && grep -r "enable_eagle_capture" crates/metal-attention/src/ && grep -r "forward_draft_token" crates/metal-attention/src/ && grep -r "SafeTensorsFile" crates/metal-attention/src/`
  - **Done when**: All acceptance criteria confirmed met
  - **Commit**: None

## Notes

- **POC shortcuts taken**:
  - Random F32 weights for all EAGLE head layers (not Q4_0 quantized)
  - Hardcoded layer indices (0, 16, 31) for Mistral-7B only
  - F32 FC layers instead of Q4_0 (higher bandwidth but simpler)
  - embed lookup for prev_token uses CPU memcpy (same as target model pattern)
  - No tree drafting -- chain (linear) only
  - No temperature sampling -- greedy only
  - No CLI integration (--eagle flag) -- library-level only
  - No SafeTensors file available yet for testing real weights

- **Production TODOs (future specs)**:
  - Q4_0 quantization for FC layers (reduce ~144MB -> ~40MB bandwidth/token)
  - Tree-structured drafting (EAGLE-2 style, requires tree attention mask kernel)
  - CLI `--eagle` flag integration with main.rs
  - Temperature > 0 speculative sampling
  - Auto-detection of EAGLE weights in GGUF
  - Training pipeline for EAGLE-3 head (PyTorch/MLX)
  - Adaptive depth based on acceptance rate
  - Configurable layer indices (not hardcoded to 0/16/31)

- **Key dependency**: The `forward_prompt_logits()` batched verify path already exists and is tested. EAGLE chain verification reuses it directly. This is the highest-leverage reuse.

- **New .metal shader note**: After creating `concat_buffers.metal`, must run `cargo clean -p metal-attention-kernels` to force build.rs to discover and compile the new shader file.

- **EagleHead scratch buffer strategy**: Allocate dedicated scratch buffers for the eagle head (not shared with GpuForwardPass) to avoid aliasing. The eagle head runs on the same GPU device but with its own command buffers.
