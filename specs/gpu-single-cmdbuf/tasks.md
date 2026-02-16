---
spec: gpu-single-cmdbuf
phase: tasks
total_tasks: 13
created: 2026-02-15
---

# Tasks: Single Command Buffer per Token

## Phase 1: Make It Work (POC)

Focus: Get 91 -> 1 command buffer working end-to-end. Skip tests, accept minimal error handling.

- [x] 1.1 Create kv_cache_copy and buffer_copy Metal shaders
  - **Do**:
    1. Create `kv_cache_copy.metal` — kernel copies kv_dim floats from scratch K/V into KV cache at row offset. 6 buffer params: k_src(0), v_src(1), k_dst(2), v_dst(3), kv_dim(4), row_idx(5). Guard: `if (tid >= kv_dim) return`.
    2. Create `buffer_copy.metal` — kernel copies count floats src->dst. 3 buffer params: src(0), dst(1), count(2). Guard: `if (tid >= count) return`.
    3. Follow `residual.metal` pattern exactly (include header, using namespace, thread_position_in_grid).
  - **Files**:
    - `metal-attention-kernels/shaders/kv_cache_copy.metal` (create)
    - `metal-attention-kernels/shaders/buffer_copy.metal` (create)
  - **Done when**: Both .metal files exist with correct kernel signatures matching design doc
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clean -p metal-attention-kernels && cargo build --workspace 2>&1 | tail -5` (exit 0, no shader compilation errors)
  - **Commit**: `feat(kernels): add kv_cache_copy and buffer_copy Metal shaders`
  - _Requirements: FR-1, FR-2, AC-1.1, AC-2.1_
  - _Design: Components 1, 2_

- [x] 1.2 Add encode_kv_append to GpuKVCache
  - **Do**:
    1. Add imports to `gpu_kv_cache.rs`: `MTLComputeCommandEncoder`, `MTLComputePipelineState`, `MTLSize` from objc2_metal; `set_buffer`, `set_bytes` from metal_attention_kernels::dispatch.
    2. Add `pub fn encode_kv_append(&mut self, encoder, pso, k_src, v_src)` method to `GpuKVCache`. Follows design doc exactly: setComputePipelineState, set 6 buffers/bytes, dispatchThreads(kv_dim, 1, 1) with threadgroup (256, 1, 1), then `self.len += 1`.
    3. Keep existing `append_kv()` CPU method (used by debug path).
  - **Files**:
    - `crates/metal-attention/src/gpu_kv_cache.rs` (modify)
  - **Done when**: `encode_kv_append` compiles, takes encoder + PSO + scratch buffers, dispatches kernel, increments len
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build -p metal-attention 2>&1 | tail -5` (exit 0)
  - **Commit**: `feat(kv-cache): add encode_kv_append for GPU-side KV cache copy`
  - _Requirements: FR-4, AC-1.2, AC-1.3, AC-1.4_
  - _Design: Component 3_

- [x] 1.3 Add encode_buffer_copy and PSO prewarm to GpuForwardPass
  - **Do**:
    1. Add `fn encode_buffer_copy(&self, encoder, src, dst, count)` private method to `GpuForwardPass`. Looks up `buffer_copy` PSO, sets 3 params (src/dst/count_u32), dispatches with grid=(count, 1, 1) threadgroup=(256, 1, 1).
    2. Add `PsoKey::simple("kv_cache_copy")` and `PsoKey::simple("buffer_copy")` to the PSO prewarm vec in `from_gguf()` (lines 196-203).
  - **Files**:
    - `crates/metal-attention/src/gpu_forward_pass.rs` (modify)
  - **Done when**: `encode_buffer_copy` method compiles; both new PSOs in prewarm list
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build -p metal-attention 2>&1 | tail -5` (exit 0)
  - **Commit**: `feat(forward): add encode_buffer_copy method and prewarm new PSOs`
  - _Requirements: FR-5, FR-7, AC-2.2, AC-3.6_
  - _Design: Components 5, existing patterns #5_

- [x] 1.4 [VERIFY] Quality checkpoint: build compiles
  - **Do**: Verify workspace builds cleanly after new shaders + Rust methods
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build --workspace 2>&1 | tail -10` (exit 0)
  - **Done when**: Zero compilation errors
  - **Commit**: `chore(build): pass quality checkpoint` (only if fixes needed)

- [x] 1.5 Extract forward_token_debug and refactor forward_token to single command buffer
  - **Do**:
    1. Create `fn forward_token_debug(&mut self, token_id: u32) -> Result<Vec<f32>, String>` private method. Move current forward_token body (lines 261-317: the per-layer loop with encode_attention_projections/append_kv/encode_attention_output/encode_ffn_block + debug eprintln blocks) into this method. Keep the embed_lookup call, debug log after embed, all 4 encode_ calls with commit+wait, CPU append_kv, CPU copy_buffer, debug readbacks, final encode_final_logits, logit readback, position increment.
    2. Rewrite `forward_token()`:
       - Keep: token_id validation, embed_lookup, debug check
       - Add: `if debug { return self.forward_token_debug(token_id); }` early return after embed_lookup + debug log
       - Create single `cmd_buf` + `encoder`
       - Loop 30 layers: inline the logic from encode_attention_projections (rmsnorm, Q/K/V matvec, 2x rope), then call `kv_caches.cache_mut(layer_idx).encode_kv_append(encoder, pso, scratch_k, scratch_v)`, then encode_decode_attention with kv_len from cache, encode_matvec_q4_0 for O proj, encode_residual_add, encode_buffer_copy (scratch_residual -> hidden_a), then FFN block (rmsnorm, gate/up matvec, ffn_silu, down matvec, residual_add, encode_buffer_copy), done
       - After loop: encode_rmsnorm (final norm), lm_head matvec (check lm_head_is_f32)
       - Single endEncoding + commit + waitUntilCompleted + validate_command_buffer
       - Read logits, increment position, return
    3. The 4 private methods (encode_attention_projections, encode_attention_output, encode_ffn_block, encode_final_logits) remain for now (used by forward_token_debug). Do NOT remove yet.
    4. `copy_buffer()` free function remains (used by debug path).
  - **Files**:
    - `crates/metal-attention/src/gpu_forward_pass.rs` (modify)
  - **Done when**: forward_token uses 1 command buffer + 1 encoder for all 30 layers; debug path preserved
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build -p metal-attention 2>&1 | tail -5` (exit 0)
  - **Commit**: `feat(forward): single command buffer forward pass with GPU-side copies`
  - _Requirements: FR-3, FR-4, FR-5, FR-6, FR-8, AC-3.1 through AC-3.5, AC-5.1, AC-5.2, AC-5.3_
  - _Design: Components 4, 6_

- [x] 1.6 [VERIFY] Correctness: all workspace tests pass
  - **Do**:
    1. Run full workspace test suite with shader validation
    2. Run GPU correctness integration tests (requires model file)
    3. If any tests fail, debug and fix
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test --workspace -- --test-threads=1 2>&1 | tail -20` (exit 0, all tests pass)
  - **Done when**: All existing tests pass with MTL_SHADER_VALIDATION=1
  - **Commit**: `fix(forward): address correctness issues` (only if fixes needed)
  - _Requirements: AC-4.1, AC-4.2, AC-4.4_

- [x] 1.7 POC Checkpoint: verify tok/s improvement with benchmark
  - **Do**:
    1. Run the ignored GPU correctness tests to verify output matches CPU baseline
    2. Run release benchmark to measure tok/s
    3. Document results in .progress.md
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --test gpu_correctness -- --ignored --test-threads=1 2>&1 | tail -20 && cargo run --release -- bench --gpu -m models/SmolLM-135M.Q4_0.gguf --gen-length 100 --iterations 3 --seq-lengths 3 2>&1 | tail -20`
  - **Done when**: GPU correctness tests pass; benchmark shows measurable tok/s improvement over 75 tok/s baseline
  - **Commit**: `feat(single-cmdbuf): complete POC — single command buffer forward pass`
  - _Requirements: NFR-1, AC-4.1, AC-4.3_

## Phase 2: Quality — Tests, Clippy, Cleanup

- [x] 2.1 Add unit tests for kv_cache_copy and buffer_copy kernels
  - **Do**:
    1. Add tests to `crates/metal-attention-kernels/tests/correctness.rs`:
       - `test_kv_cache_copy_gpu_vs_cpu`: Allocate scratch_k/scratch_v with known data (use gen_data), allocate k_cache/v_cache buffers (zero-init). Create PSO for kv_cache_copy. Dispatch kernel at row_idx=0 with kv_dim=192. Read back cache. Compare row 0 to CPU memcpy reference (bit-for-bit, atol=0). Then dispatch again at row_idx=1. Read back. Verify row 0 unchanged, row 1 matches new data.
       - `test_buffer_copy_gpu_vs_cpu`: Allocate src with known data, dst with zeros. Create PSO for buffer_copy. Dispatch with count=576. Read back dst. Compare to src (bit-for-bit, atol=0).
       - `test_buffer_copy_small`: Same but count=1 (edge case).
    2. Import needed dispatch helpers. Follow existing test patterns: GpuDevice::new(), PsoCache::new(), alloc_buffer, set_buffer/set_bytes, read_buffer_slice.
    3. Since these are new kernels without existing dispatch_ convenience functions, dispatch directly using the low-level pattern: create cmd_buf, create encoder, setComputePipelineState, set_buffer/set_bytes, dispatchThreads, endEncoding, commit, waitUntilCompleted, read_buffer_slice.
  - **Files**:
    - `crates/metal-attention-kernels/tests/correctness.rs` (modify)
  - **Done when**: 3 new tests pass, verifying bit-for-bit correctness
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test --test correctness -- --test-threads=1 test_kv_cache_copy test_buffer_copy 2>&1 | tail -20` (all 3 new tests pass)
  - **Commit**: `test(kernels): add GPU vs CPU correctness tests for kv_cache_copy and buffer_copy`
  - _Requirements: AC-1.5, AC-2.3_
  - _Design: Test Strategy #1, #2, #3_

- [x] 2.2 [VERIFY] Quality checkpoint: clippy + full test suite
  - **Do**:
    1. Run clippy on workspace
    2. Run full test suite with shader validation
    3. Fix any warnings or failures
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clippy --workspace 2>&1 | tail -20 && MTL_SHADER_VALIDATION=1 cargo test --workspace -- --test-threads=1 2>&1 | tail -20` (exit 0, zero warnings, all tests pass)
  - **Done when**: Zero clippy warnings, all tests green
  - **Commit**: `chore(quality): fix clippy warnings` (only if fixes needed)
  - _Requirements: NFR-4_

- [x] 2.3 Clean up dead code (optional, only if clippy complains)
  - **Do**:
    1. If clippy warns about unused encode_attention_projections, encode_attention_output, encode_ffn_block, encode_final_logits, or copy_buffer — add `#[allow(dead_code)]` with a comment that they're used by forward_token_debug. Or if the debug path doesn't use them, remove them.
    2. Update module doc comment at top of gpu_forward_pass.rs to reflect single command buffer architecture (remove POC comment about "one command buffer per encoding block").
    3. Update doc comment at top of gpu_kv_cache.rs to mention GPU-side append.
  - **Files**:
    - `crates/metal-attention/src/gpu_forward_pass.rs` (modify)
    - `crates/metal-attention/src/gpu_kv_cache.rs` (modify)
  - **Done when**: No dead code warnings, doc comments accurate
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clippy --workspace 2>&1 | grep -c "warning" | xargs test 0 -eq` (zero warnings)
  - **Commit**: `refactor(forward): clean up dead code and update doc comments`
  - _Design: existing patterns_

## Phase 3: PR Lifecycle

- [x] 3.1 [VERIFY] Full local CI: build + clippy + test + benchmark
  - **Do**:
    1. Run complete local CI suite
    2. Run final benchmark, capture tok/s number
    3. Verify correctness with GPU correctness tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build --workspace && cargo clippy --workspace && MTL_SHADER_VALIDATION=1 cargo test --workspace -- --test-threads=1 && cargo test --test gpu_correctness -- --ignored --test-threads=1 2>&1 | tail -30` (all pass)
  - **Done when**: Build, clippy, all tests pass; benchmark shows >= 300 tok/s (or documented if lower)
  - **Commit**: `chore(quality): pass full local CI` (only if fixes needed)
  - _Requirements: NFR-1, NFR-4, AC-4.1, AC-4.2_

- [x] 3.2 Create PR
  - **Do**:
    1. Verify on feature branch: `git branch --show-current` (should be `feat/gpu-query` or similar)
    2. Push: `git push -u origin HEAD`
    3. Create PR with summary: 91 -> 1 command buffers per token, 2 new Metal kernels (kv_cache_copy, buffer_copy), single encoder forward pass, debug fallback preserved. Include tok/s before (75) and after measurement.
  - **Verify**: `gh pr checks` (all green, or no CI configured — note in PR)
  - **Done when**: PR created and pushed
  - **Commit**: None (PR creation, not a commit)

- [ ] 3.3 [VERIFY] AC checklist
  - **Do**: Programmatically verify each acceptance criterion:
    1. AC-1.1: `grep -c "kv_cache_copy" metal-attention-kernels/shaders/kv_cache_copy.metal` (exists)
    2. AC-1.2: `grep -c "encode_kv_append" crates/metal-attention/src/gpu_forward_pass.rs` (called inline in single encoder loop)
    3. AC-1.5 + AC-2.3: `cargo test --test correctness -- --test-threads=1 test_kv_cache_copy test_buffer_copy` (pass)
    4. AC-3.1: `grep -c "commandBuffer" crates/metal-attention/src/gpu_forward_pass.rs` in forward_token (verify only 1 creation in non-debug path)
    5. AC-4.2: `MTL_SHADER_VALIDATION=1 cargo test --workspace -- --test-threads=1` (pass)
    6. AC-5.1: `grep "forward_token_debug" crates/metal-attention/src/gpu_forward_pass.rs` (exists)
  - **Verify**: All grep/test commands exit 0
  - **Done when**: All acceptance criteria confirmed met
  - **Commit**: None

## Notes

- **POC shortcuts taken**: None significant — this is a focused refactor, not a new feature. The 4 old private methods are retained (not deleted) for the debug fallback path.
- **Production TODOs**:
  - Async completion handlers (replace waitUntilCompleted with addCompletedHandler) — future spec
  - Multi-queue parallel encoding (llama.cpp thread-parallel pattern) — future spec
  - Triple-buffered command buffers — future spec
- **Key risk**: Borrow checker conflict in forward_token when accessing PsoCache and kv_caches simultaneously. Design mitigates by pre-looking up PSO ref before calling encode_kv_append.
- **cargo clean required**: After creating .metal files in task 1.1, must run `cargo clean -p metal-attention-kernels` before building. The build.rs auto-discovers .metal files but cached artifacts may not include new shaders.
