# Tasks: GPU Inference Pipeline

Total: 32 tasks across 5 phases.

**Critical change**: Full GPU attention in Phase 1. No CPU hybrid. Zero GPU-CPU sync points in decode hot path. All 5 kernels (matvec_q4_0, rmsnorm_optimized, residual_add, decode_attention, rope_apply) + GpuKVCache required for POC.

## Phase 1: Make It Work (POC)

Focus: All 5 kernels + GpuWeightStore + GpuKVCache + GpuForwardPass pipeline producing correct output with zero CPU sync points. Measure first tok/s.

- [x] 1.1 Create fused Q4_0 matvec Metal kernel
  - **Do**:
    1. Create `matvec_q4_0.metal` with `BlockQ4_0` struct (`half d; uchar qs[16]`)
    2. Implement `matvec_q4_0` kernel: per-threadgroup = 1 output row, 32 threads (1 simdgroup)
    3. Inner loop: stride over Q4_0 blocks `for (b = tid; b < n_blocks; b += 32)`
    4. Dequant: low nibble `(byte & 0x0F) - 8`, high nibble `(byte >> 4) - 8`, multiply by scale
    5. Low nibbles at input indices `[base+i]`, high at `[base+i+16]` (matches dequantize.metal)
    6. `simd_sum()` for 32-wide reduction, thread 0 writes `output[row]`
    7. Buffer bindings: weight(0), input(1), output(2), out_dim(3), in_dim(4)
    8. Include `<metal_stdlib>` only (no types.h dependency -- uses raw constant buffers)
  - **Files**: `crates/metal-attention-kernels/shaders/matvec_q4_0.metal`
  - **Done when**: Metal compiler accepts shader
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clean -p metal-attention-kernels && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(kernels): add fused Q4_0 dequant+matvec kernel with simd_sum`
  - _Requirements: FR-1_
  - _Design: Fused Q4_0 Matvec Kernel_

- [x] 1.2 Create Rust dispatch wrapper for matvec_q4_0
  - **Do**:
    1. Create `matvec_q4_0.rs` in `kernels/src/` with `dispatch_matvec_q4_0()` standalone function
    2. Accept: device, pso_cache, weight_bytes (&[u8] raw Q4_0), input (&[f32]), out_dim, in_dim
    3. Alloc buffers: `alloc_buffer_with_data` for weight bytes and input, `alloc_buffer` for output
    4. Use `setBytes` for out_dim/in_dim (u32 constants, < 4KB)
    5. Compile PSO via `PsoKey::simple("matvec_q4_0")`
    6. Create command buffer, compute encoder, bind buffers at indices 0-4
    7. Dispatch: grid=(out_dim,1,1), threadgroup=(32,1,1)
    8. commit, waitUntilCompleted, `read_buffer_slice` -> Vec<f32>
    9. Add `pub mod matvec_q4_0;` to lib.rs, re-export `dispatch_matvec_q4_0`
    10. Add unit test `test_matvec_q4_0_basic`: construct Q4_0 blocks (scale=1.0, nibbles=[0..15]), input=ones, CPU reference dequant+dot, assert diff < 1e-3
    11. Add test `test_matvec_q4_0_576x576`: random Q4_0 data at SmolLM dims
  - **Files**: `crates/metal-attention-kernels/src/matvec_q4_0.rs`, `crates/metal-attention-kernels/src/lib.rs`
  - **Done when**: Both tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention-kernels matvec_q4_0 --test-threads=1 2>&1 | tail -20`
  - **Commit**: `feat(kernels): add dispatch wrapper and tests for matvec_q4_0`
  - _Requirements: FR-1, AC-1.2, NFR-4_
  - _Design: Fused Q4_0 Matvec Kernel_

- [x] 1.3 Create residual_add kernel, dispatch wrapper, and tests
  - **Do**:
    1. Create `residual.metal`: `kernel void residual_add(a, b, output, tid) { output[tid] = a[tid] + b[tid]; }`
    2. Dispatch: grid=(hidden_dim), threadgroup=(256)
    3. Create `residual.rs` with `dispatch_residual_add()` following same pattern as matvec
    4. Add `pub mod residual;` to lib.rs, re-export
    5. Unit test: `[1,2,3]+[10,20,30]=[11,22,33]`, exact match
  - **Files**: `crates/metal-attention-kernels/shaders/residual.metal`, `crates/metal-attention-kernels/src/residual.rs`, `crates/metal-attention-kernels/src/lib.rs`
  - **Done when**: Kernel compiles, test passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clean -p metal-attention-kernels && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention-kernels residual --test-threads=1 2>&1 | tail -20`
  - **Commit**: `feat(kernels): add residual_add kernel and dispatch wrapper`
  - _Requirements: FR-8_
  - _Design: Residual Add Kernel_

- [x] 1.4 Add optimized RMSNorm kernel with simdgroup reduction
  - **Do**:
    1. Add `rmsnorm_optimized` kernel to existing `rmsnorm.metal`
    2. 32 threads, each computes partial sum-of-squares over stride `for (i=tid; i<hidden_dim; i+=32)`
    3. `simd_sum()` for cooperative reduction of sum-of-squares
    4. Each thread normalizes its stride: `output[i] = (input[i] / rms) * weight[i]`
    5. Buffer bindings: input(0), weight(1), output(2), hidden_dim(3) as `constant uint&`, eps(4) as `constant float&`
    6. Dispatch: grid=(1,1,1), threadgroup=(32,1,1)
    7. Add `dispatch_rmsnorm_optimized()` to `norm.rs`
    8. Unit test: known RMS vector [3,4,0,...], compare vs CPU reference, tolerance 1e-5
  - **Files**: `crates/metal-attention-kernels/shaders/rmsnorm.metal`, `crates/metal-attention-kernels/src/norm.rs`
  - **Done when**: Kernel compiles, test passes with 1e-5 tolerance
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clean -p metal-attention-kernels && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention-kernels rmsnorm_optimized --test-threads=1 2>&1 | tail -20`
  - **Commit**: `feat(kernels): add simdgroup-cooperative rmsnorm_optimized kernel`
  - _Requirements: FR-6_
  - _Design: Optimized RMSNorm Kernel_

- [x] 1.5 [VERIFY] Quality checkpoint: build + clippy + existing tests
  - **Do**: Run build, clippy, and full test suite after 4 kernels created
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && cargo clippy --workspace && MTL_SHADER_VALIDATION=1 cargo test --workspace --test-threads=1 2>&1 | tail -30`
  - **Done when**: No errors, all existing tests still pass
  - **Commit**: `chore(kernels): pass quality checkpoint` (only if fixes needed)

- [x] 1.6 Create GPU RoPE kernel and dispatch wrapper
  - **Do**:
    1. Create `rope_apply.metal` with `rope_apply` kernel
    2. Operates in-place on `qk` buffer `[total_heads * head_dim]`
    3. Each thread handles one (cos,sin) pair: `head = tid / (head_dim/2)`, `pair = tid % (head_dim/2)`
    4. Guard: `if (head >= num_heads) return;`
    5. Compute angle: `position / pow(theta, 2.0 * pair / head_dim)`
    6. Rotate: `qk[idx0] = v0*cos - v1*sin`, `qk[idx1] = v0*sin + v1*cos`
    7. Buffer bindings: qk(0), num_heads(1), head_dim(2), position(3), theta(4)
    8. Dispatch: grid=(num_heads * head_dim / 2), threadgroup=(32)
    9. Create `gpu_rope.rs` with `dispatch_rope_apply()` standalone function
    10. Add `pub mod gpu_rope;` to lib.rs, re-export
    11. Unit test: apply RoPE at position=0 (should be identity: cos=1, sin=0), position=1 (verify rotation matches CPU reference)
  - **Files**: `crates/metal-attention-kernels/shaders/rope_apply.metal`, `crates/metal-attention-kernels/src/gpu_rope.rs`, `crates/metal-attention-kernels/src/lib.rs`
  - **Done when**: Kernel compiles, both tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clean -p metal-attention-kernels && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention-kernels gpu_rope --test-threads=1 2>&1 | tail -20`
  - **Commit**: `feat(kernels): add GPU RoPE position encoding kernel`
  - _Requirements: FR-13_
  - _Design: GPU RoPE Kernel_

- [x] 1.7 Create decode_attention kernel and dispatch wrapper
  - **Do**:
    1. Create `decode_attention.metal` with `decode_attention` kernel
    2. Grid=(num_heads), Threadgroup=(32) -- 1 simdgroup per head
    3. GQA: `kv_head = head_id / group_size` where `group_size = num_heads / num_kv_heads`
    4. Phase 1: Compute Q.K^T scores -- each thread handles subset of kv_len positions `for (pos=tid; pos<kv_len; pos+=32)`, dot product over head_dim, store `scores[pos]` in threadgroup memory
    5. Phase 2: `simd_max` for max_score, then softmax in threadgroup memory: `exp(score - max)`, `simd_sum` for denominator
    6. Phase 3: Weighted V sum -- each thread handles subset of head_dim `for (d=tid; d<head_dim; d+=32)`, accumulate over all kv positions
    7. Declare threadgroup memory: `threadgroup float scores[2048]` (max kv_len, 8KB per threadgroup)
    8. Buffer bindings: q(0), k_cache(1), v_cache(2), output(3), num_heads(4), num_kv_heads(5), head_dim(6), kv_len(7), scale(8)
    9. Create `decode_attention.rs` with `dispatch_decode_attention()` standalone function
    10. Add `pub mod decode_attention;` to lib.rs, re-export
    11. Unit test: 2 heads, head_dim=4, kv_len=3, known Q/K/V, verify output matches CPU softmax+weighted-V reference
    12. Unit test with GQA: 9 Q heads, 3 KV heads, head_dim=64, kv_len=5, verify grouping correct
  - **Files**: `crates/metal-attention-kernels/shaders/decode_attention.metal`, `crates/metal-attention-kernels/src/decode_attention.rs`, `crates/metal-attention-kernels/src/lib.rs`
  - **Done when**: Kernel compiles, both tests pass with tolerance < 1e-3
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clean -p metal-attention-kernels && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention-kernels decode_attention --test-threads=1 2>&1 | tail -20`
  - **Commit**: `feat(kernels): add GPU decode attention kernel with GQA and simdgroup softmax`
  - _Requirements: FR-7_
  - _Design: GPU Decode Attention Kernel_

- [x] 1.8 [VERIFY] Quality checkpoint: all kernels green
  - **Do**: Run full test suite to ensure all 5 new kernels work and no regressions
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && cargo clippy --workspace && MTL_SHADER_VALIDATION=1 cargo test --workspace --test-threads=1 2>&1 | tail -30`
  - **Done when**: All existing + new kernel tests pass
  - **Commit**: `chore(kernels): pass quality checkpoint` (only if fixes needed)

- [x] 1.9 Implement GpuWeightStore with zero-copy Q4_0 buffers from GGUF
  - **Do**:
    1. Create `gpu_weight_store.rs` in `crates/metal-attention/src/`
    2. Define `GpuWeightStore` struct with:
       - `AttnProjBuffers` { q, k, v, o } per layer (Retained<MTLBuffer>, raw Q4_0 bytes)
       - `FfnBuffers` { gate, up, down } per layer (Q4_0)
       - `NormBuffers` { attn_norm, ffn_norm } per layer (F32)
       - `embed_buf` (F32), `lm_head_buf` (Q4_0), `final_norm_buf` (F32)
    3. Implement `from_gguf()`:
       a. Accept `&GgufFile`, `&ModelConfig`, device
       b. For each tensor: `gguf.tensor_data(tensor_info)` -> `&[u8]`
       c. Check page alignment: `ptr as usize % page_size == 0` where `page_size = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) as usize }` (16KB on Apple Silicon ARM64, NOT 4096)
       d. If aligned: `unsafe { create_weight_buffer(device, ptr as *mut _, len) }` -- zero-copy
       e. If not aligned: `alloc_buffer_with_data(device, bytes)` with warning
       f. F32 norm weights: always `alloc_buffer_with_data` (small, no need for zero-copy)
       g. Embedding: F32, use `alloc_buffer_with_data`
    4. Store `Arc<GgufFile>` to keep mmap alive
    5. Accessor methods: `attn_proj(layer)`, `ffn(layer)`, `norm(layer)`, `embed()`, `lm_head()`, `final_norm()`
    6. Add `pub mod gpu_weight_store;` to metal-attention/src/lib.rs
  - **Files**: `crates/metal-attention/src/gpu_weight_store.rs`, `crates/metal-attention/src/lib.rs`
  - **Done when**: Compiles without errors
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(inference): implement GpuWeightStore with zero-copy Q4_0 buffers`
  - _Requirements: FR-3, FR-5, AC-2.1_
  - _Design: GpuWeightStore_

- [x] 1.10 Implement GpuKVCache for GPU-resident key/value storage
  - **Do**:
    1. Create `gpu_kv_cache.rs` in `crates/metal-attention/src/`
    2. Define `GpuKVCache` struct:
       - `k_buf`: Retained<MTLBuffer> [max_seq, kv_dim] F32
       - `v_buf`: Retained<MTLBuffer> [max_seq, kv_dim] F32
       - `len`: usize (current sequence length)
       - `max_len`: usize (2048)
       - `kv_dim`: usize (num_kv_heads * head_dim = 3*64 = 192 for SmolLM)
    3. Implement `new(device, max_len, kv_dim)`: alloc k_buf and v_buf
    4. Implement `append_kv(&mut self, encoder, k_vec_buf, v_vec_buf)`:
       - Copy k_vec_buf into k_buf at offset `len * kv_dim * 4` bytes using blit encoder
       - Copy v_vec_buf into v_buf at same offset pattern
       - Increment len
       - Note: must end compute encoder, do blit copy, then start new compute encoder
       - Alternative: write a tiny `kv_cache_append` Metal kernel that copies a single row (avoids encoder switching)
    5. Implement `k_buffer()`, `v_buffer()`, `current_len()` accessors
    6. Define `GpuKVCacheSet` wrapping Vec<GpuKVCache> (one per layer)
    7. Add `pub mod gpu_kv_cache;` to lib.rs
  - **Files**: `crates/metal-attention/src/gpu_kv_cache.rs`, `crates/metal-attention/src/lib.rs`
  - **Done when**: Compiles without errors
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(inference): implement GPU-resident KV cache per-layer`
  - _Requirements: FR-7_
  - _Design: GPU KV Cache_

- [x] 1.11 [VERIFY] Quality checkpoint: build + clippy after infrastructure modules
  - **Do**: Verify GpuWeightStore and GpuKVCache compile cleanly
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && cargo clippy --workspace 2>&1 | tail -20`
  - **Done when**: No errors from new modules
  - **Commit**: `chore(inference): pass quality checkpoint` (only if fixes needed)

- [x] 1.12 Implement GpuForwardPass with single command buffer pipeline (full GPU)
  - **Do**:
    1. Create `gpu_forward_pass.rs` in `crates/metal-attention/src/`
    2. Define `GpuForwardPass` struct:
       - `device: &'static GpuDevice`
       - `pso_cache: PsoCache`
       - `cmd_manager: CommandManager`
       - `weight_store: GpuWeightStore`
       - `kv_caches: Vec<GpuKVCache>` (one per layer)
       - `hidden_a, hidden_b: Retained<MTLBuffer>` (ping-pong, 576*4=2304B each)
       - `scratch_pool: BufferPool`
       - `config: ModelConfig` (hidden_size, num_heads, num_kv_heads, head_dim, num_layers, vocab_size, intermediate_size)
       - `position: usize` (current decode position)
    3. Implement `from_gguf(path, device)`:
       a. Open GgufFile, extract config from metadata
       b. Build GpuWeightStore::from_gguf()
       c. Alloc hidden_a, hidden_b (2304B each)
       d. Init kv_caches: 30 GpuKVCache (max_len=2048, kv_dim=192)
       e. Init BufferPool, PsoCache
       f. Prewarm PSOs: matvec_q4_0, rmsnorm_optimized, residual_add, decode_attention, rope_apply, ffn_silu
    4. Implement `forward_token(&mut self, token_id: u32) -> Vec<f32>`:
       a. CPU embed lookup: `embed_weight[token_id * hidden_dim .. (token_id+1) * hidden_dim]` -> write to hidden_a via `contents()` memcpy
       b. Create ONE command buffer via cmd_manager.begin_frame()
       c. Create ONE compute encoder
       d. For each layer (0..num_layers):
          **Attention block:**
          i.   Encode rmsnorm_optimized: hidden_a -> hidden_b
          ii.  Encode matvec_q4_0(Q proj): hidden_b -> scratch_q [num_heads*head_dim = 576]
          iii. Encode matvec_q4_0(K proj): hidden_b -> scratch_k [num_kv_heads*head_dim = 192]
          iv.  Encode matvec_q4_0(V proj): hidden_b -> scratch_v [num_kv_heads*head_dim = 192]
          v.   Encode rope_apply on scratch_q (num_heads=9, head_dim=64, position)
          vi.  Encode rope_apply on scratch_k (num_kv_heads=3, head_dim=64, position)
          vii. Append K/V to kv_cache[layer] -- encode small copy kernel or use blit
          viii.Encode decode_attention: scratch_q, kv_cache_k, kv_cache_v -> scratch_attn_out [576]
          ix.  Encode matvec_q4_0(O proj): scratch_attn_out -> scratch_o [576]
          x.   Encode residual_add: hidden_a + scratch_o -> hidden_a
          **FFN block:**
          xi.  Encode rmsnorm_optimized: hidden_a -> hidden_b
          xii. Encode matvec_q4_0(gate): hidden_b -> scratch_gate [1536]
          xiii.Encode matvec_q4_0(up): hidden_b -> scratch_up [1536]
          xiv. Encode ffn_silu: scratch_gate, scratch_up -> scratch_hidden [1536]
          xv.  Encode matvec_q4_0(down): scratch_hidden -> scratch_ffn [576]
          xvi. Encode residual_add: hidden_a + scratch_ffn -> hidden_a
       e. Final rmsnorm_optimized: hidden_a -> hidden_b
       f. matvec_q4_0(lm_head): hidden_b -> logits_buf [49152]
       g. endEncoding, commit, waitUntilCompleted
       h. Read back logits_buf -> Vec<f32> (192KB)
       i. Increment position
    5. For inline encoding: each kernel dispatch = `encoder.setComputePipelineState(pso)` + `setBuffer_offset_atIndex` + `setBytes` + `dispatchThreads`. No new command buffers per op.
    6. For KV cache append within single encoder: write a tiny `kv_cache_copy` kernel that copies scratch_k/v into cache at offset, OR break encoder -> blit -> new encoder per layer. Prefer tiny kernel to avoid encoder break.
    7. Add `pub mod gpu_forward_pass;` to lib.rs, re-export GpuForwardPass
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`, `crates/metal-attention/src/lib.rs`
  - **Done when**: Compiles, `forward_token()` signature correct
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build 2>&1 | tail -10`
  - **Commit**: `feat(inference): implement GpuForwardPass full-GPU pipeline (zero sync points)`
  - _Requirements: FR-2, FR-4, FR-7, FR-12, FR-13, AC-3.1, AC-3.2, AC-3.3, AC-4.1, AC-4.2, AC-4.3_
  - _Design: GpuForwardPass, Data Flow_

- [x] 1.13 Wire GPU forward path into CLI and measure first tok/s
  - **Do**:
    1. Modify `src/main.rs`: add `--gpu` flag (bool, default false) to Run and Bench commands
    2. In Run with --gpu: construct `GpuForwardPass::from_gguf()`, implement simple decode loop
    3. Prefill: call `forward_token()` for each prompt token
    4. Decode loop: greedy sample from logits, feed back, print tokens
    5. Print tok/s to stderr after generation (time decode loop, count tokens)
    6. In Bench with --gpu and -m: load GpuForwardPass, measure decode tok/s over gen_length tokens
    7. Import GpuForwardPass from metal_attention crate
  - **Files**: `src/main.rs`
  - **Done when**: `cargo run -- bench --gpu -m models/SmolLM-135M.Q4_0.gguf` prints tok/s
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && cargo run -- bench --gpu -m models/SmolLM-135M.Q4_0.gguf --gen-length 5 --iterations 1 --seq-lengths 3 2>&1 | tail -10`
  - **Commit**: `feat(cli): wire GPU forward path into run/bench with --gpu flag`
  - _Requirements: AC-6.1, AC-6.2_
  - _Design: Implementation Steps_

- [x] 1.14 [VERIFY] Quality checkpoint: full build + test + GPU run
  - **Do**: Verify entire pipeline builds and existing tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && cargo clippy --workspace && MTL_SHADER_VALIDATION=1 cargo test --workspace --test-threads=1 2>&1 | tail -30`
  - **Done when**: All commands pass
  - **Commit**: `chore(inference): pass quality checkpoint` (only if fixes needed)

- [x] 1.15 POC Checkpoint: validate GPU path correct output + tok/s
  - **Do**:
    1. Run GPU forward pass on SmolLM-135M with a simple prompt
    2. Run CPU forward pass (existing HybridModel) with same prompt
    3. Compare greedy-sampled tokens for 5+ decode steps
    4. If tokens don't match: debug by comparing per-layer outputs
    5. Measure and record initial tok/s number
    6. Record results in .progress.md
  - **Done when**: GPU and CPU paths produce matching greedy tokens for 5+ steps AND tok/s measured
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo run -- bench --gpu -m models/SmolLM-135M.Q4_0.gguf --gen-length 10 --iterations 1 --seq-lengths 3 2>&1`
  - **Commit**: `feat(inference): complete full-GPU inference POC`
  - _Requirements: AC-1.2, AC-1.3, AC-5.3_

## Phase 2: Refactoring

After POC validated. Inline encoder dispatch helpers, PSO prewarm, error handling.

- [x] 2.1 Refactor inline encoder dispatch helpers
  - **Do**:
    1. Extract encode helper functions in `gpu_forward_pass.rs` (or new `gpu_dispatch.rs`):
       - `encode_matvec_q4_0(encoder, pso, weight_buf, input_buf, output_buf, out_dim, in_dim)`
       - `encode_rmsnorm(encoder, pso, input_buf, weight_buf, output_buf, hidden_dim, eps)`
       - `encode_residual_add(encoder, pso, a_buf, b_buf, output_buf, dim)`
       - `encode_ffn_silu(encoder, pso, gate_buf, up_buf, output_buf, dim)`
       - `encode_decode_attention(encoder, pso, q_buf, k_cache, v_cache, output_buf, params...)`
       - `encode_rope_apply(encoder, pso, qk_buf, num_heads, head_dim, position, theta)`
    2. Each helper: sets pipeline state, binds buffers, sets bytes for constants, dispatches
    3. No command buffer creation -- caller owns encoder
    4. Refactor `forward_token()` loop to use these helpers
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: forward_token is clean and uses helper functions
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && cargo run -- bench --gpu -m models/SmolLM-135M.Q4_0.gguf --gen-length 5 --iterations 1 --seq-lengths 3 2>&1 | tail -10`
  - **Commit**: `refactor(inference): extract encoder-level dispatch helpers`
  - _Design: Single Command Buffer Pipeline_

- [x] 2.2 Add error handling and buffer validation
  - **Do**:
    1. Return `Result<Vec<f32>, String>` from `forward_token()`
    2. Validate command buffer status after waitUntilCompleted
    3. Check Q4_0 block count: `assert_eq!(tensor_bytes.len(), expected_blocks * 18)`
    4. Log page alignment fallback warnings
    5. Add GPU family check in `from_gguf()`: verify Apple Family 7+ for simd_sum
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`, `crates/metal-attention/src/gpu_weight_store.rs`
  - **Done when**: All error paths return descriptive messages
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && cargo clippy --workspace 2>&1 | tail -20`
  - **Commit**: `refactor(inference): add error handling and buffer validation`
  - _Design: Error Handling_

- [x] 2.3 [VERIFY] Quality checkpoint: build + clippy + tests after refactoring
  - **Do**: Full quality suite after refactoring
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && cargo clippy --workspace && MTL_SHADER_VALIDATION=1 cargo test --workspace --test-threads=1 2>&1 | tail -30`
  - **Done when**: All commands pass
  - **Commit**: `chore(inference): pass quality checkpoint` (only if fixes needed)

## Phase 3: Testing

- [x] 3.1 GPU vs CPU correctness integration test
  - **Do**:
    1. Create `crates/metal-attention/tests/gpu_correctness.rs`
    2. Test `test_gpu_vs_cpu_logits`:
       - Load SmolLM-135M via CPU (HybridModel::from_gguf) and GPU (GpuForwardPass::from_gguf)
       - Prefill tokens [1, 2, 3] on both paths
       - Compare logits: max abs diff < 1e-2 (Q4_0 fused may have higher error than F32 CPU)
    3. Test `test_gpu_vs_cpu_greedy_match`:
       - Decode 10 tokens greedy on both paths
       - Assert same token sampled at each step
    4. Mark `#[ignore]` since requires model file at `models/SmolLM-135M.Q4_0.gguf`
  - **Files**: `crates/metal-attention/tests/gpu_correctness.rs`
  - **Done when**: Tests pass with real GGUF model
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention --test gpu_correctness -- --ignored --test-threads=1 2>&1 | tail -20`
  - **Commit**: `test(inference): add GPU vs CPU correctness integration test`
  - _Requirements: AC-1.2, AC-1.3, AC-5.3, NFR-4_
  - _Design: Test Strategy - Integration Tests_

- [x] 3.2 Per-kernel unit tests with edge cases
  - **Do**:
    1. In `matvec_q4_0.rs` add tests:
       - `test_matvec_q4_0_1536x576`: gate/up projection dims
       - `test_matvec_q4_0_49152x576`: lm_head (largest matvec)
       - `test_matvec_q4_0_192x576`: K/V projection (kv_heads=3)
       - `test_matvec_q4_0_zero_scale`: zero scale -> zero output
    2. In `decode_attention.rs` add tests:
       - `test_decode_attention_single_head`: 1 head, kv_len=1, trivial case
       - `test_decode_attention_long_context`: kv_len=512, verify no overflow
    3. In `gpu_rope.rs` add test:
       - `test_rope_apply_multi_position`: verify rotation angles at positions 0,1,10,100
    4. In `residual.rs` add: `test_residual_add_576`
    5. In `norm.rs` add: `test_rmsnorm_optimized_576`, `test_rmsnorm_optimized_identity_weight`
  - **Files**: `crates/metal-attention-kernels/src/matvec_q4_0.rs`, `crates/metal-attention-kernels/src/decode_attention.rs`, `crates/metal-attention-kernels/src/gpu_rope.rs`, `crates/metal-attention-kernels/src/residual.rs`, `crates/metal-attention-kernels/src/norm.rs`
  - **Done when**: All edge-case tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention-kernels --test-threads=1 2>&1 | tail -30`
  - **Commit**: `test(kernels): add edge-case tests for all GPU inference kernels`
  - _Requirements: NFR-6_
  - _Design: Test Strategy - Unit Tests_

- [x] 3.3 [VERIFY] Quality checkpoint: all tests green
  - **Do**: Full test suite including all new tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test --workspace --test-threads=1 2>&1 | tail -30`
  - **Done when**: All tests pass
  - **Commit**: `chore(inference): pass quality checkpoint` (only if fixes needed)

- [x] 3.4 Add decode throughput benchmark with real GGUF
  - **Do**:
    1. Update or create `benches/inference.rs` to use GpuForwardPass with real GGUF
    2. Add `bench_gpu_decode`: load SmolLM-135M via GpuForwardPass, warmup 3 tokens, benchmark 100 decode tokens
    3. Report tok/s via Criterion
    4. Add `bench_cpu_decode` for comparison (same model via HybridModel CPU path)
    5. Ensure bench is marked `#[ignore]` or conditionally skipped if model file absent
  - **Files**: `benches/inference.rs`
  - **Done when**: `cargo bench --bench inference` reports GPU vs CPU tok/s
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo bench --bench inference 2>&1 | tail -20`
  - **Commit**: `test(bench): add real-GGUF GPU vs CPU decode throughput benchmark`
  - _Requirements: AC-1.4, AC-6.1, AC-6.2, AC-6.3_
  - _Design: Test Strategy - benchmark_tok_s_

## Phase 4: Quality Gates

- [x] 4.1 [VERIFY] Full local CI: build + clippy + test
  - **Do**: Run complete local CI suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && cargo clippy --workspace && MTL_SHADER_VALIDATION=1 cargo test --workspace --test-threads=1`
  - **Done when**: All commands pass with no errors
  - **Commit**: `fix(inference): address lint/type issues` (if fixes needed)

- [x] 4.2 Verify no regressions in existing functionality
  - **Do**:
    1. Run existing CPU path: `cargo run -- run -m models/SmolLM-135M.Q4_0.gguf -p "Hello" -n 10 --temp 0`
    2. Verify output is coherent (not garbage)
    3. Run existing bench: `cargo run -- bench --synthetic --gen-length 10 --iterations 1`
    4. Run ignored GGUF loading tests: `cargo test --test gguf_loading -- --ignored --test-threads=1`
  - **Done when**: All existing functionality preserved
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo run -- bench --synthetic --gen-length 5 --iterations 1 2>&1 | tail -10`
  - **Commit**: none (verification only)
  - _Requirements: FR-14, AC-5.1, NFR-7_

- [x] 4.3 Create PR and verify CI
  - **Do**:
    1. Verify current branch is feature branch: `git branch --show-current`
    2. If on default branch, STOP and alert user
    3. Stage all new/modified files
    4. Push branch: `git push -u origin <branch-name>`
    5. Create PR with summary including: kernels added, pipeline architecture, tok/s measurement
    6. PR body should include measured tok/s numbers from POC checkpoint
  - **Verify**: `gh pr checks --watch` or `gh pr checks` to poll
  - **Done when**: All CI checks green, PR ready for review
  - **If CI fails**: Read failure details, fix locally, push, re-verify

## Phase 5: PR Lifecycle

- [x] 5.1 Monitor CI and fix failures
  - **Do**:
    1. After PR creation, check CI: `gh pr checks`
    2. If any check fails, read logs, fix, push
    3. Re-verify until all green
  - **Verify**: `gh pr checks` shows all green
  - **Done when**: CI passes

- [ ] 5.2 [VERIFY] AC checklist verification
  - **Do**: Programmatically verify each acceptance criterion:
    1. AC-1.2: GPU matvec matches CPU < 1e-3 -- `cargo test -p metal-attention-kernels matvec_q4_0`
    2. AC-1.3: Greedy output matches CPU -- `cargo test -p metal-attention --test gpu_correctness -- --ignored`
    3. AC-2.1: Zero-copy mmap -- grep for `create_weight_buffer` in gpu_weight_store.rs
    4. AC-3.1: Single command buffer -- grep `begin_frame` in gpu_forward_pass.rs (should appear once per forward_token)
    5. AC-3.3: Only 2 CPU-GPU transfers -- verify embed write + logits read only
    6. AC-4.1: Hidden buffers allocated once -- verify hidden_a/hidden_b in struct
    7. AC-4.3: Intermediates GPU-resident -- no read_buffer_slice in hot path loop
    8. AC-5.1: CPU path preserved -- existing tests pass
    9. AC-5.2: GPU via separate entry point -- GpuForwardPass::forward_token() exists
    10. AC-6.1: bench reports tok/s -- --gpu flag in bench subcommand
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test --workspace --test-threads=1 2>&1 | tail -10`
  - **Done when**: All ACs confirmed met
  - **Commit**: none (verification only)

- [ ] 5.3 [VERIFY] Final validation: tok/s measurement
  - **Do**:
    1. Run GPU decode benchmark with real GGUF
    2. Record tok/s number
    3. Compare against 389 tok/s target (AC-1.1)
    4. With full GPU attention (zero sync points), expect 200-500+ tok/s range
    5. Document result in .progress.md
    6. If below target, document what optimizations remain (fused RMSNorm+Matvec, GPU argmax, etc)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo run -- bench --gpu -m models/SmolLM-135M.Q4_0.gguf --gen-length 100 --iterations 3 --seq-lengths 3 2>&1`
  - **Done when**: tok/s recorded and compared against target
  - **Commit**: none (measurement only)

## Notes

- **POC shortcuts taken**:
  - Hardcoded assumptions for SmolLM dimensions in some places
  - No GPU-side argmax (full 192KB logits readback per token)
  - KV cache threadgroup memory capped at 2048 (8KB per head) -- sufficient for SmolLM
  - Single-threaded synchronous waitUntilCompleted (no async pipeline)
  - Embedding lookup on CPU with memcpy to GPU buffer

- **Full GPU Phase 1 advantage**:
  - Zero GPU-CPU sync points in decode hot path (vs 30 sync points in CPU hybrid)
  - Single command buffer encodes ALL ops for ALL layers
  - Only CPU-GPU transfers: embedding in (2.3KB) + logits out (192KB) = 199KB/token
  - Expected throughput: 200-500+ tok/s (vs ~333 tok/s cap with CPU hybrid)

- **Production TODOs** (beyond this spec):
  - Fused RMSNorm+Matvec kernel (save intermediate buffer write)
  - GPU-side argmax (eliminate 192KB logits readback)
  - Batched prefill with simdgroup_matrix
  - KV cache paging for long sequences
  - JSON output mode for bench (--json flag)

- **Critical build note**: Adding new .metal files requires `cargo clean -p metal-attention-kernels` before build. Handled in verify commands for tasks 1.1, 1.3, 1.4, 1.6, 1.7.

- **Test environment**: GPU tests require `MTL_SHADER_VALIDATION=1` and `--test-threads=1`. Tests using real GGUF are `#[ignore]` and need `-- --ignored`.

- **GGUF model location**: `models/SmolLM-135M.Q4_0.gguf` (87MB, already downloaded)

- **KV cache append strategy**: Prefer a tiny `kv_cache_copy` Metal kernel to copy K/V vectors into cache within the compute encoder, avoiding encoder break per layer. If encoder break needed for blit copy, it adds 30 encoder creation cycles per token (small overhead but measurable).

- **SmolLM-135M dimensions**: hidden=576, heads=9, kv_heads=3, head_dim=64, layers=30, ffn=1536, vocab=49152
