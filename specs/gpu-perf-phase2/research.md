---
spec: gpu-perf-phase2
phase: research
created: 2026-02-15
generated: auto
---

# Research: gpu-perf-phase2

## Executive Summary

SmolLM-135M Q4_0 GPU inference currently achieves 257 tok/s on Apple M4. Three targeted optimizations can close the gap to 389+ tok/s: GPU-side argmax (eliminate 192KB logits readback), StorageModePrivate for scratch buffers (eliminate CPU coherency overhead), and fused RMSNorm+Matvec kernel (eliminate 61 intermediate buffer writes per token). All three are feasible within existing codebase patterns.

## Codebase Analysis

### Existing Patterns
- `gpu_forward_pass.rs`: Single command buffer + single compute encoder for all 30 layers. Ping-pong hidden_a/hidden_b buffers. 12 scratch buffers allocated once, reused.
- `buffer.rs`: `alloc_buffer()` uses `StorageModeShared` exclusively. `alloc_buffer_with_data()` same. No `alloc_buffer_private()` exists yet.
- `pipeline.rs`: `PsoCache` with `PsoKey::simple()` and function constant builders. `prewarm()` compiles PSOs at startup.
- Metal shaders: `rmsnorm_optimized` uses simd_sum with 32 threads. `matvec_q4_0` uses 1 threadgroup per output row with simd_sum reduction. Both follow consistent buffer binding conventions.
- `dispatch.rs`: `set_buffer()`, `set_bytes()`, `dispatch_1d()` helpers for encoder operations.
- `main.rs`: CPU `argmax()` function on Vec<f32> after `read_buffer_slice()` readback. Bench loop uses same pattern.

### Current Hot Path (per token)
1. CPU embed_lookup: write 2.3KB to hidden_a via `contents()` memcpy
2. GPU encode: 30 layers x (rmsnorm + Q/K/V matvec + rope + kv_append + decode_attn + O matvec + residual + buffer_copy + rmsnorm + gate/up matvec + silu + down matvec + residual + buffer_copy) = ~420 kernel dispatches
3. Final rmsnorm + lm_head matvec (1 dispatch each)
4. **GPU->CPU readback**: `read_buffer_slice(&logits_buf, 49152)` = 192KB via `contents()` memcpy
5. **CPU argmax**: linear scan of 49152 f32 values

### Dependencies
- `objc2-metal` crate: `MTLResourceOptions::StorageModePrivate` available
- `simd_sum` (Apple Family 7+): already validated in `from_gguf()`
- Existing PSO cache handles new kernel names transparently

### Constraints
- Weight buffers are zero-copy mmap (`create_weight_buffer`): cannot change to Private
- embed_lookup uses `contents()`: hidden_a must stay Shared (CPU write needed)
- logits_buf currently needs `contents()` for readback: must change to GPU-side argmax first, then can go Private
- Debug path (`forward_token_debug`) uses `read_buffer_slice()` on scratch buffers: must remain Shared in debug mode, or use separate debug buffers

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| GPU Argmax | High viability, S effort | 2-stage parallel reduction, ~120 lines Metal, ~80 lines Rust. Well-understood pattern. |
| StorageModePrivate | High viability, S effort | ~30 lines change in buffer.rs + gpu_forward_pass.rs. Must gate behind non-debug path. |
| Fused RMSNorm+Matvec | Medium viability, M effort | 400-600 lines Metal. Risk: numerical drift through 30 layers. Needs careful validation. |

## Recommendations
1. Implement GPU argmax first -- highest impact, lowest risk, enables Private logits_buf
2. Switch scratch buffers to Private second -- trivial change, orthogonal to argmax
3. Fused kernel last -- highest complexity, build on validated POC foundation
