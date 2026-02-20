---
spec: gpu-perf-phase3
phase: research
created: 2026-02-15
generated: auto
---

# Research: gpu-perf-phase3

## Executive Summary

Multi-token batched forward pass (`forward_prompt`) can reach 1000+ tok/s prefill on SmolLM-135M Q4_0 by exploiting SLC cache reuse. Experiments in `path_to_1000_toks.rs` show near-linear scaling: batch=4 gives 3.9x, batch=8 gives 7.1x per-token matvec speedup. The proven `multi_token_matvec_q4_0` kernel already exists in `bandwidth_test.metal`. Three changes needed: production multi-token kernels, `forward_prompt()` method, CLI integration.

## Codebase Analysis

### Existing Patterns

**Single-token forward pass** (`gpu_forward_pass.rs`)
- `forward_token()` returns `Vec<f32>` logits, `forward_token_greedy()` returns `u32` argmax
- Single command buffer + single compute encoder for all 30 layers
- Ping-pong `hidden_a`/`hidden_b` buffers (576 floats each)
- Scratch buffers: `scratch_q` (576), `scratch_k/v` (192), `scratch_gate/up/silu` (1536)
- Encoder helpers: `encode_rmsnorm`, `encode_matvec_q4_0`, `encode_batched_matvec_q4_0`, `encode_rope_dual`, `encode_decode_attention`, `encode_ffn_silu`, `encode_matvec_q4_0_accumulate`, `encode_silu_matvec_q4_0_accumulate`

**Proven multi-token kernel** (`bandwidth_test.metal:225-281`)
- `multi_token_matvec_q4_0`: 256 threads, 8 simdgroups, 8 rows/TG
- Input `[batch_size, in_dim]`, output `[batch_size, out_dim]`, loops over `batch_size` tokens
- Weight reads cached in SLC after first token: subsequent tokens near-free
- Buffer bindings: weight[0], input[1], output[2], out_dim[3], in_dim[4], batch_size[5]

**Accumulate variants** already exist for single-token:
- `matvec_q4_0_accumulate.metal`: `output[row] += sum` (O-proj + residual)
- `silu_matvec_q4_0_accumulate.metal`: fused SiLU + down-proj + accumulate

**KV cache** (`gpu_kv_cache.rs`)
- `GpuKVCache` with `encode_kv_append()` (GPU-side) and `append_kv()` (CPU-side)
- Sequential per-token append (causal constraint: kv_len increases per token)

### Dependencies
- `objc2-metal` bindings for buffer management (`alloc_buffer`, `alloc_buffer_private`, `set_buffer`)
- `PsoCache` for pipeline state objects (prewarm at init)
- Existing `GgufTokenizer::encode()` for prompt tokenization

### Constraints
- Attention is inherently sequential (causal: token N needs K/V from tokens 0..N)
- RMSNorm, RoPE, SiLU are per-token operations (cheap, not worth batching)
- Only QKV/O/gate/up/down projections benefit from multi-token batching
- KV cache max_len=2048 limits prompt length
- Must NOT break existing `forward_token()`/`forward_token_greedy()` paths

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | Kernel proven in experiments, near-linear scaling measured |
| Effort Estimate | M | ~15 tasks, kernel already exists, mainly Rust plumbing |
| Risk Level | Low | No architectural changes to existing paths |

## Key Experimental Results

| Experiment | Finding |
|-----------|---------|
| M4 bandwidth ceiling | 200+ GB/s (SLC cache boost, not 120 GB/s LPDDR5X spec) |
| Dispatch overhead | 302 dispatches = 383 us (1.27 us/dispatch), ~18% overhead |
| Small matrix bandwidth | 0.6-5.1 GB/s on 60-486 KB matrices vs 200+ GB/s peak |
| Batch=4 matvec | 3.9x per-token speedup (weights cached in SLC after token 1) |
| Batch=8 matvec | 7.1x per-token speedup |
| Current single-token | 462-491 tok/s decode (already beats llama.cpp 389 tok/s) |
| Theoretical batch=4 prefill | ~1800 tok/s |

## Recommendations

1. Move `multi_token_matvec_q4_0` from `bandwidth_test.metal` to production shader file
2. Add accumulate variant for O-proj and down-proj residual fusion
3. Implement `forward_prompt()` as new method (not replacing existing paths)
4. Use Metal buffer offsets for per-token operations within batch buffers
5. Target realistic 1000+ tok/s (conservative given theoretical 1800+)
