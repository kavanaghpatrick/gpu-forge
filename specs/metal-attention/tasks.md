# Tasks: metal-attention

Hybrid model inference engine for Apple Silicon. Composable attention traits compiled to zero-overhead Metal kernels via function constants.

**Project dir**: `/Users/patrickkavanagh/gpu_kernel/metal-attention/`
**Proto reference**: `proto/` (8 validated prototypes, not linked to workspace)

---

## Phase 1: Foundation (Workspace + Traits + GPU Infra)

Focus: Workspace builds, shaders compile, GPU device initializes, traits defined. No model logic yet.

### Task 1.1: Create workspace skeleton and crate structure
- [x] Create 6-crate Cargo workspace under `metal-attention/`
  - Root `Cargo.toml` with workspace members + shared deps (objc2-metal 0.3, memmap2, clap 4, thiserror, tracing, criterion)
  - `crates/metal-attention-traits/` - pure Rust, zero Metal deps
  - `crates/metal-attention-kernels/` - Metal GPU layer with build.rs
  - `crates/metal-attention-gguf/` - mmap parser
  - `crates/metal-attention-models/` - model implementations
  - `crates/metal-attention/` - inference engine library
  - `src/main.rs` - CLI binary depending on the lib crate
  - Each crate gets minimal `Cargo.toml` + `src/lib.rs` with a doc comment
  - Add `.gitignore` for `target/`, `*.metallib`, `*.air`
- **Do**:
  1. Create workspace root `Cargo.toml` per TECH.md spec with all workspace.dependencies
  2. Create each crate directory with minimal `Cargo.toml` referencing workspace deps
  3. Create stub `src/lib.rs` in each crate (just `//! crate doc` + empty module)
  4. Create `src/main.rs` with `fn main() { println!("metal-attention"); }`
  5. Ensure `proto/` is NOT in workspace members (reference only)
- **Files**: `Cargo.toml`, `crates/*/Cargo.toml`, `crates/*/src/lib.rs`, `src/main.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo check --workspace`
- **Commit**: `feat(workspace): create 6-crate workspace skeleton`
- _Requirements: R1 (workspace structure)_
- _Design: Workspace Crate Structure_

### Task 1.2: Define core trait hierarchy in traits crate
- [x] Implement `SequenceBlock`, `LinearSequenceModel`, `SoftmaxAttention` traits
  - `src/lib.rs` - re-exports
  - `src/types.rs` - TensorView, DType, BlockConfig
  - `src/sequence.rs` - trait SequenceBlock with forward_prefill/forward_decode/init_state
  - `src/linear.rs` - trait LinearSequenceModel: SequenceBlock
  - `src/attention.rs` - trait SoftmaxAttention: SequenceBlock with KVCacheMode, PositionEncoding, GQAConfig
  - `src/schedule.rs` - LayerSchedule, LayerType enum with periodic/explicit/pure_transformer/pure_linear constructors
  - All types per TECH.md trait hierarchy section
- **Do**:
  1. Implement all types from TECH.md Trait Hierarchy section
  2. Add unit tests: TensorView construction, LayerSchedule::periodic(32,7) produces correct 7:1 pattern, type size assertions
  3. Ensure crate has zero dependencies on Metal (pure Rust)
- **Files**: `crates/metal-attention-traits/src/{lib,types,sequence,linear,attention,schedule}.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test -p metal-attention-traits`
- **Commit**: `feat(traits): define SequenceBlock, LinearSequenceModel, SoftmaxAttention trait hierarchy`
- _Requirements: R1 (trait hierarchy), R10 (LayerSchedule)_
- _Design: Core Trait Hierarchy, Hybrid Model Composition_

### Task 1.3: Port Metal build system and shader compilation
- [x] Port `build.rs` from proto to kernels crate, copy shaders with `types.h` extensions
  - Copy proto `build.rs` to `crates/metal-attention-kernels/build.rs`, adapt paths
  - Copy proto shaders to `crates/metal-attention-kernels/shaders/`: flash_attention.metal, linear_attention.metal, paged_attention.metal, paged_reduce.metal, rope.metal, gqa_remap.metal, types.h
  - Extend `types.h` with `LayerParams` and `SSMParams` structs per TECH.md
  - Add new stub shaders: rmsnorm.metal, ffn.metal, embedding.metal, matmul.metal, dequantize.metal (minimal valid kernels with correct function signatures)
  - Export `METALLIB_PATH` env var from build.rs
- **Do**:
  1. Copy and adapt build.rs (add METALLIB_PATH export, debug symbol support)
  2. Copy 7 proto shaders + types.h, extend types.h with new param structs
  3. Create 5 stub .metal files with valid kernel signatures that compile
  4. Verify all shaders compile to single metallib
- **Files**: `crates/metal-attention-kernels/build.rs`, `crates/metal-attention-kernels/shaders/*.metal`, `crates/metal-attention-kernels/shaders/types.h`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build -p metal-attention-kernels 2>&1 | grep "Built shaders.metallib"`
- **Commit**: `feat(kernels): port Metal build system with 12 shaders compiling to single metallib`
- _Requirements: R3 (flash), R2 (linear), R15 (paged), R16 (RoPE/GQA)_
- _Design: Shader Organization, build.rs_

### Task 1.4: Port GPU device, PsoCache, buffer helpers from proto
- [x] Port core GPU infrastructure to kernels crate
  - `src/device.rs` - GpuDevice singleton (from proto/src/device.rs), adapted for workspace metallib search
  - `src/pipeline.rs` - PsoCache with HashMap<PsoKey, MTLComputePipelineState> (from proto/src/pipeline.rs), add `prewarm()` stub
  - `src/buffer.rs` - BufferPool with acquire/release + alloc_buffer_with_data + create_weight_buffer (zero-copy mmap support), from proto/src/encode.rs
  - `src/command.rs` - CommandManager with triple buffering via dispatch_semaphore(3)
  - `src/types.rs` - AttentionParams, LayerParams, SSMParams repr(C) structs matching types.h
  - `src/dispatch.rs` - helper for encoding compute commands (set_buffer, set_bytes, dispatch_threadgroups)
  - `src/lib.rs` - re-export all public types
- **Do**:
  1. Port device.rs adapting metallib search for workspace layout
  2. Port pipeline.rs (PsoCache) with prewarm() method stub
  3. Port encode.rs into buffer.rs (BufferPool) + dispatch.rs
  4. Create CommandManager with triple buffering
  5. Port types.rs adding LayerParams and SSMParams
  6. Add tests: device init, params layout (64 bytes), buffer alloc/release
- **Files**: `crates/metal-attention-kernels/src/{lib,device,pipeline,buffer,command,dispatch,types}.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test -p metal-attention-kernels`
- **Commit**: `feat(kernels): port GpuDevice, PsoCache, BufferPool, CommandManager from proto`
- _Requirements: R4 (PsoCache), R9 (shader validation)_
- _Design: PsoCache Architecture, Buffer Management, Triple Buffering_

### Task 1.5: [VERIFY] Quality checkpoint
- [ ] Ensure workspace compiles clean, all tests pass
- **Do**: Run clippy, check types, run all tests
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clippy --workspace -- -D warnings && cargo test --workspace`
- **Done when**: Zero warnings, all tests pass
- **Commit**: `chore(workspace): pass quality checkpoint` (if fixes needed)

### Task 1.6: Wire flash attention and linear attention dispatch
- [x] Create kernel dispatch modules that call Metal compute
  - `src/flash.rs` - flash attention dispatch: load PSO, bind Q/K/V/output buffers + AttentionParams, dispatch threadgroups
  - `src/linear.rs` - FLA chunk_h/chunk_o dispatch with GPU prefix sum (eliminate CPU bottleneck)
  - Port CPU reference implementations from proto (cpu_attention_f64, cpu_linear_attention_f64) into `tests/` directory
  - Add assert_allclose test helper
  - Wire up integration tests: GPU flash output vs FP64 CPU reference (atol=5e-3), GPU linear output vs FP64 CPU reference (atol=1e-3)
- **Do**:
  1. Implement flash attention dispatch in kernels/src/flash.rs
  2. Implement linear attention dispatch (chunk_h + chunk_o) in kernels/src/linear.rs
  3. Port cpu_attention_f64 and cpu_linear_attention_f64 from proto to test utils
  4. Create `tests/correctness.rs` with GPU vs CPU reference tests
  5. Run with MTL_SHADER_VALIDATION=1
- **Files**: `crates/metal-attention-kernels/src/{flash,linear}.rs`, `tests/correctness.rs`, `tests/test_utils.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test --test correctness -- --test-threads=1`
- **Commit**: `feat(kernels): wire flash + linear attention dispatch with GPU correctness tests`
- _Requirements: R2 (linear attention), R3 (flash attention), R9 (correctness validation)_
- _Design: Kernel Dispatch Patterns, Correctness Tolerances_

### Task 1.7: Implement RMSNorm, FFN, embedding, matmul kernels
- [x] Implement the supporting kernels needed for a full inference pass
  - `shaders/rmsnorm.metal` - RMS normalization with HIDDEN_SIZE function constant
  - `shaders/ffn.metal` - SwiGLU/ReLU^2/GeGLU with FFN_TYPE function constant
  - `shaders/embedding.metal` - token ID to hidden vector lookup
  - `shaders/matmul.metal` - tiled matrix multiply with simdgroup_matrix (for projections)
  - `src/norm.rs`, `src/ffn.rs`, `src/embed.rs`, `src/matmul.rs` - dispatch code
  - CPU reference for each, basic correctness test
- **Do**:
  1. Replace stub shaders with real implementations
  2. Create dispatch modules for each kernel
  3. Add CPU references and GPU correctness tests
  4. All tests pass with MTL_SHADER_VALIDATION=1
- **Files**: `crates/metal-attention-kernels/shaders/{rmsnorm,ffn,embedding,matmul}.metal`, `crates/metal-attention-kernels/src/{norm,ffn,embed,matmul}.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention-kernels -- --test-threads=1`
- **Commit**: `feat(kernels): implement RMSNorm, FFN, embedding, matmul Metal kernels`
- _Requirements: R6 (inference pipeline needs these)_
- _Design: Shader Organization, Kernel Dispatch Patterns_

### Task 1.8: [VERIFY] Phase 1 POC checkpoint
- [ ] All foundation components working end-to-end
- **Do**:
  1. Run full test suite with shader validation
  2. Verify flash attention matches CPU reference within 5e-3
  3. Verify linear attention matches CPU reference within 1e-3
  4. Verify all kernels compile and dispatch correctly
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test --workspace -- --test-threads=1 && cargo clippy --workspace -- -D warnings`
- **Done when**: All GPU kernels pass correctness tests against CPU references
- **Commit**: `feat(foundation): complete Phase 1 - all kernels validated`

---

## Phase 2: Core (GGUF + Models + Inference)

Focus: Load a real GGUF model, run inference end-to-end.

### Task 2.1: Implement GGUF parser with mmap
- [x] Binary GGUF parser in gguf crate
  - `src/parser.rs` - GgufFile::open() with mmap, header parse, metadata KV parse, tensor info parse
  - `src/metadata.rs` - typed metadata accessors (get_string, get_u32, get_f32, get_array)
  - `src/tensor.rs` - GgufTensorInfo, tensor data byte slice accessor
  - `src/quantize.rs` - GgufType enum (Q4_0, Q4_K_M, Q8_0, F16, F32), block size table
  - `src/detect.rs` - detect_architecture() from metadata + tensor name patterns
  - `src/architectures.rs` - WeightRole enum, map_tensor_name() for llama/rwkv/jamba
  - Tests: parse a small GGUF file header, read metadata, list tensors
- **Do**:
  1. Implement binary parser following GGUF spec (magic, version, tensor_count, metadata_kv_count)
  2. Parse all GGUF metadata value types (uint8..uint64, float32/64, string, arrays)
  3. Parse tensor info array (name, ndim, shape, type, offset)
  4. Implement architecture detection for llama, rwkv, jamba, griffin
  5. Download a small test GGUF (TinyLlama Q4_0 ~600MB) for integration tests
  6. Test: parse header, read metadata, iterate tensors, detect architecture
- **Files**: `crates/metal-attention-gguf/src/{lib,parser,metadata,tensor,quantize,detect,architectures}.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test -p metal-attention-gguf`
- **Commit**: `feat(gguf): implement mmap-based GGUF parser with architecture detection`
- _Requirements: R5 (GGUF model loading)_
- _Design: GGUF Parser Design, Architecture Detection, Weight Mapping_

### Task 2.2: Implement GGUF tokenizer + dequantization kernels
- [x] Tokenizer from GGUF metadata + Q4/Q8 dequant on GPU
  - `src/tokenizer.rs` - BPE tokenizer from GGUF `tokenizer.ggml.*` metadata fields (encode/decode)
  - `shaders/dequantize.metal` - real implementations for Q4_0, Q4_K_M, Q8_0 block dequantization
  - `crates/metal-attention-kernels/src/dequant.rs` - dispatch code for dequant kernels
  - Tests: tokenize/detokenize round-trip, dequant Q4_0 block matches CPU reference
- **Do**:
  1. Parse `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, `tokenizer.ggml.model` from GGUF
  2. Implement BPE encode (text -> token IDs) and decode (token IDs -> text)
  3. Replace dequantize.metal stub with real Q4_0/Q4_K_M/Q8_0 implementations
  4. Create CPU reference dequant, test GPU matches
- **Files**: `crates/metal-attention-gguf/src/tokenizer.rs`, `crates/metal-attention-kernels/shaders/dequantize.metal`, `crates/metal-attention-kernels/src/dequant.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test -p metal-attention-gguf -- tokenizer && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention-kernels -- dequant --test-threads=1`
- **Commit**: `feat(gguf): add BPE tokenizer + Q4/Q8 dequantization kernels`
- _Requirements: R5 (tokenizer from GGUF), R17 (quantization)_
- _Design: Tokenizer Strategy, Dequantization_

### Task 2.3: [VERIFY] Quality checkpoint
- [ ] All crates compile clean, tests pass
- **Do**: Run clippy, tests across workspace
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clippy --workspace -- -D warnings && cargo test --workspace -- --test-threads=1`
- **Done when**: Zero warnings, all tests pass
- **Commit**: `chore(core): pass quality checkpoint` (if fixes needed)

### Task 2.4: RWKV-7 model implementation (first end-to-end model)
- [x] Pure linear model validates LinearSequenceModel trait end-to-end
  - `crates/metal-attention-models/src/rwkv7.rs` - RWKV7Block implementing LinearSequenceModel
    - Dynamic state evolution with vector-valued gating
    - Token-shift mechanism
    - WKV operator dispatch to `rwkv_wkv.metal` kernel (new shader)
  - `crates/metal-attention-kernels/shaders/rwkv_wkv.metal` - RWKV-7 WKV attention operator
  - `crates/metal-attention-kernels/src/rwkv.rs` - WKV dispatch code
  - Wire up weight loading from GGUF tensor names (rwkv.* pattern)
  - Test: load RWKV-7 GGUF, verify forward pass produces non-NaN logits
- **Do**:
  1. Implement RWKV-7 WKV kernel in MSL (state update + output computation)
  2. Implement RWKV7Block struct implementing LinearSequenceModel trait
  3. Map RWKV GGUF tensor names to WeightRole
  4. Test single-layer forward pass with random data
  5. Download small RWKV-7 GGUF for integration test
- **Files**: `crates/metal-attention-models/src/rwkv7.rs`, `crates/metal-attention-kernels/shaders/rwkv_wkv.metal`, `crates/metal-attention-kernels/src/rwkv.rs`, `crates/metal-attention-models/src/{lib,registry}.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention-models -- rwkv --test-threads=1`
- **Commit**: `feat(models): implement RWKV-7 with LinearSequenceModel trait`
- _Requirements: R7 (RWKV-7 end-to-end), R2 (validates linear attention)_
- _Design: Target Model Architectures (RWKV-7)_

### Task 2.5: Inference pipeline + sampling engine
- [x] Build the full prefill/decode loop and token sampling
  - `crates/metal-attention/src/inference.rs` - prefill() and decode() methods on HybridModel
  - `crates/metal-attention/src/model.rs` - HybridModel<L, A> struct with LayerSchedule-driven dispatch
  - `crates/metal-attention/src/sampling.rs` - temperature, top-p, top-k, repetition penalty sampling
  - `crates/metal-attention/src/config.rs` - runtime config (model path, sampling params)
  - Wire: GGUF load -> architecture detect -> model construct -> prefill prompt -> decode loop -> sample tokens
  - Test: load RWKV-7, prefill "Hello", decode 10 tokens, verify valid vocab IDs
- **Do**:
  1. Implement HybridModel with schedule-driven layer dispatch
  2. Implement prefill (parallel prompt processing) and decode (single-token autoregressive)
  3. Implement sampling engine (greedy, temperature, top-p, top-k, repetition penalty)
  4. Integration test: full text generation with RWKV-7
- **Files**: `crates/metal-attention/src/{lib,model,inference,sampling,config}.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test -p metal-attention -- inference --test-threads=1`
- **Commit**: `feat(engine): implement prefill/decode inference loop with sampling`
- _Requirements: R6 (token generation), R7 (RWKV-7 end-to-end)_
- _Design: Inference Pipeline, Sampling Engine_

### Task 2.6: CLI `run` command with streaming output
- [x] Wire CLI binary for text generation
  - `src/main.rs` - clap-based CLI with `run` subcommand
  - Flags: `-m <model.gguf>`, `-p "prompt"`, `-n 256` (max tokens), `-s 42` (seed), `--temp 0.8`, `--top-p 0.9`, `--top-k 40`
  - Streaming: tokens to stdout, stats (tok/s, timing) to stderr
  - Error handling: user-friendly messages for missing model, invalid format, GPU errors
- **Do**:
  1. Implement clap CLI with run subcommand + all flags per design
  2. Wire run command: parse args -> load model -> prefill -> decode loop -> stream tokens
  3. Print stats to stderr on completion (tokens generated, tok/s, prefill time)
  4. Test: build binary, run with --help, verify flag parsing
- **Files**: `src/main.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && ./target/debug/metal-attention run --help`
- **Commit**: `feat(cli): implement run command with streaming token output`
- _Requirements: R8 (CLI run command)_
- _Design: CLI Design_

### Task 2.7: [VERIFY] Phase 2 POC - generate text from GGUF model
- [ ] End-to-end text generation works
- **Do**:
  1. Download a small RWKV-7 GGUF model
  2. Run: `./target/debug/metal-attention run -m <model.gguf> -p "The meaning of life is" -n 20 --temp 0.0`
  3. Verify: produces non-empty text output, valid UTF-8, no panics
  4. Run full test suite
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && MTL_SHADER_VALIDATION=1 cargo test --workspace -- --test-threads=1`
- **Done when**: CLI generates coherent text from a real GGUF model
- **Commit**: `feat(core): complete Phase 2 - end-to-end text generation from GGUF`

---

## Phase 3: Model (Flash Attention + HybridModel + More Architectures)

Focus: Flash attention path working, hybrid dispatch, additional models.

### Task 3.1: Flash attention as SoftmaxAttention with KV cache
- [ ] Implement FlashAttention struct with SoftmaxAttention trait + dense KV cache
  - `crates/metal-attention-models/src/flash_attn.rs` - FlashAttention implementing SoftmaxAttention trait
  - `crates/metal-attention-kernels/src/kv_cache.rs` - DenseKVCache (K/V buffers, append, read)
  - Wire: prefill_attention (populate KV cache), decode_attention (single-query against full cache)
  - Add RoPE application in attention forward path via `src/rope.rs` dispatch
  - Add GQA remap via `src/gqa.rs` dispatch
  - Test: FlashAttention forward matches CPU reference, KV cache grows correctly
- **Do**:
  1. Implement DenseKVCache with append/read operations
  2. Implement FlashAttention struct with prefill (full Q*K*V) and decode (single-query) paths
  3. Wire RoPE kernel dispatch for position encoding
  4. Wire GQA remap for grouped-query attention
  5. Integration test: prefill 64 tokens, decode 10 tokens, verify no NaN
- **Files**: `crates/metal-attention-models/src/flash_attn.rs`, `crates/metal-attention-kernels/src/{kv_cache,rope,gqa}.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test -- flash_attn --test-threads=1`
- **Commit**: `feat(models): implement FlashAttention with SoftmaxAttention trait + DenseKVCache`
- _Requirements: R3 (flash attention), R16 (RoPE/GQA)_
- _Design: Flash Attention, KV Cache Management_

### Task 3.2: Llama/Mistral model (pure transformer baseline)
- [x] Pure-transformer model validates the full SoftmaxAttention path
  - `crates/metal-attention-models/src/llama.rs` - LlamaModel using HybridModel with pure_transformer schedule
  - Wire GGUF loading for llama tensor name pattern (`blk.N.attn_q/k/v/output`, `blk.N.ffn_gate/up/down`)
  - SwiGLU FFN, RoPE, GQA
  - Test: load TinyLlama Q4_0, generate text, verify coherent output
- **Do**:
  1. Implement Llama weight mapping from GGUF tensor names
  2. Wire HybridModel<FLALinear, FlashAttention> with pure_transformer schedule
  3. Run inference on TinyLlama
  4. Verify greedy decode produces non-garbage output
- **Files**: `crates/metal-attention-models/src/llama.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && ./target/debug/metal-attention run -m fixtures/models/tinyllama-q4_0.gguf -p "Hello world" -n 10 --temp 0.0 2>/dev/null | head -1`
- **Commit**: `feat(models): implement Llama/Mistral pure transformer model`
- _Requirements: R14 (Llama/Mistral baseline)_
- _Design: Data Flow, Inference Pipeline_

### Task 3.3: [VERIFY] Quality checkpoint
- [ ] Both RWKV-7 and Llama generate text, all tests pass
- **Do**: Run full test suite, verify both model paths work
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clippy --workspace -- -D warnings && MTL_SHADER_VALIDATION=1 cargo test --workspace -- --test-threads=1`
- **Done when**: Both linear (RWKV) and transformer (Llama) paths validated
- **Commit**: `chore(models): pass quality checkpoint` (if fixes needed)

### Task 3.4: HybridModel dispatch + Jamba model
- [x] Implement hybrid layer scheduling + Jamba (7:1 Mamba:Attention + MoE)
  - Wire HybridModel schedule-driven dispatch: linear layers use LinearSequenceModel, attention layers use SoftmaxAttention
  - `crates/metal-attention-models/src/jamba.rs` - Jamba model
    - 7:1 ratio via LayerSchedule::periodic(72, 7)
    - Mamba-2 blocks as LinearSequenceModel (needs ssm_scan.metal kernel)
    - FlashAttention blocks with GQA for attention layers
    - MoE routing (16 experts, top-2 selection) as specialized FFN
  - `crates/metal-attention-kernels/shaders/ssm_scan.metal` - Mamba selective scan kernel
  - `crates/metal-attention-kernels/src/ssm.rs` - SSM dispatch
  - Test: load Jamba GGUF, verify hybrid dispatch alternates layer types correctly
- **Do**:
  1. Implement Mamba selective scan MSL kernel (state update: h_t = A_t * h_{t-1} + B_t * x_t)
  2. Implement MambaBlock as LinearSequenceModel
  3. Implement MoE router (expert selection + weighted combination)
  4. Implement Jamba model with 7:1 schedule
  5. Test hybrid dispatch with both layer types executing
- **Files**: `crates/metal-attention-models/src/jamba.rs`, `crates/metal-attention-kernels/shaders/ssm_scan.metal`, `crates/metal-attention-kernels/src/ssm.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test -- jamba --test-threads=1`
- **Commit**: `feat(models): implement Jamba hybrid model with 7:1 Mamba:Attention + MoE`
- _Requirements: R10 (HybridModel dispatch), R11 (Jamba model)_
- _Design: Hybrid Model Composition, Mamba SSM_

### Task 3.5: PagedAttention V2 + prefix sum kernel
- [x] Implement paged KV cache and GPU prefix sum
  - `crates/metal-attention-kernels/src/paged.rs` - PagedAttention dispatch (partition + reduce)
  - `crates/metal-attention-kernels/src/kv_cache.rs` - extend with PagedKVCache (block table, page pool, free list)
  - `crates/metal-attention-kernels/shaders/prefix_sum.metal` - replace stub with GPU parallel prefix sum over D*D matrices
  - `crates/metal-attention-kernels/src/prefix_sum.rs` - prefix sum dispatch
  - Test: paged attention matches dense attention (atol=1e-3), prefix sum matches CPU
- **Do**:
  1. Implement PagedKVCache with block table indirection
  2. Implement paged attention dispatch (partition scatter + reduce)
  3. Implement GPU prefix sum kernel (eliminates ~300us CPU bottleneck)
  4. Test: paged vs dense produce same output, prefix sum vs CPU reference
- **Files**: `crates/metal-attention-kernels/src/{paged,prefix_sum}.rs`, `crates/metal-attention-kernels/shaders/prefix_sum.metal`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test -- paged --test-threads=1 && cargo test -- prefix_sum --test-threads=1`
- **Commit**: `feat(kernels): implement PagedAttention V2 + GPU prefix sum`
- _Requirements: R15 (PagedAttention), R2 (GPU prefix sum eliminates bottleneck)_
- _Design: Paged Attention, Kernel Fusion Strategy_

### Task 3.6: [VERIFY] Phase 3 checkpoint - hybrid inference validated
- [x] Three model architectures working, hybrid dispatch validated
- **Do**:
  1. Generate text with RWKV-7 (pure linear), Llama (pure transformer), Jamba (hybrid)
  2. All GPU correctness tests pass with shader validation
  3. Run clippy + full test suite
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clippy --workspace -- -D warnings && MTL_SHADER_VALIDATION=1 cargo test --workspace -- --test-threads=1`
- **Done when**: 3 model architectures generate text, hybrid dispatch proven
- **Commit**: `feat(model): complete Phase 3 - hybrid inference validated with 3 architectures`

---

## Phase 4: Polish (Bench + Info + Quality + Burn)

Focus: CLI completeness, performance validation, optional integrations.

### Task 4.1: CLI `bench` and `info` commands
- [x] Add benchmark and model info CLI subcommands
  - `bench` subcommand: `-m <model.gguf>`, `--seq-lengths 256,512,1024`, reports prefill tok/s, decode tok/s, TFLOPS
  - `info` subcommand: `-m <model.gguf>`, prints architecture, layer count, head dims, quantization, vocab size, estimated memory
  - `--json` flag for JSONL machine output on both commands
  - Criterion benchmarks: `benches/attention.rs` (flash + linear throughput), `benches/inference.rs` (prefill + decode tok/s)
- **Do**:
  1. Implement bench subcommand with timing loop and stats reporting
  2. Implement info subcommand parsing GGUF metadata
  3. Add --json output mode (JSONL to stdout)
  4. Create criterion benchmark suite
- **Files**: `src/main.rs`, `benches/{attention,inference}.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo build && ./target/debug/metal-attention info --help && ./target/debug/metal-attention bench --help`
- **Commit**: `feat(cli): implement bench and info commands with JSON output`
- _Requirements: R18 (CLI bench + info), R21 (JSON output)_
- _Design: CLI Design_

### Task 4.2: Griffin + Zamba models (additional architectures)
- [x] Implement remaining hybrid model architectures
  - `crates/metal-attention-models/src/griffin.rs` - Griffin (2:1 RG-LRU:Attention)
    - RG-LRU (Real-Gated Linear Recurrent Unit) as LinearSequenceModel
    - Local sliding-window attention
  - `crates/metal-attention-models/src/zamba.rs` - Zamba (6:1 Mamba:SharedAttention)
    - Shared attention layers with LoRA projectors
    - 6:1 Mamba:Attention ratio
  - Wire GGUF detection for griffin/recurrentgemma and zamba tensor patterns
  - Test: architecture detection from GGUF metadata, weight mapping
- **Do**:
  1. Implement RG-LRU block as LinearSequenceModel for Griffin
  2. Implement shared attention with LoRA for Zamba
  3. Add GGUF detection for both architectures
  4. Unit tests for architecture detection and layer schedule generation
- **Files**: `crates/metal-attention-models/src/{griffin,zamba}.rs`, `crates/metal-attention-gguf/src/detect.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test -p metal-attention-models -- griffin && cargo test -p metal-attention-models -- zamba`
- **Commit**: `feat(models): implement Griffin (2:1) and Zamba (6:1) hybrid architectures`
- _Requirements: R12 (Griffin), R13 (Zamba)_
- _Design: Target Model Architectures_

### Task 4.3: [VERIFY] Quality checkpoint
- [x] All model paths compile and pass tests
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo clippy --workspace -- -D warnings && cargo test --workspace -- --test-threads=1`
- **Done when**: Zero warnings, all tests pass
- **Commit**: `chore(polish): pass quality checkpoint` (if fixes needed)

### Task 4.4: Burn framework integration (optional crate)
- [ ] Implement `metal-attention-burn` crate for Burn interop
  - `src/backend.rs` - MetalAttentionBackend<B> newtype implementing Burn's Backend trait
  - `src/bridge.rs` - Burn tensor <-> Metal buffer bridge (2-17us overhead per proto 8)
  - `src/ops.rs` - Backend trait delegation (forward Burn ops to Metal kernels where beneficial)
  - Feature-gated: only builds when `burn-ext` feature enabled
  - Test: create Burn tensor, bridge to Metal buffer, run attention, bridge back, verify result
- **Do**:
  1. Implement MetalAttentionBackend newtype
  2. Implement zero-copy Burn tensor <-> Metal buffer bridge
  3. Delegate attention ops to Metal kernels
  4. Test: bridge round-trip with correctness check
- **Files**: `crates/metal-attention-burn/src/{lib,backend,bridge,ops}.rs`, `crates/metal-attention-burn/Cargo.toml`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test -p metal-attention-burn --features burn-ext`
- **Commit**: `feat(burn): implement Burn framework bridge with 2-17us overhead`
- _Requirements: R20 (Burn AttentionBackend)_
- _Design: Burn Integration (Proto 8)_

### Task 4.5: Performance optimization pass
- [ ] Multi-simdgroup flash attention + kernel fusion for throughput targets
  - Upgrade flash_attention.metal from 1 simdgroup (0.16 TFLOPS) to 4+ simdgroups (target >1 TFLOPS)
  - Fused QKV projection kernel (single matmul for Q, K, V)
  - Fused RMSNorm + linear projection
  - In-kernel dequantization for Q4/Q8 weights in matmul
  - PsoCache.prewarm() implementation: pre-compile all kernel variants at model load
  - Benchmark: verify >1 TFLOPS flash attention, <500ms first-token latency
- **Do**:
  1. Implement multi-simdgroup tiling in flash_attention.metal
  2. Implement fused QKV projection
  3. Implement PsoCache prewarm at model load
  4. Run criterion benchmarks, measure TFLOPS
  5. Run inference latency benchmark
- **Files**: `crates/metal-attention-kernels/shaders/flash_attention.metal`, `crates/metal-attention-kernels/src/{pipeline,flash,matmul}.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo bench -- flash_attention 2>&1 | grep -i tflops`
- **Commit**: `perf(kernels): multi-simdgroup flash attention targeting >1 TFLOPS`
- _Requirements: R3 (>1 TFLOPS flash attention)_
- _Design: Kernel Fusion Strategy, Performance Architecture_

---

## Phase 5: Testing

Focus: Comprehensive tests, property-based testing, stress tests.

### Task 5.1: Property-based tests with proptest
- [x] Add proptest-based numerical invariant tests
  - Flash attention: all-zero V -> all-zero output, GQA group_size=1 is identity
  - RoPE at position 0 is identity
  - Linear attention: identity K produces V directly
  - All kernel outputs are finite (no NaN/Inf) for random inputs
  - Sampling: temperature=0 is greedy argmax, top_k=1 always selects max
- **Do**:
  1. Add proptest dependency to relevant crates
  2. Implement numerical invariant tests per QA.md specification
  3. Implement symmetry and identity tests
  4. All run on CPU only (no GPU required)
- **Files**: `crates/metal-attention-kernels/tests/property.rs`, `crates/metal-attention/tests/sampling_property.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test -- property`
- **Commit**: `test(kernels): add property-based numerical invariant tests`
- _Requirements: R9 (correctness validation)_
- _Design: Testing Architecture (Property-based)_

### Task 5.2: GPU correctness test sweep
- [x] Comprehensive kernel correctness tests across configs
  - Flash attention: N in {64,128,256,512,1024,2048}, D in {64,128}, heads in {1,4,8,32}
  - Linear attention: N in {64,128,256,512,1024}, D=64
  - PagedAttention: paged vs dense consistency test
  - RoPE: correctness at multiple positions
  - All with MTL_SHADER_VALIDATION=1
  - Add finiteness checks, dimensional correctness assertions
  - Add numerical edge cases: large magnitude, near-zero, uniform, one-hot inputs
- **Do**:
  1. Create parameterized test matrices per QA.md specification
  2. Implement edge case data generators (large magnitude, near-zero, etc.)
  3. Add tolerance table as code constants with consistency test
  4. Run full sweep
- **Files**: `tests/correctness.rs`, `tests/test_utils.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test --test correctness -- --test-threads=1`
- **Commit**: `test(correctness): comprehensive GPU kernel correctness sweep with edge cases`
- _Requirements: R9 (tolerances: Flash 5e-3, Linear 1e-3, RoPE 1e-4, GQA 1e-6)_
- _Design: Correctness Tolerances, Numerical Stability Edge Cases_

### Task 5.3: Integration + E2E tests
- [x] Model loading, inference pipeline, text generation tests
  - `tests/model_load.rs` - GGUF parse, tensor-to-buffer mapping, architecture detection
  - `tests/e2e.rs` - end-to-end text generation (load model, prefill, decode, verify valid tokens)
  - Memory leak test: 100 iterations of inference, verify <1% memory growth
  - Test both RWKV (linear) and Llama (transformer) paths minimum
- **Do**:
  1. Create integration tests for model loading
  2. Create E2E test for text generation
  3. Create memory leak detection test
  4. Verify with real GGUF models
- **Files**: `tests/{model_load,e2e}.rs`
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test --test model_load --test e2e -- --test-threads=1`
- **Commit**: `test(e2e): add integration and end-to-end inference tests`
- _Requirements: R7 (RWKV-7 E2E), R14 (Llama baseline)_
- _Design: Testing Architecture_

### Task 5.4: [VERIFY] Full test suite green
- [x] All tests pass, all kernels correct, no memory leaks
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && MTL_SHADER_VALIDATION=1 cargo test --workspace -- --test-threads=1 && cargo clippy --workspace -- -D warnings && cargo fmt --check`
- **Done when**: 100% test pass rate, zero clippy warnings, formatted
- **Commit**: `test(all): complete Phase 5 - comprehensive test coverage`

---

## Phase 6: Quality Gates + PR

### Task 6.1: Full local CI
- [x] Run complete quality suite
- **Do**:
  1. `cargo fmt --check`
  2. `cargo clippy --workspace -- -D warnings`
  3. `cargo test --workspace -- --test-threads=1` (with MTL_SHADER_VALIDATION=1)
  4. `cargo build --release`
  5. `cargo bench -- --quick` (smoke test benchmarks)
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo fmt --check && cargo clippy --workspace -- -D warnings && MTL_SHADER_VALIDATION=1 cargo test --workspace -- --test-threads=1 && cargo build --release`
- **Done when**: All commands pass with zero errors
- **Commit**: `chore(ci): pass full local CI` (if fixes needed)

### Task 6.2: Create PR
- [x] Push branch and create pull request
- **Do**:
  1. Verify on feature branch: `git branch --show-current`
  2. Push: `git push -u origin <branch>`
  3. Create PR with summary of all phases implemented
- **Verify**: `gh pr checks --watch`
- **Done when**: PR created, CI green
- **Commit**: None

### Task 6.3: [VERIFY] AC checklist
- [x] Verify all acceptance criteria from requirements.md
- **Do**:
  1. R1: Trait hierarchy exists (SequenceBlock, LinearSequenceModel, SoftmaxAttention) - grep crate
  2. R2: Linear attention kernel passes correctness test (atol=1e-3)
  3. R3: Flash attention kernel passes correctness test (atol=5e-3)
  4. R4: PsoCache with function constants, 0% GPU overhead
  5. R5: GGUF parser loads models with mmap
  6. R6: Prefill + decode loop generates tokens
  7. R7: RWKV-7 generates text end-to-end
  8. R8: CLI `run` command works
  9. R9: FP64 CPU references with documented tolerances
- **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/metal-attention && cargo test --workspace -- --test-threads=1 2>&1 | tail -5`
- **Done when**: All P0 requirements verified by passing tests
- **Commit**: None

---

## Notes

**POC shortcuts (Phase 1-2)**:
- Single-simdgroup flash attention initially (0.16 TFLOPS), multi-simdgroup in Phase 4
- CPU prefix sum acceptable until Phase 3 GPU implementation
- Hardcoded tile sizes until prewarm() selects optimal per-device
- No MoE routing until Jamba task
- Skip Griffin/Zamba until Phase 4

**Production TODOs (post-Phase 4)**:
- M1-M4 device compatibility testing (R19)
- Speculative decoding (R23)
- Metal 4 cooperative tensor migration (R22)
- Config file support (`~/Library/Application Support/metal-attention/config.toml`)
- CI pipeline with GPU runner (self-hosted Mac mini)
- Perplexity evaluation against reference engines

**Key risk areas**:
- GGUF parser correctness: format is complex with many type variants
- Multi-simdgroup flash attention: jumping from 0.16 to >1 TFLOPS requires careful tiling
- RWKV-7 weight mapping: GGUF format for RWKV is less standardized than llama
- Mamba SSM kernel: selective scan is non-trivial to implement efficiently in MSL
