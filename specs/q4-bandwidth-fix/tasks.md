---
spec: q4-bandwidth-fix
phase: tasks
total_tasks: 14
created: 2026-02-15
generated: auto
---

# Tasks: q4-bandwidth-fix

## Phase 1: Make It Work (POC)

Focus: Get the coalesced v5 kernel producing correct results for one dimension pair (576x576). Skip fused kernel, skip multi-row -- just validate the aligned read approach works.

- [ ] 1.1 Create matvec_q4_0_v5_coalesced.metal with aligned uint4 reads
  - **Do**: Create new shader file. Cast weight buffer to `device const uchar*` for byte-level access. For each group of 7 blocks, compute byte offset = group * 7 * 18, then load data as aligned uint4 reads from `device const uint4*` cast at the row's base address. Extract scale (2 bytes fp16) and nibbles (16 bytes) for each of the 7 blocks. Start simple: 32 threads, 1 simdgroup, 1 row per threadgroup (same dispatch as baseline). Use float accumulation first. Handle remainder blocks (n_blocks % 7) with a scalar fallback loop.
  - **Files**: `crates/metal-attention-kernels/shaders/matvec_q4_0_v5_coalesced.metal`
  - **Done when**: Shader compiles without errors
  - **Verify**: `cargo build -p metal-attention-kernels 2>&1 | grep -E "error|warning"`
  - **Commit**: `feat(metal): add v5 coalesced Q4_0 matvec kernel`
  - _Requirements: FR-1, FR-2, FR-8_
  - _Design: Component A_

- [ ] 1.2 Add Rust dispatch and unit test for v5 kernel (576x576)
  - **Do**: In `matvec_q4_0.rs`, add `dispatch_matvec_q4_0_v5()` function following the existing `dispatch_matvec_q4_0()` pattern. Use `PsoKey::simple("matvec_q4_0_v5_coalesced")`. Dispatch with 32 threads per threadgroup, out_dim threadgroups (same as baseline for now). Add `test_matvec_q4_0_v5_576x576` test comparing v5 output against CPU reference `cpu_q4_0_dot()`. Tolerance: 1e-2.
  - **Files**: `crates/metal-attention-kernels/src/matvec_q4_0.rs`
  - **Done when**: `test_matvec_q4_0_v5_576x576` passes
  - **Verify**: `cargo test -p metal-attention-kernels test_matvec_q4_0_v5_576x576 -- --nocapture`
  - **Commit**: `test(metal): add v5 coalesced kernel unit test for 576x576`
  - _Requirements: FR-5, AC-1.4_
  - _Design: Component C_

- [ ] 1.3 Add unit tests for all SmolLM dimension pairs
  - **Do**: Add tests for: 192x576 (K/V), 1536x576 (gate/up), 576x1536 (down), 49152x576 (lm_head), 2x32 (basic), zero-scale. Follow existing test patterns in matvec_q4_0.rs. These validate the remainder-block handling for different n_blocks_per_row values.
  - **Files**: `crates/metal-attention-kernels/src/matvec_q4_0.rs`
  - **Done when**: All v5 unit tests pass
  - **Verify**: `cargo test -p metal-attention-kernels test_matvec_q4_0_v5 -- --nocapture`
  - **Commit**: `test(metal): v5 coalesced kernel tests for all SmolLM dimensions`
  - _Requirements: AC-1.4_
  - _Design: Component C_

- [ ] 1.4 POC Checkpoint -- v5 kernel correct for all dimensions
  - **Do**: Run the full test suite to confirm no regressions. Existing baseline tests must still pass. All v5 tests must pass.
  - **Done when**: All matvec_q4_0 tests pass (baseline + v5)
  - **Verify**: `cargo test -p metal-attention-kernels -- --nocapture 2>&1 | tail -5`
  - **Commit**: `feat(metal): POC complete -- v5 coalesced reads correct for all dims`

## Phase 2: Optimize + Integrate

After correctness validated, add multi-row and wire into forward pass.

- [ ] 2.1 Add multi-row support (8 rows per threadgroup, 256 threads)
  - **Do**: Update `matvec_q4_0_v5_coalesced.metal` to process 8 output rows per threadgroup. 256 threads = 8 simdgroups, each simdgroup handles 1 row. Add `threadgroup float partial_sums[8]` for cross-simdgroup reduction (though with 1 simdgroup per row, this is just simd_sum + write). Add `uint simd_id [[simdgroup_index_in_threadgroup]]` and `uint simd_lane [[thread_index_in_simdgroup]]` parameters. Guard: `if (row >= out_dim) return` for last threadgroup.
  - **Files**: `crates/metal-attention-kernels/shaders/matvec_q4_0_v5_coalesced.metal`
  - **Done when**: Tests still pass with multi-row dispatch
  - **Verify**: `cargo test -p metal-attention-kernels test_matvec_q4_0_v5 -- --nocapture`
  - **Commit**: `perf(metal): v5 multi-row threadgroups (8 rows, 256 threads)`
  - _Requirements: FR-3, FR-4_
  - _Design: Component A_

- [ ] 2.2 Update Rust dispatch for multi-row threadgroup geometry
  - **Do**: In `dispatch_matvec_q4_0_v5()`, change threadgroup size to 256 and grid to `ceil(out_dim / 8)`. Pass ROWS_PER_TG=8 as a constant. Update v5 unit tests to use the new dispatch dimensions.
  - **Files**: `crates/metal-attention-kernels/src/matvec_q4_0.rs`
  - **Done when**: All v5 unit tests pass with multi-row dispatch
  - **Verify**: `cargo test -p metal-attention-kernels test_matvec_q4_0_v5 -- --nocapture`
  - **Commit**: `refactor(metal): v5 Rust dispatch with 256-thread multi-row geometry`
  - _Requirements: FR-5, AC-2.1, AC-2.2, AC-2.3_
  - _Design: Component C_

- [ ] 2.3 Update fused rmsnorm_matvec_q4_0 kernel with coalesced reads
  - **Do**: In `rmsnorm_matvec_q4_0.metal`, replace the BlockQ4_0 struct read loop with the same uint4 aligned read pattern from v5. Keep the RMS phase 1 unchanged (cooperative simd_sum). Update phase 2 to use aligned reads. Since fused kernel is 1 row per threadgroup (RMS must be computed once per row), keep 32 threads/TG for now (multi-row would require each simdgroup to independently compute RMS, which is fine but keep it simple for this task).
  - **Files**: `crates/metal-attention-kernels/shaders/rmsnorm_matvec_q4_0.metal`
  - **Done when**: Fused kernel tests pass
  - **Verify**: `cargo test -p metal-attention fused_kernel -- --nocapture`
  - **Commit**: `perf(metal): fused rmsnorm_matvec_q4_0 with coalesced reads`
  - _Requirements: FR-7, AC-4.1, AC-4.2_
  - _Design: Component B_

- [ ] 2.4 Wire v5 kernel into gpu_forward_pass.rs
  - **Do**: In `encode_matvec_q4_0()`, change PSO key from `"matvec_q4_0"` to `"matvec_q4_0_v5_coalesced"`. Update threadgroup size to 256 and grid to `ceil(out_dim / 8)`. In `from_gguf()`, replace `PsoKey::simple("matvec_q4_0")` with `PsoKey::simple("matvec_q4_0_v5_coalesced")` in the prewarm list. Keep the old kernel shader file for reference but it will no longer be dispatched.
  - **Files**: `crates/metal-attention/src/gpu_forward_pass.rs`
  - **Done when**: Forward pass uses v5 kernel, builds successfully
  - **Verify**: `cargo build -p metal-attention 2>&1 | grep -E "error"`
  - **Commit**: `perf(metal): wire v5 coalesced kernel into forward pass`
  - _Requirements: FR-6, FR-9, AC-3.1_
  - _Design: Component D_

- [ ] 2.5 Add half precision accumulation
  - **Do**: In the v5 kernel, change the inner loop accumulator from `float` to `half`. Use `half(int(...) - 8)` for nibble dequantization. Keep the simd_sum reduction in float (convert just before reduction) to avoid overflow. Update the fused kernel similarly. Run correctness tests to verify tolerance is still within 1e-2.
  - **Files**: `crates/metal-attention-kernels/shaders/matvec_q4_0_v5_coalesced.metal`, `crates/metal-attention-kernels/shaders/rmsnorm_matvec_q4_0.metal`
  - **Done when**: All tests pass with half precision accumulation
  - **Verify**: `cargo test -p metal-attention-kernels test_matvec_q4_0_v5 -- --nocapture`
  - **Commit**: `perf(metal): half precision accumulation in v5 and fused kernels`
  - _Requirements: NFR-3_
  - _Design: Component A_

## Phase 3: Testing

- [ ] 3.1 Run GPU correctness integration test
  - **Do**: Run the full gpu_correctness test that compares GPU vs CPU forward pass with real SmolLM-135M weights. This validates end-to-end correctness through all 30 layers.
  - **Files**: `crates/metal-attention/tests/gpu_correctness.rs`
  - **Done when**: GPU vs CPU logits match within tolerance, generated text is identical
  - **Verify**: `cargo test -p metal-attention --test gpu_correctness -- --ignored --nocapture`
  - **Commit**: `test(metal): verify v5 kernel GPU vs CPU correctness` (if any fixes needed)
  - _Requirements: AC-3.2_

- [ ] 3.2 Run fused kernel numerical test
  - **Do**: Run the fused_kernel_test that compares fused rmsnorm+matvec against separate rmsnorm + matvec for all dimension pairs.
  - **Files**: `crates/metal-attention/tests/fused_kernel_test.rs`
  - **Done when**: All fused kernel dimension pairs pass within 1e-4
  - **Verify**: `cargo test -p metal-attention --test fused_kernel_test -- --nocapture`
  - **Commit**: `test(metal): verify fused kernel numerical accuracy with coalesced reads` (if any fixes needed)
  - _Requirements: AC-4.3_

- [ ] 3.3 Benchmark decode throughput
  - **Do**: Run the inference benchmark to measure tok/s. Capture results. Compare against BEFORE baseline of ~240 tok/s. Target: >500 tok/s.
  - **Files**: `benches/inference.rs`
  - **Done when**: Benchmark shows >500 tok/s (>2x baseline)
  - **Verify**: `cargo bench --bench inference 2>&1 | grep -E "thrpt|time"`
  - **Commit**: `perf(metal): benchmark v5 coalesced kernel -- Xtok/s (was 240)`
  - _Requirements: AC-3.3, NFR-2_

## Phase 4: Quality Gates

- [ ] 4.1 Local quality check
  - **Do**: Run full test suite (unit + integration), type check, and build. Ensure no regressions across the entire metal-attention workspace.
  - **Verify**: `cargo test --workspace 2>&1 | tail -10 && cargo clippy --workspace 2>&1 | grep -E "error|warning"`
  - **Done when**: All tests pass, no clippy errors
  - **Commit**: `fix(metal): address clippy/lint issues` (if needed)

- [ ] 4.2 Create PR with benchmark data
  - **Do**: Push branch, create PR with benchmark comparison (BEFORE vs AFTER tok/s). Include bandwidth measurement if available.
  - **Verify**: `gh pr checks --watch` all green
  - **Done when**: PR ready for review
  - **Commit**: None (PR creation only)

## Notes

- **POC shortcuts**: Phase 1 uses 32 threads/TG (same as baseline) to isolate the aligned-read improvement from multi-row. Multi-row added in Phase 2.
- **Key risk**: If byte extraction from uint4 is expensive, the aligned reads may not help. The POC will reveal this before investing in multi-row + fused kernel changes.
- **Remainder blocks**: For n_blocks=18 (most projections), 18 = 2*7 + 4. Two groups of 7 blocks via aligned reads, then 4 blocks via scalar fallback. For n_blocks=48 (down projection), 48 = 6*7 + 6.
- **BEFORE baseline**: GPU ~240 tok/s, CPU ~16 tok/s (from bench run 2026-02-15).
