---
spec: gpu-centric-transition
phase: tasks
total_tasks: 35
created: 2026-02-10
---
# Tasks: GPU-Centric Particle System Transition

## Phase 1: Make It Work (POC) — Eliminate CPU Readback + Indirect Emission Dispatch

Focus: Prove indirect dispatch works end-to-end. CPU no longer reads dead_count or computes emission threadgroups. Accept hardcoded values, skip tests.

- [x] 1.1 Add DispatchArgs and GpuEmissionParams structs to types.rs and types.h
  - **Do**:
    1. In `particle-system/src/types.rs`, add `DispatchArgs` struct: `#[repr(C)] pub struct DispatchArgs { pub threadgroups_per_grid: [u32; 3] }` with Default impl `[0,1,1]`
    2. Add `GpuEmissionParams` struct: `#[repr(C)] pub struct GpuEmissionParams { pub emission_count: u32, pub actual_burst_count: u32, pub _pad0: u32, pub _pad1: u32 }` with Default
    3. Add layout tests: `size_of::<DispatchArgs>() == 12`, `align_of == 4`, field offset checks; `size_of::<GpuEmissionParams>() == 16`
    4. In `particle-system/shaders/types.h`, add corresponding MSL structs `DispatchArgs` (3x uint) and `GpuEmissionParams` (4x uint) after DrawArgs
  - **Files**: `particle-system/src/types.rs`, `particle-system/shaders/types.h`
  - **Done when**: Layout tests pass, both Rust and MSL structs defined with matching layouts
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test -- test_dispatch_args_layout test_gpu_emission_params_layout`
  - **Commit**: `feat(types): add DispatchArgs and GpuEmissionParams structs for indirect dispatch`
  - _Requirements: FR-3, FR-4, NFR-6_
  - _Design: Section 4.2, 3.2_

- [x] 1.2 Rename emission_count to base_emission_rate in Uniforms
  - **Do**:
    1. In `types.rs`: rename field `emission_count` to `base_emission_rate` in Uniforms struct + Default impl
    2. In `types.h`: rename `uint emission_count` to `uint base_emission_rate` at offset 200
    3. In `emission.metal`: change `uniforms.emission_count` to `uniforms.base_emission_rate` (line 48). Note: this reference will be removed entirely in task 1.5 but rename now for compilation
    4. In `main.rs`: change `(*uniforms_ptr).emission_count = emission_count` to `(*uniforms_ptr).base_emission_rate = base_emission`
    5. In `tests/gpu_integration.rs`: rename all `emission_count` field references in Uniforms struct and usages to `base_emission_rate`
    6. Verify Uniforms is still 256 bytes (existing test `test_uniforms_size_256`)
  - **Files**: `particle-system/src/types.rs`, `particle-system/shaders/types.h`, `particle-system/shaders/emission.metal`, `particle-system/src/main.rs`, `particle-system/tests/gpu_integration.rs`
  - **Done when**: All files compile, Uniforms size unchanged at 256 bytes, all existing tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test`
  - **Commit**: `refactor(uniforms): rename emission_count to base_emission_rate`
  - _Requirements: FR-2_
  - _Design: Section 12.1_

- [x] 1.3 Create prepare_dispatch.metal kernel
  - **Do**:
    1. Create `particle-system/shaders/prepare_dispatch.metal` with `#include "types.h"`
    2. Implement `kernel void prepare_dispatch(...)` per design Section 3.2:
       - buffer(0) = dead_list (read count), buffer(1) = uniforms (read base_emission_rate, burst_count, pool_size), buffer(2) = write_list (reset counter to 0), buffer(3) = emission_dispatch_args (write DispatchArgs), buffer(4) = gpu_emission_params (write GpuEmissionParams)
    3. Single-thread kernel with `if (tid != 0) return` guard
    4. Logic: `dead_count = dead_list[0]`, `emission_count = min(base_emission_rate + burst_count, dead_count)`, `actual_burst = min(burst_count, emission_count)`, compute threadgroups = `ceil(emission_count / 256)` clamped to `pool_size / 256`
    5. Write to emission_params and emission_args structs, reset `write_list[0] = 0`
    6. build.rs auto-detects new .metal file (no change needed)
  - **Files**: `particle-system/shaders/prepare_dispatch.metal`
  - **Done when**: `cargo build` compiles without Metal shader errors
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | grep -E "(error|warning.*shaders)" | head -5; echo "exit: $?"`
  - **Commit**: `feat(shaders): create prepare_dispatch kernel for GPU-side emission computation`
  - _Requirements: FR-1_
  - _Design: Section 3.1, 3.2, KB 134, KB 277_

- [x] 1.4 [VERIFY] Quality checkpoint: cargo build && cargo test
  - **Do**: Run full build and test suite to catch any issues from tasks 1.1-1.3
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 && cargo test 2>&1 | tail -5`
  - **Done when**: Build succeeds, all 65 existing tests pass
  - **Commit**: `chore(gpu-centric): pass quality checkpoint` (only if fixes needed)

- [ ] 1.5 Modify emission.metal to read from GpuEmissionParams
  - **Do**:
    1. In `emission.metal`: add `device const GpuEmissionParams* emission_params [[buffer(8)]]` parameter to `emission_kernel`
    2. Change guard from `if (tid >= uniforms.base_emission_rate) return` to `if (tid >= emission_params->emission_count) return`
    3. Change burst detection from `bool is_burst = (tid < uniforms.burst_count)` to `bool is_burst = (tid < emission_params->actual_burst_count)`
  - **Files**: `particle-system/shaders/emission.metal`
  - **Done when**: Shader compiles, emission kernel reads GPU-computed params from buffer(8)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(emission): read GPU-computed emission params from GpuEmissionParams buffer`
  - _Requirements: FR-1, FR-3_
  - _Design: Section 3.5_

- [ ] 1.6 Add new buffers to ParticlePool and prepare_dispatch pipeline to GpuState
  - **Do**:
    1. In `buffers.rs` ParticlePool struct, add fields:
       - `pub emission_dispatch_args: Retained<ProtocolObject<dyn MTLBuffer>>` (12 bytes)
       - `pub gpu_emission_params: Retained<ProtocolObject<dyn MTLBuffer>>` (16 bytes)
    2. In `ParticlePool::new()`: allocate both buffers, zero-initialize. For `emission_dispatch_args`, init to `DispatchArgs { threadgroups_per_grid: [0, 1, 1] }`
    3. In `ParticlePool::grow()`: reallocate both buffers (simple realloc, no copy needed since GPU recomputes each frame)
    4. In `gpu.rs` GpuState struct, add `pub prepare_dispatch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>`
    5. In `GpuState::new()`: load "prepare_dispatch" function from metallib, create pipeline
    6. Add `read_alive_count_from_indirect(&self) -> u32` method to ParticlePool that reads `indirect_args.contents()` as DrawArgs and returns `instance_count`
  - **Files**: `particle-system/src/buffers.rs`, `particle-system/src/gpu.rs`
  - **Done when**: Build succeeds, new buffers allocated, pipeline loaded
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(buffers): add emission_dispatch_args, gpu_emission_params buffers and prepare_dispatch pipeline`
  - _Requirements: FR-1, FR-3, FR-4_
  - _Design: Section 3.3, 4.2_

- [ ] 1.7 Integrate prepare_dispatch + indirect emission dispatch in main.rs
  - **Do**:
    1. Remove CPU readback of dead_count (lines ~132-136 in `render()`): delete `let dead_count = unsafe { ... pool.dead_list.contents() ... }`
    2. Remove CPU computation of `emission_count` and `actual_burst_count` (lines ~136-138)
    3. Remove CPU write_list counter reset (lines ~94-97): delete `unsafe { let write_ptr = write_list.contents()... (*write_ptr).count = 0; }`
    4. Write `base_emission_rate` to uniforms instead of computed emission_count: `(*uniforms_ptr).base_emission_rate = base_emission` (already done in 1.2 but verify)
    5. Write `burst_count` to uniforms as the raw requested count (not clamped): `(*uniforms_ptr).burst_count = burst_count` (was `actual_burst_count`)
    6. Add prepare_dispatch compute pass BEFORE emission pass:
       - Create compute encoder with label "Prepare Dispatch"
       - Set pipeline to `gpu.prepare_dispatch_pipeline`
       - Bind buffer(0)=dead_list, buffer(1)=uniforms, buffer(2)=write_list, buffer(3)=emission_dispatch_args, buffer(4)=gpu_emission_params
       - Dispatch MTLSize{1,1,1} threadgroups of MTLSize{32,1,1} threads (SIMD-aligned per decision D11/Q1)
       - endEncoding
    7. Change emission dispatch from CPU-computed to indirect:
       - Remove `let threadgroup_count = (emission_count as usize).div_ceil(256)`
       - Remove `compute_encoder.dispatchThreadgroups_threadsPerThreadgroup(...)`
       - Add `unsafe { compute_encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(&pool.emission_dispatch_args, 0, MTLSize{width:256,height:1,depth:1}) }`
    8. Add buffer(8) binding for gpu_emission_params in emission encoder: `compute_encoder.setBuffer_offset_atIndex(Some(&pool.gpu_emission_params), 0, 8)`
    9. Change HUD alive count read: replace `pool.read_alive_count(write_list)` with `pool.read_alive_count_from_indirect()`
  - **Files**: `particle-system/src/main.rs`
  - **Done when**: Build succeeds, CPU no longer reads dead_count or computes emission params
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(main): integrate prepare_dispatch + indirect emission dispatch, remove CPU readback`
  - _Requirements: FR-1, FR-5, FR-6, FR-7, FR-8, FR-13_
  - _Design: Section 4.4, 3.1_

- [ ] 1.8 Update GPU integration tests for new emission kernel signature
  - **Do**:
    1. In `tests/gpu_integration.rs`, update `EmissionBuffers` to include `gpu_emission_params` buffer (16 bytes) and `emission_dispatch_args` buffer (12 bytes)
    2. Update `PhysicsBuffers` similarly
    3. Add `prepare_dispatch` pipeline to `GpuTestContext` and `GpuTestContextFull`
    4. Before dispatching emission in all test helpers, dispatch `prepare_dispatch` first (or manually write GpuEmissionParams values for test isolation)
    5. Update emission kernel buffer bindings to include buffer(8) = gpu_emission_params
    6. For `dispatch_emission` and `dispatch_emission_physics` helpers: either run prepare_dispatch before emission, or manually set GpuEmissionParams to desired values
    7. Verify all 4 GPU integration tests still pass
  - **Files**: `particle-system/tests/gpu_integration.rs`
  - **Done when**: All 4 GPU integration tests pass with the new emission kernel signature
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test --test gpu_integration 2>&1 | tail -10`
  - **Commit**: `test(gpu): update integration tests for prepare_dispatch + indirect emission`
  - _Requirements: NFR-5_
  - _Design: Section 11.2_

- [ ] 1.9 [VERIFY] Quality checkpoint: full test suite
  - **Do**: Run full build and complete test suite including GPU integration tests
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 && cargo test 2>&1 | tail -10`
  - **Done when**: All tests pass (61+ unit tests + 4 GPU integration tests), zero regressions
  - **Commit**: `chore(gpu-centric): pass Phase 1 quality checkpoint` (only if fixes needed)

- [ ] 1.10 POC Checkpoint — verify indirect dispatch works end-to-end
  - **Do**:
    1. Run `cargo build` to confirm compilation
    2. Run `cargo test` to confirm all tests pass
    3. Verify no CPU readback remains: `grep -n "dead_list.contents()" particle-system/src/main.rs` should return nothing
    4. Verify no CPU write_list reset remains: `grep -n "write_ptr.*count = 0" particle-system/src/main.rs` should return nothing
    5. Verify indirect dispatch is used: `grep -n "dispatchThreadgroupsWithIndirectBuffer" particle-system/src/main.rs` should find the emission dispatch
    6. Verify prepare_dispatch is encoded: `grep -n "prepare_dispatch" particle-system/src/main.rs` should find pipeline usage
  - **Files**: None (verification only)
  - **Done when**: All grep checks pass confirming CPU readback eliminated and indirect dispatch integrated
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test 2>&1 | tail -3 && ! grep -q "dead_list.contents()" src/main.rs && echo "POC PASS: no CPU readback"`
  - **Commit**: `feat(gpu-centric): complete Phase 1 POC — CPU readback eliminated`
  - _Requirements: US-T1, US-T2, US-T3 (emission only), US-T5 (partial)_

## Phase 2: Triple Buffering

Focus: Enable CPU/GPU overlap. Change MAX_FRAMES_IN_FLIGHT to 3, add uniform ring buffer.

- [ ] 2.1 Change MAX_FRAMES_IN_FLIGHT to 3 and add uniform ring buffer
  - **Do**:
    1. In `frame.rs`: change `const MAX_FRAMES_IN_FLIGHT: usize = 1` to `3`. Update comment to reflect GPU-centric architecture enables safe triple buffering
    2. In `buffers.rs` ParticlePool: add `pub uniform_ring: Retained<ProtocolObject<dyn MTLBuffer>>` (768 bytes = 3 x 256)
    3. In `ParticlePool::new()`: allocate `uniform_ring` buffer (768 bytes), init all 3 slots with default Uniforms
    4. Add helper method `pub fn uniforms_offset(frame_index: usize) -> usize { frame_index * std::mem::size_of::<Uniforms>() }`
    5. Keep the old single `uniforms` buffer for now (will be removed after main.rs update)
  - **Files**: `particle-system/src/frame.rs`, `particle-system/src/buffers.rs`
  - **Done when**: Build succeeds, MAX_FRAMES_IN_FLIGHT = 3, uniform_ring allocated
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(frame): enable triple buffering with MAX_FRAMES_IN_FLIGHT=3 and uniform ring buffer`
  - _Requirements: FR-9, FR-10_
  - _Design: Section 5.1, 5.2, KB 232_

- [ ] 2.2 Wire uniform ring buffer into main.rs render loop
  - **Do**:
    1. In `main.rs render()`: compute `let uniform_offset = ParticlePool::uniforms_offset(self.frame_ring.frame_index());`
    2. Write uniforms to `pool.uniform_ring` at `uniform_offset` instead of `pool.uniforms`
    3. Update ALL compute encoder `setBuffer_offset_atIndex` calls for uniforms to use `(&pool.uniform_ring, uniform_offset, index)` instead of `(&pool.uniforms, 0, index)`:
       - prepare_dispatch: buffer(1) = uniforms
       - emission: buffer(0) = uniforms
       - grid_populate: buffer(0) = uniforms
       - update: buffer(0) = uniforms
    4. Update render encoder vertex buffer binding for uniforms: buffer(4) = uniform_ring with offset
    5. Remove the old single `uniforms` buffer from ParticlePool (or mark deprecated)
    6. Add pool drain logic for grow under triple buffering (design Section 5.4):
       ```rust
       if let Some(new_size) = self.input.pending_grow.take() {
           for _ in 0..MAX_FRAMES_IN_FLIGHT { self.frame_ring.acquire(); }
           pool.grow(new_size);
           for _ in 0..MAX_FRAMES_IN_FLIGHT { self.frame_ring.signal(); }
       }
       ```
       This requires exposing `signal()` on FrameRing or inline semaphore access.
    7. Add `pub fn signal(&self)` to FrameRing that calls `self.semaphore.signal()`
  - **Files**: `particle-system/src/main.rs`, `particle-system/src/frame.rs`
  - **Done when**: Build succeeds, uniforms written to ring buffer with correct per-frame offset
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(main): wire uniform ring buffer for triple buffering with pool drain on grow`
  - _Requirements: FR-9, FR-10, FR-14_
  - _Design: Section 5.1, 5.4_

- [ ] 2.3 Update FrameRing Drop for triple buffering safety
  - **Do**:
    1. In `frame.rs` Drop impl: drain all in-flight frames, not just one. Current Drop logic handles single-buffer correctly. With 3 in-flight, we need to wait for all pending completions:
       ```rust
       impl Drop for FrameRing {
           fn drop(&mut self) {
               // Wait for all in-flight frames to complete
               if self.acquired_no_handler {
                   self.semaphore.signal();
               }
               // Drain remaining in-flight frames
               for _ in 0..MAX_FRAMES_IN_FLIGHT {
                   self.semaphore.wait(DispatchTime::FOREVER);
                   self.semaphore.signal();
               }
           }
       }
       ```
    2. Verify the FrameRing does not crash on drop with 3 in-flight frames
  - **Files**: `particle-system/src/frame.rs`
  - **Done when**: FrameRing::drop safely drains all in-flight frames
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `fix(frame): safe FrameRing::drop for triple buffering`
  - _Requirements: FR-9_
  - _Design: Section 5.2, KB 232, pitfall #4 from research.md_

- [ ] 2.4 [VERIFY] Quality checkpoint: build + tests after triple buffering
  - **Do**: Run full build and test suite. GPU integration tests may need updates if they depend on single-buffering assumptions.
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 && cargo test 2>&1 | tail -10`
  - **Done when**: All tests pass, build clean
  - **Commit**: `chore(gpu-centric): pass Phase 2 quality checkpoint` (only if fixes needed)

## Phase 3: Full GPU Autonomy — Indirect Dispatch for Update + Grid Populate + Debug

Focus: All variable-count dispatches use indirect args. Add debug infrastructure.

- [ ] 3.1 Extend sync_indirect_args to write next-frame update_dispatch_args
  - **Do**:
    1. In `buffers.rs` ParticlePool: add `pub update_dispatch_args: Retained<ProtocolObject<dyn MTLBuffer>>` (12 bytes)
    2. In `ParticlePool::new()`: allocate and init to `DispatchArgs { threadgroups_per_grid: [pool_size.div_ceil(256) as u32, 1, 1] }` (bootstrap value for first frame, per design R6)
    3. In `ParticlePool::grow()`: reallocate and reinit to new pool_size value
    4. In `render.metal` `sync_indirect_args` kernel: add `device DispatchArgs* update_args [[buffer(2)]]` parameter
    5. After writing draw args, add: `uint threadgroups = max((alive_count + 255) / 256, 1u); update_args->threadgroupsPerGridX = threadgroups; update_args->threadgroupsPerGridY = 1; update_args->threadgroupsPerGridZ = 1;`
  - **Files**: `particle-system/src/buffers.rs`, `particle-system/shaders/render.metal`
  - **Done when**: sync_indirect_args writes both draw args and next-frame dispatch args
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(sync): extend sync_indirect_args to write next-frame update dispatch args`
  - _Requirements: FR-11_
  - _Design: Section 6.2, 4.5_

- [ ] 3.2 Change grid_populate and update to indirect dispatch in main.rs
  - **Do**:
    1. In `main.rs` sync encoder: bind buffer(2) = `pool.update_dispatch_args` via `setBuffer_offset_atIndex`
    2. In `main.rs` grid_populate dispatch: replace CPU-computed `pool.pool_size.div_ceil(256)` with indirect dispatch via `dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(&pool.update_dispatch_args, 0, MTLSize{width:256,height:1,depth:1})`
    3. In `main.rs` update dispatch: same change -- replace CPU-computed dispatch with indirect using `pool.update_dispatch_args`
    4. Grid clear remains fixed at 1024 threadgroups (no change, per decision D3)
  - **Files**: `particle-system/src/main.rs`
  - **Done when**: grid_populate and update use indirect dispatch, grid_clear remains fixed
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3 && grep -c "dispatchThreadgroupsWithIndirectBuffer" src/main.rs | grep -q 3 && echo "3 indirect dispatches found"`
  - **Commit**: `feat(main): indirect dispatch for grid_populate and update kernels`
  - _Requirements: FR-12, US-T3_
  - _Design: Section 4.5_

- [ ] 3.3 Update GPU integration tests for extended sync_indirect_args
  - **Do**:
    1. In `tests/gpu_integration.rs`: add `update_dispatch_args` buffer to `PhysicsBuffers`
    2. Update `dispatch_sync_indirect` helper to bind buffer(2) = update_dispatch_args
    3. Add new test `test_sync_indirect_writes_update_dispatch_args`: emit 100 particles, run update, run sync_indirect_args, read back update_dispatch_args, assert threadgroupsX == ceil(100/256) == 1
    4. Update existing `test_indirect_draw_args_gpu_integration` to include the new buffer binding
  - **Files**: `particle-system/tests/gpu_integration.rs`
  - **Done when**: All GPU integration tests pass including new test
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test --test gpu_integration 2>&1 | tail -10`
  - **Commit**: `test(gpu): add test for sync_indirect_args update_dispatch_args output`
  - _Requirements: NFR-5, NFR-6_
  - _Design: Section 11.2_

- [ ] 3.4 [VERIFY] Quality checkpoint: build + all tests
  - **Do**: Run full build and all tests after Phase 3 core changes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 && cargo test 2>&1 | tail -10`
  - **Done when**: All tests pass, no regressions
  - **Commit**: `chore(gpu-centric): pass Phase 3 quality checkpoint` (only if fixes needed)

- [ ] 3.5 Add F1 GPU capture key
  - **Do**:
    1. In `input.rs` InputState: add `pub capture_next_frame: bool` field, default false
    2. In `input.rs` InputState::handle_key: add `KeyCode::F1 => { self.capture_next_frame = true; }` before pool key fallthrough
    3. In `main.rs render()`: after command buffer creation, add F1 capture logic:
       ```rust
       if self.input.capture_next_frame {
           self.input.capture_next_frame = false;
           eprintln!("[DEBUG] GPU capture requested (set MTL_CAPTURE_ENABLED=1 to enable)");
           // Programmatic capture via MTLCaptureManager
           // Full implementation deferred to Phase 3 testing; for now just log
       }
       ```
    4. Full MTLCaptureManager integration requires objc2-metal bindings for capture APIs. Check availability; if present, implement per design Section 7.1. If not available, add TODO comment.
  - **Files**: `particle-system/src/input.rs`, `particle-system/src/main.rs`
  - **Done when**: F1 key toggles capture flag, build succeeds
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3 && grep -q "capture_next_frame" src/input.rs && echo "F1 capture field exists"`
  - **Commit**: `feat(debug): add F1 GPU capture key support`
  - _Requirements: FR-15_
  - _Design: Section 7.1, KB 137_

- [ ] 3.6 Add debug-telemetry feature flag and DebugTelemetry struct
  - **Do**:
    1. In `Cargo.toml`: add `[features]` section with `debug-telemetry = []`
    2. In `types.rs`: add `DebugTelemetry` struct (8 x u32 = 32 bytes) with layout test. Gate behind `#[cfg(feature = "debug-telemetry")]`
    3. In `types.h`: add `DebugTelemetry` struct gated behind `#if DEBUG_TELEMETRY`
    4. In `build.rs`: conditionally pass `-DDEBUG_TELEMETRY=1` to metal compiler when `CARGO_FEATURE_DEBUG_TELEMETRY` env var is set
    5. In `buffers.rs`: add optional `debug_telemetry` buffer (32 bytes), allocated only when `#[cfg(feature = "debug-telemetry")]`
    6. In `render.metal` sync_indirect_args: add conditional telemetry writes under `#if DEBUG_TELEMETRY`
  - **Files**: `particle-system/Cargo.toml`, `particle-system/src/types.rs`, `particle-system/shaders/types.h`, `particle-system/build.rs`, `particle-system/src/buffers.rs`, `particle-system/shaders/render.metal`
  - **Done when**: `cargo build` succeeds, `cargo build --features debug-telemetry` also succeeds
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3 && cargo build --features debug-telemetry 2>&1 | tail -3`
  - **Commit**: `feat(debug): add debug-telemetry feature flag and DebugTelemetry struct`
  - _Requirements: FR-16, FR-17_
  - _Design: Section 7.2, 7.3_

- [ ] 3.7 [VERIFY] Quality checkpoint: full build with and without debug-telemetry
  - **Do**: Run full build and test suite in both configurations
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test 2>&1 | tail -5 && cargo test --features debug-telemetry 2>&1 | tail -5`
  - **Done when**: All tests pass in both configurations
  - **Commit**: `chore(gpu-centric): pass Phase 3 final quality checkpoint` (only if fixes needed)

## Phase 4: Testing & Verification

Focus: Comprehensive tests per design Section 11. Unit tests, GPU integration tests, conservation invariant.

- [ ] 4.1 Add unit tests for new structs and helpers
  - **Do**:
    1. In `types.rs` tests module, add/verify:
       - `test_dispatch_args_layout`: size==12, align==4, field offsets at 0/4/8
       - `test_gpu_emission_params_layout`: size==16, field offsets at 0/4/8/12
       - `test_uniforms_base_emission_rate_offset`: verify offset of base_emission_rate == 200 (same as old emission_count)
       - `test_uniforms_still_256_bytes`: size_of::<Uniforms>() == 256
    2. In `buffers.rs` tests module, add:
       - `test_uniform_ring_offset`: verify `uniforms_offset(0)==0`, `uniforms_offset(1)==256`, `uniforms_offset(2)==512`
       - `test_pool_new_allocates_dispatch_buffers`: verify emission_dispatch_args.length()==12, gpu_emission_params.length()==16, update_dispatch_args.length()==12
    3. In `frame.rs`, add test:
       - `test_frame_ring_cycles_0_1_2`: create FrameRing, call advance() 6 times, verify frame_index cycles 0,1,2,0,1,2
  - **Files**: `particle-system/src/types.rs`, `particle-system/src/buffers.rs`, `particle-system/src/frame.rs`
  - **Done when**: All new unit tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test -- test_dispatch_args test_gpu_emission test_uniforms_base test_uniform_ring test_frame_ring_cycles test_pool_new_allocates 2>&1 | tail -15`
  - **Commit**: `test(unit): add layout tests for DispatchArgs, GpuEmissionParams, uniform ring`
  - _Requirements: NFR-6_
  - _Design: Section 11.1_

- [ ] 4.2 Add GPU integration test: prepare_dispatch correctness
  - **Do**:
    1. In `tests/gpu_integration.rs`, add test `test_prepare_dispatch_correctness`:
       - Allocate dead_list with count=500 (set counter to 500, fill 500 indices)
       - Set uniforms: base_emission_rate=1000, burst_count=0, pool_size=1000
       - Run prepare_dispatch kernel
       - Read back gpu_emission_params: assert emission_count==500 (clamped to dead_count)
       - Read back emission_dispatch_args: assert threadgroupsPerGridX == ceil(500/256) == 2
       - Read back write_list: assert counter == 0 (was reset)
    2. Add test `test_prepare_dispatch_zero_dead`:
       - dead_count=0, base_emission_rate=10000
       - Assert: emission_count==0, threadgroupsPerGridX==0
    3. Add test `test_prepare_dispatch_burst_clamping`:
       - dead_count=100, base=50, burst=200
       - Assert: emission_count==100, actual_burst==50 (min(200, 100-50) ... actually min(200, 100)=100, then actual_burst=min(200,100)=100. Wait: per design, `actual_burst = min(burst_count, emission_count)` = min(200, 100) = 100. But emission_count = min(50+200, 100) = 100. So actual_burst = min(200, 100) = 100. Test that.
  - **Files**: `particle-system/tests/gpu_integration.rs`
  - **Done when**: All 3 new prepare_dispatch GPU tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test --test gpu_integration -- test_prepare_dispatch 2>&1 | tail -10`
  - **Commit**: `test(gpu): add prepare_dispatch correctness tests`
  - _Requirements: NFR-6_
  - _Design: Section 11.2_

- [ ] 4.3 Add GPU integration test: indirect dispatch round-trip
  - **Do**:
    1. Add test `test_indirect_dispatch_round_trip`:
       - Full pipeline: prepare_dispatch -> emission (indirect) -> grid_clear -> grid_populate -> update -> sync_indirect_args
       - Pool size 1000, base_emission_rate=100, burst=0
       - After full pipeline: read indirect_args.instanceCount, verify alive count > 0 and <= 100
       - Verify conservation: alive_count + dead_count == pool_size
    2. Add test `test_write_list_reset_by_gpu`:
       - Set write_list counter to 999 manually
       - Run prepare_dispatch
       - Read write_list counter: assert == 0
  - **Files**: `particle-system/tests/gpu_integration.rs`
  - **Done when**: Both new tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test --test gpu_integration -- test_indirect_dispatch_round_trip test_write_list_reset 2>&1 | tail -10`
  - **Commit**: `test(gpu): add indirect dispatch round-trip and write_list reset tests`
  - _Requirements: NFR-4, NFR-5_
  - _Design: Section 11.2, 11.5_

- [ ] 4.4 [VERIFY] Quality checkpoint: all tests including new GPU integration
  - **Do**: Run full test suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test 2>&1 | tail -10`
  - **Done when**: All tests pass (unit + GPU integration, old + new)
  - **Commit**: `chore(gpu-centric): pass Phase 4 test quality checkpoint` (only if fixes needed)

- [ ] 4.5 Add conservation invariant stress test
  - **Do**:
    1. Add test `test_conservation_invariant_stress`:
       - Pool size 10000, base_emission_rate=500
       - Run 100 frames of full pipeline (prepare_dispatch -> emission -> grid_clear -> grid_populate -> update -> sync_indirect_args)
       - After every 10th frame: read alive_count (from indirect_args.instanceCount) and dead_count (from dead_list counter)
       - Assert: alive_count + dead_count == pool_size on every check
    2. Use separate command buffers per frame, waitUntilCompleted between frames (synchronous for tests per KB 279)
  - **Files**: `particle-system/tests/gpu_integration.rs`
  - **Done when**: Stress test passes — conservation invariant holds over 100 frames
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test --test gpu_integration -- test_conservation_invariant_stress 2>&1 | tail -5`
  - **Commit**: `test(gpu): add 100-frame conservation invariant stress test`
  - _Requirements: NFR-4_
  - _Design: Section 11.5_

## Phase 5: Refactoring

Focus: Clean up code structure, extract helpers, improve error handling.

- [ ] 5.1 Extract compute dispatch encoding into helper methods
  - **Do**:
    1. In `main.rs` or a new `encode.rs` module: extract repeated compute encoder patterns into helper functions:
       - `encode_prepare_dispatch(cmd_buf, gpu, pool, write_list, uniform_offset)`
       - `encode_emission(cmd_buf, gpu, pool, read_list, uniform_offset)`
       - `encode_grid_clear(cmd_buf, gpu, pool)`
       - `encode_grid_populate(cmd_buf, gpu, pool, read_list, uniform_offset)`
       - `encode_update(cmd_buf, gpu, pool, read_list, write_list, uniform_offset)`
       - `encode_sync_indirect(cmd_buf, gpu, pool, write_list)`
    2. Refactor `render()` to call these helpers, reducing the method from ~300 lines to ~50 lines of high-level orchestration
    3. If extracting to a new module, add `mod encode;` to main.rs
  - **Files**: `particle-system/src/main.rs` (or `particle-system/src/encode.rs`)
  - **Done when**: render() is concise, all tests still pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test 2>&1 | tail -5`
  - **Commit**: `refactor(main): extract compute dispatch encoding into helper methods`
  - _Design: Architecture section_

- [ ] 5.2 Add error handling for buffer allocation failures
  - **Do**:
    1. In `buffers.rs`: ensure all new buffer allocations (emission_dispatch_args, gpu_emission_params, update_dispatch_args, uniform_ring) have clear panic messages matching existing pattern
    2. In `gpu.rs`: ensure prepare_dispatch pipeline creation has clear panic message
    3. Review all `unsafe` blocks added in Phases 1-3: add safety comments documenting why each is safe
  - **Files**: `particle-system/src/buffers.rs`, `particle-system/src/gpu.rs`, `particle-system/src/main.rs`
  - **Done when**: All allocations have descriptive error messages, all unsafe blocks documented
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `refactor(buffers): improve error handling and safety documentation`
  - _Design: Error Handling_

- [ ] 5.3 [VERIFY] Quality checkpoint: full test suite after refactoring
  - **Do**: Run full build and test suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test 2>&1 | tail -10`
  - **Done when**: All tests pass, no regressions from refactoring
  - **Commit**: `chore(gpu-centric): pass refactoring quality checkpoint` (only if fixes needed)

## Phase 6: Quality Gates & PR

- [ ] 6.1 [VERIFY] Full local CI: build + test + clippy
  - **Do**: Run complete local quality suite
  - **Verify**: All commands must pass:
    ```
    cd /Users/patrickkavanagh/gpu_kernel/particle-system && \
    cargo build 2>&1 | tail -3 && \
    cargo test 2>&1 | tail -10 && \
    cargo clippy -- -D warnings 2>&1 | tail -10 && \
    cargo build --features debug-telemetry 2>&1 | tail -3 && \
    cargo test --features debug-telemetry 2>&1 | tail -10
    ```
  - **Done when**: Build, all tests, clippy, and feature-gated build all pass
  - **Commit**: `fix(gpu-centric): address clippy/lint issues` (if fixes needed)

- [ ] 6.2 Create PR and verify CI
  - **Do**:
    1. Verify current branch is a feature branch: `git branch --show-current` (should be `feat/gpu-particle-system` or similar)
    2. If on default branch, STOP and alert user
    3. Push branch: `git push -u origin $(git branch --show-current)`
    4. Create PR using gh CLI:
       ```
       gh pr create --title "feat: GPU-centric particle system with indirect dispatch + triple buffering" --body "..."
       ```
    5. Summary should cover: eliminated CPU readback, indirect dispatch for all variable-count kernels, triple buffering, debug infrastructure
  - **Verify**: `gh pr checks --watch` (wait for CI completion), all checks green
  - **Done when**: PR created, all CI checks passing
  - **If CI fails**: Read failure details with `gh pr checks`, fix locally, push fixes, re-verify

## Phase 7: PR Lifecycle

- [ ] 7.1 Monitor CI and fix any failures
  - **Do**:
    1. Check CI status: `gh pr checks`
    2. If failures: read logs, fix issues, push fixes
    3. Re-verify: `gh pr checks --watch`
  - **Done when**: All CI checks green
  - **Commit**: `fix(ci): resolve CI failures` (if needed)

- [ ] 7.2 [VERIFY] AC checklist — all acceptance criteria verified
  - **Do**: Programmatically verify each acceptance criterion:
    1. AC US-T1: `grep -n "prepare_dispatch" particle-system/shaders/prepare_dispatch.metal | head -3` (kernel exists)
    2. AC US-T1: `! grep -q "dead_list.contents()" particle-system/src/main.rs` (no CPU dead_count read)
    3. AC US-T2: `! grep -q "write_ptr.*count = 0" particle-system/src/main.rs` (no CPU write_list reset)
    4. AC US-T3: `grep -c "dispatchThreadgroupsWithIndirectBuffer" particle-system/src/main.rs` == 3 (emission + grid_populate + update)
    5. AC US-T4: `grep -q "MAX_FRAMES_IN_FLIGHT.*3" particle-system/src/frame.rs` (triple buffering)
    6. AC US-T5: `! grep -q "dead_list.contents\|emission_count.*dead_count\|pool_size.div_ceil" particle-system/src/main.rs` (no CPU GPU-state reads in render loop)
    7. AC US-T6: `grep -q "capture_next_frame" particle-system/src/input.rs` (F1 capture)
    8. AC NFR-5: `cargo test` all pass
  - **Verify**: Run all grep checks and cargo test
  - **Done when**: All acceptance criteria confirmed met
  - **Commit**: None (verification only)
  - _Requirements: All AC-* criteria_

## Notes

- **POC shortcuts taken (Phase 1)**:
  - No debug telemetry in POC
  - GPU integration tests may manually set GpuEmissionParams instead of running full prepare_dispatch pipeline
  - F1 capture is a flag + eprintln, not full MTLCaptureManager integration initially
  - Clippy/lint deferred to Phase 6

- **Production TODOs (addressed in Phases 2-5)**:
  - Pool drain logic for grow under triple buffering
  - FrameRing Drop safety for 3 in-flight frames
  - Full MTLCaptureManager programmatic capture
  - Debug telemetry display in window title
  - ICBs as follow-on (out of scope per requirements)
  - Metal 4 spike after Phase 3 (out of scope per requirements)

- **KB findings most relevant to implementation**:
  - KB 277: Indirect dispatch eliminates CPU-GPU sync (core enabler)
  - KB 148: dispatchThreadgroups(indirectBuffer:) API details
  - KB 232: Triple buffering canonical pattern (semaphore=3, buffer[n%3])
  - KB 134: ~120us empty kernel overhead (prepare_dispatch budget)
  - KB 165: Encoders cheap, driver coalesces sequential compute encoders
  - KB 368: UMA zero-copy (validates StorageModeShared for all buffers)
  - KB 154: StorageModeShared coherency — mitigated by semaphore-guarded access

- **Existing tests baseline**: 61 unit tests + 4 GPU integration tests = 65 total. Zero regressions required (NFR-5).
