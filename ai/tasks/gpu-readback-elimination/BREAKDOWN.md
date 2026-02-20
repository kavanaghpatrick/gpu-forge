---
id: gpu-readback-elimination.BREAKDOWN
module: gpu-readback-elimination
priority: 1
status: pending
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN]
tags: [phase-1, gpu-centric, indirect-dispatch, prepare-dispatch, metal-compute]
testRequirements:
  unit:
    required: true
    pattern: "src/types.rs::test_uniforms_*"
  integration:
    required: false
    pattern: "tests/gpu_integration.rs::test_prepare_dispatch_*"
---
# BREAKDOWN: GPU Readback Elimination (Phase 1)

## Context

Phase 1 is the core architectural change: eliminating the CPU-GPU readback loop that forces strict serialization. Currently, the CPU reads `dead_count` from GPU memory every frame (4 bytes), computes `emission_count` on the CPU, writes it back to uniforms, and computes threadgroup counts for the emission dispatch. This creates a hard sequential dependency where the GPU must finish all work before the CPU can encode the next frame.

The solution is a new `prepare_dispatch` compute kernel that runs on the GPU at the start of each frame's pipeline. It reads `dead_count` directly, computes `emission_count`, writes indirect dispatch arguments, and resets the write list counter -- all without CPU involvement. The emission kernel then dispatches via `dispatchThreadgroupsWithIndirectBuffer` using GPU-written arguments.

From PM.md Section 6, Phase 1: This is HIGH PRIORITY -- eliminates the architectural anti-pattern and enables all subsequent work.
From TECH.md Section 3: Complete `prepare_dispatch` kernel design with MSL code.
From QA.md Section 4.1: Phase 1 quality gate requires 24 tests (struct layout + prepare_dispatch functional + indirect dispatch round-trip + visual parity + performance baseline).

## Tasks

### Task 1: Create `prepare_dispatch.metal` shader

**Acceptance Criteria**:
1. New file `shaders/prepare_dispatch.metal` containing the `prepare_dispatch` kernel function
2. Kernel reads `dead_list` count at buffer(0), `Uniforms` at buffer(1), writes to `write_list` counter at buffer(2), `emission_dispatch_args` at buffer(3), `gpu_emission_params` at buffer(4)
3. Computes `emission_count = min(base_emission_rate + burst_count, dead_count)`
4. Computes `actual_burst_count = min(burst_count, emission_count)`
5. Writes threadgroup count with defense-in-depth clamp to `pool_size / 256`
6. Resets write list counter to 0
7. Uses `if (tid != 0) return` single-thread guard
8. Includes `types.h` for shared struct definitions
9. Compiles successfully via `build.rs` (auto-detected by metallib glob)

**Technical Notes**:
- From TECH.md Section 3.2: Complete MSL source provided; dispatched as 1 threadgroup of 32 threads (1x32 SIMD-aligned per Q&A decision)
- From TECH.md Section 3.4: Thread safety analysis confirms no concurrent writers at this pipeline stage
- From PM.md Section 8: Kernel pseudo-code provided in technical architecture summary

### Task 2: Rename `emission_count` to `base_emission_rate` in types.h and types.rs

**Acceptance Criteria**:
1. `types.h`: Field renamed from `uint emission_count` to `uint base_emission_rate` (same byte offset)
2. `types.rs`: Field renamed from `pub emission_count: u32` to `pub base_emission_rate: u32`
3. `test_uniforms_size_256` still passes (struct size unchanged at 256 bytes)
4. New test `test_uniforms_base_emission_rate_offset` verifies byte offset == 200 (same as old field)
5. All references updated in `main.rs`, `emission.metal`, and `tests/gpu_integration.rs`

**Technical Notes**:
- From UX.md Section 2.2, R1: Semantic shift from "CPU-computed final count" to "CPU-declared intent"
- From TECH.md Section 12.1: Same offset, same size (u32), same padding -- only name and semantic change
- From QA.md Section 13.1: Test file references at lines 42, 83, 242, 560-561 must be updated

### Task 3: Allocate dispatch args and emission params buffers

**Acceptance Criteria**:
1. `ParticlePool` gains `emission_dispatch_args: Retained<ProtocolObject<dyn MTLBuffer>>` (16 bytes, StorageModeShared)
2. `ParticlePool` gains `gpu_emission_params: Retained<ProtocolObject<dyn MTLBuffer>>` (16 bytes, StorageModeShared)
3. Buffers allocated in `ParticlePool::new()` and labeled for Metal debugger ("Emission Dispatch Args", "GPU Emission Params")
4. Buffers initialized to zero
5. `ParticlePool::grow()` does NOT need to resize these buffers (pool-size-independent)

**Technical Notes**:
- From UX.md Section 4.3: Total additional memory is 32 bytes -- negligible vs 380MB SoA at 10M
- From TECH.md Section 3.3: Buffer bindings 3 and 4 for prepare_dispatch kernel
- From UX.md Section 4.5: grow() does not need to resize these buffers

### Task 4: Create `prepare_dispatch` pipeline in gpu.rs

**Acceptance Criteria**:
1. `GpuState` gains `pub prepare_dispatch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>`
2. Pipeline loaded from `prepare_dispatch` function in the metallib
3. Pipeline creation follows existing pattern (e.g., `emission_pipeline`)
4. No changes to metallib discovery or loading path

**Technical Notes**:
- From UX.md Section 4.4: One new pipeline added to GpuState
- From TECH.md Section 8, Phase 1: `build.rs` already globs all `.metal` files -- no build changes needed

### Task 5: Integrate indirect dispatch for emission kernel in main.rs

**Acceptance Criteria**:
1. New `prepare_dispatch` compute encoder added before emission dispatch
2. `prepare_dispatch` dispatched with `MTLSize { width: 1, height: 1, depth: 1 }` threadgroups of `MTLSize { width: 32, height: 1, depth: 1 }` threads (1x32 SIMD-aligned)
3. `prepare_dispatch` encoder binds: dead_list(0), uniforms(1), write_list(2), emission_dispatch_args(3), gpu_emission_params(4)
4. Emission dispatch changed from `dispatchThreadgroups_threadsPerThreadgroup` to `dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup`
5. Emission dispatch uses `pool.emission_dispatch_args` at offset 0, with `MTLSize { width: 256, height: 1, depth: 1 }` threads per threadgroup
6. Emission encoder binds `gpu_emission_params` at buffer index 8

**Technical Notes**:
- From TECH.md Section 4.3-4.4: objc2-metal API signature verified for indirect dispatch
- From TECH.md Q&A Q1: 1x32 SIMD-aligned dispatch (thread 0 does work, 1-31 exit)
- From UX.md Section 4.1: Steps 4-7 of current render() collapse; emission dispatch simplified

### Task 6: Modify emission kernel to read from `GpuEmissionParams`

**Acceptance Criteria**:
1. `emission.metal` gains `device const GpuEmissionParams* emission_params [[buffer(8)]]` parameter
2. Guard changed from `if (tid >= uniforms.emission_count) return` to `if (tid >= emission_params->emission_count) return`
3. Burst detection changed from `uniforms.burst_count` check to `emission_params->actual_burst_count` check: `bool is_burst = (tid < emission_params->actual_burst_count)`
4. `GpuEmissionParams` struct accessible via `#include "types.h"` (struct defined in types.h)

**Technical Notes**:
- From TECH.md Section 3.5: Complete emission kernel modification shown
- From QA.md R8: Buffer index 8 is additive -- does not shift existing bindings 0-7
- From QA.md Section 13.2: Existing emission tests must be updated to allocate and bind GpuEmissionParams

### Task 7: Remove CPU readback of `dead_count`

**Acceptance Criteria**:
1. Lines `let dead_count = unsafe { ... dead_list.contents() ... }` removed from `main.rs` frame path
2. `let emission_count = (base_emission + burst_count).min(dead_count)` computation removed
3. CPU no longer reads `dead_list.contents()` during frame encoding
4. No `pool.dead_list.contents()` calls remain in the per-frame render path

**Technical Notes**:
- From PM.md Section 1: This is the primary CPU-GPU bottleneck being eliminated
- From PM.md Section 4 (PT-07): Code audit verification -- zero bytes read from GPU during encoding
- From TECH.md Section 13, Phase 1 Acceptance: "`dead_list.contents()` never called in frame path"

### Task 8: Remove CPU-side `emission_count` computation

**Acceptance Criteria**:
1. CPU no longer computes `emission_count = (base_emission + burst_count).min(dead_count)`
2. CPU no longer computes `actual_burst_count = burst_count.min(emission_count)`
3. CPU writes `uniforms.base_emission_rate = self.input.physics.emission_rate` (unclamped desired rate)
4. CPU writes `uniforms.burst_count = burst_count` (unclamped, GPU handles clamping)

**Technical Notes**:
- From UX.md Section 2.2: CPU now writes declarative intent, not computed result
- From UX.md Section 5.2: Burst clamping logic moves to `prepare_dispatch` kernel
- From TECH.md Section 12.1: CPU write path simplified to a single assignment

### Task 9: Remove CPU write_list counter reset

**Acceptance Criteria**:
1. CPU no longer performs `(*write_ptr).count = 0` or equivalent pointer write to write_list
2. Write list counter reset now handled by `prepare_dispatch` kernel (Task 1)
3. No `write_list.contents()` writes remain in the per-frame render path

**Technical Notes**:
- From PM.md US-T2: GPU-side write list counter reset eliminates remaining per-frame CPU write to GPU beyond uniforms
- From TECH.md Section 3.2: `write_list[0] = 0` is the first operation in prepare_dispatch

### Task 10: GPU integration test -- `prepare_dispatch` correctness

**Acceptance Criteria**:
1. Test `test_prepare_dispatch_correctness`: dead_count=500, base=1000, burst=0 -> emission_count=500, dispatch_args.X=2
2. Test `test_prepare_dispatch_zero_dead`: dead_count=0, base=10000 -> emission_count=0, args.X=0
3. Test `test_prepare_dispatch_burst_clamping`: dead=100, base=50, burst=200 -> emission_count=100, actual_burst=50
4. Test `test_prepare_dispatch_write_list_reset`: write_list counter set to 999 -> after kernel, counter=0
5. Test `test_prepare_dispatch_exact_fit`: dead=256, base=256 -> args.X=1 (exactly one threadgroup)
6. Test `test_prepare_dispatch_max_clamping`: pool_size=1000, dead=UINT_MAX -> args.X <= 4 (pool_size/256 clamp)
7. All tests use `waitUntilCompleted()` and verify via CPU readback of StorageModeShared buffers

**Technical Notes**:
- From QA.md GI-01 through GI-06: Complete test specifications
- From QA.md Section 6.2: Synchronous execution via waitUntilCompleted acceptable for tests (KB 279)
- From QA.md Section 6.3: Buffer readback via `buffer.contents().as_ptr()` on StorageModeShared

### Task 11: GPU integration test -- indirect emission round-trip

**Acceptance Criteria**:
1. Test `test_indirect_emission_round_trip`: Run prepare_dispatch -> emission (indirect) full pipeline, verify alive_count matches expected emission_count
2. Test `test_indirect_emission_visual_parity`: Run emission via indirect dispatch AND via CPU-computed dispatch with identical params, compare alive counts
3. Conservation invariant checked: `alive_count + dead_count == pool_size`
4. Tests run with Metal Shader Validation enabled (`MTL_SHADER_VALIDATION=1`)

**Technical Notes**:
- From QA.md GI-07, GI-08: Round-trip and parity test specifications
- From QA.md Section 10.4: Conservation invariant is the critical correctness check
- From TECH.md Section 11.2: Complete GPU integration test specifications

### Task 12: Visual parity verification (manual)

**Acceptance Criteria**:
1. Run current (CPU-orchestrated) binary at 1M particles, screenshot at steady state
2. Run new (GPU-centric Phase 1) binary at 1M particles, screenshot at steady state
3. Repeat at 10M particles
4. Particle count, distribution, and motion are visually identical
5. Burst emission produces correct position and spread
6. Pool exhaustion (emission_rate > dead_count) clamps gracefully with no crash

**Technical Notes**:
- From QA.md VP-01 through VP-04: Visual parity test specifications
- From QA.md Section 3.6: Manual testing -- PRNG non-determinism makes pixel-diffing unreliable
- From PM.md Section 4: Visual parity is a primary success metric
