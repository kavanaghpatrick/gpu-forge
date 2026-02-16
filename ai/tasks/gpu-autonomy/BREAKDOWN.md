---
id: gpu-autonomy.BREAKDOWN
module: gpu-autonomy
priority: 3
status: pending
version: 1
origin: spec-workflow
dependsOn: [gpu-readback-elimination.BREAKDOWN]
tags: [phase-3, gpu-autonomy, indirect-dispatch, sync-indirect-args]
testRequirements:
  unit:
    required: false
    pattern: "src/**/*.rs"
  integration:
    required: false
    pattern: "tests/gpu_integration.rs::test_*indirect*"
---
# BREAKDOWN: Full GPU Autonomy (Phase 3)

## Context

Phase 3 completes the GPU-centric transition by making all variable-count dispatches GPU-driven. After Phase 1, only the emission dispatch uses indirect args. In Phase 3, the `grid_populate` and `update` dispatches also switch to indirect, with their threadgroup counts written by the `sync_indirect_args` kernel at the end of the previous frame. This means the CPU's only per-frame role is: poll input, write uniforms to the ring buffer, encode a fixed command buffer structure, and commit. The CPU never queries GPU state, never computes threadgroup counts, and never reads from GPU buffers during encoding.

The `sync_indirect_args` kernel evolves from a minimal 4-field draw args writer to the pipeline's "end-of-frame bookkeeper" that writes draw args, next-frame dispatch args, and optionally debug telemetry.

From PM.md Section 6, Phase 3: MEDIUM PRIORITY -- architectural completeness.
From TECH.md Section 4.5 and 6: Extended sync_indirect_args design.
From QA.md Section 4.3: Phase 3 exit criteria -- all dispatches are either fixed or indirect.

## Tasks

### Task 1: Extend `sync_indirect_args` to write `update_dispatch_args` and `grid_populate_dispatch_args`

**Acceptance Criteria**:
1. `sync_indirect_args` kernel gains `device DispatchArgs* update_args [[buffer(2)]]` parameter
2. After writing draw args, kernel computes `update_threadgroups = (alive_count + 255) / 256`
3. Writes `update_args->threadgroupsPerGridX = max(update_threadgroups, 1)`, Y=1, Z=1
4. `DispatchArgs` struct accessible via `#include "types.h"`
5. Same dispatch args buffer used for both `grid_populate` and `update` kernels (both use alive_count-based threadgroup sizing)

**Technical Notes**:
- From TECH.md Section 6.2: Complete Phase 3 evolution of sync_indirect_args shown
- From TECH.md Decision D5: Extend sync_indirect_args (not a separate kernel) to avoid ~120us overhead (KB 134)
- From TECH.md Section 4.5: "Write for next frame" pattern is safe due to command queue serialization

### Task 2: Switch `grid_populate` dispatch to indirect

**Acceptance Criteria**:
1. `grid_populate` dispatch in `main.rs` changed from `dispatchThreadgroups_threadsPerThreadgroup` to `dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup`
2. Uses `pool.update_dispatch_args` buffer at offset 0
3. Threads per threadgroup remains `MTLSize { width: 256, height: 1, depth: 1 }` (CPU-specified)
4. Existing `grid_populate` kernel source unchanged (guard `if tid >= alive_count return` still works)
5. CPU no longer computes `div_ceil(pool_size, 256)` for grid_populate threadgroup count

**Technical Notes**:
- From TECH.md Section 4.5: grid_populate and update share the same dispatch args buffer
- From PM.md US-T3: Grid populate dispatch uses indirect args based on alive count
- From TECH.md Decision D3: Grid clear remains fixed (1024 threadgroups), no benefit from indirection

### Task 3: Switch `update` dispatch to indirect

**Acceptance Criteria**:
1. `update` (physics) dispatch in `main.rs` changed from `dispatchThreadgroups_threadsPerThreadgroup` to `dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup`
2. Uses `pool.update_dispatch_args` buffer at offset 0 (shared with grid_populate)
3. Threads per threadgroup remains `MTLSize { width: 256, height: 1, depth: 1 }`
4. Existing `update_physics_kernel` source unchanged
5. CPU no longer computes `div_ceil(pool_size, 256)` for update threadgroup count

**Technical Notes**:
- From TECH.md Section 4.5: Same buffer as grid_populate (both need `alive_count / 256` threadgroups)
- From PM.md US-T3: Physics update dispatch uses indirect args based on alive count
- From QA.md GI-13: Full autonomous frame test verifies all indirect dispatches

### Task 4: Allocate `update_dispatch_args` buffer

**Acceptance Criteria**:
1. `ParticlePool` gains `update_dispatch_args: Retained<ProtocolObject<dyn MTLBuffer>>` (16 bytes, StorageModeShared)
2. Buffer allocated in `ParticlePool::new()` and labeled "Update Dispatch Args"
3. Buffer initialized to `{ pool_size / 256, 1, 1 }` (conservative first-frame bootstrap)
4. Re-initialized in `ParticlePool::grow()` with new pool_size
5. Bound to `sync_indirect_args` encoder at buffer index 2

**Technical Notes**:
- From TECH.md Section 4.5: First-frame bootstrap must initialize to pool_size / 256 (kernel guards handle over-dispatch)
- From TECH.md Risk R6: Without initialization, first frame dispatches with uninitialized values
- From UX.md Section 4.3: 16 bytes total, negligible memory

### Task 5: GPU integration test -- all indirect dispatches

**Acceptance Criteria**:
1. `test_sync_indirect_writes_update_args`: Run emission + update producing known alive_count, then sync_indirect_args. Read update_dispatch_args.X. Assert == ceil(alive_count / 256).
2. `test_full_gpu_autonomous_frame`: Run complete pipeline: prepare_dispatch -> emission(indirect) -> grid_clear -> grid_populate(indirect) -> update(indirect) -> sync_indirect_args. Verify: alive + dead == pool_size; all dispatches produce correct results.
3. Tests run with Metal Shader Validation enabled
4. Conservation invariant checked after full pipeline frame

**Technical Notes**:
- From QA.md GI-12, GI-13: Complete test specifications
- From QA.md Section 8.4: Phase 3 exit criteria -- all dispatches audited
- From TECH.md Section 11.2: GPU integration test descriptions

### Task 6: Verify CPU only writes uniforms and encodes commands (zero GPU state queries)

**Acceptance Criteria**:
1. Code audit: No `buffer.contents()` reads from any GPU buffer during frame encoding path (except HUD alive count which is passive)
2. Code audit: No CPU-computed threadgroup counts that depend on GPU state
3. Code audit: CPU's per-frame operations are exactly: write uniforms[frame_idx], encode fixed command buffer, commit
4. All variable-count dispatches (emission, grid_populate, update) use indirect dispatch
5. Only grid_clear remains CPU-specified (1024 fixed threadgroups)

**Technical Notes**:
- From PM.md US-T5: "CPU's only per-frame role is: poll input, update uniforms, encode command buffer, commit"
- From PM.md Section 4 (PT-07): CPU reads 0 bytes from GPU during encoding
- From QA.md Section 8.4: "CPU frame path audited: Only writes to uniform_ring + encodes fixed command buffer"
