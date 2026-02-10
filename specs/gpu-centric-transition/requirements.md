---
spec: gpu-centric-transition
phase: requirements
created: 2026-02-10
source: foreman-spec PM agent analysis + user Q&A
---
# Requirements: GPU-Centric Particle System Transition

## Goal

Transition the gpu-particle-system from CPU-orchestrated to GPU-centric architecture, eliminating all per-frame CPU readback of GPU state, enabling CPU/GPU overlap via triple buffering, and making all variable-count dispatches GPU-driven via indirect dispatch.

## User Stories

### US-T1: GPU-Side Emission Count Computation
**As a** GPU particle system,
**I want** the GPU to compute emission_count from dead_count and base_emission_rate,
**So that** the CPU never reads GPU buffer state per frame.

**Acceptance Criteria**:
- A `prepare_dispatch` kernel computes `emission_count = min(base_emission_rate + burst_count, dead_count)` on GPU
- CPU writes `base_emission_rate` (intent) not `emission_count` (derived value) to Uniforms
- `dead_count` is never read by CPU in the render loop
- Visual output is identical to current system

### US-T2: GPU-Side Counter Reset
**As a** GPU particle system,
**I want** the GPU to reset the write_list counter to 0,
**So that** the CPU never writes to GPU buffers per frame (beyond uniforms).

**Acceptance Criteria**:
- `prepare_dispatch` kernel resets `write_list[0] = 0`
- CPU code that resets write_list counter is removed
- No CPU pointer writes to GPU buffers in the render loop (only uniforms)

### US-T3: Indirect Dispatch for All Variable-Count Kernels
**As a** GPU particle system,
**I want** emission, grid_populate, and update kernels dispatched via indirect dispatch,
**So that** the CPU never computes threadgroup counts from GPU state.

**Acceptance Criteria**:
- Emission kernel dispatched with `dispatchThreadgroupsWithIndirectBuffer`
- Grid populate kernel dispatched with indirect buffer (alive count based)
- Update kernel dispatched with indirect buffer (alive count based)
- Grid clear remains fixed dispatch (no GPU dependency)
- `MTLDispatchThreadgroupsIndirectArguments` written by GPU kernels

### US-T4: Triple Buffering
**As a** GPU particle system,
**I want** CPU and GPU to overlap execution via triple buffering,
**So that** CPU doesn't block waiting for GPU completion before encoding the next frame.

**Acceptance Criteria**:
- `MAX_FRAMES_IN_FLIGHT` changed from 1 to 3
- Per-frame uniform ring buffer (768B = 3 x 256B)
- CPU writes uniforms at `frame_index * 256` offset
- Semaphore pacing with 3 in-flight frames
- Pool growth drains all 3 in-flight frames before resizing

### US-T5: GPU-Autonomous Pipeline
**As a** GPU particle system,
**I want** the CPU's only per-frame role to be: write uniforms + encode fixed command buffer structure,
**So that** all per-frame decision-making is GPU-resident.

**Acceptance Criteria**:
- CPU per-frame work: write uniforms, encode command buffer, commit
- Zero CPU reads of GPU buffers in render loop
- Zero CPU computation of dispatch parameters
- All variable dispatch sizes determined by GPU

### US-T6: Debug Infrastructure
**As a** developer,
**I want** F1 key GPU capture and optional telemetry buffer,
**So that** I can debug the GPU-centric pipeline without always-on overhead.

**Acceptance Criteria**:
- F1 key triggers programmatic Metal GPU Capture (MTLCaptureScope)
- 32-byte DebugTelemetry buffer written by sync_indirect_args
- Telemetry gated behind `--features debug-telemetry` Cargo feature flag
- Telemetry displayed in window title when enabled

## Functional Requirements

| ID | Requirement | Priority | Story |
|----|-------------|----------|-------|
| FR-1 | New `prepare_dispatch` compute kernel reads dead_count, computes emission_count, writes indirect dispatch args, resets write_list counter | Must | US-T1, US-T2 |
| FR-2 | Rename `emission_count` to `base_emission_rate` in Uniforms struct (types.rs + types.h) | Must | US-T1 |
| FR-3 | New `GpuEmissionParams` 16-byte side-channel buffer for GPU-computed emission_count | Must | US-T1 |
| FR-4 | New `DispatchArgs` struct matching `MTLDispatchThreadgroupsIndirectArguments` (3 x u32) | Must | US-T3 |
| FR-5 | Emission kernel dispatched via `dispatchThreadgroupsWithIndirectBuffer` | Must | US-T3 |
| FR-6 | Remove CPU readback of `dead_count` from render loop | Must | US-T1 |
| FR-7 | Remove CPU computation of `emission_count` and threadgroup counts | Must | US-T5 |
| FR-8 | Remove CPU write of `write_list[0] = 0` | Must | US-T2 |
| FR-9 | `MAX_FRAMES_IN_FLIGHT` = 3 with dispatch semaphore pacing | Should | US-T4 |
| FR-10 | Per-frame uniform ring buffer (single 768B buffer with offsets) | Should | US-T4 |
| FR-11 | `sync_indirect_args` writes `update_dispatch_args` and `grid_populate_dispatch_args` | Should | US-T3 |
| FR-12 | Grid populate and update dispatched via indirect buffer | Should | US-T3 |
| FR-13 | HUD alive count reads `indirect_args.instanceCount` | Should | US-T5 |
| FR-14 | Pool growth drains all 3 in-flight frames before resize | Should | US-T4 |
| FR-15 | F1 key triggers MTLCaptureScope programmatic capture | Could | US-T6 |
| FR-16 | DebugTelemetry buffer (32B) written by sync_indirect_args | Could | US-T6 |
| FR-17 | `debug-telemetry` feature flag in Cargo.toml | Could | US-T6 |

## Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-1 | Frame time must not regress (<=16.7ms for 60fps at 10M particles) | Must |
| NFR-2 | CPU blocked time should decrease from ~15ms to <1ms with triple buffering | Should |
| NFR-3 | Memory overhead increase < 1KB (dispatch args + emission params + uniform ring) | Must |
| NFR-4 | Visual output must be identical (conservation invariant: alive + dead = pool_size) | Must |
| NFR-5 | All existing tests must pass (zero regression) | Must |
| NFR-6 | New struct layout tests for DispatchArgs and GpuEmissionParams | Must |
| NFR-7 | Metal Shader Validation clean (zero warnings) on all GPU tests | Should |

## Out of Scope

- Indirect Command Buffers (ICBs) — follow-on after Phase 3
- Metal 4 APIs — time-boxed spike after Phase 3
- Persistent/mega-kernels — not feasible on Metal [KB 151]
- Particle rendering changes — already indirect draw
- Physics kernel optimization — orthogonal to architecture
- New particle features — no new effects
- Multi-GPU support — single GPU only

## Glossary

| Term | Definition |
|------|-----------|
| Indirect dispatch | GPU writes threadgroup count to a buffer; CPU encodes dispatch without knowing the count |
| ICB | Indirect Command Buffer — GPU encodes entire command buffers |
| prepare_dispatch | New compute kernel that runs first each frame, computing dispatch params |
| GpuEmissionParams | 16-byte buffer carrying GPU-computed emission_count to emission kernel |
| base_emission_rate | Renamed field in Uniforms: CPU declares desired rate, GPU computes actual |
| Conservation invariant | alive_count + dead_count == pool_size (always true if correct) |
