---
id: debug-infra.BREAKDOWN
module: debug-infra
priority: 4
status: pending
version: 1
origin: spec-workflow
dependsOn: [gpu-readback-elimination.BREAKDOWN]
tags: [phase-3, debug, telemetry, gpu-capture, feature-flag]
testRequirements:
  unit:
    required: true
    pattern: "src/types.rs::test_debug_telemetry_*"
  integration:
    required: false
    pattern: "tests/gpu_integration.rs::test_debug_*"
---
# BREAKDOWN: Debug Infrastructure

## Context

The GPU-centric transition removes CPU-side observability of intermediate pipeline values (dead count, emission count, threadgroup sizes). To compensate, this module provides two complementary debug tools: (1) an F1 key that triggers programmatic Metal GPU Capture for deep buffer inspection, and (2) a 32-byte debug telemetry buffer written by GPU kernels and read asynchronously by the CPU for always-on lightweight monitoring. Both tools are gated behind the `debug-telemetry` Cargo feature flag to ensure zero overhead in release builds.

From UX.md Section 3: Debugging strategy analysis across Metal GPU Capture, System Trace, and telemetry buffers.
From TECH.md Section 7: Complete F1 capture implementation and telemetry buffer design.
From QA.md Q&A Q1: User chose full debug infrastructure (capture key + telemetry buffer).

## Tasks

### Task 1: Add F1 capture key to input handling

**Acceptance Criteria**:
1. `InputState` gains `pub capture_next_frame: bool` field, initialized to `false`
2. F1 key press sets `capture_next_frame = true`
3. Debug message printed to stderr: `[DEBUG] GPU capture requested - next frame will be captured`
4. Flag is consumed (reset to false) when the capture is triggered
5. No runtime overhead when capture is not requested

**Technical Notes**:
- From TECH.md Section 7.1: F1 key binding shown; `KeyCode::F1` handler
- From UX.md Section 7.5: Debug capture key recommendation
- From QA.md Section 10.2: Programmatic capture setup requirements

### Task 2: Implement `MTLCaptureScope` programmatic capture

**Acceptance Criteria**:
1. When `capture_next_frame` is true, the next frame's command buffer is wrapped in an `MTLCaptureScope`
2. Capture outputs to `/tmp/particle_capture.gputrace` file
3. Uses `MTLCaptureManager.sharedCaptureManager()` and `MTLCaptureDescriptor`
4. Capture target is the command queue (captures entire queue)
5. After capture completes, `stopCapture()` called
6. Requires `MTL_CAPTURE_ENABLED=1` environment variable (documented)
7. Graceful fallback if capture fails (log error, continue running)

**Technical Notes**:
- From TECH.md Section 7.1: Complete Rust implementation via objc2-metal shown
- From QA.md Risk R7: Programmatic capture may fail for CLI binary; fallback to Xcode manual capture
- From UX.md Section 3.3: Workflow for inspecting bound resources in captured frame

### Task 3: Add `DebugTelemetry` struct and buffer

**Acceptance Criteria**:
1. `DebugTelemetry` struct defined in `types.rs` with `#[repr(C)]`: frame_number, dead_count_at_prepare, requested_emission, actual_emission, alive_count_post_update, emission_threadgroups, update_threadgroups, _pad (8 x u32 = 32 bytes)
2. Matching struct defined in `types.h` for MSL
3. `size_of::<DebugTelemetry>() == 32` layout test passes (SL-05)
4. 32-byte `MTLBuffer` allocated in `ParticlePool` when `debug-telemetry` feature is enabled (`#[cfg(feature = "debug-telemetry")]`)
5. Buffer labeled "Debug Telemetry" for Metal debugger

**Technical Notes**:
- From TECH.md Section 7.2: Complete struct definition for both Rust and MSL
- From QA.md SL-05: Layout test specification
- From UX.md Section 3.5: "Poor man's GPU printf" -- 32 bytes, zero cost when disabled

### Task 4: Extend `sync_indirect_args` to write telemetry (behind feature flag)

**Acceptance Criteria**:
1. `sync_indirect_args` kernel conditionally writes to telemetry buffer when `DEBUG_TELEMETRY` preprocessor define is set
2. Telemetry fields written: `frame_number`, `alive_count_post_update`, `update_threadgroups`
3. Uses `#if DEBUG_TELEMETRY` guards in MSL
4. Additional buffer binding (buffer index 3 for telemetry, buffer index 4 for uniforms to read frame_number)
5. When feature is disabled, kernel signature and behavior are unchanged
6. Single writer pattern: sync_indirect_args is the sole telemetry writer

**Technical Notes**:
- From TECH.md Section 6.2: Complete Phase 3 sync_indirect_args with `#if DEBUG_TELEMETRY` guards
- From TECH.md Q&A Q4: sync_indirect_args writes all telemetry (single writer, no layout coupling)
- From devops.BREAKDOWN Task 5: `build.rs` conditionally passes `-DDEBUG_TELEMETRY=1` to Metal compiler

### Task 5: Add telemetry display to window title (behind feature flag)

**Acceptance Criteria**:
1. When `debug-telemetry` feature is enabled, CPU reads telemetry buffer asynchronously (~once per second, same cadence as FPS update)
2. Telemetry values displayed in window title alongside existing FPS/particle count display
3. Read is non-blocking: value is 1-2 frames stale (acceptable for display)
4. Display format shows: dead_count, emission_count, alive_count, emission_threadgroups
5. No telemetry code compiled or executed when feature is disabled

**Technical Notes**:
- From TECH.md Section 7.2: CPU reads asynchronously, 1-2 frames late, non-blocking
- From UX.md R4: Display in window title alongside FPS
- From QA.md GI-14: `test_debug_telemetry_buffer` verifies telemetry values are consistent

### Task 6: Cargo.toml `debug-telemetry` feature flag integration

**Acceptance Criteria**:
1. All debug telemetry code gated behind `#[cfg(feature = "debug-telemetry")]` in Rust
2. All MSL telemetry code gated behind `#if DEBUG_TELEMETRY` preprocessor guards
3. `build.rs` passes `-DDEBUG_TELEMETRY=1` to Metal compiler when feature is enabled
4. `cargo build` succeeds with and without `--features debug-telemetry`
5. `cargo test` succeeds with and without `--features debug-telemetry`
6. Zero runtime overhead when feature is disabled (no buffer allocation, no GPU writes, no CPU reads)

**Technical Notes**:
- From devops.BREAKDOWN Task 1: Feature flag defined in Cargo.toml
- From devops.BREAKDOWN Task 5: build.rs conditional compilation flag
- From TECH.md Section 7.2: Feature flag integration across Rust and MSL shown
