---
id: triple-buffering.BREAKDOWN
module: triple-buffering
priority: 2
status: pending
version: 1
origin: spec-workflow
dependsOn: [gpu-readback-elimination.BREAKDOWN]
tags: [phase-2, triple-buffering, frame-ring, uniform-ring, performance]
testRequirements:
  unit:
    required: true
    pattern: "src/frame.rs::test_frame_ring_*"
  integration:
    required: false
    pattern: "tests/gpu_integration.rs::test_triple_buffer_*"
---
# BREAKDOWN: Triple Buffering (Phase 2)

## Context

With CPU readback eliminated in Phase 1, the system can now safely move to triple buffering. Currently, `MAX_FRAMES_IN_FLIGHT = 1` means the CPU and GPU execute strictly serially -- the CPU blocks on the semaphore for ~15ms waiting on GPU completion before it can encode the next frame's ~1ms of work. Triple buffering (`MAX_FRAMES_IN_FLIGHT = 3`) allows the CPU to encode up to 2 frames ahead, overlapping CPU and GPU work and reducing the effective CPU contribution to frame time to near zero.

The key challenge is ensuring no data races on shared resources. SoA particle buffers remain shared (safe via ping-pong index-disjoint access per TECH.md Section 5.3 analysis). Only the Uniforms buffer requires tripling, as the CPU writes different values each frame while the GPU may still be reading previous frames' uniforms.

From PM.md Section 6, Phase 2: MEDIUM PRIORITY -- enables CPU-GPU overlap, dependent on Phase 1 completion.
From TECH.md Section 5: Complete triple buffering design with data race analysis.
From QA.md Section 4.2: Phase 2 quality gate requires extended stress testing (60K frames at 10M particles).

## Tasks

### Task 1: Change `MAX_FRAMES_IN_FLIGHT` from 1 to 3

**Acceptance Criteria**:
1. `frame.rs`: `MAX_FRAMES_IN_FLIGHT` constant changed from `1` to `3`
2. `DispatchSemaphore` initialized with count `3` (was `1`)
3. Up to 3 frames can be in flight simultaneously
4. Compilation succeeds with no other changes (single change should not break anything)

**Technical Notes**:
- From TECH.md Section 5.2: `DispatchSemaphore::new(MAX_FRAMES_IN_FLIGHT as isize)`
- From PM.md US-T4: `MAX_FRAMES_IN_FLIGHT = 3` with `dispatch_semaphore(3)`
- From QA.md UT-04: Test that semaphore is initialized with count == 3

### Task 2: Add `frame_index` to `FrameRing`

**Acceptance Criteria**:
1. `FrameRing` struct gains `frame_index: usize` field, initialized to 0
2. `FrameRing::advance()` method cycles `frame_index = (frame_index + 1) % MAX_FRAMES_IN_FLIGHT`
3. `FrameRing::frame_index()` public getter returns current index
4. Frame index cycles 0, 1, 2, 0, 1, 2... for MAX_FRAMES_IN_FLIGHT=3

**Technical Notes**:
- From TECH.md Section 5.2: Complete `FrameRing` implementation shown
- From QA.md UT-01: `test_frame_ring_cycles_0_1_2` verifies cycling pattern
- From TECH.md Section 5.1: Used to index into uniform ring buffer

### Task 3: Allocate 768B uniform ring buffer

**Acceptance Criteria**:
1. A single 768-byte `MTLBuffer` allocated with `StorageModeShared` (replaces or supplements current single 256B uniform buffer)
2. Buffer labeled "Uniform Ring" for Metal debugger
3. Helper method `uniforms_offset(frame_index: usize) -> usize` returns `frame_index * size_of::<Uniforms>()`
4. `uniforms_offset(0) == 0`, `uniforms_offset(1) == 256`, `uniforms_offset(2) == 512`

**Technical Notes**:
- From TECH.md Section 5.1: Single 768B MTLBuffer, CPU writes at `frame_index * 256` offset
- From TECH.md Q&A Q2: Follows Apple canonical triple buffering pattern (KB 232)
- From QA.md UT-02, UT-03: Offset math tests and total size verification

### Task 4: Update uniform writes to use `frame_index` offset

**Acceptance Criteria**:
1. CPU writes uniforms to `uniform_ring` at byte offset `frame_index * 256` each frame
2. All compute encoders bind `uniform_ring` with the correct byte offset via `setBuffer_offset_atIndex`
3. Render encoder binds `uniform_ring` with the correct byte offset
4. Per-frame uniform values are isolated: frame N's uniforms are not overwritten until GPU signals completion for frame slot N
5. `frame_ring.advance()` called at the appropriate point to cycle frame_index

**Technical Notes**:
- From TECH.md Section 5.1: `compute_encoder.setBuffer_offset_atIndex(Some(&pool.uniform_ring), uniform_offset, 0)`
- From TECH.md Section 5.5: Semaphore guarantees ordering -- CPU does not write slot until GPU signals
- From QA.md GI-11: `test_uniform_ring_no_overwrite` verifies distinct values per frame slot

### Task 5: Update `pool.grow()` to drain all semaphores

**Acceptance Criteria**:
1. Before `pool.grow(new_size)`, all in-flight frames are drained: acquire all `MAX_FRAMES_IN_FLIGHT` semaphore slots
2. After grow completes, all semaphore slots are re-signaled to restore the original count
3. Growth causes a visible stutter (3 frames drained) but no crash or data corruption
4. New dispatch args buffers re-initialized for new pool_size if needed

**Technical Notes**:
- From TECH.md Section 5.4: Complete drain-grow-restore pattern shown
- From UX.md Section 5.3: Growth stall increases from ~1 frame to ~3 frames (acceptable for infrequent operation)
- From QA.md GI-10: `test_pool_grow_under_triple_buffer` verifies grow works correctly

### Task 6: Update HUD to read from `indirect_args.instanceCount`

**Acceptance Criteria**:
1. HUD alive count reads from `pool.indirect_args.contents().instance_count` instead of `pool.read_alive_count(write_list)`
2. Read is passive (no synchronization, no blocking): value was written by `sync_indirect_args` before render pass
3. Value is inherently stale by 1-2 frames -- acceptable for display purposes
4. Old `read_alive_count()` method can be deprecated or removed

**Technical Notes**:
- From PM.md Q&A Q3: User decision -- read from `indirect_args.instanceCount`
- From TECH.md Section 7.3: Zero-cost on Apple Silicon UMA (KB 368)
- From UX.md Section 4.5: `read_alive_count_from_indirect()` method shown

### Task 7: Unit tests -- frame ring cycling

**Acceptance Criteria**:
1. `test_frame_ring_cycles_0_1_2`: FrameRing with MAX=3 cycles through 0, 1, 2, 0, 1, 2 over 6 advances
2. `test_frame_ring_semaphore_count_3`: Semaphore initialized with count == MAX_FRAMES_IN_FLIGHT == 3
3. `test_uniform_ring_offset_math`: Offsets are 0, 256, 512 for indices 0, 1, 2
4. `test_uniform_ring_total_size`: Ring buffer total is 768 bytes (3 x 256)
5. All tests pass via `cargo test --lib`

**Technical Notes**:
- From QA.md UT-01 through UT-04: Complete unit test specifications
- From QA.md Section 8.3: Frame ring tests are Phase 2 exit criteria
- These are merge-blocking tests (run on any platform)

### Task 8: GPU stress test -- 60K frames at 10M particles

**Acceptance Criteria**:
1. Run 60,000 frames (~16 minutes at 60fps) at 10M particles with `MAX_FRAMES_IN_FLIGHT = 3`
2. Conservation invariant `alive_count + dead_count == pool_size` checked periodically (every 1000 frames)
3. No crash, no data corruption, no visual artifacts
4. Test can be run as a long-running integration test (not part of default `cargo test`)
5. Run at least 3 times to catch stochastic coherency issues (KB 154)

**Technical Notes**:
- From QA.md Q&A Q1: 60K frames / ~16 min soak test to catch timing-dependent bugs
- From QA.md RG-04: Conservation invariant is the primary correctness check
- From TECH.md Section 5.5: Semaphore-guarded access pattern should prevent coherency issues

### Task 9: Performance measurement -- CPU blocked time

**Acceptance Criteria**:
1. Capture Metal System Trace showing CPU-GPU overlap with triple buffering
2. CPU blocked time (semaphore wait) should be < 1ms (was ~15ms with single buffering)
3. CPU encoding time should remain ~1ms
4. No GPU frame time regression from Phase 1 baseline
5. Document results for Phase 2 performance baseline

**Technical Notes**:
- From QA.md PT-04: CPU-GPU overlap > 50% visible in Metal System Trace timeline
- From PM.md Section 4: CPU blocked time target is 0ms (freed for other work)
- From QA.md Section 7.1: Performance acceptance criteria table
