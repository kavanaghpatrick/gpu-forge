---
spec: gpu-centric-transition
phase: research
created: 2026-02-10
source: foreman-spec deep mode (PM + UX + TECH + QA agents)
---
# Research: GPU-Centric Particle System Transition

## Executive Summary

The gpu-particle-system achieves 10M particles at 60fps on M4 but has an artificial CPU-in-the-loop bottleneck: every frame, the CPU reads `dead_count` from a GPU buffer, computes `emission_count`, resets counters, and encodes all dispatches with CPU-computed threadgroup counts. Combined with `MAX_FRAMES_IN_FLIGHT=1`, CPU and GPU execute strictly serially with zero overlap. Industry leaders (UE5 Niagara, Unity VFX Graph, Wicked Engine) have moved to GPU-centric particle architectures using indirect dispatch, achieving millions of particles with minimal CPU involvement.

## GPU-Forge KB Findings (Primary Technical Reference)

601 findings across 11 GPU domain skills were queried. The Tech agent cited 30 findings; QA cited 14. Key findings:

### Indirect Dispatch
- **[KB 277]** gpu-perf: Indirect dispatch eliminates CPU-GPU sync when args are GPU-generated
- **[KB 148]** metal-compute: `dispatchThreadgroups(indirectBuffer:)` reads 3x uint32 threadgroup count from GPU-accessible MTLBuffer
- **[KB 301]** gpu-centric-arch: ICBs enable GPU-driven rendering with 39-44% speedup on Apple Silicon

### Indirect Command Buffers
- **[KB 149]** metal-compute: ICBs enable GPU-encoded compute dispatches, 16,384 command limit
- **[KB 440]** gpu-centric-arch: Metal doesn't support persistent kernels; ICBs are closest equivalent
- **[KB 453]** gpu-centric-arch: Metal ICBs = Apple's equivalent to D3D12 Work Graphs

### Architecture Constraints
- **[KB 151]** metal-compute: Persistent kernels NOT feasible on Metal (GPU watchdog timeout)
- **[KB 152]** metal-compute: Bounded kernel chunks with CPU re-dispatch recommended
- **[KB 232]** gpu-io: Triple buffering canonical for per-frame uniforms (semaphore=3, buffer[n%3])
- **[KB 137]** metal-compute: Single command buffer per frame is optimal
- **[KB 153]** metal-compute: StorageModeShared for CPU-accessible buffers, ring buffer pattern
- **[KB 165]** metal-compute: Encoders are cheap; driver coalesces sequential compute encoders

### Performance
- **[KB 134]** metal-compute: Empty kernel round-trip overhead ~120us on M2 Max (encoder-dominated)
- **[KB 114]** msl-kernels: Threadgroup memory practical max ~32KB for good occupancy
- **[KB 268]** gpu-perf: SoA with sequential indices is 2-3x faster than scattered access
- **[KB 283]** gpu-perf: Atomic contention scales with cores; single-thread avoids contention
- **[KB 451]** gpu-centric-arch: GPU dispatch overhead: 2.1us CUDA, ~120us Metal

### Unified Memory / Apple Silicon
- **[KB 368]** gpu-centric-arch: UMA eliminates GPU-centric barrier: zero-copy data sharing
- **[KB 154]** metal-compute: StorageModeShared coherency: gfx-rs had intermittent failures
- **[KB 279]** gpu-perf: waitUntilCompleted anti-pattern in production; acceptable for tests

### Metal 4 (Future)
- **[KB 442]** gpu-centric-arch: Metal 4 unified compute encoder with pass barriers
- **[KB 555]** metal4-api: GPU Work Graphs for dynamic work amplification
- **[KB 468]** metal4-api: Metal 4 long-lived command buffers
- **[KB 308]** metal4-api: Metal 4 ground-up API redesign

## Industry Validation

| Product | Architecture | Pattern | Scale |
|---------|-------------|---------|-------|
| UE5 Niagara | GPU simulation stages | GPU-side spawn/update/render, CPU only for parameter streaming | Millions |
| Unity VFX Graph | GPU-only simulation | Runs both simulation and rendering on GPU in single program | Millions |
| Wicked Engine | Compute + indirect dispatch | 2 dispatches + 1 indirect draw, CPU freed for gameplay | 1M+ |
| Halo Infinite | GPU-driven rendering | Mesh shaders + indirect dispatch for all geometry | Production AAA |

## Current Architecture Analysis

### CPU-GPU Data Flow Per Frame
- CPU reads: `dead_count` (4 bytes) + `alive_count` for HUD (4 bytes, 2x/sec)
- CPU writes: Uniforms (256 bytes) + write_list counter reset (4 bytes)
- CPU computes: `emission_count`, all threadgroup counts
- GPU computes: emission, grid_clear, grid_populate, physics_update, compaction, render

### Critical CPU-GPU Dependency
```
CPU reads dead_count → computes emission_count → writes uniforms → encodes emission dispatch
```
This creates a hard sequential dependency that forces `MAX_FRAMES_IN_FLIGHT=1`.

## Pitfalls from Original Build

These bugs inform the QA strategy:
1. `packed_float3` vs `float3` stride mismatch (12 vs 16 bytes) → silent data corruption
2. Unsigned atomic underflow (`prev <= 0u` always true for uint) → infinite allocation
3. Triple-buffering with shared SoA buffers → data races across frames
4. `dispatch_semaphore` dropped with value < original → libdispatch crash
5. `println!` inside Objective-C callback → SIGABRT

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|-----------|-------|
| Technical | High | All APIs exist in Metal 3 and objc2-metal |
| Performance | High | Industry-validated pattern, KB findings support |
| Risk | Medium | Triple buffering with shared SoA needs careful validation |
| Effort | 5-7 days | Phased approach, each phase independently testable |

## Recommendations for Requirements

1. Eliminate CPU readback of `dead_count` via GPU-side `prepare_dispatch` kernel
2. Use `dispatchThreadgroupsWithIndirectBuffer` for variable-count dispatches
3. Enable triple buffering (`MAX_FRAMES_IN_FLIGHT=3`) with per-frame uniform ring
4. Extend `sync_indirect_args` to compute next-frame dispatch args
5. Add debug infrastructure (F1 capture, telemetry buffer)
6. Defer ICBs and Metal 4 to follow-on phases

## Sources

- UE5 Niagara GPU Particle System architecture
- Unity VFX Graph documentation
- Wicked Engine GPU particle implementation
- Apple WWDC 2019 "Modern Rendering with Metal"
- Apple WWDC 2020 "Optimize Metal Performance for Apple Silicon"
- Apple WWDC 2025 "Discover Metal 4"
- objc2-metal crate documentation (docs.rs)
- Metal Best Practices Guide: Indirect Buffers, Triple Buffering
- gpu-forge KB: 30+ findings cited across metal-compute, gpu-perf, gpu-centric-arch, msl-kernels, metal4-api
