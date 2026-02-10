# Research: gpu-particle-system

## Executive Summary

Building a 10M+ interactive GPU particle system on Apple Silicon is **highly feasible**. All required patterns (atomic-based pool management, ping-pong buffers, indirect dispatch, compute→render pipeline) are production-proven across Metal, DirectX, WebGPU, and Vulkan. The M4/M4 Pro provides ~100 GB/s sustained bandwidth with Family 9 dual-issue FP16/FP32 and dynamic caching — sufficient for 10M particles at 60fps with headroom. The Rust host via objc2-metal provides direct Metal API control. GPU Forge knowledge base contributed 140+ verified findings across metal-compute, msl-kernels, simd-wave, gpu-perf, and unified-memory skills.

## External Research

### GPU Particle Architecture Patterns

**Core 3-Kernel Pipeline** (verified across OpenGL, DirectX, WebGPU, Metal):
1. **Emission Kernel**: Allocates particles from GPU-resident free pool using atomic counters
2. **Update Kernel**: Applies physics, forces, lifetime decay in parallel
3. **Render Pass**: Indirect draw calls render visible particles (zero CPU involvement)

**Scale References**:
- Unity GPU Particles: 1M+ at 60fps
- WebGPU interactive galaxy: 1M spiral particles
- DirectX 11 (GTX 1080): 60fps with 1M+ using compute + indirect draw

**Sources**:
- [Wicked Engine GPU Particle System](https://wickedengine.net/2017/11/gpu-based-particle-simulation/)
- [Mike Turitzin - Rendering Particles with Compute Shaders](https://miketuritzin.com/post/rendering-particles-with-compute-shaders/)
- [GPUParticles - DirectX 11 (Brian-Jiang)](https://github.com/Brian-Jiang/GPUParticles)

### Particle Pool Management: Atomic Free Lists

**Implementation**:
- **POOL**: Fixed-size buffer of all particle structs (10M × particle_size)
- **DEAD_LIST**: Index buffer of available particle slots
- **ALIVE_LIST**: Index buffer of active particles (+ depth for sorting)
- **Counters**: Atomic variables for `DEAD_LIST_FRONT`, `ALIVE_LIST_SIZE`, `EMISSION_COUNT`

**Lifecycle**:
1. Init: `DEAD_LIST = [0..POOL_SIZE-1]`, `DEAD_LIST_FRONT = POOL_SIZE`
2. Emission: `atomicAdd(&DEAD_LIST_FRONT, -N)` allocates N particles
3. Simulation: Dead particles compact into new alive list
4. Next frame: Swap alive lists (double-buffer)

**GPU Forge**: Append buffer lock-free pattern embeds `atomic_uint` at buffer[0] (16-byte aligned) to avoid serialization bottlenecks (KB ID: 190, verified).

### Indirect Rendering & Dispatch

**Metal ICBs** (Metal 3+): GPU-encoded dispatch/draw args in MTLBuffer, variable-count dispatches without CPU readback.

**Metal 4 (2025)**: Unified `MTL4ComputeCommandEncoder` consolidates compute + blit + acceleration ops. Producer/Consumer barriers replace manual memory barriers between passes.

**Pattern**:
```
Frame N:
  1. Emission Kernel → updates indirect buffer arg count
  2. Update Kernel → indirect dispatch with count from emission
  3. Render Pass → indirect draw with updated instance count
```

**GPU Forge**: ICB kernel chaining still requires separate encoder passes (KB ID: 150). True persistent kernels not feasible on Metal; use bounded kernel chains.

### Ping-Pong / Triple Buffering

**Ping-pong**: Two particle state buffers swapped each frame. Eliminates read-write hazards without complex fencing.

**Triple buffering** (GPU Forge KB ID: 232, verified): Canonical Metal pattern — 3 pre-allocated buffers in ring with `dispatch_semaphore(3)`. CPU waits on semaphore, writes buffer[n%3], encodes commands, GPU signals on completion. Two buffers can cause stalls; three balances parallelism, memory, and latency.

### Pitfalls to Avoid

- **waitUntilCompleted() anti-pattern** (KB ID: 279): Blocks CPU, creates GPU bubbles. Use completion handlers or semaphore signaling.
- **Excessive global atomics** (KB ID: 283): Bottleneck scales with core count (especially M4 Pro 20 cores). Use threadgroup atomics + reduction pattern.
- **Memory barriers after fragment stage** (KB ID: 121): Extremely expensive. Place barriers between compute passes instead.
- **Dynamic indexing in thread address space** (KB ID: 181): Avoid — use `constant` or `device` address space.
- **Empty kernel round-trip** (KB ID: 134): ~150-250μs baseline overhead per dispatch.

## Rust + Metal Integration (objc2-metal)

### objc2-metal Crate Overview

The `objc2-metal` crate provides safe Rust bindings to Apple's Metal framework through the objc2 ecosystem. It maps Objective-C Metal classes directly to Rust types.

**Key Characteristics**:
- Part of the broader `objc2` framework ecosystem (objc2, objc2-foundation, objc2-metal, etc.)
- Provides type-safe wrappers around MTLDevice, MTLCommandQueue, MTLBuffer, MTLComputePipelineState
- Relies on Objective-C runtime messaging under the hood
- Active development; API coverage expanding

**Project Setup** (Cargo.toml):
```toml
[dependencies]
objc2 = "0.6"
objc2-metal = { version = "0.3", features = ["all"] }
objc2-foundation = "0.3"
winit = "0.30"
raw-window-handle = "0.6"
```

### Metal Pipeline Setup in Rust

```rust
// Device & queue
let device = MTLCreateSystemDefaultDevice().unwrap();
let queue = device.newCommandQueue().unwrap();

// Load .metallib
let library = device.newLibraryWithURL(metallib_url)?;
let function = library.newFunctionWithName(ns_string!("particle_update"))?;
let pipeline = device.newComputePipelineStateWithFunction(function)?;

// Create buffers (StorageModeShared for unified memory)
let buffer = device.newBufferWithLength_options(size, MTLResourceStorageModeShared);
```

### Windowing Integration

- **winit** for cross-platform window creation on macOS
- **CAMetalLayer** attached to the window's view for Metal rendering
- `raw-window-handle` bridges winit and Metal's layer system
- Present drawable via `presentDrawable()` before `commit()`

### Known Limitations & Gotchas

- Feature flags in objc2-metal control which APIs are exposed — use `features = ["all"]` for full access
- Error handling differs from Swift — check return values manually
- Some newer Metal 4 APIs may not yet have bindings
- Objective-C memory management (autorelease pools) still applies in Rust — wrap tight loops in `autoreleasepool`

### Alternative Approaches

- **metal-rs**: Older, less maintained but simpler API surface
- **Raw objc2 calls**: Maximum flexibility, no type safety
- **Recommendation**: objc2-metal is the right choice — active development, type-safe, full Metal coverage

## Apple Silicon GPU Hardware & Optimization

### M4 / M4 Pro Specifications

| Spec | M4 | M4 Pro | M4 Max |
|------|-----|--------|--------|
| GPU Cores | 10 | 16-20 | 40 |
| FP32 TFLOPS | ~2.9 | ~4.6-5.8 | ~14.1 |
| Memory BW | ~100 GB/s | ~273 GB/s | ~546 GB/s |
| SIMD Width | 32 threads | 32 threads | 32 threads |
| GPU Family | 9 | 9 | 9 |
| Dual-Issue | FP16+FP32 | FP16+FP32 | FP16+FP32 |

**GPU Forge Sources**: KB IDs 259-265 (occupancy), 266-270 (memory access), 281 (bandwidth benchmarks)

### Bandwidth Budget Analysis

At 10M particles, ~64 bytes/particle state:
- **Read + Write per frame**: 10M × 64B × 2 = 1.28 GB
- **At 60fps**: 1.28 GB × 60 = 76.8 GB/s sustained
- **M4 @ 100 GB/s**: Feasible with ~23% headroom
- **M4 Pro @ 273 GB/s**: Comfortable with 3.5× headroom

**Practical limit**: 10-15M particles at 60fps due to vertex throughput and thermal constraints. Fanless devices (MacBook Air) throttle significantly under sustained load.

### Threadgroup & SIMD Optimization

- **SIMD width**: 32 threads per simdgroup (KB ID: 310, verified)
- **Optimal threadgroup size**: 256-512 threads (8-16 SIMD groups) for 1D particle compute
- **Register budget**: ≤104 registers → 2 SIMD groups/core; ≤52 registers → 4 SIMD groups (max occupancy) (KB ID: 259)
- **SIMD shuffle bandwidth**: 256 bytes/cycle per core (2× NVIDIA) (KB ID: 314)
- **Divergence penalty**: ~70 cycles for branch divergence (KB ID: 265)
- **16-bit types**: half/ushort use 2× fewer registers → 2× occupancy boost (KB ID: 263)

### Memory Layout: SoA vs AoS

- **Cache line**: 128 bytes (KB ID: 266)
- **No severe penalty for uncoalesced access** on Apple Silicon (unlike NVIDIA/AMD), but coalesced access still improves bandwidth utilization (KB ID: 268)
- **SoA preferred**: Separate arrays for position, velocity, lifetime, color allow partial reads in kernels that only need specific fields
- **Threadgroup memory**: ~60KB per core tile SRAM, purely on-chip (KB ID: 114). Use for per-threadgroup particle reductions.
- **Threadgroup memory banks**: 32 independent banks with classical bank conflict patterns (KB ID: 269)

### Unified Memory Patterns

- **StorageModeShared**: Zero-copy CPU↔GPU access with hardware cache coherence (KB ID: 153, verified)
- **All memory coherent** between CPU and GPU without explicit cache flushes (KB ID: 125, verified)
- **SLC bandwidth**: ~2× DRAM bandwidth per core (15.4-19.8 B/cycle vs 8-10 B/cycle) — hot particle data benefits from SLC residence (KB ID: 113)
- **STREAM benchmark**: M4 GPU achieves ~100 GB/s (83% of theoretical 120 GB/s peak) (KB ID: 101, verified)

### Synchronization Patterns

**Metal Synchronization Hierarchy** (KB ID: 141):
1. **MTLFence**: Cross-pass (encoder) synchronization
2. **threadgroup_barrier()**: Within threadgroup only
3. **mem_flags**: `mem_none` (execution only), `mem_device` (device fence), `mem_threadgroup` (threadgroup fence)

**Metal 4 Barriers** (KB ID: 167): Producer Barriers (block reads until write completes) and Consumer Barriers (prevent writes until reads complete) between compute/render passes.

**Best practice** (KB ID: 137): Submit 1 command buffer per frame (max 2). Each can contain multiple encoder passes (compute → render).

### TBDR Considerations

Apple GPUs use Tile-Based Deferred Rendering. For particle rendering:
- Point sprite rendering is efficient — small fragments stay within tiles
- Alpha blending with many overlapping particles can cause tile memory pressure
- Consider front-to-back sorting for depth rejection (if opaque particles)
- Avoid overdraw-heavy transparent particles without depth sorting

### Profiling Tools

- **Metal System Trace** (Xcode Instruments): Real-time GPU timeline
- **Metal Debugger**: Capture GPU frames, inspect buffers
- **Shader Cost Graph** (Xcode 15+, Family 9+): Visualizes kernel hotspots
- **Metal Counters API**: GPU utilization, Performance Limiters (compute vs. memory stalls)
- **Target metrics**: FP32/FP16 throughput ratio, memory stall %, L1 hit rate, occupancy %

## Codebase Analysis

### Project Structure

The `gpu_kernel` repository is a **Claude Code plugin** called **gpu-forge** providing GPU computing expertise. It is NOT a Rust project — the particle system will be built from scratch.

```
gpu_kernel/
├── .claude-plugin/plugin.json    # gpu-forge plugin manifest (v1.0.0-dev)
├── skills/                       # 11 GPU domain expertise skills (601 findings)
├── templates/                    # Code generation templates
│   ├── metal/                    # 5 MSL templates (blank, gemm, histogram, reduction, scan)
│   ├── swift/                    # 3 Swift templates (Package.swift, main.swift, MetalCompute.swift)
│   └── mlx/                      # 1 MLX template
├── agents/                       # 3 AI agents (architecture, investigation, knowledge)
├── commands/                     # 6 slash commands
├── data/gpu_knowledge.db         # SQLite FTS5 knowledge database (1.1 MB, 601 findings)
├── scripts/kb                    # Knowledge base CLI
├── tests/                        # 194 BATS tests (100% pass)
└── specs/gpu-particle-system/    # THIS SPEC
```

### Existing Patterns & Templates

**Available Metal templates** (production-ready, verified to compile):
- `blank.metal.tmpl` — Minimal compute kernel skeleton
- `reduction.metal.tmpl` — Parallel tree reduction with threadgroup shared memory
- `histogram.metal.tmpl` — Two-phase (threadgroup bins + atomic merge)
- `scan.metal.tmpl` — Hillis-Steele inclusive prefix scan

**Available Swift templates**:
- `MetalCompute.swift.tmpl` (154 lines) — Complete Metal compute pipeline wrapper with error handling, buffer creation, dispatch, timing, and threadgroup sizing

### Starting Point

- **No Rust code exists** — greenfield project
- **No Metal shaders exist** — will author from scratch
- **Templates provide patterns** — reduction, atomics, scan patterns applicable to particle system
- **Knowledge base provides** deep Apple Silicon GPU optimization guidance

## Related Specs

| Name | Relevance | Relationship | May Need Update |
|------|-----------|--------------|-----------------|
| (none) | — | Only spec in project | — |

## Quality Commands

| Type | Command | Source |
|------|---------|--------|
| Plugin Tests | `bats tests/` | .github/workflows/test.yml |
| Unit Tests | `bats tests/unit/` | CI pipeline |
| Integration Tests | `bats tests/integration/` | CI pipeline |
| Golden Queries | `bats tests/golden-queries.bats` | FTS5 relevance tests |
| Knowledge Search | `scripts/kb search "<query>"` | KB CLI tool |
| Rust Lint | `cargo clippy` | TBD (new project) |
| Rust Build | `cargo build --release` | TBD (new project) |
| Rust Test | `cargo test` | TBD (new project) |

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | **HIGH** | All patterns production-proven (MLX, llama.cpp, Wicked Engine, DirectX refs) |
| 10M Particles @ 60fps | **FEASIBLE** | M4: 23% bandwidth headroom. M4 Pro: 3.5× headroom. |
| M4/M4 Pro Suitability | **EXCELLENT** | Family 9 dual-issue + dynamic caching + 32-wide SIMD |
| Rust + objc2-metal | **VIABLE** | Active development, type-safe, full Metal API coverage |
| Implementation Effort | **MEDIUM-LARGE** | ~10-15 days core work across emission, physics, rendering, optimization |
| Risk Level | **MEDIUM** | Occupancy tuning needs profiling. objc2-metal Metal 4 coverage unknown. |

## Recommendations for Requirements

1. **Lock particle struct layout early** — SoA with FP16 where possible to minimize bandwidth
2. **Start with 1M particles** — prove pipeline correctness, then scale to 10M with profiling
3. **Use triple buffering** with `dispatch_semaphore(3)` for CPU-GPU sync
4. **Implement progressive pool growth** — start 2M, grow to 10M (avoids 7.7GB upfront on 16GB machines)
5. **Profile with Metal System Trace** at each milestone — don't optimize blind
6. **Use function constants** for kernel variants (emit vs. update vs. cull) — avoids divergence

## Open Questions

1. **Physics model scope**: Simple force fields only, or particle-particle interactions?
2. **Burst emission rate**: How many particles per click event?
3. **Lifetime model**: Fixed duration or random per-particle?
4. **Render style**: Point sprites with size/color variation, or instanced geometry?
5. **Camera model**: Fixed 2D orthographic, or 3D perspective with orbiting?
6. **Target hardware floor**: M4 (10 cores, ~100 GB/s) or M4 Pro minimum?

## Sources

**Apple Developer**:
- [Metal Documentation](https://developer.apple.com/metal/)
- [Indirect Command Encoding](https://developer.apple.com/documentation/metal/indirect-command-encoding)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
- [Metal 4 Overview](https://www.thinkdifferent.blog/blog/metal-4-apple-s-ground-up-gpu-api-redesign-for-the-ai-era/)

**Academic & Benchmarks**:
- [Apple vs. Oranges HPC Benchmarks (arXiv:2502.05317)](https://arxiv.org/html/2502.05317v1)
- [Metal Benchmarks - Philip Turner](https://github.com/philipturner/metal-benchmarks)

**GPU Particle System References**:
- [Wicked Engine GPU Particle System](https://wickedengine.net/2017/11/gpu-based-particle-simulation/)
- [Mike Turitzin - Rendering Particles with Compute](https://miketuritzin.com/post/rendering-particles-with-compute-shaders/)
- [GPUParticles - DirectX 11](https://github.com/Brian-Jiang/GPUParticles)
- [WebGPU Game Physics 1M Particles](https://markaicode.com/webgpu-physics-simulation-1m-particles/)

**GPU Forge Knowledge Base** (601 findings, 11 skills):
- metal-compute: IDs 136-165 (pipeline, command buffers, indirect dispatch)
- msl-kernels: IDs 173-220 (MSL language, optimization, advanced patterns)
- simd-wave: IDs 310-391 (SIMD operations, reductions, synchronization)
- gpu-perf: IDs 253-287 (profiling, occupancy, optimization)
- unified-memory: IDs 101-125 (bandwidth, SLC, storage modes, coherence)
- gpu-io: IDs 232+ (triple buffering, streaming patterns)
