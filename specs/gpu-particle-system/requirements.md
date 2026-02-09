---
spec: gpu-particle-system
phase: requirements
created: 2026-02-09
---

# Requirements: GPU Particle System

## Goal

Render 10M+ interactive particles at 60fps on Apple Silicon (M4 minimum) using a fully GPU-driven pipeline. Rust host handles only windowing, mouse input, and command buffer submission; all emission, physics, lifecycle, and rendering execute in Metal compute and render shaders.

---

## User Stories

### US-1: Window Creation and Metal Initialization

**As a** developer running the demo
**I want** the application to open a window and initialize a Metal rendering pipeline
**So that** I have a visible rendering surface backed by GPU compute and render pipelines

**Acceptance Criteria:**
- [ ] AC-1.1: Application opens a resizable window (default 1280x720) using winit
- [ ] AC-1.2: CAMetalLayer attached to window view with correct pixel format (BGRA8Unorm)
- [ ] AC-1.3: MTLDevice, MTLCommandQueue, and all compute/render pipeline states created at startup
- [ ] AC-1.4: Metal shader library (.metallib) compiled and loaded at build time
- [ ] AC-1.5: Triple-buffer ring with dispatch_semaphore(3) operational before first frame
- [ ] AC-1.6: Application exits cleanly on window close (no GPU resource leaks)

### US-2: Particle Pool Initialization

**As a** developer running the demo
**I want** the GPU particle pool allocated and initialized
**So that** particles can be emitted immediately after startup

**Acceptance Criteria:**
- [ ] AC-2.1: Initial pool allocated at 1M particles (progressive growth, not 10M upfront)
- [ ] AC-2.2: SoA buffers created for position (float3), velocity (float3), lifetime (half2: age + maxAge), color (half4), size (half)
- [ ] AC-2.3: Dead list initialized with all indices [0..pool_size-1]
- [ ] AC-2.4: Alive list and atomic counters zeroed
- [ ] AC-2.5: Ping-pong alive list buffers allocated (two sets)
- [ ] AC-2.6: Indirect dispatch/draw argument buffers allocated
- [ ] AC-2.7: All buffers use MTLResourceStorageModeShared (unified memory, zero-copy)
- [ ] AC-2.8: Total initial memory < 200 MB for 1M particle pool

### US-3: Continuous Particle Emission

**As a** user viewing the demo
**I want** particles to emit continuously from a source point
**So that** there is always visual activity on screen

**Acceptance Criteria:**
- [ ] AC-3.1: Emission kernel runs each frame, allocating N particles from dead list via atomic decrement
- [ ] AC-3.2: Emission rate configurable (default: 100K particles/frame at 60fps)
- [ ] AC-3.3: New particles initialized with randomized position (near emitter), velocity (directional spread), lifetime (1-5s range), color, and size
- [ ] AC-3.4: Random values generated on GPU using hash-based PRNG (no CPU random state)
- [ ] AC-3.5: Emission stops gracefully when dead list exhausted (no overflow, no crash)
- [ ] AC-3.6: Emitter position defaults to screen center in world space

### US-4: Burst Emission on Click

**As a** user interacting with the demo
**I want** a burst of particles to emit at the mouse cursor position when I click
**So that** I get immediate, satisfying visual feedback from my interaction

**Acceptance Criteria:**
- [ ] AC-4.1: Left mouse click emits burst of 10K-50K particles at cursor world position
- [ ] AC-4.2: Burst particles have higher initial velocity than continuous particles
- [ ] AC-4.3: Burst emission uses same dead list allocation path (atomic, no special case)
- [ ] AC-4.4: Multiple rapid clicks queue correctly without frame drops
- [ ] AC-4.5: Burst count clamped to available dead list slots (no overflow)
- [ ] AC-4.6: Mouse screen coordinates correctly unprojected to 3D world position

### US-5: GPU Physics Simulation

**As a** user viewing the demo
**I want** particles to move with physically plausible behavior
**So that** the simulation looks dynamic and organic

**Acceptance Criteria:**
- [ ] AC-5.1: Update kernel applies gravity (configurable direction and magnitude)
- [ ] AC-5.2: Velocity integration uses semi-implicit Euler (position += velocity * dt, velocity += acceleration * dt)
- [ ] AC-5.3: Global drag/damping applied per frame to prevent infinite acceleration
- [ ] AC-5.4: Particles respect a bounding volume (soft bounce or wrap at boundaries)
- [ ] AC-5.5: dt derived from actual frame time (passed as uniform), not hardcoded 1/60
- [ ] AC-5.6: Physics runs entirely in compute shader; zero CPU physics logic

### US-6: Grid-Based Particle Interactions

**As a** user viewing the demo
**I want** particles to exhibit density-aware behavior (clustering, repulsion, pressure)
**So that** particle motion appears collectively intelligent rather than independent

**Acceptance Criteria:**
- [ ] AC-6.1: 3D uniform grid (64x64x64 cells minimum) built each frame from particle positions
- [ ] AC-6.2: Grid populated via atomic increment per cell (particle count density field)
- [ ] AC-6.3: Each particle reads density from its 3x3x3 neighborhood (27 cells) to compute approximate pressure gradient
- [ ] AC-6.4: Pressure force pushes particles from high-density to low-density regions
- [ ] AC-6.5: Grid clear + populate + read completes within 3ms at 10M particles on M4
- [ ] AC-6.6: No per-particle neighbor lists; grid density approximation only (not N-body or SPH)
- [ ] AC-6.7: Interaction strength configurable via uniform parameter

### US-7: Particle Lifecycle Management

**As a** user viewing the demo
**I want** particles to age and die naturally
**So that** the pool recycles and the visual effect evolves over time

**Acceptance Criteria:**
- [ ] AC-7.1: Each particle's age incremented by dt each frame
- [ ] AC-7.2: Particle killed when age >= maxAge
- [ ] AC-7.3: Dead particles appended to dead list via atomic increment (lock-free)
- [ ] AC-7.4: Alive list compacted each frame (stream compaction: alive particles written to next-frame alive list)
- [ ] AC-7.5: Alive list swap (ping-pong) occurs each frame with zero CPU involvement
- [ ] AC-7.6: Indirect draw argument buffer updated with alive count after compaction
- [ ] AC-7.7: Color and size interpolate over lifetime (e.g., fade-out alpha, shrink)

### US-8: 3D Instanced Geometry Rendering

**As a** user viewing the demo
**I want** particles rendered as 3D instanced geometry with perspective
**So that** the visual quality exceeds basic point sprites

**Acceptance Criteria:**
- [ ] AC-8.1: Each particle rendered as instanced geometry (billboard quad or low-poly sphere)
- [ ] AC-8.2: Instance data (position, color, size) read from SoA particle buffers
- [ ] AC-8.3: Indirect draw call uses alive count from compaction (no CPU readback)
- [ ] AC-8.4: Vertex shader applies model-view-projection transform per instance
- [ ] AC-8.5: Fragment shader applies per-particle color with alpha blending
- [ ] AC-8.6: Render pass produces correct output at 10M visible instances
- [ ] AC-8.7: Depth buffer enabled; alpha-blended particles sorted or use OIT approximation

### US-9: 3D Perspective Camera

**As a** user viewing the demo
**I want** a 3D perspective camera viewing the particle field
**So that** depth and spatial structure are perceivable

**Acceptance Criteria:**
- [ ] AC-9.1: Perspective projection with configurable FOV (default 60 degrees), near/far planes
- [ ] AC-9.2: Camera positioned to frame the full particle volume at startup
- [ ] AC-9.3: View and projection matrices computed on CPU, uploaded as uniforms each frame
- [ ] AC-9.4: Camera orbits around center point (mouse drag to rotate, scroll to zoom)
- [ ] AC-9.5: Smooth camera interpolation (no jitter at 60fps)

### US-10: Mouse Attraction Force

**As a** user interacting with the demo
**I want** particles to be attracted toward my mouse cursor in 3D space
**So that** I can sculpt and play with the particle field in real time

**Acceptance Criteria:**
- [ ] AC-10.1: Mouse position unprojected to a 3D world-space ray or point each frame
- [ ] AC-10.2: Attraction force applied in update kernel: F = strength / distance^2 (clamped)
- [ ] AC-10.3: Force falloff configurable (radius, strength)
- [ ] AC-10.4: Attraction force integrated into physics step with no additional kernel dispatch
- [ ] AC-10.5: Mouse interaction latency < 1 frame (force applied same frame as input received)
- [ ] AC-10.6: Visible particle response to mouse movement at 10M particles with no framerate degradation

### US-11: Progressive Scaling

**As a** developer testing performance
**I want** the particle pool to grow progressively from 1M to 10M
**So that** I can profile and verify performance at each scale without upfront 10M allocation

**Acceptance Criteria:**
- [ ] AC-11.1: Pool starts at 1M particles
- [ ] AC-11.2: Pool grows in increments (1M -> 2M -> 5M -> 10M) triggered by keyboard input or automatic schedule
- [ ] AC-11.3: Growth allocates new SoA buffers, copies existing data, updates dead list with new indices
- [ ] AC-11.4: Growth does not stall rendering (prepare new buffers, swap on next frame boundary)
- [ ] AC-11.5: Current particle count and FPS displayed in window title or on-screen HUD
- [ ] AC-11.6: Peak memory at 10M particles < 4 GB total GPU allocation

### US-12: Performance Target

**As a** developer evaluating the demo
**I want** 10M particles rendered at 60fps on M4 hardware
**So that** the demo proves Apple Silicon GPU capability at scale

**Acceptance Criteria:**
- [ ] AC-12.1: Sustained 60fps (16.6ms frame time) with 10M active particles on M4 (10-core GPU)
- [ ] AC-12.2: Compute pass (emission + grid build + physics update + compaction) < 10ms
- [ ] AC-12.3: Render pass (instanced draw) < 5ms
- [ ] AC-12.4: CPU frame work (input handling + command buffer encoding) < 1ms
- [ ] AC-12.5: No frame drops during mouse interaction at 10M particles
- [ ] AC-12.6: On M4 Pro (16-20 core GPU), frame time < 10ms at 10M particles

---

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | Create macOS window with CAMetalLayer using winit + raw-window-handle | P0 | Window opens, renders Metal content, resizes correctly |
| FR-2 | Initialize MTLDevice, MTLCommandQueue, compile .metallib, create all pipeline states | P0 | All pipeline states created without error; shader compilation succeeds |
| FR-3 | Allocate SoA particle buffers (position, velocity, lifetime, color, size) with StorageModeShared | P0 | Buffers created at 1M capacity; correct byte sizes per field |
| FR-4 | Initialize dead list with [0..pool_size-1], alive list empty, atomic counters zeroed | P0 | Dead list count == pool_size; alive count == 0 after init |
| FR-5 | Implement triple-buffer ring with dispatch_semaphore(3) for CPU-GPU sync | P0 | No CPU stalls waiting on GPU; 3 frames in flight |
| FR-6 | Emission compute kernel: atomic-decrement dead list, initialize particle fields with GPU PRNG | P0 | Particles appear on screen; dead list shrinks; no duplicate indices |
| FR-7 | Update compute kernel: integrate velocity, apply gravity, drag, boundary conditions | P0 | Particles move plausibly; respect bounds; respond to dt changes |
| FR-8 | Lifecycle compaction: cull dead particles, build next-frame alive list, update indirect args | P0 | Dead particles recycled; alive count matches indirect draw args |
| FR-9 | Ping-pong alive list swap each frame | P0 | No read-write hazards; alive list consistent between compute and render |
| FR-10 | Instanced draw call using indirect argument buffer (alive count from compaction) | P0 | Correct number of instances drawn; no CPU readback of alive count |
| FR-11 | Vertex shader: MVP transform per instance, read SoA position/color/size | P0 | Particles rendered at correct 3D positions with correct colors |
| FR-12 | Fragment shader: per-particle color, alpha blending | P0 | Smooth color/alpha rendering; correct blend mode |
| FR-13 | Perspective camera with view/projection matrix uniforms | P1 | 3D depth visible; correct projection at various FOV values |
| FR-14 | Camera orbit controls (mouse drag rotate, scroll zoom) | P1 | Smooth rotation/zoom; no gimbal lock |
| FR-15 | Mouse attraction force in update kernel (world-space unproject) | P1 | Particles visibly attracted to cursor; force scales with distance |
| FR-16 | Click-to-burst emission (10K-50K particles at cursor position) | P1 | Burst appears at click location; no frame drop from burst |
| FR-17 | 3D uniform grid density field (64^3) for particle-particle interactions | P1 | Grid populated each frame; density values readable by particles |
| FR-18 | Pressure gradient force from grid density (3x3x3 neighborhood reads) | P1 | Particles exhibit clustering/repulsion behavior; visually distinct from no-interaction |
| FR-19 | Progressive pool growth (1M -> 2M -> 5M -> 10M) with buffer reallocation | P1 | Growth succeeds without crash; rendering continues during growth |
| FR-20 | FPS / particle count HUD display (window title or overlay) | P1 | Current FPS and particle count visible at all times |
| FR-21 | Lifetime-based color/size interpolation (fade-out, shrink) | P1 | Visual distinction between young and old particles |
| FR-22 | Billboard quad geometry for particle instances (camera-facing) | P1 | Quads face camera from all angles; correct aspect ratio |
| FR-23 | FP16 (half) for color, lifetime, size fields to reduce bandwidth | P2 | Bandwidth reduced ~30% vs all-FP32; no visual artifacts |
| FR-24 | Function constants for kernel variants (emission vs. update vs. cull) | P2 | Single .metal source compiles to specialized pipelines |
| FR-25 | Depth sorting or OIT approximation for alpha-blended particles | P2 | Reduced alpha blending artifacts at high particle overlap |
| FR-26 | Configurable physics parameters via keyboard (gravity, drag, attraction) | P2 | Runtime parameter tuning without recompilation |
| FR-27 | Metal System Trace / GPU profiler integration hooks | P2 | Signpost labels on compute/render passes for Instruments profiling |

---

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Frame rate at 10M particles | FPS (sustained) | >= 60 fps on M4 (10-core GPU) |
| NFR-2 | Total frame time | Milliseconds | < 16.6ms (compute + render + CPU) |
| NFR-3 | Compute pass time | Milliseconds | < 10ms for emission + grid + update + compaction |
| NFR-4 | Render pass time | Milliseconds | < 5ms for instanced draw at 10M |
| NFR-5 | CPU frame overhead | Milliseconds | < 1ms for input + encoding + submission |
| NFR-6 | Memory at 10M particles | GB total GPU allocation | < 4 GB (fits 16 GB machines with headroom) |
| NFR-7 | Memory at 1M particles | MB total GPU allocation | < 200 MB initial |
| NFR-8 | Bandwidth consumption | GB/s sustained | < 100 GB/s at 10M @ 60fps (M4 limit) |
| NFR-9 | GPU occupancy | % of theoretical max | > 50% average across compute kernels |
| NFR-10 | Input latency | Frames | < 1 frame (force applied same frame as input) |
| NFR-11 | Startup time | Seconds | < 3s from launch to first rendered frame |
| NFR-12 | Particle struct bandwidth | Bytes per particle per frame | <= 64 bytes read + 64 bytes write (SoA, FP16 where possible) |
| NFR-13 | Grid interaction overhead | Milliseconds | < 3ms for grid clear + populate + density reads at 10M |
| NFR-14 | Pool growth latency | Frames of stutter | < 3 frames of dropped frames during reallocation |
| NFR-15 | Build target | Platform | macOS 14+ (Sonoma), Apple Silicon only |

---

## User Decisions

Captured from interview and research phases:

| Decision | User Choice | Rationale |
|----------|-------------|-----------|
| API | objc2-metal (Rust bindings) | Direct Metal control; matches architecture; active development |
| Hardware floor | M4 (10 cores, ~100 GB/s) | Personal hardware target; M4 Pro as stretch goal |
| Physics model | Full physics WITH particle interactions | User wants visible collective behavior, not just independent motion |
| Interaction method | Grid-based density approximation (NOT N-body) | Technical constraint: N*N infeasible at 10M; grid fits bandwidth budget |
| Render style | Instanced geometry (billboard quads) | Higher visual quality than point sprites |
| Camera | 3D perspective with orbit controls | Depth perception; spatial structure visibility |
| Priority | GPU performance first | Maximize particle count and framerate above visual polish |
| Success metric | 10M particles @ 60fps on M4 | Single quantitative target driving all tradeoffs |
| Pool strategy | Progressive growth (1M -> 10M) | Avoid 4 GB upfront allocation; enable per-scale profiling |
| Burst size | 10K-50K per click | Visually satisfying without exhausting pool |

---

## Glossary

| Term | Definition |
|------|------------|
| **SoA** | Structure of Arrays -- particle fields stored in separate contiguous arrays (position[], velocity[], ...) rather than interleaved structs. Enables partial reads and better cache utilization. |
| **AoS** | Array of Structures -- interleaved layout (particle[i].position, particle[i].velocity, ...). Not used in this project due to bandwidth waste on partial reads. |
| **Dead List** | GPU buffer of indices for inactive/available particle slots. Emission pops from it; death pushes to it. |
| **Alive List** | GPU buffer of indices for active particles. Rebuilt each frame via stream compaction. |
| **Ping-Pong** | Double-buffering pattern where two buffers alternate roles (read vs write) each frame to avoid read-write hazards. |
| **Stream Compaction** | GPU algorithm that filters a sparse array into a dense output (removes dead particles from alive list). |
| **Indirect Draw/Dispatch** | Metal feature allowing GPU to specify draw/dispatch arguments (instance count, threadgroup count) without CPU readback. |
| **ICB** | Indirect Command Buffer -- Metal 3+ feature for GPU-encoded command sequences. |
| **Triple Buffering** | Three pre-allocated buffer slots in a ring, synchronized via dispatch_semaphore(3). Allows CPU and GPU to overlap across 3 frames. |
| **Threadgroup** | Metal execution unit: a block of threads (typically 256-512) that share threadgroup memory and synchronize via barriers. |
| **SIMD Group** | 32 threads executing in lockstep on Apple Silicon. Supports shuffle, reduction, and ballot intrinsics. |
| **Occupancy** | Percentage of GPU's maximum concurrent threads that are active. Higher occupancy hides memory latency. |
| **FP16 / half** | 16-bit floating point. Uses 2x fewer registers than FP32; Apple Family 9 can dual-issue FP16+FP32 simultaneously. |
| **TBDR** | Tile-Based Deferred Rendering -- Apple GPU architecture that processes geometry per-tile, reducing bandwidth for opaque overdraw. |
| **Density Field** | 3D grid where each cell stores a scalar (particle count or accumulated mass). Used for approximate particle-particle interactions without per-pair computation. |
| **MTLBuffer** | Metal GPU buffer object. With StorageModeShared on Apple Silicon, accessible by both CPU and GPU with hardware coherence. |
| **Function Constants** | Metal compile-time specialization values that produce optimized kernel variants without source duplication. |
| **CAMetalLayer** | Core Animation layer that provides Metal-compatible drawables for on-screen rendering. |
| **OIT** | Order-Independent Transparency -- technique for correctly blending transparent geometry without depth sorting. |

---

## Out of Scope

- **Networking / multiplayer** -- single-machine demo only
- **Cross-platform** -- macOS Apple Silicon only; no iOS, Windows, Linux, or Intel Mac
- **Physically accurate SPH / fluid simulation** -- grid density approximation only; not targeting fluid fidelity
- **True N-body particle interactions** -- O(N^2) infeasible at 10M; grid-based approximation is the scope
- **Audio** -- no sound effects or music
- **GUI / parameter editor** -- keyboard shortcuts for parameter tuning; no ImGui or similar
- **Persistent state / save-load** -- no serialization of particle state
- **Texture-mapped particles** -- solid color with alpha; no texture atlases
- **Post-processing effects** -- no bloom, tone mapping, or screen-space effects
- **Metal 4 API** -- target Metal 3 / Family 9; Metal 4 bindings in objc2-metal are unverified
- **Multi-GPU** -- single GPU execution only
- **Automated benchmarking suite** -- manual profiling with Instruments; no CI perf regression

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Rust** | stable (1.75+) | Host language |
| **objc2** | 0.6 | Objective-C runtime bindings |
| **objc2-metal** | 0.3 (features = ["all"]) | Metal API Rust bindings |
| **objc2-foundation** | 0.3 | Foundation framework bindings |
| **winit** | 0.30 | Cross-platform windowing and input |
| **raw-window-handle** | 0.6 | Bridge winit to Metal's CAMetalLayer |
| **Metal SDK** | macOS 14+ (Sonoma) | GPU framework (system-provided) |
| **Xcode Command Line Tools** | 15+ | Metal shader compiler (xcrun metal) |
| **Apple Silicon** | M4 family (Family 9) | Target hardware |
| **glam** (optional) | 0.29+ | Fast math library for CPU-side matrix/vector ops |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Bandwidth ceiling at 10M on M4** -- 76.8 GB/s sustained leaves only 23% headroom on 100 GB/s bus | Medium | High | SoA + FP16 reduces per-particle bytes; progressive scaling validates at each tier; profile with Metal System Trace |
| **objc2-metal API gaps** -- some Metal 3 features (ICBs, indirect dispatch args) may lack Rust bindings | Medium | Medium | Validate binding coverage early in implementation; fall back to raw objc2 calls if needed |
| **Atomic contention on dead/alive lists** -- 10M particles doing atomic appends serializes at scale | Medium | High | Threadgroup-local pre-aggregation: each threadgroup collects dead indices locally, then one atomic per threadgroup instead of per particle |
| **Grid density atomic contention** -- 10M particles atomically incrementing 262K grid cells | Medium | Medium | Use threadgroup histograms merged with global atomics (two-phase pattern from KB ID: histogram template) |
| **Thermal throttling on fanless Macs** -- sustained GPU load at 10M causes clock reduction | High | Medium | Accept graceful degradation; document M4 MacBook Air as "best effort"; M4 Pro as primary target for sustained 60fps |
| **Alpha blending overdraw** -- 10M semi-transparent particles cause excessive fragment work in TBDR | Medium | Medium | Reduce particle alpha at high density; use discard for fully transparent; consider OIT as P2 stretch |
| **Pool growth stutter** -- buffer reallocation at runtime causes multi-frame stall | Low | Medium | Double-buffer growth: allocate new pool while old pool still rendering; swap atomically on frame boundary |
| **Shader register pressure** -- complex update kernel (physics + grid read + forces) exceeds 104 registers, halving occupancy | Medium | High | Profile register usage early; split update into sub-passes if needed; use FP16 aggressively to halve register consumption |
| **winit + Metal integration instability** -- CAMetalLayer setup via raw-window-handle may have edge cases | Low | Medium | Validate window creation as first milestone; reference working examples from objc2-metal repository |

---

## Unresolved Questions

- **Grid resolution**: 64^3 cells assumed optimal; may need 128^3 for 10M particles spread over large volume. Profile to decide.
- **Burst particle velocity**: "Higher than continuous" is qualitative. Exact multiplier TBD during implementation.
- **Particle geometry**: Billboard quad vs. triangle-pair vs. low-poly icosphere. Quad is baseline; revisit based on vertex throughput at 10M.
- **Depth sorting strategy**: Full sort at 10M is expensive (~50ms). Approximate back-to-front via grid cells, or skip sorting and accept artifacts? Deferred to P2.
- **Camera orbit projection plane**: Mouse attraction force requires unprojecting cursor to 3D. Use depth plane at attraction center, or ray-cast to bounding volume?

---

## Success Criteria

1. **10M particles rendered at sustained 60fps** on M4 (10-core GPU, ~100 GB/s bandwidth)
2. **Mouse attraction visually responsive** with < 1 frame latency at 10M particles
3. **Click burst emits 10K+ particles** at cursor position without frame drops
4. **Grid-based density interactions** produce visible collective behavior (clustering/repulsion)
5. **Total GPU memory < 4 GB** at 10M particles on 16 GB machine
6. **Progressive scaling** from 1M to 10M verifiable with on-screen counter

---

## Next Steps

1. Review and approve requirements
2. Generate architecture design (system components, buffer layouts, kernel signatures, frame loop)
3. Define Metal shader interface contracts (MSL struct definitions, kernel entry points)
4. Create implementation task breakdown with milestones
5. Begin P0 implementation: window + Metal init + emission + basic rendering
