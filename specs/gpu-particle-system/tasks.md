---
spec: gpu-particle-system
phase: tasks
total_tasks: 42
created: 2026-02-09
---

# Tasks: GPU Particle System

## Phase 1: Make It Work (POC)

Focus: Get particles on screen. Prove the 4-kernel compute pipeline + indirect instanced draw works end-to-end. Skip tests, accept hardcoded values, minimal error handling. Goal: 100K+ particles moving at 60fps.

---

- [x] 1.1 Scaffold Rust project and Metal build pipeline
  - **Do**:
    1. Create `particle-system/` directory at project root
    2. Create `Cargo.toml` with dependencies: objc2 0.6, objc2-metal 0.3 (features=["all"]), objc2-foundation 0.3, winit 0.30, raw-window-handle 0.6, glam 0.29
    3. Create `build.rs` that compiles all `.metal` files in `shaders/` to `shaders.metallib` using `xcrun -sdk macosx metal` + `xcrun -sdk macosx metallib`
    4. Create `shaders/types.h` with shared struct definitions: `Uniforms` (view_matrix, projection_matrix, mouse_world_pos, dt, gravity, drag_coefficient, grid_bounds_min/max, frame_number, particle_size_scale, emission_count, pool_size), `DrawArgs` (vertexCount, instanceCount, vertexStart, baseInstance)
    5. Create placeholder `shaders/emission.metal` that includes `types.h` and has empty kernel stub
    6. Create `src/main.rs` with minimal `fn main()` that prints "particle-system"
    7. Create `src/types.rs` with Rust-side `Uniforms` struct matching MSL layout (use `#[repr(C)]`, glam types)
  - **Files**:
    - `particle-system/Cargo.toml`
    - `particle-system/build.rs`
    - `particle-system/shaders/types.h`
    - `particle-system/shaders/emission.metal`
    - `particle-system/src/main.rs`
    - `particle-system/src/types.rs`
  - **Done when**: `cargo build` succeeds; `shaders.metallib` produced in OUT_DIR
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -5 && ls target/debug/build/*/out/shaders.metallib`
  - **Commit**: `feat(particle): scaffold project with Metal shader build pipeline`
  - _Requirements: FR-2, AC-1.4, NFR-15_
  - _Design: Rust Host Structure, File Layout_

- [x] 1.2 Create window with CAMetalLayer and Metal device init
  - **Do**:
    1. In `src/main.rs`: create winit `EventLoop` and `Window` (1280x720, title "GPU Particles")
    2. Create `src/gpu.rs`: initialize `MTLCreateSystemDefaultDevice()`, create `MTLCommandQueue`
    3. Attach `CAMetalLayer` to window via `raw-window-handle` (get NSView from RawWindowHandle, set `layer` to CAMetalLayer with pixelFormat BGRA8Unorm, device assigned)
    4. Load `.metallib` from build output path using `device.newLibraryWithURL()`
    5. Implement basic event loop: handle `CloseRequested` to exit, `AboutToWait` to request redraw
    6. On redraw: get `nextDrawable()`, create command buffer, create render pass (clear to dark blue), encode empty render pass, present drawable, commit
    7. Use `objc2_foundation::MainThreadMarker` for safety
  - **Files**:
    - `particle-system/src/main.rs`
    - `particle-system/src/gpu.rs`
  - **Done when**: Window opens with dark blue clear color; closes cleanly on window close
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3 && timeout 5 cargo run 2>&1 || true`
  - **Commit**: `feat(particle): window creation with CAMetalLayer and Metal device`
  - _Requirements: FR-1, AC-1.1, AC-1.2, AC-1.3, AC-1.6_
  - _Design: CPU Host - Main Event Loop, Metal Device/Queue_

- [x] 1.3 Triple-buffer semaphore ring and frame timing
  - **Do**:
    1. Create `src/frame.rs`: implement `FrameRing` struct with `dispatch_semaphore_create(3)` via `objc2` dispatch APIs (or use `std::sync::Semaphore` with count 3)
    2. `FrameRing::acquire()` blocks on semaphore; `FrameRing::signal()` signals after GPU completion
    3. Register command buffer completed handler that signals semaphore
    4. Track frame index (0, 1, 2 ring)
    5. Compute `dt` as time between frames using `std::time::Instant`
    6. Update window title with FPS each second: "GPU Particles - {fps} FPS"
    7. Integrate into main event loop: acquire at frame start, signal in completion handler
  - **Files**:
    - `particle-system/src/frame.rs`
    - `particle-system/src/main.rs` (update)
  - **Done when**: Window shows FPS in title; no CPU stalls visible (steady ~60fps or vsync rate)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): triple-buffer semaphore ring with frame timing`
  - _Requirements: FR-5, AC-1.5, NFR-5_
  - _Design: Triple Buffering, Per-Frame Sequence_

- [x] 1.4 [VERIFY] Quality checkpoint: build and shader compilation
  - **Do**: Run cargo build and clippy; verify .metallib produced
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo clippy 2>&1 | tail -10 && cargo build 2>&1 | tail -5`
  - **Done when**: No clippy warnings, build succeeds
  - **Commit**: `chore(particle): pass quality checkpoint` (only if fixes needed)

- [x] 1.5 Allocate SoA particle buffers and free lists
  - **Do**:
    1. Create `src/buffers.rs`: implement `ParticlePool` struct
    2. Allocate SoA buffers at 1M capacity using `device.newBufferWithLength_options(size, MTLResourceStorageModeShared)`:
       - `positions`: 1M x 12B (float3) = 12 MB
       - `velocities`: 1M x 12B (float3) = 12 MB
       - `lifetimes`: 1M x 4B (half2) = 4 MB
       - `colors`: 1M x 8B (half4) = 8 MB
       - `sizes`: 1M x 4B (half padded) = 4 MB
    3. Allocate dead list: 16B (atomic counter, padded) + 1M x 4B (indices) = ~4 MB
    4. Allocate alive list A and B: same layout as dead list, 2 x ~4 MB
    5. Allocate indirect args buffer: 32 bytes (MTLDrawPrimitivesIndirectArguments)
    6. Allocate uniforms buffer: 256B (padded Uniforms struct)
    7. Initialize dead list on CPU: set counter to pool_size, fill indices [0..pool_size-1]
    8. Initialize alive lists: counter = 0, indices zeroed
    9. Initialize indirect args: vertexCount=4, instanceCount=0, vertexStart=0, baseInstance=0
    10. Log total allocated memory; verify < 200 MB
  - **Files**:
    - `particle-system/src/buffers.rs`
    - `particle-system/src/types.rs` (add buffer-related types)
  - **Done when**: All buffers allocated; dead list initialized with 1M indices; total memory logged < 200 MB
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): SoA buffer allocation with dead/alive lists`
  - _Requirements: FR-3, FR-4, AC-2.1 through AC-2.8, NFR-7_
  - _Design: Buffer Layout, Particle Data (SoA Layout), Free/Alive Lists_

- [x] 1.6 GPU PRNG and emission compute kernel
  - **Do**:
    1. Create `shaders/prng.metal`: implement `pcg_hash(uint seed)` and `rand_float(uint seed)` -> [0,1) using PCG hash variant
    2. Create `shaders/emission.metal`: implement `emission_kernel` matching design signature
       - Each thread: atomic_fetch_sub on dead list counter; check >= 0
       - Read particle index from dead_list_indices[counter_value]
       - Initialize position: emitter center (0,0,0) + random offset in sphere (radius 0.5)
       - Initialize velocity: random direction * speed (1.0-3.0 range)
       - Initialize lifetime: half2(0.0, random 1.0-5.0)
       - Initialize color: random hue (HSV->RGB), alpha=1.0
       - Initialize size: random 0.01-0.05
       - Atomic increment alive list counter; write index to alive list
    3. In `src/gpu.rs`: create MTLComputePipelineState for emission kernel
    4. In `src/frame.rs`: encode emission dispatch in compute command encoder
       - Set emission_count in uniforms (hardcode 10000/frame for POC)
       - Dispatch ceil(emission_count / 256) threadgroups of 256
    5. Reset alive list counter to 0 at frame start (CPU write, or small clear kernel)
  - **Files**:
    - `particle-system/shaders/prng.metal`
    - `particle-system/shaders/emission.metal`
    - `particle-system/shaders/types.h` (update with emission params)
    - `particle-system/src/gpu.rs` (pipeline creation)
    - `particle-system/src/frame.rs` (dispatch encoding)
  - **Done when**: Emission kernel compiles into .metallib; dispatches without GPU error; dead list counter decrements
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -5`
  - **Commit**: `feat(particle): emission kernel with GPU PRNG and dead list allocation`
  - _Requirements: FR-6, AC-3.1 through AC-3.6_
  - _Design: Emission Kernel, GPU PRNG_

- [x] 1.7 Basic vertex + fragment shaders and render pipeline
  - **Do**:
    1. Create `shaders/render.metal`:
       - `vertex_main`: read particle index from alive list via instance_id, read position/color/size from SoA, apply view*projection, emit billboard quad (4 vertices: -0.5,-0.5 / +0.5,-0.5 / -0.5,+0.5 / +0.5,+0.5 as triangle strip), scale by particle size
       - `fragment_main`: output particle color (pass-through from vertex)
    2. In `src/gpu.rs`: create MTLRenderPipelineState with vertex/fragment functions
       - Set pixelFormat BGRA8Unorm
       - Enable alpha blending: source=sourceAlpha, dest=oneMinusSourceAlpha
       - Create depth stencil state (depth write off for transparent particles)
    3. In `src/frame.rs`: after compute pass, encode render pass
       - Set render pipeline state
       - Bind alive list, positions, colors, sizes, uniforms buffers
       - Draw with indirect args buffer (MTLDrawPrimitivesIndirectArguments)
       - Use `drawPrimitives:indirectBuffer:indirectBufferOffset:` (triangle strip, 4 vertices per instance)
    4. Set up basic uniforms: identity view matrix, simple perspective projection (60 deg FOV, 1280/720 aspect, 0.1-100 near/far), camera at (0,0,5) looking at origin
  - **Files**:
    - `particle-system/shaders/render.metal`
    - `particle-system/src/gpu.rs` (render pipeline)
    - `particle-system/src/frame.rs` (render encoding)
    - `particle-system/src/types.rs` (uniform values)
  - **Done when**: Particles appear on screen as colored quads; visible when emission is running
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): vertex/fragment shaders with indirect instanced draw`
  - _Requirements: FR-10, FR-11, FR-12, AC-8.1 through AC-8.5_
  - _Design: Vertex Shader, Fragment Shader, Render Shaders_

- [x] 1.8 [VERIFY] Quality checkpoint: build + shader compilation
  - **Do**: Build entire project, verify all .metal files compile into .metallib without errors
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo clippy 2>&1 | tail -10 && cargo build 2>&1 | tail -5`
  - **Done when**: Clean build, no warnings
  - **Commit**: `chore(particle): pass quality checkpoint` (only if fixes needed)

- [x] 1.9 Physics update kernel (gravity, drag, lifetime, death)
  - **Do**:
    1. Create `shaders/update.metal`: implement `update_physics_kernel` matching design signature
       - Thread reads index from alive_list_a[tid] (guard: tid < alive_count)
       - Read position, velocity, lifetime from SoA
       - Apply gravity: velocity += gravity * dt (gravity = (0, -9.8, 0))
       - Apply drag: velocity *= (1.0 - drag * dt) (drag = 0.1)
       - Semi-implicit Euler: position += velocity * dt
       - Boundary soft-bounce: if |position.xyz| > 10.0, reflect velocity component, dampen 0.5
       - Update lifetime: age += dt
       - If age >= maxAge: atomic_fetch_add on dead list counter, write index to dead_list; skip alive write
       - Else: atomic_fetch_add on alive_list_b counter, write index to alive_list_b
    2. Skip grid density reads for POC (hardcode pressure = 0)
    3. Skip mouse attraction for POC
    4. In `src/gpu.rs`: create compute pipeline for update kernel
    5. In `src/frame.rs`: dispatch update after emission
       - Threadgroups = ceil(alive_count / 256); but alive_count is on GPU...
       - POC shortcut: dispatch ceil(pool_size / 256) threadgroups; guard in shader
  - **Files**:
    - `particle-system/shaders/update.metal`
    - `particle-system/src/gpu.rs` (pipeline)
    - `particle-system/src/frame.rs` (dispatch)
  - **Done when**: Particles fall downward with gravity; bounce at boundaries; die after lifetime expires
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): physics update kernel with gravity, drag, lifetime`
  - _Requirements: FR-7, AC-5.1 through AC-5.6, AC-7.1 through AC-7.3_
  - _Design: Physics Update Kernel_

- [x] 1.10 Alive list compaction and ping-pong swap
  - **Do**:
    1. Create `shaders/compact.metal`: implement simplified compaction
       - POC approach: use atomic append instead of prefix scan (simpler, works at 1M scale)
       - Thread reads alive_list_b[tid]; if valid (index written by update kernel), atomic_fetch_add on alive_list_a_counter, write to alive_list_a
       - Final thread (or dedicated kernel): write alive_list_a_counter value to indirect_args.instanceCount
       - Also write alive_list_a_counter / 256 to dispatch args (for next frame's update threadgroup count)
    2. Alternative simpler POC: update kernel directly builds alive_list_b with atomic append; at frame end, swap buffer pointers (A<->B) on CPU side
    3. In `src/frame.rs`: dispatch compaction after update; swap ping-pong buffers
    4. Write indirect draw args: vertexCount=4, instanceCount=alive_count (written by GPU)
    5. Reset counters at frame start: alive_list_a counter = 0 (CPU write or small kernel)
  - **Files**:
    - `particle-system/shaders/compact.metal`
    - `particle-system/src/frame.rs` (dispatch + swap)
    - `particle-system/src/buffers.rs` (swap method)
  - **Done when**: Particles recycle correctly; dead particles respawn; particle count stabilizes; indirect draw uses correct alive count
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): alive list compaction with ping-pong swap`
  - _Requirements: FR-8, FR-9, AC-7.4 through AC-7.6, AC-8.3_
  - _Design: Compaction Kernel, Ping-Pong alive list_

- [x] 1.11 [VERIFY] Quality checkpoint: full build
  - **Do**: Build, clippy, verify all shaders compile
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo clippy 2>&1 | tail -10 && cargo build 2>&1 | tail -5`
  - **Done when**: Clean build
  - **Commit**: `chore(particle): pass quality checkpoint` (only if fixes needed)

- [x] 1.12 POC Checkpoint: particles on screen at 60fps
  - **Do**:
    1. Run the application; verify:
       - Window opens with dark background
       - Particles emit from center
       - Particles fall with gravity, bounce at boundaries
       - Particles die and recycle (pool doesn't exhaust)
       - FPS displayed in title bar (target: 60fps at 100K+ active particles)
       - Window closes cleanly
    2. If not working: debug by checking alive count (CPU readback in debug mode), checking buffer contents, verifying shader compilation
    3. Increase emission rate to test 100K+ active particles
    4. Verify via automated test: build succeeds, run for 3 seconds, check exit code
  - **Done when**: 100K+ particles visible, falling with gravity, recycling, at 60fps
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build --release 2>&1 | tail -3 && timeout 5 cargo run --release 2>&1; echo "Exit: $?"`
  - **Commit**: `feat(particle): complete POC — 100K particles at 60fps`
  - _Requirements: FR-1 through FR-12, NFR-1, NFR-2, NFR-11_
  - _Design: Implementation Steps Milestone 1-3_

---

## Phase 2: Full Feature (P1 Requirements)

After POC validated, add grid interactions, camera controls, mouse interaction, progressive scaling.

---

- [x] 2.1 Grid clear + populate kernels (density field)
  - **Do**:
    1. Allocate grid density buffer in `buffers.rs`: 64x64x64 x 4B = 1.05 MB (uint32 per cell)
    2. Create `shaders/grid.metal`:
       - `grid_clear_kernel`: zero all 262144 cells (simple: tid < cell_count -> store 0)
       - `grid_populate_kernel`: simplified single-phase for POC (skip two-phase histogram initially)
         - Each alive particle: compute cell index from position (quantize to 0-63 per axis using grid_bounds)
         - `atomic_fetch_add_explicit(&grid[cell_index], 1, memory_order_relaxed)`
    3. In `src/gpu.rs`: create pipelines for grid_clear and grid_populate
    4. In `src/frame.rs`: dispatch grid_clear (1024 threadgroups x 256) then grid_populate (alive_count / 256 threadgroups) between emission and physics
    5. Add grid_bounds_min/max to Uniforms (default: -10,-10,-10 to 10,10,10)
  - **Files**:
    - `particle-system/shaders/grid.metal`
    - `particle-system/src/buffers.rs` (grid buffer)
    - `particle-system/src/gpu.rs` (pipelines)
    - `particle-system/src/frame.rs` (dispatch)
    - `particle-system/shaders/types.h` (grid params in Uniforms)
  - **Done when**: Grid populated each frame; no GPU errors; frame time still reasonable
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): grid clear and populate kernels for density field`
  - _Requirements: FR-17, AC-6.1, AC-6.2_
  - _Design: Grid Clear Kernel, Grid Populate Kernel_

- [x] 2.2 Pressure gradient force from grid density
  - **Do**:
    1. In `shaders/update.metal`: add grid density reads
       - Compute cell index from particle position
       - Read 3x3x3 neighborhood (27 cells) density values
       - Compute approximate pressure gradient: sum of (neighbor_density - center_density) * direction_to_neighbor
       - Scale by interaction_strength uniform (default: 0.001)
       - Add pressure force to velocity
    2. Bind grid buffer in update kernel dispatch
    3. Add interaction_strength to Uniforms
  - **Files**:
    - `particle-system/shaders/update.metal` (update)
    - `particle-system/shaders/types.h` (interaction_strength)
    - `particle-system/src/frame.rs` (bind grid buffer)
    - `particle-system/src/types.rs` (update Uniforms)
  - **Done when**: Particles exhibit density-aware behavior (spreading from high-density regions)
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): pressure gradient force from grid density field`
  - _Requirements: FR-18, AC-6.3, AC-6.4, AC-6.6, AC-6.7_
  - _Design: Physics Update Kernel - grid density forces_

- [x] 2.3 [VERIFY] Quality checkpoint: build + run test
  - **Do**: Build, clippy, brief run test
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo clippy 2>&1 | tail -10 && cargo build --release 2>&1 | tail -5`
  - **Done when**: Clean build
  - **Commit**: `chore(particle): pass quality checkpoint` (only if fixes needed)

- [ ] 2.4 3D perspective camera with orbit controls
  - **Do**:
    1. Create `src/camera.rs`: `OrbitCamera` struct
       - Fields: azimuth, elevation, distance, target (center point), fov, aspect, near, far
       - `view_matrix()`: compute using glam (look_at_rh from orbit position to target)
       - `projection_matrix()`: glam perspective_rh (60 deg FOV default)
       - `orbit(delta_x, delta_y)`: rotate azimuth/elevation by mouse drag delta
       - `zoom(delta)`: change distance (clamp to 1.0-100.0)
       - Default: azimuth=0, elevation=0.3, distance=15, target=(0,0,0)
    2. Create `src/input.rs`: `InputState` struct
       - Track mouse position, left button held, right button held, scroll delta
       - On right-drag: camera orbit
       - On scroll: camera zoom
       - `cursor_position()`: returns (x, y) in window coordinates
    3. In `src/main.rs`: handle MouseInput, CursorMoved, MouseWheel events
    4. Upload camera matrices to Uniforms each frame
  - **Files**:
    - `particle-system/src/camera.rs`
    - `particle-system/src/input.rs`
    - `particle-system/src/main.rs` (input handling)
    - `particle-system/src/frame.rs` (uniforms upload)
    - `particle-system/src/types.rs` (ensure Uniforms has matrices)
  - **Done when**: Camera orbits around particle field; scroll zooms; 3D depth visible
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): 3D orbit camera with mouse drag and scroll zoom`
  - _Requirements: FR-13, FR-14, AC-9.1 through AC-9.5_
  - _Design: Camera, Input Manager_

- [ ] 2.5 Mouse attraction force in physics kernel
  - **Do**:
    1. In `src/input.rs`: add mouse world-space position computation
       - Unproject screen coords to 3D ray using inverse(projection * view)
       - Intersect with z=0 plane (or fixed depth plane at distance = camera.distance * 0.5)
       - Store as `mouse_world_pos` in InputState
    2. In `src/frame.rs`: upload mouse_world_pos, attraction_radius (5.0), attraction_strength (10.0) to Uniforms
    3. In `shaders/update.metal`: add mouse attraction force
       - `dir = mouse_world_pos - position`
       - `dist = length(dir)`
       - `force = attraction_strength / max(dist * dist, 0.01) * normalize(dir)`
       - Clamp force magnitude to prevent explosion
       - Apply only if dist < attraction_radius
       - `velocity += force * dt`
    4. Verify particles visibly move toward cursor
  - **Files**:
    - `particle-system/src/input.rs` (unproject)
    - `particle-system/src/frame.rs` (upload mouse pos)
    - `particle-system/shaders/update.metal` (attraction force)
    - `particle-system/src/types.rs` (Uniforms fields)
  - **Done when**: Particles attract toward mouse cursor in 3D; force falls off with distance
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): mouse attraction force with world-space unproject`
  - _Requirements: FR-15, AC-10.1 through AC-10.6_
  - _Design: Mouse attraction force, Input Manager_

- [ ] 2.6 Click-to-burst emission
  - **Do**:
    1. In `src/input.rs`: track left mouse click events; store burst_requested flag + burst_position
    2. In `src/frame.rs`: when burst_requested, increase emission_count for this frame (add burst_count 20000 to normal emission rate)
    3. In Uniforms: add `burst_position` (float3) and `burst_count` (uint)
    4. In `shaders/emission.metal`: if thread_id < burst_count, use burst_position as emitter center with 2x velocity multiplier; else use default emitter
    5. Clamp total emission to available dead list slots
    6. Clear burst_requested after frame
  - **Files**:
    - `particle-system/src/input.rs` (click tracking)
    - `particle-system/src/frame.rs` (burst emission)
    - `particle-system/shaders/emission.metal` (burst path)
    - `particle-system/shaders/types.h` (burst params)
    - `particle-system/src/types.rs` (burst params)
  - **Done when**: Left click emits burst of particles at cursor world position; burst particles have higher velocity
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): burst emission on mouse click`
  - _Requirements: FR-16, AC-4.1 through AC-4.6_
  - _Design: Emission Kernel - burst path_

- [ ] 2.7 [VERIFY] Quality checkpoint: build + clippy
  - **Do**: Build, clippy, verify all features compile
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo clippy 2>&1 | tail -10 && cargo build --release 2>&1 | tail -5`
  - **Done when**: Clean build, no clippy warnings
  - **Commit**: `chore(particle): pass quality checkpoint` (only if fixes needed)

- [ ] 2.8 Progressive pool scaling (1M to 10M)
  - **Do**:
    1. In `src/buffers.rs`: implement `ParticlePool::grow(new_size)`:
       - Allocate new SoA buffers at new_size
       - Copy existing particle data from old buffers to new (CPU-side memcpy via SharedStorage pointer access)
       - Extend dead list: add indices [old_size..new_size-1], update counter
       - Swap buffer references
       - Drop old buffers
    2. In `src/input.rs`: handle keyboard events (keys 1,2,5,0 for 1M/2M/5M/10M)
    3. In `src/main.rs`: on key press, call pool.grow() during frame boundary
    4. Update window title: "GPU Particles - {alive_count}/{pool_size} - {fps} FPS"
       - POC: read alive count from alive list counter on CPU (SharedStorage allows it)
    5. Verify growth doesn't crash; rendering continues
  - **Files**:
    - `particle-system/src/buffers.rs` (grow method)
    - `particle-system/src/input.rs` (keyboard)
    - `particle-system/src/main.rs` (key handling)
    - `particle-system/src/frame.rs` (title update)
  - **Done when**: Press 2 -> pool grows to 2M; press 5 -> 5M; press 0 -> 10M; rendering continues without crash
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build --release 2>&1 | tail -3`
  - **Commit**: `feat(particle): progressive pool scaling with keyboard triggers`
  - _Requirements: FR-19, AC-11.1 through AC-11.6_
  - _Design: Progressive Scaling Strategy_

- [ ] 2.9 Lifetime color/size interpolation and billboard quads
  - **Do**:
    1. In `shaders/update.metal`: add lifetime-based interpolation
       - `t = age / maxAge` (0.0 at birth, 1.0 at death)
       - Color alpha: `color.w = half(1.0 - t * t)` (quadratic fade-out)
       - Size: `size *= half(1.0 - t * 0.7)` (shrink to 30% at death)
    2. In `shaders/render.metal`: ensure billboard quads face camera
       - Extract camera right/up vectors from view matrix (columns 0 and 1)
       - Scale quad vertex by particle size: `world_pos += (right * v.x + up * v.y) * particle_size`
       - This makes quads always face the camera (true billboard)
    3. In `shaders/render.metal`: pass lifetime ratio to fragment for alpha
  - **Files**:
    - `particle-system/shaders/update.metal` (interpolation)
    - `particle-system/shaders/render.metal` (billboard, alpha)
  - **Done when**: Particles fade out and shrink as they age; quads face camera from all angles
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): lifetime interpolation with billboard quads`
  - _Requirements: FR-21, FR-22, AC-7.7, AC-8.1_
  - _Design: Vertex Shader billboard, lifetime interpolation_

- [ ] 2.10 FPS/particle HUD in window title
  - **Do**:
    1. In `src/frame.rs`: read alive list counter from GPU buffer (CPU read via SharedStorage pointer)
    2. Update window title every 0.5 seconds: "GPU Particles | {alive}K/{pool}M | {fps} FPS | {frame_ms:.1}ms"
    3. Format: show alive in K (thousands), pool in M (millions), fps as integer, frame time in ms
  - **Files**:
    - `particle-system/src/frame.rs` (readback + title)
    - `particle-system/src/main.rs` (set_title)
  - **Done when**: Window title shows live particle count, pool size, FPS, frame time
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): live HUD with particle count and FPS`
  - _Requirements: FR-20, AC-11.5_
  - _Design: HUD Display_

- [ ] 2.11 Two-phase grid populate (threadgroup histogram)
  - **Do**:
    1. Replace single-phase grid populate with two-phase pattern (from histogram.metal.tmpl):
       - Phase 1 (`grid_populate_local`): each threadgroup builds local histogram in threadgroup memory (262K bins is too large; use simplified: each threadgroup processes N particles, atomically increments global grid directly but with threadgroup-local batching)
       - Alternative: since 262K bins > threadgroup memory (~60KB), use simpler per-thread atomic on global grid, BUT batch writes with threadgroup coalescing
       - Actually: revert to direct atomic on global grid cells (single phase) for cells > 256, but use threadgroup_barrier to coalesce writes
    2. Profile: if single-phase atomic is fast enough at 1M-10M, keep it
    3. Only upgrade to two-phase if profiling shows contention
  - **Files**:
    - `particle-system/shaders/grid.metal` (optimize)
  - **Done when**: Grid populate runs without errors; no measurable regression
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): optimize grid populate kernel`
  - _Requirements: AC-6.5, NFR-13_
  - _Design: Grid Populate Kernel (Two-Phase)_

- [ ] 2.12 [VERIFY] Quality checkpoint: full feature build
  - **Do**: Build release, clippy, verify all features work together
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo clippy 2>&1 | tail -10 && cargo build --release 2>&1 | tail -5`
  - **Done when**: Clean release build, no warnings
  - **Commit**: `chore(particle): pass quality checkpoint` (only if fixes needed)

- [ ] 2.13 Phase 2 Checkpoint: full feature validation
  - **Do**:
    1. Build release and run
    2. Verify all P1 features work:
       - Camera orbits (right-drag) and zooms (scroll)
       - Mouse attraction (cursor influences particles)
       - Click burst (left click emits burst)
       - Grid density visible (particles spread/cluster)
       - Pool growth (keys 1/2/5/0)
       - HUD shows live stats
       - Lifetime fade/shrink visible
    3. Run at 1M for 30 seconds, verify stable 60fps
    4. Grow to 2M, verify stable
  - **Done when**: All P1 features work; 1M particles at 60fps
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build --release 2>&1 | tail -3 && timeout 5 cargo run --release 2>&1; echo "Exit: $?"`
  - **Commit**: `feat(particle): complete Phase 2 — all P1 features`
  - _Requirements: FR-13 through FR-22, US-4 through US-11_
  - _Design: Implementation Steps Milestone 4-7_

---

## Phase 3: Testing

Unit tests for Rust logic, integration tests for GPU kernels.

---

- [ ] 3.1 Unit tests: buffer allocation and types
  - **Do**:
    1. In `src/buffers.rs`: add `#[cfg(test)] mod tests`
    2. Test `ParticlePool::new(1_000_000)`:
       - Assert position buffer length == 1M * 12
       - Assert velocity buffer length == 1M * 12
       - Assert lifetime buffer length == 1M * 4
       - Assert color buffer length == 1M * 8
       - Assert size buffer length == 1M * 4
       - Assert dead list counter == 1M (read via CPU pointer)
       - Assert alive list counter == 0
    3. Test `ParticlePool::grow(2_000_000)`:
       - Assert new buffer lengths doubled
       - Assert dead list has new indices [1M..2M-1]
    4. Test Uniforms struct size == 256 (padded)
    5. Test DrawArgs struct matches MTLDrawPrimitivesIndirectArguments layout
  - **Files**:
    - `particle-system/src/buffers.rs` (tests module)
    - `particle-system/src/types.rs` (tests module)
  - **Done when**: `cargo test` passes with all buffer allocation assertions
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test -- --nocapture 2>&1 | tail -20`
  - **Commit**: `test(particle): unit tests for buffer allocation and types`
  - _Requirements: AC-2.1 through AC-2.8_
  - _Design: Buffer Layout_

- [ ] 3.2 Unit tests: camera math
  - **Do**:
    1. In `src/camera.rs`: add `#[cfg(test)] mod tests`
    2. Test view_matrix at default position: camera at (0, sin(0.3)*15, cos(0.3)*15) looking at origin
    3. Test projection_matrix: verify aspect ratio, FOV, near/far
    4. Test orbit: azimuth changes by delta_x * sensitivity
    5. Test zoom: distance clamps between 1.0 and 100.0
    6. Test mouse unproject: screen center -> ray toward origin
  - **Files**:
    - `particle-system/src/camera.rs` (tests module)
  - **Done when**: `cargo test camera` passes all assertions
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test camera -- --nocapture 2>&1 | tail -20`
  - **Commit**: `test(particle): unit tests for camera math`
  - _Requirements: AC-9.1 through AC-9.5_
  - _Design: Camera_

- [ ] 3.3 Unit tests: input state
  - **Do**:
    1. In `src/input.rs`: add `#[cfg(test)] mod tests`
    2. Test cursor position tracking
    3. Test click detection and burst flag
    4. Test keyboard state (pool growth triggers)
    5. Test drag accumulation for camera orbit
  - **Files**:
    - `particle-system/src/input.rs` (tests module)
  - **Done when**: `cargo test input` passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test input -- --nocapture 2>&1 | tail -20`
  - **Commit**: `test(particle): unit tests for input state tracking`
  - _Requirements: AC-4.1, AC-10.1_
  - _Design: Input Manager_

- [ ] 3.4 [VERIFY] Quality checkpoint: all tests pass
  - **Do**: Run full test suite, clippy, build
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test 2>&1 | tail -10 && cargo clippy 2>&1 | tail -5`
  - **Done when**: All tests pass, no clippy warnings
  - **Commit**: `chore(particle): pass quality checkpoint` (only if fixes needed)

- [ ] 3.5 GPU integration test: emission kernel
  - **Do**:
    1. Create `tests/gpu_integration.rs` (integration test file)
    2. Test emission_kernel:
       - Initialize device, queue, pipeline, buffers at 1000 particles
       - Set dead list to 1000, alive list to 0
       - Dispatch emission with emission_count=100
       - Readback alive list counter: assert == 100
       - Readback dead list counter: assert == 900
       - Readback positions: assert all non-zero (particles initialized)
    3. Use `waitUntilCompleted()` for synchronous test execution (acceptable in tests)
  - **Files**:
    - `particle-system/tests/gpu_integration.rs`
  - **Done when**: `cargo test gpu_integration::test_emission` passes
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test gpu_integration -- --nocapture 2>&1 | tail -20`
  - **Commit**: `test(particle): GPU integration test for emission kernel`
  - _Requirements: FR-6, AC-3.1_
  - _Design: Emission Kernel, Test Strategy_

- [ ] 3.6 GPU integration test: physics and compaction
  - **Do**:
    1. In `tests/gpu_integration.rs`: add test_physics
       - Emit 100 particles, run 1 physics step with dt=0.016
       - Readback positions: verify y decreased (gravity applied)
       - Verify alive count after update (should be ~100, none dead yet at age 0.016s)
    2. Add test_compaction:
       - Emit 100 particles, set all lifetimes to age=maxAge (force death)
       - Run update + compaction
       - Verify alive count == 0; dead list count restored
    3. Add test_indirect_draw_args:
       - After compaction, read indirect args buffer
       - Verify instanceCount == alive_count; vertexCount == 4
  - **Files**:
    - `particle-system/tests/gpu_integration.rs` (add tests)
  - **Done when**: Physics and compaction integration tests pass
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test gpu_integration -- --nocapture 2>&1 | tail -20`
  - **Commit**: `test(particle): GPU integration tests for physics and compaction`
  - _Requirements: FR-7, FR-8, AC-5.1, AC-7.4_
  - _Design: Physics Update Kernel, Compaction Kernel, Test Strategy_

- [ ] 3.7 [VERIFY] Quality checkpoint: all tests pass (unit + integration)
  - **Do**: Full test suite, clippy, release build
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test 2>&1 | tail -15 && cargo clippy 2>&1 | tail -5 && cargo build --release 2>&1 | tail -3`
  - **Done when**: All tests pass, clean release build
  - **Commit**: `chore(particle): pass quality checkpoint` (only if fixes needed)

---

## Phase 4: Quality Gates

---

- [ ] 4.1 Configurable physics parameters via keyboard (P2)
  - **Do**:
    1. In `src/input.rs`: add keyboard mappings:
       - G/Shift+G: increase/decrease gravity magnitude
       - D/Shift+D: increase/decrease drag coefficient
       - A/Shift+A: increase/decrease mouse attraction strength
       - R: reset all parameters to defaults
       - E/Shift+E: increase/decrease emission rate
    2. In `src/frame.rs`: apply parameter changes to Uniforms before upload
    3. Display current parameter values in window title or log
  - **Files**:
    - `particle-system/src/input.rs` (keyboard params)
    - `particle-system/src/frame.rs` (param application)
  - **Done when**: Physics parameters adjustable at runtime via keyboard
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): runtime physics parameter tuning via keyboard`
  - _Requirements: FR-26_
  - _Design: Configurable physics parameters_

- [ ] 4.2 FP16 optimization for color/lifetime/size
  - **Do**:
    1. Verify shaders already use `half2` for lifetime, `half4` for color, `half` for size
    2. In `src/buffers.rs`: verify buffer sizes match FP16 field sizes (not FP32)
    3. In `shaders/update.metal`: ensure intermediate calculations use `half` where precision allows (color interpolation, size scaling)
    4. Profile: check if FP16 dual-issue is happening (compare FP32-only vs mixed precision frame times)
  - **Files**:
    - `particle-system/shaders/update.metal` (verify half usage)
    - `particle-system/shaders/render.metal` (verify half reads)
    - `particle-system/src/buffers.rs` (verify sizes)
  - **Done when**: All applicable fields use FP16; no visual artifacts
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `perf(particle): verify FP16 optimization for bandwidth reduction`
  - _Requirements: FR-23, NFR-8, NFR-12_
  - _Design: Mixed Precision, Technical Decisions_

- [ ] 4.3 [VERIFY] Quality checkpoint: build + all tests
  - **Do**: Run full quality suite
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo test 2>&1 | tail -10 && cargo clippy 2>&1 | tail -5 && cargo build --release 2>&1 | tail -3`
  - **Done when**: All tests pass, clean build
  - **Commit**: `chore(particle): pass quality checkpoint` (only if fixes needed)

- [ ] 4.4 Metal profiler signpost labels
  - **Do**:
    1. In `src/frame.rs`: add os_signpost labels around each compute dispatch and render pass
       - Use `objc2_foundation` or raw C calls to `os_signpost_interval_begin/end`
       - Labels: "Emission", "Grid Clear", "Grid Populate", "Physics Update", "Compaction", "Render"
    2. Alternative: use MTLCommandBuffer label property (simpler)
       - `cmd_buf.setLabel(ns_string!("Frame N"))`
       - Each compute encoder: `encoder.setLabel(ns_string!("Emission"))` etc.
    3. Verify labels visible in Instruments Metal System Trace
  - **Files**:
    - `particle-system/src/frame.rs` (labels)
  - **Done when**: All compute/render passes labeled in profiler
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build 2>&1 | tail -3`
  - **Commit**: `feat(particle): Metal profiler signpost labels for all passes`
  - _Requirements: FR-27_
  - _Design: Profiling Tools_

- [ ] 4.5 Code cleanup and documentation
  - **Do**:
    1. Add doc comments to all public functions and structs
    2. Add module-level documentation to each .rs file
    3. Remove dead code, unused imports
    4. Ensure consistent error handling pattern (unwrap with context messages, or proper Result propagation)
    5. Add MSL comments to each kernel explaining algorithm
    6. Run `cargo doc` to verify documentation builds
  - **Files**:
    - All `particle-system/src/*.rs` files
    - All `particle-system/shaders/*.metal` files
  - **Done when**: `cargo doc` succeeds; `cargo clippy` clean; all functions documented
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo doc 2>&1 | tail -5 && cargo clippy 2>&1 | tail -5`
  - **Commit**: `docs(particle): add documentation to all modules and shaders`
  - _Design: Code quality_

- [ ] 4.6 [VERIFY] Full local CI: clippy + test + build + doc
  - **Do**: Run complete local CI suite
  - **Verify**: All commands must pass:
    ```
    cd /Users/patrickkavanagh/gpu_kernel/particle-system && \
    cargo clippy -- -D warnings 2>&1 | tail -5 && \
    cargo test 2>&1 | tail -10 && \
    cargo build --release 2>&1 | tail -3 && \
    cargo doc 2>&1 | tail -3
    ```
  - **Done when**: All commands pass with zero errors/warnings
  - **Commit**: `chore(particle): pass full local CI` (if fixes needed)

- [ ] 4.7 Create PR and verify CI
  - **Do**:
    1. Verify current branch is a feature branch: `git branch --show-current`
    2. If on default branch, STOP and alert user
    3. Stage all particle-system files: `git add particle-system/`
    4. Push branch: `git push -u origin <branch-name>`
    5. Create PR: `gh pr create --title "feat: GPU particle system with Metal compute pipeline" --body "<summary>"`
    6. Wait for CI checks
  - **Verify**: `gh pr checks --watch` or `gh pr checks`
  - **Done when**: PR created; all CI checks green
  - **If CI fails**: Read failure details, fix locally, push, re-verify

- [ ] 4.8 [VERIFY] AC checklist
  - **Do**: Programmatically verify each acceptance criterion is satisfied:
    1. AC-1.1: Window creation code exists (`grep -r "WindowBuilder\|Window::new" src/main.rs`)
    2. AC-1.2: CAMetalLayer setup (`grep -r "CAMetalLayer\|BGRA8Unorm" src/`)
    3. AC-1.3: Pipeline creation (`grep -r "newComputePipelineState\|newRenderPipelineState" src/gpu.rs`)
    4. AC-1.4: .metallib loading (`grep -r "newLibraryWithURL\|metallib" src/gpu.rs build.rs`)
    5. AC-2.1-2.8: Buffer allocation tests pass (`cargo test buffers`)
    6. AC-3.1-3.6: Emission kernel exists (`grep -r "emission_kernel" shaders/emission.metal`)
    7. AC-5.1-5.6: Physics kernel exists (`grep -r "update_physics_kernel" shaders/update.metal`)
    8. AC-8.1-8.5: Render shaders exist (`grep -r "vertex_main\|fragment_main" shaders/render.metal`)
    9. AC-9.1-9.5: Camera tests pass (`cargo test camera`)
    10. AC-11.1-11.6: Pool growth code exists (`grep -r "grow" src/buffers.rs`)
  - **Verify**:
    ```
    cd /Users/patrickkavanagh/gpu_kernel/particle-system && \
    cargo test 2>&1 | tail -5 && \
    grep -l "emission_kernel" shaders/*.metal && \
    grep -l "update_physics_kernel" shaders/*.metal && \
    grep -l "vertex_main" shaders/*.metal && \
    grep -l "grid_clear_kernel\|grid_populate" shaders/*.metal
    ```
  - **Done when**: All acceptance criteria confirmed met via automated checks
  - **Commit**: None

---

## Phase 5: PR Lifecycle

Continuous PR validation until all completion criteria met.

---

- [ ] 5.1 Monitor CI and fix failures
  - **Do**:
    1. Check PR status: `gh pr checks`
    2. If any check fails, read logs: `gh pr checks --json name,state,output`
    3. Fix issues locally; push fixes
    4. Re-check: `gh pr checks`
    5. Repeat until all green
  - **Verify**: `gh pr checks` shows all passing
  - **Done when**: All CI checks green
  - **Commit**: `fix(particle): address CI failures` (if needed)

- [ ] 5.2 Address review comments
  - **Do**:
    1. Check for review comments: `gh pr view --comments`
    2. Address each comment with code changes
    3. Push fixes; re-run CI
  - **Verify**: `gh pr view --json reviewDecision` shows APPROVED or no blocking reviews
  - **Done when**: All review comments addressed; CI still green
  - **Commit**: `fix(particle): address review feedback` (if needed)

- [ ] 5.3 Final validation: run at 1M particles for 10 seconds
  - **Do**:
    1. Build release: `cargo build --release`
    2. Run for 10 seconds: `timeout 10 cargo run --release`
    3. Verify exit code 0 (clean termination via timeout, not crash)
    4. Verify no GPU errors in stderr
  - **Verify**: `cd /Users/patrickkavanagh/gpu_kernel/particle-system && cargo build --release 2>&1 | tail -3 && timeout 10 cargo run --release 2>&1; echo "Exit: $?"`
  - **Done when**: Application runs 10 seconds without crash; clean exit
  - **Commit**: None

---

## Notes

### POC Shortcuts Taken (Phase 1)
- Hardcoded emission rate (10K/frame) instead of configurable
- Hardcoded physics parameters (gravity -9.8, drag 0.1)
- Skip grid density (pressure force = 0)
- Skip mouse attraction
- Dispatch update kernel with pool_size threadgroups instead of alive_count (some wasted threads)
- Simple atomic append for compaction instead of prefix scan
- Identity view matrix instead of proper camera
- Fixed camera position instead of orbit controls

### Production TODOs (Cleaned Up in Phase 2+)
- Proper camera orbit controls (Phase 2.4)
- Mouse attraction force (Phase 2.5)
- Click burst emission (Phase 2.6)
- Progressive pool scaling (Phase 2.8)
- Grid density field (Phase 2.1-2.2)
- Lifetime interpolation (Phase 2.9)
- Two-phase grid populate optimization (Phase 2.11)
- FP16 bandwidth optimization (Phase 4.2)
- Profiler labels (Phase 4.4)

### Key Technical Risks
- **objc2-metal API coverage**: Some Metal APIs may need raw objc2 fallback calls
- **Atomic contention at 10M**: May need threadgroup pre-aggregation if single-phase atomics bottleneck
- **Register pressure**: Physics kernel with grid reads may exceed 104 registers; split if needed
- **Bandwidth at 10M**: 76.8 GB/s theoretical; grid reads push to ~111 GB/s peak (tight on M4)
- **winit + CAMetalLayer**: Raw window handle integration has edge cases; test early (Task 1.2)

### Verification Commands Summary
| Command | Purpose |
|---------|---------|
| `cargo build` | Compile Rust + Metal shaders |
| `cargo build --release` | Optimized build |
| `cargo test` | Run all unit + integration tests |
| `cargo clippy` | Lint check |
| `cargo doc` | Documentation build |
| `cargo run --release` | Run particle system |
| `timeout N cargo run --release` | Run for N seconds (auto-terminate) |
| `gh pr checks` | CI status |
