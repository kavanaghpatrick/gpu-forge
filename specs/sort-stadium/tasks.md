# Tasks: Sort Stadium

## Phase 1: Make It Work (POC)

Focus: Get a window with colorful noise on screen, then sort it to a rainbow gradient.

### Task 1: Create sort-demo project scaffold

**Description**: Create the `sort-demo/` crate at the repo root with Cargo.toml, build.rs (from metal-gpu-experiments with `-std=metal3.2`), and frame.rs (verbatim copy from particle-system). Create stub shader files so build.rs can compile them.

**Files**:
- `sort-demo/Cargo.toml` (create)
- `sort-demo/build.rs` (create)
- `sort-demo/src/frame.rs` (create — verbatim copy)
- `sort-demo/src/main.rs` (create — empty `fn main() {}` stub)
- `sort-demo/shaders/types.h` (create — stub with `#ifndef TYPES_H` guard only)
- `sort-demo/shaders/render.metal` (create — empty stub kernel)

**Do**:
1. Create `sort-demo/Cargo.toml`:
   ```toml
   [package]
   name = "sort-demo"
   version = "0.1.0"
   edition = "2021"

   [dependencies]
   objc2 = "0.6"
   objc2-metal = "0.3"
   objc2-foundation = "0.3"
   objc2-quartz-core = "0.3"
   objc2-core-foundation = "0.3"
   winit = "0.30"
   raw-window-handle = "0.6"
   block2 = "0.6"
   dispatch2 = "0.3"
   rand = "0.8"
   ```
2. Create `sort-demo/build.rs` — copy from `/Users/patrickkavanagh/gpu_kernel/metal-gpu-experiments/build.rs` verbatim. It already has `-std=metal3.2`, `-I shaders/`, release `-O2`, and `rerun-if-changed` for `types.h`.
3. Copy `/Users/patrickkavanagh/gpu_kernel/particle-system/src/frame.rs` verbatim to `sort-demo/src/frame.rs`. Zero modifications.
4. Create `sort-demo/shaders/types.h` with just the include guard:
   ```c
   #ifndef TYPES_H
   #define TYPES_H
   #endif
   ```
5. Create `sort-demo/shaders/render.metal` with a minimal placeholder kernel so the metallib links:
   ```metal
   #include <metal_stdlib>
   using namespace metal;
   kernel void placeholder(uint tid [[thread_position_in_grid]]) {}
   ```
6. Create `sort-demo/src/main.rs` with `mod frame; fn main() {}`.

**Acceptance Criteria**:
- [x] `cargo build --release -p sort-demo` compiles without errors
- [x] build.rs compiles shaders with `-std=metal3.2` flag
- [x] frame.rs compiles as part of the crate (even if unused)

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -5
```

**Commit**: `feat(sort-demo): scaffold project with Cargo.toml, build.rs, frame.rs`
_Requirements: FR-9, FR-10, NFR-4_
_Design: File Structure, frame.rs_

---

### Task 2: Create shaders — types.h, render.metal, visualize.metal

**Description**: Write the complete shader files for types.h (all struct definitions + FLAG_* constants), render.metal (fullscreen triangle vertex + fragment), and visualize.metal (gpu_random_fill + value_to_heatmap). No sort shaders yet.

**Files**:
- `sort-demo/shaders/types.h` (modify — replace stub)
- `sort-demo/shaders/render.metal` (modify — replace stub)
- `sort-demo/shaders/visualize.metal` (create)

**Do**:
1. Replace `sort-demo/shaders/types.h` with full content:
   ```c
   #ifndef TYPES_H
   #define TYPES_H

   // Decoupled lookback flags (required by exp16_partition)
   #define FLAG_NOT_READY  0u
   #define FLAG_AGGREGATE  1u
   #define FLAG_PREFIX     2u
   #define FLAG_SHIFT      30u
   #define VALUE_MASK      ((1u << FLAG_SHIFT) - 1u)

   // Demo visualization params
   struct DemoParams {
       uint element_count;
       uint texture_width;
       uint texture_height;
       uint max_value;   // 0xFFFFFFFF for u32
   };

   // exp17 structs
   struct Exp17Params {
       uint element_count;
       uint num_tiles;
       uint shift;
       uint pass;
   };

   struct BucketDesc {
       uint offset;
       uint count;
       uint tile_count;
       uint tile_base;
   };

   // exp16 struct
   struct Exp16Params {
       uint element_count;
       uint num_tiles;
       uint num_tgs;
       uint shift;
       uint pass;
   };

   #endif
   ```
2. Replace `sort-demo/shaders/render.metal` with fullscreen triangle vertex + fragment shader. Copy exactly from design.md Section "render.metal" (lines 693-729). The vertex shader generates 3 vertices from `vertex_id` covering NDC [-1,1]x[-1,1]. The fragment shader samples `heatmap` texture with a sampler.
3. Create `sort-demo/shaders/visualize.metal` with `gpu_random_fill` and `value_to_heatmap` kernels. Copy exactly from design.md Section "visualize.metal" (lines 599-649). The `gpu_random_fill` kernel uses PCG hash: `state = gid * 747796405u + 2891336453u`, `word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u`, `data[gid] = (word >> 22u) ^ word`. The `value_to_heatmap` kernel maps `float(data[gid]) / float(max_value)` to HSV rainbow with 5 sectors, writes `half4` to texture at `(gid % width, gid / width)`.

**Acceptance Criteria**:
- [ ] types.h has FLAG_NOT_READY, FLAG_AGGREGATE, FLAG_PREFIX, FLAG_SHIFT, VALUE_MASK defines
- [ ] types.h has DemoParams, Exp17Params, BucketDesc, Exp16Params structs
- [ ] render.metal has `fullscreen_vertex` and `fullscreen_fragment` functions
- [ ] visualize.metal has `gpu_random_fill` and `value_to_heatmap` kernels
- [ ] `cargo build --release -p sort-demo` compiles all shaders into metallib

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -5
```

**Commit**: `feat(sort-demo): add types.h, render.metal, visualize.metal shaders`
_Requirements: FR-1, FR-4, FR-6, FR-12_
_Design: types.h, visualize.metal, render.metal_

---

### Task 3: Create gpu.rs + main.rs — open window with solid color

**Description**: Implement gpu.rs (GpuState with device, queue, layer, library) and main.rs (App struct with winit ApplicationHandler). Window opens and renders a solid clear color via a render pass with no textures. Proves the Metal rendering pipeline works.

**Files**:
- `sort-demo/src/gpu.rs` (create)
- `sort-demo/src/main.rs` (modify — replace stub)

**Do**:
1. Create `sort-demo/src/gpu.rs` with `GpuState` struct. Follow particle-system/src/gpu.rs pattern:
   - `device: Retained<ProtocolObject<dyn MTLDevice>>` — `MTLCreateSystemDefaultDevice()`
   - `command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>` — `device.newCommandQueue()`
   - `layer: Retained<CAMetalLayer>` — `CAMetalLayer::new()`, `setDevice()`, `setPixelFormat(BGRA8Unorm)`, `setFramebufferOnly(true)`
   - `library: Retained<ProtocolObject<dyn MTLLibrary>>` — `find_metallib()` then `newLibraryWithFile_error()`
   - `pub unsafe fn attach_layer_to_view(&self, ns_view: NonNull<c_void>)` — `msg_send![view, setWantsLayer: true]`, `msg_send![view, setLayer: layer]`
   - `fn find_metallib() -> String` — search `target/release/build/sort-demo-*/out/shaders.metallib`
   - Add helper `pub fn make_compute_pipeline(&self, name: &str) -> Retained<ProtocolObject<dyn MTLComputePipelineState>>` — wraps `newFunctionWithName` + `newComputePipelineStateWithFunction_error` with `#[allow(deprecated)]`

2. Rewrite `sort-demo/src/main.rs`:
   - `mod frame; mod gpu;`
   - `use frame::FrameRing; use gpu::GpuState;`
   - `struct App { window: Option<Arc<Window>>, gpu: Option<GpuState>, frame_ring: FrameRing }`
   - `impl ApplicationHandler for App`:
     - `resumed()`: create window 1280x720 titled "Sort Stadium", init GpuState, attach layer to NSView, `setDrawableSize`
     - `window_event()`: handle `CloseRequested` (exit), `Resized` (update drawable size), `RedrawRequested` (call render)
     - `about_to_wait()`: `window.request_redraw()`
   - `fn render(&mut self)`:
     - `frame_ring.acquire()`
     - Get drawable from `layer.nextDrawable()` (return early if None)
     - Create command buffer
     - Create render pass descriptor with `colorAttachment[0]` = drawable texture, loadAction = Clear (blue color), storeAction = Store
     - Create render command encoder, immediately `endEncoding()`
     - `frame_ring.register_completion_handler(&cmd)`
     - `cmd.presentDrawable(...)`, `cmd.commit()`
     - `frame_ring.advance()`
   - `fn main()`: `EventLoop::new()`, `App::new()`, `event_loop.run_app(&mut app)`

**Acceptance Criteria**:
- [ ] `cargo build --release -p sort-demo` compiles
- [ ] Running the binary opens a window titled "Sort Stadium"
- [ ] Window shows a solid blue background (clear color)
- [ ] Window can be closed with the X button or Cmd+Q

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -3
```

**Commit**: `feat(sort-demo): gpu.rs + main.rs — window with Metal clear color`
_Requirements: FR-6, FR-9, NFR-6_
_Design: gpu.rs, main.rs_

---

### Task 4: Create vis.rs — heatmap texture + fullscreen triangle rendering

**Description**: Implement vis.rs (Visualization struct) with heatmap texture creation, compute pipeline for value_to_heatmap, render pipeline for fullscreen triangle, and sampler. Wire into main.rs render loop so it renders the heatmap texture to the drawable via the fullscreen triangle. Initially the texture is black (zeroed buffers).

**Files**:
- `sort-demo/src/vis.rs` (create)
- `sort-demo/src/main.rs` (modify — add `mod vis`, wire Visualization)

**Do**:
1. Create `sort-demo/src/vis.rs`:
   - `pub enum VisMode { Heatmap, Barchart }`
   - `pub fn texture_dim(element_count: usize) -> u32` — returns 1024/2048/4096/8192 based on thresholds
   - `#[repr(C)] pub struct DemoParams { element_count: u32, texture_width: u32, texture_height: u32, max_value: u32 }`
   - `pub struct Visualization`:
     - `pso_heatmap: Retained<ProtocolObject<dyn MTLComputePipelineState>>` — pipeline for `value_to_heatmap`
     - `render_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>` — for fullscreen triangle
     - `heatmap_texture: Retained<ProtocolObject<dyn MTLTexture>>` — created via `texture2DDescriptorWithPixelFormat_width_height_mipmapped(BGRA8Unorm, dim, dim, false)` with `MTLTextureUsage::ShaderRead | ShaderWrite` and `StorageMode::Private`
     - `sampler: Retained<ProtocolObject<dyn MTLSamplerState>>` — Linear min/mag, ClampToEdge S/T
     - `pub mode: VisMode`
     - `texture_dim: u32`
   - `pub fn new(device, library, element_count)` — create PSOs, texture, sampler, render pipeline
   - `pub fn resize_if_needed(&mut self, device, element_count)` — recreate texture if dim changes
   - `pub fn encode_visualize(&self, cmd, data_buffer, element_count)`:
     - Create compute encoder
     - Set `pso_heatmap` (or barchart based on mode — just heatmap for now)
     - `setBuffer` for data at index 0
     - `setBytes` for DemoParams at index 1 (element_count, dim, dim, 0xFFFFFFFF)
     - `setTexture` for heatmap_texture at index 0
     - Dispatch 1D: grid = `ceil(N / 256)` TGs, TG = 256
     - `endEncoding()`
   - `pub fn encode_render(&self, cmd, drawable)`:
     - Create `MTLRenderPassDescriptor`, set `colorAttachment[0]` to drawable texture, loadAction=Clear (black), storeAction=Store
     - Create render encoder from descriptor
     - Set `render_pipeline`
     - Set fragment texture at index 0 = `heatmap_texture`
     - Set fragment sampler at index 0 = `sampler`
     - `drawPrimitives(MTLPrimitiveType::Triangle, 0, 3)` — zero vertex buffers
     - `endEncoding()`
   - Render pipeline creation:
     - `MTLRenderPipelineDescriptor::new()`
     - `setVertexFunction` = `fullscreen_vertex`
     - `setFragmentFunction` = `fullscreen_fragment`
     - `colorAttachments[0].setPixelFormat(BGRA8Unorm)`, `setBlendingEnabled(false)`
     - `device.newRenderPipelineStateWithDescriptor_error()`

2. Modify `sort-demo/src/main.rs`:
   - Add `mod vis;` and `use vis::Visualization;`
   - Add `vis: Option<Visualization>` to App struct
   - In `resumed()`: create `Visualization::new(&gpu.device, &gpu.library, 16_000_000)` after GpuState init
   - In `render()`: replace the bare render pass with:
     ```
     vis.encode_visualize(&cmd, &some_buffer, n);  // skip for now, just render
     vis.encode_render(&cmd, &drawable);
     ```
   - For now, skip the visualize compute pass (no data buffer yet) — just call `encode_render` so the fullscreen triangle samples the (black/empty) texture.

**Acceptance Criteria**:
- [ ] `cargo build --release -p sort-demo` compiles
- [ ] Window opens showing a black screen (empty texture rendered via fullscreen triangle)
- [ ] Render pipeline uses fullscreen triangle with zero vertex buffers

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -3
```

**Commit**: `feat(sort-demo): vis.rs with heatmap texture + fullscreen triangle render`
_Requirements: FR-4, FR-6_
_Design: vis.rs, render.metal_

---

### Task 5: Wire gpu_random_fill — show random noise on screen

**Description**: Allocate buf_a (256MB for 64M u32) and wire the `gpu_random_fill` compute kernel to fill it with random data, then visualize it as a heatmap. The window should now show colorful random noise — proving the entire compute-write-texture + render pipeline works end-to-end.

**Files**:
- `sort-demo/src/main.rs` (modify — add buffer allocation + shuffle + visualize)
- `sort-demo/src/vis.rs` (modify if needed — ensure encode_visualize works)

**Do**:
1. In `sort-demo/src/main.rs`:
   - Add `pso_random_fill` compute pipeline (function name `gpu_random_fill` from library)
   - Allocate `buf_a`: `device.newBufferWithLength_options(64_000_000 * 4, MTLResourceOptions::StorageModeShared)` — 256 MB, pre-allocated for max 64M
   - Store `buf_a` and `pso_random_fill` in App struct (temporary — will move to SortEngine in Task 7)
   - Set initial `element_count = 16_000_000` and `scale_index = 2` (index into `[1M, 4M, 16M, 64M]`)
   - In `render()`, before `encode_visualize`:
     - Create compute encoder
     - Set `pso_random_fill` pipeline
     - `setBuffer` for `buf_a` at index 0
     - `setBytes` for DemoParams `{ element_count: 16_000_000, texture_width: 4096, texture_height: 4096, max_value: 0xFFFFFFFF }` at index 1
     - Dispatch 1D: grid = `ceil(16_000_000 / 256)` TGs, TG = 256
     - `endEncoding()`
   - Then call `vis.encode_visualize(&cmd, &buf_a, 16_000_000)` to write heatmap
   - Then call `vis.encode_render(&cmd, &drawable)` to render

   - NOTE: The random fill runs every frame for now (will be gated by Space key in Task 7). This means the noise pattern changes every frame — that's fine for POC, proves the pipeline works.

2. Ensure `vis.encode_visualize` properly binds `data_buffer` at index 0, DemoParams at index 1 (via setBytes), and `heatmap_texture` at texture index 0.

**Acceptance Criteria**:
- [ ] Window shows colorful random noise (HSV-mapped random uint32 values)
- [ ] Noise pattern changes every frame (random fill runs each frame)
- [ ] No crashes or visual artifacts
- [ ] AC-1.1: Window renders a texture with one pixel per element
- [ ] AC-1.3: Random data displays as color noise with no visible pattern

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -3
```

**Commit**: `feat(sort-demo): gpu_random_fill + heatmap visualization — colorful noise on screen`
_Requirements: FR-1, FR-4, AC-1.1, AC-1.3_
_Design: visualize.metal, Data Flow_

---

### Task 5.5: [VERIFY] Quality checkpoint

**Do**: Run quality commands.

**Verify**:
```bash
cargo clippy -p sort-demo -- -D warnings && cargo build --release -p sort-demo
```

**Done when**: Zero clippy warnings, release build succeeds.

**Commit**: `chore(sort-demo): pass quality checkpoint` (only if fixes needed)

---

## Phase 2: Sort Integration

### Task 6: Copy exp17 kernels into sort_exp17.metal

**Description**: Copy the 5 Investigation T kernels from `metal-gpu-experiments/shaders/exp17_hybrid.metal` into `sort-demo/shaders/sort_exp17.metal`. The file must be self-contained (include types.h, define its own constants, include all 5 kernel functions). No modifications to kernel logic — exact copies.

**Files**:
- `sort-demo/shaders/sort_exp17.metal` (create)

**Do**:
1. Create `sort-demo/shaders/sort_exp17.metal` with this structure:
   ```metal
   #include <metal_stdlib>
   using namespace metal;
   #include "types.h"

   // Constants from exp17_hybrid.metal lines 13-18
   #define EXP17_NUM_BINS  256u
   #define EXP17_TILE_SIZE 4096u
   #define EXP17_ELEMS     16u
   #define EXP17_THREADS   256u
   #define EXP17_NUM_SGS   8u
   #define EXP17_MAX_TPB   17u

   // [PASTE KERNEL 1 HERE]
   // [PASTE KERNEL 2 HERE]
   // [PASTE KERNEL 3 HERE]
   // [PASTE KERNEL 4 HERE]
   // [PASTE KERNEL 5 HERE]
   ```

2. Copy these 5 kernels exactly (function signature + body) from `/Users/patrickkavanagh/gpu_kernel/metal-gpu-experiments/shaders/exp17_hybrid.metal`:

   **Kernel 1: `exp17_msd_histogram`** (lines 64-119)
   - Signature: `kernel void exp17_msd_histogram(device const uint* src [[buffer(0)]], device atomic_uint* global_hist [[buffer(1)]], constant Exp17Params& params [[buffer(2)]], uint lid, uint gid, uint simd_id, uint simd_lane)`
   - Uses TG memory: `threadgroup atomic_uint sg_counts[EXP17_NUM_SGS * EXP17_NUM_BINS]` (8 KB)
   - Reads `src`, writes to `global_hist` via atomic add

   **Kernel 2: `exp17_msd_prep`** (lines 2287-2315)
   - Signature: `kernel void exp17_msd_prep(device const uint* global_hist [[buffer(0)]], device uint* counters [[buffer(1)]], device BucketDesc* bucket_descs [[buffer(2)]], constant uint& tile_size [[buffer(3)]], uint lid)`
   - Uses TG memory: `threadgroup uint prefix[EXP17_NUM_BINS]`, `threadgroup uint running_offset`
   - Thread 0 does serial prefix sum, all 256 write counters + bucket_descs

   **Kernel 3: `exp17_msd_atomic_scatter`** (lines 2186-2274)
   - Signature: `kernel void exp17_msd_atomic_scatter(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]], device atomic_uint* counters [[buffer(2)]], constant Exp17Params& params [[buffer(3)]], uint lid, uint gid, uint simd_lane, uint simd_id)`
   - Uses TG memory: `sg_hist_or_rank[EXP17_NUM_SGS * EXP17_NUM_BINS]` + `sg_prefix` + `tile_hist` + `tile_base` (18 KB)
   - Atomic scatter: reads `src`, writes `dst` at positions from atomic counters

   **Kernel 4: `exp17_inner_precompute_hists`** (lines 1613-1667)
   - Signature: `kernel void exp17_inner_precompute_hists(device const uint* src [[buffer(0)]], device uint* inner_hists [[buffer(1)]], device const BucketDesc* bucket_descs [[buffer(2)]], uint lid, uint gid, uint simd_lane, uint simd_id)`
   - Uses TG memory: `threadgroup atomic_uint sg_c[3 * EXP17_NUM_SGS * EXP17_NUM_BINS]` (24 KB)
   - Reads bucket data, computes 3-pass histograms per bucket

   **Kernel 5: `exp17_inner_fused_v3`** (lines 1830-1963)
   - Signature: `kernel void exp17_inner_fused_v3(device uint* buf_a [[buffer(0)]], device uint* buf_b [[buffer(1)]], device const BucketDesc* bucket_descs [[buffer(2)]], device const uint* inner_hists [[buffer(3)]], uint lid, uint gid, uint simd_lane, uint simd_id)`
   - Uses TG memory: ~19 KB (sg_ctr, sg_pfx, bkt_hist, glb_pfx, run_pfx, chk_tot)
   - 3-pass fused inner sort with buffer alternation (pass 0: b->a, 1: a->b, 2: b->a)

3. Do NOT include any other kernels from exp17_hybrid.metal (there are ~20 experimental kernels in the file — only these 5 are used by Investigation T).

4. Do NOT modify any kernel logic. The only changes allowed are: removing leading comments/headers between kernels that reference other investigations.

**Acceptance Criteria**:
- [ ] sort_exp17.metal contains exactly 5 kernel functions
- [ ] All 5 kernel names match: `exp17_msd_histogram`, `exp17_msd_prep`, `exp17_msd_atomic_scatter`, `exp17_inner_precompute_hists`, `exp17_inner_fused_v3`
- [ ] File includes `#include "types.h"` and all 6 `#define EXP17_*` constants
- [ ] Struct definitions `Exp17Params` and `BucketDesc` come from types.h (not redefined in this file)
- [ ] `cargo build --release -p sort-demo` compiles all shaders

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -5
```

**Commit**: `feat(sort-demo): copy exp17 Investigation T kernels (5 kernels)`
_Requirements: FR-2_
_Design: sort_exp17.metal_

---

### Task 7: Create sort.rs — SortEngine with exp17 dispatch

**Description**: Implement sort.rs with the SortEngine struct. Allocate all buffers for 64M. Implement `encode_shuffle()` (moves gpu_random_fill from main.rs into SortEngine) and `encode_exp17()` (5 dispatches in 1 encoder). Wire Space key in main.rs to trigger shuffle + sort + visualize. After pressing Space, window should transition from noise to smooth rainbow gradient.

**Files**:
- `sort-demo/src/sort.rs` (create)
- `sort-demo/src/main.rs` (modify — add sort integration, Space key, move buf_a)

**Do**:
1. Create `sort-demo/src/sort.rs`:
   - `#[derive(Clone, Copy, PartialEq)] pub enum SortAlgorithm { Hybrid, EightBit }`
   - `pub const SCALES: &[usize] = &[1_000_000, 4_000_000, 16_000_000, 64_000_000];`
   - `#[repr(C)] pub struct Exp17Params { pub element_count: u32, pub num_tiles: u32, pub shift: u32, pub pass: u32 }`
   - `#[repr(C)] pub struct Exp16Params { pub element_count: u32, pub num_tiles: u32, pub num_tgs: u32, pub shift: u32, pub pass: u32 }`
   - `#[repr(C)] pub struct BucketDesc { pub offset: u32, pub count: u32, pub tile_count: u32, pub tile_base: u32 }`
   - `pub struct SortEngine` with fields:
     - PSOs: `pso_random_fill`, `pso_msd_histogram`, `pso_msd_prep`, `pso_msd_scatter`, `pso_precompute`, `pso_fused_v3`
     - Buffers: `pub buf_a` (256 MB), `pub buf_b` (256 MB), `buf_msd_hist` (1 KB), `buf_counters_17` (1 KB), `buf_bucket_descs` (4 KB), `buf_inner_hists` (786 KB = 256*3*256*4)
     - State: `pub algorithm: SortAlgorithm`, `pub element_count: usize`, `pub last_sort_ms: f64`
   - `pub fn new(device, library)`:
     - Create all PSOs using `gpu.make_compute_pipeline()` or direct `newFunctionWithName`/`newComputePipelineStateWithFunction_error`
     - Allocate buffers with `device.newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)`
     - Buffer sizes: `buf_a` = 64M * 4 = 256 MB, `buf_b` = 64M * 4 = 256 MB, `buf_msd_hist` = 256 * 4 = 1024, `buf_counters_17` = 256 * 4 = 1024, `buf_bucket_descs` = 256 * 16 = 4096, `buf_inner_hists` = 256 * 3 * 256 * 4 = 786432
   - `pub fn encode_shuffle(&self, cmd)`:
     - Create compute encoder
     - Set `pso_random_fill`
     - Bind `buf_a` at buffer index 0
     - setBytes DemoParams `{ element_count, texture_width: texture_dim(element_count), texture_height: same, max_value: 0xFFFFFFFF }` at index 1
     - Dispatch 1D: grid = `ceil(element_count / 256)` TGs, TG = 256
     - endEncoding()
   - `pub fn encode_sort(&self, cmd)`:
     - `match self.algorithm { Hybrid => self.encode_exp17(cmd), EightBit => {} }`
   - `fn encode_exp17(&self, cmd)`:
     - **CPU-zero buf_msd_hist**: `unsafe { std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 256 * 4) }`
     - Create ONE compute encoder
     - Compute: `num_tiles = element_count.div_ceil(4096)`, `tile_size_u32: u32 = 4096`
     - **D1: exp17_msd_histogram**: set PSO, bind buf_a(0), buf_msd_hist(1), setBytes Exp17Params{element_count, num_tiles, shift:24, pass:0}(2), dispatch num_tiles TGs x 256
     - **D2: exp17_msd_prep**: set PSO, bind buf_msd_hist(0), buf_counters_17(1), buf_bucket_descs(2), setBytes tile_size_u32(3), dispatch 1 TG x 256
     - **D3: exp17_msd_atomic_scatter**: set PSO, bind buf_a(0), buf_b(1), buf_counters_17(2), setBytes Exp17Params(3), dispatch num_tiles TGs x 256
     - **D4: exp17_inner_precompute_hists**: set PSO, bind buf_b(0), buf_inner_hists(1), buf_bucket_descs(2), dispatch 256 TGs x 256
     - **D5: exp17_inner_fused_v3**: set PSO, bind buf_a(0), buf_b(1), buf_bucket_descs(2), buf_inner_hists(3), dispatch 256 TGs x 256
     - endEncoding()
   - `pub fn set_element_count(&mut self, n: usize)` — just sets `self.element_count = n`

2. Modify `sort-demo/src/main.rs`:
   - Add `mod sort;` and `use sort::{SortAlgorithm, SortEngine, SCALES};`
   - Replace buf_a / pso_random_fill with `sort_engine: Option<SortEngine>`
   - Add `scale_index: usize` (default 2 = 16M), `needs_shuffle: bool` (default false), `auto_sort: bool` (default false)
   - In `resumed()`: create `SortEngine::new(&gpu.device, &gpu.library)`
   - Handle keyboard input: Space key sets `needs_shuffle = true`
   - In `render()`:
     - If `needs_shuffle`:
       - `sort_engine.encode_shuffle(&cmd)`
       - `sort_engine.encode_sort(&cmd)`
       - `needs_shuffle = false`
     - `vis.encode_visualize(&cmd, &sort_engine.buf_a, sort_engine.element_count)`
     - `vis.encode_render(&cmd, &drawable)`
   - Remove per-frame random fill (was POC in Task 5)

3. First frame now shows all-black or all-same-color (zeroed buf_a). Press Space -> noise -> sorted rainbow.

**Acceptance Criteria**:
- [ ] `cargo build --release -p sort-demo` compiles
- [ ] Pressing Space shuffles data + sorts with exp17 -> rainbow gradient appears
- [ ] Pressing Space again -> new noise -> new rainbow (repeatable)
- [ ] AC-2.1: Space triggers shuffle + sort + re-visualize in single frame
- [ ] AC-1.2: Sorted data displays as smooth rainbow gradient

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -3
```

**Commit**: `feat(sort-demo): SortEngine with exp17 dispatch — Space key sorts to rainbow`
_Requirements: FR-1, FR-2, FR-8, AC-1.2, AC-2.1_
_Design: sort.rs, encode_exp17 dispatch sequence_

---

### Task 7.5: [VERIFY] Quality checkpoint

**Do**: Run quality commands.

**Verify**:
```bash
cargo clippy -p sort-demo -- -D warnings && cargo build --release -p sort-demo
```

**Done when**: Zero clippy warnings, release build succeeds.

**Commit**: `chore(sort-demo): pass quality checkpoint` (only if fixes needed)

---

### Task 8: Copy exp16 kernels into sort_exp16.metal

**Description**: Copy the 4 production kernels from `metal-gpu-experiments/shaders/exp16_8bit.metal` into `sort-demo/shaders/sort_exp16.metal`. Self-contained file with types.h include and EXP16_* constants.

**Files**:
- `sort-demo/shaders/sort_exp16.metal` (create)

**Do**:
1. Create `sort-demo/shaders/sort_exp16.metal` with this structure:
   ```metal
   #include <metal_stdlib>
   using namespace metal;
   #include "types.h"

   // Constants from exp16_8bit.metal lines 18-23
   #define EXP16_NUM_BINS   256u
   #define EXP16_NUM_SGS    8u
   #define EXP16_TILE_SIZE  4096u
   #define EXP16_ELEMS      16u
   #define EXP16_NUM_PASSES 4u
   #define EXP16_THREADS    256u

   // [PASTE KERNEL 1 HERE]
   // [PASTE KERNEL 2 HERE]
   // [PASTE KERNEL 3 HERE]
   // [PASTE KERNEL 4 HERE]
   ```

2. Copy these 4 kernels exactly from `/Users/patrickkavanagh/gpu_kernel/metal-gpu-experiments/shaders/exp16_8bit.metal`:

   **Kernel 1: `exp16_combined_histogram`** (lines 303-362)
   - Signature: `kernel void exp16_combined_histogram(device const uint* src [[buffer(0)]], device atomic_uint* global_hist [[buffer(1)]], constant Exp16Params& params [[buffer(2)]], uint lid, uint gid, uint simd_id, uint simd_lane)`
   - 4-pass histogram in one read. Uses TG memory: `threadgroup atomic_uint sg_counts[EXP16_NUM_SGS * EXP16_NUM_BINS]` (8 KB)

   **Kernel 2: `exp16_global_prefix`** (lines 370-391)
   - Signature: `kernel void exp16_global_prefix(device uint* global_hist [[buffer(0)]], uint lid, uint simd_lane, uint simd_id)`
   - 4 SGs do 4 passes in parallel. Each SG: 8 serial chunks of 32-bin SIMD prefix sum.

   **Kernel 3: `exp16_zero_status`** (lines 397-410)
   - Signature: `kernel void exp16_zero_status(device uint* tile_status [[buffer(0)]], device atomic_uint* counters [[buffer(1)]], constant Exp16Params& params [[buffer(2)]], uint tid)`
   - Zeros `tile_status[0..num_tiles*256]` and `counters[0]`.

   **Kernel 4: `exp16_partition`** (lines 433-576)
   - Signature: `kernel void exp16_partition(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]], device atomic_uint* tile_status [[buffer(2)]], device atomic_uint* counters [[buffer(3)]], device const uint* global_hist [[buffer(4)]], constant Exp16Params& params [[buffer(5)]], uint lid, uint gid, uint simd_lane, uint simd_id)`
   - Uses TG memory: 18 KB (sg_hist_or_rank, sg_prefix, tile_hist, exclusive_pfx)
   - REQUIRES Metal 3.2: uses `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)` at lines 512-513 and 550-551
   - Uses FLAG_NOT_READY, FLAG_AGGREGATE, FLAG_PREFIX, FLAG_SHIFT, VALUE_MASK from types.h

3. Do NOT copy diagnostic kernels (exp16_diag_copy, exp16_diag_scatter, etc.) or partition_v2/v3/v4 variants.

**Acceptance Criteria**:
- [ ] sort_exp16.metal contains exactly 4 kernel functions
- [ ] All 4 kernel names match: `exp16_combined_histogram`, `exp16_global_prefix`, `exp16_zero_status`, `exp16_partition`
- [ ] File includes `#include "types.h"` and all 6 `#define EXP16_*` constants
- [ ] Struct `Exp16Params` comes from types.h (not redefined)
- [ ] `cargo build --release -p sort-demo` compiles (shader compilation with `-std=metal3.2` handles `atomic_thread_fence`)

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -5
```

**Commit**: `feat(sort-demo): copy exp16 8-bit radix sort kernels (4 kernels)`
_Requirements: FR-3_
_Design: sort_exp16.metal_

---

### Task 9: Add exp16 dispatch to sort.rs + algorithm switching

**Description**: Add exp16 buffer allocation, PSO creation, and `encode_exp16()` dispatch to SortEngine. Add keyboard handling for key 1 (Hybrid) and key 2 (EightBit) to switch algorithms. Add sort timing (Instant wall clock) and display algorithm name in window title.

**Files**:
- `sort-demo/src/sort.rs` (modify — add exp16 fields + encode_exp16)
- `sort-demo/src/main.rs` (modify — add key 1/2 handling, window title with algorithm name)

**Do**:
1. In `sort-demo/src/sort.rs`:
   - Add exp16 PSO fields: `pso_combined_hist`, `pso_global_prefix`, `pso_zero_status`, `pso_partition`
   - Add exp16 buffer fields: `buf_global_hist` (4 * 256 * 4 = 4096 bytes), `buf_tile_status` (15625 * 256 * 4 = 16_000_000 bytes for 64M max), `buf_counters_16` (4 bytes)
   - In `new()`: create exp16 PSOs (`exp16_combined_histogram`, `exp16_global_prefix`, `exp16_zero_status`, `exp16_partition`) and allocate exp16 buffers
   - Implement `fn encode_exp16(&self, cmd)`:
     - **CPU-zero buf_global_hist**: `unsafe { std::ptr::write_bytes(buf_global_hist.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4) }`
     - Compute `num_tiles = element_count.div_ceil(4096)`
     - Compute `zero_tg_count = (num_tiles * 256).div_ceil(256)` (= num_tiles)
     - **Encoder 1: exp16_combined_histogram**: bind buf_a(0), buf_global_hist(1), setBytes Exp16Params{element_count, num_tiles, num_tiles, 0, 0}(2), dispatch num_tiles TGs x 256, endEncoding()
     - **Encoder 2: exp16_global_prefix**: bind buf_global_hist(0), dispatch 1 TG x 256, endEncoding()
     - **For pass in 0..4**:
       - `src = if pass % 2 == 0 { buf_a } else { buf_b }`
       - `dst = if pass % 2 == 0 { buf_b } else { buf_a }`
       - `shift = pass * 8`
       - `params = Exp16Params { element_count, num_tiles, num_tiles, shift, pass }`
       - **Encoder N: exp16_zero_status**: bind buf_tile_status(0), buf_counters_16(1), setBytes params(2), dispatch zero_tg_count TGs x 256, endEncoding()
       - **Encoder N+1: exp16_partition**: bind src(0), dst(1), buf_tile_status(2), buf_counters_16(3), buf_global_hist(4), setBytes params(5), dispatch num_tiles TGs x 256, endEncoding()
   - Update `encode_sort()` match arm: `EightBit => self.encode_exp16(cmd)`

2. In `sort-demo/src/main.rs`:
   - Add timing: wrap sort dispatches with `let sort_start = Instant::now()`, then after commit + waitUntilCompleted (actually after frame), compute `sort_engine.last_sort_ms`. But since we don't call waitUntilCompleted in the render loop (async), use a simpler approach: measure wall-clock around the encode calls and store it. The GPU time will be close to wall clock in single-frame mode.
   - Actually, for POC: measure `Instant::now()` around `encode_shuffle + encode_sort` and store as `last_sort_ms`. This measures encode time, not GPU time — close enough for title display.
   - Handle key 1 (`KeyCode::Digit1`): set `sort_engine.algorithm = SortAlgorithm::Hybrid`
   - Handle key 2 (`KeyCode::Digit2`): set `sort_engine.algorithm = SortAlgorithm::EightBit`
   - Update window title with algorithm name: `"Sort Stadium | Hybrid"` or `"Sort Stadium | 8-Bit"`

**Acceptance Criteria**:
- [ ] Key 1 selects Hybrid sort, key 2 selects 8-Bit sort
- [ ] Both algorithms produce correct sorted output (smooth rainbow gradient)
- [ ] Window title shows current algorithm name
- [ ] AC-3.1: Key 1 shows "Hybrid" in title
- [ ] AC-3.2: Key 2 shows "8-Bit" in title
- [ ] AC-3.3: Next Space press uses selected algorithm

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -3
```

**Commit**: `feat(sort-demo): exp16 dispatch + algorithm switching (1/2 keys)`
_Requirements: FR-3, AC-3.1, AC-3.2, AC-3.3_
_Design: sort.rs encode_exp16 dispatch sequence_

---

### Task 9.5: [VERIFY] Quality checkpoint

**Do**: Run quality commands.

**Verify**:
```bash
cargo clippy -p sort-demo -- -D warnings && cargo build --release -p sort-demo
```

**Done when**: Zero clippy warnings, release build succeeds.

**Commit**: `chore(sort-demo): pass quality checkpoint` (only if fixes needed)

---

## Phase 3: Features

### Task 10: Add element count scaling (Up/Down arrows) + texture resize

**Description**: Add Up/Down arrow key handling to cycle through element counts [1M, 4M, 16M, 64M]. When scale changes, update SortEngine element_count, resize heatmap texture in Visualization, and trigger a shuffle+sort so the new scale is immediately visible.

**Files**:
- `sort-demo/src/main.rs` (modify — Up/Down key handling, scale change logic)
- `sort-demo/src/vis.rs` (modify — ensure `resize_if_needed` handles triple-buffer drain if needed)

**Do**:
1. In `sort-demo/src/main.rs`:
   - Handle `KeyCode::ArrowUp`: `scale_index = (scale_index + 1).min(3)`, set `needs_shuffle = true`
   - Handle `KeyCode::ArrowDown`: `scale_index = scale_index.saturating_sub(1)`, set `needs_shuffle = true`
   - In `render()`, before any GPU encoding:
     - `let n = SCALES[scale_index]`
     - `sort_engine.set_element_count(n)`
     - `vis.resize_if_needed(&gpu.device, n)` — recreates texture if dim changed
   - Update window title with element count: format as `"1.0M"`, `"4.0M"`, `"16.0M"`, `"64.0M"`

2. In `sort-demo/src/vis.rs`:
   - Implement `resize_if_needed(&mut self, device, element_count)`:
     - Compute new `texture_dim(element_count)`
     - If different from `self.texture_dim`:
       - Recreate `heatmap_texture` with new dimensions
       - Update `self.texture_dim`
     - NOTE: For POC, skip triple-buffer drain. The old texture may still be referenced by in-flight frames. On Apple Silicon with unified memory this is unlikely to cause issues since the old texture isn't freed until its retain count drops. For production, add drain logic in Phase 2 refactoring.

**Acceptance Criteria**:
- [ ] Up arrow cycles 1M -> 4M -> 16M -> 64M
- [ ] Down arrow reverses
- [ ] Texture resizes correctly (1024^2, 2048^2, 4096^2, 8192^2)
- [ ] Window title shows current element count
- [ ] Sort still produces correct rainbow gradient at all scales
- [ ] AC-5.1, AC-5.2, AC-5.3

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -3
```

**Commit**: `feat(sort-demo): element count scaling with Up/Down arrows`
_Requirements: FR-8, AC-5.1, AC-5.2, AC-5.3_
_Design: vis.rs resize_if_needed_

---

### Task 11: Add CPU sort comparison (background thread, key 3)

**Description**: Add key 3 to trigger CPU std::sort on a background thread. Poll `JoinHandle::is_finished()` each frame. Show elapsed time ticking up in title while running, final time on completion.

**Files**:
- `sort-demo/src/main.rs` (modify — add CpuSortState, key 3 handler, poll logic, title update)

**Do**:
1. Add to App struct:
   ```rust
   struct CpuSortState {
       handle: std::thread::JoinHandle<f64>,  // returns elapsed seconds
       start: std::time::Instant,
       element_count: usize,
       result_ms: Option<f64>,
   }
   ```
   - `cpu_sort_handle: Option<CpuSortState>`

2. Handle key 3 (`KeyCode::Digit3`):
   - If `cpu_sort_handle` is None or previous is finished:
     - `let n = SCALES[scale_index]`
     - `let start = Instant::now()`
     - Spawn thread: `std::thread::spawn(move || { let mut data: Vec<u32> = (0..n).map(|_| rand::random()).collect(); data.sort(); start.elapsed().as_secs_f64() })`
     - Store `CpuSortState { handle, start, element_count: n, result_ms: None }`

3. In `render()`, poll CPU sort:
   - If `cpu_sort_handle.is_some()` and `result_ms.is_none()`:
     - If `handle.is_finished()`:
       - `result_ms = Some(start.elapsed().as_secs_f64() * 1000.0)` (ms)

4. Update window title:
   - While CPU sort running: `"CPU: {elapsed:.1}s..."` with `start.elapsed().as_secs_f64()`
   - When finished: `"CPU: {result_ms:.0}ms"`

**Acceptance Criteria**:
- [ ] Key 3 starts CPU sort on background thread
- [ ] UI remains responsive (60fps) during CPU sort
- [ ] Title shows elapsed time ticking up during sort
- [ ] Title shows final time on completion
- [ ] AC-4.1, AC-4.2, AC-4.3

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -3
```

**Commit**: `feat(sort-demo): CPU sort comparison on background thread (key 3)`
_Requirements: FR-7, AC-4.1, AC-4.2, AC-4.3_
_Design: main.rs CPU sort background thread_

---

### Task 12: Add bar chart view (Tab toggle)

**Description**: Add Tab key to toggle between heatmap and bar chart visualization modes. Implement `value_to_barchart` kernel in visualize.metal (already designed). Wire into vis.rs to use the correct compute kernel based on VisMode.

**Files**:
- `sort-demo/shaders/visualize.metal` (modify — add `value_to_barchart` kernel)
- `sort-demo/src/vis.rs` (modify — add pso_barchart, dispatch based on mode)
- `sort-demo/src/main.rs` (modify — Tab key handler)

**Do**:
1. Add `value_to_barchart` kernel to `sort-demo/shaders/visualize.metal`:
   - Copy from design.md lines 652-683
   - 1D dispatch over texture_width * texture_height pixels
   - Each pixel: sample element for column x, compute bar height, color below bar, black above
   - Use `gid % texture_width` for x, `gid / texture_width` for y

2. In `sort-demo/src/vis.rs`:
   - Add `pso_barchart: Retained<ProtocolObject<dyn MTLComputePipelineState>>`
   - In `new()`: create PSO for `value_to_barchart`
   - In `encode_visualize()`:
     - If `mode == VisMode::Heatmap`: dispatch `ceil(element_count / 256)` TGs (N threads)
     - If `mode == VisMode::Barchart`: dispatch `ceil(texture_dim * texture_dim / 256)` TGs (W*H threads)
     - Use the appropriate PSO based on mode

3. In `sort-demo/src/main.rs`:
   - Handle Tab key: `vis.mode = match vis.mode { Heatmap => Barchart, Barchart => Heatmap }`
   - View switch should take effect immediately on next frame (no re-sort needed)

**Acceptance Criteria**:
- [ ] Tab toggles between heatmap and bar chart views
- [ ] Sorted data: heatmap shows rainbow, bar chart shows ascending bars
- [ ] Unsorted data: heatmap shows noise, bar chart shows random heights
- [ ] View switch is instant (no re-sort)
- [ ] AC-6.1, AC-6.2, AC-6.3

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -3
```

**Commit**: `feat(sort-demo): bar chart view with Tab toggle`
_Requirements: FR-5, AC-6.1, AC-6.2, AC-6.3_
_Design: visualize.metal value_to_barchart_

---

### Task 13: Add auto-sort mode (S key) + HUD stats in window title

**Description**: Add S key to toggle auto-sort (shuffle + sort every frame). Add comprehensive window title with sort time (ms), throughput (Mkeys/s), FPS, element count, algorithm name. Update title periodically (~every 500ms via `frame_ring.should_update_fps()`).

**Files**:
- `sort-demo/src/main.rs` (modify — S key, auto_sort logic, full title format)

**Do**:
1. Handle S key (`KeyCode::KeyS`): toggle `auto_sort = !auto_sort`
2. In `render()`:
   - If `needs_shuffle || auto_sort`:
     - Record `let sort_start = Instant::now()`
     - `sort_engine.encode_shuffle(&cmd)`
     - `sort_engine.encode_sort(&cmd)`
     - `sort_engine.last_sort_ms = sort_start.elapsed().as_secs_f64() * 1000.0`
     - `needs_shuffle = false`
3. Update window title format:
   ```
   Sort Stadium | Hybrid | 16.0M | 2.90ms | 4861 Mk/s | 60 FPS
   Sort Stadium | Hybrid | 16.0M | 2.90ms | 4861 Mk/s | CPU: 823ms | Auto | 60 FPS
   ```
   - Algorithm: "Hybrid" or "8-Bit"
   - Element count: `"{:.1}M"` with `n as f64 / 1_000_000.0`
   - Sort time: `"{:.2}ms"` from `last_sort_ms`
   - Throughput: `"{:.0} Mk/s"` from `element_count as f64 / last_sort_ms / 1000.0`
   - "Auto" if auto_sort enabled
   - CPU time if available
   - FPS from `frame_ring.fps`
4. Update title when `frame_ring.should_update_fps()` returns true (approximately every 500ms)
5. Also update title immediately after each sort operation (so timing is visible right away)

**Acceptance Criteria**:
- [ ] S key toggles auto-sort on/off
- [ ] In auto-sort, each frame performs shuffle + sort + visualize + render
- [ ] Window title shows: algorithm, count, time, throughput, FPS
- [ ] Title shows "Auto" when auto-sort is enabled
- [ ] AC-7.1, AC-7.2, AC-7.3
- [ ] AC-8.1, AC-8.2, AC-8.3

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -3
```

**Commit**: `feat(sort-demo): auto-sort mode (S key) + live stats in window title`
_Requirements: FR-1, FR-2, FR-3, AC-7.1, AC-7.2, AC-7.3, AC-8.1, AC-8.2, AC-8.3_
_Design: main.rs render() flow, Window title format_

---

### Task 13.5: [VERIFY] Quality checkpoint

**Do**: Run quality commands.

**Verify**:
```bash
cargo clippy -p sort-demo -- -D warnings && cargo build --release -p sort-demo
```

**Done when**: Zero clippy warnings, release build succeeds.

**Commit**: `chore(sort-demo): pass quality checkpoint` (only if fixes needed)

---

## Phase 4: Polish + Quality Gates

### Task 14: Add autorelease pool per frame + Escape key

**Description**: Wrap the render loop body in an autorelease pool to prevent Metal object accumulation (~100GB/min without it). Add Escape key to quit.

**Files**:
- `sort-demo/src/main.rs` (modify — add autoreleasepool, Escape handler)

**Do**:
1. Wrap the body of `render()` in `objc2::rc::autoreleasepool(|_| { ... })` to prevent Objective-C object leaks
2. Handle Escape key (`KeyCode::Escape`): `event_loop.exit()`
3. Verify no memory growth by running auto-sort for several seconds

**Acceptance Criteria**:
- [ ] Render loop wrapped in autoreleasepool
- [ ] Escape key cleanly exits the application
- [ ] No observable memory growth during auto-sort

**Verify**:
```bash
cargo build --release -p sort-demo 2>&1 | tail -3
```

**Commit**: `fix(sort-demo): add autoreleasepool per frame, Escape key to quit`
_Design: Technical Decisions — autorelease strategy_

---

### Task 15: Unit tests for struct sizes

**Description**: Add unit tests verifying `#[repr(C)]` struct sizes match GPU expectations. This catches padding/alignment bugs.

**Files**:
- `sort-demo/src/sort.rs` (modify — add #[cfg(test)] module)
- `sort-demo/src/vis.rs` (modify — add #[cfg(test)] module)

**Do**:
1. In `sort-demo/src/sort.rs`, add tests:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       use std::mem::size_of;

       #[test]
       fn exp17_params_size() { assert_eq!(size_of::<Exp17Params>(), 16); }
       #[test]
       fn exp16_params_size() { assert_eq!(size_of::<Exp16Params>(), 20); }
       #[test]
       fn bucket_desc_size() { assert_eq!(size_of::<BucketDesc>(), 16); }
   }
   ```
2. In `sort-demo/src/vis.rs`, add tests:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       use std::mem::size_of;

       #[test]
       fn demo_params_size() { assert_eq!(size_of::<DemoParams>(), 16); }
       #[test]
       fn texture_dim_1m() { assert_eq!(texture_dim(1_000_000), 1024); }
       #[test]
       fn texture_dim_4m() { assert_eq!(texture_dim(4_000_000), 2048); }
       #[test]
       fn texture_dim_16m() { assert_eq!(texture_dim(16_000_000), 4096); }
       #[test]
       fn texture_dim_64m() { assert_eq!(texture_dim(64_000_000), 8192); }
   }
   ```

**Acceptance Criteria**:
- [ ] All struct size tests pass
- [ ] All texture_dim tests pass

**Verify**:
```bash
cargo test -p sort-demo 2>&1 | tail -10
```

**Commit**: `test(sort-demo): add unit tests for struct sizes and texture dimensions`
_Requirements: NFR-7_
_Design: Test Strategy_

---

### Task 15.5: [VERIFY] Full local CI

**Do**: Run complete local CI suite.

**Verify**:
```bash
cargo clippy -p sort-demo -- -D warnings && cargo test -p sort-demo && cargo build --release -p sort-demo
```

**Done when**: All commands pass with zero errors.

**Commit**: `chore(sort-demo): pass full local CI` (only if fixes needed)

---

### Task 16: Create PR and verify CI

**Do**:
1. Verify current branch is a feature branch: `git branch --show-current`
2. If on default branch, STOP and alert user
3. Stage all sort-demo files: `git add sort-demo/`
4. Push branch: `git push -u origin <branch-name>`
5. Create PR: `gh pr create --title "feat: Sort Stadium — real-time GPU radix sort visualization" --body "..."`

**Verify**:
```bash
gh pr checks --watch
```

**Done when**: All CI checks green, PR ready for review.

**Commit**: None (PR creation task).

---

## Phase 5: PR Lifecycle

### Task 17: Monitor CI and fix failures

**Do**:
1. Check CI status: `gh pr checks`
2. If any checks fail, read failure details
3. Fix issues locally
4. Push fixes: `git push`
5. Re-verify: `gh pr checks --watch`

**Verify**:
```bash
gh pr checks
```

**Done when**: All CI checks show passing.

**Commit**: `fix(sort-demo): address CI failures` (if needed)

---

### Task 18: Final AC verification

**Do**: Programmatically verify each acceptance criterion is satisfied:
1. AC-1.1: Window opens with texture rendering — verified by build success + render pipeline code
2. AC-1.2: Sorted data = rainbow gradient — verified by value_to_heatmap kernel + exp17/exp16 sort
3. AC-1.3: Unsorted data = noise — verified by gpu_random_fill kernel
4. AC-2.1: Space triggers shuffle+sort+visualize — verified by needs_shuffle flag in main.rs
5. AC-3.1/3.2/3.3: Algorithm switching — verified by key 1/2 handlers + SortAlgorithm enum
6. AC-4.1/4.2/4.3: CPU sort comparison — verified by CpuSortState + background thread
7. AC-5.1/5.2/5.3: Element count scaling — verified by Up/Down handlers + resize_if_needed
8. AC-6.1/6.2/6.3: Bar chart view — verified by Tab handler + value_to_barchart kernel
9. AC-7.1/7.2/7.3: Auto-sort mode — verified by S key handler + auto_sort flag
10. AC-8.1/8.2/8.3: Window title stats — verified by title format string

**Verify**:
```bash
cargo test -p sort-demo && cargo clippy -p sort-demo -- -D warnings && cargo build --release -p sort-demo
```

**Done when**: All tests pass, all AC items traceable to code.

**Commit**: None.

---

## Notes

### POC shortcuts taken (Phase 1):
- Sort timing via `Instant` wall clock around encode calls (not GPU start/end time)
- No triple-buffer drain on texture resize (relies on unified memory not crashing)
- First frame shows all-black or all-same-color (user must press Space)
- Random fill runs as separate encoder in encode_shuffle (could share encoder with sort)
- No proper GPU timing — encode time != GPU time, but close enough for title display

### Production TODOs (Phase 2 refactoring if needed):
- Add triple-buffer drain before texture resize (same pattern as particle-system pool grow)
- Use GPU completion handler to measure actual GPU sort time
- Consider fusing shuffle + sort into single encoder (saves ~0.8μs per encoder creation)
- Add StorageModePrivate for heatmap texture (bandwidth compression via KB #73)

### Critical implementation details:
- **build.rs MUST use metal-gpu-experiments template** (has `-std=metal3.2`) — particle-system build.rs lacks this flag and exp16_partition will fail to compile
- **buf_msd_hist must be CPU-zeroed to 0 before every exp17 sort** — `write_bytes(ptr, 0, 256*4)`
- **buf_global_hist must be CPU-zeroed before every exp16 sort** — `write_bytes(ptr, 0, 4*256*4)`
- **exp17 result in buf_a** — inner 3 passes alternate b->a, a->b, b->a (odd = back to buf_a)
- **exp16 result in buf_a** — 4 passes (even) ping-pong returns to starting buffer
- **exp16 src/dst ping-pong**: pass 0: a->b, pass 1: b->a, pass 2: a->b, pass 3: b->a
- **exp17_msd_prep buffer(3) is `constant uint& tile_size`** — pass as setBytes with u32 value 4096
- **types.h FLAG_* constants are required by exp16_partition** — without them the decoupled lookback will not compile
