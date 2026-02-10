//! GPU particle system entry point and main event loop.
//!
//! Orchestrates the per-frame GPU pipeline: emission -> grid clear -> grid populate ->
//! physics update -> compaction -> render. Handles window events, camera orbit, mouse
//! interaction, keyboard controls, and pool scaling.

mod buffers;
mod camera;
mod frame;
mod gpu;
mod input;
mod types;

use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLLoadAction, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLSize, MTLStoreAction,
};
use objc2_quartz_core::CAMetalDrawable;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::PhysicalKey;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use buffers::ParticlePool;
use camera::OrbitCamera;
use frame::{FrameRing, MAX_FRAMES_IN_FLIGHT};
use gpu::GpuState;
use input::InputState;
use types::Uniforms;

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    pool: Option<ParticlePool>,
    frame_ring: FrameRing,
    frame_number: u32,
    /// Ping-pong flag: false = A is read/B is write; true = B is read/A is write.
    ping_pong: bool,
    /// 3D orbit camera for perspective viewing
    camera: OrbitCamera,
    /// Mouse/keyboard input state tracker
    input: InputState,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            pool: None,
            frame_ring: FrameRing::new(),
            frame_number: 0,
            ping_pong: false,
            camera: OrbitCamera::default(),
            input: InputState::default(),
        }
    }

    fn render(&mut self) {
        if self.gpu.is_none() || self.pool.is_none() {
            return;
        }

        // Acquire a frame slot (blocks until GPU finishes previous frame)
        self.frame_ring.acquire();

        // --- Handle pending pool grow (drain all in-flight frames for safe reallocation) ---
        if let Some(new_size) = self.input.pending_grow.take() {
            // We already acquired 1 slot above. Drain the remaining in-flight frames
            // so no GPU work references the old buffers during reallocation.
            for _ in 0..(MAX_FRAMES_IN_FLIGHT - 1) {
                self.frame_ring.wait_one();
            }
            if let Some(pool) = &mut self.pool {
                pool.grow(new_size);
            }
            // Restore the extra semaphore slots we drained
            for _ in 0..(MAX_FRAMES_IN_FLIGHT - 1) {
                self.frame_ring.signal();
            }
        }

        // Borrow gpu and pool after potential grow
        let gpu = self.gpu.as_ref().unwrap();
        let pool = self.pool.as_ref().unwrap();

        // --- Ping-pong: determine read and write lists ---
        // read_list: contains last frame's survivors; emission appends new particles here
        // write_list: update kernel writes this frame's survivors here; render uses this
        let (read_list, write_list) = pool.get_ping_pong_lists(self.ping_pong);

        // --- Update uniforms (write to per-frame slot in uniform ring buffer) ---
        let uniform_offset = ParticlePool::uniforms_offset(self.frame_ring.frame_index());
        let base_emission: u32 = self.input.physics.emission_rate;
        let view_mat = self.camera.view_matrix();
        let proj_mat = self.camera.projection_matrix();

        // Compute mouse world position by unprojecting cursor to z=0 plane
        let (win_w, win_h) = self
            .window
            .as_ref()
            .map(|w| {
                let s = w.inner_size();
                (s.width, s.height)
            })
            .unwrap_or((1280, 720));
        let mouse_world_pos = input::unproject_cursor_to_world(
            self.input.cursor_x,
            self.input.cursor_y,
            win_w,
            win_h,
            &view_mat,
            &proj_mat,
        );

        // Determine burst parameters
        let burst_count: u32 = if self.input.burst_requested {
            // Set burst world position from the current mouse world pos
            self.input.burst_world_pos = mouse_world_pos;
            20000
        } else {
            0
        };

        // Write uniforms to the per-frame slot in the uniform ring buffer
        unsafe {
            let ring_base = pool.uniform_ring.contents().as_ptr() as *mut u8;
            let uniforms_ptr = ring_base.add(uniform_offset) as *mut Uniforms;
            (*uniforms_ptr).view_matrix = view_mat;
            (*uniforms_ptr).projection_matrix = proj_mat;
            (*uniforms_ptr).mouse_world_pos = mouse_world_pos;
            (*uniforms_ptr).dt = self.frame_ring.dt;
            (*uniforms_ptr).frame_number = self.frame_number;
            (*uniforms_ptr).base_emission_rate = base_emission;
            (*uniforms_ptr).pool_size = pool.pool_size as u32;
            // Set grid bounds for density field
            (*uniforms_ptr).grid_bounds_min = [-10.0, -10.0, -10.0];
            (*uniforms_ptr)._pad_grid_min = 0.0;
            (*uniforms_ptr).grid_bounds_max = [10.0, 10.0, 10.0];
            (*uniforms_ptr)._pad_grid_max = 0.0;
            // Physics parameters from runtime-tunable PhysicsParams
            (*uniforms_ptr).gravity = self.input.physics.gravity;
            (*uniforms_ptr).drag_coefficient = self.input.physics.drag_coefficient;
            // Pressure gradient interaction strength
            (*uniforms_ptr).interaction_strength = 0.001;
            // Mouse attraction parameters
            (*uniforms_ptr).mouse_attraction_radius = 5.0;
            (*uniforms_ptr).mouse_attraction_strength = self.input.physics.mouse_attraction_strength;
            // Burst emission parameters
            (*uniforms_ptr).burst_position = self.input.burst_world_pos;
            (*uniforms_ptr)._pad_burst = 0.0;
            (*uniforms_ptr).burst_count = burst_count;
        }

        // Clear burst_requested after uploading uniforms
        self.input.burst_requested = false;

        // Get the next drawable from the CAMetalLayer
        let drawable = match gpu.layer.nextDrawable() {
            Some(d) => d,
            None => {
                eprintln!("No drawable available");
                return;
            }
        };

        // Get the drawable's texture
        let texture = drawable.texture();

        // Create render pass descriptor with dark background
        let render_pass_desc = MTLRenderPassDescriptor::renderPassDescriptor();
        let color_attachment = unsafe {
            render_pass_desc.colorAttachments().objectAtIndexedSubscript(0)
        };
        color_attachment.setTexture(Some(&texture));
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setStoreAction(MTLStoreAction::Store);
        color_attachment.setClearColor(MTLClearColor {
            red: 0.02,
            green: 0.02,
            blue: 0.08,
            alpha: 1.0,
        });

        // Create command buffer
        let command_buffer = match gpu.command_queue.commandBuffer() {
            Some(cb) => cb,
            None => {
                eprintln!("Failed to create command buffer");
                return;
            }
        };
        command_buffer.setLabel(Some(ns_string!("Frame")));

        // Register completion handler that signals the semaphore
        self.frame_ring.register_completion_handler(&command_buffer);

        // --- Prepare dispatch compute pass ---
        // GPU-side computation of emission parameters and dispatch args.
        // Reads dead_count, computes emission_count, writes indirect dispatch args.
        // Also resets write_list counter to 0 (replaces CPU write_list reset).
        if let Some(prepare_encoder) = command_buffer.computeCommandEncoder() {
            prepare_encoder.setLabel(Some(ns_string!("Prepare Dispatch")));
            prepare_encoder.setComputePipelineState(&gpu.prepare_dispatch_pipeline);

            // buffer(0)=dead_list, buffer(1)=uniform_ring, buffer(2)=write_list,
            // buffer(3)=emission_dispatch_args, buffer(4)=gpu_emission_params
            unsafe {
                prepare_encoder.setBuffer_offset_atIndex(Some(&pool.dead_list), 0, 0);
                prepare_encoder.setBuffer_offset_atIndex(Some(&pool.uniform_ring), uniform_offset, 1);
                prepare_encoder.setBuffer_offset_atIndex(Some(write_list), 0, 2);
                prepare_encoder.setBuffer_offset_atIndex(Some(&pool.emission_dispatch_args), 0, 3);
                prepare_encoder.setBuffer_offset_atIndex(Some(&pool.gpu_emission_params), 0, 4);
            }

            // Single threadgroup of 32 threads (SIMD-aligned, only thread 0 does work)
            prepare_encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );

            prepare_encoder.endEncoding();
        }

        // --- Emission compute pass ---
        // Emission appends new particles to the READ list (alongside last frame's survivors).
        // Uses indirect dispatch: threadgroup count from GPU-computed emission_dispatch_args.
        if let Some(compute_encoder) = command_buffer.computeCommandEncoder() {
            compute_encoder.setLabel(Some(ns_string!("Emission")));
            compute_encoder.setComputePipelineState(&gpu.emission_pipeline);

            // buffer(0) = uniform_ring, buffer(1) = dead_list, buffer(2) = alive_list (read_list),
            // buffer(3) = positions, buffer(4) = velocities, buffer(5) = lifetimes,
            // buffer(6) = colors, buffer(7) = sizes, buffer(8) = gpu_emission_params
            unsafe {
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.uniform_ring), uniform_offset, 0);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.dead_list), 0, 1);
                compute_encoder.setBuffer_offset_atIndex(Some(read_list), 0, 2);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.positions), 0, 3);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.velocities), 0, 4);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.lifetimes), 0, 5);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.colors), 0, 6);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.sizes), 0, 7);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.gpu_emission_params), 0, 8);
            }

            // Indirect dispatch: threadgroup count from prepare_dispatch kernel output
            unsafe {
                compute_encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &pool.emission_dispatch_args,
                    0,
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }

            compute_encoder.endEncoding();
        }

        // --- Grid clear compute pass ---
        // Zero all 262144 cells in the grid density buffer.
        // Must run BEFORE grid_populate and update so update kernel reads fresh grid data.
        if let Some(grid_clear_encoder) = command_buffer.computeCommandEncoder() {
            grid_clear_encoder.setLabel(Some(ns_string!("Grid Clear")));
            grid_clear_encoder.setComputePipelineState(&gpu.grid_clear_pipeline);
            unsafe {
                grid_clear_encoder.setBuffer_offset_atIndex(Some(&pool.grid_density), 0, 0);
            }
            // 262144 cells / 256 threads per group = 1024 threadgroups
            grid_clear_encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1024, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            grid_clear_encoder.endEncoding();
        }

        // --- Grid populate compute pass ---
        // Each alive particle atomically increments its grid cell density.
        // Reads from read_list (last frame's survivors + this frame's new emissions).
        // Must run BEFORE update so the update kernel can read the density field.
        if let Some(grid_pop_encoder) = command_buffer.computeCommandEncoder() {
            grid_pop_encoder.setLabel(Some(ns_string!("Grid Populate")));
            grid_pop_encoder.setComputePipelineState(&gpu.grid_populate_pipeline);
            unsafe {
                grid_pop_encoder.setBuffer_offset_atIndex(Some(&pool.uniform_ring), uniform_offset, 0);
                grid_pop_encoder.setBuffer_offset_atIndex(Some(read_list), 0, 1);
                grid_pop_encoder.setBuffer_offset_atIndex(Some(&pool.positions), 0, 2);
                grid_pop_encoder.setBuffer_offset_atIndex(Some(&pool.grid_density), 0, 3);
            }
            // Indirect dispatch: threadgroup count from sync_indirect_args output (update_dispatch_args)
            unsafe {
                grid_pop_encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &pool.update_dispatch_args,
                    0,
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            grid_pop_encoder.endEncoding();
        }

        // --- Physics update compute pass ---
        // Reads from read_list (last frame's survivors + this frame's new emissions).
        // Reads grid_density for pressure gradient force.
        // Writes survivors to write_list, writes dead particles back to dead_list.
        if let Some(update_encoder) = command_buffer.computeCommandEncoder() {
            update_encoder.setLabel(Some(ns_string!("Physics Update")));
            update_encoder.setComputePipelineState(&gpu.update_pipeline);

            // buffer(0) = uniform_ring, buffer(1) = dead_list, buffer(2) = read_list (input),
            // buffer(3) = write_list (output), buffer(4-8) = SoA data, buffer(9) = grid_density
            unsafe {
                update_encoder.setBuffer_offset_atIndex(Some(&pool.uniform_ring), uniform_offset, 0);
                update_encoder.setBuffer_offset_atIndex(Some(&pool.dead_list), 0, 1);
                update_encoder.setBuffer_offset_atIndex(Some(read_list), 0, 2);
                update_encoder.setBuffer_offset_atIndex(Some(write_list), 0, 3);
                update_encoder.setBuffer_offset_atIndex(Some(&pool.positions), 0, 4);
                update_encoder.setBuffer_offset_atIndex(Some(&pool.velocities), 0, 5);
                update_encoder.setBuffer_offset_atIndex(Some(&pool.lifetimes), 0, 6);
                update_encoder.setBuffer_offset_atIndex(Some(&pool.colors), 0, 7);
                update_encoder.setBuffer_offset_atIndex(Some(&pool.sizes), 0, 8);
                update_encoder.setBuffer_offset_atIndex(Some(&pool.grid_density), 0, 9);
            }

            // Indirect dispatch: threadgroup count from sync_indirect_args output (update_dispatch_args)
            unsafe {
                update_encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &pool.update_dispatch_args,
                    0,
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }

            update_encoder.endEncoding();
        }

        // --- Sync alive count to indirect args (GPU compute, single thread) ---
        // Reads from write_list (update kernel output = this frame's survivors).
        if let Some(sync_encoder) = command_buffer.computeCommandEncoder() {
            sync_encoder.setLabel(Some(ns_string!("Compaction")));
            sync_encoder.setComputePipelineState(&gpu.sync_indirect_pipeline);
            unsafe {
                sync_encoder.setBuffer_offset_atIndex(Some(write_list), 0, 0);
                sync_encoder.setBuffer_offset_atIndex(Some(&pool.indirect_args), 0, 1);
                sync_encoder.setBuffer_offset_atIndex(Some(&pool.update_dispatch_args), 0, 2);
            }
            sync_encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 },
            );
            sync_encoder.endEncoding();
        }

        // --- Render pass: draw particle billboard quads ---
        let encoder = match command_buffer.renderCommandEncoderWithDescriptor(&render_pass_desc) {
            Some(enc) => enc,
            None => {
                eprintln!("Failed to create render command encoder");
                return;
            }
        };
        encoder.setLabel(Some(ns_string!("Render")));

        // Set render pipeline and depth stencil state
        encoder.setRenderPipelineState(&gpu.render_pipeline);
        encoder.setDepthStencilState(Some(&gpu.depth_stencil_state));

        // Bind vertex buffers: buffer(0) = write_list (survivors), rest = SoA + uniform_ring + lifetimes
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(write_list), 0, 0);
            encoder.setVertexBuffer_offset_atIndex(Some(&pool.positions), 0, 1);
            encoder.setVertexBuffer_offset_atIndex(Some(&pool.colors), 0, 2);
            encoder.setVertexBuffer_offset_atIndex(Some(&pool.sizes), 0, 3);
            encoder.setVertexBuffer_offset_atIndex(Some(&pool.uniform_ring), uniform_offset, 4);
            encoder.setVertexBuffer_offset_atIndex(Some(&pool.lifetimes), 0, 5);
        }

        // Indirect draw: triangle strip, 4 vertices per instance, instanceCount from indirect_args
        unsafe {
            encoder.drawPrimitives_indirectBuffer_indirectBufferOffset(
                MTLPrimitiveType::TriangleStrip,
                &pool.indirect_args,
                0,
            );
        }

        encoder.endEncoding();

        // Present drawable and commit
        command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
        command_buffer.commit();

        // Flip ping-pong: next frame swaps read/write roles
        self.ping_pong = !self.ping_pong;

        // Advance frame ring index and frame number
        self.frame_ring.advance();
        self.frame_number = self.frame_number.wrapping_add(1);

        // Update window title with FPS, alive count, and pool size approximately once per second
        if self.frame_ring.should_update_fps() {
            if let Some(window) = &self.window {
                // Read alive count from the write list (this frame's survivors)
                // Safe because GPU is done with this buffer (single buffering)
                let alive_count = pool.read_alive_count_from_indirect();
                let alive_k = alive_count as f64 / 1_000.0;
                let pool_m = pool.pool_size as f64 / 1_000_000.0;
                let fps = self.frame_ring.fps;
                let frame_ms = if fps > 0 { 1000.0 / fps as f64 } else { 0.0 };
                let physics_info = self.input.physics.summary();
                let title = format!(
                    "GPU Particles | {alive_k:.0}K/{pool_m:.0}M | {fps} FPS | {frame_ms:.1}ms | {physics_info}",
                );
                window.set_title(&title);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Create window
        let attrs = WindowAttributes::default()
            .with_title("GPU Particles")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        // Initialize GPU state
        let gpu = GpuState::new();

        // Attach CAMetalLayer to the window's NSView
        let handle = window.window_handle().expect("Failed to get window handle");
        match handle.as_raw() {
            RawWindowHandle::AppKit(appkit_handle) => {
                unsafe {
                    gpu.attach_layer_to_view(appkit_handle.ns_view);
                }

                // Set the drawable size to match the window's physical size
                let size = window.inner_size();
                gpu.layer.setDrawableSize(CGSize {
                    width: size.width as f64,
                    height: size.height as f64,
                });
            }
            _ => panic!("Unsupported platform - expected AppKit window handle"),
        }

        // Set camera aspect ratio from window size
        let size = window.inner_size();
        if size.height > 0 {
            self.camera.aspect = size.width as f32 / size.height as f32;
        }

        // Allocate particle buffers (1M particles)
        let pool = ParticlePool::new(&gpu.device, 1_000_000);
        self.pool = Some(pool);

        self.gpu = Some(gpu);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(PhysicalSize { width, height }) => {
                if let Some(gpu) = &self.gpu {
                    gpu.layer.setDrawableSize(CGSize {
                        width: width as f64,
                        height: height as f64,
                    });
                }
                // Update camera aspect ratio
                if height > 0 {
                    self.camera.aspect = width as f32 / height as f32;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((dx, dy)) = self.input.update_cursor(position.x, position.y) {
                    self.camera.orbit(dx, dy);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                match button {
                    MouseButton::Left => {
                        let was_held = self.input.left_held;
                        self.input.left_held = state == ElementState::Pressed;
                        // On initial press (not held before), request burst
                        if !was_held && state == ElementState::Pressed {
                            self.input.burst_requested = true;
                        }
                    }
                    MouseButton::Right => {
                        self.input.right_held = state == ElementState::Pressed;
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera.zoom(scroll_amount);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // Track shift modifier state
                if let PhysicalKey::Code(code) = event.physical_key {
                    if code == winit::keyboard::KeyCode::ShiftLeft
                        || code == winit::keyboard::KeyCode::ShiftRight
                    {
                        self.input.shift_held = event.state == ElementState::Pressed;
                    }
                }
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(key_code) = event.physical_key {
                        self.input.handle_key(key_code);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop
        .run_app(&mut app)
        .expect("Event loop error");
}
