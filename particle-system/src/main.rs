mod buffers;
mod frame;
mod gpu;
mod types;

use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_metal::{
    MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLLoadAction, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLSize, MTLStoreAction,
};
use objc2_quartz_core::CAMetalDrawable;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use buffers::ParticlePool;
use frame::FrameRing;
use gpu::GpuState;
use types::{CounterHeader, Uniforms};

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    pool: Option<ParticlePool>,
    frame_ring: FrameRing,
    frame_number: u32,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            pool: None,
            frame_ring: FrameRing::new(),
            frame_number: 0,
        }
    }

    fn render(&mut self) {
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };
        let pool = match &self.pool {
            Some(p) => p,
            None => return,
        };

        // Acquire a frame slot (blocks if all 3 are in use)
        self.frame_ring.acquire();

        // --- Reset alive list counter to 0 at frame start (CPU write) ---
        unsafe {
            let alive_ptr = pool.alive_list_a.contents().as_ptr() as *mut CounterHeader;
            (*alive_ptr).count = 0;
        }

        // --- Update uniforms ---
        let emission_count: u32 = 10000;
        let (view_mat, proj_mat) = types::default_camera_matrices();
        unsafe {
            let uniforms_ptr = pool.uniforms.contents().as_ptr() as *mut Uniforms;
            (*uniforms_ptr).view_matrix = view_mat;
            (*uniforms_ptr).projection_matrix = proj_mat;
            (*uniforms_ptr).dt = self.frame_ring.dt;
            (*uniforms_ptr).frame_number = self.frame_number;
            (*uniforms_ptr).emission_count = emission_count;
            (*uniforms_ptr).pool_size = pool.pool_size as u32;
        }

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

        // Create render pass descriptor with dark blue clear color
        let render_pass_desc = MTLRenderPassDescriptor::renderPassDescriptor();
        let color_attachment = unsafe {
            render_pass_desc.colorAttachments().objectAtIndexedSubscript(0)
        };
        color_attachment.setTexture(Some(&texture));
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setStoreAction(MTLStoreAction::Store);
        color_attachment.setClearColor(MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.2,
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

        // Register completion handler that signals the semaphore
        self.frame_ring.register_completion_handler(&command_buffer);

        // --- Emission compute pass ---
        if let Some(compute_encoder) = command_buffer.computeCommandEncoder() {
            compute_encoder.setComputePipelineState(&gpu.emission_pipeline);

            // Set buffers matching emission_kernel signature:
            // buffer(0) = uniforms, buffer(1) = dead_list, buffer(2) = alive_list,
            // buffer(3) = positions, buffer(4) = velocities, buffer(5) = lifetimes,
            // buffer(6) = colors, buffer(7) = sizes
            unsafe {
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.uniforms), 0, 0);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.dead_list), 0, 1);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.alive_list_a), 0, 2);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.positions), 0, 3);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.velocities), 0, 4);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.lifetimes), 0, 5);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.colors), 0, 6);
                compute_encoder.setBuffer_offset_atIndex(Some(&pool.sizes), 0, 7);
            }

            // Dispatch ceil(emission_count / 256) threadgroups of 256
            let threadgroup_size = 256usize;
            let threadgroup_count = (emission_count as usize).div_ceil(threadgroup_size);
            compute_encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: threadgroup_count, height: 1, depth: 1 },
                MTLSize { width: threadgroup_size, height: 1, depth: 1 },
            );

            compute_encoder.endEncoding();
        }

        // --- Sync alive count to indirect args (GPU compute, single thread) ---
        if let Some(sync_encoder) = command_buffer.computeCommandEncoder() {
            sync_encoder.setComputePipelineState(&gpu.sync_indirect_pipeline);
            unsafe {
                sync_encoder.setBuffer_offset_atIndex(Some(&pool.alive_list_a), 0, 0);
                sync_encoder.setBuffer_offset_atIndex(Some(&pool.indirect_args), 0, 1);
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

        // Set render pipeline and depth stencil state
        encoder.setRenderPipelineState(&gpu.render_pipeline);
        encoder.setDepthStencilState(Some(&gpu.depth_stencil_state));

        // Bind vertex buffers matching render.metal vertex_main signature:
        // buffer(0) = alive_list, buffer(1) = positions, buffer(2) = colors,
        // buffer(3) = sizes, buffer(4) = uniforms
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(&pool.alive_list_a), 0, 0);
            encoder.setVertexBuffer_offset_atIndex(Some(&pool.positions), 0, 1);
            encoder.setVertexBuffer_offset_atIndex(Some(&pool.colors), 0, 2);
            encoder.setVertexBuffer_offset_atIndex(Some(&pool.sizes), 0, 3);
            encoder.setVertexBuffer_offset_atIndex(Some(&pool.uniforms), 0, 4);
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

        // Advance frame ring index and frame number
        self.frame_ring.advance();
        self.frame_number = self.frame_number.wrapping_add(1);

        // Update window title with FPS approximately once per second
        if self.frame_ring.should_update_fps() {
            if let Some(window) = &self.window {
                window.set_title(&format!("GPU Particles - {} FPS", self.frame_ring.fps));
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
                println!("Window close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(PhysicalSize { width, height }) => {
                if let Some(gpu) = &self.gpu {
                    gpu.layer.setDrawableSize(CGSize {
                        width: width as f64,
                        height: height as f64,
                    });
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
