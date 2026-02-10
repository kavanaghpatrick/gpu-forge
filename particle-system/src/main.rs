mod buffers;
mod frame;
mod gpu;
mod types;

use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLLoadAction,
    MTLRenderPassDescriptor, MTLStoreAction,
};
use objc2_quartz_core::CAMetalDrawable;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use frame::FrameRing;
use gpu::GpuState;

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    frame_ring: FrameRing,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            frame_ring: FrameRing::new(),
        }
    }

    fn render(&mut self) {
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };

        // Acquire a frame slot (blocks if all 3 are in use)
        self.frame_ring.acquire();

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

        // Create render command encoder (empty pass - just clear)
        let encoder = match command_buffer.renderCommandEncoderWithDescriptor(&render_pass_desc) {
            Some(enc) => enc,
            None => {
                eprintln!("Failed to create render command encoder");
                return;
            }
        };

        // End the (empty) render pass
        encoder.endEncoding();

        // Present drawable and commit
        command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
        command_buffer.commit();

        // Advance frame ring index
        self.frame_ring.advance();

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
