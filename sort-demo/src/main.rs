mod frame;
mod gpu;

use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLLoadAction,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLStoreAction,
};
use objc2_quartz_core::CAMetalDrawable;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
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

        self.frame_ring.acquire();

        let drawable = match gpu.layer.nextDrawable() {
            Some(d) => d,
            None => return,
        };

        let command_buffer = match gpu.command_queue.commandBuffer() {
            Some(cb) => cb,
            None => return,
        };

        // Render pass: clear to blue
        let render_pass_desc = MTLRenderPassDescriptor::renderPassDescriptor();
        let color_attachment =
            unsafe { render_pass_desc.colorAttachments().objectAtIndexedSubscript(0) };
        color_attachment.setTexture(Some(&drawable.texture()));
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setStoreAction(MTLStoreAction::Store);
        color_attachment.setClearColor(MTLClearColor {
            red: 0.0,
            green: 0.1,
            blue: 0.3,
            alpha: 1.0,
        });

        if let Some(encoder) = command_buffer.renderCommandEncoderWithDescriptor(&render_pass_desc)
        {
            encoder.endEncoding();
        }

        self.frame_ring.register_completion_handler(&command_buffer);
        command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
        command_buffer.commit();
        self.frame_ring.advance();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Sort Stadium")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        let gpu = GpuState::new();

        let handle = window.window_handle().expect("Failed to get window handle");
        match handle.as_raw() {
            RawWindowHandle::AppKit(appkit_handle) => {
                unsafe {
                    gpu.attach_layer_to_view(appkit_handle.ns_view);
                }
                let size = window.inner_size();
                gpu.layer.setDrawableSize(CGSize {
                    width: size.width as f64,
                    height: size.height as f64,
                });
            }
            _ => panic!("Unsupported platform"),
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
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &self.gpu {
                    gpu.layer.setDrawableSize(CGSize {
                        width: size.width as f64,
                        height: size.height as f64,
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
    event_loop.run_app(&mut app).expect("Event loop error");
}
