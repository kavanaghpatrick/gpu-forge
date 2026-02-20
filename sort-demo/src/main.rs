mod frame;
mod gpu;
mod vis;

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use frame::FrameRing;
use gpu::GpuState;
use vis::{DemoParams, Visualization};

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    vis: Option<Visualization>,
    frame_ring: FrameRing,
    // Temp: buf_a + pso for random fill (will move to SortEngine later)
    buf_a: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_random_fill: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    element_count: usize,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            vis: None,
            frame_ring: FrameRing::new(),
            buf_a: None,
            pso_random_fill: None,
            element_count: 16_000_000,
        }
    }

    fn render(&mut self) {
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };
        let vis = match &self.vis {
            Some(v) => v,
            None => return,
        };
        let buf_a = match &self.buf_a {
            Some(b) => b,
            None => return,
        };
        let pso = match &self.pso_random_fill {
            Some(p) => p,
            None => return,
        };

        self.frame_ring.acquire();

        let drawable = match gpu.layer.nextDrawable() {
            Some(d) => d,
            None => return,
        };

        let cmd = match gpu.command_queue.commandBuffer() {
            Some(cb) => cb,
            None => return,
        };

        // Random fill buf_a every frame (POC: proves pipeline works)
        let dim = vis.texture_dim;
        let params = DemoParams {
            element_count: self.element_count as u32,
            texture_width: dim,
            texture_height: dim,
            max_value: 0xFFFFFFFF,
        };
        {
            let encoder = cmd
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");
            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const DemoParams as *mut DemoParams)
                        .unwrap()
                        .cast(),
                    std::mem::size_of::<DemoParams>(),
                    1,
                );
            }
            let tg = MTLSize { width: 256, height: 1, depth: 1 };
            let grid = MTLSize {
                width: self.element_count.div_ceil(256),
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
        }

        // Visualize: heatmap compute
        vis.encode_visualize(&cmd, buf_a.as_ref(), self.element_count);

        // Render: fullscreen triangle -> drawable
        vis.encode_render(&cmd, &drawable);

        self.frame_ring.register_completion_handler(&cmd);
        cmd.presentDrawable(ProtocolObject::from_ref(&*drawable));
        cmd.commit();
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

        // Create visualization
        let vis = Visualization::new(&gpu.device, &gpu.library, self.element_count);

        // Allocate buf_a for 64M elements (pre-allocate max)
        let buf_a = gpu
            .device
            .newBufferWithLength_options(64_000_000 * 4, MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate buf_a");

        // Create random fill PSO
        let pso_random_fill = gpu.make_compute_pipeline("gpu_random_fill");

        self.pso_random_fill = Some(pso_random_fill);
        self.buf_a = Some(buf_a);
        self.vis = Some(vis);
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
