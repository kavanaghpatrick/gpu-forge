mod frame;
mod gpu;
mod sort;
mod vis;

use std::sync::Arc;
use std::time::Instant;

use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

use frame::FrameRing;
use gpu::GpuState;
use sort::{SortAlgorithm, SortEngine, SCALES};
use vis::{VisMode, Visualization};

struct CpuSortState {
    handle: Option<std::thread::JoinHandle<f64>>,
    start: Instant,
    #[allow(dead_code)]
    element_count: usize,
    result_ms: Option<f64>,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    sort_engine: Option<SortEngine>,
    vis: Option<Visualization>,
    frame_ring: FrameRing,
    scale_index: usize,
    needs_shuffle: bool,
    auto_sort: bool,
    cpu_sort: Option<CpuSortState>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            sort_engine: None,
            vis: None,
            frame_ring: FrameRing::new(),
            scale_index: 2, // 16M
            needs_shuffle: false,
            auto_sort: false,
            cpu_sort: None,
        }
    }

    fn render(&mut self) {
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };

        self.frame_ring.acquire();

        autoreleasepool(|_| {
            let sort_engine = self.sort_engine.as_mut().unwrap();
            let vis = self.vis.as_mut().unwrap();

            // Update element count
            let n = SCALES[self.scale_index];
            sort_engine.set_element_count(n);
            vis.resize_if_needed(&gpu.device, n);

            // Poll CPU sort
            if let Some(ref mut cpu) = self.cpu_sort {
                if cpu.result_ms.is_none()
                    && cpu
                        .handle
                        .as_ref()
                        .map(|h| h.is_finished())
                        .unwrap_or(false)
                {
                    cpu.result_ms = Some(cpu.start.elapsed().as_secs_f64() * 1000.0);
                }
            }

            let drawable = match gpu.layer.nextDrawable() {
                Some(d) => d,
                None => return,
            };

            let cmd = match gpu.command_queue.commandBuffer() {
                Some(cb) => cb,
                None => return,
            };

            // Shuffle + sort if requested
            if self.needs_shuffle || self.auto_sort {
                let sort_start = Instant::now();
                sort_engine.encode_shuffle(&cmd);
                sort_engine.encode_sort(&cmd);
                sort_engine.last_sort_ms = sort_start.elapsed().as_secs_f64() * 1000.0;
                self.needs_shuffle = false;
            }

            // Visualize: compute read buf_a -> write heatmap texture
            vis.encode_visualize(&cmd, sort_engine.buf_a.as_ref(), n);

            // Render: fullscreen triangle -> drawable
            vis.encode_render(&cmd, &drawable);

            self.frame_ring.register_completion_handler(&cmd);
            cmd.presentDrawable(ProtocolObject::from_ref(&*drawable));
            cmd.commit();
        });

        self.frame_ring.advance();

        // Update title periodically
        if self.frame_ring.should_update_fps() {
            self.update_title();
        }
    }

    fn update_title(&self) {
        let window = match &self.window {
            Some(w) => w,
            None => return,
        };
        let sort_engine = match &self.sort_engine {
            Some(s) => s,
            None => return,
        };

        let algo = match sort_engine.algorithm {
            SortAlgorithm::Hybrid => "Hybrid",
            SortAlgorithm::EightBit => "8-Bit",
        };
        let n = SCALES[self.scale_index];
        let count_str = format!("{:.1}M", n as f64 / 1_000_000.0);
        let fps = self.frame_ring.fps;

        let mut title = format!("Sort Stadium | {} | {}", algo, count_str);

        if sort_engine.last_sort_ms > 0.0 {
            let throughput = n as f64 / sort_engine.last_sort_ms / 1000.0;
            title.push_str(&format!(
                " | {:.2}ms | {:.0} Mk/s",
                sort_engine.last_sort_ms, throughput
            ));
        }

        // CPU sort status
        if let Some(ref cpu) = self.cpu_sort {
            if let Some(ms) = cpu.result_ms {
                title.push_str(&format!(" | CPU: {:.0}ms", ms));
            } else {
                let elapsed = cpu.start.elapsed().as_secs_f64();
                title.push_str(&format!(" | CPU: {:.1}s...", elapsed));
            }
        }

        if self.auto_sort {
            title.push_str(" | Auto");
        }

        let vis = self.vis.as_ref().unwrap();
        let mode_str = match vis.mode {
            VisMode::Heatmap => "",
            VisMode::Barchart => " | Bar",
        };
        title.push_str(mode_str);

        title.push_str(&format!(" | {} FPS", fps));
        window.set_title(&title);
    }

    fn start_cpu_sort(&mut self) {
        let n = SCALES[self.scale_index];
        let start = Instant::now();
        let handle = std::thread::spawn(move || {
            let mut data: Vec<u32> = (0..n).map(|_| rand::random()).collect();
            data.sort();
            start.elapsed().as_secs_f64()
        });
        self.cpu_sort = Some(CpuSortState {
            handle: Some(handle),
            start,
            element_count: n,
            result_ms: None,
        });
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

        let n = SCALES[self.scale_index];
        let sort_engine = SortEngine::new(&gpu.device, &gpu.library);
        let vis = Visualization::new(&gpu.device, &gpu.library, n);

        self.sort_engine = Some(sort_engine);
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
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        match key {
                            KeyCode::Space => {
                                self.needs_shuffle = true;
                            }
                            KeyCode::KeyS => {
                                self.auto_sort = !self.auto_sort;
                            }
                            KeyCode::Digit1 => {
                                if let Some(se) = &mut self.sort_engine {
                                    se.algorithm = SortAlgorithm::Hybrid;
                                }
                            }
                            KeyCode::Digit2 => {
                                if let Some(se) = &mut self.sort_engine {
                                    se.algorithm = SortAlgorithm::EightBit;
                                }
                            }
                            KeyCode::Digit3 => {
                                self.start_cpu_sort();
                            }
                            KeyCode::Tab => {
                                if let Some(vis) = &mut self.vis {
                                    vis.mode = match vis.mode {
                                        VisMode::Heatmap => VisMode::Barchart,
                                        VisMode::Barchart => VisMode::Heatmap,
                                    };
                                }
                            }
                            KeyCode::ArrowUp => {
                                self.scale_index = (self.scale_index + 1).min(3);
                                self.needs_shuffle = true;
                            }
                            KeyCode::ArrowDown => {
                                self.scale_index = self.scale_index.saturating_sub(1);
                                self.needs_shuffle = true;
                            }
                            KeyCode::Escape => {
                                event_loop.exit();
                            }
                            _ => {}
                        }
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
    event_loop.run_app(&mut app).expect("Event loop error");
}
