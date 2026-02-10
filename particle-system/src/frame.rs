use std::ptr::NonNull;
use std::time::Instant;

use block2::RcBlock;
use dispatch2::{DispatchRetained, DispatchSemaphore, DispatchTime};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;

/// Maximum number of frames in flight.
/// Using 1 (single buffering) because particle SoA buffers and alive/dead lists
/// are shared (not per-frame). Multiple in-flight frames would race on these buffers.
/// For POC this is fine; triple buffering requires per-frame buffer copies or
/// partitioned buffer regions.
const MAX_FRAMES_IN_FLIGHT: usize = 1;

/// Triple-buffer semaphore ring for pipelining CPU/GPU work.
///
/// Uses a dispatch semaphore with count 3 to allow up to 3 frames
/// in flight simultaneously. Tracks frame index (0, 1, 2) and
/// computes delta time between frames.
pub struct FrameRing {
    semaphore: DispatchRetained<DispatchSemaphore>,
    frame_index: usize,
    last_frame_time: Instant,
    /// Delta time in seconds since last frame.
    pub dt: f32,

    // FPS tracking
    fps_frame_count: u32,
    fps_last_update: Instant,
    /// Current FPS value, updated approximately once per second.
    pub fps: u32,
}

impl FrameRing {
    /// Create a new FrameRing with a triple-buffer semaphore.
    pub fn new() -> Self {
        Self {
            semaphore: DispatchSemaphore::new(MAX_FRAMES_IN_FLIGHT as isize),
            frame_index: 0,
            last_frame_time: Instant::now(),
            dt: 0.016,
            fps_frame_count: 0,
            fps_last_update: Instant::now(),
            fps: 0,
        }
    }

    /// Acquire a frame slot by waiting on the semaphore.
    /// Blocks if all 3 frame slots are in use (GPU hasn't finished yet).
    /// Also updates dt and frame index.
    pub fn acquire(&mut self) {
        // Block until a frame slot is available
        self.semaphore.wait(DispatchTime::FOREVER);

        // Compute dt
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame_time);
        self.dt = elapsed.as_secs_f32();
        self.last_frame_time = now;

        // Update FPS counter
        self.fps_frame_count += 1;
        let fps_elapsed = now.duration_since(self.fps_last_update);
        if fps_elapsed.as_secs_f32() >= 0.5 {
            self.fps = (self.fps_frame_count as f32 / fps_elapsed.as_secs_f32()) as u32;
            self.fps_frame_count = 0;
            self.fps_last_update = now;
        }
    }

    /// Register a completion handler on the command buffer that signals the semaphore
    /// when the GPU finishes processing this frame.
    ///
    /// # Safety
    /// The command buffer must be valid and not yet committed.
    pub fn register_completion_handler(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) {
        let semaphore = self.semaphore.clone();
        let handler = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            semaphore.signal();
        });
        unsafe {
            command_buffer.addCompletedHandler(RcBlock::as_ptr(&handler));
        }
    }

    /// Get the current frame index (0, 1, or 2).
    #[allow(dead_code)]
    pub fn frame_index(&self) -> usize {
        self.frame_index
    }

    /// Advance the frame index to the next slot in the ring.
    #[allow(clippy::modulo_one)]
    pub fn advance(&mut self) {
        self.frame_index = (self.frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    /// Returns true if the FPS display should be updated (approximately once per second).
    /// Check this after acquire() to know when to update window title.
    pub fn should_update_fps(&self) -> bool {
        // FPS was just recalculated if fps_frame_count was reset
        self.fps_frame_count == 0 && self.fps > 0
    }
}
