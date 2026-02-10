//! Frame pacing with semaphore-based GPU/CPU synchronization.
//!
//! Manages a dispatch semaphore to limit in-flight frames, computes per-frame
//! delta time, and tracks FPS for the window title HUD.

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

    /// True after acquire(), false after register_completion_handler().
    /// Used by Drop to decide whether to signal the semaphore manually
    /// (no handler registered) or wait for the GPU handler to signal first.
    acquired_no_handler: bool,
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
            acquired_no_handler: false,
        }
    }

    /// Acquire a frame slot by waiting on the semaphore.
    /// Blocks if all 3 frame slots are in use (GPU hasn't finished yet).
    /// Also updates dt and frame index.
    pub fn acquire(&mut self) {
        // Block until a frame slot is available
        self.semaphore.wait(DispatchTime::FOREVER);
        self.acquired_no_handler = true;

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
        &mut self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) {
        self.acquired_no_handler = false;
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

impl Drop for FrameRing {
    fn drop(&mut self) {
        if self.acquired_no_handler {
            // Semaphore was decremented by acquire() but no completion handler was
            // registered (e.g., panic during frame setup before commit). Signal it
            // back to restore the original value so libdispatch doesn't crash.
            self.semaphore.signal();
        } else {
            // A completion handler was registered on the last command buffer.
            // Wait for the GPU to finish and signal, then signal back to restore
            // the semaphore to its original value before deallocation.
            self.semaphore.wait(DispatchTime::FOREVER);
            self.semaphore.signal();
        }
    }
}
