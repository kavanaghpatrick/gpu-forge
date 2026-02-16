//! Diagnostic watchdog for UI freeze debugging.
//!
//! Runs an independent thread that logs to stderr every 500ms, capturing:
//! - Main thread (UI) last-seen timestamp
//! - Background thread (orchestrator) last-seen timestamp
//! - Process RSS memory usage
//! - Channel queue depth approximation
//! - GPU batch count from orchestrator
//!
//! This provides visibility into the freeze even when update() stops being called.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Shared diagnostic state between threads.
///
/// All fields are atomic for lock-free cross-thread access.
/// Timestamps are stored as microseconds since `epoch` (the Instant when
/// the Watchdog was created).
pub struct DiagState {
    /// Microseconds since epoch when update() last ran on the main thread.
    pub ui_last_us: AtomicU64,
    /// Microseconds since epoch when the orchestrator bg thread last ran.
    pub bg_last_us: AtomicU64,
    /// Current GPU batch number being processed.
    pub bg_batch_num: AtomicU64,
    /// Total content matches accumulated so far (bg thread writes).
    pub bg_content_matches: AtomicU64,
    /// Total files searched so far (bg thread writes).
    pub bg_files_searched: AtomicU64,
    /// Number of messages in the update channel (UI side reads).
    pub channel_pending: AtomicU64,
    /// Content matches on UI side.
    pub ui_content_matches: AtomicU64,
    /// UI frame count.
    pub ui_frame_count: AtomicU64,
    /// Whether a search is active.
    pub search_active: AtomicBool,
    /// Signal to stop the watchdog thread.
    stop: AtomicBool,
    /// The reference instant (creation time).
    epoch: Instant,
}

impl Default for DiagState {
    fn default() -> Self {
        Self::new()
    }
}

impl DiagState {
    fn new() -> Self {
        Self {
            ui_last_us: AtomicU64::new(0),
            bg_last_us: AtomicU64::new(0),
            bg_batch_num: AtomicU64::new(0),
            bg_content_matches: AtomicU64::new(0),
            bg_files_searched: AtomicU64::new(0),
            channel_pending: AtomicU64::new(0),
            ui_content_matches: AtomicU64::new(0),
            ui_frame_count: AtomicU64::new(0),
            search_active: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            epoch: Instant::now(),
        }
    }

    /// Record a heartbeat from the UI thread.
    pub fn touch_ui(&self) {
        let us = self.epoch.elapsed().as_micros() as u64;
        self.ui_last_us.store(us, Ordering::Relaxed);
    }

    /// Record a heartbeat from the background orchestrator thread.
    pub fn touch_bg(&self) {
        let us = self.epoch.elapsed().as_micros() as u64;
        self.bg_last_us.store(us, Ordering::Relaxed);
    }

    /// Milliseconds since the UI thread last called touch_ui().
    /// Returns -1.0 if the UI was never seen.
    pub fn ui_stall_ms(&self) -> f64 {
        let stored_us = self.ui_last_us.load(Ordering::Relaxed);
        if stored_us == 0 {
            return -1.0;
        }
        let now_us = self.epoch.elapsed().as_micros() as u64;
        (now_us.saturating_sub(stored_us)) as f64 / 1000.0
    }

    /// Get elapsed seconds since epoch for a stored timestamp.
    fn age_secs(&self, stored_us: u64) -> f64 {
        let now_us = self.epoch.elapsed().as_micros() as u64;
        if stored_us == 0 {
            return -1.0; // never seen
        }
        (now_us.saturating_sub(stored_us)) as f64 / 1_000_000.0
    }
}

/// Get resident memory (RSS) in MB via mach kernel APIs.
fn rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut u8,
                task_info_outCnt: *mut u32,
            ) -> i32;
        }

        // MACH_TASK_BASIC_INFO = 20
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],    // time_value_t
            system_time: [u32; 2],  // time_value_t
            policy: i32,
            suspend_count: i32,
        }

        unsafe {
            let mut info: MachTaskBasicInfo = mem::zeroed();
            let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;
            let kr = task_info(
                mach_task_self(),
                20, // MACH_TASK_BASIC_INFO
                &mut info as *mut _ as *mut u8,
                &mut count,
            );
            if kr == 0 {
                info.resident_size as f64 / (1024.0 * 1024.0)
            } else {
                -1.0
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        -1.0
    }
}

/// Watchdog handle. Stops the background thread on drop.
pub struct Watchdog {
    pub state: Arc<DiagState>,
    _thread: Option<thread::JoinHandle<()>>,
}

impl Watchdog {
    /// Start a new watchdog that logs every 500ms.
    pub fn start() -> Self {
        let state = Arc::new(DiagState::new());
        let s = Arc::clone(&state);

        let thread = thread::Builder::new()
            .name("watchdog".into())
            .spawn(move || {
                eprintln!("[WATCHDOG] started — logging every 500ms");
                let mut tick = 0u64;
                while !s.stop.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(500));
                    tick += 1;

                    let ui_age = s.age_secs(s.ui_last_us.load(Ordering::Relaxed));
                    let bg_age = s.age_secs(s.bg_last_us.load(Ordering::Relaxed));
                    let batch = s.bg_batch_num.load(Ordering::Relaxed);
                    let bg_cm = s.bg_content_matches.load(Ordering::Relaxed);
                    let bg_files = s.bg_files_searched.load(Ordering::Relaxed);
                    let ch = s.channel_pending.load(Ordering::Relaxed);
                    let ui_cm = s.ui_content_matches.load(Ordering::Relaxed);
                    let ui_fc = s.ui_frame_count.load(Ordering::Relaxed);
                    let active = s.search_active.load(Ordering::Relaxed);
                    let rss = rss_mb();

                    // Color-code: if UI age > 2s, mark with FROZEN
                    let ui_status = if ui_age > 2.0 {
                        "FROZEN"
                    } else if ui_age < 0.0 {
                        "UNSEEN"
                    } else {
                        "ok"
                    };

                    let bg_status = if bg_age > 5.0 {
                        "STUCK"
                    } else if bg_age < 0.0 {
                        "UNSEEN"
                    } else {
                        "ok"
                    };

                    eprintln!(
                        "[WD {:>4}] ui={:.2}s({}) bg={:.2}s({}) batch={} files={} \
                         bg_cm={} ui_cm={} ch={} frames={} rss={:.0}MB search={}",
                        tick,
                        ui_age, ui_status,
                        bg_age, bg_status,
                        batch, bg_files,
                        bg_cm, ui_cm,
                        ch, ui_fc,
                        rss,
                        if active { "Y" } else { "N" },
                    );

                    // Alert: UI frozen while search active — sample the process
                    if ui_age > 3.0 && active {
                        eprintln!(
                            "[WD ALERT] UI FROZEN for {:.1}s while search active! \
                             bg_age={:.2}s batch={} rss={:.0}MB",
                            ui_age, bg_age, batch, rss,
                        );

                        // On first freeze detection (3-4s), capture a stack sample
                        // to see WHERE the main thread is blocked.
                        if ui_age > 3.0 && ui_age < 4.0 {
                            let pid = std::process::id();
                            eprintln!("[WD] Sampling process {} for 1 second...", pid);
                            let output = std::process::Command::new("sample")
                                .args([&pid.to_string(), "1", "10"])
                                .output();
                            match output {
                                Ok(o) => {
                                    let stdout = String::from_utf8_lossy(&o.stdout);
                                    // Extract just the main thread stack
                                    let mut in_main = false;
                                    let mut lines_printed = 0;
                                    for line in stdout.lines() {
                                        if line.contains("Thread_") && line.contains("DispatchQueue") {
                                            if in_main {
                                                break; // End of main thread section
                                            }
                                            if line.contains("main") || line.contains("Thread_0") {
                                                in_main = true;
                                                eprintln!("[SAMPLE] {}", line);
                                            }
                                            continue;
                                        }
                                        if in_main {
                                            eprintln!("[SAMPLE] {}", line);
                                            lines_printed += 1;
                                            if lines_printed > 80 {
                                                eprintln!("[SAMPLE] ... (truncated)");
                                                break;
                                            }
                                        }
                                    }
                                    if !in_main {
                                        // Fallback: print last 40 lines
                                        let all_lines: Vec<&str> = stdout.lines().collect();
                                        let start = all_lines.len().saturating_sub(40);
                                        for line in &all_lines[start..] {
                                            eprintln!("[SAMPLE] {}", line);
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("[WD] sample failed: {}", e);
                                }
                            }
                        }
                    }
                }
                eprintln!("[WATCHDOG] stopped");
            })
            .expect("failed to spawn watchdog thread");

        Watchdog {
            state,
            _thread: Some(thread),
        }
    }

    /// Stop the watchdog thread.
    pub fn stop(&self) {
        self.state.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.stop();
        // Don't join — if main thread is exiting, the watchdog will die with the process.
    }
}
