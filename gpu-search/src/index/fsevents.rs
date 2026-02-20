use std::path::PathBuf;

/// Represents a filesystem change event from FSEvents.
///
/// Maps FSEvents flags to typed variants for the IndexWriter to process.
#[derive(Debug, Clone)]
pub enum FsChange {
    /// A new file or directory was created.
    Created(PathBuf),
    /// An existing file was modified (content or metadata).
    Modified(PathBuf),
    /// A file or directory was deleted.
    Deleted(PathBuf),
    /// A file or directory was renamed/moved.
    Renamed { old: PathBuf, new: PathBuf },
    /// FSEvents could not deliver granular events for this subtree; rescan required.
    MustRescan(PathBuf),
    /// FSEvents has finished replaying historical events (sinceWhen catch-up complete).
    HistoryDone,
}

// ---------- FSEventsListener (macOS only) ----------

#[cfg(target_os = "macos")]
mod listener {
    use super::FsChange;
    use crate::index::exclude::ExcludeTrie;

    use crossbeam_channel::Sender;
    use std::ffi::CStr;
    use std::os::raw::c_void;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::thread::{self, JoinHandle};

    use fsevent_sys::core_foundation as cf;
    use fsevent_sys::*;

    /// Result of validating a stored FSEvents event ID against the current system state.
    #[derive(Debug, PartialEq)]
    pub enum EventIdValidity {
        /// Stored ID is valid and can be used as sinceWhen.
        Valid(u64),
        /// No stored ID (first run or ID==0); use kFSEventStreamEventIdSinceNow.
        NoStoredId,
        /// Stored ID is ahead of the current system event ID.
        /// This indicates a volume reformat or journal reset -- full rebuild required.
        Regression,
    }

    /// Validate a stored FSEvents event ID against the current system event ID.
    ///
    /// Returns `EventIdValidity::Regression` if the stored ID is greater than
    /// the current system ID (volume reformat / journal reset), signaling that
    /// a full index rebuild is required.
    pub fn validate_event_id(stored_id: u64) -> EventIdValidity {
        if stored_id == 0 {
            return EventIdValidity::NoStoredId;
        }
        let current_id = unsafe { FSEventsGetCurrentEventId() };
        if stored_id > current_id {
            // Event ID regression: stored ID is in the future relative to the
            // current journal. This happens after a volume reformat or
            // Time Machine restore.
            EventIdValidity::Regression
        } else {
            EventIdValidity::Valid(stored_id)
        }
    }

    // ----------------------------------------------------------------
    // Send/Sync wrapper for raw CFRunLoopRef
    // ----------------------------------------------------------------

    /// Wrapper around `CFRunLoopRef` (`*mut c_void`) so we can store it in
    /// `Arc<Mutex<>>`.  The pointer is only used for `CFRunLoopStop` from the
    /// owning thread or from `Drop`, so this is safe.
    #[derive(Clone, Copy)]
    struct SendableRunLoop(cf::CFRunLoopRef);
    unsafe impl Send for SendableRunLoop {}
    unsafe impl Sync for SendableRunLoop {}

    // ----------------------------------------------------------------
    // FSEventsListener struct  (task 3.6)
    // ----------------------------------------------------------------

    /// Watches the entire filesystem via macOS FSEvents and sends typed
    /// `FsChange` events through a crossbeam channel.
    pub struct FSEventsListener {
        /// Handle to the dedicated CFRunLoop thread.
        thread_handle: Option<JoinHandle<()>>,
        /// Channel sender for FsChange events.
        change_tx: Sender<FsChange>,
        /// Atomically-updated last event ID (persisted in GSIX v2 header).
        last_event_id: Arc<AtomicU64>,
        /// Path exclusion filter.
        excludes: Arc<ExcludeTrie>,
        /// Shutdown flag.
        shutdown: Arc<AtomicBool>,
        /// CFRunLoop reference for stopping from another thread.
        run_loop: Arc<std::sync::Mutex<Option<SendableRunLoop>>>,
    }

    impl FSEventsListener {
        /// Create a new FSEventsListener.
        ///
        /// * `excludes`      - Path exclusion trie
        /// * `change_tx`     - Channel to send FsChange events
        /// * `last_event_id` - Atomic event ID (0 = use kFSEventStreamEventIdSinceNow)
        pub fn new(
            excludes: Arc<ExcludeTrie>,
            change_tx: Sender<FsChange>,
            last_event_id: Arc<AtomicU64>,
        ) -> Self {
            Self {
                thread_handle: None,
                change_tx,
                last_event_id,
                excludes,
                shutdown: Arc::new(AtomicBool::new(false)),
                run_loop: Arc::new(std::sync::Mutex::new(None)),
            }
        }

        // ----------------------------------------------------------------
        // start  (tasks 3.6 + 3.7)
        // ----------------------------------------------------------------

        /// Spawn a dedicated thread that creates an FSEvents stream on a
        /// CFRunLoop.  Returns immediately; events flow via `change_tx`.
        ///
        /// Validates the stored event ID before starting:
        /// - If the ID is ahead of the current system event ID (volume reformat /
        ///   journal reset), sends `MustRescan("/")` and resets to `SinceNow`.
        /// - If the ID is zero (first run), uses `kFSEventStreamEventIdSinceNow`.
        /// - Otherwise, resumes from the stored ID.
        ///
        /// Returns `true` if a full rebuild is required (event ID regression
        /// detected), `false` for normal resume.
        pub fn start(&mut self) -> Result<bool, String> {
            if self.thread_handle.is_some() {
                return Err("FSEventsListener already started".into());
            }

            // Validate stored event ID against current system state
            let stored_id = self.last_event_id.load(Ordering::SeqCst);
            let needs_rebuild = match validate_event_id(stored_id) {
                EventIdValidity::Regression => {
                    eprintln!(
                        "FSEventsListener: stored event ID {} is ahead of current system ID; \
                         volume reformat or journal reset detected — triggering full rebuild",
                        stored_id
                    );
                    // Reset to SinceNow (0 means use kFSEventStreamEventIdSinceNow)
                    self.last_event_id.store(0, Ordering::SeqCst);
                    // Notify the index writer that a full rebuild is needed
                    let _ = self.change_tx.send(FsChange::MustRescan(PathBuf::from("/")));
                    true
                }
                EventIdValidity::NoStoredId => false,
                EventIdValidity::Valid(_) => false,
            };

            let change_tx = self.change_tx.clone();
            let last_event_id = Arc::clone(&self.last_event_id);
            let excludes = Arc::clone(&self.excludes);
            let shutdown = Arc::clone(&self.shutdown);
            let run_loop_slot = Arc::clone(&self.run_loop);

            let handle = thread::Builder::new()
                .name("fsevents-listener".into())
                .spawn(move || {
                    run_fsevents_thread(
                        change_tx,
                        last_event_id,
                        excludes,
                        shutdown,
                        run_loop_slot,
                    );
                })
                .map_err(|e| format!("failed to spawn FSEvents thread: {e}"))?;

            self.thread_handle = Some(handle);
            Ok(needs_rebuild)
        }

        /// Signal the listener to stop and join the thread.
        pub fn stop(&mut self) {
            self.shutdown.store(true, Ordering::SeqCst);

            // Stop the CFRunLoop so the thread exits
            if let Ok(guard) = self.run_loop.lock() {
                if let Some(rl) = *guard {
                    unsafe {
                        cf::CFRunLoopStop(rl.0);
                    }
                }
            }

            if let Some(handle) = self.thread_handle.take() {
                let _ = handle.join();
            }
        }
    }

    impl Drop for FSEventsListener {
        fn drop(&mut self) {
            self.stop();
        }
    }

    // ----------------------------------------------------------------
    // CFRunLoop thread  (task 3.7)
    // ----------------------------------------------------------------

    /// Context passed through the FSEvents callback `info` pointer.
    struct CallbackContext {
        change_tx: Sender<FsChange>,
        last_event_id: Arc<AtomicU64>,
        excludes: Arc<ExcludeTrie>,
        /// Tracks last rename path for pairing consecutive rename events.
        pending_rename: Option<PathBuf>,
    }

    fn run_fsevents_thread(
        change_tx: Sender<FsChange>,
        last_event_id: Arc<AtomicU64>,
        excludes: Arc<ExcludeTrie>,
        shutdown: Arc<AtomicBool>,
        run_loop_slot: Arc<std::sync::Mutex<Option<SendableRunLoop>>>,
    ) {
        unsafe {
            // --- Build the paths-to-watch CFArray containing "/" ---
            let root_path = std::ffi::CString::new("/").unwrap();
            let cf_root = cf::CFStringCreateWithCString(
                cf::kCFAllocatorDefault,
                root_path.as_ptr(),
                cf::kCFStringEncodingUTF8,
            );
            let paths = cf::CFArrayCreateMutable(
                cf::kCFAllocatorDefault,
                1,
                &cf::kCFTypeArrayCallBacks,
            );
            cf::CFArrayAppendValue(paths, cf_root);

            // --- Determine sinceWhen ---
            let stored_id = last_event_id.load(Ordering::SeqCst);
            let since_when = if stored_id == 0 {
                kFSEventStreamEventIdSinceNow
            } else {
                stored_id
            };

            // --- Build callback context (heap-allocated, leaked into C) ---
            let ctx = Box::new(CallbackContext {
                change_tx,
                last_event_id,
                excludes,
                pending_rename: None,
            });
            let ctx_ptr = Box::into_raw(ctx) as *mut c_void;

            let mut stream_context = FSEventStreamContext {
                version: 0,
                info: ctx_ptr,
                retain: None,
                release: None,
                copy_description: None,
            };

            // --- Create the FSEvents stream ---
            let flags = kFSEventStreamCreateFlagFileEvents
                | kFSEventStreamCreateFlagUseCFTypes
                | kFSEventStreamCreateFlagNoDefer;

            let stream = FSEventStreamCreate(
                cf::kCFAllocatorDefault,
                fsevents_callback,
                &mut stream_context as *mut FSEventStreamContext,
                paths,
                since_when,
                0.5, // latency in seconds
                flags,
            );

            // Clean up CF objects we no longer need
            cf::CFRelease(cf_root);
            cf::CFRelease(paths);

            if stream.is_null() {
                // Reclaim the context so it doesn't leak
                drop(Box::from_raw(ctx_ptr as *mut CallbackContext));
                eprintln!("FSEventsListener: FSEventStreamCreate returned null");
                return;
            }

            // --- Schedule on this thread's run loop ---
            let current_rl = cf::CFRunLoopGetCurrent();

            // Store the run loop ref so the main thread can stop us
            if let Ok(mut guard) = run_loop_slot.lock() {
                *guard = Some(SendableRunLoop(current_rl));
            }

            FSEventStreamScheduleWithRunLoop(stream, current_rl, cf::kCFRunLoopDefaultMode);
            FSEventStreamStart(stream);

            // --- Run until stopped ---
            if !shutdown.load(Ordering::SeqCst) {
                cf::CFRunLoopRun();
            }

            // --- Teardown ---
            FSEventStreamStop(stream);
            FSEventStreamInvalidate(stream);
            FSEventStreamRelease(stream);

            // Reclaim the context
            drop(Box::from_raw(ctx_ptr as *mut CallbackContext));

            // Clear the stored run loop ref
            if let Ok(mut guard) = run_loop_slot.lock() {
                *guard = None;
            }
        }
    }

    // ----------------------------------------------------------------
    // FSEvents callback  (task 3.8)
    // ----------------------------------------------------------------

    /// C callback invoked by FSEvents for each batch of filesystem events.
    ///
    /// With `kFSEventStreamCreateFlagUseCFTypes`, `event_paths` is a
    /// `CFArrayRef` of `CFStringRef` values.
    extern "C" fn fsevents_callback(
        _stream_ref: FSEventStreamRef,
        info: *mut c_void,
        num_events: usize,
        event_paths: *mut c_void,
        event_flags: *const FSEventStreamEventFlags,
        event_ids: *const FSEventStreamEventId,
    ) {
        let ctx = unsafe { &mut *(info as *mut CallbackContext) };
        let cf_paths = event_paths as cf::CFArrayRef;

        let mut max_id: u64 = 0;

        for i in 0..num_events {
            let flags = unsafe { *event_flags.add(i) };
            let event_id = unsafe { *event_ids.add(i) };
            if event_id > max_id {
                max_id = event_id;
            }

            // --- Extract path string from CFArray ---
            let path = unsafe {
                let cf_str = cf::CFArrayGetValueAtIndex(cf_paths, i as cf::CFIndex);
                if cf_str.is_null() {
                    continue;
                }
                cfstring_to_pathbuf(cf_str)
            };

            let path = match path {
                Some(p) => p,
                None => continue,
            };

            // --- HistoryDone is a special flag (no path filtering) ---
            if flags & kFSEventStreamEventFlagHistoryDone != 0 {
                let _ = ctx.change_tx.send(FsChange::HistoryDone);
                continue;
            }

            // --- Event ID wrapped: the 64-bit counter rolled over (extremely
            //     rare) — treat as journal truncation, trigger full rebuild ---
            if flags & kFSEventStreamEventFlagEventIdsWrapped != 0 {
                eprintln!(
                    "FSEventsListener: event IDs wrapped — triggering full rebuild"
                );
                let _ = ctx
                    .change_tx
                    .send(FsChange::MustRescan(PathBuf::from("/")));
                continue;
            }

            // --- UserDropped / KernelDropped: FSEvents could not deliver all
            //     events since sinceWhen (journal truncation). The stored event
            //     ID is older than the oldest retained event. ---
            if flags
                & (kFSEventStreamEventFlagUserDropped
                    | kFSEventStreamEventFlagKernelDropped)
                != 0
            {
                eprintln!(
                    "FSEventsListener: events dropped (journal truncation) — triggering full rebuild"
                );
                let _ = ctx
                    .change_tx
                    .send(FsChange::MustRescan(PathBuf::from("/")));
                continue;
            }

            // --- MustScanSubDirs bypasses exclude check ---
            if flags & kFSEventStreamEventFlagMustScanSubDirs != 0 {
                let _ = ctx.change_tx.send(FsChange::MustRescan(path));
                continue;
            }

            // --- Filter excluded paths ---
            if ctx.excludes.should_exclude(path.as_os_str().as_encoded_bytes()) {
                // Still need to reset pending_rename if this was a rename event
                if flags & kFSEventStreamEventFlagItemRenamed != 0 {
                    ctx.pending_rename = None;
                }
                continue;
            }

            // --- Map flags to FsChange variants ---

            // Rename handling: FSEvents emits paired rename events.
            // The first has the old path, the second has the new path.
            if flags & kFSEventStreamEventFlagItemRenamed != 0 {
                match ctx.pending_rename.take() {
                    None => {
                        // First of a rename pair -- store the old path
                        ctx.pending_rename = Some(path);
                    }
                    Some(old_path) => {
                        // Second of a rename pair -- emit the Renamed event
                        let _ = ctx.change_tx.send(FsChange::Renamed {
                            old: old_path,
                            new: path,
                        });
                    }
                }
                continue;
            }

            // If we had a pending rename but the next event is NOT a rename,
            // the old path was moved out of scope -- treat as deletion.
            if let Some(orphan) = ctx.pending_rename.take() {
                let _ = ctx.change_tx.send(FsChange::Deleted(orphan));
            }

            // Created
            if flags & kFSEventStreamEventFlagItemCreated != 0 {
                let _ = ctx.change_tx.send(FsChange::Created(path));
                continue;
            }

            // Deleted
            if flags & kFSEventStreamEventFlagItemRemoved != 0 {
                let _ = ctx.change_tx.send(FsChange::Deleted(path));
                continue;
            }

            // Modified (content or inode metadata)
            if flags
                & (kFSEventStreamEventFlagItemModified
                    | kFSEventStreamEventFlagItemInodeMetaMod)
                != 0
            {
                let _ = ctx.change_tx.send(FsChange::Modified(path));
                continue;
            }

            // Fallback: if none of the above flags matched but we got an
            // event, treat as Modified (e.g. xattr changes, owner changes).
            let _ = ctx.change_tx.send(FsChange::Modified(path));
        }

        // Update the last event ID atomically after each batch
        if max_id > 0 {
            ctx.last_event_id.store(max_id, Ordering::SeqCst);
        }
    }

    /// Extract a `PathBuf` from a `CFStringRef` using `CFStringGetCString`.
    unsafe fn cfstring_to_pathbuf(cf_str: cf::CFRef) -> Option<PathBuf> {
        // Try the fast path first: direct pointer to C string
        let cstr_ptr =
            cf::CFStringGetCStringPtr(cf_str, cf::kCFStringEncodingUTF8);
        if !cstr_ptr.is_null() {
            let cstr = CStr::from_ptr(cstr_ptr);
            return Some(PathBuf::from(
                std::str::from_utf8(cstr.to_bytes()).ok()?,
            ));
        }

        // Slow path: copy into a buffer
        let mut buf = [0u8; 1024];
        let ok = cf::CFStringGetCString(
            cf_str,
            buf.as_mut_ptr() as *mut _,
            buf.len() as cf::CFIndex,
            cf::kCFStringEncodingUTF8,
        );
        if ok {
            let cstr = CStr::from_ptr(buf.as_ptr() as *const _);
            Some(PathBuf::from(
                std::str::from_utf8(cstr.to_bytes()).ok()?,
            ))
        } else {
            None
        }
    }
}

#[cfg(target_os = "macos")]
pub use listener::FSEventsListener;
#[cfg(target_os = "macos")]
pub use listener::{validate_event_id, EventIdValidity};
