//! FSEvents-based index watcher using notify v7 + debouncer-mini.
//!
//! Watches a root directory for filesystem changes (create/modify/delete) via
//! macOS FSEvents (through notify's `RecommendedWatcher`). Events are debounced
//! at 500ms to coalesce rapid edits. On change, the watcher re-scans the root
//! directory via `FilesystemScanner`, rebuilds the `GpuResidentIndex`, and persists
//! via `SharedIndexManager::save()`.
//!
//! # Architecture
//!
//! ```text
//! FSEvents -> notify RecommendedWatcher -> debouncer-mini (500ms)
//!   -> mpsc channel -> processor thread: re-scan -> rebuild index -> persist GSIX
//! ```
//!
//! Uses a channel-based design to avoid Send/Sync issues with Metal buffer types.
//! The debouncer callback sends events to an mpsc channel, and a dedicated
//! processor thread handles index rebuilds with access to the shared index.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use notify_debouncer_mini::{new_debouncer, DebounceEventResult, Debouncer};

use crate::index::gpu_index::GpuResidentIndex;
use crate::index::scanner::FilesystemScanner;
use crate::index::shared_index::SharedIndexManager;

// ============================================================================
// IndexWatcher
// ============================================================================

/// Watches a directory tree for changes and keeps a persisted GSIX index up to date.
///
/// Uses macOS FSEvents (via notify v7 `RecommendedWatcher`) with a 500ms debounce
/// window. On detected changes, re-scans the root directory and persists the
/// updated index to disk via `SharedIndexManager`.
///
/// # Example
///
/// ```ignore
/// use std::path::Path;
///
/// let manager = SharedIndexManager::new().unwrap();
/// let mut watcher = IndexWatcher::new(manager);
/// watcher.start(Path::new("/some/project")).unwrap();
///
/// // ... index is kept up to date on disk ...
///
/// watcher.stop();
/// ```
pub struct IndexWatcher {
    /// Persists index to disk after updates.
    manager: SharedIndexManager,
    /// Active debounced watcher. None when stopped.
    debouncer: Option<Debouncer<notify::RecommendedWatcher>>,
    /// Channel to signal the processor thread to shut down.
    shutdown_tx: Option<mpsc::Sender<()>>,
    /// Handle to the processor thread.
    processor_handle: Option<JoinHandle<()>>,
    /// Root path being watched.
    root: Option<PathBuf>,
}

impl IndexWatcher {
    /// Create a new `IndexWatcher` with the given persistence manager.
    ///
    /// The watcher is not started until `start()` is called.
    pub fn new(manager: SharedIndexManager) -> Self {
        Self {
            manager,
            debouncer: None,
            shutdown_tx: None,
            processor_handle: None,
            root: None,
        }
    }

    /// Start watching the given root directory for filesystem changes.
    ///
    /// Creates a debounced watcher with a 500ms timeout. On events:
    /// 1. Debounced events are sent to an internal channel
    /// 2. A processor thread re-scans the root with `FilesystemScanner`
    /// 3. Rebuilds `GpuResidentIndex` from scanned entries
    /// 4. Persists via `SharedIndexManager::save()`
    ///
    /// Returns an error if the watcher cannot be created or the path cannot be watched.
    pub fn start(&mut self, root: &Path) -> Result<(), WatcherError> {
        // Stop any existing watcher first
        self.stop();

        let root_path = root
            .canonicalize()
            .map_err(|e| WatcherError::Io(e, root.to_path_buf()))?;

        // Channel for debounced events -> processor thread
        let (event_tx, event_rx) = mpsc::channel::<Vec<notify_debouncer_mini::DebouncedEvent>>();

        // Channel to signal shutdown to the processor thread
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();

        // Spawn processor thread that handles index rebuilds
        let processor_root = root_path.clone();
        let manager_base_dir = self.manager.base_dir().to_path_buf();
        let processor_handle = thread::Builder::new()
            .name("index-watcher-processor".into())
            .spawn(move || {
                processor_loop(
                    &event_rx,
                    &shutdown_rx,
                    &processor_root,
                    &manager_base_dir,
                );
            })
            .map_err(|e| WatcherError::Io(e, root_path.clone()))?;

        // Create debounced watcher (500ms debounce via notify-debouncer-mini)
        let mut debouncer = new_debouncer(
            Duration::from_millis(500),
            move |result: DebounceEventResult| {
                match result {
                    Ok(events) if !events.is_empty() => {
                        // Send events to processor thread (ignore send errors on shutdown)
                        let _ = event_tx.send(events);
                    }
                    Err(e) => {
                        eprintln!("[IndexWatcher] notify error: {e}");
                    }
                    _ => {}
                }
            },
        )
        .map_err(WatcherError::Notify)?;

        // Watch root recursively (uses FSEvents on macOS)
        debouncer
            .watcher()
            .watch(&root_path, notify::RecursiveMode::Recursive)
            .map_err(WatcherError::Notify)?;

        self.debouncer = Some(debouncer);
        self.shutdown_tx = Some(shutdown_tx);
        self.processor_handle = Some(processor_handle);
        self.root = Some(root_path);

        Ok(())
    }

    /// Stop watching for filesystem changes.
    ///
    /// Drops the internal watcher and shuts down the processor thread.
    /// Safe to call multiple times or when already stopped.
    pub fn stop(&mut self) {
        // Drop the debouncer first (sends shutdown to debounce thread)
        self.debouncer.take();

        // Signal processor thread to shut down
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        // Wait for processor thread to finish
        if let Some(handle) = self.processor_handle.take() {
            let _ = handle.join();
        }

        self.root.take();
    }

    /// Returns `true` if the watcher is currently active.
    pub fn is_watching(&self) -> bool {
        self.debouncer.is_some()
    }

    /// Returns the root path being watched, if active.
    pub fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    /// Get a reference to the `SharedIndexManager` for loading the latest index.
    pub fn manager(&self) -> &SharedIndexManager {
        &self.manager
    }
}

impl Drop for IndexWatcher {
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================================
// Processor Loop
// ============================================================================

/// Background loop that processes debounced events and updates the index.
///
/// Waits for events on the event channel. On receipt:
/// 1. Re-scans the root directory with `FilesystemScanner`
/// 2. Builds a new `GpuResidentIndex` from scanned entries
/// 3. Persists via `SharedIndexManager::save()`
///
/// Exits when the shutdown channel fires or the event channel disconnects.
fn processor_loop(
    event_rx: &mpsc::Receiver<Vec<notify_debouncer_mini::DebouncedEvent>>,
    shutdown_rx: &mpsc::Receiver<()>,
    root: &Path,
    manager_base_dir: &Path,
) {
    loop {
        // Check for shutdown signal (non-blocking)
        if shutdown_rx.try_recv().is_ok() {
            break;
        }

        // Wait for events with a timeout so we can check shutdown periodically
        match event_rx.recv_timeout(Duration::from_millis(200)) {
            Ok(events) => {
                // Log event summary
                let event_count = events.len();
                let paths: Vec<&Path> = events.iter().map(|e| e.path.as_path()).collect();
                eprintln!(
                    "[IndexWatcher] {} change(s) detected, re-scanning: {:?}",
                    event_count,
                    &paths[..paths.len().min(5)]
                );

                // Re-scan the entire root directory
                let scanner = FilesystemScanner::new();
                let entries = scanner.scan(root);
                let entry_count = entries.len();

                // Build new index and persist to disk
                let new_index = GpuResidentIndex::from_entries(entries);
                let manager =
                    SharedIndexManager::with_base_dir(manager_base_dir.to_path_buf());
                if let Err(e) = manager.save(&new_index, root) {
                    eprintln!("[IndexWatcher] failed to persist index: {e}");
                } else {
                    eprintln!(
                        "[IndexWatcher] index updated: {entry_count} entries for {}",
                        root.display()
                    );
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // No events, loop back to check shutdown
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Event channel closed, exit
                break;
            }
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Errors from `IndexWatcher` operations.
#[derive(Debug)]
pub enum WatcherError {
    /// Failed to create or configure the filesystem watcher
    Notify(notify::Error),
    /// I/O error (e.g., canonicalizing root path or spawning thread)
    Io(std::io::Error, PathBuf),
}

impl std::fmt::Display for WatcherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Notify(e) => write!(f, "watcher error: {e}"),
            Self::Io(e, path) => write!(f, "I/O error for {}: {e}", path.display()),
        }
    }
}

impl std::error::Error for WatcherError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Notify(e) => Some(e),
            Self::Io(e, _) => Some(e),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Test that IndexWatcher can be instantiated and started on a test directory.
    #[test]
    fn test_watcher_instantiate_and_start() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let cache_dir = TempDir::new().expect("Failed to create cache dir");

        // Create some initial files
        std::fs::write(dir.path().join("test.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("lib.rs"), "pub mod test;").unwrap();

        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Create and start watcher
        let mut watcher = IndexWatcher::new(manager);
        assert!(!watcher.is_watching());

        watcher.start(dir.path()).expect("Failed to start watcher");
        assert!(watcher.is_watching());
        assert!(watcher.root().is_some());

        // Stop the watcher
        watcher.stop();
        assert!(!watcher.is_watching());
        assert!(watcher.root().is_none());
    }

    /// Test that watcher can be stopped and restarted.
    #[test]
    fn test_watcher_stop_and_restart() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let cache_dir = TempDir::new().expect("Failed to create cache dir");

        std::fs::write(dir.path().join("file.txt"), "hello").unwrap();

        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        let mut watcher = IndexWatcher::new(manager);

        // Start
        watcher.start(dir.path()).expect("start failed");
        assert!(watcher.is_watching());

        // Stop
        watcher.stop();
        assert!(!watcher.is_watching());

        // Restart
        watcher.start(dir.path()).expect("restart failed");
        assert!(watcher.is_watching());

        // Drop should also stop cleanly
        drop(watcher);
    }

    /// Test that watcher handles invalid paths gracefully.
    #[test]
    fn test_watcher_invalid_path() {
        let cache_dir = TempDir::new().expect("Failed to create cache dir");

        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        let mut watcher = IndexWatcher::new(manager);

        // Watching a nonexistent path should fail
        let result = watcher.start(Path::new("/nonexistent/path/that/does/not/exist"));
        assert!(result.is_err());
        assert!(!watcher.is_watching());
    }

    /// Test that manager() returns the SharedIndexManager reference.
    #[test]
    fn test_watcher_manager_access() {
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let base_path = cache_dir.path().to_path_buf();

        let manager = SharedIndexManager::with_base_dir(base_path.clone());

        let watcher = IndexWatcher::new(manager);
        assert_eq!(watcher.manager().base_dir(), base_path);
    }
}
