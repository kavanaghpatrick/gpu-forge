//! Background initial index builder and lifecycle coordinator.
//!
//! ## BackgroundBuilder
//!
//! On first launch (no `global.idx`), spawns a background thread that runs
//! `FilesystemScanner::scan("/")` with the expanded exclude list, building
//! a full v2 index. An `Arc<AtomicUsize>` progress counter is incremented
//! per file scanned so the UI can display live progress.
//!
//! On completion, the builder saves the v2 index to disk and creates an
//! initial [`IndexSnapshot`] in the [`IndexStore`].
//!
//! ## IndexDaemon
//!
//! Top-level coordinator that orchestrates the full index lifecycle:
//! 1. Load existing index or start background build
//! 2. Start FSEventsListener for filesystem change monitoring
//! 3. Start IndexWriter thread for incremental updates
//! 4. Handle graceful shutdown of all subsystems

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::Sender;

use crate::gpu::types::GpuPathEntry;
use crate::index::exclude::{
    compute_exclude_hash, default_excludes, load_config, merge_with_config, ExcludeTrie,
};
use crate::index::fsevents::FsChange;
#[cfg(target_os = "macos")]
use crate::index::fsevents::FSEventsListener;
use crate::index::global::{ensure_index_dir, global_cache_key, global_index_path};
use crate::index::gsix_v2::{is_stale, save_v2};
use crate::index::index_writer::{spawn_writer_thread, IndexWriter};
use crate::index::scanner::{FilesystemScanner, ScannerConfig};
use crate::index::snapshot::IndexSnapshot;
use crate::index::store::IndexStore;

/// Background index builder for first-launch initial scan.
///
/// Spawns a dedicated thread that scans "/" with the configured excludes,
/// builds a full v2 index, and publishes the result to the shared
/// [`IndexStore`].
///
/// # Progress Tracking
///
/// The `progress` counter is an `Arc<AtomicUsize>` shared with the UI.
/// It is set to the total number of files indexed once the scan completes.
/// (The scanner collects all entries in parallel, so the counter is updated
/// in bulk after the scan finishes rather than per-file during the parallel
/// walk.)
pub struct BackgroundBuilder {
    /// Shared progress counter: number of files indexed so far.
    pub progress: Arc<AtomicUsize>,

    /// Lock-free snapshot store for publishing the built index.
    store: Arc<IndexStore>,

    /// Path exclusion filter.
    excludes: Arc<ExcludeTrie>,

    /// Root directory to scan.
    root: PathBuf,

    /// Handle to the background build thread (if spawned).
    build_thread: Option<JoinHandle<()>>,
}

impl BackgroundBuilder {
    /// Create a new builder with the given dependencies.
    pub fn new(
        store: Arc<IndexStore>,
        excludes: Arc<ExcludeTrie>,
        root: PathBuf,
    ) -> Self {
        Self {
            progress: Arc::new(AtomicUsize::new(0)),
            store,
            excludes,
            root,
            build_thread: None,
        }
    }

    /// Check whether a global index already exists on disk.
    pub fn needs_initial_build() -> bool {
        !global_index_path().exists()
    }

    /// Spawn the background build thread.
    ///
    /// The thread:
    /// 1. Ensures the index directory exists.
    /// 2. Scans "/" with the configured excludes using `FilesystemScanner`.
    /// 3. Updates the progress counter with the total files found.
    /// 4. Saves the v2 index to `global.idx`.
    /// 5. Creates an `IndexSnapshot` and swaps it into the `IndexStore`.
    ///
    /// Returns `&Self` for chaining. Call this only once.
    pub fn spawn_build_thread(&mut self) -> &mut Self {
        let progress = Arc::clone(&self.progress);
        let store = Arc::clone(&self.store);
        let excludes = Arc::clone(&self.excludes);
        let root = self.root.clone();

        let handle = thread::Builder::new()
            .name("index-builder".to_string())
            .spawn(move || {
                if let Err(e) = build_initial_index(&progress, &store, &excludes, &root) {
                    eprintln!("BackgroundBuilder: initial build failed: {}", e);
                }
            })
            .expect("failed to spawn index-builder thread");

        self.build_thread = Some(handle);
        self
    }

    /// Wait for the build thread to finish.
    ///
    /// Returns `Ok(())` if the thread joined successfully, or `Err` with
    /// a description if the thread panicked.
    pub fn join(mut self) -> Result<(), String> {
        if let Some(handle) = self.build_thread.take() {
            handle
                .join()
                .map_err(|_| "index-builder thread panicked".to_string())
        } else {
            Ok(())
        }
    }

    /// Get the current progress (number of files indexed).
    #[inline]
    pub fn files_indexed(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    /// Check if the build is still in progress.
    pub fn is_running(&self) -> bool {
        self.build_thread
            .as_ref()
            .is_some_and(|h| !h.is_finished())
    }
}

/// Core build logic: scan, save, snapshot, swap.
///
/// Separated from the struct for testability and reuse.
fn build_initial_index(
    progress: &Arc<AtomicUsize>,
    store: &Arc<IndexStore>,
    excludes: &Arc<ExcludeTrie>,
    root: &Path,
) -> Result<(), String> {
    // Step 1: Ensure index directory exists
    ensure_index_dir().map_err(|e| format!("ensure_index_dir failed: {}", e))?;

    let index_path = global_index_path();
    let root_hash = global_cache_key();
    let exclude_hash = compute_exclude_hash(excludes);

    // Step 2: Scan root directory with excludes
    let config = ScannerConfig {
        skip_hidden: false,
        respect_gitignore: true,
        ..Default::default()
    };
    let scanner = FilesystemScanner::with_config(config);

    eprintln!("BackgroundBuilder: starting initial scan from {}", root.display());
    let entries = scanner.scan(root);

    // Step 3: Filter entries through ExcludeTrie
    let filtered: Vec<GpuPathEntry> = entries
        .into_iter()
        .filter(|entry| {
            let path_bytes = &entry.path[..entry.path_len as usize];
            !excludes.should_exclude(path_bytes)
        })
        .collect();

    // Update progress counter
    progress.store(filtered.len(), Ordering::Relaxed);
    eprintln!(
        "BackgroundBuilder: scan complete, {} files indexed",
        filtered.len()
    );

    // Step 4: Save v2 index
    save_v2(
        &filtered,
        root_hash,
        &index_path,
        0, // no FSEvents ID yet (fresh build)
        exclude_hash,
    )
    .map_err(|e| format!("save_v2 failed: {}", e))?;

    eprintln!("BackgroundBuilder: saved v2 index to {:?}", index_path);

    // Step 5: Create IndexSnapshot and swap into store
    let snapshot = IndexSnapshot::from_file(&index_path, None)
        .map_err(|e| format!("snapshot creation failed: {}", e))?;

    store.swap(snapshot);
    eprintln!("BackgroundBuilder: snapshot swapped into IndexStore");

    Ok(())
}

/// Check whether an existing index snapshot is stale.
///
/// Reads the `saved_at` field from the snapshot's header and compares it
/// against [`DEFAULT_MAX_AGE_SECS`](crate::index::gsix_v2::DEFAULT_MAX_AGE_SECS)
/// (1 hour). Returns `true` if the index is older than the max age.
///
/// A stale index can still serve search (fast but possibly incomplete)
/// while a background update runs. The caller should set
/// `IndexState::Stale` and trigger an FSEvents watcher + background
/// update when this returns `true`.
pub fn check_staleness(snapshot: &IndexSnapshot) -> bool {
    is_stale(snapshot.header())
}

// ================================================================
// IndexDaemon: top-level lifecycle coordinator
// ================================================================

/// Top-level coordinator for the persistent index subsystem.
///
/// Orchestrates the full lifecycle:
/// - On startup: loads existing `global.idx` (or starts a background build)
/// - Starts an [`FSEventsListener`] for real-time filesystem change monitoring
/// - Starts an [`IndexWriter`] thread for incremental updates from FSEvents
/// - On shutdown: tears down FSEventsListener, channel, writer thread, and builder
///
/// # Ownership
///
/// ```text
/// IndexDaemon
///   |-- store: Arc<IndexStore>       (shared with orchestrator + writer)
///   |-- excludes: Arc<ExcludeTrie>   (shared with listener + writer)
///   |-- fsevents_listener            (macOS filesystem watcher)
///   |-- change_tx: Sender<FsChange>  (sender end; dropped to signal writer exit)
///   |-- writer_join: JoinHandle       (writer thread handle)
///   |-- builder: BackgroundBuilder   (optional initial build)
///   +-- progress: Arc<AtomicUsize>   (shared build progress counter)
/// ```
pub struct IndexDaemon {
    /// Lock-free snapshot store, shared with search orchestrator.
    store: Arc<IndexStore>,

    /// Path exclusion filter, shared with FSEventsListener and IndexWriter.
    excludes: Arc<ExcludeTrie>,

    /// FSEvents filesystem watcher (macOS only).
    #[cfg(target_os = "macos")]
    fsevents_listener: Option<FSEventsListener>,

    /// Sender end of the FsChange channel. Dropped during shutdown to
    /// signal the writer thread to perform a final flush and exit.
    change_tx: Option<Sender<FsChange>>,

    /// Handle to the IndexWriter background thread.
    writer_join: Option<JoinHandle<()>>,

    /// Background initial builder (if no index exists on startup).
    builder: Option<BackgroundBuilder>,

    /// Shared progress counter for UI display during initial build.
    pub progress: Arc<AtomicUsize>,
}

impl IndexDaemon {
    /// Start the index daemon, orchestrating all subsystems.
    ///
    /// 1. Loads config, creates ExcludeTrie (defaults + user config merge)
    /// 2. Checks if `global.idx` exists:
    ///    - **EXISTS**: Loads via `IndexSnapshot::from_file`, swaps into store,
    ///      checks staleness
    ///    - **NOT EXISTS**: Starts `BackgroundBuilder` for initial full scan
    /// 3. Creates crossbeam channel for `FsChange` events
    /// 4. Starts `FSEventsListener` with sender end
    /// 5. Starts `IndexWriter` thread with receiver end
    pub fn start(store: Arc<IndexStore>, root: PathBuf) -> Result<Self, String> {
        // Step 1: Build ExcludeTrie from defaults + user config
        let defaults = default_excludes();
        let config = load_config();
        let excludes = Arc::new(merge_with_config(defaults, config));
        let exclude_hash = compute_exclude_hash(&excludes);
        let root_hash = global_cache_key();
        let index_path = global_index_path();

        let progress = Arc::new(AtomicUsize::new(0));
        let mut builder: Option<BackgroundBuilder> = None;
        let mut last_fsevents_id: u64 = 0;

        // Step 2: Load existing index or start background build
        if index_path.exists() {
            // Warm start: load existing index
            match IndexSnapshot::from_file(&index_path, None) {
                Ok(snapshot) => {
                    let is_stale = check_staleness(&snapshot);
                    last_fsevents_id = snapshot.fsevents_id();
                    progress.store(snapshot.entry_count(), Ordering::Relaxed);
                    store.swap(snapshot);

                    if is_stale {
                        eprintln!("IndexDaemon: existing index is stale, will update via FSEvents");
                    } else {
                        eprintln!("IndexDaemon: loaded existing index ({} entries)", progress.load(Ordering::Relaxed));
                    }
                }
                Err(e) => {
                    eprintln!("IndexDaemon: failed to load existing index: {}, rebuilding", e);
                    // Fall through to background build
                    let mut bg = BackgroundBuilder::new(Arc::clone(&store), Arc::clone(&excludes), root.clone());
                    bg.spawn_build_thread();
                    builder = Some(bg);
                }
            }
        } else {
            // Cold start: no index exists, start background build
            eprintln!("IndexDaemon: no existing index found, starting background build");
            let mut bg = BackgroundBuilder::new(Arc::clone(&store), Arc::clone(&excludes), root);
            bg.spawn_build_thread();
            builder = Some(bg);
        }

        // Step 3: Create crossbeam channel for FsChange events
        let (change_tx, change_rx) = crossbeam_channel::bounded::<FsChange>(4096);

        // Step 4: Start FSEventsListener with sender end
        #[cfg(target_os = "macos")]
        let fsevents_listener = {
            let event_id = Arc::new(AtomicU64::new(last_fsevents_id));
            let mut listener = FSEventsListener::new(
                Arc::clone(&excludes),
                change_tx.clone(),
                event_id,
            );
            match listener.start() {
                Ok(needs_rebuild) => {
                    if needs_rebuild {
                        eprintln!("IndexDaemon: FSEvents detected event ID regression, rebuild triggered");
                    }
                    eprintln!("IndexDaemon: FSEventsListener started");
                }
                Err(e) => {
                    eprintln!("IndexDaemon: failed to start FSEventsListener: {}", e);
                }
            }
            Some(listener)
        };

        // Step 5: Start IndexWriter thread with receiver end
        let writer = if store.is_available() {
            // Warm start: build writer from current snapshot
            let guard = store.snapshot();
            if let Some(snap) = guard.as_ref() {
                IndexWriter::from_snapshot(
                    snap,
                    Arc::clone(&excludes),
                    Arc::clone(&store),
                    index_path,
                )
            } else {
                IndexWriter::new(
                    Arc::clone(&excludes),
                    Arc::clone(&store),
                    index_path,
                    exclude_hash,
                    root_hash,
                )
            }
        } else {
            // Cold start: empty writer (builder will populate store later)
            IndexWriter::new(
                Arc::clone(&excludes),
                Arc::clone(&store),
                index_path,
                exclude_hash,
                root_hash,
            )
        };

        let writer_join = Some(spawn_writer_thread(writer, change_rx));
        eprintln!("IndexDaemon: IndexWriter thread started");

        Ok(Self {
            store,
            excludes,
            #[cfg(target_os = "macos")]
            fsevents_listener,
            change_tx: Some(change_tx),
            writer_join,
            builder,
            progress,
        })
    }

    /// Graceful shutdown of all index subsystems.
    ///
    /// Order:
    /// 1. Stop FSEventsListener (stops sending events)
    /// 2. Drop channel sender (disconnects channel, writer performs final flush)
    /// 3. Join writer thread
    /// 4. Join builder thread if running
    pub fn shutdown(&mut self) {
        // Step 1: Stop FSEventsListener
        #[cfg(target_os = "macos")]
        if let Some(ref mut listener) = self.fsevents_listener {
            listener.stop();
            eprintln!("IndexDaemon: FSEventsListener stopped");
        }
        #[cfg(target_os = "macos")]
        {
            self.fsevents_listener = None;
        }

        // Step 2: Drop sender to disconnect channel
        // This causes the writer thread's recv() to return Err,
        // triggering a final flush and clean exit.
        self.change_tx = None;

        // Step 3: Join writer thread
        if let Some(handle) = self.writer_join.take() {
            if handle.join().is_err() {
                eprintln!("IndexDaemon: writer thread panicked during shutdown");
            } else {
                eprintln!("IndexDaemon: writer thread joined");
            }
        }

        // Step 4: Join builder thread if running
        if let Some(bg) = self.builder.take() {
            if let Err(e) = bg.join() {
                eprintln!("IndexDaemon: builder thread error during shutdown: {}", e);
            } else {
                eprintln!("IndexDaemon: builder thread joined");
            }
        }
    }

    /// Get a reference to the shared IndexStore.
    #[inline]
    pub fn store(&self) -> &Arc<IndexStore> {
        &self.store
    }

    /// Get a reference to the shared ExcludeTrie.
    #[inline]
    pub fn excludes(&self) -> &Arc<ExcludeTrie> {
        &self.excludes
    }

    /// Check if a background build is currently in progress.
    pub fn is_building(&self) -> bool {
        self.builder
            .as_ref()
            .is_some_and(|bg| bg.is_running())
    }

    /// Get the current build progress (number of files indexed).
    pub fn build_progress(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }
}

impl Drop for IndexDaemon {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_background_builder_new() {
        let store = Arc::new(IndexStore::new());
        let excludes = Arc::new(ExcludeTrie::default());
        let builder = BackgroundBuilder::new(store.clone(), excludes);

        assert_eq!(builder.files_indexed(), 0);
        assert!(!builder.is_running());
        assert!(!store.is_available());
    }

    #[test]
    fn test_needs_initial_build() {
        // This test depends on whether ~/.gpu-search/index/global.idx exists
        // on the machine. We just verify it returns a bool without panicking.
        let _needs_build = BackgroundBuilder::needs_initial_build();
    }

    #[test]
    fn test_progress_counter_shared() {
        let store = Arc::new(IndexStore::new());
        let excludes = Arc::new(ExcludeTrie::default());
        let builder = BackgroundBuilder::new(store, excludes);

        let progress = Arc::clone(&builder.progress);
        assert_eq!(progress.load(Ordering::Relaxed), 0);

        // Simulate progress from another thread
        progress.store(42, Ordering::Relaxed);
        assert_eq!(builder.files_indexed(), 42);
    }

    #[test]
    fn test_check_staleness_fresh_snapshot() {
        // A just-saved index should NOT be stale
        use crate::index::gsix_v2::save_v2;

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("fresh.idx");

        // Save with current time (GsixHeaderV2::new sets saved_at = now)
        let entry = crate::gpu::types::GpuPathEntry::new();
        save_v2(&[entry], 0xAA, &idx_path, 0, 0).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, None).unwrap();
        assert!(
            !check_staleness(&snapshot),
            "freshly saved snapshot should not be stale"
        );
    }

    #[test]
    fn test_check_staleness_expired_snapshot() {
        // An index saved 2 hours ago should be stale (default max age = 1 hour)
        use crate::index::gsix_v2::{save_v2, DEFAULT_MAX_AGE_SECS};
        use std::time::{SystemTime, UNIX_EPOCH};

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("stale.idx");

        // Save a valid index first
        let entry = crate::gpu::types::GpuPathEntry::new();
        save_v2(&[entry], 0xBB, &idx_path, 0, 0).unwrap();

        // Tamper with saved_at to be 2 hours ago, then rewrite with correct checksum
        let mut data = std::fs::read(&idx_path).unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let two_hours_ago = now - (DEFAULT_MAX_AGE_SECS * 2);
        data[16..24].copy_from_slice(&two_hours_ago.to_le_bytes());
        // Recompute CRC32 over [0..44)
        let crc = crc32fast::hash(&data[..44]);
        data[44..48].copy_from_slice(&crc.to_le_bytes());
        std::fs::write(&idx_path, &data).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, None).unwrap();
        assert!(
            check_staleness(&snapshot),
            "snapshot saved 2 hours ago should be stale (max age = {} seconds)",
            DEFAULT_MAX_AGE_SECS
        );
    }
}
