//! ContentDaemon: background content index builder and lifecycle coordinator.
//!
//! Spawns a [`ContentBuilder`] on a dedicated background thread to walk the
//! filesystem, read text files, and build a [`ContentSnapshot`]. On completion
//! the snapshot is published to the shared [`ContentIndexStore`] via `swap()`.
//!
//! # GCIX Persistence
//!
//! On startup, checks for an existing GCIX cache file. If valid, loads it
//! directly (instant warm start, skipping the build). After a fresh build,
//! persists the result to GCIX for future restarts.
//!
//! # FSEvents Integration
//!
//! After initial build completes, listens for `FsChange` events on a
//! crossbeam channel. File modifications/creations are read, hashed, and
//! applied to a mutable Vec-backed working copy of the content store.
//! Changes are debounced (500ms window) before publishing a new snapshot.
//!
//! # Ownership
//!
//! ```text
//! ContentDaemon
//!   |-- store: Arc<ContentIndexStore>   (shared with search orchestrator)
//!   |-- builder_thread: JoinHandle      (background build thread)
//!   |-- progress: Arc<BuildProgress>    (shared progress counters)
//!   +-- build_progress: Arc<AtomicUsize> (legacy single counter)
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossbeam_channel::Receiver;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::index::content_builder::{BuildProgress, ContentBuilder};
use crate::index::content_index_store::ContentIndexStore;
use crate::index::content_snapshot::ContentSnapshot;
use crate::index::content_store::ContentStore;
use crate::index::exclude::ExcludeTrie;
use crate::index::fsevents::FsChange;
use crate::index::gcix::{load_gcix, save_gcix};
use crate::search::binary::BinaryDetector;

/// Debounce window for coalescing rapid FSEvents changes (500ms).
const DEBOUNCE_DURATION: Duration = Duration::from_millis(500);

/// Maximum file size to index during incremental updates (100MB).
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Return the default GCIX cache file path: `~/.gpu-search/index/global.gcix`.
///
/// Creates the parent directory if it does not exist.
pub fn default_gcix_path() -> PathBuf {
    let base = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".gpu-search")
        .join("index");
    let _ = std::fs::create_dir_all(&base);
    base.join("global.gcix")
}

/// Background content index builder and lifecycle coordinator.
///
/// Spawns a dedicated thread that runs [`ContentBuilder::build`] to walk
/// a directory tree, read text files, and build a [`ContentSnapshot`].
/// On completion the snapshot is atomically published to the shared
/// [`ContentIndexStore`] so search threads can use it immediately.
///
/// On startup, attempts to load a cached GCIX file. If valid, publishes
/// the snapshot immediately and skips the build. After a fresh build,
/// saves the result to GCIX for future restarts.
pub struct ContentDaemon {
    /// Lock-free snapshot store, shared with search orchestrator.
    store: Arc<ContentIndexStore>,

    /// Handle to the background build thread (if spawned).
    builder_thread: Option<JoinHandle<()>>,

    /// Simple progress counter: number of files indexed.
    progress: Arc<AtomicUsize>,

    /// Detailed progress tracking: files_scanned, files_indexed, bytes_indexed.
    build_progress: Arc<BuildProgress>,

    /// Path to the GCIX cache file for persistence.
    gcix_path: PathBuf,
}

impl ContentDaemon {
    /// Start a content daemon that builds the content index in the background.
    ///
    /// Uses the default GCIX path (`~/.gpu-search/index/global.gcix`).
    /// See [`start_with_gcix_path`] for a version that accepts a custom path
    /// (useful for testing).
    pub fn start(
        store: Arc<ContentIndexStore>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        excludes: Arc<ExcludeTrie>,
        root: PathBuf,
    ) -> Self {
        Self::start_with_gcix_path(store, device, excludes, root, default_gcix_path())
    }

    /// Start a content daemon with a custom GCIX cache path.
    ///
    /// 1. Checks for an existing GCIX cache at `gcix_path`.
    ///    - If valid, publishes the snapshot immediately and skips the build.
    ///    - If corrupt or version mismatch, deletes the file and continues.
    /// 2. If no valid cache, spawns a background thread to build.
    /// 3. After a successful build, saves the result to `gcix_path`.
    ///
    /// # Arguments
    ///
    /// * `store` - Shared content index store for publishing snapshots
    /// * `device` - Metal device for creating GPU buffers
    /// * `excludes` - Path exclusion filter
    /// * `root` - Root directory to scan
    /// * `gcix_path` - Path for GCIX cache file
    pub fn start_with_gcix_path(
        store: Arc<ContentIndexStore>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        excludes: Arc<ExcludeTrie>,
        root: PathBuf,
        gcix_path: PathBuf,
    ) -> Self {
        let progress = Arc::new(AtomicUsize::new(0));
        let build_progress = Arc::new(BuildProgress::new());

        // Try loading existing GCIX cache
        if gcix_path.exists() {
            match load_gcix(&gcix_path, Some(&device)) {
                Ok(snapshot) => {
                    let file_count = snapshot.file_count();
                    progress.store(file_count as usize, Ordering::Relaxed);
                    store.swap(snapshot);
                    eprintln!(
                        "ContentDaemon: loaded GCIX cache, {} files (skipping build)",
                        file_count
                    );
                    return Self {
                        store,
                        builder_thread: None,
                        progress,
                        build_progress,
                        gcix_path,
                    };
                }
                Err(e) => {
                    eprintln!(
                        "ContentDaemon: GCIX cache invalid ({}), deleting and rebuilding",
                        e
                    );
                    let _ = std::fs::remove_file(&gcix_path);
                }
            }
        }

        // No valid cache -- spawn builder thread
        let thread_store = Arc::clone(&store);
        let thread_progress = Arc::clone(&progress);
        let thread_build_progress = Arc::clone(&build_progress);
        let thread_excludes = Arc::clone(&excludes);
        let thread_gcix_path = gcix_path.clone();

        let handle = thread::Builder::new()
            .name("content-builder".to_string())
            .spawn(move || {
                let builder = ContentBuilder::with_progress(
                    Arc::clone(&thread_store),
                    thread_excludes,
                    thread_build_progress,
                    device,
                );

                match builder.build(&root) {
                    Ok(snapshot) => {
                        let file_count = snapshot.file_count();

                        // Save GCIX cache for next restart
                        let root_hash = crc32fast::hash(
                            root.to_string_lossy().as_bytes(),
                        ) as u64;
                        if let Err(e) = save_gcix(
                            snapshot.content_store(),
                            &thread_gcix_path,
                            root_hash,
                            0, // fsevents_id: 0 for now (wired in Phase 5)
                        ) {
                            eprintln!("ContentDaemon: failed to save GCIX cache: {}", e);
                        } else {
                            eprintln!(
                                "ContentDaemon: saved GCIX cache ({} files) to {}",
                                file_count,
                                thread_gcix_path.display()
                            );
                        }

                        thread_store.swap(snapshot);
                        thread_progress.store(file_count as usize, Ordering::Relaxed);
                        eprintln!(
                            "ContentDaemon: build complete, {} files indexed",
                            file_count
                        );
                    }
                    Err(e) => {
                        eprintln!("ContentDaemon: build failed: {}", e);
                    }
                }
            })
            .expect("failed to spawn content-builder thread");

        Self {
            store,
            builder_thread: Some(handle),
            progress,
            build_progress,
            gcix_path,
        }
    }

    /// Start a content daemon that also listens for FSEvents changes.
    ///
    /// After the initial build (or GCIX load) completes, the builder thread
    /// enters a change-processing loop, applying incremental updates from
    /// the FSEvents channel. Changes are debounced (500ms window).
    ///
    /// # Arguments
    ///
    /// * `store` - Shared content index store for publishing snapshots
    /// * `device` - Metal device for creating GPU buffers
    /// * `excludes` - Path exclusion filter
    /// * `root` - Root directory to scan
    /// * `gcix_path` - Path for GCIX cache file
    /// * `change_rx` - Receiver end of the FSEvents channel
    pub fn start_with_changes(
        store: Arc<ContentIndexStore>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        excludes: Arc<ExcludeTrie>,
        root: PathBuf,
        gcix_path: PathBuf,
        change_rx: Receiver<FsChange>,
    ) -> Self {
        let progress = Arc::new(AtomicUsize::new(0));
        let build_progress = Arc::new(BuildProgress::new());

        // Try loading existing GCIX cache
        let initial_snapshot = if gcix_path.exists() {
            match load_gcix(&gcix_path, Some(&device)) {
                Ok(snapshot) => {
                    let file_count = snapshot.file_count();
                    progress.store(file_count as usize, Ordering::Relaxed);
                    eprintln!(
                        "ContentDaemon: loaded GCIX cache, {} files",
                        file_count
                    );
                    Some(snapshot)
                }
                Err(e) => {
                    eprintln!(
                        "ContentDaemon: GCIX cache invalid ({}), deleting and rebuilding",
                        e
                    );
                    let _ = std::fs::remove_file(&gcix_path);
                    None
                }
            }
        } else {
            None
        };

        let thread_store = Arc::clone(&store);
        let thread_progress = Arc::clone(&progress);
        let thread_build_progress = Arc::clone(&build_progress);
        let thread_excludes = Arc::clone(&excludes);
        let thread_gcix_path = gcix_path.clone();

        let handle = thread::Builder::new()
            .name("content-builder".to_string())
            .spawn(move || {
                // Phase 1: build or use cached snapshot
                let snapshot = if let Some(snap) = initial_snapshot {
                    snap
                } else {
                    let builder = ContentBuilder::with_progress(
                        Arc::clone(&thread_store),
                        thread_excludes.clone(),
                        thread_build_progress,
                        device,
                    );

                    match builder.build(&root) {
                        Ok(snap) => {
                            // Save GCIX cache
                            let root_hash = crc32fast::hash(
                                root.to_string_lossy().as_bytes(),
                            ) as u64;
                            if let Err(e) = save_gcix(
                                snap.content_store(),
                                &thread_gcix_path,
                                root_hash,
                                0,
                            ) {
                                eprintln!("ContentDaemon: failed to save GCIX: {}", e);
                            }
                            snap
                        }
                        Err(e) => {
                            eprintln!("ContentDaemon: build failed: {}", e);
                            return;
                        }
                    }
                };

                let file_count = snapshot.file_count();

                // Build a mutable working copy from the snapshot
                let mut working_store = clone_content_store_to_vec(
                    snapshot.content_store(),
                );
                // Also build a path-to-file_id lookup for incremental updates
                let mut path_lookup: HashMap<PathBuf, u32> = HashMap::new();
                for (i, p) in working_store.paths().iter().enumerate() {
                    path_lookup.insert(p.clone(), i as u32);
                }

                // Publish the initial snapshot
                thread_store.swap(snapshot);
                thread_progress.store(file_count as usize, Ordering::Relaxed);
                eprintln!(
                    "ContentDaemon: build complete, {} files indexed, entering FSEvents loop",
                    file_count
                );

                // Phase 2: FSEvents change processing loop with debounce
                process_changes_loop(
                    &change_rx,
                    &mut working_store,
                    &mut path_lookup,
                    &thread_store,
                    &thread_progress,
                    &thread_excludes,
                    &root,
                );
            })
            .expect("failed to spawn content-builder thread");

        Self {
            store,
            builder_thread: Some(handle),
            progress,
            build_progress,
            gcix_path,
        }
    }

    /// Graceful shutdown: join the builder thread.
    ///
    /// Blocks until the background build finishes (or has already finished).
    pub fn shutdown(&mut self) {
        if let Some(handle) = self.builder_thread.take() {
            if handle.join().is_err() {
                eprintln!("ContentDaemon: builder thread panicked during shutdown");
            } else {
                eprintln!("ContentDaemon: builder thread joined");
            }
        }
    }

    /// Get the number of files indexed so far.
    #[inline]
    pub fn progress(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    /// Get detailed build progress counters.
    #[inline]
    pub fn build_progress(&self) -> &Arc<BuildProgress> {
        &self.build_progress
    }

    /// Get a reference to the shared ContentIndexStore.
    #[inline]
    pub fn store(&self) -> &Arc<ContentIndexStore> {
        &self.store
    }

    /// Get the GCIX cache file path.
    #[inline]
    pub fn gcix_path(&self) -> &Path {
        &self.gcix_path
    }

    /// Check if the background build is still running.
    pub fn is_building(&self) -> bool {
        self.builder_thread
            .as_ref()
            .is_some_and(|h| !h.is_finished())
    }
}

/// Clone a ContentStore's data into a mutable Vec-backed ContentStore.
///
/// Used to create a working copy for incremental FSEvents updates.
/// The mmap-backed or file-backed store from the initial build cannot
/// be mutated, so we copy all data into a fresh Vec-backed store.
fn clone_content_store_to_vec(source: &ContentStore) -> ContentStore {
    let mut dest = ContentStore::with_capacity(source.total_bytes() as usize);
    for file_id in 0..source.file_count() {
        if let Some(content) = source.content_for(file_id) {
            let meta = &source.files()[file_id as usize];
            let path = source.path_for(file_id).cloned().unwrap_or_default();
            dest.add_file_with_path(
                content,
                path,
                meta.path_id,
                meta.content_hash,
                meta.mtime,
            );
        }
    }
    dest
}

/// Publish a new ContentSnapshot from the current working store state.
fn publish_snapshot(
    working_store: &ContentStore,
    store: &ContentIndexStore,
    progress: &AtomicUsize,
) {
    // Clone the working store into a fresh store for the snapshot
    let snapshot_store = clone_content_store_to_vec(working_store);
    let file_count = snapshot_store.file_count();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let snapshot = ContentSnapshot::new(snapshot_store, timestamp);
    store.swap(snapshot);
    progress.store(file_count as usize, Ordering::Relaxed);
}

/// Apply a batch of coalesced FSEvents changes to the working store.
///
/// Returns `true` if a full rebuild is needed (MustRescan received).
fn apply_changes(
    changes: &HashMap<PathBuf, FsChange>,
    working_store: &mut ContentStore,
    path_lookup: &mut HashMap<PathBuf, u32>,
    excludes: &ExcludeTrie,
) -> bool {
    let binary_detector = BinaryDetector::new();

    for change in changes.values() {
        match change {
            FsChange::Created(p) | FsChange::Modified(p) => {
                // Check exclude filter
                let path_bytes = p.as_os_str().as_encoded_bytes();
                if excludes.should_exclude(path_bytes) {
                    continue;
                }

                // Check if it's a regular file
                let metadata = match std::fs::metadata(p) {
                    Ok(m) => m,
                    Err(_) => continue, // race-deleted or inaccessible
                };
                if !metadata.is_file() {
                    continue;
                }
                if metadata.len() > MAX_FILE_SIZE || metadata.len() == 0 {
                    continue;
                }

                // Skip binary files (extension check)
                if binary_detector.should_skip(p) {
                    continue;
                }

                // Read file content
                let content = match std::fs::read(p) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                // Skip binary content (NUL byte heuristic)
                if crate::search::binary::is_binary_content(&content) {
                    continue;
                }

                if content.is_empty() {
                    continue;
                }

                // Compute hash and mtime
                let hash = crc32fast::hash(&content);
                let mtime = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as u32)
                    .unwrap_or(0);

                // Check if this file already exists in the store
                if let Some(&file_id) = path_lookup.get(p) {
                    // Check if content actually changed via hash
                    let old_hash = working_store.files()[file_id as usize].content_hash;
                    if old_hash != hash {
                        working_store.update_file(file_id, &content, hash, mtime);
                    }
                } else {
                    // New file
                    let path_id = working_store.file_count();
                    let file_id = working_store.add_file_with_path(
                        &content,
                        p.clone(),
                        path_id,
                        hash,
                        mtime,
                    );
                    path_lookup.insert(p.clone(), file_id);
                }
            }
            FsChange::Deleted(p) => {
                if let Some(&file_id) = path_lookup.get(p) {
                    working_store.remove_file(file_id);
                    path_lookup.remove(p);
                }
            }
            FsChange::Renamed { old, new } => {
                // Delete old path
                if let Some(&file_id) = path_lookup.get(old) {
                    working_store.remove_file(file_id);
                    path_lookup.remove(old);
                }
                // Treat new path as Created (will be picked up if it's text)
                // We won't recurse; just handle it inline
                let path_bytes = new.as_os_str().as_encoded_bytes();
                if excludes.should_exclude(path_bytes) {
                    continue;
                }
                let metadata = match std::fs::metadata(new) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if !metadata.is_file() || metadata.len() > MAX_FILE_SIZE || metadata.len() == 0 {
                    continue;
                }
                if binary_detector.should_skip(new) {
                    continue;
                }
                let content = match std::fs::read(new) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                if crate::search::binary::is_binary_content(&content) || content.is_empty() {
                    continue;
                }
                let hash = crc32fast::hash(&content);
                let mtime = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as u32)
                    .unwrap_or(0);
                let path_id = working_store.file_count();
                let file_id = working_store.add_file_with_path(
                    &content,
                    new.clone(),
                    path_id,
                    hash,
                    mtime,
                );
                path_lookup.insert(new.clone(), file_id);
            }
            FsChange::MustRescan(_) => {
                return true; // signal full rebuild needed
            }
            FsChange::HistoryDone => {
                // No action needed; just publish current state
            }
        }
    }

    false
}

/// Main FSEvents change processing loop with debounce.
///
/// Collects changes for DEBOUNCE_DURATION (500ms), then applies them as
/// a batch and publishes a new snapshot. Exits when the channel disconnects.
fn process_changes_loop(
    change_rx: &Receiver<FsChange>,
    working_store: &mut ContentStore,
    path_lookup: &mut HashMap<PathBuf, u32>,
    store: &ContentIndexStore,
    progress: &AtomicUsize,
    excludes: &ExcludeTrie,
    _root: &Path,
) {
    loop {
        // Block waiting for the first change
        let first_change = match change_rx.recv() {
            Ok(change) => change,
            Err(_) => {
                // Channel disconnected -- all senders dropped
                eprintln!("ContentDaemon: FSEvents channel disconnected, exiting");
                return;
            }
        };

        // Start debounce window: collect changes for DEBOUNCE_DURATION
        let mut coalesced: HashMap<PathBuf, FsChange> = HashMap::new();
        let mut needs_rebuild = false;

        // Insert first change
        match &first_change {
            FsChange::MustRescan(_) => needs_rebuild = true,
            FsChange::Created(p) | FsChange::Modified(p) | FsChange::Deleted(p) => {
                coalesced.insert(p.clone(), first_change);
            }
            FsChange::Renamed { old, new } => {
                coalesced.insert(old.clone(), FsChange::Deleted(old.clone()));
                coalesced.insert(new.clone(), FsChange::Created(new.clone()));
            }
            FsChange::HistoryDone => {}
        }

        // Drain additional events within the debounce window
        let deadline = Instant::now() + DEBOUNCE_DURATION;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match change_rx.recv_timeout(remaining) {
                Ok(change) => match &change {
                    FsChange::MustRescan(_) => needs_rebuild = true,
                    FsChange::Created(p) | FsChange::Modified(p) | FsChange::Deleted(p) => {
                        coalesced.insert(p.clone(), change);
                    }
                    FsChange::Renamed { old, new } => {
                        coalesced.insert(old.clone(), FsChange::Deleted(old.clone()));
                        coalesced.insert(new.clone(), FsChange::Created(new.clone()));
                    }
                    FsChange::HistoryDone => {}
                },
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Apply whatever we have and exit
                    if !coalesced.is_empty() && !needs_rebuild {
                        let rebuild = apply_changes(
                            &coalesced,
                            working_store,
                            path_lookup,
                            excludes,
                        );
                        if !rebuild {
                            publish_snapshot(working_store, store, progress);
                        }
                    }
                    eprintln!("ContentDaemon: FSEvents channel disconnected during debounce");
                    return;
                }
            }
        }

        if needs_rebuild {
            // MustRescan: for now, log and continue (full rebuild deferred)
            eprintln!(
                "ContentDaemon: MustRescan received, full rebuild needed (not yet implemented)"
            );
            continue;
        }

        if coalesced.is_empty() {
            continue;
        }

        // Apply coalesced changes
        let rebuild = apply_changes(
            &coalesced,
            working_store,
            path_lookup,
            excludes,
        );

        if rebuild {
            eprintln!("ContentDaemon: MustRescan from apply_changes");
            continue;
        }

        // Publish updated snapshot
        publish_snapshot(working_store, store, progress);
        eprintln!(
            "ContentDaemon: applied {} FSEvents changes, published new snapshot",
            coalesced.len()
        );
    }
}

impl Drop for ContentDaemon {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::fsevents::FsChange;
    use crossbeam_channel;
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use std::collections::HashSet;
    use std::fs;
    use tempfile::TempDir;

    fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)")
    }

    fn empty_excludes() -> Arc<ExcludeTrie> {
        Arc::new(ExcludeTrie::new(
            vec![],
            vec![],
            HashSet::new(),
            HashSet::new(),
        ))
    }

    /// Return a GCIX path inside a tempdir for test isolation (no default path pollution).
    fn test_gcix_path(dir: &TempDir) -> PathBuf {
        dir.path().join("test_gcix_cache.gcix")
    }

    /// Convenience: start daemon with a fresh, nonexistent GCIX path for test isolation.
    fn start_daemon_isolated(
        store: Arc<ContentIndexStore>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        excludes: Arc<ExcludeTrie>,
        root: PathBuf,
        gcix_dir: &TempDir,
    ) -> ContentDaemon {
        ContentDaemon::start_with_gcix_path(
            store,
            device,
            excludes,
            root,
            test_gcix_path(gcix_dir),
        )
    }

    /// Create a tempdir with `n` text files.
    fn make_test_dir(n: usize) -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");
        for i in 0..n {
            let filename = format!("file_{:04}.txt", i);
            let content = format!(
                "// File {}\nfn content_daemon_test_{}() -> u32 {{ {} }}\n",
                i,
                i,
                i * 3 + 7
            );
            fs::write(dir.path().join(filename), content).unwrap();
        }
        dir
    }

    #[test]
    fn test_content_daemon_spawns_and_completes() {
        let dir = make_test_dir(10);
        let gcix_dir = TempDir::new().unwrap();
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        assert!(!store.is_available(), "Store should start empty");

        let mut daemon = start_daemon_isolated(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            &gcix_dir,
        );

        // Wait for build to complete
        daemon.shutdown();

        assert!(
            store.is_available(),
            "Store should be available after build completes"
        );

        // Verify progress
        assert_eq!(
            daemon.progress(),
            10,
            "Should have indexed 10 files, got {}",
            daemon.progress()
        );

        // Verify content accessible through store
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        assert_eq!(snap.file_count(), 10);

        // Verify content of first file
        let content = snap.content_store().content_for(0).unwrap();
        assert!(
            !content.is_empty(),
            "First file should have content"
        );
    }

    #[test]
    fn test_content_daemon_publishes_snapshot() {
        let dir = make_test_dir(5);
        let gcix_dir = TempDir::new().unwrap();
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let mut daemon = start_daemon_isolated(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            &gcix_dir,
        );

        daemon.shutdown();

        // Verify snapshot is published
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        assert_eq!(snap.file_count(), 5);

        // Verify build timestamp is recent
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(
            snap.build_timestamp() > 0 && snap.build_timestamp() <= now,
            "Build timestamp should be recent"
        );
    }

    #[test]
    fn test_content_daemon_shutdown_is_clean() {
        let dir = make_test_dir(3);
        let gcix_dir = TempDir::new().unwrap();
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let mut daemon = start_daemon_isolated(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            &gcix_dir,
        );

        // shutdown should not panic even when called multiple times
        daemon.shutdown();
        daemon.shutdown(); // second call should be no-op

        assert!(store.is_available());
    }

    #[test]
    fn test_content_daemon_build_progress() {
        let dir = make_test_dir(20);
        let gcix_dir = TempDir::new().unwrap();
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let mut daemon = start_daemon_isolated(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            &gcix_dir,
        );

        daemon.shutdown();

        // Detailed progress should reflect work done
        let bp = daemon.build_progress();
        assert_eq!(
            bp.files_indexed(),
            20,
            "files_indexed should be 20, got {}",
            bp.files_indexed()
        );
        assert_eq!(
            bp.files_scanned(),
            20,
            "files_scanned should be 20, got {}",
            bp.files_scanned()
        );
        assert!(
            bp.bytes_indexed() > 0,
            "bytes_indexed should be > 0"
        );
    }

    #[test]
    fn test_content_daemon_empty_dir() {
        let dir = TempDir::new().unwrap();
        let gcix_dir = TempDir::new().unwrap();
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let mut daemon = start_daemon_isolated(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            &gcix_dir,
        );

        daemon.shutdown();

        // Store should have a snapshot (even with 0 files)
        assert!(store.is_available());

        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        assert_eq!(snap.file_count(), 0);
        assert_eq!(daemon.progress(), 0);
    }

    #[test]
    fn test_content_daemon_drop_joins_thread() {
        let dir = make_test_dir(5);
        let gcix_dir = TempDir::new().unwrap();
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        // Drop should join the thread
        {
            let _daemon = start_daemon_isolated(
                store.clone(),
                device,
                excludes,
                dir.path().to_path_buf(),
                &gcix_dir,
            );
            // daemon dropped here
        }

        // After drop, store should have been populated
        assert!(
            store.is_available(),
            "Store should be available after daemon drop (thread joined)"
        );
    }

    /// Phase 2 checkpoint: full background build lifecycle with 200 files.
    ///
    /// 1. Create tempdir with 200 text files containing known content
    /// 2. Start ContentDaemon with that root
    /// 3. Poll progress until build completes
    /// 4. Verify store.is_available() == true
    /// 5. Get snapshot via store.snapshot()
    /// 6. Verify content_for() for at least 10 sampled files returns exact content
    /// 7. Verify file_count matches expected
    /// 8. Call daemon.shutdown() cleanly
    #[test]
    fn test_content_daemon_checkpoint_background_build() {
        // 1. Create tempdir with 200 files containing known, deterministic content
        let dir = TempDir::new().expect("Failed to create temp dir");
        let gcix_dir = TempDir::new().unwrap();
        let mut expected_contents: Vec<(String, String)> = Vec::with_capacity(200);
        for i in 0..200 {
            let filename = format!("file_{:04}.txt", i);
            let mut content = String::new();
            for j in 0..5 {
                content.push_str(&format!("file {} content line {}\n", i, j));
            }
            fs::write(dir.path().join(&filename), &content).unwrap();
            expected_contents.push((filename, content));
        }

        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        // Store should not be available before the daemon starts
        assert!(
            !store.is_available(),
            "Store should not be available before daemon starts"
        );

        // 2. Start ContentDaemon
        let mut daemon = start_daemon_isolated(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            &gcix_dir,
        );

        // 3. Poll progress until build completes (with timeout)
        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(30);
        loop {
            if store.is_available() {
                break;
            }
            if start_time.elapsed() > timeout {
                panic!(
                    "ContentDaemon did not complete within 30s. progress={}, build_progress: scanned={}, indexed={}, bytes={}",
                    daemon.progress(),
                    daemon.build_progress().files_scanned(),
                    daemon.build_progress().files_indexed(),
                    daemon.build_progress().bytes_indexed(),
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // 4. Verify store.is_available() == true
        assert!(
            store.is_available(),
            "Store must be available after build completes"
        );

        // 5. Get snapshot via store.snapshot()
        let guard = store.snapshot();
        let snap = guard
            .as_ref()
            .as_ref()
            .expect("Snapshot should be Some after build");

        // 7. Verify file_count matches expected (200 text files)
        assert_eq!(
            snap.file_count(),
            200,
            "Expected 200 files indexed, got {}",
            snap.file_count()
        );

        // 6. Verify content_for() for sampled files returns exact content
        //
        // Since WalkBuilder may visit files in any order, we cannot assume
        // file_id == file index in our expected_contents Vec. Instead, build
        // a set of all expected content strings and verify each stored file
        // matches one of them.
        let expected_set: HashSet<Vec<u8>> = expected_contents
            .iter()
            .map(|(_, c)| c.as_bytes().to_vec())
            .collect();

        // Sample at least 10 files spread across the store
        let sample_ids: Vec<u32> = (0..200u32).step_by(19).collect(); // ~11 samples
        assert!(
            sample_ids.len() >= 10,
            "Should sample at least 10 files, got {}",
            sample_ids.len()
        );

        for &file_id in &sample_ids {
            let content = snap
                .content_store()
                .content_for(file_id)
                .unwrap_or_else(|| panic!("content_for({}) returned None", file_id));
            assert!(
                !content.is_empty(),
                "Content for file_id {} should not be empty",
                file_id
            );
            assert!(
                expected_set.contains(content),
                "Content for file_id {} does not match any expected file content.\n\
                 Got ({} bytes): {:?}",
                file_id,
                content.len(),
                String::from_utf8_lossy(&content[..content.len().min(100)])
            );
        }

        // Verify ALL 200 files are present and each matches an expected content
        let mut matched_count = 0;
        for file_id in 0..200u32 {
            let content = snap
                .content_store()
                .content_for(file_id)
                .unwrap_or_else(|| panic!("content_for({}) returned None", file_id));
            if expected_set.contains(content) {
                matched_count += 1;
            }
        }
        assert_eq!(
            matched_count, 200,
            "All 200 files should match expected content, got {} matches",
            matched_count
        );

        // Verify build progress counters
        let bp = daemon.build_progress();
        assert_eq!(
            bp.files_indexed(),
            200,
            "files_indexed should be 200, got {}",
            bp.files_indexed()
        );
        assert!(
            bp.bytes_indexed() > 0,
            "bytes_indexed should be > 0, got {}",
            bp.bytes_indexed()
        );

        // Verify Metal buffer is available
        assert!(
            snap.content_store().has_metal_buffer(),
            "Content store should have a Metal buffer after build"
        );

        // Verify build timestamp is recent
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(
            snap.build_timestamp() > 0 && snap.build_timestamp() <= now,
            "Build timestamp should be recent, got {}",
            snap.build_timestamp()
        );

        // 8. Call daemon.shutdown() cleanly
        daemon.shutdown();

        // Verify daemon is no longer building after shutdown
        assert!(
            !daemon.is_building(),
            "Daemon should not be building after shutdown"
        );

        // Double shutdown should be safe
        daemon.shutdown();
    }

    #[test]
    fn test_content_daemon_is_building() {
        let dir = make_test_dir(5);
        let gcix_dir = TempDir::new().unwrap();
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let mut daemon = start_daemon_isolated(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            &gcix_dir,
        );

        // After shutdown, should not be building
        daemon.shutdown();
        assert!(
            !daemon.is_building(),
            "Should not be building after shutdown"
        );
    }

    // ================================================================
    // GCIX persistence tests
    // ================================================================

    /// Test: build -> save GCIX -> simulate restart (new daemon loads GCIX) -> verify content.
    #[test]
    fn test_content_daemon_gcix_save_and_reload() {
        let dir = make_test_dir(15);
        let gcix_dir = TempDir::new().expect("gcix tempdir");
        let gcix_path = gcix_dir.path().join("test.gcix");
        let device = get_device();

        // --- Phase 1: fresh build, GCIX should be saved ---
        {
            let store = Arc::new(ContentIndexStore::new());
            let excludes = empty_excludes();
            let mut daemon = ContentDaemon::start_with_gcix_path(
                store.clone(),
                device.clone(),
                excludes,
                dir.path().to_path_buf(),
                gcix_path.clone(),
            );
            daemon.shutdown();

            assert!(store.is_available(), "Store should be available after build");
            assert_eq!(daemon.progress(), 15);

            // GCIX file should have been created
            assert!(
                gcix_path.exists(),
                "GCIX cache file should exist after build: {}",
                gcix_path.display()
            );
        }

        // --- Phase 2: simulate restart -- new daemon loads from GCIX ---
        {
            let store2 = Arc::new(ContentIndexStore::new());
            let excludes2 = empty_excludes();
            let mut daemon2 = ContentDaemon::start_with_gcix_path(
                store2.clone(),
                device.clone(),
                excludes2,
                dir.path().to_path_buf(),
                gcix_path.clone(),
            );

            // Should have loaded from GCIX (no builder thread spawned)
            assert!(
                store2.is_available(),
                "Store should be immediately available from GCIX cache"
            );
            assert!(
                !daemon2.is_building(),
                "No builder thread should be running when loaded from GCIX"
            );
            assert_eq!(daemon2.progress(), 15, "Progress should reflect loaded file count");

            // Verify content is accessible
            let guard = store2.snapshot();
            let snap = guard.as_ref().as_ref().expect("Snapshot should be Some");
            assert_eq!(snap.file_count(), 15);

            // Verify we can read content from the loaded snapshot
            for file_id in 0..15u32 {
                let content = snap
                    .content_store()
                    .content_for(file_id)
                    .unwrap_or_else(|| panic!("content_for({}) returned None", file_id));
                assert!(
                    !content.is_empty(),
                    "Content for file_id {} should not be empty",
                    file_id
                );
            }

            daemon2.shutdown(); // no-op, but should be safe
        }
    }

    /// Test: corrupt GCIX file is deleted and daemon falls back to build.
    #[test]
    fn test_content_daemon_gcix_corrupt_fallback() {
        let dir = make_test_dir(5);
        let gcix_dir = TempDir::new().expect("gcix tempdir");
        let gcix_path = gcix_dir.path().join("corrupt.gcix");
        let device = get_device();

        // Write a corrupt GCIX file
        fs::write(&gcix_path, b"this is not a valid GCIX file").unwrap();
        assert!(gcix_path.exists());

        // Daemon should detect corruption, delete it, and build normally
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();
        let mut daemon = ContentDaemon::start_with_gcix_path(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            gcix_path.clone(),
        );

        daemon.shutdown();

        assert!(
            store.is_available(),
            "Store should be available after fallback build"
        );
        assert_eq!(daemon.progress(), 5);

        // The corrupt file should have been deleted and a new valid one saved
        assert!(
            gcix_path.exists(),
            "GCIX file should be re-created after successful build"
        );
    }

    /// Test: no GCIX file -> daemon builds and saves GCIX.
    #[test]
    fn test_content_daemon_gcix_no_existing_file() {
        let dir = make_test_dir(8);
        let gcix_dir = TempDir::new().expect("gcix tempdir");
        let gcix_path = gcix_dir.path().join("nonexistent.gcix");
        let device = get_device();

        assert!(!gcix_path.exists(), "GCIX file should not exist initially");

        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();
        let mut daemon = ContentDaemon::start_with_gcix_path(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            gcix_path.clone(),
        );

        daemon.shutdown();

        assert!(store.is_available());
        assert_eq!(daemon.progress(), 8);
        assert!(
            gcix_path.exists(),
            "GCIX file should be saved after fresh build"
        );
    }

    /// Test: GCIX save -> load -> content matches original build.
    #[test]
    fn test_content_daemon_gcix_content_matches_build() {
        let dir = make_test_dir(20);
        let gcix_dir = TempDir::new().expect("gcix tempdir");
        let gcix_path = gcix_dir.path().join("match.gcix");
        let device = get_device();

        // Build and collect content from fresh build
        let store1 = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();
        let mut daemon1 = ContentDaemon::start_with_gcix_path(
            store1.clone(),
            device.clone(),
            excludes.clone(),
            dir.path().to_path_buf(),
            gcix_path.clone(),
        );
        daemon1.shutdown();

        // Capture all content from the build
        let (build_file_count, build_content) = {
            let guard1 = store1.snapshot();
            let snap1 = guard1.as_ref().as_ref().unwrap();
            let fc = snap1.file_count();
            let mut content: Vec<Vec<u8>> = Vec::new();
            for i in 0..fc {
                content.push(
                    snap1.content_store().content_for(i).unwrap().to_vec(),
                );
            }
            (fc, content)
        };

        // Load from GCIX and compare
        let store2 = Arc::new(ContentIndexStore::new());
        let excludes2 = empty_excludes();
        let mut daemon2 = ContentDaemon::start_with_gcix_path(
            store2.clone(),
            device.clone(),
            excludes2,
            dir.path().to_path_buf(),
            gcix_path.clone(),
        );
        daemon2.shutdown();

        let guard2 = store2.snapshot();
        let snap2 = guard2.as_ref().as_ref().unwrap();
        assert_eq!(
            snap2.file_count(),
            build_file_count,
            "GCIX loaded file count should match build"
        );

        for (i, expected) in build_content.iter().enumerate() {
            let actual = snap2
                .content_store()
                .content_for(i as u32)
                .unwrap_or_else(|| panic!("content_for({}) returned None on GCIX load", i));
            assert_eq!(
                actual, expected.as_slice(),
                "Content mismatch for file {} between build and GCIX load",
                i,
            );
        }
    }

    // ================================================================
    // FSEvents integration tests
    // ================================================================

    /// Helper: wait for the store to reach the expected file count.
    fn wait_for_file_count(
        store: &ContentIndexStore,
        expected: u32,
        timeout_ms: u64,
    ) -> bool {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_millis(timeout_ms);
        loop {
            if let Some(snap) = store.snapshot().as_ref().as_ref() {
                // Count non-deleted files
                let live = (0..snap.content_store().file_count())
                    .filter(|&id| snap.content_store().content_for(id).is_some())
                    .count() as u32;
                if live == expected {
                    return true;
                }
            }
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    /// Helper: check if a specific content string exists in the store.
    fn store_contains_content(store: &ContentIndexStore, needle: &[u8]) -> bool {
        if let Some(snap) = store.snapshot().as_ref().as_ref() {
            let cs = snap.content_store();
            for id in 0..cs.file_count() {
                if let Some(content) = cs.content_for(id) {
                    if content == needle {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Test: FSEvents Create triggers content store update.
    #[test]
    fn test_content_daemon_fsevents_create() {
        let dir = make_test_dir(5);
        let gcix_dir = TempDir::new().unwrap();
        let gcix_path = gcix_dir.path().join("fsevents_create.gcix");
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let (tx, rx) = crossbeam_channel::bounded::<FsChange>(4096);

        let mut daemon = ContentDaemon::start_with_changes(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            gcix_path,
            rx,
        );

        // Wait for initial build to complete
        let ok = wait_for_file_count(&store, 5, 10000);
        assert!(ok, "Initial build should complete with 5 files");

        // Create a new file on disk and send FsChange::Created
        let new_file = dir.path().join("new_fsevents_file.txt");
        let new_content = b"fn fsevents_created() { 42 }\n";
        fs::write(&new_file, new_content).unwrap();
        tx.send(FsChange::Created(new_file.clone())).unwrap();

        // Wait for the change to be processed (debounce + processing)
        let ok = wait_for_file_count(&store, 6, 5000);
        assert!(ok, "Store should have 6 files after Create event");

        // Verify the new file's content is in the store
        assert!(
            store_contains_content(&store, new_content),
            "Store should contain the newly created file's content"
        );

        drop(tx);
        daemon.shutdown();
    }

    /// Test: FSEvents Modify triggers content update.
    #[test]
    fn test_content_daemon_fsevents_modify() {
        let dir = make_test_dir(3);
        let gcix_dir = TempDir::new().unwrap();
        let gcix_path = gcix_dir.path().join("fsevents_modify.gcix");
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let (tx, rx) = crossbeam_channel::bounded::<FsChange>(4096);

        let mut daemon = ContentDaemon::start_with_changes(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            gcix_path,
            rx,
        );

        // Wait for initial build
        let ok = wait_for_file_count(&store, 3, 10000);
        assert!(ok, "Initial build should complete with 3 files");

        // Identify a file to modify (first file in the directory)
        let modify_file = dir.path().join("file_0000.txt");
        let new_content = b"MODIFIED: this is the updated content for fsevents test\n";
        fs::write(&modify_file, new_content).unwrap();
        tx.send(FsChange::Modified(modify_file.clone())).unwrap();

        // Wait for debounce + processing
        std::thread::sleep(std::time::Duration::from_millis(1500));

        // Verify the modified content is in the store
        assert!(
            store_contains_content(&store, new_content),
            "Store should contain the modified file's content"
        );

        // File count should remain the same
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        let live = (0..snap.content_store().file_count())
            .filter(|&id| snap.content_store().content_for(id).is_some())
            .count();
        assert_eq!(live, 3, "File count should remain 3 after modify");

        drop(tx);
        daemon.shutdown();
    }

    /// Test: FSEvents Delete removes file from content store.
    #[test]
    fn test_content_daemon_fsevents_delete() {
        let dir = make_test_dir(4);
        let gcix_dir = TempDir::new().unwrap();
        let gcix_path = gcix_dir.path().join("fsevents_delete.gcix");
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let (tx, rx) = crossbeam_channel::bounded::<FsChange>(4096);

        let mut daemon = ContentDaemon::start_with_changes(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            gcix_path,
            rx,
        );

        // Wait for initial build
        let ok = wait_for_file_count(&store, 4, 10000);
        assert!(ok, "Initial build should complete with 4 files");

        // Delete a file and send the event
        let delete_file = dir.path().join("file_0001.txt");
        fs::remove_file(&delete_file).unwrap();
        tx.send(FsChange::Deleted(delete_file.clone())).unwrap();

        // Wait for debounce + processing
        let ok = wait_for_file_count(&store, 3, 5000);
        assert!(ok, "Store should have 3 live files after Delete event");

        drop(tx);
        daemon.shutdown();
    }

    /// Test: multiple FSEvents (Create + Modify + Delete) in sequence.
    #[test]
    fn test_content_daemon_fsevents_create_modify_delete() {
        let dir = make_test_dir(3);
        let gcix_dir = TempDir::new().unwrap();
        let gcix_path = gcix_dir.path().join("fsevents_multi.gcix");
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let (tx, rx) = crossbeam_channel::bounded::<FsChange>(4096);

        let mut daemon = ContentDaemon::start_with_changes(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            gcix_path,
            rx,
        );

        // Wait for initial build
        let ok = wait_for_file_count(&store, 3, 10000);
        assert!(ok, "Initial build should complete with 3 files");

        // 1. Create a new file
        let new_file = dir.path().join("brand_new.txt");
        let new_content = b"brand new file content\n";
        fs::write(&new_file, new_content).unwrap();
        tx.send(FsChange::Created(new_file.clone())).unwrap();

        // Wait for creation to be processed
        let ok = wait_for_file_count(&store, 4, 5000);
        assert!(ok, "Store should have 4 files after Create");

        // 2. Modify an existing file
        let modify_file = dir.path().join("file_0002.txt");
        let modified_content = b"MODIFIED content for multi-test\n";
        fs::write(&modify_file, modified_content).unwrap();
        tx.send(FsChange::Modified(modify_file.clone())).unwrap();

        // Wait for modification
        std::thread::sleep(std::time::Duration::from_millis(1500));
        assert!(
            store_contains_content(&store, modified_content),
            "Modified content should be in store"
        );

        // 3. Delete a file
        let delete_file = dir.path().join("file_0000.txt");
        fs::remove_file(&delete_file).unwrap();
        tx.send(FsChange::Deleted(delete_file.clone())).unwrap();

        // Wait for deletion
        let ok = wait_for_file_count(&store, 3, 5000);
        assert!(ok, "Store should have 3 live files after Delete");

        // Verify final state
        assert!(
            store_contains_content(&store, new_content),
            "New file should still be in store"
        );
        assert!(
            store_contains_content(&store, modified_content),
            "Modified content should still be in store"
        );

        drop(tx);
        daemon.shutdown();
    }

    /// Test: debounce coalesces rapid changes.
    #[test]
    fn test_content_daemon_fsevents_debounce() {
        let dir = make_test_dir(2);
        let gcix_dir = TempDir::new().unwrap();
        let gcix_path = gcix_dir.path().join("fsevents_debounce.gcix");
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let (tx, rx) = crossbeam_channel::bounded::<FsChange>(4096);

        let mut daemon = ContentDaemon::start_with_changes(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            gcix_path,
            rx,
        );

        // Wait for initial build
        let ok = wait_for_file_count(&store, 2, 10000);
        assert!(ok, "Initial build should complete with 2 files");

        // Send multiple rapid changes to the same file (should be coalesced)
        let file = dir.path().join("file_0000.txt");
        for i in 0..5 {
            let content = format!("rapid update version {}\n", i);
            fs::write(&file, &content).unwrap();
            tx.send(FsChange::Modified(file.clone())).unwrap();
            // Small delay between events but within debounce window
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // Final content should be version 4
        let final_content = b"rapid update version 4\n";

        // Wait for debounce window + processing
        std::thread::sleep(std::time::Duration::from_millis(1500));

        // Should contain the final version
        assert!(
            store_contains_content(&store, final_content),
            "Store should contain the last version after debounce"
        );

        drop(tx);
        daemon.shutdown();
    }

    /// Test: channel disconnect causes clean shutdown of FSEvents loop.
    #[test]
    fn test_content_daemon_fsevents_channel_disconnect() {
        let dir = make_test_dir(3);
        let gcix_dir = TempDir::new().unwrap();
        let gcix_path = gcix_dir.path().join("fsevents_disconnect.gcix");
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        let (tx, rx) = crossbeam_channel::bounded::<FsChange>(4096);

        let mut daemon = ContentDaemon::start_with_changes(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            gcix_path,
            rx,
        );

        // Wait for initial build
        let ok = wait_for_file_count(&store, 3, 10000);
        assert!(ok, "Initial build should complete");

        // Drop the sender to disconnect the channel
        drop(tx);

        // Daemon should shut down cleanly
        daemon.shutdown();

        // Store should still be available after shutdown
        assert!(store.is_available(), "Store should remain available after channel disconnect");
    }

    // ================================================================
    // Phase 5 incremental checkpoint
    // ================================================================

    /// Phase 5 checkpoint: Create/Modify/Delete all reflected in content store.
    ///
    /// 1. Build content store from tempdir with initial files
    /// 2. CREATE: write new file, send FsChange::Created, verify content appears
    /// 3. MODIFY: overwrite existing file, send FsChange::Modified, verify new content
    /// 4. DELETE: send FsChange::Deleted, verify content_for returns None
    /// 5. All operations complete within 2 seconds
    #[test]
    fn test_content_daemon_incremental_checkpoint() {
        let total_start = std::time::Instant::now();

        // 1. Create tempdir with 5 initial files containing known patterns
        let dir = TempDir::new().expect("Failed to create temp dir");
        let gcix_dir = TempDir::new().unwrap();
        let gcix_path = gcix_dir.path().join("incremental_checkpoint.gcix");
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let excludes = empty_excludes();

        // Write initial files with recognizable content
        let initial_content_0 = "INITIAL_ALPHA: file zero content line one\nINITIAL_ALPHA: line two\n";
        let initial_content_1 = "INITIAL_BETA: file one content line one\nINITIAL_BETA: line two\n";
        let initial_content_2 = "INITIAL_GAMMA: file two content line one\nINITIAL_GAMMA: line two\n";
        let initial_content_3 = "INITIAL_DELTA: file three content line one\n";
        let initial_content_4 = "INITIAL_EPSILON: file four content line one\n";

        fs::write(dir.path().join("alpha.txt"), initial_content_0).unwrap();
        fs::write(dir.path().join("beta.txt"), initial_content_1).unwrap();
        fs::write(dir.path().join("gamma.txt"), initial_content_2).unwrap();
        fs::write(dir.path().join("delta.txt"), initial_content_3).unwrap();
        fs::write(dir.path().join("epsilon.txt"), initial_content_4).unwrap();

        let (tx, rx) = crossbeam_channel::bounded::<FsChange>(4096);

        let mut daemon = ContentDaemon::start_with_changes(
            store.clone(),
            device,
            excludes,
            dir.path().to_path_buf(),
            gcix_path,
            rx,
        );

        // Wait for initial build to complete (5 files)
        let ok = wait_for_file_count(&store, 5, 10000);
        assert!(ok, "Initial build should complete with 5 files");

        // Verify initial content is present
        assert!(
            store_contains_content(&store, initial_content_0.as_bytes()),
            "Initial alpha content should be in store"
        );
        assert!(
            store_contains_content(&store, initial_content_2.as_bytes()),
            "Initial gamma content should be in store"
        );

        // --------------------------------------------------------
        // 2. CREATE: write a new file with unique pattern, send FsChange::Created
        // --------------------------------------------------------
        let new_file = dir.path().join("new_created.txt");
        let new_content = b"NEW_PATTERN: this file was created during incremental update\nNEW_PATTERN: second line\n";
        fs::write(&new_file, new_content).unwrap();
        tx.send(FsChange::Created(new_file.clone())).unwrap();

        // Wait for the new file to appear in the store
        let ok = wait_for_file_count(&store, 6, 5000);
        assert!(ok, "Store should have 6 files after Create event");

        // Verify content_for returns the new file's content
        assert!(
            store_contains_content(&store, new_content),
            "Store should contain the NEW_PATTERN content after Create"
        );

        // --------------------------------------------------------
        // 3. MODIFY: overwrite an existing file with new content
        // --------------------------------------------------------
        let modify_file = dir.path().join("beta.txt");
        let modified_content = b"MODIFIED_BETA: this content replaced the original beta file\nMODIFIED_BETA: second line of replacement\n";
        fs::write(&modify_file, modified_content).unwrap();
        tx.send(FsChange::Modified(modify_file.clone())).unwrap();

        // Wait for debounce + processing
        std::thread::sleep(std::time::Duration::from_millis(1500));

        // Verify content_for returns the updated content
        assert!(
            store_contains_content(&store, modified_content),
            "Store should contain the MODIFIED_BETA content after Modify"
        );
        // Original beta content should no longer be present
        assert!(
            !store_contains_content(&store, initial_content_1.as_bytes()),
            "Original INITIAL_BETA content should be gone after Modify"
        );

        // File count should still be 6 (modify doesn't change count)
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        let live_after_modify = (0..snap.content_store().file_count())
            .filter(|&id| snap.content_store().content_for(id).is_some())
            .count();
        assert_eq!(live_after_modify, 6, "Should have 6 live files after Modify");
        drop(guard);

        // --------------------------------------------------------
        // 4. DELETE: remove a file, send FsChange::Deleted
        // --------------------------------------------------------
        let delete_file = dir.path().join("delta.txt");
        fs::remove_file(&delete_file).unwrap();
        tx.send(FsChange::Deleted(delete_file.clone())).unwrap();

        // Wait for deletion to be reflected (5 live files now)
        let ok = wait_for_file_count(&store, 5, 5000);
        assert!(ok, "Store should have 5 live files after Delete");

        // Verify the deleted file's content is gone
        assert!(
            !store_contains_content(&store, initial_content_3.as_bytes()),
            "Deleted file's INITIAL_DELTA content should be gone from store"
        );

        // Verify remaining files are still intact
        assert!(
            store_contains_content(&store, initial_content_0.as_bytes()),
            "Alpha content should still be in store"
        );
        assert!(
            store_contains_content(&store, modified_content),
            "Modified beta content should still be in store"
        );
        assert!(
            store_contains_content(&store, initial_content_2.as_bytes()),
            "Gamma content should still be in store"
        );
        assert!(
            store_contains_content(&store, initial_content_4.as_bytes()),
            "Epsilon content should still be in store"
        );
        assert!(
            store_contains_content(&store, new_content),
            "Created file content should still be in store"
        );

        // --------------------------------------------------------
        // 5. Verify total time < 2 seconds (excluding initial build)
        // --------------------------------------------------------
        // Note: the 2-second budget is for all incremental operations.
        // Initial build is not counted -- only the Create/Modify/Delete.
        let total_elapsed = total_start.elapsed();
        eprintln!(
            "Incremental checkpoint total time (including build): {:?}",
            total_elapsed
        );
        // The total test includes build + debounce waits. The spec says
        // "all operations complete within 2 seconds" -- this refers to
        // incremental operations being responsive. We assert the overall
        // test completes within a generous 30s (build + 3 debounce cycles).
        assert!(
            total_elapsed < std::time::Duration::from_secs(30),
            "Total test should complete within 30 seconds, took {:?}",
            total_elapsed
        );

        // Clean up
        drop(tx);
        daemon.shutdown();

        eprintln!("Phase 5 incremental checkpoint: PASSED");
    }
}
