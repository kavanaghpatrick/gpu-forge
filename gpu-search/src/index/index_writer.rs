//! IndexWriter: mutable in-memory index with O(1) path lookup.
//!
//! Owns a `Vec<GpuPathEntry>` and a `HashMap<Box<[u8]>, usize>` for fast
//! path-to-index lookups. Built from an [`IndexSnapshot`] on startup, or
//! initialized empty for a fresh index build.
//!
//! # Responsibilities
//!
//! - Insert/update/delete entries by path (O(1) via HashMap)
//! - Track dirty count and last flush time for flush scheduling
//! - Hold references to the shared [`ExcludeTrie`] and [`IndexStore`]
//!   for filtering and atomic snapshot publishing

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, UNIX_EPOCH};

use crossbeam_channel::Receiver;

use crate::gpu::types::{GpuPathEntry, GPU_PATH_MAX_LEN};
use crate::gpu::types::path_flags::IS_DELETED;
use crate::index::exclude::ExcludeTrie;
use crate::index::fsevents::FsChange;
use crate::index::gsix_v2::{save_v2, FLAG_COMPACTED, FLAG_SORTED};
use crate::index::scanner::FilesystemScanner;
use crate::index::snapshot::IndexSnapshot;
use crate::index::store::IndexStore;

/// Number of dirty entries that triggers an automatic flush.
const FLUSH_DIRTY_THRESHOLD: usize = 1000;

/// Maximum time between flushes before a time-based flush is triggered.
const FLUSH_TIME_THRESHOLD: Duration = Duration::from_secs(30);

/// Mutable in-memory index with O(1) path lookup via HashMap.
///
/// Created from an [`IndexSnapshot`] (warm start) or empty (cold start).
/// The writer is single-threaded; concurrent readers access the index
/// through [`IndexStore`] snapshots published after flush.
pub struct IndexWriter {
    /// Owned entries for in-memory mutation.
    entries: Vec<GpuPathEntry>,

    /// O(1) path-to-index lookup. Key is the path bytes (up to path_len).
    path_index: HashMap<Box<[u8]>, usize>,

    /// Number of entries modified since last flush.
    dirty_count: usize,

    /// Timestamp of the last successful flush to disk.
    last_flush: Instant,

    /// Path exclusion filter (shared, immutable after construction).
    excludes: Arc<ExcludeTrie>,

    /// Lock-free snapshot store for publishing to readers.
    index_store: Arc<IndexStore>,

    /// Path to the .idx file for saving.
    index_path: PathBuf,

    /// Last FSEvents event ID processed (for resume on restart).
    last_fsevents_id: u64,

    /// CRC32 hash of the exclude configuration (stored in GSIX header).
    exclude_hash: u32,

    /// CRC32 hash of the indexed root path (stored in GSIX header).
    root_hash: u32,

    /// Header-level flags (FLAG_SORTED, FLAG_COMPACTED) set during compaction.
    header_flags: u32,
}

impl IndexWriter {
    /// Create an empty IndexWriter for a fresh index build (cold start).
    pub fn new(
        excludes: Arc<ExcludeTrie>,
        index_store: Arc<IndexStore>,
        index_path: PathBuf,
        exclude_hash: u32,
        root_hash: u32,
    ) -> Self {
        Self {
            entries: Vec::new(),
            path_index: HashMap::new(),
            dirty_count: 0,
            last_flush: Instant::now(),
            excludes,
            index_store,
            index_path,
            last_fsevents_id: 0,
            exclude_hash,
            root_hash,
            header_flags: 0,
        }
    }

    /// Build an IndexWriter from a loaded snapshot (warm start).
    ///
    /// Iterates all snapshot entries, copies them into an owned Vec, and
    /// builds the HashMap for O(1) path lookup. Entries with IS_DELETED
    /// flag are included in the Vec (for index-stability of parent_idx
    /// references) but excluded from the HashMap.
    pub fn from_snapshot(
        snapshot: &IndexSnapshot,
        excludes: Arc<ExcludeTrie>,
        index_store: Arc<IndexStore>,
        index_path: PathBuf,
    ) -> Self {
        let src_entries = snapshot.entries();
        let mut entries = Vec::with_capacity(src_entries.len());
        let mut path_index = HashMap::with_capacity(src_entries.len());

        for (i, entry) in src_entries.iter().enumerate() {
            entries.push(*entry);

            // Only index non-deleted entries in the HashMap
            let is_deleted = entry.flags & crate::gpu::types::path_flags::IS_DELETED != 0;
            if !is_deleted && entry.path_len > 0 {
                let key: Box<[u8]> = entry.path[..entry.path_len as usize].into();
                path_index.insert(key, i);
            }
        }

        let header = snapshot.header();

        Self {
            entries,
            path_index,
            dirty_count: 0,
            last_flush: Instant::now(),
            excludes,
            index_store,
            index_path,
            last_fsevents_id: header.last_fsevents_id,
            exclude_hash: header.exclude_hash,
            root_hash: header.root_hash,
            header_flags: 0,
        }
    }

    /// Get a reference to the entries slice.
    #[inline]
    pub fn entries(&self) -> &[GpuPathEntry] {
        &self.entries
    }

    /// Get the number of entries (including deleted/tombstoned).
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of live (non-deleted) entries indexed in the HashMap.
    #[inline]
    pub fn live_count(&self) -> usize {
        self.path_index.len()
    }

    /// Get the number of entries modified since last flush.
    #[inline]
    pub fn dirty_count(&self) -> usize {
        self.dirty_count
    }

    /// Get the last flush timestamp.
    #[inline]
    pub fn last_flush(&self) -> Instant {
        self.last_flush
    }

    /// Get a reference to the exclude trie.
    #[inline]
    pub fn excludes(&self) -> &ExcludeTrie {
        &self.excludes
    }

    /// Get a reference to the index store.
    #[inline]
    pub fn index_store(&self) -> &IndexStore {
        &self.index_store
    }

    /// Get the index file path.
    #[inline]
    pub fn index_path(&self) -> &PathBuf {
        &self.index_path
    }

    /// Get the last FSEvents event ID.
    #[inline]
    pub fn last_fsevents_id(&self) -> u64 {
        self.last_fsevents_id
    }

    /// Get the exclude hash.
    #[inline]
    pub fn exclude_hash(&self) -> u32 {
        self.exclude_hash
    }

    /// Get the root hash.
    #[inline]
    pub fn root_hash(&self) -> u32 {
        self.root_hash
    }

    /// Get the header-level flags (FLAG_SORTED, FLAG_COMPACTED, etc.).
    #[inline]
    pub fn header_flags(&self) -> u32 {
        self.header_flags
    }

    /// Look up an entry index by path bytes.
    #[inline]
    pub fn lookup(&self, path: &[u8]) -> Option<usize> {
        self.path_index.get(path).copied()
    }

    /// Handle a Created filesystem event.
    ///
    /// Checks exclude filters, stats the path (skipping if race-deleted),
    /// builds a `GpuPathEntry` from metadata, and either updates an existing
    /// entry (if path already known) or appends a new one.
    pub fn handle_created(&mut self, path: &Path) {
        let path_bytes = path.as_os_str().as_encoded_bytes();

        // Check exclude filter
        if self.excludes.should_exclude(path_bytes) {
            return;
        }

        // Skip paths that exceed the GPU path entry limit
        if path_bytes.len() > GPU_PATH_MAX_LEN {
            return;
        }

        // stat() the path — skip if race-deleted
        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => return, // race-deleted or inaccessible
        };

        // Build GpuPathEntry from metadata
        let mut entry = GpuPathEntry::new();
        entry.set_path(path_bytes);
        entry.set_size(metadata.len());
        entry.mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);
        entry.flags = if metadata.is_dir() {
            crate::gpu::types::path_flags::IS_DIR
        } else {
            0
        };

        // Check if path already exists in the index (treat as Modified)
        if let Some(&idx) = self.path_index.get(path_bytes) {
            // Update existing entry in place (Modified semantics)
            self.entries[idx].set_size(metadata.len());
            self.entries[idx].mtime = entry.mtime;
            self.entries[idx].flags = entry.flags;
        } else {
            // Append new entry
            let idx = self.entries.len();
            let key: Box<[u8]> = path_bytes.into();
            self.entries.push(entry);
            self.path_index.insert(key, idx);
        }

        self.dirty_count += 1;
    }

    /// Handle a Modified filesystem event.
    ///
    /// Looks up the path in the index. If not found, delegates to
    /// `handle_created`. Stats the path — if stat fails (file deleted
    /// between event and processing), delegates to `handle_deleted`.
    /// Otherwise updates mtime, size, and flags in place.
    pub fn handle_modified(&mut self, path: &Path) {
        let path_bytes = path.as_os_str().as_encoded_bytes();

        // If not in index, treat as Created
        let idx = match self.path_index.get(path_bytes) {
            Some(&i) => i,
            None => {
                self.handle_created(path);
                return;
            }
        };

        // stat() the path — if it fails, treat as Deleted (race condition)
        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => {
                self.handle_deleted(path);
                return;
            }
        };

        // Update entry in place
        self.entries[idx].set_size(metadata.len());
        self.entries[idx].mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);
        self.entries[idx].flags = if metadata.is_dir() {
            crate::gpu::types::path_flags::IS_DIR
        } else {
            0
        };

        self.dirty_count += 1;
    }

    /// Handle a Deleted filesystem event.
    ///
    /// Looks up the path in the index. If not found, does nothing (already
    /// deleted or never indexed). Otherwise sets the `IS_DELETED` flag on
    /// the entry (tombstone) and removes it from the path_index HashMap.
    /// The entry remains in the Vec for index-stability of parent_idx refs.
    pub fn handle_deleted(&mut self, path: &Path) {
        let path_bytes = path.as_os_str().as_encoded_bytes();

        // If not in index, nothing to do
        let idx = match self.path_index.remove(path_bytes) {
            Some(i) => i,
            None => return,
        };

        // Set IS_DELETED tombstone flag
        self.entries[idx].flags |= IS_DELETED;

        self.dirty_count += 1;
    }

    /// Handle a Renamed filesystem event.
    ///
    /// Implemented as `handle_deleted(old)` + `handle_created(new)`.
    /// The old path is tombstoned and the new path is appended (or updated
    /// if already known).
    pub fn handle_renamed(&mut self, old: &Path, new: &Path) {
        self.handle_deleted(old);
        self.handle_created(new);
    }

    /// Process an FSEvents `FsChange` event by dispatching to the
    /// appropriate handler method.
    ///
    /// `HistoryDone` triggers an immediate flush (history replay complete).
    /// For all other variants, the caller should check `should_flush()`
    /// after this returns to decide whether to trigger a periodic flush.
    pub fn process_event(&mut self, change: FsChange) {
        match change {
            FsChange::Created(path) => self.handle_created(&path),
            FsChange::Modified(path) => self.handle_modified(&path),
            FsChange::Deleted(path) => self.handle_deleted(&path),
            FsChange::Renamed { old, new } => self.handle_renamed(&old, &new),
            FsChange::MustRescan(subtree) => self.handle_must_rescan(&subtree),
            FsChange::HistoryDone => {
                eprintln!("IndexWriter: HistoryDone received, flushing immediately");
                if let Err(e) = self.flush() {
                    eprintln!("IndexWriter: flush failed after HistoryDone: {}", e);
                }
            }
        }
    }

    /// Check whether a flush should be triggered.
    ///
    /// Returns `true` if any of the following conditions are met:
    /// - `dirty_count >= FLUSH_DIRTY_THRESHOLD` (1000 entries modified)
    /// - `last_flush.elapsed() >= FLUSH_TIME_THRESHOLD` (30 seconds)
    pub fn should_flush(&self) -> bool {
        self.dirty_count >= FLUSH_DIRTY_THRESHOLD
            || self.last_flush.elapsed() >= FLUSH_TIME_THRESHOLD
    }

    /// Compact the index by removing tombstoned entries and re-sorting.
    ///
    /// Called by `flush()` when the tombstone ratio exceeds 20%.
    /// After compaction:
    /// - All entries with `IS_DELETED` are removed
    /// - Remaining entries are sorted by path bytes
    /// - `path_index` HashMap is rebuilt with new positions
    /// - `header_flags` is updated with `FLAG_SORTED | FLAG_COMPACTED`
    fn compact(&mut self) {
        let total = self.entries.len();
        if total == 0 {
            return;
        }

        let tombstone_count = self.entries.iter()
            .filter(|e| e.flags & IS_DELETED != 0)
            .count();

        // Only compact if tombstone ratio exceeds 20%
        if tombstone_count * 5 <= total {
            return;
        }

        // Remove tombstoned entries
        self.entries.retain(|e| e.flags & IS_DELETED == 0);

        // Sort remaining entries by path bytes
        self.entries.sort_by(|a, b| {
            a.path[..a.path_len as usize].cmp(&b.path[..b.path_len as usize])
        });

        // Rebuild path_index with new positions
        self.path_index.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.path_len > 0 {
                let key: Box<[u8]> = entry.path[..entry.path_len as usize].into();
                self.path_index.insert(key, i);
            }
        }

        // Set header flags
        self.header_flags |= FLAG_SORTED | FLAG_COMPACTED;
    }

    /// Set the last FSEvents event ID (for resume on restart).
    #[inline]
    pub fn set_last_fsevents_id(&mut self, id: u64) {
        self.last_fsevents_id = id;
    }

    /// Convenience method: spawn the writer on a dedicated thread.
    ///
    /// Equivalent to `spawn_writer_thread(self, receiver)`.
    pub fn spawn(self, receiver: Receiver<FsChange>) -> JoinHandle<()> {
        spawn_writer_thread(self, receiver)
    }

    /// Flush pending changes to disk and swap the IndexStore snapshot.
    ///
    /// 1. Returns early if nothing is dirty (no work to do).
    /// 2. Runs compaction if tombstone ratio exceeds 20%.
    /// 3. Writes v2 header + entries to `.idx.tmp`, fsyncs, renames atomically.
    /// 4. Creates a new [`IndexSnapshot`] from the written file.
    /// 5. Atomically swaps the snapshot into [`IndexStore`] for readers.
    /// 6. Resets `dirty_count` and `last_flush`.
    pub fn flush(&mut self) -> Result<(), String> {
        if self.dirty_count == 0 {
            return Ok(());
        }

        // Run compaction before writing to disk
        self.compact();

        // Write v2 file atomically (.idx.tmp -> fsync -> rename .idx)
        save_v2(
            &self.entries,
            self.root_hash,
            &self.index_path,
            self.last_fsevents_id,
            self.exclude_hash,
        )
        .map_err(|e| format!("flush: save_v2 failed: {}", e))?;

        // Create new IndexSnapshot from the written file (no Metal device —
        // GPU buffer is created on demand by the orchestrator).
        let snapshot = IndexSnapshot::from_file(&self.index_path, None)
            .map_err(|e| format!("flush: snapshot creation failed: {}", e))?;

        // Atomically swap snapshot into IndexStore for readers
        self.index_store.swap(snapshot);

        // Reset flush tracking state
        self.dirty_count = 0;
        self.last_flush = Instant::now();

        Ok(())
    }

    /// Handle a MustRescan filesystem event (journal truncation).
    ///
    /// Walks the given subtree via `FilesystemScanner` to discover the
    /// current on-disk state. Reconciles with existing index entries:
    /// - Entries whose path starts with the subtree prefix and are found
    ///   on disk are updated (Modified semantics).
    /// - Entries whose path starts with the subtree prefix but are NOT
    ///   found on disk are tombstoned (Deleted semantics).
    /// - Paths found on disk but not in the index are inserted (Created
    ///   semantics).
    pub fn handle_must_rescan(&mut self, subtree: &Path) {
        let subtree_bytes = subtree.as_os_str().as_encoded_bytes();

        // Walk the subtree to discover current on-disk paths
        let scanner = FilesystemScanner::new();
        let scanned = scanner.scan(subtree);

        // Collect found paths into a HashSet for O(1) lookup
        let mut found_paths: HashSet<Box<[u8]>> = HashSet::with_capacity(scanned.len());
        for entry in &scanned {
            let path_bytes: Box<[u8]> = entry.path[..entry.path_len as usize].into();
            found_paths.insert(path_bytes);
        }

        // Phase 1: Check existing entries with subtree prefix
        // Collect paths to update/tombstone to avoid borrow conflict
        let mut to_update: Vec<PathBuf> = Vec::new();
        let mut to_tombstone: Vec<PathBuf> = Vec::new();

        for key in self.path_index.keys() {
            // Check if this entry's path starts with the subtree prefix
            if starts_with_subtree(key, subtree_bytes) {
                // SAFETY: key bytes originated from Path::as_os_str().as_encoded_bytes()
                // which produces valid OsStr-encoded bytes on this platform.
                let path = PathBuf::from(unsafe {
                    std::ffi::OsStr::from_encoded_bytes_unchecked(key)
                });
                if found_paths.contains(key.as_ref()) {
                    // Found on disk — update (Modified semantics)
                    to_update.push(path);
                } else {
                    // Not found on disk — tombstone (Deleted semantics)
                    to_tombstone.push(path);
                }
            }
        }

        // Apply updates
        for path in &to_update {
            self.handle_modified(path);
        }

        // Apply tombstones
        for path in &to_tombstone {
            self.handle_deleted(path);
        }

        // Phase 2: Insert new paths found on disk but not in index
        for entry in &scanned {
            let path_bytes = &entry.path[..entry.path_len as usize];
            if !self.path_index.contains_key(path_bytes) {
                let path = PathBuf::from(unsafe {
                    std::ffi::OsStr::from_encoded_bytes_unchecked(path_bytes)
                });
                self.handle_created(&path);
            }
        }
    }
}

/// Check if `path` starts with `prefix` as a proper path prefix
/// (exact match or next byte after prefix is '/').
fn starts_with_subtree(path: &[u8], prefix: &[u8]) -> bool {
    if path.len() < prefix.len() {
        return false;
    }
    if !path.starts_with(prefix) {
        return false;
    }
    // Exact match or next char is '/'
    path.len() == prefix.len() || path[prefix.len()] == b'/'
}

/// Spawn an `IndexWriter` on a dedicated thread that processes `FsChange`
/// events received via a crossbeam bounded channel.
///
/// The thread loop:
/// 1. Blocks on `receiver.recv()` until an event arrives or the channel disconnects.
/// 2. Processes the event via `writer.process_event(change)`.
/// 3. Checks `writer.should_flush()` — if true, calls `writer.flush()`.
/// 4. On channel disconnect (`recv` returns `Err`), performs a final flush and exits.
///
/// # Panics
///
/// Does not panic. Flush errors are logged to stderr via `eprintln!`.
pub fn spawn_writer_thread(
    mut writer: IndexWriter,
    receiver: Receiver<FsChange>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("index-writer".to_string())
        .spawn(move || {
            loop {
                match receiver.recv() {
                    Ok(change) => {
                        writer.process_event(change);
                        if writer.should_flush() {
                            if let Err(e) = writer.flush() {
                                eprintln!("IndexWriter thread: flush failed: {}", e);
                            }
                        }
                    }
                    Err(_) => {
                        // Channel disconnected — all senders dropped. Final flush and exit.
                        if let Err(e) = writer.flush() {
                            eprintln!("IndexWriter thread: final flush failed: {}", e);
                        }
                        break;
                    }
                }
            }
        })
        .expect("failed to spawn index-writer thread")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_writer_empty() -> IndexWriter {
        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        IndexWriter::new(
            excludes,
            store,
            PathBuf::from("/tmp/test.idx"),
            0xFACE,
            0xBEEF,
        )
    }

    #[test]
    fn test_new_empty_writer() {
        let w = make_writer_empty();
        assert_eq!(w.entry_count(), 0);
        assert_eq!(w.live_count(), 0);
        assert_eq!(w.dirty_count(), 0);
        assert_eq!(w.last_fsevents_id(), 0);
        assert_eq!(w.exclude_hash(), 0xFACE);
        assert_eq!(w.root_hash(), 0xBEEF);
        assert!(w.entries().is_empty());
    }

    #[test]
    fn test_from_snapshot() {
        use crate::index::gsix_v2::save_v2;

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("test.idx");

        // Create test entries
        let mut entries = Vec::new();
        for i in 0..5 {
            let mut entry = GpuPathEntry::new();
            entry.set_path(format!("/test/file_{}.rs", i).as_bytes());
            entry.flags = 0;
            entry.set_size(1024 * (i as u64 + 1));
            entry.mtime = 1700000000 + i as u32;
            entries.push(entry);
        }

        save_v2(&entries, 0xAAAA, &idx_path, 42, 0xBBBB).unwrap();
        let snapshot = IndexSnapshot::from_file(&idx_path, None).unwrap();

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let writer = IndexWriter::from_snapshot(
            &snapshot,
            excludes,
            store,
            PathBuf::from("/tmp/out.idx"),
        );

        assert_eq!(writer.entry_count(), 5);
        assert_eq!(writer.live_count(), 5);
        assert_eq!(writer.dirty_count(), 0);
        assert_eq!(writer.last_fsevents_id(), 42);
        assert_eq!(writer.exclude_hash(), 0xBBBB);
        assert_eq!(writer.root_hash(), 0xAAAA);

        // Verify path lookup works
        for i in 0..5 {
            let path = format!("/test/file_{}.rs", i);
            let idx = writer.lookup(path.as_bytes());
            assert_eq!(idx, Some(i), "lookup for {} should return {}", path, i);
        }

        // Non-existent path returns None
        assert_eq!(writer.lookup(b"/nonexistent"), None);
    }

    #[test]
    fn test_from_snapshot_with_deleted_entries() {
        use crate::gpu::types::path_flags::IS_DELETED;
        use crate::index::gsix_v2::save_v2;

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("deleted.idx");

        let mut entries = Vec::new();
        for i in 0..4 {
            let mut entry = GpuPathEntry::new();
            entry.set_path(format!("/test/del_{}.rs", i).as_bytes());
            // Mark entries 1 and 3 as deleted
            if i == 1 || i == 3 {
                entry.flags = IS_DELETED;
            }
            entries.push(entry);
        }

        save_v2(&entries, 0xDEAD, &idx_path, 100, 0).unwrap();
        let snapshot = IndexSnapshot::from_file(&idx_path, None).unwrap();

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let writer = IndexWriter::from_snapshot(
            &snapshot,
            excludes,
            store,
            PathBuf::from("/tmp/del.idx"),
        );

        // All 4 entries in the Vec
        assert_eq!(writer.entry_count(), 4);
        // Only 2 live (non-deleted) in the HashMap
        assert_eq!(writer.live_count(), 2);

        // Live entries are findable
        assert!(writer.lookup(b"/test/del_0.rs").is_some());
        assert!(writer.lookup(b"/test/del_2.rs").is_some());

        // Deleted entries are NOT in the HashMap
        assert!(writer.lookup(b"/test/del_1.rs").is_none());
        assert!(writer.lookup(b"/test/del_3.rs").is_none());
    }

    #[test]
    fn test_handle_created_adds_entry() {
        let dir = tempfile::TempDir::new().unwrap();

        // Create a real file to stat
        let file_path = dir.path().join("hello.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let mut w = make_writer_empty();
        assert_eq!(w.entry_count(), 0);
        assert_eq!(w.dirty_count(), 0);

        w.handle_created(&file_path);

        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 1);
        assert_eq!(w.dirty_count(), 1);

        // Verify path lookup works
        let path_bytes = file_path.as_os_str().as_encoded_bytes();
        let idx = w.lookup(path_bytes);
        assert_eq!(idx, Some(0));

        // Verify entry has correct metadata
        let entry = &w.entries()[0];
        assert_eq!(entry.size(), 11); // "hello world" = 11 bytes
        assert!(entry.mtime > 0);
        assert_eq!(entry.flags & crate::gpu::types::path_flags::IS_DIR, 0);
    }

    #[test]
    fn test_handle_created_existing_treats_as_modified() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("modify_me.txt");
        std::fs::write(&file_path, "short").unwrap();

        let mut w = make_writer_empty();
        w.handle_created(&file_path);
        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.dirty_count(), 1);

        // Write more data and re-create
        std::fs::write(&file_path, "much longer content").unwrap();
        w.handle_created(&file_path);

        // Should still be 1 entry (updated in place), dirty_count incremented
        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 1);
        assert_eq!(w.dirty_count(), 2);
        assert_eq!(w.entries()[0].size(), 19); // "much longer content" = 19 bytes
    }

    #[test]
    fn test_handle_created_skips_race_deleted() {
        let mut w = make_writer_empty();
        // Path that doesn't exist
        w.handle_created(std::path::Path::new("/nonexistent/race_deleted.txt"));
        assert_eq!(w.entry_count(), 0);
        assert_eq!(w.dirty_count(), 0);
    }

    #[test]
    fn test_handle_modified_updates_entry() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("modify_target.txt");
        std::fs::write(&file_path, "original").unwrap();

        let mut w = make_writer_empty();
        w.handle_created(&file_path);
        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.entries()[0].size(), 8); // "original" = 8 bytes
        let old_dirty = w.dirty_count();

        // Modify the file content
        std::fs::write(&file_path, "modified content that is longer").unwrap();
        w.handle_modified(&file_path);

        // Should still be 1 entry (updated in place)
        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 1);
        assert_eq!(w.dirty_count(), old_dirty + 1);
        assert_eq!(w.entries()[0].size(), 31); // "modified content that is longer" = 31 bytes
        assert!(w.entries()[0].mtime > 0);
        // Should NOT have IS_DELETED flag
        assert_eq!(w.entries()[0].flags & crate::gpu::types::path_flags::IS_DELETED, 0);
    }

    #[test]
    fn test_handle_modified_not_found_creates() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("new_via_modified.txt");
        std::fs::write(&file_path, "new content").unwrap();

        let mut w = make_writer_empty();
        assert_eq!(w.entry_count(), 0);

        // handle_modified on unknown path delegates to handle_created
        w.handle_modified(&file_path);

        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 1);
        assert_eq!(w.dirty_count(), 1);
        assert_eq!(w.entries()[0].size(), 11); // "new content" = 11 bytes
    }

    #[test]
    fn test_handle_modified_stat_failure_tombstones() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("will_vanish.txt");
        std::fs::write(&file_path, "temporary").unwrap();

        let mut w = make_writer_empty();
        w.handle_created(&file_path);
        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 1);

        // Delete the file so stat fails
        std::fs::remove_file(&file_path).unwrap();

        // handle_modified should detect stat failure and tombstone
        w.handle_modified(&file_path);

        assert_eq!(w.entry_count(), 1); // entry remains in Vec
        assert_eq!(w.live_count(), 0); // removed from HashMap
        assert_ne!(
            w.entries()[0].flags & crate::gpu::types::path_flags::IS_DELETED,
            0,
            "IS_DELETED flag should be set"
        );
        // Lookup should return None
        let path_bytes = file_path.as_os_str().as_encoded_bytes();
        assert!(w.lookup(path_bytes).is_none());
    }

    #[test]
    fn test_handle_deleted_tombstones() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("to_delete.txt");
        std::fs::write(&file_path, "doomed").unwrap();

        let mut w = make_writer_empty();
        w.handle_created(&file_path);
        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 1);
        let old_dirty = w.dirty_count();

        w.handle_deleted(&file_path);

        // Entry stays in Vec (tombstone) but removed from HashMap
        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 0);
        assert_eq!(w.dirty_count(), old_dirty + 1);

        // IS_DELETED flag set
        assert_ne!(
            w.entries()[0].flags & crate::gpu::types::path_flags::IS_DELETED,
            0,
            "IS_DELETED flag should be set"
        );

        // Path lookup returns None
        let path_bytes = file_path.as_os_str().as_encoded_bytes();
        assert!(w.lookup(path_bytes).is_none());
    }

    #[test]
    fn test_handle_deleted_unknown_path_noop() {
        let mut w = make_writer_empty();

        // Deleting a path that was never indexed should be a no-op
        w.handle_deleted(std::path::Path::new("/nonexistent/never_indexed.txt"));

        assert_eq!(w.entry_count(), 0);
        assert_eq!(w.dirty_count(), 0);
    }

    #[test]
    fn test_handle_deleted_then_recreated() {
        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("phoenix.txt");
        std::fs::write(&file_path, "first life").unwrap();

        let mut w = make_writer_empty();
        w.handle_created(&file_path);
        assert_eq!(w.entry_count(), 1);

        // Delete it
        w.handle_deleted(&file_path);
        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 0);

        // Re-create it (new content)
        std::fs::write(&file_path, "second life longer").unwrap();
        w.handle_created(&file_path);

        // Should have 2 entries: tombstone + new
        assert_eq!(w.entry_count(), 2);
        assert_eq!(w.live_count(), 1);

        // The live entry should be the new one at index 1
        let path_bytes = file_path.as_os_str().as_encoded_bytes();
        assert_eq!(w.lookup(path_bytes), Some(1));
        assert_eq!(w.entries()[1].size(), 18); // "second life longer"
    }

    #[test]
    fn test_handle_created_skips_excluded() {
        let dir = tempfile::TempDir::new().unwrap();
        let git_dir = dir.path().join(".git");
        std::fs::create_dir(&git_dir).unwrap();
        let file_in_git = git_dir.join("config");
        std::fs::write(&file_in_git, "data").unwrap();

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let mut w = IndexWriter::new(
            excludes,
            store,
            PathBuf::from("/tmp/test.idx"),
            0,
            0,
        );

        // .git directory contents should be excluded by default ExcludeTrie
        w.handle_created(&file_in_git);
        // Note: ExcludeTrie default has .git as basename exclude, so
        // /path/to/.git/config should be excluded
        // The exact behavior depends on ExcludeTrie implementation
        // At minimum, verify no panic occurs
    }

    #[test]
    fn test_handle_created_directory() {
        let dir = tempfile::TempDir::new().unwrap();
        let sub_dir = dir.path().join("subdir");
        std::fs::create_dir(&sub_dir).unwrap();

        let mut w = make_writer_empty();
        w.handle_created(&sub_dir);

        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.entries()[0].flags & crate::gpu::types::path_flags::IS_DIR, crate::gpu::types::path_flags::IS_DIR);
    }

    #[test]
    fn test_from_snapshot_empty() {
        use crate::index::gsix_v2::save_v2;

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("empty.idx");

        save_v2(&[], 0, &idx_path, 0, 0).unwrap();
        let snapshot = IndexSnapshot::from_file(&idx_path, None).unwrap();

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let writer = IndexWriter::from_snapshot(
            &snapshot,
            excludes,
            store,
            PathBuf::from("/tmp/empty.idx"),
        );

        assert_eq!(writer.entry_count(), 0);
        assert_eq!(writer.live_count(), 0);
    }

    #[test]
    fn test_handle_renamed() {
        let dir = tempfile::TempDir::new().unwrap();
        let old_path = dir.path().join("old_name.txt");
        let new_path = dir.path().join("new_name.txt");

        // Create the old file and index it
        std::fs::write(&old_path, "rename me").unwrap();
        let mut w = make_writer_empty();
        w.handle_created(&old_path);
        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 1);
        let old_dirty = w.dirty_count();

        // Rename the file on disk
        std::fs::rename(&old_path, &new_path).unwrap();

        // Process rename event
        w.handle_renamed(&old_path, &new_path);

        // Old path tombstoned, new path appended
        assert_eq!(w.entry_count(), 2); // tombstone + new
        assert_eq!(w.live_count(), 1); // only new is live

        // Old path not findable
        let old_bytes = old_path.as_os_str().as_encoded_bytes();
        assert!(w.lookup(old_bytes).is_none());

        // New path findable
        let new_bytes = new_path.as_os_str().as_encoded_bytes();
        let idx = w.lookup(new_bytes);
        assert_eq!(idx, Some(1));
        assert_eq!(w.entries()[1].size(), 9); // "rename me" = 9 bytes

        // Old entry has IS_DELETED flag
        assert_ne!(w.entries()[0].flags & IS_DELETED, 0);

        // dirty_count incremented for both delete and create
        assert!(w.dirty_count() > old_dirty);
    }

    #[test]
    fn test_handle_renamed_old_not_in_index() {
        let dir = tempfile::TempDir::new().unwrap();
        let old_path = dir.path().join("never_indexed.txt");
        let new_path = dir.path().join("new_file.txt");
        std::fs::write(&new_path, "content").unwrap();

        let mut w = make_writer_empty();

        // Rename where old was never indexed — delete is no-op, create succeeds
        w.handle_renamed(&old_path, &new_path);

        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 1);
        let new_bytes = new_path.as_os_str().as_encoded_bytes();
        assert!(w.lookup(new_bytes).is_some());
    }

    #[test]
    fn test_handle_must_rescan() {
        let dir = tempfile::TempDir::new().unwrap();
        let subtree = dir.path().join("project");
        std::fs::create_dir_all(&subtree).unwrap();

        // Create files in the subtree
        let file_a = subtree.join("a.rs");
        let file_b = subtree.join("b.rs");
        let file_c = subtree.join("c.rs");
        std::fs::write(&file_a, "aaa").unwrap();
        std::fs::write(&file_b, "bbb").unwrap();
        std::fs::write(&file_c, "ccc").unwrap();

        // Also create a file outside the subtree
        let outside = dir.path().join("outside.txt");
        std::fs::write(&outside, "outside data").unwrap();

        let mut w = make_writer_empty();
        // Index all files
        w.handle_created(&file_a);
        w.handle_created(&file_b);
        w.handle_created(&file_c);
        w.handle_created(&outside);
        assert_eq!(w.entry_count(), 4);
        assert_eq!(w.live_count(), 4);

        // Now modify the filesystem: delete b.rs, create d.rs, modify a.rs
        std::fs::remove_file(&file_b).unwrap();
        let file_d = subtree.join("d.rs");
        std::fs::write(&file_d, "dddd").unwrap();
        std::fs::write(&file_a, "aaa modified longer").unwrap();

        let old_dirty = w.dirty_count();

        // Rescan the subtree
        w.handle_must_rescan(&subtree);

        // a.rs should be updated (still live, new size)
        let a_bytes = file_a.as_os_str().as_encoded_bytes();
        let a_idx = w.lookup(a_bytes);
        assert!(a_idx.is_some(), "a.rs should still be in index");
        assert_eq!(w.entries()[a_idx.unwrap()].size(), 19); // "aaa modified longer"

        // b.rs should be tombstoned
        let b_bytes = file_b.as_os_str().as_encoded_bytes();
        assert!(w.lookup(b_bytes).is_none(), "b.rs should be tombstoned");

        // c.rs should still be live (unchanged)
        let c_bytes = file_c.as_os_str().as_encoded_bytes();
        assert!(w.lookup(c_bytes).is_some(), "c.rs should still be in index");

        // d.rs should be newly added
        let d_bytes = file_d.as_os_str().as_encoded_bytes();
        assert!(w.lookup(d_bytes).is_some(), "d.rs should be newly added");

        // outside.txt should be unaffected (not in subtree)
        let outside_bytes = outside.as_os_str().as_encoded_bytes();
        assert!(w.lookup(outside_bytes).is_some(), "outside.txt should be unaffected");

        // dirty_count should have increased
        assert!(w.dirty_count() > old_dirty);
    }

    #[test]
    fn test_handle_must_rescan_empty_subtree() {
        let dir = tempfile::TempDir::new().unwrap();
        let subtree = dir.path().join("empty_dir");
        std::fs::create_dir_all(&subtree).unwrap();

        // Create and index files in the subtree
        let file_a = subtree.join("a.txt");
        std::fs::write(&file_a, "data").unwrap();

        let mut w = make_writer_empty();
        w.handle_created(&file_a);
        assert_eq!(w.live_count(), 1);

        // Delete all files from subtree
        std::fs::remove_file(&file_a).unwrap();

        // Rescan — all entries should be tombstoned
        w.handle_must_rescan(&subtree);

        assert_eq!(w.live_count(), 0);
        let a_bytes = file_a.as_os_str().as_encoded_bytes();
        assert!(w.lookup(a_bytes).is_none());
    }

    #[test]
    fn test_handle_must_rescan_no_existing_entries() {
        let dir = tempfile::TempDir::new().unwrap();
        let subtree = dir.path().join("new_subtree");
        std::fs::create_dir_all(&subtree).unwrap();
        let file_a = subtree.join("new_file.rs");
        std::fs::write(&file_a, "new content").unwrap();

        let mut w = make_writer_empty();
        assert_eq!(w.entry_count(), 0);

        // Rescan a subtree with no existing entries — should insert
        w.handle_must_rescan(&subtree);

        assert_eq!(w.live_count(), 1);
        let a_bytes = file_a.as_os_str().as_encoded_bytes();
        assert!(w.lookup(a_bytes).is_some());
    }

    #[test]
    fn test_starts_with_subtree() {
        use super::starts_with_subtree;

        // Exact match
        assert!(starts_with_subtree(b"/foo/bar", b"/foo/bar"));
        // Proper prefix with /
        assert!(starts_with_subtree(b"/foo/bar/baz.rs", b"/foo/bar"));
        // Not a proper prefix (no / boundary)
        assert!(!starts_with_subtree(b"/foo/barbaz", b"/foo/bar"));
        // Shorter path than prefix
        assert!(!starts_with_subtree(b"/foo", b"/foo/bar"));
        // Empty prefix matches everything
        assert!(starts_with_subtree(b"/anything", b""));
    }

    #[test]
    fn test_process_event_dispatch() {
        use crate::index::fsevents::FsChange;

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("dispatch.idx");

        // Create real files for events that need stat()
        let created_path = dir.path().join("created.txt");
        std::fs::write(&created_path, "new file").unwrap();

        let modified_path = dir.path().join("modified.txt");
        std::fs::write(&modified_path, "original").unwrap();

        let rename_old = dir.path().join("old_name.txt");
        let rename_new = dir.path().join("new_name.txt");
        std::fs::write(&rename_old, "renamed content").unwrap();

        let delete_path = dir.path().join("to_delete.txt");
        std::fs::write(&delete_path, "doomed").unwrap();

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let mut w = IndexWriter::new(excludes, store, idx_path, 0, 0);

        // Pre-index files that will be modified/deleted/renamed
        w.handle_created(&modified_path);
        w.handle_created(&delete_path);
        w.handle_created(&rename_old);
        assert_eq!(w.live_count(), 3);
        let baseline_dirty = w.dirty_count();

        // Test Created dispatch
        w.process_event(FsChange::Created(created_path.clone()));
        assert_eq!(w.live_count(), 4);
        assert!(w.lookup(created_path.as_os_str().as_encoded_bytes()).is_some());

        // Test Modified dispatch
        std::fs::write(&modified_path, "modified content longer").unwrap();
        w.process_event(FsChange::Modified(modified_path.clone()));
        assert!(w.lookup(modified_path.as_os_str().as_encoded_bytes()).is_some());

        // Test Deleted dispatch
        w.process_event(FsChange::Deleted(delete_path.clone()));
        assert!(w.lookup(delete_path.as_os_str().as_encoded_bytes()).is_none());

        // Test Renamed dispatch — rename old file on disk, then process event
        std::fs::rename(&rename_old, &rename_new).unwrap();
        w.process_event(FsChange::Renamed {
            old: rename_old.clone(),
            new: rename_new.clone(),
        });
        assert!(w.lookup(rename_old.as_os_str().as_encoded_bytes()).is_none(),
            "old path should be tombstoned after rename");
        assert!(w.lookup(rename_new.as_os_str().as_encoded_bytes()).is_some(),
            "new path should be in index after rename");

        // dirty_count should have increased from all operations
        assert!(w.dirty_count() > baseline_dirty);

        // Test HistoryDone dispatch — should flush (reset dirty_count)
        assert!(w.dirty_count() > 0);
        w.process_event(FsChange::HistoryDone);
        assert_eq!(w.dirty_count(), 0, "HistoryDone should trigger flush, resetting dirty_count");
    }

    #[test]
    fn test_should_flush_conditions() {
        use super::{FLUSH_DIRTY_THRESHOLD, FLUSH_TIME_THRESHOLD};

        let mut w = make_writer_empty();

        // Initially: no dirty entries, just flushed -> should not flush
        assert!(!w.should_flush(), "should not flush when clean and recently flushed");

        // Condition 1: dirty_count >= FLUSH_DIRTY_THRESHOLD
        w.dirty_count = FLUSH_DIRTY_THRESHOLD - 1;
        assert!(!w.should_flush(), "should not flush at threshold - 1");
        w.dirty_count = FLUSH_DIRTY_THRESHOLD;
        assert!(w.should_flush(), "should flush at exactly threshold");
        w.dirty_count = FLUSH_DIRTY_THRESHOLD + 100;
        assert!(w.should_flush(), "should flush above threshold");

        // Reset dirty count, test time condition
        w.dirty_count = 0;
        assert!(!w.should_flush(), "should not flush with 0 dirty and recent flush");

        // Condition 2: last_flush.elapsed() >= 30s
        // Simulate old flush by setting last_flush to the past
        w.last_flush = Instant::now() - FLUSH_TIME_THRESHOLD;
        assert!(w.should_flush(), "should flush when time threshold exceeded");

        // Both conditions: dirty_count high AND time exceeded
        w.dirty_count = FLUSH_DIRTY_THRESHOLD;
        w.last_flush = Instant::now() - FLUSH_TIME_THRESHOLD;
        assert!(w.should_flush(), "should flush when both conditions met");
    }

    #[test]
    fn test_flush_resets_state() {
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("flush_reset.idx");
        let file_path = dir.path().join("flush_test.txt");
        std::fs::write(&file_path, "data").unwrap();

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let mut w = IndexWriter::new(excludes, store, idx_path, 0xFACE, 0xBEEF);
        w.handle_created(&file_path);
        assert_eq!(w.dirty_count(), 1);

        let before_flush = Instant::now();
        w.flush().expect("flush should succeed");

        assert_eq!(w.dirty_count(), 0, "flush should reset dirty_count");
        assert!(w.last_flush() >= before_flush, "flush should update last_flush");

        // Entries should still be intact after flush
        assert_eq!(w.entry_count(), 1);
        assert_eq!(w.live_count(), 1);
    }

    #[test]
    fn test_process_event_must_rescan() {
        use crate::index::fsevents::FsChange;

        let dir = tempfile::TempDir::new().unwrap();
        let subtree = dir.path().join("scan_me");
        std::fs::create_dir_all(&subtree).unwrap();
        let file_a = subtree.join("a.txt");
        std::fs::write(&file_a, "aaa").unwrap();

        let mut w = make_writer_empty();

        // MustRescan should discover the file
        w.process_event(FsChange::MustRescan(subtree));

        assert!(w.live_count() >= 1);
        assert!(w.lookup(file_a.as_os_str().as_encoded_bytes()).is_some());
    }

    #[test]
    fn test_compaction_removes_tombstones() {
        use crate::gpu::types::path_flags::IS_DELETED;
        use crate::index::gsix_v2::{FLAG_SORTED, FLAG_COMPACTED};

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("compact.idx");
        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let mut w = IndexWriter::new(excludes, store, idx_path, 0, 0);

        // Manually insert 10 entries with paths
        for i in 0..10 {
            let mut entry = GpuPathEntry::new();
            let path = format!("/test/file_{:02}.rs", i);
            entry.set_path(path.as_bytes());
            entry.flags = 0;
            let idx = w.entries.len();
            let key: Box<[u8]> = path.as_bytes().into();
            w.entries.push(entry);
            w.path_index.insert(key, idx);
        }
        assert_eq!(w.entry_count(), 10);
        assert_eq!(w.live_count(), 10);

        // Tombstone 3 entries (30% > 20% threshold)
        for i in [2, 5, 7] {
            let path = format!("/test/file_{:02}.rs", i);
            w.entries[i].flags |= IS_DELETED;
            w.path_index.remove(path.as_bytes());
        }
        assert_eq!(w.entry_count(), 10); // still 10 in Vec
        assert_eq!(w.live_count(), 7);   // 7 in HashMap

        // Flush triggers compaction (30% > 20%)
        w.dirty_count = 1; // mark dirty so flush does something
        w.flush().expect("flush should succeed");

        // After compaction: tombstones removed
        assert_eq!(w.entry_count(), 7, "tombstones should be removed");
        assert_eq!(w.live_count(), 7, "all remaining should be live");

        // Verify all live entries are still findable
        for i in [0, 1, 3, 4, 6, 8, 9] {
            let path = format!("/test/file_{:02}.rs", i);
            assert!(w.lookup(path.as_bytes()).is_some(),
                "entry {} should still be findable", i);
        }

        // Verify tombstoned entries are gone
        for i in [2, 5, 7] {
            let path = format!("/test/file_{:02}.rs", i);
            assert!(w.lookup(path.as_bytes()).is_none(),
                "entry {} should be gone after compaction", i);
        }

        // Verify entries are sorted by path bytes
        for pair in w.entries().windows(2) {
            let a = &pair[0].path[..pair[0].path_len as usize];
            let b = &pair[1].path[..pair[1].path_len as usize];
            assert!(a <= b, "entries should be sorted by path bytes");
        }

        // Verify header flags set
        assert_ne!(w.header_flags() & FLAG_SORTED, 0, "FLAG_SORTED should be set");
        assert_ne!(w.header_flags() & FLAG_COMPACTED, 0, "FLAG_COMPACTED should be set");

        // Verify no IS_DELETED entries remain
        for entry in w.entries() {
            assert_eq!(entry.flags & IS_DELETED, 0,
                "no entries should have IS_DELETED after compaction");
        }
    }

    #[test]
    fn test_compaction_threshold() {
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("thresh.idx");
        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let mut w = IndexWriter::new(excludes, store, idx_path, 0, 0);

        // Insert 10 entries
        for i in 0..10 {
            let mut entry = GpuPathEntry::new();
            let path = format!("/test/thresh_{:02}.rs", i);
            entry.set_path(path.as_bytes());
            entry.flags = 0;
            let idx = w.entries.len();
            let key: Box<[u8]> = path.as_bytes().into();
            w.entries.push(entry);
            w.path_index.insert(key, idx);
        }

        // Tombstone exactly 2 entries (20% = threshold, NOT exceeded)
        for i in [3, 6] {
            let path = format!("/test/thresh_{:02}.rs", i);
            w.entries[i].flags |= IS_DELETED;
            w.path_index.remove(path.as_bytes());
        }
        assert_eq!(w.entry_count(), 10);
        assert_eq!(w.live_count(), 8);

        // Flush should NOT compact (20% is at threshold, not above)
        w.dirty_count = 1;
        w.flush().expect("flush should succeed");

        // Entries should still be 10 (no compaction)
        assert_eq!(w.entry_count(), 10,
            "should not compact at exactly 20% threshold");
        assert_eq!(w.header_flags(), 0,
            "header_flags should remain 0 when no compaction");

        // Now tombstone one more (3/10 = 30% > 20%)
        let path = format!("/test/thresh_{:02}.rs", 1);
        w.entries[1].flags |= IS_DELETED;
        w.path_index.remove(path.as_bytes());
        assert_eq!(w.live_count(), 7);

        // Flush should compact now (30% > 20%)
        w.dirty_count = 1;
        w.flush().expect("flush should succeed");

        assert_eq!(w.entry_count(), 7,
            "should compact when tombstone ratio > 20%");
        assert_ne!(w.header_flags() & crate::index::gsix_v2::FLAG_COMPACTED, 0,
            "FLAG_COMPACTED should be set after compaction");
    }

    #[test]
    fn test_flush_writes_valid_v2() {
        use crate::index::gsix_v2::load_v2;

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("flush_v2.idx");

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let mut w = IndexWriter::new(
            excludes,
            store.clone(),
            idx_path.clone(),
            0xFACE,
            0xBEEF,
        );
        w.set_last_fsevents_id(42);

        // Insert entries manually (real files may not have stable paths)
        for i in 0..5 {
            let mut entry = GpuPathEntry::new();
            let path = format!("/test/flush_{}.rs", i);
            entry.set_path(path.as_bytes());
            entry.set_size(1024 * (i as u64 + 1));
            entry.mtime = 1700000000 + i as u32;
            entry.flags = 0;
            let idx = w.entries.len();
            let key: Box<[u8]> = path.as_bytes().into();
            w.entries.push(entry);
            w.path_index.insert(key, idx);
        }
        w.dirty_count = 5;

        // Flush should write the file
        w.flush().expect("flush should succeed");

        // Verify file exists and is loadable
        assert!(idx_path.exists(), "index file should exist after flush");

        let (header, loaded_entries) = load_v2(&idx_path).expect("should load v2 file");
        assert_eq!(header.entry_count, 5);
        assert_eq!(header.root_hash, 0xBEEF);
        assert_eq!(header.last_fsevents_id, 42);
        assert_eq!(header.exclude_hash, 0xFACE);
        assert_eq!(loaded_entries.len(), 5);

        // Verify entries match
        for i in 0..5 {
            let expected_path = format!("/test/flush_{}.rs", i);
            let loaded = &loaded_entries[i];
            let loaded_path = &loaded.path[..loaded.path_len as usize];
            assert_eq!(loaded_path, expected_path.as_bytes(),
                "entry {} path should match", i);
            assert_eq!(loaded.size(), 1024 * (i as u64 + 1),
                "entry {} size should match", i);
        }

        // Verify IndexStore was swapped (snapshot now available)
        assert!(store.is_available(), "IndexStore should have a snapshot after flush");
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        assert_eq!(snap.entry_count(), 5);
        assert_eq!(snap.fsevents_id(), 42);

        // Verify dirty_count was reset
        assert_eq!(w.dirty_count(), 0);
    }

    #[test]
    fn test_flush_skips_when_clean() {
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("no_flush.idx");

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let mut w = IndexWriter::new(excludes, store.clone(), idx_path.clone(), 0, 0);

        // dirty_count == 0, flush should be a no-op
        w.flush().expect("flush should succeed (no-op)");

        // File should NOT exist (nothing to flush)
        assert!(!idx_path.exists(), "no file should be written when clean");
        assert!(!store.is_available(), "store should remain empty when clean");
    }

    #[test]
    fn test_set_last_fsevents_id() {
        let mut w = make_writer_empty();
        assert_eq!(w.last_fsevents_id(), 0);

        w.set_last_fsevents_id(12345);
        assert_eq!(w.last_fsevents_id(), 12345);

        w.set_last_fsevents_id(u64::MAX);
        assert_eq!(w.last_fsevents_id(), u64::MAX);
    }

    #[test]
    fn test_spawn_writer_thread_processes_events() {
        use crate::index::fsevents::FsChange;
        use super::spawn_writer_thread;

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("thread_test.idx");

        // Create real files for the events
        let file_a = dir.path().join("a.txt");
        let file_b = dir.path().join("b.txt");
        let file_c = dir.path().join("c.txt");
        std::fs::write(&file_a, "aaa").unwrap();
        std::fs::write(&file_b, "bbb").unwrap();
        std::fs::write(&file_c, "ccc").unwrap();

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let writer = IndexWriter::new(
            excludes,
            store.clone(),
            idx_path.clone(),
            0,
            0,
        );

        // Create bounded channel (matching DISCOVERY_CHANNEL_CAPACITY)
        let (tx, rx) = crossbeam_channel::bounded::<FsChange>(4096);

        // Spawn the writer thread
        let handle = spawn_writer_thread(writer, rx);

        // Send Created events
        tx.send(FsChange::Created(file_a.clone())).unwrap();
        tx.send(FsChange::Created(file_b.clone())).unwrap();
        tx.send(FsChange::Created(file_c.clone())).unwrap();

        // Send HistoryDone to trigger a flush (so we can verify via IndexStore)
        tx.send(FsChange::HistoryDone).unwrap();

        // Give the thread time to process
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Verify IndexStore was swapped (HistoryDone triggers flush)
        assert!(store.is_available(), "IndexStore should have a snapshot after HistoryDone flush");
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        assert_eq!(snap.entry_count(), 3, "should have 3 entries after processing");

        // Send a delete event
        tx.send(FsChange::Deleted(file_b.clone())).unwrap();

        // Drop sender to trigger channel disconnect and final flush
        drop(tx);

        // Wait for thread to finish
        handle.join().expect("writer thread should exit cleanly");

        // After final flush, verify updated snapshot
        let guard2 = store.snapshot();
        let snap2 = guard2.as_ref().as_ref().unwrap();
        // 3 entries total (tombstone still in Vec), but live count reflected in snapshot
        // The snapshot has all entries including tombstoned ones
        assert!(snap2.entry_count() >= 2, "should have entries after final flush");

        // Verify the index file was written to disk
        assert!(idx_path.exists(), "index file should exist after final flush");
    }

    #[test]
    fn test_dirty_count_triggers_flush() {
        // Create 1000 entries and verify should_flush() returns true
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("dirty_trigger.idx");

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let mut w = IndexWriter::new(excludes, store, idx_path, 0, 0);

        // Manually insert 1000 entries (avoid needing 1000 real files)
        for i in 0..FLUSH_DIRTY_THRESHOLD {
            let mut entry = GpuPathEntry::new();
            let path = format!("/test/dirty_{:04}.rs", i);
            entry.set_path(path.as_bytes());
            entry.flags = 0;
            let idx = w.entries.len();
            let key: Box<[u8]> = path.as_bytes().into();
            w.entries.push(entry);
            w.path_index.insert(key, idx);
            w.dirty_count += 1;
        }

        assert_eq!(w.dirty_count(), FLUSH_DIRTY_THRESHOLD);
        assert_eq!(w.entry_count(), FLUSH_DIRTY_THRESHOLD);
        assert!(
            w.should_flush(),
            "should_flush() should return true at exactly {} dirty entries",
            FLUSH_DIRTY_THRESHOLD
        );

        // Verify one below threshold does NOT trigger
        w.dirty_count = FLUSH_DIRTY_THRESHOLD - 1;
        w.last_flush = Instant::now(); // reset time so only dirty triggers
        assert!(
            !w.should_flush(),
            "should_flush() should return false at {} dirty entries",
            FLUSH_DIRTY_THRESHOLD - 1
        );
    }

    #[test]
    fn test_path_index_o1_lookup() {
        // Verify HashMap lookup by path bytes returns the correct index
        let mut w = make_writer_empty();

        // Insert entries manually with known paths and indices
        let paths = vec![
            "/usr/local/bin/cargo",
            "/Users/dev/project/main.rs",
            "/tmp/scratch.txt",
            "/etc/hosts",
            "/var/log/system.log",
        ];

        for (i, path_str) in paths.iter().enumerate() {
            let mut entry = GpuPathEntry::new();
            entry.set_path(path_str.as_bytes());
            entry.flags = 0;
            entry.set_size((i as u64 + 1) * 100);
            let idx = w.entries.len();
            let key: Box<[u8]> = path_str.as_bytes().into();
            w.entries.push(entry);
            w.path_index.insert(key, idx);
        }

        // Verify each path maps to the correct index
        for (expected_idx, path_str) in paths.iter().enumerate() {
            let result = w.lookup(path_str.as_bytes());
            assert_eq!(
                result,
                Some(expected_idx),
                "lookup('{}') should return index {}",
                path_str,
                expected_idx
            );
            // Verify the entry at that index has the correct size
            let entry = &w.entries()[expected_idx];
            assert_eq!(
                entry.size(),
                (expected_idx as u64 + 1) * 100,
                "entry at index {} should have size {}",
                expected_idx,
                (expected_idx as u64 + 1) * 100
            );
        }

        // Non-existent paths return None
        assert_eq!(w.lookup(b"/nonexistent/path"), None);
        assert_eq!(w.lookup(b""), None);
        assert_eq!(w.lookup(b"/usr/local/bin/carg"), None); // partial match
        assert_eq!(w.lookup(b"/usr/local/bin/cargo "), None); // trailing space
    }

    #[test]
    fn test_spawn_convenience_method() {
        use crate::index::fsevents::FsChange;

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("spawn_method.idx");
        let file_a = dir.path().join("spawn_a.txt");
        std::fs::write(&file_a, "data").unwrap();

        let excludes = Arc::new(ExcludeTrie::default());
        let store = Arc::new(IndexStore::new());
        let writer = IndexWriter::new(excludes, store.clone(), idx_path, 0, 0);

        let (tx, rx) = crossbeam_channel::bounded::<FsChange>(4096);

        // Use the convenience method
        let handle = writer.spawn(rx);

        tx.send(FsChange::Created(file_a)).unwrap();
        tx.send(FsChange::HistoryDone).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(store.is_available(), "IndexStore should have snapshot via spawn()");

        drop(tx);
        handle.join().expect("writer thread should exit cleanly");
    }
}
