//! IndexStore: lock-free atomic snapshot store using arc-swap.
//!
//! Provides wait-free readers via [`arc_swap::ArcSwap`] so GPU search threads
//! can obtain the current [`IndexSnapshot`] without any mutex contention.
//! Writers call [`IndexStore::swap`] to atomically replace the snapshot,
//! making the new data visible to all subsequent readers.
//!
//! # Concurrency Model
//!
//! ```text
//! Reader threads (N)            Writer thread (1)
//!   |                             |
//!   +-- store.snapshot()          +-- store.swap(new_snapshot)
//!   |   (lock-free load)          |   (atomic pointer swap)
//!   |                             |
//!   +-- guard holds Arc ref       +-- old snapshot dropped when
//!       keeping snapshot alive        last reader releases guard
//! ```

use std::sync::Arc;

use arc_swap::{ArcSwap, Guard};

use crate::index::snapshot::IndexSnapshot;

/// Lock-free atomic store for the current index snapshot.
///
/// Wraps an `ArcSwap<Option<IndexSnapshot>>` so that:
/// - Readers get a [`Guard`] without blocking (wait-free on most architectures)
/// - The writer atomically swaps in a new snapshot
/// - Old snapshots are dropped only after the last reader releases its guard
pub struct IndexStore {
    inner: ArcSwap<Option<IndexSnapshot>>,
}

impl IndexStore {
    /// Create a new empty store (no snapshot loaded yet).
    pub fn new() -> Self {
        Self {
            inner: ArcSwap::from_pointee(None),
        }
    }

    /// Create a store pre-loaded with a snapshot.
    pub fn with_snapshot(snapshot: IndexSnapshot) -> Self {
        Self {
            inner: ArcSwap::from_pointee(Some(snapshot)),
        }
    }

    /// Get a lock-free guard to the current snapshot.
    ///
    /// The returned [`Guard`] keeps the snapshot alive for the duration of
    /// its borrow. Multiple readers can hold guards simultaneously without
    /// contention.
    #[inline]
    pub fn snapshot(&self) -> Guard<Arc<Option<IndexSnapshot>>> {
        self.inner.load()
    }

    /// Atomically swap in a new snapshot, replacing the current one.
    ///
    /// After this call, all new readers will see the new snapshot. Existing
    /// readers holding a [`Guard`] to the old snapshot continue to use it
    /// safely until they drop their guard.
    pub fn swap(&self, new: IndexSnapshot) {
        self.inner.store(Arc::new(Some(new)));
    }

    /// Check if a snapshot is currently available.
    ///
    /// Returns `true` if a snapshot has been loaded, `false` if the store
    /// is empty (e.g., before the first index build completes).
    #[inline]
    pub fn is_available(&self) -> bool {
        let guard = self.inner.load();
        guard.is_some()
    }
}

impl Default for IndexStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_initially_empty() {
        let store = IndexStore::new();
        assert!(!store.is_available());
        let guard = store.snapshot();
        assert!(guard.is_none());
    }

    #[test]
    fn test_store_default_is_empty() {
        let store = IndexStore::default();
        assert!(!store.is_available());
    }

    // ================================================================
    // mmap pipeline tests (task 2.9)
    // ================================================================

    #[test]
    fn test_index_store_snapshot_consistent() {
        // Snapshot loaded via IndexStore returns consistent data:
        // entry_count, entries, header fields all match the saved file.
        use crate::index::gsix_v2::save_v2;
        use crate::gpu::types::GpuPathEntry;

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("consistent.idx");

        let mut entries = Vec::new();
        for i in 0..12 {
            let mut entry = GpuPathEntry::new();
            entry.set_path(format!("/store/test/file_{}.txt", i).as_bytes());
            entry.flags = i as u32;
            entry.set_size(512 * (i as u64 + 1));
            entry.mtime = 1700000000 + i as u32;
            entries.push(entry);
        }

        save_v2(&entries, 0xAAAA, &idx_path, 777, 0xBBBB).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, None).unwrap();

        // Verify consistency before storing
        assert_eq!(snapshot.entry_count(), 12);
        assert_eq!(snapshot.header().root_hash, 0xAAAA);
        assert_eq!(snapshot.fsevents_id(), 777);
        assert_eq!(snapshot.header().exclude_hash, 0xBBBB);

        let store = IndexStore::with_snapshot(snapshot);
        assert!(store.is_available());

        // Read via store.snapshot() guard and verify same data
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        assert_eq!(snap.entry_count(), 12);
        assert_eq!(snap.header().root_hash, 0xAAAA);
        assert_eq!(snap.fsevents_id(), 777);

        // Verify entries are accessible and correct
        let loaded_entries = snap.entries();
        assert_eq!(loaded_entries.len(), 12);
        for (i, entry) in loaded_entries.iter().enumerate() {
            let path = std::str::from_utf8(&entry.path[..entry.path_len as usize]).unwrap();
            assert!(
                path.contains(&format!("file_{}", i)),
                "Entry {} path mismatch: {}",
                i,
                path
            );
        }
    }

    #[test]
    fn test_index_store_swap_visible() {
        // After swap(), new readers see the new snapshot, old guards still work.
        use crate::index::gsix_v2::save_v2;
        use crate::gpu::types::GpuPathEntry;

        let dir = tempfile::TempDir::new().unwrap();

        // Create first snapshot with 5 entries
        let idx_path_1 = dir.path().join("v1.idx");
        let mut entries_1 = Vec::new();
        for i in 0..5 {
            let mut entry = GpuPathEntry::new();
            entry.set_path(format!("/swap/old_{}.rs", i).as_bytes());
            entries_1.push(entry);
        }
        save_v2(&entries_1, 0x1111, &idx_path_1, 100, 0).unwrap();
        let snap_1 = IndexSnapshot::from_file(&idx_path_1, None).unwrap();

        // Create second snapshot with 10 entries
        let idx_path_2 = dir.path().join("v2.idx");
        let mut entries_2 = Vec::new();
        for i in 0..10 {
            let mut entry = GpuPathEntry::new();
            entry.set_path(format!("/swap/new_{}.rs", i).as_bytes());
            entries_2.push(entry);
        }
        save_v2(&entries_2, 0x2222, &idx_path_2, 200, 0).unwrap();
        let snap_2 = IndexSnapshot::from_file(&idx_path_2, None).unwrap();

        let store = IndexStore::with_snapshot(snap_1);

        // Take a guard to the old snapshot
        let old_guard = store.snapshot();
        let old_snap = old_guard.as_ref().as_ref().unwrap();
        assert_eq!(old_snap.entry_count(), 5);
        assert_eq!(old_snap.header().root_hash, 0x1111);

        // Swap in the new snapshot
        store.swap(snap_2);

        // New readers see the new snapshot
        let new_guard = store.snapshot();
        let new_snap = new_guard.as_ref().as_ref().unwrap();
        assert_eq!(new_snap.entry_count(), 10);
        assert_eq!(new_snap.header().root_hash, 0x2222);
        assert_eq!(new_snap.fsevents_id(), 200);

        // Old guard still sees old data (lock-free snapshot isolation)
        assert_eq!(old_snap.entry_count(), 5);
        assert_eq!(old_snap.header().root_hash, 0x1111);
    }
}
