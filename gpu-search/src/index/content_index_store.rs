//! ContentIndexStore: lock-free atomic snapshot store using arc-swap.
//!
//! Provides wait-free readers via [`arc_swap::ArcSwap`] so GPU search threads
//! can obtain the current [`ContentSnapshot`] without any mutex contention.
//! Writers call [`ContentIndexStore::swap`] to atomically replace the snapshot,
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

use crate::index::content_snapshot::ContentSnapshot;

/// Lock-free atomic store for the current content index snapshot.
///
/// Wraps an `ArcSwap<Option<ContentSnapshot>>` so that:
/// - Readers get a [`Guard`] without blocking (wait-free on most architectures)
/// - The writer atomically swaps in a new snapshot
/// - Old snapshots are dropped only after the last reader releases its guard
pub struct ContentIndexStore {
    inner: ArcSwap<Option<ContentSnapshot>>,
}

impl ContentIndexStore {
    /// Create a new empty store (no snapshot loaded yet).
    pub fn new() -> Self {
        Self {
            inner: ArcSwap::from_pointee(None),
        }
    }

    /// Get a lock-free guard to the current snapshot.
    ///
    /// The returned [`Guard`] keeps the snapshot alive for the duration of
    /// its borrow. Multiple readers can hold guards simultaneously without
    /// contention.
    #[inline]
    pub fn snapshot(&self) -> Guard<Arc<Option<ContentSnapshot>>> {
        self.inner.load()
    }

    /// Atomically swap in a new snapshot, replacing the current one.
    ///
    /// After this call, all new readers will see the new snapshot. Existing
    /// readers holding a [`Guard`] to the old snapshot continue to use it
    /// safely until they drop their guard.
    pub fn swap(&self, new: ContentSnapshot) {
        self.inner.store(Arc::new(Some(new)));
    }

    /// Check if a snapshot is currently available.
    ///
    /// Returns `true` if a snapshot has been loaded, `false` if the store
    /// is empty (e.g., before the first content build completes).
    #[inline]
    pub fn is_available(&self) -> bool {
        let guard = self.inner.load();
        guard.is_some()
    }
}

impl Default for ContentIndexStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::content_store::ContentStore;

    fn make_snapshot(file_count: usize, timestamp: u64) -> ContentSnapshot {
        let mut store = ContentStore::new();
        for i in 0..file_count {
            let content = format!("content for file {}", i);
            store.insert(content.as_bytes(), i as u32, 0, 0);
        }
        ContentSnapshot::new(store, timestamp)
    }

    #[test]
    fn test_store_initially_empty() {
        let store = ContentIndexStore::new();
        assert!(!store.is_available());
        let guard = store.snapshot();
        assert!(guard.is_none());
    }

    #[test]
    fn test_store_default_is_empty() {
        let store = ContentIndexStore::default();
        assert!(!store.is_available());
    }

    #[test]
    fn test_swap_makes_snapshot_visible() {
        let store = ContentIndexStore::new();
        assert!(!store.is_available());

        let snapshot = make_snapshot(5, 1700000000);
        store.swap(snapshot);

        assert!(store.is_available());

        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        assert_eq!(snap.file_count(), 5);
        assert_eq!(snap.build_timestamp(), 1700000000);

        // Verify content is accessible through the snapshot
        let content = snap.content_store().content_for(0).unwrap();
        assert_eq!(content, b"content for file 0");
    }

    #[test]
    fn test_old_guard_sees_old_data_after_swap() {
        let store = ContentIndexStore::new();

        // Load first snapshot with 3 files
        let snap_v1 = make_snapshot(3, 1000);
        store.swap(snap_v1);

        // Take a guard to the old snapshot
        let old_guard = store.snapshot();
        let old_snap = old_guard.as_ref().as_ref().unwrap();
        assert_eq!(old_snap.file_count(), 3);
        assert_eq!(old_snap.build_timestamp(), 1000);

        // Swap in a new snapshot with 10 files
        let snap_v2 = make_snapshot(10, 2000);
        store.swap(snap_v2);

        // New readers see the new snapshot
        let new_guard = store.snapshot();
        let new_snap = new_guard.as_ref().as_ref().unwrap();
        assert_eq!(new_snap.file_count(), 10);
        assert_eq!(new_snap.build_timestamp(), 2000);

        // Old guard still sees old data (lock-free snapshot isolation)
        assert_eq!(old_snap.file_count(), 3);
        assert_eq!(old_snap.build_timestamp(), 1000);

        // Verify content through old guard
        let old_content = old_snap.content_store().content_for(0).unwrap();
        assert_eq!(old_content, b"content for file 0");

        // Verify content through new guard
        let new_content = new_snap.content_store().content_for(9).unwrap();
        assert_eq!(new_content, b"content for file 9");
    }

    #[test]
    fn test_multiple_swaps() {
        let store = ContentIndexStore::new();

        for i in 0..5 {
            let snapshot = make_snapshot(i + 1, (i + 1) as u64 * 1000);
            store.swap(snapshot);

            let guard = store.snapshot();
            let snap = guard.as_ref().as_ref().unwrap();
            assert_eq!(snap.file_count(), (i + 1) as u32);
        }
    }
}
