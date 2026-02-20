//! ContentSnapshot: immutable content index snapshot for arc-swap.
//!
//! Bundles a [`ContentStore`] with build metadata. Used as the inner
//! value of [`ContentIndexStore`]'s `ArcSwap<Option<ContentSnapshot>>`.
//!
//! # Lifetime Semantics
//!
//! ```text
//! ContentSnapshot
//!   |-- content_store: ContentStore  (owns mmap + Metal buffer)
//!   |-- build_timestamp: u64         (epoch seconds when build completed)
//!   +-- file_count: u32              (number of files in the store)
//! ```

use crate::index::content_store::ContentStore;

/// An immutable snapshot of the content index, holding all file contents
/// in memory with optional GPU-accessible Metal buffer.
///
/// Created by `ContentBuilder` after a full or incremental build.
/// Shared across threads via `Arc<ContentSnapshot>` inside
/// [`ContentIndexStore`].
pub struct ContentSnapshot {
    /// The content store holding file data and metadata.
    content_store: ContentStore,
    /// Timestamp (seconds since epoch) when this snapshot was built.
    build_timestamp: u64,
    /// Number of files in this snapshot.
    file_count: u32,
}

// SAFETY: ContentStore is Send+Sync (mmap is read-only after finalization,
// Retained<MTLBuffer> is thread-safe for read access on Apple platforms,
// Vec<FileContentMeta> and scalars are inherently Send+Sync).
// build_timestamp and file_count are plain data.
unsafe impl Send for ContentSnapshot {}
unsafe impl Sync for ContentSnapshot {}

impl ContentSnapshot {
    /// Create a new content snapshot from a finalized content store.
    pub fn new(content_store: ContentStore, build_timestamp: u64) -> Self {
        let file_count = content_store.file_count();
        Self {
            content_store,
            build_timestamp,
            file_count,
        }
    }

    /// Get a reference to the underlying content store.
    #[inline]
    pub fn content_store(&self) -> &ContentStore {
        &self.content_store
    }

    /// Get the build timestamp (seconds since epoch).
    #[inline]
    pub fn build_timestamp(&self) -> u64 {
        self.build_timestamp
    }

    /// Get the number of files in this snapshot.
    #[inline]
    pub fn file_count(&self) -> u32 {
        self.file_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_new() {
        let store = ContentStore::new();
        let snapshot = ContentSnapshot::new(store, 1700000000);

        assert_eq!(snapshot.build_timestamp(), 1700000000);
        assert_eq!(snapshot.file_count(), 0);
        assert_eq!(snapshot.content_store().total_bytes(), 0);
    }

    #[test]
    fn test_snapshot_with_content() {
        let mut store = ContentStore::new();
        store.insert(b"hello world", 0, 0xABCD, 1000);
        store.insert(b"fn main() {}", 1, 0xEF01, 2000);

        let snapshot = ContentSnapshot::new(store, 1700000042);

        assert_eq!(snapshot.file_count(), 2);
        assert_eq!(snapshot.build_timestamp(), 1700000042);
        assert_eq!(
            snapshot.content_store().content_for(0).unwrap(),
            b"hello world"
        );
        assert_eq!(
            snapshot.content_store().content_for(1).unwrap(),
            b"fn main() {}"
        );
    }
}
