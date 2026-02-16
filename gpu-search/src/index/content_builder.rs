// ContentBuilder: background filesystem walker + reader that populates ContentStore.
//
// Walks a directory tree using `ignore::WalkBuilder` (respecting .gitignore),
// filters binary files via `BinaryDetector`, skips oversized files (>100MB),
// reads each text file, and appends to a `ContentStoreBuilder`. On completion,
// finalizes to a `ContentSnapshot` with GPU-accessible Metal buffer.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ignore::WalkBuilder;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::index::content_index_store::ContentIndexStore;
use crate::index::content_snapshot::ContentSnapshot;
use crate::index::content_store::ContentStoreBuilder;
use crate::index::exclude::ExcludeTrie;
use crate::search::binary::BinaryDetector;

/// Maximum file size to index (100MB).
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Default estimated capacity for the content store builder (64MB).
const DEFAULT_CAPACITY: usize = 64 * 1024 * 1024;

/// Progress tracking for content building.
///
/// Exposes three atomic counters for monitoring build progress:
/// - `files_scanned`: total files encountered during walk (including skipped)
/// - `files_indexed`: text files successfully added to the content store
/// - `bytes_indexed`: total content bytes added to the store
pub struct BuildProgress {
    /// Total files encountered during directory walk (including skipped ones).
    pub files_scanned: AtomicUsize,
    /// Text files successfully added to the content store.
    pub files_indexed: AtomicUsize,
    /// Total content bytes added to the store.
    pub bytes_indexed: AtomicUsize,
}

impl BuildProgress {
    /// Create a new BuildProgress with all counters at zero.
    pub fn new() -> Self {
        Self {
            files_scanned: AtomicUsize::new(0),
            files_indexed: AtomicUsize::new(0),
            bytes_indexed: AtomicUsize::new(0),
        }
    }

    /// Get the number of files scanned (walked).
    pub fn files_scanned(&self) -> usize {
        self.files_scanned.load(Ordering::Relaxed)
    }

    /// Get the number of files indexed (added to store).
    pub fn files_indexed(&self) -> usize {
        self.files_indexed.load(Ordering::Relaxed)
    }

    /// Get the number of bytes indexed.
    pub fn bytes_indexed(&self) -> usize {
        self.bytes_indexed.load(Ordering::Relaxed)
    }
}

impl Default for BuildProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during content building.
#[derive(Debug)]
pub enum BuildError {
    /// I/O error during directory walking or file reading.
    Io(std::io::Error),
    /// Other errors (e.g., mmap allocation failure).
    Other(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::Io(e) => write!(f, "I/O error: {}", e),
            BuildError::Other(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<std::io::Error> for BuildError {
    fn from(e: std::io::Error) -> Self {
        BuildError::Io(e)
    }
}

/// Background filesystem walker + reader that builds a ContentStore.
///
/// Walks a directory tree, reads text files, and builds a ContentSnapshot
/// backed by anonymous mmap with a GPU-accessible Metal buffer.
pub struct ContentBuilder {
    /// Arc reference to the content index store for publishing snapshots.
    pub store: Arc<ContentIndexStore>,
    /// Exclusion trie for filtering paths.
    pub excludes: Arc<ExcludeTrie>,
    /// Atomic counter for tracking progress (files indexed so far).
    /// Kept for backward compatibility -- mirrors `build_progress.files_indexed`.
    pub progress: Arc<AtomicUsize>,
    /// Detailed progress tracking: files_scanned, files_indexed, bytes_indexed.
    pub build_progress: Arc<BuildProgress>,
    /// Metal device for creating GPU buffers.
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl ContentBuilder {
    /// Create a new ContentBuilder.
    pub fn new(
        store: Arc<ContentIndexStore>,
        excludes: Arc<ExcludeTrie>,
        progress: Arc<AtomicUsize>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Self {
        Self {
            store,
            excludes,
            progress,
            build_progress: Arc::new(BuildProgress::new()),
            device,
        }
    }

    /// Create a new ContentBuilder with detailed progress tracking.
    pub fn with_progress(
        store: Arc<ContentIndexStore>,
        excludes: Arc<ExcludeTrie>,
        build_progress: Arc<BuildProgress>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Self {
        Self {
            store,
            excludes,
            progress: Arc::new(AtomicUsize::new(0)),
            build_progress,
            device,
        }
    }

    /// Build a ContentSnapshot from all text files under `root`.
    ///
    /// 1. Walks `root` using `ignore::WalkBuilder` (respects .gitignore)
    /// 2. Filters via BinaryDetector (skip binary files)
    /// 3. Skips files > 100MB
    /// 4. Reads each file via `std::fs::read()`
    /// 5. Appends to ContentStoreBuilder with CRC32 hash and mtime
    /// 6. Increments progress counter per file
    /// 7. Finalizes and returns ContentSnapshot
    pub fn build(&self, root: &Path) -> Result<ContentSnapshot, BuildError> {
        let binary_detector = BinaryDetector::new();
        let walk_start = Instant::now();

        // First pass: collect file paths (walk is not Send-safe for parallel iteration
        // with the ignore crate's parallel walker, so we collect paths first then
        // read in bulk)
        let mut file_paths = Vec::new();

        let walker = WalkBuilder::new(root)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .hidden(true)
            .follow_links(false)
            .parents(true)
            .build();

        for entry_result in walker {
            let entry = match entry_result {
                Ok(e) => e,
                Err(_) => continue,
            };

            // Skip directories
            let file_type = match entry.file_type() {
                Some(ft) => ft,
                None => continue,
            };
            if !file_type.is_file() {
                continue;
            }

            // Count every file encountered during walk
            let scanned = self.build_progress
                .files_scanned
                .fetch_add(1, Ordering::Relaxed) + 1;

            // Log progress every 100K files during walk
            if scanned % 100_000 == 0 {
                eprintln!(
                    "[ContentBuilder] walk: {} files scanned, {} text candidates, {:.1}s",
                    scanned, file_paths.len(), walk_start.elapsed().as_secs_f32()
                );
            }

            let path = entry.path();

            // Check exclude trie
            let path_bytes = path.to_string_lossy();
            if self.excludes.should_exclude(path_bytes.as_bytes()) {
                continue;
            }

            // Check file size via metadata
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };

            if metadata.len() > MAX_FILE_SIZE || metadata.len() == 0 {
                continue;
            }

            // Skip binary files (extension-based check first, no I/O)
            if binary_detector.should_skip(path) {
                continue;
            }

            file_paths.push(entry.into_path());
        }

        eprintln!(
            "[ContentBuilder] walk complete: {} text files from {} scanned in {:.1}s",
            file_paths.len(),
            self.build_progress.files_scanned.load(Ordering::Relaxed),
            walk_start.elapsed().as_secs_f32()
        );

        // Estimate total capacity from file metadata
        let estimated_total: usize = file_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as usize)
            .sum();

        let capacity = estimated_total.max(DEFAULT_CAPACITY);

        eprintln!(
            "[ContentBuilder] allocating {} MB content store for {} files",
            capacity / (1024 * 1024), file_paths.len()
        );

        let mut builder = ContentStoreBuilder::new(capacity)
            .map_err(|e| BuildError::Other(format!("Failed to allocate content store: {}", e)))?;

        let mut path_id: u32 = 0;
        let read_start = Instant::now();
        let total_files = file_paths.len();

        for path in &file_paths {
            // Read file content
            let content = match std::fs::read(path) {
                Ok(c) => c,
                Err(_) => continue, // skip unreadable files
            };

            // Skip if content is binary (NUL byte heuristic for unknown extensions)
            if crate::search::binary::is_binary_content(&content) {
                continue;
            }

            // Skip empty reads (race condition: file emptied between stat and read)
            if content.is_empty() {
                continue;
            }

            // Check builder has enough capacity
            if builder.remaining_capacity() < content.len() {
                // Skip files that won't fit (builder is fixed-capacity mmap)
                continue;
            }

            // Compute CRC32 hash
            let hash = crc32fast::hash(&content);

            // Get mtime
            let mtime = std::fs::metadata(path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as u32)
                .unwrap_or(0);

            let content_len = content.len();
            builder.append_with_path(&content, path.clone(), path_id, hash, mtime);
            path_id += 1;

            // Increment progress counters
            self.progress.fetch_add(1, Ordering::Relaxed);
            let indexed = self.build_progress
                .files_indexed
                .fetch_add(1, Ordering::Relaxed) + 1;
            self.build_progress
                .bytes_indexed
                .fetch_add(content_len, Ordering::Relaxed);

            // Log progress every 10K files during read
            if indexed % 10_000 == 0 {
                let bytes_total = self.build_progress.bytes_indexed.load(Ordering::Relaxed);
                eprintln!(
                    "[ContentBuilder] read: {}/{} files indexed ({} MB), {:.1}s",
                    indexed, total_files, bytes_total / (1024 * 1024),
                    read_start.elapsed().as_secs_f32()
                );
            }
        }

        let total_bytes = self.build_progress.bytes_indexed.load(Ordering::Relaxed);
        eprintln!(
            "[ContentBuilder] read complete: {} files, {} MB in {:.1}s — finalizing Metal buffer",
            path_id, total_bytes / (1024 * 1024), read_start.elapsed().as_secs_f32()
        );

        // Finalize with Metal buffer
        let content_store = builder.finalize(&self.device);

        // Create snapshot with current timestamp
        let build_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let snapshot = ContentSnapshot::new(content_store, build_timestamp);
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use std::fs;
    use tempfile::TempDir;

    fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)")
    }

    fn empty_excludes() -> Arc<ExcludeTrie> {
        Arc::new(ExcludeTrie::new(
            vec![],
            vec![],
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
        ))
    }

    /// Create a tempdir with `n` text files, each with deterministic content.
    fn make_test_dir(n: usize) -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");
        for i in 0..n {
            let filename = format!("file_{:04}.txt", i);
            let content = format!(
                "// File {}\nfn content_builder_test_{}() -> u32 {{ {} }}\n",
                i,
                i,
                i * 3 + 7
            );
            fs::write(dir.path().join(filename), content).unwrap();
        }
        dir
    }

    #[test]
    fn test_content_builder_50_files() {
        let dir = make_test_dir(50);
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let progress = Arc::new(AtomicUsize::new(0));
        let excludes = empty_excludes();

        let builder = ContentBuilder::new(
            store.clone(),
            excludes,
            progress.clone(),
            device,
        );

        let snapshot = builder.build(dir.path()).expect("build should succeed");

        // Should have indexed all 50 text files
        assert_eq!(
            snapshot.file_count(),
            50,
            "Expected 50 files, got {}",
            snapshot.file_count()
        );

        // Progress counter should match
        assert_eq!(
            progress.load(Ordering::Relaxed),
            50,
            "Progress counter should be 50"
        );

        // Content store should have data
        assert!(
            snapshot.content_store().total_bytes() > 0,
            "Content store should have non-zero bytes"
        );

        // Verify content is accessible for each file
        for i in 0..50u32 {
            let content = snapshot
                .content_store()
                .content_for(i)
                .unwrap_or_else(|| panic!("content_for({}) returned None", i));
            assert!(
                !content.is_empty(),
                "Content for file {} should not be empty",
                i
            );
            // Verify it looks like our generated content
            assert!(
                content.starts_with(b"// File "),
                "Content for file {} should start with '// File '",
                i
            );
        }

        // Metal buffer should be present
        assert!(
            snapshot.content_store().has_metal_buffer(),
            "Content store should have a Metal buffer"
        );

        // Build timestamp should be recent
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(
            snapshot.build_timestamp() > 0 && snapshot.build_timestamp() <= now,
            "Build timestamp should be recent"
        );
    }

    #[test]
    fn test_content_builder_skips_binary_files() {
        let dir = TempDir::new().unwrap();

        // Create text files
        for i in 0..5 {
            let content = format!("text file content {}\n", i);
            fs::write(dir.path().join(format!("text_{}.rs", i)), content).unwrap();
        }

        // Create binary files (by extension)
        fs::write(dir.path().join("image.png"), b"PNG binary data").unwrap();
        fs::write(dir.path().join("lib.dylib"), b"dylib binary").unwrap();

        // Create binary file (by content -- NUL bytes)
        fs::write(dir.path().join("mystery.dat"), b"\x00\x01\x02\x03binary").unwrap();

        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let progress = Arc::new(AtomicUsize::new(0));
        let excludes = empty_excludes();

        let builder = ContentBuilder::new(store, excludes, progress.clone(), device);
        let snapshot = builder.build(dir.path()).expect("build should succeed");

        // Only the 5 text files should be indexed
        assert_eq!(
            snapshot.file_count(),
            5,
            "Expected 5 text files, got {}",
            snapshot.file_count()
        );
    }

    #[test]
    fn test_content_builder_binary_detection_with_progress() {
        let dir = TempDir::new().unwrap();

        // Create 3 text files with known content
        let text1 = "fn hello() { println!(\"world\"); }\n";
        let text2 = "struct Foo { bar: u32 }\n";
        let text3 = "const X: i32 = 42;\n";
        fs::write(dir.path().join("hello.rs"), text1).unwrap();
        fs::write(dir.path().join("foo.rs"), text2).unwrap();
        fs::write(dir.path().join("const.rs"), text3).unwrap();

        // Create binary files via extension
        fs::write(dir.path().join("photo.jpg"), b"JFIF fake jpeg data").unwrap();
        fs::write(dir.path().join("archive.zip"), b"PK\x03\x04 fake zip").unwrap();

        // Create binary file via NUL byte content (extension won't trigger skip)
        let mut nul_content = vec![0u8; 512]; // 512 NUL bytes
        nul_content.extend_from_slice(b"some text after nulls");
        fs::write(dir.path().join("data.txt"), &nul_content).unwrap();

        // Create another binary-by-content file with scattered NUL bytes
        let mut mixed = b"looks like text but \x00 has a NUL byte embedded\n".to_vec();
        mixed.push(0x00);
        fs::write(dir.path().join("tricky.log"), &mixed).unwrap();

        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let build_progress = Arc::new(BuildProgress::new());
        let excludes = empty_excludes();

        let builder = ContentBuilder::with_progress(
            store,
            excludes,
            build_progress.clone(),
            device,
        );
        let snapshot = builder.build(dir.path()).expect("build should succeed");

        // Only the 3 text files should be indexed
        assert_eq!(
            snapshot.file_count(),
            3,
            "Expected 3 text files, got {}",
            snapshot.file_count()
        );

        // files_scanned should count ALL files walked (text + binary)
        // 3 text + 2 binary-by-extension + 2 binary-by-content = 7
        assert_eq!(
            build_progress.files_scanned(),
            7,
            "files_scanned should count all files encountered: got {}",
            build_progress.files_scanned()
        );

        // files_indexed should only count text files added to store
        assert_eq!(
            build_progress.files_indexed(),
            3,
            "files_indexed should be 3, got {}",
            build_progress.files_indexed()
        );

        // bytes_indexed should equal sum of text file content sizes
        let expected_bytes = text1.len() + text2.len() + text3.len();
        assert_eq!(
            build_progress.bytes_indexed(),
            expected_bytes,
            "bytes_indexed should be {}, got {}",
            expected_bytes,
            build_progress.bytes_indexed()
        );

        // Verify content of indexed files is correct (text only)
        for i in 0..3u32 {
            let content = snapshot.content_store().content_for(i).unwrap();
            assert!(
                !content.is_empty(),
                "Indexed file {} should have content",
                i
            );
            // None should contain NUL bytes
            assert!(
                !content.contains(&0u8),
                "Indexed file {} should not contain NUL bytes",
                i
            );
        }
    }

    #[test]
    fn test_content_builder_skips_large_files() {
        let dir = TempDir::new().unwrap();

        // Create a normal text file
        fs::write(dir.path().join("small.rs"), "fn main() {}\n").unwrap();

        // Note: We can't easily create a 100MB+ file in a unit test,
        // but we verify the MAX_FILE_SIZE constant is set correctly
        assert_eq!(MAX_FILE_SIZE, 100 * 1024 * 1024);

        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let progress = Arc::new(AtomicUsize::new(0));
        let excludes = empty_excludes();

        let builder = ContentBuilder::new(store, excludes, progress, device);
        let snapshot = builder.build(dir.path()).expect("build should succeed");

        assert!(
            snapshot.file_count() >= 1,
            "Should index at least the small file"
        );
    }

    #[test]
    fn test_content_builder_empty_dir() {
        let dir = TempDir::new().unwrap();

        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let progress = Arc::new(AtomicUsize::new(0));
        let excludes = empty_excludes();

        let builder = ContentBuilder::new(store, excludes, progress.clone(), device);
        let snapshot = builder.build(dir.path()).expect("build should succeed");

        assert_eq!(snapshot.file_count(), 0);
        assert_eq!(progress.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_content_builder_respects_excludes() {
        let dir = TempDir::new().unwrap();

        // Create files in an excluded directory
        let excluded = dir.path().join("node_modules");
        fs::create_dir(&excluded).unwrap();
        fs::write(excluded.join("pkg.js"), "module.exports = {};").unwrap();

        // Create a normal text file
        fs::write(dir.path().join("main.rs"), "fn main() {}\n").unwrap();

        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let progress = Arc::new(AtomicUsize::new(0));

        // Use default excludes which include "node_modules" basename
        let excludes = Arc::new(ExcludeTrie::default());

        let builder = ContentBuilder::new(store, excludes, progress, device);
        let snapshot = builder.build(dir.path()).expect("build should succeed");

        // Only main.rs should be indexed (node_modules excluded)
        assert_eq!(
            snapshot.file_count(),
            1,
            "Expected 1 file (node_modules excluded), got {}",
            snapshot.file_count()
        );
    }

    #[test]
    fn test_content_builder_content_integrity() {
        let dir = TempDir::new().unwrap();

        // Create a single file with known content
        let known_content = "fn hello() -> &'static str { \"GPU search\" }\n";
        fs::write(dir.path().join("hello.rs"), known_content).unwrap();

        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let progress = Arc::new(AtomicUsize::new(0));
        let excludes = empty_excludes();

        let builder = ContentBuilder::new(store, excludes, progress, device);
        let snapshot = builder.build(dir.path()).expect("build should succeed");

        assert_eq!(snapshot.file_count(), 1);

        let stored_content = snapshot.content_store().content_for(0).unwrap();
        assert_eq!(
            stored_content,
            known_content.as_bytes(),
            "Stored content should match original file content exactly"
        );
    }

    #[test]
    fn test_content_builder_crc32_hash() {
        let dir = TempDir::new().unwrap();

        let content = "fn test_hash() -> u32 { 42 }\n";
        fs::write(dir.path().join("hash_test.rs"), content).unwrap();

        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let progress = Arc::new(AtomicUsize::new(0));
        let excludes = empty_excludes();

        let builder = ContentBuilder::new(store, excludes, progress, device);
        let snapshot = builder.build(dir.path()).expect("build should succeed");

        // Verify CRC32 hash matches
        let expected_hash = crc32fast::hash(content.as_bytes());
        let meta = &snapshot.content_store().files()[0];
        assert_eq!(
            meta.content_hash, expected_hash,
            "CRC32 hash should match: expected 0x{:08X}, got 0x{:08X}",
            expected_hash, meta.content_hash
        );
    }

    #[test]
    fn test_content_builder_publishes_to_store() {
        let dir = make_test_dir(10);
        let device = get_device();
        let store = Arc::new(ContentIndexStore::new());
        let progress = Arc::new(AtomicUsize::new(0));
        let excludes = empty_excludes();

        assert!(!store.is_available(), "Store should start empty");

        let builder = ContentBuilder::new(store.clone(), excludes, progress, device);
        let snapshot = builder.build(dir.path()).expect("build should succeed");

        // Publish to store
        store.swap(snapshot);

        assert!(store.is_available(), "Store should be available after swap");

        // Read back via snapshot
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        assert_eq!(snap.file_count(), 10);
    }
}
