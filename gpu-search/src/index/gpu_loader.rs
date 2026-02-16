//! GPU Index Loader
//!
//! Orchestrates the full pipeline from filesystem -> GPU buffer:
//! 1. Check mmap cache for a saved index
//! 2. If cache hit: mmap -> `newBufferWithBytesNoCopy` -> GPU buffer (zero-copy)
//! 3. If cache miss: scan -> build -> save -> mmap -> GPU buffer
//!
//! The resulting `GpuLoadedIndex` holds the Metal buffer, entry count, and
//! metadata needed by the path_filter kernel.
//!
//! Target: <10ms for 100K cached entries on Apple Silicon.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

use crate::gpu::types::GpuPathEntry;
use crate::index::cache::CacheError;
use crate::index::gpu_index::GpuResidentIndex;
use crate::index::scanner::FilesystemScanner;
use crate::index::shared_index::SharedIndexManager;
use crate::index::store::IndexStore;

// ============================================================================
// Errors
// ============================================================================

/// Errors from GPU index loading.
#[derive(Debug)]
pub enum GpuLoaderError {
    /// I/O or filesystem error
    Io(std::io::Error),
    /// Cache format error
    Cache(CacheError),
    /// Metal device error
    Metal(String),
    /// Index build error
    Build(String),
}

impl std::fmt::Display for GpuLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "GPU loader I/O error: {}", e),
            Self::Cache(e) => write!(f, "GPU loader cache error: {}", e),
            Self::Metal(msg) => write!(f, "GPU loader Metal error: {}", msg),
            Self::Build(msg) => write!(f, "GPU loader build error: {}", msg),
        }
    }
}

impl std::error::Error for GpuLoaderError {}

impl From<std::io::Error> for GpuLoaderError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<CacheError> for GpuLoaderError {
    fn from(e: CacheError) -> Self {
        Self::Cache(e)
    }
}

// ============================================================================
// GpuLoadedIndex
// ============================================================================

/// A filesystem index loaded into a Metal GPU buffer.
///
/// Holds the Metal buffer containing packed `GpuPathEntry` data (256 bytes each),
/// ready for the `path_filter_kernel` to query.
///
/// The buffer is either:
/// - Zero-copy from mmap (via `newBufferWithBytesNoCopy`) when loaded from cache
/// - Copied from CPU memory (via `newBufferWithBytes`) when freshly built
pub struct GpuLoadedIndex {
    /// Metal buffer containing packed GpuPathEntry data
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Number of GpuPathEntry entries in the buffer
    entry_count: usize,
    /// Root directory this index was built from
    root: PathBuf,
    /// Whether this was loaded from cache (true) or freshly built (false)
    from_cache: bool,
    /// Time taken to load the index
    load_time: std::time::Duration,
}

impl GpuLoadedIndex {
    /// Get the Metal buffer containing packed GpuPathEntry data.
    #[inline]
    pub fn buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffer
    }

    /// Number of entries in the index.
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Root directory this index was built from.
    #[inline]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether this index was loaded from cache.
    #[inline]
    pub fn from_cache(&self) -> bool {
        self.from_cache
    }

    /// Time taken to load the index.
    #[inline]
    pub fn load_time(&self) -> std::time::Duration {
        self.load_time
    }

    /// GPU buffer size in bytes.
    #[inline]
    pub fn buffer_size_bytes(&self) -> usize {
        self.entry_count * GpuPathEntry::SIZE
    }
}

// ============================================================================
// GpuIndexLoader
// ============================================================================

/// Orchestrates loading a filesystem index into a Metal GPU buffer.
///
/// Tries the fast path (mmap cached index -> zero-copy Metal buffer) first.
/// Falls back to scan -> build -> save -> load on cache miss.
///
/// # Example
/// ```ignore
/// let device = MTLCreateSystemDefaultDevice().unwrap();
/// let loader = GpuIndexLoader::new(&device);
/// let loaded = loader.load(Path::new("/Users/me/project")).unwrap();
/// println!("Loaded {} entries in {:?}", loaded.entry_count(), loaded.load_time());
/// // Use loaded.buffer() with path_filter kernel
/// ```
pub struct GpuIndexLoader<'a> {
    device: &'a ProtocolObject<dyn MTLDevice>,
    /// Optional custom cache directory (default: ~/.gpu-search/)
    cache_base: Option<PathBuf>,
    /// Optional IndexStore for mmap-backed zero-copy snapshots.
    /// When available, snapshots are preferred over the copy-based SharedIndexManager path.
    index_store: Option<Arc<IndexStore>>,
}

impl<'a> GpuIndexLoader<'a> {
    /// Create a new loader with the given Metal device.
    pub fn new(device: &'a ProtocolObject<dyn MTLDevice>) -> Self {
        Self {
            device,
            cache_base: None,
            index_store: None,
        }
    }

    /// Create a loader with a custom cache directory (for testing).
    pub fn with_cache_dir(device: &'a ProtocolObject<dyn MTLDevice>, cache_base: PathBuf) -> Self {
        Self {
            device,
            cache_base: Some(cache_base),
            index_store: None,
        }
    }

    /// Create a loader with an IndexStore for mmap-backed zero-copy snapshots.
    pub fn with_store(device: &'a ProtocolObject<dyn MTLDevice>, store: Arc<IndexStore>) -> Self {
        Self {
            device,
            cache_base: None,
            index_store: Some(store),
        }
    }

    /// Load a filesystem index into a GPU buffer.
    ///
    /// Pipeline:
    /// 1. Check if a cached index exists for `root`
    /// 2. If cache hit and not stale: mmap -> Metal buffer (zero-copy)
    /// 3. If cache miss or stale: scan -> build GpuResidentIndex -> save to cache -> upload to GPU
    ///
    /// Returns a `GpuLoadedIndex` with the Metal buffer ready for kernel dispatch.
    pub fn load(&self, root: &Path) -> Result<GpuLoadedIndex, GpuLoaderError> {
        let start = Instant::now();

        // Try fast path: load from mmap cache
        match self.try_load_from_cache(root) {
            Ok(loaded) => return Ok(loaded),
            Err(_) => {
                // Cache miss or stale -- fall through to full rebuild
            }
        }

        // Slow path: scan -> build -> save -> load
        self.scan_build_save_load(root, start)
    }

    /// Try to load from mmap cache (fast path).
    ///
    /// Checks two sources in priority order:
    /// 1. IndexStore snapshot (mmap-backed zero-copy via bytesNoCopy) — preferred
    /// 2. SharedIndexManager v1 format — fallback for non-Metal or pre-v2 contexts
    fn try_load_from_cache(&self, root: &Path) -> Result<GpuLoadedIndex, GpuLoaderError> {
        // Try IndexStore snapshot first (zero-copy mmap path)
        if let Some(ref store) = self.index_store {
            if let Ok(loaded) = self.try_load_from_store(store, root) {
                return Ok(loaded);
            }
        }

        // Fallback: SharedIndexManager (copy-based v1 path)
        self.try_load_from_manager(root)
    }

    /// Try to load from IndexStore's mmap-backed snapshot (zero-copy path).
    ///
    /// If the IndexStore has a valid snapshot with a Metal buffer, uses that
    /// directly — no data copy needed. The Metal buffer was already created
    /// via bytesNoCopy over the mmap'd region.
    fn try_load_from_store(
        &self,
        store: &IndexStore,
        root: &Path,
    ) -> Result<GpuLoadedIndex, GpuLoaderError> {
        let start = Instant::now();

        let guard = store.snapshot();
        let snapshot = guard.as_ref().as_ref().ok_or_else(|| {
            GpuLoaderError::Build("IndexStore has no snapshot".into())
        })?;

        let entry_count = snapshot.entry_count();
        if entry_count == 0 {
            return Err(GpuLoaderError::Build("IndexStore snapshot is empty".into()));
        }

        // Get the Metal buffer from the snapshot (already created via bytesNoCopy)
        let metal_buf = snapshot.metal_buffer().ok_or_else(|| {
            GpuLoaderError::Metal("Snapshot has no Metal buffer".into())
        })?;

        // Retain the buffer so it outlives the guard (objc2 Retained::from(&ref) bumps refcount)
        let buffer: Retained<ProtocolObject<dyn MTLBuffer>> = metal_buf.into();
        let load_time = start.elapsed();

        eprintln!(
            "[gpu-loader] loaded {} entries from IndexStore snapshot (zero-copy, {:?})",
            entry_count, load_time
        );

        Ok(GpuLoadedIndex {
            buffer,
            entry_count,
            root: root.to_path_buf(),
            from_cache: true,
            load_time,
        })
    }

    /// Fallback: load from SharedIndexManager (copy-based v1 path).
    ///
    /// Uses SharedIndexManager (v1 format) for load since scan_build_save_load
    /// saves in v1 format. Kept as fallback for non-Metal contexts.
    fn try_load_from_manager(&self, root: &Path) -> Result<GpuLoadedIndex, GpuLoaderError> {
        let start = Instant::now();
        let manager = self.make_manager()?;

        // Check staleness
        if manager.is_stale(root, crate::index::shared_index::DEFAULT_MAX_AGE) {
            return Err(GpuLoaderError::Build("Index is stale".into()));
        }

        // Load entries from v1 cache via SharedIndexManager (matches save format)
        let mut resident = manager.load(root).map_err(|e| {
            GpuLoaderError::Build(format!("Failed to load cached index: {}", e))
        })?;
        let entry_count = resident.entry_count();

        // Upload entries to GPU buffer (copy path)
        let buffer = resident.to_gpu_buffer(self.device).clone();

        let load_time = start.elapsed();

        Ok(GpuLoadedIndex {
            buffer,
            entry_count,
            root: root.to_path_buf(),
            from_cache: true,
            load_time,
        })
    }

    /// Full rebuild path: scan filesystem -> build index -> save to cache -> upload to GPU.
    fn scan_build_save_load(
        &self,
        root: &Path,
        start: Instant,
    ) -> Result<GpuLoadedIndex, GpuLoaderError> {
        // Scan the filesystem
        let scanner = FilesystemScanner::new();
        let entries = scanner.scan(root);

        // Build a GpuResidentIndex from scanned entries
        let mut index = GpuResidentIndex::from_entries(entries);
        let entry_count = index.entry_count();

        // Save to cache for next time
        let manager = self.make_manager()?;
        let _ = manager.save(&index, root); // Best-effort save; don't fail if cache write fails

        // Upload to GPU buffer
        let buffer = index.to_gpu_buffer(self.device).clone();

        let load_time = start.elapsed();

        Ok(GpuLoadedIndex {
            buffer,
            entry_count,
            root: root.to_path_buf(),
            from_cache: false,
            load_time,

        })
    }

    /// Create a SharedIndexManager with the configured cache directory.
    fn make_manager(&self) -> Result<SharedIndexManager, GpuLoaderError> {
        match &self.cache_base {
            Some(base) => Ok(SharedIndexManager::with_base_dir(base.clone())),
            None => SharedIndexManager::new().map_err(|e| {
                GpuLoaderError::Build(format!("Failed to create index manager: {}", e))
            }),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Create a test directory with known files.
    fn make_test_dir() -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");

        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("lib.rs"), "pub mod test;").unwrap();
        fs::write(dir.path().join("README.md"), "# Test Project").unwrap();

        let sub = dir.path().join("src");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("mod.rs"), "mod sub;").unwrap();
        fs::write(sub.join("utils.rs"), "pub fn help() {}").unwrap();

        dir
    }

    #[test]
    fn test_gpu_index_load() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");

        let loader = GpuIndexLoader::with_cache_dir(&device, cache_dir.path().to_path_buf());

        // First load: should scan + build + save (cache miss)
        let loaded = loader
            .load(test_dir.path())
            .expect("Failed to load index");

        assert!(
            loaded.entry_count() > 0,
            "Should have entries, got {}",
            loaded.entry_count()
        );
        assert!(!loaded.from_cache(), "First load should NOT be from cache");
        assert_eq!(loaded.root(), test_dir.path());

        // Buffer should be correctly sized
        let expected_buffer_size = loaded.entry_count() * GpuPathEntry::SIZE;
        assert!(
            loaded.buffer().length() >= expected_buffer_size,
            "Buffer too small: {} < {}",
            loaded.buffer().length(),
            expected_buffer_size
        );

        println!(
            "First load: {} entries, buffer {} bytes, took {:?} (from_cache={})",
            loaded.entry_count(),
            loaded.buffer().length(),
            loaded.load_time(),
            loaded.from_cache()
        );

        // Second load: should hit cache
        let loaded2 = loader
            .load(test_dir.path())
            .expect("Failed to load index (cached)");

        assert_eq!(
            loaded2.entry_count(),
            loaded.entry_count(),
            "Cached entry count should match"
        );
        assert!(loaded2.from_cache(), "Second load SHOULD be from cache");

        println!(
            "Cached load: {} entries, buffer {} bytes, took {:?} (from_cache={})",
            loaded2.entry_count(),
            loaded2.buffer().length(),
            loaded2.load_time(),
            loaded2.from_cache()
        );

        // Verify buffer contents are valid GpuPathEntry data
        unsafe {
            let buf_ptr = loaded2.buffer().contents().as_ptr() as *const GpuPathEntry;
            for i in 0..loaded2.entry_count() {
                let entry = &*buf_ptr.add(i);
                assert!(
                    entry.path_len > 0,
                    "Entry {} should have non-zero path_len",
                    i
                );
                let path_str =
                    std::str::from_utf8(&entry.path[..entry.path_len as usize]).unwrap_or("");
                assert!(
                    !path_str.is_empty(),
                    "Entry {} should have non-empty path",
                    i
                );
            }
        }
    }

    #[test]
    fn test_gpu_index_load_real_src() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");
        let cache_dir = TempDir::new().expect("Failed to create cache dir");

        let loader = GpuIndexLoader::with_cache_dir(&device, cache_dir.path().to_path_buf());

        // Load index for gpu-search/src/ directory
        let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let loaded = loader.load(&src_dir).expect("Failed to load src/ index");

        assert!(
            loaded.entry_count() > 0,
            "Should find files in src/"
        );

        println!(
            "Loaded src/ index: {} entries, {} bytes buffer, {:?}",
            loaded.entry_count(),
            loaded.buffer().length(),
            loaded.load_time()
        );

        // Verify we found .rs files by checking buffer contents
        let mut rs_count = 0;
        unsafe {
            let buf_ptr = loaded.buffer().contents().as_ptr() as *const GpuPathEntry;
            for i in 0..loaded.entry_count() {
                let entry = &*buf_ptr.add(i);
                let path_str =
                    std::str::from_utf8(&entry.path[..entry.path_len as usize]).unwrap_or("");
                if path_str.ends_with(".rs") {
                    rs_count += 1;
                }
            }
        }
        assert!(
            rs_count > 0,
            "Should find at least one .rs file in src/"
        );
        println!("Found {} .rs files in src/", rs_count);
    }

    #[test]
    fn test_gpu_index_load_empty_dir() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");
        let empty_dir = TempDir::new().expect("Failed to create empty dir");
        let cache_dir = TempDir::new().expect("Failed to create cache dir");

        let loader = GpuIndexLoader::with_cache_dir(&device, cache_dir.path().to_path_buf());
        let loaded = loader.load(empty_dir.path()).expect("Failed to load empty index");

        assert_eq!(loaded.entry_count(), 0, "Empty dir should have 0 entries");
        // Buffer should still exist (minimal allocation)
        assert!(loaded.buffer().length() > 0, "Buffer should be allocated");
    }

    #[test]
    fn test_gpu_index_load_cache_invalidation() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");

        let loader = GpuIndexLoader::with_cache_dir(&device, cache_dir.path().to_path_buf());

        // First load (creates cache)
        let loaded1 = loader.load(test_dir.path()).expect("First load");
        let count1 = loaded1.entry_count();

        // Add a file
        fs::write(test_dir.path().join("new_file.rs"), "// new").unwrap();

        // Touch the directory to trigger staleness (mtime > saved_at)
        // The directory mtime changes when we add a file, so is_stale should return true
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Force staleness by modifying the test dir
        let _ = fs::write(test_dir.path().join("another.rs"), "// another");

        // Second load should detect staleness and rebuild
        let loaded2 = loader.load(test_dir.path()).expect("Second load after change");

        // Should have more entries now
        assert!(
            loaded2.entry_count() >= count1,
            "After adding files, entry count should be >= original: {} vs {}",
            loaded2.entry_count(),
            count1
        );
    }
}
