//! Binary Index Cache with mmap-based Loading
//!
//! Provides fast index persistence and loading using memory-mapped I/O.
//! Wraps the binary format from `shared_index.rs` (64B header + packed GpuPathEntry array)
//! but loads via mmap instead of `fs::read()` for zero-copy access.
//!
//! # Binary Format
//!
//! ```text
//! [IndexHeader: 64 bytes]   - magic "GSIX", version, entry_count, root_hash, timestamp
//! [GpuPathEntry[0]: 256 B]
//! [GpuPathEntry[1]: 256 B]
//! ...
//! [GpuPathEntry[N-1]: 256 B]
//! ```
//!
//! # Loading Path
//!
//! Uses `MmapBuffer::from_file()` to memory-map the index file, then validates
//! the header and reinterprets the entry region as `&[GpuPathEntry]` with zero copies.
//! This is significantly faster than `SharedIndexManager::load()` which reads the
//! entire file into a Vec and copies each entry individually.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::gpu::types::GpuPathEntry;
use crate::index::gpu_index::GpuResidentIndex;
use crate::io::mmap::MmapBuffer;

// ============================================================================
// Constants (must match shared_index.rs)
// ============================================================================

/// Magic bytes: "GSIX" (Gpu Search IndeX) in little-endian
const INDEX_MAGIC: u32 = 0x58495347;

/// Current binary format version
const INDEX_VERSION: u32 = 1;

/// Header size in bytes
const HEADER_SIZE: usize = 64;

// ============================================================================
// Errors
// ============================================================================

/// Errors from index cache operations.
#[derive(Debug)]
pub enum CacheError {
    /// I/O error (file not found, permission denied, etc.)
    Io(std::io::Error),
    /// Invalid or corrupt index file
    InvalidFormat(String),
    /// Index file not found at expected path
    NotFound(PathBuf),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Cache I/O error: {}", e),
            Self::InvalidFormat(msg) => write!(f, "Invalid cache format: {}", msg),
            Self::NotFound(path) => write!(f, "Cache not found: {}", path.display()),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<std::io::Error> for CacheError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================================
// MmapIndexCache
// ============================================================================

/// A memory-mapped index file providing zero-copy access to cached entries.
///
/// After `load_mmap()`, the index file stays mapped and entries are accessed
/// directly from the mmap'd region without any copying.
pub struct MmapIndexCache {
    /// The underlying mmap buffer (keeps the mapping alive)
    _mmap: MmapBuffer,
    /// Pointer to the first GpuPathEntry in the mapped region
    entries_ptr: *const GpuPathEntry,
    /// Number of entries
    entry_count: usize,
}

// SAFETY: The mmap is read-only and entries_ptr points into the mmap region.
// Multiple threads can safely read from the mapping.
unsafe impl Send for MmapIndexCache {}
unsafe impl Sync for MmapIndexCache {}

impl MmapIndexCache {
    /// Load an index file via mmap for zero-copy access.
    ///
    /// The returned `MmapIndexCache` holds the mapping open. Entries are accessed
    /// directly from the mmap'd region via `entries()`.
    ///
    /// # Arguments
    /// * `path` - Path to the `.idx` file
    /// * `expected_root_hash` - If Some, verify the root hash in the header matches
    ///
    /// # Errors
    /// Returns `CacheError` if the file doesn't exist, is corrupt, or has wrong version.
    pub fn load_mmap(
        path: impl AsRef<Path>,
        expected_root_hash: Option<u32>,
    ) -> Result<Self, CacheError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(CacheError::NotFound(path.to_path_buf()));
        }

        // Memory-map the index file
        let mmap = MmapBuffer::from_file(path).map_err(|e| {
            CacheError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to mmap index file {}: {}", path.display(), e),
            ))
        })?;

        let data = mmap.as_slice();

        // Validate minimum size (header)
        if data.len() < HEADER_SIZE {
            return Err(CacheError::InvalidFormat(format!(
                "File too small for header: {} < {}",
                data.len(),
                HEADER_SIZE
            )));
        }

        // Parse and validate header fields
        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if magic != INDEX_MAGIC {
            return Err(CacheError::InvalidFormat(format!(
                "Bad magic: 0x{:08X} (expected 0x{:08X})",
                magic, INDEX_MAGIC
            )));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != INDEX_VERSION {
            return Err(CacheError::InvalidFormat(format!(
                "Unsupported version: {} (expected {})",
                version, INDEX_VERSION
            )));
        }

        let entry_count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let root_hash = u32::from_le_bytes(data[12..16].try_into().unwrap());

        // Verify root hash if requested
        if let Some(expected) = expected_root_hash {
            if root_hash != expected {
                return Err(CacheError::InvalidFormat(format!(
                    "Root hash mismatch: file has 0x{:08X}, expected 0x{:08X}",
                    root_hash, expected
                )));
            }
        }

        // Verify data length covers all entries
        let expected_len = HEADER_SIZE + entry_count * GpuPathEntry::SIZE;
        if data.len() < expected_len {
            return Err(CacheError::InvalidFormat(format!(
                "File too short: {} < {} ({} entries expected)",
                data.len(),
                expected_len,
                entry_count
            )));
        }

        // Get pointer to the entries region
        // SAFETY: We verified data.len() >= HEADER_SIZE + entry_count * 256.
        // GpuPathEntry is #[repr(C)], 256 bytes, alignment 4.
        // The mmap region is page-aligned (16KB) so the entries start at offset 64
        // which is aligned to 4 bytes.
        let entries_ptr = unsafe {
            data.as_ptr().add(HEADER_SIZE) as *const GpuPathEntry
        };

        // Advise kernel we'll read sequentially
        mmap.advise_sequential();

        Ok(Self {
            _mmap: mmap,
            entries_ptr,
            entry_count,
        })
    }

    /// Get the number of entries in the cached index.
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Get a zero-copy slice of all entries from the mmap'd region.
    ///
    /// # Safety Note
    /// The returned slice borrows from `self`. The mmap region remains valid
    /// as long as `MmapIndexCache` is alive.
    pub fn entries(&self) -> &[GpuPathEntry] {
        if self.entry_count == 0 {
            return &[];
        }
        // SAFETY: entries_ptr was validated in load_mmap() to point to
        // entry_count * 256 bytes within the mmap region. The mmap is
        // read-only (PROT_READ) and stays alive via _mmap field.
        unsafe {
            std::slice::from_raw_parts(self.entries_ptr, self.entry_count)
        }
    }

    /// Get a single entry by index.
    pub fn get_entry(&self, idx: usize) -> Option<&GpuPathEntry> {
        if idx < self.entry_count {
            // SAFETY: idx < entry_count, validated in load_mmap()
            Some(unsafe { &*self.entries_ptr.add(idx) })
        } else {
            None
        }
    }

    /// Convert the mmap'd cache into a `GpuResidentIndex` by copying entries.
    ///
    /// This copies the data out of the mmap region into a Vec. Use this when
    /// you need a mutable index or one that outlives the cache file.
    pub fn into_resident_index(self) -> GpuResidentIndex {
        let entries = self.entries().to_vec();
        GpuResidentIndex::from_entries(entries)
    }
}

/// Compute a cache key (hash) for a root path.
///
/// Uses the same algorithm as `SharedIndexManager::cache_key()`.
pub fn cache_key(root: &Path) -> u32 {
    let canonical = root
        .canonicalize()
        .unwrap_or_else(|_| root.to_path_buf());
    let mut hasher = DefaultHasher::new();
    canonical.to_string_lossy().as_ref().hash(&mut hasher);
    hasher.finish() as u32
}

/// Get the default index directory (`~/.gpu-search/index/`).
pub fn default_index_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".gpu-search").join("index"))
}

/// Get the index file path for a given root directory within the default cache location.
pub fn index_path_for(root: &Path) -> Option<PathBuf> {
    default_index_dir().map(|dir| {
        let hash = cache_key(root);
        dir.join(format!("{:08x}.idx", hash))
    })
}

/// Load a cached index via mmap for the given root directory.
///
/// Convenience function that computes the cache path and validates the root hash.
///
/// # Example
/// ```ignore
/// let cache = load_cached_index(Path::new("/Users/me/project"))?;
/// println!("Cached {} entries", cache.entry_count());
/// for entry in cache.entries() {
///     // zero-copy access to each GpuPathEntry
/// }
/// ```
pub fn load_cached_index(root: &Path) -> Result<MmapIndexCache, CacheError> {
    let idx_path = index_path_for(root).ok_or_else(|| {
        CacheError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Could not determine home directory for cache path",
        ))
    })?;

    let root_hash = cache_key(root);
    MmapIndexCache::load_mmap(&idx_path, Some(root_hash))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::shared_index::SharedIndexManager;
    use std::fs;
    use tempfile::TempDir;

    /// Create a temp directory with known files for testing.
    fn make_test_dir() -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");

        fs::write(dir.path().join("hello.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("world.txt"), "hello world").unwrap();
        fs::write(dir.path().join("README.md"), "# Test Project").unwrap();

        let sub = dir.path().join("src");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("lib.rs"), "pub mod test;").unwrap();
        fs::write(sub.join("main.rs"), "fn main() { println!(\"hi\"); }").unwrap();

        dir
    }

    #[test]
    fn test_index_cache_save_and_mmap_load() {
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Build and save an index using SharedIndexManager
        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");
        let original_count = index.entry_count();
        assert!(original_count > 0, "Index should have entries");

        let saved_path = manager
            .save(&index, test_dir.path())
            .expect("Failed to save index");
        assert!(saved_path.exists());

        // Load via mmap cache
        let root_hash = cache_key(test_dir.path());
        let cache = MmapIndexCache::load_mmap(&saved_path, Some(root_hash))
            .expect("Failed to mmap load index");

        // Verify entry count matches
        assert_eq!(
            cache.entry_count(),
            original_count,
            "Mmap entry count should match original"
        );

        // Verify each entry is byte-identical
        let original_entries = index.entries();
        let cached_entries = cache.entries();

        for i in 0..original_count {
            let orig = &original_entries[i];
            let cached = &cached_entries[i];

            // Compare all 256 bytes
            let orig_bytes: &[u8; 256] =
                unsafe { &*(orig as *const GpuPathEntry as *const [u8; 256]) };
            let cached_bytes: &[u8; 256] =
                unsafe { &*(cached as *const GpuPathEntry as *const [u8; 256]) };

            assert_eq!(
                orig_bytes, cached_bytes,
                "Entry {} byte-level mismatch",
                i
            );
        }
    }

    #[test]
    fn test_index_cache_into_resident_index() {
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Build and save
        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");
        let original_count = index.entry_count();

        let saved_path = manager
            .save(&index, test_dir.path())
            .expect("Failed to save");

        // Load via mmap and convert to resident index
        let cache = MmapIndexCache::load_mmap(&saved_path, None)
            .expect("Failed to load");
        let loaded_index = cache.into_resident_index();

        assert_eq!(loaded_index.entry_count(), original_count);

        // Verify entries match
        for i in 0..original_count {
            let orig = index.get_entry(i).unwrap();
            let loaded = loaded_index.get_entry(i).unwrap();
            assert_eq!(
                orig.path_len, loaded.path_len,
                "Entry {} path_len mismatch",
                i
            );
            assert_eq!(
                &orig.path[..orig.path_len as usize],
                &loaded.path[..loaded.path_len as usize],
                "Entry {} path mismatch",
                i
            );
        }
    }

    #[test]
    fn test_index_cache_not_found() {
        let result = MmapIndexCache::load_mmap("/nonexistent/path.idx", None);
        assert!(result.is_err());
        match result {
            Err(CacheError::NotFound(_)) => {} // expected
            other => panic!("Expected NotFound, got: {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn test_index_cache_corrupt_file() {
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let corrupt_path = cache_dir.path().join("corrupt.idx");

        // Write garbage data (big enough to mmap)
        let garbage = vec![0xDE; 1024];
        fs::write(&corrupt_path, &garbage).unwrap();

        let result = MmapIndexCache::load_mmap(&corrupt_path, None);
        assert!(result.is_err());
        match result {
            Err(CacheError::InvalidFormat(_)) => {} // expected
            other => panic!("Expected InvalidFormat, got: {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn test_index_cache_wrong_root_hash() {
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Save with real root hash
        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");
        let saved_path = manager
            .save(&index, test_dir.path())
            .expect("Failed to save");

        // Try to load with wrong root hash
        let result = MmapIndexCache::load_mmap(&saved_path, Some(0xDEADBEEF));
        assert!(result.is_err());
        match result {
            Err(CacheError::InvalidFormat(msg)) => {
                assert!(msg.contains("hash mismatch"), "Error: {}", msg);
            }
            other => panic!("Expected InvalidFormat with hash mismatch, got: {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn test_index_cache_get_entry() {
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");
        let saved_path = manager
            .save(&index, test_dir.path())
            .expect("Failed to save");

        let cache = MmapIndexCache::load_mmap(&saved_path, None)
            .expect("Failed to load");

        // Valid index
        let entry = cache.get_entry(0);
        assert!(entry.is_some());

        // Out of bounds
        let oob = cache.get_entry(cache.entry_count());
        assert!(oob.is_none());
    }

    #[test]
    fn test_index_cache_roundtrip_fidelity() {
        // Full roundtrip: build index -> save -> mmap load -> verify byte-identical -> convert back
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Build from known paths
        let paths: Vec<PathBuf> = vec![
            test_dir.path().join("hello.rs"),
            test_dir.path().join("world.txt"),
            test_dir.path().join("README.md"),
            test_dir.path().join("src"),
            test_dir.path().join("src").join("lib.rs"),
            test_dir.path().join("src").join("main.rs"),
        ];
        let index = GpuResidentIndex::build_from_paths(&paths);
        let original_count = index.entry_count();

        // Save
        let saved_path = manager
            .save(&index, test_dir.path())
            .expect("Failed to save");

        // Mmap load
        let cache = MmapIndexCache::load_mmap(&saved_path, None)
            .expect("Failed to load");
        assert_eq!(cache.entry_count(), original_count);

        // Verify ALL 256 bytes of each entry
        for i in 0..original_count {
            let orig = index.get_entry(i).unwrap();
            let cached = cache.get_entry(i).unwrap();

            let orig_bytes: &[u8; 256] =
                unsafe { &*(orig as *const GpuPathEntry as *const [u8; 256]) };
            let cached_bytes: &[u8; 256] =
                unsafe { &*(cached as *const GpuPathEntry as *const [u8; 256]) };

            assert_eq!(
                orig_bytes, cached_bytes,
                "Entry {} full 256-byte mismatch",
                i
            );
        }

        // Convert back to GpuResidentIndex and verify
        let cache2 = MmapIndexCache::load_mmap(&saved_path, None)
            .expect("Failed to reload");
        let reloaded = cache2.into_resident_index();
        assert_eq!(reloaded.entry_count(), original_count);

        for i in 0..original_count {
            let orig = index.get_entry(i).unwrap();
            let re = reloaded.get_entry(i).unwrap();

            assert_eq!(orig.path_len, re.path_len, "Entry {} path_len", i);
            assert_eq!(
                &orig.path[..orig.path_len as usize],
                &re.path[..re.path_len as usize],
                "Entry {} path",
                i
            );
            assert_eq!(orig.flags, re.flags, "Entry {} flags", i);
            assert_eq!(orig.size_lo, re.size_lo, "Entry {} size_lo", i);
            assert_eq!(orig.size_hi, re.size_hi, "Entry {} size_hi", i);
            assert_eq!(orig.mtime, re.mtime, "Entry {} mtime", i);
        }
    }

    #[test]
    fn test_index_cache_cache_key_consistency() {
        let test_dir = make_test_dir();

        // Our cache_key should match SharedIndexManager::cache_key
        let our_key = cache_key(test_dir.path());
        let shared_key = SharedIndexManager::cache_key(test_dir.path());
        assert_eq!(
            our_key, shared_key,
            "cache_key functions must produce identical results"
        );
    }

    /// Integration test matching the verify command name: test_index_cache
    #[test]
    fn test_index_cache() {
        // Build index, save, load via mmap, verify identical entries
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Build
        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");
        let original_count = index.entry_count();
        assert!(original_count >= 5, "Should have at least 5 entries (3 files + src dir + 2 src files)");

        // Save to disk
        let saved_path = manager
            .save(&index, test_dir.path())
            .expect("Failed to save");
        assert!(saved_path.exists(), "Index file should exist on disk");

        // "Restart" -- load from disk via mmap (simulates process restart)
        let root_hash = cache_key(test_dir.path());
        let cache = MmapIndexCache::load_mmap(&saved_path, Some(root_hash))
            .expect("Failed to load cached index via mmap");

        // Verify identical entry count
        assert_eq!(
            cache.entry_count(),
            original_count,
            "Cached entry count must match original"
        );

        // Verify entries are identical
        for i in 0..original_count {
            let orig = index.get_entry(i).unwrap();
            let cached = cache.get_entry(i).unwrap();

            assert_eq!(
                orig.path_len, cached.path_len,
                "Entry {} path_len mismatch",
                i
            );
            assert_eq!(
                &orig.path[..orig.path_len as usize],
                &cached.path[..cached.path_len as usize],
                "Entry {} path content mismatch",
                i
            );
            assert_eq!(orig.flags, cached.flags, "Entry {} flags mismatch", i);
            assert_eq!(orig.size_lo, cached.size_lo, "Entry {} size_lo mismatch", i);
            assert_eq!(orig.size_hi, cached.size_hi, "Entry {} size_hi mismatch", i);
            assert_eq!(orig.mtime, cached.mtime, "Entry {} mtime mismatch", i);
        }

        // Also verify we can convert back to a GpuResidentIndex
        let cache2 = MmapIndexCache::load_mmap(&saved_path, Some(root_hash))
            .expect("Failed to reload");
        let resident = cache2.into_resident_index();
        assert_eq!(resident.entry_count(), original_count);
    }
}
