//! Binary Index Cache with mmap-based Loading
//!
//! Provides fast index persistence and loading using memory-mapped I/O.
//! Loads GSIX v2 files (16KB page-aligned header + packed GpuPathEntry array)
//! via mmap for zero-copy access suitable for Metal bytesNoCopy GPU dispatch.
//!
//! # Binary Format (v2)
//!
//! ```text
//! [GsixHeaderV2: 16384 bytes]  - magic "GSIX", version=2, entry_count, root_hash, CRC32, etc.
//! [GpuPathEntry[0]: 256 B]
//! [GpuPathEntry[1]: 256 B]
//! ...
//! [GpuPathEntry[N-1]: 256 B]
//! ```
//!
//! # Version Handling
//!
//! - v2 files are loaded normally via mmap with zero-copy entry access
//! - v1 files (64B header) are detected and rejected with `CacheError::InvalidFormat`,
//!   signaling the caller to rebuild the index in v2 format
//!
//! # Loading Path
//!
//! Uses `MmapBuffer::from_file()` to memory-map the index file, then validates
//! the v2 header (including CRC32 checksum) and reinterprets the entry region
//! as `&[GpuPathEntry]` with zero copies. The 16KB header ensures the entry
//! region starts at a page boundary on Apple Silicon (16KB pages).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::gpu::types::GpuPathEntry;
use crate::index::gpu_index::GpuResidentIndex;
use crate::index::gsix_v2::{self, GsixHeaderV2, HEADER_SIZE_V2};
use crate::io::mmap::MmapBuffer;

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

        // Detect format version (needs at least 8 bytes for magic + version)
        let version = gsix_v2::detect_version(data)?;

        // Reject v1 files — caller should rebuild in v2 format
        if version == gsix_v2::INDEX_VERSION_V1 {
            return Err(CacheError::InvalidFormat(
                "v1 index detected, rebuild required".to_string(),
            ));
        }

        // v2: validate 16KB header (magic, version, CRC32 checksum)
        let header = GsixHeaderV2::from_bytes(data)?;

        let entry_count = header.entry_count as usize;
        let root_hash = header.root_hash;

        // Verify root hash if requested
        if let Some(expected) = expected_root_hash {
            if root_hash != expected {
                return Err(CacheError::InvalidFormat(format!(
                    "Root hash mismatch: file has 0x{:08X}, expected 0x{:08X}",
                    root_hash, expected
                )));
            }
        }

        // Verify data length covers all entries (entries start at offset 16384)
        let expected_len = HEADER_SIZE_V2 + entry_count * GpuPathEntry::SIZE;
        if data.len() < expected_len {
            return Err(CacheError::InvalidFormat(format!(
                "File too short: {} < {} ({} entries expected)",
                data.len(),
                expected_len,
                entry_count
            )));
        }

        // Get pointer to the entries region (offset 16384 = page-aligned)
        // SAFETY: We verified data.len() >= HEADER_SIZE_V2 + entry_count * 256.
        // GpuPathEntry is #[repr(C)], 256 bytes, alignment 4.
        // The mmap region is page-aligned and HEADER_SIZE_V2 (16384) is a full
        // page on Apple Silicon, so entries start at a page boundary.
        let entries_ptr = unsafe {
            data.as_ptr().add(HEADER_SIZE_V2) as *const GpuPathEntry
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
    use crate::index::gsix_v2;
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

    /// Helper: build entries from test dir and save as v2 format.
    /// Returns (GpuResidentIndex, saved_path).
    fn save_v2_from_dir(test_dir: &Path, cache_dir: &Path) -> (GpuResidentIndex, PathBuf) {
        let index = GpuResidentIndex::build_from_directory(test_dir)
            .expect("Failed to build index");
        let root_hash = cache_key(test_dir);
        let idx_path = cache_dir.join(format!("{:08x}.idx", root_hash));
        gsix_v2::save_v2(index.entries(), root_hash, &idx_path, 0, 0)
            .expect("Failed to save v2 index");
        (index, idx_path)
    }

    #[test]
    fn test_index_cache_save_and_mmap_load() {
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");

        // Build and save as v2
        let (index, saved_path) = save_v2_from_dir(test_dir.path(), cache_dir.path());
        let original_count = index.entry_count();
        assert!(original_count > 0, "Index should have entries");
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

        // Build and save as v2
        let (index, saved_path) = save_v2_from_dir(test_dir.path(), cache_dir.path());
        let original_count = index.entry_count();

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

        // Save as v2 with real root hash
        let (_, saved_path) = save_v2_from_dir(test_dir.path(), cache_dir.path());

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

        let (_, saved_path) = save_v2_from_dir(test_dir.path(), cache_dir.path());

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
        // Full roundtrip: build index -> save v2 -> mmap load -> verify byte-identical -> convert back
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");

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

        // Save as v2
        let idx_path = cache_dir.path().join("roundtrip.idx");
        let root_hash = cache_key(test_dir.path());
        gsix_v2::save_v2(index.entries(), root_hash, &idx_path, 0, 0)
            .expect("Failed to save v2");

        // Mmap load
        let cache = MmapIndexCache::load_mmap(&idx_path, None)
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
        let cache2 = MmapIndexCache::load_mmap(&idx_path, None)
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

    #[test]
    fn test_index_cache_rejects_v1_file() {
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let test_dir = make_test_dir();
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Save as v1 using SharedIndexManager
        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");
        let saved_path = manager
            .save(&index, test_dir.path())
            .expect("Failed to save v1");

        // Attempt to load via mmap — should reject v1
        let result = MmapIndexCache::load_mmap(&saved_path, None);
        assert!(result.is_err(), "v1 file should be rejected");
        match result {
            Err(CacheError::InvalidFormat(msg)) => {
                assert!(
                    msg.contains("v1") && msg.contains("rebuild"),
                    "Error should mention v1 rebuild: {}",
                    msg
                );
            }
            other => panic!("Expected InvalidFormat for v1, got: {:?}", other.map(|_| ())),
        }
    }

    /// Integration test matching the verify command name: test_index_cache
    #[test]
    fn test_index_cache() {
        // Build index, save as v2, load via mmap, verify identical entries
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");

        // Build
        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");
        let original_count = index.entry_count();
        assert!(original_count >= 5, "Should have at least 5 entries (3 files + src dir + 2 src files)");

        // Save as v2
        let root_hash = cache_key(test_dir.path());
        let idx_path = cache_dir.path().join(format!("{:08x}.idx", root_hash));
        gsix_v2::save_v2(index.entries(), root_hash, &idx_path, 0, 0)
            .expect("Failed to save v2");
        assert!(idx_path.exists(), "Index file should exist on disk");

        // "Restart" -- load from disk via mmap (simulates process restart)
        let cache = MmapIndexCache::load_mmap(&idx_path, Some(root_hash))
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
        let cache2 = MmapIndexCache::load_mmap(&idx_path, Some(root_hash))
            .expect("Failed to reload");
        let resident = cache2.into_resident_index();
        assert_eq!(resident.entry_count(), original_count);
    }
}
