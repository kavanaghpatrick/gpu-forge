//! Shared Index Manager
//!
//! Manages persistent GPU filesystem indexes at `~/.gpu-search/index/`.
//! Provides save/load/staleness checks for `GpuResidentIndex` data.
//!
//! Ported from rust-experiment/src/gpu_os/shared_index.rs (921 lines), simplified:
//! - Cache path changed from `~/.gpu_os/` to `~/.gpu-search/`
//! - Removed Metal device dependency (indexes are CPU-side Vec<GpuPathEntry>)
//! - Binary format: header (magic, version, entry_count, root_hash) + packed entries
//! - Cache key: hash of root path

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::gpu::types::GpuPathEntry;
use crate::index::gpu_index::GpuResidentIndex;

// ============================================================================
// Constants
// ============================================================================

/// Magic bytes for the index file format: "GSIX" (Gpu Search IndeX)
const INDEX_MAGIC: u32 = 0x58495347; // "GSIX" in little-endian

/// Current binary format version
const INDEX_VERSION: u32 = 1;

/// Header size in bytes (page-aligned for potential mmap use)
const HEADER_SIZE: usize = 64;

/// Default max age before an index is considered stale
pub const DEFAULT_MAX_AGE: Duration = Duration::from_secs(3600); // 1 hour

/// Default patterns to exclude from indexing
pub const DEFAULT_EXCLUDES: &[&str] = &[
    // Version control
    ".git",
    ".hg",
    ".svn",
    // Build artifacts
    "target",
    "build",
    "dist",
    "out",
    "bin",
    "obj",
    // Dependencies
    "node_modules",
    "vendor",
    ".cargo",
    "venv",
    ".venv",
    "env",
    // Caches
    ".cache",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    // IDE
    ".idea",
    ".vscode",
    ".vs",
    // macOS
    ".DS_Store",
    ".Spotlight-V100",
    ".Trashes",
    ".fseventsd",
    // Temporary
    "tmp",
    "temp",
    ".tmp",
];

// ============================================================================
// Errors
// ============================================================================

/// Errors from shared index operations.
#[derive(Debug)]
pub enum SharedIndexError {
    /// I/O error reading/writing index files
    Io(std::io::Error),
    /// Invalid index file format
    InvalidFormat(String),
    /// Index file not found
    NotFound(PathBuf),
    /// Index is stale and needs rebuilding
    Stale { age: Duration, max_age: Duration },
}

impl std::fmt::Display for SharedIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::InvalidFormat(msg) => write!(f, "Invalid index format: {}", msg),
            Self::NotFound(path) => write!(f, "Index not found: {}", path.display()),
            Self::Stale { age, max_age } => {
                write!(f, "Index stale: age {:?} exceeds max {:?}", age, max_age)
            }
        }
    }
}

impl std::error::Error for SharedIndexError {}

impl From<std::io::Error> for SharedIndexError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ============================================================================
// Index File Header
// ============================================================================

/// Binary header for saved index files.
///
/// Layout (64 bytes):
/// - [0..4]   magic:       u32  "GSIX"
/// - [4..8]   version:     u32  format version
/// - [8..12]  entry_count: u32  number of GpuPathEntry records
/// - [12..16] root_hash:   u32  hash of the root path (cache key)
/// - [16..24] saved_at:    u64  unix timestamp when saved
/// - [24..64] reserved:    [u8; 40]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct IndexHeader {
    magic: u32,
    version: u32,
    entry_count: u32,
    root_hash: u32,
    saved_at: u64,
    _reserved: [u8; 40],
}

impl IndexHeader {
    #[allow(dead_code)]
    const SIZE: usize = HEADER_SIZE;

    fn new(entry_count: u32, root_hash: u32) -> Self {
        let saved_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            magic: INDEX_MAGIC,
            version: INDEX_VERSION,
            entry_count,
            root_hash,
            saved_at,
            _reserved: [0u8; 40],
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.entry_count.to_le_bytes());
        buf[12..16].copy_from_slice(&self.root_hash.to_le_bytes());
        buf[16..24].copy_from_slice(&self.saved_at.to_le_bytes());
        // rest is zero (reserved)
        buf
    }

    fn from_bytes(buf: &[u8]) -> Result<Self, SharedIndexError> {
        if buf.len() < HEADER_SIZE {
            return Err(SharedIndexError::InvalidFormat(format!(
                "header too short: {} < {}",
                buf.len(),
                HEADER_SIZE
            )));
        }

        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        if magic != INDEX_MAGIC {
            return Err(SharedIndexError::InvalidFormat(format!(
                "bad magic: 0x{:08X} (expected 0x{:08X})",
                magic, INDEX_MAGIC
            )));
        }

        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != INDEX_VERSION {
            return Err(SharedIndexError::InvalidFormat(format!(
                "unsupported version: {} (expected {})",
                version, INDEX_VERSION
            )));
        }

        let entry_count = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let root_hash = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let saved_at = u64::from_le_bytes(buf[16..24].try_into().unwrap());

        Ok(Self {
            magic,
            version,
            entry_count,
            root_hash,
            saved_at,
            _reserved: [0u8; 40],
        })
    }
}

// Compile-time check that header is exactly 64 bytes
const _: () = assert!(std::mem::size_of::<IndexHeader>() == 64);

// ============================================================================
// SharedIndexManager
// ============================================================================

/// Manages persistent GPU filesystem indexes.
///
/// Indexes are saved as binary files at `<base_dir>/index/<hash>.idx`.
/// The default base directory is `~/.gpu-search/`.
///
/// # Binary Format
///
/// ```text
/// [IndexHeader: 64 bytes]
/// [GpuPathEntry[0]: 256 bytes]
/// [GpuPathEntry[1]: 256 bytes]
/// ...
/// [GpuPathEntry[N-1]: 256 bytes]
/// ```
pub struct SharedIndexManager {
    /// Base directory for index storage (default: ~/.gpu-search/)
    base_dir: PathBuf,
}

impl SharedIndexManager {
    /// Create a manager using the default location (`~/.gpu-search/`).
    pub fn new() -> Result<Self, SharedIndexError> {
        let home = dirs::home_dir().ok_or_else(|| {
            SharedIndexError::InvalidFormat("Could not find home directory".into())
        })?;
        let base_dir = home.join(".gpu-search");
        Ok(Self::with_base_dir(base_dir))
    }

    /// Create a manager with a custom base directory (useful for testing).
    pub fn with_base_dir(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    /// Get the base directory.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Get the index directory (`<base_dir>/index/`).
    pub fn index_dir(&self) -> PathBuf {
        self.base_dir.join("index")
    }

    /// Compute a cache key (hash) for a root path.
    pub fn cache_key(root: &Path) -> u32 {
        let canonical = root
            .canonicalize()
            .unwrap_or_else(|_| root.to_path_buf());
        let mut hasher = DefaultHasher::new();
        canonical.to_string_lossy().as_ref().hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Get the index file path for a given root directory.
    pub fn index_path(&self, root: &Path) -> PathBuf {
        let hash = Self::cache_key(root);
        self.index_dir().join(format!("{:08x}.idx", hash))
    }

    /// Save a `GpuResidentIndex` to disk for the given root path.
    ///
    /// Creates the index directory if it doesn't exist.
    /// The index is written atomically (write to temp, rename).
    pub fn save(&self, index: &GpuResidentIndex, root: &Path) -> Result<PathBuf, SharedIndexError> {
        let index_dir = self.index_dir();
        fs::create_dir_all(&index_dir)?;

        let idx_path = self.index_path(root);
        let root_hash = Self::cache_key(root);
        let entries = index.entries();
        let header = IndexHeader::new(entries.len() as u32, root_hash);

        // Write to a temp file first, then rename for atomicity
        let tmp_path = idx_path.with_extension("idx.tmp");
        {
            let mut file = fs::File::create(&tmp_path)?;

            // Write header
            file.write_all(&header.to_bytes())?;

            // Write packed GpuPathEntry array
            // SAFETY: GpuPathEntry is #[repr(C)] and exactly 256 bytes.
            let entry_bytes = unsafe {
                std::slice::from_raw_parts(
                    entries.as_ptr() as *const u8,
                    entries.len() * GpuPathEntry::SIZE,
                )
            };
            file.write_all(entry_bytes)?;
            file.flush()?;
        }

        // Atomic rename
        fs::rename(&tmp_path, &idx_path)?;

        Ok(idx_path)
    }

    /// Load a cached `GpuResidentIndex` for the given root path.
    ///
    /// Returns `SharedIndexError::NotFound` if no cached index exists.
    /// Returns `SharedIndexError::InvalidFormat` if the file is corrupt.
    pub fn load(&self, root: &Path) -> Result<GpuResidentIndex, SharedIndexError> {
        let idx_path = self.index_path(root);

        if !idx_path.exists() {
            return Err(SharedIndexError::NotFound(idx_path));
        }

        let data = fs::read(&idx_path)?;

        // Parse header
        let header = IndexHeader::from_bytes(&data)?;

        // Verify root hash matches
        let expected_hash = Self::cache_key(root);
        if header.root_hash != expected_hash {
            return Err(SharedIndexError::InvalidFormat(format!(
                "root hash mismatch: file has 0x{:08X}, expected 0x{:08X}",
                header.root_hash, expected_hash
            )));
        }

        // Verify data length
        let expected_len = HEADER_SIZE + (header.entry_count as usize) * GpuPathEntry::SIZE;
        if data.len() < expected_len {
            return Err(SharedIndexError::InvalidFormat(format!(
                "file too short: {} < {} (header says {} entries)",
                data.len(),
                expected_len,
                header.entry_count
            )));
        }

        // Read entries
        let entry_data = &data[HEADER_SIZE..];
        let mut entries = Vec::with_capacity(header.entry_count as usize);

        for i in 0..header.entry_count as usize {
            let offset = i * GpuPathEntry::SIZE;
            let entry_slice = &entry_data[offset..offset + GpuPathEntry::SIZE];

            // SAFETY: GpuPathEntry is #[repr(C)], Copy, and 256 bytes.
            // We verified the data length above.
            let entry: GpuPathEntry = unsafe {
                std::ptr::read(entry_slice.as_ptr() as *const GpuPathEntry)
            };
            entries.push(entry);
        }

        // Reconstruct GpuResidentIndex directly from saved entries.
        // We don't use build_from_paths because that re-stats files (which may have changed).
        Ok(GpuResidentIndex::from_entries(entries))
    }

    /// Check if the cached index for a root path is stale.
    ///
    /// An index is considered stale if:
    /// 1. It doesn't exist
    /// 2. It's older than `max_age`
    /// 3. The root directory's modification time is newer than the index
    pub fn is_stale(&self, root: &Path, max_age: Duration) -> bool {
        let idx_path = self.index_path(root);

        // No cached index -> stale
        if !idx_path.exists() {
            return true;
        }

        // Read header to check saved_at timestamp
        let data = match fs::read(&idx_path) {
            Ok(d) => d,
            Err(_) => return true,
        };

        let header = match IndexHeader::from_bytes(&data) {
            Ok(h) => h,
            Err(_) => return true,
        };

        // Check age
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let age = Duration::from_secs(now.saturating_sub(header.saved_at));
        if age >= max_age {
            return true;
        }

        // Check if root directory was modified after index was saved
        if let Ok(meta) = fs::metadata(root) {
            if let Ok(mtime) = meta.modified() {
                let mtime_secs = mtime
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if mtime_secs > header.saved_at {
                    return true;
                }
            }
        }

        false
    }

    /// Delete the cached index for a root path.
    pub fn delete(&self, root: &Path) -> Result<(), SharedIndexError> {
        let idx_path = self.index_path(root);
        if idx_path.exists() {
            fs::remove_file(&idx_path)?;
        }
        Ok(())
    }

    /// List all cached index files.
    pub fn list_indexes(&self) -> Vec<PathBuf> {
        let index_dir = self.index_dir();
        if !index_dir.exists() {
            return Vec::new();
        }

        fs::read_dir(&index_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "idx")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_shared_index_save_and_load() {
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Build an index from the test directory
        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");

        let original_count = index.entry_count();
        assert!(original_count > 0, "Index should have entries");

        // Save
        let saved_path = manager
            .save(&index, test_dir.path())
            .expect("Failed to save index");
        assert!(saved_path.exists(), "Saved index file should exist");

        // Load
        let loaded = manager
            .load(test_dir.path())
            .expect("Failed to load index");

        // Verify identical
        assert_eq!(
            loaded.entry_count(),
            original_count,
            "Loaded entry count should match"
        );

        // Verify each entry matches
        for i in 0..original_count {
            let orig = index.get_entry(i).unwrap();
            let load = loaded.get_entry(i).unwrap();

            assert_eq!(
                orig.path_len, load.path_len,
                "Entry {} path_len mismatch",
                i
            );
            assert_eq!(
                &orig.path[..orig.path_len as usize],
                &load.path[..load.path_len as usize],
                "Entry {} path mismatch",
                i
            );
            assert_eq!(orig.flags, load.flags, "Entry {} flags mismatch", i);
            assert_eq!(orig.size_lo, load.size_lo, "Entry {} size_lo mismatch", i);
            assert_eq!(orig.size_hi, load.size_hi, "Entry {} size_hi mismatch", i);
            assert_eq!(orig.mtime, load.mtime, "Entry {} mtime mismatch", i);
        }
    }

    #[test]
    fn test_shared_index_cache_key() {
        let test_dir = make_test_dir();

        // Same path should produce same hash
        let key1 = SharedIndexManager::cache_key(test_dir.path());
        let key2 = SharedIndexManager::cache_key(test_dir.path());
        assert_eq!(key1, key2, "Same path should produce same cache key");
    }

    #[test]
    fn test_shared_index_not_found() {
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        let result = manager.load(Path::new("/nonexistent/path"));
        assert!(result.is_err(), "Loading nonexistent index should fail");

        if let Err(SharedIndexError::NotFound(_)) = result {
            // expected
        } else {
            panic!("Expected NotFound error, got: {:?}", result.err());
        }
    }

    #[test]
    fn test_shared_index_staleness() {
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // No cached index -> stale
        assert!(
            manager.is_stale(test_dir.path(), DEFAULT_MAX_AGE),
            "Missing index should be stale"
        );

        // Save an index
        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");
        manager
            .save(&index, test_dir.path())
            .expect("Failed to save");

        // Freshly saved -> not stale (with generous max_age)
        assert!(
            !manager.is_stale(test_dir.path(), Duration::from_secs(3600)),
            "Fresh index should not be stale"
        );

        // With zero max_age -> always stale
        assert!(
            manager.is_stale(test_dir.path(), Duration::from_secs(0)),
            "Index should be stale with zero max_age"
        );
    }

    #[test]
    fn test_shared_index_delete() {
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Save
        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");
        let saved_path = manager
            .save(&index, test_dir.path())
            .expect("Failed to save");
        assert!(saved_path.exists());

        // Delete
        manager
            .delete(test_dir.path())
            .expect("Failed to delete");
        assert!(!saved_path.exists(), "Index file should be deleted");

        // Load should fail
        assert!(manager.load(test_dir.path()).is_err());
    }

    #[test]
    fn test_shared_index_list_indexes() {
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Initially empty
        assert!(manager.list_indexes().is_empty());

        // Save one
        let index = GpuResidentIndex::build_from_directory(test_dir.path())
            .expect("Failed to build index");
        manager
            .save(&index, test_dir.path())
            .expect("Failed to save");

        let indexes = manager.list_indexes();
        assert_eq!(indexes.len(), 1, "Should have 1 index file");
        assert!(indexes[0].extension().unwrap() == "idx");
    }

    #[test]
    fn test_shared_index_empty_index() {
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Build an empty index
        let empty_dir = TempDir::new().expect("Failed to create empty dir");
        let index = GpuResidentIndex::build_from_directory(empty_dir.path())
            .expect("Failed to build index");
        assert_eq!(index.entry_count(), 0);

        // Save empty
        manager
            .save(&index, empty_dir.path())
            .expect("Failed to save empty index");

        // Load empty
        let loaded = manager
            .load(empty_dir.path())
            .expect("Failed to load empty index");
        assert_eq!(loaded.entry_count(), 0);
    }

    #[test]
    fn test_shared_index_index_dir_creation() {
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let base = cache_dir.path().join("deep").join("nested").join("path");
        let manager = SharedIndexManager::with_base_dir(base.clone());

        let empty_dir = TempDir::new().expect("Failed to create test dir");
        let index = GpuResidentIndex::build_from_directory(empty_dir.path())
            .expect("Failed to build index");

        // Save should create nested directories
        manager
            .save(&index, empty_dir.path())
            .expect("Failed to save (should create dirs)");

        assert!(
            base.join("index").exists(),
            "Index directory should be created"
        );
    }

    #[test]
    fn test_shared_index_corrupt_file() {
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        let test_dir = TempDir::new().expect("Failed to create test dir");
        let idx_path = manager.index_path(test_dir.path());

        // Create index directory
        fs::create_dir_all(manager.index_dir()).unwrap();

        // Write garbage data
        fs::write(&idx_path, b"this is not an index file").unwrap();

        let result = manager.load(test_dir.path());
        assert!(result.is_err(), "Loading corrupt index should fail");

        if let Err(SharedIndexError::InvalidFormat(_)) = result {
            // expected
        } else {
            panic!("Expected InvalidFormat error");
        }
    }

    #[test]
    fn test_shared_index_binary_roundtrip_fidelity() {
        let test_dir = make_test_dir();
        let cache_dir = TempDir::new().expect("Failed to create cache dir");
        let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

        // Build with known content
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

        // Save and reload
        manager.save(&index, test_dir.path()).expect("save failed");
        let loaded = manager.load(test_dir.path()).expect("load failed");

        assert_eq!(loaded.entry_count(), original_count);

        // Verify byte-level fidelity of all fields
        for i in 0..original_count {
            let orig = index.get_entry(i).unwrap();
            let load = loaded.get_entry(i).unwrap();

            // Compare ALL 256 bytes
            let orig_bytes: &[u8; 256] = unsafe { &*(orig as *const GpuPathEntry as *const [u8; 256]) };
            let load_bytes: &[u8; 256] = unsafe { &*(load as *const GpuPathEntry as *const [u8; 256]) };

            assert_eq!(
                orig_bytes, load_bytes,
                "Entry {} byte-level mismatch",
                i
            );
        }
    }
}
