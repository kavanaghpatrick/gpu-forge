// GSIX v2 binary format: 16KB page-aligned header for GPU-friendly persistent file index.
//
// The v2 format uses a 16384-byte header (matching Apple Silicon page size)
// followed by contiguous 256-byte GpuPathEntry records suitable for
// Metal bytesNoCopy zero-copy GPU dispatch.

use std::io::Write;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::gpu::types::GpuPathEntry;
use crate::index::cache::CacheError;

/// 16KB header size, matching Apple Silicon page alignment.
pub const HEADER_SIZE_V2: usize = 16384;

/// Magic bytes: "GSIX" in little-endian (0x47='G', 0x53='S', 0x49='I', 0x58='X').
pub const INDEX_MAGIC: u32 = 0x58495347;

/// Version 2 of the GSIX format.
pub const INDEX_VERSION_V2: u32 = 2;

/// Default maximum age for a stale index: 1 hour (3600 seconds).
///
/// If the index's `saved_at` timestamp is older than this many seconds
/// from the current time, the index is considered stale and a background
/// update should be triggered.
pub const DEFAULT_MAX_AGE_SECS: u64 = 3600;

/// Header flag: entries are sorted by path bytes.
pub const FLAG_SORTED: u32 = 0x1;

/// Header flag: tombstoned entries have been compacted out.
pub const FLAG_COMPACTED: u32 = 0x2;

/// GSIX v2 file header. 16KB page-aligned for Metal bytesNoCopy compatibility.
///
/// Layout (all fields little-endian):
///   [0..4)    magic           0x58495347 ("GSIX")
///   [4..8)    version         2
///   [8..12)   entry_count     number of valid entries
///   [12..16)  root_hash       hash of the indexed root path
///   [16..24)  saved_at        unix timestamp (seconds) of last save
///   [24..32)  last_fsevents_id  macOS FSEvents event ID for resume
///   [32..36)  exclude_hash    CRC32 of exclude configuration
///   [36..40)  entry_capacity  allocated entry slots (>= entry_count)
///   [40..44)  flags           bitfield (FLAG_SORTED, FLAG_COMPACTED, etc.)
///   [44..48)  checksum        CRC32 over bytes [0..44)
///   [48..16384) _reserved     zero-filled padding to 16KB page boundary
#[repr(C)]
#[derive(Clone, Debug)]
pub struct GsixHeaderV2 {
    /// Magic number: INDEX_MAGIC (0x58495347)
    pub magic: u32,
    /// Format version: INDEX_VERSION_V2 (2)
    pub version: u32,
    /// Number of valid (non-deleted) entries in the file
    pub entry_count: u32,
    /// Hash of the indexed root path
    pub root_hash: u32,
    /// Unix timestamp (seconds since epoch) of last save
    pub saved_at: u64,
    /// macOS FSEvents event ID for resumable watching
    pub last_fsevents_id: u64,
    /// CRC32 hash of the exclude configuration for change detection
    pub exclude_hash: u32,
    /// Total allocated entry slots (may be > entry_count after compaction)
    pub entry_capacity: u32,
    /// Bitfield flags (FLAG_SORTED, FLAG_COMPACTED, etc.)
    pub flags: u32,
    /// CRC32 checksum over header bytes [0..44) for integrity validation
    pub checksum: u32,
    /// Reserved padding to fill header to exactly 16384 bytes
    pub _reserved: [u8; 16336],
}

// Compile-time assertions: verify alignment invariants required for Metal bytesNoCopy.
const _: () = assert!(size_of::<GsixHeaderV2>() == 16384);
const _: () = assert!(HEADER_SIZE_V2 == 16384);
const _: () = assert!(HEADER_SIZE_V2.is_multiple_of(16384));
const _: () = assert!(size_of::<GpuPathEntry>() == 256);

/// Byte offset where the CRC32 checksum is stored in the header.
const CHECKSUM_OFFSET: usize = 44;

impl GsixHeaderV2 {
    /// Create a new GsixHeaderV2 with the given entry_count and root_hash.
    /// Sets magic, version, saved_at (now), and zeroes everything else.
    pub fn new(entry_count: u32, root_hash: u32) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            magic: INDEX_MAGIC,
            version: INDEX_VERSION_V2,
            entry_count,
            root_hash,
            saved_at: now,
            last_fsevents_id: 0,
            exclude_hash: 0,
            entry_capacity: entry_count,
            flags: 0,
            checksum: 0,
            _reserved: [0u8; 16336],
        }
    }

    /// Returns `true` if entries are sorted by path bytes.
    pub fn is_sorted(&self) -> bool {
        self.flags & FLAG_SORTED != 0
    }

    /// Returns `true` if tombstoned entries have been compacted out.
    pub fn is_compacted(&self) -> bool {
        self.flags & FLAG_COMPACTED != 0
    }

    /// Set the FLAG_SORTED bit in flags.
    pub fn set_sorted(&mut self) {
        self.flags |= FLAG_SORTED;
    }

    /// Set the FLAG_COMPACTED bit in flags.
    pub fn set_compacted(&mut self) {
        self.flags |= FLAG_COMPACTED;
    }

    /// Serialize this header to a 16384-byte array (little-endian).
    /// Computes CRC32 over bytes [0..44) and stores it at [44..48).
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE_V2] {
        let mut buf = [0u8; HEADER_SIZE_V2];

        // Write fields little-endian
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.entry_count.to_le_bytes());
        buf[12..16].copy_from_slice(&self.root_hash.to_le_bytes());
        buf[16..24].copy_from_slice(&self.saved_at.to_le_bytes());
        buf[24..32].copy_from_slice(&self.last_fsevents_id.to_le_bytes());
        buf[32..36].copy_from_slice(&self.exclude_hash.to_le_bytes());
        buf[36..40].copy_from_slice(&self.entry_capacity.to_le_bytes());
        buf[40..44].copy_from_slice(&self.flags.to_le_bytes());
        // [44..48) = checksum, computed below
        // [48..16384) = _reserved, already zeroed

        // Compute CRC32 over bytes [0..44)
        let crc = crc32fast::hash(&buf[..CHECKSUM_OFFSET]);
        buf[CHECKSUM_OFFSET..CHECKSUM_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());

        buf
    }

    /// Deserialize a GsixHeaderV2 from a byte buffer.
    /// Validates magic, version, and CRC32 checksum.
    /// Returns `Err(CacheError::InvalidFormat)` on any validation failure.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, CacheError> {
        if buf.len() < HEADER_SIZE_V2 {
            return Err(CacheError::InvalidFormat(format!(
                "header too small: {} bytes, expected {}",
                buf.len(),
                HEADER_SIZE_V2
            )));
        }

        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        if magic != INDEX_MAGIC {
            return Err(CacheError::InvalidFormat(format!(
                "bad magic: 0x{:08X}, expected 0x{:08X}",
                magic, INDEX_MAGIC
            )));
        }

        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != INDEX_VERSION_V2 {
            return Err(CacheError::InvalidFormat(format!(
                "unsupported version: {}, expected {}",
                version, INDEX_VERSION_V2
            )));
        }

        // Validate CRC32 checksum over [0..44)
        let expected_crc = crc32fast::hash(&buf[..CHECKSUM_OFFSET]);
        let stored_crc =
            u32::from_le_bytes(buf[CHECKSUM_OFFSET..CHECKSUM_OFFSET + 4].try_into().unwrap());
        if stored_crc != expected_crc {
            return Err(CacheError::InvalidFormat(format!(
                "checksum mismatch: stored 0x{:08X}, computed 0x{:08X}",
                stored_crc, expected_crc
            )));
        }

        let entry_count = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let root_hash = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let saved_at = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let last_fsevents_id = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let exclude_hash = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        let entry_capacity = u32::from_le_bytes(buf[36..40].try_into().unwrap());
        let flags = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        let checksum = stored_crc;

        let mut reserved = [0u8; 16336];
        reserved.copy_from_slice(&buf[48..HEADER_SIZE_V2]);

        Ok(Self {
            magic,
            version,
            entry_count,
            root_hash,
            saved_at,
            last_fsevents_id,
            exclude_hash,
            entry_capacity,
            flags,
            checksum,
            _reserved: reserved,
        })
    }
}

/// Save entries as a GSIX v2 index file using atomic write (write to .tmp, fsync, rename).
///
/// Writes a 16KB v2 header followed by packed 256-byte `GpuPathEntry` records.
/// The write is atomic: data goes to `path.with_extension("idx.tmp")`, is fsynced,
/// then renamed over the final `path`.
///
/// # Arguments
/// * `entries` - Slice of GpuPathEntry records to persist
/// * `root_hash` - Hash of the indexed root path
/// * `path` - Final `.idx` file path (temp file is `path.with_extension("idx.tmp")`)
/// * `last_fsevents_id` - macOS FSEvents event ID for resume on next startup
/// * `exclude_hash` - CRC32 of exclude configuration for change detection
///
/// # Returns
/// The final path on success.
pub fn save_v2(
    entries: &[GpuPathEntry],
    root_hash: u32,
    path: &Path,
    last_fsevents_id: u64,
    exclude_hash: u32,
) -> Result<PathBuf, CacheError> {
    // Build header with all fields
    let mut header = GsixHeaderV2::new(entries.len() as u32, root_hash);
    header.last_fsevents_id = last_fsevents_id;
    header.exclude_hash = exclude_hash;
    header.entry_capacity = entries.len() as u32;

    // Serialize header (to_bytes computes and stores CRC32 checksum)
    let header_bytes = header.to_bytes();

    // Write to temporary file
    let tmp_path = path.with_extension("idx.tmp");

    // Ensure parent directory exists
    if let Some(parent) = tmp_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = std::fs::File::create(&tmp_path)?;

    // Write 16KB header
    file.write_all(&header_bytes)?;

    // Write packed entries (each exactly 256 bytes, #[repr(C)])
    for entry in entries {
        // SAFETY: GpuPathEntry is #[repr(C)], exactly 256 bytes, no padding issues.
        let entry_bytes: &[u8; 256] =
            unsafe { &*(entry as *const GpuPathEntry as *const [u8; 256]) };
        file.write_all(entry_bytes)?;
    }

    // fsync to ensure data is on disk before rename
    file.sync_all()?;

    // Drop file handle before rename
    drop(file);

    // Atomic rename .idx.tmp -> .idx
    std::fs::rename(&tmp_path, path)?;

    Ok(path.to_path_buf())
}

/// Load a GSIX v2 index file, returning the validated header and entry vector.
///
/// Reads the file at `path`, validates the 16KB v2 header (magic, version, CRC32
/// checksum), parses all header fields (last_fsevents_id, exclude_hash, entry_capacity,
/// flags), and returns entries starting at byte offset 16384.
///
/// # Validation
/// - Header must be at least 16384 bytes
/// - Magic must be 0x58495347 ("GSIX")
/// - Version must be 2
/// - CRC32 checksum over bytes [0..44) must match stored checksum at [44..48)
/// - `entry_count * 256 + 16384 <= file_len` (entries must fit in file)
///
/// # Returns
/// `(GsixHeaderV2, Vec<GpuPathEntry>)` on success.
pub fn load_v2(path: &Path) -> Result<(GsixHeaderV2, Vec<GpuPathEntry>), CacheError> {
    let data = std::fs::read(path)?;

    if data.len() < HEADER_SIZE_V2 {
        return Err(CacheError::InvalidFormat(format!(
            "file too small for v2 header: {} bytes, need {}",
            data.len(),
            HEADER_SIZE_V2
        )));
    }

    // Validate and parse header (checks magic, version, CRC32)
    let header = GsixHeaderV2::from_bytes(&data)?;

    let entry_count = header.entry_count as usize;
    let required_len = HEADER_SIZE_V2 + entry_count * GpuPathEntry::SIZE;

    if data.len() < required_len {
        return Err(CacheError::InvalidFormat(format!(
            "file too short for {} entries: {} bytes, need {}",
            entry_count,
            data.len(),
            required_len
        )));
    }

    // Parse entries from bytes starting at offset 16384
    let mut entries = Vec::with_capacity(entry_count);
    for i in 0..entry_count {
        let offset = HEADER_SIZE_V2 + i * GpuPathEntry::SIZE;
        // SAFETY: GpuPathEntry is #[repr(C)], exactly 256 bytes, alignment 4.
        // We verified data.len() >= offset + 256 above.
        // Copying via ptr::read_unaligned handles any alignment issues from the
        // file buffer (Vec<u8> is only 1-byte aligned).
        let entry: GpuPathEntry = unsafe {
            std::ptr::read_unaligned(data[offset..].as_ptr() as *const GpuPathEntry)
        };
        entries.push(entry);
    }

    Ok((header, entries))
}

/// Version 1 of the GSIX format (legacy, 64-byte header).
pub const INDEX_VERSION_V1: u32 = 1;

/// Detect the GSIX format version from raw file bytes.
///
/// Reads the first 8 bytes to extract magic and version. Returns:
/// - `Ok(1)` for v1 (magic = 0x58495347, version = 1)
/// - `Ok(2)` for v2 (magic = 0x58495347, version = 2)
/// - `Err(CacheError::InvalidFormat)` for unrecognized magic or unknown version
pub fn detect_version(data: &[u8]) -> Result<u32, CacheError> {
    if data.len() < 8 {
        return Err(CacheError::InvalidFormat(format!(
            "file too small for version detection: {} bytes, need at least 8",
            data.len()
        )));
    }

    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != INDEX_MAGIC {
        return Err(CacheError::InvalidFormat(format!(
            "bad magic: 0x{:08X}, expected 0x{:08X}",
            magic, INDEX_MAGIC
        )));
    }

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    match version {
        INDEX_VERSION_V1 | INDEX_VERSION_V2 => Ok(version),
        _ => Err(CacheError::InvalidFormat(format!(
            "unknown version: {}, expected 1 or 2",
            version
        ))),
    }
}

/// Attempt to load a GSIX index file, handling v1 migration.
///
/// If the file is v1 format, logs a warning, deletes the old file, and returns
/// `CacheError::InvalidFormat` signaling that a rebuild is required.
/// If the file is v2 format, delegates to `load_v2`.
pub fn load_with_migration(path: &Path) -> Result<(GsixHeaderV2, Vec<GpuPathEntry>), CacheError> {
    let data = std::fs::read(path)?;
    let version = detect_version(&data)?;

    if version == INDEX_VERSION_V1 {
        eprintln!(
            "gpu-search: v1 index detected at {}, deleting for rebuild",
            path.display()
        );
        // Remove the old v1 file to force a rebuild
        if let Err(e) = std::fs::remove_file(path) {
            eprintln!(
                "gpu-search: warning: failed to remove v1 index {}: {}",
                path.display(),
                e
            );
        }
        return Err(CacheError::InvalidFormat(
            "v1 index detected, rebuild required".to_string(),
        ));
    }

    // v2: parse via load_v2 (re-read not needed, but load_v2 reads from path)
    load_v2(path)
}

/// Clean up legacy v1 per-directory `.idx` files in the given index directory.
///
/// Deletes all `.idx` files whose stem is NOT "global" (the v2 global index),
/// then creates a `.v2-migrated` marker file to indicate migration has occurred.
pub fn cleanup_v1_indexes(index_dir: &Path) -> std::io::Result<()> {
    if !index_dir.is_dir() {
        return Ok(());
    }

    for entry in std::fs::read_dir(index_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("idx") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if stem != "global" {
                    if let Err(e) = std::fs::remove_file(&path) {
                        eprintln!(
                            "gpu-search: warning: failed to remove v1 index {}: {}",
                            path.display(),
                            e
                        );
                    }
                }
            }
        }
    }

    // Create marker file to indicate v2 migration has been performed
    let marker = index_dir.join(".v2-migrated");
    std::fs::File::create(marker)?;

    Ok(())
}

/// Check if a GSIX v2 header indicates a stale index using the default max age.
///
/// An index is stale if `saved_at + DEFAULT_MAX_AGE_SECS < now` (i.e. the index
/// is older than 1 hour). A stale index can still serve search (fast but possibly
/// incomplete) while a background update runs.
///
/// Returns `false` if `saved_at` is 0 (never saved — handled by initial build).
pub fn is_stale(header: &GsixHeaderV2) -> bool {
    is_stale_with_age(header, DEFAULT_MAX_AGE_SECS)
}

/// Check if a GSIX v2 header indicates a stale index using a custom max age.
///
/// An index is stale if `saved_at + max_age_secs < now`.
///
/// Returns `false` if `saved_at` is 0 (never saved — handled by initial build).
pub fn is_stale_with_age(header: &GsixHeaderV2, max_age_secs: u64) -> bool {
    if header.saved_at == 0 {
        return false;
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Stale if saved_at + max_age < now, i.e. the index is older than max_age
    now.saturating_sub(header.saved_at) > max_age_secs
}

/// Test utilities for generating synthetic GpuPathEntry data.
///
/// Available only in test builds. Use `generate_synthetic_entries(count)` to
/// produce realistic entries with varied path depths, extensions, sizes, and mtimes.
#[cfg(test)]
pub(crate) mod test_helpers {
    use crate::gpu::types::{path_flags, GpuPathEntry};

    /// Directory name components for building realistic paths.
    const DIR_COMPONENTS: &[&str] = &[
        "src", "lib", "tests", "benches", "docs", "config", "build", "target", "scripts",
        "utils", "core", "api", "models", "views", "controllers", "services", "middleware",
        "handlers", "proto", "internal", "pkg", "cmd", "assets", "static", "templates",
    ];

    /// File extensions with representative distribution.
    const EXTENSIONS: &[&str] = &[
        ".rs", ".rs", ".rs", // weighted: Rust is most common in this project
        ".txt", ".js", ".py", ".md", ".toml", ".json", ".yaml", ".html", ".css", ".ts",
        ".sh", ".c", ".h", ".go", ".swift", ".metal",
    ];

    /// Root prefixes for realistic Unix paths.
    const ROOT_PREFIXES: &[&str] = &[
        "/Users/dev/project",
        "/Users/dev/workspace",
        "/home/user/code",
        "/opt/builds",
        "/var/lib/app",
        "/usr/local/src",
    ];

    /// Simple deterministic LCG pseudo-random number generator (no rand crate needed).
    struct Lcg {
        state: u32,
    }

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn next(&mut self) -> u32 {
            self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
            self.state
        }

        /// Returns a value in [0, bound).
        fn next_range(&mut self, bound: u32) -> u32 {
            (self.next() >> 4) % bound
        }
    }

    /// Generate `count` synthetic `GpuPathEntry` values with realistic paths.
    ///
    /// Paths have varied depths (1-8), common file extensions, realistic sizes and mtimes.
    /// Uses a deterministic pseudo-random sequence for reproducibility.
    ///
    /// # Example
    /// ```ignore
    /// let entries = generate_synthetic_entries(1000);
    /// assert_eq!(entries.len(), 1000);
    /// assert!(entries[0].path_len > 0);
    /// ```
    pub(crate) fn generate_synthetic_entries(count: usize) -> Vec<GpuPathEntry> {
        let mut rng = Lcg::new(0xBEEF_CAFE);
        let mut entries = Vec::with_capacity(count);

        for i in 0..count {
            let mut entry = GpuPathEntry::new();

            // Pick a root prefix
            let root = ROOT_PREFIXES[rng.next_range(ROOT_PREFIXES.len() as u32) as usize];

            // Build path with depth 1-8 directory components
            let depth = (rng.next_range(8) + 1) as usize; // 1..=8
            let mut path = String::with_capacity(224);
            path.push_str(root);

            for _ in 0..depth {
                path.push('/');
                let comp = DIR_COMPONENTS[rng.next_range(DIR_COMPONENTS.len() as u32) as usize];
                path.push_str(comp);
            }

            // Add filename with extension
            let ext = EXTENSIONS[rng.next_range(EXTENSIONS.len() as u32) as usize];
            path.push_str(&format!("/file_{}{}", i, ext));

            // Truncate if path exceeds GPU_PATH_MAX_LEN (224 bytes)
            if path.len() > 224 {
                path.truncate(224);
            }

            entry.set_path(path.as_bytes());

            // Flags: ~10% directories, ~5% hidden, ~2% symlinks
            let flag_roll = rng.next_range(100);
            if flag_roll < 10 {
                entry.flags |= path_flags::IS_DIR;
            }
            if flag_roll < 5 {
                entry.flags |= path_flags::IS_HIDDEN;
            }
            if flag_roll < 2 {
                entry.flags |= path_flags::IS_SYMLINK;
            }

            // Size: 0 for dirs, 100B-10MB for files
            if entry.flags & path_flags::IS_DIR == 0 {
                let size = (rng.next_range(10_000_000) + 100) as u64;
                entry.set_size(size);
            }

            // mtime: spread across 2023-2025 (unix timestamps ~1672531200 to ~1735689600)
            let base_mtime: u32 = 1_672_531_200; // 2023-01-01
            let offset = rng.next_range(63_158_400); // ~2 years in seconds
            entry.mtime = base_mtime + offset;

            // parent_idx: set to a plausible parent (0..i) or u32::MAX for root-level
            if i > 0 && depth > 1 {
                entry.parent_idx = rng.next_range(i as u32);
            }

            entries.push(entry);
        }

        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_entries_count() {
        for count in [0, 1, 10, 100, 1000] {
            let entries = test_helpers::generate_synthetic_entries(count);
            assert_eq!(entries.len(), count, "should generate exactly {count} entries");
        }
    }

    #[test]
    fn test_synthetic_entries_realistic_paths() {
        let entries = test_helpers::generate_synthetic_entries(50);

        for (i, entry) in entries.iter().enumerate() {
            let path_len = entry.path_len as usize;
            assert!(path_len > 0, "entry {i} should have non-zero path length");
            assert!(path_len <= 224, "entry {i} path_len {path_len} exceeds GPU_PATH_MAX_LEN");

            let path = std::str::from_utf8(&entry.path[..path_len])
                .unwrap_or_else(|_| panic!("entry {i} path should be valid UTF-8"));
            assert!(path.starts_with('/'), "entry {i} path should start with /: {path}");
            assert!(path.contains('/'), "entry {i} path should contain directory separators");

            // Should have a file extension (or be a truncated long path)
            let has_extension = path.contains('.');
            let is_long = path_len >= 200; // truncated paths may lose extension
            assert!(
                has_extension || is_long,
                "entry {i} should have extension or be a long truncated path: {path}"
            );
        }
    }

    #[test]
    fn test_synthetic_entries_varied_depths() {
        let entries = test_helpers::generate_synthetic_entries(200);

        // Count slash-separated segments to verify depth variation
        let depths: Vec<usize> = entries
            .iter()
            .map(|e| {
                let path = std::str::from_utf8(&e.path[..e.path_len as usize]).unwrap();
                path.matches('/').count()
            })
            .collect();

        let min_depth = *depths.iter().min().unwrap();
        let max_depth = *depths.iter().max().unwrap();

        // With 200 entries and depths 1-8+root, we should see meaningful variation
        assert!(max_depth - min_depth >= 3, "depths should vary: min={min_depth}, max={max_depth}");
    }

    #[test]
    fn test_synthetic_entries_varied_extensions() {
        let entries = test_helpers::generate_synthetic_entries(200);

        let mut ext_set = std::collections::HashSet::new();
        for entry in &entries {
            let path = std::str::from_utf8(&entry.path[..entry.path_len as usize]).unwrap();
            if let Some(dot_pos) = path.rfind('.') {
                ext_set.insert(path[dot_pos..].to_string());
            }
        }

        // With 19 extensions and 200 entries, we should see at least 5 unique
        assert!(ext_set.len() >= 5, "should have varied extensions: found {:?}", ext_set);
    }

    #[test]
    fn test_synthetic_entries_varied_sizes_and_mtimes() {
        let entries = test_helpers::generate_synthetic_entries(100);

        let sizes: Vec<u64> = entries.iter().map(|e| e.size()).collect();
        let mtimes: Vec<u32> = entries.iter().map(|e| e.mtime).collect();

        // Sizes should vary (dirs are 0, files are 100B-10MB)
        let unique_sizes: std::collections::HashSet<u64> = sizes.iter().copied().collect();
        assert!(unique_sizes.len() >= 10, "should have varied sizes: {} unique", unique_sizes.len());

        // mtimes should vary across the 2-year range
        let min_mtime = *mtimes.iter().min().unwrap();
        let max_mtime = *mtimes.iter().max().unwrap();
        assert!(
            max_mtime - min_mtime >= 1_000_000,
            "mtimes should span at least ~11 days: range={}",
            max_mtime - min_mtime
        );
    }

    #[test]
    fn test_synthetic_entries_deterministic() {
        let a = test_helpers::generate_synthetic_entries(50);
        let b = test_helpers::generate_synthetic_entries(50);

        for (i, (ea, eb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(ea.path_len, eb.path_len, "entry {i} path_len should be deterministic");
            assert_eq!(
                &ea.path[..ea.path_len as usize],
                &eb.path[..eb.path_len as usize],
                "entry {i} path should be deterministic"
            );
            assert_eq!(ea.mtime, eb.mtime, "entry {i} mtime should be deterministic");
            assert_eq!(ea.size(), eb.size(), "entry {i} size should be deterministic");
            assert_eq!(ea.flags, eb.flags, "entry {i} flags should be deterministic");
        }
    }

    #[test]
    fn test_header_size_is_16384() {
        assert_eq!(size_of::<GsixHeaderV2>(), HEADER_SIZE_V2);
        assert_eq!(size_of::<GsixHeaderV2>(), 16384);
    }

    #[test]
    fn test_header_magic_bytes() {
        let header = GsixHeaderV2::new(0, 0);
        assert_eq!(header.magic, 0x58495347, "magic must be 'GSIX' in LE");
        let bytes = header.to_bytes();
        // Verify raw bytes spell "GSIX" in little-endian
        assert_eq!(bytes[0], b'G');
        assert_eq!(bytes[1], b'S');
        assert_eq!(bytes[2], b'I');
        assert_eq!(bytes[3], b'X');
    }

    #[test]
    fn test_header_version_is_2() {
        let header = GsixHeaderV2::new(0, 0);
        assert_eq!(header.version, 2);
        let bytes = header.to_bytes();
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, 2);
    }

    #[test]
    fn test_header_serialization_roundtrip() {
        let mut header = GsixHeaderV2::new(42, 0xDEADBEEF);
        header.last_fsevents_id = 123456789;
        header.exclude_hash = 0xCAFEBABE;
        header.entry_capacity = 100;
        header.flags = 0x3;

        let bytes = header.to_bytes();
        let recovered = GsixHeaderV2::from_bytes(&bytes).expect("roundtrip should succeed");

        assert_eq!(recovered.magic, INDEX_MAGIC);
        assert_eq!(recovered.version, INDEX_VERSION_V2);
        assert_eq!(recovered.entry_count, 42);
        assert_eq!(recovered.root_hash, 0xDEADBEEF);
        assert_eq!(recovered.saved_at, header.saved_at);
        assert_eq!(recovered.last_fsevents_id, 123456789);
        assert_eq!(recovered.exclude_hash, 0xCAFEBABE);
        assert_eq!(recovered.entry_capacity, 100);
        assert_eq!(recovered.flags, 0x3);
        // Verify checksum field matches the bytes at offset 44..48
        let crc_bytes = &bytes[44..48];
        let crc_val = u32::from_le_bytes(crc_bytes.try_into().unwrap());
        assert_eq!(recovered.checksum, crc_val, "checksum field must match serialized bytes");

        // Byte-level identity: re-serializing recovered header produces identical bytes
        let bytes2 = recovered.to_bytes();
        assert_eq!(bytes, bytes2, "re-serialized bytes must be identical");
    }

    #[test]
    fn test_from_bytes_rejects_bad_magic() {
        let mut bytes = GsixHeaderV2::new(0, 0).to_bytes();
        bytes[0] = 0xFF; // corrupt magic
        assert!(GsixHeaderV2::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_from_bytes_rejects_bad_version() {
        let header = GsixHeaderV2::new(0, 0);
        let mut bytes = header.to_bytes();
        // Set version to 99 and recompute CRC so magic/CRC pass but version fails
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        // Don't recompute CRC — checksum will also fail, but version check happens first
        // Actually recompute so we test version check specifically
        let crc = crc32fast::hash(&bytes[..CHECKSUM_OFFSET]);
        bytes[CHECKSUM_OFFSET..CHECKSUM_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
        let err = GsixHeaderV2::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("version"), "error should mention version: {err}");
    }

    #[test]
    fn test_from_bytes_rejects_bad_checksum() {
        let mut bytes = GsixHeaderV2::new(10, 0x1234).to_bytes();
        // Corrupt a data byte (entry_count) without updating checksum
        bytes[8] = 0xFF;
        let err = GsixHeaderV2::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("checksum"), "error should mention checksum: {err}");
    }

    #[test]
    fn test_from_bytes_rejects_too_small() {
        let small = [0u8; 100];
        assert!(GsixHeaderV2::from_bytes(&small).is_err());
    }

    #[test]
    fn test_new_sets_defaults() {
        let header = GsixHeaderV2::new(5, 0xABC);
        assert_eq!(header.magic, INDEX_MAGIC);
        assert_eq!(header.version, INDEX_VERSION_V2);
        assert_eq!(header.entry_count, 5);
        assert_eq!(header.root_hash, 0xABC);
        assert_eq!(header.entry_capacity, 5);
        assert_eq!(header.flags, 0);
        assert!(header.saved_at > 0, "saved_at should be set to current time");
    }

    #[test]
    fn test_save_v2_writes_valid_file() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let idx_path = dir.path().join("test.idx");

        // Create test entries
        let mut entries = Vec::new();
        for i in 0..5 {
            let mut entry = GpuPathEntry::new();
            let path = format!("/tmp/test_{}.rs", i);
            entry.set_path(path.as_bytes());
            entry.flags = i as u32;
            entry.set_size(1024 * (i as u64 + 1));
            entry.mtime = 1700000000 + i as u32;
            entries.push(entry);
        }

        let result = save_v2(&entries, 0xABCD1234, &idx_path, 9999, 0xFACE);
        assert!(result.is_ok(), "save_v2 should succeed");

        let returned_path = result.unwrap();
        assert_eq!(returned_path, idx_path);
        assert!(idx_path.exists(), "index file should exist");

        // Verify no temp file remains
        let tmp_path = idx_path.with_extension("idx.tmp");
        assert!(!tmp_path.exists(), "temp file should not exist after rename");

        // Read file and validate header
        let data = std::fs::read(&idx_path).expect("Failed to read index file");
        let expected_len = HEADER_SIZE_V2 + entries.len() * 256;
        assert_eq!(data.len(), expected_len, "file size = header + entries");

        // Validate header via from_bytes
        let header = GsixHeaderV2::from_bytes(&data).expect("header should parse");
        assert_eq!(header.entry_count, 5);
        assert_eq!(header.root_hash, 0xABCD1234);
        assert_eq!(header.last_fsevents_id, 9999);
        assert_eq!(header.exclude_hash, 0xFACE);
        assert_eq!(header.entry_capacity, 5);
        assert!(header.saved_at > 0);

        // Validate entries are byte-identical
        for (i, orig) in entries.iter().enumerate() {
            let offset = HEADER_SIZE_V2 + i * 256;
            let orig_bytes: &[u8; 256] =
                unsafe { &*(orig as *const GpuPathEntry as *const [u8; 256]) };
            assert_eq!(
                &data[offset..offset + 256],
                orig_bytes,
                "entry {} should be byte-identical",
                i
            );
        }
    }

    #[test]
    fn test_save_v2_empty_entries() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let idx_path = dir.path().join("empty.idx");

        let result = save_v2(&[], 0, &idx_path, 0, 0);
        assert!(result.is_ok());

        let data = std::fs::read(&idx_path).unwrap();
        assert_eq!(data.len(), HEADER_SIZE_V2, "empty index = header only");

        let header = GsixHeaderV2::from_bytes(&data).unwrap();
        assert_eq!(header.entry_count, 0);
    }

    #[test]
    fn test_save_v2_creates_parent_dirs() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let idx_path = dir.path().join("nested").join("deep").join("test.idx");

        let entry = GpuPathEntry::new();
        let result = save_v2(&[entry], 1, &idx_path, 0, 0);
        assert!(result.is_ok());
        assert!(idx_path.exists());
    }

    #[test]
    fn test_save_v2_atomic_replaces_existing() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let idx_path = dir.path().join("replace.idx");

        // Write initial file
        let mut entry1 = GpuPathEntry::new();
        entry1.set_path(b"/first");
        save_v2(&[entry1], 1, &idx_path, 100, 0).unwrap();

        // Overwrite with different data
        let mut entry2 = GpuPathEntry::new();
        entry2.set_path(b"/second");
        save_v2(&[entry2], 2, &idx_path, 200, 0).unwrap();

        // Verify new data
        let data = std::fs::read(&idx_path).unwrap();
        let header = GsixHeaderV2::from_bytes(&data).unwrap();
        assert_eq!(header.root_hash, 2);
        assert_eq!(header.last_fsevents_id, 200);

        // Verify entry path is "/second"
        let entry_bytes = &data[HEADER_SIZE_V2..HEADER_SIZE_V2 + 256];
        let path_len = u32::from_le_bytes(entry_bytes[224..228].try_into().unwrap()) as usize;
        assert_eq!(&entry_bytes[..path_len], b"/second");
    }

    #[test]
    fn test_save_load_v2_roundtrip() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let idx_path = dir.path().join("roundtrip.idx");

        // Create diverse test entries
        let mut entries = Vec::new();
        for i in 0..10 {
            let mut entry = GpuPathEntry::new();
            let path = format!("/users/dev/project/src/file_{}.rs", i);
            entry.set_path(path.as_bytes());
            entry.flags = i as u32;
            entry.set_size(4096 * (i as u64 + 1));
            entry.mtime = 1700000000 + i as u32 * 100;
            entries.push(entry);
        }

        let root_hash = 0xDEAD_BEEF;
        let fsevents_id = 42424242u64;
        let exclude_hash = 0xCAFE_BABE;

        save_v2(&entries, root_hash, &idx_path, fsevents_id, exclude_hash)
            .expect("save_v2 should succeed");

        // Load and verify
        let (header, loaded) = load_v2(&idx_path).expect("load_v2 should succeed");

        // Verify header fields
        assert_eq!(header.magic, INDEX_MAGIC);
        assert_eq!(header.version, INDEX_VERSION_V2);
        assert_eq!(header.entry_count, 10);
        assert_eq!(header.root_hash, root_hash);
        assert_eq!(header.last_fsevents_id, fsevents_id);
        assert_eq!(header.exclude_hash, exclude_hash);
        assert_eq!(header.entry_capacity, 10);

        // Verify entry count matches
        assert_eq!(loaded.len(), entries.len());

        // Verify each entry is byte-identical
        for (i, (orig, loaded_entry)) in entries.iter().zip(loaded.iter()).enumerate() {
            let orig_bytes: &[u8; 256] =
                unsafe { &*(orig as *const GpuPathEntry as *const [u8; 256]) };
            let loaded_bytes: &[u8; 256] =
                unsafe { &*(loaded_entry as *const GpuPathEntry as *const [u8; 256]) };
            assert_eq!(
                orig_bytes, loaded_bytes,
                "entry {} should be byte-identical after roundtrip",
                i
            );
        }
    }

    #[test]
    fn test_entry_count_matches() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let idx_path = dir.path().join("count_check.idx");

        let counts = [0, 1, 7, 50];
        for &count in &counts {
            let mut entries = Vec::new();
            for i in 0..count {
                let mut entry = GpuPathEntry::new();
                entry.set_path(format!("/file_{}", i).as_bytes());
                entries.push(entry);
            }
            save_v2(&entries, 0xAA, &idx_path, 0, 0).expect("save should succeed");

            let (header, loaded) = load_v2(&idx_path).expect("load should succeed");
            assert_eq!(
                header.entry_count, count as u32,
                "header entry_count should match written entries for count={}",
                count
            );
            assert_eq!(
                loaded.len(), count,
                "loaded entries len should match for count={}",
                count
            );
        }
    }

    #[test]
    fn test_load_v2_empty_entries() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let idx_path = dir.path().join("empty_load.idx");

        save_v2(&[], 0, &idx_path, 0, 0).expect("save empty should succeed");

        let (header, loaded) = load_v2(&idx_path).expect("load empty should succeed");
        assert_eq!(header.entry_count, 0);
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_load_v2_rejects_truncated_entries() {
        let dir = tempfile::TempDir::new().expect("Failed to create temp dir");
        let idx_path = dir.path().join("truncated.idx");

        // Create valid file with 5 entries
        let mut entries = Vec::new();
        for i in 0..5 {
            let mut entry = GpuPathEntry::new();
            entry.set_path(format!("/file_{}", i).as_bytes());
            entries.push(entry);
        }
        save_v2(&entries, 0, &idx_path, 0, 0).unwrap();

        // Truncate file so it claims 5 entries but doesn't have space for them all
        let data = std::fs::read(&idx_path).unwrap();
        // Keep header + only 3 entries (truncate 2 entries worth)
        let truncated = &data[..HEADER_SIZE_V2 + 3 * 256];
        std::fs::write(&idx_path, truncated).unwrap();

        let err = load_v2(&idx_path);
        assert!(err.is_err(), "should reject truncated file");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("too short"), "error should mention too short: {msg}");
    }

    #[test]
    fn test_load_v2_file_not_found() {
        let result = load_v2(Path::new("/nonexistent/path/test.idx"));
        assert!(result.is_err());
    }

    // =========================================================================
    // Version detection tests
    // =========================================================================

    #[test]
    fn test_detect_version_v1() {
        // Build a minimal v1-style header: same magic, version=1
        let mut data = [0u8; 64];
        data[0..4].copy_from_slice(&INDEX_MAGIC.to_le_bytes());
        data[4..8].copy_from_slice(&INDEX_VERSION_V1.to_le_bytes());
        assert_eq!(detect_version(&data).unwrap(), 1);
    }

    #[test]
    fn test_detect_version_v2() {
        let header = GsixHeaderV2::new(0, 0);
        let bytes = header.to_bytes();
        assert_eq!(detect_version(&bytes).unwrap(), 2);
    }

    #[test]
    fn test_detect_version_unknown_magic() {
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        data[4..8].copy_from_slice(&2u32.to_le_bytes());
        let err = detect_version(&data);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("bad magic"));
    }

    #[test]
    fn test_detect_version_unknown_version() {
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&INDEX_MAGIC.to_le_bytes());
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = detect_version(&data);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("unknown version"));
    }

    #[test]
    fn test_detect_version_too_small() {
        let data = [0u8; 4]; // too small, need at least 8
        let err = detect_version(&data);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("too small"));
    }

    // =========================================================================
    // V1 migration tests
    // =========================================================================

    #[test]
    fn test_load_with_migration_v1_signals_rebuild() {
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("old.idx");

        // Write a fake v1 file: magic + version=1 + padding to 64 bytes
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&INDEX_MAGIC.to_le_bytes());
        data[4..8].copy_from_slice(&INDEX_VERSION_V1.to_le_bytes());
        std::fs::write(&idx_path, &data).unwrap();

        let result = load_with_migration(&idx_path);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("v1 index detected"), "error: {msg}");
        assert!(msg.contains("rebuild required"), "error: {msg}");

        // v1 file should be deleted
        assert!(!idx_path.exists(), "v1 file should have been deleted");
    }

    #[test]
    fn test_load_with_migration_v2_loads_normally() {
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("v2.idx");

        let mut entry = GpuPathEntry::new();
        entry.set_path(b"/test/file.rs");
        save_v2(&[entry], 0xABC, &idx_path, 42, 0).unwrap();

        let (header, entries) = load_with_migration(&idx_path).unwrap();
        assert_eq!(header.version, INDEX_VERSION_V2);
        assert_eq!(entries.len(), 1);
    }

    // =========================================================================
    // Cleanup tests
    // =========================================================================

    #[test]
    fn test_cleanup_v1_indexes_removes_non_global() {
        let dir = tempfile::TempDir::new().unwrap();
        let index_dir = dir.path();

        // Create various .idx files
        std::fs::write(index_dir.join("global.idx"), b"keep").unwrap();
        std::fs::write(index_dir.join("abc123.idx"), b"remove1").unwrap();
        std::fs::write(index_dir.join("def456.idx"), b"remove2").unwrap();
        std::fs::write(index_dir.join("notes.txt"), b"keep-non-idx").unwrap();

        cleanup_v1_indexes(index_dir).unwrap();

        // global.idx should remain
        assert!(index_dir.join("global.idx").exists(), "global.idx should survive");
        // non-global .idx should be deleted
        assert!(!index_dir.join("abc123.idx").exists(), "abc123.idx should be removed");
        assert!(!index_dir.join("def456.idx").exists(), "def456.idx should be removed");
        // non-.idx files untouched
        assert!(index_dir.join("notes.txt").exists(), "notes.txt should survive");
        // marker created
        assert!(index_dir.join(".v2-migrated").exists(), "marker should exist");
    }

    #[test]
    fn test_cleanup_v1_indexes_nonexistent_dir() {
        let result = cleanup_v1_indexes(Path::new("/nonexistent/dir"));
        assert!(result.is_ok(), "should be no-op for missing dir");
    }

    #[test]
    fn test_cleanup_v1_indexes_empty_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        cleanup_v1_indexes(dir.path()).unwrap();
        assert!(dir.path().join(".v2-migrated").exists(), "marker should be created even in empty dir");
    }

    // =========================================================================
    // Corrupt file handling tests
    // =========================================================================

    #[test]
    fn test_corrupt_entry_count_overflow() {
        // entry_count claims more entries than the file can hold
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("overflow.idx");

        // Save a valid file with 1 entry
        let mut entry = GpuPathEntry::new();
        entry.set_path(b"/test");
        save_v2(&[entry], 0xAA, &idx_path, 0, 0).unwrap();

        // Tamper: set entry_count to 999999 and recompute checksum
        let mut data = std::fs::read(&idx_path).unwrap();
        data[8..12].copy_from_slice(&999999u32.to_le_bytes());
        let crc = crc32fast::hash(&data[..CHECKSUM_OFFSET]);
        data[CHECKSUM_OFFSET..CHECKSUM_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
        std::fs::write(&idx_path, &data).unwrap();

        let result = load_v2(&idx_path);
        assert!(result.is_err(), "should reject entry_count overflow");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("too short"), "error should mention too short: {msg}");
    }

    #[test]
    fn test_corrupt_zero_byte_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("zero.idx");

        std::fs::write(&idx_path, b"").unwrap();

        let result = load_v2(&idx_path);
        assert!(result.is_err(), "should reject 0-byte file");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("too small"), "error should mention too small: {msg}");
    }

    #[test]
    fn test_corrupt_header_only_zero_entries() {
        // A valid header with entry_count=0 and no entry data should load OK
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("header_only.idx");

        save_v2(&[], 0xBB, &idx_path, 0, 0).unwrap();

        let (header, entries) = load_v2(&idx_path).expect("header-only with 0 entries should load");
        assert_eq!(header.entry_count, 0);
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_corrupt_partial_entry() {
        // Header claims 1 entry but only partial entry bytes present
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("partial.idx");

        // Save valid file with 1 entry
        let mut entry = GpuPathEntry::new();
        entry.set_path(b"/partial");
        save_v2(&[entry], 0xCC, &idx_path, 0, 0).unwrap();

        // Truncate: keep header + only 100 bytes of the 256-byte entry
        let data = std::fs::read(&idx_path).unwrap();
        let truncated = &data[..HEADER_SIZE_V2 + 100];
        std::fs::write(&idx_path, truncated).unwrap();

        let result = load_v2(&idx_path);
        assert!(result.is_err(), "should reject partial entry");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("too short"), "error should mention too short: {msg}");
    }

    #[test]
    fn test_corrupt_random_bytes() {
        // Random 1024 bytes should produce Err, never panic
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("random.idx");

        // Use a simple deterministic pseudo-random sequence
        let mut random_data = vec![0u8; 1024];
        let mut seed: u32 = 0xDEAD_BEEF;
        for byte in random_data.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *byte = (seed >> 16) as u8;
        }
        std::fs::write(&idx_path, &random_data).unwrap();

        let result = load_v2(&idx_path);
        // Must not panic — just return an error
        assert!(result.is_err(), "random bytes should produce Err");
    }

    #[test]
    fn test_corrupt_wrong_endianness() {
        // Big-endian magic: "XISG" instead of "GSIX"
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("big_endian.idx");

        let mut data = vec![0u8; HEADER_SIZE_V2];
        // Write magic in big-endian (swap bytes)
        data[0..4].copy_from_slice(&INDEX_MAGIC.to_be_bytes());
        data[4..8].copy_from_slice(&INDEX_VERSION_V2.to_le_bytes());
        std::fs::write(&idx_path, &data).unwrap();

        let result = load_v2(&idx_path);
        assert!(result.is_err(), "should reject big-endian magic");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("bad magic"), "error should mention bad magic: {msg}");
    }

    #[test]
    fn test_corrupt_trailing_garbage() {
        // Valid file + extra trailing bytes should still load successfully
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("trailing.idx");

        let mut entry = GpuPathEntry::new();
        entry.set_path(b"/trailing_test");
        save_v2(&[entry], 0xDD, &idx_path, 42, 0).unwrap();

        // Append 512 bytes of garbage
        let mut data = std::fs::read(&idx_path).unwrap();
        data.extend_from_slice(&[0xAB; 512]);
        std::fs::write(&idx_path, &data).unwrap();

        // Should still load — extra bytes are ignored
        let (header, entries) = load_v2(&idx_path).expect("trailing garbage should be tolerated");
        assert_eq!(header.entry_count, 1);
        assert_eq!(entries.len(), 1);
        assert_eq!(header.root_hash, 0xDD);
    }

    #[test]
    fn test_corrupt_all_zeros() {
        // All-zero header: magic will be 0x00000000 which is not INDEX_MAGIC
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("all_zeros.idx");

        let data = vec![0u8; HEADER_SIZE_V2];
        std::fs::write(&idx_path, &data).unwrap();

        let result = load_v2(&idx_path);
        assert!(result.is_err(), "should reject all-zeros header");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("bad magic"), "error should mention bad magic: {msg}");
    }

    // =========================================================================
    // Staleness detection tests
    // =========================================================================

    #[test]
    fn test_is_stale_fresh() {
        // A header with saved_at = now should NOT be stale
        let header = GsixHeaderV2::new(100, 0xABC);
        // GsixHeaderV2::new() sets saved_at to current time
        assert!(
            !is_stale(&header),
            "freshly created header should not be stale"
        );
        assert!(
            !is_stale_with_age(&header, 60),
            "freshly created header should not be stale even with 60s max age"
        );
    }

    #[test]
    fn test_is_stale_expired() {
        // A header with saved_at = 2 hours ago should be stale (DEFAULT_MAX_AGE = 1 hour)
        let mut header = GsixHeaderV2::new(100, 0xABC);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        header.saved_at = now - 7200; // 2 hours ago

        assert!(
            is_stale(&header),
            "header saved 2 hours ago should be stale (max age = 1 hour)"
        );
        assert!(
            is_stale_with_age(&header, 3600),
            "header saved 2 hours ago should be stale with 3600s max age"
        );
        // But not stale with a 3-hour max age
        assert!(
            !is_stale_with_age(&header, 10800),
            "header saved 2 hours ago should NOT be stale with 3-hour max age"
        );
    }

    #[test]
    fn test_is_stale_saved_at_zero() {
        // saved_at == 0 means never saved; handled by initial build, not staleness
        let mut header = GsixHeaderV2::new(0, 0);
        header.saved_at = 0;
        assert!(
            !is_stale(&header),
            "saved_at=0 should not be considered stale (handled by initial build)"
        );
    }

    #[test]
    fn test_is_stale_boundary() {
        // At exactly the max age, should NOT be stale (must be strictly older)
        let mut header = GsixHeaderV2::new(50, 0);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        header.saved_at = now - DEFAULT_MAX_AGE_SECS;

        // At exactly the boundary (age == max_age), not stale (> not >=)
        assert!(
            !is_stale(&header),
            "header at exactly max age boundary should not be stale"
        );

        // One second past the boundary: stale
        header.saved_at = now - DEFAULT_MAX_AGE_SECS - 1;
        assert!(
            is_stale(&header),
            "header one second past max age should be stale"
        );
    }

    #[test]
    fn test_is_stale_custom_age() {
        let mut header = GsixHeaderV2::new(10, 0);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // 5 minutes ago with 10-minute max age -> not stale
        header.saved_at = now - 300;
        assert!(!is_stale_with_age(&header, 600));

        // 5 minutes ago with 3-minute max age -> stale
        assert!(is_stale_with_age(&header, 180));
    }
}
