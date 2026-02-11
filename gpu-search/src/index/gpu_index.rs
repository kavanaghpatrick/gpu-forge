//! GPU-Resident Filesystem Index
//!
//! Builds an in-memory index of filesystem paths as `GpuPathEntry` structs (256B each),
//! then loads them into a Metal GPU buffer for kernel-side path filtering.
//!
//! Ported from rust-experiment/src/gpu_os/gpu_index.rs (867 lines), simplified to
//! use the existing GpuPathEntry type from gpu/types.rs and objc2-metal bindings.
//!
//! Loading methods (mmap, GPU-direct) are in separate modules (cache.rs, gpu_loader.rs).
//! This module handles: scan filesystem -> build Vec<GpuPathEntry> -> upload to GPU.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::gpu::types::{path_flags, GpuPathEntry, GPU_PATH_MAX_LEN};

// ============================================================================
// GpuResidentIndex
// ============================================================================

/// A filesystem index that can be loaded into GPU memory for kernel-side filtering.
///
/// Entries are 256-byte `GpuPathEntry` structs matching the Metal shader layout.
/// After building, call `to_gpu_buffer()` to upload to a Metal buffer.
pub struct GpuResidentIndex {
    /// All indexed path entries (CPU-side).
    entries: Vec<GpuPathEntry>,
    /// Optional Metal buffer (populated after `to_gpu_buffer()`).
    gpu_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl GpuResidentIndex {
    /// Build an index by scanning the given paths.
    ///
    /// Each path is stat'd for metadata (size, mtime, is_dir, is_hidden).
    /// Paths longer than 224 bytes are silently skipped.
    /// Unreadable paths are silently skipped.
    ///
    /// # Example
    /// ```ignore
    /// let paths: Vec<PathBuf> = walkdir(root).collect();
    /// let index = GpuResidentIndex::build_from_paths(&paths);
    /// assert!(index.entry_count() > 0);
    /// ```
    pub fn build_from_paths(paths: &[PathBuf]) -> Self {
        let mut entries = Vec::with_capacity(paths.len());

        for path in paths {
            if let Some(entry) = Self::make_entry(path) {
                entries.push(entry);
            }
        }

        Self {
            entries,
            gpu_buffer: None,
        }
    }

    /// Build an index by recursively scanning a directory.
    ///
    /// Skips common uninteresting directories (.git, node_modules, target, __pycache__).
    /// Silently skips unreadable entries.
    pub fn build_from_directory(root: &Path) -> std::io::Result<Self> {
        let mut entries = Vec::new();
        Self::scan_recursive(root, &mut entries)?;

        Ok(Self {
            entries,
            gpu_buffer: None,
        })
    }

    /// Recursively scan a directory tree, building GpuPathEntry for each item.
    fn scan_recursive(dir: &Path, entries: &mut Vec<GpuPathEntry>) -> std::io::Result<()> {
        let read_dir = match std::fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(_) => return Ok(()), // Skip unreadable directories
        };

        for entry_result in read_dir {
            let dir_entry = match entry_result {
                Ok(e) => e,
                Err(_) => continue,
            };

            let entry_path = dir_entry.path();
            let path_str = entry_path.to_string_lossy();

            // Skip paths that exceed GPU_PATH_MAX_LEN
            if path_str.len() > GPU_PATH_MAX_LEN {
                continue;
            }

            let metadata = match dir_entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };

            let is_dir = metadata.is_dir();
            let file_size = if is_dir { 0 } else { metadata.len() };
            let mtime = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as u32)
                .unwrap_or(0);

            // Determine flags
            let mut flags: u32 = 0;
            if is_dir {
                flags |= path_flags::IS_DIR;
            }
            if metadata.file_type().is_symlink() {
                flags |= path_flags::IS_SYMLINK;
            }
            // Check for hidden (starts with '.' after last '/')
            let file_name = dir_entry.file_name();
            let name_str = file_name.to_string_lossy();
            if name_str.starts_with('.') {
                flags |= path_flags::IS_HIDDEN;
            }

            let mut entry = GpuPathEntry::new();
            entry.set_path(path_str.as_bytes());
            entry.flags = flags;
            entry.set_size(file_size);
            entry.mtime = mtime;

            entries.push(entry);

            // Recurse into directories, skipping common noise
            if is_dir {
                let skip = matches!(
                    name_str.as_ref(),
                    ".git" | ".hg" | "node_modules" | "target" | "__pycache__" | ".build"
                );
                if !skip && !name_str.starts_with('.') {
                    let _ = Self::scan_recursive(&entry_path, entries);
                }
            }
        }

        Ok(())
    }

    /// Create a GpuPathEntry from a filesystem path with metadata.
    fn make_entry(path: &Path) -> Option<GpuPathEntry> {
        let path_str = path.to_str()?;
        if path_str.len() > GPU_PATH_MAX_LEN {
            return None;
        }

        let metadata = std::fs::metadata(path).ok()?;
        let is_dir = metadata.is_dir();
        let file_size = if is_dir { 0 } else { metadata.len() };
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);

        // Determine flags
        let mut flags: u32 = 0;
        if is_dir {
            flags |= path_flags::IS_DIR;
        }
        if metadata.file_type().is_symlink() {
            flags |= path_flags::IS_SYMLINK;
        }
        // Check for hidden file/dir
        if let Some(name) = path.file_name() {
            if name.to_string_lossy().starts_with('.') {
                flags |= path_flags::IS_HIDDEN;
            }
        }

        let mut entry = GpuPathEntry::new();
        entry.set_path(path_str.as_bytes());
        entry.flags = flags;
        entry.set_size(file_size);
        entry.mtime = mtime;

        Some(entry)
    }

    /// Upload entries into a Metal GPU buffer (StorageModeShared).
    ///
    /// Returns a reference to the buffer. The buffer is cached; subsequent
    /// calls return the same buffer unless entries change.
    ///
    /// The buffer contains `entry_count * 256` bytes of packed GpuPathEntry structs.
    pub fn to_gpu_buffer(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        if self.gpu_buffer.is_none() {
            let buffer = self.create_gpu_buffer(device);
            self.gpu_buffer = Some(buffer);
        }
        self.gpu_buffer.as_ref().unwrap()
    }

    /// Create a new Metal buffer from entries.
    fn create_gpu_buffer(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let byte_count = self.entries.len() * GpuPathEntry::SIZE;

        if byte_count == 0 {
            // Empty index: allocate minimal buffer
            return device
                .newBufferWithLength_options(GpuPathEntry::SIZE, MTLResourceOptions::StorageModeShared)
                .expect("Failed to allocate empty GPU index buffer");
        }

        // SAFETY: GpuPathEntry is #[repr(C)] and exactly 256 bytes.
        // The Vec<GpuPathEntry> is a contiguous array of these structs.
        let ptr = std::ptr::NonNull::new(self.entries.as_ptr() as *mut _)
            .expect("entries vec pointer is null");

        unsafe {
            device
                .newBufferWithBytes_length_options(ptr, byte_count, MTLResourceOptions::StorageModeShared)
                .expect("Failed to create GPU index buffer")
        }
    }

    // ========================================================================
    // Query methods
    // ========================================================================

    /// Number of entries in the index.
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Get an entry by index.
    #[inline]
    pub fn get_entry(&self, idx: usize) -> Option<&GpuPathEntry> {
        self.entries.get(idx)
    }

    /// Find entries whose path contains the given substring pattern.
    ///
    /// Case-sensitive byte-level match. Returns indices of matching entries.
    pub fn find_by_name(&self, pattern: &str) -> Vec<usize> {
        let pattern_bytes = pattern.as_bytes();
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                let path_len = entry.path_len as usize;
                let path_bytes = &entry.path[..path_len];
                // Substring search
                path_bytes
                    .windows(pattern_bytes.len())
                    .any(|window| window == pattern_bytes)
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get all entries as a slice.
    #[inline]
    pub fn entries(&self) -> &[GpuPathEntry] {
        &self.entries
    }

    /// Get the cached GPU buffer (if already uploaded).
    #[inline]
    pub fn gpu_buffer(&self) -> Option<&Retained<ProtocolObject<dyn MTLBuffer>>> {
        self.gpu_buffer.as_ref()
    }

    /// Get the path string from an entry.
    pub fn entry_path_str(entry: &GpuPathEntry) -> &str {
        let len = entry.path_len as usize;
        std::str::from_utf8(&entry.path[..len]).unwrap_or("")
    }

    /// Check if an entry is a directory.
    #[inline]
    pub fn is_dir(entry: &GpuPathEntry) -> bool {
        entry.flags & path_flags::IS_DIR != 0
    }

    /// Check if an entry is hidden.
    #[inline]
    pub fn is_hidden(entry: &GpuPathEntry) -> bool {
        entry.flags & path_flags::IS_HIDDEN != 0
    }

    /// Get all file entries (non-directories).
    pub fn files(&self) -> Vec<PathBuf> {
        self.entries
            .iter()
            .filter(|e| e.flags & path_flags::IS_DIR == 0)
            .map(|e| PathBuf::from(Self::entry_path_str(e)))
            .collect()
    }

    /// Get files matching a given extension (e.g. "rs", "metal").
    pub fn files_with_extension(&self, ext: &str) -> Vec<PathBuf> {
        let suffix = format!(".{}", ext);
        self.entries
            .iter()
            .filter(|e| e.flags & path_flags::IS_DIR == 0)
            .filter(|e| Self::entry_path_str(e).ends_with(&suffix))
            .map(|e| PathBuf::from(Self::entry_path_str(e)))
            .collect()
    }

    /// Total size (bytes) of all file entries in the index.
    pub fn total_file_size(&self) -> u64 {
        self.entries
            .iter()
            .filter(|e| e.flags & path_flags::IS_DIR == 0)
            .map(|e| e.size())
            .sum()
    }

    /// GPU buffer byte size (entry_count * 256).
    pub fn buffer_size_bytes(&self) -> usize {
        self.entries.len() * GpuPathEntry::SIZE
    }

    /// Create an index directly from pre-built entries (e.g., loaded from cache).
    ///
    /// Unlike `build_from_paths`, this does NOT stat the filesystem.
    /// The entries are used as-is with no GPU buffer (call `to_gpu_buffer()` if needed).
    pub fn from_entries(entries: Vec<GpuPathEntry>) -> Self {
        Self {
            entries,
            gpu_buffer: None,
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
    use tempfile::TempDir;

    /// Create a temp directory with known files for testing.
    fn make_test_dir() -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // Create some files
        fs::write(dir.path().join("hello.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("world.txt"), "hello world").unwrap();
        fs::write(dir.path().join(".hidden"), "secret").unwrap();

        // Create a subdirectory with files
        let sub = dir.path().join("subdir");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("nested.rs"), "mod test;").unwrap();

        dir
    }

    #[test]
    fn test_gpu_index() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        // Build index from gpu-search/src/ directory
        let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let index = GpuResidentIndex::build_from_directory(&src_dir)
            .expect("Failed to build index from src/");

        // Should have some entries (at least the known module files)
        assert!(
            index.entry_count() > 0,
            "Index should have entries from src/"
        );

        // Verify entry count roughly matches filesystem
        let fs_count = walkdir_count(&src_dir);
        // Allow some tolerance (hidden files, unreadable, etc.)
        let diff = (index.entry_count() as isize - fs_count as isize).unsigned_abs();
        assert!(
            diff <= fs_count / 5 + 2, // within 20% + 2
            "Entry count {} too far from filesystem count {}",
            index.entry_count(),
            fs_count
        );

        // Snapshot entry count and entries before mutable borrow for GPU upload
        let entry_count = index.entry_count();
        let entries_snapshot: Vec<GpuPathEntry> = index.entries().to_vec();

        // Load into GPU buffer
        let mut index = index;
        let buffer = index.to_gpu_buffer(&device);

        // Buffer should be large enough for all entries
        let expected_size = entry_count * GpuPathEntry::SIZE;
        assert!(
            buffer.length() >= expected_size,
            "GPU buffer too small: {} < {}",
            buffer.length(),
            expected_size
        );

        // Verify buffer contents match CPU-side entries
        unsafe {
            let buf_ptr = buffer.contents().as_ptr() as *const GpuPathEntry;
            for i in 0..entry_count {
                let gpu_entry = &*buf_ptr.add(i);
                let cpu_entry = &entries_snapshot[i];
                assert_eq!(
                    gpu_entry.path_len, cpu_entry.path_len,
                    "path_len mismatch at entry {}",
                    i
                );
                assert_eq!(
                    &gpu_entry.path[..gpu_entry.path_len as usize],
                    &cpu_entry.path[..cpu_entry.path_len as usize],
                    "path bytes mismatch at entry {}",
                    i
                );
            }
        }

        println!(
            "GPU index: {} entries, {} bytes in GPU buffer",
            entry_count,
            buffer.length()
        );
    }

    #[test]
    fn test_gpu_index_build_from_paths() {
        let dir = make_test_dir();

        let paths: Vec<PathBuf> = vec![
            dir.path().join("hello.rs"),
            dir.path().join("world.txt"),
            dir.path().join(".hidden"),
            dir.path().join("subdir"),
        ];

        let index = GpuResidentIndex::build_from_paths(&paths);
        assert_eq!(index.entry_count(), 4);

        // Check a known entry
        let entry = index.get_entry(0).unwrap();
        let path_str = GpuResidentIndex::entry_path_str(entry);
        assert!(path_str.ends_with("hello.rs"));
        assert_eq!(entry.flags & path_flags::IS_DIR, 0); // not a dir
    }

    #[test]
    fn test_gpu_index_build_from_directory() {
        let dir = make_test_dir();

        let index = GpuResidentIndex::build_from_directory(dir.path())
            .expect("Failed to build index");

        // Should find: hello.rs, world.txt, .hidden, subdir, subdir/nested.rs
        // .hidden is found but subdir entries depend on hidden dir skip logic
        assert!(
            index.entry_count() >= 3,
            "Should have at least 3 entries, got {}",
            index.entry_count()
        );

        // Verify all entries have valid path_len
        for i in 0..index.entry_count() {
            let entry = index.get_entry(i).unwrap();
            assert!(entry.path_len > 0, "Entry {} has zero path_len", i);
            assert!(
                (entry.path_len as usize) <= GPU_PATH_MAX_LEN,
                "Entry {} path_len {} exceeds max",
                i,
                entry.path_len
            );
        }
    }

    #[test]
    fn test_gpu_index_find_by_name() {
        let dir = make_test_dir();

        let paths: Vec<PathBuf> = vec![
            dir.path().join("hello.rs"),
            dir.path().join("world.txt"),
            dir.path().join(".hidden"),
        ];

        let index = GpuResidentIndex::build_from_paths(&paths);

        // Find .rs files
        let rs_matches = index.find_by_name(".rs");
        assert_eq!(rs_matches.len(), 1);
        let entry = index.get_entry(rs_matches[0]).unwrap();
        assert!(GpuResidentIndex::entry_path_str(entry).ends_with("hello.rs"));

        // Find no match
        let no_match = index.find_by_name("nonexistent");
        assert!(no_match.is_empty());
    }

    #[test]
    fn test_gpu_index_flags() {
        let dir = make_test_dir();

        let paths: Vec<PathBuf> = vec![
            dir.path().join("hello.rs"),
            dir.path().join(".hidden"),
            dir.path().join("subdir"),
        ];

        let index = GpuResidentIndex::build_from_paths(&paths);

        // hello.rs: regular file
        let e0 = index.get_entry(0).unwrap();
        assert!(!GpuResidentIndex::is_dir(e0));
        assert!(!GpuResidentIndex::is_hidden(e0));

        // .hidden: hidden file
        let e1 = index.get_entry(1).unwrap();
        assert!(GpuResidentIndex::is_hidden(e1));

        // subdir: directory
        let e2 = index.get_entry(2).unwrap();
        assert!(GpuResidentIndex::is_dir(e2));
    }

    #[test]
    fn test_gpu_index_files_with_extension() {
        let dir = make_test_dir();

        let paths: Vec<PathBuf> = vec![
            dir.path().join("hello.rs"),
            dir.path().join("world.txt"),
            dir.path().join(".hidden"),
        ];

        let index = GpuResidentIndex::build_from_paths(&paths);

        let rs_files = index.files_with_extension("rs");
        assert_eq!(rs_files.len(), 1);

        let txt_files = index.files_with_extension("txt");
        assert_eq!(txt_files.len(), 1);

        let metal_files = index.files_with_extension("metal");
        assert!(metal_files.is_empty());
    }

    #[test]
    fn test_gpu_index_empty() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        let mut index = GpuResidentIndex::build_from_paths(&[]);
        assert_eq!(index.entry_count(), 0);

        // Should still produce a valid (minimal) GPU buffer
        let buffer = index.to_gpu_buffer(&device);
        assert!(buffer.length() > 0);
    }

    #[test]
    fn test_gpu_index_size_tracking() {
        let dir = make_test_dir();

        let paths: Vec<PathBuf> = vec![
            dir.path().join("hello.rs"),  // "fn main() {}" = 12 bytes
            dir.path().join("world.txt"), // "hello world" = 11 bytes
        ];

        let index = GpuResidentIndex::build_from_paths(&paths);

        // Total file size should be sum of the two files
        let total = index.total_file_size();
        assert!(total > 0, "Total file size should be > 0");
        assert_eq!(total, 12 + 11, "Total file size should be 23 bytes");
    }

    /// Count entries in a directory recursively (for comparison).
    fn walkdir_count(dir: &Path) -> usize {
        let mut count = 0;
        if let Ok(rd) = fs::read_dir(dir) {
            for entry in rd.flatten() {
                count += 1;
                if entry.path().is_dir() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    // Mirror the skip logic
                    if !matches!(
                        name_str.as_ref(),
                        ".git" | ".hg" | "node_modules" | "target" | "__pycache__" | ".build"
                    ) && !name_str.starts_with('.')
                    {
                        count += walkdir_count(&entry.path());
                    }
                }
            }
        }
        count
    }
}
