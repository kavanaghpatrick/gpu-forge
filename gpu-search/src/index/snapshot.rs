//! IndexSnapshot: immutable mmap-anchored index with optional Metal buffer.
//!
//! Bundles an mmap'd GSIX v2 file with its validated header and an optional
//! Metal buffer created via `bytesNoCopy`. The mmap (`MmapBuffer`) owns the
//! memory, so the Metal buffer's pointer remains valid for the snapshot's
//! lifetime.
//!
//! # Lifetime Semantics
//!
//! ```text
//! IndexSnapshot
//!   |-- mmap: MmapBuffer          (owns the mapping, dropped LAST)
//!   |-- metal_buffer: Option<...> (GPU handle into mmap, dropped FIRST)
//!   |-- header: GsixHeaderV2      (copy of validated header)
//!   |-- entry_count: usize
//!   +-- fsevents_id: u64
//! ```
//!
//! Rust drops fields in declaration order, so `metal_buffer` is dropped
//! before `mmap`, ensuring the GPU buffer never outlives its backing memory.

use std::path::Path;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

use crate::gpu::types::GpuPathEntry;
use crate::index::cache::CacheError;
use crate::index::gsix_v2::{self, GsixHeaderV2, HEADER_SIZE_V2};
use crate::index::metal_buffer::create_gpu_buffer;
use crate::io::mmap::MmapBuffer;

/// An immutable snapshot of a GSIX v2 index file, backed by mmap with an
/// optional zero-copy Metal buffer for GPU dispatch.
///
/// Created via [`IndexSnapshot::from_file`]. The snapshot is immutable and
/// can be shared across threads via `Arc<IndexSnapshot>`.
///
/// # Drop Order
///
/// Fields are dropped in declaration order. `metal_buffer` is declared
/// before `mmap`, so the Metal buffer is released before the mmap is
/// unmapped. This guarantees the GPU buffer never references freed memory.
pub struct IndexSnapshot {
    // -- GPU buffer MUST be dropped before mmap --
    /// Metal buffer wrapping the mmap'd region (zero-copy via bytesNoCopy).
    /// `None` if no Metal device was provided or buffer creation failed.
    metal_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    // -- mmap keeps the memory alive --
    /// The underlying mmap buffer. Keeps the file mapping alive so that
    /// `metal_buffer` and `entries()` remain valid.
    mmap: MmapBuffer,

    /// Number of valid entries in the index.
    entry_count: usize,

    /// macOS FSEvents event ID stored in the header, used for resume.
    fsevents_id: u64,

    /// Copy of the validated GSIX v2 header.
    header: GsixHeaderV2,
}

// SAFETY: MmapBuffer is Send+Sync (read-only mmap). Retained<MTLBuffer> is
// Send+Sync on Apple platforms (Metal objects are thread-safe for read access).
// All other fields are plain data.
unsafe impl Send for IndexSnapshot {}
unsafe impl Sync for IndexSnapshot {}

impl IndexSnapshot {
    /// Load a GSIX v2 index file and optionally create a Metal buffer.
    ///
    /// 1. Memory-maps the file via `MmapBuffer::from_file`
    /// 2. Validates the v2 header (magic, version, CRC32)
    /// 3. Verifies the file is large enough for `entry_count` entries
    /// 4. If `device` is provided, creates a Metal buffer via `create_gpu_buffer`
    ///
    /// # Arguments
    /// * `path` - Path to the `.idx` file (GSIX v2 format)
    /// * `device` - Optional Metal device for GPU buffer creation
    ///
    /// # Errors
    /// Returns `CacheError` if the file is missing, corrupt, or wrong version.
    pub fn from_file(
        path: impl AsRef<Path>,
        device: Option<&ProtocolObject<dyn MTLDevice>>,
    ) -> Result<Self, CacheError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(CacheError::NotFound(path.to_path_buf()));
        }

        // Memory-map the file
        let mmap = MmapBuffer::from_file(path).map_err(|e| {
            CacheError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to mmap index file {}: {}", path.display(), e),
            ))
        })?;

        let data = mmap.as_slice();

        // Detect and validate version
        let version = gsix_v2::detect_version(data)?;
        if version == gsix_v2::INDEX_VERSION_V1 {
            return Err(CacheError::InvalidFormat(
                "v1 index detected, rebuild required".to_string(),
            ));
        }

        // Validate v2 header (magic, version, CRC32)
        let header = GsixHeaderV2::from_bytes(data)?;
        let entry_count = header.entry_count as usize;
        let fsevents_id = header.last_fsevents_id;

        // Verify file covers all entries
        let required_len = HEADER_SIZE_V2 + entry_count * GpuPathEntry::SIZE;
        if data.len() < required_len {
            return Err(CacheError::InvalidFormat(format!(
                "File too short: {} < {} ({} entries expected)",
                data.len(),
                required_len,
                entry_count
            )));
        }

        // Verify entry region is page-aligned for Metal bytesNoCopy.
        // Apple Silicon page size is 16384 bytes. The mmap base pointer is
        // page-aligned by the OS, and HEADER_SIZE_V2 is 16384, so entries
        // start at a page boundary. This debug_assert catches any violation.
        debug_assert!(
            (mmap.as_ptr() as usize + HEADER_SIZE_V2).is_multiple_of(16384),
            "Entry region must be 16KB page-aligned for Metal bytesNoCopy: ptr={:p}, offset={}",
            mmap.as_ptr(),
            HEADER_SIZE_V2
        );

        // Advise kernel about sequential access pattern
        mmap.advise_sequential();

        // Create Metal buffer if device provided
        let metal_buffer = device.and_then(|dev| {
            // SAFETY: mmap pointer is valid for the mmap lifetime, page-aligned from mmap,
            // and entry_count was validated against file size above.
            unsafe { create_gpu_buffer(dev, mmap.as_ptr(), mmap.mapped_len(), entry_count) }
        });

        Ok(Self {
            metal_buffer,
            mmap,
            entry_count,
            fsevents_id,
            header,
        })
    }

    /// Get a zero-copy slice of all entries from the mmap'd region.
    ///
    /// Entries start at byte offset `HEADER_SIZE_V2` (16384) in the file.
    /// The returned slice borrows from `self`; the mmap remains valid for
    /// the lifetime of the snapshot.
    pub fn entries(&self) -> &[GpuPathEntry] {
        if self.entry_count == 0 {
            return &[];
        }
        // SAFETY: We validated in from_file() that the mmap covers
        // HEADER_SIZE_V2 + entry_count * 256 bytes. GpuPathEntry is
        // #[repr(C)], 256 bytes, alignment 4. HEADER_SIZE_V2 is 16384
        // (page-aligned on Apple Silicon), so entries start at a page
        // boundary with sufficient alignment.
        unsafe {
            let entries_ptr = self.mmap.as_ptr().add(HEADER_SIZE_V2) as *const GpuPathEntry;
            std::slice::from_raw_parts(entries_ptr, self.entry_count)
        }
    }

    /// Get the number of entries in the index.
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Get the FSEvents event ID from the header, used for resume.
    #[inline]
    pub fn fsevents_id(&self) -> u64 {
        self.fsevents_id
    }

    /// Get a reference to the validated GSIX v2 header.
    #[inline]
    pub fn header(&self) -> &GsixHeaderV2 {
        &self.header
    }

    /// Get a reference to the Metal buffer, if available.
    #[inline]
    pub fn metal_buffer(&self) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        self.metal_buffer.as_deref()
    }

    /// Check whether this snapshot has a GPU-accessible Metal buffer.
    #[inline]
    pub fn has_metal_buffer(&self) -> bool {
        self.metal_buffer.is_some()
    }

    /// Get the mmap'd region length (page-aligned).
    #[inline]
    pub fn mapped_len(&self) -> usize {
        self.mmap.mapped_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::gsix_v2::save_v2;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)")
    }

    fn make_test_entries(count: usize) -> Vec<GpuPathEntry> {
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let mut entry = GpuPathEntry::new();
            entry.set_path(format!("/test/snapshot/file_{}.rs", i).as_bytes());
            entry.flags = i as u32;
            entry.set_size(1024 * (i as u64 + 1));
            entry.mtime = 1700000000 + i as u32;
            entries.push(entry);
        }
        entries
    }

    #[test]
    fn test_snapshot_from_file_without_device() {
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("test.idx");
        let entries = make_test_entries(10);

        save_v2(&entries, 0xABCD, &idx_path, 42, 0).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, None)
            .expect("snapshot should load without device");

        assert_eq!(snapshot.entry_count(), 10);
        assert_eq!(snapshot.fsevents_id(), 42);
        assert!(!snapshot.has_metal_buffer());
        assert_eq!(snapshot.header().root_hash, 0xABCD);
    }

    #[test]
    fn test_snapshot_from_file_with_device() {
        let device = get_device();
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("test.idx");
        let entries = make_test_entries(5);

        save_v2(&entries, 0xFF, &idx_path, 999, 0).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, Some(&device))
            .expect("snapshot should load with device");

        assert_eq!(snapshot.entry_count(), 5);
        assert_eq!(snapshot.fsevents_id(), 999);
        assert!(snapshot.has_metal_buffer(), "Metal buffer should be created");
    }

    #[test]
    fn test_snapshot_entries_match_original() {
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("verify.idx");
        let entries = make_test_entries(8);

        save_v2(&entries, 0xBEEF, &idx_path, 0, 0).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, None).unwrap();
        let loaded = snapshot.entries();

        assert_eq!(loaded.len(), entries.len());

        for (i, (orig, loaded_entry)) in entries.iter().zip(loaded.iter()).enumerate() {
            let orig_bytes: &[u8; 256] =
                unsafe { &*(orig as *const GpuPathEntry as *const [u8; 256]) };
            let loaded_bytes: &[u8; 256] =
                unsafe { &*(loaded_entry as *const GpuPathEntry as *const [u8; 256]) };
            assert_eq!(
                orig_bytes, loaded_bytes,
                "entry {} should be byte-identical",
                i
            );
        }
    }

    #[test]
    fn test_snapshot_empty_index() {
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("empty.idx");

        save_v2(&[], 0, &idx_path, 0, 0).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, None).unwrap();
        assert_eq!(snapshot.entry_count(), 0);
        assert!(snapshot.entries().is_empty());
    }

    #[test]
    fn test_snapshot_not_found() {
        let result = IndexSnapshot::from_file("/nonexistent/path.idx", None);
        assert!(result.is_err());
        match result {
            Err(CacheError::NotFound(_)) => {}
            other => panic!("Expected NotFound, got: {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn test_snapshot_rejects_corrupt_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let corrupt_path = dir.path().join("corrupt.idx");

        let garbage = vec![0xDE; 1024];
        std::fs::write(&corrupt_path, &garbage).unwrap();

        let result = IndexSnapshot::from_file(&corrupt_path, None);
        assert!(result.is_err());
        match result {
            Err(CacheError::InvalidFormat(_)) => {}
            other => panic!("Expected InvalidFormat, got: {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn test_snapshot_header_fields() {
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("header.idx");
        let entries = make_test_entries(3);

        save_v2(&entries, 0xCAFE, &idx_path, 12345, 0xFACE).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, None).unwrap();
        let header = snapshot.header();

        assert_eq!(header.root_hash, 0xCAFE);
        assert_eq!(header.last_fsevents_id, 12345);
        assert_eq!(header.exclude_hash, 0xFACE);
        assert_eq!(header.entry_count, 3);
    }

    // ================================================================
    // mmap pipeline tests (task 2.9)
    // ================================================================

    #[test]
    fn test_mmap_v2_page_aligned() {
        // Verify that entries in a v2 file start at a page boundary
        // after mmap. HEADER_SIZE_V2 == 16384 == Apple Silicon page size,
        // and mmap base is always page-aligned, so entries must be too.
        use crate::io::mmap::{MmapBuffer, PAGE_SIZE};

        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("aligned.idx");
        let entries = make_test_entries(20);

        save_v2(&entries, 0xA, &idx_path, 0, 0).unwrap();

        let mmap = MmapBuffer::from_file(&idx_path).unwrap();
        let entry_start = mmap.as_ptr() as usize + HEADER_SIZE_V2;

        assert_eq!(
            entry_start % PAGE_SIZE,
            0,
            "Entry region must be page-aligned: ptr={:#x}, offset={}, remainder={}",
            mmap.as_ptr() as usize,
            HEADER_SIZE_V2,
            entry_start % PAGE_SIZE
        );

        // Also verify the mmap base pointer itself is page-aligned
        assert_eq!(
            mmap.as_ptr() as usize % PAGE_SIZE,
            0,
            "mmap base pointer must be page-aligned"
        );
    }

    #[test]
    fn test_bytesnocopy_succeeds() {
        // Metal bytesNoCopy should succeed on page-aligned mmap of v2 file
        let device = get_device();
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("bytesnocopy.idx");
        let entries = make_test_entries(15);

        save_v2(&entries, 0xBB, &idx_path, 100, 0).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, Some(&device)).unwrap();

        assert!(
            snapshot.has_metal_buffer(),
            "bytesNoCopy should succeed for page-aligned mmap on Apple Silicon"
        );
        assert_eq!(snapshot.entry_count(), 15);
        // The Metal buffer should cover at least the full mmap region
        // (bytesNoCopy wraps the entire mmap, not just entries)
        let metal_buf = snapshot.metal_buffer().unwrap();
        assert!(
            metal_buf.length() >= snapshot.mapped_len(),
            "Metal buffer should cover full mmap region: buf_len={}, mapped_len={}",
            metal_buf.length(),
            snapshot.mapped_len()
        );
    }

    #[test]
    fn test_metal_buffer_contents_match() {
        // Buffer data at entry offset must match mmap'd entries byte-for-byte
        let device = get_device();
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("contents.idx");
        let entries = make_test_entries(7);

        save_v2(&entries, 0xCC, &idx_path, 0, 0).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, Some(&device)).unwrap();
        assert!(snapshot.has_metal_buffer());

        let metal_buf = snapshot.metal_buffer().unwrap();
        let snap_entries = snapshot.entries();

        // Compare each entry in the Metal buffer with the snapshot entries
        unsafe {
            let buf_ptr = metal_buf.contents().as_ptr() as *const u8;
            // bytesNoCopy wraps full mmap, so entries start at HEADER_SIZE_V2
            let entry_start = buf_ptr.add(HEADER_SIZE_V2);

            for (i, entry) in snap_entries.iter().enumerate() {
                let orig_bytes: &[u8; 256] =
                    &*(entry as *const GpuPathEntry as *const [u8; 256]);
                let buf_bytes =
                    std::slice::from_raw_parts(entry_start.add(i * 256), 256);
                assert_eq!(
                    buf_bytes, orig_bytes,
                    "Metal buffer entry {} does not match mmap entry byte-for-byte",
                    i
                );
            }
        }
    }

    #[test]
    fn test_buffer_length_covers_entries() {
        // Metal buffer length must be >= entry_count * 256 (plus header for bytesNoCopy)
        let device = get_device();
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("length.idx");
        let entries = make_test_entries(25);

        save_v2(&entries, 0xDD, &idx_path, 0, 0).unwrap();

        let snapshot = IndexSnapshot::from_file(&idx_path, Some(&device)).unwrap();
        assert!(snapshot.has_metal_buffer());

        let metal_buf = snapshot.metal_buffer().unwrap();
        let expected_entry_bytes = snapshot.entry_count() * 256;

        // The buffer wraps the full mmap (header + entries), so its length
        // must cover HEADER_SIZE_V2 + entry_count * 256 at minimum
        assert!(
            metal_buf.length() >= HEADER_SIZE_V2 + expected_entry_bytes,
            "Buffer length {} must cover header ({}) + entries ({})",
            metal_buf.length(),
            HEADER_SIZE_V2,
            expected_entry_bytes
        );

        // Verify the pure entry region size
        assert_eq!(
            expected_entry_bytes,
            25 * 256,
            "entry_count * 256 should be {} * 256 = {}",
            25,
            25 * 256
        );
    }
}
