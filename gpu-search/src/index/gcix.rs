// GCIX binary format: page-aligned content index for GPU-friendly persistent content store.
//
// The GCIX format persists a ContentStore to disk in a layout suitable for
// mmap + Metal bytesNoCopy reload on next startup. Layout:
//
//   [0 .. 16384)             GcixHeader (one full page, zero-padded)
//   [16384 .. meta_end)      FileContentMeta table (file_count * 32 bytes)
//   [meta_end .. content_offset)  Zero padding to next page boundary
//   [content_offset .. EOF)  Content data (page-aligned for bytesNoCopy)

use std::io::{self, BufWriter, Write};
use std::mem::size_of;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::index::cache::CacheError;
use crate::index::content_snapshot::ContentSnapshot;
use crate::index::content_store::{ContentStore, FileContentMeta};
use crate::io::mmap::{MmapBuffer, PAGE_SIZE};

/// Header size: one full Apple Silicon page (16KB).
pub const GCIX_HEADER_SIZE: usize = PAGE_SIZE; // 16384

/// Magic bytes: "GCIX" as individual bytes.
pub const GCIX_MAGIC: [u8; 4] = [b'G', b'C', b'I', b'X'];

/// Current GCIX format version.
pub const GCIX_VERSION: u32 = 1;

/// Byte offset where the CRC32 field lives in the serialized header.
/// Fields before crc32 in the serialized (packed) format:
///   magic(4) + version(4) + file_count(4) + content_bytes(8) +
///   meta_offset(8) + content_offset(8) + root_hash(8) + last_fsevents(8) + saved_at(8) = 60
const CRC32_OFFSET: usize = 60;

/// GCIX file header. Padded to 16384 bytes (one full page).
///
/// On-disk serialized layout (all fields little-endian, packed):
///   [0..4)     magic           "GCIX"
///   [4..8)     version         1
///   [8..12)    file_count      number of files
///   [12..20)   content_bytes   total content data bytes
///   [20..28)   meta_offset     byte offset of FileContentMeta table
///   [28..36)   content_offset  byte offset of content data (page-aligned)
///   [36..44)   root_hash       hash of the indexed root path
///   [44..52)   last_fsevents   macOS FSEvents event ID for resume
///   [52..60)   saved_at        unix timestamp (seconds)
///   [60..64)   header_crc32    CRC32 over bytes [0..60)
///   [64..16384) padding        zero-filled to page boundary
///
/// Note: The in-memory struct uses `#[repr(C)]` which inserts 4 bytes of
/// alignment padding between file_count (u32) and content_bytes (u64).
/// Serialization/deserialization use explicit byte-level packing (to_bytes/
/// from_bytes) so the on-disk format is always the packed layout above.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct GcixHeader {
    /// Magic bytes: "GCIX"
    pub magic: [u8; 4],
    /// Format version (currently 1)
    pub version: u32,
    /// Number of files in the content store
    pub file_count: u32,
    /// Explicit alignment padding (repr(C) would insert this anyway)
    pub _align_pad: u32,
    /// Total content data bytes
    pub content_bytes: u64,
    /// Byte offset of the FileContentMeta table (always 16384)
    pub meta_offset: u64,
    /// Byte offset of content data (page-aligned)
    pub content_offset: u64,
    /// Hash of the indexed root path
    pub root_hash: u64,
    /// macOS FSEvents event ID for resumable watching
    pub last_fsevents: u64,
    /// Unix timestamp (seconds since epoch) of save time
    pub saved_at: u64,
    /// CRC32 over serialized header bytes [0..60)
    pub header_crc32: u32,
    /// Zero-filled padding to 16384 bytes
    pub _padding: [u8; GCIX_HEADER_SIZE - 68],
}

// Compile-time assertions
const _: () = assert!(size_of::<GcixHeader>() == GCIX_HEADER_SIZE);
const _: () = assert!(GCIX_HEADER_SIZE == 16384);
const _: () = assert!(size_of::<FileContentMeta>() == 32);

impl GcixHeader {
    /// Create a new GcixHeader with the given parameters.
    pub fn new(
        file_count: u32,
        content_bytes: u64,
        content_offset: u64,
        root_hash: u64,
        last_fsevents: u64,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            magic: GCIX_MAGIC,
            version: GCIX_VERSION,
            file_count,
            _align_pad: 0,
            content_bytes,
            meta_offset: GCIX_HEADER_SIZE as u64,
            content_offset,
            root_hash,
            last_fsevents,
            saved_at: now,
            header_crc32: 0,
            _padding: [0u8; GCIX_HEADER_SIZE - 68],
        }
    }

    /// Serialize this header to a 16384-byte array (little-endian).
    /// Computes CRC32 over bytes [0..60) and stores it at [60..64).
    pub fn to_bytes(&self) -> [u8; GCIX_HEADER_SIZE] {
        let mut buf = [0u8; GCIX_HEADER_SIZE];

        // Write fields little-endian
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.file_count.to_le_bytes());
        buf[12..20].copy_from_slice(&self.content_bytes.to_le_bytes());
        buf[20..28].copy_from_slice(&self.meta_offset.to_le_bytes());
        buf[28..36].copy_from_slice(&self.content_offset.to_le_bytes());
        buf[36..44].copy_from_slice(&self.root_hash.to_le_bytes());
        buf[44..52].copy_from_slice(&self.last_fsevents.to_le_bytes());
        buf[52..60].copy_from_slice(&self.saved_at.to_le_bytes());
        // [60..64) = crc32, computed below
        // [64..16384) = padding, already zeroed

        // Compute CRC32 over bytes [0..60)
        let crc = crc32fast::hash(&buf[..CRC32_OFFSET]);
        buf[CRC32_OFFSET..CRC32_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());

        buf
    }

    /// Deserialize a GcixHeader from a byte buffer.
    /// Validates magic, version, and CRC32.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, io::Error> {
        if buf.len() < GCIX_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GCIX header too small: {} bytes, expected {}",
                    buf.len(),
                    GCIX_HEADER_SIZE,
                ),
            ));
        }

        // Check magic
        if buf[0..4] != GCIX_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GCIX bad magic: {:?}, expected {:?}",
                    &buf[0..4],
                    GCIX_MAGIC,
                ),
            ));
        }

        // Check version
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != GCIX_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GCIX unsupported version: {}, expected {}",
                    version, GCIX_VERSION,
                ),
            ));
        }

        // Validate CRC32 over [0..60)
        let expected_crc = crc32fast::hash(&buf[..CRC32_OFFSET]);
        let stored_crc =
            u32::from_le_bytes(buf[CRC32_OFFSET..CRC32_OFFSET + 4].try_into().unwrap());
        if stored_crc != expected_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GCIX checksum mismatch: stored 0x{:08X}, computed 0x{:08X}",
                    stored_crc, expected_crc,
                ),
            ));
        }

        let file_count = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let content_bytes = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        let meta_offset = u64::from_le_bytes(buf[20..28].try_into().unwrap());
        let content_offset = u64::from_le_bytes(buf[28..36].try_into().unwrap());
        let root_hash = u64::from_le_bytes(buf[36..44].try_into().unwrap());
        let last_fsevents = u64::from_le_bytes(buf[44..52].try_into().unwrap());
        let saved_at = u64::from_le_bytes(buf[52..60].try_into().unwrap());

        // Padding is all zeros in the serialized format; no need to copy.
        Ok(Self {
            magic: GCIX_MAGIC,
            version,
            file_count,
            _align_pad: 0,
            content_bytes,
            meta_offset,
            content_offset,
            root_hash,
            last_fsevents,
            saved_at,
            header_crc32: stored_crc,
            _padding: [0u8; GCIX_HEADER_SIZE - 68],
        })
    }
}

/// Align a byte offset up to the next page boundary (16KB).
fn align_up_page(offset: usize) -> usize {
    (offset + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// Save a ContentStore to a GCIX file.
///
/// Writes:
///   1. GcixHeader at offset 0 (16384 bytes)
///   2. FileContentMeta table at meta_offset (16384)
///   3. Zero padding to next page boundary
///   4. Content data at content_offset (page-aligned for bytesNoCopy)
///
/// Uses atomic write (write to .tmp, fsync, rename).
pub fn save_gcix(
    store: &ContentStore,
    path: &Path,
    root_hash: u64,
    fsevents_id: u64,
) -> io::Result<()> {
    let file_count = store.file_count();
    let files = store.files();
    let content = store.buffer();
    let content_bytes = store.total_bytes();

    // Meta table starts right after the header
    let meta_offset = GCIX_HEADER_SIZE;
    let meta_bytes = file_count as usize * size_of::<FileContentMeta>();
    let meta_end = meta_offset + meta_bytes;

    // Content data starts at next page boundary after meta table
    let content_offset = align_up_page(meta_end);

    // Build header
    let header = GcixHeader::new(
        file_count,
        content_bytes,
        content_offset as u64,
        root_hash,
        fsevents_id,
    );
    let header_bytes = header.to_bytes();

    // Atomic write: .tmp -> fsync -> rename
    let tmp_path = path.with_extension("gcix.tmp");

    // Ensure parent directory exists
    if let Some(parent) = tmp_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::File::create(&tmp_path)?;
    let mut writer = BufWriter::new(file);

    // 1. Write header (16384 bytes)
    writer.write_all(&header_bytes)?;

    // 2. Write FileContentMeta table
    for meta in files {
        // SAFETY: FileContentMeta is #[repr(C)], exactly 32 bytes, no padding issues.
        let meta_bytes: &[u8; 32] =
            unsafe { &*(meta as *const FileContentMeta as *const [u8; 32]) };
        writer.write_all(meta_bytes)?;
    }

    // 3. Write zero padding to content_offset
    let padding_needed = content_offset - meta_end;
    if padding_needed > 0 {
        let zeros = vec![0u8; padding_needed];
        writer.write_all(&zeros)?;
    }

    // 4. Write content data
    writer.write_all(content)?;

    // Flush and fsync
    writer.flush()?;
    let file = writer.into_inner().map_err(|e| e.into_error())?;
    file.sync_all()?;
    drop(file);

    // Atomic rename
    std::fs::rename(&tmp_path, path)?;

    Ok(())
}

/// Load a ContentSnapshot from a GCIX file via mmap.
///
/// 1. Memory-maps the file via `MmapBuffer::from_file()`
/// 2. Validates header: magic ("GCIX"), version (1), CRC32
/// 3. Extracts FileContentMeta table via pointer arithmetic
/// 4. Creates Metal buffer from content region via bytesNoCopy
///    (content_offset is page-aligned for zero-copy GPU access)
/// 5. Constructs ContentStore and ContentSnapshot
///
/// # Arguments
/// * `path` - Path to the .gcix file
/// * `device` - Optional Metal device. If provided, creates a GPU buffer
///   via bytesNoCopy from the mmap'd content region.
///
/// # Errors
/// Returns `CacheError` if the file cannot be read, the header is invalid,
/// or CRC32 verification fails.
pub fn load_gcix(
    path: &Path,
    device: Option<&ProtocolObject<dyn MTLDevice>>,
) -> Result<ContentSnapshot, CacheError> {
    // 1. mmap the file
    let mmap = MmapBuffer::from_file(path).map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound {
            CacheError::NotFound(path.to_path_buf())
        } else {
            CacheError::Io(e)
        }
    })?;

    let data = mmap.as_slice();

    // 2. Validate header (magic, version, CRC32)
    if data.len() < GCIX_HEADER_SIZE {
        return Err(CacheError::InvalidFormat(format!(
            "GCIX file too small: {} bytes, need at least {}",
            data.len(),
            GCIX_HEADER_SIZE,
        )));
    }

    let header = GcixHeader::from_bytes(data).map_err(|e| {
        CacheError::InvalidFormat(format!("GCIX header validation failed: {}", e))
    })?;

    // 3. Validate offsets and sizes are consistent
    let meta_offset = header.meta_offset as usize;
    let content_offset = header.content_offset as usize;
    let file_count = header.file_count as usize;
    let content_bytes = header.content_bytes as usize;
    let meta_size = file_count * size_of::<FileContentMeta>();
    let meta_end = meta_offset + meta_size;

    // Verify content_offset is page-aligned
    if !content_offset.is_multiple_of(PAGE_SIZE) {
        return Err(CacheError::InvalidFormat(format!(
            "GCIX content_offset {} is not page-aligned ({})",
            content_offset, PAGE_SIZE,
        )));
    }

    // Verify the file is large enough for meta table + content
    let required_len = if content_bytes > 0 {
        content_offset + content_bytes
    } else {
        meta_end
    };

    if data.len() < required_len {
        return Err(CacheError::InvalidFormat(format!(
            "GCIX file too small for data: {} bytes, need {}",
            data.len(),
            required_len,
        )));
    }

    // Verify meta region doesn't overlap content region
    if meta_end > content_offset && content_bytes > 0 {
        return Err(CacheError::InvalidFormat(format!(
            "GCIX meta table (ends at {}) overlaps content region (starts at {})",
            meta_end, content_offset,
        )));
    }

    // 4. Extract FileContentMeta table via pointer arithmetic
    let mut files = Vec::with_capacity(file_count);
    for i in 0..file_count {
        let offset = meta_offset + i * size_of::<FileContentMeta>();
        // SAFETY: FileContentMeta is #[repr(C)], exactly 32 bytes. We verified
        // the file is large enough above. read_unaligned handles any alignment.
        let meta: FileContentMeta = unsafe {
            std::ptr::read_unaligned(data[offset..].as_ptr() as *const FileContentMeta)
        };
        files.push(meta);
    }

    // 5. Construct ContentStore from mmap'd data with optional Metal buffer
    let store = ContentStore::from_gcix_mmap(
        mmap,
        files,
        content_offset,
        content_bytes,
        device,
    );

    // 6. Wrap in ContentSnapshot
    let snapshot = ContentSnapshot::new(store, header.saved_at);

    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcix_header_size() {
        assert_eq!(size_of::<GcixHeader>(), GCIX_HEADER_SIZE);
        assert_eq!(GCIX_HEADER_SIZE, 16384);
    }

    #[test]
    fn test_gcix_header_serialization_roundtrip() {
        let header = GcixHeader::new(42, 100_000, 32768, 0xDEAD_BEEF_CAFE_1234, 999);
        let bytes = header.to_bytes();
        let recovered = GcixHeader::from_bytes(&bytes).expect("roundtrip should succeed");

        assert_eq!(recovered.magic, GCIX_MAGIC);
        assert_eq!(recovered.version, GCIX_VERSION);
        assert_eq!(recovered.file_count, 42);
        assert_eq!(recovered.content_bytes, 100_000);
        assert_eq!(recovered.meta_offset, GCIX_HEADER_SIZE as u64);
        assert_eq!(recovered.content_offset, 32768);
        assert_eq!(recovered.root_hash, 0xDEAD_BEEF_CAFE_1234);
        assert_eq!(recovered.last_fsevents, 999);
        assert_eq!(recovered.saved_at, header.saved_at);

        // Re-serialize should produce identical bytes
        let bytes2 = recovered.to_bytes();
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn test_gcix_header_magic_bytes() {
        let header = GcixHeader::new(0, 0, 16384, 0, 0);
        let bytes = header.to_bytes();
        assert_eq!(bytes[0], b'G');
        assert_eq!(bytes[1], b'C');
        assert_eq!(bytes[2], b'I');
        assert_eq!(bytes[3], b'X');
    }

    #[test]
    fn test_gcix_header_rejects_bad_magic() {
        let mut bytes = GcixHeader::new(0, 0, 16384, 0, 0).to_bytes();
        bytes[0] = 0xFF;
        assert!(GcixHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_gcix_header_rejects_bad_version() {
        let mut bytes = GcixHeader::new(0, 0, 16384, 0, 0).to_bytes();
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        // Recompute CRC so only version check triggers
        let crc = crc32fast::hash(&bytes[..CRC32_OFFSET]);
        bytes[CRC32_OFFSET..CRC32_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
        let err = GcixHeader::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("version"), "error: {err}");
    }

    #[test]
    fn test_gcix_header_rejects_bad_crc() {
        let mut bytes = GcixHeader::new(10, 1000, 32768, 0x1234, 0).to_bytes();
        // Corrupt a data byte without updating CRC
        bytes[8] = 0xFF;
        let err = GcixHeader::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("checksum"), "error: {err}");
    }

    #[test]
    fn test_gcix_header_rejects_too_small() {
        let small = [0u8; 100];
        assert!(GcixHeader::from_bytes(&small).is_err());
    }

    #[test]
    fn test_align_up_page() {
        assert_eq!(align_up_page(0), 0);
        assert_eq!(align_up_page(1), PAGE_SIZE);
        assert_eq!(align_up_page(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(align_up_page(PAGE_SIZE + 1), PAGE_SIZE * 2);
        assert_eq!(align_up_page(PAGE_SIZE * 3), PAGE_SIZE * 3);
    }

    #[test]
    fn test_save_gcix_basic() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("test.gcix");

        // Build a content store with known data
        let mut store = ContentStore::new();
        store.insert(b"hello world", 0, 0xAAAA, 1000);
        store.insert(b"second file content here", 1, 0xBBBB, 2000);
        store.insert(b"third", 2, 0xCCCC, 3000);

        save_gcix(&store, &gcix_path, 0x1234_5678_9ABC_DEF0, 42)
            .expect("save_gcix should succeed");

        assert!(gcix_path.exists(), "GCIX file should exist");

        // Read back raw bytes
        let data = std::fs::read(&gcix_path).unwrap();

        // Verify header
        let header = GcixHeader::from_bytes(&data).expect("header should parse");
        assert_eq!(header.magic, GCIX_MAGIC);
        assert_eq!(header.version, GCIX_VERSION);
        assert_eq!(header.file_count, 3);
        assert_eq!(header.content_bytes, store.total_bytes());
        assert_eq!(header.meta_offset, GCIX_HEADER_SIZE as u64);
        assert_eq!(header.root_hash, 0x1234_5678_9ABC_DEF0);
        assert_eq!(header.last_fsevents, 42);

        // Verify content_offset is page-aligned
        let content_offset = header.content_offset as usize;
        assert_eq!(
            content_offset % PAGE_SIZE,
            0,
            "content_offset must be page-aligned"
        );

        // Verify meta table
        let meta_offset = header.meta_offset as usize;
        let meta_size = size_of::<FileContentMeta>();
        for (i, expected_meta) in store.files().iter().enumerate() {
            let offset = meta_offset + i * meta_size;
            let stored: FileContentMeta = unsafe {
                std::ptr::read_unaligned(data[offset..].as_ptr() as *const FileContentMeta)
            };
            assert_eq!(stored.content_offset, expected_meta.content_offset);
            assert_eq!(stored.content_len, expected_meta.content_len);
            assert_eq!(stored.path_id, expected_meta.path_id);
            assert_eq!(stored.content_hash, expected_meta.content_hash);
            assert_eq!(stored.mtime, expected_meta.mtime);
        }

        // Verify content data
        let content_data = &data[content_offset..];
        assert_eq!(
            &content_data[..store.buffer().len()],
            store.buffer(),
            "content data should match store buffer"
        );
    }

    #[test]
    fn test_save_gcix_empty_store() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("empty.gcix");

        let store = ContentStore::new();
        save_gcix(&store, &gcix_path, 0, 0).expect("save empty should succeed");

        let data = std::fs::read(&gcix_path).unwrap();
        let header = GcixHeader::from_bytes(&data).expect("header should parse");
        assert_eq!(header.file_count, 0);
        assert_eq!(header.content_bytes, 0);
    }

    #[test]
    fn test_save_gcix_creates_parent_dirs() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("nested").join("deep").join("test.gcix");

        let store = ContentStore::new();
        save_gcix(&store, &gcix_path, 0, 0).expect("should create parent dirs");
        assert!(gcix_path.exists());
    }

    #[test]
    fn test_save_gcix_content_offset_page_aligned() {
        // Test with varying file counts to ensure page alignment always holds
        for file_count in [1, 10, 100, 500, 512, 513] {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let gcix_path = dir.path().join(format!("test_{}.gcix", file_count));

            let mut store = ContentStore::new();
            for i in 0..file_count {
                let content = format!("content for file {}", i);
                store.insert(content.as_bytes(), i, 0, 0);
            }

            save_gcix(&store, &gcix_path, 0, 0).expect("save should succeed");

            let data = std::fs::read(&gcix_path).unwrap();
            let header = GcixHeader::from_bytes(&data).expect("header should parse");

            assert_eq!(
                header.content_offset as usize % PAGE_SIZE,
                0,
                "content_offset must be page-aligned for {} files",
                file_count,
            );
            assert_eq!(header.file_count, file_count);
        }
    }

    #[test]
    fn test_save_gcix_atomic_overwrite() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("overwrite.gcix");

        // Write first version
        let mut store1 = ContentStore::new();
        store1.insert(b"first", 0, 0xAA, 100);
        save_gcix(&store1, &gcix_path, 111, 1).expect("save 1");

        // Overwrite with second version
        let mut store2 = ContentStore::new();
        store2.insert(b"second version", 0, 0xBB, 200);
        store2.insert(b"another file", 1, 0xCC, 300);
        save_gcix(&store2, &gcix_path, 222, 2).expect("save 2");

        // Verify second version
        let data = std::fs::read(&gcix_path).unwrap();
        let header = GcixHeader::from_bytes(&data).expect("header should parse");
        assert_eq!(header.file_count, 2);
        assert_eq!(header.root_hash, 222);
        assert_eq!(header.last_fsevents, 2);

        // No temp file should remain
        assert!(!gcix_path.with_extension("gcix.tmp").exists());
    }

    #[test]
    fn test_save_gcix_large_content() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("large.gcix");

        let mut store = ContentStore::new();
        // Insert files totaling > 1 page of meta + content that spans pages
        for i in 0..50 {
            let content = format!("{:0>1000}", i); // 1000 bytes each
            store.insert(content.as_bytes(), i, crc32fast::hash(content.as_bytes()), 0);
        }

        save_gcix(&store, &gcix_path, 0xFEED, 0).expect("save large");

        let data = std::fs::read(&gcix_path).unwrap();
        let header = GcixHeader::from_bytes(&data).expect("header");
        assert_eq!(header.file_count, 50);

        // Verify all content is accessible
        let content_offset = header.content_offset as usize;
        for (i, meta) in store.files().iter().enumerate() {
            let start = content_offset + meta.content_offset as usize;
            let end = start + meta.content_len as usize;
            let expected = format!("{:0>1000}", i);
            assert_eq!(
                &data[start..end],
                expected.as_bytes(),
                "file {} content should match",
                i,
            );
        }
    }

    // ================================================================
    // load_gcix tests
    // ================================================================

    #[test]
    fn test_load_gcix_basic_roundtrip() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("roundtrip.gcix");

        // Build a content store with known data
        let mut store = ContentStore::new();
        store.insert(b"hello world", 0, 0xAAAA, 1000);
        store.insert(b"second file content here", 1, 0xBBBB, 2000);
        store.insert(b"third", 2, 0xCCCC, 3000);

        let original_file_count = store.file_count();
        let original_total_bytes = store.total_bytes();

        save_gcix(&store, &gcix_path, 0x1234_5678_9ABC_DEF0, 42)
            .expect("save_gcix should succeed");

        // Load without Metal device
        let snapshot = load_gcix(&gcix_path, None).expect("load_gcix should succeed");

        assert_eq!(snapshot.file_count(), original_file_count);

        let loaded = snapshot.content_store();
        assert_eq!(loaded.file_count(), original_file_count);
        assert_eq!(loaded.total_bytes(), original_total_bytes);

        // Verify all file content matches
        assert_eq!(loaded.content_for(0).unwrap(), b"hello world");
        assert_eq!(loaded.content_for(1).unwrap(), b"second file content here");
        assert_eq!(loaded.content_for(2).unwrap(), b"third");

        // Out of bounds
        assert!(loaded.content_for(3).is_none());
    }

    #[test]
    fn test_load_gcix_metadata_roundtrip() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("meta.gcix");

        let mut store = ContentStore::new();
        store.insert(b"file_a", 10, 0xAA, 1000);
        store.insert(b"file_b", 20, 0xBB, 2000);

        save_gcix(&store, &gcix_path, 0xDEAD, 99).expect("save");

        let snapshot = load_gcix(&gcix_path, None).expect("load");
        let loaded = snapshot.content_store();
        let files = loaded.files();

        assert_eq!(files.len(), 2);

        // Verify metadata fields match
        assert_eq!(files[0].path_id, 10);
        assert_eq!(files[0].content_hash, 0xAA);
        assert_eq!(files[0].mtime, 1000);
        assert_eq!(files[0].content_len, 6); // "file_a" = 6 bytes

        assert_eq!(files[1].path_id, 20);
        assert_eq!(files[1].content_hash, 0xBB);
        assert_eq!(files[1].mtime, 2000);
        assert_eq!(files[1].content_len, 6); // "file_b" = 6 bytes
    }

    #[test]
    fn test_load_gcix_empty_store_roundtrip() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("empty_rt.gcix");

        let store = ContentStore::new();
        save_gcix(&store, &gcix_path, 0, 0).expect("save empty");

        let snapshot = load_gcix(&gcix_path, None).expect("load empty");
        assert_eq!(snapshot.file_count(), 0);
        assert_eq!(snapshot.content_store().total_bytes(), 0);
        assert!(snapshot.content_store().content_for(0).is_none());
    }

    #[test]
    fn test_load_gcix_50_files_roundtrip() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("50files.gcix");

        let mut store = ContentStore::new();
        let mut expected: Vec<Vec<u8>> = Vec::new();
        for i in 0..50u32 {
            let content = format!("content_for_file_{:04}_with_some_extra_data", i);
            store.insert(
                content.as_bytes(),
                i,
                crc32fast::hash(content.as_bytes()),
                1700000000 + i,
            );
            expected.push(content.into_bytes());
        }

        save_gcix(&store, &gcix_path, 0xBEEF, 123).expect("save 50 files");

        let snapshot = load_gcix(&gcix_path, None).expect("load 50 files");
        let loaded = snapshot.content_store();

        assert_eq!(loaded.file_count(), 50);

        for (i, exp) in expected.iter().enumerate() {
            let actual = loaded
                .content_for(i as u32)
                .unwrap_or_else(|| panic!("content_for({}) returned None", i));
            assert_eq!(
                actual, exp.as_slice(),
                "content mismatch for file {}",
                i,
            );
        }
    }

    #[test]
    fn test_load_gcix_with_metal_device() {
        use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice};

        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("metal.gcix");

        let mut store = ContentStore::new();
        store.insert(b"GPU accessible content", 0, 0xAA, 100);
        store.insert(b"more GPU content here", 1, 0xBB, 200);

        save_gcix(&store, &gcix_path, 0xFACE, 7).expect("save");

        let snapshot = load_gcix(&gcix_path, Some(&device)).expect("load with device");
        let loaded = snapshot.content_store();

        // Verify Metal buffer was created
        assert!(
            loaded.has_metal_buffer(),
            "Metal buffer should be created via bytesNoCopy from mmap'd content region"
        );

        // Verify content is still accessible
        assert_eq!(loaded.content_for(0).unwrap(), b"GPU accessible content");
        assert_eq!(loaded.content_for(1).unwrap(), b"more GPU content here");

        // Verify Metal buffer contents match content data
        let metal_buf = loaded.metal_buffer().unwrap();
        assert!(
            metal_buf.length() >= loaded.total_bytes() as usize,
            "Metal buffer length ({}) must be >= total_bytes ({})",
            metal_buf.length(),
            loaded.total_bytes(),
        );

        // Verify Metal buffer data matches
        unsafe {
            let buf_ptr = metal_buf.contents().as_ptr() as *const u8;
            let buf_slice =
                std::slice::from_raw_parts(buf_ptr, loaded.total_bytes() as usize);
            assert_eq!(
                buf_slice,
                loaded.buffer(),
                "Metal buffer contents should match loaded content data"
            );
        }
    }

    #[test]
    fn test_load_gcix_not_found() {
        let result = load_gcix(Path::new("/nonexistent/path/test.gcix"), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_gcix_corrupt_magic() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("corrupt_magic.gcix");

        // Write a valid GCIX first
        let mut store = ContentStore::new();
        store.insert(b"test", 0, 0, 0);
        save_gcix(&store, &gcix_path, 0, 0).expect("save");

        // Corrupt the magic bytes
        let mut data = std::fs::read(&gcix_path).unwrap();
        data[0] = 0xFF;
        std::fs::write(&gcix_path, &data).unwrap();

        let result = load_gcix(&gcix_path, None);
        assert!(result.is_err(), "Should fail with corrupt magic");
    }

    #[test]
    fn test_load_gcix_corrupt_crc() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("corrupt_crc.gcix");

        let mut store = ContentStore::new();
        store.insert(b"test", 0, 0, 0);
        save_gcix(&store, &gcix_path, 0, 0).expect("save");

        // Corrupt a data byte without updating CRC
        let mut data = std::fs::read(&gcix_path).unwrap();
        data[8] ^= 0xFF; // flip file_count byte
        std::fs::write(&gcix_path, &data).unwrap();

        let result = load_gcix(&gcix_path, None);
        assert!(result.is_err(), "Should fail with CRC mismatch");
    }

    #[test]
    fn test_load_gcix_version_mismatch() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("bad_version.gcix");

        let mut store = ContentStore::new();
        store.insert(b"test content", 0, 0xAA, 100);
        save_gcix(&store, &gcix_path, 0, 0).expect("save");

        // Corrupt the version field and recompute CRC so only version check triggers
        let mut data = std::fs::read(&gcix_path).unwrap();
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        let crc = crc32fast::hash(&data[..CRC32_OFFSET]);
        data[CRC32_OFFSET..CRC32_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
        std::fs::write(&gcix_path, &data).unwrap();

        let result = load_gcix(&gcix_path, None);
        match result {
            Err(CacheError::InvalidFormat(msg)) => {
                assert!(msg.contains("version"), "error should mention version: {msg}");
            }
            Err(e) => panic!("Expected CacheError::InvalidFormat with version, got: {e}"),
            Ok(_) => panic!("Expected CacheError but load succeeded"),
        }
    }

    #[test]
    fn test_load_gcix_large_content_roundtrip() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("large_rt.gcix");

        let mut store = ContentStore::new();
        // Insert files that will cross page boundaries
        for i in 0..100u32 {
            let content = format!("{:0>500}", i); // 500 bytes each = 50KB total
            store.insert(content.as_bytes(), i, i * 7, i * 13);
        }

        save_gcix(&store, &gcix_path, 0xCAFE_BABE, 9999).expect("save large");

        let snapshot = load_gcix(&gcix_path, None).expect("load large");
        let loaded = snapshot.content_store();

        assert_eq!(loaded.file_count(), 100);

        for i in 0..100u32 {
            let expected = format!("{:0>500}", i);
            let actual = loaded.content_for(i).unwrap();
            assert_eq!(
                actual,
                expected.as_bytes(),
                "content mismatch for file {}",
                i,
            );

            // Verify metadata
            let meta = &loaded.files()[i as usize];
            assert_eq!(meta.path_id, i);
            assert_eq!(meta.content_hash, i * 7);
            assert_eq!(meta.mtime, i * 13);
        }
    }
}
