// GCIX binary format: page-aligned content index for GPU-friendly persistent content store.
//
// The GCIX format persists a ContentStore to disk in a layout suitable for
// mmap + Metal bytesNoCopy reload on next startup. Layout:
//
//   [0 .. 16384)             GcixHeader (one full page, zero-padded)
//   [16384 .. meta_end)      FileContentMeta table (file_count * 32 bytes)
//   [meta_end .. content_offset)  Zero padding to next page boundary
//   [content_offset .. content_end)  Content data (page-aligned for bytesNoCopy)
//   [content_end .. EOF)     Path table (v2): u32-len-prefixed UTF-8 strings per file_id

use std::io::{self, BufWriter, Write};
use std::mem::size_of;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::index::cache::CacheError;
use crate::index::content_snapshot::ContentSnapshot;
use crate::index::content_store::{build_chunk_metadata, ContentStore, FileContentMeta};
use crate::io::mmap::{MmapBuffer, PAGE_SIZE};

/// Header size: one full Apple Silicon page (16KB).
pub const GCIX_HEADER_SIZE: usize = PAGE_SIZE; // 16384

/// Magic bytes: "GCIX" as individual bytes.
pub const GCIX_MAGIC: [u8; 4] = [b'G', b'C', b'I', b'X'];

/// Current GCIX format version.
pub const GCIX_VERSION: u32 = 3;

/// Byte offset where the CRC32 field lives in the serialized header.
/// Fields before crc32 in the serialized (packed) format:
///   magic(4) + version(4) + file_count(4) + content_bytes(8) +
///   meta_offset(8) + content_offset(8) + root_hash(8) + last_fsevents(8) + saved_at(8) +
///   paths_offset(8) + paths_bytes(8) + chunks_offset(8) + chunks_bytes(8) = 92
const CRC32_OFFSET: usize = 92;

/// GCIX file header. Padded to 16384 bytes (one full page).
///
/// On-disk serialized layout (all fields little-endian, packed):
///   [0..4)     magic           "GCIX"
///   [4..8)     version         3
///   [8..12)    file_count      number of files
///   [12..20)   content_bytes   total content data bytes
///   [20..28)   meta_offset     byte offset of FileContentMeta table
///   [28..36)   content_offset  byte offset of content data (page-aligned)
///   [36..44)   root_hash       hash of the indexed root path
///   [44..52)   last_fsevents   macOS FSEvents event ID for resume
///   [52..60)   saved_at        unix timestamp (seconds)
///   [60..68)   paths_offset    byte offset of path table (after content data)
///   [68..76)   paths_bytes     total bytes of path table
///   [76..84)   chunks_offset   byte offset of chunk metadata table (v3)
///   [84..92)   chunks_bytes    total bytes of chunk metadata table (v3)
///   [92..96)   header_crc32    CRC32 over bytes [0..92)
///   [96..16384) padding        zero-filled to page boundary
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
    /// Format version (currently 2)
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
    /// Byte offset of path table (after content data)
    pub paths_offset: u64,
    /// Total bytes of path table
    pub paths_bytes: u64,
    /// Byte offset of chunk metadata table (v3, after path table)
    pub chunks_offset: u64,
    /// Total bytes of chunk metadata table (v3)
    pub chunks_bytes: u64,
    /// CRC32 over serialized header bytes [0..92)
    pub header_crc32: u32,
    /// Zero-filled padding to 16384 bytes
    pub _padding: [u8; GCIX_HEADER_SIZE - 100],
}

// Compile-time assertions
const _: () = assert!(size_of::<GcixHeader>() == GCIX_HEADER_SIZE);
const _: () = assert!(GCIX_HEADER_SIZE == 16384);
const _: () = assert!(size_of::<FileContentMeta>() == 32);

impl GcixHeader {
    /// Create a new GcixHeader with the given parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        file_count: u32,
        content_bytes: u64,
        content_offset: u64,
        root_hash: u64,
        last_fsevents: u64,
        paths_offset: u64,
        paths_bytes: u64,
        chunks_offset: u64,
        chunks_bytes: u64,
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
            paths_offset,
            paths_bytes,
            chunks_offset,
            chunks_bytes,
            header_crc32: 0,
            _padding: [0u8; GCIX_HEADER_SIZE - 100],
        }
    }

    /// Serialize this header to a 16384-byte array (little-endian).
    /// Computes CRC32 over bytes [0..76) and stores it at [76..80).
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
        buf[60..68].copy_from_slice(&self.paths_offset.to_le_bytes());
        buf[68..76].copy_from_slice(&self.paths_bytes.to_le_bytes());
        buf[76..84].copy_from_slice(&self.chunks_offset.to_le_bytes());
        buf[84..92].copy_from_slice(&self.chunks_bytes.to_le_bytes());
        // [92..96) = crc32, computed below
        // [96..16384) = padding, already zeroed

        // Compute CRC32 over bytes [0..76)
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

        // Check version (accept v2 for backward compat, v3 is current)
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != 2 && version != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GCIX unsupported version: {}, expected 2 or 3",
                    version,
                ),
            ));
        }

        // CRC32 offset depends on version: v2 = 76, v3 = 92
        let crc_offset = if version == 2 { 76 } else { CRC32_OFFSET };

        // Validate CRC32
        let expected_crc = crc32fast::hash(&buf[..crc_offset]);
        let stored_crc =
            u32::from_le_bytes(buf[crc_offset..crc_offset + 4].try_into().unwrap());
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
        let paths_offset = u64::from_le_bytes(buf[60..68].try_into().unwrap());
        let paths_bytes = u64::from_le_bytes(buf[68..76].try_into().unwrap());

        // v3 fields: chunks_offset and chunks_bytes (default to 0 for v2)
        let (chunks_offset, chunks_bytes) = if version >= 3 {
            (
                u64::from_le_bytes(buf[76..84].try_into().unwrap()),
                u64::from_le_bytes(buf[84..92].try_into().unwrap()),
            )
        } else {
            (0, 0)
        };

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
            paths_offset,
            paths_bytes,
            chunks_offset,
            chunks_bytes,
            header_crc32: stored_crc,
            _padding: [0u8; GCIX_HEADER_SIZE - 100],
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
    let paths = store.paths();

    // Meta table starts right after the header
    let meta_offset = GCIX_HEADER_SIZE;
    let meta_bytes = file_count as usize * size_of::<FileContentMeta>();
    let meta_end = meta_offset + meta_bytes;

    // Content data starts at next page boundary after meta table
    let content_offset = align_up_page(meta_end);

    // Path table: after content data. Each entry is u32 len + UTF-8 bytes.
    // Only written when paths are available (same length as file count).
    let content_end = content_offset + content_bytes as usize;
    let paths_offset = content_end;
    let mut path_table_buf: Vec<u8> = Vec::new();
    if !paths.is_empty() {
        for file_id in 0..file_count {
            let p = paths
                .get(file_id as usize)
                .map(|p| p.to_string_lossy())
                .unwrap_or_default();
            let p_bytes = p.as_bytes();
            path_table_buf.extend_from_slice(&(p_bytes.len() as u32).to_le_bytes());
            path_table_buf.extend_from_slice(p_bytes);
        }
    }
    let paths_bytes = path_table_buf.len() as u64;

    // Build chunk metadata table (v3)
    let chunk_metas = build_chunk_metadata(store);
    let chunks_table_offset = paths_offset + path_table_buf.len();
    let chunk_meta_size = size_of::<crate::search::content::ChunkMetadata>();
    let chunks_table_bytes = chunk_metas.len() * chunk_meta_size;

    // Build header
    let header = GcixHeader::new(
        file_count,
        content_bytes,
        content_offset as u64,
        root_hash,
        fsevents_id,
        paths_offset as u64,
        paths_bytes,
        chunks_table_offset as u64,
        chunks_table_bytes as u64,
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

    // 5. Write path table
    writer.write_all(&path_table_buf)?;

    // 6. Write chunk metadata table (v3)
    for cm in &chunk_metas {
        // SAFETY: ChunkMetadata is #[repr(C)], exactly 32 bytes, no padding issues.
        let cm_bytes: &[u8; 32] = unsafe {
            &*(cm as *const crate::search::content::ChunkMetadata as *const [u8; 32])
        };
        writer.write_all(cm_bytes)?;
    }

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
/// 2. Validates header: magic ("GCIX"), version (2), CRC32
/// 3. Extracts FileContentMeta table via pointer arithmetic
/// 4. Creates Metal buffer from content region via bytesNoCopy
///    (content_offset is page-aligned for zero-copy GPU access)
/// 5. Parses path table (GCIX v2: file_id -> path mapping)
/// 6. Constructs ContentStore and ContentSnapshot
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

    // 5. Parse path table (GCIX v2: stored after content data)
    let paths_offset = header.paths_offset as usize;
    let paths_bytes = header.paths_bytes as usize;
    let mut paths: Vec<std::path::PathBuf> = Vec::with_capacity(file_count);

    if paths_bytes > 0 {
        let paths_end = paths_offset + paths_bytes;
        if data.len() < paths_end {
            return Err(CacheError::InvalidFormat(format!(
                "GCIX file too small for path table: {} bytes, need {}",
                data.len(),
                paths_end,
            )));
        }

        let path_data = &data[paths_offset..paths_end];
        let mut cursor = 0usize;
        while cursor + 4 <= path_data.len() {
            let len = u32::from_le_bytes(
                path_data[cursor..cursor + 4].try_into().unwrap(),
            ) as usize;
            cursor += 4;
            if cursor + len > path_data.len() {
                return Err(CacheError::InvalidFormat(format!(
                    "GCIX path table truncated at offset {}",
                    paths_offset + cursor,
                )));
            }
            let path_str = std::str::from_utf8(&path_data[cursor..cursor + len])
                .map_err(|e| {
                    CacheError::InvalidFormat(format!("GCIX path table invalid UTF-8: {}", e))
                })?;
            paths.push(std::path::PathBuf::from(path_str));
            cursor += len;
        }
    }

    // 6. Parse chunk metadata table (GCIX v3)
    let chunks_offset = header.chunks_offset as usize;
    let chunks_bytes_len = header.chunks_bytes as usize;
    let chunk_meta_size = size_of::<crate::search::content::ChunkMetadata>();

    let chunk_metadata: Option<Vec<crate::search::content::ChunkMetadata>> =
        if header.version >= 3 && chunks_bytes_len > 0 && chunk_meta_size > 0 {
            let chunks_end = chunks_offset + chunks_bytes_len;
            if data.len() < chunks_end {
                return Err(CacheError::InvalidFormat(format!(
                    "GCIX file too small for chunk metadata table: {} bytes, need {}",
                    data.len(),
                    chunks_end,
                )));
            }

            let chunk_count = chunks_bytes_len / chunk_meta_size;
            let mut chunks = Vec::with_capacity(chunk_count);
            for i in 0..chunk_count {
                let offset = chunks_offset + i * chunk_meta_size;
                // SAFETY: ChunkMetadata is #[repr(C)], exactly 32 bytes. We verified
                // the file is large enough above. read_unaligned handles any alignment.
                let cm: crate::search::content::ChunkMetadata = unsafe {
                    std::ptr::read_unaligned(
                        data[offset..].as_ptr()
                            as *const crate::search::content::ChunkMetadata,
                    )
                };
                chunks.push(cm);
            }
            Some(chunks)
        } else {
            None // v2 files or empty chunk table: rebuilt on first search
        };

    // 7. Construct ContentStore from mmap'd data with optional Metal buffer
    let mut store = ContentStore::from_gcix_mmap(
        mmap,
        files,
        content_offset,
        content_bytes,
        device,
        paths,
    );

    // 8. Set chunk metadata if loaded from v3
    if let Some(chunks) = chunk_metadata {
        store.set_chunk_metadata(chunks);
    }

    // 9. Wrap in ContentSnapshot
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
        let header = GcixHeader::new(42, 100_000, 32768, 0xDEAD_BEEF_CAFE_1234, 999, 0, 0, 0, 0);
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
        let header = GcixHeader::new(0, 0, 16384, 0, 0, 0, 0, 0, 0);
        let bytes = header.to_bytes();
        assert_eq!(bytes[0], b'G');
        assert_eq!(bytes[1], b'C');
        assert_eq!(bytes[2], b'I');
        assert_eq!(bytes[3], b'X');
    }

    #[test]
    fn test_gcix_header_rejects_bad_magic() {
        let mut bytes = GcixHeader::new(0, 0, 16384, 0, 0, 0, 0, 0, 0).to_bytes();
        bytes[0] = 0xFF;
        assert!(GcixHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_gcix_header_rejects_bad_version() {
        let mut bytes = GcixHeader::new(0, 0, 16384, 0, 0, 0, 0, 0, 0).to_bytes();
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        // Recompute CRC so only version check triggers
        let crc = crc32fast::hash(&bytes[..CRC32_OFFSET]);
        bytes[CRC32_OFFSET..CRC32_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
        let err = GcixHeader::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("version"), "error: {err}");
    }

    #[test]
    fn test_gcix_header_rejects_bad_crc() {
        let mut bytes = GcixHeader::new(10, 1000, 32768, 0x1234, 0, 0, 0, 0, 0).to_bytes();
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

    // ================================================================
    // GCIX v2 path table tests
    // ================================================================

    #[test]
    fn test_save_load_gcix_path_roundtrip() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("paths.gcix");

        // Build a content store with paths
        let mut store = ContentStore::new();
        store.add_file_with_path(
            b"hello world",
            std::path::PathBuf::from("/Users/test/src/main.rs"),
            0,
            0xAAAA,
            1000,
        );
        store.add_file_with_path(
            b"fn search() {}",
            std::path::PathBuf::from("/Users/test/src/lib.rs"),
            1,
            0xBBBB,
            2000,
        );
        store.add_file_with_path(
            b"# README",
            std::path::PathBuf::from("/Users/test/README.md"),
            2,
            0xCCCC,
            3000,
        );

        save_gcix(&store, &gcix_path, 0x1234, 42).expect("save should succeed");

        // Load and verify paths survive roundtrip
        let snapshot = load_gcix(&gcix_path, None).expect("load should succeed");
        let loaded = snapshot.content_store();

        assert_eq!(loaded.file_count(), 3);

        // Verify paths
        assert_eq!(
            loaded.path_for(0).unwrap().to_str().unwrap(),
            "/Users/test/src/main.rs"
        );
        assert_eq!(
            loaded.path_for(1).unwrap().to_str().unwrap(),
            "/Users/test/src/lib.rs"
        );
        assert_eq!(
            loaded.path_for(2).unwrap().to_str().unwrap(),
            "/Users/test/README.md"
        );

        // Out of bounds
        assert!(loaded.path_for(3).is_none());

        // Verify content still works
        assert_eq!(loaded.content_for(0).unwrap(), b"hello world");
        assert_eq!(loaded.content_for(1).unwrap(), b"fn search() {}");
        assert_eq!(loaded.content_for(2).unwrap(), b"# README");
    }

    #[test]
    fn test_save_load_gcix_empty_paths() {
        // Store with NO paths should still roundtrip (paths_bytes = 0)
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("no_paths.gcix");

        let mut store = ContentStore::new();
        store.insert(b"content", 0, 0, 0);

        save_gcix(&store, &gcix_path, 0, 0).expect("save");

        let snapshot = load_gcix(&gcix_path, None).expect("load");
        let loaded = snapshot.content_store();

        assert_eq!(loaded.file_count(), 1);
        assert!(loaded.path_for(0).is_none()); // No paths stored
        assert_eq!(loaded.content_for(0).unwrap(), b"content");
    }

    #[test]
    fn test_save_load_gcix_paths_with_metal() {
        use objc2_metal::MTLCreateSystemDefaultDevice;

        let device = MTLCreateSystemDefaultDevice()
            .expect("No Metal device (test requires Apple Silicon)");

        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("paths_metal.gcix");

        let mut store = ContentStore::new();
        store.add_file_with_path(
            b"GPU content",
            std::path::PathBuf::from("/src/gpu.rs"),
            0,
            0xAA,
            100,
        );
        store.add_file_with_path(
            b"Metal content",
            std::path::PathBuf::from("/src/metal.rs"),
            1,
            0xBB,
            200,
        );

        save_gcix(&store, &gcix_path, 0xFACE, 7).expect("save");

        let snapshot = load_gcix(&gcix_path, Some(&device)).expect("load with device");
        let loaded = snapshot.content_store();

        assert!(loaded.has_metal_buffer());
        assert_eq!(loaded.path_for(0).unwrap().to_str().unwrap(), "/src/gpu.rs");
        assert_eq!(loaded.path_for(1).unwrap().to_str().unwrap(), "/src/metal.rs");
        assert_eq!(loaded.content_for(0).unwrap(), b"GPU content");
        assert_eq!(loaded.content_for(1).unwrap(), b"Metal content");
    }

    #[test]
    fn test_save_load_gcix_50_files_with_paths() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("50paths.gcix");

        let mut store = ContentStore::new();
        for i in 0..50u32 {
            let content = format!("content_for_file_{:04}", i);
            let path = std::path::PathBuf::from(format!("/project/src/module_{:04}.rs", i));
            store.add_file_with_path(content.as_bytes(), path, i, i * 7, i * 13);
        }

        save_gcix(&store, &gcix_path, 0xBEEF, 123).expect("save");

        let snapshot = load_gcix(&gcix_path, None).expect("load");
        let loaded = snapshot.content_store();

        assert_eq!(loaded.file_count(), 50);

        for i in 0..50u32 {
            let expected_path = format!("/project/src/module_{:04}.rs", i);
            let expected_content = format!("content_for_file_{:04}", i);

            assert_eq!(
                loaded.path_for(i).unwrap().to_str().unwrap(),
                expected_path,
                "path mismatch for file {}",
                i,
            );
            assert_eq!(
                loaded.content_for(i).unwrap(),
                expected_content.as_bytes(),
                "content mismatch for file {}",
                i,
            );
        }
    }

    // ================================================================
    // GCIX v3 chunk metadata persistence tests
    // ================================================================

    /// Helper: compare two ChunkMetadata slices field-by-field
    /// (ChunkMetadata does not derive PartialEq).
    fn chunk_metadata_eq(a: &[crate::search::content::ChunkMetadata], b: &[crate::search::content::ChunkMetadata]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if x.file_index != y.file_index
                || x.chunk_index != y.chunk_index
                || x.offset_in_file != y.offset_in_file
                || x.chunk_length != y.chunk_length
                || x.flags != y.flags
                || x.buffer_offset != y.buffer_offset
            {
                panic!(
                    "ChunkMetadata mismatch at index {}: {:?} vs {:?}",
                    i, x, y
                );
            }
        }
        true
    }

    #[test]
    fn test_gcix_v3_chunk_metadata_roundtrip() {
        // Save a ContentStore to GCIX v3, reload, verify chunk_metadata() matches
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("v3_chunks.gcix");

        let mut store = ContentStore::new();
        store.add_file_with_path(
            b"hello world\nsecond line\nthird line here",
            std::path::PathBuf::from("/src/a.rs"),
            0,
            0xAA,
            1000,
        );
        store.add_file_with_path(
            b"another file with content",
            std::path::PathBuf::from("/src/b.rs"),
            1,
            0xBB,
            2000,
        );

        // Build expected chunk metadata BEFORE saving
        let expected_chunks = build_chunk_metadata(&store);
        assert!(!expected_chunks.is_empty(), "should have chunks");

        save_gcix(&store, &gcix_path, 0xDEAD, 42).expect("save v3");

        // Verify header says v3
        let raw = std::fs::read(&gcix_path).unwrap();
        let header = GcixHeader::from_bytes(&raw).expect("parse header");
        assert_eq!(header.version, 3);
        assert!(header.chunks_bytes > 0, "chunks_bytes should be > 0");

        // Load and verify chunk_metadata is present
        let snapshot = load_gcix(&gcix_path, None).expect("load v3");
        let loaded = snapshot.content_store();

        let loaded_chunks = loaded
            .chunk_metadata()
            .expect("v3 GCIX should have chunk_metadata");

        assert_eq!(
            loaded_chunks.len(),
            expected_chunks.len(),
            "chunk count mismatch"
        );
        assert!(chunk_metadata_eq(loaded_chunks, &expected_chunks));

        // Also verify content and paths survived
        assert_eq!(loaded.content_for(0).unwrap(), b"hello world\nsecond line\nthird line here");
        assert_eq!(loaded.content_for(1).unwrap(), b"another file with content");
        assert_eq!(loaded.path_for(0).unwrap().to_str().unwrap(), "/src/a.rs");
        assert_eq!(loaded.path_for(1).unwrap().to_str().unwrap(), "/src/b.rs");
    }

    #[test]
    fn test_gcix_v3_chunk_metadata_large_files() {
        // Test with files larger than CHUNK_SIZE to produce multiple chunks per file
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("v3_large.gcix");

        let mut store = ContentStore::new();
        // File 0: 10000 bytes = 3 chunks (4096 + 4096 + 1808)
        let big_content: Vec<u8> = (0..10000u32).map(|i| (i % 256) as u8).collect();
        store.add_file_with_path(
            &big_content,
            std::path::PathBuf::from("/big.bin"),
            0,
            0xCC,
            3000,
        );
        // File 1: small file = 1 chunk
        store.add_file_with_path(
            b"small",
            std::path::PathBuf::from("/small.txt"),
            1,
            0xDD,
            4000,
        );

        let expected_chunks = build_chunk_metadata(&store);
        // 3 chunks for big file + 1 for small = 4
        assert_eq!(expected_chunks.len(), 3 + 1);

        save_gcix(&store, &gcix_path, 0xBEEF, 99).expect("save");

        let snapshot = load_gcix(&gcix_path, None).expect("load");
        let loaded = snapshot.content_store();
        let loaded_chunks = loaded.chunk_metadata().expect("should have chunks");

        assert!(chunk_metadata_eq(loaded_chunks, &expected_chunks));

        // Verify per-chunk fields for multi-chunk file
        assert_eq!(loaded_chunks[0].file_index, 0);
        assert_eq!(loaded_chunks[0].chunk_index, 0);
        assert_eq!(loaded_chunks[0].offset_in_file, 0);
        assert_eq!(loaded_chunks[0].chunk_length, 4096);
        assert_eq!(loaded_chunks[0].flags & 2, 2); // is_first

        assert_eq!(loaded_chunks[1].file_index, 0);
        assert_eq!(loaded_chunks[1].chunk_index, 1);
        assert_eq!(loaded_chunks[1].offset_in_file, 4096);
        assert_eq!(loaded_chunks[1].chunk_length, 4096);

        assert_eq!(loaded_chunks[2].file_index, 0);
        assert_eq!(loaded_chunks[2].chunk_index, 2);
        assert_eq!(loaded_chunks[2].offset_in_file, 8192);
        assert_eq!(loaded_chunks[2].chunk_length, 10000 - 8192);
        assert_eq!(loaded_chunks[2].flags & 4, 4); // is_last
    }

    #[test]
    fn test_gcix_v3_empty_store_no_chunks() {
        // Empty store should produce empty chunk metadata (but still be v3)
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("v3_empty.gcix");

        let store = ContentStore::new();
        save_gcix(&store, &gcix_path, 0, 0).expect("save empty v3");

        let raw = std::fs::read(&gcix_path).unwrap();
        let header = GcixHeader::from_bytes(&raw).expect("parse header");
        assert_eq!(header.version, 3);
        assert_eq!(header.chunks_bytes, 0, "empty store should have 0 chunk bytes");

        let snapshot = load_gcix(&gcix_path, None).expect("load empty v3");
        let loaded = snapshot.content_store();
        // chunks_bytes == 0, so load_gcix won't set chunk_metadata
        assert!(
            loaded.chunk_metadata().is_none(),
            "empty store should have None chunk_metadata"
        );
    }

    #[test]
    fn test_gcix_v2_backward_compat_no_chunk_metadata() {
        // Create a v3 GCIX file, then manually downgrade to v2 format:
        // - Set version to 2
        // - Zero out chunks_offset/chunks_bytes fields (bytes 76..92)
        // - Move CRC to offset 76 (v2 CRC position)
        // - Recompute CRC over first 76 bytes
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("v2_compat.gcix");

        let mut store = ContentStore::new();
        store.add_file_with_path(
            b"v2 content here",
            std::path::PathBuf::from("/v2/file.rs"),
            0,
            0xAA,
            500,
        );
        save_gcix(&store, &gcix_path, 0x1234, 10).expect("save v3");

        // Manually patch to v2 format
        let mut data = std::fs::read(&gcix_path).unwrap();

        // Set version to 2
        data[4..8].copy_from_slice(&2u32.to_le_bytes());

        // Zero the v3 chunk fields at [76..92)
        data[76..92].fill(0);

        // Recompute CRC over [0..76) and write at [76..80) (v2 CRC offset)
        let crc = crc32fast::hash(&data[..76]);
        data[76..80].copy_from_slice(&crc.to_le_bytes());

        std::fs::write(&gcix_path, &data).unwrap();

        // Load as v2 — should succeed with no chunk metadata
        let snapshot = load_gcix(&gcix_path, None).expect("load v2 file");
        let loaded = snapshot.content_store();

        assert_eq!(loaded.file_count(), 1);
        assert_eq!(loaded.content_for(0).unwrap(), b"v2 content here");
        assert!(
            loaded.chunk_metadata().is_none(),
            "v2 GCIX should have None chunk_metadata"
        );
    }

    #[test]
    fn test_gcix_v3_build_save_load_compare() {
        // Full round-trip: build_chunk_metadata -> save_gcix -> load_gcix -> compare
        let dir = tempfile::TempDir::new().expect("tempdir");
        let gcix_path = dir.path().join("v3_full_rt.gcix");

        let mut store = ContentStore::new();
        // Multiple files with varying sizes to exercise chunking edge cases
        let contents: Vec<Vec<u8>> = vec![
            b"short".to_vec(),                                // 5 bytes = 1 chunk
            vec![b'A'; 4096],                                 // exactly CHUNK_SIZE = 1 chunk
            vec![b'B'; 4097],                                 // CHUNK_SIZE+1 = 2 chunks
            vec![b'C'; 8192],                                 // 2*CHUNK_SIZE = 2 chunks
            b"line1\nline2\nline3\nline4\nline5\n".to_vec(),  // multi-line
        ];

        for (i, content) in contents.iter().enumerate() {
            let path = std::path::PathBuf::from(format!("/project/file_{}.txt", i));
            store.add_file_with_path(content, path, i as u32, (i as u32) * 0x11, (i as u32) * 100);
        }

        // Step 1: build chunk metadata from store
        let pre_save_chunks = build_chunk_metadata(&store);

        // Verify expected chunk counts:
        // file0: 5 bytes -> 1 chunk
        // file1: 4096 bytes -> 1 chunk
        // file2: 4097 bytes -> 2 chunks
        // file3: 8192 bytes -> 2 chunks
        // file4: 31 bytes -> 1 chunk
        assert_eq!(pre_save_chunks.len(), 1 + 1 + 2 + 2 + 1, "expected 7 chunks total");

        // Step 2: save GCIX v3
        save_gcix(&store, &gcix_path, 0xCAFE, 77).expect("save");

        // Step 3: load GCIX v3
        let snapshot = load_gcix(&gcix_path, None).expect("load");
        let loaded = snapshot.content_store();

        // Step 4: compare chunk metadata
        let loaded_chunks = loaded
            .chunk_metadata()
            .expect("loaded store should have chunk_metadata");

        assert_eq!(
            loaded_chunks.len(),
            pre_save_chunks.len(),
            "chunk count should match"
        );
        assert!(chunk_metadata_eq(loaded_chunks, &pre_save_chunks));

        // Verify content integrity too
        for (i, content) in contents.iter().enumerate() {
            assert_eq!(
                loaded.content_for(i as u32).unwrap(),
                content.as_slice(),
                "content mismatch for file {}",
                i,
            );
        }
    }
}
