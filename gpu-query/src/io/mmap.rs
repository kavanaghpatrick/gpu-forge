//! Memory-mapped file I/O with zero-copy Metal buffer wrapping.
//!
//! Uses `mmap(MAP_SHARED | PROT_READ)` with 16KB page alignment [KB #89] for
//! Apple Silicon. The mmap'd region is wrapped as a Metal buffer via
//! `makeBuffer(bytesNoCopy:)` with `StorageModeShared` for true zero-copy GPU
//! access on unified memory. Falls back to `makeBuffer(bytes:length:options:)`
//! copy path if bytesNoCopy fails.

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// Apple Silicon page size: 16KB (16384 bytes).
/// Required alignment for `makeBuffer(bytesNoCopy:)` [KB #89].
const PAGE_SIZE: usize = 16384;

/// Round `size` up to the next 16KB page boundary.
fn page_align(size: usize) -> usize {
    (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// A memory-mapped file with optional zero-copy Metal buffer wrapping.
///
/// The file is mmap'd read-only with `MAP_SHARED` and `PROT_READ`. The mapped
/// region is page-aligned to 16KB for `bytesNoCopy` compatibility. On drop,
/// the mapping is released via `munmap`.
pub struct MmapFile {
    /// Raw pointer to the mmap'd region.
    ptr: *mut c_void,
    /// Actual file size in bytes.
    file_size: usize,
    /// Page-aligned mapping length (>= file_size, multiple of 16KB).
    mapped_len: usize,
    /// Keep the file open so the mapping stays valid.
    _file: File,
}

impl std::fmt::Debug for MmapFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapFile")
            .field("ptr", &self.ptr)
            .field("file_size", &self.file_size)
            .field("mapped_len", &self.mapped_len)
            .finish()
    }
}

// SAFETY: The mmap'd memory is read-only (PROT_READ) and backed by a file
// (MAP_SHARED). Multiple threads can safely read from the mapping.
unsafe impl Send for MmapFile {}
unsafe impl Sync for MmapFile {}

impl MmapFile {
    /// Open and memory-map a file at `path`.
    ///
    /// The mapping uses `MAP_SHARED | PROT_READ` with 16KB page alignment.
    /// After mapping, `madvise(MADV_WILLNEED)` is called to prefetch pages.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened, is empty, or mmap fails.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path.as_ref())?;
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        if file_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Cannot mmap empty file",
            ));
        }

        let mapped_len = page_align(file_size);
        let fd = file.as_raw_fd();

        // SAFETY: We pass valid fd, non-zero length, and MAP_SHARED+PROT_READ.
        // The file descriptor is valid (just opened). mapped_len > 0 (checked above).
        // MAP_FAILED is checked below.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // Prefetch pages via madvise(MADV_WILLNEED) to reduce page faults
        // during GPU access. Non-fatal if it fails.
        // SAFETY: ptr is a valid mmap'd region of mapped_len bytes.
        unsafe {
            libc::madvise(ptr, mapped_len, libc::MADV_WILLNEED);
        }

        Ok(Self {
            ptr,
            file_size,
            mapped_len,
            _file: file,
        })
    }

    /// The actual file size in bytes (not the page-aligned mapping length).
    pub fn file_size(&self) -> usize {
        self.file_size
    }

    /// The page-aligned mapping length (multiple of 16KB, >= file_size).
    pub fn mapped_len(&self) -> usize {
        self.mapped_len
    }

    /// Raw pointer to the mmap'd region.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// Get the mmap'd content as a byte slice (file_size bytes, not the full mapping).
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for mapped_len bytes (>= file_size), and the
        // mapping is read-only. The region stays valid for the lifetime of self
        // because _file keeps the fd open and we don't munmap until Drop.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.file_size) }
    }

    /// Wrap the mmap'd region as a Metal buffer via `bytesNoCopy` (zero-copy).
    ///
    /// On Apple Silicon UMA, the mmap'd virtual address IS accessible by the
    /// GPU, so `bytesNoCopy` avoids any data copying. If `bytesNoCopy` fails
    /// (returns None), falls back to `newBufferWithBytes` which copies the data.
    ///
    /// The deallocator is set to `None` because the MmapFile owns the mapping
    /// and will munmap on drop. The Metal buffer must not outlive this MmapFile.
    pub fn as_metal_buffer(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let options = MTLResourceOptions::StorageModeShared;

        // Try zero-copy path first: bytesNoCopy wraps the mmap'd pointer directly.
        // SAFETY: self.ptr is a valid mmap'd region of at least mapped_len bytes,
        // page-aligned to 16KB. We pass no deallocator (None) because MmapFile
        // owns the memory. The caller must ensure the MmapFile outlives the buffer.
        let buffer = unsafe {
            let nn = NonNull::new(self.ptr).expect("mmap returned null pointer");
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                nn,
                self.mapped_len,
                options,
                None,
            )
        };

        if let Some(buf) = buffer {
            return buf;
        }

        // Fallback: copy path. This copies the data into a new Metal buffer.
        // ~1ms/GB on Apple Silicon, acceptable for files where bytesNoCopy
        // is not supported (e.g., alignment issues on older hardware).
        // SAFETY: self.ptr is valid for file_size bytes. newBufferWithBytes copies
        // the data, so no lifetime dependency on the mmap.
        unsafe {
            let nn = NonNull::new(self.ptr).expect("mmap returned null pointer");
            device
                .newBufferWithBytes_length_options(nn, self.file_size, options)
                .expect("Failed to create Metal buffer (copy fallback)")
        }
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        // SAFETY: self.ptr was returned by a successful mmap() call with
        // self.mapped_len bytes. We only call munmap once (in Drop).
        unsafe {
            libc::munmap(self.ptr, self.mapped_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper: create a temp file with known content.
    fn make_temp_file(content: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("Failed to create temp file");
        f.write_all(content).expect("Failed to write temp file");
        f.flush().expect("Failed to flush temp file");
        f
    }

    #[test]
    fn test_page_align() {
        assert_eq!(page_align(0), 0);
        assert_eq!(page_align(1), PAGE_SIZE);
        assert_eq!(page_align(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(page_align(PAGE_SIZE + 1), 2 * PAGE_SIZE);
        assert_eq!(page_align(100_000), 7 * PAGE_SIZE); // 114688
    }

    #[test]
    fn test_mmap_file_basic() {
        let content = b"hello mmap world\n";
        let tmp = make_temp_file(content);

        let mmap = MmapFile::open(tmp.path()).expect("mmap failed");
        assert_eq!(mmap.file_size(), content.len());
        assert_eq!(mmap.mapped_len(), PAGE_SIZE); // small file rounds up to 16KB
        assert_eq!(mmap.as_slice(), content);
    }

    #[test]
    fn test_mmap_empty_file_error() {
        let tmp = NamedTempFile::new().expect("Failed to create temp file");
        let result = MmapFile::open(tmp.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_mmap_nonexistent_file_error() {
        let result = MmapFile::open("/tmp/nonexistent_gpu_query_test_file_12345.dat");
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_page_alignment() {
        // File smaller than one page
        let small = vec![42u8; 100];
        let tmp = make_temp_file(&small);
        let mmap = MmapFile::open(tmp.path()).unwrap();
        assert_eq!(mmap.mapped_len() % PAGE_SIZE, 0);
        assert!(mmap.mapped_len() >= mmap.file_size());

        // File exactly one page
        let exact = vec![0xABu8; PAGE_SIZE];
        let tmp = make_temp_file(&exact);
        let mmap = MmapFile::open(tmp.path()).unwrap();
        assert_eq!(mmap.mapped_len(), PAGE_SIZE);
        assert_eq!(mmap.file_size(), PAGE_SIZE);

        // File spanning multiple pages
        let multi = vec![0xCDu8; PAGE_SIZE * 3 + 1];
        let tmp = make_temp_file(&multi);
        let mmap = MmapFile::open(tmp.path()).unwrap();
        assert_eq!(mmap.mapped_len(), PAGE_SIZE * 4);
        assert_eq!(mmap.file_size(), PAGE_SIZE * 3 + 1);
    }

    #[test]
    fn test_mmap_content_matches_file() {
        // Create a file with structured data to verify byte-level correctness
        let mut content = Vec::with_capacity(4096);
        for i in 0u32..1024 {
            content.extend_from_slice(&i.to_le_bytes());
        }
        let tmp = make_temp_file(&content);

        let mmap = MmapFile::open(tmp.path()).unwrap();
        assert_eq!(mmap.as_slice(), content.as_slice());

        // Spot-check specific u32 values
        let slice = mmap.as_slice();
        for i in 0u32..1024 {
            let offset = i as usize * 4;
            let val = u32::from_le_bytes([
                slice[offset],
                slice[offset + 1],
                slice[offset + 2],
                slice[offset + 3],
            ]);
            assert_eq!(val, i, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_mmap_metal_buffer_contents_match() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        // Create a file with known pattern
        let mut content = Vec::with_capacity(8192);
        for i in 0u16..4096 {
            content.extend_from_slice(&i.to_le_bytes());
        }
        let tmp = make_temp_file(&content);

        let mmap = MmapFile::open(tmp.path()).unwrap();
        let buffer = mmap.as_metal_buffer(&device);

        // Buffer length should be the page-aligned mapped length
        assert!(buffer.length() >= mmap.file_size());

        // Verify the Metal buffer contents match the original file content
        // SAFETY: The buffer was just created from our mmap, and no GPU work
        // is pending. StorageModeShared guarantees CPU-visible pointer.
        unsafe {
            let buf_ptr = buffer.contents().as_ptr() as *const u8;
            let buf_slice = std::slice::from_raw_parts(buf_ptr, content.len());
            assert_eq!(
                buf_slice,
                content.as_slice(),
                "Metal buffer contents mismatch"
            );
        }
    }

    #[test]
    fn test_mmap_metal_buffer_larger_file() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        // Create a file larger than one page
        let content = vec![0xDEu8; PAGE_SIZE + 500];
        let tmp = make_temp_file(&content);

        let mmap = MmapFile::open(tmp.path()).unwrap();
        let buffer = mmap.as_metal_buffer(&device);

        assert_eq!(mmap.mapped_len(), PAGE_SIZE * 2);
        assert!(buffer.length() >= content.len());

        // Verify content
        unsafe {
            let buf_ptr = buffer.contents().as_ptr() as *const u8;
            let buf_slice = std::slice::from_raw_parts(buf_ptr, content.len());
            assert_eq!(buf_slice, content.as_slice());
        }
    }

    #[test]
    fn test_mmap_csv_like_content() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        // Simulate a small CSV file
        let csv = b"id,name,amount\n1,alice,100\n2,bob,200\n3,charlie,300\n";
        let tmp = make_temp_file(csv);

        let mmap = MmapFile::open(tmp.path()).unwrap();
        assert_eq!(mmap.as_slice(), csv.as_slice());

        let buffer = mmap.as_metal_buffer(&device);
        unsafe {
            let buf_ptr = buffer.contents().as_ptr() as *const u8;
            let buf_slice = std::slice::from_raw_parts(buf_ptr, csv.len());
            assert_eq!(buf_slice, csv.as_slice());
        }
    }
}
