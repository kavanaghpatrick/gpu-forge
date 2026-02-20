//! Zero-copy mmap buffers for GPU access.
//!
//! Memory-maps files and wraps them as Metal buffers via
//! `newBufferWithBytesNoCopy` for true zero-copy GPU access on
//! Apple Silicon unified memory.
//!
//! Traditional: File -> CPU Read -> CPU Buffer -> GPU Copy -> GPU
//! Zero-Copy:   File -> mmap -> newBufferWithBytesNoCopy -> GPU (same memory!)

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
/// Required alignment for `makeBuffer(bytesNoCopy:)`.
pub const PAGE_SIZE: usize = 16384;

/// Round a size up to the next page boundary.
#[inline]
pub fn align_to_page(size: usize) -> usize {
    (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// A memory-mapped file exposed as a Metal buffer with zero copies.
///
/// # Example
/// ```ignore
/// let mmap = MmapBuffer::from_file("index.bin")?;
/// let buffer = mmap.as_metal_buffer(&device);
/// // GPU can now read file data directly - no copies!
/// encoder.set_buffer(0, Some(&*buffer), 0);
/// ```
///
/// # Safety
/// The mmap and Metal buffer share the same physical memory.
/// The Metal buffer must not outlive the MmapBuffer.
pub struct MmapBuffer {
    /// Raw pointer to mmap'd memory (kept for munmap on drop)
    ptr: *mut c_void,
    /// Size of mmap'd region (page-aligned)
    mapped_len: usize,
    /// Original file size (not page-aligned)
    file_size: usize,
    /// Keep file open to maintain mmap
    _file: File,
}

impl std::fmt::Debug for MmapBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapBuffer")
            .field("ptr", &self.ptr)
            .field("file_size", &self.file_size)
            .field("mapped_len", &self.mapped_len)
            .finish()
    }
}

// SAFETY: The mmap'd memory is read-only (PROT_READ) and backed by a file.
// Multiple threads can safely read from the mapping.
unsafe impl Send for MmapBuffer {}
unsafe impl Sync for MmapBuffer {}

impl MmapBuffer {
    /// Create a zero-copy buffer by memory-mapping a file.
    ///
    /// The file is mapped with `MAP_PRIVATE | PROT_READ` and 16KB page alignment.
    /// After mapping, `madvise(MADV_WILLNEED)` prefetches pages.
    ///
    /// # Errors
    /// Returns an error if the file doesn't exist, is empty, or mmap fails.
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();

        let file = File::open(path)?;
        let file_size = file.metadata()?.len() as usize;

        if file_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Cannot mmap empty file",
            ));
        }

        // Round up to page boundary for mmap alignment
        let mapped_len = align_to_page(file_size);

        // SAFETY: We pass valid fd, non-zero length, and MAP_PRIVATE+PROT_READ.
        // MAP_FAILED is checked below.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // Prefetch pages to reduce page faults during GPU access
        // SAFETY: ptr is a valid mmap'd region of mapped_len bytes.
        unsafe {
            libc::madvise(ptr, mapped_len, libc::MADV_WILLNEED);
        }

        Ok(Self {
            ptr,
            mapped_len,
            file_size,
            _file: file,
        })
    }

    /// Create a zero-copy buffer from raw bytes (for testing).
    ///
    /// Allocates page-aligned memory via anonymous mmap, copies data in,
    /// then makes it read-only.
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Cannot create buffer from empty data",
            ));
        }

        let file_size = data.len();
        let mapped_len = align_to_page(file_size);

        // Allocate page-aligned memory using anonymous mmap
        // SAFETY: No file descriptor needed for MAP_ANON.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // Copy data into mmap'd region
        // SAFETY: ptr is valid for mapped_len bytes, data is valid for file_size bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, file_size);
        }

        // Make it read-only
        // SAFETY: ptr is a valid mmap'd region.
        unsafe {
            libc::mprotect(ptr, mapped_len, libc::PROT_READ);
        }

        // Dummy file handle (from_bytes is mainly for testing)
        let dummy_file = File::open("/dev/null")?;

        Ok(Self {
            ptr,
            mapped_len,
            file_size,
            _file: dummy_file,
        })
    }

    /// Wrap the mmap'd region as a Metal buffer via `bytesNoCopy` (zero-copy).
    ///
    /// On Apple Silicon UMA, the mmap'd virtual address is directly GPU-accessible,
    /// so `bytesNoCopy` avoids any data copying. If `bytesNoCopy` fails (returns None),
    /// falls back to `newBufferWithBytes` which copies the data.
    ///
    /// The deallocator is `None` because MmapBuffer owns the mapping and will
    /// munmap on drop. The Metal buffer must not outlive this MmapBuffer.
    pub fn as_metal_buffer(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let options = MTLResourceOptions::StorageModeShared;

        // Try zero-copy path first
        // SAFETY: self.ptr is a valid mmap'd region, page-aligned to 16KB.
        // No deallocator because MmapBuffer owns the memory.
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

        // Fallback: copy path (~1ms/GB on Apple Silicon)
        // SAFETY: self.ptr is valid for file_size bytes.
        unsafe {
            let nn = NonNull::new(self.ptr).expect("mmap returned null pointer");
            device
                .newBufferWithBytes_length_options(nn, self.file_size, options)
                .expect("Failed to create Metal buffer (copy fallback)")
        }
    }

    /// Get the original file size (not page-aligned).
    #[inline]
    pub fn file_size(&self) -> usize {
        self.file_size
    }

    /// Get the aligned buffer size (multiple of PAGE_SIZE).
    #[inline]
    pub fn mapped_len(&self) -> usize {
        self.mapped_len
    }

    /// Get raw pointer to mmap'd data.
    ///
    /// # Safety
    /// The pointer is valid only while MmapBuffer is alive.
    /// Data beyond file_size() may be uninitialized.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// Get the mmap'd content as a byte slice (file_size bytes).
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for mapped_len bytes (>= file_size),
        // mapping is read-only, region valid for lifetime of self.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.file_size) }
    }

    /// Advise the kernel about sequential access pattern.
    pub fn advise_sequential(&self) {
        unsafe {
            libc::madvise(self.ptr, self.mapped_len, libc::MADV_SEQUENTIAL);
        }
    }

    /// Advise the kernel that we'll need this data soon.
    pub fn advise_willneed(&self) {
        unsafe {
            libc::madvise(self.ptr, self.mapped_len, libc::MADV_WILLNEED);
        }
    }

    /// Advise the kernel about random access pattern.
    pub fn advise_random(&self) {
        unsafe {
            libc::madvise(self.ptr, self.mapped_len, libc::MADV_RANDOM);
        }
    }
}

impl Drop for MmapBuffer {
    fn drop(&mut self) {
        // SAFETY: self.ptr was returned by a successful mmap() call with
        // self.mapped_len bytes. We only call munmap once (in Drop).
        unsafe {
            libc::munmap(self.ptr, self.mapped_len);
        }
        // File is closed automatically when _file is dropped
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
    fn test_align_to_page() {
        assert_eq!(align_to_page(0), 0);
        assert_eq!(align_to_page(1), PAGE_SIZE);
        assert_eq!(align_to_page(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(align_to_page(PAGE_SIZE + 1), 2 * PAGE_SIZE);
    }

    #[test]
    fn test_mmap_buffer_basic() {
        let content = b"hello mmap world\n";
        let tmp = make_temp_file(content);

        let mmap = MmapBuffer::from_file(tmp.path()).expect("mmap failed");
        assert_eq!(mmap.file_size(), content.len());
        assert_eq!(mmap.mapped_len(), PAGE_SIZE);
        assert_eq!(mmap.as_slice(), content.as_slice());
    }

    #[test]
    fn test_mmap_buffer_empty_file_error() {
        let tmp = NamedTempFile::new().expect("Failed to create temp file");
        let result = MmapBuffer::from_file(tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_buffer_from_bytes() {
        let data = b"test data for from_bytes";
        let mmap = MmapBuffer::from_bytes(data).expect("from_bytes failed");
        assert_eq!(mmap.file_size(), data.len());
        assert_eq!(mmap.as_slice(), data.as_slice());
    }

    #[test]
    fn test_mmap_buffer() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        // Create a file with known pattern
        let mut content = Vec::with_capacity(8192);
        for i in 0u16..4096 {
            content.extend_from_slice(&i.to_le_bytes());
        }
        let tmp = make_temp_file(&content);

        // mmap the file
        let mmap = MmapBuffer::from_file(tmp.path()).expect("mmap failed");
        assert_eq!(mmap.file_size(), content.len());
        assert_eq!(mmap.as_slice(), content.as_slice());

        // Create Metal buffer (zero-copy)
        let buffer = mmap.as_metal_buffer(&device);

        // Buffer length should be at least the file size
        assert!(buffer.length() >= mmap.file_size());

        // Verify Metal buffer contents match the original file
        // SAFETY: Buffer was just created, no GPU work pending.
        // StorageModeShared guarantees CPU-visible pointer.
        unsafe {
            let buf_ptr = buffer.contents().as_ptr() as *const u8;
            let buf_slice = std::slice::from_raw_parts(buf_ptr, content.len());
            assert_eq!(
                buf_slice,
                content.as_slice(),
                "Metal buffer contents mismatch"
            );
        }

        // Spot-check specific u16 values
        let slice = mmap.as_slice();
        for i in 0u16..4096 {
            let offset = i as usize * 2;
            let val = u16::from_le_bytes([slice[offset], slice[offset + 1]]);
            assert_eq!(val, i, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_mmap_buffer_multi_page() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        // File larger than one page
        let content = vec![0xDEu8; PAGE_SIZE + 500];
        let tmp = make_temp_file(&content);

        let mmap = MmapBuffer::from_file(tmp.path()).expect("mmap failed");
        assert_eq!(mmap.mapped_len(), PAGE_SIZE * 2);

        let buffer = mmap.as_metal_buffer(&device);
        assert!(buffer.length() >= content.len());

        unsafe {
            let buf_ptr = buffer.contents().as_ptr() as *const u8;
            let buf_slice = std::slice::from_raw_parts(buf_ptr, content.len());
            assert_eq!(buf_slice, content.as_slice());
        }
    }

    #[test]
    fn test_mmap_buffer_advise() {
        let content = b"advise test data";
        let tmp = make_temp_file(content);
        let mmap = MmapBuffer::from_file(tmp.path()).expect("mmap failed");

        // These should not panic
        mmap.advise_sequential();
        mmap.advise_willneed();
        mmap.advise_random();
    }
}
