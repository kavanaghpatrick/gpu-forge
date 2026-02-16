//! Zero-copy Metal buffer creation for GSIX v2 index files.
//!
//! Creates Metal buffers from mmap'd GSIX v2 index data using
//! `newBufferWithBytesNoCopy` for true zero-copy GPU access on Apple Silicon.
//! The buffer covers the full mmap region; GPU dispatch calls should use
//! `HEADER_SIZE_V2` (16384) as the buffer offset to skip the header and
//! access entry data directly.
//!
//! Fallback: if `bytesNoCopy` returns None (non-page-aligned pointer or
//! driver rejection), copies data via `newBufferWithBytes`.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::index::gsix_v2::HEADER_SIZE_V2;
use crate::io::mmap::PAGE_SIZE;

/// Create a GPU buffer from an mmap'd region for GSIX v2 index access.
///
/// Strategy A: mmap full file, call `bytesNoCopy` on the full mmap region.
/// GPU dispatch calls use `HEADER_SIZE_V2` (16384) as the buffer offset to
/// skip the header and access packed 256-byte `GpuPathEntry` records.
///
/// # Arguments
/// * `device` - Metal device to create the buffer on
/// * `mmap_ptr` - Pointer to the start of the mmap'd region (must be page-aligned)
/// * `mmap_len` - Total length of the mmap'd region (must be page-aligned)
/// * `entry_count` - Number of entries after the header (for validation)
///
/// # Returns
/// `Some(buffer)` on success, `None` if both zero-copy and fallback fail.
///
/// # Safety
/// * `mmap_ptr` must be a valid pointer to `mmap_len` bytes of readable memory.
/// * The memory must remain valid for the lifetime of the returned buffer.
/// * `mmap_len` must be page-aligned (multiple of 16384).
pub unsafe fn create_gpu_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    mmap_ptr: *const u8,
    mmap_len: usize,
    entry_count: usize,
) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
    if mmap_ptr.is_null() || mmap_len == 0 {
        return None;
    }

    // Validate that the mmap region is large enough for header + entries
    let required = HEADER_SIZE_V2 + entry_count * 256;
    if mmap_len < required {
        return None;
    }

    let options = MTLResourceOptions::StorageModeShared;

    // Strategy A: try bytesNoCopy on the full mmap region (zero-copy)
    // Requires page-aligned pointer and page-aligned length.
    let is_page_aligned =
        (mmap_ptr as usize).is_multiple_of(PAGE_SIZE) && mmap_len.is_multiple_of(PAGE_SIZE);

    if is_page_aligned {
        // SAFETY: caller guarantees mmap_ptr is valid for mmap_len bytes,
        // page-aligned, and will outlive the returned buffer.
        // No deallocator because the mmap owner manages the memory.
        let buffer = unsafe {
            let nn = NonNull::new(mmap_ptr as *mut u8)
                .expect("mmap_ptr null check already done above");
            let nn_void = nn.cast::<std::ffi::c_void>();
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                nn_void,
                mmap_len,
                options,
                None,
            )
        };

        if let Some(buf) = buffer {
            return Some(buf);
        }
    }

    // Fallback: copy the entry region only (skip header)
    // This avoids copying the 16KB header that the GPU doesn't need.
    let entry_data_len = entry_count * 256;
    if entry_data_len == 0 {
        // No entries to copy; create a minimal 1-byte buffer
        return device.newBufferWithLength_options(256, options);
    }

    // SAFETY: mmap_ptr is valid for mmap_len bytes, and we verified
    // mmap_len >= HEADER_SIZE_V2 + entry_data_len above.
    unsafe {
        let entry_ptr = mmap_ptr.add(HEADER_SIZE_V2);
        let nn = NonNull::new(entry_ptr as *mut u8)
            .expect("entry pointer should be non-null");
        let nn_void = nn.cast::<std::ffi::c_void>();
        device.newBufferWithBytes_length_options(nn_void, entry_data_len, options)
    }
}

/// Check whether a pointer is page-aligned for Metal bytesNoCopy.
///
/// On Apple Silicon, pages are 16KB (16384 bytes). The mmap pointer
/// must be aligned to this boundary for `bytesNoCopy` to succeed.
#[inline]
pub fn is_page_aligned(ptr: *const u8) -> bool {
    (ptr as usize).is_multiple_of(PAGE_SIZE)
}

/// Compute the buffer offset for GPU dispatch to skip the GSIX v2 header.
///
/// When using bytesNoCopy on the full mmap region, GPU dispatch calls
/// should set buffer offset to this value so the shader sees entry data
/// starting at index 0.
#[inline]
pub const fn entry_buffer_offset() -> usize {
    HEADER_SIZE_V2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::types::GpuPathEntry;
    use crate::index::gsix_v2::{save_v2, HEADER_SIZE_V2};
    use crate::io::mmap::MmapBuffer;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)")
    }

    #[test]
    fn test_create_gpu_buffer_from_v2_file() {
        let device = get_device();
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("test.idx");

        // Create a v2 index file with some entries
        let mut entries = Vec::new();
        for i in 0..10 {
            let mut entry = GpuPathEntry::new();
            entry.set_path(format!("/test/file_{}.rs", i).as_bytes());
            entry.flags = i as u32;
            entries.push(entry);
        }
        save_v2(&entries, 0xABCD, &idx_path, 42, 0).unwrap();

        // mmap the file
        let mmap = MmapBuffer::from_file(&idx_path).unwrap();

        // Create GPU buffer
        // SAFETY: mmap is valid and outlives buffer in this test scope
        let buffer = unsafe {
            create_gpu_buffer(
                &device,
                mmap.as_ptr(),
                mmap.mapped_len(),
                entries.len(),
            )
        };
        assert!(buffer.is_some(), "GPU buffer creation should succeed");

        let buf = buffer.unwrap();
        // Buffer should cover the full mmap region (bytesNoCopy path)
        // or at least the entry data (copy path)
        assert!(
            buf.length() >= entries.len() * 256,
            "Buffer must be large enough for entries"
        );
    }

    #[test]
    fn test_entry_buffer_offset() {
        assert_eq!(entry_buffer_offset(), 16384);
        assert_eq!(entry_buffer_offset(), HEADER_SIZE_V2);
    }

    #[test]
    fn test_is_page_aligned() {
        assert!(is_page_aligned(std::ptr::null()));
        assert!(is_page_aligned(0x4000 as *const u8));
        assert!(is_page_aligned(0x8000 as *const u8));
        assert!(!is_page_aligned(0x4001 as *const u8));
        assert!(!is_page_aligned(0x100 as *const u8));
    }

    #[test]
    fn test_null_pointer_returns_none() {
        let device = get_device();
        // SAFETY: null pointer is handled as early return in create_gpu_buffer
        let result = unsafe { create_gpu_buffer(&device, std::ptr::null(), 0, 0) };
        assert!(result.is_none());
    }

    #[test]
    fn test_buffer_contents_match_entries() {
        let device = get_device();
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("verify.idx");

        let mut entries = Vec::new();
        for i in 0..5 {
            let mut entry = GpuPathEntry::new();
            entry.set_path(format!("/verify/{}.txt", i).as_bytes());
            entry.set_size(1024 * (i as u64 + 1));
            entries.push(entry);
        }
        save_v2(&entries, 0xFF, &idx_path, 0, 0).unwrap();

        let mmap = MmapBuffer::from_file(&idx_path).unwrap();
        // SAFETY: mmap is valid and outlives buffer in this test scope
        let buffer = unsafe {
            create_gpu_buffer(
                &device,
                mmap.as_ptr(),
                mmap.mapped_len(),
                entries.len(),
            )
        }
        .expect("buffer creation should succeed");

        // Verify buffer data at the entry offset matches original entries
        unsafe {
            let buf_ptr = buffer.contents().as_ptr() as *const u8;
            // If bytesNoCopy succeeded, entries start at HEADER_SIZE_V2 offset
            // If copy fallback, entries start at offset 0
            let entry_start = if buffer.length() >= mmap.mapped_len() {
                // bytesNoCopy path: full mmap in buffer
                buf_ptr.add(HEADER_SIZE_V2)
            } else {
                // copy path: only entry data
                buf_ptr
            };

            for (i, orig) in entries.iter().enumerate() {
                let orig_bytes: &[u8; 256] =
                    &*(orig as *const GpuPathEntry as *const [u8; 256]);
                let buf_entry =
                    std::slice::from_raw_parts(entry_start.add(i * 256), 256);
                assert_eq!(
                    buf_entry, orig_bytes,
                    "entry {} in GPU buffer should match original",
                    i
                );
            }
        }
    }

    #[test]
    fn test_mmap_fallback_to_copy() {
        // When bytesNoCopy cannot be used (non-page-aligned pointer),
        // create_gpu_buffer falls back to a copy-based buffer.
        // We simulate this by passing a non-page-aligned pointer.
        let device = get_device();
        let dir = tempfile::TempDir::new().unwrap();
        let idx_path = dir.path().join("fallback.idx");

        let mut entries = Vec::new();
        for i in 0..4 {
            let mut entry = GpuPathEntry::new();
            entry.set_path(format!("/fallback/f_{}.rs", i).as_bytes());
            entry.set_size(256 * (i as u64 + 1));
            entries.push(entry);
        }
        save_v2(&entries, 0xEE, &idx_path, 0, 0).unwrap();

        // Read file into a heap Vec (NOT page-aligned) to force copy path
        let file_data = std::fs::read(&idx_path).unwrap();
        // Ensure we have a non-page-aligned pointer by offsetting by 1
        // Allocate a buffer 1 byte larger, use ptr+1 as the "mmap" pointer
        let mut padded = vec![0u8; file_data.len() + 1];
        padded[1..].copy_from_slice(&file_data);
        let misaligned_ptr = unsafe { padded.as_ptr().add(1) };

        assert!(
            !is_page_aligned(misaligned_ptr),
            "Pointer should be non-page-aligned for fallback test"
        );

        // The misaligned pointer + misaligned length means bytesNoCopy will
        // fail, forcing the copy fallback path.
        // SAFETY: padded buffer is valid and lives for this scope.
        let buffer = unsafe {
            create_gpu_buffer(
                &device,
                misaligned_ptr,
                file_data.len(), // not page-aligned length either
                entries.len(),
            )
        };

        assert!(
            buffer.is_some(),
            "Fallback copy path should still produce a valid GPU buffer"
        );

        let buf = buffer.unwrap();
        // Copy path creates buffer with only entry data (no header),
        // so buffer length should be entry_count * 256
        let entry_data_len = entries.len() * 256;
        assert!(
            buf.length() >= entry_data_len,
            "Fallback buffer should cover all entries: buf_len={}, expected>={}",
            buf.length(),
            entry_data_len
        );

        // Verify the copy-path buffer contents match the original entries
        unsafe {
            let buf_ptr = buf.contents().as_ptr() as *const u8;
            // Copy path: entries start at offset 0 (header was skipped)
            for (i, orig) in entries.iter().enumerate() {
                let orig_bytes: &[u8; 256] =
                    &*(orig as *const GpuPathEntry as *const [u8; 256]);
                let buf_entry =
                    std::slice::from_raw_parts(buf_ptr.add(i * 256), 256);
                assert_eq!(
                    buf_entry, orig_bytes,
                    "Fallback copy entry {} should match original",
                    i
                );
            }
        }
    }
}
