//! Buffer pool with 16KB page-aligned allocation, recycling, and peak tracking.
//!
//! Apple Silicon page size is 16KB (16384 bytes). Page-aligned buffers
//! enable zero-copy via makeBuffer(bytesNoCopy:).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// 16KB page size on Apple Silicon.
const PAGE_SIZE: usize = 16384;

/// Round up to the nearest 16KB page boundary.
fn page_align(size: usize) -> usize {
    (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// A recycling buffer pool with page-aligned allocations and peak tracking.
pub struct BufferPool {
    /// Available buffers for reuse, keyed by aligned size.
    pool: Vec<(usize, Retained<ProtocolObject<dyn MTLBuffer>>)>,
    /// Current total allocated bytes.
    allocated_bytes: usize,
    /// Peak total allocated bytes.
    peak_bytes: usize,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool {
    /// Create a new empty buffer pool.
    pub fn new() -> Self {
        Self {
            pool: Vec::new(),
            allocated_bytes: 0,
            peak_bytes: 0,
        }
    }

    /// Allocate (or recycle) a page-aligned Metal buffer of at least `size` bytes.
    pub fn alloc(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        size: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let aligned = page_align(size);

        // Try to recycle an existing buffer of matching size.
        if let Some(pos) = self.pool.iter().position(|(s, _)| *s == aligned) {
            let (_, buffer) = self.pool.swap_remove(pos);
            return buffer;
        }

        // Allocate a new page-aligned buffer.
        let options = MTLResourceOptions::StorageModeShared;
        let buffer = device
            .newBufferWithLength_options(aligned, options)
            .expect("Failed to allocate Metal buffer");

        self.allocated_bytes += aligned;
        if self.allocated_bytes > self.peak_bytes {
            self.peak_bytes = self.allocated_bytes;
        }

        buffer
    }

    /// Return a buffer to the pool for later reuse.
    pub fn recycle(&mut self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        let size = buffer.length();
        self.pool.push((size, buffer));
    }

    /// Current total allocated bytes (including recycled).
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Peak total allocated bytes observed.
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Number of buffers currently available for reuse.
    pub fn available_count(&self) -> usize {
        self.pool.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_align() {
        assert_eq!(page_align(0), 0);
        assert_eq!(page_align(1), PAGE_SIZE);
        assert_eq!(page_align(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(page_align(PAGE_SIZE + 1), PAGE_SIZE * 2);
        assert_eq!(page_align(100_000), 114688); // ceil(100000/16384)*16384
    }

    #[test]
    fn test_page_align_specific_values() {
        // From task spec: page_align(1) == 16384, page_align(16384) == 16384, page_align(16385) == 32768
        assert_eq!(page_align(1), 16384);
        assert_eq!(page_align(16384), 16384);
        assert_eq!(page_align(16385), 32768);
    }

    #[test]
    fn test_page_align_large_values() {
        // Test with typical buffer sizes
        assert_eq!(
            page_align(1_000_000),
            1_000_000_usize.next_multiple_of(PAGE_SIZE)
        );
        assert_eq!(page_align(4 * 1024 * 1024), 4 * 1024 * 1024); // 4MB already aligned
    }

    #[test]
    fn test_buffer_pool_new() {
        let pool = BufferPool::new();
        assert_eq!(pool.allocated_bytes(), 0);
        assert_eq!(pool.peak_bytes(), 0);
        assert_eq!(pool.available_count(), 0);
    }

    #[test]
    fn test_buffer_pool_default() {
        let pool = BufferPool::default();
        assert_eq!(pool.allocated_bytes(), 0);
        assert_eq!(pool.peak_bytes(), 0);
        assert_eq!(pool.available_count(), 0);
    }

    #[test]
    fn test_buffer_pool_alloc_recycle() {
        // This test requires a Metal device, so we use MTLCreateSystemDefaultDevice
        let device =
            objc2_metal::MTLCreateSystemDefaultDevice().expect("No Metal device available");

        let mut pool = BufferPool::new();
        assert_eq!(pool.allocated_bytes(), 0);
        assert_eq!(pool.available_count(), 0);

        // Allocate a small buffer (should round up to PAGE_SIZE)
        let buf = pool.alloc(&device, 100);
        assert_eq!(buf.length(), PAGE_SIZE);
        assert_eq!(pool.allocated_bytes(), PAGE_SIZE);
        assert_eq!(pool.peak_bytes(), PAGE_SIZE);
        assert_eq!(pool.available_count(), 0);

        // Recycle it
        pool.recycle(buf);
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.allocated_bytes(), PAGE_SIZE);

        // Re-alloc same size should recycle
        let buf2 = pool.alloc(&device, 100);
        assert_eq!(buf2.length(), PAGE_SIZE);
        assert_eq!(pool.available_count(), 0);
        // allocated_bytes should NOT increase (recycled)
        assert_eq!(pool.allocated_bytes(), PAGE_SIZE);

        // Allocate a larger buffer (2 pages)
        let buf3 = pool.alloc(&device, PAGE_SIZE + 1);
        assert_eq!(buf3.length(), PAGE_SIZE * 2);
        assert_eq!(pool.allocated_bytes(), PAGE_SIZE + PAGE_SIZE * 2);
        assert_eq!(pool.peak_bytes(), PAGE_SIZE + PAGE_SIZE * 2);
    }
}
