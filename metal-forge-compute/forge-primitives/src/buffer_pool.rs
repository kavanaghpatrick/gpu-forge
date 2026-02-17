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
}
