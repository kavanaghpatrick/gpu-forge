//! Shared GPU context: device, command queue, buffer allocation.
//!
//! `ForgeContext` owns a single Metal device and command queue that can be
//! shared across forge-sort and forge-filter via `with_context()` constructors.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};

/// Shared GPU context owning a Metal device and command queue.
///
/// Create once, then pass `device()` and `queue()` clones to
/// `GpuSorter::with_context()` and `GpuFilter::with_context()`.
pub struct ForgeContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl ForgeContext {
    /// Initialize Metal: acquire system default device and create a command queue.
    ///
    /// # Panics
    /// Panics if no Metal device is available or queue creation fails.
    pub fn new() -> Self {
        let device =
            MTLCreateSystemDefaultDevice().expect("Failed to get default Metal device");
        let queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");
        Self { device, queue }
    }

    /// Allocate a `StorageModeShared` buffer of `size` bytes.
    ///
    /// # Panics
    /// Panics if the allocation fails (e.g., out of memory).
    pub fn alloc_buffer(&self, size: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let options = MTLResourceOptions::StorageModeShared;
        self.device
            .newBufferWithLength_options(size, options)
            .expect("Failed to allocate Metal buffer")
    }

    /// Clone of the underlying Metal device (ARC bump, cheap).
    pub fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        self.device.clone()
    }

    /// Clone of the underlying command queue (ARC bump, cheap).
    pub fn queue(&self) -> Retained<ProtocolObject<dyn MTLCommandQueue>> {
        self.queue.clone()
    }
}

impl Default for ForgeContext {
    fn default() -> Self {
        Self::new()
    }
}
