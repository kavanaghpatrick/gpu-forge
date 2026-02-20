//! GPU-Direct Storage with MTLIOCommandQueue (native objc2-metal bindings).
//!
//! THE GPU IS THE COMPUTER. File IO bypasses CPU entirely:
//!
//! Traditional: App -> CPU Read -> Memory -> GPU Copy -> GPU
//! GPU-Direct:  App -> MTLIOCommandQueue -> GPU Buffer (CPU not involved!)
//!
//! Metal 3's MTLIOCommandQueue provides:
//! - Direct file-to-GPU-buffer transfers
//! - Async IO with event synchronization
//! - Priority-based scheduling
//! - Zero CPU involvement during transfer
//!
//! ## Binding Status
//!
//! All calls use native `objc2-metal` 0.3 bindings. Zero raw `msg_send!` calls.
//! Types used from objc2-metal:
//! - `MTLIOCommandQueueDescriptor` (class)
//! - `MTLIOCommandQueue` (protocol)
//! - `MTLIOCommandBuffer` (protocol)
//! - `MTLIOFileHandle` (protocol)
//! - `MTLIOPriority`, `MTLIOCommandQueueType`, `MTLIOStatus` (enums)
//! - `MTLDevice::newIOCommandQueueWithDescriptor_error`
//! - `MTLDevice::newIOFileHandleWithURL_error`

use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLIOCommandBuffer, MTLIOCommandQueue,
    MTLIOCommandQueueDescriptor, MTLIOCommandQueueType, MTLIOFileHandle, MTLIOPriority,
    MTLIOStatus, MTLResourceOptions,
};

/// Priority levels for IO operations (maps to MTLIOPriority).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IOPriority {
    High,
    Normal,
    Low,
}

impl IOPriority {
    fn to_mtl(self) -> MTLIOPriority {
        match self {
            IOPriority::High => MTLIOPriority::High,
            IOPriority::Normal => MTLIOPriority::Normal,
            IOPriority::Low => MTLIOPriority::Low,
        }
    }
}

/// Queue type for IO operations (maps to MTLIOCommandQueueType).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IOQueueType {
    Concurrent,
    Serial,
}

impl IOQueueType {
    fn to_mtl(self) -> MTLIOCommandQueueType {
        match self {
            IOQueueType::Concurrent => MTLIOCommandQueueType::Concurrent,
            IOQueueType::Serial => MTLIOCommandQueueType::Serial,
        }
    }
}

/// Status of an IO command buffer (maps to MTLIOStatus).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IOStatus {
    Pending,
    Cancelled,
    Error,
    Complete,
}

impl From<MTLIOStatus> for IOStatus {
    fn from(s: MTLIOStatus) -> Self {
        if s == MTLIOStatus::Pending {
            IOStatus::Pending
        } else if s == MTLIOStatus::Cancelled {
            IOStatus::Cancelled
        } else if s == MTLIOStatus::Error {
            IOStatus::Error
        } else {
            IOStatus::Complete
        }
    }
}

/// A GPU-direct IO command queue for file operations.
///
/// Wraps `MTLIOCommandQueue` to enable file reads directly into GPU buffers
/// without CPU involvement. All calls use native objc2-metal bindings.
pub struct GpuIOQueue {
    queue: Retained<ProtocolObject<dyn MTLIOCommandQueue>>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

// SAFETY: MTLIOCommandQueue protocol is Send + Sync in objc2-metal bindings.
// The device reference is also thread-safe.
unsafe impl Send for GpuIOQueue {}
unsafe impl Sync for GpuIOQueue {}

impl GpuIOQueue {
    /// Create a new GPU IO queue.
    ///
    /// Returns `Err` with description if the device doesn't support MTLIOCommandQueue
    /// (pre-Metal 3) or creation fails.
    pub fn new(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        priority: IOPriority,
        queue_type: IOQueueType,
    ) -> Result<Self, String> {
        let desc = MTLIOCommandQueueDescriptor::new();
        desc.setPriority(priority.to_mtl());
        desc.setType(queue_type.to_mtl());

        let queue = device
            .newIOCommandQueueWithDescriptor_error(&desc)
            .map_err(|e| format!("IOCommandQueue creation failed: {}", e))?;

        Ok(Self {
            queue,
            device: device.clone(),
        })
    }

    /// Create a new IO command buffer.
    pub fn command_buffer(&self) -> Retained<ProtocolObject<dyn MTLIOCommandBuffer>> {
        self.queue.commandBuffer()
    }

    /// Insert a barrier -- all commands before must complete before commands after.
    pub fn enqueue_barrier(&self) {
        self.queue.enqueueBarrier();
    }

    /// Get the underlying device.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
}

/// A file handle for GPU-direct IO.
///
/// Wraps `MTLIOFileHandle` using native objc2-metal bindings.
pub struct GpuIOFileHandle {
    handle: Retained<ProtocolObject<dyn MTLIOFileHandle>>,
}

// SAFETY: MTLIOFileHandle protocol is Send + Sync in objc2-metal bindings.
unsafe impl Send for GpuIOFileHandle {}
unsafe impl Sync for GpuIOFileHandle {}

impl GpuIOFileHandle {
    /// Open a file for GPU-direct reading.
    ///
    /// Returns `Err` if the file doesn't exist or can't be opened for IO.
    pub fn open(
        device: &ProtocolObject<dyn MTLDevice>,
        path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("Invalid UTF-8 path: {:?}", path))?;

        let ns_path = NSString::from_str(path_str);
        let url = NSURL::fileURLWithPath(&ns_path);

        let handle = device
            .newIOFileHandleWithURL_error(&url)
            .map_err(|e| format!("IOFileHandle open failed for {:?}: {}", path, e))?;

        Ok(Self { handle })
    }

    /// Get the inner protocol object for use in IO commands.
    pub fn inner(&self) -> &ProtocolObject<dyn MTLIOFileHandle> {
        &self.handle
    }
}

/// Wrapper around an `MTLIOCommandBuffer` with convenience methods.
///
/// All calls use native objc2-metal bindings.
pub struct GpuIOCommandBuffer {
    buffer: Retained<ProtocolObject<dyn MTLIOCommandBuffer>>,
}

impl GpuIOCommandBuffer {
    /// Wrap a retained IO command buffer.
    pub fn new(buffer: Retained<ProtocolObject<dyn MTLIOCommandBuffer>>) -> Self {
        Self { buffer }
    }

    /// Load data from file directly into a Metal buffer.
    ///
    /// # Safety
    /// The buffer must be large enough for `offset + size` bytes.
    /// The file handle must contain at least `file_offset + size` bytes.
    pub unsafe fn load_buffer(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        size: usize,
        file: &GpuIOFileHandle,
        file_offset: usize,
    ) {
        unsafe {
            self.buffer
                .loadBuffer_offset_size_sourceHandle_sourceHandleOffset(
                    buffer,
                    offset,
                    size,
                    file.inner(),
                    file_offset,
                );
        }
    }

    /// Load data from file directly into raw memory.
    ///
    /// # Safety
    /// The pointer must remain valid until the command buffer completes.
    /// The pointer must point to at least `size` bytes of writable memory.
    pub unsafe fn load_bytes(
        &self,
        ptr: NonNull<c_void>,
        size: usize,
        file: &GpuIOFileHandle,
        file_offset: usize,
    ) {
        unsafe {
            self.buffer
                .loadBytes_size_sourceHandle_sourceHandleOffset(
                    ptr,
                    size,
                    file.inner(),
                    file_offset,
                );
        }
    }

    /// Copy the IO completion status to a buffer (for GPU-side status checking).
    ///
    /// # Safety
    /// The buffer must have at least `offset + 4` bytes.
    pub unsafe fn copy_status_to_buffer(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
    ) {
        unsafe {
            self.buffer.copyStatusToBuffer_offset(buffer, offset);
        }
    }

    /// Add a barrier -- commands before must complete before commands after.
    pub fn add_barrier(&self) {
        self.buffer.addBarrier();
    }

    /// Enqueue the command buffer for execution.
    pub fn enqueue(&self) {
        self.buffer.enqueue();
    }

    /// Commit the command buffer for execution.
    pub fn commit(&self) {
        self.buffer.commit();
    }

    /// Wait until all IO operations complete.
    pub fn wait_until_completed(&self) {
        self.buffer.waitUntilCompleted();
    }

    /// Get the current status of the command buffer.
    pub fn status(&self) -> IOStatus {
        IOStatus::from(self.buffer.status())
    }

    /// Get the error if the command buffer failed.
    pub fn error(&self) -> Option<String> {
        self.buffer.error().map(|e| e.to_string())
    }

    /// Try to cancel pending IO operations.
    pub fn try_cancel(&self) {
        self.buffer.tryCancel();
    }
}

/// High-level: load a file directly into a GPU buffer via MTLIOCommandQueue.
///
/// This is the primary convenience API combining queue, file handle, command buffer.
///
/// Returns `(Retained<ProtocolObject<dyn MTLBuffer>>, u64)` -- the buffer and file size,
/// or an error string.
pub fn load_file_to_gpu(
    queue: &GpuIOQueue,
    path: impl AsRef<Path>,
) -> Result<(Retained<ProtocolObject<dyn MTLBuffer>>, u64), String> {
    let path = path.as_ref();

    // Get file size
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("Cannot stat {:?}: {}", path, e))?;
    let file_size = metadata.len();

    if file_size == 0 {
        return Err("Cannot load empty file via IO queue".into());
    }

    // Open file handle
    let file_handle = GpuIOFileHandle::open(queue.device(), path)?;

    // Create destination buffer (page-aligned for optimal IO)
    let aligned_size = (file_size + 4095) & !4095;
    let buffer = queue
        .device()
        .newBufferWithLength_options(aligned_size as usize, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| "Failed to allocate GPU buffer for IO".to_string())?;

    // Create and execute IO command
    let cmd_buffer = GpuIOCommandBuffer::new(queue.command_buffer());

    // SAFETY: buffer is freshly allocated with aligned_size >= file_size.
    // File handle is valid and contains file_size bytes.
    unsafe {
        cmd_buffer.load_buffer(&buffer, 0, file_size as usize, &file_handle, 0);
    }
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // Check status
    if cmd_buffer.status() != IOStatus::Complete {
        let err_msg = cmd_buffer
            .error()
            .unwrap_or_else(|| "Unknown IO error".to_string());
        return Err(format!("IO load failed: {}", err_msg));
    }

    Ok((buffer, file_size))
}

/// Check if the device supports GPU-direct IO (Metal 3+, Apple Silicon).
pub fn supports_gpu_io(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> bool {
    GpuIOQueue::new(device, IOPriority::Normal, IOQueueType::Concurrent).is_ok()
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
    fn test_io_command_queue() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        // Check support
        let supports = supports_gpu_io(&device);
        println!("GPU-direct IO supported: {}", supports);
        if !supports {
            println!("Skipping IO test -- MTLIOCommandQueue not supported on this device");
            return;
        }

        // Create IO queue
        let queue = GpuIOQueue::new(&device, IOPriority::Normal, IOQueueType::Concurrent)
            .expect("Failed to create IO queue");

        // Create a test file with known content
        let test_data: Vec<u8> = (0u16..4096).flat_map(|i| i.to_le_bytes()).collect();
        let tmp = make_temp_file(&test_data);

        // Load file to GPU via MTLIOCommandQueue
        let (buffer, file_size) =
            load_file_to_gpu(&queue, tmp.path()).expect("load_file_to_gpu failed");

        assert_eq!(file_size, test_data.len() as u64);
        assert!(buffer.length() >= test_data.len());

        // Verify GPU buffer contents match original file
        // SAFETY: buffer was created StorageModeShared, IO is complete, safe to read.
        unsafe {
            let buf_ptr = buffer.contents().as_ptr() as *const u8;
            let buf_slice = std::slice::from_raw_parts(buf_ptr, test_data.len());
            assert_eq!(buf_slice, test_data.as_slice(), "GPU buffer contents mismatch");
        }

        // Spot-check specific u16 values
        unsafe {
            let buf_ptr = buffer.contents().as_ptr() as *const u8;
            for i in 0u16..4096 {
                let offset = i as usize * 2;
                let val = u16::from_le_bytes([
                    *buf_ptr.add(offset),
                    *buf_ptr.add(offset + 1),
                ]);
                assert_eq!(val, i, "Mismatch at index {}", i);
            }
        }

        println!(
            "IO queue test passed: loaded {} bytes to GPU buffer",
            file_size
        );
    }

    #[test]
    fn test_io_queue_creation() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        // Test all priority/type combinations
        for priority in [IOPriority::High, IOPriority::Normal, IOPriority::Low] {
            for queue_type in [IOQueueType::Concurrent, IOQueueType::Serial] {
                let result = GpuIOQueue::new(&device, priority, queue_type);
                if let Ok(queue) = &result {
                    println!(
                        "Queue created: {:?}/{:?} -- device={}",
                        priority,
                        queue_type,
                        queue.device().name()
                    );
                }
                // Some combinations might not be supported on all devices
            }
        }
    }

    #[test]
    fn test_io_file_handle_open() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        if !supports_gpu_io(&device) {
            println!("Skipping -- MTLIOCommandQueue not supported");
            return;
        }

        // Open a real file
        let content = b"test file handle";
        let tmp = make_temp_file(content);

        let handle = GpuIOFileHandle::open(&device, tmp.path());
        assert!(handle.is_ok(), "Failed to open file handle: {:?}", handle.err());

        // Open nonexistent file should fail
        let bad = GpuIOFileHandle::open(&device, "/nonexistent/path/to/file.txt");
        assert!(bad.is_err(), "Should fail for nonexistent file");
    }

    #[test]
    fn test_io_empty_file_error() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        if !supports_gpu_io(&device) {
            return;
        }

        let queue = GpuIOQueue::new(&device, IOPriority::Normal, IOQueueType::Concurrent)
            .expect("Failed to create IO queue");

        // Empty file should return error
        let tmp = NamedTempFile::new().expect("Failed to create temp file");
        let result = load_file_to_gpu(&queue, tmp.path());
        assert!(result.is_err(), "Empty file should fail");
    }

    #[test]
    fn test_io_status_mapping() {
        assert_eq!(IOStatus::from(MTLIOStatus::Pending), IOStatus::Pending);
        assert_eq!(IOStatus::from(MTLIOStatus::Cancelled), IOStatus::Cancelled);
        assert_eq!(IOStatus::from(MTLIOStatus::Error), IOStatus::Error);
        assert_eq!(IOStatus::from(MTLIOStatus::Complete), IOStatus::Complete);
    }

    #[test]
    fn test_io_command_buffer_barrier() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        if !supports_gpu_io(&device) {
            return;
        }

        let queue = GpuIOQueue::new(&device, IOPriority::Normal, IOQueueType::Concurrent)
            .expect("Failed to create IO queue");

        // Create command buffer and test barrier methods (should not panic)
        let cmd = GpuIOCommandBuffer::new(queue.command_buffer());
        cmd.add_barrier();
        cmd.commit();
        cmd.wait_until_completed();

        // Queue barrier should not panic either
        queue.enqueue_barrier();

        println!("Barrier tests passed");
    }

    #[test]
    fn test_io_large_file() {
        let device =
            MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)");

        if !supports_gpu_io(&device) {
            return;
        }

        let queue = GpuIOQueue::new(&device, IOPriority::High, IOQueueType::Concurrent)
            .expect("Failed to create IO queue");

        // Create a ~1MB file
        let size = 1024 * 1024;
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let tmp = make_temp_file(&data);

        let (buffer, file_size) =
            load_file_to_gpu(&queue, tmp.path()).expect("load 1MB file failed");

        assert_eq!(file_size, size as u64);

        // Verify a sample of the data
        unsafe {
            let buf_ptr = buffer.contents().as_ptr() as *const u8;
            for i in (0..size).step_by(4096) {
                let val = *buf_ptr.add(i);
                assert_eq!(val, (i % 256) as u8, "Mismatch at byte {}", i);
            }
        }

        println!("Large file test passed: {} bytes loaded via IO queue", file_size);
    }
}
