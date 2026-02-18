//! Minimal Metal context: device, command queue, shader library.
//!
//! Self-contained — no dependency on forge-primitives.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

/// Core GPU state: device, command queue, shader library.
pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl MetalContext {
    /// Initialize Metal device, command queue, and load compiled shaders.
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");

        let queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");

        let library = Self::load_metallib(&device);

        Self {
            device,
            queue,
            library,
        }
    }

    /// Get the Metal shader library.
    pub fn library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        &self.library
    }

    /// Create a compute pipeline state for the named kernel function.
    pub fn make_pipeline(
        &self,
        name: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let ns_name = NSString::from_str(name);
        let func = self
            .library
            .newFunctionWithName(&ns_name)
            .unwrap_or_else(|| panic!("Kernel function '{name}' not found in metallib"));

        self.device
            .newComputePipelineStateWithFunction_error(&func)
            .unwrap_or_else(|e| panic!("Failed to create PSO for '{name}': {e}"))
    }

    /// Create a new command buffer.
    pub fn command_buffer(&self) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.queue
            .commandBuffer()
            .expect("Failed to create command buffer")
    }

    /// Load the shaders.metallib from the build output directory.
    fn load_metallib(
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Retained<ProtocolObject<dyn MTLLibrary>> {
        let metallib_path = Self::find_metallib()
            .expect("shaders.metallib not found — run `cargo build` first");
        let path_ns = NSString::from_str(&metallib_path);
        #[allow(deprecated)]
        device
            .newLibraryWithFile_error(&path_ns)
            .unwrap_or_else(|e| panic!("Failed to load metallib: {e}"))
    }

    /// Search build output directories for shaders.metallib.
    fn find_metallib() -> Option<String> {
        let exe_path = std::env::current_exe().ok()?;
        let target_dir = exe_path.parent()?;

        let search_dirs = [
            target_dir.to_path_buf(),
            target_dir
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_default(),
        ];

        for dir in &search_dirs {
            let build_dir = dir.join("build");
            if build_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(&build_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let metallib = path.join("out").join("shaders.metallib");
                        if metallib.exists() {
                            return Some(metallib.to_string_lossy().into_owned());
                        }
                    }
                }
            }

            let fallback = dir.join("shaders.metallib");
            if fallback.exists() {
                return Some(fallback.to_string_lossy().into_owned());
            }
        }

        None
    }
}

// ─── Buffer helpers ─────────────────────────────────────────────

/// Allocate an uninitialized Metal buffer of `size` bytes (StorageModeShared).
pub fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    size: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
        .expect("Failed to allocate Metal buffer")
}

/// Allocate a Metal buffer initialized with the given data.
pub fn alloc_buffer_with_data<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let size = std::mem::size_of_val(data);
    unsafe {
        let ptr =
            NonNull::new(data.as_ptr() as *mut std::ffi::c_void).expect("data pointer is null");
        device
            .newBufferWithBytes_length_options(ptr, size, MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate Metal buffer with data")
    }
}

/// Read back a slice of T values from a Metal buffer.
///
/// # Safety
/// The buffer must contain at least `count * size_of::<T>()` bytes.
pub unsafe fn read_buffer_slice<T: Copy>(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
) -> Vec<T> {
    let ptr = buffer.contents().as_ptr() as *const T;
    std::slice::from_raw_parts(ptr, count).to_vec()
}

// ─── Dispatch helper ────────────────────────────────────────────

/// Encode a 1D compute dispatch: set pipeline, bind buffers, dispatch threadgroups.
pub fn dispatch_1d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    total_threads: usize,
) {
    encoder.setComputePipelineState(pipeline);

    unsafe {
        for (buffer, index) in buffers {
            encoder.setBuffer_offset_atIndex(Some(*buffer), 0, *index);
        }
    }

    let threads_per_tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
    let threadgroup_count = total_threads.div_ceil(threads_per_tg);

    let grid_size = MTLSize {
        width: threadgroup_count,
        height: 1,
        depth: 1,
    };
    let tg_size = MTLSize {
        width: threads_per_tg,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
}

// ─── GPU timing ─────────────────────────────────────────────────

/// Return GPU execution time in milliseconds for a completed command buffer.
///
/// Uses Metal's GPUStartTime/GPUEndTime hardware timestamps.
/// Must be called after `commit()` + `waitUntilCompleted()`.
pub fn gpu_elapsed_ms(cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>) -> Option<f64> {
    let start = cmd_buf.GPUStartTime();
    let end = cmd_buf.GPUEndTime();
    if start == 0.0 || end == 0.0 {
        return None;
    }
    Some((end - start) * 1000.0)
}
