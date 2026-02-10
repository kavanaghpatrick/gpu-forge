//! Compute command encoder helpers for Metal GPU dispatch.
//!
//! Provides convenience functions for creating compute command encoders,
//! setting buffers, and dispatching threadgroups. Follows the per-pass
//! pattern from particle-system.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};

/// Create a compute pipeline state from a kernel function name in the library.
pub fn make_pipeline(
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let fn_name = NSString::from_str(name);
    #[allow(deprecated)]
    let function = library
        .newFunctionWithName(&fn_name)
        .unwrap_or_else(|| panic!("Kernel function '{}' not found in metallib", name));

    let device = library.device();
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|e| {
            panic!(
                "Failed to create compute pipeline for '{}': {:?}",
                name, e
            )
        })
}

/// Create a new command buffer from the queue.
pub fn make_command_buffer(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
    queue
        .commandBuffer()
        .expect("Failed to create command buffer")
}

/// Encode a compute pass: set pipeline, set buffers, dispatch 1D threadgroups.
///
/// `buffers` is a slice of (buffer, index) pairs to bind.
/// `total_threads` is the number of threads to dispatch.
/// The encoder computes the threadgroup count from the pipeline's
/// `maxTotalThreadsPerThreadgroup`.
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
    let threadgroup_count = (total_threads + threads_per_tg - 1) / threads_per_tg;

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

/// Dispatch with explicit thread count (1D) using dispatchThreads.
/// More precise than threadgroup-based dispatch for non-uniform workloads.
pub fn dispatch_threads_1d(
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

    let grid_size = MTLSize {
        width: total_threads,
        height: 1,
        depth: 1,
    };
    let tg_size = MTLSize {
        width: threads_per_tg,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
}

/// Allocate a Metal buffer of `size` bytes with StorageModeShared.
pub fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    size: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let options = objc2_metal::MTLResourceOptions::StorageModeShared;
    device
        .newBufferWithLength_options(size, options)
        .expect("Failed to allocate Metal buffer")
}

/// Allocate a Metal buffer initialized with the given bytes.
pub fn alloc_buffer_with_data<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    use std::ptr::NonNull;

    let size = std::mem::size_of_val(data);
    let options = objc2_metal::MTLResourceOptions::StorageModeShared;

    unsafe {
        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
            .expect("data pointer is null");
        device
            .newBufferWithBytes_length_options(ptr, size, options)
            .expect("Failed to allocate Metal buffer with data")
    }
}

/// Read back a value of type T from a Metal buffer at offset 0.
///
/// # Safety
/// The buffer must contain at least `size_of::<T>()` bytes, and the
/// data must be a valid representation of T.
pub unsafe fn read_buffer<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>) -> T {
    let ptr = buffer.contents().as_ptr() as *const T;
    *ptr
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
    let slice = std::slice::from_raw_parts(ptr, count);
    slice.to_vec()
}
