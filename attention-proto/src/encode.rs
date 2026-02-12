//! Compute command encoder helpers for Metal GPU dispatch.
//!
//! Provides convenience functions for buffer allocation, compute dispatch
//! (1D and 2D), and buffer readback. Adapted from gpu-query/src/gpu/encode.rs
//! with added dispatch_2d for attention grids.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
    MTLResourceOptions, MTLSize,
};

/// Allocate a Metal buffer of `size` bytes with StorageModeShared.
pub fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    size: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let options = MTLResourceOptions::StorageModeShared;
    device
        .newBufferWithLength_options(size, options)
        .expect("Failed to allocate Metal buffer")
}

/// Allocate a Metal buffer initialized with the given data slice.
pub fn alloc_buffer_with_data<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    use std::ptr::NonNull;

    let size = std::mem::size_of_val(data);
    let options = MTLResourceOptions::StorageModeShared;

    unsafe {
        let ptr =
            NonNull::new(data.as_ptr() as *mut std::ffi::c_void).expect("data pointer is null");
        device
            .newBufferWithBytes_length_options(ptr, size, options)
            .expect("Failed to allocate Metal buffer with data")
    }
}

/// Encode a 1D compute dispatch: set pipeline, bind buffers, dispatch threadgroups.
///
/// `buffers` is a slice of (buffer, index) pairs to bind at the given argument indices.
/// `total_threads` is the number of threads to dispatch.
/// Threadgroup size is clamped to min(maxTotalThreadsPerThreadgroup, 256).
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

/// Encode a 2D compute dispatch for attention grids.
///
/// `buffers` is a slice of (buffer, index) pairs to bind at the given argument indices.
/// `width` and `height` define the 2D grid dimensions (e.g., seq_len x heads).
/// Threadgroup size defaults to 16x16 (256 threads), clamped to pipeline max.
pub fn dispatch_2d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    width: usize,
    height: usize,
) {
    encoder.setComputePipelineState(pipeline);

    unsafe {
        for (buffer, index) in buffers {
            encoder.setBuffer_offset_atIndex(Some(*buffer), 0, *index);
        }
    }

    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    // Default 16x16 = 256 threads per threadgroup, but clamp if pipeline max is lower
    let tg_side = if max_threads >= 256 { 16 } else { (max_threads as f64).sqrt() as usize };

    let grid_size = MTLSize {
        width,
        height,
        depth: 1,
    };
    let tg_size = MTLSize {
        width: tg_side,
        height: tg_side,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
}

/// Read back a single value of type T from a Metal buffer at offset 0.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;

    #[test]
    fn test_buffer_roundtrip() {
        let gpu = GpuDevice::shared();

        // Test alloc_buffer_with_data + read_buffer_slice roundtrip
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let buffer = alloc_buffer_with_data(&gpu.device, &input);

        let output: Vec<f32> = unsafe { read_buffer_slice(&buffer, input.len()) };
        assert_eq!(input, output, "buffer roundtrip should preserve data");

        // Test alloc_buffer + manual write + read_buffer single value
        let buf = alloc_buffer(&gpu.device, std::mem::size_of::<u32>());
        unsafe {
            let ptr = buf.contents().as_ptr() as *mut u32;
            *ptr = 42;
        }
        let val: u32 = unsafe { read_buffer(&buf) };
        assert_eq!(val, 42, "single value roundtrip should preserve data");

        // Test alloc_buffer_with_data with u32
        let u32_input: Vec<u32> = vec![10, 20, 30, 40];
        let u32_buffer = alloc_buffer_with_data(&gpu.device, &u32_input);
        let u32_output: Vec<u32> = unsafe { read_buffer_slice(&u32_buffer, u32_input.len()) };
        assert_eq!(u32_input, u32_output, "u32 buffer roundtrip should preserve data");
    }
}
