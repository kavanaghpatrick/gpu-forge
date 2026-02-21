//! GPU gather kernel: `output[i] = source[indices[i]]`.
//!
//! Supports u32 and u64 element types. The metallib is compiled by `build.rs`
//! and loaded once per [`GpuGather`] instance.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDevice,
    MTLLibrary, MTLPipelineOption, MTLResourceOptions, MTLSize,
};

/// Error type for gather operations.
#[derive(Debug, thiserror::Error)]
pub enum GatherError {
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),
    #[error("GPU execution failed: {0}")]
    GpuExecution(String),
}

/// Pre-compiled gather pipeline state objects.
///
/// Owns the Metal library and PSOs for `gather_u32` and `gather_u64` kernels.
pub struct GpuGather {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    #[allow(dead_code)]
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pso_u32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_u64: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl GpuGather {
    /// Create a new `GpuGather` with its own device and queue.
    pub fn new() -> Result<Self, GatherError> {
        let device = objc2_metal::MTLCreateSystemDefaultDevice()
            .ok_or_else(|| GatherError::ShaderCompilation("no Metal device".into()))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| GatherError::ShaderCompilation("failed to create queue".into()))?;
        Self::with_context(device, queue)
    }

    /// Create using an externally-provided device and queue (shared context).
    pub fn with_context(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Result<Self, GatherError> {
        let metallib_path = env!("GATHER_METALLIB_PATH");
        let path_ns = NSString::from_str(metallib_path);
        #[allow(deprecated)]
        let library = device
            .newLibraryWithFile_error(&path_ns)
            .map_err(|e| GatherError::ShaderCompilation(format!("{:?}", e)))?;

        let pso_u32 = compile_pso(&library, "gather_u32")?;
        let pso_u64 = compile_pso(&library, "gather_u64")?;

        Ok(Self {
            device,
            queue,
            library,
            pso_u32,
            pso_u64,
        })
    }

    /// Encode a gather dispatch onto an existing compute encoder.
    ///
    /// - `source`: buffer of elements to gather from
    /// - `indices`: buffer of u32 indices (element offsets)
    /// - `output`: buffer to write gathered elements into
    /// - `count`: number of elements to gather
    /// - `is_64bit`: if true, uses `gather_u64`; otherwise `gather_u32`
    ///
    /// Caller is responsible for ending the encoder and committing the command buffer.
    pub fn encode_gather(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        source: &ProtocolObject<dyn MTLBuffer>,
        indices: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        count: u32,
        is_64bit: bool,
    ) {
        let pso = if is_64bit { &self.pso_u64 } else { &self.pso_u32 };
        encoder.setComputePipelineState(pso);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(source), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(indices), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new(&count as *const u32 as *mut _).unwrap(),
                std::mem::size_of::<u32>(),
                3,
            );
        }

        let threads_per_tg = 256usize;
        let num_tgs = (count as usize).div_ceil(threads_per_tg);
        let grid = MTLSize {
            width: num_tgs,
            height: 1,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg_size);
    }

    /// Convenience: gather u32 elements using a dedicated command buffer.
    ///
    /// Creates CB + encoder, encodes gather, commits, waits, returns.
    pub fn gather_u32(
        &self,
        source: &ProtocolObject<dyn MTLBuffer>,
        indices: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        count: u32,
    ) -> Result<(), GatherError> {
        self.gather_sync(source, indices, output, count, false)
    }

    /// Convenience: gather u64 elements using a dedicated command buffer.
    pub fn gather_u64(
        &self,
        source: &ProtocolObject<dyn MTLBuffer>,
        indices: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        count: u32,
    ) -> Result<(), GatherError> {
        self.gather_sync(source, indices, output, count, true)
    }

    fn gather_sync(
        &self,
        source: &ProtocolObject<dyn MTLBuffer>,
        indices: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        count: u32,
        is_64bit: bool,
    ) -> Result<(), GatherError> {
        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            GatherError::GpuExecution("failed to create command buffer".into())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            GatherError::GpuExecution("failed to create compute encoder".into())
        })?;

        self.encode_gather(&enc, source, indices, output, count, is_64bit);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if cmd.status() == MTLCommandBufferStatus::Error {
            return Err(GatherError::GpuExecution("command buffer error".into()));
        }
        Ok(())
    }

    /// Allocate a `StorageModeShared` buffer of `size` bytes.
    pub fn alloc_buffer(&self, size: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate Metal buffer")
    }
}

fn compile_pso(
    library: &ProtocolObject<dyn MTLLibrary>,
    function_name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, GatherError> {
    let fn_name = NSString::from_str(function_name);
    #[allow(deprecated)]
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| {
            GatherError::ShaderCompilation(format!("kernel '{}' not found", function_name))
        })?;

    let descriptor = MTLComputePipelineDescriptor::new();
    descriptor.setComputeFunction(Some(&function));
    descriptor.setMaxTotalThreadsPerThreadgroup(256);
    unsafe {
        descriptor.setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
    }

    let device = library.device();
    device
        .newComputePipelineStateWithDescriptor_options_reflection_error(
            &descriptor,
            MTLPipelineOption::None,
            None,
        )
        .map_err(|e| GatherError::ShaderCompilation(format!("PSO '{}': {:?}", function_name, e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_u32_permutation() {
        let gather = GpuGather::new().expect("GpuGather::new failed");

        // Source: [10, 20, 30, 40]
        let source_data: [u32; 4] = [10, 20, 30, 40];
        let source_buf = gather.alloc_buffer(4 * std::mem::size_of::<u32>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                source_data.as_ptr(),
                source_buf.contents().as_ptr() as *mut u32,
                4,
            );
        }

        // Indices: [2, 0, 1, 3]  => output should be [30, 10, 20, 40]
        let indices_data: [u32; 4] = [2, 0, 1, 3];
        let indices_buf = gather.alloc_buffer(4 * std::mem::size_of::<u32>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                indices_data.as_ptr(),
                indices_buf.contents().as_ptr() as *mut u32,
                4,
            );
        }

        let output_buf = gather.alloc_buffer(4 * std::mem::size_of::<u32>());

        gather
            .gather_u32(&source_buf, &indices_buf, &output_buf, 4)
            .expect("gather_u32 failed");

        let output = unsafe {
            std::slice::from_raw_parts(output_buf.contents().as_ptr() as *const u32, 4)
        };
        assert_eq!(output, &[30, 10, 20, 40]);
    }

    #[test]
    fn gather_u64_permutation() {
        let gather = GpuGather::new().expect("GpuGather::new failed");

        // Source: [100, 200, 300, 400] as u64
        let source_data: [u64; 4] = [100, 200, 300, 400];
        let source_buf = gather.alloc_buffer(4 * std::mem::size_of::<u64>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                source_data.as_ptr(),
                source_buf.contents().as_ptr() as *mut u64,
                4,
            );
        }

        // Indices: [3, 1, 0, 2] => output should be [400, 200, 100, 300]
        let indices_data: [u32; 4] = [3, 1, 0, 2];
        let indices_buf = gather.alloc_buffer(4 * std::mem::size_of::<u32>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                indices_data.as_ptr(),
                indices_buf.contents().as_ptr() as *mut u32,
                4,
            );
        }

        let output_buf = gather.alloc_buffer(4 * std::mem::size_of::<u64>());

        gather
            .gather_u64(&source_buf, &indices_buf, &output_buf, 4)
            .expect("gather_u64 failed");

        let output = unsafe {
            std::slice::from_raw_parts(output_buf.contents().as_ptr() as *const u64, 4)
        };
        assert_eq!(output, &[400, 200, 100, 300]);
    }

    #[test]
    fn gather_u32_identity() {
        let gather = GpuGather::new().expect("GpuGather::new failed");

        // Identity gather: indices = [0, 1, 2, 3, 4]
        let n = 5u32;
        let source_data: Vec<u32> = (0..n).map(|i| (i + 1) * 10).collect();
        let source_buf = gather.alloc_buffer(n as usize * std::mem::size_of::<u32>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                source_data.as_ptr(),
                source_buf.contents().as_ptr() as *mut u32,
                n as usize,
            );
        }

        let indices_data: Vec<u32> = (0..n).collect();
        let indices_buf = gather.alloc_buffer(n as usize * std::mem::size_of::<u32>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                indices_data.as_ptr(),
                indices_buf.contents().as_ptr() as *mut u32,
                n as usize,
            );
        }

        let output_buf = gather.alloc_buffer(n as usize * std::mem::size_of::<u32>());

        gather
            .gather_u32(&source_buf, &indices_buf, &output_buf, n)
            .expect("gather_u32 failed");

        let output = unsafe {
            std::slice::from_raw_parts(output_buf.contents().as_ptr() as *const u32, n as usize)
        };
        assert_eq!(output, source_data.as_slice());
    }

    #[test]
    fn gather_u32_reverse() {
        let gather = GpuGather::new().expect("GpuGather::new failed");

        let n = 8u32;
        let source_data: Vec<u32> = (0..n).map(|i| i * 100).collect();
        let source_buf = gather.alloc_buffer(n as usize * std::mem::size_of::<u32>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                source_data.as_ptr(),
                source_buf.contents().as_ptr() as *mut u32,
                n as usize,
            );
        }

        // Reverse indices
        let indices_data: Vec<u32> = (0..n).rev().collect();
        let indices_buf = gather.alloc_buffer(n as usize * std::mem::size_of::<u32>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                indices_data.as_ptr(),
                indices_buf.contents().as_ptr() as *mut u32,
                n as usize,
            );
        }

        let output_buf = gather.alloc_buffer(n as usize * std::mem::size_of::<u32>());

        gather
            .gather_u32(&source_buf, &indices_buf, &output_buf, n)
            .expect("gather_u32 failed");

        let output = unsafe {
            std::slice::from_raw_parts(output_buf.contents().as_ptr() as *const u32, n as usize)
        };
        let expected: Vec<u32> = source_data.iter().rev().copied().collect();
        assert_eq!(output, expected.as_slice());
    }
}
