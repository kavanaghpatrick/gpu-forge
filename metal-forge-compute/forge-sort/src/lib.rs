use std::ptr::NonNull;

use forge_primitives::{alloc_buffer, MetalContext, PsoCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLLibrary, MTLSize,
};

const TILE_SIZE: usize = 4096;
const THREADS_PER_TG: usize = 256;

#[derive(Debug, thiserror::Error)]
pub enum SortError {
    #[error("no Metal GPU device found")]
    DeviceNotFound,
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),
    #[error("GPU execution failed: {0}")]
    GpuExecution(String),
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SortParams {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    pass: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BucketDesc {
    offset: u32,
    count: u32,
    tile_count: u32,
    tile_base: u32,
}

pub struct GpuSorter {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pso_cache: PsoCache,
    buf_a: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_b: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_msd_hist: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_counters: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_bucket_descs: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    data_buf_capacity: usize,
}

impl GpuSorter {
    /// Initialize Metal device, queue, and compile 4 sort kernel PSOs.
    pub fn new() -> Result<Self, SortError> {
        let ctx = MetalContext::new();

        // Load our sort-specific metallib (embedded path from build.rs)
        let metallib_path = env!("SORT_METALLIB_PATH");
        let path_ns = NSString::from_str(metallib_path);
        #[allow(deprecated)]
        let library = ctx
            .device
            .newLibraryWithFile_error(&path_ns)
            .map_err(|e| SortError::ShaderCompilation(format!("{:?}", e)))?;

        let mut pso_cache = PsoCache::new();

        // Pre-compile all 4 PSOs
        for name in &[
            "sort_msd_histogram",
            "sort_msd_prep",
            "sort_msd_atomic_scatter",
            "sort_inner_fused",
        ] {
            pso_cache.get_or_create(&library, name);
        }

        Ok(Self {
            device: ctx.device,
            queue: ctx.queue,
            library,
            pso_cache,
            buf_a: None,
            buf_b: None,
            buf_msd_hist: None,
            buf_counters: None,
            buf_bucket_descs: None,
            data_buf_capacity: 0,
        })
    }

    /// Sort u32 slice in-place on GPU. Empty/single-element inputs return immediately.
    pub fn sort_u32(&mut self, data: &mut [u32]) -> Result<(), SortError> {
        let n = data.len();
        if n <= 1 {
            return Ok(());
        }

        self.ensure_buffers(n);
        let num_tiles = n.div_ceil(TILE_SIZE);

        let buf_a = self.buf_a.as_ref().unwrap();
        let buf_b = self.buf_b.as_ref().unwrap();
        let buf_msd_hist = self.buf_msd_hist.as_ref().unwrap();
        let buf_counters = self.buf_counters.as_ref().unwrap();
        let buf_bucket_descs = self.buf_bucket_descs.as_ref().unwrap();

        // Copy input data to buf_a
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buf_a.contents().as_ptr() as *mut u8,
                n * 4,
            );
            // Zero the MSD histogram
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 256 * 4);
        }

        let params = SortParams {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: 24,
            pass: 0,
        };
        let tile_size_u32 = TILE_SIZE as u32;

        let tg_size = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        let hist_grid = MTLSize {
            width: num_tiles,
            height: 1,
            depth: 1,
        };
        let one_tg_grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let fused_grid = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        // Create command buffer and encoder
        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            SortError::GpuExecution("failed to create command buffer".to_string())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            SortError::GpuExecution("failed to create compute encoder".to_string())
        })?;

        // Dispatch 1: MSD histogram (reads buf_a → global_hist[256])
        let pso_histogram = self.pso_cache.get_or_create(&self.library, "sort_msd_histogram");
        enc.setComputePipelineState(pso_histogram);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const SortParams as *mut _).unwrap(),
                std::mem::size_of::<SortParams>(),
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        // Dispatch 2: MSD prep (prefix sum → counters + bucket_descs)
        let pso_prep = self.pso_cache.get_or_create(&self.library, "sort_msd_prep");
        enc.setComputePipelineState(pso_prep);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(),
                4,
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

        // Dispatch 3: Atomic MSD scatter (buf_a → buf_b)
        let pso_scatter = self
            .pso_cache
            .get_or_create(&self.library, "sort_msd_atomic_scatter");
        enc.setComputePipelineState(pso_scatter);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const SortParams as *mut _).unwrap(),
                std::mem::size_of::<SortParams>(),
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        // Dispatch 4: Fused inner sort (3-pass LSD per bucket, buf_b ↔ buf_a)
        let batch_start_0 = 0u32;
        let pso_fused = self.pso_cache.get_or_create(&self.library, "sort_inner_fused");
        enc.setComputePipelineState(pso_fused);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&batch_start_0 as *const u32 as *mut _).unwrap(),
                4,
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        // Check for GPU errors
        let status = cmd.status();
        if status == MTLCommandBufferStatus::Error {
            return Err(SortError::GpuExecution(format!(
                "command buffer error: {:?}",
                cmd.error()
            )));
        }

        // Copy result back from buf_a
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf_a.contents().as_ptr() as *const u32,
                data.as_mut_ptr(),
                n,
            );
        }

        Ok(())
    }

    fn ensure_buffers(&mut self, n: usize) {
        let data_bytes = n * 4;

        if self.buf_a.is_none() || data_bytes > self.data_buf_capacity {
            self.buf_a = Some(alloc_buffer(&self.device, data_bytes));
            self.buf_b = Some(alloc_buffer(&self.device, data_bytes));
            self.data_buf_capacity = data_bytes;
        }

        if self.buf_msd_hist.is_none() {
            self.buf_msd_hist = Some(alloc_buffer(&self.device, 256 * 4));
            self.buf_counters = Some(alloc_buffer(&self.device, 256 * 4));
            self.buf_bucket_descs =
                Some(alloc_buffer(&self.device, 256 * std::mem::size_of::<BucketDesc>()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_params_size() {
        assert_eq!(std::mem::size_of::<SortParams>(), 16);
    }

    #[test]
    fn test_bucket_desc_size() {
        assert_eq!(std::mem::size_of::<BucketDesc>(), 16);
    }

    #[test]
    fn test_sort_error_display() {
        let e = SortError::DeviceNotFound;
        assert_eq!(e.to_string(), "no Metal GPU device found");

        let e = SortError::ShaderCompilation("test".to_string());
        assert_eq!(e.to_string(), "shader compilation failed: test");

        let e = SortError::GpuExecution("timeout".to_string());
        assert_eq!(e.to_string(), "GPU execution failed: timeout");
    }
}
