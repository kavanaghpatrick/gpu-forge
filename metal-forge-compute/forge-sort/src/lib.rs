use std::marker::PhantomData;
use std::ptr::NonNull;

use forge_primitives::{alloc_buffer, MetalContext, PsoCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLLibrary, MTLSize,
};

// --- SortKey sealed trait ---

mod private {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
    impl Sealed for u64 {}
    impl Sealed for i64 {}
    impl Sealed for f64 {}
}

/// Trait for types that can be sorted on the GPU via radix sort.
///
/// Sealed: only u32, i32, f32, u64, i64, f64 are valid. `SortBuffer<String>` is a compile error.
pub trait SortKey: private::Sealed + Copy + 'static {
    /// Size of the key in bytes (4 or 8).
    const KEY_SIZE: usize;
    /// Whether the key needs a bit transformation before/after sorting.
    const NEEDS_TRANSFORM: bool;
    /// Whether this is a 64-bit key type.
    const IS_64BIT: bool;
    /// Transform mode for forward (pre-sort) transformation.
    const TRANSFORM_MODE_FORWARD: u32;
    /// Transform mode for inverse (post-sort) transformation.
    const TRANSFORM_MODE_INVERSE: u32;
}

impl SortKey for u32 {
    const KEY_SIZE: usize = 4;
    const NEEDS_TRANSFORM: bool = false;
    const IS_64BIT: bool = false;
    const TRANSFORM_MODE_FORWARD: u32 = 0;
    const TRANSFORM_MODE_INVERSE: u32 = 0;
}

impl SortKey for i32 {
    const KEY_SIZE: usize = 4;
    const NEEDS_TRANSFORM: bool = true;
    const IS_64BIT: bool = false;
    const TRANSFORM_MODE_FORWARD: u32 = 0;
    const TRANSFORM_MODE_INVERSE: u32 = 0;
}

impl SortKey for f32 {
    const KEY_SIZE: usize = 4;
    const NEEDS_TRANSFORM: bool = true;
    const IS_64BIT: bool = false;
    const TRANSFORM_MODE_FORWARD: u32 = 1;
    const TRANSFORM_MODE_INVERSE: u32 = 2;
}

impl SortKey for u64 {
    const KEY_SIZE: usize = 8;
    const NEEDS_TRANSFORM: bool = false;
    const IS_64BIT: bool = true;
    const TRANSFORM_MODE_FORWARD: u32 = 0;
    const TRANSFORM_MODE_INVERSE: u32 = 0;
}

impl SortKey for i64 {
    const KEY_SIZE: usize = 8;
    const NEEDS_TRANSFORM: bool = true;
    const IS_64BIT: bool = true;
    const TRANSFORM_MODE_FORWARD: u32 = 1;
    const TRANSFORM_MODE_INVERSE: u32 = 1;
}

impl SortKey for f64 {
    const KEY_SIZE: usize = 8;
    const NEEDS_TRANSFORM: bool = true;
    const IS_64BIT: bool = true;
    const TRANSFORM_MODE_FORWARD: u32 = 1;
    const TRANSFORM_MODE_INVERSE: u32 = 2;
}

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
    #[error("length mismatch: keys={keys}, values={values}")]
    LengthMismatch { keys: usize, values: usize },
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

/// A Metal buffer that can be sorted in-place with zero memcpy.
///
/// Created via [`GpuSorter::alloc_sort_buffer`]. The buffer uses `StorageModeShared`
/// (unified memory), so CPU reads/writes go directly to the same physical pages the GPU uses.
pub struct SortBuffer<T: SortKey> {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    capacity: usize,
    _marker: PhantomData<T>,
}

impl<T: SortKey> SortBuffer<T> {
    /// Number of elements currently in this buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacity in elements.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get a mutable slice to write data directly into GPU-visible memory.
    /// Write your data here, then call `set_len()` to mark how many elements are valid.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.contents().as_ptr() as *mut T,
                self.capacity,
            )
        }
    }

    /// Get a slice to read sorted results directly from GPU-visible memory.
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.buffer.contents().as_ptr() as *const T, self.len)
        }
    }

    /// Set the number of valid elements. Must be <= capacity.
    pub fn set_len(&mut self, len: usize) {
        assert!(
            len <= self.capacity,
            "len {} exceeds capacity {}",
            len,
            self.capacity
        );
        self.len = len;
    }

    /// Copy data from a slice into the buffer. Sets len automatically.
    pub fn copy_from_slice(&mut self, data: &[T]) {
        assert!(
            data.len() <= self.capacity,
            "data len {} exceeds capacity {}",
            data.len(),
            self.capacity
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.buffer.contents().as_ptr() as *mut T,
                data.len(),
            );
        }
        self.len = data.len();
    }

    /// Copy sorted results out to a slice.
    pub fn copy_to_slice(&self, dest: &mut [T]) {
        let n = self.len.min(dest.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.contents().as_ptr() as *const T,
                dest.as_mut_ptr(),
                n,
            );
        }
    }

    /// Access the underlying Metal buffer (for advanced use / pipeline integration).
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

/// Encode and execute the 4-dispatch sort pipeline. Shared by sort_u32 and sort_buffer.
fn dispatch_sort(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    library: &ProtocolObject<dyn MTLLibrary>,
    pso_cache: &mut PsoCache,
    buf_a: &ProtocolObject<dyn MTLBuffer>,
    buf_b: &ProtocolObject<dyn MTLBuffer>,
    buf_msd_hist: &ProtocolObject<dyn MTLBuffer>,
    buf_counters: &ProtocolObject<dyn MTLBuffer>,
    buf_bucket_descs: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    num_tiles: usize,
) -> Result<(), SortError> {
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

    let cmd = queue.commandBuffer().ok_or_else(|| {
        SortError::GpuExecution("failed to create command buffer".to_string())
    })?;
    let enc = cmd.computeCommandEncoder().ok_or_else(|| {
        SortError::GpuExecution("failed to create compute encoder".to_string())
    })?;

    // Dispatch 1: MSD histogram
    let pso = pso_cache.get_or_create(library, "sort_msd_histogram");
    enc.setComputePipelineState(pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(&params as *const SortParams as *mut _).unwrap(),
            std::mem::size_of::<SortParams>(),
            2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

    // Dispatch 2: MSD prep
    let pso = pso_cache.get_or_create(library, "sort_msd_prep");
    enc.setComputePipelineState(pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_counters), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs), 0, 2);
        enc.setBytes_length_atIndex(
            NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

    // Dispatch 3: Atomic MSD scatter (buf_a → buf_b)
    let pso = pso_cache.get_or_create(library, "sort_msd_atomic_scatter");
    enc.setComputePipelineState(pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_counters), 0, 2);
        enc.setBytes_length_atIndex(
            NonNull::new(&params as *const SortParams as *mut _).unwrap(),
            std::mem::size_of::<SortParams>(),
            3,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

    // Dispatch 4: Fused inner sort (3-pass LSD, buf_b ↔ buf_a)
    let batch_start_0 = 0u32;
    let pso = pso_cache.get_or_create(library, "sort_inner_fused");
    enc.setComputePipelineState(pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs), 0, 2);
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

    if cmd.status() == MTLCommandBufferStatus::Error {
        return Err(SortError::GpuExecution(format!(
            "command buffer error: {:?}",
            cmd.error()
        )));
    }

    Ok(())
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

    /// Allocate a GPU buffer for zero-copy sorting.
    ///
    /// Returns a [`SortBuffer<T>`] backed by unified memory (`StorageModeShared`).
    /// Write data via `as_mut_slice()` or `copy_from_slice()`, then sort with
    /// [`sort_buffer()`](Self::sort_buffer). Read results via `as_slice()`.
    /// No memcpy between CPU and GPU — both see the same physical memory.
    pub fn alloc_sort_buffer<T: SortKey>(&self, capacity: usize) -> SortBuffer<T> {
        let buffer = alloc_buffer(&self.device, capacity * T::KEY_SIZE);
        SortBuffer {
            buffer,
            len: 0,
            capacity,
            _marker: PhantomData,
        }
    }

    /// Sort a [`SortBuffer<u32>`] in-place on GPU. **Zero memcpy** — full GPU speed.
    ///
    /// The buffer is sorted in-place. Read results via `buf.as_slice()`.
    /// Only scratch buffers are allocated internally (grow-only, reused across calls).
    pub fn sort_buffer(&mut self, buf: &SortBuffer<u32>) -> Result<(), SortError> {
        let n = buf.len();
        if n <= 1 {
            return Ok(());
        }

        self.ensure_scratch_buffers(n);
        let num_tiles = n.div_ceil(TILE_SIZE);

        // Zero the MSD histogram
        unsafe {
            std::ptr::write_bytes(
                self.buf_msd_hist.as_ref().unwrap().contents().as_ptr() as *mut u8,
                0,
                256 * 4,
            );
        }

        dispatch_sort(
            &self.queue,
            &self.library,
            &mut self.pso_cache,
            &buf.buffer,
            self.buf_b.as_ref().unwrap(),
            self.buf_msd_hist.as_ref().unwrap(),
            self.buf_counters.as_ref().unwrap(),
            self.buf_bucket_descs.as_ref().unwrap(),
            n,
            num_tiles,
        )
    }

    /// Sort u32 slice in-place on GPU. Empty/single-element inputs return immediately.
    ///
    /// This copies data to an internal GPU buffer and back. For zero-copy sorting,
    /// use [`alloc_sort_buffer()`](Self::alloc_sort_buffer) + [`sort_buffer()`](Self::sort_buffer).
    pub fn sort_u32(&mut self, data: &mut [u32]) -> Result<(), SortError> {
        let n = data.len();
        if n <= 1 {
            return Ok(());
        }

        self.ensure_buffers(n);
        let num_tiles = n.div_ceil(TILE_SIZE);

        // Copy input data to buf_a + zero the MSD histogram
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.buf_a.as_ref().unwrap().contents().as_ptr() as *mut u8,
                n * 4,
            );
            std::ptr::write_bytes(
                self.buf_msd_hist.as_ref().unwrap().contents().as_ptr() as *mut u8,
                0,
                256 * 4,
            );
        }

        dispatch_sort(
            &self.queue,
            &self.library,
            &mut self.pso_cache,
            self.buf_a.as_ref().unwrap(),
            self.buf_b.as_ref().unwrap(),
            self.buf_msd_hist.as_ref().unwrap(),
            self.buf_counters.as_ref().unwrap(),
            self.buf_bucket_descs.as_ref().unwrap(),
            n,
            num_tiles,
        )?;

        // Copy result back from buf_a
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf_a.as_ref().unwrap().contents().as_ptr() as *const u32,
                data.as_mut_ptr(),
                n,
            );
        }

        Ok(())
    }

    /// Allocate internal data + scratch buffers for sort_u32 (includes buf_a).
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

    /// Allocate only scratch buffers for sort_buffer (buf_b + metadata, no buf_a).
    fn ensure_scratch_buffers(&mut self, n: usize) {
        let data_bytes = n * 4;

        // buf_b is scratch space — needs to be at least as large as the data
        if self.buf_b.is_none() || data_bytes > self.data_buf_capacity {
            self.buf_b = Some(alloc_buffer(&self.device, data_bytes));
            // Also reallocate buf_a if it exists (keep them in sync)
            if self.buf_a.is_some() {
                self.buf_a = Some(alloc_buffer(&self.device, data_bytes));
            }
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

        let e = SortError::LengthMismatch {
            keys: 10,
            values: 5,
        };
        assert_eq!(e.to_string(), "length mismatch: keys=10, values=5");
    }

    #[test]
    fn test_sort_key_u32_consts() {
        assert_eq!(u32::KEY_SIZE, 4);
        assert!(!u32::NEEDS_TRANSFORM);
        assert!(!u32::IS_64BIT);
        assert_eq!(u32::TRANSFORM_MODE_FORWARD, 0);
        assert_eq!(u32::TRANSFORM_MODE_INVERSE, 0);
    }

    #[test]
    fn test_sort_key_i32_consts() {
        assert_eq!(i32::KEY_SIZE, 4);
        assert!(i32::NEEDS_TRANSFORM);
        assert!(!i32::IS_64BIT);
        assert_eq!(i32::TRANSFORM_MODE_FORWARD, 0);
        assert_eq!(i32::TRANSFORM_MODE_INVERSE, 0);
    }

    #[test]
    fn test_sort_key_f32_consts() {
        assert_eq!(f32::KEY_SIZE, 4);
        assert!(f32::NEEDS_TRANSFORM);
        assert!(!f32::IS_64BIT);
        assert_eq!(f32::TRANSFORM_MODE_FORWARD, 1);
        assert_eq!(f32::TRANSFORM_MODE_INVERSE, 2);
    }

    #[test]
    fn test_sort_key_u64_consts() {
        assert_eq!(u64::KEY_SIZE, 8);
        assert!(!u64::NEEDS_TRANSFORM);
        assert!(u64::IS_64BIT);
    }

    #[test]
    fn test_sort_key_i64_consts() {
        assert_eq!(i64::KEY_SIZE, 8);
        assert!(i64::NEEDS_TRANSFORM);
        assert!(i64::IS_64BIT);
        assert_eq!(i64::TRANSFORM_MODE_FORWARD, 1);
        assert_eq!(i64::TRANSFORM_MODE_INVERSE, 1);
    }

    #[test]
    fn test_sort_key_f64_consts() {
        assert_eq!(f64::KEY_SIZE, 8);
        assert!(f64::NEEDS_TRANSFORM);
        assert!(f64::IS_64BIT);
        assert_eq!(f64::TRANSFORM_MODE_FORWARD, 1);
        assert_eq!(f64::TRANSFORM_MODE_INVERSE, 2);
    }
}
