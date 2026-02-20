use std::marker::PhantomData;
use std::ptr::NonNull;

use forge_primitives::pso_cache::FnConstant;
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
struct InnerParams {
    start_shift: u32,
    pass_count: u32,
    batch_start: u32,
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
    // Value buffers for argsort / key-value sort (lazy allocation)
    buf_vals_a: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_vals_b: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    vals_buf_capacity: usize,
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
    let pso = pso_cache.get_or_create_specialized(library, "sort_msd_atomic_scatter", &[(0, FnConstant::Bool(false))]);
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
    let inner_params = InnerParams {
        start_shift: 0,
        pass_count: 3,
        batch_start: 0,
    };
    let pso = pso_cache.get_or_create_specialized(library, "sort_inner_fused", &[(0, FnConstant::Bool(false))]);
    enc.setComputePipelineState(pso);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs), 0, 2);
        enc.setBytes_length_atIndex(
            NonNull::new(&inner_params as *const InnerParams as *mut _).unwrap(),
            std::mem::size_of::<InnerParams>(),
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

/// Encode a single sort_transform_32 dispatch onto an existing encoder.
fn encode_transform_32(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    library: &ProtocolObject<dyn MTLLibrary>,
    pso_cache: &mut PsoCache,
    buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    mode: u32,
) {
    let pso = pso_cache.get_or_create_specialized(
        library,
        "sort_transform_32",
        &[(2, FnConstant::U32(mode))],
    );
    encoder.setComputePipelineState(pso);
    let count = n as u32;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buf), 0, 0);
        encoder.setBytes_length_atIndex(
            NonNull::new(&count as *const u32 as *mut _).unwrap(),
            4,
            1,
        );
    }
    let grid = MTLSize {
        width: n.div_ceil(THREADS_PER_TG) * THREADS_PER_TG,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: THREADS_PER_TG,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
}

/// Encode the 4-dispatch sort pipeline onto an existing encoder.
/// Like `dispatch_sort` but takes an encoder instead of creating its own command buffer.
fn encode_sort_pipeline(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    library: &ProtocolObject<dyn MTLLibrary>,
    pso_cache: &mut PsoCache,
    buf_a: &ProtocolObject<dyn MTLBuffer>,
    buf_b: &ProtocolObject<dyn MTLBuffer>,
    buf_msd_hist: &ProtocolObject<dyn MTLBuffer>,
    buf_counters: &ProtocolObject<dyn MTLBuffer>,
    buf_bucket_descs: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
    num_tiles: usize,
) {
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

    // Dispatch 1: MSD histogram
    let pso = pso_cache.get_or_create(library, "sort_msd_histogram");
    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(buf_msd_hist), 0, 1);
        encoder.setBytes_length_atIndex(
            NonNull::new(&params as *const SortParams as *mut _).unwrap(),
            std::mem::size_of::<SortParams>(),
            2,
        );
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

    // Dispatch 2: MSD prep
    let pso = pso_cache.get_or_create(library, "sort_msd_prep");
    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buf_msd_hist), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(buf_counters), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(buf_bucket_descs), 0, 2);
        encoder.setBytes_length_atIndex(
            NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

    // Dispatch 3: Atomic MSD scatter (buf_a → buf_b)
    let pso = pso_cache.get_or_create_specialized(library, "sort_msd_atomic_scatter", &[(0, FnConstant::Bool(false))]);
    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(buf_counters), 0, 2);
        encoder.setBytes_length_atIndex(
            NonNull::new(&params as *const SortParams as *mut _).unwrap(),
            std::mem::size_of::<SortParams>(),
            3,
        );
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

    // Dispatch 4: Fused inner sort (3-pass LSD, buf_b ↔ buf_a)
    let inner_params = InnerParams {
        start_shift: 0,
        pass_count: 3,
        batch_start: 0,
    };
    let pso = pso_cache.get_or_create_specialized(library, "sort_inner_fused", &[(0, FnConstant::Bool(false))]);
    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(buf_bucket_descs), 0, 2);
        encoder.setBytes_length_atIndex(
            NonNull::new(&inner_params as *const InnerParams as *mut _).unwrap(),
            std::mem::size_of::<InnerParams>(),
            3,
        );
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);
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

        // Pre-compile sort PSOs (histogram + prep use no function constants)
        for name in &["sort_msd_histogram", "sort_msd_prep"] {
            pso_cache.get_or_create(&library, name);
        }
        // scatter + inner fused require HAS_VALUES specialization (default false)
        pso_cache.get_or_create_specialized(
            &library,
            "sort_msd_atomic_scatter",
            &[(0, FnConstant::Bool(false))],
        );
        pso_cache.get_or_create_specialized(
            &library,
            "sort_inner_fused",
            &[(0, FnConstant::Bool(false))],
        );

        // Pre-compile transform PSOs for i32/f32 (index 2 = TRANSFORM_MODE)
        for mode in 0u32..=2 {
            pso_cache.get_or_create_specialized(
                &library,
                "sort_transform_32",
                &[(2, FnConstant::U32(mode))],
            );
        }

        // Pre-compile argsort helper PSOs (no function constants needed)
        for name in &["sort_init_indices", "sort_gather_values"] {
            pso_cache.get_or_create(&library, name);
        }

        // Pre-compile HAS_VALUES=true PSOs for scatter + inner fused (argsort)
        pso_cache.get_or_create_specialized(
            &library,
            "sort_msd_atomic_scatter",
            &[(0, FnConstant::Bool(true))],
        );
        pso_cache.get_or_create_specialized(
            &library,
            "sort_inner_fused",
            &[(0, FnConstant::Bool(true))],
        );

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
            buf_vals_a: None,
            buf_vals_b: None,
            vals_buf_capacity: 0,
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

    /// Sort i32 slice in-place on GPU. Uses XOR 0x80000000 transform.
    pub fn sort_i32(&mut self, data: &mut [i32]) -> Result<(), SortError> {
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

        // Create single command buffer with: transform → sort → inverse transform
        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            SortError::GpuExecution("failed to create command buffer".to_string())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            SortError::GpuExecution("failed to create compute encoder".to_string())
        })?;

        let buf_a = self.buf_a.as_ref().unwrap();
        let buf_b = self.buf_b.as_ref().unwrap();

        // Forward transform: XOR 0x80000000 (mode 0)
        encode_transform_32(&enc, &self.library, &mut self.pso_cache, buf_a, n, 0);

        // 4-dispatch sort pipeline
        encode_sort_pipeline(
            &enc,
            &self.library,
            &mut self.pso_cache,
            buf_a,
            buf_b,
            self.buf_msd_hist.as_ref().unwrap(),
            self.buf_counters.as_ref().unwrap(),
            self.buf_bucket_descs.as_ref().unwrap(),
            n,
            num_tiles,
        );

        // Inverse transform: XOR 0x80000000 (mode 0, self-inverse)
        encode_transform_32(&enc, &self.library, &mut self.pso_cache, buf_a, n, 0);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if cmd.status() == MTLCommandBufferStatus::Error {
            return Err(SortError::GpuExecution(format!(
                "command buffer error: {:?}",
                cmd.error()
            )));
        }

        // Copy result back from buf_a
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf_a.as_ref().unwrap().contents().as_ptr() as *const i32,
                data.as_mut_ptr(),
                n,
            );
        }

        Ok(())
    }

    /// Sort f32 slice in-place on GPU. Uses FloatFlip/IFloatFlip transform for IEEE 754 total ordering.
    ///
    /// Handles all special values correctly: -NaN < -Inf < -1.0 < -0.0 < 0.0 < 1.0 < Inf < NaN.
    pub fn sort_f32(&mut self, data: &mut [f32]) -> Result<(), SortError> {
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

        // Create single command buffer with: FloatFlip → sort → IFloatFlip
        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            SortError::GpuExecution("failed to create command buffer".to_string())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            SortError::GpuExecution("failed to create compute encoder".to_string())
        })?;

        let buf_a = self.buf_a.as_ref().unwrap();
        let buf_b = self.buf_b.as_ref().unwrap();

        // Forward transform: FloatFlip (mode 1)
        encode_transform_32(&enc, &self.library, &mut self.pso_cache, buf_a, n, 1);

        // 4-dispatch sort pipeline
        encode_sort_pipeline(
            &enc,
            &self.library,
            &mut self.pso_cache,
            buf_a,
            buf_b,
            self.buf_msd_hist.as_ref().unwrap(),
            self.buf_counters.as_ref().unwrap(),
            self.buf_bucket_descs.as_ref().unwrap(),
            n,
            num_tiles,
        );

        // Inverse transform: IFloatFlip (mode 2)
        encode_transform_32(&enc, &self.library, &mut self.pso_cache, buf_a, n, 2);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if cmd.status() == MTLCommandBufferStatus::Error {
            return Err(SortError::GpuExecution(format!(
                "command buffer error: {:?}",
                cmd.error()
            )));
        }

        // Copy result back from buf_a
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf_a.as_ref().unwrap().contents().as_ptr() as *const f32,
                data.as_mut_ptr(),
                n,
            );
        }

        Ok(())
    }

    /// Sort a [`SortBuffer<f32>`] in-place on GPU. **Zero memcpy**.
    ///
    /// Uses FloatFlip/IFloatFlip transforms for IEEE 754 total ordering.
    pub fn sort_f32_buffer(&mut self, buf: &SortBuffer<f32>) -> Result<(), SortError> {
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

        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            SortError::GpuExecution("failed to create command buffer".to_string())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            SortError::GpuExecution("failed to create compute encoder".to_string())
        })?;

        // Forward transform: FloatFlip (mode 1)
        encode_transform_32(&enc, &self.library, &mut self.pso_cache, &buf.buffer, n, 1);

        // 4-dispatch sort pipeline
        encode_sort_pipeline(
            &enc,
            &self.library,
            &mut self.pso_cache,
            &buf.buffer,
            self.buf_b.as_ref().unwrap(),
            self.buf_msd_hist.as_ref().unwrap(),
            self.buf_counters.as_ref().unwrap(),
            self.buf_bucket_descs.as_ref().unwrap(),
            n,
            num_tiles,
        );

        // Inverse transform: IFloatFlip (mode 2)
        encode_transform_32(&enc, &self.library, &mut self.pso_cache, &buf.buffer, n, 2);

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

    /// Sort a [`SortBuffer<i32>`] in-place on GPU. **Zero memcpy**.
    pub fn sort_i32_buffer(&mut self, buf: &SortBuffer<i32>) -> Result<(), SortError> {
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

        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            SortError::GpuExecution("failed to create command buffer".to_string())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            SortError::GpuExecution("failed to create compute encoder".to_string())
        })?;

        // Forward transform: XOR 0x80000000 (mode 0)
        encode_transform_32(&enc, &self.library, &mut self.pso_cache, &buf.buffer, n, 0);

        // 4-dispatch sort pipeline
        encode_sort_pipeline(
            &enc,
            &self.library,
            &mut self.pso_cache,
            &buf.buffer,
            self.buf_b.as_ref().unwrap(),
            self.buf_msd_hist.as_ref().unwrap(),
            self.buf_counters.as_ref().unwrap(),
            self.buf_bucket_descs.as_ref().unwrap(),
            n,
            num_tiles,
        );

        // Inverse transform: XOR 0x80000000 (mode 0, self-inverse)
        encode_transform_32(&enc, &self.library, &mut self.pso_cache, &buf.buffer, n, 0);

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

    /// Allocate data buffers + value buffers for argsort (values are u32 indices).
    fn ensure_buffers_with_values(&mut self, n: usize) {
        // Ensure key buffers (buf_a, buf_b) + scratch
        self.ensure_buffers(n);

        let val_bytes = n * 4; // values are always u32 (indices)
        if self.buf_vals_a.is_none() || val_bytes > self.vals_buf_capacity {
            self.buf_vals_a = Some(alloc_buffer(&self.device, val_bytes));
            self.buf_vals_b = Some(alloc_buffer(&self.device, val_bytes));
            self.vals_buf_capacity = val_bytes;
        }
    }

    /// Encode the full sort pipeline with optional value tracking onto an existing encoder.
    ///
    /// When `with_values` is true, uses HAS_VALUES=true PSOs and binds value buffers
    /// at buffer(4) and buffer(5) for scatter and inner fused kernels.
    fn encode_sort_pipeline_full(
        &mut self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_b: &ProtocolObject<dyn MTLBuffer>,
        vals_a: Option<&ProtocolObject<dyn MTLBuffer>>,
        vals_b: Option<&ProtocolObject<dyn MTLBuffer>>,
        n: usize,
        num_tiles: usize,
        with_values: bool,
    ) {
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

        // Dispatch 1: MSD histogram (key-only, no change)
        let pso = self.pso_cache.get_or_create(&self.library, "sort_msd_histogram");
        encoder.setComputePipelineState(pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(self.buf_msd_hist.as_ref().unwrap()), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::new(&params as *const SortParams as *mut _).unwrap(),
                std::mem::size_of::<SortParams>(),
                2,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        // Dispatch 2: MSD prep (no change)
        let pso = self.pso_cache.get_or_create(&self.library, "sort_msd_prep");
        encoder.setComputePipelineState(pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(self.buf_msd_hist.as_ref().unwrap()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(self.buf_counters.as_ref().unwrap()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(self.buf_bucket_descs.as_ref().unwrap()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(),
                4,
                3,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

        // Dispatch 3: Atomic MSD scatter (buf_a → buf_b, vals_a → vals_b if with_values)
        let pso = self.pso_cache.get_or_create_specialized(
            &self.library,
            "sort_msd_atomic_scatter",
            &[(0, FnConstant::Bool(with_values))],
        );
        encoder.setComputePipelineState(pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(self.buf_counters.as_ref().unwrap()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new(&params as *const SortParams as *mut _).unwrap(),
                std::mem::size_of::<SortParams>(),
                3,
            );
            if with_values {
                encoder.setBuffer_offset_atIndex(vals_a, 0, 4);
                encoder.setBuffer_offset_atIndex(vals_b, 0, 5);
            }
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        // Dispatch 4: Fused inner sort (3-pass LSD, buf_b ↔ buf_a, vals_b ↔ vals_a)
        let inner_params = InnerParams {
            start_shift: 0,
            pass_count: 3,
            batch_start: 0,
        };
        let pso = self.pso_cache.get_or_create_specialized(
            &self.library,
            "sort_inner_fused",
            &[(0, FnConstant::Bool(with_values))],
        );
        encoder.setComputePipelineState(pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(self.buf_bucket_descs.as_ref().unwrap()), 0, 2);
            encoder.setBytes_length_atIndex(
                NonNull::new(&inner_params as *const InnerParams as *mut _).unwrap(),
                std::mem::size_of::<InnerParams>(),
                3,
            );
            if with_values {
                encoder.setBuffer_offset_atIndex(vals_a, 0, 4);
                encoder.setBuffer_offset_atIndex(vals_b, 0, 5);
            }
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);
    }

    /// Encode sort_init_indices dispatch: fills indices[i] = i.
    fn encode_init_indices(
        &mut self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        indices_buf: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
    ) {
        let pso = self.pso_cache.get_or_create(&self.library, "sort_init_indices");
        encoder.setComputePipelineState(pso);
        let count = n as u32;
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(indices_buf), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::new(&count as *const u32 as *mut _).unwrap(),
                4,
                1,
            );
        }
        let grid = MTLSize {
            width: n.div_ceil(THREADS_PER_TG) * THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    }

    /// Return the sorted indices that would sort `data` in ascending order.
    ///
    /// `data[indices[0]] <= data[indices[1]] <= ...`. Input `data` is NOT modified.
    pub fn argsort_u32(&mut self, data: &[u32]) -> Result<Vec<u32>, SortError> {
        let n = data.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if n == 1 {
            return Ok(vec![0]);
        }

        self.ensure_buffers_with_values(n);
        let num_tiles = n.div_ceil(TILE_SIZE);

        // Copy keys to buf_a, zero MSD histogram
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

        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            SortError::GpuExecution("failed to create command buffer".to_string())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            SortError::GpuExecution("failed to create compute encoder".to_string())
        })?;

        // Init indices: buf_vals_a[i] = i
        let vals_a = self.buf_vals_a.as_ref().unwrap().clone();
        let vals_b = self.buf_vals_b.as_ref().unwrap().clone();
        self.encode_init_indices(&enc, &vals_a, n);

        // Sort pipeline with HAS_VALUES=true
        let buf_a = self.buf_a.as_ref().unwrap().clone();
        let buf_b = self.buf_b.as_ref().unwrap().clone();
        self.encode_sort_pipeline_full(
            &enc,
            &buf_a,
            &buf_b,
            Some(&vals_a),
            Some(&vals_b),
            n,
            num_tiles,
            true,
        );

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if cmd.status() == MTLCommandBufferStatus::Error {
            return Err(SortError::GpuExecution(format!(
                "command buffer error: {:?}",
                cmd.error()
            )));
        }

        // Read sorted indices from buf_vals_a
        let mut result = vec![0u32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf_vals_a.as_ref().unwrap().contents().as_ptr() as *const u32,
                result.as_mut_ptr(),
                n,
            );
        }

        Ok(result)
    }

    /// Return the sorted indices for i32 data in ascending order.
    ///
    /// Uses XOR 0x80000000 transform on keys. Input `data` is NOT modified.
    pub fn argsort_i32(&mut self, data: &[i32]) -> Result<Vec<u32>, SortError> {
        let n = data.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if n == 1 {
            return Ok(vec![0]);
        }

        self.ensure_buffers_with_values(n);
        let num_tiles = n.div_ceil(TILE_SIZE);

        // Copy keys to buf_a, zero MSD histogram
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

        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            SortError::GpuExecution("failed to create command buffer".to_string())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            SortError::GpuExecution("failed to create compute encoder".to_string())
        })?;

        let vals_a = self.buf_vals_a.as_ref().unwrap().clone();
        let vals_b = self.buf_vals_b.as_ref().unwrap().clone();
        let buf_a = self.buf_a.as_ref().unwrap().clone();
        let buf_b = self.buf_b.as_ref().unwrap().clone();

        // Init indices
        self.encode_init_indices(&enc, &vals_a, n);

        // Forward transform: XOR 0x80000000 (mode 0) on keys only
        encode_transform_32(&enc, &self.library, &mut self.pso_cache, &buf_a, n, 0);

        // Sort pipeline with values
        self.encode_sort_pipeline_full(
            &enc,
            &buf_a,
            &buf_b,
            Some(&vals_a),
            Some(&vals_b),
            n,
            num_tiles,
            true,
        );

        // No inverse transform needed for argsort (we don't read keys back)

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if cmd.status() == MTLCommandBufferStatus::Error {
            return Err(SortError::GpuExecution(format!(
                "command buffer error: {:?}",
                cmd.error()
            )));
        }

        let mut result = vec![0u32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf_vals_a.as_ref().unwrap().contents().as_ptr() as *const u32,
                result.as_mut_ptr(),
                n,
            );
        }

        Ok(result)
    }

    /// Return the sorted indices for f32 data in ascending order (IEEE 754 total ordering).
    ///
    /// Uses FloatFlip transform on keys. Input `data` is NOT modified.
    pub fn argsort_f32(&mut self, data: &[f32]) -> Result<Vec<u32>, SortError> {
        let n = data.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if n == 1 {
            return Ok(vec![0]);
        }

        self.ensure_buffers_with_values(n);
        let num_tiles = n.div_ceil(TILE_SIZE);

        // Copy keys to buf_a, zero MSD histogram
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

        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            SortError::GpuExecution("failed to create command buffer".to_string())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            SortError::GpuExecution("failed to create compute encoder".to_string())
        })?;

        let vals_a = self.buf_vals_a.as_ref().unwrap().clone();
        let vals_b = self.buf_vals_b.as_ref().unwrap().clone();
        let buf_a = self.buf_a.as_ref().unwrap().clone();
        let buf_b = self.buf_b.as_ref().unwrap().clone();

        // Init indices
        self.encode_init_indices(&enc, &vals_a, n);

        // Forward transform: FloatFlip (mode 1) on keys only
        encode_transform_32(&enc, &self.library, &mut self.pso_cache, &buf_a, n, 1);

        // Sort pipeline with values
        self.encode_sort_pipeline_full(
            &enc,
            &buf_a,
            &buf_b,
            Some(&vals_a),
            Some(&vals_b),
            n,
            num_tiles,
            true,
        );

        // No inverse transform needed for argsort (we don't read keys back)

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if cmd.status() == MTLCommandBufferStatus::Error {
            return Err(SortError::GpuExecution(format!(
                "command buffer error: {:?}",
                cmd.error()
            )));
        }

        let mut result = vec![0u32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf_vals_a.as_ref().unwrap().contents().as_ptr() as *const u32,
                result.as_mut_ptr(),
                n,
            );
        }

        Ok(result)
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

    #[test]
    fn test_sort_i32_basic() {
        let mut sorter = GpuSorter::new().unwrap();
        let mut data = vec![5i32, -3, 0, i32::MIN, i32::MAX, -1, 1];
        sorter.sort_i32(&mut data).unwrap();
        let mut expected = vec![5i32, -3, 0, i32::MIN, i32::MAX, -1, 1];
        expected.sort();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_sort_f32_basic() {
        let mut sorter = GpuSorter::new().unwrap();
        let mut data = vec![
            1.0f32,
            f32::NAN,
            -0.0,
            0.0,
            f32::NEG_INFINITY,
            -1.0,
            f32::INFINITY,
            -f32::NAN,
        ];
        sorter.sort_f32(&mut data).unwrap();
        // Verify total_cmp ordering via to_bits comparison
        let mut expected = vec![
            1.0f32,
            f32::NAN,
            -0.0,
            0.0,
            f32::NEG_INFINITY,
            -1.0,
            f32::INFINITY,
            -f32::NAN,
        ];
        expected.sort_by(f32::total_cmp);
        assert_eq!(
            data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_argsort_u32_basic() {
        let mut sorter = GpuSorter::new().unwrap();
        let data = vec![30u32, 10, 20];
        let indices = sorter.argsort_u32(&data).unwrap();
        assert_eq!(indices, vec![1, 2, 0]);
        assert_eq!(data, vec![30, 10, 20]); // unchanged
    }

    #[test]
    fn test_argsort_u32_empty() {
        let mut sorter = GpuSorter::new().unwrap();
        let data: Vec<u32> = vec![];
        let indices = sorter.argsort_u32(&data).unwrap();
        assert_eq!(indices, vec![]);
    }

    #[test]
    fn test_argsort_u32_single() {
        let mut sorter = GpuSorter::new().unwrap();
        let data = vec![42u32];
        let indices = sorter.argsort_u32(&data).unwrap();
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_argsort_i32_basic() {
        let mut sorter = GpuSorter::new().unwrap();
        let data = vec![5i32, -3, 0, -1, 1];
        let indices = sorter.argsort_i32(&data).unwrap();
        // sorted: -3, -1, 0, 1, 5 -> indices: 1, 3, 2, 4, 0
        assert_eq!(indices, vec![1, 3, 2, 4, 0]);
    }

    #[test]
    fn test_argsort_f32_basic() {
        let mut sorter = GpuSorter::new().unwrap();
        let data = vec![3.0f32, 1.0, 2.0];
        let indices = sorter.argsort_f32(&data).unwrap();
        assert_eq!(indices, vec![1, 2, 0]);
        assert_eq!(data, vec![3.0, 1.0, 2.0]); // unchanged
    }

    #[test]
    fn test_argsort_i32_empty() {
        let mut sorter = GpuSorter::new().unwrap();
        let data: Vec<i32> = vec![];
        let indices = sorter.argsort_i32(&data).unwrap();
        assert_eq!(indices, vec![]);
    }

    #[test]
    fn test_argsort_f32_empty() {
        let mut sorter = GpuSorter::new().unwrap();
        let data: Vec<f32> = vec![];
        let indices = sorter.argsort_f32(&data).unwrap();
        assert_eq!(indices, vec![]);
    }
}
