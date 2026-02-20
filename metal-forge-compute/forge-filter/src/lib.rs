use std::marker::PhantomData;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLLibrary, MTLSize,
};

mod metal_helpers;
use metal_helpers::{alloc_buffer, init_device_and_queue, FnConstant, PsoCache};

// --- FilterKey sealed trait ---

mod private {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
    impl Sealed for u64 {}
    impl Sealed for i64 {}
    impl Sealed for f64 {}
}

/// Trait for types that can be filtered on the GPU.
/// Sealed: only u32, i32, f32, u64, i64, f64 are valid.
pub trait FilterKey: private::Sealed + Copy + PartialOrd + 'static {
    /// Size in bytes (4 or 8).
    const KEY_SIZE: usize;
    /// Whether this is a 64-bit type.
    const IS_64BIT: bool;
    /// Reinterpret threshold value as raw bits for Metal.
    fn to_bits(self) -> u64;
}

impl FilterKey for u32 {
    const KEY_SIZE: usize = 4;
    const IS_64BIT: bool = false;
    fn to_bits(self) -> u64 {
        self as u64
    }
}

impl FilterKey for i32 {
    const KEY_SIZE: usize = 4;
    const IS_64BIT: bool = false;
    fn to_bits(self) -> u64 {
        (self as u32) as u64
    }
}

impl FilterKey for f32 {
    const KEY_SIZE: usize = 4;
    const IS_64BIT: bool = false;
    fn to_bits(self) -> u64 {
        self.to_bits() as u64
    }
}

impl FilterKey for u64 {
    const KEY_SIZE: usize = 8;
    const IS_64BIT: bool = true;
    fn to_bits(self) -> u64 {
        self
    }
}

impl FilterKey for i64 {
    const KEY_SIZE: usize = 8;
    const IS_64BIT: bool = true;
    fn to_bits(self) -> u64 {
        self as u64
    }
}

impl FilterKey for f64 {
    const KEY_SIZE: usize = 8;
    const IS_64BIT: bool = true;
    fn to_bits(self) -> u64 {
        self.to_bits()
    }
}

// --- Predicate enum ---

/// GPU-evaluated predicate for numeric filtering.
#[derive(Clone, Debug)]
pub enum Predicate<T: FilterKey> {
    /// `val > threshold`
    Gt(T),
    /// `val < threshold`
    Lt(T),
    /// `val >= threshold`
    Ge(T),
    /// `val <= threshold`
    Le(T),
    /// `val == threshold`
    Eq(T),
    /// `val != threshold`
    Ne(T),
    /// `lo <= val <= hi` (inclusive both ends)
    Between(T, T),
    /// All sub-predicates must match (single-column AND).
    And(Vec<Predicate<T>>),
    /// Any sub-predicate must match (single-column OR).
    Or(Vec<Predicate<T>>),
}

impl<T: FilterKey> Predicate<T> {
    /// Returns the function constant value for PRED_TYPE (0-7).
    ///
    /// 0=GT, 1=LT, 2=GE, 3=LE, 4=EQ, 5=NE, 6=BETWEEN, 7=TRUE (passthrough).
    /// Compound predicates (And/Or) return 7 (TRUE) since they are
    /// decomposed into multiple passes or mask composition.
    pub fn pred_type_id(&self) -> u32 {
        match self {
            Predicate::Gt(_) => 0,
            Predicate::Lt(_) => 1,
            Predicate::Ge(_) => 2,
            Predicate::Le(_) => 3,
            Predicate::Eq(_) => 4,
            Predicate::Ne(_) => 5,
            Predicate::Between(_, _) => 6,
            Predicate::And(_) => 7,
            Predicate::Or(_) => 7,
        }
    }

    /// Returns (lo_bits, hi_bits) for Metal shader params.
    ///
    /// For single-threshold predicates, lo_bits is the threshold and hi_bits is 0.
    /// For Between, lo_bits and hi_bits are the range endpoints.
    /// For compound predicates, returns (0, 0) — they use multi-pass evaluation.
    pub fn to_bits(&self) -> (u64, u64) {
        match self {
            Predicate::Gt(v)
            | Predicate::Lt(v)
            | Predicate::Ge(v)
            | Predicate::Le(v)
            | Predicate::Eq(v)
            | Predicate::Ne(v) => (v.to_bits(), 0),
            Predicate::Between(lo, hi) => (lo.to_bits(), hi.to_bits()),
            Predicate::And(_) | Predicate::Or(_) => (0, 0),
        }
    }
}

// --- FilterError ---

/// Errors that can occur during GPU filter operations.
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    /// No Metal GPU device found on this system.
    #[error("no Metal GPU device found")]
    DeviceNotFound,

    /// Metal shader compilation failed.
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// GPU execution failed (command buffer error).
    #[error("GPU execution failed: {0}")]
    GpuExecution(String),

    /// Empty input — use CPU for N < 1M.
    #[error("empty input (use CPU for N < 1M)")]
    EmptyInput,

    /// Invalid predicate configuration.
    #[error("invalid predicate: {0}")]
    InvalidPredicate(String),
}

// --- FilterParams (repr(C), shared with Metal) ---

/// Parameters for 32-bit filter kernels.
/// Layout matches the Metal `FilterParams` struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FilterParams {
    /// Total number of elements to filter.
    pub element_count: u32,
    /// Number of tiles: ceil(element_count / TILE_SIZE).
    pub num_tiles: u32,
    /// Reinterpreted bits of the lo threshold (or sole threshold).
    pub lo_bits: u32,
    /// Reinterpreted bits of the hi threshold (BETWEEN only, else 0).
    pub hi_bits: u32,
}

/// Parameters for 64-bit filter kernels.
/// Layout matches the Metal `FilterParams64` struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FilterParams64 {
    /// Total number of elements to filter.
    pub element_count: u32,
    /// Number of tiles: ceil(element_count / TILE_SIZE).
    pub num_tiles: u32,
    /// Low 32 bits of lo threshold.
    pub lo_lo: u32,
    /// High 32 bits of lo threshold.
    pub lo_hi: u32,
    /// Low 32 bits of hi threshold (BETWEEN only).
    pub hi_lo: u32,
    /// High 32 bits of hi threshold.
    pub hi_hi: u32,
    /// Padding to align to 32 bytes.
    pub _pad: [u32; 2],
}

// --- Constants ---

/// Threads per threadgroup for filter kernels.
pub const FILTER_THREADS: u32 = 256;

/// Elements per tile for 32-bit types (FILTER_THREADS * 16 elements/thread).
pub const FILTER_TILE_32: u32 = 4096;

/// Elements per tile for 64-bit types (FILTER_THREADS * 8 elements/thread).
pub const FILTER_TILE_64: u32 = 2048;

/// GPU filter+compact engine for Apple Silicon.
///
/// Uses a 3-dispatch scan-based pipeline for ordered filtering:
/// 1. `filter_predicate_scan` — evaluate predicate + local scan + write TG totals
/// 2. `filter_scan_partials` — single-TG exclusive scan of TG totals
/// 3. `filter_scatter` — re-evaluate predicate + scatter matching elements
///
/// All dispatches execute within a single command buffer and compute encoder,
/// using implicit memory barriers between dispatches.
pub struct GpuFilter {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pso_cache: PsoCache,
    // Scratch buffers (grow-only, reused across calls)
    buf_partials: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_idx: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_count: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    scratch_capacity: usize, // in elements
}

impl GpuFilter {
    /// Initialize Metal device, queue, and compile filter kernel PSOs.
    ///
    /// Pre-compiles 15 PSOs at construction: 7 predicate types x 2 kernels
    /// (filter_predicate_scan + filter_scatter) with IS_64BIT=false, plus
    /// filter_scan_partials. Additional PSO variants are compiled lazily.
    pub fn new() -> Result<Self, FilterError> {
        let (device, queue) = init_device_and_queue();

        // Load the filter metallib (embedded path from build.rs)
        let metallib_path = env!("FILTER_METALLIB_PATH");
        let path_ns = NSString::from_str(metallib_path);
        #[allow(deprecated)]
        let library = device
            .newLibraryWithFile_error(&path_ns)
            .map_err(|e| FilterError::ShaderCompilation(format!("{:?}", e)))?;

        let mut pso_cache = PsoCache::new();

        // Pre-compile 32-bit PSOs for all 7 predicate types (GT=0..BETWEEN=6)
        // filter_predicate_scan needs: PRED_TYPE(0) + IS_64BIT(1)
        // filter_scatter needs: PRED_TYPE(0) + IS_64BIT(1) + OUTPUT_IDX(2) + OUTPUT_VALS(3)
        for pred in 0..=6u32 {
            pso_cache.get_or_create_specialized(
                &library,
                "filter_predicate_scan",
                &[(0, FnConstant::U32(pred)), (1, FnConstant::Bool(false))],
            );
            pso_cache.get_or_create_specialized(
                &library,
                "filter_scatter",
                &[
                    (0, FnConstant::U32(pred)),
                    (1, FnConstant::Bool(false)),
                    (2, FnConstant::Bool(false)),
                    (3, FnConstant::Bool(true)),
                ],
            );
        }
        // filter_scan_partials needs no function constants
        pso_cache.get_or_create(&library, "filter_scan_partials");

        Ok(Self {
            device,
            queue,
            library,
            pso_cache,
            buf_partials: None,
            buf_output: None,
            buf_output_idx: None,
            buf_count: None,
            scratch_capacity: 0,
        })
    }

    /// Allocate a GPU buffer for zero-copy filtering.
    ///
    /// Returns a [`FilterBuffer<T>`] backed by unified memory (`StorageModeShared`).
    pub fn alloc_filter_buffer<T: FilterKey>(&self, capacity: usize) -> FilterBuffer<T> {
        let buffer = alloc_buffer(&self.device, capacity * T::KEY_SIZE);
        FilterBuffer {
            buffer,
            len: 0,
            capacity,
            _marker: PhantomData,
        }
    }

    /// Ensure scratch buffers are large enough for `n` elements of size `elem_size`.
    /// Grow-only: never shrinks. Reused across calls.
    fn ensure_scratch_buffers(&mut self, n: usize, elem_size: usize) {
        if n <= self.scratch_capacity && self.buf_partials.is_some() {
            return;
        }

        let tile_size = if elem_size == 8 {
            FILTER_TILE_64 as usize
        } else {
            FILTER_TILE_32 as usize
        };
        let num_tiles = n.div_ceil(tile_size);

        // partials: one u32 per tile
        self.buf_partials = Some(alloc_buffer(&self.device, num_tiles * 4));
        // output values: worst case = all match
        self.buf_output = Some(alloc_buffer(&self.device, n * elem_size));
        // output indices: worst case = all match, u32 per index
        self.buf_output_idx = Some(alloc_buffer(&self.device, n * 4));
        // count: single u32
        if self.buf_count.is_none() {
            self.buf_count = Some(alloc_buffer(&self.device, 4));
        }

        self.scratch_capacity = n;
    }

    /// Internal: encode and execute the 3-dispatch filter pipeline.
    ///
    /// Returns the number of matching elements.
    fn dispatch_filter<T: FilterKey>(
        &mut self,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
        pred: &Predicate<T>,
        output_vals: bool,
        output_idx: bool,
    ) -> Result<usize, FilterError> {
        if n == 0 {
            return Ok(0);
        }

        let tile_size = if T::IS_64BIT {
            FILTER_TILE_64 as usize
        } else {
            FILTER_TILE_32 as usize
        };
        let num_tiles = n.div_ceil(tile_size);

        // Build params
        let (lo_bits, hi_bits) = pred.to_bits();
        let pred_type = pred.pred_type_id();

        let tg_size = MTLSize {
            width: FILTER_THREADS as usize,
            height: 1,
            depth: 1,
        };
        let tile_grid = MTLSize {
            width: num_tiles,
            height: 1,
            depth: 1,
        };
        let one_tg_grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        let buf_partials = self.buf_partials.as_ref().unwrap();
        let buf_output = self.buf_output.as_ref().unwrap();
        let buf_output_idx = self.buf_output_idx.as_ref().unwrap();
        let buf_count = self.buf_count.as_ref().unwrap();

        // Zero count buffer
        unsafe {
            std::ptr::write_bytes(buf_count.contents().as_ptr() as *mut u8, 0, 4);
        }

        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            FilterError::GpuExecution("failed to create command buffer".to_string())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            FilterError::GpuExecution("failed to create compute encoder".to_string())
        })?;

        if T::IS_64BIT {
            let params = FilterParams64 {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                lo_lo: lo_bits as u32,
                lo_hi: (lo_bits >> 32) as u32,
                hi_lo: hi_bits as u32,
                hi_hi: (hi_bits >> 32) as u32,
                _pad: [0, 0],
            };

            // Dispatch 1: filter_predicate_scan
            let pso = self.pso_cache.get_or_create_specialized(
                &self.library,
                "filter_predicate_scan",
                &[
                    (0, FnConstant::U32(pred_type)),
                    (1, FnConstant::Bool(true)),
                ],
            );
            enc.setComputePipelineState(pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_partials), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const FilterParams64 as *mut _).unwrap(),
                    std::mem::size_of::<FilterParams64>(),
                    2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(tile_grid, tg_size);

            // Dispatch 2: filter_scan_partials
            let num_tiles_u32 = num_tiles as u32;
            let pso = self
                .pso_cache
                .get_or_create(&self.library, "filter_scan_partials");
            enc.setComputePipelineState(pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_partials), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::new(&num_tiles_u32 as *const u32 as *mut _).unwrap(),
                    4,
                    1,
                );
                enc.setBuffer_offset_atIndex(Some(buf_count), 0, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

            // Dispatch 3: filter_scatter
            let pso = self.pso_cache.get_or_create_specialized(
                &self.library,
                "filter_scatter",
                &[
                    (0, FnConstant::U32(pred_type)),
                    (1, FnConstant::Bool(true)),
                    (2, FnConstant::Bool(output_idx)),
                    (3, FnConstant::Bool(output_vals)),
                ],
            );
            enc.setComputePipelineState(pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_output), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_output_idx), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_partials), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const FilterParams64 as *mut _).unwrap(),
                    std::mem::size_of::<FilterParams64>(),
                    4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(tile_grid, tg_size);
        } else {
            let params = FilterParams {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                lo_bits: lo_bits as u32,
                hi_bits: hi_bits as u32,
            };

            // Dispatch 1: filter_predicate_scan
            let pso = self.pso_cache.get_or_create_specialized(
                &self.library,
                "filter_predicate_scan",
                &[
                    (0, FnConstant::U32(pred_type)),
                    (1, FnConstant::Bool(false)),
                ],
            );
            enc.setComputePipelineState(pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_partials), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const FilterParams as *mut _).unwrap(),
                    std::mem::size_of::<FilterParams>(),
                    2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(tile_grid, tg_size);

            // Dispatch 2: filter_scan_partials
            let num_tiles_u32 = num_tiles as u32;
            let pso = self
                .pso_cache
                .get_or_create(&self.library, "filter_scan_partials");
            enc.setComputePipelineState(pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_partials), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::new(&num_tiles_u32 as *const u32 as *mut _).unwrap(),
                    4,
                    1,
                );
                enc.setBuffer_offset_atIndex(Some(buf_count), 0, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

            // Dispatch 3: filter_scatter
            let pso = self.pso_cache.get_or_create_specialized(
                &self.library,
                "filter_scatter",
                &[
                    (0, FnConstant::U32(pred_type)),
                    (1, FnConstant::Bool(false)),
                    (2, FnConstant::Bool(output_idx)),
                    (3, FnConstant::Bool(output_vals)),
                ],
            );
            enc.setComputePipelineState(pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_output), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_output_idx), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_partials), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const FilterParams as *mut _).unwrap(),
                    std::mem::size_of::<FilterParams>(),
                    4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(tile_grid, tg_size);
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if MTLCommandBufferStatus::Error == cmd.status() {
            return Err(FilterError::GpuExecution(format!(
                "command buffer error: {:?}",
                cmd.error()
            )));
        }

        // Read count from count_buf[0]
        let count = unsafe { *(buf_count.contents().as_ptr() as *const u32) } as usize;
        Ok(count)
    }

    /// Filter a [`FilterBuffer`], returning matching values.
    ///
    /// Executes the 3-dispatch ordered pipeline. Output preserves input order.
    pub fn filter<T: FilterKey>(
        &mut self,
        buf: &FilterBuffer<T>,
        pred: &Predicate<T>,
    ) -> Result<FilterResult<T>, FilterError> {
        let n = buf.len();
        if n == 0 {
            return Ok(FilterResult {
                count: 0,
                values_buf: None,
                indices_buf: None,
                capacity: 0,
                _marker: PhantomData,
            });
        }

        self.ensure_scratch_buffers(n, T::KEY_SIZE);
        let count = self.dispatch_filter(&buf.buffer, n, pred, true, false)?;

        Ok(FilterResult {
            count,
            values_buf: self.buf_output.clone(),
            indices_buf: None,
            capacity: n,
            _marker: PhantomData,
        })
    }

    /// Filter a slice of `u32`, returning matching values as a `Vec<u32>`.
    ///
    /// Convenience method that copies data to GPU, filters, and copies results back.
    /// For zero-copy performance, use [`alloc_filter_buffer`](Self::alloc_filter_buffer)
    /// + [`filter`](Self::filter).
    pub fn filter_u32(
        &mut self,
        data: &[u32],
        pred: &Predicate<u32>,
    ) -> Result<Vec<u32>, FilterError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let mut buf = self.alloc_filter_buffer::<u32>(data.len());
        buf.copy_from_slice(data);
        let result = self.filter(&buf, pred)?;
        Ok(result.to_vec())
    }
}

// --- FilterBuffer<T> ---

/// A Metal buffer for GPU filter input with zero-copy access.
///
/// Created via [`GpuFilter::alloc_filter_buffer`]. The buffer uses `StorageModeShared`
/// (unified memory), so CPU reads/writes go directly to the same physical pages the GPU uses.
pub struct FilterBuffer<T: FilterKey> {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    capacity: usize,
    _marker: PhantomData<T>,
}

impl<T: FilterKey> FilterBuffer<T> {
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

    /// Get a slice to read data directly from GPU-visible memory.
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.buffer.contents().as_ptr() as *const T, self.len)
        }
    }

    /// Get a mutable slice to write data directly into GPU-visible memory.
    /// Returns a slice over the full capacity — call `set_len()` after writing.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.contents().as_ptr() as *mut T,
                self.capacity,
            )
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

    /// Access the underlying Metal buffer (for pipeline integration).
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

// --- FilterResult<T> ---

/// Result of a GPU filter operation.
///
/// Contains the filtered values and optionally the original indices of matching elements.
/// Buffers use `StorageModeShared` for zero-copy CPU access.
pub struct FilterResult<T: FilterKey> {
    count: usize,
    values_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    indices_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    capacity: usize,
    _marker: PhantomData<T>,
}

impl<T: FilterKey> FilterResult<T> {
    /// Number of elements that matched the predicate.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether no elements matched.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get a slice of the filtered values.
    ///
    /// Returns an empty slice if no values buffer is present (e.g., index-only mode).
    pub fn as_slice(&self) -> &[T] {
        match &self.values_buf {
            Some(buf) => unsafe {
                std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, self.count)
            },
            None => &[],
        }
    }

    /// Get a slice of the original indices of matching elements.
    ///
    /// Returns `None` if index output was not requested.
    pub fn indices(&self) -> Option<&[u32]> {
        self.indices_buf.as_ref().map(|buf| unsafe {
            std::slice::from_raw_parts(buf.contents().as_ptr() as *const u32, self.count)
        })
    }

    /// Copy filtered values to a new `Vec<T>`.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    /// Access the underlying Metal buffer for filtered values.
    ///
    /// Returns `None` if no values buffer is present (index-only mode).
    pub fn metal_buffer(&self) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        self.values_buf.as_deref()
    }
}
