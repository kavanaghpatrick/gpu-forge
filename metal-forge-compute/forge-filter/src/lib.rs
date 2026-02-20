use std::marker::PhantomData;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

mod metal_helpers;

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
pub struct GpuFilter;

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
