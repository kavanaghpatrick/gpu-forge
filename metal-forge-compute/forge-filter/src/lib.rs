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
