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
    /// Data type ID for Metal function constant DATA_TYPE (index 4).
    /// 0 = unsigned (uint/ulong), 1 = signed (int/long), 2 = float/double.
    const DATA_TYPE_ID: u32;
    /// Reinterpret threshold value as raw bits for Metal.
    fn to_bits(self) -> u64;
}

impl FilterKey for u32 {
    const KEY_SIZE: usize = 4;
    const IS_64BIT: bool = false;
    const DATA_TYPE_ID: u32 = 0; // unsigned
    fn to_bits(self) -> u64 {
        self as u64
    }
}

impl FilterKey for i32 {
    const KEY_SIZE: usize = 4;
    const IS_64BIT: bool = false;
    const DATA_TYPE_ID: u32 = 1; // signed
    fn to_bits(self) -> u64 {
        (self as u32) as u64
    }
}

impl FilterKey for f32 {
    const KEY_SIZE: usize = 4;
    const IS_64BIT: bool = false;
    const DATA_TYPE_ID: u32 = 2; // float
    fn to_bits(self) -> u64 {
        self.to_bits() as u64
    }
}

impl FilterKey for u64 {
    const KEY_SIZE: usize = 8;
    const IS_64BIT: bool = true;
    const DATA_TYPE_ID: u32 = 0; // unsigned
    fn to_bits(self) -> u64 {
        self
    }
}

impl FilterKey for i64 {
    const KEY_SIZE: usize = 8;
    const IS_64BIT: bool = true;
    const DATA_TYPE_ID: u32 = 1; // signed
    fn to_bits(self) -> u64 {
        self as u64
    }
}

impl FilterKey for f64 {
    const KEY_SIZE: usize = 8;
    const IS_64BIT: bool = true;
    const DATA_TYPE_ID: u32 = 2; // float/double
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

    /// Returns true if this is a simple (non-compound) predicate that can be
    /// dispatched directly to the GPU in a single pass.
    pub fn is_simple(&self) -> bool {
        !matches!(self, Predicate::And(_) | Predicate::Or(_))
    }

    /// Simplify the predicate tree, detecting common patterns.
    ///
    /// Currently detects:
    /// - `And([Ge(lo), Le(hi)])` -> `Between(lo, hi)`
    /// - `And([Gt(lo), Lt(hi)])` is NOT converted (open interval != BETWEEN which is closed)
    /// - Single-element `And([p])` or `Or([p])` -> p
    /// - Nested `And(And(...))` and `Or(Or(...))` are flattened
    pub fn simplify(self) -> Predicate<T> {
        match self {
            Predicate::And(preds) => {
                // Flatten nested ANDs and simplify children
                let mut flat = Vec::new();
                for p in preds {
                    let simplified = p.simplify();
                    match simplified {
                        Predicate::And(inner) => flat.extend(inner),
                        other => flat.push(other),
                    }
                }

                // Single-element AND -> unwrap
                if flat.len() == 1 {
                    return flat.into_iter().next().unwrap();
                }

                // Detect Ge(lo) + Le(hi) -> Between(lo, hi)
                if flat.len() == 2 {
                    let (a, b) = (&flat[0], &flat[1]);
                    match (a, b) {
                        (Predicate::Ge(lo), Predicate::Le(hi)) => {
                            return Predicate::Between(*lo, *hi);
                        }
                        (Predicate::Le(hi), Predicate::Ge(lo)) => {
                            return Predicate::Between(*lo, *hi);
                        }
                        _ => {}
                    }
                }

                Predicate::And(flat)
            }
            Predicate::Or(preds) => {
                // Flatten nested ORs and simplify children
                let mut flat = Vec::new();
                for p in preds {
                    let simplified = p.simplify();
                    match simplified {
                        Predicate::Or(inner) => flat.extend(inner),
                        other => flat.push(other),
                    }
                }

                // Single-element OR -> unwrap
                if flat.len() == 1 {
                    return flat.into_iter().next().unwrap();
                }

                Predicate::Or(flat)
            }
            // Simple predicates pass through
            other => other,
        }
    }

    /// Evaluate this predicate against a single value on the CPU.
    ///
    /// Used as a fallback for compound predicates that cannot be dispatched
    /// to the GPU in a single pass.
    pub fn evaluate(&self, val: &T) -> bool {
        match self {
            Predicate::Gt(t) => val.partial_cmp(t) == Some(std::cmp::Ordering::Greater),
            Predicate::Lt(t) => val.partial_cmp(t) == Some(std::cmp::Ordering::Less),
            Predicate::Ge(t) => matches!(
                val.partial_cmp(t),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            ),
            Predicate::Le(t) => matches!(
                val.partial_cmp(t),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ),
            Predicate::Eq(t) => val.partial_cmp(t) == Some(std::cmp::Ordering::Equal),
            Predicate::Ne(t) => val.partial_cmp(t) != Some(std::cmp::Ordering::Equal),
            Predicate::Between(lo, hi) => {
                matches!(
                    val.partial_cmp(lo),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                ) && matches!(
                    val.partial_cmp(hi),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
            }
            Predicate::And(preds) => preds.iter().all(|p| p.evaluate(val)),
            Predicate::Or(preds) => preds.iter().any(|p| p.evaluate(val)),
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
        // filter_predicate_scan needs: PRED_TYPE(0) + IS_64BIT(1) + DATA_TYPE(4)
        // filter_scatter needs: PRED_TYPE(0) + IS_64BIT(1) + OUTPUT_IDX(2) + OUTPUT_VALS(3) + DATA_TYPE(4)
        // Pre-compile for u32 (DATA_TYPE=0); other types compiled lazily
        for pred in 0..=6u32 {
            pso_cache.get_or_create_specialized(
                &library,
                "filter_predicate_scan",
                &[
                    (0, FnConstant::U32(pred)),
                    (1, FnConstant::Bool(false)),
                    (4, FnConstant::U32(0)),
                ],
            );
            pso_cache.get_or_create_specialized(
                &library,
                "filter_scatter",
                &[
                    (0, FnConstant::U32(pred)),
                    (1, FnConstant::Bool(false)),
                    (2, FnConstant::Bool(false)),
                    (3, FnConstant::Bool(true)),
                    (4, FnConstant::U32(0)),
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
        let num_tiles = (n + tile_size - 1) / tile_size;

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
        let num_tiles = (n + tile_size - 1) / tile_size;

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

        let data_type = T::DATA_TYPE_ID;

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
                    (4, FnConstant::U32(data_type)),
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
                    (4, FnConstant::U32(data_type)),
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
                    (4, FnConstant::U32(data_type)),
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
                    (4, FnConstant::U32(data_type)),
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
    /// For compound predicates (And/Or), simplifies first (e.g. And([Ge, Le]) -> Between),
    /// then cascades AND through multiple GPU passes, or falls back to CPU for OR.
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
                _capacity: 0,
                _marker: PhantomData,
            });
        }

        // Try simplifying compound predicates first
        let simplified = pred.clone().simplify();
        if simplified.is_simple() {
            self.ensure_scratch_buffers(n, T::KEY_SIZE);
            let count = self.dispatch_filter(&buf.buffer, n, &simplified, true, false)?;
            return Ok(FilterResult {
                count,
                values_buf: self.buf_output.clone(),
                indices_buf: None,
                _capacity: n,
                _marker: PhantomData,
            });
        }

        // For AND of all-simple sub-predicates: cascade through GPU
        if let Predicate::And(ref subs) = simplified {
            if subs.iter().all(|s| s.is_simple()) {
                return self.filter_compound_and_cascade(buf, subs);
            }
        }

        // General fallback: CPU evaluation for compound predicates
        let data = buf.as_slice();
        let result: Vec<T> = data.iter().filter(|v| simplified.evaluate(v)).copied().collect();
        self.ensure_scratch_buffers(n, T::KEY_SIZE);
        let buf_output = self.buf_output.as_ref().unwrap();
        let count = result.len();
        if count > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    result.as_ptr(),
                    buf_output.contents().as_ptr() as *mut T,
                    count,
                );
            }
        }
        Ok(FilterResult {
            count,
            values_buf: self.buf_output.clone(),
            indices_buf: None,
            _capacity: n,
            _marker: PhantomData,
        })
    }

    /// Internal: cascade AND of simple predicates through multiple GPU passes.
    ///
    /// Filters with the first sub-predicate, then filters the result with the second,
    /// and so on. Each pass uses the 3-dispatch ordered pipeline.
    fn filter_compound_and_cascade<T: FilterKey>(
        &mut self,
        buf: &FilterBuffer<T>,
        subs: &[Predicate<T>],
    ) -> Result<FilterResult<T>, FilterError> {
        let n = buf.len();
        if subs.is_empty() || n == 0 {
            return Ok(FilterResult {
                count: 0,
                values_buf: None,
                indices_buf: None,
                _capacity: 0,
                _marker: PhantomData,
            });
        }

        // First pass: filter from the original buffer
        self.ensure_scratch_buffers(n, T::KEY_SIZE);
        let count = self.dispatch_filter(&buf.buffer, n, &subs[0], true, false)?;
        if count == 0 || subs.len() == 1 {
            return Ok(FilterResult {
                count,
                values_buf: self.buf_output.clone(),
                indices_buf: None,
                _capacity: n,
                _marker: PhantomData,
            });
        }

        // Subsequent passes: filter from the previous output.
        // We need a temp buffer to hold intermediate results since dispatch_filter
        // reads from input and writes to buf_output.
        let temp_buf = alloc_buffer(&self.device, n * T::KEY_SIZE);
        let mut current_count = count;

        for sub in &subs[1..] {
            // Copy current output to temp
            unsafe {
                let src = self.buf_output.as_ref().unwrap().contents().as_ptr() as *const u8;
                let dst = temp_buf.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src, dst, current_count * T::KEY_SIZE);
            }

            self.ensure_scratch_buffers(current_count, T::KEY_SIZE);
            current_count = self.dispatch_filter(&temp_buf, current_count, sub, true, false)?;
            if current_count == 0 {
                break;
            }
        }

        Ok(FilterResult {
            count: current_count,
            values_buf: self.buf_output.clone(),
            indices_buf: None,
            _capacity: n,
            _marker: PhantomData,
        })
    }

    /// Filter a [`FilterBuffer`], returning only the indices of matching elements.
    ///
    /// Executes the 3-dispatch ordered pipeline. Output indices are in ascending order.
    /// No values are copied — only the u32 indices of matching elements.
    pub fn filter_indices<T: FilterKey>(
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
                _capacity: 0,
                _marker: PhantomData,
            });
        }

        self.ensure_scratch_buffers(n, T::KEY_SIZE);
        let count = self.dispatch_filter(&buf.buffer, n, pred, false, true)?;

        Ok(FilterResult {
            count,
            values_buf: None,
            indices_buf: self.buf_output_idx.clone(),
            _capacity: n,
            _marker: PhantomData,
        })
    }

    /// Filter a [`FilterBuffer`], returning both matching values and their indices.
    ///
    /// Executes the 3-dispatch ordered pipeline. Both values and indices preserve input order.
    pub fn filter_with_indices<T: FilterKey>(
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
                _capacity: 0,
                _marker: PhantomData,
            });
        }

        self.ensure_scratch_buffers(n, T::KEY_SIZE);
        let count = self.dispatch_filter(&buf.buffer, n, pred, true, true)?;

        Ok(FilterResult {
            count,
            values_buf: self.buf_output.clone(),
            indices_buf: self.buf_output_idx.clone(),
            _capacity: n,
            _marker: PhantomData,
        })
    }

    /// Filter a [`FilterBuffer`] using unordered atomic scatter (single dispatch).
    ///
    /// Uses the `filter_atomic_scatter` kernel with SIMD-aggregated atomics.
    /// Output order is **non-deterministic** but the result set is identical to
    /// [`filter`](Self::filter). Faster than ordered mode for aggregation queries
    /// where output order does not matter.
    pub fn filter_unordered<T: FilterKey>(
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
                _capacity: 0,
                _marker: PhantomData,
            });
        }

        self.ensure_scratch_buffers(n, T::KEY_SIZE);

        let tile_size = if T::IS_64BIT {
            FILTER_TILE_64 as usize
        } else {
            FILTER_TILE_32 as usize
        };
        let num_tiles = (n + tile_size - 1) / tile_size;

        let (lo_bits, hi_bits) = pred.to_bits();
        let pred_type = pred.pred_type_id();
        let data_type = T::DATA_TYPE_ID;

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

        let buf_output = self.buf_output.as_ref().unwrap();
        let buf_output_idx = self.buf_output_idx.as_ref().unwrap();
        let buf_count = self.buf_count.as_ref().unwrap();

        // Zero the atomic counter before dispatch
        unsafe {
            std::ptr::write_bytes(buf_count.contents().as_ptr() as *mut u8, 0, 4);
        }

        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            FilterError::GpuExecution("failed to create command buffer".to_string())
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            FilterError::GpuExecution("failed to create compute encoder".to_string())
        })?;

        let pso = self.pso_cache.get_or_create_specialized(
            &self.library,
            "filter_atomic_scatter",
            &[
                (0, FnConstant::U32(pred_type)),
                (1, FnConstant::Bool(T::IS_64BIT)),
                (2, FnConstant::Bool(false)),  // OUTPUT_IDX
                (3, FnConstant::Bool(true)),   // OUTPUT_VALS
                (4, FnConstant::U32(data_type)),
            ],
        );
        enc.setComputePipelineState(pso);

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
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf.metal_buffer()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_output), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_output_idx), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_count), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const FilterParams64 as *mut _).unwrap(),
                    std::mem::size_of::<FilterParams64>(),
                    4,
                );
            }
        } else {
            let params = FilterParams {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                lo_bits: lo_bits as u32,
                hi_bits: hi_bits as u32,
            };
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf.metal_buffer()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_output), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_output_idx), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_count), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const FilterParams as *mut _).unwrap(),
                    std::mem::size_of::<FilterParams>(),
                    4,
                );
            }
        }

        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_grid, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if MTLCommandBufferStatus::Error == cmd.status() {
            return Err(FilterError::GpuExecution(format!(
                "command buffer error: {:?}",
                cmd.error()
            )));
        }

        // Read count from atomic counter
        let count = unsafe { *(buf_count.contents().as_ptr() as *const u32) } as usize;

        Ok(FilterResult {
            count,
            values_buf: self.buf_output.clone(),
            indices_buf: None,
            _capacity: n,
            _marker: PhantomData,
        })
    }

    /// Filter a slice of `u32`, returning matching values as a `Vec<u32>`.
    ///
    /// Convenience method that copies data to GPU, filters, and copies results back.
    /// Compound predicates (And/Or) are supported: And([Ge, Le]) is optimized to Between,
    /// AND of simple predicates cascades through GPU passes, OR uses CPU fallback.
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

    /// Filter a slice of `i32`, returning matching values as a `Vec<i32>`.
    pub fn filter_i32(
        &mut self,
        data: &[i32],
        pred: &Predicate<i32>,
    ) -> Result<Vec<i32>, FilterError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let mut buf = self.alloc_filter_buffer::<i32>(data.len());
        buf.copy_from_slice(data);
        let result = self.filter(&buf, pred)?;
        Ok(result.to_vec())
    }

    /// Filter a slice of `f32`, returning matching values as a `Vec<f32>`.
    ///
    /// NaN handling follows IEEE 754: NaN comparisons return false for all
    /// ordered predicates (Gt, Lt, Ge, Le, Eq). NaN != NaN returns true.
    pub fn filter_f32(
        &mut self,
        data: &[f32],
        pred: &Predicate<f32>,
    ) -> Result<Vec<f32>, FilterError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let mut buf = self.alloc_filter_buffer::<f32>(data.len());
        buf.copy_from_slice(data);
        let result = self.filter(&buf, pred)?;
        Ok(result.to_vec())
    }

    /// Filter a slice of `u64`, returning matching values as a `Vec<u64>`.
    pub fn filter_u64(
        &mut self,
        data: &[u64],
        pred: &Predicate<u64>,
    ) -> Result<Vec<u64>, FilterError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let mut buf = self.alloc_filter_buffer::<u64>(data.len());
        buf.copy_from_slice(data);
        let result = self.filter(&buf, pred)?;
        Ok(result.to_vec())
    }

    /// Filter a slice of `i64`, returning matching values as a `Vec<i64>`.
    pub fn filter_i64(
        &mut self,
        data: &[i64],
        pred: &Predicate<i64>,
    ) -> Result<Vec<i64>, FilterError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let mut buf = self.alloc_filter_buffer::<i64>(data.len());
        buf.copy_from_slice(data);
        let result = self.filter(&buf, pred)?;
        Ok(result.to_vec())
    }

    /// Filter a slice of `f64`, returning matching values as a `Vec<f64>`.
    ///
    /// Note: f64 has 1/32 ALU throughput on Apple Silicon M4 Pro, but filter
    /// operations remain bandwidth-bound (single comparison per element).
    pub fn filter_f64(
        &mut self,
        data: &[f64],
        pred: &Predicate<f64>,
    ) -> Result<Vec<f64>, FilterError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let mut buf = self.alloc_filter_buffer::<f64>(data.len());
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
    _capacity: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_u32_gt_basic() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let pred = Predicate::Gt(500_000u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");

        let cpu_ref: Vec<u32> = data.iter().filter(|&&x| x > 500_000).copied().collect();

        println!(
            "test_filter_u32_gt_basic: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );

        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        assert_eq!(result, cpu_ref, "contents mismatch");
    }

    #[test]
    fn test_filter_u32_gt_16m() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..16_000_000u32).collect();
        let pred = Predicate::Gt(8_000_000u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");

        let cpu_ref: Vec<u32> = data.iter().filter(|&&x| x > 8_000_000).copied().collect();

        println!(
            "test_filter_u32_gt_16m: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );

        assert_eq!(result.len(), cpu_ref.len(), "length mismatch at 16M");

        // Spot-check first 100 + last 100 elements (don't compare all for speed)
        let check_n = 100.min(result.len());
        assert_eq!(
            &result[..check_n],
            &cpu_ref[..check_n],
            "first {} elements mismatch",
            check_n
        );
        if result.len() > check_n {
            assert_eq!(
                &result[result.len() - check_n..],
                &cpu_ref[cpu_ref.len() - check_n..],
                "last {} elements mismatch",
                check_n
            );
        }
    }

    #[test]
    fn test_filter_u32_empty_result() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        // Max value is 999_999, so Gt(2_000_000) matches nothing
        let pred = Predicate::Gt(2_000_000u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");

        println!("test_filter_u32_empty_result: GPU={}", result.len());
        assert_eq!(result.len(), 0, "expected empty result");
    }

    #[test]
    fn test_filter_u32_all_match() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        // Gt(0) matches all except 0 itself → 999_999 matches
        let pred = Predicate::Gt(0u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");

        println!("test_filter_u32_all_match: GPU={}", result.len());
        assert_eq!(result.len(), 999_999, "expected 999_999 matches");
    }

    #[test]
    fn test_filter_u32_lt() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let pred = Predicate::Lt(500_000u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");
        let cpu_ref: Vec<u32> = data.iter().filter(|&&x| x < 500_000).copied().collect();

        println!(
            "test_filter_u32_lt: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        assert_eq!(result, cpu_ref, "contents mismatch");
    }

    #[test]
    fn test_filter_u32_ge() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let pred = Predicate::Ge(500_000u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");
        let cpu_ref: Vec<u32> = data.iter().filter(|&&x| x >= 500_000).copied().collect();

        println!(
            "test_filter_u32_ge: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        assert_eq!(result, cpu_ref, "contents mismatch");
    }

    #[test]
    fn test_filter_u32_le() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let pred = Predicate::Le(500_000u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");
        let cpu_ref: Vec<u32> = data.iter().filter(|&&x| x <= 500_000).copied().collect();

        println!(
            "test_filter_u32_le: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        assert_eq!(result, cpu_ref, "contents mismatch");
    }

    #[test]
    fn test_filter_u32_eq() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        // Exactly one match: the value 500_000
        let pred = Predicate::Eq(500_000u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");
        let cpu_ref: Vec<u32> = data.iter().filter(|&&x| x == 500_000).copied().collect();

        println!(
            "test_filter_u32_eq: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        assert_eq!(result.len(), 1, "expected exactly 1 match");
        assert_eq!(result, cpu_ref, "contents mismatch");
    }

    #[test]
    fn test_filter_u32_ne() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        // All except 500_000 → 999_999 matches
        let pred = Predicate::Ne(500_000u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");
        let cpu_ref: Vec<u32> = data.iter().filter(|&&x| x != 500_000).copied().collect();

        println!(
            "test_filter_u32_ne: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        assert_eq!(result.len(), 999_999, "expected 999_999 matches");
        assert_eq!(result, cpu_ref, "contents mismatch");
    }

    // --- Type-specific tests ---

    #[test]
    fn test_filter_i32_gt() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        // Range -500_000..500_000
        let data: Vec<i32> = (-500_000..500_000i32).collect();
        let pred = Predicate::Gt(-250_000i32);

        let result = filter.filter_i32(&data, &pred).expect("filter_i32 failed");
        let cpu_ref: Vec<i32> = data.iter().filter(|&&x| x > -250_000).copied().collect();

        println!(
            "test_filter_i32_gt: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        // Spot-check first and last elements
        let check_n = 100.min(result.len());
        assert_eq!(
            &result[..check_n],
            &cpu_ref[..check_n],
            "first {} elements mismatch",
            check_n
        );
        if result.len() > check_n {
            assert_eq!(
                &result[result.len() - check_n..],
                &cpu_ref[cpu_ref.len() - check_n..],
                "last {} elements mismatch",
                check_n
            );
        }
    }

    #[test]
    fn test_filter_f32_gt() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        // Range 0.0..1.0 in steps of 1e-6
        let data: Vec<f32> = (0..1_000_000u32).map(|i| i as f32 / 1_000_000.0).collect();
        let pred = Predicate::Gt(0.5f32);

        let result = filter.filter_f32(&data, &pred).expect("filter_f32 failed");
        let cpu_ref: Vec<f32> = data.iter().filter(|&&x| x > 0.5).copied().collect();

        println!(
            "test_filter_f32_gt: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        let check_n = 100.min(result.len());
        assert_eq!(
            &result[..check_n],
            &cpu_ref[..check_n],
            "first {} elements mismatch",
            check_n
        );
    }

    #[test]
    fn test_filter_u64_gt() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u64> = (0..1_000_000u64).collect();
        let pred = Predicate::Gt(500_000u64);

        let result = filter.filter_u64(&data, &pred).expect("filter_u64 failed");
        let cpu_ref: Vec<u64> = data.iter().filter(|&&x| x > 500_000).copied().collect();

        println!(
            "test_filter_u64_gt: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        let check_n = 100.min(result.len());
        assert_eq!(
            &result[..check_n],
            &cpu_ref[..check_n],
            "first {} elements mismatch",
            check_n
        );
        if result.len() > check_n {
            assert_eq!(
                &result[result.len() - check_n..],
                &cpu_ref[cpu_ref.len() - check_n..],
                "last {} elements mismatch",
                check_n
            );
        }
    }

    #[test]
    fn test_filter_i64_gt() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<i64> = (-500_000..500_000i64).collect();
        let pred = Predicate::Gt(-250_000i64);

        let result = filter.filter_i64(&data, &pred).expect("filter_i64 failed");
        let cpu_ref: Vec<i64> = data.iter().filter(|&&x| x > -250_000).copied().collect();

        println!(
            "test_filter_i64_gt: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        let check_n = 100.min(result.len());
        assert_eq!(
            &result[..check_n],
            &cpu_ref[..check_n],
            "first {} elements mismatch",
            check_n
        );
        if result.len() > check_n {
            assert_eq!(
                &result[result.len() - check_n..],
                &cpu_ref[cpu_ref.len() - check_n..],
                "last {} elements mismatch",
                check_n
            );
        }
    }

    #[test]
    fn test_filter_f64_gt() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<f64> = (0..1_000_000u64).map(|i| i as f64 / 1_000_000.0).collect();
        let pred = Predicate::Gt(0.5f64);

        let result = filter.filter_f64(&data, &pred).expect("filter_f64 failed");
        let cpu_ref: Vec<f64> = data.iter().filter(|&&x| x > 0.5).copied().collect();

        println!(
            "test_filter_f64_gt: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        let check_n = 100.min(result.len());
        assert_eq!(
            &result[..check_n],
            &cpu_ref[..check_n],
            "first {} elements mismatch",
            check_n
        );
    }

    // --- BETWEEN predicate tests ---

    #[test]
    fn test_filter_u32_between() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let pred = Predicate::Between(250_000u32, 750_000u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");
        let cpu_ref: Vec<u32> = data
            .iter()
            .filter(|&&x| x >= 250_000 && x <= 750_000)
            .copied()
            .collect();

        println!(
            "test_filter_u32_between: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        assert_eq!(result, cpu_ref, "contents mismatch");
    }

    #[test]
    fn test_filter_f32_between() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<f32> = (0..1_000_000u32).map(|i| i as f32 / 1_000_000.0).collect();
        let pred = Predicate::Between(0.25f32, 0.75f32);

        let result = filter.filter_f32(&data, &pred).expect("filter_f32 failed");
        let cpu_ref: Vec<f32> = data
            .iter()
            .filter(|&&x| x >= 0.25 && x <= 0.75)
            .copied()
            .collect();

        println!(
            "test_filter_f32_between: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        // Spot-check first and last 100
        let check_n = 100.min(result.len());
        assert_eq!(
            &result[..check_n],
            &cpu_ref[..check_n],
            "first {} elements mismatch",
            check_n
        );
        if result.len() > check_n {
            assert_eq!(
                &result[result.len() - check_n..],
                &cpu_ref[cpu_ref.len() - check_n..],
                "last {} elements mismatch",
                check_n
            );
        }
    }

    #[test]
    fn test_filter_between_inclusive() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        // Use small range so we can verify both endpoints are included
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let lo = 100_000u32;
        let hi = 900_000u32;
        let pred = Predicate::Between(lo, hi);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");

        // Verify lo endpoint is included
        assert!(
            result.contains(&lo),
            "lo endpoint {} must be included",
            lo
        );
        // Verify hi endpoint is included
        assert!(
            result.contains(&hi),
            "hi endpoint {} must be included",
            hi
        );
        // Verify value just below lo is excluded
        assert!(
            !result.contains(&(lo - 1)),
            "value {} below lo must be excluded",
            lo - 1
        );
        // Verify value just above hi is excluded
        assert!(
            !result.contains(&(hi + 1)),
            "value {} above hi must be excluded",
            hi + 1
        );

        let expected_count = (hi - lo + 1) as usize;
        println!(
            "test_filter_between_inclusive: GPU={}, expected={}",
            result.len(),
            expected_count
        );
        assert_eq!(result.len(), expected_count, "count mismatch");
    }

    #[test]
    fn test_filter_between_eq() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let target = 500_000u32;

        // Between(x, x) should match exactly 1 element, same as Eq(x)
        let pred_between = Predicate::Between(target, target);
        let pred_eq = Predicate::Eq(target);

        let result_between = filter
            .filter_u32(&data, &pred_between)
            .expect("filter between failed");
        let result_eq = filter
            .filter_u32(&data, &pred_eq)
            .expect("filter eq failed");

        println!(
            "test_filter_between_eq: Between={}, Eq={}",
            result_between.len(),
            result_eq.len()
        );
        assert_eq!(result_between.len(), 1, "Between(x,x) should match exactly 1");
        assert_eq!(result_between, result_eq, "Between(x,x) must equal Eq(x)");
    }

    #[test]
    fn test_filter_between_inverted() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        // Between(750_000, 250_000): val >= 750_000 && val <= 250_000 is always false
        let pred = Predicate::Between(750_000u32, 250_000u32);

        let result = filter.filter_u32(&data, &pred).expect("filter_u32 failed");

        println!("test_filter_between_inverted: GPU={}", result.len());
        assert_eq!(result.len(), 0, "inverted Between should return empty");
    }

    #[test]
    fn test_filter_f32_nan() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");

        // Create data with NaN values interspersed
        let mut data: Vec<f32> = Vec::with_capacity(1_000_000);
        for i in 0..1_000_000u32 {
            if i % 1000 == 0 {
                data.push(f32::NAN);
            } else {
                data.push(i as f32 / 1_000_000.0);
            }
        }

        // Test 1: Gt(0.0) should exclude NaN values (NaN > 0.0 = false)
        let pred_gt = Predicate::Gt(0.0f32);
        let result_gt = filter
            .filter_f32(&data, &pred_gt)
            .expect("filter_f32 Gt failed");
        let cpu_ref_gt: Vec<f32> = data.iter().filter(|&&x| x > 0.0).copied().collect();

        println!(
            "test_filter_f32_nan Gt: GPU={}, CPU={}",
            result_gt.len(),
            cpu_ref_gt.len()
        );
        assert_eq!(
            result_gt.len(),
            cpu_ref_gt.len(),
            "Gt length mismatch (NaN should be excluded)"
        );
        // Verify no NaN in output
        assert!(
            !result_gt.iter().any(|x| x.is_nan()),
            "Gt output should not contain NaN"
        );

        // Test 2: Ne(0.0) should include NaN values (NaN != 0.0 = true)
        let pred_ne = Predicate::Ne(0.0f32);
        let result_ne = filter
            .filter_f32(&data, &pred_ne)
            .expect("filter_f32 Ne failed");
        let cpu_ref_ne: Vec<f32> = data.iter().filter(|&&x| x != 0.0).copied().collect();

        println!(
            "test_filter_f32_nan Ne: GPU={}, CPU={}",
            result_ne.len(),
            cpu_ref_ne.len()
        );
        assert_eq!(
            result_ne.len(),
            cpu_ref_ne.len(),
            "Ne length mismatch (NaN should be included)"
        );
        // Count NaN in Ne output — should be 1000 (every 1000th element)
        let nan_count = result_ne.iter().filter(|x| x.is_nan()).count();
        println!("NaN count in Ne result: {}", nan_count);
        assert_eq!(nan_count, 1000, "expected 1000 NaN values in Ne output");
    }

    // --- Index output mode tests ---

    #[test]
    fn test_filter_indices_ascending() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let pred = Predicate::Gt(500_000u32);

        let mut buf = filter.alloc_filter_buffer::<u32>(data.len());
        buf.copy_from_slice(&data);
        let result = filter
            .filter_indices(&buf, &pred)
            .expect("filter_indices failed");

        let indices = result.indices().expect("indices() should be Some");
        println!(
            "test_filter_indices_ascending: count={}, indices len={}",
            result.len(),
            indices.len()
        );

        // Verify count matches CPU reference
        let cpu_count = data.iter().filter(|&&x| x > 500_000).count();
        assert_eq!(result.len(), cpu_count, "count mismatch");

        // Verify indices are sorted ascending (ordered mode)
        for i in 1..indices.len() {
            assert!(
                indices[i] > indices[i - 1],
                "indices not ascending at pos {}: {} >= {}",
                i,
                indices[i - 1],
                indices[i]
            );
        }

        // Verify values_buf is None (index-only mode)
        assert!(
            result.as_slice().is_empty(),
            "as_slice should be empty in index-only mode"
        );

        // Verify first and last indices make sense
        assert_eq!(indices[0], 500_001, "first matching index should be 500_001");
        assert_eq!(
            indices[indices.len() - 1],
            999_999,
            "last matching index should be 999_999"
        );
    }

    #[test]
    fn test_filter_with_indices() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let pred = Predicate::Gt(500_000u32);

        let mut buf = filter.alloc_filter_buffer::<u32>(data.len());
        buf.copy_from_slice(&data);
        let result = filter
            .filter_with_indices(&buf, &pred)
            .expect("filter_with_indices failed");

        let values = result.as_slice();
        let indices = result.indices().expect("indices() should be Some");

        println!(
            "test_filter_with_indices: values={}, indices={}",
            values.len(),
            indices.len()
        );

        // Both should have the same length
        assert_eq!(
            values.len(),
            indices.len(),
            "values and indices must have same length"
        );

        // Verify count matches CPU reference
        let cpu_ref: Vec<u32> = data.iter().filter(|&&x| x > 500_000).copied().collect();
        assert_eq!(result.len(), cpu_ref.len(), "count mismatch");

        // Verify values match CPU reference
        assert_eq!(values, cpu_ref.as_slice(), "values mismatch");

        // Verify indices are ascending
        for i in 1..indices.len() {
            assert!(
                indices[i] > indices[i - 1],
                "indices not ascending at pos {}",
                i
            );
        }
    }

    #[test]
    fn test_filter_indices_match_values() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        // Use non-sequential data to make this test meaningful
        let data: Vec<u32> = (0..1_000_000u32).map(|i| i * 3 + 7).collect();
        let pred = Predicate::Gt(1_500_000u32);

        let mut buf = filter.alloc_filter_buffer::<u32>(data.len());
        buf.copy_from_slice(&data);

        // Get both values and indices
        let result = filter
            .filter_with_indices(&buf, &pred)
            .expect("filter_with_indices failed");

        let values = result.as_slice();
        let indices = result.indices().expect("indices() should be Some");

        println!(
            "test_filter_indices_match_values: count={}",
            result.len()
        );
        assert!(result.len() > 0, "expected some matches");

        // Gather original data at the reported indices and compare to values output
        for (i, &idx) in indices.iter().enumerate() {
            let original_val = data[idx as usize];
            assert_eq!(
                values[i], original_val,
                "mismatch at result pos {}: value={} but data[{}]={}",
                i, values[i], idx, original_val
            );
        }

        // Also verify the indices point to elements that actually match the predicate
        for &idx in indices.iter() {
            assert!(
                data[idx as usize] > 1_500_000,
                "index {} points to value {} which doesn't match Gt(1_500_000)",
                idx,
                data[idx as usize]
            );
        }
    }

    // --- Unordered atomic scatter tests ---

    #[test]
    fn test_filter_unordered_set_eq() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let pred = Predicate::Gt(500_000u32);

        // Get ordered result
        let ordered = filter.filter_u32(&data, &pred).expect("ordered filter failed");

        // Get unordered result via filter_unordered
        let mut buf = filter.alloc_filter_buffer::<u32>(data.len());
        buf.copy_from_slice(&data);
        let unordered_result = filter
            .filter_unordered(&buf, &pred)
            .expect("unordered filter failed");
        let mut unordered = unordered_result.to_vec();

        println!(
            "test_filter_unordered_set_eq: ordered={}, unordered={}",
            ordered.len(),
            unordered.len()
        );

        // Sort unordered result for comparison
        unordered.sort();
        let mut ordered_sorted = ordered.clone();
        ordered_sorted.sort();

        assert_eq!(
            unordered.len(),
            ordered_sorted.len(),
            "length mismatch: unordered={} vs ordered={}",
            unordered.len(),
            ordered_sorted.len()
        );
        assert_eq!(
            unordered, ordered_sorted,
            "sorted sets differ between ordered and unordered modes"
        );
    }

    #[test]
    fn test_filter_unordered_count() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let pred = Predicate::Gt(500_000u32);

        // Get ordered count
        let ordered = filter.filter_u32(&data, &pred).expect("ordered filter failed");

        // Get unordered count
        let mut buf = filter.alloc_filter_buffer::<u32>(data.len());
        buf.copy_from_slice(&data);
        let unordered_result = filter
            .filter_unordered(&buf, &pred)
            .expect("unordered filter failed");

        println!(
            "test_filter_unordered_count: ordered={}, unordered={}",
            ordered.len(),
            unordered_result.len()
        );

        assert_eq!(
            unordered_result.len(),
            ordered.len(),
            "unordered count {} != ordered count {}",
            unordered_result.len(),
            ordered.len()
        );

        // Also verify against CPU reference
        let cpu_count = data.iter().filter(|&&x| x > 500_000).count();
        assert_eq!(
            unordered_result.len(),
            cpu_count,
            "unordered count {} != CPU count {}",
            unordered_result.len(),
            cpu_count
        );
    }

    // --- Compound predicate tests ---

    #[test]
    fn test_compound_and() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000u32).collect();
        // And([Gt(100), Lt(900)]) -> values 101..=899 -> 799 matches
        let pred = Predicate::And(vec![Predicate::Gt(100u32), Predicate::Lt(900u32)]);

        let result = filter.filter_u32(&data, &pred).expect("compound AND failed");
        let cpu_ref: Vec<u32> = data.iter().filter(|&&x| x > 100 && x < 900).copied().collect();

        println!(
            "test_compound_and: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        assert_eq!(result, cpu_ref, "contents mismatch");
    }

    #[test]
    fn test_compound_or() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000u32).collect();
        // Or([Lt(100), Gt(900)]) -> values 0..=99 ∪ 901..=999 -> 199 matches
        let pred = Predicate::Or(vec![Predicate::Lt(100u32), Predicate::Gt(900u32)]);

        let result = filter.filter_u32(&data, &pred).expect("compound OR failed");
        let cpu_ref: Vec<u32> = data
            .iter()
            .filter(|&&x| x < 100 || x > 900)
            .copied()
            .collect();

        println!(
            "test_compound_or: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        assert_eq!(result, cpu_ref, "contents mismatch");
    }

    #[test]
    fn test_compound_nested() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000u32).collect();
        // And([Or([Lt(100), Gt(900)]), Ne(950)])
        // -> (x < 100 || x > 900) && x != 950
        // -> {0..=99, 901..=949, 951..=999} -> 198 matches
        let pred = Predicate::And(vec![
            Predicate::Or(vec![Predicate::Lt(100u32), Predicate::Gt(900u32)]),
            Predicate::Ne(950u32),
        ]);

        let result = filter
            .filter_u32(&data, &pred)
            .expect("compound nested failed");
        let cpu_ref: Vec<u32> = data
            .iter()
            .filter(|&&x| (x < 100 || x > 900) && x != 950)
            .copied()
            .collect();

        println!(
            "test_compound_nested: GPU={}, CPU={}",
            result.len(),
            cpu_ref.len()
        );
        assert_eq!(result.len(), cpu_ref.len(), "length mismatch");
        assert_eq!(result, cpu_ref, "contents mismatch");
    }

    // ===================================================================
    // Exhaustive 42-combo type x operator correctness matrix (Task 3.1)
    // ===================================================================
    //
    // 6 types (u32, i32, f32, u64, i64, f64) x 7 predicates (Gt, Lt, Ge, Le, Eq, Ne, Between)
    // = 42 combinations, each at 100K elements with ~50% selectivity.
    // Uses deterministic RNG (ChaCha8Rng seed 42) for reproducibility.

    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Helper: compare GPU filter output to CPU reference for a single type+predicate combo.
    fn test_filter_correctness<T: FilterKey + PartialOrd + std::fmt::Debug>(
        data: &[T],
        pred: &Predicate<T>,
        filter: &mut GpuFilter,
        label: &str,
    ) {
        // CPU reference using Predicate::evaluate()
        let cpu_ref: Vec<T> = data.iter().filter(|v| pred.evaluate(v)).copied().collect();

        // GPU path via FilterBuffer
        let mut buf = filter.alloc_filter_buffer::<T>(data.len());
        buf.copy_from_slice(data);
        let result = filter.filter(&buf, pred).expect("GPU filter failed");
        let gpu_out = result.to_vec();

        println!(
            "  {}: GPU={}, CPU={} (selectivity {:.1}%)",
            label,
            gpu_out.len(),
            cpu_ref.len(),
            100.0 * cpu_ref.len() as f64 / data.len() as f64
        );

        assert_eq!(
            gpu_out.len(),
            cpu_ref.len(),
            "{}: count mismatch GPU={} vs CPU={}",
            label,
            gpu_out.len(),
            cpu_ref.len()
        );
        // Full content comparison for correctness
        assert_eq!(gpu_out, cpu_ref, "{}: content mismatch", label);
    }

    macro_rules! matrix_test {
        ($test_name:ident, $T:ty, $type_label:expr, $gen_data:path, $preds_fn:path) => {
            #[test]
            fn $test_name() {
                let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                let data: Vec<$T> = (0..100_000).map(|_| $gen_data(&mut rng)).collect();

                println!("test_matrix_{}: {} elements", $type_label, data.len());

                let preds = $preds_fn(&data);
                for (pred_label, pred) in &preds {
                    let label = format!("{}_{}", $type_label, pred_label);
                    test_filter_correctness(&data, pred, &mut filter, &label);
                }
            }
        };
    }

    // Helper to pick threshold/range values that give ~50% selectivity.
    // For sorted data, median = 50th percentile, p25/p75 for Between.

    fn u32_preds(data: &[u32]) -> Vec<(&str, Predicate<u32>)> {
        let mut sorted = data.to_vec();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        let p25 = sorted[sorted.len() / 4];
        let p75 = sorted[3 * sorted.len() / 4];
        // For Eq: pick median value (appears at least once since it comes from the data)
        let eq_val = median;
        vec![
            ("gt", Predicate::Gt(median)),
            ("lt", Predicate::Lt(median)),
            ("ge", Predicate::Ge(median)),
            ("le", Predicate::Le(median)),
            ("eq", Predicate::Eq(eq_val)),
            ("ne", Predicate::Ne(eq_val)),
            ("between", Predicate::Between(p25, p75)),
        ]
    }

    fn i32_preds(data: &[i32]) -> Vec<(&str, Predicate<i32>)> {
        let mut sorted = data.to_vec();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        let p25 = sorted[sorted.len() / 4];
        let p75 = sorted[3 * sorted.len() / 4];
        let eq_val = median;
        vec![
            ("gt", Predicate::Gt(median)),
            ("lt", Predicate::Lt(median)),
            ("ge", Predicate::Ge(median)),
            ("le", Predicate::Le(median)),
            ("eq", Predicate::Eq(eq_val)),
            ("ne", Predicate::Ne(eq_val)),
            ("between", Predicate::Between(p25, p75)),
        ]
    }

    fn f32_preds(data: &[f32]) -> Vec<(&str, Predicate<f32>)> {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let p25 = sorted[sorted.len() / 4];
        let p75 = sorted[3 * sorted.len() / 4];
        let eq_val = median;
        vec![
            ("gt", Predicate::Gt(median)),
            ("lt", Predicate::Lt(median)),
            ("ge", Predicate::Ge(median)),
            ("le", Predicate::Le(median)),
            ("eq", Predicate::Eq(eq_val)),
            ("ne", Predicate::Ne(eq_val)),
            ("between", Predicate::Between(p25, p75)),
        ]
    }

    fn u64_preds(data: &[u64]) -> Vec<(&str, Predicate<u64>)> {
        let mut sorted = data.to_vec();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        let p25 = sorted[sorted.len() / 4];
        let p75 = sorted[3 * sorted.len() / 4];
        let eq_val = median;
        vec![
            ("gt", Predicate::Gt(median)),
            ("lt", Predicate::Lt(median)),
            ("ge", Predicate::Ge(median)),
            ("le", Predicate::Le(median)),
            ("eq", Predicate::Eq(eq_val)),
            ("ne", Predicate::Ne(eq_val)),
            ("between", Predicate::Between(p25, p75)),
        ]
    }

    fn i64_preds(data: &[i64]) -> Vec<(&str, Predicate<i64>)> {
        let mut sorted = data.to_vec();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        let p25 = sorted[sorted.len() / 4];
        let p75 = sorted[3 * sorted.len() / 4];
        let eq_val = median;
        vec![
            ("gt", Predicate::Gt(median)),
            ("lt", Predicate::Lt(median)),
            ("ge", Predicate::Ge(median)),
            ("le", Predicate::Le(median)),
            ("eq", Predicate::Eq(eq_val)),
            ("ne", Predicate::Ne(eq_val)),
            ("between", Predicate::Between(p25, p75)),
        ]
    }

    fn f64_preds(data: &[f64]) -> Vec<(&str, Predicate<f64>)> {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let p25 = sorted[sorted.len() / 4];
        let p75 = sorted[3 * sorted.len() / 4];
        let eq_val = median;
        vec![
            ("gt", Predicate::Gt(median)),
            ("lt", Predicate::Lt(median)),
            ("ge", Predicate::Ge(median)),
            ("le", Predicate::Le(median)),
            ("eq", Predicate::Eq(eq_val)),
            ("ne", Predicate::Ne(eq_val)),
            ("between", Predicate::Between(p25, p75)),
        ]
    }

    fn gen_u32(rng: &mut ChaCha8Rng) -> u32 { rng.gen_range(0u32..1_000_000) }
    fn gen_i32(rng: &mut ChaCha8Rng) -> i32 { rng.gen_range(-500_000i32..500_000) }
    fn gen_f32(rng: &mut ChaCha8Rng) -> f32 { rng.gen_range(0.0f32..1.0) }
    fn gen_u64(rng: &mut ChaCha8Rng) -> u64 { rng.gen_range(0u64..1_000_000) }
    fn gen_i64(rng: &mut ChaCha8Rng) -> i64 { rng.gen_range(-500_000i64..500_000) }
    fn gen_f64(rng: &mut ChaCha8Rng) -> f64 { rng.gen_range(0.0f64..1.0) }

    matrix_test!(test_matrix_u32, u32, "u32", gen_u32, u32_preds);
    matrix_test!(test_matrix_i32, i32, "i32", gen_i32, i32_preds);
    matrix_test!(test_matrix_f32, f32, "f32", gen_f32, f32_preds);
    matrix_test!(test_matrix_u64, u64, "u64", gen_u64, u64_preds);
    matrix_test!(test_matrix_i64, i64, "i64", gen_i64, i64_preds);
    matrix_test!(test_matrix_f64, f64, "f64", gen_f64, f64_preds);

    #[test]
    fn test_compound_and_as_between() {
        let mut filter = GpuFilter::new().expect("GpuFilter::new failed");
        let data: Vec<u32> = (0..1_000_000u32).collect();
        let lo = 250_000u32;
        let hi = 750_000u32;

        // And([Ge(lo), Le(hi)]) should be optimized to Between(lo, hi)
        let pred_compound = Predicate::And(vec![Predicate::Ge(lo), Predicate::Le(hi)]);
        let pred_between = Predicate::Between(lo, hi);

        // Verify simplify() detects the pattern
        let simplified = pred_compound.clone().simplify();
        match &simplified {
            Predicate::Between(a, b) => {
                println!(
                    "test_compound_and_as_between: simplified to Between({}, {})",
                    a.to_bits(),
                    b.to_bits()
                );
                assert_eq!(a.to_bits(), lo.to_bits(), "lo mismatch");
                assert_eq!(b.to_bits(), hi.to_bits(), "hi mismatch");
            }
            other => panic!(
                "expected Between after simplify, got {:?}",
                std::mem::discriminant(other)
            ),
        }

        // Verify results match
        let result_compound = filter
            .filter_u32(&data, &pred_compound)
            .expect("compound filter failed");
        let result_between = filter
            .filter_u32(&data, &pred_between)
            .expect("between filter failed");

        println!(
            "test_compound_and_as_between: compound={}, between={}",
            result_compound.len(),
            result_between.len()
        );
        assert_eq!(
            result_compound.len(),
            result_between.len(),
            "length mismatch between compound AND and Between"
        );
        assert_eq!(
            result_compound, result_between,
            "contents mismatch between compound AND and Between"
        );
    }
}
