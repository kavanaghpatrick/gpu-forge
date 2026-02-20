//! Autonomous query executor with persistent kernel and re-dispatch chain.
//!
//! This module implements:
//! - `FusedPsoCache`: compiled pipeline state cache for the AOT fused query kernel,
//!   keyed by (filter_count, agg_count, has_group_by) function constant triple.
//! - `RedispatchChain`: persistent kernel re-dispatch chain using Metal completion handlers.
//!   Creates a continuous chain of 16ms time-slice command buffers where each completion
//!   handler immediately dispatches the next. Idle detection stops the chain after 500ms
//!   of no queries.
//! - `AutonomousExecutor`: unified executor combining all autonomous components.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions, MTLSharedEvent, MTLSize,
};

use super::jit::JitCompiler;
use super::loader::{BinaryColumnarLoader, ColumnInfo, ResidentTable};
use super::types::{AggSpec, FilterSpec, OutputBuffer, QueryParamsSlot};
use super::work_queue::WorkQueue;
use crate::sql::physical_plan::PhysicalPlan;
use crate::sql::types::{AggFunc, CompareOp, Value};
use crate::storage::columnar::ColumnarBatch;
use crate::storage::schema::RuntimeSchema;

/// Cache of compiled PSOs for the fused query kernel, keyed by function constant triple.
///
/// Each unique (filter_count, agg_count, has_group_by) combination produces a specialized
/// Metal pipeline via function constants, avoiding runtime branches in the shader.
pub struct FusedPsoCache {
    #[allow(dead_code)]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    cache: HashMap<(u32, u32, bool), Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl FusedPsoCache {
    /// Create a new empty cache bound to the given device and shader library.
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        library: Retained<ProtocolObject<dyn MTLLibrary>>,
    ) -> Self {
        Self {
            device,
            library,
            cache: HashMap::new(),
        }
    }

    /// Get a cached PSO or compile a new one for the given function constant values.
    ///
    /// - `filter_count`: number of active filter predicates (0..4), maps to FILTER_COUNT (index 0)
    /// - `agg_count`: number of active aggregate functions (0..5), maps to AGG_COUNT (index 1)
    /// - `has_group_by`: whether GROUP BY is active, maps to HAS_GROUP_BY (index 2)
    pub fn get_or_compile(
        &mut self,
        filter_count: u32,
        agg_count: u32,
        has_group_by: bool,
    ) -> Result<&ProtocolObject<dyn MTLComputePipelineState>, String> {
        let key = (filter_count, agg_count, has_group_by);

        if !self.cache.contains_key(&key) {
            let pso = compile_fused_pso(&self.library, filter_count, agg_count, has_group_by)?;
            self.cache.insert(key, pso);
        }

        Ok(&self.cache[&key])
    }

    /// Number of cached PSOs.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

/// Compile a specialized PSO for the `fused_query` kernel with the given function constants.
///
/// Uses MTLFunctionConstantValues to set:
/// - Index 0: FILTER_COUNT (uint)
/// - Index 1: AGG_COUNT (uint)
/// - Index 2: HAS_GROUP_BY (bool)
///
/// The Metal compiler produces a specialized version of the kernel with dead code eliminated
/// for the given constant values, yielding optimal GPU performance.
fn compile_fused_pso(
    library: &ProtocolObject<dyn MTLLibrary>,
    filter_count: u32,
    agg_count: u32,
    has_group_by: bool,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    // Create function constant values
    let constant_values = MTLFunctionConstantValues::new();

    // Index 0: FILTER_COUNT (uint/u32)
    unsafe {
        let ptr = NonNull::new(&filter_count as *const u32 as *mut std::ffi::c_void)
            .expect("filter_count pointer is null");
        constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::UInt, 0);
    }

    // Index 1: AGG_COUNT (uint/u32)
    unsafe {
        let ptr = NonNull::new(&agg_count as *const u32 as *mut std::ffi::c_void)
            .expect("agg_count pointer is null");
        constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::UInt, 1);
    }

    // Index 2: HAS_GROUP_BY (bool)
    let group_by_val: u8 = if has_group_by { 1 } else { 0 };
    unsafe {
        let ptr = NonNull::new(&group_by_val as *const u8 as *mut std::ffi::c_void)
            .expect("has_group_by pointer is null");
        constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::Bool, 2);
    }

    // Get specialized function from library
    let fn_name = NSString::from_str("fused_query");
    let function = library
        .newFunctionWithName_constantValues_error(&fn_name, &constant_values)
        .map_err(|e| format!("Failed to create fused_query function with constants (filter={}, agg={}, group_by={}): {:?}", filter_count, agg_count, has_group_by, e))?;

    // Create compute pipeline state
    let device = library.device();
    let pso = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| format!("Failed to create fused_query PSO (filter={}, agg={}, group_by={}): {:?}", filter_count, agg_count, has_group_by, e))?;

    Ok(pso)
}

/// Execute a fused query kernel via one-shot Metal dispatch (standard command buffer path).
///
/// This is the POC dispatch path: create command buffer, encode, dispatch, waitUntilCompleted.
/// Proves kernel correctness before adding persistent complexity (Phase 4).
///
/// # Arguments
/// * `command_queue` - Metal command queue to create command buffers from
/// * `pso` - Compiled fused query pipeline state (from FusedPsoCache)
/// * `params` - Query parameters (filters, aggregates, group_by, row_count)
/// * `resident_table` - GPU-resident table data (column data + metadata buffers)
///
/// # Returns
/// The `OutputBuffer` read back from GPU memory after kernel completion.
pub fn execute_fused_oneshot(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    params: &QueryParamsSlot,
    resident_table: &ResidentTable,
) -> Result<OutputBuffer, String> {
    let options = MTLResourceOptions::StorageModeShared;

    // 1. Allocate and populate params Metal buffer (512 bytes)
    let params_size = std::mem::size_of::<QueryParamsSlot>();
    let params_buffer = device
        .newBufferWithLength_options(params_size, options)
        .ok_or_else(|| "Failed to allocate params Metal buffer".to_string())?;

    unsafe {
        let dst = params_buffer.contents().as_ptr() as *mut QueryParamsSlot;
        std::ptr::copy_nonoverlapping(params as *const QueryParamsSlot, dst, 1);
    }

    // 2. Allocate output Metal buffer (22560 bytes), zero-initialized
    let output_size = std::mem::size_of::<OutputBuffer>();
    let output_buffer = device
        .newBufferWithLength_options(output_size, options)
        .ok_or_else(|| "Failed to allocate output Metal buffer".to_string())?;

    unsafe {
        let dst = output_buffer.contents().as_ptr() as *mut u8;
        std::ptr::write_bytes(dst, 0, output_size);
    }

    // 2b. Initialize MIN/MAX sentinel values in the output buffer.
    // The zero-initialized output works for COUNT/SUM (identity = 0), but MIN needs
    // INT64_MAX (so first real value wins) and MAX needs INT64_MIN.
    // We iterate over all agg slots and set sentinels for MIN/MAX aggregates.
    unsafe {
        let out_ptr = output_buffer.contents().as_ptr() as *mut OutputBuffer;
        let out = &mut *out_ptr;
        for a in 0..params.agg_count as usize {
            let agg_func = params.aggs[a].agg_func;
            let is_min = agg_func == 3; // AGG_FUNC_MIN
            let is_max = agg_func == 4; // AGG_FUNC_MAX
            if is_min || is_max {
                for g in 0..super::types::MAX_GROUPS {
                    if is_min {
                        out.agg_results[g][a].value_int = i64::MAX;
                        out.agg_results[g][a].value_float = f32::MAX;
                    } else {
                        out.agg_results[g][a].value_int = i64::MIN;
                        out.agg_results[g][a].value_float = f32::MIN;
                    }
                }
            }
        }
    }

    // 3. Create command buffer
    let cmd_buf = command_queue
        .commandBuffer()
        .ok_or_else(|| "Failed to create command buffer".to_string())?;

    // 4. Create compute command encoder and configure dispatch
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| "Failed to create compute command encoder".to_string())?;

        // Set pipeline state
        encoder.setComputePipelineState(pso);

        // Set buffers matching kernel signature:
        //   buffer(0) = QueryParamsSlot
        //   buffer(1) = column data
        //   buffer(2) = ColumnMeta array
        //   buffer(3) = OutputBuffer
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*params_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*resident_table.data_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*resident_table.column_meta_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*output_buffer), 0, 3);
        }

        // Calculate dispatch dimensions
        let row_count = params.row_count as usize;
        let threads_per_threadgroup = 256usize;
        let threadgroup_count = if row_count == 0 {
            1 // Must dispatch at least 1 threadgroup for metadata
        } else {
            row_count.div_ceil(threads_per_threadgroup)
        };

        let grid_size = MTLSize {
            width: threadgroup_count,
            height: 1,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: threads_per_threadgroup,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);

        // End encoding
        encoder.endEncoding();
    }

    // 5. Commit and wait (one-shot blocking path)
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    // 6. Read back OutputBuffer from GPU memory
    let result = unsafe {
        let src = output_buffer.contents().as_ptr() as *const OutputBuffer;
        std::ptr::read(src)
    };

    Ok(result)
}

/// Execute a JIT-compiled fused query kernel via one-shot Metal dispatch.
///
/// This function combines JIT compilation with the same dispatch logic as
/// `execute_fused_oneshot`. It:
/// 1. Compiles the plan via JIT (cache hit on repeated structures)
/// 2. Dispatches the JIT PSO with the given parameters
/// 3. Reads back the output buffer
///
/// # Arguments
/// * `device` - Metal device for buffer allocation
/// * `command_queue` - Metal command queue for command buffer creation
/// * `jit_compiler` - JIT compiler instance (with PSO cache)
/// * `plan` - Physical query plan (used for JIT compilation)
/// * `schema` - Column schema for JIT source generation
/// * `params` - Query parameters (filters, aggregates, group_by, row_count)
/// * `resident_table` - GPU-resident table data
///
/// # Returns
/// The `OutputBuffer` read back from GPU memory after kernel completion.
pub fn execute_jit_oneshot(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    jit_compiler: &mut JitCompiler,
    plan: &PhysicalPlan,
    schema: &[ColumnInfo],
    params: &QueryParamsSlot,
    resident_table: &ResidentTable,
) -> Result<OutputBuffer, String> {
    // 1. Compile via JIT (cache hit on repeated plan structures)
    let compiled = jit_compiler.compile(plan, schema)?;
    let pso = &compiled.pso;

    let options = MTLResourceOptions::StorageModeShared;

    // 2. Allocate and populate params Metal buffer (512 bytes)
    let params_size = std::mem::size_of::<QueryParamsSlot>();
    let params_buffer = device
        .newBufferWithLength_options(params_size, options)
        .ok_or_else(|| "Failed to allocate params Metal buffer".to_string())?;

    unsafe {
        let dst = params_buffer.contents().as_ptr() as *mut QueryParamsSlot;
        std::ptr::copy_nonoverlapping(params as *const QueryParamsSlot, dst, 1);
    }

    // 3. Allocate output Metal buffer (22560 bytes), zero-initialized
    let output_size = std::mem::size_of::<OutputBuffer>();
    let output_buffer = device
        .newBufferWithLength_options(output_size, options)
        .ok_or_else(|| "Failed to allocate output Metal buffer".to_string())?;

    unsafe {
        let dst = output_buffer.contents().as_ptr() as *mut u8;
        std::ptr::write_bytes(dst, 0, output_size);
    }

    // 3b. Initialize MIN/MAX sentinel values (same as execute_fused_oneshot)
    unsafe {
        let out_ptr = output_buffer.contents().as_ptr() as *mut OutputBuffer;
        let out = &mut *out_ptr;
        for a in 0..params.agg_count as usize {
            let agg_func = params.aggs[a].agg_func;
            let is_min = agg_func == 3; // AGG_FUNC_MIN
            let is_max = agg_func == 4; // AGG_FUNC_MAX
            if is_min || is_max {
                for g in 0..super::types::MAX_GROUPS {
                    if is_min {
                        out.agg_results[g][a].value_int = i64::MAX;
                        out.agg_results[g][a].value_float = f32::MAX;
                    } else {
                        out.agg_results[g][a].value_int = i64::MIN;
                        out.agg_results[g][a].value_float = f32::MIN;
                    }
                }
            }
        }
    }

    // 4. Create command buffer
    let cmd_buf = command_queue
        .commandBuffer()
        .ok_or_else(|| "Failed to create command buffer".to_string())?;

    // 5. Create compute command encoder and configure dispatch
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| "Failed to create compute command encoder".to_string())?;

        encoder.setComputePipelineState(pso);

        // Set buffers matching JIT kernel signature (same as AOT):
        //   buffer(0) = QueryParamsSlot
        //   buffer(1) = column data
        //   buffer(2) = ColumnMeta array
        //   buffer(3) = OutputBuffer
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*params_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*resident_table.data_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*resident_table.column_meta_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*output_buffer), 0, 3);
        }

        // Calculate dispatch dimensions
        let row_count = params.row_count as usize;
        let threads_per_threadgroup = 256usize;
        let threadgroup_count = if row_count == 0 {
            1
        } else {
            row_count.div_ceil(threads_per_threadgroup)
        };

        let grid_size = MTLSize {
            width: threadgroup_count,
            height: 1,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: threads_per_threadgroup,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
        encoder.endEncoding();
    }

    // 6. Commit and wait (one-shot blocking path)
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    // 7. Read back OutputBuffer from GPU memory
    let result = unsafe {
        let src = output_buffer.contents().as_ptr() as *const OutputBuffer;
        std::ptr::read(src)
    };

    Ok(result)
}

/// Engine state for the re-dispatch chain.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineState {
    /// Re-dispatch chain is actively running.
    Active = 0,
    /// Chain stopped due to idle timeout (no queries for 500ms).
    Idle = 1,
    /// Permanently stopped (shutdown requested).
    Shutdown = 2,
}

impl EngineState {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => EngineState::Active,
            1 => EngineState::Idle,
            2 => EngineState::Shutdown,
            _ => EngineState::Shutdown,
        }
    }
}

/// Idle timeout in milliseconds. If no new queries arrive within this window,
/// the completion handler stops re-dispatching and the chain goes idle.
const IDLE_TIMEOUT_MS: u64 = 500;

/// Shared state for the re-dispatch chain, accessible from Metal completion handlers.
///
/// All fields are `Send + Sync` safe via `Arc<Atomic*>` wrappers. The completion
/// handler closure captures a clone of this struct.
struct RedispatchSharedState {
    /// Current engine state (Active/Idle/Shutdown).
    state: Arc<AtomicU8>,
    /// Number of re-dispatches completed so far.
    redispatch_count: Arc<AtomicU64>,
    /// Timestamp (millis since epoch) of the last query submission.
    last_query_time_ms: Arc<AtomicU64>,
    /// Number of GPU errors encountered.
    error_count: Arc<AtomicU64>,
    /// Metal command queue for creating new command buffers.
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Compiled pipeline state object for the fused kernel.
    pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// GPU-resident query params buffer (StorageModeShared).
    params_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// GPU-resident column data buffer.
    data_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// GPU-resident column metadata buffer.
    column_meta_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// GPU-resident output buffer.
    output_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Number of rows in the table (for threadgroup calculation).
    row_count: u32,
    /// MTLSharedEvent for idle/wake signaling. Monotonically increasing signaled value
    /// tracks state transitions: even values = idle, odd values = active/wake.
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Monotonically increasing event counter for shared_event signaling.
    event_counter: Arc<AtomicU64>,
}

// SAFETY: All fields are either Arc<Atomic*> (Send+Sync) or Retained<ProtocolObject<dyn MTL*>>
// which are reference-counted Objective-C objects that are thread-safe for retain/release.
// Metal command queues and buffers are explicitly documented as safe to use from any thread.
unsafe impl Send for RedispatchSharedState {}
unsafe impl Sync for RedispatchSharedState {}

/// Dispatch a single time-slice of the fused kernel and register a completion handler
/// that re-dispatches the next slice (unless idle or shutdown).
///
/// This is the core of the re-dispatch chain. Each call:
/// 1. Creates a command buffer
/// 2. Encodes the fused kernel dispatch
/// 3. Registers a completion handler that calls `dispatch_slice` again
/// 4. Pre-enqueues the next command buffer via `enqueue()` to hide the gap
/// 5. Commits the current command buffer
fn dispatch_slice(shared: Arc<RedispatchSharedState>) {
    // Check if shutdown was requested
    let state = EngineState::from_u8(shared.state.load(Ordering::Acquire));
    if state == EngineState::Shutdown {
        return;
    }

    // 1. Create command buffer
    let cmd_buf = match shared.command_queue.commandBuffer() {
        Some(cb) => cb,
        None => {
            shared.error_count.fetch_add(1, Ordering::Relaxed);
            return;
        }
    };

    // 2. Encode compute pass
    {
        let encoder = match cmd_buf.computeCommandEncoder() {
            Some(enc) => enc,
            None => {
                shared.error_count.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        encoder.setComputePipelineState(&shared.pso);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*shared.params_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*shared.data_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*shared.column_meta_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*shared.output_buffer), 0, 3);
        }

        let threads_per_tg = 256usize;
        let row_count = shared.row_count as usize;
        let tg_count = if row_count == 0 {
            1
        } else {
            row_count.div_ceil(threads_per_tg)
        };

        let grid = MTLSize {
            width: tg_count,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();
    }

    // 3. Register completion handler for re-dispatch
    let shared_for_handler = shared.clone();
    let handler =
        RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            // Increment re-dispatch count
            shared_for_handler
                .redispatch_count
                .fetch_add(1, Ordering::Relaxed);

            // Check for shutdown
            let current_state =
                EngineState::from_u8(shared_for_handler.state.load(Ordering::Acquire));
            if current_state == EngineState::Shutdown {
                return;
            }

            // Check idle timeout: if no queries for 500ms, go idle
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let last_query_ms = shared_for_handler.last_query_time_ms.load(Ordering::Acquire);

            if last_query_ms > 0 && now_ms.saturating_sub(last_query_ms) > IDLE_TIMEOUT_MS {
                // No queries for > 500ms -> go idle, don't re-dispatch
                shared_for_handler
                    .state
                    .store(EngineState::Idle as u8, Ordering::Release);
                // Signal shared event with even value to indicate idle transition
                let val = shared_for_handler.event_counter.fetch_add(1, Ordering::AcqRel) + 1;
                shared_for_handler.shared_event.setSignaledValue(val);
                return;
            }

            // Re-dispatch next slice
            dispatch_slice(shared_for_handler.clone());
        });

    unsafe {
        cmd_buf.addCompletedHandler(RcBlock::as_ptr(&handler));
    }

    // 4. Enqueue to pre-queue the command buffer (hides inter-buffer gap)
    cmd_buf.enqueue();

    // 5. Commit to start GPU execution
    cmd_buf.commit();
}

/// Re-dispatch chain for continuous GPU kernel execution.
///
/// Maintains a chain of Metal command buffers where each completion handler
/// immediately dispatches the next. The chain runs continuously while queries
/// are being submitted, and idles after 500ms of inactivity.
///
/// # Architecture
/// ```text
/// Time ─────────────────────────────────────────────>
/// GPU:  [kernel_0 ~~~~] [kernel_1 ~~~~] [kernel_2 ...]
///                        ^               ^
/// CPU:  commit(cb_0)    complete(cb_0)  complete(cb_1)
///                       -> commit(cb_1) -> commit(cb_2)
///                       enqueue(cb_2)   enqueue(cb_3)
/// ```
pub struct RedispatchChain {
    shared: Arc<RedispatchSharedState>,
}

impl RedispatchChain {
    /// Create and start a new re-dispatch chain.
    ///
    /// The chain immediately begins dispatching time-slice kernels. The caller
    /// must call `record_query()` to keep the chain alive; otherwise it will
    /// idle after 500ms.
    ///
    /// # Arguments
    /// * `device` - Metal device for creating the shared event
    /// * `command_queue` - Metal command queue (should be dedicated/separate from other work)
    /// * `pso` - Compiled fused kernel pipeline state
    /// * `params_buffer` - Query params buffer (StorageModeShared, 512 bytes)
    /// * `data_buffer` - Column data buffer
    /// * `column_meta_buffer` - Column metadata buffer
    /// * `output_buffer` - Output buffer (StorageModeShared, 22560 bytes)
    /// * `row_count` - Number of rows in the table
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        device: &ProtocolObject<dyn MTLDevice>,
        command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        params_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        data_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        column_meta_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        output_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        row_count: u32,
    ) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Create MTLSharedEvent for idle/wake signaling
        let shared_event = device
            .newSharedEvent()
            .expect("Failed to create MTLSharedEvent");
        // Initial signaled value = 1 (odd = active)
        shared_event.setSignaledValue(1);

        let shared = Arc::new(RedispatchSharedState {
            state: Arc::new(AtomicU8::new(EngineState::Active as u8)),
            redispatch_count: Arc::new(AtomicU64::new(0)),
            last_query_time_ms: Arc::new(AtomicU64::new(now_ms)),
            error_count: Arc::new(AtomicU64::new(0)),
            command_queue,
            pso,
            params_buffer,
            data_buffer,
            column_meta_buffer,
            output_buffer,
            row_count,
            shared_event,
            event_counter: Arc::new(AtomicU64::new(1)),
        });

        // Start the chain
        dispatch_slice(shared.clone());

        RedispatchChain { shared }
    }

    /// Record that a query was submitted (resets idle timer).
    pub fn record_query(&self) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.shared
            .last_query_time_ms
            .store(now_ms, Ordering::Release);
    }

    /// Get the current engine state.
    pub fn state(&self) -> EngineState {
        EngineState::from_u8(self.shared.state.load(Ordering::Acquire))
    }

    /// Get the number of re-dispatches completed so far.
    pub fn redispatch_count(&self) -> u64 {
        self.shared.redispatch_count.load(Ordering::Relaxed)
    }

    /// Get the number of GPU errors encountered.
    pub fn error_count(&self) -> u64 {
        self.shared.error_count.load(Ordering::Relaxed)
    }

    /// Request shutdown. The completion handler will not re-dispatch after the
    /// current in-flight command buffer completes.
    pub fn shutdown(&self) {
        self.shared
            .state
            .store(EngineState::Shutdown as u8, Ordering::Release);
    }

    /// Wake the chain from idle state by re-starting dispatch.
    /// Signals the MTLSharedEvent with an odd value (active) and restarts
    /// the dispatch chain. No-op if already active or shutdown.
    pub fn wake(&self) {
        let prev = self.shared.state.compare_exchange(
            EngineState::Idle as u8,
            EngineState::Active as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        if prev.is_ok() {
            // Signal shared event with odd value to indicate wake/active transition
            let val = self.shared.event_counter.fetch_add(1, Ordering::AcqRel) + 1;
            self.shared.shared_event.setSignaledValue(val);
            // Record a query to prevent immediate re-idle
            self.record_query();
            dispatch_slice(self.shared.clone());
        }
    }

    /// Get the current signaled value of the MTLSharedEvent.
    /// Even values indicate idle transitions, odd values indicate active/wake.
    pub fn shared_event_value(&self) -> u64 {
        self.shared.shared_event.signaledValue()
    }
}

impl Drop for RedispatchChain {
    fn drop(&mut self) {
        self.shutdown();
        // Give the in-flight command buffer a moment to complete.
        // The completion handler will see Shutdown and not re-dispatch.
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

// ---------------------------------------------------------------------------
// AutonomousExecutor: unified struct combining all autonomous components
// ---------------------------------------------------------------------------

/// Statistics for the autonomous executor.
#[derive(Debug, Clone, Default)]
pub struct AutonomousStats {
    pub total_queries: u64,
    pub jit_cache_hits: u64,
    pub jit_cache_misses: u64,
    pub avg_latency_us: f64,
}

/// Unified autonomous query executor combining all components:
/// JIT compiler, work queue, re-dispatch chain, resident tables.
///
/// Lifecycle: `new()` -> `load_table()` -> `submit_query()` -> `poll_ready()`
/// -> `read_result()` -> `shutdown()`.
pub struct AutonomousExecutor {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    work_queue: WorkQueue,
    /// Output buffer (22560B, StorageModeShared) for unified-memory readback.
    output_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Params buffer (512B, StorageModeShared) used by the re-dispatch chain.
    params_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    jit_compiler: JitCompiler,
    resident_tables: HashMap<String, ResidentTable>,
    redispatch_chain: Option<RedispatchChain>,
    state: Arc<AtomicU8>,
    stats: AutonomousStats,
    /// Last query sequence_id for tracking which result we're waiting for.
    last_sequence_id: u32,
}

// SAFETY: All fields are either Arc<Atomic*> (Send+Sync), HashMap (Send), primitive types,
// or Retained<ProtocolObject<dyn MTL*>> which are reference-counted Objective-C objects that
// are thread-safe for retain/release. Metal command queues and buffers are explicitly
// documented as safe to use from any thread. This is needed for background warm-up thread.
unsafe impl Send for AutonomousExecutor {}

impl AutonomousExecutor {
    /// Create a new autonomous executor with a separate command queue.
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        let command_queue = device
            .newCommandQueue()
            .expect("Failed to create autonomous command queue");

        let work_queue = WorkQueue::new(&device);

        let options = MTLResourceOptions::StorageModeShared;

        // Allocate output buffer (22560 bytes)
        let output_size = std::mem::size_of::<OutputBuffer>();
        let output_buffer = device
            .newBufferWithLength_options(output_size, options)
            .expect("Failed to allocate output Metal buffer");
        // Zero-init
        unsafe {
            let dst = output_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(dst, 0, output_size);
        }

        // Allocate params buffer (512 bytes)
        let params_size = std::mem::size_of::<QueryParamsSlot>();
        let params_buffer = device
            .newBufferWithLength_options(params_size, options)
            .expect("Failed to allocate params Metal buffer");
        unsafe {
            let dst = params_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(dst, 0, params_size);
        }

        let jit_compiler = JitCompiler::new(device.clone());

        Self {
            device,
            command_queue,
            work_queue,
            output_buffer,
            params_buffer,
            jit_compiler,
            resident_tables: HashMap::new(),
            redispatch_chain: None,
            state: Arc::new(AtomicU8::new(EngineState::Idle as u8)),
            stats: AutonomousStats::default(),
            last_sequence_id: 0,
        }
    }

    /// Load a table into GPU-resident memory.
    pub fn load_table(
        &mut self,
        name: &str,
        schema: &RuntimeSchema,
        batch: &ColumnarBatch,
    ) -> Result<(), String> {
        let resident_table =
            BinaryColumnarLoader::load_table(&self.device, name, schema, batch, None)?;
        self.resident_tables.insert(name.to_string(), resident_table);
        Ok(())
    }

    /// Get a reference to the underlying Metal device.
    ///
    /// Used by the warm-up thread to allocate `ColumnarBatch` buffers on the same
    /// device before loading them into the autonomous executor.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Submit a query for autonomous execution.
    ///
    /// 1. JIT compile (cache lookup or compile)
    /// 2. Build QueryParamsSlot from PhysicalPlan
    /// 3. Write params to the shared params buffer
    /// 4. Zero-init output buffer for fresh results
    /// 5. Initialize MIN/MAX sentinels
    /// 6. Dispatch a single command buffer (non-blocking, no waitUntilCompleted)
    /// 7. Return query sequence_id
    ///
    /// The dispatch is a single command buffer committed without waitUntilCompleted.
    /// The GPU kernel sets output_buffer.ready_flag = 1 when done. The caller polls
    /// via `poll_ready()` and reads via `read_result()`.
    pub fn submit_query(
        &mut self,
        plan: &PhysicalPlan,
        schema: &[ColumnInfo],
        table_name: &str,
    ) -> Result<u32, String> {
        let table = self
            .resident_tables
            .get(table_name)
            .ok_or_else(|| format!("Table '{}' not loaded", table_name))?;

        // 1. JIT compile
        let jit_cache_len_before = self.jit_compiler.cache_len();
        let compiled = self.jit_compiler.compile(plan, schema)?;
        let pso = compiled.pso.clone();
        let jit_cache_len_after = self.jit_compiler.cache_len();

        if jit_cache_len_after > jit_cache_len_before {
            self.stats.jit_cache_misses += 1;
        } else {
            self.stats.jit_cache_hits += 1;
        }

        // 2. Build QueryParamsSlot from PhysicalPlan
        let params = build_query_params(plan, schema, table.row_count)?;

        // 3. Write params to the work queue (for tracking) and to the shared params buffer
        self.work_queue.write_params(&params);
        let sequence_id = self.work_queue.read_latest_sequence_id();
        self.last_sequence_id = sequence_id;

        // Write params with correct sequence_id to the params buffer for GPU
        unsafe {
            let dst = self.params_buffer.contents().as_ptr() as *mut QueryParamsSlot;
            let mut params_with_seq = params;
            params_with_seq.sequence_id = sequence_id;
            std::ptr::copy_nonoverlapping(&params_with_seq as *const QueryParamsSlot, dst, 1);
        }

        // 4. Zero-init output buffer
        let output_size = std::mem::size_of::<OutputBuffer>();
        unsafe {
            let dst = self.output_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(dst, 0, output_size);
        }

        // 5. Initialize MIN/MAX sentinel values
        unsafe {
            let out_ptr = self.output_buffer.contents().as_ptr() as *mut OutputBuffer;
            let out = &mut *out_ptr;
            for a in 0..params.agg_count as usize {
                let agg_func = params.aggs[a].agg_func;
                let is_min = agg_func == 3; // AGG_FUNC_MIN
                let is_max = agg_func == 4; // AGG_FUNC_MAX
                if is_min || is_max {
                    for g in 0..super::types::MAX_GROUPS {
                        if is_min {
                            out.agg_results[g][a].value_int = i64::MAX;
                            out.agg_results[g][a].value_float = f32::MAX;
                        } else {
                            out.agg_results[g][a].value_int = i64::MIN;
                            out.agg_results[g][a].value_float = f32::MIN;
                        }
                    }
                }
            }
        }

        // 6. Dispatch single command buffer (non-blocking)
        // This is the autonomous path: no waitUntilCompleted, no per-query command buffer
        // creation from the CPU hot path. The GPU kernel sets ready_flag when done.
        let cmd_buf = self
            .command_queue
            .commandBuffer()
            .ok_or_else(|| "Failed to create command buffer".to_string())?;

        {
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or_else(|| "Failed to create compute command encoder".to_string())?;

            encoder.setComputePipelineState(&pso);

            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&*self.params_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&*table.data_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&*table.column_meta_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&*self.output_buffer), 0, 3);
            }

            let row_count = table.row_count as usize;
            let threads_per_tg = 256usize;
            let tg_count = if row_count == 0 {
                1
            } else {
                row_count.div_ceil(threads_per_tg)
            };

            let grid = MTLSize {
                width: tg_count,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            };

            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
        }

        // Non-blocking commit: GPU executes asynchronously, sets ready_flag when done
        cmd_buf.commit();

        self.state
            .store(EngineState::Active as u8, Ordering::Release);
        self.stats.total_queries += 1;

        Ok(sequence_id)
    }

    /// Non-blocking poll: check if the output buffer has a result ready.
    ///
    /// Reads the ready_flag from unified memory via pointer (no GPU readback needed).
    pub fn poll_ready(&self) -> bool {
        unsafe {
            std::sync::atomic::fence(Ordering::Acquire);
            let out_ptr = self.output_buffer.contents().as_ptr() as *const OutputBuffer;
            let ready = std::ptr::read_volatile(&(*out_ptr).ready_flag);
            ready == 1
        }
    }

    /// Read the result from the output buffer (unified memory, no readback).
    ///
    /// Resets the ready_flag to 0 after reading.
    pub fn read_result(&self) -> OutputBuffer {
        unsafe {
            std::sync::atomic::fence(Ordering::Acquire);
            let src = self.output_buffer.contents().as_ptr() as *const OutputBuffer;
            let r = std::ptr::read(src);
            // Reset ready_flag
            let flag_ptr = self.output_buffer.contents().as_ptr() as *mut u32;
            std::ptr::write_volatile(flag_ptr, 0);
            r
        }
    }

    /// Shutdown the autonomous executor: stops re-dispatch chain, sets state to Shutdown.
    pub fn shutdown(&mut self) {
        self.state
            .store(EngineState::Shutdown as u8, Ordering::Release);
        if let Some(chain) = self.redispatch_chain.take() {
            chain.shutdown();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    /// Get the current engine state.
    pub fn engine_state(&self) -> EngineState {
        EngineState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Get a clone of the current stats.
    pub fn stats(&self) -> AutonomousStats {
        self.stats.clone()
    }

    /// Get the last submitted query sequence_id.
    pub fn last_sequence_id(&self) -> u32 {
        self.last_sequence_id
    }
}

// ---------------------------------------------------------------------------
// Autonomous compatibility check
// ---------------------------------------------------------------------------

/// Maximum number of GROUP BY distinct values supported by the autonomous engine.
/// The fused kernel uses a 64-slot hash table in threadgroup memory.
pub const AUTONOMOUS_MAX_GROUP_CARDINALITY: usize = 64;

/// Result of checking whether a physical plan can run on the autonomous engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatibilityResult {
    /// Plan is fully supported by the autonomous engine.
    Autonomous,
    /// Plan requires fallback to the standard executor, with a reason string.
    Fallback(String),
}

/// Check whether a `PhysicalPlan` can be executed by the autonomous fused kernel.
///
/// Supported patterns:
/// - `GpuScan` (simple table scan)
/// - `GpuFilter` / `GpuCompoundFilter` (single or compound filters)
/// - `GpuAggregate` with <= 1 GROUP BY column (cardinality <= 64)
///
/// Unsupported (returns `Fallback` with reason):
/// - `GpuSort` (ORDER BY)
/// - `GpuLimit` (LIMIT -- requires sort first)
/// - `GpuAggregate` with > 1 GROUP BY column
///
/// This function inspects the plan tree recursively.
pub fn check_autonomous_compatibility(plan: &PhysicalPlan) -> CompatibilityResult {
    match plan {
        PhysicalPlan::GpuScan { .. } => CompatibilityResult::Autonomous,

        PhysicalPlan::GpuFilter { input, .. } => {
            check_autonomous_compatibility(input)
        }

        PhysicalPlan::GpuCompoundFilter { left, right, .. } => {
            // Both sides must be compatible (they should be filters over the same scan)
            let left_compat = check_autonomous_compatibility(left);
            if let CompatibilityResult::Fallback(reason) = left_compat {
                return CompatibilityResult::Fallback(reason);
            }
            check_autonomous_compatibility(right)
        }

        PhysicalPlan::GpuAggregate {
            group_by, input, ..
        } => {
            // Check GROUP BY column count: autonomous supports 0 or 1
            if group_by.len() > 1 {
                return CompatibilityResult::Fallback(format!(
                    "multi-column GROUP BY ({} columns) requires standard path",
                    group_by.len()
                ));
            }
            // Check input subtree compatibility
            check_autonomous_compatibility(input)
        }

        PhysicalPlan::GpuSort { .. } => {
            CompatibilityResult::Fallback("ORDER BY requires standard path".to_string())
        }

        PhysicalPlan::GpuLimit { input, .. } => {
            // If the input contains a sort, that's the blocker.
            // Otherwise LIMIT alone could theoretically work, but the autonomous
            // engine doesn't implement row-level truncation, so fall back.
            let inner = check_autonomous_compatibility(input);
            match inner {
                CompatibilityResult::Fallback(reason) => CompatibilityResult::Fallback(reason),
                CompatibilityResult::Autonomous => {
                    CompatibilityResult::Fallback("LIMIT requires standard path".to_string())
                }
            }
        }
    }
}

/// Extract filter predicates, aggregates, and group_by from a PhysicalPlan tree
/// and build a QueryParamsSlot.
fn build_query_params(
    plan: &PhysicalPlan,
    schema: &[ColumnInfo],
    row_count: u32,
) -> Result<QueryParamsSlot, String> {
    let mut params = QueryParamsSlot {
        row_count,
        ..Default::default()
    };

    // Collect filters, aggs, group_by from plan
    let mut filters: Vec<(String, CompareOp, Value)> = Vec::new();
    let mut aggregates: Vec<(AggFunc, String)> = Vec::new();
    let mut group_by: Vec<String> = Vec::new();

    fn collect_filters(node: &PhysicalPlan, out: &mut Vec<(String, CompareOp, Value)>) {
        match node {
            PhysicalPlan::GpuFilter {
                compare_op,
                column,
                value,
                input,
            } => {
                out.push((column.clone(), *compare_op, value.clone()));
                collect_filters(input, out);
            }
            PhysicalPlan::GpuCompoundFilter { left, right, .. } => {
                collect_filters(left, out);
                collect_filters(right, out);
            }
            _ => {}
        }
    }

    match plan {
        PhysicalPlan::GpuAggregate {
            functions,
            group_by: gb,
            input,
        } => {
            aggregates = functions.clone();
            group_by = gb.clone();
            collect_filters(input, &mut filters);
        }
        _ => {
            collect_filters(plan, &mut filters);
        }
    }

    // Populate filters
    params.filter_count = filters.len().min(super::types::MAX_FILTERS) as u32;
    for (i, (col_name, op, value)) in filters.iter().enumerate() {
        if i >= super::types::MAX_FILTERS {
            break;
        }
        let col_idx = schema
            .iter()
            .position(|c| c.name == *col_name)
            .unwrap_or(0);
        let col_type = match schema.get(col_idx) {
            Some(ci) => match ci.data_type {
                crate::storage::schema::DataType::Int64
                | crate::storage::schema::DataType::Date => 0, // COLUMN_TYPE_INT64
                crate::storage::schema::DataType::Float64 => 1, // COLUMN_TYPE_FLOAT32
                crate::storage::schema::DataType::Varchar => 2, // COLUMN_TYPE_DICT_U32
                crate::storage::schema::DataType::Bool => 0,
            },
            None => 0,
        };

        params.filters[i] = FilterSpec {
            column_idx: col_idx as u32,
            compare_op: *op as u32,
            column_type: col_type,
            _pad0: 0,
            value_int: match value {
                Value::Int(v) => *v,
                Value::Float(v) => *v as i64,
                _ => 0,
            },
            value_float_bits: match value {
                Value::Float(v) => (*v as f32).to_bits(),
                Value::Int(v) => (*v as f32).to_bits(),
                _ => 0,
            },
            _pad1: 0,
            has_null_check: 0,
            _pad2: [0; 3],
        };
    }

    // Populate aggregates
    params.agg_count = aggregates.len().min(super::types::MAX_AGGS) as u32;
    for (i, (func, col_name)) in aggregates.iter().enumerate() {
        if i >= super::types::MAX_AGGS {
            break;
        }
        let col_idx = schema
            .iter()
            .position(|c| c.name == *col_name)
            .unwrap_or(0);
        let col_type = match schema.get(col_idx) {
            Some(ci) => match ci.data_type {
                crate::storage::schema::DataType::Int64
                | crate::storage::schema::DataType::Date => 0,
                crate::storage::schema::DataType::Float64 => 1,
                crate::storage::schema::DataType::Varchar => 2,
                crate::storage::schema::DataType::Bool => 0,
            },
            None => 0,
        };

        params.aggs[i] = AggSpec {
            agg_func: func.to_gpu_code(),
            column_idx: col_idx as u32,
            column_type: col_type,
            _pad0: 0,
        };
    }

    // Populate group_by
    if !group_by.is_empty() {
        params.has_group_by = 1;
        let gb_idx = schema
            .iter()
            .position(|c| c.name == group_by[0])
            .unwrap_or(0);
        params.group_by_col = gb_idx as u32;
    }

    // Set query_hash from JIT plan structure hash
    params.query_hash = super::jit::plan_structure_hash(plan);

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::autonomous::loader::BinaryColumnarLoader;
    use crate::gpu::autonomous::types::AggSpec;
    use crate::gpu::device::GpuDevice;
    use crate::storage::columnar::ColumnarBatch;
    use crate::storage::schema::{ColumnDef, DataType, RuntimeSchema};

    #[test]
    fn test_fused_pso_compilation() {
        // Create real Metal device and library
        let gpu = GpuDevice::new();

        let mut cache = FusedPsoCache::new(
            gpu.device.clone(),
            gpu.library.clone(),
        );

        // Compile PSO for headline query: 2 filters, 2 aggs, group_by=true
        let pso = cache.get_or_compile(2, 2, true);
        assert!(pso.is_ok(), "PSO compilation failed: {:?}", pso.err());

        // Cache should have one entry
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_fused_pso_cache_hit() {
        let gpu = GpuDevice::new();
        let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

        // Compile same PSO twice -- should hit cache on second call
        let _pso1 = cache.get_or_compile(2, 2, true).expect("first compile failed");
        let _pso2 = cache.get_or_compile(2, 2, true).expect("second compile failed");

        // Only one entry in cache (cache hit)
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_fused_pso_different_constants() {
        let gpu = GpuDevice::new();
        let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

        // Compile different constant combinations
        cache.get_or_compile(0, 1, false).expect("count-only compile failed");
        cache.get_or_compile(1, 1, false).expect("single-filter compile failed");
        cache.get_or_compile(2, 2, true).expect("headline compile failed");

        // Three distinct entries
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_fused_pso_no_filters_no_groupby() {
        let gpu = GpuDevice::new();
        let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

        // Simplest case: COUNT(*) with no filters, no GROUP BY
        let pso = cache.get_or_compile(0, 1, false);
        assert!(pso.is_ok(), "Simplest PSO failed: {:?}", pso.err());
    }

    #[test]
    fn test_fused_pso_max_filters_and_aggs() {
        let gpu = GpuDevice::new();
        let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

        // Max configuration: 4 filters, 5 aggs, group_by=true
        let pso = cache.get_or_compile(4, 5, true);
        assert!(pso.is_ok(), "Max config PSO failed: {:?}", pso.err());
    }

    /// Helper: create a RuntimeSchema with the given column definitions.
    fn make_schema(cols: &[(&str, DataType)]) -> RuntimeSchema {
        RuntimeSchema::new(
            cols.iter()
                .map(|(name, dt)| ColumnDef {
                    name: name.to_string(),
                    data_type: *dt,
                    nullable: false,
                })
                .collect(),
        )
    }

    /// Helper: create a ColumnarBatch with deterministic INT64 data.
    /// INT64 values: (i * 7 + 13) % 1000
    fn make_test_batch(
        device: &ProtocolObject<dyn MTLDevice>,
        schema: &RuntimeSchema,
        row_count: usize,
    ) -> ColumnarBatch {
        let mut batch = ColumnarBatch::allocate(device, schema, row_count);
        batch.row_count = row_count;

        let mut int_local_idx = 0usize;
        let mut float_local_idx = 0usize;

        for col in &schema.columns {
            match col.data_type {
                DataType::Int64 | DataType::Date => {
                    unsafe {
                        let ptr = batch.int_buffer.contents().as_ptr() as *mut i64;
                        let offset = int_local_idx * batch.max_rows;
                        for i in 0..row_count {
                            *ptr.add(offset + i) = ((i * 7 + 13) % 1000) as i64;
                        }
                    }
                    int_local_idx += 1;
                }
                DataType::Float64 => {
                    unsafe {
                        let ptr = batch.float_buffer.contents().as_ptr() as *mut f32;
                        let offset = float_local_idx * batch.max_rows;
                        for i in 0..row_count {
                            *ptr.add(offset + i) = ((i * 7 + 13) % 1000) as f32;
                        }
                    }
                    float_local_idx += 1;
                }
                _ => {}
            }
        }

        batch
    }

    #[test]
    fn test_fused_oneshot_count() {
        // Test: SELECT COUNT(*) FROM sales (1000 rows, no filter, no GROUP BY)
        let gpu = GpuDevice::new();

        // 1. Prepare test data: single INT64 column, 1000 rows
        let schema = make_schema(&[("amount", DataType::Int64)]);
        let batch = make_test_batch(&gpu.device, &schema, 1000);

        // 2. Load into GPU-resident table
        let resident_table =
            BinaryColumnarLoader::load_table(&gpu.device, "sales", &schema, &batch, None)
                .expect("Failed to load test table");

        assert_eq!(resident_table.row_count, 1000);

        // 3. Compile PSO for COUNT(*): 0 filters, 1 agg, no group_by
        let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());
        let pso = cache
            .get_or_compile(0, 1, false)
            .expect("PSO compilation failed");

        // 4. Build QueryParamsSlot for COUNT(*)
        let mut params = QueryParamsSlot::default();
        params.sequence_id = 1;
        params.filter_count = 0;
        params.agg_count = 1;
        params.aggs[0] = AggSpec {
            agg_func: 0,    // COUNT
            column_idx: 0,  // column 0 (amount) -- not used for COUNT but must be valid
            column_type: 0, // INT64
            _pad0: 0,
        };
        params.has_group_by = 0;
        params.group_by_col = 0;
        params.row_count = 1000;

        // 5. Execute one-shot dispatch
        let result = execute_fused_oneshot(
            &gpu.device,
            &gpu.command_queue,
            pso,
            &params,
            &resident_table,
        )
        .expect("execute_fused_oneshot failed");

        // 6. Verify results
        // result_row_count should be 1 (scalar result, no GROUP BY)
        assert_eq!(
            result.result_row_count, 1,
            "Expected 1 result row (scalar), got {}",
            result.result_row_count
        );

        // COUNT(*) result should be 1000 (all rows)
        assert_eq!(
            result.agg_results[0][0].value_int, 1000,
            "Expected COUNT(*) = 1000, got {}",
            result.agg_results[0][0].value_int
        );

        // ready_flag should be set
        assert_eq!(
            result.ready_flag, 1,
            "Expected ready_flag = 1, got {}",
            result.ready_flag
        );

        // sequence_id should be echoed back
        assert_eq!(
            result.sequence_id, 1,
            "Expected sequence_id = 1, got {}",
            result.sequence_id
        );

        // error_code should be 0 (success)
        assert_eq!(
            result.error_code, 0,
            "Expected error_code = 0, got {}",
            result.error_code
        );
    }

    /// Helper: create Metal buffers needed for the re-dispatch chain (params, output).
    /// Sets up a COUNT(*) query on the given resident table.
    fn make_redispatch_buffers(
        device: &ProtocolObject<dyn MTLDevice>,
        resident_table: &ResidentTable,
    ) -> (
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
    ) {
        let options = MTLResourceOptions::StorageModeShared;

        // Params buffer: COUNT(*) query
        let mut params = QueryParamsSlot::default();
        params.sequence_id = 1;
        params.filter_count = 0;
        params.agg_count = 1;
        params.aggs[0] = AggSpec {
            agg_func: 0,    // COUNT
            column_idx: 0,
            column_type: 0, // INT64
            _pad0: 0,
        };
        params.has_group_by = 0;
        params.group_by_col = 0;
        params.row_count = resident_table.row_count as u32;

        let params_size = std::mem::size_of::<QueryParamsSlot>();
        let params_buffer = device
            .newBufferWithLength_options(params_size, options)
            .expect("Failed to allocate params buffer");
        unsafe {
            let dst = params_buffer.contents().as_ptr() as *mut QueryParamsSlot;
            std::ptr::copy_nonoverlapping(&params as *const QueryParamsSlot, dst, 1);
        }

        // Output buffer: zero-initialized
        let output_size = std::mem::size_of::<OutputBuffer>();
        let output_buffer = device
            .newBufferWithLength_options(output_size, options)
            .expect("Failed to allocate output buffer");
        unsafe {
            let dst = output_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(dst, 0, output_size);
        }

        (params_buffer, output_buffer)
    }

    #[test]
    fn test_redispatch_chain() {
        // Test: Re-dispatch chain runs for 10 seconds without GPU watchdog kill.
        // Then idle timeout kicks in after 500ms of no queries.
        let gpu = GpuDevice::new();

        // 1. Prepare test data: single INT64 column, 1000 rows
        let schema = make_schema(&[("amount", DataType::Int64)]);
        let batch = make_test_batch(&gpu.device, &schema, 1000);
        let resident_table =
            BinaryColumnarLoader::load_table(&gpu.device, "sales", &schema, &batch, None)
                .expect("Failed to load test table");

        // 2. Compile PSO for COUNT(*): 0 filters, 1 agg, no group_by
        let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());
        let _pso_ref = cache
            .get_or_compile(0, 1, false)
            .expect("PSO compilation failed");

        // We need a Retained PSO for the chain. Get it from the cache internals.
        let pso = cache.cache.remove(&(0, 1, false)).unwrap();

        // 3. Create a separate command queue for the chain
        let chain_queue = gpu
            .device
            .newCommandQueue()
            .expect("Failed to create chain command queue");

        // 4. Create buffers
        let (params_buffer, output_buffer) =
            make_redispatch_buffers(&gpu.device, &resident_table);

        // 5. Start the re-dispatch chain
        let chain = RedispatchChain::start(
            &gpu.device,
            chain_queue,
            pso,
            params_buffer,
            resident_table.data_buffer.clone(),
            resident_table.column_meta_buffer.clone(),
            output_buffer,
            resident_table.row_count as u32,
        );

        // 6. Chain should be Active
        assert_eq!(
            chain.state(),
            EngineState::Active,
            "Chain should start in Active state"
        );

        // 7. Run for 10 seconds, periodically recording queries to keep alive
        let start = std::time::Instant::now();
        let run_duration = std::time::Duration::from_secs(10);

        while start.elapsed() < run_duration {
            // Record a query every 100ms to keep chain alive
            chain.record_query();
            std::thread::sleep(std::time::Duration::from_millis(100));

            // Verify no GPU errors
            assert_eq!(
                chain.error_count(),
                0,
                "GPU errors detected after {:?}",
                start.elapsed()
            );

            // Verify chain is still active
            assert_eq!(
                chain.state(),
                EngineState::Active,
                "Chain went idle prematurely after {:?}",
                start.elapsed()
            );
        }

        // 8. Verify re-dispatch happened many times (at least once, but kernel is fast)
        let count = chain.redispatch_count();
        assert!(
            count > 0,
            "Expected re-dispatch_count > 0, got {}",
            count
        );

        // 9. Verify no GPU errors
        assert_eq!(
            chain.error_count(),
            0,
            "GPU errors after 10 seconds: {}",
            chain.error_count()
        );

        // 10. Test idle detection: stop recording queries and wait > 500ms
        // (don't call record_query anymore)
        std::thread::sleep(std::time::Duration::from_millis(800));

        // Chain should now be idle
        assert_eq!(
            chain.state(),
            EngineState::Idle,
            "Chain should be Idle after 800ms without queries (re-dispatches: {})",
            chain.redispatch_count()
        );

        // 11. Clean shutdown
        chain.shutdown();
        assert_eq!(chain.state(), EngineState::Shutdown);
    }

    #[test]
    fn test_idle_wake() {
        // Test: Start chain -> submit query -> idle timeout -> submit again -> wake -> correct result.
        // Proves the full idle/wake cycle with MTLSharedEvent signaling.
        //
        // Note: The persistent chain continuously re-executes the kernel, so output buffer
        // accumulates (device atomics add each dispatch). We verify:
        // 1. One-shot correctness (before chain) -> exact COUNT=1000
        // 2. Chain state transitions (Active -> Idle -> Active)
        // 3. MTLSharedEvent signaling (odd=active, even=idle)
        // 4. Chain restart after wake (re-dispatch count increases)
        // 5. One-shot correctness (after chain) -> exact COUNT=1000
        let gpu = GpuDevice::new();

        // 1. Prepare test data: single INT64 column, 1000 rows
        let schema = make_schema(&[("amount", DataType::Int64)]);
        let batch = make_test_batch(&gpu.device, &schema, 1000);
        let resident_table =
            BinaryColumnarLoader::load_table(&gpu.device, "sales", &schema, &batch, None)
                .expect("Failed to load test table");

        // 2. Compile PSO for COUNT(*): 0 filters, 1 agg, no group_by
        let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());
        let _pso_ref = cache
            .get_or_compile(0, 1, false)
            .expect("PSO compilation failed");

        // 3. Verify correctness via one-shot dispatch BEFORE the chain
        let params = {
            let mut p = QueryParamsSlot::default();
            p.sequence_id = 1;
            p.filter_count = 0;
            p.agg_count = 1;
            p.aggs[0] = AggSpec {
                agg_func: 0,    // COUNT
                column_idx: 0,
                column_type: 0, // INT64
                _pad0: 0,
            };
            p.has_group_by = 0;
            p.group_by_col = 0;
            p.row_count = 1000;
            p
        };

        let pso_ref = cache.get_or_compile(0, 1, false).unwrap();
        let result_before = execute_fused_oneshot(
            &gpu.device,
            &gpu.command_queue,
            pso_ref,
            &params,
            &resident_table,
        )
        .expect("One-shot dispatch before chain failed");
        assert_eq!(
            result_before.agg_results[0][0].value_int, 1000,
            "COUNT(*) before chain should be 1000, got {}",
            result_before.agg_results[0][0].value_int
        );

        // 4. Get a Retained PSO for the chain
        let pso = cache.cache.remove(&(0, 1, false)).unwrap();

        // 5. Create a separate command queue for the chain
        let chain_queue = gpu
            .device
            .newCommandQueue()
            .expect("Failed to create chain command queue");

        // 6. Create buffers with a COUNT(*) query
        let (params_buffer, output_buffer) =
            make_redispatch_buffers(&gpu.device, &resident_table);

        // 7. Start the re-dispatch chain
        let chain = RedispatchChain::start(
            &gpu.device,
            chain_queue,
            pso,
            params_buffer.clone(),
            resident_table.data_buffer.clone(),
            resident_table.column_meta_buffer.clone(),
            output_buffer.clone(),
            resident_table.row_count as u32,
        );

        // Verify initial state: Active, shared event value = 1 (odd = active)
        assert_eq!(chain.state(), EngineState::Active);
        let initial_event_val = chain.shared_event_value();
        assert_eq!(initial_event_val, 1, "Initial shared event value should be 1 (active)");

        // 8. Keep chain alive briefly with a query, let it process
        chain.record_query();
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Verify chain is still active and kernel has been dispatching
        assert_eq!(chain.state(), EngineState::Active);
        assert!(chain.redispatch_count() > 0, "Chain should have dispatched at least once");

        // 9. Let the chain go idle: stop recording queries and wait > 500ms
        std::thread::sleep(std::time::Duration::from_millis(800));

        // Verify chain is now Idle
        assert_eq!(
            chain.state(),
            EngineState::Idle,
            "Chain should be Idle after 800ms without queries"
        );

        // Verify shared event was signaled on idle (value increased, even = idle)
        let idle_event_val = chain.shared_event_value();
        assert!(
            idle_event_val > initial_event_val,
            "Shared event value should increase on idle: initial={}, idle={}",
            initial_event_val, idle_event_val
        );
        assert_eq!(
            idle_event_val % 2,
            0,
            "Idle event value should be even, got {}",
            idle_event_val
        );

        let redispatch_before_wake = chain.redispatch_count();

        // 10. Wake from idle
        chain.wake();

        // Verify shared event was signaled on wake (value increased, odd = active)
        let wake_event_val = chain.shared_event_value();
        assert!(
            wake_event_val > idle_event_val,
            "Shared event value should increase on wake: idle={}, wake={}",
            idle_event_val, wake_event_val
        );
        assert_eq!(
            wake_event_val % 2,
            1,
            "Wake event value should be odd, got {}",
            wake_event_val
        );

        // Verify state is Active again
        assert_eq!(
            chain.state(),
            EngineState::Active,
            "Chain should be Active after wake"
        );

        // 11. Let the chain process for a bit after waking
        chain.record_query();
        std::thread::sleep(std::time::Duration::from_millis(300));

        // Verify re-dispatch count increased (chain restarted successfully)
        let redispatch_after_wake = chain.redispatch_count();
        assert!(
            redispatch_after_wake > redispatch_before_wake,
            "Re-dispatch count should increase after wake: before={}, after={}",
            redispatch_before_wake, redispatch_after_wake
        );

        // 12. Verify no GPU errors throughout the idle/wake cycle
        assert_eq!(
            chain.error_count(),
            0,
            "No GPU errors expected, got {}",
            chain.error_count()
        );

        // 13. Clean shutdown
        chain.shutdown();
        assert_eq!(chain.state(), EngineState::Shutdown);

        // Give in-flight command buffers a moment to drain
        std::thread::sleep(std::time::Duration::from_millis(100));

        // 14. Verify correctness via one-shot dispatch AFTER the chain
        // This re-creates the PSO cache and does a fresh one-shot dispatch to prove
        // the device/data are still in a good state after the idle/wake cycle.
        let mut cache2 = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());
        let pso_ref2 = cache2.get_or_compile(0, 1, false).expect("PSO compile after chain");
        let result_after = execute_fused_oneshot(
            &gpu.device,
            &gpu.command_queue,
            pso_ref2,
            &params,
            &resident_table,
        )
        .expect("One-shot dispatch after chain failed");
        assert_eq!(
            result_after.agg_results[0][0].value_int, 1000,
            "COUNT(*) after idle/wake cycle should be 1000, got {}",
            result_after.agg_results[0][0].value_int
        );
    }

    #[test]
    fn test_full_lifecycle() {
        // Full lifecycle: new -> load_table -> submit_query -> poll_ready -> read_result -> shutdown
        //
        // This is the first test that uses the unified AutonomousExecutor struct,
        // combining JIT compiler, work queue, re-dispatch chain, and unified memory output.

        let gpu = GpuDevice::new();

        // 1. Create executor
        let mut executor = AutonomousExecutor::new(gpu.device.clone());
        assert_eq!(
            executor.engine_state(),
            EngineState::Idle,
            "New executor should start Idle"
        );

        // 2. Load table: single INT64 column, 1000 rows
        let schema = make_schema(&[("amount", DataType::Int64)]);
        let batch = make_test_batch(&gpu.device, &schema, 1000);

        executor
            .load_table("sales", &schema, &batch)
            .expect("load_table failed");

        // 3. Build a COUNT(*) query plan
        let plan = PhysicalPlan::GpuAggregate {
            functions: vec![(crate::sql::types::AggFunc::Count, "*".to_string())],
            group_by: vec![],
            input: Box::new(PhysicalPlan::GpuScan {
                table: "sales".to_string(),
                columns: vec!["amount".to_string()],
            }),
        };

        let col_schema = vec![ColumnInfo {
            name: "amount".to_string(),
            data_type: DataType::Int64,
        }];

        // 4. Submit query
        let seq_id = executor
            .submit_query(&plan, &col_schema, "sales")
            .expect("submit_query failed");

        assert!(seq_id > 0, "sequence_id should be > 0, got {}", seq_id);
        assert_eq!(
            executor.engine_state(),
            EngineState::Active,
            "After submit, executor should be Active"
        );
        assert_eq!(executor.stats().total_queries, 1);

        // 5. Poll until ready (with timeout)
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(10);
        while !executor.poll_ready() {
            if start.elapsed() > timeout {
                panic!("Timed out waiting for query result (10s)");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // 6. Read result
        let result = executor.read_result();

        // Verify COUNT(*) == 1000
        assert_eq!(
            result.agg_results[0][0].value_int, 1000,
            "COUNT(*) should be 1000, got {}",
            result.agg_results[0][0].value_int
        );
        assert_eq!(
            result.result_row_count, 1,
            "Scalar result should have 1 row, got {}",
            result.result_row_count
        );
        assert_eq!(
            result.ready_flag, 1,
            "ready_flag should be 1, got {}",
            result.ready_flag
        );
        assert_eq!(
            result.error_code, 0,
            "error_code should be 0, got {}",
            result.error_code
        );

        // 7. Shutdown
        executor.shutdown();
        assert_eq!(
            executor.engine_state(),
            EngineState::Shutdown,
            "After shutdown, state should be Shutdown"
        );
    }

    // -----------------------------------------------------------------------
    // Autonomous compatibility check tests
    // -----------------------------------------------------------------------

    /// Helper: build a simple scan plan.
    fn scan(table: &str) -> PhysicalPlan {
        PhysicalPlan::GpuScan {
            table: table.to_string(),
            columns: vec!["col".to_string()],
        }
    }

    #[test]
    fn test_compatibility_simple_scan() {
        let plan = scan("sales");
        assert_eq!(
            check_autonomous_compatibility(&plan),
            CompatibilityResult::Autonomous,
        );
    }

    #[test]
    fn test_compatibility_single_filter() {
        let plan = PhysicalPlan::GpuFilter {
            compare_op: CompareOp::Gt,
            column: "amount".to_string(),
            value: Value::Int(100),
            input: Box::new(scan("sales")),
        };
        assert_eq!(
            check_autonomous_compatibility(&plan),
            CompatibilityResult::Autonomous,
        );
    }

    #[test]
    fn test_compatibility_compound_filter() {
        use crate::sql::types::LogicalOp;
        let plan = PhysicalPlan::GpuCompoundFilter {
            op: LogicalOp::And,
            left: Box::new(PhysicalPlan::GpuFilter {
                compare_op: CompareOp::Gt,
                column: "amount".to_string(),
                value: Value::Int(200),
                input: Box::new(scan("sales")),
            }),
            right: Box::new(PhysicalPlan::GpuFilter {
                compare_op: CompareOp::Lt,
                column: "amount".to_string(),
                value: Value::Int(800),
                input: Box::new(scan("sales")),
            }),
        };
        assert_eq!(
            check_autonomous_compatibility(&plan),
            CompatibilityResult::Autonomous,
        );
    }

    #[test]
    fn test_compatibility_aggregate_no_groupby() {
        let plan = PhysicalPlan::GpuAggregate {
            functions: vec![(AggFunc::Count, "*".to_string())],
            group_by: vec![],
            input: Box::new(scan("sales")),
        };
        assert_eq!(
            check_autonomous_compatibility(&plan),
            CompatibilityResult::Autonomous,
        );
    }

    #[test]
    fn test_compatibility_aggregate_single_groupby() {
        let plan = PhysicalPlan::GpuAggregate {
            functions: vec![
                (AggFunc::Count, "*".to_string()),
                (AggFunc::Sum, "amount".to_string()),
            ],
            group_by: vec!["region".to_string()],
            input: Box::new(scan("sales")),
        };
        assert_eq!(
            check_autonomous_compatibility(&plan),
            CompatibilityResult::Autonomous,
        );
    }

    #[test]
    fn test_compatibility_aggregate_compound_filter_groupby() {
        // Headline query: compound filter + GROUP BY + multi-agg
        use crate::sql::types::LogicalOp;
        let plan = PhysicalPlan::GpuAggregate {
            functions: vec![
                (AggFunc::Count, "*".to_string()),
                (AggFunc::Sum, "amount".to_string()),
                (AggFunc::Min, "amount".to_string()),
                (AggFunc::Max, "amount".to_string()),
            ],
            group_by: vec!["region".to_string()],
            input: Box::new(PhysicalPlan::GpuCompoundFilter {
                op: LogicalOp::And,
                left: Box::new(PhysicalPlan::GpuFilter {
                    compare_op: CompareOp::Gt,
                    column: "amount".to_string(),
                    value: Value::Int(200),
                    input: Box::new(scan("sales")),
                }),
                right: Box::new(PhysicalPlan::GpuFilter {
                    compare_op: CompareOp::Lt,
                    column: "amount".to_string(),
                    value: Value::Int(800),
                    input: Box::new(scan("sales")),
                }),
            }),
        };
        assert_eq!(
            check_autonomous_compatibility(&plan),
            CompatibilityResult::Autonomous,
        );
    }

    #[test]
    fn test_compatibility_order_by_fallback() {
        let plan = PhysicalPlan::GpuSort {
            order_by: vec![("amount".to_string(), true)],
            input: Box::new(scan("sales")),
        };
        let result = check_autonomous_compatibility(&plan);
        match result {
            CompatibilityResult::Fallback(reason) => {
                assert!(
                    reason.contains("ORDER BY"),
                    "Reason should mention ORDER BY, got: {}",
                    reason
                );
            }
            other => panic!("Expected Fallback for ORDER BY, got {:?}", other),
        }
    }

    #[test]
    fn test_compatibility_multi_column_groupby_fallback() {
        let plan = PhysicalPlan::GpuAggregate {
            functions: vec![(AggFunc::Count, "*".to_string())],
            group_by: vec!["region".to_string(), "category".to_string()],
            input: Box::new(scan("sales")),
        };
        let result = check_autonomous_compatibility(&plan);
        match result {
            CompatibilityResult::Fallback(reason) => {
                assert!(
                    reason.contains("multi-column GROUP BY"),
                    "Reason should mention multi-column GROUP BY, got: {}",
                    reason
                );
            }
            other => panic!("Expected Fallback for multi-column GROUP BY, got {:?}", other),
        }
    }

    #[test]
    fn test_compatibility_limit_fallback() {
        let plan = PhysicalPlan::GpuLimit {
            count: 10,
            input: Box::new(scan("sales")),
        };
        let result = check_autonomous_compatibility(&plan);
        match result {
            CompatibilityResult::Fallback(reason) => {
                assert!(
                    reason.contains("LIMIT"),
                    "Reason should mention LIMIT, got: {}",
                    reason
                );
            }
            other => panic!("Expected Fallback for LIMIT, got {:?}", other),
        }
    }

    #[test]
    fn test_compatibility_sort_with_limit_fallback() {
        // ORDER BY ... LIMIT N -> fallback due to ORDER BY
        let plan = PhysicalPlan::GpuLimit {
            count: 10,
            input: Box::new(PhysicalPlan::GpuSort {
                order_by: vec![("amount".to_string(), true)],
                input: Box::new(scan("sales")),
            }),
        };
        let result = check_autonomous_compatibility(&plan);
        match result {
            CompatibilityResult::Fallback(reason) => {
                assert!(
                    reason.contains("ORDER BY"),
                    "Should detect ORDER BY inside LIMIT, got: {}",
                    reason
                );
            }
            other => panic!("Expected Fallback for ORDER BY+LIMIT, got {:?}", other),
        }
    }

    #[test]
    fn test_compatibility_aggregate_with_sort_input_fallback() {
        // Aggregate over a sort (unusual but possible): fallback due to sort
        let plan = PhysicalPlan::GpuAggregate {
            functions: vec![(AggFunc::Count, "*".to_string())],
            group_by: vec![],
            input: Box::new(PhysicalPlan::GpuSort {
                order_by: vec![("amount".to_string(), false)],
                input: Box::new(scan("sales")),
            }),
        };
        let result = check_autonomous_compatibility(&plan);
        match result {
            CompatibilityResult::Fallback(reason) => {
                assert!(
                    reason.contains("ORDER BY"),
                    "Should detect ORDER BY in aggregate input, got: {}",
                    reason
                );
            }
            other => panic!("Expected Fallback, got {:?}", other),
        }
    }
}
