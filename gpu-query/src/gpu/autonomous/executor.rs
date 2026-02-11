//! Autonomous query executor with persistent kernel and re-dispatch chain.
//!
//! This module implements:
//! - `FusedPsoCache`: compiled pipeline state cache for the AOT fused query kernel,
//!   keyed by (filter_count, agg_count, has_group_by) function constant triple.
//! - (Future) `AutonomousExecutor`: persistent kernel dispatch with work queue polling.

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use super::jit::JitCompiler;
use super::loader::{ColumnInfo, ResidentTable};
use super::types::{OutputBuffer, QueryParamsSlot};
use crate::sql::physical_plan::PhysicalPlan;

/// Cache of compiled PSOs for the fused query kernel, keyed by function constant triple.
///
/// Each unique (filter_count, agg_count, has_group_by) combination produces a specialized
/// Metal pipeline via function constants, avoiding runtime branches in the shader.
pub struct FusedPsoCache {
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
            (row_count + threads_per_threadgroup - 1) / threads_per_threadgroup
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
            (row_count + threads_per_threadgroup - 1) / threads_per_threadgroup
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
}
