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
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;

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
}
