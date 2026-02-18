//! Pipeline State Object (PSO) cache for compute kernels.
//!
//! Simplified version of gpu-query's PsoCache without function constants,
//! suitable for benchmark kernels that don't need runtime specialization.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLPipelineOption,
};

/// Cache of compiled Metal compute pipeline states, keyed by function name.
pub struct PsoCache {
    cache: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl Default for PsoCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PsoCache {
    /// Create a new empty PSO cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get or create a PSO for the given kernel function name.
    ///
    /// If a matching PSO exists in the cache, returns it immediately.
    /// Otherwise, compiles a new PSO and caches it.
    pub fn get_or_create(
        &mut self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
    ) -> &ProtocolObject<dyn MTLComputePipelineState> {
        if !self.cache.contains_key(function_name) {
            let pso = Self::compile_pso(library, function_name);
            self.cache.insert(function_name.to_string(), pso);
        }
        &self.cache[function_name]
    }

    /// Number of cached PSOs.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Compile a PSO for the given function name using descriptor-based creation
    /// with occupancy hints (maxTotalThreadsPerThreadgroup=256,
    /// threadGroupSizeIsMultipleOfThreadExecutionWidth=true).
    fn compile_pso(
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let fn_name = NSString::from_str(function_name);
        #[allow(deprecated)]
        let function = library
            .newFunctionWithName(&fn_name)
            .unwrap_or_else(|| panic!("Kernel function '{}' not found in metallib", function_name));

        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setMaxTotalThreadsPerThreadgroup(256);
        // SAFETY: We always dispatch with threadgroup sizes that are multiples of
        // the thread execution width (32 on Apple Silicon).
        unsafe {
            descriptor.setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        }

        let device = library.device();
        device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .unwrap_or_else(|e| panic!("Failed to create PSO for '{}': {:?}", function_name, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pso_cache_new_is_empty() {
        let cache = PsoCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_pso_cache_default_is_empty() {
        let cache = PsoCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_pso_cache_key_equality() {
        // HashMap keys are Strings, so same function name should map to same entry
        let mut map: HashMap<String, u32> = HashMap::new();
        map.insert("reduce_sum_u32".to_string(), 1);
        map.insert("reduce_sum_u32".to_string(), 2);
        assert_eq!(map.len(), 1, "Same key should overwrite, not duplicate");
        assert_eq!(map["reduce_sum_u32"], 2);
    }

    #[test]
    fn test_pso_cache_key_inequality() {
        // Different function names are different keys
        let mut map: HashMap<String, u32> = HashMap::new();
        map.insert("reduce_sum_u32".to_string(), 1);
        map.insert("reduce_sum_f32".to_string(), 2);
        assert_eq!(map.len(), 2, "Different keys should be separate entries");
        assert_ne!(map["reduce_sum_u32"], map["reduce_sum_f32"]);
    }
}
