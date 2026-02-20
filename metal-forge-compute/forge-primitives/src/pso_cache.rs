//! Pipeline State Object (PSO) cache for compute kernels.
//!
//! Supports both plain PSOs (keyed by function name) and specialized PSOs
//! with Metal function constants (keyed by `"fn_name:idx=val:idx=val"`).

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary, MTLPipelineOption,
};

/// Function constant value for PSO specialization.
#[derive(Clone, Copy, Debug)]
pub enum FnConstant {
    /// Boolean function constant.
    Bool(bool),
    /// Unsigned 32-bit integer function constant.
    U32(u32),
}

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

    /// Get or create a specialized PSO with function constant values.
    ///
    /// Each unique combination of (function_name, constants) compiles a distinct PSO.
    /// The cache key is `"fn_name:idx=val:idx=val"` so identical parameters reuse the PSO.
    pub fn get_or_create_specialized(
        &mut self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
        constants: &[(usize, FnConstant)],
    ) -> &ProtocolObject<dyn MTLComputePipelineState> {
        let key = Self::build_cache_key(function_name, constants);
        if !self.cache.contains_key(&key) {
            let pso = Self::compile_specialized_pso(library, function_name, constants);
            self.cache.insert(key.clone(), pso);
        }
        &self.cache[&key]
    }

    /// Build a cache key string: `"fn_name:idx=val:idx=val"`.
    fn build_cache_key(function_name: &str, constants: &[(usize, FnConstant)]) -> String {
        let mut key = function_name.to_string();
        for (idx, val) in constants {
            match val {
                FnConstant::Bool(b) => {
                    key.push_str(&format!(":{}={}", idx, *b as u32));
                }
                FnConstant::U32(v) => {
                    key.push_str(&format!(":{}={}", idx, v));
                }
            }
        }
        key
    }

    /// Compile a PSO specialized with function constant values.
    fn compile_specialized_pso(
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
        constants: &[(usize, FnConstant)],
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let constant_values = MTLFunctionConstantValues::new();

        for (idx, val) in constants {
            match val {
                FnConstant::Bool(b) => unsafe {
                    let ptr = NonNull::new(&*b as *const bool as *mut std::ffi::c_void)
                        .expect("constant value pointer is null");
                    constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::Bool, *idx);
                },
                FnConstant::U32(v) => unsafe {
                    let ptr = NonNull::new(v as *const u32 as *mut std::ffi::c_void)
                        .expect("constant value pointer is null");
                    constant_values.setConstantValue_type_atIndex(ptr, MTLDataType::UInt, *idx);
                },
            }
        }

        let fn_name = NSString::from_str(function_name);
        let function = library
            .newFunctionWithName_constantValues_error(&fn_name, &constant_values)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to create function '{}' with constants: {:?}",
                    function_name, e
                )
            });

        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setMaxTotalThreadsPerThreadgroup(256);
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
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to create specialized PSO for '{}': {:?}",
                    function_name, e
                )
            })
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

    #[test]
    fn test_specialized_cache_key_no_constants() {
        let key = PsoCache::build_cache_key("my_kernel", &[]);
        assert_eq!(key, "my_kernel");
    }

    #[test]
    fn test_specialized_cache_key_bool_constant() {
        let key = PsoCache::build_cache_key("sort_scatter", &[(0, FnConstant::Bool(true))]);
        assert_eq!(key, "sort_scatter:0=1");

        let key2 = PsoCache::build_cache_key("sort_scatter", &[(0, FnConstant::Bool(false))]);
        assert_eq!(key2, "sort_scatter:0=0");
    }

    #[test]
    fn test_specialized_cache_key_u32_constant() {
        let key = PsoCache::build_cache_key("sort_transform", &[(2, FnConstant::U32(42))]);
        assert_eq!(key, "sort_transform:2=42");
    }

    #[test]
    fn test_specialized_cache_key_multiple_constants() {
        let key = PsoCache::build_cache_key(
            "sort_scatter",
            &[
                (0, FnConstant::Bool(true)),
                (1, FnConstant::Bool(false)),
                (2, FnConstant::U32(1)),
            ],
        );
        assert_eq!(key, "sort_scatter:0=1:1=0:2=1");
    }

    #[test]
    fn test_specialized_cache_key_different_values_differ() {
        let k1 = PsoCache::build_cache_key("k", &[(0, FnConstant::U32(0))]);
        let k2 = PsoCache::build_cache_key("k", &[(0, FnConstant::U32(1))]);
        assert_ne!(k1, k2, "Different constant values should produce different keys");
    }

    #[test]
    fn test_specialized_cache_key_different_indices_differ() {
        let k1 = PsoCache::build_cache_key("k", &[(0, FnConstant::U32(1))]);
        let k2 = PsoCache::build_cache_key("k", &[(1, FnConstant::U32(1))]);
        assert_ne!(k1, k2, "Different constant indices should produce different keys");
    }

    #[test]
    fn test_specialized_cache_key_different_functions_differ() {
        let k1 = PsoCache::build_cache_key("sort_a", &[(0, FnConstant::Bool(true))]);
        let k2 = PsoCache::build_cache_key("sort_b", &[(0, FnConstant::Bool(true))]);
        assert_ne!(k1, k2, "Different function names should produce different keys");
    }
}
