//! Pipeline State Object (PSO) cache for compute kernels.
//!
//! Simplified version of gpu-query's PsoCache without function constants,
//! suitable for benchmark kernels that don't need runtime specialization.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

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

    /// Compile a PSO for the given function name (no function constants).
    fn compile_pso(
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let fn_name = NSString::from_str(function_name);
        #[allow(deprecated)]
        let function = library
            .newFunctionWithName(&fn_name)
            .unwrap_or_else(|| panic!("Kernel function '{}' not found in metallib", function_name));

        let device = library.device();
        device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to create PSO for '{}': {:?}",
                    function_name, e
                )
            })
    }
}
