//! Inlined from forge-primitives (pso_cache.rs, dispatch.rs, metal_ctx.rs).

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLPipelineOption, MTLResourceOptions,
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

/// Allocate a Metal buffer of `size` bytes with StorageModeShared.
pub fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    size: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let options = MTLResourceOptions::StorageModeShared;
    device
        .newBufferWithLength_options(size, options)
        .expect("Failed to allocate Metal buffer")
}

/// Initialize the default Metal device and command queue.
pub fn init_device_and_queue() -> (
    Retained<ProtocolObject<dyn MTLDevice>>,
    Retained<ProtocolObject<dyn MTLCommandQueue>>,
) {
    let device = MTLCreateSystemDefaultDevice().expect("Failed to get default Metal device");

    let queue = device
        .newCommandQueue()
        .expect("Failed to create command queue");

    (device, queue)
}
