//! Pipeline State Object (PSO) cache with function constant specialization.
//!
//! Metal function constants allow compile-time specialization of kernel code,
//! eliminating runtime branches. Each unique combination of function constants
//! produces a different PSO that must be compiled and cached.
//!
//! The `PsoCache` stores compiled PSOs keyed by function name + constant values,
//! so repeated calls with the same parameters reuse the cached PSO.
//!
//! Binary archive support (save/load) enables persisting compiled PSOs to disk,
//! avoiding cold-compilation overhead on subsequent runs.

use std::collections::HashMap;
use std::path::Path;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLPipelineOption,
};

/// Type tag for function constant serialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ConstantType {
    UInt = 0,
    Float = 1,
    Bool = 2,
}

impl ConstantType {
    fn to_mtl(self) -> MTLDataType {
        match self {
            ConstantType::UInt => MTLDataType::UInt,
            ConstantType::Float => MTLDataType::Float,
            ConstantType::Bool => MTLDataType::Bool,
        }
    }
}

/// A single function constant value to set on a PSO.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstantValue {
    /// Set by index.
    Index {
        index: usize,
        type_tag: ConstantType,
        bytes: Vec<u8>,
    },
    /// Set by name.
    Named {
        name: String,
        type_tag: ConstantType,
        bytes: Vec<u8>,
    },
}

/// Key for PSO cache lookup: function name + function constant values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PsoKey {
    /// Kernel function name (e.g., "flash_attention").
    pub function_name: String,
    /// Ordered list of function constant values for identity.
    pub constants: Vec<ConstantValue>,
}

impl PsoKey {
    /// Create a key with no function constants.
    pub fn simple(function_name: &str) -> Self {
        Self {
            function_name: function_name.to_string(),
            constants: Vec::new(),
        }
    }

    /// Add a uint constant by index.
    pub fn with_uint(mut self, index: usize, value: u32) -> Self {
        self.constants.push(ConstantValue::Index {
            index,
            type_tag: ConstantType::UInt,
            bytes: value.to_ne_bytes().to_vec(),
        });
        self
    }

    /// Add a float constant by index.
    pub fn with_float(mut self, index: usize, value: f32) -> Self {
        self.constants.push(ConstantValue::Index {
            index,
            type_tag: ConstantType::Float,
            bytes: value.to_ne_bytes().to_vec(),
        });
        self
    }

    /// Add a bool constant by index.
    pub fn with_bool(mut self, index: usize, value: bool) -> Self {
        self.constants.push(ConstantValue::Index {
            index,
            type_tag: ConstantType::Bool,
            bytes: vec![u8::from(value)],
        });
        self
    }

    /// Add a uint constant by name.
    pub fn with_named_uint(mut self, name: &str, value: u32) -> Self {
        self.constants.push(ConstantValue::Named {
            name: name.to_string(),
            type_tag: ConstantType::UInt,
            bytes: value.to_ne_bytes().to_vec(),
        });
        self
    }

    /// Add a float constant by name.
    pub fn with_named_float(mut self, name: &str, value: f32) -> Self {
        self.constants.push(ConstantValue::Named {
            name: name.to_string(),
            type_tag: ConstantType::Float,
            bytes: value.to_ne_bytes().to_vec(),
        });
        self
    }

    /// Add a bool constant by name.
    pub fn with_named_bool(mut self, name: &str, value: bool) -> Self {
        self.constants.push(ConstantValue::Named {
            name: name.to_string(),
            type_tag: ConstantType::Bool,
            bytes: vec![u8::from(value)],
        });
        self
    }
}

/// Cache of compiled Metal compute pipeline states.
///
/// Avoids recompiling the same function+constants combination on every dispatch.
/// PSO compilation with function constants triggers the Metal compiler at runtime,
/// so caching is essential for interactive performance.
pub struct PsoCache {
    cache: HashMap<PsoKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl PsoCache {
    /// Create a new empty PSO cache backed by the given Metal library.
    pub fn new(library: Retained<ProtocolObject<dyn MTLLibrary>>) -> Self {
        Self {
            cache: HashMap::new(),
            library,
        }
    }

    /// Get or compile a PSO for the given key.
    ///
    /// If a matching PSO exists in the cache, returns it immediately.
    /// Otherwise, compiles a new PSO using the Metal compiler and caches it.
    pub fn get_or_compile(
        &mut self,
        key: &PsoKey,
    ) -> &ProtocolObject<dyn MTLComputePipelineState> {
        if !self.cache.contains_key(key) {
            let pso = Self::compile_pso(&self.library, key);
            self.cache.insert(key.clone(), pso);
        }
        &self.cache[key]
    }

    /// Number of cached PSOs.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all cached PSOs.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Build MTLFunctionConstantValues from a PsoKey's constants.
    fn build_constant_values(key: &PsoKey) -> Retained<MTLFunctionConstantValues> {
        let constant_values = MTLFunctionConstantValues::new();

        for cv in &key.constants {
            match cv {
                ConstantValue::Index {
                    index,
                    type_tag,
                    bytes,
                } => {
                    let mtl_type = type_tag.to_mtl();
                    unsafe {
                        let ptr = NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void)
                            .expect("constant value pointer is null");
                        constant_values.setConstantValue_type_atIndex(ptr, mtl_type, *index);
                    }
                }
                ConstantValue::Named {
                    name,
                    type_tag,
                    bytes,
                } => {
                    let mtl_type = type_tag.to_mtl();
                    let ns_name = NSString::from_str(name);
                    unsafe {
                        let ptr = NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void)
                            .expect("constant value pointer is null");
                        constant_values.setConstantValue_type_withName(ptr, mtl_type, &ns_name);
                    }
                }
            }
        }

        constant_values
    }

    /// Compile a PSO with the given function constants.
    fn compile_pso(
        library: &ProtocolObject<dyn MTLLibrary>,
        key: &PsoKey,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let constant_values = Self::build_constant_values(key);

        let fn_name = NSString::from_str(&key.function_name);
        let function = library
            .newFunctionWithName_constantValues_error(&fn_name, &constant_values)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to create function '{}' with constants: {:?}",
                    key.function_name, e
                )
            });

        let device = library.device();
        device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|e| {
                panic!("Failed to create PSO for '{}': {:?}", key.function_name, e)
            })
    }

    /// Create a file URL from a path.
    fn file_url(path: &Path) -> Result<Retained<NSURL>, String> {
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("Non-UTF8 path: {}", path.display()))?;
        let ns_path = NSString::from_str(path_str);
        Ok(NSURL::fileURLWithPath_isDirectory(&ns_path, false))
    }

    /// Save all cached PSOs to a binary archive file.
    ///
    /// Binary archives persist compiled GPU code to disk, avoiding cold-compilation
    /// overhead on subsequent runs. This is especially useful for Proto 4 where
    /// we benchmark compilation of 72+ function constant variants.
    pub fn save_archive(&self, path: &Path) -> Result<(), String> {
        let device = self.library.device();
        let descriptor = MTLBinaryArchiveDescriptor::new();
        let archive = device
            .newBinaryArchiveWithDescriptor_error(&descriptor)
            .map_err(|e| format!("Failed to create binary archive: {:?}", e))?;

        // Add each cached PSO's descriptor to the archive
        for key in self.cache.keys() {
            let constant_values = Self::build_constant_values(key);
            let fn_name = NSString::from_str(&key.function_name);
            let function = self
                .library
                .newFunctionWithName_constantValues_error(&fn_name, &constant_values)
                .map_err(|e| {
                    format!("Failed to create function '{}': {:?}", key.function_name, e)
                })?;

            let desc = MTLComputePipelineDescriptor::new();
            desc.setComputeFunction(Some(&function));

            archive
                .addComputePipelineFunctionsWithDescriptor_error(&desc)
                .map_err(|e| format!("Failed to add PSO to archive: {:?}", e))?;
        }

        // Serialize to disk
        let url = Self::file_url(path)?;
        archive
            .serializeToURL_error(&url)
            .map_err(|e| format!("Failed to serialize archive: {:?}", e))?;

        Ok(())
    }

    /// Load a binary archive from disk.
    ///
    /// The archive acts as a compilation cache: if a PSO's compiled code is found
    /// in the archive, Metal skips recompilation. If not found, it compiles as usual.
    pub fn load_archive(
        library: &ProtocolObject<dyn MTLLibrary>,
        path: &Path,
    ) -> Result<Retained<ProtocolObject<dyn MTLBinaryArchive>>, String> {
        let device = library.device();
        let descriptor = MTLBinaryArchiveDescriptor::new();

        let url = Self::file_url(path)?;
        descriptor.setUrl(Some(&url));

        device
            .newBinaryArchiveWithDescriptor_error(&descriptor)
            .map_err(|e| {
                format!(
                    "Failed to load binary archive from {}: {:?}",
                    path.display(),
                    e
                )
            })
    }

    /// Compile a PSO using a binary archive for accelerated lookup.
    ///
    /// If the archive contains the compiled code, compilation is near-instant.
    /// Otherwise falls back to normal compilation.
    pub fn compile_with_archive(
        library: &ProtocolObject<dyn MTLLibrary>,
        key: &PsoKey,
        archive: &ProtocolObject<dyn MTLBinaryArchive>,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let constant_values = Self::build_constant_values(key);
        let fn_name = NSString::from_str(&key.function_name);

        let function = library
            .newFunctionWithName_constantValues_error(&fn_name, &constant_values)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to create function '{}' with constants: {:?}",
                    key.function_name, e
                )
            });

        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(&function));

        // Set the binary archive for accelerated lookup
        let archives = objc2_foundation::NSArray::from_slice(&[archive]);
        desc.setBinaryArchives(Some(&archives));

        let device = library.device();
        device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::empty(),
                None,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to create PSO for '{}' with archive: {:?}",
                    key.function_name, e
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;

    #[test]
    fn test_pso_cache() {
        let gpu = GpuDevice::shared();
        let mut cache = PsoCache::new(gpu.library.clone());

        // Cache starts empty
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        // Compile a PSO for the _stub kernel (exists in our stub metallib)
        let key = PsoKey::simple("_stub");
        let pso = cache.get_or_compile(&key);

        // Verify we got a valid PSO
        assert!(pso.maxTotalThreadsPerThreadgroup() > 0);
        assert_eq!(cache.len(), 1);

        // Second call should return cached version (same count)
        let _pso2 = cache.get_or_compile(&key);
        assert_eq!(cache.len(), 1, "Second call should hit cache, not recompile");

        // Different key should compile a new PSO
        // (same function, but distinct key — tests cache discrimination)
        let key2 = PsoKey::simple("_stub");
        let _pso3 = cache.get_or_compile(&key2);
        // Same key content => still 1 entry
        assert_eq!(
            cache.len(),
            1,
            "Equal keys should map to the same cache entry"
        );
    }

    #[test]
    fn test_pso_key_builders() {
        let k1 = PsoKey::simple("flash_attention")
            .with_uint(0, 64)
            .with_uint(1, 16)
            .with_bool(2, true);

        let k2 = PsoKey::simple("flash_attention")
            .with_uint(0, 64)
            .with_uint(1, 16)
            .with_bool(2, true);

        assert_eq!(k1, k2, "Same parameters should produce equal keys");

        let k3 = PsoKey::simple("flash_attention")
            .with_uint(0, 128) // different HEAD_DIM
            .with_uint(1, 16)
            .with_bool(2, true);

        assert_ne!(
            k1, k3,
            "Different constant values should produce different keys"
        );
    }

    #[test]
    fn test_pso_key_named_constants() {
        let k1 = PsoKey::simple("my_kernel")
            .with_named_uint("HEAD_DIM", 64)
            .with_named_float("SCALE", 0.125)
            .with_named_bool("USE_ALIBI", false);

        let k2 = PsoKey::simple("my_kernel")
            .with_named_uint("HEAD_DIM", 64)
            .with_named_float("SCALE", 0.125)
            .with_named_bool("USE_ALIBI", false);

        assert_eq!(k1, k2);

        let k3 = PsoKey::simple("my_kernel")
            .with_named_uint("HEAD_DIM", 64)
            .with_named_float("SCALE", 0.125)
            .with_named_bool("USE_ALIBI", true);

        assert_ne!(k1, k3);
    }

    #[test]
    fn test_pso_cache_clear() {
        let gpu = GpuDevice::shared();
        let mut cache = PsoCache::new(gpu.library.clone());

        let key = PsoKey::simple("_stub");
        let _ = cache.get_or_compile(&key);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_constant_type_to_mtl() {
        assert_eq!(ConstantType::UInt.to_mtl(), MTLDataType::UInt);
        assert_eq!(ConstantType::Float.to_mtl(), MTLDataType::Float);
        assert_eq!(ConstantType::Bool.to_mtl(), MTLDataType::Bool);
    }
}
