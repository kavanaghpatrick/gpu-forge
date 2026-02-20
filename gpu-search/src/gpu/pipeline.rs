//! Pipeline State Object (PSO) cache for gpu-search compute kernels.
//!
//! Loads the compiled `shaders.metallib` from build output and creates
//! cached PSOs for all search kernel functions. Unlike gpu-query which
//! uses function constants for specialization, gpu-search kernels are
//! fixed-function, so we cache by function name only.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

/// The four search kernel functions compiled into shaders.metallib.
pub const SEARCH_KERNELS: &[&str] = &[
    "content_search_kernel",
    "turbo_search_kernel",
    "batch_search_kernel",
    "path_filter_kernel",
];

/// Cache of compiled Metal compute pipeline states, keyed by function name.
///
/// All PSOs are created eagerly at init time since we have a small, fixed
/// set of kernels. This avoids first-use latency during search.
pub struct PsoCache {
    cache: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl PsoCache {
    /// Create a new PSO cache, loading the metallib and compiling PSOs for
    /// all search kernels.
    ///
    /// # Panics
    /// Panics if the metallib cannot be found/loaded or any kernel function
    /// is missing from the library.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let metallib_path = Self::find_metallib();
        let path_ns = NSString::from_str(&metallib_path);
        #[allow(deprecated)]
        let library = device
            .newLibraryWithFile_error(&path_ns)
            .expect("Failed to load shaders.metallib");

        let mut cache = HashMap::new();

        for &kernel_name in SEARCH_KERNELS {
            let pso = Self::compile_pso(&library, device, kernel_name);
            cache.insert(kernel_name.to_string(), pso);
        }

        Self { cache, library }
    }

    /// Get the PSO for a kernel function by name.
    ///
    /// Returns `None` if the kernel name is not in the cache.
    pub fn get(&self, kernel_name: &str) -> Option<&ProtocolObject<dyn MTLComputePipelineState>> {
        self.cache.get(kernel_name).map(|r| r.as_ref())
    }

    /// Get the underlying Metal library.
    pub fn library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        &self.library
    }

    /// Number of cached PSOs.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Compile a single PSO for the given kernel function name.
    fn compile_pso(
        library: &ProtocolObject<dyn MTLLibrary>,
        device: &ProtocolObject<dyn MTLDevice>,
        kernel_name: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let fn_name = NSString::from_str(kernel_name);
        #[allow(deprecated)]
        let function = library
            .newFunctionWithName(&fn_name)
            .unwrap_or_else(|| panic!("Kernel function '{}' not found in metallib", kernel_name));

        device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to create compute pipeline for '{}': {:?}",
                    kernel_name, e
                )
            })
    }

    /// Find the shaders.metallib in the build output directory.
    ///
    /// Searches multiple locations because the exe may be in target/debug/
    /// (for binaries) or target/debug/deps/ (for tests).
    fn find_metallib() -> String {
        let exe_path = std::env::current_exe().expect("Failed to get current exe path");
        let target_dir = exe_path.parent().expect("Failed to get parent of exe");

        // Candidate directories to search for build/<hash>/out/shaders.metallib
        let search_dirs = [
            target_dir.to_path_buf(),
            target_dir
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_default(),
        ];

        for dir in &search_dirs {
            let build_dir = dir.join("build");
            if build_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(&build_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let metallib = path.join("out").join("shaders.metallib");
                        if metallib.exists() {
                            return metallib.to_string_lossy().into_owned();
                        }
                    }
                }
            }

            // Fallback: alongside the executable/in target dir
            let fallback = dir.join("shaders.metallib");
            if fallback.exists() {
                return fallback.to_string_lossy().into_owned();
            }
        }

        panic!(
            "Could not find shaders.metallib. Searched: {} and parent",
            target_dir.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    #[test]
    fn test_pso_cache() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");

        let cache = PsoCache::new(&device);

        // All 4 kernels should be cached
        assert_eq!(cache.len(), 4, "Expected 4 cached PSOs");
        assert!(!cache.is_empty());

        // Each kernel should be retrievable
        for &kernel_name in SEARCH_KERNELS {
            let pso = cache.get(kernel_name);
            assert!(
                pso.is_some(),
                "PSO for '{}' should exist in cache",
                kernel_name
            );
        }

        // Unknown kernel should return None
        assert!(cache.get("nonexistent_kernel").is_none());

        // Library should be accessible
        let _lib = cache.library();
    }
}
