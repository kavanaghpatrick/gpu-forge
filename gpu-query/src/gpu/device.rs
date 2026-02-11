//! Metal device initialization: device, command queue, shader library.
//!
//! Reuses the particle-system/gpu.rs pattern adapted for compute-only
//! analytics workloads (no render pipeline, no CAMetalLayer).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};

/// Core GPU state for query execution: device, command queue, shader library.
pub struct GpuDevice {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl Default for GpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDevice {
    /// Initialize Metal device, command queue, and load the compiled metallib.
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("Failed to get default Metal device");

        let command_queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");

        let metallib_path = Self::find_metallib();
        let path_ns = NSString::from_str(&metallib_path);
        #[allow(deprecated)]
        let library = device
            .newLibraryWithFile_error(&path_ns)
            .expect("Failed to load shaders.metallib");

        Self {
            device,
            command_queue,
            library,
        }
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
