//! Metal device initialization: device, command queue, shader library.
//!
//! Follows the gpu-query/src/gpu/device.rs pattern adapted for
//! the forge-primitives compute benchmark library.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};

/// Core GPU state: device, command queue, shader library.
pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
}

impl MetalContext {
    /// Initialize Metal device and command queue.
    /// Library is loaded lazily when shaders are available.
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("Failed to get default Metal device");

        let queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");

        let library = Self::try_load_metallib(&device);

        Self {
            device,
            queue,
            library,
        }
    }

    /// Try to load the metallib from the build output directory.
    /// Returns None if no metallib is found (e.g., no shaders compiled yet).
    fn try_load_metallib(
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Option<Retained<ProtocolObject<dyn MTLLibrary>>> {
        let metallib_path = Self::find_metallib()?;
        let path_ns = NSString::from_str(&metallib_path);
        #[allow(deprecated)]
        device.newLibraryWithFile_error(&path_ns).ok()
    }

    /// Find the shaders.metallib in the build output directory.
    fn find_metallib() -> Option<String> {
        let exe_path = std::env::current_exe().ok()?;
        let target_dir = exe_path.parent()?;

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
                            return Some(metallib.to_string_lossy().into_owned());
                        }
                    }
                }
            }

            let fallback = dir.join("shaders.metallib");
            if fallback.exists() {
                return Some(fallback.to_string_lossy().into_owned());
            }
        }

        None
    }

    /// Get the library, panicking if not loaded.
    pub fn library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        self.library
            .as_ref()
            .expect("Metal library not loaded -- did you compile shaders?")
    }
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new()
    }
}
