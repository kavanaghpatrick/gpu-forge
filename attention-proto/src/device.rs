//! Metal device initialization: device, command queue, shader library.
//!
//! Thread-safe singleton via OnceLock. Adapted from gpu-query/src/gpu/device.rs
//! with enhanced metallib search (OUT_DIR, build dirs, fallback paths).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};
use std::sync::OnceLock;

/// Global singleton instance.
static SHARED: OnceLock<GpuDevice> = OnceLock::new();

/// Core GPU state: device, command queue, shader library.
pub struct GpuDevice {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

// SAFETY: MTLDevice, MTLCommandQueue, MTLLibrary are thread-safe Metal objects.
// Apple documents that these can be shared across threads.
unsafe impl Send for GpuDevice {}
unsafe impl Sync for GpuDevice {}

impl Default for GpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDevice {
    /// Get or initialize the global singleton.
    pub fn shared() -> &'static GpuDevice {
        SHARED.get_or_init(GpuDevice::new)
    }

    /// Create a fresh instance (useful for tests needing isolation).
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
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to load shaders.metallib from {}: {:?}",
                    metallib_path, e
                )
            });

        Self {
            device,
            command_queue,
            library,
        }
    }

    /// Find shaders.metallib in build output directories.
    ///
    /// Search order:
    /// 1. OUT_DIR env var (set by build.rs during compilation)
    /// 2. target/{debug,release}/build/*/out/ (runtime search)
    /// 3. target/{debug,release}/ fallback (alongside executable)
    fn find_metallib() -> String {
        // 1. Try OUT_DIR (available during `cargo test` and `cargo build`)
        if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let path = std::path::PathBuf::from(&out_dir).join("shaders.metallib");
            if path.exists() {
                return path.to_string_lossy().into_owned();
            }
        }

        // 2. Search relative to the running executable
        let exe_path = std::env::current_exe().expect("Failed to get current exe path");
        let target_dir = exe_path.parent().expect("Failed to get parent of exe");

        // Candidate directories: exe dir and its parent (for deps/ subdirectory)
        let search_dirs = [
            target_dir.to_path_buf(),
            target_dir
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_default(),
        ];

        for dir in &search_dirs {
            // Search build/<hash>/out/shaders.metallib
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

            // Fallback: alongside the executable / in target dir
            let fallback = dir.join("shaders.metallib");
            if fallback.exists() {
                return fallback.to_string_lossy().into_owned();
            }
        }

        // 3. Also try target/release and target/debug explicitly
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let manifest = std::path::PathBuf::from(&manifest_dir);
            for profile in &["debug", "release"] {
                let target_profile = manifest.join("target").join(profile);
                let build_dir = target_profile.join("build");
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
            }
        }

        panic!(
            "Could not find shaders.metallib. Searched: OUT_DIR, {} and parent, CARGO_MANIFEST_DIR/target/*/build/",
            target_dir.display()
        );
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        // Retained<T> handles release automatically via objc2's Drop impl.
        // This explicit Drop is a documentation marker — Metal objects are
        // reference-counted and will be deallocated when all Retained handles drop.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_init() {
        let gpu = GpuDevice::shared();

        // Verify device is valid
        let name = gpu.device.name();
        assert!(!name.to_string().is_empty(), "Device name should not be empty");

        // Verify command queue exists (non-null retained pointer)
        // Just accessing it is sufficient — Retained<T> guarantees non-null
        let _ = &gpu.command_queue;

        // Verify library loaded
        // The stub metallib should have at least the _stub function
        let _ = &gpu.library;

        // Verify singleton returns same instance
        let gpu2 = GpuDevice::shared();
        assert!(
            std::ptr::eq(gpu, gpu2),
            "shared() should return the same instance"
        );
    }
}
