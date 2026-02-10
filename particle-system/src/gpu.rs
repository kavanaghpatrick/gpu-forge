use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::msg_send;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLPixelFormat,
};
use objc2_quartz_core::CAMetalLayer;

/// Core GPU state: device, command queue, metal layer, and shader library.
pub struct GpuState {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub layer: Retained<CAMetalLayer>,
    pub library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl GpuState {
    /// Initialize Metal device, command queue, CAMetalLayer, and load the shader library.
    pub fn new() -> Self {
        // Create the default Metal device
        let device = MTLCreateSystemDefaultDevice().expect("Failed to get default Metal device");

        // Create command queue
        let command_queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");

        // Create and configure CAMetalLayer
        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setFramebufferOnly(true);

        // Load the compiled .metallib from the build output
        let metallib_path = Self::find_metallib();
        let path_ns = NSString::from_str(&metallib_path);
        #[allow(deprecated)]
        let library = device
            .newLibraryWithFile_error(&path_ns)
            .expect("Failed to load shaders.metallib");

        println!("Metal device: {:?}", device.name());
        println!("Loaded metallib from: {}", metallib_path);

        Self {
            device,
            command_queue,
            layer,
            library,
        }
    }

    /// Attach the CAMetalLayer to an NSView from raw window handle.
    ///
    /// # Safety
    /// The `ns_view` pointer must be a valid NSView. Must be called on the main thread.
    pub unsafe fn attach_layer_to_view(&self, ns_view: NonNull<std::ffi::c_void>) {
        let view = ns_view.as_ptr() as *mut AnyObject;
        let layer_ref: &CAMetalLayer = &self.layer;

        // Cast CAMetalLayer to AnyObject for setLayer:
        let layer_obj = layer_ref as *const CAMetalLayer as *const AnyObject;

        unsafe {
            // [view setWantsLayer:YES]
            let _: () = msg_send![view, setWantsLayer: true];
            // [view setLayer:layer]
            let _: () = msg_send![view, setLayer: layer_obj];
        }
    }

    /// Find the shaders.metallib in the build output directory.
    fn find_metallib() -> String {
        let exe_path = std::env::current_exe().expect("Failed to get current exe path");
        let target_dir = exe_path
            .parent()
            .expect("Failed to get parent of exe");

        // Search the build directory for shaders.metallib
        let build_dir = target_dir.join("build");
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

        // Fallback: alongside the executable
        let fallback = target_dir.join("shaders.metallib");
        if fallback.exists() {
            return fallback.to_string_lossy().into_owned();
        }

        panic!(
            "Could not find shaders.metallib. Searched: {}",
            build_dir.display()
        );
    }
}
