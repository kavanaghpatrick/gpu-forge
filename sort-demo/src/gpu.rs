use std::ptr::NonNull;

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLPixelFormat,
};
use objc2_quartz_core::CAMetalLayer;

pub struct GpuState {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub layer: Retained<CAMetalLayer>,
    pub library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl GpuState {
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("Failed to get default Metal device");
        let command_queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");

        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setFramebufferOnly(true);

        let metallib_path = Self::find_metallib();
        let path_ns = NSString::from_str(&metallib_path);
        #[allow(deprecated)]
        let library = device
            .newLibraryWithFile_error(&path_ns)
            .expect("Failed to load shaders.metallib");

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
        let layer_obj = layer_ref as *const CAMetalLayer as *const AnyObject;
        unsafe {
            let _: () = msg_send![view, setWantsLayer: true];
            let _: () = msg_send![view, setLayer: layer_obj];
        }
    }

    /// Create a compute pipeline for the named function in the library.
    pub fn make_compute_pipeline(
        &self,
        name: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let fn_name = NSString::from_str(name);
        let function = self
            .library
            .newFunctionWithName(&fn_name)
            .unwrap_or_else(|| panic!("Failed to find function '{}' in metallib", name));
        #[allow(deprecated)]
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|e| panic!("Failed to create compute pipeline for '{}': {:?}", name, e))
    }

    fn find_metallib() -> String {
        let exe_path = std::env::current_exe().expect("Failed to get current exe path");
        let target_dir = exe_path.parent().expect("Failed to get parent of exe");

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
