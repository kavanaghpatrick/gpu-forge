//! Metal GPU state: device, command queue, shader pipelines, and CAMetalLayer.
//!
//! Initializes the Metal device, loads the compiled `.metallib`, and creates
//! all compute and render pipeline state objects used by the particle system.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::msg_send;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLCommandQueue, MTLCompareFunction, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice,
    MTLLibrary, MTLPixelFormat, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
};
use objc2_quartz_core::CAMetalLayer;

/// Core GPU state: device, command queue, metal layer, shader library, compute and render pipelines.
#[allow(dead_code)]
pub struct GpuState {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub layer: Retained<CAMetalLayer>,
    pub library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pub emission_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub update_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub grid_clear_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub grid_populate_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub sync_indirect_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub prepare_dispatch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub render_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub depth_stencil_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
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

        // Create compute pipeline for emission kernel
        let emission_fn_name = NSString::from_str("emission_kernel");
        let emission_fn = library
            .newFunctionWithName(&emission_fn_name)
            .expect("Failed to find emission_kernel in metallib");
        #[allow(deprecated)]
        let emission_pipeline = device
            .newComputePipelineStateWithFunction_error(&emission_fn)
            .expect("Failed to create emission compute pipeline");

        // Create compute pipeline for update_physics_kernel
        let update_fn_name = NSString::from_str("update_physics_kernel");
        let update_fn = library
            .newFunctionWithName(&update_fn_name)
            .expect("Failed to find update_physics_kernel in metallib");
        #[allow(deprecated)]
        let update_pipeline = device
            .newComputePipelineStateWithFunction_error(&update_fn)
            .expect("Failed to create update compute pipeline");

        // Create compute pipeline for grid_clear_kernel
        let grid_clear_fn_name = NSString::from_str("grid_clear_kernel");
        let grid_clear_fn = library
            .newFunctionWithName(&grid_clear_fn_name)
            .expect("Failed to find grid_clear_kernel in metallib");
        #[allow(deprecated)]
        let grid_clear_pipeline = device
            .newComputePipelineStateWithFunction_error(&grid_clear_fn)
            .expect("Failed to create grid_clear compute pipeline");

        // Create compute pipeline for grid_populate_kernel
        let grid_populate_fn_name = NSString::from_str("grid_populate_kernel");
        let grid_populate_fn = library
            .newFunctionWithName(&grid_populate_fn_name)
            .expect("Failed to find grid_populate_kernel in metallib");
        #[allow(deprecated)]
        let grid_populate_pipeline = device
            .newComputePipelineStateWithFunction_error(&grid_populate_fn)
            .expect("Failed to create grid_populate compute pipeline");

        // Create compute pipeline for sync_indirect_args kernel
        let sync_fn_name = NSString::from_str("sync_indirect_args");
        let sync_fn = library
            .newFunctionWithName(&sync_fn_name)
            .expect("Failed to find sync_indirect_args in metallib");
        #[allow(deprecated)]
        let sync_indirect_pipeline = device
            .newComputePipelineStateWithFunction_error(&sync_fn)
            .expect("Failed to create sync_indirect_args compute pipeline");

        // Create compute pipeline for prepare_dispatch kernel
        let prepare_dispatch_fn_name = NSString::from_str("prepare_dispatch");
        let prepare_dispatch_fn = library
            .newFunctionWithName(&prepare_dispatch_fn_name)
            .expect("Failed to find prepare_dispatch in metallib");
        #[allow(deprecated)]
        let prepare_dispatch_pipeline = device
            .newComputePipelineStateWithFunction_error(&prepare_dispatch_fn)
            .expect("Failed to create prepare_dispatch compute pipeline");

        // Create render pipeline state for particle billboard quads
        let vertex_fn_name = NSString::from_str("vertex_main");
        let vertex_fn = library
            .newFunctionWithName(&vertex_fn_name)
            .expect("Failed to find vertex_main in metallib");
        let fragment_fn_name = NSString::from_str("fragment_main");
        let fragment_fn = library
            .newFunctionWithName(&fragment_fn_name)
            .expect("Failed to find fragment_main in metallib");

        let render_desc = MTLRenderPipelineDescriptor::new();
        render_desc.setVertexFunction(Some(&vertex_fn));
        render_desc.setFragmentFunction(Some(&fragment_fn));

        // Configure color attachment 0: BGRA8Unorm with alpha blending
        // SAFETY: objectAtIndexedSubscript(0) is always valid — MTLRenderPipelineDescriptor
        // always has at least one color attachment slot. The returned reference is valid for
        // the lifetime of render_desc.
        let color_attachment = unsafe {
            render_desc.colorAttachments().objectAtIndexedSubscript(0)
        };
        color_attachment.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        color_attachment.setBlendingEnabled(true);
        color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::SourceAlpha);
        color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

        #[allow(deprecated)]
        let render_pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&render_desc)
            .expect("Failed to create render pipeline state");

        // Create depth stencil state (depth write off for transparent particles)
        let depth_desc = MTLDepthStencilDescriptor::new();
        depth_desc.setDepthWriteEnabled(false);
        depth_desc.setDepthCompareFunction(MTLCompareFunction::Always);
        let depth_stencil_state = device
            .newDepthStencilStateWithDescriptor(&depth_desc)
            .expect("Failed to create depth stencil state");

        println!("Metal device: {:?}", device.name());
        println!("Loaded metallib from: {}", metallib_path);
        println!("Emission pipeline created successfully");
        println!("Update pipeline created successfully");
        println!("Grid clear/populate pipelines created successfully");
        println!("Prepare dispatch pipeline created successfully");
        println!("Render pipeline created successfully");

        Self {
            device,
            command_queue,
            layer,
            library,
            emission_pipeline,
            update_pipeline,
            grid_clear_pipeline,
            grid_populate_pipeline,
            sync_indirect_pipeline,
            prepare_dispatch_pipeline,
            render_pipeline,
            depth_stencil_state,
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

        // SAFETY: view is a valid NSView pointer (guaranteed by caller's safety contract).
        // CAMetalLayer cast to AnyObject is safe for Objective-C message passing.
        // setWantsLayer: and setLayer: are standard NSView methods available on macOS.
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
