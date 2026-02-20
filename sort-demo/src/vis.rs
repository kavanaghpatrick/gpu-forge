use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLoadAction, MTLPixelFormat, MTLRenderCommandEncoder,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLSize, MTLStoreAction,
    MTLLibrary, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};
use objc2_quartz_core::CAMetalDrawable;

#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum VisMode {
    Heatmap,
    Barchart,
}

pub fn texture_dim(element_count: usize) -> u32 {
    match element_count {
        n if n <= 1_048_576 => 1024,
        n if n <= 4_194_304 => 2048,
        n if n <= 16_777_216 => 4096,
        _ => 8192,
    }
}

#[repr(C)]
pub struct DemoParams {
    pub element_count: u32,
    pub texture_width: u32,
    pub texture_height: u32,
    pub max_value: u32,
}

pub struct Visualization {
    pso_heatmap: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_barchart: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    render_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    heatmap_texture: Retained<ProtocolObject<dyn MTLTexture>>,
    sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    pub mode: VisMode,
    pub texture_dim: u32,
}

impl Visualization {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        element_count: usize,
    ) -> Self {
        let pso_heatmap = make_compute_pso(device, library, "value_to_heatmap");
        let pso_barchart = make_compute_pso(device, library, "value_to_barchart");

        // Render pipeline: fullscreen triangle
        let vtx_name = NSString::from_str("fullscreen_vertex");
        let frag_name = NSString::from_str("fullscreen_fragment");
        let vtx_fn = library
            .newFunctionWithName(&vtx_name)
            .expect("Failed to find fullscreen_vertex");
        let frag_fn = library
            .newFunctionWithName(&frag_name)
            .expect("Failed to find fullscreen_fragment");

        let render_desc = MTLRenderPipelineDescriptor::new();
        render_desc.setVertexFunction(Some(&vtx_fn));
        render_desc.setFragmentFunction(Some(&frag_fn));
        let color_attachment =
            unsafe { render_desc.colorAttachments().objectAtIndexedSubscript(0) };
        color_attachment.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        color_attachment.setBlendingEnabled(false);
        #[allow(deprecated)]
        let render_pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&render_desc)
            .expect("Failed to create render pipeline");

        let dim = texture_dim(element_count);
        let heatmap_texture = create_heatmap_texture(device, dim);

        // Sampler: linear, clamp-to-edge
        let sampler_desc = MTLSamplerDescriptor::new();
        sampler_desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sampler_desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        sampler_desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sampler_desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        let sampler = device
            .newSamplerStateWithDescriptor(&sampler_desc)
            .expect("Failed to create sampler");

        Self {
            pso_heatmap,
            pso_barchart,
            render_pipeline,
            heatmap_texture,
            sampler,
            mode: VisMode::Heatmap,
            texture_dim: dim,
        }
    }

    #[allow(dead_code)]
    pub fn resize_if_needed(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        element_count: usize,
    ) {
        let new_dim = texture_dim(element_count);
        if new_dim != self.texture_dim {
            self.heatmap_texture = create_heatmap_texture(device, new_dim);
            self.texture_dim = new_dim;
        }
    }

    pub fn encode_visualize(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        data_buffer: &ProtocolObject<dyn MTLBuffer>,
        element_count: usize,
    ) {
        let encoder = cmd
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        let params = DemoParams {
            element_count: element_count as u32,
            texture_width: self.texture_dim,
            texture_height: self.texture_dim,
            max_value: 0xFFFFFFFF,
        };

        match self.mode {
            VisMode::Heatmap => {
                encoder.setComputePipelineState(&self.pso_heatmap);
                let thread_count = element_count;
                let tg_size = MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                };
                let grid_size = MTLSize {
                    width: thread_count.div_ceil(256),
                    height: 1,
                    depth: 1,
                };
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(data_buffer), 0, 0);
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::new(&params as *const DemoParams as *mut DemoParams)
                            .unwrap()
                            .cast(),
                        std::mem::size_of::<DemoParams>(),
                        1,
                    );
                    encoder.setTexture_atIndex(Some(&self.heatmap_texture), 0);
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
                }
            }
            VisMode::Barchart => {
                encoder.setComputePipelineState(&self.pso_barchart);
                let thread_count = (self.texture_dim as usize) * (self.texture_dim as usize);
                let tg_size = MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                };
                let grid_size = MTLSize {
                    width: thread_count.div_ceil(256),
                    height: 1,
                    depth: 1,
                };
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(data_buffer), 0, 0);
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::new(&params as *const DemoParams as *mut DemoParams)
                            .unwrap()
                            .cast(),
                        std::mem::size_of::<DemoParams>(),
                        1,
                    );
                    encoder.setTexture_atIndex(Some(&self.heatmap_texture), 0);
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
                }
            }
        }
        encoder.endEncoding();
    }

    pub fn encode_render(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        drawable: &ProtocolObject<dyn CAMetalDrawable>,
    ) {
        let render_pass_desc =
            objc2_metal::MTLRenderPassDescriptor::renderPassDescriptor();
        let color_attachment =
            unsafe { render_pass_desc.colorAttachments().objectAtIndexedSubscript(0) };
        color_attachment.setTexture(Some(&drawable.texture()));
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setStoreAction(MTLStoreAction::Store);
        color_attachment.setClearColor(MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 1.0,
        });

        let encoder = cmd
            .renderCommandEncoderWithDescriptor(&render_pass_desc)
            .expect("Failed to create render encoder");

        encoder.setRenderPipelineState(&self.render_pipeline);
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(&self.heatmap_texture), 0);
            encoder.setFragmentSamplerState_atIndex(Some(&self.sampler), 0);
            encoder.drawPrimitives_vertexStart_vertexCount(
                objc2_metal::MTLPrimitiveType::Triangle,
                0,
                3,
            );
        }
        encoder.endEncoding();
    }
}

fn create_heatmap_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    dim: u32,
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::BGRA8Unorm,
            dim as usize,
            dim as usize,
            false,
        )
    };
    desc.setUsage(MTLTextureUsage(
        MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::ShaderWrite.0,
    ));
    desc.setStorageMode(MTLStorageMode::Private);
    device
        .newTextureWithDescriptor(&desc)
        .expect("Failed to create heatmap texture")
}

fn make_compute_pso(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let fn_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&fn_name)
        .unwrap_or_else(|| panic!("Failed to find function '{}'", name));
    #[allow(deprecated)]
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|e| panic!("Failed to create pipeline '{}': {:?}", name, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn demo_params_size() {
        assert_eq!(size_of::<DemoParams>(), 16);
    }

    #[test]
    fn texture_dim_1m() {
        assert_eq!(texture_dim(1_000_000), 1024);
    }

    #[test]
    fn texture_dim_4m() {
        assert_eq!(texture_dim(4_000_000), 2048);
    }

    #[test]
    fn texture_dim_16m() {
        assert_eq!(texture_dim(16_000_000), 4096);
    }

    #[test]
    fn texture_dim_64m() {
        assert_eq!(texture_dim(64_000_000), 8192);
    }
}
