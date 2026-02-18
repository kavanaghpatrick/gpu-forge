//! Metal context with texture/sampler support for GPU experiments.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLOrigin,
    MTLPixelFormat, MTLRegion, MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter, MTLSamplerState, MTLSize, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
};

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl MetalContext {
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");
        let library = Self::load_metallib(&device);
        Self {
            device,
            queue,
            library,
        }
    }

    pub fn make_pipeline(
        &self,
        name: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let ns_name = NSString::from_str(name);
        let func = self
            .library
            .newFunctionWithName(&ns_name)
            .unwrap_or_else(|| panic!("Kernel '{name}' not found"));
        self.device
            .newComputePipelineStateWithFunction_error(&func)
            .unwrap_or_else(|e| panic!("PSO for '{name}' failed: {e}"))
    }

    pub fn command_buffer(&self) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.queue
            .commandBuffer()
            .expect("Failed to create command buffer")
    }

    fn load_metallib(
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Retained<ProtocolObject<dyn MTLLibrary>> {
        let path = Self::find_metallib().expect("shaders.metallib not found");
        let path_ns = NSString::from_str(&path);
        #[allow(deprecated)]
        device
            .newLibraryWithFile_error(&path_ns)
            .unwrap_or_else(|e| panic!("Failed to load metallib: {e}"))
    }

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
                        let metallib = entry.path().join("out").join("shaders.metallib");
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
}

// ─── Buffer helpers ──────────────────────────────────────────

pub fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    size: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
        .expect("Failed to allocate buffer")
}

pub fn alloc_buffer_with_data<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let size = std::mem::size_of_val(data);
    unsafe {
        let ptr =
            NonNull::new(data.as_ptr() as *mut std::ffi::c_void).expect("null data pointer");
        device
            .newBufferWithBytes_length_options(ptr, size, MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate buffer with data")
    }
}

pub unsafe fn read_buffer_slice<T: Copy>(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
) -> Vec<T> {
    let ptr = buffer.contents().as_ptr() as *const T;
    std::slice::from_raw_parts(ptr, count).to_vec()
}

/// Write data into an existing buffer.
pub unsafe fn write_buffer<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, data: &[T]) {
    let ptr = buffer.contents().as_ptr() as *mut T;
    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
}

// ─── Texture helpers ─────────────────────────────────────────

/// Create a 2D R32Float texture with the given dimensions and data.
pub fn create_texture_r32f(
    device: &ProtocolObject<dyn MTLDevice>,
    width: usize,
    height: usize,
    data: &[f32],
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    assert_eq!(data.len(), width * height);

    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::R32Float,
            width,
            height,
            false,
        )
    };
    desc.setUsage(MTLTextureUsage::ShaderRead);

    let texture = device
        .newTextureWithDescriptor(&desc)
        .expect("Failed to create texture");

    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width,
            height,
            depth: 1,
        },
    };

    let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).unwrap();
    unsafe {
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region,
            0,
            ptr,
            width * std::mem::size_of::<f32>(),
        );
    }

    texture
}

/// Create a sampler with linear filtering and clamp-to-edge addressing.
pub fn create_linear_sampler(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Retained<ProtocolObject<dyn MTLSamplerState>> {
    let desc = MTLSamplerDescriptor::new();
    desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
    desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
    desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
    desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
    device
        .newSamplerStateWithDescriptor(&desc)
        .expect("Failed to create sampler")
}

// ─── Dispatch helpers ────────────────────────────────────────

pub fn dispatch_1d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    total_threads: usize,
) {
    encoder.setComputePipelineState(pipeline);
    unsafe {
        for (buffer, index) in buffers {
            encoder.setBuffer_offset_atIndex(Some(*buffer), 0, *index);
        }
    }
    let tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
    let groups = total_threads.div_ceil(tg);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
}

/// Dispatch with exact threadgroup size (for experiments needing specific TG dimensions).
pub fn dispatch_1d_tg(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    num_threadgroups: usize,
    threads_per_tg: usize,
) {
    encoder.setComputePipelineState(pipeline);
    unsafe {
        for (buffer, index) in buffers {
            encoder.setBuffer_offset_atIndex(Some(*buffer), 0, *index);
        }
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: num_threadgroups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        },
    );
}

// ─── Timing ──────────────────────────────────────────────────

pub fn gpu_elapsed_ms(cmd: &ProtocolObject<dyn MTLCommandBuffer>) -> f64 {
    let start = cmd.GPUStartTime();
    let end = cmd.GPUEndTime();
    if start == 0.0 {
        return 0.0;
    }
    (end - start) * 1000.0
}

pub fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}
