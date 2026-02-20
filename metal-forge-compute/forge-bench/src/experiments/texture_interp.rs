//! Texture Hardware as Free Interpolation experiment.
//!
//! The Texture Mapping Unit (TMU) is a SEPARATE fixed-function hardware block.
//! It does bilinear interpolation, format conversion, and address clamping
//! with ZERO ALU cost. When you call texture.sample(), the TMU works while
//! the ALU can do other things.
//!
//! Compare: TMU bilinear interpolation (hardware, free)
//!     vs   Manual lerp in compute (ALU-bound)

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLPixelFormat, MTLSize, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, ExploitParams, GpuTimer, MetalContext,
    PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// Source texture size (number of float samples).
const SRC_SIZE: usize = 4096;

pub struct TextureInterpExperiment {
    src_data: Vec<f32>,
    texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    buf_src: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_tmu: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_compute: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params_tex: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params_compute: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_tmu_ms: f64,
    gpu_compute_ms: f64,
    size: usize,
}

impl TextureInterpExperiment {
    pub fn new() -> Self {
        Self {
            src_data: Vec::new(),
            texture: None,
            buf_src: None,
            buf_output_tmu: None,
            buf_output_compute: None,
            buf_params_tex: None,
            buf_params_compute: None,
            pso_cache: PsoCache::new(),
            gpu_tmu_ms: 0.0,
            gpu_compute_ms: 0.0,
            size: 0,
        }
    }
}

impl Experiment for TextureInterpExperiment {
    fn name(&self) -> &str {
        "texture_interp"
    }

    fn description(&self) -> &str {
        "Texture TMU bilinear interpolation (free, hardware) vs manual compute lerp"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // Generate source data for interpolation
        self.src_data = gen
            .uniform_u32(SRC_SIZE)
            .iter()
            .map(|&v| (v % 1000) as f32 * 0.001)
            .collect();

        // Create 2D texture (width=SRC_SIZE, height=1)
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::R32Float,
                SRC_SIZE,
                1,
                false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderRead);
        let texture = ctx.device.newTextureWithDescriptor(&desc).unwrap();

        // Fill texture with source data
        let region = objc2_metal::MTLRegion {
            origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: SRC_SIZE,
                height: 1,
                depth: 1,
            },
        };
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                std::ptr::NonNull::new(self.src_data.as_ptr() as *mut _).unwrap(),
                SRC_SIZE * std::mem::size_of::<f32>(),
            );
        }
        self.texture = Some(texture);

        // Buffer version of source data (for compute kernel)
        self.buf_src = Some(alloc_buffer_with_data(&ctx.device, &self.src_data));

        // Output buffers
        self.buf_output_tmu = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<f32>(),
        ));
        self.buf_output_compute = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<f32>(),
        ));

        // Params for texture kernel: element_count = output size
        let params_tex = ExploitParams {
            element_count: size as u32,
            num_passes: SRC_SIZE as u32,
            mode: 0,
            _pad: 0,
        };
        self.buf_params_tex = Some(alloc_buffer_with_data(&ctx.device, &[params_tex]));

        // Params for compute kernel: element_count = output size, num_passes = src size
        let params_compute = ExploitParams {
            element_count: size as u32,
            num_passes: SRC_SIZE as u32,
            mode: 0,
            _pad: 0,
        };
        self.buf_params_compute = Some(alloc_buffer_with_data(&ctx.device, &[params_compute]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_texture_interp");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_compute_interp");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let texture = self.texture.as_ref().unwrap();
        let output_tmu = self.buf_output_tmu.as_ref().unwrap();
        let output_compute = self.buf_output_compute.as_ref().unwrap();
        let params_tex = self.buf_params_tex.as_ref().unwrap();
        let params_compute = self.buf_params_compute.as_ref().unwrap();
        let buf_src = self.buf_src.as_ref().unwrap();

        // --- TMU bilinear interpolation ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_texture_interp");
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso);

            // Set texture at texture index 0
            unsafe {
                enc.setTexture_atIndex(Some(texture.as_ref()), 0);
                enc.setBuffer_offset_atIndex(Some(output_tmu.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(params_tex.as_ref()), 0, 1);
            }

            let threads_per_tg = pso.maxTotalThreadsPerThreadgroup().min(256);
            let tg_count = self.size.div_ceil(threads_per_tg);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: tg_count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_tg,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_tmu_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // --- Manual compute lerp ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_compute_interp");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (buf_src.as_ref(), 0),
                    (output_compute.as_ref(), 1),
                    (params_compute.as_ref(), 2),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_compute_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        self.gpu_tmu_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // Manual lerp on CPU
        let src_size = SRC_SIZE;
        let mut acc = 0.0f64;
        for i in 0..self.size {
            let u = i as f64 / self.size as f64 * (src_size - 1) as f64;
            let lo = u as usize;
            let hi = (lo + 1).min(src_size - 1);
            let frac = u - lo as f64;
            let v = self.src_data[lo] as f64 * (1.0 - frac) + self.src_data[hi] as f64 * frac;
            acc += v;
        }
        std::hint::black_box(acc);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_tmu_ms > 0.0 && self.gpu_compute_ms > 0.0 {
            Ok(())
        } else {
            Err("No GPU timing recorded".to_string())
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("tmu_ms".to_string(), self.gpu_tmu_ms);
        m.insert("compute_ms".to_string(), self.gpu_compute_ms);

        let tmu_advantage = if self.gpu_tmu_ms > 0.0 {
            self.gpu_compute_ms / self.gpu_tmu_ms
        } else {
            0.0
        };
        m.insert("tmu_advantage_x".to_string(), tmu_advantage);

        // Write bandwidth: size floats written = size * 4 bytes
        let bytes = size as f64 * 4.0;
        let gbs = if self.gpu_tmu_ms > 0.0 {
            bytes / (self.gpu_tmu_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
