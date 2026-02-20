//! setBytes Lookup Table experiment.
//!
//! Metal's setBytes pushes up to 4KB directly into the command stream.
//! The data lands in constant address space, which the shader compiler can
//! preload into dedicated uniform registers. Apple GPU L1 is only 8KB --
//! setBytes' 4KB limit is exactly half L1, ensuring guaranteed cache residency.
//!
//! Compare: 256-entry u32 LUT via setBytes (constant space, register-speed)
//!     vs   same LUT in a device buffer (device space, L1 cache dependent).
//! 8 dependent lookups per element amplify the cache/register speed difference.

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, ExploitParams, GpuTimer, MetalContext,
    PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// LUT size: 256 entries * 4 bytes = 1024 bytes (well under 4KB limit).
const LUT_SIZE: usize = 256;

pub struct SetBytesLutExperiment {
    input_data: Vec<u32>,
    lut_data: Vec<u32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_lut_device: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_setbytes_ms: f64,
    gpu_buffer_ms: f64,
    size: usize,
}

impl SetBytesLutExperiment {
    pub fn new() -> Self {
        Self {
            input_data: Vec::new(),
            lut_data: Vec::new(),
            buf_input: None,
            buf_output: None,
            buf_lut_device: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_setbytes_ms: 0.0,
            gpu_buffer_ms: 0.0,
            size: 0,
        }
    }
}

impl Experiment for SetBytesLutExperiment {
    fn name(&self) -> &str {
        "setbytes_lut"
    }

    fn description(&self) -> &str {
        "setBytes LUT (constant/register) vs device buffer LUT (L1 cache)"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // Random input keys
        self.input_data = gen.uniform_u32(size);

        // Random LUT: 256 entries
        self.lut_data = gen.uniform_u32(LUT_SIZE);

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.input_data));
        self.buf_output = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));

        // Device buffer for the LUT (used by buffer_lut kernel)
        self.buf_lut_device = Some(alloc_buffer_with_data(&ctx.device, &self.lut_data));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 8, // 8 dependent lookups
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_setbytes_lut");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_buffer_lut");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output = self.buf_output.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();
        let lut_device = self.buf_lut_device.as_ref().unwrap();

        let threads_per_tg: usize = 256;
        let num_tgs = self.size.div_ceil(threads_per_tg);

        // --- setBytes version (constant address space) ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_setbytes_lut");
            let enc = cmd.computeCommandEncoder().unwrap();

            enc.setComputePipelineState(pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(input.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(output.as_ref()), 0, 1);

                // Push LUT via setBytes at index 2 (constant address space)
                let lut_ptr =
                    NonNull::new(self.lut_data.as_ptr() as *mut std::ffi::c_void).unwrap();
                let lut_bytes = self.lut_data.len() * std::mem::size_of::<u32>();
                enc.setBytes_length_atIndex(lut_ptr, lut_bytes, 2);

                enc.setBuffer_offset_atIndex(Some(params.as_ref()), 0, 3);
            }

            let grid = objc2_metal::MTLSize {
                width: num_tgs,
                height: 1,
                depth: 1,
            };
            let tg = objc2_metal::MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_setbytes_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // --- Device buffer version ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_buffer_lut");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output.as_ref(), 1),
                    (lut_device.as_ref(), 2),
                    (params.as_ref(), 3),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_buffer_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        self.gpu_setbytes_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // Same 8-chain LUT lookups on CPU
        let mut acc: u32 = 0;
        for &key in &self.input_data {
            let mut val = self.lut_data[(key & 0xFF) as usize];
            val = self.lut_data[((val ^ key) & 0xFF) as usize];
            val = self.lut_data[((val ^ (key >> 8)) & 0xFF) as usize];
            val = self.lut_data[((val ^ (key >> 16)) & 0xFF) as usize];
            val = self.lut_data[((val ^ (key >> 24)) & 0xFF) as usize];
            val = self.lut_data[((val ^ key) & 0xFF) as usize];
            val = self.lut_data[((val ^ (key >> 4)) & 0xFF) as usize];
            acc ^= self.lut_data[((val ^ (key >> 12)) & 0xFF) as usize];
        }
        std::hint::black_box(acc);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_setbytes_ms > 0.0 && self.gpu_buffer_ms > 0.0 {
            Ok(())
        } else {
            Err("No GPU timing recorded".to_string())
        }
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("setbytes_ms".to_string(), self.gpu_setbytes_ms);
        m.insert("device_buf_ms".to_string(), self.gpu_buffer_ms);

        let speedup = if self.gpu_setbytes_ms > 0.0 {
            self.gpu_buffer_ms / self.gpu_setbytes_ms
        } else {
            0.0
        };
        m.insert("setbytes_speedup_x".to_string(), speedup);

        // Lookups per second
        let lookups = size as f64 * 8.0; // 8 dependent lookups per element
        let setbytes_glps = if self.gpu_setbytes_ms > 0.0 {
            lookups / (self.gpu_setbytes_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("setbytes_glookups_per_sec".to_string(), setbytes_glps);

        // For the table BW column
        let bytes = size as f64 * 4.0; // input reads
        let gbs = if self.gpu_setbytes_ms > 0.0 {
            bytes / (self.gpu_setbytes_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
