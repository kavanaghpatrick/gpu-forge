//! SIMD Register LUT vs Threadgroup Memory LUT experiment.
//!
//! Exploits simd_broadcast(value, lane_id) to build a 32-entry lookup table
//! entirely in SIMD registers -- zero memory access required.
//! Each lane holds one LUT entry. Any lane can look up any value by
//! broadcasting from the appropriate lane.
//!
//! Compare: SIMD register LUT (via simd_broadcast) vs threadgroup memory LUT

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer_slice, ExploitParams, GpuTimer,
    MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

pub struct SimdRegisterLutExperiment {
    data: Vec<u32>,
    buf_input: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_simd: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_output_tg: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_simd_ms: f64,
    gpu_tg_ms: f64,
    size: usize,
}

impl SimdRegisterLutExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_input: None,
            buf_output_simd: None,
            buf_output_tg: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_simd_ms: 0.0,
            gpu_tg_ms: 0.0,
            size: 0,
        }
    }
}

impl Experiment for SimdRegisterLutExperiment {
    fn name(&self) -> &str {
        "simd_register_lut"
    }

    fn description(&self) -> &str {
        "SIMD register LUT (simd_broadcast, zero memory) vs threadgroup memory LUT"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        self.data = gen.uniform_u32(size);

        self.buf_input = Some(alloc_buffer_with_data(&ctx.device, &self.data));
        self.buf_output_simd = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));
        self.buf_output_tg = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: 0,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_simd_register_lut");
        self.pso_cache
            .get_or_create(ctx.library(), "exploit_tg_memory_lut");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.buf_input.as_ref().unwrap();
        let output_simd = self.buf_output_simd.as_ref().unwrap();
        let output_tg = self.buf_output_tg.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        // --- SIMD register LUT (simd_broadcast) ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_simd_register_lut");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output_simd.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_simd_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        // --- Threadgroup memory LUT ---
        {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "exploit_tg_memory_lut");
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d(
                &enc,
                pso,
                &[
                    (input.as_ref(), 0),
                    (output_tg.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                self.size,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.gpu_tg_ms = GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0);
        }

        self.gpu_simd_ms
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        // Same hash chain on CPU
        let lut: Vec<u32> = (0..32).map(|i| (i * 7 + 13) & 0xFF).collect();
        let mut acc: u32 = 0;
        for &key in &self.data {
            let mut val = lut[(key & 0x1F) as usize];
            val = lut[((val ^ key) & 0x1F) as usize];
            val = lut[((val ^ (key >> 5)) & 0x1F) as usize];
            val = lut[((val ^ (key >> 10)) & 0x1F) as usize];
            val = lut[((val ^ (key >> 15)) & 0x1F) as usize];
            val = lut[((val ^ (key >> 20)) & 0x1F) as usize];
            val = lut[((val ^ (key >> 25)) & 0x1F) as usize];
            val = lut[((val ^ key) & 0x1F) as usize];
            acc ^= val;
        }
        std::hint::black_box(acc);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // Verify SIMD and TG produce identical results
        let simd_out: Vec<u32> = unsafe {
            read_buffer_slice::<u32>(self.buf_output_simd.as_ref().unwrap().as_ref(), self.size)
        };
        let tg_out: Vec<u32> = unsafe {
            read_buffer_slice::<u32>(self.buf_output_tg.as_ref().unwrap().as_ref(), self.size)
        };

        // Check first 1000 elements
        let check_count = self.size.min(1000);
        for i in 0..check_count {
            if simd_out[i] != tg_out[i] {
                return Err(format!(
                    "Mismatch at index {}: simd={}, tg={}",
                    i, simd_out[i], tg_out[i]
                ));
            }
        }
        Ok(())
    }

    fn metrics(&self, _elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        m.insert("simd_lut_ms".to_string(), self.gpu_simd_ms);
        m.insert("tg_lut_ms".to_string(), self.gpu_tg_ms);

        let speedup = if self.gpu_simd_ms > 0.0 {
            self.gpu_tg_ms / self.gpu_simd_ms
        } else {
            0.0
        };
        m.insert("simd_speedup_x".to_string(), speedup);

        // 8 dependent lookups per element = 8 reads + 1 write
        let ops = size as f64 * 9.0 * 4.0; // 9 uint ops per element
        let simd_gbs = if self.gpu_simd_ms > 0.0 {
            ops / (self.gpu_simd_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), simd_gbs);

        m
    }
}
