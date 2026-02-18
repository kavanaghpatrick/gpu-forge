//! GPU Byte Search (Pattern Matching) experiment.
//!
//! The GPU can search through raw byte data at DRAM bandwidth.
//! Each thread checks 16 bytes (4 uint) for a 4-byte pattern.
//! At 273 GB/s, this scans ~17 GB/s of data per second.
//!
//! Simulates searching through memory-mapped file contents via bytesNoCopy.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, ExploitParams, GpuTimer, MetalContext,
    PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// The 4-byte pattern we search for.
const SEARCH_PATTERN: u32 = 0xDEAD_BEEF;

pub struct ByteSearchExperiment {
    data: Vec<u32>,
    buf_data: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_count: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pso_cache: PsoCache,
    gpu_matches: u32,
    cpu_matches: u32,
    size: usize,
}

impl ByteSearchExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            buf_data: None,
            buf_count: None,
            buf_params: None,
            pso_cache: PsoCache::new(),
            gpu_matches: 0,
            cpu_matches: 0,
            size: 0,
        }
    }
}

impl Experiment for ByteSearchExperiment {
    fn name(&self) -> &str {
        "byte_search"
    }

    fn description(&self) -> &str {
        "GPU byte search: scan memory for 4-byte pattern at DRAM bandwidth"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![1_000_000, 10_000_000, 100_000_000]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // Generate random data, then inject some known patterns
        self.data = gen.uniform_u32(size * 4); // 4 uints per thread = 16 bytes

        // Inject ~0.1% patterns to find
        let inject_count = (size * 4) / 1000;
        for i in 0..inject_count {
            let idx = (i * 997) % (size * 4); // pseudo-random positions
            self.data[idx] = SEARCH_PATTERN;
        }

        self.buf_data = Some(alloc_buffer_with_data(&ctx.device, &self.data));
        self.buf_count = Some(alloc_buffer(&ctx.device, std::mem::size_of::<u32>()));

        let params = ExploitParams {
            element_count: size as u32,
            num_passes: 1,
            mode: SEARCH_PATTERN,
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        self.pso_cache
            .get_or_create(ctx.library(), "exploit_byte_search");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let data = self.buf_data.as_ref().unwrap();
        let count = self.buf_count.as_ref().unwrap();
        let params = self.buf_params.as_ref().unwrap();

        // Zero count
        unsafe {
            *(count.contents().as_ptr() as *mut u32) = 0;
        }

        let cmd = ctx.queue.commandBuffer().unwrap();
        let pso = self
            .pso_cache
            .get_or_create(ctx.library(), "exploit_byte_search");
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            pso,
            &[
                (data.as_ref(), 0),
                (count.as_ref(), 1),
                (params.as_ref(), 2),
            ],
            self.size,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        self.gpu_matches = unsafe { *(count.contents().as_ptr() as *const u32) };

        GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        use forge_primitives::BenchTimer;
        let timer = BenchTimer::start();

        let mut matches = 0u32;
        for &v in &self.data {
            if v == SEARCH_PATTERN {
                matches += 1;
            }
        }
        self.cpu_matches = matches;

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_matches == self.cpu_matches {
            Ok(())
        } else {
            Err(format!(
                "Match count mismatch: gpu={}, cpu={}",
                self.gpu_matches, self.cpu_matches
            ))
        }
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();

        // Total data scanned: each thread checks 4 uints = 16 bytes
        let bytes_scanned = size as f64 * 16.0;

        let gbs = if elapsed_ms > 0.0 {
            bytes_scanned / (elapsed_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);
        m.insert("scanned_mb".to_string(), bytes_scanned / 1e6);
        m.insert("matches_found".to_string(), self.gpu_matches as f64);
        m.insert("bw_utilization_pct".to_string(), gbs / 273.0 * 100.0);

        m
    }
}
