//! CSV parsing experiment: GPU parallel byte scanning vs CPU sequential scan.
//!
//! Demonstrates GPU text processing by parsing raw CSV byte data:
//!   - GPU: two kernels dispatched sequentially in one command buffer:
//!     1. csv_newline_detect: per-byte newline flag generation (1 if '\n', else 0)
//!     2. csv_field_count: threadgroup-atomic comma+newline counting
//!   - CPU: sequential byte scan counting commas and newlines
//!
//! Validates that GPU and CPU produce the same comma and newline counts.
//! Reports MB/s throughput and rows/sec.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, read_buffer_slice, BenchTimer,
    CsvBenchParams, GpuTimer, MetalContext, PsoCache,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// CSV parsing experiment comparing GPU parallel byte scanning vs CPU sequential scan.
pub struct JsonParseExperiment {
    /// Raw CSV byte data.
    data: Vec<u8>,
    /// Metal buffer holding the CSV bytes.
    input_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for newline flags (u32 per byte).
    flags_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for atomic counters: [comma_count, newline_count].
    counters_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for CsvBenchParams.
    params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU result: (comma_count, newline_count).
    gpu_result: (u32, u32),
    /// CPU result: (comma_count, newline_count).
    cpu_result: (u32, u32),
    /// Current byte count.
    size: usize,
}

impl JsonParseExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            input_buffer: None,
            flags_buffer: None,
            counters_buffer: None,
            params_buffer: None,
            pso_cache: PsoCache::new(),
            gpu_result: (0, 0),
            cpu_result: (0, 0),
            size: 0,
        }
    }

    /// Zero the counters buffer before each GPU run.
    fn zero_counters(&self) {
        if let Some(ref buf) = self.counters_buffer {
            unsafe {
                let ptr = buf.contents().as_ptr() as *mut u32;
                *ptr = 0;
                *ptr.add(1) = 0;
            }
        }
    }
}

impl Experiment for JsonParseExperiment {
    fn name(&self) -> &str {
        "json_parse"
    }

    fn description(&self) -> &str {
        "CSV byte parsing: GPU parallel newline detect + field count vs CPU sequential"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![
            100_000,      // 100K bytes
            1_000_000,    // 1M bytes
            10_000_000,   // 10M bytes
            100_000_000,  // 100M bytes
        ]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;
        self.data = gen.csv_records(size);

        // Input buffer: raw CSV bytes
        self.input_buffer = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Flags buffer: u32 per byte for newline detection
        self.flags_buffer = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));

        // Counters buffer: 2 x u32 (comma_count, newline_count)
        self.counters_buffer = Some(alloc_buffer(
            &ctx.device,
            2 * std::mem::size_of::<u32>(),
        ));
        self.zero_counters();

        // Params buffer
        let params = CsvBenchParams {
            byte_count: size as u32,
            _pad: [0; 3],
        };
        self.params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Pre-warm PSO cache
        self.pso_cache
            .get_or_create(ctx.library(), "csv_newline_detect");
        self.pso_cache
            .get_or_create(ctx.library(), "csv_field_count");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        // Zero counters before each run
        self.zero_counters();

        let input = self.input_buffer.as_ref().expect("setup not called");
        let flags = self.flags_buffer.as_ref().expect("setup not called");
        let counters = self.counters_buffer.as_ref().expect("setup not called");
        let params = self.params_buffer.as_ref().expect("setup not called");

        // Single command buffer with two compute passes
        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");

        // --- Pass 1: csv_newline_detect ---
        {
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "csv_newline_detect");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            dispatch_1d(
                &encoder,
                pso,
                &[
                    (input.as_ref(), 0),
                    (flags.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                self.size,
            );

            encoder.endEncoding();
        }

        // --- Pass 2: csv_field_count ---
        {
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "csv_field_count");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            dispatch_1d(
                &encoder,
                pso,
                &[
                    (input.as_ref(), 0),
                    (counters.as_ref(), 1),
                    (params.as_ref(), 2),
                ],
                self.size,
            );

            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read back counters
        let result: Vec<u32> = unsafe { read_buffer_slice(counters.as_ref(), 2) };
        self.gpu_result = (result[0], result[1]);

        GpuTimer::elapsed_ms(&cmd_buf).unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();

        let mut commas: u32 = 0;
        let mut newlines: u32 = 0;
        for &byte in &self.data {
            if byte == b',' {
                commas += 1;
            } else if byte == b'\n' {
                newlines += 1;
            }
        }
        self.cpu_result = (commas, newlines);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_result.0 != self.cpu_result.0 {
            return Err(format!(
                "Comma count mismatch: GPU={} CPU={}",
                self.gpu_result.0, self.cpu_result.0
            ));
        }
        if self.gpu_result.1 != self.cpu_result.1 {
            return Err(format!(
                "Newline count mismatch: GPU={} CPU={}",
                self.gpu_result.1, self.cpu_result.1
            ));
        }

        // Validate field count consistency:
        // total_fields = commas + newlines (each row has fields_per_row-1 commas + 1 newline)
        let gpu_fields = self.gpu_result.0 + self.gpu_result.1;
        let cpu_fields = self.cpu_result.0 + self.cpu_result.1;
        if gpu_fields != cpu_fields {
            return Err(format!(
                "Total field count mismatch: GPU={} CPU={}",
                gpu_fields, cpu_fields
            ));
        }

        Ok(())
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let seconds = elapsed_ms / 1000.0;
        let bytes = size as f64;

        // MB/s throughput
        let mb_per_sec = if seconds > 0.0 {
            bytes / seconds / 1e6
        } else {
            0.0
        };
        m.insert("mb_per_sec".to_string(), mb_per_sec);

        // GB/s bandwidth
        let gb_per_sec = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gb_per_sec);

        // Rows per second (newline_count / seconds)
        let rows = self.gpu_result.1 as f64;
        let rows_per_sec = if seconds > 0.0 {
            rows / seconds
        } else {
            0.0
        };
        m.insert("rows_per_sec".to_string(), rows_per_sec);

        m.insert("bytes_processed".to_string(), bytes);
        m.insert("comma_count".to_string(), self.gpu_result.0 as f64);
        m.insert("newline_count".to_string(), self.gpu_result.1 as f64);

        m
    }
}
