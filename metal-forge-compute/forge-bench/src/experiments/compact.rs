//! Stream compaction experiment: GPU scan-based compact vs CPU rayon filter.
//!
//! Tests stream compaction on u32 data. The GPU implementation uses a
//! 5-dispatch pipeline in a single command buffer:
//!   1. compact_flags: evaluate predicate (value > threshold)
//!   2. scan_local: per-threadgroup Blelloch scan of flags
//!   3. scan_partials: scan the partials array
//!   4. scan_add_offsets: add scanned partials to each threadgroup's elements
//!   5. compact_scatter: write selected elements to compacted output
//!
//! Uses threshold = u32::MAX / 2 for ~50% selectivity on uniform random data.
//! Validates exact set equality (sort both GPU and CPU results, compare).

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, read_buffer_slice, BenchTimer, CompactParams,
    MetalContext, PsoCache, ScanParams,
};

use crate::cpu_baselines::rayon_filter;
use crate::data_gen::DataGenerator;

use super::Experiment;

/// Number of elements each scan threadgroup processes (256 threads * 4 elements).
const SCAN_ELEMENTS_PER_TG: usize = 1024;

/// Maximum partials that can be scanned on GPU in a single threadgroup.
const MAX_GPU_PARTIALS: usize = 1024;

/// Stream compaction experiment comparing GPU scan-based compact vs rayon filter.
pub struct CompactExperiment {
    /// Input data kept for CPU baseline and validation.
    data: Vec<u32>,
    /// Threshold for the predicate (value > threshold).
    threshold: u32,
    /// Metal buffer holding input data.
    input_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for flags (0/1 per element).
    flags_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for scan output (exclusive prefix scan of flags).
    scan_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for threadgroup partial sums (used by scan passes).
    partials_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for compacted output.
    output_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for CompactParams.
    compact_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for ScanParams (full element count).
    scan_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for ScanParams (partials element count).
    partials_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU result from last run.
    gpu_result: Vec<u32>,
    /// CPU result from last run.
    cpu_result: Vec<u32>,
    /// Current element count.
    size: usize,
    /// Number of scan threadgroups for current size.
    num_scan_tgs: usize,
    /// Number of flag threadgroups for compact_flags dispatch.
    num_flag_tgs: usize,
}

impl CompactExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            threshold: 0,
            input_buffer: None,
            flags_buffer: None,
            scan_buffer: None,
            partials_buffer: None,
            output_buffer: None,
            compact_params_buffer: None,
            scan_params_buffer: None,
            partials_params_buffer: None,
            pso_cache: PsoCache::new(),
            gpu_result: Vec::new(),
            cpu_result: Vec::new(),
            size: 0,
            num_scan_tgs: 0,
            num_flag_tgs: 0,
        }
    }
}

impl Experiment for CompactExperiment {
    fn name(&self) -> &str {
        "compact"
    }

    fn description(&self) -> &str {
        "Stream compaction (u32): scan-based GPU compact vs rayon parallel filter"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![
            100_000,     // 100K
            1_000_000,   // 1M
            10_000_000,  // 10M
            100_000_000, // 100M
        ]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;
        self.data = gen.uniform_u32(size);

        // Use u32::MAX / 2 as threshold for ~50% selectivity on uniform random data
        self.threshold = u32::MAX / 2;

        // Threadgroup counts
        self.num_flag_tgs = size.div_ceil(256);
        self.num_scan_tgs = size.div_ceil(SCAN_ELEMENTS_PER_TG);

        // Input buffer
        self.input_buffer = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Flags buffer: N x u32 (0 or 1)
        self.flags_buffer = Some(alloc_buffer(&ctx.device, size * std::mem::size_of::<u32>()));

        // Scan buffer: N x u32 (exclusive prefix scan of flags)
        self.scan_buffer = Some(alloc_buffer(&ctx.device, size * std::mem::size_of::<u32>()));

        // Partials buffer: num_scan_tgs x u32
        self.partials_buffer = Some(alloc_buffer(
            &ctx.device,
            self.num_scan_tgs * std::mem::size_of::<u32>(),
        ));

        // Output buffer: N x u32 (worst case all pass predicate)
        self.output_buffer = Some(alloc_buffer(&ctx.device, size * std::mem::size_of::<u32>()));

        // CompactParams buffer
        let compact_params = CompactParams {
            element_count: size as u32,
            threshold: self.threshold,
            _pad: [0; 2],
        };
        self.compact_params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[compact_params]));

        // ScanParams for full scan (element_count = size)
        let scan_params = ScanParams {
            element_count: size as u32,
            pass: 0,
            _pad: [0; 2],
        };
        self.scan_params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[scan_params]));

        // ScanParams for partials scan (element_count = num_scan_tgs)
        let partials_params = ScanParams {
            element_count: self.num_scan_tgs as u32,
            pass: 1,
            _pad: [0; 2],
        };
        self.partials_params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[partials_params]));

        // Pre-warm PSO cache
        self.pso_cache.get_or_create(ctx.library(), "compact_flags");
        self.pso_cache
            .get_or_create(ctx.library(), "compact_scatter");
        self.pso_cache.get_or_create(ctx.library(), "scan_local");
        self.pso_cache.get_or_create(ctx.library(), "scan_partials");
        self.pso_cache
            .get_or_create(ctx.library(), "scan_add_offsets");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.input_buffer.as_ref().expect("setup not called");
        let flags = self.flags_buffer.as_ref().expect("setup not called");
        let scan_out = self.scan_buffer.as_ref().expect("setup not called");
        let partials = self.partials_buffer.as_ref().expect("setup not called");
        let output = self.output_buffer.as_ref().expect("setup not called");
        let compact_params = self
            .compact_params_buffer
            .as_ref()
            .expect("setup not called");
        let scan_params = self.scan_params_buffer.as_ref().expect("setup not called");
        let partials_params = self
            .partials_params_buffer
            .as_ref()
            .expect("setup not called");

        let timer = BenchTimer::start();

        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");

        // --- Dispatch 1: compact_flags ---
        {
            let pso = self.pso_cache.get_or_create(ctx.library(), "compact_flags");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(input.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(flags.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(compact_params.as_ref()), 0, 2);
            }

            let grid = objc2_metal::MTLSize {
                width: self.num_flag_tgs,
                height: 1,
                depth: 1,
            };
            let tg = objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
        }

        // --- Dispatch 2: scan_local on flags ---
        {
            let pso = self.pso_cache.get_or_create(ctx.library(), "scan_local");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(flags.as_ref()), 0, 0); // input = flags
                encoder.setBuffer_offset_atIndex(Some(scan_out.as_ref()), 0, 1); // output = scan
                encoder.setBuffer_offset_atIndex(Some(partials.as_ref()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(scan_params.as_ref()), 0, 3);
            }

            let grid = objc2_metal::MTLSize {
                width: self.num_scan_tgs,
                height: 1,
                depth: 1,
            };
            let tg = objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
        }

        // --- Dispatch 3: scan partials ---
        if self.num_scan_tgs <= MAX_GPU_PARTIALS {
            // GPU scan of partials (single threadgroup)
            let pso = self.pso_cache.get_or_create(ctx.library(), "scan_partials");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(partials.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(partials_params.as_ref()), 0, 1);
            }

            let grid = objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            let tg = objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
        } else {
            // Too many partials for single threadgroup: commit, CPU scan, continue
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            // CPU scan of partials
            let partials_data: Vec<u32> =
                unsafe { read_buffer_slice(partials.as_ref(), self.num_scan_tgs) };
            let scanned =
                crate::cpu_baselines::sequential::sequential_exclusive_scan(&partials_data);
            unsafe {
                let ptr = partials.contents().as_ptr() as *mut u32;
                std::ptr::copy_nonoverlapping(scanned.as_ptr(), ptr, self.num_scan_tgs);
            }

            // New command buffer for remaining dispatches
            let cmd_buf2 = ctx
                .queue
                .commandBuffer()
                .expect("Failed to create command buffer");

            // --- Dispatch 4: scan_add_offsets ---
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "scan_add_offsets");
                let encoder = cmd_buf2
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");

                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(scan_out.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(partials.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(scan_params.as_ref()), 0, 2);
                }

                let grid = objc2_metal::MTLSize {
                    width: self.num_scan_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            // --- Dispatch 5: compact_scatter ---
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "compact_scatter");
                let encoder = cmd_buf2
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");

                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(input.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(flags.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(scan_out.as_ref()), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(output.as_ref()), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(compact_params.as_ref()), 0, 4);
                }

                let grid = objc2_metal::MTLSize {
                    width: self.num_flag_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            cmd_buf2.commit();
            cmd_buf2.waitUntilCompleted();

            let elapsed = timer.stop();

            // Determine output count: scan[N-1] + flags[N-1]
            let last_scan: u32 =
                unsafe { read_buffer_slice(scan_out.as_ref(), self.size) }[self.size - 1];
            let last_flag: u32 =
                unsafe { read_buffer_slice(flags.as_ref(), self.size) }[self.size - 1];
            let output_count = (last_scan + last_flag) as usize;

            self.gpu_result = unsafe { read_buffer_slice(output.as_ref(), output_count) };
            return elapsed;
        }

        // --- Dispatch 4: scan_add_offsets (GPU partials path) ---
        {
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "scan_add_offsets");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(scan_out.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(partials.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(scan_params.as_ref()), 0, 2);
            }

            let grid = objc2_metal::MTLSize {
                width: self.num_scan_tgs,
                height: 1,
                depth: 1,
            };
            let tg = objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
        }

        // --- Dispatch 5: compact_scatter ---
        {
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "compact_scatter");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(input.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(flags.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(scan_out.as_ref()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(output.as_ref()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(compact_params.as_ref()), 0, 4);
            }

            let grid = objc2_metal::MTLSize {
                width: self.num_flag_tgs,
                height: 1,
                depth: 1,
            };
            let tg = objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
        }

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let elapsed = timer.stop();

        // Determine output count: scan[N-1] + flags[N-1]
        let last_scan: u32 =
            unsafe { read_buffer_slice(scan_out.as_ref(), self.size) }[self.size - 1];
        let last_flag: u32 = unsafe { read_buffer_slice(flags.as_ref(), self.size) }[self.size - 1];
        let output_count = (last_scan + last_flag) as usize;

        self.gpu_result = unsafe { read_buffer_slice(output.as_ref(), output_count) };

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();
        self.cpu_result = rayon_filter::par_filter_gt(&self.data, self.threshold);
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // Sort both for set equality comparison
        let mut gpu_sorted = self.gpu_result.clone();
        gpu_sorted.sort();
        let mut cpu_sorted = self.cpu_result.clone();
        cpu_sorted.sort();

        if gpu_sorted.len() != cpu_sorted.len() {
            return Err(format!(
                "Count mismatch: GPU selected {} elements, CPU selected {} elements (threshold={})",
                gpu_sorted.len(),
                cpu_sorted.len(),
                self.threshold
            ));
        }

        for (i, (&gpu, &cpu)) in gpu_sorted.iter().zip(cpu_sorted.iter()).enumerate() {
            if gpu != cpu {
                let start = i.saturating_sub(3);
                let end = (i + 4).min(gpu_sorted.len());
                let gpu_ctx: Vec<u32> = gpu_sorted[start..end].to_vec();
                let cpu_ctx: Vec<u32> = cpu_sorted[start..end].to_vec();
                return Err(format!(
                    "Sorted mismatch at index {}: GPU={} CPU={}\n  GPU[{}..{}]: {:?}\n  CPU[{}..{}]: {:?}",
                    i, gpu, cpu, start, end, gpu_ctx, start, end, cpu_ctx
                ));
            }
        }

        Ok(())
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let seconds = elapsed_ms / 1000.0;
        let elements_per_sec = if seconds > 0.0 {
            size as f64 / seconds
        } else {
            0.0
        };
        m.insert("elements_per_sec".to_string(), elements_per_sec);
        m.insert("elements".to_string(), size as f64);
        m.insert("threshold".to_string(), self.threshold as f64);

        let selectivity = if !self.data.is_empty() {
            self.gpu_result.len() as f64 / self.data.len() as f64 * 100.0
        } else {
            0.0
        };
        m.insert("selectivity_pct".to_string(), selectivity);
        m.insert("output_count".to_string(), self.gpu_result.len() as f64);

        // Bandwidth: read input(N*4) + write flags(N*4) + read/write scan(N*4*2) + read+scatter(~selectivity*N*4)
        let bytes = size as f64 * 4.0 * 5.0; // approximate: 5 passes over N elements
        let gbs = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
