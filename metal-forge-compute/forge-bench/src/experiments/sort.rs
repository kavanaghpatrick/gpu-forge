//! Radix sort experiment: GPU 4-bit radix sort vs CPU sort baselines.
//!
//! Implements a reduce-then-scan radix sort using 8 passes (4 bits per pass)
//! for u32 keys. Each pass: histogram -> prefix scan -> scatter.
//! Double-buffers with ping-pong between passes.
//!
//! Single command buffer for all 8 passes with blit fill for histogram zeroing.
//! Multi-level GPU scan for histogram prefix sums (handles up to 100M+ elements).
//!
//! CPU baselines: std::sort_unstable (single-threaded) and rayon par_sort_unstable.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, read_buffer_slice, BenchTimer, GpuTimer, MetalContext,
    PsoCache, ScanParams, SortParams,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// Threads per threadgroup for sort kernels.
const SORT_TG_SIZE: usize = 256;

/// Number of radix bits per pass.
const RADIX_BITS: usize = 4;

/// Number of bins per radix pass (2^4 = 16).
const RADIX_BINS: usize = 16;

/// Total passes for u32 (32 bits / 4 bits per pass = 8).
const NUM_PASSES: usize = 8;

/// Elements per threadgroup for scan (256 threads * 4 elements).
const SCAN_ELEMENTS_PER_TG: usize = 1024;

/// Maximum partials for GPU scan (256 threads * 4 elements/thread).
const MAX_GPU_PARTIALS: usize = 1024;

/// Radix sort experiment comparing GPU radix sort vs CPU baselines.
pub struct SortExperiment {
    /// Input data kept for CPU baseline and validation.
    data: Vec<u32>,
    /// Metal buffer A for keys (ping).
    keys_a: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer B for keys (pong).
    keys_b: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Global histogram buffer (num_tg * 16 elements).
    histogram_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Scanned histogram output buffer.
    scanned_histogram_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Partials buffer for scan of histogram.
    scan_partials_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Pre-allocated SortParams buffers for each pass (8 total).
    sort_params_buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Pre-allocated ScanParams buffers for each pass (8 total).
    scan_params_buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Pre-allocated ScanParams buffers for scan_partials (8 total).
    partials_params_buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    // --- Multi-level scan buffers (for large histogram sizes) ---
    /// Buffer for locally-scanned L1 partials.
    l1_scanned_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Buffer for L2 partial sums.
    l2_partials_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// ScanParams for scan_local on L1 partials.
    l1_scan_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// ScanParams for scan_partials on L2.
    l2_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// ScanParams for scan_add_offsets on L1 scanned.
    l1_addoffsets_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU result from last run.
    gpu_result: Vec<u32>,
    /// CPU result from last run (std::sort_unstable).
    cpu_result: Vec<u32>,
    /// Current element count.
    size: usize,
    /// Number of threadgroups for sort kernels.
    num_sort_tgs: usize,
    /// Total histogram size (num_sort_tgs * 16).
    histogram_size: usize,
}

impl SortExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            keys_a: None,
            keys_b: None,
            histogram_buffer: None,
            scanned_histogram_buffer: None,
            scan_partials_buffer: None,
            sort_params_buffers: Vec::new(),
            scan_params_buffers: Vec::new(),
            partials_params_buffers: Vec::new(),
            l1_scanned_buffer: None,
            l2_partials_buffer: None,
            l1_scan_params_buffer: None,
            l2_params_buffer: None,
            l1_addoffsets_params_buffer: None,
            pso_cache: PsoCache::new(),
            gpu_result: Vec::new(),
            cpu_result: Vec::new(),
            size: 0,
            num_sort_tgs: 0,
            histogram_size: 0,
        }
    }
}

impl Experiment for SortExperiment {
    fn name(&self) -> &str {
        "sort"
    }

    fn description(&self) -> &str {
        "Radix sort (u32): 4-bit, 8-pass reduce-then-scan with ping-pong"
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
        self.num_sort_tgs = size.div_ceil(SORT_TG_SIZE);
        self.histogram_size = self.num_sort_tgs * RADIX_BINS;

        // Allocate ping-pong key buffers
        self.keys_a = Some(alloc_buffer_with_data(&ctx.device, &self.data));
        self.keys_b = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));

        // Histogram buffer: num_tg * 16 elements
        self.histogram_buffer = Some(alloc_buffer(
            &ctx.device,
            self.histogram_size * std::mem::size_of::<u32>(),
        ));

        // Scanned histogram output buffer (same size)
        self.scanned_histogram_buffer = Some(alloc_buffer(
            &ctx.device,
            self.histogram_size * std::mem::size_of::<u32>(),
        ));

        // Partials buffer for scan: ceil(histogram_size / SCAN_ELEMENTS_PER_TG) elements
        let scan_tgs = self.histogram_size.div_ceil(SCAN_ELEMENTS_PER_TG);
        self.scan_partials_buffer = Some(alloc_buffer(
            &ctx.device,
            scan_tgs.max(1) * std::mem::size_of::<u32>(),
        ));

        // Pre-allocate 8 SortParams buffers (one per pass)
        self.sort_params_buffers.clear();
        for pass in 0..NUM_PASSES {
            let sort_params = SortParams {
                element_count: size as u32,
                bit_offset: (pass * RADIX_BITS) as u32,
                num_threadgroups: self.num_sort_tgs as u32,
                _pad: 0,
            };
            self.sort_params_buffers
                .push(alloc_buffer_with_data(&ctx.device, &[sort_params]));
        }

        // Pre-allocate 8 ScanParams buffers (one per pass, for scan_local + scan_add_offsets)
        self.scan_params_buffers.clear();
        for _pass in 0..NUM_PASSES {
            let scan_params = ScanParams {
                element_count: self.histogram_size as u32,
                pass: 0,
                _pad: [0; 2],
            };
            self.scan_params_buffers
                .push(alloc_buffer_with_data(&ctx.device, &[scan_params]));
        }

        // Pre-allocate 8 partials ScanParams buffers (for scan_partials)
        self.partials_params_buffers.clear();
        for _pass in 0..NUM_PASSES {
            let partials_params = ScanParams {
                element_count: scan_tgs as u32,
                pass: 1,
                _pad: [0; 2],
            };
            self.partials_params_buffers
                .push(alloc_buffer_with_data(&ctx.device, &[partials_params]));
        }

        // Multi-level scan buffers (when scan_tgs > MAX_GPU_PARTIALS)
        if scan_tgs > MAX_GPU_PARTIALS {
            let num_l2_groups = scan_tgs.div_ceil(SCAN_ELEMENTS_PER_TG);

            self.l1_scanned_buffer = Some(alloc_buffer(
                &ctx.device,
                scan_tgs * std::mem::size_of::<u32>(),
            ));
            self.l2_partials_buffer = Some(alloc_buffer(
                &ctx.device,
                num_l2_groups * std::mem::size_of::<u32>(),
            ));

            // ScanParams for scan_local on L1 partials (element_count = scan_tgs)
            let l1_scan_params = ScanParams {
                element_count: scan_tgs as u32,
                pass: 0,
                _pad: [0; 2],
            };
            self.l1_scan_params_buffer =
                Some(alloc_buffer_with_data(&ctx.device, &[l1_scan_params]));

            // ScanParams for scan_partials on L2
            let l2_params = ScanParams {
                element_count: num_l2_groups as u32,
                pass: 2,
                _pad: [0; 2],
            };
            self.l2_params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[l2_params]));

            // ScanParams for scan_add_offsets on L1 scanned
            let l1_addoffsets_params = ScanParams {
                element_count: scan_tgs as u32,
                pass: 3,
                _pad: [0; 2],
            };
            self.l1_addoffsets_params_buffer =
                Some(alloc_buffer_with_data(&ctx.device, &[l1_addoffsets_params]));
        } else {
            self.l1_scanned_buffer = None;
            self.l2_partials_buffer = None;
            self.l1_scan_params_buffer = None;
            self.l2_params_buffer = None;
            self.l1_addoffsets_params_buffer = None;
        }

        // Pre-warm PSO cache
        self.pso_cache
            .get_or_create(ctx.library(), "radix_histogram");
        self.pso_cache
            .get_or_create(ctx.library(), "radix_scatter");
        self.pso_cache
            .get_or_create(ctx.library(), "scan_local");
        self.pso_cache
            .get_or_create(ctx.library(), "scan_partials");
        self.pso_cache
            .get_or_create(ctx.library(), "scan_add_offsets");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        // Clone buffer references to avoid borrow conflicts with &mut self.
        let keys_a = self.keys_a.clone().expect("setup not called");
        let keys_b = self.keys_b.clone().expect("setup not called");
        let histogram_buf = self.histogram_buffer.clone().expect("setup not called");
        let scanned_buf = self.scanned_histogram_buffer.clone().expect("setup not called");
        let partials_buf = self.scan_partials_buffer.clone().expect("setup not called");

        let histogram_size = self.histogram_size;
        let num_sort_tgs = self.num_sort_tgs;
        let size = self.size;
        let scan_tgs = histogram_size.div_ceil(SCAN_ELEMENTS_PER_TG);

        // Re-upload input data to keys_a before each run
        unsafe {
            let ptr = keys_a.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(self.data.as_ptr(), ptr, size);
        }

        // Single command buffer for all 8 passes
        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");

        // 8 radix passes, ping-pong between keys_a and keys_b
        for pass in 0..NUM_PASSES {
            let (input_buf, output_buf) = if pass % 2 == 0 {
                (&keys_a, &keys_b)
            } else {
                (&keys_b, &keys_a)
            };

            let sort_params_buf = &self.sort_params_buffers[pass];
            let scan_params_buf = &self.scan_params_buffers[pass];
            let partials_params_buf = &self.partials_params_buffers[pass];

            // --- Blit fill: zero histogram buffer on GPU ---
            {
                let blit_encoder = cmd_buf
                    .blitCommandEncoder()
                    .expect("Failed to create blit encoder");
                blit_encoder.fillBuffer_range_value(
                    histogram_buf.as_ref(),
                    NSRange::new(0, histogram_size * std::mem::size_of::<u32>()),
                    0u8,
                );
                blit_encoder.endEncoding();
            }

            // --- Step 1: radix_histogram ---
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "radix_histogram");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");

                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(input_buf.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(histogram_buf.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(sort_params_buf.as_ref()), 0, 2);
                }

                let grid = objc2_metal::MTLSize {
                    width: num_sort_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: SORT_TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            // --- Step 2: Prefix scan of histogram ---
            // scan_local: histogram -> scanned_histogram + partials
            {
                let pso = self.pso_cache.get_or_create(ctx.library(), "scan_local");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");

                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(histogram_buf.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(scanned_buf.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(partials_buf.as_ref()), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(scan_params_buf.as_ref()), 0, 3);
                }

                let grid = objc2_metal::MTLSize {
                    width: scan_tgs,
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

            if scan_tgs <= MAX_GPU_PARTIALS {
                // Simple: scan_partials in single threadgroup
                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "scan_partials");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");

                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(partials_buf.as_ref()), 0, 0);
                        encoder
                            .setBuffer_offset_atIndex(Some(partials_params_buf.as_ref()), 0, 1);
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
                }

                // scan_add_offsets on scanned histogram
                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "scan_add_offsets");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");

                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(scanned_buf.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(partials_buf.as_ref()), 0, 1);
                        encoder
                            .setBuffer_offset_atIndex(Some(scan_params_buf.as_ref()), 0, 2);
                    }

                    let grid = objc2_metal::MTLSize {
                        width: scan_tgs,
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
            } else {
                // Multi-level scan for large histogram sizes.
                // 3-level: scan_local(L1) -> scan_partials(L2) -> scan_add_offsets(L1)
                let l1_scanned = self.l1_scanned_buffer.as_ref().expect("setup not called");
                let l2_partials = self.l2_partials_buffer.as_ref().expect("setup not called");
                let l1_scan_params = self
                    .l1_scan_params_buffer
                    .as_ref()
                    .expect("setup not called");
                let l2_params = self.l2_params_buffer.as_ref().expect("setup not called");
                let l1_addoffsets_params = self
                    .l1_addoffsets_params_buffer
                    .as_ref()
                    .expect("setup not called");

                let num_l2_groups = scan_tgs.div_ceil(SCAN_ELEMENTS_PER_TG);

                // Pass 2a: scan_local on L1 partials
                {
                    let pso = self.pso_cache.get_or_create(ctx.library(), "scan_local");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");

                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder
                            .setBuffer_offset_atIndex(Some(partials_buf.as_ref()), 0, 0);
                        encoder
                            .setBuffer_offset_atIndex(Some(l1_scanned.as_ref()), 0, 1);
                        encoder
                            .setBuffer_offset_atIndex(Some(l2_partials.as_ref()), 0, 2);
                        encoder
                            .setBuffer_offset_atIndex(Some(l1_scan_params.as_ref()), 0, 3);
                    }

                    let grid = objc2_metal::MTLSize {
                        width: num_l2_groups,
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

                // Pass 2b: scan_partials on L2
                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "scan_partials");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");

                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder
                            .setBuffer_offset_atIndex(Some(l2_partials.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(l2_params.as_ref()), 0, 1);
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
                }

                // Pass 2c: scan_add_offsets on L1 scanned using L2 partials
                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "scan_add_offsets");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");

                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder
                            .setBuffer_offset_atIndex(Some(l1_scanned.as_ref()), 0, 0);
                        encoder
                            .setBuffer_offset_atIndex(Some(l2_partials.as_ref()), 0, 1);
                        encoder.setBuffer_offset_atIndex(
                            Some(l1_addoffsets_params.as_ref()),
                            0,
                            2,
                        );
                    }

                    let grid = objc2_metal::MTLSize {
                        width: num_l2_groups,
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

                // scan_add_offsets on scanned histogram using l1_scanned as partials
                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "scan_add_offsets");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");

                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder
                            .setBuffer_offset_atIndex(Some(scanned_buf.as_ref()), 0, 0);
                        encoder
                            .setBuffer_offset_atIndex(Some(l1_scanned.as_ref()), 0, 1);
                        encoder
                            .setBuffer_offset_atIndex(Some(scan_params_buf.as_ref()), 0, 2);
                    }

                    let grid = objc2_metal::MTLSize {
                        width: scan_tgs,
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
            }

            // --- Step 3: radix_scatter ---
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "radix_scatter");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");

                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(input_buf.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(output_buf.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(scanned_buf.as_ref()), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(sort_params_buf.as_ref()), 0, 3);
                }

                let grid = objc2_metal::MTLSize {
                    width: num_sort_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: SORT_TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }
        }

        // Single commit + wait for all 8 passes
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let elapsed = GpuTimer::elapsed_ms(&cmd_buf).unwrap_or(0.0);

        // After 8 passes (even number), result is in keys_a
        self.gpu_result = unsafe { read_buffer_slice(keys_a.as_ref(), size) };

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();
        self.cpu_result = self.data.clone();
        self.cpu_result.sort_unstable();
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_result.len() != self.cpu_result.len() {
            return Err(format!(
                "Length mismatch: GPU={} CPU={}",
                self.gpu_result.len(),
                self.cpu_result.len()
            ));
        }

        for (i, (&gpu, &cpu)) in self
            .gpu_result
            .iter()
            .zip(self.cpu_result.iter())
            .enumerate()
        {
            if gpu != cpu {
                // Show surrounding context for debugging
                let start = i.saturating_sub(3);
                let end = (i + 4).min(self.gpu_result.len());
                let gpu_ctx: Vec<u32> = self.gpu_result[start..end].to_vec();
                let cpu_ctx: Vec<u32> = self.cpu_result[start..end].to_vec();
                return Err(format!(
                    "Mismatch at index {}: GPU={} CPU={}\n  GPU[{}..{}]: {:?}\n  CPU[{}..{}]: {:?}",
                    i, gpu, cpu, start, end, gpu_ctx, start, end, cpu_ctx
                ));
            }
        }

        Ok(())
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let seconds = elapsed_ms / 1000.0;

        // Elements per second
        let elements_per_sec = if seconds > 0.0 {
            size as f64 / seconds
        } else {
            0.0
        };
        m.insert("elements_per_sec".to_string(), elements_per_sec);

        // Total bytes moved: 8 passes * 2 (read + write) * N * 4 bytes
        let bytes = (NUM_PASSES as f64) * 2.0 * (size as f64) * 4.0;
        let gbs = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);
        m.insert("bytes_processed".to_string(), bytes);
        m.insert("elements".to_string(), size as f64);
        m.insert("num_passes".to_string(), NUM_PASSES as f64);
        m.insert("threadgroups".to_string(), self.num_sort_tgs as f64);

        m
    }
}
