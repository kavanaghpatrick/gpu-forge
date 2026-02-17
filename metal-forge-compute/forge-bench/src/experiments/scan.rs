//! Prefix scan experiment: GPU multi-pass SIMD scan vs CPU sequential scan.
//!
//! Tests exclusive prefix scan on u32 data. The GPU implementation uses
//! a multi-pass SIMD prefix scan approach:
//!   1. scan_local: per-threadgroup SIMD prefix scan + write partial sums
//!   2. scan_partials: scan the partials array (single TG, up to 1024 partials)
//!      OR multi-level: scan_local on partials -> scan_partials on L2 -> add offsets
//!   3. scan_add_offsets: add scanned partials to each threadgroup's elements
//!
//! At 10M elements with 1024 elem/TG = 9766 partials. Since 9766 > 1024,
//! a 3-level scan is used: 9766 -> 10 -> 1 via scan_local + scan_partials.
//!
//! Validates exact match between GPU and CPU results.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, read_buffer_slice, BenchTimer, GpuTimer,
    MetalContext, PsoCache, ScanParams,
};

use crate::cpu_baselines::sequential;
use crate::data_gen::DataGenerator;

use super::Experiment;

/// Number of elements each threadgroup processes (256 threads * 4 elements).
const ELEMENTS_PER_TG: usize = 1024;

/// Maximum partials that can be scanned on GPU in a single threadgroup
/// (256 threads * 4 elements/thread).
const MAX_GPU_PARTIALS: usize = 1024;

/// Prefix scan experiment comparing GPU multi-pass scan vs sequential CPU scan.
pub struct ScanExperiment {
    /// Input data kept for CPU baseline and validation.
    data: Vec<u32>,
    /// Metal buffer holding the input data.
    input_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for scan output.
    output_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for threadgroup partial sums (level 1).
    partials_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for ScanParams (used by scan_local and scan_add_offsets).
    params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for ScanParams (used by scan_partials, element_count = num_threadgroups).
    partials_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    // --- Multi-level scan buffers (only used when num_threadgroups > MAX_GPU_PARTIALS) ---
    /// Metal buffer for scanned L1 partials (output of scan_local on partials).
    l1_scanned_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for level-2 partial sums from scanning L1 partials.
    l2_partials_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// ScanParams for scan_partials on L2 partials (element_count = num_l2_groups).
    l2_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// ScanParams for scan_add_offsets on L1 scanned (element_count = num_threadgroups).
    l1_addoffsets_params_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU result from last run.
    gpu_result: Vec<u32>,
    /// CPU result from last run.
    cpu_result: Vec<u32>,
    /// Current element count.
    size: usize,
    /// Number of threadgroups for current size.
    num_threadgroups: usize,
}

impl ScanExperiment {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            input_buffer: None,
            output_buffer: None,
            partials_buffer: None,
            params_buffer: None,
            partials_params_buffer: None,
            l1_scanned_buffer: None,
            l2_partials_buffer: None,
            l2_params_buffer: None,
            l1_addoffsets_params_buffer: None,
            pso_cache: PsoCache::new(),
            gpu_result: Vec::new(),
            cpu_result: Vec::new(),
            size: 0,
            num_threadgroups: 0,
        }
    }
}

impl Experiment for ScanExperiment {
    fn name(&self) -> &str {
        "scan"
    }

    fn description(&self) -> &str {
        "Exclusive prefix scan (u32): SIMD prefix reduce-then-scan"
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
        // Use small values to avoid u32 overflow in prefix scan
        // Max value per element: prefix sum at last position would be ~size * max_val
        // For 100M elements with max_val=10: ~1B which fits in u32 (max 4.29B)
        self.data = gen.uniform_u32(size);
        // Clamp values to avoid overflow: for 100M elements, keep values < 40
        let max_val = if size >= 100_000_000 {
            40u32
        } else if size >= 10_000_000 {
            400u32
        } else {
            4000u32
        };
        for v in &mut self.data {
            *v = *v % max_val;
        }

        self.num_threadgroups = size.div_ceil(ELEMENTS_PER_TG);

        // Input buffer: N x u32
        self.input_buffer = Some(alloc_buffer_with_data(&ctx.device, &self.data));

        // Output buffer: N x u32
        self.output_buffer = Some(alloc_buffer(&ctx.device, size * std::mem::size_of::<u32>()));

        // Partials buffer: num_threadgroups x u32
        self.partials_buffer = Some(alloc_buffer(
            &ctx.device,
            self.num_threadgroups * std::mem::size_of::<u32>(),
        ));

        // Params buffer for scan_local and scan_add_offsets
        let params = ScanParams {
            element_count: size as u32,
            pass: 0,
            _pad: [0; 2],
        };
        self.params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Params buffer for scan_partials (element_count = num_threadgroups)
        let partials_params = ScanParams {
            element_count: self.num_threadgroups as u32,
            pass: 1,
            _pad: [0; 2],
        };
        self.partials_params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[partials_params]));

        // Multi-level scan buffers (only needed when num_threadgroups > MAX_GPU_PARTIALS)
        if self.num_threadgroups > MAX_GPU_PARTIALS {
            let num_l2_groups = self.num_threadgroups.div_ceil(ELEMENTS_PER_TG);

            // Buffer for locally-scanned L1 partials (output of scan_local on partials)
            self.l1_scanned_buffer = Some(alloc_buffer(
                &ctx.device,
                self.num_threadgroups * std::mem::size_of::<u32>(),
            ));

            // Buffer for L2 partial sums
            self.l2_partials_buffer = Some(alloc_buffer(
                &ctx.device,
                num_l2_groups * std::mem::size_of::<u32>(),
            ));

            // ScanParams for scan_partials on L2 (element_count = num_l2_groups)
            let l2_params = ScanParams {
                element_count: num_l2_groups as u32,
                pass: 2,
                _pad: [0; 2],
            };
            self.l2_params_buffer = Some(alloc_buffer_with_data(&ctx.device, &[l2_params]));

            // ScanParams for scan_add_offsets on L1 scanned (element_count = num_threadgroups)
            let l1_addoffsets_params = ScanParams {
                element_count: self.num_threadgroups as u32,
                pass: 3,
                _pad: [0; 2],
            };
            self.l1_addoffsets_params_buffer =
                Some(alloc_buffer_with_data(&ctx.device, &[l1_addoffsets_params]));
        } else {
            self.l1_scanned_buffer = None;
            self.l2_partials_buffer = None;
            self.l2_params_buffer = None;
            self.l1_addoffsets_params_buffer = None;
        }

        // Pre-warm PSO cache
        self.pso_cache
            .get_or_create(ctx.library(), "scan_local");
        self.pso_cache
            .get_or_create(ctx.library(), "scan_partials");
        self.pso_cache
            .get_or_create(ctx.library(), "scan_add_offsets");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let input = self.input_buffer.as_ref().expect("setup not called");
        let output = self.output_buffer.as_ref().expect("setup not called");
        let partials = self.partials_buffer.as_ref().expect("setup not called");
        let params = self.params_buffer.as_ref().expect("setup not called");
        let partials_params = self
            .partials_params_buffer
            .as_ref()
            .expect("setup not called");

        // Create single command buffer for all passes
        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");

        // --- Pass 1: scan_local on input data ---
        {
            let pso = self.pso_cache.get_or_create(ctx.library(), "scan_local");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(input.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(output.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(partials.as_ref()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(params.as_ref()), 0, 3);
            }

            let grid = objc2_metal::MTLSize {
                width: self.num_threadgroups,
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

        // --- Pass 2: scan the partials ---
        // After this block, the buffer used for the final scan_add_offsets
        // contains the fully-scanned partials.
        let final_partials: &ProtocolObject<dyn MTLBuffer>;

        if self.num_threadgroups <= MAX_GPU_PARTIALS {
            // Simple case: scan_partials in-place (single threadgroup)
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

            final_partials = partials.as_ref();
        } else {
            // Multi-level scan for large partials count.
            // 3-level: scan_local(L1) -> scan_partials(L2) -> scan_add_offsets(L1)
            let l1_scanned = self.l1_scanned_buffer.as_ref().expect("setup not called");
            let l2_partials = self.l2_partials_buffer.as_ref().expect("setup not called");
            let l2_params = self.l2_params_buffer.as_ref().expect("setup not called");
            let l1_addoffsets_params = self
                .l1_addoffsets_params_buffer
                .as_ref()
                .expect("setup not called");

            let num_l2_groups = self.num_threadgroups.div_ceil(ELEMENTS_PER_TG);

            // Pass 2a: scan_local on L1 partials -> locally-scanned L1 + L2 partials
            {
                let pso = self.pso_cache.get_or_create(ctx.library(), "scan_local");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");

                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(partials.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(l1_scanned.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(l2_partials.as_ref()), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(partials_params.as_ref()), 0, 3);
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

            // Pass 2b: scan_partials on L2 partials (in-place, single TG)
            {
                let pso = self.pso_cache.get_or_create(ctx.library(), "scan_partials");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");

                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(l2_partials.as_ref()), 0, 0);
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
                    encoder.setBuffer_offset_atIndex(Some(l1_scanned.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(l2_partials.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(l1_addoffsets_params.as_ref()), 0, 2);
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

            // l1_scanned now contains the fully-scanned L1 partials
            final_partials = l1_scanned.as_ref();
        }

        // --- Pass 3: scan_add_offsets on main output ---
        {
            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "scan_add_offsets");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(output.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(final_partials), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(params.as_ref()), 0, 2);
            }

            let grid = objc2_metal::MTLSize {
                width: self.num_threadgroups,
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

        let elapsed = GpuTimer::elapsed_ms(&cmd_buf).unwrap_or(0.0);

        // Read back result
        self.gpu_result = unsafe { read_buffer_slice(output.as_ref(), self.size) };

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();
        self.cpu_result = sequential::sequential_exclusive_scan(&self.data);
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

        for (i, (&gpu, &cpu)) in self.gpu_result.iter().zip(self.cpu_result.iter()).enumerate() {
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
        // Scan reads N elements and writes N elements = 2 * N * 4 bytes
        let bytes = size as f64 * 4.0 * 2.0;
        let seconds = elapsed_ms / 1000.0;
        let gbs = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);
        m.insert("bytes_processed".to_string(), bytes);
        m.insert("elements".to_string(), size as f64);
        m.insert("threadgroups".to_string(), self.num_threadgroups as f64);

        m
    }
}
