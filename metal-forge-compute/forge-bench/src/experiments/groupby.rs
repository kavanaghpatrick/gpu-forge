//! Group-by aggregate experiment: GPU sort-based group-by vs CPU HashMap.
//!
//! Pipeline: radix sort keys (with paired values) -> boundary detect ->
//! compute group offsets on CPU -> segmented reduce per group.
//!
//! CPU baseline: HashMap<u32, (sum, count, min, max)> in single pass.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, read_buffer_slice, BenchTimer,
    GroupByParams, MetalContext, PsoCache, ScanParams, SortParams,
};

use crate::cpu_baselines::hashmap_ops;
use crate::data_gen::DataGenerator;

use super::Experiment;

/// Threads per threadgroup for sort and groupby kernels.
const TG_SIZE: usize = 256;

/// Radix sort constants (same as sort experiment).
const RADIX_BITS: usize = 4;
const RADIX_BINS: usize = 16;
const NUM_PASSES: usize = 8;
const SCAN_ELEMENTS_PER_TG: usize = 512;
const MAX_GPU_PARTIALS: usize = 512;

/// Number of distinct group keys to generate.
/// Uses 1000 groups -- reasonable cardinality for benchmarking.
const NUM_GROUPS: u32 = 1000;

/// Group-by aggregate experiment comparing GPU sort-based approach vs CPU HashMap.
pub struct GroupByExperiment {
    /// Key column (unsorted).
    keys: Vec<u32>,
    /// Value column (f32).
    values: Vec<f32>,
    /// Sorted keys after GPU radix sort (for validation readback).
    sorted_keys_gpu: Vec<u32>,
    /// Sorted values after GPU sort (for validation readback).
    sorted_values_gpu: Vec<f32>,
    /// GPU aggregate results: per-group (sum, count, min, max).
    gpu_agg: Vec<(f64, u32, f32, f32)>,
    /// CPU aggregate results: per-group (sum, count, min, max).
    cpu_agg: HashMap<u32, hashmap_ops::GroupAgg>,
    /// Number of groups detected on GPU.
    gpu_num_groups: usize,
    /// Current element count.
    size: usize,
    /// Number of groups target.
    num_groups: u32,

    // === Sort buffers (reuse radix sort pipeline) ===
    /// Metal buffer A for keys (ping).
    keys_a: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer B for keys (pong).
    keys_b: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer A for values (ping, paired with keys).
    values_a: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer B for values (pong).
    values_b: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Global histogram buffer (num_tg * 16 elements).
    histogram_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Scanned histogram output buffer.
    scanned_histogram_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Partials buffer for scan of histogram.
    scan_partials_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    // === Group-by buffers ===
    /// Boundary flags buffer (N elements).
    flags_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Group offsets buffer (num_groups elements, computed on CPU from scanned flags).
    group_offsets_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Aggregate output: sum per group.
    agg_sum_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Aggregate output: count per group.
    agg_count_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Aggregate output: min per group.
    agg_min_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Aggregate output: max per group.
    agg_max_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// Number of threadgroups for sort.
    num_sort_tgs: usize,
    /// Histogram size (num_sort_tgs * 16).
    histogram_size: usize,
}

impl GroupByExperiment {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            sorted_keys_gpu: Vec::new(),
            sorted_values_gpu: Vec::new(),
            gpu_agg: Vec::new(),
            cpu_agg: HashMap::new(),
            gpu_num_groups: 0,
            size: 0,
            num_groups: NUM_GROUPS,
            keys_a: None,
            keys_b: None,
            values_a: None,
            values_b: None,
            histogram_buffer: None,
            scanned_histogram_buffer: None,
            scan_partials_buffer: None,
            flags_buffer: None,
            group_offsets_buffer: None,
            agg_sum_buffer: None,
            agg_count_buffer: None,
            agg_min_buffer: None,
            agg_max_buffer: None,
            pso_cache: PsoCache::new(),
            num_sort_tgs: 0,
            histogram_size: 0,
        }
    }
}

impl Experiment for GroupByExperiment {
    fn name(&self) -> &str {
        "groupby"
    }

    fn description(&self) -> &str {
        "Sort-based group-by aggregate: GPU radix sort + boundary detect + segmented reduce vs CPU HashMap"
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

        // Generate key column with target cardinality
        // Keys are random u32 values modulo num_groups
        let raw_keys = gen.uniform_u32(size);
        self.keys = raw_keys
            .iter()
            .map(|&k| k % self.num_groups)
            .collect();

        // Generate value column (f32 in [0, 1000))
        self.values = gen
            .uniform_f32(size)
            .iter()
            .map(|&v| v * 1000.0)
            .collect();

        self.num_sort_tgs = size.div_ceil(TG_SIZE);
        self.histogram_size = self.num_sort_tgs * RADIX_BINS;

        // === Allocate sort buffers ===
        self.keys_a = Some(alloc_buffer_with_data(&ctx.device, &self.keys));
        self.keys_b = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));
        self.values_a = Some(alloc_buffer_with_data(&ctx.device, &self.values));
        self.values_b = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<f32>(),
        ));
        self.histogram_buffer = Some(alloc_buffer(
            &ctx.device,
            self.histogram_size * std::mem::size_of::<u32>(),
        ));
        self.scanned_histogram_buffer = Some(alloc_buffer(
            &ctx.device,
            self.histogram_size * std::mem::size_of::<u32>(),
        ));
        let scan_tgs = self.histogram_size.div_ceil(SCAN_ELEMENTS_PER_TG);
        self.scan_partials_buffer = Some(alloc_buffer(
            &ctx.device,
            scan_tgs.max(1) * std::mem::size_of::<u32>(),
        ));

        // === Allocate group-by buffers ===
        self.flags_buffer = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));
        // Over-allocate group offsets to handle max possible groups
        let max_groups = size.min(self.num_groups as usize + 1);
        self.group_offsets_buffer = Some(alloc_buffer(
            &ctx.device,
            max_groups * std::mem::size_of::<u32>(),
        ));
        self.agg_sum_buffer = Some(alloc_buffer(
            &ctx.device,
            max_groups * std::mem::size_of::<f32>(),
        ));
        self.agg_count_buffer = Some(alloc_buffer(
            &ctx.device,
            max_groups * std::mem::size_of::<u32>(),
        ));
        self.agg_min_buffer = Some(alloc_buffer(
            &ctx.device,
            max_groups * std::mem::size_of::<f32>(),
        ));
        self.agg_max_buffer = Some(alloc_buffer(
            &ctx.device,
            max_groups * std::mem::size_of::<f32>(),
        ));

        // Pre-warm PSO cache
        self.pso_cache.get_or_create(ctx.library(), "radix_histogram");
        self.pso_cache.get_or_create(ctx.library(), "radix_scatter");
        self.pso_cache.get_or_create(ctx.library(), "scan_local");
        self.pso_cache.get_or_create(ctx.library(), "scan_partials");
        self.pso_cache.get_or_create(ctx.library(), "scan_add_offsets");
        self.pso_cache.get_or_create(ctx.library(), "groupby_boundary_detect");
        self.pso_cache.get_or_create(ctx.library(), "groupby_segmented_reduce");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let size = self.size;
        let num_sort_tgs = self.num_sort_tgs;
        let histogram_size = self.histogram_size;
        let scan_tgs = histogram_size.div_ceil(SCAN_ELEMENTS_PER_TG);

        // Clone buffer pointers to avoid borrow conflicts
        let keys_a = self.keys_a.clone().expect("setup not called");
        let keys_b = self.keys_b.clone().expect("setup not called");
        let values_a = self.values_a.clone().expect("setup not called");
        let values_b = self.values_b.clone().expect("setup not called");
        let histogram_buf = self.histogram_buffer.clone().expect("setup not called");
        let scanned_buf = self.scanned_histogram_buffer.clone().expect("setup not called");
        let partials_buf = self.scan_partials_buffer.clone().expect("setup not called");
        let flags_buf = self.flags_buffer.clone().expect("setup not called");
        let group_offsets_buf = self.group_offsets_buffer.clone().expect("setup not called");
        let agg_sum_buf = self.agg_sum_buffer.clone().expect("setup not called");
        let agg_count_buf = self.agg_count_buffer.clone().expect("setup not called");
        let agg_min_buf = self.agg_min_buffer.clone().expect("setup not called");
        let agg_max_buf = self.agg_max_buffer.clone().expect("setup not called");

        // Re-upload input data before each run
        unsafe {
            let ptr = keys_a.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(self.keys.as_ptr(), ptr, size);
            let vptr = values_a.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(self.values.as_ptr(), vptr, size);
        }

        let timer = BenchTimer::start();

        // === PHASE 1: Radix sort keys (keeping values paired) ===
        // We sort keys and apply the same permutation to values.
        // The radix sort sorts keys; we scatter values alongside using
        // the same scatter indices.
        for pass in 0..NUM_PASSES {
            let (key_in, key_out) = if pass % 2 == 0 {
                (&keys_a, &keys_b)
            } else {
                (&keys_b, &keys_a)
            };
            let (_val_in, _val_out) = if pass % 2 == 0 {
                (&values_a, &values_b)
            } else {
                (&values_b, &values_a)
            };

            let bit_offset = (pass * RADIX_BITS) as u32;

            // Zero histogram buffer
            unsafe {
                let ptr = histogram_buf.contents().as_ptr() as *mut u32;
                for i in 0..histogram_size {
                    *ptr.add(i) = 0;
                }
            }

            let sort_params = SortParams {
                element_count: size as u32,
                bit_offset,
                num_threadgroups: num_sort_tgs as u32,
                _pad: 0,
            };
            let sort_params_buf = alloc_buffer_with_data(&ctx.device, &[sort_params]);

            let scan_params = ScanParams {
                element_count: histogram_size as u32,
                pass: 0,
                _pad: [0; 2],
            };
            let scan_params_buf = alloc_buffer_with_data(&ctx.device, &[scan_params]);

            let cmd_buf = ctx
                .queue
                .commandBuffer()
                .expect("Failed to create command buffer");

            // Step 1: radix_histogram
            {
                let pso = self.pso_cache.get_or_create(ctx.library(), "radix_histogram");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(key_in.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(histogram_buf.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(sort_params_buf.as_ref()), 0, 2);
                }
                let grid = objc2_metal::MTLSize { width: num_sort_tgs, height: 1, depth: 1 };
                let tg = objc2_metal::MTLSize { width: TG_SIZE, height: 1, depth: 1 };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            // Step 2: scan histogram
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
                let grid = objc2_metal::MTLSize { width: scan_tgs, height: 1, depth: 1 };
                let tg = objc2_metal::MTLSize { width: 256, height: 1, depth: 1 };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            if scan_tgs <= MAX_GPU_PARTIALS {
                // GPU partials scan
                let partials_params = ScanParams {
                    element_count: scan_tgs as u32,
                    pass: 1,
                    _pad: [0; 2],
                };
                let partials_params_buf = alloc_buffer_with_data(&ctx.device, &[partials_params]);

                {
                    let pso = self.pso_cache.get_or_create(ctx.library(), "scan_partials");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");
                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(partials_buf.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(partials_params_buf.as_ref()), 0, 1);
                    }
                    let grid = objc2_metal::MTLSize { width: 1, height: 1, depth: 1 };
                    let tg = objc2_metal::MTLSize { width: 256, height: 1, depth: 1 };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    encoder.endEncoding();
                }

                // scan_add_offsets
                {
                    let pso = self.pso_cache.get_or_create(ctx.library(), "scan_add_offsets");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");
                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(scanned_buf.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(partials_buf.as_ref()), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(scan_params_buf.as_ref()), 0, 2);
                    }
                    let grid = objc2_metal::MTLSize { width: scan_tgs, height: 1, depth: 1 };
                    let tg = objc2_metal::MTLSize { width: 256, height: 1, depth: 1 };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    encoder.endEncoding();
                }

                // Step 3: radix_scatter (keys + values)
                {
                    let pso = self.pso_cache.get_or_create(ctx.library(), "radix_scatter");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");
                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(key_in.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(key_out.as_ref()), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(scanned_buf.as_ref()), 0, 2);
                        encoder.setBuffer_offset_atIndex(Some(sort_params_buf.as_ref()), 0, 3);
                    }
                    let grid = objc2_metal::MTLSize { width: num_sort_tgs, height: 1, depth: 1 };
                    let tg = objc2_metal::MTLSize { width: TG_SIZE, height: 1, depth: 1 };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    encoder.endEncoding();
                }

                cmd_buf.commit();
                cmd_buf.waitUntilCompleted();
            } else {
                // CPU fallback for partials scan
                cmd_buf.commit();
                cmd_buf.waitUntilCompleted();

                let partials_data: Vec<u32> =
                    unsafe { read_buffer_slice(partials_buf.as_ref(), scan_tgs) };
                let scanned_partials =
                    crate::cpu_baselines::sequential::sequential_exclusive_scan(&partials_data);
                unsafe {
                    let ptr = partials_buf.contents().as_ptr() as *mut u32;
                    std::ptr::copy_nonoverlapping(scanned_partials.as_ptr(), ptr, scan_tgs);
                }

                let cmd_buf2 = ctx
                    .queue
                    .commandBuffer()
                    .expect("Failed to create command buffer");

                // scan_add_offsets
                {
                    let pso = self.pso_cache.get_or_create(ctx.library(), "scan_add_offsets");
                    let encoder = cmd_buf2
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");
                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(scanned_buf.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(partials_buf.as_ref()), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(scan_params_buf.as_ref()), 0, 2);
                    }
                    let grid = objc2_metal::MTLSize { width: scan_tgs, height: 1, depth: 1 };
                    let tg = objc2_metal::MTLSize { width: 256, height: 1, depth: 1 };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    encoder.endEncoding();
                }

                // radix_scatter (keys)
                {
                    let pso = self.pso_cache.get_or_create(ctx.library(), "radix_scatter");
                    let encoder = cmd_buf2
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");
                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(key_in.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(key_out.as_ref()), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(scanned_buf.as_ref()), 0, 2);
                        encoder.setBuffer_offset_atIndex(Some(sort_params_buf.as_ref()), 0, 3);
                    }
                    let grid = objc2_metal::MTLSize { width: num_sort_tgs, height: 1, depth: 1 };
                    let tg = objc2_metal::MTLSize { width: TG_SIZE, height: 1, depth: 1 };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    encoder.endEncoding();
                }

                cmd_buf2.commit();
                cmd_buf2.waitUntilCompleted();
            }

            // Now scatter values using the same scanned histogram offsets.
            // We need to re-read the scatter positions, but the radix_scatter kernel
            // only handles keys. For paired key-value sort, we apply the same permutation
            // by sorting on CPU side after reading back sorted keys.
            // Actually, we need to pair-sort values too. The simplest approach:
            // read back scanned histogram and apply the same scatter to values on CPU.
            // But that defeats the purpose. Instead, let's sort values by building
            // an index permutation from the sort.
        }

        // After 8 passes (even), sorted keys are in keys_a, but values are NOT permuted.
        // We need to co-sort values. Since our radix_scatter kernel only scatters keys,
        // we'll reconstruct the sorted values by reading back sorted keys and building
        // the permutation on CPU. This is a hybrid approach.
        //
        // For a real implementation, we'd modify radix_scatter to also scatter values.
        // For the benchmark, we do the value permutation on CPU as part of the GPU timing
        // to keep it honest.

        // Read sorted keys to build value permutation on CPU
        let sorted_keys: Vec<u32> = unsafe { read_buffer_slice(keys_a.as_ref(), size) };

        // Build sort permutation: sort original (key, index) pairs by key to match GPU order
        let mut indexed: Vec<(u32, usize)> = self.keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
        indexed.sort_by_key(|&(k, _)| k);

        // Apply permutation to values and upload to values_a
        let sorted_values: Vec<f32> = indexed.iter().map(|&(_, orig_idx)| self.values[orig_idx]).collect();
        unsafe {
            let vptr = values_a.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(sorted_values.as_ptr(), vptr, size);
        }

        // === PHASE 2: Boundary detection ===
        let groupby_params = GroupByParams {
            element_count: size as u32,
            num_groups: 0, // Will be determined after boundary detect
            _pad: [0; 2],
        };
        let groupby_params_buf = alloc_buffer_with_data(&ctx.device, &[groupby_params]);

        {
            let cmd_buf = ctx
                .queue
                .commandBuffer()
                .expect("Failed to create command buffer");

            let pso = self.pso_cache.get_or_create(ctx.library(), "groupby_boundary_detect");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");
            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(keys_a.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(flags_buf.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(groupby_params_buf.as_ref()), 0, 2);
            }
            let tg_count = size.div_ceil(TG_SIZE);
            let grid = objc2_metal::MTLSize { width: tg_count, height: 1, depth: 1 };
            let tg = objc2_metal::MTLSize { width: TG_SIZE, height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        // === PHASE 3: Compute group offsets on CPU from boundary flags ===
        let flags: Vec<u32> = unsafe { read_buffer_slice(flags_buf.as_ref(), size) };
        let mut group_offsets: Vec<u32> = Vec::new();
        for (i, &f) in flags.iter().enumerate() {
            if f == 1 {
                group_offsets.push(i as u32);
            }
        }
        let num_groups = group_offsets.len();
        self.gpu_num_groups = num_groups;

        // Upload group offsets to GPU
        unsafe {
            let ptr = group_offsets_buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(group_offsets.as_ptr(), ptr, num_groups);
        }

        // === PHASE 4: Segmented reduce on GPU ===
        let reduce_params = GroupByParams {
            element_count: size as u32,
            num_groups: num_groups as u32,
            _pad: [0; 2],
        };
        let reduce_params_buf = alloc_buffer_with_data(&ctx.device, &[reduce_params]);

        {
            let cmd_buf = ctx
                .queue
                .commandBuffer()
                .expect("Failed to create command buffer");

            let pso = self.pso_cache.get_or_create(ctx.library(), "groupby_segmented_reduce");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");
            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(values_a.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(group_offsets_buf.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(agg_sum_buf.as_ref()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(agg_count_buf.as_ref()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(agg_min_buf.as_ref()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(agg_max_buf.as_ref()), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(reduce_params_buf.as_ref()), 0, 6);
            }
            let tg_count = num_groups.div_ceil(TG_SIZE);
            let grid = objc2_metal::MTLSize { width: tg_count.max(1), height: 1, depth: 1 };
            let tg = objc2_metal::MTLSize { width: TG_SIZE, height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        let elapsed = timer.stop();

        // Read back results
        self.sorted_keys_gpu = sorted_keys;
        self.sorted_values_gpu = sorted_values;

        let sums: Vec<f32> = unsafe { read_buffer_slice(agg_sum_buf.as_ref(), num_groups) };
        let counts: Vec<u32> = unsafe { read_buffer_slice(agg_count_buf.as_ref(), num_groups) };
        let mins: Vec<f32> = unsafe { read_buffer_slice(agg_min_buf.as_ref(), num_groups) };
        let maxs: Vec<f32> = unsafe { read_buffer_slice(agg_max_buf.as_ref(), num_groups) };

        self.gpu_agg = (0..num_groups)
            .map(|i| (sums[i] as f64, counts[i], mins[i], maxs[i]))
            .collect();

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();
        self.cpu_agg = hashmap_ops::hashmap_groupby(&self.keys, &self.values);
        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // Build a map from sorted key -> GPU aggregate for comparison
        // Group offsets give us the start of each group in sorted order
        // The key at each group offset is the group key
        let flags: Vec<u32> = {
            let mut f = Vec::new();
            for i in 0..self.sorted_keys_gpu.len() {
                if i == 0 || self.sorted_keys_gpu[i] != self.sorted_keys_gpu[i - 1] {
                    f.push(i);
                }
            }
            f.iter().map(|&x| x as u32).collect()
        };

        if self.gpu_agg.len() != self.cpu_agg.len() {
            return Err(format!(
                "Group count mismatch: GPU={} CPU={}",
                self.gpu_agg.len(),
                self.cpu_agg.len()
            ));
        }

        // For each GPU group, look up the key and compare with CPU
        for (group_idx, &(gpu_sum, gpu_count, gpu_min, gpu_max)) in self.gpu_agg.iter().enumerate()
        {
            let key_offset = flags[group_idx] as usize;
            let group_key = self.sorted_keys_gpu[key_offset];

            let cpu_group = self.cpu_agg.get(&group_key).ok_or_else(|| {
                format!("GPU group key {} not found in CPU results", group_key)
            })?;

            // Compare count (exact)
            if gpu_count != cpu_group.count {
                return Err(format!(
                    "Group {} count mismatch: GPU={} CPU={}",
                    group_key, gpu_count, cpu_group.count
                ));
            }

            // Compare sum (relative error 1e-3)
            let rel_sum = if cpu_group.sum.abs() > 1e-10 {
                (gpu_sum - cpu_group.sum).abs() / cpu_group.sum.abs()
            } else {
                (gpu_sum - cpu_group.sum).abs()
            };
            if rel_sum > 1e-3 {
                return Err(format!(
                    "Group {} sum mismatch: GPU={:.4} CPU={:.4} rel_err={:.6}",
                    group_key, gpu_sum, cpu_group.sum, rel_sum
                ));
            }

            // Compare min (relative error 1e-3)
            let rel_min = if cpu_group.min.abs() > 1e-10 {
                (gpu_min - cpu_group.min).abs() / cpu_group.min.abs()
            } else {
                (gpu_min - cpu_group.min).abs()
            };
            if rel_min > 1e-3 {
                return Err(format!(
                    "Group {} min mismatch: GPU={:.4} CPU={:.4}",
                    group_key, gpu_min, cpu_group.min
                ));
            }

            // Compare max (relative error 1e-3)
            let rel_max = if cpu_group.max.abs() > 1e-10 {
                (gpu_max - cpu_group.max).abs() / cpu_group.max.abs()
            } else {
                (gpu_max - cpu_group.max).abs()
            };
            if rel_max > 1e-3 {
                return Err(format!(
                    "Group {} max mismatch: GPU={:.4} CPU={:.4}",
                    group_key, gpu_max, cpu_group.max
                ));
            }
        }

        Ok(())
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let seconds = elapsed_ms / 1000.0;

        let rows_per_sec = if seconds > 0.0 {
            size as f64 / seconds
        } else {
            0.0
        };
        m.insert("rows_per_sec".to_string(), rows_per_sec);
        m.insert("elements".to_string(), size as f64);
        m.insert("num_groups".to_string(), self.gpu_num_groups as f64);
        m.insert("cardinality".to_string(), self.num_groups as f64);

        // Bandwidth: sort reads+writes (8 passes * 2 * N * 4) + boundary detect (N * 4 read + N * 4 write) + reduce (N * 4 read)
        let sort_bytes = (NUM_PASSES as f64) * 2.0 * (size as f64) * 4.0;
        let boundary_bytes = (size as f64) * 8.0; // read keys + write flags
        let reduce_bytes = (size as f64) * 4.0; // read values
        let total_bytes = sort_bytes + boundary_bytes + reduce_bytes;
        let gbs = if seconds > 0.0 {
            total_bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);

        m
    }
}
