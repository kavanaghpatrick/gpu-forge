//! DuckDB comparison experiment.
//!
//! Runs the same analytical pipeline as the pipeline experiment:
//!   SELECT key, SUM(value) FROM data WHERE key > threshold
//!   GROUP BY key ORDER BY SUM(value) DESC LIMIT K
//!
//! GPU side: reuses the pipeline experiment's dispatch chain
//!   (filter -> compact -> sort -> groupby -> top-K)
//!
//! CPU side: shells out to `duckdb` CLI with the equivalent SQL query.
//!   Times wall-clock including DuckDB startup overhead (~50ms).
//!
//! If DuckDB is not installed, the experiment gracefully skips with a warning.

use std::collections::HashMap;
use std::path::PathBuf;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder,
};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, read_buffer_slice, BenchTimer,
    CompactParams, GroupByParams, MetalContext, PsoCache, ScanParams, SortParams,
};

use crate::cpu_baselines::duckdb_runner;
use crate::data_gen::DataGenerator;

use super::Experiment;

/// Threads per threadgroup.
const TG_SIZE: usize = 256;

/// Radix sort constants (same as pipeline experiment).
const RADIX_BITS: usize = 4;
const RADIX_BINS: usize = 16;
const NUM_PASSES: usize = 8;
const SCAN_ELEMENTS_PER_TG: usize = 512;
const MAX_GPU_PARTIALS: usize = 512;

/// Number of groups for the pipeline.
const NUM_GROUPS: u32 = 1000;

/// Top-K results to extract.
const TOP_K: usize = 10;

/// DuckDB comparison experiment.
///
/// Compares GPU analytical pipeline vs DuckDB CLI for the same query.
pub struct DuckDbExperiment {
    /// Key column (u32).
    keys: Vec<u32>,
    /// Value column (f32).
    values: Vec<f32>,
    /// Filter threshold for keys (u32).
    key_threshold: u32,
    /// Current element count.
    size: usize,
    /// Whether duckdb CLI is available.
    duckdb_available: bool,
    /// Path to temporary CSV file for DuckDB.
    csv_path: PathBuf,

    // === GPU buffers (same as pipeline) ===
    keys_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    values_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    // Compact stage
    compact_flags_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    compact_scan_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    compact_partials_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    compacted_keys_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    compacted_values_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    // Sort stage (ping-pong)
    sort_keys_b: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    sort_histogram_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    sort_scanned_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    sort_partials_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    // Groupby stage
    groupby_flags_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    group_offsets_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    agg_sum_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    agg_count_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    agg_min_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    agg_max_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    pso_cache: PsoCache,

    // Results
    gpu_topk: Vec<(u32, f64)>,
    duckdb_topk: Vec<(u32, f64)>,
    /// Last DuckDB elapsed time in ms.
    last_duckdb_ms: f64,
}

impl DuckDbExperiment {
    pub fn new() -> Self {
        let duckdb_available = duckdb_runner::duckdb_available();
        if !duckdb_available {
            eprintln!("WARNING: duckdb CLI not found -- DuckDB experiment will use CPU fallback");
        }

        // Use temp directory for CSV file
        let csv_path = std::env::temp_dir().join("forge_bench_duckdb_data.csv");

        Self {
            keys: Vec::new(),
            values: Vec::new(),
            key_threshold: 0,
            size: 0,
            duckdb_available,
            csv_path,
            keys_buffer: None,
            values_buffer: None,
            compact_flags_buffer: None,
            compact_scan_buffer: None,
            compact_partials_buffer: None,
            compacted_keys_buffer: None,
            compacted_values_buffer: None,
            sort_keys_b: None,
            sort_histogram_buffer: None,
            sort_scanned_buffer: None,
            sort_partials_buffer: None,
            groupby_flags_buffer: None,
            group_offsets_buffer: None,
            agg_sum_buffer: None,
            agg_count_buffer: None,
            agg_min_buffer: None,
            agg_max_buffer: None,
            pso_cache: PsoCache::new(),
            gpu_topk: Vec::new(),
            duckdb_topk: Vec::new(),
            last_duckdb_ms: 0.0,
        }
    }
}

impl Experiment for DuckDbExperiment {
    fn name(&self) -> &str {
        "duckdb"
    }

    fn description(&self) -> &str {
        "DuckDB comparison: GPU pipeline vs DuckDB CLI for analytical query"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![
            100_000,     // 100K
            1_000_000,   // 1M
            10_000_000,  // 10M
        ]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;

        // Generate same columnar data as pipeline experiment
        let raw_keys = gen.uniform_u32(size);
        self.keys = raw_keys.iter().map(|&k| k % NUM_GROUPS).collect();
        self.values = gen.uniform_f32(size).iter().map(|&v| v * 1000.0).collect();

        // Filter threshold: keep keys > NUM_GROUPS/2 (~50% selectivity)
        self.key_threshold = NUM_GROUPS / 2;

        // Write CSV file for DuckDB
        if self.duckdb_available {
            if let Err(e) = duckdb_runner::write_csv_data(&self.csv_path, &self.keys, &self.values)
            {
                eprintln!("WARNING: Failed to write CSV for DuckDB: {}", e);
            }
        }

        let max_compacted = size;

        // Input buffers
        self.keys_buffer = Some(alloc_buffer_with_data(&ctx.device, &self.keys));
        let values_as_u32: Vec<u32> = self.values.iter().map(|v| v.to_bits()).collect();
        self.values_buffer = Some(alloc_buffer_with_data(&ctx.device, &values_as_u32));

        // Compact stage buffers
        self.compact_flags_buffer = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));
        self.compact_scan_buffer = Some(alloc_buffer(
            &ctx.device,
            size * std::mem::size_of::<u32>(),
        ));
        let num_scan_tgs = size.div_ceil(SCAN_ELEMENTS_PER_TG);
        self.compact_partials_buffer = Some(alloc_buffer(
            &ctx.device,
            num_scan_tgs.max(1) * std::mem::size_of::<u32>(),
        ));
        self.compacted_keys_buffer = Some(alloc_buffer(
            &ctx.device,
            max_compacted * std::mem::size_of::<u32>(),
        ));
        self.compacted_values_buffer = Some(alloc_buffer(
            &ctx.device,
            max_compacted * std::mem::size_of::<f32>(),
        ));

        // Sort stage buffers
        let num_sort_tgs = max_compacted.div_ceil(TG_SIZE);
        let histogram_size = num_sort_tgs * RADIX_BINS;

        self.sort_keys_b = Some(alloc_buffer(
            &ctx.device,
            max_compacted * std::mem::size_of::<u32>(),
        ));
        self.sort_histogram_buffer = Some(alloc_buffer(
            &ctx.device,
            histogram_size * std::mem::size_of::<u32>(),
        ));
        self.sort_scanned_buffer = Some(alloc_buffer(
            &ctx.device,
            histogram_size * std::mem::size_of::<u32>(),
        ));
        let sort_scan_tgs = histogram_size.div_ceil(SCAN_ELEMENTS_PER_TG);
        self.sort_partials_buffer = Some(alloc_buffer(
            &ctx.device,
            sort_scan_tgs.max(1) * std::mem::size_of::<u32>(),
        ));

        // Groupby stage buffers
        self.groupby_flags_buffer = Some(alloc_buffer(
            &ctx.device,
            max_compacted * std::mem::size_of::<u32>(),
        ));
        let max_groups = (NUM_GROUPS as usize + 1).min(max_compacted);
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
        self.pso_cache.get_or_create(ctx.library(), "compact_flags");
        self.pso_cache
            .get_or_create(ctx.library(), "compact_scatter");
        self.pso_cache.get_or_create(ctx.library(), "scan_local");
        self.pso_cache
            .get_or_create(ctx.library(), "scan_partials");
        self.pso_cache
            .get_or_create(ctx.library(), "scan_add_offsets");
        self.pso_cache
            .get_or_create(ctx.library(), "radix_histogram");
        self.pso_cache
            .get_or_create(ctx.library(), "radix_scatter");
        self.pso_cache
            .get_or_create(ctx.library(), "groupby_boundary_detect");
        self.pso_cache
            .get_or_create(ctx.library(), "groupby_segmented_reduce");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let size = self.size;

        // Clone buffer pointers to avoid borrow conflicts with pso_cache
        let keys_buf = self.keys_buffer.clone().expect("setup not called");
        let values_buf = self.values_buffer.clone().expect("setup not called");
        let compact_flags = self.compact_flags_buffer.clone().expect("setup not called");
        let compact_scan = self.compact_scan_buffer.clone().expect("setup not called");
        let compact_partials = self
            .compact_partials_buffer
            .clone()
            .expect("setup not called");
        let compacted_keys = self
            .compacted_keys_buffer
            .clone()
            .expect("setup not called");
        let compacted_values = self
            .compacted_values_buffer
            .clone()
            .expect("setup not called");
        let sort_keys_b = self.sort_keys_b.clone().expect("setup not called");
        let sort_histogram = self
            .sort_histogram_buffer
            .clone()
            .expect("setup not called");
        let sort_scanned = self.sort_scanned_buffer.clone().expect("setup not called");
        let sort_partials = self.sort_partials_buffer.clone().expect("setup not called");
        let groupby_flags = self
            .groupby_flags_buffer
            .clone()
            .expect("setup not called");
        let group_offsets_buf = self
            .group_offsets_buffer
            .clone()
            .expect("setup not called");
        let agg_sum_buf = self.agg_sum_buffer.clone().expect("setup not called");
        let agg_count_buf = self.agg_count_buffer.clone().expect("setup not called");
        let agg_min_buf = self.agg_min_buffer.clone().expect("setup not called");
        let agg_max_buf = self.agg_max_buffer.clone().expect("setup not called");

        // Re-upload input data before each run
        unsafe {
            let kptr = keys_buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(self.keys.as_ptr(), kptr, size);
            let values_as_u32: Vec<u32> = self.values.iter().map(|v| v.to_bits()).collect();
            let vptr = values_buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(values_as_u32.as_ptr(), vptr, size);
        }

        let num_scan_tgs = size.div_ceil(SCAN_ELEMENTS_PER_TG);

        let timer = BenchTimer::start();

        // ====================================================================
        // STAGE 1+2: Filter + Compact (same as pipeline experiment)
        // ====================================================================
        let compact_params = CompactParams {
            element_count: size as u32,
            threshold: self.key_threshold,
            _pad: [0; 2],
        };
        let compact_params_buf = alloc_buffer_with_data(&ctx.device, &[compact_params]);

        let scan_params = ScanParams {
            element_count: size as u32,
            pass: 0,
            _pad: [0; 2],
        };
        let scan_params_buf = alloc_buffer_with_data(&ctx.device, &[scan_params]);

        let partials_scan_params = ScanParams {
            element_count: num_scan_tgs as u32,
            pass: 1,
            _pad: [0; 2],
        };
        let partials_scan_params_buf =
            alloc_buffer_with_data(&ctx.device, &[partials_scan_params]);

        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .expect("Failed to create command buffer");

        // --- compact_flags on keys ---
        {
            let pso = self.pso_cache.get_or_create(ctx.library(), "compact_flags");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");
            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(keys_buf.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(compact_flags.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(compact_params_buf.as_ref()), 0, 2);
            }
            let num_tgs = size.div_ceil(TG_SIZE);
            let grid = objc2_metal::MTLSize {
                width: num_tgs,
                height: 1,
                depth: 1,
            };
            let tg = objc2_metal::MTLSize {
                width: TG_SIZE,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
        }

        // --- scan_local on flags ---
        {
            let pso = self.pso_cache.get_or_create(ctx.library(), "scan_local");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");
            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(compact_flags.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(compact_scan.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(compact_partials.as_ref()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(scan_params_buf.as_ref()), 0, 3);
            }
            let grid = objc2_metal::MTLSize {
                width: num_scan_tgs,
                height: 1,
                depth: 1,
            };
            let tg = objc2_metal::MTLSize {
                width: TG_SIZE,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
        }

        let compacted_count;

        if num_scan_tgs <= MAX_GPU_PARTIALS {
            // GPU partials scan
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "scan_partials");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(compact_partials.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(
                        Some(partials_scan_params_buf.as_ref()),
                        0,
                        1,
                    );
                }
                let grid = objc2_metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            // scan_add_offsets
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "scan_add_offsets");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(compact_scan.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(compact_partials.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(scan_params_buf.as_ref()), 0, 2);
                }
                let grid = objc2_metal::MTLSize {
                    width: num_scan_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            // compact_scatter keys
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "compact_scatter");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(keys_buf.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(compact_flags.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(compact_scan.as_ref()), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(compacted_keys.as_ref()), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(compact_params_buf.as_ref()), 0, 4);
                }
                let num_tgs = size.div_ceil(TG_SIZE);
                let grid = objc2_metal::MTLSize {
                    width: num_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            // compact_scatter values
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "compact_scatter");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(values_buf.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(compact_flags.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(compact_scan.as_ref()), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(compacted_values.as_ref()), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(compact_params_buf.as_ref()), 0, 4);
                }
                let num_tgs = size.div_ceil(TG_SIZE);
                let grid = objc2_metal::MTLSize {
                    width: num_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            let last_scan: u32 =
                unsafe { read_buffer_slice(compact_scan.as_ref(), size) }[size - 1];
            let last_flag: u32 =
                unsafe { read_buffer_slice(compact_flags.as_ref(), size) }[size - 1];
            compacted_count = (last_scan + last_flag) as usize;
        } else {
            // CPU fallback for partials scan
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            let partials_data: Vec<u32> =
                unsafe { read_buffer_slice(compact_partials.as_ref(), num_scan_tgs) };
            let scanned =
                crate::cpu_baselines::sequential::sequential_exclusive_scan(&partials_data);
            unsafe {
                let ptr = compact_partials.contents().as_ptr() as *mut u32;
                std::ptr::copy_nonoverlapping(scanned.as_ptr(), ptr, num_scan_tgs);
            }

            let cmd_buf2 = ctx
                .queue
                .commandBuffer()
                .expect("Failed to create command buffer");

            // scan_add_offsets
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "scan_add_offsets");
                let encoder = cmd_buf2
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(compact_scan.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(compact_partials.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(scan_params_buf.as_ref()), 0, 2);
                }
                let grid = objc2_metal::MTLSize {
                    width: num_scan_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            // compact_scatter keys
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "compact_scatter");
                let encoder = cmd_buf2
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(keys_buf.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(compact_flags.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(compact_scan.as_ref()), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(compacted_keys.as_ref()), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(compact_params_buf.as_ref()), 0, 4);
                }
                let num_tgs = size.div_ceil(TG_SIZE);
                let grid = objc2_metal::MTLSize {
                    width: num_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            // compact_scatter values
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "compact_scatter");
                let encoder = cmd_buf2
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(values_buf.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(compact_flags.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(compact_scan.as_ref()), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(compacted_values.as_ref()), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(compact_params_buf.as_ref()), 0, 4);
                }
                let num_tgs = size.div_ceil(TG_SIZE);
                let grid = objc2_metal::MTLSize {
                    width: num_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            cmd_buf2.commit();
            cmd_buf2.waitUntilCompleted();

            let last_scan: u32 =
                unsafe { read_buffer_slice(compact_scan.as_ref(), size) }[size - 1];
            let last_flag: u32 =
                unsafe { read_buffer_slice(compact_flags.as_ref(), size) }[size - 1];
            compacted_count = (last_scan + last_flag) as usize;
        }

        if compacted_count == 0 {
            let elapsed = timer.stop();
            self.gpu_topk = Vec::new();
            return elapsed;
        }

        // ====================================================================
        // STAGE 3: Radix sort compacted keys
        // ====================================================================
        let sort_n = compacted_count;
        let sort_tgs = sort_n.div_ceil(TG_SIZE);
        let sort_hist_size = sort_tgs * RADIX_BINS;
        let sort_scan_tgs = sort_hist_size.div_ceil(SCAN_ELEMENTS_PER_TG);

        for pass in 0..NUM_PASSES {
            let (key_in, key_out) = if pass % 2 == 0 {
                (&compacted_keys, &sort_keys_b)
            } else {
                (&sort_keys_b, &compacted_keys)
            };

            let bit_offset = (pass * RADIX_BITS) as u32;

            // Zero histogram
            unsafe {
                let ptr = sort_histogram.contents().as_ptr() as *mut u32;
                for i in 0..sort_hist_size {
                    *ptr.add(i) = 0;
                }
            }

            let sort_params = SortParams {
                element_count: sort_n as u32,
                bit_offset,
                num_threadgroups: sort_tgs as u32,
                _pad: 0,
            };
            let sort_params_buf = alloc_buffer_with_data(&ctx.device, &[sort_params]);

            let hist_scan_params = ScanParams {
                element_count: sort_hist_size as u32,
                pass: 0,
                _pad: [0; 2],
            };
            let hist_scan_params_buf = alloc_buffer_with_data(&ctx.device, &[hist_scan_params]);

            let cmd_buf = ctx
                .queue
                .commandBuffer()
                .expect("Failed to create command buffer");

            // radix_histogram
            {
                let pso = self
                    .pso_cache
                    .get_or_create(ctx.library(), "radix_histogram");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(key_in.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(sort_histogram.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(sort_params_buf.as_ref()), 0, 2);
                }
                let grid = objc2_metal::MTLSize {
                    width: sort_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            // scan_local on histogram
            {
                let pso = self.pso_cache.get_or_create(ctx.library(), "scan_local");
                let encoder = cmd_buf
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");
                encoder.setComputePipelineState(pso);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(sort_histogram.as_ref()), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(sort_scanned.as_ref()), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(sort_partials.as_ref()), 0, 2);
                    encoder.setBuffer_offset_atIndex(
                        Some(hist_scan_params_buf.as_ref()),
                        0,
                        3,
                    );
                }
                let grid = objc2_metal::MTLSize {
                    width: sort_scan_tgs,
                    height: 1,
                    depth: 1,
                };
                let tg = objc2_metal::MTLSize {
                    width: TG_SIZE,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                encoder.endEncoding();
            }

            if sort_scan_tgs <= MAX_GPU_PARTIALS {
                let part_params = ScanParams {
                    element_count: sort_scan_tgs as u32,
                    pass: 1,
                    _pad: [0; 2],
                };
                let part_params_buf = alloc_buffer_with_data(&ctx.device, &[part_params]);

                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "scan_partials");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");
                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(sort_partials.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(part_params_buf.as_ref()), 0, 1);
                    }
                    let grid = objc2_metal::MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    };
                    let tg = objc2_metal::MTLSize {
                        width: TG_SIZE,
                        height: 1,
                        depth: 1,
                    };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    encoder.endEncoding();
                }

                // scan_add_offsets
                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "scan_add_offsets");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");
                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(sort_scanned.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(sort_partials.as_ref()), 0, 1);
                        encoder.setBuffer_offset_atIndex(
                            Some(hist_scan_params_buf.as_ref()),
                            0,
                            2,
                        );
                    }
                    let grid = objc2_metal::MTLSize {
                        width: sort_scan_tgs,
                        height: 1,
                        depth: 1,
                    };
                    let tg = objc2_metal::MTLSize {
                        width: TG_SIZE,
                        height: 1,
                        depth: 1,
                    };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    encoder.endEncoding();
                }

                // radix_scatter
                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "radix_scatter");
                    let encoder = cmd_buf
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");
                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(key_in.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(key_out.as_ref()), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(sort_scanned.as_ref()), 0, 2);
                        encoder.setBuffer_offset_atIndex(Some(sort_params_buf.as_ref()), 0, 3);
                    }
                    let grid = objc2_metal::MTLSize {
                        width: sort_tgs,
                        height: 1,
                        depth: 1,
                    };
                    let tg = objc2_metal::MTLSize {
                        width: TG_SIZE,
                        height: 1,
                        depth: 1,
                    };
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
                    unsafe { read_buffer_slice(sort_partials.as_ref(), sort_scan_tgs) };
                let scanned_partials =
                    crate::cpu_baselines::sequential::sequential_exclusive_scan(&partials_data);
                unsafe {
                    let ptr = sort_partials.contents().as_ptr() as *mut u32;
                    std::ptr::copy_nonoverlapping(
                        scanned_partials.as_ptr(),
                        ptr,
                        sort_scan_tgs,
                    );
                }

                let cmd_buf2 = ctx
                    .queue
                    .commandBuffer()
                    .expect("Failed to create command buffer");

                // scan_add_offsets
                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "scan_add_offsets");
                    let encoder = cmd_buf2
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");
                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(sort_scanned.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(sort_partials.as_ref()), 0, 1);
                        encoder.setBuffer_offset_atIndex(
                            Some(hist_scan_params_buf.as_ref()),
                            0,
                            2,
                        );
                    }
                    let grid = objc2_metal::MTLSize {
                        width: sort_scan_tgs,
                        height: 1,
                        depth: 1,
                    };
                    let tg = objc2_metal::MTLSize {
                        width: TG_SIZE,
                        height: 1,
                        depth: 1,
                    };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    encoder.endEncoding();
                }

                // radix_scatter
                {
                    let pso = self
                        .pso_cache
                        .get_or_create(ctx.library(), "radix_scatter");
                    let encoder = cmd_buf2
                        .computeCommandEncoder()
                        .expect("Failed to create compute encoder");
                    encoder.setComputePipelineState(pso);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(key_in.as_ref()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(key_out.as_ref()), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(sort_scanned.as_ref()), 0, 2);
                        encoder.setBuffer_offset_atIndex(Some(sort_params_buf.as_ref()), 0, 3);
                    }
                    let grid = objc2_metal::MTLSize {
                        width: sort_tgs,
                        height: 1,
                        depth: 1,
                    };
                    let tg = objc2_metal::MTLSize {
                        width: TG_SIZE,
                        height: 1,
                        depth: 1,
                    };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    encoder.endEncoding();
                }

                cmd_buf2.commit();
                cmd_buf2.waitUntilCompleted();
            }
        }

        // === Value reorder (hybrid: CPU permutation) ===
        let filtered_kv: Vec<(u32, f32)> = self
            .keys
            .iter()
            .zip(self.values.iter())
            .filter(|(&k, _)| k > self.key_threshold)
            .map(|(&k, &v)| (k, v))
            .collect();

        let mut sorted_kv = filtered_kv;
        sorted_kv.sort_by_key(|&(k, _)| k);

        // Upload sorted values for segmented reduce
        unsafe {
            let vptr = compacted_values.contents().as_ptr() as *mut f32;
            for (i, &(_, v)) in sorted_kv.iter().enumerate() {
                *vptr.add(i) = v;
            }
        }

        // ====================================================================
        // STAGE 4: Boundary detect on sorted keys
        // ====================================================================
        let groupby_params = GroupByParams {
            element_count: sort_n as u32,
            num_groups: 0,
            _pad: [0; 2],
        };
        let groupby_params_buf = alloc_buffer_with_data(&ctx.device, &[groupby_params]);

        {
            let cmd_buf = ctx
                .queue
                .commandBuffer()
                .expect("Failed to create command buffer");

            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "groupby_boundary_detect");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");
            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(compacted_keys.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(groupby_flags.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(groupby_params_buf.as_ref()), 0, 2);
            }
            let tg_count = sort_n.div_ceil(TG_SIZE);
            let grid = objc2_metal::MTLSize {
                width: tg_count.max(1),
                height: 1,
                depth: 1,
            };
            let tg = objc2_metal::MTLSize {
                width: TG_SIZE,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        // Compute group offsets on CPU from boundary flags
        let flags: Vec<u32> =
            unsafe { read_buffer_slice(groupby_flags.as_ref(), sort_n) };
        let mut group_offsets: Vec<u32> = Vec::new();
        for (i, &f) in flags.iter().enumerate() {
            if f == 1 {
                group_offsets.push(i as u32);
            }
        }
        let num_groups = group_offsets.len();

        unsafe {
            let ptr = group_offsets_buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(group_offsets.as_ptr(), ptr, num_groups);
        }

        // ====================================================================
        // STAGE 5: Segmented reduce on GPU
        // ====================================================================
        let reduce_params = GroupByParams {
            element_count: sort_n as u32,
            num_groups: num_groups as u32,
            _pad: [0; 2],
        };
        let reduce_params_buf = alloc_buffer_with_data(&ctx.device, &[reduce_params]);

        {
            let cmd_buf = ctx
                .queue
                .commandBuffer()
                .expect("Failed to create command buffer");

            let pso = self
                .pso_cache
                .get_or_create(ctx.library(), "groupby_segmented_reduce");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");
            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(compacted_values.as_ref()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(group_offsets_buf.as_ref()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(agg_sum_buf.as_ref()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(agg_count_buf.as_ref()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(agg_min_buf.as_ref()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(agg_max_buf.as_ref()), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(reduce_params_buf.as_ref()), 0, 6);
            }
            let tg_count = num_groups.div_ceil(TG_SIZE);
            let grid = objc2_metal::MTLSize {
                width: tg_count.max(1),
                height: 1,
                depth: 1,
            };
            let tg = objc2_metal::MTLSize {
                width: TG_SIZE,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }

        let elapsed = timer.stop();

        // ====================================================================
        // STAGE 6: Top-K extraction (on CPU)
        // ====================================================================
        let sums: Vec<f32> =
            unsafe { read_buffer_slice(agg_sum_buf.as_ref(), num_groups) };

        let mut group_results: Vec<(u32, f64)> = Vec::with_capacity(num_groups);
        for (g, &sum_val) in sums.iter().enumerate() {
            let key_offset = group_offsets[g] as usize;
            let group_key = sorted_kv[key_offset].0;
            group_results.push((group_key, sum_val as f64));
        }
        group_results
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        group_results.truncate(TOP_K);

        self.gpu_topk = group_results;

        elapsed
    }

    fn run_cpu(&mut self) -> f64 {
        if !self.duckdb_available {
            // Fallback: CPU HashMap (same as pipeline) -- so the experiment still runs
            let timer = BenchTimer::start();

            let mut groups: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
            for (&k, &v) in self.keys.iter().zip(self.values.iter()) {
                if k > self.key_threshold {
                    *groups.entry(k).or_insert(0.0) += v as f64;
                }
            }

            let mut sorted: Vec<(u32, f64)> = groups.into_iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sorted.truncate(TOP_K);

            self.duckdb_topk = sorted;
            return timer.stop();
        }

        // DuckDB CLI path: run SQL query against CSV file
        match duckdb_runner::run_pipeline_query(
            &self.csv_path,
            self.key_threshold,
            TOP_K,
        ) {
            Some((elapsed_ms, results)) => {
                self.duckdb_topk = results;
                self.last_duckdb_ms = elapsed_ms;
                elapsed_ms
            }
            None => {
                eprintln!("WARNING: DuckDB query failed, falling back to CPU HashMap");
                let timer = BenchTimer::start();

                let mut groups: std::collections::HashMap<u32, f64> =
                    std::collections::HashMap::new();
                for (&k, &v) in self.keys.iter().zip(self.values.iter()) {
                    if k > self.key_threshold {
                        *groups.entry(k).or_insert(0.0) += v as f64;
                    }
                }

                let mut sorted: Vec<(u32, f64)> = groups.into_iter().collect();
                sorted.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                sorted.truncate(TOP_K);

                self.duckdb_topk = sorted;
                timer.stop()
            }
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.gpu_topk.len() != self.duckdb_topk.len() {
            return Err(format!(
                "Top-K count mismatch: GPU={} DuckDB/CPU={}",
                self.gpu_topk.len(),
                self.duckdb_topk.len()
            ));
        }

        for (i, (gpu, ddb)) in self
            .gpu_topk
            .iter()
            .zip(self.duckdb_topk.iter())
            .enumerate()
        {
            let rel_err = if ddb.1.abs() > 1e-10 {
                (gpu.1 - ddb.1).abs() / ddb.1.abs()
            } else {
                (gpu.1 - ddb.1).abs()
            };

            if gpu.0 != ddb.0 && rel_err > 0.01 {
                return Err(format!(
                    "Top-{}: key mismatch GPU=({}, {:.2}) DuckDB=({}, {:.2}) rel_err={:.6}",
                    i + 1,
                    gpu.0,
                    gpu.1,
                    ddb.0,
                    ddb.1,
                    rel_err
                ));
            }

            if rel_err > 0.01 {
                return Err(format!(
                    "Top-{}: sum mismatch GPU=({}, {:.2}) DuckDB=({}, {:.2}) rel_err={:.6}",
                    i + 1,
                    gpu.0,
                    gpu.1,
                    ddb.0,
                    ddb.1,
                    rel_err
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
        m.insert("top_k".to_string(), TOP_K as f64);
        m.insert("num_groups".to_string(), NUM_GROUPS as f64);
        m.insert(
            "duckdb_available".to_string(),
            if self.duckdb_available { 1.0 } else { 0.0 },
        );

        if self.last_duckdb_ms > 0.0 {
            m.insert("duckdb_ms".to_string(), self.last_duckdb_ms);
            m.insert("gpu_ms".to_string(), elapsed_ms);
            let ratio = if elapsed_ms > 0.0 {
                self.last_duckdb_ms / elapsed_ms
            } else {
                0.0
            };
            m.insert("duckdb_gpu_ratio".to_string(), ratio);
        }

        m
    }
}
