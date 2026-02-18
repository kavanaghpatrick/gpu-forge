//! Spreadsheet formula experiment: GPU column SUM/AVERAGE/VLOOKUP vs CPU sequential.
//!
//! Simulates spreadsheet workloads on a 2D grid of cells (rows x cols).
//! The "size" parameter maps to total cell count (1M = 1000x1000 grid).
//!
//! Three formula types tested per run:
//!   - Column SUM: reduce each column across all rows
//!   - Column AVERAGE: SUM / row_count per column
//!   - VLOOKUP: binary search in sorted lookup column
//!
//! Primary metric: cells/sec (total cells processed per second).

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use forge_primitives::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, dispatch_2d, read_buffer_slice, BenchTimer,
    GpuTimer, MetalContext, PsoCache, SpreadsheetParams,
};

use crate::data_gen::DataGenerator;

use super::Experiment;

/// Spreadsheet formula evaluation experiment.
pub struct SpreadsheetExperiment {
    /// Grid data: rows x cols, row-major f32.
    grid_data: Vec<f32>,
    /// Sorted lookup keys for VLOOKUP.
    lookup_keys: Vec<f32>,
    /// Lookup values corresponding to sorted keys.
    lookup_vals: Vec<f32>,
    /// Search keys for VLOOKUP (random subset of lookup keys).
    search_keys: Vec<f32>,
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Metal buffer for grid data.
    buf_grid: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for SUM output (cols elements).
    buf_col_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for AVERAGE output (cols elements).
    buf_avg_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for row-chunk partials (num_row_chunks * cols elements).
    buf_partials: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for SpreadsheetParams.
    buf_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for sorted lookup keys.
    buf_lookup_keys: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for lookup values.
    buf_lookup_vals: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for search keys.
    buf_search_keys: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for VLOOKUP output.
    buf_vlookup_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Metal buffer for VLOOKUP params.
    buf_vlookup_params: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// PSO cache for kernel lookup.
    pso_cache: PsoCache,
    /// GPU column sums from last run.
    gpu_col_sums: Vec<f32>,
    /// GPU column averages from last run.
    gpu_col_avgs: Vec<f32>,
    /// GPU VLOOKUP results from last run.
    gpu_vlookup: Vec<f32>,
    /// CPU column sums from last run.
    cpu_col_sums: Vec<f32>,
    /// CPU column averages from last run.
    cpu_col_avgs: Vec<f32>,
    /// CPU VLOOKUP results from last run.
    cpu_vlookup: Vec<f32>,
    /// Current total size (rows * cols).
    size: usize,
}

impl SpreadsheetExperiment {
    pub fn new() -> Self {
        Self {
            grid_data: Vec::new(),
            lookup_keys: Vec::new(),
            lookup_vals: Vec::new(),
            search_keys: Vec::new(),
            rows: 0,
            cols: 0,
            buf_grid: None,
            buf_col_output: None,
            buf_avg_output: None,
            buf_partials: None,
            buf_params: None,
            buf_lookup_keys: None,
            buf_lookup_vals: None,
            buf_search_keys: None,
            buf_vlookup_output: None,
            buf_vlookup_params: None,
            pso_cache: PsoCache::new(),
            gpu_col_sums: Vec::new(),
            gpu_col_avgs: Vec::new(),
            gpu_vlookup: Vec::new(),
            cpu_col_sums: Vec::new(),
            cpu_col_avgs: Vec::new(),
            cpu_vlookup: Vec::new(),
            size: 0,
        }
    }

    /// Derive grid dimensions from total cell count.
    /// Target: square-ish grid. For 1M -> 1000x1000.
    fn grid_dims(total_cells: usize) -> (usize, usize) {
        let cols = (total_cells as f64).sqrt() as usize;
        let rows = total_cells / cols;
        (rows.max(1), cols.max(1))
    }

    /// CPU: compute column sums.
    fn cpu_column_sums(grid: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut sums = vec![0.0f32; cols];
        for row in 0..rows {
            for col in 0..cols {
                sums[col] += grid[row * cols + col];
            }
        }
        sums
    }

    /// CPU: compute column averages.
    fn cpu_column_averages(grid: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let sums = Self::cpu_column_sums(grid, rows, cols);
        sums.iter().map(|s| s / rows as f32).collect()
    }

    /// CPU: VLOOKUP via binary search in sorted lookup_keys.
    fn cpu_vlookup(
        lookup_keys: &[f32],
        lookup_vals: &[f32],
        search_keys: &[f32],
    ) -> Vec<f32> {
        search_keys
            .iter()
            .map(|key| {
                // Binary search: find largest index where lookup_keys[idx] <= key
                let mut lo = 0usize;
                let mut hi = lookup_keys.len();
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if lookup_keys[mid] <= *key {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                if lo > 0 {
                    lookup_vals[lo - 1]
                } else {
                    -1.0f32
                }
            })
            .collect()
    }
}

impl Experiment for SpreadsheetExperiment {
    fn name(&self) -> &str {
        "spreadsheet"
    }

    fn description(&self) -> &str {
        "Spreadsheet formulas (SUM, AVERAGE, VLOOKUP): GPU parallel vs CPU sequential"
    }

    fn supported_sizes(&self) -> Vec<usize> {
        vec![
            100_000,     // 100K cells
            1_000_000,   // 1M cells (1000x1000)
            10_000_000,  // 10M cells
            100_000_000, // 100M cells
        ]
    }

    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator) {
        self.size = size;
        let (rows, cols) = Self::grid_dims(size);
        self.rows = rows;
        self.cols = cols;

        // Generate grid data: random f32 values in [0, 1)
        self.grid_data = gen.uniform_f32(rows * cols);

        // Generate sorted lookup keys (0..rows as float, evenly spaced)
        self.lookup_keys = (0..rows).map(|i| i as f32).collect();
        // Lookup values: random
        self.lookup_vals = gen.uniform_f32(rows);
        // Search keys: random indices as floats (within range of lookup keys)
        self.search_keys = gen.uniform_f32(rows)
            .iter()
            .map(|v| v * (rows as f32 - 1.0))
            .collect();

        // Allocate Metal buffers for SUM/AVERAGE
        self.buf_grid = Some(alloc_buffer_with_data(&ctx.device, &self.grid_data));
        self.buf_col_output = Some(alloc_buffer(
            &ctx.device,
            cols * std::mem::size_of::<f32>(),
        ));
        self.buf_avg_output = Some(alloc_buffer(
            &ctx.device,
            cols * std::mem::size_of::<f32>(),
        ));

        // Partials buffer for 2D row-chunked approach: num_row_chunks * cols
        let ss_rows_per_chunk = 64usize;
        let num_row_chunks = rows.div_ceil(ss_rows_per_chunk);
        self.buf_partials = Some(alloc_buffer(
            &ctx.device,
            num_row_chunks * cols * std::mem::size_of::<f32>(),
        ));

        let params = SpreadsheetParams {
            rows: rows as u32,
            cols: cols as u32,
            formula_type: 0, // SUM
            _pad: 0,
        };
        self.buf_params = Some(alloc_buffer_with_data(&ctx.device, &[params]));

        // Allocate Metal buffers for VLOOKUP
        self.buf_lookup_keys = Some(alloc_buffer_with_data(&ctx.device, &self.lookup_keys));
        self.buf_lookup_vals = Some(alloc_buffer_with_data(&ctx.device, &self.lookup_vals));
        self.buf_search_keys = Some(alloc_buffer_with_data(&ctx.device, &self.search_keys));
        self.buf_vlookup_output = Some(alloc_buffer(
            &ctx.device,
            rows * std::mem::size_of::<f32>(),
        ));

        let vlookup_params = SpreadsheetParams {
            rows: rows as u32,
            cols: cols as u32,
            formula_type: 2, // VLOOKUP
            _pad: 0,
        };
        self.buf_vlookup_params = Some(alloc_buffer_with_data(&ctx.device, &[vlookup_params]));

        // Pre-warm PSO cache
        self.pso_cache
            .get_or_create(ctx.library(), "spreadsheet_sum_v2");
        self.pso_cache
            .get_or_create(ctx.library(), "spreadsheet_sum_reduce");
        self.pso_cache
            .get_or_create(ctx.library(), "spreadsheet_avg_reduce");
        self.pso_cache
            .get_or_create(ctx.library(), "spreadsheet_vlookup");
    }

    fn run_gpu(&mut self, ctx: &MetalContext) -> f64 {
        let grid = self.buf_grid.as_ref().expect("setup not called");
        let partials = self.buf_partials.as_ref().expect("setup not called");
        let sum_out = self.buf_col_output.as_ref().expect("setup not called");
        let avg_out = self.buf_avg_output.as_ref().expect("setup not called");
        let prm = self.buf_params.as_ref().expect("setup not called");

        let ss_rows_per_chunk = 64usize;
        let num_row_chunks = self.rows.div_ceil(ss_rows_per_chunk);

        // Single command buffer for all kernels
        let cmd = ctx.queue.commandBuffer().expect("cmd buf");

        // --- Encoder 1: spreadsheet_sum_v2 (2D: columns x row_chunks → partials) ---
        {
            let pso = self.pso_cache.get_or_create(ctx.library(), "spreadsheet_sum_v2");
            let enc = cmd.computeCommandEncoder().expect("encoder");
            let tg_x = 256usize;
            dispatch_2d(
                &enc,
                pso,
                &[(grid.as_ref(), 0), (partials.as_ref(), 1), (prm.as_ref(), 2)],
                self.cols.div_ceil(tg_x),   // threadgroup count X
                num_row_chunks,              // threadgroup count Y
                tg_x.min(self.cols),         // threads per TG X
                1,                           // threads per TG Y
            );
            enc.endEncoding();
        }

        // --- Encoder 2: spreadsheet_sum_reduce (1D: partials → sum output) ---
        {
            let pso = self.pso_cache.get_or_create(ctx.library(), "spreadsheet_sum_reduce");
            let enc = cmd.computeCommandEncoder().expect("encoder");
            dispatch_1d(
                &enc,
                pso,
                &[(partials.as_ref(), 0), (sum_out.as_ref(), 1), (prm.as_ref(), 2)],
                self.cols,
            );
            enc.endEncoding();
        }

        // --- Encoder 3: spreadsheet_avg_reduce (1D: same partials → avg output) ---
        {
            let pso = self.pso_cache.get_or_create(ctx.library(), "spreadsheet_avg_reduce");
            let enc = cmd.computeCommandEncoder().expect("encoder");
            dispatch_1d(
                &enc,
                pso,
                &[(partials.as_ref(), 0), (avg_out.as_ref(), 1), (prm.as_ref(), 2)],
                self.cols,
            );
            enc.endEncoding();
        }

        // --- Encoder 4: VLOOKUP (unchanged) ---
        {
            let pso = self.pso_cache.get_or_create(ctx.library(), "spreadsheet_vlookup");
            let keys = self.buf_lookup_keys.as_ref().expect("setup not called");
            let vals = self.buf_lookup_vals.as_ref().expect("setup not called");
            let skeys = self.buf_search_keys.as_ref().expect("setup not called");
            let vout = self.buf_vlookup_output.as_ref().expect("setup not called");
            let vprm = self.buf_vlookup_params.as_ref().expect("setup not called");

            let enc = cmd.computeCommandEncoder().expect("encoder");
            dispatch_1d(
                &enc,
                pso,
                &[
                    (keys.as_ref(), 0),
                    (vals.as_ref(), 1),
                    (skeys.as_ref(), 2),
                    (vout.as_ref(), 3),
                    (vprm.as_ref(), 4),
                ],
                self.rows,
            );
            enc.endEncoding();
        }

        // Single commit + wait
        cmd.commit();
        cmd.waitUntilCompleted();

        // Read back results
        self.gpu_col_sums = unsafe { read_buffer_slice::<f32>(sum_out.as_ref(), self.cols) };
        self.gpu_col_avgs = unsafe { read_buffer_slice::<f32>(avg_out.as_ref(), self.cols) };
        let vout = self.buf_vlookup_output.as_ref().expect("setup not called");
        self.gpu_vlookup = unsafe { read_buffer_slice::<f32>(vout.as_ref(), self.rows) };

        GpuTimer::elapsed_ms(&cmd).unwrap_or(0.0)
    }

    fn run_cpu(&mut self) -> f64 {
        let timer = BenchTimer::start();

        self.cpu_col_sums = Self::cpu_column_sums(&self.grid_data, self.rows, self.cols);
        self.cpu_col_avgs = Self::cpu_column_averages(&self.grid_data, self.rows, self.cols);
        self.cpu_vlookup = Self::cpu_vlookup(&self.lookup_keys, &self.lookup_vals, &self.search_keys);

        timer.stop()
    }

    fn validate(&self) -> Result<(), String> {
        // Validate column SUM
        if self.gpu_col_sums.len() != self.cpu_col_sums.len() {
            return Err(format!(
                "SUM size mismatch: GPU {} vs CPU {}",
                self.gpu_col_sums.len(),
                self.cpu_col_sums.len()
            ));
        }

        for (i, (gpu, cpu)) in self
            .gpu_col_sums
            .iter()
            .zip(self.cpu_col_sums.iter())
            .enumerate()
        {
            let abs_err = (*gpu as f64 - *cpu as f64).abs();
            let denom = (*cpu as f64).abs().max(1e-8);
            let rel_err = abs_err / denom;
            if rel_err > 1e-3 {
                return Err(format!(
                    "SUM col {} mismatch: GPU={:.6} CPU={:.6} rel_err={:.6e}",
                    i, gpu, cpu, rel_err
                ));
            }
        }

        // Validate column AVERAGE
        for (i, (gpu, cpu)) in self
            .gpu_col_avgs
            .iter()
            .zip(self.cpu_col_avgs.iter())
            .enumerate()
        {
            let abs_err = (*gpu as f64 - *cpu as f64).abs();
            let denom = (*cpu as f64).abs().max(1e-8);
            let rel_err = abs_err / denom;
            if rel_err > 1e-3 {
                return Err(format!(
                    "AVERAGE col {} mismatch: GPU={:.6} CPU={:.6} rel_err={:.6e}",
                    i, gpu, cpu, rel_err
                ));
            }
        }

        // Validate VLOOKUP
        if self.gpu_vlookup.len() != self.cpu_vlookup.len() {
            return Err(format!(
                "VLOOKUP size mismatch: GPU {} vs CPU {}",
                self.gpu_vlookup.len(),
                self.cpu_vlookup.len()
            ));
        }

        let mut mismatches = 0;
        for (i, (gpu, cpu)) in self
            .gpu_vlookup
            .iter()
            .zip(self.cpu_vlookup.iter())
            .enumerate()
        {
            if (*gpu - *cpu).abs() > 1e-5 {
                mismatches += 1;
                if mismatches <= 3 {
                    eprintln!(
                        "  VLOOKUP mismatch at {}: GPU={:.6} CPU={:.6} key={:.6}",
                        i, gpu, cpu, self.search_keys[i]
                    );
                }
            }
        }

        if mismatches > 0 {
            return Err(format!(
                "VLOOKUP: {} mismatches out of {} lookups",
                mismatches,
                self.gpu_vlookup.len()
            ));
        }

        Ok(())
    }

    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        let seconds = elapsed_ms / 1000.0;

        // Total cells processed = grid cells (SUM + AVERAGE) + VLOOKUP rows
        // SUM touches rows*cols, AVERAGE touches rows*cols, VLOOKUP touches rows*log(rows)
        let total_cells = (self.rows * self.cols * 2 + self.rows) as f64;
        let cells_per_sec = if seconds > 0.0 {
            total_cells / seconds
        } else {
            0.0
        };

        m.insert("cells_per_sec".to_string(), cells_per_sec);
        m.insert("total_cells".to_string(), total_cells);
        m.insert("grid_cells".to_string(), size as f64);
        m.insert("rows".to_string(), self.rows as f64);
        m.insert("cols".to_string(), self.cols as f64);

        // Bandwidth: read grid twice (SUM + AVG) + lookup arrays
        let bytes = (self.rows * self.cols * 2 + self.rows * 4) as f64 * 4.0;
        let gbs = if seconds > 0.0 {
            bytes / seconds / 1e9
        } else {
            0.0
        };
        m.insert("gb_per_sec".to_string(), gbs);
        m.insert("bytes_processed".to_string(), bytes);

        m
    }
}
