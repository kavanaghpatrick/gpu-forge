//! Experiment trait and registry for GPU compute benchmarks.
//!
//! Each experiment implements the `Experiment` trait, providing GPU and CPU
//! implementations of the same computation for comparison.

pub mod compact;
pub mod duckdb;
pub mod filter;
pub mod gemm;
pub mod gemv;
pub mod groupby;
pub mod hash_join;
pub mod histogram;
pub mod json_parse;
pub mod pipeline;
pub mod reduce;
pub mod scan;
pub mod sort;
pub mod spreadsheet;
pub mod timeseries;

use std::collections::HashMap;

use forge_primitives::MetalContext;

use crate::data_gen::DataGenerator;

/// A benchmark experiment comparing GPU and CPU implementations.
pub trait Experiment {
    /// Short name used for CLI selection (e.g., "reduce").
    fn name(&self) -> &str;

    /// Human-readable description of the experiment.
    fn description(&self) -> &str;

    /// Supported element counts for this experiment.
    fn supported_sizes(&self) -> Vec<usize>;

    /// Prepare data and Metal buffers for the given size.
    fn setup(&mut self, ctx: &MetalContext, size: usize, gen: &mut DataGenerator);

    /// Run the GPU implementation. Returns elapsed time in milliseconds.
    fn run_gpu(&mut self, ctx: &MetalContext) -> f64;

    /// Run the CPU baseline. Returns elapsed time in milliseconds.
    fn run_cpu(&mut self) -> f64;

    /// Validate GPU result against CPU result.
    fn validate(&self) -> Result<(), String>;

    /// Compute performance metrics for the given elapsed time and size.
    fn metrics(&self, elapsed_ms: f64, size: usize) -> HashMap<String, f64>;
}

/// Registry of all available experiments.
pub fn all_experiments() -> Vec<Box<dyn Experiment>> {
    vec![
        Box::new(reduce::ReduceExperiment::new()),
        Box::new(histogram::HistogramExperiment::new()),
        Box::new(scan::ScanExperiment::new()),
        Box::new(compact::CompactExperiment::new()),
        Box::new(sort::SortExperiment::new()),
        Box::new(filter::FilterExperiment::new()),
        Box::new(groupby::GroupByExperiment::new()),
        Box::new(gemm::GemmExperiment::new()),
        Box::new(gemv::GemvExperiment::new()),
        Box::new(spreadsheet::SpreadsheetExperiment::new()),
        Box::new(timeseries::TimeSeriesExperiment::new()),
        Box::new(hash_join::HashJoinExperiment::new()),
        Box::new(json_parse::JsonParseExperiment::new()),
        Box::new(pipeline::PipelineExperiment::new()),
        Box::new(duckdb::DuckDbExperiment::new()),
    ]
}
