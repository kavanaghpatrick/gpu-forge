//! Experiment trait and registry for GPU compute benchmarks.
//!
//! Each experiment implements the `Experiment` trait, providing GPU and CPU
//! implementations of the same computation for comparison.

pub mod atomic_contention;
pub mod atomic_throughput;
pub mod ballot_compact;
pub mod bank_conflicts;
pub mod branch_diverge;
pub mod byte_search;
pub mod cache_probe;
pub mod compact;
pub mod cross_tg;
pub mod dispatch_overhead;
pub mod duckdb;
pub mod dynamic_cache;
pub mod encoder_reuse;
pub mod filter;
pub mod fp16_advantage;
pub mod fsm_runtime;
pub mod gemm;
pub mod gemv;
pub mod gpu_scheduler;
pub mod groupby;
pub mod hash_join;
pub mod histogram;
pub mod ilp_width;
pub mod indirect_cond;
pub mod json_parse;
pub mod mem_compute_overlap;
pub mod occupancy_sweep;
pub mod pipeline;
pub mod reduce;
pub mod register_pressure;
pub mod scan;
pub mod setbytes_lut;
pub mod simd_matmul;
pub mod simd_pipeline;
pub mod simd_register_lut;
pub mod simd_sort32;
pub mod simd_taskqueue;
pub mod simd_vs_tg;
pub mod slc_lockfree;
pub mod slc_persistence;
pub mod slc_residency;
pub mod sort;
pub mod spreadsheet;
pub mod stride_bandwidth;
pub mod texture_interp;
pub mod tg_gradient;
pub mod tick_chain;
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
        Box::new(simd_vs_tg::SimdVsTgExperiment::new()),
        Box::new(slc_residency::SlcResidencyExperiment::new()),
        Box::new(dynamic_cache::DynamicCacheExperiment::new()),
        Box::new(indirect_cond::IndirectCondExperiment::new()),
        Box::new(fp16_advantage::Fp16AdvantageExperiment::new()),
        Box::new(occupancy_sweep::OccupancySweepExperiment::new()),
        Box::new(setbytes_lut::SetBytesLutExperiment::new()),
        Box::new(tg_gradient::TgGradientExperiment::new()),
        Box::new(encoder_reuse::EncoderReuseExperiment::new()),
        // Wave 3: Practical kernel design patterns
        Box::new(atomic_contention::AtomicContentionExperiment::new()),
        Box::new(stride_bandwidth::StrideBandwidthExperiment::new()),
        Box::new(branch_diverge::BranchDivergeExperiment::new()),
        // Wave 4: Creative Metal tricks
        Box::new(ballot_compact::BallotCompactExperiment::new()),
        Box::new(simd_register_lut::SimdRegisterLutExperiment::new()),
        Box::new(byte_search::ByteSearchExperiment::new()),
        Box::new(texture_interp::TextureInterpExperiment::new()),
        // Wave 5: Hardware probing & SIMD-as-CPU
        Box::new(register_pressure::RegisterPressureExperiment::new()),
        Box::new(ilp_width::IlpWidthExperiment::new()),
        Box::new(cache_probe::CacheProbeExperiment::new()),
        Box::new(simd_pipeline::SimdPipelineExperiment::new()),
        Box::new(simd_sort32::SimdSort32Experiment::new()),
        Box::new(simd_taskqueue::SimdTaskqueueExperiment::new()),
        // Wave 6: Micro-architecture deep probes
        Box::new(slc_persistence::SlcPersistenceExperiment::new()),
        Box::new(dispatch_overhead::DispatchOverheadExperiment::new()),
        Box::new(mem_compute_overlap::MemComputeOverlapExperiment::new()),
        Box::new(bank_conflicts::BankConflictsExperiment::new()),
        Box::new(simd_matmul::SimdMatmulExperiment::new()),
        Box::new(atomic_throughput::AtomicThroughputExperiment::new()),
        Box::new(cross_tg::CrossTgExperiment::new()),
        // Wave 7: GPU OS Primitives
        Box::new(slc_lockfree::SlcLockfreeExperiment::new()),
        Box::new(gpu_scheduler::GpuSchedulerExperiment::new()),
        Box::new(tick_chain::TickChainExperiment::new()),
        Box::new(fsm_runtime::FsmRuntimeExperiment::new()),
    ]
}
