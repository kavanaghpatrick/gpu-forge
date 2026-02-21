//! Measurement harness: runs experiments through warmup + measured loops,
//! collects statistics, and returns structured DataPoint results.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use forge_primitives::{HardwareInfo, MetalContext};

use crate::data_gen::DataGenerator;
use crate::experiments::Experiment;
use crate::stats::{compute_stats, Stats};

/// A single benchmark data point for one experiment at one size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub experiment: String,
    pub size: usize,
    pub gpu_stats: Stats,
    pub cpu_stats: Stats,
    pub speedup: f64,
    pub metrics: HashMap<String, f64>,
}

/// Configuration for a benchmark run.
pub struct BenchConfig {
    pub sizes: Vec<usize>,
    pub runs: u32,
    pub warmup: u32,
}

/// Run an experiment across all configured sizes.
///
/// For each size:
/// 1. setup(ctx, size, gen)
/// 2. validate() -- correctness check
/// 3. warmup loop (discard timings)
/// 4. measured loop (GPU + CPU)
/// 5. compute_stats -> DataPoint
pub fn run_experiment(
    exp: &mut dyn Experiment,
    config: &BenchConfig,
    ctx: &MetalContext,
    hardware: &HardwareInfo,
    progress_cb: Option<&dyn Fn(&str)>,
) -> Vec<DataPoint> {
    let mut results = Vec::new();
    let mut gen = DataGenerator::new(42);

    for &size in &config.sizes {
        let size_label = format_size(size);
        if let Some(cb) = progress_cb {
            cb(&format!("{} @ {}: setup", exp.name(), size_label));
        }

        // Setup data and buffers
        exp.setup(ctx, size, &mut gen);

        // Validate correctness (run both GPU and CPU once first)
        let _ = exp.run_gpu(ctx);
        let _ = exp.run_cpu();
        if let Err(e) = exp.validate() {
            eprintln!(
                "WARNING: {} @ {} validation failed: {}",
                exp.name(),
                size_label,
                e
            );
            // Continue anyway -- report the data point with a note
        }

        // Warmup loop (discard timings)
        if let Some(cb) = progress_cb {
            cb(&format!(
                "{} @ {}: warmup ({} runs)",
                exp.name(),
                size_label,
                config.warmup
            ));
        }
        for _ in 0..config.warmup {
            let _ = exp.run_gpu(ctx);
        }

        // Measured GPU loop
        if let Some(cb) = progress_cb {
            cb(&format!(
                "{} @ {}: measuring GPU ({} runs)",
                exp.name(),
                size_label,
                config.runs
            ));
        }
        let gpu_times: Vec<f64> = (0..config.runs).map(|_| exp.run_gpu(ctx)).collect();

        // Measured CPU loop
        if let Some(cb) = progress_cb {
            cb(&format!(
                "{} @ {}: measuring CPU ({} runs)",
                exp.name(),
                size_label,
                config.runs
            ));
        }
        let cpu_times: Vec<f64> = (0..config.runs).map(|_| exp.run_cpu()).collect();

        // Compute statistics
        let gpu_stats = compute_stats(&gpu_times);
        let cpu_stats = compute_stats(&cpu_times);

        // Speedup = CPU_mean / GPU_mean
        let speedup = if gpu_stats.mean > 0.0 {
            cpu_stats.mean / gpu_stats.mean
        } else {
            0.0
        };

        // Compute metrics using GPU mean time
        let mut metrics = exp.metrics(gpu_stats.mean, size);

        // Add bandwidth utilization if available
        if let Some(&gbs) = metrics.get("gb_per_sec") {
            let bw_pct = hardware.bandwidth_utilization(gbs);
            metrics.insert("bw_utilization_pct".to_string(), bw_pct);
        }

        results.push(DataPoint {
            experiment: exp.name().to_string(),
            size,
            gpu_stats,
            cpu_stats,
            speedup,
            metrics,
        });

        if let Some(cb) = progress_cb {
            cb(&format!(
                "{} @ {}: done (speedup={:.1}x)",
                exp.name(),
                size_label,
                speedup
            ));
        }
    }

    results
}

/// Format a size number for display (e.g., 1000000 -> "1M").
pub fn format_size(size: usize) -> String {
    if size >= 1_000_000 && size.is_multiple_of(1_000_000) {
        format!("{}M", size / 1_000_000)
    } else if size >= 1_000 && size.is_multiple_of(1_000) {
        format!("{}K", size / 1_000)
    } else {
        format!("{}", size)
    }
}
