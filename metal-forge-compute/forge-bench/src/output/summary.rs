//! Summary output with speedup verdicts and crossover-point analysis.
//!
//! Prints a compact summary table showing experiment results with
//! verdict classification based on speedup ranges:
//!   >10x = DOMINANT, >5x = STRONG, 2-5x = SOLID, 1-2x = MARGINAL, <1x = SLOWER
//!
//! Also provides crossover-point analysis: for each experiment, finds the
//! smallest N where GPU > CPU (speedup crosses 1.0x).

use std::collections::BTreeMap;

use crate::harness::{format_size, DataPoint};

/// Classify a speedup value into a verdict string.
fn verdict(speedup: f64) -> &'static str {
    if speedup > 10.0 {
        "DOMINANT"
    } else if speedup > 5.0 {
        "STRONG"
    } else if speedup >= 2.0 {
        "SOLID"
    } else if speedup >= 1.0 {
        "MARGINAL"
    } else {
        "SLOWER"
    }
}

/// ANSI color code for a verdict.
fn verdict_color(v: &str) -> &'static str {
    match v {
        "DOMINANT" => "\x1b[1;32m", // bold green
        "STRONG" => "\x1b[32m",     // green
        "SOLID" => "\x1b[36m",      // cyan
        "MARGINAL" => "\x1b[33m",   // yellow
        "SLOWER" => "\x1b[31m",     // red
        _ => "\x1b[0m",
    }
}

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

/// Print a summary table of all results with verdicts.
pub fn print_summary(data: &[DataPoint]) {
    if data.is_empty() {
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("  SUMMARY");
    println!("{}", "=".repeat(70));
    println!(
        "  {:<20} {:>10} {:>10} {:>10}  {}",
        "Experiment", "Size", "Speedup", "GPU (ms)", "Verdict"
    );
    println!("  {}", "-".repeat(66));

    for dp in data {
        let size_str = format_size(dp.size);
        let v = verdict(dp.speedup);
        let color = verdict_color(v);

        println!(
            "  {:<20} {:>10} {:>9.1}x {:>10.3}  {}{}{}",
            dp.experiment, size_str, dp.speedup, dp.gpu_stats.mean, color, v, RESET,
        );
    }

    println!("{}", "=".repeat(70));

    // Count verdicts
    let mut dominant = 0u32;
    let mut strong = 0u32;
    let mut solid = 0u32;
    let mut marginal = 0u32;
    let mut slower = 0u32;

    for dp in data {
        match verdict(dp.speedup) {
            "DOMINANT" => dominant += 1,
            "STRONG" => strong += 1,
            "SOLID" => solid += 1,
            "MARGINAL" => marginal += 1,
            "SLOWER" => slower += 1,
            _ => {}
        }
    }

    println!(
        "  DOMINANT: {}  STRONG: {}  SOLID: {}  MARGINAL: {}  SLOWER: {}",
        dominant, strong, solid, marginal, slower,
    );
    println!();
}

/// Crossover info for a single experiment.
pub struct CrossoverInfo {
    /// Experiment name
    pub experiment: String,
    /// Smallest size where GPU > CPU (speedup >= 1.0), or None if always slower
    pub crossover_size: Option<usize>,
    /// Best speedup observed across all sizes
    pub best_speedup: f64,
    /// Size at which best speedup was observed
    pub best_size: usize,
}

/// Analyze crossover points: for each experiment, find the smallest N where GPU > CPU.
///
/// Groups data points by experiment name, sorts by size, and finds the first
/// size where speedup crosses 1.0x (GPU becomes faster than CPU).
pub fn analyze_crossover(data: &[DataPoint]) -> Vec<CrossoverInfo> {
    // Group by experiment name, preserving insertion order
    let mut by_experiment: BTreeMap<String, Vec<&DataPoint>> = BTreeMap::new();
    for dp in data {
        by_experiment
            .entry(dp.experiment.clone())
            .or_default()
            .push(dp);
    }

    let mut results = Vec::new();
    for (name, mut points) in by_experiment {
        // Sort by size ascending
        points.sort_by_key(|p| p.size);

        // Find crossover: first size where speedup >= 1.0
        let crossover_size = points.iter().find(|p| p.speedup >= 1.0).map(|p| p.size);

        // Find best speedup
        let best = points
            .iter()
            .max_by(|a, b| {
                a.speedup
                    .partial_cmp(&b.speedup)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        results.push(CrossoverInfo {
            experiment: name,
            crossover_size,
            best_speedup: best.speedup,
            best_size: best.size,
        });
    }

    results
}

/// Print crossover-point analysis table.
pub fn print_crossover_analysis(data: &[DataPoint]) {
    let crossovers = analyze_crossover(data);
    if crossovers.is_empty() {
        return;
    }

    println!("{}", "=".repeat(70));
    println!("  CROSSOVER ANALYSIS (smallest N where GPU > CPU)");
    println!("{}", "=".repeat(70));
    println!(
        "  {:<20} {:>15} {:>12} {:>12}",
        "Experiment", "Crossover @", "Best Speedup", "Best @"
    );
    println!("  {}", "-".repeat(63));

    for info in &crossovers {
        let crossover_str = match info.crossover_size {
            Some(size) => format_size(size),
            None => "never".to_string(),
        };

        let v = verdict(info.best_speedup);
        let color = verdict_color(v);

        println!(
            "  {:<20} {:>15} {:>10.1}x {} {:>10}{}",
            info.experiment,
            crossover_str,
            info.best_speedup,
            color,
            format_size(info.best_size),
            RESET,
        );
    }

    println!("{}", "=".repeat(70));

    // Count experiments that are GPU-favorable
    let gpu_wins = crossovers
        .iter()
        .filter(|c| c.crossover_size.is_some())
        .count();
    let total = crossovers.len();
    let always_slower = total - gpu_wins;

    println!(
        "  GPU wins: {}/{}  Always slower: {}",
        gpu_wins, total, always_slower,
    );
    println!();
}

/// Print suite-level summary after running all experiments.
///
/// Shows overall statistics: total experiments, total data points,
/// best/worst speedups, and a recommendation for which operations to prioritize.
pub fn print_suite_summary(data: &[DataPoint]) {
    if data.is_empty() {
        return;
    }

    // Collect unique experiment names
    let mut experiments: Vec<String> = data.iter().map(|d| d.experiment.clone()).collect();
    experiments.sort();
    experiments.dedup();

    let total_experiments = experiments.len();
    let total_points = data.len();

    // Find best and worst speedups
    let best = data
        .iter()
        .max_by(|a, b| {
            a.speedup
                .partial_cmp(&b.speedup)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    let worst = data
        .iter()
        .min_by(|a, b| {
            a.speedup
                .partial_cmp(&b.speedup)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    // Average speedup across all data points
    let avg_speedup: f64 = data.iter().map(|d| d.speedup).sum::<f64>() / data.len() as f64;

    // GPU-favorable experiments (at least one size where speedup >= 1.0)
    let gpu_favorable: Vec<&str> = experiments
        .iter()
        .filter(|name| {
            data.iter()
                .any(|d| d.experiment == **name && d.speedup >= 1.0)
        })
        .map(|s| s.as_str())
        .collect();

    // Dominant experiments (any size with speedup > 10x)
    let dominant_exps: Vec<&str> = experiments
        .iter()
        .filter(|name| {
            data.iter()
                .any(|d| d.experiment == **name && d.speedup > 10.0)
        })
        .map(|s| s.as_str())
        .collect();

    println!("{}{}SUITE SUMMARY{}", BOLD, "=".repeat(5).as_str(), RESET);
    println!("{}", "=".repeat(70));
    println!("  Experiments run:    {}", total_experiments);
    println!("  Data points:        {}", total_points);
    println!("  Average speedup:    {:.1}x", avg_speedup);
    println!(
        "  Best speedup:       {:.1}x ({} @ {})",
        best.speedup,
        best.experiment,
        format_size(best.size)
    );
    println!(
        "  Worst speedup:      {:.1}x ({} @ {})",
        worst.speedup,
        worst.experiment,
        format_size(worst.size)
    );
    println!(
        "  GPU favorable:      {}/{} experiments",
        gpu_favorable.len(),
        total_experiments
    );

    if !dominant_exps.is_empty() {
        println!("  Dominant (>10x):    {}", dominant_exps.join(", "));
    }

    println!("{}", "=".repeat(70));

    // Priority recommendation
    if !dominant_exps.is_empty() {
        println!(
            "\n  PRIORITY: Invest in GPU acceleration for: {}",
            dominant_exps.join(", ")
        );
    }
    if gpu_favorable.len() < total_experiments {
        let cpu_better: Vec<&str> = experiments
            .iter()
            .filter(|name| {
                !data
                    .iter()
                    .any(|d| d.experiment == **name && d.speedup >= 1.0)
            })
            .map(|s| s.as_str())
            .collect();
        if !cpu_better.is_empty() {
            println!("  CPU preferred for: {}", cpu_better.join(", "));
        }
    }
    println!();
}
