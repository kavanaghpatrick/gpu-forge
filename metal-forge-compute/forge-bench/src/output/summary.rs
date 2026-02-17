//! Summary output with speedup verdicts.
//!
//! Prints a compact summary table showing experiment results with
//! verdict classification based on speedup ranges:
//!   >10x = DOMINANT, >5x = STRONG, 2-5x = SOLID, 1-2x = MARGINAL, <1x = SLOWER

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
