//! ASCII roofline diagram showing bandwidth utilization for each experiment.
//!
//! Renders a simple horizontal bar chart:
//! ```text
//! reduce@10M   |████████████░░░░░░░░| 13.5% of 273 GB/s
//! scan@10M     |██████████████████░░| 45.2% of 273 GB/s
//! ```

use crate::harness::{format_size, DataPoint};
use forge_primitives::HardwareInfo;

const BAR_WIDTH: usize = 30;

/// Print an ASCII roofline diagram for all results.
pub fn print_roofline(data: &[DataPoint], hardware: &HardwareInfo) {
    // Only show experiments that have bandwidth data
    let with_bw: Vec<&DataPoint> = data
        .iter()
        .filter(|dp| dp.metrics.contains_key("bw_utilization_pct"))
        .collect();

    if with_bw.is_empty() {
        return;
    }

    println!("\n  BANDWIDTH ROOFLINE ({} @ {} GB/s)", hardware.chip_name, hardware.bandwidth_gbs);
    println!("  {}", "-".repeat(60));

    // Find max label width for alignment
    let max_label: usize = with_bw
        .iter()
        .map(|dp| {
            let label = format!("{}@{}", dp.experiment, format_size(dp.size));
            label.len()
        })
        .max()
        .unwrap_or(10);

    for dp in &with_bw {
        let bw_pct = dp
            .metrics
            .get("bw_utilization_pct")
            .copied()
            .unwrap_or(0.0);
        let gb_per_sec = dp.metrics.get("gb_per_sec").copied().unwrap_or(0.0);

        let label = format!("{}@{}", dp.experiment, format_size(dp.size));

        // Compute filled/empty portions of bar
        let fill_count = ((bw_pct / 100.0) * BAR_WIDTH as f64)
            .round()
            .min(BAR_WIDTH as f64) as usize;
        let empty_count = BAR_WIDTH - fill_count;

        let filled: String = "\u{2588}".repeat(fill_count);
        let empty: String = "\u{2591}".repeat(empty_count);

        // Color based on utilization
        let color = if bw_pct >= 50.0 {
            "\x1b[32m" // green
        } else if bw_pct >= 20.0 {
            "\x1b[36m" // cyan
        } else if bw_pct >= 5.0 {
            "\x1b[33m" // yellow
        } else {
            "\x1b[31m" // red
        };
        let reset = "\x1b[0m";

        println!(
            "  {:<width$}  |{}{}{}{}| {:>5.1}% ({:.1} GB/s)",
            label,
            color,
            filled,
            reset,
            empty,
            bw_pct,
            gb_per_sec,
            width = max_label,
        );
    }

    println!();
}
