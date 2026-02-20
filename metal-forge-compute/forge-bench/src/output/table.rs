//! Table output using comfy-table.
//!
//! Renders benchmark results as a formatted ASCII table with columns:
//! Size | GPU (ms) | CPU (ms) | Speedup | GB/s | BW% | CV%

use comfy_table::{Attribute, Cell, CellAlignment, Color, ContentArrangement, Table};

use crate::harness::{format_size, DataPoint};

/// Render a vector of DataPoints as a comfy-table.
pub fn render_table(data: &[DataPoint]) {
    if data.is_empty() {
        println!("No results to display.");
        return;
    }

    let experiment_name = &data[0].experiment;

    let mut table = Table::new();
    table
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Size").add_attribute(Attribute::Bold),
            Cell::new("GPU (ms)").add_attribute(Attribute::Bold),
            Cell::new("CPU (ms)").add_attribute(Attribute::Bold),
            Cell::new("Speedup").add_attribute(Attribute::Bold),
            Cell::new("GB/s").add_attribute(Attribute::Bold),
            Cell::new("BW%").add_attribute(Attribute::Bold),
            Cell::new("CV%").add_attribute(Attribute::Bold),
        ]);

    for dp in data {
        let size_str = format_size(dp.size);
        let gpu_ms = format!("{:.3}", dp.gpu_stats.mean);
        let cpu_ms = format!("{:.3}", dp.cpu_stats.mean);

        let speedup_str = format!("{:.1}x", dp.speedup);
        let speedup_cell = if dp.speedup >= 5.0 {
            Cell::new(&speedup_str).fg(Color::Green)
        } else if dp.speedup >= 2.0 {
            Cell::new(&speedup_str).fg(Color::Cyan)
        } else if dp.speedup >= 1.0 {
            Cell::new(&speedup_str).fg(Color::Yellow)
        } else {
            Cell::new(&speedup_str).fg(Color::Red)
        };

        let gbs = dp
            .metrics
            .get("gb_per_sec")
            .map(|v| format!("{:.1}", v))
            .unwrap_or_else(|| "-".to_string());

        let bw_pct = dp
            .metrics
            .get("bw_utilization_pct")
            .map(|v| format!("{:.1}", v))
            .unwrap_or_else(|| "-".to_string());

        let cv = format!("{:.1}", dp.gpu_stats.cv_percent);

        table.add_row(vec![
            Cell::new(&size_str).set_alignment(CellAlignment::Right),
            Cell::new(&gpu_ms).set_alignment(CellAlignment::Right),
            Cell::new(&cpu_ms).set_alignment(CellAlignment::Right),
            speedup_cell.set_alignment(CellAlignment::Right),
            Cell::new(&gbs).set_alignment(CellAlignment::Right),
            Cell::new(&bw_pct).set_alignment(CellAlignment::Right),
            Cell::new(&cv).set_alignment(CellAlignment::Right),
        ]);
    }

    println!("\n=== {} ===", experiment_name);
    println!("{table}");
}

/// Render results grouped by experiment name.
pub fn render_all_tables(data: &[DataPoint]) {
    if data.is_empty() {
        println!("No results to display.");
        return;
    }

    // Group by experiment
    let mut groups: Vec<(String, Vec<&DataPoint>)> = Vec::new();
    for dp in data {
        if let Some(group) = groups.iter_mut().find(|(name, _)| name == &dp.experiment) {
            group.1.push(dp);
        } else {
            groups.push((dp.experiment.clone(), vec![dp]));
        }
    }

    for (_, points) in &groups {
        let owned: Vec<DataPoint> = points.iter().map(|p| (*p).clone()).collect();
        render_table(&owned);
    }
}
