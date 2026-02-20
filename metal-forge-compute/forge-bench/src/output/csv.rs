//! CSV output for benchmark results.
//!
//! Writes DataPoint vec to CSV file with columns:
//! experiment,size,gpu_mean_ms,cpu_mean_ms,speedup,gb_per_sec,bw_pct,cv_pct

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::DataPoint;

/// Write benchmark results to a CSV file.
pub fn write_csv(path: &str, data: &[DataPoint]) -> Result<(), String> {
    // Ensure parent directory exists
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
        }
    }

    let mut file =
        fs::File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;

    // Header
    writeln!(
        file,
        "experiment,size,gpu_mean_ms,cpu_mean_ms,speedup,gb_per_sec,bw_pct,cv_pct"
    )
    .map_err(|e| format!("Write error: {}", e))?;

    // Data rows
    for dp in data {
        let gb_per_sec = dp.metrics.get("gb_per_sec").copied().unwrap_or(0.0);
        let bw_pct = dp.metrics.get("bw_utilization_pct").copied().unwrap_or(0.0);

        writeln!(
            file,
            "{},{},{:.4},{:.4},{:.2},{:.2},{:.1},{:.1}",
            dp.experiment,
            dp.size,
            dp.gpu_stats.mean,
            dp.cpu_stats.mean,
            dp.speedup,
            gb_per_sec,
            bw_pct,
            dp.gpu_stats.cv_percent,
        )
        .map_err(|e| format!("Write error: {}", e))?;
    }

    println!("CSV results written to: {}", path);
    Ok(())
}
