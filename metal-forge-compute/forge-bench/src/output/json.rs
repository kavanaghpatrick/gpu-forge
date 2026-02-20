//! JSON output for benchmark results.
//!
//! Serializes Vec<DataPoint> to JSON with a hardware info header.

use std::fs;
use std::path::Path;

use serde::Serialize;

use forge_primitives::HardwareInfo;

use crate::harness::DataPoint;

/// JSON output wrapper with hardware metadata.
#[derive(Serialize)]
struct JsonReport {
    hardware: HardwareHeader,
    timestamp: String,
    results: Vec<DataPoint>,
}

#[derive(Serialize)]
struct HardwareHeader {
    chip: String,
    bandwidth_gbs: f64,
}

/// Write benchmark results to a JSON file.
pub fn write_json(path: &str, data: &[DataPoint], hardware: &HardwareInfo) -> Result<(), String> {
    let report = JsonReport {
        hardware: HardwareHeader {
            chip: hardware.chip_name.clone(),
            bandwidth_gbs: hardware.bandwidth_gbs,
        },
        timestamp: chrono::Utc::now().to_rfc3339(),
        results: data.to_vec(),
    };

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("JSON serialization failed: {}", e))?;

    // Ensure parent directory exists
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
        }
    }

    fs::write(path, json).map_err(|e| format!("Failed to write {}: {}", path, e))?;

    println!("JSON results written to: {}", path);
    Ok(())
}
