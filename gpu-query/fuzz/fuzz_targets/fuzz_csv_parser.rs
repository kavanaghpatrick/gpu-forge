//! Fuzz target for the CPU-side CSV parser.
//!
//! Feeds arbitrary bytes as CSV content to `parse_header` and
//! `infer_schema_from_csv`, ensuring neither panics on any input.
//! GPU kernels are NOT invoked (fuzz targets must run on CI without Metal).

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Skip very large inputs to keep iteration speed high.
    if data.len() > 1_048_576 {
        return;
    }

    // Write fuzz input to a temporary file (parse_header takes a path).
    let dir = std::env::temp_dir().join("gpu_query_fuzz_csv");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("input.csv");
    {
        let mut f = match std::fs::File::create(&path) {
            Ok(f) => f,
            Err(_) => return,
        };
        if f.write_all(data).is_err() {
            return;
        }
        if f.flush().is_err() {
            return;
        }
    }

    // Exercise CSV header parser -- must not panic.
    let _ = gpu_query::io::csv::parse_header(&path);

    // Exercise format detection -- must not panic.
    let _ = gpu_query::io::format_detect::detect_format(&path);

    // Clean up.
    let _ = std::fs::remove_file(&path);
});
