//! Fuzz target for the CPU-side NDJSON parser.
//!
//! Feeds arbitrary bytes as NDJSON content to `parse_ndjson_header`,
//! ensuring it never panics on any input.
//! GPU kernels are NOT invoked (fuzz targets must run on CI without Metal).

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Skip very large inputs to keep iteration speed high.
    if data.len() > 1_048_576 {
        return;
    }

    // Write fuzz input to a temporary file (parse_ndjson_header takes a path).
    let dir = std::env::temp_dir().join("gpu_query_fuzz_json");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("input.ndjson");
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

    // Exercise NDJSON header parser -- must not panic.
    let _ = gpu_query::io::json::parse_ndjson_header(&path);

    // Exercise format detection -- must not panic.
    let _ = gpu_query::io::format_detect::detect_format(&path);

    // Clean up.
    let _ = std::fs::remove_file(&path);
});
