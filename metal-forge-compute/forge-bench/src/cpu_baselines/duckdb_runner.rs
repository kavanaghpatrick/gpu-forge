//! DuckDB CLI wrapper for running SQL queries via subprocess.
//!
//! Shells out to the `duckdb` CLI binary, passing SQL via stdin.
//! Handles the case where duckdb is not installed by returning None.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

/// Result of a DuckDB query execution.
pub struct DuckDbResult {
    /// Wall-clock time in milliseconds (includes startup overhead ~50ms).
    pub elapsed_ms: f64,
    /// Raw CSV output from DuckDB.
    pub output: String,
}

/// Check if duckdb CLI is available on PATH.
pub fn duckdb_available() -> bool {
    Command::new("duckdb")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run a SQL query against a CSV file using the duckdb CLI.
///
/// Returns None if duckdb is not installed or the query fails.
/// The query should reference the data via `read_csv('path')`.
pub fn run_duckdb_query(sql: &str) -> Option<DuckDbResult> {
    let start = Instant::now();

    let mut child = Command::new("duckdb")
        .arg("-csv")        // CSV output mode
        .arg("-noheader")   // Skip column headers
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    // Write SQL to stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(sql.as_bytes()).ok()?;
    }

    let output = child.wait_with_output().ok()?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("DuckDB query failed: {}", stderr);
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    Some(DuckDbResult {
        elapsed_ms,
        output: stdout,
    })
}

/// Run the pipeline-equivalent SQL query against a CSV file.
///
/// Query: SELECT key, SUM(value) FROM read_csv('path') WHERE key > threshold
///        GROUP BY key ORDER BY 2 DESC LIMIT K
///
/// Returns (elapsed_ms, Vec<(key, sum)>) or None if duckdb unavailable.
pub fn run_pipeline_query(
    csv_path: &Path,
    threshold: u32,
    top_k: usize,
) -> Option<(f64, Vec<(u32, f64)>)> {
    let sql = format!(
        "SELECT column0, SUM(column1) as s FROM read_csv('{}', header=false, columns={{'column0': 'INTEGER', 'column1': 'DOUBLE'}}) WHERE column0 > {} GROUP BY column0 ORDER BY s DESC LIMIT {};",
        csv_path.display(),
        threshold,
        top_k
    );

    let result = run_duckdb_query(&sql)?;

    // Parse CSV output: each line is "key,sum"
    let mut rows: Vec<(u32, f64)> = Vec::new();
    for line in result.output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let (Ok(key), Ok(sum)) = (parts[0].parse::<u32>(), parts[1].parse::<f64>()) {
                rows.push((key, sum));
            }
        }
    }

    Some((result.elapsed_ms, rows))
}

/// Write columnar data (keys, values) to a CSV file.
///
/// Format: key,value (no header) -- one row per element.
pub fn write_csv_data(path: &Path, keys: &[u32], values: &[f32]) -> std::io::Result<()> {
    use std::io::BufWriter;

    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    for (&k, &v) in keys.iter().zip(values.iter()) {
        writeln!(writer, "{},{}", k, v)?;
    }

    writer.flush()?;
    Ok(())
}
