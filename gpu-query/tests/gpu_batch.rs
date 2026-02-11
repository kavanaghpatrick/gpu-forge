//! Integration tests for batched query execution.
//!
//! Tests the batched execution path: large files are split into chunks,
//! each chunk processed independently through the GPU pipeline, and partial
//! results merged on CPU.
//!
//! Uses a small batch threshold (e.g., 50 bytes) to force batching on small
//! test CSV files, then verifies merged results match single-pass results.

use std::io::Write;
use std::path::Path;

use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

/// Helper: create a CSV file in a directory.
fn make_csv(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("create csv file");
    f.write_all(content.as_bytes()).expect("write csv");
    f.flush().expect("flush csv");
}

/// Helper: run a SQL query with standard (non-batched) execution.
fn run_query(dir: &Path, sql: &str) -> gpu_query::gpu::executor::QueryResult {
    let catalog = catalog::scan_directory(dir).expect("scan directory");
    let logical_plan = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let physical_plan = gpu_query::sql::physical_plan::plan(&logical_plan).expect("plan SQL");
    let mut executor = QueryExecutor::new().expect("create executor");
    executor
        .execute(&physical_plan, &catalog)
        .expect("execute query")
}

/// Helper: run a SQL query with batched execution at a given threshold.
fn run_batched_query(
    dir: &Path,
    sql: &str,
    batch_threshold: usize,
) -> gpu_query::gpu::executor::QueryResult {
    let catalog = catalog::scan_directory(dir).expect("scan directory");
    let logical_plan = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let physical_plan = gpu_query::sql::physical_plan::plan(&logical_plan).expect("plan SQL");
    let mut executor = QueryExecutor::new().expect("create executor");
    executor
        .execute_with_batching(&physical_plan, &catalog, batch_threshold)
        .expect("execute batched query")
}

// ============================================================
// Test: Batched COUNT(*) matches non-batched
// ============================================================

#[test]
fn test_batched_count_matches_non_batched() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,100\n\
               2,200\n\
               3,300\n\
               4,400\n\
               5,500\n\
               6,600\n\
               7,700\n\
               8,800\n\
               9,900\n\
               10,1000\n";
    make_csv(tmp.path(), "data.csv", csv);

    let sql = "SELECT count(*) FROM data";
    let normal = run_query(tmp.path(), sql);
    // Force batching with tiny threshold (50 bytes) -- file is ~100 bytes
    let batched = run_batched_query(tmp.path(), sql, 50);

    assert_eq!(
        normal.rows[0][0], batched.rows[0][0],
        "Batched count should match non-batched count"
    );
    assert_eq!(batched.rows[0][0], "10");
}

// ============================================================
// Test: Batched SUM matches non-batched
// ============================================================

#[test]
fn test_batched_sum_matches_non_batched() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,100\n\
               2,200\n\
               3,300\n\
               4,400\n\
               5,500\n\
               6,600\n\
               7,700\n\
               8,800\n\
               9,900\n\
               10,1000\n";
    make_csv(tmp.path(), "data.csv", csv);

    let sql = "SELECT sum(amount) FROM data";
    let normal = run_query(tmp.path(), sql);
    let batched = run_batched_query(tmp.path(), sql, 50);

    assert_eq!(
        normal.rows[0][0], batched.rows[0][0],
        "Batched sum should match non-batched sum"
    );
    assert_eq!(batched.rows[0][0], "5500");
}

// ============================================================
// Test: Batched COUNT + SUM with filter matches non-batched
// ============================================================

#[test]
fn test_batched_filtered_aggregate_matches() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,100\n\
               2,200\n\
               3,300\n\
               4,400\n\
               5,500\n\
               6,600\n\
               7,700\n\
               8,800\n\
               9,900\n\
               10,1000\n";
    make_csv(tmp.path(), "data.csv", csv);

    let sql = "SELECT count(*), sum(amount) FROM data WHERE amount > 500";
    let normal = run_query(tmp.path(), sql);
    let batched = run_batched_query(tmp.path(), sql, 50);

    assert_eq!(
        normal.rows[0][0], batched.rows[0][0],
        "Batched filtered count should match"
    );
    assert_eq!(
        normal.rows[0][1], batched.rows[0][1],
        "Batched filtered sum should match"
    );
    // amount > 500: rows 600,700,800,900,1000 -> count=5, sum=4000
    assert_eq!(batched.rows[0][0], "5");
    assert_eq!(batched.rows[0][1], "4000");
}

// ============================================================
// Test: Non-batched path when file is small (threshold large)
// ============================================================

#[test]
fn test_no_batching_when_below_threshold() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,100\n\
               2,200\n\
               3,300\n";
    make_csv(tmp.path(), "data.csv", csv);

    let sql = "SELECT count(*), sum(amount) FROM data";
    // Use large threshold -- should fall through to standard execution
    let result = run_batched_query(tmp.path(), sql, 1024 * 1024);

    assert_eq!(result.rows[0][0], "3");
    assert_eq!(result.rows[0][1], "600");
}

// ============================================================
// Test: Batched execution with very small batch size (1 row per chunk)
// ============================================================

#[test]
fn test_batched_tiny_threshold() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,val\n\
               1,10\n\
               2,20\n\
               3,30\n\
               4,40\n\
               5,50\n";
    make_csv(tmp.path(), "data.csv", csv);

    let sql = "SELECT count(*), sum(val) FROM data";
    // Threshold of 10 bytes: each ~5-byte row gets its own chunk
    let batched = run_batched_query(tmp.path(), sql, 10);

    assert_eq!(batched.rows[0][0], "5");
    assert_eq!(batched.rows[0][1], "150");
}

// ============================================================
// Test: Batched MIN/MAX aggregation
// ============================================================

#[test]
fn test_batched_min_max() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,val\n\
               1,50\n\
               2,10\n\
               3,90\n\
               4,30\n\
               5,70\n\
               6,20\n\
               7,80\n\
               8,40\n\
               9,60\n\
               10,100\n";
    make_csv(tmp.path(), "data.csv", csv);

    let sql = "SELECT min(val), max(val) FROM data";
    let normal = run_query(tmp.path(), sql);
    let batched = run_batched_query(tmp.path(), sql, 30);

    assert_eq!(
        normal.rows[0][0], batched.rows[0][0],
        "Batched min should match"
    );
    assert_eq!(
        normal.rows[0][1], batched.rows[0][1],
        "Batched max should match"
    );
    assert_eq!(batched.rows[0][0], "10");
    assert_eq!(batched.rows[0][1], "100");
}

// ============================================================
// Test: Batched AVG aggregation
// ============================================================

#[test]
fn test_batched_avg() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,val\n\
               1,10\n\
               2,20\n\
               3,30\n\
               4,40\n";
    make_csv(tmp.path(), "data.csv", csv);

    let sql = "SELECT avg(val) FROM data";
    let normal = run_query(tmp.path(), sql);
    let batched = run_batched_query(tmp.path(), sql, 20);

    assert_eq!(
        normal.rows[0][0], batched.rows[0][0],
        "Batched avg should match non-batched"
    );
    // avg(10,20,30,40) = 25
    assert_eq!(batched.rows[0][0], "25");
}

// ============================================================
// Test: Batched with all aggregate functions combined
// ============================================================

#[test]
fn test_batched_all_aggregates() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,val\n\
               1,100\n\
               2,200\n\
               3,300\n\
               4,400\n\
               5,500\n";
    make_csv(tmp.path(), "data.csv", csv);

    let sql = "SELECT count(*), sum(val), min(val), max(val), avg(val) FROM data";
    let normal = run_query(tmp.path(), sql);
    let batched = run_batched_query(tmp.path(), sql, 30);

    for i in 0..5 {
        assert_eq!(
            normal.rows[0][i], batched.rows[0][i],
            "Function {} mismatch: normal={} batched={}",
            i, normal.rows[0][i], batched.rows[0][i]
        );
    }
}

// ============================================================
// Test: needs_batching static method
// ============================================================

#[test]
fn test_needs_batching_detection() {
    assert!(!QueryExecutor::needs_batching(100, 1024));
    assert!(QueryExecutor::needs_batching(1025, 1024));
    assert!(!QueryExecutor::needs_batching(
        gpu_query::gpu::executor::BATCH_SIZE_BYTES,
        gpu_query::gpu::executor::BATCH_SIZE_BYTES
    ));
    assert!(QueryExecutor::needs_batching(
        gpu_query::gpu::executor::BATCH_SIZE_BYTES + 1,
        gpu_query::gpu::executor::BATCH_SIZE_BYTES
    ));
}
