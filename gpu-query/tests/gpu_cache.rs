//! Integration tests for the scan cache in QueryExecutor.
//!
//! Tests: cache hit on repeated query, FIFO eviction at 8 entries,
//! compound filter single scan via cache, and file-change invalidation.
//!
//! These require a Metal GPU device (Apple Silicon integration tests).

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

/// Helper: run a SQL query against a CSV directory using a given executor.
fn run_query_with(
    executor: &mut QueryExecutor,
    dir: &Path,
    sql: &str,
) -> gpu_query::gpu::executor::QueryResult {
    let cat = catalog::scan_directory(dir).expect("scan directory");
    let logical_plan = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let physical_plan = gpu_query::sql::physical_plan::plan(&logical_plan).expect("plan SQL");
    executor.execute(&physical_plan, &cat).expect("execute query")
}

// ============================================================
// test_scan_cache_hit -- same table queried twice, cache has 1 entry
// ============================================================

#[test]
fn test_scan_cache_hit() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n1,100\n2,200\n3,300\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let mut executor = QueryExecutor::new().expect("create executor");
    assert_eq!(executor.scan_cache_len(), 0, "cache starts empty");

    // First query scans the table
    let r1 = run_query_with(&mut executor, tmp.path(), "SELECT count(*) FROM sales");
    assert_eq!(r1.rows[0][0], "3");
    assert_eq!(executor.scan_cache_len(), 1, "one entry cached after first query");

    // Second query should hit cache (still 1 entry, not 2)
    let r2 = run_query_with(
        &mut executor,
        tmp.path(),
        "SELECT sum(amount) FROM sales",
    );
    assert_eq!(r2.rows[0][0], "600");
    assert_eq!(
        executor.scan_cache_len(),
        1,
        "still one entry -- cache hit, no duplicate"
    );
}

// ============================================================
// test_scan_cache_eviction -- 9 distinct tables, oldest evicted at 8
// ============================================================

#[test]
fn test_scan_cache_eviction() {
    let tmp = TempDir::new().unwrap();

    // Create 9 distinct CSV tables
    for i in 0..9 {
        let name = format!("t{}.csv", i);
        let csv = format!("id,val\n1,{}\n2,{}\n", i * 10, i * 10 + 1);
        make_csv(tmp.path(), &name, &csv);
    }

    let mut executor = QueryExecutor::new().expect("create executor");

    // Query tables 0..8 (9 tables total)
    for i in 0..9 {
        let sql = format!("SELECT count(*) FROM t{}", i);
        let result = run_query_with(&mut executor, tmp.path(), &sql);
        assert_eq!(result.rows[0][0], "2", "each table has 2 rows");
    }

    // FIFO eviction at capacity 8: after inserting 9th, oldest was evicted
    assert_eq!(
        executor.scan_cache_len(),
        8,
        "cache capped at 8 entries after 9 inserts"
    );
}

// ============================================================
// test_compound_filter_single_scan -- AND query uses cache, single scan
// ============================================================

#[test]
fn test_compound_filter_single_scan() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,50,10\n\
               2,150,5\n\
               3,200,20\n\
               4,300,3\n\
               5,250,15\n";
    make_csv(tmp.path(), "data.csv", csv);

    let mut executor = QueryExecutor::new().expect("create executor");

    // Compound AND filter: both predicates reference the same table
    // The scan cache should deduplicate -- only 1 scan entry for 'data'
    let result = run_query_with(
        &mut executor,
        tmp.path(),
        "SELECT count(*) FROM data WHERE amount > 100 AND quantity > 10",
    );
    // amount > 100 AND quantity > 10 => rows 3 (200,20) and 5 (250,15)
    assert_eq!(result.rows[0][0], "2");

    // Only one scan cache entry for 'data' (not two)
    assert_eq!(
        executor.scan_cache_len(),
        1,
        "compound filter used cache -- single scan entry"
    );
}

// ============================================================
// test_scan_cache_invalidation_on_file_change -- modify file, verify re-scan
// ============================================================

#[test]
fn test_scan_cache_invalidation_on_file_change() {
    let tmp = TempDir::new().unwrap();
    let csv_v1 = "id,amount\n1,100\n2,200\n3,300\n";
    make_csv(tmp.path(), "sales.csv", csv_v1);

    let mut executor = QueryExecutor::new().expect("create executor");

    // First query: populates cache
    let r1 = run_query_with(
        &mut executor,
        tmp.path(),
        "SELECT sum(amount) FROM sales",
    );
    assert_eq!(r1.rows[0][0], "600", "sum before file change");
    assert_eq!(executor.scan_cache_len(), 1);

    // Modify the file: add a row, changing both size and mtime
    // Small sleep to ensure mtime differs (filesystem granularity)
    std::thread::sleep(std::time::Duration::from_millis(100));
    let csv_v2 = "id,amount\n1,100\n2,200\n3,300\n4,400\n";
    make_csv(tmp.path(), "sales.csv", csv_v2);

    // Second query: should detect stale cache entry and re-scan
    let r2 = run_query_with(
        &mut executor,
        tmp.path(),
        "SELECT sum(amount) FROM sales",
    );
    assert_eq!(r2.rows[0][0], "1000", "sum after file change reflects new data");
    assert_eq!(
        executor.scan_cache_len(),
        1,
        "still 1 entry -- stale replaced"
    );
}
