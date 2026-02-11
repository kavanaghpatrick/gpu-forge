//! Integration tests for GPU NDJSON parser with query execution.
//!
//! Tests the full path: NDJSON file -> catalog scan -> SQL parse ->
//! GPU execution (structural index + field extraction + filter/aggregate)
//! -> verify results.

use std::io::Write;
use std::path::Path;

use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

/// Helper: create an NDJSON file in a directory.
fn make_ndjson(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("create ndjson file");
    f.write_all(content.as_bytes()).expect("write ndjson");
    f.flush().expect("flush ndjson");
}

/// Helper: build NDJSON content from rows of (id, amount, quantity).
fn build_ndjson(rows: &[(i64, i64, i64)]) -> String {
    let mut content = String::new();
    for (id, amount, quantity) in rows {
        content.push_str(&format!(
            r#"{{"id":{},"amount":{},"quantity":{}}}"#,
            id, amount, quantity
        ));
        content.push('\n');
    }
    content
}

/// Helper: run a SQL query against an NDJSON file directory.
fn run_query(dir: &Path, sql: &str) -> gpu_query::gpu::executor::QueryResult {
    let catalog = catalog::scan_directory(dir).expect("scan directory");
    let logical_plan = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let physical_plan = gpu_query::sql::physical_plan::plan(&logical_plan).expect("plan SQL");
    let mut executor = QueryExecutor::new().expect("create executor");
    executor
        .execute(&physical_plan, &catalog)
        .expect("execute query")
}

// ============================================================
// Basic NDJSON scan tests
// ============================================================

#[test]
fn test_json_count_star() {
    let tmp = TempDir::new().unwrap();
    let rows: Vec<(i64, i64, i64)> = (1..=10).map(|i| (i, i * 100, i * 10)).collect();
    let content = build_ndjson(&rows);
    make_ndjson(tmp.path(), "sales.ndjson", &content);

    let result = run_query(tmp.path(), "SELECT count(*) FROM sales");
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "10");
}

#[test]
fn test_json_count_filtered() {
    let tmp = TempDir::new().unwrap();
    let rows: Vec<(i64, i64, i64)> = vec![
        (1, 100, 10),
        (2, 200, 20),
        (3, 300, 30),
        (4, 400, 40),
        (5, 500, 50),
        (6, 600, 60),
        (7, 700, 70),
        (8, 800, 80),
        (9, 900, 90),
        (10, 1000, 100),
    ];
    let content = build_ndjson(&rows);
    make_ndjson(tmp.path(), "sales.ndjson", &content);

    let result = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount > 500");
    assert_eq!(result.row_count, 1);
    // amount > 500: 600, 700, 800, 900, 1000 = 5 rows
    assert_eq!(result.rows[0][0], "5");
}

#[test]
fn test_json_sum() {
    let tmp = TempDir::new().unwrap();
    let rows: Vec<(i64, i64, i64)> = vec![
        (1, 100, 10),
        (2, 200, 20),
        (3, 300, 30),
        (4, 400, 40),
        (5, 500, 50),
    ];
    let content = build_ndjson(&rows);
    make_ndjson(tmp.path(), "sales.ndjson", &content);

    let result = run_query(tmp.path(), "SELECT sum(amount) FROM sales");
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "1500"); // 100+200+300+400+500
}

#[test]
fn test_json_sum_filtered() {
    let tmp = TempDir::new().unwrap();
    let rows: Vec<(i64, i64, i64)> = vec![
        (1, 100, 10),
        (2, 200, 20),
        (3, 300, 30),
        (4, 400, 40),
        (5, 500, 50),
    ];
    let content = build_ndjson(&rows);
    make_ndjson(tmp.path(), "sales.ndjson", &content);

    let result = run_query(
        tmp.path(),
        "SELECT sum(amount) FROM sales WHERE amount > 200",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "1200"); // 300+400+500
}

#[test]
fn test_json_count_and_sum() {
    let tmp = TempDir::new().unwrap();
    let rows: Vec<(i64, i64, i64)> = vec![
        (1, 100, 10),
        (2, 200, 20),
        (3, 300, 30),
        (4, 400, 40),
        (5, 500, 50),
    ];
    let content = build_ndjson(&rows);
    make_ndjson(tmp.path(), "sales.ndjson", &content);

    let result = run_query(
        tmp.path(),
        "SELECT count(*), sum(amount) FROM sales WHERE amount > 200",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "3"); // 300, 400, 500
    assert_eq!(result.rows[0][1], "1200"); // 300+400+500
}

#[test]
fn test_json_filter_on_different_column() {
    let tmp = TempDir::new().unwrap();
    let rows: Vec<(i64, i64, i64)> = vec![
        (1, 100, 10),
        (2, 200, 20),
        (3, 300, 30),
        (4, 400, 40),
        (5, 500, 50),
    ];
    let content = build_ndjson(&rows);
    make_ndjson(tmp.path(), "sales.ndjson", &content);

    // Filter on quantity, aggregate amount
    let result = run_query(
        tmp.path(),
        "SELECT sum(amount) FROM sales WHERE quantity > 30",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "900"); // amounts for qty>30: 400+500
}

#[test]
fn test_json_single_row() {
    let tmp = TempDir::new().unwrap();
    let content = build_ndjson(&[(42, 100, 1)]);
    make_ndjson(tmp.path(), "single.ndjson", &content);

    let result = run_query(tmp.path(), "SELECT count(*), sum(amount) FROM single");
    assert_eq!(result.rows[0][0], "1");
    assert_eq!(result.rows[0][1], "100");
}

#[test]
fn test_json_larger_dataset() {
    let tmp = TempDir::new().unwrap();
    let n = 1000i64;
    let rows: Vec<(i64, i64, i64)> = (1..=n).map(|i| (i, i * 10, i % 100)).collect();
    let content = build_ndjson(&rows);
    make_ndjson(tmp.path(), "big.ndjson", &content);

    let result = run_query(tmp.path(), "SELECT count(*) FROM big");
    assert_eq!(result.rows[0][0], "1000");

    // Sum of 1..=1000 * 10 = 10 * (1000 * 1001 / 2) = 5005000
    let result = run_query(tmp.path(), "SELECT sum(amount) FROM big");
    assert_eq!(result.rows[0][0], "5005000");

    // Filter: amount > 5000 means i*10 > 5000, so i > 500, i.e. 501..=1000 = 500 rows
    let result = run_query(tmp.path(), "SELECT count(*) FROM big WHERE amount > 5000");
    assert_eq!(result.rows[0][0], "500");
}

#[test]
fn test_json_negative_values() {
    let tmp = TempDir::new().unwrap();
    let rows: Vec<(i64, i64, i64)> = vec![
        (1, -100, 1),
        (2, -50, 1),
        (3, 0, 1),
        (4, 50, 1),
        (5, 100, 1),
    ];
    let content = build_ndjson(&rows);
    make_ndjson(tmp.path(), "neg.ndjson", &content);

    let result = run_query(tmp.path(), "SELECT sum(amount) FROM neg");
    assert_eq!(result.rows[0][0], "0"); // -100-50+0+50+100 = 0

    let result = run_query(tmp.path(), "SELECT count(*) FROM neg WHERE amount > 0");
    assert_eq!(result.rows[0][0], "2"); // 50, 100
}

#[test]
fn test_json_eq_filter() {
    let tmp = TempDir::new().unwrap();
    let rows: Vec<(i64, i64, i64)> = vec![
        (1, 100, 1),
        (2, 200, 2),
        (3, 200, 3),
        (4, 300, 4),
        (5, 200, 5),
    ];
    let content = build_ndjson(&rows);
    make_ndjson(tmp.path(), "data.ndjson", &content);

    let result = run_query(tmp.path(), "SELECT count(*) FROM data WHERE amount = 200");
    assert_eq!(result.rows[0][0], "3"); // Three rows with amount=200
}
