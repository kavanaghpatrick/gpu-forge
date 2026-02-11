//! End-to-end NDJSON query tests with golden file oracle.
//!
//! Each test creates NDJSON data in a temp directory, executes SQL through
//! the full pipeline (parse -> optimize -> plan -> GPU execute), and
//! verifies results.

use std::io::Write;
use std::path::Path;
use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

// ============================================================
// Helpers
// ============================================================

/// Write an NDJSON file to a directory.
fn write_ndjson(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("create ndjson file");
    f.write_all(content.as_bytes()).expect("write ndjson");
    f.flush().expect("flush ndjson");
}

/// Build NDJSON content from (id, amount, quantity) rows.
fn build_ndjson(rows: &[(i64, i64, i64)]) -> String {
    let mut s = String::new();
    for (id, amount, quantity) in rows {
        s.push_str(&format!(
            r#"{{"id":{},"amount":{},"quantity":{}}}"#,
            id, amount, quantity
        ));
        s.push('\n');
    }
    s
}

/// Run SQL through full pipeline.
fn run_query(dir: &Path, sql: &str) -> gpu_query::gpu::executor::QueryResult {
    let catalog = catalog::scan_directory(dir).expect("scan directory");
    let logical = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let optimized = gpu_query::sql::optimizer::optimize(logical);
    let physical = gpu_query::sql::physical_plan::plan(&optimized).expect("plan SQL");
    let mut executor = QueryExecutor::new().expect("create executor");
    executor
        .execute(&physical, &catalog)
        .expect("execute query")
}

fn assert_int(result: &gpu_query::gpu::executor::QueryResult, expected: i64) {
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], expected.to_string());
}

/// Standard 10-row dataset matching the CSV tests.
fn make_sales_10(dir: &Path) {
    let rows: Vec<(i64, i64, i64)> = vec![
        (1, 50, 2),
        (2, 150, 3),
        (3, 200, 1),
        (4, 75, 4),
        (5, 300, 2),
        (6, 25, 1),
        (7, 500, 5),
        (8, 100, 3),
        (9, 450, 2),
        (10, 80, 1),
    ];
    write_ndjson(dir, "sales.ndjson", &build_ndjson(&rows));
}

// ============================================================
// 1. Basic aggregates
// ============================================================

#[test]
fn json_count_star() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales");
    assert_int(&r, 10);
}

#[test]
fn json_sum() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT sum(amount) FROM sales");
    assert_int(&r, 1930);
}

#[test]
fn json_min() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT min(amount) FROM sales");
    assert_int(&r, 25);
}

#[test]
fn json_max() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT max(amount) FROM sales");
    assert_int(&r, 500);
}

#[test]
fn json_count_and_sum() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*), sum(amount) FROM sales");
    assert_eq!(r.rows[0][0], "10");
    assert_eq!(r.rows[0][1], "1930");
}

// ============================================================
// 2. WHERE clause
// ============================================================

#[test]
fn json_where_gt() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount > 100");
    assert_int(&r, 5);
}

#[test]
fn json_where_lt() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount < 100");
    assert_int(&r, 4);
}

#[test]
fn json_where_eq() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount = 100");
    assert_int(&r, 1);
}

#[test]
fn json_where_sum_filtered() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(
        tmp.path(),
        "SELECT sum(amount) FROM sales WHERE amount > 100",
    );
    assert_int(&r, 1600);
}

#[test]
fn json_where_min_filtered() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(
        tmp.path(),
        "SELECT min(amount) FROM sales WHERE amount > 100",
    );
    assert_int(&r, 150);
}

#[test]
fn json_where_cross_column() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(
        tmp.path(),
        "SELECT sum(amount) FROM sales WHERE quantity > 3",
    );
    assert_int(&r, 575);
}

// ============================================================
// 3. Compound predicates
// ============================================================

#[test]
fn json_where_and() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(
        tmp.path(),
        "SELECT count(*) FROM sales WHERE amount > 100 AND amount < 400",
    );
    assert_int(&r, 3);
}

#[test]
fn json_where_or() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(
        tmp.path(),
        "SELECT count(*) FROM sales WHERE amount < 50 OR amount > 400",
    );
    assert_int(&r, 3);
}

// ============================================================
// 4. Edge cases
// ============================================================

#[test]
fn json_single_row() {
    let tmp = TempDir::new().unwrap();
    write_ndjson(tmp.path(), "one.ndjson", &build_ndjson(&[(1, 42, 1)]));
    let r = run_query(tmp.path(), "SELECT count(*), sum(amount) FROM one");
    assert_eq!(r.rows[0][0], "1");
    assert_eq!(r.rows[0][1], "42");
}

#[test]
fn json_negative_values() {
    let tmp = TempDir::new().unwrap();
    let rows: Vec<(i64, i64, i64)> = vec![
        (1, -100, 1),
        (2, -50, 1),
        (3, 0, 1),
        (4, 50, 1),
        (5, 100, 1),
    ];
    write_ndjson(tmp.path(), "neg.ndjson", &build_ndjson(&rows));
    let r = run_query(tmp.path(), "SELECT sum(amount) FROM neg");
    assert_int(&r, 0);
}

#[test]
fn json_filter_no_matches() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount > 9999");
    assert_int(&r, 0);
}
