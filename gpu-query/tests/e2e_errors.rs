//! End-to-end error handling tests.
//!
//! Tests that invalid SQL, missing tables, bad columns, and other error
//! conditions are properly caught and reported through the full pipeline.

use std::path::Path;
use tempfile::TempDir;

use gpu_query::io::catalog;

// ============================================================
// Helpers
// ============================================================

fn write_csv(dir: &Path, name: &str, content: &str) {
    std::fs::write(dir.join(name), content).expect("write csv");
}

/// Attempt to parse SQL. Returns Err message or Ok if parse succeeds.
fn try_parse(sql: &str) -> Result<(), String> {
    gpu_query::sql::parser::parse_query(sql)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Attempt full pipeline. Returns Err message or Ok if query succeeds.
fn try_query(dir: &Path, sql: &str) -> Result<gpu_query::gpu::executor::QueryResult, String> {
    let catalog = catalog::scan_directory(dir).map_err(|e| e.to_string())?;
    let logical = gpu_query::sql::parser::parse_query(sql).map_err(|e| e.to_string())?;
    let optimized = gpu_query::sql::optimizer::optimize(logical);
    let physical =
        gpu_query::sql::physical_plan::plan(&optimized).map_err(|e| format!("{:?}", e))?;
    let mut executor = gpu_query::gpu::executor::QueryExecutor::new().map_err(|e| e.to_string())?;
    executor.execute(&physical, &catalog)
}

fn make_test_dir() -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().unwrap();
    write_csv(
        dir.path(),
        "sales.csv",
        "id,amount,quantity\n1,100,10\n2,200,20\n3,300,30\n",
    );
    let p = dir.path().to_path_buf();
    (dir, p)
}

// ============================================================
// 1. Parse errors: invalid SQL
// ============================================================

#[test]
fn error_empty_sql() {
    let result = try_parse("");
    assert!(result.is_err(), "empty SQL should fail");
}

#[test]
fn error_gibberish() {
    let result = try_parse("NOT VALID SQL AT ALL");
    assert!(result.is_err(), "gibberish should fail");
}

#[test]
fn error_update_statement() {
    let result = try_parse("UPDATE sales SET amount = 0");
    assert!(result.is_err(), "UPDATE should be rejected");
}

#[test]
fn error_delete_statement() {
    let result = try_parse("DELETE FROM sales");
    assert!(result.is_err(), "DELETE should be rejected");
}

#[test]
fn error_insert_statement() {
    let result = try_parse("INSERT INTO sales VALUES (1, 2, 3)");
    assert!(result.is_err(), "INSERT should be rejected");
}

// ============================================================
// 2. Planning/execution errors: missing tables, bad columns
// ============================================================

#[test]
fn error_missing_table() {
    let (_d, p) = make_test_dir();
    let result = try_query(&p, "SELECT count(*) FROM nonexistent");
    assert!(result.is_err(), "missing table should fail");
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected error"),
    };
    assert!(
        err.contains("not found") || err.contains("nonexistent"),
        "error should mention missing table, got: {}",
        err
    );
}

#[test]
fn error_bad_column_in_where() {
    let (_d, p) = make_test_dir();
    let result = try_query(&p, "SELECT count(*) FROM sales WHERE nonexistent > 0");
    assert!(
        result.is_err(),
        "bad column in WHERE should fail: {:?}",
        result.as_ref().err()
    );
}

#[test]
fn error_bad_column_in_aggregate() {
    let (_d, p) = make_test_dir();
    let result = try_query(&p, "SELECT sum(nonexistent) FROM sales");
    assert!(
        result.is_err(),
        "bad column in aggregate should fail: {:?}",
        result.as_ref().err()
    );
}

#[test]
fn error_empty_directory() {
    let dir = TempDir::new().unwrap();
    let result = try_query(dir.path(), "SELECT count(*) FROM anything");
    assert!(result.is_err(), "empty directory should fail");
}

// ============================================================
// 3. Unsupported SQL features
// ============================================================

#[test]
fn error_join() {
    let result = try_parse("SELECT * FROM a JOIN b ON a.id = b.id");
    assert!(result.is_err(), "JOIN should be rejected");
}

#[test]
fn error_multiple_statements() {
    let result = try_parse("SELECT 1; SELECT 2");
    assert!(result.is_err(), "multiple statements should be rejected");
}
