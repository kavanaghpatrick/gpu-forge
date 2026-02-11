//! End-to-end Parquet query tests with golden file oracle.
//!
//! Each test writes a Parquet file, executes SQL through the full pipeline
//! (parse -> optimize -> plan -> GPU execute), and verifies results.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

// ============================================================
// Helpers
// ============================================================

/// Create a Parquet file with 3 INT64 columns (id, amount, quantity).
fn make_parquet(dir: &Path, name: &str, ids: &[i64], amounts: &[i64], quantities: &[i64]) {
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;

    let schema_str = "
        message test_schema {
            REQUIRED INT64 id;
            REQUIRED INT64 amount;
            REQUIRED INT64 quantity;
        }
    ";
    let schema = Arc::new(parse_message_type(schema_str).expect("parse schema"));
    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .build(),
    );

    let path = dir.join(name);
    let file = File::create(&path).expect("create parquet file");
    let mut writer =
        SerializedFileWriter::new(file, schema, props).expect("create parquet writer");

    let mut rg = writer.next_row_group().expect("next row group");

    // Write id column
    {
        let mut col = rg.next_column().expect("next column").unwrap();
        col.typed::<parquet::data_type::Int64Type>()
            .write_batch(ids, None, None)
            .expect("write ids");
        col.close().expect("close column");
    }
    // Write amount column
    {
        let mut col = rg.next_column().expect("next column").unwrap();
        col.typed::<parquet::data_type::Int64Type>()
            .write_batch(amounts, None, None)
            .expect("write amounts");
        col.close().expect("close column");
    }
    // Write quantity column
    {
        let mut col = rg.next_column().expect("next column").unwrap();
        col.typed::<parquet::data_type::Int64Type>()
            .write_batch(quantities, None, None)
            .expect("write quantities");
        col.close().expect("close column");
    }

    rg.close().expect("close row group");
    writer.close().expect("close writer");
}

/// Run SQL through full pipeline: parse -> optimize -> plan -> execute.
fn run_query(dir: &Path, sql: &str) -> gpu_query::gpu::executor::QueryResult {
    let catalog = catalog::scan_directory(dir).expect("scan directory");
    let logical = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let optimized = gpu_query::sql::optimizer::optimize(logical);
    let physical = gpu_query::sql::physical_plan::plan(&optimized).expect("plan SQL");
    let mut executor = QueryExecutor::new().expect("create executor");
    executor.execute(&physical, &catalog).expect("execute query")
}

fn assert_int(result: &gpu_query::gpu::executor::QueryResult, expected: i64) {
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], expected.to_string());
}

/// Standard 10-row dataset.
fn make_sales_10(dir: &Path) {
    let ids: Vec<i64> = (1..=10).collect();
    let amounts: Vec<i64> = vec![50, 150, 200, 75, 300, 25, 500, 100, 450, 80];
    let quantities: Vec<i64> = vec![2, 3, 1, 4, 2, 1, 5, 3, 2, 1];
    make_parquet(dir, "sales.parquet", &ids, &amounts, &quantities);
}

// ============================================================
// 1. Basic aggregates
// ============================================================

#[test]
fn parquet_count_star() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales");
    assert_int(&r, 10);
}

#[test]
fn parquet_sum() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT sum(amount) FROM sales");
    assert_int(&r, 1930);
}

#[test]
fn parquet_min() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT min(amount) FROM sales");
    assert_int(&r, 25);
}

#[test]
fn parquet_max() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT max(amount) FROM sales");
    assert_int(&r, 500);
}

#[test]
fn parquet_count_and_sum() {
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
fn parquet_where_gt() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount > 100");
    // >100: 150,200,300,500,450 = 5
    assert_int(&r, 5);
}

#[test]
fn parquet_where_lt() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount < 100");
    // <100: 50,75,25,80 = 4
    assert_int(&r, 4);
}

#[test]
fn parquet_where_eq() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount = 100");
    assert_int(&r, 1);
}

#[test]
fn parquet_where_ne() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount != 100");
    assert_int(&r, 9);
}

#[test]
fn parquet_where_sum_filtered() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT sum(amount) FROM sales WHERE amount > 100");
    // 150+200+300+500+450 = 1600
    assert_int(&r, 1600);
}

#[test]
fn parquet_where_min_filtered() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT min(amount) FROM sales WHERE amount > 100");
    assert_int(&r, 150);
}

#[test]
fn parquet_where_cross_column() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT sum(amount) FROM sales WHERE quantity > 3");
    // qty>3: row4(75,4), row7(500,5) => 575
    assert_int(&r, 575);
}

// ============================================================
// 3. Compound predicates
// ============================================================

#[test]
fn parquet_where_and() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount > 100 AND amount < 400");
    // 150,200,300 = 3
    assert_int(&r, 3);
}

#[test]
fn parquet_where_or() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount < 50 OR amount > 400");
    // <50: 25. >400: 500,450 => 3
    assert_int(&r, 3);
}

// ============================================================
// 4. Edge cases
// ============================================================

#[test]
fn parquet_single_row() {
    let tmp = TempDir::new().unwrap();
    make_parquet(tmp.path(), "one.parquet", &[1], &[42], &[1]);
    let r = run_query(tmp.path(), "SELECT count(*), sum(amount) FROM one");
    assert_eq!(r.rows[0][0], "1");
    assert_eq!(r.rows[0][1], "42");
}

#[test]
fn parquet_negative_values() {
    let tmp = TempDir::new().unwrap();
    let ids: Vec<i64> = vec![1, 2, 3, 4, 5];
    let amounts: Vec<i64> = vec![-100, -50, 0, 50, 100];
    let quantities: Vec<i64> = vec![1, 1, 1, 1, 1];
    make_parquet(tmp.path(), "neg.parquet", &ids, &amounts, &quantities);
    let r = run_query(tmp.path(), "SELECT sum(amount) FROM neg");
    assert_int(&r, 0);
}

#[test]
fn parquet_filter_no_matches() {
    let tmp = TempDir::new().unwrap();
    make_sales_10(tmp.path());
    let r = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount > 9999");
    assert_int(&r, 0);
}

#[test]
fn parquet_large_dataset() {
    let tmp = TempDir::new().unwrap();
    let n = 1000i64;
    let ids: Vec<i64> = (1..=n).collect();
    let amounts: Vec<i64> = (1..=n).map(|i| i * 10).collect();
    let quantities: Vec<i64> = (1..=n).map(|i| i % 100).collect();
    make_parquet(tmp.path(), "big.parquet", &ids, &amounts, &quantities);

    let r = run_query(tmp.path(), "SELECT count(*) FROM big");
    assert_int(&r, 1000);

    // sum = 10*(1000*1001/2) = 5005000
    let r = run_query(tmp.path(), "SELECT sum(amount) FROM big");
    assert_int(&r, 5005000);
}
