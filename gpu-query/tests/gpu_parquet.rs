//! Integration tests for Parquet reader with GPU query execution.
//!
//! Tests the full path: write Parquet file -> catalog scan -> SQL parse ->
//! GPU execution (filter/aggregate) -> verify results.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

// Helper: create a Parquet file with INT64 columns (id, amount, quantity).
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
    let mut writer = SerializedFileWriter::new(file, schema, props).expect("create parquet writer");

    let mut rg_writer = writer.next_row_group().expect("next row group");

    // Write id column
    {
        let mut col_writer = rg_writer.next_column().expect("next column").unwrap();
        col_writer
            .typed::<parquet::data_type::Int64Type>()
            .write_batch(ids, None, None)
            .expect("write ids");
        col_writer.close().expect("close column");
    }

    // Write amount column
    {
        let mut col_writer = rg_writer.next_column().expect("next column").unwrap();
        col_writer
            .typed::<parquet::data_type::Int64Type>()
            .write_batch(amounts, None, None)
            .expect("write amounts");
        col_writer.close().expect("close column");
    }

    // Write quantity column
    {
        let mut col_writer = rg_writer.next_column().expect("next column").unwrap();
        col_writer
            .typed::<parquet::data_type::Int64Type>()
            .write_batch(quantities, None, None)
            .expect("write quantities");
        col_writer.close().expect("close column");
    }

    rg_writer.close().expect("close row group");
    writer.close().expect("close writer");
}

/// Helper: run a SQL query against a Parquet file directory.
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
// Basic Parquet scan tests
// ============================================================

#[test]
fn test_parquet_count_star() {
    let tmp = TempDir::new().unwrap();
    let ids: Vec<i64> = (1..=10).collect();
    let amounts: Vec<i64> = (10..=100).step_by(10).collect();
    let quantities: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    make_parquet(tmp.path(), "sales.parquet", &ids, &amounts, &quantities);

    let result = run_query(tmp.path(), "SELECT count(*) FROM sales");
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "10");
}

#[test]
fn test_parquet_count_filtered() {
    let tmp = TempDir::new().unwrap();
    let ids: Vec<i64> = (1..=10).collect();
    let amounts: Vec<i64> = vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
    let quantities: Vec<i64> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    make_parquet(tmp.path(), "sales.parquet", &ids, &amounts, &quantities);

    let result = run_query(tmp.path(), "SELECT count(*) FROM sales WHERE amount > 500");
    assert_eq!(result.row_count, 1);
    // amount > 500: 600, 700, 800, 900, 1000 = 5 rows
    assert_eq!(result.rows[0][0], "5");
}

#[test]
fn test_parquet_sum() {
    let tmp = TempDir::new().unwrap();
    let ids: Vec<i64> = (1..=5).collect();
    let amounts: Vec<i64> = vec![100, 200, 300, 400, 500];
    let quantities: Vec<i64> = vec![10, 20, 30, 40, 50];
    make_parquet(tmp.path(), "sales.parquet", &ids, &amounts, &quantities);

    let result = run_query(tmp.path(), "SELECT sum(amount) FROM sales");
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "1500"); // 100+200+300+400+500
}

#[test]
fn test_parquet_sum_filtered() {
    let tmp = TempDir::new().unwrap();
    let ids: Vec<i64> = (1..=5).collect();
    let amounts: Vec<i64> = vec![100, 200, 300, 400, 500];
    let quantities: Vec<i64> = vec![10, 20, 30, 40, 50];
    make_parquet(tmp.path(), "sales.parquet", &ids, &amounts, &quantities);

    let result = run_query(
        tmp.path(),
        "SELECT sum(amount) FROM sales WHERE amount > 200",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "1200"); // 300+400+500
}

#[test]
fn test_parquet_count_and_sum() {
    let tmp = TempDir::new().unwrap();
    let ids: Vec<i64> = (1..=5).collect();
    let amounts: Vec<i64> = vec![100, 200, 300, 400, 500];
    let quantities: Vec<i64> = vec![10, 20, 30, 40, 50];
    make_parquet(tmp.path(), "sales.parquet", &ids, &amounts, &quantities);

    let result = run_query(
        tmp.path(),
        "SELECT count(*), sum(amount) FROM sales WHERE amount > 200",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "3"); // 300, 400, 500
    assert_eq!(result.rows[0][1], "1200"); // 300+400+500
}

#[test]
fn test_parquet_filter_on_different_column() {
    let tmp = TempDir::new().unwrap();
    let ids: Vec<i64> = (1..=5).collect();
    let amounts: Vec<i64> = vec![100, 200, 300, 400, 500];
    let quantities: Vec<i64> = vec![10, 20, 30, 40, 50];
    make_parquet(tmp.path(), "sales.parquet", &ids, &amounts, &quantities);

    // Filter on quantity, aggregate amount
    let result = run_query(
        tmp.path(),
        "SELECT sum(amount) FROM sales WHERE quantity > 30",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "900"); // amounts for qty>30: 400+500
}

#[test]
fn test_parquet_single_row() {
    let tmp = TempDir::new().unwrap();
    make_parquet(tmp.path(), "single.parquet", &[42], &[100], &[1]);

    let result = run_query(tmp.path(), "SELECT count(*), sum(amount) FROM single");
    assert_eq!(result.rows[0][0], "1");
    assert_eq!(result.rows[0][1], "100");
}

#[test]
fn test_parquet_larger_dataset() {
    let tmp = TempDir::new().unwrap();
    let n = 1000;
    let ids: Vec<i64> = (1..=n).collect();
    let amounts: Vec<i64> = (1..=n).map(|i| i * 10).collect();
    let quantities: Vec<i64> = (1..=n).map(|i| i % 100).collect();
    make_parquet(tmp.path(), "big.parquet", &ids, &amounts, &quantities);

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
fn test_parquet_negative_values() {
    let tmp = TempDir::new().unwrap();
    let ids: Vec<i64> = vec![1, 2, 3, 4, 5];
    let amounts: Vec<i64> = vec![-100, -50, 0, 50, 100];
    let quantities: Vec<i64> = vec![1, 1, 1, 1, 1];
    make_parquet(tmp.path(), "neg.parquet", &ids, &amounts, &quantities);

    let result = run_query(tmp.path(), "SELECT sum(amount) FROM neg");
    assert_eq!(result.rows[0][0], "0"); // -100-50+0+50+100 = 0

    let result = run_query(tmp.path(), "SELECT count(*) FROM neg WHERE amount > 0");
    assert_eq!(result.rows[0][0], "2"); // 50, 100
}

#[test]
fn test_parquet_eq_filter() {
    let tmp = TempDir::new().unwrap();
    let ids: Vec<i64> = vec![1, 2, 3, 4, 5];
    let amounts: Vec<i64> = vec![100, 200, 200, 300, 200];
    let quantities: Vec<i64> = vec![1, 2, 3, 4, 5];
    make_parquet(tmp.path(), "data.parquet", &ids, &amounts, &quantities);

    let result = run_query(tmp.path(), "SELECT count(*) FROM data WHERE amount = 200");
    assert_eq!(result.rows[0][0], "3"); // Three rows with amount=200
}
