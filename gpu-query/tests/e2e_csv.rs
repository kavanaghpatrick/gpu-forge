//! End-to-end CSV integration tests for DESCRIBE and GPU-parallel column statistics.
//!
//! Tests the execute_describe() pathway directly via Rust integration tests.
//! Uses temporary CSV files to verify per-column stats (count, null%, distinct, min, max, sample).

use tempfile::TempDir;

/// Create a temp directory with a sales CSV file containing known data.
fn make_sales_csv() -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().expect("create temp dir");
    let csv_path = dir.path().join("sales.csv");
    std::fs::write(
        &csv_path,
        "id,name,amount,region\n\
         1,Alice,500,East\n\
         2,Bob,200,West\n\
         3,Charlie,800,East\n\
         4,Diana,150,West\n\
         5,Eve,1000,East\n",
    )
    .expect("write csv");
    let dir_path = dir.path().to_path_buf();
    (dir, dir_path)
}

#[test]
fn test_describe() {
    let (_dir, dir_path) = make_sales_csv();

    // Scan catalog
    let catalog = gpu_query::io::catalog::scan_directory(&dir_path)
        .expect("scan directory should succeed");

    assert!(!catalog.is_empty(), "catalog should have at least one table");

    // Create executor and run DESCRIBE
    let mut executor =
        gpu_query::gpu::executor::QueryExecutor::new().expect("GPU executor should init");

    let describe_result = executor
        .execute_describe("sales", &catalog)
        .expect("execute_describe should succeed");

    // Verify table name
    assert_eq!(describe_result.table_name, "sales");

    // Verify column count (id, name, amount, region = 4 columns)
    assert_eq!(
        describe_result.columns.len(),
        4,
        "sales table should have 4 columns, got {}",
        describe_result.columns.len()
    );

    // Check column names
    let col_names: Vec<&str> = describe_result.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(col_names, &["id", "name", "amount", "region"]);

    // Check types
    let col_types: Vec<&str> = describe_result
        .columns
        .iter()
        .map(|c| c.data_type.as_str())
        .collect();
    assert_eq!(col_types[0], "INT64"); // id
    assert_eq!(col_types[1], "VARCHAR"); // name
    assert_eq!(col_types[2], "INT64"); // amount
    assert_eq!(col_types[3], "VARCHAR"); // region

    // Verify row counts (5 rows)
    for col in &describe_result.columns {
        assert_eq!(col.count, 5, "each column should have count=5, got {} for {}", col.count, col.name);
    }

    // Verify null percentages (no nulls in our test data)
    for col in &describe_result.columns {
        assert_eq!(col.null_count, 0, "no nulls expected for column {}", col.name);
    }

    // Check id column: INT64, distinct=5, min=1, max=5
    let id_col = &describe_result.columns[0];
    assert_eq!(id_col.distinct_count, 5);
    assert_eq!(id_col.min_value, "1");
    assert_eq!(id_col.max_value, "5");
    assert_eq!(id_col.sample_value, "1");

    // Check amount column: INT64, distinct=5, min=150, max=1000
    let amount_col = &describe_result.columns[2];
    assert_eq!(amount_col.distinct_count, 5);
    assert_eq!(amount_col.min_value, "150");
    assert_eq!(amount_col.max_value, "1000");

    // Check name column: VARCHAR, distinct=5
    let name_col = &describe_result.columns[1];
    assert_eq!(name_col.distinct_count, 5);

    // Check region column: VARCHAR, distinct=2 (East, West)
    let region_col = &describe_result.columns[3];
    assert_eq!(region_col.distinct_count, 2);

    // Verify we can format as QueryResult
    let qr = describe_result.to_query_result();
    assert_eq!(qr.columns.len(), 8); // column, type, count, null%, distinct, min, max, sample
    assert_eq!(qr.row_count, 4); // 4 columns
    assert_eq!(qr.rows.len(), 4);

    // First row should be id column stats
    assert_eq!(qr.rows[0][0], "id");
    assert_eq!(qr.rows[0][1], "INT64");
    assert_eq!(qr.rows[0][2], "5"); // count
    assert_eq!(qr.rows[0][3], "0%"); // null%
}
