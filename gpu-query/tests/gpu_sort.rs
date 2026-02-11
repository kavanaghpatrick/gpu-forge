//! Integration tests for ORDER BY (CPU-side sort via GpuSort plan node).
//!
//! Tests: ORDER BY ASC/DESC on INT64 column, ORDER BY with WHERE filter,
//! ORDER BY with LIMIT, multi-row result ordering verification.
//!
//! Uses the full pipeline: CSV file -> catalog scan -> SQL parse ->
//! GPU execution (scan + optional filter + CPU sort) -> verify results.

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

/// Helper: run a SQL query against a CSV file directory.
fn run_query(dir: &Path, sql: &str) -> gpu_query::gpu::executor::QueryResult {
    let catalog = catalog::scan_directory(dir).expect("scan directory");
    let logical_plan = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let physical_plan =
        gpu_query::sql::physical_plan::plan(&logical_plan).expect("plan SQL");
    let mut executor = QueryExecutor::new().expect("create executor");
    executor
        .execute(&physical_plan, &catalog)
        .expect("execute query")
}

// ============================================================
// ORDER BY ASC on INT64 column
// ============================================================

#[test]
fn test_order_by_asc_int64() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,500,10\n\
               2,100,20\n\
               3,300,30\n\
               4,200,40\n\
               5,400,50\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 5);
    // Amounts should be sorted ascending: 100, 200, 300, 400, 500
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["100", "200", "300", "400", "500"]);
}

// ============================================================
// ORDER BY DESC on INT64 column
// ============================================================

#[test]
fn test_order_by_desc_int64() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,500,10\n\
               2,100,20\n\
               3,300,30\n\
               4,200,40\n\
               5,400,50\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount DESC");
    assert_eq!(result.row_count, 5);
    // Amounts should be sorted descending: 500, 400, 300, 200, 100
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["500", "400", "300", "200", "100"]);
}

// ============================================================
// ORDER BY with WHERE filter
// ============================================================

#[test]
fn test_order_by_with_filter() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,500,10\n\
               2,100,20\n\
               3,300,30\n\
               4,200,40\n\
               5,400,50\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(
        tmp.path(),
        "SELECT * FROM sales WHERE amount > 200 ORDER BY amount ASC",
    );
    // amount > 200: 500, 300, 400 -> sorted ASC: 300, 400, 500
    assert_eq!(result.row_count, 3);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["300", "400", "500"]);
}

// ============================================================
// ORDER BY with LIMIT
// ============================================================

#[test]
fn test_order_by_with_limit() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,500,10\n\
               2,100,20\n\
               3,300,30\n\
               4,200,40\n\
               5,400,50\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(
        tmp.path(),
        "SELECT * FROM sales ORDER BY amount DESC LIMIT 3",
    );
    // All 5 rows sorted DESC by amount: 500, 400, 300, 200, 100 -> top 3: 500, 400, 300
    assert_eq!(result.row_count, 3);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["500", "400", "300"]);
}

// ============================================================
// ORDER BY verifies all columns are preserved
// ============================================================

#[test]
fn test_order_by_preserves_all_columns() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,300,30\n\
               2,100,10\n\
               3,200,20\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 3);
    // Sorted by amount ASC: (2,100,10), (3,200,20), (1,300,30)
    assert_eq!(result.rows[0], vec!["2", "100", "10"]);
    assert_eq!(result.rows[1], vec!["3", "200", "20"]);
    assert_eq!(result.rows[2], vec!["1", "300", "30"]);
}

// ============================================================
// ORDER BY on first column (id)
// ============================================================

#[test]
fn test_order_by_id_column() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               5,500,50\n\
               3,300,30\n\
               1,100,10\n\
               4,400,40\n\
               2,200,20\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY id ASC");
    assert_eq!(result.row_count, 5);
    let ids: Vec<&str> = result.rows.iter().map(|r| r[0].as_str()).collect();
    assert_eq!(ids, vec!["1", "2", "3", "4", "5"]);
}

// ============================================================
// ORDER BY with negative values
// ============================================================

#[test]
fn test_order_by_negative_values() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,-100,10\n\
               2,200,20\n\
               3,-50,30\n\
               4,0,40\n\
               5,100,50\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 5);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["-100", "-50", "0", "100", "200"]);
}

// ============================================================
// ORDER BY with filter and LIMIT combined
// ============================================================

#[test]
fn test_order_by_filter_and_limit() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,100,10\n\
               2,200,20\n\
               3,300,30\n\
               4,400,40\n\
               5,500,50\n\
               6,600,60\n\
               7,700,70\n\
               8,800,80\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(
        tmp.path(),
        "SELECT * FROM sales WHERE amount > 300 ORDER BY amount DESC LIMIT 3",
    );
    // amount > 300: 400, 500, 600, 700, 800 -> DESC: 800, 700, 600, 500, 400 -> top 3
    assert_eq!(result.row_count, 3);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["800", "700", "600"]);
}

// ============================================================
// ORDER BY single row
// ============================================================

#[test]
fn test_order_by_single_row() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               42,999,1\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0], vec!["42", "999", "1"]);
}

// ============================================================
// ORDER BY with duplicate values
// ============================================================

#[test]
fn test_order_by_duplicate_values() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,200,10\n\
               2,100,20\n\
               3,200,30\n\
               4,100,40\n\
               5,300,50\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 5);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    // 100, 100, 200, 200, 300
    assert_eq!(amounts, vec!["100", "100", "200", "200", "300"]);
}

// ============================================================
// ORDER BY returns correct column names
// ============================================================

#[test]
fn test_order_by_column_names() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,100,10\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY id ASC");
    assert_eq!(result.columns, vec!["id", "amount", "quantity"]);
}

// ============================================================
// ORDER BY on already sorted data
// ============================================================

#[test]
fn test_order_by_already_sorted_asc() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,100\n\
               2,200\n\
               3,300\n\
               4,400\n\
               5,500\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 5);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["100", "200", "300", "400", "500"]);
}

#[test]
fn test_order_by_already_sorted_desc() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               5,500\n\
               4,400\n\
               3,300\n\
               2,200\n\
               1,100\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount DESC");
    assert_eq!(result.row_count, 5);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["500", "400", "300", "200", "100"]);
}

// ============================================================
// ORDER BY reverse sorted input
// ============================================================

#[test]
fn test_order_by_reverse_input_asc() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               5,500\n\
               4,400\n\
               3,300\n\
               2,200\n\
               1,100\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 5);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["100", "200", "300", "400", "500"]);
}

// ============================================================
// ORDER BY all same values
// ============================================================

#[test]
fn test_order_by_all_same_values() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,100\n\
               2,100\n\
               3,100\n\
               4,100\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 4);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["100", "100", "100", "100"]);
}

// ============================================================
// ORDER BY two elements
// ============================================================

#[test]
fn test_order_by_two_elements_asc() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,200\n\
               2,100\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 2);
    assert_eq!(result.rows[0][1], "100");
    assert_eq!(result.rows[1][1], "200");
}

#[test]
fn test_order_by_two_elements_desc() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,100\n\
               2,200\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount DESC");
    assert_eq!(result.row_count, 2);
    assert_eq!(result.rows[0][1], "200");
    assert_eq!(result.rows[1][1], "100");
}

// ============================================================
// ORDER BY larger dataset
// ============================================================

#[test]
fn test_order_by_20_rows() {
    let tmp = TempDir::new().unwrap();
    // Insert rows in random-ish order
    let mut csv = String::from("id,amount\n");
    let order = [15, 3, 18, 7, 12, 1, 9, 20, 5, 14, 2, 16, 8, 11, 19, 4, 13, 6, 17, 10];
    for &v in &order {
        csv.push_str(&format!("{},{}\n", v, v * 100));
    }
    make_csv(tmp.path(), "sales.csv", &csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 20);
    let amounts: Vec<i64> = result.rows.iter().map(|r| r[1].parse::<i64>().unwrap()).collect();
    let expected: Vec<i64> = (1..=20).map(|v| v * 100).collect();
    assert_eq!(amounts, expected);
}

// ============================================================
// ORDER BY with all negative values
// ============================================================

#[test]
fn test_order_by_all_negative() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,-300\n\
               2,-100\n\
               3,-500\n\
               4,-200\n\
               5,-400\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 5);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["-500", "-400", "-300", "-200", "-100"]);
}

// ============================================================
// ORDER BY DESC with LIMIT 1 (finds max)
// ============================================================

#[test]
fn test_order_by_desc_limit_1() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,300\n\
               2,100\n\
               3,500\n\
               4,200\n\
               5,400\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount DESC LIMIT 1");
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][1], "500");
}

// ============================================================
// ORDER BY ASC with LIMIT 1 (finds min)
// ============================================================

#[test]
fn test_order_by_asc_limit_1() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,300\n\
               2,100\n\
               3,500\n\
               4,200\n\
               5,400\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC LIMIT 1");
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][1], "100");
}

// ============================================================
// ORDER BY alternating values
// ============================================================

#[test]
fn test_order_by_alternating_values() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,100\n\
               2,500\n\
               3,100\n\
               4,500\n\
               5,100\n\
               6,500\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY amount ASC");
    assert_eq!(result.row_count, 6);
    let amounts: Vec<&str> = result.rows.iter().map(|r| r[1].as_str()).collect();
    assert_eq!(amounts, vec!["100", "100", "100", "500", "500", "500"]);
}

// ============================================================
// ORDER BY with filter that eliminates all rows except one
// ============================================================

#[test]
fn test_order_by_filter_single_result() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,100\n\
               2,200\n\
               3,300\n\
               4,400\n\
               5,500\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(
        tmp.path(),
        "SELECT * FROM sales WHERE amount > 400 ORDER BY amount ASC",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][1], "500");
}

// ============================================================
// ORDER BY on third column
// ============================================================

#[test]
fn test_order_by_third_column() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,500,30\n\
               2,100,10\n\
               3,300,50\n\
               4,200,20\n\
               5,400,40\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(tmp.path(), "SELECT * FROM sales ORDER BY quantity ASC");
    assert_eq!(result.row_count, 5);
    let quantities: Vec<&str> = result.rows.iter().map(|r| r[2].as_str()).collect();
    assert_eq!(quantities, vec!["10", "20", "30", "40", "50"]);
}
