//! End-to-end CSV query tests with golden file oracle.
//!
//! Each test creates CSV data in a temp directory, executes a SQL query
//! via the full pipeline (parse -> optimize -> plan -> GPU execute),
//! and compares results against expected values.

use std::path::Path;
use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

// ============================================================
// Test helpers
// ============================================================

/// Write a CSV file to a directory.
fn write_csv(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    std::fs::write(&path, content).expect("write csv");
}

/// Run a SQL query through the full pipeline: parse -> optimize -> plan -> execute.
fn run_query(dir: &Path, sql: &str) -> gpu_query::gpu::executor::QueryResult {
    let catalog = catalog::scan_directory(dir).expect("scan directory");
    let logical = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let optimized = gpu_query::sql::optimizer::optimize(logical);
    let physical = gpu_query::sql::physical_plan::plan(&optimized).expect("plan SQL");
    let mut executor = QueryExecutor::new().expect("create executor");
    executor.execute(&physical, &catalog).expect("execute query")
}

/// Run a query and expect it to fail.
fn run_query_err(dir: &Path, sql: &str) -> String {
    let catalog = catalog::scan_directory(dir).expect("scan directory");
    let logical = gpu_query::sql::parser::parse_query(sql);
    if let Err(e) = logical {
        return e.to_string();
    }
    let logical = logical.unwrap();
    let optimized = gpu_query::sql::optimizer::optimize(logical);
    let physical = gpu_query::sql::physical_plan::plan(&optimized);
    if let Err(e) = physical {
        return e.to_string();
    }
    let physical = physical.unwrap();
    let mut executor = QueryExecutor::new().expect("create executor");
    match executor.execute(&physical, &catalog) {
        Ok(_) => "expected error but query succeeded".to_string(),
        Err(e) => e,
    }
}

/// Assert a single-row single-column result equals expected integer.
fn assert_int_result(result: &gpu_query::gpu::executor::QueryResult, expected: i64) {
    assert_eq!(result.row_count, 1, "expected 1 row, got {}", result.row_count);
    assert_eq!(
        result.rows[0][0],
        expected.to_string(),
        "expected {}, got {}",
        expected,
        result.rows[0][0]
    );
}

/// Assert a single-row result with multiple int columns.
fn assert_int_row(result: &gpu_query::gpu::executor::QueryResult, expected: &[i64]) {
    assert_eq!(result.row_count, 1, "expected 1 row, got {}", result.row_count);
    for (i, exp) in expected.iter().enumerate() {
        assert_eq!(
            result.rows[0][i],
            exp.to_string(),
            "column {} mismatch: expected {}, got {}",
            i,
            exp,
            result.rows[0][i]
        );
    }
}

/// Golden file comparison helper. Reads golden file from tests/golden/{name}.csv.
/// Compares result columns+rows against the golden file content.
fn assert_golden(result: &gpu_query::gpu::executor::QueryResult, golden_name: &str) {
    let golden_path = format!(
        "{}/tests/golden/{}.csv",
        env!("CARGO_MANIFEST_DIR"),
        golden_name
    );
    let golden_content = std::fs::read_to_string(&golden_path)
        .unwrap_or_else(|e| panic!("read golden file {}: {}", golden_path, e));
    let mut lines = golden_content.trim().lines();
    let header = lines.next().expect("golden file must have header");
    let expected_cols: Vec<&str> = header.split(',').collect();

    // Compare column names
    assert_eq!(
        result.columns.len(),
        expected_cols.len(),
        "column count mismatch: result={:?}, golden={:?}",
        result.columns,
        expected_cols
    );
    for (i, exp) in expected_cols.iter().enumerate() {
        assert_eq!(
            result.columns[i], *exp,
            "column name mismatch at {}: {} vs {}",
            i, result.columns[i], exp
        );
    }

    // Compare rows
    let golden_rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
    assert_eq!(
        result.row_count,
        golden_rows.len(),
        "row count mismatch: got {}, golden has {}",
        result.row_count,
        golden_rows.len()
    );
    for (r, golden_row) in golden_rows.iter().enumerate() {
        for (c, golden_val) in golden_row.iter().enumerate() {
            assert_eq!(
                result.rows[r][c], *golden_val,
                "mismatch at row {} col {}: got '{}', expected '{}'",
                r, c, result.rows[r][c], golden_val
            );
        }
    }
}

/// Standard 10-row sales test data (id, amount, quantity).
/// amounts: 50,150,200,75,300,25,500,100,450,80
/// quantities: 2,3,1,4,2,1,5,3,2,1
const SALES_CSV: &str = "id,amount,quantity\n\
1,50,2\n\
2,150,3\n\
3,200,1\n\
4,75,4\n\
5,300,2\n\
6,25,1\n\
7,500,5\n\
8,100,3\n\
9,450,2\n\
10,80,1\n";

fn make_sales_dir() -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().unwrap();
    write_csv(dir.path(), "sales.csv", SALES_CSV);
    let p = dir.path().to_path_buf();
    (dir, p)
}

// ============================================================
// 1. Basic SELECT with aggregates (no WHERE)
// ============================================================

#[test]
fn csv_count_star() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales");
    assert_int_result(&r, 10);
    assert_golden(&r, "csv_count_star");
}

#[test]
fn csv_sum_amount() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT sum(amount) FROM sales");
    // 50+150+200+75+300+25+500+100+450+80 = 1930
    assert_int_result(&r, 1930);
    assert_golden(&r, "csv_sum_amount");
}

#[test]
fn csv_sum_quantity() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT sum(quantity) FROM sales");
    // 2+3+1+4+2+1+5+3+2+1 = 24
    assert_int_result(&r, 24);
}

#[test]
fn csv_count_and_sum() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*), sum(amount) FROM sales");
    assert_int_row(&r, &[10, 1930]);
}

#[test]
fn csv_min_amount() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT min(amount) FROM sales");
    assert_int_result(&r, 25);
}

#[test]
fn csv_max_amount() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT max(amount) FROM sales");
    assert_int_result(&r, 500);
}

#[test]
fn csv_all_aggregates() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*), sum(amount), min(amount), max(amount) FROM sales");
    assert_eq!(r.row_count, 1);
    assert_eq!(r.rows[0][0], "10");   // count
    assert_eq!(r.rows[0][1], "1930"); // sum
    assert_eq!(r.rows[0][2], "25");   // min
    assert_eq!(r.rows[0][3], "500");  // max
}

// ============================================================
// 2. WHERE clause with all comparison operators
// ============================================================

#[test]
fn csv_where_gt() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales WHERE amount > 100");
    // >100: 150,200,300,500,450 = 5
    assert_int_result(&r, 5);
    assert_golden(&r, "csv_count_filtered_gt");
}

#[test]
fn csv_where_ge() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales WHERE amount >= 100");
    // >=100: 150,200,300,500,100,450 = 6
    assert_int_result(&r, 6);
}

#[test]
fn csv_where_lt() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales WHERE amount < 100");
    // <100: 50,75,25,80 = 4
    assert_int_result(&r, 4);
}

#[test]
fn csv_where_le() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales WHERE amount <= 100");
    // <=100: 50,75,25,100,80 = 5
    assert_int_result(&r, 5);
}

#[test]
fn csv_where_eq() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales WHERE amount = 100");
    assert_int_result(&r, 1);
}

#[test]
fn csv_where_ne() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales WHERE amount != 100");
    assert_int_result(&r, 9);
}

#[test]
fn csv_where_sum_filtered() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT sum(amount) FROM sales WHERE amount > 100");
    // >100: 150+200+300+500+450 = 1600
    assert_int_result(&r, 1600);
    assert_golden(&r, "csv_sum_filtered");
}

#[test]
fn csv_where_filter_on_different_column() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT sum(amount) FROM sales WHERE quantity > 3");
    // qty>3: row4(75,qty4), row7(500,qty5) = 575
    assert_int_result(&r, 575);
}

#[test]
fn csv_where_min_filtered() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT min(amount) FROM sales WHERE amount > 100");
    // >100: 150,200,300,500,450 => min=150
    assert_int_result(&r, 150);
}

#[test]
fn csv_where_max_filtered() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT max(amount) FROM sales WHERE amount < 200");
    // <200: 50,150,75,25,100,80 => max=150
    assert_int_result(&r, 150);
}

// ============================================================
// 3. Compound predicates (AND/OR)
// ============================================================

#[test]
fn csv_where_and() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales WHERE amount > 100 AND amount < 400");
    // 150,200,300 = 3
    assert_int_result(&r, 3);
}

#[test]
fn csv_where_or() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales WHERE amount < 50 OR amount > 400");
    // <50: 25. >400: 500,450 => total 3
    assert_int_result(&r, 3);
}

#[test]
fn csv_where_and_sum() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT sum(amount) FROM sales WHERE amount >= 100 AND quantity >= 3");
    // amount>=100: 150,200,300,500,100,450
    // AND qty>=3: row2(150,3), row7(500,5), row8(100,3)
    // sum = 150+500+100 = 750
    assert_int_result(&r, 750);
}

// ============================================================
// 4. GROUP BY
// ============================================================

#[test]
fn csv_group_by_count() {
    let dir = TempDir::new().unwrap();
    write_csv(
        dir.path(),
        "orders.csv",
        "id,region,amount\n\
         1,East,100\n\
         2,West,200\n\
         3,East,300\n\
         4,West,400\n\
         5,East,500\n",
    );
    let r = run_query(dir.path(), "SELECT region, count(*) FROM orders GROUP BY region");
    assert_eq!(r.row_count, 2);
    // Groups sorted alphabetically: East, West
    assert_eq!(r.rows[0][0], "East");
    assert_eq!(r.rows[0][1], "3");
    assert_eq!(r.rows[1][0], "West");
    assert_eq!(r.rows[1][1], "2");
}

#[test]
fn csv_group_by_sum() {
    let dir = TempDir::new().unwrap();
    write_csv(
        dir.path(),
        "orders.csv",
        "id,region,amount\n\
         1,East,100\n\
         2,West,200\n\
         3,East,300\n\
         4,West,400\n\
         5,East,500\n",
    );
    let r = run_query(dir.path(), "SELECT region, sum(amount) FROM orders GROUP BY region");
    assert_eq!(r.row_count, 2);
    assert_eq!(r.rows[0][0], "East");
    assert_eq!(r.rows[0][1], "900");  // 100+300+500
    assert_eq!(r.rows[1][0], "West");
    assert_eq!(r.rows[1][1], "600");  // 200+400
}

// ============================================================
// 5. ORDER BY
// ============================================================

#[test]
fn csv_order_by_asc() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT id, amount FROM sales ORDER BY amount");
    // Ascending by amount: 25,50,75,80,100,150,200,300,450,500
    assert_eq!(r.row_count, 10);
    assert_eq!(r.rows[0][1], "25");
    assert_eq!(r.rows[9][1], "500");
}

#[test]
fn csv_order_by_desc() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT id, amount FROM sales ORDER BY amount DESC");
    assert_eq!(r.row_count, 10);
    assert_eq!(r.rows[0][1], "500");
    assert_eq!(r.rows[9][1], "25");
}

// ============================================================
// 6. LIMIT
// ============================================================

#[test]
fn csv_limit() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT id, amount FROM sales ORDER BY amount LIMIT 3");
    assert_eq!(r.row_count, 3);
    assert_eq!(r.rows[0][1], "25");
    assert_eq!(r.rows[1][1], "50");
    assert_eq!(r.rows[2][1], "75");
}

#[test]
fn csv_limit_larger_than_data() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT id, amount FROM sales ORDER BY amount LIMIT 100");
    assert_eq!(r.row_count, 10);
}

// ============================================================
// 7. Edge cases
// ============================================================

#[test]
fn csv_single_row() {
    let dir = TempDir::new().unwrap();
    write_csv(dir.path(), "tiny.csv", "val\n42\n");
    let r = run_query(dir.path(), "SELECT count(*), sum(val) FROM tiny");
    assert_eq!(r.rows[0][0], "1");
    assert_eq!(r.rows[0][1], "42");
}

#[test]
fn csv_large_values() {
    let dir = TempDir::new().unwrap();
    write_csv(
        dir.path(),
        "big.csv",
        "id,value\n1,1000000\n2,2000000\n3,3000000\n",
    );
    let r = run_query(dir.path(), "SELECT sum(value) FROM big");
    assert_int_result(&r, 6000000);
}

#[test]
fn csv_negative_values() {
    let dir = TempDir::new().unwrap();
    write_csv(
        dir.path(),
        "neg.csv",
        "id,amount\n1,-100\n2,-50\n3,0\n4,50\n5,100\n",
    );
    let r = run_query(dir.path(), "SELECT sum(amount) FROM neg");
    assert_int_result(&r, 0); // -100-50+0+50+100 = 0
}

#[test]
fn csv_all_zeros() {
    let dir = TempDir::new().unwrap();
    write_csv(dir.path(), "zeros.csv", "id,val\n1,0\n2,0\n3,0\n");
    let r = run_query(dir.path(), "SELECT sum(val), min(val), max(val) FROM zeros");
    assert_eq!(r.rows[0][0], "0"); // sum
    assert_eq!(r.rows[0][1], "0"); // min
    assert_eq!(r.rows[0][2], "0"); // max
}

#[test]
fn csv_filter_no_matches() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales WHERE amount > 9999");
    assert_int_result(&r, 0);
}

#[test]
fn csv_filter_all_match() {
    let (_d, p) = make_sales_dir();
    let r = run_query(&p, "SELECT count(*) FROM sales WHERE amount > 0");
    assert_int_result(&r, 10);
}

// ============================================================
// 8. DESCRIBE test (kept from original)
// ============================================================

#[test]
fn csv_describe() {
    let dir = TempDir::new().unwrap();
    write_csv(
        dir.path(),
        "sales.csv",
        "id,name,amount,region\n\
         1,Alice,500,East\n\
         2,Bob,200,West\n\
         3,Charlie,800,East\n\
         4,Diana,150,West\n\
         5,Eve,1000,East\n",
    );

    let catalog = catalog::scan_directory(dir.path()).expect("scan directory");
    assert!(!catalog.is_empty());

    let mut executor = QueryExecutor::new().expect("GPU executor should init");
    let describe_result = executor
        .execute_describe("sales", &catalog)
        .expect("execute_describe should succeed");

    assert_eq!(describe_result.table_name, "sales");
    assert_eq!(describe_result.columns.len(), 4);

    let col_names: Vec<&str> = describe_result.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(col_names, &["id", "name", "amount", "region"]);

    // Verify row counts
    for col in &describe_result.columns {
        assert_eq!(col.count, 5);
    }

    // Check id column
    let id_col = &describe_result.columns[0];
    assert_eq!(id_col.distinct_count, 5);
    assert_eq!(id_col.min_value, "1");
    assert_eq!(id_col.max_value, "5");

    // Check amount column
    let amount_col = &describe_result.columns[2];
    assert_eq!(amount_col.distinct_count, 5);
    assert_eq!(amount_col.min_value, "150");
    assert_eq!(amount_col.max_value, "1000");
}
