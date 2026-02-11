//! Integration tests for GPU schema inference with type voting.
//!
//! Tests the enhanced multi-row schema inference that samples up to 100 rows
//! to determine column types (INT64, FLOAT64, VARCHAR) and nullable flags.
//! Also tests the NullBitmap data structure.

use std::io::Write;
use tempfile::NamedTempFile;

use gpu_query::gpu::executor::infer_schema_from_csv;
use gpu_query::io::csv::CsvMetadata;
use gpu_query::storage::null_bitmap::NullBitmap;
use gpu_query::storage::schema::DataType;

// ---- Schema inference tests ----

/// Helper: write CSV content to a temp file and return the path + metadata.
fn make_csv(content: &str) -> (NamedTempFile, CsvMetadata) {
    let mut f = NamedTempFile::new().expect("create temp file");
    f.write_all(content.as_bytes()).expect("write CSV");
    f.flush().expect("flush");

    // Parse header to build CsvMetadata
    let first_line = content.lines().next().unwrap_or("");
    let delimiter = b',';
    let column_names: Vec<String> = first_line
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let column_count = column_names.len();
    let meta = CsvMetadata {
        column_names,
        delimiter,
        column_count,
        file_path: f.path().to_path_buf(),
    };
    (f, meta)
}

#[test]
fn test_infer_all_int_columns() {
    let csv = "id,count,value\n1,10,100\n2,20,200\n3,30,300\n";
    let (f, meta) = make_csv(csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    assert_eq!(schema.num_columns(), 3);
    assert_eq!(schema.columns[0].data_type, DataType::Int64);
    assert_eq!(schema.columns[1].data_type, DataType::Int64);
    assert_eq!(schema.columns[2].data_type, DataType::Int64);
    assert!(!schema.columns[0].nullable);
}

#[test]
fn test_infer_float_column() {
    let csv = "id,price\n1,10.5\n2,20.3\n3,30.7\n";
    let (f, meta) = make_csv(csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    assert_eq!(schema.columns[0].data_type, DataType::Int64);
    assert_eq!(schema.columns[1].data_type, DataType::Float64);
}

#[test]
fn test_infer_varchar_column() {
    let csv = "id,name\n1,Alice\n2,Bob\n3,Charlie\n";
    let (f, meta) = make_csv(csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    assert_eq!(schema.columns[0].data_type, DataType::Int64);
    assert_eq!(schema.columns[1].data_type, DataType::Varchar);
}

#[test]
fn test_infer_mixed_types_across_rows() {
    // First row has "1" in price (looks like int), but later rows have floats
    let csv = "id,price\n1,100\n2,200.5\n3,300\n";
    let (f, meta) = make_csv(csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    // price should be Float64 because row 2 has a float value
    assert_eq!(schema.columns[1].data_type, DataType::Float64);
}

#[test]
fn test_infer_nullable_from_empty_field() {
    let csv = "id,value\n1,100\n2,\n3,300\n";
    let (f, meta) = make_csv(csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    // value column should be nullable (row 2 is empty)
    assert!(schema.columns[1].nullable);
    // Non-empty values are all int64
    assert_eq!(schema.columns[1].data_type, DataType::Int64);
}

#[test]
fn test_infer_varchar_upgrade_from_mixed() {
    // Column starts as int but later row has a string
    let csv = "id,code\n1,100\n2,ABC\n3,300\n";
    let (f, meta) = make_csv(csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    // code should be Varchar because "ABC" forces upgrade
    assert_eq!(schema.columns[1].data_type, DataType::Varchar);
}

#[test]
fn test_infer_many_rows_consistency() {
    // Generate 50 rows of consistent int data
    let mut csv = String::from("id,amount\n");
    for i in 1..=50 {
        csv.push_str(&format!("{},{}\n", i, i * 100));
    }
    let (f, meta) = make_csv(&csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    assert_eq!(schema.columns[0].data_type, DataType::Int64);
    assert_eq!(schema.columns[1].data_type, DataType::Int64);
    assert!(!schema.columns[0].nullable);
    assert!(!schema.columns[1].nullable);
}

#[test]
fn test_infer_float_upgrade_at_row_50() {
    // 49 rows of int, then a float on row 50
    let mut csv = String::from("id,value\n");
    for i in 1..=49 {
        csv.push_str(&format!("{},{}\n", i, i * 10));
    }
    csv.push_str("50,500.5\n");
    let (f, meta) = make_csv(&csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    assert_eq!(schema.columns[1].data_type, DataType::Float64);
}

#[test]
fn test_infer_all_empty_defaults_to_varchar() {
    let csv = "id,notes\n1,\n2,\n3,\n";
    let (f, meta) = make_csv(csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    // All empty -> Varchar default, nullable
    assert_eq!(schema.columns[1].data_type, DataType::Varchar);
    assert!(schema.columns[1].nullable);
}

#[test]
fn test_infer_negative_numbers() {
    let csv = "id,value\n1,-100\n2,-200\n3,300\n";
    let (f, meta) = make_csv(csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    assert_eq!(schema.columns[1].data_type, DataType::Int64);
}

#[test]
fn test_infer_negative_floats() {
    let csv = "id,value\n1,-10.5\n2,20.3\n3,-30.7\n";
    let (f, meta) = make_csv(csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    assert_eq!(schema.columns[1].data_type, DataType::Float64);
}

#[test]
fn test_infer_single_row() {
    let csv = "a,b,c\n42,3.14,hello\n";
    let (f, meta) = make_csv(csv);
    let schema = infer_schema_from_csv(f.path(), &meta).unwrap();

    assert_eq!(schema.columns[0].data_type, DataType::Int64);
    assert_eq!(schema.columns[1].data_type, DataType::Float64);
    assert_eq!(schema.columns[2].data_type, DataType::Varchar);
}

#[test]
fn test_infer_no_data_rows_error() {
    let csv = "a,b,c\n";
    let (f, meta) = make_csv(csv);
    let result = infer_schema_from_csv(f.path(), &meta);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("no data rows"));
}

// ---- NullBitmap tests ----

#[test]
fn test_null_bitmap_basic_ops() {
    let mut bm = NullBitmap::new(100);
    assert_eq!(bm.null_count(), 0);
    assert_eq!(bm.row_count(), 100);

    bm.set_null(0);
    bm.set_null(50);
    bm.set_null(99);
    assert!(bm.is_null(0));
    assert!(bm.is_null(50));
    assert!(bm.is_null(99));
    assert!(!bm.is_null(1));
    assert!(!bm.is_null(49));
    assert_eq!(bm.null_count(), 3);
}

#[test]
fn test_null_bitmap_word_boundaries() {
    let mut bm = NullBitmap::new(64);
    // Test at word boundaries: 31 (end of word 0), 32 (start of word 1)
    bm.set_null(31);
    bm.set_null(32);
    assert!(bm.is_null(31));
    assert!(bm.is_null(32));
    assert!(!bm.is_null(30));
    assert!(!bm.is_null(33));
    assert_eq!(bm.null_count(), 2);
}

#[test]
fn test_null_bitmap_clear_and_recount() {
    let mut bm = NullBitmap::new(32);
    for i in 0..32 {
        bm.set_null(i);
    }
    assert_eq!(bm.null_count(), 32);

    bm.clear_null(0);
    bm.clear_null(15);
    bm.clear_null(31);
    assert_eq!(bm.null_count(), 29);
    assert!(!bm.is_null(0));
    assert!(!bm.is_null(15));
    assert!(!bm.is_null(31));
}

#[test]
fn test_null_bitmap_gpu_buffer_format() {
    // Verify the word layout matches what GPU expects
    let mut bm = NullBitmap::new(64);
    bm.set_null(0); // bit 0 of word 0
    bm.set_null(1); // bit 1 of word 0
    bm.set_null(32); // bit 0 of word 1
    bm.set_null(63); // bit 31 of word 1

    let words = bm.as_words();
    assert_eq!(words.len(), 2);
    assert_eq!(words[0], 0b11); // bits 0 and 1
    assert_eq!(words[1], (1 << 0) | (1 << 31)); // bits 0 and 31
}
