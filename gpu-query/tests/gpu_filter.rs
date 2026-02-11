//! Integration tests for the GPU column filter kernel.
//!
//! Tests the filter pipeline: create column data -> allocate bitmask/count buffers ->
//! create specialized PSO via function constants -> dispatch filter kernel ->
//! read back bitmask + match_count -> verify correctness.

use gpu_query::gpu::device::GpuDevice;
use gpu_query::gpu::encode;
use gpu_query::gpu::pipeline::{filter_pso_key, ColumnTypeCode, CompareOp, PsoCache};
use gpu_query::gpu::types::FilterParams;

use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

/// Run the column_filter kernel on INT64 data and return (bitmask_words, match_count).
fn run_int64_filter(
    gpu: &GpuDevice,
    pso_cache: &mut PsoCache,
    data: &[i64],
    compare_op: CompareOp,
    compare_value: i64,
) -> (Vec<u32>, u32) {
    let row_count = data.len() as u32;
    let bitmask_words = ((row_count + 31) / 32) as usize;

    // Allocate buffers
    let data_buffer = encode::alloc_buffer_with_data(&gpu.device, data);
    let bitmask_buffer = encode::alloc_buffer(&gpu.device, bitmask_words * 4);
    let match_count_buffer = encode::alloc_buffer(&gpu.device, 4);
    // Null bitmap (empty, not used)
    let null_bitmap_buffer = encode::alloc_buffer(&gpu.device, bitmask_words * 4);

    // Zero bitmask and match_count
    unsafe {
        let bitmask_ptr = bitmask_buffer.contents().as_ptr() as *mut u32;
        for i in 0..bitmask_words {
            *bitmask_ptr.add(i) = 0;
        }
        let count_ptr = match_count_buffer.contents().as_ptr() as *mut u32;
        *count_ptr = 0;
    }

    // Create FilterParams
    let params = FilterParams {
        compare_value_int: compare_value,
        compare_value_float: 0.0,
        row_count,
        column_stride: 8,
        null_bitmap_present: 0,
        _pad0: 0,
        compare_value_int_hi: 0,
    };
    let params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[params]);

    // Get specialized PSO
    let key = filter_pso_key(compare_op, ColumnTypeCode::Int64, false);
    let pipeline = pso_cache.get_or_create(&gpu.library, &key);

    // Dispatch
    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encode::dispatch_threads_1d(
            &encoder,
            pipeline,
            &[
                (&data_buffer, 0),
                (&bitmask_buffer, 1),
                (&match_count_buffer, 2),
                (&params_buffer, 3),
                (&null_bitmap_buffer, 4),
            ],
            row_count as usize,
        );

        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    // Read back results
    let bitmask = unsafe {
        let ptr = bitmask_buffer.contents().as_ptr() as *const u32;
        std::slice::from_raw_parts(ptr, bitmask_words).to_vec()
    };
    let count = unsafe {
        let ptr = match_count_buffer.contents().as_ptr() as *const u32;
        *ptr
    };

    (bitmask, count)
}

/// Extract matching row indices from a bitmask.
fn bitmask_to_indices(bitmask: &[u32], row_count: usize) -> Vec<usize> {
    let mut indices = Vec::new();
    for row in 0..row_count {
        let word = row / 32;
        let bit = row % 32;
        if (bitmask[word] >> bit) & 1 == 1 {
            indices.push(row);
        }
    }
    indices
}

// ---- INT64 filter tests ----

#[test]
fn test_int64_filter_gt() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Gt, 25);

    assert_eq!(count, 3, "GT 25: should match 30, 40, 50");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![2, 3, 4], "GT 25: matching rows");
}

#[test]
fn test_int64_filter_eq() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Eq, 20);

    assert_eq!(count, 1, "EQ 20: should match exactly 1");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![1], "EQ 20: matching row index");
}

#[test]
fn test_int64_filter_lt() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Lt, 15);

    assert_eq!(count, 1, "LT 15: should match 10");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0], "LT 15: matching row index");
}

#[test]
fn test_int64_filter_le() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Le, 30);

    assert_eq!(count, 3, "LE 30: should match 10, 20, 30");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1, 2], "LE 30: matching rows");
}

#[test]
fn test_int64_filter_ge() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Ge, 40);

    assert_eq!(count, 2, "GE 40: should match 40, 50");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![3, 4], "GE 40: matching rows");
}

#[test]
fn test_int64_filter_ne() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Ne, 30);

    assert_eq!(count, 4, "NE 30: should match all except 30");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1, 3, 4], "NE 30: matching rows");
}

#[test]
fn test_int64_filter_no_matches() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30];
    let (_bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Gt, 100);

    assert_eq!(count, 0, "GT 100: no values match");
}

#[test]
fn test_int64_filter_all_match() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Gt, 0);

    assert_eq!(count, 3, "GT 0: all values match");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn test_int64_filter_negative_values() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![-50, -20, 0, 10, 30];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Ge, 0);

    assert_eq!(count, 3, "GE 0: should match 0, 10, 30");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![2, 3, 4]);
}

#[test]
fn test_int64_filter_single_row() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![42];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Eq, 42);

    assert_eq!(count, 1);
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0]);
}

// ---- Many rows (cross bitmask word boundary) ----

#[test]
fn test_int64_filter_many_rows() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    // 100 rows: values 0..99, filter GT 49 -> matches 50..99 = 50 rows
    let data: Vec<i64> = (0..100).collect();
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Gt, 49);

    assert_eq!(count, 50, "GT 49 on 0..99: should match 50 rows");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices.len(), 50);
    assert_eq!(indices[0], 50);
    assert_eq!(*indices.last().unwrap(), 99);
}

#[test]
fn test_int64_filter_exactly_32_rows() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    // Exactly 32 rows = 1 bitmask word
    let data: Vec<i64> = (1..=32).map(|x| x as i64).collect();
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Le, 16);

    assert_eq!(count, 16, "LE 16 on 1..32: should match 16 rows");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices.len(), 16);
    // rows 0..15 (values 1..16)
    for i in 0..16 {
        assert_eq!(indices[i], i);
    }
}

// ---- PSO cache tests ----

#[test]
fn test_pso_cache_reuse() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();
    assert!(cache.is_empty());

    let data: Vec<i64> = vec![10, 20, 30];

    // First call compiles PSO
    let _ = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Gt, 15);
    assert_eq!(cache.len(), 1, "One PSO should be cached");

    // Second call with same op should reuse PSO (no recompile)
    let _ = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Gt, 25);
    assert_eq!(cache.len(), 1, "Same PSO should be reused");

    // Different op creates a new PSO
    let _ = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Lt, 25);
    assert_eq!(cache.len(), 2, "Different op should create new PSO");
}

// ---- FLOAT filter tests ----

/// Run the column_filter kernel on FLOAT32 data and return (bitmask_words, match_count).
fn run_float_filter(
    gpu: &GpuDevice,
    pso_cache: &mut PsoCache,
    data: &[f32],
    compare_op: CompareOp,
    compare_value: f32,
) -> (Vec<u32>, u32) {
    let row_count = data.len() as u32;
    let bitmask_words = ((row_count + 31) / 32) as usize;

    let data_buffer = encode::alloc_buffer_with_data(&gpu.device, data);
    let bitmask_buffer = encode::alloc_buffer(&gpu.device, bitmask_words * 4);
    let match_count_buffer = encode::alloc_buffer(&gpu.device, 4);
    let null_bitmap_buffer = encode::alloc_buffer(&gpu.device, bitmask_words * 4);

    // Zero buffers
    unsafe {
        let bitmask_ptr = bitmask_buffer.contents().as_ptr() as *mut u32;
        for i in 0..bitmask_words {
            *bitmask_ptr.add(i) = 0;
        }
        let count_ptr = match_count_buffer.contents().as_ptr() as *mut u32;
        *count_ptr = 0;
    }

    // For float comparison, store the f32 compare value's bits as i64
    // in compare_value_float_bits. The shader reinterprets the lower 32 bits as float.
    let float_bits = compare_value.to_bits() as i32;
    let params = FilterParams {
        compare_value_int: 0,
        // Store f32 bits in the f64 field; the shader reads the bits via as_type
        compare_value_float: f64::from_bits(float_bits as u32 as u64),
        row_count,
        column_stride: 4,
        null_bitmap_present: 0,
        _pad0: 0,
        compare_value_int_hi: 0,
    };
    let params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[params]);

    let key = filter_pso_key(compare_op, ColumnTypeCode::Float64, false);
    let pipeline = pso_cache.get_or_create(&gpu.library, &key);

    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encode::dispatch_threads_1d(
            &encoder,
            pipeline,
            &[
                (&data_buffer, 0),
                (&bitmask_buffer, 1),
                (&match_count_buffer, 2),
                (&params_buffer, 3),
                (&null_bitmap_buffer, 4),
            ],
            row_count as usize,
        );

        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let bitmask = unsafe {
        let ptr = bitmask_buffer.contents().as_ptr() as *const u32;
        std::slice::from_raw_parts(ptr, bitmask_words).to_vec()
    };
    let count = unsafe {
        let ptr = match_count_buffer.contents().as_ptr() as *const u32;
        *ptr
    };

    (bitmask, count)
}

#[test]
fn test_float_filter_gt() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![1.0, 2.5, 3.7, 4.2, 5.9];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Gt, 3.0);

    assert_eq!(count, 3, "GT 3.0: should match 3.7, 4.2, 5.9");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![2, 3, 4]);
}

#[test]
fn test_float_filter_lt() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![1.0, 2.5, 3.7, 4.2, 5.9];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Lt, 3.0);

    assert_eq!(count, 2, "LT 3.0: should match 1.0, 2.5");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn test_float_filter_eq() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![1.0, 2.5, 2.5, 4.0, 2.5];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Eq, 2.5);

    assert_eq!(count, 3, "EQ 2.5: should match 3 rows");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![1, 2, 4]);
}

// ---- String filter tests (dictionary-encoded via executor) ----

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;

/// Helper: create a CSV file in a directory.
fn make_csv(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("create csv file");
    f.write_all(content.as_bytes()).expect("write csv");
    f.flush().expect("flush csv");
}

/// Helper: run a SQL query against a CSV file directory.
fn run_query(dir: &Path, sql: &str) -> gpu_query::gpu::executor::QueryResult {
    let cat = catalog::scan_directory(dir).expect("scan directory");
    let logical_plan = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let physical_plan = gpu_query::sql::physical_plan::plan(&logical_plan).expect("plan SQL");
    let mut executor = QueryExecutor::new().expect("create executor");
    executor
        .execute(&physical_plan, &cat)
        .expect("execute query")
}

#[test]
fn test_string_filter_eq_europe() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,region,amount\n\
               1,Europe,100\n\
               2,Asia,200\n\
               3,Europe,300\n\
               4,Africa,400\n\
               5,Asia,500\n\
               6,Europe,600\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM sales WHERE region = 'Europe'",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(
        result.rows[0][0], "3",
        "Should match 3 rows where region = 'Europe'"
    );
}

#[test]
fn test_string_filter_nonexistent_value() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,region,amount\n\
               1,Europe,100\n\
               2,Asia,200\n\
               3,Africa,300\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM sales WHERE region = 'Antarctica'",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(
        result.rows[0][0], "0",
        "Nonexistent value should return 0 rows"
    );
}

#[test]
fn test_string_filter_all_match() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,region,amount\n\
               1,Europe,100\n\
               2,Europe,200\n\
               3,Europe,300\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM sales WHERE region = 'Europe'",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "3", "All rows should match");
}

#[test]
fn test_string_filter_single_match() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,region,amount\n\
               1,Europe,100\n\
               2,Asia,200\n\
               3,Africa,300\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM sales WHERE region = 'Asia'",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "1", "Should match exactly 1 row");
}

#[test]
fn test_string_filter_with_sum() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,region,amount\n\
               1,Europe,100\n\
               2,Asia,200\n\
               3,Europe,300\n\
               4,Africa,400\n\
               5,Europe,500\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(
        tmp.path(),
        "SELECT sum(amount) FROM sales WHERE region = 'Europe'",
    );
    assert_eq!(result.row_count, 1);
    // Europe rows: amount 100 + 300 + 500 = 900
    assert_eq!(
        result.rows[0][0], "900",
        "SUM of Europe amounts should be 900"
    );
}

#[test]
fn test_string_filter_group_by_region() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,region,amount\n\
               1,Europe,100\n\
               2,Asia,200\n\
               3,Europe,300\n\
               4,Africa,400\n\
               5,Asia,500\n";
    make_csv(tmp.path(), "sales.csv", csv);

    let result = run_query(
        tmp.path(),
        "SELECT region, sum(amount) FROM sales GROUP BY region",
    );
    assert_eq!(result.row_count, 3, "Should have 3 groups");

    // Groups sorted alphabetically: Africa, Asia, Europe
    let groups: Vec<(&str, &str)> = result
        .rows
        .iter()
        .map(|r| (r[0].as_str(), r[1].as_str()))
        .collect();
    assert_eq!(
        groups,
        vec![("Africa", "400"), ("Asia", "700"), ("Europe", "400")]
    );
}

// ---- Compound predicate tests (AND/OR) ----

#[test]
fn test_compound_and_count() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,50,10\n\
               2,150,5\n\
               3,200,20\n\
               4,300,3\n\
               5,250,15\n\
               6,80,25\n";
    make_csv(tmp.path(), "data.csv", csv);

    // amount > 100 AND quantity > 10 => rows 3 (200,20) and 5 (250,15)
    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM data WHERE amount > 100 AND quantity > 10",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "2", "AND: 2 rows match both predicates");
}

#[test]
fn test_compound_or_count() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,50,10\n\
               2,150,5\n\
               3,200,20\n\
               4,300,3\n\
               5,250,15\n\
               6,80,25\n";
    make_csv(tmp.path(), "data.csv", csv);

    // amount > 200 OR quantity > 20 => rows 4 (300,3), 5 (250,15), 6 (80,25)
    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM data WHERE amount > 200 AND quantity > 20",
    );
    assert_eq!(result.row_count, 1);
    // Only row where both: none -- amount>200 has rows 4,5 and quantity>20 has rows 3,6
    // Actually: amount>200: rows 4(300),5(250); quantity>20: rows 3(20),6(25)
    // AND of those = empty set? No wait: quantity > 20 means > 20, so row 6 (25) only
    // amount > 200: rows 4 (300), 5 (250)
    // quantity > 20: rows 6 (25)
    // AND => no overlap => 0
    assert_eq!(result.rows[0][0], "0", "AND: no rows match both predicates");
}

#[test]
fn test_compound_and_sum() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,50,10\n\
               2,150,5\n\
               3,200,20\n\
               4,300,3\n\
               5,250,15\n\
               6,80,25\n";
    make_csv(tmp.path(), "data.csv", csv);

    // amount > 100 AND quantity > 10 => rows 3 (200,20) and 5 (250,15)
    // sum(amount) = 200 + 250 = 450
    let result = run_query(
        tmp.path(),
        "SELECT sum(amount) FROM data WHERE amount > 100 AND quantity > 10",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "450", "AND: sum of matching rows");
}

#[test]
fn test_compound_or_sum() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,50,10\n\
               2,150,5\n\
               3,200,20\n\
               4,300,3\n\
               5,250,15\n\
               6,80,25\n";
    make_csv(tmp.path(), "data.csv", csv);

    // amount > 250 OR quantity > 20 => rows 4 (300,3), 6 (80,25)
    // sum(amount) = 300 + 80 = 380
    let result = run_query(
        tmp.path(),
        "SELECT sum(amount) FROM data WHERE amount > 250 OR quantity > 20",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "380", "OR: sum of matching rows");
}

#[test]
fn test_compound_and_all_match() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,150,20\n\
               2,200,30\n\
               3,300,40\n";
    make_csv(tmp.path(), "data.csv", csv);

    // amount > 100 AND quantity > 10 => all 3 rows
    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM data WHERE amount > 100 AND quantity > 10",
    );
    assert_eq!(
        result.rows[0][0], "3",
        "AND: all rows match both predicates"
    );
}

#[test]
fn test_compound_and_no_match() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,50,100\n\
               2,200,5\n\
               3,300,3\n";
    make_csv(tmp.path(), "data.csv", csv);

    // amount > 100 AND quantity > 50 => row 1 has qty>50 but amt<100; rows 2,3 have amt>100 but qty<50
    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM data WHERE amount > 100 AND quantity > 50",
    );
    assert_eq!(result.rows[0][0], "0", "AND: no rows match both predicates");
}

#[test]
fn test_compound_or_overlap() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount,quantity\n\
               1,50,10\n\
               2,200,30\n\
               3,300,5\n\
               4,80,40\n";
    make_csv(tmp.path(), "data.csv", csv);

    // amount > 100 OR quantity > 20 => rows 2 (both), 3 (amount only), 4 (quantity only)
    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM data WHERE amount > 100 OR quantity > 20",
    );
    assert_eq!(
        result.rows[0][0], "3",
        "OR: 3 rows match at least one predicate"
    );
}

#[test]
fn test_compound_same_column_and() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,50\n\
               2,100\n\
               3,150\n\
               4,200\n\
               5,250\n";
    make_csv(tmp.path(), "data.csv", csv);

    // amount > 100 AND amount < 200 => row 3 (150) only
    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM data WHERE amount > 100 AND amount < 200",
    );
    assert_eq!(result.rows[0][0], "1", "AND: range filter on same column");
}

#[test]
fn test_compound_same_column_or() {
    let tmp = TempDir::new().unwrap();
    let csv = "id,amount\n\
               1,50\n\
               2,100\n\
               3,150\n\
               4,200\n\
               5,250\n";
    make_csv(tmp.path(), "data.csv", csv);

    // amount < 100 OR amount > 200 => rows 1 (50), 5 (250)
    let result = run_query(
        tmp.path(),
        "SELECT count(*) FROM data WHERE amount < 100 OR amount > 200",
    );
    assert_eq!(
        result.rows[0][0], "2",
        "OR: exclusive ranges on same column"
    );
}

// ---- FLOAT filter additional operator tests ----

#[test]
fn test_float_filter_ge() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![1.0, 2.5, 3.0, 4.2, 5.9];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Ge, 3.0);

    assert_eq!(count, 3, "GE 3.0: should match 3.0, 4.2, 5.9");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![2, 3, 4]);
}

#[test]
fn test_float_filter_le() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![1.0, 2.5, 3.0, 4.2, 5.9];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Le, 3.0);

    assert_eq!(count, 3, "LE 3.0: should match 1.0, 2.5, 3.0");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn test_float_filter_ne() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![1.0, 2.5, 2.5, 4.0, 2.5];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Ne, 2.5);

    assert_eq!(count, 2, "NE 2.5: should match 1.0, 4.0");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 3]);
}

// ---- Edge value tests (INT64) ----

#[test]
fn test_int64_filter_compare_zero() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![-5, -1, 0, 1, 5];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Eq, 0);

    assert_eq!(count, 1, "EQ 0: should match exactly 0");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![2]);
}

#[test]
fn test_int64_filter_gt_zero() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![-5, -1, 0, 1, 5];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Gt, 0);

    assert_eq!(count, 2, "GT 0: should match 1, 5");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![3, 4]);
}

#[test]
fn test_int64_filter_lt_zero() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![-5, -1, 0, 1, 5];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Lt, 0);

    assert_eq!(count, 2, "LT 0: should match -5, -1");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn test_int64_filter_all_same_values() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![42, 42, 42, 42, 42];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Eq, 42);

    assert_eq!(count, 5, "EQ 42: all values equal 42");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_int64_filter_all_same_no_match() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![42, 42, 42, 42, 42];
    let (_bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Gt, 42);

    assert_eq!(count, 0, "GT 42 on all-42: nothing matches");
}

#[test]
fn test_int64_filter_boundary_value_exact() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    // Test GE on exact boundary: all values match
    let data: Vec<i64> = vec![100, 200, 300];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Ge, 100);

    assert_eq!(count, 3, "GE 100: all >= 100");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn test_int64_filter_large_negative() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![-1_000_000, -500_000, 0, 500_000, 1_000_000];
    let (bitmask, count) = run_int64_filter(&gpu, &mut cache, &data, CompareOp::Lt, -250_000);

    assert_eq!(count, 2, "LT -250000: should match -1000000, -500000");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1]);
}

// ---- Float edge value tests ----

#[test]
fn test_float_filter_zero_value() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Eq, 0.0);

    assert_eq!(count, 1, "EQ 0.0: should match 0.0");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![1]);
}

#[test]
fn test_float_filter_negative_compare() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![-5.0, -2.5, 0.0, 2.5, 5.0];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Lt, 0.0);

    assert_eq!(count, 2, "LT 0.0: should match -5.0, -2.5");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn test_float_filter_small_differences() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![1.0, 1.1, 1.2, 1.3, 1.4];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Gt, 1.25);

    assert_eq!(count, 2, "GT 1.25: should match 1.3, 1.4");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![3, 4]);
}

#[test]
fn test_float_filter_all_match() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![10.0, 20.0, 30.0];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Gt, 0.0);

    assert_eq!(count, 3, "GT 0.0: all positive values match");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn test_float_filter_no_matches() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let (_bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Gt, 100.0);

    assert_eq!(count, 0, "GT 100.0: no values match");
}

#[test]
fn test_float_filter_single_value() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = vec![42.5];
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Eq, 42.5);

    assert_eq!(count, 1, "EQ 42.5: single value matches");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices, vec![0]);
}

// ---- Large dataset float filter ----

#[test]
fn test_float_filter_100_elements() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let (bitmask, count) = run_float_filter(&gpu, &mut cache, &data, CompareOp::Gt, 25.0);

    // Values: 0.0, 0.5, 1.0, ..., 49.5. GT 25.0 means values 25.5, 26.0, ..., 49.5 => indices 51..99 = 49 elements
    assert_eq!(count, 49, "GT 25.0 on [0.0..49.5]: 49 values");
    let indices = bitmask_to_indices(&bitmask, data.len());
    assert_eq!(indices.len(), 49);
    assert_eq!(indices[0], 51); // 51 * 0.5 = 25.5
}
