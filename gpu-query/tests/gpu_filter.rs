//! Integration tests for the GPU column filter kernel.
//!
//! Tests the filter pipeline: create column data -> allocate bitmask/count buffers ->
//! create specialized PSO via function constants -> dispatch filter kernel ->
//! read back bitmask + match_count -> verify correctness.

use gpu_query::gpu::device::GpuDevice;
use gpu_query::gpu::encode;
use gpu_query::gpu::pipeline::{
    filter_pso_key, ColumnTypeCode, CompareOp, PsoCache,
};
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
    let cat = catalog::scan_directory(dir).expect("scan directory");
    let logical_plan = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let physical_plan =
        gpu_query::sql::physical_plan::plan(&logical_plan).expect("plan SQL");
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
    assert_eq!(result.rows[0][0], "3", "Should match 3 rows where region = 'Europe'");
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
    assert_eq!(result.rows[0][0], "0", "Nonexistent value should return 0 rows");
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
    assert_eq!(result.rows[0][0], "900", "SUM of Europe amounts should be 900");
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
