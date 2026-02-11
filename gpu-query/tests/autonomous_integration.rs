//! Integration tests for the fused query kernel across diverse query patterns.
//!
//! Tests 1-8 cover: COUNT(*), SUM, MIN/MAX, AVG, single filter, compound filter,
//! GROUP BY, and the full headline query (compound filter + GROUP BY + multi-agg).
//!
//! All tests use deterministic test data:
//!   Column 0 (amount): INT64, values = (i*7+13)%1000 for i in 0..1000
//!   Column 1 (region): INT64, values = i%5 for i in 0..1000

use gpu_query::gpu::autonomous::executor::{execute_fused_oneshot, FusedPsoCache};
use gpu_query::gpu::autonomous::loader::BinaryColumnarLoader;
use gpu_query::gpu::autonomous::types::{AggSpec, FilterSpec, OutputBuffer, QueryParamsSlot};
use gpu_query::gpu::device::GpuDevice;
use gpu_query::storage::columnar::ColumnarBatch;
use gpu_query::storage::schema::{ColumnDef, DataType, RuntimeSchema};

// ============================================================================
// Test helpers
// ============================================================================

/// Create a RuntimeSchema for the standard 2-column test table.
fn test_schema() -> RuntimeSchema {
    RuntimeSchema::new(vec![
        ColumnDef {
            name: "amount".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
        ColumnDef {
            name: "region".to_string(),
            data_type: DataType::Int64,
            nullable: false,
        },
    ])
}

/// Create deterministic test data with 2 INT64 columns:
///   Column 0 (amount): (i*7+13)%1000
///   Column 1 (region): i%5
fn make_test_batch(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    schema: &RuntimeSchema,
    row_count: usize,
) -> ColumnarBatch {
    use objc2_metal::MTLBuffer;

    let mut batch = ColumnarBatch::allocate(device, schema, row_count);
    batch.row_count = row_count;

    unsafe {
        let ptr = batch.int_buffer.contents().as_ptr() as *mut i64;

        // Column 0 (amount): local_int_idx=0, offset = 0 * max_rows
        for i in 0..row_count {
            *ptr.add(i) = ((i * 7 + 13) % 1000) as i64;
        }

        // Column 1 (region): local_int_idx=1, offset = 1 * max_rows
        let offset = batch.max_rows;
        for i in 0..row_count {
            *ptr.add(offset + i) = (i % 5) as i64;
        }
    }

    batch
}

/// Execute a fused oneshot query and return the OutputBuffer.
fn run_query(
    gpu: &GpuDevice,
    pso_cache: &mut FusedPsoCache,
    params: &QueryParamsSlot,
    filter_count: u32,
    agg_count: u32,
    has_group_by: bool,
    resident_table: &gpu_query::gpu::autonomous::loader::ResidentTable,
) -> OutputBuffer {
    let pso = pso_cache
        .get_or_compile(filter_count, agg_count, has_group_by)
        .expect("PSO compilation failed");

    execute_fused_oneshot(&gpu.device, &gpu.command_queue, pso, params, resident_table)
        .expect("execute_fused_oneshot failed")
}

/// CPU-side amount values for the test dataset.
fn cpu_amounts(row_count: usize) -> Vec<i64> {
    (0..row_count).map(|i| ((i * 7 + 13) % 1000) as i64).collect()
}

/// CPU-side region values for the test dataset.
fn cpu_regions(row_count: usize) -> Vec<i64> {
    (0..row_count).map(|i| (i % 5) as i64).collect()
}

/// Relative float comparison with tolerance.
fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    let denom = a.abs().max(b.abs()).max(1.0);
    ((a - b) / denom).abs() < tol
}

const ROW_COUNT: usize = 1000;

// ============================================================================
// Test 1: COUNT(*) no filter
// ============================================================================

#[test]
fn test_fused_count_star_no_filter() {
    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);

    let resident_table =
        BinaryColumnarLoader::load_table(&gpu.device, "test", &schema, &batch, None)
            .expect("load_table failed");

    let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

    let mut params = QueryParamsSlot::default();
    params.sequence_id = 1;
    params.filter_count = 0;
    params.agg_count = 1;
    params.aggs[0] = AggSpec {
        agg_func: 0, // COUNT
        column_idx: 0,
        column_type: 0, // INT64
        _pad0: 0,
    };
    params.has_group_by = 0;
    params.group_by_col = 0;
    params.row_count = ROW_COUNT as u32;

    let result = run_query(&gpu, &mut cache, &params, 0, 1, false, &resident_table);

    assert_eq!(result.ready_flag, 1, "ready_flag should be 1");
    assert_eq!(result.result_row_count, 1, "should have 1 result row (scalar)");
    assert_eq!(
        result.agg_results[0][0].value_int, ROW_COUNT as i64,
        "COUNT(*) should be {}",
        ROW_COUNT
    );
}

// ============================================================================
// Test 2: SUM(amount) no filter
// ============================================================================

#[test]
fn test_fused_sum_no_filter() {
    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);

    let resident_table =
        BinaryColumnarLoader::load_table(&gpu.device, "test", &schema, &batch, None)
            .expect("load_table failed");

    let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

    // CPU expected
    let amounts = cpu_amounts(ROW_COUNT);
    let expected_sum: i64 = amounts.iter().sum();

    let mut params = QueryParamsSlot::default();
    params.sequence_id = 2;
    params.filter_count = 0;
    params.agg_count = 1;
    params.aggs[0] = AggSpec {
        agg_func: 1, // SUM
        column_idx: 0,
        column_type: 0, // INT64
        _pad0: 0,
    };
    params.has_group_by = 0;
    params.group_by_col = 0;
    params.row_count = ROW_COUNT as u32;

    let result = run_query(&gpu, &mut cache, &params, 0, 1, false, &resident_table);

    assert_eq!(result.ready_flag, 1);
    assert_eq!(result.result_row_count, 1);
    assert_eq!(
        result.agg_results[0][0].value_int, expected_sum,
        "SUM(amount): expected {}, got {}",
        expected_sum, result.agg_results[0][0].value_int
    );
}

// ============================================================================
// Test 3: MIN/MAX(amount) no filter
// ============================================================================

#[test]
fn test_fused_min_max_no_filter() {
    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);

    let resident_table =
        BinaryColumnarLoader::load_table(&gpu.device, "test", &schema, &batch, None)
            .expect("load_table failed");

    let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

    let amounts = cpu_amounts(ROW_COUNT);
    let expected_min = *amounts.iter().min().unwrap();
    let expected_max = *amounts.iter().max().unwrap();

    // Two aggregates: MIN and MAX
    let mut params = QueryParamsSlot::default();
    params.sequence_id = 3;
    params.filter_count = 0;
    params.agg_count = 2;
    params.aggs[0] = AggSpec {
        agg_func: 3, // MIN
        column_idx: 0,
        column_type: 0, // INT64
        _pad0: 0,
    };
    params.aggs[1] = AggSpec {
        agg_func: 4, // MAX
        column_idx: 0,
        column_type: 0, // INT64
        _pad0: 0,
    };
    params.has_group_by = 0;
    params.group_by_col = 0;
    params.row_count = ROW_COUNT as u32;

    let result = run_query(&gpu, &mut cache, &params, 0, 2, false, &resident_table);

    assert_eq!(result.ready_flag, 1);
    assert_eq!(result.result_row_count, 1);
    assert_eq!(
        result.agg_results[0][0].value_int, expected_min,
        "MIN(amount): expected {}, got {}",
        expected_min, result.agg_results[0][0].value_int
    );
    assert_eq!(
        result.agg_results[0][1].value_int, expected_max,
        "MAX(amount): expected {}, got {}",
        expected_max, result.agg_results[0][1].value_int
    );
}

// ============================================================================
// Test 4: AVG(amount) no filter
// ============================================================================

#[test]
fn test_fused_avg_no_filter() {
    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);

    let resident_table =
        BinaryColumnarLoader::load_table(&gpu.device, "test", &schema, &batch, None)
            .expect("load_table failed");

    let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

    let amounts = cpu_amounts(ROW_COUNT);
    let expected_sum: i64 = amounts.iter().sum();
    let expected_avg = expected_sum as f64 / ROW_COUNT as f64;

    // AVG is implemented as SUM + COUNT, result read from value_int (sum) and count
    let mut params = QueryParamsSlot::default();
    params.sequence_id = 4;
    params.filter_count = 0;
    params.agg_count = 1;
    params.aggs[0] = AggSpec {
        agg_func: 2, // AVG
        column_idx: 0,
        column_type: 0, // INT64
        _pad0: 0,
    };
    params.has_group_by = 0;
    params.group_by_col = 0;
    params.row_count = ROW_COUNT as u32;

    let result = run_query(&gpu, &mut cache, &params, 0, 1, false, &resident_table);

    assert_eq!(result.ready_flag, 1);
    assert_eq!(result.result_row_count, 1);

    // AVG for INT64 columns: kernel stores sum in value_int, count in count
    let gpu_sum = result.agg_results[0][0].value_int;
    let gpu_count = result.agg_results[0][0].count;
    let gpu_avg = gpu_sum as f64 / gpu_count as f64;

    assert!(
        approx_eq(gpu_avg, expected_avg, 1e-5),
        "AVG(amount): expected {}, got {} (sum={}, count={})",
        expected_avg,
        gpu_avg,
        gpu_sum,
        gpu_count
    );
}

// ============================================================================
// Test 5: Single filter — COUNT(*) WHERE amount > 500
// ============================================================================

#[test]
fn test_fused_single_filter_gt() {
    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);

    let resident_table =
        BinaryColumnarLoader::load_table(&gpu.device, "test", &schema, &batch, None)
            .expect("load_table failed");

    let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

    // CPU expected
    let amounts = cpu_amounts(ROW_COUNT);
    let expected_count: i64 = amounts.iter().filter(|&&v| v > 500).count() as i64;

    let mut params = QueryParamsSlot::default();
    params.sequence_id = 5;
    params.filter_count = 1;
    params.filters[0] = FilterSpec {
        column_idx: 0,
        compare_op: 4, // GT
        column_type: 0, // INT64
        _pad0: 0,
        value_int: 500,
        value_float_bits: 0,
        _pad1: 0,
        has_null_check: 0,
        _pad2: [0; 3],
    };
    params.agg_count = 1;
    params.aggs[0] = AggSpec {
        agg_func: 0, // COUNT
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params.has_group_by = 0;
    params.group_by_col = 0;
    params.row_count = ROW_COUNT as u32;

    let result = run_query(&gpu, &mut cache, &params, 1, 1, false, &resident_table);

    assert_eq!(result.ready_flag, 1);
    assert_eq!(result.result_row_count, 1);
    assert_eq!(
        result.agg_results[0][0].value_int, expected_count,
        "COUNT(*) WHERE amount > 500: expected {}, got {}",
        expected_count, result.agg_results[0][0].value_int
    );
}

// ============================================================================
// Test 6: Compound AND filter — COUNT(*) WHERE amount > 200 AND amount < 800
// ============================================================================

#[test]
fn test_fused_compound_filter() {
    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);

    let resident_table =
        BinaryColumnarLoader::load_table(&gpu.device, "test", &schema, &batch, None)
            .expect("load_table failed");

    let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

    // CPU expected
    let amounts = cpu_amounts(ROW_COUNT);
    let expected_count: i64 = amounts.iter().filter(|&&v| v > 200 && v < 800).count() as i64;

    let mut params = QueryParamsSlot::default();
    params.sequence_id = 6;
    params.filter_count = 2;
    params.filters[0] = FilterSpec {
        column_idx: 0,
        compare_op: 4, // GT
        column_type: 0, // INT64
        _pad0: 0,
        value_int: 200,
        value_float_bits: 0,
        _pad1: 0,
        has_null_check: 0,
        _pad2: [0; 3],
    };
    params.filters[1] = FilterSpec {
        column_idx: 0,
        compare_op: 2, // LT
        column_type: 0, // INT64
        _pad0: 0,
        value_int: 800,
        value_float_bits: 0,
        _pad1: 0,
        has_null_check: 0,
        _pad2: [0; 3],
    };
    params.agg_count = 1;
    params.aggs[0] = AggSpec {
        agg_func: 0, // COUNT
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params.has_group_by = 0;
    params.group_by_col = 0;
    params.row_count = ROW_COUNT as u32;

    let result = run_query(&gpu, &mut cache, &params, 2, 1, false, &resident_table);

    assert_eq!(result.ready_flag, 1);
    assert_eq!(result.result_row_count, 1);
    assert_eq!(
        result.agg_results[0][0].value_int, expected_count,
        "COUNT(*) WHERE amount > 200 AND amount < 800: expected {}, got {}",
        expected_count, result.agg_results[0][0].value_int
    );
}

// ============================================================================
// Test 7: GROUP BY region (no filter) — COUNT per group
// ============================================================================

#[test]
fn test_fused_group_by_no_filter() {
    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);

    let resident_table =
        BinaryColumnarLoader::load_table(&gpu.device, "test", &schema, &batch, None)
            .expect("load_table failed");

    let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

    // CPU expected: 5 groups (region 0,1,2,3,4), each with 200 rows
    let amounts = cpu_amounts(ROW_COUNT);
    let regions = cpu_regions(ROW_COUNT);

    // Per-group expected count and sum
    let mut expected_count = [0i64; 5];
    let mut expected_sum = [0i64; 5];
    for i in 0..ROW_COUNT {
        let g = regions[i] as usize;
        expected_count[g] += 1;
        expected_sum[g] += amounts[i];
    }

    let mut params = QueryParamsSlot::default();
    params.sequence_id = 7;
    params.filter_count = 0;
    params.agg_count = 2;
    params.aggs[0] = AggSpec {
        agg_func: 0, // COUNT
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params.aggs[1] = AggSpec {
        agg_func: 1, // SUM(amount)
        column_idx: 0,
        column_type: 0, // INT64
        _pad0: 0,
    };
    params.has_group_by = 1;
    params.group_by_col = 1; // region column
    params.row_count = ROW_COUNT as u32;

    let result = run_query(&gpu, &mut cache, &params, 0, 2, true, &resident_table);

    assert_eq!(result.ready_flag, 1);
    assert_eq!(
        result.result_row_count, 5,
        "Should have 5 groups, got {}",
        result.result_row_count
    );

    // Verify each group. The kernel hashes group keys as abs(key) % MAX_GROUPS.
    // For keys 0,1,2,3,4 the bucket indices are 0,1,2,3,4 directly.
    for g in 0..5usize {
        let bucket = g; // abs(g) % 64 = g for g in 0..5
        let gpu_count = result.agg_results[bucket][0].value_int;
        let gpu_sum = result.agg_results[bucket][1].value_int;

        assert_eq!(
            gpu_count, expected_count[g],
            "Group {}: COUNT expected {}, got {}",
            g, expected_count[g], gpu_count
        );
        assert_eq!(
            gpu_sum, expected_sum[g],
            "Group {}: SUM expected {}, got {}",
            g, expected_sum[g], gpu_sum
        );

        // Verify group key
        assert_eq!(
            result.group_keys[bucket], g as i64,
            "Group key at bucket {}: expected {}, got {}",
            bucket, g, result.group_keys[bucket]
        );
    }
}

// ============================================================================
// Test 8: Headline query — compound filter + GROUP BY + multi-agg
//   WHERE amount > 200 AND amount < 800
//   GROUP BY region
//   SELECT COUNT(*), SUM(amount), MIN(amount), MAX(amount)
// ============================================================================

#[test]
fn test_fused_headline_query() {
    let gpu = GpuDevice::new();
    let schema = test_schema();
    let batch = make_test_batch(&gpu.device, &schema, ROW_COUNT);

    let resident_table =
        BinaryColumnarLoader::load_table(&gpu.device, "test", &schema, &batch, None)
            .expect("load_table failed");

    let mut cache = FusedPsoCache::new(gpu.device.clone(), gpu.library.clone());

    // CPU expected: filter first, then group, then aggregate
    let amounts = cpu_amounts(ROW_COUNT);
    let regions = cpu_regions(ROW_COUNT);

    let mut expected_count = [0i64; 5];
    let mut expected_sum = [0i64; 5];
    let mut expected_min = [i64::MAX; 5];
    let mut expected_max = [i64::MIN; 5];

    for i in 0..ROW_COUNT {
        let v = amounts[i];
        if v > 200 && v < 800 {
            let g = regions[i] as usize;
            expected_count[g] += 1;
            expected_sum[g] += v;
            if v < expected_min[g] {
                expected_min[g] = v;
            }
            if v > expected_max[g] {
                expected_max[g] = v;
            }
        }
    }

    // Count how many groups have data
    let active_groups: usize = expected_count.iter().filter(|&&c| c > 0).count();

    let mut params = QueryParamsSlot::default();
    params.sequence_id = 8;
    params.filter_count = 2;
    params.filters[0] = FilterSpec {
        column_idx: 0,
        compare_op: 4, // GT
        column_type: 0,
        _pad0: 0,
        value_int: 200,
        value_float_bits: 0,
        _pad1: 0,
        has_null_check: 0,
        _pad2: [0; 3],
    };
    params.filters[1] = FilterSpec {
        column_idx: 0,
        compare_op: 2, // LT
        column_type: 0,
        _pad0: 0,
        value_int: 800,
        value_float_bits: 0,
        _pad1: 0,
        has_null_check: 0,
        _pad2: [0; 3],
    };
    params.agg_count = 4;
    params.aggs[0] = AggSpec {
        agg_func: 0, // COUNT
        column_idx: 0,
        column_type: 0,
        _pad0: 0,
    };
    params.aggs[1] = AggSpec {
        agg_func: 1, // SUM
        column_idx: 0,
        column_type: 0, // INT64
        _pad0: 0,
    };
    params.aggs[2] = AggSpec {
        agg_func: 3, // MIN
        column_idx: 0,
        column_type: 0, // INT64
        _pad0: 0,
    };
    params.aggs[3] = AggSpec {
        agg_func: 4, // MAX
        column_idx: 0,
        column_type: 0, // INT64
        _pad0: 0,
    };
    params.has_group_by = 1;
    params.group_by_col = 1; // region column
    params.row_count = ROW_COUNT as u32;

    let result = run_query(&gpu, &mut cache, &params, 2, 4, true, &resident_table);

    assert_eq!(result.ready_flag, 1);
    assert_eq!(
        result.result_row_count, active_groups as u32,
        "Should have {} active groups, got {}",
        active_groups, result.result_row_count
    );

    // Verify each group
    for g in 0..5usize {
        if expected_count[g] == 0 {
            continue;
        }

        let bucket = g; // abs(g) % 64 = g

        // COUNT
        assert_eq!(
            result.agg_results[bucket][0].value_int, expected_count[g],
            "Group {}: COUNT expected {}, got {}",
            g, expected_count[g], result.agg_results[bucket][0].value_int
        );
        // SUM
        assert_eq!(
            result.agg_results[bucket][1].value_int, expected_sum[g],
            "Group {}: SUM expected {}, got {}",
            g, expected_sum[g], result.agg_results[bucket][1].value_int
        );
        // MIN
        assert_eq!(
            result.agg_results[bucket][2].value_int, expected_min[g],
            "Group {}: MIN expected {}, got {}",
            g, expected_min[g], result.agg_results[bucket][2].value_int
        );
        // MAX
        assert_eq!(
            result.agg_results[bucket][3].value_int, expected_max[g],
            "Group {}: MAX expected {}, got {}",
            g, expected_max[g], result.agg_results[bucket][3].value_int
        );

        // Verify group key
        assert_eq!(
            result.group_keys[bucket], g as i64,
            "Group key at bucket {}: expected {}, got {}",
            bucket, g, result.group_keys[bucket]
        );
    }
}
