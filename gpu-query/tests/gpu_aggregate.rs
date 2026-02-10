//! Integration tests for all GPU aggregation kernels.
//!
//! Tests: COUNT, SUM (int64 + float), MIN (int64), MAX (int64), AVG,
//! and GROUP BY (CPU-side).
//!
//! Tests the aggregation pipeline: create data + selection_mask -> dispatch
//! aggregate kernel -> read back result -> verify against CPU reference.
//!
//! Uses the filter pipeline from task 1.6 to generate selection masks for
//! realistic filtered aggregation tests.

use gpu_query::gpu::device::GpuDevice;
use gpu_query::gpu::encode;
use gpu_query::gpu::pipeline::{
    filter_pso_key, ColumnTypeCode, CompareOp, PsoCache,
};
use gpu_query::gpu::types::{AggParams, FilterParams};

use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState,
};

// ---- Helper: build an all-ones bitmask (no filter) ----

fn build_all_ones_mask(device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>, row_count: usize) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>> {
    let num_words = (row_count + 31) / 32;
    let buf = encode::alloc_buffer(device, num_words * 4);
    unsafe {
        let ptr = buf.contents().as_ptr() as *mut u32;
        for i in 0..num_words {
            if i == num_words - 1 {
                let valid_bits = row_count % 32;
                if valid_bits == 0 {
                    *ptr.add(i) = 0xFFFFFFFF;
                } else {
                    *ptr.add(i) = (1u32 << valid_bits) - 1;
                }
            } else {
                *ptr.add(i) = 0xFFFFFFFF;
            }
        }
    }
    buf
}

// ---- Helper: build bitmask from filter ----

fn run_filter_to_bitmask(
    gpu: &GpuDevice,
    pso_cache: &mut PsoCache,
    data: &[i64],
    compare_op: CompareOp,
    compare_value: i64,
) -> (objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>, u32) {
    let row_count = data.len() as u32;
    let bitmask_words = ((row_count + 31) / 32) as usize;

    let data_buffer = encode::alloc_buffer_with_data(&gpu.device, data);
    let bitmask_buffer = encode::alloc_buffer(&gpu.device, bitmask_words * 4);
    let match_count_buffer = encode::alloc_buffer(&gpu.device, 4);
    let null_bitmap_buffer = encode::alloc_buffer(&gpu.device, bitmask_words * 4);

    // Zero bitmask and match_count
    unsafe {
        let bitmask_ptr = bitmask_buffer.contents().as_ptr() as *mut u32;
        for i in 0..bitmask_words {
            *bitmask_ptr.add(i) = 0;
        }
        let count_ptr = match_count_buffer.contents().as_ptr() as *mut u32;
        *count_ptr = 0;
        let null_ptr = null_bitmap_buffer.contents().as_ptr() as *mut u32;
        for i in 0..bitmask_words {
            *null_ptr.add(i) = 0;
        }
    }

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

    let key = filter_pso_key(compare_op, ColumnTypeCode::Int64, false);
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

    let match_count = unsafe {
        let ptr = match_count_buffer.contents().as_ptr() as *const u32;
        *ptr
    };

    (bitmask_buffer, match_count)
}

// ---- Helper: run aggregate_count kernel ----

fn run_aggregate_count(
    gpu: &GpuDevice,
    mask_buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    row_count: u32,
) -> u64 {
    let num_words = ((row_count + 31) / 32) as usize;

    // Result buffer: single uint32 (atomic_uint)
    let result_buffer = encode::alloc_buffer(&gpu.device, 4);
    unsafe {
        let ptr = result_buffer.contents().as_ptr() as *mut u32;
        *ptr = 0;
    }

    let params = AggParams {
        row_count,
        group_count: 0,
        agg_function: 0, // COUNT
        _pad0: 0,
    };
    let params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[params]);

    let pipeline = encode::make_pipeline(&gpu.library, "aggregate_count");

    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encode::dispatch_1d(
            &encoder,
            &pipeline,
            &[
                (mask_buffer, 0),
                (&result_buffer, 1),
                (&params_buffer, 2),
            ],
            num_words,
        );

        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    unsafe {
        let ptr = result_buffer.contents().as_ptr() as *const u32;
        *ptr as u64
    }
}

// ---- Helper: run aggregate_sum_int64 kernel ----

fn run_aggregate_sum_int64(
    gpu: &GpuDevice,
    data: &[i64],
    mask_buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    row_count: u32,
) -> i64 {
    let data_buffer = encode::alloc_buffer_with_data(&gpu.device, data);

    // Result: two uint32 atomics (lo + hi) = 8 bytes
    let result_buffer = encode::alloc_buffer(&gpu.device, 8);
    unsafe {
        let ptr = result_buffer.contents().as_ptr() as *mut u32;
        *ptr = 0;       // lo
        *ptr.add(1) = 0; // hi
    }

    let params = AggParams {
        row_count,
        group_count: 0,
        agg_function: 1, // SUM
        _pad0: 0,
    };
    let params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[params]);

    let pipeline = encode::make_pipeline(&gpu.library, "aggregate_sum_int64");

    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        // buffer(0): column_data
        // buffer(1): selection_mask
        // buffer(2): result_lo (first uint32 of result_buffer)
        // buffer(3): result_hi (second uint32 of result_buffer, set via offset)
        // buffer(4): params
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&data_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(mask_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&result_buffer), 0, 2);   // result_lo
            encoder.setBuffer_offset_atIndex(Some(&result_buffer), 4, 3);   // result_hi (offset 4 bytes)
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 4);
        }

        let threads_per_tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
        let threadgroup_count = (row_count as usize + threads_per_tg - 1) / threads_per_tg;

        let grid_size = objc2_metal::MTLSize {
            width: threadgroup_count,
            height: 1,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    // Read back lo + hi and reconstruct int64
    unsafe {
        let ptr = result_buffer.contents().as_ptr() as *const u32;
        let lo = *ptr as u64;
        let hi = *ptr.add(1) as u64;
        let val = lo | (hi << 32);
        val as i64
    }
}

// ---- CPU reference helpers ----

fn cpu_count_filtered(data: &[i64], compare_op: CompareOp, threshold: i64) -> u64 {
    data.iter()
        .filter(|&&v| match compare_op {
            CompareOp::Gt => v > threshold,
            CompareOp::Ge => v >= threshold,
            CompareOp::Lt => v < threshold,
            CompareOp::Le => v <= threshold,
            CompareOp::Eq => v == threshold,
            CompareOp::Ne => v != threshold,
        })
        .count() as u64
}

fn cpu_sum_filtered(data: &[i64], compare_op: CompareOp, threshold: i64) -> i64 {
    data.iter()
        .filter(|&&v| match compare_op {
            CompareOp::Gt => v > threshold,
            CompareOp::Ge => v >= threshold,
            CompareOp::Lt => v < threshold,
            CompareOp::Le => v <= threshold,
            CompareOp::Eq => v == threshold,
            CompareOp::Ne => v != threshold,
        })
        .sum()
}

// ============================================================
// COUNT tests
// ============================================================

#[test]
fn test_count_filtered_gt_25() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (mask, _match_count) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Gt, 25);

    let gpu_count = run_aggregate_count(&gpu, &mask, data.len() as u32);
    let cpu_count = cpu_count_filtered(&data, CompareOp::Gt, 25);

    assert_eq!(gpu_count, cpu_count, "COUNT filtered GT 25: expected {}, got {}", cpu_count, gpu_count);
    assert_eq!(gpu_count, 3);
}

#[test]
fn test_count_unfiltered() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_count = run_aggregate_count(&gpu, &mask, data.len() as u32);
    assert_eq!(gpu_count, 5, "COUNT(*) unfiltered should be 5");
}

#[test]
fn test_count_no_matches() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30];
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Gt, 100);

    let gpu_count = run_aggregate_count(&gpu, &mask, data.len() as u32);
    assert_eq!(gpu_count, 0, "COUNT with no matches should be 0");
}

#[test]
fn test_count_all_match() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30];
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Gt, 0);

    let gpu_count = run_aggregate_count(&gpu, &mask, data.len() as u32);
    assert_eq!(gpu_count, 3, "COUNT with all matching should be 3");
}

#[test]
fn test_count_single_row() {
    let gpu = GpuDevice::new();

    let mask = build_all_ones_mask(&gpu.device, 1);
    let gpu_count = run_aggregate_count(&gpu, &mask, 1);
    assert_eq!(gpu_count, 1, "COUNT of single row");
}

#[test]
fn test_count_exactly_32_rows() {
    let gpu = GpuDevice::new();

    // 32 rows = exactly 1 bitmask word
    let mask = build_all_ones_mask(&gpu.device, 32);
    let gpu_count = run_aggregate_count(&gpu, &mask, 32);
    assert_eq!(gpu_count, 32, "COUNT of exactly 32 rows");
}

#[test]
fn test_count_33_rows() {
    let gpu = GpuDevice::new();

    // 33 rows = 2 bitmask words, last word has 1 valid bit
    let mask = build_all_ones_mask(&gpu.device, 33);
    let gpu_count = run_aggregate_count(&gpu, &mask, 33);
    assert_eq!(gpu_count, 33, "COUNT of 33 rows (crosses word boundary)");
}

// ============================================================
// SUM tests
// ============================================================

#[test]
fn test_sum_filtered_gt_25() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Gt, 25);

    let gpu_sum = run_aggregate_sum_int64(&gpu, &data, &mask, data.len() as u32);
    let cpu_sum = cpu_sum_filtered(&data, CompareOp::Gt, 25);

    assert_eq!(gpu_sum, cpu_sum, "SUM filtered GT 25: expected {}, got {}", cpu_sum, gpu_sum);
    assert_eq!(gpu_sum, 120); // 30 + 40 + 50
}

#[test]
fn test_sum_unfiltered() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_sum = run_aggregate_sum_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_sum, 150, "SUM unfiltered should be 150");
}

#[test]
fn test_sum_no_matches() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30];
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Gt, 100);

    let gpu_sum = run_aggregate_sum_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_sum, 0, "SUM with no matches should be 0");
}

#[test]
fn test_sum_single_value() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![42];
    let mask = build_all_ones_mask(&gpu.device, 1);

    let gpu_sum = run_aggregate_sum_int64(&gpu, &data, &mask, 1);
    assert_eq!(gpu_sum, 42, "SUM of single value");
}

#[test]
fn test_sum_negative_values() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![-10, -20, 30, 40, -50];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_sum = run_aggregate_sum_int64(&gpu, &data, &mask, data.len() as u32);
    let cpu_sum: i64 = data.iter().sum();
    assert_eq!(gpu_sum, cpu_sum, "SUM with negative values: expected {}, got {}", cpu_sum, gpu_sum);
    assert_eq!(gpu_sum, -10); // -10 + -20 + 30 + 40 + -50 = -10
}

// ============================================================
// Larger dataset tests (1000 elements)
// ============================================================

#[test]
fn test_count_1000_elements() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    // 1000 elements: 0..999, filter GT 499 -> 500 matches
    let data: Vec<i64> = (0..1000).collect();
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Gt, 499);

    let gpu_count = run_aggregate_count(&gpu, &mask, data.len() as u32);
    let cpu_count = cpu_count_filtered(&data, CompareOp::Gt, 499);

    assert_eq!(gpu_count, cpu_count, "COUNT 1000 elements GT 499");
    assert_eq!(gpu_count, 500);
}

#[test]
fn test_sum_1000_elements() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = (0..1000).collect();
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Gt, 499);

    let gpu_sum = run_aggregate_sum_int64(&gpu, &data, &mask, data.len() as u32);
    let cpu_sum = cpu_sum_filtered(&data, CompareOp::Gt, 499);

    assert_eq!(gpu_sum, cpu_sum, "SUM 1000 elements GT 499: expected {}, got {}", cpu_sum, gpu_sum);
    // sum of 500..999 = (500+999)*500/2 = 374750
    assert_eq!(gpu_sum, 374750);
}

#[test]
fn test_count_1000_unfiltered() {
    let gpu = GpuDevice::new();

    let mask = build_all_ones_mask(&gpu.device, 1000);
    let gpu_count = run_aggregate_count(&gpu, &mask, 1000);
    assert_eq!(gpu_count, 1000, "COUNT(*) on 1000 rows");
}

#[test]
fn test_sum_1000_unfiltered() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = (0..1000).collect();
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_sum = run_aggregate_sum_int64(&gpu, &data, &mask, data.len() as u32);
    let cpu_sum: i64 = data.iter().sum();
    assert_eq!(gpu_sum, cpu_sum, "SUM 0..999 unfiltered: expected {}, got {}", cpu_sum, gpu_sum);
    assert_eq!(gpu_sum, 499500); // sum of 0..999
}

// ============================================================
// Edge cases
// ============================================================

#[test]
fn test_sum_large_values() {
    let gpu = GpuDevice::new();

    // Values that exercise 64-bit range
    let data: Vec<i64> = vec![1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_sum = run_aggregate_sum_int64(&gpu, &data, &mask, data.len() as u32);
    let cpu_sum: i64 = data.iter().sum();
    assert_eq!(gpu_sum, cpu_sum, "SUM of large int64 values");
    assert_eq!(gpu_sum, 10_000_000_000i64);
}

#[test]
fn test_sum_zeros() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![0, 0, 0, 0, 0];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_sum = run_aggregate_sum_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_sum, 0, "SUM of all zeros");
}

#[test]
fn test_count_filtered_eq() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 20, 30, 20];
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Eq, 20);

    let gpu_count = run_aggregate_count(&gpu, &mask, data.len() as u32);
    assert_eq!(gpu_count, 3, "COUNT where value == 20");
}

#[test]
fn test_sum_filtered_eq() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 20, 30, 20];
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Eq, 20);

    let gpu_sum = run_aggregate_sum_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_sum, 60, "SUM where value == 20: 3 * 20 = 60");
}

// ============================================================
// Helper: run aggregate_min_int64 kernel
// ============================================================

fn run_aggregate_min_int64(
    gpu: &GpuDevice,
    data: &[i64],
    mask_buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    row_count: u32,
) -> i64 {
    let data_buffer = encode::alloc_buffer_with_data(&gpu.device, data);

    let pipeline = encode::make_pipeline(&gpu.library, "aggregate_min_int64");
    let threads_per_tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
    let threadgroup_count = (row_count as usize + threads_per_tg - 1) / threads_per_tg;

    // Partials buffer: one i64 per threadgroup, initialized to INT64_MAX
    let partials_buffer = encode::alloc_buffer(
        &gpu.device,
        threadgroup_count * std::mem::size_of::<i64>(),
    );
    unsafe {
        let ptr = partials_buffer.contents().as_ptr() as *mut i64;
        for i in 0..threadgroup_count {
            *ptr.add(i) = i64::MAX;
        }
    }

    let params = AggParams {
        row_count,
        group_count: 0,
        agg_function: 3, // MIN
        _pad0: 0,
    };
    let params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[params]);

    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&data_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(mask_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&partials_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        }

        let grid_size = objc2_metal::MTLSize {
            width: threadgroup_count,
            height: 1,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    // CPU final reduction over partials
    unsafe {
        let ptr = partials_buffer.contents().as_ptr() as *const i64;
        let mut min_val = i64::MAX;
        for i in 0..threadgroup_count {
            let v = *ptr.add(i);
            if v < min_val {
                min_val = v;
            }
        }
        min_val
    }
}

// ============================================================
// Helper: run aggregate_max_int64 kernel
// ============================================================

fn run_aggregate_max_int64(
    gpu: &GpuDevice,
    data: &[i64],
    mask_buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    row_count: u32,
) -> i64 {
    let data_buffer = encode::alloc_buffer_with_data(&gpu.device, data);

    let pipeline = encode::make_pipeline(&gpu.library, "aggregate_max_int64");
    let threads_per_tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
    let threadgroup_count = (row_count as usize + threads_per_tg - 1) / threads_per_tg;

    // Partials buffer: one i64 per threadgroup, initialized to INT64_MIN
    let partials_buffer = encode::alloc_buffer(
        &gpu.device,
        threadgroup_count * std::mem::size_of::<i64>(),
    );
    unsafe {
        let ptr = partials_buffer.contents().as_ptr() as *mut i64;
        for i in 0..threadgroup_count {
            *ptr.add(i) = i64::MIN;
        }
    }

    let params = AggParams {
        row_count,
        group_count: 0,
        agg_function: 4, // MAX
        _pad0: 0,
    };
    let params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[params]);

    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&data_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(mask_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&partials_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        }

        let grid_size = objc2_metal::MTLSize {
            width: threadgroup_count,
            height: 1,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    // CPU final reduction over partials
    unsafe {
        let ptr = partials_buffer.contents().as_ptr() as *const i64;
        let mut max_val = i64::MIN;
        for i in 0..threadgroup_count {
            let v = *ptr.add(i);
            if v > max_val {
                max_val = v;
            }
        }
        max_val
    }
}

// ============================================================
// Helper: run aggregate_sum_float kernel
// ============================================================

fn run_aggregate_sum_float(
    gpu: &GpuDevice,
    data: &[f32],
    mask_buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    row_count: u32,
) -> f32 {
    let data_buffer = encode::alloc_buffer_with_data(&gpu.device, data);

    // Result: single uint32 storing float bits, initialized to 0.0f
    let result_buffer = encode::alloc_buffer(&gpu.device, 4);
    unsafe {
        let ptr = result_buffer.contents().as_ptr() as *mut u32;
        *ptr = 0; // 0.0f as bits
    }

    let params = AggParams {
        row_count,
        group_count: 0,
        agg_function: 1, // SUM
        _pad0: 0,
    };
    let params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[params]);

    let pipeline = encode::make_pipeline(&gpu.library, "aggregate_sum_float");

    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&data_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(mask_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&result_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        }

        let threads_per_tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
        let threadgroup_count = (row_count as usize + threads_per_tg - 1) / threads_per_tg;

        let grid_size = objc2_metal::MTLSize {
            width: threadgroup_count,
            height: 1,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    unsafe {
        let ptr = result_buffer.contents().as_ptr() as *const u32;
        f32::from_bits(*ptr)
    }
}

// ============================================================
// MIN tests
// ============================================================

#[test]
fn test_min_basic() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![50, 10, 30, 20, 40];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_min = run_aggregate_min_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_min, 10, "MIN of [50,10,30,20,40] should be 10");
}

#[test]
fn test_min_single_value() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![42];
    let mask = build_all_ones_mask(&gpu.device, 1);

    let gpu_min = run_aggregate_min_int64(&gpu, &data, &mask, 1);
    assert_eq!(gpu_min, 42, "MIN of single value");
}

#[test]
fn test_min_negative_values() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![-10, 5, -50, 20, -3];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_min = run_aggregate_min_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_min, -50, "MIN with negatives should be -50");
}

#[test]
fn test_min_filtered() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Gt, 25);

    let gpu_min = run_aggregate_min_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_min, 30, "MIN of values > 25 should be 30");
}

#[test]
fn test_min_1000_elements() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = (100..1100).collect();
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_min = run_aggregate_min_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_min, 100, "MIN of 100..1099 should be 100");
}

#[test]
fn test_min_all_same() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![42; 100];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_min = run_aggregate_min_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_min, 42, "MIN of all-same should be 42");
}

// ============================================================
// MAX tests
// ============================================================

#[test]
fn test_max_basic() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![50, 10, 30, 20, 40];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_max = run_aggregate_max_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_max, 50, "MAX of [50,10,30,20,40] should be 50");
}

#[test]
fn test_max_single_value() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![42];
    let mask = build_all_ones_mask(&gpu.device, 1);

    let gpu_max = run_aggregate_max_int64(&gpu, &data, &mask, 1);
    assert_eq!(gpu_max, 42, "MAX of single value");
}

#[test]
fn test_max_negative_values() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![-10, -5, -50, -20, -3];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_max = run_aggregate_max_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_max, -3, "MAX with all-negatives should be -3");
}

#[test]
fn test_max_filtered() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    let data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &data, CompareOp::Lt, 35);

    let gpu_max = run_aggregate_max_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_max, 30, "MAX of values < 35 should be 30");
}

#[test]
fn test_max_1000_elements() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = (0..1000).collect();
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_max = run_aggregate_max_int64(&gpu, &data, &mask, data.len() as u32);
    assert_eq!(gpu_max, 999, "MAX of 0..999 should be 999");
}

#[test]
fn test_min_max_combined() {
    let gpu = GpuDevice::new();

    let data: Vec<i64> = vec![-100, 0, 50, 200, -300];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_min = run_aggregate_min_int64(&gpu, &data, &mask, data.len() as u32);
    let gpu_max = run_aggregate_max_int64(&gpu, &data, &mask, data.len() as u32);

    assert_eq!(gpu_min, -300, "MIN should be -300");
    assert_eq!(gpu_max, 200, "MAX should be 200");
}

// ============================================================
// SUM float tests
// ============================================================

#[test]
fn test_sum_float_basic() {
    let gpu = GpuDevice::new();

    let data: Vec<f32> = vec![1.5, 2.5, 3.0, 4.0, 5.0];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_sum = run_aggregate_sum_float(&gpu, &data, &mask, data.len() as u32);
    let cpu_sum: f32 = data.iter().sum();

    assert!(
        (gpu_sum - cpu_sum).abs() < 0.01,
        "SUM float: expected {}, got {}",
        cpu_sum,
        gpu_sum
    );
}

#[test]
fn test_sum_float_filtered() {
    let gpu = GpuDevice::new();
    let mut cache = PsoCache::new();

    // Use int filter, then sum different float data
    let int_data: Vec<i64> = vec![10, 20, 30, 40, 50];
    let (mask, _) = run_filter_to_bitmask(&gpu, &mut cache, &int_data, CompareOp::Gt, 25);

    // Only rows 2,3,4 are selected (values 30,40,50)
    let float_data: Vec<f32> = vec![1.0, 2.0, 3.5, 4.5, 5.5];
    let gpu_sum = run_aggregate_sum_float(&gpu, &float_data, &mask, float_data.len() as u32);

    // Expected: 3.5 + 4.5 + 5.5 = 13.5
    assert!(
        (gpu_sum - 13.5).abs() < 0.01,
        "SUM float filtered: expected 13.5, got {}",
        gpu_sum
    );
}

#[test]
fn test_sum_float_zeros() {
    let gpu = GpuDevice::new();

    let data: Vec<f32> = vec![0.0, 0.0, 0.0];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_sum = run_aggregate_sum_float(&gpu, &data, &mask, data.len() as u32);
    assert!(
        gpu_sum.abs() < 0.001,
        "SUM float zeros: expected 0, got {}",
        gpu_sum
    );
}

#[test]
fn test_sum_float_negative() {
    let gpu = GpuDevice::new();

    let data: Vec<f32> = vec![-1.5, 2.5, -3.0, 4.0];
    let mask = build_all_ones_mask(&gpu.device, data.len());

    let gpu_sum = run_aggregate_sum_float(&gpu, &data, &mask, data.len() as u32);
    let cpu_sum: f32 = data.iter().sum();

    assert!(
        (gpu_sum - cpu_sum).abs() < 0.01,
        "SUM float with negatives: expected {}, got {}",
        cpu_sum,
        gpu_sum
    );
}

// ============================================================
// End-to-end executor tests for all aggregate functions + GROUP BY
// ============================================================

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

/// Helper: create a test CSV, scan it with the catalog, and execute a query.
fn run_query(csv_content: &str, sql: &str) -> gpu_query::gpu::executor::QueryResult {
    let tmp_dir = tempfile::tempdir().expect("tempdir");
    let csv_path = tmp_dir.path().join("test.csv");
    std::fs::write(&csv_path, csv_content).expect("write csv");

    let catalog_entries = catalog::scan_directory(tmp_dir.path()).expect("scan");
    let plan = {
        let parsed = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
        gpu_query::sql::physical_plan::plan(&parsed).expect("plan")
    };

    let mut executor = QueryExecutor::new().expect("executor");
    executor.execute(&plan, &catalog_entries).expect("execute")
}

#[test]
fn test_executor_count_star() {
    let csv = "id,amount\n1,100\n2,200\n3,300\n4,400\n5,500\n";
    let result = run_query(csv, "SELECT count(*) FROM test");
    assert_eq!(result.rows[0][0], "5");
}

#[test]
fn test_executor_sum() {
    let csv = "id,amount\n1,100\n2,200\n3,300\n4,400\n5,500\n";
    let result = run_query(csv, "SELECT sum(amount) FROM test");
    assert_eq!(result.rows[0][0], "1500");
}

#[test]
fn test_executor_min() {
    let csv = "id,amount\n1,300\n2,100\n3,500\n4,200\n5,400\n";
    let result = run_query(csv, "SELECT min(amount) FROM test");
    assert_eq!(result.rows[0][0], "100");
}

#[test]
fn test_executor_max() {
    let csv = "id,amount\n1,300\n2,100\n3,500\n4,200\n5,400\n";
    let result = run_query(csv, "SELECT max(amount) FROM test");
    assert_eq!(result.rows[0][0], "500");
}

#[test]
fn test_executor_avg() {
    let csv = "id,amount\n1,100\n2,200\n3,300\n4,400\n5,500\n";
    let result = run_query(csv, "SELECT avg(amount) FROM test");
    // AVG = 1500 / 5 = 300.0, should display as "300"
    assert_eq!(result.rows[0][0], "300");
}

#[test]
fn test_executor_multiple_aggregates() {
    let csv = "id,amount\n1,100\n2,200\n3,300\n4,400\n5,500\n";
    let result = run_query(
        csv,
        "SELECT count(*), sum(amount), min(amount), max(amount), avg(amount) FROM test",
    );
    assert_eq!(result.columns.len(), 5);
    assert_eq!(result.rows[0][0], "5");     // count
    assert_eq!(result.rows[0][1], "1500");  // sum
    assert_eq!(result.rows[0][2], "100");   // min
    assert_eq!(result.rows[0][3], "500");   // max
    assert_eq!(result.rows[0][4], "300");   // avg
}

#[test]
fn test_executor_filtered_aggregates() {
    let csv = "id,amount\n1,100\n2,200\n3,300\n4,400\n5,500\n";
    let result = run_query(
        csv,
        "SELECT count(*), sum(amount), min(amount), max(amount) FROM test WHERE amount > 200",
    );
    assert_eq!(result.rows[0][0], "3");     // count: 300,400,500
    assert_eq!(result.rows[0][1], "1200");  // sum: 300+400+500
    assert_eq!(result.rows[0][2], "300");   // min
    assert_eq!(result.rows[0][3], "500");   // max
}

#[test]
fn test_executor_group_by_int() {
    let csv = "region,amount\n1,100\n2,200\n1,300\n2,400\n1,500\n";
    let result = run_query(
        csv,
        "SELECT region, count(*), sum(amount) FROM test GROUP BY region",
    );
    // Two groups: region=1 (rows: 100,300,500) and region=2 (rows: 200,400)
    assert_eq!(result.row_count, 2);

    // Rows should be sorted by group key
    // Region 1: count=3, sum=900
    assert_eq!(result.rows[0][0], "1");
    assert_eq!(result.rows[0][1], "3");
    assert_eq!(result.rows[0][2], "900");

    // Region 2: count=2, sum=600
    assert_eq!(result.rows[1][0], "2");
    assert_eq!(result.rows[1][1], "2");
    assert_eq!(result.rows[1][2], "600");
}

#[test]
fn test_executor_group_by_all_aggregates() {
    let csv = "region,amount\n1,100\n2,200\n1,300\n2,400\n1,500\n";
    let result = run_query(
        csv,
        "SELECT region, count(*), sum(amount), avg(amount), min(amount), max(amount) FROM test GROUP BY region",
    );
    assert_eq!(result.row_count, 2);

    // Region 1: count=3, sum=900, avg=300, min=100, max=500
    assert_eq!(result.rows[0][0], "1");
    assert_eq!(result.rows[0][1], "3");
    assert_eq!(result.rows[0][2], "900");
    assert_eq!(result.rows[0][3], "300");
    assert_eq!(result.rows[0][4], "100");
    assert_eq!(result.rows[0][5], "500");

    // Region 2: count=2, sum=600, avg=300, min=200, max=400
    assert_eq!(result.rows[1][0], "2");
    assert_eq!(result.rows[1][1], "2");
    assert_eq!(result.rows[1][2], "600");
    assert_eq!(result.rows[1][3], "300");
    assert_eq!(result.rows[1][4], "200");
    assert_eq!(result.rows[1][5], "400");
}

#[test]
fn test_executor_group_by_single_group() {
    // All rows same group
    let csv = "region,amount\n1,100\n1,200\n1,300\n";
    let result = run_query(
        csv,
        "SELECT region, count(*), sum(amount) FROM test GROUP BY region",
    );
    assert_eq!(result.row_count, 1);
    assert_eq!(result.rows[0][0], "1");
    assert_eq!(result.rows[0][1], "3");
    assert_eq!(result.rows[0][2], "600");
}

#[test]
fn test_executor_group_by_filtered() {
    let csv = "region,amount\n1,100\n2,200\n1,300\n2,400\n1,500\n";
    let result = run_query(
        csv,
        "SELECT region, count(*), sum(amount) FROM test WHERE amount > 200 GROUP BY region",
    );
    // After filter (amount > 200): region=1: [300,500], region=2: [400]
    assert_eq!(result.row_count, 2);

    // Region 1: count=2, sum=800
    assert_eq!(result.rows[0][0], "1");
    assert_eq!(result.rows[0][1], "2");
    assert_eq!(result.rows[0][2], "800");

    // Region 2: count=1, sum=400
    assert_eq!(result.rows[1][0], "2");
    assert_eq!(result.rows[1][1], "1");
    assert_eq!(result.rows[1][2], "400");
}
