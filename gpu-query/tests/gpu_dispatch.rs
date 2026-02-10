//! Integration tests for the indirect dispatch pipeline (prepare_query_dispatch kernel).
//!
//! Tests that the prepare_query_dispatch kernel correctly computes threadgroup counts
//! from match_count, and that indirect dispatch produces the same results as direct
//! dispatch for various filter selectivities.
//!
//! Pipeline: filter -> prepare_query_dispatch -> indirect aggregate
//! vs.
//! Pipeline: filter -> CPU readback match_count -> direct aggregate

use gpu_query::gpu::device::GpuDevice;
use gpu_query::gpu::encode;
use gpu_query::gpu::pipeline::{filter_pso_key, ColumnTypeCode, CompareOp, PsoCache};
use gpu_query::gpu::types::{AggParams, DispatchArgs, FilterParams};

use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState,
};

// ---- Helper: run filter and return (bitmask_buffer, match_count_buffer) ----

fn run_filter(
    gpu: &GpuDevice,
    pso_cache: &mut PsoCache,
    data: &[i64],
    compare_op: CompareOp,
    compare_value: i64,
) -> (
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>,
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>,
    u32, // row_count
) {
    let row_count = data.len() as u32;
    let bitmask_words = ((row_count + 31) / 32) as usize;

    let data_buffer = encode::alloc_buffer_with_data(&gpu.device, data);
    let bitmask_buffer = encode::alloc_buffer(&gpu.device, std::cmp::max(bitmask_words * 4, 4));
    let match_count_buffer = encode::alloc_buffer(&gpu.device, 4);
    let null_bitmap_buffer = encode::alloc_buffer(&gpu.device, std::cmp::max(bitmask_words * 4, 4));

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

    (bitmask_buffer, match_count_buffer, row_count)
}

// ---- Helper: run prepare_query_dispatch and read back DispatchArgs ----

fn run_prepare_dispatch(
    gpu: &GpuDevice,
    match_count_buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    threads_per_tg: u32,
) -> DispatchArgs {
    let dispatch_args_buffer = encode::alloc_buffer(
        &gpu.device,
        std::mem::size_of::<DispatchArgs>(),
    );
    let tpt_buffer = encode::alloc_buffer_with_data(&gpu.device, &[threads_per_tg]);

    let pipeline = encode::make_pipeline(&gpu.library, "prepare_query_dispatch");

    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(match_count_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&dispatch_args_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&tpt_buffer), 0, 2);
        }

        let grid = objc2_metal::MTLSize { width: 1, height: 1, depth: 1 };
        let tg = objc2_metal::MTLSize { width: 1, height: 1, depth: 1 };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    unsafe {
        let ptr = dispatch_args_buffer.contents().as_ptr() as *const DispatchArgs;
        *ptr
    }
}

// ---- Helper: run aggregate_count with direct dispatch ----

fn run_count_direct(
    gpu: &GpuDevice,
    mask_buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    row_count: u32,
) -> u32 {
    let num_words = ((row_count + 31) / 32) as usize;
    let result_buffer = encode::alloc_buffer(&gpu.device, 4);
    unsafe {
        let ptr = result_buffer.contents().as_ptr() as *mut u32;
        *ptr = 0;
    }

    let params = AggParams {
        row_count,
        group_count: 0,
        agg_function: 0,
        _pad0: 0,
    };
    let params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[params]);

    let pipeline = encode::make_pipeline(&gpu.library, "aggregate_count");

    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("compute encoder");

        encode::dispatch_1d(
            &encoder,
            &pipeline,
            &[(mask_buffer, 0), (&result_buffer, 1), (&params_buffer, 2)],
            num_words,
        );
        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    unsafe {
        let ptr = result_buffer.contents().as_ptr() as *const u32;
        *ptr
    }
}

// ---- Helper: run aggregate_count with indirect dispatch ----

fn run_count_indirect(
    gpu: &GpuDevice,
    mask_buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    dispatch_args_buffer: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    row_count: u32,
) -> u32 {
    let result_buffer = encode::alloc_buffer(&gpu.device, 4);
    unsafe {
        let ptr = result_buffer.contents().as_ptr() as *mut u32;
        *ptr = 0;
    }

    let params = AggParams {
        row_count,
        group_count: 0,
        agg_function: 0,
        _pad0: 0,
    };
    let params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[params]);

    let pipeline = encode::make_pipeline(&gpu.library, "aggregate_count");

    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("compute encoder");

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(mask_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&result_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
        }

        let tg_size = objc2_metal::MTLSize {
            width: pipeline.maxTotalThreadsPerThreadgroup().min(256),
            height: 1,
            depth: 1,
        };

        // Indirect dispatch: threadgroup count comes from GPU buffer
        unsafe {
            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                dispatch_args_buffer,
                0,
                tg_size,
            );
        }
        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    unsafe {
        let ptr = result_buffer.contents().as_ptr() as *const u32;
        *ptr
    }
}

// ---- Helper: run filter -> prepare_dispatch -> indirect aggregate_count in one command buffer ----

fn run_filter_indirect_count_single_cmdbuf(
    gpu: &GpuDevice,
    pso_cache: &mut PsoCache,
    data: &[i64],
    compare_op: CompareOp,
    compare_value: i64,
) -> u32 {
    let row_count = data.len() as u32;
    let bitmask_words = ((row_count + 31) / 32) as usize;

    // Allocate all buffers upfront
    let data_buffer = encode::alloc_buffer_with_data(&gpu.device, data);
    let bitmask_buffer = encode::alloc_buffer(&gpu.device, std::cmp::max(bitmask_words * 4, 4));
    let match_count_buffer = encode::alloc_buffer(&gpu.device, 4);
    let null_bitmap_buffer = encode::alloc_buffer(&gpu.device, std::cmp::max(bitmask_words * 4, 4));
    let dispatch_args_buffer = encode::alloc_buffer(
        &gpu.device,
        std::mem::size_of::<DispatchArgs>(),
    );
    let agg_result_buffer = encode::alloc_buffer(&gpu.device, 4);

    // Zero buffers
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
        let agg_ptr = agg_result_buffer.contents().as_ptr() as *mut u32;
        *agg_ptr = 0;
    }

    let filter_params = FilterParams {
        compare_value_int: compare_value,
        compare_value_float: 0.0,
        row_count,
        column_stride: 8,
        null_bitmap_present: 0,
        _pad0: 0,
        compare_value_int_hi: 0,
    };
    let filter_params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[filter_params]);

    let agg_params = AggParams {
        row_count,
        group_count: 0,
        agg_function: 0,
        _pad0: 0,
    };
    let agg_params_buffer = encode::alloc_buffer_with_data(&gpu.device, &[agg_params]);

    let filter_key = filter_pso_key(compare_op, ColumnTypeCode::Int64, false);
    let filter_pipeline = pso_cache.get_or_create(&gpu.library, &filter_key);
    let prepare_pipeline = encode::make_pipeline(&gpu.library, "prepare_query_dispatch");
    let count_pipeline = encode::make_pipeline(&gpu.library, "aggregate_count");

    let tpt = count_pipeline.maxTotalThreadsPerThreadgroup().min(256) as u32;
    let tpt_buffer = encode::alloc_buffer_with_data(&gpu.device, &[tpt]);

    // Single command buffer: filter -> prepare_dispatch -> indirect aggregate_count
    let cmd_buf = encode::make_command_buffer(&gpu.command_queue);

    // Stage 1: Filter
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("compute encoder");

        encode::dispatch_threads_1d(
            &encoder,
            filter_pipeline,
            &[
                (&data_buffer, 0),
                (&bitmask_buffer, 1),
                (&match_count_buffer, 2),
                (&filter_params_buffer, 3),
                (&null_bitmap_buffer, 4),
            ],
            row_count as usize,
        );
        encoder.endEncoding();
    }

    // Stage 2: Prepare dispatch args (single thread)
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("compute encoder");

        encoder.setComputePipelineState(&prepare_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&match_count_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&dispatch_args_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&tpt_buffer), 0, 2);
        }

        let grid = objc2_metal::MTLSize { width: 1, height: 1, depth: 1 };
        let tg = objc2_metal::MTLSize { width: 1, height: 1, depth: 1 };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();
    }

    // Stage 3: Aggregate count with indirect dispatch
    {
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("compute encoder");

        encoder.setComputePipelineState(&count_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&bitmask_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&agg_result_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&agg_params_buffer), 0, 2);
        }

        let tg_size = objc2_metal::MTLSize {
            width: tpt as usize,
            height: 1,
            depth: 1,
        };
        unsafe {
            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &dispatch_args_buffer,
                0,
                tg_size,
            );
        }
        encoder.endEncoding();
    }

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    unsafe {
        let ptr = agg_result_buffer.contents().as_ptr() as *const u32;
        *ptr
    }
}


// ============================================================
// Tests
// ============================================================

#[test]
fn test_prepare_dispatch_basic() {
    // Given a known match_count, verify DispatchArgs are computed correctly
    let gpu = GpuDevice::new();

    let match_count: u32 = 1000;
    let threads_per_tg: u32 = 256;
    let match_count_buffer = encode::alloc_buffer_with_data(&gpu.device, &[match_count]);

    let args = run_prepare_dispatch(&gpu, &match_count_buffer, threads_per_tg);

    let expected_tg = (1000 + 255) / 256; // = 4
    assert_eq!(args.threadgroups_x, expected_tg);
    assert_eq!(args.threadgroups_y, 1);
    assert_eq!(args.threadgroups_z, 1);
}

#[test]
fn test_prepare_dispatch_zero_matches() {
    // Zero matches should produce at least 1 threadgroup (max(0, 1) = 1)
    let gpu = GpuDevice::new();

    let match_count: u32 = 0;
    let match_count_buffer = encode::alloc_buffer_with_data(&gpu.device, &[match_count]);

    let args = run_prepare_dispatch(&gpu, &match_count_buffer, 256);

    assert_eq!(args.threadgroups_x, 1, "zero matches should still dispatch 1 threadgroup");
    assert_eq!(args.threadgroups_y, 1);
    assert_eq!(args.threadgroups_z, 1);
}

#[test]
fn test_prepare_dispatch_exact_multiple() {
    // match_count that is exact multiple of threads_per_tg
    let gpu = GpuDevice::new();

    let match_count: u32 = 512;
    let match_count_buffer = encode::alloc_buffer_with_data(&gpu.device, &[match_count]);

    let args = run_prepare_dispatch(&gpu, &match_count_buffer, 256);

    assert_eq!(args.threadgroups_x, 2);
    assert_eq!(args.threadgroups_y, 1);
    assert_eq!(args.threadgroups_z, 1);
}

#[test]
fn test_prepare_dispatch_one_match() {
    let gpu = GpuDevice::new();

    let match_count: u32 = 1;
    let match_count_buffer = encode::alloc_buffer_with_data(&gpu.device, &[match_count]);

    let args = run_prepare_dispatch(&gpu, &match_count_buffer, 256);

    assert_eq!(args.threadgroups_x, 1);
}

#[test]
fn test_prepare_dispatch_large_count() {
    let gpu = GpuDevice::new();

    let match_count: u32 = 1_000_000;
    let match_count_buffer = encode::alloc_buffer_with_data(&gpu.device, &[match_count]);

    let args = run_prepare_dispatch(&gpu, &match_count_buffer, 256);

    let expected = (1_000_000 + 255) / 256; // = 3907
    assert_eq!(args.threadgroups_x, expected);
}

#[test]
fn test_indirect_vs_direct_all_match() {
    // All rows match: indirect and direct should give same count
    let gpu = GpuDevice::new();
    let mut pso_cache = PsoCache::new();

    let data: Vec<i64> = (1..=1000).collect();
    let (bitmask, _match_count_buf, row_count) =
        run_filter(&gpu, &mut pso_cache, &data, CompareOp::Gt, 0);

    // Direct dispatch count
    let direct_count = run_count_direct(&gpu, &bitmask, row_count);

    // Prepare indirect dispatch args for aggregate (aggregate_count dispatches per bitmask word)
    let num_words = ((row_count + 31) / 32) as u32;
    let word_count_buffer = encode::alloc_buffer_with_data(&gpu.device, &[num_words]);
    let threads_per_tg = 256u32;
    let args = run_prepare_dispatch(&gpu, &word_count_buffer, threads_per_tg);

    // Create dispatch args buffer for indirect
    let dispatch_args_buffer = encode::alloc_buffer_with_data(
        &gpu.device,
        &[args],
    );
    let indirect_count = run_count_indirect(&gpu, &bitmask, &dispatch_args_buffer, row_count);

    assert_eq!(direct_count, 1000);
    assert_eq!(indirect_count, 1000);
    assert_eq!(direct_count, indirect_count);
}

#[test]
fn test_indirect_vs_direct_none_match() {
    // No rows match: both should give 0
    let gpu = GpuDevice::new();
    let mut pso_cache = PsoCache::new();

    let data: Vec<i64> = (1..=1000).collect();
    let (bitmask, _match_count_buf, row_count) =
        run_filter(&gpu, &mut pso_cache, &data, CompareOp::Gt, 2000);

    let direct_count = run_count_direct(&gpu, &bitmask, row_count);

    // Prepare indirect dispatch for bitmask word count
    let num_words = ((row_count + 31) / 32) as u32;
    let word_count_buffer = encode::alloc_buffer_with_data(&gpu.device, &[num_words]);
    let args = run_prepare_dispatch(&gpu, &word_count_buffer, 256);
    let dispatch_args_buffer = encode::alloc_buffer_with_data(&gpu.device, &[args]);
    let indirect_count = run_count_indirect(&gpu, &bitmask, &dispatch_args_buffer, row_count);

    assert_eq!(direct_count, 0);
    assert_eq!(indirect_count, 0);
}

#[test]
fn test_indirect_vs_direct_partial_match() {
    // ~50% selectivity
    let gpu = GpuDevice::new();
    let mut pso_cache = PsoCache::new();

    let data: Vec<i64> = (1..=1000).collect();
    let (bitmask, _match_count_buf, row_count) =
        run_filter(&gpu, &mut pso_cache, &data, CompareOp::Gt, 500);

    let direct_count = run_count_direct(&gpu, &bitmask, row_count);

    let num_words = ((row_count + 31) / 32) as u32;
    let word_count_buffer = encode::alloc_buffer_with_data(&gpu.device, &[num_words]);
    let args = run_prepare_dispatch(&gpu, &word_count_buffer, 256);
    let dispatch_args_buffer = encode::alloc_buffer_with_data(&gpu.device, &[args]);
    let indirect_count = run_count_indirect(&gpu, &bitmask, &dispatch_args_buffer, row_count);

    assert_eq!(direct_count, 500, "values 501..=1000 should match GT 500");
    assert_eq!(indirect_count, direct_count);
}

#[test]
fn test_single_cmdbuf_filter_indirect_aggregate() {
    // Full pipeline in one command buffer: filter -> prepare -> indirect aggregate
    // This is the actual pattern used in the executor
    let gpu = GpuDevice::new();
    let mut pso_cache = PsoCache::new();

    let data: Vec<i64> = (1..=1000).collect();
    let count = run_filter_indirect_count_single_cmdbuf(
        &gpu,
        &mut pso_cache,
        &data,
        CompareOp::Gt,
        500,
    );

    assert_eq!(count, 500, "single cmdbuf pipeline: 501..=1000 match GT 500");
}

#[test]
fn test_single_cmdbuf_all_match() {
    let gpu = GpuDevice::new();
    let mut pso_cache = PsoCache::new();

    let data: Vec<i64> = (1..=2048).collect();
    let count = run_filter_indirect_count_single_cmdbuf(
        &gpu,
        &mut pso_cache,
        &data,
        CompareOp::Gt,
        0,
    );

    assert_eq!(count, 2048);
}

#[test]
fn test_single_cmdbuf_none_match() {
    let gpu = GpuDevice::new();
    let mut pso_cache = PsoCache::new();

    let data: Vec<i64> = (1..=2048).collect();
    let count = run_filter_indirect_count_single_cmdbuf(
        &gpu,
        &mut pso_cache,
        &data,
        CompareOp::Gt,
        9999,
    );

    assert_eq!(count, 0);
}
