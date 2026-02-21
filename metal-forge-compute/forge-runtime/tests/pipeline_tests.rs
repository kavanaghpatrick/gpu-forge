//! Integration tests for Pipeline: filter->sort, filter->sort->gather,
//! sort->gather, zero-match filter, and multi-type coverage.

use forge_filter::{GpuFilter, Predicate};
use forge_runtime::{filter_result_to_sort_buffer, ForgeContext, GpuGather, Pipeline};
use forge_sort::GpuSorter;
use objc2_metal::MTLBuffer as _;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// LCG deterministic RNG (same as POC tests).
fn lcg_u32(seed: u32, n: usize) -> Vec<u32> {
    let mut vals = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        vals.push(s);
    }
    vals
}

/// Write `data` into a freshly-allocated MTLBuffer.
fn write_buf<T: Copy>(ctx: &ForgeContext, data: &[T]) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>> {
    let buf = ctx.alloc_buffer(data.len() * std::mem::size_of::<T>());
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents().as_ptr() as *mut T,
            data.len(),
        );
    }
    buf
}

/// Read `n` elements of type `T` from an MTLBuffer.
unsafe fn read_buf<T: Copy>(buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>, n: usize) -> Vec<T> {
    let ptr = buf.contents().as_ptr() as *const T;
    std::slice::from_raw_parts(ptr, n).to_vec()
}

// ---------------------------------------------------------------------------
// 1. Filter -> Sort (u32, 1M elements)
// ---------------------------------------------------------------------------

#[test]
fn pipeline_filter_then_sort_u32_1m() {
    let ctx = ForgeContext::new();
    let mut sorter = GpuSorter::with_context(ctx.device(), ctx.queue()).unwrap();
    let mut filter = GpuFilter::with_context(ctx.device(), ctx.queue()).unwrap();

    let n = 1_000_000usize;
    let data = lcg_u32(42, n);
    let threshold = 500_000u32;

    // Load data into a Metal buffer
    let input_buf = write_buf(&ctx, &data);

    // Pipeline: filter Gt(500_000) then sort
    let pipeline = Pipeline::new(&ctx).unwrap();
    let pending = pipeline
        .filter(&mut filter, &input_buf, n, &Predicate::Gt(threshold))
        .unwrap();

    // After filter, we need the result to build a SortBuffer.
    // But the count isn't known until after execute. We must execute first,
    // then resolve, then do sort in a *second* pipeline.
    pipeline.execute().unwrap();

    let filter_result = pending.resolve();
    let count = filter_result.len();
    assert!(count > 0, "expected some matches for Gt(500000)");

    // Convert to SortBuffer and sort in a second pipeline
    let sort_buf = filter_result_to_sort_buffer::<u32>(filter_result)
        .expect("filter_result should have values buffer");

    let pipeline2 = Pipeline::new(&ctx).unwrap();
    pipeline2.sort(&mut sorter, &sort_buf).unwrap();
    pipeline2.execute().unwrap();

    // Verify: all elements > threshold and sorted ascending
    let result = sort_buf.as_slice();
    assert_eq!(result.len(), count);
    for (i, &v) in result.iter().enumerate() {
        assert!(v > threshold, "element {} = {} should be > {}", i, v, threshold);
        if i > 0 {
            assert!(result[i - 1] <= v, "not sorted at index {}: {} > {}", i, result[i - 1], v);
        }
    }

    // Cross-validate against CPU reference
    let mut cpu_ref: Vec<u32> = data.iter().copied().filter(|&x| x > threshold).collect();
    cpu_ref.sort();
    assert_eq!(result.len(), cpu_ref.len());
    assert_eq!(result, cpu_ref.as_slice());
}

// ---------------------------------------------------------------------------
// 2. Filter -> Sort -> Gather (top 1000)
// ---------------------------------------------------------------------------

#[test]
fn pipeline_filter_sort_gather_top_1000() {
    let ctx = ForgeContext::new();
    let mut sorter = GpuSorter::with_context(ctx.device(), ctx.queue()).unwrap();
    let mut filter = GpuFilter::with_context(ctx.device(), ctx.queue()).unwrap();
    let gather = GpuGather::with_context(ctx.device(), ctx.queue()).unwrap();

    let n = 1_000_000usize;
    let data = lcg_u32(99, n);
    let threshold = 500_000u32;

    let input_buf = write_buf(&ctx, &data);

    // Step 1: Filter
    let pipeline = Pipeline::new(&ctx).unwrap();
    let pending = pipeline
        .filter(&mut filter, &input_buf, n, &Predicate::Gt(threshold))
        .unwrap();
    pipeline.execute().unwrap();
    let filter_result = pending.resolve();

    // Step 2: Sort
    let sort_buf = filter_result_to_sort_buffer::<u32>(filter_result)
        .expect("should have values buffer");
    let sorted_count = sort_buf.len();
    assert!(sorted_count >= 1000, "need at least 1000 matches, got {}", sorted_count);

    let pipeline2 = Pipeline::new(&ctx).unwrap();
    pipeline2.sort(&mut sorter, &sort_buf).unwrap();

    // Step 3: Gather top 1000 (first 1000 after ascending sort = smallest 1000)
    let gather_count = 1000u32;
    let indices: Vec<u32> = (0..gather_count).collect();
    let indices_buf = write_buf(&ctx, &indices);
    let output_buf = ctx.alloc_buffer(gather_count as usize * std::mem::size_of::<u32>());

    pipeline2.gather(
        &gather,
        sort_buf.metal_buffer(),
        &indices_buf,
        &output_buf,
        gather_count,
        false,
    );
    pipeline2.execute().unwrap();

    // Verify gathered elements
    let gathered: Vec<u32> = unsafe { read_buf(&output_buf, gather_count as usize) };
    assert_eq!(gathered.len(), 1000);
    for (i, &v) in gathered.iter().enumerate() {
        assert!(v > threshold, "gathered[{}] = {} should be > {}", i, v, threshold);
        if i > 0 {
            assert!(gathered[i - 1] <= v, "gathered not sorted at {}", i);
        }
    }

    // Cross-validate
    let mut cpu_ref: Vec<u32> = data.iter().copied().filter(|&x| x > threshold).collect();
    cpu_ref.sort();
    let cpu_top_1000 = &cpu_ref[..1000];
    assert_eq!(gathered.as_slice(), cpu_top_1000);
}

// ---------------------------------------------------------------------------
// 3. Sort -> Gather (100K elements, gather first 100)
// ---------------------------------------------------------------------------

#[test]
fn pipeline_sort_then_gather_100k() {
    let ctx = ForgeContext::new();
    let mut sorter = GpuSorter::with_context(ctx.device(), ctx.queue()).unwrap();
    let gather = GpuGather::with_context(ctx.device(), ctx.queue()).unwrap();

    let n = 100_000usize;
    let data = lcg_u32(7, n);

    // Allocate sort buffer and fill
    let mut sort_buf = sorter.alloc_sort_buffer::<u32>(n);
    sort_buf.copy_from_slice(&data);

    // Pipeline: sort then gather first 100
    let gather_count = 100u32;
    let indices: Vec<u32> = (0..gather_count).collect();
    let indices_buf = write_buf(&ctx, &indices);
    let output_buf = ctx.alloc_buffer(gather_count as usize * std::mem::size_of::<u32>());

    let pipeline = Pipeline::new(&ctx).unwrap();
    pipeline.sort(&mut sorter, &sort_buf).unwrap();
    pipeline.gather(
        &gather,
        sort_buf.metal_buffer(),
        &indices_buf,
        &output_buf,
        gather_count,
        false,
    );
    pipeline.execute().unwrap();

    // Verify: gathered should be the 100 smallest elements, sorted
    let gathered: Vec<u32> = unsafe { read_buf(&output_buf, gather_count as usize) };
    let mut cpu_ref = data.clone();
    cpu_ref.sort();
    let cpu_first_100 = &cpu_ref[..100];
    assert_eq!(gathered.as_slice(), cpu_first_100);
}

// ---------------------------------------------------------------------------
// 4. Filter with zero matches
// ---------------------------------------------------------------------------

#[test]
fn pipeline_filter_zero_matches() {
    let ctx = ForgeContext::new();
    let mut filter = GpuFilter::with_context(ctx.device(), ctx.queue()).unwrap();

    // All values in [0, 9999], filter Gt(u32::MAX - 1) => 0 matches
    let n = 10_000usize;
    let data: Vec<u32> = (0..n as u32).collect();
    let input_buf = write_buf(&ctx, &data);

    let pipeline = Pipeline::new(&ctx).unwrap();
    let pending = pipeline
        .filter(&mut filter, &input_buf, n, &Predicate::Gt(u32::MAX - 1))
        .unwrap();
    pipeline.execute().unwrap();

    let result = pending.resolve();
    assert_eq!(result.len(), 0, "expected zero matches");
    assert!(result.is_empty());

    // filter_result_to_sort_buffer on empty result should still work
    // (values buffer exists but count = 0)
    let sort_buf_opt = filter_result_to_sort_buffer::<u32>(result);
    if let Some(sb) = sort_buf_opt {
        assert_eq!(sb.len(), 0);
    }
    // Either None or Some with len=0 is acceptable
}

// ---------------------------------------------------------------------------
// 5. Filter with empty input (n=0)
// ---------------------------------------------------------------------------

#[test]
fn pipeline_filter_empty_input() {
    let ctx = ForgeContext::new();
    let mut filter = GpuFilter::with_context(ctx.device(), ctx.queue()).unwrap();

    // Allocate a 4-byte buffer (can't alloc 0 bytes on Metal)
    let input_buf = ctx.alloc_buffer(4);

    let pipeline = Pipeline::new(&ctx).unwrap();
    let pending = pipeline
        .filter(&mut filter, &input_buf, 0, &Predicate::Gt(0u32))
        .unwrap();
    pipeline.execute().unwrap();

    let result = pending.resolve();
    assert_eq!(result.len(), 0);
}

// ---------------------------------------------------------------------------
// 6. Numeric type tests: u32 filter
// ---------------------------------------------------------------------------

#[test]
fn pipeline_filter_u32() {
    let ctx = ForgeContext::new();
    let mut filter = GpuFilter::with_context(ctx.device(), ctx.queue()).unwrap();

    let data: Vec<u32> = (0..10_000u32).collect();
    let input_buf = write_buf(&ctx, &data);

    let pipeline = Pipeline::new(&ctx).unwrap();
    let pending = pipeline
        .filter(&mut filter, &input_buf, data.len(), &Predicate::Ge(5000u32))
        .unwrap();
    pipeline.execute().unwrap();

    let result = pending.resolve();
    let values = result.to_vec();
    assert_eq!(values.len(), 5000);
    for &v in &values {
        assert!(v >= 5000);
    }

    // CPU reference
    let cpu: Vec<u32> = data.iter().copied().filter(|&x| x >= 5000).collect();
    assert_eq!(values, cpu);
}

// ---------------------------------------------------------------------------
// 7. Numeric type tests: i32 filter
// ---------------------------------------------------------------------------

#[test]
fn pipeline_filter_i32() {
    let ctx = ForgeContext::new();
    let mut filter = GpuFilter::with_context(ctx.device(), ctx.queue()).unwrap();

    // Range [-5000, 4999]
    let data: Vec<i32> = (-5000..5000i32).collect();
    let input_buf = write_buf(&ctx, &data);

    let pipeline = Pipeline::new(&ctx).unwrap();
    let pending = pipeline
        .filter(&mut filter, &input_buf, data.len(), &Predicate::Gt(0i32))
        .unwrap();
    pipeline.execute().unwrap();

    let result = pending.resolve();
    let values = result.to_vec();
    // Values > 0: 1..=4999 = 4999 elements
    assert_eq!(values.len(), 4999, "expected 4999 positive values");
    for &v in &values {
        assert!(v > 0, "expected > 0, got {}", v);
    }

    let cpu: Vec<i32> = data.iter().copied().filter(|&x| x > 0).collect();
    assert_eq!(values, cpu);
}

// ---------------------------------------------------------------------------
// 8. Numeric type tests: f32 filter
// ---------------------------------------------------------------------------

#[test]
fn pipeline_filter_f32() {
    let ctx = ForgeContext::new();
    let mut filter = GpuFilter::with_context(ctx.device(), ctx.queue()).unwrap();

    // 10K floats from 0.0 to 9999.0
    let data: Vec<f32> = (0..10_000u32).map(|i| i as f32).collect();
    let input_buf = write_buf(&ctx, &data);

    let pipeline = Pipeline::new(&ctx).unwrap();
    let pending = pipeline
        .filter(&mut filter, &input_buf, data.len(), &Predicate::Lt(100.0f32))
        .unwrap();
    pipeline.execute().unwrap();

    let result = pending.resolve();
    let values = result.to_vec();
    // Values < 100.0: 0.0..99.0 = 100 elements
    assert_eq!(values.len(), 100, "expected 100 values < 100.0");
    for &v in &values {
        assert!(v < 100.0, "expected < 100.0, got {}", v);
    }

    let cpu: Vec<f32> = data.iter().copied().filter(|&x| x < 100.0).collect();
    assert_eq!(values, cpu);
}

// ---------------------------------------------------------------------------
// 9. Verify pipeline results match standalone sequential execution
// ---------------------------------------------------------------------------

#[test]
fn pipeline_matches_standalone_filter_sort() {
    let ctx = ForgeContext::new();
    let mut sorter = GpuSorter::with_context(ctx.device(), ctx.queue()).unwrap();
    let mut filter = GpuFilter::with_context(ctx.device(), ctx.queue()).unwrap();

    let n = 100_000usize;
    let data = lcg_u32(123, n);
    let threshold = 2_000_000_000u32;

    // --- Standalone sequential path (separate CBs) ---
    let mut standalone_filter_buf = filter.alloc_filter_buffer::<u32>(n);
    standalone_filter_buf.copy_from_slice(&data);
    let standalone_result = filter
        .filter(&standalone_filter_buf, &Predicate::Gt(threshold))
        .unwrap();
    let standalone_values = standalone_result.to_vec();
    let standalone_sort_buf = filter_result_to_sort_buffer::<u32>(standalone_result)
        .expect("standalone should have values");
    // Sort standalone (uses its own CB internally)
    sorter.sort_buffer(&standalone_sort_buf).unwrap();
    let standalone_sorted = standalone_sort_buf.as_slice().to_vec();

    // --- Pipeline path (single CB per stage) ---
    let input_buf = write_buf(&ctx, &data);

    let pipeline = Pipeline::new(&ctx).unwrap();
    let pending = pipeline
        .filter(&mut filter, &input_buf, n, &Predicate::Gt(threshold))
        .unwrap();
    pipeline.execute().unwrap();
    let filter_result = pending.resolve();
    let pipeline_values = filter_result.to_vec();

    let sort_buf = filter_result_to_sort_buffer::<u32>(filter_result)
        .expect("pipeline should have values");

    let pipeline2 = Pipeline::new(&ctx).unwrap();
    pipeline2.sort(&mut sorter, &sort_buf).unwrap();
    pipeline2.execute().unwrap();
    let pipeline_sorted = sort_buf.as_slice().to_vec();

    // Both paths should produce identical results
    assert_eq!(standalone_values.len(), pipeline_values.len(),
        "filter counts differ: standalone={} pipeline={}", standalone_values.len(), pipeline_values.len());

    // Filter results may differ in order (GPU scheduling), but sorted results must match
    assert_eq!(standalone_sorted.len(), pipeline_sorted.len());
    assert_eq!(standalone_sorted, pipeline_sorted,
        "sorted output differs between standalone and pipeline");
}
