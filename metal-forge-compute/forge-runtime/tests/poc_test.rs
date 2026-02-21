//! POC integration test: filter 100K random u32, then sort the survivors.
//!
//! Proves that filter output flows into sort via `filter_result_to_sort_buffer`,
//! sharing a single Metal device/queue through `ForgeContext`.

use forge_filter::{GpuFilter, Predicate};
use forge_runtime::{filter_result_to_sort_buffer, ForgeContext};
use forge_sort::GpuSorter;

#[test]
fn poc_filter_then_sort() {
    // 1. Shared context (one device + queue)
    let ctx = ForgeContext::new();

    // 2. Create primitives sharing the context
    let mut filter = GpuFilter::with_context(ctx.device(), ctx.queue())
        .expect("GpuFilter::with_context failed");
    let mut sorter = GpuSorter::with_context(ctx.device(), ctx.queue())
        .expect("GpuSorter::with_context failed");

    // 3. Generate 100K random-ish u32 values (deterministic LCG for reproducibility)
    let n = 100_000usize;
    let mut data = Vec::with_capacity(n);
    let mut rng: u64 = 42;
    for _ in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        data.push((rng >> 33) as u32);
    }

    // 4. Filter: keep values > 50_000
    let threshold = 50_000u32;
    let pred = Predicate::Gt(threshold);

    // Use the buffer-based API: allocate FilterBuffer, copy data, filter
    let mut fbuf = filter.alloc_filter_buffer::<u32>(n);
    fbuf.copy_from_slice(&data);
    let filter_result = filter.filter(&fbuf, &pred).expect("filter failed");

    let filtered_count = filter_result.len();
    assert!(filtered_count > 0, "expected some values > 50_000");

    // Sanity: filtered values must all be > threshold
    let filtered_vals = filter_result.as_slice();
    for &v in filtered_vals {
        assert!(v > threshold, "filter passed through value {v} <= {threshold}");
    }

    // 5. Convert FilterResult -> SortBuffer (zero-copy buffer hand-off)
    let sort_buf =
        filter_result_to_sort_buffer(filter_result).expect("conversion returned None");
    assert_eq!(sort_buf.len(), filtered_count);

    // 6. Sort in-place
    sorter.sort_buffer(&sort_buf).expect("sort failed");

    // 7. Verify: sorted ascending AND all values > threshold
    let sorted = sort_buf.as_slice();
    assert_eq!(sorted.len(), filtered_count);
    for i in 0..sorted.len() {
        assert!(
            sorted[i] > threshold,
            "sorted[{i}] = {} <= {threshold}",
            sorted[i]
        );
        if i > 0 {
            assert!(
                sorted[i] >= sorted[i - 1],
                "not sorted: sorted[{}] = {} < sorted[{}] = {}",
                i - 1,
                sorted[i - 1],
                i,
                sorted[i]
            );
        }
    }

    // Quick summary
    eprintln!(
        "POC passed: {n} input -> {filtered_count} filtered -> sorted OK (min={}, max={})",
        sorted.first().unwrap(),
        sorted.last().unwrap()
    );
}
