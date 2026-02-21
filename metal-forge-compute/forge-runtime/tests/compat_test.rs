//! Backward compatibility tests: standalone ::new() vs shared ::with_context()
//! must produce identical results for both GpuSorter and GpuFilter.

use forge_filter::{GpuFilter, Predicate};
use forge_runtime::ForgeContext;
use forge_sort::GpuSorter;

/// Deterministic LCG data generator (same seed = same sequence).
fn gen_data(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as u32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GpuSorter: standalone new() works
// ---------------------------------------------------------------------------
#[test]
fn sort_standalone_new() {
    let mut sorter = GpuSorter::new().expect("GpuSorter::new() failed");
    let data = gen_data(10_000, 42);

    let mut buf = sorter.alloc_sort_buffer::<u32>(data.len());
    buf.copy_from_slice(&data);
    sorter.sort_buffer(&buf).expect("sort_buffer failed");

    let sorted = buf.as_slice();
    for i in 1..sorted.len() {
        assert!(
            sorted[i] >= sorted[i - 1],
            "standalone sort not ordered at [{i}]"
        );
    }
}

// ---------------------------------------------------------------------------
// GpuSorter: with_context() produces identical results
// ---------------------------------------------------------------------------
#[test]
fn sort_with_context_matches_standalone() {
    let data = gen_data(10_000, 42);

    // Standalone
    let mut standalone = GpuSorter::new().expect("standalone new");
    let mut buf_s = standalone.alloc_sort_buffer::<u32>(data.len());
    buf_s.copy_from_slice(&data);
    standalone.sort_buffer(&buf_s).expect("standalone sort");
    let result_standalone: Vec<u32> = buf_s.as_slice().to_vec();

    // Shared context
    let ctx = ForgeContext::new();
    let mut shared = GpuSorter::with_context(ctx.device(), ctx.queue()).expect("with_context");
    let mut buf_c = shared.alloc_sort_buffer::<u32>(data.len());
    buf_c.copy_from_slice(&data);
    shared.sort_buffer(&buf_c).expect("context sort");
    let result_context: Vec<u32> = buf_c.as_slice().to_vec();

    assert_eq!(
        result_standalone, result_context,
        "standalone and with_context sort results differ"
    );
}

// ---------------------------------------------------------------------------
// GpuFilter: standalone new() works
// ---------------------------------------------------------------------------
#[test]
fn filter_standalone_new() {
    let mut filter = GpuFilter::new().expect("GpuFilter::new() failed");
    let data = gen_data(10_000, 42);
    let pred = Predicate::Gt(50_000u32);

    let mut buf = filter.alloc_filter_buffer::<u32>(data.len());
    buf.copy_from_slice(&data);
    let result = filter.filter(&buf, &pred).expect("filter failed");

    assert!(result.len() > 0, "expected some values > 50_000");
    for &v in result.as_slice() {
        assert!(v > 50_000, "filter passed {v} <= 50_000");
    }
}

// ---------------------------------------------------------------------------
// GpuFilter: with_context() produces identical results
// ---------------------------------------------------------------------------
#[test]
fn filter_with_context_matches_standalone() {
    let data = gen_data(10_000, 42);
    let pred = Predicate::Gt(50_000u32);

    // Standalone
    let mut standalone = GpuFilter::new().expect("standalone new");
    let mut buf_s = standalone.alloc_filter_buffer::<u32>(data.len());
    buf_s.copy_from_slice(&data);
    let res_s = standalone.filter(&buf_s, &pred).expect("standalone filter");
    let result_standalone: Vec<u32> = res_s.as_slice().to_vec();

    // Shared context
    let ctx = ForgeContext::new();
    let mut shared = GpuFilter::with_context(ctx.device(), ctx.queue()).expect("with_context");
    let mut buf_c = shared.alloc_filter_buffer::<u32>(data.len());
    buf_c.copy_from_slice(&data);
    let res_c = shared.filter(&buf_c, &pred).expect("context filter");
    let result_context: Vec<u32> = res_c.as_slice().to_vec();

    assert_eq!(
        result_standalone.len(),
        result_context.len(),
        "count mismatch: standalone={} vs context={}",
        result_standalone.len(),
        result_context.len()
    );
    assert_eq!(
        result_standalone, result_context,
        "standalone and with_context filter results differ"
    );
}

// ---------------------------------------------------------------------------
// Combined: both primitives via shared context produce correct pipeline result
// ---------------------------------------------------------------------------
#[test]
fn combined_filter_sort_shared_vs_standalone() {
    let data = gen_data(50_000, 99);
    let pred = Predicate::Gt(100_000u32);

    // --- Standalone path (separate device/queue per primitive) ---
    let mut filter_s = GpuFilter::new().expect("standalone filter");
    let mut sorter_s = GpuSorter::new().expect("standalone sorter");

    let mut fbuf_s = filter_s.alloc_filter_buffer::<u32>(data.len());
    fbuf_s.copy_from_slice(&data);
    let fres_s = filter_s.filter(&fbuf_s, &pred).expect("filter");
    let filtered_standalone: Vec<u32> = fres_s.as_slice().to_vec();

    let mut sbuf_s = sorter_s.alloc_sort_buffer::<u32>(filtered_standalone.len());
    sbuf_s.copy_from_slice(&filtered_standalone);
    sorter_s.sort_buffer(&sbuf_s).expect("sort");
    let sorted_standalone: Vec<u32> = sbuf_s.as_slice().to_vec();

    // --- Shared context path ---
    let ctx = ForgeContext::new();
    let mut filter_c =
        GpuFilter::with_context(ctx.device(), ctx.queue()).expect("context filter");
    let mut sorter_c =
        GpuSorter::with_context(ctx.device(), ctx.queue()).expect("context sorter");

    let mut fbuf_c = filter_c.alloc_filter_buffer::<u32>(data.len());
    fbuf_c.copy_from_slice(&data);
    let fres_c = filter_c.filter(&fbuf_c, &pred).expect("filter");
    let filtered_context: Vec<u32> = fres_c.as_slice().to_vec();

    let mut sbuf_c = sorter_c.alloc_sort_buffer::<u32>(filtered_context.len());
    sbuf_c.copy_from_slice(&filtered_context);
    sorter_c.sort_buffer(&sbuf_c).expect("sort");
    let sorted_context: Vec<u32> = sbuf_c.as_slice().to_vec();

    // Both paths must yield identical results
    assert_eq!(
        sorted_standalone, sorted_context,
        "standalone vs shared-context filter->sort pipeline results differ"
    );

    // Sanity: result is sorted and all values pass predicate
    for i in 0..sorted_context.len() {
        assert!(sorted_context[i] > 100_000, "value <= predicate threshold");
        if i > 0 {
            assert!(
                sorted_context[i] >= sorted_context[i - 1],
                "not sorted at [{i}]"
            );
        }
    }
}
