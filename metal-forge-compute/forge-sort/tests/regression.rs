mod common;

use forge_sort::GpuSorter;
use rand::Rng;

/// Verify sort_u32 behavior is unchanged from v1.
#[test]
fn test_v1_compat_sort_u32_unchanged() {
    let mut rng = common::seeded_rng(100);
    let n = 1_000_000;
    let data: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut expected = data.clone();
    expected.sort();

    let mut actual = data;
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u32(&mut actual).unwrap();
    assert_eq!(actual, expected);
}

/// Verify sort_buffer with SortBuffer<u32> is unchanged from v1.
#[test]
fn test_v1_compat_sort_buffer_u32() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut rng = common::seeded_rng(101);
    let n = 1_000_000;
    let data: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut expected = data.clone();
    expected.sort();

    let mut buf = sorter.alloc_sort_buffer(n);
    buf.copy_from_slice(&data);
    sorter.sort_buffer(&buf).unwrap();
    assert_eq!(buf.as_slice(), &expected[..]);
}

/// Sort u32 -> f32 -> i32 -> u32 on the same GpuSorter, all must be correct.
#[test]
fn test_cross_type_interleave() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut rng = common::seeded_rng(200);
    let n = 100_000;

    // u32
    let mut u32_data: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut u32_exp = u32_data.clone();
    u32_exp.sort();
    sorter.sort_u32(&mut u32_data).unwrap();
    assert_eq!(u32_data, u32_exp);

    // f32
    let mut f32_data: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 2000.0 - 1000.0).collect();
    let mut f32_exp = f32_data.clone();
    f32_exp.sort_by(f32::total_cmp);
    sorter.sort_f32(&mut f32_data).unwrap();
    let f32_bits: Vec<u32> = f32_data.iter().map(|v| v.to_bits()).collect();
    let f32_exp_bits: Vec<u32> = f32_exp.iter().map(|v| v.to_bits()).collect();
    assert_eq!(f32_bits, f32_exp_bits);

    // i32
    let mut i32_data: Vec<i32> = (0..n).map(|_| rng.gen()).collect();
    let mut i32_exp = i32_data.clone();
    i32_exp.sort();
    sorter.sort_i32(&mut i32_data).unwrap();
    assert_eq!(i32_data, i32_exp);

    // u32 again
    let mut u32_data2: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut u32_exp2 = u32_data2.clone();
    u32_exp2.sort();
    sorter.sort_u32(&mut u32_data2).unwrap();
    assert_eq!(u32_data2, u32_exp2);
}

/// Mix sort + argsort calls on the same GpuSorter.
#[test]
fn test_cross_type_interleave_with_argsort() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut rng = common::seeded_rng(201);
    let n = 50_000;

    // sort_u32
    let mut u32_data: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut u32_exp = u32_data.clone();
    u32_exp.sort();
    sorter.sort_u32(&mut u32_data).unwrap();
    assert_eq!(u32_data, u32_exp);

    // argsort_f32
    let f32_data: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 100.0 - 50.0).collect();
    let indices = sorter.argsort_f32(&f32_data).unwrap();
    assert!(common::verify_permutation(&indices, n));
    assert!(common::verify_sorted_by_indices_f32(&f32_data, &indices));

    // sort_i32
    let mut i32_data: Vec<i32> = (0..n).map(|_| rng.gen()).collect();
    let mut i32_exp = i32_data.clone();
    i32_exp.sort();
    sorter.sort_i32(&mut i32_data).unwrap();
    assert_eq!(i32_data, i32_exp);

    // argsort_u32
    let u32_data2: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let indices2 = sorter.argsort_u32(&u32_data2).unwrap();
    assert!(common::verify_permutation(&indices2, n));
    assert!(common::verify_sorted_by_indices(&u32_data2, &indices2));
}

/// Mix 32-bit and 64-bit sorts on the same GpuSorter.
#[test]
fn test_cross_type_interleave_64bit() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut rng = common::seeded_rng(202);
    let n = 50_000;

    // sort_u32
    let mut u32_data: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut u32_exp = u32_data.clone();
    u32_exp.sort();
    sorter.sort_u32(&mut u32_data).unwrap();
    assert_eq!(u32_data, u32_exp);

    // sort_u64
    let mut u64_data: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
    let mut u64_exp = u64_data.clone();
    u64_exp.sort();
    sorter.sort_u64(&mut u64_data).unwrap();
    assert_eq!(u64_data, u64_exp);

    // sort_i64
    let mut i64_data: Vec<i64> = (0..n).map(|_| rng.gen()).collect();
    let mut i64_exp = i64_data.clone();
    i64_exp.sort();
    sorter.sort_i64(&mut i64_data).unwrap();
    assert_eq!(i64_data, i64_exp);

    // sort_f32 (back to 32-bit after 64-bit)
    let mut f32_data: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 100.0 - 50.0).collect();
    let mut f32_exp = f32_data.clone();
    f32_exp.sort_by(f32::total_cmp);
    sorter.sort_f32(&mut f32_data).unwrap();
    let f32_bits: Vec<u32> = f32_data.iter().map(|v| v.to_bits()).collect();
    let f32_exp_bits: Vec<u32> = f32_exp.iter().map(|v| v.to_bits()).collect();
    assert_eq!(f32_bits, f32_exp_bits);

    // sort_f64
    let mut f64_data: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 100.0 - 50.0).collect();
    let mut f64_exp = f64_data.clone();
    f64_exp.sort_by(f64::total_cmp);
    sorter.sort_f64(&mut f64_data).unwrap();
    let f64_bits: Vec<u64> = f64_data.iter().map(|v| v.to_bits()).collect();
    let f64_exp_bits: Vec<u64> = f64_exp.iter().map(|v| v.to_bits()).collect();
    assert_eq!(f64_bits, f64_exp_bits);
}

/// Different types use different PSOs -- sort u32, i32, f32 back to back, all correct.
#[test]
fn test_pso_cache_isolation() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut rng = common::seeded_rng(300);
    let n = 100_000;

    // u32
    let mut u32_data: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut u32_exp = u32_data.clone();
    u32_exp.sort();
    sorter.sort_u32(&mut u32_data).unwrap();
    assert_eq!(u32_data, u32_exp);

    // i32
    let mut i32_data: Vec<i32> = (0..n).map(|_| rng.gen()).collect();
    let mut i32_exp = i32_data.clone();
    i32_exp.sort();
    sorter.sort_i32(&mut i32_data).unwrap();
    assert_eq!(i32_data, i32_exp);

    // f32
    let mut f32_data: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 1000.0 - 500.0).collect();
    let mut f32_exp = f32_data.clone();
    f32_exp.sort_by(f32::total_cmp);
    sorter.sort_f32(&mut f32_data).unwrap();
    let f32_bits: Vec<u32> = f32_data.iter().map(|v| v.to_bits()).collect();
    let f32_exp_bits: Vec<u32> = f32_exp.iter().map(|v| v.to_bits()).collect();
    assert_eq!(f32_bits, f32_exp_bits);

    // u32 again (verify PSO reuse works correctly)
    let mut u32_data2: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut u32_exp2 = u32_data2.clone();
    u32_exp2.sort();
    sorter.sort_u32(&mut u32_data2).unwrap();
    assert_eq!(u32_data2, u32_exp2);
}

/// 16M sort_u32 must complete in < 50ms (same bound as existing perf sanity test).
#[test]
fn test_sort_u32_perf_no_regression() {
    let mut rng = common::seeded_rng(400);
    let n = 16_000_000;
    let mut data: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut sorter = GpuSorter::new().unwrap();

    // warmup
    let mut warmup = data.clone();
    sorter.sort_u32(&mut warmup).unwrap();

    let start = std::time::Instant::now();
    sorter.sort_u32(&mut data).unwrap();
    let elapsed = start.elapsed();

    // Verify sorted
    for i in 1..data.len() {
        assert!(data[i - 1] <= data[i], "Not sorted at index {}", i);
    }

    assert!(
        elapsed.as_millis() < 50,
        "16M sort_u32 took {}ms, expected < 50ms",
        elapsed.as_millis()
    );
}
