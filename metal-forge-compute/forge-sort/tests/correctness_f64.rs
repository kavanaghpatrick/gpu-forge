mod common;

use common::seeded_rng;
use forge_sort::GpuSorter;
use rand::Rng;

/// Compare two f64 slices via to_bits() (handles NaN correctly).
fn assert_bits_eq(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "bit mismatch at index {i}: actual={a} (0x{:016x}), expected={e} (0x{:016x})",
            a.to_bits(),
            e.to_bits(),
        );
    }
}

#[test]
fn test_sort_f64_total_cmp_order() {
    let mut data = vec![
        f64::NAN,
        f64::INFINITY,
        1.0,
        0.0,
        -0.0,
        -1.0,
        f64::NEG_INFINITY,
        f64::from_bits(0xFFF8_0000_0000_0000), // -NaN
    ];
    let mut expected = data.clone();
    expected.sort_by(f64::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f64_ieee754_special() {
    let mut data = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        0.0_f64,
        -0.0_f64,
        1.0,
        -1.0,
        f64::EPSILON,
        -f64::EPSILON,
        f64::from_bits(0x7FF8_0000_0000_0001), // +NaN variant
        f64::from_bits(0xFFF8_0000_0000_0001), // -NaN variant
        f64::from_bits(0x0000_0000_0000_0001), // smallest subnormal
        f64::from_bits(0x8000_0000_0000_0001), // smallest negative subnormal
        f64::from_bits(0x000F_FFFF_FFFF_FFFF), // largest subnormal
        f64::from_bits(0x800F_FFFF_FFFF_FFFF), // largest negative subnormal
    ];
    let mut expected = data.clone();
    expected.sort_by(f64::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f64_neg_zero_before_pos_zero() {
    let mut data = vec![0.0_f64, -0.0_f64, 0.0, -0.0, 0.0, -0.0];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    let neg_zero_bits = (-0.0_f64).to_bits();
    let pos_zero_bits = 0.0_f64.to_bits();
    let first_pos = data.iter().position(|x| x.to_bits() == pos_zero_bits).unwrap();
    let last_neg = data.iter().rposition(|x| x.to_bits() == neg_zero_bits).unwrap();
    assert!(last_neg < first_pos, "-0.0 should sort before +0.0");
}

#[test]
fn test_sort_f64_nan_variants() {
    let nan_payloads: Vec<f64> = (1..=10u64)
        .map(|i| f64::from_bits(0x7FF8_0000_0000_0000 | i)) // positive NaN variants
        .chain((1..=10u64).map(|i| f64::from_bits(0xFFF8_0000_0000_0000 | i))) // negative NaN variants
        .collect();
    let mut data = nan_payloads;
    let mut expected = data.clone();
    expected.sort_by(f64::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f64_denormals() {
    let mut data: Vec<f64> = (1..=100u64)
        .map(|i| f64::from_bits(i)) // positive subnormals
        .chain((1..=100u64).map(|i| f64::from_bits(0x8000_0000_0000_0000 | i))) // negative subnormals
        .chain(vec![0.0, -0.0, f64::MIN_POSITIVE, -f64::MIN_POSITIVE])
        .collect();
    let mut expected = data.clone();
    expected.sort_by(f64::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f64_bit_exact() {
    let mut rng = seeded_rng(6600);
    let mut data: Vec<f64> = (0..100_000)
        .map(|_| {
            loop {
                let v = f64::from_bits(rng.gen::<u64>());
                if !v.is_nan() {
                    break v;
                }
            }
        })
        .collect();
    data.push(f64::NAN);
    data.push(f64::from_bits(0xFFF8_0000_0000_0000)); // -NaN
    data.push(f64::from_bits(0x7FF8_0000_0000_0001)); // +NaN variant

    let mut expected = data.clone();
    expected.sort_by(f64::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f64_random_1m() {
    let mut rng = seeded_rng(6601);
    let mut data: Vec<f64> = (0..1_000_000).map(|_| rng.gen::<f64>()).collect();
    let mut expected = data.clone();
    expected.sort_by(f64::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f64_random_16m() {
    let mut rng = seeded_rng(6602);
    let mut data: Vec<f64> = (0..16_000_000).map(|_| rng.gen::<f64>()).collect();
    let mut expected = data.clone();
    expected.sort_by(f64::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f64_buffer_basic() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut buf = sorter.alloc_sort_buffer::<f64>(100);
    let data = vec![3.0_f64, -1.0, f64::NAN, 0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY];
    let mut expected = data.clone();
    expected.sort_by(f64::total_cmp);
    buf.copy_from_slice(&data);
    sorter.sort_f64_buffer(&buf).unwrap();
    assert_bits_eq(buf.as_slice(), &expected);
}

#[test]
fn test_sort_f64_buffer_16m() {
    let mut rng = seeded_rng(6603);
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut buf = sorter.alloc_sort_buffer::<f64>(n);
    let data: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    let mut expected = data.clone();
    expected.sort_by(f64::total_cmp);
    buf.copy_from_slice(&data);
    sorter.sort_f64_buffer(&buf).unwrap();
    assert_bits_eq(buf.as_slice(), &expected);
}

#[test]
fn test_sort_f64_floatflip_roundtrip() {
    let test_values: Vec<f64> = vec![
        0.0, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY,
        f64::NAN, f64::from_bits(0xFFF8_0000_0000_0000), // -NaN
        f64::MIN_POSITIVE, f64::MAX, f64::MIN,
        f64::from_bits(0x0000_0000_0000_0001), // smallest subnormal
        f64::from_bits(0x8000_0000_0000_0001), // smallest negative subnormal
    ];
    for &v in &test_values {
        let bits = v.to_bits();
        // FloatFlip64 forward
        let flipped = if bits & 0x8000_0000_0000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000_0000_0000
        };
        // IFloatFlip64 inverse
        let restored = if flipped & 0x8000_0000_0000_0000 != 0 {
            flipped ^ 0x8000_0000_0000_0000
        } else {
            !flipped
        };
        assert_eq!(restored, bits, "FloatFlip64 roundtrip failed for {v} (0x{bits:016x})");
    }
}

#[test]
fn test_sort_f64_empty() {
    let mut data: Vec<f64> = vec![];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    assert!(data.is_empty());
}

#[test]
fn test_sort_f64_single() {
    let mut data = vec![42.0_f64];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    assert_eq!(data[0].to_bits(), 42.0_f64.to_bits());
}

#[test]
fn test_argsort_f64_basic() {
    let data = vec![3.0_f64, -1.0, 2.0];
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_f64(&data).unwrap();
    // -1.0 < 2.0 < 3.0 -> indices [1, 2, 0]
    assert_eq!(indices, vec![1, 2, 0]);
}

#[test]
fn test_sort_f64_per_byte_isolation() {
    // Values that differ only in specific bytes, to exercise all 8 radix passes
    let mut data: Vec<f64> = Vec::new();
    // Generate values where only one byte differs at a time
    for byte_idx in 0..8u32 {
        for val in [0x01u64, 0x80, 0xFF] {
            let bits = val << (byte_idx * 8);
            data.push(f64::from_bits(bits));
        }
    }
    let mut expected = data.clone();
    expected.sort_by(f64::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f64(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}
