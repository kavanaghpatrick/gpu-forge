use forge_sort::GpuSorter;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Compare two f32 slices via to_bits() (handles NaN correctly).
fn assert_bits_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "bit mismatch at index {i}: actual={a} (0x{:08x}), expected={e} (0x{:08x})",
            a.to_bits(),
            e.to_bits(),
        );
    }
}

#[test]
fn test_sort_f32_ieee754_special() {
    let mut data = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        0.0_f32,
        -0.0_f32,
        1.0,
        -1.0,
        f32::EPSILON,
        -f32::EPSILON,
        f32::from_bits(0x7FC0_0001), // +NaN variant
        f32::from_bits(0xFFC0_0001), // -NaN variant
        f32::from_bits(0x0000_0001), // smallest subnormal
        f32::from_bits(0x8000_0001), // smallest negative subnormal
        f32::from_bits(0x007F_FFFF), // largest subnormal
        f32::from_bits(0x807F_FFFF), // largest negative subnormal
    ];
    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f32_total_cmp_order() {
    let mut data = vec![
        f32::NAN,
        f32::INFINITY,
        1.0,
        0.0,
        -0.0,
        -1.0,
        f32::NEG_INFINITY,
        -f32::NAN,
    ];
    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f32_neg_zero_before_pos_zero() {
    let mut data = vec![0.0_f32, -0.0_f32, 0.0, -0.0, 0.0, -0.0];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    // All -0.0 should come before +0.0
    let neg_zero_bits = (-0.0_f32).to_bits();
    let pos_zero_bits = 0.0_f32.to_bits();
    let first_pos = data.iter().position(|x| x.to_bits() == pos_zero_bits).unwrap();
    let last_neg = data.iter().rposition(|x| x.to_bits() == neg_zero_bits).unwrap();
    assert!(last_neg < first_pos, "-0.0 should sort before +0.0");
}

#[test]
fn test_sort_f32_nan_variants() {
    let nan_payloads: Vec<f32> = (1..=10u32)
        .map(|i| f32::from_bits(0x7FC0_0000 | i)) // positive NaN variants
        .chain((1..=10u32).map(|i| f32::from_bits(0xFFC0_0000 | i))) // negative NaN variants
        .collect();
    let mut data = nan_payloads.clone();
    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f32_denormals() {
    let mut data: Vec<f32> = (1..=100u32)
        .map(|i| f32::from_bits(i)) // positive subnormals
        .chain((1..=100u32).map(|i| f32::from_bits(0x8000_0000 | i))) // negative subnormals
        .chain(vec![0.0, -0.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE])
        .collect();
    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f32_bit_exact() {
    let mut rng = seeded_rng(600);
    // Generate random f32 values, filtering NaN for clean oracle comparison
    let mut data: Vec<f32> = (0..100_000)
        .map(|_| {
            loop {
                let v = f32::from_bits(rng.gen::<u32>());
                if !v.is_nan() {
                    break v;
                }
            }
        })
        .collect();
    // Also add some NaN values
    data.push(f32::NAN);
    data.push(-f32::NAN);
    data.push(f32::from_bits(0x7FC0_0001));

    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f32_random_1m() {
    let mut rng = seeded_rng(700);
    let mut data: Vec<f32> = (0..1_000_000).map(|_| rng.gen::<f32>()).collect();
    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f32_random_16m() {
    let mut rng = seeded_rng(800);
    let mut data: Vec<f32> = (0..16_000_000).map(|_| rng.gen::<f32>()).collect();
    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f32_all_nan() {
    let mut data: Vec<f32> = vec![f32::NAN; 1000];
    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f32_all_inf() {
    let mut data = vec![
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert_bits_eq(&data, &expected);
}

#[test]
fn test_sort_f32_buffer_basic() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut buf = sorter.alloc_sort_buffer::<f32>(100);
    let data = vec![3.0_f32, -1.0, f32::NAN, 0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY];
    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    buf.copy_from_slice(&data);
    sorter.sort_f32_buffer(&buf).unwrap();
    assert_bits_eq(buf.as_slice(), &expected);
}

#[test]
fn test_sort_f32_buffer_16m() {
    let mut rng = seeded_rng(900);
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut buf = sorter.alloc_sort_buffer::<f32>(n);
    let data: Vec<f32> = (0..n).map(|_| rng.gen::<f32>()).collect();
    let mut expected = data.clone();
    expected.sort_by(f32::total_cmp);
    buf.copy_from_slice(&data);
    sorter.sort_f32_buffer(&buf).unwrap();
    assert_bits_eq(buf.as_slice(), &expected);
}

#[test]
fn test_sort_f32_floatflip_roundtrip() {
    // FloatFlip forward then inverse should return original bit pattern
    let test_values: Vec<f32> = vec![
        0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY,
        f32::NAN, -f32::NAN, f32::MIN_POSITIVE, f32::MAX, f32::MIN,
        f32::from_bits(0x0000_0001), // smallest subnormal
        f32::from_bits(0x8000_0001), // smallest negative subnormal
    ];
    for &v in &test_values {
        let bits = v.to_bits();
        // FloatFlip forward
        let flipped = if bits & 0x8000_0000 != 0 {
            !bits // negative: flip all bits
        } else {
            bits ^ 0x8000_0000 // positive: flip sign bit
        };
        // IFloatFlip inverse
        let restored = if flipped & 0x8000_0000 != 0 {
            flipped ^ 0x8000_0000 // was positive: flip sign bit back
        } else {
            !flipped // was negative: flip all bits back
        };
        assert_eq!(restored, bits, "FloatFlip roundtrip failed for {v} (0x{bits:08x})");
    }
}

#[test]
fn test_sort_f32_empty() {
    let mut data: Vec<f32> = vec![];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert!(data.is_empty());
}

#[test]
fn test_sort_f32_single() {
    let mut data = vec![42.0_f32];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_f32(&mut data).unwrap();
    assert_eq!(data[0].to_bits(), 42.0_f32.to_bits());
}
