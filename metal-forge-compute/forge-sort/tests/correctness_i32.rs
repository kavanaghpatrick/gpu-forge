use forge_sort::GpuSorter;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

#[test]
fn test_sort_i32_boundaries() {
    let mut data = vec![i32::MAX, 0, i32::MIN, -1, 1];
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i32(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i32_all_negative() {
    let mut rng = seeded_rng(100);
    let mut data: Vec<i32> = (0..100_000).map(|_| -(rng.gen::<u32>() as i32).abs().max(1)).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i32(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i32_mixed_1m() {
    let mut rng = seeded_rng(200);
    let mut data: Vec<i32> = (0..1_000_000).map(|_| rng.gen::<u32>() as i32).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i32(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i32_16m() {
    let mut rng = seeded_rng(300);
    let mut data: Vec<i32> = (0..16_000_000).map(|_| rng.gen::<u32>() as i32).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i32(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i32_presorted() {
    let mut data: Vec<i32> = (-500_000..500_000).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i32(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i32_reverse() {
    let mut data: Vec<i32> = (-500_000..500_000).rev().collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i32(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i32_empty() {
    let mut data: Vec<i32> = vec![];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i32(&mut data).unwrap();
    assert!(data.is_empty());
}

#[test]
fn test_sort_i32_single() {
    let mut data = vec![-42i32];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i32(&mut data).unwrap();
    assert_eq!(data, vec![-42]);
}

#[test]
fn test_sort_i32_buffer_basic() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut buf = sorter.alloc_sort_buffer::<i32>(100);
    let data: Vec<i32> = vec![5, -3, 0, i32::MAX, i32::MIN, 1, -1];
    let mut expected = data.clone();
    expected.sort();
    buf.copy_from_slice(&data);
    sorter.sort_i32_buffer(&buf).unwrap();
    assert_eq!(buf.as_slice(), &expected[..]);
}

#[test]
fn test_sort_i32_buffer_16m() {
    let mut rng = seeded_rng(400);
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut buf = sorter.alloc_sort_buffer::<i32>(n);
    let data: Vec<i32> = (0..n).map(|_| rng.gen::<u32>() as i32).collect();
    let mut expected = data.clone();
    expected.sort();
    buf.copy_from_slice(&data);
    sorter.sort_i32_buffer(&buf).unwrap();
    assert_eq!(buf.as_slice(), &expected[..]);
}

#[test]
fn test_sort_i32_xor_self_inverse() {
    // XOR 0x80000000 applied twice should return original values
    let values: Vec<i32> = vec![i32::MIN, -1, 0, 1, i32::MAX, -12345, 12345];
    for &v in &values {
        let bits = v as u32;
        let transformed = bits ^ 0x8000_0000;
        let restored = transformed ^ 0x8000_0000;
        assert_eq!(restored, bits, "XOR self-inverse failed for {v}");
    }
}

#[test]
fn test_sort_i32_non_tile_aligned() {
    let mut rng = seeded_rng(500);
    let mut data: Vec<i32> = (0..4097).map(|_| rng.gen::<u32>() as i32).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i32(&mut data).unwrap();
    assert_eq!(data, expected);
}
