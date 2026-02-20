use forge_sort::GpuSorter;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

#[test]
fn test_sort_u64_boundaries() {
    let mut data = vec![u64::MAX, 0, 1, u64::MAX - 1];
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_u64_byte_boundary_values() {
    // Values that isolate each of the 8 bytes
    let mut data: Vec<u64> = (0..8u32)
        .map(|b| 0xFFu64 << (b * 8))
        .chain((0..8u32).map(|b| 0x01u64 << (b * 8)))
        .chain(std::iter::once(0u64))
        .chain(std::iter::once(u64::MAX))
        .collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_u64_random_1m() {
    let mut rng = seeded_rng(6400);
    let mut data: Vec<u64> = (0..1_000_000).map(|_| rng.gen::<u64>()).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_u64_random_16m() {
    let mut rng = seeded_rng(6401);
    let mut data: Vec<u64> = (0..16_000_000).map(|_| rng.gen::<u64>()).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_u64_buffer_basic() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut buf = sorter.alloc_sort_buffer::<u64>(100);
    let data: Vec<u64> = vec![100, 3, u64::MAX, 0, 42, 1];
    let mut expected = data.clone();
    expected.sort();
    buf.copy_from_slice(&data);
    sorter.sort_u64_buffer(&buf).unwrap();
    assert_eq!(buf.as_slice(), &expected[..]);
}

#[test]
fn test_sort_u64_buffer_16m() {
    let mut rng = seeded_rng(6402);
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut buf = sorter.alloc_sort_buffer::<u64>(n);
    let data: Vec<u64> = (0..n).map(|_| rng.gen::<u64>()).collect();
    let mut expected = data.clone();
    expected.sort();
    buf.copy_from_slice(&data);
    sorter.sort_u64_buffer(&buf).unwrap();
    assert_eq!(buf.as_slice(), &expected[..]);
}

#[test]
fn test_sort_u64_presorted() {
    let mut data: Vec<u64> = (0..100_000u64).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_u64_reverse() {
    let mut data: Vec<u64> = (0..100_000u64).rev().collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_u64_empty() {
    let mut data: Vec<u64> = vec![];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u64(&mut data).unwrap();
    assert!(data.is_empty());
}

#[test]
fn test_sort_u64_single() {
    let mut data = vec![42u64];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u64(&mut data).unwrap();
    assert_eq!(data, vec![42u64]);
}

#[test]
fn test_sort_u64_non_tile_aligned() {
    let mut rng = seeded_rng(6403);
    // 2049 is not aligned to any power-of-two tile size
    let mut data: Vec<u64> = (0..2049).map(|_| rng.gen::<u64>()).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_u64_powers_of_two() {
    let mut data: Vec<u64> = (0..64).map(|i| 1u64 << i).collect();
    data.push(0);
    data.push(u64::MAX);
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u64(&mut data).unwrap();
    assert_eq!(data, expected);
}
