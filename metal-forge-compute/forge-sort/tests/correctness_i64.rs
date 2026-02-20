mod common;

use common::{seeded_rng, verify_permutation, verify_sorted_by_indices};
use forge_sort::GpuSorter;
use rand::Rng;

#[test]
fn test_sort_i64_boundaries() {
    let mut data = vec![i64::MIN, -1, 0, 1, i64::MAX];
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i64_sign_boundary() {
    let mut data: Vec<i64> = (-1000..=1000).collect();
    data.push(i64::MIN);
    data.push(i64::MAX);
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i64_all_negative() {
    let mut rng = seeded_rng(6500);
    let mut data: Vec<i64> = (0..100_000)
        .map(|_| -(rng.gen::<u64>() as i64).abs().max(1))
        .collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i64_mixed_1m() {
    let mut rng = seeded_rng(6501);
    let mut data: Vec<i64> = (0..1_000_000).map(|_| rng.gen::<u64>() as i64).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i64_xor_self_inverse() {
    // XOR 0x8000000000000000 applied twice should return original values
    let values: Vec<i64> = vec![i64::MIN, -1, 0, 1, i64::MAX, -123456789, 123456789];
    for &v in &values {
        let bits = v as u64;
        let transformed = bits ^ 0x8000_0000_0000_0000;
        let restored = transformed ^ 0x8000_0000_0000_0000;
        assert_eq!(restored, bits, "XOR self-inverse failed for {v}");
    }
}

#[test]
fn test_sort_i64_buffer_basic() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut buf = sorter.alloc_sort_buffer::<i64>(100);
    let data: Vec<i64> = vec![5, -3, 0, i64::MAX, i64::MIN, 1, -1];
    let mut expected = data.clone();
    expected.sort();
    buf.copy_from_slice(&data);
    sorter.sort_i64_buffer(&buf).unwrap();
    assert_eq!(buf.as_slice(), &expected[..]);
}

#[test]
fn test_sort_i64_random_16m() {
    let mut rng = seeded_rng(6502);
    let mut data: Vec<i64> = (0..16_000_000).map(|_| rng.gen::<u64>() as i64).collect();
    let mut expected = data.clone();
    expected.sort();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_sort_i64_empty() {
    let mut data: Vec<i64> = vec![];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i64(&mut data).unwrap();
    assert!(data.is_empty());
}

#[test]
fn test_sort_i64_single() {
    let mut data = vec![-42i64];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_i64(&mut data).unwrap();
    assert_eq!(data, vec![-42i64]);
}

#[test]
fn test_argsort_i64_basic() {
    let data = vec![30i64, -10, 20];
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_i64(&data).unwrap();
    assert!(verify_permutation(&indices, data.len()));
    assert!(verify_sorted_by_indices(
        &data.iter().map(|&x| x).collect::<Vec<_>>(),
        &indices,
    ));
    // -10 < 20 < 30, so indices should be [1, 2, 0]
    assert_eq!(indices, vec![1, 2, 0]);
}

#[test]
fn test_sort_pairs_i64_basic() {
    let mut keys = vec![30i64, -10, 20, 0];
    let mut values = vec![100u32, 200, 300, 400];
    let orig_keys = keys.clone();
    let orig_values = values.clone();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_pairs_i64(&mut keys, &mut values).unwrap();
    // Keys should be sorted
    assert_eq!(keys, vec![-10, 0, 20, 30]);
    // Values should follow their original keys
    assert_eq!(values, vec![200, 400, 300, 100]);
    // Verify multiset preservation
    let mut orig_pairs: Vec<(i64, u32)> = orig_keys.into_iter().zip(orig_values).collect();
    let mut sorted_pairs: Vec<(i64, u32)> = keys.into_iter().zip(values).collect();
    orig_pairs.sort();
    sorted_pairs.sort();
    assert_eq!(orig_pairs, sorted_pairs);
}
