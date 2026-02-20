mod common;

use common::{seeded_rng, verify_permutation, verify_sorted_by_indices, verify_sorted_by_indices_f32};
use forge_sort::GpuSorter;
use rand::Rng;

#[test]
fn test_argsort_u32_basic() {
    let data = vec![30u32, 10, 20];
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_u32(&data).unwrap();
    assert_eq!(indices, vec![1, 2, 0]);
}

#[test]
fn test_argsort_u32_permutation_valid() {
    let mut rng = seeded_rng(1000);
    let data: Vec<u32> = (0..10_000).map(|_| rng.gen()).collect();
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_u32(&data).unwrap();
    assert!(verify_permutation(&indices, data.len()));
}

#[test]
fn test_argsort_u32_sorted_order() {
    let mut rng = seeded_rng(1001);
    let data: Vec<u32> = (0..10_000).map(|_| rng.gen()).collect();
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_u32(&data).unwrap();
    assert!(verify_sorted_by_indices(&data, &indices));
}

#[test]
fn test_argsort_u32_input_unmodified() {
    let data = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6, 0];
    let original = data.clone();
    let mut sorter = GpuSorter::new().unwrap();
    let _indices = sorter.argsort_u32(&data).unwrap();
    assert_eq!(data, original, "argsort must not modify input data");
}

#[test]
fn test_argsort_u32_stable() {
    // Equal keys: indices for equal keys should preserve relative order
    let data = vec![5u32, 3, 5, 3, 5];
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_u32(&data).unwrap();
    // Verify sorted order
    assert!(verify_sorted_by_indices(&data, &indices));
    // For keys==3: original indices 1,3 should appear in that order
    let threes: Vec<u32> = indices.iter().copied().filter(|&i| data[i as usize] == 3).collect();
    assert_eq!(threes, vec![1, 3], "equal keys should preserve relative index order");
    // For keys==5: original indices 0,2,4 should appear in that order
    let fives: Vec<u32> = indices.iter().copied().filter(|&i| data[i as usize] == 5).collect();
    assert_eq!(fives, vec![0, 2, 4], "equal keys should preserve relative index order");
}

#[test]
fn test_argsort_u32_1m() {
    let mut rng = seeded_rng(1002);
    let data: Vec<u32> = (0..1_000_000).map(|_| rng.gen()).collect();
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_u32(&data).unwrap();
    assert!(verify_permutation(&indices, data.len()));
    assert!(verify_sorted_by_indices(&data, &indices));
}

#[test]
fn test_argsort_u32_16m() {
    let mut rng = seeded_rng(1003);
    let data: Vec<u32> = (0..16_000_000).map(|_| rng.gen()).collect();
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_u32(&data).unwrap();
    assert!(verify_permutation(&indices, data.len()));
    assert!(verify_sorted_by_indices(&data, &indices));
}

#[test]
fn test_argsort_i32_basic() {
    let data = vec![5i32, -3, 0, -1, 2];
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_i32(&data).unwrap();
    // sorted: [-3, -1, 0, 2, 5] -> indices [1, 3, 2, 4, 0]
    assert_eq!(indices, vec![1, 3, 2, 4, 0]);
}

#[test]
fn test_argsort_i32_boundaries() {
    let data = vec![i32::MAX, 0, i32::MIN, -1, 1];
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_i32(&data).unwrap();
    // sorted: [MIN, -1, 0, 1, MAX] -> indices [2, 3, 1, 4, 0]
    assert_eq!(indices, vec![2, 3, 1, 4, 0]);
    // Verify the order is correct
    for w in indices.windows(2) {
        assert!(data[w[0] as usize] <= data[w[1] as usize]);
    }
}

#[test]
fn test_argsort_f32_basic() {
    let data = vec![3.0f32, -1.0, 0.0, 2.0];
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_f32(&data).unwrap();
    // sorted: [-1.0, 0.0, 2.0, 3.0] -> indices [1, 2, 3, 0]
    assert_eq!(indices, vec![1, 2, 3, 0]);
}

#[test]
fn test_argsort_f32_nan() {
    let data = vec![f32::NAN, 1.0, f32::NEG_INFINITY, -f32::NAN, 0.0];
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_f32(&data).unwrap();
    assert!(verify_permutation(&indices, data.len()));
    assert!(verify_sorted_by_indices_f32(&data, &indices));
}

#[test]
fn test_argsort_empty() {
    let mut sorter = GpuSorter::new().unwrap();
    assert_eq!(sorter.argsort_u32(&[]).unwrap(), Vec::<u32>::new());
    assert_eq!(sorter.argsort_i32(&[]).unwrap(), Vec::<u32>::new());
    assert_eq!(sorter.argsort_f32(&[]).unwrap(), Vec::<u32>::new());
}

#[test]
fn test_argsort_single() {
    let mut sorter = GpuSorter::new().unwrap();
    assert_eq!(sorter.argsort_u32(&[42]).unwrap(), vec![0]);
    assert_eq!(sorter.argsort_i32(&[-1]).unwrap(), vec![0]);
    assert_eq!(sorter.argsort_f32(&[3.14]).unwrap(), vec![0]);
}

#[test]
fn test_argsort_u32_all_same() {
    let data = vec![7u32; 100];
    let mut sorter = GpuSorter::new().unwrap();
    let indices = sorter.argsort_u32(&data).unwrap();
    assert!(verify_permutation(&indices, data.len()));
    // All keys equal, stable sort should return 0..n
    let expected: Vec<u32> = (0..100).collect();
    assert_eq!(indices, expected);
}
