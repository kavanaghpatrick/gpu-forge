mod common;

use common::{
    seeded_rng, verify_pairs_preserved, verify_pairs_preserved_f32, verify_pairs_preserved_i32,
};
use forge_sort::{GpuSorter, SortError};
use rand::Rng;

#[test]
fn test_sort_pairs_u32_basic() {
    let mut keys = vec![30u32, 10, 20];
    let mut values = vec![300u32, 100, 200];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_pairs_u32(&mut keys, &mut values).unwrap();
    assert_eq!(keys, vec![10, 20, 30]);
    assert_eq!(values, vec![100, 200, 300]);
}

#[test]
fn test_sort_pairs_u32_multiset_preserved() {
    let mut rng = seeded_rng(2000);
    let n = 10_000;
    let orig_keys: Vec<u32> = (0..n).map(|_| rng.gen::<u32>() % 1000).collect();
    let orig_vals: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut keys = orig_keys.clone();
    let mut values = orig_vals.clone();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_pairs_u32(&mut keys, &mut values).unwrap();
    assert!(
        verify_pairs_preserved(&orig_keys, &orig_vals, &keys, &values),
        "multiset of (key, value) pairs must be preserved"
    );
}

#[test]
fn test_sort_pairs_u32_stable() {
    // Equal keys: values should preserve original order
    let mut keys = vec![5u32, 3, 5, 3, 5];
    let mut values = vec![50u32, 30, 51, 31, 52];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_pairs_u32(&mut keys, &mut values).unwrap();
    assert_eq!(keys, vec![3, 3, 5, 5, 5]);
    // For equal keys, values should maintain relative order
    assert_eq!(values, vec![30, 31, 50, 51, 52]);
}

#[test]
fn test_sort_pairs_i32_basic() {
    let mut keys = vec![5i32, -3, 0, -1, 2];
    let mut values = vec![50u32, 30, 0, 10, 20];
    let orig_keys = keys.clone();
    let orig_vals = values.clone();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_pairs_i32(&mut keys, &mut values).unwrap();
    assert_eq!(keys, vec![-3, -1, 0, 2, 5]);
    assert!(verify_pairs_preserved_i32(&orig_keys, &orig_vals, &keys, &values));
}

#[test]
fn test_sort_pairs_f32_basic() {
    let mut keys = vec![3.0f32, -1.0, 0.0, 2.0];
    let mut values = vec![30u32, 10, 0, 20];
    let orig_keys = keys.clone();
    let orig_vals = values.clone();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_pairs_f32(&mut keys, &mut values).unwrap();
    assert_eq!(keys, vec![-1.0, 0.0, 2.0, 3.0]);
    assert!(verify_pairs_preserved_f32(&orig_keys, &orig_vals, &keys, &values));
}

#[test]
fn test_sort_pairs_f32_nan_with_values() {
    let mut keys = vec![f32::NAN, 1.0, f32::NEG_INFINITY, -f32::NAN, 0.0];
    let mut values = vec![99u32, 11, 22, 88, 0];
    let orig_keys = keys.clone();
    let orig_vals = values.clone();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_pairs_f32(&mut keys, &mut values).unwrap();
    // Verify keys are in total_cmp order
    for w in keys.windows(2) {
        assert!(
            w[0].total_cmp(&w[1]) != std::cmp::Ordering::Greater,
            "keys not sorted: {} > {}",
            w[0],
            w[1]
        );
    }
    // Verify pairs preserved
    assert!(verify_pairs_preserved_f32(&orig_keys, &orig_vals, &keys, &values));
}

#[test]
fn test_sort_pairs_length_mismatch() {
    let mut keys = vec![1u32, 2, 3];
    let mut values = vec![10u32, 20];
    let mut sorter = GpuSorter::new().unwrap();
    let err = sorter.sort_pairs_u32(&mut keys, &mut values).unwrap_err();
    match err {
        SortError::LengthMismatch { keys: k, values: v } => {
            assert_eq!(k, 3);
            assert_eq!(v, 2);
        }
        other => panic!("expected LengthMismatch, got: {other}"),
    }
}

#[test]
fn test_sort_pairs_u32_1m() {
    let mut rng = seeded_rng(2001);
    let n = 1_000_000;
    let orig_keys: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let orig_vals: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut keys = orig_keys.clone();
    let mut values = orig_vals.clone();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_pairs_u32(&mut keys, &mut values).unwrap();
    // Keys sorted
    for w in keys.windows(2) {
        assert!(w[0] <= w[1]);
    }
    assert!(verify_pairs_preserved(&orig_keys, &orig_vals, &keys, &values));
}

#[test]
fn test_sort_pairs_f32_16m() {
    let mut rng = seeded_rng(2002);
    let n = 16_000_000;
    let orig_keys: Vec<f32> = (0..n).map(|_| rng.gen::<f32>()).collect();
    let orig_vals: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    let mut keys = orig_keys.clone();
    let mut values = orig_vals.clone();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_pairs_f32(&mut keys, &mut values).unwrap();
    // Keys sorted by total_cmp
    for w in keys.windows(2) {
        assert!(w[0].total_cmp(&w[1]) != std::cmp::Ordering::Greater);
    }
    assert!(verify_pairs_preserved_f32(&orig_keys, &orig_vals, &keys, &values));
}

#[test]
fn test_sort_pairs_empty() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut k: Vec<u32> = vec![];
    let mut v: Vec<u32> = vec![];
    sorter.sort_pairs_u32(&mut k, &mut v).unwrap();
    assert!(k.is_empty());
    assert!(v.is_empty());
}

#[test]
fn test_sort_pairs_single() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut k = vec![42u32];
    let mut v = vec![99u32];
    sorter.sort_pairs_u32(&mut k, &mut v).unwrap();
    assert_eq!(k, vec![42]);
    assert_eq!(v, vec![99]);
}

#[test]
fn test_sort_pairs_u32_all_same() {
    let mut keys = vec![7u32; 100];
    let mut values: Vec<u32> = (0..100).collect();
    let orig_values = values.clone();
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_pairs_u32(&mut keys, &mut values).unwrap();
    assert!(keys.iter().all(|&k| k == 7));
    // Stable sort: equal keys preserve value order
    assert_eq!(values, orig_values);
}
