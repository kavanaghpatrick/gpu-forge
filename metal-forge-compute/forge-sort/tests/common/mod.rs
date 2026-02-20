#![allow(dead_code)]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Verify that `indices` is a valid permutation of 0..n.
pub fn verify_permutation(indices: &[u32], n: usize) -> bool {
    if indices.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &idx in indices {
        let i = idx as usize;
        if i >= n || seen[i] {
            return false;
        }
        seen[i] = true;
    }
    true
}

/// Verify that data[indices[i]] <= data[indices[i+1]] for all consecutive pairs.
pub fn verify_sorted_by_indices<T: Ord>(data: &[T], indices: &[u32]) -> bool {
    for w in indices.windows(2) {
        if data[w[0] as usize] > data[w[1] as usize] {
            return false;
        }
    }
    true
}

/// Verify that the same multiset of (key, value) pairs exists before and after sorting.
pub fn verify_pairs_preserved(
    orig_keys: &[u32],
    orig_vals: &[u32],
    sorted_keys: &[u32],
    sorted_vals: &[u32],
) -> bool {
    if orig_keys.len() != sorted_keys.len() || orig_vals.len() != sorted_vals.len() {
        return false;
    }
    let mut orig_pairs: Vec<(u32, u32)> = orig_keys.iter().copied().zip(orig_vals.iter().copied()).collect();
    let mut sorted_pairs: Vec<(u32, u32)> = sorted_keys.iter().copied().zip(sorted_vals.iter().copied()).collect();
    orig_pairs.sort();
    sorted_pairs.sort();
    orig_pairs == sorted_pairs
}

/// Verify sorted_by_indices for f32 using total_cmp ordering.
pub fn verify_sorted_by_indices_f32(data: &[f32], indices: &[u32]) -> bool {
    for w in indices.windows(2) {
        if data[w[0] as usize].total_cmp(&data[w[1] as usize]) == std::cmp::Ordering::Greater {
            return false;
        }
    }
    true
}

/// Verify pairs preserved for i32 keys.
pub fn verify_pairs_preserved_i32(
    orig_keys: &[i32],
    orig_vals: &[u32],
    sorted_keys: &[i32],
    sorted_vals: &[u32],
) -> bool {
    if orig_keys.len() != sorted_keys.len() {
        return false;
    }
    let mut orig_pairs: Vec<(i32, u32)> = orig_keys.iter().copied().zip(orig_vals.iter().copied()).collect();
    let mut sorted_pairs: Vec<(i32, u32)> = sorted_keys.iter().copied().zip(sorted_vals.iter().copied()).collect();
    orig_pairs.sort();
    sorted_pairs.sort();
    orig_pairs == sorted_pairs
}

/// Verify pairs preserved for f32 keys (compare by bits for NaN correctness).
pub fn verify_pairs_preserved_f32(
    orig_keys: &[f32],
    orig_vals: &[u32],
    sorted_keys: &[f32],
    sorted_vals: &[u32],
) -> bool {
    if orig_keys.len() != sorted_keys.len() {
        return false;
    }
    let mut orig_pairs: Vec<(u32, u32)> = orig_keys
        .iter()
        .map(|k| k.to_bits())
        .zip(orig_vals.iter().copied())
        .collect();
    let mut sorted_pairs: Vec<(u32, u32)> = sorted_keys
        .iter()
        .map(|k| k.to_bits())
        .zip(sorted_vals.iter().copied())
        .collect();
    orig_pairs.sort();
    sorted_pairs.sort();
    orig_pairs == sorted_pairs
}
