//! Rayon-based parallel sort CPU baseline.
//!
//! Provides parallel sort for u32 arrays as a CPU comparison point
//! for the GPU radix sort kernel.

use rayon::prelude::*;

/// Parallel sort of u32 values using rayon's par_sort_unstable.
///
/// Returns a new sorted vector (does not modify input).
pub fn par_sort_u32(data: &[u32]) -> Vec<u32> {
    let mut sorted = data.to_vec();
    sorted.par_sort_unstable();
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_sort_basic() {
        let data = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        let sorted = par_sort_u32(&data);
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_par_sort_already_sorted() {
        let data: Vec<u32> = (0..100).collect();
        let sorted = par_sort_u32(&data);
        assert_eq!(sorted, data);
    }

    #[test]
    fn test_par_sort_empty() {
        let data: Vec<u32> = vec![];
        let sorted = par_sort_u32(&data);
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_par_sort_duplicates() {
        let data = vec![3u32, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let sorted = par_sort_u32(&data);
        let mut expected = data.clone();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
    }
}
