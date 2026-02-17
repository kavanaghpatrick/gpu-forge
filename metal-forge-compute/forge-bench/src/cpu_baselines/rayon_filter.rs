//! Rayon-based parallel filter (stream compaction) CPU baseline.
//!
//! Provides parallel filter for u32 arrays as CPU comparison
//! for the GPU compact kernels.

use rayon::prelude::*;

/// Parallel filter: collect all elements greater than threshold.
///
/// Returns a sorted vector of matching elements for deterministic comparison.
pub fn par_filter_gt(data: &[u32], threshold: u32) -> Vec<u32> {
    data.par_iter()
        .copied()
        .filter(|&x| x > threshold)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_filter_basic() {
        let data = vec![1u32, 5, 3, 8, 2, 7, 4, 6];
        let result = par_filter_gt(&data, 4);
        let mut sorted = result.clone();
        sorted.sort();
        assert_eq!(sorted, vec![5, 6, 7, 8]);
    }

    #[test]
    fn test_par_filter_none() {
        let data = vec![1u32, 2, 3];
        let result = par_filter_gt(&data, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_par_filter_all() {
        let data = vec![5u32, 6, 7];
        let result = par_filter_gt(&data, 0);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_par_filter_empty() {
        let data: Vec<u32> = vec![];
        let result = par_filter_gt(&data, 0);
        assert!(result.is_empty());
    }
}
