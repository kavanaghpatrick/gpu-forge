//! Rayon-based parallel CPU reduction baselines.
//!
//! Provides parallel sum, min, and max for u32 and f32 arrays
//! as CPU comparison points for the GPU reduce kernels.

use rayon::prelude::*;

/// Parallel sum of u32 values, returning u64 to avoid overflow.
pub fn par_sum_u32(data: &[u32]) -> u64 {
    data.par_iter().map(|&x| x as u64).sum::<u64>()
}

/// Parallel sum of f32 values, returning f64 for precision.
pub fn par_sum_f32(data: &[f32]) -> f64 {
    data.par_iter().map(|&x| x as f64).sum::<f64>()
}

/// Parallel min of u32 values.
pub fn par_min_u32(data: &[u32]) -> u32 {
    data.par_iter().copied().min().unwrap_or(u32::MAX)
}

/// Parallel max of u32 values.
pub fn par_max_u32(data: &[u32]) -> u32 {
    data.par_iter().copied().max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_sum_u32() {
        let data: Vec<u32> = (1..=100).collect();
        assert_eq!(par_sum_u32(&data), 5050);
    }

    #[test]
    fn test_par_sum_f32() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let sum = par_sum_f32(&data);
        assert!((sum - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_par_min_u32() {
        let data = vec![5u32, 3, 8, 1, 9];
        assert_eq!(par_min_u32(&data), 1);
    }

    #[test]
    fn test_par_max_u32() {
        let data = vec![5u32, 3, 8, 1, 9];
        assert_eq!(par_max_u32(&data), 9);
    }

    #[test]
    fn test_empty_slice() {
        let data: Vec<u32> = vec![];
        assert_eq!(par_sum_u32(&data), 0);
        assert_eq!(par_min_u32(&data), u32::MAX);
        assert_eq!(par_max_u32(&data), 0);
    }
}
