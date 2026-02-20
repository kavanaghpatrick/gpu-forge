//! Sequential CPU baselines for scan and histogram operations.
//!
//! These are inherently sequential algorithms that serve as CPU comparison
//! points for the GPU parallel implementations.

/// Sequential exclusive prefix scan of u32 values.
///
/// Returns a vector where output[i] = sum(data[0..i]).
/// output[0] = 0, output[1] = data[0], output[2] = data[0]+data[1], etc.
pub fn sequential_exclusive_scan(data: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(data.len());
    let mut acc: u64 = 0;
    for &val in data {
        result.push(acc as u32);
        acc += val as u64;
    }
    result
}

/// Sequential histogram of u32 values into `num_bins` bins.
///
/// Each element is assigned to bin `value % num_bins`.
pub fn sequential_histogram(data: &[u32], num_bins: usize) -> Vec<u32> {
    let mut bins = vec![0u32; num_bins];
    for &val in data {
        let bin = (val as usize) % num_bins;
        bins[bin] = bins[bin].wrapping_add(1);
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exclusive_scan_basic() {
        let data = vec![1u32, 2, 3, 4, 5];
        let result = sequential_exclusive_scan(&data);
        assert_eq!(result, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn test_exclusive_scan_empty() {
        let data: Vec<u32> = vec![];
        let result = sequential_exclusive_scan(&data);
        assert!(result.is_empty());
    }

    #[test]
    fn test_exclusive_scan_single() {
        let data = vec![42u32];
        let result = sequential_exclusive_scan(&data);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_histogram_basic() {
        let data = vec![0u32, 1, 2, 3, 0, 1, 2, 0];
        let result = sequential_histogram(&data, 4);
        assert_eq!(result, vec![3, 2, 2, 1]);
    }

    #[test]
    fn test_histogram_empty() {
        let data: Vec<u32> = vec![];
        let result = sequential_histogram(&data, 256);
        assert_eq!(result.len(), 256);
        assert!(result.iter().all(|&v| v == 0));
    }
}
