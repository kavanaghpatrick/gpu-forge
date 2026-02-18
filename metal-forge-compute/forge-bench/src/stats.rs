use serde::{Deserialize, Serialize};

/// Descriptive statistics for a set of benchmark samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
    pub cv_percent: f64,
    pub sample_count: usize,
    pub outliers_removed: usize,
}

/// Compute descriptive statistics from timing samples.
///
/// Applies IQR-based outlier detection: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
/// are removed before computing final statistics.
pub fn compute_stats(samples: &[f64]) -> Stats {
    if samples.is_empty() {
        return Stats {
            mean: 0.0,
            median: 0.0,
            min: 0.0,
            max: 0.0,
            stddev: 0.0,
            cv_percent: 0.0,
            sample_count: 0,
            outliers_removed: 0,
        };
    }

    if samples.len() == 1 {
        return Stats {
            mean: samples[0],
            median: samples[0],
            min: samples[0],
            max: samples[0],
            stddev: 0.0,
            cv_percent: 0.0,
            sample_count: 1,
            outliers_removed: 0,
        };
    }

    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // IQR outlier detection
    let q1 = percentile(&sorted, 25.0);
    let q3 = percentile(&sorted, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;

    let filtered: Vec<f64> = sorted
        .iter()
        .copied()
        .filter(|&v| v >= lower && v <= upper)
        .collect();

    let outliers_removed = samples.len() - filtered.len();

    // Use filtered data if we have enough samples, otherwise use all
    let data = if filtered.len() >= 2 {
        &filtered
    } else {
        &sorted
    };

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let median = compute_median(data);
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let variance = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    let stddev = variance.sqrt();
    let cv_percent = if mean > 0.0 {
        (stddev / mean) * 100.0
    } else {
        0.0
    };

    Stats {
        mean,
        median,
        min,
        max,
        stddev,
        cv_percent,
        sample_count: data.len(),
        outliers_removed,
    }
}

/// Compute the median of a sorted slice.
fn compute_median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute a percentile (0-100) from a sorted slice using linear interpolation.
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_samples() {
        let s = compute_stats(&[]);
        assert_eq!(s.sample_count, 0);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn test_single_sample() {
        let s = compute_stats(&[42.0]);
        assert_eq!(s.mean, 42.0);
        assert_eq!(s.median, 42.0);
        assert_eq!(s.stddev, 0.0);
        assert_eq!(s.cv_percent, 0.0);
    }

    #[test]
    fn test_basic_stats() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = compute_stats(&samples);
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.median - 3.0).abs() < 1e-10);
        assert!((s.min - 1.0).abs() < 1e-10);
        assert!((s.max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cv_percent() {
        // Known stddev: values [10, 20] -> mean=15, stddev=sqrt(50)=7.071, cv=47.14%
        let samples = vec![10.0, 20.0];
        let s = compute_stats(&samples);
        assert!((s.mean - 15.0).abs() < 1e-10);
        assert!(s.cv_percent > 40.0 && s.cv_percent < 50.0);
    }

    #[test]
    fn test_outlier_removal() {
        // One extreme outlier in otherwise tight data
        let mut samples = vec![10.0, 10.1, 10.2, 9.9, 9.8, 10.0, 10.1, 10.0, 9.9, 10.0];
        samples.push(1000.0); // extreme outlier
        let s = compute_stats(&samples);
        assert_eq!(s.outliers_removed, 1);
        assert!(s.mean < 11.0); // outlier removed, mean near 10
    }

    #[test]
    fn test_median_even() {
        let s = compute_stats(&[1.0, 2.0, 3.0, 4.0]);
        assert!((s.median - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_median_odd() {
        let s = compute_stats(&[1.0, 2.0, 3.0]);
        assert!((s.median - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cv_zero_for_constant_values() {
        let samples = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let s = compute_stats(&samples);
        assert!((s.cv_percent - 0.0).abs() < 1e-10);
        assert!((s.stddev - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cv_known_values() {
        // Values: [3, 4, 4, 5, 5, 5, 6, 6, 7, 7] -- no outliers
        // No outlier removal expected (tight distribution)
        let samples = vec![3.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0];
        let s = compute_stats(&samples);
        assert_eq!(s.outliers_removed, 0);
        // Mean = 5.2, values are clustered -> CV should be moderate (< 30%)
        assert!(
            s.cv_percent > 0.0 && s.cv_percent < 30.0,
            "CV={} should be between 0 and 30 for this distribution",
            s.cv_percent
        );
        // Verify CV = stddev / mean * 100
        let expected_cv = (s.stddev / s.mean) * 100.0;
        assert!((s.cv_percent - expected_cv).abs() < 1e-10);
    }

    #[test]
    fn test_stddev_known_values() {
        // [1, 2, 3, 4, 5] -> mean = 3, variance = 10/4 = 2.5, stddev = 1.5811
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = compute_stats(&samples);
        let expected_stddev = (2.5_f64).sqrt();
        assert!(
            (s.stddev - expected_stddev).abs() < 1e-6,
            "stddev {} vs expected {}",
            s.stddev,
            expected_stddev
        );
    }

    #[test]
    fn test_multiple_outliers_removed() {
        // Tight cluster around 100, with two extreme outliers
        let mut samples = vec![
            100.0, 100.1, 99.9, 100.2, 99.8, 100.0, 100.1, 99.9, 100.0, 100.0,
        ];
        samples.push(500.0); // outlier high
        samples.push(1.0); // outlier low
        let s = compute_stats(&samples);
        assert!(
            s.outliers_removed >= 2,
            "Expected at least 2 outliers removed, got {}",
            s.outliers_removed
        );
        assert!(
            s.mean > 99.0 && s.mean < 101.0,
            "Mean {} should be near 100 after outlier removal",
            s.mean
        );
    }

    #[test]
    fn test_no_outliers_in_uniform_data() {
        // Evenly spaced data should have few or no outliers
        let samples: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let s = compute_stats(&samples);
        assert_eq!(s.outliers_removed, 0);
        assert_eq!(s.sample_count, 10);
    }

    #[test]
    fn test_percentile_function() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&sorted, 50.0) - 3.0).abs() < 1e-10);
        assert!((percentile(&sorted, 100.0) - 5.0).abs() < 1e-10);
        assert!((percentile(&sorted, 25.0) - 2.0).abs() < 1e-10);
        assert!((percentile(&sorted, 75.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_correct() {
        let samples = vec![5.0, 3.0, 8.0, 1.0, 7.0];
        let s = compute_stats(&samples);
        assert!((s.min - 1.0).abs() < 1e-10);
        assert!((s.max - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_samples() {
        let samples = vec![3.0, 7.0];
        let s = compute_stats(&samples);
        assert!((s.mean - 5.0).abs() < 1e-10);
        assert!((s.median - 5.0).abs() < 1e-10);
        // sample stddev = sqrt((4+4)/1) = sqrt(8) = 2.828
        let expected_stddev = (8.0_f64).sqrt();
        assert!((s.stddev - expected_stddev).abs() < 1e-6);
    }
}
