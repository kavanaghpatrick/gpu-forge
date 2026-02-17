use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Deterministic data generator seeded for reproducible benchmarks.
pub struct DataGenerator {
    rng: StdRng,
}

impl DataGenerator {
    /// Create a new generator with a fixed seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate `count` uniformly distributed u32 values.
    pub fn uniform_u32(&mut self, count: usize) -> Vec<u32> {
        (0..count).map(|_| self.rng.gen::<u32>()).collect()
    }

    /// Generate `count` uniformly distributed f32 values in [0.0, 1.0).
    pub fn uniform_f32(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.rng.gen::<f32>()).collect()
    }

    /// Generate random CSV data as a byte buffer of approximately `target_bytes` size.
    ///
    /// Produces rows with 10 comma-separated random integer fields (~50 bytes/row).
    /// Returns the raw byte buffer. The data always ends with a trailing newline.
    pub fn csv_records(&mut self, target_bytes: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(target_bytes + 128);
        let fields_per_row = 10;

        while buf.len() < target_bytes {
            for f in 0..fields_per_row {
                let val: u32 = self.rng.gen_range(0..100_000);
                // Write the integer as ASCII digits
                let s = val.to_string();
                buf.extend_from_slice(s.as_bytes());
                if f < fields_per_row - 1 {
                    buf.push(b',');
                }
            }
            buf.push(b'\n');
        }

        // Truncate to exactly target_bytes (ensure last byte is still valid)
        buf.truncate(target_bytes);
        // Ensure trailing newline for clean row boundary
        if buf.last() != Some(&b'\n') {
            if let Some(last) = buf.last_mut() {
                *last = b'\n';
            }
        }

        buf
    }

    /// Generate time series data: prices in [50.0, 200.0] and volumes in [1000.0, 100000.0].
    ///
    /// Returns (prices, volumes) each of length `count`.
    pub fn time_series(&mut self, count: usize) -> (Vec<f32>, Vec<f32>) {
        let prices: Vec<f32> = (0..count)
            .map(|_| self.rng.gen_range(50.0f32..200.0f32))
            .collect();
        let volumes: Vec<f32> = (0..count)
            .map(|_| self.rng.gen_range(1000.0f32..100000.0f32))
            .collect();
        (prices, volumes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_u32() {
        let a = DataGenerator::new(42).uniform_u32(100);
        let b = DataGenerator::new(42).uniform_u32(100);
        assert_eq!(a, b);
    }

    #[test]
    fn test_deterministic_f32() {
        let a = DataGenerator::new(42).uniform_f32(100);
        let b = DataGenerator::new(42).uniform_f32(100);
        assert_eq!(a, b);
    }

    #[test]
    fn test_correct_count() {
        let mut gen = DataGenerator::new(0);
        assert_eq!(gen.uniform_u32(1000).len(), 1000);
        assert_eq!(gen.uniform_f32(500).len(), 500);
    }

    #[test]
    fn test_f32_range() {
        let data = DataGenerator::new(123).uniform_f32(10_000);
        for v in &data {
            assert!(*v >= 0.0 && *v < 1.0, "f32 value {} out of range", v);
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let a = DataGenerator::new(1).uniform_u32(100);
        let b = DataGenerator::new(2).uniform_u32(100);
        assert_ne!(a, b);
    }

    #[test]
    fn test_time_series_reproducibility() {
        let (prices_a, volumes_a) = DataGenerator::new(42).time_series(1000);
        let (prices_b, volumes_b) = DataGenerator::new(42).time_series(1000);
        assert_eq!(prices_a, prices_b);
        assert_eq!(volumes_a, volumes_b);
    }

    #[test]
    fn test_time_series_correct_count() {
        let (prices, volumes) = DataGenerator::new(0).time_series(500);
        assert_eq!(prices.len(), 500);
        assert_eq!(volumes.len(), 500);
    }

    #[test]
    fn test_time_series_price_range() {
        let (prices, _) = DataGenerator::new(99).time_series(10_000);
        for p in &prices {
            assert!(*p >= 50.0 && *p < 200.0, "Price {} out of [50, 200) range", p);
        }
    }

    #[test]
    fn test_time_series_volume_range() {
        let (_, volumes) = DataGenerator::new(99).time_series(10_000);
        for v in &volumes {
            assert!(*v >= 1000.0 && *v < 100000.0, "Volume {} out of [1000, 100000) range", v);
        }
    }

    #[test]
    fn test_csv_records_correct_size() {
        let mut gen = DataGenerator::new(42);
        let data = gen.csv_records(1000);
        assert_eq!(data.len(), 1000);
    }

    #[test]
    fn test_csv_records_ends_with_newline() {
        let mut gen = DataGenerator::new(42);
        let data = gen.csv_records(500);
        assert_eq!(*data.last().unwrap(), b'\n');
    }

    #[test]
    fn test_csv_records_reproducibility() {
        let a = DataGenerator::new(42).csv_records(1000);
        let b = DataGenerator::new(42).csv_records(1000);
        assert_eq!(a, b);
    }

    #[test]
    fn test_uniform_u32_zero_count() {
        let mut gen = DataGenerator::new(0);
        assert!(gen.uniform_u32(0).is_empty());
    }

    #[test]
    fn test_uniform_f32_zero_count() {
        let mut gen = DataGenerator::new(0);
        assert!(gen.uniform_f32(0).is_empty());
    }

    #[test]
    fn test_sequential_calls_differ() {
        // Two sequential calls from same generator should produce different data
        let mut gen = DataGenerator::new(42);
        let a = gen.uniform_u32(100);
        let b = gen.uniform_u32(100);
        assert_ne!(a, b, "Sequential calls should produce different data");
    }
}
