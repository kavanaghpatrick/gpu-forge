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
}
