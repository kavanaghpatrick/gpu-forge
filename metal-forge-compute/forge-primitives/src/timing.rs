//! Wall-clock timing utilities for benchmark measurement.

use std::time::Instant;

/// Simple wall-clock timer for benchmarking.
pub struct BenchTimer {
    start: Instant,
}

impl BenchTimer {
    /// Start the timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Stop the timer and return elapsed time in milliseconds.
    pub fn stop(&self) -> f64 {
        let elapsed = self.start.elapsed();
        elapsed.as_secs_f64() * 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_measures_positive() {
        let timer = BenchTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed > 0.0, "Timer should measure positive time");
        assert!(elapsed >= 5.0, "Timer should measure at least ~10ms (got {elapsed}ms)");
    }
}
