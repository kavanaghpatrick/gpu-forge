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

    #[test]
    fn test_timer_stop_returns_milliseconds() {
        let timer = BenchTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let elapsed = timer.stop();
        // Should be between 40ms and 200ms (generous bounds for CI)
        assert!(elapsed >= 30.0, "Expected >= 30ms, got {elapsed}ms");
        assert!(elapsed < 500.0, "Expected < 500ms, got {elapsed}ms");
    }

    #[test]
    fn test_timer_multiple_stops() {
        // stop() is non-destructive -- can be called multiple times
        let timer = BenchTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let first = timer.stop();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let second = timer.stop();
        assert!(second > first, "Second stop should be later than first");
    }
}
