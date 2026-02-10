//! GPU metrics collection for the dashboard.
//!
//! Collects timing, memory, and throughput metrics from query execution.
//! Uses wall-clock timing (std::time::Instant) for POC; Metal Counters API
//! can replace this for hardware-accurate GPU utilization in the future.

use std::time::Instant;

/// Maximum number of history samples retained for sparkline display.
pub const MAX_HISTORY: usize = 64;

/// Metrics from a single query execution.
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    /// Wall-clock GPU time in milliseconds.
    pub gpu_time_ms: f64,
    /// GPU memory currently allocated in bytes.
    pub memory_used_bytes: u64,
    /// Scan throughput in GB/s (bytes_scanned / gpu_time).
    pub scan_throughput_gbps: f64,
    /// Number of rows processed.
    pub rows_processed: u64,
    /// Bytes scanned from source file(s).
    pub bytes_scanned: u64,
}

impl QueryMetrics {
    /// Create metrics with zero values.
    pub fn zero() -> Self {
        Self {
            gpu_time_ms: 0.0,
            memory_used_bytes: 0,
            scan_throughput_gbps: 0.0,
            rows_processed: 0,
            bytes_scanned: 0,
        }
    }
}

/// Accumulates metrics across queries and provides history for sparkline charts.
#[derive(Debug, Clone)]
pub struct GpuMetricsCollector {
    /// Running count of queries executed.
    pub query_count: u64,
    /// Most recent query metrics.
    pub latest: QueryMetrics,
    /// History of GPU times (ms) for sparkline.
    pub time_history: Vec<f64>,
    /// History of throughput (GB/s) for sparkline.
    pub throughput_history: Vec<f64>,
    /// History of memory usage (bytes) for sparkline.
    pub memory_history: Vec<u64>,
    /// Peak memory usage observed (bytes).
    pub peak_memory_bytes: u64,
}

impl GpuMetricsCollector {
    /// Create a new empty collector.
    pub fn new() -> Self {
        Self {
            query_count: 0,
            latest: QueryMetrics::zero(),
            time_history: Vec::with_capacity(MAX_HISTORY),
            throughput_history: Vec::with_capacity(MAX_HISTORY),
            memory_history: Vec::with_capacity(MAX_HISTORY),
            peak_memory_bytes: 0,
        }
    }

    /// Record metrics from a completed query.
    pub fn record(&mut self, metrics: QueryMetrics) {
        self.query_count += 1;

        // Track peak memory
        if metrics.memory_used_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = metrics.memory_used_bytes;
        }

        // Push to histories (ring-buffer style: drop oldest if at capacity)
        push_bounded(&mut self.time_history, metrics.gpu_time_ms, MAX_HISTORY);
        push_bounded(
            &mut self.throughput_history,
            metrics.scan_throughput_gbps,
            MAX_HISTORY,
        );
        push_bounded_u64(&mut self.memory_history, metrics.memory_used_bytes, MAX_HISTORY);

        self.latest = metrics;
    }

    /// GPU utilization estimate (0.0..=1.0).
    ///
    /// Rough heuristic: compare latest throughput against theoretical M4 bandwidth.
    /// M4 base: ~100 GB/s memory bandwidth. Utilization = throughput / peak_bandwidth.
    pub fn utilization_estimate(&self) -> f32 {
        const M4_PEAK_BANDWIDTH_GBPS: f64 = 100.0;
        let util = self.latest.scan_throughput_gbps / M4_PEAK_BANDWIDTH_GBPS;
        (util as f32).clamp(0.0, 1.0)
    }

    /// Memory utilization (0.0..=1.0) relative to peak observed.
    pub fn memory_utilization(&self) -> f32 {
        if self.peak_memory_bytes == 0 {
            return 0.0;
        }
        (self.latest.memory_used_bytes as f32 / self.peak_memory_bytes as f32).clamp(0.0, 1.0)
    }

    /// Get time history as u64 values (microseconds) for ratatui Sparkline.
    pub fn time_history_us(&self) -> Vec<u64> {
        self.time_history
            .iter()
            .map(|ms| (ms * 1000.0) as u64)
            .collect()
    }

    /// Get throughput history as u64 values (MB/s) for ratatui Sparkline.
    pub fn throughput_history_mbs(&self) -> Vec<u64> {
        self.throughput_history
            .iter()
            .map(|gbps| (gbps * 1000.0) as u64)
            .collect()
    }

    /// Average query time across history (ms).
    pub fn avg_time_ms(&self) -> f64 {
        if self.time_history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.time_history.iter().sum();
        sum / self.time_history.len() as f64
    }

    /// Average throughput across history (GB/s).
    pub fn avg_throughput_gbps(&self) -> f64 {
        if self.throughput_history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.throughput_history.iter().sum();
        sum / self.throughput_history.len() as f64
    }
}

impl Default for GpuMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple timer for measuring GPU execution stages.
pub struct GpuTimer {
    start: Instant,
}

impl GpuTimer {
    /// Start a new timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Stop the timer and return elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Build a QueryMetrics from the timer result and additional info.
    pub fn into_metrics(self, bytes_scanned: u64, rows_processed: u64, memory_used_bytes: u64) -> QueryMetrics {
        let gpu_time_ms = self.elapsed_ms();
        let scan_throughput_gbps = if gpu_time_ms > 0.0 {
            (bytes_scanned as f64 / 1_000_000_000.0) / (gpu_time_ms / 1000.0)
        } else {
            0.0
        };
        QueryMetrics {
            gpu_time_ms,
            memory_used_bytes,
            scan_throughput_gbps,
            rows_processed,
            bytes_scanned,
        }
    }
}

/// Format bytes into a human-readable string (B, KB, MB, GB).
pub fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Format throughput in GB/s with appropriate precision.
pub fn format_throughput(gbps: f64) -> String {
    if gbps < 0.001 {
        format!("{:.1} MB/s", gbps * 1000.0)
    } else if gbps < 1.0 {
        format!("{:.2} GB/s", gbps)
    } else {
        format!("{:.1} GB/s", gbps)
    }
}

/// Format time in adaptive units (us/ms/s).
pub fn format_time(ms: f64) -> String {
    if ms < 0.001 {
        format!("{:.1} us", ms * 1000.0)
    } else if ms < 1.0 {
        format!("{:.2} ms", ms)
    } else if ms < 1000.0 {
        format!("{:.1} ms", ms)
    } else {
        format!("{:.2} s", ms / 1000.0)
    }
}

/// Push a value to a Vec, dropping the oldest if it exceeds max_len.
fn push_bounded(vec: &mut Vec<f64>, value: f64, max_len: usize) {
    if vec.len() >= max_len {
        vec.remove(0);
    }
    vec.push(value);
}

/// Push a u64 value to a Vec, dropping the oldest if it exceeds max_len.
fn push_bounded_u64(vec: &mut Vec<u64>, value: u64, max_len: usize) {
    if vec.len() >= max_len {
        vec.remove(0);
    }
    vec.push(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_metrics_zero() {
        let m = QueryMetrics::zero();
        assert_eq!(m.gpu_time_ms, 0.0);
        assert_eq!(m.memory_used_bytes, 0);
        assert_eq!(m.scan_throughput_gbps, 0.0);
        assert_eq!(m.rows_processed, 0);
        assert_eq!(m.bytes_scanned, 0);
    }

    #[test]
    fn test_collector_new() {
        let c = GpuMetricsCollector::new();
        assert_eq!(c.query_count, 0);
        assert!(c.time_history.is_empty());
        assert!(c.throughput_history.is_empty());
        assert!(c.memory_history.is_empty());
        assert_eq!(c.peak_memory_bytes, 0);
    }

    #[test]
    fn test_collector_record() {
        let mut c = GpuMetricsCollector::new();
        c.record(QueryMetrics {
            gpu_time_ms: 5.0,
            memory_used_bytes: 1_000_000,
            scan_throughput_gbps: 50.0,
            rows_processed: 100_000,
            bytes_scanned: 10_000_000,
        });
        assert_eq!(c.query_count, 1);
        assert_eq!(c.time_history.len(), 1);
        assert_eq!(c.time_history[0], 5.0);
        assert_eq!(c.throughput_history[0], 50.0);
        assert_eq!(c.memory_history[0], 1_000_000);
        assert_eq!(c.peak_memory_bytes, 1_000_000);
    }

    #[test]
    fn test_collector_peak_memory() {
        let mut c = GpuMetricsCollector::new();
        c.record(QueryMetrics {
            gpu_time_ms: 1.0,
            memory_used_bytes: 500,
            scan_throughput_gbps: 1.0,
            rows_processed: 10,
            bytes_scanned: 100,
        });
        c.record(QueryMetrics {
            gpu_time_ms: 1.0,
            memory_used_bytes: 2000,
            scan_throughput_gbps: 1.0,
            rows_processed: 10,
            bytes_scanned: 100,
        });
        c.record(QueryMetrics {
            gpu_time_ms: 1.0,
            memory_used_bytes: 800,
            scan_throughput_gbps: 1.0,
            rows_processed: 10,
            bytes_scanned: 100,
        });
        assert_eq!(c.peak_memory_bytes, 2000);
        assert_eq!(c.latest.memory_used_bytes, 800);
    }

    #[test]
    fn test_utilization_estimate() {
        let mut c = GpuMetricsCollector::new();
        // 50 GB/s out of 100 GB/s = 0.5
        c.record(QueryMetrics {
            gpu_time_ms: 10.0,
            memory_used_bytes: 0,
            scan_throughput_gbps: 50.0,
            rows_processed: 0,
            bytes_scanned: 0,
        });
        let util = c.utilization_estimate();
        assert!((util - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_utilization_clamp() {
        let mut c = GpuMetricsCollector::new();
        // Over 100 GB/s should clamp to 1.0
        c.record(QueryMetrics {
            gpu_time_ms: 1.0,
            memory_used_bytes: 0,
            scan_throughput_gbps: 200.0,
            rows_processed: 0,
            bytes_scanned: 0,
        });
        assert_eq!(c.utilization_estimate(), 1.0);
    }

    #[test]
    fn test_memory_utilization() {
        let mut c = GpuMetricsCollector::new();
        c.record(QueryMetrics {
            gpu_time_ms: 1.0,
            memory_used_bytes: 100,
            scan_throughput_gbps: 1.0,
            rows_processed: 0,
            bytes_scanned: 0,
        });
        // Only one sample, peak = current = 100
        assert!((c.memory_utilization() - 1.0).abs() < 0.01);

        c.record(QueryMetrics {
            gpu_time_ms: 1.0,
            memory_used_bytes: 50,
            scan_throughput_gbps: 1.0,
            rows_processed: 0,
            bytes_scanned: 0,
        });
        // Peak = 100, current = 50 => 0.5
        assert!((c.memory_utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_time_history_us() {
        let mut c = GpuMetricsCollector::new();
        c.record(QueryMetrics {
            gpu_time_ms: 2.5,
            memory_used_bytes: 0,
            scan_throughput_gbps: 0.0,
            rows_processed: 0,
            bytes_scanned: 0,
        });
        let history = c.time_history_us();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0], 2500);
    }

    #[test]
    fn test_throughput_history_mbs() {
        let mut c = GpuMetricsCollector::new();
        c.record(QueryMetrics {
            gpu_time_ms: 1.0,
            memory_used_bytes: 0,
            scan_throughput_gbps: 1.5,
            rows_processed: 0,
            bytes_scanned: 0,
        });
        let history = c.throughput_history_mbs();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0], 1500);
    }

    #[test]
    fn test_avg_time_ms() {
        let mut c = GpuMetricsCollector::new();
        assert_eq!(c.avg_time_ms(), 0.0);
        c.record(QueryMetrics {
            gpu_time_ms: 10.0,
            memory_used_bytes: 0,
            scan_throughput_gbps: 0.0,
            rows_processed: 0,
            bytes_scanned: 0,
        });
        c.record(QueryMetrics {
            gpu_time_ms: 20.0,
            memory_used_bytes: 0,
            scan_throughput_gbps: 0.0,
            rows_processed: 0,
            bytes_scanned: 0,
        });
        assert!((c.avg_time_ms() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_avg_throughput() {
        let mut c = GpuMetricsCollector::new();
        assert_eq!(c.avg_throughput_gbps(), 0.0);
        c.record(QueryMetrics {
            gpu_time_ms: 1.0,
            memory_used_bytes: 0,
            scan_throughput_gbps: 40.0,
            rows_processed: 0,
            bytes_scanned: 0,
        });
        c.record(QueryMetrics {
            gpu_time_ms: 1.0,
            memory_used_bytes: 0,
            scan_throughput_gbps: 60.0,
            rows_processed: 0,
            bytes_scanned: 0,
        });
        assert!((c.avg_throughput_gbps() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_history_bounded() {
        let mut c = GpuMetricsCollector::new();
        for i in 0..MAX_HISTORY + 10 {
            c.record(QueryMetrics {
                gpu_time_ms: i as f64,
                memory_used_bytes: i as u64,
                scan_throughput_gbps: i as f64,
                rows_processed: 0,
                bytes_scanned: 0,
            });
        }
        assert_eq!(c.time_history.len(), MAX_HISTORY);
        assert_eq!(c.throughput_history.len(), MAX_HISTORY);
        assert_eq!(c.memory_history.len(), MAX_HISTORY);
        // Oldest should have been dropped
        assert_eq!(c.time_history[0], 10.0);
    }

    #[test]
    fn test_gpu_timer() {
        let timer = GpuTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 5.0); // At least ~5ms (sleep jitter)
        assert!(elapsed < 100.0); // Sanity upper bound
    }

    #[test]
    fn test_gpu_timer_into_metrics() {
        let timer = GpuTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let metrics = timer.into_metrics(1_000_000_000, 1_000_000, 500_000);
        assert!(metrics.gpu_time_ms >= 5.0);
        assert_eq!(metrics.bytes_scanned, 1_000_000_000);
        assert_eq!(metrics.rows_processed, 1_000_000);
        assert_eq!(metrics.memory_used_bytes, 500_000);
        // 1 GB scanned in ~10ms = ~100 GB/s
        assert!(metrics.scan_throughput_gbps > 1.0);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.0 MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_format_throughput() {
        assert_eq!(format_throughput(0.0005), "0.5 MB/s");
        assert_eq!(format_throughput(0.5), "0.50 GB/s");
        assert_eq!(format_throughput(50.0), "50.0 GB/s");
    }

    #[test]
    fn test_format_time() {
        assert_eq!(format_time(0.0005), "0.5 us");
        assert_eq!(format_time(0.5), "0.50 ms");
        assert_eq!(format_time(5.0), "5.0 ms");
        assert_eq!(format_time(1500.0), "1.50 s");
    }

    #[test]
    fn test_push_bounded() {
        let mut v = Vec::new();
        for i in 0..5 {
            push_bounded(&mut v, i as f64, 3);
        }
        assert_eq!(v, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_push_bounded_u64() {
        let mut v = Vec::new();
        for i in 0..5u64 {
            push_bounded_u64(&mut v, i, 3);
        }
        assert_eq!(v, vec![2, 3, 4]);
    }

    #[test]
    fn test_default_collector() {
        let c = GpuMetricsCollector::default();
        assert_eq!(c.query_count, 0);
    }
}
