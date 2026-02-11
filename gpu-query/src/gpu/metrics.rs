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
    /// Whether this is a warm query (data already in memory/cache).
    pub is_warm: bool,
}

/// CPU comparison estimate based on assumed CPU memory bandwidth.
#[derive(Debug, Clone)]
pub struct CpuEstimate {
    /// Estimated time the same scan would take on CPU (ms).
    pub cpu_estimate_ms: f64,
    /// Speedup factor: cpu_estimate / gpu_time.
    pub speedup_vs_cpu: f64,
}

/// Assumed CPU memory bandwidth for comparison (single-threaded sequential scan).
/// M4 has ~60 GB/s DRAM bandwidth but typical single-threaded scan throughput
/// is much lower due to cache/memory hierarchy: ~6.5 GB/s is realistic for
/// a single-core sequential CSV/Parquet scan.
pub const CPU_BANDWIDTH_GBPS: f64 = 6.5;

impl CpuEstimate {
    /// Compute CPU comparison from bytes_scanned and actual GPU time.
    ///
    /// Formula: cpu_estimate_ms = (bytes_scanned / CPU_BANDWIDTH_GBPS) * 1000
    ///          speedup = cpu_estimate_ms / gpu_time_ms
    pub fn from_metrics(bytes_scanned: u64, gpu_time_ms: f64) -> Self {
        let bytes_gb = bytes_scanned as f64 / 1_000_000_000.0;
        let cpu_estimate_ms = (bytes_gb / CPU_BANDWIDTH_GBPS) * 1000.0;
        let speedup_vs_cpu = if gpu_time_ms > 0.0 {
            cpu_estimate_ms / gpu_time_ms
        } else {
            0.0
        };
        Self {
            cpu_estimate_ms,
            speedup_vs_cpu,
        }
    }
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
            is_warm: false,
        }
    }

    /// Compute the CPU comparison estimate for this query.
    pub fn cpu_estimate(&self) -> CpuEstimate {
        CpuEstimate::from_metrics(self.bytes_scanned, self.gpu_time_ms)
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
            is_warm: false,
        }
    }

    /// Build a QueryMetrics with warm/cold indicator.
    pub fn into_metrics_with_warmth(self, bytes_scanned: u64, rows_processed: u64, memory_used_bytes: u64, is_warm: bool) -> QueryMetrics {
        let mut m = self.into_metrics(bytes_scanned, rows_processed, memory_used_bytes);
        m.is_warm = is_warm;
        m
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

/// Format a row count with SI suffixes (e.g., 142_000_000 -> "142M").
pub fn format_row_count(rows: u64) -> String {
    if rows >= 1_000_000_000 {
        let val = rows as f64 / 1_000_000_000.0;
        if val >= 100.0 {
            format!("{:.0}B", val)
        } else if val >= 10.0 {
            format!("{:.1}B", val)
        } else {
            format!("{:.2}B", val)
        }
    } else if rows >= 1_000_000 {
        let val = rows as f64 / 1_000_000.0;
        if val >= 100.0 {
            format!("{:.0}M", val)
        } else if val >= 10.0 {
            format!("{:.1}M", val)
        } else {
            format!("{:.2}M", val)
        }
    } else if rows >= 1_000 {
        let val = rows as f64 / 1_000.0;
        if val >= 100.0 {
            format!("{:.0}K", val)
        } else if val >= 10.0 {
            format!("{:.1}K", val)
        } else {
            format!("{:.2}K", val)
        }
    } else {
        format!("{}", rows)
    }
}

/// Format bytes with SI-style units for display (e.g., 8_400_000_000 -> "8.4 GB").
/// Uses decimal units (1 GB = 1,000,000,000 bytes) like data throughput conventions.
pub fn format_data_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        let val = bytes as f64 / 1_000_000_000.0;
        if val >= 100.0 {
            format!("{:.0} GB", val)
        } else if val >= 10.0 {
            format!("{:.1} GB", val)
        } else {
            format!("{:.2} GB", val)
        }
    } else if bytes >= 1_000_000 {
        let val = bytes as f64 / 1_000_000.0;
        if val >= 100.0 {
            format!("{:.0} MB", val)
        } else if val >= 10.0 {
            format!("{:.1} MB", val)
        } else {
            format!("{:.2} MB", val)
        }
    } else if bytes >= 1_000 {
        let val = bytes as f64 / 1_000.0;
        if val >= 10.0 {
            format!("{:.1} KB", val)
        } else {
            format!("{:.2} KB", val)
        }
    } else {
        format!("{} B", bytes)
    }
}

/// Format a CPU comparison speedup (e.g., 312.5 -> "~312x vs CPU").
pub fn format_speedup(speedup: f64) -> String {
    if speedup < 1.0 {
        format!("~{:.1}x vs CPU", speedup)
    } else if speedup < 10.0 {
        format!("~{:.1}x vs CPU", speedup)
    } else {
        format!("~{:.0}x vs CPU", speedup)
    }
}

/// Build the full performance summary line from QueryMetrics.
///
/// Format: "142M rows | 8.4 GB | 2.3ms | GPU 94% | ~312x vs CPU"
pub fn build_metrics_performance_line(metrics: &QueryMetrics, utilization: f32) -> String {
    let rows = format_row_count(metrics.rows_processed);
    let data = format_data_bytes(metrics.bytes_scanned);
    let time = format_time(metrics.gpu_time_ms);
    let gpu_pct = format!("GPU {:.0}%", utilization * 100.0);
    let estimate = metrics.cpu_estimate();
    let speedup = format_speedup(estimate.speedup_vs_cpu);

    let warm_cold = if metrics.is_warm { " (warm)" } else { " (cold)" };

    format!(
        " {} rows | {} | {}{} | {} | {}",
        rows, data, time, warm_cold, gpu_pct, speedup
    )
}

/// Per-stage timing profile for the query pipeline.
///
/// Each field represents wall-clock time in milliseconds for that pipeline stage.
/// Uses CPU-side timing (Instant::now()) around each stage.
/// Metal timestamp counters (MTLCounterSampleBuffer) can replace this for
/// hardware-accurate GPU-side timing in the future.
#[derive(Debug, Clone)]
pub struct PipelineProfile {
    /// SQL parsing time.
    pub parse_ms: f64,
    /// Query planning / optimization time.
    pub plan_ms: f64,
    /// mmap / buffer warming time.
    pub mmap_ms: f64,
    /// GPU scan (CSV/JSON/Parquet parse) time.
    pub scan_ms: f64,
    /// GPU filter kernel time.
    pub filter_ms: f64,
    /// GPU aggregation kernel time.
    pub aggregate_ms: f64,
    /// GPU sort kernel time.
    pub sort_ms: f64,
    /// GPU -> CPU data transfer / readback time.
    pub transfer_ms: f64,
    /// Result formatting time.
    pub format_ms: f64,
    /// Total end-to-end time.
    pub total_ms: f64,
}

impl PipelineProfile {
    /// Create a profile with all zeros.
    pub fn zero() -> Self {
        Self {
            parse_ms: 0.0,
            plan_ms: 0.0,
            mmap_ms: 0.0,
            scan_ms: 0.0,
            filter_ms: 0.0,
            aggregate_ms: 0.0,
            sort_ms: 0.0,
            transfer_ms: 0.0,
            format_ms: 0.0,
            total_ms: 0.0,
        }
    }

    /// Return all stages as (label, time_ms) pairs.
    pub fn stages(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("Parse", self.parse_ms),
            ("Plan", self.plan_ms),
            ("mmap", self.mmap_ms),
            ("Scan", self.scan_ms),
            ("Filter", self.filter_ms),
            ("Aggregate", self.aggregate_ms),
            ("Sort", self.sort_ms),
            ("Transfer", self.transfer_ms),
            ("Format", self.format_ms),
        ]
    }

    /// Sum of all individual stage times (for validation against total_ms).
    pub fn stages_sum_ms(&self) -> f64 {
        self.parse_ms
            + self.plan_ms
            + self.mmap_ms
            + self.scan_ms
            + self.filter_ms
            + self.aggregate_ms
            + self.sort_ms
            + self.transfer_ms
            + self.format_ms
    }
}

/// Maximum bar width in characters for the profile timeline.
const MAX_BAR_WIDTH: usize = 50;

/// Render a proportional ASCII bar chart of the pipeline profile.
///
/// Output format:
/// ```text
/// Parse:      0.1ms  ██
/// Plan:       0.0ms  █
/// mmap:       0.3ms  ████
/// Scan:      12.5ms  ████████████████████████████████████████████████
/// Filter:     1.2ms  █████
/// Aggregate:  0.8ms  ███
/// Sort:       0.0ms
/// Transfer:   0.2ms  █
/// Format:     0.1ms  █
/// Total:     15.2ms
/// ```
///
/// Bar widths are proportional to the maximum stage time.
/// Stages with zero time get no bar.
pub fn render_profile_timeline(profile: &PipelineProfile) -> String {
    let stages = profile.stages();

    // Find maximum stage time for proportional bar calculation
    let max_time = stages
        .iter()
        .map(|(_, t)| *t)
        .fold(0.0_f64, f64::max);

    let mut lines = Vec::with_capacity(stages.len() + 1);

    for (label, time_ms) in &stages {
        let bar = if max_time > 0.0 && *time_ms > 0.0 {
            let width = ((*time_ms / max_time) * MAX_BAR_WIDTH as f64).round() as usize;
            let width = width.max(1); // at least 1 block if > 0
            "\u{2588}".repeat(width) // U+2588 FULL BLOCK
        } else {
            String::new()
        };

        lines.push(format!(
            "{:<10} {:>6.1}ms  {}",
            format!("{}:", label),
            time_ms,
            bar,
        ));
    }

    // Total line (no bar)
    lines.push(format!(
        "{:<10} {:>6.1}ms",
        "Total:", profile.total_ms,
    ));

    lines.join("\n")
}

/// Compute proportional bar width for a given value relative to max.
/// Returns width in characters (0..=max_width).
pub fn compute_bar_width(value: f64, max_value: f64, max_width: usize) -> usize {
    if max_value <= 0.0 || value <= 0.0 {
        return 0;
    }
    let width = ((value / max_value) * max_width as f64).round() as usize;
    width.clamp(1, max_width) // at least 1 if value > 0
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

    /// Helper to build a test QueryMetrics quickly.
    fn test_metrics(gpu_time_ms: f64, memory_used_bytes: u64, scan_throughput_gbps: f64, rows_processed: u64, bytes_scanned: u64) -> QueryMetrics {
        QueryMetrics {
            gpu_time_ms,
            memory_used_bytes,
            scan_throughput_gbps,
            rows_processed,
            bytes_scanned,
            is_warm: false,
        }
    }

    #[test]
    fn test_query_metrics_zero() {
        let m = QueryMetrics::zero();
        assert_eq!(m.gpu_time_ms, 0.0);
        assert_eq!(m.memory_used_bytes, 0);
        assert_eq!(m.scan_throughput_gbps, 0.0);
        assert_eq!(m.rows_processed, 0);
        assert_eq!(m.bytes_scanned, 0);
        assert!(!m.is_warm);
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
        c.record(test_metrics(5.0, 1_000_000, 50.0, 100_000, 10_000_000));
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
        c.record(test_metrics(1.0, 500, 1.0, 10, 100));
        c.record(test_metrics(1.0, 2000, 1.0, 10, 100));
        c.record(test_metrics(1.0, 800, 1.0, 10, 100));
        assert_eq!(c.peak_memory_bytes, 2000);
        assert_eq!(c.latest.memory_used_bytes, 800);
    }

    #[test]
    fn test_utilization_estimate() {
        let mut c = GpuMetricsCollector::new();
        // 50 GB/s out of 100 GB/s = 0.5
        c.record(test_metrics(10.0, 0, 50.0, 0, 0));
        let util = c.utilization_estimate();
        assert!((util - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_utilization_clamp() {
        let mut c = GpuMetricsCollector::new();
        // Over 100 GB/s should clamp to 1.0
        c.record(test_metrics(1.0, 0, 200.0, 0, 0));
        assert_eq!(c.utilization_estimate(), 1.0);
    }

    #[test]
    fn test_memory_utilization() {
        let mut c = GpuMetricsCollector::new();
        c.record(test_metrics(1.0, 100, 1.0, 0, 0));
        // Only one sample, peak = current = 100
        assert!((c.memory_utilization() - 1.0).abs() < 0.01);

        c.record(test_metrics(1.0, 50, 1.0, 0, 0));
        // Peak = 100, current = 50 => 0.5
        assert!((c.memory_utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_time_history_us() {
        let mut c = GpuMetricsCollector::new();
        c.record(test_metrics(2.5, 0, 0.0, 0, 0));
        let history = c.time_history_us();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0], 2500);
    }

    #[test]
    fn test_throughput_history_mbs() {
        let mut c = GpuMetricsCollector::new();
        c.record(test_metrics(1.0, 0, 1.5, 0, 0));
        let history = c.throughput_history_mbs();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0], 1500);
    }

    #[test]
    fn test_avg_time_ms() {
        let mut c = GpuMetricsCollector::new();
        assert_eq!(c.avg_time_ms(), 0.0);
        c.record(test_metrics(10.0, 0, 0.0, 0, 0));
        c.record(test_metrics(20.0, 0, 0.0, 0, 0));
        assert!((c.avg_time_ms() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_avg_throughput() {
        let mut c = GpuMetricsCollector::new();
        assert_eq!(c.avg_throughput_gbps(), 0.0);
        c.record(test_metrics(1.0, 0, 40.0, 0, 0));
        c.record(test_metrics(1.0, 0, 60.0, 0, 0));
        assert!((c.avg_throughput_gbps() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_history_bounded() {
        let mut c = GpuMetricsCollector::new();
        for i in 0..MAX_HISTORY + 10 {
            c.record(test_metrics(i as f64, i as u64, i as f64, 0, 0));
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
        assert!(!metrics.is_warm);
        // 1 GB scanned in ~10ms = ~100 GB/s
        assert!(metrics.scan_throughput_gbps > 1.0);
    }

    #[test]
    fn test_gpu_timer_into_metrics_with_warmth() {
        let timer = GpuTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let metrics = timer.into_metrics_with_warmth(500_000_000, 500_000, 100_000, true);
        assert!(metrics.is_warm);
        assert_eq!(metrics.rows_processed, 500_000);
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

    // ---- CPU estimate tests ----

    #[test]
    fn test_cpu_estimate_basic() {
        // 1 GB scanned in 1ms on GPU
        // CPU estimate: 1 GB / 6.5 GB/s = 153.8ms
        // Speedup: 153.8 / 1.0 = ~153.8x
        let est = CpuEstimate::from_metrics(1_000_000_000, 1.0);
        assert!((est.cpu_estimate_ms - 153.846).abs() < 1.0);
        assert!((est.speedup_vs_cpu - 153.846).abs() < 1.0);
    }

    #[test]
    fn test_cpu_estimate_small_data() {
        // 10 MB scanned in 0.5ms
        // CPU estimate: 0.01 GB / 6.5 GB/s = 1.538ms
        // Speedup: 1.538 / 0.5 = ~3.08x
        let est = CpuEstimate::from_metrics(10_000_000, 0.5);
        assert!((est.cpu_estimate_ms - 1.538).abs() < 0.1);
        assert!((est.speedup_vs_cpu - 3.077).abs() < 0.1);
    }

    #[test]
    fn test_cpu_estimate_zero_time() {
        let est = CpuEstimate::from_metrics(1_000_000_000, 0.0);
        assert_eq!(est.speedup_vs_cpu, 0.0);
    }

    #[test]
    fn test_cpu_estimate_zero_bytes() {
        let est = CpuEstimate::from_metrics(0, 5.0);
        assert_eq!(est.cpu_estimate_ms, 0.0);
        assert_eq!(est.speedup_vs_cpu, 0.0);
    }

    #[test]
    fn test_query_metrics_cpu_estimate() {
        let m = QueryMetrics {
            gpu_time_ms: 2.3,
            memory_used_bytes: 0,
            scan_throughput_gbps: 0.0,
            rows_processed: 142_000_000,
            bytes_scanned: 8_400_000_000,
            is_warm: true,
        };
        let est = m.cpu_estimate();
        // 8.4 GB / 6.5 GB/s = 1292.3ms
        // Speedup: 1292.3 / 2.3 = ~561.9x
        assert!((est.cpu_estimate_ms - 1292.3).abs() < 1.0);
        assert!(est.speedup_vs_cpu > 500.0);
    }

    // ---- format_row_count tests ----

    #[test]
    fn test_format_row_count_small() {
        assert_eq!(format_row_count(0), "0");
        assert_eq!(format_row_count(42), "42");
        assert_eq!(format_row_count(999), "999");
    }

    #[test]
    fn test_format_row_count_thousands() {
        assert_eq!(format_row_count(1_000), "1.00K");
        assert_eq!(format_row_count(1_500), "1.50K");
        assert_eq!(format_row_count(10_000), "10.0K");
        assert_eq!(format_row_count(100_000), "100K");
        assert_eq!(format_row_count(999_999), "1000K");
    }

    #[test]
    fn test_format_row_count_millions() {
        assert_eq!(format_row_count(1_000_000), "1.00M");
        assert_eq!(format_row_count(1_500_000), "1.50M");
        assert_eq!(format_row_count(10_000_000), "10.0M");
        assert_eq!(format_row_count(142_000_000), "142M");
        assert_eq!(format_row_count(999_000_000), "999M");
    }

    #[test]
    fn test_format_row_count_billions() {
        assert_eq!(format_row_count(1_000_000_000), "1.00B");
        assert_eq!(format_row_count(10_000_000_000), "10.0B");
    }

    // ---- format_data_bytes tests ----

    #[test]
    fn test_format_data_bytes_small() {
        assert_eq!(format_data_bytes(500), "500 B");
    }

    #[test]
    fn test_format_data_bytes_kb() {
        assert_eq!(format_data_bytes(1_500), "1.50 KB");
        assert_eq!(format_data_bytes(50_000), "50.0 KB");
    }

    #[test]
    fn test_format_data_bytes_mb() {
        assert_eq!(format_data_bytes(1_500_000), "1.50 MB");
        assert_eq!(format_data_bytes(50_000_000), "50.0 MB");
        assert_eq!(format_data_bytes(500_000_000), "500 MB");
    }

    #[test]
    fn test_format_data_bytes_gb() {
        assert_eq!(format_data_bytes(1_000_000_000), "1.00 GB");
        assert_eq!(format_data_bytes(8_400_000_000), "8.40 GB");
        assert_eq!(format_data_bytes(50_000_000_000), "50.0 GB");
        assert_eq!(format_data_bytes(500_000_000_000), "500 GB");
    }

    // ---- format_speedup tests ----

    #[test]
    fn test_format_speedup_small() {
        assert_eq!(format_speedup(0.5), "~0.5x vs CPU");
        assert_eq!(format_speedup(3.7), "~3.7x vs CPU");
    }

    #[test]
    fn test_format_speedup_large() {
        assert_eq!(format_speedup(312.5), "~312x vs CPU");
        assert_eq!(format_speedup(15.0), "~15x vs CPU");
    }

    // ---- build_metrics_performance_line tests ----

    #[test]
    fn test_build_metrics_performance_line() {
        let m = QueryMetrics {
            gpu_time_ms: 2.3,
            memory_used_bytes: 0,
            scan_throughput_gbps: 94.0, // near M4 peak
            rows_processed: 142_000_000,
            bytes_scanned: 8_400_000_000,
            is_warm: false,
        };
        let line = build_metrics_performance_line(&m, 0.94);
        assert!(line.contains("142M rows"));
        assert!(line.contains("8.40 GB"));
        assert!(line.contains("2.3 ms"));
        assert!(line.contains("GPU 94%"));
        assert!(line.contains("vs CPU"));
        assert!(line.contains("(cold)"));
    }

    #[test]
    fn test_build_metrics_performance_line_warm() {
        let m = QueryMetrics {
            gpu_time_ms: 1.0,
            memory_used_bytes: 0,
            scan_throughput_gbps: 50.0,
            rows_processed: 1_000_000,
            bytes_scanned: 50_000_000,
            is_warm: true,
        };
        let line = build_metrics_performance_line(&m, 0.50);
        assert!(line.contains("1.00M rows"));
        assert!(line.contains("50.0 MB"));
        assert!(line.contains("(warm)"));
        assert!(line.contains("GPU 50%"));
    }

    // ---- PipelineProfile tests ----

    #[test]
    fn test_pipeline_profile_zero() {
        let p = PipelineProfile::zero();
        assert_eq!(p.parse_ms, 0.0);
        assert_eq!(p.scan_ms, 0.0);
        assert_eq!(p.total_ms, 0.0);
        assert_eq!(p.stages_sum_ms(), 0.0);
    }

    #[test]
    fn test_pipeline_profile_stages() {
        let p = PipelineProfile {
            parse_ms: 0.1,
            plan_ms: 0.05,
            mmap_ms: 0.3,
            scan_ms: 12.5,
            filter_ms: 1.2,
            aggregate_ms: 0.8,
            sort_ms: 0.0,
            transfer_ms: 0.2,
            format_ms: 0.1,
            total_ms: 15.2,
        };
        let stages = p.stages();
        assert_eq!(stages.len(), 9);
        assert_eq!(stages[0].0, "Parse");
        assert_eq!(stages[0].1, 0.1);
        assert_eq!(stages[3].0, "Scan");
        assert_eq!(stages[3].1, 12.5);
    }

    #[test]
    fn test_pipeline_profile_stages_sum() {
        let p = PipelineProfile {
            parse_ms: 1.0,
            plan_ms: 2.0,
            mmap_ms: 3.0,
            scan_ms: 4.0,
            filter_ms: 5.0,
            aggregate_ms: 6.0,
            sort_ms: 0.0,
            transfer_ms: 1.0,
            format_ms: 1.0,
            total_ms: 23.0,
        };
        assert!((p.stages_sum_ms() - 23.0).abs() < 0.001);
    }

    // ---- render_profile_timeline tests ----

    #[test]
    fn test_render_profile_timeline_basic() {
        let p = PipelineProfile {
            parse_ms: 0.1,
            plan_ms: 0.0,
            mmap_ms: 0.3,
            scan_ms: 12.5,
            filter_ms: 1.2,
            aggregate_ms: 0.8,
            sort_ms: 0.0,
            transfer_ms: 0.2,
            format_ms: 0.1,
            total_ms: 15.2,
        };
        let output = render_profile_timeline(&p);

        // Should have 10 lines (9 stages + 1 total)
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 10, "expected 10 lines, got: {:?}", lines);

        // First line should be Parse
        assert!(lines[0].contains("Parse:"), "first line: {}", lines[0]);
        assert!(lines[0].contains("0.1ms"), "first line: {}", lines[0]);

        // Scan should have the longest bar (12.5ms is max)
        assert!(lines[3].contains("Scan:"), "scan line: {}", lines[3]);
        assert!(lines[3].contains("12.5ms"), "scan line: {}", lines[3]);
        // Scan bar should contain the most blocks
        let scan_blocks: usize = lines[3].matches('\u{2588}').count();
        assert_eq!(scan_blocks, MAX_BAR_WIDTH, "scan should have max bar width");

        // Total line should be last
        assert!(lines[9].contains("Total:"), "last line: {}", lines[9]);
        assert!(lines[9].contains("15.2ms"), "last line: {}", lines[9]);
        // Total line should have no bar
        assert_eq!(lines[9].matches('\u{2588}').count(), 0);
    }

    #[test]
    fn test_render_profile_timeline_all_zero() {
        let p = PipelineProfile::zero();
        let output = render_profile_timeline(&p);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 10);
        // No bars should appear
        for line in &lines {
            assert_eq!(line.matches('\u{2588}').count(), 0, "no bars expected in: {}", line);
        }
    }

    #[test]
    fn test_render_profile_timeline_single_stage() {
        let mut p = PipelineProfile::zero();
        p.scan_ms = 5.0;
        p.total_ms = 5.0;
        let output = render_profile_timeline(&p);
        let lines: Vec<&str> = output.lines().collect();

        // Only scan should have a bar (full width)
        let scan_blocks: usize = lines[3].matches('\u{2588}').count();
        assert_eq!(scan_blocks, MAX_BAR_WIDTH);

        // Other stages (with 0.0) should have no bar
        assert_eq!(lines[0].matches('\u{2588}').count(), 0); // Parse
        assert_eq!(lines[1].matches('\u{2588}').count(), 0); // Plan
    }

    // ---- compute_bar_width tests ----

    #[test]
    fn test_compute_bar_width_basic() {
        assert_eq!(compute_bar_width(50.0, 100.0, 50), 25);
        assert_eq!(compute_bar_width(100.0, 100.0, 50), 50);
        assert_eq!(compute_bar_width(0.0, 100.0, 50), 0);
    }

    #[test]
    fn test_compute_bar_width_minimum() {
        // Very small but positive value should get at least 1
        assert_eq!(compute_bar_width(0.001, 100.0, 50), 1);
    }

    #[test]
    fn test_compute_bar_width_zero_max() {
        assert_eq!(compute_bar_width(10.0, 0.0, 50), 0);
    }

    #[test]
    fn test_compute_bar_width_negative() {
        assert_eq!(compute_bar_width(-5.0, 100.0, 50), 0);
    }
}
