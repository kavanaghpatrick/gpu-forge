//! Pipeline profiler for GPU search stages.
//!
//! Tracks per-stage timing and counters across the full search pipeline:
//! walk -> filter -> batch -> GPU load -> GPU dispatch -> resolve -> total.

use std::fmt;

/// Per-stage timing and counters for the full search pipeline.
///
/// All `_us` fields are microseconds measured via `Instant::now().elapsed()`.
/// Counters track volume at each stage for throughput and selectivity analysis.
#[derive(Debug, Clone, Default)]
pub struct PipelineProfile {
    // ---- Stage timings (microseconds) ----

    /// Time spent walking the filesystem or reading index.
    pub walk_us: u64,
    /// Time spent filtering files (gitignore, binary detection, size).
    pub filter_us: u64,
    /// Time spent batching files into GPU-sized chunks.
    pub batch_us: u64,
    /// Time spent loading file content into GPU buffers.
    pub gpu_load_us: u64,
    /// Time spent in GPU compute dispatches.
    pub gpu_dispatch_us: u64,
    /// Time spent in CPU-side match resolution (byte_offset -> line/column).
    pub resolve_us: u64,
    /// Total wall-clock time from search start to final result.
    pub total_us: u64,
    /// Time to first result (first SearchUpdate::ContentMatches sent).
    pub ttfr_us: u64,

    // ---- Counters ----

    /// Files discovered by walk (before filtering).
    pub files_walked: u32,
    /// Files remaining after filter (gitignore, binary, size).
    pub files_filtered: u32,
    /// Files actually searched by GPU.
    pub files_searched: u32,
    /// Total bytes sent to GPU for search.
    pub bytes_searched: u64,
    /// Number of GPU compute dispatches.
    pub gpu_dispatches: u32,
    /// Raw match count from GPU (before CPU verification/resolution).
    pub matches_raw: u32,
    /// Matches successfully resolved to file:line:column.
    pub matches_resolved: u32,
    /// Matches rejected by CPU verification or resolution.
    pub matches_rejected: u32,
}

impl PipelineProfile {
    /// Compute walk-to-filter selectivity (fraction of files that pass filter).
    pub fn filter_selectivity(&self) -> f64 {
        if self.files_walked == 0 {
            return 0.0;
        }
        self.files_filtered as f64 / self.files_walked as f64
    }

    /// Compute GPU throughput in GB/s.
    pub fn gpu_throughput_gbps(&self) -> f64 {
        if self.gpu_dispatch_us == 0 {
            return 0.0;
        }
        let gb = self.bytes_searched as f64 / (1024.0 * 1024.0 * 1024.0);
        let secs = self.gpu_dispatch_us as f64 / 1_000_000.0;
        gb / secs
    }

    /// Compute match rejection rate.
    pub fn rejection_rate(&self) -> f64 {
        if self.matches_raw == 0 {
            return 0.0;
        }
        self.matches_rejected as f64 / self.matches_raw as f64
    }
}

impl fmt::Display for PipelineProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pipeline Profile")?;
        writeln!(f, "  Stage Breakdown:")?;
        writeln!(f, "    walk:         {:>8} us", self.walk_us)?;
        writeln!(f, "    filter:       {:>8} us", self.filter_us)?;
        writeln!(f, "    batch:        {:>8} us", self.batch_us)?;
        writeln!(f, "    gpu_load:     {:>8} us", self.gpu_load_us)?;
        writeln!(f, "    gpu_dispatch: {:>8} us", self.gpu_dispatch_us)?;
        writeln!(f, "    resolve:      {:>8} us", self.resolve_us)?;
        writeln!(
            f,
            "    total:        {:>8} us ({:.1} ms)",
            self.total_us,
            self.total_us as f64 / 1000.0
        )?;
        writeln!(
            f,
            "    ttfr:         {:>8} us ({:.1} ms)",
            self.ttfr_us,
            self.ttfr_us as f64 / 1000.0
        )?;
        writeln!(f, "  Counters:")?;
        writeln!(f, "    files_walked:    {:>8}", self.files_walked)?;
        writeln!(f, "    files_filtered:  {:>8}", self.files_filtered)?;
        writeln!(f, "    files_searched:  {:>8}", self.files_searched)?;
        writeln!(
            f,
            "    bytes_searched:  {:>8} ({:.2} MB)",
            self.bytes_searched,
            self.bytes_searched as f64 / (1024.0 * 1024.0)
        )?;
        writeln!(f, "    gpu_dispatches:  {:>8}", self.gpu_dispatches)?;
        writeln!(f, "    matches_raw:     {:>8}", self.matches_raw)?;
        writeln!(f, "    matches_resolved:{:>8}", self.matches_resolved)?;
        writeln!(f, "    matches_rejected:{:>8}", self.matches_rejected)?;

        if self.gpu_dispatch_us > 0 {
            writeln!(f, "  Derived:")?;
            writeln!(
                f,
                "    gpu_throughput:  {:>8.1} GB/s",
                self.gpu_throughput_gbps()
            )?;
            writeln!(
                f,
                "    filter_select:   {:>7.1}%",
                self.filter_selectivity() * 100.0
            )?;
            writeln!(
                f,
                "    rejection_rate:  {:>7.1}%",
                self.rejection_rate() * 100.0
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_profile() {
        let p = PipelineProfile::default();
        assert_eq!(p.walk_us, 0);
        assert_eq!(p.total_us, 0);
        assert_eq!(p.files_walked, 0);
        assert_eq!(p.matches_raw, 0);
    }

    #[test]
    fn test_display_formats_stage_breakdown() {
        let p = PipelineProfile {
            walk_us: 1200,
            filter_us: 300,
            batch_us: 50,
            gpu_load_us: 800,
            gpu_dispatch_us: 5000,
            resolve_us: 200,
            total_us: 7550,
            ttfr_us: 2300,
            files_walked: 10000,
            files_filtered: 8500,
            files_searched: 8500,
            bytes_searched: 50 * 1024 * 1024,
            gpu_dispatches: 4,
            matches_raw: 120,
            matches_resolved: 115,
            matches_rejected: 5,
        };
        let output = format!("{}", p);
        assert!(output.contains("Pipeline Profile"));
        assert!(output.contains("walk:"));
        assert!(output.contains("filter:"));
        assert!(output.contains("batch:"));
        assert!(output.contains("gpu_load:"));
        assert!(output.contains("gpu_dispatch:"));
        assert!(output.contains("resolve:"));
        assert!(output.contains("total:"));
        assert!(output.contains("ttfr:"));
        assert!(output.contains("files_walked:"));
        assert!(output.contains("files_searched:"));
        assert!(output.contains("bytes_searched:"));
        assert!(output.contains("gpu_dispatches:"));
        assert!(output.contains("matches_raw:"));
        assert!(output.contains("matches_resolved:"));
        assert!(output.contains("matches_rejected:"));
        assert!(output.contains("gpu_throughput:"));
        assert!(output.contains("filter_select:"));
        assert!(output.contains("rejection_rate:"));
    }

    #[test]
    fn test_derived_metrics() {
        let p = PipelineProfile {
            files_walked: 1000,
            files_filtered: 800,
            bytes_searched: 1024 * 1024 * 1024, // 1 GB
            gpu_dispatch_us: 1_000_000,          // 1 second
            matches_raw: 100,
            matches_rejected: 10,
            ..Default::default()
        };
        assert!((p.filter_selectivity() - 0.8).abs() < 0.001);
        assert!((p.gpu_throughput_gbps() - 1.0).abs() < 0.001);
        assert!((p.rejection_rate() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_derived_metrics_zero_division() {
        let p = PipelineProfile::default();
        assert_eq!(p.filter_selectivity(), 0.0);
        assert_eq!(p.gpu_throughput_gbps(), 0.0);
        assert_eq!(p.rejection_rate(), 0.0);
    }
}
