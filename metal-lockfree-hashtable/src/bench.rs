//! Benchmark harness with statistical analysis.

use crate::table::{GpuHashTable, Version};
use rand::Rng;

const EMPTY_KEY: u32 = 0xFFFFFFFF;

/// Benchmark configuration.
pub struct BenchConfig {
    /// Table capacity (will be rounded up to next power of 2).
    pub capacity: u32,
    /// Number of keys to insert/lookup.
    pub num_keys: u32,
    /// Which versions to benchmark.
    pub versions: Vec<Version>,
    /// Number of timed runs per operation.
    pub runs: usize,
    /// Number of warmup runs (not counted).
    pub warmup: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            capacity: 2_097_152, // 2M
            num_keys: 1_000_000,
            versions: vec![Version::V1, Version::V2, Version::V3],
            runs: 10,
            warmup: 3,
        }
    }
}

/// Result of a single benchmark.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub version: Version,
    pub op: String,
    pub num_keys: u32,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub stddev_ms: f64,
    pub mops: f64,
    pub hit_rate: Option<f64>,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:>2} {:>6} | {:>6} keys | {:>8.3} ms median | {:>8.1} Mops | stddev {:.3} ms",
            self.version,
            self.op,
            format_count(self.num_keys),
            self.median_ms,
            self.mops,
            self.stddev_ms,
        )?;
        if let Some(hr) = self.hit_rate {
            write!(f, " | {:.1}% hits", hr * 100.0)?;
        }
        Ok(())
    }
}

fn format_count(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}

/// Compute statistics from a sorted slice of timings.
fn stats(sorted: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    let median = sorted[sorted.len() / 2];
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();
    (mean, median, min, max, stddev)
}

/// Remove outliers using IQR method.
fn remove_outliers(timings: &mut Vec<f64>) {
    if timings.len() < 4 {
        return;
    }
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = timings[timings.len() / 4];
    let q3 = timings[3 * timings.len() / 4];
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    timings.retain(|&x| x >= lower && x <= upper);
}

/// Generate random keys, avoiding the EMPTY_KEY sentinel.
fn gen_keys(rng: &mut impl Rng, n: usize) -> Vec<u32> {
    (0..n)
        .map(|_| {
            let k: u32 = rng.gen();
            if k == EMPTY_KEY {
                0xFFFFFFFE
            } else {
                k
            }
        })
        .collect()
}

/// Generate random values.
fn gen_values(rng: &mut impl Rng, n: usize) -> Vec<u32> {
    (0..n).map(|_| rng.gen()).collect()
}

/// Run benchmarks for all configured versions.
///
/// Returns a vector of results (2 per version: insert + lookup).
pub fn run_benchmarks(config: &BenchConfig) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let mut rng = rand::thread_rng();

    let keys = gen_keys(&mut rng, config.num_keys as usize);
    let values = gen_values(&mut rng, config.num_keys as usize);

    for &version in &config.versions {
        let mut table = GpuHashTable::new(version, config.capacity);

        // ── INSERT benchmark ──
        let mut insert_timings = Vec::with_capacity(config.warmup + config.runs);

        for i in 0..(config.warmup + config.runs) {
            table.clear();
            let ms = table.insert(&keys, &values);
            if i >= config.warmup {
                insert_timings.push(ms);
            }
        }

        remove_outliers(&mut insert_timings);
        insert_timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (mean, median, min, max, stddev) = stats(&insert_timings);
        let mops = config.num_keys as f64 / median / 1000.0;

        results.push(BenchResult {
            version,
            op: "insert".to_string(),
            num_keys: config.num_keys,
            mean_ms: mean,
            median_ms: median,
            min_ms: min,
            max_ms: max,
            stddev_ms: stddev,
            mops,
            hit_rate: None,
        });

        // ── LOOKUP benchmark ──
        // Insert once, then benchmark lookups
        table.clear();
        table.insert(&keys, &values);

        let mut lookup_timings = Vec::with_capacity(config.warmup + config.runs);
        let mut last_hit_rate = 0.0;

        for i in 0..(config.warmup + config.runs) {
            let (results_vec, ms) = table.lookup(&keys);
            if i >= config.warmup {
                lookup_timings.push(ms);
            }
            // Count hits on last run
            if i == config.warmup + config.runs - 1 {
                let hits = results_vec.iter().filter(|&&v| v != EMPTY_KEY).count();
                last_hit_rate = hits as f64 / keys.len() as f64;
            }
        }

        remove_outliers(&mut lookup_timings);
        lookup_timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (mean, median, min, max, stddev) = stats(&lookup_timings);
        let mops = config.num_keys as f64 / median / 1000.0;

        results.push(BenchResult {
            version,
            op: "lookup".to_string(),
            num_keys: config.num_keys,
            mean_ms: mean,
            median_ms: median,
            min_ms: min,
            max_ms: max,
            stddev_ms: stddev,
            mops,
            hit_rate: Some(last_hit_rate),
        });
    }

    results
}

/// Run a quick CPU baseline for comparison.
pub fn cpu_baseline(keys: &[u32], values: &[u32]) -> f64 {
    use std::collections::HashMap;
    use std::time::Instant;

    let start = Instant::now();

    let mut map = HashMap::with_capacity(keys.len());
    for i in 0..keys.len() {
        map.insert(keys[i], values[i]);
    }
    let mut hits = 0u64;
    for &k in keys {
        if map.get(&k).is_some() {
            hits += 1;
        }
    }
    std::hint::black_box(hits);

    start.elapsed().as_secs_f64() * 1000.0
}
