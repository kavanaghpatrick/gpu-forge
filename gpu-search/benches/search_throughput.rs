//! Search throughput benchmark: raw GPU content search at 1MB, 10MB, 100MB.
//!
//! Reports throughput in GB/s. Target: 55-80 GB/s on Apple M4 Pro.
//! Sample size: 20 per benchmark to keep runtime reasonable.

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::content::{ContentSearchEngine, SearchMode, SearchOptions};

/// Generate synthetic data with embedded search patterns.
///
/// Creates repeating text lines (~80 chars each) with the pattern "SEARCHME"
/// embedded roughly every 1000 bytes, so we get realistic match density
/// without saturating the MAX_MATCHES buffer.
fn generate_synthetic_data(size_bytes: usize) -> Vec<u8> {
    // Base line: 78 chars + newline = 79 bytes
    let base_line = "The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet__\n";
    // Line with embedded pattern: same length
    let match_line = "The quick brown fox SEARCHME over the lazy dog. Lorem ipsum dolor sit__\n";

    let base_len = base_line.len();
    let mut data = Vec::with_capacity(size_bytes);
    let mut line_count = 0u64;

    while data.len() < size_bytes {
        // Insert a match line roughly every 13 lines (~1000 bytes)
        if line_count % 13 == 7 {
            let remaining = size_bytes - data.len();
            let to_copy = remaining.min(match_line.len());
            data.extend_from_slice(&match_line.as_bytes()[..to_copy]);
        } else {
            let remaining = size_bytes - data.len();
            let to_copy = remaining.min(base_len);
            data.extend_from_slice(&base_line.as_bytes()[..to_copy]);
        }
        line_count += 1;
    }

    data.truncate(size_bytes);
    data
}

fn bench_search_throughput(c: &mut Criterion) {
    // Initialize GPU once -- reused across all benchmarks
    let gpu = GpuDevice::new();
    let pso_cache = PsoCache::new(&gpu.device);

    let sizes: &[(usize, &str)] = &[
        (1 * 1024 * 1024, "1MB"),
        (10 * 1024 * 1024, "10MB"),
        (100 * 1024 * 1024, "100MB"),
    ];

    let pattern = b"SEARCHME";
    let options = SearchOptions {
        case_sensitive: true,
        max_results: 10000,
        mode: SearchMode::Turbo, // Max throughput mode
    };

    let mut group = c.benchmark_group("search_throughput");
    group.sample_size(20);

    for &(size, label) in sizes {
        let data = generate_synthetic_data(size);

        // max_files needs to support enough chunks: size/4096 chunks, max_chunks = max_files*10
        // 100MB = 25600 chunks -> max_files >= 2560
        let max_files = (size / 4096 / 10).max(100) + 100;

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("gpu_search", label), &data, |b, data| {
            // Create engine once per benchmark (not per iteration)
            let mut engine = ContentSearchEngine::new(&gpu.device, &pso_cache, max_files);

            b.iter(|| {
                engine.reset();
                engine.load_content(data, 0);
                let results = engine.search(pattern, &options);
                // Return results to prevent optimization
                results.len()
            });
        });
    }

    group.finish();
}

/// Ripgrep comparison benchmark.
///
/// Generates a ~100MB synthetic corpus as a temp file, then benchmarks:
/// 1. gpu-search raw content search (in-memory, GPU compute)
/// 2. ripgrep (rg) via subprocess on the same file (disk I/O + CPU SIMD)
///
/// NOTE: This is NOT an apples-to-apples comparison:
/// - gpu-search operates on in-memory data (no disk I/O)
/// - ripgrep reads from disk (though OS page cache likely warm after first run)
/// - gpu-search uses Metal GPU compute kernels
/// - ripgrep uses CPU SIMD (AVX2/NEON) with mmap
///
/// The comparison shows the raw throughput advantage of GPU parallelism
/// over highly-optimized CPU search. Target: gpu-search 4-7x faster.
fn bench_vs_ripgrep(c: &mut Criterion) {
    use std::io::Write;
    use std::process::Command;

    const CORPUS_SIZE: usize = 100 * 1024 * 1024; // 100MB
    let pattern = b"SEARCHME";
    let pattern_str = "SEARCHME";

    // Check if rg is available
    let rg_available = Command::new("rg")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    // Generate corpus
    let data = generate_synthetic_data(CORPUS_SIZE);

    // Write corpus to temp file for ripgrep
    let tmp_dir = tempfile::tempdir().expect("create temp dir");
    let corpus_path = tmp_dir.path().join("corpus.txt");
    {
        let mut f = std::fs::File::create(&corpus_path).expect("create corpus file");
        f.write_all(&data).expect("write corpus");
        f.flush().expect("flush corpus");
    }

    // Warm the OS page cache by reading once
    if rg_available {
        let _ = Command::new("rg")
            .arg("--count")
            .arg(pattern_str)
            .arg(&corpus_path)
            .output();
    }

    let mut group = c.benchmark_group("vs_ripgrep");
    group.sample_size(20);
    group.throughput(criterion::Throughput::Bytes(CORPUS_SIZE as u64));

    // GPU search benchmark
    {
        let gpu = GpuDevice::new();
        let pso_cache = PsoCache::new(&gpu.device);
        // 100MB = 25600 chunks -> max_files >= 2560
        let max_files = (CORPUS_SIZE / 4096 / 10).max(100) + 100;

        let options = SearchOptions {
            case_sensitive: true,
            max_results: 10000,
            mode: SearchMode::Turbo,
        };

        group.bench_function("gpu_search_100MB", |b| {
            let mut engine = ContentSearchEngine::new(&gpu.device, &pso_cache, max_files);
            b.iter(|| {
                engine.reset();
                engine.load_content(&data, 0);
                let results = engine.search(pattern, &options);
                results.len()
            });
        });
    }

    // Ripgrep benchmark (only if rg is available)
    if rg_available {
        let corpus_path_str = corpus_path.to_str().unwrap().to_string();
        group.bench_function("ripgrep_100MB", |b| {
            b.iter(|| {
                let output = Command::new("rg")
                    .arg("--count")
                    .arg("--no-filename")
                    .arg(pattern_str)
                    .arg(&corpus_path_str)
                    .output()
                    .expect("rg failed");
                // Parse match count to prevent optimization
                let count_str = String::from_utf8_lossy(&output.stdout);
                count_str.trim().parse::<u64>().unwrap_or(0)
            });
        });
    } else {
        eprintln!("NOTE: rg (ripgrep) not available -- skipping ripgrep benchmark");
        eprintln!("      Install with: brew install ripgrep");
    }

    group.finish();
}

criterion_group!(benches, bench_search_throughput, bench_vs_ripgrep);
criterion_main!(benches);
