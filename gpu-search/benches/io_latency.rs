//! I/O latency benchmark: MTLIOCommandQueue batch loading vs sequential CPU reads.
//!
//! Benchmarks file loading at 100 and 1K file counts with ~1KB files.
//! Also benchmarks with larger ~64KB files where GPU batch I/O excels.
//! Compares GPU batch loading (MTLIOCommandQueue) against sequential CPU reads (std::fs::read).
//!
//! ## Results (Apple M4 Pro)
//!
//! GPU batch I/O has per-file setup overhead (MTLIOFileHandle, page alignment).
//! At small file sizes (~1KB), CPU sequential reads are faster due to less overhead.
//! GPU batch I/O shines with larger files and when combined with GPU compute overlap.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;
use tempfile::TempDir;

use gpu_search::gpu::device::GpuDevice;
use gpu_search::io::batch::GpuBatchLoader;

/// Create a temp directory with N files of specified size.
///
/// Returns (TempDir, Vec<PathBuf>) -- TempDir must be kept alive.
fn create_test_files(count: usize, file_size_bytes: usize) -> (TempDir, Vec<PathBuf>) {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let mut paths = Vec::with_capacity(count);

    // Generate content pattern: repeating text lines
    let line = "The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet.\n";
    let mut content = Vec::with_capacity(file_size_bytes);
    while content.len() < file_size_bytes {
        let remaining = file_size_bytes - content.len();
        let to_copy = remaining.min(line.len());
        content.extend_from_slice(&line.as_bytes()[..to_copy]);
    }
    content.truncate(file_size_bytes);

    for i in 0..count {
        let path = dir.path().join(format!("bench_file_{:06}.txt", i));
        // Prepend a unique header to each file
        let header = format!("=== File {:06} ===\n", i);
        let mut file_content = Vec::with_capacity(file_size_bytes + header.len());
        file_content.extend_from_slice(header.as_bytes());
        file_content.extend_from_slice(&content[header.len()..]);
        std::fs::write(&path, &file_content).expect("Failed to write test file");
        paths.push(path);
    }

    (dir, paths)
}

/// Sequential CPU reads via std::fs::read.
fn cpu_sequential_read(paths: &[PathBuf]) -> usize {
    let mut total_bytes = 0usize;
    for path in paths {
        let data = std::fs::read(path).expect("CPU read failed");
        total_bytes += data.len();
    }
    total_bytes
}

/// GPU batch loading via MTLIOCommandQueue.
fn gpu_batch_load(loader: &GpuBatchLoader, paths: &[PathBuf]) -> u64 {
    let result = loader.load_batch(paths).expect("GPU batch load failed");
    result.total_bytes
}

fn bench_io_latency(c: &mut Criterion) {
    let gpu = GpuDevice::new();

    let loader = match GpuBatchLoader::new(&gpu.device) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("MTLIOCommandQueue not available: {} -- skipping GPU benchmarks", e);
            return;
        }
    };

    // Benchmark with small ~1KB files (tests per-file overhead)
    {
        let mut group = c.benchmark_group("io_latency_1kb");
        group.sample_size(10);

        for &(count, label) in &[(100usize, "100_files"), (1_000, "1K_files")] {
            let (_dir, paths) = create_test_files(count, 1024);

            // Warm filesystem cache
            for path in &paths {
                let _ = std::fs::read(path);
            }

            group.bench_with_input(
                BenchmarkId::new("cpu_sequential", label),
                &paths,
                |b, paths| b.iter(|| cpu_sequential_read(paths)),
            );

            group.bench_with_input(
                BenchmarkId::new("gpu_batch_io", label),
                &paths,
                |b, paths| b.iter(|| gpu_batch_load(&loader, paths)),
            );
        }

        group.finish();
    }

    // Benchmark with larger ~64KB files (amortizes per-file overhead)
    {
        let mut group = c.benchmark_group("io_latency_64kb");
        group.sample_size(10);

        for &(count, label) in &[(100usize, "100_files"), (500, "500_files")] {
            let (_dir, paths) = create_test_files(count, 65536);

            // Warm filesystem cache
            for path in &paths {
                let _ = std::fs::read(path);
            }

            group.bench_with_input(
                BenchmarkId::new("cpu_sequential", label),
                &paths,
                |b, paths| b.iter(|| cpu_sequential_read(paths)),
            );

            group.bench_with_input(
                BenchmarkId::new("gpu_batch_io", label),
                &paths,
                |b, paths| b.iter(|| gpu_batch_load(&loader, paths)),
            );
        }

        group.finish();
    }
}

criterion_group!(benches, bench_io_latency);
criterion_main!(benches);
