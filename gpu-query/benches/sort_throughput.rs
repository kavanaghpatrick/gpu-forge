//! Sort throughput benchmarks: measure ORDER BY performance at 1K, 10K, 100K rows.

use std::path::Path;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

// ============================================================
// Data generation
// ============================================================

/// Generate CSV with `n_rows` rows where amount has pseudo-random distribution.
fn generate_csv(n_rows: usize) -> String {
    let mut s = String::with_capacity(n_rows * 30);
    s.push_str("id,amount,quantity\n");
    for i in 0..n_rows {
        // Pseudo-random via LCG-like formula for non-trivial sort
        let amount = ((i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 33)
            % 100000;
        s.push_str(&format!("{},{},{}\n", i, amount, (i * 3 + 1) % 50));
    }
    s
}

fn write_csv(dir: &Path, name: &str, content: &str) {
    std::fs::write(dir.join(name), content).expect("write csv");
}

fn run_query(dir: &Path, sql: &str) -> gpu_query::gpu::executor::QueryResult {
    let catalog = catalog::scan_directory(dir).expect("scan dir");
    let logical = gpu_query::sql::parser::parse_query(sql).expect("parse");
    let optimized = gpu_query::sql::optimizer::optimize(logical);
    let physical = gpu_query::sql::physical_plan::plan(&optimized).expect("plan");
    let mut executor = QueryExecutor::new().expect("executor");
    executor.execute(&physical, &catalog).expect("execute")
}

// ============================================================
// Benchmarks
// ============================================================

/// Benchmark ORDER BY ASC at various row counts.
fn bench_sort_asc(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_asc");
    group.sample_size(20);

    let sizes = [1_000, 10_000, 100_000];
    for &n_rows in &sizes {
        let csv = generate_csv(n_rows);
        let dir = TempDir::new().expect("tempdir");
        write_csv(dir.path(), "data.csv", &csv);

        group.throughput(Throughput::Elements(n_rows as u64));

        group.bench_with_input(BenchmarkId::new("rows", n_rows), &dir, |b, dir| {
            b.iter(|| {
                run_query(dir.path(), "SELECT id, amount FROM data ORDER BY amount");
            });
        });
    }

    group.finish();
}

/// Benchmark ORDER BY DESC at various row counts.
fn bench_sort_desc(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_desc");
    group.sample_size(20);

    let sizes = [1_000, 10_000, 100_000];
    for &n_rows in &sizes {
        let csv = generate_csv(n_rows);
        let dir = TempDir::new().expect("tempdir");
        write_csv(dir.path(), "data.csv", &csv);

        group.throughput(Throughput::Elements(n_rows as u64));

        group.bench_with_input(BenchmarkId::new("rows", n_rows), &dir, |b, dir| {
            b.iter(|| {
                run_query(
                    dir.path(),
                    "SELECT id, amount FROM data ORDER BY amount DESC",
                );
            });
        });
    }

    group.finish();
}

/// Benchmark ORDER BY with LIMIT (top-N query pattern).
fn bench_sort_with_limit(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_limit");
    group.sample_size(20);

    let n_rows = 100_000;
    let csv = generate_csv(n_rows);
    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "data.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    let limits = [10, 100, 1000];
    for &limit in &limits {
        group.bench_with_input(BenchmarkId::new("top_n", limit), &dir, |b, dir| {
            let sql = format!(
                "SELECT id, amount FROM data ORDER BY amount DESC LIMIT {}",
                limit
            );
            b.iter(|| {
                run_query(dir.path(), &sql);
            });
        });
    }

    group.finish();
}

/// Benchmark filtered sort (WHERE + ORDER BY).
fn bench_filtered_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_filtered");
    group.sample_size(20);

    let n_rows = 100_000;
    let csv = generate_csv(n_rows);
    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "data.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    group.bench_with_input(BenchmarkId::new("where_order", n_rows), &dir, |b, dir| {
        b.iter(|| {
            run_query(
                dir.path(),
                "SELECT id, amount FROM data WHERE amount > 50000 ORDER BY amount",
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sort_asc,
    bench_sort_desc,
    bench_sort_with_limit,
    bench_filtered_sort
);
criterion_main!(benches);
