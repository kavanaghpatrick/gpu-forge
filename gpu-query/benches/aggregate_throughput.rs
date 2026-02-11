//! Aggregation throughput benchmarks: COUNT, SUM, MIN, MAX, AVG
//! with and without GROUP BY at various group cardinalities.

use std::path::Path;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

// ============================================================
// Data generation
// ============================================================

/// Generate CSV with `n_rows` rows and a group column cycling 0..n_groups.
fn generate_csv_grouped(n_rows: usize, n_groups: usize) -> String {
    let mut s = String::with_capacity(n_rows * 40);
    s.push_str("id,amount,quantity,grp\n");
    for i in 0..n_rows {
        let grp = i % n_groups;
        s.push_str(&format!(
            "{},{},{},{}\n",
            i,
            (i * 7 + 13) % 10000,
            (i * 3 + 1) % 50,
            grp
        ));
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

/// Benchmark individual aggregate functions on 100K rows (no GROUP BY).
fn bench_aggregate_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregate_functions");
    group.sample_size(20);

    let n_rows = 100_000;
    let csv = generate_csv_grouped(n_rows, 100);

    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "data.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    let queries = [
        ("count", "SELECT count(*) FROM data"),
        ("sum", "SELECT sum(amount) FROM data"),
        ("min", "SELECT min(amount) FROM data"),
        ("max", "SELECT max(amount) FROM data"),
        ("avg", "SELECT avg(amount) FROM data"),
        (
            "count_sum_min_max",
            "SELECT count(*), sum(amount), min(amount), max(amount) FROM data",
        ),
    ];

    for (name, sql) in &queries {
        group.bench_with_input(BenchmarkId::new(*name, n_rows), &dir, |b, dir| {
            b.iter(|| {
                run_query(dir.path(), sql);
            });
        });
    }

    group.finish();
}

/// Benchmark filtered aggregation (filter + aggregate pipeline).
fn bench_filtered_aggregate(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtered_aggregate");
    group.sample_size(20);

    let n_rows = 100_000;
    let csv = generate_csv_grouped(n_rows, 100);

    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "data.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    group.bench_with_input(
        BenchmarkId::new("count_where", n_rows),
        &dir,
        |b, dir| {
            b.iter(|| {
                run_query(
                    dir.path(),
                    "SELECT count(*) FROM data WHERE amount > 5000",
                );
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("sum_where", n_rows),
        &dir,
        |b, dir| {
            b.iter(|| {
                run_query(
                    dir.path(),
                    "SELECT sum(amount) FROM data WHERE amount > 5000",
                );
            });
        },
    );

    group.finish();
}

/// Benchmark GROUP BY at various group cardinalities (10, 100, 1000 groups).
fn bench_group_by(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_by");
    group.sample_size(20);

    let n_rows = 100_000;

    let group_counts = [10, 100, 1000];
    for &n_groups in &group_counts {
        let csv = generate_csv_grouped(n_rows, n_groups);
        let dir = TempDir::new().expect("tempdir");
        write_csv(dir.path(), "data.csv", &csv);

        group.throughput(Throughput::Elements(n_rows as u64));

        group.bench_with_input(
            BenchmarkId::new("count_groups", n_groups),
            &dir,
            |b, dir| {
                b.iter(|| {
                    run_query(dir.path(), "SELECT grp, count(*) FROM data GROUP BY grp");
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sum_groups", n_groups),
            &dir,
            |b, dir| {
                b.iter(|| {
                    run_query(
                        dir.path(),
                        "SELECT grp, sum(amount) FROM data GROUP BY grp",
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_aggregate_functions,
    bench_filtered_aggregate,
    bench_group_by
);
criterion_main!(benches);
