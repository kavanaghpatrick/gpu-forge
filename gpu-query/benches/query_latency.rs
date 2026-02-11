//! End-to-end query latency benchmarks: representative SQL queries
//! measuring full pipeline latency (parse -> optimize -> plan -> GPU execute).
//! Includes TPC-H Q1 adapted for gpu-query.

use std::path::Path;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

// ============================================================
// Data generation
// ============================================================

/// Generate a "lineitem" CSV adapted from TPC-H for gpu-query.
/// Columns: orderkey, quantity, extendedprice, discount, tax, shipdate_epoch, returnflag
fn generate_lineitem(n_rows: usize) -> String {
    let mut s = String::with_capacity(n_rows * 60);
    s.push_str("orderkey,quantity,extendedprice,discount,tax,returnflag\n");
    for i in 0..n_rows {
        let orderkey = i + 1;
        let quantity = (i % 50) + 1;
        let price = ((i * 7 + 13) % 100000) + 100;
        let discount = i % 10; // 0-9 representing 0-9%
        let tax = i % 8; // 0-7 representing 0-7%
                         // returnflag cycles 0,1,2 (representing A, N, R)
        let returnflag = i % 3;
        s.push_str(&format!(
            "{},{},{},{},{},{}\n",
            orderkey, quantity, price, discount, tax, returnflag
        ));
    }
    s
}

/// Generate simple sales CSV.
fn generate_sales(n_rows: usize) -> String {
    let mut s = String::with_capacity(n_rows * 30);
    s.push_str("id,amount,region\n");
    for i in 0..n_rows {
        let amount = (i * 7 + 13) % 1000;
        let region = i % 5; // 0-4 representing regions
        s.push_str(&format!("{},{},{}\n", i, amount, region));
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

/// TPC-H Q1 adapted for gpu-query: aggregate pricing summary.
/// Original Q1: SELECT returnflag, linestatus, sum(quantity), sum(extendedprice),
///   sum(extendedprice * (1 - discount)), ... FROM lineitem
///   WHERE shipdate <= '1998-12-01' - interval '90' day GROUP BY returnflag, linestatus
///
/// Adapted (no date/string support in filter, simplified):
///   SELECT returnflag, count(*), sum(quantity), sum(extendedprice)
///   FROM lineitem WHERE discount < 5 GROUP BY returnflag
fn bench_tpch_q1(c: &mut Criterion) {
    let mut group = c.benchmark_group("tpch_q1");
    group.sample_size(20);

    let n_rows = 100_000;
    let csv = generate_lineitem(n_rows);
    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "lineitem.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    group.bench_with_input(
        BenchmarkId::new("100K_rows", n_rows),
        &dir,
        |b, dir| {
            b.iter(|| {
                run_query(
                    dir.path(),
                    "SELECT returnflag, count(*), sum(quantity), sum(extendedprice) FROM lineitem WHERE discount < 5 GROUP BY returnflag",
                );
            });
        },
    );

    group.finish();
}

/// Simple point query: SELECT count(*) WHERE.
fn bench_point_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_query");
    group.sample_size(20);

    let sizes = [10_000, 100_000];
    for &n_rows in &sizes {
        let csv = generate_sales(n_rows);
        let dir = TempDir::new().expect("tempdir");
        write_csv(dir.path(), "sales.csv", &csv);

        group.throughput(Throughput::Elements(n_rows as u64));

        group.bench_with_input(BenchmarkId::new("count_where", n_rows), &dir, |b, dir| {
            b.iter(|| {
                run_query(dir.path(), "SELECT count(*) FROM sales WHERE amount > 500");
            });
        });
    }

    group.finish();
}

/// Aggregate query: SELECT sum, count WHERE.
fn bench_aggregate_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregate_query");
    group.sample_size(20);

    let n_rows = 100_000;
    let csv = generate_sales(n_rows);
    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "sales.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    group.bench_with_input(
        BenchmarkId::new("sum_count_where", n_rows),
        &dir,
        |b, dir| {
            b.iter(|| {
                run_query(
                    dir.path(),
                    "SELECT count(*), sum(amount) FROM sales WHERE amount > 500",
                );
            });
        },
    );

    group.bench_with_input(BenchmarkId::new("min_max_avg", n_rows), &dir, |b, dir| {
        b.iter(|| {
            run_query(
                dir.path(),
                "SELECT min(amount), max(amount), avg(amount) FROM sales",
            );
        });
    });

    group.finish();
}

/// GROUP BY query latency.
fn bench_group_by_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_by_query");
    group.sample_size(20);

    let n_rows = 100_000;
    let csv = generate_sales(n_rows);
    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "sales.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    group.bench_with_input(BenchmarkId::new("group_sum", n_rows), &dir, |b, dir| {
        b.iter(|| {
            run_query(
                dir.path(),
                "SELECT region, sum(amount) FROM sales GROUP BY region",
            );
        });
    });

    group.bench_with_input(
        BenchmarkId::new("group_count_sum", n_rows),
        &dir,
        |b, dir| {
            b.iter(|| {
                run_query(
                    dir.path(),
                    "SELECT region, count(*), sum(amount) FROM sales GROUP BY region",
                );
            });
        },
    );

    group.finish();
}

/// Full pipeline: filter + aggregate + sort.
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.sample_size(20);

    let n_rows = 100_000;
    let csv = generate_sales(n_rows);
    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "sales.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    group.bench_with_input(
        BenchmarkId::new("filter_agg_sort", n_rows),
        &dir,
        |b, dir| {
            b.iter(|| {
                run_query(
                    dir.path(),
                    "SELECT region, count(*), sum(amount) FROM sales WHERE amount > 200 GROUP BY region",
                );
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_tpch_q1,
    bench_point_query,
    bench_aggregate_query,
    bench_group_by_query,
    bench_full_pipeline
);
criterion_main!(benches);
