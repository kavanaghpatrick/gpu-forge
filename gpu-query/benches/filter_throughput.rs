//! Filter throughput benchmarks: measure GPU filter kernel performance
//! at various selectivities (10%, 50%, 90%) on 100K rows.

use std::path::Path;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

// ============================================================
// Data generation
// ============================================================

/// Generate CSV with `n_rows` rows where amount is uniformly distributed 0..999.
fn generate_csv(n_rows: usize) -> String {
    let mut s = String::with_capacity(n_rows * 30);
    s.push_str("id,amount,quantity\n");
    for i in 0..n_rows {
        // amount cycles 0..999 for predictable selectivity
        let amount = i % 1000;
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

/// Filter at ~10% selectivity (amount > 899 → 100 out of 1000 match).
/// Filter at ~50% selectivity (amount > 499 → 500 out of 1000 match).
/// Filter at ~90% selectivity (amount > 99 → 900 out of 1000 match).
fn bench_filter_selectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_selectivity");
    group.sample_size(20); // fewer samples for GPU benchmarks

    let n_rows = 100_000;
    let csv = generate_csv(n_rows);
    let _csv_size = csv.len() as u64;

    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "data.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    // 10% selectivity: amount > 899
    group.bench_with_input(
        BenchmarkId::new("selectivity_10pct", n_rows),
        &dir,
        |b, dir| {
            b.iter(|| {
                run_query(dir.path(), "SELECT count(*) FROM data WHERE amount > 899");
            });
        },
    );

    // 50% selectivity: amount > 499
    group.bench_with_input(
        BenchmarkId::new("selectivity_50pct", n_rows),
        &dir,
        |b, dir| {
            b.iter(|| {
                run_query(dir.path(), "SELECT count(*) FROM data WHERE amount > 499");
            });
        },
    );

    // 90% selectivity: amount > 99
    group.bench_with_input(
        BenchmarkId::new("selectivity_90pct", n_rows),
        &dir,
        |b, dir| {
            b.iter(|| {
                run_query(dir.path(), "SELECT count(*) FROM data WHERE amount > 99");
            });
        },
    );

    group.finish();
}

/// Filter with compound predicates (AND/OR).
fn bench_filter_compound(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_compound");
    group.sample_size(20);

    let n_rows = 100_000;
    let csv = generate_csv(n_rows);

    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "data.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    // AND predicate
    group.bench_with_input(BenchmarkId::new("and_predicate", n_rows), &dir, |b, dir| {
        b.iter(|| {
            run_query(
                dir.path(),
                "SELECT count(*) FROM data WHERE amount > 200 AND amount < 800",
            );
        });
    });

    // OR predicate
    group.bench_with_input(BenchmarkId::new("or_predicate", n_rows), &dir, |b, dir| {
        b.iter(|| {
            run_query(
                dir.path(),
                "SELECT count(*) FROM data WHERE amount < 100 OR amount > 900",
            );
        });
    });

    group.finish();
}

/// Filter with different comparison operators.
fn bench_filter_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_operators");
    group.sample_size(20);

    let n_rows = 100_000;
    let csv = generate_csv(n_rows);

    let dir = TempDir::new().expect("tempdir");
    write_csv(dir.path(), "data.csv", &csv);

    group.throughput(Throughput::Elements(n_rows as u64));

    let ops = [
        ("gt", "SELECT count(*) FROM data WHERE amount > 500"),
        ("lt", "SELECT count(*) FROM data WHERE amount < 500"),
        ("eq", "SELECT count(*) FROM data WHERE amount = 500"),
        ("ge", "SELECT count(*) FROM data WHERE amount >= 500"),
        ("le", "SELECT count(*) FROM data WHERE amount <= 500"),
        ("ne", "SELECT count(*) FROM data WHERE amount != 500"),
    ];

    for (name, sql) in &ops {
        group.bench_with_input(BenchmarkId::new(*name, n_rows), &dir, |b, dir| {
            b.iter(|| {
                run_query(dir.path(), sql);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_filter_selectivity,
    bench_filter_compound,
    bench_filter_operators
);
criterion_main!(benches);
