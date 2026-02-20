//! Scan throughput benchmarks: measure CSV, Parquet, and JSON scan rates
//! at practical data sizes (1MB and 10MB) for fast CI.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::TempDir;

use gpu_query::gpu::executor::QueryExecutor;
use gpu_query::io::catalog;

// ============================================================
// Data generation helpers
// ============================================================

/// Generate a CSV string with `n_rows` rows of (id, amount, quantity).
fn generate_csv(n_rows: usize) -> String {
    let mut s = String::with_capacity(n_rows * 30);
    s.push_str("id,amount,quantity\n");
    for i in 0..n_rows {
        s.push_str(&format!(
            "{},{},{}\n",
            i,
            (i * 7 + 13) % 1000,
            (i * 3 + 1) % 50
        ));
    }
    s
}

/// Write CSV data to a file in the given directory.
fn write_csv(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    std::fs::write(&path, content).expect("write csv");
}

/// Generate NDJSON with `n_rows` rows.
fn generate_ndjson(n_rows: usize) -> String {
    let mut s = String::with_capacity(n_rows * 50);
    for i in 0..n_rows {
        s.push_str(&format!(
            r#"{{"id":{},"amount":{},"quantity":{}}}"#,
            i,
            (i * 7 + 13) % 1000,
            (i * 3 + 1) % 50
        ));
        s.push('\n');
    }
    s
}

/// Write NDJSON to file.
fn write_ndjson(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    let mut f = File::create(&path).expect("create ndjson file");
    f.write_all(content.as_bytes()).expect("write ndjson");
    f.flush().expect("flush ndjson");
}

/// Create a Parquet file with `n_rows` rows of (id, amount, quantity).
fn write_parquet(dir: &Path, name: &str, n_rows: usize) {
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;

    let schema_str = "
        message bench_schema {
            REQUIRED INT64 id;
            REQUIRED INT64 amount;
            REQUIRED INT64 quantity;
        }
    ";
    let schema = Arc::new(parse_message_type(schema_str).expect("parse schema"));
    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .build(),
    );

    let path = dir.join(name);
    let file = File::create(&path).expect("create parquet file");
    let mut writer = SerializedFileWriter::new(file, schema, props).expect("create writer");

    // Write in chunks to avoid massive Vec allocations
    let chunk_size = 10_000.min(n_rows);
    let mut written = 0;
    while written < n_rows {
        let batch = chunk_size.min(n_rows - written);
        let ids: Vec<i64> = (written..written + batch).map(|i| i as i64).collect();
        let amounts: Vec<i64> = (written..written + batch)
            .map(|i| ((i * 7 + 13) % 1000) as i64)
            .collect();
        let quantities: Vec<i64> = (written..written + batch)
            .map(|i| ((i * 3 + 1) % 50) as i64)
            .collect();

        let mut rg = writer.next_row_group().expect("next row group");
        {
            let mut col = rg.next_column().expect("next col").unwrap();
            col.typed::<parquet::data_type::Int64Type>()
                .write_batch(&ids, None, None)
                .expect("write ids");
            col.close().expect("close");
        }
        {
            let mut col = rg.next_column().expect("next col").unwrap();
            col.typed::<parquet::data_type::Int64Type>()
                .write_batch(&amounts, None, None)
                .expect("write amounts");
            col.close().expect("close");
        }
        {
            let mut col = rg.next_column().expect("next col").unwrap();
            col.typed::<parquet::data_type::Int64Type>()
                .write_batch(&quantities, None, None)
                .expect("write quantities");
            col.close().expect("close");
        }
        rg.close().expect("close row group");
        written += batch;
    }
    writer.close().expect("close writer");
}

/// Execute a SQL query through the full pipeline.
fn run_query(dir: &Path, sql: &str) -> gpu_query::gpu::executor::QueryResult {
    let catalog = catalog::scan_directory(dir).expect("scan directory");
    let logical = gpu_query::sql::parser::parse_query(sql).expect("parse SQL");
    let optimized = gpu_query::sql::optimizer::optimize(logical);
    let physical = gpu_query::sql::physical_plan::plan(&optimized).expect("plan SQL");
    let mut executor = QueryExecutor::new().expect("create executor");
    executor
        .execute(&physical, &catalog)
        .expect("execute query")
}

// ============================================================
// Benchmarks
// ============================================================

/// Benchmark CSV scan throughput at different data sizes.
fn bench_csv_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("csv_scan");

    // ~1MB CSV: ~35K rows at ~30 bytes/row
    let rows_1mb = 35_000;
    let csv_1mb = generate_csv(rows_1mb);
    let size_1mb = csv_1mb.len() as u64;

    let dir_1mb = TempDir::new().expect("tempdir");
    write_csv(dir_1mb.path(), "bench.csv", &csv_1mb);

    group.throughput(Throughput::Bytes(size_1mb));
    group.bench_with_input(BenchmarkId::new("1MB", size_1mb), &dir_1mb, |b, dir| {
        b.iter(|| {
            run_query(dir.path(), "SELECT count(*) FROM bench");
        });
    });

    // ~10MB CSV: ~350K rows
    let rows_10mb = 350_000;
    let csv_10mb = generate_csv(rows_10mb);
    let size_10mb = csv_10mb.len() as u64;

    let dir_10mb = TempDir::new().expect("tempdir");
    write_csv(dir_10mb.path(), "bench.csv", &csv_10mb);

    group.throughput(Throughput::Bytes(size_10mb));
    group.bench_with_input(BenchmarkId::new("10MB", size_10mb), &dir_10mb, |b, dir| {
        b.iter(|| {
            run_query(dir.path(), "SELECT count(*) FROM bench");
        });
    });

    group.finish();
}

/// Benchmark Parquet scan throughput.
fn bench_parquet_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_scan");

    // ~1MB Parquet: ~40K rows (Parquet is more compact)
    let rows_1mb = 40_000;
    let dir_1mb = TempDir::new().expect("tempdir");
    write_parquet(dir_1mb.path(), "bench.parquet", rows_1mb);
    let file_size = std::fs::metadata(dir_1mb.path().join("bench.parquet"))
        .expect("metadata")
        .len();

    group.throughput(Throughput::Bytes(file_size));
    group.bench_with_input(BenchmarkId::new("1MB", file_size), &dir_1mb, |b, dir| {
        b.iter(|| {
            run_query(dir.path(), "SELECT count(*) FROM bench");
        });
    });

    // ~10MB Parquet
    let rows_10mb = 400_000;
    let dir_10mb = TempDir::new().expect("tempdir");
    write_parquet(dir_10mb.path(), "bench.parquet", rows_10mb);
    let file_size_10mb = std::fs::metadata(dir_10mb.path().join("bench.parquet"))
        .expect("metadata")
        .len();

    group.throughput(Throughput::Bytes(file_size_10mb));
    group.bench_with_input(
        BenchmarkId::new("10MB", file_size_10mb),
        &dir_10mb,
        |b, dir| {
            b.iter(|| {
                run_query(dir.path(), "SELECT count(*) FROM bench");
            });
        },
    );

    group.finish();
}

/// Benchmark JSON (NDJSON) scan throughput.
fn bench_json_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_scan");

    // ~1MB NDJSON: ~20K rows at ~50 bytes/row
    let rows_1mb = 20_000;
    let json_1mb = generate_ndjson(rows_1mb);
    let size_1mb = json_1mb.len() as u64;

    let dir_1mb = TempDir::new().expect("tempdir");
    write_ndjson(dir_1mb.path(), "bench.json", &json_1mb);

    group.throughput(Throughput::Bytes(size_1mb));
    group.bench_with_input(BenchmarkId::new("1MB", size_1mb), &dir_1mb, |b, dir| {
        b.iter(|| {
            run_query(dir.path(), "SELECT count(*) FROM bench");
        });
    });

    // ~10MB NDJSON
    let rows_10mb = 200_000;
    let json_10mb = generate_ndjson(rows_10mb);
    let size_10mb = json_10mb.len() as u64;

    let dir_10mb = TempDir::new().expect("tempdir");
    write_ndjson(dir_10mb.path(), "bench.json", &json_10mb);

    group.throughput(Throughput::Bytes(size_10mb));
    group.bench_with_input(BenchmarkId::new("10MB", size_10mb), &dir_10mb, |b, dir| {
        b.iter(|| {
            run_query(dir.path(), "SELECT count(*) FROM bench");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_csv_scan, bench_parquet_scan, bench_json_scan);
criterion_main!(benches);
