use criterion::{criterion_group, criterion_main, Criterion};

fn bench_function_stitch(_c: &mut Criterion) {
    // TODO: Proto 2 — Function stitching overhead benchmark
}

criterion_group!(benches, bench_function_stitch);
criterion_main!(benches);
