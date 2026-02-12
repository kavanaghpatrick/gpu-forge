use criterion::{criterion_group, criterion_main, Criterion};

fn bench_constant_overhead(_c: &mut Criterion) {
    // TODO: Proto 4 — Function constant compilation overhead benchmark
}

criterion_group!(benches, bench_constant_overhead);
criterion_main!(benches);
