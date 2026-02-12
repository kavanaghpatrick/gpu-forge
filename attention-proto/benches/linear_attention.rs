use criterion::{criterion_group, criterion_main, Criterion};

fn bench_linear_attention(_c: &mut Criterion) {
    // TODO: Proto 6 — FLA linear attention benchmark
}

criterion_group!(benches, bench_linear_attention);
criterion_main!(benches);
