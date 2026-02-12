use criterion::{criterion_group, criterion_main, Criterion};

fn bench_burn_extension(_c: &mut Criterion) {
    // TODO: Proto 8 — Burn extension trait dispatch overhead benchmark
}

criterion_group!(benches, bench_burn_extension);
criterion_main!(benches);
