use criterion::{criterion_group, criterion_main, Criterion};

fn bench_variant_overhead(_c: &mut Criterion) {
    // TODO: Proto 7 — RoPE/ALiBi/GQA variant overhead benchmark
}

criterion_group!(benches, bench_variant_overhead);
criterion_main!(benches);
