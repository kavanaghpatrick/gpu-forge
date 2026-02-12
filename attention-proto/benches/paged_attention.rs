use criterion::{criterion_group, criterion_main, Criterion};

fn bench_paged_attention(_c: &mut Criterion) {
    // TODO: Proto 3 — PagedAttention V2 throughput benchmark
}

criterion_group!(benches, bench_paged_attention);
criterion_main!(benches);
