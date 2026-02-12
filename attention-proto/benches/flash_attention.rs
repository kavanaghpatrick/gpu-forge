use criterion::{criterion_group, criterion_main, Criterion};

fn bench_flash_attention(_c: &mut Criterion) {
    // TODO: Proto 1 — Flash Attention tile size sweep benchmark
}

criterion_group!(benches, bench_flash_attention);
criterion_main!(benches);
