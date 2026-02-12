use criterion::{criterion_group, criterion_main, Criterion};

fn bench_cubecl_comparison(_c: &mut Criterion) {
    // TODO: Proto 5 — CubeCL vs hand-written MSL benchmark
}

criterion_group!(benches, bench_cubecl_comparison);
criterion_main!(benches);
