use criterion::{criterion_group, criterion_main};

fn placeholder(_c: &mut criterion::Criterion) {
    // Benchmarks will be added in Task 4.2
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
