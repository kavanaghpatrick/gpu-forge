//! Criterion benchmark for Proto 8: Burn extension trait dispatch overhead.
//!
//! Measures the overhead of routing attention through Burn tensors (NdArray backend)
//! vs calling Proto 1 directly with raw f32 slices. The difference isolates the
//! tensor data extraction (into_data + to_vec) and re-wrapping (TensorData::new +
//! Tensor::from_data) cost — the "bridge overhead".
//!
//! Feature-gated behind `burn-ext`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::{Duration, Instant};

use attention_proto::device::GpuDevice;
use attention_proto::proto1_flash::run_flash_attention;
use attention_proto::proto8_burn::metal_flash_attention_bridge;

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

/// Generate deterministic pseudo-random f32 data via LCG.
fn lcg_data(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn bench_burn_extension(c: &mut Criterion) {
    let gpu = GpuDevice::shared();
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let head_dim: usize = 64;

    let mut group = c.benchmark_group("burn_extension");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(3));

    for &seq_len in &[64usize, 128, 256, 512] {
        let n_elements = seq_len * head_dim;

        // Pre-generate data
        let q_data = lcg_data(n_elements, 42);
        let k_data = lcg_data(n_elements, 137);
        let v_data = lcg_data(n_elements, 256);

        // --- Benchmark 1: Direct Proto 1 (raw f32 slices -> Metal -> f32 slices) ---
        group.bench_with_input(
            BenchmarkId::new(format!("direct_N={seq_len}"), seq_len),
            &seq_len,
            |b, &seq_len| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _output =
                            run_flash_attention(gpu, &q_data, &k_data, &v_data, seq_len, head_dim);
                    }
                    start.elapsed()
                });
            },
        );

        // --- Benchmark 2: Bridge (Burn tensor -> Metal -> Burn tensor) ---
        group.bench_with_input(
            BenchmarkId::new(format!("bridge_N={seq_len}"), seq_len),
            &seq_len,
            |b, _| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        // Recreate tensors each iteration (tensor data is consumed by into_data)
                        let q_t: Tensor<NdArray, 3> = Tensor::from_data(
                            TensorData::new(q_data.clone(), [1, seq_len, head_dim]),
                            &device,
                        );
                        let k_t: Tensor<NdArray, 3> = Tensor::from_data(
                            TensorData::new(k_data.clone(), [1, seq_len, head_dim]),
                            &device,
                        );
                        let v_t: Tensor<NdArray, 3> = Tensor::from_data(
                            TensorData::new(v_data.clone(), [1, seq_len, head_dim]),
                            &device,
                        );
                        let _output = metal_flash_attention_bridge::<NdArray>(q_t, k_t, v_t, None);
                    }
                    start.elapsed()
                });
            },
        );
    }

    group.finish();

    // --- Post-benchmark: estimate bridge overhead ---
    eprintln!();
    eprintln!("=== Burn Bridge Overhead Estimate ===");
    eprintln!(
        "{:>6} {:>12} {:>12} {:>12}",
        "N", "Direct", "Bridge", "Overhead"
    );
    eprintln!("{:-<50}", "");

    for &seq_len in &[64usize, 128, 256, 512] {
        let n_elements = seq_len * head_dim;
        let q_data = lcg_data(n_elements, 42);
        let k_data = lcg_data(n_elements, 137);
        let v_data = lcg_data(n_elements, 256);

        let num_samples = 20;

        // Measure direct
        let mut direct_times = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            let start = Instant::now();
            let _o = run_flash_attention(gpu, &q_data, &k_data, &v_data, seq_len, head_dim);
            direct_times.push(start.elapsed());
        }

        // Measure bridge
        let mut bridge_times = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            let q_t: Tensor<NdArray, 3> = Tensor::from_data(
                TensorData::new(q_data.clone(), [1, seq_len, head_dim]),
                &device,
            );
            let k_t: Tensor<NdArray, 3> = Tensor::from_data(
                TensorData::new(k_data.clone(), [1, seq_len, head_dim]),
                &device,
            );
            let v_t: Tensor<NdArray, 3> = Tensor::from_data(
                TensorData::new(v_data.clone(), [1, seq_len, head_dim]),
                &device,
            );
            let start = Instant::now();
            let _o = metal_flash_attention_bridge::<NdArray>(q_t, k_t, v_t, None);
            bridge_times.push(start.elapsed());
        }

        let mean_direct =
            direct_times.iter().map(|d| d.as_nanos()).sum::<u128>() / num_samples as u128;
        let mean_bridge =
            bridge_times.iter().map(|d| d.as_nanos()).sum::<u128>() / num_samples as u128;
        let overhead_ns = mean_bridge.saturating_sub(mean_direct);

        eprintln!(
            "{:>6} {:>10.1}us {:>10.1}us {:>10.1}us",
            seq_len,
            mean_direct as f64 / 1000.0,
            mean_bridge as f64 / 1000.0,
            overhead_ns as f64 / 1000.0,
        );
    }
    eprintln!("{:-<50}", "");
    eprintln!("Overhead = bridge - direct (tensor extraction + re-wrapping cost)");
}

criterion_group!(benches, bench_burn_extension);
criterion_main!(benches);
