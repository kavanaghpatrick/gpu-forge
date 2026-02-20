use forge_sort::GpuSorter;
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;

const WARMUP: usize = 3;
const RUNS: usize = 10;

fn gen_random_u32(n: usize) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}

fn gen_random_i32(n: usize) -> Vec<i32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}

fn gen_random_f32(n: usize) -> Vec<f32> {
    // Uniform random bit patterns (not narrow-range).
    // Narrow-range floats (e.g. -1e6..1e6) cluster in ~20 MSD bins,
    // causing pathological bucket imbalance for MSD radix sort.
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| loop {
            let v = f32::from_bits(rng.gen::<u32>());
            if v.is_finite() {
                return v;
            }
        })
        .collect()
}

fn gen_random_u64(n: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}

fn gen_random_i64(n: usize) -> Vec<i64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}

fn gen_random_f64(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| loop {
            let v = f64::from_bits(rng.gen::<u64>());
            if v.is_finite() {
                return v;
            }
        })
        .collect()
}

fn percentile(times: &[f64], p: f64) -> f64 {
    let idx = (p / 100.0 * (times.len() - 1) as f64).round() as usize;
    times[idx.min(times.len() - 1)]
}

fn size_str(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else {
        format!("{}K", n / 1_000)
    }
}

/// Benchmark a sort that operates on &mut [T]. Clones data before each run,
/// only times the sort_fn call (not the clone).
fn bench_typed<T: Clone, F: FnMut(&mut [T])>(data: &[T], mut sort_fn: F) -> (f64, f64, f64) {
    let mut copy = data.to_vec();
    let mut times = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        copy.clear();
        copy.extend_from_slice(data);
        let start = Instant::now();
        sort_fn(&mut copy);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if i >= WARMUP {
            times.push(ms);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile(&times, 50.0), percentile(&times, 5.0), percentile(&times, 95.0))
}

/// Benchmark a function that takes no mutable input (e.g. argsort, which reads &[T]).
fn bench_fn<F: FnMut()>(mut f: F) -> (f64, f64, f64) {
    let mut times = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let start = Instant::now();
        f();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if i >= WARMUP {
            times.push(ms);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile(&times, 50.0), percentile(&times, 5.0), percentile(&times, 95.0))
}

/// Benchmark sort_pairs: clones both keys and values before each run.
fn bench_pairs<K: Clone, V: Clone, F: FnMut(&mut [K], &mut [V])>(
    keys: &[K], vals: &[V], mut sort_fn: F,
) -> (f64, f64, f64) {
    let mut k = keys.to_vec();
    let mut v = vals.to_vec();
    let mut times = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        k.clear(); k.extend_from_slice(keys);
        v.clear(); v.extend_from_slice(vals);
        let start = Instant::now();
        sort_fn(&mut k, &mut v);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if i >= WARMUP {
            times.push(ms);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile(&times, 50.0), percentile(&times, 5.0), percentile(&times, 95.0))
}

fn bench_gpu_sort_zerocopy(sorter: &mut GpuSorter, data: &[u32]) -> (f64, f64, f64) {
    let n = data.len();
    let mut buf = sorter.alloc_sort_buffer(n);
    let mut times = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        buf.copy_from_slice(data);
        let start = Instant::now();
        sorter.sort_buffer(&buf).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if i >= WARMUP {
            times.push(ms);
        }
        assert!(buf.as_slice().first() <= buf.as_slice().last());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile(&times, 50.0), percentile(&times, 5.0), percentile(&times, 95.0))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    forge-sort v2 Benchmark — Apple M4 Pro                   ║");
    println!("║  GPU Radix Sort: u32, i32, f32, u64, i64, f64, argsort, sort_pairs          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let sizes_32: &[usize] = &[100_000, 1_000_000, 4_000_000, 16_000_000];
    let sizes_64: &[usize] = &[100_000, 1_000_000, 4_000_000, 16_000_000];

    let mut sorter = GpuSorter::new().unwrap();

    // Warmup GPU
    {
        let mut warmup = gen_random_u32(1_000_000);
        sorter.sort_u32(&mut warmup).unwrap();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Section 1: u32 (baseline, with CPU comparisons)
    // ═══════════════════════════════════════════════════════════════════════
    println!("  ═══ u32 (baseline, with CPU comparisons) ═══\n");
    println!("  {:>5} │ {:>12} {:>9} │ {:>12} {:>9} │ {:>12} {:>9} │ {:>7} {:>7}",
        "Size", "std_unstable", "Mk/s", "rayon_par", "Mk/s", "sort_u32", "Mk/s", "vs CPU", "vs par");
    println!("  ──────┼────────────────────────┼────────────────────────┼────────────────────────┼────────────────");

    for &n in sizes_32 {
        let data = gen_random_u32(n);
        let (std_p50, _, _) = bench_typed(&data, |d| d.sort_unstable());
        let (par_p50, _, _) = bench_typed(&data, |d| d.par_sort_unstable());
        let (gpu_p50, _, _) = bench_typed(&data, |d| sorter.sort_u32(d).unwrap());

        let std_mk = n as f64 / std_p50 / 1e3;
        let par_mk = n as f64 / par_p50 / 1e3;
        let gpu_mk = n as f64 / gpu_p50 / 1e3;

        println!(
            "  {:>5} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0} │ {:>5.1}x {:>5.1}x",
            size_str(n), std_p50, std_mk, par_p50, par_mk, gpu_p50, gpu_mk,
            gpu_mk / std_mk, gpu_mk / par_mk
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Section 2: All 32-bit types head-to-head
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n  ═══ 32-bit types: u32 vs i32 vs f32 ═══\n");
    println!("  {:>5} │ {:>12} {:>9} │ {:>12} {:>9} │ {:>12} {:>9}",
        "Size", "sort_u32", "Mk/s", "sort_i32", "Mk/s", "sort_f32", "Mk/s");
    println!("  ──────┼────────────────────────┼────────────────────────┼────────────────────────");

    for &n in sizes_32 {
        let data_u32 = gen_random_u32(n);
        let data_i32 = gen_random_i32(n);
        let data_f32 = gen_random_f32(n);

        let (u32_p50, _, _) = bench_typed(&data_u32, |d| sorter.sort_u32(d).unwrap());
        let (i32_p50, _, _) = bench_typed(&data_i32, |d| sorter.sort_i32(d).unwrap());
        let (f32_p50, _, _) = bench_typed(&data_f32, |d| sorter.sort_f32(d).unwrap());

        println!(
            "  {:>5} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0}",
            size_str(n),
            u32_p50, n as f64 / u32_p50 / 1e3,
            i32_p50, n as f64 / i32_p50 / 1e3,
            f32_p50, n as f64 / f32_p50 / 1e3,
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Section 3: All 64-bit types head-to-head
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n  ═══ 64-bit types: u64 vs i64 vs f64 ═══\n");
    println!("  {:>5} │ {:>12} {:>9} │ {:>12} {:>9} │ {:>12} {:>9}",
        "Size", "sort_u64", "Mk/s", "sort_i64", "Mk/s", "sort_f64", "Mk/s");
    println!("  ──────┼────────────────────────┼────────────────────────┼────────────────────────");

    for &n in sizes_64 {
        let data_u64 = gen_random_u64(n);
        let data_i64 = gen_random_i64(n);
        let data_f64 = gen_random_f64(n);

        let (u64_p50, _, _) = bench_typed(&data_u64, |d| sorter.sort_u64(d).unwrap());
        let (i64_p50, _, _) = bench_typed(&data_i64, |d| sorter.sort_i64(d).unwrap());
        let (f64_p50, _, _) = bench_typed(&data_f64, |d| sorter.sort_f64(d).unwrap());

        println!(
            "  {:>5} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0}",
            size_str(n),
            u64_p50, n as f64 / u64_p50 / 1e3,
            i64_p50, n as f64 / i64_p50 / 1e3,
            f64_p50, n as f64 / f64_p50 / 1e3,
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Section 4: 32-bit vs 64-bit (same type family)
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n  ═══ 32-bit vs 64-bit bandwidth cost ═══\n");
    println!("  {:>5} │ {:>12} {:>9} │ {:>12} {:>9} │ {:>7}",
        "Size", "sort_u32", "Mk/s", "sort_u64", "Mk/s", "ratio");
    println!("  ──────┼────────────────────────┼────────────────────────┼────────");

    for &n in sizes_32 {
        let data_u32 = gen_random_u32(n);
        let data_u64 = gen_random_u64(n);

        let (u32_p50, _, _) = bench_typed(&data_u32, |d| sorter.sort_u32(d).unwrap());
        let (u64_p50, _, _) = bench_typed(&data_u64, |d| sorter.sort_u64(d).unwrap());

        let u32_mk = n as f64 / u32_p50 / 1e3;
        let u64_mk = n as f64 / u64_p50 / 1e3;

        println!(
            "  {:>5} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0} │ {:>5.2}x",
            size_str(n), u32_p50, u32_mk, u64_p50, u64_mk, u32_mk / u64_mk
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Section 5: Argsort
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n  ═══ argsort (returns index permutation) ═══\n");
    println!("  {:>5} │ {:>12} {:>9} │ {:>12} {:>9} │ {:>12} {:>9}",
        "Size", "argsort_u32", "Mk/s", "argsort_f32", "Mk/s", "argsort_f64", "Mk/s");
    println!("  ──────┼────────────────────────┼────────────────────────┼────────────────────────");

    for &n in sizes_32 {
        let data_u32 = gen_random_u32(n);
        let data_f32 = gen_random_f32(n);
        let data_f64 = gen_random_f64(n);

        let (u32_p50, _, _) = bench_fn(|| { let _ = sorter.argsort_u32(&data_u32).unwrap(); });
        let (f32_p50, _, _) = bench_fn(|| { let _ = sorter.argsort_f32(&data_f32).unwrap(); });
        let (f64_p50, _, _) = bench_fn(|| { let _ = sorter.argsort_f64(&data_f64).unwrap(); });

        println!(
            "  {:>5} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0}",
            size_str(n),
            u32_p50, n as f64 / u32_p50 / 1e3,
            f32_p50, n as f64 / f32_p50 / 1e3,
            f64_p50, n as f64 / f64_p50 / 1e3,
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Section 6: sort_pairs
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n  ═══ sort_pairs (co-sort values by key order) ═══\n");
    println!("  {:>5} │ {:>12} {:>9} │ {:>12} {:>9} │ {:>12} {:>9}",
        "Size", "pairs_u32", "Mk/s", "pairs_f32", "Mk/s", "pairs_u64", "Mk/s");
    println!("  ──────┼────────────────────────┼────────────────────────┼────────────────────────");

    for &n in sizes_32 {
        let keys_u32 = gen_random_u32(n);
        let vals = gen_random_u32(n);
        let keys_f32 = gen_random_f32(n);
        let keys_u64 = gen_random_u64(n);

        let (u32_p50, _, _) = bench_pairs(&keys_u32, &vals, |k, v| sorter.sort_pairs_u32(k, v).unwrap());
        let (f32_p50, _, _) = bench_pairs(&keys_f32, &vals, |k, v| sorter.sort_pairs_f32(k, v).unwrap());
        let (u64_p50, _, _) = bench_pairs(&keys_u64, &vals, |k, v| sorter.sort_pairs_u64(k, v).unwrap());

        println!(
            "  {:>5} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0}",
            size_str(n),
            u32_p50, n as f64 / u32_p50 / 1e3,
            f32_p50, n as f64 / f32_p50 / 1e3,
            u64_p50, n as f64 / u64_p50 / 1e3,
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Section 7: Zero-copy u32 (original benchmark)
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n  ═══ Zero-Copy u32 (sort_buffer, no memcpy) ═══\n");
    println!("  {:>5} │ {:>12} {:>9} │ {:>12} {:>9} │ {:>7}",
        "Size", "sort_u32", "Mk/s", "sort_buffer", "Mk/s", "speedup");
    println!("  ──────┼────────────────────────┼────────────────────────┼────────");

    for &n in sizes_32 {
        let data = gen_random_u32(n);
        let (gpu_p50, _, _) = bench_typed(&data, |d| sorter.sort_u32(d).unwrap());
        let (zc_p50, _, _) = bench_gpu_sort_zerocopy(&mut sorter, &data);

        let gpu_mk = n as f64 / gpu_p50 / 1e3;
        let zc_mk = n as f64 / zc_p50 / 1e3;

        println!(
            "  {:>5} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0} │ {:>5.2}x",
            size_str(n), gpu_p50, gpu_mk, zc_p50, zc_mk, zc_mk / gpu_mk
        );
    }

    println!("\n  ─── Hardware ───");
    println!("  Apple M4 Pro (12-core CPU, 20-core GPU, 48GB unified memory)");
    println!("  forge-sort v0.2.0: MSD+fused-inner 8-bit radix, Metal 3.2");
    println!("  {} warmup + {} timed runs per measurement, reporting p50\n", WARMUP, RUNS);
}
