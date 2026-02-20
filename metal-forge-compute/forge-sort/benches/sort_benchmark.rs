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

fn percentile(times: &[f64], p: f64) -> f64 {
    let idx = (p / 100.0 * (times.len() - 1) as f64).round() as usize;
    times[idx.min(times.len() - 1)]
}

fn bench_std_sort(data: &[u32]) -> (f64, f64, f64) {
    let mut times = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let mut copy = data.to_vec();
        let start = Instant::now();
        copy.sort_unstable();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if i >= WARMUP {
            times.push(ms);
        }
        // Prevent optimization
        assert!(copy.first() <= copy.last());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile(&times, 50.0), percentile(&times, 5.0), percentile(&times, 95.0))
}

fn bench_rayon_sort(data: &[u32]) -> (f64, f64, f64) {
    let mut times = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let mut copy = data.to_vec();
        let start = Instant::now();
        copy.par_sort_unstable();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if i >= WARMUP {
            times.push(ms);
        }
        assert!(copy.first() <= copy.last());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile(&times, 50.0), percentile(&times, 5.0), percentile(&times, 95.0))
}

fn bench_gpu_sort(sorter: &mut GpuSorter, data: &[u32]) -> (f64, f64, f64) {
    let mut times = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let mut copy = data.to_vec();
        let start = Instant::now();
        sorter.sort_u32(&mut copy).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if i >= WARMUP {
            times.push(ms);
        }
        assert!(copy.first() <= copy.last());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile(&times, 50.0), percentile(&times, 5.0), percentile(&times, 95.0))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              forge-sort Benchmark — Apple M4 Pro               ║");
    println!("║  GPU Radix Sort vs CPU sort_unstable vs Rayon par_sort         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let sizes: &[usize] = &[
        100_000,
        1_000_000,
        2_000_000,
        4_000_000,
        8_000_000,
        16_000_000,
        32_000_000,
    ];

    let mut sorter = GpuSorter::new().unwrap();

    // Warmup GPU
    {
        let mut warmup = gen_random_u32(1_000_000);
        sorter.sort_u32(&mut warmup).unwrap();
    }

    println!("  {:>5} │ {:>12} {:>9} │ {:>12} {:>9} │ {:>12} {:>9} │ {:>7} {:>7}",
        "Size", "std_unstable", "Mk/s", "rayon_par", "Mk/s", "forge-sort", "Mk/s", "vs CPU", "vs par");
    println!("  ──────┼────────────────────────┼────────────────────────┼────────────────────────┼────────────────");

    for &n in sizes {
        let data = gen_random_u32(n);

        let (std_p50, _, _) = bench_std_sort(&data);
        let (par_p50, _, _) = bench_rayon_sort(&data);
        let (gpu_p50, _, _) = bench_gpu_sort(&mut sorter, &data);

        let std_mkeys = n as f64 / std_p50 / 1e3;
        let par_mkeys = n as f64 / par_p50 / 1e3;
        let gpu_mkeys = n as f64 / gpu_p50 / 1e3;

        let vs_cpu = gpu_mkeys / std_mkeys;
        let vs_par = gpu_mkeys / par_mkeys;

        let size_str = if n >= 1_000_000 {
            format!("{}M", n / 1_000_000)
        } else {
            format!("{}K", n / 1_000)
        };

        println!(
            "  {:>5} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0} │ {:>8.3} ms {:>7.0} │ {:>5.1}x {:>5.1}x",
            size_str, std_p50, std_mkeys, par_p50, par_mkeys, gpu_p50, gpu_mkeys, vs_cpu, vs_par
        );
    }

    println!("\n  ─── Published Reference Numbers (other implementations) ───");
    println!("  VSort (Apple Silicon LSD radix, M4 Pro):");
    println!("    1M random:       10.09 ms =   99 Mk/s");
    println!("    1M nearly sorted: 4.81 ms =  208 Mk/s");
    println!("  VkRadixSort (Vulkan, RTX 3070):");
    println!("    1M random:       18.97 ms =   53 Mk/s");
    println!("  Note: MPS (Metal Performance Shaders) has NO sort function");

    println!("\n  ─── Hardware ───");
    println!("  Apple M4 Pro (12-core CPU, 20-core GPU, 48GB unified memory)");
    println!("  forge-sort: MSD+fused-inner 8-bit radix, 4 dispatches, Metal 3.2\n");
}
