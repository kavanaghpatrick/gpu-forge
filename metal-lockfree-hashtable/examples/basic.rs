//! Basic usage: insert 1M keys, lookup, verify, print timing.

use metal_lockfree_hashtable::{
    bench::{cpu_baseline, run_benchmarks, BenchConfig},
    GpuHashTable, Version,
};
use rand::Rng;

const EMPTY_KEY: u32 = 0xFFFFFFFF;

fn main() {
    println!("=== Lock-Free GPU Hash Table ===\n");

    // ── Quick correctness check ──
    let table = GpuHashTable::new(Version::V3, 2_000_000);
    let mut rng = rand::thread_rng();

    let n = 1_000_000;
    let keys: Vec<u32> = (0..n)
        .map(|_| {
            let k: u32 = rng.gen();
            if k == EMPTY_KEY { 0xFFFFFFFE } else { k }
        })
        .collect();
    let values: Vec<u32> = (0..n).map(|_| rng.gen()).collect();

    let insert_ms = table.insert(&keys, &values);
    let (results, lookup_ms) = table.lookup(&keys);

    let hits = results.iter().filter(|&&v| v != EMPTY_KEY).count();
    let hit_rate = hits as f64 / n as f64 * 100.0;

    println!("Quick check (V3, 1M keys):");
    println!("  Insert: {:.3} ms ({:.0} Mops)", insert_ms, n as f64 / insert_ms / 1000.0);
    println!("  Lookup: {:.3} ms ({:.0} Mops)", lookup_ms, n as f64 / lookup_ms / 1000.0);
    println!("  Hit rate: {:.1}% ({}/{})", hit_rate, hits, n);
    println!();

    // ── CPU baseline ──
    let cpu_ms = cpu_baseline(&keys, &values);
    println!("CPU baseline (std::HashMap insert+lookup): {:.1} ms", cpu_ms);
    println!("GPU speedup: {:.0}x\n", cpu_ms / (insert_ms + lookup_ms));

    // ── Full benchmark suite ──
    println!("--- Benchmark: 1M keys, 2M capacity ---\n");

    let config = BenchConfig {
        capacity: 2_097_152,
        num_keys: 1_000_000,
        versions: vec![Version::V1, Version::V2, Version::V3],
        runs: 10,
        warmup: 3,
    };

    let results = run_benchmarks(&config);
    for r in &results {
        println!("  {}", r);
    }
    println!();

    // ── Scaling test: 64M keys (DRAM-resident) ──
    println!("--- Benchmark: 32M keys, 64M capacity (DRAM) ---\n");

    let config_large = BenchConfig {
        capacity: 67_108_864,
        num_keys: 32_000_000,
        versions: vec![Version::V1, Version::V2, Version::V3],
        runs: 5,
        warmup: 2,
    };

    let results_large = run_benchmarks(&config_large);
    for r in &results_large {
        println!("  {}", r);
    }
    println!();

    // ── Summary ──
    let v3_lookup_1m = results.iter().find(|r| r.version == Version::V3 && r.op == "lookup");
    let v3_lookup_32m = results_large.iter().find(|r| r.version == Version::V3 && r.op == "lookup");

    if let (Some(small), Some(large)) = (v3_lookup_1m, v3_lookup_32m) {
        println!("=== Summary ===");
        println!("  V3 Lookup @ 1M:  {:.0} Mops (SLC-resident)", small.mops);
        println!("  V3 Lookup @ 32M: {:.0} Mops (DRAM-resident)", large.mops);
    }
}
