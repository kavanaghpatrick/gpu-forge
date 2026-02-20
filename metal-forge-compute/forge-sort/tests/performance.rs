#![cfg(feature = "perf-test")]

mod common;

use forge_sort::GpuSorter;
use rand::Rng;
use std::time::{Duration, Instant};

fn median(times: &mut Vec<Duration>) -> Duration {
    times.sort();
    times[times.len() / 2]
}

// --- 32-bit memcpy sorts ---

#[test]
fn test_perf_sort_u32_memcpy() {
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut rng = common::seeded_rng(42);
    let data_orig: Vec<u32> = (0..n).map(|_| rng.gen()).collect();

    // warmup
    let mut data = data_orig.clone();
    sorter.sort_u32(&mut data).unwrap();

    let mut times = Vec::new();
    for _ in 0..10 {
        let mut data = data_orig.clone();
        let start = Instant::now();
        sorter.sort_u32(&mut data).unwrap();
        times.push(start.elapsed());
    }
    let med = median(&mut times);
    let mks = n as f64 / med.as_secs_f64() / 1_000_000.0;
    println!("sort_u32 memcpy 16M: {:.0} Mk/s ({:.2} ms)", mks, med.as_secs_f64() * 1000.0);
    assert!(mks >= 2000.0, "sort_u32 memcpy: {:.0} Mk/s < 2000 threshold", mks);
}

#[test]
fn test_perf_sort_i32_memcpy() {
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut rng = common::seeded_rng(43);
    let data_orig: Vec<i32> = (0..n).map(|_| rng.gen()).collect();

    let mut data = data_orig.clone();
    sorter.sort_i32(&mut data).unwrap();

    let mut times = Vec::new();
    for _ in 0..10 {
        let mut data = data_orig.clone();
        let start = Instant::now();
        sorter.sort_i32(&mut data).unwrap();
        times.push(start.elapsed());
    }
    let med = median(&mut times);
    let mks = n as f64 / med.as_secs_f64() / 1_000_000.0;
    println!("sort_i32 memcpy 16M: {:.0} Mk/s ({:.2} ms)", mks, med.as_secs_f64() * 1000.0);
    assert!(mks >= 1900.0, "sort_i32 memcpy: {:.0} Mk/s < 1900 threshold", mks);
}

#[test]
fn test_perf_sort_f32_memcpy() {
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut rng = common::seeded_rng(44);
    let data_orig: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 2000.0 - 1000.0).collect();

    let mut data = data_orig.clone();
    sorter.sort_f32(&mut data).unwrap();

    let mut times = Vec::new();
    for _ in 0..10 {
        let mut data = data_orig.clone();
        let start = Instant::now();
        sorter.sort_f32(&mut data).unwrap();
        times.push(start.elapsed());
    }
    let med = median(&mut times);
    let mks = n as f64 / med.as_secs_f64() / 1_000_000.0;
    println!("sort_f32 memcpy 16M: {:.0} Mk/s ({:.2} ms)", mks, med.as_secs_f64() * 1000.0);
    assert!(mks >= 500.0, "sort_f32 memcpy: {:.0} Mk/s < 500 threshold", mks);
}

// --- argsort ---

#[test]
fn test_perf_argsort_u32() {
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut rng = common::seeded_rng(45);
    let data: Vec<u32> = (0..n).map(|_| rng.gen()).collect();

    // warmup
    let _ = sorter.argsort_u32(&data).unwrap();

    let mut times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _ = sorter.argsort_u32(&data).unwrap();
        times.push(start.elapsed());
    }
    let med = median(&mut times);
    let mks = n as f64 / med.as_secs_f64() / 1_000_000.0;
    println!("argsort_u32 16M: {:.0} Mk/s ({:.2} ms)", mks, med.as_secs_f64() * 1000.0);
    assert!(mks >= 750.0, "argsort_u32: {:.0} Mk/s < 750 threshold", mks);
}

// --- sort_pairs ---

#[test]
fn test_perf_sort_pairs_f32() {
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut rng = common::seeded_rng(46);
    let keys_orig: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 2000.0 - 1000.0).collect();
    let vals_orig: Vec<u32> = (0..n).map(|_| rng.gen()).collect();

    // warmup
    let mut keys = keys_orig.clone();
    let mut vals = vals_orig.clone();
    sorter.sort_pairs_f32(&mut keys, &mut vals).unwrap();

    let mut times = Vec::new();
    for _ in 0..10 {
        let mut keys = keys_orig.clone();
        let mut vals = vals_orig.clone();
        let start = Instant::now();
        sorter.sort_pairs_f32(&mut keys, &mut vals).unwrap();
        times.push(start.elapsed());
    }
    let med = median(&mut times);
    let mks = n as f64 / med.as_secs_f64() / 1_000_000.0;
    println!("sort_pairs_f32 16M: {:.0} Mk/s ({:.2} ms)", mks, med.as_secs_f64() * 1000.0);
    assert!(mks >= 250.0, "sort_pairs_f32: {:.0} Mk/s < 250 threshold", mks);
}

// --- 64-bit sorts ---

#[test]
fn test_perf_sort_u64_memcpy() {
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut rng = common::seeded_rng(47);
    let data_orig: Vec<u64> = (0..n).map(|_| rng.gen()).collect();

    let mut data = data_orig.clone();
    sorter.sort_u64(&mut data).unwrap();

    let mut times = Vec::new();
    for _ in 0..10 {
        let mut data = data_orig.clone();
        let start = Instant::now();
        sorter.sort_u64(&mut data).unwrap();
        times.push(start.elapsed());
    }
    let med = median(&mut times);
    let mks = n as f64 / med.as_secs_f64() / 1_000_000.0;
    println!("sort_u64 memcpy 16M: {:.0} Mk/s ({:.2} ms)", mks, med.as_secs_f64() * 1000.0);
    assert!(mks >= 600.0, "sort_u64 memcpy: {:.0} Mk/s < 600 threshold", mks);
}

#[test]
fn test_perf_sort_i64_memcpy() {
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut rng = common::seeded_rng(48);
    let data_orig: Vec<i64> = (0..n).map(|_| rng.gen()).collect();

    let mut data = data_orig.clone();
    sorter.sort_i64(&mut data).unwrap();

    let mut times = Vec::new();
    for _ in 0..10 {
        let mut data = data_orig.clone();
        let start = Instant::now();
        sorter.sort_i64(&mut data).unwrap();
        times.push(start.elapsed());
    }
    let med = median(&mut times);
    let mks = n as f64 / med.as_secs_f64() / 1_000_000.0;
    println!("sort_i64 memcpy 16M: {:.0} Mk/s ({:.2} ms)", mks, med.as_secs_f64() * 1000.0);
    assert!(mks >= 550.0, "sort_i64 memcpy: {:.0} Mk/s < 550 threshold", mks);
}

#[test]
fn test_perf_sort_f64_memcpy() {
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut rng = common::seeded_rng(49);
    let data_orig: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 2000.0 - 1000.0).collect();

    let mut data = data_orig.clone();
    sorter.sort_f64(&mut data).unwrap();

    let mut times = Vec::new();
    for _ in 0..10 {
        let mut data = data_orig.clone();
        let start = Instant::now();
        sorter.sort_f64(&mut data).unwrap();
        times.push(start.elapsed());
    }
    let med = median(&mut times);
    let mks = n as f64 / med.as_secs_f64() / 1_000_000.0;
    println!("sort_f64 memcpy 16M: {:.0} Mk/s ({:.2} ms)", mks, med.as_secs_f64() * 1000.0);
    assert!(mks >= 100.0, "sort_f64 memcpy: {:.0} Mk/s < 100 threshold", mks);
}
