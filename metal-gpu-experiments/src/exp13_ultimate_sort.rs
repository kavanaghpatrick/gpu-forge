//! Experiment 13: Ultimate Radix Sort
//!
//! Combines all proven optimizations:
//!   1. Fence-fixed single dispatch (atomic_thread_fence seq_cst device_scope)
//!   2. SIMD rank (2.83x faster scatter)
//!   3. Tile-status reuse (single section, cleared between passes)
//!   4. u32, u64, and key-value pair support
//!
//! Scale tests: 1M to 16M elements with 50-run percentile analysis.

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

const TILE_SIZE: usize = 256;
const NUM_BINS: usize = 256;
const NUM_TGS: usize = 64;
const WARMUP: usize = 5;
const RUNS: usize = 50;

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp13Params {
    element_count: u32,
    num_tiles: u32,
    num_tgs: u32,
    num_passes: u32,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo as f64)
    }
}

fn print_stats(label: &str, n: usize, times: &[f64], correct: bool) {
    let p5 = percentile(times, 5.0);
    let p50 = percentile(times, 50.0);
    let p95 = percentile(times, 95.0);
    let spread = if p5 > 0.0 {
        (p95 - p5) / p5 * 100.0
    } else {
        0.0
    };
    let throughput = n as f64 / p50 / 1000.0; // Mkeys/s
    let status = if correct { "ok" } else { "FAIL" };
    println!(
        "    {:>4}: {:>7.3} ms  {:>6.0} Mkeys/s  {:>4}  [p5={:.2} p95={:.2} spread={:.0}%]",
        label, p50, throughput, status, p5, p95, spread
    );
}

/// Benchmark u32 sort at given element count.
fn bench_u32(
    ctx: &MetalContext,
    pso: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(TILE_SIZE);
    let num_passes: usize = 4;
    let ts_section = num_tiles * NUM_BINS;
    let num_counters = 2 + 3 * (num_passes - 1) + 1 + 16;

    let input: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };
    let mut expected = input.clone();
    expected.sort();

    let buf_input = alloc_buffer_with_data(&ctx.device, &input);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_ts = alloc_buffer(&ctx.device, ts_section * 4);
    let buf_counters = alloc_buffer(&ctx.device, num_counters * 4);

    let params = Exp13Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        num_tgs: NUM_TGS as u32,
        num_passes: num_passes as u32,
    };
    let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
        std::ptr::write_bytes(buf_ts.contents().as_ptr() as *mut u8, 0, ts_section * 4);
        std::ptr::write_bytes(
            buf_counters.contents().as_ptr() as *mut u8,
            0,
            num_counters * 4,
        );
    }

    let cmd = ctx.command_buffer();
    let enc = cmd.computeCommandEncoder().unwrap();
    dispatch_1d_tg(
        &enc,
        pso,
        &[
            (buf_a.as_ref(), 0),
            (buf_b.as_ref(), 1),
            (buf_ts.as_ref(), 2),
            (buf_counters.as_ref(), 3),
            (buf_params.as_ref(), 4),
        ],
        NUM_TGS,
        TILE_SIZE,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    // 4 passes (even): pass 0 a→b, pass 1 b→a, pass 2 a→b, pass 3 b→a → result in buf_a
    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "      !! u32 {}: {} / {} mismatched ({:.1}%)",
            n,
            mismatches,
            n,
            mismatches as f64 / n as f64 * 100.0
        );
    }

    // Benchmark
    let mut times = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_ts.contents().as_ptr() as *mut u8, 0, ts_section * 4);
            std::ptr::write_bytes(
                buf_counters.contents().as_ptr() as *mut u8,
                0,
                num_counters * 4,
            );
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc,
            pso,
            &[
                (buf_a.as_ref(), 0),
                (buf_b.as_ref(), 1),
                (buf_ts.as_ref(), 2),
                (buf_counters.as_ref(), 3),
                (buf_params.as_ref(), 4),
            ],
            NUM_TGS,
            TILE_SIZE,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let elapsed = gpu_elapsed_ms(&cmd);
        if iter >= WARMUP {
            times.push(elapsed);
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times, correct)
}

/// Benchmark u64 sort at given element count.
fn bench_u64(
    ctx: &MetalContext,
    pso: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(TILE_SIZE);
    let num_passes: usize = 8;
    let ts_section = num_tiles * NUM_BINS;
    let num_counters = 2 + 3 * (num_passes - 1) + 1 + 16;

    let input: Vec<u64> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u64>()).collect()
    };
    let mut expected = input.clone();
    expected.sort();

    let buf_input = alloc_buffer_with_data(&ctx.device, &input);
    let buf_a = alloc_buffer(&ctx.device, n * 8);
    let buf_b = alloc_buffer(&ctx.device, n * 8);
    let buf_ts = alloc_buffer(&ctx.device, ts_section * 4);
    let buf_counters = alloc_buffer(&ctx.device, num_counters * 4);

    let params = Exp13Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        num_tgs: NUM_TGS as u32,
        num_passes: num_passes as u32,
    };
    let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 8);
        std::ptr::write_bytes(buf_ts.contents().as_ptr() as *mut u8, 0, ts_section * 4);
        std::ptr::write_bytes(
            buf_counters.contents().as_ptr() as *mut u8,
            0,
            num_counters * 4,
        );
    }

    let cmd = ctx.command_buffer();
    let enc = cmd.computeCommandEncoder().unwrap();
    dispatch_1d_tg(
        &enc,
        pso,
        &[
            (buf_a.as_ref(), 0),
            (buf_b.as_ref(), 1),
            (buf_ts.as_ref(), 2),
            (buf_counters.as_ref(), 3),
            (buf_params.as_ref(), 4),
        ],
        NUM_TGS,
        TILE_SIZE,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    // 8 passes: last pass index=7 (odd) → result in buf_a
    let results: Vec<u64> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "      !! u64 {}: {} / {} mismatched ({:.1}%)",
            n,
            mismatches,
            n,
            mismatches as f64 / n as f64 * 100.0
        );
    }

    // Benchmark
    let mut times = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 8);
            std::ptr::write_bytes(buf_ts.contents().as_ptr() as *mut u8, 0, ts_section * 4);
            std::ptr::write_bytes(
                buf_counters.contents().as_ptr() as *mut u8,
                0,
                num_counters * 4,
            );
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc,
            pso,
            &[
                (buf_a.as_ref(), 0),
                (buf_b.as_ref(), 1),
                (buf_ts.as_ref(), 2),
                (buf_counters.as_ref(), 3),
                (buf_params.as_ref(), 4),
            ],
            NUM_TGS,
            TILE_SIZE,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let elapsed = gpu_elapsed_ms(&cmd);
        if iter >= WARMUP {
            times.push(elapsed);
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times, correct)
}

/// Benchmark key-value u32 sort at given element count.
fn bench_kv32(
    ctx: &MetalContext,
    pso: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(TILE_SIZE);
    let num_passes: usize = 4;
    let ts_section = num_tiles * NUM_BINS;
    let num_counters = 2 + 3 * (num_passes - 1) + 1 + 16;

    let keys: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };
    let values: Vec<u32> = (0..n as u32).collect();

    // Expected: sort by key, track where each index ended up
    let mut pairs: Vec<(u32, u32)> = keys.iter().copied().zip(values.iter().copied()).collect();
    pairs.sort_by_key(|(k, _)| *k);
    let expected_keys: Vec<u32> = pairs.iter().map(|(k, _)| *k).collect();

    let buf_key_input = alloc_buffer_with_data(&ctx.device, &keys);
    let buf_val_input = alloc_buffer_with_data(&ctx.device, &values);
    let buf_key_a = alloc_buffer(&ctx.device, n * 4);
    let buf_key_b = alloc_buffer(&ctx.device, n * 4);
    let buf_val_a = alloc_buffer(&ctx.device, n * 4);
    let buf_val_b = alloc_buffer(&ctx.device, n * 4);
    let buf_ts = alloc_buffer(&ctx.device, ts_section * 4);
    let buf_counters = alloc_buffer(&ctx.device, num_counters * 4);

    let params = Exp13Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        num_tgs: NUM_TGS as u32,
        num_passes: num_passes as u32,
    };
    let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

    // Correctness check
    unsafe {
        let src = buf_key_input.contents().as_ptr() as *const u8;
        let dst = buf_key_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
        let src = buf_val_input.contents().as_ptr() as *const u8;
        let dst = buf_val_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
        std::ptr::write_bytes(buf_ts.contents().as_ptr() as *mut u8, 0, ts_section * 4);
        std::ptr::write_bytes(
            buf_counters.contents().as_ptr() as *mut u8,
            0,
            num_counters * 4,
        );
    }

    let cmd = ctx.command_buffer();
    let enc = cmd.computeCommandEncoder().unwrap();
    dispatch_1d_tg(
        &enc,
        pso,
        &[
            (buf_key_a.as_ref(), 0),
            (buf_key_b.as_ref(), 1),
            (buf_val_a.as_ref(), 2),
            (buf_val_b.as_ref(), 3),
            (buf_ts.as_ref(), 4),
            (buf_counters.as_ref(), 5),
            (buf_params.as_ref(), 6),
        ],
        NUM_TGS,
        TILE_SIZE,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    // 4 passes → result in key_a, val_a
    let result_keys: Vec<u32> = unsafe { read_buffer_slice(&buf_key_a, n) };
    let result_vals: Vec<u32> = unsafe { read_buffer_slice(&buf_val_a, n) };

    let keys_correct = result_keys == expected_keys;
    let vals_correct = result_vals
        .iter()
        .zip(result_keys.iter())
        .all(|(v, k)| (*v as usize) < n && keys[*v as usize] == *k);
    let correct = keys_correct && vals_correct;

    if !correct {
        let key_mismatches = result_keys
            .iter()
            .zip(expected_keys.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "      !! kv32 {}: keys {} / {} mismatched, vals_valid: {}",
            n, key_mismatches, n, vals_correct
        );
    }

    // Benchmark
    let mut times = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_key_input.contents().as_ptr() as *const u8;
            let dst = buf_key_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            let src = buf_val_input.contents().as_ptr() as *const u8;
            let dst = buf_val_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_ts.contents().as_ptr() as *mut u8, 0, ts_section * 4);
            std::ptr::write_bytes(
                buf_counters.contents().as_ptr() as *mut u8,
                0,
                num_counters * 4,
            );
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc,
            pso,
            &[
                (buf_key_a.as_ref(), 0),
                (buf_key_b.as_ref(), 1),
                (buf_val_a.as_ref(), 2),
                (buf_val_b.as_ref(), 3),
                (buf_ts.as_ref(), 4),
                (buf_counters.as_ref(), 5),
                (buf_params.as_ref(), 6),
            ],
            NUM_TGS,
            TILE_SIZE,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let elapsed = gpu_elapsed_ms(&cmd);
        if iter >= WARMUP {
            times.push(elapsed);
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times, correct)
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 13: Ultimate Radix Sort");
    println!("{}", "=".repeat(60));
    println!("Fence-fixed single dispatch + SIMD rank + tile-status reuse");
    println!("64 persistent TGs × 256 threads, 50 runs per config\n");

    let pso_u32 = ctx.make_pipeline("exp13_sort_u32");
    let pso_u64 = ctx.make_pipeline("exp13_sort_u64");
    let pso_kv32 = ctx.make_pipeline("exp13_sort_kv32");

    // ── u32 Scale Test ──────────────────────────────────────────
    println!("  ── u32 Scale Test ──────────────────────────────────────");
    for &n in &[1_000_000, 4_000_000, 8_000_000, 16_000_000] {
        let label = format!("{}M", n / 1_000_000);
        let (times, correct) = bench_u32(ctx, &pso_u32, n);
        print_stats(&label, n, &times, correct);
    }

    // ── u64 Sort ────────────────────────────────────────────────
    println!("\n  ── u64 Sort (8-pass) ───────────────────────────────────");
    for &n in &[1_000_000, 4_000_000] {
        let label = format!("{}M", n / 1_000_000);
        let (times, correct) = bench_u64(ctx, &pso_u64, n);
        print_stats(&label, n, &times, correct);
    }

    // ── Key-Value u32 Sort ──────────────────────────────────────
    println!("\n  ── Key-Value u32 Sort ──────────────────────────────────");
    for &n in &[1_000_000, 4_000_000] {
        let label = format!("{}M", n / 1_000_000);
        let (times, correct) = bench_kv32(ctx, &pso_kv32, n);
        print_stats(&label, n, &times, correct);
    }

    // ── Throughput summary ──────────────────────────────────────
    println!("\n  ── Peak Throughput Summary ─────────────────────────────");
    println!("    u32:  best p5 across all scales");
    println!("    u64:  2x memory per element, 2x passes → expect ~4x slower");
    println!("    kv32: same passes as u32, extra value scatter overhead");
}
