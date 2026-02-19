//! Experiment 16: 8-Bit Radix Sort (3000+ Mkeys/s target)
//!
//! Core change from exp15: 8-bit radix with 256 bins and 4 passes
//! instead of 4-bit radix with 16 bins and 8 passes.
//! Halves pass count → halves bandwidth → target ~1.5x speedup.
//!
//! Architecture: non-persistent (1 TG/tile), decoupled lookback,
//! per-SG atomic histogram (256 bins too many for SIMD butterfly),
//! 2048 elements/tile, 256 threads/TG.

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
use std::ptr::NonNull;

const NUM_BINS: usize = 256;
const TILE_SIZE: usize = 2048;
const NUM_PASSES: usize = 4;
const THREADS_PER_TG: usize = 256;
const WARMUP: usize = 5;
const RUNS: usize = 50;

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp16Params {
    element_count: u32,
    num_tiles: u32,
    num_tgs: u32,
    shift: u32,
    pass: u32,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] } else { sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo as f64) }
}

fn print_stats(label: &str, n: usize, times: &[f64], correct: bool) {
    let p5 = percentile(times, 5.0);
    let p50 = percentile(times, 50.0);
    let p95 = percentile(times, 95.0);
    let spread = if p5 > 0.0 { (p95 - p5) / p5 * 100.0 } else { 0.0 };
    let throughput = n as f64 / p50 / 1000.0;
    let status = if correct { "ok" } else { "FAIL" };
    println!(
        "    {:>4}: {:>7.3} ms  {:>6.0} Mkeys/s  {:>4}  [p5={:.2} p95={:.2} spread={:.0}%]",
        label, p50, throughput, status, p5, p95, spread
    );
}

type Pso = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

fn bench_8bit(
    ctx: &MetalContext,
    pso_combined_hist: &Pso,
    pso_global_prefix: &Pso,
    pso_zero_status: &Pso,
    pso_partition: &Pso,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(TILE_SIZE);

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
    let buf_global_hist = alloc_buffer(&ctx.device, NUM_PASSES * NUM_BINS * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * NUM_BINS * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
    let hist_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    // Zero TGs: num_tiles * 256 entries / 256 threads per TG = num_tiles
    let zero_tg_count = (num_tiles * NUM_BINS).div_ceil(THREADS_PER_TG);

    let params_size = std::mem::size_of::<Exp16Params>();

    let encode_sort = |cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>| {
        // Zero global_hist (CPU-side, before GPU starts)
        unsafe {
            std::ptr::write_bytes(
                buf_global_hist.contents().as_ptr() as *mut u8, 0,
                NUM_PASSES * NUM_BINS * 4,
            );
        }

        let base_params = Exp16Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            num_tgs: num_tiles as u32,
            shift: 0,
            pass: 0,
        };

        // ── Encoder 1: Combined histogram (all 4 passes in one read) ──
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_combined_hist);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&base_params as *const Exp16Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_tgs, tg_size);
        enc.endEncoding();

        // ── Encoder 2: Global prefix sum (in-place on global_hist) ──
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        // ── 4 passes: zero_status + partition with ping-pong ──
        for pass in 0..NUM_PASSES as u32 {
            let params = Exp16Params {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                num_tgs: num_tiles as u32,
                shift: pass * 8, // 8-bit radix: shift by 8 per pass
                pass,
            };

            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };

            // Zero tile_status + work counter
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_zero_status);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const Exp16Params as *mut _).unwrap(),
                    params_size, 2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: zero_tg_count, height: 1, depth: 1 }, tg_size,
            );
            enc.endEncoding();

            // Partition: 1 TG per tile with decoupled lookback
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_partition);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 4);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const Exp16Params as *mut _).unwrap(),
                    params_size, 5,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tiles, height: 1, depth: 1 }, tg_size,
            );
            enc.endEncoding();
        }
    };

    // ── Correctness check ──
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }

    let cmd = ctx.command_buffer();
    encode_sort(&cmd);
    cmd.commit();
    cmd.waitUntilCompleted();

    // After 4 passes (even count), result is in buf_a
    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! 8-bit radix {}: {} / {} mismatched ({:.1}%)",
            n, mismatches, n, mismatches as f64 / n as f64 * 100.0
        );
        let mut shown = 0;
        for (i, (got, exp)) in results.iter().zip(expected.iter()).enumerate() {
            if got != exp && shown < 5 {
                println!("         [{}] got={} exp={}", i, got, exp);
                shown += 1;
            }
        }
    }

    // ── Benchmark ──
    let mut times = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
        }

        let cmd = ctx.command_buffer();
        encode_sort(&cmd);
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
    println!("Experiment 16: 8-Bit Radix Sort");
    println!("{}", "=".repeat(60));
    println!("8-bit radix, 4 passes, 256 bins, 2048 elem/tile");
    println!("Per-SG atomic histogram, decoupled lookback (256 threads)\n");

    let pso_combined_hist = ctx.make_pipeline("exp16_combined_histogram");
    let pso_global_prefix = ctx.make_pipeline("exp16_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");

    println!("  ── 8-bit Radix Sort (4 passes, 256 bins) ──");
    for &n in &[1_000_000, 4_000_000, 16_000_000] {
        let label = format!("{}M", n / 1_000_000);
        let (times, correct) = bench_8bit(
            ctx, &pso_combined_hist, &pso_global_prefix,
            &pso_zero_status, &pso_partition, n,
        );
        print_stats(&label, n, &times, correct);
    }
}
