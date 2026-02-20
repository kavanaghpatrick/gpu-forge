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
const TILE_SIZE: usize = 4096;
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

fn bench_8bit_generic(
    ctx: &MetalContext,
    pso_combined_hist: &Pso,
    pso_global_prefix: &Pso,
    pso_zero_status: &Pso,
    pso_partition: &Pso,
    n: usize,
    part_tile_size: usize,
    num_tiles: usize,
) -> (Vec<f64>, bool) {
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
    let hist_tile_count = n.div_ceil(TILE_SIZE); // histogram always uses 4096 tiles
    let zero_tg_count = (num_tiles * NUM_BINS).div_ceil(THREADS_PER_TG);
    let params_size = std::mem::size_of::<Exp16Params>();

    let encode_sort = |cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>| {
        unsafe {
            std::ptr::write_bytes(
                buf_global_hist.contents().as_ptr() as *mut u8, 0,
                NUM_PASSES * NUM_BINS * 4,
            );
        }

        let hist_params = Exp16Params {
            element_count: n as u32,
            num_tiles: hist_tile_count as u32,
            num_tgs: hist_tile_count as u32,
            shift: 0, pass: 0,
        };

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_combined_hist);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&hist_params as *const Exp16Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: hist_tile_count, height: 1, depth: 1 }, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        for pass in 0..NUM_PASSES as u32 {
            let params = Exp16Params {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                num_tgs: num_tiles as u32,
                shift: pass * 8,
                pass,
            };

            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };

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
                MTLSize { width: zero_tg_count, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();

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
                MTLSize { width: num_tiles, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();
        }
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }

    let cmd = ctx.command_buffer();
    encode_sort(&cmd);
    cmd.commit();
    cmd.waitUntilCompleted();

    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! generic radix {}: {} / {} mismatched ({:.1}%)",
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

const V2_TILE_SIZE: usize = 2048;

fn bench_8bit_v2(
    ctx: &MetalContext,
    pso_combined_hist: &Pso,
    pso_global_prefix: &Pso,
    pso_zero_status: &Pso,
    pso_partition_v2: &Pso,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(V2_TILE_SIZE);

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
    let zero_tg_count = (num_tiles * NUM_BINS).div_ceil(THREADS_PER_TG);

    let params_size = std::mem::size_of::<Exp16Params>();

    let encode_sort = |cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>| {
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

        // Combined histogram (always uses 4096-element tiles regardless of partition tile size)
        let hist_tile_count = n.div_ceil(TILE_SIZE);
        let hist_params = Exp16Params {
            element_count: n as u32,
            num_tiles: hist_tile_count as u32,
            num_tgs: hist_tile_count as u32,
            shift: 0,
            pass: 0,
        };
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_combined_hist);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&hist_params as *const Exp16Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: hist_tile_count, height: 1, depth: 1 }, tg_size);
        enc.endEncoding();

        // Global prefix sum
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        // 4 passes: zero_status + partition_v2
        for pass in 0..NUM_PASSES as u32 {
            let params = Exp16Params {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                num_tgs: num_tiles as u32,
                shift: pass * 8,
                pass,
            };

            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };

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

            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_partition_v2);
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

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }

    let cmd = ctx.command_buffer();
    encode_sort(&cmd);
    cmd.commit();
    cmd.waitUntilCompleted();

    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! v2 radix {}: {} / {} mismatched ({:.1}%)",
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

    // Benchmark
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

fn bench_8bit_perpass(
    ctx: &MetalContext,
    pso_combined_hist: &Pso,
    pso_global_prefix: &Pso,
    pso_zero_status: &Pso,
    pso_partition: &Pso,
    n: usize,
) {
    let num_tiles = n.div_ceil(TILE_SIZE);

    let input: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };

    let buf_input = alloc_buffer_with_data(&ctx.device, &input);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_global_hist = alloc_buffer(&ctx.device, NUM_PASSES * NUM_BINS * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * NUM_BINS * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
    let hist_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * NUM_BINS).div_ceil(THREADS_PER_TG);
    let params_size = std::mem::size_of::<Exp16Params>();

    let runs = 20;
    let warmup = 3;

    // Collect per-phase times: [hist+prefix, pass0, pass1, pass2, pass3]
    let mut phase_times: Vec<Vec<f64>> = vec![Vec::new(); 5];

    for iter in 0..(warmup + runs) {
        // Reset input
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
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

        // Phase 0: Combined histogram + global prefix (separate cmd buf)
        let cmd = ctx.command_buffer();
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

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= warmup {
            phase_times[0].push(gpu_elapsed_ms(&cmd));
        }

        // Passes 0-3: each in separate cmd buf
        for pass in 0..NUM_PASSES as u32 {
            let params = Exp16Params {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                num_tgs: num_tiles as u32,
                shift: pass * 8,
                pass,
            };

            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };

            // Zero tile_status
            unsafe {
                std::ptr::write_bytes(
                    buf_tile_status.contents().as_ptr() as *mut u8, 0,
                    num_tiles * NUM_BINS * 4,
                );
                std::ptr::write_bytes(
                    buf_counters.contents().as_ptr() as *mut u8, 0, 4,
                );
            }

            let cmd = ctx.command_buffer();
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

            cmd.commit();
            cmd.waitUntilCompleted();
            if iter >= warmup {
                phase_times[1 + pass as usize].push(gpu_elapsed_ms(&cmd));
            }
        }
    }

    // Print results
    println!("    Per-phase timing @ {}M (median of {} runs):", n / 1_000_000, runs);
    let labels = ["hist+pfx", "pass 0  ", "pass 1  ", "pass 2  ", "pass 3  "];
    let mut total = 0.0;
    for (i, label) in labels.iter().enumerate() {
        phase_times[i].sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&phase_times[i], 50.0);
        let bw = if i == 0 {
            // hist reads n elements once
            (n * 4) as f64 / p50 / 1e6 // GB/s
        } else {
            // each pass reads + writes n elements
            (n * 4 * 2) as f64 / p50 / 1e6
        };
        total += p50;
        println!("      {}: {:>6.3} ms  ({:.0} GB/s)", label, p50, bw);
    }
    let implied = n as f64 / total / 1000.0;
    println!("      total:    {:>6.3} ms  ({:.0} Mkeys/s implied)", total, implied);
    println!("      overhead: {:>6.3} ms  (vs {:.3} ms theoretical @ 245 GB/s)",
        total - (n as f64 * 4.0 * 9.0 / 245e9 * 1000.0),
        n as f64 * 4.0 * 9.0 / 245e9 * 1000.0);
}

// ═══════════════════════════════════════════════════════════════════
// 11-bit 3-pass radix sort benchmark
// ═══════════════════════════════════════════════════════════════════

const V5_MAX_BINS: usize = 2048;
const V5_NUM_PASSES: usize = 3;
const V5_TILE_SIZE_3P: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy)]
struct V5Params {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    mask: u32,
    pass: u32,
}

fn bench_3pass(
    ctx: &MetalContext,
    pso_hist: &Pso,
    pso_prefix: &Pso,
    pso_zero: &Pso,
    pso_partition: &Pso,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(V5_TILE_SIZE_3P);

    // 11+11+10 = 32 bits
    let shifts: [u32; 3] = [0, 11, 22];
    let masks: [u32; 3] = [0x7FF, 0x7FF, 0x3FF];

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
    let buf_global_hist = alloc_buffer(&ctx.device, V5_NUM_PASSES * V5_MAX_BINS * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * V5_MAX_BINS * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
    let hist_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let zero_entries = (num_tiles * V5_MAX_BINS) as u32;
    let zero_tg_count = (zero_entries as usize).div_ceil(THREADS_PER_TG);

    let v5_params_size = std::mem::size_of::<V5Params>();

    let encode_sort = |cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>| {
        // Zero global histogram
        unsafe {
            std::ptr::write_bytes(
                buf_global_hist.contents().as_ptr() as *mut u8, 0,
                V5_NUM_PASSES * V5_MAX_BINS * 4,
            );
        }

        // Encoder 1: Combined histogram (all 3 passes in one read)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_hist);
        let element_count = n as u32;
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&element_count as *const u32 as *mut _).unwrap(),
                4, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_tgs, tg_size);
        enc.endEncoding();

        // Encoder 2: Global prefix sum (in-place)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        // 3 passes: zero_status + partition with ping-pong
        for pass in 0..V5_NUM_PASSES as u32 {
            // Zero tile_status
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_zero);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::new(&zero_entries as *const u32 as *mut _).unwrap(),
                    4, 1,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: zero_tg_count, height: 1, depth: 1 }, tg_size,
            );
            enc.endEncoding();

            let params = V5Params {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
                shift: shifts[pass as usize],
                mask: masks[pass as usize],
                pass,
            };

            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };

            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_partition);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const V5Params as *mut _).unwrap(),
                    v5_params_size, 4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tiles, height: 1, depth: 1 }, tg_size,
            );
            enc.endEncoding();
        }
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }

    let cmd = ctx.command_buffer();
    encode_sort(&cmd);
    cmd.commit();
    cmd.waitUntilCompleted();

    // After 3 passes (odd): pass 0 (A→B), pass 1 (B→A), pass 2 (A→B)
    // Result is in buf_b
    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_b, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! 3-pass radix {}: {} / {} mismatched ({:.1}%)",
            n, mismatches, n, mismatches as f64 / n as f64 * 100.0
        );
        let mut shown = 0;
        for (i, (got, exp)) in results.iter().zip(expected.iter()).enumerate() {
            if got != exp && shown < 5 {
                println!("         [{}] got=0x{:08X} exp=0x{:08X}", i, got, exp);
                shown += 1;
            }
        }
    }

    // Benchmark
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
    println!("8-bit radix, 4 passes, 256 bins, {} elem/tile", TILE_SIZE);
    println!("Per-SG atomic histogram, decoupled lookback (256 threads)\n");

    let pso_combined_hist = ctx.make_pipeline("exp16_combined_histogram");
    let pso_global_prefix = ctx.make_pipeline("exp16_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");

    println!("  ── 8-bit Radix Sort (4 passes, 256 bins) ──");
    // Test small sizes to measure SLC-resident throughput
    for &n in &[62_500, 250_000, 1_000_000, 4_000_000, 16_000_000] {
        let label = if n >= 1_000_000 { format!("{}M", n / 1_000_000) }
                    else { format!("{}K", n / 1_000) };
        let (times, correct) = bench_8bit(
            ctx, &pso_combined_hist, &pso_global_prefix,
            &pso_zero_status, &pso_partition, n,
        );
        print_stats(&label, n, &times, correct);
    }

    // ── 11-bit 3-pass radix sort ──
    println!("\n  ── 11-bit 3-Pass Radix Sort (2048 bins, 3 passes) ──");
    let pso_3pass_hist = ctx.make_pipeline("exp16_3pass_histogram");
    let pso_3pass_prefix = ctx.make_pipeline("exp16_3pass_prefix");
    let pso_3pass_zero = ctx.make_pipeline("exp16_3pass_zero");
    let pso_3pass_partition = ctx.make_pipeline("exp16_3pass_partition");

    for &n in &[1_000_000usize, 4_000_000, 16_000_000] {
        let label = if n >= 1_000_000 { format!("{}M", n / 1_000_000) }
                    else { format!("{}K", n / 1_000) };
        let (times, correct) = bench_3pass(
            ctx, &pso_3pass_hist, &pso_3pass_prefix,
            &pso_3pass_zero, &pso_3pass_partition, n,
        );
        print_stats(&label, n, &times, correct);
    }

    println!("\n  ── Per-Pass Diagnostic (16M) ──");
    bench_8bit_perpass(
        ctx, &pso_combined_hist, &pso_global_prefix,
        &pso_zero_status, &pso_partition, 16_000_000,
    );

    println!("\n  ── Scatter Penalty Diagnostic (16M) ──");
    let n = 16_000_000usize;
    let pso_copy = ctx.make_pipeline("exp16_diag_copy");
    let pso_scatter = ctx.make_pipeline("exp16_diag_scatter");
    let pso_noscat = ctx.make_pipeline("exp16_diag_noscat");

    let buf_src = alloc_buffer(&ctx.device, n * 4);
    let buf_dst = alloc_buffer(&ctx.device, n * 4);

    // Create random permutation for scatter test
    let perm: Vec<u32> = {
        use rand::Rng;
        let mut v: Vec<u32> = (0..n as u32).collect();
        let mut rng = rand::thread_rng();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            v.swap(i, j);
        }
        v
    };
    let buf_perm = alloc_buffer_with_data(&ctx.device, &perm);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let grid_tgs = MTLSize { width: n.div_ceil(THREADS_PER_TG), height: 1, depth: 1 };

    // Benchmark sequential copy
    let mut copy_times = Vec::new();
    for iter in 0..25 {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_copy);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= 5 { copy_times.push(gpu_elapsed_ms(&cmd)); }
    }
    copy_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let copy_p50 = percentile(&copy_times, 50.0);
    let copy_bw = (n * 4 * 2) as f64 / copy_p50 / 1e6;
    println!("    seq copy:     {:>6.3} ms  ({:.0} GB/s)", copy_p50, copy_bw);

    // Benchmark random scatter
    let mut scat_times = Vec::new();
    for iter in 0..25 {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_scatter);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_perm.as_ref()), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= 5 { scat_times.push(gpu_elapsed_ms(&cmd)); }
    }
    scat_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let scat_p50 = percentile(&scat_times, 50.0);
    let scat_bw = (n * 4 * 2) as f64 / scat_p50 / 1e6;
    println!("    rng scatter:  {:>6.3} ms  ({:.0} GB/s)  ({:.1}x slower)",
        scat_p50, scat_bw, scat_p50 / copy_p50);

    // Benchmark partition without scatter (load+hist+lookback only)
    let num_tiles = n.div_ceil(TILE_SIZE);
    let buf_global_hist2 = alloc_buffer(&ctx.device, NUM_PASSES * NUM_BINS * 4);
    let buf_tile_status2 = alloc_buffer(&ctx.device, num_tiles * NUM_BINS * 4);
    let buf_counters2 = alloc_buffer(&ctx.device, 4);

    // First run the histogram to populate global_hist
    {
        let input: Vec<u32> = {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            (0..n).map(|_| rng.gen::<u32>()).collect()
        };
        let buf_input = alloc_buffer_with_data(&ctx.device, &input);
        unsafe {
            let s = buf_input.contents().as_ptr() as *const u8;
            let d = buf_src.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(s, d, n * 4);
            std::ptr::write_bytes(buf_global_hist2.contents().as_ptr() as *mut u8, 0, NUM_PASSES * NUM_BINS * 4);
        }

        let base_params = Exp16Params { element_count: n as u32, num_tiles: num_tiles as u32, num_tgs: num_tiles as u32, shift: 0, pass: 0 };
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_combined_hist);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist2.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&base_params as *const Exp16Params as *mut _).unwrap(),
                std::mem::size_of::<Exp16Params>(), 2,
            );
        }
        let hist_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_tgs, tg_size);
        enc.endEncoding();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_global_prefix);
        unsafe { enc.setBuffer_offset_atIndex(Some(buf_global_hist2.as_ref()), 0, 0); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1}, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    // Benchmark no-scatter partition (pass 0 only)
    let mut noscat_times = Vec::new();
    for iter in 0..25 {
        unsafe {
            std::ptr::write_bytes(buf_tile_status2.contents().as_ptr() as *mut u8, 0, num_tiles * NUM_BINS * 4);
            std::ptr::write_bytes(buf_counters2.contents().as_ptr() as *mut u8, 0, 4);
        }
        let params = Exp16Params { element_count: n as u32, num_tiles: num_tiles as u32, num_tgs: num_tiles as u32, shift: 0, pass: 0 };
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_noscat);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_status2.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_counters2.as_ref()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist2.as_ref()), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp16Params as *mut _).unwrap(),
                std::mem::size_of::<Exp16Params>(), 5,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: num_tiles, height: 1, depth: 1 }, tg_size,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= 5 { noscat_times.push(gpu_elapsed_ms(&cmd)); }
    }
    noscat_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let noscat_p50 = percentile(&noscat_times, 50.0);
    println!("    no-scatter:   {:>6.3} ms  (load+hist+lookback only)", noscat_p50);

    // Benchmark random gather (read random, write sequential)
    let pso_gather = ctx.make_pipeline("exp16_diag_gather");
    let mut gather_times = Vec::new();
    for iter in 0..25 {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_gather);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_perm.as_ref()), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= 5 { gather_times.push(gpu_elapsed_ms(&cmd)); }
    }
    gather_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let gather_p50 = percentile(&gather_times, 50.0);
    let gather_bw = (n * 4 * 2) as f64 / gather_p50 / 1e6;
    println!("    rng gather:   {:>6.3} ms  ({:.0} GB/s)  ({:.1}x slower)",
        gather_p50, gather_bw, gather_p50 / copy_p50);

    // Benchmark blocked gather (32-element cache-line-aligned reads)
    let pso_gather_blocked = ctx.make_pipeline("exp16_diag_gather_blocked");
    let num_blocks = n / 32;
    let block_starts: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // Random block-aligned start positions within the buffer
        (0..num_blocks).map(|_| (rng.gen_range(0..n / 32) * 32) as u32).collect()
    };
    let buf_block_starts = alloc_buffer_with_data(&ctx.device, &block_starts);
    let n_u32 = n as u32;
    let mut blocked_gather_times = Vec::new();
    for iter in 0..25 {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_gather_blocked);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_block_starts.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(),
                4, 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= 5 { blocked_gather_times.push(gpu_elapsed_ms(&cmd)); }
    }
    blocked_gather_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let blocked_p50 = percentile(&blocked_gather_times, 50.0);
    let blocked_bw = (n * 4 * 2) as f64 / blocked_p50 / 1e6;
    println!("    blk gather:   {:>6.3} ms  ({:.0} GB/s)  ({:.1}x slower)  [32-elem blocks]",
        blocked_p50, blocked_bw, blocked_p50 / copy_p50);

    // Benchmark 256-bin structured scatter (radix sort pattern)
    let pso_scatter_binned = ctx.make_pipeline("exp16_diag_scatter_binned");
    // Create a radix-sort-like scatter pattern: partition by byte 0 into 256 bins
    let bin_offsets: Vec<u32> = {
        let data: Vec<u32> = {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            (0..n).map(|_| rng.gen::<u32>()).collect()
        };
        // Count per-bin histogram
        let mut hist = vec![0u32; 256];
        for &v in &data { hist[(v & 0xFF) as usize] += 1; }
        // Prefix sum
        let mut prefix = vec![0u32; 256];
        let mut sum = 0u32;
        for i in 0..256 { prefix[i] = sum; sum += hist[i]; }
        // Compute per-element destination
        let mut counters = prefix.clone();
        let mut offsets = vec![0u32; n];
        for i in 0..n {
            let bin = (data[i] & 0xFF) as usize;
            offsets[i] = counters[bin];
            counters[bin] += 1;
        }
        offsets
    };
    let buf_bin_offsets = alloc_buffer_with_data(&ctx.device, &bin_offsets);
    let mut binned_scat_times = Vec::new();
    for iter in 0..25 {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_scatter_binned);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bin_offsets.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(),
                4, 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= 5 { binned_scat_times.push(gpu_elapsed_ms(&cmd)); }
    }
    binned_scat_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let binned_p50 = percentile(&binned_scat_times, 50.0);
    let binned_bw = (n * 4 * 2) as f64 / binned_p50 / 1e6;
    println!("    256-bin scat: {:>6.3} ms  ({:.0} GB/s)  ({:.1}x slower)  [radix pattern]",
        binned_p50, binned_bw, binned_p50 / copy_p50);

    // Benchmark bitonic sort of tiles (no DRAM scatter)
    let pso_bitonic = ctx.make_pipeline("exp16_diag_bitonic_tile");
    let element_count_u32 = n as u32;
    let bitonic_tgs = MTLSize { width: n.div_ceil(4096), height: 1, depth: 1 };
    let mut bitonic_times = Vec::new();
    for iter in 0..25 {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_bitonic);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&element_count_u32 as *const u32 as *mut _).unwrap(),
                4, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(bitonic_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= 5 { bitonic_times.push(gpu_elapsed_ms(&cmd)); }
    }
    bitonic_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bitonic_p50 = percentile(&bitonic_times, 50.0);
    let bitonic_mkeys = n as f64 / bitonic_p50 / 1000.0;
    println!("    bitonic tile:  {:>5.3} ms  ({:.0} Mkeys/s tile-sort, {:.0} GB/s)",
        bitonic_p50, bitonic_mkeys, (n * 4 * 2) as f64 / bitonic_p50 / 1e6);

    // Summary
    let scatter_penalty = scat_p50 / copy_p50;
    let noscat_bw = (n * 4) as f64 / noscat_p50 / 1e6;
    println!("\n    Summary:");
    println!("      Scatter penalty: {:.2}x (random vs sequential writes)", scatter_penalty);
    println!("      Gather penalty:  {:.2}x (random vs sequential reads)", gather_p50 / copy_p50);
    println!("      Gather vs scatter: {:.2}x ({} is faster)",
        if gather_p50 < scat_p50 { scat_p50 / gather_p50 } else { gather_p50 / scat_p50 },
        if gather_p50 < scat_p50 { "GATHER" } else { "scatter" });
    println!("      No-scatter partition: {:.3} ms = {:.0} GB/s read-effective", noscat_p50, noscat_bw);
    println!("      Bitonic tile sort: {:.3} ms for {} tiles of 4096", bitonic_p50, n / 4096);
    println!("      If scatter were free: 4 × {:.3} ms + hist {:.3} ms = {:.3} ms = {:.0} Mkeys/s",
        noscat_p50, copy_p50 / 2.0,
        4.0 * noscat_p50 + copy_p50 / 2.0,
        n as f64 / (4.0 * noscat_p50 + copy_p50 / 2.0) / 1000.0);

    // Hybrid approach estimate
    let merge_passes = (n as f64 / 4096.0).log2().ceil() as u32;
    let merge_pass_time = copy_p50; // sequential read+write = copy bandwidth
    println!("\n    Hybrid estimate (bitonic tiles + merge):");
    println!("      Tile sort: {:.3} ms ({} tiles)", bitonic_p50, n / 4096);
    println!("      Merge: {} passes × {:.3} ms = {:.3} ms",
        merge_passes, merge_pass_time, merge_passes as f64 * merge_pass_time);
    let hybrid_total = bitonic_p50 + merge_passes as f64 * merge_pass_time;
    println!("      Total: {:.3} ms = {:.0} Mkeys/s", hybrid_total, n as f64 / hybrid_total / 1000.0);
    println!("      vs radix 4-pass: {:.3} ms = {:.0} Mkeys/s", 5.0 * copy_p50 + noscat_p50 * 4.0,
        n as f64 / (5.0 * copy_p50 + noscat_p50 * 4.0) / 1000.0);
}
