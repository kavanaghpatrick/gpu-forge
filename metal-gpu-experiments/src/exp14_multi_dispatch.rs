//! Experiment 14: Multi-Dispatch Radix Sort (Phase 1)
//!
//! 3 kernels per pass: histogram → prefix_scan → scatter
//! Key optimization: local sort in TG shared memory for coalesced scatter
//! Phase 1: scalar loads, serial O(32) rank (TILE_SIZE=256)
//!
//! Compare against exp13 (persistent kernel, random scatter) to measure
//! the improvement from coalesced writes alone.

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
use std::ptr::NonNull;

const TILE_SIZE: usize = 256;
const V2_TILE_SIZE: usize = 1024;
const NUM_BINS: usize = 256;
const V4_NUM_BINS: usize = 16;
const V5_TILE_SIZE: usize = 2048;
const THREADS_PER_TG: usize = 256;
const NUM_PASSES: usize = 4; // 8-bit radix, 32-bit keys
const V4_NUM_PASSES: usize = 8; // 4-bit radix, 32-bit keys
const V8_NUM_BINS: usize = 32;
const V8_TILE_SIZE: usize = 2048;
const V8_NUM_PASSES: usize = 7; // 5-bit radix: ceil(32/5) = 7 passes
const WARMUP: usize = 5;
const RUNS: usize = 50;

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp14Params {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    pass: u32,
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
    let throughput = n as f64 / p50 / 1000.0;
    let status = if correct { "ok" } else { "FAIL" };
    println!(
        "    {:>4}: {:>7.3} ms  {:>6.0} Mkeys/s  {:>4}  [p5={:.2} p95={:.2} spread={:.0}%]",
        label, p50, throughput, status, p5, p95, spread
    );
}

/// Run one sort: correctness check + benchmark.
fn bench_u32(
    ctx: &MetalContext,
    pso_histogram: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_prefix_scan: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_global_prefix: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(TILE_SIZE);

    // Generate random input + CPU reference sort
    let input: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };
    let mut expected = input.clone();
    expected.sort();

    // Allocate GPU buffers
    let buf_input = alloc_buffer_with_data(&ctx.device, &input);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let tile_hist_bytes = num_tiles * NUM_BINS * 4;
    let buf_tile_hist = alloc_buffer(&ctx.device, tile_hist_bytes);
    let buf_global_hist = alloc_buffer(&ctx.device, NUM_BINS * 4);

    let tg_size = MTLSize {
        width: THREADS_PER_TG,
        height: 1,
        depth: 1,
    };
    let tile_tgs = MTLSize {
        width: num_tiles,
        height: 1,
        depth: 1,
    };
    let one_tg = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    // Helper: run one sort pass
    let run_sort = |src_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                    dst_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                    cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                    pass: u32| {
        let params = Exp14Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: pass * 8,
            pass,
        };
        let params_size = std::mem::size_of::<Exp14Params>();

        // Dispatch 1: Histogram
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_histogram);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size,
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();

        // Dispatch 2: Prefix Scan (256 TGs, one per bin)
        let scan_tgs = MTLSize { width: NUM_BINS, height: 1, depth: 1 };
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_prefix_scan);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size,
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(scan_tgs, tg_size);
        enc.endEncoding();

        // Dispatch 3: Global Prefix (1 TG, converts counts → exclusive prefix)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        // Dispatch 4: Scatter
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size,
                4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
    };

    // ── Correctness check ──────────────────────────────────────
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }

    let cmd = ctx.command_buffer();
    for pass in 0..NUM_PASSES as u32 {
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a.as_ref(), buf_b.as_ref())
        } else {
            (buf_b.as_ref(), buf_a.as_ref())
        };
        run_sort(src_buf, dst_buf, &cmd, pass);
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    // 4 passes: 0:a→b, 1:b→a, 2:a→b, 3:b→a → result in buf_a
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
        // Print first few mismatches for debugging
        let mut shown = 0;
        for (i, (got, exp)) in results.iter().zip(expected.iter()).enumerate() {
            if got != exp && shown < 5 {
                println!("         [{}] got={} exp={}", i, got, exp);
                shown += 1;
            }
        }
    }

    // ── Benchmark ──────────────────────────────────────────────
    let mut times = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
        }

        let cmd = ctx.command_buffer();
        for pass in 0..NUM_PASSES as u32 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };
            run_sort(src_buf, dst_buf, &cmd, pass);
        }
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

/// V2 bench: TILE_SIZE=1024, 4 elements/thread
fn bench_u32_v2(
    ctx: &MetalContext,
    pso_histogram_v2: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_prefix_scan: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_global_prefix: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter_v2: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
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
    let tile_hist_bytes = num_tiles * NUM_BINS * 4;
    let buf_tile_hist = alloc_buffer(&ctx.device, tile_hist_bytes);
    let buf_global_hist = alloc_buffer(&ctx.device, NUM_BINS * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let scan_tgs = MTLSize { width: NUM_BINS, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };

    let run_sort_v2 = |src_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       dst_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                       pass: u32| {
        let params = Exp14Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: pass * 8,
            pass,
        };
        let params_size = std::mem::size_of::<Exp14Params>();

        // Dispatch 1: V2 Histogram (1024 elements/tile)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_histogram_v2);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();

        // Dispatch 2: Prefix Scan (same kernel, fewer tiles)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_prefix_scan);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(scan_tgs, tg_size);
        enc.endEncoding();

        // Dispatch 3: Global Prefix
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        // Dispatch 4: V2 Scatter (1024 elements/tile)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter_v2);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }

    let cmd = ctx.command_buffer();
    for pass in 0..NUM_PASSES as u32 {
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a.as_ref(), buf_b.as_ref())
        } else {
            (buf_b.as_ref(), buf_a.as_ref())
        };
        run_sort_v2(src_buf, dst_buf, &cmd, pass);
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!("      !! V2 u32 {}: {} / {} mismatched ({:.1}%)",
            n, mismatches, n, mismatches as f64 / n as f64 * 100.0);
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
        for pass in 0..NUM_PASSES as u32 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };
            run_sort_v2(src_buf, dst_buf, &cmd, pass);
        }
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

/// V3 bench: fused histogram + decoupled lookback + scatter (single dispatch/pass)
fn bench_u32_v3(
    ctx: &MetalContext,
    pso_sort_v3: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
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
    let tile_status_bytes = num_tiles * NUM_BINS * 4;
    // One tile_status buffer per pass (avoids GPU-side clearing)
    let buf_tile_status: Vec<_> = (0..NUM_PASSES)
        .map(|_| alloc_buffer(&ctx.device, tile_status_bytes))
        .collect();

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };

    // Helper: zero all tile_status buffers from CPU
    let clear_tile_status = || {
        for ts in &buf_tile_status {
            unsafe {
                std::ptr::write_bytes(
                    ts.contents().as_ptr() as *mut u8,
                    0,
                    tile_status_bytes,
                );
            }
        }
    };

    let run_sort_v3 = |src_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       dst_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       tile_status_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                       pass: u32| {
        let params = Exp14Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: pass * 8,
            pass,
        };
        let params_size = std::mem::size_of::<Exp14Params>();

        // Single dispatch: fused histogram + lookback + scatter
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_sort_v3);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(tile_status_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size,
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }
    clear_tile_status();

    let cmd = ctx.command_buffer();
    for pass in 0..NUM_PASSES as u32 {
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a.as_ref(), buf_b.as_ref())
        } else {
            (buf_b.as_ref(), buf_a.as_ref())
        };
        run_sort_v3(src_buf, dst_buf, buf_tile_status[pass as usize].as_ref(), &cmd, pass);
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! V3 u32 {}: {} / {} mismatched ({:.1}%)",
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
        clear_tile_status();

        let cmd = ctx.command_buffer();
        for pass in 0..NUM_PASSES as u32 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };
            run_sort_v3(src_buf, dst_buf, buf_tile_status[pass as usize].as_ref(), &cmd, pass);
        }
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

/// V3 bench: 3 dispatches/pass (upsweep_v3 + global_prefix + scatter_v2)
/// Fuses histogram + prefix_scan into one lookback dispatch.
fn bench_u32_v3_3disp(
    ctx: &MetalContext,
    pso_upsweep_v3: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_global_prefix: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter_v2: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
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
    let tile_hist_bytes = num_tiles * NUM_BINS * 4;
    let buf_tile_hist = alloc_buffer(&ctx.device, tile_hist_bytes);
    let buf_global_hist = alloc_buffer(&ctx.device, NUM_BINS * 4);
    let tile_status_bytes = num_tiles * NUM_BINS * 4;
    // One tile_status buffer per pass (avoids GPU-side clearing)
    let buf_tile_status: Vec<_> = (0..NUM_PASSES)
        .map(|_| alloc_buffer(&ctx.device, tile_status_bytes))
        .collect();

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };

    let clear_tile_status = || {
        for ts in &buf_tile_status {
            unsafe {
                std::ptr::write_bytes(ts.contents().as_ptr() as *mut u8, 0, tile_status_bytes);
            }
        }
    };

    let run_sort_v3 = |src_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       dst_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       tile_status_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                       pass: u32| {
        let params = Exp14Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: pass * 8,
            pass,
        };
        let params_size = std::mem::size_of::<Exp14Params>();

        // Dispatch 1: Fused histogram + lookback (upsweep)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_upsweep_v3);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(tile_status_buf), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();

        // Dispatch 2: Global prefix (1 TG)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        // Dispatch 3: Scatter (same V2 scatter)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter_v2);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }
    clear_tile_status();

    let cmd = ctx.command_buffer();
    for pass in 0..NUM_PASSES as u32 {
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a.as_ref(), buf_b.as_ref())
        } else {
            (buf_b.as_ref(), buf_a.as_ref())
        };
        run_sort_v3(src_buf, dst_buf, buf_tile_status[pass as usize].as_ref(), &cmd, pass);
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! V3-3disp u32 {}: {} / {} mismatched ({:.1}%)",
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
        clear_tile_status();

        let cmd = ctx.command_buffer();
        for pass in 0..NUM_PASSES as u32 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };
            run_sort_v3(src_buf, dst_buf, buf_tile_status[pass as usize].as_ref(), &cmd, pass);
        }
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

/// Profile individual dispatch times by using separate command buffers per kernel.
fn profile_dispatches(
    ctx: &MetalContext,
    pso_histogram: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_prefix_scan: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_global_prefix: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) {
    let num_tiles = n.div_ceil(TILE_SIZE);

    let input: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };

    let buf_src = alloc_buffer_with_data(&ctx.device, &input);
    let buf_dst = alloc_buffer(&ctx.device, n * 4);
    let buf_tile_hist = alloc_buffer(&ctx.device, num_tiles * NUM_BINS * 4);
    let buf_global_hist = alloc_buffer(&ctx.device, NUM_BINS * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let scan_tgs = MTLSize { width: NUM_BINS, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };

    let params = Exp14Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        shift: 0,
        pass: 0,
    };
    let params_size = std::mem::size_of::<Exp14Params>();

    const PROFILE_RUNS: usize = 20;
    let mut t_hist = Vec::new();
    let mut t_scan = Vec::new();
    let mut t_gpfx = Vec::new();
    let mut t_scat = Vec::new();

    for _ in 0..PROFILE_RUNS {
        // Histogram
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_histogram);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_hist.push(gpu_elapsed_ms(&cmd));

        // Prefix Scan
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_prefix_scan);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(scan_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_scan.push(gpu_elapsed_ms(&cmd));

        // Global Prefix
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_gpfx.push(gpu_elapsed_ms(&cmd));

        // Scatter
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_scat.push(gpu_elapsed_ms(&cmd));
    }

    t_hist.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_scan.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_gpfx.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_scat.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let h = percentile(&t_hist, 50.0);
    let s = percentile(&t_scan, 50.0);
    let g = percentile(&t_gpfx, 50.0);
    let sc = percentile(&t_scat, 50.0);
    let total = h + s + g + sc;

    println!("    histogram:    {:>7.3} ms  ({:>4.1}%)", h, h / total * 100.0);
    println!("    prefix_scan:  {:>7.3} ms  ({:>4.1}%)", s, s / total * 100.0);
    println!("    global_prefix:{:>7.3} ms  ({:>4.1}%)", g, g / total * 100.0);
    println!("    scatter:      {:>7.3} ms  ({:>4.1}%)", sc, sc / total * 100.0);
    println!("    ─────────────────────────────────");
    println!("    sum (1 pass):  {:>6.3} ms", total);
    println!("    × 4 passes:   {:>6.3} ms", total * 4.0);
    println!("    actual bench:  ~3.05 ms (includes encoder transition overhead)");
}

/// Profile V2 dispatch times (TILE_SIZE=1024)
fn profile_dispatches_v2(
    ctx: &MetalContext,
    pso_histogram_v2: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_prefix_scan: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_global_prefix: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter_v2: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) {
    let num_tiles = n.div_ceil(V2_TILE_SIZE);

    let input: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };

    let buf_src = alloc_buffer_with_data(&ctx.device, &input);
    let buf_dst = alloc_buffer(&ctx.device, n * 4);
    let buf_tile_hist = alloc_buffer(&ctx.device, num_tiles * NUM_BINS * 4);
    let buf_global_hist = alloc_buffer(&ctx.device, NUM_BINS * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let scan_tgs = MTLSize { width: NUM_BINS, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };

    let params = Exp14Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        shift: 0,
        pass: 0,
    };
    let params_size = std::mem::size_of::<Exp14Params>();

    const PROFILE_RUNS: usize = 20;
    let mut t_hist = Vec::new();
    let mut t_scan = Vec::new();
    let mut t_gpfx = Vec::new();
    let mut t_scat = Vec::new();

    for _ in 0..PROFILE_RUNS {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_histogram_v2);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_hist.push(gpu_elapsed_ms(&cmd));

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_prefix_scan);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(scan_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_scan.push(gpu_elapsed_ms(&cmd));

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_gpfx.push(gpu_elapsed_ms(&cmd));

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter_v2);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_scat.push(gpu_elapsed_ms(&cmd));
    }

    t_hist.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_scan.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_gpfx.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_scat.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let h = percentile(&t_hist, 50.0);
    let s = percentile(&t_scan, 50.0);
    let g = percentile(&t_gpfx, 50.0);
    let sc = percentile(&t_scat, 50.0);
    let total = h + s + g + sc;

    println!("    histogram_v2: {:>7.3} ms  ({:>4.1}%)  [{} TGs]", h, h / total * 100.0, num_tiles);
    println!("    prefix_scan:  {:>7.3} ms  ({:>4.1}%)  [{} TGs]", s, s / total * 100.0, NUM_BINS);
    println!("    global_prefix:{:>7.3} ms  ({:>4.1}%)", g, g / total * 100.0);
    println!("    scatter_v2:   {:>7.3} ms  ({:>4.1}%)  [{} TGs]", sc, sc / total * 100.0, num_tiles);
    println!("    ─────────────────────────────────");
    println!("    sum (1 pass):  {:>6.3} ms", total);
    println!("    × 4 passes:   {:>6.3} ms", total * 4.0);
    let throughput = n as f64 / (total * 4.0) / 1000.0;
    println!("    projected:    {:>6.0} Mkeys/s", throughput);
}

/// V4 bench: 4-bit radix (16 bins, 8 passes)
fn bench_u32_v4(
    ctx: &MetalContext,
    pso_histogram_v4: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_prefix_scan_v4: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_global_prefix_v4: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter_v4: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
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
    let tile_hist_bytes = num_tiles * V4_NUM_BINS * 4;
    let buf_tile_hist = alloc_buffer(&ctx.device, tile_hist_bytes);
    let buf_global_hist = alloc_buffer(&ctx.device, V4_NUM_BINS * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let scan_tgs = MTLSize { width: V4_NUM_BINS, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };

    let run_sort_v4 = |src_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       dst_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                       pass: u32| {
        let params = Exp14Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: pass * 4, // 4-bit radix
            pass,
        };
        let params_size = std::mem::size_of::<Exp14Params>();

        // Dispatch 1: V4 Histogram
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_histogram_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();

        // Dispatch 2: V4 Prefix Scan (16 TGs)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_prefix_scan_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(scan_tgs, tg_size);
        enc.endEncoding();

        // Dispatch 3: V4 Global Prefix (1 TG)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        // Dispatch 4: V4 Scatter
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }

    let cmd = ctx.command_buffer();
    for pass in 0..V4_NUM_PASSES as u32 {
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a.as_ref(), buf_b.as_ref())
        } else {
            (buf_b.as_ref(), buf_a.as_ref())
        };
        run_sort_v4(src_buf, dst_buf, &cmd, pass);
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    // 8 passes: 0:a→b, 1:b→a, ..., 7:b→a → result in buf_a
    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! V4 u32 {}: {} / {} mismatched ({:.1}%)",
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
        for pass in 0..V4_NUM_PASSES as u32 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };
            run_sort_v4(src_buf, dst_buf, &cmd, pass);
        }
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

/// V5 bench: 4-bit radix, 8 elements/thread (2048/tile)
fn bench_u32_v5(
    ctx: &MetalContext,
    pso_histogram_v5: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_prefix_scan_v4: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_global_prefix_v4: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter_v5: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(V5_TILE_SIZE);

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
    let tile_hist_bytes = num_tiles * V4_NUM_BINS * 4;
    let buf_tile_hist = alloc_buffer(&ctx.device, tile_hist_bytes);
    let buf_global_hist = alloc_buffer(&ctx.device, V4_NUM_BINS * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let scan_tgs = MTLSize { width: V4_NUM_BINS, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };

    let run_sort_v5 = |src_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       dst_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                       pass: u32| {
        let params = Exp14Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: pass * 4,
            pass,
        };
        let params_size = std::mem::size_of::<Exp14Params>();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_histogram_v5);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_prefix_scan_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(scan_tgs, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter_v5);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }

    let cmd = ctx.command_buffer();
    for pass in 0..V4_NUM_PASSES as u32 {
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a.as_ref(), buf_b.as_ref())
        } else {
            (buf_b.as_ref(), buf_a.as_ref())
        };
        run_sort_v5(src_buf, dst_buf, &cmd, pass);
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! V5 u32 {}: {} / {} mismatched ({:.1}%)",
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
        for pass in 0..V4_NUM_PASSES as u32 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };
            run_sort_v5(src_buf, dst_buf, &cmd, pass);
        }
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

/// V6 bench: fused 4-bit radix (single dispatch/pass, decoupled lookback)
fn bench_u32_v6(
    ctx: &MetalContext,
    pso_sort_v6: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(V5_TILE_SIZE);

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
    let tile_status_bytes = num_tiles * V4_NUM_BINS * 4;
    // One tile_status buffer per pass (avoids GPU-side clearing)
    let buf_tile_status: Vec<_> = (0..V4_NUM_PASSES)
        .map(|_| alloc_buffer(&ctx.device, tile_status_bytes))
        .collect();

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };

    let clear_tile_status = || {
        for ts in &buf_tile_status {
            unsafe {
                std::ptr::write_bytes(ts.contents().as_ptr() as *mut u8, 0, tile_status_bytes);
            }
        }
    };

    let run_sort_v6 = |src_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       dst_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       tile_status_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                       pass: u32| {
        let params = Exp14Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: pass * 4,
            pass,
        };
        let params_size = std::mem::size_of::<Exp14Params>();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_sort_v6);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(tile_status_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }
    clear_tile_status();

    let cmd = ctx.command_buffer();
    for pass in 0..V4_NUM_PASSES as u32 {
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a.as_ref(), buf_b.as_ref())
        } else {
            (buf_b.as_ref(), buf_a.as_ref())
        };
        run_sort_v6(src_buf, dst_buf, buf_tile_status[pass as usize].as_ref(), &cmd, pass);
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! V6 u32 {}: {} / {} mismatched ({:.1}%)",
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
        clear_tile_status();

        let cmd = ctx.command_buffer();
        for pass in 0..V4_NUM_PASSES as u32 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };
            run_sort_v6(src_buf, dst_buf, buf_tile_status[pass as usize].as_ref(), &cmd, pass);
        }
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

const V7_NUM_BINS: usize = 256;
const V7_NUM_PASSES: usize = 4; // 8-bit radix, 32-bit keys

/// V7 bench: 8-bit radix, 4 passes, 2 dispatches/pass, uint4 vectorized loads
/// Halves total bandwidth vs 4-bit/8-pass. 256 threads = 256 bins = full lookback util.
fn bench_u32_v7(
    ctx: &MetalContext,
    pso_upsweep_v7: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter_v7: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(V5_TILE_SIZE); // 2048 elems/tile

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
    let tile_hist_bytes = num_tiles * V7_NUM_BINS * 4;
    let buf_tile_hist = alloc_buffer(&ctx.device, tile_hist_bytes);
    let buf_global_hist = alloc_buffer(&ctx.device, V7_NUM_BINS * 4);
    let tile_status_bytes = num_tiles * V7_NUM_BINS * 4;
    let buf_tile_status: Vec<_> = (0..V7_NUM_PASSES)
        .map(|_| alloc_buffer(&ctx.device, tile_status_bytes))
        .collect();

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };

    let clear_tile_status = || {
        for ts in &buf_tile_status {
            unsafe {
                std::ptr::write_bytes(ts.contents().as_ptr() as *mut u8, 0, tile_status_bytes);
            }
        }
    };

    let run_sort = |src_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                    dst_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                    tile_status_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                    cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                    pass: u32| {
        let params = Exp14Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: pass * 8, // 8-bit radix
            pass,
        };
        let params_size = std::mem::size_of::<Exp14Params>();

        // Dispatch 1: Fused upsweep
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_upsweep_v7);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(tile_status_buf), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();

        // Dispatch 2: Scatter
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter_v7);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }
    clear_tile_status();

    let cmd = ctx.command_buffer();
    for pass in 0..V7_NUM_PASSES as u32 {
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a.as_ref(), buf_b.as_ref())
        } else {
            (buf_b.as_ref(), buf_a.as_ref())
        };
        run_sort(src_buf, dst_buf, buf_tile_status[pass as usize].as_ref(), &cmd, pass);
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! V7 u32 {}: {} / {} mismatched ({:.1}%)",
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
        clear_tile_status();

        let cmd = ctx.command_buffer();
        for pass in 0..V7_NUM_PASSES as u32 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };
            run_sort(src_buf, dst_buf, buf_tile_status[pass as usize].as_ref(), &cmd, pass);
        }
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

/// V6-2disp bench: fused upsweep (histogram+lookback+global_prefix) + scatter
/// 2 dispatches per pass × 8 passes = 16 encoders, ONE cmd.commit()
fn bench_u32_v6_2disp(
    ctx: &MetalContext,
    pso_upsweep_v6: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter_v5: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(V5_TILE_SIZE);

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
    let tile_hist_bytes = num_tiles * V4_NUM_BINS * 4;
    let buf_tile_hist = alloc_buffer(&ctx.device, tile_hist_bytes);
    let buf_global_hist = alloc_buffer(&ctx.device, V4_NUM_BINS * 4);
    let tile_status_bytes = num_tiles * V4_NUM_BINS * 4;
    let buf_tile_status: Vec<_> = (0..V4_NUM_PASSES)
        .map(|_| alloc_buffer(&ctx.device, tile_status_bytes))
        .collect();

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };

    let clear_tile_status = || {
        for ts in &buf_tile_status {
            unsafe {
                std::ptr::write_bytes(ts.contents().as_ptr() as *mut u8, 0, tile_status_bytes);
            }
        }
    };

    let run_sort = |src_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                    dst_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                    tile_status_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                    cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                    pass: u32| {
        let params = Exp14Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: pass * 4,
            pass,
        };
        let params_size = std::mem::size_of::<Exp14Params>();

        // Dispatch 1: Fused upsweep (histogram + lookback + global prefix)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_upsweep_v6);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(tile_status_buf), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();

        // Dispatch 2: Scatter (reuses V5 scatter kernel)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter_v5);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }
    clear_tile_status();

    let cmd = ctx.command_buffer();
    for pass in 0..V4_NUM_PASSES as u32 {
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a.as_ref(), buf_b.as_ref())
        } else {
            (buf_b.as_ref(), buf_a.as_ref())
        };
        run_sort(src_buf, dst_buf, buf_tile_status[pass as usize].as_ref(), &cmd, pass);
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! V6-2disp u32 {}: {} / {} mismatched ({:.1}%)",
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
        clear_tile_status();

        let cmd = ctx.command_buffer();
        for pass in 0..V4_NUM_PASSES as u32 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };
            run_sort(src_buf, dst_buf, buf_tile_status[pass as usize].as_ref(), &cmd, pass);
        }
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

/// Profile V4 dispatch times (4-bit radix)
fn profile_dispatches_v4(
    ctx: &MetalContext,
    pso_histogram_v4: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_prefix_scan_v4: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_global_prefix_v4: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter_v4: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) {
    let num_tiles = n.div_ceil(V2_TILE_SIZE);

    let input: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };

    let buf_src = alloc_buffer_with_data(&ctx.device, &input);
    let buf_dst = alloc_buffer(&ctx.device, n * 4);
    let buf_tile_hist = alloc_buffer(&ctx.device, num_tiles * V4_NUM_BINS * 4);
    let buf_global_hist = alloc_buffer(&ctx.device, V4_NUM_BINS * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let scan_tgs = MTLSize { width: V4_NUM_BINS, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };

    let params = Exp14Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        shift: 0,
        pass: 0,
    };
    let params_size = std::mem::size_of::<Exp14Params>();

    const PROFILE_RUNS: usize = 20;
    let mut t_hist = Vec::new();
    let mut t_scan = Vec::new();
    let mut t_gpfx = Vec::new();
    let mut t_scat = Vec::new();

    for _ in 0..PROFILE_RUNS {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_histogram_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_hist.push(gpu_elapsed_ms(&cmd));

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_prefix_scan_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(scan_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_scan.push(gpu_elapsed_ms(&cmd));

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_gpfx.push(gpu_elapsed_ms(&cmd));

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        t_scat.push(gpu_elapsed_ms(&cmd));
    }

    t_hist.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_scan.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_gpfx.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_scat.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let h = percentile(&t_hist, 50.0);
    let s = percentile(&t_scan, 50.0);
    let g = percentile(&t_gpfx, 50.0);
    let sc = percentile(&t_scat, 50.0);
    let total = h + s + g + sc;

    println!("    histogram_v4: {:>7.3} ms  ({:>4.1}%)  [{} TGs]", h, h / total * 100.0, num_tiles);
    println!("    pfx_scan_v4:  {:>7.3} ms  ({:>4.1}%)  [{} TGs]", s, s / total * 100.0, V4_NUM_BINS);
    println!("    glb_prefix_v4:{:>7.3} ms  ({:>4.1}%)", g, g / total * 100.0);
    println!("    scatter_v4:   {:>7.3} ms  ({:>4.1}%)  [{} TGs]", sc, sc / total * 100.0, num_tiles);
    println!("    ─────────────────────────────────");
    println!("    sum (1 pass):  {:>6.3} ms", total);
    println!("    × 8 passes:   {:>6.3} ms", total * 8.0);
    let throughput = n as f64 / (total * 8.0) / 1000.0;
    println!("    projected:    {:>6.0} Mkeys/s", throughput);
}

/// V8 bench: 5-bit radix, 32 bins, 7 passes (4 dispatches/pass)
fn bench_u32_v8(
    ctx: &MetalContext,
    pso_histogram_v8: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_prefix_scan_v8: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_global_prefix_v8: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    pso_scatter_v8: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    n: usize,
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(V8_TILE_SIZE);

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
    let tile_hist_bytes = num_tiles * V8_NUM_BINS * 4;
    let buf_tile_hist = alloc_buffer(&ctx.device, tile_hist_bytes);
    let buf_global_hist = alloc_buffer(&ctx.device, V8_NUM_BINS * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let tile_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let scan_tgs = MTLSize { width: V8_NUM_BINS, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };

    let run_sort_v8 = |src_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       dst_buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
                       cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
                       pass: u32| {
        let params = Exp14Params {
            element_count: n as u32,
            num_tiles: num_tiles as u32,
            shift: pass * 5,
            pass,
        };
        let params_size = std::mem::size_of::<Exp14Params>();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_histogram_v8);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_prefix_scan_v8);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(scan_tgs, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_global_prefix_v8);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_scatter_v8);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hist.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp14Params as *mut _).unwrap(),
                params_size, 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tile_tgs, tg_size);
        enc.endEncoding();
    };

    // Correctness check
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }

    let cmd = ctx.command_buffer();
    for pass in 0..V8_NUM_PASSES as u32 {
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a.as_ref(), buf_b.as_ref())
        } else {
            (buf_b.as_ref(), buf_a.as_ref())
        };
        run_sort_v8(src_buf, dst_buf, &cmd, pass);
    }
    cmd.commit();
    cmd.waitUntilCompleted();

    // 7 passes (odd): last pass writes to buf_b, result is in buf_b
    let result_buf = if V8_NUM_PASSES % 2 == 0 { &buf_a } else { &buf_b };
    let results: Vec<u32> = unsafe { read_buffer_slice(result_buf, n) };
    let correct = results == expected;

    if !correct {
        let mismatches = results.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        println!(
            "      !! V8 u32 {}: {} / {} mismatched ({:.1}%)",
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
        for pass in 0..V8_NUM_PASSES as u32 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };
            run_sort_v8(src_buf, dst_buf, &cmd, pass);
        }
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
    println!("Experiment 14: Multi-Dispatch Radix Sort (Phase 1)");
    println!("{}", "=".repeat(60));
    println!("3 kernels/pass × 4 passes, coalesced scatter via local sort");
    println!("TILE_SIZE=256, serial rank, scalar loads\n");

    let pso_histogram = ctx.make_pipeline("exp14_histogram");
    let pso_prefix_scan = ctx.make_pipeline("exp14_prefix_scan");
    let pso_global_prefix = ctx.make_pipeline("exp14_global_prefix");
    let pso_scatter = ctx.make_pipeline("exp14_scatter");
    let pso_scatter_direct = ctx.make_pipeline("exp14_scatter_direct");
    let pso_scatter_ballot = ctx.make_pipeline("exp14_scatter_ballot");
    let pso_histogram_v2 = ctx.make_pipeline("exp14_histogram_v2");
    let pso_scatter_v2 = ctx.make_pipeline("exp14_scatter_v2");
    let pso_sort_v3 = ctx.make_pipeline("exp14_sort_v3");
    let pso_upsweep_v3 = ctx.make_pipeline("exp14_upsweep_v3");
    let pso_histogram_v4 = ctx.make_pipeline("exp14_histogram_v4");
    let pso_prefix_scan_v4 = ctx.make_pipeline("exp14_prefix_scan_v4");
    let pso_global_prefix_v4 = ctx.make_pipeline("exp14_global_prefix_v4");
    let pso_scatter_v4 = ctx.make_pipeline("exp14_scatter_v4");
    let pso_histogram_v5 = ctx.make_pipeline("exp14_histogram_v5");
    let pso_scatter_v5 = ctx.make_pipeline("exp14_scatter_v5");
    let pso_sort_v6 = ctx.make_pipeline("exp14_sort_v6");

    // ── GPU Quick Health Check ──────────────────────────────────
    {
        let buf = alloc_buffer(&ctx.device, NUM_BINS * 4);
        let tg = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
        let one = MTLSize { width: 1, height: 1, depth: 1 };
        // Warmup
        for _ in 0..5 {
            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_global_prefix);
            unsafe { enc.setBuffer_offset_atIndex(Some(buf.as_ref()), 0, 0); }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one, tg);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_global_prefix);
        unsafe { enc.setBuffer_offset_atIndex(Some(buf.as_ref()), 0, 0); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu_us = gpu_elapsed_ms(&cmd) * 1000.0;
        if gpu_us > 100.0 {
            println!("  ⚠ GPU THROTTLED: dispatch latency {:.0} us (expected <10 us)", gpu_us);
            println!("    GPU watchdog may be active — reboot to clear\n");
        } else {
            println!("  ✓ GPU healthy: dispatch latency {:.0} us\n", gpu_us);
        }
    }

    let pso_histogram_v8 = ctx.make_pipeline("exp14_histogram_v8");
    let pso_prefix_scan_v8 = ctx.make_pipeline("exp14_prefix_scan_v8");
    let pso_global_prefix_v8 = ctx.make_pipeline("exp14_global_prefix_v8");
    let pso_scatter_v8 = ctx.make_pipeline("exp14_scatter_v8");

    // ── V8: 5-bit radix, 7 passes (12.5% less bandwidth) ────────
    println!("  ── V8: 5-bit radix, 32 bins, 7 passes ──");
    for &n in &[1_000_000, 4_000_000, 16_000_000] {
        let label = format!("{}M", n / 1_000_000);
        let (times, correct) = bench_u32_v8(
            ctx, &pso_histogram_v8, &pso_prefix_scan_v8,
            &pso_global_prefix_v8, &pso_scatter_v8, n,
        );
        print_stats(&label, n, &times, correct);
    }

    // V5 reference (4-bit, 4 disp/pass)
    println!("\n  ── V5 ref: 4-bit, 16 bins, 8 passes ──");
    for &n in &[1_000_000, 4_000_000, 16_000_000] {
        let label = format!("{}M", n / 1_000_000);
        let (times, correct) = bench_u32_v5(
            ctx, &pso_histogram_v5, &pso_prefix_scan_v4,
            &pso_global_prefix_v4, &pso_scatter_v5, n,
        );
        print_stats(&label, n, &times, correct);
    }
}
