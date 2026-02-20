//! Experiment 21: Pre-Sort Scatter (Stehle-Jacobsen Technique)
//!
//! A/B test: MSD + 3×LSD hybrid sort where the inner scatter kernel
//! pre-sorts elements by digit in TG memory before writing to global.
//!
//! Variant A: exp21_inner_presort_scatter_v2 (digit-sorted TG memory → sequential write)
//! Variant B: exp21_inner_random_scatter (control — random scatter like exp17)
//!
//! Both use the same MSD phase (exp16_partition + decoupled lookback).
//! The ONLY difference is the inner scatter kernel.

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
use std::ptr::NonNull;

const WARMUP: usize = 5;
const RUNS: usize = 30;
const THREADS_PER_TG: usize = 256;
const TILE_SIZE: usize = 4096;
const MAX_TPB: usize = 17; // max tiles per bucket (ceil(62500/4096))
const NUM_BINS: usize = 256;

type Pso = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

// ════════════════════════════════════════════════════════════════════
// Structs matching Metal layout
// ════════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Clone, Copy)]
struct E21Params {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    pass: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct E21InnerParams {
    shift: u32,
    tg_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BucketDesc {
    offset: u32,
    count: u32,
    tile_count: u32,
    tile_base: u32,
}

// Reuse exp16's partition structs
#[repr(C)]
#[derive(Clone, Copy)]
struct Exp16Params {
    element_count: u32,
    num_tiles: u32,
    num_tgs: u32,
    shift: u32,
    pass: u32,
}

// ════════════════════════════════════════════════════════════════════
// Utility helpers
// ════════════════════════════════════════════════════════════════════

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] } else { sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo as f64) }
}

fn gen_random_u32(n: usize) -> Vec<u32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<u32>()).collect()
}

fn check_correctness(result: &[u32], expected: &[u32]) -> usize {
    let mut mismatches = 0usize;
    for i in 0..result.len().min(expected.len()) {
        if result[i] != expected[i] {
            mismatches += 1;
            if mismatches <= 5 {
                println!(
                    "      mismatch idx={}: got=0x{:08X} expected=0x{:08X}",
                    i, result[i], expected[i]
                );
            }
        }
    }
    mismatches
}

// ════════════════════════════════════════════════════════════════════
// A/B benchmark: presort scatter vs random scatter
// ════════════════════════════════════════════════════════════════════

fn bench_variant(
    ctx: &MetalContext,
    n: usize,
    label: &str,
    pso_inner_scatter: &Pso,
) -> (f64, bool) {
    let num_tiles = n.div_ceil(TILE_SIZE);
    let inner_tg_count = NUM_BINS * MAX_TPB; // 256 * 17 = 4352
    let tile_hists_entries = NUM_BINS * MAX_TPB * NUM_BINS; // 256 * 17 * 256 = 1,114,112
    let inner_zero_tg_count = tile_hists_entries.div_ceil(THREADS_PER_TG);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    // Allocate buffers
    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * NUM_BINS * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * NUM_BINS * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, NUM_BINS * std::mem::size_of::<BucketDesc>());
    let buf_tile_hists = alloc_buffer(&ctx.device, tile_hists_entries * 4);

    // PSOs — MSD phase reuses exp16/exp17 kernels
    let pso_msd_histogram = ctx.make_pipeline("exp21_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp21_compute_bucket_descs");
    let pso_global_prefix = ctx.make_pipeline("exp21_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_inner_zero = ctx.make_pipeline("exp21_inner_zero");
    let pso_inner_histogram = ctx.make_pipeline("exp21_inner_histogram");

    // Params
    let msd_params = E21Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        shift: 24,
        pass: 0,
    };
    let exp16_params = Exp16Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        num_tgs: num_tiles as u32,
        shift: 24,
        pass: 0,
    };
    let tile_size_u32 = TILE_SIZE as u32;
    let tile_hists_total = tile_hists_entries as u32;

    // Grid sizes
    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * NUM_BINS).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
    let inner_grid = MTLSize { width: inner_tg_count, height: 1, depth: 1 };
    let inner_zero_grid = MTLSize { width: inner_zero_tg_count, height: 1, depth: 1 };

    let inner_shifts: [u32; 3] = [0, 8, 16];

    let encode_sort = |cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>| {
        // Zero MSD hist (CPU-side)
        unsafe {
            std::ptr::write_bytes(
                buf_msd_hist.contents().as_ptr() as *mut u8, 0,
                4 * NUM_BINS * 4,
            );
        }

        let enc = cmd.computeCommandEncoder().unwrap();

        // ── MSD Phase (5 dispatches) ──

        // 1. MSD histogram
        enc.setComputePipelineState(&pso_msd_histogram);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&msd_params as *const E21Params as *mut _).unwrap(),
                std::mem::size_of::<E21Params>(), 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        // 2. Compute bucket descriptors
        enc.setComputePipelineState(&pso_bucket_descs);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(), 4, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);

        // 3. Global prefix sum
        enc.setComputePipelineState(&pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);

        // 4. Zero tile status
        enc.setComputePipelineState(&pso_zero_status);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp16_params as *const Exp16Params as *mut _).unwrap(),
                std::mem::size_of::<Exp16Params>(), 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(zero_grid, tg_size);

        // 5. MSD partition (decoupled lookback scatter)
        enc.setComputePipelineState(&pso_partition);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp16_params as *const Exp16Params as *mut _).unwrap(),
                std::mem::size_of::<Exp16Params>(), 5,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        // ── Inner Sort (3 passes × 3 dispatches = 9 dispatches) ──
        for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
            let inner_params = E21InnerParams { shift, tg_offset: 0 };
            // Ping-pong: MSD writes to buf_b, inner pass 0 reads buf_b → buf_a
            let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                (buf_b.as_ref(), buf_a.as_ref())
            } else {
                (buf_a.as_ref(), buf_b.as_ref())
            };

            // Zero tile_hists
            enc.setComputePipelineState(&pso_inner_zero);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::new(&tile_hists_total as *const u32 as *mut _).unwrap(), 4, 1,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_zero_grid, tg_size);

            // Inner histogram
            enc.setComputePipelineState(&pso_inner_histogram);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&inner_params as *const E21InnerParams as *mut _).unwrap(),
                    std::mem::size_of::<E21InnerParams>(), 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_grid, tg_size);

            // Inner scatter (A: presort or B: random)
            enc.setComputePipelineState(pso_inner_scatter);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&inner_params as *const E21InnerParams as *mut _).unwrap(),
                    std::mem::size_of::<E21InnerParams>(), 4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_grid, tg_size);
        }

        enc.endEncoding();
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

    // After 3 inner passes (odd count), final result is in buf_a
    // MSD: a→b. Inner0: b→a. Inner1: a→b. Inner2: b→a. Result in buf_a.
    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
    let mismatches = check_correctness(&results, &expected);
    let correct = mismatches == 0;

    if !correct {
        println!("    {} CORRECTNESS FAILED: {} mismatches out of {}", label, mismatches, n);
        println!("    First 10 result:   {:?}", &results[..10.min(n)]);
        println!("    First 10 expected: {:?}", &expected[..10.min(n)]);
        return (0.0, false);
    }
    println!("    {} Correctness: PASS", label);

    // ── Benchmark ──
    let mut times: Vec<f64> = Vec::new();
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

        let gpu_time = gpu_elapsed_ms(&cmd);
        if iter >= WARMUP {
            times.push(gpu_time);
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p5 = percentile(&times, 5.0);
    let p50 = percentile(&times, 50.0);
    let p95 = percentile(&times, 95.0);
    let mkeys = n as f64 / p50 / 1000.0;
    let bw = (n as f64 * 4.0 * 2.0 * 4.0) / p50 / 1e6; // 4 passes total (1 MSD + 3 inner) × r+w

    println!("    {} Median: {:.3} ms | {:.0} Mkeys/s | {:.0} GB/s eff",
             label, p50, mkeys, bw);
    println!("    {} [p5={:.3} p95={:.3} spread={:.0}%]",
             label, p5, p95, if p5 > 0.0 { (p95 - p5) / p5 * 100.0 } else { 0.0 });

    (mkeys, correct)
}

// ════════════════════════════════════════════════════════════════════
// Phase timing: measure inner scatter only (isolate the A/B diff)
// ════════════════════════════════════════════════════════════════════

fn bench_inner_scatter_only(
    ctx: &MetalContext,
    n: usize,
    label: &str,
    pso_inner_scatter: &Pso,
) -> f64 {
    let num_tiles = n.div_ceil(TILE_SIZE);
    let inner_tg_count = NUM_BINS * MAX_TPB;
    let tile_hists_entries = NUM_BINS * MAX_TPB * NUM_BINS;
    let inner_zero_tg_count = tile_hists_entries.div_ceil(THREADS_PER_TG);

    let data = gen_random_u32(n);

    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * NUM_BINS * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * NUM_BINS * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, NUM_BINS * std::mem::size_of::<BucketDesc>());
    let buf_tile_hists = alloc_buffer(&ctx.device, tile_hists_entries * 4);

    let pso_msd_histogram = ctx.make_pipeline("exp21_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp21_compute_bucket_descs");
    let pso_global_prefix = ctx.make_pipeline("exp21_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_inner_zero = ctx.make_pipeline("exp21_inner_zero");
    let pso_inner_histogram = ctx.make_pipeline("exp21_inner_histogram");

    let msd_params = E21Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        shift: 24,
        pass: 0,
    };
    let exp16_params = Exp16Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        num_tgs: num_tiles as u32,
        shift: 24,
        pass: 0,
    };
    let tile_size_u32 = TILE_SIZE as u32;
    let tile_hists_total = tile_hists_entries as u32;

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * NUM_BINS).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
    let inner_grid = MTLSize { width: inner_tg_count, height: 1, depth: 1 };
    let inner_zero_grid = MTLSize { width: inner_zero_tg_count, height: 1, depth: 1 };

    // First, run MSD phase to set up bucket_descs (only once)
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
        std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * NUM_BINS * 4);
    }

    {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        enc.setComputePipelineState(&pso_msd_histogram);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&msd_params as *const E21Params as *mut _).unwrap(),
                std::mem::size_of::<E21Params>(), 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        enc.setComputePipelineState(&pso_bucket_descs);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(), 4, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);

        enc.setComputePipelineState(&pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);

        enc.setComputePipelineState(&pso_zero_status);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp16_params as *const Exp16Params as *mut _).unwrap(),
                std::mem::size_of::<Exp16Params>(), 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(zero_grid, tg_size);

        enc.setComputePipelineState(&pso_partition);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp16_params as *const Exp16Params as *mut _).unwrap(),
                std::mem::size_of::<Exp16Params>(), 5,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    // Now benchmark JUST a single inner pass (shift=0) repeatedly
    let inner_params = E21InnerParams { shift: 0, tg_offset: 0 };
    let mut times: Vec<f64> = Vec::new();

    for iter in 0..(WARMUP + RUNS) {
        // Inner histogram + scatter only (buf_b → buf_a for pass 0)
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        enc.setComputePipelineState(&pso_inner_zero);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 0);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_hists_total as *const u32 as *mut _).unwrap(), 4, 1,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(inner_zero_grid, tg_size);

        enc.setComputePipelineState(&pso_inner_histogram);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&inner_params as *const E21InnerParams as *mut _).unwrap(),
                std::mem::size_of::<E21InnerParams>(), 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(inner_grid, tg_size);

        enc.setComputePipelineState(pso_inner_scatter);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&inner_params as *const E21InnerParams as *mut _).unwrap(),
                std::mem::size_of::<E21InnerParams>(), 4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(inner_grid, tg_size);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if iter >= WARMUP {
            times.push(gpu_elapsed_ms(&cmd));
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&times, 50.0);
    let bw = (n as f64 * 4.0 * 2.0) / p50 / 1e6; // single pass: read + write
    println!("    {} inner pass p50: {:.3} ms | {:.0} GB/s (single pass)",
             label, p50, bw);
    p50
}

// ════════════════════════════════════════════════════════════════════
// Main entry point
// ════════════════════════════════════════════════════════════════════

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "═".repeat(60));
    println!("  EXP 21: Pre-Sort Scatter (Stehle-Jacobsen Technique)");
    println!("  A/B test: pre-sort vs random scatter in inner LSD passes");
    println!("  Architecture: MSD(24:31) + 3×LSD(0:7,8:15,16:23)");
    println!("{}", "═".repeat(60));

    let n = 16_000_000usize;

    // Create PSOs for both variants
    let pso_presort = ctx.make_pipeline("exp21_inner_presort_scatter_v2");
    let pso_random = ctx.make_pipeline("exp21_inner_random_scatter");

    // ── Test 1: Correctness + full pipeline benchmark at 16M ──
    println!("\n  ── Test 1: Full Pipeline @ 16M (correctness + perf) ──\n");

    // Run smaller sizes first for correctness checks
    let small_sizes = [62_500usize, 250_000, 1_000_000];
    for &sz in &small_sizes {
        println!("  ── Correctness @ {} ──", sz);
        let (_, correct_a) = bench_variant(ctx, sz, "[PRESORT]", &pso_presort);
        let (_, correct_b) = bench_variant(ctx, sz, "[RANDOM] ", &pso_random);
        if !correct_a || !correct_b {
            println!("  ABORT: Correctness failure at size {}", sz);
            return;
        }
    }

    println!("\n  ── Benchmark @ 16M ──\n");

    let (mkeys_presort, correct_presort) = bench_variant(ctx, n, "[PRESORT]", &pso_presort);
    let (mkeys_random, correct_random) = bench_variant(ctx, n, "[RANDOM] ", &pso_random);

    if correct_presort && correct_random && mkeys_random > 0.0 {
        let speedup = mkeys_presort / mkeys_random;
        println!("\n    Pre-sort speedup: {:.2}x ({:.0} vs {:.0} Mkeys/s)",
                 speedup, mkeys_presort, mkeys_random);
    }

    // ── Test 2: Isolated inner scatter timing ──
    println!("\n  ── Test 2: Isolated Inner Scatter Timing @ 16M ──\n");

    let p50_presort = bench_inner_scatter_only(ctx, n, "[PRESORT]", &pso_presort);
    let p50_random = bench_inner_scatter_only(ctx, n, "[RANDOM] ", &pso_random);

    if p50_random > 0.0 {
        let inner_speedup = p50_random / p50_presort;
        println!("\n    Inner scatter speedup: {:.2}x ({:.3} vs {:.3} ms)",
                 inner_speedup, p50_presort, p50_random);
        println!("    Time saved per inner pass: {:.3} ms", p50_random - p50_presort);
        println!("    Time saved × 3 passes: {:.3} ms", (p50_random - p50_presort) * 3.0);
    }

    // ── Summary ──
    println!("\n  ── Summary ──");
    println!("    vs exp16 (3003 Mk/s): presort {:.2}x, random {:.2}x",
             mkeys_presort / 3003.0, mkeys_random / 3003.0);
    println!("    vs exp17 (3461 Mk/s): presort {:.2}x, random {:.2}x",
             mkeys_presort / 3461.0, mkeys_random / 3461.0);
    println!("    Target: 5000+ Mk/s");
}
