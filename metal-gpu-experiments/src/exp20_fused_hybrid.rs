//! Experiment 20: Batched Hybrid Sort (SLC-Resident Inner Passes)
//!
//! Same MSD+LSD architecture as exp17 (3461 Mkeys/s baseline).
//! Key optimization: batch inner passes so each batch's working set
//! fits in SLC (24MB), achieving 469 GB/s instead of 245 GB/s DRAM.
//!
//! At 16M elements: 256 buckets × 62.5K × 4B = 64MB total.
//! With B=64 buckets per batch: 16MB working set < 24MB SLC.
//! With B=96: 24MB ≈ SLC boundary.
//!
//! Dispatch structure per inner pass:
//!   1× zero tile_hists (full buffer)
//!   For each batch of B buckets:
//!     1× inner_histogram (B × MAX_TPB TGs)
//!     1× inner_scan_scatter (B × MAX_TPB TGs)
//! Total dispatches = 1 + num_batches × 2 per pass.

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
use std::ptr::NonNull;

const WARMUP: usize = 5;
const RUNS: usize = 50;
const THREADS_PER_TG: usize = 256;
const TILE_SIZE: usize = 4096;
const MAX_TPB: usize = 17;
const NUM_BINS: usize = 256;

type Pso = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

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

#[repr(C)]
#[derive(Clone, Copy)]
struct BucketDesc {
    offset: u32,
    count: u32,
    tile_count: u32,
    tile_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp16Params {
    element_count: u32,
    num_tiles: u32,
    num_tgs: u32,
    shift: u32,
    pass: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp17Params {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    pass: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp17InnerParams {
    shift: u32,
    tg_offset: u32,
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "═".repeat(60));
    println!("  EXP 20: Batched Hybrid Sort (SLC-Resident Inner Passes)");
    println!("  SLC=469 GB/s vs DRAM=245 GB/s — 1.9x advantage");
    println!("  Batch inner passes to keep working set < 24MB SLC");
    println!("{}", "═".repeat(60));

    // PSOs
    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_msd_partition = ctx.make_pipeline("exp16_partition");
    let pso_inner_zero = ctx.make_pipeline("exp17_inner_zero");
    let pso_inner_histogram = ctx.make_pipeline("exp17_inner_histogram");
    let pso_inner_scan_scatter = ctx.make_pipeline("exp17_inner_scan_scatter");

    let n: usize = 16_000_000;
    let num_tiles = n.div_ceil(TILE_SIZE);
    let tile_hists_entries: usize = NUM_BINS * MAX_TPB * NUM_BINS; // 1,114,112

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    // Allocate buffers
    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * NUM_BINS * 4);
    let buf_msd_tile_status = alloc_buffer(&ctx.device, num_tiles * NUM_BINS * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, NUM_BINS * std::mem::size_of::<BucketDesc>());
    let buf_tile_hists = alloc_buffer(&ctx.device, tile_hists_entries * 4);

    // Params
    let exp17_params = Exp17Params {
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
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let zero_msd_tg_count = (num_tiles * NUM_BINS).div_ceil(THREADS_PER_TG);
    let zero_msd_grid = MTLSize { width: zero_msd_tg_count, height: 1, depth: 1 };
    let inner_zero_tg_count = tile_hists_entries.div_ceil(THREADS_PER_TG);
    let inner_zero_grid = MTLSize { width: inner_zero_tg_count, height: 1, depth: 1 };

    let inner_shifts: [u32; 3] = [0, 8, 16];

    // Batch sizes to sweep: [4, 16, 32, 64, 96, 128, 256]
    let batch_sizes: &[usize] = &[4, 16, 32, 64, 96, 128, 256];

    // First, run unbatched (B=256) for correctness baseline
    println!("\n  ── Correctness check (unbatched B=256) ──");
    {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * NUM_BINS * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        // MSD phase
        encode_msd_phase(
            &enc, &pso_histogram, &pso_bucket_descs, &pso_prefix,
            &pso_zero_status, &pso_msd_partition,
            &buf_a, &buf_b, &buf_msd_hist, &buf_msd_tile_status,
            &buf_counters, &buf_bucket_descs,
            &exp17_params, &exp16_params, tile_size_u32,
            hist_grid, one_tg, zero_msd_grid, tg_size,
        );

        // Inner passes (unbatched)
        for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
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

            // All buckets at once
            let inner_params = Exp17InnerParams { shift, tg_offset: 0 };
            let inner_grid = MTLSize { width: NUM_BINS * MAX_TPB, height: 1, depth: 1 };

            enc.setComputePipelineState(&pso_inner_histogram);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                    std::mem::size_of::<Exp17InnerParams>(), 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_grid, tg_size);

            enc.setComputePipelineState(&pso_inner_scan_scatter);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                    std::mem::size_of::<Exp17InnerParams>(), 4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_grid, tg_size);
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        // After 3 inner passes (odd), result in buf_a
        let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
        let mismatches = check_correctness(&results, &expected);
        if mismatches > 0 {
            println!("    BASELINE FAILED: {} mismatches — aborting", mismatches);
            return;
        }
        println!("    Baseline correctness: PASS");
    }

    // Now benchmark each batch size
    println!("\n  ── Batch Size Sweep @ {}M ──", n / 1_000_000);
    println!("    {:>6} | {:>8} | {:>8} | {:>8} | {:>7} | {:>6} | {:<8}",
             "Batch", "WS (MB)", "Batches", "Dispatches", "p50 ms", "Mkeys", "vs exp17");
    println!("    {}", "-".repeat(72));

    for &batch_size in batch_sizes {
        let num_batches = NUM_BINS.div_ceil(batch_size);
        let working_set_mb = batch_size as f64 * (n as f64 / 256.0) * 4.0 / 1024.0 / 1024.0;
        // Dispatches: 5 MSD + 3 passes × (1 zero + num_batches × 2)
        let total_dispatches = 5 + 3 * (1 + num_batches * 2);

        let mut times: Vec<f64> = Vec::new();
        let mut correct = false;

        for iter in 0..(WARMUP + RUNS) {
            unsafe {
                let src = buf_input.contents().as_ptr() as *const u8;
                let dst = buf_a.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src, dst, n * 4);
                std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * NUM_BINS * 4);
            }

            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();

            // MSD phase (5 dispatches)
            encode_msd_phase(
                &enc, &pso_histogram, &pso_bucket_descs, &pso_prefix,
                &pso_zero_status, &pso_msd_partition,
                &buf_a, &buf_b, &buf_msd_hist, &buf_msd_tile_status,
                &buf_counters, &buf_bucket_descs,
                &exp17_params, &exp16_params, tile_size_u32,
                hist_grid, one_tg, zero_msd_grid, tg_size,
            );

            // Inner passes (batched)
            for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
                let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                    (buf_b.as_ref(), buf_a.as_ref())
                } else {
                    (buf_a.as_ref(), buf_b.as_ref())
                };

                // Zero all tile_hists once per pass
                enc.setComputePipelineState(&pso_inner_zero);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 0);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&tile_hists_total as *const u32 as *mut _).unwrap(), 4, 1,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(inner_zero_grid, tg_size);

                // Process batches of buckets
                for batch_idx in 0..num_batches {
                    let bucket_start = batch_idx * batch_size;
                    let bucket_end = (bucket_start + batch_size).min(NUM_BINS);
                    let buckets_in_batch = bucket_end - bucket_start;
                    let tg_offset = (bucket_start * MAX_TPB) as u32;
                    let batch_tgs = buckets_in_batch * MAX_TPB;
                    let inner_params = Exp17InnerParams { shift, tg_offset };
                    let batch_grid = MTLSize { width: batch_tgs, height: 1, depth: 1 };

                    // Histogram for this batch
                    enc.setComputePipelineState(&pso_inner_histogram);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                            std::mem::size_of::<Exp17InnerParams>(), 3,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid, tg_size);

                    // Scan+scatter for this batch
                    enc.setComputePipelineState(&pso_inner_scan_scatter);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 3);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                            std::mem::size_of::<Exp17InnerParams>(), 4,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid, tg_size);
                }
            }

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let gpu_time = unsafe { cmd.GPUEndTime() - cmd.GPUStartTime() };
            if iter >= WARMUP {
                times.push(gpu_time * 1000.0);
            }

            // Correctness on last run
            if iter == WARMUP + RUNS - 1 {
                let results: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
                correct = check_correctness(&results, &expected) == 0;
            }
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&times, 50.0);
        let mkeys = n as f64 / p50 / 1000.0;
        let vs_exp17 = mkeys / 3461.0;
        let status = if correct { "OK" } else { "FAIL" };

        println!(
            "    {:>4}   | {:>6.1}  | {:>6}  | {:>8}  | {:>6.3} | {:>5.0} | {:.2}x  {}",
            batch_size, working_set_mb, num_batches, total_dispatches,
            p50, mkeys, vs_exp17, status
        );
    }

    // Phase timing for best batch size
    println!("\n  ── Phase Timing: Batched(64) vs Unbatched(256) @ 16M ──");

    for &batch_size in &[64usize, 256] {
        let num_batches = NUM_BINS.div_ceil(batch_size);
        let label = if batch_size == 256 { "Unbatched" } else { &format!("B={}", batch_size) };

        // Measure MSD phase separately
        let mut msd_times: Vec<f64> = Vec::new();
        let mut inner_times: Vec<f64> = Vec::new();

        for iter in 0..(WARMUP + 20) {
            unsafe {
                let src = buf_input.contents().as_ptr() as *const u8;
                let dst = buf_a.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src, dst, n * 4);
                std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * NUM_BINS * 4);
            }

            // MSD phase in separate command buffer
            {
                let cmd = ctx.command_buffer();
                let enc = cmd.computeCommandEncoder().unwrap();
                encode_msd_phase(
                    &enc, &pso_histogram, &pso_bucket_descs, &pso_prefix,
                    &pso_zero_status, &pso_msd_partition,
                    &buf_a, &buf_b, &buf_msd_hist, &buf_msd_tile_status,
                    &buf_counters, &buf_bucket_descs,
                    &exp17_params, &exp16_params, tile_size_u32,
                    hist_grid, one_tg, zero_msd_grid, tg_size,
                );
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if iter >= WARMUP {
                    msd_times.push(unsafe { (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0 });
                }
            }

            // Inner phases in separate command buffer
            {
                let cmd = ctx.command_buffer();
                let enc = cmd.computeCommandEncoder().unwrap();
                for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
                    let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                        (buf_b.as_ref(), buf_a.as_ref())
                    } else {
                        (buf_a.as_ref(), buf_b.as_ref())
                    };

                    enc.setComputePipelineState(&pso_inner_zero);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 0);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&tile_hists_total as *const u32 as *mut _).unwrap(), 4, 1,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(inner_zero_grid, tg_size);

                    for batch_idx in 0..num_batches {
                        let bucket_start = batch_idx * batch_size;
                        let bucket_end = (bucket_start + batch_size).min(NUM_BINS);
                        let buckets_in_batch = bucket_end - bucket_start;
                        let tg_offset = (bucket_start * MAX_TPB) as u32;
                        let batch_tgs = buckets_in_batch * MAX_TPB;
                        let inner_params = Exp17InnerParams { shift, tg_offset };
                        let batch_grid = MTLSize { width: batch_tgs, height: 1, depth: 1 };

                        enc.setComputePipelineState(&pso_inner_histogram);
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 1);
                            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
                            enc.setBytes_length_atIndex(
                                NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                                std::mem::size_of::<Exp17InnerParams>(), 3,
                            );
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid, tg_size);

                        enc.setComputePipelineState(&pso_inner_scan_scatter);
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                            enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 2);
                            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 3);
                            enc.setBytes_length_atIndex(
                                NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                                std::mem::size_of::<Exp17InnerParams>(), 4,
                            );
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid, tg_size);
                    }
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if iter >= WARMUP {
                    inner_times.push(unsafe { (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0 });
                }
            }
        }

        msd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        inner_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let msd_p50 = percentile(&msd_times, 50.0);
        let inner_p50 = percentile(&inner_times, 50.0);
        let total_p50 = msd_p50 + inner_p50;
        let inner_bw = (n as f64 * 4.0 * 2.0 * 3.0) / inner_p50 / 1e6;
        let mkeys = n as f64 / total_p50 / 1000.0;

        println!(
            "    {:<10}: MSD={:.3}ms  Inner={:.3}ms  Total={:.3}ms  {:.0}Mk/s  InnerBW={:.0}GB/s",
            label, msd_p50, inner_p50, total_p50, mkeys, inner_bw
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_msd_phase(
    enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso_histogram: &Pso,
    pso_bucket_descs: &Pso,
    pso_prefix: &Pso,
    pso_zero_status: &Pso,
    pso_msd_partition: &Pso,
    buf_a: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_b: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_msd_hist: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_msd_tile_status: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_counters: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_bucket_descs: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    exp17_params: &Exp17Params,
    exp16_params: &Exp16Params,
    tile_size_u32: u32,
    hist_grid: MTLSize,
    one_tg: MTLSize,
    zero_msd_grid: MTLSize,
    tg_size: MTLSize,
) {
    // 1. MSD histogram
    enc.setComputePipelineState(pso_histogram);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(exp17_params as *const Exp17Params as *mut _).unwrap(),
            std::mem::size_of::<Exp17Params>(), 2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

    // 2. Compute bucket descs
    enc.setComputePipelineState(pso_bucket_descs);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(), 4, 2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);

    // 3. Global prefix
    enc.setComputePipelineState(pso_prefix);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist), 0, 0);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);

    // 4. Zero MSD tile status
    enc.setComputePipelineState(pso_zero_status);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_msd_tile_status), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_counters), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(exp16_params as *const Exp16Params as *mut _).unwrap(),
            std::mem::size_of::<Exp16Params>(), 2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(zero_msd_grid, tg_size);

    // 5. MSD scatter
    enc.setComputePipelineState(pso_msd_partition);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_msd_tile_status), 0, 2);
        enc.setBuffer_offset_atIndex(Some(buf_counters), 0, 3);
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist), 0, 4);
        enc.setBytes_length_atIndex(
            NonNull::new(exp16_params as *const Exp16Params as *mut _).unwrap(),
            std::mem::size_of::<Exp16Params>(), 5,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);
}
