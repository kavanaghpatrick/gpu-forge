//! Experiment 17: MSD+LSD Hybrid Radix Sort (5000+ Mkeys/s target)
//!
//! Architecture: 1 MSD scatter (bits 24:31) creates 256 buckets of ~62K
//! elements (~250KB each, SLC-resident), then 3 per-bucket LSD passes
//! at SLC speed. Single encoder, 14 dispatches, zero CPU readback.
//!
//! Phases:
//!   0. SLC scatter bandwidth benchmark (go/no-go gate)
//!   1. End-to-end hybrid sort at multiple sizes
//!   2. Per-phase timing breakdown
//!   3. Fallback analysis (if target not met)

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
use std::ptr::NonNull;

const WARMUP: usize = 5;
const RUNS: usize = 20;
const BENCH_WARMUP: usize = 5;
const BENCH_RUNS: usize = 50;
const THREADS_PER_TG: usize = 256;
const TILE_SIZE: usize = 4096;
const BASELINE_MKEYS: f64 = 3003.0; // exp16 8-bit 4-pass baseline at 16M

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

fn format_size(n: usize) -> String {
    if n >= 1_000_000 { format!("{}M", n / 1_000_000) }
    else if n >= 1_000 { format!("{}K", n / 1_000) }
    else { format!("{}", n) }
}

fn gen_random_u32(n: usize) -> Vec<u32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<u32>()).collect()
}

/// Check correctness of GPU-sorted result vs CPU reference.
/// Returns number of mismatches.
fn check_correctness(result: &[u32], expected: &[u32]) -> usize {
    let mut mismatches = 0usize;
    for i in 0..result.len().min(expected.len()) {
        if result[i] != expected[i] {
            mismatches += 1;
            if mismatches <= 3 {
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
// Shared structs (must match Metal layout)
// ════════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Clone, Copy, Debug)]
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
}

// ════════════════════════════════════════════════════════════════════
// Phase 0: SLC Scatter Bandwidth Benchmark
// ════════════════════════════════════════════════════════════════════

/// Benchmark 256-bin scatter bandwidth at various working set sizes.
/// Returns the 250K scatter bandwidth in GB/s (used for go/no-go gate).
fn bench_slc_scatter(ctx: &MetalContext) -> f64 {
    println!("\n  ── Phase 0: SLC Scatter Bandwidth ──");
    println!("  Kernel: exp16_diag_scatter_binned");
    println!("  Goal: measure scatter BW at SLC-resident sizes\n");

    let pso_scatter_binned = ctx.make_pipeline("exp16_diag_scatter_binned");
    let sizes: [usize; 5] = [62_500, 250_000, 1_000_000, 4_000_000, 16_000_000];
    let mut bw_250k: f64 = 0.0;

    for &n in &sizes {
        let label = format_size(n);
        let ws_kb = n * 4 * 2 / 1024;
        let ws_label = if ws_kb >= 1024 { format!("{}MB", ws_kb / 1024) }
                       else { format!("{}KB", ws_kb) };

        let data = gen_random_u32(n);

        // Compute 256-bin scatter offsets on CPU
        let bin_offsets: Vec<u32> = {
            let mut hist = vec![0u32; 256];
            for &v in &data { hist[(v & 0xFF) as usize] += 1; }
            let mut prefix = vec![0u32; 256];
            let mut sum = 0u32;
            for i in 0..256 { prefix[i] = sum; sum += hist[i]; }
            let mut counters = prefix.clone();
            let mut offsets = vec![0u32; n];
            for i in 0..n {
                let bin = (data[i] & 0xFF) as usize;
                offsets[i] = counters[bin];
                counters[bin] += 1;
            }
            offsets
        };

        let buf_src = alloc_buffer_with_data(&ctx.device, &data);
        let buf_dst = alloc_buffer(&ctx.device, n * 4);
        let buf_offsets = alloc_buffer_with_data(&ctx.device, &bin_offsets);
        let n_u32 = n as u32;

        let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
        let grid_tgs = MTLSize { width: n.div_ceil(THREADS_PER_TG), height: 1, depth: 1 };

        let mut times = Vec::new();
        for iter in 0..(WARMUP + RUNS) {
            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_scatter_binned);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_offsets.as_ref()), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(), 4, 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_tgs, tg_size);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            if iter >= WARMUP { times.push(gpu_elapsed_ms(&cmd)); }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p5 = percentile(&times, 5.0);
        let p50 = percentile(&times, 50.0);
        let p95 = percentile(&times, 95.0);
        let bw = (n * 4 * 2) as f64 / p50 / 1e6;

        println!(
            "    {:>8} ({:>5}): {:>7.3} ms  {:>6.0} GB/s  [p5={:.3} p95={:.3}]",
            label, ws_label, p50, bw, p5, p95
        );

        if n == 250_000 { bw_250k = bw; }
    }

    // Go/no-go gate
    println!();
    if bw_250k < 80.0 {
        println!(
            "  ABORT: SLC scatter too slow -- 250K = {:.0} GB/s (need >= 80)",
            bw_250k
        );
        println!("  Continuing anyway (gate is informational in POC)");
    } else {
        println!(
            "  GO: SLC scatter at 250K = {:.0} GB/s (>= 80 threshold)",
            bw_250k
        );
    }

    bw_250k
}

// ════════════════════════════════════════════════════════════════════
// Benchmark: full hybrid sort pipeline at size N
// ════════════════════════════════════════════════════════════════════

/// Benchmark the full hybrid sort pipeline at size n.
/// Returns (sorted_times_ms, correctness_ok).
fn bench_hybrid(ctx: &MetalContext, n: usize) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(TILE_SIZE);
    let inner_tg_count: usize = 256 * 17;
    let tile_hists_entries: usize = 256 * 17 * 256;
    let inner_zero_tg_count = tile_hists_entries.div_ceil(THREADS_PER_TG);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    // Allocate buffers (sized for this N)
    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());
    let buf_tile_hists = alloc_buffer(&ctx.device, tile_hists_entries * 4);

    // PSOs
    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_inner_zero = ctx.make_pipeline("exp17_inner_zero");
    let pso_inner_histogram = ctx.make_pipeline("exp17_inner_histogram");
    let pso_inner_scan_scatter = ctx.make_pipeline("exp17_inner_scan_scatter");

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
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
    let inner_grid = MTLSize { width: inner_tg_count, height: 1, depth: 1 };
    let inner_zero_grid = MTLSize { width: inner_zero_tg_count, height: 1, depth: 1 };

    let inner_shifts: [u32; 3] = [0, 8, 16];

    let mut times = Vec::new();
    let mut correct = false;

    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        // Reset: copy input to buf_a, zero msd_hist
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        // MSD phase (5 dispatches)
        enc.setComputePipelineState(&pso_histogram);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp17_params as *const Exp17Params as *mut _).unwrap(),
                std::mem::size_of::<Exp17Params>(), 2,
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
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

        enc.setComputePipelineState(&pso_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

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

        // Inner sort (3 passes x 3 dispatches = 9 dispatches)
        for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
            let inner_params = Exp17InnerParams { shift };
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

        let ms = gpu_elapsed_ms(&cmd);
        if iter >= BENCH_WARMUP { times.push(ms); }

        // Correctness check on last iteration
        if iter == BENCH_WARMUP + BENCH_RUNS - 1 {
            let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
            correct = check_correctness(&result, &expected) == 0;
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times, correct)
}

// ════════════════════════════════════════════════════════════════════
// Per-phase timing breakdown
// ════════════════════════════════════════════════════════════════════

/// Run each pipeline phase in separate command buffers to measure individual GPU times.
/// Returns (phase_p50s, full_p50) where phase_p50s = [msd_hist, msd_scatter, inner0, inner1, inner2].
fn bench_phases(ctx: &MetalContext, n: usize) -> (Vec<f64>, f64) {
    let num_tiles = n.div_ceil(TILE_SIZE);
    let inner_tg_count: usize = 256 * 17;
    let tile_hists_entries: usize = 256 * 17 * 256;
    let inner_zero_tg_count = tile_hists_entries.div_ceil(THREADS_PER_TG);

    let data = gen_random_u32(n);

    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());
    let buf_tile_hists = alloc_buffer(&ctx.device, tile_hists_entries * 4);

    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_inner_zero = ctx.make_pipeline("exp17_inner_zero");
    let pso_inner_histogram = ctx.make_pipeline("exp17_inner_histogram");
    let pso_inner_scan_scatter = ctx.make_pipeline("exp17_inner_scan_scatter");

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

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
    let inner_grid = MTLSize { width: inner_tg_count, height: 1, depth: 1 };
    let inner_zero_grid = MTLSize { width: inner_zero_tg_count, height: 1, depth: 1 };

    let inner_shifts: [u32; 3] = [0, 8, 16];
    let phase_warmup = 3;
    let phase_runs = 20;

    let mut phase_times: Vec<Vec<f64>> = vec![Vec::new(); 5];
    let mut full_times: Vec<f64> = Vec::new();

    for iter in 0..(phase_warmup + phase_runs) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
        }

        // Phase A: MSD histogram
        {
            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_histogram);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp17_params as *const Exp17Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp17Params>(), 2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            if iter >= phase_warmup { phase_times[0].push(gpu_elapsed_ms(&cmd)); }
        }

        // Phase B: BucketDesc + prefix + zero + scatter
        {
            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();

            enc.setComputePipelineState(&pso_bucket_descs);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(), 4, 2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

            enc.setComputePipelineState(&pso_prefix);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

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
            if iter >= phase_warmup { phase_times[1].push(gpu_elapsed_ms(&cmd)); }
        }

        // Phases C/D/E: Inner passes 0/1/2
        for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
            let inner_params = Exp17InnerParams { shift };
            let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                (buf_b.as_ref(), buf_a.as_ref())
            } else {
                (buf_a.as_ref(), buf_b.as_ref())
            };

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

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            if iter >= phase_warmup { phase_times[2 + pass_idx].push(gpu_elapsed_ms(&cmd)); }
        }

        if iter >= phase_warmup {
            let idx = iter - phase_warmup;
            let full = phase_times[0][idx] + phase_times[1][idx]
                + phase_times[2][idx] + phase_times[3][idx] + phase_times[4][idx];
            full_times.push(full);
        }
    }

    for t in phase_times.iter_mut() {
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    full_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let labels = [
        "MSD histogram",
        "MSD bd+pfx+zero+scatter",
        "Inner pass 0",
        "Inner pass 1",
        "Inner pass 2",
    ];

    let phase_runs_count = phase_runs;
    println!("    Phase breakdown (median of {} runs):", phase_runs_count);
    let mut sum_p50 = 0.0;
    let mut phase_p50s = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        let p50 = percentile(&phase_times[i], 50.0);
        phase_p50s.push(p50);
        sum_p50 += p50;
        println!("      {:.<28} {:>6.3} ms", label, p50);
    }
    println!("      {:.<28} {:>6.3} ms", "Sum of phases", sum_p50);
    let full_p50 = percentile(&full_times, 50.0);
    println!("      {:.<28} {:>6.3} ms", "Full pipeline (sum)", full_p50);
    let overhead = if full_p50 > sum_p50 { full_p50 - sum_p50 } else { 0.0 };
    println!("      {:.<28} {:>6.3} ms", "Overhead estimate", overhead);
    let implied_mkeys = n as f64 / full_p50 / 1e3;
    println!("      Implied throughput: {:.0} Mkeys/s", implied_mkeys);

    (phase_p50s, full_p50)
}

// ════════════════════════════════════════════════════════════════════
// Fallback analysis
// ════════════════════════════════════════════════════════════════════

/// Print per-phase bandwidth utilization and identify bottleneck.
/// Called when p50 < 5000 Mkeys/s.
fn print_fallback_analysis(n: usize, measured_mkeys: f64, phase_p50s: &[f64]) {
    println!("\n  ── Fallback Analysis (target not met) ──\n");

    let n_f = n as f64;
    let mb = n_f * 4.0 / 1e6;

    struct PhaseInfo {
        label: &'static str,
        measured_ms: f64,
        bytes_mb: f64,
        theoretical_bw_gbs: f64,
        bw_label: &'static str,
    }

    let phases = [
        PhaseInfo { label: "MSD histogram",  measured_ms: phase_p50s[0], bytes_mb: mb,       theoretical_bw_gbs: 245.0, bw_label: "245 (DRAM)" },
        PhaseInfo { label: "MSD scatter",    measured_ms: phase_p50s[1], bytes_mb: mb * 2.0, theoretical_bw_gbs: 131.0, bw_label: "131 (scatter)" },
        PhaseInfo { label: "Inner pass 0",   measured_ms: phase_p50s[2], bytes_mb: mb * 2.0, theoretical_bw_gbs: 469.0, bw_label: "469 (SLC)" },
        PhaseInfo { label: "Inner pass 1",   measured_ms: phase_p50s[3], bytes_mb: mb * 2.0, theoretical_bw_gbs: 469.0, bw_label: "469 (SLC)" },
        PhaseInfo { label: "Inner pass 2",   measured_ms: phase_p50s[4], bytes_mb: mb * 2.0, theoretical_bw_gbs: 469.0, bw_label: "469 (SLC)" },
    ];

    println!(
        "    {:<20}| {:<10}| {:<9}| {:<10}| {:<14}| {}",
        "Phase", "Measured", "Bytes", "Actual BW", "Theoretical", "Utilization"
    );
    println!("    {}", "-".repeat(82));

    let mut bottleneck_idx = 0usize;
    let mut worst_ratio = f64::MAX;
    let mut total_theoretical_ms = 0.0f64;

    for (i, p) in phases.iter().enumerate() {
        let actual_bw_gbs = if p.measured_ms > 0.0 { p.bytes_mb / p.measured_ms } else { 0.0 };
        let theoretical_ms = p.bytes_mb / p.theoretical_bw_gbs;
        total_theoretical_ms += theoretical_ms;
        let utilization = if p.measured_ms > 0.0 { actual_bw_gbs / p.theoretical_bw_gbs * 100.0 } else { 0.0 };

        println!(
            "    {:<20}| {:<10.3}| {:<4.0} MB  | {:<4.0} GB/s  | {:<14}| {:.0}%",
            p.label, p.measured_ms, p.bytes_mb, actual_bw_gbs, p.bw_label, utilization
        );

        if utilization < worst_ratio {
            worst_ratio = utilization;
            bottleneck_idx = i;
        }
    }

    println!(
        "\n    Bottleneck: {} ({:.0}% utilization)",
        phases[bottleneck_idx].label, worst_ratio
    );
    if bottleneck_idx >= 2 {
        println!("    Inner passes NOT running at SLC speed -- scatter write amplification likely.");
    }

    let sum_measured_ms: f64 = phase_p50s.iter().sum();
    let measured_ceiling = n_f / sum_measured_ms / 1e3;
    let theoretical_ceiling = n_f / total_theoretical_ms / 1e3;

    println!(
        "\n    Theoretical ceiling (BW limits):    {:.0} Mkeys/s ({:.3} ms)",
        theoretical_ceiling, total_theoretical_ms
    );
    println!(
        "    Measured ceiling (sum of phases):    {:.0} Mkeys/s ({:.3} ms)",
        measured_ceiling, sum_measured_ms
    );

    println!("\n    Root cause: Inner sort not achieving SLC bandwidth.");
    println!("    1. 256 buckets * 250KB = 64MB total -- exceeds SLC capacity");
    println!("    2. Scatter write amplification even at SLC-resident sizes");
    println!("    3. tile_hists reads add bandwidth not in simple model");
}

// ════════════════════════════════════════════════════════════════════
// Main entry point
// ════════════════════════════════════════════════════════════════════

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 17: MSD+LSD Hybrid Radix Sort");
    println!("{}", "=".repeat(60));
    println!("  1 MSD scatter (bits 24:31) + 3 inner LSD passes");
    println!("  Single encoder, 14 dispatches, {} elem/tile", TILE_SIZE);
    println!("  Target: 5000 Mkeys/s @ 16M | Baseline: {:.0} Mkeys/s\n", BASELINE_MKEYS);

    // ── Phase 0: SLC Scatter Bandwidth ──
    let _bw_250k = bench_slc_scatter(ctx);

    // ── Phase 1: End-to-end sort at multiple sizes ──
    println!("\n  ── Phase 1: Hybrid Sort Benchmark ──");

    let bench_sizes: [usize; 3] = [1_000_000, 4_000_000, 16_000_000];
    let mut mkeys_16m = 0.0f64;

    for &n in &bench_sizes {
        let label = format_size(n);
        let (times, correct) = bench_hybrid(ctx, n);
        let p5 = percentile(&times, 5.0);
        let p50 = percentile(&times, 50.0);
        let p95 = percentile(&times, 95.0);
        let spread = if p50 > 0.0 { (p95 - p5) / p50 * 100.0 } else { 0.0 };
        let mkeys = n as f64 / p50 / 1e3;
        let status = if correct { "ok" } else { "FAIL" };

        println!(
            "    {:>4}: {:>7.3} ms  {:>6.0} Mkeys/s  {:>4}  [p5={:.3} p95={:.3} spread={:.0}%]",
            label, p50, mkeys, status, p5, p95, spread
        );

        if n == 16_000_000 { mkeys_16m = mkeys; }
    }

    // ── Phase 2: Per-phase timing breakdown at 16M ──
    println!("\n  ── Phase 2: Per-Phase Timing @ 16M ──");
    let (phase_p50s, _full_p50) = bench_phases(ctx, 16_000_000);

    // ── Summary ──
    println!("\n  ── Summary ──");
    println!("    exp16 baseline:  {:>6.0} Mkeys/s", BASELINE_MKEYS);
    println!("    exp17 hybrid:    {:>6.0} Mkeys/s", mkeys_16m);
    println!("    Speedup:         {:>6.2}x", mkeys_16m / BASELINE_MKEYS);
    println!("    Target:          {:>6.0} Mkeys/s", 5000.0);
    if mkeys_16m >= 5000.0 {
        println!("    Status:          TARGET MET");
    } else {
        println!("    Gap:             {:>6.0} Mkeys/s", 5000.0 - mkeys_16m);
    }

    // ── Phase 3: Fallback analysis (if target not met) ──
    if mkeys_16m < 5000.0 {
        print_fallback_analysis(16_000_000, mkeys_16m, &phase_p50s);
    }
}
