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

type Pso = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

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
    tg_offset: u32,  // For batched dispatch: gid += tg_offset
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
            let inner_params = Exp17InnerParams { shift, tg_offset: 0 };
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
            let inner_params = Exp17InnerParams { shift, tg_offset: 0 };
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
// Tile size comparison: 4096 vs 8192
// ════════════════════════════════════════════════════════════════════

const TILE_SIZE_LARGE: usize = 8192;
const MAX_TPB_V2: usize = 9;

/// Benchmark inner passes with both tile sizes (4096 vs 8192) at 16M.
/// Runs full pipeline for each, compares total time and inner pass times.
fn bench_tile_comparison(ctx: &MetalContext) {
    let n: usize = 16_000_000;
    let num_tiles = n.div_ceil(TILE_SIZE);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    // Allocate buffers
    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);

    // Separate bucket_descs and tile_hists for each tile size
    let buf_bucket_descs_4k = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());
    let buf_bucket_descs_8k = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());
    let tile_hists_entries_4k: usize = 256 * 17 * 256;
    let tile_hists_entries_8k: usize = 256 * MAX_TPB_V2 * 256;
    let buf_tile_hists_4k = alloc_buffer(&ctx.device, tile_hists_entries_4k * 4);
    let buf_tile_hists_8k = alloc_buffer(&ctx.device, tile_hists_entries_8k * 4);

    // PSOs
    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_inner_zero = ctx.make_pipeline("exp17_inner_zero");
    let pso_inner_histogram = ctx.make_pipeline("exp17_inner_histogram");
    let pso_inner_scan_scatter = ctx.make_pipeline("exp17_inner_scan_scatter");
    let pso_inner_histogram_v2 = ctx.make_pipeline("exp17_inner_histogram_v2");
    let pso_inner_scan_scatter_v2 = ctx.make_pipeline("exp17_inner_scan_scatter_v2");

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
    let tile_size_4k_u32 = TILE_SIZE as u32;
    let tile_size_8k_u32 = TILE_SIZE_LARGE as u32;
    let tile_hists_total_4k = tile_hists_entries_4k as u32;
    let tile_hists_total_8k = tile_hists_entries_8k as u32;

    // Grid sizes
    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
    let inner_grid_4k = MTLSize { width: 256 * 17, height: 1, depth: 1 };
    let inner_zero_grid_4k = MTLSize { width: tile_hists_entries_4k.div_ceil(THREADS_PER_TG), height: 1, depth: 1 };
    let inner_grid_8k = MTLSize { width: 256 * MAX_TPB_V2, height: 1, depth: 1 };
    let inner_zero_grid_8k = MTLSize { width: tile_hists_entries_8k.div_ceil(THREADS_PER_TG), height: 1, depth: 1 };

    let inner_shifts: [u32; 3] = [0, 8, 16];
    let compare_warmup = 5;
    let compare_runs = 30;

    // Macro to run MSD phase (shared between both variants)
    macro_rules! run_msd_phase {
        ($buf_src:expr, $buf_dst:expr, $bucket_descs_buf:expr, $tile_size_param:expr) => {{
            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();

            enc.setComputePipelineState(&pso_histogram);
            unsafe {
                enc.setBuffer_offset_atIndex(Some($buf_src.as_ref()), 0, 0);
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
                enc.setBuffer_offset_atIndex(Some($bucket_descs_buf.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new($tile_size_param as *const u32 as *mut _).unwrap(), 4, 2,
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
                enc.setBuffer_offset_atIndex(Some($buf_src.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some($buf_dst.as_ref()), 0, 1);
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
        }};
    }

    println!("\n  ── Tile Size Comparison: 4096 vs 8192 @ 16M ──\n");

    // ── Benchmark 4096 tile size (original) ──
    let mut times_4k = Vec::new();
    let mut correct_4k = false;
    for iter in 0..(compare_warmup + compare_runs) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
        }

        run_msd_phase!(buf_a, buf_b, buf_bucket_descs_4k, &tile_size_4k_u32);

        // Inner passes with 4096 tile
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
            let inner_params = Exp17InnerParams { shift, tg_offset: 0 };
            let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                (buf_b.as_ref(), buf_a.as_ref())
            } else {
                (buf_a.as_ref(), buf_b.as_ref())
            };

            enc.setComputePipelineState(&pso_inner_zero);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists_4k.as_ref()), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::new(&tile_hists_total_4k as *const u32 as *mut _).unwrap(), 4, 1,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_zero_grid_4k, tg_size);

            enc.setComputePipelineState(&pso_inner_histogram);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists_4k.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_4k.as_ref()), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                    std::mem::size_of::<Exp17InnerParams>(), 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_grid_4k, tg_size);

            enc.setComputePipelineState(&pso_inner_scan_scatter);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists_4k.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_4k.as_ref()), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                    std::mem::size_of::<Exp17InnerParams>(), 4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_grid_4k, tg_size);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ms = gpu_elapsed_ms(&cmd);
        if iter >= compare_warmup { times_4k.push(ms); }

        if iter == compare_warmup + compare_runs - 1 {
            let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
            correct_4k = check_correctness(&result, &expected) == 0;
        }
    }
    times_4k.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // ── Benchmark 8192 tile size (v2) ──
    let mut times_8k = Vec::new();
    let mut correct_8k = false;
    for iter in 0..(compare_warmup + compare_runs) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
        }

        run_msd_phase!(buf_a, buf_b, buf_bucket_descs_8k, &tile_size_8k_u32);

        // Inner passes with 8192 tile
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
            let inner_params = Exp17InnerParams { shift, tg_offset: 0 };
            let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                (buf_b.as_ref(), buf_a.as_ref())
            } else {
                (buf_a.as_ref(), buf_b.as_ref())
            };

            enc.setComputePipelineState(&pso_inner_zero);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists_8k.as_ref()), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::new(&tile_hists_total_8k as *const u32 as *mut _).unwrap(), 4, 1,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_zero_grid_8k, tg_size);

            enc.setComputePipelineState(&pso_inner_histogram_v2);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists_8k.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_8k.as_ref()), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                    std::mem::size_of::<Exp17InnerParams>(), 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_grid_8k, tg_size);

            enc.setComputePipelineState(&pso_inner_scan_scatter_v2);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_tile_hists_8k.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_8k.as_ref()), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                    std::mem::size_of::<Exp17InnerParams>(), 4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(inner_grid_8k, tg_size);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ms = gpu_elapsed_ms(&cmd);
        if iter >= compare_warmup { times_8k.push(ms); }

        if iter == compare_warmup + compare_runs - 1 {
            let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
            correct_8k = check_correctness(&result, &expected) == 0;
        }
    }
    times_8k.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // ── Print comparison ──
    let p50_4k = percentile(&times_4k, 50.0);
    let p50_8k = percentile(&times_8k, 50.0);
    let p5_4k = percentile(&times_4k, 5.0);
    let p5_8k = percentile(&times_8k, 5.0);
    let p95_4k = percentile(&times_4k, 95.0);
    let p95_8k = percentile(&times_8k, 95.0);
    let status_4k = if correct_4k { "ok" } else { "FAIL" };
    let status_8k = if correct_8k { "ok" } else { "FAIL" };

    println!("    {:>14} {:>10} {:>10} {:>10} {:>8} {:>6}", "Tile size", "p5 (ms)", "p50 (ms)", "p95 (ms)", "TGs", "Status");
    println!("    {}", "-".repeat(68));
    println!("    {:>14} {:>10.3} {:>10.3} {:>10.3} {:>8} {:>6}",
        "4096 (16/thr)", p5_4k, p50_4k, p95_4k, 256 * 17, status_4k);
    println!("    {:>14} {:>10.3} {:>10.3} {:>10.3} {:>8} {:>6}",
        "8192 (32/thr)", p5_8k, p50_8k, p95_8k, 256 * MAX_TPB_V2, status_8k);

    let speedup = p50_4k / p50_8k;
    let delta_ms = p50_4k - p50_8k;
    println!();
    if p50_8k < p50_4k {
        println!("    Winner: 8192 tile ({:.2}x faster, {:.3}ms saved per 3 inner passes)", speedup, delta_ms);
    } else if p50_4k < p50_8k {
        println!("    Winner: 4096 tile ({:.2}x faster, {:.3}ms saved per 3 inner passes)", 1.0/speedup, -delta_ms);
    } else {
        println!("    Result: Tie (identical median times)");
    }
    println!("    Keeping 4096 as default (original tile size)");
}

// ════════════════════════════════════════════════════════════════════
// Investigation A: Scatter BW vs Bin Count @ 16M
//
// Key question: How much does bin count degrade scatter bandwidth?
// If 2048-bin scatter is close to 256-bin, we can do 3-pass (11+11+10).
// 3 passes at 129 GB/s = 2.98 ms = 5370 Mkeys/s → hits target!
// ════════════════════════════════════════════════════════════════════

fn bench_scatter_bw_vs_bins(ctx: &MetalContext) {
    println!("\n  ── Investigation A: Scatter BW vs Bin Count @ 16M ──");
    println!("  Kernel: exp16_diag_scatter_binned (pre-computed offsets)");
    println!("  Q: Does bin count affect scatter bandwidth?\n");

    let n: usize = 16_000_000;
    let data = gen_random_u32(n);
    let pso = ctx.make_pipeline("exp16_diag_scatter_binned");
    let buf_src = alloc_buffer_with_data(&ctx.device, &data);
    let buf_dst = alloc_buffer(&ctx.device, n * 4);
    let n_u32 = n as u32;
    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let grid = MTLSize { width: n.div_ceil(THREADS_PER_TG), height: 1, depth: 1 };

    println!("    {:>10} {:>6} {:>10} {:>10} {:>10}", "Bins", "Bits", "p50 (ms)", "GB/s", "vs 256");
    println!("    {}", "-".repeat(54));

    // Sequential copy baseline (identity offsets)
    let seq_offsets: Vec<u32> = (0..n as u32).collect();
    let buf_seq = alloc_buffer_with_data(&ctx.device, &seq_offsets);
    let mut seq_times = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_seq.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(), 4, 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg_size);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= WARMUP { seq_times.push(gpu_elapsed_ms(&cmd)); }
    }
    seq_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let seq_p50 = percentile(&seq_times, 50.0);
    let seq_bw = (n * 4 * 2) as f64 / seq_p50 / 1e6;
    println!(
        "    {:>10} {:>6} {:>10.3} {:>10.0} {:>10}",
        "1 (seq)", "0", seq_p50, seq_bw, "baseline"
    );

    let mut bw_256: f64 = 0.0;

    // Scatter at various bin counts
    for &bits in &[4u32, 6, 8, 10, 11, 12] {
        let num_bins = 1u32 << bits;
        let mask = num_bins - 1;

        // Compute scatter offsets on CPU
        let offsets: Vec<u32> = {
            let mut hist = vec![0u32; num_bins as usize];
            for &v in &data {
                hist[((v >> 0) & mask) as usize] += 1;
            }
            let mut prefix = vec![0u32; num_bins as usize];
            let mut sum = 0u32;
            for i in 0..num_bins as usize {
                prefix[i] = sum;
                sum += hist[i];
            }
            let mut counters = prefix.clone();
            let mut offs = vec![0u32; n];
            for i in 0..n {
                let bin = ((data[i] >> 0) & mask) as usize;
                offs[i] = counters[bin];
                counters[bin] += 1;
            }
            offs
        };

        let buf_offsets = alloc_buffer_with_data(&ctx.device, &offsets);

        let mut times = Vec::new();
        for iter in 0..(WARMUP + RUNS) {
            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_src.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_dst.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_offsets.as_ref()), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(), 4, 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg_size);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            if iter >= WARMUP { times.push(gpu_elapsed_ms(&cmd)); }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&times, 50.0);
        let bw = (n * 4 * 2) as f64 / p50 / 1e6;
        if bits == 8 { bw_256 = bw; }
        let vs = if bw_256 > 0.0 { format!("{:.2}x", bw / bw_256) }
                 else { format!("{:.0}", bw) };
        println!(
            "    {:>10} {:>6} {:>10.3} {:>10.0} {:>10}",
            num_bins, bits, p50, bw, vs
        );
    }

    println!("\n    Analysis:");
    println!("    - Sequential copy: {:.0} GB/s (upper bound)", seq_bw);
    println!("    - 256-bin scatter: {:.0} GB/s", bw_256);
    println!("    - If 2048-bin ≥ 80% of 256-bin → 3-pass 11-bit viable");
    println!("    - 3 passes @ {:.0} GB/s = {:.2} ms = {:.0} Mkeys/s",
        bw_256, 3.0 * (n * 4 * 2) as f64 / bw_256 / 1e6, n as f64 / (3.0 * (n * 4 * 2) as f64 / bw_256 / 1e6) / 1e3);
}

// ════════════════════════════════════════════════════════════════════
// Investigation B: Fused inner sort (1 TG per bucket, no tile_hists)
//
// Eliminates: tile_hists buffer (4.5 MB), zeroing dispatch, serial scan
// Each TG serially processes all tiles of its bucket.
// 256 TGs total (one per bucket).
// ════════════════════════════════════════════════════════════════════

fn bench_inner_fused(ctx: &MetalContext) {
    println!("\n  ── Investigation B: Fused Inner Sort (1 TG/bucket) ──");
    println!("  Kernel: exp17_inner_fused (no tile_hists, serial tiles)");
    println!("  Q: Does eliminating tile_hists improve inner pass BW?\n");

    let n: usize = 16_000_000;
    let num_tiles = n.div_ceil(TILE_SIZE);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    // Allocate buffers
    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());

    // PSOs for MSD phase (reuse existing)
    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    // New fused inner PSO
    let pso_inner_fused = ctx.make_pipeline("exp17_inner_fused");

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

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
    // Fused inner: 256 TGs (one per bucket)
    let fused_grid = MTLSize { width: 256, height: 1, depth: 1 };

    let inner_shifts: [u32; 3] = [0, 8, 16];

    let mut times = Vec::new();
    let mut correct = false;

    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        // Reset
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        // MSD phase (5 dispatches — same as before)
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

        // Inner sort: 3 passes × 1 fused dispatch each
        for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
            let inner_params = Exp17InnerParams { shift, tg_offset: 0 };
            let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                (buf_b.as_ref(), buf_a.as_ref())
            } else {
                (buf_a.as_ref(), buf_b.as_ref())
            };

            enc.setComputePipelineState(&pso_inner_fused);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                    std::mem::size_of::<Exp17InnerParams>(), 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);
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
    let p50 = percentile(&times, 50.0);
    let p5 = percentile(&times, 5.0);
    let p95 = percentile(&times, 95.0);
    let mkeys = n as f64 / p50 / 1e3;
    let status = if correct { "ok" } else { "FAIL" };

    println!("    Full pipeline (5 MSD + 3 fused inner):");
    println!("      p50={:.3} ms  p5={:.3} p95={:.3}  {} Mkeys/s  {}",
        p50, p5, p95, mkeys as u64, status);
    println!("      Dispatches: 8 total (was 14 with separate hist+scatter)");
    println!("      Eliminated: tile_hists buffer (4.5 MB), 3 zero dispatches, 3 histogram dispatches");
}

// ════════════════════════════════════════════════════════════════════
// Investigation Q: Batched Fused Inner — SLC-Resident Sort
//
// Process buckets in batches so working set fits in SLC (24MB).
// Each batch: all 3 inner passes for N buckets, then move to next batch.
// 48 buckets × ~250KB × 2 (R+W) = 24MB → SLC-resident.
// ════════════════════════════════════════════════════════════════════

fn bench_batched_fused(ctx: &MetalContext) {
    println!("\n  ── Investigation Q: Batched Fused Inner (SLC Target) ──");
    println!("  Sweep batch sizes to find SLC sweet spot");
    println!("  Each bucket ~250KB, SLC=24MB → 48 buckets/batch optimal\n");

    let n: usize = 16_000_000;
    let num_tiles = n.div_ceil(TILE_SIZE);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    // Allocate buffers
    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());

    // PSOs
    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_inner_fused = ctx.make_pipeline("exp17_inner_fused");

    let exp17_params = Exp17Params {
        element_count: n as u32, num_tiles: num_tiles as u32, shift: 24, pass: 0,
    };
    let exp16_params = Exp16Params {
        element_count: n as u32, num_tiles: num_tiles as u32,
        num_tgs: num_tiles as u32, shift: 24, pass: 0,
    };
    let tile_size_u32 = TILE_SIZE as u32;

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };

    let inner_shifts: [u32; 3] = [0, 8, 16];

    // Sweep batch sizes
    for &batch_size in &[16usize, 32, 48, 64, 96, 128, 256] {
        let num_batches = 256usize.div_ceil(batch_size);
        let total_dispatches = 5 + num_batches * 3; // MSD + batches × 3 passes
        let ws_mb = batch_size as f64 * (n as f64 / 256.0) * 4.0 * 2.0 / 1024.0 / 1024.0;

        let mut times = Vec::new();
        let mut correct = false;

        for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf_input.contents().as_ptr() as *const u8,
                    buf_a.contents().as_ptr() as *mut u8,
                    n * 4,
                );
                std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
            }

            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();

            // MSD phase (5 dispatches — same as bench_inner_fused)
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
            unsafe { enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0); }
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

            // Batched fused inner: process batch_size buckets at a time
            // All 3 passes for each batch before moving to next batch
            for batch in 0..num_batches {
                let bucket_start = (batch * batch_size) as u32;
                let buckets_this_batch = batch_size.min(256 - batch * batch_size);
                let fused_grid = MTLSize { width: buckets_this_batch, height: 1, depth: 1 };

                for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
                    let inner_params = Exp17InnerParams { shift, tg_offset: bucket_start };
                    let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                        (buf_b.as_ref(), buf_a.as_ref())
                    } else {
                        (buf_a.as_ref(), buf_b.as_ref())
                    };

                    enc.setComputePipelineState(&pso_inner_fused);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                            std::mem::size_of::<Exp17InnerParams>(), 3,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);
                }
            }

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let ms = gpu_elapsed_ms(&cmd);
            if iter >= BENCH_WARMUP { times.push(ms); }

            if iter == BENCH_WARMUP + BENCH_RUNS - 1 {
                let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
                correct = check_correctness(&result, &expected) == 0;
            }
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&times, 50.0);
        let p5 = percentile(&times, 5.0);
        let p95 = percentile(&times, 95.0);
        let mkeys = n as f64 / p50 / 1e3;
        let status = if correct { "ok" } else { "FAIL" };

        println!("    batch={:>3} | ws={:>5.1}MB | disp={:>3} | {:.3}ms | {:>4} Mk/s | {} | [p5={:.3} p95={:.3}]",
            batch_size, ws_mb, total_dispatches, p50, mkeys as u64, status, p5, p95);
    }
}

// ════════════════════════════════════════════════════════════════════
// Investigation R: Pre-computed Histogram + Fused V2
//
// Pre-compute all 3 inner digit histograms in a single pass after MSD
// scatter, then use fused_v2 kernel that skips Phase 1. This eliminates
// 3 × 64MB of histogram reads at the cost of 1 × 64MB precompute read.
// Net: 128MB less data movement.
// ════════════════════════════════════════════════════════════════════

fn bench_precomputed_fused(ctx: &MetalContext) {
    println!("\n  ── Investigation R: Pre-computed Histogram + Fused V2 ──");
    println!("  Pre-compute 3×256-bin inner histograms → skip Phase 1 in fused kernel");
    println!("  Saves 3×64MB reads, costs 1×64MB precompute. Net: -128MB\n");

    let n: usize = 16_000_000;
    let num_tiles = n.div_ceil(TILE_SIZE);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    // Allocate buffers
    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());
    let buf_inner_hists = alloc_buffer(&ctx.device, 256 * 3 * 256 * 4); // 768 KB

    // PSOs
    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_precompute = ctx.make_pipeline("exp17_inner_precompute_hists");
    let pso_inner_fused_v2 = ctx.make_pipeline("exp17_inner_fused_v2");

    let exp17_params = Exp17Params {
        element_count: n as u32, num_tiles: num_tiles as u32, shift: 24, pass: 0,
    };
    let exp16_params = Exp16Params {
        element_count: n as u32, num_tiles: num_tiles as u32,
        num_tgs: num_tiles as u32, shift: 24, pass: 0,
    };
    let tile_size_u32 = TILE_SIZE as u32;

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
    let precompute_grid = MTLSize { width: 256, height: 1, depth: 1 };

    let inner_shifts: [u32; 3] = [0, 8, 16];

    // Sweep batch sizes
    for &batch_size in &[64usize, 128, 256] {
        let num_batches = 256usize.div_ceil(batch_size);
        let total_dispatches = 5 + 1 + num_batches * 3; // MSD + precompute + batches × 3 passes
        let ws_mb = batch_size as f64 * (n as f64 / 256.0) * 4.0 * 2.0 / 1024.0 / 1024.0;

        let mut times = Vec::new();
        let mut correct = false;

        for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf_input.contents().as_ptr() as *const u8,
                    buf_a.contents().as_ptr() as *mut u8,
                    n * 4,
                );
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
            unsafe { enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0); }
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

            // Pre-compute inner histograms (1 dispatch: 256 TGs)
            // MSD scattered data is in buf_b
            enc.setComputePipelineState(&pso_precompute);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_inner_hists.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(precompute_grid, tg_size);

            // Batched fused V2 inner: process batch_size buckets at a time
            for batch in 0..num_batches {
                let bucket_start = (batch * batch_size) as u32;
                let buckets_this_batch = batch_size.min(256 - batch * batch_size);
                let fused_grid = MTLSize { width: buckets_this_batch, height: 1, depth: 1 };

                for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
                    let inner_params = Exp17InnerParams { shift, tg_offset: bucket_start };
                    let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                        (buf_b.as_ref(), buf_a.as_ref())
                    } else {
                        (buf_a.as_ref(), buf_b.as_ref())
                    };

                    enc.setComputePipelineState(&pso_inner_fused_v2);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                            std::mem::size_of::<Exp17InnerParams>(), 3,
                        );
                        enc.setBuffer_offset_atIndex(Some(buf_inner_hists.as_ref()), 0, 4);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);
                }
            }

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let ms = gpu_elapsed_ms(&cmd);
            if iter >= BENCH_WARMUP { times.push(ms); }

            if iter == BENCH_WARMUP + BENCH_RUNS - 1 {
                let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
                correct = check_correctness(&result, &expected) == 0;
            }
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&times, 50.0);
        let p5 = percentile(&times, 5.0);
        let p95 = percentile(&times, 95.0);
        let mkeys = n as f64 / p50 / 1e3;
        let status = if correct { "ok" } else { "FAIL" };

        println!("    batch={:>3} | ws={:>5.1}MB | disp={:>3} | {:.3}ms | {:>4} Mk/s | {} | [p5={:.3} p95={:.3}]",
            batch_size, ws_mb, total_dispatches, p50, mkeys as u64, status, p5, p95);
    }
}

// ════════════════════════════════════════════════════════════════════
// Investigation S: Fused 3-pass inner — all 3 passes in 1 dispatch
//
// Uses exp17_inner_fused_v3: each TG does all 3 inner passes for its
// bucket, with threadgroup_barrier(mem_device) between passes.
// L2 cache locality between passes (250KB/bucket fits in L2).
// Total: 5 MSD + 1 precompute + 1 fused-3-pass = 7 dispatches.
// ════════════════════════════════════════════════════════════════════

fn bench_fused_3pass(ctx: &MetalContext) {
    println!("\n  ── Investigation S: Fused 3-Pass Inner (1 dispatch) ──");
    println!("  All 3 inner passes in single kernel: L2 locality between passes");
    println!("  7 dispatches total (5 MSD + 1 precompute + 1 fused-3-pass)\n");

    let n: usize = 16_000_000;
    let num_tiles = n.div_ceil(TILE_SIZE);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());
    let buf_inner_hists = alloc_buffer(&ctx.device, 256 * 3 * 256 * 4);

    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_precompute = ctx.make_pipeline("exp17_inner_precompute_hists");
    let pso_fused_v3 = ctx.make_pipeline("exp17_inner_fused_v3");

    let exp17_params = Exp17Params {
        element_count: n as u32, num_tiles: num_tiles as u32, shift: 24, pass: 0,
    };
    let exp16_params = Exp16Params {
        element_count: n as u32, num_tiles: num_tiles as u32,
        num_tgs: num_tiles as u32, shift: 24, pass: 0,
    };
    let tile_size_u32 = TILE_SIZE as u32;

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
    let precompute_grid = MTLSize { width: 256, height: 1, depth: 1 };
    let fused_grid = MTLSize { width: 256, height: 1, depth: 1 };

    let mut times = Vec::new();
    let mut correct = false;

    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf_input.contents().as_ptr() as *const u8,
                buf_a.contents().as_ptr() as *mut u8,
                n * 4,
            );
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
        unsafe { enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0); }
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

        // Pre-compute inner histograms (1 dispatch: 256 TGs, reads buf_b)
        enc.setComputePipelineState(&pso_precompute);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_inner_hists.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(precompute_grid, tg_size);

        // Fused 3-pass inner (1 dispatch: 256 TGs, does all 3 passes internally)
        // Pass 0: buf_b→buf_a, Pass 1: buf_a→buf_b, Pass 2: buf_b→buf_a
        // Final sorted output in buf_a
        enc.setComputePipelineState(&pso_fused_v3);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_inner_hists.as_ref()), 0, 3);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ms = gpu_elapsed_ms(&cmd);
        if iter >= BENCH_WARMUP { times.push(ms); }

        if iter == BENCH_WARMUP + BENCH_RUNS - 1 {
            let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
            correct = check_correctness(&result, &expected) == 0;
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&times, 50.0);
    let p5 = percentile(&times, 5.0);
    let p95 = percentile(&times, 95.0);
    let mkeys = n as f64 / p50 / 1e3;
    let status = if correct { "ok" } else { "FAIL" };

    println!("    7 dispatches | {:.3}ms | {:>4} Mk/s | {} | [p5={:.3} p95={:.3}]",
        p50, mkeys as u64, status, p5, p95);
}

// ════════════════════════════════════════════════════════════════════
// Investigation T: Atomic MSD Scatter (no lookback)
//
// Replace decoupled lookback with atomic_fetch_add on global counters.
// Eliminates tile_status buffer, zero_status dispatch, and lookback
// spin-waiting (~1.0ms overhead). counters[] initialized to
// exclusive_prefix[d], so atomic_fetch_add returns exact global position.
//
// MSD phase: 3 dispatches (histogram + prep + atomic_scatter)
// Inner phase: 2 dispatches (precompute + fused_v3)
// Total: 5 dispatches
// ════════════════════════════════════════════════════════════════════

fn bench_atomic_msd(ctx: &MetalContext) {
    println!("\n  ── Investigation T: Atomic MSD Scatter (no lookback) ──");
    println!("  Replaces decoupled lookback with atomic_fetch_add");
    println!("  5 dispatches total (3 MSD + 1 precompute + 1 fused-3-pass)\n");

    let n: usize = 16_000_000;
    let num_tiles = n.div_ceil(TILE_SIZE);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 256 * 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());
    let buf_inner_hists = alloc_buffer(&ctx.device, 256 * 3 * 256 * 4);

    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_prep = ctx.make_pipeline("exp17_msd_prep");
    let pso_scatter = ctx.make_pipeline("exp17_msd_atomic_scatter");
    let pso_precompute = ctx.make_pipeline("exp17_inner_precompute_hists");
    let pso_fused_v3 = ctx.make_pipeline("exp17_inner_fused_v3");

    let exp17_params = Exp17Params {
        element_count: n as u32, num_tiles: num_tiles as u32, shift: 24, pass: 0,
    };
    let tile_size_u32 = TILE_SIZE as u32;

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let precompute_grid = MTLSize { width: 256, height: 1, depth: 1 };
    let fused_grid = MTLSize { width: 256, height: 1, depth: 1 };

    let mut times = Vec::new();
    let mut correct = false;

    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf_input.contents().as_ptr() as *const u8,
                buf_a.contents().as_ptr() as *mut u8,
                n * 4,
            );
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 256 * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        // Dispatch 1: MSD histogram (reads buf_a → global_hist[256])
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

        // Dispatch 2: MSD prep (prefix sum → counters + bucket_descs)
        enc.setComputePipelineState(&pso_prep);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(), 4, 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

        // Dispatch 3: Atomic MSD scatter (buf_a → buf_b, no lookback)
        enc.setComputePipelineState(&pso_scatter);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp17_params as *const Exp17Params as *mut _).unwrap(),
                std::mem::size_of::<Exp17Params>(), 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        // Dispatch 4: Pre-compute inner histograms
        enc.setComputePipelineState(&pso_precompute);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_inner_hists.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(precompute_grid, tg_size);

        // Dispatch 5: Fused 3-pass inner sort
        enc.setComputePipelineState(&pso_fused_v3);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_inner_hists.as_ref()), 0, 3);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ms = gpu_elapsed_ms(&cmd);
        if iter >= BENCH_WARMUP { times.push(ms); }

        if iter == BENCH_WARMUP + BENCH_RUNS - 1 {
            let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
            correct = check_correctness(&result, &expected) == 0;
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&times, 50.0);
    let p5 = percentile(&times, 5.0);
    let p95 = percentile(&times, 95.0);
    let mkeys = n as f64 / p50 / 1e3;
    let status = if correct { "ok" } else { "FAIL" };

    println!("    5 dispatches | {:.3}ms | {:>4} Mk/s | {} | [p5={:.3} p95={:.3}]",
        p50, mkeys as u64, status, p5, p95);
}

// ════════════════════════════════════════════════════════════════════
// Investigation V: Large-tile Atomic MSD (8192/tile)
//
// Same as Investigation T but with 8192-element tiles (32/thread).
// Halves TG count: 1953 vs 3906. Less atomic contention, less overhead.
// ════════════════════════════════════════════════════════════════════

fn bench_atomic_msd_large(ctx: &MetalContext) {
    println!("\n  ── Investigation V: Large-Tile Atomic MSD (8192/tile) ──");
    println!("  32 elements/thread, 1953 TGs (half of Investigation T)");
    println!("  5 dispatches total (3 MSD + 1 precompute + 1 fused-3-pass)\n");

    let n: usize = 16_000_000;
    let num_tiles_large = n.div_ceil(TILE_SIZE_LARGE);
    let num_tiles_small = n.div_ceil(TILE_SIZE);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 256 * 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());
    let buf_inner_hists = alloc_buffer(&ctx.device, 256 * 3 * 256 * 4);

    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram_large");
    let pso_prep = ctx.make_pipeline("exp17_msd_prep");
    let pso_scatter = ctx.make_pipeline("exp17_msd_atomic_scatter_large");
    let pso_precompute = ctx.make_pipeline("exp17_inner_precompute_hists");
    let pso_fused_v3 = ctx.make_pipeline("exp17_inner_fused_v3");

    let exp17_params = Exp17Params {
        element_count: n as u32, num_tiles: num_tiles_large as u32, shift: 24, pass: 0,
    };
    let tile_size_u32 = TILE_SIZE as u32;  // for bucket_descs (inner sort still uses 4096)

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles_large, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let precompute_grid = MTLSize { width: 256, height: 1, depth: 1 };
    let fused_grid = MTLSize { width: 256, height: 1, depth: 1 };

    let mut times = Vec::new();
    let mut correct = false;

    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf_input.contents().as_ptr() as *const u8,
                buf_a.contents().as_ptr() as *mut u8,
                n * 4,
            );
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 256 * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        // Dispatch 1: MSD histogram (large tiles)
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

        // Dispatch 2: MSD prep
        enc.setComputePipelineState(&pso_prep);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(), 4, 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

        // Dispatch 3: Atomic MSD scatter (large tiles)
        enc.setComputePipelineState(&pso_scatter);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp17_params as *const Exp17Params as *mut _).unwrap(),
                std::mem::size_of::<Exp17Params>(), 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        // Dispatch 4: Pre-compute inner histograms
        enc.setComputePipelineState(&pso_precompute);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_inner_hists.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(precompute_grid, tg_size);

        // Dispatch 5: Fused 3-pass inner sort
        enc.setComputePipelineState(&pso_fused_v3);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_inner_hists.as_ref()), 0, 3);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ms = gpu_elapsed_ms(&cmd);
        if iter >= BENCH_WARMUP { times.push(ms); }

        if iter == BENCH_WARMUP + BENCH_RUNS - 1 {
            let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
            correct = check_correctness(&result, &expected) == 0;
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&times, 50.0);
    let p5 = percentile(&times, 5.0);
    let p95 = percentile(&times, 95.0);
    let mkeys = n as f64 / p50 / 1e3;
    let status = if correct { "ok" } else { "FAIL" };

    println!("    5 dispatches | {:.3}ms | {:>4} Mk/s | {} | [p5={:.3} p95={:.3}]",
        p50, mkeys as u64, status, p5, p95);
}

// ════════════════════════════════════════════════════════════════════
// Investigation W: Self-Contained Inner Sort (fused_v4)
//
// Same as Investigation T but fused_v4 computes its own inner
// histograms during pass 0. Eliminates the precompute dispatch
// and inner_hists buffer.
//
// MSD phase: 3 dispatches (histogram + prep + atomic_scatter)
// Inner phase: 1 dispatch (fused_v4 — self-contained)
// Total: 4 dispatches
// ════════════════════════════════════════════════════════════════════

fn bench_investigation_w_at_size(ctx: &MetalContext, n: usize) -> (f64, f64, f64, bool) {
    let num_tiles = n.div_ceil(TILE_SIZE);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 256 * 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());

    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_prep = ctx.make_pipeline("exp17_msd_prep");
    let pso_scatter = ctx.make_pipeline("exp17_msd_atomic_scatter");
    let pso_fused_v4 = ctx.make_pipeline("exp17_inner_fused_v4");

    let exp17_params = Exp17Params {
        element_count: n as u32, num_tiles: num_tiles as u32, shift: 24, pass: 0,
    };
    let tile_size_u32 = TILE_SIZE as u32;

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let fused_grid = MTLSize { width: 256, height: 1, depth: 1 };

    let mut times = Vec::new();
    let mut correct = false;

    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf_input.contents().as_ptr() as *const u8,
                buf_a.contents().as_ptr() as *mut u8,
                n * 4,
            );
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 256 * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        // Dispatch 1: MSD histogram (reads buf_a → global_hist[256])
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

        // Dispatch 2: MSD prep (prefix sum → counters + bucket_descs)
        enc.setComputePipelineState(&pso_prep);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(), 4, 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

        // Dispatch 3: Atomic MSD scatter (buf_a → buf_b, no lookback)
        enc.setComputePipelineState(&pso_scatter);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp17_params as *const Exp17Params as *mut _).unwrap(),
                std::mem::size_of::<Exp17Params>(), 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        // Dispatch 4: Fused v4 inner sort (self-contained — computes own histograms)
        let batch_start_0 = 0u32;
        enc.setComputePipelineState(&pso_fused_v4);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&batch_start_0 as *const u32 as *mut _).unwrap(), 4, 3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ms = gpu_elapsed_ms(&cmd);
        if iter >= BENCH_WARMUP { times.push(ms); }

        if iter == BENCH_WARMUP + BENCH_RUNS - 1 {
            let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
            correct = check_correctness(&result, &expected) == 0;
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&times, 50.0);
    let p5 = percentile(&times, 5.0);
    let p95 = percentile(&times, 95.0);
    (p50, p5, p95, correct)
}

fn bench_investigation_w(ctx: &MetalContext) {
    println!("\n  ── Investigation W: Self-Contained Inner (fused_v4) ──");
    println!("  fused_v4 computes own histograms — no precompute dispatch");
    println!("  4 dispatches total (3 MSD + 1 fused-v4)\n");

    let sizes: &[usize] = &[1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000, 32_000_000];

    for &n in sizes {
        let (p50, p5, p95, correct) = bench_investigation_w_at_size(ctx, n);
        let mkeys = n as f64 / p50 / 1e3;
        let status = if correct { "ok" } else { "FAIL" };
        println!("    {:>3} | {:.3}ms | {:>4} Mk/s | {} | [p5={:.3} p95={:.3}]",
            format_size(n), p50, mkeys as u64, status, p5, p95);
    }
}

// ════════════════════════════════════════════════════════════════════
// Investigation W+: Batched Inner Dispatch for SLC Residency
//
// Same MSD phase as Investigation T (3 dispatches).
// Inner phase: fused_v4 dispatched in batches of ~48 TGs.
// Each batch's working set (~12MB) fits in 24MB SLC.
// Expected: inner passes run at SLC speed (469 GB/s) vs DRAM (245 GB/s).
//
// Serialization: separate encoders per batch (0 overhead per MEMORY.md).
// Batches touch disjoint memory (no cross-batch dependencies).
//
// Sweep batch sizes: [16, 32, 48, 64, 128, 256]
// ════════════════════════════════════════════════════════════════════

fn bench_investigation_w_batched(ctx: &MetalContext) {
    println!("\n  ── Investigation W+: Batched Inner for SLC Residency ──");
    println!("  Separate encoders per batch → serialized execution");
    println!("  Working set per batch = batch_size × ~250KB/bucket\n");

    let n: usize = 16_000_000;
    let num_tiles = n.div_ceil(TILE_SIZE);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 256 * 4);
    let buf_counters = alloc_buffer(&ctx.device, 256 * 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());

    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_prep = ctx.make_pipeline("exp17_msd_prep");
    let pso_scatter = ctx.make_pipeline("exp17_msd_atomic_scatter");
    let pso_fused_v4 = ctx.make_pipeline("exp17_inner_fused_v4");

    let exp17_params = Exp17Params {
        element_count: n as u32, num_tiles: num_tiles as u32, shift: 24, pass: 0,
    };
    let tile_size_u32 = TILE_SIZE as u32;

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };

    let batch_sizes: &[u32] = &[16, 32, 48, 64, 128, 256];

    for &batch_sz in batch_sizes {
        let num_batches = (256u32 + batch_sz - 1) / batch_sz;
        let ws_mb = batch_sz as f64 * (n as f64 / 256.0) * 4.0 / 1024.0 / 1024.0;

        let mut times = Vec::new();
        let mut correct = false;

        for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf_input.contents().as_ptr() as *const u8,
                    buf_a.contents().as_ptr() as *mut u8,
                    n * 4,
                );
                std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 256 * 4);
            }

            let cmd = ctx.command_buffer();

            // Encoder 1: MSD phase (3 dispatches)
            {
                let enc = cmd.computeCommandEncoder().unwrap();

                // Dispatch 1: MSD histogram
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

                // Dispatch 2: MSD prep
                enc.setComputePipelineState(&pso_prep);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(), 4, 3,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

                // Dispatch 3: Atomic MSD scatter
                enc.setComputePipelineState(&pso_scatter);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&exp17_params as *const Exp17Params as *mut _).unwrap(),
                        std::mem::size_of::<Exp17Params>(), 3,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

                enc.endEncoding();
            }

            // Inner phase: dispatch fused_v4 in batches with separate encoders
            for batch in 0..num_batches {
                let start = batch * batch_sz;
                let count = (256 - start).min(batch_sz);
                let batch_grid = MTLSize { width: count as usize, height: 1, depth: 1 };

                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&pso_fused_v4);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&start as *const u32 as *mut _).unwrap(), 4, 3,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid, tg_size);
                enc.endEncoding();
            }

            cmd.commit();
            cmd.waitUntilCompleted();

            let ms = gpu_elapsed_ms(&cmd);
            if iter >= BENCH_WARMUP { times.push(ms); }

            if iter == BENCH_WARMUP + BENCH_RUNS - 1 {
                let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
                correct = check_correctness(&result, &expected) == 0;
            }
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&times, 50.0);
        let p5 = percentile(&times, 5.0);
        let p95 = percentile(&times, 95.0);
        let mkeys = n as f64 / p50 / 1e3;
        let status = if correct { "ok" } else { "FAIL" };

        println!("    batch={:>3} ({:>2} batches, ~{:.0}MB/batch) | {:.3}ms | {:>4} Mk/s | {} | [p5={:.3} p95={:.3}]",
            batch_sz, num_batches, ws_mb, p50, mkeys as u64, status, p5, p95);
    }
}

// ════════════════════════════════════════════════════════════════════
// Investigation U: Fused MSD (single dispatch — histogram+scatter)
//
// Single kernel does histogram → global sync → prefix → scatter.
// Eliminates one 64MB read (histogram + scatter share same data load).
// Total: 3 dispatches (1 fused-MSD + 1 precompute + 1 fused-3-pass)
// ════════════════════════════════════════════════════════════════════

fn bench_fused_msd(ctx: &MetalContext) {
    println!("\n  ── Investigation U: Fused MSD (1 dispatch) ──");
    println!("  Single kernel: histogram + global sync + scatter");
    println!("  3 dispatches total (1 fused-MSD + 1 precompute + 1 fused-3-pass)\n");

    let n: usize = 16_000_000;
    let num_tiles = n.div_ceil(TILE_SIZE);

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_global_counts = alloc_buffer(&ctx.device, 256 * 4);
    let buf_completion = alloc_buffer(&ctx.device, 2 * 4);  // [0]=done count, [1]=prefix ready
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());
    let buf_inner_hists = alloc_buffer(&ctx.device, 256 * 3 * 256 * 4);

    let pso_fused_msd = ctx.make_pipeline("exp17_msd_fused_scatter");
    let pso_precompute = ctx.make_pipeline("exp17_inner_precompute_hists");
    let pso_fused_v3 = ctx.make_pipeline("exp17_inner_fused_v3");

    let exp17_params = Exp17Params {
        element_count: n as u32, num_tiles: num_tiles as u32, shift: 24, pass: 0,
    };

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let msd_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let precompute_grid = MTLSize { width: 256, height: 1, depth: 1 };
    let fused_grid = MTLSize { width: 256, height: 1, depth: 1 };

    let mut times = Vec::new();
    let mut correct = false;

    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf_input.contents().as_ptr() as *const u8,
                buf_a.contents().as_ptr() as *mut u8,
                n * 4,
            );
            std::ptr::write_bytes(buf_global_counts.contents().as_ptr() as *mut u8, 0, 256 * 4);
            std::ptr::write_bytes(buf_completion.contents().as_ptr() as *mut u8, 0, 2 * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        // Dispatch 1: Fused MSD (histogram + sync + scatter)
        enc.setComputePipelineState(&pso_fused_msd);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_global_counts.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_completion.as_ref()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp17_params as *const Exp17Params as *mut _).unwrap(),
                std::mem::size_of::<Exp17Params>(), 5,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(msd_grid, tg_size);

        // Dispatch 2: Pre-compute inner histograms
        enc.setComputePipelineState(&pso_precompute);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_inner_hists.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(precompute_grid, tg_size);

        // Dispatch 3: Fused 3-pass inner sort
        enc.setComputePipelineState(&pso_fused_v3);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_inner_hists.as_ref()), 0, 3);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(fused_grid, tg_size);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ms = gpu_elapsed_ms(&cmd);
        if iter >= BENCH_WARMUP { times.push(ms); }

        if iter == BENCH_WARMUP + BENCH_RUNS - 1 {
            let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };
            correct = check_correctness(&result, &expected) == 0;
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&times, 50.0);
    let p5 = percentile(&times, 5.0);
    let p95 = percentile(&times, 95.0);
    let mkeys = n as f64 / p50 / 1e3;
    let status = if correct { "ok" } else { "FAIL" };

    println!("    3 dispatches | {:.3}ms | {:>4} Mk/s | {} | [p5={:.3} p95={:.3}]",
        p50, mkeys as u64, status, p5, p95);
}

// ════════════════════════════════════════════════════════════════════
// Investigation C: Pure 4-pass overhead measurement
//
// Run just 4 passes of 256-bin scatter at 16M (pre-computed offsets)
// to establish the pure scatter ceiling without any radix sort overhead.
// ════════════════════════════════════════════════════════════════════

fn bench_pure_4pass_scatter(ctx: &MetalContext) {
    println!("\n  ── Investigation C: Pure 4-Pass Scatter Ceiling @ 16M ──");
    println!("  4× exp16_diag_scatter_binned with pre-computed 256-bin offsets");
    println!("  Q: What's the absolute maximum for 4-pass 8-bit sort?\n");

    let n: usize = 16_000_000;
    let pso = ctx.make_pipeline("exp16_diag_scatter_binned");
    let n_u32 = n as u32;
    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let grid = MTLSize { width: n.div_ceil(THREADS_PER_TG), height: 1, depth: 1 };

    // Generate data and pre-compute offsets for all 4 passes
    let data = gen_random_u32(n);
    let buf_a = alloc_buffer_with_data(&ctx.device, &data);
    let buf_b = alloc_buffer(&ctx.device, n * 4);

    // Pre-compute: simulate 4-pass LSD by computing offsets at each stage
    // Pass k: digit = (val >> (k*8)) & 0xFF, scatter by digit
    // We pre-compute ALL offsets assuming correct intermediate results
    let mut current = data.clone();
    let mut offset_bufs = Vec::new();

    for pass in 0..4u32 {
        let shift = pass * 8;
        let mask = 0xFFu32;
        let mut hist = vec![0u32; 256];
        for &v in &current {
            hist[((v >> shift) & mask) as usize] += 1;
        }
        let mut prefix = vec![0u32; 256];
        let mut sum = 0u32;
        for i in 0..256 {
            prefix[i] = sum;
            sum += hist[i];
        }
        let mut counters = prefix.clone();
        let mut offsets = vec![0u32; n];
        let mut next = vec![0u32; n];
        for i in 0..n {
            let bin = ((current[i] >> shift) & mask) as usize;
            offsets[i] = counters[bin];
            next[counters[bin] as usize] = current[i];
            counters[bin] += 1;
        }
        offset_bufs.push(alloc_buffer_with_data(&ctx.device, &offsets));
        current = next;
    }

    // Benchmark: 4 scatter passes in single encoder
    let mut times = Vec::new();
    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        // Reset buf_a with original data
        unsafe {
            let src_ptr = data.as_ptr() as *const u8;
            let dst_ptr = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, n * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        for pass in 0..4 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };

            enc.setComputePipelineState(&pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(offset_bufs[pass].as_ref()), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(), 4, 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg_size);
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= BENCH_WARMUP { times.push(gpu_elapsed_ms(&cmd)); }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&times, 50.0);
    let p5 = percentile(&times, 5.0);
    let p95 = percentile(&times, 95.0);
    let mkeys = n as f64 / p50 / 1e3;
    let per_pass = p50 / 4.0;
    let bw_per_pass = (n * 4 * 2) as f64 / per_pass / 1e6;

    println!("    4-pass scatter: p50={:.3} ms  p5={:.3} p95={:.3}", p50, p5, p95);
    println!("    Per pass:       {:.3} ms  ({:.0} GB/s)", per_pass, bw_per_pass);
    println!("    Throughput:     {:.0} Mkeys/s", mkeys);
    println!("    vs exp16:       {:.2}x ({:.0} vs {:.0})", mkeys / BASELINE_MKEYS, mkeys, BASELINE_MKEYS);
    println!();
    println!("    This is the CEILING for 4-pass 8-bit radix sort.");
    println!("    Any real sort must be ≤ this due to histogram/prefix overhead.");

    // Also do 3-pass version
    let mut times_3 = Vec::new();
    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            let src_ptr = data.as_ptr() as *const u8;
            let dst_ptr = buf_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, n * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        for pass in 0..3 {
            let (src_buf, dst_buf) = if pass % 2 == 0 {
                (buf_a.as_ref(), buf_b.as_ref())
            } else {
                (buf_b.as_ref(), buf_a.as_ref())
            };

            enc.setComputePipelineState(&pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(offset_bufs[pass].as_ref()), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(), 4, 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg_size);
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= BENCH_WARMUP { times_3.push(gpu_elapsed_ms(&cmd)); }
    }
    times_3.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_3 = percentile(&times_3, 50.0);
    let mkeys_3 = n as f64 / p50_3 / 1e3;
    println!("\n    3-pass scatter: p50={:.3} ms → {:.0} Mkeys/s (ceiling for 3-pass sort)", p50_3, mkeys_3);
}

// ════════════════════════════════════════════════════════════════════
// Investigation D: 3-pass 11-bit sort — v1 (serial) vs v2 (parallel)
// ════════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Clone, Copy)]
struct V5Params {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    mask: u32,
    pass: u32,
}

const V5_MAX_BINS: usize = 2048;
const V5_TILE_SIZE_3P: usize = 4096;

/// Run the 3-pass 11-bit sort with the given partition PSO and return (times, correct).
fn bench_3pass_generic(
    ctx: &MetalContext,
    pso_hist: &Pso,
    pso_prefix: &Pso,
    pso_zero: &Pso,
    pso_partition: &Pso,
    n: usize,
    input: &[u32],
    expected: &[u32],
) -> (Vec<f64>, bool) {
    let num_tiles = n.div_ceil(V5_TILE_SIZE_3P);
    let shifts: [u32; 3] = [0, 11, 22];
    let masks: [u32; 3] = [0x7FF, 0x7FF, 0x3FF];

    let buf_input = alloc_buffer_with_data(&ctx.device, input);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);
    let buf_global_hist = alloc_buffer(&ctx.device, 3 * V5_MAX_BINS * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * V5_MAX_BINS * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
    let hist_tgs = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let zero_entries = (num_tiles * V5_MAX_BINS) as u32;
    let zero_tg_count = (zero_entries as usize).div_ceil(THREADS_PER_TG);
    let v5_params_size = std::mem::size_of::<V5Params>();

    let encode_sort = |cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>| {
        unsafe {
            std::ptr::write_bytes(
                buf_global_hist.contents().as_ptr() as *mut u8, 0,
                3 * V5_MAX_BINS * 4,
            );
        }

        // Combined histogram
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_hist);
        let element_count = n as u32;
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&element_count as *const u32 as *mut _).unwrap(), 4, 2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_tgs, tg_size);
        enc.endEncoding();

        // Global prefix sum
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        // 3 passes
        for pass in 0..3u32 {
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_zero);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::new(&zero_entries as *const u32 as *mut _).unwrap(), 4, 1,
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

    // 3 passes (odd): result in buf_b
    let results: Vec<u32> = unsafe { read_buffer_slice(&buf_b, n) };
    let correct = results == expected;
    if !correct {
        let mismatches = check_correctness(&results, expected);
        println!("      !! {} / {} mismatched ({:.1}%)",
            mismatches, n, mismatches as f64 / n as f64 * 100.0);
    }

    // Benchmark
    let mut times = Vec::new();
    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
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
        if iter >= BENCH_WARMUP {
            times.push(elapsed);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times, correct)
}

fn bench_3pass_comparison(ctx: &MetalContext) {
    println!("\n  ── Investigation D: 3-Pass 11-Bit Sort ──");
    println!("    v1: serial SG ranking (original, stable)");
    println!("    v2: parallel ranking (unstable, all threads active)\n");

    let pso_hist = ctx.make_pipeline("exp16_3pass_histogram");
    let pso_prefix = ctx.make_pipeline("exp16_3pass_prefix");
    let pso_zero = ctx.make_pipeline("exp16_3pass_zero");
    let pso_v1 = ctx.make_pipeline("exp16_3pass_partition");
    let pso_v2 = ctx.make_pipeline("exp16_3pass_partition_v2");

    let n = 16_000_000usize;
    let input = gen_random_u32(n);
    let mut expected = input.clone();
    expected.sort();

    // v1: serial ranking
    let (times_v1, ok_v1) = bench_3pass_generic(
        ctx, &pso_hist, &pso_prefix, &pso_zero, &pso_v1, n, &input, &expected,
    );
    let p50_v1 = percentile(&times_v1, 50.0);
    let mkeys_v1 = n as f64 / p50_v1 / 1e3;
    let status_v1 = if ok_v1 { "ok" } else { "FAIL" };
    println!("    v1 (serial):   p50={:.3} ms  {:.0} Mkeys/s  {}",
        p50_v1, mkeys_v1, status_v1);

    // v2: parallel ranking
    let (times_v2, ok_v2) = bench_3pass_generic(
        ctx, &pso_hist, &pso_prefix, &pso_zero, &pso_v2, n, &input, &expected,
    );
    let p50_v2 = percentile(&times_v2, 50.0);
    let mkeys_v2 = n as f64 / p50_v2 / 1e3;
    let status_v2 = if ok_v2 { "ok" } else { "FAIL" };
    println!("    v2 (parallel): p50={:.3} ms  {:.0} Mkeys/s  {}",
        p50_v2, mkeys_v2, status_v2);

    let speedup = p50_v1 / p50_v2;
    println!("\n    Speedup: {:.2}x  ({:.0} → {:.0} Mkeys/s)",
        speedup, mkeys_v1, mkeys_v2);
    if mkeys_v2 >= 5000.0 {
        println!("    *** TARGET HIT: {:.0} Mkeys/s >= 5000 ***", mkeys_v2);
    } else {
        let gap = 5000.0 - mkeys_v2;
        println!("    Gap to 5000: {:.0} Mkeys/s ({:.1}%)", gap, gap / 5000.0 * 100.0);
    }

    // v3: large tiles (8192 elements) for SLC-resident tile_status
    let pso_v3 = ctx.make_pipeline("exp16_3pass_partition_v3");
    let v3_tile_size: usize = 8192;
    let v3_num_tiles = n.div_ceil(v3_tile_size);
    let v3_buf_tile_status = alloc_buffer(&ctx.device, v3_num_tiles * V5_MAX_BINS * 4);
    let v3_status_mb = (v3_num_tiles * V5_MAX_BINS * 4) as f64 / 1024.0 / 1024.0;
    println!("    v3 tile_status: {:.1} MB ({} tiles × {} bins) — {}",
        v3_status_mb, v3_num_tiles, V5_MAX_BINS,
        if v3_status_mb <= 24.0 { "SLC-resident" } else { "DRAM" });

    let buf_input_v3 = alloc_buffer_with_data(&ctx.device, &input);
    let buf_a_v3 = alloc_buffer(&ctx.device, n * 4);
    let buf_b_v3 = alloc_buffer(&ctx.device, n * 4);
    let buf_gh_v3 = alloc_buffer(&ctx.device, 3 * V5_MAX_BINS * 4);
    let buf_ctr_v3 = alloc_buffer(&ctx.device, 4);
    let v3_zero_entries = (v3_num_tiles * V5_MAX_BINS) as u32;
    let v3_zero_tg_count = (v3_zero_entries as usize).div_ceil(THREADS_PER_TG);
    let v5_params_size = std::mem::size_of::<V5Params>();
    let v3_shifts: [u32; 3] = [0, 11, 22];
    let v3_masks: [u32; 3] = [0x7FF, 0x7FF, 0x3FF];
    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
    // Histogram kernel uses V5_TILE_SIZE=4096, not v3's 8192
    let v3_hist_tgs = n.div_ceil(4096);

    // Correctness check for v3
    unsafe {
        let src = buf_input_v3.contents().as_ptr() as *const u8;
        let dst = buf_a_v3.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
        std::ptr::write_bytes(buf_gh_v3.contents().as_ptr() as *mut u8, 0, 3 * V5_MAX_BINS * 4);
    }
    {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_hist);
        let element_count = n as u32;
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a_v3.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_gh_v3.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&element_count as *const u32 as *mut _).unwrap(), 4, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: v3_hist_tgs, height: 1, depth: 1 }, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_prefix);
        unsafe { enc.setBuffer_offset_atIndex(Some(buf_gh_v3.as_ref()), 0, 0); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        for pass in 0..3u32 {
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_zero);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(v3_buf_tile_status.as_ref()), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::new(&v3_zero_entries as *const u32 as *mut _).unwrap(), 4, 1);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: v3_zero_tg_count, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();

            let params = V5Params {
                element_count: n as u32, num_tiles: v3_num_tiles as u32,
                shift: v3_shifts[pass as usize], mask: v3_masks[pass as usize], pass,
            };
            let (s, d) = if pass % 2 == 0 { (buf_a_v3.as_ref(), buf_b_v3.as_ref()) }
                         else { (buf_b_v3.as_ref(), buf_a_v3.as_ref()) };
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_v3);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(d), 0, 1);
                enc.setBuffer_offset_atIndex(Some(v3_buf_tile_status.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_gh_v3.as_ref()), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const V5Params as *mut _).unwrap(), v5_params_size, 4);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: v3_num_tiles, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    let results_v3: Vec<u32> = unsafe { read_buffer_slice(&buf_b_v3, n) };
    let ok_v3 = results_v3 == expected;
    if !ok_v3 {
        let mis = check_correctness(&results_v3, &expected);
        println!("      !! v3: {} / {} mismatched ({:.1}%)", mis, n, mis as f64 / n as f64 * 100.0);
    }

    // Benchmark v3
    let mut times_v3 = Vec::new();
    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            let src = buf_input_v3.contents().as_ptr() as *const u8;
            let dst = buf_a_v3.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_gh_v3.contents().as_ptr() as *mut u8, 0, 3 * V5_MAX_BINS * 4);
        }
        let cmd = ctx.command_buffer();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_hist);
        let element_count = n as u32;
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a_v3.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_gh_v3.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&element_count as *const u32 as *mut _).unwrap(), 4, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: v3_hist_tgs, height: 1, depth: 1 }, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_prefix);
        unsafe { enc.setBuffer_offset_atIndex(Some(buf_gh_v3.as_ref()), 0, 0); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        for pass in 0..3u32 {
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_zero);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(v3_buf_tile_status.as_ref()), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::new(&v3_zero_entries as *const u32 as *mut _).unwrap(), 4, 1);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: v3_zero_tg_count, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();

            let params = V5Params {
                element_count: n as u32, num_tiles: v3_num_tiles as u32,
                shift: v3_shifts[pass as usize], mask: v3_masks[pass as usize], pass,
            };
            let (s, d) = if pass % 2 == 0 { (buf_a_v3.as_ref(), buf_b_v3.as_ref()) }
                         else { (buf_b_v3.as_ref(), buf_a_v3.as_ref()) };
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_v3);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(d), 0, 1);
                enc.setBuffer_offset_atIndex(Some(v3_buf_tile_status.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_gh_v3.as_ref()), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const V5Params as *mut _).unwrap(), v5_params_size, 4);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: v3_num_tiles, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();
        }

        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= BENCH_WARMUP { times_v3.push(gpu_elapsed_ms(&cmd)); }
    }
    times_v3.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_v3 = percentile(&times_v3, 50.0);
    let mkeys_v3 = n as f64 / p50_v3 / 1e3;
    let status_v3 = if ok_v3 { "ok" } else { "FAIL" };
    println!("    v3 (8K tiles): p50={:.3} ms  {:.0} Mkeys/s  {}",
        p50_v3, mkeys_v3, status_v3);
    if mkeys_v3 >= 5000.0 {
        println!("    *** TARGET HIT: {:.0} Mkeys/s >= 5000 ***", mkeys_v3);
    }

    // Also benchmark 8-bit 4-pass for reference
    println!("\n    Reference: 8-bit 4-pass sort");
    let pso_8bit_hist = ctx.make_pipeline("exp16_combined_histogram");
    let pso_8bit_prefix = ctx.make_pipeline("exp16_global_prefix");
    let pso_8bit_zero = ctx.make_pipeline("exp16_zero_status");
    let pso_8bit_part = ctx.make_pipeline("exp16_partition");

    let num_tiles_8 = n.div_ceil(TILE_SIZE);
    let buf_input_8 = alloc_buffer_with_data(&ctx.device, &input);
    let buf_a_8 = alloc_buffer(&ctx.device, n * 4);
    let buf_b_8 = alloc_buffer(&ctx.device, n * 4);
    let buf_gh_8 = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_ts_8 = alloc_buffer(&ctx.device, num_tiles_8 * 256 * 4);
    let buf_ctr_8 = alloc_buffer(&ctx.device, 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
    let params_size_8 = std::mem::size_of::<Exp16Params>();
    let zero_tg_count_8 = (num_tiles_8 * 256).div_ceil(THREADS_PER_TG);

    let mut times_8 = Vec::new();
    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            let s = buf_input_8.contents().as_ptr() as *const u8;
            let d = buf_a_8.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(s, d, n * 4);
            std::ptr::write_bytes(buf_gh_8.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
        }
        let cmd = ctx.command_buffer();

        // Combined histogram
        let params = Exp16Params { element_count: n as u32, num_tiles: num_tiles_8 as u32,
            num_tgs: num_tiles_8 as u32, shift: 0, pass: 0 };
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_8bit_hist);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a_8.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_gh_8.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&params as *const Exp16Params as *mut _).unwrap(), params_size_8, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: num_tiles_8, height: 1, depth: 1 }, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_8bit_prefix);
        unsafe { enc.setBuffer_offset_atIndex(Some(buf_gh_8.as_ref()), 0, 0); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        for pass in 0..4u32 {
            let p = Exp16Params { element_count: n as u32, num_tiles: num_tiles_8 as u32,
                num_tgs: num_tiles_8 as u32, shift: pass * 8, pass };
            let (s, d) = if pass % 2 == 0 { (buf_a_8.as_ref(), buf_b_8.as_ref()) }
                         else { (buf_b_8.as_ref(), buf_a_8.as_ref()) };

            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_8bit_zero);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_ts_8.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_ctr_8.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&p as *const Exp16Params as *mut _).unwrap(), params_size_8, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: zero_tg_count_8, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();

            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_8bit_part);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(d), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_ts_8.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_ctr_8.as_ref()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(buf_gh_8.as_ref()), 0, 4);
                enc.setBytes_length_atIndex(
                    NonNull::new(&p as *const Exp16Params as *mut _).unwrap(), params_size_8, 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tiles_8, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();
        }

        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= BENCH_WARMUP { times_8.push(gpu_elapsed_ms(&cmd)); }
    }
    times_8.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_8 = percentile(&times_8, 50.0);
    let mkeys_8 = n as f64 / p50_8 / 1e3;
    println!("    8-bit 4-pass:  p50={:.3} ms  {:.0} Mkeys/s", p50_8, mkeys_8);

    // ── Investigation E: Coalesced scatter (v2) vs random scatter ──
    println!("\n  ── Investigation E: Coalesced Scatter (TG Reorder) ──");
    let pso_v2_part = ctx.make_pipeline("exp16_partition_v2");

    let v2_tile_size: usize = 2048;
    let v2_num_tiles = n.div_ceil(v2_tile_size);
    let v2_status_mb = (v2_num_tiles * 256 * 4) as f64 / 1024.0 / 1024.0;
    println!("    v2 tiles: {} (2K each), tile_status: {:.1} MB",
        v2_num_tiles, v2_status_mb);

    let buf_input_v2 = alloc_buffer_with_data(&ctx.device, &input);
    let buf_a_v2 = alloc_buffer(&ctx.device, n * 4);
    let buf_b_v2 = alloc_buffer(&ctx.device, n * 4);
    let buf_gh_v2 = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_ts_v2 = alloc_buffer(&ctx.device, v2_num_tiles * 256 * 4);
    let buf_ctr_v2 = alloc_buffer(&ctx.device, 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
    let v2_zero_tg_count = (v2_num_tiles * 256).div_ceil(THREADS_PER_TG);
    let params_size = std::mem::size_of::<Exp16Params>();

    // Correctness check
    unsafe {
        let src = buf_input_v2.contents().as_ptr() as *const u8;
        let dst = buf_a_v2.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
        std::ptr::write_bytes(buf_gh_v2.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
    }
    {
        let cmd = ctx.command_buffer();
        // Histogram (uses original tile count — histogram kernel uses 4096-element tiles)
        let p0 = Exp16Params { element_count: n as u32, num_tiles: num_tiles_8 as u32,
            num_tgs: num_tiles_8 as u32, shift: 0, pass: 0 };
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_8bit_hist);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a_v2.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_gh_v2.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&p0 as *const Exp16Params as *mut _).unwrap(), params_size, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: num_tiles_8, height: 1, depth: 1 }, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_8bit_prefix);
        unsafe { enc.setBuffer_offset_atIndex(Some(buf_gh_v2.as_ref()), 0, 0); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        for pass in 0..4u32 {
            let p = Exp16Params { element_count: n as u32, num_tiles: v2_num_tiles as u32,
                num_tgs: v2_num_tiles as u32, shift: pass * 8, pass };
            let (s, d) = if pass % 2 == 0 { (buf_a_v2.as_ref(), buf_b_v2.as_ref()) }
                         else { (buf_b_v2.as_ref(), buf_a_v2.as_ref()) };
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_8bit_zero);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_ts_v2.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_ctr_v2.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&p as *const Exp16Params as *mut _).unwrap(), params_size, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: v2_zero_tg_count, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();

            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_v2_part);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(d), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_ts_v2.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_ctr_v2.as_ref()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(buf_gh_v2.as_ref()), 0, 4);
                enc.setBytes_length_atIndex(
                    NonNull::new(&p as *const Exp16Params as *mut _).unwrap(), params_size, 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: v2_num_tiles, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    // 4 passes (even): result in buf_a_v2
    let results_v2f: Vec<u32> = unsafe { read_buffer_slice(&buf_a_v2, n) };
    let ok_v2f = results_v2f == expected;
    if !ok_v2f {
        let mis = check_correctness(&results_v2f, &expected);
        println!("    !! v2-coalesced: {} / {} mismatched ({:.1}%)",
            mis, n, mis as f64 / n as f64 * 100.0);
    } else {
        println!("    v2-coalesced: correctness OK");
    }

    // Benchmark v2 coalesced
    let mut times_v2f = Vec::new();
    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            let src = buf_input_v2.contents().as_ptr() as *const u8;
            let dst = buf_a_v2.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_gh_v2.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
        }
        let cmd = ctx.command_buffer();

        let p0 = Exp16Params { element_count: n as u32, num_tiles: num_tiles_8 as u32,
            num_tgs: num_tiles_8 as u32, shift: 0, pass: 0 };
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_8bit_hist);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a_v2.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_gh_v2.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&p0 as *const Exp16Params as *mut _).unwrap(), params_size, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: num_tiles_8, height: 1, depth: 1 }, tg_size);
        enc.endEncoding();

        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_8bit_prefix);
        unsafe { enc.setBuffer_offset_atIndex(Some(buf_gh_v2.as_ref()), 0, 0); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
        enc.endEncoding();

        for pass in 0..4u32 {
            let p = Exp16Params { element_count: n as u32, num_tiles: v2_num_tiles as u32,
                num_tgs: v2_num_tiles as u32, shift: pass * 8, pass };
            let (s, d) = if pass % 2 == 0 { (buf_a_v2.as_ref(), buf_b_v2.as_ref()) }
                         else { (buf_b_v2.as_ref(), buf_a_v2.as_ref()) };
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_8bit_zero);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_ts_v2.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_ctr_v2.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&p as *const Exp16Params as *mut _).unwrap(), params_size, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: v2_zero_tg_count, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();

            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_v2_part);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(d), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_ts_v2.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_ctr_v2.as_ref()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(buf_gh_v2.as_ref()), 0, 4);
                enc.setBytes_length_atIndex(
                    NonNull::new(&p as *const Exp16Params as *mut _).unwrap(), params_size, 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: v2_num_tiles, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= BENCH_WARMUP { times_v2f.push(gpu_elapsed_ms(&cmd)); }
    }
    times_v2f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_v2f = percentile(&times_v2f, 50.0);
    let mkeys_v2f = n as f64 / p50_v2f / 1e3;
    let status_v2f = if ok_v2f { "ok" } else { "FAIL" };
    println!("    v2-coalesced:  p50={:.3} ms  {:.0} Mkeys/s  {}",
        p50_v2f, mkeys_v2f, status_v2f);
    println!("    vs original:   {:.2}x ({:.0} vs {:.0} Mkeys/s)",
        mkeys_v2f / mkeys_8, mkeys_v2f, mkeys_8);
    if mkeys_v2f >= 5000.0 {
        println!("    *** TARGET HIT: {:.0} Mkeys/s >= 5000 ***", mkeys_v2f);
    }
}

// ════════════════════════════════════════════════════════════════════
// Main entry point — INVESTIGATION MODE
// ════════════════════════════════════════════════════════════════════

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 17: GPU Sort Investigation");
    println!("{}", "=".repeat(60));
    println!("  Current: 3504 Mkeys/s (hybrid MSD+3 LSD)");
    println!("  Target:  5000 Mkeys/s @ 16M uint32");
    println!("  Gap:     1496 Mkeys/s\n");

    // Investigation W: Self-contained inner (fused_v4) — size sweep
    bench_investigation_w(ctx);

    // Dead code below — previous investigations G through P retained for reference
    #[allow(unreachable_code)]
    return;

    // Investigation G: Batched inner sort — sweep batch sizes for SLC effect
    let n = 16_000_000usize;
    let full_mkeys = 3456.0f64; // placeholder for dead code references
    println!("\n  ── Investigation G: Batched Inner Sort (SLC Sweep) ──");
    let batch_sizes_to_test: &[usize] = &[4, 16, 64, 256];
    let max_tpb: usize = 17;  // EXP17_MAX_TPB

    for &buckets_per_batch in batch_sizes_to_test {
    let num_batches = 256usize.div_ceil(buckets_per_batch);
    let tgs_per_batch = buckets_per_batch * max_tpb;
    let batch_data_mb = buckets_per_batch as f64 * (n as f64 / 256.0) * 4.0 / 1024.0 / 1024.0;

    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix_b = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition_b = ctx.make_pipeline("exp16_partition");
    let pso_inner_zero = ctx.make_pipeline("exp17_inner_zero");
    let pso_inner_histogram = ctx.make_pipeline("exp17_inner_histogram");
    let pso_inner_scan_scatter = ctx.make_pipeline("exp17_inner_scan_scatter");

    let num_tiles = n.div_ceil(TILE_SIZE);
    let tile_hists_entries: usize = 256 * max_tpb * 256;

    let data = gen_random_u32(n);
    let mut expected = data.clone();
    expected.sort();

    let buf_input_g = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a_g = alloc_buffer(&ctx.device, n * 4);
    let buf_b_g = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_tile_status_g = alloc_buffer(&ctx.device, num_tiles * 256 * 4);
    let buf_counters_g = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * 16); // BucketDesc = 4 × u32 = 16B
    let buf_tile_hists = alloc_buffer(&ctx.device, tile_hists_entries * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let one_tg_b = MTLSize { width: 1, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
    let tile_hists_total = tile_hists_entries as u32;
    let inner_zero_tg_count = tile_hists_entries.div_ceil(THREADS_PER_TG);
    let inner_zero_grid = MTLSize { width: inner_zero_tg_count, height: 1, depth: 1 };

    let exp17_params = Exp17Params {
        element_count: n as u32, num_tiles: num_tiles as u32, shift: 24, pass: 0,
    };
    let exp16_params = Exp16Params {
        element_count: n as u32, num_tiles: num_tiles as u32,
        num_tgs: num_tiles as u32, shift: 24, pass: 0,
    };
    let tile_size_u32 = TILE_SIZE as u32;
    let inner_shifts: [u32; 3] = [0, 8, 16];

    let mut times_batched = Vec::new();
    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            let src = buf_input_g.contents().as_ptr() as *const u8;
            let dst = buf_a_g.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
        }

        let cmd = ctx.command_buffer();

        // MSD phase (single encoder, 5 dispatches)
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_histogram);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a_g.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp17_params as *const Exp17Params as *mut _).unwrap(),
                std::mem::size_of::<Exp17Params>(), 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

        enc.setComputePipelineState(&pso_bucket_descs);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(), 4, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_b, tg_size);

        enc.setComputePipelineState(&pso_prefix_b);
        unsafe { enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_b, tg_size);

        enc.setComputePipelineState(&pso_zero_status);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_status_g.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_counters_g.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp16_params as *const Exp16Params as *mut _).unwrap(),
                std::mem::size_of::<Exp16Params>(), 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(zero_grid, tg_size);

        enc.setComputePipelineState(&pso_partition_b);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a_g.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b_g.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_status_g.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_counters_g.as_ref()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp16_params as *const Exp16Params as *mut _).unwrap(),
                std::mem::size_of::<Exp16Params>(), 5);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);
        enc.endEncoding();

        // Inner sort: batched for SLC residency
        // Zero tile_hists once
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&pso_inner_zero);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 0);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_hists_total as *const u32 as *mut _).unwrap(), 4, 1);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(inner_zero_grid, tg_size);
        enc.endEncoding();

        // For each batch of buckets, do all 3 inner passes
        for batch in 0..num_batches {
            let tg_off = (batch * buckets_per_batch * max_tpb) as u32;
            let batch_tg_count = tgs_per_batch.min(256 * max_tpb - batch * tgs_per_batch);
            let batch_grid = MTLSize { width: batch_tg_count, height: 1, depth: 1 };

            for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
                // Zero tile_hists for this batch's portion
                let batch_hist_entries = (batch_tg_count * 256) as u32;
                let batch_zero_tgs = (batch_hist_entries as usize).div_ceil(THREADS_PER_TG);

                let enc = cmd.computeCommandEncoder().unwrap();

                // Zero just this batch's portion of tile_hists
                // (we set tg_offset in the params so the zero writes to the right location)
                // Actually, tile_hists for this batch starts at tg_off * 256
                // The inner_zero kernel zeros tile_hists from tid=0. We need to offset it.
                // Simplest: just zero the whole thing once before each pass.
                // For now, skip re-zeroing since we zeroed the whole buffer above for pass 0.
                // For passes 1 and 2, we need to re-zero the batch's portion.
                if pass_idx > 0 {
                    enc.setComputePipelineState(&pso_inner_zero);
                    unsafe {
                        // Zero the batch portion: offset = tg_off * 256 * 4 bytes
                        // But inner_zero zeros from tid=0. We need batch-aware zeroing.
                        // Simplest: zero the whole tile_hists before each pass.
                        enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 0);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&tile_hists_total as *const u32 as *mut _).unwrap(), 4, 1);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(inner_zero_grid, tg_size);
                }

                let inner_params = Exp17InnerParams { shift, tg_offset: tg_off };
                let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                    (buf_b_g.as_ref(), buf_a_g.as_ref())
                } else {
                    (buf_a_g.as_ref(), buf_b_g.as_ref())
                };

                enc.setComputePipelineState(&pso_inner_histogram);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                        std::mem::size_of::<Exp17InnerParams>(), 3);
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
                        std::mem::size_of::<Exp17InnerParams>(), 4);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid, tg_size);

                enc.endEncoding();  // Barrier: histogram+scatter must finish before next pass
            }
        }

        cmd.commit();
        cmd.waitUntilCompleted();

        if iter >= BENCH_WARMUP { times_batched.push(gpu_elapsed_ms(&cmd)); }

        // Correctness check on first real iteration
        if iter == BENCH_WARMUP {
            // After MSD (1 pass, even→buf_b) + 3 inner passes (odd count→buf_a)
            let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a_g, n) };
            let mis = check_correctness(&result, &expected);
            if mis == 0 {
                println!("    Batched: correctness OK");
            } else {
                println!("    !! Batched: {} / {} mismatched ({:.1}%)",
                    mis, n, mis as f64 / n as f64 * 100.0);
            }
        }
    }
    times_batched.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_batched = percentile(&times_batched, 50.0);
    let mkeys_batched = n as f64 / p50_batched / 1e3;
    println!("    batch={:>3}: p50={:.3} ms  {:.0} Mkeys/s  {:.2}x vs unbatched  ({:.1} MB/batch)",
        buckets_per_batch, p50_batched, mkeys_batched,
        mkeys_batched / full_mkeys, batch_data_mb);
    if mkeys_batched >= 5000.0 {
        println!("    *** TARGET HIT: {:.0} Mkeys/s >= 5000 ***", mkeys_batched);
    }
    } // end batch_sizes_to_test sweep

    // ══════════════════════════════════════════════════════════════
    // Investigation I: Single-Encoder Batched Inner Sort
    // Hypothesis: Investigation G batching was slow because of
    // encoder barriers (~45us each). With ALL dispatches in ONE
    // encoder, we eliminate barrier overhead. Batch-first ordering
    // ensures each batch's ~16MB working set stays SLC-resident
    // across all 3 inner passes. Scan_scatter re-reads data that
    // histogram just cached → SLC hit instead of DRAM re-read.
    // ══════════════════════════════════════════════════════════════
    println!("\n  ── Investigation I: Single-Encoder Batched (SLC) ──");
    println!("    Key: ALL dispatches in 1 encoder, batch-first ordering");
    println!("    Goal: SLC hits on scan_scatter re-reads of histogram data\n");

    let batch_sizes_i: &[usize] = &[8, 16, 32, 64, 128, 256];
    let max_tpb_i: usize = 17;

    for &bpb in batch_sizes_i {
    let num_batches_i = 256usize.div_ceil(bpb);
    let tgs_per_batch_i = bpb * max_tpb_i;
    let batch_data_mb = bpb as f64 * (n as f64 / 256.0) * 4.0 / 1024.0 / 1024.0;

    let pso_histogram_i = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs_i = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix_i = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status_i = ctx.make_pipeline("exp16_zero_status");
    let pso_partition_i = ctx.make_pipeline("exp16_partition");
    let pso_inner_zero_i = ctx.make_pipeline("exp17_inner_zero");
    let pso_inner_histogram_i = ctx.make_pipeline("exp17_inner_histogram");
    let pso_inner_scan_scatter_i = ctx.make_pipeline("exp17_inner_scan_scatter");

    let num_tiles_i = n.div_ceil(TILE_SIZE);
    let tile_hists_entries_i: usize = 256 * max_tpb_i * 256;

    let data_i = gen_random_u32(n);
    let mut expected_i = data_i.clone();
    expected_i.sort();

    let buf_input_i = alloc_buffer_with_data(&ctx.device, &data_i);
    let buf_a_i = alloc_buffer(&ctx.device, n * 4);
    let buf_b_i = alloc_buffer(&ctx.device, n * 4);
    let buf_msd_hist_i = alloc_buffer(&ctx.device, 4 * 256 * 4);
    let buf_tile_status_i = alloc_buffer(&ctx.device, num_tiles_i * 256 * 4);
    let buf_counters_i = alloc_buffer(&ctx.device, 4);
    let buf_bucket_descs_i = alloc_buffer(&ctx.device, 256 * 16);
    let buf_tile_hists_i = alloc_buffer(&ctx.device, tile_hists_entries_i * 4);

    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let one_tg_i = MTLSize { width: 1, height: 1, depth: 1 };
    let hist_grid_i = MTLSize { width: num_tiles_i, height: 1, depth: 1 };
    let zero_tg_count_i = (num_tiles_i * 256).div_ceil(THREADS_PER_TG);
    let zero_grid_i = MTLSize { width: zero_tg_count_i, height: 1, depth: 1 };
    let batch_grid_i = MTLSize { width: tgs_per_batch_i, height: 1, depth: 1 };

    // Batch-aware zero: entries per batch region
    let batch_tile_hists_entries = (bpb * max_tpb_i * 256) as u32;
    let batch_zero_tgs = (batch_tile_hists_entries as usize).div_ceil(THREADS_PER_TG);
    let batch_zero_grid = MTLSize { width: batch_zero_tgs, height: 1, depth: 1 };

    let exp17_params_i = Exp17Params {
        element_count: n as u32, num_tiles: num_tiles_i as u32, shift: 24, pass: 0,
    };
    let exp16_params_i = Exp16Params {
        element_count: n as u32, num_tiles: num_tiles_i as u32,
        num_tgs: num_tiles_i as u32, shift: 24, pass: 0,
    };
    let tile_size_u32_i = TILE_SIZE as u32;
    let inner_shifts_i: [u32; 3] = [0, 8, 16];

    let mut times_i = Vec::new();

    for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
        unsafe {
            let src = buf_input_i.contents().as_ptr() as *const u8;
            let dst = buf_a_i.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            std::ptr::write_bytes(buf_msd_hist_i.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        // ── MSD phase (5 dispatches, same encoder) ──
        enc.setComputePipelineState(&pso_histogram_i);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a_i.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist_i.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp17_params_i as *const Exp17Params as *mut _).unwrap(),
                std::mem::size_of::<Exp17Params>(), 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid_i, tg_size);

        enc.setComputePipelineState(&pso_bucket_descs_i);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist_i.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_i.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_size_u32_i as *const u32 as *mut _).unwrap(), 4, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_i, tg_size);

        enc.setComputePipelineState(&pso_prefix_i);
        unsafe { enc.setBuffer_offset_atIndex(Some(buf_msd_hist_i.as_ref()), 0, 0); }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_i, tg_size);

        enc.setComputePipelineState(&pso_zero_status_i);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_status_i.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_counters_i.as_ref()), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp16_params_i as *const Exp16Params as *mut _).unwrap(),
                std::mem::size_of::<Exp16Params>(), 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(zero_grid_i, tg_size);

        enc.setComputePipelineState(&pso_partition_i);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_a_i.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_b_i.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_status_i.as_ref()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_counters_i.as_ref()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(buf_msd_hist_i.as_ref()), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(&exp16_params_i as *const Exp16Params as *mut _).unwrap(),
                std::mem::size_of::<Exp16Params>(), 5);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid_i, tg_size);

        // ── Inner sort: single encoder, batch-first ordering ──
        // Each batch does all 3 passes before moving to next batch.
        // This keeps the batch's working set SLC-resident across passes.
        for batch in 0..num_batches_i {
            let tg_off = (batch * bpb * max_tpb_i) as u32;
            let batch_offset_bytes = batch * bpb * max_tpb_i * 256 * 4; // byte offset into tile_hists

            for (pass_idx, &shift) in inner_shifts_i.iter().enumerate() {
                let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                    (buf_b_i.as_ref(), buf_a_i.as_ref())
                } else {
                    (buf_a_i.as_ref(), buf_b_i.as_ref())
                };

                // Zero this batch's tile_hists region using buffer offset
                enc.setComputePipelineState(&pso_inner_zero_i);
                unsafe {
                    enc.setBuffer_offset_atIndex(
                        Some(buf_tile_hists_i.as_ref()), batch_offset_bytes, 0);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&batch_tile_hists_entries as *const u32 as *mut _).unwrap(),
                        4, 1);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(batch_zero_grid, tg_size);

                // Histogram: reads data, writes to tile_hists (full buffer, absolute indexing)
                let inner_params = Exp17InnerParams { shift, tg_offset: tg_off };
                enc.setComputePipelineState(&pso_inner_histogram_i);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(buf_tile_hists_i.as_ref()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_i.as_ref()), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                        std::mem::size_of::<Exp17InnerParams>(), 3);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid_i, tg_size);

                // Scan+scatter: reads data + tile_hists, writes to dst
                enc.setComputePipelineState(&pso_inner_scan_scatter_i);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_tile_hists_i.as_ref()), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_i.as_ref()), 0, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                        std::mem::size_of::<Exp17InnerParams>(), 4);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid_i, tg_size);
            }
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ms = gpu_elapsed_ms(&cmd);
        if iter >= BENCH_WARMUP { times_i.push(ms); }

        // Correctness check on first real iteration
        if iter == BENCH_WARMUP {
            let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a_i, n) };
            let mis = check_correctness(&result, &expected_i);
            if mis == 0 {
                println!("    batch={:>3}: correctness OK", bpb);
            } else {
                println!("    !! batch={:>3}: {} / {} mismatched ({:.1}%)",
                    bpb, mis, n, mis as f64 / n as f64 * 100.0);
            }
        }
    }
    times_i.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_i = percentile(&times_i, 50.0);
    let mkeys_i = n as f64 / p50_i / 1e3;
    println!("    batch={:>3}: p50={:.3} ms  {:.0} Mkeys/s  {:.2}x vs full  ({:.1} MB/batch)",
        bpb, p50_i, mkeys_i, mkeys_i / full_mkeys, batch_data_mb);
    if mkeys_i >= 5000.0 {
        println!("    *** TARGET HIT: {:.0} Mkeys/s >= 5000 ***", mkeys_i);
    }
    } // end batch_sizes_i sweep

    // ══════════════════════════════════════════════════════════════
    // Investigation H: Fused Inner Partition — ABANDONED
    // Fan-in spin-wait on bucket_ready causes GPU livelock:
    // 20KB TG memory → 1 TG/core → spinners starve last tiles.
    // Even batched dispatch (16 buckets/encoder) only reduces
    // deadlock from 10s to 676ms, introduces MORE mismatches.
    // The 192MB savings (~0.78ms at DRAM) isn't worth the complexity.
    // ══════════════════════════════════════════════════════════════
    if false { // DISABLED — see comment above
    println!("\n  ── Investigation H: Fused Inner Partition ──");
    println!("    Kernel: exp17_inner_partition (lookback within bucket)");
    println!("    Saves: 64 MB/pass × 3 = 192 MB total data reads\n");

    {
        let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
        let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
        let pso_prefix_h = ctx.make_pipeline("exp17_global_prefix");
        let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
        let pso_partition_h = ctx.make_pipeline("exp16_partition");
        let pso_inner_zero = ctx.make_pipeline("exp17_inner_zero");
        let pso_inner_partition = ctx.make_pipeline("exp17_inner_partition");

        let num_tiles = n.div_ceil(TILE_SIZE);
        let max_tpb: usize = 17;
        let inner_tg_count: usize = 256 * max_tpb;
        let tile_status_entries: usize = 256 * max_tpb * 256; // same size as tile_hists

        let data = gen_random_u32(n);
        let mut expected = data.clone();
        expected.sort();

        let buf_input_h = alloc_buffer_with_data(&ctx.device, &data);
        let buf_a_h = alloc_buffer(&ctx.device, n * 4);
        let buf_b_h = alloc_buffer(&ctx.device, n * 4);
        let buf_msd_hist_h = alloc_buffer(&ctx.device, 4 * 256 * 4);
        let buf_tile_status_msd = alloc_buffer(&ctx.device, num_tiles * 256 * 4);
        let buf_counters_h = alloc_buffer(&ctx.device, 4);
        let buf_bucket_descs_h = alloc_buffer(&ctx.device, 256 * 16);
        // Inner partition buffers
        let buf_inner_tile_status = alloc_buffer(&ctx.device, tile_status_entries * 4);
        let buf_bucket_digit_pfx = alloc_buffer(&ctx.device, 256 * 256 * 4); // 256KB
        let buf_bucket_ready = alloc_buffer(&ctx.device, 256 * 4); // 1KB

        let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
        let one_tg_h = MTLSize { width: 1, height: 1, depth: 1 };
        let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
        let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
        let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
        let inner_grid = MTLSize { width: inner_tg_count, height: 1, depth: 1 };
        let inner_zero_tg_count = tile_status_entries.div_ceil(THREADS_PER_TG);
        let inner_zero_grid = MTLSize { width: inner_zero_tg_count, height: 1, depth: 1 };
        // bucket_ready is only 256 uints = 1 TG
        let ready_zero_entries = 256u32;
        let ready_zero_grid = MTLSize { width: 1, height: 1, depth: 1 };

        let exp17_params = Exp17Params {
            element_count: n as u32, num_tiles: num_tiles as u32, shift: 24, pass: 0,
        };
        let exp16_params = Exp16Params {
            element_count: n as u32, num_tiles: num_tiles as u32,
            num_tgs: num_tiles as u32, shift: 24, pass: 0,
        };
        let tile_size_u32 = TILE_SIZE as u32;
        let tile_status_total = tile_status_entries as u32;
        let inner_shifts: [u32; 3] = [0, 8, 16];

        let mut times_fused = Vec::new();
        let mut correct_fused = false;

        for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
            unsafe {
                let src = buf_input_h.contents().as_ptr() as *const u8;
                let dst = buf_a_h.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src, dst, n * 4);
                std::ptr::write_bytes(buf_msd_hist_h.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
            }

            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();

            // ── MSD phase (5 dispatches, identical to bench_hybrid) ──
            enc.setComputePipelineState(&pso_histogram);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a_h.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_h.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp17_params as *const Exp17Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp17Params>(), 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

            enc.setComputePipelineState(&pso_bucket_descs);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_h.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_h.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(), 4, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_h, tg_size);

            enc.setComputePipelineState(&pso_prefix_h);
            unsafe { enc.setBuffer_offset_atIndex(Some(buf_msd_hist_h.as_ref()), 0, 0); }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_h, tg_size);

            enc.setComputePipelineState(&pso_zero_status);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_tile_status_msd.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_counters_h.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp16_params as *const Exp16Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp16Params>(), 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(zero_grid, tg_size);

            enc.setComputePipelineState(&pso_partition_h);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a_h.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_b_h.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_tile_status_msd.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_counters_h.as_ref()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_h.as_ref()), 0, 4);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp16_params as *const Exp16Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp16Params>(), 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);
            enc.endEncoding();

            // ── Inner sort: 3 passes × batched inner_partition ──
            // Dispatch in batches of 16 buckets (272 TGs) per encoder to prevent
            // GPU livelock: with all 4352 TGs in one dispatch, ~4096 non-last tiles
            // spin on bucket_ready, starving the 256 last tiles from getting scheduled.
            // Batching limits concurrent spinners to ~256 per encoder.
            let buckets_per_batch: usize = 16;
            let num_batches = 256 / buckets_per_batch; // = 16

            for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
                let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                    (buf_b_h.as_ref(), buf_a_h.as_ref())
                } else {
                    (buf_a_h.as_ref(), buf_b_h.as_ref())
                };

                // Encoder 0: Zero tile_status + bucket_ready
                {
                    let enc = cmd.computeCommandEncoder().unwrap();
                    enc.setComputePipelineState(&pso_inner_zero);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(buf_inner_tile_status.as_ref()), 0, 0);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&tile_status_total as *const u32 as *mut _).unwrap(), 4, 1);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(inner_zero_grid, tg_size);

                    enc.setComputePipelineState(&pso_inner_zero);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_ready.as_ref()), 0, 0);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&ready_zero_entries as *const u32 as *mut _).unwrap(), 4, 1);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(ready_zero_grid, tg_size);
                    enc.endEncoding();
                }

                // Encoders 1-16: Batched inner_partition (16 buckets per batch)
                for batch in 0..num_batches {
                    let tg_offset = (batch * buckets_per_batch * max_tpb) as u32;
                    let batch_tg_count = buckets_per_batch * max_tpb; // 16 * 17 = 272
                    let batch_grid = MTLSize { width: batch_tg_count, height: 1, depth: 1 };
                    let inner_params = Exp17InnerParams { shift, tg_offset };

                    let enc = cmd.computeCommandEncoder().unwrap();
                    enc.setComputePipelineState(&pso_inner_partition);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(buf_inner_tile_status.as_ref()), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_h.as_ref()), 0, 3);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_digit_pfx.as_ref()), 0, 4);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_ready.as_ref()), 0, 5);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                            std::mem::size_of::<Exp17InnerParams>(), 6);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid, tg_size);
                    enc.endEncoding();
                }
            }

            cmd.commit();
            cmd.waitUntilCompleted();

            let ms = gpu_elapsed_ms(&cmd);
            if iter >= BENCH_WARMUP { times_fused.push(ms); }

            // Correctness check on first real iteration
            if iter == BENCH_WARMUP {
                let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a_h, n) };
                let mis = check_correctness(&result, &expected);
                correct_fused = mis == 0;
                if mis == 0 {
                    println!("    Fused: correctness OK");
                } else {
                    println!("    !! Fused: {} / {} mismatched ({:.1}%)", mis, n,
                        mis as f64 / n as f64 * 100.0);
                }
            }
        }

        times_fused.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50_fused = percentile(&times_fused, 50.0);
        let p5_fused = percentile(&times_fused, 5.0);
        let p95_fused = percentile(&times_fused, 95.0);
        let mkeys_fused = n as f64 / p50_fused / 1e3;
        let speedup = mkeys_fused / full_mkeys;

        println!("    Fused partition: p50={:.3} ms  {:.0} Mkeys/s  {:.2}x vs current",
            p50_fused, mkeys_fused, speedup);
        println!("      [p5={:.3} p95={:.3}]  correct={}", p5_fused, p95_fused, correct_fused);

        if mkeys_fused >= 5000.0 {
            println!("    *** TARGET HIT: {:.0} Mkeys/s >= 5000 ***", mkeys_fused);
        }
    }
    } // end if false (Investigation H disabled)

    if false { // SKIP J-O (already ran, results captured)
    // ══════════════════════════════════════════════════════════════
    // Investigation J: Scatter BW at SLC-Resident Sizes
    // CRITICAL UNKNOWN: Does scatter BW improve at small (SLC-fit) sizes?
    // If scatter @ 250K ≈ 400 GB/s, hybrid MSD+LSD can reach target.
    // If scatter @ 250K ≈ 130 GB/s (same as 16M), need different approach.
    // ══════════════════════════════════════════════════════════════
    println!("\n  ── Investigation J: Scatter BW × Size × Bins (SLC Test) ──");
    println!("    Kernel: exp16_diag_scatter_binned (CPU pre-computed offsets)");
    println!("    GPU hypothesis: scatter BW is SIZE-dependent, not just bin-dependent");
    println!("    If SLC-resident scatter >> DRAM scatter, hybrid wins\n");

    let pso_scatter_j = ctx.make_pipeline("exp16_diag_scatter_binned");
    let tg_size_j = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };

    println!("    {:>10} {:>6} {:>10} {:>10} {:>12}",
        "Size", "Bins", "p50 (ms)", "GB/s", "SLC?");
    println!("    {}", "-".repeat(54));

    for &sz in &[62_500usize, 250_000, 1_000_000, 4_000_000, 16_000_000] {
        let data_j = gen_random_u32(sz);
        let buf_src_j = alloc_buffer_with_data(&ctx.device, &data_j);
        let buf_dst_j = alloc_buffer(&ctx.device, sz * 4);
        let sz_u32 = sz as u32;
        let grid_j = MTLSize { width: sz.div_ceil(THREADS_PER_TG), height: 1, depth: 1 };
        let data_mb = sz as f64 * 4.0 / 1024.0 / 1024.0;
        let slc_tag = if data_mb * 2.0 < 24.0 { "SLC" } else { "DRAM" };

        for &bins in &[64u32, 256, 1024] {
            let mask = bins - 1;
            // CPU pre-compute scatter offsets
            let offsets_j: Vec<u32> = {
                let mut hist = vec![0u32; bins as usize];
                for &v in &data_j { hist[((v >> 0) & mask) as usize] += 1; }
                let mut prefix = vec![0u32; bins as usize];
                let mut sum = 0u32;
                for i in 0..bins as usize { prefix[i] = sum; sum += hist[i]; }
                let mut counters = prefix.clone();
                let mut offs = vec![0u32; sz];
                for i in 0..sz {
                    let bin = ((data_j[i] >> 0) & mask) as usize;
                    offs[i] = counters[bin];
                    counters[bin] += 1;
                }
                offs
            };
            let buf_offs_j = alloc_buffer_with_data(&ctx.device, &offsets_j);

            let mut times_j = Vec::new();
            for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
                let cmd = ctx.command_buffer();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&pso_scatter_j);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_src_j.as_ref()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(buf_dst_j.as_ref()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_offs_j.as_ref()), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&sz_u32 as *const u32 as *mut _).unwrap(), 4, 3);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid_j, tg_size_j);
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if iter >= BENCH_WARMUP { times_j.push(gpu_elapsed_ms(&cmd)); }
            }
            times_j.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50_j = percentile(&times_j, 50.0);
            let bw_j = (sz * 4 * 2) as f64 / p50_j / 1e6;
            println!("    {:>10} {:>6} {:>10.4} {:>10.0} {:>12}",
                format_size(sz), bins, p50_j, bw_j, slc_tag);
        }
    }

    // ══════════════════════════════════════════════════════════════
    // Investigation K: TG Reorder Inner (Coalesced Scatter)
    // Replace inner_scan_scatter with inner_scan_scatter_reorder.
    // GPU hypothesis: TG reorder converts random scatter to sequential
    // writes, potentially 2-3x better scatter BW.
    // Uses Investigation I's best pattern (single-encoder, batch=64).
    // ══════════════════════════════════════════════════════════════
    println!("\n  ── Investigation K: TG Reorder Inner (Coalesced) ──");
    println!("    Kernel: exp17_inner_scan_scatter_reorder");
    println!("    GPU hypothesis: TG reorder buffer → sequential device writes");
    println!("    vs random scatter in inner_scan_scatter\n");

    {
        let bpb_k = 64usize; // Best batch size from Investigation I
        let num_batches_k = 256usize.div_ceil(bpb_k);
        let max_tpb_k = 17usize;
        let tgs_per_batch_k = bpb_k * max_tpb_k;
        let batch_data_mb = bpb_k as f64 * (n as f64 / 256.0) * 4.0 / 1024.0 / 1024.0;

        let pso_hist_k = ctx.make_pipeline("exp17_msd_histogram");
        let pso_bdesc_k = ctx.make_pipeline("exp17_compute_bucket_descs");
        let pso_prefix_k = ctx.make_pipeline("exp17_global_prefix");
        let pso_zero_st_k = ctx.make_pipeline("exp16_zero_status");
        let pso_part_k = ctx.make_pipeline("exp16_partition");
        let pso_inner_zero_k = ctx.make_pipeline("exp17_inner_zero");
        let pso_inner_hist_k = ctx.make_pipeline("exp17_inner_histogram");
        let pso_inner_reorder_k = ctx.make_pipeline("exp17_inner_scan_scatter_reorder");

        let num_tiles_k = n.div_ceil(TILE_SIZE);
        let tile_hists_entries_k: usize = 256 * max_tpb_k * 256;

        let data_k = gen_random_u32(n);
        let mut expected_k = data_k.clone();
        expected_k.sort();

        let buf_input_k = alloc_buffer_with_data(&ctx.device, &data_k);
        let buf_a_k = alloc_buffer(&ctx.device, n * 4);
        let buf_b_k = alloc_buffer(&ctx.device, n * 4);
        let buf_msd_hist_k = alloc_buffer(&ctx.device, 4 * 256 * 4);
        let buf_tile_status_k = alloc_buffer(&ctx.device, num_tiles_k * 256 * 4);
        let buf_counters_k = alloc_buffer(&ctx.device, 4);
        let buf_bucket_descs_k = alloc_buffer(&ctx.device, 256 * 16);
        let buf_tile_hists_k = alloc_buffer(&ctx.device, tile_hists_entries_k * 4);

        let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
        let one_tg_k = MTLSize { width: 1, height: 1, depth: 1 };
        let hist_grid_k = MTLSize { width: num_tiles_k, height: 1, depth: 1 };
        let zero_tg_count_k = (num_tiles_k * 256).div_ceil(THREADS_PER_TG);
        let zero_grid_k = MTLSize { width: zero_tg_count_k, height: 1, depth: 1 };
        let batch_grid_k = MTLSize { width: tgs_per_batch_k, height: 1, depth: 1 };

        let batch_tile_hists_k = (bpb_k * max_tpb_k * 256) as u32;
        let batch_zero_tgs_k = (batch_tile_hists_k as usize).div_ceil(THREADS_PER_TG);
        let batch_zero_grid_k = MTLSize { width: batch_zero_tgs_k, height: 1, depth: 1 };

        let exp17_p_k = Exp17Params {
            element_count: n as u32, num_tiles: num_tiles_k as u32, shift: 24, pass: 0,
        };
        let exp16_p_k = Exp16Params {
            element_count: n as u32, num_tiles: num_tiles_k as u32,
            num_tgs: num_tiles_k as u32, shift: 24, pass: 0,
        };
        let tile_size_u32_k = TILE_SIZE as u32;
        let inner_shifts_k: [u32; 3] = [0, 8, 16];

        let mut times_k = Vec::new();
        for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
            unsafe {
                let src = buf_input_k.contents().as_ptr() as *const u8;
                let dst = buf_a_k.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src, dst, n * 4);
                std::ptr::write_bytes(buf_msd_hist_k.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
            }

            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();

            // MSD phase (identical to Investigation I)
            enc.setComputePipelineState(&pso_hist_k);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a_k.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_k.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp17_p_k as *const Exp17Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp17Params>(), 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid_k, tg_size);

            enc.setComputePipelineState(&pso_bdesc_k);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_k.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_k.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&tile_size_u32_k as *const u32 as *mut _).unwrap(), 4, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_k, tg_size);

            enc.setComputePipelineState(&pso_prefix_k);
            unsafe { enc.setBuffer_offset_atIndex(Some(buf_msd_hist_k.as_ref()), 0, 0); }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_k, tg_size);

            enc.setComputePipelineState(&pso_zero_st_k);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_tile_status_k.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_counters_k.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp16_p_k as *const Exp16Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp16Params>(), 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(zero_grid_k, tg_size);

            enc.setComputePipelineState(&pso_part_k);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a_k.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_b_k.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_tile_status_k.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_counters_k.as_ref()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_k.as_ref()), 0, 4);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp16_p_k as *const Exp16Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp16Params>(), 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid_k, tg_size);

            // Inner sort: batch=64, single encoder, TG REORDER kernel
            for batch in 0..num_batches_k {
                let tg_off = (batch * bpb_k * max_tpb_k) as u32;
                let batch_offset_bytes = batch * bpb_k * max_tpb_k * 256 * 4;

                for (pass_idx, &shift) in inner_shifts_k.iter().enumerate() {
                    let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                        (buf_b_k.as_ref(), buf_a_k.as_ref())
                    } else {
                        (buf_a_k.as_ref(), buf_b_k.as_ref())
                    };

                    enc.setComputePipelineState(&pso_inner_zero_k);
                    unsafe {
                        enc.setBuffer_offset_atIndex(
                            Some(buf_tile_hists_k.as_ref()), batch_offset_bytes, 0);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&batch_tile_hists_k as *const u32 as *mut _).unwrap(), 4, 1);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(batch_zero_grid_k, tg_size);

                    let inner_params = Exp17InnerParams { shift, tg_offset: tg_off };
                    enc.setComputePipelineState(&pso_inner_hist_k);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(buf_tile_hists_k.as_ref()), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_k.as_ref()), 0, 2);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                            std::mem::size_of::<Exp17InnerParams>(), 3);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid_k, tg_size);

                    // KEY CHANGE: use reorder kernel instead of scan_scatter
                    enc.setComputePipelineState(&pso_inner_reorder_k);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(buf_tile_hists_k.as_ref()), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_k.as_ref()), 0, 3);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                            std::mem::size_of::<Exp17InnerParams>(), 4);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid_k, tg_size);
                }
            }

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let ms = gpu_elapsed_ms(&cmd);
            if iter >= BENCH_WARMUP { times_k.push(ms); }

            if iter == BENCH_WARMUP {
                let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a_k, n) };
                let mis = check_correctness(&result, &expected_k);
                if mis == 0 {
                    println!("    TG Reorder: correctness OK");
                } else {
                    println!("    !! TG Reorder: {} / {} mismatched ({:.1}%)",
                        mis, n, mis as f64 / n as f64 * 100.0);
                }
            }
        }
        times_k.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50_k = percentile(&times_k, 50.0);
        let mkeys_k = n as f64 / p50_k / 1e3;
        println!("    TG Reorder: p50={:.3} ms  {:.0} Mkeys/s  {:.2}x vs baseline ({:.1} MB/batch)",
            p50_k, mkeys_k, mkeys_k / BASELINE_MKEYS, batch_data_mb);
        if mkeys_k >= 5000.0 {
            println!("    *** TARGET HIT: {:.0} Mkeys/s >= 5000 ***", mkeys_k);
        }
    }

    // ══════════════════════════════════════════════════════════════
    // Investigation L: Fused 256-TG Inner (1 TG per bucket)
    // Uses exp17_inner_fused — each TG serially processes ALL tiles
    // of its bucket. No tile_hists, no inter-TG coordination.
    // GPU hypothesis: eliminates tile_hists BW entirely, trades for
    // serial processing within bucket. Good if per-bucket work is
    // compute-bound rather than memory-bound.
    // ══════════════════════════════════════════════════════════════
    println!("\n  ── Investigation L: Fused Inner (256 TGs, serial tiles) ──");
    println!("    Kernel: exp17_inner_fused (no tile_hists, 1 TG/bucket)");
    println!("    GPU hypothesis: eliminate tile_hists overhead via serial processing\n");

    {
        let pso_hist_l = ctx.make_pipeline("exp17_msd_histogram");
        let pso_bdesc_l = ctx.make_pipeline("exp17_compute_bucket_descs");
        let pso_prefix_l = ctx.make_pipeline("exp17_global_prefix");
        let pso_zero_st_l = ctx.make_pipeline("exp16_zero_status");
        let pso_part_l = ctx.make_pipeline("exp16_partition");
        let pso_fused_l = ctx.make_pipeline("exp17_inner_fused");

        let num_tiles_l = n.div_ceil(TILE_SIZE);

        let data_l = gen_random_u32(n);
        let mut expected_l = data_l.clone();
        expected_l.sort();

        let buf_input_l = alloc_buffer_with_data(&ctx.device, &data_l);
        let buf_a_l = alloc_buffer(&ctx.device, n * 4);
        let buf_b_l = alloc_buffer(&ctx.device, n * 4);
        let buf_msd_hist_l = alloc_buffer(&ctx.device, 4 * 256 * 4);
        let buf_tile_status_l = alloc_buffer(&ctx.device, num_tiles_l * 256 * 4);
        let buf_counters_l = alloc_buffer(&ctx.device, 4);
        let buf_bucket_descs_l = alloc_buffer(&ctx.device, 256 * 16);

        let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
        let one_tg_l = MTLSize { width: 1, height: 1, depth: 1 };
        let hist_grid_l = MTLSize { width: num_tiles_l, height: 1, depth: 1 };
        let zero_tg_count_l = (num_tiles_l * 256).div_ceil(THREADS_PER_TG);
        let zero_grid_l = MTLSize { width: zero_tg_count_l, height: 1, depth: 1 };
        let fused_grid_l = MTLSize { width: 256, height: 1, depth: 1 }; // 1 TG per bucket

        let exp17_p_l = Exp17Params {
            element_count: n as u32, num_tiles: num_tiles_l as u32, shift: 24, pass: 0,
        };
        let exp16_p_l = Exp16Params {
            element_count: n as u32, num_tiles: num_tiles_l as u32,
            num_tgs: num_tiles_l as u32, shift: 24, pass: 0,
        };
        let tile_size_u32_l = TILE_SIZE as u32;
        let inner_shifts_l: [u32; 3] = [0, 8, 16];

        let mut times_l = Vec::new();
        for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
            unsafe {
                let src = buf_input_l.contents().as_ptr() as *const u8;
                let dst = buf_a_l.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src, dst, n * 4);
                std::ptr::write_bytes(buf_msd_hist_l.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
            }

            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();

            // MSD phase
            enc.setComputePipelineState(&pso_hist_l);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a_l.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_l.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp17_p_l as *const Exp17Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp17Params>(), 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid_l, tg_size);

            enc.setComputePipelineState(&pso_bdesc_l);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_l.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_l.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&tile_size_u32_l as *const u32 as *mut _).unwrap(), 4, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_l, tg_size);

            enc.setComputePipelineState(&pso_prefix_l);
            unsafe { enc.setBuffer_offset_atIndex(Some(buf_msd_hist_l.as_ref()), 0, 0); }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_l, tg_size);

            enc.setComputePipelineState(&pso_zero_st_l);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_tile_status_l.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_counters_l.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp16_p_l as *const Exp16Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp16Params>(), 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(zero_grid_l, tg_size);

            enc.setComputePipelineState(&pso_part_l);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a_l.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_b_l.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_tile_status_l.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_counters_l.as_ref()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_l.as_ref()), 0, 4);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp16_p_l as *const Exp16Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp16Params>(), 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid_l, tg_size);

            // Inner sort: 3 passes, fused (256 TGs, 1 per bucket)
            for &shift in &inner_shifts_l {
                let (src_buf, dst_buf) = if shift == 0 || shift == 16 {
                    (buf_b_l.as_ref(), buf_a_l.as_ref())
                } else {
                    (buf_a_l.as_ref(), buf_b_l.as_ref())
                };

                let inner_params = Exp17InnerParams { shift, tg_offset: 0 };
                enc.setComputePipelineState(&pso_fused_l);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_l.as_ref()), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                        std::mem::size_of::<Exp17InnerParams>(), 3);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(fused_grid_l, tg_size);
            }

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let ms = gpu_elapsed_ms(&cmd);
            if iter >= BENCH_WARMUP { times_l.push(ms); }

            if iter == BENCH_WARMUP {
                let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a_l, n) };
                let mis = check_correctness(&result, &expected_l);
                if mis == 0 {
                    println!("    Fused 256-TG: correctness OK");
                } else {
                    println!("    !! Fused 256-TG: {} / {} mismatched ({:.1}%)",
                        mis, n, mis as f64 / n as f64 * 100.0);
                }
            }
        }
        times_l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50_l = percentile(&times_l, 50.0);
        let mkeys_l = n as f64 / p50_l / 1e3;
        println!("    Fused 256-TG: p50={:.3} ms  {:.0} Mkeys/s  {:.2}x vs baseline",
            p50_l, mkeys_l, mkeys_l / BASELINE_MKEYS);
        if mkeys_l >= 5000.0 {
            println!("    *** TARGET HIT: {:.0} Mkeys/s >= 5000 ***", mkeys_l);
        }
    }

    // ══════════════════════════════════════════════════════════════
    // Investigation M: Pure V4 4-Pass LSD (no MSD)
    // Control experiment: how fast is exp16_partition_v4 as a straight
    // 4-pass sort? This is the "what if we just made the scatter
    // kernel better" baseline.
    // GPU hypothesis: V4's TG reorder at full 16M gives the scatter
    // BW improvement that MSD+LSD tries to get via SLC residency.
    // ══════════════════════════════════════════════════════════════
    println!("\n  ── Investigation M: Pure V4 4-Pass LSD (no MSD) ──");
    println!("    Kernel: exp16_partition_v4 (TG reorder, 4 passes at 16M)");
    println!("    GPU hypothesis: V4 TG reorder may match SLC advantage\n");

    {
        let pso_8bit_hist_m = ctx.make_pipeline("exp16_combined_histogram");
        let pso_8bit_prefix_m = ctx.make_pipeline("exp16_global_prefix");
        let pso_8bit_zero_m = ctx.make_pipeline("exp16_zero_status");
        let pso_v4_part_m = ctx.make_pipeline("exp16_partition_v4");

        let num_tiles_m = n.div_ceil(TILE_SIZE);
        // V4 uses 2048-element tiles (TG reorder buffer = 2048 uints = 8KB)
        let v4_tile_size = 2048usize;
        let v4_num_tiles = n.div_ceil(v4_tile_size);

        let data_m = gen_random_u32(n);
        let mut expected_m = data_m.clone();
        expected_m.sort();

        let buf_input_m = alloc_buffer_with_data(&ctx.device, &data_m);
        let buf_a_m = alloc_buffer(&ctx.device, n * 4);
        let buf_b_m = alloc_buffer(&ctx.device, n * 4);
        let buf_gh_m = alloc_buffer(&ctx.device, 4 * 256 * 4);
        let buf_ts_m = alloc_buffer(&ctx.device, v4_num_tiles * 256 * 4);
        let buf_ctr_m = alloc_buffer(&ctx.device, 4);

        let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
        let one_tg_m = MTLSize { width: 1, height: 1, depth: 1 };
        let v4_zero_tg_count = (v4_num_tiles * 256).div_ceil(THREADS_PER_TG);
        let params_size_m = std::mem::size_of::<Exp16Params>();

        let mut times_m = Vec::new();
        for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
            unsafe {
                let src = buf_input_m.contents().as_ptr() as *const u8;
                let dst = buf_a_m.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src, dst, n * 4);
                std::ptr::write_bytes(buf_gh_m.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
            }

            let cmd = ctx.command_buffer();

            // Histogram (uses original 4096-element tiles)
            let p0 = Exp16Params { element_count: n as u32, num_tiles: num_tiles_m as u32,
                num_tgs: num_tiles_m as u32, shift: 0, pass: 0 };
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_8bit_hist_m);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a_m.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_gh_m.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&p0 as *const Exp16Params as *mut _).unwrap(), params_size_m, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tiles_m, height: 1, depth: 1 }, tg_size);
            enc.endEncoding();

            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_8bit_prefix_m);
            unsafe { enc.setBuffer_offset_atIndex(Some(buf_gh_m.as_ref()), 0, 0); }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_m, tg_size);
            enc.endEncoding();

            // 4 passes of V4
            for pass in 0..4u32 {
                let p = Exp16Params { element_count: n as u32, num_tiles: v4_num_tiles as u32,
                    num_tgs: v4_num_tiles as u32, shift: pass * 8, pass };
                let (s, d) = if pass % 2 == 0 { (buf_a_m.as_ref(), buf_b_m.as_ref()) }
                             else { (buf_b_m.as_ref(), buf_a_m.as_ref()) };
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&pso_8bit_zero_m);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_ts_m.as_ref()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(buf_ctr_m.as_ref()), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&p as *const Exp16Params as *mut _).unwrap(), params_size_m, 2);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: v4_zero_tg_count, height: 1, depth: 1 }, tg_size);
                enc.endEncoding();

                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&pso_v4_part_m);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(s), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(d), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_ts_m.as_ref()), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(buf_ctr_m.as_ref()), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(buf_gh_m.as_ref()), 0, 4);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&p as *const Exp16Params as *mut _).unwrap(), params_size_m, 5);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: v4_num_tiles, height: 1, depth: 1 }, tg_size);
                enc.endEncoding();
            }

            cmd.commit();
            cmd.waitUntilCompleted();
            if iter >= BENCH_WARMUP { times_m.push(gpu_elapsed_ms(&cmd)); }

            if iter == BENCH_WARMUP {
                // 4 passes (even): result in buf_a
                let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a_m, n) };
                let mis = check_correctness(&result, &expected_m);
                if mis == 0 {
                    println!("    V4 4-pass: correctness OK");
                } else {
                    println!("    !! V4 4-pass: {} / {} mismatched ({:.1}%)",
                        mis, n, mis as f64 / n as f64 * 100.0);
                }
            }
        }
        times_m.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50_m = percentile(&times_m, 50.0);
        let mkeys_m = n as f64 / p50_m / 1e3;
        println!("    V4 4-pass: p50={:.3} ms  {:.0} Mkeys/s  {:.2}x vs baseline",
            p50_m, mkeys_m, mkeys_m / BASELINE_MKEYS);
        if mkeys_m >= 5000.0 {
            println!("    *** TARGET HIT: {:.0} Mkeys/s >= 5000 ***", mkeys_m);
        }
    }

    // ══════════════════════════════════════════════════════════════
    // Investigation N: 6-Bit Inner Radix (64 bins, 4 inner passes)
    // GPU hypothesis: 64-bin scatter has better coalescing than 256-bin.
    // Trade: 4 inner passes (6+6+6+6=24 bits) vs 3 inner passes (8+8+8=24).
    // If 64-bin scatter BW > 256-bin scatter BW × 4/3, 6-bit wins.
    // ══════════════════════════════════════════════════════════════
    println!("\n  ── Investigation N: 6-Bit Inner (64 bins, 4 passes) ──");
    println!("    Kernels: exp17_inner_histogram_6bit + exp17_inner_scan_scatter_6bit");
    println!("    GPU hypothesis: 64-bin scatter coalesces better than 256-bin");
    println!("    Trade: 4 passes × better BW vs 3 passes × worse BW\n");

    {
        let bpb_n = 64usize;
        let num_batches_n = 256usize.div_ceil(bpb_n);
        let max_tpb_n = 17usize;
        let tgs_per_batch_n = bpb_n * max_tpb_n;

        let pso_hist_n = ctx.make_pipeline("exp17_msd_histogram");
        let pso_bdesc_n = ctx.make_pipeline("exp17_compute_bucket_descs");
        let pso_prefix_n = ctx.make_pipeline("exp17_global_prefix");
        let pso_zero_st_n = ctx.make_pipeline("exp16_zero_status");
        let pso_part_n = ctx.make_pipeline("exp16_partition");
        let pso_inner_zero_n = ctx.make_pipeline("exp17_inner_zero");
        let pso_inner_hist_6_n = ctx.make_pipeline("exp17_inner_histogram_6bit");
        let pso_inner_scat_6_n = ctx.make_pipeline("exp17_inner_scan_scatter_6bit");

        let num_tiles_n = n.div_ceil(TILE_SIZE);
        // 6-bit: 64 bins instead of 256 → tile_hists is 64 × max_tpb × 256 buckets
        let tile_hists_6_entries: usize = 64 * max_tpb_n * 256;

        let data_n = gen_random_u32(n);
        let mut expected_n = data_n.clone();
        expected_n.sort();

        let buf_input_n = alloc_buffer_with_data(&ctx.device, &data_n);
        let buf_a_n = alloc_buffer(&ctx.device, n * 4);
        let buf_b_n = alloc_buffer(&ctx.device, n * 4);
        let buf_msd_hist_n = alloc_buffer(&ctx.device, 4 * 256 * 4);
        let buf_tile_status_n = alloc_buffer(&ctx.device, num_tiles_n * 256 * 4);
        let buf_counters_n = alloc_buffer(&ctx.device, 4);
        let buf_bucket_descs_n = alloc_buffer(&ctx.device, 256 * 16);
        let buf_tile_hists_n = alloc_buffer(&ctx.device, tile_hists_6_entries * 4);

        let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
        let one_tg_n = MTLSize { width: 1, height: 1, depth: 1 };
        let hist_grid_n = MTLSize { width: num_tiles_n, height: 1, depth: 1 };
        let zero_tg_count_n = (num_tiles_n * 256).div_ceil(THREADS_PER_TG);
        let zero_grid_n = MTLSize { width: zero_tg_count_n, height: 1, depth: 1 };
        let batch_grid_n = MTLSize { width: tgs_per_batch_n, height: 1, depth: 1 };

        let batch_tile_hists_6 = (bpb_n * max_tpb_n * 64) as u32;
        let batch_zero_6_tgs = (batch_tile_hists_6 as usize).div_ceil(THREADS_PER_TG);
        let batch_zero_6_grid = MTLSize { width: batch_zero_6_tgs, height: 1, depth: 1 };

        let exp17_p_n = Exp17Params {
            element_count: n as u32, num_tiles: num_tiles_n as u32, shift: 24, pass: 0,
        };
        let exp16_p_n = Exp16Params {
            element_count: n as u32, num_tiles: num_tiles_n as u32,
            num_tgs: num_tiles_n as u32, shift: 24, pass: 0,
        };
        let tile_size_u32_n = TILE_SIZE as u32;
        // 4 passes of 6 bits: 0, 6, 12, 18 (covers bits 0-23, MSD covers 24-31)
        let inner_shifts_6: [u32; 4] = [0, 6, 12, 18];

        let mut times_n = Vec::new();
        for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
            unsafe {
                let src = buf_input_n.contents().as_ptr() as *const u8;
                let dst = buf_a_n.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src, dst, n * 4);
                std::ptr::write_bytes(buf_msd_hist_n.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
            }

            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();

            // MSD phase
            enc.setComputePipelineState(&pso_hist_n);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a_n.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_n.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp17_p_n as *const Exp17Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp17Params>(), 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid_n, tg_size);

            enc.setComputePipelineState(&pso_bdesc_n);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_n.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_n.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&tile_size_u32_n as *const u32 as *mut _).unwrap(), 4, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_n, tg_size);

            enc.setComputePipelineState(&pso_prefix_n);
            unsafe { enc.setBuffer_offset_atIndex(Some(buf_msd_hist_n.as_ref()), 0, 0); }
            enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_n, tg_size);

            enc.setComputePipelineState(&pso_zero_st_n);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_tile_status_n.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_counters_n.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp16_p_n as *const Exp16Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp16Params>(), 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(zero_grid_n, tg_size);

            enc.setComputePipelineState(&pso_part_n);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a_n.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_b_n.as_ref()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_tile_status_n.as_ref()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_counters_n.as_ref()), 0, 3);
                enc.setBuffer_offset_atIndex(Some(buf_msd_hist_n.as_ref()), 0, 4);
                enc.setBytes_length_atIndex(
                    NonNull::new(&exp16_p_n as *const Exp16Params as *mut _).unwrap(),
                    std::mem::size_of::<Exp16Params>(), 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid_n, tg_size);

            // Inner sort: 4 passes of 6-bit, batch=64
            for batch in 0..num_batches_n {
                let tg_off = (batch * bpb_n * max_tpb_n) as u32;
                let batch_offset_bytes = batch * bpb_n * max_tpb_n * 64 * 4; // 64 bins

                for (pass_idx, &shift) in inner_shifts_6.iter().enumerate() {
                    let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
                        (buf_b_n.as_ref(), buf_a_n.as_ref())
                    } else {
                        (buf_a_n.as_ref(), buf_b_n.as_ref())
                    };

                    enc.setComputePipelineState(&pso_inner_zero_n);
                    unsafe {
                        enc.setBuffer_offset_atIndex(
                            Some(buf_tile_hists_n.as_ref()), batch_offset_bytes, 0);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&batch_tile_hists_6 as *const u32 as *mut _).unwrap(), 4, 1);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(batch_zero_6_grid, tg_size);

                    let inner_params = Exp17InnerParams { shift, tg_offset: tg_off };
                    enc.setComputePipelineState(&pso_inner_hist_6_n);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(buf_tile_hists_n.as_ref()), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_n.as_ref()), 0, 2);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                            std::mem::size_of::<Exp17InnerParams>(), 3);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid_n, tg_size);

                    enc.setComputePipelineState(&pso_inner_scat_6_n);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(buf_tile_hists_n.as_ref()), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs_n.as_ref()), 0, 3);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&inner_params as *const Exp17InnerParams as *mut _).unwrap(),
                            std::mem::size_of::<Exp17InnerParams>(), 4);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(batch_grid_n, tg_size);
                }
            }

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let ms = gpu_elapsed_ms(&cmd);
            if iter >= BENCH_WARMUP { times_n.push(ms); }

            if iter == BENCH_WARMUP {
                // 4 even inner passes: result in buf_a (pass 0→a, 1→b, 2→a, 3→b)
                // MSD: a→b (1 pass). Inner: 4 passes from b. pass0: b→a, pass1: a→b, pass2: b→a, pass3: a→b
                // Final result in buf_b
                let result: Vec<u32> = unsafe { read_buffer_slice(&buf_b_n, n) };
                let mis = check_correctness(&result, &expected_n);
                if mis == 0 {
                    println!("    6-bit inner: correctness OK");
                } else {
                    println!("    !! 6-bit inner: {} / {} mismatched ({:.1}%)",
                        mis, n, mis as f64 / n as f64 * 100.0);
                    // Also check buf_a in case parity is wrong
                    let result_a: Vec<u32> = unsafe { read_buffer_slice(&buf_a_n, n) };
                    let mis_a = check_correctness(&result_a, &expected_n);
                    if mis_a == 0 {
                        println!("    6-bit inner: correctness OK (in buf_a, parity=even)");
                    } else {
                        println!("    !! 6-bit inner: buf_a also {} mismatched", mis_a);
                    }
                }
            }
        }
        times_n.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50_n = percentile(&times_n, 50.0);
        let mkeys_n = n as f64 / p50_n / 1e3;
        println!("    6-bit inner: p50={:.3} ms  {:.0} Mkeys/s  {:.2}x vs baseline",
            p50_n, mkeys_n, mkeys_n / BASELINE_MKEYS);
        if mkeys_n >= 5000.0 {
            println!("    *** TARGET HIT: {:.0} Mkeys/s >= 5000 ***", mkeys_n);
        }
    }

    // ══════════════════════════════════════════════════════════════
    // Investigation O: TG Bitonic Sort Throughput
    // Measures pure in-TG-memory sort speed at various sub-bucket sizes.
    // GPU hypothesis: if TG sort at ~4K elements is fast enough,
    // MSD → 1 inner LSD → bitonic could beat MSD → 3 inner LSD.
    // Saves 2 scatter passes (256 MB), replaces with TG compute.
    // ══════════════════════════════════════════════════════════════
    println!("\n  ── Investigation O: TG Bitonic Sort Throughput ──");
    println!("    Kernel: exp17_tg_bitonic_sort");
    println!("    GPU hypothesis: TG-memory-only sort faster than 2 scatter passes");
    println!("    Reference: 2 inner passes ≈ 2.0 ms at 125 GB/s\n");

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Exp17TgSortParams {
        bucket_count: u32,
        max_sub_size: u32,
    }

    {
        let pso_bitonic = ctx.make_pipeline("exp17_tg_bitonic_sort");
        let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };

        // Test at different sub-bucket sizes (simulated from MSD + inner LSD)
        // After MSD (256 buckets of ~62.5K) + 1 inner LSD pass (256 bins),
        // each sub-bucket has ~244 elements (62500/256).
        // Also test larger sizes for MSD-only sub-buckets.
        for &sub_size in &[244usize, 512, 1024, 2048, 4096] {
            let num_subs = n / sub_size.max(1);
            if num_subs == 0 { continue; }
            let total_elements = num_subs * sub_size;

            // Generate data: num_subs sub-arrays of sub_size elements
            let data_o = gen_random_u32(total_elements);
            // CPU sort each sub-bucket for expected result
            let mut expected_o = data_o.clone();
            for chunk in expected_o.chunks_mut(sub_size) {
                chunk.sort();
            }

            let buf_data_o = alloc_buffer_with_data(&ctx.device, &data_o);
            // Offsets: 0, sub_size, 2*sub_size, ...
            let offsets_o: Vec<u32> = (0..num_subs).map(|i| (i * sub_size) as u32).collect();
            let counts_o: Vec<u32> = vec![sub_size as u32; num_subs];
            let buf_offsets_o = alloc_buffer_with_data(&ctx.device, &offsets_o);
            let buf_counts_o = alloc_buffer_with_data(&ctx.device, &counts_o);

            let params_o = Exp17TgSortParams {
                bucket_count: num_subs as u32,
                max_sub_size: 4096,
            };

            let grid_o = MTLSize { width: num_subs, height: 1, depth: 1 };

            let mut times_o = Vec::new();
            for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
                // Reset data each iteration
                unsafe {
                    let src = data_o.as_ptr() as *const u8;
                    let dst = buf_data_o.contents().as_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(src, dst, total_elements * 4);
                }

                let cmd = ctx.command_buffer();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&pso_bitonic);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_data_o.as_ref()), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(buf_offsets_o.as_ref()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_counts_o.as_ref()), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&params_o as *const Exp17TgSortParams as *mut _).unwrap(),
                        std::mem::size_of::<Exp17TgSortParams>(), 3);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid_o, tg_size);
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();

                let ms = gpu_elapsed_ms(&cmd);
                if iter >= BENCH_WARMUP { times_o.push(ms); }

                if iter == BENCH_WARMUP {
                    let result: Vec<u32> = unsafe { read_buffer_slice(&buf_data_o, total_elements) };
                    let mis = check_correctness(&result, &expected_o);
                    if mis == 0 {
                        println!("    sub={}×{}: correctness OK", sub_size, num_subs);
                    } else {
                        println!("    !! sub={}×{}: {} mismatched",
                            sub_size, num_subs, mis);
                    }
                }
            }
            times_o.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50_o = percentile(&times_o, 50.0);
            let total_mb = total_elements as f64 * 4.0 / 1024.0 / 1024.0;
            let equiv_bw = total_mb * 2.0 / p50_o * 1000.0; // read + write
            let mkeys_equiv = total_elements as f64 / p50_o / 1e3;
            println!("    sub={:>4}×{:>5}: p50={:.3} ms  equiv {:.0} GB/s  {:.0} Mkeys/s",
                sub_size, num_subs, p50_o, equiv_bw, mkeys_equiv);
        }

        // Summary: how does bitonic sort time compare to 2 inner LSD passes?
        println!("\n    Comparison point:");
        println!("    2 inner passes @ 125 GB/s = {:.2} ms",
            2.0 * (n * 4 * 2) as f64 / 125e3);
        println!("    If bitonic < this → MSD+1inner+bitonic beats MSD+3inner");
    }
    } // end if false (J-O skip)

    // ══════════════════════════════════════════════════════════════
    // Investigation P: 3-Pass Scatter Ceiling Validation
    // THE decisive experiment. Pre-compute scatter offsets on CPU,
    // then benchmark 3 scatter dispatches at various radix widths.
    // This is the ABSOLUTE CEILING for any 3-pass sort implementation.
    // If 3-pass ceiling > 5000 Mkeys/s, path to target is proven.
    // ══════════════════════════════════════════════════════════════
    println!("\n  ── Investigation P: 3-Pass Scatter Ceiling ──");
    println!("    Kernel: exp16_diag_scatter_binned (CPU pre-computed offsets)");
    println!("    GPU hypothesis: 3 passes at ~130 GB/s → 5000+ Mkeys/s ceiling");
    println!("    This is the ABSOLUTE UPPER BOUND for any 3-pass sort.\n");

    {
        let pso_scat_p = ctx.make_pipeline("exp16_diag_scatter_binned");
        let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
        let grid_p = MTLSize { width: n.div_ceil(THREADS_PER_TG), height: 1, depth: 1 };
        let n_u32 = n as u32;

        // Test multiple radix configs for 3 passes
        let configs: &[(u32, u32, u32, &str)] = &[
            (8, 8, 8, "8+8+8 (256/256/256) — only 24 bits, NOT full sort"),
            (11, 11, 10, "11+11+10 (2048/2048/1024) — full 32-bit sort"),
            (10, 11, 11, "10+11+11 (1024/2048/2048) — full 32-bit sort"),
            (8, 12, 12, "8+12+12 (256/4096/4096) — full 32-bit sort"),
        ];

        // Also benchmark 4-pass for comparison
        println!("    {:>20} {:>10} {:>10} {:>10}", "Config", "p50 (ms)", "Mkeys/s", "vs 5000");
        println!("    {}", "-".repeat(56));

        for &(b0, b1, b2, label) in configs {
            let total_bits = b0 + b1 + b2;
            let data_p = gen_random_u32(n);
            let buf_a_p = alloc_buffer_with_data(&ctx.device, &data_p);
            let buf_b_p = alloc_buffer(&ctx.device, n * 4);

            // Pre-compute offsets for 3-pass stable sort
            let shifts = [0u32, b0, b0 + b1];
            let widths = [b0, b1, b2];
            let mut current = data_p.clone();
            let mut offset_bufs_p = Vec::new();

            for pass in 0..3usize {
                let shift = shifts[pass];
                let bits = widths[pass];
                let num_bins = 1u32 << bits;
                let mask = num_bins - 1;

                let mut hist = vec![0u32; num_bins as usize];
                for &v in &current { hist[((v >> shift) & mask) as usize] += 1; }
                let mut prefix = vec![0u32; num_bins as usize];
                let mut sum = 0u32;
                for i in 0..num_bins as usize { prefix[i] = sum; sum += hist[i]; }
                let mut counters = prefix.clone();
                let mut offsets = vec![0u32; n];
                let mut next = vec![0u32; n];
                for i in 0..n {
                    let bin = ((current[i] >> shift) & mask) as usize;
                    offsets[i] = counters[bin];
                    next[counters[bin] as usize] = current[i];
                    counters[bin] += 1;
                }
                offset_bufs_p.push(alloc_buffer_with_data(&ctx.device, &offsets));
                current = next;
            }

            // Verify correctness (if 32 bits covered)
            if total_bits >= 32 {
                let mut expected_p = data_p.clone();
                expected_p.sort();
                let mis = check_correctness(&current, &expected_p);
                if mis > 0 {
                    println!("    !! CPU pre-sort: {} mismatches for {}", mis, label);
                }
            }

            // Benchmark: 3 scatter passes in single encoder
            let mut times_p = Vec::new();
            for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
                unsafe {
                    let src = data_p.as_ptr() as *const u8;
                    let dst = buf_a_p.contents().as_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(src, dst, n * 4);
                }

                let cmd = ctx.command_buffer();
                let enc = cmd.computeCommandEncoder().unwrap();
                for pass in 0..3usize {
                    let (s, d) = if pass % 2 == 0 {
                        (buf_a_p.as_ref(), buf_b_p.as_ref())
                    } else {
                        (buf_b_p.as_ref(), buf_a_p.as_ref())
                    };
                    enc.setComputePipelineState(&pso_scat_p);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(s), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(d), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(offset_bufs_p[pass].as_ref()), 0, 2);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(), 4, 3);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_p, tg_size);
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if iter >= BENCH_WARMUP { times_p.push(gpu_elapsed_ms(&cmd)); }
            }
            times_p.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50_p = percentile(&times_p, 50.0);
            let mkeys_p = n as f64 / p50_p / 1e3;
            let per_pass = p50_p / 3.0;
            let bw_per_pass = (n * 4 * 2) as f64 / per_pass / 1e6;
            let vs5000 = if mkeys_p >= 5000.0 { format!("✓ {:.0}%", mkeys_p / 50.0) }
                         else { format!("✗ {:.0}%", mkeys_p / 50.0) };
            println!("    {:>20}: {:>10.3} {:>10.0} {:>10}",
                format!("{}b", total_bits), p50_p, mkeys_p, vs5000);
            println!("      {} — per-pass: {:.3} ms @ {:.0} GB/s",
                label, per_pass, bw_per_pass);
        }

        // 4-pass control (current approach)
        {
            let data_p4 = gen_random_u32(n);
            let buf_a_p4 = alloc_buffer_with_data(&ctx.device, &data_p4);
            let buf_b_p4 = alloc_buffer(&ctx.device, n * 4);
            let mut current = data_p4.clone();
            let mut offset_bufs_4 = Vec::new();
            for pass in 0..4u32 {
                let shift = pass * 8;
                let mask = 0xFFu32;
                let mut hist = vec![0u32; 256];
                for &v in &current { hist[((v >> shift) & mask) as usize] += 1; }
                let mut prefix = vec![0u32; 256];
                let mut sum = 0u32;
                for i in 0..256 { prefix[i] = sum; sum += hist[i]; }
                let mut counters = prefix.clone();
                let mut offsets = vec![0u32; n];
                let mut next = vec![0u32; n];
                for i in 0..n {
                    let bin = ((current[i] >> shift) & mask) as usize;
                    offsets[i] = counters[bin];
                    next[counters[bin] as usize] = current[i];
                    counters[bin] += 1;
                }
                offset_bufs_4.push(alloc_buffer_with_data(&ctx.device, &offsets));
                current = next;
            }
            let mut times_4 = Vec::new();
            for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
                unsafe {
                    let src = data_p4.as_ptr() as *const u8;
                    let dst = buf_a_p4.contents().as_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(src, dst, n * 4);
                }
                let cmd = ctx.command_buffer();
                let enc = cmd.computeCommandEncoder().unwrap();
                for pass in 0..4usize {
                    let (s, d) = if pass % 2 == 0 {
                        (buf_a_p4.as_ref(), buf_b_p4.as_ref())
                    } else {
                        (buf_b_p4.as_ref(), buf_a_p4.as_ref())
                    };
                    enc.setComputePipelineState(&pso_scat_p);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(s), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(d), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(offset_bufs_4[pass].as_ref()), 0, 2);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(), 4, 3);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_p, tg_size);
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if iter >= BENCH_WARMUP { times_4.push(gpu_elapsed_ms(&cmd)); }
            }
            times_4.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50_4 = percentile(&times_4, 50.0);
            let mkeys_4 = n as f64 / p50_4 / 1e3;
            let per_pass_4 = p50_4 / 4.0;
            let bw_4 = (n * 4 * 2) as f64 / per_pass_4 / 1e6;
            println!("    {:>20}: {:>10.3} {:>10.0} {:>10}",
                "32b (4-pass ctrl)", p50_4, mkeys_4,
                if mkeys_4 >= 5000.0 { format!("✓ {:.0}%", mkeys_4 / 50.0) }
                else { format!("✗ {:.0}%", mkeys_4 / 50.0) });
            println!("      4×8-bit (256 bins) control — per-pass: {:.3} ms @ {:.0} GB/s",
                per_pass_4, bw_4);
        }

        // ── PART 2: Exhaustive 3-pass config sweep ──
        println!("\n    ── 3-Pass Config Sweep (all viable 32-bit splits) ──");
        println!("    {:>16} {:>10} {:>10} {:>10}", "Config", "p50 (ms)", "Mkeys/s", "per-pass BW");
        println!("    {}", "-".repeat(52));

        let three_pass_configs: &[(u32, u32, u32)] = &[
            // Narrowest first pass (pass 0 is most random → fewer bins = better coalescing)
            (8, 12, 12),
            (9, 12, 11),
            (9, 11, 12),
            (10, 11, 11),
            (10, 12, 10),
            (10, 10, 12),
            (11, 11, 10),
            (11, 10, 11),
            (12, 10, 10),
            (12, 11, 9),
        ];

        let mut best_3p_mkeys = 0.0f64;
        let mut best_3p_config = String::new();

        for &(b0, b1, b2) in three_pass_configs {
            if b0 + b1 + b2 != 32 { continue; }
            let data_s = gen_random_u32(n);
            let buf_a_s = alloc_buffer_with_data(&ctx.device, &data_s);
            let buf_b_s = alloc_buffer(&ctx.device, n * 4);

            let shifts = [0u32, b0, b0 + b1];
            let widths = [b0, b1, b2];
            let mut current = data_s.clone();
            let mut off_bufs = Vec::new();
            for pass in 0..3usize {
                let num_bins = 1u32 << widths[pass];
                let mask = num_bins - 1;
                let shift = shifts[pass];
                let mut hist = vec![0u32; num_bins as usize];
                for &v in &current { hist[((v >> shift) & mask) as usize] += 1; }
                let mut prefix = vec![0u32; num_bins as usize];
                let mut sum = 0u32;
                for i in 0..num_bins as usize { prefix[i] = sum; sum += hist[i]; }
                let mut counters = prefix.clone();
                let mut offs = vec![0u32; n];
                let mut next = vec![0u32; n];
                for i in 0..n {
                    let bin = ((current[i] >> shift) & mask) as usize;
                    offs[i] = counters[bin];
                    next[counters[bin] as usize] = current[i];
                    counters[bin] += 1;
                }
                off_bufs.push(alloc_buffer_with_data(&ctx.device, &offs));
                current = next;
            }

            let mut times_s = Vec::new();
            for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
                unsafe {
                    let src = data_s.as_ptr() as *const u8;
                    let dst = buf_a_s.contents().as_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(src, dst, n * 4);
                }
                let cmd = ctx.command_buffer();
                let enc = cmd.computeCommandEncoder().unwrap();
                for pass in 0..3usize {
                    let (s, d) = if pass % 2 == 0 { (buf_a_s.as_ref(), buf_b_s.as_ref()) }
                                 else { (buf_b_s.as_ref(), buf_a_s.as_ref()) };
                    enc.setComputePipelineState(&pso_scat_p);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(s), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(d), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(off_bufs[pass].as_ref()), 0, 2);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(), 4, 3);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_p, tg_size);
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if iter >= BENCH_WARMUP { times_s.push(gpu_elapsed_ms(&cmd)); }
            }
            times_s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50_s = percentile(&times_s, 50.0);
            let mkeys_s = n as f64 / p50_s / 1e3;
            let bw_s = (n * 4 * 2) as f64 / (p50_s / 3.0) / 1e6;
            let label = format!("{}+{}+{}", b0, b1, b2);
            let marker = if mkeys_s >= 5000.0 { " ★" } else { "" };
            println!("    {:>16}: {:>10.3} {:>10.0} {:>8.0} GB/s{}",
                label, p50_s, mkeys_s, bw_s, marker);
            if mkeys_s > best_3p_mkeys {
                best_3p_mkeys = mkeys_s;
                best_3p_config = label;
            }
        }
        println!("\n    Best 3-pass: {} → {:.0} Mkeys/s", best_3p_config, best_3p_mkeys);

        // ── PART 3: 2-Pass ceiling (the moonshot) ──
        // 2 passes at 128 GB/s = 2.0 ms → 8000 Mkeys/s theoretical
        // But 16+16 = 65536 bins. Does scatter hold up?
        println!("\n    ── 2-Pass Ceiling (Moonshot) ──");
        println!("    {:>16} {:>10} {:>10} {:>10}", "Config", "p50 (ms)", "Mkeys/s", "per-pass BW");
        println!("    {}", "-".repeat(52));

        let two_pass_configs: &[(u32, u32)] = &[
            (16, 16),
            (14, 18),
            (12, 20),
            (13, 19),
            (15, 17),
        ];

        for &(b0, b1) in two_pass_configs {
            if b0 + b1 != 32 { continue; }
            // Skip if bins > 2^20 (1M) — too much CPU pre-compute time
            if b0 > 20 || b1 > 20 { continue; }
            let data_2 = gen_random_u32(n);
            let buf_a_2 = alloc_buffer_with_data(&ctx.device, &data_2);
            let buf_b_2 = alloc_buffer(&ctx.device, n * 4);

            let shifts = [0u32, b0];
            let widths = [b0, b1];
            let mut current = data_2.clone();
            let mut off_bufs_2 = Vec::new();
            for pass in 0..2usize {
                let num_bins = 1u32 << widths[pass];
                let mask = num_bins - 1;
                let shift = shifts[pass];
                let mut hist = vec![0u32; num_bins as usize];
                for &v in &current { hist[((v >> shift) & mask) as usize] += 1; }
                let mut prefix = vec![0u32; num_bins as usize];
                let mut sum = 0u32;
                for i in 0..num_bins as usize { prefix[i] = sum; sum += hist[i]; }
                let mut counters = prefix.clone();
                let mut offs = vec![0u32; n];
                let mut next = vec![0u32; n];
                for i in 0..n {
                    let bin = ((current[i] >> shift) & mask) as usize;
                    offs[i] = counters[bin];
                    next[counters[bin] as usize] = current[i];
                    counters[bin] += 1;
                }
                off_bufs_2.push(alloc_buffer_with_data(&ctx.device, &offs));
                current = next;
            }

            let mut times_2 = Vec::new();
            for iter in 0..(BENCH_WARMUP + BENCH_RUNS) {
                unsafe {
                    let src = data_2.as_ptr() as *const u8;
                    let dst = buf_a_2.contents().as_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(src, dst, n * 4);
                }
                let cmd = ctx.command_buffer();
                let enc = cmd.computeCommandEncoder().unwrap();
                for pass in 0..2usize {
                    let (s, d) = if pass % 2 == 0 { (buf_a_2.as_ref(), buf_b_2.as_ref()) }
                                 else { (buf_b_2.as_ref(), buf_a_2.as_ref()) };
                    enc.setComputePipelineState(&pso_scat_p);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(s), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(d), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(off_bufs_2[pass].as_ref()), 0, 2);
                        enc.setBytes_length_atIndex(
                            NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(), 4, 3);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_p, tg_size);
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                if iter >= BENCH_WARMUP { times_2.push(gpu_elapsed_ms(&cmd)); }
            }
            times_2.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50_2 = percentile(&times_2, 50.0);
            let mkeys_2 = n as f64 / p50_2 / 1e3;
            let bw_2 = (n * 4 * 2) as f64 / (p50_2 / 2.0) / 1e6;
            let label = format!("{}+{}", b0, b1);
            let marker = if mkeys_2 >= 5000.0 { " ★" } else { "" };
            println!("    {:>16}: {:>10.3} {:>10.0} {:>8.0} GB/s{}",
                label, p50_2, mkeys_2, bw_2, marker);
        }

        println!("\n    VALIDATION:");
        println!("    If 3-pass 32b ceiling > 5000 → path exists (build efficient 3-pass kernel)");
        println!("    If 2-pass 32b ceiling > 5000 → even faster possible (but harder kernel)");
    }

    println!("\n{}", "=".repeat(60));
    println!("Investigations J-P Complete");
    println!("{}", "=".repeat(60));
}
