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

    // Investigation F: Phase timing breakdown of hybrid
    let n = 16_000_000usize;
    println!("  ── Investigation F: Phase Timing @ {}M ──", n / 1_000_000);
    let (phase_p50s, full_p50) = bench_phases(ctx, n);
    let full_mkeys = n as f64 / full_p50 / 1e3;
    println!("    Phase breakdown (p50, ms):");
    let labels = ["MSD histogram", "MSD scatter  ", "Inner pass 0 ",
                   "Inner pass 1 ", "Inner pass 2 "];
    for (i, label) in labels.iter().enumerate() {
        let pct = phase_p50s[i] / full_p50 * 100.0;
        println!("      {}: {:.3} ms ({:.1}%)", label, phase_p50s[i], pct);
    }
    let inner_total = phase_p50s[2] + phase_p50s[3] + phase_p50s[4];
    let msd_total = phase_p50s[0] + phase_p50s[1];
    println!("    MSD total: {:.3} ms, Inner total: {:.3} ms", msd_total, inner_total);
    println!("    Full: {:.3} ms = {:.0} Mkeys/s", full_p50, full_mkeys);
    let inner_bw = (n * 4 * 2 * 3) as f64 / inner_total / 1e6;
    println!("    Inner effective BW: {:.0} GB/s (SLC=469, DRAM=245)", inner_bw);

    // Investigation G: Batched inner sort — sweep batch sizes for SLC effect
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

    println!("\n{}", "=".repeat(60));
    println!("Investigation Complete");
    println!("{}", "=".repeat(60));
}
