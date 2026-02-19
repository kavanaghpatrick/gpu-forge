//! Experiment 17: MSD+LSD Hybrid Radix Sort (5000+ Mkeys/s target)
//!
//! Architecture: 1 MSD scatter (bits 24:31) creates 256 buckets of ~62K
//! elements (~250KB each, SLC-resident), then 3 per-bucket LSD passes
//! at SLC speed. Single encoder, 14 dispatches, zero CPU readback.
//!
//! Phase 0: SLC scatter bandwidth benchmark — gates the entire experiment.

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
use std::ptr::NonNull;

const WARMUP: usize = 5;
const RUNS: usize = 20;
const THREADS_PER_TG: usize = 256;

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

fn print_stats(label: &str, n: usize, times: &[f64]) {
    let p5 = percentile(times, 5.0);
    let p50 = percentile(times, 50.0);
    let p95 = percentile(times, 95.0);
    let bw = (n * 4 * 2) as f64 / p50 / 1e6; // read + write = 2x data
    println!(
        "    {:>8}: {:>7.3} ms  {:>6.0} GB/s  [p5={:.3} p95={:.3}]",
        label, p50, bw, p5, p95
    );
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 17: MSD+LSD Hybrid Radix Sort");
    println!("{}", "=".repeat(60));
    println!("Phase 0: SLC Scatter Bandwidth Benchmark");
    println!("  Reusing exp16_diag_scatter_binned kernel");
    println!("  Goal: measure scatter BW at SLC-resident sizes\n");

    let pso_scatter_binned = ctx.make_pipeline("exp16_diag_scatter_binned");

    let sizes: [usize; 5] = [62_500, 250_000, 1_000_000, 4_000_000, 16_000_000];
    let mut bw_250k: f64 = 0.0;

    println!("  ── 256-bin Scatter Bandwidth vs Working Set Size ──");

    for &n in &sizes {
        let label = if n >= 1_000_000 {
            format!("{}M", n / 1_000_000)
        } else {
            format!("{}K", n / 1_000)
        };
        let ws_kb = n * 4 * 2 / 1024; // working set in KB (src + dst)
        let ws_label = if ws_kb >= 1024 {
            format!("{}MB", ws_kb / 1024)
        } else {
            format!("{}KB", ws_kb)
        };

        // Generate random data
        let data: Vec<u32> = {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            (0..n).map(|_| rng.gen::<u32>()).collect()
        };

        // Compute 256-bin scatter offsets on CPU (same pattern as exp16 lines 1200-1224)
        let bin_offsets: Vec<u32> = {
            let mut hist = vec![0u32; 256];
            for &v in &data {
                hist[(v & 0xFF) as usize] += 1;
            }
            let mut prefix = vec![0u32; 256];
            let mut sum = 0u32;
            for i in 0..256 {
                prefix[i] = sum;
                sum += hist[i];
            }
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

        let tg_size = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        let grid_tgs = MTLSize {
            width: n.div_ceil(THREADS_PER_TG),
            height: 1,
            depth: 1,
        };

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
                    NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(),
                    4,
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_tgs, tg_size);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            if iter >= WARMUP {
                times.push(gpu_elapsed_ms(&cmd));
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&times, 50.0);
        let bw = (n * 4 * 2) as f64 / p50 / 1e6;

        print_stats(&format!("{} ({})", label, ws_label), n, &times);

        if n == 250_000 {
            bw_250k = bw;
        }
    }

    // Go/no-go gate
    println!();
    if bw_250k < 80.0 {
        println!(
            "  ABORT: SLC scatter too slow — 250K scatter = {:.0} GB/s (need >= 80 GB/s)",
            bw_250k
        );
        println!("  Hybrid approach is not viable on this hardware.");
        // Continue anyway for development -- Phase 0 gate is informational in POC
    } else {
        println!(
            "  GO: SLC scatter at 250K = {:.0} GB/s (>= 80 GB/s threshold)",
            bw_250k
        );
        println!("  Hybrid approach is viable — proceeding with MSD+LSD sort.");
    }

    // Phase 1: MSD histogram
    bench_msd_histogram(ctx);

    // Phase 1b: MSD scatter (full pipeline: histogram → bucket_descs → prefix → zero → scatter)
    run_msd_scatter(ctx);

    // Phase 2: End-to-end hybrid sort (MSD scatter + 3 inner LSD passes)
    run_hybrid_sort(ctx);
}

/// BucketDesc: describes one MSD bucket (offset, count, tile_count, tile_base)
/// Must match the Metal struct layout exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct BucketDesc {
    offset: u32,
    count: u32,
    tile_count: u32,
    tile_base: u32,
}

/// Phase 1: MSD histogram benchmark
/// Dispatches exp17_msd_histogram (1-pass, bits 24:31) + exp17_compute_bucket_descs + exp17_global_prefix.
/// Verifies: 256 bins sum to N, prefix sums monotonically increasing, BucketDesc offsets correct.
fn bench_msd_histogram(ctx: &MetalContext) {
    println!("\n{}", "-".repeat(60));
    println!("Phase 1: MSD Histogram (1-pass, bits 24:31)");
    println!("{}", "-".repeat(60));

    let n: usize = 16_000_000;
    let tile_size: usize = 4096;
    let num_tiles = n.div_ceil(tile_size);

    println!("  N = {}M, num_tiles = {}", n / 1_000_000, num_tiles);

    // Generate random input
    let data: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };

    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    // Allocate 256 bins for histogram (zeroed)
    let hist_zeros = vec![0u32; 256];
    let buf_msd_hist = alloc_buffer_with_data(&ctx.device, &hist_zeros);
    // Allocate BucketDesc[256] (256 * 16 bytes)
    let bucket_desc_zeros = vec![0u8; 256 * std::mem::size_of::<BucketDesc>()];
    let buf_bucket_descs = alloc_buffer_with_data(&ctx.device, &bucket_desc_zeros);

    // Exp17Params: element_count, num_tiles, shift, pass
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Exp17Params {
        element_count: u32,
        num_tiles: u32,
        shift: u32,
        pass: u32,
    }

    let params = Exp17Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        shift: 24,
        pass: 0,
    };

    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");

    let tg_size = MTLSize {
        width: THREADS_PER_TG,
        height: 1,
        depth: 1,
    };
    let hist_grid = MTLSize {
        width: num_tiles,
        height: 1,
        depth: 1,
    };
    let one_tg_grid = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    // --- Step 1: Run histogram ---
    // Zero the histogram buffer first
    unsafe {
        let ptr = buf_msd_hist.contents().as_ptr() as *mut u32;
        std::ptr::write_bytes(ptr, 0, 256);
    }

    let cmd = ctx.command_buffer();
    let enc = cmd.computeCommandEncoder().unwrap();

    // Dispatch histogram
    enc.setComputePipelineState(&pso_histogram);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_input.as_ref()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(&params as *const Exp17Params as *mut _).unwrap(),
            std::mem::size_of::<Exp17Params>(),
            2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    // Read back raw histogram (before prefix sum / bucket descs destroy it)
    let raw_hist: Vec<u32> = unsafe { read_buffer_slice(&buf_msd_hist, 256) };
    let hist_sum: u64 = raw_hist.iter().map(|&x| x as u64).sum();
    let min_bucket = *raw_hist.iter().min().unwrap();
    let max_bucket = *raw_hist.iter().max().unwrap();
    let avg_bucket = hist_sum as f64 / 256.0;

    println!("  Histogram sum: {} (expected {})", hist_sum, n);
    if hist_sum == n as u64 {
        println!("  Histogram sum check: ok");
    } else {
        println!("  Histogram sum check: FAIL (off by {})", (hist_sum as i64 - n as i64).abs());
    }
    println!(
        "  Bucket stats: min={}, max={}, avg={:.0}",
        min_bucket, max_bucket, avg_bucket
    );

    // --- Step 2: Compute BucketDesc (reads raw histogram, must run BEFORE global_prefix) ---
    let tile_size_u32 = tile_size as u32;
    let cmd_bd = ctx.command_buffer();
    let enc_bd = cmd_bd.computeCommandEncoder().unwrap();

    enc_bd.setComputePipelineState(&pso_bucket_descs);
    unsafe {
        enc_bd.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
        enc_bd.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 1);
        enc_bd.setBytes_length_atIndex(
            NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(),
            4,
            2,
        );
    }
    enc_bd.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

    enc_bd.endEncoding();
    cmd_bd.commit();
    cmd_bd.waitUntilCompleted();

    // Read back BucketDesc[256]
    let bucket_descs: Vec<BucketDesc> = unsafe {
        let ptr = buf_bucket_descs.contents().as_ptr() as *const BucketDesc;
        std::slice::from_raw_parts(ptr, 256).to_vec()
    };

    // Verify BucketDesc: sum of counts == N
    let bd_count_sum: u64 = bucket_descs.iter().map(|d| d.count as u64).sum();
    println!("\n  BucketDesc verification:");
    println!("  Sum of counts: {} (expected {})", bd_count_sum, n);
    if bd_count_sum == n as u64 {
        println!("  BucketDesc count sum check: ok");
    } else {
        println!("  BucketDesc count sum check: FAIL");
    }

    // Verify offsets monotonically increasing
    let mut bd_monotonic = true;
    for i in 1..256 {
        if bucket_descs[i].offset < bucket_descs[i - 1].offset {
            bd_monotonic = false;
            println!(
                "  BucketDesc offset monotonicity FAIL at bucket {}: {} < {}",
                i, bucket_descs[i].offset, bucket_descs[i - 1].offset
            );
            break;
        }
    }
    if bd_monotonic {
        println!("  BucketDesc offsets: monotonically increasing: ok");
    }

    // Verify offsets[255] + counts[255] == N
    let last_end = bucket_descs[255].offset as u64 + bucket_descs[255].count as u64;
    if last_end == n as u64 {
        println!("  BucketDesc final check (offset[255] + count[255] = N): ok");
    } else {
        println!(
            "  BucketDesc final check: FAIL (offset[255]={} + count[255]={} = {}, expected {})",
            bucket_descs[255].offset, bucket_descs[255].count, last_end, n
        );
    }

    // Verify counts match raw histogram
    let mut counts_match = true;
    for i in 0..256 {
        if bucket_descs[i].count != raw_hist[i] {
            counts_match = false;
            println!(
                "  BucketDesc count mismatch at bucket {}: desc={} vs hist={}",
                i, bucket_descs[i].count, raw_hist[i]
            );
            break;
        }
    }
    if counts_match {
        println!("  BucketDesc counts match raw histogram: ok");
    }

    // Print first few bucket descs
    println!("  First 4 BucketDescs:");
    for i in 0..4 {
        let d = &bucket_descs[i];
        println!(
            "    [{}] offset={}, count={}, tile_count={}, tile_base={}",
            i, d.offset, d.count, d.tile_count, d.tile_base
        );
    }
    println!("  Last BucketDesc:");
    {
        let d = &bucket_descs[255];
        println!(
            "    [255] offset={}, count={}, tile_count={}, tile_base={}",
            d.offset, d.count, d.tile_count, d.tile_base
        );
    }

    // --- Step 3: Run global prefix sum ---
    let cmd2 = ctx.command_buffer();
    let enc2 = cmd2.computeCommandEncoder().unwrap();

    enc2.setComputePipelineState(&pso_prefix);
    unsafe {
        enc2.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
    }
    enc2.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

    enc2.endEncoding();
    cmd2.commit();
    cmd2.waitUntilCompleted();

    // Read back prefix sums
    let prefix_sums: Vec<u32> = unsafe { read_buffer_slice(&buf_msd_hist, 256) };

    // Verify monotonically increasing
    let mut monotonic = true;
    for i in 1..256 {
        if prefix_sums[i] < prefix_sums[i - 1] {
            monotonic = false;
            println!(
                "  Prefix monotonicity FAIL at bin {}: {} < {}",
                i, prefix_sums[i], prefix_sums[i - 1]
            );
            break;
        }
    }
    if monotonic {
        println!("  Prefix sums: monotonically increasing: ok");
    }

    // Verify prefix sums match BucketDesc offsets
    let mut prefix_match = true;
    for i in 0..256 {
        if prefix_sums[i] != bucket_descs[i].offset {
            prefix_match = false;
            println!(
                "  Prefix vs BucketDesc offset mismatch at bin {}: prefix={} vs bd_offset={}",
                i, prefix_sums[i], bucket_descs[i].offset
            );
            break;
        }
    }
    if prefix_match {
        println!("  Prefix sums match BucketDesc offsets: ok");
    }

    // Verify last prefix + last raw count = N
    // (prefix_sums[255] = sum of raw_hist[0..254], so prefix_sums[255] + raw_hist[255] should == N)
    let last_prefix_plus_count = prefix_sums[255] as u64 + raw_hist[255] as u64;
    if last_prefix_plus_count == n as u64 {
        println!("  Prefix final check (prefix[255] + count[255] = N): ok");
    } else {
        println!(
            "  Prefix final check: FAIL (prefix[255]={} + count[255]={} = {}, expected {})",
            prefix_sums[255], raw_hist[255], last_prefix_plus_count, n
        );
    }

    println!("  First 8 prefix sums: {:?}", &prefix_sums[..8]);
    println!("  Last 4 prefix sums: {:?}", &prefix_sums[252..256]);
    println!("  First 8 raw counts: {:?}", &raw_hist[..8]);
}

/// Exp16Params: needed for exp16_partition and exp16_zero_status (5-field struct).
/// Must match the Metal Exp16Params layout exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct Exp16Params {
    element_count: u32,
    num_tiles: u32,
    num_tgs: u32,
    shift: u32,
    pass: u32,
}

/// Phase 1b: MSD Scatter — full single-encoder pipeline.
///
/// Dispatches in one command buffer, one encoder, PSO switching:
///   1. exp17_msd_histogram (3907 TGs) — 256-bin histogram of bits[24:31]
///   2. exp17_compute_bucket_descs (1 TG) — reads raw histogram to build BucketDesc[256]
///   3. exp17_global_prefix (1 TG) — in-place prefix sum on histogram
///   4. exp16_zero_status (3907 TGs) — zero tile_status + counters for scatter
///   5. exp16_partition (3907 TGs) — decoupled lookback scatter, shift=24, pass=0
///
/// Correctness check: every element in buf_b at bucket position has matching MSD byte.
fn run_msd_scatter(ctx: &MetalContext) {
    println!("\n{}", "-".repeat(60));
    println!("Phase 1b: MSD Scatter (single-encoder pipeline)");
    println!("{}", "-".repeat(60));

    let n: usize = 16_000_000;
    let tile_size: usize = 4096;
    let num_tiles = n.div_ceil(tile_size); // 3907

    println!("  N = {}M, num_tiles = {}", n / 1_000_000, num_tiles);

    // Generate random input
    let data: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };

    // ── Allocate buffers ──
    let buf_input = alloc_buffer_with_data(&ctx.device, &data);
    let buf_a = alloc_buffer(&ctx.device, n * 4);
    let buf_b = alloc_buffer(&ctx.device, n * 4);

    // Copy input to buf_a (scatter reads from buf_a, writes to buf_b)
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, n * 4);
    }

    // MSD histogram buffer: 4 passes * 256 bins * 4 bytes = 4096 bytes
    // (exp16_partition reads global_hist[pass*256+d]; with pass=0 it reads [0..255])
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * 256 * 4);
    // Zero the histogram buffer
    unsafe {
        std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
    }

    // Tile status for decoupled lookback: num_tiles * 256 entries * 4 bytes
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * 256 * 4);

    // Atomic work counter for exp16_partition: 4 bytes
    let buf_counters = alloc_buffer(&ctx.device, 4);

    // BucketDesc[256]: 256 * 16 bytes
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());

    // ── Build PSOs ──
    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");

    // ── Params ──
    // Exp17Params for MSD histogram (4 fields)
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Exp17Params {
        element_count: u32,
        num_tiles: u32,
        shift: u32,
        pass: u32,
    }

    let exp17_params = Exp17Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        shift: 24,
        pass: 0,
    };

    // Exp16Params for zero_status and partition (5 fields)
    let exp16_params = Exp16Params {
        element_count: n as u32,
        num_tiles: num_tiles as u32,
        num_tgs: num_tiles as u32,
        shift: 24,
        pass: 0,
    };

    let tile_size_u32 = tile_size as u32;

    // ── Grid sizes ──
    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    // Zero TGs: num_tiles * 256 entries / 256 threads per TG = num_tiles
    let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };

    // ════════════════════════════════════════════════════════════════
    // Single command buffer, single encoder, PSO switching
    // ════════════════════════════════════════════════════════════════
    let cmd = ctx.command_buffer();
    let enc = cmd.computeCommandEncoder().unwrap();

    // ── Dispatch 1: MSD histogram (3907 TGs) ──
    enc.setComputePipelineState(&pso_histogram);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);       // src data
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 1); // global_hist
        enc.setBytes_length_atIndex(
            NonNull::new(&exp17_params as *const Exp17Params as *mut _).unwrap(),
            std::mem::size_of::<Exp17Params>(), 2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

    // ── Dispatch 2: Compute BucketDesc (1 TG) ──
    // Must run AFTER histogram but BEFORE global_prefix (prefix destroys raw counts)
    enc.setComputePipelineState(&pso_bucket_descs);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);       // global_hist (raw counts)
        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 1);   // bucket_descs output
        enc.setBytes_length_atIndex(
            NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(),
            4, 2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

    // ── Dispatch 3: Global prefix sum (1 TG) ──
    enc.setComputePipelineState(&pso_prefix);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0); // in-place prefix sum
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

    // ── Dispatch 4: Zero tile_status + counters (3907 TGs) ──
    enc.setComputePipelineState(&pso_zero_status);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 0); // tile_status
        enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 1);    // counters
        enc.setBytes_length_atIndex(
            NonNull::new(&exp16_params as *const Exp16Params as *mut _).unwrap(),
            std::mem::size_of::<Exp16Params>(), 2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(zero_grid, tg_size);

    // ── Dispatch 5: MSD scatter / exp16_partition (3907 TGs) ──
    // shift=24, pass=0 → reads global_hist[0*256+d] = global_hist[d]
    enc.setComputePipelineState(&pso_partition);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);              // src
        enc.setBuffer_offset_atIndex(Some(buf_b.as_ref()), 0, 1);              // dst
        enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 2);    // tile_status
        enc.setBuffer_offset_atIndex(Some(buf_counters.as_ref()), 0, 3);       // counters
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 4);       // global_hist (prefix sums)
        enc.setBytes_length_atIndex(
            NonNull::new(&exp16_params as *const Exp16Params as *mut _).unwrap(),
            std::mem::size_of::<Exp16Params>(), 5,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);

    // ── Commit + wait ──
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let elapsed = gpu_elapsed_ms(&cmd);
    println!("  Pipeline time: {:.3} ms", elapsed);

    // ════════════════════════════════════════════════════════════════
    // Correctness check: verify each element is in the correct bucket
    // ════════════════════════════════════════════════════════════════
    let output: Vec<u32> = unsafe { read_buffer_slice(&buf_b, n) };
    let bucket_descs: Vec<BucketDesc> = unsafe {
        let ptr = buf_bucket_descs.contents().as_ptr() as *const BucketDesc;
        std::slice::from_raw_parts(ptr, 256).to_vec()
    };

    let mut mismatches = 0u64;
    let mut first_mismatches: Vec<String> = Vec::new();

    for b in 0..256u32 {
        let desc = &bucket_descs[b as usize];
        let start = desc.offset as usize;
        let end = start + desc.count as usize;
        for i in start..end {
            let elem = output[i];
            let msd_byte = (elem >> 24) & 0xFF;
            if msd_byte != b {
                mismatches += 1;
                if first_mismatches.len() < 5 {
                    first_mismatches.push(format!(
                        "    idx={}: elem=0x{:08X}, MSD byte={}, expected bucket={}",
                        i, elem, msd_byte, b
                    ));
                }
            }
        }
    }

    println!("  MSD scatter correctness: {} mismatches out of {} elements", mismatches, n);
    if mismatches == 0 {
        println!("  MSD scatter check: ok");
    } else {
        println!("  MSD scatter check: FAIL");
        for msg in &first_mismatches {
            println!("{}", msg);
        }
    }

    // Print a few bucket stats
    println!("  First 4 buckets:");
    for b in 0..4 {
        let d = &bucket_descs[b];
        println!("    [{}] offset={}, count={}", b, d.offset, d.count);
    }
    println!("  Last bucket:");
    {
        let d = &bucket_descs[255];
        println!("    [255] offset={}, count={}", d.offset, d.count);
    }

    // Verify total element count across all buckets
    let total: u64 = bucket_descs.iter().map(|d| d.count as u64).sum();
    println!("  Total elements across buckets: {} (expected {})", total, n);
}

/// Exp17InnerParams: shift for inner sort passes. Must match Metal struct.
#[repr(C)]
#[derive(Clone, Copy)]
struct Exp17InnerParams {
    shift: u32,
}

/// Phase 2: End-to-end hybrid sort — MSD scatter + 3 inner LSD passes.
///
/// Single command buffer, single encoder, 14 dispatches:
///   1-5: MSD pipeline (histogram → bucket_descs → prefix → zero → scatter)
///   6-14: Inner sort (3 passes × [zero + histogram + scan_scatter])
///
/// After MSD: data in buf_b (partitioned by MSD byte).
/// Inner passes ping-pong: pass0 buf_b→buf_a, pass1 buf_a→buf_b, pass2 buf_b→buf_a.
/// Final result in buf_a.
fn run_hybrid_sort(ctx: &MetalContext) {
    println!("\n{}", "-".repeat(60));
    println!("Phase 2: End-to-End Hybrid Sort (MSD + 3 Inner LSD)");
    println!("{}", "-".repeat(60));

    let n: usize = 16_000_000;
    let tile_size: usize = 4096;
    let num_tiles = n.div_ceil(tile_size); // 3907

    println!("  N = {}M, num_tiles = {}", n / 1_000_000, num_tiles);

    // Generate random input
    let data: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };

    // CPU-side expected result
    let mut expected = data.clone();
    expected.sort();

    // ── Allocate buffers ──
    let buf_a = alloc_buffer(&ctx.device, n * 4);       // 64 MB
    let buf_b = alloc_buffer(&ctx.device, n * 4);       // 64 MB

    // Copy input to buf_a
    unsafe {
        let src_ptr = data.as_ptr() as *const u8;
        let dst_ptr = buf_a.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, n * 4);
    }

    // MSD histogram: 4 passes * 256 bins * 4 bytes (exp16_partition compat)
    let buf_msd_hist = alloc_buffer(&ctx.device, 4 * 256 * 4);
    unsafe {
        std::ptr::write_bytes(buf_msd_hist.contents().as_ptr() as *mut u8, 0, 4 * 256 * 4);
    }

    // Tile status for decoupled lookback: num_tiles * 256 * 4 bytes
    let buf_tile_status = alloc_buffer(&ctx.device, num_tiles * 256 * 4);

    // Atomic work counter for exp16_partition: 4 bytes
    let buf_counters = alloc_buffer(&ctx.device, 4);

    // BucketDesc[256]: 256 * 16 bytes
    let buf_bucket_descs = alloc_buffer(&ctx.device, 256 * std::mem::size_of::<BucketDesc>());

    // Inner tile histograms: 256 buckets * 17 tiles * 256 bins * 4 bytes = ~4.5 MB
    let tile_hists_entries: usize = 256 * 17 * 256;
    let buf_tile_hists = alloc_buffer(&ctx.device, tile_hists_entries * 4);

    // ── Build PSOs ──
    let pso_histogram = ctx.make_pipeline("exp17_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp17_compute_bucket_descs");
    let pso_prefix = ctx.make_pipeline("exp17_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_inner_zero = ctx.make_pipeline("exp17_inner_zero");
    let pso_inner_histogram = ctx.make_pipeline("exp17_inner_histogram");
    let pso_inner_scan_scatter = ctx.make_pipeline("exp17_inner_scan_scatter");

    // ── Params ──
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Exp17Params {
        element_count: u32,
        num_tiles: u32,
        shift: u32,
        pass: u32,
    }

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

    let tile_size_u32 = tile_size as u32;
    let tile_hists_total = tile_hists_entries as u32;

    // Inner dispatch: 4352 TGs = 256 buckets * 17 tiles/bucket
    let inner_tg_count: usize = 256 * 17; // 4352
    // Inner zero dispatch: ceil(tile_hists_entries / 256)
    let inner_zero_tg_count = tile_hists_entries.div_ceil(THREADS_PER_TG); // 4352

    // ── Grid sizes ──
    let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
    let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };
    let one_tg_grid = MTLSize { width: 1, height: 1, depth: 1 };
    let zero_tg_count = (num_tiles * 256).div_ceil(THREADS_PER_TG);
    let zero_grid = MTLSize { width: zero_tg_count, height: 1, depth: 1 };
    let inner_grid = MTLSize { width: inner_tg_count, height: 1, depth: 1 };
    let inner_zero_grid = MTLSize { width: inner_zero_tg_count, height: 1, depth: 1 };

    // ════════════════════════════════════════════════════════════════
    // Single command buffer, single encoder, 14 dispatches
    // ════════════════════════════════════════════════════════════════
    let cmd = ctx.command_buffer();
    let enc = cmd.computeCommandEncoder().unwrap();

    // ── Dispatch 1: MSD histogram (3907 TGs) ──
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

    // ── Dispatch 2: Compute BucketDesc (1 TG) ──
    enc.setComputePipelineState(&pso_bucket_descs);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_bucket_descs.as_ref()), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(&tile_size_u32 as *const u32 as *mut _).unwrap(),
            4, 2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

    // ── Dispatch 3: Global prefix sum (1 TG) ──
    enc.setComputePipelineState(&pso_prefix);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_msd_hist.as_ref()), 0, 0);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg_grid, tg_size);

    // ── Dispatch 4: Zero tile_status + counters ──
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

    // ── Dispatch 5: MSD scatter / exp16_partition (3907 TGs) ──
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

    // ════════════════════════════════════════════════════════════════
    // Inner sort: 3 passes (shift=0, 8, 16), each = zero + histogram + scan_scatter
    // Ping-pong: pass0 buf_b→buf_a, pass1 buf_a→buf_b, pass2 buf_b→buf_a
    // ════════════════════════════════════════════════════════════════
    let inner_shifts: [u32; 3] = [0, 8, 16];

    for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
        let inner_params = Exp17InnerParams { shift };

        let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
            (buf_b.as_ref(), buf_a.as_ref())
        } else {
            (buf_a.as_ref(), buf_b.as_ref())
        };

        // ── Inner dispatch 1: Zero tile_hists ──
        enc.setComputePipelineState(&pso_inner_zero);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hists.as_ref()), 0, 0);
            enc.setBytes_length_atIndex(
                NonNull::new(&tile_hists_total as *const u32 as *mut _).unwrap(),
                4, 1,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(inner_zero_grid, tg_size);

        // ── Inner dispatch 2: Inner histogram (4352 TGs) ──
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

        // ── Inner dispatch 3: Inner scan+scatter (4352 TGs) ──
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

    // ── Commit + wait ──
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let elapsed = gpu_elapsed_ms(&cmd);
    println!("  Total pipeline time: {:.3} ms", elapsed);
    println!("  Throughput: {:.0} Mkeys/s", n as f64 / elapsed / 1e3);

    // ════════════════════════════════════════════════════════════════
    // Correctness check: compare buf_a with CPU-sorted expected
    // After 3 inner passes (odd count), result is in buf_a
    // ════════════════════════════════════════════════════════════════
    let result: Vec<u32> = unsafe { read_buffer_slice(&buf_a, n) };

    let mut mismatches = 0usize;
    let mut first_mismatches: Vec<String> = Vec::new();
    for i in 0..n {
        if result[i] != expected[i] {
            mismatches += 1;
            if first_mismatches.len() < 5 {
                first_mismatches.push(format!(
                    "    idx={}: got=0x{:08X}, expected=0x{:08X}",
                    i, result[i], expected[i]
                ));
            }
        }
    }

    println!("  End-to-end correctness: {} mismatches out of {} elements", mismatches, n);
    if mismatches == 0 {
        println!("  Hybrid sort check: ok");
    } else {
        println!("  Hybrid sort check: FAIL");
        for msg in &first_mismatches {
            println!("{}", msg);
        }
    }

    // ── Per-bucket correctness: verify elements within each bucket range are sorted ──
    let bucket_descs_data: Vec<BucketDesc> = unsafe {
        let ptr = buf_bucket_descs.contents().as_ptr() as *const BucketDesc;
        std::slice::from_raw_parts(ptr, 256).to_vec()
    };

    let mut bucket_failures = 0usize;
    for b in 0..256u32 {
        let desc = &bucket_descs_data[b as usize];
        let start = desc.offset as usize;
        let end = start + desc.count as usize;
        if end <= start { continue; }

        // Check elements in this bucket are sorted
        let mut sorted = true;
        for i in start..(end - 1) {
            if result[i] > result[i + 1] {
                sorted = false;
                if bucket_failures < 5 {
                    println!(
                        "    Bucket {} unsorted at idx {}: 0x{:08X} > 0x{:08X}",
                        b, i, result[i], result[i + 1]
                    );
                }
                break;
            }
        }
        if !sorted {
            bucket_failures += 1;
        }
    }

    if bucket_failures == 0 {
        println!("  Per-bucket sorted check: ok (all 256 buckets internally sorted)");
    } else {
        println!("  Per-bucket sorted check: FAIL ({} buckets not sorted)", bucket_failures);
    }
}
