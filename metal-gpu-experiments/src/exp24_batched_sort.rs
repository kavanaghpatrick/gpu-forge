//! Experiment 24: Batched Inner Passes for SLC Residency
//!
//! Architecture: 1 MSD scatter (bits 24-31) → 256 buckets (~250KB each)
//!               + 3 per-bucket LSD passes with BATCHED dispatch
//!
//! Key insight from multi-AI analysis (Codex, Grok, KB Agent):
//! exp23 dispatched ALL ~4096 inner TGs simultaneously, creating a
//! 64MB working set that exceeds 24MB SLC → DRAM speed (73 GB/s).
//!
//! Fix: Dispatch inner passes in batches of ~90 buckets (~22MB),
//! keeping the active working set under SLC capacity (24MB).
//!
//! Expected: inner passes at SLC speed (131-469 GB/s) → 5000+ Mkeys/s

use crate::metal_ctx::*;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};
use std::ptr::NonNull;

const WARMUP: usize = 5;
const RUNS: usize = 30;
const TG_SIZE: usize = 256;
const MSD_TILE: usize = 4096;
const INNER_TILE: usize = 4096;
const NUM_BINS: usize = 256;
const SLC_BUDGET: usize = 20_000_000; // 20MB — best tested config

type Pso =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

#[repr(C)]
#[derive(Clone, Copy)]
struct E24Params {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    pass: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct E24BatchParams {
    shift: u32,
    batch_start: u32,
    batch_count: u32,
    tile_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct E24BucketDesc {
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

fn percentile(s: &[f64], p: f64) -> f64 {
    let i = (p / 100.0) * (s.len() - 1) as f64;
    let (lo, hi) = (i.floor() as usize, i.ceil() as usize);
    if lo == hi {
        s[lo]
    } else {
        s[lo] + (s[hi] - s[lo]) * (i - lo as f64)
    }
}

fn gen_random(n: usize) -> Vec<u32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}

/// Compute batch boundaries from bucket descriptors.
/// Groups consecutive buckets where cumulative data size < SLC_BUDGET.
/// Returns Vec<(batch_start, batch_count, batch_tiles)>.
fn compute_batches(descs: &[E24BucketDesc]) -> Vec<(u32, u32, u32)> {
    let mut batches = Vec::new();
    let mut batch_start = 0u32;
    let mut batch_bytes = 0usize;
    let mut batch_tiles = 0u32;

    for (i, desc) in descs.iter().enumerate() {
        let bucket_bytes = desc.count as usize * 4 * 2; // src + dst
        if batch_bytes + bucket_bytes > SLC_BUDGET && i as u32 > batch_start {
            // Close current batch
            batches.push((batch_start, i as u32 - batch_start, batch_tiles));
            batch_start = i as u32;
            batch_bytes = 0;
            batch_tiles = 0;
        }
        batch_bytes += bucket_bytes;
        batch_tiles += desc.tile_count;
    }
    // Last batch
    if batch_start < descs.len() as u32 {
        batches.push((batch_start, descs.len() as u32 - batch_start, batch_tiles));
    }
    batches
}

/// Encode the full sort pipeline with batched inner passes.
fn encode_pipeline(
    cmd: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
    n: usize,
    buf_a: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_b: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_hist: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_descs: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_msd_status: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_inner_status: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_bucket_hist: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_ctr: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    pso_msd_hist: &Pso,
    pso_bucket_descs: &Pso,
    pso_global_pfx: &Pso,
    pso_zero_status: &Pso,
    pso_partition: &Pso,
    pso_inner_hist: &Pso,
    pso_inner_pfx: &Pso,
    pso_inner_partition: &Pso,
    batches: &[(u32, u32, u32)], // (batch_start, batch_count, batch_tiles)
    descs: &[E24BucketDesc],
    total_inner_tiles: usize,
    skip_inner: bool,
) {
    let mt = n.div_ceil(MSD_TILE);

    let msd_p = E24Params {
        element_count: n as u32,
        num_tiles: mt as u32,
        shift: 24,
        pass: 0,
    };
    let e16_p = Exp16Params {
        element_count: n as u32,
        num_tiles: mt as u32,
        num_tgs: mt as u32,
        shift: 24,
        pass: 0,
    };

    let tg = MTLSize {
        width: TG_SIZE,
        height: 1,
        depth: 1,
    };

    // Zero histogram (CPU-side)
    unsafe {
        std::ptr::write_bytes(buf_hist.contents().as_ptr() as *mut u8, 0, NUM_BINS * 4);
    }

    // ── Encoder 1: MSD phase ─────────────────────────────────────
    let enc = cmd.computeCommandEncoder().unwrap();

    // 1) MSD histogram
    enc.setComputePipelineState(pso_msd_hist);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_hist), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(&msd_p as *const _ as *mut _).unwrap(),
            std::mem::size_of_val(&msd_p),
            2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: mt, height: 1, depth: 1 },
        tg,
    );

    // 2) Bucket descriptors
    enc.setComputePipelineState(pso_bucket_descs);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_hist), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_descs), 0, 1);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: 1, height: 1, depth: 1 },
        tg,
    );

    // 3) Global prefix sum
    enc.setComputePipelineState(pso_global_pfx);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_hist), 0, 0);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: 1, height: 1, depth: 1 },
        tg,
    );

    // 4) Zero MSD tile status + counter
    let zt = (mt * NUM_BINS).div_ceil(TG_SIZE);
    enc.setComputePipelineState(pso_zero_status);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_msd_status), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_ctr), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(&e16_p as *const _ as *mut _).unwrap(),
            std::mem::size_of_val(&e16_p),
            2,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: zt, height: 1, depth: 1 },
        tg,
    );

    // 5) MSD partition: buf_a → buf_b (shift=24)
    enc.setComputePipelineState(pso_partition);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_msd_status), 0, 2);
        enc.setBuffer_offset_atIndex(Some(buf_ctr), 0, 3);
        enc.setBuffer_offset_atIndex(Some(buf_hist), 0, 4);
        enc.setBytes_length_atIndex(
            NonNull::new(&e16_p as *const _ as *mut _).unwrap(),
            std::mem::size_of_val(&e16_p),
            5,
        );
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: mt, height: 1, depth: 1 },
        tg,
    );

    enc.endEncoding();

    if skip_inner {
        return;
    }

    // ── BATCHED Inner LSD passes ──────────────────────────────────
    // Pass 0 (shift=0):  buf_b → buf_a
    // Pass 1 (shift=8):  buf_a → buf_b
    // Pass 2 (shift=16): buf_b → buf_a
    // Result in buf_a after 3 passes (odd).
    let inner_shifts = [0u32, 8, 16];

    // Single encoder for all inner passes
    let enc = cmd.computeCommandEncoder().unwrap();

    for (pass_idx, &shift) in inner_shifts.iter().enumerate() {
        let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
            (buf_b as &_, buf_a as &_)
        } else {
            (buf_a as &_, buf_b as &_)
        };

        // Process each batch separately
        for &(batch_start, batch_count, batch_tiles) in batches {
            if batch_tiles == 0 {
                continue;
            }

            // Compute tile_offset: sum of tile_counts for buckets before batch_start
            let tile_offset = descs[batch_start as usize].tile_base;

            let batch_p = E24BatchParams {
                shift,
                batch_start,
                batch_count,
                tile_offset,
            };

            // 1) Per-bucket histogram (batch_count TGs)
            enc.setComputePipelineState(pso_inner_hist);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_bucket_hist), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_descs), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&batch_p as *const _ as *mut _).unwrap(),
                    std::mem::size_of_val(&batch_p),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: batch_count as usize, height: 1, depth: 1 },
                tg,
            );

            // 2) Per-bucket exclusive prefix sum (batch_count TGs)
            enc.setComputePipelineState(pso_inner_pfx);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_bucket_hist), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::new(&batch_p as *const _ as *mut _).unwrap(),
                    std::mem::size_of_val(&batch_p),
                    1,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: batch_count as usize, height: 1, depth: 1 },
                tg,
            );

            // 3) Zero ONLY this batch's tile_status entries using buffer offset
            // Offset into tile_status: tile_offset * NUM_BINS * 4 bytes
            let batch_zero_p = Exp16Params {
                element_count: 0,
                num_tiles: batch_tiles,
                num_tgs: 0,
                shift: 0,
                pass: 0,
            };
            let status_byte_offset = tile_offset as usize * NUM_BINS * 4;
            enc.setComputePipelineState(pso_zero_status);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_inner_status), status_byte_offset, 0);
                enc.setBuffer_offset_atIndex(Some(buf_ctr), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&batch_zero_p as *const _ as *mut _).unwrap(),
                    std::mem::size_of_val(&batch_zero_p),
                    2,
                );
            }
            let inner_zero_tgs = (batch_tiles as usize * NUM_BINS).div_ceil(TG_SIZE);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: inner_zero_tgs, height: 1, depth: 1 },
                tg,
            );

            // 4) Inner partition (batch_tiles TGs)
            enc.setComputePipelineState(pso_inner_partition);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(buf_inner_status), 0, 2);
                enc.setBuffer_offset_atIndex(Some(buf_descs), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&batch_p as *const _ as *mut _).unwrap(),
                    std::mem::size_of_val(&batch_p),
                    4,
                );
                enc.setBuffer_offset_atIndex(Some(buf_bucket_hist), 0, 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: batch_tiles as usize, height: 1, depth: 1 },
                tg,
            );
        }
    }

    enc.endEncoding();
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "═".repeat(60));
    println!("  EXP 24: Batched Inner Passes for SLC Residency");
    println!("  MSD(24:31) → 3 batched LSD passes (~22MB/batch)");
    println!("  Keeps working set < 24MB SLC for inner passes");
    println!("{}", "═".repeat(60));

    let pso_msd_hist = ctx.make_pipeline("exp24_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp24_compute_bucket_descs");
    let pso_global_pfx = ctx.make_pipeline("exp24_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_inner_hist = ctx.make_pipeline("exp24_inner_histogram");
    let pso_inner_pfx = ctx.make_pipeline("exp24_inner_prefix");
    let pso_inner_partition = ctx.make_pipeline("exp24_inner_partition");

    // ── Step 1: Run MSD to get bucket descriptors ──
    let n = 16_000_000usize;
    let data = gen_random(n);
    let total_inner_tiles = n.div_ceil(INNER_TILE) + NUM_BINS;

    let bi = alloc_buffer_with_data(&ctx.device, &data);
    let ba = alloc_buffer(&ctx.device, n * 4);
    let bb = alloc_buffer(&ctx.device, n * 4);
    let bh = alloc_buffer(&ctx.device, NUM_BINS * 4);
    let bd = alloc_buffer(&ctx.device, NUM_BINS * std::mem::size_of::<E24BucketDesc>());
    let bs_msd = alloc_buffer(&ctx.device, n.div_ceil(MSD_TILE) * NUM_BINS * 4);
    let bs_inner = alloc_buffer(&ctx.device, total_inner_tiles * NUM_BINS * 4);
    let bbh = alloc_buffer(&ctx.device, NUM_BINS * NUM_BINS * 4);
    let bc = alloc_buffer(&ctx.device, 4);

    // Run MSD-only to get bucket descriptors
    unsafe {
        std::ptr::copy_nonoverlapping(
            bi.contents().as_ptr() as *const u8,
            ba.contents().as_ptr() as *mut u8,
            n * 4,
        );
    }
    let cmd = ctx.command_buffer();
    encode_pipeline(
        &cmd, n, &ba, &bb, &bh, &bd, &bs_msd, &bs_inner, &bbh, &bc,
        &pso_msd_hist, &pso_bucket_descs, &pso_global_pfx,
        &pso_zero_status, &pso_partition,
        &pso_inner_hist, &pso_inner_pfx, &pso_inner_partition,
        &[], &[], total_inner_tiles, true,
    );
    cmd.commit();
    cmd.waitUntilCompleted();

    // Read bucket descriptors from GPU
    let descs: Vec<E24BucketDesc> = unsafe { read_buffer_slice(&bd, NUM_BINS) };
    let batches = compute_batches(&descs);

    // Print batch info
    let actual_inner_tiles: u32 = descs.iter().map(|d| d.tile_count).sum();
    let max_bucket = descs.iter().map(|d| d.count).max().unwrap_or(0);
    let avg_count = n as f64 / NUM_BINS as f64;
    let bucket_bytes = avg_count * 4.0;

    println!("\n  Batch Configuration:");
    println!("    SLC budget: {} MB", SLC_BUDGET / 1_000_000);
    println!("    Batches: {}", batches.len());
    for (i, &(start, count, tiles)) in batches.iter().enumerate() {
        let batch_bytes: u64 = descs[start as usize..(start + count) as usize]
            .iter()
            .map(|d| d.count as u64 * 4 * 2)
            .sum();
        println!(
            "    Batch {}: buckets {}..{} ({} tiles, {:.1} MB)",
            i, start, start + count - 1, tiles, batch_bytes as f64 / 1e6
        );
    }
    println!(
        "    Inner tiles: {} | max bucket {}K | avg {:.0}K ({:.1} KB)",
        actual_inner_tiles, max_bucket / 1000, avg_count / 1000.0, bucket_bytes / 1024.0
    );

    // ── Correctness tests ──
    for &sz in &[2048usize, 8192, 62_500, 250_000, 1_000_000, 4_000_000, 16_000_000] {
        let label = if sz >= 1_000_000 {
            format!("{}M", sz / 1_000_000)
        } else if sz >= 1_000 {
            format!("{}K", sz / 1_000)
        } else {
            format!("{}", sz)
        };
        print!("  Correctness @ {} ... ", label);

        let data = gen_random(sz);
        let mut expected = data.clone();
        expected.sort();

        let total_inner_tiles = sz.div_ceil(INNER_TILE) + NUM_BINS;

        let ba = alloc_buffer_with_data(&ctx.device, &data);
        let bb = alloc_buffer(&ctx.device, sz * 4);
        let bh = alloc_buffer(&ctx.device, NUM_BINS * 4);
        let bd = alloc_buffer(&ctx.device, NUM_BINS * std::mem::size_of::<E24BucketDesc>());
        let bs_msd = alloc_buffer(&ctx.device, sz.div_ceil(MSD_TILE) * NUM_BINS * 4);
        let bs_inner = alloc_buffer(&ctx.device, total_inner_tiles * NUM_BINS * 4);
        let bbh = alloc_buffer(&ctx.device, NUM_BINS * NUM_BINS * 4);
        let bc = alloc_buffer(&ctx.device, 4);

        // Run MSD-only first to get descs for batching
        let cmd = ctx.command_buffer();
        encode_pipeline(
            &cmd, sz, &ba, &bb, &bh, &bd, &bs_msd, &bs_inner, &bbh, &bc,
            &pso_msd_hist, &pso_bucket_descs, &pso_global_pfx,
            &pso_zero_status, &pso_partition,
            &pso_inner_hist, &pso_inner_pfx, &pso_inner_partition,
            &[], &[], total_inner_tiles, true,
        );
        cmd.commit();
        cmd.waitUntilCompleted();

        let descs: Vec<E24BucketDesc> = unsafe { read_buffer_slice(&bd, NUM_BINS) };
        let batches = compute_batches(&descs);

        // Re-copy data and run full sort
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ba.contents().as_ptr() as *mut u8,
                sz * 4,
            );
        }
        // Re-zero histogram since MSD-only corrupted it
        unsafe {
            std::ptr::write_bytes(bh.contents().as_ptr() as *mut u8, 0, NUM_BINS * 4);
        }

        let cmd = ctx.command_buffer();
        encode_pipeline(
            &cmd, sz, &ba, &bb, &bh, &bd, &bs_msd, &bs_inner, &bbh, &bc,
            &pso_msd_hist, &pso_bucket_descs, &pso_global_pfx,
            &pso_zero_status, &pso_partition,
            &pso_inner_hist, &pso_inner_pfx, &pso_inner_partition,
            &batches, &descs, total_inner_tiles, false,
        );
        cmd.commit();
        cmd.waitUntilCompleted();

        let result: Vec<u32> = unsafe { read_buffer_slice(&ba, sz) };
        let mm = result.iter().zip(&expected).filter(|(a, b)| a != b).count();
        if mm > 0 {
            println!("FAIL ({} mismatches / {})", mm, sz);
            let mut shown = 0;
            for i in 0..sz {
                if result[i] != expected[i] {
                    println!("    [{}] got=0x{:08X} exp=0x{:08X}", i, result[i], expected[i]);
                    shown += 1;
                    if shown >= 10 { break; }
                }
            }
            return;
        }
        println!("PASS");
    }

    // ── Benchmark @ 16M ──
    println!("\n  ── Benchmark @ 16M ──\n");

    // Full pipeline benchmark
    let mut full_times: Vec<f64> = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                bi.contents().as_ptr() as *const u8,
                ba.contents().as_ptr() as *mut u8,
                n * 4,
            );
        }
        let cmd = ctx.command_buffer();
        encode_pipeline(
            &cmd, n, &ba, &bb, &bh, &bd, &bs_msd, &bs_inner, &bbh, &bc,
            &pso_msd_hist, &pso_bucket_descs, &pso_global_pfx,
            &pso_zero_status, &pso_partition,
            &pso_inner_hist, &pso_inner_pfx, &pso_inner_partition,
            &batches, &descs, total_inner_tiles, false,
        );
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= WARMUP {
            full_times.push(gpu_elapsed_ms(&cmd));
        }
    }

    // MSD-only benchmark
    let mut msd_times: Vec<f64> = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                bi.contents().as_ptr() as *const u8,
                ba.contents().as_ptr() as *mut u8,
                n * 4,
            );
        }
        let cmd = ctx.command_buffer();
        encode_pipeline(
            &cmd, n, &ba, &bb, &bh, &bd, &bs_msd, &bs_inner, &bbh, &bc,
            &pso_msd_hist, &pso_bucket_descs, &pso_global_pfx,
            &pso_zero_status, &pso_partition,
            &pso_inner_hist, &pso_inner_pfx, &pso_inner_partition,
            &batches, &descs, total_inner_tiles, true,
        );
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= WARMUP {
            msd_times.push(gpu_elapsed_ms(&cmd));
        }
    }

    full_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    msd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let fp50 = percentile(&full_times, 50.0);
    let fp5 = percentile(&full_times, 5.0);
    let fp95 = percentile(&full_times, 95.0);
    let mp50 = percentile(&msd_times, 50.0);
    let inner_p50 = fp50 - mp50;

    let mkeys = n as f64 / fp50 / 1000.0;
    let bw = (n as f64 * 4.0 * 8.0) / fp50 / 1e6;

    // Verify correctness at 16M
    let mut expected = data.clone();
    expected.sort();
    unsafe {
        std::ptr::copy_nonoverlapping(
            bi.contents().as_ptr() as *const u8,
            ba.contents().as_ptr() as *mut u8,
            n * 4,
        );
    }
    let cmd = ctx.command_buffer();
    encode_pipeline(
        &cmd, n, &ba, &bb, &bh, &bd, &bs_msd, &bs_inner, &bbh, &bc,
        &pso_msd_hist, &pso_bucket_descs, &pso_global_pfx,
        &pso_zero_status, &pso_partition,
        &pso_inner_hist, &pso_inner_pfx, &pso_inner_partition,
        &batches, &descs, total_inner_tiles, false,
    );
    cmd.commit();
    cmd.waitUntilCompleted();
    let result: Vec<u32> = unsafe { read_buffer_slice(&ba, n) };
    let mm = result.iter().zip(&expected).filter(|(a, b)| a != b).count();

    if mm == 0 {
        println!("    Correctness: PASS (16M fully sorted)");
    } else {
        println!("    Correctness: FAIL ({} mismatches at 16M)", mm);
        let mut shown = 0;
        for i in 0..n {
            if result[i] != expected[i] {
                println!("    [{}] got=0x{:08X} exp=0x{:08X}", i, result[i], expected[i]);
                shown += 1;
                if shown >= 10 { break; }
            }
        }
    }

    println!(
        "    Full:  {:.3} ms | {:.0} Mkeys/s | {:.0} GB/s eff",
        fp50, mkeys, bw
    );
    println!(
        "    [p5={:.3} p95={:.3} spread={:.0}%]",
        fp5, fp95, (fp95 - fp5) / fp5 * 100.0
    );
    println!(
        "    MSD:   {:.3} ms | {:.0} GB/s",
        mp50, (n as f64 * 4.0 * 2.0) / mp50 / 1e6
    );
    if inner_p50 > 0.0 {
        println!(
            "    Inner: {:.3} ms | {:.0} GB/s (3 passes, {} batches)",
            inner_p50,
            (n as f64 * 4.0 * 6.0) / inner_p50 / 1e6,
            batches.len()
        );
        println!("    Per inner pass: {:.3} ms", inner_p50 / 3.0);
    }

    // Summary
    println!("\n  ── Summary ──");
    println!("    exp24: {:.0} Mkeys/s ({:.3} ms)", mkeys, fp50);
    println!("    vs exp23 (2410 Mk/s): {:.2}x", mkeys / 2410.0);
    println!("    vs exp16 (3003 Mk/s): {:.2}x", mkeys / 3003.0);
    println!("    Target: 5000+ Mk/s");
    println!(
        "    SLC advantage: {} batches × ~{:.0} MB each",
        batches.len(),
        SLC_BUDGET as f64 / 1e6
    );
}
