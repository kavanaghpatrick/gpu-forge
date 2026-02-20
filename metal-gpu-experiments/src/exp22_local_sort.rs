//! Experiment 22: Work-Queue Local Sort (KB #3460 Approach 2)
//!
//! Architecture: 1 MSD scatter (bits 24-31) + per-bucket local sort (bits 0-23)
//! Local sort: 3-pass 8-bit counting sort ENTIRELY in TG memory (2048-element tiles)
//! Only 2 global data passes (MSD read+write, local sort read+write)
//!
//! vs exp17: 4 global passes (1 MSD + 3 LSD through full 16M array)
//! vs exp21: proved pre-sort scatter is dead (4% slower) — random scatter is fine

use crate::metal_ctx::*;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};
use std::ptr::NonNull;

const WARMUP: usize = 5;
const RUNS: usize = 30;
const TG_SIZE: usize = 256;
const MSD_TILE: usize = 4096;
const LOCAL_TILE: usize = 1024;
const NUM_BINS: usize = 256;

type Pso =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

// ═══════════════════════════════════════════════════════════════════
// Structs matching Metal layout
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Clone, Copy)]
struct E22Params {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    pass: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct E22InnerParams {
    total_inner_tiles: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct E22BucketDesc {
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

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// Pipeline encoding
// ═══════════════════════════════════════════════════════════════════

/// Encode the full sort pipeline (or MSD-only if skip_local=true).
/// Flow: buf_a (input) → MSD scatter → buf_b → local sort → buf_a (sorted output)
fn encode_pipeline(
    cmd: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
    n: usize,
    buf_a: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_b: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_hist: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_descs: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_status: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_ctr: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    pso_msd_hist: &Pso,
    pso_bucket_descs: &Pso,
    pso_global_pfx: &Pso,
    pso_zero_status: &Pso,
    pso_partition: &Pso,
    pso_local_sort: &Pso,
    skip_local: bool,
) {
    let mt = n.div_ceil(MSD_TILE);
    // Upper bound for total inner tiles: exact + 256 (one extra per bucket for ceiling)
    let total_it = (n.div_ceil(LOCAL_TILE) + NUM_BINS) as u32;

    let msd_p = E22Params {
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
    let inn_p = E22InnerParams {
        total_inner_tiles: total_it,
    };

    let tg = MTLSize { width: TG_SIZE, height: 1, depth: 1 };

    // Zero histogram (CPU-side, visible to GPU after commit)
    unsafe {
        std::ptr::write_bytes(buf_hist.contents().as_ptr() as *mut u8, 0, NUM_BINS * 4);
    }

    let enc = cmd.computeCommandEncoder().unwrap();

    // 1) MSD histogram: buf_a → buf_hist
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

    // 2) Bucket descriptors: buf_hist (counts) → buf_descs
    enc.setComputePipelineState(pso_bucket_descs);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_hist), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_descs), 0, 1);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: 1, height: 1, depth: 1 },
        tg,
    );

    // 3) Global prefix: buf_hist in-place (counts → exclusive prefix sums)
    enc.setComputePipelineState(pso_global_pfx);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_hist), 0, 0);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: 1, height: 1, depth: 1 },
        tg,
    );

    // 4) Zero tile status + counter
    let zt = (mt * NUM_BINS).div_ceil(TG_SIZE);
    enc.setComputePipelineState(pso_zero_status);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_status), 0, 0);
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

    // 5) MSD partition (decoupled lookback): buf_a → buf_b
    enc.setComputePipelineState(pso_partition);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_status), 0, 2);
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

    // 6) Local sort: buf_b → buf_a (3-pass counting sort in TG memory)
    if !skip_local {
        enc.setComputePipelineState(pso_local_sort);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_b), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_a), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_descs), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&inn_p as *const _ as *mut _).unwrap(),
                std::mem::size_of_val(&inn_p),
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: total_it as usize, height: 1, depth: 1 },
            tg,
        );
    }

    enc.endEncoding();
}

// ═══════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "═".repeat(60));
    println!("  EXP 22: Work-Queue Local Sort (KB #3460)");
    println!("  MSD(24:31) scatter → per-bucket local sort(0:23)");
    println!("  2 global passes vs exp17's 4 global passes");
    println!("{}", "═".repeat(60));

    let pso_msd_hist = ctx.make_pipeline("exp22_msd_histogram");
    let pso_bucket_descs = ctx.make_pipeline("exp22_compute_bucket_descs");
    let pso_global_pfx = ctx.make_pipeline("exp22_global_prefix");
    let pso_zero_status = ctx.make_pipeline("exp16_zero_status");
    let pso_partition = ctx.make_pipeline("exp16_partition");
    let pso_local_sort = ctx.make_pipeline("exp22_local_sort");

    // ── Correctness tests ──
    // NOTE: Local sort only works when all MSD buckets fit in one tile (≤1024 elements).
    // At n=256*1024=262144 with uniform MSD distribution, avg bucket=1024, some overflow.
    // Safe sizes: n ≤ ~200K (avg bucket ~781, max ~950 with 3σ margin).
    for &sz in &[2048usize, 8192, 62_500, 200_000] {
        print!("  Correctness @ {} ... ", sz);

        let data = gen_random(sz);
        let mut expected = data.clone();
        expected.sort();

        let ba = alloc_buffer_with_data(&ctx.device, &data);
        let bb = alloc_buffer(&ctx.device, sz * 4);
        let bh = alloc_buffer(&ctx.device, NUM_BINS * 4);
        let bd = alloc_buffer(&ctx.device, NUM_BINS * std::mem::size_of::<E22BucketDesc>());
        let bs = alloc_buffer(&ctx.device, sz.div_ceil(MSD_TILE) * NUM_BINS * 4);
        let bc = alloc_buffer(&ctx.device, 4);

        let cmd = ctx.command_buffer();
        encode_pipeline(
            &cmd, sz, &ba, &bb, &bh, &bd, &bs, &bc,
            &pso_msd_hist, &pso_bucket_descs, &pso_global_pfx,
            &pso_zero_status, &pso_partition, &pso_local_sort, false,
        );
        cmd.commit();
        cmd.waitUntilCompleted();

        let result: Vec<u32> = unsafe { read_buffer_slice(&ba, sz) };
        let mm = result.iter().zip(&expected).filter(|(a, b)| a != b).count();
        if mm > 0 {
            println!("FAIL ({} mismatches)", mm);
            // Check max bucket size
            let descs: Vec<E22BucketDesc> = unsafe { read_buffer_slice(&bd, NUM_BINS) };
            let max_bucket = descs.iter().map(|d| d.count).max().unwrap_or(0);
            println!("    Max bucket size: {} (tile={})", max_bucket, LOCAL_TILE);
            if max_bucket as usize > LOCAL_TILE {
                println!("    CAUSE: Bucket exceeds tile size — multi-tile merge needed");
            }
            // Show first 10 mismatches
            let mut shown = 0;
            for i in 0..sz {
                if result[i] != expected[i] {
                    println!(
                        "    [{}] got=0x{:08X} exp=0x{:08X}",
                        i, result[i], expected[i]
                    );
                    shown += 1;
                    if shown >= 10 { break; }
                }
            }
            return;
        }
        println!("PASS");
    }

    // ── Benchmark @ 16M ──
    let n = 16_000_000usize;
    println!("\n  ── Benchmark @ 16M ──\n");

    let data = gen_random(n);
    let mut expected = data.clone();
    expected.sort();

    let bi = alloc_buffer_with_data(&ctx.device, &data);
    let ba = alloc_buffer(&ctx.device, n * 4);
    let bb = alloc_buffer(&ctx.device, n * 4);
    let bh = alloc_buffer(&ctx.device, NUM_BINS * 4);
    let bd = alloc_buffer(&ctx.device, NUM_BINS * std::mem::size_of::<E22BucketDesc>());
    let bs = alloc_buffer(&ctx.device, n.div_ceil(MSD_TILE) * NUM_BINS * 4);
    let bc = alloc_buffer(&ctx.device, 4);

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
            &cmd, n, &ba, &bb, &bh, &bd, &bs, &bc,
            &pso_msd_hist, &pso_bucket_descs, &pso_global_pfx,
            &pso_zero_status, &pso_partition, &pso_local_sort, false,
        );
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= WARMUP {
            full_times.push(gpu_elapsed_ms(&cmd));
        }
    }

    // Verify correctness at 16M (will fail due to multi-tile buckets)
    let result: Vec<u32> = unsafe { read_buffer_slice(&ba, n) };
    let mm = result.iter().zip(&expected).filter(|(a, b)| a != b).count();
    // Check intra-tile correctness: within each tile, elements should be sorted
    let descs_chk: Vec<E22BucketDesc> = unsafe { read_buffer_slice(&bd, NUM_BINS) };
    let mut intra_tile_ok = true;
    let mut intra_tile_fails = 0u32;
    for d in &descs_chk {
        for t in 0..d.tile_count {
            let start = d.offset as usize + t as usize * LOCAL_TILE;
            let end = std::cmp::min(start + LOCAL_TILE, (d.offset + d.count) as usize);
            let tile = &result[start..end];
            if !tile.windows(2).all(|w| w[0] <= w[1]) {
                intra_tile_ok = false;
                intra_tile_fails += 1;
            }
        }
    }

    // MSD-only benchmark (for phase timing)
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
            &cmd, n, &ba, &bb, &bh, &bd, &bs, &bc,
            &pso_msd_hist, &pso_bucket_descs, &pso_global_pfx,
            &pso_zero_status, &pso_partition, &pso_local_sort, true,
        );
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= WARMUP {
            msd_times.push(gpu_elapsed_ms(&cmd));
        }
    }

    // Results
    full_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    msd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let fp50 = percentile(&full_times, 50.0);
    let fp5 = percentile(&full_times, 5.0);
    let fp95 = percentile(&full_times, 95.0);
    let mp50 = percentile(&msd_times, 50.0);
    let lp50 = fp50 - mp50; // estimated local sort time

    let mkeys = n as f64 / fp50 / 1000.0;
    // 2 global passes: MSD(r+w) + local(r+w) = 4 × N × 4 bytes
    let bw = (n as f64 * 4.0 * 4.0) / fp50 / 1e6;

    if mm == 0 {
        println!("    Correctness: PASS (full sort)");
    } else {
        println!("    Correctness: {} mismatches (expected — multi-tile buckets)", mm);
        println!("    Intra-tile:  {} (tiles with internal sort errors: {})",
            if intra_tile_ok { "PASS" } else { "FAIL" }, intra_tile_fails);
    }
    println!(
        "    Full:  {:.3} ms | {:.0} Mkeys/s | {:.0} GB/s eff",
        fp50, mkeys, bw
    );
    println!(
        "    [p5={:.3} p95={:.3} spread={:.0}%]",
        fp5,
        fp95,
        (fp95 - fp5) / fp5 * 100.0
    );
    println!(
        "    MSD:   {:.3} ms | {:.0} GB/s",
        mp50,
        (n as f64 * 4.0 * 2.0) / mp50 / 1e6
    );
    if lp50 > 0.0 {
        println!(
            "    Local: {:.3} ms | {:.0} GB/s (est)",
            lp50,
            (n as f64 * 4.0 * 2.0) / lp50 / 1e6
        );
    }

    // Bucket statistics
    let descs: Vec<E22BucketDesc> = unsafe { read_buffer_slice(&bd, NUM_BINS) };
    let total_tiles: u32 = descs.iter().map(|d| d.tile_count).sum();
    let avg_count = n as f64 / NUM_BINS as f64;
    let avg_tiles = total_tiles as f64 / NUM_BINS as f64;
    println!(
        "    Buckets: {} tiles total | avg {:.0} elements ({:.1} tiles) per bucket",
        total_tiles, avg_count, avg_tiles
    );

    // Summary
    println!("\n  ── Summary ──");
    println!("    exp22: {:.0} Mkeys/s ({:.3} ms)", mkeys, fp50);
    println!("    vs exp16 (3003 Mk/s): {:.2}x", mkeys / 3003.0);
    println!("    vs exp17 (3461 Mk/s): {:.2}x", mkeys / 3461.0);
    println!(
        "    vs exp21-random (3513 Mk/s): {:.2}x",
        mkeys / 3513.0
    );
    println!("    Target: 5000+ Mk/s");
}
