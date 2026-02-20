//! Experiment 25: Fence-Free Radix Sort
//!
//! Eliminates ALL device-scope fences from the partition kernel.
//! Instead of decoupled lookback, uses precomputed tile prefix sums.
//!
//! Per pass (4 dispatches):
//!   1. Tile histogram: read source → per-tile 256-bin counts
//!   2. Tile prefix: serial scan across tiles → per-tile offsets + global hist
//!   3. Global prefix: 256 counts → exclusive prefix sums
//!   4. Scatter: re-read source, rank within tile, use precomputed offsets
//!
//! Total: 16 dispatches (4 per pass × 4 passes) in a single encoder.
//! Zero device-scope fences in any kernel.
//!
//! Hypothesis: fences cost ~0.6ms/pass in exp16 → removing = faster partition.

use crate::metal_ctx::*;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};
use std::ptr::NonNull;

const WARMUP: usize = 5;
const RUNS: usize = 30;
const TG_SIZE: usize = 256;
const TILE_SIZE: usize = 4096;
const NUM_BINS: usize = 256;
const NUM_PASSES: usize = 4;

type Pso =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

#[repr(C)]
#[derive(Clone, Copy)]
struct E25Params {
    element_count: u32,
    num_tiles: u32,
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

/// Encode full fence-free radix sort pipeline in a SINGLE encoder.
fn encode_pipeline(
    cmd: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
    n: usize,
    buf_a: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_b: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_tile_hists: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_tile_prefix: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_global_hist: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    pso_tile_hist: &Pso,
    pso_tile_prefix: &Pso,
    pso_global_prefix: &Pso,
    pso_scatter: &Pso,
) {
    let nt = n.div_ceil(TILE_SIZE);

    let tg = MTLSize {
        width: TG_SIZE,
        height: 1,
        depth: 1,
    };

    let enc = cmd.computeCommandEncoder().unwrap();

    // 4 passes: 0→shift0, 1→shift8, 2→shift16, 3→shift24
    // Pass 0: buf_a → buf_b
    // Pass 1: buf_b → buf_a
    // Pass 2: buf_a → buf_b
    // Pass 3: buf_b → buf_a
    // Result in buf_a (even number of passes).
    for pass in 0..NUM_PASSES as u32 {
        let shift = pass * 8;
        let (src_buf, dst_buf) = if pass % 2 == 0 {
            (buf_a as &_, buf_b as &_)
        } else {
            (buf_b as &_, buf_a as &_)
        };

        let pp = E25Params {
            element_count: n as u32,
            num_tiles: nt as u32,
            shift,
            pass,
        };

        // 1) Tile histogram — nt TGs, reads current source
        enc.setComputePipelineState(pso_tile_hist);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hists), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&pp as *const _ as *mut _).unwrap(),
                std::mem::size_of_val(&pp),
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: nt, height: 1, depth: 1 },
            tg,
        );

        // 2) Tile prefix sum — 256 TGs (one per bin)
        enc.setComputePipelineState(pso_tile_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hists), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_tile_prefix), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&pp as *const _ as *mut _).unwrap(),
                std::mem::size_of_val(&pp),
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: NUM_BINS, height: 1, depth: 1 },
            tg,
        );

        // 3) Global prefix sum — 1 TG
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist), 0, 0);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            tg,
        );

        // 4) Scatter — nt TGs, NO fences
        enc.setComputePipelineState(pso_scatter);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_prefix), 0, 2);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&pp as *const _ as *mut _).unwrap(),
                std::mem::size_of_val(&pp),
                4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: nt, height: 1, depth: 1 },
            tg,
        );
    }

    enc.endEncoding();
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "═".repeat(60));
    println!("  EXP 25: Fence-Free Radix Sort");
    println!("  Precomputed tile prefix — no decoupled lookback, no fences");
    println!("  Hypothesis: fences cost ~0.6ms/pass → removing = faster");
    println!("{}", "═".repeat(60));

    let pso_tile_hist = ctx.make_pipeline("exp25_tile_histogram");
    let pso_tile_prefix = ctx.make_pipeline("exp25_tile_prefix");
    let pso_global_prefix = ctx.make_pipeline("exp25_global_prefix");
    let pso_scatter = ctx.make_pipeline("exp25_scatter");

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

        let nt = sz.div_ceil(TILE_SIZE);

        let ba = alloc_buffer_with_data(&ctx.device, &data);
        let bb = alloc_buffer(&ctx.device, sz * 4);
        let bth = alloc_buffer(&ctx.device, nt * NUM_BINS * 4);
        let btp = alloc_buffer(&ctx.device, nt * NUM_BINS * 4);
        let bgh = alloc_buffer(&ctx.device, NUM_BINS * 4);

        let cmd = ctx.command_buffer();
        encode_pipeline(
            &cmd, sz, &ba, &bb, &bth, &btp, &bgh,
            &pso_tile_hist, &pso_tile_prefix, &pso_global_prefix, &pso_scatter,
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
                    if shown >= 10 {
                        break;
                    }
                }
            }
            return;
        }
        println!("PASS");
    }

    // ── Benchmark @ 16M ──
    println!("\n  ── Benchmark @ 16M ──\n");

    let n = 16_000_000usize;
    let data = gen_random(n);
    let nt = n.div_ceil(TILE_SIZE);

    let bi = alloc_buffer_with_data(&ctx.device, &data);
    let ba = alloc_buffer(&ctx.device, n * 4);
    let bb = alloc_buffer(&ctx.device, n * 4);
    let bth = alloc_buffer(&ctx.device, nt * NUM_BINS * 4);
    let btp = alloc_buffer(&ctx.device, nt * NUM_BINS * 4);
    let bgh = alloc_buffer(&ctx.device, NUM_BINS * 4);

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
            &cmd, n, &ba, &bb, &bth, &btp, &bgh,
            &pso_tile_hist, &pso_tile_prefix, &pso_global_prefix, &pso_scatter,
        );
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= WARMUP {
            full_times.push(gpu_elapsed_ms(&cmd));
        }
    }

    full_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let fp50 = percentile(&full_times, 50.0);
    let fp5 = percentile(&full_times, 5.0);
    let fp95 = percentile(&full_times, 95.0);

    let mkeys = n as f64 / fp50 / 1000.0;
    // Total bandwidth: per pass = 64MB (hist read) + 4MB (hist write) + 8MB (prefix r/w)
    //                  + 64MB (scatter read) + 64MB (scatter write) = 204MB
    // × 4 passes = 816MB
    let bw = 816.0 / fp50; // GB/s

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
        &cmd, n, &ba, &bb, &bth, &btp, &bgh,
        &pso_tile_hist, &pso_tile_prefix, &pso_global_prefix, &pso_scatter,
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
                if shown >= 10 {
                    break;
                }
            }
        }
    }

    let per_pass = fp50 / NUM_PASSES as f64;
    println!(
        "    Full:  {:.3} ms | {:.0} Mkeys/s | {:.0} GB/s eff",
        fp50, mkeys, bw
    );
    println!(
        "    [p5={:.3} p95={:.3} spread={:.0}%]",
        fp5, fp95, (fp95 - fp5) / fp5 * 100.0
    );
    println!(
        "    Per pass: {:.3} ms (hist+prefix+scatter, ZERO fences)",
        per_pass
    );
    println!(
        "    Per pass BW: {:.0} GB/s (read 64+4MB, write 64+4MB+1KB)",
        (n as f64 * 4.0 * 2.0 + nt as f64 * NUM_BINS as f64 * 4.0 * 2.0 + NUM_BINS as f64 * 4.0)
            / per_pass / 1e6
    );

    // Summary
    println!("\n  ── Summary ──");
    println!("    exp25: {:.0} Mkeys/s ({:.3} ms)", mkeys, fp50);
    println!("    vs exp16 (3003 Mk/s): {:.2}x", mkeys / 3003.0);
    println!("    Target: 5000+ Mk/s");
    println!("    Approach: 4 dispatches/pass, ZERO device-scope fences");
    if mkeys > 3003.0 {
        println!("    FENCE OVERHEAD CONFIRMED: {:.0} Mk/s improvement!", mkeys - 3003.0);
    } else {
        println!("    Fence overhead hypothesis: DISPROVEN (fences not the bottleneck)");
    }
}
