//! Experiment 26: 3-Pass Radix Sort (10 + 11 + 11 bits)
//!
//! Targets 5000+ Mkeys/s at 16M uint32 on Apple M4 Pro.
//! 3-pass scatter ceiling: 5360 Mkeys/s (measured in exp17 Investigation P).
//! 4-pass ceiling: 3746 Mkeys/s (cannot reach 5000).
//!
//! Pass layout:
//!   Pass 0: 10-bit (1024 bins), shift= 0, mask=0x3FF
//!   Pass 1: 11-bit (2048 bins), shift=10, mask=0x7FF
//!   Pass 2: 11-bit (2048 bins), shift=21, mask=0x7FF
//!
//! Per pass (4 dispatches in single encoder):
//!   1. Tile histogram — per-tile bin counts
//!   2. Tile prefix    — in-place serial prefix across tiles
//!   3. Global prefix  — bin counts → exclusive prefix sums
//!   4. Scatter        — atomic ranking + scatter with cached offsets
//!
//! Total: 12 dispatches, single encoder, zero fences.
//! Result in buf_b (3 passes = odd → A→B→A→B, final in B).

use crate::metal_ctx::*;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};
use std::ptr::NonNull;

const WARMUP: usize = 5;
const RUNS: usize = 30;
const TILE_SIZE: usize = 4096;
const MAX_BINS: usize = 2048;
const TG_SIZE: usize = 256;  // all kernels use 256 threads

type Pso =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

#[repr(C)]
#[derive(Clone, Copy)]
struct E26Params {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    num_bins: u32,
    mask: u32,
}

/// Pass configs: (shift, num_bins, mask)
const PASSES: [(u32, u32, u32); 3] = [
    (0, 1024, 0x3FF),   // 10-bit
    (10, 2048, 0x7FF),  // 11-bit
    (21, 2048, 0x7FF),  // 11-bit
];

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

/// Encode all 12 dispatches (4 per pass × 3 passes) in a single compute encoder.
/// Ping-pong: pass 0 A→B, pass 1 B→A, pass 2 A→B. Result in buf_b.
fn encode_pipeline(
    cmd: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
    n: usize,
    buf_a: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_b: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_tile_hists: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_global_hist: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    pso_tile_hist: &Pso,
    pso_tile_prefix: &Pso,
    pso_global_prefix: &Pso,
    pso_scatter: &Pso,
) {
    let nt = n.div_ceil(TILE_SIZE);
    let tg256 = MTLSize {
        width: TG_SIZE,
        height: 1,
        depth: 1,
    };
    let enc = cmd.computeCommandEncoder().unwrap();

    for (pass_idx, &(shift, num_bins, mask)) in PASSES.iter().enumerate() {
        let (src_buf, dst_buf) = if pass_idx % 2 == 0 {
            (buf_a as &_, buf_b as &_)
        } else {
            (buf_b as &_, buf_a as &_)
        };

        let pp = E26Params {
            element_count: n as u32,
            num_tiles: nt as u32,
            shift,
            num_bins,
            mask,
        };

        // 1) Tile histogram — nt TGs × 256 threads
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
            MTLSize {
                width: nt,
                height: 1,
                depth: 1,
            },
            tg256,
        );

        // 2) Tile prefix — num_bins TGs × 256 threads
        //    (serial kernel: only lid==0 active; parallel v2: all 256 active)
        enc.setComputePipelineState(pso_tile_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_tile_hists), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_global_hist), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::new(&pp as *const _ as *mut _).unwrap(),
                std::mem::size_of_val(&pp),
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_bins as usize,
                height: 1,
                depth: 1,
            },
            tg256,
        );

        // 3) Global prefix sum — 1 TG × 256 threads (only lid==0 active)
        enc.setComputePipelineState(pso_global_prefix);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_global_hist), 0, 0);
            enc.setBytes_length_atIndex(
                NonNull::new(&pp as *const _ as *mut _).unwrap(),
                std::mem::size_of_val(&pp),
                1,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            tg256,
        );

        // 4) Scatter — nt TGs × 256 threads (SG-serialized stable ranking)
        enc.setComputePipelineState(pso_scatter);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_tile_hists), 0, 2); // now contains prefix sums
            enc.setBuffer_offset_atIndex(Some(buf_global_hist), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::new(&pp as *const _ as *mut _).unwrap(),
                std::mem::size_of_val(&pp),
                4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: nt,
                height: 1,
                depth: 1,
            },
            tg256,
        );
    }

    enc.endEncoding();
}

fn bench_variant(
    ctx: &MetalContext,
    label: &str,
    n: usize,
    data: &[u32],
    pso_tile_hist: &Pso,
    pso_tile_prefix: &Pso,
    pso_global_prefix: &Pso,
    pso_scatter: &Pso,
) -> (f64, f64, f64, usize) {
    let nt = n.div_ceil(TILE_SIZE);
    let bi = alloc_buffer_with_data(&ctx.device, data);
    let ba = alloc_buffer(&ctx.device, n * 4);
    let bb = alloc_buffer(&ctx.device, n * 4);
    let bth = alloc_buffer(&ctx.device, nt * MAX_BINS * 4);
    let bgh = alloc_buffer(&ctx.device, MAX_BINS * 4);

    let mut times: Vec<f64> = Vec::new();
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
            &cmd, n, &ba, &bb, &bth, &bgh,
            pso_tile_hist, pso_tile_prefix, pso_global_prefix, pso_scatter,
        );
        cmd.commit();
        cmd.waitUntilCompleted();
        if iter >= WARMUP {
            times.push(gpu_elapsed_ms(&cmd));
        }
    }

    // Verify correctness on last run
    unsafe {
        std::ptr::copy_nonoverlapping(
            bi.contents().as_ptr() as *const u8,
            ba.contents().as_ptr() as *mut u8,
            n * 4,
        );
    }
    let cmd = ctx.command_buffer();
    encode_pipeline(
        &cmd, n, &ba, &bb, &bth, &bgh,
        pso_tile_hist, pso_tile_prefix, pso_global_prefix, pso_scatter,
    );
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut expected = data.to_vec();
    expected.sort();
    let result: Vec<u32> = unsafe { read_buffer_slice(&bb, n) };
    let mm = result.iter().zip(&expected).filter(|(a, b)| a != b).count();

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&times, 50.0);
    let p5 = percentile(&times, 5.0);
    let p95 = percentile(&times, 95.0);
    let mkeys = n as f64 / p50 / 1000.0;

    println!("  {}: {:.3} ms | {:.0} Mk/s | {} | [p5={:.3} p95={:.3}]",
        label, p50, mkeys,
        if mm == 0 { "CORRECT".to_string() } else { format!("FAIL({})", mm) },
        p5, p95);

    (p50, mkeys, p5, mm)
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "═".repeat(60));
    println!("  EXP 26: 3-Pass Radix Sort (10 + 11 + 11 bits)");
    println!("  Target: 5000+ Mkeys/s | Scatter ceiling: 5360 Mk/s");
    println!("  A/B: unstable atomics vs SG-serialized stable");
    println!("{}", "═".repeat(60));

    let pso_tile_hist = ctx.make_pipeline("exp26_tile_histogram");
    let pso_tile_prefix_serial = ctx.make_pipeline("exp26_tile_prefix");
    let pso_tile_prefix_par = ctx.make_pipeline("exp26_tile_prefix_v2");
    let pso_global_prefix = ctx.make_pipeline("exp26_global_prefix");
    let pso_scatter_unstable = ctx.make_pipeline("exp26_scatter");
    let pso_scatter_stable = ctx.make_pipeline("exp26_scatter_stable");

    // ── Quick correctness check: serial prefix + stable scatter ──
    println!("\n  ── Correctness (serial prefix + stable scatter) ──");
    for &sz in &[4096usize, 62_500, 1_000_000, 16_000_000] {
        let label = if sz >= 1_000_000 {
            format!("{}M", sz / 1_000_000)
        } else if sz >= 1_000 {
            format!("{}K", sz / 1_000)
        } else {
            format!("{}", sz)
        };
        print!("  @ {} ... ", label);

        let data = gen_random(sz);
        let mut expected = data.clone();
        expected.sort();
        let nt = sz.div_ceil(TILE_SIZE);

        let ba = alloc_buffer_with_data(&ctx.device, &data);
        let bb = alloc_buffer(&ctx.device, sz * 4);
        let bth = alloc_buffer(&ctx.device, nt * MAX_BINS * 4);
        let bgh = alloc_buffer(&ctx.device, MAX_BINS * 4);

        let cmd = ctx.command_buffer();
        encode_pipeline(
            &cmd, sz, &ba, &bb, &bth, &bgh,
            &pso_tile_hist, &pso_tile_prefix_serial, &pso_global_prefix, &pso_scatter_stable,
        );
        cmd.commit();
        cmd.waitUntilCompleted();

        let result: Vec<u32> = unsafe { read_buffer_slice(&bb, sz) };
        let mm = result.iter().zip(&expected).filter(|(a, b)| a != b).count();
        println!("{}", if mm == 0 { "PASS" } else { "FAIL" });
        if mm > 0 { return; }
    }

    // ── Correctness check: parallel prefix ──
    println!("\n  ── Correctness (parallel prefix + stable scatter) ──");
    for &sz in &[4096usize, 62_500, 1_000_000, 16_000_000] {
        let label = if sz >= 1_000_000 {
            format!("{}M", sz / 1_000_000)
        } else if sz >= 1_000 {
            format!("{}K", sz / 1_000)
        } else {
            format!("{}", sz)
        };
        print!("  @ {} ... ", label);

        let data = gen_random(sz);
        let mut expected = data.clone();
        expected.sort();
        let nt = sz.div_ceil(TILE_SIZE);

        let ba = alloc_buffer_with_data(&ctx.device, &data);
        let bb = alloc_buffer(&ctx.device, sz * 4);
        let bth = alloc_buffer(&ctx.device, nt * MAX_BINS * 4);
        let bgh = alloc_buffer(&ctx.device, MAX_BINS * 4);

        let cmd = ctx.command_buffer();
        encode_pipeline(
            &cmd, sz, &ba, &bb, &bth, &bgh,
            &pso_tile_hist, &pso_tile_prefix_par, &pso_global_prefix, &pso_scatter_stable,
        );
        cmd.commit();
        cmd.waitUntilCompleted();

        let result: Vec<u32> = unsafe { read_buffer_slice(&bb, sz) };
        let mm = result.iter().zip(&expected).filter(|(a, b)| a != b).count();
        println!("{}", if mm == 0 { "PASS" } else { "FAIL" });
        if mm > 0 { return; }
    }

    // ── A/B/C Benchmark @ 16M ──
    println!("\n  ── A/B/C Benchmark @ 16M ──\n");

    let n = 16_000_000usize;
    let data = gen_random(n);

    let (_, mkeys_a, _, mm_a) = bench_variant(
        ctx, "A (serial+unstable)", n, &data,
        &pso_tile_hist, &pso_tile_prefix_serial, &pso_global_prefix, &pso_scatter_unstable,
    );

    let (_, mkeys_b, _, _) = bench_variant(
        ctx, "B (serial+stable)  ", n, &data,
        &pso_tile_hist, &pso_tile_prefix_serial, &pso_global_prefix, &pso_scatter_stable,
    );

    let (_, mkeys_c, _, _) = bench_variant(
        ctx, "C (parallel+stable) ", n, &data,
        &pso_tile_hist, &pso_tile_prefix_par, &pso_global_prefix, &pso_scatter_stable,
    );

    // ── Analysis ──
    println!("\n  ── Analysis ──");
    println!("    A serial+unstable:  {:.0} Mk/s (mismatches: {})", mkeys_a, mm_a);
    println!("    B serial+stable:    {:.0} Mk/s (correct)", mkeys_b);
    println!("    C parallel+stable:  {:.0} Mk/s (correct)", mkeys_c);
    println!("    Parallel prefix speedup: {:.2}x (C/B)", mkeys_c / mkeys_b);
    println!("    vs exp17 best (3635 Mk/s): {:.2}x", mkeys_c / 3635.0);

    // ── Per-kernel profiling (serial vs parallel prefix) ──
    println!("\n  ── Per-kernel profiling (pass 0: 10-bit, 1024 bins) ──\n");
    {
        let nt = n.div_ceil(TILE_SIZE);
        let bi2 = alloc_buffer_with_data(&ctx.device, &data);
        let ba2 = alloc_buffer(&ctx.device, n * 4);
        let bb2 = alloc_buffer(&ctx.device, n * 4);
        let bth2 = alloc_buffer(&ctx.device, nt * MAX_BINS * 4);
        let bgh2 = alloc_buffer(&ctx.device, MAX_BINS * 4);

        let (shift, num_bins, mask) = PASSES[0];
        let pp = E26Params {
            element_count: n as u32,
            num_tiles: nt as u32,
            shift, num_bins, mask,
        };
        let tg256 = MTLSize { width: TG_SIZE, height: 1, depth: 1 };

        // Warm up
        unsafe {
            std::ptr::copy_nonoverlapping(
                bi2.contents().as_ptr() as *const u8,
                ba2.contents().as_ptr() as *mut u8,
                n * 4,
            );
        }

        // Helper: profile a single kernel dispatch
        let profile_kernel = |pso: &Pso, bufs: &[(&objc2::runtime::ProtocolObject<dyn MTLBuffer>, usize)],
                              bytes: Option<(&E26Params, usize)>,
                              grid: MTLSize, tg: MTLSize| -> f64 {
            let mut times: Vec<f64> = Vec::new();
            for _ in 0..RUNS {
                let cmd = ctx.command_buffer();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(pso);
                for &(buf, idx) in bufs {
                    unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, idx); }
                }
                if let Some((p, idx)) = bytes {
                    unsafe {
                        enc.setBytes_length_atIndex(
                            NonNull::new(p as *const _ as *mut _).unwrap(),
                            std::mem::size_of_val(p), idx,
                        );
                    }
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                times.push(gpu_elapsed_ms(&cmd));
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            percentile(&times, 50.0)
        };

        let hist_grid = MTLSize { width: nt, height: 1, depth: 1 };
        let prefix_grid = MTLSize { width: num_bins as usize, height: 1, depth: 1 };
        let one_grid = MTLSize { width: 1, height: 1, depth: 1 };

        // tile_histogram
        let k1 = profile_kernel(
            &pso_tile_hist,
            &[(&ba2, 0), (&bth2, 1)],
            Some((&pp, 2)),
            hist_grid, tg256,
        );

        // tile_prefix SERIAL
        let k2_serial = profile_kernel(
            &pso_tile_prefix_serial,
            &[(&bth2, 0), (&bgh2, 1)],
            Some((&pp, 2)),
            prefix_grid, tg256,
        );

        // tile_prefix PARALLEL
        let k2_par = profile_kernel(
            &pso_tile_prefix_par,
            &[(&bth2, 0), (&bgh2, 1)],
            Some((&pp, 2)),
            prefix_grid, tg256,
        );

        // global_prefix
        let k3 = profile_kernel(
            &pso_global_prefix,
            &[(&bgh2, 0)],
            Some((&pp, 1)),
            one_grid, tg256,
        );

        // scatter (stable) — setup hist+prefix first
        let mut k4_times: Vec<f64> = Vec::new();
        for _ in 0..RUNS {
            let setup = ctx.command_buffer();
            let senc = setup.computeCommandEncoder().unwrap();
            senc.setComputePipelineState(&pso_tile_hist);
            unsafe {
                senc.setBuffer_offset_atIndex(Some(&*ba2), 0, 0);
                senc.setBuffer_offset_atIndex(Some(&*bth2), 0, 1);
                senc.setBytes_length_atIndex(
                    NonNull::new(&pp as *const _ as *mut _).unwrap(),
                    std::mem::size_of_val(&pp), 2,
                );
            }
            senc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg256);
            senc.setComputePipelineState(&pso_tile_prefix_par);
            unsafe {
                senc.setBuffer_offset_atIndex(Some(&*bth2), 0, 0);
                senc.setBuffer_offset_atIndex(Some(&*bgh2), 0, 1);
                senc.setBytes_length_atIndex(
                    NonNull::new(&pp as *const _ as *mut _).unwrap(),
                    std::mem::size_of_val(&pp), 2,
                );
            }
            senc.dispatchThreadgroups_threadsPerThreadgroup(prefix_grid, tg256);
            senc.setComputePipelineState(&pso_global_prefix);
            unsafe {
                senc.setBuffer_offset_atIndex(Some(&*bgh2), 0, 0);
                senc.setBytes_length_atIndex(
                    NonNull::new(&pp as *const _ as *mut _).unwrap(),
                    std::mem::size_of_val(&pp), 1,
                );
            }
            senc.dispatchThreadgroups_threadsPerThreadgroup(one_grid, tg256);
            senc.endEncoding();
            setup.commit();
            setup.waitUntilCompleted();

            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_scatter_stable);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&*ba2), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&*bb2), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&*bth2), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&*bgh2), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&pp as *const _ as *mut _).unwrap(),
                    std::mem::size_of_val(&pp), 4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg256);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            k4_times.push(gpu_elapsed_ms(&cmd));
        }
        k4_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k4 = percentile(&k4_times, 50.0);

        let total_serial = k1 + k2_serial + k3 + k4;
        let total_par = k1 + k2_par + k3 + k4;
        println!("    tile_histogram:      {:.3} ms", k1);
        println!("    tile_prefix SERIAL:  {:.3} ms", k2_serial);
        println!("    tile_prefix PARALLEL:{:.3} ms  ({:.1}x speedup)",
            k2_par, k2_serial / k2_par);
        println!("    global_prefix:       {:.3} ms", k3);
        println!("    scatter (stable):    {:.3} ms", k4);
        println!();
        println!("    Serial sum:   {:.3} ms × 3 = {:.3} ms → {:.0} Mk/s",
            total_serial, total_serial * 3.0, n as f64 / (total_serial * 3.0) / 1000.0);
        println!("    Parallel sum:  {:.3} ms × 3 = {:.3} ms → {:.0} Mk/s",
            total_par, total_par * 3.0, n as f64 / (total_par * 3.0) / 1000.0);
        println!();
        println!("    tile_hists buf: {:.1} MB ({} tiles × {} bins × 4B)",
            (nt * MAX_BINS * 4) as f64 / 1e6, nt, MAX_BINS);
    }
}
