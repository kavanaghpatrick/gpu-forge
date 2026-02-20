//! Experiment 18: Monster 3-Pass Radix Sort (5000+ Mkeys/s target)
//!
//! Config: 11+11+10 bits (2048/2048/1024 bins)
//! Architecture: Onesweep-style decoupled lookback, TG-wide atomics,
//! non-persistent dispatch, 4096-element tiles, 256 threads/TG.
//!
//! Key insight: at 2048 bins, TG-wide atomic contention = 4096/2048 = 2,
//! identical to per-SG contention at 256 bins (512/256 = 2).

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
use std::ptr::NonNull;

const TILE_SIZE: usize = 4096;
const THREADS_PER_TG: usize = 256;
const WARMUP: usize = 5;
const RUNS: usize = 30;

const BINS_P0: usize = 2048; // 11-bit
const BINS_P1: usize = 2048; // 11-bit
const BINS_P2: usize = 1024; // 10-bit
const TOTAL_BINS: usize = BINS_P0 + BINS_P1 + BINS_P2; // 5120

type Pso = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>;

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp18Params {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    num_bins: u32,
    pass: u32,
}

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

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "═".repeat(60));
    println!("  EXP 18: Monster 3-Pass Radix Sort (5000+ Mkeys/s target)");
    println!("  Config: 11+11+10 bits | 2048/2048/1024 bins");
    println!("  TG-wide atomics | Decoupled lookback | 4096-elem tiles");
    println!("{}", "═".repeat(60));

    let pso_hist = ctx.make_pipeline("exp18_combined_histogram");
    let pso_prefix = ctx.make_pipeline("exp18_global_prefix");
    let pso_zero = ctx.make_pipeline("exp18_zero_status");
    let pso_partition = ctx.make_pipeline("exp18_partition");

    let sizes = [16_000_000usize];

    for &n in &sizes {
        let num_tiles = n.div_ceil(TILE_SIZE);
        let max_tile_status = num_tiles * BINS_P0; // 2048 bins max

        println!("\n  ── {} elements ({} tiles) ──", n, num_tiles);

        // Generate random data + CPU reference sort
        let input = gen_random_u32(n);
        let mut expected = input.clone();
        expected.sort();

        // Allocate GPU buffers
        let buf_input = alloc_buffer_with_data(&ctx.device, &input);
        let buf_a = alloc_buffer(&ctx.device, n * 4);
        let buf_b = alloc_buffer(&ctx.device, n * 4);
        let buf_global_hist = alloc_buffer(&ctx.device, TOTAL_BINS * 4);
        let buf_tile_status = alloc_buffer(&ctx.device, max_tile_status * 4);

        let tg_size = MTLSize { width: THREADS_PER_TG, height: 1, depth: 1 };
        let one_tg = MTLSize { width: 1, height: 1, depth: 1 };
        let hist_grid = MTLSize { width: num_tiles, height: 1, depth: 1 };

        let params_size = std::mem::size_of::<Exp18Params>();

        // Pass configs: (shift, num_bins, pass_index, pass_offset_in_global_hist)
        let pass_configs: [(u32, u32, u32, u32); 3] = [
            (0,  2048, 0, 0),                              // bits 0-10
            (11, 2048, 1, BINS_P0 as u32),                 // bits 11-21
            (22, 1024, 2, (BINS_P0 + BINS_P1) as u32),    // bits 22-31
        ];

        let encode_sort = |cmd: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>| {
            // Zero global_hist (CPU-side)
            unsafe {
                std::ptr::write_bytes(
                    buf_global_hist.contents().as_ptr() as *mut u8, 0,
                    TOTAL_BINS * 4,
                );
            }

            let n_u32 = n as u32;

            // ── Encoder 1: Combined histogram ──
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_hist);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_a.as_ref()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut _).unwrap(),
                    4, 2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(hist_grid, tg_size);
            enc.endEncoding();

            // ── Encoder 2: Global prefix for all 3 passes ──
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pso_prefix);
            for &(_, num_bins, _, pass_offset) in &pass_configs {
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 0);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&pass_offset as *const u32 as *mut _).unwrap(),
                        4, 1,
                    );
                    enc.setBytes_length_atIndex(
                        NonNull::new(&num_bins as *const u32 as *mut _).unwrap(),
                        4, 2,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(one_tg, tg_size);
            }
            enc.endEncoding();

            // ── 3 passes: zero_status + partition with ping-pong ──
            for (i, &(shift, num_bins, pass_idx, _pass_offset)) in pass_configs.iter().enumerate() {
                let params = Exp18Params {
                    element_count: n as u32,
                    num_tiles: num_tiles as u32,
                    shift,
                    num_bins,
                    pass: pass_idx,
                };

                let (src_buf, dst_buf) = if i % 2 == 0 {
                    (buf_a.as_ref(), buf_b.as_ref())
                } else {
                    (buf_b.as_ref(), buf_a.as_ref())
                };

                let status_entries = (num_tiles as u32) * num_bins;
                let zero_tgs = (status_entries as usize).div_ceil(THREADS_PER_TG);

                // Zero tile_status
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&pso_zero);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 0);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&status_entries as *const u32 as *mut _).unwrap(),
                        4, 1,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: zero_tgs, height: 1, depth: 1 }, tg_size,
                );
                enc.endEncoding();

                // Partition
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&pso_partition);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_tile_status.as_ref()), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(buf_global_hist.as_ref()), 0, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&params as *const Exp18Params as *mut _).unwrap(),
                        params_size, 4,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: num_tiles, height: 1, depth: 1 }, tg_size,
                );
                enc.endEncoding();
            }
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

        // After 3 passes (odd), result is in buf_b
        let results: Vec<u32> = unsafe { read_buffer_slice(&buf_b, n) };
        let mismatches = check_correctness(&results, &expected);
        let correct = mismatches == 0;

        if !correct {
            println!("    CORRECTNESS FAILED: {} mismatches out of {}", mismatches, n);
            // Show some debug info
            println!("    First 10 result:   {:?}", &results[..10.min(n)]);
            println!("    First 10 expected: {:?}", &expected[..10.min(n)]);
        } else {
            println!("    Correctness: PASS");
        }

        // ── Benchmark ──
        let mut times: Vec<f64> = Vec::new();

        for iter in 0..(WARMUP + RUNS) {
            // Re-copy input for each run
            unsafe {
                let src = buf_input.contents().as_ptr() as *const u8;
                let dst = buf_a.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src, dst, n * 4);
            }

            let cmd = ctx.command_buffer();
            encode_sort(&cmd);
            cmd.commit();
            cmd.waitUntilCompleted();

            let gpu_time = unsafe { cmd.GPUEndTime() - cmd.GPUStartTime() };
            if iter >= WARMUP {
                times.push(gpu_time * 1000.0); // ms
            }
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p5 = percentile(&times, 5.0);
        let p50 = percentile(&times, 50.0);
        let p95 = percentile(&times, 95.0);
        let mkeys = n as f64 / p50 / 1000.0;
        let bw = (n as f64 * 4.0 * 2.0 * 3.0) / p50 / 1e6; // 3 passes R+W

        println!("    Median: {:.3} ms | {:.0} Mkeys/s | {:.0} GB/s effective",
                 p50, mkeys, bw);
        println!("    [p5={:.3} p95={:.3} spread={:.0}%]",
                 p5, p95, if p5 > 0.0 { (p95 - p5) / p5 * 100.0 } else { 0.0 });

        let status = if correct { "PASS" } else { "FAIL" };
        let vs_baseline = mkeys / 3003.0;
        println!("    vs exp16 baseline (3003): {:.2}x | Status: {}", vs_baseline, status);
    }
}
