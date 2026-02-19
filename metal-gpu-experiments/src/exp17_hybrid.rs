//! Experiment 17: MSD+LSD Hybrid Radix Sort (5000+ Mkeys/s target)
//!
//! Architecture: 1 MSD scatter (bits 24:31) creates 256 buckets of ~62K
//! elements (~250KB each, SLC-resident), then 3 per-bucket LSD passes
//! at SLC speed. Single encoder, 14 dispatches, zero CPU readback.
//!
//! Phase 0: SLC scatter bandwidth benchmark — gates the entire experiment.

use crate::metal_ctx::*;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
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
        return;
    } else {
        println!(
            "  GO: SLC scatter at 250K = {:.0} GB/s (>= 80 GB/s threshold)",
            bw_250k
        );
        println!("  Hybrid approach is viable — proceeding with MSD+LSD sort.");
    }
}
