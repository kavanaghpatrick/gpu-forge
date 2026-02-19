//! Experiment 8: Single-Dispatch Megasort (1M elements)
//!
//! Removes exp7's 16K limitation using occupancy-bound dispatch:
//! 64 persistent TGs loop over 4096 tiles via atomic work-stealing.
//!
//! Version A: 8 dispatches in 8 command buffers (2/pass × 4 passes)
//!   Traditional: histogram dispatch → CPU prefix sum → scatter dispatch
//!
//! Version B: 4 persistent dispatches in 1 command buffer
//!   Each compute encoder does histogram + lookback + scatter in one shot.
//!   Metal's inter-encoder barriers provide cross-pass coherence.

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

const TILE_SIZE: usize = 256;
const NUM_BINS: usize = 256;
const N: usize = 1_048_576; // 1M elements
const NUM_TILES: usize = N / TILE_SIZE; // 4096
const NUM_TGS: usize = 64; // Occupancy-bound: all simultaneously resident
const NUM_PASSES: usize = 4;
const WARMUP: usize = 5;
const RUNS: usize = 20;

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp8PassParams {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp8PersistParams {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    num_tgs: u32,
    counter_base: u32,
    ts_offset: u32,
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 8: Persistent Megasort");
    println!("{}", "=".repeat(60));
    println!("Removes exp7's 16K limit via occupancy-bound dispatch");
    println!(
        "Setup: {}M elements, {} tiles, {} persistent TGs",
        N / 1_000_000,
        NUM_TILES,
        NUM_TGS
    );
    println!(
        "Version A: 8 dispatches (8 cmdbufs)  vs  Version B: 4 persistent (1 cmdbuf)\n"
    );

    // Pipelines
    let pso_histogram = ctx.make_pipeline("exp8_histogram");
    let pso_scatter = ctx.make_pipeline("exp8_scatter");
    let pso_persistent = ctx.make_pipeline("exp8_persistent_pass");

    // Random input
    let input: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..N).map(|_| rng.gen::<u32>()).collect()
    };
    let buf_input = alloc_buffer_with_data(&ctx.device, &input);

    // CPU reference sort
    let mut expected = input.clone();
    expected.sort();

    // ── Version A: 8-dispatch traditional ──────────────────────────

    let buf_a1 = alloc_buffer(&ctx.device, N * 4);
    let buf_a2 = alloc_buffer(&ctx.device, N * 4);
    let buf_tile_hists = alloc_buffer(&ctx.device, NUM_TILES * NUM_BINS * 4);
    let buf_offsets = alloc_buffer(&ctx.device, NUM_TILES * NUM_BINS * 4);

    let mut times_a = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        let mut total_ms = 0.0;

        // Copy input to buf_a1
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
        }

        for pass in 0..NUM_PASSES {
            let shift = (pass * 8) as u32;
            let params = Exp8PassParams {
                element_count: N as u32,
                num_tiles: NUM_TILES as u32,
                shift,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            let (src, dst) = if pass % 2 == 0 {
                (buf_a1.as_ref(), buf_a2.as_ref())
            } else {
                (buf_a2.as_ref(), buf_a1.as_ref())
            };

            // Dispatch 1: Histogram
            let cmd1 = ctx.command_buffer();
            let enc1 = cmd1.computeCommandEncoder().unwrap();
            dispatch_1d_tg(
                &enc1,
                &pso_histogram,
                &[
                    (src, 0),
                    (buf_tile_hists.as_ref(), 1),
                    (buf_params.as_ref(), 2),
                ],
                NUM_TILES,
                TILE_SIZE,
            );
            enc1.endEncoding();
            cmd1.commit();
            cmd1.waitUntilCompleted();
            total_ms += gpu_elapsed_ms(&cmd1);

            // CPU: compute offsets from tile histograms
            let tile_hists: Vec<u32> =
                unsafe { read_buffer_slice(&buf_tile_hists, NUM_TILES * NUM_BINS) };
            let offsets = compute_offsets(&tile_hists, NUM_TILES);
            unsafe { write_buffer(&buf_offsets, &offsets) };

            // Dispatch 2: Scatter
            let cmd2 = ctx.command_buffer();
            let enc2 = cmd2.computeCommandEncoder().unwrap();
            dispatch_1d_tg(
                &enc2,
                &pso_scatter,
                &[
                    (src, 0),
                    (dst, 1),
                    (buf_offsets.as_ref(), 2),
                    (buf_params.as_ref(), 3),
                ],
                NUM_TILES,
                TILE_SIZE,
            );
            enc2.endEncoding();
            cmd2.commit();
            cmd2.waitUntilCompleted();
            total_ms += gpu_elapsed_ms(&cmd2);
        }

        if iter >= WARMUP {
            times_a.push(total_ms);
        }
    }

    // Pass 0: a1→a2, Pass 1: a2→a1, Pass 2: a1→a2, Pass 3: a2→a1
    // Final result in a1
    let results_a: Vec<u32> = unsafe { read_buffer_slice(&buf_a1, N) };
    let correct_a = results_a == expected;

    // ── Version B: 4 persistent dispatches in 1 command buffer ───

    let buf_b1 = alloc_buffer(&ctx.device, N * 4);
    let buf_b2 = alloc_buffer(&ctx.device, N * 4);
    // tile_status: 4 passes × 4096 tiles × 256 bins × 4 bytes = 16 MB
    // Each pass uses its own section (no clearing needed between passes)
    let ts_section = NUM_TILES * NUM_BINS;
    let buf_tile_status = alloc_buffer(&ctx.device, NUM_PASSES * ts_section * 4);
    // counters: 4 passes × 3 slots × 4 bytes = 48 bytes (allocate 64 for alignment)
    let buf_counters = alloc_buffer(&ctx.device, 16 * 4);

    // ── Incremental correctness check ──
    println!("  Pass-by-pass correctness check:");
    for test_passes in 1..=4u32 {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_b1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
            // Zero tile_status and counters
            let ptr = buf_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 16 * 4);
        }

        let cmd = ctx.command_buffer();
        for pass in 0..test_passes as usize {
            let shift = (pass * 8) as u32;
            let params = Exp8PersistParams {
                element_count: N as u32,
                num_tiles: NUM_TILES as u32,
                shift,
                num_tgs: NUM_TGS as u32,
                counter_base: (pass * 3) as u32,
                ts_offset: (pass * ts_section) as u32,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            let (src, dst) = if pass % 2 == 0 {
                (buf_b1.as_ref(), buf_b2.as_ref())
            } else {
                (buf_b2.as_ref(), buf_b1.as_ref())
            };

            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d_tg(
                &enc,
                &pso_persistent,
                &[
                    (src, 0),
                    (dst, 1),
                    (buf_tile_status.as_ref(), 2),
                    (buf_counters.as_ref(), 3),
                    (buf_params.as_ref(), 4),
                ],
                NUM_TGS,
                TILE_SIZE,
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        // Result: even passes write to buf_b2, odd passes write to buf_b1
        // After test_passes passes, the last pass index is (test_passes-1)
        // If (test_passes-1) is even → dst=buf_b2, if odd → dst=buf_b1
        let result_buf = if (test_passes - 1) % 2 == 0 {
            buf_b2.as_ref()
        } else {
            buf_b1.as_ref()
        };
        let results: Vec<u32> = unsafe { read_buffer_slice(result_buf, N) };
        let mut sorted_check = results.clone();
        sorted_check.sort();
        let is_perm = sorted_check == expected;
        let is_sorted = results.windows(2).all(|w| w[0] <= w[1]);
        println!(
            "    {} pass(es): permutation={}, sorted={}",
            test_passes, is_perm, is_sorted
        );
        if !is_perm {
            println!("      First 5: {:?}", &results[..5]);
            break;
        }
    }
    println!();

    // ── Benchmark Version B ──
    let mut times_b = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        // Copy input to buf_b1
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_b1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
        }

        // Clear tile_status and counters
        unsafe {
            let ptr = buf_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 16 * 4);
        }

        // Single command buffer, 4 compute encoders
        let cmd = ctx.command_buffer();
        for pass in 0..NUM_PASSES {
            let shift = (pass * 8) as u32;
            let params = Exp8PersistParams {
                element_count: N as u32,
                num_tiles: NUM_TILES as u32,
                shift,
                num_tgs: NUM_TGS as u32,
                counter_base: (pass * 3) as u32,
                ts_offset: (pass * ts_section) as u32,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            let (src, dst) = if pass % 2 == 0 {
                (buf_b1.as_ref(), buf_b2.as_ref())
            } else {
                (buf_b2.as_ref(), buf_b1.as_ref())
            };

            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d_tg(
                &enc,
                &pso_persistent,
                &[
                    (src, 0),
                    (dst, 1),
                    (buf_tile_status.as_ref(), 2),
                    (buf_counters.as_ref(), 3),
                    (buf_params.as_ref(), 4),
                ],
                NUM_TGS,
                TILE_SIZE,
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        let elapsed = gpu_elapsed_ms(&cmd);

        if iter >= WARMUP {
            times_b.push(elapsed);
        }
    }

    // Final result: 4 passes, last pass index=3 (odd) → dst=buf_b1
    let results_b: Vec<u32> = unsafe { read_buffer_slice(&buf_b1, N) };
    let correct_b = results_b == expected;

    // ── Results ────────────────────────────────────────────────────

    let med_a = median(&mut times_a);
    let med_b = median(&mut times_b);

    let throughput_a = N as f64 / med_a / 1000.0;
    let throughput_b = N as f64 / med_b / 1000.0;

    println!("Results (full 32-bit sort, {}M elements):", N / 1_000_000);
    println!(
        "  Version A (8 dispatches, 8 cmdbufs): {:.3} ms  ({:.0} Mkeys/s)  correct: {}",
        med_a, throughput_a, correct_a
    );
    println!(
        "  Version B (4 persistent, 1 cmdbuf):  {:.3} ms  ({:.0} Mkeys/s)  correct: {}",
        med_b, throughput_b, correct_b
    );

    let speedup = med_a / med_b;
    println!("  Speedup:                             {:.2}x", speedup);
    println!();

    println!("  Dispatches:       A=8, B=4  (2x fewer)");
    println!("  Command buffers:  A=8, B=1  (8x fewer round-trips)");
    println!(
        "  CPU work:         A=4 × {}KB histogram read+prefix, B=none",
        NUM_TILES * NUM_BINS * 4 / 1024
    );
    println!(
        "  Global syncs:     B=4 (1 per pass, ~1us each)"
    );

    if !correct_a {
        println!("  WARNING: Version A INCORRECT!");
        show_mismatch(&results_a, &expected);
    }
    if !correct_b {
        println!("  WARNING: Version B INCORRECT!");
        show_mismatch(&results_b, &expected);
    }
}

fn show_mismatch(got: &[u32], expected: &[u32]) {
    let mismatches = got.iter().zip(expected.iter())
        .filter(|(a, b)| a != b).count();
    println!("    Mismatched: {} / {} ({:.1}%)",
        mismatches, got.len(), mismatches as f64 / got.len() as f64 * 100.0);

    // Show first few mismatches
    let mut shown = 0;
    for i in 0..got.len() {
        if got[i] != expected[i] {
            println!("    [{}]: got {}, expected {}", i, got[i], expected[i]);
            shown += 1;
            if shown >= 5 { break; }
        }
    }

    // Check if output is sorted
    let sorted = got.windows(2).all(|w| w[0] <= w[1]);
    println!("    Output sorted: {}", sorted);

    // Check if valid permutation
    let mut got_sorted = got.to_vec();
    got_sorted.sort();
    let mut exp_sorted = expected.to_vec();
    exp_sorted.sort();
    println!("    Valid permutation: {}", got_sorted == exp_sorted);
}

/// Compute scatter offsets from per-tile histograms (CPU-side for Version A).
fn compute_offsets(tile_hists: &[u32], num_tiles: usize) -> Vec<u32> {
    // Column-wise exclusive prefix per bin across tiles
    let mut tile_prefix = vec![0u32; num_tiles * NUM_BINS];
    for b in 0..NUM_BINS {
        let mut running = 0u32;
        for t in 0..num_tiles {
            tile_prefix[t * NUM_BINS + b] = running;
            running += tile_hists[t * NUM_BINS + b];
        }
    }

    // Global bin starts
    let mut global_totals = vec![0u32; NUM_BINS];
    for b in 0..NUM_BINS {
        for t in 0..num_tiles {
            global_totals[b] += tile_hists[t * NUM_BINS + b];
        }
    }
    let mut global_bin_start = vec![0u32; NUM_BINS];
    let mut running = 0u32;
    for b in 0..NUM_BINS {
        global_bin_start[b] = running;
        running += global_totals[b];
    }

    // Final: global_bin_start[bin] + tile_prefix[tile][bin]
    let mut offsets = vec![0u32; num_tiles * NUM_BINS];
    for t in 0..num_tiles {
        for b in 0..NUM_BINS {
            offsets[t * NUM_BINS + b] = global_bin_start[b] + tile_prefix[t * NUM_BINS + b];
        }
    }
    offsets
}
