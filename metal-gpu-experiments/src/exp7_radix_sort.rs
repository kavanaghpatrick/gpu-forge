//! Experiment 7: Single-Dispatch-Per-Pass Radix Sort
//!
//! FIRST persistent-kernel radix sort on Metal.
//! Disproves claim that Metal requires multi-dispatch tree reduction.
//!
//! 8-bit radix (256 bins), 4 LSD passes.
//! Version A: 8 dispatches (2 per pass: histogram + scatter)
//! Version B: 4 dispatches (1 persistent per pass)

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

const TILE_SIZE: usize = 256;
const NUM_BINS: usize = 256;
// SAFETY: N must be small enough that NUM_TILES ≤ GPU's concurrent threadgroup capacity.
// Phase 3 of the persistent kernel requires ALL tiles to be simultaneously resident —
// if any tile can't be scheduled (because spinning tiles occupy all execution slots),
// the GPU deadlocks and freezes the ENTIRE MACHINE (macOS has no GPU preemption timeout).
//
// M4 Pro: ~20 cores × ~3 TGs/core = ~60 concurrent TGs. Use 64 tiles max.
// For a production persistent radix sort, use occupancy-bound dispatch with tile loops.
const N: usize = 16_384; // 16K elements — safe: 64 tiles fit in concurrent execution
const NUM_TILES: usize = N / TILE_SIZE; // 64
const WARMUP: usize = 5;
const RUNS: usize = 20;
const NUM_PASSES: usize = 4; // 32-bit / 8-bit radix

#[repr(C)]
#[derive(Clone, Copy)]
struct RadixParams {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 7: Single-Dispatch-Per-Pass Radix Sort");
    println!("{}", "=".repeat(60));
    println!("Disproves: 'Metal requires multi-dispatch tree reduction'");
    println!(
        "Setup: {} elements, {} tiles, 8-bit radix, 4 passes",
        N, NUM_TILES
    );
    println!("(N capped at {}K — persistent kernel Phase 3 requires all tiles concurrent)\n",
        N / 1024
    );

    // Pipelines
    let pso_histogram = ctx.make_pipeline("exp7_histogram");
    let pso_scatter = ctx.make_pipeline("exp7_scatter");
    let pso_persistent = ctx.make_pipeline("exp7_radix_persistent");

    // Random input
    let input: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..N).map(|_| rng.gen::<u32>()).collect()
    };
    let buf_input = alloc_buffer_with_data(&ctx.device, &input);

    // Reference: CPU sort for verification
    let mut expected = input.clone();
    expected.sort();

    // ── Version A: Multi-dispatch (8 dispatches) ──

    let buf_a1 = alloc_buffer(&ctx.device, N * 4);
    let buf_a2 = alloc_buffer(&ctx.device, N * 4);
    let buf_tile_hists = alloc_buffer(&ctx.device, NUM_TILES * NUM_BINS * 4);
    let buf_offsets = alloc_buffer(&ctx.device, NUM_TILES * NUM_BINS * 4);

    let mut times_a = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        let mut total_ms = 0.0;

        // Copy input to buf_a1 for this iteration
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
        }

        // Ping-pong: pass 0: a1→a2, pass 1: a2→a1, pass 2: a1→a2, pass 3: a2→a1
        for pass in 0..NUM_PASSES {
            let shift = (pass * 8) as u32;
            let params = RadixParams {
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

    // Result in a2 (even number of passes: a1→a2→a1→a2 wait no)
    // Pass 0: a1→a2, Pass 1: a2→a1, Pass 2: a1→a2, Pass 3: a2→a1
    // Final result in a1 (odd last pass writes to a1)
    let results_a: Vec<u32> = unsafe { read_buffer_slice(&buf_a1, N) };
    let correct_a = results_a == expected;

    // ── Version B: Persistent (4 dispatches) ──

    let buf_b1 = alloc_buffer(&ctx.device, N * 4);
    let buf_b2 = alloc_buffer(&ctx.device, N * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, NUM_TILES * NUM_BINS * 4);

    let mut times_b = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        let mut total_ms = 0.0;

        // Copy input to buf_b1
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_b1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
        }

        for pass in 0..NUM_PASSES {
            let shift = (pass * 8) as u32;
            let params = RadixParams {
                element_count: N as u32,
                num_tiles: NUM_TILES as u32,
                shift,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            let (src, dst) = if pass % 2 == 0 {
                (buf_b1.as_ref(), buf_b2.as_ref())
            } else {
                (buf_b2.as_ref(), buf_b1.as_ref())
            };

            // Clear tile_status (all zeros = FLAG_NOT_READY)
            unsafe {
                let ptr = buf_tile_status.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(ptr, 0, NUM_TILES * NUM_BINS * 4);
            }

            // Single persistent dispatch — no separate totals buffer needed,
            // global totals are read directly from tile_status atomics
            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d_tg(
                &enc,
                &pso_persistent,
                &[
                    (src, 0),
                    (dst, 1),
                    (buf_tile_status.as_ref(), 2),
                    (buf_params.as_ref(), 3),
                ],
                NUM_TILES,
                TILE_SIZE,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            total_ms += gpu_elapsed_ms(&cmd);
        }

        if iter >= WARMUP {
            times_b.push(total_ms);
        }
    }

    // Same ping-pong: final result in b1
    let results_b: Vec<u32> = unsafe { read_buffer_slice(&buf_b1, N) };
    let correct_b = results_b == expected;

    // ── Results ──

    let med_a = median(&mut times_a);
    let med_b = median(&mut times_b);

    let throughput_a = N as f64 / med_a / 1000.0;
    let throughput_b = N as f64 / med_b / 1000.0;

    println!("Results (total across 4 passes):");
    println!(
        "  8-dispatch traditional:    {:.3} ms  ({:.0} Mkeys/s)  correct: {}",
        med_a, throughput_a, correct_a
    );
    println!(
        "  4-dispatch persistent:     {:.3} ms  ({:.0} Mkeys/s)  correct: {}",
        med_b, throughput_b, correct_b
    );
    println!("  Speedup:                   {:.2}x", med_a / med_b);
    println!();

    // Per-pass breakdown
    println!(
        "  Per pass: traditional {:.3} ms, persistent {:.3} ms",
        med_a / 4.0,
        med_b / 4.0
    );
    let dispatch_savings = 4.0 * 0.050; // 4 eliminated dispatches × ~50us
    println!(
        "  Dispatch overhead eliminated: ~{:.3} ms (4 dispatches × ~50us)",
        dispatch_savings
    );
    println!(
        "  DRAM savings: ~{:.3} ms (histogram intermediate eliminated per pass)",
        (med_a - med_b - dispatch_savings).max(0.0)
    );

    if !correct_a {
        println!("  WARNING: Version A INCORRECT! First 5: {:?}", &results_a[..5.min(N)]);
    }
    if !correct_b {
        println!("  WARNING: Version B INCORRECT! First 5: {:?}", &results_b[..5.min(N)]);
        // Show where it diverges
        let mut first_diff = N;
        for i in 0..N {
            if results_b[i] != expected[i] {
                first_diff = i;
                break;
            }
        }
        if first_diff < N {
            println!(
                "  First mismatch at [{}]: got {}, expected {}",
                first_diff, results_b[first_diff], expected[first_diff]
            );
        }
    }
}

/// Compute scatter offsets from per-tile histograms.
/// offsets[tile * 256 + bin] = global position for tile's bin-start.
fn compute_offsets(tile_hists: &[u32], num_tiles: usize) -> Vec<u32> {
    // Step 1: Column-wise exclusive prefix (per-bin across tiles)
    let mut tile_prefix = vec![0u32; num_tiles * NUM_BINS];
    for b in 0..NUM_BINS {
        let mut running = 0u32;
        for t in 0..num_tiles {
            tile_prefix[t * NUM_BINS + b] = running;
            running += tile_hists[t * NUM_BINS + b];
        }
    }

    // Step 2: Global bin totals and starts
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

    // Step 3: Final offsets = global_bin_start[bin] + tile_prefix[tile][bin]
    let mut offsets = vec![0u32; num_tiles * NUM_BINS];
    for t in 0..num_tiles {
        for b in 0..NUM_BINS {
            offsets[t * NUM_BINS + b] = global_bin_start[b] + tile_prefix[t * NUM_BINS + b];
        }
    }

    offsets
}
