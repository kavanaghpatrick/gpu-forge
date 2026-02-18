//! Experiment 4: Decoupled Lookback Prefix Sum
//!
//! FIRST Apple Silicon implementation of the decoupled lookback algorithm.
//! Single-dispatch inclusive prefix sum using cross-threadgroup relaxed
//! atomics. Tests whether Apple Silicon's memory model allows this despite
//! Metal's spec only guaranteeing relaxed ordering.
//!
//! Comparison: 3-dispatch multi-pass vs 1-dispatch decoupled lookback.

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

const TILE_SIZE: usize = 256;
const N: usize = 262_144; // 256 * 1024 = 1024 tiles
const NUM_TILES: usize = N / TILE_SIZE;
const WARMUP: usize = 3;
const RUNS: usize = 10;

#[repr(C)]
#[derive(Clone, Copy)]
struct ExpParams {
    element_count: u32,
    num_passes: u32,
    mode: u32,
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 4: Decoupled Lookback Prefix Sum");
    println!("{}", "=".repeat(60));
    println!("Hypothesis: Single-dispatch scan via cross-TG atomics works on Apple Silicon");
    println!(
        "Setup: {} elements, {} tiles of {}\n",
        N, NUM_TILES, TILE_SIZE
    );

    let pso_reduce = ctx.make_pipeline("exp4_reduce");
    let pso_scan_add = ctx.make_pipeline("exp4_local_scan_and_add");
    let pso_decoupled = ctx.make_pipeline("exp4_decoupled_lookback");

    // Input: all ones → prefix sum should be [1, 2, 3, ..., N]
    let input: Vec<u32> = vec![1u32; N];
    let buf_input = alloc_buffer_with_data(&ctx.device, &input);
    let buf_output = alloc_buffer(&ctx.device, N * std::mem::size_of::<u32>());

    // Tile buffers for multi-pass
    let buf_tile_sums = alloc_buffer(&ctx.device, NUM_TILES * std::mem::size_of::<u32>());
    let buf_tile_prefix = alloc_buffer_with_data(&ctx.device, &vec![0u32; NUM_TILES]);

    // Tile status for decoupled lookback (packed flag|value)
    let buf_tile_status = alloc_buffer(&ctx.device, NUM_TILES * std::mem::size_of::<u32>());

    let params = ExpParams {
        element_count: N as u32,
        num_passes: NUM_TILES as u32,
        mode: 0,
    };
    let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

    // --- Version A: Multi-pass (reduce + CPU scan + local_scan_and_add) ---
    let mut times_a = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        // Phase 1: Reduce per tile
        let cmd1 = ctx.command_buffer();
        let enc1 = cmd1.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc1,
            &pso_reduce,
            &[
                (buf_input.as_ref(), 0),
                (buf_tile_sums.as_ref(), 1),
                (buf_params.as_ref(), 2),
            ],
            NUM_TILES,
            TILE_SIZE,
        );
        enc1.endEncoding();
        cmd1.commit();
        cmd1.waitUntilCompleted();

        // Phase 2: CPU exclusive prefix sum of tile sums
        let tile_sums: Vec<u32> = unsafe { read_buffer_slice(&buf_tile_sums, NUM_TILES) };
        let mut tile_prefixes = vec![0u32; NUM_TILES];
        for j in 1..NUM_TILES {
            tile_prefixes[j] = tile_prefixes[j - 1] + tile_sums[j - 1];
        }
        unsafe { write_buffer(&buf_tile_prefix, &tile_prefixes) };

        // Phase 3: Local scan + add tile prefix
        let cmd2 = ctx.command_buffer();
        let enc2 = cmd2.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc2,
            &pso_scan_add,
            &[
                (buf_input.as_ref(), 0),
                (buf_output.as_ref(), 1),
                (buf_tile_prefix.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            NUM_TILES,
            TILE_SIZE,
        );
        enc2.endEncoding();
        cmd2.commit();
        cmd2.waitUntilCompleted();

        if i >= WARMUP {
            let ms1 = gpu_elapsed_ms(&cmd1);
            let ms2 = gpu_elapsed_ms(&cmd2);
            times_a.push(ms1 + ms2); // GPU time only (CPU scan is ~free)
        }
    }

    // Verify multi-pass correctness
    let results_a: Vec<u32> = unsafe { read_buffer_slice(&buf_output, N) };
    let correct_a = results_a.iter().enumerate().all(|(i, &v)| v == (i + 1) as u32);

    // --- Version B: Single-pass decoupled lookback ---
    let mut times_b = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        // Clear tile status to 0 (FLAG_NOT_READY)
        unsafe {
            let ptr = buf_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_TILES * std::mem::size_of::<u32>());
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc,
            &pso_decoupled,
            &[
                (buf_input.as_ref(), 0),
                (buf_output.as_ref(), 1),
                (buf_tile_status.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            NUM_TILES,
            TILE_SIZE,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if i >= WARMUP {
            times_b.push(gpu_elapsed_ms(&cmd));
        }
    }

    // Verify decoupled lookback correctness
    let results_b: Vec<u32> = unsafe { read_buffer_slice(&buf_output, N) };
    let correct_b = results_b
        .iter()
        .enumerate()
        .all(|(i, &v)| v == (i + 1) as u32);

    // Check first few and last few values for diagnostics
    let first_5_b: Vec<u32> = results_b.iter().take(5).copied().collect();
    let last_5_b: Vec<u32> = results_b.iter().rev().take(5).rev().copied().collect();

    let med_a = median(&mut times_a);
    let med_b = median(&mut times_b);
    let mops_a = N as f64 / med_a / 1000.0;
    let mops_b = N as f64 / med_b / 1000.0;

    println!("Results:");
    println!(
        "  Multi-pass (3 dispatch): {:.3} ms  ({:.0} Mops)  correct: {}",
        med_a, mops_a, correct_a
    );
    println!(
        "  Decoupled (1 dispatch):  {:.3} ms  ({:.0} Mops)  correct: {}",
        med_b, mops_b, correct_b
    );
    println!("  Speedup:                 {:.2}x", med_a / med_b);
    println!();

    if correct_b {
        println!("  ** HOLY SHIT: Decoupled lookback WORKS on Apple Silicon! **");
        println!("  ** Single-dispatch prefix sum with cross-TG relaxed atomics **");
    } else {
        println!("  Decoupled lookback produced INCORRECT results.");
        println!("  First 5: {:?} (expected [1,2,3,4,5])", first_5_b);
        println!("  Last 5:  {:?} (expected [{},{},{},{},{}])",
            last_5_b, N-4, N-3, N-2, N-1, N);
        println!("  Apple Silicon's memory model may not support this pattern.");
    }
}
