//! Experiment 6: Single-Dispatch Stream Compaction
//!
//! Combines Exp 4 (decoupled lookback) + Exp 5 (kernel fusion).
//! Traditional stream compaction: 4 dispatches (predicate → reduce → scan → scatter).
//! Persistent kernel version: 1 dispatch does ALL phases, data stays in registers.
//!
//! Predicate: value % 3 == 0  (keeps ~33% of elements)

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

const TILE_SIZE: usize = 256;
const N: usize = 1_048_576; // 1M elements
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
    println!("Experiment 6: Single-Dispatch Stream Compaction");
    println!("{}", "=".repeat(60));
    println!("Combines: decoupled lookback (Exp 4) + kernel fusion (Exp 5)");
    println!("Predicate: value % 3 == 0  (~33% selectivity)");
    println!(
        "Setup: {} elements, {} tiles of {}\n",
        N, NUM_TILES, TILE_SIZE
    );

    // Pipelines
    let pso_predicate = ctx.make_pipeline("exp6_predicate");
    let pso_reduce = ctx.make_pipeline("exp4_reduce");
    let pso_scan_add = ctx.make_pipeline("exp4_local_scan_and_add");
    let pso_scatter = ctx.make_pipeline("exp6_scatter");
    let pso_single = ctx.make_pipeline("exp6_compact_single");

    // Input: [0, 1, 2, 3, ..., N-1]
    let input: Vec<u32> = (0..N as u32).collect();
    let buf_input = alloc_buffer_with_data(&ctx.device, &input);

    // Expected: elements where value % 3 == 0
    let expected: Vec<u32> = input.iter().copied().filter(|v| v % 3 == 0).collect();
    let expected_count = expected.len();

    let params = ExpParams {
        element_count: N as u32,
        num_passes: NUM_TILES as u32,
        mode: 0,
    };
    let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

    // ── Version A: 4-dispatch traditional approach ──

    let buf_pred = alloc_buffer(&ctx.device, N * 4);
    let buf_tile_sums = alloc_buffer(&ctx.device, NUM_TILES * 4);
    let buf_tile_prefix = alloc_buffer_with_data(&ctx.device, &vec![0u32; NUM_TILES]);
    let buf_prefix = alloc_buffer(&ctx.device, N * 4); // inclusive prefix sum of predicates
    let buf_output_a = alloc_buffer(&ctx.device, N * 4);

    let mut times_a = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let mut total_ms = 0.0;

        // Dispatch 1: Compute predicates
        let cmd1 = ctx.command_buffer();
        let enc1 = cmd1.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc1,
            &pso_predicate,
            &[
                (buf_input.as_ref(), 0),
                (buf_pred.as_ref(), 1),
                (buf_params.as_ref(), 2),
            ],
            N,
        );
        enc1.endEncoding();
        cmd1.commit();
        cmd1.waitUntilCompleted();
        total_ms += gpu_elapsed_ms(&cmd1);

        // Dispatch 2: Reduce predicate tiles
        let cmd2 = ctx.command_buffer();
        let enc2 = cmd2.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc2,
            &pso_reduce,
            &[
                (buf_pred.as_ref(), 0),
                (buf_tile_sums.as_ref(), 1),
                (buf_params.as_ref(), 2),
            ],
            NUM_TILES,
            TILE_SIZE,
        );
        enc2.endEncoding();
        cmd2.commit();
        cmd2.waitUntilCompleted();
        total_ms += gpu_elapsed_ms(&cmd2);

        // CPU: exclusive prefix sum of tile sums
        let tile_sums: Vec<u32> = unsafe { read_buffer_slice(&buf_tile_sums, NUM_TILES) };
        let mut tile_prefixes = vec![0u32; NUM_TILES];
        for j in 1..NUM_TILES {
            tile_prefixes[j] = tile_prefixes[j - 1] + tile_sums[j - 1];
        }
        unsafe { write_buffer(&buf_tile_prefix, &tile_prefixes) };

        // Dispatch 3: Local scan + add tile prefix
        let cmd3 = ctx.command_buffer();
        let enc3 = cmd3.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc3,
            &pso_scan_add,
            &[
                (buf_pred.as_ref(), 0),
                (buf_prefix.as_ref(), 1),
                (buf_tile_prefix.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            NUM_TILES,
            TILE_SIZE,
        );
        enc3.endEncoding();
        cmd3.commit();
        cmd3.waitUntilCompleted();
        total_ms += gpu_elapsed_ms(&cmd3);

        // Dispatch 4: Scatter
        let cmd4 = ctx.command_buffer();
        let enc4 = cmd4.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc4,
            &pso_scatter,
            &[
                (buf_input.as_ref(), 0),
                (buf_pred.as_ref(), 1),
                (buf_prefix.as_ref(), 2),
                (buf_output_a.as_ref(), 3),
                (buf_params.as_ref(), 4),
            ],
            N,
        );
        enc4.endEncoding();
        cmd4.commit();
        cmd4.waitUntilCompleted();
        total_ms += gpu_elapsed_ms(&cmd4);

        if i >= WARMUP {
            times_a.push(total_ms);
        }
    }

    // Verify multi-dispatch results
    let results_a: Vec<u32> = unsafe { read_buffer_slice(&buf_output_a, expected_count) };
    let correct_a = results_a == expected;

    // ── Version B: Single-dispatch persistent kernel ──

    let buf_output_b = alloc_buffer(&ctx.device, N * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, NUM_TILES * 4);
    let buf_total_count = alloc_buffer_with_data(&ctx.device, &[0u32]);

    let mut times_b = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        // Clear tile status
        unsafe {
            let ptr = buf_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_TILES * 4);
            let cptr = buf_total_count.contents().as_ptr() as *mut u32;
            *cptr = 0;
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc,
            &pso_single,
            &[
                (buf_input.as_ref(), 0),
                (buf_output_b.as_ref(), 1),
                (buf_tile_status.as_ref(), 2),
                (buf_total_count.as_ref(), 3),
                (buf_params.as_ref(), 4),
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

    // Verify single-dispatch results
    let count_b: Vec<u32> = unsafe { read_buffer_slice(&buf_total_count, 1) };
    let actual_count = count_b[0] as usize;
    let results_b: Vec<u32> = unsafe { read_buffer_slice(&buf_output_b, actual_count.min(N)) };
    let correct_b = actual_count == expected_count && results_b == expected;

    // ── Results ──

    let med_a = median(&mut times_a);
    let med_b = median(&mut times_b);

    let throughput_a = N as f64 / med_a / 1000.0; // Mops
    let throughput_b = N as f64 / med_b / 1000.0;
    let bandwidth_a = (N * 4) as f64 / (med_a / 1000.0) / 1e9; // GB/s (input reads)
    let bandwidth_b = (N * 4) as f64 / (med_b / 1000.0) / 1e9;

    println!("Results:");
    println!(
        "  4-dispatch traditional:    {:.3} ms  ({:.0} Mops, {:.0} GB/s)  correct: {}",
        med_a, throughput_a, bandwidth_a, correct_a
    );
    println!(
        "  1-dispatch persistent:     {:.3} ms  ({:.0} Mops, {:.0} GB/s)  correct: {}",
        med_b, throughput_b, bandwidth_b, correct_b
    );
    println!("  Speedup:                   {:.2}x", med_a / med_b);
    println!();
    println!(
        "  Expected output count:     {} ({:.1}% selectivity)",
        expected_count,
        expected_count as f64 / N as f64 * 100.0
    );
    println!(
        "  Actual output count (B):   {}",
        actual_count
    );
    println!();

    // Break down where the savings come from
    let dispatch_savings_ms = 3.0 * 0.050; // ~50us per dispatch × 3 eliminated dispatches
    let bandwidth_savings_ms = med_a - med_b - dispatch_savings_ms;

    if med_a / med_b > 3.0 {
        println!("  ** Persistent kernel DOMINATES: {:.1}x faster **", med_a / med_b);
        println!("  Dispatch overhead eliminated:  ~{:.3} ms (3 × ~50us)", dispatch_savings_ms);
        println!("  DRAM round-trips eliminated:   ~{:.3} ms (3 intermediate buffers)", bandwidth_savings_ms.max(0.0));
        println!("  Data stays in registers between predicate → scan → scatter");
    } else if med_a / med_b > 1.5 {
        println!("  ** Significant win from persistent kernel approach **");
    } else {
        println!("  Modest improvement — 1M elements may not fully expose overhead");
    }
}
