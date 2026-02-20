//! Experiment 10: SIMD-Optimized Stable Rank
//!
//! The serial stable_rank O(TILE_SIZE) loop is the compute bottleneck in
//! persistent radix sort. This experiment replaces it with:
//!   Phase 1: simd_shuffle within-SG rank — O(31) max
//!   Phase 2: per-SG histogram cross-SG rank — O(7) reads
//!   Total: O(38) vs O(255) — ~6.7x fewer operations
//!
//! Version A: Serial rank (baseline, same kernel as exp8)
//! Version B: SIMD rank (shared_digits[] within-SG + TG histogram cross-SG)

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

const TILE_SIZE: usize = 256;
const NUM_BINS: usize = 256;
const N: usize = 1_048_576; // 1M elements
const NUM_TILES: usize = N / TILE_SIZE; // 4096
const NUM_TGS: usize = 64;
const NUM_PASSES: usize = 4;
const WARMUP: usize = 5;
const RUNS: usize = 20;

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp10PersistParams {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    num_tgs: u32,
    counter_base: u32,
    ts_offset: u32,
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 10: SIMD-Optimized Stable Rank");
    println!("{}", "=".repeat(60));
    println!("Serial O(255) rank vs SIMD O(38) rank in persistent radix sort");
    println!(
        "Setup: {}M elements, {} tiles, {} persistent TGs",
        N / 1_000_000,
        NUM_TILES,
        NUM_TGS
    );
    println!("A: serial rank (baseline)  B: SIMD shuffle + histogram rank\n");

    // Pipelines
    let pso_serial = ctx.make_pipeline("exp10_serial_pass");
    let pso_simd = ctx.make_pipeline("exp10_simd_pass");

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

    let ts_section = NUM_TILES * NUM_BINS;

    // ── Version A: Serial rank ──────────────────────────────────────

    let buf_a1 = alloc_buffer(&ctx.device, N * 4);
    let buf_a2 = alloc_buffer(&ctx.device, N * 4);
    let buf_a_tile_status = alloc_buffer(&ctx.device, NUM_PASSES * ts_section * 4);
    let buf_a_counters = alloc_buffer(&ctx.device, 16 * 4);

    let mut times_a = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
            let ptr = buf_a_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_a_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 16 * 4);
        }

        let cmd = ctx.command_buffer();
        for pass in 0..NUM_PASSES {
            let shift = (pass * 8) as u32;
            let params = Exp10PersistParams {
                element_count: N as u32,
                num_tiles: NUM_TILES as u32,
                shift,
                num_tgs: NUM_TGS as u32,
                counter_base: (pass * 3) as u32,
                ts_offset: (pass * ts_section) as u32,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            let (src, dst) = if pass % 2 == 0 {
                (buf_a1.as_ref(), buf_a2.as_ref())
            } else {
                (buf_a2.as_ref(), buf_a1.as_ref())
            };

            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d_tg(
                &enc,
                &pso_serial,
                &[
                    (src, 0),
                    (dst, 1),
                    (buf_a_tile_status.as_ref(), 2),
                    (buf_a_counters.as_ref(), 3),
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
            times_a.push(elapsed);
        }
    }

    let results_a: Vec<u32> = unsafe { read_buffer_slice(&buf_a1, N) };
    let correct_a = results_a == expected;

    // ── Version B: SIMD rank ────────────────────────────────────────

    let buf_b1 = alloc_buffer(&ctx.device, N * 4);
    let buf_b2 = alloc_buffer(&ctx.device, N * 4);
    let buf_b_tile_status = alloc_buffer(&ctx.device, NUM_PASSES * ts_section * 4);
    let buf_b_counters = alloc_buffer(&ctx.device, 16 * 4);

    // Pass-by-pass correctness check
    println!("  SIMD rank pass-by-pass correctness:");
    for test_passes in 1..=4u32 {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_b1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
            let ptr = buf_b_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_b_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 16 * 4);
        }

        let cmd = ctx.command_buffer();
        for pass in 0..test_passes as usize {
            let shift = (pass * 8) as u32;
            let params = Exp10PersistParams {
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
                &pso_simd,
                &[
                    (src, 0),
                    (dst, 1),
                    (buf_b_tile_status.as_ref(), 2),
                    (buf_b_counters.as_ref(), 3),
                    (buf_params.as_ref(), 4),
                ],
                NUM_TGS,
                TILE_SIZE,
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

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

    // Benchmark Version B
    let mut times_b = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_b1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
            let ptr = buf_b_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_b_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 16 * 4);
        }

        let cmd = ctx.command_buffer();
        for pass in 0..NUM_PASSES {
            let shift = (pass * 8) as u32;
            let params = Exp10PersistParams {
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
                &pso_simd,
                &[
                    (src, 0),
                    (dst, 1),
                    (buf_b_tile_status.as_ref(), 2),
                    (buf_b_counters.as_ref(), 3),
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

    let results_b: Vec<u32> = unsafe { read_buffer_slice(&buf_b1, N) };
    let correct_b = results_b == expected;

    // ── Results ────────────────────────────────────────────────────

    let med_a = median(&mut times_a);
    let med_b = median(&mut times_b);

    let throughput_a = N as f64 / med_a / 1000.0;
    let throughput_b = N as f64 / med_b / 1000.0;

    println!("Results (full 32-bit sort, {}M elements):", N / 1_000_000);
    println!(
        "  A (serial rank O(255)):   {:.3} ms  ({:.0} Mkeys/s)  correct: {}",
        med_a, throughput_a, correct_a
    );
    println!(
        "  B (SIMD rank O(38)):      {:.3} ms  ({:.0} Mkeys/s)  correct: {}",
        med_b, throughput_b, correct_b
    );

    let speedup = med_a / med_b;
    println!("  Speedup:                  {:.2}x", speedup);
    println!();

    println!("  Analysis:");
    println!(
        "    Rank ops per thread: serial=127 avg, SIMD=19 avg (shuffle+hist)"
    );
    println!("    Extra TG memory: +8 KB (sg_hist[8][256])");
    println!("    Extra barriers: +1 per tile (histogram contribute)");
    let rank_fraction = 1.0 - (med_b / med_a);
    println!(
        "    Rank compute was {:.0}% of total runtime",
        rank_fraction * 100.0
    );

    if !correct_a {
        println!("\n  WARNING: Version A INCORRECT!");
    }
    if !correct_b {
        println!("\n  WARNING: Version B INCORRECT!");
        // Show diagnostic info
        let mismatches = results_b
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "    Mismatched: {} / {} ({:.1}%)",
            mismatches,
            N,
            mismatches as f64 / N as f64 * 100.0
        );
        let mut shown = 0;
        for i in 0..N {
            if results_b[i] != expected[i] {
                println!("    [{}]: got {}, expected {}", i, results_b[i], expected[i]);
                shown += 1;
                if shown >= 5 {
                    break;
                }
            }
        }
    }
}
