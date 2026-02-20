//! Experiment 9: Single-Dispatch Coherence Fixes
//!
//! Tests whether `volatile` qualifiers on data buffers fix the stale-read
//! bug that killed exp8's single-dispatch megasort.
//!
//! Version A (baseline): Multi-encoder approach — 4 compute encoders in
//!   1 command buffer. Known correct from exp8.
//!
//! Version B: Single-dispatch with `device volatile uint*` for buf_a/buf_b.
//!   volatile prevents compiler/hardware read caching.
//!
//! Version C: Single-dispatch with plain `device uint*` (non-volatile).
//!   Expected to reproduce exp8's stale-read bug.

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
struct Exp9Params {
    element_count: u32,
    num_tiles: u32,
    num_tgs: u32,
    num_passes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp9PersistParams {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    num_tgs: u32,
    counter_base: u32,
    ts_offset: u32,
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 9: Single-Dispatch Coherence Fixes");
    println!("{}", "=".repeat(60));
    println!("Tests: does `volatile` fix stale reads in single-dispatch megasort?");
    println!(
        "Setup: {}M elements, {} tiles, {} persistent TGs",
        N / 1_000_000,
        NUM_TILES,
        NUM_TGS
    );
    println!(
        "A: 4 encoders/1 cmdbuf (baseline)  B: volatile single-dispatch  C: plain single-dispatch\n"
    );

    // Pipelines
    let pso_persistent = ctx.make_pipeline("exp9_persistent_pass");
    let pso_volatile = ctx.make_pipeline("exp9_megasort_volatile");
    let pso_plain = ctx.make_pipeline("exp9_megasort_plain");

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

    // ── Version A: Multi-encoder baseline (4 persistent dispatches in 1 cmdbuf) ──

    let ts_section = NUM_TILES * NUM_BINS;
    let buf_a1 = alloc_buffer(&ctx.device, N * 4);
    let buf_a2 = alloc_buffer(&ctx.device, N * 4);
    // 4 passes x (num_tiles * num_bins) tile_status entries
    let buf_a_tile_status = alloc_buffer(&ctx.device, NUM_PASSES * ts_section * 4);
    // 4 passes x 3 counters (ws1, ws2, barrier) = 12 slots, allocate 16 for alignment
    let buf_a_counters = alloc_buffer(&ctx.device, 16 * 4);

    let mut times_a = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        // Copy input to buf_a1
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_a1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
        }

        // Clear tile_status and counters
        unsafe {
            let ptr = buf_a_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_a_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 16 * 4);
        }

        // Single command buffer, 4 compute encoders (one per pass)
        let cmd = ctx.command_buffer();
        for pass in 0..NUM_PASSES {
            let shift = (pass * 8) as u32;
            let params = Exp9PersistParams {
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
                &pso_persistent,
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

    // 4 passes: last pass index=3 (odd) -> dst=buf_a1
    let results_a: Vec<u32> = unsafe { read_buffer_slice(&buf_a1, N) };
    let correct_a = results_a == expected;

    // ── Version B: Single-dispatch volatile ──────────────────────────

    let buf_b1 = alloc_buffer(&ctx.device, N * 4);
    let buf_b2 = alloc_buffer(&ctx.device, N * 4);
    // tile_status: 4 passes x (num_tiles * num_bins) — each pass uses own section
    let buf_b_tile_status = alloc_buffer(&ctx.device, NUM_PASSES * ts_section * 4);
    // counters: [0..7] work-steal (pass*2), [8..21] barriers (up to 14)
    // 4 passes x 3 barriers_between_passes(max) = up to ~10 barriers + 8 ws = 22
    // Allocate 32 slots for safety
    let buf_b_counters = alloc_buffer(&ctx.device, 32 * 4);

    // ── Pass-by-pass correctness check for Version B ──
    println!("  Version B (volatile) pass-by-pass correctness:");
    for test_passes in 1..=4u32 {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_b1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
            let ptr = buf_b_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_b_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 32 * 4);
        }

        let mega_params = Exp9Params {
            element_count: N as u32,
            num_tiles: NUM_TILES as u32,
            num_tgs: NUM_TGS as u32,
            num_passes: test_passes,
        };
        let buf_params = alloc_buffer_with_data(&ctx.device, &[mega_params]);

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc,
            &pso_volatile,
            &[
                (buf_b1.as_ref(), 0),
                (buf_b2.as_ref(), 1),
                (buf_b_tile_status.as_ref(), 2),
                (buf_b_counters.as_ref(), 3),
                (buf_params.as_ref(), 4),
            ],
            NUM_TGS,
            TILE_SIZE,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        // After test_passes: last pass index = (test_passes-1)
        // Even pass writes to buf_b2, odd pass writes to buf_b1
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
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_b1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
            let ptr = buf_b_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_b_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 32 * 4);
        }

        let mega_params = Exp9Params {
            element_count: N as u32,
            num_tiles: NUM_TILES as u32,
            num_tgs: NUM_TGS as u32,
            num_passes: NUM_PASSES as u32,
        };
        let buf_params = alloc_buffer_with_data(&ctx.device, &[mega_params]);

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc,
            &pso_volatile,
            &[
                (buf_b1.as_ref(), 0),
                (buf_b2.as_ref(), 1),
                (buf_b_tile_status.as_ref(), 2),
                (buf_b_counters.as_ref(), 3),
                (buf_params.as_ref(), 4),
            ],
            NUM_TGS,
            TILE_SIZE,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let elapsed = gpu_elapsed_ms(&cmd);

        if iter >= WARMUP {
            times_b.push(elapsed);
        }
    }

    // 4 passes, last pass odd -> result in buf_b1
    let results_b: Vec<u32> = unsafe { read_buffer_slice(&buf_b1, N) };
    let correct_b = results_b == expected;

    // ── Version C: Single-dispatch plain (expected to fail) ──────────

    let buf_c1 = alloc_buffer(&ctx.device, N * 4);
    let buf_c2 = alloc_buffer(&ctx.device, N * 4);
    let buf_c_tile_status = alloc_buffer(&ctx.device, NUM_PASSES * ts_section * 4);
    let buf_c_counters = alloc_buffer(&ctx.device, 32 * 4);

    // ── Pass-by-pass correctness check for Version C ──
    println!("  Version C (plain) pass-by-pass correctness:");
    for test_passes in 1..=4u32 {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_c1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
            let ptr = buf_c_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_c_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 32 * 4);
        }

        let mega_params = Exp9Params {
            element_count: N as u32,
            num_tiles: NUM_TILES as u32,
            num_tgs: NUM_TGS as u32,
            num_passes: test_passes,
        };
        let buf_params = alloc_buffer_with_data(&ctx.device, &[mega_params]);

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc,
            &pso_plain,
            &[
                (buf_c1.as_ref(), 0),
                (buf_c2.as_ref(), 1),
                (buf_c_tile_status.as_ref(), 2),
                (buf_c_counters.as_ref(), 3),
                (buf_params.as_ref(), 4),
            ],
            NUM_TGS,
            TILE_SIZE,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let result_buf = if (test_passes - 1) % 2 == 0 {
            buf_c2.as_ref()
        } else {
            buf_c1.as_ref()
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

    // ── Benchmark Version C ──
    let mut times_c = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf_c1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
            let ptr = buf_c_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_c_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 32 * 4);
        }

        let mega_params = Exp9Params {
            element_count: N as u32,
            num_tiles: NUM_TILES as u32,
            num_tgs: NUM_TGS as u32,
            num_passes: NUM_PASSES as u32,
        };
        let buf_params = alloc_buffer_with_data(&ctx.device, &[mega_params]);

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc,
            &pso_plain,
            &[
                (buf_c1.as_ref(), 0),
                (buf_c2.as_ref(), 1),
                (buf_c_tile_status.as_ref(), 2),
                (buf_c_counters.as_ref(), 3),
                (buf_params.as_ref(), 4),
            ],
            NUM_TGS,
            TILE_SIZE,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let elapsed = gpu_elapsed_ms(&cmd);

        if iter >= WARMUP {
            times_c.push(elapsed);
        }
    }

    // 4 passes, last pass odd -> result in buf_c1
    let results_c: Vec<u32> = unsafe { read_buffer_slice(&buf_c1, N) };
    let correct_c = results_c == expected;

    // ── Results ────────────────────────────────────────────────────

    let med_a = median(&mut times_a);
    let med_b = median(&mut times_b);
    let med_c = median(&mut times_c);

    let throughput_a = N as f64 / med_a / 1000.0;
    let throughput_b = N as f64 / med_b / 1000.0;
    let throughput_c = N as f64 / med_c / 1000.0;

    println!("Results (full 32-bit sort, {}M elements):", N / 1_000_000);
    println!(
        "  A (4 encoders, 1 cmdbuf):    {:.3} ms  ({:.0} Mkeys/s)  correct: {}",
        med_a, throughput_a, correct_a
    );
    println!(
        "  B (single-dispatch volatile): {:.3} ms  ({:.0} Mkeys/s)  correct: {}",
        med_b, throughput_b, correct_b
    );
    println!(
        "  C (single-dispatch plain):    {:.3} ms  ({:.0} Mkeys/s)  correct: {}",
        med_c, throughput_c, correct_c
    );
    println!();

    if correct_b {
        println!(
            "  volatile FIXES stale reads! Speedup vs multi-encoder: {:.2}x",
            med_a / med_b
        );
    } else {
        println!("  volatile did NOT fix stale reads.");
    }

    if correct_c {
        println!(
            "  plain ALSO works! mem_device barrier is sufficient. Speedup: {:.2}x",
            med_a / med_c
        );
    } else {
        println!(
            "  plain FAILS as expected — confirms exp8's stale-read bug."
        );
    }

    if correct_b && correct_c {
        println!("  Both single-dispatch variants correct!");
        println!(
            "  volatile overhead: {:.3} ms ({:.1}%)",
            med_b - med_c,
            (med_b - med_c) / med_c * 100.0
        );
    } else if correct_b && !correct_c {
        println!("  volatile is necessary for cross-pass coherence in single-dispatch.");
        println!("  mem_flags::mem_device alone is insufficient for read cache invalidation.");
    }
    println!();

    // Dispatch count comparison
    println!("  Dispatch analysis:");
    println!("    A: 4 dispatches (4 compute encoders, 1 command buffer)");
    println!("    B: 1 dispatch  (volatile data buffers, 1 compute encoder)");
    println!("    C: 1 dispatch  (plain data buffers, 1 compute encoder)");

    if !correct_a {
        println!("\n  WARNING: Version A INCORRECT!");
        show_mismatch(&results_a, &expected);
    }
    if !correct_b {
        println!("\n  WARNING: Version B INCORRECT!");
        show_mismatch(&results_b, &expected);
    }
    if !correct_c {
        println!("\n  Version C details (expected failure):");
        show_mismatch(&results_c, &expected);
    }
}

fn show_mismatch(got: &[u32], expected: &[u32]) {
    let mismatches = got
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "    Mismatched: {} / {} ({:.1}%)",
        mismatches,
        got.len(),
        mismatches as f64 / got.len() as f64 * 100.0
    );

    let mut shown = 0;
    for i in 0..got.len() {
        if got[i] != expected[i] {
            println!("    [{}]: got {}, expected {}", i, got[i], expected[i]);
            shown += 1;
            if shown >= 5 {
                break;
            }
        }
    }

    let sorted = got.windows(2).all(|w| w[0] <= w[1]);
    println!("    Output sorted: {}", sorted);

    let mut got_sorted = got.to_vec();
    got_sorted.sort();
    let mut exp_sorted = expected.to_vec();
    exp_sorted.sort();
    println!("    Valid permutation: {}", got_sorted == exp_sorted);
}
