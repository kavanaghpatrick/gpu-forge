//! Experiment 12: Cross-TG Coherence Strategy Benchmark
//!
//! Tests Metal 3.2 coherence mechanisms for fixing the cross-threadgroup
//! stale-read bug in single-dispatch persistent radix sort.
//!
//! Version A: Multi-encoder baseline (4 encoders, 1 cmdbuf) — known correct
//! Version B: Single dispatch + coherent(device) + atomic_thread_fence
//! Version C: Single dispatch + atomic_thread_fence only (no coherent)
//! Version D: Single dispatch + coherent(device) only (no fence)

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
struct Exp12PassParams {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    num_tgs: u32,
    counter_base: u32,
    ts_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Exp12Params {
    element_count: u32,
    num_tiles: u32,
    num_tgs: u32,
    num_passes: u32,
    mode: u32, // 0 = use fence, 1 = no fence
}

/// Run the multi-encoder baseline (Version A).
/// 4 separate compute encoders in 1 command buffer.
fn run_multi_encoder(
    ctx: &MetalContext,
    pso: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    input: &[u32],
    expected: &[u32],
) -> (f64, bool) {
    let ts_section = NUM_TILES * NUM_BINS;

    let buf_input = alloc_buffer_with_data(&ctx.device, input);
    let buf1 = alloc_buffer(&ctx.device, N * 4);
    let buf2 = alloc_buffer(&ctx.device, N * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, NUM_PASSES * ts_section * 4);
    let buf_counters = alloc_buffer(&ctx.device, 16 * 4);

    let mut times = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
            let ptr = buf_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 16 * 4);
        }

        let cmd = ctx.command_buffer();
        for pass in 0..NUM_PASSES {
            let shift = (pass * 8) as u32;
            let params = Exp12PassParams {
                element_count: N as u32,
                num_tiles: NUM_TILES as u32,
                shift,
                num_tgs: NUM_TGS as u32,
                counter_base: (pass * 3) as u32,
                ts_offset: (pass * ts_section) as u32,
            };
            let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

            let (src, dst) = if pass % 2 == 0 {
                (buf1.as_ref(), buf2.as_ref())
            } else {
                (buf2.as_ref(), buf1.as_ref())
            };

            let enc = cmd.computeCommandEncoder().unwrap();
            dispatch_1d_tg(
                &enc,
                pso,
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
            times.push(elapsed);
        }
    }

    // 4 passes: last pass index=3 (odd) -> result in buf1
    let results: Vec<u32> = unsafe { read_buffer_slice(&buf1, N) };
    let correct = results == expected;
    let med = median(&mut times);
    (med, correct)
}

/// Run a single-dispatch megasort variant (Versions B, C, D).
fn run_single_dispatch(
    ctx: &MetalContext,
    pso: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    input: &[u32],
    expected: &[u32],
    mode: u32, // 0 = fence, 1 = no fence
) -> (f64, bool) {
    let ts_section = NUM_TILES * NUM_BINS;

    let buf_input = alloc_buffer_with_data(&ctx.device, input);
    let buf1 = alloc_buffer(&ctx.device, N * 4);
    let buf2 = alloc_buffer(&ctx.device, N * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, NUM_PASSES * ts_section * 4);
    // Megasort needs more counters: [0..7] work-steal, [8..21] barriers
    let buf_counters = alloc_buffer(&ctx.device, 32 * 4);

    let params = Exp12Params {
        element_count: N as u32,
        num_tiles: NUM_TILES as u32,
        num_tgs: NUM_TGS as u32,
        num_passes: NUM_PASSES as u32,
        mode,
    };
    let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

    // First: correctness check (single run)
    unsafe {
        let src = buf_input.contents().as_ptr() as *const u8;
        let dst = buf1.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src, dst, N * 4);
        let ptr = buf_tile_status.contents().as_ptr() as *mut u8;
        std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
        let ptr = buf_counters.contents().as_ptr() as *mut u8;
        std::ptr::write_bytes(ptr, 0, 32 * 4);
    }

    let cmd = ctx.command_buffer();
    let enc = cmd.computeCommandEncoder().unwrap();
    dispatch_1d_tg(
        &enc,
        pso,
        &[
            (buf1.as_ref(), 0),
            (buf2.as_ref(), 1),
            (buf_tile_status.as_ref(), 2),
            (buf_counters.as_ref(), 3),
            (buf_params.as_ref(), 4),
        ],
        NUM_TGS,
        TILE_SIZE,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    // 4 passes: last pass index=3, even passes write to buf2, odd to buf1
    // Pass 0: buf1→buf2, Pass 1: buf2→buf1, Pass 2: buf1→buf2, Pass 3: buf2→buf1
    // Result is in buf1 after 4 passes
    let results: Vec<u32> = unsafe { read_buffer_slice(&buf1, N) };
    let correct = results == expected;

    if !correct {
        // Show diagnostic info
        let mismatches = results
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a != b)
            .count();
        // Check if it's still a valid permutation
        let mut sorted_check = results.clone();
        sorted_check.sort();
        let is_perm = sorted_check == *expected;
        println!(
            "      Mismatched: {} / {} ({:.1}%), is_permutation: {}",
            mismatches,
            N,
            mismatches as f64 / N as f64 * 100.0,
            is_perm
        );

        // Benchmark anyway (even if incorrect, timing is still informative)
    }

    // Benchmark
    let mut times = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, N * 4);
            let ptr = buf_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 32 * 4);
        }

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d_tg(
            &enc,
            pso,
            &[
                (buf1.as_ref(), 0),
                (buf2.as_ref(), 1),
                (buf_tile_status.as_ref(), 2),
                (buf_counters.as_ref(), 3),
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
            times.push(elapsed);
        }
    }

    let med = median(&mut times);
    (med, correct)
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 12: Cross-TG Coherence Strategy Benchmark");
    println!("{}", "=".repeat(60));
    println!("Metal 3.2 coherence mechanisms for single-dispatch radix sort");
    println!(
        "Setup: {}M elements, {} tiles, {} persistent TGs\n",
        N / 1_000_000,
        NUM_TILES,
        NUM_TGS
    );

    // Pipelines
    let pso_pass = ctx.make_pipeline("exp12_pass");
    let pso_coherent = ctx.make_pipeline("exp12_megasort_coherent");
    let pso_fence = ctx.make_pipeline("exp12_megasort_fence");

    // Random input
    let input: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..N).map(|_| rng.gen::<u32>()).collect()
    };

    // CPU reference sort
    let mut expected = input.clone();
    expected.sort();

    // ── Version A: Multi-encoder baseline ─────────────────────────────
    print!("  A (multi-encoder):       ");
    let (med_a, correct_a) = run_multi_encoder(ctx, &pso_pass, &input, &expected);
    let throughput_a = N as f64 / med_a / 1000.0;
    println!(
        "{:.3} ms  {:>6.0} Mkeys/s  correct: {}",
        med_a, throughput_a, correct_a
    );

    // ── Version B: coherent(device) + fence ───────────────────────────
    print!("  B (coherent + fence):    ");
    let (med_b, correct_b) = run_single_dispatch(ctx, &pso_coherent, &input, &expected, 0);
    let throughput_b = N as f64 / med_b / 1000.0;
    println!(
        "{:.3} ms  {:>6.0} Mkeys/s  correct: {}",
        med_b, throughput_b, correct_b
    );

    // ── Version C: fence only (no coherent) ───────────────────────────
    print!("  C (fence only):          ");
    let (med_c, correct_c) = run_single_dispatch(ctx, &pso_fence, &input, &expected, 0);
    let throughput_c = N as f64 / med_c / 1000.0;
    println!(
        "{:.3} ms  {:>6.0} Mkeys/s  correct: {}",
        med_c, throughput_c, correct_c
    );

    // ── Version D: coherent(device) only (no fence) ──────────────────
    print!("  D (coherent only):       ");
    let (med_d, correct_d) = run_single_dispatch(ctx, &pso_coherent, &input, &expected, 1);
    let throughput_d = N as f64 / med_d / 1000.0;
    println!(
        "{:.3} ms  {:>6.0} Mkeys/s  correct: {}",
        med_d, throughput_d, correct_d
    );

    // ── Summary ──────────────────────────────────────────────────────
    println!("\n  Summary:");
    println!("  {:>25}  {:>8}  {:>12}  {:>7}", "version", "time", "throughput", "correct");
    println!("  {}", "-".repeat(58));

    let versions = [
        ("A multi-encoder", med_a, throughput_a, correct_a),
        ("B coherent+fence", med_b, throughput_b, correct_b),
        ("C fence only", med_c, throughput_c, correct_c),
        ("D coherent only", med_d, throughput_d, correct_d),
    ];

    for (name, med, tput, correct) in &versions {
        let speedup = med_a / med;
        println!(
            "  {:>25}  {:.3} ms  {:>6.0} Mkeys/s  {:>5}  ({:.2}x vs A)",
            name, med, tput, correct, speedup
        );
    }

    println!("\n  Interpretation:");
    if correct_b && correct_c {
        println!("    Both coherent and fence work independently → fence alone sufficient");
    } else if correct_b && !correct_c {
        println!("    coherent(device) required for cross-TG visibility");
        if correct_d {
            println!("    coherent alone works — fence is redundant");
        } else {
            println!("    Both coherent AND fence needed together");
        }
    } else if !correct_b && correct_c {
        println!("    fence alone works without coherent — surprising!");
    } else if !correct_b && !correct_c && !correct_d {
        println!("    None of the single-dispatch approaches work");
        println!("    Multi-encoder remains the only viable approach");
    }
}
