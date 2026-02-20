//! Experiment 11: Radix Sort Scale Test
//!
//! Runs the persistent radix sort at 1M, 2M, and 4M elements to measure
//! throughput scaling. Uses both serial and SIMD rank kernels from exp10.
//!
//! Key question: does throughput improve with larger N (better GPU utilization)?

use crate::metal_ctx::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

const TILE_SIZE: usize = 256;
const NUM_BINS: usize = 256;
const NUM_TGS: usize = 64;
const NUM_PASSES: usize = 4;
const WARMUP: usize = 5;
const RUNS: usize = 20;

#[repr(C)]
#[derive(Clone, Copy)]
struct ScaleParams {
    element_count: u32,
    num_tiles: u32,
    shift: u32,
    num_tgs: u32,
    counter_base: u32,
    ts_offset: u32,
}

fn bench_at_scale(
    ctx: &MetalContext,
    n: usize,
    pso: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,
    label: &str,
) -> (f64, bool) {
    let num_tiles = n / TILE_SIZE;
    let ts_section = num_tiles * NUM_BINS;

    // Random input
    let input: Vec<u32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<u32>()).collect()
    };
    let buf_input = alloc_buffer_with_data(&ctx.device, &input);

    let mut expected = input.clone();
    expected.sort();

    let buf1 = alloc_buffer(&ctx.device, n * 4);
    let buf2 = alloc_buffer(&ctx.device, n * 4);
    let buf_tile_status = alloc_buffer(&ctx.device, NUM_PASSES * ts_section * 4);
    let buf_counters = alloc_buffer(&ctx.device, 16 * 4);

    let mut times = Vec::new();
    for iter in 0..(WARMUP + RUNS) {
        unsafe {
            let src = buf_input.contents().as_ptr() as *const u8;
            let dst = buf1.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, n * 4);
            let ptr = buf_tile_status.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, NUM_PASSES * ts_section * 4);
            let ptr = buf_counters.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, 16 * 4);
        }

        let cmd = ctx.command_buffer();
        for pass in 0..NUM_PASSES {
            let shift = (pass * 8) as u32;
            let params = ScaleParams {
                element_count: n as u32,
                num_tiles: num_tiles as u32,
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

    // 4 passes, last pass index=3 (odd) -> result in buf1
    let results: Vec<u32> = unsafe { read_buffer_slice(&buf1, n) };
    let correct = results == expected;

    let med = median(&mut times);
    let throughput = n as f64 / med / 1000.0;

    println!(
        "  {:>10} {:>5}  {:.3} ms  {:>6.0} Mkeys/s  correct: {}",
        label,
        format!("{}M", n / 1_000_000),
        med,
        throughput,
        correct
    );

    (med, correct)
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 11: Radix Sort Scale Test");
    println!("{}", "=".repeat(60));
    println!("Persistent radix sort throughput at 1M, 2M, 4M elements");
    println!("Both serial and SIMD rank kernels tested\n");

    let pso_serial = ctx.make_pipeline("exp10_serial_pass");
    let pso_simd = ctx.make_pipeline("exp10_simd_pass");

    let sizes = [1_048_576, 2_097_152, 4_194_304]; // 1M, 2M, 4M

    println!("  {:>10} {:>5}  {:>8}  {:>14}  {}", "kernel", "size", "time", "throughput", "correct");
    println!("  {}", "-".repeat(55));

    let mut serial_results = Vec::new();
    let mut simd_results = Vec::new();

    for &n in &sizes {
        let (med, correct) = bench_at_scale(ctx, n, &pso_serial, "serial");
        serial_results.push((n, med, correct));
    }

    println!();

    for &n in &sizes {
        let (med, correct) = bench_at_scale(ctx, n, &pso_simd, "SIMD");
        simd_results.push((n, med, correct));
    }

    // Summary
    println!("\n  Scale analysis:");
    for i in 0..sizes.len() {
        let n = sizes[i];
        let (_, s_med, _) = serial_results[i];
        let (_, m_med, _) = simd_results[i];
        let speedup = s_med / m_med;
        println!(
            "    {}M: SIMD {:.2}x faster than serial",
            n / 1_000_000,
            speedup
        );
    }

    // Throughput scaling
    println!("\n  Throughput scaling (SIMD):");
    let base_throughput = sizes[0] as f64 / simd_results[0].1 / 1000.0;
    for i in 0..sizes.len() {
        let n = sizes[i];
        let throughput = n as f64 / simd_results[i].1 / 1000.0;
        println!(
            "    {}M: {:.0} Mkeys/s ({:.1}x vs 1M)",
            n / 1_000_000,
            throughput,
            throughput / base_throughput
        );
    }
}
