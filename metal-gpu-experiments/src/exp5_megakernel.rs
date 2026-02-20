//! Experiment 5: Megakernel (Single Dispatch vs N Dispatches)
//!
//! Tests the overhead of N separate command buffer commits vs. a single
//! fused megakernel dispatch. Quantifies dispatch overhead + memory
//! bandwidth savings from kernel fusion.

use crate::metal_ctx::*;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState};

const N: usize = 1_000_000;
const NUM_OPS: usize = 10; // 10 operations per element
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
    println!("Experiment 5: Megakernel vs N Separate Dispatches");
    println!("{}", "=".repeat(60));
    println!("Hypothesis: Single fused dispatch eliminates dispatch overhead + extra DRAM passes");
    println!("Setup: {} elements, {} ops (scale/add/sqrt/negate repeated)\n", N, NUM_OPS);

    let pso_scale = ctx.make_pipeline("exp5_scale");
    let pso_add = ctx.make_pipeline("exp5_add");
    let pso_sqrt = ctx.make_pipeline("exp5_sqrt_op");
    let pso_negate = ctx.make_pipeline("exp5_negate");
    let pso_mega = ctx.make_pipeline("exp5_megakernel");

    // Scalars for the operations
    let scalars: [f32; 6] = [1.5, 0.3, 2.0, -0.7, 0.8, 1.1];
    let buf_scalars = alloc_buffer_with_data(&ctx.device, &scalars);

    let params = ExpParams {
        element_count: N as u32,
        num_passes: NUM_OPS as u32,
        mode: 0,
    };
    let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

    // Initialize source data
    let src_data: Vec<f32> = (0..N).map(|i| (i as f32 + 1.0) * 0.001).collect();

    // --- Version A: 10 separate command buffer dispatches ---
    let buf_data_a = alloc_buffer_with_data(&ctx.device, &src_data);
    let mut times_a = Vec::new();

    for i in 0..(WARMUP + RUNS) {
        // Reset data
        unsafe { write_buffer(&buf_data_a, &src_data) };

        let mut total_ms = 0.0;

        // 10 operations: scale, add, sqrt, negate, scale, add, sqrt, negate, scale, add
        let ops: Vec<(&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>, Option<usize>)> = vec![
            (&*pso_scale, Some(0)),
            (&*pso_add, Some(1)),
            (&*pso_sqrt, None),
            (&*pso_negate, None),
            (&*pso_scale, Some(2)),
            (&*pso_add, Some(3)),
            (&*pso_sqrt, None),
            (&*pso_negate, None),
            (&*pso_scale, Some(4)),
            (&*pso_add, Some(5)),
        ];

        for (pso, scalar_idx) in &ops {
            let cmd = ctx.command_buffer();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(*pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf_data_a.as_ref()), 0, 0);
            }

            if let Some(idx) = scalar_idx {
                // Scale/Add: buffer 1 = scalar (as setBytes-like via buffer)
                let scalar_val = scalars[*idx];
                let buf_scalar = alloc_buffer_with_data(&ctx.device, &[scalar_val]);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_scalar.as_ref()), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(buf_params.as_ref()), 0, 2);
                }
            } else {
                // Sqrt/Negate: buffer 1 = params
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(buf_params.as_ref()), 0, 1);
                }
            }

            let tg = pso.maxTotalThreadsPerThreadgroup().min(256);
            let groups = N.div_ceil(tg);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize { width: groups, height: 1, depth: 1 },
                objc2_metal::MTLSize { width: tg, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            total_ms += gpu_elapsed_ms(&cmd);
        }

        if i >= WARMUP {
            times_a.push(total_ms);
        }
    }

    // Read results from version A for comparison
    let results_a: Vec<f32> = unsafe { read_buffer_slice(&buf_data_a, N) };

    // --- Version B: Single megakernel dispatch ---
    let buf_data_b = alloc_buffer_with_data(&ctx.device, &src_data);
    let mut times_b = Vec::new();

    for i in 0..(WARMUP + RUNS) {
        unsafe { write_buffer(&buf_data_b, &src_data) };

        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            &pso_mega,
            &[
                (buf_data_b.as_ref(), 0),
                (buf_scalars.as_ref(), 1),
                (buf_params.as_ref(), 2),
            ],
            N,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if i >= WARMUP {
            times_b.push(gpu_elapsed_ms(&cmd));
        }
    }

    let results_b: Vec<f32> = unsafe { read_buffer_slice(&buf_data_b, N) };

    // Check if results are close (floating point may differ slightly due to fusion)
    let max_diff = results_a
        .iter()
        .zip(results_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let med_a = median(&mut times_a);
    let med_b = median(&mut times_b);
    let dispatch_overhead = med_a - med_b;

    println!("Results:");
    println!("  {} separate dispatches: {:.3} ms", NUM_OPS, med_a);
    println!("  1 megakernel dispatch:  {:.3} ms", med_b);
    println!("  Speedup:                {:.2}x", med_a / med_b);
    println!("  Dispatch overhead:      {:.3} ms ({:.1} us/dispatch)",
        dispatch_overhead, dispatch_overhead * 1000.0 / (NUM_OPS - 1) as f64);
    println!("  Max value diff:         {:.6}", max_diff);
    println!();
    if med_a / med_b > 2.0 {
        println!("  ** Massive win: kernel fusion eliminates dispatch + memory overhead **");
    } else if med_a / med_b > 1.3 {
        println!("  ** Significant win from kernel fusion **");
    } else {
        println!("  Modest improvement — dispatches are already fast on Apple Silicon");
    }
}
