//! Experiment 2: Texture Bilinear Interpolation as Free ALU
//!
//! Tests whether hardware bilinear filtering can replace ALU-computed
//! LUT interpolation — effectively getting interpolation "for free"
//! from the texture unit while ALUs do other work.

use crate::metal_ctx::*;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState};

const N: usize = 1_000_000;
const LUT_SIZE: usize = 256;
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
    println!("Experiment 2: Texture Bilinear as Free ALU");
    println!("{}", "=".repeat(60));
    println!("Hypothesis: HW bilinear LUT lookup is faster than manual lerp");
    println!("Setup: {} elements, {}-entry LUT (sigmoid)\n", N, LUT_SIZE);

    let pso_manual = ctx.make_pipeline("exp2_manual_lerp");
    let pso_hw = ctx.make_pipeline("exp2_hw_bilinear");

    // Build a sigmoid LUT
    let lut: Vec<f32> = (0..LUT_SIZE)
        .map(|i| {
            let x = (i as f32 / (LUT_SIZE - 1) as f32) * 12.0 - 6.0; // range [-6, 6]
            1.0 / (1.0 + (-x).exp())
        })
        .collect();
    let buf_lut = alloc_buffer_with_data(&ctx.device, &lut);

    // Random input values in [0, 1]
    let input: Vec<f32> = (0..N)
        .map(|i| ((i as f32 * 0.618033) % 1.0).abs())
        .collect();
    let buf_input = alloc_buffer_with_data(&ctx.device, &input);
    let buf_output = alloc_buffer(&ctx.device, N * std::mem::size_of::<f32>());

    let params = ExpParams {
        element_count: N as u32,
        num_passes: 0,
        mode: 0,
    };
    let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

    // Create texture LUT (256×1 R32Float) for HW bilinear
    let texture = create_texture_r32f(&ctx.device, LUT_SIZE, 1, &lut);
    let sampler = create_linear_sampler(&ctx.device);

    // --- Version A: Manual lerp (2 buffer reads + ALU lerp) ---
    let mut times_a = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            &pso_manual,
            &[
                (buf_lut.as_ref(), 0),
                (buf_input.as_ref(), 1),
                (buf_output.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            N,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if i >= WARMUP {
            times_a.push(gpu_elapsed_ms(&cmd));
        }
    }

    // --- Version B: HW bilinear (1 texture sample) ---
    let mut times_b = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        enc.setComputePipelineState(&pso_hw);
        unsafe {
            enc.setTexture_atIndex(Some(texture.as_ref()), 0);
            enc.setSamplerState_atIndex(Some(sampler.as_ref()), 0);
            enc.setBuffer_offset_atIndex(Some(buf_input.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_output.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_params.as_ref()), 0, 2);
        }

        let tg = pso_hw.maxTotalThreadsPerThreadgroup().min(256);
        let groups = N.div_ceil(tg);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            objc2_metal::MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if i >= WARMUP {
            times_b.push(gpu_elapsed_ms(&cmd));
        }
    }

    let med_a = median(&mut times_a);
    let med_b = median(&mut times_b);
    let mops_a = N as f64 / med_a / 1000.0;
    let mops_b = N as f64 / med_b / 1000.0;

    println!("Results:");
    println!("  Manual lerp:       {:.3} ms  ({:.0} Mops)", med_a, mops_a);
    println!("  HW bilinear:       {:.3} ms  ({:.0} Mops)", med_b, mops_b);
    println!("  Speedup:           {:.2}x", med_a / med_b);
    println!();
    if med_b < med_a * 0.9 {
        println!("  ** HW bilinear is faster — texture unit handles interpolation **");
    } else if med_a < med_b * 0.9 {
        println!("  Manual lerp wins — texture sampling has overhead at this scale");
    } else {
        println!("  Roughly equivalent — both paths saturate at 1M elements");
    }
}
