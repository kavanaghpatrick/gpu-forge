//! Experiment 1: Texture L1 Cache Doubling
//!
//! Tests whether texture unit L1 and shader core L1 are separate caches.
//! If so, reading data through both paths gives ~2x effective L1 bandwidth.

use crate::metal_ctx::*;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState};

const DATA_SIZE: usize = 4096; // 16KB of floats — fits in L1
const NUM_THREADS: usize = 65536;
const NUM_PASSES: u32 = 256;
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
    println!("Experiment 1: Texture L1 Cache Doubling");
    println!("{}", "=".repeat(60));
    println!("Hypothesis: Separate texture/buffer L1 caches → 2x bandwidth");
    println!("Setup: {}KB dataset, {} threads, {} passes each\n", DATA_SIZE * 4 / 1024, NUM_THREADS, NUM_PASSES);

    let pso_buf = ctx.make_pipeline("exp1_buffer_only");
    let pso_dual = ctx.make_pipeline("exp1_dual_read");

    // Create data buffer (L1-sized)
    let data: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
    let buf_data = alloc_buffer_with_data(&ctx.device, &data);
    let buf_output = alloc_buffer(&ctx.device, NUM_THREADS * std::mem::size_of::<f32>());
    let params = ExpParams {
        element_count: DATA_SIZE as u32,
        num_passes: NUM_PASSES,
        mode: 0,
    };
    let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

    // Create texture with same data (64×64 = 4096)
    let tex_w = 64usize;
    let tex_h = 64usize;
    let texture = create_texture_r32f(&ctx.device, tex_w, tex_h, &data);

    // --- Version A: Buffer-only reads ---
    let mut times_a = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            &pso_buf,
            &[
                (buf_data.as_ref(), 0),
                (buf_output.as_ref(), 1),
                (buf_params.as_ref(), 2),
            ],
            NUM_THREADS,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if i >= WARMUP {
            times_a.push(gpu_elapsed_ms(&cmd));
        }
    }

    // --- Version B: Dual texture+buffer reads ---
    let mut times_b = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        enc.setComputePipelineState(&pso_dual);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_data.as_ref()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_output.as_ref()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(buf_params.as_ref()), 0, 2);
            enc.setTexture_atIndex(Some(texture.as_ref()), 0);
        }

        let tg = pso_dual.maxTotalThreadsPerThreadgroup().min(256);
        let groups = NUM_THREADS.div_ceil(tg);
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
    let total_reads = NUM_THREADS as f64 * NUM_PASSES as f64 * 4.0; // bytes
    let bw_a = total_reads / (med_a / 1000.0) / 1e9;
    let bw_b = total_reads / (med_b / 1000.0) / 1e9;

    println!("Results:");
    println!("  Buffer-only:       {:.3} ms  ({:.0} GB/s effective)", med_a, bw_a);
    println!("  Dual tex+buf:      {:.3} ms  ({:.0} GB/s effective)", med_b, bw_b);
    println!("  Speedup:           {:.2}x", med_a / med_b);
    println!();
    if med_b < med_a * 0.9 {
        println!("  ** CONFIRMED: Separate L1 caches give measurable bandwidth gain **");
    } else {
        println!("  No significant difference — L1 caches may share bandwidth");
    }
}
