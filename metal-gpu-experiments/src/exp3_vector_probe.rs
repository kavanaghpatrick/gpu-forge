//! Experiment 3: Vectorized Hash Probe (Batch-4 vs Scalar)
//!
//! Tests whether pre-loading 4 consecutive hash table entries per loop
//! iteration reduces branch overhead and hides pipeline latency vs.
//! the standard one-entry-per-iteration approach.

use crate::metal_ctx::*;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};

const NUM_KEYS: usize = 1_000_000;
const TABLE_CAP: usize = 2_097_152; // 2M slots, 50% load factor
const EMPTY_SLOT: u32 = 0xFFFFFFFF;
const WARMUP: usize = 3;
const RUNS: usize = 10;

#[repr(C)]
#[derive(Clone, Copy)]
struct ExpParams {
    element_count: u32,
    num_passes: u32, // table capacity
    mode: u32,
}

fn murmur3_finalize(mut key: u32) -> u32 {
    key ^= key >> 16;
    key = key.wrapping_mul(0x85ebca6b);
    key ^= key >> 13;
    key = key.wrapping_mul(0xc2b2ae35);
    key ^= key >> 16;
    key
}

pub fn run(ctx: &MetalContext) {
    println!("\n{}", "=".repeat(60));
    println!("Experiment 3: Vectorized Hash Probe (Batch-4 vs Scalar)");
    println!("{}", "=".repeat(60));
    println!("Hypothesis: Loading 4 entries/iteration hides latency");
    println!(
        "Setup: {}M keys, {}M capacity (50% load)\n",
        NUM_KEYS / 1_000_000,
        TABLE_CAP / 1_000_000
    );

    let pso_scalar = ctx.make_pipeline("exp3_scalar_probe");
    let pso_batch4 = ctx.make_pipeline("exp3_batch4_probe");

    // Build hash table (AoS: uint2 = {key, value})
    let mut table = vec![[EMPTY_SLOT, EMPTY_SLOT]; TABLE_CAP];
    let mask = (TABLE_CAP - 1) as u32;

    // Generate unique keys and insert into table
    let mut keys = Vec::with_capacity(NUM_KEYS);
    let mut key: u32 = 1;
    for i in 0..NUM_KEYS {
        // Generate unique keys avoiding EMPTY_SLOT
        while key == EMPTY_SLOT {
            key = key.wrapping_add(1);
        }
        let value = i as u32;

        // Insert via linear probing
        let mut slot = murmur3_finalize(key) & mask;
        for _ in 0..64 {
            if table[slot as usize][0] == EMPTY_SLOT {
                table[slot as usize] = [key, value];
                break;
            }
            slot = (slot + 1) & mask;
        }

        keys.push(key);
        key = key.wrapping_add(1);
    }

    let buf_table = alloc_buffer_with_data(&ctx.device, &table);
    let buf_queries = alloc_buffer_with_data(&ctx.device, &keys);
    let buf_output = alloc_buffer(&ctx.device, NUM_KEYS * std::mem::size_of::<u32>());
    let params = ExpParams {
        element_count: NUM_KEYS as u32,
        num_passes: TABLE_CAP as u32,
        mode: 0,
    };
    let buf_params = alloc_buffer_with_data(&ctx.device, &[params]);

    // --- Version A: Scalar probe ---
    let mut times_a = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            &pso_scalar,
            &[
                (buf_table.as_ref(), 0),
                (buf_queries.as_ref(), 1),
                (buf_output.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            NUM_KEYS,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if i >= WARMUP {
            times_a.push(gpu_elapsed_ms(&cmd));
        }
    }

    // Verify correctness of scalar version
    let results_a: Vec<u32> = unsafe { read_buffer_slice(&buf_output, NUM_KEYS) };
    let hits_a = results_a.iter().filter(|&&v| v != EMPTY_SLOT).count();

    // --- Version B: Batch-4 probe ---
    let mut times_b = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        let cmd = ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();
        dispatch_1d(
            &enc,
            &pso_batch4,
            &[
                (buf_table.as_ref(), 0),
                (buf_queries.as_ref(), 1),
                (buf_output.as_ref(), 2),
                (buf_params.as_ref(), 3),
            ],
            NUM_KEYS,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if i >= WARMUP {
            times_b.push(gpu_elapsed_ms(&cmd));
        }
    }

    let results_b: Vec<u32> = unsafe { read_buffer_slice(&buf_output, NUM_KEYS) };
    let hits_b = results_b.iter().filter(|&&v| v != EMPTY_SLOT).count();

    let med_a = median(&mut times_a);
    let med_b = median(&mut times_b);
    let mops_a = NUM_KEYS as f64 / med_a / 1000.0;
    let mops_b = NUM_KEYS as f64 / med_b / 1000.0;

    println!("Results:");
    println!(
        "  Scalar probe:      {:.3} ms  ({:.0} Mops)  hits: {}/{}",
        med_a, mops_a, hits_a, NUM_KEYS
    );
    println!(
        "  Batch-4 probe:     {:.3} ms  ({:.0} Mops)  hits: {}/{}",
        med_b, mops_b, hits_b, NUM_KEYS
    );
    println!("  Speedup:           {:.2}x", med_a / med_b);

    // Verify both produce same results
    let agree = results_a
        .iter()
        .zip(results_b.iter())
        .all(|(a, b)| a == b);
    println!("  Results match:     {}", if agree { "YES" } else { "NO" });
}
