mod metal_ctx;
mod exp1_texture_l1;
mod exp2_texture_alu;
mod exp3_vector_probe;
mod exp4_decoupled_scan;
mod exp5_megakernel;
mod exp6_stream_compact;
mod exp7_radix_sort;
mod exp8_megasort;
mod exp9_sort_fixes;
mod exp10_simd_rank;
mod exp11_scale;
mod exp12_coherence;
mod exp13_ultimate_sort;
mod exp14_multi_dispatch;
mod exp15_onesweep;
mod exp16_8bit;
mod exp17_hybrid;
mod exp18_monster;
mod exp19_wlms;
mod exp20_fused_hybrid;
mod exp21_presort;
mod exp22_local_sort;
mod exp23_slc_sort;
mod exp24_batched_sort;
mod exp25_fence_free;
mod exp26_3pass;

use objc2_metal::MTLDevice;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Metal GPU Experiments — Apple M4 Pro            ║");
    println!("║         Novel uses of Metal APIs on Apple Silicon       ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let ctx = metal_ctx::MetalContext::new();

    println!("\nDevice: {}", ctx.device.name());

    // // Experiments 1-6 (original set)
    // exp1_texture_l1::run(&ctx);
    // exp2_texture_alu::run(&ctx);
    // exp3_vector_probe::run(&ctx);
    // exp4_decoupled_scan::run(&ctx);
    // exp5_megakernel::run(&ctx);
    // exp6_stream_compact::run(&ctx);

    // // Experiments 7-12 (radix sort series)
    // exp7_radix_sort::run(&ctx);
    // exp8_megasort::run(&ctx);
    // exp9_sort_fixes::run(&ctx);
    // exp10_simd_rank::run(&ctx);
    // exp11_scale::run(&ctx);
    // exp12_coherence::run(&ctx);

    // // Experiment 13: Ultimate sort (all optimizations combined)
    // exp13_ultimate_sort::run(&ctx);

    // // Experiment 14: Multi-dispatch sort (coalesced scatter)
    // exp14_multi_dispatch::run(&ctx);

    // // Experiment 15: Onesweep fused radix sort (baseline)
    // exp15_onesweep::run(&ctx);

    // // Experiment 16: 8-bit radix sort + 3-pass variant
    // exp16_8bit::run(&ctx);

    // Experiment 17: MSD+LSD hybrid — targeting 5000+ Mk/s
    exp17_hybrid::run(&ctx);

    // Experiment 18: Monster 3-pass radix sort — 1589 Mkeys/s
    // exp18_monster::run(&ctx);

    // Experiment 19: WLMS 3-pass radix sort (Plan A) — dead at 1048 Mkeys/s
    // exp19_wlms::run(&ctx);

    // Experiment 20: Fused hybrid (inner_partition replaces histogram+scan_scatter)
    // exp20_fused_hybrid::run(&ctx);

    // Experiment 21: Pre-sort scatter (Stehle-Jacobsen) — DEAD (4% slower)
    // exp21_presort::run(&ctx);

    // Experiment 22: Work-queue local sort (KB #3460) — dead at 1354 Mkeys/s
    // exp22_local_sort::run(&ctx);

    // Experiment 23: SLC-speed per-bucket global LSD sort — 2410 Mk/s (DRAM speed)
    // exp23_slc_sort::run(&ctx);

    // Experiment 24: Batched inner passes for SLC residency — DEAD (SLC scatter ≠ faster)
    // exp24_batched_sort::run(&ctx);

    // Experiment 25: Fence-free radix sort (precomputed tile prefix) — 2377 Mkeys/s
    // exp25_fence_free::run(&ctx);

    println!("\n{}", "=".repeat(60));
    println!("All experiments complete.");
    println!("{}", "=".repeat(60));
}
