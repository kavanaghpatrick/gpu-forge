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

    // Experiment 17: MSD+LSD hybrid radix sort
    exp17_hybrid::run(&ctx);

    println!("\n{}", "=".repeat(60));
    println!("All experiments complete.");
    println!("{}", "=".repeat(60));
}
