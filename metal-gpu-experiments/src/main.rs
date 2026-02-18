mod metal_ctx;
mod exp1_texture_l1;
mod exp2_texture_alu;
mod exp3_vector_probe;
mod exp4_decoupled_scan;
mod exp5_megakernel;
mod exp6_stream_compact;

use objc2_metal::MTLDevice;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Metal GPU Experiments — Apple M4 Pro            ║");
    println!("║         Novel uses of Metal APIs on Apple Silicon       ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let ctx = metal_ctx::MetalContext::new();

    println!("\nDevice: {}", ctx.device.name());

    exp1_texture_l1::run(&ctx);
    exp2_texture_alu::run(&ctx);
    exp3_vector_probe::run(&ctx);
    exp4_decoupled_scan::run(&ctx);
    exp5_megakernel::run(&ctx);
    exp6_stream_compact::run(&ctx);

    println!("\n{}", "=".repeat(60));
    println!("All experiments complete.");
    println!("{}", "=".repeat(60));
}
