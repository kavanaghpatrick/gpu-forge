use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("shaders");

    // Collect all .metal files in the shaders/ directory
    let metal_files: Vec<PathBuf> = fs::read_dir(shader_dir)
        .expect("Failed to read shaders/ directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "metal") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if metal_files.is_empty() {
        panic!("No .metal files found in shaders/ directory");
    }

    // Check if debug-telemetry feature is enabled via Cargo
    let debug_telemetry = env::var("CARGO_FEATURE_DEBUG_TELEMETRY").is_ok();

    // Compile each .metal file to .air (Apple Intermediate Representation)
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_file = out_dir.join(format!("{}.air", stem));

        let mut cmd = Command::new("xcrun");
        cmd.args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "-I",
            shader_dir.to_str().unwrap(),
        ]);

        // Conditionally define DEBUG_TELEMETRY for shader compilation
        if debug_telemetry {
            cmd.arg("-DDEBUG_TELEMETRY=1");
        }

        cmd.args([
            metal_file.to_str().unwrap(),
            "-o",
            air_file.to_str().unwrap(),
        ]);

        let status = cmd
            .status()
            .expect("Failed to run xcrun metal compiler");

        if !status.success() {
            panic!(
                "Metal shader compilation failed for {}",
                metal_file.display()
            );
        }

        air_files.push(air_file);

        // Re-run build if shader source changes
        println!("cargo:rerun-if-changed={}", metal_file.display());
    }

    // Also re-run if shared headers change
    println!("cargo:rerun-if-changed=shaders/types.h");
    println!("cargo:rerun-if-changed=shaders/prng.metal");

    // Link all .air files into a single .metallib
    let metallib_path = out_dir.join("shaders.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air_file in &air_files {
        cmd.arg(air_file.to_str().unwrap());
    }
    cmd.args(["-o", metallib_path.to_str().unwrap()]);

    let status = cmd.status().expect("Failed to run xcrun metallib linker");

    if !status.success() {
        panic!("Metal library linking failed");
    }

    println!(
        "cargo:warning=Built shaders.metallib at {}",
        metallib_path.display()
    );
}
