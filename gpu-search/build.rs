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
        // No .metal files yet — create a minimal metallib from a stub
        // so downstream code can load the library at runtime.
        let stub_metal = out_dir.join("_stub.metal");
        fs::write(
            &stub_metal,
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void _stub(uint tid [[thread_position_in_grid]]) {}\n",
        )
        .expect("Failed to write stub shader");

        let stub_air = out_dir.join("_stub.air");
        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                "-I",
                shader_dir.to_str().unwrap(),
                stub_metal.to_str().unwrap(),
                "-o",
                stub_air.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run xcrun metal compiler");

        if !status.success() {
            panic!("Metal stub shader compilation failed");
        }

        let metallib_path = out_dir.join("shaders.metallib");
        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metallib",
                stub_air.to_str().unwrap(),
                "-o",
                metallib_path.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run xcrun metallib linker");

        if !status.success() {
            panic!("Metal library linking failed");
        }

        println!(
            "cargo:warning=Built shaders.metallib (stub) at {}",
            metallib_path.display()
        );
        println!("cargo:rerun-if-changed=shaders/");
        return;
    }

    // Compile each .metal file to .air (Apple Intermediate Representation)
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_file = out_dir.join(format!("{stem}.air"));

        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                "-I",
                shader_dir.to_str().unwrap(),
                metal_file.to_str().unwrap(),
                "-o",
                air_file.to_str().unwrap(),
            ])
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

    // Re-run if any header files change
    for entry in fs::read_dir(shader_dir).expect("Failed to read shaders/ directory").flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "h") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

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
