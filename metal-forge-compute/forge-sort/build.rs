use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile = env::var("PROFILE").unwrap_or_default();

    let metal_file = "shaders/sort.metal";
    let air_file = out_dir.join("sort.air");

    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metal", "-std=metal3.2", "-c"]);
    if profile == "release" {
        cmd.arg("-O2");
    }
    cmd.args([metal_file, "-o", air_file.to_str().unwrap()]);

    let status = cmd.status().expect("Failed to run xcrun metal compiler");
    if !status.success() {
        panic!("Metal shader compilation failed for {}", metal_file);
    }
    println!("cargo:rerun-if-changed={}", metal_file);

    let metallib_path = out_dir.join("shaders.metallib");
    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metallib",
            air_file.to_str().unwrap(),
            "-o",
            metallib_path.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run xcrun metallib linker");
    if !status.success() {
        panic!("Metal library linking failed");
    }

    println!(
        "cargo:rustc-env=SORT_METALLIB_PATH={}",
        metallib_path.display()
    );
    println!(
        "cargo:warning=Built shaders.metallib at {}",
        metallib_path.display()
    );
}
