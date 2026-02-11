//! POC end-to-end GPU search -- validates the entire pipeline works.
//!
//! Usage: cargo run -- "pattern" ./directory
//!
//! Wires together: GpuDevice -> PsoCache -> StreamingSearchEngine -> results.
//! Walks the target directory recursively, searches all text files for the
//! given pattern, prints results in grep-like format (path:line:content).

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::streaming::StreamingSearchEngine;
use gpu_search::search::content::SearchOptions;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <pattern> <directory>", args[0]);
        eprintln!("  Searches for <pattern> in all files under <directory>");
        eprintln!("  Output format: path:match_offset (grep-like)");
        std::process::exit(1);
    }

    let pattern = &args[1];
    let dir_path = Path::new(&args[2]);

    if !dir_path.exists() {
        eprintln!("Error: directory '{}' does not exist", dir_path.display());
        std::process::exit(1);
    }

    eprintln!("gpu-search v0.1.0 -- POC end-to-end GPU search");
    eprintln!("Pattern: \"{}\"", pattern);
    eprintln!("Directory: {}", dir_path.display());

    // Step 1: Initialize GPU
    let init_start = Instant::now();
    let gpu = GpuDevice::new();
    eprintln!("GPU: {} (init: {:.1}ms)", gpu.device_name, init_start.elapsed().as_secs_f64() * 1000.0);

    // Step 2: Create PSO cache (compile/load Metal shaders)
    let pso_start = Instant::now();
    let pso_cache = PsoCache::new(&gpu.device);
    eprintln!("PSO cache: {} kernels (init: {:.1}ms)", pso_cache.len(), pso_start.elapsed().as_secs_f64() * 1000.0);

    // Step 3: Create streaming search engine
    let mut engine = StreamingSearchEngine::new(&gpu.device, &pso_cache)
        .expect("Failed to create streaming search engine");

    // Step 4: Walk directory to collect files
    let walk_start = Instant::now();
    let files = walk_directory(dir_path);
    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Files found: {} (walk: {:.1}ms)", files.len(), walk_ms);

    if files.is_empty() {
        eprintln!("No files found in {}", dir_path.display());
        std::process::exit(0);
    }

    // Step 5: GPU streaming search
    let search_start = Instant::now();
    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };
    let (results, profile) = engine.search_files_with_profile(&files, pattern.as_bytes(), &options);
    let search_ms = search_start.elapsed().as_secs_f64() * 1000.0;

    // Step 6: Resolve line numbers and print results in grep-like format
    let output_start = Instant::now();
    let mut printed = 0;
    for m in &results {
        // Read the file to resolve byte_offset -> line:content
        if let Some((line_num, line_content)) = resolve_line(&m.file_path, m.byte_offset as usize) {
            println!("{}:{}:{}", m.file_path.display(), line_num, line_content);
            printed += 1;
        }
    }
    let output_ms = output_start.elapsed().as_secs_f64() * 1000.0;

    // Step 7: Print summary
    eprintln!("---");
    eprintln!("GPU matches: {} ({} printed)", results.len(), printed);
    eprintln!("Files searched: {}", profile.files_processed);
    eprintln!("Data searched: {:.2} MB", profile.bytes_processed as f64 / (1024.0 * 1024.0));
    eprintln!("Search time: {:.1}ms (I/O: {:.1}ms, GPU: {:.1}ms, output: {:.1}ms)",
        search_ms,
        profile.io_us as f64 / 1000.0,
        profile.search_us as f64 / 1000.0,
        output_ms,
    );
    if profile.total_us > 0 {
        let throughput = (profile.bytes_processed as f64 / (1024.0 * 1024.0))
            / (profile.total_us as f64 / 1_000_000.0);
        eprintln!("Throughput: {:.1} MB/s", throughput);
    }
}

/// Recursively walk a directory, collecting all regular file paths.
/// Skips hidden directories (.git, .hg, etc.), binary files, and symlinks.
fn walk_directory(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk_recursive(dir, &mut files);
    files.sort();
    files
}

fn walk_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden directories and common non-text directories
        if name_str.starts_with('.') {
            continue;
        }
        if matches!(name_str.as_ref(), "target" | "node_modules" | "__pycache__") {
            continue;
        }

        if path.is_dir() {
            walk_recursive(&path, files);
        } else if path.is_file() {
            // Skip known binary extensions
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if is_binary_extension(ext) {
                    continue;
                }
            }
            // Skip empty or very large files
            if let Ok(meta) = path.metadata() {
                let size = meta.len();
                if size == 0 || size > 100 * 1024 * 1024 {
                    continue;
                }
            }
            files.push(path);
        }
    }
}

/// Check if a file extension indicates a binary file.
fn is_binary_extension(ext: &str) -> bool {
    matches!(
        ext.to_lowercase().as_str(),
        "o" | "a" | "dylib" | "so" | "dll" | "exe"
        | "metallib" | "air" | "metalar"
        | "png" | "jpg" | "jpeg" | "gif" | "bmp" | "ico" | "webp"
        | "mp3" | "mp4" | "wav" | "avi" | "mov"
        | "zip" | "gz" | "tar" | "bz2" | "xz" | "7z"
        | "pdf" | "doc" | "docx" | "xls" | "xlsx"
        | "wasm" | "class" | "pyc" | "pyo"
        | "rlib" | "rmeta" | "d"
    )
}

/// Given a file path and byte offset, resolve to (line_number, line_content).
///
/// Reads the file, counts newlines up to byte_offset, extracts the line.
/// Returns None if the file can't be read or offset is out of bounds.
fn resolve_line(path: &Path, byte_offset: usize) -> Option<(usize, String)> {
    let content = std::fs::read(path).ok()?;

    if byte_offset >= content.len() {
        // Offset might be approximate from GPU -- clamp to valid range
        return None;
    }

    // Count newlines before byte_offset to get line number
    let mut line_num = 1usize;
    let mut line_start = 0usize;

    for (i, &b) in content[..byte_offset].iter().enumerate() {
        if b == b'\n' {
            line_num += 1;
            line_start = i + 1;
        }
    }

    // Find end of current line
    let line_end = content[byte_offset..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|pos| byte_offset + pos)
        .unwrap_or(content.len());

    // Extract line content (lossy UTF-8)
    let line = String::from_utf8_lossy(&content[line_start..line_end]).to_string();

    Some((line_num, line))
}
