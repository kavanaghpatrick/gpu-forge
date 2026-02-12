//! Search accuracy tests for false positive detection.
//!
//! These tests validate that SearchOrchestrator returns correct results by
//! asserting that EVERY `ContentMatch.line_content` contains the search pattern.
//! They are designed to catch the P0 accuracy bug where GPU byte_offset mapping
//! produces false positives in multi-file batches.
//!
//! IMPORTANT: Some tests MAY FAIL initially -- they exist to catch the P0 bug.
//! That is expected behavior.
//!
//! Run with Metal shader validation:
//!   MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 cargo test --test test_search_accuracy
//!
//! REQUIRES: Real Apple Silicon GPU (Metal device must be available).

use std::collections::HashSet;
use std::fs;
use std::io::Write;

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::SearchRequest;

// ============================================================================
// Helpers
// ============================================================================

/// Create GPU device + PSO cache + orchestrator (shared setup).
fn create_orchestrator() -> (GpuDevice, PsoCache, SearchOrchestrator) {
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);
    let orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
        .expect("Failed to create SearchOrchestrator");
    (device, pso_cache, orchestrator)
}

/// Generate file content of approximately `target_size` bytes containing
/// `num_occurrences` of `pattern` at roughly evenly spaced positions.
/// The rest is filled with filler text that does NOT contain the pattern.
fn generate_corpus_file(pattern: &str, target_size: usize, num_occurrences: usize) -> Vec<u8> {
    let mut content = Vec::with_capacity(target_size + 256);

    if num_occurrences == 0 || target_size == 0 {
        // Fill with safe filler that won't match any test pattern
        let filler_line = b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n";
        while content.len() < target_size {
            let remaining = target_size - content.len();
            let to_write = remaining.min(filler_line.len());
            content.extend_from_slice(&filler_line[..to_write]);
        }
        return content;
    }

    // Calculate spacing between pattern insertions
    let spacing = if num_occurrences > 1 {
        target_size / num_occurrences
    } else {
        target_size / 2
    };

    let filler_char = b'_';
    let mut next_insert = spacing / 2; // Start inserting after half a spacing

    while content.len() < target_size {
        if content.len() >= next_insert && num_occurrences > 0 {
            // Insert pattern on its own line
            if !content.is_empty() && content.last() != Some(&b'\n') {
                content.push(b'\n');
            }
            content.extend_from_slice(b"prefix ");
            content.extend_from_slice(pattern.as_bytes());
            content.extend_from_slice(b" suffix\n");
            next_insert += spacing;
        } else {
            // Filler line -- use characters that won't form any test pattern
            let line_len = 60.min(target_size.saturating_sub(content.len()));
            for _ in 0..line_len.saturating_sub(1) {
                content.push(filler_char);
            }
            if line_len > 0 {
                content.push(b'\n');
            }
        }
    }

    content.truncate(target_size);
    content
}

// ============================================================================
// Test 1: Assert every returned line contains the pattern
// ============================================================================

/// For each of 10 patterns, run SearchOrchestrator::search() over a temp corpus,
/// and assert EVERY ContentMatch.line_content contains the pattern.
#[test]
fn test_accuracy_line_content_contains_pattern() {
    let (_device, _pso, mut orchestrator) = create_orchestrator();

    let patterns = [
        "ALPHA_MARKER",
        "fn search",
        "impl Iterator",
        "TODO_FIX",
        "ERROR_CODE_42",
        "pub struct",
        "return Ok",
        "DEADBEEF",
        "match self",
        "async fn",
    ];

    // File sizes: 10 files from ~100B to ~1MB
    let file_sizes: [usize; 10] = [
        100, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 200_000, 1_000_000,
    ];

    let dir = tempfile::TempDir::new().expect("create temp dir");

    // Create corpus: each file has a different size and contains 1-5 occurrences
    // of each pattern, embedded in non-matching filler text.
    for (i, &size) in file_sizes.iter().enumerate() {
        let mut file_content = Vec::with_capacity(size + 4096);

        // Add occurrences of ALL patterns to each file (with some variation)
        for (j, pattern) in patterns.iter().enumerate() {
            let num_occurrences = 1 + ((i + j) % 5); // 1-5 occurrences
            let chunk = generate_corpus_file(pattern, size / patterns.len(), num_occurrences);
            file_content.extend_from_slice(&chunk);
        }

        let filename = format!("corpus_{:02}.txt", i);
        let filepath = dir.path().join(&filename);
        let mut f = fs::File::create(&filepath).expect("create corpus file");
        f.write_all(&file_content).expect("write corpus file");
    }

    // Search for each pattern and verify accuracy
    let mut total_matches = 0;
    let mut total_false_positives = 0;

    for pattern in &patterns {
        let request = SearchRequest {
            pattern: pattern.to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        let match_count = response.content_matches.len();
        total_matches += match_count;

        println!(
            "Pattern '{}': {} matches in {} files searched",
            pattern, match_count, response.total_files_searched
        );

        // Core accuracy assertion: EVERY line_content must contain the pattern
        for cm in &response.content_matches {
            if !cm.line_content.contains(pattern) {
                total_false_positives += 1;
                eprintln!(
                    "  FALSE POSITIVE: pattern='{}' file={:?} line={} content='{}'",
                    pattern,
                    cm.path.file_name().unwrap_or_default().to_string_lossy(),
                    cm.line_number,
                    cm.line_content.trim()
                );
            }
        }
    }

    println!("\n=== ACCURACY SUMMARY ===");
    println!("Total matches: {}", total_matches);
    println!("False positives: {}", total_false_positives);
    println!(
        "Accuracy: {:.2}%",
        if total_matches > 0 {
            (1.0 - total_false_positives as f64 / total_matches as f64) * 100.0
        } else {
            100.0
        }
    );

    assert_eq!(
        total_false_positives, 0,
        "Found {} false positives out of {} matches -- EVERY line_content must contain its pattern",
        total_false_positives, total_matches
    );
}

// ============================================================================
// Test 2: No cross-file contamination in multi-file batches
// ============================================================================

/// 5 files of different sizes, each containing a UNIQUE pattern that appears
/// ONLY in that file. Search for each unique pattern and assert:
/// 1. Matches come ONLY from the correct file
/// 2. Every line_content contains the pattern
/// 3. No cross-file contamination from batch processing
#[test]
fn test_no_false_positives_multi_file_batch() {
    let (_device, _pso, mut orchestrator) = create_orchestrator();

    let dir = tempfile::TempDir::new().expect("create temp dir");

    // Each file has a unique pattern that exists ONLY in that file
    let file_specs: Vec<(&str, &str, usize)> = vec![
        ("small.txt", "UNIQUE_PATTERN_SMALL_7x9q", 150),
        ("medium.txt", "UNIQUE_PATTERN_MEDIUM_3k8m", 4_000),
        ("large.txt", "UNIQUE_PATTERN_LARGE_5n2p", 50_000),
        ("xlarge.txt", "UNIQUE_PATTERN_XLARGE_1w4r", 200_000),
        ("huge.txt", "UNIQUE_PATTERN_HUGE_9t6j", 800_000),
    ];

    // Create each file with its unique pattern (3 occurrences each)
    for (filename, pattern, size) in &file_specs {
        let content = generate_corpus_file(pattern, *size, 3);
        let filepath = dir.path().join(filename);
        fs::write(&filepath, &content).expect("write file");
    }

    // Search for each unique pattern
    for (expected_file, pattern, _size) in &file_specs {
        let request = SearchRequest {
            pattern: pattern.to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        println!(
            "Pattern '{}': {} matches (expected file: {})",
            pattern,
            response.content_matches.len(),
            expected_file
        );

        // Collect all files that had matches
        let matched_files: HashSet<String> = response
            .content_matches
            .iter()
            .map(|cm| {
                cm.path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            })
            .collect();

        // Assert: no cross-file contamination
        for file in &matched_files {
            if file.as_str() != *expected_file {
                panic!(
                    "CROSS-FILE CONTAMINATION: pattern '{}' expected only in '{}' but found in '{}'",
                    pattern, expected_file, file
                );
            }
        }

        // Assert: every line_content contains the pattern
        for cm in &response.content_matches {
            assert!(
                cm.line_content.contains(pattern),
                "FALSE POSITIVE: pattern='{}' file={:?} line={} content='{}'",
                pattern,
                cm.path.file_name().unwrap_or_default().to_string_lossy(),
                cm.line_number,
                cm.line_content.trim()
            );
        }
    }

    println!("\n=== MULTI-FILE BATCH: No cross-file contamination detected ===");
}
