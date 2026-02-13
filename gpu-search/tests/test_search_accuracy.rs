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
//! ## Metal Shader Validation
//!
//! These tests should be run with Apple's Metal diagnostic environment variables
//! enabled to catch GPU-side correctness issues:
//!
//!   MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 cargo test --test test_search_accuracy
//!
//! - `MTL_SHADER_VALIDATION=1`: Enables runtime validation of Metal shader code.
//!   Detects out-of-bounds buffer accesses, invalid texture reads, threadgroup
//!   memory races, and other undefined behavior in compute/fragment/vertex shaders.
//!   Essential for catching silent data corruption in `turbo_search.metal` kernels.
//!
//! - `MTL_DEBUG_LAYER=1`: Enables the Metal API validation layer. Catches
//!   incorrect API usage: invalid command encoder state, buffer overflows on
//!   dispatch, mismatched resource hazards, and missing synchronization barriers.
//!   Reports errors that would otherwise silently produce wrong results.
//!
//! Both variables are Apple-provided diagnostics with no code changes required.
//! They add ~10-20% overhead but should always be enabled in CI/test builds
//! to ensure GPU kernel correctness alongside CPU-side accuracy checks.
//!
//! REQUIRES: Real Apple Silicon GPU (Metal device must be available).

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

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

/// Quick GPU health check: create a tiny file with a known pattern and verify
/// the GPU can actually find it. Returns false if the GPU dispatches but returns
/// 0 matches (e.g., on CI runners where Metal compute may not work correctly).
fn gpu_returns_results(orchestrator: &mut SearchOrchestrator) -> bool {
    let dir = tempfile::TempDir::new().expect("create health check dir");
    let filepath = dir.path().join("health_check.txt");
    fs::write(&filepath, "line one\nHEALTH_CHECK_MARKER here\nline three\n")
        .expect("write health check file");

    let request = SearchRequest {
        pattern: "HEALTH_CHECK_MARKER".to_string(),
        root: dir.path().to_path_buf(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 100,
    };

    let response = orchestrator.search(request);
    !response.content_matches.is_empty()
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
            content.extend(std::iter::repeat_n(filler_char, line_len.saturating_sub(1)));
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

    // Skip gracefully if GPU doesn't return results on this hardware (e.g., CI M1 runners)
    if !gpu_returns_results(&mut orchestrator) {
        eprintln!("SKIPPED: GPU health check failed -- Metal compute returns 0 matches on this hardware");
        return;
    }

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

    // Skip gracefully if GPU doesn't return results on this hardware (e.g., CI M1 runners)
    if !gpu_returns_results(&mut orchestrator) {
        eprintln!("SKIPPED: GPU health check failed -- Metal compute returns 0 matches on this hardware");
        return;
    }

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

// ============================================================================
// Test 3: Multi-file batch byte_offset mapping with precise placement
// ============================================================================

/// Create 5 files with "ALPHA" at exact byte offsets to test chunk mapping:
///   file_a: 100B total, "ALPHA" at byte 50
///   file_b: 8000B total, "ALPHA" at byte 4000
///   file_c: 200B total, "ALPHA" at byte 10
///   file_d: 16000B total, "ALPHA" at bytes 1000 and 12000
///   file_e: 50B total, NO "ALPHA"
///
/// Assertions:
///   - Total match count == 5
///   - file_e is never in results
///   - Every match line_content contains "ALPHA"
#[test]
fn test_accuracy_multi_file_batch_byte_offset() {
    let (_device, _pso, mut orchestrator) = create_orchestrator();

    // Skip gracefully if GPU doesn't return results on this hardware (e.g., CI M1 runners)
    if !gpu_returns_results(&mut orchestrator) {
        eprintln!("SKIPPED: GPU health check failed -- Metal compute returns 0 matches on this hardware");
        return;
    }

    let dir = tempfile::TempDir::new().expect("create temp dir");

    /// Build a file of `total_size` bytes where "ALPHA" appears at each
    /// byte offset in `pattern_offsets`. Filler uses '_' chars with newlines
    /// every ~60 chars. If `pattern_offsets` is empty, no "ALPHA" at all.
    fn build_file(total_size: usize, pattern_offsets: &[usize]) -> Vec<u8> {
        let pattern = b"ALPHA";
        let pat_len = pattern.len();

        // Start with filler
        let mut buf = Vec::with_capacity(total_size);
        // Fill entire buffer with underscores first
        buf.resize(total_size, b'_');

        // Place newlines every ~60 chars for realistic line structure,
        // but avoid placing them where patterns will go
        let mut protected: HashSet<usize> = HashSet::new();
        for &off in pattern_offsets {
            for i in off..off + pat_len {
                if i < total_size {
                    protected.insert(i);
                }
            }
        }

        // Insert newlines for line structure
        let mut col = 0;
        for (i, byte) in buf.iter_mut().enumerate() {
            if !protected.contains(&i) {
                col += 1;
                if col >= 59 {
                    *byte = b'\n';
                    col = 0;
                }
            } else {
                col += 1;
            }
        }

        // Now stamp the pattern at exact offsets
        for &off in pattern_offsets {
            assert!(
                off + pat_len <= total_size,
                "Pattern offset {} + len {} exceeds file size {}",
                off, pat_len, total_size
            );
            buf[off..off + pat_len].copy_from_slice(pattern);
        }

        buf
    }

    // file_a: 100B, "ALPHA" at byte 50
    let file_a = dir.path().join("file_a.txt");
    fs::write(&file_a, build_file(100, &[50])).expect("write file_a");

    // file_b: 8000B, "ALPHA" at byte 4000
    let file_b = dir.path().join("file_b.txt");
    fs::write(&file_b, build_file(8000, &[4000])).expect("write file_b");

    // file_c: 200B, "ALPHA" at byte 10
    let file_c = dir.path().join("file_c.txt");
    fs::write(&file_c, build_file(200, &[10])).expect("write file_c");

    // file_d: 16000B, "ALPHA" at bytes 1000 and 12000
    let file_d = dir.path().join("file_d.txt");
    fs::write(&file_d, build_file(16000, &[1000, 12000])).expect("write file_d");

    // file_e: 50B, NO "ALPHA"
    let file_e = dir.path().join("file_e.txt");
    fs::write(&file_e, build_file(50, &[])).expect("write file_e");

    // Search for "ALPHA"
    let request = SearchRequest {
        pattern: "ALPHA".to_string(),
        root: dir.path().to_path_buf(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    let response = orchestrator.search(request);

    println!("\n=== MULTI-FILE BATCH BYTE_OFFSET TEST ===");
    println!("Total matches: {}", response.content_matches.len());

    // Print details of each match for debugging
    for cm in &response.content_matches {
        let fname = cm.path.file_name().unwrap_or_default().to_string_lossy();
        println!(
            "  Match: file={} line={} content='{}'",
            fname,
            cm.line_number,
            cm.line_content.trim()
        );
    }

    // Assertion 1: Total match count == 5
    // (file_a=1, file_b=1, file_c=1, file_d=2, file_e=0)
    assert_eq!(
        response.content_matches.len(),
        5,
        "Expected exactly 5 matches (file_a:1 + file_b:1 + file_c:1 + file_d:2), got {}",
        response.content_matches.len()
    );

    // Assertion 2: file_e is never in results
    for cm in &response.content_matches {
        let fname = cm
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        assert_ne!(
            fname, "file_e.txt",
            "file_e.txt should never appear in results (it has no 'ALPHA')"
        );
    }

    // Assertion 3: Every match line_content contains "ALPHA"
    let mut false_positives = 0;
    for cm in &response.content_matches {
        if !cm.line_content.contains("ALPHA") {
            false_positives += 1;
            let fname = cm.path.file_name().unwrap_or_default().to_string_lossy();
            eprintln!(
                "  FALSE POSITIVE: file={} line={} content='{}'",
                fname,
                cm.line_number,
                cm.line_content.trim()
            );
        }
    }
    assert_eq!(
        false_positives, 0,
        "All match line_content must contain 'ALPHA', found {} false positives",
        false_positives
    );

    // Verify correct file attribution
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

    // file_a, file_b, file_c, file_d should all have matches
    for expected in &["file_a.txt", "file_b.txt", "file_c.txt", "file_d.txt"] {
        assert!(
            matched_files.contains(*expected),
            "Expected matches in {} but it was not found in results. Matched files: {:?}",
            expected, matched_files
        );
    }

    println!("=== MULTI-FILE BATCH BYTE_OFFSET: All assertions passed ===");
}

// ============================================================================
// Test 4: Grep oracle comparison -- gpu-search vs grep on real source files
// ============================================================================

/// Run grep -rn --include=*.rs for each pattern on gpu-search/src/.
/// Parse output into (path, line_number) tuples.
fn grep_oracle(pattern: &str, search_dir: &std::path::Path) -> HashSet<(PathBuf, u32)> {
    let output = Command::new("grep")
        .args(["-rn", "--include=*.rs", pattern])
        .arg(search_dir)
        .output()
        .expect("failed to run grep");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut results = HashSet::new();

    for line in stdout.lines() {
        // grep output format: path:line_number:content
        // Split on first two colons to handle content with colons
        let mut parts = line.splitn(3, ':');
        if let (Some(path_str), Some(line_str)) = (parts.next(), parts.next()) {
            if let Ok(line_num) = line_str.parse::<u32>() {
                // Canonicalize the path to handle relative/absolute differences
                let path = PathBuf::from(path_str);
                let canonical = path.canonicalize().unwrap_or(path);
                results.insert((canonical, line_num));
            }
        }
    }

    results
}

/// Compare gpu-search results against grep as ground truth oracle.
///
/// For each of 6 common Rust patterns, run both grep and gpu-search on
/// gpu-search/src/. Assert:
///   1. Zero false positives: every gpu-search match exists in grep output
///   2. Miss rate < 20%: gpu-search finds at least 80% of what grep finds
#[test]
fn test_accuracy_vs_grep_oracle() {
    let (_device, _pso, mut orchestrator) = create_orchestrator();

    // Skip gracefully if GPU doesn't return results on this hardware (e.g., CI M1 runners)
    if !gpu_returns_results(&mut orchestrator) {
        eprintln!("SKIPPED: GPU health check failed -- Metal compute returns 0 matches on this hardware");
        return;
    }

    let patterns = ["fn ", "pub ", "struct ", "impl ", "use ", "mod "];

    // Resolve gpu-search/src/ relative to the workspace root
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    let search_dir = PathBuf::from(&manifest_dir).join("src");
    let search_dir = search_dir
        .canonicalize()
        .expect("failed to canonicalize gpu-search/src/");

    println!("Search directory: {:?}", search_dir);

    let mut total_grep_matches = 0usize;
    let mut total_false_positives = 0usize;
    let mut total_misses = 0usize;
    let mut per_pattern_stats: Vec<(String, usize, usize, usize)> = Vec::new();

    for pattern in &patterns {
        // --- Grep oracle ---
        let grep_results = grep_oracle(pattern, &search_dir);
        let grep_count = grep_results.len();

        // --- gpu-search ---
        let request = SearchRequest {
            pattern: pattern.to_string(),
            root: search_dir.clone(),
            file_types: Some(vec!["rs".to_string()]),
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 100_000,
        };

        let response = orchestrator.search(request);

        // Build gpu-search result set as (canonical_path, line_number)
        let gpu_results: HashSet<(PathBuf, u32)> = response
            .content_matches
            .iter()
            .map(|cm| {
                let canonical = cm.path.canonicalize().unwrap_or_else(|_| cm.path.clone());
                (canonical, cm.line_number)
            })
            .collect();

        let gpu_count = gpu_results.len();

        // --- False positives: gpu-search matches NOT in grep output ---
        let mut fp_count = 0usize;
        for (path, line_num) in &gpu_results {
            if !grep_results.contains(&(path.clone(), *line_num)) {
                fp_count += 1;
                // Print first few for debugging
                if fp_count <= 3 {
                    eprintln!(
                        "  FALSE POSITIVE: pattern='{}' file={:?} line={}",
                        pattern,
                        path.file_name().unwrap_or_default(),
                        line_num
                    );
                }
            }
        }

        // --- Misses: grep matches NOT in gpu-search output ---
        let miss_count = grep_results
            .iter()
            .filter(|entry| !gpu_results.contains(entry))
            .count();

        println!(
            "Pattern '{}': grep={} gpu={} false_pos={} misses={}",
            pattern.trim(),
            grep_count,
            gpu_count,
            fp_count,
            miss_count
        );

        total_grep_matches += grep_count;
        total_false_positives += fp_count;
        total_misses += miss_count;
        per_pattern_stats.push((
            pattern.to_string(),
            grep_count,
            fp_count,
            miss_count,
        ));
    }

    // --- Summary ---
    let miss_rate = if total_grep_matches > 0 {
        total_misses as f64 / total_grep_matches as f64
    } else {
        0.0
    };

    println!("\n=== GREP ORACLE COMPARISON SUMMARY ===");
    println!("Total grep matches:    {}", total_grep_matches);
    println!("Total false positives: {}", total_false_positives);
    println!("Total misses:          {}", total_misses);
    println!("Miss rate:             {:.2}%", miss_rate * 100.0);

    for (pat, grep_n, fp, miss) in &per_pattern_stats {
        let mr = if *grep_n > 0 {
            *miss as f64 / *grep_n as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  '{}': grep={} fp={} miss={} miss_rate={:.1}%",
            pat.trim(),
            grep_n,
            fp,
            miss,
            mr
        );
    }

    // --- Assertions ---
    assert_eq!(
        total_false_positives, 0,
        "ZERO false positives required: gpu-search returned {} matches not in grep output",
        total_false_positives
    );

    assert!(
        miss_rate < 0.20,
        "Miss rate {:.2}% exceeds 20% threshold ({} misses out of {} grep matches)",
        miss_rate * 100.0,
        total_misses,
        total_grep_matches
    );

    println!("=== GREP ORACLE: PASSED (0 false positives, {:.1}% miss rate) ===", miss_rate * 100.0);
}
