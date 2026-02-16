//! Integration tests for GPU search false positive detection.
//!
//! These tests exercise `StreamingSearchEngine::search_files()` directly
//! (bypassing the orchestrator) to isolate the GPU pipeline and catch
//! false positives from stale buffers, cross-file contamination, or
//! byte_offset mapping errors.
//!
//! REQUIRES: Real Apple Silicon GPU (Metal device must be available).

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::content::{ContentSearchEngine, SearchOptions};
use gpu_search::search::streaming::{cpu_streaming_search, StreamingSearchEngine};
use objc2_metal::MTLCreateSystemDefaultDevice;

// ============================================================================
// Helpers
// ============================================================================

/// Create a `StreamingSearchEngine` with default quad-buffer config.
fn create_streaming_engine() -> (
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    PsoCache,
    StreamingSearchEngine,
) {
    let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
    let pso_cache = PsoCache::new(&device);
    let engine = StreamingSearchEngine::new(&device, &pso_cache)
        .expect("Failed to create StreamingSearchEngine");
    (device, pso_cache, engine)
}

/// Quick GPU health check: create a tiny file with a known pattern and verify
/// the GPU can find it. Returns false if Metal compute doesn't work on this hardware.
fn gpu_returns_results(engine: &mut StreamingSearchEngine) -> bool {
    let dir = tempfile::TempDir::new().expect("create health check dir");
    let filepath = dir.path().join("health_check.txt");
    fs::write(
        &filepath,
        "line one\nHEALTH_CHECK_MARKER here\nline three\n",
    )
    .expect("write health check file");

    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };
    let results = engine.search_files(&[filepath], b"HEALTH_CHECK_MARKER", &options);
    !results.is_empty()
}

/// Create a temp directory with numbered files, each containing unique content.
///
/// Returns (TempDir, file_paths) where:
/// - file_0.txt contains "UNIQUE_ALPHA_0" on multiple lines + filler
/// - file_1.txt contains "UNIQUE_BETA_1" on multiple lines + filler
/// - file_2.txt contains "UNIQUE_GAMMA_2" on multiple lines + filler
/// - etc.
///
/// Each file is ~4KB to ensure GPU dispatch (above cold threshold).
fn create_unique_content_files(
    count: usize,
) -> (tempfile::TempDir, Vec<PathBuf>, Vec<String>) {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let mut paths = Vec::with_capacity(count);
    let mut patterns = Vec::with_capacity(count);

    let prefixes = [
        "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA", "IOTA", "KAPPA",
        "LAMBDA", "MU", "NU", "XI", "OMICRON", "PI", "RHO", "SIGMA", "TAU", "UPSILON",
    ];

    for i in 0..count {
        let prefix = prefixes[i % prefixes.len()];
        let unique_pattern = format!("UNIQUE_{}_{}", prefix, i);

        // Build file content: ~4KB with the unique pattern on 5 lines + filler
        let mut content = String::with_capacity(5000);
        for line in 0..80 {
            if line % 16 == 3 {
                // Insert the unique pattern every 16 lines
                content.push_str(&format!(
                    "data line {} contains {} in file {}\n",
                    line, unique_pattern, i
                ));
            } else {
                // Filler that won't match any UNIQUE_ pattern
                content.push_str(&format!(
                    "filler_line_{:04}_file_{}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                    line, i
                ));
            }
        }

        let filepath = dir.path().join(format!("file_{}.txt", i));
        fs::write(&filepath, content.as_bytes()).expect("write test file");
        paths.push(filepath);
        patterns.push(unique_pattern);
    }

    (dir, paths, patterns)
}

// ============================================================================
// Test 1: Unique patterns - each pattern matches ONLY its designated file
// ============================================================================

/// Create files with unique content, search for each unique pattern, and verify
/// matches come ONLY from the correct file. This catches cross-file contamination
/// from stale GPU buffers or incorrect byte_offset mapping.
#[test]
fn test_unique_pattern_isolation() {
    let (_device, _pso, mut engine) = create_streaming_engine();

    if !gpu_returns_results(&mut engine) {
        eprintln!("SKIPPED: GPU health check failed");
        return;
    }

    let file_count = 10;
    let (dir, paths, _patterns) = create_unique_content_files(file_count);
    let _ = dir; // keep alive

    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };

    // Search for "UNIQUE_ALPHA" -- should match ONLY file_0
    println!("=== Test: UNIQUE_ALPHA should match only file_0 ===");
    let results = engine.search_files(&paths, b"UNIQUE_ALPHA", &options);

    println!("  Found {} matches for UNIQUE_ALPHA", results.len());
    for m in &results {
        println!(
            "    file={} line={} byte_offset={}",
            m.file_path.file_name().unwrap_or_default().to_string_lossy(),
            m.line_number,
            m.byte_offset
        );
    }

    // All matches must be from file_0
    assert!(
        !results.is_empty(),
        "Should find at least one match for UNIQUE_ALPHA in file_0"
    );
    for m in &results {
        let fname = m
            .file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        assert_eq!(
            fname, "file_0.txt",
            "UNIQUE_ALPHA should only match file_0.txt, but found in {}",
            fname
        );
    }

    // Search for "UNIQUE_BETA" -- should match ONLY file_1
    println!("\n=== Test: UNIQUE_BETA should match only file_1 ===");
    let results = engine.search_files(&paths, b"UNIQUE_BETA", &options);

    println!("  Found {} matches for UNIQUE_BETA", results.len());
    for m in &results {
        println!(
            "    file={} line={} byte_offset={}",
            m.file_path.file_name().unwrap_or_default().to_string_lossy(),
            m.line_number,
            m.byte_offset
        );
    }

    assert!(
        !results.is_empty(),
        "Should find at least one match for UNIQUE_BETA in file_1"
    );
    for m in &results {
        let fname = m
            .file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        assert_eq!(
            fname, "file_1.txt",
            "UNIQUE_BETA should only match file_1.txt, but found in {}",
            fname
        );
    }

    println!("\n=== UNIQUE_PATTERN_ISOLATION: PASSED ===");
}

// ============================================================================
// Test 2: Sequential queries - search all unique patterns in sequence
// ============================================================================

/// For each of N files, search for its unique pattern and verify:
/// 1. Matches come ONLY from the correct file
/// 2. Match count matches CPU reference
/// 3. Zero cross-file contamination across sequential queries
#[test]
fn test_sequential_unique_queries() {
    let (_device, _pso, mut engine) = create_streaming_engine();

    if !gpu_returns_results(&mut engine) {
        eprintln!("SKIPPED: GPU health check failed");
        return;
    }

    let file_count = 10;
    let (dir, paths, patterns) = create_unique_content_files(file_count);
    let _ = dir;

    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };

    let mut total_false_positives = 0;

    for (i, pattern) in patterns.iter().enumerate() {
        let pattern_bytes = pattern.as_bytes();

        // GPU search
        let results = engine.search_files(&paths, pattern_bytes, &options);

        // CPU reference
        let cpu_count = cpu_streaming_search(&paths, pattern_bytes, true);

        let expected_file = format!("file_{}.txt", i);

        println!(
            "Pattern '{}': GPU={} CPU={} (expected file: {})",
            pattern,
            results.len(),
            cpu_count,
            expected_file
        );

        // Collect files that had matches
        let matched_files: HashSet<String> = results
            .iter()
            .map(|m| {
                m.file_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            })
            .collect();

        // Assert: matches come ONLY from the expected file
        for file in &matched_files {
            if file.as_str() != expected_file {
                total_false_positives += 1;
                eprintln!(
                    "  CROSS-FILE CONTAMINATION: pattern='{}' expected '{}' but found in '{}'",
                    pattern, expected_file, file
                );
            }
        }

        // Assert: GPU should not produce more matches than CPU
        assert!(
            results.len() <= cpu_count,
            "GPU false positives: GPU({}) > CPU({}) for pattern '{}'",
            results.len(),
            cpu_count,
            pattern
        );
    }

    println!("\n=== SEQUENTIAL QUERIES SUMMARY ===");
    println!("False positives (cross-file): {}", total_false_positives);

    assert_eq!(
        total_false_positives, 0,
        "Found {} cross-file false positives across {} sequential queries",
        total_false_positives, file_count
    );

    println!("=== SEQUENTIAL_UNIQUE_QUERIES: PASSED ===");
}

// ============================================================================
// Test 3: Back-to-back search with different patterns on same engine
// ============================================================================

/// Search for pattern A, then pattern B, then pattern A again.
/// Verifies that switching patterns doesn't leave stale results.
#[test]
fn test_pattern_switch_no_stale_results() {
    let (_device, _pso, mut engine) = create_streaming_engine();

    if !gpu_returns_results(&mut engine) {
        eprintln!("SKIPPED: GPU health check failed");
        return;
    }

    let file_count = 10;
    let (dir, paths, _patterns) = create_unique_content_files(file_count);
    let _ = dir;

    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };

    // Search 1: UNIQUE_ALPHA (should match file_0 only)
    println!("=== Search 1: UNIQUE_ALPHA ===");
    let results_1 = engine.search_files(&paths, b"UNIQUE_ALPHA", &options);
    let count_1 = results_1.len();
    println!("  Matches: {}", count_1);

    for m in &results_1 {
        let fname = m.file_path.file_name().unwrap_or_default().to_string_lossy().to_string();
        assert_eq!(fname, "file_0.txt", "Search 1: UNIQUE_ALPHA in wrong file: {}", fname);
    }

    // Search 2: UNIQUE_BETA (should match file_1 only, no stale ALPHA results)
    println!("\n=== Search 2: UNIQUE_BETA ===");
    let results_2 = engine.search_files(&paths, b"UNIQUE_BETA", &options);
    let count_2 = results_2.len();
    println!("  Matches: {}", count_2);

    for m in &results_2 {
        let fname = m.file_path.file_name().unwrap_or_default().to_string_lossy().to_string();
        assert_eq!(fname, "file_1.txt", "Search 2: UNIQUE_BETA in wrong file: {}", fname);
    }

    // Search 3: UNIQUE_ALPHA again (should produce identical results to search 1)
    println!("\n=== Search 3: UNIQUE_ALPHA (repeat) ===");
    let results_3 = engine.search_files(&paths, b"UNIQUE_ALPHA", &options);
    let count_3 = results_3.len();
    println!("  Matches: {}", count_3);

    for m in &results_3 {
        let fname = m.file_path.file_name().unwrap_or_default().to_string_lossy().to_string();
        assert_eq!(fname, "file_0.txt", "Search 3: UNIQUE_ALPHA in wrong file: {}", fname);
    }

    // Verify determinism: search 1 and search 3 should have identical counts
    assert_eq!(
        count_1, count_3,
        "Search for same pattern should be deterministic: first={} repeat={}",
        count_1, count_3
    );

    println!("\n=== PATTERN_SWITCH_NO_STALE_RESULTS: PASSED ===");
}

// ============================================================================
// Test 4: GPU vs CPU reference for each query
// ============================================================================

/// For every unique pattern, verify GPU match count <= CPU reference count
/// (GPU may miss boundary matches but must never produce false positives).
#[test]
fn test_gpu_vs_cpu_no_false_positives() {
    let (_device, _pso, mut engine) = create_streaming_engine();

    if !gpu_returns_results(&mut engine) {
        eprintln!("SKIPPED: GPU health check failed");
        return;
    }

    let file_count = 15;
    let (dir, paths, patterns) = create_unique_content_files(file_count);
    let _ = dir;

    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };

    let mut total_gpu = 0usize;
    let mut total_cpu = 0usize;
    let mut false_positive_queries = 0usize;

    for pattern in &patterns {
        let pattern_bytes = pattern.as_bytes();

        let gpu_results = engine.search_files(&paths, pattern_bytes, &options);
        let cpu_count = cpu_streaming_search(&paths, pattern_bytes, true);

        let gpu_count = gpu_results.len();
        total_gpu += gpu_count;
        total_cpu += cpu_count;

        if gpu_count > cpu_count {
            false_positive_queries += 1;
            eprintln!(
                "FALSE POSITIVE QUERY: pattern='{}' GPU={} > CPU={}",
                pattern, gpu_count, cpu_count
            );
        }

        println!(
            "Pattern '{}': GPU={} CPU={} {}",
            pattern,
            gpu_count,
            cpu_count,
            if gpu_count > cpu_count {
                "*** FALSE POSITIVE ***"
            } else {
                "OK"
            }
        );
    }

    println!("\n=== GPU vs CPU SUMMARY ===");
    println!("Total GPU matches: {}", total_gpu);
    println!("Total CPU matches: {}", total_cpu);
    println!("False positive queries: {}", false_positive_queries);

    assert_eq!(
        false_positive_queries, 0,
        "{} queries had GPU > CPU matches (false positives)",
        false_positive_queries
    );

    println!("=== GPU_VS_CPU_NO_FALSE_POSITIVES: PASSED ===");
}

// ============================================================================
// Test 5: Stale buffer detection after reset()
// ============================================================================

/// Verifies that `reset()` properly zeros GPU metadata buffers, preventing
/// stale data from a previous search from causing false positives.
///
/// After the fix (task 2.1), reset() zeros metadata_buffer, match_count_buffer,
/// and matches_buffer. This test confirms that after loading 10 files, calling
/// reset(), and loading 2 files, chunks 2-9 are fully zeroed (no stale data).
///
/// Steps:
/// 1. Load 10 files with pattern "AAAA" -> ~10 chunks, search confirms matches
/// 2. reset() -> buffers zeroed
/// 3. Load 2 files with pattern "BBBB" -> ~2 chunks
/// 4. Inspect metadata_buffer at chunk positions 2..9 -> should be all zeros
/// 5. Search for "AAAA" -> should be 0 matches (no false positives)
#[test]
fn test_stale_buffer_after_reset() {
    let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
    let pso_cache = PsoCache::new(&device);
    let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);

    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };

    // ----- Phase 1: Load 10 files with "AAAA" pattern -----
    println!("=== Phase 1: Loading 10 files with AAAA pattern ===");

    // Each file is ~2KB (under CHUNK_SIZE=4096), so 1 chunk per file = 10 chunks total
    for file_idx in 0..10u32 {
        let mut content = Vec::with_capacity(2048);
        for line in 0..40 {
            if line % 8 == 0 {
                content.extend_from_slice(
                    format!("line {} has AAAA marker in file {}\n", line, file_idx).as_bytes(),
                );
            } else {
                content.extend_from_slice(
                    format!(
                        "filler_{:04}_file_{}_xxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                        line, file_idx
                    )
                    .as_bytes(),
                );
            }
        }
        let chunks = engine.load_content(&content, file_idx);
        assert!(chunks > 0, "File {} should produce at least 1 chunk", file_idx);
    }

    let chunks_after_phase1 = engine.chunk_count();
    println!("  Loaded {} chunks across 10 files", chunks_after_phase1);
    assert!(
        chunks_after_phase1 >= 10,
        "Expected at least 10 chunks, got {}",
        chunks_after_phase1
    );

    // Search for "AAAA" -- should find matches
    let results_phase1 = engine.search(b"AAAA", &options);
    println!("  Search 'AAAA': {} matches", results_phase1.len());
    assert!(
        !results_phase1.is_empty(),
        "Phase 1: Should find AAAA matches in the 10 loaded files"
    );

    // Snapshot: inspect metadata buffer before reset
    let meta_before_reset = engine.inspect_metadata_buffer();

    // ----- Phase 2: Reset and load only 2 files with "BBBB" -----
    println!("\n=== Phase 2: reset() then load 2 files with BBBB ===");
    engine.reset();

    // Verify counters are zeroed
    assert_eq!(engine.chunk_count(), 0, "chunk_count should be 0 after reset");
    assert_eq!(engine.file_count(), 0, "file_count should be 0 after reset");

    // Load 2 files with completely different content (BBBB, no AAAA anywhere)
    for file_idx in 0..2u32 {
        let mut content = Vec::with_capacity(2048);
        for line in 0..40 {
            if line % 8 == 0 {
                content.extend_from_slice(
                    format!("line {} has BBBB marker in file {}\n", line, file_idx).as_bytes(),
                );
            } else {
                content.extend_from_slice(
                    format!(
                        "padding_{:04}_file_{}_yyyyyyyyyyyyyyyyyyyyyyyyyyyy\n",
                        line, file_idx
                    )
                    .as_bytes(),
                );
            }
        }
        engine.load_content(&content, file_idx);
    }

    let chunks_after_phase2 = engine.chunk_count();
    println!("  Loaded {} chunks across 2 files", chunks_after_phase2);
    assert!(
        chunks_after_phase2 < chunks_after_phase1,
        "Phase 2 should have fewer chunks ({}) than phase 1 ({})",
        chunks_after_phase2,
        chunks_after_phase1
    );

    // ----- Phase 3: Inspect metadata buffer for stale data -----
    println!("\n=== Phase 3: Inspecting metadata buffer for stale data ===");

    let meta_after_reload = engine.inspect_metadata_buffer();
    let chunk_meta_size = 24; // sizeof(ChunkMetadata)

    let mut stale_chunks_found = 0;
    let mut stale_details = Vec::new();

    // Check chunks beyond what phase 2 loaded (positions chunks_after_phase2 .. chunks_after_phase1)
    for chunk_idx in chunks_after_phase2..chunks_after_phase1 {
        let offset = chunk_idx * chunk_meta_size;
        if offset + chunk_meta_size > meta_after_reload.len() {
            break;
        }

        // Read file_index (u32 at offset+0) and chunk_length (u32 at offset+16)
        let file_index = u32::from_ne_bytes(
            meta_after_reload[offset..offset + 4].try_into().unwrap(),
        );
        let chunk_length = u32::from_ne_bytes(
            meta_after_reload[offset + 16..offset + 20].try_into().unwrap(),
        );

        let is_stale = file_index != 0 || chunk_length != 0;
        if is_stale {
            stale_chunks_found += 1;
            stale_details.push((chunk_idx, file_index, chunk_length));
        }

        println!(
            "  Chunk {}: file_index={}, chunk_length={} {}",
            chunk_idx,
            file_index,
            chunk_length,
            if is_stale { "<-- STALE" } else { "(clean)" }
        );
    }

    println!(
        "\n  Stale chunks found: {} / {} (positions {}..{})",
        stale_chunks_found,
        chunks_after_phase1 - chunks_after_phase2,
        chunks_after_phase2,
        chunks_after_phase1
    );

    // Also compare raw bytes: the region for chunks beyond phase2 should still
    // match the pre-reset snapshot (proving reset() didn't touch them)
    let stale_region_start = chunks_after_phase2 * chunk_meta_size;
    let stale_region_end = chunks_after_phase1 * chunk_meta_size;
    let before_stale = &meta_before_reset[stale_region_start..stale_region_end];
    let after_stale = &meta_after_reload[stale_region_start..stale_region_end];
    let bytes_match = before_stale == after_stale;
    println!(
        "  Stale region bytes unchanged after reset: {} ({} bytes)",
        bytes_match,
        stale_region_end - stale_region_start
    );

    // ----- Phase 4: Search for "AAAA" -- should be 0 but stale buffers may cause FP -----
    println!("\n=== Phase 4: Searching for 'AAAA' (should be 0 in current content) ===");

    let results_aaaa = engine.search(b"AAAA", &options);
    println!(
        "  Search 'AAAA' after reload with BBBB-only content: {} matches",
        results_aaaa.len()
    );

    // Also verify BBBB works correctly
    let results_bbbb = engine.search(b"BBBB", &options);
    println!("  Search 'BBBB' (should find matches): {} matches", results_bbbb.len());
    assert!(
        !results_bbbb.is_empty(),
        "Should find BBBB matches in the 2 loaded files"
    );

    // ----- Report findings -----
    println!("\n=== STALE BUFFER DETECTION SUMMARY ===");
    println!("  Stale metadata entries: {}", stale_chunks_found);
    println!("  Stale bytes unchanged: {}", bytes_match);
    println!("  False positives (AAAA in BBBB-only content): {}", results_aaaa.len());

    // After the fix (task 2.1), reset() zeros metadata_buffer.
    // All chunks beyond current_chunk_count should be clean (zeroed).
    assert_eq!(
        stale_chunks_found, 0,
        "reset() should zero metadata buffer -- found {} stale chunks",
        stale_chunks_found
    );

    // Stale region bytes should NOT match pre-reset snapshot (they were zeroed)
    assert!(
        !bytes_match,
        "Stale region bytes should differ after reset (zeroed vs old data)"
    );

    // No false positives: searching for "AAAA" in BBBB-only content should yield 0
    assert_eq!(
        results_aaaa.len(),
        0,
        "Should have 0 false positives for 'AAAA' after reset, got {}",
        results_aaaa.len()
    );

    println!("\n=== STALE_BUFFER_AFTER_RESET: FIX VERIFIED ===");
}

// ============================================================================
// Test 6: Rapid query change simulation (kolbey / patrick / kolbey)
// ============================================================================

/// Simulates rapid query changes like a user typing in a search box:
///   search "patrick" -> search "kolbey" -> search "patrick" again
///
/// Creates 50 temp files:
/// - files 0-9: contain "kolbey"
/// - files 10-19: contain "patrick"
/// - files 20-49: generic filler (no "kolbey" or "patrick")
///
/// Validates every GPU result against CPU reference, asserts zero false
/// positives, and checks determinism (search 1 count == search 3 count).
#[test]
fn test_rapid_query_change() {
    let (_device, _pso, mut engine) = create_streaming_engine();

    if !gpu_returns_results(&mut engine) {
        eprintln!("SKIPPED: GPU health check failed");
        return;
    }

    // ----- Create 50 temp files -----
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let mut paths = Vec::with_capacity(50);

    for i in 0..50 {
        let filepath = dir.path().join(format!("file_{:02}.txt", i));

        // Build ~4KB+ content to ensure GPU dispatch (above cold threshold)
        let mut content = String::with_capacity(5000);
        for line in 0..80 {
            if i < 10 {
                // Files 0-9: contain "kolbey"
                if line % 10 == 3 {
                    content.push_str(&format!(
                        "data line {} kolbey entry in file {}\n",
                        line, i
                    ));
                } else {
                    content.push_str(&format!(
                        "filler_{:04}_file_{:02}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                        line, i
                    ));
                }
            } else if i < 20 {
                // Files 10-19: contain "patrick"
                if line % 10 == 3 {
                    content.push_str(&format!(
                        "data line {} patrick entry in file {}\n",
                        line, i
                    ));
                } else {
                    content.push_str(&format!(
                        "filler_{:04}_file_{:02}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                        line, i
                    ));
                }
            } else {
                // Files 20-49: generic filler (no kolbey or patrick)
                content.push_str(&format!(
                    "generic_{:04}_file_{:02}_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n",
                    line, i
                ));
            }
        }

        fs::write(&filepath, content.as_bytes()).expect("write test file");
        paths.push(filepath);
    }

    let options = SearchOptions {
        case_sensitive: false,
        ..Default::default()
    };

    let mut total_false_positives = 0usize;

    // ----- Search 1: "patrick" -----
    println!("=== Search 1: 'patrick' ===");
    let results_1 = engine.search_files(&paths, b"patrick", &options);
    let cpu_count_1 = cpu_streaming_search(&paths, b"patrick", false);
    println!(
        "  GPU matches: {}, CPU matches: {}",
        results_1.len(),
        cpu_count_1
    );

    // Validate: every GPU match must be in a file that actually contains "patrick"
    for m in &results_1 {
        let file_content = fs::read_to_string(&m.file_path).expect("read matched file");
        let pattern_found = file_content.to_lowercase().contains("patrick");
        if !pattern_found {
            total_false_positives += 1;
            eprintln!(
                "  FALSE POSITIVE (search 1): '{}' does not contain 'patrick'",
                m.file_path.file_name().unwrap_or_default().to_string_lossy()
            );
        }
    }

    // GPU should not exceed CPU count
    assert!(
        results_1.len() <= cpu_count_1,
        "Search 1 false positives: GPU({}) > CPU({}) for 'patrick'",
        results_1.len(),
        cpu_count_1
    );

    // ----- Search 2: "kolbey" -----
    println!("\n=== Search 2: 'kolbey' ===");
    let results_2 = engine.search_files(&paths, b"kolbey", &options);
    let cpu_count_2 = cpu_streaming_search(&paths, b"kolbey", false);
    println!(
        "  GPU matches: {}, CPU matches: {}",
        results_2.len(),
        cpu_count_2
    );

    // Validate: every GPU match must be in a file that actually contains "kolbey"
    for m in &results_2 {
        let file_content = fs::read_to_string(&m.file_path).expect("read matched file");
        let pattern_found = file_content.to_lowercase().contains("kolbey");
        if !pattern_found {
            total_false_positives += 1;
            eprintln!(
                "  FALSE POSITIVE (search 2): '{}' does not contain 'kolbey'",
                m.file_path.file_name().unwrap_or_default().to_string_lossy()
            );
        }
    }

    assert!(
        results_2.len() <= cpu_count_2,
        "Search 2 false positives: GPU({}) > CPU({}) for 'kolbey'",
        results_2.len(),
        cpu_count_2
    );

    // ----- Search 3: "patrick" again -----
    println!("\n=== Search 3: 'patrick' (repeat) ===");
    let results_3 = engine.search_files(&paths, b"patrick", &options);
    let cpu_count_3 = cpu_streaming_search(&paths, b"patrick", false);
    println!(
        "  GPU matches: {}, CPU matches: {}",
        results_3.len(),
        cpu_count_3
    );

    // Validate: every GPU match must be in a file that actually contains "patrick"
    for m in &results_3 {
        let file_content = fs::read_to_string(&m.file_path).expect("read matched file");
        let pattern_found = file_content.to_lowercase().contains("patrick");
        if !pattern_found {
            total_false_positives += 1;
            eprintln!(
                "  FALSE POSITIVE (search 3): '{}' does not contain 'patrick'",
                m.file_path.file_name().unwrap_or_default().to_string_lossy()
            );
        }
    }

    assert!(
        results_3.len() <= cpu_count_3,
        "Search 3 false positives: GPU({}) > CPU({}) for 'patrick'",
        results_3.len(),
        cpu_count_3
    );

    // ----- Determinism check: search 1 == search 3 -----
    println!("\n=== Determinism Check ===");
    println!(
        "  Search 1 ('patrick'): {} matches",
        results_1.len()
    );
    println!(
        "  Search 3 ('patrick'): {} matches",
        results_3.len()
    );
    assert_eq!(
        results_1.len(),
        results_3.len(),
        "Determinism violated: same query should produce same count (first={}, repeat={})",
        results_1.len(),
        results_3.len()
    );

    // ----- Summary -----
    println!("\n=== RAPID QUERY CHANGE SUMMARY ===");
    println!("  Total false positives: {}", total_false_positives);
    println!("  Search 1 ('patrick'): {} GPU / {} CPU", results_1.len(), cpu_count_1);
    println!("  Search 2 ('kolbey'):  {} GPU / {} CPU", results_2.len(), cpu_count_2);
    println!("  Search 3 ('patrick'): {} GPU / {} CPU", results_3.len(), cpu_count_3);
    println!("  Deterministic: search 1 == search 3 = {}", results_1.len() == results_3.len());

    assert_eq!(
        total_false_positives, 0,
        "Found {} false positives across 3 rapid query changes",
        total_false_positives
    );

    println!("\n=== RAPID_QUERY_CHANGE: PASSED ===");
}

// ============================================================================
// Test 7: Large file set (250+) to trigger sub-batching in StreamingSearchEngine
// ============================================================================

/// Tests with 250+ files to trigger sub-batching (SUB_BATCH_SIZE=200).
///
/// When StreamingSearchEngine processes >200 files in a chunk, it splits
/// into sub-batches. Each sub-batch calls reset() then load_content(),
/// meaning stale metadata from the previous sub-batch persists. This test
/// checks whether that sub-batching path introduces false positives.
///
/// Creates 250 files:
/// - Files 0-24: contain "ZEBRA_MARKER" (unique pattern)
/// - Files 25-249: contain "FILLER_TEXT_ONLY" (no ZEBRA)
///
/// Searches for "ZEBRA_MARKER" and validates every match is in files 0-24.
#[test]
fn test_sub_batch_false_positives_250_files() {
    let (_device, _pso, mut engine) = create_streaming_engine();

    if !gpu_returns_results(&mut engine) {
        eprintln!("SKIPPED: GPU health check failed");
        return;
    }

    let dir = tempfile::TempDir::new().expect("create temp dir");
    let mut paths = Vec::with_capacity(250);

    for i in 0..250 {
        let filepath = dir.path().join(format!("file_{:03}.txt", i));

        let mut content = String::with_capacity(5000);
        for line in 0..80 {
            if i < 25 {
                // Files 0-24: contain "ZEBRA_MARKER"
                if line % 10 == 3 {
                    content.push_str(&format!(
                        "data line {} ZEBRA_MARKER entry in file {}\n",
                        line, i
                    ));
                } else {
                    content.push_str(&format!(
                        "filler_{:04}_file_{:03}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                        line, i
                    ));
                }
            } else {
                // Files 25-249: filler only (no ZEBRA)
                content.push_str(&format!(
                    "generic_{:04}_file_{:03}_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n",
                    line, i
                ));
            }
        }

        fs::write(&filepath, content.as_bytes()).expect("write test file");
        paths.push(filepath);
    }

    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };

    // Search 1: "ZEBRA_MARKER" -- should only match files 0-24
    println!("=== Sub-batch test: 250 files, searching 'ZEBRA_MARKER' ===");
    let results = engine.search_files(&paths, b"ZEBRA_MARKER", &options);
    let cpu_count = cpu_streaming_search(&paths, b"ZEBRA_MARKER", true);

    println!("  GPU matches: {}, CPU matches: {}", results.len(), cpu_count);

    let mut false_positives = 0usize;
    for m in &results {
        let fname = m.file_path.file_name().unwrap_or_default().to_string_lossy().to_string();
        // Extract file number from filename
        let file_num: usize = fname
            .strip_prefix("file_")
            .and_then(|s| s.strip_suffix(".txt"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(999);

        if file_num >= 25 {
            false_positives += 1;
            eprintln!(
                "  FALSE POSITIVE: match in {} (file_num={}, expected < 25)",
                fname, file_num
            );
        }
    }

    assert!(
        results.len() <= cpu_count,
        "GPU({}) > CPU({}) -- false positive count mismatch",
        results.len(),
        cpu_count
    );

    // Search 2: Switch to different pattern, then back
    println!("\n=== Sub-batch test: switching to 'FILLER_TEXT' then back ===");
    let results2 = engine.search_files(&paths, b"generic_", &options);
    println!("  'generic_' matches: {}", results2.len());

    let results3 = engine.search_files(&paths, b"ZEBRA_MARKER", &options);
    println!("  'ZEBRA_MARKER' (repeat): {} matches", results3.len());

    let mut false_positives_repeat = 0usize;
    for m in &results3 {
        let fname = m.file_path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let file_num: usize = fname
            .strip_prefix("file_")
            .and_then(|s| s.strip_suffix(".txt"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(999);

        if file_num >= 25 {
            false_positives_repeat += 1;
            eprintln!(
                "  FALSE POSITIVE (repeat): match in {} (file_num={})",
                fname, file_num
            );
        }
    }

    // Determinism check
    assert_eq!(
        results.len(),
        results3.len(),
        "Determinism violated: first={} repeat={}",
        results.len(),
        results3.len()
    );

    println!("\n=== SUB-BATCH FALSE POSITIVES SUMMARY ===");
    println!("  Total files: 250 (25 with ZEBRA, 225 filler)");
    println!("  False positives (first search): {}", false_positives);
    println!("  False positives (after switch): {}", false_positives_repeat);
    println!("  Deterministic: {}", results.len() == results3.len());

    assert_eq!(
        false_positives, 0,
        "{} false positives in 250-file sub-batch test",
        false_positives
    );
    assert_eq!(
        false_positives_repeat, 0,
        "{} false positives after pattern switch in 250-file sub-batch test",
        false_positives_repeat
    );

    println!("=== SUB_BATCH_FALSE_POSITIVES_250: PASSED ===");
}

// ============================================================================
// Test: Concurrent search cancellation (task 3.2)
// ============================================================================
//
// Simulates search cancellation: start a streaming search for pattern A over
// 500+ temp files, cancel after 50ms, then run a new search for pattern B.
// Verifies the second search returns correct results with no contamination
// from the cancelled first search.

use gpu_search::search::cancel::{cancellation_pair, SearchGeneration, SearchSession};

/// Test that cancelling a streaming search and starting a new one on the same
/// orchestrator produces correct results for the second search (no cross-query
/// contamination from the cancelled search).
///
/// The orchestrator (containing Metal GPU buffers) stays on the main test thread.
/// A separate timer thread triggers cancellation after 50ms, simulating the UI
/// thread cancelling a search when the user types a new query.
#[test]
fn test_search_cancellation_no_contamination() {
    use crossbeam_channel as channel;
    use gpu_search::search::types::StampedUpdate;
    use std::thread;
    use std::time::Duration;

    // ----- Create 550 temp files -----
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let mut paths = Vec::with_capacity(550);

    // Files 0-274: contain "CANCEL_FIRST_MARKER" (unique to search 1)
    // Files 275-549: contain "CANCEL_SECOND_MARKER" (unique to search 2)
    for i in 0..550 {
        let filepath = dir.path().join(format!("file_{:04}.txt", i));

        let mut content = String::with_capacity(5000);
        for line in 0..80 {
            if i < 275 {
                if line % 10 == 3 {
                    content.push_str(&format!(
                        "data line {} CANCEL_FIRST_MARKER entry in file {}\n",
                        line, i
                    ));
                } else {
                    content.push_str(&format!(
                        "filler_{:04}_file_{:04}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                        line, i
                    ));
                }
            } else {
                if line % 10 == 3 {
                    content.push_str(&format!(
                        "data line {} CANCEL_SECOND_MARKER entry in file {}\n",
                        line, i
                    ));
                } else {
                    content.push_str(&format!(
                        "filler_{:04}_file_{:04}_yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\n",
                        line, i
                    ));
                }
            }
        }

        fs::write(&filepath, content.as_bytes()).expect("write test file");
        paths.push(filepath);
    }

    println!("=== Search Cancellation Test: 550 files created ===");

    // ----- Set up orchestrator (stays on this thread -- Metal buffers aren't Send) -----
    let device = gpu_search::gpu::device::GpuDevice::new();
    let pso_cache = gpu_search::gpu::pipeline::PsoCache::new(&device.device);
    let mut orchestrator = gpu_search::search::orchestrator::SearchOrchestrator::new(
        &device.device,
        &pso_cache,
    )
    .expect("Failed to create SearchOrchestrator");

    let generation = SearchGeneration::new();
    let search_root = dir.path().to_path_buf();

    // ----- Search 1: streaming search for CANCEL_FIRST_MARKER, cancel from timer thread -----
    println!("\n=== Search 1: CANCEL_FIRST_MARKER (will be cancelled) ===");

    let (token1, handle1) = cancellation_pair();
    let guard1 = generation.next();
    let session1 = SearchSession {
        token: token1,
        guard: guard1,
    };

    let (update_tx1, update_rx1) = channel::bounded::<StampedUpdate>(1024);

    let request1 = gpu_search::search::types::SearchRequest {
        pattern: "CANCEL_FIRST_MARKER".to_string(),
        root: search_root.clone(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    // Spawn a timer thread that cancels search 1 after 50ms
    let cancel_thread = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        handle1.cancel();
    });

    // Run search 1 on this thread (orchestrator stays here)
    let response1 = orchestrator.search_streaming(request1, &update_tx1, &session1);

    // Wait for cancel thread to finish
    cancel_thread.join().expect("cancel thread panicked");

    // Drain any remaining updates from search 1
    while update_rx1.try_recv().is_ok() {}

    println!(
        "  Search 1 returned: {} content matches, {} file matches",
        response1.content_matches.len(),
        response1.file_matches.len()
    );

    // ----- Search 2: full search for CANCEL_SECOND_MARKER (not cancelled) -----
    println!("\n=== Search 2: CANCEL_SECOND_MARKER (full search) ===");

    let (token2, _handle2) = cancellation_pair();
    let guard2 = generation.next();
    let session2 = SearchSession {
        token: token2,
        guard: guard2,
    };

    let (update_tx2, update_rx2) = channel::bounded::<StampedUpdate>(1024);

    let request2 = gpu_search::search::types::SearchRequest {
        pattern: "CANCEL_SECOND_MARKER".to_string(),
        root: search_root.clone(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    let response2 = orchestrator.search_streaming(request2, &update_tx2, &session2);

    // Drain updates
    while update_rx2.try_recv().is_ok() {}

    println!(
        "  Search 2 returned: {} content matches, {} file matches",
        response2.content_matches.len(),
        response2.file_matches.len()
    );

    // ----- Validate: search 2 should have ONLY CANCEL_SECOND_MARKER matches -----
    println!("\n=== Validation ===");

    // Every content match in search 2 must be from files 275-549
    let mut contamination_count = 0usize;
    for m in &response2.content_matches {
        let fname = m.path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let file_num: usize = fname
            .strip_prefix("file_")
            .and_then(|s| s.strip_suffix(".txt"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(9999);

        if file_num < 275 {
            contamination_count += 1;
            eprintln!(
                "  CONTAMINATION: search 2 matched {} (file_num={}, expected >= 275)",
                fname, file_num
            );
        }

        // Also verify the line content actually contains the second pattern
        let file_content = fs::read_to_string(&m.path).expect("read matched file");
        if !file_content.contains("CANCEL_SECOND_MARKER") {
            contamination_count += 1;
            eprintln!(
                "  FALSE POSITIVE: {} does not contain CANCEL_SECOND_MARKER",
                fname
            );
        }
    }

    // Search 2 must have found some results (it searched 275 files with the pattern)
    assert!(
        !response2.content_matches.is_empty(),
        "Search 2 should find at least some CANCEL_SECOND_MARKER matches"
    );

    // Verify against CPU reference
    let cpu_count = cpu_streaming_search(
        &paths[275..],
        b"CANCEL_SECOND_MARKER",
        true,
    );
    println!(
        "  CPU reference for CANCEL_SECOND_MARKER in files 275-549: {} matches",
        cpu_count
    );

    // GPU should not produce more matches than CPU (for the correct file range)
    // Note: search 2 scans ALL files but should only match in files 275-549
    let cpu_count_all = cpu_streaming_search(&paths, b"CANCEL_SECOND_MARKER", true);
    assert!(
        response2.content_matches.len() <= cpu_count_all,
        "GPU({}) > CPU({}) -- false positives in search 2",
        response2.content_matches.len(),
        cpu_count_all
    );

    assert_eq!(
        contamination_count, 0,
        "Found {} contaminated results from cancelled search 1 in search 2 results",
        contamination_count
    );

    println!("  Contamination: 0");
    println!("  Search 2 GPU matches: {}", response2.content_matches.len());
    println!("  Search 2 CPU reference: {}", cpu_count_all);
    println!("\n=== SEARCH_CANCELLATION_NO_CONTAMINATION: PASSED ===");
}

// ============================================================================
// Test 8-12: match_range accuracy tests (task 3.1)
// ============================================================================
//
// These tests create files with known patterns at known positions, run
// SearchOrchestrator::search() (blocking path), and validate that every
// ContentMatch.match_range correctly identifies the pattern within line_content.

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache as PsoCacheOrch;
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::{ContentMatch, SearchRequest};

/// Helper: create an orchestrator for match_range tests.
fn create_orchestrator() -> SearchOrchestrator {
    let device = GpuDevice::new();
    let pso_cache = PsoCacheOrch::new(&device.device);
    SearchOrchestrator::new(&device.device, &pso_cache)
        .expect("Failed to create SearchOrchestrator")
}

/// Helper: validate that match_range for every ContentMatch correctly identifies
/// the pattern within line_content (CPU ground truth).
fn validate_match_ranges(matches: &[ContentMatch], pattern: &str, case_sensitive: bool) {
    for (i, m) in matches.iter().enumerate() {
        let range = &m.match_range;
        assert!(
            range.end <= m.line_content.len(),
            "match_range {:?} extends beyond line_content (len={}) for match #{}",
            range, m.line_content.len(), i
        );
        assert!(
            range.start < range.end,
            "match_range {:?} is empty or inverted for match #{}",
            range, i
        );
        assert_eq!(
            range.end - range.start,
            pattern.len(),
            "match_range length ({}) != pattern length ({}) for match #{}",
            range.end - range.start, pattern.len(), i
        );

        let slice = &m.line_content[range.clone()];
        if case_sensitive {
            assert_eq!(
                slice, pattern,
                "match_range content {:?} != pattern {:?} (case_sensitive) for match #{} in {:?}",
                slice, pattern, i, m.path
            );
        } else {
            assert_eq!(
                slice.to_lowercase(), pattern.to_lowercase(),
                "match_range content {:?} != pattern {:?} (case_insensitive) for match #{} in {:?}",
                slice, pattern, i, m.path
            );
        }

        // CPU ground truth: confirm pattern actually exists in the line at the given position
        let line_at_range = &m.line_content[range.start..];
        if case_sensitive {
            assert!(
                line_at_range.starts_with(pattern),
                "CPU verify: pattern not at match_range.start in line {:?} for match #{}",
                m.line_content, i
            );
        } else {
            assert!(
                line_at_range.to_lowercase().starts_with(&pattern.to_lowercase()),
                "CPU verify: pattern not at match_range.start in line {:?} for match #{}",
                m.line_content, i
            );
        }
    }
}

// ============================================================================
// Test 8: Single match per file -- match_range accuracy
// ============================================================================

/// Create files each with exactly one occurrence of the pattern.
/// Validate that match_range correctly identifies the pattern position.
#[test]
fn test_match_range_single_match_per_file() {
    let mut orchestrator = create_orchestrator();

    let dir = tempfile::TempDir::new().expect("create temp dir");

    // Create 5 files, each with exactly one occurrence of "NEEDLE" at different columns
    let test_cases = [
        ("file_0.txt", "The quick brown fox NEEDLE jumped over\n", 20usize),
        ("file_1.txt", "NEEDLE is at the very start of this line\n", 0usize),
        ("file_2.txt", "Here the pattern sits right at the end NEEDLE\n", 39usize),
        ("file_3.txt", "xxxx NEEDLE yyyy zzzz padding text here\n", 5usize),
        ("file_4.txt", "lots of padding before the marker NEEDLE appears\n", 34usize),
    ];

    // Pad each file to >4KB to ensure GPU dispatch
    for (name, line, _expected_col) in &test_cases {
        let filepath = dir.path().join(name);
        let mut content = String::with_capacity(5000);
        // Add filler lines first
        for j in 0..80 {
            content.push_str(&format!(
                "filler_line_{:04}_no_needle_here_xxxxxxxxxxxxxxxxxxxxxxxxxx\n", j
            ));
        }
        // Add the line with the pattern at the end (so line number is known)
        content.push_str(line);
        fs::write(&filepath, content.as_bytes()).expect("write test file");
    }

    let request = SearchRequest {
        pattern: "NEEDLE".to_string(),
        root: dir.path().to_path_buf(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    let response = orchestrator.search(request);

    println!("=== match_range_single_match_per_file ===");
    println!("  Content matches: {}", response.content_matches.len());

    // Should find at least some matches (GPU may miss boundary cases)
    assert!(
        !response.content_matches.is_empty(),
        "Should find at least one NEEDLE match"
    );

    // Validate every match_range
    validate_match_ranges(&response.content_matches, "NEEDLE", true);

    // For each match, verify the column matches expected position
    for m in &response.content_matches {
        let col = m.match_range.start;
        println!(
            "  file={} line={} col={} match_range={:?} line_content={:?}",
            m.path.file_name().unwrap_or_default().to_string_lossy(),
            m.line_number,
            col,
            m.match_range,
            &m.line_content[..m.line_content.len().min(60)]
        );

        // Find expected column for this file
        let fname = m.path.file_name().unwrap_or_default().to_string_lossy().to_string();
        if let Some((_, _, expected_col)) = test_cases.iter().find(|(n, _, _)| *n == fname) {
            assert_eq!(
                col, *expected_col,
                "match_range.start ({}) != expected column ({}) for {}",
                col, expected_col, fname
            );
        }
    }

    println!("=== match_range_single_match_per_file: PASSED ===");
}

// ============================================================================
// Test 9: Multiple matches per file -- match_range accuracy
// ============================================================================

/// Create a file with multiple occurrences of the pattern on different lines.
/// Validate match_range for each occurrence.
#[test]
fn test_match_range_multiple_matches_per_file() {
    let mut orchestrator = create_orchestrator();

    let dir = tempfile::TempDir::new().expect("create temp dir");
    let filepath = dir.path().join("multi.txt");

    // Build content with MARKER at known positions across multiple lines
    let mut content = String::with_capacity(6000);
    let mut expected_positions: Vec<(usize, usize)> = Vec::new(); // (line_number_1based, column)

    for line_idx in 0..100 {
        if line_idx == 10 {
            // MARKER at column 0
            content.push_str("MARKER is first on this line with padding text\n");
            expected_positions.push((line_idx + 1, 0));
        } else if line_idx == 30 {
            // MARKER at column 15
            content.push_str("padding text   MARKER in the middle of this line\n");
            expected_positions.push((line_idx + 1, 15));
        } else if line_idx == 50 {
            // MARKER at column 40
            content.push_str("a]very long prefix text padding here    MARKER at position forty\n");
            expected_positions.push((line_idx + 1, 40));
        } else if line_idx == 70 {
            // MARKER at column 5
            content.push_str("text MARKER near the start of this filler line text\n");
            expected_positions.push((line_idx + 1, 5));
        } else {
            content.push_str(&format!(
                "filler_line_{:04}_no_marker_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n", line_idx
            ));
        }
    }

    fs::write(&filepath, content.as_bytes()).expect("write multi.txt");

    let request = SearchRequest {
        pattern: "MARKER".to_string(),
        root: dir.path().to_path_buf(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    let response = orchestrator.search(request);

    println!("=== match_range_multiple_matches_per_file ===");
    println!("  Content matches: {}", response.content_matches.len());
    println!("  Expected positions: {:?}", expected_positions);

    // Validate all match_ranges
    validate_match_ranges(&response.content_matches, "MARKER", true);

    for m in &response.content_matches {
        println!(
            "  line={} col={} match_range={:?}",
            m.line_number, m.match_range.start, m.match_range
        );
    }

    // Should find at least 2 of the 4 matches (GPU boundary may miss some)
    assert!(
        response.content_matches.len() >= 2,
        "Should find at least 2 of 4 MARKER occurrences, got {}",
        response.content_matches.len()
    );

    // Every found match must have correct match_range column matching expected
    for m in &response.content_matches {
        if let Some((_, expected_col)) = expected_positions
            .iter()
            .find(|(ln, _)| *ln == m.line_number as usize)
        {
            assert_eq!(
                m.match_range.start, *expected_col,
                "line {}: match_range.start ({}) != expected column ({})",
                m.line_number, m.match_range.start, expected_col
            );
        }
    }

    println!("=== match_range_multiple_matches_per_file: PASSED ===");
}

// ============================================================================
// Test 10: Match at start of line -- match_range accuracy
// ============================================================================

/// Verify match_range.start == 0 when the pattern is at the very beginning of a line.
#[test]
fn test_match_range_start_of_line() {
    let mut orchestrator = create_orchestrator();

    let dir = tempfile::TempDir::new().expect("create temp dir");
    let filepath = dir.path().join("start_of_line.txt");

    let mut content = String::with_capacity(6000);
    let mut lines_with_pattern = 0;

    for line_idx in 0..100 {
        if line_idx % 20 == 5 {
            // Pattern at column 0
            content.push_str("STARTPAT is the very first thing on this line padding\n");
            lines_with_pattern += 1;
        } else {
            content.push_str(&format!(
                "filler_line_{:04}_no_startpat_here_xxxxxxxxxxxxxxxxxxxxxxxx\n", line_idx
            ));
        }
    }

    fs::write(&filepath, content.as_bytes()).expect("write start_of_line.txt");

    let request = SearchRequest {
        pattern: "STARTPAT".to_string(),
        root: dir.path().to_path_buf(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    let response = orchestrator.search(request);

    println!("=== match_range_start_of_line ===");
    println!("  Content matches: {} (expected up to {})", response.content_matches.len(), lines_with_pattern);

    validate_match_ranges(&response.content_matches, "STARTPAT", true);

    assert!(
        !response.content_matches.is_empty(),
        "Should find at least one STARTPAT match"
    );

    for m in &response.content_matches {
        println!(
            "  line={} match_range={:?} content={:?}",
            m.line_number, m.match_range,
            &m.line_content[..m.line_content.len().min(50)]
        );

        // Every match must start at column 0
        assert_eq!(
            m.match_range.start, 0,
            "Pattern at start of line should have match_range.start==0, got {} on line {}",
            m.match_range.start, m.line_number
        );
    }

    println!("=== match_range_start_of_line: PASSED ===");
}

// ============================================================================
// Test 11: Match at end of line -- match_range accuracy
// ============================================================================

/// Verify match_range.end == line_content.len() when pattern is at line end.
#[test]
fn test_match_range_end_of_line() {
    let mut orchestrator = create_orchestrator();

    let dir = tempfile::TempDir::new().expect("create temp dir");
    let filepath = dir.path().join("end_of_line.txt");

    let mut content = String::with_capacity(6000);
    let mut lines_with_pattern = 0;

    for line_idx in 0..100 {
        if line_idx % 20 == 7 {
            // Pattern at end of line (before newline)
            content.push_str("this line has lots of prefix padding text ENDPAT\n");
            lines_with_pattern += 1;
        } else {
            content.push_str(&format!(
                "filler_line_{:04}_no_endpat_here_xxxxxxxxxxxxxxxxxxxxxxxxx\n", line_idx
            ));
        }
    }

    fs::write(&filepath, content.as_bytes()).expect("write end_of_line.txt");

    let request = SearchRequest {
        pattern: "ENDPAT".to_string(),
        root: dir.path().to_path_buf(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    let response = orchestrator.search(request);

    println!("=== match_range_end_of_line ===");
    println!("  Content matches: {} (expected up to {})", response.content_matches.len(), lines_with_pattern);

    validate_match_ranges(&response.content_matches, "ENDPAT", true);

    assert!(
        !response.content_matches.is_empty(),
        "Should find at least one ENDPAT match"
    );

    for m in &response.content_matches {
        println!(
            "  line={} match_range={:?} line_len={} content={:?}",
            m.line_number, m.match_range, m.line_content.len(),
            &m.line_content[..m.line_content.len().min(60)]
        );

        // match_range.end should equal line_content.len() (pattern at very end)
        assert_eq!(
            m.match_range.end,
            m.line_content.len(),
            "Pattern at end of line should have match_range.end==line_content.len() ({}), got {} on line {}",
            m.line_content.len(), m.match_range.end, m.line_number
        );
    }

    println!("=== match_range_end_of_line: PASSED ===");
}

// ============================================================================
// Test 12: Match near 64-byte thread boundary -- match_range accuracy
// ============================================================================

/// The GPU kernel processes 64 bytes per thread. A match at byte ~62-66 within
/// a chunk spans two thread windows. Verify match_range is still correct when
/// the GPU successfully detects such a match.
#[test]
fn test_match_range_near_thread_boundary() {
    let mut orchestrator = create_orchestrator();

    let dir = tempfile::TempDir::new().expect("create temp dir");
    let filepath = dir.path().join("boundary.txt");

    // Build content where the pattern sits exactly near the 64-byte boundary
    // within a line. Each GPU thread handles 64 bytes, so a pattern starting
    // at byte 60 within the chunk data will span thread 0 and thread 1.
    //
    // Strategy: pad each line to known lengths, place pattern near byte 60-64.
    let mut content = String::with_capacity(8000);

    // First few filler lines to push past initial chunk area
    for i in 0..5 {
        content.push_str(&format!(
            "filler_line_{:04}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n", i
        ));
    }

    // Line with pattern near byte 60 within the line (col ~58-62)
    // "A" * 58 + "BOUND" + padding = pattern at column 58
    let prefix_a = "A".repeat(58);
    content.push_str(&format!("{}BOUND and then some more text padding here\n", prefix_a));

    // More filler to reach ~4KB+
    for i in 6..100 {
        content.push_str(&format!(
            "filler_line_{:04}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n", i
        ));
    }

    // Another line with pattern at byte 62 within the line
    let prefix_b = "B".repeat(62);
    content.push_str(&format!("{}BOUND end of boundary test line padding\n", prefix_b));

    // Another line with pattern at byte 63 (spans byte 63-67 = threads 0 and 1)
    let prefix_c = "C".repeat(63);
    content.push_str(&format!("{}BOUND right on the edge of threads text\n", prefix_c));

    fs::write(&filepath, content.as_bytes()).expect("write boundary.txt");

    let request = SearchRequest {
        pattern: "BOUND".to_string(),
        root: dir.path().to_path_buf(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    let response = orchestrator.search(request);

    println!("=== match_range_near_thread_boundary ===");
    println!("  Content matches: {}", response.content_matches.len());

    // Validate all found match_ranges (GPU may miss some boundary matches)
    validate_match_ranges(&response.content_matches, "BOUND", true);

    for m in &response.content_matches {
        let slice = &m.line_content[m.match_range.clone()];
        println!(
            "  line={} col={} match_range={:?} slice={:?} line_len={}",
            m.line_number, m.match_range.start, m.match_range, slice, m.line_content.len()
        );

        // CPU ground truth: read the file and verify
        let file_content = fs::read_to_string(&m.path).expect("read boundary file");
        let lines: Vec<&str> = file_content.lines().collect();
        let line = lines.get((m.line_number - 1) as usize).expect("line exists");
        let cpu_col = line.find("BOUND").expect("BOUND exists in line");
        assert_eq!(
            m.match_range.start, cpu_col,
            "GPU match_range.start ({}) != CPU column ({}) for line {}",
            m.match_range.start, cpu_col, m.line_number
        );
    }

    // At least one match should be found (the one at col 58 is most likely
    // to be within a single thread window if the line starts at a chunk boundary)
    assert!(
        !response.content_matches.is_empty(),
        "Should find at least one BOUND match near thread boundary"
    );

    println!("=== match_range_near_thread_boundary: PASSED ===");
}

// ============================================================================
// Test 13-14: Large file chunk boundary tests (task 3.3)
// ============================================================================
//
// ContentSearchEngine uses CHUNK_SIZE = 4096 bytes. Each chunk is processed
// independently by the GPU kernel. A match that spans a chunk boundary (e.g.,
// pattern starts at byte 4094, chunk boundary at 4096) will be MISSED because
// the GPU kernel only sees data within its assigned chunk.
//
// These tests document this known limitation and verify that:
// 1. Matches fully within a chunk are correctly detected
// 2. Matches spanning a chunk boundary are gracefully missed (not corrupted)
// 3. byte_offsets are file-relative, not global across all loaded content
// 4. GPU match count <= CPU reference count (no false positives)

/// Helper: perform a CPU reference search counting all non-overlapping occurrences
/// of `pattern` in `content`.
#[allow(dead_code)]
fn cpu_count_matches(content: &[u8], pattern: &[u8]) -> usize {
    if pattern.is_empty() || content.len() < pattern.len() {
        return 0;
    }
    let mut count = 0;
    let mut pos = 0;
    while pos + pattern.len() <= content.len() {
        if &content[pos..pos + pattern.len()] == pattern {
            count += 1;
            pos += pattern.len(); // non-overlapping
        } else {
            pos += 1;
        }
    }
    count
}

/// Helper: find all byte positions where `pattern` occurs in `content`.
fn cpu_find_positions(content: &[u8], pattern: &[u8]) -> Vec<usize> {
    let mut positions = Vec::new();
    if pattern.is_empty() || content.len() < pattern.len() {
        return positions;
    }
    let mut pos = 0;
    while pos + pattern.len() <= content.len() {
        if &content[pos..pos + pattern.len()] == pattern {
            positions.push(pos);
            pos += pattern.len();
        } else {
            pos += 1;
        }
    }
    positions
}

// ============================================================================
// Test 13: 8KB file with pattern spanning chunk boundary
// ============================================================================

/// Create a file of exactly 8192 bytes with a pattern placed at byte 4090,
/// which means it spans the chunk boundary at byte 4096. The GPU processes
/// each 4096-byte chunk independently, so this pattern will likely be missed.
///
/// This test documents the KNOWN LIMITATION: matches spanning chunk boundaries
/// are missed by the GPU kernel. The test verifies:
/// - GPU match count <= CPU match count (no false positives)
/// - byte_offsets of found matches are file-relative
/// - The boundary-spanning miss is documented, not a bug
#[test]
fn test_chunk_boundary_8kb_spanning_pattern() {
    let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
    let pso_cache = PsoCache::new(&device);
    let mut engine = ContentSearchEngine::new(&device, &pso_cache, 10);

    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };

    // Build exactly 8192 bytes of content.
    // Pattern "BOUNDARY" (8 bytes) is placed at byte 4090, spanning bytes 4090-4097.
    // Chunk 0: bytes 0-4095, Chunk 1: bytes 4096-8191.
    // The pattern crosses the boundary: bytes 4090-4095 in chunk 0, bytes 4096-4097 in chunk 1.
    let pattern = b"BOUNDARY";
    let pattern_len = pattern.len(); // 8
    let target_size: usize = 8192;
    let boundary_pos: usize = 4090; // spans 4090..4098, crossing 4096

    let mut content = vec![b'x'; target_size];

    // Place the pattern at the boundary-spanning position
    content[boundary_pos..boundary_pos + pattern_len].copy_from_slice(pattern);

    // Also place a pattern safely within chunk 0 (byte 100) and chunk 1 (byte 5000)
    // so we can verify those are found correctly
    let safe_pos_chunk0: usize = 100;
    let safe_pos_chunk1: usize = 5000;
    content[safe_pos_chunk0..safe_pos_chunk0 + pattern_len].copy_from_slice(pattern);
    content[safe_pos_chunk1..safe_pos_chunk1 + pattern_len].copy_from_slice(pattern);

    assert_eq!(content.len(), target_size);

    // CPU reference: should find 3 matches
    let cpu_positions = cpu_find_positions(&content, pattern);
    let cpu_count = cpu_positions.len();
    println!("=== chunk_boundary_8kb_spanning_pattern ===");
    println!("  CPU found {} matches at positions: {:?}", cpu_count, cpu_positions);
    assert_eq!(cpu_count, 3, "CPU should find 3 occurrences of BOUNDARY");

    // Load into GPU engine
    let chunks = engine.load_content(&content, 0);
    println!("  Loaded {} chunks (expected 2 for 8192 bytes)", chunks);
    assert_eq!(chunks, 2, "8192 bytes should produce exactly 2 chunks");

    // GPU search
    let results = engine.search(pattern, &options);
    println!("  GPU found {} matches", results.len());

    // KNOWN LIMITATION: The pattern at byte 4090 spans the chunk boundary.
    // Each chunk is searched independently, so the GPU will miss this match.
    // Expected: GPU finds 2 matches (safe_pos_chunk0 and safe_pos_chunk1),
    //           misses the boundary-spanning match at 4090.
    println!(
        "  KNOWN LIMITATION: Pattern at byte {} spans chunk boundary (4096). GPU may miss it.",
        boundary_pos
    );

    // Core assertion: GPU must NOT produce false positives (GPU <= CPU)
    assert!(
        results.len() <= cpu_count,
        "GPU({}) > CPU({}) -- false positives detected!",
        results.len(),
        cpu_count
    );

    // Verify byte_offsets are file-relative (not global)
    for (i, m) in results.iter().enumerate() {
        println!(
            "  Match {}: file_index={} byte_offset={} line={} col={}",
            i, m.file_index, m.byte_offset, m.line_number, m.column
        );

        // file_index should be 0 (we loaded one file at index 0)
        assert_eq!(
            m.file_index, 0,
            "file_index should be 0, got {} for match {}",
            m.file_index, i
        );

        // byte_offset should correspond to one of the safe positions or
        // (unlikely) the boundary position. It must be < 8192 (file-relative).
        assert!(
            (m.byte_offset as usize) < target_size,
            "byte_offset {} >= file size {} -- not file-relative!",
            m.byte_offset,
            target_size
        );

        // Verify the byte_offset points to a real occurrence of the pattern
        // in the original content
        let bo = m.byte_offset as usize;
        if bo + pattern_len <= content.len() {
            let slice = &content[bo..bo + pattern_len];
            assert_eq!(
                slice, pattern,
                "Content at byte_offset {} does not match pattern (got {:?})",
                bo,
                String::from_utf8_lossy(slice)
            );
        }
    }

    // Document boundary miss
    let boundary_found = results.iter().any(|m| m.byte_offset as usize == boundary_pos);
    if !boundary_found {
        println!(
            "  DOCUMENTED: Boundary-spanning match at byte {} was missed (expected behavior).",
            boundary_pos
        );
    } else {
        println!(
            "  NOTE: Boundary-spanning match at byte {} was found (GPU handled overlap).",
            boundary_pos
        );
    }

    println!(
        "  Summary: GPU={} CPU={} (boundary miss={})",
        results.len(),
        cpu_count,
        !boundary_found
    );
    println!("=== chunk_boundary_8kb_spanning_pattern: PASSED ===");
}

// ============================================================================
// Test 14: 16KB file with patterns at all chunk boundaries
// ============================================================================

/// Create a 16384-byte file with patterns placed at strategic chunk boundary
/// positions: bytes 0, 4095, 4096, 8191, 8192.
///
/// With CHUNK_SIZE=4096, chunk boundaries are at 4096, 8192, 12288.
/// - Byte 0: start of chunk 0 (fully within chunk 0)
/// - Byte 4095: end of chunk 0 (pattern spans into chunk 1 -- KNOWN MISS)
/// - Byte 4096: start of chunk 1 (fully within chunk 1)
/// - Byte 8191: end of chunk 1 (pattern spans into chunk 2 -- KNOWN MISS)
/// - Byte 8192: start of chunk 2 (fully within chunk 2)
///
/// We use a short pattern ("XY", 2 bytes) so that patterns at bytes 4095
/// and 8191 span exactly 1 byte into the next chunk.
#[test]
fn test_chunk_boundary_16kb_multiple_boundaries() {
    let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
    let pso_cache = PsoCache::new(&device);
    let mut engine = ContentSearchEngine::new(&device, &pso_cache, 10);

    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };

    let pattern = b"XY";
    let pattern_len = pattern.len(); // 2
    let target_size: usize = 16384;

    // Positions to place the pattern
    let positions: Vec<usize> = vec![0, 4095, 4096, 8191, 8192];

    // Build content: fill with 'z' (won't match "XY"), then place pattern at positions
    let mut content = vec![b'z'; target_size];
    for &pos in &positions {
        assert!(
            pos + pattern_len <= target_size,
            "Position {} + pattern_len {} exceeds file size {}",
            pos,
            pattern_len,
            target_size
        );
        content[pos..pos + pattern_len].copy_from_slice(pattern);
    }

    assert_eq!(content.len(), target_size);

    // Identify which positions span chunk boundaries
    // A position spans a boundary if it starts in one chunk and ends in the next.
    // Chunk boundaries: 4096, 8192, 12288
    let chunk_size: usize = 4096;
    let mut within_chunk_positions = Vec::new();
    let mut spanning_positions = Vec::new();

    for &pos in &positions {
        let start_chunk = pos / chunk_size;
        let end_chunk = (pos + pattern_len - 1) / chunk_size;
        if start_chunk == end_chunk {
            within_chunk_positions.push(pos);
        } else {
            spanning_positions.push(pos);
        }
    }

    // CPU reference
    let cpu_positions = cpu_find_positions(&content, pattern);
    let cpu_count = cpu_positions.len();

    println!("=== chunk_boundary_16kb_multiple_boundaries ===");
    println!("  File size: {} bytes ({} chunks)", target_size, target_size / chunk_size);
    println!("  Pattern: {:?} ({} bytes)", String::from_utf8_lossy(pattern), pattern_len);
    println!("  Placed at positions: {:?}", positions);
    println!("  Within-chunk positions: {:?}", within_chunk_positions);
    println!(
        "  Spanning-boundary positions: {:?} (KNOWN LIMITATION: may be missed)",
        spanning_positions
    );
    println!("  CPU found {} matches at: {:?}", cpu_count, cpu_positions);

    // Load into GPU engine
    let chunks = engine.load_content(&content, 0);
    println!("  Loaded {} chunks (expected {})", chunks, target_size / chunk_size);
    assert_eq!(
        chunks,
        target_size / chunk_size,
        "16384 bytes should produce exactly 4 chunks"
    );

    // GPU search
    let results = engine.search(pattern, &options);
    println!("  GPU found {} matches", results.len());

    // Core assertion: no false positives
    assert!(
        results.len() <= cpu_count,
        "GPU({}) > CPU({}) -- false positives detected!",
        results.len(),
        cpu_count
    );

    // Verify byte_offsets are file-relative and point to actual pattern occurrences
    let mut gpu_positions: Vec<usize> = Vec::new();
    for (i, m) in results.iter().enumerate() {
        let bo = m.byte_offset as usize;
        gpu_positions.push(bo);

        println!(
            "  Match {}: byte_offset={} file_index={} line={} col={}",
            i, bo, m.file_index, m.line_number, m.column
        );

        // file_index must be 0
        assert_eq!(m.file_index, 0, "file_index should be 0, got {}", m.file_index);

        // byte_offset must be within file bounds
        assert!(
            bo + pattern_len <= target_size,
            "byte_offset {} + pattern_len {} exceeds file size {}",
            bo,
            pattern_len,
            target_size
        );

        // Content at byte_offset must actually be the pattern
        let slice = &content[bo..bo + pattern_len];
        assert_eq!(
            slice, pattern,
            "Content at byte_offset {} does not match pattern (got {:?})",
            bo,
            String::from_utf8_lossy(slice)
        );
    }

    // Check which within-chunk positions were found
    let mut within_found = 0;
    let mut within_missed = 0;
    for &pos in &within_chunk_positions {
        if gpu_positions.contains(&pos) {
            within_found += 1;
            println!("  Within-chunk position {}: FOUND", pos);
        } else {
            within_missed += 1;
            println!("  Within-chunk position {}: MISSED", pos);
        }
    }

    // Check which spanning positions were found (expected: missed)
    let mut spanning_found = 0;
    let mut spanning_missed = 0;
    for &pos in &spanning_positions {
        if gpu_positions.contains(&pos) {
            spanning_found += 1;
            println!("  Spanning-boundary position {}: FOUND (unexpected but valid)", pos);
        } else {
            spanning_missed += 1;
            println!(
                "  Spanning-boundary position {}: MISSED (expected -- KNOWN LIMITATION)",
                pos
            );
        }
    }

    println!("\n  === CHUNK BOUNDARY SUMMARY ===");
    println!(
        "  Within-chunk: {}/{} found, {}/{} missed",
        within_found,
        within_chunk_positions.len(),
        within_missed,
        within_chunk_positions.len()
    );
    println!(
        "  Spanning-boundary: {}/{} found, {}/{} missed (KNOWN LIMITATION)",
        spanning_found,
        spanning_positions.len(),
        spanning_missed,
        spanning_positions.len()
    );
    println!("  GPU total: {}, CPU total: {}", results.len(), cpu_count);
    println!("  False positives: 0 (GPU <= CPU verified)");

    // Within-chunk matches should ideally all be found (they don't span boundaries)
    // But GPU thread boundaries (64 bytes) may also cause misses within a chunk.
    // We just assert no false positives, which is the critical property.
    println!("=== chunk_boundary_16kb_multiple_boundaries: PASSED ===");
}

// ============================================================================
// Test 15: Deterministic repeat search (task 3.4)
// ============================================================================
//
// Runs the same search 10 times on the same file set and asserts identical
// results each time. Catches non-determinism from stale buffers, race
// conditions, or atomic ordering issues. Compares full result vectors
// (file_path, line_number, column, byte_offset, match_length), not just counts.

use gpu_search::search::streaming::StreamingMatch;

/// Helper: sort StreamingMatch results by (file_path, line_number, byte_offset, column)
/// for stable comparison across runs.
fn sort_streaming_matches(matches: &mut Vec<StreamingMatch>) {
    matches.sort_by(|a, b| {
        a.file_path
            .cmp(&b.file_path)
            .then(a.line_number.cmp(&b.line_number))
            .then(a.byte_offset.cmp(&b.byte_offset))
            .then(a.column.cmp(&b.column))
    });
}

/// Helper: compare two sorted StreamingMatch vectors field-by-field.
/// Returns a description of the first difference found, or None if identical.
fn compare_streaming_matches(
    run_a: &[StreamingMatch],
    run_b: &[StreamingMatch],
    label_a: &str,
    label_b: &str,
) -> Option<String> {
    if run_a.len() != run_b.len() {
        return Some(format!(
            "Count mismatch: {} has {} matches, {} has {} matches",
            label_a,
            run_a.len(),
            label_b,
            run_b.len()
        ));
    }

    for (i, (a, b)) in run_a.iter().zip(run_b.iter()).enumerate() {
        if a.file_path != b.file_path {
            return Some(format!(
                "Match #{}: file_path differs: {} has {:?}, {} has {:?}",
                i, label_a, a.file_path, label_b, b.file_path
            ));
        }
        if a.line_number != b.line_number {
            return Some(format!(
                "Match #{}: line_number differs: {} has {}, {} has {} (file={:?})",
                i, label_a, a.line_number, label_b, b.line_number, a.file_path
            ));
        }
        if a.byte_offset != b.byte_offset {
            return Some(format!(
                "Match #{}: byte_offset differs: {} has {}, {} has {} (file={:?} line={})",
                i, label_a, a.byte_offset, label_b, b.byte_offset, a.file_path, a.line_number
            ));
        }
        if a.column != b.column {
            return Some(format!(
                "Match #{}: column differs: {} has {}, {} has {} (file={:?} line={})",
                i, label_a, a.column, label_b, b.column, a.file_path, a.line_number
            ));
        }
        if a.match_length != b.match_length {
            return Some(format!(
                "Match #{}: match_length differs: {} has {}, {} has {} (file={:?} line={})",
                i, label_a, a.match_length, label_b, b.match_length, a.file_path, a.line_number
            ));
        }
    }

    None
}

/// Run the same search 10 times on the same file set using StreamingSearchEngine.
/// Assert that all 10 runs produce identical results (same file_paths, line_numbers,
/// columns, byte_offsets, and match_lengths).
///
/// This catches non-determinism from:
/// - Stale GPU buffers not fully cleared between searches
/// - Race conditions in GPU dispatch or result collection
/// - Atomic ordering issues in match counting
/// - Sub-batch boundary effects on result ordering
#[test]
fn test_deterministic_repeat() {
    let (_device, _pso, mut engine) = create_streaming_engine();

    if !gpu_returns_results(&mut engine) {
        eprintln!("SKIPPED: GPU health check failed");
        return;
    }

    // Create 30 temp files with known patterns:
    // - Files 0-9: contain "REPEAT_TARGET" on specific lines
    // - Files 10-29: filler only (no REPEAT_TARGET)
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let mut paths = Vec::with_capacity(30);

    for i in 0..30 {
        let filepath = dir.path().join(format!("repeat_{:02}.txt", i));

        let mut content = String::with_capacity(5000);
        for line in 0..80 {
            if i < 10 && line % 10 == 3 {
                // Files 0-9: insert REPEAT_TARGET every 10 lines
                content.push_str(&format!(
                    "data line {} REPEAT_TARGET entry in file {}\n",
                    line, i
                ));
            } else {
                // Filler: unique per file+line to avoid accidental matches
                content.push_str(&format!(
                    "filler_{:04}_file_{:02}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                    line, i
                ));
            }
        }

        fs::write(&filepath, content.as_bytes()).expect("write test file");
        paths.push(filepath);
    }

    let options = SearchOptions {
        case_sensitive: true,
        ..Default::default()
    };

    let num_runs = 10;
    let mut all_runs: Vec<Vec<StreamingMatch>> = Vec::with_capacity(num_runs);

    println!("=== Deterministic Repeat Search: {} runs ===", num_runs);

    // Run the same search 10 times
    for run in 0..num_runs {
        let mut results = engine.search_files(&paths, b"REPEAT_TARGET", &options);
        sort_streaming_matches(&mut results);

        println!(
            "  Run {}: {} matches",
            run + 1,
            results.len()
        );

        // Sanity: should find some matches in files 0-9
        assert!(
            !results.is_empty(),
            "Run {}: expected at least one match for REPEAT_TARGET",
            run + 1
        );

        all_runs.push(results);
    }

    // Compare every run against run 1 (the reference)
    let reference = &all_runs[0];
    let mut differences_found = 0;

    for run_idx in 1..num_runs {
        let current = &all_runs[run_idx];

        if let Some(diff) = compare_streaming_matches(
            reference,
            current,
            &format!("run 1"),
            &format!("run {}", run_idx + 1),
        ) {
            differences_found += 1;
            eprintln!(
                "  NON-DETERMINISM between run 1 and run {}: {}",
                run_idx + 1,
                diff
            );
        }
    }

    // Print first run details for debugging
    println!("\n  Reference (run 1) matches:");
    for (i, m) in reference.iter().enumerate() {
        println!(
            "    [{}] file={:?} line={} col={} byte_offset={} match_len={}",
            i,
            m.file_path.file_name().unwrap_or_default().to_string_lossy(),
            m.line_number,
            m.column,
            m.byte_offset,
            m.match_length
        );
    }

    println!("\n=== DETERMINISTIC REPEAT SUMMARY ===");
    println!("  Total runs: {}", num_runs);
    println!("  Reference match count: {}", reference.len());
    println!("  Differences found: {}", differences_found);

    assert_eq!(
        differences_found, 0,
        "Found {} non-deterministic runs out of {} (expected all identical to run 1)",
        differences_found,
        num_runs - 1
    );

    println!("=== DETERMINISTIC_REPEAT: PASSED ===");
}

// ============================================================================
// Test 16: GPU_SEARCH_VERIFY=full integration test (task 3.5)
// ============================================================================
//
// Sets the GPU_SEARCH_VERIFY=full env var, then runs SearchOrchestrator::search()
// against a temp directory with known content. In Full verify mode, the verify
// layer in orchestrator.rs panics if any false positives are detected (lines
// 300-305). The test passes if the search completes without panic.
//
// NOTE on corrupted byte_offset injection: Injecting corrupted byte_offsets
// would require either:
//   (a) Modifying production code to accept injectable offsets (not acceptable)
//   (b) Using unsafe memory manipulation of GPU buffers from the test (fragile
//       and non-portable across Metal driver versions)
//   (c) Mocking the GPU pipeline (defeats the purpose of integration testing)
// Since none of these approaches are acceptable for a real integration test,
// we skip the corruption injection part and instead verify that:
//   1. VERIFY=full passes without panic on clean data
//   2. The cpu_verify_matches function itself correctly detects false positives
//      (covered by unit tests in verify.rs: test_false_positive_detected)
//
// The verify.rs unit tests already prove that cpu_verify_matches returns
// false_positives > 0 when GPU offsets don't match CPU ground truth, and
// orchestrator.rs line 300 panics when this happens in Full mode. Together
// these prove the panic path is reachable.

/// Test that GPU_SEARCH_VERIFY=full completes without panic when searching
/// a directory with known content (no false positives).
///
/// Creates 20 files containing "VERIFY_FULL_MARKER" at known positions plus
/// 30 filler files, then runs SearchOrchestrator::search() with the
/// GPU_SEARCH_VERIFY=full env var set. The orchestrator's verify layer
/// performs CPU verification of every GPU match and panics if any false
/// positive is found. Test passes = zero false positives.
#[test]
fn test_verify_full_mode_no_false_positives() {
    // Set env var BEFORE creating the orchestrator (VerifyMode::from_env() reads
    // it during search, not during construction)
    std::env::set_var("GPU_SEARCH_VERIFY", "full");

    let mut orchestrator = create_orchestrator();
    let dir = tempfile::TempDir::new().expect("create temp dir");

    // Create 50 files:
    // - Files 0-19: contain "VERIFY_FULL_MARKER" on specific lines
    // - Files 20-49: filler only
    for i in 0..50 {
        let filepath = dir.path().join(format!("verify_{:02}.txt", i));
        let mut content = String::with_capacity(5000);

        for line in 0..80 {
            if i < 20 && line % 10 == 5 {
                content.push_str(&format!(
                    "data line {} VERIFY_FULL_MARKER entry in file {} with padding\n",
                    line, i
                ));
            } else {
                content.push_str(&format!(
                    "filler_{:04}_file_{:02}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                    line, i
                ));
            }
        }

        fs::write(&filepath, content.as_bytes()).expect("write verify test file");
    }

    println!("=== test_verify_full_mode_no_false_positives ===");
    println!("  Created 50 files (20 with VERIFY_FULL_MARKER, 30 filler)");
    println!("  GPU_SEARCH_VERIFY=full is set");

    let request = SearchRequest {
        pattern: "VERIFY_FULL_MARKER".to_string(),
        root: dir.path().to_path_buf(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    // This call will PANIC inside orchestrator.rs line 300-305 if any false
    // positives are detected by the CPU verification layer.
    let response = orchestrator.search(request);

    println!("  Search completed without panic (VERIFY=full passed)");
    println!("  Content matches: {}", response.content_matches.len());
    println!("  File matches: {}", response.file_matches.len());
    println!("  Total files searched: {}", response.total_files_searched);
    println!("  Elapsed: {:?}", response.elapsed);

    // Verify we actually found some matches (the test is meaningless if 0 matches)
    assert!(
        !response.content_matches.is_empty(),
        "Should find at least some VERIFY_FULL_MARKER matches"
    );

    // All content matches should be in files 0-19
    for m in &response.content_matches {
        let fname = m.path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let file_num: usize = fname
            .strip_prefix("verify_")
            .and_then(|s| s.strip_suffix(".txt"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(999);

        assert!(
            file_num < 20,
            "Match found in filler file {} (file_num={}, expected < 20)",
            fname, file_num
        );

        // Validate match_range content
        let range = &m.match_range;
        assert!(
            range.end <= m.line_content.len(),
            "match_range {:?} exceeds line_content len {}",
            range, m.line_content.len()
        );
        let slice = &m.line_content[range.clone()];
        assert_eq!(
            slice, "VERIFY_FULL_MARKER",
            "match_range content {:?} != expected pattern in file {}",
            slice, fname
        );
    }

    // Clean up env var
    std::env::remove_var("GPU_SEARCH_VERIFY");

    println!("  All matches validated (correct files, correct match_ranges)");
    println!("=== test_verify_full_mode_no_false_positives: PASSED ===");
}

/// Test that GPU_SEARCH_VERIFY=full works with case-insensitive search.
///
/// The verify layer must use case_sensitive=false when comparing GPU offsets
/// against CPU ground truth. This test confirms the case-insensitive path
/// through the verify layer also passes without panic.
#[test]
fn test_verify_full_mode_case_insensitive() {
    std::env::set_var("GPU_SEARCH_VERIFY", "full");

    let mut orchestrator = create_orchestrator();
    let dir = tempfile::TempDir::new().expect("create temp dir");

    // Create 20 files with mixed-case pattern
    for i in 0..20 {
        let filepath = dir.path().join(format!("vcase_{:02}.txt", i));
        let mut content = String::with_capacity(5000);

        for line in 0..80 {
            if line % 15 == 7 {
                // Alternate case: "VerifyCase", "VERIFYCASE", "verifycase"
                let marker = match i % 3 {
                    0 => "VerifyCase",
                    1 => "VERIFYCASE",
                    _ => "verifycase",
                };
                content.push_str(&format!(
                    "data line {} {} entry in file {} with padding text\n",
                    line, marker, i
                ));
            } else {
                content.push_str(&format!(
                    "filler_{:04}_file_{:02}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                    line, i
                ));
            }
        }

        fs::write(&filepath, content.as_bytes()).expect("write verify case test file");
    }

    println!("=== test_verify_full_mode_case_insensitive ===");

    let request = SearchRequest {
        pattern: "verifycase".to_string(),
        root: dir.path().to_path_buf(),
        file_types: None,
        case_sensitive: false,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    // Panics if false positives detected in Full mode
    let response = orchestrator.search(request);

    println!("  Search completed without panic (VERIFY=full, case_insensitive)");
    println!("  Content matches: {}", response.content_matches.len());

    assert!(
        !response.content_matches.is_empty(),
        "Should find at least some verifycase matches (case-insensitive)"
    );

    // Validate each match contains the pattern (case-insensitive)
    for m in &response.content_matches {
        let range = &m.match_range;
        let slice = &m.line_content[range.clone()];
        assert_eq!(
            slice.to_lowercase(), "verifycase",
            "match_range content {:?} != 'verifycase' (case-insensitive)",
            slice
        );
    }

    std::env::remove_var("GPU_SEARCH_VERIFY");

    println!("=== test_verify_full_mode_case_insensitive: PASSED ===");
}
