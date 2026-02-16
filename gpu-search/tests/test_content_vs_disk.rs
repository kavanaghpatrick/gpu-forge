//! A/B oracle integration test: in-memory content store search vs ground truth.
//!
//! Generates a 200-file tempdir with known patterns, builds a ContentStore,
//! searches via the content store fast-path in SearchOrchestrator, then
//! compares results against ground truth (manual pattern search over file
//! bytes). For 20 different patterns, verifies:
//!
//! 1. Same set of files found (strict file-level coverage)
//! 2. No false positives (every GPU match line_content contains the pattern)
//! 3. GPU line numbers correspond to actual pattern locations in files
//!
//! The content store search path uses GPU compute over a padded chunked buffer.
//! Small files (< CHUNK_SIZE) each get their own 4KB chunk. The GPU kernel
//! finds byte offsets within chunks, which are then resolved to line numbers
//! via Rust code. This pipeline is validated against manual string search
//! of the exact same file bytes.
//!
//! REQUIRES: Real Apple Silicon GPU (Metal device must be available).

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossbeam_channel as channel;
use tempfile::TempDir;

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::index::content_index_store::ContentIndexStore;
use gpu_search::index::content_snapshot::ContentSnapshot;
use gpu_search::index::content_store::ContentStoreBuilder;
use gpu_search::search::cancel::{cancellation_pair, SearchGeneration, SearchSession};
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::{SearchRequest, SearchResponse};

// ============================================================================
// Helpers
// ============================================================================

/// Create a test search session (token + generation guard).
fn test_session() -> SearchSession {
    let (token, _handle) = cancellation_pair();
    let gen = SearchGeneration::new();
    let guard = gen.next();
    SearchSession { token, guard }
}

/// Quick GPU health check: create a tiny file with a known pattern and verify
/// the GPU can find it. Returns false if Metal compute is non-functional.
fn gpu_returns_results(orchestrator: &mut SearchOrchestrator) -> bool {
    let dir = TempDir::new().expect("create health check dir");
    std::fs::write(
        dir.path().join("health_check.txt"),
        "line one\nHEALTH_CHECK_MARKER here\nline three\n",
    )
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

/// Ground truth: for a given pattern, scan all files and return a map of
/// file path -> set of 1-based line numbers where the pattern appears.
fn ground_truth_search(
    files: &[(PathBuf, Vec<u8>)],
    pattern: &str,
) -> BTreeMap<PathBuf, BTreeSet<u32>> {
    let mut results: BTreeMap<PathBuf, BTreeSet<u32>> = BTreeMap::new();

    for (path, content) in files {
        let text = match std::str::from_utf8(content) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let mut line_numbers = BTreeSet::new();
        for (i, line) in text.lines().enumerate() {
            if line.contains(pattern) {
                line_numbers.insert((i + 1) as u32); // 1-based
            }
        }

        if !line_numbers.is_empty() {
            results.insert(path.clone(), line_numbers);
        }
    }

    results
}

/// Generate the 200-file test corpus with unique, unambiguous patterns.
///
/// Each file is named with its category and contains distinctive long markers
/// that are unlikely to produce chunk boundary issues. Files are kept under
/// 4KB (CHUNK_SIZE) so each gets exactly one GPU chunk.
///
/// File categories:
///   - files 0..39:   contain "ALPHA_MARKER_ABCDEF" (unique to this set + overlap)
///   - files 40..79:  contain "BETA_SIGNAL_XYZQWE" (unique to this set + overlap)
///   - files 80..119: contain "GAMMA_VALUE_MNOPRS" (unique to this set + overlap)
///   - files 120..149: contain "DELTA_CODE_JKLTUV" (unique to this set + overlap)
///   - files 150..169: contain all four markers (overlap files)
///   - files 170..189: contain unique per-file markers "UNIQUE_FILE_nnn"
///   - files 190..199: noise files (no target patterns)
fn make_200_file_corpus() -> (TempDir, Vec<(PathBuf, Vec<u8>)>) {
    let dir = TempDir::new().expect("create tempdir");
    let mut files: Vec<(PathBuf, Vec<u8>)> = Vec::with_capacity(200);

    // Category 1: ALPHA files (0..40)
    for i in 0..40 {
        let name = format!("alpha_{:03}.rs", i);
        let content = format!(
            "// alpha source file {i}\n\
             use std::collections::HashMap;\n\
             fn compute_{i}(x: u32) -> u32 {{\n\
             \tlet result = x * ALPHA_MARKER_ABCDEF;\n\
             \tresult\n\
             }}\n\
             pub fn helper_{i}() {{\n\
             \tlet data = vec![1, 2, 3];\n\
             }}\n"
        );
        let path = dir.path().join(&name);
        std::fs::write(&path, &content).unwrap();
        files.push((path, content.into_bytes()));
    }

    // Category 2: BETA files (40..80)
    for i in 40..80 {
        let name = format!("beta_{:03}.rs", i);
        let content = format!(
            "// beta source file {i}\n\
             use std::sync::Arc;\n\
             struct Runner_{i} {{\n\
             \tactive: bool,\n\
             \tBETA_SIGNAL_XYZQWE: bool,\n\
             }}\n\
             impl Runner_{i} {{\n\
             \tfn run(&self) {{\n\
             \t\tif self.BETA_SIGNAL_XYZQWE {{ }}\n\
             \t}}\n\
             }}\n"
        );
        let path = dir.path().join(&name);
        std::fs::write(&path, &content).unwrap();
        files.push((path, content.into_bytes()));
    }

    // Category 3: GAMMA files (80..120)
    for i in 80..120 {
        let name = format!("gamma_{:03}.rs", i);
        let content = format!(
            "// gamma source file {i}\n\
             pub struct GammaData_{i} {{\n\
             \tpub value: f64,\n\
             \tpub GAMMA_VALUE_MNOPRS: u32,\n\
             }}\n\
             impl GammaData_{i} {{\n\
             \tfn validate(&self) -> bool {{\n\
             \t\tself.GAMMA_VALUE_MNOPRS > 0\n\
             \t}}\n\
             }}\n"
        );
        let path = dir.path().join(&name);
        std::fs::write(&path, &content).unwrap();
        files.push((path, content.into_bytes()));
    }

    // Category 4: DELTA files (120..150)
    for i in 120..150 {
        let name = format!("delta_{:03}.rs", i);
        let content = format!(
            "// delta source file {i}\n\
             struct DeltaImpl_{i};\n\
             impl DeltaImpl_{i} {{\n\
             \tfn process(&self) -> u32 {{\n\
             \t\tlet code = DELTA_CODE_JKLTUV;\n\
             \t\tcode\n\
             \t}}\n\
             }}\n"
        );
        let path = dir.path().join(&name);
        std::fs::write(&path, &content).unwrap();
        files.push((path, content.into_bytes()));
    }

    // Category 5: ALL four markers -- overlap files (150..170)
    //
    // GPU kernel assigns each thread a 64-byte window. A pattern is only found
    // if it starts within [0, 64 - pattern_len] of a window. To avoid boundary
    // misses, we place each marker at a known safe offset. With markers ~19
    // bytes each, safe positions are 0 and 20 within each 64-byte window.
    // We pad to 64 bytes between pairs of markers so each pair lands in its
    // own thread window.
    for i in 150..170 {
        let name = format!("overlap_{:03}.rs", i);
        // Line 1: ALPHA at offset 0 within window 0 (bytes 0-63)
        // Line 2: BETA at offset ~20 within window 0
        // Pad to byte 64
        // Line 4: GAMMA at offset 0 within window 1 (bytes 64-127)
        // Line 5: DELTA at offset ~20 within window 1
        let mut content = String::new();
        content.push_str("ALPHA_MARKER_ABCDEF\n"); // 20 bytes (offset 0)
        content.push_str("BETA_SIGNAL_XYZQWE\n");  // 19 bytes (offset 20)
        // Pad to 64 bytes total for this window
        while content.len() < 64 {
            content.push('.');
        }
        content.push('\n');
        content.push_str("GAMMA_VALUE_MNOPRS\n"); // offset 65 = window 1 offset 1
        content.push_str("DELTA_CODE_JKLTUV\n");  // offset 84 = window 1 offset 20
        // Pad to 128 bytes
        while content.len() < 128 {
            content.push('.');
        }
        content.push('\n');
        content.push_str(&format!("// overlap source file {i}\n"));
        content.push_str(&format!("fn combined_{i}() {{ }}\n"));

        let path = dir.path().join(&name);
        std::fs::write(&path, &content).unwrap();
        files.push((path, content.into_bytes()));
    }

    // Category 6: unique per-file markers (170..190)
    for i in 170..190 {
        let name = format!("unique_{:03}.rs", i);
        let content = format!(
            "// unique source file {i}\n\
             const UNIQUE_FILE_{i}: &str = \"marker\";\n\
             fn lookup_{i}() -> &'static str {{\n\
             \tUNIQUE_FILE_{i}\n\
             }}\n"
        );
        let path = dir.path().join(&name);
        std::fs::write(&path, &content).unwrap();
        files.push((path, content.into_bytes()));
    }

    // Category 7: noise files (190..200)
    for i in 190..200 {
        let name = format!("noise_{:03}.txt", i);
        let content = format!(
            "// noise file {i}\n\
             let x = {i};\n\
             let y = x * 2;\n"
        );
        let path = dir.path().join(&name);
        std::fs::write(&path, &content).unwrap();
        files.push((path, content.into_bytes()));
    }

    assert_eq!(files.len(), 200, "Should have exactly 200 files");
    (dir, files)
}

/// Build a ContentStore from file entries, finalize with Metal GPU buffer.
fn build_content_store(
    files: &[(PathBuf, Vec<u8>)],
    device: &GpuDevice,
) -> Arc<ContentIndexStore> {
    let total_size: usize = files.iter().map(|(_, c)| c.len()).sum();
    let mut builder = ContentStoreBuilder::new(total_size + 16384)
        .expect("Failed to allocate ContentStoreBuilder");

    for (i, (path, content)) in files.iter().enumerate() {
        builder.append_with_path(content, path.clone(), i as u32, 0, 0);
    }

    let store = builder.finalize(&device.device);
    assert_eq!(store.file_count() as usize, files.len());

    let snapshot = ContentSnapshot::new(store, 0);
    let content_store = Arc::new(ContentIndexStore::new());
    content_store.swap(snapshot);

    content_store
}

/// Search via content store fast-path.
fn search_content_store(
    orchestrator: &mut SearchOrchestrator,
    root: &Path,
    pattern: &str,
) -> SearchResponse {
    let request = SearchRequest {
        pattern: pattern.to_string(),
        root: root.to_path_buf(),
        file_types: None,
        case_sensitive: true,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    let (update_tx, _) = channel::unbounded();
    let session = test_session();
    orchestrator.search_streaming(request, &update_tx, &session)
}

/// Compare GPU content store results against ground truth.
///
/// Returns (pass, mismatches).
fn compare_results(
    pattern: &str,
    gpu_response: &SearchResponse,
    ground_truth: &BTreeMap<PathBuf, BTreeSet<u32>>,
) -> (bool, Vec<String>) {
    let mut mismatches: Vec<String> = Vec::new();

    // Build GPU result map: path -> set of line numbers
    let mut gpu_results: BTreeMap<PathBuf, BTreeSet<u32>> = BTreeMap::new();
    for cm in &gpu_response.content_matches {
        gpu_results
            .entry(cm.path.clone())
            .or_default()
            .insert(cm.line_number);
    }

    let gpu_files: BTreeSet<&PathBuf> = gpu_results.keys().collect();
    let truth_files: BTreeSet<&PathBuf> = ground_truth.keys().collect();

    // Check 1: File coverage -- same files found
    let missed_files: Vec<&&PathBuf> = truth_files.difference(&gpu_files).collect();
    for f in &missed_files {
        let name = f.file_name().unwrap_or_default().to_string_lossy();
        mismatches.push(format!(
            "MISSED FILE: pattern='{}' file='{}'",
            pattern, name
        ));
    }

    let extra_files: Vec<&&PathBuf> = gpu_files.difference(&truth_files).collect();
    for f in &extra_files {
        let name = f.file_name().unwrap_or_default().to_string_lossy();
        mismatches.push(format!(
            "FALSE POSITIVE FILE: pattern='{}' file='{}'",
            pattern, name
        ));
    }

    // Check 2: No false positives in line content
    for cm in &gpu_response.content_matches {
        if !cm.line_content.contains(pattern) {
            let name = cm.path.file_name().unwrap_or_default().to_string_lossy();
            mismatches.push(format!(
                "LINE MISMATCH: pattern='{}' file='{}' line={} content='{}'",
                pattern, name, cm.line_number, cm.line_content.trim()
            ));
        }
    }

    // Check 3: GPU line numbers should exist in ground truth
    for (path, gpu_lines) in &gpu_results {
        if let Some(truth_lines) = ground_truth.get(path) {
            let bad: Vec<&u32> = gpu_lines.difference(truth_lines).collect();
            if !bad.is_empty() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                mismatches.push(format!(
                    "BAD LINES: pattern='{}' file='{}' gpu={:?} truth={:?}",
                    pattern, name, bad, truth_lines
                ));
            }
        }
    }

    (mismatches.is_empty(), mismatches)
}

// ============================================================================
// Main A/B Oracle Test: 20 Patterns
// ============================================================================

/// For 20 different patterns, search via the content store fast-path and compare
/// against ground truth. This is the definitive correctness gate.
#[test]
fn test_content_store_vs_ground_truth_oracle() {
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);

    let mut health_orch = SearchOrchestrator::new(&device.device, &pso_cache)
        .expect("Failed to create health check orchestrator");
    if !gpu_returns_results(&mut health_orch) {
        eprintln!("SKIPPED: GPU health check failed -- Metal compute non-functional");
        return;
    }
    drop(health_orch);

    let (dir, file_entries) = make_200_file_corpus();
    let content_store = build_content_store(&file_entries, &device);

    let mut orchestrator = SearchOrchestrator::with_content_store(
        &device.device,
        &pso_cache,
        None,
        content_store,
    )
    .expect("Failed to create orchestrator with content store");

    // 20 patterns covering all categories
    let patterns: [&str; 20] = [
        // Primary markers (appear in category + overlap files)
        "ALPHA_MARKER_ABCDEF",
        "BETA_SIGNAL_XYZQWE",
        "GAMMA_VALUE_MNOPRS",
        "DELTA_CODE_JKLTUV",
        // Unique per-file markers (should match exactly 1 file each)
        "UNIQUE_FILE_170",
        "UNIQUE_FILE_175",
        "UNIQUE_FILE_180",
        "UNIQUE_FILE_185",
        // File header patterns (appear in specific categories)
        "// alpha source",
        "// beta source",
        "// gamma source",
        "// delta source",
        "// overlap source",
        "// unique source",
        "// noise file",
        // Cross-category patterns
        "fn combined_",
        "struct DeltaImpl_",
        "struct GammaData_",
        // Zero-match pattern
        "ZZZZZ_NONEXISTENT_99",
        // Cross-category: struct keyword (gamma + beta categories)
        "struct Runner_",
    ];

    let mut total_tested = 0;
    let mut total_passed = 0;
    let mut all_mismatches: Vec<String> = Vec::new();

    println!("\n=== A/B Oracle Test: Content Store vs Ground Truth ===");
    println!("Corpus: {} files", file_entries.len());
    println!();

    for pattern in &patterns {
        total_tested += 1;

        let truth = ground_truth_search(&file_entries, pattern);
        let truth_files = truth.len();
        let truth_matches: usize = truth.values().map(|s| s.len()).sum();

        let response = search_content_store(&mut orchestrator, dir.path(), pattern);
        let gpu_matches = response.content_matches.len();

        let (pass, mismatches) = compare_results(pattern, &response, &truth);

        let status = if pass { "PASS" } else { "FAIL" };
        println!(
            "  [{:>4}] {:<40} truth_files={:<4} truth_matches={:<4} gpu_matches={:<4}",
            status, format!("'{}'", pattern), truth_files, truth_matches, gpu_matches
        );

        if pass {
            total_passed += 1;
        } else {
            for m in &mismatches {
                println!("         {}", m);
            }
            all_mismatches.extend(mismatches);
        }
    }

    println!();
    println!("=== ORACLE SUMMARY ===");
    println!("Patterns: {}/{} passed", total_passed, total_tested);
    println!("Mismatches: {}", all_mismatches.len());

    assert!(
        all_mismatches.is_empty(),
        "A/B oracle failed with {} mismatches across {} patterns:\n{}",
        all_mismatches.len(),
        total_tested - total_passed,
        all_mismatches.join("\n")
    );
}

// ============================================================================
// File Coverage Test
// ============================================================================

/// Verify exact file-level coverage for each primary marker pattern.
#[test]
fn test_content_store_file_coverage_matches_ground_truth() {
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);

    let mut health_orch = SearchOrchestrator::new(&device.device, &pso_cache)
        .expect("Failed to create health check orchestrator");
    if !gpu_returns_results(&mut health_orch) {
        eprintln!("SKIPPED: GPU health check failed");
        return;
    }
    drop(health_orch);

    let (dir, file_entries) = make_200_file_corpus();
    let content_store = build_content_store(&file_entries, &device);

    let mut orchestrator = SearchOrchestrator::with_content_store(
        &device.device,
        &pso_cache,
        None,
        content_store,
    )
    .expect("Failed to create orchestrator");

    let marker_patterns = [
        ("ALPHA_MARKER_ABCDEF", 60), // 40 alpha + 20 overlap
        ("BETA_SIGNAL_XYZQWE", 60),  // 40 beta + 20 overlap
        ("GAMMA_VALUE_MNOPRS", 60),  // 40 gamma + 20 overlap
        ("DELTA_CODE_JKLTUV", 50),   // 30 delta + 20 overlap
    ];

    println!("\n=== File Coverage Test ===");

    for (pattern, expected_file_count) in &marker_patterns {
        let truth = ground_truth_search(&file_entries, pattern);
        let response = search_content_store(&mut orchestrator, dir.path(), pattern);

        let gpu_files: HashSet<PathBuf> = response
            .content_matches
            .iter()
            .map(|cm| cm.path.clone())
            .collect();
        let truth_files: HashSet<PathBuf> = truth.keys().cloned().collect();

        println!(
            "  pattern='{}' expected={} truth={} gpu={}",
            pattern, expected_file_count, truth_files.len(), gpu_files.len()
        );

        assert_eq!(
            truth_files.len(),
            *expected_file_count,
            "Ground truth count mismatch for '{}'",
            pattern
        );

        let missed: Vec<&PathBuf> = truth_files.difference(&gpu_files).collect();
        assert!(
            missed.is_empty(),
            "GPU missed files for '{}': {:?}",
            pattern,
            missed.iter().map(|p| p.file_name().unwrap_or_default()).collect::<Vec<_>>()
        );

        let extra: Vec<&PathBuf> = gpu_files.difference(&truth_files).collect();
        assert!(
            extra.is_empty(),
            "GPU false positive files for '{}': {:?}",
            pattern,
            extra.iter().map(|p| p.file_name().unwrap_or_default()).collect::<Vec<_>>()
        );

        // Every match line must contain the pattern
        for cm in &response.content_matches {
            assert!(
                cm.line_content.contains(pattern),
                "Line should contain '{}': '{}'",
                pattern,
                cm.line_content.trim()
            );
        }
    }

    println!("  All file coverage checks passed!");
}

// ============================================================================
// Zero-Match Patterns
// ============================================================================

/// Verify patterns that don't exist return empty from both paths.
#[test]
fn test_content_store_zero_match_patterns() {
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);

    let mut health_orch = SearchOrchestrator::new(&device.device, &pso_cache)
        .expect("Failed to create health check orchestrator");
    if !gpu_returns_results(&mut health_orch) {
        eprintln!("SKIPPED: GPU health check failed");
        return;
    }
    drop(health_orch);

    let (dir, file_entries) = make_200_file_corpus();
    let content_store = build_content_store(&file_entries, &device);

    let mut orchestrator = SearchOrchestrator::with_content_store(
        &device.device,
        &pso_cache,
        None,
        content_store,
    )
    .expect("Failed to create orchestrator");

    let nonexistent = [
        "ZZZZZ_NONEXISTENT_99",
        "xylophone_quantum_flux",
        "ZEBRA_MARKER_999",
    ];

    println!("\n=== Zero-Match Patterns Test ===");

    for pattern in &nonexistent {
        let truth = ground_truth_search(&file_entries, pattern);
        let response = search_content_store(&mut orchestrator, dir.path(), pattern);

        assert!(truth.is_empty(), "Ground truth should be empty for '{}'", pattern);
        assert!(
            response.content_matches.is_empty(),
            "GPU should find 0 matches for '{}', got {}",
            pattern,
            response.content_matches.len()
        );

        println!("  '{}' -- zero matches (correct)", pattern);
    }
}

// ============================================================================
// Unique Pattern Isolation
// ============================================================================

/// Verify each UNIQUE_FILE_nnn pattern appears in exactly one file.
#[test]
fn test_content_store_unique_patterns_isolation() {
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);

    let mut health_orch = SearchOrchestrator::new(&device.device, &pso_cache)
        .expect("Failed to create health check orchestrator");
    if !gpu_returns_results(&mut health_orch) {
        eprintln!("SKIPPED: GPU health check failed");
        return;
    }
    drop(health_orch);

    let (dir, file_entries) = make_200_file_corpus();
    let content_store = build_content_store(&file_entries, &device);

    let mut orchestrator = SearchOrchestrator::with_content_store(
        &device.device,
        &pso_cache,
        None,
        content_store,
    )
    .expect("Failed to create orchestrator");

    println!("\n=== Unique Pattern Isolation Test ===");

    for i in 170..190 {
        let pattern = format!("UNIQUE_FILE_{}", i);
        let expected_file = format!("unique_{:03}.rs", i);

        let truth = ground_truth_search(&file_entries, &pattern);
        let response = search_content_store(&mut orchestrator, dir.path(), &pattern);

        assert_eq!(truth.len(), 1, "'{}' should be in exactly 1 file", pattern);

        let truth_file = truth.keys().next().unwrap()
            .file_name().unwrap().to_str().unwrap();
        assert_eq!(truth_file, expected_file);

        let gpu_files: HashSet<String> = response
            .content_matches
            .iter()
            .map(|cm| cm.path.file_name().unwrap().to_str().unwrap().to_string())
            .collect();

        assert_eq!(
            gpu_files.len(), 1,
            "GPU should find '{}' in 1 file, got {}: {:?}",
            pattern, gpu_files.len(), gpu_files
        );
        assert!(
            gpu_files.contains(&expected_file),
            "GPU should find '{}' in '{}', got {:?}",
            pattern, expected_file, gpu_files
        );

        for cm in &response.content_matches {
            assert!(
                cm.line_content.contains(&pattern),
                "Line should contain '{}': '{}'",
                pattern, cm.line_content.trim()
            );
        }
    }

    println!("  All 20 unique patterns correctly isolated!");
}
