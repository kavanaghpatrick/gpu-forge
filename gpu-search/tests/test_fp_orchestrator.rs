//! Orchestrator-level false positive integration tests.
//!
//! Tests the full pipeline: directory walk -> filter -> GPU dispatch -> resolve_match,
//! validating that every returned ContentMatch genuinely contains the search pattern.
//! This catches false positives that originate above the StreamingSearchEngine level
//! (e.g., stale buffers, byte_offset mapping, resolve_match amplification).
//!
//! REQUIRES: Real Apple Silicon GPU (Metal device must be available).

use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use crossbeam_channel as channel;

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::cancel::{cancellation_pair, CancellationHandle, SearchGeneration, SearchSession};
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::{ContentMatch, SearchRequest, SearchUpdate, StampedUpdate};

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

/// Create a test directory with 20 temp files:
/// - 5 files containing "kolbey" (kolbey_0..4.txt)
/// - 5 files containing "Patrick Kavanagh" (kavanagh_0..4.txt)
/// - 10 files containing generic filler (filler_0..9.txt)
///
/// Each file is ~4KB+ to ensure it exceeds the cold threshold and goes through
/// the GPU dispatch path rather than being handled by small-file CPU fallback.
///
/// Returns (TempDir, kolbey_paths, kavanagh_paths, filler_paths).
fn create_test_directory() -> (
    tempfile::TempDir,
    Vec<PathBuf>,
    Vec<PathBuf>,
    Vec<PathBuf>,
) {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let root = dir.path();

    let mut kolbey_paths = Vec::new();
    let mut kavanagh_paths = Vec::new();
    let mut filler_paths = Vec::new();

    // --- 5 files containing "kolbey" ---
    for i in 0..5 {
        let path = root.join(format!("kolbey_{}.txt", i));
        let mut content = String::with_capacity(5000);
        for line in 0..80 {
            if line % 16 == 5 {
                content.push_str(&format!(
                    "line {} here is kolbey mentioned in document {} for testing\n",
                    line, i
                ));
            } else {
                content.push_str(&format!(
                    "filler_line_{:04}_kolbey_file_{}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                    line, i
                ));
            }
        }
        fs::write(&path, &content).expect("write kolbey file");
        kolbey_paths.push(path);
    }

    // --- 5 files containing "Patrick Kavanagh" ---
    for i in 0..5 {
        let path = root.join(format!("kavanagh_{}.txt", i));
        let mut content = String::with_capacity(5000);
        for line in 0..80 {
            if line % 16 == 5 {
                content.push_str(&format!(
                    "line {} the poet Patrick Kavanagh wrote about Monaghan in piece {}\n",
                    line, i
                ));
            } else {
                content.push_str(&format!(
                    "filler_line_{:04}_kavanagh_file_{}_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
                    line, i
                ));
            }
        }
        fs::write(&path, &content).expect("write kavanagh file");
        kavanagh_paths.push(path);
    }

    // --- 10 files containing generic filler ---
    for i in 0..10 {
        let path = root.join(format!("filler_{}.txt", i));
        let mut content = String::with_capacity(5000);
        for line in 0..80 {
            content.push_str(&format!(
                "generic_filler_line_{:04}_document_{}_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n",
                line, i
            ));
        }
        fs::write(&path, &content).expect("write filler file");
        filler_paths.push(path);
    }

    (dir, kolbey_paths, kavanagh_paths, filler_paths)
}

/// Validate that every ContentMatch in the results genuinely contains the search
/// pattern. For each match:
/// 1. Read the file at cm.path
/// 2. Assert the file content (lowercased) contains the pattern (lowercased)
/// 3. Assert cm.line_content (lowercased) contains the pattern (lowercased)
///
/// Returns the number of false positives found (0 = all good).
fn validate_content_matches(matches: &[ContentMatch], pattern: &str) -> usize {
    let pattern_lower = pattern.to_lowercase();
    let mut false_positives = 0;

    for (i, cm) in matches.iter().enumerate() {
        // Check 1: line_content contains pattern
        let line_lower = cm.line_content.to_lowercase();
        if !line_lower.contains(&pattern_lower) {
            eprintln!(
                "FALSE POSITIVE [line_content] match #{}: path={} line={} content={:?}",
                i,
                cm.path.display(),
                cm.line_number,
                cm.line_content.trim()
            );
            false_positives += 1;
            continue;
        }

        // Check 2: actual file on disk contains pattern
        if cm.path.exists() {
            let file_content = fs::read_to_string(&cm.path).unwrap_or_default();
            let file_lower = file_content.to_lowercase();
            if !file_lower.contains(&pattern_lower) {
                eprintln!(
                    "FALSE POSITIVE [file_content] match #{}: path={} line={} -- file does NOT contain {:?}",
                    i,
                    cm.path.display(),
                    cm.line_number,
                    pattern
                );
                false_positives += 1;
            }
        } else {
            eprintln!(
                "WARNING: match #{} path does not exist: {}",
                i,
                cm.path.display()
            );
        }
    }

    false_positives
}

// ============================================================================
// Test: infrastructure smoke test
// ============================================================================

/// Verify that the test infrastructure works: orchestrator can search the test
/// directory and return results for known patterns.
#[test]
fn test_infrastructure_smoke() {
    let (_device, _pso, mut orchestrator) = create_orchestrator();
    let (dir, kolbey_paths, _kavanagh_paths, _filler_paths) = create_test_directory();

    // Verify files were created
    assert_eq!(kolbey_paths.len(), 5);
    for p in &kolbey_paths {
        assert!(p.exists(), "kolbey file should exist: {}", p.display());
        let content = fs::read_to_string(p).unwrap();
        assert!(
            content.contains("kolbey"),
            "kolbey file should contain 'kolbey': {}",
            p.display()
        );
        assert!(
            content.len() >= 4000,
            "kolbey file should be >= 4KB: {} has {} bytes",
            p.display(),
            content.len()
        );
    }

    // Search for "kolbey" -- should find matches
    let request = SearchRequest {
        pattern: "kolbey".to_string(),
        root: dir.path().to_path_buf(),
        file_types: None,
        case_sensitive: false,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    let response = orchestrator.search(request);

    println!("Infrastructure smoke test:");
    println!("  Files searched: {}", response.total_files_searched);
    println!("  Content matches: {}", response.content_matches.len());
    println!("  Elapsed: {:.2}ms", response.elapsed.as_secs_f64() * 1000.0);

    for cm in &response.content_matches {
        println!(
            "  {}:{}: {}",
            cm.path.file_name().unwrap_or_default().to_string_lossy(),
            cm.line_number,
            cm.line_content.trim()
        );
    }

    // Should have searched all 20 files
    assert!(
        response.total_files_searched >= 15,
        "Expected to search >= 15 files, got {}",
        response.total_files_searched
    );

    // Should find at least some matches for "kolbey"
    assert!(
        !response.content_matches.is_empty(),
        "Expected at least 1 content match for 'kolbey'"
    );

    // Validate no false positives in results
    let fp_count = validate_content_matches(&response.content_matches, "kolbey");
    println!("  False positives: {}", fp_count);
}

// ============================================================================
// Test: orchestrator-level false positive detection
// ============================================================================

/// Search for "kolbey" and "patrick" using orchestrator.search(), then validate
/// that EVERY returned ContentMatch genuinely contains the searched pattern in
/// both the file on disk and the line_content field. Asserts zero false positives.
#[test]
fn test_orchestrator_no_false_positives() {
    let (_device, _pso, mut orchestrator) = create_orchestrator();
    let (dir, _kolbey_paths, _kavanagh_paths, _filler_paths) = create_test_directory();

    // --- Test pattern: "kolbey" ---
    {
        let request = SearchRequest {
            pattern: "kolbey".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: false,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        println!("[kolbey] Content matches: {}", response.content_matches.len());
        for cm in &response.content_matches {
            println!(
                "  [kolbey] {}:{}: {}",
                cm.path.file_name().unwrap_or_default().to_string_lossy(),
                cm.line_number,
                cm.line_content.trim()
            );
        }

        // Validate every match: file on disk contains "kolbey"
        let mut fp_count = 0;
        for (i, cm) in response.content_matches.iter().enumerate() {
            // Check line_content contains pattern
            if !cm.line_content.to_lowercase().contains("kolbey") {
                eprintln!(
                    "FP [kolbey/line] #{}: {}:{} content={:?}",
                    i,
                    cm.path.display(),
                    cm.line_number,
                    cm.line_content.trim()
                );
                fp_count += 1;
                continue;
            }

            // Check file on disk contains pattern
            let file_content = fs::read_to_string(&cm.path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {}", cm.path.display(), e));
            if !file_content.to_lowercase().contains("kolbey") {
                eprintln!(
                    "FP [kolbey/file] #{}: {} -- file does NOT contain 'kolbey'",
                    i,
                    cm.path.display()
                );
                fp_count += 1;
            }
        }

        // Also validate via the shared helper
        let fp_helper = validate_content_matches(&response.content_matches, "kolbey");

        println!("[kolbey] False positives (manual): {}", fp_count);
        println!("[kolbey] False positives (helper): {}", fp_helper);
        assert_eq!(fp_count, 0, "Expected zero false positives for 'kolbey' (manual check)");
        assert_eq!(fp_helper, 0, "Expected zero false positives for 'kolbey' (helper check)");

        // Should have found at least some matches
        assert!(
            !response.content_matches.is_empty(),
            "Expected at least 1 content match for 'kolbey'"
        );
    }

    // --- Test pattern: "patrick" ---
    {
        let request = SearchRequest {
            pattern: "patrick".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: false,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        println!("[patrick] Content matches: {}", response.content_matches.len());
        for cm in &response.content_matches {
            println!(
                "  [patrick] {}:{}: {}",
                cm.path.file_name().unwrap_or_default().to_string_lossy(),
                cm.line_number,
                cm.line_content.trim()
            );
        }

        // Validate every match: file on disk contains "patrick"
        let mut fp_count = 0;
        for (i, cm) in response.content_matches.iter().enumerate() {
            // Check line_content contains pattern
            if !cm.line_content.to_lowercase().contains("patrick") {
                eprintln!(
                    "FP [patrick/line] #{}: {}:{} content={:?}",
                    i,
                    cm.path.display(),
                    cm.line_number,
                    cm.line_content.trim()
                );
                fp_count += 1;
                continue;
            }

            // Check file on disk contains pattern
            let file_content = fs::read_to_string(&cm.path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {}", cm.path.display(), e));
            if !file_content.to_lowercase().contains("patrick") {
                eprintln!(
                    "FP [patrick/file] #{}: {} -- file does NOT contain 'patrick'",
                    i,
                    cm.path.display()
                );
                fp_count += 1;
            }
        }

        // Also validate via the shared helper
        let fp_helper = validate_content_matches(&response.content_matches, "patrick");

        println!("[patrick] False positives (manual): {}", fp_count);
        println!("[patrick] False positives (helper): {}", fp_helper);
        assert_eq!(fp_count, 0, "Expected zero false positives for 'patrick' (manual check)");
        assert_eq!(fp_helper, 0, "Expected zero false positives for 'patrick' (helper check)");

        // Should have found at least some matches (Patrick Kavanagh files)
        assert!(
            !response.content_matches.is_empty(),
            "Expected at least 1 content match for 'patrick'"
        );
    }

    println!("test_orchestrator_no_false_positives: ALL PASSED");
}

// ============================================================================
// Test: match_range integrity validation
// ============================================================================

/// For every ContentMatch from an orchestrator search, validate that:
/// 1. match_range.start < match_range.end
/// 2. match_range.end <= line_content.len()
/// 3. line_content[match_range] case-insensitively equals the search pattern
///
/// Tests with both "kolbey" and "patrick" patterns.
#[test]
fn test_match_range_integrity() {
    let (_device, _pso, mut orchestrator) = create_orchestrator();
    let (dir, _kolbey_paths, _kavanagh_paths, _filler_paths) = create_test_directory();

    for pattern in &["kolbey", "patrick"] {
        let request = SearchRequest {
            pattern: pattern.to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: false,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        println!(
            "[{}] Validating match_range for {} content matches",
            pattern,
            response.content_matches.len()
        );

        assert!(
            !response.content_matches.is_empty(),
            "Expected at least 1 content match for '{}'",
            pattern
        );

        for (i, cm) in response.content_matches.iter().enumerate() {
            // (1) match_range.start < match_range.end
            assert!(
                cm.match_range.start < cm.match_range.end,
                "[{}] match #{}: match_range.start ({}) must be < match_range.end ({}) -- path={} line={}",
                pattern,
                i,
                cm.match_range.start,
                cm.match_range.end,
                cm.path.display(),
                cm.line_number
            );

            // (2) match_range.end <= line_content.len()
            assert!(
                cm.match_range.end <= cm.line_content.len(),
                "[{}] match #{}: match_range.end ({}) must be <= line_content.len() ({}) -- path={} line={} content={:?}",
                pattern,
                i,
                cm.match_range.end,
                cm.line_content.len(),
                cm.path.display(),
                cm.line_number,
                cm.line_content.trim()
            );

            // (3) extracted text at match_range must case-insensitively equal the pattern
            let extracted = &cm.line_content[cm.match_range.clone()];
            assert!(
                extracted.to_lowercase() == pattern.to_lowercase(),
                "[{}] match #{}: extracted text {:?} != pattern {:?} -- path={} line={} range={:?} content={:?}",
                pattern,
                i,
                extracted,
                pattern,
                cm.path.display(),
                cm.line_number,
                cm.match_range,
                cm.line_content.trim()
            );

            println!(
                "  [{}] match #{}: OK range={:?} extracted={:?}",
                pattern, i, cm.match_range, extracted
            );
        }

        println!(
            "[{}] All {} match_ranges validated successfully",
            pattern,
            response.content_matches.len()
        );
    }

    println!("test_match_range_integrity: ALL PASSED");
}

// ============================================================================
// Test: rapid query change simulation (orchestrator level)
// ============================================================================

/// Command sent from the simulated UI thread to the background orchestrator thread.
/// Mirrors `OrchestratorCommand` from app.rs (which is private).
enum TestOrchestratorCommand {
    Search(SearchRequest, SearchSession),
    Shutdown,
}

/// Simulates the orchestrator_thread() pattern from app.rs:
///
/// 1. Dispatches search for "patrick" (gen=1)
/// 2. After 10ms, cancels and dispatches "kolbey" (gen=2)
/// 3. After 10ms, cancels and dispatches "patrick" (gen=3)
/// 4. Collects all StampedUpdate messages
/// 5. Filters to final generation only
/// 6. Validates all surviving ContentMatches are genuine (file contains pattern)
///
/// This exercises the cancel/restart race that triggers false positives when
/// stale GPU results leak through generation guards.
#[test]
fn test_rapid_query_change_orchestrator() {
    // Set GPU_SEARCH_VERIFY=full for strict CPU verification
    std::env::set_var("GPU_SEARCH_VERIFY", "full");

    let (dir, _kolbey_paths, _kavanagh_paths, _filler_paths) = create_test_directory();
    let search_root = dir.path().to_path_buf();

    // Create channels mimicking app.rs orchestrator_thread pattern:
    //   cmd_tx/cmd_rx: UI -> background thread (search commands)
    //   update_tx/update_rx: background thread -> UI (stamped updates)
    let (cmd_tx, cmd_rx) = channel::unbounded::<TestOrchestratorCommand>();
    let (update_tx, update_rx) = channel::unbounded::<StampedUpdate>();

    // Shared generation counter (like GpuSearchApp.search_generation)
    let search_generation = SearchGeneration::new();

    // Spawn background orchestrator thread (mirrors app.rs orchestrator_thread)
    let bg_update_tx = update_tx.clone();
    let bg_thread = thread::spawn(move || {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create SearchOrchestrator");

        loop {
            match cmd_rx.recv() {
                Ok(TestOrchestratorCommand::Search(request, session)) => {
                    // Drain queued commands -- only process the latest (like app.rs)
                    let mut latest_request = request;
                    let mut latest_session = session;
                    while let Ok(cmd) = cmd_rx.try_recv() {
                        match cmd {
                            TestOrchestratorCommand::Search(r, s) => {
                                latest_request = r;
                                latest_session = s;
                            }
                            TestOrchestratorCommand::Shutdown => return,
                        }
                    }

                    // Skip if already cancelled/superseded before we start
                    if latest_session.should_stop() {
                        continue;
                    }

                    eprintln!(
                        "[bg] starting search for '{}' gen={}",
                        latest_request.pattern,
                        latest_session.guard.generation_id()
                    );

                    orchestrator.search_streaming(
                        latest_request,
                        &bg_update_tx,
                        &latest_session,
                    );

                    eprintln!("[bg] search complete");
                }
                Ok(TestOrchestratorCommand::Shutdown) | Err(_) => {
                    eprintln!("[bg] shutting down");
                    return;
                }
            }
        }
    });

    // Helper closure: dispatch a search (mirrors app.rs dispatch_search)
    let mut current_cancel_handle: Option<CancellationHandle> = None;

    let dispatch = |pattern: &str,
                    generation: &SearchGeneration,
                    cancel_handle: &mut Option<CancellationHandle>,
                    root: &PathBuf,
                    tx: &channel::Sender<TestOrchestratorCommand>,
                    rx: &channel::Receiver<StampedUpdate>| {
        // Cancel previous search
        if let Some(handle) = cancel_handle.take() {
            handle.cancel();
        }

        let request = SearchRequest {
            pattern: pattern.to_string(),
            root: root.clone(),
            file_types: None,
            case_sensitive: false,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let (token, handle) = cancellation_pair();
        let guard = generation.next();
        let gen_id = guard.generation_id();
        let session = SearchSession { token, guard };
        *cancel_handle = Some(handle);

        // Drain pending updates from previous search (like app.rs:430)
        while rx.try_recv().is_ok() {}

        tx.send(TestOrchestratorCommand::Search(request, session))
            .expect("send search command");

        eprintln!("[ui] dispatched '{}' gen={}", pattern, gen_id);
        gen_id
    };

    // === Rapid query change sequence ===

    // Step 1: Search for "patrick" (gen=1)
    let _gen1 = dispatch(
        "patrick",
        &search_generation,
        &mut current_cancel_handle,
        &search_root,
        &cmd_tx,
        &update_rx,
    );

    // Step 2: After 10ms, cancel and search for "kolbey" (gen=2)
    thread::sleep(Duration::from_millis(10));
    let _gen2 = dispatch(
        "kolbey",
        &search_generation,
        &mut current_cancel_handle,
        &search_root,
        &cmd_tx,
        &update_rx,
    );

    // Step 3: After 10ms, cancel and search for "patrick" (gen=3)
    thread::sleep(Duration::from_millis(10));
    let final_gen = dispatch(
        "patrick",
        &search_generation,
        &mut current_cancel_handle,
        &search_root,
        &cmd_tx,
        &update_rx,
    );

    let final_pattern = "patrick";

    // Wait for the final search to complete.
    // Collect all updates, looking for a Complete message with the final generation.
    let mut all_updates: Vec<StampedUpdate> = Vec::new();
    let timeout = Duration::from_secs(30);
    let start = std::time::Instant::now();
    let mut got_complete = false;

    while start.elapsed() < timeout {
        match update_rx.recv_timeout(Duration::from_millis(500)) {
            Ok(stamped) => {
                let is_complete_for_final = stamped.generation == final_gen
                    && matches!(stamped.update, SearchUpdate::Complete(_));
                all_updates.push(stamped);
                if is_complete_for_final {
                    got_complete = true;
                    // Drain any remaining buffered updates
                    while let Ok(extra) = update_rx.try_recv() {
                        all_updates.push(extra);
                    }
                    break;
                }
            }
            Err(channel::RecvTimeoutError::Timeout) => {
                // Keep waiting
            }
            Err(channel::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }

    // Shutdown the background thread
    cmd_tx
        .send(TestOrchestratorCommand::Shutdown)
        .expect("send shutdown");
    bg_thread.join().expect("bg thread panicked");

    assert!(
        got_complete,
        "Timed out waiting for Complete message for final generation {}",
        final_gen
    );

    // === Analyze results ===

    println!("Total StampedUpdate messages received: {}", all_updates.len());

    // Count updates per generation
    let mut gen_counts = std::collections::HashMap::new();
    for u in &all_updates {
        *gen_counts.entry(u.generation).or_insert(0usize) += 1;
    }
    for (gen, count) in &gen_counts {
        println!("  Generation {}: {} updates", gen, count);
    }

    // Filter to final generation only (mirrors poll_updates generation guard)
    let final_updates: Vec<&StampedUpdate> = all_updates
        .iter()
        .filter(|u| u.generation == final_gen)
        .collect();

    println!(
        "Final generation {} updates: {}",
        final_gen,
        final_updates.len()
    );

    // Extract all ContentMatches from final generation
    let mut final_content_matches: Vec<ContentMatch> = Vec::new();
    for stamped in &final_updates {
        match &stamped.update {
            SearchUpdate::ContentMatches(cms) => {
                final_content_matches.extend(cms.iter().cloned());
            }
            SearchUpdate::Complete(response) => {
                // The Complete message also has the full content_matches
                final_content_matches.extend(response.content_matches.iter().cloned());
            }
            _ => {}
        }
    }

    // Deduplicate: the Complete response includes all matches already sent
    // via ContentMatches updates. Use the Complete response's matches as
    // the authoritative set if available, otherwise use accumulated.
    let authoritative_matches: Vec<ContentMatch> = final_updates
        .iter()
        .filter_map(|u| {
            if let SearchUpdate::Complete(ref resp) = u.update {
                Some(resp.content_matches.clone())
            } else {
                None
            }
        })
        .next()
        .unwrap_or(final_content_matches);

    println!(
        "Final generation content matches: {}",
        authoritative_matches.len()
    );

    // === Validate: zero false positives ===
    let fp_count = validate_content_matches(&authoritative_matches, final_pattern);
    println!(
        "False positives in final generation for '{}': {}",
        final_pattern, fp_count
    );

    assert_eq!(
        fp_count, 0,
        "Rapid query change produced {} false positives in final generation for '{}'",
        fp_count, final_pattern
    );

    // Additional check: every match should be from a kavanagh file (contains "patrick")
    for (i, cm) in authoritative_matches.iter().enumerate() {
        let file_content = fs::read_to_string(&cm.path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", cm.path.display(), e));
        assert!(
            file_content.to_lowercase().contains(final_pattern),
            "Match #{} from {} does NOT contain '{}' in file content",
            i,
            cm.path.display(),
            final_pattern
        );
        assert!(
            cm.line_content.to_lowercase().contains(final_pattern),
            "Match #{} from {} line_content does NOT contain '{}'",
            i,
            cm.path.display(),
            final_pattern
        );
    }

    // Should have found at least some matches for "patrick"
    assert!(
        !authoritative_matches.is_empty(),
        "Expected at least 1 content match for '{}' in final generation",
        final_pattern
    );

    println!(
        "test_rapid_query_change_orchestrator: ALL PASSED ({} matches, 0 false positives)",
        authoritative_matches.len()
    );
}

// ============================================================================
// Test: character-by-character typing simulation
// ============================================================================

/// Simulates a user typing "kolbey" character by character with 5ms between
/// keystrokes. Each keystroke dispatches a new search, cancelling the previous.
///
/// Sequence: "k" -> "ko" -> "kol" -> "kolb" -> "kolbe" -> "kolbey"
///
/// After the final "kolbey" search completes, validates:
/// 1. Every ContentMatch genuinely contains "kolbey" in both line_content and file
/// 2. Zero matches come from "Patrick Kavanagh" files
/// 3. Zero false positives
///
/// This reproduces the exact user behavior that triggers false positives when
/// stale GPU results from earlier prefix searches leak into later results.
#[test]
fn test_typing_simulation() {
    // Set GPU_SEARCH_VERIFY=full for strict CPU verification
    std::env::set_var("GPU_SEARCH_VERIFY", "full");

    let (dir, _kolbey_paths, kavanagh_paths, _filler_paths) = create_test_directory();
    let search_root = dir.path().to_path_buf();

    // Collect kavanagh file names for later assertion
    let kavanagh_file_names: Vec<String> = kavanagh_paths
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();

    // Create channels mimicking app.rs orchestrator_thread pattern
    let (cmd_tx, cmd_rx) = channel::unbounded::<TestOrchestratorCommand>();
    let (update_tx, update_rx) = channel::unbounded::<StampedUpdate>();

    // Shared generation counter
    let search_generation = SearchGeneration::new();

    // Spawn background orchestrator thread
    let bg_update_tx = update_tx.clone();
    let bg_thread = thread::spawn(move || {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create SearchOrchestrator");

        loop {
            match cmd_rx.recv() {
                Ok(TestOrchestratorCommand::Search(request, session)) => {
                    // Drain queued commands -- only process the latest
                    let mut latest_request = request;
                    let mut latest_session = session;
                    while let Ok(cmd) = cmd_rx.try_recv() {
                        match cmd {
                            TestOrchestratorCommand::Search(r, s) => {
                                latest_request = r;
                                latest_session = s;
                            }
                            TestOrchestratorCommand::Shutdown => return,
                        }
                    }

                    // Skip if already cancelled/superseded
                    if latest_session.should_stop() {
                        continue;
                    }

                    eprintln!(
                        "[bg] starting search for '{}' gen={}",
                        latest_request.pattern,
                        latest_session.guard.generation_id()
                    );

                    orchestrator.search_streaming(
                        latest_request,
                        &bg_update_tx,
                        &latest_session,
                    );

                    eprintln!("[bg] search complete");
                }
                Ok(TestOrchestratorCommand::Shutdown) | Err(_) => {
                    eprintln!("[bg] shutting down");
                    return;
                }
            }
        }
    });

    // Helper: dispatch a search, cancel previous, return generation id
    let mut current_cancel_handle: Option<CancellationHandle> = None;

    let dispatch = |pattern: &str,
                    generation: &SearchGeneration,
                    cancel_handle: &mut Option<CancellationHandle>,
                    root: &PathBuf,
                    tx: &channel::Sender<TestOrchestratorCommand>,
                    rx: &channel::Receiver<StampedUpdate>| -> u64 {
        // Cancel previous search
        if let Some(handle) = cancel_handle.take() {
            handle.cancel();
        }

        let request = SearchRequest {
            pattern: pattern.to_string(),
            root: root.clone(),
            file_types: None,
            case_sensitive: false,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let (token, handle) = cancellation_pair();
        let guard = generation.next();
        let gen_id = guard.generation_id();
        let session = SearchSession { token, guard };
        *cancel_handle = Some(handle);

        // Drain pending updates from previous search
        while rx.try_recv().is_ok() {}

        tx.send(TestOrchestratorCommand::Search(request, session))
            .expect("send search command");

        eprintln!("[ui] dispatched '{}' gen={}", pattern, gen_id);
        gen_id
    };

    // === Character-by-character typing sequence ===
    let prefixes = ["k", "ko", "kol", "kolb", "kolbe", "kolbey"];
    let mut final_gen = 0u64;

    for (i, prefix) in prefixes.iter().enumerate() {
        let gen = dispatch(
            prefix,
            &search_generation,
            &mut current_cancel_handle,
            &search_root,
            &cmd_tx,
            &update_rx,
        );

        // The last prefix is the final search we care about
        if i == prefixes.len() - 1 {
            final_gen = gen;
        } else {
            // 5ms gap between keystrokes
            thread::sleep(Duration::from_millis(5));
        }
    }

    let final_pattern = "kolbey";

    // Wait for the final search to complete
    let mut all_updates: Vec<StampedUpdate> = Vec::new();
    let timeout = Duration::from_secs(30);
    let start = std::time::Instant::now();
    let mut got_complete = false;

    while start.elapsed() < timeout {
        match update_rx.recv_timeout(Duration::from_millis(500)) {
            Ok(stamped) => {
                let is_complete_for_final = stamped.generation == final_gen
                    && matches!(stamped.update, SearchUpdate::Complete(_));
                all_updates.push(stamped);
                if is_complete_for_final {
                    got_complete = true;
                    // Drain remaining buffered updates
                    while let Ok(extra) = update_rx.try_recv() {
                        all_updates.push(extra);
                    }
                    break;
                }
            }
            Err(channel::RecvTimeoutError::Timeout) => {
                // Keep waiting
            }
            Err(channel::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }

    // Shutdown the background thread
    cmd_tx
        .send(TestOrchestratorCommand::Shutdown)
        .expect("send shutdown");
    bg_thread.join().expect("bg thread panicked");

    assert!(
        got_complete,
        "Timed out waiting for Complete message for final generation {} (pattern='{}')",
        final_gen, final_pattern
    );

    // === Analyze results ===

    println!("Total StampedUpdate messages received: {}", all_updates.len());

    // Count updates per generation
    let mut gen_counts = std::collections::HashMap::new();
    for u in &all_updates {
        *gen_counts.entry(u.generation).or_insert(0usize) += 1;
    }
    for (gen, count) in &gen_counts {
        println!("  Generation {}: {} updates", gen, count);
    }

    // Filter to final generation only
    let final_updates: Vec<&StampedUpdate> = all_updates
        .iter()
        .filter(|u| u.generation == final_gen)
        .collect();

    println!(
        "Final generation {} updates: {}",
        final_gen,
        final_updates.len()
    );

    // Extract authoritative ContentMatches from final generation
    let authoritative_matches: Vec<ContentMatch> = final_updates
        .iter()
        .filter_map(|u| {
            if let SearchUpdate::Complete(ref resp) = u.update {
                Some(resp.content_matches.clone())
            } else {
                None
            }
        })
        .next()
        .unwrap_or_else(|| {
            // Fallback: accumulate from ContentMatches updates
            let mut cms = Vec::new();
            for stamped in &final_updates {
                if let SearchUpdate::ContentMatches(ref matches) = stamped.update {
                    cms.extend(matches.iter().cloned());
                }
            }
            cms
        });

    println!(
        "Final generation '{}' content matches: {}",
        final_pattern,
        authoritative_matches.len()
    );

    // === Validate: zero false positives ===
    let fp_count = validate_content_matches(&authoritative_matches, final_pattern);
    println!(
        "False positives in final generation for '{}': {}",
        final_pattern, fp_count
    );

    assert_eq!(
        fp_count, 0,
        "Typing simulation produced {} false positives for '{}'",
        fp_count, final_pattern
    );

    // === Validate: zero matches from "Patrick Kavanagh" files ===
    let kavanagh_match_count = authoritative_matches
        .iter()
        .filter(|cm| {
            let fname = cm
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            kavanagh_file_names.contains(&fname)
        })
        .count();

    println!(
        "Matches from 'Patrick Kavanagh' files: {}",
        kavanagh_match_count
    );

    assert_eq!(
        kavanagh_match_count, 0,
        "Expected 0 matches from kavanagh files, got {} -- these are false positives from stale prefix searches",
        kavanagh_match_count
    );

    // === Validate: every matched file actually contains "kolbey" ===
    for (i, cm) in authoritative_matches.iter().enumerate() {
        let file_content = fs::read_to_string(&cm.path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", cm.path.display(), e));
        assert!(
            file_content.to_lowercase().contains(final_pattern),
            "Match #{} from {} does NOT contain '{}' in file content",
            i,
            cm.path.display(),
            final_pattern
        );
        assert!(
            cm.line_content.to_lowercase().contains(final_pattern),
            "Match #{} from {} line_content does NOT contain '{}'",
            i,
            cm.path.display(),
            final_pattern
        );
    }

    // Should have found at least some matches for "kolbey"
    assert!(
        !authoritative_matches.is_empty(),
        "Expected at least 1 content match for '{}' in final generation",
        final_pattern
    );

    println!(
        "test_typing_simulation: ALL PASSED ({} matches, 0 false positives, 0 kavanagh matches)",
        authoritative_matches.len()
    );
}

// ============================================================================
// Test: kolbey-in-kavanagh regression (exact reported scenario)
// ============================================================================

/// Regression test for the exact reported false-positive bug:
///
///   Searching "kolbey" returns 9 matches from "Patrick Kavanagh" files.
///
/// This test reproduces the scenario with:
/// - Files named `kavanagh_bio.txt`, `kavanagh_poems.txt`, etc. containing
///   "Patrick Kavanagh" text (NEVER "kolbey")
/// - Files named `kolbey_notes.txt`, `kolbey_project.txt`, etc. containing
///   "kolbey" text (NEVER "Patrick Kavanagh")
/// - Many filler files to increase GPU batch pressure
///
/// Validates:
/// 1. ZERO matches from any kavanagh file
/// 2. ALL matches come exclusively from kolbey files
/// 3. Every match_range is valid and extracts "kolbey"
/// 4. Every matched file on disk actually contains "kolbey"
#[test]
fn test_kolbey_kavanagh_regression() {
    // Enable full CPU verification
    std::env::set_var("GPU_SEARCH_VERIFY", "full");

    let (_device, _pso, mut orchestrator) = create_orchestrator();

    // --- Build the test directory ---
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let root = dir.path();

    // Kavanagh files: realistic names, contain "Patrick Kavanagh" text, NEVER "kolbey"
    let kavanagh_file_names = [
        "kavanagh_bio.txt",
        "kavanagh_poems.txt",
        "kavanagh_monaghan.txt",
        "kavanagh_letters.txt",
        "kavanagh_tarry_flynn.txt",
        "patrick_kavanagh_collected.txt",
        "kavanagh_great_hunger.txt",
        "kavanagh_bibliography.txt",
        "kavanagh_canal_bank.txt",
    ];

    let mut kavanagh_paths = Vec::new();
    for (i, name) in kavanagh_file_names.iter().enumerate() {
        let path = root.join(name);
        let mut content = String::with_capacity(6000);
        // Each file: ~100 lines, every 10th line mentions Patrick Kavanagh
        for line in 0..100 {
            if line % 10 == 3 {
                content.push_str(&format!(
                    "Patrick Kavanagh was born in Inniskeen, County Monaghan in 1904 (ref {})\n",
                    i
                ));
            } else if line % 10 == 7 {
                content.push_str(&format!(
                    "The poet Kavanagh wrote about rural Irish life in document section {}\n",
                    i
                ));
            } else {
                content.push_str(&format!(
                    "This is line {} of the biography document number {} with various literary notes and references\n",
                    line, i
                ));
            }
        }
        // Verify no accidental "kolbey" in content
        assert!(
            !content.to_lowercase().contains("kolbey"),
            "BUG IN TEST: kavanagh file {} contains 'kolbey'",
            name
        );
        fs::write(&path, &content).expect("write kavanagh file");
        kavanagh_paths.push(path);
    }

    // Kolbey files: contain "kolbey" text, NEVER "Patrick Kavanagh"
    let kolbey_file_names = [
        "kolbey_notes.txt",
        "kolbey_project.txt",
        "kolbey_research.txt",
        "kolbey_data.txt",
        "kolbey_analysis.txt",
        "kolbey_report.txt",
    ];

    let mut kolbey_paths = Vec::new();
    for (i, name) in kolbey_file_names.iter().enumerate() {
        let path = root.join(name);
        let mut content = String::with_capacity(6000);
        for line in 0..100 {
            if line % 8 == 2 {
                content.push_str(&format!(
                    "The kolbey algorithm processes batch {} with optimized throughput\n",
                    i
                ));
            } else if line % 8 == 6 {
                content.push_str(&format!(
                    "Results from kolbey experiment run {} show improved performance metrics\n",
                    i
                ));
            } else {
                content.push_str(&format!(
                    "Generic experiment line {} in dataset {} with numerical data points and measurements\n",
                    line, i
                ));
            }
        }
        // Verify no accidental "kavanagh" or "patrick" in content
        assert!(
            !content.to_lowercase().contains("kavanagh"),
            "BUG IN TEST: kolbey file {} contains 'kavanagh'",
            name
        );
        assert!(
            !content.to_lowercase().contains("patrick"),
            "BUG IN TEST: kolbey file {} contains 'patrick'",
            name
        );
        fs::write(&path, &content).expect("write kolbey file");
        kolbey_paths.push(path);
    }

    // Filler files: 30 files with generic content (no "kolbey", no "kavanagh")
    let mut filler_paths = Vec::new();
    for i in 0..30 {
        let path = root.join(format!("document_{:03}.txt", i));
        let mut content = String::with_capacity(6000);
        for line in 0..100 {
            content.push_str(&format!(
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, document {} line {} sed do eiusmod\n",
                i, line
            ));
        }
        assert!(
            !content.to_lowercase().contains("kolbey"),
            "BUG IN TEST: filler file {} contains 'kolbey'",
            i
        );
        fs::write(&path, &content).expect("write filler file");
        filler_paths.push(path);
    }

    println!("=== kolbey-kavanagh regression test ===");
    println!(
        "Test directory: {} ({} kavanagh + {} kolbey + {} filler = {} total files)",
        root.display(),
        kavanagh_paths.len(),
        kolbey_paths.len(),
        filler_paths.len(),
        kavanagh_paths.len() + kolbey_paths.len() + filler_paths.len()
    );

    // --- Search for "kolbey" ---
    let request = SearchRequest {
        pattern: "kolbey".to_string(),
        root: root.to_path_buf(),
        file_types: None,
        case_sensitive: false,
        respect_gitignore: false,
        include_binary: false,
        max_results: 10_000,
    };

    let response = orchestrator.search(request);

    println!(
        "Search 'kolbey': {} files searched, {} content matches, {:.2}ms",
        response.total_files_searched,
        response.content_matches.len(),
        response.elapsed.as_secs_f64() * 1000.0
    );

    // --- Assertion 1: ZERO matches from kavanagh files ---
    let kavanagh_names_set: std::collections::HashSet<String> = kavanagh_file_names
        .iter()
        .map(|s| s.to_string())
        .collect();

    let kavanagh_matches: Vec<&ContentMatch> = response
        .content_matches
        .iter()
        .filter(|cm| {
            let fname = cm
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            kavanagh_names_set.contains(&fname)
        })
        .collect();

    if !kavanagh_matches.is_empty() {
        eprintln!("!!! FALSE POSITIVES from kavanagh files:");
        for cm in &kavanagh_matches {
            eprintln!(
                "  {}:{}: {:?}",
                cm.path.file_name().unwrap_or_default().to_string_lossy(),
                cm.line_number,
                cm.line_content.trim()
            );
        }
    }

    assert_eq!(
        kavanagh_matches.len(),
        0,
        "REGRESSION: searching 'kolbey' returned {} matches from kavanagh files (expected 0). \
         This is the exact reported false positive bug.",
        kavanagh_matches.len()
    );

    // --- Assertion 2: ALL matches come from kolbey files ---
    let kolbey_names_set: std::collections::HashSet<String> = kolbey_file_names
        .iter()
        .map(|s| s.to_string())
        .collect();

    for (i, cm) in response.content_matches.iter().enumerate() {
        let fname = cm
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        assert!(
            kolbey_names_set.contains(&fname),
            "Match #{} came from unexpected file '{}' (expected only kolbey files). line={}: {:?}",
            i,
            fname,
            cm.line_number,
            cm.line_content.trim()
        );
    }

    println!(
        "All {} matches are from kolbey files (0 from kavanagh, 0 from filler)",
        response.content_matches.len()
    );

    // --- Assertion 3: Validate match_range for every result ---
    for (i, cm) in response.content_matches.iter().enumerate() {
        // match_range bounds
        assert!(
            cm.match_range.start < cm.match_range.end,
            "Match #{}: match_range.start ({}) must be < end ({}) -- {}:{}",
            i,
            cm.match_range.start,
            cm.match_range.end,
            cm.path.file_name().unwrap_or_default().to_string_lossy(),
            cm.line_number
        );

        assert!(
            cm.match_range.end <= cm.line_content.len(),
            "Match #{}: match_range.end ({}) exceeds line_content.len() ({}) -- {}:{}",
            i,
            cm.match_range.end,
            cm.line_content.len(),
            cm.path.file_name().unwrap_or_default().to_string_lossy(),
            cm.line_number
        );

        // Extracted text must be "kolbey" (case-insensitive)
        let extracted = &cm.line_content[cm.match_range.clone()];
        assert!(
            extracted.to_lowercase() == "kolbey",
            "Match #{}: extracted text {:?} != 'kolbey' -- {}:{} range={:?}",
            i,
            extracted,
            cm.path.file_name().unwrap_or_default().to_string_lossy(),
            cm.line_number,
            cm.match_range
        );
    }

    println!("All {} match_ranges validated (bounds + extracted text)", response.content_matches.len());

    // --- Assertion 4: Every matched file on disk actually contains "kolbey" ---
    let fp_count = validate_content_matches(&response.content_matches, "kolbey");
    assert_eq!(
        fp_count, 0,
        "Found {} false positives where file on disk does NOT contain 'kolbey'",
        fp_count
    );

    // --- Assertion 5: Should have found a reasonable number of matches ---
    // 6 kolbey files x 25 lines each with "kolbey" = ~150 matches expected
    assert!(
        response.content_matches.len() >= 10,
        "Expected at least 10 matches for 'kolbey', got {} -- test directory may be misconfigured",
        response.content_matches.len()
    );

    // --- Assertion 6: Zero matches from filler files ---
    let filler_match_count = response
        .content_matches
        .iter()
        .filter(|cm| {
            let fname = cm
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            fname.starts_with("document_")
        })
        .count();

    assert_eq!(
        filler_match_count, 0,
        "Unexpected {} matches from filler files (they should not contain 'kolbey')",
        filler_match_count
    );

    println!(
        "test_kolbey_kavanagh_regression: PASSED ({} matches, 0 kavanagh FP, 0 filler FP, all match_ranges valid)",
        response.content_matches.len()
    );
}

// ============================================================================
// Test: 100 rapid queries stress test with content validation
// ============================================================================

/// Stress test: run 100 sequential searches alternating between 5 distinct
/// patterns, validating every ContentMatch for each query.
///
/// Setup:
/// - 50 files total: 5 groups of 5 files each containing a unique pattern
///   ("alpha", "bravo", "charlie", "delta", "echo"), plus 25 filler files
/// - Each pattern file has ~100 lines, with ~25 lines containing the pattern
/// - Files are 4KB+ to go through GPU dispatch path
///
/// For each of the 100 queries:
/// 1. Pick a pattern (rotating through alpha/bravo/charlie/delta/echo)
/// 2. Run orchestrator.search() (blocking)
/// 3. Validate every ContentMatch: line_content and file on disk contain pattern
/// 4. Accumulate false positive count
///
/// Asserts total false positives across all 100 queries == 0.
#[test]
fn test_100_rapid_queries_no_false_positives() {
    let (_device, _pso, mut orchestrator) = create_orchestrator();

    // --- Build test directory with 50 files ---
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let root = dir.path();

    let patterns = ["alpha", "bravo", "charlie", "delta", "echo"];

    // 5 groups of 5 files, each group containing one unique pattern
    let mut pattern_paths: Vec<Vec<PathBuf>> = Vec::new();
    for (group_idx, pattern) in patterns.iter().enumerate() {
        let mut group_paths = Vec::new();
        for file_idx in 0..5 {
            let path = root.join(format!("{}_{}.txt", pattern, file_idx));
            let mut content = String::with_capacity(6000);
            for line in 0..100 {
                if line % 4 == 0 {
                    // ~25 lines per file contain the pattern
                    content.push_str(&format!(
                        "line {} contains {} data for group {} file {} with measurement results\n",
                        line, pattern, group_idx, file_idx
                    ));
                } else {
                    // Filler lines -- use group-specific but pattern-free content
                    content.push_str(&format!(
                        "measurement line {} in dataset group {} file {} with numeric values and observations\n",
                        line, group_idx, file_idx
                    ));
                }
            }
            // Verify content contains the pattern and NOT the other patterns
            assert!(
                content.to_lowercase().contains(pattern),
                "BUG IN TEST: {} file should contain '{}'",
                path.display(),
                pattern
            );
            for other in &patterns {
                if *other != *pattern {
                    assert!(
                        !content.to_lowercase().contains(other),
                        "BUG IN TEST: {} file contains '{}' (should only contain '{}')",
                        path.display(),
                        other,
                        pattern
                    );
                }
            }
            fs::write(&path, &content).expect("write pattern file");
            group_paths.push(path);
        }
        pattern_paths.push(group_paths);
    }

    // 25 filler files -- no patterns at all
    let mut filler_paths = Vec::new();
    for i in 0..25 {
        let path = root.join(format!("filler_{:03}.txt", i));
        let mut content = String::with_capacity(6000);
        for line in 0..100 {
            content.push_str(&format!(
                "generic filler line {} in document {} with lorem ipsum dolor sit amet consectetur\n",
                line, i
            ));
        }
        // Verify no accidental pattern content
        for pattern in &patterns {
            assert!(
                !content.to_lowercase().contains(pattern),
                "BUG IN TEST: filler file {} contains '{}'",
                path.display(),
                pattern
            );
        }
        fs::write(&path, &content).expect("write filler file");
        filler_paths.push(path);
    }

    println!("=== 100 rapid queries stress test ===");
    println!(
        "Test directory: {} ({} pattern files + {} filler = {} total)",
        root.display(),
        patterns.len() * 5,
        filler_paths.len(),
        patterns.len() * 5 + filler_paths.len()
    );

    // --- Run 100 sequential searches ---
    let mut total_false_positives = 0usize;
    let mut total_matches = 0usize;

    for query_idx in 0..100 {
        let pattern = patterns[query_idx % patterns.len()];

        let request = SearchRequest {
            pattern: pattern.to_string(),
            root: root.to_path_buf(),
            file_types: None,
            case_sensitive: false,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);
        let match_count = response.content_matches.len();
        total_matches += match_count;

        // Validate every ContentMatch
        let fp_count = validate_content_matches(&response.content_matches, pattern);
        if fp_count > 0 {
            eprintln!(
                "Query #{} '{}': {} false positives out of {} matches!",
                query_idx, pattern, fp_count, match_count
            );
        }
        total_false_positives += fp_count;

        // Also verify no matches come from files belonging to other patterns
        for cm in &response.content_matches {
            let fname = cm
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            // Check it's not a filler file
            if fname.starts_with("filler_") {
                eprintln!(
                    "Query #{} '{}': unexpected match from filler file {} line {}",
                    query_idx, pattern, fname, cm.line_number
                );
                total_false_positives += 1;
            }

            // Check it's not from a different pattern's file
            for other in &patterns {
                if *other != pattern && fname.starts_with(other) {
                    eprintln!(
                        "Query #{} '{}': cross-contamination match from {} file {} line {}",
                        query_idx, pattern, other, fname, cm.line_number
                    );
                    total_false_positives += 1;
                }
            }
        }

        // Progress output every 10 queries
        if (query_idx + 1) % 10 == 0 {
            println!(
                "  Queries {}-{}: pattern='{}', matches={}, FP so far={}",
                query_idx.saturating_sub(8),
                query_idx + 1,
                pattern,
                match_count,
                total_false_positives
            );
        }
    }

    println!(
        "=== Results: 100 queries, {} total matches, {} total false positives ===",
        total_matches, total_false_positives
    );

    assert_eq!(
        total_false_positives, 0,
        "Stress test: {} false positives across 100 queries ({} total matches). \
         Expected zero false positives.",
        total_false_positives, total_matches
    );

    // Sanity: should have found a reasonable number of total matches
    // 100 queries * ~125 matches per query (5 files * 25 lines) = ~12500
    assert!(
        total_matches >= 100,
        "Expected at least 100 total matches across 100 queries, got {}",
        total_matches
    );

    println!(
        "test_100_rapid_queries_no_false_positives: PASSED ({} queries, {} matches, 0 FP)",
        100, total_matches
    );
}

// ============================================================================
// Test: match_range corruption stress test (20 varied searches)
// ============================================================================

/// Stress test that match_range is never corrupted across 20 searches with
/// patterns of varying length: short ("fn"), medium ("kolbey"), and long
/// ("Patrick Kavanagh").
///
/// For every ContentMatch from every search, validates:
/// 1. match_range.start < match_range.end
/// 2. match_range.end <= line_content.len()
/// 3. line_content[match_range] case-insensitively equals the search pattern
///
/// This catches byte-level corruption where match_range points to wrong bytes,
/// extends beyond line boundaries, or extracts text that doesn't match the pattern.
#[test]
fn test_match_range_never_corrupted() {
    let (_device, _pso, mut orchestrator) = create_orchestrator();

    // --- Build test directory with files containing all three patterns ---
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let root = dir.path();

    // Short pattern files: contain "fn" at known positions
    for i in 0..5 {
        let path = root.join(format!("fn_source_{}.txt", i));
        let mut content = String::with_capacity(6000);
        for line in 0..100 {
            if line % 5 == 0 {
                content.push_str(&format!(
                    "pub fn process_batch_{}(data: &[u8]) -> Result<(), Error> {{\n",
                    line + i * 100
                ));
            } else if line % 5 == 2 {
                content.push_str(&format!(
                    "    fn helper_{}_{}(x: i32) -> i32 {{ x + 1 }}\n",
                    i, line
                ));
            } else {
                content.push_str(&format!(
                    "    let value_{} = compute_result({}, {});\n",
                    line, i, line
                ));
            }
        }
        assert!(
            content.to_lowercase().contains("fn"),
            "BUG IN TEST: fn file should contain 'fn'"
        );
        fs::write(&path, &content).expect("write fn file");
    }

    // Medium pattern files: contain "kolbey" at known positions
    for i in 0..5 {
        let path = root.join(format!("kolbey_data_{}.txt", i));
        let mut content = String::with_capacity(6000);
        for line in 0..100 {
            if line % 5 == 1 {
                content.push_str(&format!(
                    "the kolbey algorithm iteration {} processes batch {} efficiently\n",
                    line, i
                ));
            } else if line % 5 == 3 {
                content.push_str(&format!(
                    "results from kolbey experiment {} run {} show improvements\n",
                    i, line
                ));
            } else {
                content.push_str(&format!(
                    "measurement line {} in dataset {} with numeric observations recorded\n",
                    line, i
                ));
            }
        }
        assert!(
            content.to_lowercase().contains("kolbey"),
            "BUG IN TEST: kolbey file should contain 'kolbey'"
        );
        fs::write(&path, &content).expect("write kolbey file");
    }

    // Long pattern files: contain "Patrick Kavanagh" at known positions
    for i in 0..5 {
        let path = root.join(format!("kavanagh_text_{}.txt", i));
        let mut content = String::with_capacity(6000);
        for line in 0..100 {
            if line % 5 == 0 {
                content.push_str(&format!(
                    "the poet Patrick Kavanagh was born in Inniskeen reference {} line {}\n",
                    i, line
                ));
            } else if line % 5 == 4 {
                content.push_str(&format!(
                    "Patrick Kavanagh wrote about rural Ireland in collection {} section {}\n",
                    i, line
                ));
            } else {
                content.push_str(&format!(
                    "this is a literary analysis document {} line {} with various notes\n",
                    i, line
                ));
            }
        }
        assert!(
            content.to_lowercase().contains("patrick kavanagh"),
            "BUG IN TEST: kavanagh file should contain 'Patrick Kavanagh'"
        );
        fs::write(&path, &content).expect("write kavanagh file");
    }

    // Filler files to add GPU batch pressure
    for i in 0..10 {
        let path = root.join(format!("filler_{:03}.txt", i));
        let mut content = String::with_capacity(6000);
        for line in 0..100 {
            content.push_str(&format!(
                "generic document {} line {} with lorem ipsum dolor sit amet content padding\n",
                i, line
            ));
        }
        fs::write(&path, &content).expect("write filler file");
    }

    println!("=== match_range corruption stress test ===");
    println!(
        "Test directory: {} (5 fn + 5 kolbey + 5 kavanagh + 10 filler = 25 files)",
        root.display()
    );

    // --- Run 20 searches alternating short/medium/long patterns ---
    let search_patterns = ["fn", "kolbey", "Patrick Kavanagh"];
    let mut total_matches = 0usize;
    let mut total_corruptions = 0usize;

    for query_idx in 0..20 {
        let pattern = search_patterns[query_idx % search_patterns.len()];

        let request = SearchRequest {
            pattern: pattern.to_string(),
            root: root.to_path_buf(),
            file_types: None,
            case_sensitive: false,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);
        let match_count = response.content_matches.len();
        total_matches += match_count;

        // Validate every match_range
        for (i, cm) in response.content_matches.iter().enumerate() {
            // (1) match_range.start < match_range.end
            if cm.match_range.start >= cm.match_range.end {
                eprintln!(
                    "CORRUPTION query#{} '{}' match#{}: start ({}) >= end ({}) -- {}:{}",
                    query_idx,
                    pattern,
                    i,
                    cm.match_range.start,
                    cm.match_range.end,
                    cm.path.file_name().unwrap_or_default().to_string_lossy(),
                    cm.line_number
                );
                total_corruptions += 1;
                continue;
            }

            // (2) match_range.end <= line_content.len()
            if cm.match_range.end > cm.line_content.len() {
                eprintln!(
                    "CORRUPTION query#{} '{}' match#{}: end ({}) > line_content.len() ({}) -- {}:{} content={:?}",
                    query_idx,
                    pattern,
                    i,
                    cm.match_range.end,
                    cm.line_content.len(),
                    cm.path.file_name().unwrap_or_default().to_string_lossy(),
                    cm.line_number,
                    cm.line_content.trim()
                );
                total_corruptions += 1;
                continue;
            }

            // (3) extracted text at match_range must case-insensitively equal pattern
            let extracted = &cm.line_content[cm.match_range.clone()];
            if extracted.to_lowercase() != pattern.to_lowercase() {
                eprintln!(
                    "CORRUPTION query#{} '{}' match#{}: extracted {:?} != pattern {:?} -- {}:{} range={:?} content={:?}",
                    query_idx,
                    pattern,
                    i,
                    extracted,
                    pattern,
                    cm.path.file_name().unwrap_or_default().to_string_lossy(),
                    cm.line_number,
                    cm.match_range,
                    cm.line_content.trim()
                );
                total_corruptions += 1;
            }
        }

        // Progress: print every query result
        println!(
            "  Query {:2}/{}: pattern={:20} matches={:4} corruptions={}",
            query_idx + 1,
            20,
            format!("{:?}", pattern),
            match_count,
            total_corruptions
        );
    }

    println!(
        "=== Results: 20 queries, {} total matches, {} total corruptions ===",
        total_matches, total_corruptions
    );

    assert_eq!(
        total_corruptions, 0,
        "match_range corruption stress test: {} corruptions across 20 queries ({} total matches). \
         Expected zero corruptions.",
        total_corruptions, total_matches
    );

    // Sanity: should have found matches for each pattern type
    assert!(
        total_matches >= 20,
        "Expected at least 20 total matches across 20 queries, got {} -- test files may be misconfigured",
        total_matches
    );

    println!(
        "test_match_range_never_corrupted: PASSED (20 queries, {} matches, 0 corruptions)",
        total_matches
    );
}
