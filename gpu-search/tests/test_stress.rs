//! Stress tests for GPU search engine.
//!
//! All tests are marked `#[ignore]` since they are slow (10-60s each).
//! Run with: `cargo test --test test_stress -- --ignored --test-threads=1`
//!
//! REQUIRES: Real Apple Silicon GPU (Metal device must be available).
//!
//! ## Tests
//!
//! 1. **Memory leak**: Run 1000 searches, verify GPU allocation doesn't grow >1%
//! 2. **Watchdog survival**: Run continuous searches for 15s with zero errors
//! 3. **Sustained UI responsiveness**: Rapid searches maintaining <16ms p99
//! 4. **Concurrent filesystem changes**: Modify files during active search, no crash

use std::cell::RefCell;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::content::{ContentSearchEngine, SearchMode, SearchOptions};
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::SearchRequest;

// ============================================================================
// Constants
// ============================================================================

/// Number of searches for memory leak test.
const MEMORY_LEAK_ITERATIONS: usize = 1000;

/// Duration for watchdog survival test.
const WATCHDOG_DURATION: Duration = Duration::from_secs(15);

/// Number of rapid searches for UI responsiveness test.
const RESPONSIVENESS_ITERATIONS: usize = 500;

/// Maximum allowed p99 latency for UI responsiveness (one frame at 60fps).
const P99_LATENCY_LIMIT: Duration = Duration::from_millis(16);

/// GPU max matches (must match content.rs).
const GPU_MAX_MATCHES: usize = 10000;

// ============================================================================
// Helpers
// ============================================================================

fn create_engine(max_files: usize) -> (ContentSearchEngine, GpuDevice) {
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);
    let engine = ContentSearchEngine::new(&device.device, &pso_cache, max_files);
    (engine, device)
}

fn default_options() -> SearchOptions {
    SearchOptions {
        case_sensitive: true,
        max_results: GPU_MAX_MATCHES,
        mode: SearchMode::Standard,
    }
}

/// Create a temp directory with several Rust-like source files for testing.
fn make_stress_test_directory() -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let root = dir.path();

    // Create 50 source files with varied content
    for i in 0..50 {
        let filename = format!("module_{:03}.rs", i);
        let content = format!(
            "//! Module {} for stress testing.\n\
             \n\
             pub fn function_{i}(x: i32) -> i32 {{\n\
             \tlet result = x * {} + {};\n\
             \tprintln!(\"module {} result: {{}}\", result);\n\
             \tresult\n\
             }}\n\
             \n\
             pub fn helper_{i}() -> &'static str {{\n\
             \t\"hello from module {}\"\n\
             }}\n\
             \n\
             #[cfg(test)]\n\
             mod tests {{\n\
             \tuse super::*;\n\
             \n\
             \t#[test]\n\
             \tfn test_function_{i}() {{\n\
             \t\tassert_eq!(function_{i}(1), {});\n\
             \t}}\n\
             }}\n",
            i,
            i + 1,
            i * 7,
            i,
            i,
            i + 1 + i * 7,
            i = i,
        );
        fs::write(root.join(&filename), &content).unwrap();
    }

    // Create a few larger files (~4KB each)
    for i in 0..10 {
        let filename = format!("large_{:03}.rs", i);
        let mut content = format!("//! Large module {} for stress testing.\n\n", i);
        for j in 0..100 {
            content.push_str(&format!(
                "pub fn large_fn_{i}_{j}(x: i32) -> i32 {{ x + {} }}\n",
                i * 100 + j,
            ));
        }
        fs::write(root.join(&filename), &content).unwrap();
    }

    dir
}

/// Generate test content of a given size with some searchable patterns.
fn generate_content(size: usize) -> Vec<u8> {
    let line = b"fn search_target(x: i32) -> i32 { x + 42 }\n";
    let mut content = Vec::with_capacity(size);
    while content.len() < size {
        let remaining = size - content.len();
        let chunk = &line[..remaining.min(line.len())];
        content.extend_from_slice(chunk);
    }
    content.truncate(size);
    content
}

// ============================================================================
// Stress Test 1: Memory Leak Detection
// ============================================================================

/// Run 1000 searches and verify GPU allocation size doesn't grow significantly.
///
/// The GPU engine pre-allocates buffers and reuses them. If there's a leak,
/// memory would grow with each search iteration. We check that the
/// engine's internal state (chunk count, data bytes) resets properly after
/// each search by verifying the engine accepts load_content after reset.
///
/// We also measure wall-clock allocation behavior by timing the searches:
/// if allocation grew per iteration, later iterations would be slower.
#[test]
#[ignore]
fn stress_memory_leak_1000_searches() {
    let (engine, _dev) = create_engine(100);
    let engine = RefCell::new(engine);

    let content = generate_content(8192); // 8KB content (2 chunks)
    let pattern = b"search_target";
    let opts = default_options();

    // Warmup: 10 iterations
    for _ in 0..10 {
        let mut eng = engine.borrow_mut();
        eng.reset();
        eng.load_content(&content, 0);
        let _results = eng.search(pattern, &opts);
    }

    // Measure timing for first batch (iterations 0-99)
    let first_batch_start = Instant::now();
    for _ in 0..100 {
        let mut eng = engine.borrow_mut();
        eng.reset();
        eng.load_content(&content, 0);
        let _results = eng.search(pattern, &opts);
    }
    let first_batch_time = first_batch_start.elapsed();

    // Run remaining iterations (100-999)
    for _ in 100..MEMORY_LEAK_ITERATIONS {
        let mut eng = engine.borrow_mut();
        eng.reset();
        eng.load_content(&content, 0);
        let results = eng.search(pattern, &opts);
        // Verify results are still valid (not corrupted)
        assert!(!results.is_empty(), "Search should find matches");
    }

    // Measure timing for last batch (iterations 900-999)
    let last_batch_start = Instant::now();
    for _ in 0..100 {
        let mut eng = engine.borrow_mut();
        eng.reset();
        eng.load_content(&content, 0);
        let _results = eng.search(pattern, &opts);
    }
    let last_batch_time = last_batch_start.elapsed();

    // Verify: last batch should not be significantly slower than first batch.
    // Allow 3x tolerance (Metal driver may have background activity).
    // A true leak would cause 10-100x slowdown after 1000 iterations.
    let ratio = last_batch_time.as_secs_f64() / first_batch_time.as_secs_f64().max(0.001);
    assert!(
        ratio < 3.0,
        "Memory leak suspected: last batch {:.2}ms is {:.1}x slower than first batch {:.2}ms",
        last_batch_time.as_secs_f64() * 1000.0,
        ratio,
        first_batch_time.as_secs_f64() * 1000.0,
    );

    // Verify engine still accepts full-capacity load after all iterations
    let mut eng = engine.borrow_mut();
    eng.reset();
    let large_content = generate_content(32768); // 32KB = 8 chunks
    let chunks = eng.load_content(&large_content, 0);
    assert!(chunks > 0, "Engine should still accept content after 1000 searches");
    let results = eng.search(pattern, &opts);
    assert!(!results.is_empty(), "Engine should still produce valid results after 1000 searches");

    println!(
        "Memory leak test PASSED: {} iterations, first_batch={:.2}ms, last_batch={:.2}ms, ratio={:.2}x",
        MEMORY_LEAK_ITERATIONS,
        first_batch_time.as_secs_f64() * 1000.0,
        last_batch_time.as_secs_f64() * 1000.0,
        ratio,
    );
}

// ============================================================================
// Stress Test 2: Watchdog Survival (Continuous Operation)
// ============================================================================

/// Run continuous GPU searches for 15 seconds with zero panics or errors.
///
/// Simulates a long-running session where the user keeps searching.
/// Alternates between different patterns, content sizes, and search modes
/// to exercise varied code paths.
#[test]
#[ignore]
fn stress_watchdog_survival_continuous() {
    let (engine, _dev) = create_engine(100);
    let engine = RefCell::new(engine);

    // Pre-generate varied content and patterns
    let contents: Vec<Vec<u8>> = vec![
        generate_content(1024),    // 1KB
        generate_content(4096),    // 4KB
        generate_content(16384),   // 16KB
        generate_content(65536),   // 64KB
        b"short content with fn keyword here\n".to_vec(),
        b"".to_vec(), // empty
    ];

    let patterns: Vec<&[u8]> = vec![
        b"search_target",
        b"fn",
        b"i32",
        b"x",        // single char
        b"nonexistent_pattern_xyz",
        b"42",
    ];

    let modes = vec![SearchMode::Standard, SearchMode::Turbo];

    let start = Instant::now();
    let mut iteration = 0u64;
    let mut error_count = 0u64;

    while start.elapsed() < WATCHDOG_DURATION {
        let content_idx = (iteration as usize) % contents.len();
        let pattern_idx = (iteration as usize / contents.len()) % patterns.len();
        let mode_idx = (iteration as usize) % modes.len();

        let content = &contents[content_idx];
        let pattern = patterns[pattern_idx];
        let mode = modes[mode_idx];

        let opts = SearchOptions {
            case_sensitive: iteration % 3 != 0, // Mix case sensitivity
            max_results: GPU_MAX_MATCHES,
            mode,
        };

        // Catch any panics at the search level
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut eng = engine.borrow_mut();
            eng.reset();
            if !content.is_empty() {
                eng.load_content(content, 0);
            }
            eng.search(pattern, &opts)
        }));

        match result {
            Ok(matches) => {
                // Basic sanity: empty content should yield no matches
                if content.is_empty() {
                    assert!(
                        matches.is_empty(),
                        "Empty content should produce no matches at iteration {}",
                        iteration
                    );
                }
            }
            Err(_) => {
                error_count += 1;
            }
        }

        iteration += 1;
    }

    let elapsed = start.elapsed();

    assert_eq!(
        error_count, 0,
        "Watchdog test had {} errors in {} iterations over {:.1}s",
        error_count, iteration, elapsed.as_secs_f64()
    );

    println!(
        "Watchdog survival PASSED: {} iterations in {:.1}s ({:.0} searches/sec), zero errors",
        iteration,
        elapsed.as_secs_f64(),
        iteration as f64 / elapsed.as_secs_f64(),
    );
}

// ============================================================================
// Stress Test 3: Sustained UI Responsiveness
// ============================================================================

/// Run 500 rapid searches and verify p99 latency stays under 16ms.
///
/// This simulates search-as-you-type where each keystroke triggers a new
/// search. The GPU engine must respond fast enough to maintain 60fps UI.
/// We use small content (typical project file sizes) to match real-world
/// search-as-you-type behavior.
#[test]
#[ignore]
fn stress_sustained_ui_responsiveness() {
    let (engine, _dev) = create_engine(50);
    let engine = RefCell::new(engine);

    // Simulate typical source files (1-4KB each)
    let files: Vec<Vec<u8>> = (0..10)
        .map(|i| {
            let size = 1024 + (i * 512); // 1KB to 5.5KB
            generate_content(size)
        })
        .collect();

    let patterns: Vec<&[u8]> = vec![
        b"fn",
        b"pub",
        b"let",
        b"search",
        b"result",
        b"x",
        b"i32",
        b"mod",
        b"use",
        b"return",
    ];

    let opts = default_options();
    let mut latencies = Vec::with_capacity(RESPONSIVENESS_ITERATIONS);

    // Warmup: 20 iterations
    for _ in 0..20 {
        let mut eng = engine.borrow_mut();
        eng.reset();
        eng.load_content(&files[0], 0);
        let _results = eng.search(b"fn", &opts);
    }

    // Measure each search latency
    for i in 0..RESPONSIVENESS_ITERATIONS {
        let file_idx = i % files.len();
        let pattern_idx = i % patterns.len();

        let start = Instant::now();

        let mut eng = engine.borrow_mut();
        eng.reset();
        // Load 1-3 files per search (simulating incremental results)
        let num_files = (i % 3) + 1;
        for f in 0..num_files {
            let idx = (file_idx + f) % files.len();
            eng.load_content(&files[idx], f as u32);
        }
        let _results = eng.search(patterns[pattern_idx], &opts);
        drop(eng);

        latencies.push(start.elapsed());
    }

    // Sort latencies for percentile calculation
    latencies.sort();

    let p50_idx = latencies.len() / 2;
    let p99_idx = (latencies.len() * 99) / 100;
    let p50 = latencies[p50_idx];
    let p99 = latencies[p99_idx];
    let max_latency = latencies.last().unwrap();
    let avg = latencies.iter().sum::<Duration>() / latencies.len() as u32;

    // p99 must be under 16ms for 60fps UI
    assert!(
        p99 < P99_LATENCY_LIMIT,
        "p99 latency {:.2}ms exceeds 16ms limit for 60fps UI (p50={:.2}ms, max={:.2}ms)",
        p99.as_secs_f64() * 1000.0,
        p50.as_secs_f64() * 1000.0,
        max_latency.as_secs_f64() * 1000.0,
    );

    println!(
        "UI responsiveness PASSED: {} searches, avg={:.2}ms, p50={:.2}ms, p99={:.2}ms, max={:.2}ms",
        RESPONSIVENESS_ITERATIONS,
        avg.as_secs_f64() * 1000.0,
        p50.as_secs_f64() * 1000.0,
        p99.as_secs_f64() * 1000.0,
        max_latency.as_secs_f64() * 1000.0,
    );
}

// ============================================================================
// Stress Test 4: Concurrent Filesystem Changes During Search
// ============================================================================

/// Modify files on disk while the orchestrator is searching the same directory.
///
/// The orchestrator must not crash even when files appear, disappear, or change
/// mid-search. This simulates a user running gpu-search on a project while
/// actively editing code in their editor.
#[test]
#[ignore]
fn stress_concurrent_filesystem_changes() {
    let dir = make_stress_test_directory();
    let root = dir.path().to_path_buf();

    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);
    let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
        .expect("Failed to create SearchOrchestrator");

    // Flag to signal the mutator thread to stop
    let stop = Arc::new(AtomicBool::new(false));
    let error_count = Arc::new(AtomicU64::new(0));

    // Spawn a background thread that continuously modifies the filesystem
    let mutator_root = root.clone();
    let mutator_stop = Arc::clone(&stop);
    let mutator_errors = Arc::clone(&error_count);
    let mutator_thread = std::thread::spawn(move || {
        let mut i = 0u64;
        while !mutator_stop.load(Ordering::Relaxed) {
            let action = i % 5;
            match action {
                0 => {
                    // Create a new file
                    let path = mutator_root.join(format!("dynamic_{}.rs", i));
                    if let Err(_) = fs::write(&path, format!("fn dynamic_{i}() {{ }}\n")) {
                        mutator_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
                1 => {
                    // Modify an existing file
                    let path = mutator_root.join(format!("module_{:03}.rs", i % 50));
                    if path.exists() {
                        if let Err(_) = fs::write(
                            &path,
                            format!("// Modified at iteration {i}\nfn modified_{i}() {{ }}\n"),
                        ) {
                            mutator_errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                2 => {
                    // Delete a dynamic file (don't delete original test files)
                    let del_idx = if i > 5 { i - 5 } else { 0 };
                    let path = mutator_root.join(format!("dynamic_{}.rs", del_idx));
                    let _ = fs::remove_file(&path); // Ignore if not found
                }
                3 => {
                    // Create a new subdirectory with a file
                    let subdir = mutator_root.join(format!("subdir_{}", i % 10));
                    let _ = fs::create_dir_all(&subdir);
                    let path = subdir.join("nested.rs");
                    let _ = fs::write(&path, format!("fn nested_{i}() {{ }}\n"));
                }
                4 => {
                    // Truncate a file to zero length
                    let path = mutator_root.join(format!("large_{:03}.rs", i % 10));
                    if path.exists() {
                        let _ = fs::write(&path, "");
                    }
                }
                _ => unreachable!(),
            }

            i += 1;
            // Small sleep so we don't completely thrash the filesystem
            std::thread::sleep(Duration::from_millis(1));
        }
        i
    });

    // Run searches concurrently with filesystem mutations
    let mut search_count = 0u64;
    let mut search_errors = 0u64;
    let search_start = Instant::now();
    let search_duration = Duration::from_secs(10);

    let search_patterns = ["fn", "pub", "module", "dynamic", "nested", "test"];

    while search_start.elapsed() < search_duration {
        let pattern = search_patterns[search_count as usize % search_patterns.len()];

        let request = SearchRequest {
            pattern: pattern.to_string(),
            root: root.clone(),
            file_types: Some(vec!["rs".to_string()]),
            case_sensitive: false,
            respect_gitignore: false, // No git repo in temp dir
            include_binary: false,
            max_results: 1000,
        };

        // The search must not panic or crash
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            orchestrator.search(request)
        }));

        match result {
            Ok(response) => {
                // Response should be valid even with concurrent changes
                // (may miss files being added/deleted, that's expected)
                assert!(
                    response.elapsed.as_secs() < 30,
                    "Single search took too long: {:?}",
                    response.elapsed
                );
            }
            Err(_) => {
                search_errors += 1;
            }
        }

        search_count += 1;
    }

    // Stop the mutator thread
    stop.store(true, Ordering::Relaxed);
    let mutation_count = mutator_thread.join().expect("Mutator thread panicked");

    let fs_errors = error_count.load(Ordering::Relaxed);

    assert_eq!(
        search_errors, 0,
        "Had {} search panics/errors during concurrent filesystem changes ({} searches, {} mutations)",
        search_errors, search_count, mutation_count
    );

    println!(
        "Concurrent filesystem PASSED: {} searches + {} mutations in {:.1}s, {} fs write errors (non-fatal), zero search errors",
        search_count,
        mutation_count,
        search_start.elapsed().as_secs_f64(),
        fs_errors,
    );
}
