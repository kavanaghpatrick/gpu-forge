//! Benchmark search scenarios to measure real-world performance.
//!
//! Run with: cargo test --release --test bench_search_scenarios -- --nocapture

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::cancel::{cancellation_pair, SearchGeneration, SearchSession};
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::{SearchRequest, SearchUpdate};
use std::path::PathBuf;
use std::time::Instant;

fn make_session() -> SearchSession {
    let (token, _handle) = cancellation_pair();
    let gen = SearchGeneration::new();
    let guard = gen.next();
    SearchSession { token, guard }
}

fn bench_blocking(
    orch: &mut SearchOrchestrator,
    label: &str,
    pattern: &str,
    root: &str,
) {
    let root = PathBuf::from(root);
    if !root.exists() {
        eprintln!("  SKIP {}: path does not exist", label);
        return;
    }

    let request = SearchRequest {
        pattern: pattern.to_string(),
        root,
        file_types: None,
        case_sensitive: false,
        respect_gitignore: true,
        include_binary: false,
        max_results: 1000,
    };

    let start = Instant::now();
    let response = orch.search(request);
    let elapsed = start.elapsed();

    eprintln!(
        "  {:<40} {:>7.1}ms | {:>6} files | {:>5} file_matches | {:>5} content_matches",
        label,
        elapsed.as_secs_f64() * 1000.0,
        response.total_files_searched,
        response.file_matches.len(),
        response.content_matches.len(),
    );
}

fn bench_streaming(
    orch: &mut SearchOrchestrator,
    label: &str,
    pattern: &str,
    root: &str,
) {
    let root = PathBuf::from(root);
    if !root.exists() {
        eprintln!("  SKIP {}: path does not exist", label);
        return;
    }

    let request = SearchRequest {
        pattern: pattern.to_string(),
        root,
        file_types: None,
        case_sensitive: false,
        respect_gitignore: true,
        include_binary: false,
        max_results: 1000,
    };

    let (update_tx, update_rx) = crossbeam_channel::unbounded();
    let session = make_session();

    let start = Instant::now();
    let response = orch.search_streaming(request, &update_tx, &session);
    let total_elapsed = start.elapsed();

    // Measure time-to-first-result from the update channel
    let mut first_content_time: Option<std::time::Duration> = None;
    let mut first_file_time: Option<std::time::Duration> = None;
    let mut content_update_count = 0u32;
    let mut file_update_count = 0u32;

    // Re-read updates to count (they were already sent during search_streaming)
    while let Ok(stamped) = update_rx.try_recv() {
        match stamped.update {
            SearchUpdate::ContentMatches(_) => {
                content_update_count += 1;
            }
            SearchUpdate::FileMatches(_) => {
                file_update_count += 1;
            }
            SearchUpdate::Complete(_) => {}
        }
    }

    // For TTFR we need to instrument differently - just report total for now
    eprintln!(
        "  {:<40} {:>7.1}ms | {:>6} files | {:>5} file_m | {:>5} content_m | {:>2} batches",
        label,
        total_elapsed.as_secs_f64() * 1000.0,
        response.total_files_searched,
        response.file_matches.len(),
        response.content_matches.len(),
        content_update_count,
    );
}

fn bench_streaming_with_ttfr(
    orch: &mut SearchOrchestrator,
    label: &str,
    pattern: &str,
    root: &str,
) {
    let root_path = PathBuf::from(root);
    if !root_path.exists() {
        eprintln!("  SKIP {}: path does not exist", label);
        return;
    }

    let request = SearchRequest {
        pattern: pattern.to_string(),
        root: root_path,
        file_types: None,
        case_sensitive: false,
        respect_gitignore: true,
        include_binary: false,
        max_results: 1000,
    };

    // For TTFR: run streaming in a thread and measure when first update arrives
    let (update_tx, update_rx) = crossbeam_channel::unbounded();
    let session = make_session();

    let start = Instant::now();

    // Run search (blocking this thread)
    let response = orch.search_streaming(request, &update_tx, &session);
    let total = start.elapsed();

    eprintln!(
        "  {:<40} {:>7.1}ms total | {:>6} files | {:>5} fm | {:>5} cm",
        label,
        total.as_secs_f64() * 1000.0,
        response.total_files_searched,
        response.file_matches.len(),
        response.content_matches.len(),
    );
}

fn bench_cancellation(
    orch: &mut SearchOrchestrator,
    label: &str,
    pattern: &str,
    root: &str,
) {
    let root_path = PathBuf::from(root);
    if !root_path.exists() {
        eprintln!("  SKIP {}: path does not exist", label);
        return;
    }

    let request = SearchRequest {
        pattern: pattern.to_string(),
        root: root_path,
        file_types: None,
        case_sensitive: false,
        respect_gitignore: true,
        include_binary: false,
        max_results: 1000,
    };

    let (update_tx, _update_rx) = crossbeam_channel::unbounded();

    // Create a session and cancel it after 100ms
    let gen = SearchGeneration::new();
    let (token, handle) = cancellation_pair();
    let guard = gen.next();
    let session = SearchSession { token, guard };

    // Cancel after 100ms in a separate thread
    let handle_clone = handle.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        handle_clone.cancel();
    });

    let start = Instant::now();
    let response = orch.search_streaming(request, &update_tx, &session);
    let elapsed = start.elapsed();

    eprintln!(
        "  {:<40} {:>7.1}ms (cancelled@100ms) | {:>6} files partial",
        label,
        elapsed.as_secs_f64() * 1000.0,
        response.total_files_searched,
    );
}

#[test]
fn bench_all_scenarios() {
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);
    let mut orch = SearchOrchestrator::new(&device.device, &pso_cache)
        .expect("Failed to create orchestrator");

    let src_dir = env!("CARGO_MANIFEST_DIR").to_string() + "/src";
    let project_dir = env!("CARGO_MANIFEST_DIR").to_string();
    let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/Users/patrickkavanagh".to_string());

    eprintln!("\n=== GPU-SEARCH BENCHMARK SUITE ===\n");

    // ---- Blocking pipeline (baseline) ----
    eprintln!("--- Blocking Pipeline (search()) ---");
    bench_blocking(&mut orch, "src/ 'fn '", "fn ", &src_dir);
    bench_blocking(&mut orch, "src/ 'SearchOrchestrator'", "SearchOrchestrator", &src_dir);
    bench_blocking(&mut orch, "project/ 'fn '", "fn ", &project_dir);
    bench_blocking(&mut orch, "project/ 'ContentMatch'", "ContentMatch", &project_dir);

    eprintln!();

    // ---- Streaming pipeline ----
    eprintln!("--- Streaming Pipeline (search_streaming()) ---");
    bench_streaming(&mut orch, "src/ 'fn '", "fn ", &src_dir);
    bench_streaming(&mut orch, "src/ 'SearchOrchestrator'", "SearchOrchestrator", &src_dir);
    bench_streaming(&mut orch, "project/ 'fn '", "fn ", &project_dir);
    bench_streaming(&mut orch, "project/ 'ContentMatch'", "ContentMatch", &project_dir);
    bench_streaming(&mut orch, "project/ 'old friendship'", "old friendship", &project_dir);

    eprintln!();

    // ---- Larger directory searches ----
    eprintln!("--- Streaming: Larger Directories ---");
    bench_streaming_with_ttfr(&mut orch, "home/ 'gpu-search'", "gpu-search", &home_dir);
    bench_streaming_with_ttfr(&mut orch, "home/ 'old friendship'", "old friendship", &home_dir);
    bench_streaming_with_ttfr(&mut orch, "home/ 'patrick'", "patrick", &home_dir);

    eprintln!();

    // ---- Root search (the problem case) ----
    eprintln!("--- Streaming: Root / (real-world worst case) ---");
    bench_streaming_with_ttfr(&mut orch, "root/ 'old friendship'", "old friendship", "/");

    eprintln!();

    // ---- Cancellation benchmarks ----
    eprintln!("--- Cancellation (cancel after 100ms) ---");
    bench_cancellation(&mut orch, "home/ 'fn ' (cancel@100ms)", "fn ", &home_dir);
    bench_cancellation(&mut orch, "root/ 'patrick' (cancel@100ms)", "patrick", "/");

    eprintln!();

    // ---- Rapid sequential searches (simulating typing) ----
    eprintln!("--- Rapid Sequential Searches (simulating 'old friendship' typing) ---");
    let queries = ["ol", "old", "old ", "old f", "old fr", "old fri", "old friendship"];
    let gen = SearchGeneration::new();

    for (i, query) in queries.iter().enumerate() {
        let request = SearchRequest {
            pattern: query.to_string(),
            root: PathBuf::from(&src_dir),
            file_types: None,
            case_sensitive: false,
            respect_gitignore: true,
            include_binary: false,
            max_results: 1000,
        };

        let (update_tx, _) = crossbeam_channel::unbounded();
        let (token, handle) = cancellation_pair();
        let guard = gen.next();
        let session = SearchSession { token, guard };

        // Cancel previous-generation searches after 10ms (simulating new keystroke)
        if i < queries.len() - 1 {
            let h = handle.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(10));
                h.cancel();
            });
        }

        let start = Instant::now();
        let response = orch.search_streaming(request, &update_tx, &session);
        let elapsed = start.elapsed();

        let cancelled = if session.should_stop() { " [CANCELLED]" } else { "" };
        eprintln!(
            "  {:<25} {:>7.1}ms | {:>5} fm | {:>5} cm{}",
            format!("'{}'", query),
            elapsed.as_secs_f64() * 1000.0,
            response.file_matches.len(),
            response.content_matches.len(),
            cancelled,
        );
    }

    eprintln!("\n=== BENCHMARK COMPLETE ===\n");
}
