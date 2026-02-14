//! Test streaming search from root / to diagnose performance.
use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::search::cancel::{cancellation_pair, SearchGeneration, SearchSession};
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::{SearchRequest, SearchUpdate};
use std::path::PathBuf;
use std::time::Instant;

#[test]
fn test_streaming_from_root() {
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);
    let mut orch = SearchOrchestrator::new(&device.device, &pso_cache)
        .expect("Failed to create orchestrator");

    let root = PathBuf::from("/");
    let request = SearchRequest {
        pattern: "patrick".to_string(),
        root,
        file_types: None,
        case_sensitive: false,
        respect_gitignore: true,
        include_binary: false,
        max_results: 100,
    };

    let (update_tx, update_rx) = crossbeam_channel::unbounded();
    let (token, _handle) = cancellation_pair();
    let gen = SearchGeneration::new();
    let guard = gen.next();
    let session = SearchSession { token, guard };
    let start = Instant::now();

    eprintln!("=== Starting streaming search from / for 'patrick' ===");
    let response = orch.search_streaming(request, &update_tx, &session);

    eprintln!("=== RESULTS ===");
    eprintln!("Files searched: {}", response.total_files_searched);
    eprintln!("File matches: {}", response.file_matches.len());
    eprintln!("Content matches: {}", response.content_matches.len());
    eprintln!("Total time: {:.2}s", start.elapsed().as_secs_f64());

    let mut file_updates = 0u32;
    let mut content_updates = 0u32;
    while let Ok(stamped) = update_rx.try_recv() {
        match stamped.update {
            SearchUpdate::FileMatches(_) => file_updates += 1,
            SearchUpdate::ContentMatches(_) => content_updates += 1,
            SearchUpdate::Complete(_) => {},
        }
    }
    eprintln!("Progressive updates: {} file, {} content", file_updates, content_updates);

    assert!(response.total_files_searched > 0, "Should search some files");
}
