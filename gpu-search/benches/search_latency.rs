//! End-to-end search latency benchmark.
//!
//! Measures full pipeline: SearchRequest -> SearchOrchestrator -> SearchResponse.
//!
//! Benchmarks:
//! - **cold search**: First search including index building from scratch
//! - **cached search**: Subsequent searches with warm index + GPU engine
//! - **index load**: Filesystem scan + GPU buffer loading
//!
//! Targets:
//! - Cached/incremental search: <5ms p50
//! - Cold first search: <200ms
//! - Index load: <10ms for ~100-500 files

use std::fs;

use criterion::{criterion_group, criterion_main, Criterion};
use tempfile::TempDir;

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::index::scanner::{FilesystemScanner, ScannerConfig};
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::SearchRequest;

// ============================================================================
// Test corpus generation
// ============================================================================

/// Create a temp directory with ~200 Rust-like source files for benchmarking.
///
/// Files contain realistic content with embedded "SEARCHME" patterns
/// to ensure the search pipeline has actual matches to process.
fn create_test_corpus() -> TempDir {
    let dir = TempDir::new().expect("Failed to create temp dir");

    // Create a few subdirectories for realism
    let subdirs = ["src", "src/utils", "src/models", "tests", "examples"];
    for sub in &subdirs {
        fs::create_dir_all(dir.path().join(sub)).unwrap();
    }

    // Generate ~200 source files with varied content
    let extensions = ["rs", "rs", "rs", "rs", "md", "toml", "txt"];

    for i in 0..200 {
        let ext = extensions[i % extensions.len()];
        let subdir_idx = i % subdirs.len();
        let filename = format!("file_{:04}.{}", i, ext);
        let path = dir.path().join(subdirs[subdir_idx]).join(&filename);

        // Generate realistic content (~500 bytes per file)
        let content = generate_file_content(i, ext);
        fs::write(&path, content).unwrap();
    }

    dir
}

/// Generate realistic file content with embedded search patterns.
fn generate_file_content(index: usize, ext: &str) -> String {
    match ext {
        "rs" => format!(
            r#"//! Module file_{index:04}
//!
//! Auto-generated for benchmarking.

use std::collections::HashMap;

/// Main struct for file_{index:04}.
pub struct Handler_{index} {{
    data: Vec<u8>,
    cache: HashMap<String, usize>,
}}

impl Handler_{index} {{
    pub fn new() -> Self {{
        Self {{
            data: Vec::new(),
            cache: HashMap::new(),
        }}
    }}

    /// Process input data -- SEARCHME marker for benchmark.
    pub fn process(&mut self, input: &[u8]) -> usize {{
        self.data.extend_from_slice(input);
        self.data.len()
    }}

    pub fn lookup(&self, key: &str) -> Option<usize> {{
        self.cache.get(key).copied()
    }}

    fn internal_helper(&self) -> bool {{
        !self.data.is_empty()
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_process() {{
        let mut h = Handler_{index}::new();
        assert_eq!(h.process(b"hello"), 5);
    }}

    #[test]
    fn test_lookup() {{
        let h = Handler_{index}::new();
        assert!(h.lookup("missing").is_none());
    }}
}}
"#,
        ),
        "md" => format!(
            "# File {index:04}\n\nDocumentation for module {index}.\n\n## Overview\n\nThis module provides SEARCHME functionality.\n\n## Usage\n\n```rust\nlet handler = Handler::new();\nhandler.process(data);\n```\n"
        ),
        "toml" => format!(
            "[package]\nname = \"file-{index:04}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n# SEARCHME: benchmark marker\nserde = \"1\"\n"
        ),
        _ => format!(
            "File {index:04}: generic text content.\nSEARCHME pattern for benchmark.\nLine 3 of content.\nLine 4 of content.\n"
        ),
    }
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_search_latency(c: &mut Criterion) {
    // Initialize GPU once -- shared across all benchmarks
    let gpu = GpuDevice::new();
    let pso_cache = PsoCache::new(&gpu.device);

    // Create the test corpus once
    let corpus = create_test_corpus();
    let corpus_path = corpus.path().to_path_buf();

    // ----------------------------------------------------------------
    // Benchmark: cached (warm) search latency
    // ----------------------------------------------------------------
    // This is the key metric: <5ms p50 for search-as-you-type.
    // The orchestrator + GPU engine are already initialized and warm.
    {
        let mut group = c.benchmark_group("search_latency");
        group.sample_size(10);

        // Pre-warm: create orchestrator and run one search to warm caches
        let mut orchestrator = SearchOrchestrator::new(&gpu.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let warmup_request = SearchRequest {
            pattern: "SEARCHME".to_string(),
            root: corpus_path.clone(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };
        let warmup_response = orchestrator.search(warmup_request.clone());
        println!(
            "Warmup: {} content matches, {} file matches, {:.2}ms",
            warmup_response.content_matches.len(),
            warmup_response.file_matches.len(),
            warmup_response.elapsed.as_secs_f64() * 1000.0
        );

        group.bench_function("cached_search", |b| {
            b.iter(|| {
                let request = SearchRequest {
                    pattern: "SEARCHME".to_string(),
                    root: corpus_path.clone(),
                    file_types: None,
                    case_sensitive: true,
                    respect_gitignore: false,
                    include_binary: false,
                    max_results: 10_000,
                };
                let response = orchestrator.search(request);
                response.content_matches.len()
            });
        });

        // ----------------------------------------------------------------
        // Benchmark: cached search with filetype filter
        // ----------------------------------------------------------------
        group.bench_function("cached_search_filtered", |b| {
            b.iter(|| {
                let request = SearchRequest {
                    pattern: "fn ".to_string(),
                    root: corpus_path.clone(),
                    file_types: Some(vec!["rs".to_string()]),
                    case_sensitive: true,
                    respect_gitignore: false,
                    include_binary: false,
                    max_results: 10_000,
                };
                let response = orchestrator.search(request);
                response.content_matches.len()
            });
        });

        // ----------------------------------------------------------------
        // Benchmark: cold search (new orchestrator each time)
        // ----------------------------------------------------------------
        group.bench_function("cold_search", |b| {
            b.iter(|| {
                let mut fresh_orchestrator =
                    SearchOrchestrator::new(&gpu.device, &pso_cache)
                        .expect("Failed to create orchestrator");

                let request = SearchRequest {
                    pattern: "SEARCHME".to_string(),
                    root: corpus_path.clone(),
                    file_types: None,
                    case_sensitive: true,
                    respect_gitignore: false,
                    include_binary: false,
                    max_results: 10_000,
                };
                let response = fresh_orchestrator.search(request);
                response.content_matches.len()
            });
        });

        group.finish();
    }

    // ----------------------------------------------------------------
    // Benchmark: index load (filesystem scan + index build)
    // ----------------------------------------------------------------
    {
        let mut group = c.benchmark_group("index_load");
        group.sample_size(10);

        group.bench_function("scan_200_files", |b| {
            let scanner = FilesystemScanner::with_config(ScannerConfig {
                respect_gitignore: false,
                skip_hidden: true,
                ..Default::default()
            });

            b.iter(|| {
                let entries = scanner.scan(&corpus_path);
                entries.len()
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_search_latency);
criterion_main!(benches);
