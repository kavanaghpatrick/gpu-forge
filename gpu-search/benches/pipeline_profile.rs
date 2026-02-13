//! Pipeline profiler benchmarks with Criterion baseline support.
//!
//! Three benchmarks covering the full search pipeline:
//! 1. **project_search**: Search real gpu-search/src/ for "fn " (realistic project search)
//! 2. **index_load**: Load GSIX index via MmapIndexCache (zero-copy mmap)
//! 3. **full_pipeline**: End-to-end orchestrator search on test corpus
//!
//! ## Baseline management
//!
//! Save initial baseline:
//!   cargo bench -p gpu-search --bench pipeline_profile -- --save-baseline initial
//!
//! Compare against baseline:
//!   cargo bench -p gpu-search --bench pipeline_profile -- --baseline initial

use std::fs;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};
use tempfile::TempDir;

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::index::cache::MmapIndexCache;
use gpu_search::index::gpu_index::GpuResidentIndex;
use gpu_search::index::shared_index::SharedIndexManager;
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::SearchRequest;

// ============================================================================
// Helpers
// ============================================================================

/// Locate gpu-search/src/ directory for real-project benchmarks.
fn gpu_search_src_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("src")
}

/// Create a test corpus with ~100 Rust-like files for end-to-end benchmarks.
fn create_bench_corpus() -> TempDir {
    let dir = TempDir::new().expect("create temp dir");
    let subdirs = ["src", "src/core", "src/util", "tests"];
    for sub in &subdirs {
        fs::create_dir_all(dir.path().join(sub)).unwrap();
    }

    for i in 0..100 {
        let subdir = subdirs[i % subdirs.len()];
        let path = dir.path().join(subdir).join(format!("mod_{:04}.rs", i));
        let content = format!(
            r#"//! Module mod_{i:04}

use std::collections::HashMap;

pub struct Widget_{i} {{
    data: Vec<u8>,
    index: HashMap<String, usize>,
}}

impl Widget_{i} {{
    pub fn new() -> Self {{
        Self {{ data: Vec::new(), index: HashMap::new() }}
    }}

    /// Process input -- fn marker for search.
    pub fn process(&mut self, input: &[u8]) -> usize {{
        self.data.extend_from_slice(input);
        self.data.len()
    }}

    fn internal_helper(&self) -> bool {{
        !self.data.is_empty()
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_new() {{
        let w = Widget_{i}::new();
        assert!(w.data.is_empty());
    }}
}}
"#,
        );
        fs::write(&path, content).unwrap();
    }

    dir
}

/// Build a GSIX index for the given root, return the index file path.
fn build_index_for(root: &std::path::Path) -> PathBuf {
    let manager = SharedIndexManager::new().expect("create SharedIndexManager");
    let index = GpuResidentIndex::build_from_directory(root).expect("build index");
    manager
        .save(&index, root)
        .expect("save index");
    manager.index_path(root)
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_pipeline_profile(c: &mut Criterion) {
    let gpu = GpuDevice::new();
    let pso_cache = PsoCache::new(&gpu.device);

    // ------------------------------------------------------------------
    // 1. bench_project_search: search gpu-search/src/ for "fn "
    // ------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("pipeline_project_search");
        group.sample_size(10);

        let src_dir = gpu_search_src_dir();
        let mut orchestrator = SearchOrchestrator::new(&gpu.device, &pso_cache)
            .expect("create orchestrator");

        // Warmup
        let warmup_req = SearchRequest {
            pattern: "fn ".to_string(),
            root: src_dir.clone(),
            file_types: Some(vec!["rs".to_string()]),
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };
        let warmup = orchestrator.search(warmup_req);
        println!(
            "project_search warmup: {} matches in {:.2}ms",
            warmup.content_matches.len(),
            warmup.elapsed.as_secs_f64() * 1000.0
        );

        group.bench_function("search_fn_in_src", |b| {
            b.iter(|| {
                let request = SearchRequest {
                    pattern: "fn ".to_string(),
                    root: src_dir.clone(),
                    file_types: Some(vec!["rs".to_string()]),
                    case_sensitive: true,
                    respect_gitignore: false,
                    include_binary: false,
                    max_results: 10_000,
                };
                let resp = orchestrator.search(request);
                resp.content_matches.len()
            });
        });

        group.finish();
    }

    // ------------------------------------------------------------------
    // 2. bench_index_load: load GSIX index via MmapIndexCache
    // ------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("pipeline_index_load");
        group.sample_size(10);

        let corpus = create_bench_corpus();
        let idx_path = build_index_for(corpus.path());

        // Warmup mmap
        let _ = MmapIndexCache::load_mmap(&idx_path, None);

        group.bench_function("mmap_load_100_files", |b| {
            b.iter(|| {
                let cache = MmapIndexCache::load_mmap(&idx_path, None)
                    .expect("load index");
                cache.entry_count()
            });
        });

        group.finish();
    }

    // ------------------------------------------------------------------
    // 3. bench_full_pipeline: end-to-end orchestrator search
    // ------------------------------------------------------------------
    {
        let mut group = c.benchmark_group("pipeline_full");
        group.sample_size(10);

        let corpus = create_bench_corpus();
        let corpus_path = corpus.path().to_path_buf();

        let mut orchestrator = SearchOrchestrator::new(&gpu.device, &pso_cache)
            .expect("create orchestrator");

        // Warmup
        let warmup_req = SearchRequest {
            pattern: "fn ".to_string(),
            root: corpus_path.clone(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };
        let warmup = orchestrator.search(warmup_req);
        println!(
            "full_pipeline warmup: {} matches in {:.2}ms",
            warmup.content_matches.len(),
            warmup.elapsed.as_secs_f64() * 1000.0
        );

        group.bench_function("orchestrator_e2e", |b| {
            b.iter(|| {
                let request = SearchRequest {
                    pattern: "fn ".to_string(),
                    root: corpus_path.clone(),
                    file_types: None,
                    case_sensitive: true,
                    respect_gitignore: false,
                    include_binary: false,
                    max_results: 10_000,
                };
                let resp = orchestrator.search(request);
                resp.content_matches.len()
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_pipeline_profile);
criterion_main!(benches);
