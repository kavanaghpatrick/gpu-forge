//! Benchmark: content store search vs disk-based search.
//!
//! Generates a 1000-file corpus (~5MB), then benchmarks:
//! - `disk_search`: existing streaming pipeline reading files from disk via fs::read
//! - `content_store_search`: content store fast-path (zero disk I/O, GPU from memory)
//!
//! Target: content_store_search latency < 10% of disk_search latency.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tempfile::TempDir;

use gpu_search::gpu::device::GpuDevice;
use gpu_search::gpu::pipeline::PsoCache;
use gpu_search::index::content_index_store::ContentIndexStore;
use gpu_search::index::content_snapshot::ContentSnapshot;
use gpu_search::index::content_store::ContentStoreBuilder;
use gpu_search::search::orchestrator::SearchOrchestrator;
use gpu_search::search::types::SearchRequest;

// ============================================================================
// Test corpus generation -- 1000 files, ~5KB each, ~5MB total
// ============================================================================

/// Create a temp directory with 1000 source files for benchmarking.
///
/// Files contain realistic Rust/Markdown/TOML content with embedded
/// "SEARCHME" patterns to ensure actual GPU matches.
fn create_bench_corpus() -> TempDir {
    let dir = TempDir::new().expect("Failed to create temp dir");

    let subdirs = [
        "src",
        "src/core",
        "src/utils",
        "src/models",
        "src/handlers",
        "tests",
        "tests/unit",
        "tests/integration",
        "examples",
        "docs",
    ];
    for sub in &subdirs {
        fs::create_dir_all(dir.path().join(sub)).unwrap();
    }

    let extensions = ["rs", "rs", "rs", "rs", "md", "toml", "txt", "rs", "rs", "rs"];

    for i in 0..1000 {
        let ext = extensions[i % extensions.len()];
        let subdir_idx = i % subdirs.len();
        let filename = format!("file_{:04}.{}", i, ext);
        let path = dir.path().join(subdirs[subdir_idx]).join(&filename);

        let content = generate_file_content(i, ext);
        fs::write(&path, content).unwrap();
    }

    dir
}

/// Generate ~5KB of realistic file content per file.
fn generate_file_content(index: usize, ext: &str) -> String {
    // Base content per type, then pad to ~5KB
    let base = match ext {
        "rs" => format!(
            r#"//! Module file_{index:04}
//!
//! Auto-generated benchmark corpus file.

use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for module file_{index:04}.
#[derive(Debug, Clone)]
pub struct Config_{index} {{
    pub name: String,
    pub value: usize,
    pub enabled: bool,
}}

/// Main handler for file_{index:04}.
pub struct Handler_{index} {{
    data: Vec<u8>,
    cache: HashMap<String, usize>,
    config: Config_{index},
}}

impl Handler_{index} {{
    pub fn new(config: Config_{index}) -> Self {{
        Self {{
            data: Vec::with_capacity(1024),
            cache: HashMap::new(),
            config,
        }}
    }}

    /// Process input data -- SEARCHME marker for benchmark.
    pub fn process(&mut self, input: &[u8]) -> Result<usize, String> {{
        if input.is_empty() {{
            return Err("empty input".to_string());
        }}
        self.data.extend_from_slice(input);
        self.cache.insert(
            format!("batch_{{}}", self.cache.len()),
            self.data.len(),
        );
        Ok(self.data.len())
    }}

    /// Lookup a cached value by key.
    pub fn lookup(&self, key: &str) -> Option<usize> {{
        self.cache.get(key).copied()
    }}

    /// Reset all internal state.
    pub fn reset(&mut self) {{
        self.data.clear();
        self.cache.clear();
    }}

    fn validate_input(&self, input: &[u8]) -> bool {{
        !input.is_empty() && input.len() < 1_000_000
    }}

    fn compute_hash(&self, data: &[u8]) -> u64 {{
        let mut hash = 0u64;
        for &byte in data {{
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }}
        hash
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_process() {{
        let config = Config_{index} {{
            name: "test".to_string(),
            value: 42,
            enabled: true,
        }};
        let mut h = Handler_{index}::new(config);
        assert_eq!(h.process(b"hello").unwrap(), 5);
    }}

    #[test]
    fn test_lookup() {{
        let config = Config_{index} {{
            name: "test".to_string(),
            value: 0,
            enabled: false,
        }};
        let h = Handler_{index}::new(config);
        assert!(h.lookup("missing").is_none());
    }}

    #[test]
    fn test_reset() {{
        let config = Config_{index} {{
            name: "bench".to_string(),
            value: 99,
            enabled: true,
        }};
        let mut h = Handler_{index}::new(config);
        h.process(b"data").unwrap();
        h.reset();
        assert!(h.lookup("batch_0").is_none());
    }}
}}
"#,
        ),
        "md" => format!(
            r#"# File {index:04}

Documentation for module {index}.

## Overview

This module provides SEARCHME functionality for the benchmark corpus.
It handles data processing, caching, and validation.

## Architecture

The module is organized into three layers:

1. **Input Layer**: Validates and normalizes incoming data
2. **Processing Layer**: Transforms data using configurable pipelines
3. **Output Layer**: Serializes results for downstream consumers

## Configuration

```toml
[module.file_{index:04}]
enabled = true
max_batch_size = 1024
timeout_ms = 5000
```

## Usage

```rust
use crate::file_{index:04}::Handler_{index};

let handler = Handler_{index}::new(config);
handler.process(data)?;
```

## Performance Notes

- Average throughput: 100MB/s
- P99 latency: <5ms
- Memory overhead: ~2KB per instance

## Changelog

- v0.3.0: Added batch processing support
- v0.2.0: Improved error handling
- v0.1.0: Initial implementation
"#,
        ),
        "toml" => format!(
            r#"[package]
name = "file-{index:04}"
version = "0.1.0"
edition = "2021"
description = "Benchmark corpus file {index}"

[dependencies]
# SEARCHME: benchmark marker
serde = {{ version = "1", features = ["derive"] }}
tokio = {{ version = "1", features = ["full"] }}
tracing = "0.1"

[dev-dependencies]
criterion = "0.5"
tempfile = "3"

[[bench]]
name = "bench_{index:04}"
harness = false

[features]
default = []
experimental = []
"#,
        ),
        _ => format!(
            r#"File {index:04}: generic text content for benchmark corpus.
SEARCHME pattern for GPU search benchmark.

This file contains various text patterns that exercise the search pipeline.
Line 5 of content with some keywords: function, struct, module.
Line 6: additional filler text for realistic file sizes.
Line 7: more content to bulk up the file to target size.

Section: Data Processing
The handler processes input data in configurable batches.
Each batch is validated, transformed, and cached for retrieval.

Section: Error Handling
Errors are categorized into recoverable and fatal classes.
Recoverable errors trigger automatic retry with backoff.

Section: Metrics
Performance metrics are collected at each pipeline stage.
Throughput and latency histograms enable SLO monitoring.
"#,
        ),
    };

    // Pad to ~5KB by repeating filler lines
    let mut content = base;
    let filler = format!(
        "// Filler line {index} -- padding to reach target file size for benchmark corpus.\n"
    );
    while content.len() < 5000 {
        content.push_str(&filler);
    }
    content
}

// ============================================================================
// Helper: build a ContentIndexStore from the corpus on disk
// ============================================================================

fn build_content_store(corpus_path: &std::path::Path, device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>) -> Arc<ContentIndexStore> {
    // Walk corpus and load files into ContentStoreBuilder
    let mut total_bytes = 0usize;
    let mut file_entries: Vec<(PathBuf, Vec<u8>)> = Vec::new();

    fn walk_dir(dir: &std::path::Path, entries: &mut Vec<(PathBuf, Vec<u8>)>) {
        if let Ok(rd) = fs::read_dir(dir) {
            for entry in rd.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk_dir(&path, entries);
                } else if path.is_file() {
                    if let Ok(content) = fs::read(&path) {
                        entries.push((path, content));
                    }
                }
            }
        }
    }

    walk_dir(corpus_path, &mut file_entries);
    for (_, content) in &file_entries {
        total_bytes += content.len();
    }

    // Allocate builder with total capacity
    let mut builder = ContentStoreBuilder::new(total_bytes + 4096)
        .expect("Failed to create ContentStoreBuilder");

    for (i, (path, content)) in file_entries.iter().enumerate() {
        let hash = crc32fast::hash(content);
        let mtime = path
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;
        builder.append_with_path(content, path.clone(), i as u32, hash, mtime);
    }

    let store = builder.finalize(device);
    let snapshot = ContentSnapshot::new(store, 0);
    let index_store = Arc::new(ContentIndexStore::new());
    index_store.swap(snapshot);
    index_store
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_content_search(c: &mut Criterion) {
    // Initialize GPU and corpus once
    let gpu = GpuDevice::new();
    let pso_cache = PsoCache::new(&gpu.device);
    let corpus = create_bench_corpus();
    let corpus_path = corpus.path().to_path_buf();

    // Pre-build content store
    let content_store = build_content_store(&corpus_path, &gpu.device);

    // Print corpus stats
    {
        let guard = content_store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        let cs = snap.content_store();
        eprintln!(
            "\n=== Benchmark corpus: {} files, {:.2} MB total ===\n",
            cs.file_count(),
            cs.total_bytes() as f64 / (1024.0 * 1024.0)
        );
    }

    let mut group = c.benchmark_group("content_search");
    group.sample_size(10);

    // ------------------------------------------------------------------
    // Benchmark 1: Disk-based search (streaming pipeline with fs::read)
    // ------------------------------------------------------------------
    group.bench_function(BenchmarkId::new("disk_search", "SEARCHME"), |b| {
        let mut orchestrator = SearchOrchestrator::new(&gpu.device, &pso_cache)
            .expect("Failed to create orchestrator (disk)");

        // Warmup
        let warmup_req = SearchRequest {
            pattern: "SEARCHME".to_string(),
            root: corpus_path.clone(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };
        let _ = orchestrator.search(warmup_req);

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

    // ------------------------------------------------------------------
    // Benchmark 2: Content store search (zero disk I/O, GPU from memory)
    // ------------------------------------------------------------------
    group.bench_function(
        BenchmarkId::new("content_store_search", "SEARCHME"),
        |b| {
            let mut orchestrator = SearchOrchestrator::with_content_store(
                &gpu.device,
                &pso_cache,
                None,
                content_store.clone(),
            )
            .expect("Failed to create orchestrator (content store)");

            // Warmup
            let warmup_req = SearchRequest {
                pattern: "SEARCHME".to_string(),
                root: corpus_path.clone(),
                file_types: None,
                case_sensitive: true,
                respect_gitignore: false,
                include_binary: false,
                max_results: 10_000,
            };
            let _ = orchestrator.search(warmup_req);

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
        },
    );

    group.finish();
}

criterion_group!(benches, bench_content_search);
criterion_main!(benches);
