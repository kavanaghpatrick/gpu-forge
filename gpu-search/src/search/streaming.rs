//! Quad-buffered streaming search pipeline.
//!
//! Ported from rust-experiment/src/gpu_os/streaming_search.rs (1071 lines).
//!
//! THE GPU IS THE COMPUTER. Don't wait for ALL files to load before searching.
//!
//! Traditional:  Load ALL files (283ms) -> THEN search (50ms) = 333ms total
//! Streaming:    Load chunk 1 -> [Load 2 + Search 1] -> [Load 3 + Search 2] -> ...
//!
//! Key insight: GPU can search one chunk while the next batch of files loads.
//! This overlaps I/O and compute for ~30%+ speedup on I/O-bound workloads.
//!
//! ## Architecture
//!
//! 4 StreamChunks (quad-buffering), each holding up to `CHUNK_BYTES` of file data.
//! Files are partitioned into chunks by cumulative size. The pipeline:
//! 1. Load chunk 0 files (batch I/O)
//! 2. While searching chunk N, load chunk N+1 files
//! 3. Collect results from each chunk, merge and deduplicate
//!
//! ## POC Simplification
//!
//! This POC uses CPU file reads (std::fs::read) into ContentSearchEngine buffers
//! rather than true MTLIOCommandQueue overlap. The streaming partition logic and
//! multi-buffer architecture are in place for Phase 2 GPU I/O integration.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::gpu::pipeline::PsoCache;
use super::content::{ContentSearchEngine, SearchOptions};

// ============================================================================
// Constants
// ============================================================================

/// Default number of stream chunks (quad-buffering).
const DEFAULT_CHUNK_COUNT: usize = 4;

/// Maximum files per chunk (prevents memory bloat).
const MAX_FILES_PER_CHUNK: usize = 5000;

/// Default chunk size in bytes (64 MB).
const DEFAULT_CHUNK_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum single file size to include (skip files > 100MB).
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Maximum files per GPU dispatch sub-batch.
/// Keeps match count well under MAX_MATCHES (10000) per dispatch.
const SUB_BATCH_SIZE: usize = 200;

// ============================================================================
// StreamChunk
// ============================================================================

/// A stream chunk: a partition of files with metadata for pipeline tracking.
///
/// Each chunk holds references to files that fit within `CHUNK_BYTES`.
/// The `ready` flag signals when I/O is complete and search can begin.
pub struct StreamChunk {
    /// File paths assigned to this chunk.
    pub file_paths: Vec<PathBuf>,
    /// File sizes (actual, not aligned).
    pub file_sizes: Vec<u64>,
    /// Total bytes in this chunk (sum of file sizes).
    pub total_bytes: u64,
    /// Number of files in this chunk.
    pub file_count: usize,
    /// Set to true when I/O is complete and data is loaded.
    pub ready: Arc<AtomicBool>,
}

impl StreamChunk {
    /// Create a new empty chunk.
    fn new() -> Self {
        Self {
            file_paths: Vec::with_capacity(MAX_FILES_PER_CHUNK),
            file_sizes: Vec::with_capacity(MAX_FILES_PER_CHUNK),
            total_bytes: 0,
            file_count: 0,
            ready: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Reset chunk for reuse.
    fn reset(&mut self) {
        self.file_paths.clear();
        self.file_sizes.clear();
        self.total_bytes = 0;
        self.file_count = 0;
        self.ready.store(false, Ordering::Release);
    }

    /// Add a file to this chunk.
    fn add_file(&mut self, path: PathBuf, size: u64) {
        self.file_paths.push(path);
        self.file_sizes.push(size);
        self.total_bytes += size;
        self.file_count += 1;
    }

    /// Check if adding a file would exceed chunk limits.
    fn would_exceed(&self, size: u64, max_bytes: u64) -> bool {
        self.total_bytes + size > max_bytes || self.file_count + 1 > MAX_FILES_PER_CHUNK
    }

    /// Mark chunk as ready (I/O complete).
    fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    /// Check if chunk is ready for search.
    #[allow(dead_code)]
    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

// ============================================================================
// StreamingProfile
// ============================================================================

/// Profiling data for streaming search operations.
#[derive(Debug, Clone, Default)]
pub struct StreamingProfile {
    /// Time spent partitioning files (microseconds).
    pub partition_us: u64,
    /// Time spent loading file data (I/O).
    pub io_us: u64,
    /// Time spent in GPU search.
    pub search_us: u64,
    /// Total elapsed time.
    pub total_us: u64,
    /// Number of chunks processed.
    pub chunk_count: usize,
    /// Number of files processed.
    pub files_processed: usize,
    /// Total bytes searched.
    pub bytes_processed: u64,
    /// Number of matches found.
    pub match_count: usize,
}

impl StreamingProfile {
    /// Print formatted profile summary.
    pub fn print(&self) {
        println!("Streaming Search Profile:");
        println!("  Partition:  {:>8}us", self.partition_us);
        println!("  I/O:        {:>8}us", self.io_us);
        println!("  GPU Search: {:>8}us", self.search_us);
        println!(
            "  Total:      {:>8}us ({:.1}ms)",
            self.total_us,
            self.total_us as f64 / 1000.0
        );
        println!("  Chunks:     {}", self.chunk_count);
        println!("  Files:      {}", self.files_processed);
        println!(
            "  Data:       {:.2} MB",
            self.bytes_processed as f64 / (1024.0 * 1024.0)
        );
        println!("  Matches:    {}", self.match_count);

        if self.total_us > 0 {
            let throughput = (self.bytes_processed as f64 / (1024.0 * 1024.0))
                / (self.total_us as f64 / 1_000_000.0);
            println!("  Throughput: {:.1} MB/s", throughput);
        }
    }
}

// ============================================================================
// StreamingSearchEngine
// ============================================================================

/// Streaming search engine with quad-buffered pipeline.
///
/// Manages multiple StreamChunks and a ContentSearchEngine to process
/// large file sets in overlapped I/O + compute batches. Files are
/// partitioned into chunks by cumulative size, then each chunk is
/// loaded and searched sequentially (with overlap in Phase 2).
///
/// ## Usage
///
/// ```ignore
/// let mut engine = StreamingSearchEngine::new(&device, &pso_cache)?;
/// let results = engine.search_files(&files, b"pattern", &options);
/// ```
pub struct StreamingSearchEngine {
    /// The underlying content search engine (reused per chunk).
    search_engine: ContentSearchEngine,
    /// Pre-allocated chunk slots for file partitioning.
    chunks: Vec<StreamChunk>,
    /// Maximum bytes per chunk.
    chunk_bytes: u64,
}

impl StreamingSearchEngine {
    /// Create a new streaming search engine with default quad-buffer config.
    ///
    /// Returns `None` if GPU initialization fails.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        pso_cache: &PsoCache,
    ) -> Option<Self> {
        Self::with_config(device, pso_cache, DEFAULT_CHUNK_COUNT, DEFAULT_CHUNK_BYTES)
    }

    /// Create with custom configuration.
    pub fn with_config(
        device: &ProtocolObject<dyn MTLDevice>,
        pso_cache: &PsoCache,
        chunk_count: usize,
        chunk_bytes: u64,
    ) -> Option<Self> {
        // ContentSearchEngine with generous max_files -- will be reset per chunk
        let search_engine = ContentSearchEngine::new(device, pso_cache, MAX_FILES_PER_CHUNK);

        let mut chunks = Vec::with_capacity(chunk_count);
        for _ in 0..chunk_count {
            chunks.push(StreamChunk::new());
        }

        Some(Self {
            search_engine,
            chunks,
            chunk_bytes,
        })
    }

    /// Get number of stream chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get mutable access to the inner ContentSearchEngine.
    ///
    /// Used by the content store fast-path in SearchOrchestrator to call
    /// `search_with_buffer()` directly against in-memory content.
    pub fn content_engine_mut(&mut self) -> &mut ContentSearchEngine {
        &mut self.search_engine
    }

    /// Reset all chunks for a new search.
    fn reset_chunks(&mut self) {
        for chunk in &mut self.chunks {
            chunk.reset();
        }
    }

    /// Partition files into chunks based on cumulative size.
    ///
    /// Each chunk holds files up to `chunk_bytes` total or `MAX_FILES_PER_CHUNK`.
    /// Returns the number of chunks populated.
    fn partition_files(&mut self, files: &[PathBuf]) -> usize {
        self.reset_chunks();

        if files.is_empty() {
            return 0;
        }

        let mut chunk_idx = 0;

        for path in files {
            // Get file size, skip invalid
            let size = match std::fs::metadata(path) {
                Ok(m) => m.len(),
                Err(_) => continue,
            };

            // Skip empty or oversized files
            if size == 0 || size > MAX_FILE_SIZE {
                continue;
            }

            // Check if current chunk would overflow
            if chunk_idx < self.chunks.len()
                && self.chunks[chunk_idx].would_exceed(size, self.chunk_bytes)
            {
                // Move to next chunk
                chunk_idx += 1;
            }

            // If we've exhausted all chunks, pack remaining into last chunk
            if chunk_idx >= self.chunks.len() {
                chunk_idx = self.chunks.len() - 1;
            }

            self.chunks[chunk_idx].add_file(path.clone(), size);
        }

        // Count non-empty chunks
        self.chunks.iter().filter(|c| c.file_count > 0).count()
    }

    /// Search files using the streaming pipeline.
    ///
    /// Partitions files into chunks, then for each chunk:
    /// 1. Reads file data (CPU I/O for POC)
    /// 2. Loads into ContentSearchEngine
    /// 3. Dispatches GPU search
    /// 4. Collects results with file path annotation
    ///
    /// Returns matches with `file_index` mapped to actual file indices,
    /// plus a profiling summary.
    pub fn search_files(
        &mut self,
        files: &[PathBuf],
        pattern: &[u8],
        options: &SearchOptions,
    ) -> Vec<StreamingMatch> {
        let (results, _profile) = self.search_files_with_profile(files, pattern, options);
        results
    }

    /// Search with detailed profiling.
    pub fn search_files_with_profile(
        &mut self,
        files: &[PathBuf],
        pattern: &[u8],
        options: &SearchOptions,
    ) -> (Vec<StreamingMatch>, StreamingProfile) {
        let total_start = Instant::now();
        let mut profile = StreamingProfile::default();

        if files.is_empty() || pattern.is_empty() || pattern.len() > 64 {
            return (vec![], profile);
        }

        // Phase 1: Partition files into chunks
        let partition_start = Instant::now();
        let active_chunks = self.partition_files(files);
        profile.partition_us = partition_start.elapsed().as_micros() as u64;
        profile.chunk_count = active_chunks;

        if active_chunks == 0 {
            return (vec![], profile);
        }

        let mut all_results: Vec<StreamingMatch> = Vec::new();

        // Phase 2: Process each chunk (sequential for POC, overlapped in Phase 2)
        for chunk_idx in 0..active_chunks {
            let chunk = &self.chunks[chunk_idx];
            if chunk.file_count == 0 {
                continue;
            }

            // Build list of (path, content) for this chunk
            let io_start = Instant::now();
            let mut file_contents: Vec<(usize, Vec<u8>)> = Vec::with_capacity(chunk.file_count);

            for (local_idx, path) in chunk.file_paths.iter().enumerate() {
                match std::fs::read(path) {
                    Ok(data) => {
                        if !data.is_empty() {
                            file_contents.push((local_idx, data));
                        }
                    }
                    Err(_) => continue,
                }
            }
            profile.io_us += io_start.elapsed().as_micros() as u64;

            if file_contents.is_empty() {
                continue;
            }

            // Process files in sub-batches to stay within GPU match buffer limits.
            // Each dispatch has MAX_MATCHES=10000 slots; sub-batching prevents overflow.
            let search_start = Instant::now();

            for sub_batch in file_contents.chunks(SUB_BATCH_SIZE) {
                // Wrap each sub-batch in autoreleasepool to drain Metal objects
                // (command buffers, encoders) created during GPU dispatch.
                autoreleasepool(|_| {
                    self.search_engine.reset();

                    // Track mapping: engine_file_index -> (chunk_local_idx, start_chunk_offset)
                    // start_chunk_offset = engine chunk index where this file's data begins,
                    // used to convert engine-global byte_offset to file-relative byte_offset.
                    let mut loaded_files: Vec<(u32, usize, usize)> = Vec::new();

                    for (local_idx, content) in sub_batch {
                        let file_index = loaded_files.len() as u32;
                        let start_chunk = self.search_engine.chunk_count();
                        let chunks_loaded = self.search_engine.load_content(content, file_index);
                        if chunks_loaded > 0 {
                            loaded_files.push((file_index, *local_idx, start_chunk));
                            profile.bytes_processed += content.len() as u64;
                            profile.files_processed += 1;
                        }
                    }

                    if loaded_files.is_empty() {
                        return;
                    }

                    // Dispatch GPU search for this sub-batch
                    let gpu_results = self.search_engine.search(pattern, options);

                    // Map results back to file paths
                    for m in &gpu_results {
                        let engine_file_idx = m.file_index;

                        let file_info = loaded_files
                            .iter()
                            .find(|(fi, _, _)| *fi as usize == engine_file_idx);

                        if let Some((_, local_idx, start_chunk)) = file_info {
                            if let Some(path) = chunk.file_paths.get(*local_idx) {
                                // Convert engine-global byte_offset to file-relative:
                                // byte_offset from GPU = global_chunk_index * 4096 + context_start
                                // file-relative = byte_offset - (start_chunk * 4096)
                                let file_byte_offset = m.byte_offset
                                    .saturating_sub((*start_chunk as u32) * 4096);

                                // Validate: engine byte_offset should be >= start_chunk * 4096.
                                // If saturating_sub would have underflowed, the byte_offset chain is broken.
                                debug_assert!(
                                    m.byte_offset >= (*start_chunk as u32) * 4096,
                                    "byte_offset {} < start_chunk offset {} (start_chunk={}): file_byte_offset would underflow",
                                    m.byte_offset, (*start_chunk as u32) * 4096, start_chunk,
                                );

                                all_results.push(StreamingMatch {
                                    file_path: path.clone(),
                                    line_number: m.line_number,
                                    column: m.column,
                                    byte_offset: file_byte_offset,
                                    match_length: m.match_length,
                                });
                            }
                        }
                    }
                });
            }
            profile.search_us += search_start.elapsed().as_micros() as u64;

            // Mark chunk complete
            chunk.mark_ready();
        }

        profile.total_us = total_start.elapsed().as_micros() as u64;
        profile.match_count = all_results.len();

        // Sort by file path then line number
        all_results.sort_by(|a, b| {
            a.file_path
                .cmp(&b.file_path)
                .then(a.line_number.cmp(&b.line_number))
                .then(a.column.cmp(&b.column))
        });

        (all_results, profile)
    }
}

// ============================================================================
// StreamingMatch -- result type with file path
// ============================================================================

/// A match from streaming search, annotated with the full file path.
#[derive(Debug, Clone)]
pub struct StreamingMatch {
    /// Full path to the file containing this match.
    pub file_path: PathBuf,
    /// Line number (1-based, may be local to GPU thread window).
    pub line_number: u32,
    /// Column (byte offset within the line/chunk).
    pub column: u32,
    /// Byte offset in the file.
    pub byte_offset: u32,
    /// Length of the matched pattern.
    pub match_length: u32,
}

// ============================================================================
// CPU reference for streaming verification
// ============================================================================

/// CPU reference: count pattern occurrences across multiple files.
///
/// Used to verify streaming GPU search results.
pub fn cpu_streaming_search(files: &[PathBuf], pattern: &[u8], case_sensitive: bool) -> usize {
    let mut total = 0;
    for path in files {
        if let Ok(content) = std::fs::read(path) {
            total += super::content::cpu_search(&content, pattern, case_sensitive);
        }
    }
    total
}

// ============================================================================
// Hot/Cold Dispatch Logic
// ============================================================================

/// Threshold below which files are always dispatched to CPU (4 KB).
const COLD_THRESHOLD_BYTES: u64 = 4 * 1024;

/// Threshold above which files are always dispatched to GPU (128 KB).
const HOT_THRESHOLD_BYTES: u64 = 128 * 1024;

/// Determine whether a file should be searched on the GPU or fall back to CPU.
///
/// Decision logic:
/// - Files > 128 KB: always GPU (large sequential reads amortize dispatch overhead)
/// - Files < 4 KB: always CPU (memchr is faster than GPU dispatch latency)
/// - Middle range (4 KB - 128 KB): use `page_cache_hint` -- if the file is likely
///   hot in the page cache, CPU memchr wins; otherwise GPU batch amortizes I/O.
///
/// This is a stub for future page-cache detection. The `page_cache_hint` parameter
/// will eventually be replaced by actual mincore(2) / fs_usage probing.
///
/// TODO(KB #1377): Replace page_cache_hint with real hot/cold detection using
/// mincore(2) for resident-page ratio and adaptive threshold learning from
/// per-file search latency history. See also KB #1290 for CPU/GPU crossover analysis.
pub fn should_use_gpu(file_size: u64, page_cache_hint: bool) -> bool {
    if file_size >= HOT_THRESHOLD_BYTES {
        return true;
    }
    if file_size < COLD_THRESHOLD_BYTES {
        return false;
    }
    // Middle range: if page cache hint says hot, CPU is faster; otherwise GPU.
    !page_cache_hint
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use tempfile::TempDir;

    /// Create a temp directory with test files, each containing known patterns.
    ///
    /// Returns (dir, file_paths, expected_total_matches_for_pattern).
    fn make_streaming_test_files(
        count: usize,
        content_per_file: &str,
    ) -> (TempDir, Vec<PathBuf>) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let mut paths = Vec::with_capacity(count);

        for i in 0..count {
            let path = dir.path().join(format!("stream_file_{:05}.txt", i));
            // Each file gets the base content plus a unique identifier
            let content = format!(
                "=== File {} ===\n{}\n--- end file {} ---\n",
                i, content_per_file, i
            );
            std::fs::write(&path, content.as_bytes()).expect("Failed to write test file");
            paths.push(path);
        }

        (dir, paths)
    }

    /// Create test files totaling more than the given size in bytes.
    ///
    /// Uses larger files (~64KB each) with sparse pattern placement to minimize
    /// GPU 64-byte thread boundary misses. Patterns are placed at line starts
    /// (not crossing boundaries) with long filler lines between them.
    fn make_large_test_data(target_bytes: u64, pattern: &str) -> (TempDir, Vec<PathBuf>) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let mut paths = Vec::new();
        let mut total_bytes = 0u64;

        // Larger files reduce per-file overhead and boundary miss ratio
        let file_size = 65536; // 64KB per file
        let mut file_idx = 0u32;

        // Long filler line (~120 chars) ensures patterns don't land near
        // 64-byte boundaries too often
        let filler_base = "abcdefghijklmnopqrstuvwxyz_0123456789_ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz_0123456789_FILLER";

        while total_bytes < target_bytes {
            let path = dir.path().join(format!("data_{:06}.txt", file_idx));

            let mut content = String::with_capacity(file_size + 256);
            let mut line = 0u32;
            while content.len() < file_size {
                if line % 40 == 7 {
                    // Pattern line: start of line, well-separated from boundaries
                    content.push_str(pattern);
                    content.push_str(" found in file ");
                    content.push_str(&file_idx.to_string());
                    content.push_str(" at line ");
                    content.push_str(&line.to_string());
                    content.push('\n');
                } else {
                    // Long filler line
                    content.push_str(filler_base);
                    content.push('_');
                    content.push_str(&line.to_string());
                    content.push('\n');
                }
                line += 1;
            }

            let bytes = content.as_bytes();
            std::fs::write(&path, bytes).expect("write test file");
            total_bytes += bytes.len() as u64;
            paths.push(path);
            file_idx += 1;
        }

        (dir, paths)
    }

    #[test]
    fn test_streaming_search() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);

        let mut engine = StreamingSearchEngine::new(&device, &pso_cache)
            .expect("Failed to create streaming engine");

        // Create >64MB of test data with known pattern
        let search_pattern = "SEARCH_TARGET";
        let target_bytes = 68 * 1024 * 1024; // 68MB > 64MB threshold
        let (dir, paths) = make_large_test_data(target_bytes, search_pattern);

        let total_file_bytes: u64 = paths
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum();
        println!(
            "Created {} files totaling {:.1} MB",
            paths.len(),
            total_file_bytes as f64 / (1024.0 * 1024.0)
        );
        assert!(
            total_file_bytes > 64 * 1024 * 1024,
            "Test data must exceed 64MB, got {} bytes",
            total_file_bytes
        );

        // GPU streaming search
        let options = SearchOptions {
            case_sensitive: true,
            ..Default::default()
        };
        let (results, profile) =
            engine.search_files_with_profile(&paths, search_pattern.as_bytes(), &options);

        profile.print();

        // CPU reference count
        let cpu_count = cpu_streaming_search(&paths, search_pattern.as_bytes(), true);

        println!(
            "GPU streaming matches: {}, CPU reference: {}",
            results.len(),
            cpu_count
        );

        // GPU count may be slightly less due to 64-byte thread boundary misses.
        // The content_search_kernel processes 64 bytes per thread -- matches spanning
        // a boundary are missed. For a 13-byte pattern, up to ~20% of matches may be
        // at boundaries. This is a known GPU kernel limitation documented in content.rs.
        assert!(
            results.len() <= cpu_count,
            "GPU should not produce false positives: GPU({}) > CPU({})",
            results.len(),
            cpu_count
        );

        // Allow up to ~20% boundary misses (known GPU kernel thread-boundary limitation)
        let min_expected = cpu_count * 80 / 100;
        assert!(
            results.len() >= min_expected,
            "Too many misses: GPU({}) < 80% of CPU({})",
            results.len(),
            min_expected
        );

        // Verify results have valid file paths
        for m in &results {
            assert!(
                m.file_path.exists(),
                "Match file path should exist: {:?}",
                m.file_path
            );
            assert!(m.match_length > 0, "Match length should be positive");
        }

        // Verify >64MB was actually processed
        assert!(
            profile.bytes_processed > 64 * 1024 * 1024,
            "Must process >64MB, got {} bytes",
            profile.bytes_processed
        );
        assert!(
            profile.chunk_count > 0,
            "Should have processed at least 1 chunk"
        );
        assert!(
            profile.files_processed > 0,
            "Should have processed files"
        );

        println!("Streaming search PASSED: {} matches across {} chunks, {:.1} MB processed",
            results.len(), profile.chunk_count,
            profile.bytes_processed as f64 / (1024.0 * 1024.0));

        drop(dir);
    }

    #[test]
    fn test_streaming_search_small_files() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);

        let mut engine = StreamingSearchEngine::new(&device, &pso_cache)
            .expect("Failed to create streaming engine");

        // Create 50 small files
        let content = "fn hello_world() {\n    println!(\"hello\");\n}\n";
        let (_dir, paths) = make_streaming_test_files(50, content);

        let options = SearchOptions::default();
        let results = engine.search_files(&paths, b"hello", &options);
        let cpu_count = cpu_streaming_search(&paths, b"hello", true);

        println!(
            "Small files: GPU={}, CPU={} matches for 'hello' in {} files",
            results.len(),
            cpu_count,
            paths.len()
        );

        // Each file has "hello_world" and "hello" in println = 2 matches per file
        // GPU should find at least 90% of CPU count
        assert!(results.len() > 0, "Should find matches");
        assert!(
            results.len() <= cpu_count,
            "No false positives: GPU({}) > CPU({})",
            results.len(),
            cpu_count
        );
        let min_expected = cpu_count * 90 / 100;
        assert!(
            results.len() >= min_expected,
            "Too many misses: GPU({}) < 90% of CPU({})",
            results.len(),
            min_expected
        );
    }

    #[test]
    fn test_streaming_search_no_matches() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);

        let mut engine = StreamingSearchEngine::new(&device, &pso_cache)
            .expect("Failed to create streaming engine");

        let content = "this content has nothing special\n";
        let (_dir, paths) = make_streaming_test_files(10, content);

        let options = SearchOptions::default();
        let results = engine.search_files(&paths, b"ZZZZZ_NOT_FOUND", &options);

        assert_eq!(results.len(), 0, "Should find zero matches for absent pattern");
    }

    #[test]
    fn test_streaming_partition() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);

        // Use small chunk size to force multiple partitions
        let mut engine = StreamingSearchEngine::with_config(
            &device,
            &pso_cache,
            4,       // 4 chunks
            8 * 1024, // 8KB per chunk (tiny, for testing)
        )
        .expect("Failed to create streaming engine");

        // Create files that exceed 8KB per chunk
        let content = "x".repeat(2048); // 2KB per file
        let (_dir, paths) = make_streaming_test_files(20, &content);

        let active = engine.partition_files(&paths);

        println!("Partitioned {} files into {} chunks (max 4)", paths.len(), active);
        for (i, chunk) in engine.chunks.iter().enumerate() {
            if chunk.file_count > 0 {
                println!(
                    "  Chunk {}: {} files, {} bytes",
                    i, chunk.file_count, chunk.total_bytes
                );
            }
        }

        assert!(active > 1, "Should use multiple chunks with 8KB limit");
        assert!(active <= 4, "Should not exceed 4 chunks");
    }

    #[test]
    fn test_streaming_empty_input() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);

        let mut engine = StreamingSearchEngine::new(&device, &pso_cache)
            .expect("Failed to create streaming engine");

        // Empty file list
        let results = engine.search_files(&[], b"pattern", &SearchOptions::default());
        assert_eq!(results.len(), 0);

        // Empty pattern
        let (_dir, paths) = make_streaming_test_files(5, "content");
        let results = engine.search_files(&paths, b"", &SearchOptions::default());
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_streaming_case_insensitive() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);

        let mut engine = StreamingSearchEngine::new(&device, &pso_cache)
            .expect("Failed to create streaming engine");

        let content = "Hello HELLO hello hElLo World\n";
        let (_dir, paths) = make_streaming_test_files(5, content);

        let options = SearchOptions {
            case_sensitive: false,
            ..Default::default()
        };
        let results = engine.search_files(&paths, b"hello", &options);
        let cpu_count = cpu_streaming_search(&paths, b"hello", false);

        println!(
            "Case insensitive: GPU={}, CPU={} for 'hello'",
            results.len(),
            cpu_count
        );

        assert!(results.len() > 0, "Should find case-insensitive matches");
        assert!(
            results.len() <= cpu_count,
            "No false positives: GPU({}) > CPU({})",
            results.len(),
            cpu_count
        );
    }

    #[test]
    fn test_streaming_profile() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);

        let mut engine = StreamingSearchEngine::new(&device, &pso_cache)
            .expect("Failed to create streaming engine");

        let content = "fn test() { return 42; }\n".repeat(100);
        let (_dir, paths) = make_streaming_test_files(20, &content);

        let options = SearchOptions::default();
        let (_results, profile) =
            engine.search_files_with_profile(&paths, b"test", &options);

        assert!(profile.total_us > 0, "Total time should be recorded");
        assert!(profile.files_processed > 0, "Should process files");
        assert!(profile.bytes_processed > 0, "Should process bytes");
        assert!(profile.chunk_count > 0, "Should have chunks");

        profile.print();
    }

    // ====================================================================
    // Hot/cold dispatch threshold tests
    // ====================================================================

    #[test]
    fn test_should_use_gpu_large_files() {
        // Files >= 128KB always go to GPU regardless of page_cache_hint
        assert!(should_use_gpu(128 * 1024, false), "128KB cold -> GPU");
        assert!(should_use_gpu(128 * 1024, true), "128KB hot -> GPU");
        assert!(should_use_gpu(1024 * 1024, false), "1MB cold -> GPU");
        assert!(should_use_gpu(1024 * 1024, true), "1MB hot -> GPU");
        assert!(should_use_gpu(100 * 1024 * 1024, true), "100MB hot -> GPU");
    }

    #[test]
    fn test_should_use_gpu_small_files() {
        // Files < 4KB always go to CPU regardless of page_cache_hint
        assert!(!should_use_gpu(0, false), "0B cold -> CPU");
        assert!(!should_use_gpu(0, true), "0B hot -> CPU");
        assert!(!should_use_gpu(1024, false), "1KB cold -> CPU");
        assert!(!should_use_gpu(1024, true), "1KB hot -> CPU");
        assert!(!should_use_gpu(4 * 1024 - 1, false), "4KB-1 cold -> CPU");
        assert!(!should_use_gpu(4 * 1024 - 1, true), "4KB-1 hot -> CPU");
    }

    #[test]
    fn test_should_use_gpu_middle_range() {
        // Middle range (4KB to 128KB): page_cache_hint decides
        // hot (in page cache) -> CPU; cold (not cached) -> GPU
        let mid = 64 * 1024; // 64KB -- squarely in the middle

        assert!(should_use_gpu(mid, false), "64KB cold -> GPU");
        assert!(!should_use_gpu(mid, true), "64KB hot -> CPU");

        // Boundary: exactly 4KB
        assert!(should_use_gpu(4 * 1024, false), "4KB cold -> GPU");
        assert!(!should_use_gpu(4 * 1024, true), "4KB hot -> CPU");

        // Boundary: just under 128KB
        assert!(should_use_gpu(128 * 1024 - 1, false), "128KB-1 cold -> GPU");
        assert!(!should_use_gpu(128 * 1024 - 1, true), "128KB-1 hot -> CPU");
    }

    #[test]
    fn test_should_use_gpu_boundary_values() {
        // Exact boundaries
        assert!(!should_use_gpu(COLD_THRESHOLD_BYTES - 1, false), "just below cold threshold -> CPU");
        assert!(should_use_gpu(COLD_THRESHOLD_BYTES, false), "exactly cold threshold, cold -> GPU");
        assert!(should_use_gpu(HOT_THRESHOLD_BYTES, false), "exactly hot threshold -> GPU");
        assert!(should_use_gpu(HOT_THRESHOLD_BYTES, true), "exactly hot threshold, hot -> GPU");
    }
}
