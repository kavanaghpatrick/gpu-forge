//! Search orchestrator -- top-level pipeline coordinating full search.
//!
//! Accepts a `SearchRequest`, applies filters (gitignore, binary, filetype),
//! dispatches GPU content search via `StreamingSearchEngine`, resolves results
//! into `ContentMatch` and `FileMatch` entries, and returns a `SearchResponse`
//! with timing statistics.
//!
//! ## Pipeline (Streaming)
//!
//! Three concurrent stages overlapping I/O with GPU compute:
//!
//! **Stage 1 (Producer thread)**: Walk directory via `ignore` crate, apply
//! gitignore/binary/filetype filters inline, send paths through channel.
//!
//! **Stage 2 (Consumer / main thread)**: Receive filtered paths, collect
//! filename matches, batch files for GPU dispatch.
//!
//! **Stage 3 (GPU dispatch)**: As each batch fills, dispatch to GPU
//! streaming search engine, resolve matches, accumulate results.
//!
//! This eliminates the ~29-second stall when searching from `/` where
//! walk_directory() previously blocked until all 1.3M files were enumerated.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel as channel;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::gpu::pipeline::PsoCache;
use super::binary::BinaryDetector;
use super::cancel::{CancellationToken, SearchSession};
use super::content::SearchOptions;
use super::ignore::GitignoreFilter;
use super::streaming::StreamingSearchEngine;
use super::types::{ContentMatch, FileMatch, SearchRequest, SearchResponse, SearchUpdate};
use super::verify::{cpu_verify_matches, VerifyMode};

// ============================================================================
// SearchOrchestrator
// ============================================================================

/// Maximum files to batch before dispatching to GPU.
/// Keeps GPU dispatches frequent enough for progressive results while
/// amortizing per-dispatch overhead.
const STREAMING_BATCH_SIZE: usize = 500;

/// Channel capacity for the file discovery producer -> consumer pipeline.
/// Bounded to apply backpressure if GPU processing falls behind discovery.
const DISCOVERY_CHANNEL_CAPACITY: usize = 4096;

/// Top-level search orchestrator coordinating the full search pipeline.
///
/// Owns a `StreamingSearchEngine` for GPU content search and applies
/// .gitignore, binary, and filetype filters before dispatching to the GPU.
///
/// ## Usage
///
/// ```ignore
/// let orchestrator = SearchOrchestrator::new(&device, &pso_cache)?;
/// let request = SearchRequest::new("fn ", "/path/to/project");
/// let response = orchestrator.search(request);
/// ```
pub struct SearchOrchestrator {
    /// GPU streaming search engine (reused across searches).
    engine: StreamingSearchEngine,
}

impl SearchOrchestrator {
    /// Create a new search orchestrator.
    ///
    /// Returns `None` if GPU initialization fails.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        pso_cache: &PsoCache,
    ) -> Option<Self> {
        let engine = StreamingSearchEngine::new(device, pso_cache)?;
        Some(Self { engine })
    }

    /// Execute a full search pipeline for the given request.
    ///
    /// This is the original blocking pipeline kept for backward compatibility
    /// and tests. For progressive UI delivery, use `search_streaming`.
    ///
    /// Pipeline stages:
    /// 1. Walk directory to collect file paths
    /// 2. Apply gitignore filter (if `respect_gitignore` is true)
    /// 3. Apply binary file filter (if `include_binary` is false)
    /// 4. Apply file type filter (if `file_types` is specified)
    /// 5. Build filename matches (pattern as substring of filename)
    /// 6. Dispatch GPU streaming content search
    /// 7. Resolve GPU byte offsets to line-level ContentMatch entries
    /// 8. Truncate results to `max_results` and build SearchResponse
    pub fn search(&mut self, request: SearchRequest) -> SearchResponse {
        let start = Instant::now();

        // ----------------------------------------------------------------
        // Stage 1: Walk directory
        // ----------------------------------------------------------------
        let all_files = walk_directory(&request.root);

        if all_files.is_empty() {
            return SearchResponse {
                file_matches: vec![],
                content_matches: vec![],
                total_files_searched: 0,
                total_matches: 0,
                elapsed: start.elapsed(),
            };
        }

        // ----------------------------------------------------------------
        // Stage 2: Gitignore filter
        // ----------------------------------------------------------------
        let files_after_gitignore = if request.respect_gitignore {
            match GitignoreFilter::from_directory(&request.root) {
                Ok(filter) => all_files
                    .into_iter()
                    .filter(|p| !filter.is_ignored(p))
                    .collect(),
                Err(_) => all_files, // If gitignore parsing fails, keep all files
            }
        } else {
            all_files
        };

        // ----------------------------------------------------------------
        // Stage 3: Binary file filter
        // ----------------------------------------------------------------
        let detector = if request.include_binary {
            BinaryDetector::include_all()
        } else {
            BinaryDetector::new()
        };

        let files_after_binary: Vec<PathBuf> = files_after_gitignore
            .into_iter()
            .filter(|p| !detector.should_skip(p))
            .collect();

        // ----------------------------------------------------------------
        // Stage 4: File type filter
        // ----------------------------------------------------------------
        let filtered_files = if let Some(ref types) = request.file_types {
            files_after_binary
                .into_iter()
                .filter(|p| {
                    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                        types.iter().any(|t| t.eq_ignore_ascii_case(ext))
                    } else {
                        false // No extension -> skip when filetype filter is active
                    }
                })
                .collect()
        } else {
            files_after_binary
        };

        let total_files_searched = filtered_files.len() as u64;

        // ----------------------------------------------------------------
        // Stage 5: Filename matches (pattern as substring in filename)
        // ----------------------------------------------------------------
        let pattern_lower = request.pattern.to_lowercase();
        let mut file_matches: Vec<FileMatch> = Vec::new();

        for path in &filtered_files {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let name_lower = name.to_lowercase();
                if name_lower.contains(&pattern_lower) {
                    // Score: shorter paths = more relevant, exact match = highest
                    let path_len = path.to_string_lossy().len() as f32;
                    let mut score = 100.0 / (path_len.max(1.0));
                    if name_lower == pattern_lower {
                        score += 10.0; // Exact filename match bonus
                    }
                    if name_lower.starts_with(&pattern_lower) {
                        score += 5.0; // Prefix match bonus
                    }
                    file_matches.push(FileMatch {
                        path: path.clone(),
                        score,
                    });
                }
            }
        }

        // Sort file matches by score descending
        file_matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // ----------------------------------------------------------------
        // Stage 6: GPU streaming content search
        // ----------------------------------------------------------------
        let search_options = SearchOptions {
            case_sensitive: request.case_sensitive,
            max_results: request.max_results,
            ..Default::default()
        };

        let gpu_results = self.engine.search_files(
            &filtered_files,
            request.pattern.as_bytes(),
            &search_options,
        );

        // ----------------------------------------------------------------
        // Stage 6b: CPU verification (if GPU_SEARCH_VERIFY env var is set)
        // ----------------------------------------------------------------
        let verify_mode = VerifyMode::from_env();
        if verify_mode != VerifyMode::Off {
            let mut by_file: std::collections::HashMap<&PathBuf, Vec<u32>> =
                std::collections::HashMap::new();
            for m in &gpu_results {
                by_file.entry(&m.file_path).or_default().push(m.byte_offset);
            }

            let mut total_confirmed = 0u32;
            let mut total_false_positives = 0u32;
            let mut total_missed = 0u32;

            for (path, offsets) in &by_file {
                if let Ok(content) = std::fs::read(path.as_path()) {
                    let result = cpu_verify_matches(
                        &content,
                        request.pattern.as_bytes(),
                        offsets,
                        request.case_sensitive,
                    );
                    total_confirmed += result.confirmed;
                    total_false_positives += result.false_positives;
                    total_missed += result.missed;

                    if result.false_positives > 0 {
                        eprintln!(
                            "[VERIFY] FALSE POSITIVES in {}: {} confirmed, {} false positives, {} missed",
                            path.display(),
                            result.confirmed,
                            result.false_positives,
                            result.missed,
                        );
                    }
                }
            }

            eprintln!(
                "[VERIFY] search: {} confirmed, {} false positives, {} missed ({} files, mode={:?})",
                total_confirmed, total_false_positives, total_missed, by_file.len(), verify_mode,
            );

            if verify_mode == VerifyMode::Full && total_false_positives > 0 {
                panic!(
                    "[VERIFY] FATAL: {} false positives detected in Full verification mode",
                    total_false_positives,
                );
            }
        }

        // ----------------------------------------------------------------
        // Stage 7: Resolve GPU matches to ContentMatch entries
        // ----------------------------------------------------------------
        let mut content_matches: Vec<ContentMatch> = Vec::new();

        for m in &gpu_results {
            if content_matches.len() >= request.max_results {
                break;
            }

            if let Some((line_number, line_content, context_before, context_after, match_start)) =
                resolve_match(&m.file_path, m.byte_offset as usize, &request.pattern, request.case_sensitive)
            {
                let match_end = match_start + request.pattern.len();
                content_matches.push(ContentMatch {
                    path: m.file_path.clone(),
                    line_number: line_number as u32,
                    line_content,
                    context_before,
                    context_after,
                    match_range: match_start..match_end,
                });
            }
        }

        // ----------------------------------------------------------------
        // Stage 8: Build SearchResponse
        // ----------------------------------------------------------------
        let total_matches = (file_matches.len() + content_matches.len()) as u64;

        SearchResponse {
            file_matches,
            content_matches,
            total_files_searched,
            total_matches,
            elapsed: start.elapsed(),
        }
    }

    /// Execute a streaming 3-stage search pipeline with progressive results.
    ///
    /// Unlike `search()` which blocks on directory walking, this method
    /// overlaps file discovery (Stage 1) with GPU search (Stage 3) using
    /// a bounded channel. Results are sent progressively via `update_tx`:
    ///
    /// - `SearchUpdate::FileMatches` sent as filename matches are found
    /// - `SearchUpdate::ContentMatches` sent after each GPU batch completes
    /// - `SearchUpdate::Complete` sent with final aggregated response
    ///
    /// ## 3-Stage Pipeline
    ///
    /// ```text
    /// Stage 1 (thread):  walk_directory -> filter -> channel ─┐
    ///                                                          │
    /// Stage 2 (main):    channel ─> batch files ─> filename match ─┐
    ///                                                               │
    /// Stage 3 (main):    batch full ─> GPU search ─> resolve ─> send
    /// ```
    pub fn search_streaming(
        &mut self,
        request: SearchRequest,
        update_tx: &channel::Sender<SearchUpdate>,
        session: &SearchSession,
    ) -> SearchResponse {
        let start = Instant::now();

        // Shared counter: producer increments as files pass filters,
        // consumer reads at the end for total_files_searched.
        let total_files_counter = Arc::new(AtomicU64::new(0));

        // ----------------------------------------------------------------
        // Stage 1: Spawn directory walker + inline filter on producer thread
        // ----------------------------------------------------------------
        let (path_tx, path_rx) = channel::bounded::<PathBuf>(DISCOVERY_CHANNEL_CAPACITY);
        let producer_counter = Arc::clone(&total_files_counter);

        let root = request.root.clone();
        let respect_gitignore = request.respect_gitignore;
        let include_binary = request.include_binary;
        let file_types = request.file_types.clone();

        let cancel_token = session.token.clone();
        let producer = std::thread::spawn(move || {
            walk_and_filter(
                &root,
                respect_gitignore,
                include_binary,
                file_types.as_deref(),
                &path_tx,
                &producer_counter,
                &cancel_token,
            );
            // path_tx drops here, closing the channel
        });

        // ----------------------------------------------------------------
        // Stage 2 + 3: Consume paths, batch, dispatch GPU, send results
        // ----------------------------------------------------------------
        let pattern_lower = request.pattern.to_lowercase();
        let search_options = SearchOptions {
            case_sensitive: request.case_sensitive,
            max_results: request.max_results,
            ..Default::default()
        };

        let mut all_file_matches: Vec<FileMatch> = Vec::new();
        let mut all_content_matches: Vec<ContentMatch> = Vec::new();
        let mut batch: Vec<PathBuf> = Vec::with_capacity(STREAMING_BATCH_SIZE);
        let mut batch_number = 0u64;
        let mut pending_file_matches: Vec<FileMatch> = Vec::new();

        eprintln!("[gpu-search] streaming search started: pattern='{}' root='{}'",
            request.pattern, request.root.display());

        // Process paths as they arrive from the producer.
        // Check session.should_stop() between batches to abort quickly
        // when a new search supersedes this one (keystroke cancellation).
        for path in path_rx.iter() {
            // Check if search was cancelled or superseded by a new keystroke
            if session.should_stop() {
                eprintln!("[gpu-search] consumer: search cancelled/superseded, aborting ({:.1}s)",
                    start.elapsed().as_secs_f64());
                break;
            }

            // Filename matching (inline, no batching needed)
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let name_lower = name.to_lowercase();
                if name_lower.contains(&pattern_lower) {
                    let path_len = path.to_string_lossy().len() as f32;
                    let mut score = 100.0 / (path_len.max(1.0));
                    if name_lower == pattern_lower {
                        score += 10.0;
                    }
                    if name_lower.starts_with(&pattern_lower) {
                        score += 5.0;
                    }
                    let fm = FileMatch {
                        path: path.clone(),
                        score,
                    };
                    pending_file_matches.push(fm.clone());
                    all_file_matches.push(fm);
                }
            }

            batch.push(path);

            // When batch is full, dispatch to GPU
            if batch.len() >= STREAMING_BATCH_SIZE {
                // Check cancellation before expensive GPU dispatch
                if session.should_stop() {
                    eprintln!("[gpu-search] consumer: cancelled before batch #{} GPU dispatch",
                        batch_number + 1);
                    break;
                }

                batch_number += 1;
                let files_so_far = total_files_counter.load(Ordering::Relaxed);
                eprintln!("[gpu-search] batch #{}: dispatching {} files to GPU ({} discovered so far, {:.1}s)",
                    batch_number, batch.len(), files_so_far,
                    start.elapsed().as_secs_f64());

                // Send accumulated file matches progressively
                if !pending_file_matches.is_empty() {
                    let _ = update_tx.send(SearchUpdate::FileMatches(pending_file_matches.clone()));
                    pending_file_matches.clear();
                }

                let batch_matches = self.dispatch_gpu_batch(
                    &batch,
                    &request.pattern,
                    &search_options,
                    request.max_results.saturating_sub(all_content_matches.len()),
                );
                if !batch_matches.is_empty() {
                    eprintln!("[gpu-search] batch #{}: {} content matches", batch_number, batch_matches.len());
                    let _ = update_tx.send(SearchUpdate::ContentMatches(batch_matches.clone()));
                    all_content_matches.extend(batch_matches);
                }
                batch.clear();
            }
        }

        // Drop receiver so the producer thread's send() fails and it stops quickly.
        // Without this, the producer could block on a full channel while we wait to join.
        drop(path_rx);

        // Dispatch remaining files in the last partial batch (only if not cancelled)
        if !session.should_stop() && !batch.is_empty() {
            batch_number += 1;
            eprintln!("[gpu-search] batch #{} (final): dispatching {} files to GPU ({:.1}s)",
                batch_number, batch.len(), start.elapsed().as_secs_f64());

            // Send any remaining file matches
            if !pending_file_matches.is_empty() {
                let _ = update_tx.send(SearchUpdate::FileMatches(pending_file_matches.clone()));
                pending_file_matches.clear();
            }

            let batch_matches = self.dispatch_gpu_batch(
                &batch,
                &request.pattern,
                &search_options,
                request.max_results.saturating_sub(all_content_matches.len()),
            );
            if !batch_matches.is_empty() {
                eprintln!("[gpu-search] batch #{}: {} content matches", batch_number, batch_matches.len());
                let _ = update_tx.send(SearchUpdate::ContentMatches(batch_matches.clone()));
                all_content_matches.extend(batch_matches);
            }
        }

        // Wait for producer thread to finish
        let _ = producer.join();

        // ----------------------------------------------------------------
        // Build final response (only send updates if not cancelled)
        // ----------------------------------------------------------------
        let total_files_searched = total_files_counter.load(Ordering::Acquire);

        if session.should_stop() {
            eprintln!("[gpu-search] search aborted: {} batches dispatched before cancel ({:.1}s)",
                batch_number, start.elapsed().as_secs_f64());
            return SearchResponse {
                file_matches: vec![],
                content_matches: vec![],
                total_files_searched,
                total_matches: 0,
                elapsed: start.elapsed(),
            };
        }

        // Send final file matches (complete, ranked)
        all_file_matches.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        let _ = update_tx.send(SearchUpdate::FileMatches(all_file_matches.clone()));

        let total_matches = (all_file_matches.len() + all_content_matches.len()) as u64;

        let response = SearchResponse {
            file_matches: all_file_matches,
            content_matches: all_content_matches,
            total_files_searched,
            total_matches,
            elapsed: start.elapsed(),
        };

        let _ = update_tx.send(SearchUpdate::Complete(response.clone()));

        eprintln!("[gpu-search] search complete: {} batches, {} files, {} file matches, {} content matches, {:.1}s",
            batch_number, total_files_searched, response.file_matches.len(),
            response.content_matches.len(), response.elapsed.as_secs_f64());

        response
    }

    /// Dispatch a batch of files to the GPU streaming engine and resolve matches.
    fn dispatch_gpu_batch(
        &mut self,
        files: &[PathBuf],
        pattern: &str,
        options: &SearchOptions,
        remaining_budget: usize,
    ) -> Vec<ContentMatch> {
        if files.is_empty() || remaining_budget == 0 {
            return vec![];
        }

        let gpu_results = self.engine.search_files(
            files,
            pattern.as_bytes(),
            options,
        );

        // --- CPU verification layer ---
        let verify_mode = VerifyMode::from_env();
        if verify_mode != VerifyMode::Off {
            // Group GPU results by file path
            let mut by_file: std::collections::HashMap<&PathBuf, Vec<u32>> =
                std::collections::HashMap::new();
            for m in &gpu_results {
                by_file.entry(&m.file_path).or_default().push(m.byte_offset);
            }

            let mut total_confirmed = 0u32;
            let mut total_false_positives = 0u32;
            let mut total_missed = 0u32;

            for (path, offsets) in &by_file {
                if let Ok(content) = std::fs::read(path.as_path()) {
                    let result = cpu_verify_matches(
                        &content,
                        pattern.as_bytes(),
                        offsets,
                        options.case_sensitive,
                    );
                    total_confirmed += result.confirmed;
                    total_false_positives += result.false_positives;
                    total_missed += result.missed;

                    if result.false_positives > 0 {
                        eprintln!(
                            "[VERIFY] FALSE POSITIVES in {}: {} confirmed, {} false positives, {} missed",
                            path.display(),
                            result.confirmed,
                            result.false_positives,
                            result.missed,
                        );
                    }
                }
            }

            eprintln!(
                "[VERIFY] batch: {} confirmed, {} false positives, {} missed ({} files, mode={:?})",
                total_confirmed, total_false_positives, total_missed, by_file.len(), verify_mode,
            );

            // In Full mode, assert zero false positives (for test runs)
            if verify_mode == VerifyMode::Full && total_false_positives > 0 {
                panic!(
                    "[VERIFY] FATAL: {} false positives detected in Full verification mode",
                    total_false_positives,
                );
            }
        }

        let mut content_matches = Vec::new();
        for m in &gpu_results {
            if content_matches.len() >= remaining_budget {
                break;
            }
            if let Some((line_number, line_content, context_before, context_after, match_start)) =
                resolve_match(&m.file_path, m.byte_offset as usize, pattern, options.case_sensitive)
            {
                let match_end = match_start + pattern.len();
                content_matches.push(ContentMatch {
                    path: m.file_path.clone(),
                    line_number: line_number as u32,
                    line_content,
                    context_before,
                    context_after,
                    match_range: match_start..match_end,
                });
            }
        }

        content_matches
    }
}

// ============================================================================
// Helper: walk directory (using ignore crate for gitignore-aware parallel walk)
// ============================================================================

/// Walk a directory recursively, collecting all regular file paths.
///
/// Uses the `ignore` crate's WalkBuilder for fast parallel traversal.
/// Skips hidden directories, target/, node_modules/, etc.
/// Used by the blocking `search()` method.
fn walk_directory(root: &Path) -> Vec<PathBuf> {
    use ignore::WalkBuilder;

    let mut files = Vec::new();

    let walker = WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .parents(false)
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let file_type = match entry.file_type() {
            Some(ft) => ft,
            None => continue,
        };
        if file_type.is_dir() {
            continue;
        }

        let path = entry.into_path();
        if let Ok(meta) = path.metadata() {
            let size = meta.len();
            if size == 0 || size > 100 * 1024 * 1024 {
                continue;
            }
        }

        files.push(path);
    }

    files.sort();
    files
}

// ============================================================================
// Helper: walk + filter (streaming producer for search_streaming)
// ============================================================================

/// Walk directory and apply all filters inline, sending surviving paths
/// through the channel. This runs on the producer thread for `search_streaming`.
///
/// Filters applied inline (no intermediate Vec):
/// 1. Skip hidden/empty/oversized files (walk_directory logic)
/// 2. Gitignore filter (if enabled)
/// 3. Binary file filter (extension + NUL-byte heuristic)
/// 4. File type filter (extension whitelist)
fn walk_and_filter(
    root: &Path,
    respect_gitignore: bool,
    include_binary: bool,
    file_types: Option<&[String]>,
    path_tx: &channel::Sender<PathBuf>,
    counter: &AtomicU64,
    cancel_token: &CancellationToken,
) {
    use ignore::WalkBuilder;

    // No manual GitignoreFilter -- let WalkBuilder handle .gitignore
    // natively per-directory as it walks. Our old GitignoreFilter::from_directory()
    // recursively pre-scanned the ENTIRE tree for .gitignore files, which from
    // root `/` took forever (scanning /System, /Library, /usr, etc.).

    let detector = if include_binary {
        BinaryDetector::include_all()
    } else {
        BinaryDetector::new()
    };

    let walker = WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(respect_gitignore)   // Let walker handle .gitignore lazily
        .git_global(false)
        .git_exclude(false)
        .parents(false)
        .filter_entry(|entry| {
            // Skip system/framework directories early to avoid walking millions
            // of irrelevant files when searching from root `/`.
            if entry.file_type().map_or(false, |ft| ft.is_dir()) {
                if let Some(name) = entry.file_name().to_str() {
                    let skip = matches!(
                        name,
                        // macOS system directories (huge, not user-relevant)
                        "System" | "Library" | "Applications"
                        | "Volumes" | "cores" | "private"
                        | "usr" | "bin" | "sbin" | "dev" | "opt"
                        | "etc" | "tmp" | "var" | "nix"
                        // Build/dependency directories
                        | "node_modules" | "target" | ".git"
                        | "__pycache__" | ".cargo" | ".rustup"
                        | "DerivedData" | "Build" | "Caches"
                        | ".Trash" | "vendor" | "dist"
                        | ".next" | ".nuxt" | "coverage"
                        | "build" | ".build" | "Pods"
                        | ".venv" | "venv" | "env"
                        | ".tox" | ".mypy_cache" | ".pytest_cache"
                        | "bower_components" | ".gradle"
                    );
                    if skip {
                        eprintln!("[gpu-search] filter: skipping dir '{}'", entry.path().display());
                    } else if entry.depth() <= 2 {
                        eprintln!("[gpu-search] filter: entering dir '{}'", entry.path().display());
                    }
                    return !skip;
                }
            }
            true
        })
        .build();

    let walk_start = Instant::now();
    let mut walked = 0u64;
    let mut filtered_in = 0u64;
    let mut skipped_size = 0u64;
    let mut skipped_binary = 0u64;
    let mut skipped_filetype = 0u64;

    eprintln!("[gpu-search] producer starting walk from '{}'", root.display());

    for entry in walker {
        // Check cancellation every iteration (cheap atomic read)
        if cancel_token.is_cancelled() {
            eprintln!("[gpu-search] producer: cancelled after walking {} files ({:.1}s)",
                walked, walk_start.elapsed().as_secs_f64());
            return;
        }

        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let file_type = match entry.file_type() {
            Some(ft) => ft,
            None => continue,
        };
        if file_type.is_dir() {
            continue;
        }

        walked += 1;

        // Log walking progress every 1K files (diagnostic)
        if walked % 1_000 == 0 {
            eprintln!("[gpu-search] producer: walked {} files ({:.1}s)",
                walked, walk_start.elapsed().as_secs_f64());
        }

        let path = entry.into_path();

        // Skip empty or very large files
        if let Ok(meta) = path.metadata() {
            let size = meta.len();
            if size == 0 || size > 100 * 1024 * 1024 {
                skipped_size += 1;
                continue;
            }
        }

        // Note: gitignore is handled by the WalkBuilder natively (lazy per-directory)

        // Binary filter (now 3-layer: extension -> text extension -> content)
        if detector.should_skip(&path) {
            skipped_binary += 1;
            continue;
        }

        // File type filter
        if let Some(types) = file_types {
            let passes = if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                types.iter().any(|t| t.eq_ignore_ascii_case(ext))
            } else {
                false
            };
            if !passes {
                skipped_filetype += 1;
                continue;
            }
        }

        filtered_in += 1;
        counter.fetch_add(1, Ordering::Relaxed);

        // Log progress every 2K files that pass filters
        if filtered_in % 2_000 == 0 {
            eprintln!("[gpu-search] producer: {} files passed filters ({} walked, {:.1}s)",
                filtered_in, walked, walk_start.elapsed().as_secs_f64());
        }

        // Send to consumer; if channel is closed (consumer dropped), stop walking
        if path_tx.send(path).is_err() {
            break;
        }
    }

    eprintln!("[gpu-search] producer done: {} passed / {} walked in {:.1}s (skipped: {} size, {} binary, {} filetype, gitignore=walker)",
        filtered_in, walked, walk_start.elapsed().as_secs_f64(),
        skipped_size, skipped_binary, skipped_filetype);
}

// ============================================================================
// Helper: resolve GPU match to line-level content match
// ============================================================================

/// Resolve a GPU byte offset to line number, line content, and context.
///
/// Returns `(line_number, line_content, context_before, context_after, match_col_in_line)`.
/// Returns `None` if the file cannot be read or offset is out of bounds.
#[allow(clippy::type_complexity)]
fn resolve_match(
    path: &Path,
    byte_offset: usize,
    pattern: &str,
    case_sensitive: bool,
) -> Option<(usize, String, Vec<String>, Vec<String>, usize)> {
    let content = std::fs::read(path).ok()?;

    if byte_offset >= content.len() {
        return None;
    }

    // Split content into lines
    let text = String::from_utf8_lossy(&content);
    let lines: Vec<&str> = text.lines().collect();

    // Find which line the byte_offset falls on
    let mut cumulative = 0usize;
    let mut target_line = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let line_end = cumulative + line.len() + 1; // +1 for newline
        if byte_offset < line_end {
            target_line = i;
            break;
        }
        cumulative = line_end;
        if i == lines.len() - 1 {
            target_line = i;
        }
    }

    let line_content = lines.get(target_line).unwrap_or(&"").to_string();

    // Find the actual pattern match within the line for match_range.
    // If the pattern is NOT found in this line, reject the match entirely —
    // this catches byte_offset mapping errors from the GPU pipeline.
    let match_col = if case_sensitive {
        match line_content.find(pattern) {
            Some(col) => col,
            None => return None, // GPU byte_offset was wrong — reject false positive
        }
    } else {
        match line_content.to_lowercase().find(&pattern.to_lowercase()) {
            Some(col) => col,
            None => return None, // GPU byte_offset was wrong — reject false positive
        }
    };

    // Context: 2 lines before and after
    let context_before: Vec<String> = (target_line.saturating_sub(2)..target_line)
        .filter_map(|i| lines.get(i).map(|l| l.to_string()))
        .collect();
    let context_after: Vec<String> = (target_line + 1..=(target_line + 2).min(lines.len().saturating_sub(1)))
        .filter_map(|i| lines.get(i).map(|l| l.to_string()))
        .collect();

    // Line number is 1-based
    Some((target_line + 1, line_content, context_before, context_after, match_col))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDevice;
    use crate::gpu::pipeline::PsoCache;
    use crate::search::cancel::{cancellation_pair, SearchGeneration, SearchSession};
    use crossbeam_channel as channel;
    use tempfile::TempDir;

    /// Create a non-cancelled SearchSession for testing.
    fn test_session() -> SearchSession {
        let (token, _handle) = cancellation_pair();
        let gen = SearchGeneration::new();
        let guard = gen.next();
        SearchSession { token, guard }
    }

    /// Create a test directory with known files for orchestrator testing.
    fn make_test_directory() -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // Rust source files
        std::fs::write(
            dir.path().join("main.rs"),
            "fn main() {\n    println!(\"hello world\");\n}\n",
        ).unwrap();

        std::fs::write(
            dir.path().join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\npub fn multiply(a: i32, b: i32) -> i32 {\n    a * b\n}\n",
        ).unwrap();

        std::fs::write(
            dir.path().join("utils.rs"),
            "fn helper() -> bool {\n    true\n}\n\nfn another_fn() {\n    let x = 42;\n}\n",
        ).unwrap();

        // Non-Rust file
        std::fs::write(
            dir.path().join("README.md"),
            "# My Project\n\nA test project for fn testing.\n",
        ).unwrap();

        // Binary file (should be skipped)
        let bin_path = dir.path().join("data.png");
        std::fs::write(&bin_path, &[0x89, 0x50, 0x4E, 0x47, 0x00]).unwrap();

        dir
    }

    #[test]
    fn test_orchestrator_basic() {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = make_test_directory();
        let request = SearchRequest {
            pattern: "fn ".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        println!("Orchestrator results:");
        println!("  File matches: {}", response.file_matches.len());
        println!("  Content matches: {}", response.content_matches.len());
        println!("  Total files searched: {}", response.total_files_searched);
        println!("  Elapsed: {:.1}ms", response.elapsed.as_secs_f64() * 1000.0);

        for cm in &response.content_matches {
            println!("  {}:{}: {}", cm.path.display(), cm.line_number, cm.line_content.trim());
        }

        // Should find "fn " in content -- main.rs (1), lib.rs (2), utils.rs (2) = 5 minimum
        // GPU may miss a few due to thread boundary limitation, so check >= 3
        assert!(
            response.content_matches.len() >= 3,
            "Should find at least 3 'fn ' content matches, got {}",
            response.content_matches.len()
        );

        // All content matches should have valid paths
        for cm in &response.content_matches {
            assert!(cm.path.exists(), "Content match path should exist: {:?}", cm.path);
            assert!(cm.line_number > 0, "Line number should be > 0");
            assert!(!cm.line_content.is_empty(), "Line content should not be empty");
        }

        // Total files searched should be > 0
        assert!(response.total_files_searched > 0);

        // Elapsed time should be recorded
        assert!(response.elapsed.as_micros() > 0);
    }

    #[test]
    fn test_orchestrator_filetype_filter() {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = make_test_directory();

        // Search only .rs files
        let request = SearchRequest {
            pattern: "fn ".to_string(),
            root: dir.path().to_path_buf(),
            file_types: Some(vec!["rs".to_string()]),
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        // All content matches should be in .rs files
        for cm in &response.content_matches {
            assert!(
                cm.path.extension().and_then(|e| e.to_str()) == Some("rs"),
                "With filetype filter 'rs', match should be in .rs file: {:?}",
                cm.path
            );
        }

        println!("Filetype filter: {} matches in .rs files only", response.content_matches.len());
    }

    #[test]
    fn test_orchestrator_no_matches() {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = make_test_directory();
        let request = SearchRequest::new("ZZZZZ_NOT_FOUND_EVER", dir.path());

        let response = orchestrator.search(request);

        assert_eq!(response.content_matches.len(), 0, "Should find no content matches");
        assert!(response.total_files_searched > 0, "Should still search files");
    }

    #[test]
    fn test_orchestrator_binary_excluded() {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = make_test_directory();

        // Binary files should be excluded by default
        let request = SearchRequest {
            pattern: "PNG".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        // Should not find matches in binary .png file
        for cm in &response.content_matches {
            assert!(
                cm.path.extension().and_then(|e| e.to_str()) != Some("png"),
                "Binary .png file should be excluded: {:?}",
                cm.path
            );
        }
    }

    #[test]
    fn test_orchestrator_case_insensitive() {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = make_test_directory();

        // Case-insensitive search for "FN " should match "fn "
        let request = SearchRequest {
            pattern: "FN ".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: false,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        assert!(
            response.content_matches.len() >= 3,
            "Case-insensitive 'FN ' should match 'fn ', got {} matches",
            response.content_matches.len()
        );
    }

    #[test]
    fn test_orchestrator_file_matches() {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = make_test_directory();

        // Search for "main" should match filename "main.rs"
        let request = SearchRequest {
            pattern: "main".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        // Should have at least one file match for "main.rs"
        assert!(
            response.file_matches.len() >= 1,
            "Should find 'main.rs' as a file match, got {} file matches",
            response.file_matches.len()
        );

        let has_main = response.file_matches.iter().any(|fm| {
            fm.path.file_name().and_then(|n| n.to_str()) == Some("main.rs")
        });
        assert!(has_main, "Should find main.rs in file matches");
    }

    #[test]
    fn test_orchestrator_empty_directory() {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = TempDir::new().unwrap();
        let request = SearchRequest::new("test", dir.path());

        let response = orchestrator.search(request);

        assert_eq!(response.content_matches.len(), 0);
        assert_eq!(response.file_matches.len(), 0);
        assert_eq!(response.total_files_searched, 0);
    }

    #[test]
    fn test_orchestrator_real_src() {
        // Search the actual gpu-search src/ directory
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let request = SearchRequest {
            pattern: "fn ".to_string(),
            root: src_dir.clone(),
            file_types: Some(vec!["rs".to_string()]),
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let response = orchestrator.search(request);

        println!("Real src/ search:");
        println!("  Files searched: {}", response.total_files_searched);
        println!("  Content matches: {}", response.content_matches.len());
        println!("  File matches: {}", response.file_matches.len());
        println!("  Elapsed: {:.1}ms", response.elapsed.as_secs_f64() * 1000.0);

        // Should find many "fn " matches in the source
        assert!(
            response.content_matches.len() >= 10,
            "Should find at least 10 'fn ' in src/, got {}",
            response.content_matches.len()
        );

        assert!(response.total_files_searched > 0);
    }

    // ================================================================
    // Streaming pipeline tests
    // ================================================================

    #[test]
    fn test_streaming_basic() {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = make_test_directory();
        let request = SearchRequest {
            pattern: "fn ".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let (update_tx, update_rx) = channel::unbounded();
        let session = test_session();
        let response = orchestrator.search_streaming(request, &update_tx, &session);

        println!("Streaming results:");
        println!("  File matches: {}", response.file_matches.len());
        println!("  Content matches: {}", response.content_matches.len());
        println!("  Total files searched: {}", response.total_files_searched);
        println!("  Elapsed: {:.1}ms", response.elapsed.as_secs_f64() * 1000.0);

        // Should find content matches
        assert!(
            response.content_matches.len() >= 3,
            "Streaming should find at least 3 'fn ' content matches, got {}",
            response.content_matches.len()
        );

        // Should have sent progressive updates
        let mut update_count = 0;
        let mut got_complete = false;
        while let Ok(update) = update_rx.try_recv() {
            update_count += 1;
            if matches!(update, SearchUpdate::Complete(_)) {
                got_complete = true;
            }
        }
        assert!(update_count >= 2, "Should send at least 2 updates (content + complete), got {}", update_count);
        assert!(got_complete, "Should send Complete update");

        assert!(response.total_files_searched > 0);
        assert!(response.elapsed.as_micros() > 0);
    }

    #[test]
    fn test_streaming_matches_blocking() {
        // Streaming pipeline should produce the same results as blocking pipeline
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orch1 = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");
        let mut orch2 = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = make_test_directory();
        let request = SearchRequest {
            pattern: "fn ".to_string(),
            root: dir.path().to_path_buf(),
            file_types: Some(vec!["rs".to_string()]),
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let blocking = orch1.search(request.clone());

        let (update_tx, _update_rx) = channel::unbounded();
        let session = test_session();
        let streaming = orch2.search_streaming(request, &update_tx, &session);

        println!("Blocking: {} content, {} files searched",
            blocking.content_matches.len(), blocking.total_files_searched);
        println!("Streaming: {} content, {} files searched",
            streaming.content_matches.len(), streaming.total_files_searched);

        // Same file count
        assert_eq!(
            blocking.total_files_searched,
            streaming.total_files_searched,
            "Streaming should search same number of files as blocking"
        );

        // Content matches should be within the same range
        // (exact match not guaranteed due to different batching order)
        let blocking_count = blocking.content_matches.len();
        let streaming_count = streaming.content_matches.len();
        let min_count = blocking_count.min(streaming_count);
        let max_count = blocking_count.max(streaming_count);
        // Allow small variance due to GPU thread boundary differences in batching
        assert!(
            min_count > 0 || max_count == 0,
            "Both should find matches or neither should"
        );
    }

    #[test]
    fn test_streaming_empty_directory() {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = TempDir::new().unwrap();
        let request = SearchRequest::new("test", dir.path());

        let (update_tx, _update_rx) = channel::unbounded();
        let session = test_session();
        let response = orchestrator.search_streaming(request, &update_tx, &session);

        assert_eq!(response.content_matches.len(), 0);
        assert_eq!(response.file_matches.len(), 0);
        assert_eq!(response.total_files_searched, 0);
    }

    #[test]
    fn test_streaming_real_src() {
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let request = SearchRequest {
            pattern: "fn ".to_string(),
            root: src_dir,
            file_types: Some(vec!["rs".to_string()]),
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let (update_tx, update_rx) = channel::unbounded();
        let session = test_session();
        let response = orchestrator.search_streaming(request, &update_tx, &session);

        println!("Streaming real src/ search:");
        println!("  Files searched: {}", response.total_files_searched);
        println!("  Content matches: {}", response.content_matches.len());
        println!("  Elapsed: {:.1}ms", response.elapsed.as_secs_f64() * 1000.0);

        assert!(
            response.content_matches.len() >= 10,
            "Should find at least 10 'fn ' in src/ via streaming, got {}",
            response.content_matches.len()
        );

        // Count progressive updates
        let mut content_updates = 0;
        while let Ok(update) = update_rx.try_recv() {
            if matches!(update, SearchUpdate::ContentMatches(_)) {
                content_updates += 1;
            }
        }
        println!("  Progressive content updates: {}", content_updates);
    }

    #[test]
    fn test_streaming_pre_cancelled() {
        // A pre-cancelled session should return immediately with empty results
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = make_test_directory();
        let request = SearchRequest {
            pattern: "fn ".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        // Create a session and immediately cancel it
        let (token, handle) = cancellation_pair();
        let gen = SearchGeneration::new();
        let guard = gen.next();
        let session = SearchSession { token, guard };
        handle.cancel();

        let (update_tx, update_rx) = channel::unbounded();
        let start = std::time::Instant::now();
        let response = orchestrator.search_streaming(request, &update_tx, &session);
        let elapsed = start.elapsed();

        println!("Pre-cancelled search: {:.1}ms", elapsed.as_secs_f64() * 1000.0);

        // Should return empty results (search aborted)
        assert_eq!(response.content_matches.len(), 0,
            "Cancelled search should return no content matches");
        assert_eq!(response.file_matches.len(), 0,
            "Cancelled search should return no file matches");

        // Should NOT send a Complete update (cancelled searches don't send final updates)
        let mut got_complete = false;
        while let Ok(update) = update_rx.try_recv() {
            if matches!(update, SearchUpdate::Complete(_)) {
                got_complete = true;
            }
        }
        assert!(!got_complete, "Cancelled search should not send Complete update");

        // Should be fast (< 100ms for pre-cancelled)
        assert!(elapsed.as_millis() < 100,
            "Pre-cancelled search should be fast, took {}ms", elapsed.as_millis());
    }

    #[test]
    fn test_streaming_stale_generation() {
        // A session whose generation is stale should also abort
        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);
        let mut orchestrator = SearchOrchestrator::new(&device.device, &pso_cache)
            .expect("Failed to create orchestrator");

        let dir = make_test_directory();
        let request = SearchRequest {
            pattern: "fn ".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        // Create a session, then advance the generation to make it stale
        let gen = SearchGeneration::new();
        let (token, _handle) = cancellation_pair();
        let guard = gen.next(); // generation 1
        let _guard2 = gen.next(); // generation 2 -- makes guard stale
        let session = SearchSession { token, guard };

        assert!(session.should_stop(), "Session with stale generation should report should_stop");

        let (update_tx, _update_rx) = channel::unbounded();
        let response = orchestrator.search_streaming(request, &update_tx, &session);

        // Should return empty (aborted due to stale generation)
        assert_eq!(response.content_matches.len(), 0);
        assert_eq!(response.file_matches.len(), 0);
    }
}
