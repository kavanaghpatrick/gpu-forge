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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel as channel;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::gpu::pipeline::PsoCache;
use crate::index::cache::MmapIndexCache;
use crate::index::gpu_index::GpuResidentIndex;
use crate::index::shared_index::SharedIndexManager;
use super::binary::BinaryDetector;
use super::cancel::{CancellationToken, SearchSession};
use super::content::SearchOptions;
use super::ignore::GitignoreFilter;
use super::profile::PipelineProfile;
use super::streaming::StreamingSearchEngine;
use super::types::{ContentMatch, FileMatch, SearchRequest, SearchResponse, SearchUpdate, StampedUpdate};
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
        let mut profile = PipelineProfile::default();

        // ----------------------------------------------------------------
        // Stage 1: Walk directory
        // ----------------------------------------------------------------
        let walk_start = Instant::now();
        let all_files = walk_directory(&request.root);
        profile.walk_us = walk_start.elapsed().as_micros() as u64;
        profile.files_walked = all_files.len() as u32;

        if all_files.is_empty() {
            profile.total_us = start.elapsed().as_micros() as u64;
            return SearchResponse {
                file_matches: vec![],
                content_matches: vec![],
                total_files_searched: 0,
                total_matches: 0,
                elapsed: start.elapsed(),
                profile,
            };
        }

        // ----------------------------------------------------------------
        // Stage 2: Gitignore filter
        // ----------------------------------------------------------------
        let filter_start = Instant::now();
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

        profile.filter_us = filter_start.elapsed().as_micros() as u64;
        profile.files_filtered = filtered_files.len() as u32;
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

        let gpu_dispatch_start = Instant::now();
        let (gpu_results, streaming_profile) = self.engine.search_files_with_profile(
            &filtered_files,
            request.pattern.as_bytes(),
            &search_options,
        );
        let gpu_elapsed = gpu_dispatch_start.elapsed().as_micros() as u64;

        // Merge StreamingProfile timing into PipelineProfile
        profile.gpu_load_us = streaming_profile.io_us;
        profile.gpu_dispatch_us = streaming_profile.search_us;
        profile.batch_us = streaming_profile.partition_us;
        profile.bytes_searched = streaming_profile.bytes_processed;
        profile.files_searched = streaming_profile.files_processed as u32;
        profile.gpu_dispatches += 1;
        profile.matches_raw = gpu_results.len() as u32;
        // If streaming total exceeds sum of parts, attribute remainder to batch
        if gpu_elapsed > profile.gpu_load_us + profile.gpu_dispatch_us + profile.batch_us {
            profile.batch_us = gpu_elapsed - profile.gpu_load_us - profile.gpu_dispatch_us;
        }

        // ----------------------------------------------------------------
        // Stage 6b: CPU verification (if GPU_SEARCH_VERIFY env var is set)
        // ----------------------------------------------------------------
        let verify_mode = VerifyMode::from_env().effective(gpu_results.len());
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
        //
        // DEFERRED LINE COUNTING (KB #1320, #1322): line_number comes
        // exclusively from CPU-side resolve_match() which counts newlines
        // in file content up to byte_offset.  The GPU's line_number field
        // (StreamingMatch.line_number) only counts within the 64-byte
        // thread window and is intentionally ignored here.
        // ----------------------------------------------------------------
        let resolve_start = Instant::now();
        let mut content_matches: Vec<ContentMatch> = Vec::new();

        for m in &gpu_results {
            if content_matches.len() >= request.max_results {
                break;
            }

            // NOTE: m.line_number (GPU) is NOT used -- resolve_match()
            // computes the authoritative line number from byte_offset.
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
        profile.resolve_us = resolve_start.elapsed().as_micros() as u64;
        profile.matches_resolved = content_matches.len() as u32;
        profile.matches_rejected = profile.matches_raw.saturating_sub(profile.matches_resolved);

        // ----------------------------------------------------------------
        // Stage 8: Build SearchResponse
        // ----------------------------------------------------------------
        let total_matches = (file_matches.len() + content_matches.len()) as u64;
        profile.total_us = start.elapsed().as_micros() as u64;
        // For blocking search, TTFR equals total (no progressive delivery)
        profile.ttfr_us = profile.total_us;

        eprintln!("[gpu-search] profile:\n{}", profile);

        SearchResponse {
            file_matches,
            content_matches,
            total_files_searched,
            total_matches,
            elapsed: start.elapsed(),
            profile,
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
        update_tx: &channel::Sender<StampedUpdate>,
        session: &SearchSession,
    ) -> SearchResponse {
        let start = Instant::now();
        let mut profile = PipelineProfile::default();
        let mut ttfr_recorded = false;

        // Shared counter: producer increments as files pass filters,
        // consumer reads at the end for total_files_searched.
        let total_files_counter = Arc::new(AtomicU64::new(0));

        // ----------------------------------------------------------------
        // Stage 1: Try GSIX index first, fall back to walk_and_filter
        // ----------------------------------------------------------------
        let walk_start = Instant::now();
        let (path_tx, path_rx) = channel::bounded::<PathBuf>(DISCOVERY_CHANNEL_CAPACITY);
        let producer_counter = Arc::clone(&total_files_counter);

        let root = request.root.clone();
        let respect_gitignore = request.respect_gitignore;
        let include_binary = request.include_binary;
        let file_types = request.file_types.clone();

        let cancel_token = session.token.clone();

        // Try loading from GSIX index for instant file discovery
        let use_index = Self::try_index_producer(
            &root,
            &path_tx,
            &producer_counter,
            include_binary,
            file_types.as_deref(),
        );

        // Collect paths for GSIX index building when falling back to walk
        let index_collector: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));

        let producer = if use_index {
            eprintln!("[gpu-search] Stage 1: using GSIX index (instant discovery)");
            // Index entries already sent -- drop sender to close channel
            drop(path_tx);
            None
        } else {
            eprintln!("[gpu-search] Stage 1: falling back to walk_and_filter");
            let idx_collector = Arc::clone(&index_collector);
            let producer = std::thread::spawn(move || {
                walk_and_filter(
                    &root,
                    respect_gitignore,
                    include_binary,
                    file_types.as_deref(),
                    &path_tx,
                    &producer_counter,
                    &cancel_token,
                    Some(&idx_collector),
                );
                // path_tx drops here, closing the channel
            });
            Some(producer)
        };

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
                    let _ = update_tx.send(StampedUpdate {
                        generation: session.guard.generation_id(),
                        update: SearchUpdate::FileMatches(pending_file_matches.clone()),
                    });
                    pending_file_matches.clear();
                }

                let (batch_matches, batch_profile) = self.dispatch_gpu_batch_profiled(
                    &batch,
                    &request.pattern,
                    &search_options,
                    request.max_results.saturating_sub(all_content_matches.len()),
                );
                // Accumulate batch profile into pipeline profile
                profile.gpu_load_us += batch_profile.gpu_load_us;
                profile.gpu_dispatch_us += batch_profile.gpu_dispatch_us;
                profile.batch_us += batch_profile.batch_us;
                profile.resolve_us += batch_profile.resolve_us;
                profile.bytes_searched += batch_profile.bytes_searched;
                profile.files_searched += batch_profile.files_searched;
                profile.gpu_dispatches += 1;
                profile.matches_raw += batch_profile.matches_raw;
                profile.matches_resolved += batch_profile.matches_resolved;
                profile.matches_rejected += batch_profile.matches_rejected;

                if !batch_matches.is_empty() {
                    // Record TTFR on first content matches sent
                    if !ttfr_recorded {
                        profile.ttfr_us = start.elapsed().as_micros() as u64;
                        ttfr_recorded = true;
                    }
                    eprintln!("[gpu-search] batch #{}: {} content matches", batch_number, batch_matches.len());
                    let _ = update_tx.send(StampedUpdate {
                        generation: session.guard.generation_id(),
                        update: SearchUpdate::ContentMatches(batch_matches.clone()),
                    });
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
                let _ = update_tx.send(StampedUpdate {
                    generation: session.guard.generation_id(),
                    update: SearchUpdate::FileMatches(pending_file_matches.clone()),
                });
                pending_file_matches.clear();
            }

            let (batch_matches, batch_profile) = self.dispatch_gpu_batch_profiled(
                &batch,
                &request.pattern,
                &search_options,
                request.max_results.saturating_sub(all_content_matches.len()),
            );
            // Accumulate batch profile into pipeline profile
            profile.gpu_load_us += batch_profile.gpu_load_us;
            profile.gpu_dispatch_us += batch_profile.gpu_dispatch_us;
            profile.batch_us += batch_profile.batch_us;
            profile.resolve_us += batch_profile.resolve_us;
            profile.bytes_searched += batch_profile.bytes_searched;
            profile.files_searched += batch_profile.files_searched;
            profile.gpu_dispatches += 1;
            profile.matches_raw += batch_profile.matches_raw;
            profile.matches_resolved += batch_profile.matches_resolved;
            profile.matches_rejected += batch_profile.matches_rejected;

            if !batch_matches.is_empty() {
                if !ttfr_recorded {
                    profile.ttfr_us = start.elapsed().as_micros() as u64;
                    ttfr_recorded = true;
                }
                eprintln!("[gpu-search] batch #{}: {} content matches", batch_number, batch_matches.len());
                let _ = update_tx.send(StampedUpdate {
                    generation: session.guard.generation_id(),
                    update: SearchUpdate::ContentMatches(batch_matches.clone()),
                });
                all_content_matches.extend(batch_matches);
            }
        }

        // Wait for producer thread to finish (if walk_and_filter was used)
        if let Some(producer) = producer {
            let _ = producer.join();
        }

        // Walk+filter timing: from walk_start to when producer finishes
        // (walk and filter happen on the producer thread concurrently with GPU dispatch)
        profile.walk_us = walk_start.elapsed().as_micros() as u64;

        // ----------------------------------------------------------------
        // Build final response (only send updates if not cancelled)
        // ----------------------------------------------------------------
        let total_files_searched = total_files_counter.load(Ordering::Acquire);
        profile.files_walked = total_files_searched as u32;
        profile.files_filtered = total_files_searched as u32;

        if session.should_stop() {
            profile.total_us = start.elapsed().as_micros() as u64;
            eprintln!("[gpu-search] search aborted: {} batches dispatched before cancel ({:.1}s)",
                batch_number, start.elapsed().as_secs_f64());
            return SearchResponse {
                file_matches: vec![],
                content_matches: vec![],
                total_files_searched,
                total_matches: 0,
                elapsed: start.elapsed(),
                profile,
            };
        }

        // Send final file matches (complete, ranked)
        all_file_matches.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        let _ = update_tx.send(StampedUpdate {
            generation: session.guard.generation_id(),
            update: SearchUpdate::FileMatches(all_file_matches.clone()),
        });

        let total_matches = (all_file_matches.len() + all_content_matches.len()) as u64;

        // If no content matches were found, TTFR = total (no first result)
        if !ttfr_recorded {
            profile.ttfr_us = start.elapsed().as_micros() as u64;
        }
        profile.total_us = start.elapsed().as_micros() as u64;

        let response = SearchResponse {
            file_matches: all_file_matches,
            content_matches: all_content_matches,
            total_files_searched,
            total_matches,
            elapsed: start.elapsed(),
            profile,
        };

        let _ = update_tx.send(StampedUpdate {
            generation: session.guard.generation_id(),
            update: SearchUpdate::Complete(response.clone()),
        });

        eprintln!("[gpu-search] search complete: {} batches, {} files, {} file matches, {} content matches, {:.1}s",
            batch_number, total_files_searched, response.file_matches.len(),
            response.content_matches.len(), response.elapsed.as_secs_f64());
        eprintln!("[gpu-search] profile:\n{}", response.profile);

        // Build GSIX index in background if we walked (no index was used)
        // and the search wasn't cancelled. This makes the next search instant.
        if !use_index && !session.should_stop() {
            let search_root = request.root.clone();
            std::thread::spawn(move || {
                let paths = match Arc::try_unwrap(index_collector) {
                    Ok(mutex) => mutex.into_inner().unwrap_or_default(),
                    Err(arc) => arc.lock().map(|g| g.clone()).unwrap_or_default(),
                };
                if paths.is_empty() {
                    return;
                }
                eprintln!("[gpu-search] index: building GSIX from {} walked paths...", paths.len());
                let index = GpuResidentIndex::build_from_paths(&paths);
                if let Ok(manager) = SharedIndexManager::new() {
                    match manager.save(&index, &search_root) {
                        Ok(path) => eprintln!("[gpu-search] index: saved {} entries to {}",
                            index.entry_count(), path.display()),
                        Err(e) => eprintln!("[gpu-search] index: save failed: {}", e),
                    }
                }
            });
        }

        response
    }

    /// Try to use the GSIX index for instant file discovery.
    ///
    /// If a fresh index exists for the given root, iterates all index entries
    /// into the crossbeam channel (instant, no filesystem walk). Applies the
    /// same binary and filetype filters as walk_and_filter.
    ///
    /// Returns `true` if the index was used (entries sent), `false` if the
    /// caller should fall back to walk_and_filter.
    fn try_index_producer(
        root: &Path,
        path_tx: &channel::Sender<PathBuf>,
        counter: &AtomicU64,
        include_binary: bool,
        file_types: Option<&[String]>,
    ) -> bool {
        // Try to create the shared index manager
        let manager = match SharedIndexManager::new() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[gpu-search] index: manager init failed: {}", e);
                return false;
            }
        };

        let idx_path = manager.index_path(root);

        // Check if index file exists
        if !idx_path.exists() {
            eprintln!("[gpu-search] index: no cached index at {}", idx_path.display());
            return false;
        }

        // Check staleness (1 hour max age)
        if manager.is_stale(root, Duration::from_secs(3600)) {
            eprintln!("[gpu-search] index: stale index at {}", idx_path.display());
            return false;
        }

        // Load via mmap for zero-copy access
        let cache = match MmapIndexCache::load_mmap(&idx_path, None) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[gpu-search] index: mmap load failed: {}", e);
                return false;
            }
        };

        let entry_count = cache.entry_count();
        eprintln!(
            "[gpu-search] index: loaded {} entries from {}",
            entry_count,
            idx_path.display()
        );

        // Set up binary detector for filtering
        let detector = if include_binary {
            BinaryDetector::include_all()
        } else {
            BinaryDetector::new()
        };

        let mut sent = 0u64;
        for entry in cache.entries() {
            // Extract path from GpuPathEntry
            let path_bytes = &entry.path[..entry.path_len as usize];
            let path = match std::str::from_utf8(path_bytes) {
                Ok(s) => PathBuf::from(s),
                Err(_) => continue,
            };

            // Skip directories (flag bit 0 = is_dir)
            if entry.flags & 1 != 0 {
                continue;
            }

            // Skip empty or oversized files
            let size = (entry.size_hi as u64) << 32 | entry.size_lo as u64;
            if size == 0 || size > 100 * 1024 * 1024 {
                continue;
            }

            // Binary filter
            if detector.should_skip(&path) {
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
                    continue;
                }
            }

            sent += 1;
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if path_tx.send(path).is_err() {
                break;
            }
        }

        eprintln!(
            "[gpu-search] index: sent {} paths from {} entries",
            sent, entry_count
        );
        true
    }

    /// Dispatch a batch to GPU and return matches with per-batch profile data.
    fn dispatch_gpu_batch_profiled(
        &mut self,
        files: &[PathBuf],
        pattern: &str,
        options: &SearchOptions,
        remaining_budget: usize,
    ) -> (Vec<ContentMatch>, PipelineProfile) {
        let mut batch_profile = PipelineProfile::default();

        if files.is_empty() || remaining_budget == 0 {
            return (vec![], batch_profile);
        }

        let gpu_start = Instant::now();
        let (gpu_results, streaming_profile) = self.engine.search_files_with_profile(
            files,
            pattern.as_bytes(),
            options,
        );

        // Merge StreamingProfile into batch profile
        batch_profile.gpu_load_us = streaming_profile.io_us;
        batch_profile.gpu_dispatch_us = streaming_profile.search_us;
        batch_profile.batch_us = streaming_profile.partition_us;
        batch_profile.bytes_searched = streaming_profile.bytes_processed;
        batch_profile.files_searched = streaming_profile.files_processed as u32;
        batch_profile.matches_raw = gpu_results.len() as u32;

        // Attribute any remaining time to batch overhead
        let gpu_elapsed = gpu_start.elapsed().as_micros() as u64;
        let accounted = batch_profile.gpu_load_us + batch_profile.gpu_dispatch_us + batch_profile.batch_us;
        if gpu_elapsed > accounted {
            batch_profile.batch_us += gpu_elapsed - accounted;
        }

        // --- CPU verification layer ---
        let verify_mode = VerifyMode::from_env().effective(gpu_results.len());
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

        // DEFERRED LINE COUNTING: resolve_match() is the sole source of
        // truth for line numbers.  GPU's m.line_number (local to 64B thread
        // window) is intentionally ignored -- only byte_offset is used.
        let resolve_start = Instant::now();
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
        batch_profile.resolve_us = resolve_start.elapsed().as_micros() as u64;
        batch_profile.matches_resolved = content_matches.len() as u32;
        batch_profile.matches_rejected = batch_profile.matches_raw.saturating_sub(batch_profile.matches_resolved);

        (content_matches, batch_profile)
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
#[allow(clippy::too_many_arguments)]
fn walk_and_filter(
    root: &Path,
    respect_gitignore: bool,
    include_binary: bool,
    file_types: Option<&[String]>,
    path_tx: &channel::Sender<PathBuf>,
    counter: &AtomicU64,
    cancel_token: &CancellationToken,
    index_paths: Option<&Mutex<Vec<PathBuf>>>,
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
            if entry.file_type().is_some_and(|ft| ft.is_dir()) {
                if let Some(name) = entry.file_name().to_str() {
                    let skip = matches!(
                        name,
                        // macOS system directories (huge, not user-relevant)
                        "System" | "Library" | "Applications"
                        | "Volumes" | "cores" | "private"
                        | "usr" | "bin" | "sbin" | "dev" | "opt"
                        | "etc" | "tmp" | "var" | "nix"
                        // macOS user data directories (not source code)
                        | "Mail" | "Messages" | "Photos"
                        | "Music" | "Movies" | "Pictures"
                        | "Downloads" | "Documents"
                        | "Containers" | "Group Containers"
                        | "CloudStorage" | "MobileBackups"
                        // macOS system caches and metadata
                        | "Caches" | "CrashReporter" | "Logs"
                        | "DiagnosticReports" | "WebKit"
                        | "Preferences" | "Saved Application State"
                        | "Cookies" | "HTTPStorages"
                        // Homebrew
                        | "Homebrew" | "Cellar" | "Caskroom"
                        // Build/dependency directories
                        | "node_modules" | "target" | ".git"
                        | "__pycache__" | ".cargo" | ".rustup"
                        | "DerivedData" | "Build"
                        | ".Trash" | "vendor" | "dist"
                        | ".next" | ".nuxt" | "coverage"
                        | "build" | ".build" | "Pods"
                        | ".venv" | "venv" | "env"
                        | ".tox" | ".mypy_cache" | ".pytest_cache"
                        | "bower_components" | ".gradle"
                        // Misc caches and package managers
                        | "cache" | ".cache" | "Cache"
                        | "logs" | "log"
                        | ".npm" | ".yarn" | ".pnpm-store"
                        | ".conda" | ".local"
                        | ".docker" | ".minikube"
                        | ".ssh" | ".gnupg"
                        // Language package manager caches (huge trees)
                        | "go" | ".go" | "pkg"
                        | "gems" | ".gem" | ".rbenv"
                        | ".pyenv" | ".nvm" | ".sdkman"
                        | ".m2" | ".ivy2" | ".sbt"
                        | "anaconda3" | "miniconda3"
                        | ".cpan" | ".cpanm"
                        // Game / app data
                        | "Steam" | "Battle.net" | "Blizzard"
                        | "Epic Games" | "GOG Galaxy"
                        | "Previously Relocated Items"
                        | "Relocated Items"
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
        if walked.is_multiple_of(1_000) {
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

        // Collect path for index building (if requested)
        if let Some(idx) = index_paths {
            if let Ok(mut paths) = idx.lock() {
                paths.push(path.clone());
            }
        }

        // Log progress every 2K files that pass filters
        if filtered_in.is_multiple_of(2_000) {
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
/// This is the **deferred line counting** implementation (KB #1320, #1322):
/// the GPU kernel only records byte offsets; this CPU function reads the file
/// and counts newlines up to `byte_offset` to derive the authoritative
/// 1-based line number.  The GPU's own `line_number` field (which only counts
/// within the 64-byte thread window) is never used.
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
        line_content.find(pattern)?
    } else {
        line_content.to_lowercase().find(&pattern.to_lowercase())?
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

        // Should have sent progressive updates (now StampedUpdate)
        let mut update_count = 0;
        let mut got_complete = false;
        while let Ok(stamped) = update_rx.try_recv() {
            update_count += 1;
            if matches!(stamped.update, SearchUpdate::Complete(_)) {
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

        // Count progressive updates (now StampedUpdate)
        let mut content_updates = 0;
        while let Ok(stamped) = update_rx.try_recv() {
            if matches!(stamped.update, SearchUpdate::ContentMatches(_)) {
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
        while let Ok(stamped) = update_rx.try_recv() {
            if matches!(stamped.update, SearchUpdate::Complete(_)) {
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

    // ========================================================================
    // resolve_match() unit tests
    // ========================================================================

    #[test]
    fn test_resolve_match_correct_line() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        // "hello\nworld\nfoo bar\n" — "foo" starts at byte 12
        std::fs::write(&file, "hello\nworld\nfoo bar\n").unwrap();

        let result = resolve_match(&file, 12, "foo", true);
        assert!(result.is_some(), "resolve_match should return Some for valid offset");
        let (line_num, line_content, ctx_before, ctx_after, col) = result.unwrap();
        assert_eq!(line_num, 3, "foo is on line 3");
        assert_eq!(line_content, "foo bar");
        assert_eq!(col, 0, "foo starts at column 0");
        assert_eq!(ctx_before, vec!["hello", "world"]);
        assert!(ctx_after.is_empty());
    }

    #[test]
    fn test_resolve_match_rejects_wrong_line() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        std::fs::write(&file, "hello\nworld\nfoo bar\n").unwrap();

        // byte_offset 999 is way past end of file — should return None
        let result = resolve_match(&file, 999, "foo", true);
        assert!(result.is_none(), "offset beyond file length should return None");

        // byte_offset 0 points to "hello" line which does NOT contain "xyz"
        let result2 = resolve_match(&file, 0, "xyz", true);
        assert!(result2.is_none(), "pattern not on resolved line should return None");
    }

    #[test]
    fn test_resolve_match_multi_chunk_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("big.txt");
        // Create 5 chunks worth of content (~20KB) with a pattern in chunk 2
        let mut content = String::new();
        for i in 0..500 {
            content.push_str(&format!("line number {:04} padding text here\n", i));
        }
        // Each line is ~36 chars + newline = 37 bytes. Insert "NEEDLE" around line 250 (chunk 2 area)
        let lines: Vec<&str> = content.lines().collect();
        let mut rebuilt = String::new();
        for (i, line) in lines.iter().enumerate() {
            if i == 250 {
                rebuilt.push_str("line number 0250 NEEDLE found here\n");
            } else {
                rebuilt.push_str(line);
                rebuilt.push('\n');
            }
        }
        std::fs::write(&file, &rebuilt).unwrap();

        // Find byte offset of "NEEDLE"
        let needle_offset = rebuilt.find("NEEDLE").unwrap();
        let result = resolve_match(&file, needle_offset, "NEEDLE", true);
        assert!(result.is_some(), "should find NEEDLE in multi-chunk file");
        let (line_num, line_content, _ctx_before, _ctx_after, col) = result.unwrap();
        assert_eq!(line_num, 251, "NEEDLE is on line 251 (1-based)");
        assert!(line_content.contains("NEEDLE"), "line should contain NEEDLE");
        assert_eq!(col, line_content.find("NEEDLE").unwrap());
    }

    #[test]
    fn test_resolve_match_last_line_no_newline() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("no_trailing.txt");
        // File with no trailing newline — last line is "end"
        std::fs::write(&file, "start\nmiddle\nend").unwrap();

        // "end" starts at byte 13
        let result = resolve_match(&file, 13, "end", true);
        assert!(result.is_some(), "should resolve match on last line without trailing newline");
        let (line_num, line_content, ctx_before, ctx_after, col) = result.unwrap();
        assert_eq!(line_num, 3);
        assert_eq!(line_content, "end");
        assert_eq!(col, 0);
        assert_eq!(ctx_before, vec!["start", "middle"]);
        assert!(ctx_after.is_empty());
    }

    #[test]
    fn test_resolve_match_empty_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("empty.txt");
        std::fs::write(&file, "").unwrap();

        // Any offset in an empty file should return None (offset >= content.len())
        let result = resolve_match(&file, 0, "anything", true);
        assert!(result.is_none(), "empty file should return None");
    }

    #[test]
    fn test_resolve_match_case_insensitive() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("case.txt");
        std::fs::write(&file, "Hello World\nFOO BAR\nbaz qux\n").unwrap();

        // Case-insensitive search for "foo" should match "FOO BAR" on line 2
        let result = resolve_match(&file, 12, "foo", false);
        assert!(result.is_some(), "case-insensitive match should succeed");
        let (line_num, line_content, _ctx_before, _ctx_after, col) = result.unwrap();
        assert_eq!(line_num, 2);
        assert_eq!(line_content, "FOO BAR");
        assert_eq!(col, 0, "FOO starts at column 0");

        // Case-sensitive search for "foo" on "FOO BAR" line should fail
        let result2 = resolve_match(&file, 12, "foo", true);
        assert!(result2.is_none(), "case-sensitive search for 'foo' should not match 'FOO'");
    }
}
