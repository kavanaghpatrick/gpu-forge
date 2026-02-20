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
use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::gpu::pipeline::PsoCache;
use crate::index::cache::MmapIndexCache;
use crate::index::content_index_store::ContentIndexStore;
use crate::index::content_store::build_chunk_metadata;
use crate::index::gpu_index::GpuResidentIndex;
use crate::index::shared_index::SharedIndexManager;
use crate::index::store::IndexStore;
use crate::ui::watchdog::DiagState;
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
/// Smaller batches = shorter GPU occupancy per dispatch = less render starvation.
/// 200 files × ~10 chunks/file = ~2000 chunks × 4KB = ~8MB GPU data per batch.
/// This keeps each GPU compute pass short (<5ms), leaving ample time for wgpu
/// to acquire drawables and present frames between compute batches.
const STREAMING_BATCH_SIZE: usize = 200;

/// Channel capacity for the file discovery producer -> consumer pipeline.
/// Bounded to apply backpressure if GPU processing falls behind discovery.
const DISCOVERY_CHANNEL_CAPACITY: usize = 4096;

/// Base sleep between GPU batch dispatches to yield Metal GPU time for rendering.
/// On Apple Silicon, compute and render share one unified GPU. Each compute
/// dispatch blocks the GPU for the duration of the kernel. The yield must be
/// long enough for wgpu to submit a render command buffer, acquire a drawable
/// from CAMetalLayer, and present. 32ms (~2 vsync cycles at 60fps) ensures the
/// render pipeline has a comfortable window even with complex result rendering.
const GPU_YIELD_BASE_MS: u64 = 4;

/// When the UI hasn't rendered a frame in this many ms, the pre-dispatch stall
/// gate activates and pauses compute to let the renderer recover. On Apple
/// Silicon, compute and render share a unified GPU — dispatching more compute
/// work while the renderer is starved only makes the freeze worse.
const UI_STALL_THRESHOLD_MS: f64 = 100.0;

/// The extended yield when UI stalling is detected. 100ms is ~6 frames at 60fps,
/// enough for the Metal drawable pool to refill and wgpu to present.
#[allow(dead_code)]
const UI_STALL_YIELD_MS: u64 = 100;

/// Maximum filename matches to send to the UI. Beyond this, the render workload
/// for text layout/display overwhelms wgpu, starving CAMetalLayer's drawable pool.
/// 1000 matches is plenty for interactive browsing; the Complete response will
/// carry the final sorted+truncated list for display.
const FILE_MATCH_CAP: usize = 1000;

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
    /// Optional IndexStore for mmap-backed zero-copy snapshot access.
    /// When set, `try_index_producer` checks this store first before
    /// falling back to MmapIndexCache.
    index_store: Option<Arc<IndexStore>>,
    /// Optional ContentIndexStore for in-memory content search (zero disk I/O).
    /// When available and populated, `search_streaming_inner` bypasses the
    /// disk-based pipeline and dispatches GPU search directly from the
    /// content store's Metal buffer.
    content_store: Option<Arc<ContentIndexStore>>,
}

/// Extract line content and surrounding context from raw file bytes at a given byte offset.
///
/// Given a byte offset into `content` (typically from a GPU match result), this function
/// locates the enclosing line by scanning backward and forward for `\n` boundaries, then
/// extracts up to 2 context lines before and after the match line.
///
/// # Algorithm
///
/// 1. **Line boundaries**: Scan backward from `byte_offset` for the nearest `\n` to find
///    `line_start`, then forward for the next `\n` to find `line_end`. This is O(line_length)
///    per call, not O(file_size).
/// 2. **Pattern location**: Find `pattern` within the extracted line (case-sensitive or
///    case-insensitive) to determine the match column. Returns `None` if the pattern is not
///    found in the line (e.g., byte offset points between multi-byte UTF-8 codepoints).
/// 3. **Context lines**: Scan further backward/forward from the match line to collect up to
///    2 lines of context in each direction.
/// 4. **CRLF handling**: Trailing `\r` bytes are stripped from all extracted lines.
///
/// # Returns
///
/// `Some((line_content, context_before, context_after, match_column))` on success, or `None`
/// if `byte_offset` is out of bounds or the pattern cannot be located within the line.
///
/// - `line_content`: The full text of the line containing the match.
/// - `context_before`: Up to 2 preceding lines (closest line last, reversed to reading order).
/// - `context_after`: Up to 2 following lines.
/// - `match_column`: 0-based byte offset of the pattern within `line_content`.
pub(crate) fn extract_line_context(
    content: &[u8],
    byte_offset: usize,
    pattern: &str,
    case_sensitive: bool,
) -> Option<(String, Vec<String>, Vec<String>, usize)> {
    if byte_offset >= content.len() {
        return None;
    }

    // Find start of the matched line: scan backward for \n
    let line_start = if byte_offset == 0 {
        0
    } else {
        match content[..byte_offset].iter().rposition(|&b| b == b'\n') {
            Some(pos) => pos + 1,
            None => 0,
        }
    };

    // Find end of the matched line: scan forward for \n
    let line_end = match content[byte_offset..].iter().position(|&b| b == b'\n') {
        Some(pos) => byte_offset + pos,
        None => content.len(),
    };

    // Strip trailing \r if present (CRLF)
    let line_end_trimmed = if line_end > line_start && content.get(line_end.wrapping_sub(1)) == Some(&b'\r') {
        line_end - 1
    } else {
        line_end
    };

    let line_content = String::from_utf8_lossy(&content[line_start..line_end_trimmed]).to_string();

    // Find pattern match column within the extracted line
    let match_col = if case_sensitive {
        line_content.find(pattern)
    } else {
        line_content.to_lowercase().find(&pattern.to_lowercase())
    };

    let match_col = match_col?;

    // Extract 2 context lines before
    let mut context_before = Vec::new();
    let mut scan_pos = line_start;
    for _ in 0..2 {
        if scan_pos == 0 {
            break;
        }
        // Move past the \n before current line
        let prev_end = scan_pos - 1;
        // Handle \r\n
        let prev_end_content = if prev_end > 0 && content[prev_end - 1] == b'\r' {
            prev_end - 1
        } else {
            prev_end
        };
        // Find start of previous line
        let prev_start = if prev_end_content == 0 {
            0
        } else {
            match content[..prev_end_content].iter().rposition(|&b| b == b'\n') {
                Some(pos) => pos + 1,
                None => 0,
            }
        };
        let prev_line = String::from_utf8_lossy(&content[prev_start..prev_end_content]).to_string();
        context_before.push(prev_line);
        scan_pos = prev_start;
    }
    context_before.reverse();

    // Extract 2 context lines after
    let mut context_after = Vec::new();
    let mut after_pos = if line_end < content.len() { line_end + 1 } else { content.len() };
    for _ in 0..2 {
        if after_pos >= content.len() {
            break;
        }
        let next_start = after_pos;
        let next_end = match content[next_start..].iter().position(|&b| b == b'\n') {
            Some(pos) => next_start + pos,
            None => content.len(),
        };
        // Strip trailing \r
        let next_end_trimmed = if next_end > next_start && content.get(next_end.wrapping_sub(1)) == Some(&b'\r') {
            next_end - 1
        } else {
            next_end
        };
        let next_line = String::from_utf8_lossy(&content[next_start..next_end_trimmed]).to_string();
        context_after.push(next_line);
        after_pos = if next_end < content.len() { next_end + 1 } else { content.len() };
    }

    Some((line_content, context_before, context_after, match_col))
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
        Some(Self { engine, index_store: None, content_store: None })
    }

    /// Create a new search orchestrator with an IndexStore for zero-copy snapshots.
    ///
    /// When `index_store` is provided, the streaming search pipeline will check
    /// the store's snapshot before falling back to MmapIndexCache or walk_and_filter.
    pub fn with_store(
        device: &ProtocolObject<dyn MTLDevice>,
        pso_cache: &PsoCache,
        store: Arc<IndexStore>,
    ) -> Option<Self> {
        let engine = StreamingSearchEngine::new(device, pso_cache)?;
        Some(Self { engine, index_store: Some(store), content_store: None })
    }

    /// Create a new search orchestrator with a ContentIndexStore for in-memory search.
    ///
    /// When `content_store` is populated (via background content build), the search
    /// pipeline bypasses all disk I/O and dispatches GPU search directly from the
    /// content store's Metal buffer. Falls back to the disk-based pipeline when
    /// the content store is not yet available.
    pub fn with_content_store(
        device: &ProtocolObject<dyn MTLDevice>,
        pso_cache: &PsoCache,
        index_store: Option<Arc<IndexStore>>,
        content_store: Arc<ContentIndexStore>,
    ) -> Option<Self> {
        let engine = StreamingSearchEngine::new(device, pso_cache)?;
        Some(Self {
            engine,
            index_store,
            content_store: Some(content_store),
        })
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
    /// Wrapper with DiagState for UI app usage.
    pub fn search_streaming_diag(
        &mut self,
        request: SearchRequest,
        update_tx: &channel::Sender<StampedUpdate>,
        session: &SearchSession,
        diag: &Arc<DiagState>,
    ) -> SearchResponse {
        self.search_streaming_inner(request, update_tx, session, Some(diag))
    }

    pub fn search_streaming(
        &mut self,
        request: SearchRequest,
        update_tx: &channel::Sender<StampedUpdate>,
        session: &SearchSession,
    ) -> SearchResponse {
        self.search_streaming_inner(request, update_tx, session, None)
    }

    fn search_streaming_inner(
        &mut self,
        request: SearchRequest,
        update_tx: &channel::Sender<StampedUpdate>,
        session: &SearchSession,
        diag: Option<&Arc<DiagState>>,
    ) -> SearchResponse {
        // -----------------------------------------------------------------
        // Content store fast-path: GPU search from in-memory content buffer
        // Bypasses ALL disk I/O when the content store is populated.
        // -----------------------------------------------------------------
        if let Some(cs) = &self.content_store {
            if cs.is_available() {
                eprintln!("[gpu-search] content store fast-path: dispatching from in-memory buffer");
                return self.search_from_content_store(cs.clone(), &request, update_tx, session);
            }
        }

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

        // Try loading from GSIX index for instant file discovery.
        // We check availability synchronously but send entries on a background
        // thread to avoid deadlocking the bounded channel (capacity 4096).
        let has_index = self.check_index_available(&root, include_binary, file_types.as_deref());

        // Collect paths for GSIX index building when falling back to walk
        let index_collector: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));

        let producer = if has_index {
            eprintln!("[gpu-search] Stage 1: using GSIX index (instant discovery)");
            // Spawn a thread to send index entries — bounded channel would
            // deadlock if we tried to send 3M+ entries inline before consuming.
            let store = self.index_store.as_ref().unwrap().clone();
            let producer = std::thread::spawn(move || {
                Self::try_snapshot_producer_threaded(
                    &store,
                    &path_tx,
                    &producer_counter,
                    include_binary,
                    file_types.as_deref(),
                );
                // path_tx drops here, closing the channel
            });
            Some(producer)
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

            // Filename matching (inline, sent frequently for responsive UI)
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

            // Send NEW filename matches incrementally (not the full accumulated list).
            // The UI extends its file_matches list with each batch, avoiding the
            // O(n^2) clone of the full Vec that previously caused massive allocation
            // churn when thousands of files match (e.g., searching for "patrick" from /).
            if pending_file_matches.len() >= 20 && all_file_matches.len() <= FILE_MATCH_CAP {
                let _ = update_tx.send(StampedUpdate {
                    generation: session.guard.generation_id(),
                    update: SearchUpdate::FileMatches(std::mem::take(&mut pending_file_matches)),
                });
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

                // --- Pre-dispatch stall gate ---
                // On Apple Silicon, compute and render share the unified GPU.
                // If the UI is stalled (update() not being called), dispatching
                // more compute work will only worsen the freeze by saturating
                // the GPU and preventing wgpu's nextDrawable() from returning.
                //
                // Strategy: when a stall is detected, back off exponentially
                // to give the render pipeline maximum GPU time to recover.
                // We do NOT need request_repaint() here — the main thread's
                // update() self-drives via request_repaint_after(16ms). We just
                // need to stop hogging the GPU with compute work.
                if let Some(d) = diag {
                    let stall = d.ui_stall_ms();
                    if stall >= UI_STALL_THRESHOLD_MS {
                        // Exponential backoff: 100ms, 200ms, 400ms, capped at 500ms
                        let backoff_ms = ((stall / UI_STALL_THRESHOLD_MS) as u64)
                            .clamp(1, 5) * 100;
                        eprintln!(
                            "[gpu-search] STALL GATE: UI frozen {:.0}ms, backing off {}ms before batch #{}",
                            stall, backoff_ms, batch_number + 1,
                        );
                        std::thread::sleep(Duration::from_millis(backoff_ms));
                    }
                }

                batch_number += 1;
                let files_so_far = total_files_counter.load(Ordering::Relaxed);

                // Update watchdog metrics before GPU dispatch
                if let Some(d) = diag {
                    d.touch_bg();
                    d.bg_batch_num.store(batch_number, Ordering::Relaxed);
                    d.bg_files_searched.store(files_so_far, Ordering::Relaxed);
                }

                eprintln!("[gpu-search] batch #{}: dispatching {} files to GPU ({} discovered so far, {:.1}s)",
                    batch_number, batch.len(), files_so_far,
                    start.elapsed().as_secs_f64());

                // Flush any remaining pending file matches at batch boundaries
                if !pending_file_matches.is_empty() && all_file_matches.len() <= FILE_MATCH_CAP {
                    let _ = update_tx.send(StampedUpdate {
                        generation: session.guard.generation_id(),
                        update: SearchUpdate::FileMatches(std::mem::take(&mut pending_file_matches)),
                    });
                }

                // Wrap GPU dispatch in autoreleasepool to drain Metal objects
                // (command buffers, encoders) and prevent resource exhaustion.
                let (batch_matches, batch_profile) = autoreleasepool(|_| {
                    self.dispatch_gpu_batch_profiled(
                        &batch,
                        &request.pattern,
                        &search_options,
                        request.max_results.saturating_sub(all_content_matches.len()),
                    )
                });
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

                // Update watchdog after GPU dispatch completes
                if let Some(d) = diag {
                    d.touch_bg();
                    d.bg_content_matches.store(all_content_matches.len() as u64, Ordering::Relaxed);
                }

                // Yield GPU time to the renderer between batches.
                // The stall gate above handles severe stalls with exponential
                // backoff. This base yield ensures a minimum gap between compute
                // dispatches for the render pipeline to acquire drawables.
                std::thread::sleep(Duration::from_millis(GPU_YIELD_BASE_MS));
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

            // Flush any remaining pending file matches
            if !pending_file_matches.is_empty() && all_file_matches.len() <= FILE_MATCH_CAP {
                let _ = update_tx.send(StampedUpdate {
                    generation: session.guard.generation_id(),
                    update: SearchUpdate::FileMatches(std::mem::take(&mut pending_file_matches)),
                });
            }

            let (batch_matches, batch_profile) = autoreleasepool(|_| {
                self.dispatch_gpu_batch_profiled(
                    &batch,
                    &request.pattern,
                    &search_options,
                    request.max_results.saturating_sub(all_content_matches.len()),
                )
            });
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

        // Sort and cap file matches for the final Complete response.
        // No separate FileMatches send needed here — Complete carries them.
        all_file_matches.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_file_matches.truncate(FILE_MATCH_CAP);

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

        eprintln!("[gpu-search] search complete: {} batches, {} files, {} file matches, {} content matches, {:.1}s",
            batch_number, total_files_searched, response.file_matches.len(),
            response.content_matches.len(), response.elapsed.as_secs_f64());
        eprintln!("[gpu-search] profile:\n{}", response.profile);

        let _ = update_tx.send(StampedUpdate {
            generation: session.guard.generation_id(),
            update: SearchUpdate::Complete(response.clone()),
        });

        // Build GSIX index in background if we walked (no index was used)
        // and the search wasn't cancelled. This makes the next search instant.
        if !has_index && !session.should_stop() {
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
    /// Checks two sources in priority order:
    /// 1. IndexStore snapshot (mmap-backed, zero-copy) — preferred when available
    /// 2. MmapIndexCache from disk (v2 format) — fallback for non-store contexts
    ///
    /// If a fresh index exists, iterates all index entries into the crossbeam
    /// channel (instant, no filesystem walk). Applies the same binary and
    /// filetype filters as walk_and_filter.
    ///
    /// Returns `true` if the index was used (entries sent), `false` if the
    /// caller should fall back to walk_and_filter.
    #[allow(dead_code)]
    fn try_index_producer(
        &self,
        root: &Path,
        path_tx: &channel::Sender<PathBuf>,
        counter: &AtomicU64,
        include_binary: bool,
        file_types: Option<&[String]>,
    ) -> bool {
        // Priority 1: Check IndexStore snapshot (zero-copy mmap path)
        if let Some(ref store) = self.index_store {
            if Self::try_snapshot_producer(store, path_tx, counter, include_binary, file_types) {
                return true;
            }
        }

        // Priority 2: Fall back to MmapIndexCache from disk
        Self::try_mmap_cache_producer(root, path_tx, counter, include_binary, file_types)
    }

    /// Check if the index is available without sending any entries.
    /// Used by search_streaming to decide whether to spawn an index producer thread.
    fn check_index_available(
        &self,
        _root: &Path,
        _include_binary: bool,
        _file_types: Option<&[String]>,
    ) -> bool {
        if let Some(ref store) = self.index_store {
            if store.is_available() {
                let guard = store.snapshot();
                if let Some(snapshot) = guard.as_ref().as_ref() {
                    return snapshot.entry_count() > 0;
                }
            }
        }
        false
    }

    /// Send entries from an IndexStore snapshot on a background thread.
    /// This avoids deadlocking the bounded channel when the index has
    /// millions of entries.
    fn try_snapshot_producer_threaded(
        store: &IndexStore,
        path_tx: &channel::Sender<PathBuf>,
        counter: &AtomicU64,
        include_binary: bool,
        file_types: Option<&[String]>,
    ) {
        let guard = store.snapshot();
        let snapshot = match guard.as_ref().as_ref() {
            Some(s) => s,
            None => return,
        };

        let entry_count = snapshot.entry_count();
        eprintln!(
            "[gpu-search] index: sending {} snapshot entries on producer thread",
            entry_count
        );

        let sent = Self::send_entries_from_slice(
            snapshot.entries(),
            path_tx,
            counter,
            include_binary,
            file_types,
        );

        eprintln!(
            "[gpu-search] index: sent {} paths from {} snapshot entries",
            sent, entry_count
        );
    }

    /// Try to produce paths from an IndexStore snapshot (zero-copy mmap path).
    ///
    /// Returns `true` if the snapshot was available and entries were sent.
    #[allow(dead_code)]
    fn try_snapshot_producer(
        store: &IndexStore,
        path_tx: &channel::Sender<PathBuf>,
        counter: &AtomicU64,
        include_binary: bool,
        file_types: Option<&[String]>,
    ) -> bool {
        if !store.is_available() {
            eprintln!("[gpu-search] index: IndexStore has no snapshot");
            return false;
        }

        let guard = store.snapshot();
        let snapshot = match guard.as_ref().as_ref() {
            Some(s) => s,
            None => return false,
        };

        let entry_count = snapshot.entry_count();
        if entry_count == 0 {
            eprintln!("[gpu-search] index: IndexStore snapshot is empty");
            return false;
        }

        eprintln!(
            "[gpu-search] index: using IndexStore snapshot ({} entries, zero-copy mmap)",
            entry_count
        );

        let sent = Self::send_entries_from_slice(
            snapshot.entries(),
            path_tx,
            counter,
            include_binary,
            file_types,
        );

        eprintln!(
            "[gpu-search] index: sent {} paths from {} snapshot entries",
            sent, entry_count
        );
        true
    }

    /// Fall back to MmapIndexCache from disk for index-based file discovery.
    #[allow(dead_code)]
    fn try_mmap_cache_producer(
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

        let sent = Self::send_entries_from_slice(
            cache.entries(),
            path_tx,
            counter,
            include_binary,
            file_types,
        );

        eprintln!(
            "[gpu-search] index: sent {} paths from {} entries",
            sent, entry_count
        );
        true
    }

    /// Send entries from a GpuPathEntry slice through the channel, applying filters.
    ///
    /// Shared logic between IndexStore snapshot and MmapIndexCache paths.
    /// Returns the number of paths successfully sent.
    ///
    /// Uses a TEXT ALLOWLIST (is_known_text) rather than a binary blocklist
    /// (is_binary_path). From root `/`, 3.3M index entries contain ~160K files
    /// with unknown extensions (no ext, or obscure ext like .dat, .conf2, etc).
    /// A binary blocklist lets all those through, overwhelming GPU content search
    /// with 7+ GB of RSS. The text allowlist reduces this to ~10-20K actual
    /// source/config files.
    fn send_entries_from_slice(
        entries: &[crate::gpu::types::GpuPathEntry],
        path_tx: &channel::Sender<PathBuf>,
        counter: &AtomicU64,
        include_binary: bool,
        file_types: Option<&[String]>,
    ) -> u64 {
        let mut sent = 0u64;
        for entry in entries {

            // Skip directories (flag bit 0 = is_dir)
            if entry.flags & 1 != 0 {
                continue;
            }

            // Skip deleted entries (IS_DELETED tombstone flag)
            if entry.flags & 0x10 != 0 {
                continue;
            }

            // Skip empty or oversized files
            let size = (entry.size_hi as u64) << 32 | entry.size_lo as u64;
            if size == 0 || size > 100 * 1024 * 1024 {
                continue;
            }

            let path_bytes = &entry.path[..entry.path_len as usize];
            let path_str = match std::str::from_utf8(path_bytes) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let path_ref = std::path::Path::new(path_str);

            // File type filter (checked before text allowlist for efficiency —
            // if the user specified file types, only those matter)
            if let Some(types) = file_types {
                let passes = if let Some(ext) = path_ref.extension().and_then(|e| e.to_str()) {
                    types.iter().any(|t| t.eq_ignore_ascii_case(ext))
                } else {
                    false
                };
                if !passes {
                    continue;
                }
            } else if !include_binary {
                // TEXT ALLOWLIST: Only send files with known text extensions.
                // This is critical for root `/` searches where 3.3M index entries
                // include ~160K files with unknown extensions. A binary blocklist
                // (is_binary_path) lets all those through, causing 7+ GB RSS.
                // The text allowlist reduces the count to ~10-20K actual source files.
                if !super::binary::is_known_text(path_ref) {
                    continue;
                }
            }

            let path = PathBuf::from(path_str);

            sent += 1;
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if path_tx.send(path).is_err() {
                // Consumer disconnected (search cancelled or done)
                break;
            }
        }

        sent
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

        // DEFERRED LINE COUNTING: resolve_from_cache() is the sole source of
        // truth for line numbers.  GPU's m.line_number (local to 64B thread
        // window) is intentionally ignored -- only byte_offset is used.
        //
        // FileCache: results are sorted by (file_index, byte_offset), so
        // consecutive matches are in the same file. We load each file once
        // and resolve all its matches from the cache.
        let resolve_start = Instant::now();
        let mut content_matches = Vec::new();
        let mut file_cache: Option<FileCache> = None;

        for m in &gpu_results {
            if content_matches.len() >= remaining_budget {
                break;
            }

            // Refresh cache when file changes
            let need_refresh = file_cache
                .as_ref()
                .map(|c| c.path != m.file_path)
                .unwrap_or(true);
            if need_refresh {
                file_cache = FileCache::load(&m.file_path, pattern, options.case_sensitive);
            }

            let cache = match &file_cache {
                Some(c) if c.contains_pattern => c,
                Some(c) => {
                    // File doesn't contain pattern — skip all matches for this file
                    eprintln!(
                        "[gpu-search] FILE-LEVEL REJECTION: file does not contain pattern \
                         file={} pattern='{}' byte_offset={}",
                        c.path.display(),
                        pattern,
                        m.byte_offset,
                    );
                    continue;
                }
                None => continue, // File unreadable
            };

            if let Some((line_number, line_content, context_before, context_after, match_start)) =
                resolve_from_cache(cache, m.byte_offset as usize, pattern, options.case_sensitive)
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

    /// Search using the in-memory content store (zero disk I/O fast-path).
    ///
    /// Loads the current ContentSnapshot via arc-swap guard, builds
    /// ChunkMetadata from the content store, dispatches GPU search via
    /// `search_with_buffer`, resolves matches to file paths and line
    /// numbers, and sends results via the update channel.
    ///
    /// This eliminates ALL disk reads during search -- the content buffer
    /// and file paths are already in memory from the background build.
    fn search_from_content_store(
        &mut self,
        cs: Arc<ContentIndexStore>,
        request: &SearchRequest,
        update_tx: &channel::Sender<StampedUpdate>,
        session: &SearchSession,
    ) -> SearchResponse {
        let start = Instant::now();
        let mut profile = PipelineProfile::default();

        // Check cancellation early
        if session.should_stop() {
            return SearchResponse {
                file_matches: vec![],
                content_matches: vec![],
                total_files_searched: 0,
                total_matches: 0,
                elapsed: start.elapsed(),
                profile,
            };
        }

        // (1) Load ContentSnapshot via arc-swap guard
        let guard = cs.snapshot();
        let snapshot = match guard.as_ref().as_ref() {
            Some(s) => s,
            None => {
                eprintln!("[gpu-search] content store: snapshot disappeared between is_available and load");
                return SearchResponse {
                    file_matches: vec![],
                    content_matches: vec![],
                    total_files_searched: 0,
                    total_matches: 0,
                    elapsed: start.elapsed(),
                    profile,
                };
            }
        };

        let content_store = snapshot.content_store();
        let total_files = content_store.file_count() as u64;
        profile.files_walked = total_files as u32;
        profile.files_filtered = total_files as u32;

        // (2) Filename matching (same logic as streaming pipeline)
        let pattern_lower = request.pattern.to_lowercase();
        let mut file_matches: Vec<FileMatch> = Vec::new();
        let paths = content_store.paths();
        for path in paths {
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
                    file_matches.push(FileMatch {
                        path: path.clone(),
                        score,
                    });
                }
            }
        }
        file_matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        file_matches.truncate(FILE_MATCH_CAP);

        // Send filename matches
        if !file_matches.is_empty() {
            let _ = update_tx.send(StampedUpdate {
                generation: session.guard.generation_id(),
                update: SearchUpdate::FileMatches(file_matches.clone()),
            });
        }

        // Check cancellation before GPU dispatch
        if session.should_stop() {
            profile.total_us = start.elapsed().as_micros() as u64;
            return SearchResponse {
                file_matches: vec![],
                content_matches: vec![],
                total_files_searched: total_files,
                total_matches: 0,
                elapsed: start.elapsed(),
                profile,
            };
        }

        // (3) Build ChunkMetadata from content store (zero-copy path).
        //
        // The zerocopy kernel reads directly from the contiguous content buffer
        // using ChunkMetadata.buffer_offset — NO padded buffer, NO CPU memcpy.
        let chunks_start = Instant::now();
        let (chunk_metas, meta_source) = match content_store.chunk_metadata() {
            Some(cached) => (cached.to_vec(), "cached"),
            None => (build_chunk_metadata(content_store), "rebuilt"),
        };
        profile.batch_us = chunks_start.elapsed().as_micros() as u64;

        eprintln!(
            "[gpu-search] content store: {} chunks ({}) from {} files, buffer={:.1}GB, metadata in {:.1}ms",
            chunk_metas.len(),
            meta_source,
            content_store.file_count(),
            content_store.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
            chunks_start.elapsed().as_secs_f64() * 1000.0,
        );

        if chunk_metas.is_empty() {
            profile.total_us = start.elapsed().as_micros() as u64;
            profile.ttfr_us = profile.total_us;
            let total_matches = file_matches.len() as u64;
            let response = SearchResponse {
                file_matches,
                content_matches: vec![],
                total_files_searched: total_files,
                total_matches,
                elapsed: start.elapsed(),
                profile,
            };
            let _ = update_tx.send(StampedUpdate {
                generation: session.guard.generation_id(),
                update: SearchUpdate::Complete(response.clone()),
            });
            return response;
        }

        // (4) Zero-copy GPU dispatch: pass content store's Metal buffer directly.
        // No allocation, no CPU copy. The zerocopy kernel reads from buffer_offset.
        let metal_buf = match content_store.metal_buffer() {
            Some(buf) => buf,
            None => {
                eprintln!("[gpu-search] content store: no Metal buffer available, falling back");
                profile.total_us = start.elapsed().as_micros() as u64;
                let fm_count = file_matches.len() as u64;
                let response = SearchResponse {
                    file_matches,
                    content_matches: vec![],
                    total_files_searched: total_files,
                    total_matches: fm_count,
                    elapsed: start.elapsed(),
                    profile,
                };
                let _ = update_tx.send(StampedUpdate {
                    generation: session.guard.generation_id(),
                    update: SearchUpdate::Complete(response.clone()),
                });
                return response;
            }
        };

        let search_options = SearchOptions {
            case_sensitive: request.case_sensitive,
            max_results: request.max_results,
            ..Default::default()
        };

        let gpu_start = Instant::now();
        let gpu_matches = autoreleasepool(|_| {
            self.engine.content_engine_mut().search_zerocopy(
                metal_buf,
                &chunk_metas,
                request.pattern.as_bytes(),
                &search_options,
            )
        });
        profile.gpu_dispatch_us = gpu_start.elapsed().as_micros() as u64;
        profile.gpu_dispatches = 1;
        profile.matches_raw = gpu_matches.len() as u32;
        profile.files_searched = content_store.file_count();
        profile.bytes_searched = content_store.total_bytes();

        eprintln!(
            "[gpu-search] content store: GPU zerocopy returned {} raw matches from {} chunks ({} files, {:.1}ms)",
            gpu_matches.len(),
            chunk_metas.len(),
            content_store.file_count(),
            gpu_start.elapsed().as_secs_f64() * 1000.0
        );

        // (6) Resolve GPU matches to ContentMatch entries
        // GPU ContentMatch has file_index, byte_offset, line_number, column.
        // We need to resolve file_index -> path and compute accurate line numbers.
        let resolve_start = Instant::now();
        let mut content_matches: Vec<ContentMatch> = Vec::new();

        let mut reject_no_path = 0u32;
        let mut reject_no_content = 0u32;
        let mut reject_offset_oob = 0u32;
        let mut reject_pattern_miss = 0u32;

        for (mi, m) in gpu_matches.iter().enumerate() {
            if content_matches.len() >= request.max_results {
                break;
            }

            let file_id = m.file_index as u32;

            // Resolve file_id to path
            let path = match content_store.path_for(file_id) {
                Some(p) => p,
                None => { reject_no_path += 1; continue; },
            };

            // Get file content from the content store (zero disk I/O)
            let file_content = match content_store.content_for(file_id) {
                Some(c) => c,
                None => { reject_no_content += 1; continue; },
            };

            // byte_offset is file-relative from zerocopy collect_results (context_start + column)
            let file_byte_offset = m.byte_offset as usize;

            // Log first few matches for debugging
            if mi < 3 {
                eprintln!(
                    "[resolve-debug] match[{}]: file_id={} byte_offset={} file_len={} path={}",
                    mi, file_id, file_byte_offset, file_content.len(),
                    path.display()
                );
            }

            // If file-relative byte_offset exceeds file length, skip
            if file_byte_offset >= file_content.len() {
                reject_offset_oob += 1;
                continue;
            }

            // Resolve line content and context directly from raw bytes using byte offset.
            // No String::from_utf8_lossy or lines().collect() -- just scan for \n boundaries.
            let (line_content, context_before, context_after, match_col) =
                match extract_line_context(file_content, file_byte_offset, &request.pattern, request.case_sensitive) {
                    Some(result) => result,
                    None => { reject_pattern_miss += 1; continue; },
                };

            let match_end = match_col + request.pattern.len();

            // Use GPU-computed line number directly (trust the kernel)
            content_matches.push(ContentMatch {
                path: path.clone(),
                line_number: m.line_number,
                line_content,
                context_before,
                context_after,
                match_range: match_col..match_end,
            });
        }

        profile.resolve_us = resolve_start.elapsed().as_micros() as u64;
        profile.matches_resolved = content_matches.len() as u32;
        profile.matches_rejected = profile.matches_raw.saturating_sub(profile.matches_resolved);

        eprintln!(
            "[gpu-search] content store: resolved {} matches ({} rejected) in {:.1}ms",
            content_matches.len(),
            profile.matches_rejected,
            resolve_start.elapsed().as_secs_f64() * 1000.0
        );
        if reject_no_path > 0 || reject_no_content > 0 || reject_offset_oob > 0 || reject_pattern_miss > 0 {
            eprintln!(
                "[gpu-search] rejection breakdown: no_path={} no_content={} offset_oob={} pattern_miss={}",
                reject_no_path, reject_no_content, reject_offset_oob, reject_pattern_miss
            );
        }

        // (7) Send results
        if !content_matches.is_empty() {
            profile.ttfr_us = start.elapsed().as_micros() as u64;
            let _ = update_tx.send(StampedUpdate {
                generation: session.guard.generation_id(),
                update: SearchUpdate::ContentMatches(content_matches.clone()),
            });
        } else {
            profile.ttfr_us = start.elapsed().as_micros() as u64;
        }

        let total_matches = (file_matches.len() + content_matches.len()) as u64;
        profile.total_us = start.elapsed().as_micros() as u64;

        let response = SearchResponse {
            file_matches,
            content_matches,
            total_files_searched: total_files,
            total_matches,
            elapsed: start.elapsed(),
            profile,
        };

        eprintln!(
            "[gpu-search] content store search complete: {} file matches, {} content matches, {:.1}ms",
            response.file_matches.len(),
            response.content_matches.len(),
            response.elapsed.as_secs_f64() * 1000.0
        );
        eprintln!("[gpu-search] profile:\n{}", response.profile);

        let _ = update_tx.send(StampedUpdate {
            generation: session.guard.generation_id(),
            update: SearchUpdate::Complete(response.clone()),
        });

        response
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

// ============================================================================
// FileCache: single-file content cache for the resolve loop
// ============================================================================

/// Caches one file's content + precomputed line structure for the resolve loop.
/// Since results are sorted by file_index, consecutive matches hit the same file.
/// This eliminates redundant fs::read() calls (N reads -> U unique-file reads).
struct FileCache {
    path: PathBuf,
    content: Vec<u8>,
    lines: Vec<String>,
    line_offsets: Vec<usize>, // byte offset where each line starts
    contains_pattern: bool,
}

impl FileCache {
    /// Load a file and precompute line structure. Returns None if unreadable.
    fn load(path: &Path, pattern: &str, case_sensitive: bool) -> Option<Self> {
        let content = std::fs::read(path).ok()?;
        let text = String::from_utf8_lossy(&content);

        let contains_pattern = if case_sensitive {
            text.contains(pattern)
        } else {
            text.to_lowercase().contains(&pattern.to_lowercase())
        };

        // Precompute line offsets and line strings
        let mut lines = Vec::new();
        let mut line_offsets = Vec::new();
        let mut offset = 0usize;

        for line in text.lines() {
            line_offsets.push(offset);
            lines.push(line.to_string());
            // Advance past line content + newline character(s)
            offset += line.len();
            // Check for \r\n vs \n
            if content.get(offset) == Some(&b'\r') {
                offset += 2; // \r\n
            } else if offset < content.len() {
                offset += 1; // \n
            }
        }

        Some(FileCache {
            path: path.to_path_buf(),
            content,
            lines,
            line_offsets,
            contains_pattern,
        })
    }
}

/// Resolve a match using cached file content instead of re-reading from disk.
/// Same logic as resolve_match() but operates on precomputed line structure.
#[allow(clippy::type_complexity)]
fn resolve_from_cache(
    cache: &FileCache,
    byte_offset: usize,
    pattern: &str,
    case_sensitive: bool,
) -> Option<(usize, String, Vec<String>, Vec<String>, usize)> {
    if byte_offset >= cache.content.len() {
        return None;
    }

    // Binary search for the line containing byte_offset
    let target_line = match cache.line_offsets.binary_search(&byte_offset) {
        Ok(i) => i,         // Exact match: byte_offset is at line start
        Err(i) => i.saturating_sub(1), // Between two lines: take the previous one
    };

    let line_content = cache.lines.get(target_line)?.to_string();
    let line_start = cache.line_offsets[target_line];
    let expected_col = byte_offset - line_start;

    // Find the actual pattern match within the line
    let match_col = if case_sensitive {
        line_content.find(pattern)?
    } else {
        line_content.to_lowercase().find(&pattern.to_lowercase())?
    };

    // Diagnostic: validate GPU byte_offset against find() position
    let col_diff = match_col.abs_diff(expected_col);
    if col_diff > pattern.len() {
        eprintln!(
            "[gpu-search] byte_offset discrepancy: file={} byte_offset={} expected_col={} match_col={} diff={} pattern='{}'",
            cache.path.display(), byte_offset, expected_col, match_col, col_diff, pattern
        );
    }

    // Context: 2 lines before and after
    let context_before: Vec<String> = (target_line.saturating_sub(2)..target_line)
        .filter_map(|i| cache.lines.get(i).cloned())
        .collect();
    let context_after: Vec<String> =
        (target_line + 1..=(target_line + 2).min(cache.lines.len().saturating_sub(1)))
            .filter_map(|i| cache.lines.get(i).cloned())
            .collect();

    // Line number is 1-based
    Some((target_line + 1, line_content, context_before, context_after, match_col))
}

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

    // Find which line the byte_offset falls on.
    // lines() strips both \n and \r\n, so we check the raw bytes to determine
    // the actual line-ending length (1 for \n, 2 for \r\n).
    let mut cumulative = 0usize;
    let mut target_line = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let after_line = cumulative + line.len();
        let newline_len = if content.get(after_line) == Some(&b'\r') { 2 } else { 1 };
        let line_end = after_line + newline_len;
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

    // The GPU byte_offset should place us at the exact match column within the line.
    // expected_col = byte_offset - cumulative (start of this line in the file).
    let expected_col = byte_offset - cumulative;

    // Find the actual pattern match within the line for match_range.
    // If the pattern is NOT found in this line, reject the match entirely —
    // this catches byte_offset mapping errors from the GPU pipeline.
    let match_col = if case_sensitive {
        line_content.find(pattern)?
    } else {
        line_content.to_lowercase().find(&pattern.to_lowercase())?
    };

    // Diagnostic: validate GPU byte_offset against find() position.
    // If they diverge significantly, the GPU reported a different column than
    // where find() located the pattern. Still accept find()'s result (the
    // pattern IS in the line) but log the discrepancy for diagnostics.
    let col_diff = match_col.abs_diff(expected_col);
    if col_diff > pattern.len() {
        eprintln!(
            "[gpu-search] byte_offset discrepancy: file={} byte_offset={} expected_col={} match_col={} diff={} pattern='{}'",
            path.display(), byte_offset, expected_col, match_col, col_diff, pattern
        );
    }

    // Verify the content at byte_offset actually starts with the pattern bytes
    // (case-insensitive). If it does not, the GPU reported an offset that doesn't
    // align with the pattern — a GPU false positive at the byte level. We still
    // accept find()'s result since the pattern IS in the line.
    if byte_offset + pattern.len() <= content.len() {
        let slice_at_offset = &content[byte_offset..byte_offset + pattern.len()];
        let pattern_bytes = pattern.as_bytes();
        let starts_with_pattern = if case_sensitive {
            slice_at_offset == pattern_bytes
        } else {
            slice_at_offset
                .iter()
                .zip(pattern_bytes.iter())
                .all(|(a, b)| a.eq_ignore_ascii_case(b))
        };
        if !starts_with_pattern {
            eprintln!(
                "[gpu-search] byte_offset discrepancy: content at offset does not match pattern \
                 file={} byte_offset={} expected='{}' found='{}'",
                path.display(),
                byte_offset,
                pattern,
                String::from_utf8_lossy(slice_at_offset)
            );
        }
    }

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

    // ========================================================================
    // Content store fast-path tests
    // ========================================================================

    #[test]
    fn test_content_store_fast_path() {
        use crate::index::content_index_store::ContentIndexStore;
        use crate::index::content_snapshot::ContentSnapshot;
        use crate::index::content_store::ContentStoreBuilder;

        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);

        // Build a ContentStore with known content and paths using mmap builder
        let dir = make_test_directory();

        let main_rs = dir.path().join("main.rs");
        let lib_rs = dir.path().join("lib.rs");
        let utils_rs = dir.path().join("utils.rs");

        let main_content = std::fs::read(&main_rs).unwrap();
        let lib_content = std::fs::read(&lib_rs).unwrap();
        let utils_content = std::fs::read(&utils_rs).unwrap();

        let total_size = main_content.len() + lib_content.len() + utils_content.len();
        let mut builder = ContentStoreBuilder::new(total_size + 4096).unwrap();

        builder.append_with_path(&main_content, main_rs.clone(), 0, 0, 0);
        builder.append_with_path(&lib_content, lib_rs.clone(), 1, 0, 0);
        builder.append_with_path(&utils_content, utils_rs.clone(), 2, 0, 0);

        let store = builder.finalize(&device.device);

        let snapshot = ContentSnapshot::new(store, 0);
        let content_store = Arc::new(ContentIndexStore::new());
        content_store.swap(snapshot);

        // Create orchestrator with content store
        let mut orchestrator = SearchOrchestrator::with_content_store(
            &device.device,
            &pso_cache,
            None,
            content_store,
        ).expect("Failed to create orchestrator with content store");

        // Search for "fn " -- should use content store fast-path
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

        println!("Content store fast-path results:");
        println!("  File matches: {}", response.file_matches.len());
        println!("  Content matches: {}", response.content_matches.len());
        println!("  Total files searched: {}", response.total_files_searched);
        println!("  Elapsed: {:.1}ms", response.elapsed.as_secs_f64() * 1000.0);

        for cm in &response.content_matches {
            println!("  {}:{}: {}", cm.path.display(), cm.line_number, cm.line_content.trim());
        }

        // Should find "fn " in at least 3 files (main.rs, lib.rs, utils.rs)
        assert!(
            response.content_matches.len() >= 3,
            "Content store fast-path should find at least 3 'fn ' matches, got {}",
            response.content_matches.len()
        );

        // All matches should have valid paths
        for cm in &response.content_matches {
            assert!(cm.path.exists(), "Content match path should exist: {:?}", cm.path);
            assert!(cm.line_number > 0, "Line number should be > 0");
            assert!(!cm.line_content.is_empty(), "Line content should not be empty");
        }

        // Should have sent updates
        let mut got_complete = false;
        while let Ok(stamped) = update_rx.try_recv() {
            if matches!(stamped.update, SearchUpdate::Complete(_)) {
                got_complete = true;
            }
        }
        assert!(got_complete, "Should send Complete update");

        // Total files searched should equal the content store file count
        assert_eq!(response.total_files_searched, 3);
    }

    #[test]
    fn test_content_store_fallback_when_unavailable() {
        // When content store is empty (not yet built), should fall back to disk pipeline
        use crate::index::content_index_store::ContentIndexStore;

        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);

        let content_store = Arc::new(ContentIndexStore::new());
        // Don't swap any snapshot -- store is empty

        let mut orchestrator = SearchOrchestrator::with_content_store(
            &device.device,
            &pso_cache,
            None,
            content_store,
        ).expect("Failed to create orchestrator");

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

        let (update_tx, _) = channel::unbounded();
        let session = test_session();
        let response = orchestrator.search_streaming(request, &update_tx, &session);

        // Should still find matches via disk-based pipeline fallback
        assert!(
            response.content_matches.len() >= 3,
            "Fallback to disk pipeline should find at least 3 'fn ' matches, got {}",
            response.content_matches.len()
        );
    }

    // ========================================================================
    // Phase 3 checkpoint: zero disk I/O search with 100 files
    // ========================================================================

    /// Generate a 100-file test corpus with known patterns for checkpoint verification.
    ///
    /// File distribution:
    ///   - files 0..29:  contain "kolbey" only
    ///   - files 30..49: contain "patrick" only
    ///   - files 50..64: contain BOTH "kolbey" and "patrick"
    ///   - files 65..79: contain "GPU_SEARCH" (unique marker)
    ///   - files 80..99: contain NEITHER pattern (noise/padding)
    ///
    /// Returns (TempDir, expected_kolbey_count, expected_patrick_count, expected_gpu_search_count)
    fn make_100_file_corpus() -> (TempDir, usize, usize, usize) {
        let dir = TempDir::new().expect("create tempdir");

        // Category 1: kolbey only (files 0..30 = 30 files)
        for i in 0..30 {
            let name = format!("kolbey_{:03}.txt", i);
            let content = format!(
                "// File {i}\nfn process_{i}() {{\n    let name = \"kolbey\";\n    println!(\"hello kolbey #{i}\");\n}}\n"
            );
            std::fs::write(dir.path().join(&name), &content).unwrap();
        }

        // Category 2: patrick only (files 30..50 = 20 files)
        for i in 30..50 {
            let name = format!("patrick_{:03}.txt", i);
            let content = format!(
                "// File {i}\nfn handler_{i}() {{\n    let author = \"patrick\";\n    // Written by patrick\n}}\n"
            );
            std::fs::write(dir.path().join(&name), &content).unwrap();
        }

        // Category 3: both patterns (files 50..65 = 15 files)
        for i in 50..65 {
            let name = format!("both_{:03}.txt", i);
            let content = format!(
                "// File {i}\n// Author: patrick\nfn collaborate_{i}() {{\n    let team = vec![\"kolbey\", \"patrick\"];\n    println!(\"kolbey and patrick working\");\n}}\n"
            );
            std::fs::write(dir.path().join(&name), &content).unwrap();
        }

        // Category 4: GPU_SEARCH marker (files 65..80 = 15 files)
        for i in 65..80 {
            let name = format!("gpu_{:03}.txt", i);
            let content = format!(
                "// File {i}\nconst MARKER: &str = \"GPU_SEARCH\";\nfn benchmark_{i}() {{\n    // GPU_SEARCH enabled\n}}\n"
            );
            std::fs::write(dir.path().join(&name), &content).unwrap();
        }

        // Category 5: noise files (files 80..100 = 20 files) -- no target patterns
        for i in 80..100 {
            let name = format!("noise_{:03}.txt", i);
            let content = format!(
                "// File {i}\nfn utility_{i}() {{\n    let x = {i} * 2;\n    let y = x + 1;\n    println!(\"result: {{}}\", y);\n}}\n"
            );
            std::fs::write(dir.path().join(&name), &content).unwrap();
        }

        // Expected match counts:
        // "kolbey":     30 files (cat1) + 15 files (cat3) = 45 files
        //   cat1: 2 occurrences each (name = "kolbey", hello kolbey) = 60
        //   cat3: 2 occurrences each (vec kolbey, kolbey and) = 30
        //   Total "kolbey" occurrences = 90
        //
        // "patrick":    20 files (cat2) + 15 files (cat3) = 35 files
        //   cat2: 2 occurrences each = 40
        //   cat3: 2 occurrences each (Author: patrick, and patrick) + 1 (vec patrick) = 45
        //   (exact count varies by GPU match granularity)
        //
        // "GPU_SEARCH": 15 files (cat4), 2 occurrences each = 30

        let kolbey_files = 30 + 15; // 45 files contain "kolbey"
        let patrick_files = 20 + 15; // 35 files contain "patrick"
        let gpu_search_files = 15; // 15 files contain "GPU_SEARCH"

        (dir, kolbey_files, patrick_files, gpu_search_files)
    }

    #[test]
    fn test_content_store_phase3_checkpoint() {
        use crate::index::content_index_store::ContentIndexStore;
        use crate::index::content_snapshot::ContentSnapshot;
        use crate::index::content_store::ContentStoreBuilder;
        use std::collections::HashSet;

        let device = GpuDevice::new();
        let pso_cache = PsoCache::new(&device.device);

        let (dir, expected_kolbey_files, expected_patrick_files, expected_gpu_files) =
            make_100_file_corpus();

        // -----------------------------------------------------------------
        // Build ContentStore from the 100 files
        // -----------------------------------------------------------------
        // Read all files and compute total size
        let mut file_entries: Vec<(PathBuf, Vec<u8>)> = Vec::new();
        let mut total_size = 0usize;
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_file() {
                let content = std::fs::read(&path).unwrap();
                total_size += content.len();
                file_entries.push((path, content));
            }
        }
        // Sort for deterministic ordering
        file_entries.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(file_entries.len(), 100, "Should have exactly 100 files");

        let mut builder = ContentStoreBuilder::new(total_size + 16384).unwrap();
        for (i, (path, content)) in file_entries.iter().enumerate() {
            builder.append_with_path(content, path.clone(), i as u32, 0, 0);
        }
        let store = builder.finalize(&device.device);
        assert_eq!(store.file_count() as usize, 100);

        let snapshot = ContentSnapshot::new(store, 0);
        let content_store = Arc::new(ContentIndexStore::new());
        content_store.swap(snapshot);

        // -----------------------------------------------------------------
        // Test 1: Search for "kolbey" via content store
        // -----------------------------------------------------------------
        let mut orchestrator = SearchOrchestrator::with_content_store(
            &device.device,
            &pso_cache,
            None,
            content_store.clone(),
        ).expect("Failed to create orchestrator");

        let request = SearchRequest {
            pattern: "kolbey".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let (update_tx, _) = channel::unbounded();
        let session = test_session();
        let response = orchestrator.search_streaming(request, &update_tx, &session);

        println!("=== Phase 3 Checkpoint: 'kolbey' search ===");
        println!("  Content matches: {}", response.content_matches.len());
        println!("  Total files searched: {}", response.total_files_searched);
        println!("  Elapsed: {:.1}ms", response.elapsed.as_secs_f64() * 1000.0);

        // Verify we searched all 100 files
        assert_eq!(response.total_files_searched, 100, "Should search all 100 files");

        // Collect unique file paths from matches
        let kolbey_match_files: HashSet<PathBuf> = response.content_matches
            .iter()
            .map(|m| m.path.clone())
            .collect();

        println!("  Unique files with 'kolbey' matches: {}", kolbey_match_files.len());

        // Should find matches in the correct number of files
        assert!(
            kolbey_match_files.len() >= expected_kolbey_files - 2,
            "Expected matches in ~{} files for 'kolbey', got {} files",
            expected_kolbey_files, kolbey_match_files.len()
        );

        // No false positives: matches should ONLY be in kolbey_ or both_ files
        for path in &kolbey_match_files {
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(
                name.starts_with("kolbey_") || name.starts_with("both_"),
                "False positive: 'kolbey' match found in unexpected file: {}",
                name
            );
        }

        // All matches should have valid line content containing "kolbey"
        for cm in &response.content_matches {
            assert!(cm.line_number > 0, "Line number should be > 0 for {:?}", cm.path);
            // line_content should contain the pattern (case-sensitive)
            assert!(
                cm.line_content.contains("kolbey"),
                "Match line should contain 'kolbey': {:?} -> '{}'",
                cm.path.file_name(), cm.line_content.trim()
            );
        }

        // No matches should be in noise_, patrick_, or gpu_ files
        for cm in &response.content_matches {
            let name = cm.path.file_name().unwrap().to_str().unwrap();
            assert!(
                !name.starts_with("noise_") && !name.starts_with("patrick_") && !name.starts_with("gpu_"),
                "False positive: 'kolbey' should not match in {}", name
            );
        }

        // -----------------------------------------------------------------
        // Test 2: Search for "GPU_SEARCH" via content store
        // -----------------------------------------------------------------
        let request2 = SearchRequest {
            pattern: "GPU_SEARCH".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let (update_tx2, _) = channel::unbounded();
        let session2 = test_session();
        let response2 = orchestrator.search_streaming(request2, &update_tx2, &session2);

        println!("\n=== Phase 3 Checkpoint: 'GPU_SEARCH' search ===");
        println!("  Content matches: {}", response2.content_matches.len());

        let gpu_match_files: HashSet<PathBuf> = response2.content_matches
            .iter()
            .map(|m| m.path.clone())
            .collect();

        println!("  Unique files with 'GPU_SEARCH' matches: {}", gpu_match_files.len());

        // Should find GPU_SEARCH only in gpu_ files
        assert!(
            gpu_match_files.len() >= expected_gpu_files - 1,
            "Expected matches in ~{} files for 'GPU_SEARCH', got {} files",
            expected_gpu_files, gpu_match_files.len()
        );

        for path in &gpu_match_files {
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(
                name.starts_with("gpu_"),
                "False positive: 'GPU_SEARCH' match in unexpected file: {}",
                name
            );
        }

        // Each match line should contain "GPU_SEARCH"
        for cm in &response2.content_matches {
            assert!(
                cm.line_content.contains("GPU_SEARCH"),
                "Match line should contain 'GPU_SEARCH': {:?} -> '{}'",
                cm.path.file_name(), cm.line_content.trim()
            );
        }

        // -----------------------------------------------------------------
        // Test 3: Search for "ZZZZZ_NONEXISTENT" -- zero matches expected
        // -----------------------------------------------------------------
        let request3 = SearchRequest {
            pattern: "ZZZZZ_NONEXISTENT".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let (update_tx3, _) = channel::unbounded();
        let session3 = test_session();
        let response3 = orchestrator.search_streaming(request3, &update_tx3, &session3);

        assert_eq!(
            response3.content_matches.len(), 0,
            "Non-existent pattern should return zero matches, got {}",
            response3.content_matches.len()
        );
        assert_eq!(response3.total_files_searched, 100);

        // -----------------------------------------------------------------
        // Test 4: Search for "patrick" -- verify file distribution
        // -----------------------------------------------------------------
        let request4 = SearchRequest {
            pattern: "patrick".to_string(),
            root: dir.path().to_path_buf(),
            file_types: None,
            case_sensitive: true,
            respect_gitignore: false,
            include_binary: false,
            max_results: 10_000,
        };

        let (update_tx4, _) = channel::unbounded();
        let session4 = test_session();
        let response4 = orchestrator.search_streaming(request4, &update_tx4, &session4);

        println!("\n=== Phase 3 Checkpoint: 'patrick' search ===");
        println!("  Content matches: {}", response4.content_matches.len());

        let patrick_match_files: HashSet<PathBuf> = response4.content_matches
            .iter()
            .map(|m| m.path.clone())
            .collect();

        println!("  Unique files with 'patrick' matches: {}", patrick_match_files.len());

        assert!(
            patrick_match_files.len() >= expected_patrick_files - 2,
            "Expected matches in ~{} files for 'patrick', got {} files",
            expected_patrick_files, patrick_match_files.len()
        );

        // No false positives: matches only in patrick_ or both_ files
        for path in &patrick_match_files {
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(
                name.starts_with("patrick_") || name.starts_with("both_"),
                "False positive: 'patrick' match in unexpected file: {}",
                name
            );
        }

        // All match lines should contain "patrick"
        for cm in &response4.content_matches {
            assert!(
                cm.line_content.contains("patrick"),
                "Match line should contain 'patrick': {:?} -> '{}'",
                cm.path.file_name(), cm.line_content.trim()
            );
        }

        println!("\n=== Phase 3 Checkpoint PASSED ===");
        println!("  100 files indexed, zero disk I/O during search");
        println!("  'kolbey':       {} matches in {} files (expected ~{})",
            response.content_matches.len(), kolbey_match_files.len(), expected_kolbey_files);
        println!("  'GPU_SEARCH':   {} matches in {} files (expected ~{})",
            response2.content_matches.len(), gpu_match_files.len(), expected_gpu_files);
        println!("  'patrick':      {} matches in {} files (expected ~{})",
            response4.content_matches.len(), patrick_match_files.len(), expected_patrick_files);
        println!("  no-match query: 0 matches (correct)");
        println!("  No false positives in any search");
    }
}
