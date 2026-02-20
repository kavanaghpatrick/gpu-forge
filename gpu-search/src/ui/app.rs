//! egui application shell for gpu-search.
//!
//! Wires the GPU engine -> search orchestrator -> UI. The orchestrator runs on a
//! dedicated background thread with its own `MTLCommandQueue`. Communication uses
//! crossbeam channels: UI sends `SearchRequest`, orchestrator sends `SearchUpdate`
//! back. UI polls `try_recv` every frame to update results progressively.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::ui::watchdog::Watchdog;

use crossbeam_channel::{Receiver, Sender};
use eframe::egui;

use crate::gpu::device::GpuDevice;
use crate::gpu::pipeline::PsoCache;
use crate::index::content_daemon::ContentDaemon;
use crate::index::content_index_store::ContentIndexStore;
use crate::index::daemon::IndexDaemon;
use crate::index::exclude::{default_excludes, load_config, merge_with_config};
use crate::index::store::IndexStore;
use crate::search::cancel::{cancellation_pair, CancellationHandle, SearchGeneration, SearchSession};
use crate::search::orchestrator::SearchOrchestrator;
use crate::search::types::{ContentMatch, FileMatch, SearchRequest, SearchUpdate, StampedUpdate};

use super::actions;
use super::filters::FilterBar;
use super::highlight::SyntaxHighlighter;
use super::keybinds::{self, KeyAction};
use super::path_utils::abbreviate_path;
use super::results_list::{ContentGroup, FlatRowModel, ResultsList};
use super::search_bar::{SearchBar, SearchMode};
use super::status_bar::{IndexState, StatusBar};
use super::theme;

/// Message from UI -> orchestrator background thread.
enum OrchestratorCommand {
    /// Execute a new search, cancelling any in-progress search.
    /// Carries a `SearchSession` for cooperative cancellation + generation tracking.
    Search(SearchRequest, SearchSession),
    /// Shut down the background thread.
    Shutdown,
}

/// Main application state for the gpu-search UI.
pub struct GpuSearchApp {
    /// Current text in the search input field.
    pub search_input: String,

    /// Filename matches from the most recent search.
    pub file_matches: Vec<FileMatch>,

    /// Content matches from the most recent search.
    pub content_matches: Vec<ContentMatch>,

    /// Content matches grouped by file path for grouped rendering.
    pub content_groups: Vec<ContentGroup>,

    /// Map from file path to index in `content_groups` for O(1) group lookup.
    group_index_map: HashMap<PathBuf, usize>,

    /// Index into `content_matches` up to which grouping has been computed.
    /// Enables incremental grouping -- only new matches are processed.
    last_grouped_index: usize,

    /// Index of the currently selected result in the combined list.
    pub selected_index: usize,

    /// Whether a search is currently in progress.
    pub is_searching: bool,

    /// Search bar with 30ms debounce.
    search_bar: SearchBar,

    /// Scrollable results list with virtual scroll.
    results_list: ResultsList,

    /// Status bar (match count, elapsed, root, filter count).
    status_bar: StatusBar,

    /// Filter bar with dismissible pills.
    filter_bar: FilterBar,

    /// Syntax highlighter (cached per extension).
    highlighter: SyntaxHighlighter,

    /// Flat row model for grouped virtual scroll rendering.
    flat_row_model: FlatRowModel,

    /// Channel to send commands to the orchestrator background thread.
    cmd_tx: Sender<OrchestratorCommand>,

    /// Channel to receive generation-stamped search updates from the orchestrator.
    update_rx: Receiver<StampedUpdate>,

    /// Root directory for searches (current working dir by default).
    search_root: PathBuf,

    /// Last search elapsed time.
    last_elapsed: Duration,

    /// Generation counter for search sessions (monotonically increasing).
    /// Used to detect and discard stale results from superseded searches.
    search_generation: SearchGeneration,

    /// Handle to cancel the currently in-flight search.
    /// Set on each `dispatch_search`, cancelled when a new search starts.
    current_cancel_handle: Option<CancellationHandle>,

    /// When true, the next valid update in `poll_updates()` will clear old
    /// results before applying the new ones. This keeps the previous search's
    /// results visible until the new search starts producing output, avoiding
    /// a flash of "Searching..." between keystrokes.
    pending_result_clear: bool,

    /// The query for which currently displayed results are valid.
    /// Used for client-side refinement filtering: when the new query extends
    /// this one (user typed more characters), existing results are filtered
    /// immediately instead of showing stale data while the background search
    /// restarts from scratch.
    last_displayed_query: String,

    /// Current state of the persistent file index lifecycle.
    pub index_state: IndexState,

    /// Handle to the background orchestrator thread.
    _bg_thread: Option<thread::JoinHandle<()>>,

    /// Persistent index daemon (loads/builds global.idx, runs FSEvents + IndexWriter).
    /// Kept alive for the lifetime of the app.
    _index_daemon: Option<IndexDaemon>,

    /// Content index daemon (builds in-memory content store for zero-disk-I/O search).
    /// Kept alive for the lifetime of the app; publishes snapshots to ContentIndexStore.
    _content_daemon: Option<ContentDaemon>,

    /// Diagnostic watchdog (independent thread logging to stderr).
    _watchdog: Option<Watchdog>,

    /// Shared diagnostic state (atomics for lock-free cross-thread access).
    diag: Arc<crate::ui::watchdog::DiagState>,

    /// Frame counter for heartbeat diagnostics.
    frame_count: u64,
    /// Last heartbeat log time.
    last_heartbeat: Instant,
    /// App creation time.
    app_start: Instant,
    /// When the previous update() call ended. Used to measure the "frame gap"
    /// — time eframe spends outside update() for rendering, presentation,
    /// and event processing. A large gap reveals drawable stalls or GPU contention.
    last_update_end: Instant,
    /// Auto-search query from --search CLI argument, dispatched on first update.
    auto_search: Option<String>,
    /// Second auto-search: simulates rapid typing to reproduce the freeze.
    auto_search_2: Option<String>,
    /// Counter for simulated typing (dispatches chars one by one).
    auto_search_2_idx: usize,
    /// When to dispatch the next simulated keystroke.
    auto_search_2_next: Option<Instant>,
}

impl Default for GpuSearchApp {
    fn default() -> Self {
        // Create channels (StampedUpdate for generation-stamped messages)
        let (cmd_tx, _cmd_rx) = crossbeam_channel::unbounded();
        let (_update_tx, update_rx) = crossbeam_channel::unbounded::<StampedUpdate>();

        Self {
            search_input: String::new(),
            file_matches: Vec::new(),
            content_matches: Vec::new(),
            content_groups: Vec::new(),
            group_index_map: HashMap::new(),
            last_grouped_index: 0,
            selected_index: 0,
            is_searching: false,
            search_bar: SearchBar::default(),
            results_list: ResultsList::new(),
            status_bar: StatusBar::default(),
            filter_bar: FilterBar::new(),
            highlighter: SyntaxHighlighter::new(),
            flat_row_model: FlatRowModel::default(),
            cmd_tx,
            update_rx,
            search_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            last_elapsed: Duration::ZERO,
            search_generation: SearchGeneration::new(),
            current_cancel_handle: None,
            pending_result_clear: false,
            last_displayed_query: String::new(),
            index_state: IndexState::Loading,
            _bg_thread: None,
            _index_daemon: None,
            _content_daemon: None,
            _watchdog: None,
            diag: Arc::new(crate::ui::watchdog::DiagState::default()),
            frame_count: 0,
            last_heartbeat: Instant::now(),
            app_start: Instant::now(),
            last_update_end: Instant::now(),
            auto_search: None,
            auto_search_2: None,
            auto_search_2_idx: 0,
            auto_search_2_next: None,
        }
    }
}

impl GpuSearchApp {
    /// Create a new GpuSearchApp, initializing the GPU device, pipeline,
    /// and orchestrator on a background thread.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let search_root = parse_root_arg()
            .unwrap_or_else(|| PathBuf::from("/"));

        // Start the persistent index daemon (loads global.idx or builds in background)
        let index_store = Arc::new(IndexStore::new());
        let index_daemon = match IndexDaemon::start(Arc::clone(&index_store), search_root.clone()) {
            Ok(daemon) => {
                eprintln!("[gpu-search] IndexDaemon started");
                Some(daemon)
            }
            Err(e) => {
                eprintln!("[gpu-search] IndexDaemon failed to start: {}, falling back to walk", e);
                None
            }
        };

        // Start the content index daemon (builds in-memory content store for
        // zero-disk-I/O search). Uses the same exclude config as IndexDaemon.
        let content_store = Arc::new(ContentIndexStore::new());
        let content_daemon = {
            use objc2_metal::MTLCreateSystemDefaultDevice;
            let excludes = Arc::new(merge_with_config(default_excludes(), load_config()));
            match MTLCreateSystemDefaultDevice() {
                Some(device) => {
                    let daemon = ContentDaemon::start(
                        Arc::clone(&content_store),
                        device,
                        excludes,
                        search_root.clone(),
                    );
                    eprintln!("[gpu-search] ContentDaemon started (building content index in background)");
                    Some(daemon)
                }
                None => {
                    eprintln!("[gpu-search] ContentDaemon: no Metal device, content index disabled");
                    None
                }
            }
        };

        // Channels: UI -> orchestrator commands, orchestrator -> UI updates
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<OrchestratorCommand>();
        let (update_tx, update_rx) = crossbeam_channel::unbounded::<StampedUpdate>();

        // Start diagnostic watchdog (independent thread logging to stderr)
        let watchdog = Watchdog::start();
        let diag = Arc::clone(&watchdog.state);

        // Spawn background orchestrator thread with its own GPU resources.
        // Pass the IndexStore so the orchestrator can use mmap snapshots.
        // Pass ContentIndexStore so the orchestrator can use in-memory content search.
        // Pass DiagState so the orchestrator can update watchdog metrics.
        // NOTE: We intentionally do NOT pass egui::Context to the BG thread.
        // The BG thread must never call request_repaint() — doing so floods the
        // event loop with immediate repaint requests, causing wgpu to produce
        // frames faster than the compositor returns drawables to the
        // CAMetalLayer pool. Once all 3 drawables are consumed, nextDrawable()
        // blocks permanently, freezing the UI. Instead, update() self-drives
        // repaints at 60fps via request_repaint_after(16ms).
        let store_for_thread = Arc::clone(&index_store);
        let content_store_for_thread = Arc::clone(&content_store);
        let diag_for_thread = Arc::clone(&diag);
        let bg_thread = thread::spawn(move || {
            orchestrator_thread(cmd_rx, update_tx, store_for_thread, content_store_for_thread, diag_for_thread);
        });

        let mut status_bar = StatusBar::default();
        status_bar.set_search_root(&search_root);

        // Apply Tokyo Night theme ONCE at startup (not every frame).
        // Setting the style in update() causes ctx to mark itself as changed,
        // triggering a repaint every frame → 120fps idle rendering → 26MB/s
        // memory growth from wgpu staging buffers that never get reclaimed.
        let mut style = (*cc.egui_ctx.style()).clone();
        style.visuals.override_text_color = Some(theme::TEXT_PRIMARY);
        style.visuals.panel_fill = theme::BG_BASE;
        style.visuals.window_fill = theme::BG_BASE;
        cc.egui_ctx.set_style(style);

        Self {
            search_input: String::new(),
            file_matches: Vec::new(),
            content_matches: Vec::new(),
            content_groups: Vec::new(),
            group_index_map: HashMap::new(),
            last_grouped_index: 0,
            selected_index: 0,
            is_searching: false,
            search_bar: SearchBar::default(),
            results_list: ResultsList::new(),
            status_bar,
            filter_bar: FilterBar::new(),
            highlighter: SyntaxHighlighter::new(),
            flat_row_model: FlatRowModel::default(),
            cmd_tx,
            update_rx,
            search_root,
            last_elapsed: Duration::ZERO,
            search_generation: SearchGeneration::new(),
            current_cancel_handle: None,
            pending_result_clear: false,
            last_displayed_query: String::new(),
            index_state: IndexState::Loading,
            _bg_thread: Some(bg_thread),
            _index_daemon: index_daemon,
            _content_daemon: content_daemon,
            _watchdog: Some(watchdog),
            diag,
            frame_count: 0,
            last_heartbeat: Instant::now(),
            app_start: Instant::now(),
            last_update_end: Instant::now(),
            auto_search: parse_search_arg(),
            auto_search_2: parse_search2_arg(),
            auto_search_2_idx: 0,
            auto_search_2_next: None,
        }
    }

    /// Update the index lifecycle state.
    ///
    /// Called by the index daemon / background builder to signal transitions:
    /// - `Loading` -> `Building` on scan start
    /// - `Building` -> `Ready` on completion
    /// - `Ready` -> `Updating` on FSEvents flush
    /// - `Ready` -> `Stale` on staleness detection
    /// - any -> `Error` on failure
    pub fn set_index_state(&mut self, state: IndexState) {
        self.index_state = state;
    }

    /// Total number of results (file + content matches).
    pub fn total_results(&self) -> usize {
        self.file_matches.len() + self.content_matches.len()
    }

    /// Send a search request to the orchestrator background thread.
    ///
    /// Cancels any in-flight search via the cancellation handle before starting
    /// a new one. Creates a fresh `SearchSession` (token + generation guard) so
    /// the orchestrator can detect cancellation and discard stale results.
    fn dispatch_search(&mut self, query: String) {
        // Always cancel previous in-flight search first
        if let Some(handle) = self.current_cancel_handle.take() {
            handle.cancel();
        }

        if query.is_empty() {
            self.file_matches.clear();
            self.content_matches.clear();
            self.clear_groups();
            self.flat_row_model = FlatRowModel::default();
            self.is_searching = false;
            self.status_bar.is_searching = false;
            self.status_bar.search_start = None;
            self.results_list.selected_index = 0;
            self.status_bar.update(0, Duration::ZERO, self.filter_bar.count());
            self.last_displayed_query.clear();
            return;
        }

        // === Client-side refinement filtering ===
        //
        // When the user types additional characters (e.g. "kol" → "kolbe"),
        // the new query is a refinement of the old one. Instead of showing
        // stale results for "kol" while the 18-second walk restarts, we
        // immediately filter existing results to only those matching the
        // new query. This gives instant visual feedback.
        //
        // Three cases:
        // 1. Refinement (new query extends old): filter in-place
        // 2. Different query (completely unrelated): clear stale results
        // 3. First search or backspace: keep old results via pending_result_clear
        let query_lower = query.to_lowercase();
        let prev_lower = self.last_displayed_query.to_lowercase();
        let has_results = !self.content_matches.is_empty() || !self.file_matches.is_empty();

        if !prev_lower.is_empty() && query_lower.len() > prev_lower.len()
            && query_lower.contains(&prev_lower) && has_results
        {
            // Case 1: Refinement — filter existing results immediately
            let before_content = self.content_matches.len();
            let before_files = self.file_matches.len();

            // Filter and update match_range in a single pass.
            // Drop any match where the NEW pattern is not found in line_content
            // or where the extracted text doesn't case-insensitively match.
            self.content_matches.retain_mut(|cm| {
                let line_lower = cm.line_content.to_lowercase();
                if let Some(pos) = line_lower.find(&query_lower) {
                    let new_range = pos..pos + query.len();
                    // Validate extracted text matches the pattern
                    if new_range.end <= cm.line_content.len() {
                        let extracted = &cm.line_content[new_range.clone()];
                        if extracted.to_lowercase() == query_lower {
                            cm.match_range = new_range;
                            return true;
                        }
                    }
                    // Range valid but extracted text doesn't match — drop
                    false
                } else {
                    // Pattern not found in line — false positive from refinement
                    false
                }
            });
            self.file_matches.retain(|fm| {
                fm.path.to_string_lossy().to_lowercase().contains(&query_lower)
            });

            // Recompute groups from filtered matches
            self.clear_groups();
            self.recompute_groups();
            self.rebuild_flat_model();
            if self.results_list.selected_index >= self.flat_row_model.rows.len() {
                self.results_list.selected_index = 0;
            }

            eprintln!(
                "[refine] '{}' → '{}': content {}->{}, files {}->{}",
                self.last_displayed_query, query,
                before_content, self.content_matches.len(),
                before_files, self.file_matches.len(),
            );

            // Background search will replace these filtered results when ready
            self.pending_result_clear = true;
        } else if !prev_lower.is_empty() && !query_lower.contains(&prev_lower)
            && !prev_lower.contains(&query_lower)
        {
            // Case 2: Completely different query — clear stale results now
            // rather than confusingly showing results for the old query.
            self.file_matches.clear();
            self.content_matches.clear();
            self.clear_groups();
            self.flat_row_model = FlatRowModel::default();
            self.results_list.selected_index = 0;
            self.pending_result_clear = false; // already cleared
        } else {
            // Case 3: First search, backspace, or substring — keep old results
            // visible until new search produces output (existing behavior).
            self.pending_result_clear = true;
        }

        self.last_displayed_query = query.clone();

        let mut request = SearchRequest::new(&query, &self.search_root);
        self.filter_bar.apply_to_request(&mut request);

        // Create new cancellation pair + generation guard for this search
        let (token, handle) = cancellation_pair();
        let guard = self.search_generation.next();
        let session = SearchSession { token, guard };
        self.current_cancel_handle = Some(handle);

        self.is_searching = true;
        self.status_bar.is_searching = true;
        self.status_bar.search_start = Some(Instant::now());
        // Drain any pending updates from previous (now-cancelled) searches
        while self.update_rx.try_recv().is_ok() {}

        // Drain guard: after draining the channel, always set pending_result_clear
        // so the first valid update for the new generation clears accumulated results.
        // This prevents stale results that were in-flight (sent by the orchestrator
        // between our drain and the generation guard check in poll_updates) from
        // leaking through. Even for Case 2 (already cleared), this is harmless --
        // clearing empty vectors is a no-op -- but it guarantees that no code path
        // can leave pending_result_clear=false with stale data still visible.
        self.pending_result_clear = true;

        let _ = self.cmd_tx.send(OrchestratorCommand::Search(request, session));
    }

    /// Poll for search updates from the orchestrator.
    ///
    /// The streaming pipeline sends multiple `ContentMatches` updates as
    /// GPU batches complete. These are accumulated (not replaced) until
    /// the `Complete` message arrives with the final aggregated results.
    ///
    /// Generation guard: updates from stale (superseded) searches are
    /// silently discarded by comparing the stamped generation ID against
    /// the current generation. This fixes the P0 stale-results race.
    fn poll_updates(&mut self) {
        // Track what changed so we rebuild the flat model at most ONCE per
        // frame, not once per message.  With 400+ batches queued this avoids
        // hundreds of redundant O(N) rebuilds that freeze the UI.
        let mut dirty = false;
        let mut completed = false;

        // Process up to MAX_MSGS_PER_FRAME messages per frame (non-blocking).
        // This prevents the channel drain from monopolising the UI thread when
        // hundreds of batches queue up.  Remaining messages are picked up on
        // subsequent frames (request_repaint keeps polling while is_searching).
        const MAX_MSGS_PER_FRAME: usize = 64;
        let mut msgs_processed = 0usize;
        while msgs_processed < MAX_MSGS_PER_FRAME {
            let stamped = match self.update_rx.try_recv() {
                Ok(s) => s,
                Err(_) => break,
            };
            msgs_processed += 1;
            // Discard updates from stale (superseded) search generations.
            // Strict equality is intentional: only the EXACT current generation
            // is accepted. Combined with the drain guard in dispatch_search()
            // (which sets pending_result_clear=true after draining), this ensures
            // no stale results can leak through even if in-flight messages arrive
            // between the channel drain and the first poll_updates() call.
            if stamped.generation != self.search_generation.current_id() {
                continue;
            }

            // Clear old results on the first valid update for a new search.
            // This keeps the previous search's results visible until the new
            // search produces output, avoiding a "Searching..." flash.
            if self.pending_result_clear {
                self.file_matches.clear();
                self.content_matches.clear();
                self.clear_groups();
                self.results_list.selected_index = 0;
                self.pending_result_clear = false;
            }

            match stamped.update {
                SearchUpdate::FileMatches(matches) => {
                    // Extend incrementally — the orchestrator now sends only
                    // NEW matches per update, not the full accumulated list.
                    self.file_matches.extend(matches);
                    if self.file_matches.len() <= 20 {
                        // Only reset selection for early results
                        self.results_list.selected_index = 0;
                    }
                    dirty = true;
                }
                SearchUpdate::ContentMatches(matches) => {
                    // Accumulate progressive content match batches from
                    // the streaming pipeline (each GPU batch sends one update).
                    // Cap at 10K to prevent UI OOM on broad patterns like "patrick".
                    const UI_CONTENT_CAP: usize = 10_000;
                    if self.content_matches.len() < UI_CONTENT_CAP {
                        let remaining = UI_CONTENT_CAP - self.content_matches.len();
                        if matches.len() <= remaining {
                            self.content_matches.extend(matches);
                        } else {
                            self.content_matches.extend(matches.into_iter().take(remaining));
                        }
                        dirty = true;
                    }
                    // Once capped, skip accumulation to avoid CPU churn
                }
                SearchUpdate::Complete(response) => {
                    self.file_matches = response.file_matches;
                    self.content_matches = response.content_matches;
                    // Complete replaces all content_matches, so reset groups
                    self.clear_groups();
                    self.is_searching = false;
                    self.last_elapsed = response.elapsed;
                    self.status_bar.is_searching = false;
                    self.status_bar.search_start = None;
                    self.status_bar.update(
                        self.file_matches.len() + self.content_matches.len(),
                        response.elapsed,
                        self.filter_bar.count(),
                    );
                    dirty = true;
                    completed = true;
                }
            }
        }

        // Single rebuild per frame: recompute groups + rebuild flat model
        if dirty {
            let rebuild_start = Instant::now();
            self.recompute_groups();
            if completed {
                self.sort_groups_by_count();
            }
            self.rebuild_flat_model();
            let rebuild_ms = rebuild_start.elapsed().as_secs_f64() * 1000.0;
            if rebuild_ms > 10.0 {
                eprintln!(
                    "[SLOW-REBUILD] {:.1}ms: msgs={} content={} groups={} rows={}",
                    rebuild_ms, msgs_processed, self.content_matches.len(),
                    self.content_groups.len(), self.flat_row_model.rows.len(),
                );
            }
        }

        // Keep status bar count in sync with displayed results every frame
        self.update_status_from_displayed();
    }

    /// Derive status bar match count from the currently displayed results.
    ///
    /// Called after every poll loop iteration so the status bar always reflects
    /// the actual number of visible results, even during progressive streaming.
    fn update_status_from_displayed(&mut self) {
        let displayed_count = self.file_matches.len() + self.content_matches.len();
        self.status_bar.update(
            displayed_count,
            self.last_elapsed,
            self.filter_bar.count(),
        );
    }

    /// Incrementally group new content matches by file path.
    ///
    /// Only processes matches from `last_grouped_index` onward, so it is
    /// efficient for progressive streaming where new batches arrive each frame.
    /// Uses `group_index_map` for O(1) lookup of existing groups.
    fn recompute_groups(&mut self) {
        let search_root = self.search_root.clone();
        for i in self.last_grouped_index..self.content_matches.len() {
            let cm = &self.content_matches[i];
            if let Some(&group_idx) = self.group_index_map.get(&cm.path) {
                // Existing group: append match index
                self.content_groups[group_idx].match_indices.push(i);
            } else {
                // New group: compute abbreviated path and extension
                let (dir_display, filename) = abbreviate_path(&cm.path, &search_root);
                let extension = cm
                    .path
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                let group_idx = self.content_groups.len();
                self.group_index_map.insert(cm.path.clone(), group_idx);
                self.content_groups.push(ContentGroup {
                    path: cm.path.clone(),
                    dir_display,
                    filename,
                    extension,
                    match_indices: vec![i],
                });
            }
        }
        self.last_grouped_index = self.content_matches.len();
    }

    /// Sort content groups by match count descending (most matches first).
    ///
    /// Called on `Complete` to produce the final display order.
    fn sort_groups_by_count(&mut self) {
        self.content_groups
            .sort_by(|a, b| b.match_indices.len().cmp(&a.match_indices.len()));
        // Rebuild the index map after sorting
        self.group_index_map.clear();
        for (idx, group) in self.content_groups.iter().enumerate() {
            self.group_index_map.insert(group.path.clone(), idx);
        }
    }

    /// Clear all grouping state. Called when starting a new search.
    fn clear_groups(&mut self) {
        self.content_groups.clear();
        self.group_index_map.clear();
        self.last_grouped_index = 0;
    }

    /// Rebuild the flat row model from current file_matches and content_groups.
    ///
    /// Called after any change to results or groups so the virtual scroll
    /// layout stays in sync.
    fn rebuild_flat_model(&mut self) {
        let selected = if self.results_list.selected_index < usize::MAX {
            Some(self.results_list.selected_index)
        } else {
            None
        };
        self.flat_row_model = FlatRowModel::rebuild(
            &self.file_matches,
            &self.content_groups,
            selected,
        );
    }

    /// Handle a keyboard action.
    fn handle_key_action(&mut self, action: KeyAction, ctx: &egui::Context) {
        match action {
            KeyAction::NavigateUp => {
                if let Some(idx) = self.flat_row_model.prev_selectable_row(self.results_list.selected_index) {
                    self.results_list.selected_index = idx;
                    self.selected_index = idx;
                    self.results_list.scroll_to_selected = true;
                    self.rebuild_flat_model();
                }
            }
            KeyAction::NavigateDown => {
                if let Some(idx) = self.flat_row_model.next_selectable_row(self.results_list.selected_index) {
                    self.results_list.selected_index = idx;
                    self.selected_index = idx;
                    self.results_list.scroll_to_selected = true;
                    self.rebuild_flat_model();
                }
            }
            KeyAction::ClearSearch => {
                self.search_bar.clear();
                self.search_input.clear();
                self.file_matches.clear();
                self.content_matches.clear();
                self.clear_groups();
                self.flat_row_model = FlatRowModel::default();
                self.is_searching = false;
                self.results_list.selected_index = 0;
                self.selected_index = 0;
                self.last_displayed_query.clear();
            }
            KeyAction::Dismiss => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            KeyAction::JumpToContentSection => {
                if let Some(idx) = self.flat_row_model.first_selectable_in_content_section() {
                    self.results_list.selected_index = idx;
                    self.selected_index = idx;
                    self.results_list.scroll_to_selected = true;
                    self.rebuild_flat_model();
                }
            }
            KeyAction::JumpToFileSection => {
                if let Some(idx) = self.flat_row_model.first_selectable_in_file_section() {
                    self.results_list.selected_index = idx;
                    self.selected_index = idx;
                    self.results_list.scroll_to_selected = true;
                    self.rebuild_flat_model();
                }
            }
            KeyAction::OpenFile => {
                if let Some(path) = self
                    .results_list
                    .get_selected(&self.file_matches, &self.content_matches)
                {
                    let _ = actions::open_file(path);
                }
            }
            KeyAction::OpenInEditor => {
                if let Some(path) = self
                    .results_list
                    .get_selected(&self.file_matches, &self.content_matches)
                {
                    let line = self
                        .results_list
                        .get_selected_line(&self.file_matches, &self.content_matches)
                        .unwrap_or(1) as usize;
                    let _ = actions::open_in_editor(path, line);
                }
            }
            KeyAction::CopyPath => {
                if let Some(path) = self
                    .results_list
                    .get_selected(&self.file_matches, &self.content_matches)
                {
                    ctx.copy_text(path.display().to_string());
                }
            }
            KeyAction::None => {}
        }
    }
}

impl eframe::App for GpuSearchApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let frame_start = Instant::now();

        // Measure "frame gap" — time eframe spent OUTSIDE update() for
        // rendering, presentation, and event processing. A large gap reveals
        // drawable stalls, GPU contention, or event loop issues.
        let frame_gap_ms = self.last_update_end.elapsed().as_secs_f64() * 1000.0;

        // Touch watchdog: "UI thread is alive"
        self.diag.touch_ui();
        self.frame_count += 1;
        self.diag.ui_frame_count.store(self.frame_count, Ordering::Relaxed);
        self.diag.search_active.store(self.is_searching, Ordering::Relaxed);
        self.diag.ui_content_matches.store(self.content_matches.len() as u64, Ordering::Relaxed);

        // Approximate channel depth: count how many messages are pending
        // (crossbeam doesn't expose len(), but we can check is_empty)
        self.diag.channel_pending.store(
            if self.update_rx.is_empty() { 0 } else { 1 },
            Ordering::Relaxed,
        );

        // Heartbeat: log every frame for the first 15 seconds, then every 1s
        let since_start = self.app_start.elapsed().as_secs_f64();
        let log_this_frame = if since_start < 15.0 {
            self.frame_count.is_multiple_of(100)
        } else {
            self.last_heartbeat.elapsed() >= Duration::from_secs(1)
        };
        if log_this_frame {
            eprintln!(
                "[HB] t={:.1}s f={} gap={:.1}ms search={} content={} groups={} rows={}",
                since_start, self.frame_count, frame_gap_ms,
                self.is_searching,
                self.content_matches.len(),
                self.content_groups.len(),
                self.flat_row_model.rows.len(),
            );
            self.last_heartbeat = Instant::now();
        }

        // Auto-search from --search CLI argument (dispatched once after 1s startup).
        if self.auto_search.is_some() && self.app_start.elapsed() > Duration::from_secs(1) {
            if let Some(query) = self.auto_search.take() {
                eprintln!("[auto-search] dispatching query: '{}'", query);
                // Sync both query AND prev_query to prevent the debounce
                // from detecting a "change" and re-dispatching 100ms later.
                self.search_bar.query = query.clone();
                self.search_bar.suppress_next_debounce();
                self.dispatch_search(query);
            }
        }

        // Second auto-search from --search2: simulates rapid typing to reproduce
        // the cancel-restart cycle that causes the UI freeze.  Dispatches one
        // character at a time ("p", "pa", "pat", …) with 50ms between keystrokes,
        // starting 5s after app launch.
        if let Some(ref full_query) = self.auto_search_2.clone() {
            let start_delay = Duration::from_secs(5);
            let keystroke_interval = Duration::from_millis(50);

            if self.auto_search_2_idx == 0 && self.auto_search_2_next.is_none() {
                // Haven't started yet — wait for 5s startup delay.
                if self.app_start.elapsed() >= start_delay {
                    eprintln!(
                        "[auto-search-2] starting rapid-fire typing simulation for '{}' ({} chars, 50ms interval)",
                        full_query, full_query.len(),
                    );
                    self.auto_search_2_next = Some(Instant::now());
                }
            }

            if let Some(next_time) = self.auto_search_2_next {
                if Instant::now() >= next_time && self.auto_search_2_idx < full_query.len() {
                    self.auto_search_2_idx += 1;
                    let partial = &full_query[..self.auto_search_2_idx];
                    eprintln!(
                        "[auto-search-2] keystroke {}/{}: dispatching '{}'",
                        self.auto_search_2_idx, full_query.len(), partial,
                    );
                    self.search_bar.query = partial.to_string();
                    self.dispatch_search(partial.to_string());
                    self.auto_search_2_next = Some(Instant::now() + keystroke_interval);
                }
                // All characters dispatched — clean up.
                if self.auto_search_2_idx >= full_query.len() {
                    eprintln!("[auto-search-2] all {} keystrokes dispatched", full_query.len());
                    self.auto_search_2 = None;
                    self.auto_search_2_next = None;
                }
            }
        }

        // 1. Process keyboard input
        let t0 = Instant::now();
        let action = keybinds::process_input(ctx);
        self.handle_key_action(action, ctx);
        let keys_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. Poll for search updates from background thread
        let t1 = Instant::now();
        self.poll_updates();
        let poll_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // 3. Request repaint while active, throttled to ~60fps.
        // CRITICAL: Use request_repaint_after to prevent flooding the macOS
        // event loop with thousands of repaint events per second. The 16ms
        // timer ensures smooth 60fps updates without starving the run loop.
        // Also always request a slow keepalive repaint to prevent the event
        // loop from permanently going idle (the root cause of the freeze).
        if self.is_searching
            || matches!(self.index_state, IndexState::Building { .. })
            || matches!(self.index_state, IndexState::Updating)
        {
            ctx.request_repaint_after(Duration::from_millis(16));
        } else {
            // Keepalive: ensure the event loop never permanently sleeps.
            // Without this, if request_repaint() signals are lost during a
            // system stall, the event loop may never wake up again.
            ctx.request_repaint_after(Duration::from_millis(200));
        }

        // Theme is set once in new() — do NOT set it here per-frame.
        // Calling ctx.set_style() every frame marks context as changed,
        // causing infinite repaint at 120Hz and 26MB/s memory growth.

        let render_start = Instant::now();

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::BG_BASE).inner_margin(egui::Margin::same(8)))
            .show(ctx, |ui| {
                // Search bar at top
                let fired_query =
                    self.search_bar.show(ui, SearchMode::Content);

                // Sync search bar query to app state
                self.search_input = self.search_bar.query.clone();

                // If debounce fired, dispatch search
                if let Some(query) = fired_query {
                    self.dispatch_search(query);
                }

                // Filter bar (shown when filter toggle is active)
                if self.search_bar.filter_active {
                    ui.add_space(4.0);
                    self.filter_bar.render(ui);
                }

                ui.add_space(4.0);
                ui.separator();

                // Results area
                if self.is_searching && self.file_matches.is_empty() && self.content_matches.is_empty() {
                    ui.centered_and_justified(|ui| {
                        ui.colored_label(theme::TEXT_MUTED, "Searching...");
                    });
                } else if self.search_input.is_empty() {
                    ui.centered_and_justified(|ui| {
                        ui.colored_label(theme::TEXT_MUTED, "Type to search files and content");
                    });
                } else if self.total_results() == 0 && !self.is_searching {
                    ui.centered_and_justified(|ui| {
                        ui.colored_label(theme::TEXT_MUTED, "No results");
                    });
                } else {
                    // Results list with grouped virtual scroll
                    let file_matches = &self.file_matches;
                    let content_matches = &self.content_matches;
                    let content_groups = &self.content_groups;
                    let flat_row_model = &self.flat_row_model;
                    let search_input = &self.search_input;
                    let search_root = &self.search_root;
                    let prev_selected = self.results_list.selected_index;
                    self.results_list.show(
                        ui,
                        file_matches,
                        content_matches,
                        content_groups,
                        flat_row_model,
                        search_input,
                        search_root,
                        &mut self.highlighter,
                    );
                    // Rebuild flat model if selection changed via click
                    if self.results_list.selected_index != prev_selected {
                        self.selected_index = self.results_list.selected_index;
                        self.rebuild_flat_model();
                    }
                }

                // Status bar at bottom
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    self.status_bar.render(ui);
                });
            });

        let render_ms = render_start.elapsed().as_secs_f64() * 1000.0;

        // Log slow frames with FULL timing including render pass.
        // Previous version only timed pre-render logic, completely missing
        // the egui rendering pass which could be the bottleneck.
        let total_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        // Log when: (a) update() itself is slow (>16ms), (b) frame gap shows a
        // stall during active search (>100ms — lowered from 500ms to catch early
        // signs of GPU contention before the event loop dies).
        let is_stall = frame_gap_ms > 100.0 && self.is_searching;
        if total_ms > 16.0 || is_stall {
            eprintln!(
                "[SLOW-FRAME] {:.1}ms total (gap={:.1}ms keys={:.1}ms poll={:.1}ms render={:.1}ms) \
                 content={} groups={} rows={}",
                total_ms, frame_gap_ms, keys_ms, poll_ms, render_ms,
                self.content_matches.len(),
                self.content_groups.len(),
                self.flat_row_model.rows.len(),
            );
        }

        // Record end of update() for next frame's gap measurement
        self.last_update_end = Instant::now();
    }
}

impl Drop for GpuSearchApp {
    fn drop(&mut self) {
        // Signal the background thread to shut down
        let _ = self.cmd_tx.send(OrchestratorCommand::Shutdown);
    }
}

/// Background thread function for the search orchestrator.
///
/// Creates its own GpuDevice + PsoCache + SearchOrchestrator on this thread
/// (Metal objects are created per-thread for safety). Loops waiting for commands
/// from the UI via crossbeam channel.
///
/// Uses `search_streaming` for progressive result delivery -- directory walking
/// overlaps with GPU search so results start arriving immediately instead of
/// blocking until all files are enumerated.
fn orchestrator_thread(
    cmd_rx: Receiver<OrchestratorCommand>,
    update_tx: Sender<StampedUpdate>,
    index_store: Arc<IndexStore>,
    content_store: Arc<ContentIndexStore>,
    diag: Arc<crate::ui::watchdog::DiagState>,
) {
    // Initialize GPU resources on this thread
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);
    let mut orchestrator = match SearchOrchestrator::with_content_store(
        &device.device, &pso_cache, Some(index_store), content_store,
    ) {
        Some(o) => o,
        None => {
            eprintln!("gpu-search: failed to initialize search orchestrator");
            return;
        }
    };

    eprintln!("[orchestrator] thread started, waiting for commands");

    // Event loop: wait for commands
    loop {
        // Touch watchdog while idle (waiting for commands)
        diag.touch_bg();

        match cmd_rx.recv() {
            Ok(OrchestratorCommand::Search(request, session)) => {
                diag.touch_bg();

                // Drain any queued search commands -- only process the latest
                let mut latest_request = request;
                let mut latest_session = session;
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        OrchestratorCommand::Search(r, s) => {
                            latest_request = r;
                            latest_session = s;
                        }
                        OrchestratorCommand::Shutdown => return,
                    }
                }

                // Skip if already cancelled/superseded before we even start
                if latest_session.should_stop() {
                    continue;
                }

                // Reset batch counters for this search
                diag.bg_batch_num.store(0, Ordering::Relaxed);
                diag.bg_content_matches.store(0, Ordering::Relaxed);
                diag.bg_files_searched.store(0, Ordering::Relaxed);

                eprintln!("[orchestrator] starting search for '{}' from {:?}",
                    latest_request.pattern, latest_request.root);

                // Execute streaming search with cancellation support and diagnostics.
                orchestrator.search_streaming_diag(
                    latest_request, &update_tx, &latest_session,
                    &diag,
                );

                diag.touch_bg();
                eprintln!("[orchestrator] search complete");
            }
            Ok(OrchestratorCommand::Shutdown) | Err(_) => {
                eprintln!("[orchestrator] shutting down");
                return;
            }
        }
    }
}

/// Launch the gpu-search application window.
///
/// Configures eframe with a 720x400 floating window, then runs the event loop.
/// This function blocks until the window is closed.
/// Parse `--root <path>` from CLI arguments.
///
/// Returns `Some(path)` if `--root` was provided and the path exists,
/// `None` otherwise (caller falls back to current working directory).
fn parse_root_arg() -> Option<PathBuf> {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--root" {
            if let Some(path_str) = args.get(i + 1) {
                let path = PathBuf::from(path_str);
                if path.is_dir() {
                    return Some(path);
                }
                eprintln!("gpu-search: --root path does not exist: {}", path_str);
                return None;
            }
        }
    }
    None
}

/// Parse `--search <pattern>` from CLI arguments for automated testing.
fn parse_search_arg() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--search" {
            if let Some(pattern) = args.get(i + 1) {
                return Some(pattern.clone());
            }
        }
    }
    None
}

/// Parse `--search2 <pattern>` — dispatched after first search completes.
/// Used to reproduce the "second search freezes" scenario.
fn parse_search2_arg() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--search2" {
            if let Some(pattern) = args.get(i + 1) {
                return Some(pattern.clone());
            }
        }
    }
    None
}

pub fn run_app() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([720.0, 400.0])
            .with_decorations(false),
        // Use AutoVsync (Fifo/Mailbox) to prevent drawable pool exhaustion.
        // AutoNoVsync allows unlimited frame production which, combined with
        // rapid repaint requests, exhausts CAMetalLayer's 3-drawable pool and
        // causes nextDrawable() to block permanently. AutoVsync synchronizes
        // presentation with the display refresh rate, ensuring drawables are
        // returned to the pool before new ones are acquired.
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            present_mode: eframe::wgpu::PresentMode::AutoVsync,
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "gpu-search",
        options,
        Box::new(|cc| Ok(Box::new(GpuSearchApp::new(cc)))),
    )
}
