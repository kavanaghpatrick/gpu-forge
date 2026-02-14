//! egui application shell for gpu-search.
//!
//! Wires the GPU engine -> search orchestrator -> UI. The orchestrator runs on a
//! dedicated background thread with its own `MTLCommandQueue`. Communication uses
//! crossbeam channels: UI sends `SearchRequest`, orchestrator sends `SearchUpdate`
//! back. UI polls `try_recv` every frame to update results progressively.

use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use eframe::egui;

use crate::gpu::device::GpuDevice;
use crate::gpu::pipeline::PsoCache;
use crate::search::cancel::{cancellation_pair, CancellationHandle, SearchGeneration, SearchSession};
use crate::search::orchestrator::SearchOrchestrator;
use crate::search::types::{ContentMatch, FileMatch, SearchRequest, SearchUpdate, StampedUpdate};

use super::actions;
use super::filters::FilterBar;
use super::highlight::SyntaxHighlighter;
use super::keybinds::{self, KeyAction};
use super::results_list::ResultsList;
use super::search_bar::{SearchBar, SearchMode};
use super::status_bar::StatusBar;
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
    _highlighter: SyntaxHighlighter,

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

    /// Handle to the background orchestrator thread.
    _bg_thread: Option<thread::JoinHandle<()>>,
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
            selected_index: 0,
            is_searching: false,
            search_bar: SearchBar::default(),
            results_list: ResultsList::new(),
            status_bar: StatusBar::default(),
            filter_bar: FilterBar::new(),
            _highlighter: SyntaxHighlighter::new(),
            cmd_tx,
            update_rx,
            search_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            last_elapsed: Duration::ZERO,
            search_generation: SearchGeneration::new(),
            current_cancel_handle: None,
            _bg_thread: None,
        }
    }
}

impl GpuSearchApp {
    /// Create a new GpuSearchApp, initializing the GPU device, pipeline,
    /// and orchestrator on a background thread.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let search_root = parse_root_arg()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        // Channels: UI -> orchestrator commands, orchestrator -> UI updates
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<OrchestratorCommand>();
        let (update_tx, update_rx) = crossbeam_channel::unbounded::<StampedUpdate>();

        // Spawn background orchestrator thread with its own GPU resources.
        // Each Metal device + command queue is created on the background thread
        // for thread safety.
        let bg_thread = thread::spawn(move || {
            orchestrator_thread(cmd_rx, update_tx);
        });

        let mut status_bar = StatusBar::default();
        status_bar.set_search_root(&search_root);

        Self {
            search_input: String::new(),
            file_matches: Vec::new(),
            content_matches: Vec::new(),
            selected_index: 0,
            is_searching: false,
            search_bar: SearchBar::default(),
            results_list: ResultsList::new(),
            status_bar,
            filter_bar: FilterBar::new(),
            _highlighter: SyntaxHighlighter::new(),
            cmd_tx,
            update_rx,
            search_root,
            last_elapsed: Duration::ZERO,
            search_generation: SearchGeneration::new(),
            current_cancel_handle: None,
            _bg_thread: Some(bg_thread),
        }
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
            self.is_searching = false;
            self.results_list.selected_index = 0;
            self.status_bar.update(0, Duration::ZERO, self.filter_bar.count());
            return;
        }

        let mut request = SearchRequest::new(&query, &self.search_root);
        self.filter_bar.apply_to_request(&mut request);

        // Create new cancellation pair + generation guard for this search
        let (token, handle) = cancellation_pair();
        let guard = self.search_generation.next();
        let session = SearchSession { token, guard };
        self.current_cancel_handle = Some(handle);

        self.is_searching = true;
        self.file_matches.clear();
        self.content_matches.clear();
        // Drain any pending updates from previous (now-cancelled) searches
        while self.update_rx.try_recv().is_ok() {}

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
        // Process all available updates this frame (non-blocking)
        while let Ok(stamped) = self.update_rx.try_recv() {
            // Discard updates from stale (superseded) search generations
            if stamped.generation != self.search_generation.current_id() {
                continue;
            }

            match stamped.update {
                SearchUpdate::FileMatches(matches) => {
                    self.file_matches = matches;
                    self.results_list.selected_index = 0;
                }
                SearchUpdate::ContentMatches(matches) => {
                    // Accumulate progressive content match batches from
                    // the streaming pipeline (each GPU batch sends one update)
                    self.content_matches.extend(matches);
                }
                SearchUpdate::Complete(response) => {
                    self.file_matches = response.file_matches;
                    self.content_matches = response.content_matches;
                    self.is_searching = false;
                    self.last_elapsed = response.elapsed;
                    self.status_bar.update(
                        self.file_matches.len() + self.content_matches.len(),
                        response.elapsed,
                        self.filter_bar.count(),
                    );
                }
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

    /// Handle a keyboard action.
    fn handle_key_action(&mut self, action: KeyAction, ctx: &egui::Context) {
        match action {
            KeyAction::NavigateUp => {
                self.results_list
                    .select_prev(&self.file_matches, &self.content_matches);
                self.selected_index = self.results_list.selected_index;
            }
            KeyAction::NavigateDown => {
                self.results_list
                    .select_next(&self.file_matches, &self.content_matches);
                self.selected_index = self.results_list.selected_index;
            }
            KeyAction::ClearSearch => {
                self.search_bar.clear();
                self.search_input.clear();
                self.file_matches.clear();
                self.content_matches.clear();
                self.is_searching = false;
                self.results_list.selected_index = 0;
                self.selected_index = 0;
            }
            KeyAction::Dismiss => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            KeyAction::CycleFilter => {
                self.search_bar.filter_active = !self.search_bar.filter_active;
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
        // 1. Process keyboard input
        let action = keybinds::process_input(ctx);
        self.handle_key_action(action, ctx);

        // 2. Poll for search updates from background thread
        self.poll_updates();

        // 3. Request repaint while searching to keep polling
        if self.is_searching {
            ctx.request_repaint();
        }

        // Apply Tokyo Night theme
        let mut style = (*ctx.style()).clone();
        style.visuals.override_text_color = Some(theme::TEXT_PRIMARY);
        style.visuals.panel_fill = theme::BG_BASE;
        style.visuals.window_fill = theme::BG_BASE;
        ctx.set_style(style);

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
                    // Results list with virtual scroll
                    self.results_list.show(
                        ui,
                        &self.file_matches,
                        &self.content_matches,
                        &self.search_input,
                        &self.search_root,
                    );
                }

                // Status bar at bottom
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    self.status_bar.render(ui);
                });
            });
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
) {
    // Initialize GPU resources on this thread
    let device = GpuDevice::new();
    let pso_cache = PsoCache::new(&device.device);
    let mut orchestrator = match SearchOrchestrator::new(&device.device, &pso_cache) {
        Some(o) => o,
        None => {
            eprintln!("gpu-search: failed to initialize search orchestrator");
            return;
        }
    };

    // Event loop: wait for commands
    loop {
        match cmd_rx.recv() {
            Ok(OrchestratorCommand::Search(request, session)) => {
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

                // Execute streaming search with cancellation support.
                // search_streaming checks session.should_stop() between GPU batches
                // and aborts early if a new search has been dispatched.
                orchestrator.search_streaming(latest_request, &update_tx, &latest_session);
            }
            Ok(OrchestratorCommand::Shutdown) | Err(_) => {
                // Channel closed or explicit shutdown
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

pub fn run_app() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([720.0, 400.0])
            .with_decorations(false),
        ..Default::default()
    };

    eframe::run_native(
        "gpu-search",
        options,
        Box::new(|cc| Ok(Box::new(GpuSearchApp::new(cc)))),
    )
}
