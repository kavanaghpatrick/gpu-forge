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
use crate::search::orchestrator::SearchOrchestrator;
use crate::search::types::{ContentMatch, FileMatch, SearchRequest, SearchUpdate};

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
    Search(SearchRequest),
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

    /// Channel to receive search updates from the orchestrator.
    update_rx: Receiver<SearchUpdate>,

    /// Root directory for searches (current working dir by default).
    search_root: PathBuf,

    /// Last search elapsed time.
    last_elapsed: Duration,

    /// Handle to the background orchestrator thread.
    _bg_thread: Option<thread::JoinHandle<()>>,
}

impl Default for GpuSearchApp {
    fn default() -> Self {
        // Create channels
        let (cmd_tx, _cmd_rx) = crossbeam_channel::unbounded();
        let (_update_tx, update_rx) = crossbeam_channel::unbounded();

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
            _bg_thread: None,
        }
    }
}

impl GpuSearchApp {
    /// Create a new GpuSearchApp, initializing the GPU device, pipeline,
    /// and orchestrator on a background thread.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let search_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        // Channels: UI -> orchestrator commands, orchestrator -> UI updates
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<OrchestratorCommand>();
        let (update_tx, update_rx) = crossbeam_channel::unbounded::<SearchUpdate>();

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
            _bg_thread: Some(bg_thread),
        }
    }

    /// Total number of results (file + content matches).
    pub fn total_results(&self) -> usize {
        self.file_matches.len() + self.content_matches.len()
    }

    /// Send a search request to the orchestrator background thread.
    fn dispatch_search(&mut self, query: String) {
        if query.is_empty() {
            // Clear results
            self.file_matches.clear();
            self.content_matches.clear();
            self.is_searching = false;
            self.results_list.selected_index = 0;
            self.status_bar.update(0, Duration::ZERO, self.filter_bar.count());
            return;
        }

        let mut request = SearchRequest::new(&query, &self.search_root);
        self.filter_bar.apply_to_request(&mut request);

        self.is_searching = true;
        // Drain any pending updates from previous searches
        while self.update_rx.try_recv().is_ok() {}

        let _ = self.cmd_tx.send(OrchestratorCommand::Search(request));
    }

    /// Poll for search updates from the orchestrator.
    fn poll_updates(&mut self) {
        // Process all available updates this frame (non-blocking)
        while let Ok(update) = self.update_rx.try_recv() {
            match update {
                SearchUpdate::FileMatches(matches) => {
                    self.file_matches = matches;
                    self.results_list.selected_index = 0;
                }
                SearchUpdate::ContentMatches(matches) => {
                    self.content_matches = matches;
                }
                SearchUpdate::Complete(response) => {
                    self.file_matches = response.file_matches;
                    self.content_matches = response.content_matches;
                    self.is_searching = false;
                    self.last_elapsed = response.elapsed;
                    self.status_bar.update(
                        response.total_matches as usize,
                        response.elapsed,
                        self.filter_bar.count(),
                    );
                }
            }
        }
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
            KeyAction::OpenFile | KeyAction::OpenInEditor | KeyAction::CopyPath => {
                // These actions require a selected result path -- deferred to Phase 3
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
fn orchestrator_thread(
    cmd_rx: Receiver<OrchestratorCommand>,
    update_tx: Sender<SearchUpdate>,
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
            Ok(OrchestratorCommand::Search(request)) => {
                // Drain any queued search commands -- only process the latest
                let mut latest_request = request;
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        OrchestratorCommand::Search(r) => latest_request = r,
                        OrchestratorCommand::Shutdown => return,
                    }
                }

                // Execute the search
                let response = orchestrator.search(latest_request);

                // Send progressive updates
                let _ = update_tx.send(SearchUpdate::FileMatches(response.file_matches.clone()));
                let _ = update_tx.send(SearchUpdate::ContentMatches(
                    response.content_matches.clone(),
                ));
                let _ = update_tx.send(SearchUpdate::Complete(response));
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
