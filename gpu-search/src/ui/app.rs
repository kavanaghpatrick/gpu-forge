//! egui application shell for gpu-search.
//!
//! Provides the main application struct (`GpuSearchApp`) implementing `eframe::App`,
//! with search input state, result storage, and the basic UI layout.

use eframe::egui;

use crate::search::types::{ContentMatch, FileMatch};

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
}

impl Default for GpuSearchApp {
    fn default() -> Self {
        Self {
            search_input: String::new(),
            file_matches: Vec::new(),
            content_matches: Vec::new(),
            selected_index: 0,
            is_searching: false,
        }
    }
}

impl GpuSearchApp {
    /// Create a new GpuSearchApp with default (empty) state.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }

    /// Total number of results (file + content matches).
    pub fn total_results(&self) -> usize {
        self.file_matches.len() + self.content_matches.len()
    }
}

impl eframe::App for GpuSearchApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Search input at top
            ui.horizontal(|ui| {
                ui.label("Search:");
                ui.text_edit_singleline(&mut self.search_input);
            });

            ui.separator();

            // Results area
            if self.is_searching {
                ui.label("Searching...");
            } else if self.search_input.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Type to search files and content");
                });
            } else if self.total_results() == 0 {
                ui.centered_and_justified(|ui| {
                    ui.label("No results");
                });
            } else {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // File matches section
                    if !self.file_matches.is_empty() {
                        ui.strong(format!("FILENAME MATCHES ({})", self.file_matches.len()));
                        for (i, fm) in self.file_matches.iter().enumerate() {
                            let selected = i == self.selected_index;
                            let label = egui::SelectableLabel::new(
                                selected,
                                fm.path.display().to_string(),
                            );
                            if ui.add(label).clicked() {
                                self.selected_index = i;
                            }
                        }
                    }

                    // Content matches section
                    if !self.content_matches.is_empty() {
                        ui.add_space(8.0);
                        ui.strong(format!("CONTENT MATCHES ({})", self.content_matches.len()));
                        let offset = self.file_matches.len();
                        for (i, cm) in self.content_matches.iter().enumerate() {
                            let selected = (offset + i) == self.selected_index;
                            let text = format!(
                                "{}:{} {}",
                                cm.path.display(),
                                cm.line_number,
                                cm.line_content.trim(),
                            );
                            let label = egui::SelectableLabel::new(selected, text);
                            if ui.add(label).clicked() {
                                self.selected_index = offset + i;
                            }
                        }
                    }
                });
            }
        });
    }
}

/// Launch the gpu-search application window.
///
/// Configures eframe with a 720x400 window, then runs the event loop.
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
