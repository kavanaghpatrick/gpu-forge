//! Status bar widget for gpu-search.
//!
//! Displays match count, elapsed search time, search root path, and active
//! filter count in a horizontal bar at the bottom of the search panel.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;

use super::path_utils::abbreviate_root;
use super::theme;

/// Represents the current state of the persistent file index.
///
/// Used by the UI to display index status in the status bar and by the app
/// to track index lifecycle events (building, ready, updating, stale, error).
#[derive(Debug, Clone, Default)]
pub enum IndexState {
    /// Index is being loaded from disk (initial startup).
    #[default]
    Loading,
    /// Index is being built from a filesystem scan.
    Building {
        /// Number of files indexed so far.
        files_indexed: usize,
    },
    /// Index is ready and serving searches.
    Ready {
        /// Total number of files in the index.
        file_count: usize,
    },
    /// Index is being updated with incremental changes from FSEvents.
    Updating,
    /// Index exists but is older than the staleness threshold.
    Stale {
        /// Total number of files in the stale index.
        file_count: usize,
    },
    /// Index encountered an error.
    Error {
        /// Human-readable error description.
        message: String,
    },
}


/// Format a file count for compact display.
///
/// - < 1000: exact number (e.g., "420")
/// - 1000-999999: K format (e.g., "1.5K", "420K")
/// - >= 1000000: M format (e.g., "1.3M")
pub fn format_file_count(count: usize) -> String {
    if count < 1_000 {
        format!("{count}")
    } else if count < 1_000_000 {
        let k = count as f64 / 1_000.0;
        if count < 100_000 {
            // Show one decimal for 1.0K-99.9K
            format!("{k:.1}K")
        } else {
            // 100K-999K: no decimal needed
            format!("{k:.0}K")
        }
    } else {
        let m = count as f64 / 1_000_000.0;
        format!("{m:.1}M")
    }
}

/// Status bar displaying search statistics.
///
/// When searching: "Searching... | N matches | X.Xs | ~root | 2 filters"
/// When idle:      "N matches in X.Xms | ~root | 2 filters"
/// Uses TEXT_MUTED for labels and TEXT_PRIMARY for values.
pub struct StatusBar {
    /// Current state of the persistent file index.
    pub index_state: IndexState,
    /// Number of matches found.
    pub match_count: usize,
    /// Time elapsed for the search.
    pub elapsed: Duration,
    /// Root directory being searched.
    pub search_root: PathBuf,
    /// Number of active filters.
    pub active_filter_count: usize,
    /// Whether a search is currently in progress.
    pub is_searching: bool,
    /// When the current search started (for live elapsed display).
    pub search_start: Option<Instant>,
}

impl Default for StatusBar {
    fn default() -> Self {
        Self {
            index_state: IndexState::default(),
            match_count: 0,
            elapsed: Duration::ZERO,
            search_root: PathBuf::from("."),
            active_filter_count: 0,
            is_searching: false,
            search_start: None,
        }
    }
}

impl StatusBar {
    /// Create a new StatusBar with the given values.
    pub fn new(
        match_count: usize,
        elapsed: Duration,
        search_root: impl Into<PathBuf>,
        active_filter_count: usize,
    ) -> Self {
        Self {
            index_state: IndexState::default(),
            match_count,
            elapsed,
            search_root: search_root.into(),
            active_filter_count,
            is_searching: false,
            search_start: None,
        }
    }

    /// Returns the display text and color for the current index state.
    pub fn index_status_text(&self) -> (String, egui::Color32) {
        match &self.index_state {
            IndexState::Loading => ("Loading...".to_string(), theme::ACCENT),
            IndexState::Building { files_indexed } => {
                let formatted = format_file_count(*files_indexed);
                (format!("Indexing: {formatted} files"), theme::ACCENT)
            }
            IndexState::Ready { file_count } => {
                let formatted = format_file_count(*file_count);
                (format!("{formatted} files"), theme::SUCCESS)
            }
            IndexState::Updating => ("Updating...".to_string(), theme::ACCENT),
            IndexState::Stale { file_count } => {
                let formatted = format_file_count(*file_count);
                (format!("{formatted} files (stale)"), theme::ACCENT)
            }
            IndexState::Error { .. } => ("Index: error".to_string(), theme::ERROR),
        }
    }

    /// Update the status bar with new search results.
    pub fn update(&mut self, match_count: usize, elapsed: Duration, active_filter_count: usize) {
        self.match_count = match_count;
        self.elapsed = elapsed;
        self.active_filter_count = active_filter_count;
    }

    /// Set the search root path.
    pub fn set_search_root(&mut self, root: impl Into<PathBuf>) {
        self.search_root = root.into();
    }

    /// Render the status bar as a horizontal row.
    ///
    /// When searching: "Searching... | N matches | X.Xs | ~root | filters"
    /// When idle:      "N matches in X.Xms | ~root | filters"
    pub fn render(&self, ui: &mut egui::Ui) {
        let bar_rect = ui.available_rect_before_wrap();
        // Draw separator line at top of status bar
        ui.painter().line_segment(
            [bar_rect.left_top(), bar_rect.right_top()],
            egui::Stroke::new(1.0, theme::BORDER),
        );

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;

            // Index status segment (prepended before search stats)
            let (index_text, index_color) = self.index_status_text();
            ui.colored_label(index_color, &index_text);
            ui.colored_label(theme::BORDER, "|");

            if self.is_searching {
                // Live searching state
                ui.colored_label(theme::ACCENT, "Searching...");
                ui.colored_label(theme::BORDER, "|");

                // Live match count
                ui.colored_label(
                    theme::TEXT_PRIMARY,
                    format!("{}", self.match_count),
                );
                ui.colored_label(theme::TEXT_MUTED, "matches");
                ui.colored_label(theme::BORDER, "|");

                // Live elapsed time from search_start
                let live_elapsed = self
                    .search_start
                    .map(|start| start.elapsed())
                    .unwrap_or(Duration::ZERO);
                let secs = live_elapsed.as_secs_f64();
                ui.colored_label(
                    theme::TEXT_PRIMARY,
                    format!("{:.1}s", secs),
                );
            } else {
                // Final results state
                ui.colored_label(
                    theme::TEXT_PRIMARY,
                    format!("{}", self.match_count),
                );
                ui.colored_label(theme::TEXT_MUTED, "matches in");
                ui.colored_label(
                    theme::TEXT_PRIMARY,
                    format!("{:.1}ms", self.elapsed.as_secs_f64() * 1000.0),
                );
            }

            // Separator
            ui.colored_label(theme::BORDER, "|");

            // Search root (with ~ substitution, truncated if too long)
            let abbreviated = abbreviate_root(&self.search_root);
            let display_root = truncate_path(&abbreviated, 40);
            ui.colored_label(theme::TEXT_PRIMARY, display_root);

            // Active filters (only show if > 0)
            if self.active_filter_count > 0 {
                ui.colored_label(theme::BORDER, "|");
                ui.colored_label(
                    theme::ACCENT,
                    format!(
                        "{} filter{}",
                        self.active_filter_count,
                        if self.active_filter_count == 1 { "" } else { "s" }
                    ),
                );
            }
        });
        ui.add_space(2.0);
    }

    /// Format the status bar as a string (for testing).
    pub fn format_status(&self) -> String {
        let (index_text, _) = self.index_status_text();
        let abbreviated = abbreviate_root(&self.search_root);
        let display_root = truncate_path(&abbreviated, 40);
        let mut status = if self.is_searching {
            let live_elapsed = self
                .search_start
                .map(|start| start.elapsed())
                .unwrap_or(Duration::ZERO);
            format!(
                "{} | Searching... | {} matches | {:.1}s | {}",
                index_text,
                self.match_count,
                live_elapsed.as_secs_f64(),
                display_root,
            )
        } else {
            format!(
                "{} | {} matches in {:.1}ms | {}",
                index_text,
                self.match_count,
                self.elapsed.as_secs_f64() * 1000.0,
                display_root,
            )
        };
        if self.active_filter_count > 0 {
            status.push_str(&format!(
                " | {} filter{}",
                self.active_filter_count,
                if self.active_filter_count == 1 { "" } else { "s" }
            ));
        }
        status
    }
}

/// Truncate a path string to max_len characters, keeping the end.
fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("...{}", &path[path.len() - (max_len - 3)..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_bar_default() {
        let bar = StatusBar::default();
        assert!(matches!(bar.index_state, IndexState::Loading));
        assert_eq!(bar.match_count, 0);
        assert_eq!(bar.elapsed, Duration::ZERO);
        assert_eq!(bar.search_root, PathBuf::from("."));
        assert_eq!(bar.active_filter_count, 0);
        assert!(!bar.is_searching);
        assert!(bar.search_start.is_none());
    }

    #[test]
    fn test_status_bar_new() {
        let bar = StatusBar::new(42, Duration::from_millis(15), "/Users/dev/project", 2);
        assert_eq!(bar.match_count, 42);
        assert_eq!(bar.elapsed, Duration::from_millis(15));
        assert_eq!(bar.search_root, PathBuf::from("/Users/dev/project"));
        assert_eq!(bar.active_filter_count, 2);
    }

    #[test]
    fn test_status_bar_update() {
        let mut bar = StatusBar::default();
        bar.set_search_root("/home/user/code");
        bar.update(100, Duration::from_millis(8), 3);
        assert_eq!(bar.match_count, 100);
        assert_eq!(bar.elapsed, Duration::from_millis(8));
        assert_eq!(bar.active_filter_count, 3);
        assert_eq!(bar.search_root, PathBuf::from("/home/user/code"));
    }

    #[test]
    fn test_status_bar_format_no_filters() {
        let bar = StatusBar::new(256, Duration::from_millis(12), "/src", 0);
        let status = bar.format_status();
        assert!(status.contains("256 matches"));
        assert!(status.contains("12.0ms"));
        assert!(status.contains("/src"));
        assert!(!status.contains("filter"));
    }

    #[test]
    fn test_status_bar_format_with_filters() {
        let bar = StatusBar::new(42, Duration::from_millis(5), "/project", 2);
        let status = bar.format_status();
        assert!(status.contains("42 matches"));
        assert!(status.contains("5.0ms"));
        assert!(status.contains("/project"));
        assert!(status.contains("2 filters"));
    }

    #[test]
    fn test_status_bar_format_single_filter() {
        let bar = StatusBar::new(10, Duration::from_millis(3), "/root", 1);
        let status = bar.format_status();
        assert!(status.contains("1 filter"));
        assert!(!status.contains("1 filters"));
    }

    #[test]
    fn test_status_bar_truncate_long_path() {
        let long_path = "/Users/developer/very/deeply/nested/project/directory/structure/src";
        let bar = StatusBar::new(10, Duration::from_millis(5), long_path, 0);
        let status = bar.format_status();
        // Path should be truncated
        assert!(status.contains("..."));
        // But still contains the tail of the path
        assert!(status.contains("src"));
    }

    #[test]
    fn test_status_bar_short_path_not_truncated() {
        let short_path = "/src";
        let truncated = truncate_path(short_path, 40);
        assert_eq!(truncated, "/src");
        assert!(!truncated.contains("..."));
    }

    #[test]
    fn test_status_bar_zero_matches() {
        let bar = StatusBar::new(0, Duration::from_millis(1), ".", 0);
        let status = bar.format_status();
        assert!(status.contains("0 matches"));
        assert!(status.contains("1.0ms"));
    }

    #[test]
    fn test_status_bar_large_numbers() {
        let bar = StatusBar::new(10_000, Duration::from_millis(2500), "/big/project", 5);
        let status = bar.format_status();
        assert!(status.contains("10000 matches"));
        assert!(status.contains("2500.0ms"));
        assert!(status.contains("5 filters"));
    }

    #[test]
    fn test_status_bar_searching_state() {
        let mut bar = StatusBar::new(50, Duration::ZERO, "/project", 0);
        bar.is_searching = true;
        bar.search_start = Some(Instant::now());
        let status = bar.format_status();
        assert!(status.contains("Searching..."));
        assert!(status.contains("50 matches"));
        assert!(status.contains("/project"));
    }

    // U-SB-3: Root abbreviation -- HOME path should be substituted with ~
    #[test]
    fn test_status_bar_root_abbreviation() {
        // Use the actual HOME env var to construct a path under it
        if let Ok(home) = std::env::var("HOME") {
            let root = format!("{}/projects/my-app", home);
            let bar = StatusBar::new(10, Duration::from_millis(5), &root, 0);
            let status = bar.format_status();
            // Should contain ~ substitution, not the full home path
            assert!(
                status.contains("~/projects/my-app"),
                "Expected ~ substitution in status: {status}"
            );
            assert!(
                !status.contains(&home),
                "Should not contain raw HOME path in status: {status}"
            );
        } else {
            // If HOME is not set, the path should appear as-is
            let bar = StatusBar::new(10, Duration::from_millis(5), "/some/path", 0);
            let status = bar.format_status();
            assert!(status.contains("/some/path"));
        }
    }

    #[test]
    fn test_status_bar_searching_without_start() {
        let mut bar = StatusBar::default();
        bar.is_searching = true;
        let status = bar.format_status();
        assert!(status.contains("Searching..."));
        assert!(status.contains("0.0s"));
    }

    // --- format_file_count tests ---

    #[test]
    fn test_format_file_count_exact() {
        assert_eq!(format_file_count(0), "0");
        assert_eq!(format_file_count(1), "1");
        assert_eq!(format_file_count(420), "420");
        assert_eq!(format_file_count(999), "999");
    }

    #[test]
    fn test_format_file_count_k() {
        assert_eq!(format_file_count(1_000), "1.0K");
        assert_eq!(format_file_count(1_500), "1.5K");
        assert_eq!(format_file_count(10_000), "10.0K");
        assert_eq!(format_file_count(99_999), "100.0K");
        assert_eq!(format_file_count(100_000), "100K");
        assert_eq!(format_file_count(420_000), "420K");
        assert_eq!(format_file_count(999_999), "1000K");
    }

    #[test]
    fn test_format_file_count_m() {
        assert_eq!(format_file_count(1_000_000), "1.0M");
        assert_eq!(format_file_count(1_300_000), "1.3M");
        assert_eq!(format_file_count(10_000_000), "10.0M");
    }

    // --- IndexState rendering tests ---

    #[test]
    fn test_index_status_loading() {
        let bar = StatusBar::default();
        let (text, color) = bar.index_status_text();
        assert_eq!(text, "Loading...");
        assert_eq!(color, theme::ACCENT);
    }

    #[test]
    fn test_index_status_building() {
        let mut bar = StatusBar::default();
        bar.index_state = IndexState::Building { files_indexed: 420_000 };
        let (text, color) = bar.index_status_text();
        assert_eq!(text, "Indexing: 420K files");
        assert_eq!(color, theme::ACCENT);
    }

    #[test]
    fn test_index_status_ready() {
        let mut bar = StatusBar::default();
        bar.index_state = IndexState::Ready { file_count: 1_300_000 };
        let (text, color) = bar.index_status_text();
        assert_eq!(text, "1.3M files");
        assert_eq!(color, theme::SUCCESS);
    }

    #[test]
    fn test_index_status_updating() {
        let mut bar = StatusBar::default();
        bar.index_state = IndexState::Updating;
        let (text, color) = bar.index_status_text();
        assert_eq!(text, "Updating...");
        assert_eq!(color, theme::ACCENT);
    }

    #[test]
    fn test_index_status_stale() {
        let mut bar = StatusBar::default();
        bar.index_state = IndexState::Stale { file_count: 1_300_000 };
        let (text, color) = bar.index_status_text();
        assert_eq!(text, "1.3M files (stale)");
        assert_eq!(color, theme::ACCENT);
    }

    #[test]
    fn test_index_status_error() {
        let mut bar = StatusBar::default();
        bar.index_state = IndexState::Error { message: "disk full".to_string() };
        let (text, color) = bar.index_status_text();
        assert_eq!(text, "Index: error");
        assert_eq!(color, theme::ERROR);
    }

    #[test]
    fn test_format_status_includes_index_state() {
        let mut bar = StatusBar::new(42, Duration::from_millis(5), "/project", 0);
        bar.index_state = IndexState::Ready { file_count: 500 };
        let status = bar.format_status();
        assert!(
            status.starts_with("500 files |"),
            "Expected index state prefix in: {status}"
        );
        assert!(status.contains("42 matches"));
    }

    #[test]
    fn test_format_status_building_state() {
        let mut bar = StatusBar::new(0, Duration::ZERO, "/", 0);
        bar.index_state = IndexState::Building { files_indexed: 1_500 };
        let status = bar.format_status();
        assert!(
            status.contains("Indexing: 1.5K files"),
            "Expected building text in: {status}"
        );
    }

    // --- Task 5.12: Named IndexState status bar rendering tests ---

    #[test]
    fn test_index_state_ready_display() {
        let mut bar = StatusBar::default();
        bar.index_state = IndexState::Ready { file_count: 1_300_000 };
        let (text, _color) = bar.index_status_text();
        assert_eq!(text, "1.3M files");
    }

    #[test]
    fn test_index_state_building_display() {
        let mut bar = StatusBar::default();
        bar.index_state = IndexState::Building { files_indexed: 420_000 };
        let (text, _color) = bar.index_status_text();
        assert_eq!(text, "Indexing: 420K files");
    }

    #[test]
    fn test_index_state_stale_display() {
        let mut bar = StatusBar::default();
        bar.index_state = IndexState::Stale { file_count: 1_300_000 };
        let (text, _color) = bar.index_status_text();
        assert_eq!(text, "1.3M files (stale)");
    }

    #[test]
    fn test_index_state_error_display() {
        let mut bar = StatusBar::default();
        bar.index_state = IndexState::Error { message: "corrupt".to_string() };
        let (text, _color) = bar.index_status_text();
        assert_eq!(text, "Index: error");
    }

    #[test]
    fn test_file_count_k_formatting() {
        assert_eq!(format_file_count(1_500), "1.5K");
    }

    #[test]
    fn test_file_count_m_formatting() {
        assert_eq!(format_file_count(1_300_000), "1.3M");
    }
}
