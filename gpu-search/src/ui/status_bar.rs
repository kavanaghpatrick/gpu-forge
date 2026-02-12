//! Status bar widget for gpu-search.
//!
//! Displays match count, elapsed search time, search root path, and active
//! filter count in a horizontal bar at the bottom of the search panel.

use std::path::PathBuf;
use std::time::Duration;

use eframe::egui;

use super::theme;

/// Status bar displaying search statistics.
///
/// Shows: "N matches in Xms | /path/to/root | 2 filters"
/// Uses TEXT_MUTED for labels and TEXT_PRIMARY for values.
pub struct StatusBar {
    /// Number of matches found.
    pub match_count: usize,
    /// Time elapsed for the search.
    pub elapsed: Duration,
    /// Root directory being searched.
    pub search_root: PathBuf,
    /// Number of active filters.
    pub active_filter_count: usize,
}

impl Default for StatusBar {
    fn default() -> Self {
        Self {
            match_count: 0,
            elapsed: Duration::ZERO,
            search_root: PathBuf::from("."),
            active_filter_count: 0,
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
            match_count,
            elapsed,
            search_root: search_root.into(),
            active_filter_count,
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

            // Match count
            ui.colored_label(theme::TEXT_MUTED, "Matches:");
            ui.colored_label(
                theme::TEXT_PRIMARY,
                format!("{}", self.match_count),
            );

            // Elapsed time
            ui.colored_label(theme::TEXT_MUTED, "in");
            ui.colored_label(
                theme::TEXT_PRIMARY,
                format!("{}ms", self.elapsed.as_millis()),
            );

            // Separator
            ui.colored_label(theme::BORDER, "|");

            // Search root (truncated if too long)
            ui.colored_label(theme::TEXT_MUTED, "Root:");
            let root_str = self.search_root.display().to_string();
            let display_root = truncate_path(&root_str, 40);
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
        let root_str = self.search_root.display().to_string();
        let display_root = truncate_path(&root_str, 40);
        let mut status = format!(
            "{} matches in {}ms | {}",
            self.match_count,
            self.elapsed.as_millis(),
            display_root,
        );
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
        assert_eq!(bar.match_count, 0);
        assert_eq!(bar.elapsed, Duration::ZERO);
        assert_eq!(bar.search_root, PathBuf::from("."));
        assert_eq!(bar.active_filter_count, 0);
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
        assert!(status.contains("12ms"));
        assert!(status.contains("/src"));
        assert!(!status.contains("filter"));
    }

    #[test]
    fn test_status_bar_format_with_filters() {
        let bar = StatusBar::new(42, Duration::from_millis(5), "/project", 2);
        let status = bar.format_status();
        assert!(status.contains("42 matches"));
        assert!(status.contains("5ms"));
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
        assert!(status.contains("1ms"));
    }

    #[test]
    fn test_status_bar_large_numbers() {
        let bar = StatusBar::new(10_000, Duration::from_millis(2500), "/big/project", 5);
        let status = bar.format_status();
        assert!(status.contains("10000 matches"));
        assert!(status.contains("2500ms"));
        assert!(status.contains("5 filters"));
    }
}
