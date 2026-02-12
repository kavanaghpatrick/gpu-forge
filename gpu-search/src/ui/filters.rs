//! Filter pills for gpu-search.
//!
//! Provides dismissible filter pills for file extension (`.rs`), exclude directory
//! (`!target/`), and case-sensitive toggle (`Cmd+I`). Filters are applied before
//! GPU dispatch to reduce unnecessary work.

use std::fmt;

use eframe::egui;

use crate::search::types::SearchRequest;
use super::theme;

/// A search filter that can be applied before GPU dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SearchFilter {
    /// Filter to only include files with this extension (e.g., "rs", "toml").
    Extension(String),
    /// Exclude files under this directory name (e.g., "target", "node_modules").
    ExcludeDir(String),
    /// Enable case-sensitive search.
    CaseSensitive,
}

impl fmt::Display for SearchFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchFilter::Extension(ext) => write!(f, ".{}", ext),
            SearchFilter::ExcludeDir(dir) => write!(f, "!{}/", dir),
            SearchFilter::CaseSensitive => write!(f, "Aa"),
        }
    }
}

impl SearchFilter {
    /// Short label for the pill display.
    pub fn label(&self) -> String {
        self.to_string()
    }

    /// Descriptive tooltip for the filter pill.
    pub fn tooltip(&self) -> String {
        match self {
            SearchFilter::Extension(ext) => format!("Only .{} files", ext),
            SearchFilter::ExcludeDir(dir) => format!("Exclude {}/ directory", dir),
            SearchFilter::CaseSensitive => "Case-sensitive search".to_string(),
        }
    }
}

/// Filter bar managing a collection of active search filters.
///
/// Renders dismissible pills and provides methods to apply filters
/// to a SearchRequest before GPU dispatch.
pub struct FilterBar {
    /// Currently active filters.
    pub filters: Vec<SearchFilter>,
}

impl Default for FilterBar {
    fn default() -> Self {
        Self {
            filters: Vec::new(),
        }
    }
}

impl FilterBar {
    /// Create a new empty FilterBar.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a filter. Duplicate filters are ignored.
    pub fn add_filter(&mut self, filter: SearchFilter) {
        if !self.filters.contains(&filter) {
            self.filters.push(filter);
        }
    }

    /// Remove a filter by value. Returns true if the filter was found and removed.
    pub fn remove_filter(&mut self, filter: &SearchFilter) -> bool {
        let len_before = self.filters.len();
        self.filters.retain(|f| f != filter);
        self.filters.len() < len_before
    }

    /// Remove a filter by index. Returns the removed filter if valid.
    pub fn remove_at(&mut self, index: usize) -> Option<SearchFilter> {
        if index < self.filters.len() {
            Some(self.filters.remove(index))
        } else {
            None
        }
    }

    /// Toggle a filter: add if absent, remove if present.
    pub fn toggle_filter(&mut self, filter: SearchFilter) {
        if self.filters.contains(&filter) {
            self.remove_filter(&filter);
        } else {
            self.add_filter(filter);
        }
    }

    /// Number of active filters.
    pub fn count(&self) -> usize {
        self.filters.len()
    }

    /// Whether any filters are active.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Clear all filters.
    pub fn clear(&mut self) {
        self.filters.clear();
    }

    /// Check if case-sensitive filter is active.
    pub fn is_case_sensitive(&self) -> bool {
        self.filters.contains(&SearchFilter::CaseSensitive)
    }

    /// Get all active extension filters.
    pub fn extension_filters(&self) -> Vec<&str> {
        self.filters
            .iter()
            .filter_map(|f| match f {
                SearchFilter::Extension(ext) => Some(ext.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Get all active exclude directory filters.
    pub fn exclude_dir_filters(&self) -> Vec<&str> {
        self.filters
            .iter()
            .filter_map(|f| match f {
                SearchFilter::ExcludeDir(dir) => Some(dir.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Apply active filters to a SearchRequest, modifying it in place.
    ///
    /// - Extension filters set `file_types`
    /// - CaseSensitive filter sets `case_sensitive = true`
    /// - ExcludeDir filters are not directly in SearchRequest (applied during walk)
    pub fn apply_to_request(&self, request: &mut SearchRequest) {
        // Collect extension filters
        let extensions: Vec<String> = self
            .filters
            .iter()
            .filter_map(|f| match f {
                SearchFilter::Extension(ext) => Some(ext.clone()),
                _ => None,
            })
            .collect();

        if !extensions.is_empty() {
            request.file_types = Some(extensions);
        }

        // Apply case sensitivity
        if self.is_case_sensitive() {
            request.case_sensitive = true;
        }
    }

    /// Convert filters to search parameters.
    ///
    /// Returns a tuple of (file_types, excluded_dirs, case_sensitive).
    pub fn to_search_params(&self) -> (Option<Vec<String>>, Vec<String>, bool) {
        let extensions: Vec<String> = self
            .filters
            .iter()
            .filter_map(|f| match f {
                SearchFilter::Extension(ext) => Some(ext.clone()),
                _ => None,
            })
            .collect();

        let excluded_dirs: Vec<String> = self
            .filters
            .iter()
            .filter_map(|f| match f {
                SearchFilter::ExcludeDir(dir) => Some(dir.clone()),
                _ => None,
            })
            .collect();

        let file_types = if extensions.is_empty() {
            None
        } else {
            Some(extensions)
        };

        (file_types, excluded_dirs, self.is_case_sensitive())
    }

    /// Render the filter bar with dismissible pills.
    ///
    /// Returns the index of any pill that was dismissed (X button clicked),
    /// or None if no pill was dismissed.
    pub fn render(&mut self, ui: &mut egui::Ui) -> Option<usize> {
        if self.filters.is_empty() {
            return None;
        }

        let mut dismissed_index = None;

        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;

            for (i, filter) in self.filters.iter().enumerate() {
                let pill_dismissed = render_pill(ui, filter);
                if pill_dismissed {
                    dismissed_index = Some(i);
                }
            }
        });

        // Remove dismissed filter after iteration
        if let Some(idx) = dismissed_index {
            self.filters.remove(idx);
        }

        dismissed_index
    }
}

/// Render a single filter pill with an X dismiss button.
/// Returns true if the X button was clicked.
fn render_pill(ui: &mut egui::Ui, filter: &SearchFilter) -> bool {
    let mut dismissed = false;

    let label_text = filter.label();
    let tooltip_text = filter.tooltip();

    // Pill frame
    egui::Frame::new()
        .fill(theme::BG_SURFACE)
        .stroke(egui::Stroke::new(1.0, theme::BORDER))
        .corner_radius(egui::CornerRadius::same(12))
        .inner_margin(egui::Margin::symmetric(8, 3))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;

                // Filter label
                let color = match filter {
                    SearchFilter::CaseSensitive => theme::ACCENT,
                    _ => theme::TEXT_PRIMARY,
                };
                ui.colored_label(color, &label_text).on_hover_text(&tooltip_text);

                // Dismiss button (X)
                let x_response = ui.add(
                    egui::Button::new(
                        egui::RichText::new("\u{2715}") // Unicode multiplication sign (X)
                            .color(theme::TEXT_MUTED)
                            .size(10.0),
                    )
                    .frame(false),
                );
                if x_response.clicked() {
                    dismissed = true;
                }
                x_response.on_hover_text("Remove filter");
            });
        });

    dismissed
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_filter_pills_add_remove() {
        let mut bar = FilterBar::new();
        assert!(bar.is_empty());
        assert_eq!(bar.count(), 0);

        // Add extension filter
        bar.add_filter(SearchFilter::Extension("rs".to_string()));
        assert_eq!(bar.count(), 1);
        assert!(!bar.is_empty());

        // Add exclude dir filter
        bar.add_filter(SearchFilter::ExcludeDir("target".to_string()));
        assert_eq!(bar.count(), 2);

        // Add case sensitive
        bar.add_filter(SearchFilter::CaseSensitive);
        assert_eq!(bar.count(), 3);

        // Remove extension filter
        let removed = bar.remove_filter(&SearchFilter::Extension("rs".to_string()));
        assert!(removed);
        assert_eq!(bar.count(), 2);

        // Remove non-existent filter
        let removed = bar.remove_filter(&SearchFilter::Extension("py".to_string()));
        assert!(!removed);
        assert_eq!(bar.count(), 2);
    }

    #[test]
    fn test_filter_pills_no_duplicates() {
        let mut bar = FilterBar::new();
        bar.add_filter(SearchFilter::Extension("rs".to_string()));
        bar.add_filter(SearchFilter::Extension("rs".to_string()));
        assert_eq!(bar.count(), 1);
    }

    #[test]
    fn test_filter_pills_toggle() {
        let mut bar = FilterBar::new();

        // Toggle on
        bar.toggle_filter(SearchFilter::CaseSensitive);
        assert!(bar.is_case_sensitive());
        assert_eq!(bar.count(), 1);

        // Toggle off
        bar.toggle_filter(SearchFilter::CaseSensitive);
        assert!(!bar.is_case_sensitive());
        assert_eq!(bar.count(), 0);
    }

    #[test]
    fn test_filter_pills_remove_at() {
        let mut bar = FilterBar::new();
        bar.add_filter(SearchFilter::Extension("rs".to_string()));
        bar.add_filter(SearchFilter::ExcludeDir("target".to_string()));
        bar.add_filter(SearchFilter::CaseSensitive);

        // Remove middle element
        let removed = bar.remove_at(1);
        assert_eq!(removed, Some(SearchFilter::ExcludeDir("target".to_string())));
        assert_eq!(bar.count(), 2);

        // Remove out of bounds
        let removed = bar.remove_at(10);
        assert_eq!(removed, None);
    }

    #[test]
    fn test_filter_pills_clear() {
        let mut bar = FilterBar::new();
        bar.add_filter(SearchFilter::Extension("rs".to_string()));
        bar.add_filter(SearchFilter::CaseSensitive);
        assert_eq!(bar.count(), 2);

        bar.clear();
        assert!(bar.is_empty());
    }

    #[test]
    fn test_filter_pills_extension_filters() {
        let mut bar = FilterBar::new();
        bar.add_filter(SearchFilter::Extension("rs".to_string()));
        bar.add_filter(SearchFilter::Extension("toml".to_string()));
        bar.add_filter(SearchFilter::CaseSensitive);

        let exts = bar.extension_filters();
        assert_eq!(exts.len(), 2);
        assert!(exts.contains(&"rs"));
        assert!(exts.contains(&"toml"));
    }

    #[test]
    fn test_filter_pills_exclude_dir_filters() {
        let mut bar = FilterBar::new();
        bar.add_filter(SearchFilter::ExcludeDir("target".to_string()));
        bar.add_filter(SearchFilter::ExcludeDir("node_modules".to_string()));
        bar.add_filter(SearchFilter::Extension("rs".to_string()));

        let dirs = bar.exclude_dir_filters();
        assert_eq!(dirs.len(), 2);
        assert!(dirs.contains(&"target"));
        assert!(dirs.contains(&"node_modules"));
    }

    #[test]
    fn test_filter_pills_apply_to_request() {
        let mut bar = FilterBar::new();
        bar.add_filter(SearchFilter::Extension("rs".to_string()));
        bar.add_filter(SearchFilter::Extension("toml".to_string()));
        bar.add_filter(SearchFilter::CaseSensitive);
        bar.add_filter(SearchFilter::ExcludeDir("target".to_string()));

        let mut request = SearchRequest::new("test", "/src");
        assert!(!request.case_sensitive);
        assert!(request.file_types.is_none());

        bar.apply_to_request(&mut request);
        assert!(request.case_sensitive);
        assert_eq!(
            request.file_types,
            Some(vec!["rs".to_string(), "toml".to_string()])
        );
    }

    #[test]
    fn test_filter_pills_to_search_params() {
        let mut bar = FilterBar::new();
        bar.add_filter(SearchFilter::Extension("rs".to_string()));
        bar.add_filter(SearchFilter::ExcludeDir("target".to_string()));
        bar.add_filter(SearchFilter::CaseSensitive);

        let (file_types, excluded_dirs, case_sensitive) = bar.to_search_params();
        assert_eq!(file_types, Some(vec!["rs".to_string()]));
        assert_eq!(excluded_dirs, vec!["target".to_string()]);
        assert!(case_sensitive);
    }

    #[test]
    fn test_filter_pills_to_search_params_empty() {
        let bar = FilterBar::new();
        let (file_types, excluded_dirs, case_sensitive) = bar.to_search_params();
        assert!(file_types.is_none());
        assert!(excluded_dirs.is_empty());
        assert!(!case_sensitive);
    }

    #[test]
    fn test_filter_display() {
        assert_eq!(
            SearchFilter::Extension("rs".to_string()).to_string(),
            ".rs"
        );
        assert_eq!(
            SearchFilter::ExcludeDir("target".to_string()).to_string(),
            "!target/"
        );
        assert_eq!(SearchFilter::CaseSensitive.to_string(), "Aa");
    }

    #[test]
    fn test_filter_label_and_tooltip() {
        let ext = SearchFilter::Extension("rs".to_string());
        assert_eq!(ext.label(), ".rs");
        assert_eq!(ext.tooltip(), "Only .rs files");

        let dir = SearchFilter::ExcludeDir("target".to_string());
        assert_eq!(dir.label(), "!target/");
        assert_eq!(dir.tooltip(), "Exclude target/ directory");

        let cs = SearchFilter::CaseSensitive;
        assert_eq!(cs.label(), "Aa");
        assert_eq!(cs.tooltip(), "Case-sensitive search");
    }

    #[test]
    fn test_filter_pills_case_sensitive_check() {
        let mut bar = FilterBar::new();
        assert!(!bar.is_case_sensitive());

        bar.add_filter(SearchFilter::CaseSensitive);
        assert!(bar.is_case_sensitive());

        bar.remove_filter(&SearchFilter::CaseSensitive);
        assert!(!bar.is_case_sensitive());
    }
}
