//! Search bar widget with 100ms debounce.
//!
//! Renders a single-line `egui::TextEdit` with placeholder text, a search icon on the left,
//! and a filter toggle on the right. Implements a 100ms debounce timer: on each keystroke
//! the timer resets, and when elapsed > debounce_ms the query fires.
//!
//! Minimum query lengths: 1 character for filename search, 2 characters for content search.

use std::time::Instant;

use eframe::egui;

use super::theme;

/// Minimum query length for filename search.
pub const MIN_FILENAME_QUERY_LEN: usize = 1;

/// Minimum query length for content search.
pub const MIN_CONTENT_QUERY_LEN: usize = 2;

/// Default debounce duration in milliseconds.
const DEFAULT_DEBOUNCE_MS: u64 = 100;

/// Search mode determining minimum query length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Filename search: minimum 1 character.
    Filename,
    /// Content search: minimum 2 characters.
    Content,
}

impl SearchMode {
    /// Minimum query length for this search mode.
    pub fn min_query_len(self) -> usize {
        match self {
            SearchMode::Filename => MIN_FILENAME_QUERY_LEN,
            SearchMode::Content => MIN_CONTENT_QUERY_LEN,
        }
    }
}

/// Search bar with debounced input.
///
/// Tracks the current query string and a debounce timer. On each frame,
/// [`SearchBar::show`] renders the text input and returns `Some(query)` when
/// the debounce timer has expired since the last keystroke.
pub struct SearchBar {
    /// Current query text.
    pub query: String,

    /// Timestamp of the last text change.
    last_change: Option<Instant>,

    /// Debounce duration in milliseconds.
    debounce_ms: u64,

    /// Whether a search is pending (text changed, debounce not yet fired).
    pending_search: bool,

    /// Whether the filter toggle is active.
    pub filter_active: bool,

    /// Previous query snapshot (to detect changes).
    prev_query: String,
}

impl Default for SearchBar {
    fn default() -> Self {
        Self {
            query: String::new(),
            last_change: None,
            debounce_ms: DEFAULT_DEBOUNCE_MS,
            pending_search: false,
            filter_active: false,
            prev_query: String::new(),
        }
    }
}

impl SearchBar {
    /// Create a new SearchBar with the given debounce duration.
    pub fn with_debounce_ms(debounce_ms: u64) -> Self {
        Self {
            debounce_ms,
            ..Default::default()
        }
    }

    /// Render the search bar and return `Some(query)` when debounce fires.
    ///
    /// Returns `Some(query)` when:
    /// - Text has changed and debounce_ms elapsed since last change
    /// - Query meets minimum length for the given search mode
    ///
    /// Returns `None` otherwise (debounce still pending, or query too short).
    pub fn show(&mut self, ui: &mut egui::Ui, mode: SearchMode) -> Option<String> {
        let mut fired_query = None;

        ui.horizontal(|ui| {
            // Search icon (left)
            ui.colored_label(theme::TEXT_MUTED, "\u{1F50D}");

            // Text input
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.query)
                    .hint_text("Search files and content...")
                    .desired_width(ui.available_width() - 30.0)
                    .text_color(theme::TEXT_PRIMARY),
            );

            // Detect text change
            if self.query != self.prev_query {
                self.last_change = Some(Instant::now());
                self.pending_search = true;
                self.prev_query = self.query.clone();
            }

            // Request repaint while debounce is pending so we check again next frame
            if self.pending_search {
                response.ctx.request_repaint();
            }

            // Filter toggle (right)
            let filter_label = if self.filter_active {
                egui::RichText::new("\u{2699}").color(theme::ACCENT)
            } else {
                egui::RichText::new("\u{2699}").color(theme::TEXT_MUTED)
            };
            if ui.button(filter_label).clicked() {
                self.filter_active = !self.filter_active;
            }
        });

        // Check debounce timer
        if self.pending_search {
            if let Some(last_change) = self.last_change {
                let elapsed_ms = last_change.elapsed().as_millis() as u64;
                if elapsed_ms >= self.debounce_ms {
                    self.pending_search = false;
                    let min_len = mode.min_query_len();
                    if self.query.len() >= min_len {
                        fired_query = Some(self.query.clone());
                    } else if self.query.is_empty() {
                        // Fire empty query to clear results
                        fired_query = Some(String::new());
                    }
                }
            }
        }

        fired_query
    }

    /// Whether a search is currently pending (debounce hasn't fired yet).
    pub fn has_pending_search(&self) -> bool {
        self.pending_search
    }

    /// Suppress the next debounce trigger after programmatic query changes.
    ///
    /// When code sets `self.query` directly (e.g., auto-search from CLI args),
    /// `show()` would detect `query != prev_query` and set a 100ms debounce
    /// timer, causing a duplicate dispatch. Call this after setting `query`
    /// to sync `prev_query` and prevent that.
    pub fn suppress_next_debounce(&mut self) {
        self.prev_query = self.query.clone();
        self.pending_search = false;
        self.last_change = None;
    }

    /// Clear the search bar state.
    pub fn clear(&mut self) {
        self.query.clear();
        self.prev_query.clear();
        self.last_change = None;
        self.pending_search = false;
    }

    /// Record a text change at the given instant (for testing).
    #[cfg(test)]
    pub(crate) fn simulate_input(&mut self, text: &str, at: Instant) {
        self.query = text.to_string();
        self.prev_query = String::new(); // Force change detection
        self.last_change = Some(at);
        self.pending_search = true;
    }

    /// Check if debounce has expired given the current time.
    /// Returns `Some(query)` if debounce fired and query meets min length for mode.
    #[cfg(test)]
    pub(crate) fn check_debounce(&mut self, mode: SearchMode) -> Option<String> {
        if self.pending_search {
            if let Some(last_change) = self.last_change {
                let elapsed_ms = last_change.elapsed().as_millis() as u64;
                if elapsed_ms >= self.debounce_ms {
                    self.pending_search = false;
                    let min_len = mode.min_query_len();
                    if self.query.len() >= min_len {
                        return Some(self.query.clone());
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_debounce_logic() {
        let mut bar = SearchBar::with_debounce_ms(30);

        // Initially no pending search
        assert!(!bar.has_pending_search());

        // Simulate input
        let now = Instant::now();
        bar.simulate_input("te", now);

        // Should be pending immediately
        assert!(bar.has_pending_search());

        // Should NOT fire before debounce expires
        assert!(bar.check_debounce(SearchMode::Content).is_none());
        assert!(bar.has_pending_search());

        // Wait for debounce to expire
        thread::sleep(Duration::from_millis(35));

        // Now should fire for content search (2 chars meets min)
        let result = bar.check_debounce(SearchMode::Content);
        assert_eq!(result, Some("te".to_string()));
        assert!(!bar.has_pending_search());
    }

    #[test]
    fn test_debounce_filename_mode_single_char() {
        let mut bar = SearchBar::with_debounce_ms(30);
        let now = Instant::now();
        bar.simulate_input("a", now);

        thread::sleep(Duration::from_millis(35));

        // Filename mode: 1 char is enough
        let result = bar.check_debounce(SearchMode::Filename);
        assert_eq!(result, Some("a".to_string()));
    }

    #[test]
    fn test_debounce_content_mode_single_char_rejected() {
        let mut bar = SearchBar::with_debounce_ms(30);
        let now = Instant::now();
        bar.simulate_input("a", now);

        thread::sleep(Duration::from_millis(35));

        // Content mode: 1 char is not enough
        let result = bar.check_debounce(SearchMode::Content);
        assert!(result.is_none());
    }

    #[test]
    fn test_debounce_reset_on_new_input() {
        let mut bar = SearchBar::with_debounce_ms(30);

        // First input
        let now = Instant::now();
        bar.simulate_input("fo", now);

        thread::sleep(Duration::from_millis(15));

        // Second input resets timer
        bar.simulate_input("foo", Instant::now());
        assert!(bar.has_pending_search());

        // Should not fire yet (only ~0ms since second input)
        assert!(bar.check_debounce(SearchMode::Content).is_none());

        // Wait for debounce
        thread::sleep(Duration::from_millis(35));

        let result = bar.check_debounce(SearchMode::Content);
        assert_eq!(result, Some("foo".to_string()));
    }

    #[test]
    fn test_clear() {
        let mut bar = SearchBar::with_debounce_ms(30);
        bar.simulate_input("test", Instant::now());
        assert!(bar.has_pending_search());

        bar.clear();
        assert!(!bar.has_pending_search());
        assert!(bar.query.is_empty());
    }

    #[test]
    fn test_default_debounce_is_100ms() {
        let bar = SearchBar::default();
        assert_eq!(bar.debounce_ms, 100);
    }

    #[test]
    fn test_search_mode_min_len() {
        assert_eq!(SearchMode::Filename.min_query_len(), 1);
        assert_eq!(SearchMode::Content.min_query_len(), 2);
    }
}
