//! Application state model for the TUI dashboard.
//!
//! Holds query state, results, GPU metrics, running flags, and theme.

use super::catalog::CatalogState;
use super::results::ResultsState;
use super::themes::Theme;
use crate::gpu::executor::QueryResult;
use crate::gpu::metrics::QueryMetrics;
use std::path::PathBuf;

/// Focus panel in the dashboard layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPanel {
    /// File catalog / table tree.
    Catalog,
    /// SQL query editor.
    Editor,
    /// Query results display.
    Results,
}

/// Current state of query execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryState {
    /// No query running, idle.
    Idle,
    /// Query is currently executing on GPU.
    Running,
    /// Query completed with result.
    Complete,
    /// Query failed with error message.
    Error(String),
}

/// Main application state for the dashboard.
pub struct AppState {
    /// Whether the app is running (false = quit).
    pub running: bool,

    /// Current focused panel.
    pub focus: FocusPanel,

    /// Data directory being queried.
    pub data_dir: PathBuf,

    /// Current SQL text in the editor.
    pub query_text: String,

    /// Query execution state.
    pub query_state: QueryState,

    /// Most recent query result.
    pub last_result: Option<QueryResult>,

    /// Query history (most recent first).
    pub history: Vec<String>,

    /// Current theme.
    pub theme: Theme,

    /// Frame counter for FPS tracking.
    pub frame_count: u64,

    /// Last execution time in microseconds.
    pub last_exec_us: Option<u64>,

    /// Last query metrics (for CPU comparison and performance line).
    pub last_query_metrics: Option<QueryMetrics>,

    /// Table names discovered in data directory.
    pub tables: Vec<String>,

    /// Status bar message.
    pub status_message: String,

    /// Tick rate in milliseconds (target ~60fps = 16ms).
    pub tick_rate_ms: u64,

    /// Results table pagination/scroll state.
    pub results_state: ResultsState,

    /// Data catalog tree view state.
    pub catalog_state: CatalogState,
}

impl AppState {
    /// Create a new AppState with the given data directory and theme name.
    pub fn new(data_dir: PathBuf, theme_name: &str) -> Self {
        Self {
            running: true,
            focus: FocusPanel::Editor,
            data_dir,
            query_text: String::new(),
            query_state: QueryState::Idle,
            last_result: None,
            history: Vec::new(),
            theme: Theme::by_name(theme_name),
            frame_count: 0,
            last_exec_us: None,
            last_query_metrics: None,
            tables: Vec::new(),
            status_message: "Ready. Type SQL and press Ctrl+Enter to execute.".into(),
            tick_rate_ms: 16, // ~60fps
            results_state: ResultsState::new(),
            catalog_state: CatalogState::new(),
        }
    }

    /// Cycle focus to the next panel.
    pub fn cycle_focus(&mut self) {
        self.focus = match self.focus {
            FocusPanel::Catalog => FocusPanel::Editor,
            FocusPanel::Editor => FocusPanel::Results,
            FocusPanel::Results => FocusPanel::Catalog,
        };
    }

    /// Set the query result.
    pub fn set_result(&mut self, result: QueryResult) {
        self.query_state = QueryState::Complete;
        self.results_state.reset();
        self.last_result = Some(result);
    }

    /// Set query error.
    pub fn set_error(&mut self, msg: String) {
        self.query_state = QueryState::Error(msg.clone());
        self.status_message = format!("Error: {}", msg);
    }

    /// Increment frame counter.
    pub fn tick(&mut self) {
        self.frame_count = self.frame_count.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_app_state_new() {
        let state = AppState::new(PathBuf::from("/tmp"), "thermal");
        assert!(state.running);
        assert_eq!(state.focus, FocusPanel::Editor);
        assert_eq!(state.query_state, QueryState::Idle);
        assert!(state.last_result.is_none());
        assert_eq!(state.theme.name, "thermal");
        assert_eq!(state.tick_rate_ms, 16);
    }

    #[test]
    fn test_cycle_focus() {
        let mut state = AppState::new(PathBuf::from("/tmp"), "thermal");
        assert_eq!(state.focus, FocusPanel::Editor);
        state.cycle_focus();
        assert_eq!(state.focus, FocusPanel::Results);
        state.cycle_focus();
        assert_eq!(state.focus, FocusPanel::Catalog);
        state.cycle_focus();
        assert_eq!(state.focus, FocusPanel::Editor);
    }

    #[test]
    fn test_set_result() {
        let mut state = AppState::new(PathBuf::from("/tmp"), "thermal");
        let result = QueryResult {
            columns: vec!["count".into()],
            rows: vec![vec!["42".into()]],
            row_count: 1,
        };
        state.set_result(result);
        assert_eq!(state.query_state, QueryState::Complete);
        assert!(state.last_result.is_some());
    }

    #[test]
    fn test_set_error() {
        let mut state = AppState::new(PathBuf::from("/tmp"), "thermal");
        state.set_error("parse error".into());
        assert!(matches!(state.query_state, QueryState::Error(_)));
    }

    #[test]
    fn test_tick() {
        let mut state = AppState::new(PathBuf::from("/tmp"), "thermal");
        assert_eq!(state.frame_count, 0);
        state.tick();
        assert_eq!(state.frame_count, 1);
        state.tick();
        assert_eq!(state.frame_count, 2);
    }
}
