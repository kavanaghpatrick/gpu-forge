//! Application state model for the TUI dashboard.
//!
//! Holds query state, results, GPU metrics, running flags, and theme.

use super::catalog::CatalogState;
use super::editor::EditorState;
use super::results::ResultsState;
use super::themes::Theme;
use crate::gpu::autonomous::executor::{AutonomousExecutor, CompatibilityResult};
use crate::gpu::executor::{QueryExecutor, QueryResult};
use crate::gpu::metrics::{GpuMetricsCollector, PipelineProfile, QueryMetrics};
use crate::io::catalog_cache::CatalogCache;
use crate::sql::physical_plan::PhysicalPlan;
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

/// Status of the autonomous GPU engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineStatus {
    /// Engine not initialized.
    Off,
    /// Loading tables into GPU-resident buffers.
    WarmingUp,
    /// JIT-compiling a new kernel variant.
    Compiling,
    /// Engine ready and actively processing queries.
    Live,
    /// Engine idle (no recent queries, will wake on next submit).
    Idle,
    /// Query routed to standard executor (unsupported pattern).
    Fallback,
    /// Engine encountered an error.
    Error,
}

/// SQL text validity state for live mode feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlValidity {
    /// Editor is empty.
    Empty,
    /// SQL is syntactically incomplete (e.g., missing semicolon or clause).
    Incomplete,
    /// SQL has a parse error.
    ParseError,
    /// SQL parses successfully.
    Valid,
}

/// Whether the current query can use the autonomous engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryCompatibility {
    /// Not yet determined.
    Unknown,
    /// Query is compatible with autonomous engine.
    Autonomous,
    /// Query requires fallback to standard executor.
    Fallback,
    /// Query is invalid (cannot be executed by either path).
    Invalid,
}

/// TUI-side statistics for autonomous engine display.
#[derive(Debug, Clone)]
pub struct TuiAutonomousStats {
    /// Total autonomous queries executed.
    pub total_queries: u64,
    /// Queries that fell back to standard executor.
    pub fallback_queries: u64,
    /// Average latency in microseconds.
    pub avg_latency_us: f64,
    /// 99th percentile latency in microseconds.
    pub p99_latency_us: u64,
    /// Count of consecutive sub-1ms query results.
    pub consecutive_sub_1ms: u64,
}

impl Default for TuiAutonomousStats {
    fn default() -> Self {
        Self {
            total_queries: 0,
            fallback_queries: 0,
            avg_latency_us: 0.0,
            p99_latency_us: 0,
            consecutive_sub_1ms: 0,
        }
    }
}

/// Current state of query execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryState {
    /// No query running, idle.
    Idle,
    /// Query is currently executing on GPU.
    Running,
    /// Query submitted to autonomous engine (polling for result).
    AutonomousSubmitted,
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

    /// Multi-line SQL editor state (cursor, lines, syntax highlighting).
    pub editor_state: EditorState,

    /// GPU metrics collector for dashboard sparklines and utilization bars.
    pub gpu_metrics: GpuMetricsCollector,

    /// Whether profile mode is enabled (.profile on/off).
    pub profile_mode: bool,

    /// Last pipeline profile (per-kernel timing breakdown).
    pub last_pipeline_profile: Option<PipelineProfile>,

    /// Persistent GPU query executor (lazily initialized on first query).
    pub executor: Option<QueryExecutor>,

    /// Cached catalog of tables in the data directory.
    pub catalog_cache: CatalogCache,

    // -- Autonomous engine state fields --

    /// Autonomous GPU executor (initialized during warm-up).
    pub autonomous_executor: Option<AutonomousExecutor>,

    /// Whether live mode is active (queries execute on every keystroke).
    pub live_mode: bool,

    /// Current autonomous engine status.
    pub engine_status: EngineStatus,

    /// Warm-up progress (0.0 to 1.0).
    pub warmup_progress: f32,

    /// Last autonomous query latency in microseconds.
    pub last_autonomous_us: Option<u64>,

    /// Autonomous engine statistics for dashboard display.
    pub autonomous_stats: TuiAutonomousStats,

    /// Current SQL text validity for live mode feedback.
    pub sql_validity: SqlValidity,

    /// Whether current query is autonomous-compatible.
    pub query_compatibility: QueryCompatibility,

    /// Cached physical plan for current SQL (avoids re-planning on each tick).
    pub cached_plan: Option<PhysicalPlan>,

    /// Whether the last completed result came from the autonomous engine.
    pub last_result_autonomous: bool,

    /// Fallback reason when the last query used the standard path.
    pub last_fallback_reason: Option<String>,
}

impl AppState {
    /// Create a new AppState with the given data directory and theme name.
    pub fn new(data_dir: PathBuf, theme_name: &str) -> Self {
        let catalog_cache = CatalogCache::new(data_dir.clone());
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
            editor_state: EditorState::new(),
            gpu_metrics: GpuMetricsCollector::new(),
            profile_mode: false,
            last_pipeline_profile: None,
            executor: None,
            catalog_cache,
            autonomous_executor: None,
            live_mode: false,
            engine_status: EngineStatus::Off,
            warmup_progress: 0.0,
            last_autonomous_us: None,
            autonomous_stats: TuiAutonomousStats::default(),
            sql_validity: SqlValidity::Empty,
            query_compatibility: QueryCompatibility::Unknown,
            cached_plan: None,
            last_result_autonomous: false,
            last_fallback_reason: None,
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

    /// Get or lazily initialize the persistent GPU query executor.
    ///
    /// The executor is created on first use (not at TUI startup) to avoid
    /// Metal device initialization overhead when the user hasn't run a query yet.
    pub fn get_or_init_executor(&mut self) -> Result<&mut QueryExecutor, String> {
        if self.executor.is_none() {
            self.executor = Some(QueryExecutor::new()?);
        }
        Ok(self.executor.as_mut().unwrap())
    }

    /// Increment frame counter.
    pub fn tick(&mut self) {
        self.frame_count = self.frame_count.wrapping_add(1);
    }
}

/// Update the SQL validity state based on the current editor text.
///
/// Attempts to parse the SQL text and sets `app.sql_validity` to:
/// - `Empty` if the text is blank
/// - `Valid` if it parses successfully (also caches the physical plan)
/// - `Incomplete` if it looks like a partial statement (common typing patterns)
/// - `ParseError` otherwise
pub fn update_sql_validity(app: &mut AppState) {
    let text = app.editor_state.text();
    let trimmed = text.trim();

    if trimmed.is_empty() {
        app.sql_validity = SqlValidity::Empty;
        app.cached_plan = None;
        return;
    }

    // Try to parse
    match crate::sql::parser::parse_query(trimmed) {
        Ok(logical_plan) => {
            // Parse succeeded -- try to plan
            match crate::sql::physical_plan::plan(&logical_plan) {
                Ok(physical) => {
                    app.sql_validity = SqlValidity::Valid;
                    app.cached_plan = Some(physical);
                }
                Err(_) => {
                    // Parses but cannot be planned (e.g., unsupported expression)
                    app.sql_validity = SqlValidity::ParseError;
                    app.cached_plan = None;
                }
            }
        }
        Err(_) => {
            // Heuristic: if it starts with SELECT but doesn't parse, it's likely incomplete
            let upper = trimmed.to_uppercase();
            if upper.starts_with("SELECT") && !trimmed.ends_with(';') {
                app.sql_validity = SqlValidity::Incomplete;
            } else {
                app.sql_validity = SqlValidity::ParseError;
            }
            app.cached_plan = None;
        }
    }
}

/// Update the query compatibility state based on the cached physical plan.
///
/// If `app.cached_plan` is `Some`, runs the autonomous compatibility check.
/// Otherwise sets `Unknown`.
pub fn update_query_compatibility(app: &mut AppState) {
    match &app.cached_plan {
        Some(plan) => {
            let result =
                crate::gpu::autonomous::executor::check_autonomous_compatibility(plan);
            app.query_compatibility = match result {
                CompatibilityResult::Autonomous => {
                    app.last_fallback_reason = None;
                    QueryCompatibility::Autonomous
                }
                CompatibilityResult::Fallback(reason) => {
                    app.last_fallback_reason = Some(reason);
                    QueryCompatibility::Fallback
                }
            };
        }
        None => {
            app.query_compatibility = match app.sql_validity {
                SqlValidity::Empty | SqlValidity::Incomplete => QueryCompatibility::Unknown,
                SqlValidity::ParseError => QueryCompatibility::Invalid,
                SqlValidity::Valid => QueryCompatibility::Unknown,
            };
        }
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
