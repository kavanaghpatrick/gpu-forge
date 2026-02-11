//! Full dashboard layout composition with responsive breakpoints.
//!
//! Provides the `render_ui` function that composes all TUI panels based on
//! terminal width:
//! - >= 120 cols: full three-panel layout (catalog | editor+results | GPU dashboard)
//! - 80-119 cols: two-panel layout (editor+results | GPU dashboard, catalog hidden)
//! - < 80 cols: minimal REPL mode (editor + results only, no chrome)
//!
//! Panel focus: Tab cycles, Ctrl+1/2/3 for direct panel selection.
//! Ctrl+Enter executes the current query from the editor.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::app::{AppState, FocusPanel};
use super::catalog;
use super::dashboard;
use super::gradient::gradient_text;
use super::results;
use crate::gpu::metrics::GpuMetricsCollector;

/// Layout mode determined by terminal width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    /// >= 120 columns: catalog (left) | editor+results (center) | GPU dashboard (right)
    Full,
    /// 80-119 columns: editor+results (left) | GPU dashboard (right), no catalog
    TwoPanel,
    /// < 80 columns: minimal REPL (editor + results stacked, no chrome)
    Minimal,
}

/// Determine the layout mode based on terminal width.
pub fn layout_mode(width: u16) -> LayoutMode {
    if width >= 120 {
        LayoutMode::Full
    } else if width >= 80 {
        LayoutMode::TwoPanel
    } else {
        LayoutMode::Minimal
    }
}

/// Compute content area panel split for the full (3-panel) layout.
/// Returns (catalog_pct, main_pct, dashboard_pct).
pub fn full_layout_percentages() -> (u16, u16, u16) {
    (18, 55, 27)
}

/// Compute content area panel split for the two-panel layout.
/// Returns (main_pct, dashboard_pct).
pub fn two_panel_percentages() -> (u16, u16) {
    (65, 35)
}

/// Compute the editor/results vertical split within the main panel.
/// Returns (editor_pct, results_pct).
pub fn editor_results_split() -> (u16, u16) {
    (30, 70)
}

/// Render the entire UI frame, dispatching to the appropriate layout mode.
pub fn render_ui(f: &mut Frame, app: &mut AppState, metrics: &GpuMetricsCollector) {
    let size = f.area();
    let mode = layout_mode(size.width);

    match mode {
        LayoutMode::Full => render_full_layout(f, size, app, metrics),
        LayoutMode::TwoPanel => render_two_panel_layout(f, size, app, metrics),
        LayoutMode::Minimal => render_minimal_layout(f, size, app),
    }
}

/// Full three-panel layout: title | catalog | editor+results | GPU dashboard | status
fn render_full_layout(
    f: &mut Frame,
    size: Rect,
    app: &mut AppState,
    metrics: &GpuMetricsCollector,
) {
    // Main vertical layout: title (3) + content (fill) + status (3)
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title bar
            Constraint::Min(5),    // content area
            Constraint::Length(3), // status bar
        ])
        .split(size);

    render_title_bar(f, main_chunks[0], app);

    // Content: catalog (18%) | main (55%) | GPU dashboard (27%)
    let (cat_pct, main_pct, dash_pct) = full_layout_percentages();
    let h_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(cat_pct),
            Constraint::Percentage(main_pct),
            Constraint::Percentage(dash_pct),
        ])
        .split(main_chunks[1]);

    // Left: catalog
    render_catalog_panel(f, h_chunks[0], app);

    // Center: editor (top) + results (bottom)
    let (ed_pct, res_pct) = editor_results_split();
    let v_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(ed_pct),
            Constraint::Percentage(res_pct),
        ])
        .split(h_chunks[1]);

    render_editor_panel(f, v_chunks[0], app);
    render_results_panel(f, v_chunks[1], app);

    // Right: GPU dashboard
    dashboard::render_gpu_dashboard(f, h_chunks[2], metrics, &app.theme, app);

    render_status_bar(f, main_chunks[2], app);
}

/// Two-panel layout (no catalog): title | editor+results | GPU dashboard | status
fn render_two_panel_layout(
    f: &mut Frame,
    size: Rect,
    app: &mut AppState,
    metrics: &GpuMetricsCollector,
) {
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title bar
            Constraint::Min(5),    // content area
            Constraint::Length(3), // status bar
        ])
        .split(size);

    render_title_bar(f, main_chunks[0], app);

    // Content: main (65%) | GPU dashboard (35%)
    let (main_pct, dash_pct) = two_panel_percentages();
    let h_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(main_pct),
            Constraint::Percentage(dash_pct),
        ])
        .split(main_chunks[1]);

    // Left: editor (top) + results (bottom)
    let (ed_pct, res_pct) = editor_results_split();
    let v_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(ed_pct),
            Constraint::Percentage(res_pct),
        ])
        .split(h_chunks[0]);

    render_editor_panel(f, v_chunks[0], app);
    render_results_panel(f, v_chunks[1], app);

    // Right: GPU dashboard
    dashboard::render_gpu_dashboard(f, h_chunks[1], metrics, &app.theme, app);

    render_status_bar(f, main_chunks[2], app);
}

/// Minimal REPL mode (no chrome, no GPU dashboard): just editor + results
fn render_minimal_layout(f: &mut Frame, size: Rect, app: &mut AppState) {
    // No title bar or status bar in minimal mode -- maximize usable area
    let v_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // compact editor
            Constraint::Min(3),    // results fill
        ])
        .split(size);

    render_editor_panel(f, v_chunks[0], app);
    render_results_panel(f, v_chunks[1], app);
}

// ---- Panel rendering helpers ----

/// Render the gradient-colored title bar.
fn render_title_bar(f: &mut Frame, area: Rect, app: &AppState) {
    let title_text = " gpu-query  GPU-Native Data Analytics ";
    let gradient_chars = gradient_text(title_text, &app.theme.title_gradient);

    let spans: Vec<Span> = gradient_chars
        .into_iter()
        .map(|(ch, color)| {
            Span::styled(
                ch.to_string(),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )
        })
        .collect();

    let title_line = Line::from(spans);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(app.theme.focus_border_style);

    let paragraph = Paragraph::new(title_line).block(block);
    f.render_widget(paragraph, area);
}

/// Render the catalog panel with tree view.
fn render_catalog_panel(f: &mut Frame, area: Rect, app: &mut AppState) {
    let is_focused = app.focus == FocusPanel::Catalog;

    catalog::render_catalog_tree(
        f,
        area,
        &mut app.catalog_state,
        is_focused,
        app.theme.border_style,
        app.theme.focus_border_style,
        app.theme.text,
        app.theme.muted,
        app.theme.accent,
        app.theme.selection,
    );
}

/// Render the query editor panel with syntax highlighting and cursor.
fn render_editor_panel(f: &mut Frame, area: Rect, app: &AppState) {
    let is_focused = app.focus == FocusPanel::Editor;
    let border_style = if is_focused {
        app.theme.focus_border_style
    } else {
        app.theme.border_style
    };

    // Use the EditorState if it has content, otherwise show placeholder
    let has_content = !app.editor_state.is_empty();

    let block = Block::default()
        .title(Span::styled(
            " Query (Ctrl+Enter to execute) ",
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(border_style);

    if has_content || is_focused {
        // Render syntax-highlighted editor with cursor
        let lines = if is_focused {
            app.editor_state.render_lines_with_cursor()
        } else {
            app.editor_state.render_lines()
        };
        let paragraph = Paragraph::new(lines).block(block);
        f.render_widget(paragraph, area);
    } else {
        // Show placeholder
        let style = Style::default().fg(app.theme.muted);
        let paragraph = Paragraph::new(Line::from(Span::styled(
            "SELECT * FROM ... WHERE ... LIMIT 10",
            style,
        )))
        .block(block);
        f.render_widget(paragraph, area);
    }
}

/// Render the results panel using the scrollable table widget.
fn render_results_panel(f: &mut Frame, area: Rect, app: &AppState) {
    results::render_results_table(f, area, app);
}

/// Render the status bar at the bottom.
fn render_status_bar(f: &mut Frame, area: Rect, app: &AppState) {
    let status_text = &app.status_message;
    let mode = layout_mode(area.width);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(app.theme.border_style);

    // Show focus panel indicator and keyboard shortcuts
    let focus_label = match app.focus {
        FocusPanel::Catalog => "Catalog",
        FocusPanel::Editor => "Editor",
        FocusPanel::Results => "Results",
    };

    let shortcuts = match mode {
        LayoutMode::Full => "Tab: cycle | Ctrl+1/2/3: panel | Ctrl+Enter: run",
        LayoutMode::TwoPanel => "Tab: cycle | Ctrl+Enter: run",
        LayoutMode::Minimal => "Ctrl+Enter: run",
    };

    let style = Style::default().fg(app.theme.muted);
    let paragraph = Paragraph::new(Line::from(vec![
        Span::styled(format!(" {} ", status_text), style),
        Span::styled(
            format!("| {} ", focus_label),
            Style::default().fg(app.theme.accent),
        ),
        Span::styled(
            format!("| {} ", shortcuts),
            Style::default().fg(app.theme.muted),
        ),
        Span::styled(
            format!("| Theme: {} ", app.theme.name),
            Style::default().fg(app.theme.accent),
        ),
    ]))
    .block(block);

    f.render_widget(paragraph, area);
}

// ---- Query execution from editor ----

/// Execute the SQL query currently in the editor.
///
/// Parses, optimizes, plans, and executes via GPU. Updates app state with results.
/// When the autonomous engine is live but the query requires fallback (e.g. ORDER BY),
/// sets engine_status to Fallback during execution and restores to Live afterward.
/// Returns Ok(()) on success, Err(msg) on failure (already sets app error).
pub fn execute_editor_query(app: &mut AppState) -> Result<(), String> {
    let sql = app.editor_state.text();
    if sql.trim().is_empty() {
        app.set_error("No query to execute".into());
        return Err("empty query".into());
    }

    app.query_state = super::app::QueryState::Running;
    app.status_message = "Executing query...".into();

    // Scan catalog
    let catalog = app
        .catalog_cache
        .get_or_refresh()
        .map(|s| s.to_vec())
        .map_err(|e| {
            let msg = format!("Catalog scan error: {}", e);
            app.set_error(msg.clone());
            msg
        })?;

    if catalog.is_empty() {
        let msg = format!("No data files in '{}'", app.data_dir.display());
        app.set_error(msg.clone());
        return Err(msg);
    }

    // Parse SQL
    let logical_plan = crate::sql::parser::parse_query(&sql).map_err(|e| {
        let msg = format!("SQL parse error: {}", e);
        app.set_error(msg.clone());
        msg
    })?;

    // Optimize
    let logical_plan = crate::sql::optimizer::optimize(logical_plan);

    // Plan
    let physical_plan = crate::sql::physical_plan::plan(&logical_plan).map_err(|e| {
        let msg = format!("Plan error: {:?}", e);
        app.set_error(msg.clone());
        msg
    })?;

    // Check autonomous compatibility and set fallback status if needed
    let has_autonomous = app.autonomous_executor.is_some();
    if has_autonomous {
        let compat =
            crate::gpu::autonomous::executor::check_autonomous_compatibility(&physical_plan);
        match compat {
            crate::gpu::autonomous::executor::CompatibilityResult::Fallback(reason) => {
                app.engine_status = super::app::EngineStatus::Fallback;
                app.last_fallback_reason = Some(reason);
                app.autonomous_stats.fallback_queries += 1;
            }
            crate::gpu::autonomous::executor::CompatibilityResult::Autonomous => {
                // Query is autonomous-compatible but being executed via standard path
                // (e.g., F5 during warm-up). No special handling needed.
            }
        }
    }

    // Execute on GPU -- use persistent executor from AppState (take/put-back for borrow safety)
    let mut executor =
        app.executor
            .take()
            .unwrap_or(crate::gpu::executor::QueryExecutor::new().map_err(|e| {
                let msg = format!("GPU init error: {}", e);
                app.set_error(msg.clone());
                msg
            })?);

    let start = std::time::Instant::now();
    let result = executor.execute(&physical_plan, &catalog);
    let elapsed = start.elapsed();

    // Put executor back before handling result (which needs &mut app)
    app.executor = Some(executor);

    let result = result.map_err(|e| {
        let msg = format!("Execution error: {}", e);
        app.set_error(msg.clone());
        msg
    })?;

    // Update timing
    let exec_us = elapsed.as_micros() as u64;
    app.last_exec_us = Some(exec_us);
    app.last_result_autonomous = false;

    // Store query in history
    if !sql.trim().is_empty() {
        app.history.push(sql.clone());
    }

    // Build status message with fallback reason if applicable
    let fallback_suffix = match &app.last_fallback_reason {
        Some(reason) if app.engine_status == super::app::EngineStatus::Fallback => {
            format!(" | standard path ({})", reason)
        }
        _ => String::new(),
    };

    app.status_message = format!(
        "Query completed: {} rows in {:.1}ms{}",
        result.row_count,
        exec_us as f64 / 1000.0,
        fallback_suffix,
    );

    app.set_result(result);

    // Restore engine status to Live if autonomous executor is still warm
    if has_autonomous && app.engine_status == super::app::EngineStatus::Fallback {
        app.engine_status = super::app::EngineStatus::Live;
    }

    Ok(())
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_mode_full() {
        assert_eq!(layout_mode(120), LayoutMode::Full);
        assert_eq!(layout_mode(200), LayoutMode::Full);
        assert_eq!(layout_mode(150), LayoutMode::Full);
    }

    #[test]
    fn test_layout_mode_two_panel() {
        assert_eq!(layout_mode(80), LayoutMode::TwoPanel);
        assert_eq!(layout_mode(100), LayoutMode::TwoPanel);
        assert_eq!(layout_mode(119), LayoutMode::TwoPanel);
    }

    #[test]
    fn test_layout_mode_minimal() {
        assert_eq!(layout_mode(79), LayoutMode::Minimal);
        assert_eq!(layout_mode(40), LayoutMode::Minimal);
        assert_eq!(layout_mode(0), LayoutMode::Minimal);
    }

    #[test]
    fn test_full_layout_percentages_sum() {
        let (a, b, c) = full_layout_percentages();
        assert_eq!(a + b + c, 100);
    }

    #[test]
    fn test_two_panel_percentages_sum() {
        let (a, b) = two_panel_percentages();
        assert_eq!(a + b, 100);
    }

    #[test]
    fn test_editor_results_split_sum() {
        let (a, b) = editor_results_split();
        assert_eq!(a + b, 100);
    }

    #[test]
    fn test_layout_mode_boundary_120() {
        // 120 is the boundary: >= 120 = Full
        assert_eq!(layout_mode(119), LayoutMode::TwoPanel);
        assert_eq!(layout_mode(120), LayoutMode::Full);
    }

    #[test]
    fn test_layout_mode_boundary_80() {
        // 80 is the boundary: >= 80 = TwoPanel
        assert_eq!(layout_mode(79), LayoutMode::Minimal);
        assert_eq!(layout_mode(80), LayoutMode::TwoPanel);
    }

    #[test]
    fn test_full_layout_catalog_not_too_wide() {
        let (cat, _, _) = full_layout_percentages();
        // Catalog should not take more than 25% of screen
        assert!(cat <= 25);
        assert!(cat >= 10);
    }

    #[test]
    fn test_editor_split_editor_smaller_than_results() {
        let (ed, res) = editor_results_split();
        // Editor should be smaller than results (query text is usually short)
        assert!(ed < res);
    }

    #[test]
    fn test_layout_mode_all_variants() {
        // Ensure all three modes are reachable
        let modes: Vec<LayoutMode> = (0..200).map(layout_mode).collect();
        assert!(modes.contains(&LayoutMode::Minimal));
        assert!(modes.contains(&LayoutMode::TwoPanel));
        assert!(modes.contains(&LayoutMode::Full));
    }
}
