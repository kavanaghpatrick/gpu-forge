//! Scrollable results table widget with streaming pagination.
//!
//! Renders query results as a formatted table with:
//! - Number formatting (thousands separators, right-aligned numerics)
//! - NULL rendering (dim gray)
//! - Column auto-width based on content and header
//! - Pagination for large result sets (Space: next page, arrow keys: scroll)
//! - Performance line below results

use ratatui::{
    layout::{Constraint, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Row, Table},
    Frame,
};

use super::app::AppState;
use super::themes::Theme;
use crate::gpu::executor::QueryResult;
use crate::gpu::metrics::{self, PipelineProfile, QueryMetrics};

/// Pagination state for the results table.
#[derive(Debug, Clone)]
pub struct ResultsState {
    /// Current page (0-indexed).
    pub current_page: usize,
    /// Number of rows per page (computed from available height).
    pub page_size: usize,
    /// Scroll offset within the current page (for arrow key scrolling).
    pub scroll_offset: usize,
    /// Selected row index (global, across all pages).
    pub selected_row: usize,
}

impl Default for ResultsState {
    fn default() -> Self {
        Self {
            current_page: 0,
            page_size: 50,
            scroll_offset: 0,
            selected_row: 0,
        }
    }
}

impl ResultsState {
    /// Create a new results state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset pagination (e.g., when a new query completes).
    pub fn reset(&mut self) {
        self.current_page = 0;
        self.scroll_offset = 0;
        self.selected_row = 0;
    }

    /// Total number of pages for the given row count.
    pub fn total_pages(&self, total_rows: usize) -> usize {
        if self.page_size == 0 {
            return 1;
        }
        total_rows.div_ceil(self.page_size)
    }

    /// Go to next page.
    pub fn next_page(&mut self, total_rows: usize) {
        let max_page = self.total_pages(total_rows).saturating_sub(1);
        if self.current_page < max_page {
            self.current_page += 1;
            self.scroll_offset = 0;
            self.selected_row = self.current_page * self.page_size;
        }
    }

    /// Go to previous page.
    pub fn prev_page(&mut self, total_rows: usize) {
        if self.current_page > 0 {
            self.current_page -= 1;
            self.scroll_offset = 0;
            self.selected_row = self.current_page * self.page_size;
        }
        let _ = total_rows; // suppress unused warning
    }

    /// Move selection down by one row.
    pub fn scroll_down(&mut self, total_rows: usize) {
        if total_rows == 0 {
            return;
        }
        if self.selected_row + 1 < total_rows {
            self.selected_row += 1;
            // Auto-advance page if needed
            let new_page = self.selected_row / self.page_size;
            if new_page != self.current_page {
                self.current_page = new_page;
                self.scroll_offset = 0;
            }
        }
    }

    /// Move selection up by one row.
    pub fn scroll_up(&mut self, _total_rows: usize) {
        if self.selected_row > 0 {
            self.selected_row -= 1;
            let new_page = self.selected_row / self.page_size;
            if new_page != self.current_page {
                self.current_page = new_page;
                self.scroll_offset = 0;
            }
        }
    }

    /// First row index visible on the current page.
    pub fn page_start(&self) -> usize {
        self.current_page * self.page_size
    }

    /// Last row index (exclusive) visible on the current page.
    pub fn page_end(&self, total_rows: usize) -> usize {
        let end = self.page_start() + self.page_size;
        end.min(total_rows)
    }
}

// ---- Number formatting ----

/// Format an integer with thousands separators (e.g., 1234567 -> "1,234,567").
pub fn format_number(s: &str) -> String {
    // Try to parse as integer
    if let Ok(n) = s.parse::<i64>() {
        return format_i64(n);
    }
    // Try to parse as float
    if let Ok(f) = s.parse::<f64>() {
        return format_f64(f);
    }
    // Return as-is if not numeric
    s.to_string()
}

/// Format an i64 with thousands separators.
pub fn format_i64(n: i64) -> String {
    let negative = n < 0;
    let abs = if n == i64::MIN {
        // Handle i64::MIN overflow
        "9223372036854775808".to_string()
    } else {
        n.unsigned_abs().to_string()
    };

    let with_sep = insert_thousands_sep(&abs);
    if negative {
        format!("-{}", with_sep)
    } else {
        with_sep
    }
}

/// Format an f64 with thousands separators on the integer part.
pub fn format_f64(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f.is_sign_positive() { "Inf" } else { "-Inf" }.to_string();
    }

    let s = format!("{:.2}", f);
    if let Some(dot_pos) = s.find('.') {
        let int_part = &s[..dot_pos];
        let frac_part = &s[dot_pos..];

        let negative = int_part.starts_with('-');
        let digits = if negative { &int_part[1..] } else { int_part };
        let formatted_int = insert_thousands_sep(digits);

        if negative {
            format!("-{}{}", formatted_int, frac_part)
        } else {
            format!("{}{}", formatted_int, frac_part)
        }
    } else {
        s
    }
}

/// Insert comma thousands separators into a digit string.
fn insert_thousands_sep(digits: &str) -> String {
    let len = digits.len();
    if len <= 3 {
        return digits.to_string();
    }

    let mut result = String::with_capacity(len + len / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(ch);
    }
    result
}

/// Check if a string value looks numeric (integer or float).
pub fn is_numeric(s: &str) -> bool {
    if s.is_empty() || s.eq_ignore_ascii_case("null") {
        return false;
    }
    s.parse::<i64>().is_ok() || s.parse::<f64>().is_ok()
}

/// Check if a cell value represents NULL.
pub fn is_null(s: &str) -> bool {
    s.eq_ignore_ascii_case("null") || s == "NULL" || s.is_empty()
}

// ---- Column width computation ----

/// Compute optimal column widths based on header and data content.
/// Returns a Vec of widths (in characters) for each column.
pub fn compute_column_widths(result: &QueryResult, max_total_width: u16) -> Vec<u16> {
    if result.columns.is_empty() {
        return vec![];
    }

    let ncols = result.columns.len();
    let mut widths: Vec<u16> = vec![0; ncols];

    // Start with header widths
    for (i, col) in result.columns.iter().enumerate() {
        widths[i] = col.len() as u16;
    }

    // Scan data rows for max widths (sample first 200 rows for performance)
    let sample_rows = result.rows.len().min(200);
    for row in result.rows.iter().take(sample_rows) {
        for (i, val) in row.iter().enumerate() {
            if i < ncols {
                let formatted = format_number(val);
                let w = formatted.len() as u16;
                if w > widths[i] {
                    widths[i] = w;
                }
            }
        }
    }

    // Add padding (2 chars per column)
    for w in widths.iter_mut() {
        *w += 2;
    }

    // Clamp individual columns
    for w in widths.iter_mut() {
        *w = (*w).clamp(4, 40);
    }

    // If total exceeds available width, scale down proportionally
    let total: u16 = widths.iter().sum();
    let available = max_total_width.saturating_sub(ncols as u16 + 1); // borders/separators
    if total > available && available > 0 {
        let scale = available as f32 / total as f32;
        for w in widths.iter_mut() {
            *w = ((*w as f32) * scale).max(4.0) as u16;
        }
    }

    widths
}

// ---- Rendering ----

/// Render the results table into the given area.
/// This replaces the inline rendering in mod.rs with a proper table widget.
pub fn render_results_table(f: &mut Frame, area: Rect, app: &AppState) {
    let is_focused = app.focus == super::app::FocusPanel::Results;
    let theme = &app.theme;
    let border_style = if is_focused {
        theme.focus_border_style
    } else {
        theme.border_style
    };

    match &app.query_state {
        super::app::QueryState::Idle => {
            render_placeholder(
                f,
                area,
                theme,
                border_style,
                "Run a query to see results here.",
            );
        }
        super::app::QueryState::Running => {
            let dots = ".".repeat(((app.frame_count / 10) % 4) as usize);
            let msg = format!("Executing{}", dots);
            render_placeholder_styled(
                f,
                area,
                theme,
                border_style,
                &msg,
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            );
        }
        super::app::QueryState::Complete => {
            if let Some(ref result) = app.last_result {
                let profile_info = if app.profile_mode {
                    app.last_pipeline_profile.as_ref()
                } else {
                    None
                };
                render_data_table(
                    f,
                    area,
                    result,
                    &app.results_state,
                    app.last_exec_us,
                    app.last_query_metrics.as_ref(),
                    profile_info,
                    theme,
                    border_style,
                    is_focused,
                );
            } else {
                render_placeholder(f, area, theme, border_style, "(no results)");
            }
        }
        super::app::QueryState::Error(msg) => {
            let err_style = Style::default().fg(ratatui::style::Color::Rgb(255, 80, 80));
            render_placeholder_styled(
                f,
                area,
                theme,
                border_style,
                &format!("Error: {}", msg),
                err_style,
            );
        }
    }
}

/// Render a simple placeholder message in the results panel.
fn render_placeholder(f: &mut Frame, area: Rect, theme: &Theme, border_style: Style, msg: &str) {
    render_placeholder_styled(
        f,
        area,
        theme,
        border_style,
        msg,
        Style::default().fg(theme.muted),
    );
}

/// Render a placeholder message with a custom style.
fn render_placeholder_styled(
    f: &mut Frame,
    area: Rect,
    theme: &Theme,
    border_style: Style,
    msg: &str,
    style: Style,
) {
    let block = Block::default()
        .title(Span::styled(
            " Results ",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(border_style);

    let paragraph =
        ratatui::widgets::Paragraph::new(Line::from(Span::styled(format!("  {}", msg), style)))
            .block(block);
    f.render_widget(paragraph, area);
}

/// Render the data table with headers, formatted rows, pagination, and performance line.
#[allow(clippy::too_many_arguments)]
fn render_data_table(
    f: &mut Frame,
    area: Rect,
    result: &QueryResult,
    results_state: &ResultsState,
    exec_time_us: Option<u64>,
    query_metrics: Option<&QueryMetrics>,
    pipeline_profile: Option<&PipelineProfile>,
    theme: &Theme,
    border_style: Style,
    is_focused: bool,
) {
    // Block with title
    let title = format!(" Results ({} rows) ", format_i64(result.row_count as i64));
    let block = Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(border_style);

    let inner_area = block.inner(area);
    f.render_widget(block, area);

    if inner_area.height < 3 || inner_area.width < 10 {
        return; // Too small to render
    }

    // Reserve space: 1 line for perf info, 1 line for pagination status,
    // + 10 lines for profile timeline when active (9 stages + 1 total)
    let profile_lines: u16 = if pipeline_profile.is_some() { 10 } else { 0 };
    let table_height = inner_area.height.saturating_sub(2 + profile_lines) as usize;

    // Compute column widths
    let col_widths = compute_column_widths(result, inner_area.width);
    let constraints: Vec<Constraint> = col_widths.iter().map(|&w| Constraint::Length(w)).collect();

    // Detect which columns are numeric (by scanning first rows)
    let ncols = result.columns.len();
    let is_col_numeric = detect_numeric_columns(result);

    // Header row
    let header_cells: Vec<Cell> = result
        .columns
        .iter()
        .enumerate()
        .map(|(i, col)| {
            let style = Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD);
            if i < is_col_numeric.len() && is_col_numeric[i] {
                // Right-align numeric headers by padding
                Cell::from(col.clone()).style(style)
            } else {
                Cell::from(col.clone()).style(style)
            }
        })
        .collect();
    let header = Row::new(header_cells).height(1).bottom_margin(0);

    // Compute visible row range (page)
    let page_start = results_state.page_start();
    let page_end = results_state.page_end(result.rows.len());
    let visible_rows = &result.rows[page_start..page_end];

    // Build data rows
    let rows: Vec<Row> = visible_rows
        .iter()
        .enumerate()
        .map(|(local_idx, row)| {
            let global_idx = page_start + local_idx;
            let cells: Vec<Cell> = (0..ncols)
                .map(|col_idx| {
                    let val = row.get(col_idx).map(|s| s.as_str()).unwrap_or("");
                    if is_null(val) {
                        Cell::from("NULL".to_string()).style(
                            Style::default()
                                .fg(ratatui::style::Color::Rgb(100, 100, 110))
                                .add_modifier(Modifier::DIM),
                        )
                    } else if col_idx < is_col_numeric.len() && is_col_numeric[col_idx] {
                        Cell::from(format_number(val)).style(Style::default().fg(theme.text))
                    } else {
                        Cell::from(val.to_string()).style(Style::default().fg(theme.text))
                    }
                })
                .collect();

            let row_style = if is_focused && global_idx == results_state.selected_row {
                theme.selection
            } else {
                Style::default()
            };
            Row::new(cells).style(row_style).height(1)
        })
        .collect();

    // Render table (limited to table_height rows)
    let visible_count = rows.len().min(table_height);
    let table_rows: Vec<Row> = rows.into_iter().take(visible_count).collect();

    let table = Table::new(table_rows, &constraints)
        .header(header)
        .column_spacing(1);

    // Render into upper portion of inner area
    let table_area = Rect {
        x: inner_area.x,
        y: inner_area.y,
        width: inner_area.width,
        height: (visible_count as u16 + 1).min(inner_area.height.saturating_sub(2)), // +1 for header
    };
    f.render_widget(table, table_area);

    // Performance line (with CPU comparison when metrics available)
    let footer_start_y = inner_area.y + inner_area.height.saturating_sub(2 + profile_lines);
    let perf_y = footer_start_y;
    if perf_y < inner_area.y + inner_area.height {
        let perf_text = if let Some(m) = query_metrics {
            // Full metrics-based line: "142M rows | 8.4 GB | 2.3ms (cold) | GPU 94% | ~312x vs CPU"
            let utilization = m.scan_throughput_gbps as f32 / 100.0; // rough M4 estimate
            metrics::build_metrics_performance_line(m, utilization.clamp(0.0, 1.0))
        } else {
            build_performance_line(result, exec_time_us)
        };
        let perf_line = Line::from(Span::styled(perf_text, Style::default().fg(theme.muted)));
        let perf_area = Rect {
            x: inner_area.x,
            y: perf_y,
            width: inner_area.width,
            height: 1,
        };
        f.render_widget(ratatui::widgets::Paragraph::new(perf_line), perf_area);
    }

    // Profile timeline (when profile mode is on and profile data available)
    if let Some(profile) = pipeline_profile {
        let timeline_text = metrics::render_profile_timeline(profile);
        let timeline_lines: Vec<Line> = timeline_text
            .lines()
            .map(|l| {
                Line::from(Span::styled(
                    l.to_string(),
                    Style::default().fg(theme.muted),
                ))
            })
            .collect();
        let timeline_y = perf_y + 1;
        let timeline_height = profile_lines.min(inner_area.y + inner_area.height - timeline_y);
        if timeline_height > 0 {
            let timeline_area = Rect {
                x: inner_area.x + 1,
                y: timeline_y,
                width: inner_area.width.saturating_sub(1),
                height: timeline_height,
            };
            let timeline_widget = ratatui::widgets::Paragraph::new(timeline_lines);
            f.render_widget(timeline_widget, timeline_area);
        }
    }

    // Pagination status line
    let page_y = inner_area.y + inner_area.height.saturating_sub(1);
    if page_y < inner_area.y + inner_area.height && result.rows.len() > results_state.page_size {
        let total_pages = results_state.total_pages(result.rows.len());
        let page_text = format!(
            " Page {}/{} | Space: next | Shift+Space: prev | Arrow keys: scroll ",
            results_state.current_page + 1,
            total_pages,
        );
        let page_line = Line::from(Span::styled(
            page_text,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::DIM),
        ));
        let page_area = Rect {
            x: inner_area.x,
            y: page_y,
            width: inner_area.width,
            height: 1,
        };
        f.render_widget(ratatui::widgets::Paragraph::new(page_line), page_area);
    }
}

/// Detect which columns appear to be numeric by sampling rows.
fn detect_numeric_columns(result: &QueryResult) -> Vec<bool> {
    let ncols = result.columns.len();
    let mut numeric_count = vec![0usize; ncols];
    let mut total_count = vec![0usize; ncols];

    let sample_size = result.rows.len().min(100);
    for row in result.rows.iter().take(sample_size) {
        for (i, val) in row.iter().enumerate() {
            if i < ncols && !is_null(val) {
                total_count[i] += 1;
                if is_numeric(val) {
                    numeric_count[i] += 1;
                }
            }
        }
    }

    // Column is numeric if >80% of non-null values are numeric
    (0..ncols)
        .map(|i| {
            if total_count[i] == 0 {
                false
            } else {
                numeric_count[i] * 100 / total_count[i] > 80
            }
        })
        .collect()
}

/// Build the performance summary line.
fn build_performance_line(result: &QueryResult, exec_time_us: Option<u64>) -> String {
    let row_count = format_i64(result.row_count as i64);
    match exec_time_us {
        Some(us) if us >= 1_000_000 => {
            format!(" {} rows | {:.2}s", row_count, us as f64 / 1_000_000.0,)
        }
        Some(us) if us >= 1_000 => {
            format!(" {} rows | {:.1}ms", row_count, us as f64 / 1_000.0,)
        }
        Some(us) => {
            format!(" {} rows | {}us", row_count, format_i64(us as i64))
        }
        None => {
            format!(" {} rows", row_count)
        }
    }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_i64_zero() {
        assert_eq!(format_i64(0), "0");
    }

    #[test]
    fn test_format_i64_small() {
        assert_eq!(format_i64(42), "42");
        assert_eq!(format_i64(999), "999");
    }

    #[test]
    fn test_format_i64_thousands() {
        assert_eq!(format_i64(1_000), "1,000");
        assert_eq!(format_i64(1_234_567), "1,234,567");
        assert_eq!(format_i64(1_000_000_000), "1,000,000,000");
    }

    #[test]
    fn test_format_i64_negative() {
        assert_eq!(format_i64(-1_234), "-1,234");
        assert_eq!(format_i64(-999), "-999");
        assert_eq!(format_i64(-1_000_000), "-1,000,000");
    }

    #[test]
    fn test_format_f64_basic() {
        assert_eq!(format_f64(0.0), "0.00");
        assert_eq!(format_f64(3.14), "3.14");
    }

    #[test]
    fn test_format_f64_thousands() {
        assert_eq!(format_f64(1234.56), "1,234.56");
        assert_eq!(format_f64(1_000_000.5), "1,000,000.50");
    }

    #[test]
    fn test_format_f64_negative() {
        assert_eq!(format_f64(-1234.56), "-1,234.56");
    }

    #[test]
    fn test_format_f64_special() {
        assert_eq!(format_f64(f64::NAN), "NaN");
        assert_eq!(format_f64(f64::INFINITY), "Inf");
        assert_eq!(format_f64(f64::NEG_INFINITY), "-Inf");
    }

    #[test]
    fn test_format_number_integer() {
        assert_eq!(format_number("1234567"), "1,234,567");
    }

    #[test]
    fn test_format_number_float() {
        assert_eq!(format_number("1234.5"), "1,234.50");
    }

    #[test]
    fn test_format_number_non_numeric() {
        assert_eq!(format_number("hello"), "hello");
        assert_eq!(format_number("NULL"), "NULL");
    }

    #[test]
    fn test_is_numeric_true() {
        assert!(is_numeric("42"));
        assert!(is_numeric("-100"));
        assert!(is_numeric("3.14"));
        assert!(is_numeric("0"));
    }

    #[test]
    fn test_is_numeric_false() {
        assert!(!is_numeric(""));
        assert!(!is_numeric("null"));
        assert!(!is_numeric("hello"));
        assert!(!is_numeric("NULL"));
    }

    #[test]
    fn test_is_null_values() {
        assert!(is_null("NULL"));
        assert!(is_null("null"));
        assert!(is_null("Null"));
        assert!(is_null(""));
        assert!(!is_null("0"));
        assert!(!is_null("hello"));
    }

    #[test]
    fn test_results_state_pagination() {
        let mut state = ResultsState::new();
        state.page_size = 10;

        assert_eq!(state.total_pages(100), 10);
        assert_eq!(state.total_pages(0), 0);
        assert_eq!(state.total_pages(5), 1);
        assert_eq!(state.total_pages(15), 2);

        assert_eq!(state.page_start(), 0);
        assert_eq!(state.page_end(100), 10);
    }

    #[test]
    fn test_results_state_next_prev_page() {
        let mut state = ResultsState::new();
        state.page_size = 10;

        state.next_page(100);
        assert_eq!(state.current_page, 1);
        assert_eq!(state.page_start(), 10);
        assert_eq!(state.page_end(100), 20);

        state.prev_page(100);
        assert_eq!(state.current_page, 0);
        assert_eq!(state.page_start(), 0);
    }

    #[test]
    fn test_results_state_no_overflow() {
        let mut state = ResultsState::new();
        state.page_size = 10;

        // Can't go below page 0
        state.prev_page(100);
        assert_eq!(state.current_page, 0);

        // Can't go past last page
        state.current_page = 9;
        state.next_page(100);
        assert_eq!(state.current_page, 9);
    }

    #[test]
    fn test_scroll_down_up() {
        let mut state = ResultsState::new();
        state.page_size = 5;

        state.scroll_down(20);
        assert_eq!(state.selected_row, 1);

        state.scroll_down(20);
        assert_eq!(state.selected_row, 2);

        state.scroll_up(20);
        assert_eq!(state.selected_row, 1);
    }

    #[test]
    fn test_scroll_auto_page_advance() {
        let mut state = ResultsState::new();
        state.page_size = 3;

        // Scroll to end of page 0
        state.scroll_down(10); // row 1
        state.scroll_down(10); // row 2
        assert_eq!(state.current_page, 0);

        // Cross page boundary
        state.scroll_down(10); // row 3 -> page 1
        assert_eq!(state.selected_row, 3);
        assert_eq!(state.current_page, 1);
    }

    #[test]
    fn test_scroll_no_overflow() {
        let mut state = ResultsState::new();
        state.page_size = 5;

        state.scroll_up(10);
        assert_eq!(state.selected_row, 0);

        state.selected_row = 9;
        state.scroll_down(10);
        assert_eq!(state.selected_row, 9); // at last row
    }

    #[test]
    fn test_reset() {
        let mut state = ResultsState::new();
        state.current_page = 5;
        state.selected_row = 42;
        state.scroll_offset = 3;

        state.reset();
        assert_eq!(state.current_page, 0);
        assert_eq!(state.selected_row, 0);
        assert_eq!(state.scroll_offset, 0);
    }

    #[test]
    fn test_compute_column_widths_basic() {
        let result = QueryResult {
            columns: vec!["id".into(), "name".into(), "amount".into()],
            rows: vec![
                vec!["1".into(), "Alice".into(), "1000".into()],
                vec!["2".into(), "Bob".into(), "2000000".into()],
            ],
            row_count: 2,
        };

        let widths = compute_column_widths(&result, 120);
        assert_eq!(widths.len(), 3);
        // Each width should be at least 4 (minimum)
        for w in &widths {
            assert!(*w >= 4);
        }
    }

    #[test]
    fn test_compute_column_widths_empty() {
        let result = QueryResult {
            columns: vec![],
            rows: vec![],
            row_count: 0,
        };

        let widths = compute_column_widths(&result, 120);
        assert!(widths.is_empty());
    }

    #[test]
    fn test_insert_thousands_sep() {
        assert_eq!(insert_thousands_sep("1"), "1");
        assert_eq!(insert_thousands_sep("12"), "12");
        assert_eq!(insert_thousands_sep("123"), "123");
        assert_eq!(insert_thousands_sep("1234"), "1,234");
        assert_eq!(insert_thousands_sep("1234567"), "1,234,567");
        assert_eq!(insert_thousands_sep("1234567890"), "1,234,567,890");
    }

    #[test]
    fn test_build_performance_line_no_timing() {
        let result = QueryResult {
            columns: vec!["count".into()],
            rows: vec![vec!["42".into()]],
            row_count: 42,
        };
        let line = build_performance_line(&result, None);
        assert!(line.contains("42 rows"));
    }

    #[test]
    fn test_build_performance_line_us() {
        let result = QueryResult {
            columns: vec!["count".into()],
            rows: vec![vec!["1000".into()]],
            row_count: 1000,
        };
        let line = build_performance_line(&result, Some(500));
        assert!(line.contains("1,000 rows"));
        assert!(line.contains("500us"));
    }

    #[test]
    fn test_build_performance_line_ms() {
        let result = QueryResult {
            columns: vec!["count".into()],
            rows: vec![vec!["1000".into()]],
            row_count: 1000,
        };
        let line = build_performance_line(&result, Some(5_500));
        assert!(line.contains("1,000 rows"));
        assert!(line.contains("5.5ms"));
    }

    #[test]
    fn test_build_performance_line_seconds() {
        let result = QueryResult {
            columns: vec!["count".into()],
            rows: vec![vec!["1000000".into()]],
            row_count: 1_000_000,
        };
        let line = build_performance_line(&result, Some(2_500_000));
        assert!(line.contains("1,000,000 rows"));
        assert!(line.contains("2.50s"));
    }

    #[test]
    fn test_detect_numeric_columns() {
        let result = QueryResult {
            columns: vec!["id".into(), "name".into(), "amount".into()],
            rows: vec![
                vec!["1".into(), "Alice".into(), "1000.50".into()],
                vec!["2".into(), "Bob".into(), "2000.75".into()],
                vec!["3".into(), "Charlie".into(), "NULL".into()],
            ],
            row_count: 3,
        };
        let numeric = detect_numeric_columns(&result);
        assert_eq!(numeric, vec![true, false, true]);
    }

    #[test]
    fn test_format_i64_min_max() {
        let min_str = format_i64(i64::MIN);
        assert!(min_str.starts_with('-'));
        assert!(min_str.contains(','));

        let max_str = format_i64(i64::MAX);
        assert!(max_str.contains(','));
    }

    // ---- CPU comparison and metrics-aware performance line tests ----

    #[test]
    fn test_build_performance_line_with_metrics() {
        use crate::gpu::metrics::{build_metrics_performance_line, QueryMetrics};

        let m = QueryMetrics {
            gpu_time_ms: 2.3,
            memory_used_bytes: 1_000_000,
            scan_throughput_gbps: 94.0,
            rows_processed: 142_000_000,
            bytes_scanned: 8_400_000_000,
            is_warm: false,
        };
        let line = build_metrics_performance_line(&m, 0.94);
        // Verify all components present
        assert!(line.contains("142M rows"), "line was: {}", line);
        assert!(line.contains("8.40 GB"), "line was: {}", line);
        assert!(line.contains("2.3 ms"), "line was: {}", line);
        assert!(line.contains("(cold)"), "line was: {}", line);
        assert!(line.contains("GPU 94%"), "line was: {}", line);
        assert!(line.contains("vs CPU"), "line was: {}", line);
    }

    #[test]
    fn test_build_performance_line_with_metrics_warm() {
        use crate::gpu::metrics::{build_metrics_performance_line, QueryMetrics};

        let m = QueryMetrics {
            gpu_time_ms: 0.5,
            memory_used_bytes: 0,
            scan_throughput_gbps: 20.0,
            rows_processed: 1_000_000,
            bytes_scanned: 12_000_000,
            is_warm: true,
        };
        let line = build_metrics_performance_line(&m, 0.20);
        assert!(line.contains("1.00M rows"), "line was: {}", line);
        assert!(line.contains("12.0 MB"), "line was: {}", line);
        assert!(line.contains("(warm)"), "line was: {}", line);
        assert!(line.contains("GPU 20%"), "line was: {}", line);
    }

    #[test]
    fn test_build_performance_line_fallback_no_metrics() {
        // When no QueryMetrics available, old format still works
        let result = QueryResult {
            columns: vec!["count".into()],
            rows: vec![vec!["500000".into()]],
            row_count: 500_000,
        };
        let line = build_performance_line(&result, Some(3_500));
        assert!(line.contains("500,000 rows"), "line was: {}", line);
        assert!(line.contains("3.5ms"), "line was: {}", line);
    }

    #[test]
    fn test_format_row_count_integration() {
        use crate::gpu::metrics::format_row_count;
        // Test the row count formatting used in performance line
        assert_eq!(format_row_count(142_000_000), "142M");
        assert_eq!(format_row_count(1_500_000), "1.50M");
        assert_eq!(format_row_count(500), "500");
    }

    #[test]
    fn test_format_data_bytes_integration() {
        use crate::gpu::metrics::format_data_bytes;
        assert_eq!(format_data_bytes(8_400_000_000), "8.40 GB");
        assert_eq!(format_data_bytes(12_000_000), "12.0 MB");
    }

    #[test]
    fn test_format_speedup_integration() {
        use crate::gpu::metrics::format_speedup;
        assert_eq!(format_speedup(312.0), "~312x vs CPU");
        assert_eq!(format_speedup(5.3), "~5.3x vs CPU");
        assert_eq!(format_speedup(0.8), "~0.8x vs CPU");
    }
}
