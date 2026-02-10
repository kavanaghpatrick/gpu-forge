//! TUI dashboard module for gpu-query.
//!
//! Provides an interactive terminal dashboard with gradient-colored rendering,
//! query editor, results display, and GPU metrics. Built on ratatui + crossterm.

pub mod app;
pub mod autocomplete;
pub mod catalog;
pub mod dashboard;
pub mod editor;
pub mod event;
pub mod gradient;
pub mod results;
pub mod themes;

use app::AppState;
use event::{poll_event, handle_key, AppEvent};
use gradient::gradient_text;

use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};
use std::io;
use std::path::PathBuf;
use std::time::Duration;

/// Run the interactive TUI dashboard.
/// This takes over the terminal until the user quits (q/Ctrl+C).
pub fn run_dashboard(data_dir: PathBuf, theme_name: &str) -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Initialize app state
    let mut app = AppState::new(data_dir.clone(), theme_name);

    // Scan for tables in data directory and populate catalog tree
    if let Ok(catalog_entries) = crate::io::catalog::scan_directory(&data_dir) {
        app.tables = catalog_entries.iter().map(|e| e.name.clone()).collect();

        let tree_entries: Vec<catalog::CatalogEntry> = catalog_entries
            .iter()
            .map(catalog::CatalogEntry::from_table_entry)
            .collect();

        let dir_name = data_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("data")
            .to_string();

        app.catalog_state.load(tree_entries, dir_name);
    }
    app.status_message = format!(
        "gpu-query dashboard | {} tables loaded | Press q to quit",
        app.tables.len()
    );

    // Main event loop (~60fps)
    let tick_rate = Duration::from_millis(app.tick_rate_ms);

    loop {
        // Render
        terminal.draw(|f| render_frame(f, &mut app))?;

        // Handle events
        match poll_event(tick_rate)? {
            AppEvent::Quit => break,
            AppEvent::Key(key) => {
                handle_key(&key, &mut app);
            }
            AppEvent::Resize(_, _) => {
                // ratatui handles resize automatically
            }
            AppEvent::Tick => {
                app.tick();
            }
        }

        if !app.running {
            break;
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

/// Render a single frame of the dashboard.
fn render_frame(f: &mut Frame, app: &mut AppState) {
    let size = f.area();

    // Main layout: title bar (3) + content (fill) + status bar (3)
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title bar
            Constraint::Min(5),   // content area
            Constraint::Length(3), // status bar
        ])
        .split(size);

    // Render title bar with gradient
    render_title_bar(f, main_chunks[0], app);

    // Render content area (placeholder panels)
    render_content(f, main_chunks[1], app);

    // Render status bar
    render_status_bar(f, main_chunks[2], app);
}

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

/// Render the main content area with placeholder panels.
fn render_content(f: &mut Frame, area: Rect, app: &mut AppState) {
    // Three-column layout: catalog (20%) | editor+results (80%)
    let h_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(20), Constraint::Percentage(80)])
        .split(area);

    // Left panel: catalog
    render_catalog_panel(f, h_chunks[0], app);

    // Right panel: editor (top) + results (bottom)
    let v_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(h_chunks[1]);

    render_editor_panel(f, v_chunks[0], app);
    render_results_panel(f, v_chunks[1], app);
}

/// Render the catalog panel with tree view.
fn render_catalog_panel(f: &mut Frame, area: Rect, app: &mut AppState) {
    let is_focused = app.focus == app::FocusPanel::Catalog;

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

/// Render the query editor panel (placeholder).
fn render_editor_panel(f: &mut Frame, area: Rect, app: &AppState) {
    let is_focused = app.focus == app::FocusPanel::Editor;
    let border_style = if is_focused {
        app.theme.focus_border_style
    } else {
        app.theme.border_style
    };

    let editor_text = if app.query_text.is_empty() {
        "SELECT * FROM ... WHERE ... LIMIT 10"
    } else {
        &app.query_text
    };

    let style = if app.query_text.is_empty() {
        Style::default().fg(app.theme.muted)
    } else {
        Style::default().fg(app.theme.text)
    };

    let block = Block::default()
        .title(Span::styled(
            " Query (Ctrl+Enter to execute) ",
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(border_style);

    let paragraph = Paragraph::new(Line::from(Span::styled(editor_text, style))).block(block);
    f.render_widget(paragraph, area);
}

/// Render the results panel using the scrollable table widget.
fn render_results_panel(f: &mut Frame, area: Rect, app: &AppState) {
    results::render_results_table(f, area, app);
}

/// Render the status bar at the bottom.
fn render_status_bar(f: &mut Frame, area: Rect, app: &AppState) {
    // Build gradient-colored status items
    let status_text = &app.status_message;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(app.theme.border_style);

    let style = Style::default().fg(app.theme.muted);
    let paragraph = Paragraph::new(Line::from(vec![
        Span::styled(format!(" {} ", status_text), style),
        Span::styled(
            format!("| Theme: {} ", app.theme.name),
            Style::default().fg(app.theme.accent),
        ),
    ]))
    .block(block);

    f.render_widget(paragraph, area);
}
