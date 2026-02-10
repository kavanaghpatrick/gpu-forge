//! TUI dashboard module for gpu-query.
//!
//! Provides an interactive terminal dashboard with gradient-colored rendering,
//! query editor, results display, and GPU metrics. Built on ratatui + crossterm.
//!
//! Layout composition is handled by `ui::render_ui` with responsive breakpoints:
//! - >= 120 cols: full three-panel (catalog | editor+results | GPU dashboard)
//! - 80-119 cols: two-panel (editor+results | GPU dashboard)
//! - < 80 cols: minimal REPL (editor + results)

pub mod app;
pub mod autocomplete;
pub mod catalog;
pub mod dashboard;
pub mod editor;
pub mod event;
pub mod gradient;
pub mod results;
pub mod themes;
pub mod ui;

use app::AppState;
use event::{poll_event, handle_key, AppEvent};

use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
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
        "gpu-query dashboard | {} tables loaded | Type SQL + Ctrl+Enter to execute",
        app.tables.len()
    );

    // Main event loop (~60fps)
    let tick_rate = Duration::from_millis(app.tick_rate_ms);

    loop {
        // Render using the responsive layout system
        let metrics_snapshot = app.gpu_metrics.clone();
        terminal.draw(|f| ui::render_ui(f, &mut app, &metrics_snapshot))?;

        // Handle events
        match poll_event(tick_rate)? {
            AppEvent::Quit => break,
            AppEvent::Key(key) => {
                handle_key(&key, &mut app);
            }
            AppEvent::Resize(_, _) => {
                // ratatui handles resize automatically; layout recalculates
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
