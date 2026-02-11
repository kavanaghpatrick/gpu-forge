//! Event loop with crossterm for terminal input handling.
//!
//! Polls for keyboard/mouse events at 60fps tick rate.
//! Separates input handling from rendering.
//!
//! Key bindings:
//! - Tab: cycle focus panel
//! - Ctrl+1/2/3: direct panel focus (catalog/editor/results)
//! - Ctrl+Enter: execute query from editor
//! - Esc: return to editor
//! - q/Ctrl+C/Ctrl+Q: quit
//! - Editor panel: full text editing (arrows, insert, backspace, etc.)

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use std::time::Duration;

use crate::cli::commands::{
    is_dot_command, parse_dot_command, handle_dot_command, append_to_history,
    DotCommandContext, DotCommandResult,
};

/// Events produced by the event loop.
#[derive(Debug)]
pub enum AppEvent {
    /// Terminal key press.
    Key(KeyEvent),
    /// Tick (render frame).
    Tick,
    /// Quit requested.
    Quit,
    /// Window resize.
    Resize(u16, u16),
}

/// Poll for the next event with the given timeout.
/// Returns AppEvent::Tick if no event is available within the timeout.
pub fn poll_event(tick_rate: Duration) -> std::io::Result<AppEvent> {
    if event::poll(tick_rate)? {
        match event::read()? {
            Event::Key(key) => {
                // Ctrl+C or Ctrl+Q = quit
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('q'))
                {
                    return Ok(AppEvent::Quit);
                }
                Ok(AppEvent::Key(key))
            }
            Event::Resize(w, h) => Ok(AppEvent::Resize(w, h)),
            _ => Ok(AppEvent::Tick),
        }
    } else {
        Ok(AppEvent::Tick)
    }
}

/// Handle a key event against the application state.
/// Returns true if the event was consumed.
pub fn handle_key(key: &KeyEvent, app: &mut super::app::AppState) -> bool {
    use super::app::FocusPanel;

    let has_ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

    // Global bindings (work regardless of focused panel)

    // Ctrl+Enter: execute query or dot command
    if has_ctrl && key.code == KeyCode::Enter {
        let text = app.editor_state.text();
        if is_dot_command(&text) {
            execute_dot_command(app);
        } else {
            // Record query in persistent history
            append_to_history(&mut app.history, &text);
            let _ = super::ui::execute_editor_query(app);
        }
        return true;
    }

    // Ctrl+1/2/3: direct panel focus
    if has_ctrl {
        match key.code {
            KeyCode::Char('1') => {
                app.focus = FocusPanel::Catalog;
                return true;
            }
            KeyCode::Char('2') => {
                app.focus = FocusPanel::Editor;
                return true;
            }
            KeyCode::Char('3') => {
                app.focus = FocusPanel::Results;
                return true;
            }
            _ => {}
        }
    }

    // Tab cycles focus
    if key.code == KeyCode::Tab {
        app.cycle_focus();
        return true;
    }

    // Escape returns to editor
    if key.code == KeyCode::Esc {
        app.focus = FocusPanel::Editor;
        return true;
    }

    // Panel-specific key handling
    match app.focus {
        FocusPanel::Editor => handle_editor_key(key, app),
        FocusPanel::Results => handle_results_key(key, app),
        FocusPanel::Catalog => handle_catalog_key(key, app),
    }
}

/// Execute a dot command from the editor text and update app state.
fn execute_dot_command(app: &mut super::app::AppState) {
    let text = app.editor_state.text();

    let cmd = match parse_dot_command(&text) {
        Some(c) => c,
        None => {
            app.status_message = format!("Unknown command: {}", text.trim());
            return;
        }
    };

    // Build context from app state
    let tables: Vec<(String, String, Vec<(String, String)>)> = app
        .catalog_state
        .entries
        .iter()
        .map(|entry| {
            let cols: Vec<(String, String)> = entry
                .columns
                .iter()
                .map(|c| (c.name.clone(), c.type_name.clone()))
                .collect();
            let fmt = match entry.format {
                crate::io::format_detect::FileFormat::Csv => "CSV",
                crate::io::format_detect::FileFormat::Parquet => "Parquet",
                crate::io::format_detect::FileFormat::Json => "JSON",
                crate::io::format_detect::FileFormat::Unknown => "?",
            };
            (entry.name.clone(), fmt.to_string(), cols)
        })
        .collect();

    let last_result_text = app.last_result.as_ref().map(|r| {
        crate::cli::format_result(r, &crate::cli::args::OutputFormat::Table)
    });
    let last_result_ref = last_result_text.as_deref();

    let comparison_text = app.last_query_metrics.as_ref().map(|m| {
        let est = crate::gpu::metrics::CpuEstimate::from_metrics(m.bytes_scanned, m.gpu_time_ms);
        format!(
            "GPU {:.1}ms vs CPU ~{:.1}ms (~{:.1}x speedup)",
            m.gpu_time_ms, est.cpu_estimate_ms, est.speedup_vs_cpu
        )
    });

    let format = crate::cli::args::OutputFormat::Table; // default format for TUI
    let ctx = DotCommandContext {
        tables: &tables,
        history: &app.history,
        last_result_text: last_result_ref,
        current_format: &format,
        profile_on: app.profile_mode,
        timer_on: true,
        gpu_device_name: "Apple Silicon GPU".into(),
        gpu_memory_bytes: app.gpu_metrics.peak_memory_bytes,
        last_comparison: comparison_text,
    };

    let result = handle_dot_command(&cmd, &ctx);

    match result {
        DotCommandResult::Output(text) => {
            // Display as a "query result" with single column
            let result = crate::gpu::executor::QueryResult {
                columns: vec!["output".into()],
                rows: text.lines().map(|l| vec![l.to_string()]).collect(),
                row_count: text.lines().count(),
            };
            app.set_result(result);
            app.status_message = "Command executed.".into();
        }
        DotCommandResult::StateChange(msg) => {
            // Parse state changes to update AppState fields
            if msg.contains("Profile mode: ON") {
                app.profile_mode = true;
            } else if msg.contains("Profile mode: OFF") {
                app.profile_mode = false;
            }
            app.status_message = msg;
        }
        DotCommandResult::ClearScreen => {
            app.last_result = None;
            app.results_state.reset();
            app.status_message = "Cleared.".into();
        }
        DotCommandResult::Quit => {
            app.running = false;
        }
        DotCommandResult::Error(msg) => {
            app.set_error(msg);
        }
    }
}

/// Handle key events when the editor panel is focused.
fn handle_editor_key(key: &KeyEvent, app: &mut super::app::AppState) -> bool {
    match key.code {
        // Character input
        KeyCode::Char(ch) => {
            app.editor_state.insert_char(ch);
            // Keep query_text in sync for backward compat
            app.query_text = app.editor_state.text();
            true
        }
        // Newline
        KeyCode::Enter => {
            app.editor_state.insert_newline();
            app.query_text = app.editor_state.text();
            true
        }
        // Backspace
        KeyCode::Backspace => {
            app.editor_state.backspace();
            app.query_text = app.editor_state.text();
            true
        }
        // Delete
        KeyCode::Delete => {
            app.editor_state.delete_char();
            app.query_text = app.editor_state.text();
            true
        }
        // Cursor movement
        KeyCode::Left => {
            app.editor_state.move_left();
            true
        }
        KeyCode::Right => {
            app.editor_state.move_right();
            true
        }
        KeyCode::Up => {
            app.editor_state.move_up();
            true
        }
        KeyCode::Down => {
            app.editor_state.move_down();
            true
        }
        KeyCode::Home => {
            app.editor_state.move_home();
            true
        }
        KeyCode::End => {
            app.editor_state.move_end();
            true
        }
        _ => false,
    }
}

/// Handle key events when the results panel is focused.
fn handle_results_key(key: &KeyEvent, app: &mut super::app::AppState) -> bool {
    let total_rows = app.last_result.as_ref().map_or(0, |r| r.rows.len());

    match key.code {
        KeyCode::Down => {
            app.results_state.scroll_down(total_rows);
            true
        }
        KeyCode::Up => {
            app.results_state.scroll_up(total_rows);
            true
        }
        KeyCode::Char(' ') => {
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                app.results_state.prev_page(total_rows);
            } else {
                app.results_state.next_page(total_rows);
            }
            true
        }
        _ => false,
    }
}

/// Handle key events when the catalog panel is focused.
fn handle_catalog_key(key: &KeyEvent, app: &mut super::app::AppState) -> bool {
    match key.code {
        KeyCode::Down | KeyCode::Char('j') => {
            app.catalog_state.move_down();
            true
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.catalog_state.move_up();
            true
        }
        KeyCode::Enter => {
            app.catalog_state.handle_enter();
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
    use std::path::PathBuf;

    fn make_key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        }
    }

    #[test]
    fn test_handle_tab_cycles_focus() {
        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        assert_eq!(app.focus, super::super::app::FocusPanel::Editor);

        let tab = make_key(KeyCode::Tab, KeyModifiers::empty());
        assert!(handle_key(&tab, &mut app));
        assert_eq!(app.focus, super::super::app::FocusPanel::Results);
    }

    #[test]
    fn test_handle_esc_returns_to_editor() {
        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        app.focus = super::super::app::FocusPanel::Results;

        let esc = make_key(KeyCode::Esc, KeyModifiers::empty());
        assert!(handle_key(&esc, &mut app));
        assert_eq!(app.focus, super::super::app::FocusPanel::Editor);
    }

    #[test]
    fn test_unhandled_key_returns_false() {
        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        // Focus on results, send a non-handled key
        app.focus = super::super::app::FocusPanel::Results;
        let key = make_key(KeyCode::Char('x'), KeyModifiers::empty());
        assert!(!handle_key(&key, &mut app));
    }

    #[test]
    fn test_ctrl_1_focuses_catalog() {
        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        assert_eq!(app.focus, super::super::app::FocusPanel::Editor);

        let key = make_key(KeyCode::Char('1'), KeyModifiers::CONTROL);
        assert!(handle_key(&key, &mut app));
        assert_eq!(app.focus, super::super::app::FocusPanel::Catalog);
    }

    #[test]
    fn test_ctrl_2_focuses_editor() {
        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        app.focus = super::super::app::FocusPanel::Catalog;

        let key = make_key(KeyCode::Char('2'), KeyModifiers::CONTROL);
        assert!(handle_key(&key, &mut app));
        assert_eq!(app.focus, super::super::app::FocusPanel::Editor);
    }

    #[test]
    fn test_ctrl_3_focuses_results() {
        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        assert_eq!(app.focus, super::super::app::FocusPanel::Editor);

        let key = make_key(KeyCode::Char('3'), KeyModifiers::CONTROL);
        assert!(handle_key(&key, &mut app));
        assert_eq!(app.focus, super::super::app::FocusPanel::Results);
    }

    #[test]
    fn test_editor_char_input() {
        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        assert_eq!(app.focus, super::super::app::FocusPanel::Editor);

        let key_s = make_key(KeyCode::Char('S'), KeyModifiers::empty());
        assert!(handle_key(&key_s, &mut app));
        assert_eq!(app.editor_state.text(), "S");
        assert_eq!(app.query_text, "S");
    }

    #[test]
    fn test_editor_backspace() {
        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        app.editor_state.insert_char('A');
        app.editor_state.insert_char('B');
        app.query_text = app.editor_state.text();

        let bs = make_key(KeyCode::Backspace, KeyModifiers::empty());
        assert!(handle_key(&bs, &mut app));
        assert_eq!(app.editor_state.text(), "A");
        assert_eq!(app.query_text, "A");
    }

    #[test]
    fn test_editor_enter_newline() {
        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        app.editor_state.insert_char('X');

        let enter = make_key(KeyCode::Enter, KeyModifiers::empty());
        assert!(handle_key(&enter, &mut app));
        assert_eq!(app.editor_state.lines.len(), 2);
    }

    #[test]
    fn test_results_scroll() {
        use crate::gpu::executor::QueryResult;

        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        app.focus = super::super::app::FocusPanel::Results;
        app.set_result(QueryResult {
            columns: vec!["a".into()],
            rows: vec![vec!["1".into()], vec!["2".into()], vec!["3".into()]],
            row_count: 3,
        });

        let down = make_key(KeyCode::Down, KeyModifiers::empty());
        assert!(handle_key(&down, &mut app));
        assert_eq!(app.results_state.selected_row, 1);

        let up = make_key(KeyCode::Up, KeyModifiers::empty());
        assert!(handle_key(&up, &mut app));
        assert_eq!(app.results_state.selected_row, 0);
    }

    #[test]
    fn test_catalog_navigation() {
        use crate::tui::catalog::{CatalogEntry, ColumnInfo};
        use crate::io::format_detect::FileFormat;

        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        app.focus = super::super::app::FocusPanel::Catalog;

        let entries = vec![
            CatalogEntry {
                name: "a".into(),
                format: FileFormat::Csv,
                row_count: None,
                columns: vec![ColumnInfo { name: "x".into(), type_name: "INT64".into() }],
            },
            CatalogEntry {
                name: "b".into(),
                format: FileFormat::Csv,
                row_count: None,
                columns: vec![],
            },
        ];
        app.catalog_state.load(entries, "data".into());

        let down = make_key(KeyCode::Down, KeyModifiers::empty());
        assert!(handle_key(&down, &mut app));
        assert_eq!(app.catalog_state.selected, 1);

        let up = make_key(KeyCode::Up, KeyModifiers::empty());
        assert!(handle_key(&up, &mut app));
        assert_eq!(app.catalog_state.selected, 0);

        // Enter toggles expand
        let enter = make_key(KeyCode::Enter, KeyModifiers::empty());
        assert!(handle_key(&enter, &mut app));
        assert!(app.catalog_state.is_expanded(0));
    }

    #[test]
    fn test_q_in_editor_does_not_quit() {
        // When editor is focused, 'q' should type 'q', not quit
        let mut app = super::super::app::AppState::new(PathBuf::from("/tmp"), "thermal");
        assert_eq!(app.focus, super::super::app::FocusPanel::Editor);

        let key_q = make_key(KeyCode::Char('q'), KeyModifiers::empty());
        // handle_key is called (not poll_event which catches quit)
        let consumed = handle_key(&key_q, &mut app);
        assert!(consumed);
        assert_eq!(app.editor_state.text(), "q");
        assert!(app.running); // should NOT quit
    }
}
