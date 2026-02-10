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

    // Ctrl+Enter: execute query
    if has_ctrl && key.code == KeyCode::Enter {
        let _ = super::ui::execute_editor_query(app);
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
