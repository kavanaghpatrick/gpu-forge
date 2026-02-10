//! Event loop with crossterm for terminal input handling.
//!
//! Polls for keyboard/mouse events at 60fps tick rate.
//! Separates input handling from rendering.

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
                // 'q' without modifiers = quit
                if key.code == KeyCode::Char('q') && key.modifiers.is_empty() {
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
    match key.code {
        // Tab cycles focus
        KeyCode::Tab => {
            app.cycle_focus();
            true
        }
        // Escape returns to editor
        KeyCode::Esc => {
            app.focus = super::app::FocusPanel::Editor;
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
        let key = make_key(KeyCode::Char('x'), KeyModifiers::empty());
        assert!(!handle_key(&key, &mut app));
    }
}
