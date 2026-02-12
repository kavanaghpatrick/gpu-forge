//! Keyboard shortcut handling for gpu-search.
//!
//! Maps keyboard input to semantic actions. The mapping logic is separated
//! from egui context access so it can be tested without a running UI.

use eframe::egui;

/// Actions that can be triggered by keyboard shortcuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Move selection up (Up arrow or Ctrl+P).
    NavigateUp,
    /// Move selection down (Down arrow or Ctrl+N).
    NavigateDown,
    /// Open the selected file (Enter).
    OpenFile,
    /// Open the selected file in $EDITOR (Cmd+Enter).
    OpenInEditor,
    /// Dismiss / close the search panel (Escape).
    Dismiss,
    /// Clear the search input (Cmd+Backspace).
    ClearSearch,
    /// Copy the selected result's path to clipboard (Cmd+C).
    CopyPath,
    /// Cycle through filter modes (Tab).
    CycleFilter,
    /// No action matched.
    None,
}

/// Snapshot of modifier key state for mapping logic.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModifierState {
    /// Command key (Cmd on macOS, Ctrl on other platforms).
    pub command: bool,
    /// Control key.
    pub ctrl: bool,
}

impl From<egui::Modifiers> for ModifierState {
    fn from(m: egui::Modifiers) -> Self {
        Self {
            command: m.command,
            ctrl: m.ctrl,
        }
    }
}

/// Key press event for testable mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyPress {
    ArrowUp,
    ArrowDown,
    Enter,
    Escape,
    Backspace,
    Tab,
    P,
    N,
    C,
}

/// Map a key press + modifier state to a KeyAction.
///
/// This is the core mapping logic, separated from egui for testability.
pub fn map_key_action(key: KeyPress, mods: ModifierState) -> KeyAction {
    match (key, mods.command, mods.ctrl) {
        // Cmd+Enter -> OpenInEditor (must check before plain Enter)
        (KeyPress::Enter, true, _) => KeyAction::OpenInEditor,
        // Plain Enter -> OpenFile
        (KeyPress::Enter, false, _) => KeyAction::OpenFile,
        // Cmd+Backspace -> ClearSearch
        (KeyPress::Backspace, true, _) => KeyAction::ClearSearch,
        // Cmd+C -> CopyPath
        (KeyPress::C, true, _) => KeyAction::CopyPath,
        // Ctrl+P -> NavigateUp (Emacs-style)
        (KeyPress::P, _, true) => KeyAction::NavigateUp,
        // Ctrl+N -> NavigateDown (Emacs-style)
        (KeyPress::N, _, true) => KeyAction::NavigateDown,
        // Up arrow -> NavigateUp
        (KeyPress::ArrowUp, false, false) => KeyAction::NavigateUp,
        // Down arrow -> NavigateDown
        (KeyPress::ArrowDown, false, false) => KeyAction::NavigateDown,
        // Escape -> Dismiss
        (KeyPress::Escape, _, _) => KeyAction::Dismiss,
        // Tab -> CycleFilter
        (KeyPress::Tab, false, false) => KeyAction::CycleFilter,
        // No match
        _ => KeyAction::None,
    }
}

/// Process keyboard input from an egui context and return the action to perform.
///
/// Uses `input_mut` with `consume_key` so that matched shortcuts are not
/// forwarded to other widgets (e.g., the text input).
pub fn process_input(ctx: &egui::Context) -> KeyAction {
    ctx.input_mut(|input| {
        let mods = input.modifiers;

        // Check modifier+key combos first (order matters: Cmd+Enter before Enter)
        if mods.command && input.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter) {
            return KeyAction::OpenInEditor;
        }
        if mods.command && input.consume_key(egui::Modifiers::COMMAND, egui::Key::Backspace) {
            return KeyAction::ClearSearch;
        }
        if mods.command && input.consume_key(egui::Modifiers::COMMAND, egui::Key::C) {
            return KeyAction::CopyPath;
        }

        // Ctrl+P / Ctrl+N (Emacs navigation)
        if input.consume_key(egui::Modifiers::CTRL, egui::Key::P) {
            return KeyAction::NavigateUp;
        }
        if input.consume_key(egui::Modifiers::CTRL, egui::Key::N) {
            return KeyAction::NavigateDown;
        }

        // Plain keys (no modifiers)
        if input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp) {
            return KeyAction::NavigateUp;
        }
        if input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown) {
            return KeyAction::NavigateDown;
        }
        if input.consume_key(egui::Modifiers::NONE, egui::Key::Enter) {
            return KeyAction::OpenFile;
        }
        if input.consume_key(egui::Modifiers::NONE, egui::Key::Escape) {
            return KeyAction::Dismiss;
        }
        if input.consume_key(egui::Modifiers::NONE, egui::Key::Tab) {
            return KeyAction::CycleFilter;
        }

        KeyAction::None
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_mods() -> ModifierState {
        ModifierState::default()
    }

    fn cmd() -> ModifierState {
        ModifierState {
            command: true,
            ctrl: false,
        }
    }

    fn ctrl() -> ModifierState {
        ModifierState {
            command: false,
            ctrl: true,
        }
    }

    #[test]
    fn test_keybind_actions() {
        // Navigation
        assert_eq!(
            map_key_action(KeyPress::ArrowUp, no_mods()),
            KeyAction::NavigateUp,
        );
        assert_eq!(
            map_key_action(KeyPress::ArrowDown, no_mods()),
            KeyAction::NavigateDown,
        );

        // Emacs-style navigation
        assert_eq!(
            map_key_action(KeyPress::P, ctrl()),
            KeyAction::NavigateUp,
        );
        assert_eq!(
            map_key_action(KeyPress::N, ctrl()),
            KeyAction::NavigateDown,
        );

        // Open file
        assert_eq!(
            map_key_action(KeyPress::Enter, no_mods()),
            KeyAction::OpenFile,
        );

        // Open in editor (Cmd+Enter)
        assert_eq!(
            map_key_action(KeyPress::Enter, cmd()),
            KeyAction::OpenInEditor,
        );

        // Dismiss
        assert_eq!(
            map_key_action(KeyPress::Escape, no_mods()),
            KeyAction::Dismiss,
        );

        // Clear search (Cmd+Backspace)
        assert_eq!(
            map_key_action(KeyPress::Backspace, cmd()),
            KeyAction::ClearSearch,
        );

        // Copy path (Cmd+C)
        assert_eq!(
            map_key_action(KeyPress::C, cmd()),
            KeyAction::CopyPath,
        );

        // Cycle filter (Tab)
        assert_eq!(
            map_key_action(KeyPress::Tab, no_mods()),
            KeyAction::CycleFilter,
        );

        // No action for unbound keys
        assert_eq!(
            map_key_action(KeyPress::Tab, cmd()),
            KeyAction::None,
        );
    }

    #[test]
    fn test_key_action_enum_variants() {
        // Verify all expected variants exist and are distinct
        let actions = [
            KeyAction::NavigateUp,
            KeyAction::NavigateDown,
            KeyAction::OpenFile,
            KeyAction::OpenInEditor,
            KeyAction::Dismiss,
            KeyAction::ClearSearch,
            KeyAction::CopyPath,
            KeyAction::CycleFilter,
            KeyAction::None,
        ];
        // All 9 variants should be unique
        for (i, a) in actions.iter().enumerate() {
            for (j, b) in actions.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Variants at index {} and {} should differ", i, j);
                }
            }
        }
    }

    #[test]
    fn test_modifier_priority() {
        // Cmd+Enter should be OpenInEditor, not OpenFile
        assert_eq!(
            map_key_action(KeyPress::Enter, cmd()),
            KeyAction::OpenInEditor,
        );
        assert_ne!(
            map_key_action(KeyPress::Enter, cmd()),
            KeyAction::OpenFile,
        );

        // Plain Enter should be OpenFile, not OpenInEditor
        assert_eq!(
            map_key_action(KeyPress::Enter, no_mods()),
            KeyAction::OpenFile,
        );
    }

    #[test]
    fn test_key_action_clone_copy_debug() {
        let action = KeyAction::NavigateUp;
        let cloned = action.clone();
        let copied = action;
        assert_eq!(action, cloned);
        assert_eq!(action, copied);
        let debug_str = format!("{:?}", action);
        assert!(debug_str.contains("NavigateUp"));
    }

    #[test]
    fn test_modifier_state_from_egui() {
        let egui_mods = egui::Modifiers {
            command: true,
            ctrl: false,
            ..Default::default()
        };
        let state = ModifierState::from(egui_mods);
        assert!(state.command);
        assert!(!state.ctrl);
    }
}
