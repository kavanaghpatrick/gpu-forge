//! Open-in-editor actions for gpu-search.
//!
//! Detects the user's preferred editor from environment variables ($VISUAL,
//! $EDITOR) or PATH, then generates the correct command to open a file at a
//! specific line number.

use std::path::Path;
use std::process::Command;

/// Known editor types with line-number argument conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorKind {
    /// VS Code: `code --goto file:line`
    VsCode,
    /// Neovim: `nvim +line file`
    Neovim,
    /// Vim: `vim +line file`
    Vim,
    /// Sublime Text: `subl file:line`
    Sublime,
    /// Emacs: `emacs +line file`
    Emacs,
    /// macOS default: `open file`
    Default,
}

/// Detect the user's preferred editor.
///
/// Checks, in order:
/// 1. `$VISUAL`
/// 2. `$EDITOR`
/// 3. Common editor binaries on `$PATH`
/// 4. Falls back to `Default` (macOS `open`)
pub fn detect_editor() -> EditorKind {
    // Check $VISUAL first, then $EDITOR
    for var in &["VISUAL", "EDITOR"] {
        if let Ok(val) = std::env::var(var) {
            if let Some(kind) = classify_editor(&val) {
                return kind;
            }
        }
    }

    // Probe PATH for common editors
    let candidates = [
        ("code", EditorKind::VsCode),
        ("nvim", EditorKind::Neovim),
        ("vim", EditorKind::Vim),
        ("subl", EditorKind::Sublime),
        ("emacs", EditorKind::Emacs),
    ];

    for (bin, kind) in &candidates {
        if which_exists(bin) {
            return *kind;
        }
    }

    EditorKind::Default
}

/// Classify an editor string (from env var) into an EditorKind.
///
/// Handles full paths like `/usr/local/bin/code` and bare names like `nvim`.
fn classify_editor(editor: &str) -> Option<EditorKind> {
    let basename = Path::new(editor)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(editor);

    match basename {
        "code" | "code-insiders" => Some(EditorKind::VsCode),
        "nvim" | "neovim" => Some(EditorKind::Neovim),
        "vim" | "vi" => Some(EditorKind::Vim),
        "subl" | "sublime_text" => Some(EditorKind::Sublime),
        "emacs" | "emacsclient" => Some(EditorKind::Emacs),
        _ => None,
    }
}

/// Check if a binary exists on PATH using `which`.
fn which_exists(bin: &str) -> bool {
    Command::new("which")
        .arg(bin)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Build the command (program + args) to open a file at a specific line.
///
/// Returns `(program, args)` so callers can inspect without spawning.
/// This is the primary testable interface.
pub fn editor_command(editor: EditorKind, path: &Path, line: usize) -> (String, Vec<String>) {
    let path_str = path.to_string_lossy().to_string();

    match editor {
        EditorKind::VsCode => (
            "code".to_string(),
            vec!["--goto".to_string(), format!("{}:{}", path_str, line)],
        ),
        EditorKind::Neovim => (
            "nvim".to_string(),
            vec![format!("+{}", line), path_str],
        ),
        EditorKind::Vim => (
            "vim".to_string(),
            vec![format!("+{}", line), path_str],
        ),
        EditorKind::Sublime => (
            "subl".to_string(),
            vec![format!("{}:{}", path_str, line)],
        ),
        EditorKind::Emacs => (
            "emacs".to_string(),
            vec![format!("+{}", line), path_str],
        ),
        EditorKind::Default => (
            "open".to_string(),
            vec![path_str],
        ),
    }
}

/// Open a file using macOS `open` command (Enter key action).
pub fn open_file(path: &Path) -> std::io::Result<()> {
    Command::new("open")
        .arg(path)
        .spawn()
        .map(|_| ())
}

/// Open a file in the detected editor at a specific line (Cmd+Enter action).
pub fn open_in_editor(path: &Path, line: usize) -> std::io::Result<()> {
    let editor = detect_editor();
    let (program, args) = editor_command(editor, path, line);
    Command::new(&program)
        .args(&args)
        .spawn()
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_editor_command_vscode() {
        let path = PathBuf::from("/src/main.rs");
        let (prog, args) = editor_command(EditorKind::VsCode, &path, 42);
        assert_eq!(prog, "code");
        assert_eq!(args, vec!["--goto", "/src/main.rs:42"]);
    }

    #[test]
    fn test_editor_command_neovim() {
        let path = PathBuf::from("/src/main.rs");
        let (prog, args) = editor_command(EditorKind::Neovim, &path, 10);
        assert_eq!(prog, "nvim");
        assert_eq!(args, vec!["+10", "/src/main.rs"]);
    }

    #[test]
    fn test_editor_command_vim() {
        let path = PathBuf::from("/src/lib.rs");
        let (prog, args) = editor_command(EditorKind::Vim, &path, 99);
        assert_eq!(prog, "vim");
        assert_eq!(args, vec!["+99", "/src/lib.rs"]);
    }

    #[test]
    fn test_editor_command_sublime() {
        let path = PathBuf::from("/tmp/test.py");
        let (prog, args) = editor_command(EditorKind::Sublime, &path, 7);
        assert_eq!(prog, "subl");
        assert_eq!(args, vec!["/tmp/test.py:7"]);
    }

    #[test]
    fn test_editor_command_emacs() {
        let path = PathBuf::from("/home/user/file.el");
        let (prog, args) = editor_command(EditorKind::Emacs, &path, 1);
        assert_eq!(prog, "emacs");
        assert_eq!(args, vec!["+1", "/home/user/file.el"]);
    }

    #[test]
    fn test_editor_command_default() {
        let path = PathBuf::from("/tmp/readme.txt");
        let (prog, args) = editor_command(EditorKind::Default, &path, 50);
        assert_eq!(prog, "open");
        // Default ignores line number -- just opens the file
        assert_eq!(args, vec!["/tmp/readme.txt"]);
    }

    #[test]
    fn test_editor_command_line_one() {
        let path = PathBuf::from("/a.rs");
        let (_, args) = editor_command(EditorKind::VsCode, &path, 1);
        assert_eq!(args, vec!["--goto", "/a.rs:1"]);
    }

    #[test]
    fn test_classify_editor_bare_names() {
        assert_eq!(classify_editor("code"), Some(EditorKind::VsCode));
        assert_eq!(classify_editor("code-insiders"), Some(EditorKind::VsCode));
        assert_eq!(classify_editor("nvim"), Some(EditorKind::Neovim));
        assert_eq!(classify_editor("vim"), Some(EditorKind::Vim));
        assert_eq!(classify_editor("vi"), Some(EditorKind::Vim));
        assert_eq!(classify_editor("subl"), Some(EditorKind::Sublime));
        assert_eq!(classify_editor("emacs"), Some(EditorKind::Emacs));
        assert_eq!(classify_editor("emacsclient"), Some(EditorKind::Emacs));
        assert_eq!(classify_editor("unknown-editor"), None);
    }

    #[test]
    fn test_classify_editor_full_paths() {
        assert_eq!(
            classify_editor("/usr/local/bin/code"),
            Some(EditorKind::VsCode)
        );
        assert_eq!(
            classify_editor("/opt/homebrew/bin/nvim"),
            Some(EditorKind::Neovim)
        );
        assert_eq!(
            classify_editor("/usr/bin/vim"),
            Some(EditorKind::Vim)
        );
        assert_eq!(
            classify_editor("/Applications/Sublime Text.app/Contents/SharedSupport/bin/subl"),
            Some(EditorKind::Sublime)
        );
    }

    #[test]
    fn test_editor_kind_variants_distinct() {
        let kinds = [
            EditorKind::VsCode,
            EditorKind::Neovim,
            EditorKind::Vim,
            EditorKind::Sublime,
            EditorKind::Emacs,
            EditorKind::Default,
        ];
        for (i, a) in kinds.iter().enumerate() {
            for (j, b) in kinds.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_editor_command_path_with_spaces() {
        let path = PathBuf::from("/my project/src/main.rs");
        let (prog, args) = editor_command(EditorKind::VsCode, &path, 5);
        assert_eq!(prog, "code");
        assert_eq!(args, vec!["--goto", "/my project/src/main.rs:5"]);
    }

    #[test]
    fn test_detect_editor_returns_valid_kind() {
        // detect_editor should always return a valid EditorKind
        let editor = detect_editor();
        // Just verify it doesn't panic and returns something
        let path = PathBuf::from("/tmp/test.rs");
        let (prog, _) = editor_command(editor, &path, 1);
        assert!(!prog.is_empty());
    }
}
