//! Path abbreviation utilities for display in the UI.
//!
//! Converts absolute paths to user-friendly display strings by:
//! - Stripping the search root prefix for relative display
//! - Substituting `$HOME` with `~`
//! - Middle-truncating long directory paths

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Cached home directory from `$HOME` environment variable.
static HOME_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Maximum length for the directory portion before middle-truncation kicks in.
const DIR_MAX_LEN: usize = 50;

/// Get the cached home directory.
fn home_dir() -> Option<&'static Path> {
    HOME_DIR
        .get_or_init(|| std::env::var_os("HOME").map(PathBuf::from))
        .as_deref()
}

/// Abbreviate a file path for display relative to the search root.
///
/// Returns `(dir_display, filename)` where:
/// - `dir_display` is the abbreviated directory portion (with trailing `/` if non-empty)
/// - `filename` is the file name component
///
/// Abbreviation strategy:
/// 1. Try `strip_prefix(search_root)` for paths under the search root
/// 2. Try `$HOME` substitution (replaces home prefix with `~`)
/// 3. Middle-truncate if directory portion exceeds 50 characters
pub fn abbreviate_path(path: &Path, search_root: &Path) -> (String, String) {
    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => return (String::new(), filename),
    };

    // Strategy (a): strip search_root prefix for relative display
    if let Ok(relative) = parent.strip_prefix(search_root) {
        let rel_str = relative.to_string_lossy();
        if rel_str.is_empty() {
            return (String::new(), filename);
        }
        let dir_display = format!("{}/", rel_str);
        let dir_display = middle_truncate(&dir_display, DIR_MAX_LEN);
        return (dir_display, filename);
    }

    // Strategy (b): substitute $HOME with ~
    if let Some(home) = home_dir() {
        if let Ok(relative) = parent.strip_prefix(home) {
            let rel_str = relative.to_string_lossy();
            let dir_display = if rel_str.is_empty() {
                "~/".to_string()
            } else {
                format!("~/{}/", rel_str)
            };
            let dir_display = middle_truncate(&dir_display, DIR_MAX_LEN);
            return (dir_display, filename);
        }
    }

    // Strategy (c): use full path, middle-truncate if needed
    let dir_str = format!("{}/", parent.display());
    let dir_display = middle_truncate(&dir_str, DIR_MAX_LEN);
    (dir_display, filename)
}

/// Abbreviate a root path for status bar display.
///
/// Applies `$HOME` -> `~` substitution.
pub fn abbreviate_root(path: &Path) -> String {
    if let Some(home) = home_dir() {
        if let Ok(relative) = path.strip_prefix(home) {
            let rel_str = relative.to_string_lossy();
            if rel_str.is_empty() {
                return "~".to_string();
            }
            return format!("~/{}", rel_str);
        }
    }
    path.display().to_string()
}

/// Middle-truncate a string if it exceeds `max_len`.
///
/// Keeps the first ~40% and last ~60% of characters with `...` in the middle.
/// Returns the original string unchanged if it fits within `max_len`.
pub fn middle_truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len || max_len < 5 {
        return s.to_string();
    }
    // Reserve 3 chars for "..."
    let available = max_len - 3;
    let prefix_len = available * 2 / 5; // ~40% at start
    let suffix_len = available - prefix_len; // ~60% at end
    format!("{}...{}", &s[..prefix_len], &s[s.len() - suffix_len..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abbreviate_path_relative_to_root() {
        let root = Path::new("/Users/dev/project");
        let path = Path::new("/Users/dev/project/src/main.rs");
        let (dir, name) = abbreviate_path(path, root);
        assert_eq!(dir, "src/");
        assert_eq!(name, "main.rs");
    }

    #[test]
    fn test_abbreviate_path_root_level_file() {
        let root = Path::new("/Users/dev/project");
        let path = Path::new("/Users/dev/project/Cargo.toml");
        let (dir, name) = abbreviate_path(path, root);
        assert_eq!(dir, "");
        assert_eq!(name, "Cargo.toml");
    }

    #[test]
    fn test_abbreviate_path_deep_relative() {
        let root = Path::new("/Users/dev/project");
        let path = Path::new("/Users/dev/project/src/ui/widgets/button.rs");
        let (dir, name) = abbreviate_path(path, root);
        assert_eq!(dir, "src/ui/widgets/");
        assert_eq!(name, "button.rs");
    }

    #[test]
    fn test_abbreviate_path_no_parent() {
        let root = Path::new("/root");
        let path = Path::new("file.txt");
        let (dir, name) = abbreviate_path(path, root);
        assert_eq!(dir, "");
        assert_eq!(name, "file.txt");
    }

    #[test]
    fn test_middle_truncate_short_string() {
        assert_eq!(middle_truncate("hello", 50), "hello");
    }

    #[test]
    fn test_middle_truncate_exact_length() {
        let s = "a".repeat(50);
        assert_eq!(middle_truncate(&s, 50), s);
    }

    #[test]
    fn test_middle_truncate_long_string() {
        let s = "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuv";
        let result = middle_truncate(s, 30);
        assert!(result.len() <= 30, "got len={}: {}", result.len(), result);
        assert!(result.contains("..."));
        // Starts with the beginning of the original (~40% prefix)
        assert!(result.starts_with("abcdefghij"));
        // Ends with the tail of the original (~60% suffix)
        assert!(result.ends_with("stuv"));
    }

    #[test]
    fn test_middle_truncate_very_small_max() {
        // Below 5 chars, return as-is to avoid panic
        assert_eq!(middle_truncate("hello world", 4), "hello world");
    }

    #[test]
    fn test_abbreviate_root_with_home() {
        // This test depends on the actual $HOME env var being set.
        // It tests the ~ substitution path.
        if let Some(home) = home_dir() {
            let test_path = home.join("projects/test");
            let result = abbreviate_root(&test_path);
            assert!(result.starts_with("~/"));
            assert!(result.contains("projects/test"));
        }
    }

    #[test]
    fn test_abbreviate_root_outside_home() {
        let path = Path::new("/opt/system/config");
        let result = abbreviate_root(path);
        assert_eq!(result, "/opt/system/config");
    }
}
