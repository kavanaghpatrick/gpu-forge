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

    // Strategy (a): substitute $HOME with ~ (try first for best UX)
    // This must come before search_root stripping, because when search_root
    // is "/" every path matches but we still want ~/... display.
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

    // Strategy (b): strip search_root prefix for relative display
    if let Ok(relative) = parent.strip_prefix(search_root) {
        let rel_str = relative.to_string_lossy();
        if rel_str.is_empty() {
            return (String::new(), filename);
        }
        let dir_display = format!("{}/", rel_str);
        let dir_display = middle_truncate(&dir_display, DIR_MAX_LEN);
        return (dir_display, filename);
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

    // ── Existing tests ──────────────────────────────────────────────

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

    // ── U-PATH-1 through U-PATH-12: comprehensive path abbreviation tests ──

    /// U-PATH-1: Relative to root with single directory component
    #[test]
    fn u_path_1_relative_to_root_single_dir() {
        let root = Path::new("/workspace");
        let path = Path::new("/workspace/lib/utils.rs");
        let (dir, name) = abbreviate_path(path, root);
        assert_eq!(dir, "lib/");
        assert_eq!(name, "utils.rs");
    }

    /// U-PATH-2: Root-level file returns empty dir, just filename
    #[test]
    fn u_path_2_root_level_file_returns_empty_dir() {
        let root = Path::new("/a/b/c");
        let path = Path::new("/a/b/c/README.md");
        let (dir, name) = abbreviate_path(path, root);
        assert_eq!(dir, "");
        assert_eq!(name, "README.md");
    }

    /// U-PATH-3: Home substitution -- path inside $HOME but outside search root
    /// gets ~ prefix. Uses actual $HOME from environment since OnceLock is cached.
    #[test]
    fn u_path_3_home_substitution() {
        if let Some(home) = home_dir() {
            // Use a search root that does NOT contain the target path
            let search_root = Path::new("/nonexistent/root");
            let path = home.join("Documents/notes.txt");
            let (dir, name) = abbreviate_path(&path, search_root);
            assert!(
                dir.starts_with("~/"),
                "expected dir starting with ~/, got: {}",
                dir
            );
            assert!(dir.contains("Documents"));
            assert_eq!(name, "notes.txt");
        }
    }

    /// U-PATH-4: Deep nested path (5+ levels) under search root
    #[test]
    fn u_path_4_deep_path() {
        let root = Path::new("/project");
        let path = Path::new("/project/src/core/engine/render/pipeline/shader.glsl");
        let (dir, name) = abbreviate_path(path, root);
        assert_eq!(dir, "src/core/engine/render/pipeline/");
        assert_eq!(name, "shader.glsl");
    }

    /// U-PATH-5: Path outside home and outside search root falls back to absolute
    #[test]
    fn u_path_5_outside_home_and_root() {
        let root = Path::new("/workspace/project");
        let path = Path::new("/var/log/system.log");
        let (dir, name) = abbreviate_path(path, root);
        // Not under root or home, should show full absolute parent
        assert!(
            dir.starts_with("/var/log"),
            "expected absolute path prefix, got: {}",
            dir
        );
        assert_eq!(name, "system.log");
    }

    /// U-PATH-6: Middle truncation triggers when directory exceeds DIR_MAX_LEN (50 chars)
    #[test]
    fn u_path_6_middle_truncation_long_dir() {
        let root = Path::new("/r");
        // Build a path with a directory portion exceeding 50 characters
        let long_dir = "a".repeat(20);
        let deep = format!(
            "/r/{}/{}/{}/file.rs",
            long_dir, long_dir, long_dir
        );
        let path = PathBuf::from(&deep);
        let (dir, name) = abbreviate_path(&path, root);
        // The relative dir is "aaaa.../aaaa.../aaaa.../" which is 62 chars
        // Should be middle-truncated to <= 50 chars
        assert!(
            dir.len() <= DIR_MAX_LEN,
            "dir len {} exceeds max {}: {}",
            dir.len(),
            DIR_MAX_LEN,
            dir
        );
        assert!(dir.contains("..."), "expected truncation marker, got: {}", dir);
        assert_eq!(name, "file.rs");
    }

    /// U-PATH-7: No parent -- bare filename returns empty dir
    #[test]
    fn u_path_7_no_parent_bare_filename() {
        let root = Path::new("/some/root");
        let path = Path::new("bare_file.txt");
        let (dir, name) = abbreviate_path(path, root);
        assert_eq!(dir, "");
        assert_eq!(name, "bare_file.txt");
    }

    /// U-PATH-8: Unicode filename and directory components
    #[test]
    fn u_path_8_unicode_path() {
        let root = Path::new("/proyecto");
        let path = Path::new("/proyecto/src/datos/archivo_\u{00e9}special.rs");
        let (dir, name) = abbreviate_path(path, root);
        assert_eq!(dir, "src/datos/");
        assert_eq!(name, "archivo_\u{00e9}special.rs");
    }

    /// U-PATH-9: Empty filename (directory-like path with trailing slash)
    #[test]
    fn u_path_9_empty_filename() {
        let root = Path::new("/root");
        // Path::new("/root/subdir/") treats it as dir with no filename
        let path = Path::new("/root/subdir/");
        let (dir, name) = abbreviate_path(path, root);
        // For a path ending in `/`, file_name() returns the last component "subdir"
        // and parent() returns "/root" -- so it behaves like a file named "subdir"
        assert_eq!(name, "subdir");
        assert_eq!(dir, "");
    }

    /// U-PATH-10: HOME unset fallback -- abbreviate_root returns absolute path
    /// Since OnceLock caches HOME on first access (and HOME IS set in our env),
    /// we test the outside-home fallback path instead.
    #[test]
    fn u_path_10_home_unset_fallback() {
        // When path is outside home, abbreviate_root should return the full absolute path
        let path = Path::new("/tmp/cache/data");
        let result = abbreviate_root(path);
        assert_eq!(result, "/tmp/cache/data");
        // Also verify it doesn't contain ~ substitution
        assert!(!result.contains('~'), "should not have ~ for path outside home");
    }

    /// U-PATH-11: abbreviate_root at home directory returns just "~"
    #[test]
    fn u_path_11_abbreviate_root_at_home() {
        if let Some(home) = home_dir() {
            let result = abbreviate_root(home);
            assert_eq!(result, "~");
        }
    }

    /// U-PATH-12: middle_truncate preserves boundary chars and ellipsis format
    #[test]
    fn u_path_12_middle_truncate_boundary() {
        // Exactly at max_len: no truncation
        let s = "a".repeat(50);
        assert_eq!(middle_truncate(&s, 50), s);

        // One char over: truncation kicks in
        let s51 = "a".repeat(51);
        let result = middle_truncate(&s51, 50);
        assert!(result.len() <= 50);
        assert!(result.contains("..."));
        // Verify prefix + "..." + suffix = 50
        assert_eq!(result.len(), 50);

        // Max_len = 5 (minimum non-bypass): should still truncate
        let result5 = middle_truncate("abcdefghij", 5);
        assert!(result5.len() <= 5);
        assert!(result5.contains("..."));
        // available = 2 (5-3), prefix = 0 (2*2/5=0), suffix = 2
        // Result: "...ij"
        assert_eq!(result5, "...ij");
    }
}
