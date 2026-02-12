//! .gitignore filtering for search results.
//!
//! Uses the `ignore` crate (ripgrep's) to parse .gitignore files at each
//! directory level, supporting globs, negation patterns, and directory markers.
//! Applied before GPU search to exclude irrelevant files from the pipeline.

use std::path::Path;

use ignore::gitignore::{Gitignore, GitignoreBuilder};

// ============================================================================
// GitignoreFilter
// ============================================================================

/// Filter that applies .gitignore rules to file paths.
///
/// Wraps the `ignore` crate's `Gitignore` matcher, which supports:
/// - Standard glob patterns (`*.log`, `build/`)
/// - Negation patterns (`!important.log`)
/// - Directory-only markers (`logs/`)
/// - Nested .gitignore files at each directory level
/// - Comment lines starting with `#`
///
/// # Example
/// ```ignore
/// let filter = GitignoreFilter::from_directory(Path::new("/my/project")).unwrap();
/// assert!(filter.is_ignored(Path::new("/my/project/target/debug/binary")));
/// assert!(!filter.is_ignored(Path::new("/my/project/src/main.rs")));
/// ```
pub struct GitignoreFilter {
    /// The compiled gitignore matcher.
    matcher: Gitignore,
    /// Root directory this filter was built from.
    root: std::path::PathBuf,
}

impl GitignoreFilter {
    /// Build a GitignoreFilter from all .gitignore files in the directory tree.
    ///
    /// Walks from `root` upward and downward, parsing .gitignore files at each
    /// level. The root .gitignore is parsed first, then subdirectory-level ones
    /// override as appropriate (matching git's behavior).
    ///
    /// Returns `Ok(GitignoreFilter)` even if no .gitignore files are found
    /// (in which case nothing is ignored).
    pub fn from_directory(root: &Path) -> Result<Self, ignore::Error> {
        let mut builder = GitignoreBuilder::new(root);

        // Add the root .gitignore if present
        let root_gitignore = root.join(".gitignore");
        if root_gitignore.exists() {
            builder.add(&root_gitignore);
        }

        // Walk subdirectories to find nested .gitignore files
        Self::collect_nested_gitignores(root, root, &mut builder);

        let matcher = builder.build()?;

        Ok(Self {
            matcher,
            root: root.to_path_buf(),
        })
    }

    /// Build a GitignoreFilter from a single .gitignore content string.
    ///
    /// Useful for testing or when .gitignore content is already in memory.
    pub fn from_str(root: &Path, content: &str) -> Result<Self, ignore::Error> {
        let mut builder = GitignoreBuilder::new(root);

        for line in content.lines() {
            builder.add_line(None, line)?;
        }

        let matcher = builder.build()?;

        Ok(Self {
            matcher,
            root: root.to_path_buf(),
        })
    }

    /// Check if a path should be ignored based on .gitignore rules.
    ///
    /// The path can be absolute or relative to the root directory.
    /// For directory paths, set `is_dir` appropriately since some .gitignore
    /// patterns only match directories (trailing `/`).
    pub fn is_ignored(&self, path: &Path) -> bool {
        let is_dir = path.is_dir();
        self.is_ignored_with_dir_hint(path, is_dir)
    }

    /// Check if a path should be ignored, with an explicit directory hint.
    ///
    /// Use this when you already know whether the path is a directory,
    /// avoiding an extra filesystem stat.
    pub fn is_ignored_with_dir_hint(&self, path: &Path, is_dir: bool) -> bool {
        self.matcher
            .matched_path_or_any_parents(path, is_dir)
            .is_ignore()
    }

    /// Filter a list of paths, returning only those NOT ignored.
    pub fn filter_paths<'a>(&self, paths: &'a [&Path]) -> Vec<&'a Path> {
        paths
            .iter()
            .filter(|p| !self.is_ignored(p))
            .copied()
            .collect()
    }

    /// Get the root directory this filter was built from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Recursively collect .gitignore files from subdirectories.
    fn collect_nested_gitignores(
        dir: &Path,
        _root: &Path,
        builder: &mut GitignoreBuilder,
    ) {
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                // Skip hidden directories and common non-project dirs
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.') || name == "node_modules" || name == "target" {
                        continue;
                    }
                }

                // Check for .gitignore in this subdirectory
                let sub_gitignore = path.join(".gitignore");
                if sub_gitignore.exists() {
                    builder.add(&sub_gitignore);
                }

                // Recurse
                Self::collect_nested_gitignores(&path, _root, builder);
            }
        }
    }
}

impl std::fmt::Debug for GitignoreFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GitignoreFilter")
            .field("root", &self.root)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Create a test directory with .gitignore and various files.
    fn make_gitignore_test_dir() -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // Initialize git repo (ignore crate may need this for some features)
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .expect("git init failed");

        // Create .gitignore
        fs::write(
            dir.path().join(".gitignore"),
            "# Build artifacts\n\
             target/\n\
             *.o\n\
             *.log\n\
             \n\
             # But keep important.log\n\
             !important.log\n\
             \n\
             # Temp files\n\
             *.tmp\n\
             build/\n",
        )
        .unwrap();

        // Create source files (should NOT be ignored)
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("lib.rs"), "pub mod test;").unwrap();
        fs::write(dir.path().join("README.md"), "# Project").unwrap();

        // Create files that SHOULD be ignored
        fs::write(dir.path().join("debug.log"), "debug output").unwrap();
        fs::write(dir.path().join("app.tmp"), "temp data").unwrap();
        fs::write(dir.path().join("module.o"), "object file").unwrap();

        // Create negated file (should NOT be ignored despite *.log)
        fs::write(dir.path().join("important.log"), "keep this").unwrap();

        // Create ignored directory
        let build_dir = dir.path().join("build");
        fs::create_dir(&build_dir).unwrap();
        fs::write(build_dir.join("output.bin"), "binary").unwrap();

        let target_dir = dir.path().join("target");
        fs::create_dir(&target_dir).unwrap();
        fs::write(target_dir.join("release"), "executable").unwrap();

        // Create src directory with nested .gitignore
        let src_dir = dir.path().join("src");
        fs::create_dir(&src_dir).unwrap();
        fs::write(src_dir.join("code.rs"), "fn code() {}").unwrap();
        fs::write(
            src_dir.join(".gitignore"),
            "# Additional ignores for src/\n*.generated\n",
        )
        .unwrap();
        fs::write(src_dir.join("auto.generated"), "generated code").unwrap();

        dir
    }

    #[test]
    fn test_gitignore_basic_glob() {
        let dir = make_gitignore_test_dir();
        let filter = GitignoreFilter::from_directory(dir.path()).unwrap();

        // *.log should be ignored
        assert!(
            filter.is_ignored(&dir.path().join("debug.log")),
            "*.log files should be ignored"
        );

        // *.tmp should be ignored
        assert!(
            filter.is_ignored(&dir.path().join("app.tmp")),
            "*.tmp files should be ignored"
        );

        // *.o should be ignored
        assert!(
            filter.is_ignored(&dir.path().join("module.o")),
            "*.o files should be ignored"
        );
    }

    #[test]
    fn test_gitignore_source_files_not_ignored() {
        let dir = make_gitignore_test_dir();
        let filter = GitignoreFilter::from_directory(dir.path()).unwrap();

        // Source files should NOT be ignored
        assert!(
            !filter.is_ignored(&dir.path().join("main.rs")),
            "main.rs should not be ignored"
        );
        assert!(
            !filter.is_ignored(&dir.path().join("lib.rs")),
            "lib.rs should not be ignored"
        );
        assert!(
            !filter.is_ignored(&dir.path().join("README.md")),
            "README.md should not be ignored"
        );
    }

    #[test]
    fn test_gitignore_negation() {
        let dir = make_gitignore_test_dir();
        let filter = GitignoreFilter::from_directory(dir.path()).unwrap();

        // important.log should NOT be ignored (negation pattern)
        assert!(
            !filter.is_ignored(&dir.path().join("important.log")),
            "important.log should not be ignored (negation pattern !important.log)"
        );
    }

    #[test]
    fn test_gitignore_directory_patterns() {
        let dir = make_gitignore_test_dir();
        let filter = GitignoreFilter::from_directory(dir.path()).unwrap();

        // build/ directory contents should be ignored
        assert!(
            filter.is_ignored_with_dir_hint(&dir.path().join("build"), true),
            "build/ directory should be ignored"
        );

        // target/ directory contents should be ignored
        assert!(
            filter.is_ignored_with_dir_hint(&dir.path().join("target"), true),
            "target/ directory should be ignored"
        );

        // Files inside ignored directories should be ignored
        assert!(
            filter.is_ignored(&dir.path().join("build").join("output.bin")),
            "Files inside build/ should be ignored"
        );
    }

    #[test]
    fn test_gitignore_nested_gitignore() {
        let dir = make_gitignore_test_dir();
        let filter = GitignoreFilter::from_directory(dir.path()).unwrap();

        // src/.gitignore adds *.generated
        assert!(
            filter.is_ignored(&dir.path().join("src").join("auto.generated")),
            "*.generated in src/ should be ignored by nested .gitignore"
        );

        // But src/code.rs should NOT be ignored
        assert!(
            !filter.is_ignored(&dir.path().join("src").join("code.rs")),
            "src/code.rs should not be ignored"
        );
    }

    #[test]
    fn test_gitignore_from_str() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        let filter = GitignoreFilter::from_str(
            dir.path(),
            "*.log\n*.tmp\n!keep.log\ntarget/\n",
        )
        .unwrap();

        assert!(filter.is_ignored(&dir.path().join("debug.log")));
        assert!(filter.is_ignored(&dir.path().join("app.tmp")));
        assert!(!filter.is_ignored(&dir.path().join("keep.log")));
        assert!(!filter.is_ignored(&dir.path().join("main.rs")));
    }

    #[test]
    fn test_gitignore_filter_paths() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // Create actual files so is_dir check works
        fs::write(dir.path().join("main.rs"), "code").unwrap();
        fs::write(dir.path().join("debug.log"), "log").unwrap();
        fs::write(dir.path().join("notes.txt"), "notes").unwrap();

        let filter =
            GitignoreFilter::from_str(dir.path(), "*.log\n").unwrap();

        let p1 = dir.path().join("main.rs");
        let p2 = dir.path().join("debug.log");
        let p3 = dir.path().join("notes.txt");
        let paths: Vec<&Path> = vec![&p1, &p2, &p3];

        let filtered = filter.filter_paths(&paths);

        assert_eq!(filtered.len(), 2, "Should have 2 non-ignored paths");
        assert!(
            filtered
                .iter()
                .any(|p| p.to_string_lossy().contains("main.rs")),
            "main.rs should pass filter"
        );
        assert!(
            filtered
                .iter()
                .any(|p| p.to_string_lossy().contains("notes.txt")),
            "notes.txt should pass filter"
        );
        assert!(
            !filtered
                .iter()
                .any(|p| p.to_string_lossy().contains("debug.log")),
            "debug.log should be filtered out"
        );
    }

    #[test]
    fn test_gitignore_no_gitignore_file() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // No .gitignore file -- nothing should be ignored
        fs::write(dir.path().join("main.rs"), "code").unwrap();
        fs::write(dir.path().join("debug.log"), "log").unwrap();

        let filter = GitignoreFilter::from_directory(dir.path()).unwrap();

        assert!(
            !filter.is_ignored(&dir.path().join("main.rs")),
            "Nothing should be ignored without .gitignore"
        );
        assert!(
            !filter.is_ignored(&dir.path().join("debug.log")),
            "Nothing should be ignored without .gitignore"
        );
    }

    #[test]
    fn test_gitignore_comment_and_empty_lines() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        let content = "# This is a comment\n\
                       \n\
                       *.log\n\
                       \n\
                       # Another comment\n\
                       *.tmp\n";

        let filter = GitignoreFilter::from_str(dir.path(), content).unwrap();

        assert!(filter.is_ignored(&dir.path().join("test.log")));
        assert!(filter.is_ignored(&dir.path().join("test.tmp")));
        assert!(!filter.is_ignored(&dir.path().join("test.rs")));
    }

    #[test]
    fn test_gitignore_root_accessor() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let filter = GitignoreFilter::from_directory(dir.path()).unwrap();
        assert_eq!(filter.root(), dir.path());
    }
}
