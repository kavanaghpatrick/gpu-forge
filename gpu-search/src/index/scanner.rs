//! Parallel filesystem scanner using rayon + ignore crate.
//!
//! Uses `ignore::WalkBuilder` for fast parallel directory traversal that
//! respects .gitignore rules. Builds `Vec<GpuPathEntry>` for GPU-resident
//! path indexing.

use std::path::Path;
use std::sync::Mutex;
use std::time::SystemTime;

use ignore::WalkBuilder;

use crate::gpu::types::{path_flags, GpuPathEntry, GPU_PATH_MAX_LEN};

// ============================================================================
// Scanner Configuration
// ============================================================================

/// Configuration for the filesystem scanner.
#[derive(Debug, Clone)]
pub struct ScannerConfig {
    /// Respect .gitignore rules (default: true)
    pub respect_gitignore: bool,
    /// Follow symbolic links (default: false)
    pub follow_symlinks: bool,
    /// Skip hidden files/directories starting with '.' (default: true)
    pub skip_hidden: bool,
    /// Maximum directory depth (None = unlimited)
    pub max_depth: Option<usize>,
    /// Number of parallel threads (0 = rayon default)
    pub threads: usize,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            respect_gitignore: true,
            follow_symlinks: false,
            skip_hidden: true,
            max_depth: None,
            threads: 0,
        }
    }
}

// ============================================================================
// FilesystemScanner
// ============================================================================

/// Parallel filesystem scanner that builds `Vec<GpuPathEntry>`.
///
/// Uses the `ignore` crate's `WalkBuilder` which provides:
/// - Built-in .gitignore parsing at each directory level
/// - Parallel directory traversal
/// - Symlink handling
/// - Hidden file filtering
pub struct FilesystemScanner {
    config: ScannerConfig,
}

impl FilesystemScanner {
    /// Create a scanner with default configuration.
    pub fn new() -> Self {
        Self {
            config: ScannerConfig::default(),
        }
    }

    /// Create a scanner with custom configuration.
    pub fn with_config(config: ScannerConfig) -> Self {
        Self { config }
    }

    /// Scan a directory tree in parallel, returning `GpuPathEntry` for each file.
    ///
    /// Only files are indexed (directories are skipped in the output).
    /// Paths longer than 224 bytes are silently skipped.
    /// Unreadable entries are silently skipped.
    ///
    /// # Example
    /// ```ignore
    /// let scanner = FilesystemScanner::new();
    /// let entries = scanner.scan(Path::new("/some/project"));
    /// assert!(entries.len() > 0);
    /// ```
    pub fn scan(&self, root: &Path) -> Vec<GpuPathEntry> {
        let mut builder = WalkBuilder::new(root);

        // Configure the walker
        builder
            .git_ignore(self.config.respect_gitignore)
            .git_global(self.config.respect_gitignore)
            .git_exclude(self.config.respect_gitignore)
            .hidden(self.config.skip_hidden)
            .follow_links(self.config.follow_symlinks)
            .parents(self.config.respect_gitignore);

        if let Some(depth) = self.config.max_depth {
            builder.max_depth(Some(depth));
        }

        // Set thread count (ignore crate uses its own thread pool)
        let thread_count = if self.config.threads > 0 {
            self.config.threads
        } else {
            // Use available parallelism or default to 4
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        };
        builder.threads(thread_count);

        // Collect results using ignore's parallel walker
        let entries: Mutex<Vec<GpuPathEntry>> = Mutex::new(Vec::new());

        builder.build_parallel().run(|| {
            Box::new(|result| {
                let dir_entry = match result {
                    Ok(entry) => entry,
                    Err(_) => return ignore::WalkState::Continue,
                };

                // Skip directories -- we only index files
                let file_type = match dir_entry.file_type() {
                    Some(ft) => ft,
                    None => return ignore::WalkState::Continue,
                };

                if file_type.is_dir() {
                    return ignore::WalkState::Continue;
                }

                // Skip symlinks if not following them
                if file_type.is_symlink() && !self.config.follow_symlinks {
                    return ignore::WalkState::Continue;
                }

                let path = dir_entry.path();
                let path_str = path.to_string_lossy();

                // Skip paths that exceed GPU_PATH_MAX_LEN
                if path_str.len() > GPU_PATH_MAX_LEN {
                    return ignore::WalkState::Continue;
                }

                // Get metadata
                let metadata = match dir_entry.metadata() {
                    Ok(m) => m,
                    Err(_) => return ignore::WalkState::Continue,
                };

                let file_size = metadata.len();
                let mtime = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as u32)
                    .unwrap_or(0);

                // Determine flags
                let mut flags: u32 = 0;
                if metadata.file_type().is_symlink() {
                    flags |= path_flags::IS_SYMLINK;
                }
                // Check for hidden (starts with '.' after last '/')
                if let Some(name) = path.file_name() {
                    if name.to_string_lossy().starts_with('.') {
                        flags |= path_flags::IS_HIDDEN;
                    }
                }

                let mut entry = GpuPathEntry::new();
                entry.set_path(path_str.as_bytes());
                entry.flags = flags;
                entry.set_size(file_size);
                entry.mtime = mtime;

                // Thread-safe append
                if let Ok(mut vec) = entries.lock() {
                    vec.push(entry);
                }

                ignore::WalkState::Continue
            })
        });

        // Sort by path for deterministic output
        let mut result = entries.into_inner().unwrap_or_default();
        result.sort_by(|a, b| {
            let a_path = &a.path[..a.path_len as usize];
            let b_path = &b.path[..b.path_len as usize];
            a_path.cmp(b_path)
        });

        result
    }

    /// Get the scanner configuration.
    pub fn config(&self) -> &ScannerConfig {
        &self.config
    }
}

impl Default for FilesystemScanner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper: extract path string from GpuPathEntry
// ============================================================================

/// Extract the path string from a GpuPathEntry.
fn entry_path_str(entry: &GpuPathEntry) -> &str {
    let len = entry.path_len as usize;
    std::str::from_utf8(&entry.path[..len]).unwrap_or("")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Create a test directory with known structure.
    fn make_test_dir() -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // Regular files
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("lib.rs"), "pub mod test;").unwrap();
        fs::write(dir.path().join("README.md"), "# Test").unwrap();

        // Hidden file
        fs::write(dir.path().join(".hidden_file"), "secret").unwrap();

        // Subdirectory with files
        let sub = dir.path().join("src");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("mod.rs"), "mod sub;").unwrap();
        fs::write(sub.join("utils.rs"), "pub fn help() {}").unwrap();

        // Nested subdirectory
        let nested = sub.join("nested");
        fs::create_dir(&nested).unwrap();
        fs::write(nested.join("deep.rs"), "// deep file").unwrap();

        // .git-like directory (should be skipped when skip_hidden=true)
        let git_dir = dir.path().join(".git");
        fs::create_dir(&git_dir).unwrap();
        fs::write(git_dir.join("HEAD"), "ref: refs/heads/main").unwrap();

        dir
    }

    #[test]
    fn test_scanner_basic() {
        let dir = make_test_dir();
        let scanner = FilesystemScanner::new();
        let entries = scanner.scan(dir.path());

        // Should find files but not directories, not hidden, not .git/
        // Expected: main.rs, lib.rs, README.md, src/mod.rs, src/utils.rs, src/nested/deep.rs
        // NOT: .hidden_file, .git/HEAD (hidden)
        assert!(
            entries.len() >= 3,
            "Should have at least 3 entries, got {}",
            entries.len()
        );

        // All entries should have valid paths
        for entry in &entries {
            let path_str = entry_path_str(entry);
            assert!(!path_str.is_empty(), "Path should not be empty");
            let path = PathBuf::from(path_str);
            assert!(path.exists(), "Path should exist: {}", path_str);
        }

        // No directories in output
        for entry in &entries {
            assert_eq!(
                entry.flags & path_flags::IS_DIR,
                0,
                "Should not contain directories"
            );
        }
    }

    #[test]
    fn test_scanner_includes_nested() {
        let dir = make_test_dir();
        let scanner = FilesystemScanner::new();
        let entries = scanner.scan(dir.path());

        // Should find nested file
        let paths: Vec<String> = entries.iter().map(|e| entry_path_str(e).to_string()).collect();
        let has_deep = paths.iter().any(|p| p.contains("deep.rs"));
        assert!(has_deep, "Should find nested deep.rs file. Found: {:?}", paths);
    }

    #[test]
    fn test_scanner_skips_hidden() {
        let dir = make_test_dir();
        let scanner = FilesystemScanner::new();
        let entries = scanner.scan(dir.path());

        let paths: Vec<String> = entries.iter().map(|e| entry_path_str(e).to_string()).collect();

        // Should NOT contain hidden files or .git/ contents
        for path in &paths {
            assert!(
                !path.contains(".hidden_file"),
                "Should skip hidden file, found: {}",
                path
            );
            assert!(
                !path.contains(".git/"),
                "Should skip .git directory, found: {}",
                path
            );
        }
    }

    #[test]
    fn test_scanner_show_hidden() {
        let dir = make_test_dir();
        let scanner = FilesystemScanner::with_config(ScannerConfig {
            skip_hidden: false,
            ..Default::default()
        });
        let entries = scanner.scan(dir.path());

        let paths: Vec<String> = entries.iter().map(|e| entry_path_str(e).to_string()).collect();

        // With skip_hidden=false, should find .hidden_file
        let has_hidden = paths.iter().any(|p| p.contains(".hidden_file"));
        assert!(
            has_hidden,
            "Should find hidden file when skip_hidden=false. Found: {:?}",
            paths
        );
    }

    #[test]
    fn test_scanner_paths_exist() {
        // Scan the actual gpu-search src/ directory
        let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let scanner = FilesystemScanner::new();
        let entries = scanner.scan(&src_dir);

        assert!(
            !entries.is_empty(),
            "Should find files in src/ directory"
        );

        // Every path should exist on disk
        for entry in &entries {
            let path_str = entry_path_str(entry);
            let path = PathBuf::from(path_str);
            assert!(
                path.exists(),
                "Scanned path should exist on disk: {}",
                path_str
            );
        }

        // Should find .rs files
        let rs_count = entries
            .iter()
            .filter(|e| entry_path_str(e).ends_with(".rs"))
            .count();
        assert!(
            rs_count > 0,
            "Should find at least one .rs file in src/"
        );

        println!(
            "Scanner found {} entries ({} .rs files) in src/",
            entries.len(),
            rs_count
        );
    }

    #[test]
    fn test_scanner_entry_metadata() {
        let dir = make_test_dir();
        let scanner = FilesystemScanner::new();
        let entries = scanner.scan(dir.path());

        for entry in &entries {
            // path_len should be valid
            assert!(entry.path_len > 0, "path_len should be > 0");
            assert!(
                (entry.path_len as usize) <= GPU_PATH_MAX_LEN,
                "path_len should not exceed max"
            );

            // mtime should be nonzero for newly created files
            assert!(entry.mtime > 0, "mtime should be > 0 for test files");

            // Size should be > 0 for files with content
            assert!(entry.size() > 0, "File size should be > 0");
        }
    }

    #[test]
    fn test_scanner_max_depth() {
        let dir = make_test_dir();
        let scanner = FilesystemScanner::with_config(ScannerConfig {
            max_depth: Some(1),
            ..Default::default()
        });
        let entries = scanner.scan(dir.path());

        // max_depth=1 means only root-level files (depth 0 is root dir, depth 1 is files in root)
        let paths: Vec<String> = entries.iter().map(|e| entry_path_str(e).to_string()).collect();
        for path in &paths {
            // Should not find files in subdirectories
            let relative = path.strip_prefix(&dir.path().to_string_lossy().to_string())
                .unwrap_or(path);
            let depth = relative.chars().filter(|c| *c == '/').count();
            assert!(
                depth <= 1,
                "max_depth=1 should not find deeply nested files: {}",
                path
            );
        }
    }

    #[test]
    fn test_scanner_empty_dir() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let scanner = FilesystemScanner::new();
        let entries = scanner.scan(dir.path());

        assert!(
            entries.is_empty(),
            "Empty directory should produce no entries"
        );
    }

    #[test]
    fn test_scanner_deterministic_order() {
        let dir = make_test_dir();
        let scanner = FilesystemScanner::new();

        // Scan twice -- results should be in same order (sorted by path)
        let entries1 = scanner.scan(dir.path());
        let entries2 = scanner.scan(dir.path());

        assert_eq!(
            entries1.len(),
            entries2.len(),
            "Two scans should produce same count"
        );

        for (a, b) in entries1.iter().zip(entries2.iter()) {
            assert_eq!(
                entry_path_str(a),
                entry_path_str(b),
                "Scan results should be deterministically ordered"
            );
        }
    }

    #[test]
    fn test_scanner_gitignore() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // The ignore crate requires a git repo to recognize .gitignore files
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .expect("git init failed");

        // Create a .gitignore that ignores *.log files
        fs::write(dir.path().join(".gitignore"), "*.log\nbuild/\n").unwrap();

        // Create files
        fs::write(dir.path().join("code.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("debug.log"), "log output").unwrap();
        fs::write(dir.path().join("notes.txt"), "notes").unwrap();

        // Create ignored directory
        let build_dir = dir.path().join("build");
        fs::create_dir(&build_dir).unwrap();
        fs::write(build_dir.join("output.o"), "binary").unwrap();

        let scanner = FilesystemScanner::new();
        let entries = scanner.scan(dir.path());

        let paths: Vec<String> = entries.iter().map(|e| entry_path_str(e).to_string()).collect();

        // Should find code.rs and notes.txt
        let has_code = paths.iter().any(|p| p.contains("code.rs"));
        let has_notes = paths.iter().any(|p| p.contains("notes.txt"));
        assert!(has_code, "Should find code.rs. Found: {:?}", paths);
        assert!(has_notes, "Should find notes.txt. Found: {:?}", paths);

        // Should NOT find debug.log or build/output.o
        let has_log = paths.iter().any(|p| p.contains("debug.log"));
        let has_build = paths.iter().any(|p| p.contains("build/"));
        assert!(!has_log, "Should skip .log files per .gitignore. Found: {:?}", paths);
        assert!(!has_build, "Should skip build/ per .gitignore. Found: {:?}", paths);
    }
}
