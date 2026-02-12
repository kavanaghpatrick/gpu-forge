//! Filesystem edge case tests for gpu-search.
//!
//! Tests the search pipeline handles unusual filesystem conditions gracefully:
//! empty files, single-byte files, NUL-only files (binary detection), Unicode
//! filenames, hidden files/directories, deeply nested directories, .gitignore
//! patterns, and symlinks.
//!
//! NOTE: >4GB file test is intentionally skipped (too slow for CI).
//! This is documented as a known limitation.

use std::fs;
use std::io::Write;

use gpu_search::index::scanner::{FilesystemScanner, ScannerConfig};
use gpu_search::search::binary::{is_binary_content, BinaryDetector};
use gpu_search::search::ignore::GitignoreFilter;

// ============================================================================
// Helper: create a temp dir with git init (needed for ignore crate)
// ============================================================================

fn make_git_dir() -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    std::process::Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .output()
        .expect("git init");
    dir
}

// ============================================================================
// Empty files (0 bytes)
// ============================================================================

#[test]
fn test_empty_file_scanner_includes() {
    let dir = make_git_dir();
    fs::File::create(dir.path().join("empty.txt")).unwrap();
    fs::write(dir.path().join("notempty.txt"), "hello").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());
    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    // Scanner should include empty files (they are valid files)
    assert!(
        paths.iter().any(|p| p.contains("empty.txt")),
        "Scanner should include empty files. Found: {:?}",
        paths
    );
}

#[test]
fn test_empty_file_not_binary() {
    // Empty files should NOT be classified as binary
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("empty.txt");
    fs::File::create(&path).unwrap();

    let detector = BinaryDetector::new();
    assert!(
        !detector.should_skip(&path),
        "Empty files should not be detected as binary"
    );
}

#[test]
fn test_empty_content_not_binary() {
    let empty: &[u8] = &[];
    assert!(
        !is_binary_content(empty),
        "Empty content should not be binary"
    );
}

// ============================================================================
// Single-byte files
// ============================================================================

#[test]
fn test_single_byte_file() {
    let dir = make_git_dir();
    fs::write(dir.path().join("one.txt"), "x").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        paths.iter().any(|p| p.contains("one.txt")),
        "Single-byte file should be found. Found: {:?}",
        paths
    );
}

#[test]
fn test_single_byte_not_binary() {
    let data = [b'A'];
    assert!(
        !is_binary_content(&data),
        "Single ASCII byte should not be binary"
    );
}

#[test]
fn test_single_nul_byte_is_binary() {
    let data = [0u8];
    assert!(
        is_binary_content(&data),
        "Single NUL byte should be detected as binary"
    );
}

// ============================================================================
// Files with only NUL bytes (binary detection)
// ============================================================================

#[test]
fn test_nul_only_file_detected_as_binary() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("allnul.dat");
    {
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(&[0u8; 1024]).unwrap();
    }

    let detector = BinaryDetector::new();
    assert!(
        detector.should_skip(&path),
        "File with only NUL bytes should be detected as binary"
    );
}

#[test]
fn test_nul_only_content_is_binary() {
    let data = vec![0u8; 4096];
    assert!(
        is_binary_content(&data),
        "All-NUL content should be binary"
    );
}

#[test]
fn test_mixed_text_with_nul_is_binary() {
    let mut data = b"Hello, world! This is text.".to_vec();
    data.push(0u8); // append NUL
    data.extend_from_slice(b"More text after NUL.");
    assert!(
        is_binary_content(&data),
        "Text with embedded NUL should be binary"
    );
}

// ============================================================================
// Unicode filenames: CJK characters, emoji, mixed scripts
// ============================================================================

#[test]
fn test_unicode_filename_cjk() {
    let dir = make_git_dir();

    // CJK characters
    let cjk_name = dir.path().join("\u{4F60}\u{597D}\u{4E16}\u{754C}.txt"); // "hello world" in Chinese
    fs::write(&cjk_name, "CJK content").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    assert!(
        !entries.is_empty(),
        "Scanner should handle CJK filenames without crash"
    );

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        paths.iter().any(|p| p.contains("\u{4F60}\u{597D}")),
        "CJK filename should be found. Paths: {:?}",
        paths
    );
}

#[test]
fn test_unicode_filename_emoji() {
    let dir = make_git_dir();

    // Emoji filename
    let emoji_name = dir.path().join("\u{1F680}_rocket.txt"); // rocket emoji
    fs::write(&emoji_name, "emoji content").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    assert!(
        !entries.is_empty(),
        "Scanner should handle emoji filenames without crash"
    );

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        paths.iter().any(|p| p.contains("rocket")),
        "Emoji filename should be found. Paths: {:?}",
        paths
    );
}

#[test]
fn test_unicode_filename_mixed_scripts() {
    let dir = make_git_dir();

    // Mixed: Latin + Cyrillic + Arabic
    let mixed_name = dir.path().join("hello_\u{041F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}_\u{0645}\u{0631}\u{062D}\u{0628}\u{0627}.txt");
    fs::write(&mixed_name, "mixed script content").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    assert!(
        !entries.is_empty(),
        "Scanner should handle mixed-script filenames without crash"
    );
}

#[test]
fn test_unicode_filename_nfc_nfd() {
    let dir = make_git_dir();

    // NFC form: single codepoint for accented character (e with acute = U+00E9)
    let nfc_name = dir.path().join("caf\u{00E9}.txt");
    fs::write(&nfc_name, "NFC content").unwrap();

    // NFD form: base letter + combining mark (e + U+0301)
    // Note: macOS HFS+ normalizes to NFD, APFS preserves original form.
    // The scanner should handle both without crashing.
    let nfd_name = dir.path().join("re\u{0301}sume\u{0301}.txt");
    fs::write(&nfd_name, "NFD content").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    // Both files should be found (or at least no crash)
    assert!(
        entries.len() >= 2,
        "Scanner should find NFC and NFD files. Found {} entries",
        entries.len()
    );
}

// ============================================================================
// Hidden files and directories
// ============================================================================

#[test]
fn test_hidden_files_skipped_by_default() {
    let dir = make_git_dir();

    fs::write(dir.path().join(".hidden"), "secret").unwrap();
    fs::write(dir.path().join("visible.txt"), "public").unwrap();

    let scanner = FilesystemScanner::new(); // default: skip_hidden=true
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        !paths.iter().any(|p| p.contains(".hidden")),
        "Hidden files should be skipped by default. Found: {:?}",
        paths
    );
    assert!(
        paths.iter().any(|p| p.contains("visible.txt")),
        "Visible files should be included. Found: {:?}",
        paths
    );
}

#[test]
fn test_hidden_directories_skipped_by_default() {
    let dir = make_git_dir();

    let hidden_dir = dir.path().join(".secret_dir");
    fs::create_dir(&hidden_dir).unwrap();
    fs::write(hidden_dir.join("inside.txt"), "hidden dir content").unwrap();

    fs::write(dir.path().join("visible.txt"), "public").unwrap();

    let scanner = FilesystemScanner::new();
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        !paths.iter().any(|p| p.contains(".secret_dir")),
        "Files inside hidden directories should be skipped. Found: {:?}",
        paths
    );
}

#[test]
fn test_hidden_files_included_when_configured() {
    let dir = make_git_dir();

    fs::write(dir.path().join(".env"), "SECRET=value").unwrap();
    fs::write(dir.path().join("app.rs"), "fn main() {}").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        skip_hidden: false,
        respect_gitignore: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        paths.iter().any(|p| p.contains(".env")),
        "Hidden files should be included when skip_hidden=false. Found: {:?}",
        paths
    );
}

// ============================================================================
// Deeply nested directories (50+ levels)
// ============================================================================

#[test]
fn test_deeply_nested_directories() {
    let dir = make_git_dir();

    // Create 60-level deep directory structure
    let depth = 60;
    let mut current = dir.path().to_path_buf();
    for i in 0..depth {
        current = current.join(format!("d{}", i));
    }
    fs::create_dir_all(&current).unwrap();
    fs::write(current.join("deep.txt"), "found at the bottom").unwrap();

    // Also put a file at the top
    fs::write(dir.path().join("top.txt"), "at the top").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        paths.iter().any(|p| p.contains("top.txt")),
        "Should find top-level file. Found: {:?}",
        paths
    );

    // The deeply nested file may or may not be found depending on path length
    // (224-byte path limit in GpuPathEntry). The key is NO crash.
    // Just verify the scanner completes without panic.
    println!(
        "Deep nesting test: found {} entries. Deep file path length: {} bytes",
        entries.len(),
        current.join("deep.txt").to_string_lossy().len()
    );
}

#[test]
fn test_deeply_nested_no_crash_with_max_depth() {
    let dir = make_git_dir();

    // Create 100-level deep structure
    let depth = 100;
    let mut current = dir.path().to_path_buf();
    for i in 0..depth {
        current = current.join(format!("l{}", i));
    }
    fs::create_dir_all(&current).unwrap();
    fs::write(current.join("bottom.txt"), "very deep").unwrap();

    // Scanner with max_depth should gracefully limit traversal
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        max_depth: Some(10),
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    // Should complete without panic. Deeply nested file should NOT appear.
    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        !paths.iter().any(|p| p.contains("bottom.txt")),
        "File beyond max_depth should not be found. Found: {:?}",
        paths
    );
}

// ============================================================================
// .gitignore patterns
// ============================================================================

#[test]
fn test_gitignore_glob_patterns() {
    let dir = make_git_dir();

    fs::write(dir.path().join(".gitignore"), "*.log\n*.tmp\nbuild/\n").unwrap();

    fs::write(dir.path().join("app.rs"), "fn main() {}").unwrap();
    fs::write(dir.path().join("debug.log"), "log output").unwrap();
    fs::write(dir.path().join("temp.tmp"), "temporary").unwrap();

    let build_dir = dir.path().join("build");
    fs::create_dir(&build_dir).unwrap();
    fs::write(build_dir.join("output.bin"), "binary").unwrap();

    let scanner = FilesystemScanner::new();
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        paths.iter().any(|p| p.contains("app.rs")),
        "app.rs should not be ignored"
    );
    assert!(
        !paths.iter().any(|p| p.contains("debug.log")),
        "*.log should be ignored. Found: {:?}",
        paths
    );
    assert!(
        !paths.iter().any(|p| p.contains("temp.tmp")),
        "*.tmp should be ignored. Found: {:?}",
        paths
    );
    assert!(
        !paths.iter().any(|p| p.contains("build/")),
        "build/ contents should be ignored. Found: {:?}",
        paths
    );
}

#[test]
fn test_gitignore_negation_pattern() {
    let dir = make_git_dir();

    fs::write(
        dir.path().join(".gitignore"),
        "*.log\n!important.log\n",
    )
    .unwrap();

    fs::write(dir.path().join("debug.log"), "debug").unwrap();
    fs::write(dir.path().join("important.log"), "keep me").unwrap();
    fs::write(dir.path().join("app.rs"), "code").unwrap();

    let filter = GitignoreFilter::from_directory(dir.path()).unwrap();

    assert!(
        filter.is_ignored(&dir.path().join("debug.log")),
        "debug.log should be ignored"
    );
    assert!(
        !filter.is_ignored(&dir.path().join("important.log")),
        "important.log should NOT be ignored (negation)"
    );
    assert!(
        !filter.is_ignored(&dir.path().join("app.rs")),
        "app.rs should not be ignored"
    );
}

#[test]
fn test_gitignore_nested_directory_pattern() {
    let dir = make_git_dir();

    fs::write(dir.path().join(".gitignore"), "target/\nnode_modules/\n").unwrap();

    let target = dir.path().join("target");
    fs::create_dir(&target).unwrap();
    fs::write(target.join("debug.exe"), "binary").unwrap();

    let nm = dir.path().join("node_modules");
    fs::create_dir(&nm).unwrap();
    fs::write(nm.join("package.json"), "{}").unwrap();

    fs::write(dir.path().join("src.rs"), "code").unwrap();

    let filter = GitignoreFilter::from_directory(dir.path()).unwrap();

    assert!(
        filter.is_ignored_with_dir_hint(&target, true),
        "target/ should be ignored"
    );
    assert!(
        filter.is_ignored_with_dir_hint(&nm, true),
        "node_modules/ should be ignored"
    );
    assert!(
        !filter.is_ignored(&dir.path().join("src.rs")),
        "src.rs should not be ignored"
    );
}

// ============================================================================
// Symlinks
// ============================================================================

#[cfg(unix)]
#[test]
fn test_symlink_to_file() {
    let dir = make_git_dir();

    let real_file = dir.path().join("real.txt");
    fs::write(&real_file, "real content").unwrap();

    let link = dir.path().join("link.txt");
    std::os::unix::fs::symlink(&real_file, &link).unwrap();

    // Default: follow_symlinks=false, scanner should skip symlinks
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    // real.txt should always be found
    assert!(
        paths.iter().any(|p| p.contains("real.txt")),
        "real.txt should be found. Found: {:?}",
        paths
    );

    // No crash regardless of symlink handling
    println!("Symlink test: found {} entries", entries.len());
}

#[cfg(unix)]
#[test]
fn test_broken_symlink_no_crash() {
    let dir = make_git_dir();

    // Create a symlink to a nonexistent target
    let broken_link = dir.path().join("broken_link.txt");
    std::os::unix::fs::symlink("/nonexistent/target/file.txt", &broken_link).unwrap();

    fs::write(dir.path().join("valid.txt"), "valid content").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        follow_symlinks: true, // explicitly try to follow broken symlink
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    // Should not crash. valid.txt should be found.
    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        paths.iter().any(|p| p.contains("valid.txt")),
        "valid.txt should be found despite broken symlink. Found: {:?}",
        paths
    );
}

#[cfg(unix)]
#[test]
fn test_circular_symlink_no_crash() {
    let dir = make_git_dir();

    // Create circular symlinks: a -> b, b -> a
    let link_a = dir.path().join("link_a");
    let link_b = dir.path().join("link_b");
    std::os::unix::fs::symlink(&link_b, &link_a).unwrap();
    std::os::unix::fs::symlink(&link_a, &link_b).unwrap();

    fs::write(dir.path().join("safe.txt"), "safe content").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        follow_symlinks: true, // try to follow circular symlinks
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    // The key assertion: no infinite loop, no crash
    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        paths.iter().any(|p| p.contains("safe.txt")),
        "safe.txt should be found despite circular symlinks. Found: {:?}",
        paths
    );
}

#[cfg(unix)]
#[test]
fn test_symlink_to_directory_no_crash() {
    let dir = make_git_dir();

    let subdir = dir.path().join("real_dir");
    fs::create_dir(&subdir).unwrap();
    fs::write(subdir.join("inside.txt"), "inside real dir").unwrap();

    let dir_link = dir.path().join("dir_link");
    std::os::unix::fs::symlink(&subdir, &dir_link).unwrap();

    fs::write(dir.path().join("top.txt"), "top level").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    // No crash expected
    let entries = scanner.scan(dir.path());

    assert!(
        !entries.is_empty(),
        "Should find at least some files with symlinked directory"
    );
}

// ============================================================================
// Permission denied
// ============================================================================

#[cfg(unix)]
#[test]
fn test_permission_denied_file_no_crash() {
    let dir = make_git_dir();

    let forbidden = dir.path().join("forbidden.txt");
    fs::write(&forbidden, "you can't read me").unwrap();

    // Remove all permissions
    std::process::Command::new("chmod")
        .args(["000", &forbidden.to_string_lossy()])
        .output()
        .expect("chmod");

    fs::write(dir.path().join("readable.txt"), "you can read me").unwrap();

    // Binary detector should handle permission-denied gracefully
    let detector = BinaryDetector::new();
    // should_skip returns false on read error (assumes text)
    let result = detector.should_skip(&forbidden);
    // No crash is the key assertion
    println!("Permission denied file: should_skip={}", result);

    // Restore permissions for cleanup
    std::process::Command::new("chmod")
        .args(["644", &forbidden.to_string_lossy()])
        .output()
        .expect("chmod restore");
}

#[cfg(unix)]
#[test]
fn test_permission_denied_directory_no_crash() {
    let dir = make_git_dir();

    let forbidden_dir = dir.path().join("noaccess");
    fs::create_dir(&forbidden_dir).unwrap();
    fs::write(forbidden_dir.join("hidden.txt"), "hidden").unwrap();

    // Remove directory read permission
    std::process::Command::new("chmod")
        .args(["000", &forbidden_dir.to_string_lossy()])
        .output()
        .expect("chmod");

    fs::write(dir.path().join("accessible.txt"), "accessible").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    // Should not crash -- just skip the inaccessible directory
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    assert!(
        paths.iter().any(|p| p.contains("accessible.txt")),
        "Should still find accessible files. Found: {:?}",
        paths
    );

    // Restore permissions for cleanup
    std::process::Command::new("chmod")
        .args(["755", &forbidden_dir.to_string_lossy()])
        .output()
        .expect("chmod restore");
}

// ============================================================================
// Binary detector edge cases
// ============================================================================

#[test]
fn test_binary_detector_nul_at_various_positions() {
    // NUL at position 0
    let data0 = [0u8, b'h', b'e', b'l', b'l', b'o'];
    assert!(is_binary_content(&data0));

    // NUL in middle
    let data_mid = [b'h', b'e', 0u8, b'l', b'o'];
    assert!(is_binary_content(&data_mid));

    // NUL at end (within check window)
    let mut data_end = vec![b'A'; 100];
    data_end[99] = 0u8;
    assert!(is_binary_content(&data_end));
}

#[test]
fn test_binary_detector_unicode_text_not_binary() {
    // Valid UTF-8 multibyte (CJK, emoji) should NOT be flagged as binary
    let text = "\u{4F60}\u{597D}\u{4E16}\u{754C}\u{1F680}".as_bytes(); // "hello world" in Chinese + rocket
    assert!(
        !is_binary_content(text),
        "Valid UTF-8 text (CJK + emoji) should not be binary"
    );
}

#[test]
fn test_binary_detector_latin_extended_not_binary() {
    // Latin-1 extended characters in UTF-8 (accented letters)
    let text = "caf\u{00E9} r\u{00E9}sum\u{00E9}".as_bytes();
    assert!(
        !is_binary_content(text),
        "UTF-8 accented text should not be binary"
    );
}

// ============================================================================
// GitignoreFilter edge cases
// ============================================================================

#[test]
fn test_gitignore_empty_gitignore() {
    let dir = make_git_dir();
    fs::write(dir.path().join(".gitignore"), "").unwrap();
    fs::write(dir.path().join("anything.txt"), "content").unwrap();

    let filter = GitignoreFilter::from_directory(dir.path()).unwrap();
    assert!(
        !filter.is_ignored(&dir.path().join("anything.txt")),
        "Empty .gitignore should ignore nothing"
    );
}

#[test]
fn test_gitignore_comments_only() {
    let dir = make_git_dir();
    fs::write(
        dir.path().join(".gitignore"),
        "# comment 1\n# comment 2\n# comment 3\n",
    )
    .unwrap();
    fs::write(dir.path().join("file.txt"), "content").unwrap();

    let filter = GitignoreFilter::from_directory(dir.path()).unwrap();
    assert!(
        !filter.is_ignored(&dir.path().join("file.txt")),
        "Comments-only .gitignore should ignore nothing"
    );
}

#[test]
fn test_gitignore_wildcard_patterns() {
    let dir = tempfile::TempDir::new().unwrap();

    // Double-star glob
    let filter = GitignoreFilter::from_str(dir.path(), "**/test_*.txt\n").unwrap();

    assert!(
        filter.is_ignored(&dir.path().join("test_foo.txt")),
        "**/test_*.txt should match test_foo.txt"
    );
    assert!(
        filter.is_ignored(&dir.path().join("sub").join("test_bar.txt")),
        "**/test_*.txt should match sub/test_bar.txt"
    );
    assert!(
        !filter.is_ignored(&dir.path().join("main.rs")),
        "**/test_*.txt should not match main.rs"
    );
}

// ============================================================================
// Path length edge cases
// ============================================================================

#[test]
fn test_path_exceeding_gpu_path_max_len() {
    let dir = make_git_dir();

    // Create a path that will exceed the 224-byte GPU_PATH_MAX_LEN
    // Build with shorter segments to avoid OS max filename limits
    let mut current = dir.path().to_path_buf();
    // Each segment is 20 chars + path separator, we need > 224 bytes total
    for i in 0..15 {
        current = current.join(format!("segment_{:04}_abcdefgh", i));
    }
    fs::create_dir_all(&current).unwrap();
    fs::write(current.join("long_path_file.txt"), "content").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    // Should not crash. File with too-long path is silently skipped.
    let entries = scanner.scan(dir.path());

    // Verify no entry has path_len > 224
    for entry in &entries {
        assert!(
            (entry.path_len as usize) <= 224,
            "No entry should have path_len > 224 (GPU_PATH_MAX_LEN)"
        );
    }
}

// ============================================================================
// Combined edge cases: multiple edge conditions simultaneously
// ============================================================================

#[test]
fn test_combined_edge_cases() {
    let dir = make_git_dir();

    // .gitignore
    fs::write(dir.path().join(".gitignore"), "*.log\nbuild/\n").unwrap();

    // Empty file
    fs::File::create(dir.path().join("empty.rs")).unwrap();

    // Single byte
    fs::write(dir.path().join("one.rs"), "x").unwrap();

    // NUL-only (binary)
    {
        let mut f = fs::File::create(dir.path().join("binary.dat")).unwrap();
        f.write_all(&[0u8; 256]).unwrap();
    }

    // Unicode filename
    fs::write(dir.path().join("\u{1F60A}_smile.txt"), "smile").unwrap();

    // Hidden file
    fs::write(dir.path().join(".secret"), "hidden").unwrap();

    // Ignored file
    fs::write(dir.path().join("output.log"), "ignored").unwrap();

    // Nested dir with file
    let sub = dir.path().join("src");
    fs::create_dir(&sub).unwrap();
    fs::write(sub.join("main.rs"), "fn main() {}").unwrap();

    // Ignored dir
    let build = dir.path().join("build");
    fs::create_dir(&build).unwrap();
    fs::write(build.join("out.o"), "binary build output").unwrap();

    // Scan with default settings
    let scanner = FilesystemScanner::new();
    let entries = scanner.scan(dir.path());

    // Key assertion: no crash, and reasonable results
    assert!(
        !entries.is_empty(),
        "Should find at least some files in combined test"
    );

    let paths: Vec<String> = entries
        .iter()
        .map(|e| {
            let len = e.path_len as usize;
            std::str::from_utf8(&e.path[..len]).unwrap_or("").to_string()
        })
        .collect();

    // .gitignore should exclude *.log and build/
    assert!(
        !paths.iter().any(|p| p.contains("output.log")),
        "*.log should be ignored"
    );
    assert!(
        !paths.iter().any(|p| p.contains("build/")),
        "build/ should be ignored"
    );

    // Hidden should be excluded by default
    assert!(
        !paths.iter().any(|p| p.contains(".secret")),
        ".secret should be hidden"
    );

    // Source files should be found
    assert!(
        paths.iter().any(|p| p.contains("main.rs")),
        "src/main.rs should be found"
    );

    println!("Combined edge case test: found {} files", entries.len());
}

// ============================================================================
// NOTE: >4GB file test intentionally skipped
// ============================================================================
// Creating a >4GB file in a test is too slow for CI and wastes disk space.
// The system handles large files via:
// - MmapBuffer: uses 64-bit offsets, supports files up to process address space limit
// - StreamingSearchEngine: processes files in 64MB chunks
// - GpuPathEntry: size stored as split u32 (size_lo/size_hi) supporting up to 2^64 bytes
// Manual verification: tested with large files during development.
