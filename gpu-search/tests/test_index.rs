//! Index Lifecycle Integration Tests
//!
//! Tests the full filesystem index pipeline:
//! - FilesystemScanner: correct entry count, paths match
//! - SharedIndexManager: save/load cycle preserves entries
//! - Incremental updates: create/delete files reflected in new scans
//! - GpuIndexLoader: GPU buffer loading performance
//!
//! Run: cargo test --test test_index

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use tempfile::TempDir;

use objc2_metal::MTLBuffer;

use gpu_search::gpu::types::{GpuPathEntry, GPU_PATH_MAX_LEN};
use gpu_search::index::cache::{cache_key, MmapIndexCache};
use gpu_search::index::gpu_index::GpuResidentIndex;
use gpu_search::index::gpu_loader::GpuIndexLoader;
use gpu_search::index::scanner::{FilesystemScanner, ScannerConfig};
use gpu_search::index::shared_index::SharedIndexManager;

// ============================================================================
// Helpers
// ============================================================================

/// Create a temp directory with a known file structure.
fn make_test_dir() -> TempDir {
    let dir = TempDir::new().expect("Failed to create temp dir");

    // Root-level files
    fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
    fs::write(dir.path().join("lib.rs"), "pub mod test;").unwrap();
    fs::write(dir.path().join("README.md"), "# Test Project").unwrap();
    fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();

    // src/ subdirectory
    let src = dir.path().join("src");
    fs::create_dir(&src).unwrap();
    fs::write(src.join("mod.rs"), "mod sub;").unwrap();
    fs::write(src.join("utils.rs"), "pub fn help() {}").unwrap();
    fs::write(src.join("types.rs"), "pub struct Foo;").unwrap();

    // src/nested/ subdirectory
    let nested = src.join("nested");
    fs::create_dir(&nested).unwrap();
    fs::write(nested.join("deep.rs"), "// deep file").unwrap();

    dir
}

/// Extract path string from a GpuPathEntry.
fn entry_path(entry: &GpuPathEntry) -> &str {
    let len = entry.path_len as usize;
    std::str::from_utf8(&entry.path[..len]).unwrap_or("")
}

/// Count the expected files in make_test_dir():
/// main.rs, lib.rs, README.md, Cargo.toml, src/mod.rs, src/utils.rs, src/types.rs, src/nested/deep.rs
const EXPECTED_FILE_COUNT: usize = 8;

// ============================================================================
// 1. FilesystemScanner: correct entry count
// ============================================================================

#[test]
fn test_scanner_correct_entry_count() {
    let dir = make_test_dir();
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false, // no git init, so disable
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    assert_eq!(
        entries.len(),
        EXPECTED_FILE_COUNT,
        "Scanner should find exactly {} files, got {}. Paths: {:?}",
        EXPECTED_FILE_COUNT,
        entries.len(),
        entries.iter().map(|e| entry_path(e).to_string()).collect::<Vec<_>>()
    );
}

// ============================================================================
// 2. FilesystemScanner: paths match expected files
// ============================================================================

#[test]
fn test_scanner_paths_match_expected() {
    let dir = make_test_dir();
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries.iter().map(|e| entry_path(e).to_string()).collect();

    // Every expected file should be present
    let expected_suffixes = [
        "main.rs",
        "lib.rs",
        "README.md",
        "Cargo.toml",
        "mod.rs",
        "utils.rs",
        "types.rs",
        "deep.rs",
    ];

    for suffix in &expected_suffixes {
        assert!(
            paths.iter().any(|p| p.ends_with(suffix)),
            "Expected to find file ending with '{}' in scanned paths: {:?}",
            suffix,
            paths
        );
    }

    // Every scanned path should exist on disk
    for path_str in &paths {
        let path = PathBuf::from(path_str);
        assert!(
            path.exists(),
            "Scanned path should exist on disk: {}",
            path_str
        );
    }

    // Every scanned path should be under the root directory
    let root_str = dir.path().to_string_lossy().to_string();
    for path_str in &paths {
        assert!(
            path_str.starts_with(&root_str),
            "Scanned path '{}' should be under root '{}'",
            path_str,
            root_str
        );
    }
}

// ============================================================================
// 3. Index persistence: save to binary and reload
// ============================================================================

#[test]
fn test_index_save_to_binary_and_reload() {
    let dir = make_test_dir();
    let cache_dir = TempDir::new().expect("Failed to create cache dir");
    let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

    // Build index via scanner
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());
    let index = GpuResidentIndex::from_entries(entries);
    let original_count = index.entry_count();

    assert_eq!(original_count, EXPECTED_FILE_COUNT);

    // Save to disk
    let saved_path = manager
        .save(&index, dir.path())
        .expect("Failed to save index");
    assert!(saved_path.exists(), "Saved index file should exist on disk");

    // Verify file size is correct: 64 (header) + N * 256 (entries)
    let file_size = fs::metadata(&saved_path).unwrap().len() as usize;
    let expected_size = 64 + original_count * 256;
    assert_eq!(
        file_size, expected_size,
        "Index file size should be {} (64 header + {} * 256), got {}",
        expected_size, original_count, file_size
    );

    // Reload
    let loaded = manager
        .load(dir.path())
        .expect("Failed to load index");

    assert_eq!(
        loaded.entry_count(),
        original_count,
        "Loaded index should have same entry count"
    );
}

// ============================================================================
// 4. Save/load cycle preserves all entries
// ============================================================================

#[test]
fn test_save_load_preserves_all_entries() {
    let dir = make_test_dir();
    let cache_dir = TempDir::new().expect("Failed to create cache dir");
    let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

    // Build original index
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());
    let index = GpuResidentIndex::from_entries(entries);

    // Save and reload
    manager.save(&index, dir.path()).expect("Failed to save");
    let loaded = manager.load(dir.path()).expect("Failed to load");

    // Verify every entry field matches
    assert_eq!(index.entry_count(), loaded.entry_count());
    for i in 0..index.entry_count() {
        let orig = index.get_entry(i).unwrap();
        let load = loaded.get_entry(i).unwrap();

        assert_eq!(
            entry_path(orig),
            entry_path(load),
            "Entry {} path mismatch",
            i
        );
        assert_eq!(orig.path_len, load.path_len, "Entry {} path_len mismatch", i);
        assert_eq!(orig.flags, load.flags, "Entry {} flags mismatch", i);
        assert_eq!(orig.size_lo, load.size_lo, "Entry {} size_lo mismatch", i);
        assert_eq!(orig.size_hi, load.size_hi, "Entry {} size_hi mismatch", i);
        assert_eq!(orig.mtime, load.mtime, "Entry {} mtime mismatch", i);
        assert_eq!(
            orig.parent_idx, load.parent_idx,
            "Entry {} parent_idx mismatch",
            i
        );

        // Also verify byte-for-byte equality of the full 256-byte struct
        let orig_bytes: &[u8; 256] =
            unsafe { &*(orig as *const GpuPathEntry as *const [u8; 256]) };
        let load_bytes: &[u8; 256] =
            unsafe { &*(load as *const GpuPathEntry as *const [u8; 256]) };
        assert_eq!(
            orig_bytes, load_bytes,
            "Entry {} full 256-byte mismatch",
            i
        );
    }
}

// ============================================================================
// 5. Save/load via mmap cache (zero-copy path)
// ============================================================================

#[test]
fn test_mmap_cache_roundtrip() {
    let dir = make_test_dir();
    let cache_dir = TempDir::new().expect("Failed to create cache dir");
    let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

    // Build and save
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());
    let index = GpuResidentIndex::from_entries(entries);

    let saved_path = manager
        .save(&index, dir.path())
        .expect("Failed to save");

    // Load via mmap
    let root_hash = cache_key(dir.path());
    let mmap_cache = MmapIndexCache::load_mmap(&saved_path, Some(root_hash))
        .expect("Failed to mmap load");

    assert_eq!(mmap_cache.entry_count(), index.entry_count());

    // Verify entries via mmap match original
    for i in 0..index.entry_count() {
        let orig = index.get_entry(i).unwrap();
        let cached = mmap_cache.get_entry(i).unwrap();
        assert_eq!(
            entry_path(orig),
            entry_path(cached),
            "Mmap entry {} path mismatch",
            i
        );
    }

    // Convert to resident index and verify
    let resident = mmap_cache.into_resident_index();
    assert_eq!(resident.entry_count(), index.entry_count());
}

// ============================================================================
// 6. Incremental update: file creation reflected in new scan
// ============================================================================

#[test]
fn test_file_creation_reflected_in_scan() {
    let dir = make_test_dir();
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });

    // Initial scan
    let entries_before = scanner.scan(dir.path());
    let count_before = entries_before.len();
    assert_eq!(count_before, EXPECTED_FILE_COUNT);

    // Create new files
    fs::write(dir.path().join("new_file.rs"), "// new file").unwrap();
    fs::write(
        dir.path().join("src").join("extra.rs"),
        "pub fn extra() {}",
    )
    .unwrap();

    // Rescan
    let entries_after = scanner.scan(dir.path());
    let count_after = entries_after.len();

    assert_eq!(
        count_after,
        count_before + 2,
        "After adding 2 files, count should increase by 2: {} -> {}",
        count_before,
        count_after
    );

    // Verify new files appear in scan
    let paths: Vec<String> = entries_after
        .iter()
        .map(|e| entry_path(e).to_string())
        .collect();
    assert!(
        paths.iter().any(|p| p.ends_with("new_file.rs")),
        "Should find new_file.rs after creation"
    );
    assert!(
        paths.iter().any(|p| p.ends_with("extra.rs")),
        "Should find extra.rs after creation"
    );
}

// ============================================================================
// 7. Incremental update: file deletion reflected in new scan
// ============================================================================

#[test]
fn test_file_deletion_reflected_in_scan() {
    let dir = make_test_dir();
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });

    // Initial scan
    let entries_before = scanner.scan(dir.path());
    let count_before = entries_before.len();
    assert_eq!(count_before, EXPECTED_FILE_COUNT);

    // Delete files
    fs::remove_file(dir.path().join("README.md")).unwrap();
    fs::remove_file(dir.path().join("src").join("types.rs")).unwrap();

    // Rescan
    let entries_after = scanner.scan(dir.path());
    let count_after = entries_after.len();

    assert_eq!(
        count_after,
        count_before - 2,
        "After deleting 2 files, count should decrease by 2: {} -> {}",
        count_before,
        count_after
    );

    // Verify deleted files are absent
    let paths: Vec<String> = entries_after
        .iter()
        .map(|e| entry_path(e).to_string())
        .collect();
    assert!(
        !paths.iter().any(|p| p.ends_with("README.md")),
        "README.md should be absent after deletion"
    );
    assert!(
        !paths.iter().any(|p| p.ends_with("types.rs")),
        "types.rs should be absent after deletion"
    );

    // Verify remaining files are still present
    assert!(
        paths.iter().any(|p| p.ends_with("main.rs")),
        "main.rs should still be present"
    );
    assert!(
        paths.iter().any(|p| p.ends_with("deep.rs")),
        "deep.rs should still be present"
    );
}

// ============================================================================
// 8. GPU buffer loading creates valid Metal buffer
// ============================================================================

#[test]
fn test_gpu_buffer_loading_creates_valid_buffer() {
    use objc2_metal::MTLCreateSystemDefaultDevice;

    let device = MTLCreateSystemDefaultDevice()
        .expect("No Metal device (test requires Apple Silicon)");

    let dir = make_test_dir();
    let cache_dir = TempDir::new().expect("Failed to create cache dir");

    let loader = GpuIndexLoader::with_cache_dir(&device, cache_dir.path().to_path_buf());

    // Load index (scan + build + save)
    let loaded = loader
        .load(dir.path())
        .expect("Failed to load index into GPU");

    // Verify entry count
    assert!(
        loaded.entry_count() > 0,
        "GPU loaded index should have entries"
    );

    // Verify buffer size
    let expected_buffer_size = loaded.entry_count() * GpuPathEntry::SIZE;
    assert!(
        loaded.buffer().length() >= expected_buffer_size,
        "GPU buffer too small: {} < {}",
        loaded.buffer().length(),
        expected_buffer_size
    );

    // Verify buffer contents are valid GpuPathEntry data
    unsafe {
        let buf_ptr = loaded.buffer().contents().as_ptr() as *const GpuPathEntry;
        for i in 0..loaded.entry_count() {
            let entry = &*buf_ptr.add(i);
            assert!(
                entry.path_len > 0 && (entry.path_len as usize) <= GPU_PATH_MAX_LEN,
                "Entry {} has invalid path_len: {}",
                i,
                entry.path_len
            );
            let path_str = entry_path(entry);
            assert!(!path_str.is_empty(), "Entry {} has empty path", i);
            let path = PathBuf::from(path_str);
            assert!(
                path.exists(),
                "GPU buffer entry {} path should exist on disk: {}",
                i,
                path_str
            );
        }
    }

    // Verify root path
    assert_eq!(loaded.root(), dir.path());

    println!(
        "GPU index: {} entries, {} bytes buffer, loaded in {:?}",
        loaded.entry_count(),
        loaded.buffer().length(),
        loaded.load_time()
    );
}

// ============================================================================
// 9. GPU buffer loading from cache (second load)
// ============================================================================

#[test]
fn test_gpu_buffer_loading_from_cache() {
    use objc2_metal::MTLCreateSystemDefaultDevice;

    let device = MTLCreateSystemDefaultDevice()
        .expect("No Metal device (test requires Apple Silicon)");

    let dir = make_test_dir();
    let cache_dir = TempDir::new().expect("Failed to create cache dir");

    let loader = GpuIndexLoader::with_cache_dir(&device, cache_dir.path().to_path_buf());

    // First load: cache miss
    let loaded1 = loader.load(dir.path()).expect("First load failed");
    assert!(!loaded1.from_cache(), "First load should not be from cache");

    // Second load: cache hit
    let loaded2 = loader.load(dir.path()).expect("Second load failed");
    assert!(loaded2.from_cache(), "Second load should be from cache");
    assert_eq!(
        loaded2.entry_count(),
        loaded1.entry_count(),
        "Cached load should have same entry count"
    );
}

// ============================================================================
// 10. Performance: scan of 1000+ files completes in reasonable time
// ============================================================================

#[test]
fn test_scan_1000_files_performance() {
    let dir = TempDir::new().expect("Failed to create temp dir");

    // Create 1000+ files across subdirectories
    for i in 0..10 {
        let subdir = dir.path().join(format!("dir_{:03}", i));
        fs::create_dir(&subdir).unwrap();
        for j in 0..100 {
            let filename = format!("file_{:04}.rs", i * 100 + j);
            fs::write(subdir.join(&filename), format!("// file {}", i * 100 + j)).unwrap();
        }
    }

    // Also create 50 files at root level
    for k in 0..50 {
        fs::write(
            dir.path().join(format!("root_{:03}.txt", k)),
            format!("root file {}", k),
        )
        .unwrap();
    }

    let total_files = 1000 + 50;

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });

    let start = Instant::now();
    let entries = scanner.scan(dir.path());
    let elapsed = start.elapsed();

    assert_eq!(
        entries.len(),
        total_files,
        "Should find exactly {} files, got {}",
        total_files,
        entries.len()
    );

    // Scan of 1050 files should complete in under 5 seconds (generous limit)
    assert!(
        elapsed.as_secs() < 5,
        "Scanning {} files took {:?} (expected < 5s)",
        total_files,
        elapsed
    );

    println!(
        "Scanned {} files in {:?} ({:.0} files/sec)",
        entries.len(),
        elapsed,
        entries.len() as f64 / elapsed.as_secs_f64()
    );
}

// ============================================================================
// 11. GPU buffer loading performance (< 10ms for cached entries)
// ============================================================================

#[test]
fn test_gpu_buffer_loading_performance() {
    use objc2_metal::MTLCreateSystemDefaultDevice;

    let device = MTLCreateSystemDefaultDevice()
        .expect("No Metal device (test requires Apple Silicon)");

    // Use actual gpu-search/src directory for a realistic file count
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let cache_dir = TempDir::new().expect("Failed to create cache dir");

    let loader = GpuIndexLoader::with_cache_dir(&device, cache_dir.path().to_path_buf());

    // First load to populate cache
    let _ = loader.load(&src_dir).expect("Initial load failed");

    // Measure cached load time
    let start = Instant::now();
    let loaded = loader.load(&src_dir).expect("Cached load failed");
    let elapsed = start.elapsed();

    assert!(loaded.from_cache(), "Should be loaded from cache");
    assert!(loaded.entry_count() > 0, "Should have entries");

    // Target: <10ms for cached load
    // Use generous 100ms limit for CI stability
    assert!(
        elapsed.as_millis() < 100,
        "Cached GPU buffer load took {:?} for {} entries (expected < 100ms)",
        elapsed,
        loaded.entry_count()
    );

    println!(
        "Cached GPU load: {} entries in {:?} ({:.2} ms)",
        loaded.entry_count(),
        elapsed,
        elapsed.as_secs_f64() * 1000.0
    );
}

// ============================================================================
// 12. Index staleness detection after file modification
// ============================================================================

#[test]
fn test_index_staleness_after_modification() {
    let dir = make_test_dir();
    let cache_dir = TempDir::new().expect("Failed to create cache dir");
    let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

    // Build and save initial index
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());
    let index = GpuResidentIndex::from_entries(entries);
    manager.save(&index, dir.path()).expect("Failed to save");

    // Index should not be stale immediately
    assert!(
        !manager.is_stale(dir.path(), std::time::Duration::from_secs(3600)),
        "Freshly saved index should not be stale"
    );

    // Sleep > 1 second to ensure mtime change is detectable
    // (saved_at uses unix seconds, so directory mtime must be strictly greater)
    std::thread::sleep(std::time::Duration::from_millis(1100));

    // Modify the directory (add a file)
    fs::write(dir.path().join("trigger_stale.txt"), "trigger").unwrap();

    // Now the index should be stale (dir mtime > index saved_at)
    assert!(
        manager.is_stale(dir.path(), std::time::Duration::from_secs(3600)),
        "Index should be stale after directory modification"
    );
}

// ============================================================================
// 13. Scanner with .gitignore respects rules
// ============================================================================

#[test]
fn test_scanner_gitignore_integration() {
    let dir = TempDir::new().expect("Failed to create temp dir");

    // Initialize git repo (required for ignore crate to parse .gitignore)
    std::process::Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .output()
        .expect("git init failed");

    // Create .gitignore
    fs::write(dir.path().join(".gitignore"), "*.log\nbuild/\n").unwrap();

    // Create files
    fs::write(dir.path().join("code.rs"), "fn main() {}").unwrap();
    fs::write(dir.path().join("debug.log"), "log output").unwrap();
    fs::write(dir.path().join("notes.txt"), "notes").unwrap();

    let build_dir = dir.path().join("build");
    fs::create_dir(&build_dir).unwrap();
    fs::write(build_dir.join("output.o"), "binary").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: true,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries.iter().map(|e| entry_path(e).to_string()).collect();

    // Should find code.rs and notes.txt
    assert!(
        paths.iter().any(|p| p.ends_with("code.rs")),
        "Should find code.rs. Found: {:?}",
        paths
    );
    assert!(
        paths.iter().any(|p| p.ends_with("notes.txt")),
        "Should find notes.txt. Found: {:?}",
        paths
    );

    // Should NOT find .log or build/
    assert!(
        !paths.iter().any(|p| p.ends_with("debug.log")),
        "Should skip *.log files. Found: {:?}",
        paths
    );
    assert!(
        !paths.iter().any(|p| p.contains("build/")),
        "Should skip build/ dir. Found: {:?}",
        paths
    );
}

// ============================================================================
// 14. Empty directory produces valid empty index
// ============================================================================

#[test]
fn test_empty_directory_index() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_dir = TempDir::new().expect("Failed to create cache dir");
    let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());
    assert!(entries.is_empty(), "Empty dir should produce no entries");

    // Save and reload empty index
    let index = GpuResidentIndex::from_entries(entries);
    manager.save(&index, dir.path()).expect("Failed to save empty index");
    let loaded = manager.load(dir.path()).expect("Failed to load empty index");
    assert_eq!(loaded.entry_count(), 0, "Loaded empty index should have 0 entries");
}

// ============================================================================
// 15. Large scan with full pipeline: scan -> build -> save -> mmap load -> GPU
// ============================================================================

#[test]
fn test_full_pipeline_scan_to_gpu() {
    use objc2_metal::MTLCreateSystemDefaultDevice;

    let device = MTLCreateSystemDefaultDevice()
        .expect("No Metal device (test requires Apple Silicon)");

    let dir = TempDir::new().expect("Failed to create temp dir");

    // Create 200 files
    for i in 0..200 {
        fs::write(
            dir.path().join(format!("file_{:04}.rs", i)),
            format!("// file content {}", i),
        )
        .unwrap();
    }

    let cache_dir = TempDir::new().expect("Failed to create cache dir");
    let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());

    // Step 1: Scan
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());
    assert_eq!(entries.len(), 200, "Should scan exactly 200 files");

    // Step 2: Build GpuResidentIndex
    let index = GpuResidentIndex::from_entries(entries);
    assert_eq!(index.entry_count(), 200);

    // Step 3: Save to binary
    let saved_path = manager.save(&index, dir.path()).expect("Failed to save");
    assert!(saved_path.exists());

    // Step 4: Load via mmap
    let root_hash = cache_key(dir.path());
    let mmap_cache = MmapIndexCache::load_mmap(&saved_path, Some(root_hash))
        .expect("Failed to mmap load");
    assert_eq!(mmap_cache.entry_count(), 200);

    // Step 5: Convert to resident + upload to GPU
    let mut resident = mmap_cache.into_resident_index();
    assert_eq!(resident.entry_count(), 200);

    let buffer = resident.to_gpu_buffer(&device);
    let expected_size = 200 * GpuPathEntry::SIZE;
    assert!(
        buffer.length() >= expected_size,
        "GPU buffer too small: {} < {}",
        buffer.length(),
        expected_size
    );

    // Verify GPU buffer contents
    unsafe {
        let buf_ptr = buffer.contents().as_ptr() as *const GpuPathEntry;
        for i in 0..200 {
            let entry = &*buf_ptr.add(i);
            assert!(entry.path_len > 0, "Entry {} should have non-zero path_len", i);
        }
    }

    println!(
        "Full pipeline: 200 files -> {} bytes GPU buffer",
        buffer.length()
    );
}

// ============================================================================
// 16. IndexWatcher integration test: create/delete/rename detection
// ============================================================================

#[test]
fn test_watcher_detects_file_changes() {
    use gpu_search::index::watcher::IndexWatcher;

    let dir = TempDir::new().expect("Failed to create temp dir");
    let cache_dir = TempDir::new().expect("Failed to create cache dir");

    // Create initial files (not hidden, so default scanner finds them)
    fs::write(dir.path().join("alpha.rs"), "fn alpha() {}").unwrap();
    fs::write(dir.path().join("beta.rs"), "fn beta() {}").unwrap();
    fs::write(dir.path().join("gamma.rs"), "fn gamma() {}").unwrap();

    // Build and persist the initial index
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());
    let initial_count = entries.len();
    assert_eq!(initial_count, 3, "Should start with 3 files");

    let manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());
    let index = GpuResidentIndex::from_entries(entries);
    manager.save(&index, dir.path()).expect("Failed to save initial index");

    // Start the watcher (uses same cache_dir so it persists to the same location)
    let watcher_manager = SharedIndexManager::with_base_dir(cache_dir.path().to_path_buf());
    let mut watcher = IndexWatcher::new(watcher_manager);
    watcher.start(dir.path()).expect("Failed to start watcher");
    assert!(watcher.is_watching());

    // Allow watcher to fully initialize before making changes
    std::thread::sleep(std::time::Duration::from_millis(500));

    // --- Test 1: Create a new file ---
    eprintln!("[test] Creating new file delta.rs");
    fs::write(dir.path().join("delta.rs"), "fn delta() {}").unwrap();

    // Wait for debounce (500ms) + processing time + margin
    std::thread::sleep(std::time::Duration::from_secs(3));

    // Reload the persisted index and verify it has 4 entries
    let loaded = manager.load(dir.path()).expect("Failed to load index after create");
    let paths_after_create: Vec<String> = (0..loaded.entry_count())
        .map(|i| entry_path(loaded.get_entry(i).unwrap()).to_string())
        .collect();
    eprintln!("[test] After create: {} entries: {:?}", loaded.entry_count(), paths_after_create);

    assert!(
        paths_after_create.iter().any(|p| p.ends_with("delta.rs")),
        "Index should contain delta.rs after creation. Paths: {:?}",
        paths_after_create
    );
    assert_eq!(
        loaded.entry_count(),
        initial_count + 1,
        "Should have {} entries after adding 1 file, got {}. Paths: {:?}",
        initial_count + 1,
        loaded.entry_count(),
        paths_after_create
    );

    // --- Test 2: Delete a file ---
    eprintln!("[test] Deleting beta.rs");
    fs::remove_file(dir.path().join("beta.rs")).unwrap();

    std::thread::sleep(std::time::Duration::from_secs(3));

    let loaded = manager.load(dir.path()).expect("Failed to load index after delete");
    let paths_after_delete: Vec<String> = (0..loaded.entry_count())
        .map(|i| entry_path(loaded.get_entry(i).unwrap()).to_string())
        .collect();
    eprintln!("[test] After delete: {} entries: {:?}", loaded.entry_count(), paths_after_delete);

    assert!(
        !paths_after_delete.iter().any(|p| p.ends_with("beta.rs")),
        "Index should NOT contain beta.rs after deletion. Paths: {:?}",
        paths_after_delete
    );
    assert_eq!(
        loaded.entry_count(),
        initial_count, // 3 original + 1 added - 1 deleted = 3
        "Should have {} entries after delete, got {}. Paths: {:?}",
        initial_count,
        loaded.entry_count(),
        paths_after_delete
    );

    // --- Test 3: Rename a file ---
    eprintln!("[test] Renaming alpha.rs -> omega.rs");
    fs::rename(
        dir.path().join("alpha.rs"),
        dir.path().join("omega.rs"),
    )
    .unwrap();

    std::thread::sleep(std::time::Duration::from_secs(3));

    let loaded = manager.load(dir.path()).expect("Failed to load index after rename");
    let paths_after_rename: Vec<String> = (0..loaded.entry_count())
        .map(|i| entry_path(loaded.get_entry(i).unwrap()).to_string())
        .collect();
    eprintln!("[test] After rename: {} entries: {:?}", loaded.entry_count(), paths_after_rename);

    assert!(
        !paths_after_rename.iter().any(|p| p.ends_with("alpha.rs")),
        "Index should NOT contain alpha.rs after rename. Paths: {:?}",
        paths_after_rename
    );
    assert!(
        paths_after_rename.iter().any(|p| p.ends_with("omega.rs")),
        "Index should contain omega.rs after rename. Paths: {:?}",
        paths_after_rename
    );
    assert_eq!(
        loaded.entry_count(),
        initial_count, // count should be same as after delete (3)
        "Entry count should remain {} after rename, got {}. Paths: {:?}",
        initial_count,
        loaded.entry_count(),
        paths_after_rename
    );

    // Stop watcher
    watcher.stop();
    assert!(!watcher.is_watching());

    eprintln!("[test] IndexWatcher integration test PASSED");
}
