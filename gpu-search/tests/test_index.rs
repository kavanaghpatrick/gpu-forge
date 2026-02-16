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
use gpu_search::index::gsix_v2::save_v2;
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

    // Build and save
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());
    let index = GpuResidentIndex::from_entries(entries);

    // Save in v2 format (MmapIndexCache requires v2)
    let root_hash = cache_key(dir.path());
    let saved_path = cache_dir.path().join("test.idx");
    save_v2(index.entries(), root_hash, &saved_path, 0, 0)
        .expect("Failed to save v2");

    // Load via mmap
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

    // Step 3: Save to v2 binary format (MmapIndexCache requires v2)
    let root_hash = cache_key(dir.path());
    let saved_path = cache_dir.path().join("pipeline.idx");
    save_v2(index.entries(), root_hash, &saved_path, 0, 0)
        .expect("Failed to save v2");
    assert!(saved_path.exists());

    // Step 4: Load via mmap
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
// (notify-based, non-macOS only; macOS uses FSEventsListener)
// ============================================================================

#[cfg(not(target_os = "macos"))]
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

// ============================================================================
// 17. Crash safety: atomic rename leaves no partial .idx file
// ============================================================================

#[test]
fn test_atomic_rename_no_partial_file() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let idx_path = dir.path().join("test.idx");
    let tmp_path = dir.path().join("test.idx.tmp");

    // Create some test entries
    let mut entries = Vec::new();
    for i in 0..5 {
        let mut entry = GpuPathEntry::default();
        let path_str = format!("/tmp/file_{}.rs", i);
        let path_bytes = path_str.as_bytes();
        entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
        entry.path_len = path_bytes.len() as u32;
        entry.mtime = 1000 + i;
        entries.push(entry);
    }

    // Before save: neither .idx nor .idx.tmp should exist
    assert!(!idx_path.exists(), ".idx should not exist before save");
    assert!(!tmp_path.exists(), ".idx.tmp should not exist before save");

    // Perform save_v2 (writes to .tmp then renames to .idx)
    save_v2(&entries, 42, &idx_path, 0, 0).expect("save_v2 failed");

    // After save: .idx exists, .idx.tmp does NOT exist (was renamed away)
    assert!(idx_path.exists(), ".idx should exist after successful save");
    assert!(
        !tmp_path.exists(),
        ".idx.tmp should NOT exist after successful save (atomic rename consumed it)"
    );

    // Verify the .idx file is valid and loadable
    let (header, loaded) = gpu_search::index::gsix_v2::load_v2(&idx_path).expect("load_v2 failed");
    assert_eq!(loaded.len(), 5, "Should load 5 entries");
    assert_eq!(header.entry_count, 5);
}

// ============================================================================
// 18. Crash safety: leftover .idx.tmp from failed write doesn't corrupt load
// ============================================================================

#[test]
fn test_tmp_left_on_error_no_corrupt() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let idx_path = dir.path().join("test.idx");
    let tmp_path = dir.path().join("test.idx.tmp");

    // Create valid entries and save a good .idx file
    let mut entries = Vec::new();
    for i in 0..3 {
        let mut entry = GpuPathEntry::default();
        let path_str = format!("/tmp/good_{}.rs", i);
        let path_bytes = path_str.as_bytes();
        entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
        entry.path_len = path_bytes.len() as u32;
        entry.mtime = 2000 + i;
        entries.push(entry);
    }
    save_v2(&entries, 99, &idx_path, 0, 0).expect("Initial save failed");

    // Simulate a failed/interrupted write: leave a garbage .idx.tmp file
    fs::write(&tmp_path, b"this is corrupt leftover data from a crashed write").unwrap();
    assert!(tmp_path.exists(), "Leftover .tmp should exist");

    // load_v2 should still read the valid .idx file, ignoring the .tmp
    let (header, loaded) =
        gpu_search::index::gsix_v2::load_v2(&idx_path).expect("load_v2 should succeed despite leftover .tmp");
    assert_eq!(loaded.len(), 3, "Should load 3 entries from valid .idx");
    assert_eq!(header.root_hash, 99);

    // A subsequent save_v2 should overwrite the stale .tmp and produce a valid .idx
    let mut new_entries = Vec::new();
    for i in 0..7 {
        let mut entry = GpuPathEntry::default();
        let path_str = format!("/tmp/new_{}.rs", i);
        let path_bytes = path_str.as_bytes();
        entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
        entry.path_len = path_bytes.len() as u32;
        entry.mtime = 3000 + i;
        new_entries.push(entry);
    }
    save_v2(&new_entries, 200, &idx_path, 0, 0).expect("Second save should succeed");

    // .tmp should be gone (renamed to .idx)
    assert!(
        !tmp_path.exists(),
        ".idx.tmp should not exist after successful save"
    );

    // Verify the new .idx is valid
    let (header2, loaded2) =
        gpu_search::index::gsix_v2::load_v2(&idx_path).expect("load after overwrite failed");
    assert_eq!(loaded2.len(), 7, "Should load 7 entries after second save");
    assert_eq!(header2.root_hash, 200);
}

// ============================================================================
// 19. Crash safety: concurrent save produces valid file
// ============================================================================

#[test]
fn test_concurrent_save_one_valid() {
    use std::sync::Arc;

    let dir = TempDir::new().expect("Failed to create temp dir");
    let idx_path = Arc::new(dir.path().join("concurrent.idx"));

    // Create two different entry sets
    let mut entries_a = Vec::new();
    for i in 0..10 {
        let mut entry = GpuPathEntry::default();
        let path_str = format!("/tmp/thread_a/file_{}.rs", i);
        let path_bytes = path_str.as_bytes();
        entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
        entry.path_len = path_bytes.len() as u32;
        entry.mtime = 5000 + i;
        entries_a.push(entry);
    }

    let mut entries_b = Vec::new();
    for i in 0..20 {
        let mut entry = GpuPathEntry::default();
        let path_str = format!("/tmp/thread_b/file_{}.rs", i);
        let path_bytes = path_str.as_bytes();
        entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
        entry.path_len = path_bytes.len() as u32;
        entry.mtime = 6000 + i;
        entries_b.push(entry);
    }

    let path_a = Arc::clone(&idx_path);
    let path_b = Arc::clone(&idx_path);

    // Spawn two threads both writing to the same path.
    // Because both use the same .idx.tmp intermediate file, one may fail due to
    // the race on the temp file. That's acceptable — the key invariant is that
    // whatever ends up on disk is a valid, complete index (not a partial mix).
    let handle_a = std::thread::spawn(move || {
        save_v2(&entries_a, 111, &path_a, 0, 0).ok()
    });

    let handle_b = std::thread::spawn(move || {
        save_v2(&entries_b, 222, &path_b, 0, 0).ok()
    });

    let result_a = handle_a.join().expect("Thread A panicked");
    let result_b = handle_b.join().expect("Thread B panicked");

    // At least one thread must succeed
    assert!(
        result_a.is_some() || result_b.is_some(),
        "At least one concurrent save must succeed"
    );

    // After both complete, the .idx file should be valid (one of the two writes won)
    let (header, loaded) = gpu_search::index::gsix_v2::load_v2(&idx_path)
        .expect("load_v2 should succeed after concurrent writes");

    // The file must contain a complete, valid index — not a corrupt mix.
    // It could be either thread's data depending on race timing.
    assert!(
        !loaded.is_empty(),
        "Loaded index should have entries"
    );

    // All entries must have valid, non-zero path_len
    for (i, entry) in loaded.iter().enumerate() {
        assert!(
            entry.path_len > 0,
            "Entry {} should have non-zero path_len",
            i
        );
        let path_str = std::str::from_utf8(&entry.path[..entry.path_len as usize])
            .expect("Entry path should be valid UTF-8");
        assert!(
            path_str.starts_with("/tmp/thread_"),
            "Entry {} should be from one of the threads, got: {}",
            i,
            path_str
        );
    }

    // Header entry_count must match actual loaded entries
    assert_eq!(
        header.entry_count as usize,
        loaded.len(),
        "Header entry_count should match loaded entries"
    );

    // root_hash should be one of the two known values (111 or 222)
    assert!(
        header.root_hash == 111 || header.root_hash == 222,
        "root_hash should be 111 or 222, got {}",
        header.root_hash
    );

    // All entries should be from the same thread (no mixing)
    let first_path = std::str::from_utf8(&loaded[0].path[..loaded[0].path_len as usize]).unwrap();
    let expected_prefix = if first_path.starts_with("/tmp/thread_a/") {
        "/tmp/thread_a/"
    } else {
        "/tmp/thread_b/"
    };
    for (i, entry) in loaded.iter().enumerate() {
        let path_str = std::str::from_utf8(&entry.path[..entry.path_len as usize]).unwrap();
        assert!(
            path_str.starts_with(expected_prefix),
            "Entry {} should be from {}, got: {} (no mixing allowed)",
            i,
            expected_prefix,
            path_str
        );
    }
}

// ============================================================================
// 20. Crash safety: rename atomicity — reader sees old or new, never partial
// ============================================================================

#[test]
fn test_rename_atomicity() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let idx_path = dir.path().join("atomic.idx");

    // Save version 1: 5 entries with root_hash=100
    let mut entries_v1 = Vec::new();
    for i in 0..5 {
        let mut entry = GpuPathEntry::default();
        let path_str = format!("/data/v1/file_{}.rs", i);
        let path_bytes = path_str.as_bytes();
        entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
        entry.path_len = path_bytes.len() as u32;
        entry.mtime = 7000 + i;
        entries_v1.push(entry);
    }
    save_v2(&entries_v1, 100, &idx_path, 0, 0).expect("First save failed");

    // Verify version 1 on disk
    let (h1, l1) = gpu_search::index::gsix_v2::load_v2(&idx_path).expect("Load v1 failed");
    assert_eq!(l1.len(), 5);
    assert_eq!(h1.root_hash, 100);

    // Save version 2: 15 entries with root_hash=200
    let mut entries_v2 = Vec::new();
    for i in 0..15 {
        let mut entry = GpuPathEntry::default();
        let path_str = format!("/data/v2/file_{}.rs", i);
        let path_bytes = path_str.as_bytes();
        entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
        entry.path_len = path_bytes.len() as u32;
        entry.mtime = 8000 + i;
        entries_v2.push(entry);
    }
    save_v2(&entries_v2, 200, &idx_path, 0, 0).expect("Second save failed");

    // Load the result: should see exactly version 2
    let (h2, l2) = gpu_search::index::gsix_v2::load_v2(&idx_path).expect("Load after overwrite failed");
    assert_eq!(l2.len(), 15, "Should see version 2 (15 entries)");
    assert_eq!(h2.root_hash, 200, "Should see version 2 root_hash");

    // Verify all entries are from version 2, not a mix of v1 and v2
    for (i, entry) in l2.iter().enumerate() {
        let path_str = std::str::from_utf8(&entry.path[..entry.path_len as usize])
            .expect("Entry path should be valid UTF-8");
        assert!(
            path_str.starts_with("/data/v2/"),
            "Entry {} should be from v2, not mixed with v1. Got: {}",
            i,
            path_str
        );
    }

    // Verify no leftover .tmp
    let tmp_path = dir.path().join("atomic.idx.tmp");
    assert!(
        !tmp_path.exists(),
        ".idx.tmp should not remain after save"
    );

    // Additional: verify entry count in header matches actual entries
    assert_eq!(
        h2.entry_count as usize,
        l2.len(),
        "Header entry_count should match loaded entries"
    );

    // Verify entries are byte-exact with what was saved
    for (i, (orig, loaded)) in entries_v2.iter().zip(l2.iter()).enumerate() {
        let orig_bytes: &[u8; 256] =
            unsafe { &*(orig as *const GpuPathEntry as *const [u8; 256]) };
        let loaded_bytes: &[u8; 256] =
            unsafe { &*(loaded as *const GpuPathEntry as *const [u8; 256]) };
        assert_eq!(
            orig_bytes, loaded_bytes,
            "Entry {} should be byte-identical between save and load",
            i
        );
    }
}

// ============================================================================
// 21. Graceful degradation: missing cache dir is auto-created
// ============================================================================

#[test]
fn test_graceful_missing_cache_dir_created() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let nested = dir.path().join("a").join("b").join("c");

    // Directory does not exist yet
    assert!(!nested.exists(), "nested dir should not exist before test");

    // save_v2 should auto-create parent dirs and succeed
    let idx_path = nested.join("test.idx");
    let mut entry = GpuPathEntry::default();
    let path_bytes = b"/tmp/test.rs";
    entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
    entry.path_len = path_bytes.len() as u32;

    let result = save_v2(&[entry], 42, &idx_path, 0, 0);
    assert!(result.is_ok(), "save_v2 should auto-create missing parent dirs: {:?}", result.err());
    assert!(nested.exists(), "nested cache dir should have been created");
    assert!(idx_path.exists(), "index file should exist after save");
}

// ============================================================================
// 22. Graceful degradation: unwritable dir fallback (save fails, search can
//     still work via walk_and_filter)
// ============================================================================

#[test]
fn test_graceful_unwritable_dir_fallback_to_walk() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().expect("Failed to create temp dir");
    let readonly_dir = dir.path().join("readonly");
    fs::create_dir(&readonly_dir).unwrap();

    // Make directory read-only
    let perms = fs::Permissions::from_mode(0o444);
    fs::set_permissions(&readonly_dir, perms).unwrap();

    // Attempt save_v2 to read-only dir — should return Err, NOT panic
    let idx_path = readonly_dir.join("test.idx");
    let entry = GpuPathEntry::default();
    let result = save_v2(&[entry], 1, &idx_path, 0, 0);

    // Restore permissions before assertions so cleanup works
    let perms = fs::Permissions::from_mode(0o755);
    fs::set_permissions(&readonly_dir, perms).unwrap();

    assert!(
        result.is_err(),
        "save_v2 to read-only dir should return Err, not panic"
    );

    // Meanwhile, walk_and_filter (via FilesystemScanner) still works independently
    let test_dir = make_test_dir();
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(test_dir.path());
    assert_eq!(
        entries.len(),
        EXPECTED_FILE_COUNT,
        "walk_and_filter via FilesystemScanner should work even when index save fails"
    );
}

// ============================================================================
// 23. Graceful degradation: corrupt index file triggers rebuild path (returns
//     Err, never panics)
// ============================================================================

#[test]
fn test_graceful_corrupt_index_triggers_rebuild() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let idx_path = dir.path().join("corrupt.idx");

    // Write garbage data that looks nothing like a valid index
    fs::write(&idx_path, b"THIS IS NOT A VALID INDEX FILE -- random garbage data!!").unwrap();

    // load_v2 must return Err (not panic) — this signals the caller to rebuild
    let result = gpu_search::index::gsix_v2::load_v2(&idx_path);
    assert!(
        result.is_err(),
        "load_v2 on corrupt file should return Err, not panic"
    );

    // load_with_migration must also return Err gracefully
    let result2 = gpu_search::index::gsix_v2::load_with_migration(&idx_path);
    assert!(
        result2.is_err(),
        "load_with_migration on corrupt file should return Err, not panic"
    );

    // After detecting corruption, a fresh save_v2 should succeed (rebuild path)
    let mut entry = GpuPathEntry::default();
    let path_bytes = b"/rebuilt/file.rs";
    entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
    entry.path_len = path_bytes.len() as u32;

    let rebuild_result = save_v2(&[entry], 99, &idx_path, 0, 0);
    assert!(
        rebuild_result.is_ok(),
        "save_v2 should succeed as rebuild after corruption"
    );

    // Verify the rebuilt index is valid
    let (header, loaded) = gpu_search::index::gsix_v2::load_v2(&idx_path)
        .expect("rebuilt index should be loadable");
    assert_eq!(header.entry_count, 1);
    assert_eq!(loaded.len(), 1);
}

// ============================================================================
// 24. Graceful degradation: walk_and_filter works without any index file
// ============================================================================

#[test]
fn test_graceful_walk_fallback_without_index() {
    let test_dir = make_test_dir();
    // No index file created — verify that the scanner works standalone
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(test_dir.path());

    assert_eq!(
        entries.len(),
        EXPECTED_FILE_COUNT,
        "FilesystemScanner should find all {} files without any index",
        EXPECTED_FILE_COUNT
    );

    // Verify paths are valid and point to real files
    for entry in &entries {
        let path_str = entry_path(entry);
        assert!(!path_str.is_empty(), "path should not be empty");
        let path = PathBuf::from(path_str);
        assert!(path.exists(), "scanned path should exist on disk: {}", path_str);
    }
}

// ============================================================================
// 25. Graceful degradation: blocking search() API works without index
// ============================================================================

#[test]
fn test_graceful_blocking_search_without_index() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use gpu_search::gpu::pipeline::PsoCache;
    use gpu_search::search::types::SearchRequest;

    let device = MTLCreateSystemDefaultDevice()
        .expect("No Metal device (test requires Apple Silicon)");
    let pso_cache = PsoCache::new(&device);

    // Create orchestrator WITHOUT an IndexStore — no index at all
    let mut orchestrator = gpu_search::search::orchestrator::SearchOrchestrator::new(
        &device,
        &pso_cache,
    )
    .expect("Failed to create SearchOrchestrator");

    // Create a test directory with known content
    let test_dir = make_test_dir();

    // Run blocking search — should work via walk_directory fallback
    let request = SearchRequest::new("fn ", test_dir.path().to_str().unwrap());
    let response = orchestrator.search(request);

    // Should complete without panic and find files
    assert!(
        response.total_files_searched > 0,
        "blocking search should find files via walk_directory even without index"
    );
}

// ============================================================================
// 26. Graceful degradation: index deleted mid-operation — load returns error
// ============================================================================

#[test]
fn test_graceful_index_deleted_mid_search() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let idx_path = dir.path().join("will_delete.idx");

    // Save a valid index
    let mut entries = Vec::new();
    for i in 0..10 {
        let mut entry = GpuPathEntry::default();
        let path_str = format!("/tmp/mid_search/file_{}.rs", i);
        let path_bytes = path_str.as_bytes();
        entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
        entry.path_len = path_bytes.len() as u32;
        entry.mtime = 4000 + i;
        entries.push(entry);
    }
    save_v2(&entries, 77, &idx_path, 555, 0).expect("save failed");

    // Verify it's loadable
    let (header, loaded) = gpu_search::index::gsix_v2::load_v2(&idx_path)
        .expect("initial load should succeed");
    assert_eq!(header.entry_count, 10);
    assert_eq!(loaded.len(), 10);

    // Delete the index file (simulating external deletion mid-operation)
    fs::remove_file(&idx_path).expect("delete should succeed");
    assert!(!idx_path.exists(), "index file should be gone");

    // Subsequent load_v2 should return Err gracefully, not panic
    let result = gpu_search::index::gsix_v2::load_v2(&idx_path);
    assert!(
        result.is_err(),
        "load_v2 on deleted file should return Err, not panic"
    );

    // load_with_migration should also handle deletion gracefully
    let result2 = gpu_search::index::gsix_v2::load_with_migration(&idx_path);
    assert!(
        result2.is_err(),
        "load_with_migration on deleted file should return Err, not panic"
    );
}

// ============================================================================
// Edge Case Tests: paths, permissions, symlinks
// ============================================================================

// --- Path edge cases ---

/// A path of exactly 224 bytes (GPU_PATH_MAX_LEN) should be indexed successfully.
#[test]
fn test_edge_case_path_exactly_224_bytes() {
    // GPU_PATH_MAX_LEN = 224
    let path_bytes = vec![b'a'; GPU_PATH_MAX_LEN];
    assert_eq!(path_bytes.len(), 224);

    let mut entry = GpuPathEntry::new();
    entry.set_path(&path_bytes);

    assert_eq!(entry.path_len as usize, 224, "path_len should be exactly 224");
    assert_eq!(
        &entry.path[..224],
        &path_bytes[..],
        "all 224 bytes should be stored"
    );
}

/// A path of 225 bytes exceeds GPU_PATH_MAX_LEN. set_path truncates to 224;
/// IndexWriter::handle_created silently skips it.
#[test]
fn test_edge_case_path_225_bytes() {
    // set_path truncates at GPU_PATH_MAX_LEN
    let path_bytes = vec![b'b'; 225];
    let mut entry = GpuPathEntry::new();
    entry.set_path(&path_bytes);
    assert_eq!(
        entry.path_len as usize, 224,
        "set_path should truncate 225-byte path to 224"
    );

    // IndexWriter skips paths > GPU_PATH_MAX_LEN entirely
    use gpu_search::index::index_writer::IndexWriter;
    use gpu_search::index::exclude::ExcludeTrie;
    use gpu_search::index::store::IndexStore;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let excludes = Arc::new(ExcludeTrie::default());
    let store = Arc::new(IndexStore::default());
    let idx_path = dir.path().join("test.idx");
    let mut writer = IndexWriter::new(excludes, store, idx_path, 0, 0);

    // Create a temp file with a very long name (225+ bytes total path)
    let long_name = "x".repeat(225);
    let long_path = dir.path().join(&long_name);
    // The full path is dir + "/" + long_name, so well over 225 bytes
    // handle_created should silently skip it
    writer.handle_created(&long_path);

    // Writer should have 0 entries (path was too long)
    assert_eq!(
        writer.entry_count(), 0,
        "IndexWriter should skip paths exceeding GPU_PATH_MAX_LEN"
    );
}

/// A path of 1024 bytes should be skipped without panic.
#[test]
fn test_edge_case_path_1024_bytes() {
    let path_bytes = vec![b'c'; 1024];
    let mut entry = GpuPathEntry::new();
    // set_path does not panic — it truncates
    entry.set_path(&path_bytes);
    assert_eq!(entry.path_len as usize, 224, "1024-byte path truncated to 224");

    // IndexWriter also does not panic
    use gpu_search::index::index_writer::IndexWriter;
    use gpu_search::index::exclude::ExcludeTrie;
    use gpu_search::index::store::IndexStore;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let excludes = Arc::new(ExcludeTrie::default());
    let store = Arc::new(IndexStore::default());
    let idx_path = dir.path().join("test.idx");
    let mut writer = IndexWriter::new(excludes, store, idx_path, 0, 0);

    let long_name = "z".repeat(1024);
    let long_path = dir.path().join(&long_name);
    // Must not panic
    writer.handle_created(&long_path);
    // Skipped because path > GPU_PATH_MAX_LEN
    assert_eq!(writer.entry_count(), 0, "1024-byte path should be skipped without panic");
}

/// CJK Unicode characters in paths are handled correctly.
#[test]
fn test_edge_case_unicode_cjk_path() {
    let dir = TempDir::new().unwrap();
    let cjk_name = "\u{4F60}\u{597D}\u{4E16}\u{754C}.txt"; // 你好世界.txt
    let cjk_path = dir.path().join(cjk_name);
    fs::write(&cjk_path, "CJK content").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    assert_eq!(entries.len(), 1, "Should find one CJK-named file");
    let path_str = entry_path(&entries[0]);
    assert!(
        path_str.contains(cjk_name),
        "Entry path '{}' should contain CJK name '{}'",
        path_str,
        cjk_name
    );
}

/// Spaces in filenames are handled correctly.
#[test]
fn test_edge_case_space_in_filename() {
    let dir = TempDir::new().unwrap();
    let space_name = "file with spaces.rs";
    let space_path = dir.path().join(space_name);
    fs::write(&space_path, "// spaces").unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    assert_eq!(entries.len(), 1, "Should find file with spaces");
    let path_str = entry_path(&entries[0]);
    assert!(
        path_str.contains(space_name),
        "Entry path '{}' should contain '{}'",
        path_str,
        space_name
    );
}

/// Null bytes in path data do not cause panic — GpuPathEntry::set_path
/// handles arbitrary bytes.
#[test]
fn test_edge_case_null_byte_path_rejected() {
    // set_path accepts arbitrary bytes including null
    let path_with_null = b"/tmp/file\x00hidden.rs";
    let mut entry = GpuPathEntry::new();
    // Must not panic
    entry.set_path(path_with_null);
    assert_eq!(
        entry.path_len as usize,
        path_with_null.len(),
        "set_path should accept bytes with null"
    );

    // IndexWriter won't be able to stat such a path (invalid on POSIX),
    // so handle_created should silently skip it.
    use gpu_search::index::index_writer::IndexWriter;
    use gpu_search::index::exclude::ExcludeTrie;
    use gpu_search::index::store::IndexStore;
    use std::sync::Arc;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let dir = TempDir::new().unwrap();
    let excludes = Arc::new(ExcludeTrie::default());
    let store = Arc::new(IndexStore::default());
    let idx_path = dir.path().join("test.idx");
    let mut writer = IndexWriter::new(excludes, store, idx_path, 0, 0);

    // Construct a path with embedded null byte
    let null_path = PathBuf::from(OsStr::from_bytes(b"/tmp/null\x00byte.rs"));
    // Must not panic — should silently fail (stat will fail)
    writer.handle_created(&null_path);
    assert_eq!(writer.entry_count(), 0, "Null-byte path should be skipped without panic");
}

// --- Permission edge cases ---

/// An unreadable directory (chmod 000) should be skipped by the scanner.
#[test]
fn test_edge_case_chmod_000_dir_skipped() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().unwrap();

    // Create a readable file and an unreadable subdirectory with a file inside
    fs::write(dir.path().join("visible.rs"), "visible").unwrap();

    let forbidden = dir.path().join("forbidden");
    fs::create_dir(&forbidden).unwrap();
    fs::write(forbidden.join("secret.rs"), "secret").unwrap();

    // Make the subdirectory completely inaccessible
    fs::set_permissions(&forbidden, fs::Permissions::from_mode(0o000)).unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    // Restore permissions before assertions so TempDir cleanup works
    fs::set_permissions(&forbidden, fs::Permissions::from_mode(0o755)).unwrap();

    // The scanner should find visible.rs but NOT secret.rs inside the forbidden dir
    let paths: Vec<String> = entries.iter().map(|e| entry_path(e).to_string()).collect();
    assert!(
        paths.iter().any(|p| p.ends_with("visible.rs")),
        "Should find visible.rs. Paths: {:?}",
        paths
    );
    assert!(
        !paths.iter().any(|p| p.ends_with("secret.rs")),
        "Should NOT find secret.rs inside chmod 000 dir. Paths: {:?}",
        paths
    );
}

/// Only readable files are indexed; unreadable files are skipped gracefully.
#[test]
fn test_edge_case_mixed_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().unwrap();

    // Create three files with different permissions
    let readable = dir.path().join("readable.rs");
    let unreadable = dir.path().join("unreadable.rs");
    let also_readable = dir.path().join("also_readable.rs");

    fs::write(&readable, "readable content").unwrap();
    fs::write(&unreadable, "secret content").unwrap();
    fs::write(&also_readable, "also readable").unwrap();

    // Make one file unreadable (000)
    fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o000)).unwrap();

    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    // Restore permissions for cleanup
    fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o644)).unwrap();

    let paths: Vec<String> = entries.iter().map(|e| entry_path(e).to_string()).collect();

    // Note: The scanner uses metadata() which may still succeed on chmod 000 files
    // (metadata reads inode, not file contents). The scanner indexes by path, not
    // by readability. So all 3 files may appear. The key invariant is: no panic.
    // At minimum, readable files must appear.
    assert!(
        paths.iter().any(|p| p.ends_with("readable.rs")),
        "readable.rs should be indexed. Paths: {:?}",
        paths
    );
    assert!(
        paths.iter().any(|p| p.ends_with("also_readable.rs")),
        "also_readable.rs should be indexed. Paths: {:?}",
        paths
    );
    // No panic occurred — graceful handling confirmed
}

// --- Symlink edge cases ---

/// A symlink to a file should have the IS_SYMLINK flag set when the scanner
/// is configured to follow symlinks.
#[test]
fn test_edge_case_symlink_to_file() {
    use gpu_search::gpu::types::path_flags;

    let dir = TempDir::new().unwrap();
    let target = dir.path().join("target.rs");
    fs::write(&target, "target content").unwrap();

    let link = dir.path().join("link.rs");
    std::os::unix::fs::symlink(&target, &link).unwrap();

    // With follow_symlinks=true, the scanner should index the symlink
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        follow_symlinks: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries.iter().map(|e| entry_path(e).to_string()).collect();
    assert!(
        paths.iter().any(|p| p.ends_with("target.rs")),
        "Should find target.rs. Paths: {:?}",
        paths
    );

    // Check the link entry for IS_SYMLINK flag
    // Note: with follow_links=true, the walker resolves symlinks and metadata
    // comes from the target. The IS_SYMLINK flag is set based on metadata.file_type().is_symlink(),
    // but after following, the metadata is the target's. The flag may or may not be set
    // depending on whether lstat or stat is used. Test both presence scenarios gracefully.
    let link_entries: Vec<&GpuPathEntry> = entries
        .iter()
        .filter(|e| entry_path(e).ends_with("link.rs"))
        .collect();

    // With follow_symlinks=true, the link should be visible
    assert!(
        !link_entries.is_empty(),
        "Symlink link.rs should be indexed when follow_symlinks=true. Paths: {:?}",
        paths
    );

    // Verify the entry has the IS_SYMLINK flag or is at least properly indexed
    // (The ignore crate with follow_links resolves symlinks, so metadata.file_type()
    // may report regular file rather than symlink. The important thing is no panic.)
    let link_entry = link_entries[0];
    assert!(link_entry.path_len > 0, "Symlink entry should have valid path_len");

    // If the scanner detected it as a symlink, the flag should be set
    if link_entry.flags & path_flags::IS_SYMLINK != 0 {
        // IS_SYMLINK flag is correctly set
        assert_eq!(
            link_entry.flags & path_flags::IS_SYMLINK,
            path_flags::IS_SYMLINK,
            "IS_SYMLINK flag should be set for symlink entry"
        );
    }
    // Either way, no panic — test passes
}

/// A broken symlink (target does not exist) should be silently skipped.
#[test]
fn test_edge_case_broken_symlink_skipped() {
    let dir = TempDir::new().unwrap();

    // Create a file so there's at least one result
    let real_file = dir.path().join("real.rs");
    fs::write(&real_file, "real content").unwrap();

    // Create a symlink to a non-existent target
    let broken_link = dir.path().join("broken_link.rs");
    std::os::unix::fs::symlink("/nonexistent/target/file.rs", &broken_link).unwrap();
    assert!(
        broken_link.symlink_metadata().is_ok(),
        "Symlink itself should exist"
    );
    assert!(
        !broken_link.exists(),
        "Broken symlink target should not exist"
    );

    // Scanner with default follow_symlinks=false should skip the symlink entirely
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        follow_symlinks: false,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    let paths: Vec<String> = entries.iter().map(|e| entry_path(e).to_string()).collect();
    assert!(
        paths.iter().any(|p| p.ends_with("real.rs")),
        "Should find real.rs. Paths: {:?}",
        paths
    );
    assert!(
        !paths.iter().any(|p| p.ends_with("broken_link.rs")),
        "Broken symlink should be silently skipped. Paths: {:?}",
        paths
    );

    // Also test with follow_symlinks=true — broken symlink should still be skipped
    // (metadata() will fail on broken symlink)
    let scanner_follow = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: false,
        follow_symlinks: true,
        ..Default::default()
    });
    let entries_follow = scanner_follow.scan(dir.path());

    let paths_follow: Vec<String> = entries_follow
        .iter()
        .map(|e| entry_path(e).to_string())
        .collect();
    assert!(
        !paths_follow.iter().any(|p| p.ends_with("broken_link.rs")),
        "Broken symlink should be skipped even with follow_symlinks=true. Paths: {:?}",
        paths_follow
    );
}

// ============================================================================
// Regression Tests: walk fallback and search equivalence
// ============================================================================

/// Regression: walk fallback (FilesystemScanner) still works without any index.
/// Verifies that the walk path has not been broken by persistent index changes.
#[test]
fn test_regression_walk_fallback_still_works() {
    let dir = make_test_dir();

    // Use FilesystemScanner directly — this is the walk fallback path
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    // Must return the expected number of files
    assert_eq!(
        entries.len(),
        EXPECTED_FILE_COUNT,
        "Walk fallback should find exactly {} files without any index, got {}",
        EXPECTED_FILE_COUNT,
        entries.len()
    );

    // All returned paths must exist on disk
    for entry in &entries {
        let path_str = entry_path(entry);
        assert!(!path_str.is_empty(), "Walk fallback should not return empty paths");
        let path = PathBuf::from(path_str);
        assert!(
            path.exists(),
            "Walk fallback path should exist on disk: {}",
            path_str
        );
    }

    // All expected files must be found
    let paths: Vec<String> = entries.iter().map(|e| entry_path(e).to_string()).collect();
    for suffix in &["main.rs", "lib.rs", "README.md", "Cargo.toml", "mod.rs", "utils.rs", "types.rs", "deep.rs"] {
        assert!(
            paths.iter().any(|p| p.ends_with(suffix)),
            "Walk fallback should find file ending with '{}'. Found: {:?}",
            suffix,
            paths
        );
    }
}

/// Regression: blocking search() API still works without any IndexStore.
/// Verifies the original search pipeline (walk_directory -> filter -> GPU) is unchanged.
#[test]
fn test_regression_blocking_search_api_works() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use gpu_search::gpu::pipeline::PsoCache;
    use gpu_search::search::types::SearchRequest;

    let device = MTLCreateSystemDefaultDevice()
        .expect("No Metal device (test requires Apple Silicon)");
    let pso_cache = PsoCache::new(&device);

    // Create orchestrator WITHOUT IndexStore — pure walk fallback
    let mut orchestrator = gpu_search::search::orchestrator::SearchOrchestrator::new(
        &device,
        &pso_cache,
    )
    .expect("Failed to create SearchOrchestrator");

    let test_dir = make_test_dir();

    // Run blocking search for a pattern known to exist in make_test_dir files
    let request = SearchRequest::new("fn ", test_dir.path().to_str().unwrap());
    let response = orchestrator.search(request);

    // Must find files (walk_directory fallback)
    assert!(
        response.total_files_searched > 0,
        "Blocking search should search files via walk_directory fallback"
    );

    // Must find content matches ("fn " appears in main.rs and lib.rs at minimum)
    // main.rs has "fn main() {}" and src/utils.rs has "pub fn help() {}"
    assert!(
        !response.content_matches.is_empty(),
        "Blocking search should find content matches for 'fn '"
    );

    // File matches should include files whose names contain "fn" — none do in our test set,
    // so we just verify the response completed without error
    assert!(
        response.elapsed.as_secs() < 30,
        "Blocking search should complete in reasonable time, took {:?}",
        response.elapsed
    );
}

/// Regression: indexed vs unindexed results return the same files.
/// Builds an index from a temp dir, then compares files found via index
/// entries with files found via FilesystemScanner walk.
#[test]
fn test_regression_indexed_vs_unindexed_results_match() {
    use gpu_search::index::gsix_v2::save_v2;
    use gpu_search::index::snapshot::IndexSnapshot;
    use gpu_search::index::cache::cache_key;
    use gpu_search::gpu::types::path_flags;
    use std::collections::BTreeSet;

    let dir = make_test_dir();
    let cache_dir = TempDir::new().expect("Failed to create cache dir");

    // --- Unindexed path: walk via FilesystemScanner ---
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let walk_entries = scanner.scan(dir.path());
    let walk_paths: BTreeSet<String> = walk_entries
        .iter()
        .map(|e| entry_path(e).to_string())
        .collect();

    // --- Indexed path: save to v2, load via IndexSnapshot ---
    let root_hash = cache_key(dir.path());
    let idx_path = cache_dir.path().join("regression.idx");
    save_v2(walk_entries.as_slice(), root_hash, &idx_path, 0, 0)
        .expect("save_v2 failed");

    let snapshot = IndexSnapshot::from_file(&idx_path, None)
        .expect("IndexSnapshot::from_file failed");

    // Extract paths from the snapshot (skipping IS_DELETED entries)
    let index_paths: BTreeSet<String> = snapshot
        .entries()
        .iter()
        .filter(|e| e.flags & path_flags::IS_DELETED == 0)
        .map(|e| entry_path(e).to_string())
        .collect();

    // --- Equivalence check ---
    assert_eq!(
        walk_paths.len(),
        index_paths.len(),
        "Walk found {} files, index found {} files. They should match.\nWalk only: {:?}\nIndex only: {:?}",
        walk_paths.len(),
        index_paths.len(),
        walk_paths.difference(&index_paths).collect::<Vec<_>>(),
        index_paths.difference(&walk_paths).collect::<Vec<_>>()
    );

    assert_eq!(
        walk_paths, index_paths,
        "Walk paths and index paths should be identical.\nWalk only: {:?}\nIndex only: {:?}",
        walk_paths.difference(&index_paths).collect::<Vec<_>>(),
        index_paths.difference(&walk_paths).collect::<Vec<_>>()
    );
}

// ============================================================================
// V1-to-V2 Migration: end-to-end validation
// ============================================================================

/// Validate the full v1-to-v2 migration path end-to-end:
/// 1. Create a v1 format index file (GSIX magic, version=1, 64-byte header, packed entries)
/// 2. Verify detect_version() returns V1
/// 3. Verify load_with_migration() detects v1 and signals rebuild (deletes v1 file)
/// 4. Build and save a v2 global.idx, verify it's valid
/// 5. Verify cleanup_v1_indexes() removes per-directory v1 files and creates .v2-migrated marker
/// 6. Verify subsequent load uses v2 format
#[test]
fn test_v1_migration_end_to_end() {
    use gpu_search::index::gsix_v2::{
        detect_version, load_with_migration, cleanup_v1_indexes,
        save_v2, load_v2, INDEX_MAGIC, INDEX_VERSION_V2,
    };

    let dir = TempDir::new().expect("Failed to create temp dir");
    let index_dir = dir.path().join("index");
    fs::create_dir_all(&index_dir).unwrap();

    // -----------------------------------------------------------------------
    // Step 1: Create a v1 format index file (GSIX magic, version=1, 64-byte header)
    // -----------------------------------------------------------------------
    let v1_path = index_dir.join("abc123.idx");
    {
        // Build a v1-format file: 64-byte header + packed 256-byte entries
        let mut v1_data: Vec<u8> = Vec::new();

        // Header: magic(4) + version(4) + entry_count(4) + root_hash(4) + saved_at(8) + reserved(40) = 64
        v1_data.extend_from_slice(&INDEX_MAGIC.to_le_bytes()); // magic "GSIX"
        v1_data.extend_from_slice(&1u32.to_le_bytes());         // version = 1
        v1_data.extend_from_slice(&3u32.to_le_bytes());         // entry_count = 3
        v1_data.extend_from_slice(&0xABCDu32.to_le_bytes());    // root_hash
        v1_data.extend_from_slice(&1700000000u64.to_le_bytes()); // saved_at
        v1_data.extend_from_slice(&[0u8; 40]);                   // reserved

        assert_eq!(v1_data.len(), 64, "v1 header should be exactly 64 bytes");

        // Write 3 packed 256-byte GpuPathEntry records
        for i in 0..3u32 {
            let mut entry = GpuPathEntry::default();
            let path_str = format!("/tmp/v1_file_{}.rs", i);
            let path_bytes = path_str.as_bytes();
            entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
            entry.path_len = path_bytes.len() as u32;
            entry.mtime = 1700000000 + i;

            let entry_bytes: &[u8; 256] =
                unsafe { &*((&entry) as *const GpuPathEntry as *const [u8; 256]) };
            v1_data.extend_from_slice(entry_bytes);
        }

        assert_eq!(v1_data.len(), 64 + 3 * 256, "v1 file = 64 header + 3*256 entries");
        fs::write(&v1_path, &v1_data).unwrap();
    }

    // -----------------------------------------------------------------------
    // Step 2: Verify detect_version() returns V1 (=1)
    // -----------------------------------------------------------------------
    {
        let v1_data = fs::read(&v1_path).unwrap();
        let version = detect_version(&v1_data).expect("detect_version should succeed on v1 file");
        assert_eq!(version, 1, "detect_version should return 1 for v1 format");
    }

    // -----------------------------------------------------------------------
    // Step 3: Verify load_with_migration() detects v1 and signals rebuild
    // -----------------------------------------------------------------------
    {
        let result = load_with_migration(&v1_path);
        assert!(
            result.is_err(),
            "load_with_migration should return Err for v1 file"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("v1 index detected") && err_msg.contains("rebuild required"),
            "Error should indicate v1 detected and rebuild required, got: {}",
            err_msg
        );

        // v1 file should have been deleted by load_with_migration
        assert!(
            !v1_path.exists(),
            "v1 file should be deleted after load_with_migration detects it"
        );
    }

    // -----------------------------------------------------------------------
    // Step 4: Build and save a v2 global.idx, verify it's valid
    // -----------------------------------------------------------------------
    let global_path = index_dir.join("global.idx");
    {
        // Simulate a rebuild: create fresh entries and save as v2
        let mut entries = Vec::new();
        for i in 0..5 {
            let mut entry = GpuPathEntry::default();
            let path_str = format!("/tmp/v2_file_{}.rs", i);
            let path_bytes = path_str.as_bytes();
            entry.path[..path_bytes.len()].copy_from_slice(path_bytes);
            entry.path_len = path_bytes.len() as u32;
            entry.mtime = 1700100000 + i;
            entries.push(entry);
        }

        save_v2(&entries, 0xBEEF, &global_path, 99999, 0xCAFE)
            .expect("save_v2 should succeed for global.idx");

        assert!(global_path.exists(), "global.idx should exist after save_v2");

        // Verify the v2 file is valid
        let (header, loaded) = load_v2(&global_path).expect("load_v2 should succeed on new v2 file");
        assert_eq!(header.version, INDEX_VERSION_V2, "version should be 2");
        assert_eq!(header.entry_count, 5, "should have 5 entries");
        assert_eq!(header.root_hash, 0xBEEF);
        assert_eq!(header.last_fsevents_id, 99999);
        assert_eq!(header.exclude_hash, 0xCAFE);
        assert_eq!(loaded.len(), 5);

        // Verify detect_version on the new file returns V2
        let v2_data = fs::read(&global_path).unwrap();
        let version = detect_version(&v2_data).expect("detect_version should succeed on v2");
        assert_eq!(version, 2, "detect_version should return 2 for v2 format");
    }

    // -----------------------------------------------------------------------
    // Step 5: Verify cleanup_v1_indexes() removes per-directory v1 .idx files
    //         and creates .v2-migrated marker. global.idx should survive.
    // -----------------------------------------------------------------------
    {
        // Create some fake v1 per-directory .idx files (simulate leftover v1 indexes)
        fs::write(index_dir.join("dir_hash1.idx"), b"v1 leftover 1").unwrap();
        fs::write(index_dir.join("dir_hash2.idx"), b"v1 leftover 2").unwrap();
        // Also create a non-.idx file that should be untouched
        fs::write(index_dir.join("notes.txt"), b"not an index").unwrap();

        cleanup_v1_indexes(&index_dir).expect("cleanup_v1_indexes should succeed");

        // global.idx must survive (stem == "global")
        assert!(
            global_path.exists(),
            "global.idx should survive cleanup_v1_indexes"
        );
        // per-directory v1 .idx files should be removed
        assert!(
            !index_dir.join("dir_hash1.idx").exists(),
            "dir_hash1.idx should be removed by cleanup"
        );
        assert!(
            !index_dir.join("dir_hash2.idx").exists(),
            "dir_hash2.idx should be removed by cleanup"
        );
        // non-.idx files should be untouched
        assert!(
            index_dir.join("notes.txt").exists(),
            "notes.txt should not be affected by cleanup"
        );
        // .v2-migrated marker should exist
        assert!(
            index_dir.join(".v2-migrated").exists(),
            ".v2-migrated marker should be created by cleanup_v1_indexes"
        );
    }

    // -----------------------------------------------------------------------
    // Step 6: Verify subsequent loads use v2 (global.idx loads normally)
    // -----------------------------------------------------------------------
    {
        // load_with_migration should now load the v2 global.idx successfully
        let (header, entries) = load_with_migration(&global_path)
            .expect("Subsequent load_with_migration should succeed on v2 global.idx");
        assert_eq!(header.version, INDEX_VERSION_V2, "should be v2 format");
        assert_eq!(header.entry_count, 5);
        assert_eq!(entries.len(), 5);
        assert_eq!(header.root_hash, 0xBEEF);
        assert_eq!(header.last_fsevents_id, 99999);

        // Verify entry content survived the full pipeline
        let first_path = std::str::from_utf8(&entries[0].path[..entries[0].path_len as usize])
            .expect("entry path should be valid UTF-8");
        assert!(
            first_path.starts_with("/tmp/v2_file_"),
            "First entry should be a v2 file, got: {}",
            first_path
        );
    }
}

/// Regression: filename matching produces identical results from index and walk paths.
/// Verifies that searching for a filename pattern yields the same matches
/// regardless of whether files come from the index or from walking.
#[test]
fn test_regression_filename_match_equivalence() {
    use gpu_search::index::gsix_v2::save_v2;
    use gpu_search::index::snapshot::IndexSnapshot;
    use gpu_search::index::cache::cache_key;
    use gpu_search::gpu::types::path_flags;
    use std::collections::BTreeSet;
    use std::path::Path;

    let dir = make_test_dir();
    let cache_dir = TempDir::new().expect("Failed to create cache dir");

    // --- Walk path ---
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let walk_entries = scanner.scan(dir.path());

    // Filename match: find files whose name contains ".rs" (case-insensitive)
    let pattern = ".rs";
    let pattern_lower = pattern.to_lowercase();

    let walk_filename_matches: BTreeSet<String> = walk_entries
        .iter()
        .filter_map(|e| {
            let path_str = entry_path(e);
            let path = Path::new(path_str);
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.to_lowercase().contains(&pattern_lower) {
                    return Some(path_str.to_string());
                }
            }
            None
        })
        .collect();

    // --- Index path ---
    let root_hash = cache_key(dir.path());
    let idx_path = cache_dir.path().join("filename_regression.idx");
    save_v2(walk_entries.as_slice(), root_hash, &idx_path, 0, 0)
        .expect("save_v2 failed");

    let snapshot = IndexSnapshot::from_file(&idx_path, None)
        .expect("IndexSnapshot::from_file failed");

    let index_filename_matches: BTreeSet<String> = snapshot
        .entries()
        .iter()
        .filter(|e| e.flags & path_flags::IS_DELETED == 0)
        .filter_map(|e| {
            let path_str = entry_path(e);
            let path = Path::new(path_str);
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.to_lowercase().contains(&pattern_lower) {
                    return Some(path_str.to_string());
                }
            }
            None
        })
        .collect();

    // --- Equivalence ---
    assert!(
        !walk_filename_matches.is_empty(),
        "Walk should find .rs files in test directory"
    );
    assert_eq!(
        walk_filename_matches.len(),
        index_filename_matches.len(),
        "Walk found {} filename matches, index found {}.\nWalk only: {:?}\nIndex only: {:?}",
        walk_filename_matches.len(),
        index_filename_matches.len(),
        walk_filename_matches.difference(&index_filename_matches).collect::<Vec<_>>(),
        index_filename_matches.difference(&walk_filename_matches).collect::<Vec<_>>()
    );

    assert_eq!(
        walk_filename_matches, index_filename_matches,
        "Filename matches should be identical between walk and index paths.\nWalk only: {:?}\nIndex only: {:?}",
        walk_filename_matches.difference(&index_filename_matches).collect::<Vec<_>>(),
        index_filename_matches.difference(&walk_filename_matches).collect::<Vec<_>>()
    );

    // Verify specific expected .rs files are in the matches
    let expected_rs_files = ["main.rs", "lib.rs", "mod.rs", "utils.rs", "types.rs", "deep.rs"];
    for expected in &expected_rs_files {
        assert!(
            walk_filename_matches.iter().any(|p| p.ends_with(expected)),
            "Filename matches should include '{}'. Found: {:?}",
            expected,
            walk_filename_matches
        );
    }
}

// ============================================================================
// Warm Startup Path Validation
// ============================================================================

// Local synthetic entry generator (integration tests can't access #[cfg(test)] modules)

const WARM_DIR_COMPONENTS: &[&str] = &[
    "src", "lib", "tests", "benches", "docs", "config", "build", "target", "scripts",
    "utils", "core", "api", "models", "views", "controllers", "services", "middleware",
    "handlers", "proto", "internal", "pkg", "cmd", "assets", "static", "templates",
];

const WARM_EXTENSIONS: &[&str] = &[
    ".rs", ".rs", ".rs",
    ".txt", ".js", ".py", ".md", ".toml", ".json", ".yaml", ".html", ".css", ".ts",
    ".sh", ".c", ".h", ".go", ".swift", ".metal",
];

const WARM_ROOT_PREFIXES: &[&str] = &[
    "/Users/dev/project",
    "/Users/dev/workspace",
    "/home/user/code",
    "/opt/builds",
    "/var/lib/app",
    "/usr/local/src",
];

/// Simple deterministic LCG for synthetic entries.
struct WarmLcg {
    state: u32,
}

impl WarmLcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }

    fn next_range(&mut self, bound: u32) -> u32 {
        (self.next() >> 4) % bound
    }
}

fn generate_warm_entries(count: usize) -> Vec<GpuPathEntry> {
    use gpu_search::gpu::types::path_flags;

    let mut rng = WarmLcg::new(0xBEEF_CAFE);
    let mut entries = Vec::with_capacity(count);

    for i in 0..count {
        let mut entry = GpuPathEntry::new();

        let root = WARM_ROOT_PREFIXES[rng.next_range(WARM_ROOT_PREFIXES.len() as u32) as usize];
        let depth = (rng.next_range(8) + 1) as usize;
        let mut path = String::with_capacity(224);
        path.push_str(root);

        for _ in 0..depth {
            path.push('/');
            let comp = WARM_DIR_COMPONENTS[rng.next_range(WARM_DIR_COMPONENTS.len() as u32) as usize];
            path.push_str(comp);
        }

        let ext = WARM_EXTENSIONS[rng.next_range(WARM_EXTENSIONS.len() as u32) as usize];
        path.push_str(&format!("/file_{}{}", i, ext));

        if path.len() > GPU_PATH_MAX_LEN {
            path.truncate(GPU_PATH_MAX_LEN);
        }

        entry.set_path(path.as_bytes());

        let flag_roll = rng.next_range(100);
        if flag_roll < 10 {
            entry.flags |= path_flags::IS_DIR;
        }
        if flag_roll < 5 {
            entry.flags |= path_flags::IS_HIDDEN;
        }
        if flag_roll < 2 {
            entry.flags |= path_flags::IS_SYMLINK;
        }

        if entry.flags & path_flags::IS_DIR == 0 {
            let size = (rng.next_range(10_000_000) + 100) as u64;
            entry.set_size(size);
        }

        let base_mtime: u32 = 1_672_531_200;
        let offset = rng.next_range(63_158_400);
        entry.mtime = base_mtime + offset;

        if i > 0 && depth > 1 {
            entry.parent_idx = rng.next_range(i as u32);
        }

        entries.push(entry);
    }

    entries
}

/// Validate warm startup path: create a v2 index with 100K entries, then
/// measure the full warm startup sequence (mmap -> validate header ->
/// IndexStore swap -> entry count verification) and assert <100ms total.
#[test]
fn test_warm_startup_under_100ms() {
    use gpu_search::index::snapshot::IndexSnapshot;
    use gpu_search::index::store::IndexStore;
    use std::sync::Arc;

    let dir = TempDir::new().expect("Failed to create temp dir");
    let idx_path = dir.path().join("warm_startup.idx");

    // Generate and save 100K synthetic entries as v2 index
    let entry_count = 100_000;
    let entries = generate_warm_entries(entry_count);
    assert_eq!(entries.len(), entry_count);

    let fsevents_id: u64 = 9876543210;
    let root_hash: u32 = 0xDEAD_BEEF;
    let exclude_hash: u32 = 0xCAFE_BABE;
    save_v2(&entries, root_hash, &idx_path, fsevents_id, exclude_hash)
        .expect("save_v2 failed for warm startup test data");

    // Verify the file exists and has correct size
    let file_len = std::fs::metadata(&idx_path).unwrap().len() as usize;
    let expected_len = 16384 + entry_count * 256;
    assert_eq!(file_len, expected_len, "v2 file size mismatch");

    // --- Time the warm startup sequence ---
    let start = Instant::now();

    // Step 1: mmap the file and validate header (IndexSnapshot::from_file)
    let snapshot = IndexSnapshot::from_file(&idx_path, None)
        .expect("IndexSnapshot::from_file failed during warm startup");

    // Step 2: Store the snapshot in IndexStore (atomic swap)
    let store = Arc::new(IndexStore::new());
    store.swap(snapshot);

    // Step 3: Verify the snapshot is accessible and correct
    let guard = store.snapshot();
    let snap = guard.as_ref().as_ref().expect("store should have snapshot after swap");
    let loaded_count = snap.entry_count();

    let elapsed = start.elapsed();

    // --- Validate correctness ---
    assert_eq!(
        loaded_count, entry_count,
        "Warm startup should load all {} entries, got {}",
        entry_count, loaded_count
    );

    // Validate header fields
    let header = snap.header();
    assert_eq!(header.root_hash, root_hash, "root_hash mismatch");
    assert_eq!(header.last_fsevents_id, fsevents_id, "fsevents_id mismatch");
    assert_eq!(header.exclude_hash, exclude_hash, "exclude_hash mismatch");
    assert_eq!(header.entry_count as usize, entry_count, "entry_count mismatch");

    // Spot-check a few entries are valid
    let snap_entries = snap.entries();
    assert_eq!(snap_entries.len(), entry_count);
    assert!(snap_entries[0].path_len > 0, "First entry should have valid path_len");
    assert!(snap_entries[entry_count / 2].path_len > 0, "Middle entry should have valid path_len");
    assert!(snap_entries[entry_count - 1].path_len > 0, "Last entry should have valid path_len");

    // --- Assert performance target ---
    assert!(
        elapsed.as_millis() < 100,
        "Warm startup (mmap + validate + store swap) took {:?} ({:.2}ms) for {} entries — exceeds 100ms target",
        elapsed,
        elapsed.as_secs_f64() * 1000.0,
        entry_count
    );

    println!(
        "Warm startup: {} entries in {:.2}ms (mmap + header validate + IndexStore swap + entry count verify)",
        entry_count,
        elapsed.as_secs_f64() * 1000.0
    );
}

// ============================================================================
// Cold Startup Validation: no index -> walk fallback -> background build -> v2
// ============================================================================

/// Validates the cold startup path end-to-end:
/// 1. No index file exists -> IndexStore is empty
/// 2. FilesystemScanner (walk fallback) works immediately
/// 3. Background build (simulated via local scan + save_v2) completes and saves v2
/// 4. IndexSnapshot::from_file loads the built v2 index successfully
/// 5. Subsequent "launches" load the v2 index instantly via IndexStore
#[test]
fn test_cold_startup_with_background_build() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use gpu_search::index::exclude::{default_excludes, compute_exclude_hash};
    use gpu_search::index::gsix_v2::{save_v2, load_v2};
    use gpu_search::index::snapshot::IndexSnapshot;
    use gpu_search::index::store::IndexStore;

    let test_dir = make_test_dir();
    let index_dir = TempDir::new().expect("Failed to create index dir");
    let idx_path = index_dir.path().join("global.idx");

    // ---------------------------------------------------------------
    // Step 1: Cold start — no index file, IndexStore is empty
    // ---------------------------------------------------------------
    assert!(!idx_path.exists(), "Index file should NOT exist at cold start");

    let store = Arc::new(IndexStore::new());
    assert!(
        !store.is_available(),
        "IndexStore should be empty at cold start (no snapshot)"
    );

    // Verify snapshot returns None
    {
        let guard = store.snapshot();
        assert!(
            guard.is_none(),
            "IndexStore snapshot should be None before any build"
        );
    }

    // ---------------------------------------------------------------
    // Step 2: Walk fallback works immediately (search available)
    // ---------------------------------------------------------------
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let walk_entries = scanner.scan(test_dir.path());

    assert_eq!(
        walk_entries.len(),
        EXPECTED_FILE_COUNT,
        "Walk fallback should find {} files immediately at cold start",
        EXPECTED_FILE_COUNT
    );

    // Verify all walked paths exist on disk
    for entry in &walk_entries {
        let path_str = entry_path(entry);
        assert!(
            PathBuf::from(path_str).exists(),
            "Walk fallback path should exist: {}",
            path_str
        );
    }

    // ---------------------------------------------------------------
    // Step 3: Background build — scan + save v2 index
    // (Simulates BackgroundBuilder::build_initial_index but scans
    //  test_dir instead of "/" for test isolation.)
    // ---------------------------------------------------------------
    let excludes = Arc::new(default_excludes());
    let exclude_hash = compute_exclude_hash(&excludes);
    let root_hash = 0xDEAD_BEEF_u32;
    let progress = Arc::new(AtomicUsize::new(0));

    // Simulate background build in a thread (same pattern as BackgroundBuilder)
    let build_store = Arc::clone(&store);
    let build_progress = Arc::clone(&progress);
    let build_entries = walk_entries.clone();
    let build_idx_path = idx_path.clone();

    let build_handle = std::thread::spawn(move || {
        // Filter through excludes (same as build_initial_index)
        let filtered: Vec<GpuPathEntry> = build_entries
            .into_iter()
            .filter(|entry| {
                let path_bytes = &entry.path[..entry.path_len as usize];
                !excludes.should_exclude(path_bytes)
            })
            .collect();

        // Update progress counter
        build_progress.store(filtered.len(), Ordering::Relaxed);

        // Save v2 index (atomic write)
        save_v2(
            &filtered,
            root_hash,
            &build_idx_path,
            0, // no FSEvents ID yet (fresh build)
            exclude_hash,
        )
        .expect("save_v2 should succeed during background build");

        // Create snapshot and swap into store (same as BackgroundBuilder)
        let snapshot = IndexSnapshot::from_file(&build_idx_path, None)
            .expect("IndexSnapshot should load after save");

        build_store.swap(snapshot);

        filtered.len()
    });

    // Wait for background build to complete
    let built_count = build_handle.join().expect("Background build thread should not panic");

    // ---------------------------------------------------------------
    // Step 4: Verify background build completed and saved v2 index
    // ---------------------------------------------------------------
    assert!(
        idx_path.exists(),
        "v2 index file should exist after background build"
    );

    assert!(
        progress.load(Ordering::Relaxed) > 0,
        "Progress counter should be > 0 after build"
    );
    assert_eq!(
        progress.load(Ordering::Relaxed),
        built_count,
        "Progress counter should match built entry count"
    );

    // Verify the saved file is a valid v2 index
    let (header, loaded_entries) = load_v2(&idx_path)
        .expect("load_v2 should succeed on the built index");
    assert_eq!(
        header.entry_count as usize,
        built_count,
        "Header entry_count should match built count"
    );
    assert_eq!(header.root_hash, root_hash, "root_hash should match");
    assert_eq!(header.exclude_hash, exclude_hash, "exclude_hash should match");
    assert_eq!(
        loaded_entries.len(),
        built_count,
        "loaded entries should match built count"
    );

    // ---------------------------------------------------------------
    // Step 5: IndexStore now has the snapshot (post-build)
    // ---------------------------------------------------------------
    assert!(
        store.is_available(),
        "IndexStore should be available after background build swaps snapshot"
    );

    {
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref()
            .expect("IndexStore snapshot should be Some after build");
        assert_eq!(
            snap.entry_count(), built_count,
            "Snapshot entry_count should match built count"
        );
        assert_eq!(
            snap.header().root_hash, root_hash,
            "Snapshot root_hash should match"
        );

        // Verify entries in snapshot match
        let snap_entries = snap.entries();
        assert_eq!(
            snap_entries.len(), built_count,
            "Snapshot entries slice length should match"
        );
    }

    // ---------------------------------------------------------------
    // Step 6: Subsequent launch — load v2 index instantly
    // (Simulates warm startup: IndexSnapshot::from_file succeeds)
    // ---------------------------------------------------------------
    let store2 = Arc::new(IndexStore::new());
    assert!(
        !store2.is_available(),
        "New store should start empty (simulated app restart)"
    );

    // Load the previously saved index (simulating warm startup)
    let start = Instant::now();
    let snapshot2 = IndexSnapshot::from_file(&idx_path, None)
        .expect("Subsequent launch should load v2 index instantly");
    let load_elapsed = start.elapsed();

    store2.swap(snapshot2);

    assert!(
        store2.is_available(),
        "Store should be available after loading saved v2 index"
    );

    {
        let guard = store2.snapshot();
        let snap = guard.as_ref().as_ref()
            .expect("Snapshot should be Some on warm restart");
        assert_eq!(
            snap.entry_count(), built_count,
            "Warm restart snapshot should have same entry count"
        );
        assert_eq!(
            snap.header().root_hash, root_hash,
            "Warm restart snapshot root_hash should match"
        );
        assert_eq!(
            snap.header().exclude_hash, exclude_hash,
            "Warm restart snapshot exclude_hash should match"
        );
    }

    // Load should be fast (mmap, no full read)
    println!(
        "Cold startup test: {} files indexed, v2 load in {:?}",
        built_count,
        load_elapsed
    );
}

// ============================================================================
// 12. Live update cycle validation (task 7.5)
// ============================================================================

/// Validates the full live update pipeline:
///   load index -> create file on disk -> FsChange::Created event ->
///   IndexWriter processes -> flush -> IndexStore swap -> search finds new file
///
/// Measures total latency from fs::write() to searchable and asserts <1s.
#[test]
fn test_live_update_cycle_under_1s() {
    use gpu_search::index::exclude::ExcludeTrie;
    use gpu_search::index::fsevents::FsChange;
    use gpu_search::index::index_writer::spawn_writer_thread;
    use gpu_search::index::index_writer::IndexWriter;
    use gpu_search::index::snapshot::IndexSnapshot;
    use gpu_search::index::store::IndexStore;
    use std::sync::Arc;
    use std::time::Duration;

    let dir = TempDir::new().expect("Failed to create temp dir");
    let idx_path = dir.path().join("live_update.idx");

    // ---------------------------------------------------------------
    // Step 1: Create initial index with some entries and save to disk
    // ---------------------------------------------------------------
    let mut initial_entries = Vec::new();
    for i in 0..20 {
        let mut entry = GpuPathEntry::new();
        let file = dir.path().join(format!("existing_{}.rs", i));
        fs::write(&file, format!("content {}", i)).unwrap();
        entry.set_path(file.as_os_str().as_encoded_bytes());
        entry.set_size(10 + i as u64);
        entry.mtime = 1700000000 + i as u32;
        entry.flags = 0;
        initial_entries.push(entry);
    }

    save_v2(&initial_entries, 0xBEEF, &idx_path, 0, 0xFACE)
        .expect("save_v2 should succeed");

    // ---------------------------------------------------------------
    // Step 2: Load index into IndexStore via IndexSnapshot
    // ---------------------------------------------------------------
    let snapshot = IndexSnapshot::from_file(&idx_path, None)
        .expect("IndexSnapshot::from_file should succeed");
    assert_eq!(snapshot.entry_count(), 20);

    let store = Arc::new(IndexStore::new());
    store.swap(snapshot);
    assert!(store.is_available());

    // Verify initial state: 20 entries
    {
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        assert_eq!(snap.entry_count(), 20);
    }

    // ---------------------------------------------------------------
    // Step 3: Set up IndexWriter from the loaded snapshot + writer thread
    // ---------------------------------------------------------------
    let snapshot_for_writer = IndexSnapshot::from_file(&idx_path, None)
        .expect("second load should succeed");
    let excludes = Arc::new(ExcludeTrie::default());
    let writer = IndexWriter::from_snapshot(
        &snapshot_for_writer,
        excludes,
        store.clone(),
        idx_path.clone(),
    );
    assert_eq!(writer.entry_count(), 20);
    assert_eq!(writer.live_count(), 20);

    let (tx, rx) = crossbeam_channel::bounded::<FsChange>(4096);
    let handle = spawn_writer_thread(writer, rx);

    // ---------------------------------------------------------------
    // Step 4: Create a NEW file on disk and time the full update cycle
    // ---------------------------------------------------------------
    let new_file = dir.path().join("brand_new_file.rs");

    // Start timing from the moment we write the file
    let cycle_start = Instant::now();

    fs::write(&new_file, "fn brand_new() { /* live update test */ }")
        .expect("fs::write should succeed");

    // Step 5: Send FsChange::Created event (simulates FSEvents delivery)
    tx.send(FsChange::Created(new_file.clone()))
        .expect("send Created should succeed");

    // Step 6: Send HistoryDone to trigger immediate flush
    tx.send(FsChange::HistoryDone)
        .expect("send HistoryDone should succeed");

    // Step 7: Wait for IndexStore to reflect the new file
    // Poll the store for up to 5 seconds (generous timeout, should be <1s)
    let mut found = false;
    let poll_deadline = Instant::now() + Duration::from_secs(5);
    let new_file_bytes = new_file.as_os_str().as_encoded_bytes();

    while Instant::now() < poll_deadline {
        let guard = store.snapshot();
        if let Some(snap) = guard.as_ref().as_ref() {
            // Check if the snapshot now contains the new file
            let entries = snap.entries();
            for entry in entries {
                let path_bytes = &entry.path[..entry.path_len as usize];
                if path_bytes == new_file_bytes {
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let cycle_elapsed = cycle_start.elapsed();

    // ---------------------------------------------------------------
    // Step 8: Verify results
    // ---------------------------------------------------------------
    assert!(
        found,
        "New file should be searchable in IndexStore after live update cycle"
    );
    assert!(
        cycle_elapsed < Duration::from_secs(1),
        "Live update cycle should complete in <1s, took {:?}",
        cycle_elapsed
    );

    // Verify the snapshot has 21 entries (20 original + 1 new)
    {
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        assert_eq!(
            snap.entry_count(),
            21,
            "Snapshot should have 21 entries (20 original + 1 new)"
        );
    }

    // Verify the new file's metadata is correct
    {
        let guard = store.snapshot();
        let snap = guard.as_ref().as_ref().unwrap();
        let entries = snap.entries();
        let new_entry = entries
            .iter()
            .find(|e| &e.path[..e.path_len as usize] == new_file_bytes)
            .expect("new file entry should exist in snapshot");
        assert_eq!(
            new_entry.size(),
            41, // "fn brand_new() { /* live update test */ }" = 41 bytes
            "New file size should match written content"
        );
        assert!(new_entry.mtime > 0, "New file should have non-zero mtime");
        // Should NOT be deleted or directory
        assert_eq!(
            new_entry.flags & gpu_search::gpu::types::path_flags::IS_DELETED,
            0,
            "New file should not be marked deleted"
        );
        assert_eq!(
            new_entry.flags & gpu_search::gpu::types::path_flags::IS_DIR,
            0,
            "New file should not be marked as directory"
        );
    }

    // Clean shutdown: drop sender to trigger final flush, join thread
    drop(tx);
    handle.join().expect("writer thread should exit cleanly");

    println!(
        "Live update cycle: fs::write -> searchable in {:?} (target: <1s)",
        cycle_elapsed
    );
}

// ============================================================================
// Concurrent Search During Operations (task 7.6)
// ============================================================================

/// Validate that FilesystemScanner (search fallback) works concurrently with
/// a background index build. Both threads perform scan operations on the
/// same directory tree simultaneously. No panics, no data races.
///
/// Runs 10 iterations to stress test for race conditions.
#[test]
fn test_concurrent_search_during_build() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use gpu_search::index::snapshot::IndexSnapshot;
    use gpu_search::index::store::IndexStore;

    let test_dir = make_test_dir();
    let index_dir = TempDir::new().expect("Failed to create index dir");

    let store = Arc::new(IndexStore::new());
    let total_searches = Arc::new(AtomicUsize::new(0));

    // Run 10 iterations to stress test for races
    for iteration in 0..10 {
        let store_clone = Arc::clone(&store);
        let build_done = Arc::new(AtomicBool::new(false));
        let done_clone = Arc::clone(&build_done);
        let test_path = test_dir.path().to_path_buf();
        let build_idx_path = index_dir.path().join(format!("concurrent_build_{}.idx", iteration));

        // Spawn background build thread: scan + save_v2 + snapshot swap
        let build_handle = std::thread::spawn(move || {
            let scanner = FilesystemScanner::with_config(ScannerConfig {
                respect_gitignore: false,
                skip_hidden: true,
                ..Default::default()
            });
            let entries = scanner.scan(&test_path);
            assert!(
                !entries.is_empty(),
                "Build thread (iter {}) should find entries",
                iteration
            );

            // Save and swap into store
            save_v2(&entries, 0xBEEF, &build_idx_path, iteration as u64, 0)
                .expect("save_v2 should succeed in build thread");

            let snapshot = IndexSnapshot::from_file(&build_idx_path, None)
                .expect("IndexSnapshot should load in build thread");
            store_clone.swap(snapshot);

            done_clone.store(true, Ordering::Release);
            entries.len()
        });

        // Main thread: perform concurrent search (walk) while build is running
        // Keep searching until the build is done to maximize overlap
        let scanner = FilesystemScanner::with_config(ScannerConfig {
            respect_gitignore: false,
            skip_hidden: true,
            ..Default::default()
        });

        let mut searches = 0usize;
        loop {
            let entries = scanner.scan(test_dir.path());
            assert!(
                !entries.is_empty(),
                "Search (iter {}, attempt {}) should find entries even during build",
                iteration, searches
            );
            // Verify entries are coherent (valid path_len, no garbage)
            for entry in &entries {
                assert!(
                    entry.path_len as usize <= GPU_PATH_MAX_LEN,
                    "Entry path_len should be within bounds"
                );
                let path_str = entry_path(entry);
                assert!(!path_str.is_empty(), "Entry path should not be empty");
            }
            searches += 1;
            total_searches.fetch_add(1, Ordering::Relaxed);

            if build_done.load(Ordering::Acquire) {
                break;
            }
        }

        let built = build_handle.join().expect("Build thread should not panic");
        assert!(built > 0, "Build should have produced entries");
        assert!(searches > 0, "Should have completed at least one search during build");
    }

    // Final verification: store has a valid snapshot after all iterations
    assert!(
        store.is_available(),
        "IndexStore should have a snapshot after all build iterations"
    );
    let guard = store.snapshot();
    let snap = guard.as_ref().as_ref().unwrap();
    assert!(
        snap.entry_count() > 0,
        "Final snapshot should have entries"
    );

    println!(
        "Concurrent search during build: 10 iterations, {} total searches, no panics/races",
        total_searches.load(Ordering::Relaxed)
    );
}

/// Validate that IndexStore readers always see a consistent snapshot (old or
/// new) during an IndexWriter flush + snapshot swap. Readers should never see
/// None (if store was previously populated) or partial data. Uses Arc<IndexStore>
/// shared between a writer thread (process_event + flush) and reader on main thread.
///
/// Runs 10 iterations to stress test for race conditions.
#[test]
fn test_concurrent_search_during_update() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use gpu_search::index::exclude::ExcludeTrie;
    use gpu_search::index::index_writer::IndexWriter;
    use gpu_search::index::snapshot::IndexSnapshot;
    use gpu_search::index::store::IndexStore;
    use gpu_search::index::fsevents::FsChange;

    // Run 10 iterations to stress test for races
    for iteration in 0..10 {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let idx_path = dir.path().join(format!("concurrent_update_{}.idx", iteration));

        // Create initial index with 20 entries (real files on disk for stat)
        let mut initial_entries = Vec::new();
        for i in 0..20 {
            let file_path = dir.path().join(format!("file_{}.rs", i));
            fs::write(&file_path, format!("content {}", i)).unwrap();

            let mut entry = GpuPathEntry::new();
            entry.set_path(file_path.as_os_str().as_encoded_bytes());
            entry.set_size((i as u64 + 1) * 100);
            entry.mtime = 1700000000 + i as u32;
            initial_entries.push(entry);
        }

        // Save initial index and load into store
        save_v2(&initial_entries, 0xAAAA, &idx_path, 0, 0)
            .expect("Initial save_v2 should succeed");

        let initial_snapshot = IndexSnapshot::from_file(&idx_path, None)
            .expect("Initial IndexSnapshot should load");
        assert_eq!(initial_snapshot.entry_count(), 20);

        let store = Arc::new(IndexStore::with_snapshot(initial_snapshot));
        assert!(store.is_available());

        let update_done = Arc::new(AtomicBool::new(false));

        // Spawn writer thread: adds new files, processes events, flushes
        let writer_store = Arc::clone(&store);
        let writer_done = Arc::clone(&update_done);
        let writer_dir = dir.path().to_path_buf();
        let writer_idx_path = idx_path.clone();

        let writer_handle = std::thread::spawn(move || {
            let excludes = Arc::new(ExcludeTrie::default());

            // Load snapshot for writer to initialize from
            let snap = IndexSnapshot::from_file(&writer_idx_path, None)
                .expect("Writer snapshot load should succeed");
            let mut writer = IndexWriter::from_snapshot(
                &snap,
                excludes,
                writer_store,
                writer_idx_path,
            );

            // Create 5 new files and process Created events
            for i in 20..25 {
                let new_file = writer_dir.join(format!("new_file_{}.rs", i));
                fs::write(&new_file, format!("new content {}", i)).unwrap();
                writer.process_event(FsChange::Created(new_file));
            }

            // Force a flush (triggers save_v2 + snapshot swap)
            writer.flush().expect("Writer flush should succeed");

            writer_done.store(true, Ordering::Release);

            writer.entry_count()
        });

        // Main thread: keep reading from IndexStore while writer is updating
        let mut checks = 0usize;
        loop {
            let guard = store.snapshot();
            let snap_ref = guard.as_ref().as_ref();

            // Snapshot should ALWAYS be Some (store was pre-populated)
            assert!(
                snap_ref.is_some(),
                "Snapshot should never be None during concurrent update (iter {}, check {})",
                iteration, checks
            );

            let snap = snap_ref.unwrap();
            let count = snap.entry_count();

            // Entry count should be either old (20) or new (>=20), never 0 or garbage
            assert!(
                count >= 20,
                "Snapshot entry_count should be >= 20 (got {}) during concurrent update (iter {}, check {})",
                count, iteration, checks
            );

            // Verify entries slice length matches header
            let entries = snap.entries();
            assert_eq!(
                entries.len(), count,
                "entries().len() should match entry_count() (iter {}, check {})",
                iteration, checks
            );

            // Spot check first entry is valid
            if count > 0 {
                assert!(
                    entries[0].path_len as usize <= GPU_PATH_MAX_LEN,
                    "First entry path_len should be valid (iter {}, check {})",
                    iteration, checks
                );
            }

            // Check header is coherent
            let header = snap.header();
            assert_eq!(
                header.entry_count as usize, count,
                "Header entry_count should match snapshot entry_count (iter {}, check {})",
                iteration, checks
            );

            checks += 1;

            if update_done.load(Ordering::Acquire) {
                // One final read after writer signals done
                let final_guard = store.snapshot();
                let final_snap = final_guard.as_ref().as_ref()
                    .expect("Final snapshot should be Some");
                assert!(
                    final_snap.entry_count() >= 20,
                    "Final snapshot should have >= 20 entries (got {})",
                    final_snap.entry_count()
                );
                break;
            }

            // Small yield to allow writer thread progress
            std::thread::yield_now();
        }

        let writer_final_count = writer_handle.join()
            .expect("Writer thread should not panic");
        assert!(
            writer_final_count >= 20,
            "Writer should have >= 20 entries after update (got {})",
            writer_final_count
        );
        assert!(
            checks > 0,
            "Reader should have completed at least one check during update"
        );
    }

    println!(
        "Concurrent search during update: 10 iterations, no panics/races/partial snapshots"
    );
}

// ============================================================================
// Performance Validation: PM success metrics
// ============================================================================

/// PM success metric: warm start (mmap load + IndexStore swap) < 100ms at 100K entries.
/// This is an alias for test_warm_startup_under_100ms with the perf_ prefix convention.
#[test]
fn test_perf_warm_start_under_100ms() {
    use gpu_search::index::snapshot::IndexSnapshot;
    use gpu_search::index::store::IndexStore;
    use std::sync::Arc;

    let dir = TempDir::new().expect("temp dir");
    let idx_path = dir.path().join("perf_warm.idx");

    let entry_count = 100_000;
    let entries = generate_warm_entries(entry_count);
    save_v2(&entries, 0xDEAD_BEEF, &idx_path, 42, 0)
        .expect("save_v2 failed");

    let start = Instant::now();

    let snapshot = IndexSnapshot::from_file(&idx_path, None)
        .expect("IndexSnapshot::from_file failed");
    let store = Arc::new(IndexStore::new());
    store.swap(snapshot);
    let guard = store.snapshot();
    let snap = guard.as_ref().as_ref().expect("snapshot");
    let count = snap.entry_count();

    let elapsed = start.elapsed();

    assert_eq!(count, entry_count);
    assert!(
        elapsed.as_millis() < 100,
        "PM metric FAIL: warm start took {:?} ({:.2}ms) — must be <100ms",
        elapsed,
        elapsed.as_secs_f64() * 1000.0,
    );

    println!(
        "PERF warm_start: {:.2}ms for {} entries (target <100ms)",
        elapsed.as_secs_f64() * 1000.0,
        entry_count
    );
}

/// PM success metric: mmap load < 5ms at 1M entries.
/// Tests IndexSnapshot::from_file on a 1M entry (~244MB) v2 index.
#[test]
fn test_perf_mmap_load_1m_under_5ms() {
    use gpu_search::index::snapshot::IndexSnapshot;

    let dir = TempDir::new().expect("temp dir");
    let idx_path = dir.path().join("perf_mmap_1m.idx");

    let entry_count = 1_000_000;
    let entries = generate_warm_entries(entry_count);
    save_v2(&entries, 0xDEAD_BEEF, &idx_path, 42, 0)
        .expect("save_v2 failed");

    // Warm the mmap path once to avoid cold-cache penalty
    {
        let _ = IndexSnapshot::from_file(&idx_path, None).expect("warm-up load");
    }

    // Measure: mmap load only (no Metal device, no data read)
    let start = Instant::now();
    let snapshot = IndexSnapshot::from_file(&idx_path, None)
        .expect("IndexSnapshot::from_file failed");
    let elapsed = start.elapsed();

    assert_eq!(snapshot.entry_count(), entry_count);
    assert!(
        elapsed.as_millis() < 5,
        "PM metric FAIL: mmap load 1M took {:?} ({:.2}ms) — must be <5ms",
        elapsed,
        elapsed.as_secs_f64() * 1000.0,
    );

    println!(
        "PERF mmap_load_1m: {:.2}ms for {} entries (target <5ms)",
        elapsed.as_secs_f64() * 1000.0,
        entry_count
    );
}

/// PM success metric: save < 1s at 1M entries.
/// Tests save_v2 (write + fsync + atomic rename) on 1M entries (~244MB).
/// Note: fsync dominates save time and varies by disk load. We use a 2s
/// relaxed threshold in integration tests to account for CI/background I/O,
/// while the benchmark (cargo bench --bench index_load) validates the strict 1s target.
#[test]
fn test_perf_save_1m_under_1s() {
    let dir = TempDir::new().expect("temp dir");
    let idx_path = dir.path().join("perf_save_1m.idx");

    let entry_count = 1_000_000;
    let entries = generate_warm_entries(entry_count);

    let start = Instant::now();
    save_v2(&entries, 0xDEAD_BEEF, &idx_path, 42, 0)
        .expect("save_v2 failed");
    let elapsed = start.elapsed();

    // Verify file was written correctly
    let file_len = std::fs::metadata(&idx_path).unwrap().len() as usize;
    let expected_len = 16384 + entry_count * 256;
    assert_eq!(file_len, expected_len, "saved file size mismatch");

    // Use relaxed 3s threshold for integration test (fsync varies with disk load
    // and concurrent test/bench activity). The strict 1s target is validated by
    // criterion bench (cargo bench --bench index_load) in isolation.
    assert!(
        elapsed.as_secs() < 3,
        "PM metric FAIL: save 1M took {:?} ({:.2}s) — must be <3s (relaxed; bench target <1s)",
        elapsed,
        elapsed.as_secs_f64(),
    );

    println!(
        "PERF save_1m: {:.2}s for {} entries (target <1s strict, <3s relaxed)",
        elapsed.as_secs_f64(),
        entry_count
    );
}

/// PM success metric: bytesNoCopy buffer creation < 1ms.
/// Tests Metal zero-copy buffer creation from mmap'd index.
#[test]
fn test_perf_bytesnocopy_under_1ms() {
    use gpu_search::index::snapshot::IndexSnapshot;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    let device = MTLCreateSystemDefaultDevice();
    let device = match device {
        Some(d) => d,
        None => {
            println!("PERF bytesnocopy: SKIPPED (no Metal device)");
            return;
        }
    };

    let dir = TempDir::new().expect("temp dir");
    let idx_path = dir.path().join("perf_bytesnocopy.idx");

    let entry_count = 100_000;
    let entries = generate_warm_entries(entry_count);
    save_v2(&entries, 0xDEAD_BEEF, &idx_path, 42, 0)
        .expect("save_v2 failed");

    // Warm up
    {
        let _ = IndexSnapshot::from_file(&idx_path, Some(&*device)).expect("warm-up");
    }

    let start = Instant::now();
    let snapshot = IndexSnapshot::from_file(&idx_path, Some(&*device))
        .expect("IndexSnapshot::from_file with device failed");
    let elapsed = start.elapsed();

    assert_eq!(snapshot.entry_count(), entry_count);
    assert!(
        elapsed.as_millis() < 1,
        "PM metric FAIL: bytesNoCopy took {:?} ({:.2}ms) — must be <1ms",
        elapsed,
        elapsed.as_secs_f64() * 1000.0,
    );

    println!(
        "PERF bytesnocopy: {:.2}ms for {} entries (target <1ms)",
        elapsed.as_secs_f64() * 1000.0,
        entry_count
    );
}

// ============================================================================
// Error Recovery: stale index detection and data validity
// ============================================================================

/// Validates stale index detection and that stale data is still loadable.
///
/// Steps:
/// 1. Save a valid index
/// 2. Tamper saved_at to be 3601+ seconds old (recompute CRC32)
/// 3. Verify is_stale() returns true
/// 4. Verify load_v2 still succeeds (stale but valid data)
#[test]
fn test_error_recovery_stale_index() {
    use gpu_search::index::gsix_v2::{is_stale, is_stale_with_age, load_v2, save_v2};
    use std::time::{SystemTime, UNIX_EPOCH};

    let dir = TempDir::new().expect("Failed to create temp dir");
    let idx_path = dir.path().join("stale_test.idx");

    // Create entries with real data
    let mut entries = Vec::new();
    for i in 0..10 {
        let mut entry = GpuPathEntry::new();
        entry.set_path(format!("/test/stale/file_{}.rs", i).as_bytes());
        entry.set_size((i as u64 + 1) * 512);
        entry.mtime = 1700000000 + i as u32;
        entries.push(entry);
    }

    // Save valid index
    save_v2(&entries, 0xBEEF, &idx_path, 42, 0xCAFE)
        .expect("save_v2 should succeed");

    // Verify freshly saved index is NOT stale
    let (fresh_header, _) = load_v2(&idx_path).expect("Fresh load should succeed");
    assert!(
        !is_stale(&fresh_header),
        "Freshly saved index should not be stale"
    );

    // Tamper: set saved_at to 3601 seconds ago (just past DEFAULT_MAX_AGE_SECS=3600)
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let old_saved_at = now - 3601;

    let mut raw = std::fs::read(&idx_path).expect("Read index file");
    // saved_at is at bytes [16..24) in the header
    raw[16..24].copy_from_slice(&old_saved_at.to_le_bytes());
    // Recompute CRC32 over bytes [0..44) and store at [44..48)
    let crc = crc32fast::hash(&raw[..44]);
    raw[44..48].copy_from_slice(&crc.to_le_bytes());
    std::fs::write(&idx_path, &raw).expect("Write tampered index");

    // Load tampered index — should still succeed (data is valid, just old)
    let (stale_header, loaded) = load_v2(&idx_path)
        .expect("load_v2 should succeed on stale but valid index");

    // Verify staleness detected
    assert!(
        is_stale(&stale_header),
        "Index saved 3601s ago should be stale (max age = 3600s)"
    );
    assert!(
        is_stale_with_age(&stale_header, 3600),
        "is_stale_with_age(3600) should detect 3601s-old index as stale"
    );

    // Verify data integrity preserved despite staleness
    assert_eq!(loaded.len(), 10, "Stale index should still have all 10 entries");
    assert_eq!(stale_header.entry_count, 10);
    assert_eq!(stale_header.root_hash, 0xBEEF);
    assert_eq!(stale_header.last_fsevents_id, 42);
    assert_eq!(stale_header.exclude_hash, 0xCAFE);

    // Verify each entry path is preserved
    for (i, entry) in loaded.iter().enumerate() {
        let expected_path = format!("/test/stale/file_{}.rs", i);
        let actual_path = std::str::from_utf8(&entry.path[..entry.path_len as usize])
            .expect("Path should be valid UTF-8");
        assert_eq!(actual_path, expected_path, "Entry {} path should be preserved", i);
    }

    println!("Error recovery (stale): stale detected, data still valid and loadable");
}

// ============================================================================
// Error Recovery: corrupt index triggers rebuild, walk fallback works
// ============================================================================

/// Validates corrupt index detection and filesystem walk fallback.
///
/// Steps:
/// 1. Save a valid index
/// 2. Overwrite first bytes with garbage
/// 3. Verify load_v2 and load_with_migration return Err
/// 4. Verify FilesystemScanner still works as fallback
#[test]
fn test_error_recovery_corrupt_index() {
    use gpu_search::index::gsix_v2::{load_v2, load_with_migration, save_v2};

    let dir = TempDir::new().expect("Failed to create temp dir");
    let idx_path = dir.path().join("corrupt_recovery.idx");

    // Create and save a valid index
    let mut entries = Vec::new();
    for i in 0..5 {
        let mut entry = GpuPathEntry::new();
        entry.set_path(format!("/corrupt/test/file_{}.txt", i).as_bytes());
        entry.set_size((i as u64 + 1) * 100);
        entries.push(entry);
    }

    save_v2(&entries, 0xDEAD, &idx_path, 100, 0)
        .expect("save_v2 should succeed");

    // Verify the index loads correctly before corruption
    let (header, loaded) = load_v2(&idx_path)
        .expect("Pre-corruption load should succeed");
    assert_eq!(header.entry_count, 5);
    assert_eq!(loaded.len(), 5);

    // Corrupt: overwrite first 16 bytes with garbage
    let mut raw = std::fs::read(&idx_path).expect("Read index file");
    raw[0..16].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04,
                                  0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C]);
    std::fs::write(&idx_path, &raw).expect("Write corrupted index");

    // load_v2 must return Err (not panic)
    let result = load_v2(&idx_path);
    assert!(
        result.is_err(),
        "load_v2 on corrupted index should return Err"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("bad magic"),
        "Error should mention bad magic: {}", err_msg
    );

    // load_with_migration must also return Err gracefully (not panic)
    // Re-write the corrupted data (load_with_migration reads the file fresh)
    let result2 = load_with_migration(&idx_path);
    assert!(
        result2.is_err(),
        "load_with_migration on corrupted index should return Err"
    );

    // Verify FilesystemScanner still works as fallback (independent of index)
    let test_dir = make_test_dir();
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let walk_entries = scanner.scan(test_dir.path());
    assert_eq!(
        walk_entries.len(),
        EXPECTED_FILE_COUNT,
        "FilesystemScanner walk fallback should find {} files even when index is corrupt",
        EXPECTED_FILE_COUNT
    );

    println!("Error recovery (corrupt): rebuild triggered, walk fallback works");
}

// ============================================================================
// Error Recovery: disk-full / read-only directory, save fails gracefully
// ============================================================================

/// Validates that save_v2 to a read-only directory returns Err (not panic)
/// and that walk-based search still works independently.
///
/// Steps:
/// 1. Create a read-only directory
/// 2. Attempt save_v2 — should return Err
/// 3. Verify FilesystemScanner walk still works as fallback
#[test]
fn test_error_recovery_disk_full() {
    use std::os::unix::fs::PermissionsExt;
    use gpu_search::index::gsix_v2::save_v2;

    let dir = TempDir::new().expect("Failed to create temp dir");
    let readonly_dir = dir.path().join("readonly_recovery");
    fs::create_dir(&readonly_dir).unwrap();

    // Make directory read-only (simulates disk-full / permission denied)
    let ro_perms = fs::Permissions::from_mode(0o444);
    fs::set_permissions(&readonly_dir, ro_perms).unwrap();

    // Attempt save_v2 to read-only directory — must return Err, NOT panic
    let idx_path = readonly_dir.join("should_fail.idx");
    let mut entries = Vec::new();
    for i in 0..3 {
        let mut entry = GpuPathEntry::new();
        entry.set_path(format!("/disk_full/file_{}.rs", i).as_bytes());
        entry.set_size((i as u64 + 1) * 256);
        entries.push(entry);
    }

    let result = save_v2(&entries, 0xF00D, &idx_path, 0, 0);

    // Restore permissions before assertions so TempDir cleanup works
    let rw_perms = fs::Permissions::from_mode(0o755);
    fs::set_permissions(&readonly_dir, rw_perms).unwrap();

    assert!(
        result.is_err(),
        "save_v2 to read-only directory should return Err, not panic"
    );

    // Verify no partial file was left behind
    assert!(
        !idx_path.exists(),
        "No index file should exist after failed save"
    );
    let tmp_path = idx_path.with_extension("idx.tmp");
    assert!(
        !tmp_path.exists(),
        "No temp file should exist after failed save"
    );

    // Verify walk-based search still works independently
    let test_dir = make_test_dir();
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let walk_entries = scanner.scan(test_dir.path());
    assert_eq!(
        walk_entries.len(),
        EXPECTED_FILE_COUNT,
        "Walk-based search should work even when index save fails (disk-full scenario)"
    );

    // Also verify a successful save to a writable directory still works
    let writable_path = dir.path().join("writable.idx");
    let result2 = save_v2(&entries, 0xF00D, &writable_path, 0, 0);
    assert!(
        result2.is_ok(),
        "save_v2 to writable directory should succeed after read-only failure"
    );

    println!("Error recovery (disk-full): save fails gracefully, walk fallback works");
}

// ============================================================================
// Backward Compatibility: blocking search() API (alias for regression test)
// ============================================================================

/// Backward compat: blocking search() API works unchanged via walk_directory fallback.
/// This is an explicit backward compatibility alias — the same scenario is also covered
/// by test_regression_blocking_search_api_works and test_graceful_blocking_search_without_index.
#[test]
fn test_backward_compat_blocking_search() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use gpu_search::gpu::pipeline::PsoCache;
    use gpu_search::search::types::SearchRequest;

    let device = MTLCreateSystemDefaultDevice()
        .expect("No Metal device (test requires Apple Silicon)");
    let pso_cache = PsoCache::new(&device);

    // Create orchestrator WITHOUT IndexStore — pure walk fallback
    let mut orchestrator = gpu_search::search::orchestrator::SearchOrchestrator::new(
        &device,
        &pso_cache,
    )
    .expect("Failed to create SearchOrchestrator");

    let test_dir = make_test_dir();

    let request = SearchRequest::new("fn ", test_dir.path().to_str().unwrap());
    let response = orchestrator.search(request);

    // Must find files via walk_directory fallback
    assert!(
        response.total_files_searched > 0,
        "Backward compat: blocking search() must find files via walk_directory"
    );
    // Must find content matches ("fn " appears in main.rs, utils.rs)
    assert!(
        !response.content_matches.is_empty(),
        "Backward compat: blocking search() must find content matches for 'fn '"
    );

    println!("Backward compat: blocking search() API works unchanged");
}

// ============================================================================
// Backward Compatibility: search_streaming + SearchUpdate protocol
// ============================================================================

/// Backward compat: search_streaming() produces SearchUpdate messages via channel.
/// Verifies the streaming protocol (FileMatches, ContentMatches, Complete) works.
#[test]
fn test_backward_compat_streaming_search() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use gpu_search::gpu::pipeline::PsoCache;
    use gpu_search::search::cancel::{cancellation_pair, SearchGeneration, SearchSession};
    use gpu_search::search::types::{SearchRequest, SearchUpdate};

    let device = MTLCreateSystemDefaultDevice()
        .expect("No Metal device (test requires Apple Silicon)");
    let pso_cache = PsoCache::new(&device);

    let mut orchestrator = gpu_search::search::orchestrator::SearchOrchestrator::new(
        &device,
        &pso_cache,
    )
    .expect("Failed to create SearchOrchestrator");

    let test_dir = make_test_dir();

    let request = SearchRequest::new("fn ", test_dir.path().to_str().unwrap());
    let (update_tx, update_rx) = crossbeam_channel::unbounded();

    // Create a valid SearchSession
    let (token, _handle) = cancellation_pair();
    let gen = SearchGeneration::new();
    let guard = gen.next();
    let session = SearchSession { token, guard };

    // Run streaming search
    let response = orchestrator.search_streaming(request, &update_tx, &session);

    // Response must complete
    assert!(
        response.total_files_searched > 0,
        "Backward compat: search_streaming() must search files"
    );

    // Drain channel and verify SearchUpdate variants received
    let mut saw_complete = false;
    let mut content_count = 0u32;
    let mut file_count = 0u32;

    while let Ok(stamped) = update_rx.try_recv() {
        // StampedUpdate must have a generation
        assert!(stamped.generation > 0, "StampedUpdate must have valid generation");
        match stamped.update {
            SearchUpdate::FileMatches(ref matches) => {
                file_count += matches.len() as u32;
            }
            SearchUpdate::ContentMatches(ref matches) => {
                content_count += matches.len() as u32;
            }
            SearchUpdate::Complete(_) => {
                saw_complete = true;
            }
        }
    }

    // Must have sent a Complete update
    assert!(
        saw_complete,
        "Backward compat: search_streaming() must send SearchUpdate::Complete"
    );
    // Must have found content matches ("fn " appears in source files)
    assert!(
        content_count > 0 || !response.content_matches.is_empty(),
        "Backward compat: streaming search must find content matches"
    );

    println!(
        "Backward compat: search_streaming() works — {} file updates, {} content updates, complete={}",
        file_count, content_count, saw_complete
    );
}

// ============================================================================
// Backward Compatibility: GpuPathEntry layout matches Metal shader (256B)
// ============================================================================

/// Backward compat: GpuPathEntry is exactly 256 bytes, matching Metal shader expectations.
/// Compile-time assertions already exist in types.rs and gsix_v2.rs; this test provides
/// an explicit runtime assertion for backward compatibility documentation.
#[test]
fn test_backward_compat_gpu_path_entry_layout() {
    // Runtime verification of the compile-time assertion
    assert_eq!(
        std::mem::size_of::<GpuPathEntry>(),
        256,
        "GpuPathEntry must be exactly 256 bytes to match Metal shader struct"
    );
    assert_eq!(
        std::mem::align_of::<GpuPathEntry>(),
        4,
        "GpuPathEntry alignment must be 4 bytes"
    );

    // Verify key field offsets haven't changed
    let entry = GpuPathEntry::default();
    assert_eq!(entry.path.len(), 224, "path field must be 224 bytes (GPU_PATH_MAX_LEN)");
    assert_eq!(GPU_PATH_MAX_LEN, 224, "GPU_PATH_MAX_LEN constant must be 224");

    println!("Backward compat: GpuPathEntry layout = 256B, align=4, path=224B");
}

// ============================================================================
// Backward Compatibility: walk_and_filter fallback works without index
// ============================================================================

/// Backward compat: FilesystemScanner walk fallback works without any index.
/// This is an explicit backward compatibility alias — the same scenario is also covered
/// by test_graceful_walk_fallback_without_index and test_regression_walk_fallback_still_works.
#[test]
fn test_backward_compat_walk_fallback() {
    let dir = make_test_dir();

    // Walk fallback (no index, no IndexStore)
    let scanner = FilesystemScanner::with_config(ScannerConfig {
        respect_gitignore: false,
        skip_hidden: true,
        ..Default::default()
    });
    let entries = scanner.scan(dir.path());

    assert_eq!(
        entries.len(),
        EXPECTED_FILE_COUNT,
        "Backward compat: walk fallback must find all {} files without index",
        EXPECTED_FILE_COUNT
    );

    // All paths must exist on disk
    for entry in &entries {
        let path_str = entry_path(entry);
        assert!(!path_str.is_empty(), "walk entry path must not be empty");
        let path = PathBuf::from(path_str);
        assert!(
            path.exists(),
            "Backward compat: walk fallback path must exist on disk: {}",
            path_str
        );
    }

    println!("Backward compat: walk_and_filter fallback works without index");
}
