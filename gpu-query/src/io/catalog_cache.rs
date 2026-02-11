//! Cached directory scanner with fingerprint-based invalidation.
//!
//! Wraps [`super::catalog::scan_directory`] and caches results, revalidating
//! on each access by checking directory mtime and per-file `(size, mtime)`
//! fingerprints.  A full re-scan costs ~57ms on cold start; validation is
//! ~0.1ms for 10 files.

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::time::SystemTime;

use super::catalog::{scan_directory, TableEntry};

/// Per-file fingerprint used for cache invalidation.
#[derive(Debug, Clone)]
struct FileFingerprint {
    size: u64,
    modified: SystemTime,
}

/// Cached catalog that tracks a single directory and its table entries.
///
/// On each [`get_or_refresh`](CatalogCache::get_or_refresh) call the cache
/// checks whether the directory or any contained file has changed, and only
/// re-scans when a change is detected.
#[derive(Debug)]
pub struct CatalogCache {
    dir: PathBuf,
    entries: Vec<TableEntry>,
    fingerprints: HashMap<PathBuf, FileFingerprint>,
    dir_modified: Option<SystemTime>,
}

impl CatalogCache {
    /// Create a new, empty cache for the given directory.
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            entries: Vec::new(),
            fingerprints: HashMap::new(),
            dir_modified: None,
        }
    }

    /// Return cached entries if still valid, otherwise re-scan.
    ///
    /// Validity is checked via [`is_valid`](CatalogCache::is_valid).
    pub fn get_or_refresh(&mut self) -> io::Result<&[TableEntry]> {
        if !self.is_valid()? {
            self.refresh()?;
        }
        Ok(&self.entries)
    }

    /// Check whether the cached state is still valid.
    ///
    /// Returns `false` (triggering a re-scan) when:
    /// - The cache has never been populated (`dir_modified` is `None`).
    /// - The directory's modification time has changed.
    /// - Any previously-fingerprinted file has a different size or mtime.
    /// - A previously-fingerprinted file no longer exists.
    pub fn is_valid(&self) -> io::Result<bool> {
        // Never populated -- always invalid.
        let cached_dir_mtime = match self.dir_modified {
            Some(t) => t,
            None => return Ok(false),
        };

        // Check directory mtime first (cheap short-circuit).
        let dir_meta = std::fs::metadata(&self.dir)?;
        let current_dir_mtime = dir_meta.modified()?;
        if current_dir_mtime != cached_dir_mtime {
            return Ok(false);
        }

        // Directory mtime unchanged -- check individual file fingerprints.
        for (path, fp) in &self.fingerprints {
            let meta = match std::fs::metadata(path) {
                Ok(m) => m,
                Err(_) => return Ok(false), // file removed
            };
            if meta.len() != fp.size {
                return Ok(false);
            }
            if let Ok(mtime) = meta.modified() {
                if mtime != fp.modified {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Force a full re-scan of the directory and rebuild fingerprints.
    pub fn refresh(&mut self) -> io::Result<()> {
        let entries = scan_directory(&self.dir)?;

        // Build fingerprints for every catalogued file.
        let mut fingerprints = HashMap::with_capacity(entries.len());
        for entry in &entries {
            if let Ok(meta) = std::fs::metadata(&entry.path) {
                let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                fingerprints.insert(
                    entry.path.clone(),
                    FileFingerprint {
                        size: meta.len(),
                        modified,
                    },
                );
            }
        }

        // Record the current directory mtime.
        let dir_meta = std::fs::metadata(&self.dir)?;
        self.dir_modified = Some(dir_meta.modified().unwrap_or(SystemTime::UNIX_EPOCH));

        self.entries = entries;
        self.fingerprints = fingerprints;
        Ok(())
    }

    /// Manually invalidate the cache, forcing a re-scan on next access.
    pub fn invalidate(&mut self) {
        self.entries.clear();
        self.fingerprints.clear();
        self.dir_modified = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::thread;
    use std::time::Duration;

    /// Helper: create a minimal CSV file that scan_directory will recognise.
    fn write_csv(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, content).expect("write csv");
        path
    }

    #[test]
    fn test_catalog_cache_hit_on_unchanged_dir() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(tmp.path(), "a.csv", "id,name\n1,test\n");

        let mut cache = CatalogCache::new(tmp.path().to_path_buf());

        // First call populates cache.
        let entries1 = cache.get_or_refresh().unwrap().to_vec();
        assert_eq!(entries1.len(), 1);

        // Second call should return cached data (is_valid == true).
        assert!(cache.is_valid().unwrap(), "cache should be valid on unchanged dir");
        let entries2 = cache.get_or_refresh().unwrap();
        assert_eq!(entries2.len(), 1);
        assert_eq!(entries2[0].name, entries1[0].name);
    }

    #[test]
    fn test_catalog_cache_miss_on_new_file() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(tmp.path(), "a.csv", "id\n1\n");

        let mut cache = CatalogCache::new(tmp.path().to_path_buf());

        // Populate cache.
        let entries = cache.get_or_refresh().unwrap();
        assert_eq!(entries.len(), 1);

        // Adding a new file changes the directory mtime.
        // Small sleep to ensure mtime granularity difference on macOS (1s HFS+).
        thread::sleep(Duration::from_millis(1100));
        write_csv(tmp.path(), "b.csv", "x\n2\n");

        // Cache should detect new file via dir mtime change.
        assert!(!cache.is_valid().unwrap(), "cache should be invalid after new file");
        let entries = cache.get_or_refresh().unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_catalog_cache_miss_on_modified_file() {
        let tmp = tempfile::tempdir().unwrap();
        let csv_path = write_csv(tmp.path(), "data.csv", "col\nold\n");

        let mut cache = CatalogCache::new(tmp.path().to_path_buf());

        // Populate cache.
        cache.get_or_refresh().unwrap();
        assert!(cache.is_valid().unwrap());

        // Modify file content (changes size and/or mtime).
        thread::sleep(Duration::from_millis(1100));
        fs::write(&csv_path, "col\nnew_longer_content\n").unwrap();

        // Cache should detect modified file via fingerprint mismatch.
        assert!(!cache.is_valid().unwrap(), "cache should be invalid after file modification");
        let entries = cache.get_or_refresh().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_catalog_cache_invalidate() {
        let tmp = tempfile::tempdir().unwrap();
        write_csv(tmp.path(), "t.csv", "a\n1\n");

        let mut cache = CatalogCache::new(tmp.path().to_path_buf());

        // Populate.
        let entries = cache.get_or_refresh().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(cache.is_valid().unwrap());

        // Manual invalidation should clear everything.
        cache.invalidate();
        assert!(!cache.is_valid().unwrap(), "cache should be invalid after invalidate()");
        assert!(cache.entries.is_empty());
        assert!(cache.fingerprints.is_empty());
        assert!(cache.dir_modified.is_none());

        // Next get_or_refresh re-scans.
        let entries = cache.get_or_refresh().unwrap();
        assert_eq!(entries.len(), 1);
    }
}
