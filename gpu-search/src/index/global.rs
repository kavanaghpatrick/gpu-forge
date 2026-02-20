//! Global index path helpers for the single root "/" index.
//!
//! The global index is the system-wide persistent file index built from root "/".
//! It lives at `~/.gpu-search/index/global.idx` and uses a deterministic cache key
//! derived from hashing the "/" path string.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

/// Return the global index directory: `~/.gpu-search/index/`.
pub fn global_index_dir() -> PathBuf {
    let home = dirs::home_dir().expect("HOME directory must be available");
    home.join(".gpu-search").join("index")
}

/// Return the global index file path: `~/.gpu-search/index/global.idx`.
pub fn global_index_path() -> PathBuf {
    global_index_dir().join("global.idx")
}

/// Ensure the global index directory exists, creating it if necessary.
pub fn ensure_index_dir() -> std::io::Result<PathBuf> {
    let dir = global_index_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Compute the cache key for the global root "/" path.
///
/// Uses the same hashing algorithm as `cache::cache_key()` (DefaultHasher / SipHash)
/// but for the fixed "/" root path. The result is a deterministic u32 hash.
pub fn global_cache_key() -> u32 {
    let mut hasher = DefaultHasher::new();
    "/".hash(&mut hasher);
    hasher.finish() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_index_path() {
        let path = global_index_path();
        let path_str = path.to_string_lossy();
        assert!(
            path_str.ends_with(".gpu-search/index/global.idx"),
            "path should end with .gpu-search/index/global.idx, got: {}",
            path_str
        );
        assert!(
            path_str.contains("/.gpu-search/index/global.idx"),
            "path should contain /.gpu-search/index/global.idx, got: {}",
            path_str
        );
    }

    #[test]
    fn test_global_index_dir() {
        let dir = global_index_dir();
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.ends_with(".gpu-search/index"),
            "dir should end with .gpu-search/index, got: {}",
            dir_str
        );
    }

    #[test]
    fn test_global_cache_key() {
        let key1 = global_cache_key();
        let key2 = global_cache_key();
        // Must be deterministic
        assert_eq!(key1, key2, "global_cache_key must be deterministic");
        // Must be non-zero (extremely unlikely for "/" hash to be zero)
        assert_ne!(key1, 0, "cache key for '/' should be non-zero");
    }

    #[test]
    fn test_global_cache_key_matches_manual_hash() {
        // Verify it produces the same result as manually hashing "/"
        let mut hasher = DefaultHasher::new();
        "/".hash(&mut hasher);
        let expected = hasher.finish() as u32;
        assert_eq!(
            global_cache_key(),
            expected,
            "global_cache_key should match manual DefaultHasher('/')"
        );
    }

    #[test]
    fn test_ensure_index_dir() {
        // ensure_index_dir should create the directory and return its path
        let dir = ensure_index_dir().expect("ensure_index_dir should succeed");
        assert!(dir.is_dir(), "index directory should exist after ensure_index_dir");
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.ends_with(".gpu-search/index"),
            "returned dir should end with .gpu-search/index, got: {}",
            dir_str
        );
    }
}
