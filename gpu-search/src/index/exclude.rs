// ExcludeTrie: 3-tier path filtering for the persistent index.
//
// Tier 1: absolute path prefix matching (e.g. /System, /Volumes)
// Tier 2: user-relative prefixes expanded at construction (e.g. ~/Library/Caches)
// Tier 3: directory basename matching via HashSet (e.g. .git, node_modules)

use std::collections::HashSet;
use std::path::PathBuf;

use serde::Deserialize;

/// Three-tier path exclusion filter operating on raw bytes.
///
/// Check order: include overrides (return false) -> absolute prefixes ->
/// user-relative prefixes -> basename components.
pub struct ExcludeTrie {
    /// Tier 1: absolute path prefixes (e.g. b"/System", b"/Volumes")
    absolute_prefixes: Vec<Vec<u8>>,
    /// Tier 2: user-relative prefixes, expanded with real HOME at construction
    user_prefixes: Vec<Vec<u8>>,
    /// Tier 3: directory basenames to exclude (e.g. b".git", b"node_modules")
    basenames: HashSet<Vec<u8>>,
    /// Paths that bypass exclusion checks entirely
    include_overrides: HashSet<Vec<u8>>,
}

impl ExcludeTrie {
    /// Create an ExcludeTrie with custom configuration.
    pub fn new(
        absolute_prefixes: Vec<Vec<u8>>,
        user_relative_prefixes: Vec<&[u8]>,
        basenames: HashSet<Vec<u8>>,
        include_overrides: HashSet<Vec<u8>>,
    ) -> Self {
        let home = std::env::var("HOME").unwrap_or_default();
        let home_bytes = home.as_bytes();

        let user_prefixes = user_relative_prefixes
            .into_iter()
            .map(|rel| {
                // rel starts with b"~/" — replace ~ with home
                let mut expanded = Vec::with_capacity(home_bytes.len() + rel.len() - 1);
                expanded.extend_from_slice(home_bytes);
                expanded.extend_from_slice(&rel[1..]); // skip the '~'
                expanded
            })
            .collect();

        Self {
            absolute_prefixes,
            user_prefixes,
            basenames,
            include_overrides,
        }
    }

    /// Check if a path should be excluded from indexing.
    ///
    /// Returns `true` if the path matches any exclusion rule and is not
    /// in the include overrides list.
    pub fn should_exclude(&self, path: &[u8]) -> bool {
        // Include overrides take priority — never exclude these
        if self.include_overrides.contains(path) {
            return false;
        }

        // Tier 1: absolute prefix match
        for prefix in &self.absolute_prefixes {
            if starts_with_prefix(path, prefix) {
                return true;
            }
        }

        // Tier 2: user-relative prefix match (already expanded)
        for prefix in &self.user_prefixes {
            if starts_with_prefix(path, prefix) {
                return true;
            }
        }

        // Tier 3: check each path component against basenames
        if !self.basenames.is_empty() {
            for component in path_components(path) {
                if self.basenames.contains(component) {
                    return true;
                }
            }
        }

        false
    }

    /// Returns true if the trie has no rules at all.
    pub fn is_empty(&self) -> bool {
        self.absolute_prefixes.is_empty()
            && self.user_prefixes.is_empty()
            && self.basenames.is_empty()
    }

    /// Compute a deterministic CRC32 hash of all exclude patterns.
    ///
    /// Sorts all patterns (tagged with tier prefix), joins with NUL separators,
    /// and hashes via CRC32. Stored in GSIX v2 header; mismatch on startup
    /// triggers a full index rebuild.
    pub fn compute_exclude_hash(&self) -> u32 {
        let patterns = self.all_patterns_sorted();
        let mut hasher = crc32fast::Hasher::new();
        for pattern in &patterns {
            hasher.update(pattern);
            hasher.update(b"\0");
        }
        hasher.finalize()
    }

    /// Access the sorted list of all patterns for hashing.
    pub fn all_patterns_sorted(&self) -> Vec<Vec<u8>> {
        let mut patterns = Vec::new();

        for p in &self.absolute_prefixes {
            let mut tagged = b"abs:".to_vec();
            tagged.extend_from_slice(p);
            patterns.push(tagged);
        }

        for p in &self.user_prefixes {
            let mut tagged = b"usr:".to_vec();
            tagged.extend_from_slice(p);
            patterns.push(tagged);
        }

        for b in &self.basenames {
            let mut tagged = b"base:".to_vec();
            tagged.extend_from_slice(b);
            patterns.push(tagged);
        }

        for o in &self.include_overrides {
            let mut tagged = b"inc:".to_vec();
            tagged.extend_from_slice(o);
            patterns.push(tagged);
        }

        patterns.sort();
        patterns
    }
}

impl Default for ExcludeTrie {
    fn default() -> Self {
        let absolute_prefixes: Vec<Vec<u8>> = vec![
            b"/System".to_vec(),
            b"/Library/Caches".to_vec(),
            b"/private/var".to_vec(),
            b"/private/tmp".to_vec(),
            b"/Volumes".to_vec(),
            b"/dev".to_vec(),
            b"/cores".to_vec(),
        ];

        let user_relative: Vec<&[u8]> = vec![b"~/Library/Caches", b"~/.Trash"];

        let basenames: HashSet<Vec<u8>> = [
            b".git".as_slice(),
            b"node_modules",
            b"target",
            b"__pycache__",
            b"vendor",
            b"dist",
            b"build",
            b".cache",
            b"venv",
            b".venv",
            b".Spotlight-V100",
            b".fseventsd",
            b".DS_Store",
            b".Trashes",
            b".hg",
            b".svn",
            b".idea",
            b".vscode",
        ]
        .iter()
        .map(|s| s.to_vec())
        .collect();

        Self::new(absolute_prefixes, user_relative, basenames, HashSet::new())
    }
}

/// Compute a deterministic CRC32 hash of exclude configuration.
///
/// Convenience wrapper around `ExcludeTrie::compute_exclude_hash()`.
/// The hash is stored in the GSIX v2 header; a mismatch on startup
/// triggers a full index rebuild.
pub fn compute_exclude_hash(excludes: &ExcludeTrie) -> u32 {
    excludes.compute_exclude_hash()
}

/// Create default excludes covering all system paths from TECH.md.
///
/// Returns an `ExcludeTrie` with:
/// - Tier 1: `/System`, `/Library/Caches`, `/private/var`, `/private/tmp`, `/Volumes`, `/dev`, `/cores`
/// - Tier 2: `~/Library/Caches`, `~/.Trash`
/// - Tier 3: `.git`, `node_modules`, `target`, `__pycache__`, `vendor`, `dist`, `build`,
///   `.cache`, `venv`, `.venv`, `.Spotlight-V100`, `.fseventsd`, `.DS_Store`, `.Trashes`,
///   `.hg`, `.svn`, `.idea`, `.vscode`
pub fn default_excludes() -> ExcludeTrie {
    ExcludeTrie::default()
}

/// User-facing config file structure loaded from `~/.gpu-search/config.json`.
///
/// All fields are optional; missing fields leave defaults unchanged.
/// ```json
/// {
///   "exclude_dirs": ["node_modules", ".git"],
///   "exclude_absolute": ["/some/custom/path"],
///   "include_override": ["/Volumes/MyDisk"]
/// }
/// ```
#[derive(Debug, Deserialize, Default)]
pub struct ExcludeConfig {
    /// Additional directory basenames to exclude (e.g. `["node_modules", ".git"]`)
    #[serde(default)]
    pub exclude_dirs: Vec<String>,
    /// Additional absolute path prefixes to exclude (e.g. `["/some/custom/path"]`)
    #[serde(default)]
    pub exclude_absolute: Vec<String>,
    /// Absolute paths that bypass all exclusion checks (e.g. `["/Volumes/MyDisk"]`)
    #[serde(default)]
    pub include_override: Vec<String>,
}

/// Attempt to load config from `~/.gpu-search/config.json`.
///
/// Returns `None` if the file does not exist or cannot be parsed.
/// Parsing errors are logged to stderr but do not prevent startup.
pub fn load_config() -> Option<ExcludeConfig> {
    let home = std::env::var("HOME").ok()?;
    let config_path = PathBuf::from(home).join(".gpu-search").join("config.json");
    load_config_from(&config_path)
}

/// Load config from a specific path (for testing or custom locations).
///
/// Returns `None` if the file does not exist or cannot be parsed.
pub fn load_config_from(path: &std::path::Path) -> Option<ExcludeConfig> {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            eprintln!("gpu-search: failed to read config {:?}: {}", path, e);
            return None;
        }
    };

    match serde_json::from_str::<ExcludeConfig>(&contents) {
        Ok(cfg) => Some(cfg),
        Err(e) => {
            eprintln!("gpu-search: failed to parse config {:?}: {}", path, e);
            None
        }
    }
}

/// Merge user config with hardcoded defaults.
///
/// Config additions are additive: `exclude_dirs` extends the basename set,
/// `exclude_absolute` extends absolute prefixes, `include_override` extends
/// the include overrides set. If `config` is `None`, defaults are returned unchanged.
pub fn merge_with_config(defaults: ExcludeTrie, config: Option<ExcludeConfig>) -> ExcludeTrie {
    let config = match config {
        Some(c) => c,
        None => return defaults,
    };

    let mut absolute_prefixes = defaults.absolute_prefixes;
    let mut basenames = defaults.basenames;
    let mut include_overrides = defaults.include_overrides;
    // user_prefixes are already expanded at construction -- keep as-is
    let user_prefixes = defaults.user_prefixes;

    // Merge additional absolute exclusions
    for path in config.exclude_absolute {
        let bytes = path.into_bytes();
        if !absolute_prefixes.contains(&bytes) {
            absolute_prefixes.push(bytes);
        }
    }

    // Merge additional basename exclusions
    for dir in config.exclude_dirs {
        basenames.insert(dir.into_bytes());
    }

    // Merge include overrides
    for path in config.include_override {
        include_overrides.insert(path.into_bytes());
    }

    ExcludeTrie {
        absolute_prefixes,
        user_prefixes,
        basenames,
        include_overrides,
    }
}

/// Check if `path` starts with `prefix`, ensuring the match is at a path boundary.
/// The prefix must match either exactly or be followed by b'/'.
fn starts_with_prefix(path: &[u8], prefix: &[u8]) -> bool {
    if path.len() < prefix.len() {
        return false;
    }
    if path[..prefix.len()] != prefix[..] {
        return false;
    }
    // Exact match or next char is '/'
    path.len() == prefix.len() || path[prefix.len()] == b'/'
}

/// Iterator over path components split by b'/'.
fn path_components(path: &[u8]) -> impl Iterator<Item = &[u8]> {
    path.split(|&b| b == b'/').filter(|c| !c.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exclude_absolute_prefix() {
        let trie = ExcludeTrie::default();
        assert!(trie.should_exclude(b"/System/Library/Frameworks"));
        assert!(trie.should_exclude(b"/Volumes/External"));
        assert!(trie.should_exclude(b"/dev/null"));
        assert!(trie.should_exclude(b"/cores/core.1234"));
        assert!(trie.should_exclude(b"/private/var/folders"));
        assert!(trie.should_exclude(b"/private/tmp/scratch"));
        // Exact match (no trailing path)
        assert!(trie.should_exclude(b"/System"));
    }

    #[test]
    fn test_exclude_basename() {
        let trie = ExcludeTrie::default();
        assert!(trie.should_exclude(b"/Users/dev/project/.git/config"));
        assert!(trie.should_exclude(b"/Users/dev/project/node_modules/react/index.js"));
        assert!(trie.should_exclude(b"/Users/dev/rust-project/target/debug/main"));
        assert!(trie.should_exclude(b"/Users/dev/python/__pycache__/mod.pyc"));
        assert!(trie.should_exclude(b"/Users/dev/.vscode/settings.json"));
    }

    #[test]
    fn test_include_override() {
        let mut overrides = HashSet::new();
        overrides.insert(b"/System/special".to_vec());
        let trie = ExcludeTrie::new(
            vec![b"/System".to_vec()],
            vec![],
            HashSet::new(),
            overrides,
        );
        // The exact override path is not excluded
        assert!(!trie.should_exclude(b"/System/special"));
        // But other /System paths are still excluded
        assert!(trie.should_exclude(b"/System/Library"));
    }

    #[test]
    fn test_empty_trie() {
        let trie = ExcludeTrie::new(vec![], vec![], HashSet::new(), HashSet::new());
        assert!(trie.is_empty());
        assert!(!trie.should_exclude(b"/System"));
        assert!(!trie.should_exclude(b"/Users/dev/.git/config"));
        assert!(!trie.should_exclude(b"/anything"));
    }

    #[test]
    fn test_deeply_nested_excluded() {
        let trie = ExcludeTrie::default();
        assert!(trie.should_exclude(b"/foo/bar/.git/objects/pack/data"));
        assert!(trie.should_exclude(b"/a/b/c/d/e/node_modules/pkg/lib/index.js"));
    }

    #[test]
    fn test_non_excluded_passes() {
        let trie = ExcludeTrie::default();
        assert!(!trie.should_exclude(b"/Users/dev/code/main.rs"));
        assert!(!trie.should_exclude(b"/Users/dev/projects/app/src/lib.rs"));
        assert!(!trie.should_exclude(b"/tmp/scratch.txt"));
        assert!(!trie.should_exclude(b"/usr/local/bin/tool"));
    }

    #[test]
    fn test_user_relative_prefix() {
        let trie = ExcludeTrie::default();
        let home = std::env::var("HOME").unwrap();
        let cache_path = format!("{}/Library/Caches/com.apple.Safari", home);
        assert!(trie.should_exclude(cache_path.as_bytes()));

        let trash_path = format!("{}/.Trash/deleted_file.txt", home);
        assert!(trie.should_exclude(trash_path.as_bytes()));

        // Non-excluded user path
        let code_path = format!("{}/code/project/main.rs", home);
        assert!(!trie.should_exclude(code_path.as_bytes()));
    }

    #[test]
    fn test_prefix_boundary() {
        // /SystemFoo should NOT match /System prefix
        let trie = ExcludeTrie::default();
        assert!(!trie.should_exclude(b"/SystemFoo/bar"));
        assert!(!trie.should_exclude(b"/developer/tool"));
    }

    #[test]
    fn test_all_patterns_sorted_deterministic() {
        let trie1 = ExcludeTrie::default();
        let trie2 = ExcludeTrie::default();
        assert_eq!(trie1.all_patterns_sorted(), trie2.all_patterns_sorted());
    }

    #[test]
    fn test_exclude_hash_deterministic() {
        // Same excludes must produce the same hash every time
        let trie1 = ExcludeTrie::default();
        let trie2 = ExcludeTrie::default();
        let hash1 = trie1.compute_exclude_hash();
        let hash2 = trie2.compute_exclude_hash();
        assert_eq!(hash1, hash2, "same ExcludeTrie config must produce same hash");
        // Also non-zero (extremely unlikely for real patterns)
        assert_ne!(hash1, 0, "hash of non-empty trie should be non-zero");
    }

    #[test]
    fn test_exclude_hash_different_configs_differ() {
        let trie_default = ExcludeTrie::default();

        // Build a different ExcludeTrie with fewer patterns
        let trie_small = ExcludeTrie::new(
            vec![b"/System".to_vec()],
            vec![],
            HashSet::new(),
            HashSet::new(),
        );

        let hash_default = trie_default.compute_exclude_hash();
        let hash_small = trie_small.compute_exclude_hash();
        assert_ne!(
            hash_default, hash_small,
            "different exclude configs must produce different hashes"
        );
    }

    #[test]
    fn test_exclude_hash_empty_trie() {
        let trie = ExcludeTrie::new(vec![], vec![], HashSet::new(), HashSet::new());
        // Empty trie should still produce a valid (deterministic) hash
        let hash1 = trie.compute_exclude_hash();
        let hash2 = trie.compute_exclude_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_compute_exclude_hash_free_function() {
        // The free function should produce the same result as the method
        let trie = ExcludeTrie::default();
        let method_hash = trie.compute_exclude_hash();
        let fn_hash = compute_exclude_hash(&trie);
        assert_eq!(method_hash, fn_hash);
    }

    #[test]
    fn test_default_excludes_comprehensive() {
        let trie = default_excludes();

        // Tier 1: absolute prefixes -- all system-root paths from TECH.md
        assert!(trie.should_exclude(b"/System/Library/Frameworks/Metal.framework"));
        assert!(trie.should_exclude(b"/Library/Caches/com.apple.Safari"));
        assert!(trie.should_exclude(b"/private/var/folders/xx/tmp"));
        assert!(trie.should_exclude(b"/private/tmp/scratch"));
        assert!(trie.should_exclude(b"/Volumes/External/backup"));
        assert!(trie.should_exclude(b"/dev/null"));
        assert!(trie.should_exclude(b"/cores/core.1234"));

        // Tier 1: exact match (no trailing path)
        assert!(trie.should_exclude(b"/System"));
        assert!(trie.should_exclude(b"/Library/Caches"));
        assert!(trie.should_exclude(b"/Volumes"));
        assert!(trie.should_exclude(b"/dev"));
        assert!(trie.should_exclude(b"/cores"));

        // Tier 1: boundary check -- similar names must NOT match
        assert!(!trie.should_exclude(b"/SystemFoo/bar"));
        assert!(!trie.should_exclude(b"/Library/Extensions"));
        assert!(!trie.should_exclude(b"/developer/tools"));

        // Tier 2: user-relative prefixes
        let home = std::env::var("HOME").unwrap();
        let cache_path = format!("{}/Library/Caches/com.apple.dt.Xcode", home);
        assert!(trie.should_exclude(cache_path.as_bytes()));
        let trash_path = format!("{}/.Trash/deleted.txt", home);
        assert!(trie.should_exclude(trash_path.as_bytes()));

        // Tier 2: non-excluded user paths must pass
        let code_path = format!("{}/Developer/project/main.rs", home);
        assert!(!trie.should_exclude(code_path.as_bytes()));

        // Tier 3: all basename patterns
        let basenames = [
            ".git",
            "node_modules",
            "target",
            "__pycache__",
            "vendor",
            "dist",
            "build",
            ".cache",
            "venv",
            ".venv",
            ".Spotlight-V100",
            ".fseventsd",
            ".DS_Store",
            ".Trashes",
            ".hg",
            ".svn",
            ".idea",
            ".vscode",
        ];
        for name in &basenames {
            let path = format!("/Users/dev/project/{}/file.txt", name);
            assert!(
                trie.should_exclude(path.as_bytes()),
                "basename '{}' should be excluded",
                name
            );
        }

        // Non-excluded paths must pass
        assert!(!trie.should_exclude(b"/Users/dev/code/main.rs"));
        assert!(!trie.should_exclude(b"/usr/local/bin/tool"));
        assert!(!trie.should_exclude(b"/tmp/scratch.txt"));
        assert!(!trie.should_exclude(b"/Applications/Xcode.app"));
    }

    #[test]
    fn test_default_excludes_fn_matches_default_impl() {
        let from_fn = default_excludes();
        let from_default = ExcludeTrie::default();
        assert_eq!(
            from_fn.compute_exclude_hash(),
            from_default.compute_exclude_hash(),
            "default_excludes() must match ExcludeTrie::default()"
        );
    }

    #[test]
    fn test_exclude_hash_order_independent_of_insertion() {
        // Even if basenames are inserted in different order, hash should be the same
        // because all_patterns_sorted() sorts them
        let mut basenames1 = HashSet::new();
        basenames1.insert(b".git".to_vec());
        basenames1.insert(b"node_modules".to_vec());
        basenames1.insert(b"target".to_vec());

        let mut basenames2 = HashSet::new();
        basenames2.insert(b"target".to_vec());
        basenames2.insert(b"node_modules".to_vec());
        basenames2.insert(b".git".to_vec());

        let trie1 = ExcludeTrie::new(vec![], vec![], basenames1, HashSet::new());
        let trie2 = ExcludeTrie::new(vec![], vec![], basenames2, HashSet::new());

        assert_eq!(
            trie1.compute_exclude_hash(),
            trie2.compute_exclude_hash(),
            "insertion order of basenames must not affect hash"
        );
    }

    #[test]
    fn test_config_merge() {
        let defaults = ExcludeTrie::default();
        let default_hash = defaults.compute_exclude_hash();

        // Create config with additional exclusions and overrides
        let config = ExcludeConfig {
            exclude_dirs: vec!["custom_dir".to_string(), "my_cache".to_string()],
            exclude_absolute: vec!["/opt/secret".to_string(), "/data/tmp".to_string()],
            include_override: vec!["/Volumes/MyDisk".to_string()],
        };

        let merged = merge_with_config(ExcludeTrie::default(), Some(config));

        // New basename exclusions work
        assert!(
            merged.should_exclude(b"/Users/dev/project/custom_dir/file.txt"),
            "custom_dir should be excluded after merge"
        );
        assert!(
            merged.should_exclude(b"/Users/dev/project/my_cache/data.bin"),
            "my_cache should be excluded after merge"
        );

        // New absolute exclusions work
        assert!(
            merged.should_exclude(b"/opt/secret/keys"),
            "/opt/secret should be excluded after merge"
        );
        assert!(
            merged.should_exclude(b"/data/tmp/scratch"),
            "/data/tmp should be excluded after merge"
        );

        // Include override works -- /Volumes is normally excluded but /Volumes/MyDisk is overridden
        assert!(
            !merged.should_exclude(b"/Volumes/MyDisk"),
            "/Volumes/MyDisk should be included via override"
        );

        // Original defaults still work
        assert!(
            merged.should_exclude(b"/System/Library/something"),
            "default /System exclusion should still work"
        );
        assert!(
            merged.should_exclude(b"/Users/dev/project/.git/config"),
            "default .git basename exclusion should still work"
        );
        assert!(
            !merged.should_exclude(b"/Users/dev/code/main.rs"),
            "non-excluded paths should still pass"
        );

        // Hash should differ from defaults due to added patterns
        let merged_hash = merged.compute_exclude_hash();
        assert_ne!(
            default_hash, merged_hash,
            "merged config must produce different hash than defaults"
        );
    }

    #[test]
    fn test_config_merge_none_returns_defaults() {
        let defaults = ExcludeTrie::default();
        let default_hash = defaults.compute_exclude_hash();

        let merged = merge_with_config(ExcludeTrie::default(), None);
        let merged_hash = merged.compute_exclude_hash();

        assert_eq!(
            default_hash, merged_hash,
            "merge with None config must return identical defaults"
        );
    }

    #[test]
    fn test_config_merge_empty_config_returns_defaults() {
        let defaults = ExcludeTrie::default();
        let default_hash = defaults.compute_exclude_hash();

        let empty_config = ExcludeConfig::default();
        let merged = merge_with_config(ExcludeTrie::default(), Some(empty_config));
        let merged_hash = merged.compute_exclude_hash();

        assert_eq!(
            default_hash, merged_hash,
            "merge with empty config must return identical defaults"
        );
    }

    #[test]
    fn test_config_merge_no_duplicate_absolute() {
        // Adding an absolute prefix that already exists should not duplicate it
        let config = ExcludeConfig {
            exclude_dirs: vec![],
            exclude_absolute: vec!["/System".to_string()],
            include_override: vec![],
        };

        let defaults = ExcludeTrie::default();
        let default_patterns = defaults.all_patterns_sorted();

        let merged = merge_with_config(ExcludeTrie::default(), Some(config));
        let merged_patterns = merged.all_patterns_sorted();

        assert_eq!(
            default_patterns, merged_patterns,
            "adding existing absolute prefix should not create duplicate"
        );
    }

    #[test]
    fn test_config_missing_file() {
        // Loading from a non-existent path should return None
        let path = std::path::Path::new("/nonexistent/path/config.json");
        let result = load_config_from(path);
        assert!(result.is_none(), "missing config file should return None");
    }

    #[test]
    fn test_config_load_valid_file() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        let json = r#"{
            "exclude_dirs": ["custom_dir", "my_cache"],
            "exclude_absolute": ["/opt/secret"],
            "include_override": ["/Volumes/MyDisk"]
        }"#;
        std::fs::write(&config_path, json).unwrap();

        let config = load_config_from(&config_path).expect("should parse valid config");
        assert_eq!(config.exclude_dirs, vec!["custom_dir", "my_cache"]);
        assert_eq!(config.exclude_absolute, vec!["/opt/secret"]);
        assert_eq!(config.include_override, vec!["/Volumes/MyDisk"]);
    }

    #[test]
    fn test_config_load_partial_fields() {
        // Config with only some fields -- missing fields should default to empty
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        let json = r#"{ "exclude_dirs": ["custom_only"] }"#;
        std::fs::write(&config_path, json).unwrap();

        let config = load_config_from(&config_path).expect("should parse partial config");
        assert_eq!(config.exclude_dirs, vec!["custom_only"]);
        assert!(config.exclude_absolute.is_empty());
        assert!(config.include_override.is_empty());
    }

    #[test]
    fn test_config_load_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(&config_path, "not valid json {{{").unwrap();

        let result = load_config_from(&config_path);
        assert!(result.is_none(), "invalid JSON should return None");
    }

    #[test]
    fn test_config_load_empty_object() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(&config_path, "{}").unwrap();

        let config = load_config_from(&config_path).expect("should parse empty object");
        assert!(config.exclude_dirs.is_empty());
        assert!(config.exclude_absolute.is_empty());
        assert!(config.include_override.is_empty());
    }

    #[test]
    fn test_config_end_to_end() {
        // Full pipeline: write config file -> load -> merge -> verify behavior
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        let json = r#"{
            "exclude_dirs": ["my_build"],
            "exclude_absolute": ["/scratch"],
            "include_override": ["/Volumes/Work"]
        }"#;
        std::fs::write(&config_path, json).unwrap();

        let config = load_config_from(&config_path);
        let trie = merge_with_config(ExcludeTrie::default(), config);

        // Custom dir excluded
        assert!(trie.should_exclude(b"/Users/dev/project/my_build/out"));
        // Custom absolute excluded
        assert!(trie.should_exclude(b"/scratch/temp"));
        // Override works
        assert!(!trie.should_exclude(b"/Volumes/Work"));
        // Defaults still active
        assert!(trie.should_exclude(b"/System/Library"));
        assert!(trie.should_exclude(b"/Users/dev/.git/HEAD"));
    }
}
