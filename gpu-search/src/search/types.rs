// Search API types for orchestrator and UI layers

use std::ops::Range;
use std::path::PathBuf;
use std::time::Duration;

use super::profile::PipelineProfile;

/// A search request describing what to search for and where.
#[derive(Debug, Clone)]
pub struct SearchRequest {
    /// The pattern to search for (literal string for MVP).
    pub pattern: String,
    /// Root directory to search in.
    pub root: PathBuf,
    /// Optional file type filters (e.g. ["rs", "toml"]).
    pub file_types: Option<Vec<String>>,
    /// Whether the search is case-sensitive.
    pub case_sensitive: bool,
    /// Whether to respect .gitignore rules.
    pub respect_gitignore: bool,
    /// Whether to include binary files in search.
    pub include_binary: bool,
    /// Maximum number of results to return.
    pub max_results: usize,
}

impl SearchRequest {
    /// Create a new search request with sensible defaults.
    ///
    /// Defaults: case-insensitive, respect .gitignore, exclude binary, max 10_000 results.
    pub fn new(pattern: impl Into<String>, root: impl Into<PathBuf>) -> Self {
        Self {
            pattern: pattern.into(),
            root: root.into(),
            file_types: None,
            case_sensitive: false,
            respect_gitignore: true,
            include_binary: false,
            max_results: 10_000,
        }
    }
}

/// A filename match result (path matched the query pattern).
#[derive(Debug, Clone)]
pub struct FileMatch {
    /// Path to the matched file.
    pub path: PathBuf,
    /// Relevance score (higher = more relevant). Based on path length, match quality.
    pub score: f32,
}

/// A content match result (file contents matched the query pattern).
#[derive(Debug, Clone)]
pub struct ContentMatch {
    /// Path to the file containing the match.
    pub path: PathBuf,
    /// 1-based line number of the match.
    pub line_number: u32,
    /// Full content of the matched line.
    pub line_content: String,
    /// Lines before the match (for context display).
    pub context_before: Vec<String>,
    /// Lines after the match (for context display).
    pub context_after: Vec<String>,
    /// Byte range within `line_content` where the match occurs.
    pub match_range: Range<usize>,
}

/// Aggregated search response with all results and statistics.
#[derive(Debug, Clone)]
pub struct SearchResponse {
    /// Filename matches (path matched query).
    pub file_matches: Vec<FileMatch>,
    /// Content matches (file contents matched query).
    pub content_matches: Vec<ContentMatch>,
    /// Total number of files scanned.
    pub total_files_searched: u64,
    /// Total number of matches found.
    pub total_matches: u64,
    /// Wall-clock time for the search.
    pub elapsed: Duration,
    /// Per-stage pipeline profile with timing and counters.
    pub profile: PipelineProfile,
}

/// Progressive search update for streaming results to the UI.
///
/// Results arrive in waves: filename matches first, then content matches,
/// finally the complete response with statistics.
#[derive(Debug, Clone)]
pub enum SearchUpdate {
    /// Wave 1: Filename matches (fast, from GPU path filter).
    FileMatches(Vec<FileMatch>),
    /// Wave 2: Content matches (from GPU content search).
    ContentMatches(Vec<ContentMatch>),
    /// Final: Complete response with all results and stats.
    Complete(SearchResponse),
}

/// A generation-stamped search update.
///
/// Wraps a `SearchUpdate` with the generation ID of the search session that
/// produced it. The UI uses the generation to discard stale results from
/// superseded searches (the P0 stale-results race condition fix).
#[derive(Debug, Clone)]
pub struct StampedUpdate {
    /// Generation ID of the search session that produced this update.
    pub generation: u64,
    /// The underlying search update payload.
    pub update: SearchUpdate,
}

#[cfg(test)]
mod stamped_generation_tests {
    use super::*;
    use super::super::profile::PipelineProfile;

    /// Helper that mimics `poll_updates()` generation guard logic.
    ///
    /// Takes a vec of `StampedUpdate` and a `current_gen`, returns only
    /// the updates whose generation matches `current_gen` (discards stale
    /// and future generations).
    fn filter_by_generation(updates: Vec<StampedUpdate>, current_gen: u64) -> Vec<SearchUpdate> {
        updates
            .into_iter()
            .filter(|stamped| stamped.generation == current_gen)
            .map(|stamped| stamped.update)
            .collect()
    }

    /// Helper to create a FileMatches update with N dummy file matches.
    fn file_matches_update(gen: u64, count: usize) -> StampedUpdate {
        let matches = (0..count)
            .map(|i| FileMatch {
                path: PathBuf::from(format!("file_{i}.rs")),
                score: 1.0,
            })
            .collect();
        StampedUpdate {
            generation: gen,
            update: SearchUpdate::FileMatches(matches),
        }
    }

    /// Helper to create a ContentMatches update with N dummy content matches.
    fn content_matches_update(gen: u64, count: usize) -> StampedUpdate {
        let matches = (0..count)
            .map(|i| ContentMatch {
                path: PathBuf::from(format!("file_{i}.rs")),
                line_number: (i + 1) as u32,
                line_content: format!("match line {i}"),
                context_before: vec![],
                context_after: vec![],
                match_range: 0..5,
            })
            .collect();
        StampedUpdate {
            generation: gen,
            update: SearchUpdate::ContentMatches(matches),
        }
    }

    /// Helper to create a Complete update.
    fn complete_update(gen: u64) -> StampedUpdate {
        StampedUpdate {
            generation: gen,
            update: SearchUpdate::Complete(SearchResponse {
                file_matches: vec![],
                content_matches: vec![],
                total_files_searched: 100,
                total_matches: 5,
                elapsed: Duration::from_millis(42),
                profile: PipelineProfile::default(),
            }),
        }
    }

    /// U-GEN-1: Updates with the current generation are accepted.
    #[test]
    fn stamped_current_gen_accepted() {
        let current_gen = 5;
        let updates = vec![
            file_matches_update(5, 3),
            content_matches_update(5, 2),
            complete_update(5),
        ];
        let accepted = filter_by_generation(updates, current_gen);
        assert_eq!(accepted.len(), 3, "All 3 updates with current gen should be accepted");
        assert!(matches!(accepted[0], SearchUpdate::FileMatches(ref fm) if fm.len() == 3));
        assert!(matches!(accepted[1], SearchUpdate::ContentMatches(ref cm) if cm.len() == 2));
        assert!(matches!(accepted[2], SearchUpdate::Complete(_)));
    }

    /// U-GEN-2: Updates with a stale (older) generation are discarded.
    #[test]
    fn stamped_stale_gen_discarded() {
        let current_gen = 5;
        let updates = vec![
            file_matches_update(3, 10),    // stale gen 3
            content_matches_update(4, 5),  // stale gen 4
            complete_update(2),            // stale gen 2
        ];
        let accepted = filter_by_generation(updates, current_gen);
        assert!(accepted.is_empty(), "All stale updates should be discarded");
    }

    /// U-GEN-3: Updates with a future generation are discarded.
    ///
    /// This shouldn't happen in practice, but the guard uses strict equality,
    /// so future generations are also rejected.
    #[test]
    fn stamped_future_gen_discarded() {
        let current_gen = 5;
        let updates = vec![
            file_matches_update(6, 2),
            content_matches_update(100, 1),
        ];
        let accepted = filter_by_generation(updates, current_gen);
        assert!(accepted.is_empty(), "Future gen updates should be discarded");
    }

    /// U-GEN-4: Rapid generation advance -- only the latest generation survives.
    ///
    /// Simulates rapid typing where generations advance quickly. Only the final
    /// generation's results should pass through the filter.
    #[test]
    fn stamped_rapid_advance_only_latest_survives() {
        let current_gen = 10;
        let mut updates = Vec::new();
        // Simulate results from generations 6..=10 arriving interleaved
        for gen in 6..=10 {
            updates.push(file_matches_update(gen, 1));
            updates.push(content_matches_update(gen, 1));
        }
        let accepted = filter_by_generation(updates, current_gen);
        // Only gen 10 updates survive (1 FileMatches + 1 ContentMatches)
        assert_eq!(accepted.len(), 2, "Only gen 10 updates should pass");
        assert!(matches!(accepted[0], SearchUpdate::FileMatches(_)));
        assert!(matches!(accepted[1], SearchUpdate::ContentMatches(_)));
    }

    /// U-GEN-5: Complete update is only applied if it matches the current generation.
    ///
    /// A stale Complete should NOT finalize results, otherwise we'd display
    /// counts and timing from a superseded search.
    #[test]
    fn stamped_complete_only_applied_if_current_gen() {
        let current_gen = 7;
        let updates = vec![
            complete_update(6),  // stale Complete
            complete_update(7),  // current Complete
            complete_update(8),  // future Complete
        ];
        let accepted = filter_by_generation(updates, current_gen);
        assert_eq!(accepted.len(), 1, "Only current gen Complete should pass");
        assert!(matches!(accepted[0], SearchUpdate::Complete(_)));
    }

    /// U-GEN-6: Mixed stale and current updates -- only current gen passes.
    ///
    /// Simulates the common case: results from both old and new searches
    /// arrive in the same poll cycle.
    #[test]
    fn stamped_mixed_stale_and_current() {
        let current_gen = 3;
        let updates = vec![
            file_matches_update(2, 5),     // stale
            file_matches_update(3, 2),     // current
            content_matches_update(1, 10), // stale
            content_matches_update(3, 3),  // current
            complete_update(2),            // stale
            complete_update(3),            // current
        ];
        let accepted = filter_by_generation(updates, current_gen);
        assert_eq!(accepted.len(), 3, "Only 3 current-gen updates should pass");
        assert!(matches!(accepted[0], SearchUpdate::FileMatches(ref fm) if fm.len() == 2));
        assert!(matches!(accepted[1], SearchUpdate::ContentMatches(ref cm) if cm.len() == 3));
        assert!(matches!(accepted[2], SearchUpdate::Complete(_)));
    }

    /// U-GEN-7: u64::MAX boundary -- generation at max value works correctly.
    ///
    /// Tests that the generation guard handles the u64 boundary. In practice
    /// wrapping would require 2^64 searches, but correctness at the boundary
    /// is important for robustness.
    #[test]
    fn stamped_u64_max_boundary() {
        let current_gen = u64::MAX;
        let updates = vec![
            file_matches_update(u64::MAX, 1),     // current
            file_matches_update(u64::MAX - 1, 1), // stale
            content_matches_update(0, 1),          // wrapped/stale
        ];
        let accepted = filter_by_generation(updates, current_gen);
        assert_eq!(accepted.len(), 1, "Only u64::MAX gen should pass");
        assert!(matches!(accepted[0], SearchUpdate::FileMatches(ref fm) if fm.len() == 1));
    }
}
