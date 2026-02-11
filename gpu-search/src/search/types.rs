// Search API types for orchestrator and UI layers

use std::ops::Range;
use std::path::PathBuf;
use std::time::Duration;

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
