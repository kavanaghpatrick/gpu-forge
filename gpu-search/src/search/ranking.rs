//! Result ranking for search results.
//!
//! Filename matches ranked by path length (shorter = more relevant).
//! Content matches ranked by exact word boundary > partial match > path depth.
//! All results capped at 10K.

use super::types::{ContentMatch, FileMatch};

/// Maximum number of results to keep after truncation.
pub const MAX_RESULTS: usize = 10_000;

// ============================================================================
// Scoring
// ============================================================================

/// Score a filename match. Shorter path = higher score.
///
/// Returns a score in `[0.0, 1.0]` range where 1.0 is the best match.
pub fn score_file_match(path_len: usize, _pattern: &str) -> f32 {
    // Inverse path length: shorter paths score higher.
    // Clamp to avoid division by zero.
    let len = (path_len as f32).max(1.0);
    // Normalize: a 1-char path gets ~1.0, a 200-char path gets ~0.005
    1.0 / len * 10.0
}

/// Score a content match for relevance.
///
/// - Exact word boundary match (whitespace/punctuation around match) = 1.0
/// - Partial (substring inside a word) = 0.5
/// - Then penalize by path depth (more `/` separators = less relevant).
pub fn score_content_match(
    line_content: &str,
    match_start: usize,
    match_len: usize,
    path_depth: usize,
    _pattern: &str,
) -> f32 {
    let base_score = if is_word_boundary_match(line_content, match_start, match_len) {
        1.0
    } else {
        0.5
    };

    // Penalize deep paths: each level of depth reduces score by 0.02
    let depth_penalty = (path_depth as f32) * 0.02;
    (base_score - depth_penalty).max(0.01)
}

/// Check if a match at the given position is on a word boundary.
///
/// A word boundary means the character before `start` and the character
/// after `start + len` are non-alphanumeric (or the match is at the
/// start/end of the string).
fn is_word_boundary_match(text: &str, start: usize, len: usize) -> bool {
    let bytes = text.as_bytes();
    let end = start + len;

    // Check character before match
    let left_boundary = if start == 0 {
        true
    } else if start <= bytes.len() {
        !bytes[start - 1].is_ascii_alphanumeric() && bytes[start - 1] != b'_'
    } else {
        false
    };

    // Check character after match
    let right_boundary = if end >= bytes.len() {
        true
    } else {
        !bytes[end].is_ascii_alphanumeric() && bytes[end] != b'_'
    };

    left_boundary && right_boundary
}

// ============================================================================
// Ranking
// ============================================================================

/// Rank filename matches by path length (shorter first), then alphabetically.
pub fn rank_file_matches(matches: &mut Vec<FileMatch>) {
    matches.sort_by(|a, b| {
        let len_a = a.path.to_string_lossy().len();
        let len_b = b.path.to_string_lossy().len();
        len_a.cmp(&len_b).then_with(|| a.path.cmp(&b.path))
    });
}

/// Rank content matches: exact word boundary first, then partial, then by path depth.
///
/// Uses the match_range and line_content to determine if the match is on a word
/// boundary. Ties broken by path depth (shallower = better).
pub fn rank_content_matches(matches: &mut Vec<ContentMatch>) {
    matches.sort_by(|a, b| {
        let a_exact = is_word_boundary_match(
            &a.line_content,
            a.match_range.start,
            a.match_range.end - a.match_range.start,
        );
        let b_exact = is_word_boundary_match(
            &b.line_content,
            b.match_range.start,
            b.match_range.end - b.match_range.start,
        );

        // Exact word boundary matches first
        let boundary_cmp = b_exact.cmp(&a_exact);
        if boundary_cmp != std::cmp::Ordering::Equal {
            return boundary_cmp;
        }

        // Then by path depth (fewer components = more relevant)
        let depth_a = a.path.components().count();
        let depth_b = b.path.components().count();
        depth_a.cmp(&depth_b)
    });
}

/// Truncate a vector to at most `max` elements.
pub fn truncate_results<T>(items: &mut Vec<T>, max: usize) {
    items.truncate(max);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_result_ranking() {
        // Test file match ranking: shorter paths first
        let mut file_matches = vec![
            FileMatch {
                path: PathBuf::from("/very/long/nested/path/file.rs"),
                score: 1.0,
            },
            FileMatch {
                path: PathBuf::from("/short/file.rs"),
                score: 1.0,
            },
            FileMatch {
                path: PathBuf::from("/medium/path/file.rs"),
                score: 1.0,
            },
        ];

        rank_file_matches(&mut file_matches);

        let paths: Vec<String> = file_matches
            .iter()
            .map(|m| m.path.to_string_lossy().to_string())
            .collect();

        // Shortest path first
        assert!(
            paths[0].len() <= paths[1].len(),
            "First path should be shortest: {} vs {}",
            paths[0],
            paths[1]
        );
        assert!(
            paths[1].len() <= paths[2].len(),
            "Second path should be shorter than third: {} vs {}",
            paths[1],
            paths[2]
        );

        // Test content match ranking: exact word > partial
        let mut content_matches = vec![
            ContentMatch {
                path: PathBuf::from("/a/file.rs"),
                line_number: 1,
                line_content: "substring_match here".to_string(),
                context_before: vec![],
                context_after: vec![],
                match_range: 10..15, // "match" inside "substring_match" -- partial
            },
            ContentMatch {
                path: PathBuf::from("/a/file.rs"),
                line_number: 2,
                line_content: "exact match here".to_string(),
                context_before: vec![],
                context_after: vec![],
                match_range: 6..11, // "match" with space before and after -- word boundary
            },
        ];

        rank_content_matches(&mut content_matches);

        // Word boundary match should come first
        assert_eq!(
            content_matches[0].line_number, 2,
            "Word boundary match (line 2) should rank first"
        );
        assert_eq!(
            content_matches[1].line_number, 1,
            "Partial match (line 1) should rank second"
        );
    }

    #[test]
    fn test_score_file_match() {
        let short_score = score_file_match(10, "test");
        let long_score = score_file_match(100, "test");

        assert!(
            short_score > long_score,
            "Shorter path should score higher: {} vs {}",
            short_score,
            long_score
        );
    }

    #[test]
    fn test_score_content_match_word_boundary() {
        // Word boundary match
        let exact = score_content_match("fn test() {", 3, 4, 2, "test");
        // Partial match (inside a word)
        let partial = score_content_match("fn testing() {", 3, 4, 2, "test");

        assert!(
            exact > partial,
            "Word boundary match should score higher: {} vs {}",
            exact,
            partial
        );
    }

    #[test]
    fn test_is_word_boundary() {
        // "test" at word boundary
        assert!(is_word_boundary_match("fn test() {", 3, 4));
        // "test" at start of string
        assert!(is_word_boundary_match("test foo", 0, 4));
        // "test" at end of string
        assert!(is_word_boundary_match("foo test", 4, 4));
        // "test" inside a word
        assert!(!is_word_boundary_match("testing", 0, 4));
        // "test" inside another word
        assert!(!is_word_boundary_match("attest", 2, 4));
    }

    #[test]
    fn test_truncate_results() {
        let mut items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        truncate_results(&mut items, 5);
        assert_eq!(items.len(), 5);
        assert_eq!(items, vec![1, 2, 3, 4, 5]);

        // Truncating to larger than current length is a no-op
        let mut items2 = vec![1, 2, 3];
        truncate_results(&mut items2, 100);
        assert_eq!(items2.len(), 3);
    }

    #[test]
    fn test_rank_file_matches_alphabetical_tiebreak() {
        let mut matches = vec![
            FileMatch {
                path: PathBuf::from("/src/zebra.rs"),
                score: 1.0,
            },
            FileMatch {
                path: PathBuf::from("/src/alpha.rs"),
                score: 1.0,
            },
        ];

        rank_file_matches(&mut matches);

        // Same length paths should be sorted alphabetically
        assert_eq!(
            matches[0].path,
            PathBuf::from("/src/alpha.rs"),
            "Alphabetically first path should rank first on tie"
        );
    }

    #[test]
    fn test_rank_content_matches_depth_tiebreak() {
        // Both are word boundary matches, but different depths
        let mut matches = vec![
            ContentMatch {
                path: PathBuf::from("/a/b/c/d/file.rs"),
                line_number: 1,
                line_content: "fn test() {}".to_string(),
                context_before: vec![],
                context_after: vec![],
                match_range: 3..7, // "test" on word boundary
            },
            ContentMatch {
                path: PathBuf::from("/a/file.rs"),
                line_number: 2,
                line_content: "fn test() {}".to_string(),
                context_before: vec![],
                context_after: vec![],
                match_range: 3..7, // "test" on word boundary
            },
        ];

        rank_content_matches(&mut matches);

        // Shallower path should rank first when boundary type is the same
        assert_eq!(
            matches[0].line_number, 2,
            "Shallower path should rank first on tie"
        );
    }

    #[test]
    fn test_rank_empty() {
        let mut file_matches: Vec<FileMatch> = vec![];
        rank_file_matches(&mut file_matches);
        assert!(file_matches.is_empty());

        let mut content_matches: Vec<ContentMatch> = vec![];
        rank_content_matches(&mut content_matches);
        assert!(content_matches.is_empty());
    }
}
