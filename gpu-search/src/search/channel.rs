//! Progressive delivery channel for streaming search results to the UI.
//!
//! Results arrive in waves:
//! - **Wave 1**: `FileMatches` -- filename matches from GPU path filter (fast).
//! - **Wave 2**: `ContentMatches` -- content matches from GPU content search.
//! - **Complete**: Final `SearchResponse` with aggregated stats.
//!
//! Uses `std::sync::mpsc` channels for cross-thread communication.

use std::sync::mpsc;
use std::time::Duration;

use super::ranking;
use super::types::{ContentMatch, FileMatch, SearchResponse, SearchUpdate};

// ============================================================================
// SearchChannel
// ============================================================================

/// A channel for progressive delivery of search results.
///
/// Wraps `std::sync::mpsc` channel for `SearchUpdate` messages.
/// The sender side sends results in waves; the receiver side
/// (typically the UI) polls for updates.
pub struct SearchChannel {
    tx: mpsc::Sender<SearchUpdate>,
}

/// The receiving half of a `SearchChannel`.
pub struct SearchReceiver {
    rx: mpsc::Receiver<SearchUpdate>,
}

/// Create a new search channel pair (sender, receiver).
pub fn search_channel() -> (SearchChannel, SearchReceiver) {
    let (tx, rx) = mpsc::channel();
    (SearchChannel { tx }, SearchReceiver { rx })
}

impl SearchChannel {
    /// Send Wave 1: filename matches.
    ///
    /// Ranks matches by path length (shorter first) and truncates to MAX_RESULTS.
    /// Returns `true` if the send succeeded, `false` if the receiver was dropped.
    pub fn send_file_matches(&self, mut matches: Vec<FileMatch>) -> bool {
        ranking::rank_file_matches(&mut matches);
        ranking::truncate_results(&mut matches, ranking::MAX_RESULTS);
        self.tx.send(SearchUpdate::FileMatches(matches)).is_ok()
    }

    /// Send Wave 2: content matches.
    ///
    /// Ranks matches by exact word > partial > path depth and truncates to MAX_RESULTS.
    /// Returns `true` if the send succeeded, `false` if the receiver was dropped.
    pub fn send_content_matches(&self, mut matches: Vec<ContentMatch>) -> bool {
        ranking::rank_content_matches(&mut matches);
        ranking::truncate_results(&mut matches, ranking::MAX_RESULTS);
        self.tx.send(SearchUpdate::ContentMatches(matches)).is_ok()
    }

    /// Send Complete: final response with all results and stats.
    ///
    /// Returns `true` if the send succeeded, `false` if the receiver was dropped.
    pub fn send_complete(&self, response: SearchResponse) -> bool {
        self.tx.send(SearchUpdate::Complete(response)).is_ok()
    }
}

impl SearchReceiver {
    /// Try to receive the next update without blocking.
    ///
    /// Returns `None` if no update is available yet.
    pub fn try_recv(&self) -> Option<SearchUpdate> {
        self.rx.try_recv().ok()
    }

    /// Block until the next update arrives, with a timeout.
    ///
    /// Returns `None` if the timeout expires or the sender was dropped.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<SearchUpdate> {
        self.rx.recv_timeout(timeout).ok()
    }

    /// Block until the next update arrives.
    ///
    /// Returns `None` if the sender was dropped.
    pub fn recv(&self) -> Option<SearchUpdate> {
        self.rx.recv().ok()
    }

    /// Collect all updates into a vector, blocking until the channel is closed
    /// or a `Complete` message is received.
    ///
    /// Useful for tests.
    pub fn collect_all(&self) -> Vec<SearchUpdate> {
        let mut updates = Vec::new();
        while let Some(update) = self.recv() {
            let is_complete = matches!(&update, SearchUpdate::Complete(_));
            updates.push(update);
            if is_complete {
                break;
            }
        }
        updates
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn test_progressive_results() {
        let (tx, rx) = search_channel();

        // Wave 1: file matches
        let file_matches = vec![
            FileMatch {
                path: PathBuf::from("/long/nested/path/config.rs"),
                score: 1.0,
            },
            FileMatch {
                path: PathBuf::from("/src/main.rs"),
                score: 2.0,
            },
        ];
        assert!(tx.send_file_matches(file_matches));

        // Wave 2: content matches
        let content_matches = vec![ContentMatch {
            path: PathBuf::from("/src/lib.rs"),
            line_number: 10,
            line_content: "fn test() {}".to_string(),
            context_before: vec![],
            context_after: vec![],
            match_range: 3..7,
        }];
        assert!(tx.send_content_matches(content_matches));

        // Complete
        let response = SearchResponse {
            file_matches: vec![],
            content_matches: vec![],
            total_files_searched: 42,
            total_matches: 3,
            elapsed: Duration::from_millis(5),
            profile: Default::default(),
        };
        assert!(tx.send_complete(response));

        // Receive in order: Wave 1, Wave 2, Complete
        let updates = rx.collect_all();
        assert_eq!(updates.len(), 3, "Should receive exactly 3 updates");

        // First update should be FileMatches (Wave 1)
        assert!(
            matches!(&updates[0], SearchUpdate::FileMatches(fm) if fm.len() == 2),
            "First update should be FileMatches with 2 items"
        );

        // File matches should be ranked (shorter path first)
        if let SearchUpdate::FileMatches(ref fm) = updates[0] {
            assert!(
                fm[0].path.to_string_lossy().len() <= fm[1].path.to_string_lossy().len(),
                "File matches should be ranked by path length: {:?} should be before {:?}",
                fm[0].path,
                fm[1].path
            );
        }

        // Second update should be ContentMatches (Wave 2)
        assert!(
            matches!(&updates[1], SearchUpdate::ContentMatches(cm) if cm.len() == 1),
            "Second update should be ContentMatches with 1 item"
        );

        // Third update should be Complete
        assert!(
            matches!(&updates[2], SearchUpdate::Complete(r) if r.total_files_searched == 42),
            "Third update should be Complete with correct stats"
        );
    }

    #[test]
    fn test_channel_try_recv_empty() {
        let (_tx, rx) = search_channel();

        // No messages sent yet
        assert!(rx.try_recv().is_none(), "Should return None when empty");
    }

    #[test]
    fn test_channel_recv_timeout() {
        let (_tx, rx) = search_channel();

        // Should timeout quickly
        let result = rx.recv_timeout(Duration::from_millis(10));
        assert!(result.is_none(), "Should return None on timeout");
    }

    #[test]
    fn test_channel_sender_dropped() {
        let (tx, rx) = search_channel();
        drop(tx);

        // Receiver should get None after sender drops
        assert!(rx.recv().is_none(), "Should return None when sender dropped");
    }

    #[test]
    fn test_send_file_matches_ranked() {
        let (tx, rx) = search_channel();

        // Send unordered file matches
        let matches = vec![
            FileMatch {
                path: PathBuf::from("/very/deeply/nested/path/file.rs"),
                score: 0.5,
            },
            FileMatch {
                path: PathBuf::from("/a.rs"),
                score: 5.0,
            },
            FileMatch {
                path: PathBuf::from("/src/lib.rs"),
                score: 2.0,
            },
        ];
        tx.send_file_matches(matches);

        if let Some(SearchUpdate::FileMatches(ranked)) = rx.recv() {
            assert_eq!(ranked.len(), 3);
            // Shortest path first
            let lens: Vec<usize> = ranked
                .iter()
                .map(|m| m.path.to_string_lossy().len())
                .collect();
            assert!(
                lens[0] <= lens[1] && lens[1] <= lens[2],
                "File matches should be sorted by path length: {:?}",
                lens
            );
        } else {
            panic!("Expected FileMatches update");
        }
    }

    #[test]
    fn test_send_content_matches_ranked() {
        let (tx, rx) = search_channel();

        let matches = vec![
            // Partial match (inside word "testing")
            ContentMatch {
                path: PathBuf::from("/a/file.rs"),
                line_number: 1,
                line_content: "testing something".to_string(),
                context_before: vec![],
                context_after: vec![],
                match_range: 0..4, // "test" inside "testing"
            },
            // Exact word boundary match
            ContentMatch {
                path: PathBuf::from("/a/file.rs"),
                line_number: 2,
                line_content: "fn test() {}".to_string(),
                context_before: vec![],
                context_after: vec![],
                match_range: 3..7, // "test" with boundary
            },
        ];
        tx.send_content_matches(matches);

        if let Some(SearchUpdate::ContentMatches(ranked)) = rx.recv() {
            assert_eq!(ranked.len(), 2);
            // Word boundary match should be first
            assert_eq!(
                ranked[0].line_number, 2,
                "Word boundary match should rank first"
            );
        } else {
            panic!("Expected ContentMatches update");
        }
    }

    #[test]
    fn test_truncation() {
        let (tx, rx) = search_channel();

        // Create more than MAX_RESULTS file matches
        let matches: Vec<FileMatch> = (0..ranking::MAX_RESULTS + 500)
            .map(|i| FileMatch {
                path: PathBuf::from(format!("/path/{}.rs", i)),
                score: 1.0,
            })
            .collect();

        let original_len = matches.len();
        assert!(original_len > ranking::MAX_RESULTS);

        tx.send_file_matches(matches);

        if let Some(SearchUpdate::FileMatches(truncated)) = rx.recv() {
            assert_eq!(
                truncated.len(),
                ranking::MAX_RESULTS,
                "Should be truncated to MAX_RESULTS ({})",
                ranking::MAX_RESULTS
            );
        } else {
            panic!("Expected FileMatches update");
        }
    }

    #[test]
    fn test_wave_order_preserved() {
        // Verify that sending waves in order results in receiving them in order
        let (tx, rx) = search_channel();

        // Spawn a thread to send waves
        let handle = std::thread::spawn(move || {
            // Wave 1
            tx.send_file_matches(vec![FileMatch {
                path: PathBuf::from("/a.rs"),
                score: 1.0,
            }]);

            // Wave 2
            tx.send_content_matches(vec![ContentMatch {
                path: PathBuf::from("/a.rs"),
                line_number: 1,
                line_content: "test".to_string(),
                context_before: vec![],
                context_after: vec![],
                match_range: 0..4,
            }]);

            // Complete
            tx.send_complete(SearchResponse {
                file_matches: vec![],
                content_matches: vec![],
                total_files_searched: 1,
                total_matches: 2,
                elapsed: Duration::from_millis(1),
                profile: Default::default(),
            });
        });

        handle.join().unwrap();

        let updates = rx.collect_all();
        assert_eq!(updates.len(), 3);

        // Verify wave order
        assert!(
            matches!(&updates[0], SearchUpdate::FileMatches(_)),
            "Wave 1 should be FileMatches"
        );
        assert!(
            matches!(&updates[1], SearchUpdate::ContentMatches(_)),
            "Wave 2 should be ContentMatches"
        );
        assert!(
            matches!(&updates[2], SearchUpdate::Complete(_)),
            "Wave 3 should be Complete"
        );
    }
}
