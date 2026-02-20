//! Integration tests for stale result filtering.
//!
//! These tests simulate the generation-stamped update filtering that
//! `poll_updates()` performs, without requiring real channels or timing.
//! Each test constructs `StampedUpdate` messages manually and verifies
//! that generation-based filtering produces correct, deterministic results.

use std::path::PathBuf;
use std::time::Duration;

use gpu_search::search::cancel::SearchGeneration;
use gpu_search::search::profile::PipelineProfile;
use gpu_search::search::types::{
    ContentMatch, FileMatch, SearchResponse, SearchUpdate, StampedUpdate,
};

// ============================================================================
// Test harness: simulates poll_updates() generation guard + result accumulation
// ============================================================================

/// Simulated app state for testing the poll_updates() logic end-to-end.
///
/// This mirrors the relevant fields of `GpuSearchApp` that participate in
/// the generation guard and result accumulation, without any egui dependency.
struct SimulatedApp {
    /// Current search generation (monotonically increasing).
    generation: SearchGeneration,
    /// Accumulated file matches for the current search.
    file_matches: Vec<FileMatch>,
    /// Accumulated content matches for the current search.
    content_matches: Vec<ContentMatch>,
    /// Whether the most recent search has completed.
    is_searching: bool,
    /// Status bar match count (derived from displayed results).
    status_count: usize,
}

impl SimulatedApp {
    fn new() -> Self {
        Self {
            generation: SearchGeneration::new(),
            file_matches: Vec::new(),
            content_matches: Vec::new(),
            is_searching: false,
            status_count: 0,
        }
    }

    /// Simulate dispatching a new search: advance generation, clear results.
    fn dispatch_search(&mut self) -> u64 {
        let guard = self.generation.next();
        let gen_id = guard.generation_id();
        self.file_matches.clear();
        self.content_matches.clear();
        self.is_searching = true;
        self.status_count = 0;
        gen_id
    }

    /// Simulate poll_updates(): process a batch of stamped updates.
    ///
    /// Mirrors the exact logic from `GpuSearchApp::poll_updates()`:
    /// - Discard updates where `stamped.generation != current_id()`
    /// - FileMatches: replace file_matches
    /// - ContentMatches: extend (accumulate progressive batches)
    /// - Complete: replace both, mark not searching
    /// - After all updates: sync status_count to displayed count
    fn poll_updates(&mut self, updates: Vec<StampedUpdate>) {
        let current_gen = self.generation.current_id();

        for stamped in updates {
            // Generation guard: discard stale/future generations
            if stamped.generation != current_gen {
                continue;
            }

            match stamped.update {
                SearchUpdate::FileMatches(matches) => {
                    self.file_matches = matches;
                }
                SearchUpdate::ContentMatches(matches) => {
                    // Accumulate progressive content match batches
                    self.content_matches.extend(matches);
                }
                SearchUpdate::Complete(response) => {
                    self.file_matches = response.file_matches;
                    self.content_matches = response.content_matches;
                    self.is_searching = false;
                }
            }
        }

        // Status bar always matches displayed count
        self.update_status_from_displayed();
    }

    /// Derive status bar count from displayed results (mirrors app logic).
    fn update_status_from_displayed(&mut self) {
        self.status_count = self.file_matches.len() + self.content_matches.len();
    }

    /// Total displayed result count.
    fn displayed_count(&self) -> usize {
        self.file_matches.len() + self.content_matches.len()
    }
}

// ============================================================================
// Test helpers: construct StampedUpdate messages
// ============================================================================

/// Create a FileMatches update with N dummy file matches.
fn file_matches_update(gen: u64, count: usize) -> StampedUpdate {
    let matches = (0..count)
        .map(|i| FileMatch {
            path: PathBuf::from(format!("gen{gen}_file_{i}.rs")),
            score: 1.0,
        })
        .collect();
    StampedUpdate {
        generation: gen,
        update: SearchUpdate::FileMatches(matches),
    }
}

/// Create a ContentMatches update with N dummy content matches.
fn content_matches_update(gen: u64, count: usize) -> StampedUpdate {
    let matches = (0..count)
        .map(|i| ContentMatch {
            path: PathBuf::from(format!("gen{gen}_file_{i}.rs")),
            line_number: (i + 1) as u32,
            line_content: format!("match in gen {gen} line {i}"),
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

/// Create a Complete update with N file matches and M content matches.
fn complete_update(gen: u64, file_count: usize, content_count: usize) -> StampedUpdate {
    let file_matches = (0..file_count)
        .map(|i| FileMatch {
            path: PathBuf::from(format!("gen{gen}_file_{i}.rs")),
            score: 1.0,
        })
        .collect();
    let content_matches = (0..content_count)
        .map(|i| ContentMatch {
            path: PathBuf::from(format!("gen{gen}_content_{i}.rs")),
            line_number: (i + 1) as u32,
            line_content: format!("final match {i}"),
            context_before: vec![],
            context_after: vec![],
            match_range: 0..5,
        })
        .collect();
    StampedUpdate {
        generation: gen,
        update: SearchUpdate::Complete(SearchResponse {
            file_matches,
            content_matches,
            total_files_searched: 1000,
            total_matches: (file_count + content_count) as u64,
            elapsed: Duration::from_millis(42),
            profile: PipelineProfile::default(),
        }),
    }
}

// ============================================================================
// I-STALE-1: Rapid dispatch (5 generations), only last gen results survive
// ============================================================================

/// Simulate rapid typing that dispatches 5 searches in quick succession.
///
/// Each search advances the generation. Results from all 5 searches arrive
/// interleaved. Only the last generation's results should be visible.
///
/// This is the core stale-results race condition: typing "hello" produces
/// searches for "h", "he", "hel", "hell", "hello". Only "hello" results
/// should be displayed.
#[test]
fn i_stale_1_rapid_dispatch_only_last_gen_survives() {
    let mut app = SimulatedApp::new();

    // Simulate 5 rapid dispatches (typing h -> he -> hel -> hell -> hello)
    let mut gen_ids = Vec::new();
    for _ in 0..5 {
        gen_ids.push(app.dispatch_search());
    }
    // gen_ids = [1, 2, 3, 4, 5]
    assert_eq!(gen_ids, vec![1, 2, 3, 4, 5]);
    assert_eq!(app.generation.current_id(), 5);

    // Results arrive interleaved from all 5 generations
    // (simulates the GPU pipeline having queued work from earlier searches)
    let updates = vec![
        // Gen 1 ("h") results arrive first -- these are stale
        file_matches_update(1, 100),
        content_matches_update(1, 50),
        // Gen 2 ("he") results -- stale
        file_matches_update(2, 80),
        content_matches_update(2, 40),
        // Gen 3 ("hel") results -- stale
        file_matches_update(3, 30),
        content_matches_update(3, 20),
        // Gen 4 ("hell") results -- stale
        file_matches_update(4, 15),
        content_matches_update(4, 10),
        // Gen 5 ("hello") results -- CURRENT, these should survive
        file_matches_update(5, 5),
        content_matches_update(5, 3),
        complete_update(5, 5, 3),
    ];

    app.poll_updates(updates);

    // VERIFY: Only gen 5 results are visible
    assert_eq!(
        app.file_matches.len(),
        5,
        "Should have exactly 5 file matches from gen 5 Complete"
    );
    assert_eq!(
        app.content_matches.len(),
        3,
        "Should have exactly 3 content matches from gen 5 Complete"
    );
    assert_eq!(app.displayed_count(), 8, "Total displayed should be 8");
    assert!(!app.is_searching, "Search should be marked complete");

    // All file match paths should be from gen 5
    for fm in &app.file_matches {
        assert!(
            fm.path.to_string_lossy().starts_with("gen5_"),
            "File match path should be from gen 5, got: {:?}",
            fm.path
        );
    }
    // All content match paths should be from gen 5
    for cm in &app.content_matches {
        assert!(
            cm.path.to_string_lossy().starts_with("gen5_"),
            "Content match path should be from gen 5, got: {:?}",
            cm.path
        );
    }
}

// ============================================================================
// I-STALE-2: Manually inject stale gen, verify discarded
// ============================================================================

/// Inject a stale generation update into an active search session.
///
/// After dispatching gen 3, manually inject updates tagged with gen 1 and gen 2.
/// These must be silently discarded. Only gen 3 updates affect state.
#[test]
fn i_stale_2_manually_injected_stale_gen_discarded() {
    let mut app = SimulatedApp::new();

    // Dispatch 3 searches
    let _gen1 = app.dispatch_search(); // gen 1
    let _gen2 = app.dispatch_search(); // gen 2
    let gen3 = app.dispatch_search(); // gen 3 (current)
    assert_eq!(gen3, 3);

    // Poll with a mix: stale gen 1 and gen 2 results injected alongside gen 3
    let updates = vec![
        // Stale gen 1: these should be completely ignored
        file_matches_update(1, 999),
        content_matches_update(1, 999),
        complete_update(1, 999, 999),
        // Stale gen 2: also ignored
        file_matches_update(2, 500),
        content_matches_update(2, 500),
        // Current gen 3: only these count
        file_matches_update(3, 7),
        content_matches_update(3, 4),
    ];

    app.poll_updates(updates);

    // VERIFY: Only gen 3 results are visible
    assert_eq!(
        app.file_matches.len(),
        7,
        "Should have 7 file matches from gen 3 only"
    );
    assert_eq!(
        app.content_matches.len(),
        4,
        "Should have 4 content matches from gen 3 only"
    );
    assert!(
        app.is_searching,
        "Search should still be in progress (no Complete for gen 3 yet)"
    );

    // Now inject a stale Complete from gen 2 -- must NOT finalize
    let stale_complete = vec![complete_update(2, 0, 0)];
    app.poll_updates(stale_complete);

    // VERIFY: Stale Complete did not affect state
    assert!(
        app.is_searching,
        "Stale Complete should not mark search as finished"
    );
    assert_eq!(
        app.file_matches.len(),
        7,
        "File matches should be unchanged after stale Complete"
    );
    assert_eq!(
        app.content_matches.len(),
        4,
        "Content matches should be unchanged after stale Complete"
    );

    // Now send the real gen 3 Complete
    let real_complete = vec![complete_update(3, 7, 4)];
    app.poll_updates(real_complete);

    // VERIFY: Now search is complete with gen 3 results
    assert!(!app.is_searching, "Gen 3 Complete should finalize search");
    assert_eq!(app.file_matches.len(), 7);
    assert_eq!(app.content_matches.len(), 4);
}

// ============================================================================
// I-STALE-3: Status bar count == displayed count invariant
// ============================================================================

/// Verify that `status_count` always equals the sum of displayed file and
/// content matches after every poll cycle.
///
/// This is the invariant that prevents the status bar from showing "150 matches"
/// when only 8 are actually displayed (the original bug symptom).
#[test]
fn i_stale_3_status_bar_count_equals_displayed_count() {
    let mut app = SimulatedApp::new();

    // === Phase 1: Empty state ===
    assert_eq!(app.status_count, 0, "Initial status count should be 0");
    assert_eq!(
        app.status_count,
        app.displayed_count(),
        "Invariant: status == displayed (empty)"
    );

    // === Phase 2: Start search, progressive updates ===
    let gen = app.dispatch_search();
    assert_eq!(app.status_count, 0, "Status should be 0 after dispatch clear");
    assert_eq!(
        app.status_count,
        app.displayed_count(),
        "Invariant holds after dispatch"
    );

    // First batch: 5 file matches
    app.poll_updates(vec![file_matches_update(gen, 5)]);
    assert_eq!(app.status_count, 5, "Status should reflect 5 file matches");
    assert_eq!(
        app.status_count,
        app.displayed_count(),
        "Invariant: status == displayed after file matches"
    );

    // Second batch: 10 content matches (progressive)
    app.poll_updates(vec![content_matches_update(gen, 10)]);
    assert_eq!(
        app.status_count, 15,
        "Status should reflect 5 files + 10 content"
    );
    assert_eq!(
        app.status_count,
        app.displayed_count(),
        "Invariant: status == displayed after content batch 1"
    );

    // Third batch: 7 more content matches (progressive accumulation)
    app.poll_updates(vec![content_matches_update(gen, 7)]);
    assert_eq!(
        app.status_count, 22,
        "Status should reflect 5 files + 17 content"
    );
    assert_eq!(
        app.status_count,
        app.displayed_count(),
        "Invariant: status == displayed after content batch 2"
    );

    // === Phase 3: Complete with final counts ===
    app.poll_updates(vec![complete_update(gen, 5, 17)]);
    assert_eq!(
        app.status_count, 22,
        "Status should reflect final 5 + 17 = 22"
    );
    assert_eq!(
        app.status_count,
        app.displayed_count(),
        "Invariant: status == displayed after Complete"
    );

    // === Phase 4: New search clears and starts fresh ===
    let gen2 = app.dispatch_search();
    assert_eq!(app.status_count, 0, "Status should be 0 after new dispatch");
    assert_eq!(
        app.status_count,
        app.displayed_count(),
        "Invariant holds after second dispatch"
    );

    // Stale gen 1 results arrive (should not affect anything)
    app.poll_updates(vec![
        file_matches_update(gen, 999),
        content_matches_update(gen, 999),
    ]);
    assert_eq!(
        app.status_count, 0,
        "Stale results should not affect status count"
    );
    assert_eq!(
        app.status_count,
        app.displayed_count(),
        "Invariant holds even with stale results in the update batch"
    );

    // Gen 2 results arrive
    app.poll_updates(vec![
        file_matches_update(gen2, 3),
        content_matches_update(gen2, 2),
    ]);
    assert_eq!(
        app.status_count, 5,
        "Status should reflect gen2 results: 3 + 2"
    );
    assert_eq!(
        app.status_count,
        app.displayed_count(),
        "Invariant: status == displayed for gen2 progressive"
    );
}

// ============================================================================
// I-STALE-4: Simulate drain + late arrival with old gen
// ============================================================================

/// Simulate the real-world scenario where:
/// 1. User types "fo" (gen 1) -- results start arriving
/// 2. User types "foo" (gen 2) -- dispatch_search drains channel, starts gen 2
/// 3. Late gen 1 results arrive AFTER the drain (race condition!)
/// 4. Only gen 2 results should be visible
///
/// This is the exact P0 bug scenario. The drain in dispatch_search() clears
/// buffered updates, but results that arrive after the drain (but before
/// gen 2 results) must still be rejected by the generation guard.
#[test]
fn i_stale_4_drain_plus_late_arrival_with_old_gen() {
    let mut app = SimulatedApp::new();

    // Step 1: User types "fo" -> gen 1 dispatched
    let gen1 = app.dispatch_search();
    assert_eq!(gen1, 1);

    // Step 2: Some gen 1 results arrive and are processed
    app.poll_updates(vec![
        file_matches_update(1, 20),
        content_matches_update(1, 15),
    ]);
    assert_eq!(
        app.displayed_count(),
        35,
        "Gen 1 progressive results: 20 + 15"
    );

    // Step 3: User types "foo" -> gen 2 dispatched (clears results)
    let gen2 = app.dispatch_search();
    assert_eq!(gen2, 2);
    assert_eq!(
        app.displayed_count(),
        0,
        "dispatch_search() should clear all results"
    );

    // Step 4: Late gen 1 results arrive AFTER the drain
    // These are the "race condition" results -- they were in-flight when
    // the new search was dispatched and arrive after the channel drain.
    let late_gen1_updates = vec![
        content_matches_update(1, 30), // gen 1 content batch (late)
        complete_update(1, 20, 45),    // gen 1 Complete (late)
    ];
    app.poll_updates(late_gen1_updates);

    // VERIFY: Late gen 1 results are rejected
    assert_eq!(
        app.displayed_count(),
        0,
        "Late gen 1 results must be discarded by generation guard"
    );
    assert!(
        app.is_searching,
        "Late gen 1 Complete must not finalize gen 2 search"
    );
    assert_eq!(
        app.status_count, 0,
        "Status bar must show 0 (no gen 2 results yet)"
    );

    // Step 5: Gen 2 results arrive
    app.poll_updates(vec![
        file_matches_update(2, 8),
        content_matches_update(2, 5),
    ]);
    assert_eq!(
        app.displayed_count(),
        13,
        "Gen 2 progressive: 8 + 5 = 13"
    );

    // Step 6: More late gen 1 results trickle in
    // (e.g., a slow GPU batch from the old search finally completes)
    app.poll_updates(vec![content_matches_update(1, 100)]);
    assert_eq!(
        app.displayed_count(),
        13,
        "Late gen 1 trickle must be discarded; count unchanged"
    );

    // Step 7: Gen 2 completes normally
    app.poll_updates(vec![complete_update(2, 8, 5)]);
    assert!(!app.is_searching, "Gen 2 should be complete");
    assert_eq!(
        app.displayed_count(),
        13,
        "Final gen 2 results: 8 + 5 = 13"
    );
    assert_eq!(
        app.status_count,
        app.displayed_count(),
        "Status bar invariant holds at completion"
    );

    // Verify all results belong to gen 2
    for fm in &app.file_matches {
        assert!(
            fm.path.to_string_lossy().starts_with("gen2_"),
            "All file matches should be gen 2, got: {:?}",
            fm.path
        );
    }
    for cm in &app.content_matches {
        assert!(
            cm.path.to_string_lossy().starts_with("gen2_"),
            "All content matches should be gen 2, got: {:?}",
            cm.path
        );
    }
}
