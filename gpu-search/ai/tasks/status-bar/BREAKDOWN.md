---
id: status-bar.BREAKDOWN
module: status-bar
priority: 7
status: failing
version: 1
origin: spec-workflow
dependsOn: [path-utils, stale-results]
tags: [ui, status-bar, correctness]
testRequirements:
  unit:
    required: true
    pattern: "src/ui/status_bar.rs::tests"
---
# Status Bar Breakdown

## Context

The status bar currently has three problems: (1) it only updates on `SearchUpdate::Complete`, showing stale "Matches: 0" during streaming while results accumulate; (2) it uses `SearchResponse.total_matches` which can contradict the actual displayed result count (the stale results bug manifestation); (3) it displays the full absolute path for the search root.

The fix ensures the status bar always derives its match count from the actual displayed data vectors (`file_matches.len() + content_matches.len()`), adds a "Searching..." state with live match count and elapsed time during active searches, and applies `~` substitution to the search root path.

From PM P0-3, PM P2-3, UX.md Section 8, TECH.md Section 9, TECH.md Section 9.4.

## Acceptance Criteria

1. `StatusBar` struct gains `is_searching: bool` and `search_start: Option<Instant>` fields
2. During search (`is_searching == true`), status bar shows: `Searching... | N matches | X.Xs | ~/root | filters`
3. After search (`is_searching == false`), status bar shows: `N matches in X.Xms | files | Root: ~/root | filters`
4. Match count always derived from `file_matches.len() + content_matches.len()` (not from `SearchResponse.total_matches`)
5. `update_status_from_displayed()` method called after every `poll_updates()` cycle, not just on `Complete`
6. Search root path abbreviated using `path_utils::abbreviate_path()` or a dedicated `abbreviate_root()` function with `~` substitution
7. `is_searching` set to `true` in `dispatch_search()` and `false` on `SearchUpdate::Complete`
8. `search_start` set to `Instant::now()` in `dispatch_search()`
9. Critical invariant maintained: `status_bar.match_count == file_matches.len() + content_matches.len()` at all times
10. Unit test U-SB-1 passes: count derived from `update()` parameter
11. Unit test U-SB-2 passes: searching state renders "Searching..."
12. Unit test U-SB-3 passes: root path shows `~/project` instead of full path
13. Unit test U-SB-4 passes: zero matches shows "0 matches"
14. Integration test I-STALE-3 passes: status bar count always equals displayed result count
15. All existing tests pass: `cargo test -p gpu-search`

## Technical Notes

- Reference: [spec/OVERVIEW.md] status-bar is priority 7, depends on path-utils and stale-results
- UX: From UX.md Section 8 -- proposed status bar format for searching and complete states
- Test: From QA.md Section 2.10 (U-SB-1..4) and Section 3.2 (I-STALE-3)
- Defense-in-depth: even after generation fix, status bar never contradicts visible results (TECH TD-7)
- The status bar update happens in the UI thread (not the background thread), so it is always synchronized with the displayed data
- Animated "Searching..." dots (cycle . .. ... every 300ms) are a nice-to-have, not required for acceptance
- Active filters still display as text tags in the status bar (existing behavior preserved)
