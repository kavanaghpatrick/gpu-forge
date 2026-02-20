---
id: stale-results.BREAKDOWN
module: stale-results
priority: 2
status: failing
version: 1
origin: spec-workflow
dependsOn: [data-model]
tags: [p0, bug-fix, correctness, race-condition]
testRequirements:
  unit:
    required: true
    pattern: "src/ui/app.rs::tests, tests/test_stale_results.rs"
---
# Stale Results Breakdown

## Context

This is the P0 critical bug fix. The stale results race condition is the most severe defect in gpu-search: when a user types a query character-by-character, debounced prefix searches produce overlapping `SearchUpdate::ContentMatches` batches. Due to a TOCTOU race in the channel drain (`app.rs:196-202`), stale batches from prefix searches arrive after the drain completes but before the orchestrator checks `should_stop()`. These stale results accumulate via `content_matches.extend()` in `poll_updates()`, causing the UI to display hundreds of results that do not match the current query while the status bar shows "Matches: 0."

The fix has two parts: (1) generation-stamp every outgoing SearchUpdate in the orchestrator, and (2) add a generation guard in poll_updates() that discards updates whose generation does not match the current search generation. Additionally, the debounce interval is increased from 30ms to 100ms to reduce the frequency of intermediate prefix searches.

From PM.md P0-1, TECH.md Sections 3.1-3.4, PM Q1, PM Q4.

## Acceptance Criteria

1. Channel type changed from `Receiver<SearchUpdate>` to `Receiver<StampedUpdate>` in `src/ui/app.rs`
2. Orchestrator (`src/search/orchestrator.rs`) wraps every `tx.send()` call with `StampedUpdate { generation: session.guard.generation_id(), update: ... }`
3. `search_streaming()` signature updated to use `Sender<StampedUpdate>` instead of `Sender<SearchUpdate>`
4. `poll_updates()` in `app.rs` reads `StampedUpdate`, compares `stamped.generation` against `self.search_generation.current_id()`, and discards mismatches with `continue`
5. `dispatch_search()` increments generation before sending the new search command
6. Debounce constant changed from `DEFAULT_DEBOUNCE_MS: u64 = 30` to `100` in `src/ui/search_bar.rs`
7. Unit test U-GEN-2 passes: poll_updates accepts current generation
8. Unit test U-GEN-3 passes: poll_updates discards stale generation
9. Unit test U-GEN-5 passes: rapid generation advance discards all stale
10. Unit test U-GEN-6 passes: Complete message only applied if current generation
11. Integration test I-STALE-1 passes: rapid dispatch ("a","ab","abc","abcd","abcde") only shows final query results
12. Integration test I-STALE-3 passes: status bar count always equals displayed result count
13. Unit test U-DEB-1 passes: default debounce is 100ms
14. Existing test `test_default_debounce_is_30ms` updated to expect 100ms
15. All existing tests pass: `cargo test -p gpu-search`

## Technical Notes

- Reference: [spec/OVERVIEW.md] stale-results is priority 2, depends on data-model (StampedUpdate type)
- UX: From UX.md Section 1.2 Defect 1 -- stale results display is the #1 UX priority; once fixed, no UX mitigation needed
- Test: From QA.md Section 2.1 (U-GEN-2..7), Section 3.2 (I-STALE-1..4) -- use synchronous test harness per QA Q2
- The generation guard is a single `u64 !=` comparison per message -- effectively zero cost (TECH.md Section 3.5)
- Debounce change reduces intermediate prefix searches by ~70% but does NOT eliminate the race -- generation stamping is the actual fix
- Test I-STALE-1 should be written BEFORE the fix (confirmed to FAIL), then confirmed to PASS after (QA Section 3.2)
- Stale result tests use synchronous harness with manual StampedUpdate construction, not real channel timing (QA Q2)
