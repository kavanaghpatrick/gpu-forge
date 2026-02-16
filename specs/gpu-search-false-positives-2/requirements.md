---
spec: gpu-search-false-positives-2
phase: requirements
created: 2026-02-16
generated: auto
---

# Requirements: gpu-search-false-positives-2

## Summary

Fix GPU content search false positives where searching "kolbey" returns matches in files containing "Patrick Kavanagh" but not "kolbey". Build orchestrator-level integration tests with rapid query simulation, byte_offset validation, and match_range verification. Then fix the root cause.

## User Stories

### US-1: Zero False Positives in Content Search
As a user, I want every content match returned by gpu-search to actually contain my search pattern so that I can trust the results.

**Acceptance Criteria**:
- AC-1.1: Every ContentMatch's `line_content` contains the search pattern (case-insensitive match)
- AC-1.2: Every ContentMatch's `match_range` correctly highlights the pattern within `line_content`
- AC-1.3: Searching "kolbey" returns 0 matches in files that do not contain "kolbey"
- AC-1.4: CPU verification layer confirms 0 false positives in Full mode

### US-2: Accurate Results During Rapid Query Changes
As a user, I want search results to remain accurate when I type quickly (character by character) so that intermediate results don't contaminate final results.

**Acceptance Criteria**:
- AC-2.1: Searching "k" -> "ko" -> "kol" -> "kolb" -> "kolbe" -> "kolbey" produces correct results at each step
- AC-2.2: No matches from a previous query appear in results for a subsequent query
- AC-2.3: Cancel/restart cycles do not leak results between generations

### US-3: Validated match_range for UI Highlighting
As a user, I want the highlighted portion of each result line to correctly correspond to my search pattern so that I can see exactly what matched.

**Acceptance Criteria**:
- AC-3.1: `line_content[match_range.clone()]` case-insensitively equals the search pattern for every ContentMatch
- AC-3.2: match_range.start < match_range.end for every ContentMatch
- AC-3.3: match_range.end <= line_content.len() for every ContentMatch

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Every ContentMatch must contain the search pattern in its line_content | Must | US-1 |
| FR-2 | resolve_match() must validate byte_offset consistency (not just line-level find) | Must | US-1 |
| FR-3 | Orchestrator-level integration tests covering full pipeline (walk -> GPU -> resolve -> UI) | Must | US-1, US-2 |
| FR-4 | Rapid query change test simulating character-by-character typing with cancel/restart | Must | US-2 |
| FR-5 | match_range validation: extracted text must equal pattern | Must | US-3 |
| FR-6 | Byte_offset validation: content at byte_offset must match pattern | Should | US-1 |
| FR-7 | CPU verification (verify.rs) enabled in Full mode during test runs | Should | US-1 |
| FR-8 | Regression tests preventing reintroduction of false positives | Must | US-1 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Integration tests complete in < 60 seconds | Performance |
| NFR-2 | No increase in normal search latency from fix | Performance |
| NFR-3 | All existing tests continue to pass | Compatibility |

## Out of Scope
- GPU kernel (Metal shader) modifications
- UI rendering changes
- Index daemon modifications
- Performance optimization of search pipeline

## Dependencies
- tempfile (dev-dependency, already in Cargo.toml)
- crossbeam-channel (already in Cargo.toml)
- Real Metal GPU device (Apple Silicon required for tests)
